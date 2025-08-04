"""
Copyright to FOA Authors ICML 2024
zo_version:
use zeroth-order optimization to minimize the defined loss function 
and optimize a set of prompts to generate the best output.
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from models.vpt import PromptViT
import numpy as np
import time
import math

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *

RUNNING_IMAGNET_R = False

class FOA_ZO_ADAM(nn.Module):
    """test-time Forward Optimization Adaptation(Zero-order update with Adam optimizer)
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, 
                 model:PromptViT, 
                 epsilon=0.1, 
                 lr=0.01, 
                 num_samples=20, 
                 fitness_lambda=0.4,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8):
        super().__init__()

        self.model = model
        self.epsilon = epsilon  # Zero-order扰动幅度
        self.lr = lr  # 学习率
        self.num_samples = num_samples  # Zero-order梯度估计的随机样本数
        self.fitness_lambda = fitness_lambda
        
        # Adam优化器参数
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.best_loss = np.inf
        self.current_loss = np.inf
        self.decay_factor = 0.1
        self.best_prompts = None
        self.hist_stat = None
        
        # 初始化Adam状态
        self.step = 0
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat

    def _compute_loss(self, images, shift_vector, imagenet_mask):#调用有问题！！先试试
        features = self.model.layers_cls_features_with_prompts(images)#从PromptViT模型中提取图像特征
        cls_features = features[:, -768:]# the feature of classification token，ViT模型的最后一个分类token的特征768维，e_N^0
        
        """Compute loss (Eqn. 5) for prompt evaluation"""
        batch_std, batch_mean = torch.std_mean(features, dim=0)#OOD域数据特征的均值和方差
        #ID和OOD域数据特征MSE损失
        std_mse = criterion_mse(batch_std, self.train_info[0])
        mean_mse = criterion_mse(batch_mean, self.train_info[1])
        # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
        discrepancy_loss = self.fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64

        output = self.model.vit.head(cls_features)#通过分类头，将cls_features映射到分类输出
        
        """entropy loss for Eqn. (5)"""
        if imagenet_mask is not None:#有需要可以利用imagenet_mask对输出进行筛选
            output = output[:, imagenet_mask]
        entropy_loss = softmax_entropy(output).sum()#对batch求和得到熵损失
        
        loss = discrepancy_loss + entropy_loss#总损失fitness function

        """activation shifting, Eqn. (7)"""
        if shift_vector is not None:
            output = self.model.vit.head(cls_features + 1. * shift_vector)
            if imagenet_mask is not None:
                output = output[:, imagenet_mask]
        return output, loss, batch_mean
    
 
    def _compute_zero_order_gradient(self, images, shift_vector, imagenet_mask):
        """MeZO风格的Zero-order Adam gradient estimation"""
        prompts_flat = self.model.prompts.flatten().detach()
        gradient = torch.zeros_like(prompts_flat).cuda()
        batch_means = []

        for _ in range(self.num_samples):
            u = torch.randn_like(prompts_flat).cuda()  # Random direction
            u /= torch.norm(u, p=2)  # Normalize direction，L2范数归一化
            
            # 正向扰动评估
            p_plus = (prompts_flat + self.epsilon * u).reshape_as(self.model.prompts)
            self.model.prompts = nn.Parameter(p_plus)
            _, loss_plus, batch_mean_plus = self._compute_loss(images, shift_vector, imagenet_mask)
            batch_means.append(batch_mean_plus[-768:].unsqueeze(0))

            # 反向扰动评估
            p_minus = (prompts_flat - self.epsilon * u).reshape_as(self.model.prompts)
            self.model.prompts = nn.Parameter(p_minus)
            _, loss_minus, batch_mean_minus = self._compute_loss(images, shift_vector, imagenet_mask)
            batch_means.append(batch_mean_minus[-768:].unsqueeze(0))

            del batch_mean_minus, batch_mean_plus
            
            # 计算ZO梯度估计
            grad_estimate = (loss_plus.item() - loss_minus.item()) / (2 * self.epsilon) * u
            gradient += grad_estimate

        return gradient / self.num_samples, batch_means
    
    
    def forward(self, images):
        imagenet_mask = None
        shift_vector = self._get_shift_vector()
        self.current_loss, batch_means = np.inf, []
        
        # 计算ZO梯度
        gradient, batch_means = self._compute_zero_order_gradient(images, shift_vector, imagenet_mask)
        
        # Adam更新步骤
        self.step += 1
        
        # 初始化动量
        if self.m is None:
            self.m = torch.zeros_like(gradient)
            self.v = torch.zeros_like(gradient)
        
        # 更新偏差修正的一阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        # 更新偏差修正的二阶矩估计
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # 计算偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.step)
        v_hat = self.v / (1 - self.beta2 ** self.step)
        
        # 更新prompts
        prompts_flat = self.model.prompts.flatten().detach()
        new_prompts = prompts_flat - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        self.model.prompts = nn.Parameter(new_prompts.reshape_as(self.model.prompts))
        
        # 评估更新后的prompts
        outputs, loss, _ = self._compute_loss(images, shift_vector, imagenet_mask)

        # 更新统计信息
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        """
        更新总体测试统计信息，公式(9)
        """
        
        # Update best results
        self.current_loss = loss.item()
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_prompts = new_prompts

        return outputs
    
    def obtain_origin_stat(self, train_loader):
        """
        计算训练集特征的均值和方差。

        该函数用于计算训练集特征的均值和方差，为模型量化做准备。
        它首先遍历训练集以提取特征，然后计算这些特征的均值和方差。
        最后，它为快速适应准备量化模型。

        参数:
        - train_loader: DataLoader 类型，训练集的数据加载器，用于批量加载训练数据和标签。

        返回:
        无
        """
        print('===> 开始计算均值和方差')
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()               #dl[0]是图片，dl[1]是标签
                feature = self.model.layers_cls_features(images)    #从PromptViT模型中提取图像特征
                features.append(feature)
                # break
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0)#计算源域数据特征的均值和方差，dim=0表示计算每个特征的均值和方差
        del features#释放内存
        
        # 为快速适应准备量化模型
        for _, m in self.model.vit.named_modules():#遍历PromptViT模型的所有模块
            if type(m) == PTQSLBatchingQuantMatMul:#如果是PTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,64)).cuda(), torch.zeros((1,12,64,197+self.model.num_prompts)).cuda())
            elif type(m) == SoSPTQSLBatchingQuantMatMul:#如果是SoSPTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,197+self.model.num_prompts)).cuda(), torch.zeros((1,12,197+self.model.num_prompts,64)).cuda())
        print('===> 计算均值和方差结束')

    def reset(self):
    # Reset historical statistics and prompts
        self.hist_stat = None
        self.model.prompts.data.zero_()
        self.step = 0
        self.m = None
        self.v = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.熵的softmax分布从对数中得到"""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()