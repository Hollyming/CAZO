"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from models.vpt import PromptViT
import cma
import numpy as np
import time
import math

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *

RUNNING_IMAGNET_R = False

class FOA(nn.Module):
    """test-time Forward Optimization Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, model:PromptViT, fitness_lambda=0.4):
        super().__init__()
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.es = self._init_cma() # initialization for CMA-ES in algorithm 1

        self.best_prompts = model.prompts
        self.best_loss = np.inf
        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)

    def _init_cma(self):
        """CMA-ES initialization"""
        dim = self.model.prompts.numel()    #总元素个数
        popsize = 27 #种群大小 which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

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
        #train_info[1]是ID的均值，hist_stat是OOD域的均值，二者相减得到shift_vector
        #之前的batch_mean取得也是-768维度的CLS特征，因此train_info[1]也取-768维度的CLS特征

    def forward(self, x):
        """
        实现前向传播逻辑，根据公式(8)计算移位方向。
        该方法旨在通过使用协方差矩阵自适应进化策略（CMA-ES）最小化定义的损失函数，优化一组提示以生成最佳输出。

        参数:
        - x: 模型的输入数据。

        返回:
        - best_outputs: 使用优化后的提示生成的输出。
        """
        # 获取用于计算移位方向的移位向量。
        shift_vector = self._get_shift_vector()

        # 初始化变量以跟踪最佳损失、对应的输出和批量均值。
        self.best_loss, self.best_outputs, batch_means = np.inf, None, []#初始化loss为无穷大，输出为None，batch_means为空列表

        """
        从CMA-ES采样并评估新解。
        注意我们还将当前解与之前的最佳解进行比较。
        """
        # 从CMA-ES分布生成新解（提示）并评估它们。algorithm 3
        #修改！！！！
        prompts, losses = self.es.ask() + [self.best_prompts.flatten().cpu()], []#ask产生es.popsize个样本，然后加上最佳prompt展平的样本
        for j, prompt in enumerate(prompts):
            # 使用新解更新模型的提示，并确保不需要梯度计算。
            self.model.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).
                                                        reshape_as(self.model.prompts).cuda())
            self.model.prompts.requires_grad_(False)

            # 执行前向传播并计算当前提示的损失。
            outputs, loss, batch_mean = forward_and_get_loss(x, self.model, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask)
            # batch_mean包含当前batch特征所有维度的均值，将batch_mean的CLStoken均值部分附加到batch_means中，以供后续使用。
            batch_means.append(batch_mean[-768:].unsqueeze(0))  #大小为(1,768)的列表，for循环后列表长度为27
            del batch_mean

            # 如果当前解更好，则更新最佳提示及其对应的损失。
            if self.best_loss > loss.item():#一般loss计算结果都是标量张量，所以直接用item()方法取出数值
                self.best_prompts = self.model.prompts
                self.best_loss = loss.item()
                self.best_outputs = outputs
                outputs = None
            losses.append(loss.item())
            del outputs

            # 记录当前解的评估结果。
            print(f'Solution:[{j+1}/{len(prompts)}], Loss: {loss.item()}')

        """
        CMA-ES 更新，公式(6)
        """
        # 根据评估的解及其损失更新CMA-ES参数。
        #修改！！！！
        self.es.tell(prompts, losses)
        
        """
        利用_update_hist更新总体OOD域批均值的统计信息，公式(9)
        """
        # 使用当前批均值OOD域的平均值更新历史记录中的测试统计信息。
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        #列表经过cat拼接后大小为(27,768)，然后对第0维度求均值，得到(768,)的张量,此时与batch_mean的大小一致
        self._update_hist(batch_means)#更新历史记录中的测试统计信息，计算activate shift的历史均值，便于后续方向计算
        return self.best_outputs
    
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
        self.es = self._init_cma()
        self.hist_stat = None

        self.model.reset()
        self.best_prompts = self.model.prompts
        

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.熵的softmax分布从对数中得到"""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model:PromptViT, fitness_lambda, train_info, shift_vector, imagenet_mask):
    features = model.layers_cls_features_with_prompts(images)#从PromptViT模型中提取图像特征
    cls_features = features[:, -768:] # the feature of classification token，ViT模型的最后一个分类token的特征768维，e_N^0
    
    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0) #OOD域数据特征的均值和方差
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])#ID和OOD域数据特征MSE损失，train_info[0]是ID的方差，train_info[1]是ID的均值
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    #images.shape[0]是样本数batch_size
    output = model.vit.head(cls_features)#通过分类头，将cls_features映射到分类输出

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]#有需要可以利用imagenet_mask对输出进行筛选
    entropy_loss = softmax_entropy(output).sum()#对batch求和得到熵损失
    loss = discrepancy_loss + entropy_loss
    
    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)#添加activate shift 偏移量，覆盖原有output
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean