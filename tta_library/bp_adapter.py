import torch
import torch.nn as nn
import numpy as np
import time
import os
from copy import deepcopy
from models.adaformer import AdaFormerViT

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from quant_library.quant_layers.matmul import *

class BP_Adapter(nn.Module):
    """
    BP_Adapter: 使用反向传播更新adapter参数的测试时适应算法
    参考FOA_BP的实现方式，简化为只支持ViT架构
    """
    def __init__(self, model: AdaFormerViT, fitness_lambda=0.4, lr=0.01, 
                 optimizer_type='sgd', momentum=0.9, use_pure_entropy=False):
        """
        初始化BP_Adapter算法
        
        Args:
            model: AdaFormerViT模型
            fitness_lambda: 适应度函数的平衡因子
            lr: 学习率
            optimizer_type: 优化器类型，'sgd'或'adam'
            momentum: SGD动量系数
            use_pure_entropy: 如果为True，仅使用熵损失而不使用mse损失
        """
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.use_pure_entropy = use_pure_entropy
        self.final_loss = np.inf  # 添加这一行，用于记录最终的损失值
        self.model = model
        
        # 收集adapter参数
        adapter_params = []
        for adapter in self.model.adapters.values():
            for param in adapter.parameters():
                param.requires_grad_(True)
                adapter_params.append(param)
        
        # 初始化优化器
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(adapter_params, lr=lr, momentum=momentum)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(adapter_params, lr=lr)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 保存初始状态用于reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.hist_stat = None
        self.train_info = None
        self.imagenet_mask = None
    
    def _update_hist(self, batch_mean):
        """Update overall test statistics"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def _get_shift_vector(self):
        """Calculate shift direction"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat
    
    def forward(self, x):
        """前向传播，在内部完成反向传播和参数更新"""
        shift_vector = self._get_shift_vector()
        
        # 调用forward_and_get_loss，内部会完成backward和optimizer.step
        outputs, self.final_loss, batch_mean = forward_and_get_loss(
            x, self.model, self.optimizer, self.fitness_lambda, 
            self.train_info, shift_vector, self.imagenet_mask, self.use_pure_entropy
        )
        
        # 更新历史统计
        self._update_hist(batch_mean[-768:])
        
        return outputs

    def obtain_origin_stat(self, train_loader):
        """计算源域统计信息（与ZO_Base相同）"""
        print('===> begin calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'train_info_adapter.pt')
        
        if os.path.exists(save_path):
            print('===> 从文件加载预计算的均值和方差')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            self.model.eval()
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.layers_cls_features(images)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)
            del features
            
            print('===> 保存计算结果到文件')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)
        
        # preparing quantized model
        num_layers = len(self.model.vit.blocks)
        head_dim = self.model.vit.blocks[0].attn.head_dim
        for _, m in self.model.vit.named_modules():
            if type(m) == PTQSLBatchingQuantMatMul:
                m._get_padding_parameters(
                    torch.zeros((1, num_layers, 197, head_dim)).cuda(),
                    torch.zeros((1, num_layers, head_dim, 197)).cuda()
                )
            elif type(m) == SoSPTQSLBatchingQuantMatMul:
                m._get_padding_parameters(
                    torch.zeros((1, num_layers, 197, 197)).cuda(),
                    torch.zeros((1, num_layers, 197, head_dim)).cuda()
                )
        print('===> 计算均值和方差结束')

    def reset(self):
        """重置模型和优化器到初始状态"""
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.hist_stat = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='mean').cuda()

@torch.enable_grad()  # 确保梯度可用
def forward_and_get_loss(images, model: AdaFormerViT, optimizer, fitness_lambda, 
                        train_info, shift_vector, imagenet_mask, use_pure_entropy=False):
    """前向传播并计算损失，参考foa_bp.py的实现"""
    # 提取特征
    features = model.layers_cls_features_with_adapters(images)
    cls_features = features[:, -768:] # the feature of classification token，ViT模型的最后一个分类token的特征768维，e_N^0
    
    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    
    # 计算discrepancy loss
    if not use_pure_entropy:
        std_mse = criterion_mse(batch_std, train_info[0])
        mean_mse = criterion_mse(batch_mean, train_info[1])
        discrepancy_loss = fitness_lambda * (std_mse + mean_mse)
    else:
        discrepancy_loss = 0.0
    del features

    output = model.vit.head(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).mean()
    loss = entropy_loss if use_pure_entropy else (discrepancy_loss + entropy_loss)

    # 使用shift_vector获取最终输出（不需要梯度）
    with torch.no_grad():
        if shift_vector is not None:
            output = model.vit.head(cls_features + 1. * shift_vector)
            if imagenet_mask is not None:
                output = output[:, imagenet_mask]
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, loss, batch_mean

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)