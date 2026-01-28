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

class BP_Adapter_FT(nn.Module):
    """
    BP_Adapter_FT: 使用反向传播和有标签数据微调adapter参数
    参考BP_Adapter的实现方式，简化为只支持ViT架构
    
    与BP_Adapter的主要区别：
    1. 使用交叉熵损失（有标签）而非熵损失（无标签）
    2. forward接收(x, targets)而不是只有x
    3. 适用于有监督的微调场景
    """
    def __init__(self, model: AdaFormerViT, fitness_lambda=0.4, lr=0.01, 
                 optimizer_type='sgd', momentum=0.9, use_pure_ce=False):
        """
        初始化BP_Adapter_FT算法
        
        Args:
            model: AdaFormerViT模型
            fitness_lambda: 适应度函数的平衡因子
            lr: 学习率
            optimizer_type: 优化器类型，'sgd'或'adam'
            momentum: SGD动量系数
            use_pure_ce: 如果为True，仅使用交叉熵损失而不使用mse损失
        """
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.use_pure_ce = use_pure_ce
        self.final_loss = np.inf  # 记录最终损失值
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
    
    def forward(self, x, targets):
        """
        前向传播，在内部完成反向传播和参数更新
        
        Args:
            x: 输入图像
            targets: 标签（有监督微调）
        """
        # 调用forward_and_get_loss_ft，内部会完成backward和optimizer.step
        outputs, self.final_loss, batch_mean = forward_and_get_loss_ft(
            x, targets, self.model, self.optimizer, self.fitness_lambda, 
            self.train_info, self.imagenet_mask, self.use_pure_ce
        )
        
        # 注意：微调场景下不需要更新hist_stat，因为不使用shift_vector
        # 微调依赖有监督信号，不需要activation shifting
        
        return outputs

    def obtain_origin_stat(self, train_loader):
        """计算源域统计信息（与BP_Adapter相同，但微调时可选）"""
        print('===> begin calculating mean and variance')
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

criterion_mse = nn.MSELoss(reduction='mean').cuda()
criterion_ce = nn.CrossEntropyLoss().cuda()

@torch.enable_grad()  # 确保梯度可用
def forward_and_get_loss_ft(images, targets, model: AdaFormerViT, optimizer, fitness_lambda, 
                            train_info, imagenet_mask, use_pure_ce=False):
    """
    前向传播并计算有监督损失，参考bp_adapter的实现
    
    与BP_Adapter的区别：
    1. 使用交叉熵损失（有标签）而不是熵损失（无标签）
    2. 不需要shift_vector（有监督信号足够强）
    """
    # 提取特征
    features = model.layers_cls_features_with_adapters(images)
    cls_features = features[:, -768:]
    
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    
    # Discrepancy loss (可选，微调时通常不需要)
    if not use_pure_ce and train_info is not None:
        std_mse = criterion_mse(batch_std, train_info[0])
        mean_mse = criterion_mse(batch_mean, train_info[1])
        discrepancy_loss = fitness_lambda * (std_mse + mean_mse)
    else:
        discrepancy_loss = 0.0
    
    del features
    
    output = model.vit.head(cls_features)
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    
    # 交叉熵损失（有监督）
    ce_loss = criterion_ce(output, targets)
    loss = ce_loss if use_pure_ce else (discrepancy_loss + ce_loss)
    
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