import torch
import torch.nn as nn
import numpy as np
import time
import os
from models.adaformer import AdaFormerViT
from models.deit_adapter import DeiTAdapter
from models.swin_adapter import SwinAdapter
from models.resnet_adapter import ResNetAdapter

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from quant_library.quant_layers.matmul import *

class Optimizer:
    """优化器基类"""
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, grad_estimate):
        """更新参数"""
        raise NotImplementedError

class SGD(Optimizer):
    """SGD优化器"""
    def __init__(self, lr):
        super().__init__(lr)
    
    def step(self, grad_estimate):
        return -self.lr * grad_estimate     #w-lr*g

class SGD_Momentum(Optimizer):
    """带动量的SGD优化器"""
    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.momentum = None
    
    def step(self, grad_estimate):
        if self.momentum is None:
            self.momentum = (1 - self.beta) * grad_estimate
        else:
            self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_estimate
        return -self.lr * self.momentum

    def reset(self):
        """重置动量"""
        self.momentum = None

class ZO_Base_FT(nn.Module):
    """
    ZO_Base_FT: 零阶优化的有标签微调版本
    
    使用零阶优化估计梯度，但使用交叉熵损失进行有监督训练
    """
    def __init__(self, model, fitness_lambda=0.4, lr=0.01, 
                 pertub=20, epsilon=0.1,
                 optimizer_type='sgd', beta=0.9, use_pure_ce=False):
        """
        初始化ZO_Base_FT算法
        
        Args:
            model: 支持AdaFormerViT, DeiTAdapter, SwinAdapter, ResNetAdapter
            fitness_lambda: 适应度函数的平衡因子
            lr: 学习率
            pertub: 扰动数量k
            epsilon: 扰动大小ε
            optimizer_type: 优化器类型，'sgd'或'sgd_momentum'
            beta: 动量系数
            use_pure_ce: if True, only use cross-entropy loss without mse loss
        """
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.epsilon = epsilon
        self.pertub = pertub
        self.use_pure_ce = use_pure_ce
        
        self.model = model
        self.model_type = self._detect_model_type()
        
        # 确保所有adapter参数不需要梯度
        for adapter in self.model.adapters.values():
            for param in adapter.parameters():
                param.requires_grad_(False)
        
        # 保存最佳adapter参数
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        self.best_loss = np.inf
        self.final_loss = np.inf
        self.hist_stat = None
        self.train_info = None
        self.imagenet_mask = None
        
        # 交叉熵损失
        self.criterion_ce = nn.CrossEntropyLoss().cuda()
        
        # 初始化优化器
        if optimizer_type == 'sgd':
            self.optimizer = SGD(lr)
        elif optimizer_type == 'sgd_momentum':
            self.optimizer = SGD_Momentum(lr, beta)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _detect_model_type(self):
        """自动检测模型类型"""
        if isinstance(self.model, AdaFormerViT):
            return 'vit'
        elif isinstance(self.model, DeiTAdapter):
            return 'deit'
        elif isinstance(self.model, SwinAdapter):
            return 'swin'
        elif isinstance(self.model, ResNetAdapter):
            return 'resnet'
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
    
    def _get_model_info(self):
        """获取模型的层数和特征维度信息"""
        if self.model_type in ['vit', 'deit']:
            base_model = self.model.vit if self.model_type == 'vit' else self.model.deit
            num_layers = len(base_model.blocks)
            embed_dim = base_model.embed_dim
            head_dim = base_model.blocks[0].attn.head_dim
            return {
                'num_layers': num_layers,
                'embed_dim': embed_dim,
                'head_dim': head_dim,
                'feature_dim': num_layers * embed_dim
            }
        elif self.model_type == 'swin':
            total_blocks = self.model.total_blocks
            stage_dims = []
            for stage_idx, layer in enumerate(self.model.swin.layers):
                stage_dim = int(self.model.swin.embed_dim * 2 ** stage_idx)
                stage_dims.extend([stage_dim] * len(layer.blocks))
            feature_dim = sum(stage_dims)
            return {
                'num_layers': total_blocks,
                'stage_dims': stage_dims,
                'feature_dim': feature_dim
            }
        elif self.model_type == 'resnet':
            total_blocks = self.model.total_blocks
            block_dims = self.model.block_dims
            feature_dim = sum(block_dims)
            return {
                'num_layers': total_blocks,
                'block_dims': block_dims,
                'feature_dim': feature_dim
            }
    
    def _extract_features(self, images, with_adapter=False):
        """根据模型类型提取特征"""
        if with_adapter:
            if self.model_type in ['vit', 'deit']:
                return self.model.layers_cls_features_with_adapters(images)
            elif self.model_type == 'swin':
                return self.model.layers_features_with_adapters(images)
            elif self.model_type == 'resnet':
                return self.model.layers_features_with_adapters(images)
        else:
            if self.model_type in ['vit', 'deit']:
                return self.model.layers_cls_features(images)
            elif self.model_type == 'swin':
                return self.model.layers_features(images)
            elif self.model_type == 'resnet':
                return self.model.layers_features(images)
    
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
            if self.model_type in ['vit', 'deit']:
                model_info = self._get_model_info()
                embed_dim = model_info['embed_dim']
                return self.train_info[1][-embed_dim:] - self.hist_stat
            elif self.model_type == 'swin':
                final_dim = self.model.swin.num_features
                return self.train_info[1][-final_dim:] - self.hist_stat[-final_dim:]
            elif self.model_type == 'resnet':
                model_info = self._get_model_info()
                final_dim = model_info['block_dims'][-1]
                return self.train_info[1][-final_dim:] - self.hist_stat[-final_dim:]
    
    def _sample_perturbations(self, num_samples, dim):
        """采样k个标准正态分布的扰动向量"""
        return torch.randn(num_samples, dim, device='cuda')
    
    def _apply_perturbation(self, perturbation, sign=1):
        """应用扰动到所有adapter参数"""
        start_idx = 0
        
        # 遍历所有adapter的参数
        for layer_idx in self.model.adapter_layers:
            adapter = self.model.adapters[f'adapter_{layer_idx}']
            for name, param in adapter.named_parameters():
                num_params = param.numel()
                param_perturbation = perturbation[start_idx:start_idx + num_params]
                if not isinstance(param_perturbation, torch.Tensor):
                    param_perturbation = torch.tensor(
                        param_perturbation, dtype=torch.float, device=param.device
                    )
                param.data += sign * self.epsilon * param_perturbation.reshape(param.shape)
                start_idx += num_params

    def _save_current_adapter(self):
        """保存当前所有adapter参数的深拷贝"""
        return {
            f'adapter_{layer}': {
                k: v.clone() for k, v in self.model.adapters[f'adapter_{layer}'].state_dict().items()
            } for layer in self.model.adapter_layers
        }

    def _load_adapter(self, adapter_states):
        """加载所有adapter参数"""
        for layer in self.model.adapter_layers:
            self.model.adapters[f'adapter_{layer}'].load_state_dict(
                adapter_states[f'adapter_{layer}']
            )
    
    def forward(self, x, targets):
        """
        使用零阶优化方法和有标签数据微调adapter参数
        
        Args:
            x: 输入图像
            targets: 标签（有监督）
        """
        shift_vector = self._get_shift_vector()

        # 初始化变量跟踪最佳损失和输出
        self.best_loss, self.best_outputs, batch_means = np.inf, None, []
        
        # 保存当前模型adapter参数
        current_adapter = self._save_current_adapter()
        losses = []
        
        # 记录总体开始时间
        total_start = time.time()
        
        # 获取参数总数
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        
        # 初始化梯度估计为零向量
        grad_estimate = torch.zeros(total_params, device='cuda')
        
        # 生成k个标准正态分布的扰动
        perturbations = self._sample_perturbations(self.pertub, total_params)
        
        # 使用零阶方法估计梯度
        print("开始使用零阶方法估计梯度（有监督）...")
        for i, z in enumerate(perturbations):
            # 正向扰动: f(x + εz)
            self._load_adapter(current_adapter)
            self._apply_perturbation(z, sign=1)
            outputs_pos, loss_pos, batch_mean = forward_and_get_loss_supervised(
                x, targets, self.model, self.fitness_lambda, self.train_info, 
                shift_vector, self.imagenet_mask, self.model_type, self.use_pure_ce, self.criterion_ce
            )
            
            # 根据模型类型提取合适维度的batch_mean
            model_info = self._get_model_info()
            if self.model_type in ['vit', 'deit']:
                embed_dim = model_info['embed_dim']
                batch_means.append(batch_mean[-embed_dim:].unsqueeze(0))
            elif self.model_type in ['swin', 'resnet']:
                batch_means.append(batch_mean.unsqueeze(0))
            del batch_mean
            
            # 负向扰动: f(x - εz)
            self._load_adapter(current_adapter)
            self._apply_perturbation(z, sign=-1)
            outputs_neg, loss_neg, batch_mean = forward_and_get_loss_supervised(
                x, targets, self.model, self.fitness_lambda, self.train_info, 
                shift_vector, self.imagenet_mask, self.model_type, self.use_pure_ce, self.criterion_ce
            )
            if self.model_type in ['vit', 'deit']:
                embed_dim = model_info['embed_dim']
                batch_means.append(batch_mean[-embed_dim:].unsqueeze(0))
            elif self.model_type in ['swin', 'resnet']:
                batch_means.append(batch_mean.unsqueeze(0))
            del batch_mean
            
            # 使用公式: ∇f(x;ε) ≈ (f(x+εz) - f(x-εz))/(2ε) * z
            grad_i = (loss_pos - loss_neg) / (2 * self.epsilon) * z
            grad_estimate += grad_i
            
            print(f'扰动 [{i+1}/{self.pertub}], 梯度贡献范数: {torch.norm(grad_i).item():.6f}')
        
        # 平均所有扰动的梯度估计
        grad_estimate /= self.pertub
        
        # 使用优化器计算更新
        update = self.optimizer.step(grad_estimate)
        
        # 应用更新
        self._load_adapter(current_adapter)  # 重置到初始状态
        self._apply_perturbation(update, sign=1)  # 使用优化器计算的更新量
        
        # 计算最终输出和损失
        final_outputs, self.final_loss, _ = forward_and_get_loss_supervised(
            x, targets, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask, self.model_type, self.use_pure_ce, self.criterion_ce
        )
        losses.append(self.final_loss.item())
        
        # 更新历史统计信息
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        # 记录总体结束时间
        total_end = time.time()
        print(f"总计算完成，总耗时: {total_end - total_start:.4f}秒")
        print('perturbation min/max:', z.min().item(), z.max().item())
        
        return final_outputs

    def obtain_origin_stat(self, train_loader):
        """
        计算训练集特征的均值和方差，支持保存和加载计算结果。

        该函数用于计算训练集（源域）特征的均值和方差，为模型量化做准备。
        该函数首先尝试从保存的文件加载预计算的统计信息。
        如果找不到保存的数据，则重新计算并保存结果。
        先遍历训练集以提取特征，然后计算这些特征的均值和方差。
        最后，它为快速适应准备量化模型。

        参数:
        - train_loader: DataLoader 类型，训练集的数据加载器。

        返回:
        无
        """
        print('===> Start calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        model_name = f'{self.model_type}_adapter'
        save_path = os.path.join(save_dir, f'train_info_{model_name}.pt')
        
        if os.path.exists(save_path):
            print(f'===> 从文件加载预计算的均值和方差: {save_path}')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()               #dl[0]是图片，dl[1]是标签
                    feature = self.model.layers_cls_features(images)    #从ViT模型中提取图像特征
                    features.append(feature)
                    # break
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)#计算源域数据特征的均值和方差，dim=0表示计算每个特征的均值和方差
            del features#释放内存
            
            # 保存计算结果
            print('===> 保存计算结果到文件')
            torch.save({
                'train_info': self.train_info,
                'model_type': self.model_type,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)
        
        # 为快速适应准备量化模型
        num_layers = len(self.model.vit.blocks)  # 自动检测层数，vit_base_patch16_224有12层
        head_dim = self.model.vit.blocks[0].attn.head_dim  # 自动检测head维度，vit_base_patch16_224的head维度是64
        for _, m in self.model.vit.named_modules():  # 遍历ViT模型的所有模块
            if type(m) == PTQSLBatchingQuantMatMul:  # 如果是PTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(
                    torch.zeros((1,num_layers,197,head_dim)).cuda(),  # 移除prompt tokens
                    torch.zeros((1,num_layers,64,197)).cuda()   # 移除prompt tokens
                )
            elif type(m) == SoSPTQSLBatchingQuantMatMul:  # 如果是SoSPTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(
                    torch.zeros((1,num_layers,197,197)).cuda(),  # 移除prompt tokens
                    torch.zeros((1,num_layers,197,head_dim)).cuda()    # 移除prompt tokens
                )
        print('===> 计算均值和方差结束')

    def reset(self):
        """重置优化器和模型状态"""
        self.hist_stat = None
        self.model.reset_adapters()
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss_supervised(images, targets, model, fitness_lambda, train_info, 
                                      shift_vector, imagenet_mask, model_type, use_pure_ce, criterion_ce):
    """前向传播并计算有监督损失"""
    if model_type in ['vit', 'deit']:
        features = model.layers_cls_features_with_adapters(images)
        base_model = model.vit if model_type == 'vit' else model.deit
        embed_dim = base_model.embed_dim
        cls_features = features[:, -embed_dim:]
    elif model_type == 'swin':
        features = model.layers_features_with_adapters(images)
        cls_features = model.forward_features(images)
    elif model_type == 'resnet':
        features = model.layers_features_with_adapters(images)
        x = model.forward_features(images)
        cls_features = model.resnet.avgpool(x)
        cls_features = cls_features.view(cls_features.size(0), -1)
    
    batch_std, batch_mean = torch.std_mean(features, dim=0) #OOD域数据特征的均值和方差
    
    # Discrepancy loss (可选)
    if not use_pure_ce and train_info is not None:
        std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
        discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    else:
        discrepancy_loss = 0.0
    
    # 通过分类头获取输出
    if model_type in ['vit', 'deit']:
        base_model = model.vit if model_type == 'vit' else model.deit
        output = base_model.head(cls_features)
    elif model_type == 'swin':
        output = model.swin.head.fc(cls_features)
    elif model_type == 'resnet':
        output = model.resnet.fc(cls_features)
    
    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    
    # 交叉熵损失（有监督）
    ce_loss = criterion_ce(output, targets)
    loss = ce_loss if use_pure_ce else (discrepancy_loss + ce_loss)

    # 因为有标签，因此不用activation shifting了（冗余）
    
    return output, loss, batch_mean 