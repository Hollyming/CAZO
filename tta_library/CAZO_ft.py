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
    """Base optimizer class"""
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, grad_estimate):
        """Update parameters"""
        raise NotImplementedError

class SGD(Optimizer):
    """SGD optimizer"""
    def __init__(self, lr):
        super().__init__(lr)
    
    def step(self, grad_estimate):
        return -self.lr * grad_estimate     #w-lr*g

class SGD_Momentum(Optimizer):
    """SGD optimizer with momentum"""
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
        """Reset momentum"""
        self.momentum = None

class CAZO_FT(nn.Module):
    """
    CAZO_FT: CAZO算法的有标签微调版本
    
    使用零阶优化估计梯度和Hessian对角，但使用交叉熵损失进行有监督训练
    支持多种模型架构：AdaFormerViT, DeiTAdapter, SwinAdapter, ResNetAdapter
    """
    def __init__(self, model, fitness_lambda=0.4, lr=0.01, 
                 pertub=20, epsilon=0.1, optimizer_type='sgd', beta=0.9, nu=0.1, use_pure_ce=False):
        """
        Initialize CAZO_FT algorithm
        
        Args:
            model: 支持AdaFormerViT, DeiTAdapter, SwinAdapter, ResNetAdapter
            fitness_lambda: balance factor for fitness function
            lr: learning rate
            pertub: number of perturbations k
            epsilon: perturbation size ε
            optimizer_type: optimizer type, 'sgd' or 'sgd_momentum'
            beta: momentum coefficient
            nu: decay factor for diagonal Hessian estimation matrix
            use_pure_ce: if True, only use cross-entropy loss without mse loss
        """
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.epsilon = epsilon
        self.pertub = pertub
        self.use_pure_ce = use_pure_ce
        
        self.model = model
        # 检测模型类型
        self.model_type = self._detect_model_type()
        
        # ensure all adapter parameters do not require gradients
        for adapter in self.model.adapters.values():
            for param in adapter.parameters():
                param.requires_grad_(False)
        
        # save best adapter parameters
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        self.best_loss = np.inf
        self.final_loss = np.inf
        self.hist_stat = None
        self.train_info = None
        self.imagenet_mask = None
        
        # initialize diagonal Hessian estimation matrix D as identity matrix
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        self.D = torch.ones(total_params, device='cuda')
        
        # decay factor for diagonal Hessian estimation matrix
        self.nu = nu
        
        # gradient estimate from previous step
        self.prev_grad_estimate = None
        
        # time step counter
        self.t = 0
        
        # 交叉熵损失
        self.criterion_ce = nn.CrossEntropyLoss().cuda()
        
        # initialize optimizer
        if optimizer_type == 'sgd':
            self.optimizer = SGD(lr)
        elif optimizer_type == 'sgd_momentum':
            self.optimizer = SGD_Momentum(lr, beta)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
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
            # ViT/DeiT: 获取transformer blocks数量和embed_dim
            base_model = self.model.vit if self.model_type == 'vit' else self.model.deit
            num_layers = len(base_model.blocks)
            embed_dim = base_model.embed_dim
            head_dim = base_model.blocks[0].attn.head_dim
            return {
                'num_layers': num_layers,
                'embed_dim': embed_dim,
                'head_dim': head_dim,
                'feature_dim': num_layers * embed_dim  # 所有层的CLS token特征拼接
            }
        elif self.model_type == 'swin':
            # Swin Transformer: 获取所有blocks总数和各stage维度
            total_blocks = self.model.total_blocks
            stage_dims = []
            for stage_idx, layer in enumerate(self.model.swin.layers):
                stage_dim = int(self.model.swin.embed_dim * 2 ** stage_idx)
                stage_dims.extend([stage_dim] * len(layer.blocks))
            feature_dim = sum(stage_dims)  # 所有block特征拼接
            return {
                'num_layers': total_blocks,
                'stage_dims': stage_dims,
                'feature_dim': feature_dim
            }
        elif self.model_type == 'resnet':
            # ResNet: 获取所有bottleneck数量和维度
            total_blocks = self.model.total_blocks
            block_dims = self.model.block_dims
            feature_dim = sum(block_dims)  # 所有block特征拼接
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
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def _get_shift_vector(self):
        if self.hist_stat is None:
            return None
        else:
            # return self.train_info[1] [-768:] - self.hist_stat  # 默认使用最后768维度作为shift vector
            if self.model_type in ['vit', 'deit']:
                model_info = self._get_model_info()
                embed_dim = model_info['embed_dim']
                return self.train_info[1][-embed_dim:] - self.hist_stat
            elif self.model_type == 'swin':
                # Swin: 使用最后一个stage的维度(768 for Swin-Tiny)
                final_dim = self.model.swin.num_features
                return self.train_info[1][-final_dim:] - self.hist_stat[-final_dim:]
            elif self.model_type == 'resnet':
                model_info = self._get_model_info()
                final_dim = model_info['block_dims'][-1]
                return self.train_info[1][-final_dim:] - self.hist_stat[-final_dim:]
    
    def _sample_perturbations(self, num_samples, dim):
        """
        Sample k perturbation vectors based on inverse of diagonal Hessian estimation matrix
        
        Args:
            num_samples: number of samples k
            dim: parameter dimension
            
        Returns:
            List of perturbation vectors
        """

        if self.t == 0:
            # initial time step, use D directly
            H_diag = self.D
        else:
            # use formula: H̃_t = diag(D_t / (1 - (1 - ν)^t))
            H_diag = self.D / (1 - (1 - self.nu)**self.t)
        
        # sample from standard normal distribution, then multiply by inverse square root of H_t diagonal elements
        perturbations = []
        for _ in range(num_samples):
            z = torch.randn(dim, device='cuda')
            scaled_z = z / torch.sqrt(H_diag)
            perturbations.append(scaled_z)
            
        return perturbations
    
    def _apply_perturbation(self, perturbation, sign=1):
        """Apply perturbation to all adapter parameters"""
        start_idx = 0
        
        # traverse all adapter parameters
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
        """Save deep copy of all adapter parameters"""
        return {
            f'adapter_{layer}': {
                k: v.clone() for k, v in self.model.adapters[f'adapter_{layer}'].state_dict().items()
            } for layer in self.model.adapter_layers
        }

    def _load_adapter(self, adapter_states):
        """Load all adapter parameters"""
        for layer in self.model.adapter_layers:
            self.model.adapters[f'adapter_{layer}'].load_state_dict(
                adapter_states[f'adapter_{layer}']
            )
    
    def forward(self, x, targets):
        """
        Use zero-order optimization with Hessian approximation and supervised loss
        
        Args:
            x: input images
            targets: labels (supervised)
        """
        shift_vector = self._get_shift_vector()

        # initialize variables to track best loss and outputs
        self.best_loss, self.best_outputs, batch_means = np.inf, None, []
        
        # save current model adapter parameters
        current_adapter = self._save_current_adapter()
        losses = []
        
        # record total start time
        total_start = time.time()
        
        # get total number of parameters
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        
        # initialize gradient estimate as zero vector
        grad_estimate = torch.zeros(total_params, device='cuda')
        
        # generate k perturbations based on Hessian diagonal
        perturbations = self._sample_perturbations(self.pertub, total_params)
        
        # use zero-order method to estimate gradient
        print("Start using zero-order method to estimate gradient (supervised)...")
        for i, z in enumerate(perturbations):
            # positive perturbation: f(x + εz)
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
                # Swin和ResNet使用全部特征
                batch_means.append(batch_mean.unsqueeze(0))
            del batch_mean
            
            # negative perturbation: f(x - εz)
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
            
            # use formula: ĝ(x_t+1) = 1/k ∑ (L(w_t + μū_i) - L(w_t - μū_i))/(2μ) * ū_i
            grad_i = (loss_pos - loss_neg) / (2 * self.epsilon) * z
            grad_estimate += grad_i
            
            print(f'Perturbation [{i+1}/{self.pertub}], Gradient contribution norm: {torch.norm(grad_i).item():.6f}')
        
        # average gradient estimates of all perturbations
        grad_estimate /= self.pertub
        
        # update diagonal Hessian estimation matrix D
        if self.prev_grad_estimate is not None:
            self.D = (1 - self.nu) * self.D + self.nu * grad_estimate**2
        
        # save current gradient estimate for next iteration
        self.prev_grad_estimate = grad_estimate.clone()
        
        # increment time step
        self.t += 1
        
        # use optimizer to calculate update
        update = self.optimizer.step(grad_estimate)
        
        # apply update
        self._load_adapter(current_adapter)  # reset to initial state
        self._apply_perturbation(update, sign=1)
        
        final_outputs, self.final_loss, _ = forward_and_get_loss_supervised(
            x, targets, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask, self.model_type, self.use_pure_ce, self.criterion_ce
        )
        losses.append(self.final_loss.item())
        
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        total_end = time.time()
        print(f"Total computation completed, total time: {total_end - total_start:.4f} seconds")
        print('perturbations min/max:', z.min().item(), z.max().item())
        
        return final_outputs

    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        # 根据模型类型生成不同的文件名
        model_name = f'{self.model_type}_adapter'
        save_path = os.path.join(save_dir, f'train_info_{model_name}.pt')
        
        # 尝试加载已保存的统计信息
        if os.path.exists(save_path):
            print(f'===> 从文件加载预计算的均值和方差: {save_path}')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    # 根据模型类型提取特征
                    feature = self._extract_features(images, with_adapter=False)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)
            del features

            # 保存计算结果
            print(f'===> 保存计算结果到文件: {save_path}')
            torch.save({
                'train_info': self.train_info,
                'model_type': self.model_type,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)

        # preparing quantized model (仅对ViT/DeiT有效)
        if self.model_type in ['vit', 'deit']:
            model_info = self._get_model_info()
            num_layers = model_info['num_layers']
            head_dim = model_info['head_dim']
            
            base_model = self.model.vit if self.model_type == 'vit' else self.model.deit
            for _, m in base_model.named_modules():
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
        """Reset optimizer, model state and Hessian estimation"""
        self.hist_stat = None
        self.model.reset_adapters()
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        self.D = torch.ones(total_params, device='cuda')
        self.prev_grad_estimate = None
        self.t = 0

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss_supervised(images, targets, model, fitness_lambda, train_info, 
                                      shift_vector, imagenet_mask, model_type, use_pure_ce, criterion_ce):
    """Forward propagation and calculate supervised loss"""
    if model_type in ['vit', 'deit']:
        # ViT/DeiT: 提取所有层的CLS token特征
        features = model.layers_cls_features_with_adapters(images)
        # 获取embed_dim
        base_model = model.vit if model_type == 'vit' else model.deit
        embed_dim = base_model.embed_dim
        cls_features = features[:, -embed_dim:]
    elif model_type == 'swin':
        # Swin Transformer: 提取所有block的平均特征
        features = model.layers_features_with_adapters(images)
        # 使用前向传播获取最终分类特征
        cls_features = model.forward_features(images)
    elif model_type == 'resnet':
        # ResNet: 提取所有block的平均池化特征
        features = model.layers_features_with_adapters(images)
        # 获取最终分类特征
        x = model.forward_features(images)
        cls_features = model.resnet.avgpool(x)
        cls_features = cls_features.view(cls_features.size(0), -1)
    
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    
    # Discrepancy loss (optional)
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
        # Swin: cls_features已经是(B, C)，直接用fc层
        output = model.swin.head.fc(cls_features)
    elif model_type == 'resnet':
        output = model.resnet.fc(cls_features)
    
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    
    # Cross-entropy loss (supervised)
    ce_loss = criterion_ce(output, targets)
    loss = ce_loss if use_pure_ce else (discrepancy_loss + ce_loss)
    
    return output, loss, batch_mean 