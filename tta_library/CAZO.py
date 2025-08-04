import torch
import torch.nn as nn
import numpy as np
import time
import os
from models.adaformer import AdaFormerViT

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

class CAZO(nn.Module):
    """
    CAZO: Zero-order optimization algorithm based on diagonal Hessian approximation
    This algorithm improves gradient estimation by sampling perturbation vectors using the inverse of diagonal Hessian estimation matrix
    """
    def __init__(self, model: AdaFormerViT, fitness_lambda=0.4, lr=0.01, 
                 pertub=20, epsilon=0.1, optimizer_type='sgd', beta=0.9, nu=0.1):
        """
        Initialize CAZO algorithm
        
        Args:
            model: AdaFormerViT model
            fitness_lambda: balance factor for fitness function
            lr: learning rate
            pertub: number of perturbations k
            epsilon: perturbation size ε
            optimizer_type: optimizer type, 'sgd' or 'sgd_momentum'
            beta: momentum coefficient
            nu: decay factor for diagonal Hessian estimation matrix
        """
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.epsilon = epsilon
        self.pertub = pertub
        
        self.model = model
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
        
        # initialize optimizer
        if optimizer_type == 'sgd':
            self.optimizer = SGD(lr)
        elif optimizer_type == 'sgd_momentum':
            self.optimizer = SGD_Momentum(lr, beta)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _update_hist(self, batch_mean):
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def _get_shift_vector(self):
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat
    
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
    
    def forward(self, x):
        """
        Use zero-order optimization method based on diagonal Hessian approximation to optimize adapter parameters
        """
        # get shift vector for calculating shift direction
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
        
        # generate k standard normal distribution perturbations
        perturbations = self._sample_perturbations(self.pertub, total_params)
        
        # use zero-order method to estimate gradient
        print("Start using zero-order method to estimate gradient...")
        for i, z in enumerate(perturbations):
            # positive perturbation: f(x + εz)
            self._load_adapter(current_adapter)
            self._apply_perturbation(z, sign=1)
            outputs_pos, loss_pos, batch_mean = forward_and_get_loss(
                x, self.model, self.fitness_lambda, self.train_info, 
                shift_vector, self.imagenet_mask
            )
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean
            
            # negative perturbation: f(x - εz)
            self._load_adapter(current_adapter)
            self._apply_perturbation(z, sign=-1)
            outputs_neg, loss_neg, batch_mean = forward_and_get_loss(
                x, self.model, self.fitness_lambda, self.train_info, 
                shift_vector, self.imagenet_mask
            )
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean
            
            # use formula: ĝ(x_t+1) = 1/k ∑ (L(w_t + μū_i) - L(w_t - μū_i))/(2μ) * ū_i
            grad_i = (loss_pos - loss_neg) / (2 * self.epsilon) * z
            grad_estimate += grad_i
            
            print(f'Perturbation [{i+1}/{self.pertub}], Gradient contribution norm: {torch.norm(grad_i).item():.6f}')
        
        # average gradient estimates of all perturbations
        grad_estimate /= self.pertub
        
        # update diagonal Hessian estimation matrix D
        if self.prev_grad_estimate is not None:
            # D_t = (1 - ν)D_{t-1} + νĝ²(x_{t-1})
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
        
        final_outputs, self.final_loss, _ = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask
        )
        losses.append(self.final_loss.item())
        
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        total_end = time.time()
        print(f"Total computation completed, total time: {total_end - total_start:.4f} seconds")
        print('perturbations min/max:', z.min().item(), z.max().item())
        
        return final_outputs
    
    # def obtain_origin_stat(self, train_loader):
    #     """
    #     Calculate mean and variance of training set features, support saving and loading calculation results.

    #     This function is used to calculate the mean and variance of training set (source domain) features for model quantization preparation.
    #     This function first tries to load precalculated statistics from saved files.
    #     If no saved data is found, it recalculates and saves the results.
    #     First traverse the training set to extract features, then calculate the mean and variance of these features.
    #     Finally, it prepares the quantized model for quick adaptation.

    #     Args:
    #     - train_loader: DataLoader type, training set data loader.

    #     Returns:
    #     None
    #     """
    #     print('===> Start calculating mean and variance')
    #     save_dir = os.path.join('dataset', 'train_stats')
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     save_path = os.path.join(save_dir, f'train_info_adapter.pt')
        
    #     if os.path.exists(save_path):
    #         print('===> Load precalculated mean and variance from file')
    #         saved_data = torch.load(save_path)
    #         self.train_info = saved_data['train_info']
    #     else:
    #         print('===> Start calculating mean and variance')
    #         features = []
    #         with torch.no_grad():
    #             for _, dl in enumerate(train_loader):
    #                 images = dl[0].cuda()               #dl[0] is image, dl[1] is label
    #                 feature = self.model.layers_cls_features(images)    # extract image features from ViT model
    #                 features.append(feature)
    #             features = torch.cat(features, dim=0)
    #             self.train_info = torch.std_mean(features, dim=0)# calculate mean and variance of source domain data features, dim=0 means calculating mean and variance of each feature
    #         del features# release memory
            
    #         # save calculation results
    #         print('===> Save calculation results to file')
    #         torch.save({
    #             'train_info': self.train_info,
    #             'timestamp': time.strftime("%Y%m%d-%H%M%S")
    #         }, save_path)
        
    #     for _, m in self.model.vit.named_modules():  # traverse all modules of ViT model
    #         if type(m) == PTQSLBatchingQuantMatMul:  # if PTQSLBatchingQuantMatMul type
    #             m._get_padding_parameters(
    #                 torch.zeros((1,12,197,64)).cuda(),  # remove prompt tokens
    #                 torch.zeros((1,12,64,197)).cuda()   # remove prompt tokens
    #             )
    #         elif type(m) == SoSPTQSLBatchingQuantMatMul:  # if SoSPTQSLBatchingQuantMatMul type
    #             m._get_padding_parameters(
    #                 torch.zeros((1,12,197,197)).cuda(),  # remove prompt tokens
    #                 torch.zeros((1,12,197,64)).cuda()    # remove prompt tokens
    #             )
    #     print('===> Calculating mean and variance finished')

    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'train_info_adapter.pt')
        
        # 尝试加载已保存的统计信息
        if os.path.exists(save_path):
            print('===> 从文件加载预计算的均值和方差')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.layers_cls_features(images)
                    features.append(feature)
                    # break
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)
            del features

            # 保存计算结果
            print('===> 保存计算结果到文件')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)

        # preparing quantized model for prompt adaptation
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

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Calculate softmax distribution entropy"""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model: AdaFormerViT, fitness_lambda, train_info, shift_vector, imagenet_mask):
    """Forward propagation and calculate loss"""
    features = model.layers_cls_features_with_adapters(images)# extract 12 layer cls features, turn (64,12*768)=(64,9216)
    cls_features = features[:, -768:] # the feature of classification token, 768-dimensional feature of the last classification token of ViT model
    batch_std, batch_mean = torch.std_mean(features, dim=0) #OOD domain data feature mean and variance
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])#ID and OOD domain data feature MSE loss
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    output = model.vit.head(cls_features)# through classification head, map cls_features to classification output
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]# if needed, use imagenet_mask to filter output
    entropy_loss = softmax_entropy(output).sum()# sum over batch to get entropy loss
    loss = discrepancy_loss + entropy_loss
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)# add activate shift offset, cover original output
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean 