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
        return -self.lr * grad_estimate

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

class HiZOO_TTA(nn.Module):
    """
    HiZOO: Zero-order optimization algorithm based on diagonal Hessian approximation using second-order finite difference
    This algorithm improves gradient estimation by using curvature information estimated via f(x+z), f(x-z), and f(x)
    """
    def __init__(self, model: AdaFormerViT, fitness_lambda=0.4, lr=0.01, 
                 pertub=20, epsilon=0.1, optimizer_type='sgd', beta=0.9, 
                 hessian_smooth=0.1):
        """
        Initialize HiZOO algorithm
        
        Args:
            model: AdaFormerViT model
            fitness_lambda: balance factor for fitness function
            lr: learning rate
            pertub: number of perturbations k
            epsilon: perturbation size ε
            optimizer_type: optimizer type, 'sgd' or 'sgd_momentum'
            beta: momentum coefficient
            hessian_smooth: smoothing factor for Hessian estimation (corresponds to nu in HiZOO)
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
        
        # initialize diagonal Hessian estimation matrix H as identity matrix
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        self.Hessian_matrix = torch.ones(total_params, device='cuda')
        
        # smoothing factor for Hessian estimation (EMA coefficient)
        self.hessian_smooth = hessian_smooth
        
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
        Use HiZOO zero-order optimization method with second-order Hessian approximation
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
        
        # Step 1: Evaluate loss at current point f(θ)
        print("Evaluating loss at current point...")
        self._load_adapter(current_adapter)
        outputs_original, loss_original, batch_mean = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info, 
            shift_vector, self.imagenet_mask
        )
        batch_means.append(batch_mean[-768:].unsqueeze(0))
        print(f"Original loss: {loss_original.item():.6f}")
        
        # Step 2: Sample perturbations and estimate gradient using HiZOO method
        print("Start using HiZOO method to estimate gradient and update Hessian...")
        
        # Use a single perturbation for Hessian estimation (HiZOO style)
        # Generate random perturbation
        z = torch.randn(total_params, device='cuda')
        
        # Scale perturbation by inverse square root of Hessian diagonal
        scaled_z = z / torch.sqrt(self.Hessian_matrix)
        
        # Positive perturbation: f(θ + ε * z/√H)
        self._load_adapter(current_adapter)
        self._apply_perturbation(scaled_z, sign=1)
        outputs_pos, loss_pos, batch_mean = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info, 
            shift_vector, self.imagenet_mask
        )
        batch_means.append(batch_mean[-768:].unsqueeze(0))
        print(f"Loss at +perturbation: {loss_pos.item():.6f}")
        
        # Negative perturbation: f(θ - ε * z/√H)
        self._load_adapter(current_adapter)
        self._apply_perturbation(scaled_z, sign=-1)
        outputs_neg, loss_neg, batch_mean = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info, 
            shift_vector, self.imagenet_mask
        )
        batch_means.append(batch_mean[-768:].unsqueeze(0))
        print(f"Loss at -perturbation: {loss_neg.item():.6f}")
        
        # Step 3: Update Hessian matrix using second-order finite difference
        # H̃ = |f(θ+εz) + f(θ-εz) - 2f(θ)| / (2ε²) * H * z²
        Hessian_temp = self.Hessian_matrix * z * z
        second_order_diff = torch.abs(loss_pos + loss_neg - 2 * loss_original)
        Hessian_estimator = (second_order_diff * Hessian_temp * self.hessian_smooth / 
                            (2 * self.epsilon * self.epsilon))
        
        # Update Hessian with exponential moving average
        # H_new = (1 - λ) * H_old + Ĥ
        self.Hessian_matrix = ((1 - self.hessian_smooth) * self.Hessian_matrix + 
                               Hessian_estimator)
        
        print(f"Hessian statistics - min: {self.Hessian_matrix.min().item():.6f}, "
              f"max: {self.Hessian_matrix.max().item():.6f}, "
              f"mean: {self.Hessian_matrix.mean().item():.6f}")
        
        # Step 4: Estimate gradient using first-order finite difference
        # ĝ = (f(θ+εz) - f(θ-εz)) / (2ε) * z / √H
        grad_estimate = (loss_pos - loss_neg) / (2 * self.epsilon) * z / torch.sqrt(self.Hessian_matrix)
        
        print(f"Gradient estimate norm: {torch.norm(grad_estimate).item():.6f}")
        
        # Step 5: Use optimizer to calculate update
        update = self.optimizer.step(grad_estimate)
        
        # Step 6: Apply update
        self._load_adapter(current_adapter)  # reset to initial state
        self._apply_perturbation(update, sign=1)
        
        final_outputs, self.final_loss, _ = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask
        )
        losses.append(self.final_loss.item())
        
        print(f"Final loss after update: {self.final_loss.item():.6f}")
        
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        
        total_end = time.time()
        print(f"Total computation completed, total time: {total_end - total_start:.4f} seconds")
        print(f"Perturbations min/max: {z.min().item():.4f}, {z.max().item():.4f}")
        
        return final_outputs

    def obtain_origin_stat(self, train_loader):
        """
        Calculate mean and variance of training set features, support saving and loading calculation results.
        """
        print('===> begin calculating mean and variance')
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'train_info_adapter.pt')
        
        if os.path.exists(save_path):
            print('===> Load precalculated mean and variance from file')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> Start calculating mean and variance')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.layers_cls_features(images)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)
            del features

            print('===> Save calculation results to file')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)

        # preparing quantized model for prompt adaptation
        num_layers = len(self.model.vit.blocks)
        head_dim = self.model.vit.blocks[0].attn.head_dim
        for _, m in self.model.vit.named_modules():
            if type(m) == PTQSLBatchingQuantMatMul:
                m._get_padding_parameters(
                    torch.zeros((1,num_layers,197,head_dim)).cuda(),
                    torch.zeros((1,num_layers,64,197)).cuda()
                )
            elif type(m) == SoSPTQSLBatchingQuantMatMul:
                m._get_padding_parameters(
                    torch.zeros((1,num_layers,197,197)).cuda(),
                    torch.zeros((1,num_layers,197,head_dim)).cuda()
                )
        print('===> Calculating mean and variance finished')

    def reset(self):
        """Reset optimizer, model state and Hessian estimation"""
        self.hist_stat = None
        self.model.reset_adapters()
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        self.Hessian_matrix = torch.ones(total_params, device='cuda')

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
    features = model.layers_cls_features_with_adapters(images)
    cls_features = features[:, -768:]
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    output = model.vit.head(cls_features)
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).sum()
    loss = discrepancy_loss + entropy_loss
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean