"""
CAZO for ResNet - Covariance-Aware Zero-Order Optimization for ResNet
Adapted from CAZO.py to work with ResNet50 using PromptResNet
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os

from models.prompt_resnet import PromptResNet


class CAZO_ResNet(nn.Module):
    """
    CAZO adapted for ResNet50 architecture
    Uses PromptResNet for parameter-efficient adaptation
    """
    def __init__(self, args, model: PromptResNet, fitness_lambda=0.4, lr=0.01, 
                 pertub=20, epsilon=0.1, optimizer_type='sgd', beta=0.9, nu=0.8):
        super().__init__()
        self.args = args
        self.model = model
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.pertub = pertub
        self.epsilon = epsilon
        self.optimizer_type = optimizer_type
        self.beta = beta
        self.nu = nu  # decay factor for diagonal Hessian
        
        # Get trainable parameters (prompts)
        self.params = [self.model.padding.weight]
        self.param_shapes = [p.shape for p in self.params]
        self.total_params = sum(p.numel() for p in self.params)
        
        # Initialize optimizer state
        if self.optimizer_type == 'sgd':
            self.velocity = [torch.zeros_like(p) for p in self.params]
        elif self.optimizer_type == 'sgd_momentum':
            self.velocity = [torch.zeros_like(p) for p in self.params]
        
        # Initialize diagonal Hessian approximation
        self.diag_hessian = [torch.ones_like(p) * 1e-4 for p in self.params]
        
        # Statistics
        self.num_features = model.num_features
        self.best_loss = np.inf
        self.final_loss = np.inf
        self.hist_stat = None
        self.train_info = None
        
    def _update_hist(self, batch_mean):
        """Update overall test statistics"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction"""
        if self.hist_stat is None or self.train_info is None:
            return None
        else:
            return self.train_info[1][-1] - self.hist_stat
    
    def _estimate_gradient_and_hessian(self, x, shift_vector):
        """
        Estimate gradient and diagonal Hessian using zero-order methods
        """
        # Flatten parameters
        param_vec = torch.cat([p.flatten() for p in self.params])
        
        # Store original parameters
        original_params = [p.clone() for p in self.params]
        
        # Sample random directions
        gradients = []
        hessian_samples = []
        
        for _ in range(self.pertub):
            # Sample random direction
            u = torch.randn_like(param_vec)
            u = u / (torch.norm(u) + 1e-8)
            
            # Forward perturbation
            self._update_params(param_vec + self.epsilon * u)
            _, loss_plus, _ = forward_and_get_loss(
                x, self.model, self.fitness_lambda, 
                self.train_info, shift_vector, self.imagenet_mask
            )
            
            # Backward perturbation
            self._update_params(param_vec - self.epsilon * u)
            _, loss_minus, _ = forward_and_get_loss(
                x, self.model, self.fitness_lambda,
                self.train_info, shift_vector, self.imagenet_mask
            )
            
            # Gradient estimation: (f(x+εu) - f(x-εu)) / (2ε)
            grad_estimate = (loss_plus.item() - loss_minus.item()) / (2 * self.epsilon)
            gradients.append(grad_estimate * u)
            
            # Hessian diagonal estimation: (f(x+εu) + f(x-εu) - 2f(x)) / ε²
            # First get f(x)
            self._update_params(param_vec)
            _, loss_curr, _ = forward_and_get_loss(
                x, self.model, self.fitness_lambda,
                self.train_info, shift_vector, self.imagenet_mask
            )
            
            hess_estimate = (loss_plus.item() + loss_minus.item() - 2 * loss_curr.item()) / (self.epsilon ** 2)
            hessian_samples.append(hess_estimate * (u ** 2))
        
        # Average gradient estimates
        grad_vec = torch.stack(gradients).mean(dim=0)
        
        # Average Hessian diagonal estimates
        hess_diag_vec = torch.stack(hessian_samples).mean(dim=0)
        
        # Restore original parameters
        for p, orig_p in zip(self.params, original_params):
            p.data.copy_(orig_p)
        
        # Reshape gradients
        gradients_list = []
        hessian_list = []
        start_idx = 0
        for shape in self.param_shapes:
            numel = np.prod(shape)
            gradients_list.append(grad_vec[start_idx:start_idx+numel].reshape(shape))
            hessian_list.append(hess_diag_vec[start_idx:start_idx+numel].reshape(shape))
            start_idx += numel
        
        return gradients_list, hessian_list
    
    def _update_params(self, param_vec):
        """Update parameters from flattened vector"""
        start_idx = 0
        for p, shape in zip(self.params, self.param_shapes):
            numel = np.prod(shape)
            p.data = param_vec[start_idx:start_idx+numel].reshape(shape)
            start_idx += numel
    
    def _update_diag_hessian(self, new_hessian):
        """Update diagonal Hessian with exponential moving average"""
        for i in range(len(self.diag_hessian)):
            self.diag_hessian[i] = self.nu * self.diag_hessian[i] + (1 - self.nu) * torch.abs(new_hessian[i])
    
    def _apply_optimizer_step(self, gradients):
        """Apply optimizer step with curvature-aware scaling"""
        if self.optimizer_type == 'sgd':
            for p, g, h in zip(self.params, gradients, self.diag_hessian):
                # Curvature-aware step: scale by inverse of Hessian diagonal
                step = g / (h + 1e-8)
                p.data.sub_(self.lr * step)
                
        elif self.optimizer_type == 'sgd_momentum':
            for i, (p, g, h) in enumerate(zip(self.params, gradients, self.diag_hessian)):
                # Curvature-aware step with momentum
                step = g / (h + 1e-8)
                self.velocity[i] = self.beta * self.velocity[i] + step
                p.data.sub_(self.lr * self.velocity[i])

    def forward(self, x):
        """Forward pass with CAZO adaptation"""
        shift_vector = self._get_shift_vector()
        
        # Estimate gradients and Hessian
        gradients, hessian_estimates = self._estimate_gradient_and_hessian(x, shift_vector)
        
        # Update diagonal Hessian
        self._update_diag_hessian(hessian_estimates)
        
        # Apply optimizer step
        self._apply_optimizer_step(gradients)
        
        # Forward pass with updated parameters
        outputs, loss, batch_mean = forward_and_get_loss(
            x, self.model, self.fitness_lambda,
            self.train_info, shift_vector, self.imagenet_mask
        )
        
        # Update statistics
        batch_means = batch_mean[-self.num_features:].unsqueeze(0)
        self._update_hist(batch_means)
        
        # Record final loss
        self.final_loss = loss.item()
        
        return outputs
    
    def obtain_origin_stat(self, train_loader):
        """Calculate and cache training statistics"""
        print('===> begin calculating mean and variance')
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'train_info_promptresnet.pt')
        self.model.eval()

        if os.path.exists(save_path):
            print('===> Loading precomputed statistics from file')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> Computing statistics')
            features, layer_stds, layer_means = [], [], []
            with torch.no_grad():
                for i, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.forward_features(images)
                    features.append(feature)
                    if i == 24: 
                        break

            for i in range(len(features[0])):
                layer_features = [feature[i] for feature in features]
                layer_features = torch.cat(layer_features, dim=0).cuda()

                if len(layer_features.shape) == 4:
                    dim = (0, 2, 3)
                else:
                    dim = (0)

                layer_stds.append(layer_features.std(dim=dim))
                layer_means.append(layer_features.mean(dim=dim))

            layer_stds = torch.cat(layer_stds, dim=0)
            layer_means = torch.cat(layer_means, dim=0)
            self.train_info = (layer_stds, layer_means)
            
            print('===> Saving computed statistics to file')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)
        
        print('===> Statistics computation complete')

    def reset(self):
        """Reset model to initial state"""
        # Reset optimizer state
        if hasattr(self, 'velocity'):
            self.velocity = [torch.zeros_like(p) for p in self.params]
        
        # Reset Hessian
        self.diag_hessian = [torch.ones_like(p) * 1e-4 for p in self.params]
        
        # Reset statistics
        self.hist_stat = None
        self.best_loss = np.inf
        self.final_loss = np.inf


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


criterion_mse = nn.MSELoss(reduction='mean').cuda()


@torch.no_grad()
def forward_and_get_loss(images, model: PromptResNet, fitness_lambda, 
                        train_info, shift_vector, imagenet_mask):
    """Forward pass and loss calculation"""
    features = model.forward_features_with_prompts(images)
    discrepancy_loss = 0

    loss_std, loss_mean = 0, 0
    si, ei = 0, 0
    
    # Discrepancy loss
    for i in range(len(features)):
        layer_features = features[i]
        si = ei
        ei = si + features[i].shape[1]
        
        if len(layer_features.shape) == 4:
            dim = (0, 2, 3)
        else:
            dim = (0)

        batch_std, batch_mean = layer_features.std(dim=dim), layer_features.mean(dim=dim)
        loss_std += criterion_mse(batch_std, train_info[0][si:ei])
        loss_mean += criterion_mse(batch_mean, train_info[1][si:ei])
    
    loss_std = loss_std / len(features)
    loss_mean = loss_mean / len(features)
    discrepancy_loss += loss_std + loss_mean
    
    # Classification
    cls_features = features[-1]
    output = model.model.fc(cls_features)

    # Entropy loss
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).mean()
    loss = fitness_lambda * discrepancy_loss + entropy_loss

    # Activation shifting
    if shift_vector is not None:
        output = model.model.fc(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean


def configure_model(model):
    """Configure model for CAZO"""
    model.train()
    model.requires_grad_(False)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    
    return model