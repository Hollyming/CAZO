"""
DeYO: Destroying Your Object-centric Inductive Biases
Adapted from LCoTTA: https://github.com/Jhyun17/DeYO
Paper: https://openreview.net/pdf?id=9w3iw8wDuE

Integrated into CAZO framework for single-round testing (non-lifelong setting)
"""

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange


class DeYO(nn.Module):
    """
    DeYO: Destroying Your Object with entropy minimization and PLPD filtering
    Adapted for CAZO framework
    """
    def __init__(self, model, optimizer, num_classes=1000, 
                 reweight_ent=True, reweight_plpd=True, plpd_threshold=0.2,
                 margin=0.5, margin_e0=0.4, aug_type='pixel',
                 occlusion_size=112, row_start=56, column_start=56, patch_len=4,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.num_classes = num_classes
        
        # DeYO specific parameters
        self.reweight_ent = reweight_ent
        self.reweight_plpd = reweight_plpd
        self.plpd_threshold = plpd_threshold
        self.deyo_margin = margin * math.log(num_classes)
        self.margin_e0 = margin_e0 * math.log(num_classes)
        
        # Augmentation parameters
        self.aug_type = aug_type  # 'occ', 'patch', or 'pixel'
        self.occlusion_size = occlusion_size
        self.row_start = row_start
        self.column_start = column_start
        self.patch_len = patch_len
        
        # Save initial state for reset
        self.model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.optimizer_state = {k: v if not isinstance(v, torch.Tensor) else v.clone() 
                                for k, v in self.optimizer.state_dict().items()}
        
        # Configure model
        self._configure_model()
        
        self.imagenet_mask = None  # For ImageNet-R compatibility
        self.num_forwards = 0
        self.num_backwards = 0
        self.final_loss = None

    def _configure_model(self):
        """Configure model for use with DeYO"""
        self.model.eval()  # eval mode to avoid stochastic depth
        self.model.requires_grad_(False)
        
        # Configure norm for DeYO updates: enable grad
        for nm, m in self.model.named_modules():
            # Skip top layers for adaptation
            if 'layer4' in nm or 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm:
                continue
            if 'norm.' in nm or nm in ['norm']:
                continue
                
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # Force use of batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, loss_value = self.forward_and_adapt(x)
            self.num_forwards += x.size(0)
            if loss_value is not None:
                self.num_backwards += x.size(0)
            self.final_loss = loss_value
            
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Forward and adapt model on batch of data
        """
        imgs_test = x
        outputs = self.model(imgs_test)
        
        # Apply ImageNet-R mask if needed
        if self.imagenet_mask is not None:
            outputs = outputs[:, self.imagenet_mask]
        
        # Calculate entropy
        entropys = softmax_entropy(outputs)
        
        # First filter: remove high-entropy samples
        filter_ids_1 = torch.where(entropys < self.deyo_margin)
        entropys_filtered = entropys[filter_ids_1]
        
        if len(entropys_filtered) == 0:
            return outputs, None
        
        # Create augmented views
        x_prime = imgs_test[filter_ids_1].detach()
        x_prime = self._apply_augmentation(x_prime, imgs_test.shape[-1])
        
        # Get predictions on augmented data
        with torch.no_grad():
            outputs_prime = self.model(x_prime)
            if self.imagenet_mask is not None:
                outputs_prime = outputs_prime[:, self.imagenet_mask]
        
        # Calculate PLPD (Pseudo Label Probability Drop)
        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        plpd = (torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - 
                torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1)))
        plpd = plpd.reshape(-1)
        
        # Second filter: keep samples with high PLPD
        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys_final = entropys_filtered[filter_ids_2]
        
        if len(entropys_final) == 0:
            return outputs, None
        
        plpd_final = plpd[filter_ids_2]
        
        # Apply reweighting
        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp((entropys_final.clone().detach() - self.margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd_final.clone().detach()))))
            entropys_final = entropys_final.mul(coeff)
        
        loss = entropys_final.mean(0)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outputs, loss.item()

    def _apply_augmentation(self, x_prime, img_size):
        """Apply destructive augmentation based on aug_type"""
        if self.aug_type == 'occ':
            # Occlusion augmentation
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, 
                    self.column_start:self.column_start + self.occlusion_size] = occlusion_window
                    
        elif self.aug_type == 'patch':
            # Patch shuffling
            resize_t = torchvision.transforms.Resize(((img_size // self.patch_len) * self.patch_len,
                                                       (img_size // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((img_size, img_size))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', 
                                ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', 
                                ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
            
        elif self.aug_type == 'pixel':
            # Pixel shuffling
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=img_size, ps2=img_size)
        
        return x_prime

    def reset(self):
        """Reset model to initial state"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        
        self.model.load_state_dict(self.model_state, strict=False)
        self.optimizer.load_state_dict(self.optimizer_state)


# ============ Helper Functions ============

@torch.jit.script
def softmax_entropy(x):
    """Entropy of softmax distribution from logits"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# ============ Configuration Functions (for compatibility) ============

def configure_model(model):
    """Configure model for DeYO"""
    model.eval()
    model.requires_grad_(False)
    
    for nm, m in model.named_modules():
        # Skip top layers
        if 'layer4' in nm or 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm:
            continue
        if 'norm.' in nm or nm in ['norm']:
            continue
            
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    
    return model


def collect_params(model):
    """Collect normalization parameters for adaptation"""
    params = []
    names = []
    
    for nm, m in model.named_modules():
        # Skip top layers
        if 'layer4' in nm or 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm:
            continue
        if 'norm.' in nm or nm in ['norm']:
            continue
            
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    
    return params, names
