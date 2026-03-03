"""
EATA: Efficient Test-Time Adaptation through Automatic Test-time Sample Selection
ETA: Entropy minimization with Test-time Adaptation (EATA without Fisher regularization)
Adapted from LCoTTA: https://github.com/mr-eggplant/EATA
Paper: https://arxiv.org/abs/2204.02610

Integrated into CAZO framework for single-round testing (non-lifelong setting)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EATA(nn.Module):
    """
    EATA: Efficient Test-Time Adaptation with sample filtering and Fisher regularization
    When fisher_alpha = 0, it becomes ETA (pure entropy minimization)
    Adapted for CAZO framework
    """
    def __init__(self, model, optimizer, num_classes=1000,
                 margin_e0=0.4, d_margin=0.05, fisher_alpha=2000.0,
                 fishers=None, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.num_classes = num_classes
        
        # EATA specific parameters
        self.e_margin = margin_e0 * math.log(num_classes)  # entropy margin
        self.d_margin = d_margin  # diversity margin for cosine similarity
        self.fisher_alpha = fisher_alpha  # trade-off for Fisher regularization
        
        # Fisher information matrix for EWC (anti-forgetting)
        self.fishers = fishers  # Should be pre-computed if fisher_alpha > 0
        
        # Sample filtering statistics
        self.num_samples_update_1 = 0  # after entropy filtering
        self.num_samples_update_2 = 0  # after entropy + diversity filtering
        self.current_model_probs = None  # moving average of probability vector
        
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
        """Configure model for use with EATA/ETA"""
        self.model.eval()  # eval mode to avoid stochastic depth
        self.model.requires_grad_(False)
        
        # Configure norm layers for adaptation
        for m in self.model.modules():
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
            outputs, loss_value, perform_update = self.forward_and_adapt(x)
            self.num_forwards += x.size(0)
            if perform_update:
                self.num_backwards += x.size(0)
            self.final_loss = loss_value
            
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Forward and adapt model on batch of data with sample filtering
        """
        imgs_test = x
        outputs = self.model(imgs_test)
        
        # Apply ImageNet-R mask if needed
        if self.imagenet_mask is not None:
            outputs_masked = outputs[:, self.imagenet_mask]
        else:
            outputs_masked = outputs
        
        # Calculate entropy
        entropys = softmax_entropy(outputs_masked)
        
        # First filter: remove unreliable samples (high entropy)
        filter_ids_1 = torch.where(entropys < self.e_margin)
        entropys_filtered = entropys[filter_ids_1]
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        
        if len(entropys_filtered) == 0:
            return outputs, None, False
        
        # Second filter: remove redundant samples (high similarity with current model)
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                outputs_masked[filter_ids_1].softmax(1),
                dim=1
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys_final = entropys_filtered[filter_ids_2]
            updated_probs = update_model_probs(
                self.current_model_probs,
                outputs_masked[filter_ids_1][filter_ids_2].softmax(1)
            )
        else:
            entropys_final = entropys_filtered
            updated_probs = update_model_probs(
                self.current_model_probs,
                outputs_masked[filter_ids_1].softmax(1)
            )
        
        self.num_samples_update_2 += entropys_final.size(0)
        
        if len(entropys_final) == 0:
            self.current_model_probs = updated_probs
            return outputs, None, False
        
        # Reweight entropy losses
        coeff = 1 / (torch.exp(entropys_final.clone().detach() - self.e_margin))
        entropys_final = entropys_final.mul(coeff)
        loss = entropys_final.mean(0)
        
        # Add Fisher regularization (EWC) if enabled
        if self.fishers is not None and self.fisher_alpha > 0:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (
                        self.fishers[name][0] * (param - self.fishers[name][1]) ** 2
                    ).sum()
            loss += ewc_loss
        
        # Update current model probabilities
        self.current_model_probs = updated_probs
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return outputs, loss.item(), True

    def reset(self):
        """Reset model to initial state"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        
        self.model.load_state_dict(self.model_state, strict=False)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0


class ETA(EATA):
    """
    ETA: Entropy minimization with Test-time Adaptation
    This is EATA without Fisher regularization (fisher_alpha = 0)
    """
    def __init__(self, model, optimizer, num_classes=1000,
                 margin_e0=0.4, d_margin=0.05, steps=1, episodic=False):
        super().__init__(
            model=model,
            optimizer=optimizer,
            num_classes=num_classes,
            margin_e0=margin_e0,
            d_margin=d_margin,
            fisher_alpha=0.0,  # No Fisher regularization for ETA
            fishers=None,
            steps=steps,
            episodic=episodic
        )


# ============ Helper Functions ============

@torch.jit.script
def softmax_entropy(x):
    """Entropy of softmax distribution from logits"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_model_probs(current_model_probs, new_probs):
    """Update moving average of model probabilities"""
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def compute_fishers(model, fisher_loader, device='cuda'):
    """
    Compute Fisher information matrix for EWC regularization
    This should be called before adaptation using source data
    
    Args:
        model: The neural network model
        fisher_loader: DataLoader containing source domain samples
        device: Device to run computation on
    
    Returns:
        fishers: Dictionary containing Fisher matrix and original parameters
    """
    params = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
    
    optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().to(device)
    
    for iter_, batch in enumerate(fisher_loader, start=1):
        images = batch[0].to(device, non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        
        optimizer.zero_grad()
    
    return fishers


# ============ Configuration Functions (for compatibility) ============

def configure_model(model):
    """Configure model for EATA/ETA"""
    model.eval()
    model.requires_grad_(False)
    
    for m in model.modules():
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
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    
    return params, names
