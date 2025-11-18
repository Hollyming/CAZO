"""
RoTTA: Robust Test-Time Adaptation via Category-balanced Memory and Dynamic Entropy Filtering
Adapted from LCoTTA: https://github.com/BIT-DA/RoTTA
Paper: https://arxiv.org/pdf/2303.13899.pdf

Integrated into CAZO framework for single-round testing (non-lifelong setting)
"""

import math
import torch
import torch.nn as nn
from copy import deepcopy
import PIL
import torchvision.transforms as transforms
from . import my_transforms as my_transforms


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std), 
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema(ema_model, model, alpha_teacher):
    """
    Update teacher model with exponential moving average
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class RoTTA(nn.Module):
    """
    RoTTA: Robust Test-Time Adaptation 
    Adapted for CAZO framework - single round testing without lifelong continual adaptation
    """
    def __init__(self, model, optimizer, nu=0.001, memory_size=64, update_frequency=64, 
                 lambda_t=1.0, lambda_u=1.0, alpha=0.05, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        
        # RoTTA specific parameters
        self.memory_size = memory_size
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.nu = 1 - nu  # EMA momentum
        self.alpha = alpha  # for robust BN
        self.update_frequency = update_frequency
        self.current_instance = 0
        
        # Initialize memory bank
        self.num_classes = 1000  # ImageNet classes
        self.mem = CSTU(capacity=self.memory_size, num_class=self.num_classes, 
                        lambda_t=self.lambda_t, lambda_u=self.lambda_u)
        
        # Setup EMA model
        self.model_ema = self._copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        
        # Save initial state for reset
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.model_ema_state = deepcopy(self.model_ema.state_dict())
        
        # Create test-time transformations
        self.transform = get_tta_transforms()
        
        # Replace BN layers with RobustBN
        self._configure_model()
        
        self.imagenet_mask = None  # For ImageNet-R compatibility
        self.num_forwards = 0
        self.num_backwards = 0

    def _copy_model(self, model):
        """Copy model"""
        model_copy = deepcopy(model)
        return model_copy

    def _configure_model(self):
        """Replace BatchNorm layers with RobustBN"""
        self.model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
            elif isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
                sub_module.requires_grad_(True)

        for name in normlayer_names:
            bn_layer = self._get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                continue

            momentum_bn = NewBN(bn_layer, self.alpha)
            momentum_bn.requires_grad_(True)
            self._set_named_submodule(self.model, name, momentum_bn)

    def _get_named_submodule(self, model, sub_name: str):
        """Get submodule by name"""
        names = sub_name.split(".")
        module = model
        for name in names:
            module = getattr(module, name)
        return module

    def _set_named_submodule(self, model, sub_name, value):
        """Set submodule by name"""
        names = sub_name.split(".")
        module = model
        for i in range(len(names)):
            if i != len(names) - 1:
                module = getattr(module, names[i])
            else:
                setattr(module, names[i], value)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
            self.num_forwards += x.size(0)
            
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Forward and adapt model on batch of data
        """
        imgs_test = x
        
        with torch.no_grad():
            self.model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(imgs_test)
            
            # Apply ImageNet-R mask if needed
            if self.imagenet_mask is not None:
                ema_out = ema_out[:, self.imagenet_mask]
            
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # Add samples to memory bank
        for i, data in enumerate(imgs_test):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            # Update model when reaching update frequency
            if self.current_instance % self.update_frequency == 0:
                loss = self._loss_calculation()
                if loss is not None and not torch.isnan(loss):
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.num_backwards += self.update_frequency

                    # Update EMA model
                    self.model_ema = update_ema(self.model_ema, self.model, self.nu)

        return ema_out

    def _loss_calculation(self):
        """Calculate supervised loss from memory bank"""
        self.model.train()
        self.model_ema.train()
        
        # Get memory data
        sup_data, ages = self.mem.get_memory()
        
        if len(sup_data) == 0:
            return None
            
        sup_data = torch.stack(sup_data)
        strong_sup_aug = self.transform(sup_data)
        
        ema_sup_out = self.model_ema(sup_data)
        stu_sup_out = self.model(strong_sup_aug)
        
        instance_weight = timeliness_reweighting(ages, device=sup_data.device)
        loss_sup = (softmax_cross_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()
        
        return loss_sup

    def reset(self):
        """Reset model to initial state"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        
        self.model.load_state_dict(self.model_state, strict=False)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.model_ema.load_state_dict(self.model_ema_state, strict=False)
        
        # Reset memory and counters
        self.current_instance = 0
        self.mem = CSTU(capacity=self.memory_size, num_class=self.num_classes,
                        lambda_t=self.lambda_t, lambda_u=self.lambda_u)


# ============ Helper Functions ============

@torch.jit.script
def softmax_cross_entropy(x, x_ema):
    """Cross entropy between student and teacher predictions"""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def timeliness_reweighting(ages, device):
    """Compute timeliness weights based on sample age"""
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().to(device)
    return torch.exp(-ages) / (1 + torch.exp(-ages))


# ============ Robust Batch Normalization ============

class MomentumBN(nn.Module):
    """Base class for momentum-based batch normalization"""
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        
        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    """Robust Batch Normalization for 1D"""
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)
        return x * weight + bias


class RobustBN2d(MomentumBN):
    """Robust Batch Normalization for 2D"""
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * weight + bias


# ============ Memory Management ============

class MemoryItem:
    """Single item in memory bank"""
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class CSTU:
    """
    Category-balanced Sampling with Timeliness and Uncertainty (CSTU)
    Memory bank for RoTTA
    """
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        """Get total number of samples in memory"""
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        """Get per-class sample count"""
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)
        return per_class_occupied

    def add_instance(self, instance):
        """Add new instance to memory"""
        assert len(instance) == 3
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        """Remove instance if needed to make space"""
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        """Remove worst sample from specified classes"""
        max_class = None
        max_index = None
        max_score = None
        
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        """Get classes with most samples"""
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)
        return classes

    def heuristic_score(self, age, uncertainty):
        """Compute heuristic score for sample removal"""
        return (self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + 
                self.lambda_u * uncertainty / math.log(self.num_class))

    def add_age(self):
        """Increment age for all samples in memory"""
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        """Get all samples and their ages from memory"""
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        # Normalize ages
        tmp_age = [x / self.capacity for x in tmp_age]
        return tmp_data, tmp_age


# ============ Configuration Functions (for compatibility) ============

def configure_model(model):
    """Configure model for RoTTA - this is handled in __init__"""
    model.eval()
    return model


def collect_params(model):
    """Collect normalization parameters for adaptation"""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, 
                          RobustBN1d, RobustBN2d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
