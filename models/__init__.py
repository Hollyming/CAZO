"""
Models package for CAZO
Includes adapter implementations for various architectures
"""

from .adaformer import AdaFormerViT, freeze_vit_parameters
from .deit_adapter import DeiTAdapter, freeze_deit_parameters
from .swin_adapter import SwinAdapter, freeze_swin_parameters
from .resnet_adapter import ResNetAdapter, freeze_resnet_parameters

__all__ = [
    'AdaFormerViT',
    'DeiTAdapter', 
    'SwinAdapter',
    'ResNetAdapter',
    'freeze_vit_parameters',
    'freeze_deit_parameters',
    'freeze_swin_parameters',
    'freeze_resnet_parameters'
]
