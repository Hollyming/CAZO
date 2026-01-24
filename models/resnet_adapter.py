import torch
import torch.nn as nn
from models.resnet import ResNet, Bottleneck, BasicBlock
import math
from typing import Union, List

class ResNetAdapter(nn.Module):
    '''
    ResNet-50 with Residual Adapter
    在ResNet的每个Bottleneck后并联adapter
    '''
    def __init__(self, 
                resnet: ResNet,
                adapter_layer: Union[str, int, List[int], None] = None,
                reduction_factor: int = 16,  
                dropout: float = 0.1,
                init_option: str = "lora",
                adapter_scalar: str = "0.1",
                adapter_layernorm_option: str = "none"):  # ResNet通常不用LayerNorm
        super().__init__()
        self.resnet = resnet
        self.reduction_factor = reduction_factor
        
        # 获取所有blocks（ResNet50有16个bottleneck blocks）
        self.all_blocks = []
        self.block_dims = []
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(resnet, layer_name)
            for block in layer:
                self.all_blocks.append(block)
                # 获取每个block的输出维度
                if isinstance(block, Bottleneck):
                    self.block_dims.append(block.conv3.out_channels)
                elif isinstance(block, BasicBlock):
                    self.block_dims.append(block.conv2.out_channels)
        
        self.total_blocks = len(self.all_blocks)
        
        # 处理adapter_layer参数
        if adapter_layer is None:
            self.adapter_layers = []
        elif isinstance(adapter_layer, str):
            if ',' in adapter_layer:
                self.adapter_layers = [int(layer.strip()) for layer in adapter_layer.split(',')]
            else:
                self.adapter_layers = [int(adapter_layer)]
        elif isinstance(adapter_layer, int):
            self.adapter_layers = [adapter_layer]
        else:
            self.adapter_layers = sorted(adapter_layer)
        
        # 为每个指定block创建adapter
        self.adapters = nn.ModuleDict()
        for block_idx in self.adapter_layers:
            if block_idx < self.total_blocks:
                block_dim = self.block_dims[block_idx]
                
                class Config:
                    def __init__(self, d_model, reduction_factor):
                        self.d_model = d_model
                        self.attn_bn = d_model // reduction_factor
                
                config = Config(d_model=block_dim, 
                               reduction_factor=reduction_factor)
                
                self.adapters[f'adapter_{block_idx}'] = ResidualAdapter(
                    config=config,
                    dropout=dropout,
                    init_option=init_option,
                    adapter_scalar=adapter_scalar,
                    adapter_layernorm_option=adapter_layernorm_option
                )
    
    def reset_adapters(self):
        """重置所有adapter参数"""
        for adapter in self.adapters.values():
            if adapter.init_option == "lora":
                with torch.no_grad():
                    nn.init.kaiming_uniform_(adapter.down_proj.weight, a=math.sqrt(5))
                    nn.init.zeros_(adapter.up_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)
                    nn.init.zeros_(adapter.up_proj.bias)

    def _collect_block_features(self, x, with_adapter=False):
        """收集每个block的特征（经过全局平均池化）"""
        features = []
        block_idx = 0
        
        # 初始卷积层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # 遍历所有layer
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.resnet, layer_name)
            for block in layer:
                identity = x
                
                # 执行block的前向传播
                if isinstance(block, Bottleneck):
                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)
                    
                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)
                    
                    out = block.conv3(out)
                    out = block.bn3(out)
                    
                    if block.downsample is not None:
                        identity = block.downsample(x)
                    
                    out += identity
                    
                    # Adapter部分
                    if with_adapter and block_idx in self.adapter_layers:
                        # 在ReLU之前加adapter
                        adapt_out = self.adapters[f'adapter_{block_idx}'](out, add_residual=False)
                        out = out + adapt_out
                    
                    out = block.relu(out)
                    
                elif isinstance(block, BasicBlock):
                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)
                    
                    out = block.conv2(out)
                    out = block.bn2(out)
                    
                    if block.downsample is not None:
                        identity = block.downsample(x)
                    
                    out += identity
                    
                    # Adapter部分
                    if with_adapter and block_idx in self.adapter_layers:
                        adapt_out = self.adapters[f'adapter_{block_idx}'](out, add_residual=False)
                        out = out + adapt_out
                    
                    out = block.relu(out)
                
                x = out
                
                # 对特征图进行全局平均池化
                feat = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                feat = feat.view(feat.size(0), -1)
                features.append(feat)
                
                block_idx += 1
        
        features = torch.cat(features, dim=1)
        return features
    
    def forward_features(self, x):
        '''带adapter的前向传播'''
        block_idx = 0
        
        # 初始卷积层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # 遍历所有layer
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.resnet, layer_name)
            for block in layer:
                identity = x
                
                if isinstance(block, Bottleneck):
                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)
                    
                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)
                    
                    out = block.conv3(out)
                    out = block.bn3(out)
                    
                    if block.downsample is not None:
                        identity = block.downsample(x)
                    
                    out += identity
                    
                    # Adapter
                    if block_idx in self.adapter_layers:
                        adapt_out = self.adapters[f'adapter_{block_idx}'](out, add_residual=False)
                        out = out + adapt_out
                    
                    out = block.relu(out)
                    
                elif isinstance(block, BasicBlock):
                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)
                    
                    out = block.conv2(out)
                    out = block.bn2(out)
                    
                    if block.downsample is not None:
                        identity = block.downsample(x)
                    
                    out += identity
                    
                    # Adapter
                    if block_idx in self.adapter_layers:
                        adapt_out = self.adapters[f'adapter_{block_idx}'](out, add_residual=False)
                        out = out + adapt_out
                    
                    out = block.relu(out)
                
                x = out
                block_idx += 1
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def layers_features_with_adapters(self, x):
        """从带adapter的ResNet中提取特征"""
        return self._collect_block_features(x, with_adapter=True)
        
    def layers_features(self, x):
        """从原始ResNet中提取特征"""
        return self._collect_block_features(x, with_adapter=False)


class ResidualAdapter(nn.Module):
    """
    Residual Adapter for ResNet
    与Transformer的Adapter类似，但针对CNN特征图设计
    """
    def __init__(self,
                 config=None,
                 d_model=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn
        self.adapter_layernorm_option = adapter_layernorm_option
        self.init_option = init_option

        # ResNet通常使用BatchNorm而不是LayerNorm
        self.adapter_norm_before = None
        if adapter_layernorm_option == "bn":
            self.adapter_norm_before = nn.BatchNorm2d(self.n_embd)
        elif adapter_layernorm_option == "ln":
            self.adapter_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # 使用1x1卷积实现adapter
        self.down_proj = nn.Conv2d(self.n_embd, self.down_size, kernel_size=1)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Conv2d(self.down_size, self.n_embd, kernel_size=1)
        self.dropout = dropout
        
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                if self.down_proj.bias is not None:
                    nn.init.zeros_(self.down_proj.bias)
                if self.up_proj.bias is not None:
                    nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
            add_residual: 是否添加残差连接
            residual: 残差输入，如果为None则使用x
        """
        residual = x if residual is None else residual
        
        if self.adapter_layernorm_option == 'bn':
            x = self.adapter_norm_before(x)
        elif self.adapter_layernorm_option == 'ln':
            # LayerNorm需要转换维度
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
            x = self.adapter_norm_before(x)
            x = x.transpose(1, 2).view(B, C, H, W)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def freeze_resnet_parameters(model: ResNetAdapter):
    """冻结ResNet参数,只训练adapter参数"""
    for param in model.resnet.parameters():
        param.requires_grad = False
