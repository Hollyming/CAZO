import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
import math
from typing import Union, List

class SwinAdapter(nn.Module):
    '''
    Swin Transformer with Adapter
    在Swin Transformer的每个block后并联adapter（AdaFormer风格）
    '''
    def __init__(self, 
                swin: SwinTransformer,
                adapter_layer: Union[str, int, List[int], None] = None,
                reduction_factor: int = 16,  
                dropout: float = 0.1,
                init_option: str = "lora",
                adapter_scalar: str = "0.1",
                adapter_layernorm_option: str = "in",
                adapter_style: str = "parallel"):
        super().__init__()
        self.swin = swin
        self.reduction_factor = reduction_factor
        self.adapter_style = adapter_style
        
        # Swin Transformer的特征维度
        self.hidden_size = swin.num_features
        
        # 获取所有blocks的总数
        self.total_blocks = sum(len(layer.blocks) for layer in swin.layers)
        
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
        
        # 创建Adapter配置 - 为每个stage的不同维度创建adapter
        self.adapters = nn.ModuleDict()
        block_idx = 0
        for stage_idx, layer in enumerate(self.swin.layers):
            # 直接从block中获取实际维度（考虑downsample的影响）
            if len(layer.blocks) > 0:
                # 通过第一个block的norm1层获取该stage的实际特征维度
                stage_dim = layer.blocks[0].norm1.normalized_shape[0]
            else:
                stage_dim = int(self.swin.embed_dim * 2 ** stage_idx)
            
            for local_block_idx in range(len(layer.blocks)):
                if block_idx in self.adapter_layers:
                    class Config:
                        def __init__(self, d_model, reduction_factor):
                            self.d_model = d_model
                            self.attn_bn = d_model // reduction_factor
                    
                    config = Config(d_model=stage_dim, 
                                   reduction_factor=reduction_factor)
                    
                    self.adapters[f'adapter_{block_idx}'] = Adapter(
                        config=config,
                        dropout=dropout,
                        init_option=init_option,
                        adapter_scalar=adapter_scalar,
                        adapter_layernorm_option=adapter_layernorm_option
                    )
                block_idx += 1
    
    def reset_adapters(self):
        """重置所有adapter参数"""
        for adapter in self.adapters.values():
            if adapter.init_option == "lora":
                with torch.no_grad():
                    nn.init.kaiming_uniform_(adapter.down_proj.weight, a=math.sqrt(5))
                    nn.init.zeros_(adapter.up_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)
                    nn.init.zeros_(adapter.up_proj.bias)

    def forward_features(self, x):
        '''带adapter的前向传播'''
        x = self.swin.patch_embed(x)  # (B, H, W, C)
        
        block_idx = 0
        for stage_idx, layer in enumerate(self.swin.layers):
            # Downsample在每个stage开始前
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                x = layer.downsample(x)
            
            for block in layer.blocks:
                # 使用block的forward
                x = block(x)  # (B, H, W, C)
                
                # Adapter并联
                if block_idx in self.adapter_layers:
                    B, H, W, C = x.shape
                    x_flat = x.view(B, H * W, C)  # (B, H*W, C)
                    
                    if self.adapter_style == "parallel":
                        adapt_x = self.adapters[f'adapter_{block_idx}'](x_flat, add_residual=False)
                        x_flat = x_flat + adapt_x
                    
                    x = x_flat.view(B, H, W, C)  # (B, H, W, C)
                
                block_idx += 1
        
        x = self.swin.norm(x)  # (B, H, W, C)
        # 全局平均池化
        B, H, W, C = x.shape
        x = x.view(B, -1, C).mean(dim=1)  # (B, H*W, C) -> (B, C)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.swin.head(x)
        return x

    def layers_features_with_adapters(self, x):
        """从带adapter的Swin中提取所有block特征（用于CAZO的MSE损失计算）"""
        x = self.swin.patch_embed(x)
        
        features_list = []
        block_idx = 0
        for stage_idx, layer in enumerate(self.swin.layers):
            # Downsample在每个stage开始前
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                x = layer.downsample(x)
            
            for block in layer.blocks:
                x = block(x)
                
                # Adapter
                if block_idx in self.adapter_layers:
                    B, H, W, C = x.shape
                    x_flat = x.view(B, H * W, C)
                    
                    if self.adapter_style == "parallel":
                        adapt_x = self.adapters[f'adapter_{block_idx}'](x_flat, add_residual=False)
                        x_flat = x_flat + adapt_x
                    
                    x = x_flat.view(B, H, W, C)
                
                # 提取特征：全局平均池化
                B, H, W, C = x.shape
                feat = x.view(B, -1, C).mean(dim=1)  # (B, C)
                features_list.append(feat)
                block_idx += 1
        
        return torch.cat(features_list, dim=1)  # (B, sum of all C_i)
        
    def layers_features(self, x):
        """从原始Swin中提取所有block特征（用于obtain_origin_stat）"""
        x = self.swin.patch_embed(x)
        
        features_list = []
        for stage_idx, layer in enumerate(self.swin.layers):
            # Downsample在每个stage开始前
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                x = layer.downsample(x)
            
            for block in layer.blocks:
                x = block(x)  # (B, H, W, C)
                
                # 提取特征
                B, H, W, C = x.shape
                feat = x.view(B, -1, C).mean(dim=1)  # (B, C)
                features_list.append(feat)
        
        return torch.cat(features_list, dim=1)


class Adapter(nn.Module):
    """Adapter模块（AdaFormer风格）"""
    def __init__(self,
                 config=None,
                 d_model=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn
        self.adapter_layernorm_option = adapter_layernorm_option
        self.init_option = init_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        # 确保down的维度正确
        if down.dim() == 1:
            down = down.unsqueeze(0)
        
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def freeze_swin_parameters(model: SwinAdapter):
    """冻结Swin Transformer参数,只训练adapter参数"""
    for param in model.swin.parameters():
        param.requires_grad = False
