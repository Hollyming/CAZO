import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import math
from typing import Union, List

class AdaFormerViT(nn.Module):
    '''
    Vision Transformer with AdaFormer layer at specified position
    支持串联(sequential)和并联(parallel)两种方式
    '''
    def __init__(self, 
                vit: VisionTransformer,
                adapter_layer: Union[str, int, List[int], None] = None,
                reduction_factor: int = 16,  
                dropout: float = 0.1,
                init_option: str = "lora",
                adapter_scalar: str = "0.1",
                adapter_layernorm_option: str = "in",
                adapter_style: str = "parallel"):  # sequential或parallel
        super().__init__()
        self.vit = vit
        self.hidden_size = vit.embed_dim
        self.reduction_factor = reduction_factor
        self.adapter_style = adapter_style
        
        # 处理adapter_layer参数
        if adapter_layer is None:   #None表示不添加adapter
            self.adapter_layers = []
        elif isinstance(adapter_layer, str):    #默认输入格式为str
            if ',' in adapter_layer:
                # 处理形如 "3,7,11" 的输入
                self.adapter_layers = [int(layer.strip()) for layer in adapter_layer.split(',')]
            else:
                # 处理形如 "11" 的输入
                self.adapter_layers = [int(adapter_layer)]
        elif isinstance(adapter_layer, int):
            self.adapter_layers = [adapter_layer]
        else:
            self.adapter_layers = sorted(adapter_layer)  # 确保层序号有序
        
        # 创建AdaFormer配置
        class Config:
            def __init__(self, d_model, reduction_factor):
                self.d_model = d_model
                self.attn_bn = d_model // reduction_factor
        
        config = Config(d_model=self.hidden_size, 
                       reduction_factor=reduction_factor)
        
        # 为每个指定层创建独立的adapter
        self.adapters = nn.ModuleDict({
            f'adapter_{layer}': Adapter(
                config=config,
                dropout=dropout,
                init_option=init_option,
                adapter_scalar=adapter_scalar,
                adapter_layernorm_option=adapter_layernorm_option
            ) for layer in self.adapter_layers
        })
    
    def reset_adapters(self):
        """重置所有adapter参数"""
        for adapter in self.adapters.values():
            if adapter.init_option == "lora":
                with torch.no_grad():
                    nn.init.kaiming_uniform_(adapter.down_proj.weight, a=math.sqrt(5))
                    nn.init.zeros_(adapter.up_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)
                    nn.init.zeros_(adapter.up_proj.bias)

    def _collect_layers_features(self, x):
        """收集每个transformer block的CLS token特征"""
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features

    def _collect_layers_features_with_adapters(self, x):
        """收集带adapter的每个transformer block的CLS token特征"""
        cls_features = []
        for i, block in enumerate(self.vit.blocks):
            # 原始block处理
            h = x
            x = block.norm1(x)
            attn_output = block.attn(x)
            x = h + block.drop_path1(attn_output)
            
            # FFN
            h = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = block.drop_path2(x)
            
            # Adapter 部分
            if i in self.adapter_layers:
                if self.adapter_style == "sequential":
                    x = self.adapters[f'adapter_{i}'](x)
                elif self.adapter_style == "parallel":
                    adapt_x = self.adapters[f'adapter_{i}'](h, add_residual=False)
                    x = x + adapt_x
            
            x = x + h
            
            # 收集特征
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features
    
    def forward_features(self, x):
        '''带adapter的前向传播'''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        
        for i, block in enumerate(self.vit.blocks):
            # 注意力模块
            h = x
            x = block.norm1(x)
            attn_output = block.attn(x)
            x = h + block.drop_path1(attn_output)
            
            # FFN
            h = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = block.drop_path2(x)
            
            # Adapter 部分
            if i in self.adapter_layers:
                if self.adapter_style == "sequential":
                    x = self.adapters[f'adapter_{i}'](x)
                elif self.adapter_style == "parallel":
                    adapt_x = self.adapters[f'adapter_{i}'](h, add_residual=False)
                    x = x + adapt_x
            
            x = x + h
            
        x = self.vit.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x

    def layers_cls_features_with_adapters(self, x):
        """从带adapter的ViT中提取CLS token特征"""
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features_with_adapters(x)
        
    def layers_cls_features(self, x):
        """从原始ViT中提取CLS token特征"""
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)


class Adapter(nn.Module):
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
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def freeze_vit_parameters(model: AdaFormerViT):
    """冻结ViT参数,只训练adapter参数"""
    for param in model.vit.parameters():
        param.requires_grad = False 