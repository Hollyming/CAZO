import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import math

class AdapterViT(nn.Module):
    '''
    Vision Transformer with Adapter layer at specified position
    '''
    def __init__(self, 
                vit: VisionTransformer,
                adapter_layer: int = 11,  # 在第几层添加adapter (0-11)
                reduction_factor: int = 16,  # adapter bottleneck的降维比例
                adapter_style: str = "Pfeiffer"):# adapter类型
        super().__init__()
        self.vit = vit
        self.hidden_size = vit.embed_dim
        self.reduction_factor = reduction_factor
        self.adapter_style = adapter_style
        self.adapter_layer = adapter_layer
        
        # 只在指定层创建adapter
        self.adapter = AdapterBlock(
            hidden_size=self.hidden_size,
            reduction_factor=reduction_factor
        )
        
        # 添加新的LayerNorm层
        self.adapter_norm = nn.LayerNorm(self.hidden_size)
    
    def reset_adapters(self):
        """重置adapter参数"""
        self.adapter.reset_parameters()

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
            # 修改 attention 的处理方式
            attn_output = block.attn(x)
            x = h + block.drop_path1(attn_output)
            
            # FFN 部分
            h = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = block.drop_path2(x)
            
            # 只在指定层添加Adapter，并在adapter后添加norm
            if i == self.adapter_layer:
                x = self.adapter(x)
                # x = self.adapter_norm(x)
            
            x = x + h

            # # 简单方式outlayer
            # x = self.vit.blocks[i](x)
            # if i == self.adapter_layer:
            #     x = self.adapter(x)
            #     # x = self.adapter_norm(x)
            
            # 收集特征
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))#提取197的第一个cls特征，变成(64,768)
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)#12个cls特征，变成(64,12*768)=(64,9216)
        return cls_features
    
    def forward_features(self, x):
        '''带adapter的前向传播'''
        # 1. Patch Embedding
        x = self.vit.patch_embed(x)
        
        # 2. Position Embedding
        x = self.vit._pos_embed(x)
        
        # 3. 前向传播每个block
        x = self.vit.norm_pre(x)
        # 原本此处应为 x = self.vit.blocks(x)
        for i, block in enumerate(self.vit.blocks):
            # 原始block处理
            h = x
            x = block.norm1(x)
            # 修改 attention 的处理方式
            attn_output = block.attn(x)
            x = h + block.drop_path1(attn_output)
            
            # FFN
            h = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = block.drop_path2(x)
            # 只在指定层添加Adapter
            #参考Parameter-Efficient Transfer Learning for NLP论文建议的adapter位置
            if i == self.adapter_layer:
                x = self.adapter(x)
                # x = self.adapter_norm(x)
            
            x = x + h
            
            # # 简单方式outlayer
            # x = self.vit.blocks[i](x)
            # if i == self.adapter_layer:
            #     x = self.adapter(x)
            # #     x = self.adapter_norm(x)
            
        x = self.vit.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x

    def layers_cls_features_with_adapters(self, x):
        """从带adapter的ViT中提取CLS token特征"""
        x = self.vit.patch_embed(x)# 1. Patch Embedding
        x = self.vit._pos_embed(x)# 2. Position Embedding
        x = self.vit.norm_pre(x)# 3. 收集每个block的特征
        return self._collect_layers_features_with_adapters(x)
        
    def layers_cls_features(self, x):
        """从原始ViT中提取CLS token特征"""
        x = self.vit.patch_embed(x)# 1. Patch Embedding
        x = self.vit._pos_embed(x)# 2. Position Embedding
        x = self.vit.norm_pre(x)# 3. 收集每个block的特征
        return self._collect_layers_features(x)


class AdapterBlock(nn.Module):
    """单个Adapter模块"""
    def __init__(self, hidden_size: int, reduction_factor: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size // reduction_factor)
        self.up_proj = nn.Linear(hidden_size // reduction_factor, hidden_size)
        self.act_fn = nn.GELU()
        
        # 初始化为0
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化adapter参数为0"""
        '''
        初始化0是为了避免在训练开始时adapter层不会影响原始模型的输出
        随着训练进行,adapter层会逐渐学习到有用的适应性变换
        '''
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # Adapter forward
        identity = x
        x = self.down_proj(x)
        x = self.act_fn(x)
        x = self.up_proj(x)      
        x = x + identity  # 残差连接
        return x


def freeze_vit_parameters(model: AdapterViT):
    """冻结ViT参数,只训练adapter参数"""
    for param in model.vit.parameters():
        param.requires_grad = False 