import timm
import torch

print("=== Checking MLP dimensions for each layer ===\n")

# DeiT-Base
print("DeiT-Base:")
deit = timm.create_model('deit_base_patch16_224', pretrained=False)
for i, block in enumerate(deit.blocks):
    mlp_dim = block.mlp.fc1.out_features  # MLP hidden dimension
    embed_dim = block.mlp.fc2.out_features  # Embedding dimension
    print(f"  Layer {i}: embed_dim={embed_dim}, mlp_hidden_dim={mlp_dim}")

# Swin-Tiny
print("\nSwin-Tiny:")
swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
block_idx = 0
for stage_idx, layer in enumerate(swin.layers):
    print(f"  Stage {stage_idx}:")
    for local_idx, block in enumerate(layer.blocks):
        # Swin block structure: norm1 contains the dimension
        dim = block.norm1.normalized_shape[0]
        mlp_dim = block.mlp.fc1.out_features
        print(f"    Layer {block_idx} (Stage {stage_idx}, Block {local_idx}): dim={dim}, mlp_hidden_dim={mlp_dim}")
        block_idx += 1

# Calculate reduction factors for bottleneck=2
print("\n=== Reduction factors for bottleneck=2 ===\n")

print("DeiT-Base:")
for i, block in enumerate(deit.blocks):
    embed_dim = block.mlp.fc2.out_features
    reduction_factor = embed_dim // 2
    print(f"  Layer {i}: embed_dim={embed_dim} → reduction_factor={reduction_factor} (bottleneck={embed_dim // reduction_factor})")

print("\nSwin-Tiny:")
block_idx = 0
reduction_factors = []
for stage_idx, layer in enumerate(swin.layers):
    for local_idx, block in enumerate(layer.blocks):
        dim = block.norm1.normalized_shape[0]
        reduction_factor = dim // 2
        reduction_factors.append(reduction_factor)
        print(f"  Layer {block_idx}: dim={dim} → reduction_factor={reduction_factor} (bottleneck={dim // reduction_factor})")
        block_idx += 1

print(f"\nSwin-Tiny reduction_factor array: {reduction_factors}")
print(f"Unique values: {sorted(set(reduction_factors))}")
