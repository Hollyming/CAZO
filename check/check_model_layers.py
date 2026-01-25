import timm

# Check DeiT-Base
print("=== DeiT-Base ===")
deit = timm.create_model('deit_base_patch16_224', pretrained=False)
print(f"Number of transformer blocks: {len(deit.blocks)}")
print(f"Valid adapter_layer range: 0 to {len(deit.blocks) - 1}")

# Check Swin-Tiny
print("\n=== Swin-Tiny ===")
swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
total_blocks = sum(len(layer.blocks) for layer in swin.layers)
print(f"Number of transformer blocks: {total_blocks}")
print(f"Stage structure: {[len(layer.blocks) for layer in swin.layers]}")
print(f"Valid adapter_layer range: 0 to {total_blocks - 1}")

# Check ViT-Base for reference
print("\n=== ViT-Base (for reference) ===")
vit = timm.create_model('vit_base_patch16_224', pretrained=False)
print(f"Number of transformer blocks: {len(vit.blocks)}")
print(f"Valid adapter_layer range: 0 to {len(vit.blocks) - 1}")
