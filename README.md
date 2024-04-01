# Note: You can use our Nuwa to help you discover the key parts in the Dynamic System by following the tutorial below, using the present simple tense.

### (1) Discovery ``` python nvwa_upstream_pretrain.py ```
The following hyperparameters are modified to select the appropriate data dimensions for the self-supervised reconstruction process.
```python
img_size = 64
patch_size = 16
in_c = 1 
out_c = 1
embed_dim = 256  
depth = 1
num_heads = 1
mlp_ratio = 4.0  
betch_size = 10
time_step = 10

vit_model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_c=in_c * time_step,
    out_chans=out_c * time_step,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio
)

input = torch.randn(betch_size, time_step, in_c, img_size, img_size)
print("input shape:", input.shape)
output = vit_model(input)

print("output shape:", output.shape)

