# Note: You can use our Nuwa to help you discover the key parts in the Dynamic System by following the tutorial below, using the present simple tense.

### (1) Discovering ``` python nvwa_upstream_pretrain.py ```
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
```

### (2) Updating ``` python nvwa_downstream_pred.py ```
We provided an example of simvp, and fundamentally, Nuwa offers a vast array of potential data distributions to enhance out-of-distribution perception capabilities.
```python

class Nvwa_enchane_SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Nvwa_enchane_SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


if __name__ == "__main__":
    model = Nvwa_enchane_SimVP((10, 1, 64, 64))
    inputs = torch.rand(10, 10, 1, 64, 64)
    outputs = model(inputs)
    print(outputs.shape)


```

### (3) Data augmentation 

In the folder attention_map, we give an example of data augmentation, combined with the paper, to find out the top-ordered and bottom-ordered patch regions, and we use top-down and bottom-up Mixup for data augmentation strategy for the bottom-ordered patch regions.

```python
import numpy as np

def mixup_patch(input_frame, min_attention_position, alpha=0.5):
    """
    Performs mixup enhancement on the minimum attention weight patch of the specified image and its right-hand patch.
    :param input_frame: single image data, assuming shape [C, H, W].
    :param min_attention_position: The position of the minimum attention weight patch, in the form (row, col).
    :param alpha: The mixup's mixing ratio.
    :return: The enhanced image.
    """
    patch_size = 16 
    row, col = min_attention_position
    start_x = col * patch_size
    start_y = row * patch_size
    
    if start_x + patch_size < input_frame.shape[2] - patch_size:
        target_patch = input_frame[:, start_y:start_y+patch_size, start_x:start_x+patch_size]
        right_patch = input_frame[:, start_y:start_y+patch_size, start_x+patch_size:start_x+2*patch_size]
        mixed_patch = alpha * target_patch + (1 - alpha) * right_patch
        input_frame[:, start_y:start_y+patch_size, start_x:start_x+patch_size] = mixed_patch
    
    return input_frame


input_frame = input_frames[0] 
min_attention_position = (min_attention_positions[0][0], min_attention_positions[1][0]) 
enhanced_frame = mixup_patch(input_frame, min_attention_position, alpha=0.5)
print(enhanced_frame.reshape(10, 1, 64, 64).shape)
```



#### If you have any questions, please feel free to contact us Email: wuhao2022@mail.ustc.edu.cn






