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
