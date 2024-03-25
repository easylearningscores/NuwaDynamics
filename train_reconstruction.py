import torch
from MM.causal_physical_system.nvwa_upstream_pretrain import VisionTransformer
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from dataloader import load_data
import numpy as np

dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=10, 
                                                                                    val_batch_size=10, 
                                                                                    data_root='/data/workspace/yancheng/MM/neural_manifold_operator/data/',
                                                                                    num_workers=8)


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

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
vit_model.to(device)

optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

def validate_model(model, dataloader_valid, mse_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_frames, output_frames in dataloader_valid:
            input_frames = input_frames.to(device)
            output_frames = output_frames.to(device)
            preds = model(input_frames)
            loss = mse_loss(preds, input_frames)
            total_loss += loss.item()
    return total_loss / len(dataloader_valid)

def train_model(model, dataloader_train, dataloader_valid, optimizer, mse_loss, epochs=10):
    min_valid_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for input_frames, output_frames in dataloader_train:
            input_frames = input_frames.to(device)
            output_frames = output_frames.to(device)

            optimizer.zero_grad()
            preds = model(input_frames)
            loss = mse_loss(preds, input_frames)
            loss.backward()
            optimizer.step()

        valid_loss = validate_model(model, dataloader_valid, mse_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Valid Loss: {valid_loss}")

        # Save the best model
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), '/data/workspace/yancheng/MM/causal_physical_system/pretrain_model_save/2024big_best_model.pth')
            print("Model saved")

train_model(vit_model, dataloader_train, dataloader_validation, optimizer, mse_loss, epochs=1)

# select attention map
attention_maps = vit_model.blocks[0].attn.attention_maps
np.save('/data/workspace/yancheng/MM/causal_physical_system/attention_map/Attention_map.npy', attention_maps.cpu())
print(attention_maps.shape)