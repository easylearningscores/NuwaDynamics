import torch
from nvwa_upstream_pretrain import VisionTransformer
import torch.nn as nn
import numpy as np
from API.dataloader_sevir import load_data

# Load data
dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(
    batch_size=1, val_batch_size=1, data_root='./data', num_workers=8)

# Model parameters
img_size = 128
patch_size = 16
in_c = 1
out_c = 1
embed_dim = 128
depth = 1
num_heads = 1
mlp_ratio = 4.0
batch_size = 1
time_step = 10
drop_ratio = 0.5
attn_drop_ratio=0.
drop_path_ratio=0.3
# Initialize model
vit_model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_c=in_c * time_step,
    out_chans=out_c * time_step,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop_ratio=drop_ratio,
    attn_drop_ratio=attn_drop_ratio,
    drop_path_ratio=drop_path_ratio
)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
vit_model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

# Validation function
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

# Training function
def train_model(model, dataloader_train, dataloader_valid, optimizer, mse_loss, epochs=10):
    min_valid_loss = float('inf')
    for epoch in range(epochs):
        print(epoch)
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
            torch.save(model.state_dict(), './data/2024big_best_model.pth')
            print("Model saved")

# First train for 100 epochs
train_model(vit_model, dataloader_train, dataloader_validation, optimizer, mse_loss, epochs=30)

# Load the best model
vit_model.load_state_dict(torch.load('./data/2024big_best_model.pth'))
vit_model.to(device)

# Function to generate attention maps
def generate_attention_maps(model, dataloader):
    model.eval()
    all_attention_maps = []
    with torch.no_grad():
        for input_frames, output_frames in dataloader:
            input_frames = input_frames.to(device)
            preds = model(input_frames)
            current_attention_maps = model.blocks[0].attn.attention_maps.detach().cpu().numpy()
            all_attention_maps.append(current_attention_maps)
    return all_attention_maps

# Train for 1 more epoch and generate attention maps
optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)  # Reinitialize optimizer
train_model(vit_model, dataloader_train, dataloader_validation, optimizer, mse_loss, epochs=1)
attention_maps = generate_attention_maps(vit_model, dataloader_train)

# Save attention maps
np.save('./data/all_attention_maps.npy', np.array(attention_maps, dtype=object))
