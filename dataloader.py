import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io as sio


class NS2DDataset1e5(Dataset):
    def __init__(self, data_path, transform=None):
        self.metadata = sio.loadmat(data_path)
        self.data = self.metadata["u"]
        self.data = torch.from_numpy(self.data)
        self.data = self.data.permute(0, 3, 1, 2)
        self.data.unsqueeze_(2)  # 在通道维度上添加一个维度
        self.transform = transform
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = self.data[idx][:10]
        output_frames = self.data[idx][10:]

        input_frames = (input_frames - self.mean) / self.std
        output_frames = (output_frames - self.mean) / self.std

        return input_frames, output_frames

def load_data(batch_size, val_batch_size, data_root, num_workers):
    train_dataset = NS2DDataset1e5(data_path=data_root + 'train_data.mat', transform=None)
    test_dataset = NS2DDataset1e5(data_path=data_root + 'test_data.mat', transform=None)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = train_dataset.mean, train_dataset.std
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

if __name__ == '__main__':
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=10, 
                                                                                    val_batch_size=10, 
                                                                                    data_root='/data/workspace/yancheng/MM/neural_manifold_operator/data/',
                                                                                    num_workers=8)
    for input_frames, output_frames in iter(dataloader_train):
        print(input_frames.shape, output_frames.shape)
        break
