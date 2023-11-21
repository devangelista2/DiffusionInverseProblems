import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, config):
        # Load data from numpy array
        self.data = np.load(os.path.join(config.data.data_path, "train.npy"))

        # Convert the data (B, D, D, C) -> (B, C, D, D)
        self.data = torch.permute(torch.tensor(self.data), (0, 3, 1, 2))

        # Normalize the data
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        # Infos
        self.shape = self.data.shape

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)
