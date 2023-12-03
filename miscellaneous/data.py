import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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


def load_data(config):
    # Check wether the dataset is configured
    match config.data.dataset:
        case "MNIST":
            # Define the transform
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            # Load training set
            x_train = datasets.MNIST(
                f"../data/{config.data.data_path}",
                train=True,
                download=True,
                transform=transform,
            )

            # Load test set
            x_test = datasets.MNIST(
                f"../data/{config.data.data_path}",
                train=False,
                download=True,
                transform=transform,
            )

            return x_train, x_test

        case "CIFAR10":
            # Define the transform
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            # Load training set
            x_train = datasets.CIFAR10(
                f"../data/{config.data.data_path}",
                train=True,
                download=True,
                transform=transform,
            )

            # Load test set
            x_test = datasets.CIFAR10(
                f"../data/{config.data.data_path}",
                train=False,
                download=True,
                transform=transform,
            )

            return x_train, x_test

        case "SimpleCelebA":
            x_train = ImageDataset(config)

            return x_train, []

        case "LSUNChurch":
            # Define the transform
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            # Load training set
            x_train = datasets.LSUN(
                f"../data/{config.data.data_path}",
                classes=["church_outdoor_train"],
                transform=transform,
            )

            return x_train, []

        case _:
            return None
