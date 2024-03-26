import glob
import os

import matplotlib.pyplot as plt
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
    
class MayoDataset(Dataset):
    def __init__(self, data_path, transform=None) -> None:
        super().__init__()

        # Get transforms
        self.transform = transform

        # Set the data_path
        self.data_path = data_path
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        # Get the filename list
        self.f_list = self.get_f_list()

        # Compute the shape
        N = self.__len__()
        c, m, n = self.__getitem__(0).shape
        self.shape = (N, c, m, n)

    def __getitem__(self, index):
        x = plt.imread(self.f_list[index])[:, :, 0:1]
        x = torch.permute(torch.tensor(x), (2, 0, 1))

        return self.normalize(self.transform(x))

    def __len__(self):
        return len(self.f_list)

    def get_f_list(self):
        folders = glob.glob(os.path.join(self.train_path, "*"))

        f_list = []
        for folder in folders:
            f_list = f_list + glob.glob(os.path.join(folder, "*"))
        return tuple(f_list)

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())


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

            return x_train, x_train
        
        case "Mayo256":
            # Define the transform
            transform = transforms.Resize((config.data.image_size, config.data.image_size))

            # Get dataset
            x_train = MayoDataset(config.data.data_path, transform=transform)

            return x_train, []

        case "LSUNChurch":
            # Define the transform
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(
                        (config.data.image_size, config.data.image_size)
                    ),
                ]
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
