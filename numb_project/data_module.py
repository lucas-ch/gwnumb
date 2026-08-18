from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F

from numb_project.constants import BASE


def rotate_item(item: dict[str, Any], degrees: float) -> dict[str, Any]:
    rotated = dict(item)
    if "image" in rotated:
        rotated["image"] = TF.rotate(rotated["image"], degrees)
    return rotated

class MnistDataset(Dataset):
    def __init__(self, mnist_dataset: datasets.MNIST, domain: list[str]) -> None:
        self.mnist_dataset = mnist_dataset
        self.domain = domain

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image = self.mnist_dataset[idx][0]
        digit = self.mnist_dataset[idx][1]
        digit_one_hot = F.one_hot(torch.tensor(digit), num_classes=BASE).float()

        available = {
            "image": image,
            "digit": digit_one_hot,
        }

        return {key: available[key] for key in self.domain}

class MnistDataModule(LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.mnist_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

        self.domains = [
            ['image'],
            ['digit'],
            ['image', 'digit']
        ]

    def __get_dataloaders(self, split: str) -> dict[frozenset[str], DataLoader]:
        mnist_split = self.mnist_train if split == 'train' else self.mnist_test

        return {
            frozenset(domain): DataLoader(
                MnistDataset(mnist_split, domain),
                batch_size=self.batch_size,
                shuffle=True,
            )
            for domain in self.domains
        }

    def train_dataloader(self) -> CombinedLoader:
        loaders = self.__get_dataloaders('train')
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self) -> CombinedLoader:
        loaders = self.__get_dataloaders('test')
        return CombinedLoader(loaders, mode="max_size_cycle")
    
