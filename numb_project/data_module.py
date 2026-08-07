from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

class MnistDataset(Dataset):
    def __init__(self, mnist_dataset, domain):
        self.mnist_dataset = mnist_dataset
        self.domain = domain

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image = self.mnist_dataset[idx][0]
        digit = self.mnist_dataset[idx][1]
        digit_one_hot = F.one_hot(torch.tensor(digit), num_classes=10).float()

        if self.domain == ['image']:
            return {"image": image}
        elif self.domain == ['digit']:
            return {"digit": digit_one_hot}
        else:
            return {
                "image": image,
                "digit": digit_one_hot
            }

class MnistDataModule(LightningDataModule):
    def __init__(self, batch_size):
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

    def __get_dataloaders(self, split):
        mnist_split = self.mnist_train if split == 'train' else self.mnist_test

        return {
            frozenset(domain): DataLoader(
                MnistDataset(mnist_split, domain),
                batch_size=self.batch_size,
                shuffle=True,
            )
            for domain in self.domains
        }

    def train_dataloader(self):
        loaders = self.__get_dataloaders('train')
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        loaders = self.__get_dataloaders('test')
        return CombinedLoader(loaders, mode="max_size_cycle")
    
