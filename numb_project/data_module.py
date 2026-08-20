from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from shimmer import RawDomainGroupsT
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

def rotate_image_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    theta = angles_deg * torch.pi / 180
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(theta)
    rotation_matrices = torch.stack(
        [torch.stack([cos, sin, zeros], dim=-1), torch.stack([-sin, cos, zeros], dim=-1)], dim=-2
    )
    grid = F.affine_grid(rotation_matrices, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)

def rotated_raw_image_groups(raw_data: RawDomainGroupsT) -> RawDomainGroupsT:
    pair_key = frozenset({"digit", "image"})
    image = raw_data[pair_key]["image"]
    digit_raw = raw_data[pair_key]["digit"]

    angle_deg = torch.rand(image.shape[0], device=image.device) * 360.0
    rotated_image = rotate_image_batch(image, angle_deg)

    single_key = frozenset({"image"})
    return {
        single_key: {"image": rotated_image},
        pair_key: {"image": rotated_image, "digit": digit_raw},
    }

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
    
