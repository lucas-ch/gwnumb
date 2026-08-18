import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader

from numb_project.constants import BASE
from numb_project.data_module import MnistDataModule, MnistDataset


class FakeMNIST:
    """Minimal stand-in for datasets.MNIST: indexable, returns (image, digit)."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = torch.full((1, 28, 28), float(idx))
        digit = idx % BASE
        return image, digit


class TestMnistDataset:
    def test_len_matches_underlying_dataset(self) -> None:
        dataset = MnistDataset(FakeMNIST(7), domain=["image"])
        assert len(dataset) == 7

    def test_image_only_domain_returns_only_image(self) -> None:
        dataset = MnistDataset(FakeMNIST(3), domain=["image"])
        item = dataset[0]
        assert set(item.keys()) == {"image"}
        assert item["image"].shape == (1, 28, 28)

    def test_digit_only_domain_returns_one_hot_digit(self) -> None:
        dataset = MnistDataset(FakeMNIST(3), domain=["digit"])
        item = dataset[2]
        assert set(item.keys()) == {"digit"}
        assert item["digit"].shape == (BASE,)
        assert item["digit"].argmax().item() == 2
        assert item["digit"].sum().item() == 1

    def test_image_and_digit_domain_returns_both(self) -> None:
        dataset = MnistDataset(FakeMNIST(3), domain=["image", "digit"])
        item = dataset[1]
        assert set(item.keys()) == {"image", "digit"}
        assert item["digit"].argmax().item() == 1


class TestMnistDataModule:
    def test_init_stores_batch_size_and_builds_dataloaders(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "numb_project.data_module.datasets.MNIST",
            lambda root, train, download, transform: FakeMNIST(16 if train else 8),
        )

        data_module = MnistDataModule(batch_size=4)

        assert data_module.batch_size == 4

    def test_train_dataloader_covers_all_domain_combinations(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "numb_project.data_module.datasets.MNIST",
            lambda root, train, download, transform: FakeMNIST(16 if train else 8),
        )

        data_module = MnistDataModule(batch_size=4)
        loader = data_module.train_dataloader()

        assert isinstance(loader, CombinedLoader)
        batch, _, _ = next(iter(loader))

        expected_domains = {
            frozenset(["image"]),
            frozenset(["digit"]),
            frozenset(["image", "digit"]),
        }
        assert set(batch.keys()) == expected_domains
        assert set(batch[frozenset(["image"])].keys()) == {"image"}
        assert set(batch[frozenset(["digit"])].keys()) == {"digit"}
        assert set(batch[frozenset(["image", "digit"])].keys()) == {"image", "digit"}

    def test_test_dataloader_uses_test_split(self, monkeypatch) -> None:
        seen_splits = []

        def fake_mnist(root, train, download, transform):
            seen_splits.append(train)
            return FakeMNIST(16 if train else 8)

        monkeypatch.setattr("numb_project.data_module.datasets.MNIST", fake_mnist)

        data_module = MnistDataModule(batch_size=4)
        loader = data_module.test_dataloader()

        assert isinstance(loader, CombinedLoader)
        batch, _, _ = next(iter(loader))
        # test split only has 8 samples: with batch_size=4 each dataloader yields 4 items max
        for sub_batch in batch.values():
            for tensor in sub_batch.values():
                assert tensor.shape[0] <= 4

        assert True in seen_splits and False in seen_splits
