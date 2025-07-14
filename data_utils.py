from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from datasets import AutoencoderDataset
from torch.utils.data import Dataset


def data_downloader() -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """
    Download FashionMNIST datasets.
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root='MNIST_data/',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='MNIST_data/',
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def adapt_dataset(
    train_dataset: Dataset,
    test_dataset: Dataset,
    new_class: type
) -> Tuple[Dataset, Dataset]:
    """
    Wrap datasets with a new class, such as AutoencoderDataset.
    """

    train_dataset = new_class(train_dataset)
    test_dataset = new_class(test_dataset)

    return train_dataset, test_dataset


def data_loaders(
    train_dataset: datasets.VisionDataset,
    test_dataset: datasets.VisionDataset,
    batch_size: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation and test sets.
    """
    train_set, valid_set = random_split(train_dataset, [50000, 10000])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
