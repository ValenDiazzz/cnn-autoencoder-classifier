from typing import Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def data_downloader() -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """
    Download FashionMNIST datasets.

    Returns:
        Tuple[datasets.VisionDataset, datasets.VisionDataset]:
            The training and test datasets.
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
    dataset_wrapper: type
) -> Tuple[Dataset, Dataset]:
    """
    Wrap datasets with a new class, such as AutoencoderDataset.
    """

    wrapped_train_dataset = dataset_wrapper(train_dataset)
    wrapped_test_dataset = dataset_wrapper(test_dataset)
    return wrapped_train_dataset, wrapped_test_dataset


def data_loaders(
    train_dataset: datasets.VisionDataset,
    test_dataset: datasets.VisionDataset,
    batch_size: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation and test sets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The test dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 100.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            DataLoaders for train, validation, and test datasets.
    """
    train_set, valid_set = random_split(train_dataset, [50000, 10000])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
