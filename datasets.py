from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    """
    Dataset wrapper for training autoencoders.

    This class takes an existing dataset and adapts it
    to be used for training an autoencoder, where both
    the input and the target are the same image.

    Attributes:
        dataset (Dataset): The original dataset containing images.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, image

    def __len__(self):
        return len(self.dataset)


class ClassificationDataset(Dataset):
    """
    Dataset wrapper for training classifiers.

    This class takes an existing dataset and adapts it
    to return (image, label) pairs for classification.

    Attributes:
        dataset (Dataset): The original dataset containing images and labels.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

    def __len__(self):
        return len(self.dataset)
