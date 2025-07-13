from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    """
    Dataset wrapper for training autoencoders.

    This class takes an existing dataset and adapts it
    to be used for training an autoencoder, where both
    the input and the target are the same image.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, image

    def __len__(self):
        return len(self.dataset)
