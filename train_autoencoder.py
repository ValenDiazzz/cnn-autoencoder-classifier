import torch
import argparse
import os
import numpy as np
from torch import nn
from data_utils import data_downloader, adapt_dataset, data_loaders
from train import autoencoder_training
from plots import plot_autoencoder_losses
from datasets import AutoencoderDataset


WEIGHTS_PATH = "Autoencoder_weights"


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Autoencoder on Fashion-MNIST")
    parser.add_argument(
        "--optimizer",
        '-opt',
        type=str,
        choices=["ADAM", "SGD"],
        default="ADAM",
        help="Optimizer to use: ADAM or SGD (default: ADAM)"
    )
    parser.add_argument(
        "--learning_rate",
        '-lr',
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--epochs",
        '-ep',
        type=int,
        default=20,
        help="Number of epochs to train (default: 20)"
    )
    return parser.parse_args()


def set_seeds(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    args = parse_args()
    # ------------------------
    # 1. Set seeds and device
    # ------------------------
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # 2. Download and prepare datasets
    # ------------------------
    print('----------------Downloading Fashio-MNIST dataset----------------')

    try:
        train_dataset, test_dataset = data_downloader()
        train_dataset, test_dataset = adapt_dataset(
            train_dataset,
            test_dataset,
            AutoencoderDataset
        )
        train_loader, valid_loader, test_loader = data_loaders(
            train_dataset,
            test_dataset,
            batch_size=100
        )
        print('Data downloaded successfully')
    except Exception as e:
        print(f'Error in downloadinf the data:\n {str(e)}')

    # ------------------------
    # 3. Setup optimizer and loss
    # ------------------------
    optimizer_class = torch.optim.Adam if args.optimizer == "ADAM" else torch.optim.SGD
    criterion = nn.MSELoss()

    # ------------------------
    # 4. Train autoencoder
    # ------------------------
    print('----------------Training Autoencoder----------------')
    autoencoder_data = autoencoder_training(
        train_loader,
        valid_loader,
        test_loader,
        optimizer_class,
        criterion,
        device,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    autoenc_train_losses = autoencoder_data[0]
    autoenc_valid_losses = autoencoder_data[1]
    autoenc_test_loss = autoencoder_data[2]
    model = autoencoder_data[3]
    print(f"\nFinal Test Loss (MSE): {autoenc_test_loss:.5f}")

    # ------------------------
    # 5. Plot and save losses
    # ------------------------
    plot_autoencoder_losses(
        autoenc_train_losses,
        autoenc_valid_losses,
        autoenc_test_loss,
        filename="autoencoder_loss_curve.png"
    )

    # ------------------------
    # 6. Save Autoencoder weights.
    # ------------------------
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    torch.save(model.encoder.state_dict(), F"{WEIGHTS_PATH}/encoder_weights.pth")
    print("Encoder saved to encoder_weights.pth")


if __name__ == "__main__":
    main()
