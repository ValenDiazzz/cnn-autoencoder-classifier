import os
import argparse
import itertools
from typing import List

import numpy as np
import torch
from torch import nn

from data_utils import data_downloader, adapt_dataset, data_loaders
from train import autoencoder_training
from plots import plot_autoencoder_losses
from datasets import AutoencoderDataset
from models import ConvAutoencoder

WEIGHTS_PATH = "Weights"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the autoencoder.
    """
    parser = argparse.ArgumentParser(
        description="Train Autoencoder on Fashion-MNIST"
    )
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
    parser.add_argument(
        "--dropout",
        '-dp',
        type=float,
        default=0.2,
        help="Dropout for training (Default: 0.2)"
    )
    parser.add_argument(
        "--latent_dim",
        '-ld',
        type=int,
        default=64,
        help="Latent dimension"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning grid search"
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


def run_hyperparameter_tuning(
    train_loader,
    valid_loader,
    test_loader,
    optimizer_class,
    criterion,
    device: torch.device,
    learning_rates: List[float],
    dropouts: List[float],
    latent_dims: List[int],
    epochs: int
) -> None:
    """
    Perform grid search over hyperparameters and save the best model.

    Saves the best encoder weights to Weights/encoder_weights.pth
    """
    best_valid_loss = float('inf')
    best_model = None
    best_config = None

    for lr, p_dropout, latent_dim in itertools.product(learning_rates, dropouts, latent_dims):
        print(f"\nTrying: lr={lr}, dropout={p_dropout}, latent_dim={latent_dim}")

        model = ConvAutoencoder(n=latent_dim, p_dropout=p_dropout).to(device)

        train_losses, valid_losses, test_loss, final_model = autoencoder_training(
            train_loader,
            valid_loader,
            test_loader,
            optimizer_class,
            criterion=criterion,
            device=device,
            model=model,
            learning_rate=lr,
            epochs=epochs
        )

        min_valid_loss = min(valid_losses)
        print(f"Finished: min_valid_loss={min_valid_loss:.5f}, test_loss={test_loss:.5f}")

        if min_valid_loss < best_valid_loss:
            best_valid_loss = min_valid_loss
            best_model = final_model
            best_config = (lr, p_dropout, latent_dim)

    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    torch.save(best_model.state_dict(), f"{WEIGHTS_PATH}/encoder_weights.pth")
    print(f"\nBest config: lr={best_config[0]}, dropout={best_config[1]}, latent_dim={best_config[2]}")
    print(f"Best min valid loss: {best_valid_loss:.5f}")


def main() -> None:
    args = parse_args()

    # ------------------------
    # Set seeds and device
    # ------------------------
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # Download and prepare datasets
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
    # Setup optimizer and loss
    # ------------------------
    optimizer_class = torch.optim.Adam if args.optimizer == "ADAM" else torch.optim.SGD
    criterion = nn.MSELoss()

    if args.tune:
        # ------------------------
        # Run hyperparameter tuning
        # ------------------------
        learning_rates = [1e-3, 5e-4]
        dropouts = [0.1, 0.2, 0.3]
        latent_dims = [32, 64, 128]
        run_hyperparameter_tuning(
            train_loader,
            valid_loader,
            test_loader,
            optimizer_class,
            criterion,
            device,
            learning_rates,
            dropouts,
            latent_dims,
            epochs=args.epochs
        )
    else:
        # ------------------------
        # Train autoencoder
        # ------------------------
        print('----------------Training Autoencoder----------------')
        model = ConvAutoencoder(n=args.latent_dim, p_dropout=args.dropout).to(device)
        autoencoder_data = autoencoder_training(
            train_loader,
            valid_loader,
            test_loader,
            optimizer_class,
            criterion,
            device,
            model=model,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        autoenc_train_losses = autoencoder_data[0]
        autoenc_valid_losses = autoencoder_data[1]
        autoenc_test_loss = autoencoder_data[2]
        model = autoencoder_data[3]

        print(f"\nFinal Test Loss (MSE): {autoenc_test_loss:.5f}")

        # ------------------------
        # Plot and save losses
        # ------------------------
        plot_autoencoder_losses(
            autoenc_train_losses,
            autoenc_valid_losses,
            autoenc_test_loss,
            filename="autoencoder_loss_curve.png"
        )

        # ------------------------
        # Save Autoencoder weights.
        # ------------------------
        os.makedirs(WEIGHTS_PATH, exist_ok=True)
        torch.save(model.encoder.state_dict(), F"{WEIGHTS_PATH}/encoder_weights.pth")
        print("Encoder saved to encoder_weights.pth")


if __name__ == "__main__":
    main()
