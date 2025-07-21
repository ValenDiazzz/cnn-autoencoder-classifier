import os
import argparse

import numpy as np
import torch
from torch import nn

from data_utils import data_downloader, adapt_dataset, data_loaders
from models import ConvAutoencoder, Classifier
from train import classifier_training
from plots import (
    plot_classifier_loss,
    plot_classifier_accuracy,
    plot_confusion_matrix
)
from datasets import ClassificationDataset


WEIGHTS_PATH = "Weights"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train CNN with Autoencoder on Fashion-MNIST"
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
        "--freeze_encoder",
        '-fe',
        action="store_true",
        help="If specified, encoder weights are frozen during classifier training."
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


def main() -> None:
    args = parse_args()

    # ------------------------
    # Set seeds and device
    # ------------------------
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load datasets for classification
    # -----------------------
    train_dataset, test_dataset = data_downloader()
    train_dataset, test_dataset = adapt_dataset(
        train_dataset,
        test_dataset,
        dataset_wrapper=ClassificationDataset
    )
    train_loader, valid_loader, test_loader = data_loaders(
        train_dataset,
        test_dataset,
        batch_size=100
    )

    # -----------------------
    # Load pretrained encoder (with automatic latent_dim detection)
    # -----------------------
    encoder_weights_path = WEIGHTS_PATH + "/encoder_weights.pth"
    state_dict = torch.load(encoder_weights_path, map_location=device)

    # Detect latent_dim
    try:
        latent_dim = state_dict['9.weight'].shape[0]
        print(f"[INFO] Detected latent_dim from encoder weights: {latent_dim}")
    except KeyError:
        raise RuntimeError(
            "Could not find '9.weight' in state_dict. "
            "Make sure your encoder architecture hasn't changed."
        )

    autoencoder = ConvAutoencoder(n=latent_dim)
    autoencoder.encoder.load_state_dict(state_dict)
    encoder = autoencoder.encoder

    # -----------------------
    # Create classifier model
    # -----------------------
    classifier = Classifier(
        encoder=encoder,
        n_classes=10,
        p_dropout=0.3,
        freeze_encoder=args.freeze_encoder
    )
    classifier.to(device)

    # -----------------------
    # Training loop
    # -----------------------
    print('----------------Training Classifier----------------')

    train_data = classifier_training(
        train_loader,
        valid_loader,
        test_loader,
        torch.optim.Adam,
        nn.CrossEntropyLoss(),
        device,
        classifier,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )

    train_losses, valid_losses = train_data[0], train_data[1]
    train_accuracies, valid_accuracies = train_data[2], train_data[3]
    test_loss, test_acc = train_data[4], train_data[5]
    classifier = train_data[6]

    print(f"\nFinal Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

    # -----------------------
    # Plot results
    # -----------------------
    os.makedirs("Images", exist_ok=True)
    plot_classifier_loss(
        train_losses, valid_losses,
        filename="Images/classifier_loss.png"
    )
    plot_classifier_accuracy(
        train_accuracies, valid_accuracies,
        filename="Images/classifier_accuracy.png"
    )
    plot_confusion_matrix(
        classifier, test_loader, device,
        class_names=[str(i) for i in range(10)]
    )

    # -----------------------
    # Save classifier weights
    # -----------------------
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    torch.save(classifier.state_dict(), f"{WEIGHTS_PATH}/classifier_weights.pth")
    print(f"Classifier weights saved to '{WEIGHTS_PATH}/classifier_weights.pth'")


if __name__ == "__main__":
    main()
