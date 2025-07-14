import os
import torch
import argparse
import numpy as np
from torch import nn
from data_utils import data_downloader, adapt_dataset, data_loaders
from models import ConvAutoencoder, Classifier
from train import train_loop_classifier, eval_loop_classifier
from plots import plot_classifier_loss, plot_classifier_accuracy, plot_confusion_matrix
from datasets import ClassificationDataset


WEIGHTS_PATH = "Weights"


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train CNN with Autoencoder on Fashion-MNIST")
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    args = parse_args()
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
        new_class=ClassificationDataset
    )
    train_loader, valid_loader, test_loader = data_loaders(
        train_dataset,
        test_dataset,
        batch_size=100
    )

    # -----------------------
    # Load pretrained encoder
    # -----------------------
    autoencoder = ConvAutoencoder()
    encoder_weights_path = WEIGHTS_PATH + "/encoder_weights.pth"
    autoencoder.encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    encoder = autoencoder.encoder
    encoder.eval()

    # -----------------------
    # Create classifier model
    # -----------------------
    classifier = Classifier(encoder=encoder, n_classes=10, p_dropout=0.3)
    classifier.to(device)

    # -----------------------
    # Setup optimizer and loss
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs

    # -----------------------
    # Training loop
    # -----------------------
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(epochs):
        valid_loss, valid_acc = eval_loop_classifier(
            classifier, valid_loader, criterion, device
        )
        train_loss, train_acc = train_loop_classifier(
            classifier, train_loader, criterion, optimizer, device
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}"
            )

    # -----------------------
    # Evaluate on test set
    # -----------------------
    test_loss, test_acc = eval_loop_classifier(
        classifier, test_loader, criterion, device
    )
    print(f"\nTest Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

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
