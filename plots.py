import os
from typing import List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_autoencoder_losses(
    train_losses: List[float],
    valid_losses: List[float],
    test_loss: Optional[float] = None,
    filename: str = "autoencoder_loss.png"
) -> None:
    """
    Plot training and validation MSE loss for the autoencoder.

    Args:
        train_losses (List[float]): Training losses over epochs.
        valid_losses (List[float]): Validation losses over epochs.
        test_loss (Optional[float]): Optional test loss to draw as a line.
        filename (str): Filename for saving the plot.

    Saves:
        A PNG figure in the 'Images' directory.
    """

    os.makedirs("Images", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(valid_losses, label='Validation MSE')

    if test_loss is not None:
        plt.axhline(
            y=test_loss,
            color='gray',
            linestyle='--',
            linewidth=1.2,
            label=f'Test MSE = {test_loss:.5f}'
        )

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join("Images", filename)
    plt.savefig(save_path)
    print(f"Saved loss plot to '{save_path}'")
    plt.close()


def plot_confusion_matrix(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    filename: str = "confusion_matrix.png"
) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        model (torch.nn.Module): Trained classifier model.
        loader (DataLoader): DataLoader to evaluate.
        device (torch.device): CPU or CUDA device.
        class_names (List[str]): Names for each class.
        filename (str): Filename for saving the plot.

    Saves:
        A PNG figure in the 'Images' directory.
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    os.makedirs("Images", exist_ok=True)
    save_path = os.path.join("Images", filename)
    plt.savefig(save_path)
    print(f"Saved confusion matrix to '{save_path}'")
    plt.close()


def plot_classifier_loss(
    train_losses: List[float],
    valid_losses: List[float],
    filename: str = "Images/classifier_loss.png"
) -> None:
    """
    Plot training and validation CrossEntropy loss.

    Args:
        train_losses (List[float]): Training losses over epochs.
        valid_losses (List[float]): Validation losses over epochs.
        filename (str): Filename for saving the plot.

    Saves:
        A PNG figure in the 'Images' directory.
    """
    os.makedirs("Images", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linestyle="-")
    plt.plot(valid_losses, label="Valid Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title("Classifier Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved classifier loss plot to '{filename}'")
    plt.close()


def plot_classifier_accuracy(
    train_accuracies: List[float],
    valid_accuracies: List[float],
    filename: str = "Images/classifier_accuracy.png"
) -> None:
    """
    Plot training and validation accuracy.

    Args:
        train_accuracies (List[float]): Training accuracy over epochs.
        valid_accuracies (List[float]): Validation accuracy over epochs.
        filename (str): Filename for saving the plot.

    Saves:
        A PNG figure in the 'Images' directory.
    """
    os.makedirs("Images", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Train Accuracy", linestyle="-")
    plt.plot(valid_accuracies, label="Valid Accuracy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved classifier accuracy plot to '{filename}'")
    plt.close()
