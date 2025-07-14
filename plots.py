import matplotlib.pyplot as plt
from typing import List, Optional
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_autoencoder_losses(
    train_losses: List[float],
    valid_losses: List[float],
    test_loss: Optional[float] = None,
    filename: str = "autoencoder_loss.png"
) -> None:
    """
    Plot training and validation loss curves for the autoencoder,
    and save the figure as a PNG in the 'Images' folder.
    """
    # Ensure the Images directory exists
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

    # Save the plot to the Images directory
    save_path = os.path.join("Images", filename)
    plt.savefig(save_path)
    print(f"Saved loss plot to '{save_path}'")
    plt.close()


def plot_confusion_matrix(model, loader, device, class_names, filename = "confusion_matrix.png"):
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
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    # Save figure
    save_path = os.path.join("Images", filename)
    plt.savefig(save_path)
    print(f"Saved loss/accuracy plot to 'Images/{filename}'")
    plt.close()


def plot_loss_accuracy(
    train_losses: List[float],
    valid_losses: List[float],
    train_accuracies: List[float],
    valid_accuracies: List[float],
    filename: str = "classifier_loss_accuracy.png"
) -> None:
    """
    Plot loss and accuracy curves for training and validation.

    Saves the figure to the 'Images' directory.
    """
    os.makedirs("Images", exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot losses on left axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CrossEntropy)", color="tab:red")
    ax1.plot(train_losses, label="Train Loss", color="tab:red", linestyle="-")
    ax1.plot(valid_losses, label="Valid Loss", color="tab:red", linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    # Plot accuracies on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(train_accuracies, label="Train Acc", color="tab:blue", linestyle="-")
    ax2.plot(valid_accuracies, label="Valid Acc", color="tab:blue", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Combined legend
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    fig.tight_layout()

    # Save figure
    save_path = os.path.join("Images", filename)
    plt.savefig(filename)
    print(f"Saved loss/accuracy plot to '{filename}'")
    plt.close()
