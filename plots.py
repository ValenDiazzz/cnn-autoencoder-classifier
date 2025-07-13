import matplotlib.pyplot as plt
from typing import List, Optional
import os


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
