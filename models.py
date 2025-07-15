import torch
import torch.nn as nn
from typing import Optional


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for Fashion-MNIST.

    This model uses two convolutional blocks in the encoder
    and two transposed convolutional blocks in the decoder
    to compress and reconstruct the input image.

    Args:
        n (int): Dimension of the latent space. Default is 64.
        p_dropout (float): Dropout probability. Default is 0.2.
    """
    def __init__(self, n: int = 64, p_dropout: float = 0.2) -> None:
        super().__init__()
        self.n = n
        self.p_dropout = p_dropout

        # Encoder: 2 convolutional layers followed by a fully connected layer
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # (1,28,28) -> (16,26,26)
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.MaxPool2d(2),                 # (16,26,26) -> (16,13,13)

            nn.Conv2d(16, 32, kernel_size=3),  # (16,13,13) -> (32,11,11)
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.MaxPool2d(2),                 # (32,11,11) -> (32,5,5)

            nn.Flatten(),                    # (32,5,5) -> (800,)
            nn.Linear(32 * 5 * 5, self.n),
            nn.ReLU(),
            nn.Dropout(self.p_dropout)
        )

        # Decoder: linear layer + transposed convolutions
        self.decoder = nn.Sequential(
            nn.Linear(self.n, 32 * 5 * 5),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Unflatten(1, (32, 5, 5)),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class Classifier(nn.Module):
    """
    A classifier that uses a pre-trained encoder to extract latent features
    and a small MLP to classify into categories.

    Args:
        encoder (nn.Module): A frozen encoder network.
        n_classes (int): Number of output classes. Default is 10.
        p_dropout (float): Dropout probability in the classifier head. Default is 0.1.
    """
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int = 10,
        p_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        device = next(self.encoder.parameters()).device

        # Automatically infer latent dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28, device=device)
            latent = self.encoder(dummy_input)
            latent_dim = latent.shape[1]

        # Build classifier head based on inferred latent dimension
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        """
        Perform a forward pass through the classifier.
        """
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits
