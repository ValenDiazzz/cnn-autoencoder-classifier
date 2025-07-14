import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for Fashion-MNIST dataset.

    Parameters
    ----------
    n : int, optional
        Dimension of the latent space. Default is 64.
    p_dropout : float, optional
        Dropout probability. Default is 0.2.
    """
    def __init__(
        self,
        n: int = 64,
        p_dropout: float = 0.2
    ):
        super().__init__()

        self.n = n
        self.p_dropout = p_dropout

        # Encoder: 2 convolutional blocks + fully connected layer
        self.encoder = nn.Sequential(
            nn.Conv2d(                      # (1,28,28) -> (16,26,26)
                in_channels=1,
                out_channels=16,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.MaxPool2d(kernel_size=2),    # (16,26,26) -> (16,13,13)

            nn.Conv2d(                      # (16,13,13) -> (32,11,11)
                in_channels=16,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.MaxPool2d(kernel_size=2),    # (32,11,11) -> (32,5,5)

            nn.Flatten(),                   # (32,5,5) -> (32 * 5 * 5)
            nn.Linear(32 * 5 * 5, self.n),  # (32 * 5 * 5) -> (n)
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n, 32 * 5 * 5),  # (n) -> (32 * 5 * 5)
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Unflatten(                   # (32 * 5 * 5) -> (32,5,5)
                dim=1,
                unflattened_size=(32, 5, 5)
            ),

            nn.ConvTranspose2d(             # (32,5,5) -> (16,13,13)
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                output_padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),

            nn.ConvTranspose2d(             # (16,13,13) -> (1,28,28)
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                output_padding=1
            ),
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
    def __init__(self, encoder, n_classes=10, p_dropout=0.3):
        super().__init__()
        
        self.encoder = encoder
        self.encoder.eval()
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        with torch.no_grad():
            latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits