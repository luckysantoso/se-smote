# super_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from separator import Separator


class SuperEncoder(nn.Module):
    """
    Autoencoder + classifier head (separator) yang berbagi latent space.
    - Encoder menghasilkan vektor laten.
    - Decoder merekonstruksi gambar dari latent.
    - Separator mengklasifikasikan dari latent (logits).
    """
    def __init__(
        self,
        latent_dim: int = 4096,
        num_classes: int = 5,
        dropout_prob: float = 0.5,
        l2_normalize_latent: bool = True,
        use_adaptive_pool: bool = True
    ):
        super().__init__()
        self.l2_normalize_latent = l2_normalize_latent

        self.encoder = Encoder(
            in_channels=3,
            latent_dim=latent_dim,
            dropout=dropout_prob,
            use_adaptive_pool=use_adaptive_pool
        )
        self.decoder = Decoder(latent_dim=latent_dim)
        # IMPORTANT: CrossEntropyLoss expects logits, so disable softmax here
        self.separator = Separator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            dropout=dropout_prob,
            use_softmax=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.l2_normalize_latent:
            z = F.normalize(z, p=2, dim=1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.separator(z)  # logits

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat: reconstructed image in [0,1] (Sigmoid decoder)
            logits: class logits for CrossEntropyLoss
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        logits = self.classify(z)
        return x_hat, logits
