# separator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Separator(nn.Module):
    def __init__(self, latent_dim: int = 4096, num_classes: int = 5, dropout: float = 0.5, use_softmax: bool = True):
        super().__init__()
        self.use_softmax = use_softmax

        # (latent_dim) -> (num_classes)
        self.block = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.block(z)
        if self.use_softmax:
            return F.softmax(logits, dim=-1)
        return logits
