import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels:int=3, latent_dim:int=4096, dropout:float=0.5, use_adaptive_pool:bool=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_adaptive_pool = use_adaptive_pool

        # Block 1 (3 -> 64)
        self.block1 = nn.Sequential(
            # (Batch_size, 3, Height, Width) -> (Batch_size, 64, Height, Width)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            # (Batch_size, 3, Height, Width) -> (Batch_size, 64, Height, Width)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            # (Batch_size, 64, Height, Width) -> (Batch_size, 64, Height/2, Width/2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
        )

        # Block 2 (64 -> 128)
        self.block2 = nn.Sequential(
            # (Batch_size, 64, Height/2, Width/2) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 128, Height/4, Width/4)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout),
        )

        # Block 3 (128 -> 256)
        self.block3 = nn.Sequential(
            # (Batch_size, 128, Height/4, Width/4) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            # (Batch_size, 256, Height/4, Width/4) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            # (Batch_size, 256, Height/4, Width/4) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            # (Batch_size, 256, Height/4, Width/4) -> (Batch_size, 256, Height/8, Width/8)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout),
        )

        # Block 4 (256 -> 512)
        self.block4 = nn.Sequential(
            # (Batch_size, 256, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # Block 5 (512 -> latent_dim)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
        )

        # Pool ke (4,4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))

        # Flatten + Linear ke latent vector
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width)
        x = self.block1(x)   # -> (Batch_size, 64, Height/2, Width/2)
        x = self.block2(x)   # -> (Batch_size, 128, Height/4, Width/4)
        x = self.block3(x)   # -> (Batch_size, 256, Height/8, Width/8)
        x = self.block4(x)   # -> (Batch_size, 512, Height/8, Width/8)
        x = self.block5(x)   # -> (Batch_size, latent_dim, Height/16, Width/16)

        if self.use_adaptive_pool:
            x = self.adaptive_pool(x)  # -> (Batch_size, latent_dim, 4, 4)

        x = self.flaten(x)    # -> (Batch_size, latent_dim * 4 * 4)
        x = self.fc(x)        # -> (Batch_size, latent_dim)

        return x




        