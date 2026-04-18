import torch
import torch.nn as nn

class DeconvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=2, padding:int=1, output_padding:int=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim:int=4096, proj_dim=8192, out_channels:int=3, dropout:float=0.5, initial_shape=(512, 4, 4)):
        super().__init__()

        self.initial_c, self.initial_h, self.initial_w = initial_shape

        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.out_channels = out_channels

        self.proj_dim = nn.Linear(latent_dim, proj_dim)

        # Block 1
        # (Batch_size, 512, Height/16, Width/16) -> (Batch_size, 512, Height/16, Width/16)
        self.block1 = DeconvBlock(512, 512)
        
        # Block 2
        # (Batch_size, 512, Height/16, Width/16) -> (Batch_size, 256, Height/8, Width/8)
        self.block2 = DeconvBlock(512, 256)

        # Block 3
        # (Batch_size, 256, Height/8, Width/8) -> (Batch_size, 128, Height/4, Width/4)
        self.block3 = DeconvBlock(256, 128)

        # Block 4
        # (Batch_size, 128, Height/4, Width/4) -> (Batch_size, 64, Height/2, Width/2)
        self.block4 = DeconvBlock(128, 64)

        # Block 5
        # (Batch_size, 64, Height/2, Width/2) -> (Batch_size, out_channels, Height, Width)
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.proj_dim(x)
        x = x.view(-1, self.initial_c, self.initial_h, self.initial_w)  # Reshape to (Batch_size, 512, Height/16, Width/16)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x






