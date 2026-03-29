import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        
        # Encoder: (1, 10, 10) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Drops spatial dimensions from 10x10 to 5x5
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),     # Flattens 8 channels of 5x5 into a 200-element vector
            nn.Linear(200, latent_dim)
        )

        # Decoder: latent_dim -> (1, 10, 10)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Unflatten(1, (8, 5, 5)), 
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), # Upscales 5x5 back to 10x10
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh() # Tanh allows outputs in the range [-1.0, 1.0]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
