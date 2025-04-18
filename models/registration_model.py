import torch
import torch.nn as nn

class RegistrationCNN(nn.Module):
    def __init__(self):
        super(RegistrationCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )
        
    def forward(self, fixed, moving):
        # Concatenate fixed and moving images
        x = torch.cat([fixed, moving], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x 