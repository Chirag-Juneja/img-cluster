import torch.nn as nn
import torchvision.models as models


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.efficientnet_b0(weights="IMAGENET1K_V1").features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                1280, 360, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                360, 80, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                80, 36, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                36, 12, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                12, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
