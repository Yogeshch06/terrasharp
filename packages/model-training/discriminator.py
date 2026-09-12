import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN-style 4-band discriminator for Sentinel-2 SR image realism.

    Input is a 4-band SR or HR chip tensor with shape (N, 4, H, W). The output is
    a patch-real/fake prediction map rather than a single scalar, following the
    LSGAN/no-sigmoid formulation requested for the TerraSharp GAN augmentation.
    """

    def __init__(self, in_chans=4):
        super().__init__()
        self.model = nn.Sequential(
            # First layer: no batch norm.
            nn.Conv2d(in_chans, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv layers 2-5: BatchNorm on all but first and final output layer.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final map layer: no BatchNorm, no sigmoid. LSGAN works directly on raw map score.
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.model(x)
