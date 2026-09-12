import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except Exception:
            vgg = models.vgg19(weights=None).features

        self.slice = nn.Sequential(*[vgg[x] for x in range(16)]).eval()
        for param in self.slice.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr_rgb, hr_rgb):
        sr_norm = (sr_rgb - self.mean) / self.std
        hr_norm = (hr_rgb - self.mean) / self.std
        sr_feats = self.slice(sr_norm)
        hr_feats = self.slice(hr_norm)
        return F.l1_loss(sr_feats, hr_feats)


class SobelEdgeLoss(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)

        kernel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        kernel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, sr, hr):
        grad_sr_x = F.conv2d(sr, self.kernel_x, padding=1, groups=self.channels)
        grad_sr_y = F.conv2d(sr, self.kernel_y, padding=1, groups=self.channels)
        grad_hr_x = F.conv2d(hr, self.kernel_x, padding=1, groups=self.channels)
        grad_hr_y = F.conv2d(hr, self.kernel_y, padding=1, groups=self.channels)

        loss_x = F.l1_loss(grad_sr_x, grad_hr_x)
        loss_y = F.l1_loss(grad_sr_y, grad_hr_y)
        return loss_x + loss_y


class NDVILoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def compute_ndvi(self, x):
        red = x[:, 2:3, :, :]
        nir = x[:, 3:4, :, :]
        return (nir - red) / (nir + red + self.eps)

    def forward(self, sr, hr):
        ndvi_sr = self.compute_ndvi(sr)
        ndvi_hr = self.compute_ndvi(hr)
        return F.l1_loss(ndvi_sr, ndvi_hr)


class TerraSharpLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        perceptual_weight=0.1,
        edge_weight=0.5,
        ndvi_weight=0.3,
        adversarial_weight=0.005,
        use_adversarial=False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.ndvi_weight = ndvi_weight
        self.adversarial_weight = adversarial_weight
        self.use_adversarial = use_adversarial

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.edge_loss = SobelEdgeLoss(channels=4)
        self.ndvi_loss = NDVILoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr, discriminator_output=None):
        loss_l1 = self.l1_loss(sr, hr)

        sr_rgb = sr[:, [2, 1, 0], :, :]
        hr_rgb = hr[:, [2, 1, 0], :, :]
        loss_perceptual = self.perceptual_loss(sr_rgb, hr_rgb)

        loss_edge = self.edge_loss(sr, hr)
        loss_ndvi = self.ndvi_loss(sr, hr)

        total_loss = (
            self.l1_weight * loss_l1
            + self.perceptual_weight * loss_perceptual
            + self.edge_weight * loss_edge
            + self.ndvi_weight * loss_ndvi
        )

        loss_gan_adversarial = 0.0
        if discriminator_output is not None:
            # LSGAN generator objective: D(G(z)) should be judged as real (target 1.0)
            # The required map is scalar patch prediction map; MSE is the stable formulation.
            target = torch.ones_like(discriminator_output)
            loss_gan_adversarial = self.mse_loss(discriminator_output, target)
            total_loss = total_loss + self.adversarial_weight * loss_gan_adversarial

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_l1": loss_l1.item(),
            "loss_perceptual": loss_perceptual.item(),
            "loss_edge": loss_edge.item(),
            "loss_ndvi": loss_ndvi.item(),
            "loss_gan_adversarial": loss_gan_adversarial.item() if isinstance(loss_gan_adversarial, torch.Tensor) else float(loss_gan_adversarial),
        }

        return total_loss, loss_dict
