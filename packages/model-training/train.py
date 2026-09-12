import os
import time
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_arch import SwinIRLight
from losses import TerraSharpLoss
from dataset import Sentinel2WaldDataset
from discriminator import PatchGANDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="TerraSharp SwinIR-Light Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start from when resuming")
    parser.add_argument("--use_gan", action="store_true", default=False,
                        help="Enable optional GAN adversarial training path while keeping the supervised-only path available.")
    parser.add_argument("--gan_weight", type=float, default=None,
                        help="Override the adversarial LSGAN weight from the config.yaml file.")
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    args = parse_args()
    config = load_config(args.config)

    # Optional GAN config defaults remain in config.yaml and are additive.
    gan_cfg = config.get("gan", {}) if isinstance(config.get("gan"), dict) else {}
    gan_weight_default = float(gan_cfg.get("adversarial_weight", 0.005))
    if args.gan_weight is not None:
        gan_weight_default = args.gan_weight

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({gpu_name})")
        print("Estimated time per epoch (GPU): ~20-40 seconds")
    else:
        print(f"Using device: {device}")
        print("Estimated time per epoch (CPU): ~5-10 minutes")

    m_cfg = config.get("model", {})
    model = SwinIRLight(
        in_chans=m_cfg.get("in_chans", 4),
        out_chans=m_cfg.get("out_chans", 4),
        embed_dim=m_cfg.get("embed_dim", 64),
        depths=m_cfg.get("depths", [2, 2]),
        num_heads=m_cfg.get("num_heads", [4, 4]),
        window_size=m_cfg.get("window_size", 8),
        mlp_ratio=m_cfg.get("mlp_ratio", 2.0),
        upscale=m_cfg.get("upscale", 4)
    ).to(device)

    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("lr", 0.0002))

    l_cfg = config.get("loss", {})
    criterion = TerraSharpLoss(
        l1_weight=l_cfg.get("l1_weight", 1.0),
        perceptual_weight=l_cfg.get("perceptual_weight", 0.1),
        edge_weight=l_cfg.get("edge_weight", 0.5),
        ndvi_weight=l_cfg.get("ndvi_weight", 0.3),
        adversarial_weight=gan_weight_default,
        use_adversarial=args.use_gan,
    ).to(device)

    discriminator = None
    discriminator_optimizer = None
    if args.use_gan:
        discriminator = PatchGANDiscriminator(in_chans=m_cfg.get("in_chans", 4)).to(device)
        discriminator_lr = max(lr * 0.5, 0.00001)
        discriminator_optimizer = AdamW(
            discriminator.parameters(),
            lr=discriminator_lr,
            weight_decay=float(t_cfg.get("weight_decay", 0.0001))
        )

    epochs = t_cfg.get("epochs", 50)
    batch_size = t_cfg.get("batch_size", 8)
    min_lr = float(t_cfg.get("min_lr", 0.00001))
    weight_decay = float(t_cfg.get("weight_decay", 0.0001))
    grad_clip_norm = float(t_cfg.get("grad_clip_norm", config.get("grad_clip_norm", 1.0)))
    checkpoint_dir = t_cfg.get("checkpoint_dir", "packages/model-training/checkpoints")
    checkpoint_name = t_cfg.get("checkpoint_name", "best.pth")
    if checkpoint_dir.startswith("packages/model-training") and not os.path.exists("packages"):
        checkpoint_dir = os.path.relpath(checkpoint_dir, "packages/model-training")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    d_cfg = config.get("dataset", {})
    train_dataset = Sentinel2WaldDataset(
        hr_dir=d_cfg.get("train_hr_dir", "datasets/samples/train/hr"),
        lr_dir=d_cfg.get("train_lr_dir", "datasets/samples/train/lr"),
        hr_size=d_cfg.get("hr_size", 256),
        lr_size=d_cfg.get("lr_size", 64),
        scale=d_cfg.get("scale", 4),
        blur_kernel_size=d_cfg.get("blur_kernel_size", 5),
        blur_sigma=d_cfg.get("blur_sigma", 1.2)
    )

    val_dataset = Sentinel2WaldDataset(
        hr_dir=d_cfg.get("val_hr_dir", "datasets/samples/val/hr"),
        lr_dir=d_cfg.get("val_lr_dir", "datasets/samples/val/lr"),
        hr_size=d_cfg.get("hr_size", 256),
        lr_size=d_cfg.get("lr_size", 64),
        scale=d_cfg.get("scale", 4),
        blur_kernel_size=d_cfg.get("blur_kernel_size", 5),
        blur_sigma=d_cfg.get("blur_sigma", 1.2)
    )

    num_workers = d_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    best_val_loss = float("inf")
    start_epoch = args.start_epoch

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        is_structured_checkpoint = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint

        if is_structured_checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            resume_epoch = checkpoint_epoch + 1
            if args.start_epoch != 1:
                resume_epoch = args.start_epoch
            start_epoch = resume_epoch

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and args.start_epoch == 1:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                if "scheduler_state_dict" not in checkpoint:
                    print("Warning: checkpoint has no scheduler state; rebuilding the cosine schedule.")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
                for _ in range(max(start_epoch - 1, 0)):
                    scheduler.step()

            best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
            print(f"Resuming from {args.resume} at epoch {start_epoch} (checkpoint epoch {checkpoint_epoch}).")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = args.start_epoch
            print(
                "Warning: Resuming from raw weights only — optimizer and scheduler state reset, "
                "starting from epoch 1's schedule but with epoch-33 weights"
            )
            print(f"Resuming from {args.resume} at epoch {start_epoch}.")
    else:
        print("Starting a fresh training run from epoch 1.")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        if discriminator is not None:
            discriminator.train()
        train_loss_total = 0.0
        train_loss_l1 = 0.0
        train_loss_perceptual = 0.0
        train_loss_edge = 0.0
        train_loss_ndvi = 0.0
        train_loss_gan = 0.0
        train_batches = 0
        train_disc_loss = 0.0
        generator_step = 0

        for batch in train_loader:
            lr_imgs = batch["lr"].to(device)
            hr_imgs = batch["hr"].to(device)

            if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
                # One discriminator update every two generator steps.
                # On odd generator steps, skip the discriminator update but still
                # let the generator see the discriminator's last state.
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)

                disc_loss = None
                if generator_step % 2 == 0:
                    discriminator_optimizer.zero_grad()
                    real_logits = discriminator(hr_imgs)
                    fake_logits = discriminator(sr_imgs.detach())
                    real_target = torch.full_like(real_logits, 0.9)
                    fake_target = torch.zeros_like(fake_logits)
                    disc_real_loss = torch.nn.functional.mse_loss(real_logits, real_target)
                    disc_fake_loss = torch.nn.functional.mse_loss(fake_logits, fake_target)
                    disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)
                    disc_loss.backward()
                    discriminator_optimizer.step()

                # Generator loss with current discriminator judgment every step.
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                fake_logits_gen = discriminator(sr_imgs)
                loss, loss_dict = criterion(sr_imgs, hr_imgs, discriminator_output=fake_logits_gen)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if epoch == 1 and ((train_batches + 1) == 1 or (train_batches + 1) % 50 == 0):
                    print(
                        f"  [Epoch 1 | Step {train_batches + 1:03d}] Pre-clip Gradient Norm: {grad_norm.item():.4f} (clip limit: {grad_clip_norm})",
                        flush=True
                    )

                optimizer.step()
                if disc_loss is not None:
                    train_disc_loss += disc_loss.item()
                train_loss_gan += loss_dict.get("loss_gan_adversarial", 0.0)
                generator_step += 1
            else:
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                loss, loss_dict = criterion(sr_imgs, hr_imgs)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if epoch == 1 and ((train_batches + 1) == 1 or (train_batches + 1) % 50 == 0):
                    print(
                        f"  [Epoch 1 | Step {train_batches + 1:03d}] Pre-clip Gradient Norm: {grad_norm.item():.4f} (clip limit: {grad_clip_norm})",
                        flush=True
                    )

                optimizer.step()

            train_loss_total += loss_dict["loss_total"]
            train_loss_l1 += loss_dict["loss_l1"]
            train_loss_perceptual += loss_dict["loss_perceptual"]
            train_loss_edge += loss_dict["loss_edge"]
            train_loss_ndvi += loss_dict["loss_ndvi"]
            train_batches += 1

        scheduler.step()

        model.eval()
        if discriminator is not None:
            discriminator.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                lr_imgs = batch["lr"].to(device)
                hr_imgs = batch["hr"].to(device)
                sr_imgs = model(lr_imgs)
                if args.use_gan and discriminator is not None:
                    # Validation remains supervised-only as a stable metric path.
                    v_loss, _ = criterion(sr_imgs, hr_imgs)
                else:
                    v_loss, _ = criterion(sr_imgs, hr_imgs)
                val_loss_total += v_loss.item()
                val_batches += 1

        avg_train_loss = train_loss_total / max(train_batches, 1)
        avg_l1 = train_loss_l1 / max(train_batches, 1)
        avg_perc = train_loss_perceptual / max(train_batches, 1)
        avg_edge = train_loss_edge / max(train_batches, 1)
        avg_ndvi = train_loss_ndvi / max(train_batches, 1)
        avg_gan = train_loss_gan / max(train_batches, 1)
        avg_disc_loss = train_disc_loss / max(train_batches, 1)
        avg_val_loss = val_loss_total / max(val_batches, 1)

        current_lr = scheduler.get_last_lr()[0]
        epoch_duration = time.time() - epoch_start
        if args.use_gan:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Time: {epoch_duration:.2f}s | LR: {current_lr:.6f} | "
                f"Train Total: {avg_train_loss:.4f} (L1: {avg_l1:.4f}, VGG: {avg_perc:.4f}, Edge: {avg_edge:.4f}, NDVI: {avg_ndvi:.4f}, GANAdv: {avg_gan:.4f}) | "
                f"Disc Loss: {avg_disc_loss:.4f} | Val Total: {avg_val_loss:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Time: {epoch_duration:.2f}s | LR: {current_lr:.6f} | "
                f"Train Total: {avg_train_loss:.4f} (L1: {avg_l1:.4f}, VGG: {avg_perc:.4f}, Edge: {avg_edge:.4f}, NDVI: {avg_ndvi:.4f}) | "
                f"Val Total: {avg_val_loss:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "config": config
            }, best_checkpoint_path)
            print(f"--> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
