import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_arch import SwinIRLight
from losses import TerraSharpLoss
from dataset import Sentinel2WaldDataset


def parse_args():
    parser = argparse.ArgumentParser(description="TerraSharp SwinIR-Light Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    l_cfg = config.get("loss", {})
    criterion = TerraSharpLoss(
        l1_weight=l_cfg.get("l1_weight", 1.0),
        perceptual_weight=l_cfg.get("perceptual_weight", 0.1),
        edge_weight=l_cfg.get("edge_weight", 0.5),
        ndvi_weight=l_cfg.get("ndvi_weight", 0.3)
    ).to(device)

    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 50)
    batch_size = t_cfg.get("batch_size", 8)
    lr = float(t_cfg.get("lr", 0.0002))
    min_lr = float(t_cfg.get("min_lr", 0.00001))
    weight_decay = float(t_cfg.get("weight_decay", 0.0001))
    checkpoint_dir = t_cfg.get("checkpoint_dir", "packages/model-training/checkpoints")
    checkpoint_name = t_cfg.get("checkpoint_name", "best.pth")

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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_loss_l1 = 0.0
        train_loss_perceptual = 0.0
        train_loss_edge = 0.0
        train_loss_ndvi = 0.0
        train_batches = 0

        for batch in train_loader:
            lr_imgs = batch["lr"].to(device)
            hr_imgs = batch["hr"].to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss, loss_dict = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            train_loss_total += loss_dict["loss_total"]
            train_loss_l1 += loss_dict["loss_l1"]
            train_loss_perceptual += loss_dict["loss_perceptual"]
            train_loss_edge += loss_dict["loss_edge"]
            train_loss_ndvi += loss_dict["loss_ndvi"]
            train_batches += 1

        scheduler.step()

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                lr_imgs = batch["lr"].to(device)
                hr_imgs = batch["hr"].to(device)
                sr_imgs = model(lr_imgs)
                v_loss, _ = criterion(sr_imgs, hr_imgs)
                val_loss_total += v_loss.item()
                val_batches += 1

        avg_train_loss = train_loss_total / max(train_batches, 1)
        avg_l1 = train_loss_l1 / max(train_batches, 1)
        avg_perc = train_loss_perceptual / max(train_batches, 1)
        avg_edge = train_loss_edge / max(train_batches, 1)
        avg_ndvi = train_loss_ndvi / max(train_batches, 1)
        avg_val_loss = val_loss_total / max(val_batches, 1)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | LR: {current_lr:.6f} | "
            f"Train Total: {avg_train_loss:.4f} (L1: {avg_l1:.4f}, VGG: {avg_perc:.4f}, Edge: {avg_edge:.4f}, NDVI: {avg_ndvi:.4f}) | "
            f"Val Total: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config
            }, best_checkpoint_path)
            print(f"--> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
