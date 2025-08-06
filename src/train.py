import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from model import UNet, CombinedLoss
from dataset import create_dataloaders
from utils import (
    load_config, save_config, save_checkpoint, setup_wandb,
    log_images_to_wandb, calculate_metrics, get_device,
    set_seed, print_model_summary, create_experiment_dir, load_checkpoint
)

# ---------------------------------------------------------------------------
# Helper to safely coerce YAML strings -> float so optimizers don’t crash
# ---------------------------------------------------------------------------

def _to_float(cfg_section: dict, key: str, fallback: float | None = None) -> float:
    """Return cfg_section[key] as float.

    Args:
        cfg_section: Sub‑dict (e.g. config['training']).
        key: Key to look up.
        fallback: Default if key absent. If *None* and key missing or
                   value is not convertible, a ValueError is raised.
    """
    try:
        return float(cfg_section[key])
    except (KeyError, TypeError, ValueError):
        if fallback is not None:
            return fallback
        raise ValueError(
            f"Expected numeric value for '{key}', got {cfg_section.get(key)!r}")

# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_metrics = {"l1_loss": 0, "mse_loss": 0, "psnr": 0}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input"].to(device)
        targets = batch["output"].to(device)
        color_indices = batch["color_idx"].to(device)

        optimizer.zero_grad()
        preds = model(inputs, color_indices)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(preds, targets)
        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        if batch_idx % 100 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_l1_loss": metrics["l1_loss"],
                "train/batch_mse_loss": metrics["mse_loss"],
                "train/batch_psnr": metrics["psnr"],
            }, step=epoch * len(train_loader) + batch_idx)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{metrics['psnr']:.2f}"})

    n = len(train_loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {"l1_loss": 0, "mse_loss": 0, "psnr": 0}

    inputs_all, targets_all, preds_all, color_names = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            inputs = batch["input"].to(device)
            targets = batch["output"].to(device)
            color_indices = batch["color_idx"].to(device)
            color_names.extend(batch["color_name"])

            preds = model(inputs, color_indices)
            loss = criterion(preds, targets)

            metrics = calculate_metrics(preds, targets)
            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]

            inputs_all.append(inputs.cpu())
            targets_all.append(targets.cpu())
            preds_all.append(preds.cpu())

    n = len(val_loader)
    inputs_all = torch.cat(inputs_all)
    targets_all = torch.cat(targets_all)
    preds_all = torch.cat(preds_all)
    return (
        total_loss / n,
        {k: v / n for k, v in total_metrics.items()},
        inputs_all,
        targets_all,
        preds_all,
        color_names,
    )

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UNet for polygon colorization")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    device = get_device()
    print(f"▶ Using device: {device}")

    exp_name = f"unet_colorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = create_experiment_dir("experiments", exp_name)
    save_config(config, os.path.join(exp_dir, "config.yaml"))

    setup_wandb(config)

    train_loader, val_loader, num_colors = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        image_size=config["model"]["image_size"],
    )

    model = UNet(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"],
        bilinear=config["model"]["bilinear"],
        color_embedding_dim=config["model"]["color_embedding_dim"],
        num_colors=num_colors,
    ).to(device)
    print_model_summary(model)

    criterion = CombinedLoss(
        l1_weight=_to_float(config["training"], "l1_weight", 1.0),
        ssim_weight=_to_float(config["training"], "ssim_weight", 1.0),
    )

    lr = _to_float(config["training"], "learning_rate")
    weight_decay = _to_float(config["training"], "weight_decay", 0.0)
    min_lr = _to_float(config["training"], "min_lr", 0.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=min_lr)

    start_epoch = 0
    if args.resume:
        print(f"⏯ Resuming from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics, v_in, v_out, v_pred, v_names = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/l1_loss": train_metrics["l1_loss"],
            "train/mse_loss": train_metrics["mse_loss"],
            "train/psnr": train_metrics["psnr"],
            "val/loss": val_loss,
            "val/l1_loss": val_metrics["l1_loss"],
            "val/mse_loss": val_metrics["mse_loss"],
            "val/psnr": val_metrics["psnr"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if epoch % 5 == 0:
            log_images_to_wandb(v_in[:4], v_out[:4], v_pred[:4], v_names[:4], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(exp_dir, "best_model.pth"), config)
            print(f"💾 New best model (val loss {val_loss:.4f}) saved.")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(exp_dir, f"checkpoint_epoch_{epoch + 1}.pth"), config)

        print(
            f"Train Loss {train_loss:.4f} · Val Loss {val_loss:.4f} · "
            f"Train PSNR {train_metrics['psnr']:.2f} · Val PSNR {val_metrics['psnr']:.2f}"
        )

    save_checkpoint(model, optimizer, epochs - 1, val_loss, os.path.join(exp_dir, "final_model.pth"), config)
    print(f"🏁 Training completed! Best val loss {best_val_loss:.4f}\nExperiment artifacts saved to {exp_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from model import UNet, CombinedLoss
from dataset import create_dataloaders
from utils import (
    load_config, save_config, save_checkpoint, setup_wandb,
    log_images_to_wandb, calculate_metrics, get_device,
    set_seed, print_model_summary, create_experiment_dir, load_checkpoint
)

# ---------------------------------------------------------------------------
# Helper to safely coerce YAML strings -> float so optimizers don’t crash
# ---------------------------------------------------------------------------

def _to_float(cfg_section: dict, key: str, fallback: float | None = None) -> float:
    """Return cfg_section[key] as float.

    Args:
        cfg_section: Sub‑dict (e.g. config['training']).
        key: Key to look up.
        fallback: Default if key absent. If *None* and key missing or
                   value is not convertible, a ValueError is raised.
    """
    try:
        return float(cfg_section[key])
    except (KeyError, TypeError, ValueError):
        if fallback is not None:
            return fallback
        raise ValueError(
            f"Expected numeric value for '{key}', got {cfg_section.get(key)!r}")

# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_metrics = {"l1_loss": 0, "mse_loss": 0, "psnr": 0}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input"].to(device)
        targets = batch["output"].to(device)
        color_indices = batch["color_idx"].to(device)

        optimizer.zero_grad()
        preds = model(inputs, color_indices)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(preds, targets)
        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        if batch_idx % 100 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_l1_loss": metrics["l1_loss"],
                "train/batch_mse_loss": metrics["mse_loss"],
                "train/batch_psnr": metrics["psnr"],
            }, step=epoch * len(train_loader) + batch_idx)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{metrics['psnr']:.2f}"})

    n = len(train_loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {"l1_loss": 0, "mse_loss": 0, "psnr": 0}

    inputs_all, targets_all, preds_all, color_names = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            inputs = batch["input"].to(device)
            targets = batch["output"].to(device)
            color_indices = batch["color_idx"].to(device)
            color_names.extend(batch["color_name"])

            preds = model(inputs, color_indices)
            loss = criterion(preds, targets)

            metrics = calculate_metrics(preds, targets)
            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]

            inputs_all.append(inputs.cpu())
            targets_all.append(targets.cpu())
            preds_all.append(preds.cpu())

    n = len(val_loader)
    inputs_all = torch.cat(inputs_all)
    targets_all = torch.cat(targets_all)
    preds_all = torch.cat(preds_all)
    return (
        total_loss / n,
        {k: v / n for k, v in total_metrics.items()},
        inputs_all,
        targets_all,
        preds_all,
        color_names,
    )

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UNet for polygon colorization")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    device = get_device()
    print(f"▶ Using device: {device}")

    exp_name = f"unet_colorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = create_experiment_dir("experiments", exp_name)
    save_config(config, os.path.join(exp_dir, "config.yaml"))

    setup_wandb(config)

    train_loader, val_loader, num_colors = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        image_size=config["model"]["image_size"],
    )

    model = UNet(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"],
        bilinear=config["model"]["bilinear"],
        color_embedding_dim=config["model"]["color_embedding_dim"],
        num_colors=num_colors,
    ).to(device)
    print_model_summary(model)

    criterion = CombinedLoss(
        l1_weight=_to_float(config["training"], "l1_weight", 1.0),
        ssim_weight=_to_float(config["training"], "ssim_weight", 1.0),
    )

    lr = _to_float(config["training"], "learning_rate")
    weight_decay = _to_float(config["training"], "weight_decay", 0.0)
    min_lr = _to_float(config["training"], "min_lr", 0.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=min_lr)

    start_epoch = 0
    if args.resume:
        print(f"⏯ Resuming from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics, v_in, v_out, v_pred, v_names = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/l1_loss": train_metrics["l1_loss"],
            "train/mse_loss": train_metrics["mse_loss"],
            "train/psnr": train_metrics["psnr"],
            "val/loss": val_loss,
            "val/l1_loss": val_metrics["l1_loss"],
            "val/mse_loss": val_metrics["mse_loss"],
            "val/psnr": val_metrics["psnr"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if epoch % 5 == 0:
            log_images_to_wandb(v_in[:4], v_out[:4], v_pred[:4], v_names[:4], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(exp_dir, "best_model.pth"), config)
            print(f"💾 New best model (val loss {val_loss:.4f}) saved.")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(exp_dir, f"checkpoint_epoch_{epoch + 1}.pth"), config)

        print(
            f"Train Loss {train_loss:.4f} · Val Loss {val_loss:.4f} · "
            f"Train PSNR {train_metrics['psnr']:.2f} · Val PSNR {val_metrics['psnr']:.2f}"
        )

    save_checkpoint(model, optimizer, epochs - 1, val_loss, os.path.join(exp_dir, "final_model.pth"), config)
    print(f"🏁 Training completed! Best val loss {best_val_loss:.4f}\nExperiment artifacts saved to {exp_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
