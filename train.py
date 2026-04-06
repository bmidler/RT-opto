"""Training loop with early stopping, live plotting, and model checkpointing."""

import json
import os
import time
from pathlib import Path

from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import SessionChunkDataset, load_labels, split_sessions
from model import VideoClassifier


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, scaler):
    model.train()
    total_loss, total_acc, n_chunks = 0.0, 0.0, 0

    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc="  train", leave=False, unit="batch")

    for step, (frames, labels) in pbar:
        frames = frames.to(device, non_blocking=True)   # (B, T, 1, H, W)
        labels = labels.to(device, non_blocking=True)    # (B, T)

        h = model.init_hidden(frames.size(0), device)

        with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
            logits, _ = model(frames, h)    # (B, T, C)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_loss = loss.item() * cfg.grad_accum_steps
        batch_acc = compute_accuracy(logits, labels)
        total_loss += batch_loss
        total_acc += batch_acc
        n_chunks += 1

        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

        del frames, labels, logits, loss, h

    pbar.close()

    # Handle leftover gradients when len(loader) is not divisible by
    # grad_accum_steps.
    if len(loader) % cfg.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_chunks, 1), total_acc / max(n_chunks, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss, total_acc, n_chunks = 0.0, 0.0, 0

    pbar = tqdm(loader, total=len(loader),
                desc="  val  ", leave=False, unit="batch")

    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        h = model.init_hidden(frames.size(0), device)

        with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
            logits, _ = model(frames, h)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, labels)
        n_chunks += 1

        del frames, labels, logits, loss, h

    pbar.close()

    return total_loss / max(n_chunks, 1), total_acc / max(n_chunks, 1)


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(cfg: Config):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    # --- Data ---
    labels_dict = load_labels(cfg.labels_pkl)
    train_sessions, val_sessions = split_sessions(labels_dict, cfg)
    print(f"Sessions — train: {len(train_sessions)}, val: {len(val_sessions)}",
          flush=True)

    num_classes = int(max(arr.max() for arr in labels_dict.values())) + 1
    print(f"Number of classes: {num_classes}", flush=True)

    print(f"Loading datasets with batch size {cfg.batch_size}")
    print(f"Training sessions: {train_sessions}", flush=True)
    print(f"Validation sessions: {val_sessions}", flush=True)

    train_ds = SessionChunkDataset(train_sessions, labels_dict, cfg.video_root, cfg)
    val_ds = SessionChunkDataset(val_sessions, labels_dict, cfg.video_root, cfg)

    # Make data loaders.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size // cfg.grad_accum_steps,
        shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size // cfg.grad_accum_steps,
        shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    micro_bs = cfg.batch_size // cfg.grad_accum_steps
    print(f"Loaded datasets:", flush=True)
    print(f"  Train: {len(train_ds)} chunks, "
          f"Val: {len(val_ds)} chunks", flush=True)
    print(f"  Effective batch size: {micro_bs} x {cfg.grad_accum_steps} "
          f"= {micro_bs * cfg.grad_accum_steps}", flush=True)
    print(f"  DataLoader workers: {cfg.num_workers}", flush=True)

    # --- Model ---
    model = VideoClassifier(
        num_classes=num_classes,
        cnn_channels=cfg.cnn_channels,
        gru_hidden=cfg.gru_hidden,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    # --- Training loop ---
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_time_sec": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\nStarting training for up to {cfg.max_epochs} epochs...",
          flush=True)

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, scaler)
        vl_loss, vl_acc = evaluate(
            model, val_loader, criterion, device, cfg)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(elapsed)

        print(
            f"Epoch {epoch:3d}/{cfg.max_epochs} | "
            f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
            f"val loss {vl_loss:.4f}  acc {vl_acc:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        # Checkpoint best
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "num_classes": num_classes,
                "config": cfg.__dict__,
            }, cfg.model_save_path)
            print(f"  ✓ Saved best model (val_loss={vl_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1

        # Save history every epoch (for live monitoring)
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {cfg.patience} epochs)", flush=True)
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"Best model saved to: {cfg.model_save_path}", flush=True)
    return history, cfg
