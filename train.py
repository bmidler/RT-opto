"""Training loop with early stopping, live plotting, model checkpointing,
and optional multi-GPU distributed training via PyTorch DDP.

Launch single-GPU:
    python run.py ...

Launch multi-GPU (e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 run.py ...

SLURM multi-node example:
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE \
         --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         run.py ...
"""

import json
import os
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config
from dataset import (SessionChunkDataset, load_labels, split_sessions,
                     compute_class_weights)
from model import VideoClassifier


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialise DDP if launched via torchrun / srun, otherwise single-GPU.

    Returns (local_rank, global_rank, world_size).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def _get_raw_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP / compiled wrappers to get the original module."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


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
                desc="  train", leave=False, unit="batch",
                disable=not is_main_process(
                    dist.get_rank() if dist.is_initialized() else 0))

    for step, (frames, labels) in pbar:
        frames = frames.to(device, non_blocking=True)   # (B, T, 1, H, W)
        labels = labels.to(device, non_blocking=True)    # (B, T)

        h = _get_raw_model(model).init_hidden(frames.size(0), device)

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
                desc="  val  ", leave=False, unit="batch",
                disable=not is_main_process(
                    dist.get_rank() if dist.is_initialized() else 0))

    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        h = _get_raw_model(model).init_hidden(frames.size(0), device)

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
    local_rank, global_rank, world_size = setup_distributed()
    rank0 = is_main_process(global_rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if rank0:
        print(f"Using device: {device}  (world_size={world_size})", flush=True)

    # --- Data ---
    labels_dict = load_labels(cfg.labels_pkl)
    train_sessions, val_sessions = split_sessions(labels_dict, cfg)
    if rank0:
        print(f"Sessions — train: {len(train_sessions)}, "
              f"val: {len(val_sessions)}", flush=True)

    num_classes = int(max(arr.max() for arr in labels_dict.values())) + 1
    if rank0:
        print(f"Number of classes: {num_classes}", flush=True)

    # Class-weight balancing (inverse-frequency)
    class_weights = compute_class_weights(
        labels_dict, train_sessions, num_classes).to(device)
    if rank0:
        print(f"Class weights: {class_weights.cpu().tolist()}", flush=True)

    if rank0:
        print(f"Loading datasets with batch size {cfg.batch_size}")
        print(f"Training sessions: {train_sessions}", flush=True)
        print(f"Validation sessions: {val_sessions}", flush=True)

    train_ds = SessionChunkDataset(
        train_sessions, labels_dict, cfg.video_root, cfg, training=True)
    val_ds = SessionChunkDataset(
        val_sessions, labels_dict, cfg.video_root, cfg, training=False)

    # Resolve number of DataLoader workers
    num_workers = cfg.resolve_num_workers(world_size)

    # Distributed samplers (None when single-GPU)
    train_sampler = (DistributedSampler(train_ds, shuffle=True)
                     if world_size > 1 else None)
    val_sampler = (DistributedSampler(val_ds, shuffle=False)
                   if world_size > 1 else None)

    micro_bs = cfg.batch_size // cfg.grad_accum_steps
    train_loader = DataLoader(
        train_ds, batch_size=micro_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,
    )

    if rank0:
        print(f"Loaded datasets:", flush=True)
        print(f"  Train: {len(train_ds)} chunks, "
              f"Val: {len(val_ds)} chunks", flush=True)
        print(f"  Effective batch size: {micro_bs} x {cfg.grad_accum_steps} "
              f"= {micro_bs * cfg.grad_accum_steps}", flush=True)
        print(f"  DataLoader workers per rank: {num_workers}", flush=True)

    # --- Model ---
    model = VideoClassifier(
        num_classes=num_classes,
        cnn_channels=cfg.cnn_channels,
        gru_hidden=cfg.gru_hidden,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
    ).to(device)

    if rank0:
        print(f"Model parameters: "
              f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    # torch.compile for kernel fusion / speed
    if device.type == "cuda":
        model = torch.compile(model)
        if rank0:
            print("Model compiled with torch.compile", flush=True)

    # Wrap in DDP after compile (recommended order for PyTorch ≥ 2.0)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank0:
            print("Model wrapped with DistributedDataParallel", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    # --- Training loop ---
    out = Path(cfg.output_dir)
    if rank0:
        out.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_time_sec": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if rank0:
        print(f"\nStarting training for up to {cfg.max_epochs} epochs...",
              flush=True)

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        # Tell distributed sampler which epoch we're in so it shuffles
        # differently each epoch.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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

        if rank0:
            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
                f"val loss {vl_loss:.4f}  acc {vl_acc:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )

        # Checkpoint best (rank 0 only)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            if rank0:
                raw = _get_raw_model(model)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vl_loss,
                    "val_acc": vl_acc,
                    "num_classes": num_classes,
                    "config": cfg.__dict__,
                }, cfg.model_save_path)
                print(f"  ✓ Saved best model (val_loss={vl_loss:.4f})",
                      flush=True)
        else:
            epochs_no_improve += 1

        # Save history every epoch (for live monitoring)
        if rank0:
            with open(out / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Early stopping
        if epochs_no_improve >= cfg.patience:
            if rank0:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)",
                      flush=True)
            break

    if rank0:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}",
              flush=True)
        print(f"Best model saved to: {cfg.model_save_path}", flush=True)

    cleanup_distributed()
    return history, cfg
