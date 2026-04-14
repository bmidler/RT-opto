"""Training loop with TBPTT, early stopping, model checkpointing,
and optional multi-GPU distributed training via PyTorch DDP.

Training uses Truncated Backpropagation Through Time (TBPTT): sessions are
processed chunk-by-chunk in sequence, carrying the GRU hidden state across
chunk boundaries via .detach().  This faithfully replicates the deployment
setting (where state is carried across an entire 5-minute session) and exposes
the GRU to real temporal continuity rather than independent random chunks.

B parallel session streams are processed in lockstep per group, where
B = batch_size // grad_accum_steps.  When a session in a group ends before
the others, its stream is masked out of the loss computation and its hidden
state slot is zeroed.

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

from config import Config
from dataset import SessionChunkDataset, load_labels, split_sessions, compute_class_weights
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
# TBPTT training epoch
# ---------------------------------------------------------------------------

def train_tbptt(
    model: nn.Module,
    train_sessions: list[str],
    dataset: SessionChunkDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> tuple[float, float]:
    """One training epoch using Truncated Backpropagation Through Time.

    Sessions are shuffled, then grouped into batches of B parallel streams.
    The GRU hidden state is carried across chunk boundaries within each session
    (via .detach()) and zeroed only when a new session begins.

    Returns (mean_loss, mean_acc) averaged over all active chunks.
    """
    model.train()
    raw_model = _get_raw_model(model)
    rank0 = is_main_process(dist.get_rank() if dist.is_initialized() else 0)

    B = cfg.batch_size // cfg.grad_accum_steps  # parallel streams per forward

    # Shuffle sessions reproducibly per epoch
    rng = np.random.default_rng(cfg.seed + epoch)
    sessions = list(train_sessions)
    rng.shuffle(sessions)

    # Pad to a multiple of B so every group is full
    remainder = len(sessions) % B
    if remainder:
        # Repeat from the start to pad — these sessions will appear twice in
        # this epoch but that is acceptable for padding purposes.
        sessions += sessions[: B - remainder]

    n_groups = len(sessions) // B
    total_loss, total_acc, n_chunks = 0.0, 0.0, 0
    t_data = t_forward = t_backward = t_optim = 0.0
    global_step = 0

    optimizer.zero_grad()

    pbar = tqdm(range(n_groups), desc="  train", leave=False, unit="group",
                disable=not rank0)

    for g in pbar:
        group = sessions[g * B: (g + 1) * B]

        # Pre-compute how many chunks each session in this group has
        n_chunks_per_sess = [dataset.get_session_num_chunks(s) for s in group]
        max_chunks = max(n_chunks_per_sess)

        if max_chunks == 0:
            continue

        # Initialise hidden state for the whole group; carry across chunks
        h = raw_model.init_hidden(B, device)

        for chunk_idx in range(max_chunks):
            # Which streams are still active at this chunk
            active = [chunk_idx < n_chunks_per_sess[b] for b in range(B)]
            active_indices = [b for b, a in enumerate(active) if a]

            if not active_indices:
                break

            # ---- Data loading ----
            _t = time.perf_counter()
            chunks_frames, chunks_labels = [], []
            for b in range(B):
                if active[b]:
                    f, l = dataset.get_session_chunk(group[b], chunk_idx)
                else:
                    # Provide dummy tensors for inactive slots (masked from loss)
                    f, l = dataset.get_session_chunk(
                        group[active_indices[0]], chunk_idx)
                chunks_frames.append(f)
                chunks_labels.append(l)

            frames = torch.stack(chunks_frames).to(device, non_blocking=True)  # (B, T, 1, H, W)
            labels = torch.stack(chunks_labels).to(device, non_blocking=True)  # (B, T)
            t_data += time.perf_counter() - _t

            # ---- Forward ----
            _t = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
                logits, h_new = model(frames, h)  # (B, T, C)

                # Only compute loss on active streams to avoid polluting
                # gradients with dummy data from padded/finished sessions.
                active_t = torch.tensor(active, device=device)  # (B,)
                loss = criterion(
                    logits[active_t].reshape(-1, logits.size(-1)),
                    labels[active_t].reshape(-1),
                ) / cfg.grad_accum_steps

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward += time.perf_counter() - _t

            # ---- Backward ----
            _t = time.perf_counter()
            scaler.scale(loss).backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_backward += time.perf_counter() - _t

            # Carry hidden state; zero out streams that just finished
            h = h_new.detach()
            if not all(active):
                # Zero hidden state for inactive slots so they don't poison the
                # next group's initialisation (h is re-initialised per group,
                # so this is defensive).
                active_mask = torch.tensor(
                    active, dtype=h.dtype, device=device).view(1, -1, 1)
                h = h * active_mask

            # ---- Optimiser step (every grad_accum_steps chunks) ----
            global_step += 1
            if global_step % cfg.grad_accum_steps == 0:
                _t = time.perf_counter()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_optim += time.perf_counter() - _t

            batch_loss = loss.item() * cfg.grad_accum_steps
            batch_acc = compute_accuracy(
                logits[active_t], labels[active_t])
            total_loss += batch_loss
            total_acc += batch_acc
            n_chunks += 1

            del frames, labels, logits, loss, h_new

        pbar.set_postfix(loss=f"{total_loss/max(n_chunks,1):.4f}",
                         acc=f"{total_acc/max(n_chunks,1):.4f}")

    pbar.close()

    # Flush any leftover gradients
    if global_step % cfg.grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if rank0:
        t_total = t_data + t_forward + t_backward + t_optim
        if t_total > 0:
            print(
                f"  Timing — "
                f"data: {t_data:.1f}s ({100*t_data/t_total:.0f}%) | "
                f"forward: {t_forward:.1f}s ({100*t_forward/t_total:.0f}%) | "
                f"backward: {t_backward:.1f}s ({100*t_backward/t_total:.0f}%) | "
                f"optim: {t_optim:.1f}s ({100*t_optim/t_total:.0f}%)",
                flush=True,
            )

    return total_loss / max(n_chunks, 1), total_acc / max(n_chunks, 1)


# ---------------------------------------------------------------------------
# TBPTT validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_tbptt(
    model: nn.Module,
    val_sessions: list[str],
    dataset: SessionChunkDataset,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, float]:
    """Evaluate each validation session independently with a carried GRU state.

    Each session is processed as a single stream (B=1), carrying hidden state
    across chunks exactly as the deployed model would.  Sessions are sorted for
    reproducibility.  Returns (mean_loss, mean_acc) over all chunks.
    """
    model.eval()
    raw_model = _get_raw_model(model)
    rank0 = is_main_process(dist.get_rank() if dist.is_initialized() else 0)

    total_loss, total_acc, n_chunks = 0.0, 0.0, 0

    pbar = tqdm(sorted(val_sessions), desc="  val  ", leave=False, unit="sess",
                disable=not rank0)

    for sess in pbar:
        n = dataset.get_session_num_chunks(sess)
        if n == 0:
            continue

        h = raw_model.init_hidden(1, device)

        for chunk_idx in range(n):
            frames, labels = dataset.get_session_chunk(sess, chunk_idx)
            # Add batch dimension: (1, T, 1, H, W) and (1, T)
            frames = frames.unsqueeze(0).to(device, non_blocking=True)
            labels = labels.unsqueeze(0).to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
                logits, h = model(frames, h)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            h = h.detach()
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)
            n_chunks += 1

            del frames, labels, logits, loss

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
        torch.backends.cudnn.benchmark = True
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
        print(f"Training sessions: {train_sessions}", flush=True)
        print(f"Validation sessions: {val_sessions}", flush=True)

    if rank0:
        print("Building datasets...", flush=True)
    t0_ds = time.time()
    train_ds = SessionChunkDataset(
        train_sessions, labels_dict, cfg.video_root, cfg)
    val_ds = SessionChunkDataset(
        val_sessions, labels_dict, cfg.video_root, cfg)
    if rank0:
        print(f"Datasets ready in {time.time() - t0_ds:.1f}s.", flush=True)
        print(f"  Train chunks: {len(train_ds)}, "
              f"Val chunks: {len(val_ds)}", flush=True)
        B = cfg.batch_size // cfg.grad_accum_steps
        print(f"  Parallel streams per forward: {B} "
              f"(batch_size={cfg.batch_size}, "
              f"grad_accum_steps={cfg.grad_accum_steps})", flush=True)

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

    # Wrap in DDP after compile (recommended order for PyTorch >= 2.0)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank0:
            print("Model wrapped with DistributedDataParallel", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
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
        print(f"\nStarting TBPTT training for up to {cfg.max_epochs} epochs...",
              flush=True)

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_tbptt(
            model, train_sessions, train_ds,
            optimizer, criterion, device, cfg, scaler, epoch)
        vl_loss, vl_acc = validate_tbptt(
            model, val_sessions, val_ds, criterion, device, cfg)
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
                print(f"  Saved best model (val_loss={vl_loss:.4f})",
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
