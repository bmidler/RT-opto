"""Post-training evaluation: plots, confusion matrix, per-class metrics,
and real-time inference latency benchmarking."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from config import Config
from dataset import (
    SessionChunkDataset, load_labels, split_sessions, find_session_video,
    get_video_info, preprocess_frame,
)
from model import VideoClassifier


# ===================================================================
# 1.  Training-curve plots  (can be run DURING or AFTER training)
# ===================================================================

def plot_training_curves(history_path: str, output_dir: str):
    """Read history.json and produce loss / accuracy / LR plots."""
    with open(history_path) as f:
        h = json.load(f)

    out = Path(output_dir)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, h["train_loss"], label="train")
    ax.plot(epochs, h["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, h["train_acc"], label="train")
    ax.plot(epochs, h["val_acc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, h["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Epoch duration
    ax = axes[1, 1]
    ax.bar(epochs, h["epoch_time_sec"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_title("Epoch Duration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out / 'training_curves.png'}")


# ===================================================================
# 2.  Full validation-set evaluation
# ===================================================================

@torch.no_grad()
def full_evaluation(cfg: Config):
    """Load best model, run on all val chunks, produce reports + plots."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    out = Path(cfg.output_dir)

    # Load model
    ckpt = torch.load(cfg.model_save_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]

    model = VideoClassifier(
        num_classes=num_classes,
        cnn_channels=cfg.cnn_channels,
        gru_hidden=cfg.gru_hidden,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Data
    labels_dict = load_labels(cfg.labels_pkl)
    _, val_sessions = split_sessions(labels_dict, cfg)
    val_ds = SessionChunkDataset(val_sessions, labels_dict, cfg.video_root, cfg)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    all_preds, all_labels = [], []
    for frames, labels in val_loader:
        frames = frames.to(device)
        h = model.init_hidden(frames.size(0), device)
        logits, _ = model(frames, h)
        preds = logits.argmax(dim=-1).cpu().numpy().ravel()
        all_preds.append(preds)
        all_labels.append(labels.numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # --- Classification report ---
    report = classification_report(all_labels, all_preds, zero_division=0)
    print("\n=== Classification Report (Validation Set) ===")
    print(report)
    with open(out / "classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6),
                                     max(8, num_classes * 0.6)))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (Validation Set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {out / 'confusion_matrix.png'}")

    # --- Per-class accuracy bar chart ---
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.4), 5))
    ax.bar(range(num_classes), per_class_acc, color="steelblue", alpha=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "per_class_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved per-class accuracy → {out / 'per_class_accuracy.png'}")

    return all_preds, all_labels


# ===================================================================
# 3.  Real-time latency benchmark
# ===================================================================

@torch.no_grad()
def benchmark_latency(cfg: Config, n_warmup: int = 50, n_frames: int = 500):
    """Simulate real-time streaming: decode one frame, run CNN+GRU, measure
    end-to-end latency per frame (decode + preprocess + inference)."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    out = Path(cfg.output_dir)

    ckpt = torch.load(cfg.model_save_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]

    model = VideoClassifier(
        num_classes=num_classes,
        cnn_channels=cfg.cnn_channels,
        gru_hidden=cfg.gru_hidden,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pick a val session to stream from
    labels_dict = load_labels(cfg.labels_pkl)
    _, val_sessions = split_sessions(labels_dict, cfg)
    vid_path = find_session_video(cfg.video_root, val_sessions[0])

    # Detect native frame size and compute output size (respects max_dimension)
    _, native_h, native_w = get_video_info(vid_path)
    out_h, out_w = compute_output_size(native_h, native_w, cfg.max_dimension)
    print(f"  Native frame size:  {native_h}H x {native_w}W")
    print(f"  Inference frame size: {out_h}H x {out_w}W")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {vid_path}")

    h = model.init_hidden(1, device)
    latencies = []

    for i in range(n_warmup + n_frames):
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        gray = preprocess_frame(frame, out_h, out_w, native_h, native_w)
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,1,H,W)

        logits, h = model(tensor, h)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            latencies.append(elapsed_ms)

    cap.release()
    latencies = np.array(latencies)

    # --- Report ---
    stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "max_fps": float(1000.0 / np.mean(latencies)),
        "device": str(device),
        "n_frames": n_frames,
    }
    frame_budget_ms = 1000.0 / cfg.fps  # 8.33 ms at 120 fps

    print("\n=== Real-Time Latency Benchmark ===")
    print(f"  Device:     {device}")
    print(f"  Mean:       {stats['mean_ms']:.2f} ms")
    print(f"  Median:     {stats['median_ms']:.2f} ms")
    print(f"  P95:        {stats['p95_ms']:.2f} ms")
    print(f"  P99:        {stats['p99_ms']:.2f} ms")
    print(f"  Max FPS:    {stats['max_fps']:.1f}")
    print(f"  Budget:     {frame_budget_ms:.2f} ms/frame ({cfg.fps} fps)")
    meets = stats["p95_ms"] < frame_budget_ms
    print(f"  Meets budget (p95 < {frame_budget_ms:.1f}ms): "
          f"{'YES' : <4}" if meets else f"  Meets budget (p95 < {frame_budget_ms:.1f}ms): NO")

    with open(out / "latency_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Latency histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latencies, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(frame_budget_ms, color="red", linestyle="--", linewidth=2,
               label=f"Budget ({frame_budget_ms:.1f} ms)")
    ax.axvline(stats["mean_ms"], color="orange", linestyle="-", linewidth=2,
               label=f"Mean ({stats['mean_ms']:.2f} ms)")
    ax.axvline(stats["p95_ms"], color="green", linestyle="-.", linewidth=2,
               label=f"P95 ({stats['p95_ms']:.2f} ms)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Frame Inference Latency (Streaming)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "latency_histogram.png", dpi=150)
    plt.close(fig)
    print(f"Saved latency histogram → {out / 'latency_histogram.png'}")

    # --- Latency over time ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(latencies, linewidth=0.5, alpha=0.7)
    ax.axhline(frame_budget_ms, color="red", linestyle="--", label="Budget")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Over Time (Streaming)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "latency_timeline.png", dpi=150)
    plt.close(fig)
    print(f"Saved latency timeline → {out / 'latency_timeline.png'}")

    return stats
