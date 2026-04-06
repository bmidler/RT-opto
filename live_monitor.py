#!/usr/bin/env python3
"""Live training monitor — run in a separate terminal while training.

Watches output/history.json and refreshes plots every few seconds so you
can track progress in real time.

Usage:
    python live_monitor.py                        # defaults
    python live_monitor.py --output_dir output/   # custom dir
    python live_monitor.py --interval 10          # refresh every 10s
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt


def monitor(output_dir: str, interval: float):
    path = Path(output_dir) / "history.json"
    print(f"Watching {path} (refresh every {interval}s). Close the window to stop.")

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Live Training Monitor", fontsize=13, fontweight="bold")

    while True:
        if not path.exists():
            print("Waiting for history.json to appear...")
            time.sleep(interval)
            continue

        with open(path) as f:
            h = json.load(f)

        epochs = range(1, len(h["train_loss"]) + 1)

        for ax in axes:
            ax.clear()

        # Loss
        axes[0].plot(epochs, h["train_loss"], "o-", markersize=3, label="train")
        axes[0].plot(epochs, h["val_loss"], "o-", markersize=3, label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, h["train_acc"], "o-", markersize=3, label="train")
        axes[1].plot(epochs, h["val_acc"], "o-", markersize=3, label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="output/")
    p.add_argument("--interval", type=float, default=5.0)
    args = p.parse_args()
    monitor(args.output_dir, args.interval)
