#!/usr/bin/env python3
"""Main entry point: train, evaluate, and benchmark.

Usage
-----
    # Full pipeline (train → evaluate → benchmark)
    python run.py --labels labels.pkl --video_root /path/to/sessions

    # Only plot training curves from an existing run
    python run.py --plot_only --output_dir output/

    # Only run evaluation + latency benchmark on a saved model
    python run.py --eval_only --labels labels.pkl --video_root /path/to/sessions

    # Override any config field
    python run.py --labels labels.pkl --video_root /data --lr 5e-4 --max_epochs 30
"""

import argparse
import sys

from config import Config
from train import train
from evaluate import plot_training_curves, full_evaluation, benchmark_latency


def parse_args():
    p = argparse.ArgumentParser(
        description="Train & evaluate a CNN-GRU video classifier.")

    # Modes
    p.add_argument("--plot_only", action="store_true",
                   help="Only regenerate training plots from history.json")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; run evaluation + latency on saved model")

    # Paths
    p.add_argument("--labels", type=str, help="Path to labels pkl file")
    p.add_argument("--video_root", type=str, help="Root dir containing sessions")
    p.add_argument("--output_dir", type=str, default="output/")
    p.add_argument("--model_save_path", type=str, default=None)

    # Training overrides
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--gru_hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max_dimension", type=int, default=None,
                   help="Downscale frames so the longest side is at most this "
                        "many pixels (aspect ratio preserved). Omit to use "
                        "the video's native resolution.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None,
                   help="Gradient accumulation steps (default 4)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed-precision training")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.labels:
        cfg.labels_pkl = args.labels
    if args.video_root:
        cfg.video_root = args.video_root
    cfg.output_dir = args.output_dir
    if args.model_save_path:
        cfg.model_save_path = args.model_save_path
    else:
        cfg.model_save_path = f"{cfg.output_dir}/best_model.pt"

    for field in ["lr", "batch_size", "max_epochs", "patience",
                  "gru_hidden", "dropout", "max_dimension", "seed",
                  "grad_accum_steps"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    if args.no_amp:
        cfg.use_amp = False

    # --- Plot-only mode ---
    if args.plot_only:
        plot_training_curves(f"{cfg.output_dir}/history.json", cfg.output_dir)
        return

    # --- Train ---
    if not args.eval_only:
        history, cfg = train(cfg)
        plot_training_curves(f"{cfg.output_dir}/history.json", cfg.output_dir)

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)
    full_evaluation(cfg)

    # --- Latency benchmark ---
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)
    benchmark_latency(cfg)


if __name__ == "__main__":
    main()
