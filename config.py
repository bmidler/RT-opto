"""Configuration for the video classification pipeline."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Paths ---
    labels_pkl: str = "labels.pkl"          # Path to pkl with {session: cluster_ids}
    video_root: str = "data/"               # Root directory to search for session videos
    output_dir: str = "output/"             # Where to save model, plots, logs
    model_save_path: str = "output/best_model.pt"

    # --- Video ---
    fps: int = 120
    temporal_context_sec: float = 2.0       # Seconds of temporal context for GRU

    @property
    def seq_len(self) -> int:
        """Number of frames of temporal context."""
        return int(self.fps * self.temporal_context_sec)  # 240

    # --- Model ---
    cnn_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.3

    # --- Training ---
    batch_size: int = 16                    # Number of sequences per batch
    chunk_len: int = 240                    # Frames per training chunk (= seq_len)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 1000
    patience: int = 50                       # Early-stopping patience (epochs)
    val_fraction: float = 0.10              # Fraction of sessions for validation
    num_workers: int = 0                    # 0 = auto-detect (os.cpu_count())
    seed: int = 42
    grad_accum_steps: int = 4               # Gradient accumulation steps
    use_amp: bool = True                    # Mixed-precision training

    def resolve_num_workers(self, world_size: int = 1) -> int:
        """Return the actual number of DataLoader workers to use.

        When num_workers == 0 (the default), auto-detect from os.cpu_count()
        divided by the DDP world size so workers are not over-subscribed.
        """
        if self.num_workers > 0:
            return self.num_workers
        cpus = os.cpu_count() or 4
        return max(1, cpus // world_size)
