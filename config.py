"""Configuration for the video classification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Paths ---
    labels_pkl: str = "labels.pkl"          # Path to pkl with {session: cluster_ids}
    video_root: str = "data/"               # Root directory to search for session videos
    output_dir: str = "output/"             # Where to save model, plots, logs
    model_save_path: str = "output/best_model.pt"

    # --- Video ---
    fps: int = 120
    # Frame dimensions are detected automatically from the video at runtime.
    # If max_dimension is set, frames are downscaled so their longest side is
    # at most max_dimension pixels (aspect ratio preserved).  Leave as None to
    # use the native video resolution.
    max_dimension: int | None = 256
    temporal_context_sec: float = 2.0       # Seconds of temporal context for GRU

    @property
    def seq_len(self) -> int:
        """Number of frames of temporal context."""
        return int(self.fps * self.temporal_context_sec)  # 240

    # --- Model ---
    cnn_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 16                    # Number of sequences per batch
    chunk_len: int = 240                    # Frames per training chunk (= seq_len)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 1000
    patience: int = 50                       # Early-stopping patience (epochs)
    val_fraction: float = 0.10              # Fraction of sessions for validation
    num_workers: int = 4
    seed: int = 42
    grad_accum_steps: int = 4               # Gradient accumulation steps
    use_amp: bool = True                    # Mixed-precision training
