"""Dataset and data-loading utilities for video frame classification.

Key design choice: videos are decoded once (sequentially) during dataset
construction and stored as memory-mapped numpy arrays on disk.  This avoids
the critical bottleneck of seeking into compressed H.264/H.265 mp4 files on
every __getitem__ call — OpenCV's CAP_PROP_POS_FRAMES seek must decode
forward from the nearest keyframe, which can take seconds per chunk and
causes training to appear "hung".

Frames are stored as uint8 (0–255) and normalised to float32 on-the-fly in
__getitem__, reducing memmap size by 4× compared to float32 storage.

Temporal downsampling is applied during decode: only every *stride*-th frame
is written to the memmap, where stride = round(native_fps / cfg.target_fps).
If a video's native fps is below cfg.target_fps a warning is emitted and
stride is forced to 1.

Memmaps are opened lazily in __getitem__ so that the Dataset can be safely
pickled into DataLoader worker processes (num_workers > 0).
"""

import hashlib
import math
import multiprocessing as mp
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_session_video(root: str, session_name: str) -> Path:
    """Search root recursively for a directory matching session_name,
    then return the path to the non-tracked mp4 in Camera0/."""
    root = Path(root)
    matches = [d for d in root.rglob(session_name) if d.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"No directory named '{session_name}' found under {root}"
        )
    session_dir = matches[0]
    cam_dir = session_dir / "Camera0"
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Camera0 not found in {session_dir}")

    mp4s = [
        f for f in cam_dir.glob("*.mp4")
        if f.name != "tracked_video.mp4"
    ]
    if not mp4s:
        raise FileNotFoundError(
            f"No non-tracked mp4 found in {cam_dir}"
        )
    return mp4s[0]


def load_labels(pkl_path: str) -> dict[str, np.ndarray]:
    """Load {session_name: cluster_ids} from a pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Video introspection helpers
# ---------------------------------------------------------------------------

def get_video_info(video_path: str | Path) -> tuple[float, int, int]:
    """Return (fps, height, width) for the video without decoding frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return fps, h, w


def preprocess_frame(frame_bgr: np.ndarray,
                     spatial_scale: float = 1.0) -> np.ndarray:
    """BGR frame -> normalised float32 grayscale, optionally downscaled.

    Used by evaluate.py for real-time benchmarking.  The training pipeline
    stores uint8 frames in memmaps and normalises in __getitem__ instead.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if spatial_scale != 1.0:
        h, w = gray.shape[:2]
        new_w, new_h = int(w * spatial_scale), int(h * spatial_scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# One-time video -> memmap extraction
# ---------------------------------------------------------------------------

def _cache_key(video_path: Path, spatial_scale: float,
               stride: int = 1) -> str:
    """Deterministic short hash for cache-file naming.

    Includes stride and a version tag so that caches from different
    configurations (or older float32 caches) are never reused.
    """
    raw = f"{video_path.resolve()}:scale={spatial_scale}:stride={stride}:v2"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _mmap_path_for(video_path: Path, cache_dir: Path,
                   spatial_scale: float, stride: int = 1) -> Path:
    """Return the expected cache file path for a given video."""
    return cache_dir / f"{_cache_key(video_path, spatial_scale, stride)}.npy"


def _decode_worker(
    video_path_str: str,
    sess_name: str,
    n_frames: int,
    n_mmap_frames: int,
    out_h: int,
    out_w: int,
    cache_dir_str: str,
    spatial_scale: float = 1.0,
    stride: int = 1,
) -> tuple[str, str, int, int, int]:
    """Worker function for parallel decoding (runs in a subprocess).

    Reads every *stride*-th frame from the video and writes it as uint8 to a
    memory-mapped file.  Non-selected frames are advanced with cap.grab(),
    which skips the BGR→numpy conversion for a modest speed improvement.

    All arguments are plain types so they pickle cleanly.  Returns
    (sess_name, mmap_path, n_mmap_frames, out_h, out_w).
    """
    video_path = Path(video_path_str)
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = _mmap_path_for(video_path, cache_dir, spatial_scale, stride)

    if mmap_path.exists():
        return (sess_name, str(mmap_path), n_mmap_frames, out_h, out_w)

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    mmap = np.memmap(str(mmap_path), dtype=np.uint8, mode="w+",
                     shape=(n_mmap_frames, out_h, out_w))

    mmap_idx = 0
    for i in range(n_frames):
        if i % stride == 0:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if spatial_scale != 1.0:
                fh, fw = gray.shape[:2]
                new_w = int(fw * spatial_scale)
                new_h = int(fh * spatial_scale)
                gray = cv2.resize(gray, (new_w, new_h),
                                  interpolation=cv2.INTER_AREA)
            if mmap_idx < n_mmap_frames:
                mmap[mmap_idx] = gray
                mmap_idx += 1
        else:
            if not cap.grab():
                break

    mmap.flush()
    del mmap
    cap.release()
    return (sess_name, str(mmap_path), n_mmap_frames, out_h, out_w)


# ---------------------------------------------------------------------------
# Data augmentation (applied per-chunk during training)
# ---------------------------------------------------------------------------

def augment_chunk(frames: torch.Tensor) -> torch.Tensor:
    """Apply random spatial/photometric augmentations to a chunk [T, 1, H, W].

    Spatial transforms are consistent across the temporal chunk so that the
    GRU sees coherent motion.  Photometric noise is per-frame.
    """
    # Random horizontal flip (consistent across chunk)
    if torch.rand(1).item() < 0.5:
        frames = frames.flip(-1)

    # Random vertical flip (consistent across chunk)
    if torch.rand(1).item() < 0.3:
        frames = frames.flip(-2)

    # Random brightness shift
    if torch.rand(1).item() < 0.5:
        delta = (torch.rand(1).item() - 0.5) * 0.3  # [-0.15, +0.15]
        frames = (frames + delta).clamp_(0.0, 1.0)

    # Random contrast adjustment
    if torch.rand(1).item() < 0.5:
        factor = 0.7 + torch.rand(1).item() * 0.6  # [0.7, 1.3]
        mean = frames.mean()
        frames = ((frames - mean) * factor + mean).clamp_(0.0, 1.0)

    # Per-frame Gaussian noise
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(frames) * 0.02
        frames = (frames + noise).clamp_(0.0, 1.0)

    return frames


# ---------------------------------------------------------------------------
# Video reader (sequential, memory-efficient) — kept for evaluate.py compat
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """Iterate over grayscale frames of an mp4, optionally downscaled and
    temporally subsampled to cfg.target_fps."""

    def __init__(self, video_path: str | Path, cfg: Config):
        self.path = str(video_path)
        self.spatial_scale = cfg.spatial_scale
        fps_native, native_h, native_w = get_video_info(video_path)
        self.out_h = int(native_h * self.spatial_scale)
        self.out_w = int(native_w * self.spatial_scale)
        if fps_native < cfg.target_fps:
            warnings.warn(
                f"VideoFrameReader: native fps ({fps_native:.1f}) < "
                f"target fps ({cfg.target_fps}). stride=1."
            )
            self.stride = 1
        else:
            self.stride = max(1, round(fps_native / cfg.target_fps))
        print(f"    Frame HxW: {self.out_h}x{self.out_w} "
              f"(scale={self.spatial_scale}, stride={self.stride})", flush=True)

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        try:
            i = 0
            while True:
                if i % self.stride == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield preprocess_frame(frame, self.spatial_scale)
                else:
                    if not cap.grab():
                        break
                i += 1
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(labels_dict: dict[str, np.ndarray],
                          sessions: list[str],
                          num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Uses the "balanced" heuristic:  weight_c = n_total / (n_classes * n_c)
    This is the same formula used by sklearn.utils.class_weight.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for sess in sessions:
        labels = labels_dict[sess]
        for c in range(num_classes):
            counts[c] += (labels == c).sum()

    n_total = counts.sum()
    # Avoid division by zero for classes absent from the training set
    counts = np.maximum(counts, 1.0)
    weights = n_total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset: produces (chunk_of_frames, chunk_of_labels) pairs
# ---------------------------------------------------------------------------

class SessionChunkDataset(Dataset):
    """Pre-indexes all non-overlapping chunks across sessions for one epoch.

    On first construction the videos are decoded at cfg.target_fps (keeping
    every stride-th frame) and cached as uint8 memory-mapped arrays.
    Subsequent runs reuse the cache, making startup near-instant.

    Frames are stored as uint8 to reduce memmap size 4× vs float32.
    Normalisation to float32 (÷255) happens in __getitem__.

    Memmap file handles are opened *lazily* inside __getitem__ so the
    Dataset object can be pickled into DataLoader worker processes.

    Each item is (frames, labels) where:
        frames: float32 tensor  [chunk_len, 1, H, W]  (values in [0, 1])
        labels: int64 tensor    [chunk_len]
    """

    def __init__(self, sessions: list[str], labels_dict: dict[str, np.ndarray],
                 video_root: str, cfg: Config, training: bool = False):
        self.cfg = cfg
        self.labels_dict = labels_dict
        self.training = training

        # Build an index: list of (session_name, start_frame)
        self.index: list[tuple[str, int]] = []
        self.frame_sizes: dict[str, tuple[int, int]] = {}

        # Memmap metadata: {sess: (path_str, (n_mmap_frames, out_h, out_w))}
        self.mmap_meta: dict[str, tuple[str, tuple[int, int, int]]] = {}

        # Worker-local cache of open memmaps (not pickled — each worker opens
        # its own handle).
        self._mmap_cache: dict[str, np.memmap] = {}

        # Per-session temporal downsampling stride and subsampled labels.
        self._strides: dict[str, int] = {}
        self.subsampled_labels: dict[str, np.ndarray] = {}

        cache_dir = Path(cfg.output_dir) / "frame_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        scale = cfg.spatial_scale

        # ---- Phase 1: gather metadata and compute per-session strides ----
        print("Resolving session video paths and detecting frame rates...",
              flush=True)
        sess_meta: dict[str, dict] = {}
        for sess in sessions:
            vid = find_session_video(video_root, sess)
            fps_native, native_h, native_w = get_video_info(vid)

            # Confirm native fps and compute temporal downsampling stride.
            if fps_native < cfg.target_fps:
                warnings.warn(
                    f"Session '{sess}': native fps ({fps_native:.1f}) is less "
                    f"than target fps ({cfg.target_fps}). "
                    f"No temporal downsampling applied (stride=1)."
                )
                stride = 1
            else:
                stride = max(1, round(fps_native / cfg.target_fps))

            effective_fps = fps_native / stride
            print(
                f"  {sess}: {fps_native:.1f} fps native → "
                f"stride={stride} → {effective_fps:.1f} fps effective",
                flush=True,
            )
            self._strides[sess] = stride

            out_h = int(native_h * scale)
            out_w = int(native_w * scale)
            self.frame_sizes[sess] = (out_h, out_w)

            n_frames_native = len(labels_dict[sess])
            n_mmap_frames = math.ceil(n_frames_native / stride)

            sess_meta[sess] = dict(
                vid=vid,
                n_frames=n_frames_native,
                n_mmap_frames=n_mmap_frames,
                out_h=out_h,
                out_w=out_w,
                stride=stride,
            )

        # ---- Phase 2: decode videos (parallel, cached on disk) ----
        to_decode: dict[str, dict] = {}
        for sess, m in sess_meta.items():
            mmap_path = _mmap_path_for(m["vid"], cache_dir, scale, m["stride"])
            if mmap_path.exists():
                self.mmap_meta[sess] = (
                    str(mmap_path),
                    (m["n_mmap_frames"], m["out_h"], m["out_w"]),
                )
            else:
                to_decode[sess] = m

        if to_decode:
            max_workers = min(len(to_decode), min(4, os.cpu_count() or 1))
            print(
                f"Decoding {len(to_decode)} video(s) across "
                f"{max_workers} worker(s) as uint8 with temporal stride...",
                flush=True,
            )
            t_decode_start = time.time()

            futures = {}
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers,
                                     mp_context=mp_ctx) as pool:
                for sess, m in to_decode.items():
                    fut = pool.submit(
                        _decode_worker,
                        str(m["vid"]), sess,
                        m["n_frames"], m["n_mmap_frames"],
                        m["out_h"], m["out_w"],
                        str(cache_dir), scale, m["stride"],
                    )
                    futures[fut] = sess

                for i, fut in enumerate(as_completed(futures), 1):
                    sess_name, mmap_path_str, n_mmap_frames, out_h, out_w = \
                        fut.result()   # raises if worker failed
                    self.mmap_meta[sess_name] = (
                        mmap_path_str, (n_mmap_frames, out_h, out_w))
                    print(
                        f"  [{i}/{len(futures)}] {sess_name} done "
                        f"({n_mmap_frames} frames after temporal downsampling)",
                        flush=True,
                    )

            print(
                f"All videos decoded in {time.time() - t_decode_start:.1f}s.",
                flush=True,
            )
        else:
            print("All videos found in cache — skipping decode.", flush=True)

        # ---- Phase 3: build subsampled labels and chunk index ----
        for sess in sessions:
            stride = self._strides[sess]
            sub_labels = labels_dict[sess][::stride]
            self.subsampled_labels[sess] = sub_labels
            n_sub_frames = len(sub_labels)
            for start in range(0, n_sub_frames - cfg.chunk_len + 1,
                               cfg.chunk_len):
                self.index.append((sess, start))

        unique_sizes = set(self.frame_sizes.values())
        for sz in unique_sizes:
            print(f"  Frame size in use: {sz[0]}H x {sz[1]}W", flush=True)
        if len(unique_sizes) > 1:
            raise ValueError(
                "Multiple native resolutions detected across sessions. "
                "All videos must share the same resolution for batched "
                "training. Found: " + ", ".join(
                    f"{h}x{w}" for h, w in sorted(unique_sizes)))
        print(f"  Total chunks: {len(self.index)}", flush=True)

    # -- Lazy memmap accessor (safe across DataLoader workers) --

    def _get_mmap(self, sess: str) -> np.memmap:
        """Return an open uint8 memmap for *sess*, creating on first access."""
        mmap = self._mmap_cache.get(sess)
        if mmap is None:
            path_str, shape = self.mmap_meta[sess]
            mmap = np.memmap(path_str, dtype=np.uint8, mode="r", shape=shape)
            self._mmap_cache[sess] = mmap
        return mmap

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sess, start = self.index[idx]
        end = start + self.cfg.chunk_len

        # Load uint8 from memmap, normalise to float32 on-the-fly.
        frames = np.array(self._get_mmap(sess)[start:end])
        labels = self.subsampled_labels[sess][start:end]

        frames_t = torch.from_numpy(frames).unsqueeze(1).float() / 255.0
        labels_t = torch.from_numpy(labels).long()

        if self.training:
            frames_t = augment_chunk(frames_t)

        return frames_t, labels_t

    def __getstate__(self):
        """Drop the live memmap cache before pickling (for DataLoader
        workers).  Each worker will re-open memmaps lazily."""
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state


# ---------------------------------------------------------------------------
# Train / val split at session level
# ---------------------------------------------------------------------------

def split_sessions(labels_dict: dict[str, np.ndarray], cfg: Config):
    """Return (train_sessions, val_sessions) lists."""
    rng = np.random.RandomState(cfg.seed)
    sessions = sorted(labels_dict.keys())
    rng.shuffle(sessions)
    n_val = max(1, int(len(sessions) * cfg.val_fraction))
    return sessions[n_val:], sessions[:n_val]
