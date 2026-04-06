"""Dataset and data-loading utilities for video frame classification.

Key design choice: videos are decoded once (sequentially) during dataset
construction and stored as memory-mapped numpy arrays on disk.  This avoids
the critical bottleneck of seeking into compressed H.264/H.265 mp4 files on
every __getitem__ call — OpenCV's CAP_PROP_POS_FRAMES seek must decode
forward from the nearest keyframe, which can take seconds per chunk and
causes training to appear "hung".

Memmaps are opened lazily in __getitem__ so that the Dataset can be safely
pickled into DataLoader worker processes (num_workers > 0).
"""

import hashlib
import os
import pickle
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


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> normalised float32 grayscale (native resolution)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# One-time video -> memmap extraction
# ---------------------------------------------------------------------------

def _cache_key(video_path: Path) -> str:
    """Deterministic short hash so we can tell if a cache is still valid."""
    raw = f"{video_path.resolve()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _mmap_path_for(video_path: Path, cache_dir: Path) -> Path:
    """Return the expected cache file path for a given video."""
    return cache_dir / f"{_cache_key(video_path)}.npy"


def _decode_worker(
    video_path_str: str,
    sess_name: str,
    n_frames: int,
    out_h: int,
    out_w: int,
    cache_dir_str: str,
) -> tuple[str, str, int, int, int]:
    """Worker function for parallel decoding (runs in a subprocess).

    All arguments are plain types so they pickle cleanly.  Returns
    (sess_name, mmap_path, n_frames, out_h, out_w) so the parent can
    record where the file landed.
    """
    video_path = Path(video_path_str)
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = _mmap_path_for(video_path, cache_dir)

    if mmap_path.exists():
        return (sess_name, str(mmap_path), n_frames, out_h, out_w)

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    mmap = np.memmap(str(mmap_path), dtype=np.float32, mode="w+",
                     shape=(n_frames, out_h, out_w))

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        mmap[i] = preprocess_frame(frame)

    mmap.flush()
    del mmap
    cap.release()
    return (sess_name, str(mmap_path), n_frames, out_h, out_w)


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
    """Iterate over grayscale frames of an mp4 at their native resolution."""

    def __init__(self, video_path: str | Path, cfg: Config):
        self.path = str(video_path)
        _, self.native_h, self.native_w = get_video_info(video_path)
        print(f"    Native HxW: {self.native_h}x{self.native_w}", flush=True)

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield preprocess_frame(frame)
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

    On first construction the videos are decoded sequentially (fast,
    no seeking) and cached as memory-mapped numpy arrays.  Subsequent
    runs reuse the cache, making startup near-instant.

    Memmap file handles are opened *lazily* inside __getitem__ so the
    Dataset object can be pickled into DataLoader worker processes.

    Each item is (frames, labels) where:
        frames: float32 tensor  [chunk_len, 1, H, W]
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

        # Instead of storing live memmap objects (unpicklable), store the
        # path + shape so each DataLoader worker can open its own handle.
        self.mmap_meta: dict[str, tuple[str, tuple[int, int, int]]] = {}
        #   {sess: (mmap_path_str, (n_frames, out_h, out_w))}

        # Worker-local cache of open memmaps (populated lazily in __getitem__).
        # This dict is *not* pickled — each worker starts with an empty one.
        self._mmap_cache: dict[str, np.memmap] = {}

        cache_dir = Path(cfg.output_dir) / "frame_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ---- Phase 1: gather metadata for every session ----
        print("Resolving session video paths...", flush=True)
        sess_meta: dict[str, dict] = {}
        for sess in sessions:
            vid = find_session_video(video_root, sess)
            _, native_h, native_w = get_video_info(vid)
            self.frame_sizes[sess] = (native_h, native_w)

            n_frames = len(labels_dict[sess])
            sess_meta[sess] = dict(
                vid=vid, n_frames=n_frames, out_h=native_h, out_w=native_w)

        # ---- Phase 2: decode videos (parallel, cached on disk) ----
        to_decode: dict[str, dict] = {}
        for sess, m in sess_meta.items():
            mmap_path = _mmap_path_for(m["vid"], cache_dir)
            if mmap_path.exists():
                self.mmap_meta[sess] = (
                    str(mmap_path),
                    (m["n_frames"], m["out_h"], m["out_w"]),
                )
            else:
                to_decode[sess] = m

        if to_decode:
            max_workers = min(len(to_decode), os.cpu_count() or 1)
            print(f"Decoding {len(to_decode)} video(s) across "
                  f"{max_workers} workers...", flush=True)

            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                for sess, m in to_decode.items():
                    fut = pool.submit(
                        _decode_worker,
                        str(m["vid"]), sess, m["n_frames"],
                        m["out_h"], m["out_w"],
                        str(cache_dir),
                    )
                    futures[fut] = sess

                for i, fut in enumerate(as_completed(futures), 1):
                    sess_name, mmap_path_str, n_frames, out_h, out_w = \
                        fut.result()   # raises if worker failed
                    self.mmap_meta[sess_name] = (
                        mmap_path_str, (n_frames, out_h, out_w))
                    print(f"  [{i}/{len(futures)}] {sess_name} done "
                          f"({n_frames} frames)", flush=True)

            print("All videos decoded.", flush=True)
        else:
            print("All videos found in cache — skipping decode.", flush=True)

        # ---- Phase 3: build chunk index ----
        for sess in sessions:
            n_frames = len(labels_dict[sess])
            for start in range(0, n_frames - cfg.chunk_len + 1, cfg.chunk_len):
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
        """Return an open memmap for *sess*, creating the file handle on
        first access in this process."""
        mmap = self._mmap_cache.get(sess)
        if mmap is None:
            path_str, shape = self.mmap_meta[sess]
            mmap = np.memmap(path_str, dtype=np.float32, mode="r",
                             shape=shape)
            self._mmap_cache[sess] = mmap
        return mmap

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sess, start = self.index[idx]
        end = start + self.cfg.chunk_len

        # Fast array slice from the memory-mapped file
        frames = np.array(self._get_mmap(sess)[start:end])
        labels = self.labels_dict[sess][start:end]

        frames_t = torch.from_numpy(frames).unsqueeze(1)   # [T, 1, H, W]
        labels_t = torch.from_numpy(labels).long()          # [T]

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
