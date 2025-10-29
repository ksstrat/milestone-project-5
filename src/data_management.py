# src/data_management.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageSpec:
    """Target spatial size for images."""
    width: int = 100
    height: int = 100

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


def load_manifest(csv_path: Path) -> pd.DataFrame:
    """
    Load a split manifest (CSV with columns: filepath, label).
    Validates schema and normalizes file paths to absolute.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = {"filepath", "label"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {expected}, got: {set(df.columns)}")

    df["filepath"] = df["filepath"].apply(lambda p: str(Path(p).resolve()))
    return df


def load_manifests(manifest_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load train/val/test manifests from a directory into a dict.
    Keys: 'train', 'val', 'test'.
    """
    manifest_dir = Path(manifest_dir)
    paths = {
        "train": manifest_dir / "train.csv",
        "val": manifest_dir / "val.csv",
        "test": manifest_dir / "test.csv",
    }
    return {k: load_manifest(v) for k, v in paths.items()}


def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a stable label->index mapping from labels present in the dataframe.
    Sorted for determinism across runs/environments.
    """
    classes = sorted(df["label"].unique())
    return {label: idx for idx, label in enumerate(classes)}


def read_image_rgb01(path: Path, spec: ImageSpec) -> np.ndarray:
    """
    Read an image from disk, convert to RGB, resize to spec.size, and return float32 array in [0,1], shape [H,W,C].
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported extension: {path.suffix} for file {path}")

    with Image.open(path) as im:
        im = im.convert("RGB").resize(spec.size)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def batch_iterator(
    df: pd.DataFrame,
    batch_size: int = 32,
    spec: ImageSpec = ImageSpec(),
    shuffle: bool = True,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield batches (X, y) from a manifest DataFrame.

    X: float32 [B, H, W, C] in [0,1]
    y: int64 [B] with class indices as defined by build_label_map()
    """
    n = len(df)
    if n == 0:
        return
    rng = np.random.default_rng(seed) if shuffle else None
    label_map = build_label_map(df)
    indices = np.arange(n)

    if shuffle:
        rng.shuffle(indices)

    paths = df["filepath"].values
    labels = df["label"].map(label_map).astype(np.int64).values

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        X_list = [read_image_rgb01(Path(paths[i]), spec) for i in batch_idx]
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = labels[batch_idx]
        yield X, y