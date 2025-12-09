
# experiments.py

"""
Experiment runner and result aggregation.

Responsibilities:
- Define which images and image sizes will be tested
- Run:
    * LZW-BST vs LZW-Hashmap (runtime comparison)
    * LZW-Hashmap vs RLE (runtime comparison)
    * LZW-Hashmap vs RLE (compression size comparison)
- Average results across the three images per data point
- Save raw results to disk (e.g., CSV)
- Plot the three required graphs (using matplotlib)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Callable, Tuple

import time

from lzw_bst import LZWBSTCompressor
from lzw_hashmap import LZWHashmapCompressor
from rle import RLECompressor
from image_io import IMAGE_PATHS, load_image, image_to_bytes


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment setting."""
    image_sizes: List[int]  # e.g. [256, 512, 1024, 2048]
    output_dir: Path


@dataclass
class Measurement:
    """Stores timing and size statistics for one run."""
    image_name: str
    size: int  # image dimension, e.g., 256 for 256x256
    algo_name: str
    runtime_seconds: float
    compressed_bytes: int


def run_single_experiment(
    image_path: str,
    output_dir: str,
) -> None:
    """Skeleton for running a single experiment on one image."""
    cfg = ExperimentConfig(
        image_sizes=[256, 512, 1024, 2048],
        output_dir=Path(output_dir),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement an actual experiment on this single image
    pass


def run_all_experiments(output_dir: str) -> None:
    """Run all required experiments and produce plots.

    High-level steps:
    1. Define image sizes and which algorithms to compare.
    2. Loop over sizes; for each size, run on all three images.
    3. Record runtime and compressed size for:
        - LZW-BST
        - LZW-Hashmap
        - RLE
    4. Average across images for each size and algorithm.
    5. Save CSV and plot the three graphs:
        - Graph 1: LZW-BST vs LZW-Hashmap (runtime)
        - Graph 2: LZW-Hashmap vs RLE (runtime)
        - Graph 3: LZW-Hashmap vs RLE (compression size)
    """
    cfg = ExperimentConfig(
        image_sizes=[256, 512, 1024, 2048],
        output_dir=Path(output_dir),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: build the main experimental loop here
    # Use helper functions (below) for timing and measuring sizes.
    pass


def time_compression(
    data: bytes,
    compress_fn: Callable[[bytes], object],
) -> Tuple[float, object]:
    """Measure runtime of compress_fn(data) and return (seconds, compressed_obj)."""
    start = time.perf_counter()
    compressed = compress_fn(data)
    end = time.perf_counter()
    return end - start, compressed


def estimate_compressed_size(compressed_obj: object) -> int:
    """Return the compressed size in bytes (depends on representation).

    For example:
    - LZW outputs a list of ints (codes).
    - RLE outputs a list of RLEToken objects.
    You'll have to decide how to translate those into a byte size.
    """
    # TODO: implement size estimation for each representation
    return 0


def save_results_to_csv(
    measurements: List[Measurement],
    csv_path: Path,
) -> None:
    """Save measurement results to a CSV file."""
    # TODO: write columns like: image_name, size, algo_name, runtime_seconds, compressed_bytes
    pass


def plot_results(csv_path: Path, output_dir: Path) -> None:
    """Read CSV results and generate the three required graphs using matplotlib.

    Graphs:
    1) Running time vs image size (LZW-BST vs LZW-Hashmap)
    2) Running time vs image size (LZW-Hashmap vs RLE)
    3) Compressed size vs image size (LZW-Hashmap vs RLE)
    """
    # TODO: implement plotting (matplotlib)
    pass
