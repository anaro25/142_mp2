# experiments.py

"""
Experiment runner and result aggregation.

Responsibilities:
- Define which images and image sizes will be tested
- Run:
    * LZW-BST vs LZW-Hashmap (runtime comparison)
    * LZW-Hashmap vs RLE (runtime comparison)
    * LZW-Hashmap vs RLE (compression size comparison)
- Average results across the three images per data point (done at plotting stage)
- Save raw results to disk (CSV)
- Plot the three required graphs (using matplotlib)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Tuple
from collections import defaultdict
import csv
import time

from lzw_bst import LZWBSTCompressor
from lzw_hashmap import LZWHashmapCompressor
from rle import RLECompressor, RLEToken
from io_pipeline import (
    load_image,
    image_to_bytes,
    generate_resized_versions,
    prepare_image_bytes_for_sizes,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


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

    Representation assumptions:
    - LZW (BST / Hashmap) -> list[int], each code stored as a 16-bit value (2 bytes).
    - RLE -> list[RLEToken], each token stores (value: 1 byte, length: 1 byte) = 2 bytes.
    - bytes / bytearray -> len() bytes.

    If an unknown representation is encountered, this returns 0.
    """
    # Empty
    if compressed_obj is None:
        return 0

    # Raw byte-like
    if isinstance(compressed_obj, (bytes, bytearray)):
        return len(compressed_obj)

    # Lists (LZW codes or RLE tokens)
    if isinstance(compressed_obj, list):
        if not compressed_obj:
            return 0

        first = compressed_obj[0]

        # LZW: list of ints
        if isinstance(first, int):
            # Assume each code is stored in 2 bytes (16 bits)
            return 2 * len(compressed_obj)

        # RLE: list of RLEToken
        if isinstance(first, RLEToken):
            # Each token: 1 byte value + 1 byte length = 2 bytes
            return 2 * len(compressed_obj)

    # Fallback
    return 0


def save_results_to_csv(
    measurements: List[Measurement],
    csv_path: Path,
) -> None:
    """Save measurement results to a CSV file.

    Columns:
        image_name, size, algo_name, runtime_seconds, compressed_bytes
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_name",
        "size",
        "algo_name",
        "runtime_seconds",
        "compressed_bytes",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in measurements:
            writer.writerow(
                {
                    "image_name": m.image_name,
                    "size": m.size,
                    "algo_name": m.algo_name,
                    "runtime_seconds": m.runtime_seconds,
                    "compressed_bytes": m.compressed_bytes,
                }
            )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(csv_path: Path, output_dir: Path) -> None:
    """Read CSV results and generate the three required graphs using matplotlib.

    Graphs:
    1) Running time vs image size (LZW-BST vs LZW-Hashmap)
    2) Running time vs image size (LZW-Hashmap vs RLE)
    3) Compressed size vs image size (LZW-Hashmap vs RLE)

    For each (algorithm, size) pair we average across all images.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - plotting is optional
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with:\n"
            "    pip install matplotlib"
        ) from exc

    # Read CSV and group measurements
    runtime_by_algo_size = defaultdict(list)   # (algo, size) -> [runtimes]
    size_by_algo_size = defaultdict(list)      # (algo, size) -> [compressed_bytes]

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algo_name"]
            size = int(row["size"])
            runtime = float(row["runtime_seconds"])
            compressed_bytes = int(row["compressed_bytes"])

            key = (algo, size)
            runtime_by_algo_size[key].append(runtime)
            size_by_algo_size[key].append(compressed_bytes)

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Compute averages
    avg_runtime = {
        key: avg(vals)
        for key, vals in runtime_by_algo_size.items()
    }
    avg_compressed_size = {
        key: avg(vals)
        for key, vals in size_by_algo_size.items()
    }

    def make_runtime_plot(algos: List[str], filename: str, title: str) -> None:
        plt.figure()
        for algo in algos:
            xs = sorted(
                {size for (a, size) in avg_runtime.keys() if a == algo}
            )
            if not xs:
                continue
            ys = [avg_runtime[(algo, s)] for s in xs]
            plt.plot(xs, ys, marker="o", label=algo)
        plt.xlabel("Image size (pixels per side)")
        plt.ylabel("Average runtime (seconds)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    def make_size_plot(algos: List[str], filename: str, title: str) -> None:
        plt.figure()
        for algo in algos:
            xs = sorted(
                {size for (a, size) in avg_compressed_size.keys() if a == algo}
            )
            if not xs:
                continue
            ys = [avg_compressed_size[(algo, s)] for s in xs]
            plt.plot(xs, ys, marker="o", label=algo)
            # xs/ys may be empty if the algo wasn't run for some sizes
        plt.xlabel("Image size (pixels per side)")
        plt.ylabel("Average compressed size (bytes)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    # Graph 1: LZW-BST vs LZW-Hashmap (runtime)
    make_runtime_plot(
        ["LZW-BST", "LZW-Hashmap"],
        "graph1_runtime_lzw_bst_vs_hashmap.png",
        "Runtime: LZW-BST vs LZW-Hashmap",
    )

    # Graph 2: LZW-Hashmap vs RLE (runtime)
    make_runtime_plot(
        ["LZW-Hashmap", "RLE"],
        "graph2_runtime_lzw_hashmap_vs_rle.png",
        "Runtime: LZW-Hashmap vs RLE",
    )

    # Graph 3: LZW-Hashmap vs RLE (compressed size)
    make_size_plot(
        ["LZW-Hashmap", "RLE"],
        "graph3_size_lzw_hashmap_vs_rle.png",
        "Compressed size: LZW-Hashmap vs RLE",
    )


# ---------------------------------------------------------------------------
# Experiment frontends
# ---------------------------------------------------------------------------


def run_single_experiment(
    image_path: str,
    output_dir: str,
) -> None:
    """Run experiments on a single image across all configured sizes.

    This is mainly for quick testing / debugging. It will:
    - Resize the given image to each target size.
    - Run LZW-BST, LZW-Hashmap, and RLE.
    - Save a CSV with raw per-image measurements.
    - Generate the same three plots (with averages, though here there's
      only one image so averages = raw values).
    """
    cfg = ExperimentConfig(
        image_sizes=[256, 512, 1024, 2048],
        output_dir=Path(output_dir),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and resize the single image
    img_path = Path(image_path)
    img = load_image(img_path)
    resized_imgs = generate_resized_versions(img, cfg.image_sizes)

    compressors = {
        "LZW-BST": LZWBSTCompressor(),
        "LZW-Hashmap": LZWHashmapCompressor(),
        "RLE": RLECompressor(),
    }

    measurements: List[Measurement] = []
    image_name = img_path.stem

    for dim, resized_img in zip(cfg.image_sizes, resized_imgs):
        data = image_to_bytes(resized_img)

        for algo_name, compressor in compressors.items():
            runtime, compressed_obj = time_compression(data, compressor.compress)
            compressed_size_bytes = estimate_compressed_size(compressed_obj)

            measurements.append(
                Measurement(
                    image_name=image_name,
                    size=dim,
                    algo_name=algo_name,
                    runtime_seconds=runtime,
                    compressed_bytes=compressed_size_bytes,
                )
            )

    csv_path = cfg.output_dir / f"single_{image_name}.csv"
    save_results_to_csv(measurements, csv_path)
    plot_results(csv_path, cfg.output_dir)


def run_all_experiments(output_dir: str) -> None:
    """Run all required experiments and produce plots.

    High-level steps:
    1. Define image sizes and which algorithms to compare.
    2. Loop over sizes; for each size, run on all three images.
    3. Record runtime and compressed size for:
        - LZW-BST
        - LZW-Hashmap
        - RLE
    4. Save a CSV of raw per-image measurements.
    5. Plot the three graphs (averaging over images at plotting time):
        - Graph 1: LZW-BST vs LZW-Hashmap (runtime)
        - Graph 2: LZW-Hashmap vs RLE (runtime)
        - Graph 3: LZW-Hashmap vs RLE (compression size)
    """
    cfg = ExperimentConfig(
        image_sizes=[256, 512, 1024, 2048],
        output_dir=Path(output_dir),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare image bytes for each size and each image.
    # Structure: data_by_size[size][image_name] -> bytes
    data_by_size = prepare_image_bytes_for_sizes(cfg.image_sizes)

    compressors = {
        "LZW-BST": LZWBSTCompressor(),
        "LZW-Hashmap": LZWHashmapCompressor(),
        "RLE": RLECompressor(),
    }

    measurements: List[Measurement] = []

    for dim in cfg.image_sizes:
        size_dict = data_by_size.get(dim, {})
        for image_name, data in size_dict.items():
            for algo_name, compressor in compressors.items():
                runtime, compressed_obj = time_compression(data, compressor.compress)
                compressed_size_bytes = estimate_compressed_size(compressed_obj)

                measurements.append(
                    Measurement(
                        image_name=image_name,
                        size=dim,
                        algo_name=algo_name,
                        runtime_seconds=runtime,
                        compressed_bytes=compressed_size_bytes,
                    )
                )

    csv_path = cfg.output_dir / "all_measurements.csv"
    save_results_to_csv(measurements, csv_path)
    plot_results(csv_path, cfg.output_dir)
