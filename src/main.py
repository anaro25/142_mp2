# main.py
"""
Entry point for running compression experiments.

Responsibilities:
- Parse command-line arguments (e.g., which experiment, output folder, etc.)
- Call the high-level experiment runner in experiments.py
- Optionally print a short summary of results to the console

Usage examples:

    # Run all experiments, save results under ./results
    python -m main

    # Run all experiments with a custom results folder
    python -m main --mode all --output-dir out_results

    # Run a single experiment on one image
    python -m main --mode single --image houses/houses_2048.bmp --output-dir single_out

"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import run_all_experiments, run_single_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return an argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="Run image-compression experiments (LZW-BST vs LZW-Hashmap vs RLE)."
    )

    parser.add_argument(
        "--mode",
        choices=["all", "single"],
        default="all",
        help="Which experiments to run: "
             "'all' (default) runs the full suite, "
             "'single' runs a single-image experiment.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default="results",
        help="Directory where result CSVs and plots will be written (default: %(default)s).",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image when running with --mode single.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    output_dir = args.output_dir

    if args.mode == "all":
        print(f"[main] Running ALL experiments. Output directory: {output_dir}")
        run_all_experiments(output_dir=output_dir)
        print("[main] All experiments finished.")
    else:
        # Single-experiment mode
        if args.image is None:
            raise SystemExit(
                "Error: --image is required when --mode single.\n"
                "Example:\n"
                "  python -m main --mode single --image houses/houses_2048.bmp"
            )

        image_path = Path(args.image)
        print(
            f"[main] Running SINGLE experiment on {image_path}.\n"
            f"[main] Output directory: {output_dir}"
        )
        run_single_experiment(
            image_path=str(image_path),
            output_dir=output_dir,
        )
        print("[main] Single experiment finished.")


if __name__ == "__main__":
    main()
