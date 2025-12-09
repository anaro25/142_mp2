# main.py

"""
Entry point for running compression experiments.

Responsibilities:
- Parse command-line arguments (e.g., which experiment, output folder, etc.)
- Call the high-level experiment runner in experiments.py
- Optionally print a short summary of results to the console
"""

from __future__ import annotations

from experiments import run_all_experiments, run_single_experiment


def parse_args() -> dict:
    """Parse command-line arguments and return a config dictionary.

    For now this is just a skeleton; fill in with argparse later.
    """
    config: dict = {
        "mode": "all",  # "all" or "single"
        "output_dir": "results",
    }
    return config


def main() -> None:
    """Main entry point."""
    config = parse_args()

    if config["mode"] == "all":
        run_all_experiments(output_dir=config["output_dir"])
    else:
        # Skeleton for a single experiment
        run_single_experiment(
            image_path="houses/houses_2048.bmp",
            output_dir=config["output_dir"],
        )


if __name__ == "__main__":
    main()
