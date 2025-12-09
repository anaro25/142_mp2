# image_io.py

"""
Image loading and conversion helpers.

Responsibilities:
- Load 2048x2048 BMP images from the specified folders:
    houses/houses_2048.bmp
    sunflowers/sunflowers_2048.bmp
    birthday/birthday_2048.bmp
- Convert image data to and from 1D byte arrays suitable for compression
- (Optionally) resize / crop images to smaller sizes for experiments
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

# You can later decide whether to use Pillow (PIL) or another library.
try:
    from PIL import Image
except ImportError:  # skeleton fallback
    Image = None  # type: ignore[assignment]


PROJECT_ROOT = Path(".")
IMAGE_PATHS = {
    "houses": PROJECT_ROOT / "houses" / "houses_2048.bmp",
    "sunflowers": PROJECT_ROOT / "sunflowers" / "sunflowers_2048.bmp",
    "birthday": PROJECT_ROOT / "birthday" / "birthday_2048.bmp",
}


def load_image(path: Path) -> "Image.Image":
    """Load an image from disk and return a PIL Image object."""
    if Image is None:
        raise RuntimeError("Pillow (PIL) is not installed.")
    return Image.open(path)


def image_to_bytes(img: "Image.Image") -> bytes:
    """Convert an image to a flat bytes object (e.g., grayscale or RGB)."""
    # TODO: decide on color mode (e.g., 'L' or 'RGB') and flatten logic
    return b""


def bytes_to_image(data: bytes, size: Tuple[int, int]) -> "Image.Image":
    """Convert raw bytes back into a PIL Image with the given size."""
    # TODO: implement reconstruction based on chosen color mode
    if Image is None:
        raise RuntimeError("Pillow (PIL) is not installed.")
    return Image.new("L", size)


def generate_resized_versions(
    img: "Image.Image", target_sizes: List[int]
) -> List["Image.Image"]:
    """Given a 2048x2048 image, generate smaller versions (e.g., 256x256, 512x512).

    target_sizes contains the dimension for a square image (e.g., 256 -> 256x256).
    """
    # TODO: implement resize pipeline
    return []
