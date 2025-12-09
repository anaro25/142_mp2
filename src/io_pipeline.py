# io_pipeline.py
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
from typing import Tuple, List, Dict

# You can later decide whether to use Pillow (PIL) or another library.
try:
    from PIL import Image
except ImportError:  # skeleton fallback
    Image = None  # type: ignore[assignment]

# Root of the project (directory that contains src/, data/, results/, ...)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root of the images folder
IMAGES_ROOT = PROJECT_ROOT / "data" / "images"

# Known image locations (relative to PROJECT_ROOT).
IMAGE_PATHS: Dict[str, Path] = {
    "houses": IMAGES_ROOT / "houses" / "houses_2048.bmp",
    "sunflowers": IMAGES_ROOT / "sunflowers" / "sunflowers_2048.bmp",
    "birthday": IMAGES_ROOT / "birthday" / "birthday_2048.bmp",
}

# Chosen color mode for the experiments.
# "L" = 8-bit grayscale (values 0–255).
COLOR_MODE = "L"


def _ensure_pillow() -> None:
    """Raise a helpful error if Pillow is not installed."""
    if Image is None:
        raise RuntimeError(
            "Pillow (PIL) is not installed. Install it with:\n"
            "    pip install Pillow"
        )


def load_image(path: Path) -> "Image.Image":
    """Load an image from disk and return a PIL Image object in COLOR_MODE.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    Image.Image
        Loaded image converted to COLOR_MODE.
    """
    _ensure_pillow()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    # Convert to a consistent mode (e.g., grayscale) to make byte handling simple.
    return img.convert(COLOR_MODE)


def image_to_bytes(img: "Image.Image") -> bytes:
    """Convert an image to a flat bytes object (e.g., grayscale).

    The image is converted to COLOR_MODE (e.g., 'L') if necessary, then flattened
    row-major into a single bytes object.

    Parameters
    ----------
    img : Image.Image
        Input image.

    Returns
    -------
    bytes
        Flat byte representation suitable for feeding into compressors.
    """
    _ensure_pillow()

    if img.mode != COLOR_MODE:
        img = img.convert(COLOR_MODE)

    return img.tobytes()


def bytes_to_image(data: bytes, size: Tuple[int, int]) -> "Image.Image":
    """Convert raw bytes back into a PIL Image with the given size.

    Parameters
    ----------
    data : bytes
        Flat byte data.
    size : (int, int)
        (width, height) of the target image.

    Returns
    -------
    Image.Image
        Reconstructed image in COLOR_MODE.
    """
    _ensure_pillow()

    width, height = size
    expected_len = width * height
    if len(data) != expected_len:
        raise ValueError(
            f"Expected {expected_len} bytes for image of size {size}, "
            f"got {len(data)}."
        )

    return Image.frombytes(COLOR_MODE, size, data)


def generate_resized_versions(
    img: "Image.Image",
    target_sizes: List[int],
) -> List["Image.Image"]:
    """Given a source image, generate smaller square versions.

    Parameters
    ----------
    img : Image.Image
        Original image (e.g., 2048x2048).
    target_sizes : list[int]
        List of side lengths for square images (e.g., [256, 512]).

    Returns
    -------
    list[Image.Image]
        List of resized images, one per size in target_sizes.
    """
    _ensure_pillow()

    if img.mode != COLOR_MODE:
        img = img.convert(COLOR_MODE)

    resized: List["Image.Image"] = []
    for dim in target_sizes:
        if dim <= 0:
            raise ValueError("All target_sizes must be positive integers.")
        # NEAREST keeps a crisp look; you can switch to BILINEAR/BICUBIC if needed.
        resized_img = img.resize((dim, dim), resample=Image.NEAREST)
        resized.append(resized_img)

    return resized


# --- Optional convenience helpers for experiments ---------------------------

def load_all_original_images() -> Dict[str, "Image.Image"]:
    """Load all known original images defined in IMAGE_PATHS.

    Returns
    -------
    dict[str, Image.Image]
        Mapping from image name (e.g., 'houses') to loaded Image object.
    """
    return {name: load_image(path) for name, path in IMAGE_PATHS.items()}


def prepare_image_bytes_for_sizes(
    target_sizes: List[int],
) -> Dict[int, Dict[str, bytes]]:
    """Prepare flattened byte arrays for each image and each size.

    This is a small helper that the experiments module can use if desired.

    Parameters
    ----------
    target_sizes : list[int]
        Side lengths for square resized images.

    Returns
    -------
    dict[int, dict[str, bytes]]
        A nested mapping: size -> image_name -> byte_data.
        For example:
            result[256]["houses"]  -> bytes of houses at 256x256
            result[512]["birthday"] -> bytes of birthday at 512x512
    """
    originals = load_all_original_images()

    data_by_size: Dict[int, Dict[str, bytes]] = {s: {} for s in target_sizes}
    for name, img in originals.items():
        resized_list = generate_resized_versions(img, target_sizes)
        for dim, resized_img in zip(target_sizes, resized_list):
            data_by_size[dim][name] = image_to_bytes(resized_img)

    return data_by_size
