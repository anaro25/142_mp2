# rle.py

"""
Run-Length Encoding (RLE) compressor.

Responsibilities:
- Implement a simple RLE scheme for comparison with LZW
- Encode a byte sequence to (value, run_length) pairs
- Decode such pairs back into raw bytes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class RLEToken:
    """Represents a single (value, run_length) pair."""
    value: int
    length: int


class RLECompressor:
    """Simple Run-Length Encoding compressor."""

    def compress(self, data: bytes) -> List[RLEToken]:
        """Compress a sequence of bytes into RLE tokens."""
        # TODO: implement basic RLE compression
        return []

    def decompress(self, tokens: List[RLEToken]) -> bytes:
        """Decompress RLE tokens back into raw bytes."""
        # TODO: implement basic RLE decompression
        return b""
