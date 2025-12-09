# lzw_hashmap.py

"""
LZW compressor where the dictionary is stored as a Python hashmap (dict).

Responsibilities:
- Manage a dictionary from byte-strings to integer codes
- Implement LZW compress() and decompress() using O(1) average-time lookups
"""

from __future__ import annotations
from typing import Dict, List


class LZWHashmapCompressor:
    """LZW compressor using a dict as the dictionary structure."""

    def __init__(self) -> None:
        # Maps byte-sequences to integer codes
        self.dictionary: Dict[bytes, int] = {}
        # Maps codes to byte-sequences (for decompression)
        self.reverse_dictionary: Dict[int, bytes] = {}
        self.next_code: int = 0

    def initialize_with_single_bytes(self) -> None:
        """Initialize the dictionary with all possible single-byte values."""
        # TODO: fill dictionary and reverse_dictionary
        pass

    def compress(self, data: bytes) -> List[int]:
        """Compress raw data into a list of integer codes."""
        # TODO: implement hashmap-based LZW compression
        return []

    def decompress(self, codes: List[int]) -> bytes:
        """Decompress a list of integer codes back into raw data."""
        # TODO: implement hashmap-based LZW decompression
        return b""
