# lzw_bst.py

"""
LZW compressor where the dictionary is stored as a Binary Search Tree (BST).

Responsibilities:
- Define a BST node structure
- Implement dictionary operations using a BST
- Implement LZW compress() and decompress() functions/classes that use the BST
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class BSTNode:
    """Node used in the BST-based LZW dictionary."""
    key: bytes
    value: int
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None


class LZWBSTDictionary:
    """BST-based dictionary for LZW."""

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
        self.next_code: int = 0

    def initialize_with_single_bytes(self) -> None:
        """Insert all possible single-byte values into the BST."""
        # TODO: implement actual initialization
        pass

    def lookup(self, key: bytes) -> Optional[int]:
        """Find the code corresponding to a key (if present)."""
        # TODO: implement BST search
        return None

    def insert(self, key: bytes) -> int:
        """Insert a new key into the BST and return the assigned code."""
        # TODO: implement BST insert with O(n) worst-case cost
        return -1


class LZWBSTCompressor:
    """LZW compressor using a BST as the dictionary structure."""

    def __init__(self) -> None:
        self.dictionary = LZWBSTDictionary()

    def compress(self, data: bytes) -> List[int]:
        """Compress raw data into a list of integer codes."""
        # TODO: implement LZW compression using BST dictionary
        return []

    def decompress(self, codes: List[int]) -> bytes:
        """Decompress a list of integer codes back into raw data."""
        # TODO: implement LZW decompression
        return b""
