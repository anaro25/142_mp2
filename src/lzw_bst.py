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
from typing import Optional, List, Dict


@dataclass
class BSTNode:
    """Node used in the BST-based LZW dictionary."""
    key: bytes
    value: int
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None


class LZWBSTDictionary:
    """BST-based dictionary for LZW.

    Notes
    -----
    - Keys are byte sequences (bytes), compared lexicographically.
    - Values are integer codes.
    - Insert is implemented as a simple, unbalanced BST insert
      (O(n) worst case), on purpose.
    """

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
        self.next_code: int = 0

    def initialize_with_single_bytes(self) -> None:
        """Insert all possible single-byte values into the BST."""
        self.root = None
        self.next_code = 0
        for i in range(256):
            self.insert(bytes([i]))

    def lookup(self, key: bytes) -> Optional[int]:
        """Find the code corresponding to a key (if present)."""
        node = self.root
        while node is not None:
            if key == node.key:
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def insert(self, key: bytes) -> int:
        """Insert a new key into the BST and return the assigned code.

        If the key already exists, its existing code is returned and the tree
        is not modified.
        """
        # Empty tree
        if self.root is None:
            code = self.next_code
            self.root = BSTNode(key=key, value=code)
            self.next_code += 1
            return code

        node = self.root
        parent: Optional[BSTNode] = None
        went_left = False

        # Search for existing key / insertion point
        while node is not None:
            parent = node
            if key == node.key:
                return node.value  # already present
            elif key < node.key:
                went_left = True
                node = node.left
            else:
                went_left = False
                node = node.right

        # Insert new node
        code = self.next_code
        new_node = BSTNode(key=key, value=code)
        if went_left:
            assert parent is not None
            parent.left = new_node
        else:
            assert parent is not None
            parent.right = new_node

        self.next_code += 1
        return code


class LZWBSTCompressor:
    """LZW compressor using a BST as the dictionary structure."""

    def __init__(self) -> None:
        self.dictionary = LZWBSTDictionary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, data: bytes) -> List[int]:
        """Compress raw data into a list of integer codes."""
        if not data:
            return []

        # Fresh dictionary per compression run
        self.dictionary.initialize_with_single_bytes()

        result: List[int] = []

        # Start with the first byte as the initial "w"
        w = bytes([data[0]])

        for b in data[1:]:
            c = bytes([b])
            wc = w + c
            code_wc = self.dictionary.lookup(wc)
            if code_wc is not None:
                # The extended sequence exists; keep growing w
                w = wc
            else:
                # Output code for w, then add wc to the dictionary
                code_w = self.dictionary.lookup(w)
                if code_w is None:
                    raise ValueError("Internal BST dictionary error: missing sequence.")
                result.append(code_w)
                self.dictionary.insert(wc)
                w = c

        # Output the code for the last w
        code_w = self.dictionary.lookup(w)
        if code_w is None:
            raise ValueError("Internal BST dictionary error at final step.")
        result.append(code_w)

        return result

    def decompress(self, codes: List[int]) -> bytes:
        """Decompress a list of integer codes back into raw data.

        Decompression does not use the BST; instead it reconstructs the
        dictionary from the code stream in the standard LZW way.
        """
        if not codes:
            return b""

        # code -> bytes
        code_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_code = 256

        result = bytearray()

        # First code
        first_code = codes[0]
        if first_code not in code_to_bytes:
            raise ValueError("Invalid LZW code in stream.")
        w = code_to_bytes[first_code]
        result.extend(w)

        for code in codes[1:]:
            if code in code_to_bytes:
                entry = code_to_bytes[code]
            elif code == next_code:
                # Special LZW case: entry = w + first_char(w)
                entry = w + w[:1]
            else:
                raise ValueError("Invalid LZW code encountered during decompression.")

            result.extend(entry)

            # Add new sequence to dictionary
            code_to_bytes[next_code] = w + entry[:1]
            next_code += 1

            w = entry

        return bytes(result)
