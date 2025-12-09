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
        # Maps byte-sequences to integer codes (for compression)
        self.dictionary: Dict[bytes, int] = {}
        # Maps integer codes to byte-sequences (for decompression)
        self.reverse_dictionary: Dict[int, bytes] = {}
        self.next_code: int = 0

    # ------------------------------------------------------------------
    # Dictionary management
    # ------------------------------------------------------------------

    def initialize_with_single_bytes(self) -> None:
        """Initialize the dictionary with all possible single-byte values."""
        self.dictionary.clear()
        self.reverse_dictionary.clear()

        for i in range(256):
            b = bytes([i])
            self.dictionary[b] = i
            self.reverse_dictionary[i] = b

        self.next_code = 256

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, data: bytes) -> List[int]:
        """Compress raw data into a list of integer codes."""
        if not data:
            return []

        # Fresh dictionary per compression run
        self.initialize_with_single_bytes()

        result: List[int] = []

        # Start with the first byte as the initial "w"
        w = bytes([data[0]])

        for b in data[1:]:
            c = bytes([b])
            wc = w + c
            if wc in self.dictionary:
                # The extended sequence exists; keep growing w
                w = wc
            else:
                # Output the code for w, then add wc to the dictionary
                result.append(self.dictionary[w])
                self.dictionary[wc] = self.next_code
                self.reverse_dictionary[self.next_code] = wc
                self.next_code += 1
                w = c

        # Output the code for the last w
        result.append(self.dictionary[w])

        return result

    def decompress(self, codes: List[int]) -> bytes:
        """Decompress a list of integer codes back into raw data."""
        if not codes:
            return b""

        # Fresh dictionary per decompression run (replay dictionary growth)
        self.initialize_with_single_bytes()

        result = bytearray()

        # First code
        first_code = codes[0]
        if first_code not in self.reverse_dictionary:
            raise ValueError("Invalid LZW code in stream.")
        w = self.reverse_dictionary[first_code]
        result.extend(w)

        for code in codes[1:]:
            if code in self.reverse_dictionary:
                entry = self.reverse_dictionary[code]
            elif code == self.next_code:
                # Special LZW case: entry = w + first_char(w)
                entry = w + w[:1]
            else:
                raise ValueError("Invalid LZW code encountered during decompression.")

            result.extend(entry)

            # Add new sequence to dictionary
            new_seq = w + entry[:1]
            self.dictionary[new_seq] = self.next_code
            self.reverse_dictionary[self.next_code] = new_seq
            self.next_code += 1

            w = entry

        return bytes(result)
