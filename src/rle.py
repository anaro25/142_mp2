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
        """Compress a sequence of bytes into RLE tokens.

        Each token stores a byte value and the length of its run.
        For simplicity, run lengths are capped at 255 so that a token
        can be thought of as (1 byte value, 1 byte length).
        """
        if not data:
            return []

        tokens: List[RLEToken] = []

        current_value = data[0]
        run_length = 1

        for b in data[1:]:
            if b == current_value and run_length < 255:
                run_length += 1
            else:
                tokens.append(RLEToken(value=current_value, length=run_length))
                current_value = b
                run_length = 1

        # Flush the final run
        tokens.append(RLEToken(value=current_value, length=run_length))

        return tokens

    def decompress(self, tokens: List[RLEToken]) -> bytes:
        """Decompress RLE tokens back into raw bytes."""
        if not tokens:
            return b""

        out = bytearray()

        for token in tokens:
            if token.length < 0:
                raise ValueError("RLEToken has negative run length.")
            if not (0 <= token.value <= 255):
                raise ValueError("RLEToken value is out of byte range (0–255).")

            # Repeat the value 'length' times
            out.extend([token.value] * token.length)

        return bytes(out)
