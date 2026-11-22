"""Small shared helpers."""

from __future__ import annotations


def cdiv(a: int, b: int) -> int:
    """Ceiling division for non-negative integers."""
    return -(-a // b)
