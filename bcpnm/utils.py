"""Small utilities: random number generation, state encoding, softmax, environment capture."""
from __future__ import annotations

import platform
import sys
from typing import Iterable, Sequence

import numpy as np


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a NumPy ``Generator``.

    Passing an integer gives a reproducible stream; passing an existing generator
    returns it unchanged so that functions can share one stream.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def softmax(u: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    u = np.asarray(u, dtype=float)
    m = np.max(u, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(u - m)
    return e / e.sum(axis=axis, keepdims=True)


def sample_categorical(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one index per row from row-wise probability vectors ``p`` (shape R x S)."""
    p = np.asarray(p, dtype=float)
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0
    u = rng.random((p.shape[0], 1))
    return (u > c).sum(axis=1)


def radices(sizes: Sequence[int]) -> np.ndarray:
    """Place values for mixed-radix encoding of states (last node varies fastest)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    r = np.ones_like(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        r[i] = r[i + 1] * sizes[i + 1]
    return r


def encode_indices(idx: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Encode per-node state indices (R x N) into single integers (R,)."""
    idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
    return idx @ radices(sizes)


def decode_indices(codes: np.ndarray | int, sizes: Sequence[int]) -> np.ndarray:
    """Decode integers into per-node state indices (R x N)."""
    codes = np.atleast_1d(np.asarray(codes, dtype=np.int64))
    sizes = np.asarray(sizes, dtype=np.int64)
    out = np.empty((codes.size, sizes.size), dtype=np.int64)
    rem = codes.copy()
    r = radices(sizes)
    for i in range(sizes.size):
        out[:, i] = rem // r[i]
        rem = rem % r[i]
    return out


def environment_report(packages: Iterable[str] = ("numpy", "scipy", "networkx", "matplotlib", "yaml")) -> dict:
    """Versions of Python, the operating system and key packages, for reporting."""
    import importlib

    from . import __version__

    rep = {"python": sys.version.split()[0], "platform": platform.platform(), "bcpnm": __version__}
    for name in packages:
        try:
            mod = importlib.import_module(name)
            rep[name] = getattr(mod, "__version__", "unknown")
        except ImportError:  # pragma: no cover - optional dependency missing
            rep[name] = "not installed"
    return rep
