"""Shared absolute-RoPE math helpers (no consumer policy)."""

from __future__ import annotations

import enum
import threading
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_ABS_ROPE_NUM_TILES = 4


class RopeAxis(enum.Enum):
    """An axis of the channels-last ``(B, T, H, W, ...)`` volume abs-RoPE rotates."""

    T = "t"
    H = "h"
    W = "w"


@dataclass(frozen=True)
class RopePositions:
    """The integer positions ``offset + arange(extent)`` on ``axis``, named by value for the table cache."""

    axis: RopeAxis
    extent: int
    offset: int = 0


class AbsRopeTableCache:
    """The abs-RoPE inputs one decode scope reuses: cos/sin tables by key and inverse frequencies by configuration.
    Opened by :func:`abs_rope_table_cache`, which hands it to the decoder's attention modules for the scope.
    Only the thread that opened it reads it, so every table is used on the stream order it was built in. Keys
    name positions by value (:class:`RopePositions`) and inverse frequencies by this cache's own tensors, so a
    cached table is exactly the one a fresh build would produce.
    """

    def __init__(self) -> None:
        self._owner = threading.get_ident()
        self._tables: dict[Hashable, tuple[torch.Tensor, ...]] = {}
        self._inv_freqs: dict[tuple[int, float, torch.dtype, torch.device], torch.Tensor] = {}

    def usable(self) -> bool:
        """Whether this thread may read the cache: the opening thread, outside CUDA graph capture. Traced code
        never reaches the cache (see ``det_attn_rope._rope_config``)."""
        if threading.get_ident() != self._owner:
            return False
        return not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def tables(self, key: Hashable, build: Callable[[], tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        """``build()``, kept under ``key`` for the rest of the scope."""
        tables = self._tables.get(key)
        if tables is None:
            tables = build()
            self._tables[key] = tables
        return tables

    def inv_freqs(self, dim: int, base: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """:func:`rope_inv_freqs` as ``dtype`` on ``device``, one tensor per configuration for the scope.
        Attention blocks of one configuration rotate with this tensor instead of their own equal-valued
        buffers, so table keys (which name the frequencies by ``data_ptr``) match across blocks.
        """
        key = (dim, base, dtype, device)
        inv = self._inv_freqs.get(key)
        if inv is None:
            inv = rope_inv_freqs(dim, base).to(device=device, dtype=dtype)
            self._inv_freqs[key] = inv
        return inv


@contextmanager
def abs_rope_table_cache(module: object) -> Iterator[AbsRopeTableCache]:
    """Give ``module``'s attention blocks (submodules with a ``rope_table_cache`` slot) one
    :class:`AbsRopeTableCache` for the scope, so same-extent tiles and blocks build their tables once.
    A decoder that is not an ``nn.Module`` has no such blocks and gets a cache nothing reads.
    Re-entrant on one thread (an inner scope uses the outer cache); the cache is dropped when the scope
    that opened it exits. Blocks already holding another thread's cache keep it, and this thread builds
    its tables per call there.
    """
    modules = module.modules() if isinstance(module, torch.nn.Module) else ()
    holders = [m for m in modules if hasattr(m, "rope_table_cache")]
    for holder in holders:
        if holder.rope_table_cache is not None and holder.rope_table_cache.usable():
            yield holder.rope_table_cache
            return
    cache = AbsRopeTableCache()
    claimed = [m for m in holders if m.rope_table_cache is None]
    for holder in claimed:
        holder.rope_table_cache = cache
    try:
        yield cache
    finally:
        for holder in claimed:
            holder.rope_table_cache = None


def cached_rope_tables(
    cache: AbsRopeTableCache | None, key: Hashable | None, build: Callable[[], tuple[torch.Tensor, ...]]
) -> tuple[torch.Tensor, ...]:
    """``build()``, reused under ``key`` from ``cache``; always rebuilt without a cache or a key."""
    if cache is None or key is None:
        return build()
    return cache.tables(key, build)


def t_positions(t: int, device: torch.device) -> torch.Tensor:
    return torch.arange(t, dtype=torch.float32, device=device)


def h_positions(h: int, device: torch.device) -> torch.Tensor:
    return torch.arange(h, dtype=torch.float32, device=device)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Default split of head_dim across (T, H, W) RoPE chunks."""
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be a multiple of 8 for default split"
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    assert d_t > 0
    assert d_hw > 0
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Inverse RoPE frequencies: ``1 / base**(i/dim)`` for ``i`` in ``[0, dim, 2)``."""
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    inv_freqs = 1.0 / np.power(float(base), exponents)
    return torch.from_numpy(inv_freqs).to(torch.float32)


def rot_abs_axis_impl(
    xc: torch.Tensor,
    pos: torch.Tensor,
    inv: torch.Tensor,
    axis: int,
    *,
    compute_dtype: torch.dtype,
    pos_key: RopePositions | None = None,
    cache: AbsRopeTableCache | None = None,
) -> torch.Tensor:
    """Absolute RoPE on one axis chunk ``xc[..., D]`` (D even) → new tensor.
    ``pos_key`` names ``pos`` by value, so the tables come from ``cache`` when one is given.
    """
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(compute_dtype)
    xo = pairs[..., 1].to(compute_dtype)

    def build() -> tuple[torch.Tensor, torch.Tensor]:
        shape = [1, 1, 1, 1, 1, inv.shape[0]]
        shape[axis] = pos.shape[0]
        ang = (pos[:, None] * inv[None, :]).reshape(shape)
        return ang.cos().to(compute_dtype), ang.sin().to(compute_dtype)

    key = None if pos_key is None else ("axis", axis, pos_key, compute_dtype, inv.data_ptr(), inv.numel(), pos.device)
    c, s = cached_rope_tables(cache, key, build)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out
