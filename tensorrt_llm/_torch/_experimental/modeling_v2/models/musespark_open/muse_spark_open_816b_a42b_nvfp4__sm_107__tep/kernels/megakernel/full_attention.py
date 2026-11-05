# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The megakernel's full-attention stages of one full-attention (NoPE) layer on one rank: the stock
``full_attention_prep_fold`` + ``combine_norm_gate`` pair as megakernel units (``stages/fullAttention.cuh``).

Input: the raw QKV projection ``qkv`` BF16 ``[8, 2304]`` (checkpoint order, ``Io.qkv``), the layer's FP8 pool view
``[blocks, 2, 32, 64]``, its int32 block table ``[2, >= pages]`` (K row, V row), ``kv_lens`` int32 ``[1]`` (the
context including the step's 8 rows) and the FP32 KV scale. Output: the o_proj input BF16 ``[8, 1024]`` (and, in check
mode, the FP32 merged attention before its BF16 rounding). The step's K/V are appended to the pool with the stock bytes.

Numerics are the stock pair's rule: the same Q/K prep statements, Q as ``Q_hi + Q_lo``,
P in E4M3 against the lazily raised running max, normalized FP16 partials with an FP32 log2-sum-exp, a fixed-order
merge, BF16, the norm-gate. Not bit-identical to the stock (MMA shapes and summation orders differ); the tests gate the
error against the float64 attention (``reference``) at 1.25 x the stock pair's.
"""

from __future__ import annotations

import ctypes
import math
from typing import Dict, Optional, Tuple

import torch

from . import library, schedule_fa

#: The checkpoint's Q / K norm epsilon and query pre-scale (``preattn_prep.CKPT_EPS``, ``CKPT_Q_SCALE``).
EPS = 1e-5
Q_SCALE = 43.7840518911 / 8.0
HEAD_DIM = 64
Q_HEADS = 16
KV_HEADS = 2
ROWS = 8
QKV_WIDTH = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM
OUT_WIDTH = Q_HEADS * HEAD_DIM


def _f32_bits(x: float) -> int:
    return int(torch.tensor(x, dtype=torch.float32).view(torch.int32))


def tensor_map(lib: library.Library, pool: torch.Tensor) -> torch.Tensor:
    """The pool's two TMA tensor maps (256 B) on the pool's device: box 64 x 32 (one head of a block; K) at 0 and box
    64 x 64 (a whole block; V) at 128."""
    assert pool.dtype == torch.float8_e4m3fn and pool.is_contiguous()
    assert tuple(pool.shape[1:]) == (KV_HEADS, 32, HEAD_DIM) and pool.data_ptr() % 16 == 0
    fn = lib.lib.msoMkFaTensorMap
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int]
    host = (ctypes.c_uint8 * 256)()
    for i, rows in enumerate((32, 64)):
        rc = fn(pool.data_ptr(), pool.shape[0], ctypes.addressof(host) + 128 * i, rows)
        assert rc == 0, f"msoMkFaTensorMap: error {rc}"
    out = torch.frombuffer(bytearray(host), dtype=torch.uint8).to(pool.device)
    assert out.data_ptr() % 128 == 0
    return out


def _abi_int(lib: library.Library, name: str) -> int:
    f = getattr(lib.lib, name)
    f.restype = ctypes.c_int
    f.argtypes = []
    return int(f())


class FaStage:
    """The full-attention stages on their own: one ``Megakernel`` of ``schedule_fa.fa_tables`` and per layer its
    parameter blocks.

    Args:
        mode: the library's compile mode (``check``, ``lean`` or a ring variant).
        units: attention units per layer (one per CTA).
        min_pages: fewest pages per keyed unit.
    """

    def __init__(
        self,
        mode: str = "check",
        units: Optional[int] = None,
        min_pages: int = schedule_fa.DEFAULT_MIN_PAGES,
        workspace: Optional[library.Workspace] = None,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        self.ws = workspace or library.Workspace(self.lib)
        self.units = int(units or self.lib.ctas)
        self.min_pages = int(min_pages)
        self.region_bytes = _abi_int(self.lib, "msoMkFaRegionBytes")
        self.ring = _abi_int(self.lib, "msoMkFaRing")
        assert _abi_int(self.lib, "msoMkFaMaxUnitPages") == schedule_fa.MAX_UNIT_PAGES
        self.tables = schedule_fa.fa_tables(self.lib.ctas, self.units, self.region_bytes)
        self.mk = library.Megakernel(self.lib, self.ws, self.tables, timeout_ms=timeout_ms)
        dev = self.ws.arena.device
        n = schedule_fa.ROWS * self.units
        # Single-reader slots: every 32-bit word starts (and is left by its reader) as kEmpty.
        self.part_o = torch.full((n * HEAD_DIM // 2,), -0.0, dtype=torch.float32, device=dev)
        self.part_l = torch.full((n,), -0.0, dtype=torch.float32, device=dev)

    def make_layer(
        self,
        pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_scale: torch.Tensor,
        out: torch.Tensor,
        out_f32: Optional[torch.Tensor] = None,
        entry_pages: Optional[int] = None,
        prefetch_tiles: int = 0,
        check: bool = False,
        stamps: bool = False,
        eps: float = EPS,
        q_scale: float = Q_SCALE,
        dbg_partials: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        iflags: int = 0,
    ):
        """One layer's parameter blocks (eager). ``out`` BF16 ``[8, 1024]`` and ``out_f32`` FP32 ``[8, 1024]`` are
        written by every launch of the layer. ``entry_pages`` (default: the ring) are requested at kernel entry;
        ``prefetch_tiles`` tiles of 4 pages beyond the ring are kept prefetched into L2. ``dbg_partials`` (tests):
        FP16 ``[128, units, 64]`` and FP32 ``[128, units]`` that also receive the keyed units' partials in the stock
        split op's layout."""
        assert (
            block_tables.dtype == torch.int32
            and block_tables.dim() == 2
            and block_tables.shape[0] == 2
        )
        assert block_tables.stride(1) == 1
        assert kv_lens.dtype == torch.int32 and kv_scale.dtype == torch.float32
        assert (
            out.dtype == torch.bfloat16
            and tuple(out.shape) == (ROWS, OUT_WIDTH)
            and out.is_contiguous()
        )
        if out_f32 is not None:
            assert out_f32.dtype == torch.float32 and tuple(out_f32.shape) == (ROWS, OUT_WIDTH)
        tmap = tensor_map(self.lib, pool)
        fields: Dict[str, int] = dict(
            FaTmap=tmap.data_ptr(),
            FaPool=pool.data_ptr(),
            FaTable=block_tables.data_ptr(),
            FaTableStride=block_tables.stride(0),
            FaKvLens=kv_lens.data_ptr(),
            FaKvScale=kv_scale.data_ptr(),
            FaPartO=self.part_o.data_ptr(),
            FaPartL=self.part_l.data_ptr(),
            FaUnits=self.units,
            FaMinPages=self.min_pages,
            FaEntryPages=self.ring if entry_pages is None else int(entry_pages),
            FaQScale=_f32_bits(q_scale),
            FaEps=_f32_bits(eps),
            FaOut=out.data_ptr(),
            FaOutF32=0 if out_f32 is None else out_f32.data_ptr(),
            FaPrefetch=int(prefetch_tiles),
        )
        if (
            iflags
        ):  # kPIFlags bits of the stage internals (e.g. kIFlagF3FaPingPong, kIFlagFaLateLight)
            fields["IFlags"] = int(iflags)
        if dbg_partials is not None:
            dbg_o, dbg_l = dbg_partials
            assert dbg_o.dtype == torch.float16 and tuple(dbg_o.shape) == (
                schedule_fa.ROWS,
                self.units,
                HEAD_DIM,
            )
            assert dbg_l.dtype == torch.float32 and tuple(dbg_l.shape) == (
                schedule_fa.ROWS,
                self.units,
            )
            assert dbg_o.is_contiguous() and dbg_l.is_contiguous()
            fields.update(FaDbgO=dbg_o.data_ptr(), FaDbgL=dbg_l.data_ptr())
        flags = library.FLAG_EVICT_FIRST | (library.FLAG_CHECK if check else 0)
        keep = (tmap, pool, block_tables, kv_lens, kv_scale, out) + (
            () if out_f32 is None else (out_f32,)
        )
        keep = keep + (() if dbg_partials is None else tuple(dbg_partials))
        return self.mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep)

    def __call__(self, layer, qkv: torch.Tensor, pdl: bool = True) -> None:
        """Launch the layer on the raw projection ``qkv`` BF16 ``[8, 2304]``."""
        assert (
            qkv.dtype == torch.bfloat16
            and tuple(qkv.shape) == (ROWS, QKV_WIDTH)
            and qkv.is_contiguous()
        )
        self.mk.launch(layer, library.KernelIo(qkv=qkv), pdl=pdl)


# --- references (float64) ---------------------------------------------------------------------------------------


def gate_of(qkv: torch.Tensor) -> torch.Tensor:
    """The output gate ``[8, 16, 64]`` of the raw projection (checkpoint order)."""
    return qkv[:, : 2 * Q_HEADS * HEAD_DIM].view(qkv.shape[0], Q_HEADS, 2, HEAD_DIM)[:, :, 1, :]


def reference(
    qkv: torch.Tensor,
    pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_len: int,
    kv_scale: float = 1.0,
    eps: float = EPS,
    q_scale: float = Q_SCALE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(attention, gated)`` in float64: the attention ``[8, 16, 64]`` of the float64-normalized query over the
    cache bytes of ``pool`` (which must hold the step's appended K/V), and the norm-gated output ``[8, 1024]`` of it
    (no intermediate rounding). The cache bytes are the model's storage: both the stock pair and the megakernel read
    the same ones."""
    rows = qkv.shape[0]
    q = qkv[:, : 2 * Q_HEADS * HEAD_DIM].view(rows, Q_HEADS, 2, HEAD_DIM)[:, :, 0, :].double()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * q_scale
    keys = torch.arange(seq_len, device=qkv.device)
    raw = pool.view(torch.uint8)
    out = torch.empty(rows, Q_HEADS, HEAD_DIM, dtype=torch.float64, device=qkv.device)
    past = seq_len - rows
    for h in range(KV_HEADS):
        k = (
            raw[block_tables[0, keys // 32].long(), h, keys % 32].view(torch.float8_e4m3fn).double()
            / kv_scale
        )
        v = (
            raw[block_tables[1, keys // 32].long(), h, keys % 32].view(torch.float8_e4m3fn).double()
            / kv_scale
        )
        for t in range(rows):
            end = past + t + 1
            s = q[t, 8 * h : 8 * h + 8] @ k[:end].T / math.sqrt(HEAD_DIM)
            out[t, 8 * h : 8 * h + 8] = torch.softmax(s, dim=-1) @ v[:end]
        del k, v
    g = gate_of(qkv).double()
    gated = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps) * torch.sigmoid(g)
    return out, gated.reshape(rows, OUT_WIDTH)
