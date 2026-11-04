# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The megakernel's QKV stage of one decoder layer on one rank: ``qkv = x_attn @ W_qkv^T``, ``[8, 8192] x [2304,
8192]^T`` BF16, for the 8 verify rows of a K=7 step.

The weight is repacked once (``pack_bf16``) into the MMA fragment order ``[144 tiles][128 chunks][2048 B]``; the row
order is the checkpoint's (``[q_0 | g_0 | ... | q_15 | g_15 | k_0 | k_1 | v_0 | v_1]``, 64 rows each), so every
``QkvSum`` unit (64 rows) holds one head. Two input modes:

* mode A (the attention-half contract): ``x`` = the normed BF16 attention input the previous layer's boundary
  wrote (``bf16(c * rsqrt(mean(c^2) + eps) * (w + 1))``, ``c = bf16(residual)``); one BF16 rounding of the FP32 sum.
* mode B (``FLAG_X_PRIME``): ``res`` = the FP32 residual; the run CTAs form ``x' = bf16(bf16(r) * (1 + w))``, the
  ``XSsq`` units the per-token sums of ``bf16(r)^2``, and ``QkvSum`` rounds ``bf16(sum * rsqrt(S / 8192 + eps))``
  once.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from . import library, schedule

#: The pre-QKV norm's epsilon (the checkpoint's pre-QKV Gemma norm).
PRE_QKV_EPS = 1e-5


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """``[M, K]`` BF16 -> ``[M/16, K/64, 2048]`` bytes: per (tile, chunk) pieces p = (row half, element half) of 32
    threads x 16 B; thread (g, c) holds row g's (pieces 0-1) and row g + 8's (2-3) elements ``[16 c, +16)`` of the
    chunk."""
    m, k = w.shape
    assert w.dtype == torch.bfloat16 and m % 16 == 0 and k % 64 == 0
    v = w.contiguous().view(m // 16, 2, 8, k // 64, 4, 2, 8)  # tile, rh, g, q, c, eh, e8
    v = v.permute(0, 3, 1, 5, 2, 4, 6)  # tile, q, rh, eh, g, c, e8
    return v.contiguous().view(torch.uint8).reshape(m // 16, k // 64, 2048)


def unpack_bf16(packed: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """The inverse of ``pack_bf16``."""
    v = (
        packed.contiguous().view(torch.bfloat16).view(m // 16, k // 64, 2, 2, 8, 4, 8)
    )  # tile, q, rh, eh, g, c, e8
    return v.permute(0, 2, 4, 1, 5, 3, 6).contiguous().view(m, k)


def pack_qkv(w: torch.Tensor, slices: int = 1) -> torch.Tensor:
    """The QKV weight ``[2304, 8192]`` BF16 in the kernel's packed order: ``pack_bf16`` blocks as ``[144 tiles]
    [128 chunks]`` (``slices`` = 1) or ``[4 K slices][144 tiles][32 chunks]`` (``slices`` = 4); ``[144, 128, 2048]``
    bytes."""
    packed = pack_bf16(w)
    if slices == 1:
        return packed
    L = schedule.seg_chunks(slices)
    return (
        packed.view(schedule.QKV_TILES, slices, L, 2048)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(packed.shape)
    )


def qkv_tables(ctas: int, x_prime: bool, slices: int = 1) -> schedule.Tables:
    """The QKV stage's role tables."""
    plan = schedule.plan_qkv(ctas, x_prime=x_prime, slices=slices)
    return schedule.build_tables(plan, lambda st, u: schedule.unit_bytes_qkv(st, u, ctas))


def qkv_fields(
    packed: torch.Tensor, gamma: Optional[torch.Tensor] = None, eps: float = PRE_QKV_EPS
) -> Dict[str, int]:
    """Parameter-block fields of one layer's QKV stage (``packed`` = ``pack_bf16(W_qkv)``, ``gamma`` = the
    pre-QKV norm weight BF16 ``[8192]`` for mode B)."""
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (
        schedule.QKV_TILES,
        schedule.CHUNKS_PER_ROW,
        2048,
    )
    fields = dict(
        WQkv=packed.data_ptr(), Eps=int(torch.tensor(eps, dtype=torch.float32).view(torch.int32))
    )
    if gamma is not None:
        assert (
            gamma.dtype == torch.bfloat16
            and gamma.numel() == schedule.HIDDEN
            and gamma.is_contiguous()
        )
        fields["GammaQkv"] = gamma.data_ptr()
    return fields


class QkvStage:
    """The QKV stage on its own: one ``Megakernel`` of ``qkv_tables`` and per layer its parameter blocks.

    Args:
        mode: the library's compile mode (``check`` or ``lean``).
        x_prime: mode B (FP32 residual input, 1 / rms after the GEMV).
        copies: arena copies of the shared workspace.
    """

    def __init__(
        self,
        mode: str = "check",
        x_prime: bool = False,
        copies: int = 3,
        workspace: Optional[library.Workspace] = None,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        self.ws = workspace or library.Workspace(self.lib, copies)
        self.x_prime = x_prime
        self.slices = self.lib.qkv_slices
        self.tables = qkv_tables(self.lib.ctas, x_prime, self.slices)
        self.mk = library.Megakernel(self.lib, self.ws, self.tables, timeout_ms=timeout_ms)

    def make_layer(
        self,
        w: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        check: bool = False,
        stamps: bool = False,
        evict_first: bool = True,
        one_piece: bool = False,
        packed: Optional[torch.Tensor] = None,
    ):
        """One layer's parameter blocks from its QKV weight ``w`` BF16 ``[2304, 8192]`` (packed here in this build's
        order), or from ``packed`` = ``pack_qkv(w, self.slices)``; ``gamma`` = the pre-QKV norm weight (mode B)."""
        if packed is None:
            packed = pack_qkv(w, self.slices)
        flags = (
            (library.FLAG_EVICT_FIRST if evict_first else 0)
            | (library.FLAG_ONE_PIECE if one_piece else 0)
            | (library.FLAG_X_PRIME if self.x_prime else 0)
            | (library.FLAG_CHECK if check else 0)
        )
        assert (gamma is not None) == self.x_prime, "mode B needs the norm weight"
        keep = (packed,) if gamma is None else (packed, gamma)
        return self.mk.make_layer(qkv_fields(packed, gamma), flags=flags, stamps=stamps, keep=keep)

    def __call__(
        self,
        layer,
        x: Optional[torch.Tensor] = None,
        res: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        out_f32: Optional[torch.Tensor] = None,
        pdl: bool = True,
    ):
        """``qkv [8, 2304]`` BF16 of the layer (into ``out`` if given); ``x`` BF16 ``[8, 8192]`` (mode A) or ``res``
        FP32 ``[8, 8192]`` (mode B); ``out_f32`` receives the FP32 sums when the layer was made with ``check``."""
        if self.x_prime:
            assert (
                res is not None
                and res.dtype == torch.float32
                and tuple(res.shape) == (8, schedule.HIDDEN)
            )
            assert res.is_contiguous()
        else:
            assert (
                x is not None
                and x.dtype == torch.bfloat16
                and tuple(x.shape) == (8, schedule.HIDDEN)
            )
            assert x.is_contiguous()
        dev = (x if x is not None else res).device
        if out is None:
            out = torch.empty(8, schedule.QKV_ROWS, dtype=torch.bfloat16, device=dev)
        io = library.KernelIo(x=x, res=res, qkv=out, qkv_f32=out_f32)
        self.mk.launch(layer, io, pdl=pdl)
        return out
