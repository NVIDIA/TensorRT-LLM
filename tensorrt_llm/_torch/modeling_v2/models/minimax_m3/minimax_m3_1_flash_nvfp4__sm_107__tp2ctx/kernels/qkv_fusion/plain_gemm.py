# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain MXFP8 x MXFP8 -> bf16 GEMM on the fused QKV kernel's persistent CLC mainloop.

``out[m, n] = bf16_rn(alpha * sum_k act[m, k] * weight[n, k])`` with E4M3
operands and UE8M0 K32 block scales in the swizzled layout of
``torch.ops.trtllm.mxfp8_quantize(x, True)`` (the layout ``MXFP8LinearMethod``
stores for ``weight_scale``), FP32 accumulation.

This is the ``debug_gemm_out=True`` handle of ``M3QkvGemmNormRopeKvInsert``
with N mapped onto its head slots: ``n = (num_heads_q + 2 * 2 + 2 + 1) * 128``.
In that mode the kernel reads only the four TMA operands (act, act_sf,
weight, weight_sf), ``alpha[0]`` and ``kv_quant_scale[1:3]`` (two scalar loads
that feed only the producer branch), and stores only ``gemm_out``. Every
other producer tensor (Q / index-Q outputs, caches, cache locations, RoPE
table, positions, norm weights) is turned into an address and never
dereferenced, so one set of tiny per-device dummies serves every call and M.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .kernel import HEAD_DIM, M3QkvFusedRunner

NUM_HEADS_KV = 2
NUM_HEADS_INDEX = 2
# K, V, index-q head slots plus the single index-K slot of the head map.
NUM_FIXED_SLOTS = 2 * NUM_HEADS_KV + NUM_HEADS_INDEX + 1
SF_VEC_SIZE = 32
# Weight-side DLPack arguments kept per runner; the cache holds the tensors, so a key's pointer cannot be reused.
_MAX_CACHED_WEIGHTS = 1024


def num_heads_q_for(n: int) -> int:
    """The head-slot count that makes the fused kernel's packed N equal ``n``."""
    if n % HEAD_DIM:
        raise ValueError(f"N={n} is not a multiple of {HEAD_DIM}")
    num_heads_q = n // HEAD_DIM - NUM_FIXED_SLOTS
    if num_heads_q < 1:
        raise ValueError(f"N={n} is too small for the head-slot map")
    return num_heads_q


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def swizzled_sf_numel(rows: int, k: int) -> int:
    """Bytes of a swizzled UE8M0 K32 scale tensor (rows padded to 128, K blocks to 4)."""
    return _pad_up(rows, 128) * _pad_up(k // SF_VEC_SIZE, 4)


def _dl(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=align, enable_tvm_ffi=True)


_DUMMY_ARGS: Dict[torch.device, Tuple] = {}


def _dummy_producer_args(device: torch.device) -> Tuple:
    """DLPack arguments for the producer slots the GEMM-only handle never dereferences.

    Order: q_out, index_q_out, kv_data_cache, kv_scale_cache, index_k_cache,
    out_cache_loc, kv_quant_scale, rotary_cos_sin, position_ids, norm_weights.
    ``kv_quant_scale`` holds three finite fp32 values since the kernel loads
    [1] and [2] ahead of the (compiled-out) producer branch. ``norm_weights``
    has the 4 x 128 extent the compiled signature fixes.
    """
    args = _DUMMY_ARGS.get(device)
    if args is None:
        u8 = torch.zeros(16, dtype=torch.uint8, device=device)
        i32 = torch.zeros(4, dtype=torch.int32, device=device)
        f32 = torch.ones(4, dtype=torch.float32, device=device)
        nw = torch.zeros(4 * HEAD_DIM, dtype=torch.bfloat16, device=device)
        args = (
            _dl(u8),
            _dl(u8),
            _dl(u8),
            _dl(u8, 8),
            _dl(u8),
            _dl(i32, 4),
            _dl(f32, 4),
            _dl(f32),
            _dl(i32, 4),
            _dl(nw),
            (u8, i32, f32, nw),  # keeps the storage alive
        )
        _DUMMY_ARGS[device] = args
    return args[:10]


# Cache strides (9), num_pages, eps: unused by the GEMM-only handle.
_TAIL_ARGS = (0,) * 9 + (0, 0.0)


class M3ClcMxfp8Gemm:
    """One compiled GEMM-only handle for a fixed N (any M, any K that is a multiple of 128)."""

    def __init__(self, n: int, heads_per_tile: int = 4, **cfg):
        self.n = int(n)
        self.heads_per_tile = int(heads_per_tile)
        self.runner = M3QkvFusedRunner(
            num_heads_q=num_heads_q_for(self.n),
            num_heads_kv=NUM_HEADS_KV,
            num_heads_index=NUM_HEADS_INDEX,
            heads_per_tile=self.heads_per_tile,
            debug_gemm_out=True,
            **cfg,
        )
        assert self.runner.op.n_total == self.n
        self._weight_args: Dict[Tuple, Tuple] = {}

    def check(self, act, act_sf, weight, weight_sf, alpha) -> str:
        """Why these tensors do not fit the kernel's contract, or "" when they do."""
        if weight.dim() != 2 or weight.shape[0] != self.n:
            return f"weight shape {tuple(weight.shape)} is not [{self.n}, K]"
        k = weight.shape[1]
        if act.dim() != 2 or act.shape[1] != k:
            return f"activation shape {tuple(act.shape)} does not match K={k}"
        if k % 128:
            return f"K={k} is not a multiple of 128"
        if act.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
            return f"operand dtypes {act.dtype}/{weight.dtype} are not float8_e4m3fn"
        if act_sf.dtype not in (torch.uint8, torch.float8_e8m0fnu) or weight_sf.dtype not in (
            torch.uint8,
            torch.float8_e8m0fnu,
        ):
            return f"scale dtypes {act_sf.dtype}/{weight_sf.dtype} are not UE8M0 bytes"
        if not (
            act.is_contiguous()
            and weight.is_contiguous()
            and act_sf.is_contiguous()
            and weight_sf.is_contiguous()
        ):
            return "non-contiguous operand or scale"
        need = swizzled_sf_numel(act.shape[0], k)
        if act_sf.numel() < need:
            return (
                f"activation scales hold {act_sf.numel()} bytes, the swizzled layout needs {need}"
            )
        need = swizzled_sf_numel(self.n, k)
        if weight_sf.numel() < need:
            return f"weight scales hold {weight_sf.numel()} bytes, the swizzled layout needs {need}"
        if alpha.dtype != torch.float32 or alpha.numel() != 1 or alpha.dim() != 1:
            return "alpha is not a 1-element fp32 vector"
        for name, t, al in (
            ("act", act, 16),
            ("act_sf", act_sf, 16),
            ("weight", weight, 16),
            ("weight_sf", weight_sf, 16),
            ("alpha", alpha, 4),
        ):
            if t.data_ptr() % al:
                return f"{name} is not {al}-byte aligned"
        devices = {t.device for t in (act, act_sf, weight, weight_sf, alpha)}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return "operands are not on one CUDA device"
        return ""

    def _weight_dl(self, weight, weight_sf, alpha) -> Tuple:
        key = (
            weight.data_ptr(),
            tuple(weight.shape),
            weight_sf.data_ptr(),
            weight_sf.numel(),
            alpha.data_ptr(),
        )
        cached = self._weight_args.get(key)
        if cached is None:
            if len(self._weight_args) >= _MAX_CACHED_WEIGHTS:
                self._weight_args.clear()
            cached = (
                _dl(weight),
                _dl(weight_sf.view(torch.float8_e8m0fnu).reshape(-1)),
                _dl(alpha, 4),
                (weight, weight_sf, alpha),
            )
            self._weight_args[key] = cached
        return cached[:3]

    def make_args(self, act, act_sf, weight, weight_sf, alpha, gemm_out) -> Tuple:
        """TVM-FFI arguments for one call; callers validate with ``check`` first."""
        assert gemm_out.shape == (act.shape[0], self.n) and gemm_out.dtype == torch.bfloat16
        assert gemm_out.is_contiguous() and gemm_out.data_ptr() % 16 == 0
        w_dl, wsf_dl, alpha_dl = self._weight_dl(weight, weight_sf, alpha)
        return (
            _dl(act),
            _dl(act_sf.view(torch.float8_e8m0fnu).reshape(-1)),
            w_dl,
            wsf_dl,
            alpha_dl,
            *_dummy_producer_args(act.device),
            _dl(gemm_out.view(-1)),
            *_TAIL_ARGS,
        )

    def run_args(self, args, stream=None) -> None:
        self.runner.run_args(args, stream)

    def __call__(self, act, act_sf, weight, weight_sf, alpha, out=None) -> torch.Tensor:
        reason = self.check(act, act_sf, weight, weight_sf, alpha)
        assert not reason, reason
        if out is None:
            out = torch.empty((act.shape[0], self.n), dtype=torch.bfloat16, device=act.device)
        self.run_args(self.make_args(act, act_sf, weight, weight_sf, alpha, out))
        return out


_GEMMS: Dict[Tuple[int, int], M3ClcMxfp8Gemm] = {}


def get_gemm(n: int, heads_per_tile: int) -> M3ClcMxfp8Gemm:
    """The cached handle for (N, heads per tile); compiles on first use."""
    key = (int(n), int(heads_per_tile))
    gemm = _GEMMS.get(key)
    if gemm is None:
        gemm = M3ClcMxfp8Gemm(n, heads_per_tile=heads_per_tile)
        _GEMMS[key] = gemm
    return gemm


def m3_clc_mxfp8_gemm(act, act_sf, weight, weight_sf, alpha, out=None, heads_per_tile: int = 4):
    """``bf16_rn(alpha * act @ weight.T)`` for MXFP8 operands with swizzled UE8M0 K32 scales."""
    return get_gemm(weight.shape[0], heads_per_tile)(act, act_sf, weight, weight_sf, alpha, out)
