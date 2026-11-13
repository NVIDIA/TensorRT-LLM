# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's CuTe DSL MXFP8 GEMM (``Sm100BlockScaledPersistentDenseGemmKernel``) writing a row-strided output.

DHEAD_MX4 (``dspark_draft_head``) computes the drafter's vocab-shard logits ``x [M, K] @ W[:N]^T`` straight into the S4
padded logits buffer, whose rows are ``ldc`` > N elements apart and whose columns N..ldc-1 hold -inf for the Markov
chain. FlashInfer's own entry (``gemm_base._cute_dsl_gemm_mxfp8_runner``) builds a compact output layout (row stride
N), so it cannot address that buffer. This module runs the same kernel class at the same tactic through its own host
wrapper, ``wrapper_ldc``, which differs from FlashInfer's ``wrapper`` in one place: the C tensor is the logical
[M, N] window of a row-major [rows, ldc] buffer (row stride ``ldc`` read from the buffer). The kernel stores C with a
TMA tensor store bounded by that logical shape, so columns N..ldc-1 (and rows past M) are never written: the GEMM runs
at the logical N with the store predicated by the TMA bounds, and the buffer's -inf tail survives every call.

Operands are FlashInfer's MXFP8 ones: e4m3 codes row-major [rows, K] and UE8M0 scales in the 128x4-swizzled layout
with rows padded to 128 (``torch.ops.trtllm.mxfp8_quantize(x, True)``). The weight's codes are passed as its logical
[N, K] rows (TMA zero-fills the last tile's rows past N); its scales cover ceil(N / 128) * 128 rows.

A tactic is FlashInfer's (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, split_k) with split_k 1 (the
split-K kernel needs a contiguous output). Compiled once per (tactic, PDL, max active clusters) through
``cute_aot`` (TVM-FFI), outside capture; a forward never compiles. The AOT key's ``sources`` are those of every other
CuTe DSL kernel of this target (the target package and the shared Blackwell helpers), so all of a run's compiles carry
one source fingerprint; FlashInfer's kernel files, whose code the kernel body is traced from, enter the key as the
``flashinfer_kernel`` argument (a digest of their contents) instead.
"""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot

Tactic = Tuple[Tuple[int, int], Tuple[int, int], bool, bool, int]

SF_VEC = 32  # columns per UE8M0 scale (MXFP8)
SF_ROW_TILE = 128  # rows per swizzled scale tile


def _kernel_base():
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    return Sm100BlockScaledPersistentDenseGemmKernel


@functools.lru_cache(maxsize=None)
def _kernel_class():
    """FlashInfer's kernel class with the row-strided host wrapper (built once, on first use)."""
    base = _kernel_base()

    class StridedCKernel(base):
        @cute.jit
        def wrapper_ldc(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            sf_m: cutlass.Int64,
            sf_n: cutlass.Int64,
            sf_k: cutlass.Int64,
            num_l: cutlass.Constexpr,
            a_sf_ptr: cute.Pointer,
            b_sf_ptr: cute.Pointer,
            alpha_tensor: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            current_stream,
            swap_ab: cutlass.Constexpr = False,
            epilogue_op: cutlass.Constexpr = lambda x: x,
        ):
            """FlashInfer's ``wrapper`` (MXFP8 path) with C = the logical window of the row-major buffer ``mC``
            [rows, ldc]: swap_ab writes C^T's (m = weight rows, n = activation rows) at strides (1, ldc), else C's
            (m = activation rows, n = weight rows) at (ldc, 1)."""
            m = cute.size(mA, mode=[0])
            k = cute.size(mA, mode=[1])
            n = cute.size(mB, mode=[0])
            ldc = cute.size(mC, mode=[1])
            a_tensor = cute.make_tensor(
                mA.iterator, layout=cute.make_ordered_layout((m, k, num_l), order=(1, 0, 2))
            )
            b_tensor = cute.make_tensor(
                mB.iterator, layout=cute.make_ordered_layout((n, k, num_l), order=(1, 0, 2))
            )
            if cutlass.const_expr(swap_ab):
                c_tensor = cute.make_tensor(
                    mC.iterator, layout=cute.make_layout((m, n, num_l), stride=(1, ldc, ldc * n))
                )
            else:
                c_tensor = cute.make_tensor(
                    mC.iterator, layout=cute.make_layout((m, n, num_l), stride=(ldc, 1, ldc * m))
                )
            sfa_tensor = cute.make_tensor(
                a_sf_ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, sf_m, 4, sf_k, num_l), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            sfb_tensor = cute.make_tensor(
                b_sf_ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, sf_n, 4, sf_k, num_l), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
            )

    return StridedCKernel


def why_invalid(tactic, n: int, m: int, k: int) -> Optional[str]:
    """Why ``tactic`` cannot run an [m, k] x [n, k]^T call into a row-strided output (None: it can)."""
    try:
        (tile_m, tile_n), (cluster_m, cluster_n), swap_ab, use_prefetch, split_k = tactic
        tactic = (
            (int(tile_m), int(tile_n)),
            (int(cluster_m), int(cluster_n)),
            bool(swap_ab),
            bool(use_prefetch),
            int(split_k),
        )
    except (TypeError, ValueError):
        return f"tactic {tactic!r} is not ((tile_m, tile_n), (cluster_m, cluster_n), swap_ab, prefetch, split_k)"
    if tactic[4] != 1:
        return f"tactic {tactic}: split-K needs a contiguous output (split_k must be 1)"
    if m <= 0 or n <= 0:
        return f"empty problem: M {m}, N {n}"
    if k % (4 * SF_VEC):
        return f"K {k} is not a multiple of {4 * SF_VEC}"
    kernel_m, kernel_n = (n, m) if tactic[2] else (m, n)
    ok = _kernel_base().can_implement(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        SF_VEC,
        cutlass.BFloat16,
        tactic[0],
        tactic[1],
        kernel_m,
        kernel_n,
        k,
        1,
        "k",
        "k",
        "m" if tactic[2] else "n",
    )
    return (
        None
        if ok
        else f"tactic {tactic} cannot implement M {m}, N {n}, K {k} (FlashInfer can_implement)"
    )


def tactics(n: int, m: int, k: int) -> List[Tactic]:
    """Every non-split-K tactic FlashInfer lists that runs this shape (the sweep a micro bench times)."""
    from flashinfer.gemm.kernels.utils import (
        _SM100_CLUSTER_SHAPE_MN_CANDIDATES,
        _SM100_MMA_TILER_MN_CANDIDATES,
    )

    out = []
    for tile in _SM100_MMA_TILER_MN_CANDIDATES:
        for cluster in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
            for swap_ab in (True, False):
                for prefetch in (False, True):
                    tactic = (tuple(tile), tuple(cluster), swap_ab, prefetch, 1)
                    if why_invalid(tactic, n, m, k) is None:
                        out.append(tactic)
    return out


def _aot_sources():
    """The ``sources`` of the shipped AOT object: the target package (the wrapper) and the shared Blackwell kernel
    helpers, the same files as every other CuTe DSL kernel of this target (one source fingerprint per run)."""
    target = Path(__file__).resolve().parents[1]
    shared = cute_aot.BLACKWELL_DIR
    return cute_aot.py_files(str(target), str(shared))


@functools.lru_cache(maxsize=None)
def flashinfer_kernel_digest() -> str:
    """A digest of FlashInfer's kernel files the kernel body is traced from (the kernel module and its ``_common``
    sibling): the ``flashinfer_kernel`` argument of the AOT key, so an object compiled from other FlashInfer kernel
    code is not served."""
    kernel = Path(sys.modules[_kernel_base().__module__].__file__).resolve()
    files = [
        f
        for f in (kernel, kernel.with_name("dense_blockscaled_gemm_sm100_common.py"))
        if f.is_file()
    ]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


@cute_aot.cached_compile("m3_draft_head_gemm._compiled", lambda: _aot_sources(), tvm_ffi=True)
def _compiled(tactic: Tactic, enable_pdl: bool, max_active_clusters: int, flashinfer_kernel: str):
    """``flashinfer_kernel`` (``flashinfer_kernel_digest()``) keys the object only; the compile reads the kernel."""
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr

    (tile, cluster, swap_ab, use_prefetch, _) = tactic
    gemm = _kernel_class()(SF_VEC, tile, cluster, use_prefetch, enable_pdl)
    sym_m, sym_n, sym_k, sym_rows, sym_ldc = (cute.sym_int() for _ in range(5))
    a_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
    )
    b_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_n, sym_k), stride_order=(1, 0), assumed_align=16
    )
    c_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_rows, sym_ldc), stride_order=(1, 0), assumed_align=16
    )
    a_sf_ptr = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    return cute.compile(
        gemm.wrapper_ldc,
        a_fake,
        b_fake,
        c_fake,
        1,
        1,
        1,
        1,
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        max_active_clusters,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        swap_ab,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _max_active_clusters(tactic: Tactic) -> int:
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    cluster = tactic[1]
    return int(get_max_active_clusters(cluster[0] * cluster[1]))


class StridedMxfp8Gemm:
    """``out[:M, :N] = bf16(x @ W[:N]^T)`` for MXFP8 x [M, K] and W [>= N, K] into a row-major BF16 ``out`` [>= M, ldc]
    at one pinned tactic; nothing else of ``out`` is written. ``enable_pdl``: launch with programmatic dependent launch
    (the caller passes the process's ``get_env_enable_pdl()``, as the kernels around it do)."""

    def __init__(self, tactic: Tactic, enable_pdl: bool = True) -> None:
        (tile, cluster, swap_ab, use_prefetch, split_k) = tactic
        self.tactic: Tactic = (
            (int(tile[0]), int(tile[1])),
            (int(cluster[0]), int(cluster[1])),
            bool(swap_ab),
            bool(use_prefetch),
            int(split_k),
        )
        self.enable_pdl = bool(enable_pdl)
        self._fn = None
        self._alpha = {}

    def compile(self) -> None:
        """Compile (or load the shipped object of) the kernel; outside CUDA-graph capture."""
        if self._fn is None:
            self._fn = _compiled(
                self.tactic,
                self.enable_pdl,
                _max_active_clusters(self.tactic),
                flashinfer_kernel_digest(),
            )

    def _alpha_for(self, device) -> torch.Tensor:
        alpha = self._alpha.get(device)
        if alpha is None:
            alpha = self._alpha[device] = torch.ones(1, dtype=torch.float32, device=device)
        return alpha

    def __call__(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        wq: torch.Tensor,
        wsf: torch.Tensor,
        n: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """``xq`` e4m3 [M, K] and ``xsf`` its swizzled scales; ``wq`` e4m3 [>= N, K] contiguous and ``wsf`` its
        swizzled scales (rows padded to 128); ``out`` row-major BF16 [>= M, ldc], ldc >= N, 16-byte aligned rows.
        Returns ``out``."""
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("m3_dhead_gemm: compile outside CUDA-graph capture first")
            self.compile()
        m, k = int(xq.shape[0]), int(xq.shape[1])
        n = int(n)
        if out.dim() != 2 or not out.is_contiguous() or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be a contiguous BF16 [rows, ldc] buffer, got {out.dtype} {tuple(out.shape)}"
            )
        if out.shape[0] < m or out.shape[1] < n or (out.shape[1] * 2) % 16:
            raise ValueError(
                f"out {tuple(out.shape)} cannot hold [{m}, {n}] with 16-byte aligned rows"
            )
        weight = wq[:n]
        if self.tactic[2]:
            kernel_a, kernel_b, a_sf, b_sf, kernel_m, kernel_n = weight, xq, wsf, xsf, n, m
        else:
            kernel_a, kernel_b, a_sf, b_sf, kernel_m, kernel_n = xq, weight, xsf, wsf, m, n
        sf_k = (k // SF_VEC + 3) // 4
        self._fn(
            kernel_a,
            kernel_b,
            out[:m] if out.shape[0] != m else out,
            (kernel_m + SF_ROW_TILE - 1) // SF_ROW_TILE,
            (kernel_n + SF_ROW_TILE - 1) // SF_ROW_TILE,
            sf_k,
            a_sf.data_ptr(),
            b_sf.data_ptr(),
            self._alpha_for(xq.device),
        )
        return out
