# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA verify kernels' 32-value MXFP8 epilogue (``mxfp8_exact_store``) on given BF16 rows.

A check harness, not a model op: one warp per (row, head) holds the head's 128 values as the verify
epilogue holds them (lane ``l``: columns ``32 i + l``) and runs the very ``mxfp8_exact_store`` the
``MXFP8_EXACT`` epilogues of ``rubin_kda_mtp_decode_v2`` and ``rubin_kda_mtp_decode_2cta`` run, so a
test can compare that code with ``torch.ops.trtllm.mxfp8_quantize`` on chosen values (zeros,
denormals, block maxima at every binade, ``448 * 2^k``) that a verify output does not reach.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from .rubin_kda_mtp_decode_v2 import mxfp8_exact_store

    @cute.kernel
    def _mxfp8_k32_rows_kernel(
        x: cute.Tensor, o8: cute.Tensor, osf: cute.Tensor, HV: cutlass.Constexpr[int]
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        i_hv = tidx // 32
        hv_off = i_hv * 128
        xs = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        for i in cutlass.range(4, unroll_full=True):
            # x holds the bf16 bits (int16): the FP32 value has those 16 bits on top (no arithmetic,
            # so subnormals and NaNs reach the epilogue as they are)
            xs[i] = (cutlass.Int32(x[row, hv_off + i * 32 + lane]) << cutlass.Int32(16)).bitcast(
                cutlass.Float32
            )
        mxfp8_exact_store(xs, o8, osf, row, hv_off, i_hv, HV, lane)

    @cute.jit
    def _mxfp8_k32_rows_launch(
        x: cute.Tensor,
        o8: cute.Tensor,
        osf: cute.Tensor,
        rows: cutlass.Int32,
        HV: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        _mxfp8_k32_rows_kernel(x, o8, osf, HV).launch(
            grid=(rows, 1, 1), block=[32 * HV, 1, 1], stream=stream
        )


_compiled = {}


def _arg(t: torch.Tensor):
    arg = from_dlpack(t, assumed_align=16)
    return arg.mark_layout_dynamic(t.dim() - 1)


def mxfp8_k32_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(E4M3 [T, HV * 128], R128c4 UE8M0 bytes [pad128(T) * HV * 4])`` of BF16 ``x``
    ``[T, HV * 128]`` by the verify kernels' ``MXFP8_EXACT`` epilogue code. The scale bytes of the
    padding rows are zero, as ``torch.ops.trtllm.mxfp8_quantize`` writes them."""
    assert IS_CUTLASS_DSL_AVAILABLE, "needs the CUTLASS DSL"
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
    rows, cols = x.shape
    assert cols % 128 == 0 and 0 < cols // 128 <= 32 and x.data_ptr() % 16 == 0
    heads = cols // 128
    codes = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.zeros((rows + 127) // 128 * 128 * heads * 4, dtype=torch.uint8, device=x.device)
    if rows == 0:
        return codes, scales
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = [_arg(t) for t in (x.view(torch.int16), codes, scales)]
    if heads not in _compiled:
        _compiled[heads] = cute.compile(
            _mxfp8_k32_rows_launch, *args, rows, HV=heads, stream=stream
        )
    _compiled[heads](*args, rows, stream)
    return codes, scales
