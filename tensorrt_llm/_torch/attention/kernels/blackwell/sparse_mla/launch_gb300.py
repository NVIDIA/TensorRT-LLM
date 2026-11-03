# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host entry of the GB300 (512 TMEM columns) sparse MLA attention (``kernel_gb300.py``)."""

from __future__ import annotations

import torch

_COMPILED: dict = {}


def sparse_mla_attention_gb300(
    q: torch.Tensor,
    bank_ptr: int,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sinks: torch.Tensor,
    scale: torch.Tensor,
    vscale: torch.Tensor,
    out: torch.Tensor | None,
    *,
    epi: str = "stm",
    positions: torch.Tensor | None = None,
    rope_table: torch.Tensor | None = None,
    out_fp8: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    pdl: bool = False,
) -> None:
    """Attention of ``q`` ``[T, 64, 512]`` E4M3 over each token's bank rows ``indices[t, :lengths[t]]``.

    ``bank_ptr`` is the base address of the ``[R, 512]`` E4M3 bank rows; ``indices`` is ``[T, W]`` int32 with
    W <= 640 and W % 4 == 0 (row stride W; entries past the length are ignored), ``lengths`` ``[T]`` int32;
    ``sinks`` ``[64]`` FP32 logits; ``scale`` FP32 q_dequant * kv_dequant * softmax_scale * log2(e) and ``vscale``
    FP32 kv_dequant. ``epi="stm"`` writes ``out`` ``[T, 64, 512]`` BF16.

    ``epi="rope_mxfp8"`` writes ``grouped_output_mxfp8``'s input instead, bit for bit what
    ``inverse_rope_mxfp8(out, positions, rope_table, 8)`` gives on the BF16 O: ``out_fp8`` E4M3 ``[8, T, 4096]`` and
    ``out_scale`` uint8 ``[8 * align128(T) * 128]``, per group the CUTLASS-swizzled UE8M0 scales of 1x32 blocks
    (padding rows zero), from ``positions`` int32 ``[T]`` and the layer's inverse-RoPE ``rope_table`` FP32
    ``[max_pos, 2, 32]`` (cos, sin).

    ``pdl``: launch with a programmatic edge to the preceding kernel; the kernel's setup runs before it waits for
    that kernel, and it releases its dependents after its last store.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from . import kernel_gb300 as _k

    tokens = q.shape[0]
    if q.dtype != torch.float8_e4m3fn or q.shape[1:] != (_k.HEADS, _k.DIM) or not q.is_contiguous():
        raise ValueError("Q must be a contiguous [T, 64, 512] E4M3 tensor")
    if bank_ptr % 16:
        raise ValueError("bank_ptr must be 16-byte aligned")
    if indices.dtype != torch.int32 or indices.dim() != 2 or indices.shape[0] < tokens:
        raise ValueError("indices must be an int32 [T, W] tensor")
    width = indices.stride(0)
    if indices.stride(1) != 1 or width % 4 or indices.shape[1] > _k.LIST_COLS:
        raise ValueError(
            f"indices rows must be contiguous, 16-byte aligned and at most {_k.LIST_COLS} wide"
        )
    if lengths.dtype != torch.int32 or lengths.shape[0] < tokens:
        raise ValueError("lengths must be an int32 [T] tensor")
    rope = epi == "rope_mxfp8"
    scale_m = (tokens + 127) // 128 * 128
    if rope:
        if (
            positions is None
            or positions.dtype != torch.int32
            or positions.shape[0] < tokens
            or not positions.is_contiguous()
        ):
            raise ValueError("rope_mxfp8 needs int32 positions [T]")
        if (
            rope_table is None
            or rope_table.dtype != torch.float32
            or not rope_table.is_contiguous()
            or rope_table.numel() % 64
        ):
            raise ValueError("rope_mxfp8 needs the FP32 [max_pos, 2, 32] inverse-RoPE table")
        if (
            out_fp8 is None
            or out_fp8.dtype != torch.float8_e4m3fn
            or tuple(out_fp8.shape) != (8, tokens, 4096)
            or not out_fp8.is_contiguous()
        ):
            raise ValueError("rope_mxfp8 needs out_fp8 E4M3 [8, T, 4096]")
        if (
            out_scale is None
            or out_scale.dtype != torch.uint8
            or out_scale.numel() != 8 * scale_m * 128
            or not out_scale.is_contiguous()
        ):
            raise ValueError("rope_mxfp8 needs out_scale uint8 [8 * align128(T) * 128]")
        for t_ in (out_fp8, out_scale):
            if t_.data_ptr() % 16:
                raise ValueError("rope_mxfp8 outputs must be 16-byte aligned")
    gmem = cute.AddressSpace.gmem
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    args = (
        make_ptr(cutlass.Float8E4M3FN, q.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, bank_ptr, gmem, assumed_align=16),
        make_ptr(cutlass.Int32, indices.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Int32, lengths.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Float32, sinks.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Float32, scale.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Float32, vscale.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, (out_fp8 if rope else out).data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Int32, (positions if rope else lengths).data_ptr(), gmem, assumed_align=4),
        make_ptr(
            cutlass.Float32, (rope_table if rope else scale).data_ptr(), gmem, assumed_align=4
        ),
        make_ptr(cutlass.Uint8, (out_fp8 if rope else q).data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float32, (out_scale if rope else scale).data_ptr(), gmem, assumed_align=4),
        cutlass.Int32(tokens),
        cutlass.Int32(width),
        cutlass.Int32(scale_m),
        cutlass.Int32(min(tokens, sms)),
        cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    major, minor = torch.cuda.get_device_capability(q.device)
    arch = f"sm_{major}{minor}"
    key = (arch, epi, bool(pdl))
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compile the GB300 sparse MLA kernel before CUDA graph capture")
        fn = _COMPILED[key] = cute.compile(
            _k.SparseMlaAttentionGB300(arch, epi=epi, pdl=bool(pdl)), *args
        )
    fn(*args)
