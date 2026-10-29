# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import ClassVar, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

IS_K3_CONTEXT_FMHA_AVAILABLE = False
K3_CONTEXT_FMHA_IMPORT_ERROR: Optional[ImportError] = None

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Float32, Int32

    from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_context_fmha_helpers as fmha_utils
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.k3_context_fmha import (
        RubinFusedMultiHeadAttentionForward,
    )

    IS_K3_CONTEXT_FMHA_AVAILABLE = bool(
        getattr(
            RubinFusedMultiHeadAttentionForward,
            "supports_split_k_epilogue_merge",
            False,
        )
    )
except ImportError as error:
    K3_CONTEXT_FMHA_IMPORT_ERROR = error


if IS_K3_CONTEXT_FMHA_AVAILABLE:

    class CuteDSLK3ContextFmhaRunner:
        """Compile and launch the Rubin FP8 split-K context FMHA."""

        kernel_cache: ClassVar[dict] = {}

        @classmethod
        def run(
            cls,
            q: torch.Tensor,
            k_nope: torch.Tensor,
            k_pe: torch.Tensor,
            v: torch.Tensor,
            output: torch.Tensor,
            cu_q_seqlens: torch.Tensor,
            cu_kv_seqlens: torch.Tensor,
            merge_stats: torch.Tensor,
            merge_op: torch.Tensor,
            max_q_len: int,
            max_kv_len: int,
            softmax_scale: float,
            is_causal: bool,
        ) -> None:
            num_tokens_q, num_heads, head_dim = q.shape
            num_tokens_kv = k_nope.shape[0]
            batch_size = merge_op.numel()

            q_5d = q.view(1, num_tokens_q, num_heads, 1, head_dim)
            k_nope_5d = k_nope.view(1, num_tokens_kv, num_heads, 1, 128)
            k_pe_5d = k_pe.view(1, num_tokens_kv, 1, 1, 64)
            v_5d = v.view(1, num_tokens_kv, num_heads, 1, 128)
            output_5d = output.view(1, num_tokens_q, num_heads, 1, 128)

            cache_key = (q.dtype, num_heads, is_causal, cu_kv_seqlens.dtype)
            compiled_fmha = cls.kernel_cache.get(cache_key)
            problem_size = (
                batch_size,
                max_q_len,
                num_tokens_q,
                max_kv_len,
                num_heads,
                num_heads,
                192,
                128,
            )
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            scale = Float32(softmax_scale)

            if compiled_fmha is None:
                logger.info(
                    "Compiling Rubin K3 FP8 context FMHA: split K=(128,64), "
                    "epilogue merge, causal=%s",
                    is_causal,
                )
                mask_type = (
                    fmha_utils.MaskEnum.WINDOW_MASK
                    if is_causal
                    else fmha_utils.MaskEnum.RESIDUAL_MASK
                )
                fmha = RubinFusedMultiHeadAttentionForward(
                    cutlass.Float32,
                    cutlass.Float32,
                    (128, 128),
                    (192, 128),
                    False,
                    mask_type,
                    False,
                    False,
                    use_tma_store=False,
                    split_k_dims=(128, 64),
                )

                def as_dynamic_tensor(tensor: torch.Tensor, align: int = 16):
                    return from_dlpack(tensor, assumed_align=align).mark_layout_dynamic(
                        leading_dim=cutlass_torch.get_leading_dim(tensor)
                    )

                compiled_fmha = cute.compile(
                    fmha,
                    as_dynamic_tensor(q_5d),
                    as_dynamic_tensor(k_nope_5d),
                    as_dynamic_tensor(k_pe_5d),
                    as_dynamic_tensor(v_5d),
                    as_dynamic_tensor(output_5d),
                    problem_size,
                    as_dynamic_tensor(cu_q_seqlens, 4),
                    as_dynamic_tensor(cu_kv_seqlens, 4),
                    None,
                    None,
                    as_dynamic_tensor(merge_stats),
                    as_dynamic_tensor(merge_op, 8),
                    Float32(softmax_scale * math.log2(math.e)),
                    scale,
                    Float32(1.0),
                    None,
                    None,
                    Int32(0) if is_causal else None,
                    None,
                    None,
                    stream,
                    False,
                    options="--opt-level=2 --ptxas-options '--uumn'",
                )
                cls.kernel_cache[cache_key] = compiled_fmha

            compiled_fmha(
                q_5d,
                k_nope_5d,
                k_pe_5d,
                v_5d,
                output_5d,
                problem_size,
                cu_q_seqlens,
                cu_kv_seqlens,
                None,
                None,
                merge_stats,
                merge_op,
                softmax_scale * math.log2(math.e),
                softmax_scale,
                1.0,
                None,
                None,
                0 if is_causal else None,
                None,
                None,
                stream,
                False,
            )

    def _validate_k3_context_fmha_inputs(
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
    ) -> None:
        if get_sm_version() != 107:
            raise ValueError("K3 context FMHA requires Rubin SM107.")
        if any(tensor.dtype != torch.float8_e4m3fn for tensor in (q, k_nope, k_pe, v)):
            raise ValueError("K3 context FMHA requires FP8 E4M3 Q/K/V inputs.")
        if output.dtype != torch.bfloat16:
            raise ValueError("K3 context FMHA requires BF16 output.")
        if q.dim() != 3 or q.shape[1] <= 0 or q.shape[2] != 192:
            raise ValueError(f"Expected q [T, H, 192] with H > 0, got {tuple(q.shape)}.")
        num_heads = q.shape[1]
        if k_nope.dim() != 3 or k_nope.shape[1:] != (num_heads, 128):
            raise ValueError(f"Expected k_nope [T, {num_heads}, 128], got {tuple(k_nope.shape)}.")
        if k_pe.dim() != 3 or k_pe.shape[1:] != (1, 64):
            raise ValueError(f"Expected k_pe [T, 1, 64], got {tuple(k_pe.shape)}.")
        if v.dim() != 3 or v.shape[1:] != (num_heads, 128):
            raise ValueError(f"Expected v [T, {num_heads}, 128], got {tuple(v.shape)}.")
        if output.shape != (q.shape[0], num_heads, 128):
            raise ValueError(
                f"Expected output {(q.shape[0], num_heads, 128)}, got {tuple(output.shape)}."
            )
        if merge_stats.shape != (q.shape[0], num_heads, 2):
            raise ValueError(
                f"Expected merge_stats {(q.shape[0], num_heads, 2)}, got {tuple(merge_stats.shape)}."
            )
        if merge_stats.dtype != torch.float32:
            raise ValueError("merge_stats must be float32.")
        if merge_op.dtype != torch.int64 or merge_op.dim() != 1:
            raise ValueError("merge_op must be a one-dimensional int64 tensor.")
        expected_cu_shape = (merge_op.numel() + 1,)
        for name, cu_seqlens, dtypes in (
            ("cu_q_seqlens", cu_q_seqlens, (torch.int32,)),
            ("cu_kv_seqlens", cu_kv_seqlens, (torch.int32, torch.int64)),
        ):
            if cu_seqlens.dtype not in dtypes or cu_seqlens.shape != expected_cu_shape:
                raise ValueError(
                    f"{name} must have dtype in {dtypes} and shape {expected_cu_shape}, got "
                    f"dtype={cu_seqlens.dtype}, shape={tuple(cu_seqlens.shape)}."
                )
        if k_nope.shape[0] != k_pe.shape[0] or k_nope.shape[0] != v.shape[0]:
            raise ValueError("K-nope, K-pe, and V token counts must match.")
        for name, tensor in (("k_nope", k_nope), ("v", v)):
            if tensor.stride()[1:] != (128, 1):
                raise ValueError(
                    f"{name} must have contiguous head dimensions, got stride {tensor.stride()}."
                )
        if k_pe.stride(2) != 1:
            raise ValueError(
                f"k_pe must have a contiguous head dimension, got stride {k_pe.stride()}."
            )
        for name, tensor in (
            ("q", q),
            ("output", output),
            ("merge_stats", merge_stats),
        ):
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")

    @torch.library.custom_op(
        "trtllm::cute_dsl_k3_context_fmha_rubin",
        mutates_args=("output", "merge_stats"),
        device_types="cuda",
    )
    def cute_dsl_k3_context_fmha_rubin(
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> None:
        _validate_k3_context_fmha_inputs(
            q,
            k_nope,
            k_pe,
            v,
            output,
            cu_q_seqlens,
            cu_kv_seqlens,
            merge_stats,
            merge_op,
        )
        CuteDSLK3ContextFmhaRunner.run(
            q,
            k_nope,
            k_pe,
            v,
            output,
            cu_q_seqlens,
            cu_kv_seqlens,
            merge_stats,
            merge_op,
            max_q_len,
            max_kv_len,
            softmax_scale,
            is_causal,
        )

    @torch.library.register_fake("trtllm::cute_dsl_k3_context_fmha_rubin")
    def _(
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> None:
        return None
