# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-bounded full-query FP8 DSA logits for the Rubin DEP16 target."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger

from .full_query_policy import fp8_full_query_weight_cache_limit

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}
_L2_EVICT_FIRST_POLICY = 0x12F0000000000000


@lru_cache(maxsize=None)
def _get_num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def fp8_logits_early_kv_supported(
    next_n: int,
    num_heads: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> bool:
    """Shapes of the early-KV FP8 paged MQA logits kernel (PDL launch, N split).

    The kernel splits each group's accumulator into two N halves of next_n / 2
    query rows; a half must be a whole number of 16-column UMMA atoms, and the
    split epilogue is the FP32 one with a single sub-tile.
    """
    return (
        next_n % 2 == 0
        and (next_n * num_heads // 2) % 16 == 0
        and num_epi_subtiles == 1
        and epi_dtype == torch.float32
        and acc_dtype == torch.float32
        and output_dtype == torch.float32
    )


class CuteDSLPagedMQALogitsRunner:
    """Runner for CuTe DSL FP8 Paged MQA Logits kernel (Blackwell SM100).

    Caches compiled kernels keyed by static params
    (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n, num_sms).
    """

    kernel_cache = dict()

    @classmethod
    def _compile(
        cls,
        compute_block_kv,
        phys_block_kv,
        num_heads,
        head_dim,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        acc_dtype,
        output_dtype,
        weight_cache_limit=None,
        early_kv=False,
        kv_cache_policy=None,
    ):
        """Compile kernel using fake tensors + TVM FFI."""
        key = (
            compute_block_kv,
            phys_block_kv,
            num_heads,
            head_dim,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )
        if weight_cache_limit is not None:
            key += (weight_cache_limit,)
        if early_kv:
            key += ("early_kv",)
        if kv_cache_policy is not None:
            key += ("kv_cache_policy", kv_cache_policy)
        if key in cls.kernel_cache:
            return

        to_cutlass = _TORCH_TO_CUTLASS_DTYPE
        N = next_n * num_heads
        block_bytes = phys_block_kv * (head_dim + 4)

        sym_num_phys_blocks = cute.sym_int()
        sym_B = cute.sym_int()
        max_ctx = cute.sym_int()
        max_blocks_per_seq = cute.sym_int()
        num_ctas = cute.sym_int()

        # KV may come from the indexer K-cache pool view, which is
        # strided in dim 0 (pool layout interleaves layers:
        # [num_blocks, num_layers, kvFactor, blockSize]). Declare outer
        # stride as sym so the actual per-block stride is read at
        # runtime; innermost stride is fixed to 1 (byte-contig within a
        # logical block view).
        kv_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8, (sym_num_phys_blocks, block_bytes), stride=(cute.sym_int64(), 1)
        )

        q_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (N, head_dim, sym_B), stride_order=(1, 0, 2)
        )

        w_dtype = cutlass.Float16 if epi_dtype == torch.float16 else to_cutlass[epi_dtype]
        w_fake = cute.runtime.make_fake_compact_tensor(w_dtype, (N, sym_B), stride_order=(0, 1))

        logits_fake = cute.runtime.make_fake_tensor(
            to_cutlass[output_dtype], (cute.sym_int(), max_ctx), stride=(cute.sym_int64(), 1)
        )

        bt_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_B, max_blocks_per_seq), stride_order=(1, 0)
        )

        cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_B,), stride_order=(0,))

        sm_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
        )

        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        from ...kernels.attention.fp8_paged_mqa_logits import FP8MQALogitsKernel

        kernel = FP8MQALogitsKernel(
            block_kv=compute_block_kv,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=next_n,
            num_sms=num_sms,
            num_epi_subtiles=num_epi_subtiles,
            epi_dtype=to_cutlass[epi_dtype],
            acc_dtype=to_cutlass[acc_dtype],
            output_dtype=to_cutlass[output_dtype],
            weight_cache_limit=weight_cache_limit,
            split_n=early_kv,
            early_kv=early_kv,
            kv_cache_policy=kv_cache_policy,
        )

        compiled = cute.compile(
            kernel,
            kv_fake,
            q_fake,
            w_fake,
            logits_fake,
            bt_fake,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )
        cls.kernel_cache[key] = compiled
        if kv_cache_policy is not None:
            logger.info_once(
                "Mid-batch attention-DP decode: the FP8 paged MQA logits "
                "kernel loads the K cache evict-first",
                key="dsa_logits_tuned",
            )
        logger.debug(
            f"[compile cute_dsl fp8_paged_mqa_logits] {key}"
            f" kv_stages={kernel.num_kv_stages}"
            f" umma_stages={kernel.num_umma_stages}"
        )

    @classmethod
    def forward(
        cls,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
        early_kv: bool = False,
    ) -> torch.Tensor:
        """Execute FP8 paged MQA logits kernel.

        Args:
            q: [B, next_n, H, D] FP8
            kv_fused: [num_blocks, phys_block_kv, 1, D+4] uint8
            weights: [B*next_n, H] float32
            context_lens: [B] int32
            block_table: [B, max_blocks] int32
            schedule_meta: [num_sms+1, 2] int32
            max_context_len: int
            num_epi_subtiles: epilogue sub-tile count (1, 2, or 4)
            epi_dtype: epilogue compute dtype
            acc_dtype: MMA accumulator dtype
            output_dtype: output logits dtype
            early_kv: the q rows are each request's newest next_n tokens
                and the kernels this launch depends on write no older KV
                row: run the N-split variant launched with PDL, which
                loads the older KV blocks before griddepcontrol.wait
                (shapes of fp8_logits_early_kv_supported; otherwise the
                default kernel runs)
        Returns:
            logits: [B*next_n, max_context_len] output_dtype
        """
        B, next_n, H, D = q.shape
        N = next_n * H
        phys_block_kv = kv_fused.shape[1]
        compute_block_kv = 128
        num_phys_blocks = kv_fused.shape[0]
        num_sms = _get_num_sms()

        # Reshape Q: [B, next_n, H, D] -> [B, N, D] -> [N, D, B]
        q_3d = q.reshape(B, N, D).permute(1, 2, 0)

        # Reshape weights: [B*next_n, H] -> [B, N] -> [N, B]
        if epi_dtype == torch.float16:
            # TODO: move type conversion to weight loading
            w_2d = weights.reshape(B, N).half().t()
        else:
            w_2d = weights.reshape(B, N).t()

        # Flatten fused KV to [num_phys_blocks, block_bytes]
        kv_flat = kv_fused.reshape(num_phys_blocks, -1)

        # Allocate output with alignment padding
        SPLIT_KV = compute_block_kv * 2  # NUM_MATH_WG = 2
        aligned_max_ctx = ((max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
        # Use a persistent arena buffer instead of a per-forward torch.empty
        # so the output address stays stable across CUDA-graph replays.
        _reserve = torch.cuda.is_current_stream_capturing()
        logits = get_memory_buffers().get_buffer(
            [B * next_n, aligned_max_ctx],
            output_dtype,
            buffer_name="glm_rubin_high_throughput_mqa_logits",
            reserve_buffer=_reserve,
        )
        logits = logits[:, :max_context_len]

        # Compile if needed (fake tensors, no real data required)
        weight_cache_limit = None
        if True:
            weight_cache_limit = fp8_full_query_weight_cache_limit(
                next_n,
                H,
                D,
                phys_block_kv,
                q.dtype,
                num_epi_subtiles,
                epi_dtype,
                acc_dtype,
                output_dtype,
            )
        early_kv = early_kv and fp8_logits_early_kv_supported(
            next_n, H, num_epi_subtiles, epi_dtype, acc_dtype, output_dtype
        )
        # On the mid-batch attention-DP decode (decode_shape.py) the kernel
        # loads the K cache (KV and scales) evict-first, so the logits it
        # writes stay in L2 for the top-k.
        kv_cache_policy = _L2_EVICT_FIRST_POLICY if True else None
        key = (
            compute_block_kv,
            phys_block_kv,
            H,
            D,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )
        if weight_cache_limit is not None:
            key += (weight_cache_limit,)
        if early_kv:
            key += ("early_kv",)
        if kv_cache_policy is not None:
            key += ("kv_cache_policy", kv_cache_policy)
        if key not in cls.kernel_cache:
            cls._compile(
                compute_block_kv,
                phys_block_kv,
                H,
                D,
                next_n,
                num_sms,
                num_epi_subtiles,
                epi_dtype,
                acc_dtype,
                output_dtype,
                weight_cache_limit,
                early_kv,
                kv_cache_policy,
            )
        compiled = cls.kernel_cache[key]

        # FP8 q needs uint8 view to match compile-time dtype
        q_for_ffi = (
            q_3d.view(torch.uint8)
            if q_3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else q_3d
        )

        # TVM FFI: pass raw tensors, no dlpack/stream needed
        compiled(
            kv_flat,
            q_for_ffi,
            w_2d,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            num_phys_blocks,
            B,
        )
        return logits


@torch.library.custom_op(
    "trtllm::glm_rubin_high_throughput_fp8_logits", mutates_args=(), device_types="cuda"
)
def cute_dsl_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
    early_kv: bool = False,
) -> torch.Tensor:
    if not is_sm_100f():
        raise ValueError(
            f"CuteDSL: SM version {get_sm_version()} is not supported. "
            f"CuteDSL FP8 Paged MQA Logits only supports SM 100 family."
        )
    # Caller (dsa.py) prepares all tensors with metadata-guaranteed
    # dtype/shape; skip per-call validation to keep decode-hot-path
    # latency low. Log inputs once for debugging.
    logger.info_once(
        f"cute_dsl_fp8_paged_mqa_logits inputs: "
        f"q dtype={q.dtype} shape={tuple(q.shape)} stride={q.stride()}; "
        f"kv_fused dtype={kv_fused.dtype} shape={tuple(kv_fused.shape)} stride={kv_fused.stride()}; "
        f"weights dtype={weights.dtype} shape={tuple(weights.shape)} stride={weights.stride()}; "
        f"context_lens dtype={context_lens.dtype} shape={tuple(context_lens.shape)}; "
        f"block_table dtype={block_table.dtype} shape={tuple(block_table.shape)} stride={block_table.stride()}; "
        f"schedule_meta dtype={schedule_meta.dtype} shape={tuple(schedule_meta.shape)}; "
        f"max_context_len={max_context_len} num_epi_subtiles={num_epi_subtiles} "
        f"epi_dtype={epi_dtype} acc_dtype={acc_dtype} output_dtype={output_dtype}",
        key="cute_dsl_fp8_paged_mqa_logits_inputs",
    )
    return CuteDSLPagedMQALogitsRunner.forward(
        q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        acc_dtype=acc_dtype,
        output_dtype=output_dtype,
        early_kv=early_kv,
    )


@cute_dsl_fp8_paged_mqa_logits.register_fake
def _fake_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
    early_kv: bool = False,
) -> torch.Tensor:
    return torch.empty(
        (q.shape[0] * q.shape[1], max_context_len), dtype=output_dtype, device=q.device
    )
