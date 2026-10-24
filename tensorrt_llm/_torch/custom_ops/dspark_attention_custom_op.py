# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged DSpark attention and Rubin FP8 projection operators."""

import functools
from collections.abc import Callable
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.typing import Numeric, Pointer, Type

from ..._utils import get_sm_version
from ...logger import logger
from ...math_utils import ceil_div, pad_up
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from ..cute_dsl_kernels.blackwell.dspark.attention import DSparkAttention
from ..cute_dsl_kernels.blackwell.utils import make_ptr
from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from ..utils import fp8_scale_infer_shape, mxfp8_scale_infer_shape, next_positive_power_of_2
from .dspark_rmsnorm_rope_custom_op import _get_dspark_arch_str

_DSPARK_NUM_HEADS = 128
_DSPARK_HEAD_DIM = 512
_DSPARK_WINDOW_SIZE = 128
_DSPARK_PAGE_SIZE = 8
_DSPARK_BLOCK_SIZES = (5, 6)
_DSPARK_ROPE_DIM = 64

_DSPARK_CONTEXT_WKV_MIN_M_BUCKET = 128
_DSPARK_GENERATION_M_SPEC = DynamicTensorSpec(0, 0)
# These are request-capacity buckets for tactic selection. The Rubin GEMM
# artifact has runtime-dynamic M/N/K, so this does not specialize a kernel for
# each batch value.
_DSPARK_GENERATION_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@functools.cache
def _get_dspark_generation_m_spec(
    rows_per_request: int,
    max_batch: int,
    input_idx: int = 0,
    dim_idx: int = 0,
) -> DynamicTensorSpec:
    """Build the DSpark operator-owned generation-M bucket contract."""
    if rows_per_request < 1 or max_batch < 1:
        raise ValueError(
            "DSpark generation tuning requires positive rows_per_request and max_batch, "
            f"got {rows_per_request=} and {max_batch=}."
        )

    batch_buckets = tuple(
        batch for batch in _DSPARK_GENERATION_BATCH_BUCKETS if batch < max_batch
    ) + (max_batch,)
    m_buckets = tuple(batch * rows_per_request for batch in batch_buckets)

    def map_to_bucket(m: int) -> int:
        batch = ceil_div(m, rows_per_request)
        for bucket in batch_buckets:
            if batch <= bucket:
                return bucket * rows_per_request
        # A shape beyond the configured worker capacity is not expected, but
        # keeping it exact gives the autotuner its normal cache-miss fallback.
        return m

    return DynamicTensorSpec(
        input_idx,
        dim_idx,
        m_buckets,
        map_to_bucket,
    )


@functools.cache
def _get_dspark_generation_mxfp8_tuning_config(
    rows_per_request: int,
    max_batch: int,
) -> TuningConfig:
    """Return generation buckets for a DSpark MXFP8 projection."""
    return TuningConfig(
        dynamic_tensor_specs=(_get_dspark_generation_m_spec(rows_per_request, max_batch),),
        constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
        use_cold_l2_cache=True,
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        use_cuda_graph=True,
    )


def _map_dspark_context_wkv_m_profile(m: int) -> int:
    """Bucket CTX seed and batched backfill rows without changing kernel shapes."""
    return max(_DSPARK_CONTEXT_WKV_MIN_M_BUCKET, next_positive_power_of_2(m))


def _gen_dspark_context_wkv_m_profiles(max_m: int) -> Tuple[int, ...]:
    max_bucket = _map_dspark_context_wkv_m_profile(max_m)
    buckets = []
    bucket = _DSPARK_CONTEXT_WKV_MIN_M_BUCKET
    while bucket <= max_bucket:
        buckets.append(bucket)
        bucket *= 2
    return tuple(buckets)


# Worker initialization is process-local and single-threaded. These transient
# values let the memoized builder consume the real cache only during precompile.
_DSV4_DSPARK_NUM_HEADS = (64, 128)
_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_WINDOW_SIZE = 128
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSV4_DSPARK_BLOCK_SIZES = (5, 6)
_DSV4_DSPARK_ROPE_DIM = 64


_dspark_attention_kernel_cache: dict[
    tuple[int, int, int, str, int | None], Callable[..., None]
] = {}


def _make_compile_gmem_pointer(dtype: Type[Numeric], assumed_align: int) -> Pointer:
    """Create a compile-only pointer specimen without allocating device memory."""
    return make_ptr(
        dtype,
        0,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _compile_dspark_attention(
    block_size: int,
    arch_str: str,
    page_size: int,
    output_groups: int | None,
    num_heads: int,
) -> Callable[..., None]:
    """Compile the pointer host wrapper without runtime tensor specimens."""
    head_dim = _DSV4_DSPARK_HEAD_DIM
    batch = cute.sym_int()
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, block_size, num_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    output_scale_fake = None
    if output_groups is not None:
        group_dim = num_heads * head_dim // output_groups
        rows = batch * block_size
        output_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (output_groups, rows, group_dim),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        output_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Real tensor specimens used to make PyTorch's primary CUDA context current
    # implicitly. Fake-only compilation must do so before querying occupancy;
    # the returned stream is not part of the compiled TVM-FFI launch ABI.
    torch.cuda.current_stream()
    hardware_info = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(2)
    kernel = DSparkAttention(
        cutlass.Float32,
        (128, 128),
        (128, 256),
        max_active_clusters,
        _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
        _DSV4_DSPARK_WINDOW_SIZE,
        0.0,
        seq_len_q=block_size,
        mma_qk_tiler_k=128,
        inverse_rope_dim=_DSV4_DSPARK_ROPE_DIM,
        arch_str=arch_str,
        history_page_size=page_size,
        output_groups=output_groups,
        num_heads=num_heads,
    )
    # Scalars and typed, aligned compile-only pointers define the runtime ABI.
    # ``output_fake`` keeps one tensor argument for TVM-FFI environment-stream
    # detection without tying compilation to a real worker buffer.
    compiled = cute.compile(
        kernel.wrapper,
        1,
        1,
        page_size * head_dim,
        1,
        1,
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.Int32, 4),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        output_fake,
        output_scale_fake,
        cutlass.Float32(1.0),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    logger.info(
        "DSpark Attention enabled: implementation=dspark_attn, "
        f"block={block_size}, heads={num_heads}, head_dim={head_dim}"
    )
    return compiled


def _run_paged_dspark_attention(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Attend to a 128-token context window and fuse the inverse-RoPE epilogue."""
    arch_str = _get_dspark_arch_str()
    if arch_str is None:
        raise RuntimeError("Embedded DSpark attention requires SM100, SM103, or SM107")
    batch, block, heads, dim = q.shape
    page_size = kv_pages.shape[1]
    if (
        block not in _DSV4_DSPARK_BLOCK_SIZES
        or heads not in _DSV4_DSPARK_NUM_HEADS
        or dim != _DSV4_DSPARK_HEAD_DIM
        or page_size not in (16, 32, 64, 128, 256)
    ):
        raise ValueError(
            "DSpark requires block 5/6, 64/128 heads, head_dim 512 and 16/32/64/128/256-token pages"
        )
    if q.dtype != torch.bfloat16 or draft_block.dtype != q.dtype or kv_pages.dtype != q.dtype:
        raise ValueError("DSpark Q, draft KV and pages must use BF16")
    if draft_block.shape != (batch, 8, dim) or kv_pages.shape[2] != dim:
        raise ValueError("DSpark draft KV must have shape [batch, 8, 512]")
    if (
        block_tables.dtype != torch.int32
        or block_tables.shape[0] != batch
        or block_tables.stride(1) != 1
    ):
        raise ValueError("DSpark page tables must have contiguous INT32 rows")
    for value in (positions, valid_lengths, capacities):
        if value.dtype != torch.int64 or value.shape != (batch,) or not value.is_contiguous():
            raise ValueError(
                "DSpark positions, lengths and capacities must be contiguous INT64 batch vectors"
            )
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (heads,):
        raise ValueError("DSpark attention sink must be FP32 with one entry per head")
    if inverse_rope_freqs.dtype != torch.float32 or inverse_rope_freqs.shape != (
        batch,
        block,
        32,
        2,
    ):
        raise ValueError("DSpark inverse RoPE frequencies must be FP32 [batch, block, 32, 2]")
    if kv_pages.stride(1) != dim or kv_pages.stride(2) != 1 or kv_pages.stride(0) % 8:
        raise ValueError("DSpark pages require contiguous tokens and an aligned page stride")
    for value in (q, draft_block, attn_sink, inverse_rope_freqs):
        if not value.is_contiguous():
            raise ValueError("DSpark query, draft, sink and frequencies must be contiguous")
    if not all(
        value.is_cuda and value.device == q.device
        for value in (
            q,
            draft_block,
            kv_pages,
            block_tables,
            positions,
            valid_lengths,
            capacities,
            attn_sink,
            inverse_rope_freqs,
        )
    ):
        raise ValueError("All DSpark attention inputs must be on the same CUDA device")
    cache_key = (block, heads, page_size, arch_str, output_groups)
    compiled = _dspark_attention_kernel_cache.get(cache_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark paged attention must run eagerly before graph capture")
        compiled = _compile_dspark_attention(block, arch_str, page_size, output_groups, heads)
        _dspark_attention_kernel_cache[cache_key] = compiled
    output = torch.empty_like(q)
    output_scale = None
    if output_groups is not None:
        rows = batch * block
        group_dim = heads * dim // output_groups
        output = torch.empty(
            (output_groups, rows, group_dim), dtype=torch.float8_e4m3fn, device=q.device
        )
        output_scale = torch.empty(
            (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
            dtype=torch.float32,
            device=q.device,
        )
    compiled(
        batch,
        kv_pages.shape[0],
        kv_pages.stride(0),
        block_tables.shape[1],
        block_tables.stride(0),
        q.data_ptr(),
        kv_pages.data_ptr(),
        draft_block.data_ptr(),
        block_tables.data_ptr(),
        positions.data_ptr(),
        valid_lengths.data_ptr(),
        capacities.data_ptr(),
        attn_sink.data_ptr(),
        inverse_rope_freqs.data_ptr(),
        output,
        output_scale,
        softmax_scale,
    )
    return output, output_scale


@torch.library.custom_op(
    "trtllm::fused_dsv4_dspark_attention", mutates_args=(), device_types="cuda"
)
def fused_dsv4_dspark_attention(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return _run_paged_dspark_attention(
        q,
        draft_block,
        kv_pages,
        block_tables,
        positions,
        valid_lengths,
        capacities,
        attn_sink,
        inverse_rope_freqs,
        softmax_scale,
    )[0]


@torch.library.register_fake("trtllm::fused_dsv4_dspark_attention")
def _(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "trtllm::fused_dsv4_dspark_attention_fp8", mutates_args=(), device_types="cuda"
)
def fused_dsv4_dspark_attention_fp8(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, scales = _run_paged_dspark_attention(
        q,
        draft_block,
        kv_pages,
        block_tables,
        positions,
        valid_lengths,
        capacities,
        attn_sink,
        inverse_rope_freqs,
        softmax_scale,
        output_groups,
    )
    assert scales is not None
    return output, scales


@torch.library.register_fake("trtllm::fused_dsv4_dspark_attention_fp8")
def _(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
    output_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q.shape[0] * q.shape[1]
    group_dim = q.shape[2] * q.shape[3] // output_groups
    return (
        torch.empty((output_groups, rows, group_dim), dtype=torch.float8_e4m3fn, device=q.device),
        torch.empty(
            (output_groups, group_dim // 128, (rows + 3) // 4 * 4),
            dtype=torch.float32,
            device=q.device,
        ),
    )


if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    from .cute_dsl_custom_ops import CuteDSLFp8RubinBmmQuantRunner, CuteDSLMXFP8RubinLinear

    @functools.cache
    def _get_dspark_generation_fp8_bmm_quant_tuning_config(
        rows_per_request: int,
        max_batch: int,
    ) -> TuningConfig:
        # One-engine DSpark runs target verification (K+1 rows/request)
        # before the K-row draft. Tune the shared OA BMM for both real
        # generation families when the draft first publishes its capacity.
        draft_spec = _get_dspark_generation_m_spec(
            rows_per_request, max_batch, input_idx=0, dim_idx=1
        )
        target_spec = _get_dspark_generation_m_spec(
            rows_per_request + 1, max_batch, input_idx=0, dim_idx=1
        )
        m_buckets = tuple(
            sorted(set(draft_spec.gen_tuning_buckets + target_spec.gen_tuning_buckets))
        )

        def map_to_bucket(m: int) -> int:
            for bucket in m_buckets:
                if m <= bucket:
                    return bucket
            return m

        return TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=0,
                    dim_idx=1,
                    gen_tuning_buckets=m_buckets,
                    map_to_tuning_buckets=map_to_bucket,
                ),
            ),
            constraint_specs=(
                ConstraintSpec(2, 2, fp8_scale_infer_shape),
                ConstraintSpec(4, 1, lambda input_shapes: input_shapes[0][1]),
                ConstraintSpec(
                    5,
                    0,
                    lambda input_shapes: pad_up(input_shapes[0][1], 128)
                    * pad_up(
                        ceil_div(input_shapes[0][0] * input_shapes[1][1], 32),
                        4,
                    ),
                ),
            ),
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

    class CuteDSLDSparkFp8RubinBmmQuantRunner(CuteDSLFp8RubinBmmQuantRunner):
        """DSpark OA runner with operator-owned M buckets and launch plans."""

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(0, 1),),
            constraint_specs=(
                ConstraintSpec(2, 2, fp8_scale_infer_shape),
                ConstraintSpec(4, 1, lambda input_shapes: input_shapes[0][1]),
                ConstraintSpec(
                    5,
                    0,
                    lambda input_shapes: pad_up(input_shapes[0][1], 128)
                    * pad_up(
                        ceil_div(input_shapes[0][0] * input_shapes[1][1], 32),
                        4,
                    ),
                ),
            ),
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def _forward_tvm_ffi_prepared(
            self,
            inputs: List[torch.Tensor],
            compiled_gemm,
        ) -> None:
            if self.fp8_quantize_1x128:
                (
                    a_tensor,
                    b_tensor,
                    a_sf_tensor,
                    b_sf_tensor,
                    c_tensor,
                    scale_tensor,
                ) = inputs
            else:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, c_tensor = inputs
                scale_tensor = None

            batch_size, m, k = a_tensor.shape
            n = b_tensor.shape[1]
            runtime_args = (
                m,
                n,
                k,
                pad_up(m, 4),
                ceil_div(n, 128),
                ceil_div(k, 128),
                batch_size,
                a_tensor.data_ptr(),
                b_tensor.data_ptr(),
                a_sf_tensor.data_ptr(),
                b_sf_tensor.data_ptr(),
                c_tensor.permute(1, 2, 0),
            )
            if self.fp8_quantize_1x128:
                compiled_gemm(*runtime_args, scale_tensor)
            else:
                compiled_gemm(*runtime_args)

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__(use_tvm_ffi=use_tvm_ffi)
            self._tvm_ffi_launch_plans = {}

        def unique_id(self):
            return (self.output_dtype, self.use_tvm_ffi)

    @functools.cache
    def _get_cute_dsl_dspark_fp8_rubin_bmm_quant_runner(
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkFp8RubinBmmQuantRunner:
        return CuteDSLDSparkFp8RubinBmmQuantRunner(use_tvm_ffi=use_tvm_ffi)

    class CuteDSLDSparkMXFP8RubinLinear(CuteDSLMXFP8RubinLinear):
        """DSpark MXFP8 runner with process-local tuning and launch plans."""

        # DSpark runs its own launch plans: no B-stationary tactics (TRTLLM_CUTEDSL_MXFP8_STATIONARY).
        stationary_supported = False

        tuning_config = TuningConfig(
            # DSpark warmup runs the real generation shapes. Keep those M
            # values exact so runtime lookup uses the tactic measured for
            # the captured graph instead of remapping it to a power of two.
            dynamic_tensor_specs=(_DSPARK_GENERATION_M_SPEC,),
            constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
            use_cuda_graph=True,
        )

        def _forward_tvm_ffi_prepared(
            self,
            inputs: List[torch.Tensor],
            launch_plan,
            bias: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            compiled_gemm, swap_ab = launch_plan
            if len(inputs) == 5:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs
                output_tensor = None
            else:
                (
                    a_tensor,
                    b_tensor,
                    a_sf_tensor,
                    b_sf_tensor,
                    alpha_tensor,
                    output_tensor,
                ) = inputs

            m, k = a_tensor.shape
            n = b_tensor.shape[0]
            partition_id = kwargs.get("partition_id", -1)
            if output_tensor is not None:
                c_tensor = output_tensor[:, partition_id * n : (partition_id + 1) * n]
            elif self.to_userbuffers:
                c_tensor = torch.ops.trtllm.create_userbuffers_tensor([m, n], self.output_dtype)
            else:
                c_tensor = torch.empty(
                    (m, n),
                    dtype=self.output_dtype,
                    device=a_tensor.device,
                )

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)

            real_k = k * self.packed_k_multiplier
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // self.sf_vec_size, 4)
            sf_n = pad_up(n, 128)
            if swap_ab:
                kernel_m = n
                kernel_n = m
                kernel_sf_m = sf_n
                kernel_sf_n = sf_m
                kernel_a_tensor = b_tensor
                kernel_a_sf_tensor = b_sf_tensor
                kernel_b_tensor = a_tensor
                kernel_b_sf_tensor = a_sf_tensor
            else:
                kernel_m = m
                kernel_n = n
                kernel_sf_m = sf_m
                kernel_sf_n = sf_n
                kernel_a_tensor = a_tensor
                kernel_a_sf_tensor = a_sf_tensor
                kernel_b_tensor = b_tensor
                kernel_b_sf_tensor = b_sf_tensor

            c_ld = output_tensor.shape[1] if output_tensor is not None else 0
            compiled_gemm(
                kernel_m,
                kernel_n,
                real_k,
                kernel_sf_m // 128,
                kernel_sf_n // 128,
                sf_k // 4,
                kernel_a_tensor.data_ptr(),
                kernel_b_tensor.data_ptr(),
                kernel_a_sf_tensor.data_ptr(),
                kernel_b_sf_tensor.data_ptr(),
                c_tensor.data_ptr(),
                alpha_tensor,
                c_ld,
            )

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)
            if output_tensor is None:
                if bias is not None:
                    if bias.ndim != 1 or bias.shape[0] != c_tensor.shape[-1]:
                        raise ValueError(
                            f"bias must be a 1-D tensor of shape [N]={c_tensor.shape[-1]}, "
                            f"got shape {tuple(bias.shape)}"
                        )
                    c_tensor = c_tensor + bias
                return c_tensor
            return

        def __init__(
            self,
            output_dtype: torch.dtype,
            to_userbuffers: bool = False,
            use_tvm_ffi: bool = True,
        ):
            super().__init__(
                output_dtype=output_dtype,
                to_userbuffers=to_userbuffers,
                use_tvm_ffi=use_tvm_ffi,
            )
            self._tvm_ffi_launch_plans = {}

    @functools.cache
    def _get_cute_dsl_mxfp8_rubin_alpha(device: torch.device) -> torch.Tensor:
        """Keep the scalar-one GEMM input alive for each CUDA device."""
        return torch.ones((), dtype=torch.float32, device=device)

    @functools.cache
    def _get_cute_dsl_dspark_mxfp8_rubin_runner(
        output_dtype: torch.dtype,
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkMXFP8RubinLinear:
        return CuteDSLDSparkMXFP8RubinLinear(
            output_dtype=output_dtype,
            use_tvm_ffi=use_tvm_ffi,
        )

    _mxfp8_prevalidated_launch_plans = {}

    def _materialize_dspark_tuning_profile_winners(
        custom_op: str,
        runner: TunableRunner,
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
    ) -> None:
        """Compile every cached DSpark bucket winner in this process."""
        tuner = AutoTuner.get()
        if not tuner.is_tuning_mode:
            return

        profiles = tuner._optimization_profiles(tuning_config, inputs)
        key = (
            custom_op,
            inputs[0].device,
            runner.unique_id(),
            tuple(profile.get_opt_shapes() for profile in profiles),
        )
        materialized = getattr(runner, "_dspark_materialized_profiles", None)
        if materialized is None:
            materialized = set()
            runner._dspark_materialized_profiles = materialized
        if key in materialized:
            return

        complete = True
        for profile in profiles:
            found, _, tactic, _ = tuner.profiling_cache.search_cache(
                custom_op,
                [runner],
                profile.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False,
            )
            if not found:
                complete = False
                continue
            profile_inputs = tuner._prepare_input_tensors(profile, inputs)
            runner(profile_inputs, tactic=tactic)
        if complete:
            materialized.add(key)

    def _run_cute_dsl_dspark_mxfp8_gemm_rubin_prevalidated(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
        materialize_launch: bool = False,
        tuning_config: Optional[TuningConfig] = None,
    ) -> torch.Tensor:
        """Run DSpark MXFP8 after bind-time contract validation."""
        alpha = _get_cute_dsl_mxfp8_rubin_alpha(input.device)
        runner = _get_cute_dsl_dspark_mxfp8_rubin_runner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        tuner = AutoTuner.get()
        materialized_key = None
        if materialize_launch and use_tvm_ffi and not tuner.is_tuning_mode:
            m, k = input.shape
            n = weight.shape[0]
            materialized_key = (
                input.device,
                output_dtype,
                m,
                n,
                k,
                input_scale.numel(),
                weight_scale.numel(),
            )
            launch_plan = _mxfp8_prevalidated_launch_plans.get(materialized_key)
            if launch_plan is not None:
                return runner._forward_tvm_ffi_prepared(inputs, launch_plan)

        active_tuning_config = tuning_config or runner.__class__.tuning_config
        custom_op = "trtllm::dspark_cute_dsl_mxfp8_gemm_rubin"
        _, best_tactic = tuner.choose_one(
            custom_op,
            [runner],
            active_tuning_config,
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        if tuning_config is not None and tuner.is_tuning_mode:
            _materialize_dspark_tuning_profile_winners(
                custom_op, runner, active_tuning_config, inputs
            )
        if materialized_key is not None:
            launch_plan_key = (
                best_tactic,
                m,
                n,
                k,
                input_scale.numel(),
                weight_scale.numel(),
                0,
                -1,
            )
            launch_plan = runner._tvm_ffi_launch_plans.get(launch_plan_key)
            if launch_plan is not None:
                _mxfp8_prevalidated_launch_plans[materialized_key] = launch_plan
        return output

    class CuteDSLDSparkContextWkvRubinLinear(CuteDSLDSparkMXFP8RubinLinear):
        """MXFP8 runner with operator-owned buckets for DSpark CTX M."""

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    0,
                    0,
                    _gen_dspark_context_wkv_m_profiles,
                    _map_dspark_context_wkv_m_profile,
                ),
            ),
            constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
            use_cuda_graph=True,
        )

    @functools.cache
    def _get_cute_dsl_dspark_context_wkv_rubin_runner(
        output_dtype: torch.dtype,
        use_tvm_ffi: bool,
    ) -> CuteDSLDSparkContextWkvRubinLinear:
        return CuteDSLDSparkContextWkvRubinLinear(
            output_dtype=output_dtype,
            use_tvm_ffi=use_tvm_ffi,
        )

    @torch.library.custom_op(
        "trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin_out",
        mutates_args=("output", "packed_scale"),
        device_types="cuda",
    )
    def cute_dsl_dspark_fp8_bmm_quantize_rubin_out(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        packed_scale: torch.Tensor,
        use_tvm_ffi: bool = True,
        dspark_max_batch: int = 0,
        dspark_rows_per_request: int = 0,
    ) -> None:
        if get_sm_version() != 107:
            raise ValueError("CuTe DSL FP8 BMM quantization requires SM107.")
        batch_size, m, _ = input.shape
        n = weight.shape[1]
        if n % 128 != 0:
            raise ValueError(f"Fused FP8 BMM quantization requires N divisible by 128, got {n}.")
        expected_scale_numel = pad_up(m, 128) * pad_up(ceil_div(batch_size * n, 32), 4)
        if output.shape != (batch_size, m, n):
            raise ValueError(
                "Fused FP8 BMM output must have shape "
                f"{(batch_size, m, n)}, got {tuple(output.shape)}."
            )
        if (
            output.dtype != torch.float8_e4m3fn
            or packed_scale.dtype != torch.uint8
            or not packed_scale.is_contiguous()
            or packed_scale.shape != (expected_scale_numel,)
        ):
            raise ValueError(
                "Fused FP8 BMM requires E4M3 output and a flat contiguous "
                f"uint8 R128c4 scale tensor with {expected_scale_numel} elements."
            )

        runner = _get_cute_dsl_dspark_fp8_rubin_bmm_quant_runner(use_tvm_ffi)
        inputs = [
            input,
            weight,
            input_scale,
            weight_scale,
            output,
            packed_scale,
        ]
        if dspark_max_batch > 0 and dspark_rows_per_request > 0:
            tuning_config = _get_dspark_generation_fp8_bmm_quant_tuning_config(
                dspark_rows_per_request,
                dspark_max_batch,
            )
            runner._dspark_generation_tuning_config = tuning_config
        else:
            tuning_config = getattr(
                runner, "_dspark_generation_tuning_config", runner.__class__.tuning_config
            )
        custom_op = "trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin::gemm"
        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            custom_op,
            [runner],
            tuning_config,
            inputs,
        )
        runner(inputs, tactic=best_tactic)
        if tuning_config is not runner.__class__.tuning_config and tuner.is_tuning_mode:
            _materialize_dspark_tuning_profile_winners(
                custom_op,
                runner,
                tuning_config,
                inputs,
            )

    @torch.library.register_fake("trtllm::cute_dsl_dspark_fp8_bmm_quantize_rubin_out")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        packed_scale: torch.Tensor,
        use_tvm_ffi: bool = True,
        dspark_max_batch: int = 0,
        dspark_rows_per_request: int = 0,
    ) -> None:
        del input_scale, weight_scale, use_tvm_ffi
        del dspark_max_batch, dspark_rows_per_request
        batch_size, m, _ = input.shape
        n = weight.shape[1]
        assert output.dtype == torch.float8_e4m3fn
        assert output.shape == (batch_size, m, n)
        expected_scale_numel = pad_up(m, 128) * pad_up(ceil_div(batch_size * n, 32), 4)
        assert packed_scale.dtype == torch.uint8
        assert packed_scale.shape == (expected_scale_numel,)
        assert packed_scale.is_contiguous()

    @torch.library.custom_op(
        "trtllm::cute_dsl_dspark_context_wkv_gemm_rubin",
        mutates_args=(),
        device_types="cuda",
    )
    def cute_dsl_dspark_context_wkv_gemm_rubin(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """Run stacked DSpark CTX WKV with operator-owned M buckets."""
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"DSpark context WKV GEMM only supports bfloat16 output, got {output_dtype}"
            )
        if (sm_version := get_sm_version()) != 107:
            raise ValueError(f"DSpark context WKV GEMM requires SM107, got SM{sm_version}.")
        alpha = _get_cute_dsl_mxfp8_rubin_alpha(input.device)
        runner = _get_cute_dsl_dspark_context_wkv_rubin_runner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, best_tactic = AutoTuner.get().choose_one(
            "trtllm::cute_dsl_dspark_context_wkv_gemm_rubin",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_dspark_context_wkv_gemm_rubin")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-2]
        return mat_a.new_empty(shape, dtype=torch.bfloat16)
