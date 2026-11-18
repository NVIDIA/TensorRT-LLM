# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV-only FP4 MLA generation cache update of K3's fused query.

:func:`fp4_mla_kv_append_generation` is the FP4 MLA backend's fused generation cache update
(RoPE on the new K tail, FP4 tiles, HP ring, V scales and packed V) without the query rows: it
prepares the packed-Q buffers of a query of ``num_q_heads`` heads on ``metadata.fp4_mla_state``
and leaves their rows to the caller (``k3_mla_q_fused`` writes them on another stream). The Triton
kernel is the backend's ``_fp4_mla_generation_fused_qk_rope_cache_update_kernel`` with a
``WRITE_Q_ROWS`` switch; launched with it off and a grid of the KV programs only, it writes the
cache bytes of the full update.
"""

from typing import Any, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    _FP4_MLA_CUTEDSL_BACKEND,
    _FP4_MLA_K_RESIDUAL_BACKENDS,
    FP4_BLOCK_SIZE,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_Q1_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_DIM,
    FP4_MLA_Q_SF_GROUPS,
    FP4_MLA_SCALE_ROW_GROUP,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
    _fp4_mla_packed_rows_per_seq,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_hp_pool_layout,
    _get_fp4_mla_kv_cache_tensors,
    _get_fp4_mla_q_global_scale,
    _validate_fp4_mla_cache_shape,
    _validate_fp4_mla_hp_generation_width,
    _validate_fp4_mla_kv_storage_shape,
    get_fp4_mla_v_scale_pool_view,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _fp4_mla_generation_hp_page_ids,
    _fp4_mla_generation_num_blocks_device,
    _fp4_mla_generation_page_ids,
    _fp4_mla_uniform_generation_lengths,
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.packed_plan import _fp4_mla_helix_plan_bounds

# The backend's plan geometry with the packed decode's task slots per request.
from ...ops.attention._fp4_mla_decode import backend_plan_geometry

# The family copies of the FP4 MLA cache-update code outside the KV-cache-manager contract.
from ...ops.attention._k3_fp4_mla_cache import (
    _get_cutedsl_persistent_v_packed_cache,
    _get_fp4_mla_v_packed_pool_base,
    _maybe_update_triton_v_packed_cache,
    _prepare_fp4_mla_q_buffers,
    _repack_cutedsl_v_packed_cache,
)
from ...ops.attention._k3_fp4_mla_kernels import (
    _fp4_mla_generation_q1_kv_tail8_update,
    _fp4_mla_generation_q1_kv_tile_update,
    _fp4_mla_q_sf_offset,
    _fp4_mla_q_sf_offset_h128,
    _fp4_mla_swizzled_sf_offset,
)

# ---------------------------------------------------------------------------
# Launch policy and element-wise FP4 helpers of the FP4 MLA generation update (the backend's
# ``fp4_mla.config`` / ``fp4_mla.fp4_mla_kernels``), used by this module's kernel.
# ---------------------------------------------------------------------------

_FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH = 4
_FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH = 16
_FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH = 32
_FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD = 256
_FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD = 512
_FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD = 640
_FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD = 768


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _fp4_mla_q1_kv_blocks_per_program(num_gen: int, v_head_dim: int) -> int:
    """Select the Q1 KV work per program without changing the launch boundary."""
    large_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH * FP4_BLOCK_SIZE
    if num_gen >= _FP4_MLA_Q1_KV_LARGE_BATCH_THRESHOLD and v_head_dim % large_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    medium_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH * FP4_BLOCK_SIZE
    if (
        num_gen >= _FP4_MLA_Q1_KV_MEDIUM_BATCH_THRESHOLD
        and v_head_dim % medium_batch_block_dim == 0
    ):
        return _FP4_MLA_Q1_KV_BLOCKS_MEDIUM_BATCH
    small_batch_block_dim = _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH * FP4_BLOCK_SIZE
    if v_head_dim % small_batch_block_dim == 0:
        return _FP4_MLA_Q1_KV_BLOCKS_SMALL_BATCH
    return 1


def _fp4_mla_q1_prefix_blocks_per_program(
    num_gen: int,
    q1_kv_blocks_per_program: int,
) -> int:
    """Select rolled Q-prefix work only when the batch keeps it efficient."""
    max_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // FP4_MLA_Q1_PREFIX_BLOCK_DIM
    if (
        num_gen >= _FP4_MLA_Q1_PREFIX_PAIR_BATCH_THRESHOLD
        and q1_kv_blocks_per_program == _FP4_MLA_Q1_KV_BLOCKS_LARGE_BATCH
    ):
        if num_gen >= _FP4_MLA_Q1_PREFIX_GROUP4_BATCH_THRESHOLD:
            return min(4, max_prefix_blocks)
        return min(2, max_prefix_blocks)
    return 1


@triton.jit
def _fp4_mla_pack_e2m1x2_rne(even, odd):
    """Match cvt.rn.satfinite.e2m1x2.f32 used by arcquantFP4.cu."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            cvt.rn.satfinite.e2m1x2.f32 packed0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 packed1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 packed2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 packed3, $8, $4;
            mov.b32 $0, {packed0, packed1, packed2, packed3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[even.to(tl.float32), odd.to(tl.float32)],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _fp4_mla_pack_e2m1x8_rne(even0, even1, even2, even3, odd0, odd1, odd2, odd3):
    """Pack four consecutive E2M1x2 bytes into one word."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            cvt.rn.satfinite.e2m1x2.f32 packed0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 packed1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 packed2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 packed3, $8, $4;
            mov.b32 $0, {packed0, packed1, packed2, packed3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[even0, even1, even2, even3, odd0, odd1, odd2, odd3],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_unpack_e2m1x2(packed):
    """Decode packed E2M1 through the native GR100 conversion path."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            .reg .b16 low, high;
            .reg .b32 half2;
            mov.b32 {packed0, packed1, packed2, packed3}, $2;
            cvt.rn.f16x2.e2m1x2 half2, packed0;
            mov.b32 {low, high}, half2;
            cvt.f32.f16 $0, low;
            cvt.f32.f16 $1, high;
        }
        """,
        constraints="=f,=f,r",
        args=[packed.to(tl.uint32)],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_rcp_approx_ftz(value):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_rope_fp32(even, odd, cos, sin):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 even_cos, odd_sin, odd_cos, even_sin;
            mul.rn.f32 even_cos, $2, $4;
            mul.rn.f32 odd_sin, $3, $5;
            sub.rn.f32 $0, even_cos, odd_sin;
            mul.rn.f32 odd_cos, $3, $4;
            mul.rn.f32 even_sin, $2, $5;
            add.rn.f32 $1, odd_cos, even_sin;
        }
        """,
        constraints="=f,=f,f,f,f,f",
        args=[even, odd, cos, sin],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_quantize_q_groups(values_even, values_odd):
    amax = tl.max(
        tl.maximum(tl.abs(values_even), tl.abs(values_odd)),
        axis=2,
    )
    scale = tl.maximum(amax / 6.0, 0.001953125)
    scale_fp8 = scale.to(tl.float8e4nv)
    inv_scale = _fp4_mla_rcp_approx_ftz(scale_fp8.to(tl.float32))
    packed = _fp4_mla_pack_e2m1x2_rne(
        values_even * inv_scale[:, :, None],
        values_odd * inv_scale[:, :, None],
    )
    return packed, scale_fp8


@triton.jit
def _fp4_mla_quantize_q_groups_words(
    values_even,
    values_odd,
    BLOCK_Q_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Quantize FP4 groups while retaining native packed-word ownership."""
    tl.static_assert(values_even.shape[2] == 8)
    tl.static_assert(values_odd.shape[2] == 8)
    amax = tl.max(
        tl.maximum(tl.abs(values_even), tl.abs(values_odd)),
        axis=2,
    )
    scale = tl.maximum(amax / 6.0, 0.001953125)
    scale_fp8 = scale.to(tl.float8e4nv)
    inv_scale = _fp4_mla_rcp_approx_ftz(scale_fp8.to(tl.float32))
    scaled_even = tl.reshape(
        values_even * inv_scale[:, :, None],
        (BLOCK_Q_HEADS, GROUPS, 2, 2, 2),
    )
    scaled_odd = tl.reshape(
        values_odd * inv_scale[:, :, None],
        (BLOCK_Q_HEADS, GROUPS, 2, 2, 2),
    )
    even_02, even_13 = tl.split(scaled_even)
    odd_02, odd_13 = tl.split(scaled_odd)
    even0, even2 = tl.split(even_02)
    even1, even3 = tl.split(even_13)
    odd0, odd2 = tl.split(odd_02)
    odd1, odd3 = tl.split(odd_13)
    packed_words = _fp4_mla_pack_e2m1x8_rne(
        even0,
        even1,
        even2,
        even3,
        odd0,
        odd1,
        odd2,
        odd3,
    )
    return packed_words, scale_fp8


@triton.jit
def _fp4_mla_quantize_q_residual_groups(values_even, values_odd):
    packed, scale_fp8 = _fp4_mla_quantize_q_groups(values_even, values_odd)
    rounded_scale = scale_fp8.to(tl.float32)
    main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
    residual_even = values_even - main_even * rounded_scale[:, :, None]
    residual_odd = values_odd - main_odd * rounded_scale[:, :, None]
    residual_packed, residual_scale_fp8 = _fp4_mla_quantize_q_groups(residual_even, residual_odd)
    return packed, scale_fp8, residual_packed, residual_scale_fp8


@triton.jit
def _fp4_e2m1_to_f32(nibble):
    magnitude = nibble & 0x7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    sign = (nibble & 0x8) != 0
    return tl.where(sign, -value, value)


@triton.jit
def _fp4_mla_floor_half_up(abs_value, multiplier, bias):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 rounded;
            fma.rz.f32 rounded, $1, $2, $3;
            cvt.rzi.s32.f32 $0, rounded;
        }
        """,
        constraints="=r,f,f,f",
        args=[abs_value, multiplier, bias],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_e2m1_quantize(x):
    abs_x = tl.abs(x)
    bounded = tl.where(abs_x < 6.0, abs_x, 6.0)
    small = bounded < 2.0
    multiplier = tl.where(small, 2.0, 1.0)
    bias = tl.where(small, 0.5, 2.5)
    magnitude = _fp4_mla_floor_half_up(bounded, multiplier, bias)
    magnitude = tl.minimum(magnitude, 6) + (~(abs_x < 5.0)).to(tl.int32)
    sign = tl.where(x < 0.0, 8, 0)
    return (magnitude | sign).to(tl.uint8)


def fp4_mla_kv_append_generation(
    mqa: Any,
    latent_cache: torch.Tensor,
    metadata: Any,
    num_q_heads: int,
) -> None:
    """Append this step's latent rows of the FP4 MLA layer behind ``mqa`` (its
    ``TrtllmAttention``) to the FP4 cache and HP ring, and prepare the packed-Q buffers of
    ``num_q_heads`` heads on ``metadata`` for the decode, without writing the query rows."""
    assert mqa.kv_lora_rank is not None
    assert mqa.qk_rope_head_dim is not None
    if not mqa.uses_fp4_mla_attention:
        raise RuntimeError("The FP4 MLA KV append requires FP4 MLA attention.")
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    if latent_cache.shape[-1] != mqa.kv_lora_rank + mqa.qk_rope_head_dim:
        raise RuntimeError(
            "FP4 MLA latent_cache last dimension must be "
            f"{mqa.kv_lora_rank + mqa.qk_rope_head_dim}, got {latent_cache.shape[-1]}."
        )
    if (
        mqa.rotary_cos_sin is None
        or latent_cache.dtype != torch.bfloat16
        or mqa.rotary_cos_sin.dtype != torch.float32
    ):
        raise RuntimeError(
            "FP4 MLA generation requires a BF16 latent cache and an FP32 rotary table."
        )
    hp_pool_updated = _scatter_generation_kv_only(
        metadata,
        latent_cache,
        mqa.layer_idx,
        token_offset=getattr(metadata, "num_ctx_tokens", 0),
        local_layer=mqa.get_fp4_mla_local_layer_idx(metadata),
        v_head_dim=mqa.kv_lora_rank,
        rotary_cos_sin=mqa.rotary_cos_sin,
        helix_position_offsets=metadata.helix_position_offsets,
        helix_is_inactive_rank=metadata.helix_is_inactive_rank,
        num_q_heads=num_q_heads,
    )
    if not hp_pool_updated:
        raise RuntimeError("FP4 MLA generation KV append did not update the HP pool.")


def _scatter_generation_kv_only(
    metadata: Any,
    latent_cache: torch.Tensor,
    layer_idx: int,
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    rotary_cos_sin: torch.Tensor,
    helix_position_offsets: torch.Tensor | None,
    helix_is_inactive_rank: torch.Tensor | None,
    num_q_heads: int,
) -> bool:
    """The generation phase of the backend's ``scatter_fp4_mla_kv_cache`` with the query rows
    left to the caller; returns whether it updated the HP pool."""
    state = metadata.fp4_mla_state
    state.generation_cache_scattered = False
    state.prequantized_q = None
    state.prequantized_q_sf = None
    state.q_batch_capacity = None
    state.prequantized_q_rows_per_seq = None
    if latent_cache.numel() == 0:
        raise ValueError("FP4 MLA cache scatter requires at least one latent token.")

    latent_cache = latent_cache.reshape(latent_cache.shape[0], -1).contiguous()
    num_tokens = latent_cache.shape[0]
    head_dim = latent_cache.shape[-1]
    if head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must be divisible by {FP4_BLOCK_SIZE}, got {head_dim}."
        )
    indices_len = state.batch_indices.shape[0]
    positions_len = state.positions.shape[0]
    if token_offset + num_tokens > indices_len or token_offset + num_tokens > positions_len:
        raise RuntimeError(
            f"FP4 MLA scatter would read batch_indices[{token_offset}:"
            f"{token_offset + num_tokens}] / positions[{token_offset}:"
            f"{token_offset + num_tokens}], but only {indices_len} / "
            f"{positions_len} entries are available: latent_cache must hold the generation "
            "tokens only."
        )

    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)

    backend = _fp4_mla_attention_backend()
    global_scale = _get_fp4_mla_global_scale(metadata, latent_cache.device)
    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=backend,
    )
    sf_per_token = storage_head_dim // FP4_BLOCK_SIZE

    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    if getattr(kv_cache_manager, "fp4_mla_context_only_cache", False) is True:
        raise RuntimeError("FP4 MLA context-only cache cannot execute generation.")
    if getattr(state, "v_scale_pool", None) is None:
        raise RuntimeError("FP4 MLA scatter requires the auxiliary V scale pool.")
    if metadata.page_size % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA scatter requires page_size divisible by "
            f"{FP4_BLOCK_SIZE}, got {metadata.page_size}."
        )
    if v_head_dim > head_dim:
        raise ValueError(f"FP4 MLA v_head_dim={v_head_dim} cannot exceed head_dim={head_dim}.")
    if head_dim - v_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual quantization requires the K-only tail to match "
            f"the {FP4_MLA_K_RESIDUAL_DIM}-channel residual, got "
            f"head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if v_head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA v_head_dim must be divisible by {FP4_BLOCK_SIZE}, got {v_head_dim}."
        )

    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
    num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE

    if (helix_position_offsets is None) != (helix_is_inactive_rank is None):
        raise ValueError(
            "FP4 MLA Helix position-offset and inactive-rank metadata must be provided together."
        )
    if rotary_cos_sin is None:
        raise ValueError("FP4 MLA generation with external query rows requires rotary_cos_sin.")
    num_gen_for_q = metadata.num_seqs - metadata.num_contexts
    q_rows_per_seq = 0
    if num_gen_for_q > 0 and num_tokens % num_gen_for_q == 0:
        q_rows_per_seq = _fp4_mla_packed_rows_per_seq(num_q_heads, num_tokens // num_gen_for_q)
    q_fp4_out, q_sf_out, q_batch_capacity = _prepare_fp4_mla_q_buffers(
        metadata,
        num_tokens,
        num_q_heads,
        latent_cache.device,
        rows_per_seq=q_rows_per_seq,
    )
    state.prequantized_q = q_fp4_out
    state.prequantized_q_sf = q_sf_out
    state.q_batch_capacity = q_batch_capacity
    state.prequantized_q_rows_per_seq = q_rows_per_seq
    hp_pool_updated = True

    cutedsl_backend = backend == _FP4_MLA_CUTEDSL_BACKEND
    fused_v_transpose = cutedsl_backend and _fp4_mla_cutedsl_fused_v_transpose_enabled()
    persistent_v_packed = None
    v_packed_base = None
    v_page_offset = 0
    direct_v_packed_write = False
    if cutedsl_backend and not fused_v_transpose:
        persistent_v_packed = _get_cutedsl_persistent_v_packed_cache(
            metadata,
            local_layer,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
        )
        # Reused/imported pages may not carry this process-local sidecar. Write
        # packed V directly only when the cache pages are exclusively owned.
        num_gen = metadata.num_seqs - metadata.num_contexts
        block_reuse = getattr(kv_cache_manager, "enable_block_reuse", True)
        direct_v_packed_write = (
            num_gen > 0
            and num_tokens >= num_gen
            and num_tokens % num_gen == 0
            and block_reuse is False
        )
        if direct_v_packed_write:
            v_packed_base = _get_fp4_mla_v_packed_pool_base(metadata)
            get_v_page_offset = getattr(kv_cache_manager, "get_mla_v_packed_page_offset", None)
            v_page_offset = (
                int(get_v_page_offset(local_layer))
                if callable(get_v_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            expected_row_width = metadata.page_size // 2
            required_base_rows = (v_page_offset + kv_cache.shape[0]) * v_head_dim
            if (
                not isinstance(v_packed_base, torch.Tensor)
                or v_packed_base.dtype != torch.uint8
                or v_packed_base.device != kv_cache.device
                or v_packed_base.ndim != 2
                or v_packed_base.shape[0] < required_base_rows
                or v_packed_base.shape[1] != expected_row_width
                or not v_packed_base.is_contiguous()
            ):
                raise RuntimeError(
                    "FP4 MLA direct V-packed cache update requires the "
                    "stable full-pool base to be a contiguous uint8 tensor "
                    f"with at least {required_base_rows} rows and "
                    f"{expected_row_width} columns on {kv_cache.device}."
                )
            expected_layer_ptr = v_packed_base.data_ptr() + (
                v_page_offset * v_head_dim * expected_row_width
            )
            if expected_layer_ptr != persistent_v_packed.data_ptr():
                raise RuntimeError(
                    "FP4 MLA V-packed layer view does not match its stable "
                    "full-pool base and page offset."
                )

    generation_state = _scatter_generation_kv_only_2d(
        metadata,
        latent_cache,
        kv_cache,
        sf_cache,
        v_sf,
        global_scale,
        token_offset=token_offset,
        local_layer=local_layer,
        v_head_dim=v_head_dim,
        head_dim=head_dim,
        num_tokens=num_tokens,
        num_dim_blocks=num_dim_blocks,
        sf_per_token=sf_per_token,
        sf_per_page=sf_per_page,
        rotary_cos_sin=rotary_cos_sin,
        num_q_heads=num_q_heads,
        q_fp4_out=q_fp4_out,
        q_sf_out=q_sf_out,
        v_packed_base=v_packed_base,
        v_page_offset=v_page_offset,
        helix_position_offsets=helix_position_offsets,
        helix_is_inactive_rank=helix_is_inactive_rank,
    )
    v_pack_page_ids = _fp4_mla_generation_page_ids(
        metadata, metadata.num_seqs - metadata.num_contexts
    )
    v_pack_num_valid = None
    if getattr(metadata, "is_cuda_graph", False):
        # Frozen launch grids cannot follow the per-replay page count;
        # the repack kernels stride over this device-side count instead.
        v_pack_num_valid = _fp4_mla_generation_num_blocks_device(metadata)
    if cutedsl_backend and not fused_v_transpose and not direct_v_packed_write:
        cutedsl_repack_page_indptr = None
        cutedsl_repack_kv_lens = None
        cutedsl_repack_generation_lens = None
        cutedsl_repack_max_touched_pages = 1
        if generation_state is not None:
            kv_lens_gen, gen_lens_gen, generation_page_ids = generation_state
            num_gen = kv_lens_gen.numel()
            cutedsl_v_pack_page_ids = generation_page_ids
            cutedsl_repack_page_indptr = state.paged_kv_indptr_decode
            cutedsl_repack_kv_lens = kv_lens_gen
            cutedsl_repack_generation_lens = gen_lens_gen
            cutedsl_repack_max_touched_pages = _ceil_div(
                num_tokens // num_gen + metadata.page_size - 1,
                metadata.page_size,
            )
        else:
            cutedsl_v_pack_page_ids = v_pack_page_ids
        _repack_cutedsl_v_packed_cache(
            persistent_v_packed,
            kv_cache,
            cutedsl_v_pack_page_ids,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
            page_indptr=cutedsl_repack_page_indptr,
            kv_lens=cutedsl_repack_kv_lens,
            generation_lens=cutedsl_repack_generation_lens,
            max_touched_pages=cutedsl_repack_max_touched_pages,
        )
    _maybe_update_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_pack_page_ids,
        num_queries=num_tokens,
        v_head_dim=v_head_dim,
        page_size=metadata.page_size,
        local_layer=local_layer,
        v_sf=v_sf[local_layer],
        num_valid_pages=v_pack_num_valid,
    )
    state.generation_cache_scattered = hp_pool_updated
    return hp_pool_updated


def _scatter_generation_kv_only_2d(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
    rotary_cos_sin: torch.Tensor,
    num_q_heads: int,
    q_fp4_out: torch.Tensor,
    q_sf_out: torch.Tensor,
    v_packed_base: Optional[torch.Tensor],
    v_page_offset: int,
    helix_position_offsets: Optional[torch.Tensor],
    helix_is_inactive_rank: Optional[torch.Tensor],
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The generation KV update of a query of ``num_q_heads`` heads whose rows the caller writes."""
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return
    if num_tokens < num_gen:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {num_gen} generation "
            f"tokens, got {num_tokens}."
        )
    if num_tokens % num_gen != 0:
        raise NotImplementedError(
            "FP4 MLA no-dequant generation scatter requires a uniform linear MTP "
            f"generation length, got {num_tokens} tokens for {num_gen} sequences."
        )
    # The prompt_lens/kv_lens runtime aliases can lag at the decode anchor
    # (seq_lens == 1) under CUDA graph / one-engine MTP while each generation
    # sequence really appends num_tokens // num_gen tokens this step. Recover the
    # true per-sequence lengths for the no-dequant kernel below (a no-op when the
    # aliases already match). As the first FP4 MLA kernel of the step, the same
    # launch also writes the packed split plan the decode layers reuse.
    plan_geometry = None
    if num_q_heads > 0:
        plan_geometry = backend_plan_geometry(
            metadata,
            num_heads=num_q_heads,
            query_len=num_tokens // num_gen,
            num_requests=num_gen,
            device=latent_cache.device,
            helix_kv_bounds=_fp4_mla_helix_plan_bounds(
                metadata,
                num_tokens,
                use_helix=helix_position_offsets is not None or helix_is_inactive_rank is not None,
            ),
        )
    kv_lens_gen, gen_lens_gen = _fp4_mla_uniform_generation_lengths(
        metadata, num_tokens, num_gen, plan_geometry=plan_geometry
    )
    _materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens_gen)

    pool = getattr(metadata.fp4_mla_state, "hp_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA 2D generation scatter requires the HP KV pool.")
    try:
        hp_pool_size, hp_head_dim = _get_fp4_mla_hp_pool_layout(metadata, pool)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if hp_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {head_dim} HP channels, got "
            f"{hp_head_dim}."
        )
    hp_page_ids = _fp4_mla_generation_hp_page_ids(metadata, num_gen)
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA generation requires HP page metadata.")
    num_hp_pages = pool.shape[0]

    max_gen_len = num_tokens // num_gen
    use_helix = helix_position_offsets is not None or helix_is_inactive_rank is not None
    use_helix_local_slots = use_helix and bool(getattr(metadata, "_helix_spec_tokens_valid", False))
    if use_helix:
        if helix_position_offsets is None or helix_is_inactive_rank is None:
            raise RuntimeError(
                "FP4 MLA Helix requires both position-offset and inactive-rank metadata."
            )
        if max_gen_len != 1 and not use_helix_local_slots:
            raise NotImplementedError(
                "FP4 MLA multi-token Helix requires speculative per-token metadata."
            )
        if (
            helix_position_offsets.dtype != torch.int32
            or helix_position_offsets.device != latent_cache.device
            or helix_position_offsets.ndim != 1
            or helix_position_offsets.numel() < num_tokens
            or not helix_position_offsets.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix position offsets must be a contiguous same-device "
                "int32 tensor covering every generation token."
            )
        if (
            helix_is_inactive_rank.dtype != torch.bool
            or helix_is_inactive_rank.device != latent_cache.device
            or helix_is_inactive_rank.ndim != 1
            or helix_is_inactive_rank.numel() < num_gen
            or not helix_is_inactive_rank.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix inactive-rank metadata must be a contiguous "
                "same-device bool tensor covering every generation sequence."
            )
        if use_helix_local_slots:
            helix_local_slots = getattr(metadata, "helix_local_slots", None)
            if (
                not isinstance(helix_local_slots, torch.Tensor)
                or helix_local_slots.dtype != torch.int32
                or helix_local_slots.device != latent_cache.device
                or helix_local_slots.ndim != 1
                or helix_local_slots.numel() < num_tokens
                or not helix_local_slots.is_contiguous()
            ):
                raise ValueError(
                    "FP4 MLA speculative Helix local slots must be a contiguous "
                    "same-device int32 tensor covering every generation token."
                )
        else:
            helix_local_slots = helix_position_offsets
    else:
        helix_position_offsets = kv_lens_gen
        helix_local_slots = kv_lens_gen
        helix_is_inactive_rank = gen_lens_gen
    _validate_fp4_mla_hp_generation_width(hp_pool_size, max_gen_len)
    page_ids = _fp4_mla_generation_page_ids(metadata, num_gen)
    rope_dim = head_dim - v_head_dim
    block_q_heads = 32
    q1_kv_blocks_per_program = 1
    # Grouped Q1 kernels reuse K's packed codes for V. Keep one dimension
    # block per program until their warp-specialized V quantizer is split out.
    if max_gen_len == 1:
        q1_kv_blocks_per_program = _fp4_mla_q1_kv_blocks_per_program(num_gen, v_head_dim)
    q_prefix_block_dim = (
        FP4_MLA_Q1_PREFIX_BLOCK_DIM if max_gen_len == 1 else FP4_MLA_Q_PREFIX_BLOCK_DIM
    )
    q_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // q_prefix_block_dim
    q_prefix_blocks_per_program = _fp4_mla_q1_prefix_blocks_per_program(
        num_gen,
        q1_kv_blocks_per_program,
    )
    q_work_blocks = q_prefix_blocks // q_prefix_blocks_per_program + 1  # noqa: F841
    if latent_cache.dtype != torch.bfloat16:
        raise TypeError(
            "Fused FP4 MLA Q/K RoPE and cache storage requires BF16 latent KV, "
            f"got {latent_cache.dtype}."
        )
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a positive even K tail, "
            f"got head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if (
        rotary_cos_sin is None
        or rotary_cos_sin.device != latent_cache.device
        or rotary_cos_sin.dtype != torch.float32
        or rotary_cos_sin.numel() % (rope_dim * 2) != 0
    ):
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a same-device FP32 rotary "
            f"table with rows of {rope_dim * 2} values."
        )
    if num_q_heads <= 0:
        raise ValueError(
            f"FP4 MLA generation needs a positive query head count, got {num_q_heads}."
        )
    q_head_blocks = _ceil_div(num_q_heads, block_q_heads)
    q_rows_per_seq = _fp4_mla_packed_rows_per_seq(num_q_heads, max_gen_len)
    if q_rows_per_seq and q_fp4_out.shape[0] < num_gen * q_rows_per_seq:
        raise RuntimeError(
            "FP4 MLA packed Q buffer is smaller than the generation batch: "
            f"{q_fp4_out.shape[0]} rows < {num_gen} x {q_rows_per_seq}."
        )
    max_gen_tiles = _ceil_div(max_gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
    rotary_table = rotary_cos_sin
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, latent_cache.device)
    # Without query rows the kernel reads none of the three; any device tensor fills their slots.
    q_pe_input = latent_cache
    q_rope_output = latent_cache
    q_full_input = latent_cache
    q_fp4_output = q_fp4_out
    q_sf_output = q_sf_out
    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0
    kv_work_blocks = (
        v_head_dim // FP4_BLOCK_SIZE // q1_kv_blocks_per_program + 1
        if max_gen_len == 1
        else num_dim_blocks
    )
    launch_grid = (num_gen, kv_work_blocks)
    store_k_residual = _fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS

    def launch_generation_update(
        grid: tuple[int, ...],
        *,
        page_ids_len: int,
        indptr_len: int,
        max_gen_tiles_variant: int,
        q_prefix_block_dim_variant: int,
        q_prefix_blocks_variant: int,
        q_prefix_blocks_per_program_variant: int,
        q1_kv_blocks_per_program_variant: int,
    ) -> None:
        q_work_blocks_variant = q_prefix_blocks_variant // q_prefix_blocks_per_program_variant + 1
        _fp4_mla_generation_kv_update_kernel[grid](
            kv_cache,
            sf_cache,
            v_sf,
            v_packed_output,
            pool,
            latent_cache,
            global_scale,
            q_global_scale,
            rotary_table,
            q_pe_input,
            q_rope_output,
            q_full_input,
            q_fp4_output,
            q_sf_output,
            kv_lens_gen,
            gen_lens_gen,
            helix_position_offsets,
            helix_local_slots,
            helix_is_inactive_rank,
            page_ids,
            hp_page_ids,
            metadata.fp4_mla_state.paged_kv_indptr_decode,
            page_ids_len,
            hp_page_ids.numel(),
            indptr_len,
            v_sf.shape[1],
            num_hp_pages,
            v_sf.shape[0],
            local_layer,
            v_page_offset if write_v_packed else 0,
            metadata.page_size,
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            sf_cache.stride(0),
            pool.stride(0),
            pool.stride(1),
            v_sf.stride(0),
            v_sf.stride(1),
            v_packed_s0,
            v_packed_s1,
            0,
            0,
            0,
            0,
            0,
            0,
            HEAD_D=hp_head_dim,
            V_HEAD_D=v_head_dim,
            HP_BLOCK=FP4_BLOCK_SIZE,
            HP_POOL_SIZE=hp_pool_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_TOKEN=sf_per_token,
            SF_PER_PAGE=sf_per_page,
            K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
            STORE_K_RESIDUAL=store_k_residual,
            FUSE_ROPE_CACHE_STORE=True,
            USE_HELIX=use_helix,
            USE_HELIX_LOCAL_SLOTS=use_helix_local_slots,
            WRITE_V_PACKED=write_v_packed,
            MAX_GEN_TILES=max_gen_tiles_variant,
            ROPE_DIM=rope_dim,
            ROPE_PAIR_BLOCK=triton.next_power_of_2(rope_dim // 2),
            NUM_DIM_BLOCKS=num_dim_blocks,
            NUM_Q_HEADS=num_q_heads,
            Q_HEAD_BLOCKS=max(q_head_blocks, 1),
            BLOCK_Q_HEADS=block_q_heads,
            Q_PREFIX_D=FP4_MLA_Q_PREFIX_DIM,
            Q_PREFIX_BLOCK_D=q_prefix_block_dim_variant,
            Q_PREFIX_BLOCKS=q_prefix_blocks_variant,
            Q_PREFIX_BLOCKS_PER_PROGRAM=q_prefix_blocks_per_program_variant,
            Q_WORK_BLOCKS=q_work_blocks_variant,
            Q_SF_COLS=FP4_MLA_Q_SF_GROUPS,
            WRITE_Q=True,
            Q1_KV_BLOCKS_PER_PROGRAM=q1_kv_blocks_per_program_variant,
            Q_ROWS_PER_SEQ=q_rows_per_seq,
            WRITE_Q_ROWS=False,
            maxnreg=56,
        )

    launch_generation_update(
        launch_grid,
        page_ids_len=page_ids.shape[0],
        indptr_len=metadata.fp4_mla_state.paged_kv_indptr_decode.shape[0],
        max_gen_tiles_variant=max(max_gen_tiles, 1),
        q_prefix_block_dim_variant=q_prefix_block_dim,
        q_prefix_blocks_variant=q_prefix_blocks,
        q_prefix_blocks_per_program_variant=q_prefix_blocks_per_program,
        q1_kv_blocks_per_program_variant=q1_kv_blocks_per_program,
    )
    return kv_lens_gen, gen_lens_gen, page_ids


@triton.jit(
    do_not_specialize=[
        "page_ids_len",
        "hp_page_ids_len",
        "indptr_len",
        "num_pages",
        "num_hp_pages",
        "num_layers",
        "local_layer",
        "v_page_offset",
    ],
    do_not_specialize_on_alignment=[
        "page_ids_len",
        "hp_page_ids_len",
        "indptr_len",
        "num_pages",
        "num_hp_pages",
        "num_layers",
        "local_layer",
        "v_page_offset",
    ],
)
def _fp4_mla_generation_kv_update_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    hp_pool_ptr,
    latent_cache_ptr,
    global_scale_ptr,
    q_global_scale_ptr,
    rotary_cos_sin_ptr,
    q_pe_ptr,
    q_rope_out_ptr,
    q_full_ptr,
    q_fp4_out_ptr,
    q_sf_out_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    helix_position_offsets_ptr,
    helix_local_slots_ptr,
    helix_is_inactive_rank_ptr,
    page_ids_ptr,
    hp_page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    hp_page_ids_len,
    indptr_len,
    num_pages,
    num_hp_pages,
    num_layers,
    local_layer,
    v_page_offset,
    page_size: tl.constexpr,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    vsf_s0,
    vsf_s1,
    v_packed_s0,
    v_packed_s1,
    q_pe_s0,
    q_pe_s1,
    q_pe_s2,
    q_out_s0,
    q_out_s1,
    q_out_s2,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    HP_POOL_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    K_RESIDUAL_D: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    FUSE_ROPE_CACHE_STORE: tl.constexpr,
    USE_HELIX: tl.constexpr,
    USE_HELIX_LOCAL_SLOTS: tl.constexpr,
    WRITE_V_PACKED: tl.constexpr,
    MAX_GEN_TILES: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE_PAIR_BLOCK: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    Q_HEAD_BLOCKS: tl.constexpr,
    BLOCK_Q_HEADS: tl.constexpr,
    Q_PREFIX_D: tl.constexpr,
    Q_PREFIX_BLOCK_D: tl.constexpr,
    Q_PREFIX_BLOCKS: tl.constexpr,
    Q_PREFIX_BLOCKS_PER_PROGRAM: tl.constexpr,
    Q_WORK_BLOCKS: tl.constexpr,
    Q_SF_COLS: tl.constexpr,
    WRITE_Q: tl.constexpr,
    Q1_KV_BLOCKS_PER_PROGRAM: tl.constexpr,
    Q_ROWS_PER_SEQ: tl.constexpr = 0,
    WRITE_Q_ROWS: tl.constexpr = True,
):
    q1_shared_main: tl.constexpr = FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1
    q1_grouped_kv: tl.constexpr = q1_shared_main and Q1_KV_BLOCKS_PER_PROGRAM > 1
    q1_main_kv_programs: tl.constexpr = V_HEAD_D // FP4_BLOCK // Q1_KV_BLOCKS_PER_PROGRAM
    q_prefix_programs: tl.constexpr = Q_PREFIX_BLOCKS // Q_PREFIX_BLOCKS_PER_PROGRAM
    q1_kv_blocks_per_iteration: tl.constexpr = 2
    tl.static_assert(Q1_KV_BLOCKS_PER_PROGRAM == 1 or q1_grouped_kv)
    tl.static_assert(
        Q1_KV_BLOCKS_PER_PROGRAM == 1 or Q1_KV_BLOCKS_PER_PROGRAM % q1_kv_blocks_per_iteration == 0
    )
    tl.static_assert(V_HEAD_D % (FP4_BLOCK * Q1_KV_BLOCKS_PER_PROGRAM) == 0)
    tl.static_assert(Q_PREFIX_BLOCKS % Q_PREFIX_BLOCKS_PER_PROGRAM == 0)
    seq_idx = tl.program_id(0)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return
    gen_len = tl.load(prompt_lens_ptr + seq_idx)
    if gen_len <= 0:
        return
    kv_len = tl.load(kv_lens_ptr + seq_idx)
    # Generation tokens are request-major. Deriving their absolute position
    # here avoids materializing per-token metadata in a CUDA graph.
    first_new_pos = kv_len - gen_len
    rope_first_new_pos = first_new_pos
    if USE_HELIX:
        rope_first_new_pos = tl.load(helix_position_offsets_ptr + seq_idx * gen_len)
    if USE_HELIX_LOCAL_SLOTS:
        # Helix owns a contiguous rank-local subset of a linear verify group,
        # but that subset can begin at any token in the group.
        first_new_pos = kv_len
        for token_idx in tl.static_range(0, MAX_GEN_TILES * HP_BLOCK):
            local_slot = tl.load(
                helix_local_slots_ptr + seq_idx * gen_len + token_idx,
                mask=token_idx < gen_len,
                other=-1,
            )
            first_new_pos = tl.minimum(first_new_pos, tl.where(local_slot >= 0, local_slot, kv_len))
    # HP ring slots are addressed by global position (the context writer and
    # the ring transfer use it). Helix cache positions are rank-local; the
    # difference only cancels when HP_POOL_SIZE divides the page size.
    hp_pos_offset = first_new_pos - first_new_pos
    if USE_HELIX_LOCAL_SLOTS:
        for token_idx in tl.static_range(0, MAX_GEN_TILES * HP_BLOCK):
            local_slot = tl.load(
                helix_local_slots_ptr + seq_idx * gen_len + token_idx,
                mask=token_idx < gen_len,
                other=-1,
            )
            global_pos = tl.load(
                helix_position_offsets_ptr + seq_idx * gen_len + token_idx,
                mask=token_idx < gen_len,
                other=0,
            )
            hp_pos_offset = tl.where(
                local_slot == first_new_pos, global_pos - local_slot, hp_pos_offset
            )
    elif USE_HELIX:
        hp_pos_offset = rope_first_new_pos - first_new_pos
    if FUSE_ROPE_CACHE_STORE:
        work_idx = tl.program_id(1)
        dim_block = work_idx
        if work_idx >= 0:
            q_program = work_idx
            if (
                q1_grouped_kv
                and Q_WORK_BLOCKS == 2
                and Q_HEAD_BLOCKS >= 3
                and q_prefix_programs == 1
            ):
                # Move the final prefix program ahead of the last tail
                # programs to shorten the grid drain without changing Q work.
                penultimate_tail: tl.constexpr = Q_HEAD_BLOCKS * Q_WORK_BLOCKS - 3
                final_prefix: tl.constexpr = penultimate_tail + 1
                q_program = tl.where(
                    work_idx == penultimate_tail,
                    final_prefix,
                    tl.where(
                        work_idx == final_prefix,
                        penultimate_tail,
                        work_idx,
                    ),
                )
            q_token_head_program = q_program // Q_WORK_BLOCKS
            q_work_block = q_program - q_token_head_program * Q_WORK_BLOCKS
            q_token_idx = q_token_head_program // Q_HEAD_BLOCKS
            if not WRITE_Q_ROWS:
                # Another kernel writes the query rows: only the KV programs work.
                q_token_idx = gen_len
            q_head_block = q_token_head_program - q_token_idx * Q_HEAD_BLOCKS
            if q_token_idx < gen_len:
                q_token = seq_idx * gen_len + q_token_idx
                head_lanes = tl.arange(0, BLOCK_Q_HEADS)
                head_offsets = q_head_block * BLOCK_Q_HEADS + head_lanes
                # The BF16 Q input is token-major; the FP4 output follows the
                # decode layout: token-major, or packed per request where each
                # request owns Q_ROWS_PER_SEQ rows (a multiple of 128) holding
                # its [query token, head] pairs.
                q_in_rows = q_token * NUM_Q_HEADS + head_offsets
                if Q_ROWS_PER_SEQ > 0:
                    q_rows = seq_idx * Q_ROWS_PER_SEQ + q_token_idx * NUM_Q_HEADS + head_offsets
                else:
                    q_rows = q_in_rows
                q_input_row_stride: tl.constexpr = Q_PREFIX_D + ROPE_DIM
                q_packed_row_stride: tl.constexpr = (Q_PREFIX_D + 2 * ROPE_DIM) // 2
                # Every lane is valid only when the launch covers the head
                # dimension with an exact number of full head blocks.
                if NUM_Q_HEADS == Q_HEAD_BLOCKS * BLOCK_Q_HEADS:
                    row_mask = tl.full(
                        [BLOCK_Q_HEADS],
                        True,
                        dtype=tl.int1,
                    )
                else:
                    row_mask = head_offsets < NUM_Q_HEADS
                if WRITE_Q and q_work_block < q_prefix_programs:
                    if Q_PREFIX_BLOCKS_PER_PROGRAM >= 1:
                        tl.static_assert(Q_PREFIX_D % FP4_BLOCK == 0)
                        tl.static_assert(Q_PREFIX_D % Q_PREFIX_BLOCK_D == 0)
                        tl.static_assert(Q_PREFIX_BLOCK_D % FP4_BLOCK == 0)
                        q_global_scale = tl.load(q_global_scale_ptr).to(tl.float32)
                        prefix_dims = tl.arange(0, Q_PREFIX_BLOCK_D)
                        prefix_base = q_in_rows[:, None].to(tl.int64) * q_input_row_stride
                        prefix_groups: tl.constexpr = Q_PREFIX_BLOCK_D // FP4_BLOCK
                        q_prefix_iterations: tl.constexpr = Q_PREFIX_BLOCKS_PER_PROGRAM
                        for q_prefix_iteration in tl.range(
                            0,
                            q_prefix_iterations,
                            num_stages=1,
                            loop_unroll_factor=1,
                            disable_licm=True,
                        ):
                            prefix_work_block = (
                                q_work_block * Q_PREFIX_BLOCKS_PER_PROGRAM + q_prefix_iteration
                            )
                            prefix_dim_offset = prefix_work_block * Q_PREFIX_BLOCK_D
                            prefix_offsets = prefix_base + (
                                prefix_dim_offset + prefix_dims[None, :]
                            ).to(tl.int64)
                            prefix_values = tl.load(
                                q_full_ptr + prefix_offsets,
                                mask=row_mask[:, None],
                                other=0.0,
                            ).to(tl.float32)
                            prefix_pairs = tl.reshape(
                                prefix_values * q_global_scale,
                                (
                                    BLOCK_Q_HEADS,
                                    prefix_groups,
                                    FP4_BLOCK // 2,
                                    2,
                                ),
                            )
                            prefix_even, prefix_odd = tl.split(prefix_pairs)
                            prefix_packed_words, prefix_scales = _fp4_mla_quantize_q_groups_words(
                                prefix_even,
                                prefix_odd,
                                BLOCK_Q_HEADS,
                                prefix_groups,
                            )
                            prefix_group_offsets = prefix_work_block * prefix_groups + tl.arange(
                                0, prefix_groups
                            )
                            prefix_word_offsets = q_rows[:, None, None].to(
                                tl.int64
                            ) * q_packed_row_stride + (
                                prefix_group_offsets[None, :, None] * (FP4_BLOCK // 2)
                                + tl.arange(0, FP4_BLOCK // 8)[None, None, :] * 4
                            ).to(tl.int64)
                            tl.store(
                                (q_fp4_out_ptr + prefix_word_offsets).to(
                                    tl.pointer_type(tl.uint32)
                                ),
                                prefix_packed_words,
                                mask=row_mask[:, None, None],
                            )
                            if NUM_Q_HEADS == 128 and BLOCK_Q_HEADS == 32:
                                tl.static_assert(NUM_Q_HEADS == 128)
                                tl.static_assert(BLOCK_Q_HEADS == 32)
                                prefix_sf_offsets = _fp4_mla_q_sf_offset_h128(
                                    q_token,
                                    q_head_block,
                                    head_lanes[:, None],
                                    prefix_group_offsets[None, :],
                                    Q_SF_COLS,
                                )
                            else:
                                prefix_sf_offsets = _fp4_mla_q_sf_offset(
                                    q_rows[:, None],
                                    prefix_group_offsets[None, :],
                                    Q_SF_COLS,
                                )
                            tl.store(
                                q_sf_out_ptr + prefix_sf_offsets,
                                prefix_scales,
                                mask=row_mask[:, None],
                            )
                else:
                    position = (rope_first_new_pos + q_token_idx).to(tl.int64)
                    if USE_HELIX_LOCAL_SLOTS:
                        position = tl.load(helix_position_offsets_ptr + q_token).to(tl.int64)
                    valid_position = position >= 0
                    pair_offsets = tl.arange(0, ROPE_PAIR_BLOCK)
                    pair_mask = pair_offsets < ROPE_DIM // 2
                    rotary_offsets = position * (ROPE_DIM * 2) + pair_offsets * 2
                    cos = tl.load(
                        rotary_cos_sin_ptr + rotary_offsets,
                        mask=valid_position & pair_mask,
                        other=0.0,
                    ).to(tl.float32)
                    sin = tl.load(
                        rotary_cos_sin_ptr + rotary_offsets + 1,
                        mask=valid_position & pair_mask,
                        other=0.0,
                    ).to(tl.float32)
                    q_base = q_token * q_pe_s0 + head_offsets[:, None].to(tl.int64) * q_pe_s1
                    q_dims = tl.arange(0, ROPE_DIM)
                    q_dim_mask = valid_position & row_mask[:, None] & (q_dims[None, :] < ROPE_DIM)
                    q_offsets = q_base + q_dims[None, :].to(tl.int64) * q_pe_s2
                    q_values = tl.load(
                        q_pe_ptr + q_offsets,
                        mask=q_dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    q_pairs = tl.reshape(q_values, (BLOCK_Q_HEADS, ROPE_DIM // 2, 2))
                    q_even, q_odd = tl.split(q_pairs)
                    q_roped_even, q_roped_odd = _fp4_mla_rope_fp32(
                        q_even, q_odd, cos[None, :], sin[None, :]
                    )
                    q_roped_even = q_roped_even.to(tl.bfloat16)
                    q_roped_odd = q_roped_odd.to(tl.bfloat16)
                    q_out_base = q_token * q_out_s0 + head_offsets[:, None].to(tl.int64) * q_out_s1
                    q_out_offsets = q_out_base + q_dims[None, :].to(tl.int64) * q_out_s2
                    q_roped = tl.reshape(
                        tl.join(q_roped_even, q_roped_odd),
                        (BLOCK_Q_HEADS, ROPE_DIM),
                    )
                    tl.store(
                        q_rope_out_ptr + q_out_offsets,
                        q_roped,
                        mask=q_dim_mask,
                    )
                    if WRITE_Q:
                        tl.static_assert(ROPE_DIM == 64)
                        tl.static_assert(ROPE_PAIR_BLOCK == 32)
                        q_global_scale = tl.load(q_global_scale_ptr).to(tl.float32)
                        tail_even = tl.reshape(
                            q_roped_even.to(tl.float32) * q_global_scale,
                            (
                                BLOCK_Q_HEADS,
                                ROPE_DIM // FP4_BLOCK,
                                FP4_BLOCK // 2,
                            ),
                        )
                        tail_odd = tl.reshape(
                            q_roped_odd.to(tl.float32) * q_global_scale,
                            (
                                BLOCK_Q_HEADS,
                                ROPE_DIM // FP4_BLOCK,
                                FP4_BLOCK // 2,
                            ),
                        )
                        packed, scales, residual_packed, residual_scales = (
                            _fp4_mla_quantize_q_residual_groups(tail_even, tail_odd)
                        )
                        prefix_groups: tl.constexpr = Q_PREFIX_D // FP4_BLOCK
                        tail_group_count: tl.constexpr = ROPE_DIM // FP4_BLOCK
                        output_groups = prefix_groups + tl.arange(0, tail_group_count * 2)
                        bytes_in_group = tl.arange(0, FP4_BLOCK // 2)
                        packed_tail = tl.reshape(
                            tl.join(packed, residual_packed).permute(0, 1, 3, 2),
                            (
                                BLOCK_Q_HEADS,
                                tail_group_count * 2,
                                FP4_BLOCK // 2,
                            ),
                        )
                        output_offsets = q_rows[:, None, None].to(
                            tl.int64
                        ) * q_packed_row_stride + (
                            output_groups[None, :, None] * (FP4_BLOCK // 2)
                            + bytes_in_group[None, None, :]
                        ).to(tl.int64)
                        tail_row_mask = valid_position & row_mask
                        tl.store(
                            q_fp4_out_ptr + output_offsets,
                            packed_tail,
                            mask=tail_row_mask[:, None, None],
                        )
                        tail_scales = tl.reshape(
                            tl.join(scales, residual_scales),
                            (BLOCK_Q_HEADS, tail_group_count * 2),
                        )
                        if NUM_Q_HEADS == 128 and BLOCK_Q_HEADS == 32:
                            sf_offsets = _fp4_mla_q_sf_offset_h128(
                                q_token,
                                q_head_block,
                                head_lanes[:, None],
                                output_groups[None, :],
                                Q_SF_COLS,
                            )
                        else:
                            sf_offsets = _fp4_mla_q_sf_offset(
                                q_rows[:, None],
                                output_groups[None, :],
                                Q_SF_COLS,
                            )
                        tl.store(
                            q_sf_out_ptr + sf_offsets,
                            tail_scales,
                            mask=tail_row_mask[:, None],
                        )
            if q1_grouped_kv:
                if work_idx > q1_main_kv_programs:
                    return
                dim_block = tl.where(
                    work_idx == q1_main_kv_programs,
                    V_HEAD_D // FP4_BLOCK,
                    work_idx,
                )
            else:
                if work_idx >= NUM_DIM_BLOCKS:
                    return
    else:
        dim_block = tl.program_id(2)
    if USE_HELIX_LOCAL_SLOTS:
        if first_new_pos >= kv_len:
            return
    elif USE_HELIX:
        if tl.load(helix_is_inactive_rank_ptr + seq_idx):
            return
    if FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1:
        if dim_block * FP4_BLOCK > V_HEAD_D:
            return
    first_tile_pos = (first_new_pos // HP_BLOCK) * HP_BLOCK
    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
    hp_index = tl.full((), 0, tl.int64)

    if q1_grouped_kv:
        # A single V owner can share the K-residual program.  Multi-owner
        # specializations retain their interleaved dimension ownership.
        if q1_main_kv_programs == 1:
            kv_owner = work_idx == q1_main_kv_programs
            kv_work_idx = work_idx - q1_main_kv_programs
        else:
            kv_owner = work_idx < q1_main_kv_programs
            kv_work_idx = work_idx
        if kv_owner:
            block_base_pos = first_tile_pos
            if block_base_pos >= kv_len:
                return
            if block_base_pos + HP_BLOCK <= first_new_pos:
                return

            page_idx = block_base_pos // page_size
            page_pos = block_base_pos - page_idx * page_size
            physical_page_offset = page_start + page_idx
            if (
                (page_pos < 0)
                | (page_pos >= page_size)
                | (physical_page_offset < page_start)
                | (physical_page_offset >= page_end)
                | (physical_page_offset < 0)
                | (physical_page_offset >= page_ids_len)
            ):
                return
            physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
            if (physical_page < 0) | (physical_page >= num_pages):
                return
            if physical_page_offset >= hp_page_ids_len:
                return
            hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
            if (hp_index < 0) | (hp_index >= num_hp_pages):
                return

            latent_token = seq_idx * gen_len
            global_scale = tl.load(global_scale_ptr)
            # Each grouped CTA stages only the dimensions it owns.  The barrier
            # then makes the new BF16 token visible to that CTA's rolled loop.
            prestage_linear = tl.arange(0, Q1_KV_BLOCKS_PER_PROGRAM * FP4_BLOCK)
            prestage_iteration = prestage_linear // FP4_BLOCK
            prestage_lane = prestage_linear % FP4_BLOCK
            prestage_dim = (
                kv_work_idx + prestage_iteration * q1_main_kv_programs
            ) * FP4_BLOCK + prestage_lane
            prestage_values = tl.load(
                latent_cache_ptr
                + latent_token.to(tl.int64) * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + prestage_dim.to(tl.int64)
            )
            prestage_slot = (first_new_pos + hp_pos_offset) % HP_POOL_SIZE
            tl.store(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + prestage_slot * HEAD_D
                + prestage_dim,
                prestage_values,
            )
            tl.debug_barrier()
            tail_count = kv_len - block_base_pos
            if tail_count <= HP_BLOCK // 2:
                tl.static_assert(Q1_KV_BLOCKS_PER_PROGRAM % 4 == 0)
                for dim_group in tl.range(
                    0,
                    Q1_KV_BLOCKS_PER_PROGRAM // 4,
                    num_stages=1,
                    loop_unroll_factor=1,
                    disable_licm=True,
                ):
                    dim_block_base = kv_work_idx + dim_group * 4 * q1_main_kv_programs
                    _fp4_mla_generation_q1_kv_tail8_update(
                        kv_cache_ptr,
                        sf_cache_ptr,
                        v_sf_ptr,
                        v_packed_ptr,
                        hp_pool_ptr,
                        hp_index,
                        kv_len,
                        block_base_pos,
                        hp_pos_offset,
                        page_pos,
                        physical_page,
                        dim_block_base,
                        global_scale,
                        local_layer,
                        v_page_offset,
                        kv_s0,
                        kv_s2,
                        kv_s4,
                        sf_s0,
                        pool_s0,
                        pool_s1,
                        vsf_s0,
                        vsf_s1,
                        v_packed_s0,
                        v_packed_s1,
                        HEAD_D,
                        V_HEAD_D,
                        HP_BLOCK,
                        HP_POOL_SIZE,
                        FP4_BLOCK,
                        SF_PER_TOKEN,
                        SF_PER_PAGE,
                        WRITE_V_PACKED,
                        q1_main_kv_programs,
                    )
            else:
                for dim_group in tl.range(
                    0,
                    Q1_KV_BLOCKS_PER_PROGRAM // q1_kv_blocks_per_iteration,
                    num_stages=1,
                    loop_unroll_factor=1,
                    disable_licm=True,
                ):
                    dim_iterations = dim_group * q1_kv_blocks_per_iteration + tl.arange(
                        0, q1_kv_blocks_per_iteration
                    )
                    pair_dim_blocks = kv_work_idx + dim_iterations * q1_main_kv_programs
                    _fp4_mla_generation_q1_kv_tile_update(
                        kv_cache_ptr,
                        sf_cache_ptr,
                        v_sf_ptr,
                        v_packed_ptr,
                        hp_pool_ptr,
                        hp_index,
                        kv_len,
                        block_base_pos,
                        hp_pos_offset,
                        page_pos,
                        physical_page,
                        pair_dim_blocks,
                        global_scale,
                        local_layer,
                        v_page_offset,
                        kv_s0,
                        kv_s2,
                        kv_s4,
                        sf_s0,
                        pool_s0,
                        pool_s1,
                        vsf_s0,
                        vsf_s1,
                        v_packed_s0,
                        v_packed_s1,
                        HEAD_D,
                        V_HEAD_D,
                        HP_BLOCK,
                        HP_POOL_SIZE,
                        FP4_BLOCK,
                        SF_PER_TOKEN,
                        SF_PER_PAGE,
                        WRITE_V_PACKED,
                    )
        if work_idx < q1_main_kv_programs:
            return

    byte_offsets = tl.arange(0, HP_BLOCK // 2)
    token_offsets = tl.arange(0, HP_BLOCK)
    even_d = dim_block * FP4_BLOCK + byte_offsets * 2
    odd_d = even_d + 1
    all_d = dim_block * FP4_BLOCK + tl.arange(0, FP4_BLOCK)
    mask_even_d = even_d < HEAD_D
    mask_odd_d = odd_d < HEAD_D
    mask_all_d = all_d < HEAD_D
    safe_even_d = tl.where(mask_even_d, even_d, 0)
    safe_odd_d = tl.where(mask_odd_d, odd_d, 0)
    safe_all_d = tl.where(mask_all_d, all_d, 0)

    if FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1:
        if dim_block * FP4_BLOCK >= V_HEAD_D:
            tl.static_assert(STORE_K_RESIDUAL)
            tl.static_assert(HEAD_D - K_RESIDUAL_D == V_HEAD_D)
            tl.static_assert(K_RESIDUAL_D % FP4_BLOCK == 0)
            position = first_new_pos.to(tl.int64)
            if position < 0:
                return
            # Under helix the cache slot is rank-local while RoPE must use the
            # GLOBAL position; they coincide everywhere else.
            rope_position = rope_first_new_pos.to(tl.int64)
            page_idx = position // page_size
            page_pos = position - page_idx * page_size
            physical_page_offset = page_start + page_idx
            if (
                (page_pos < 0)
                | (page_pos >= page_size)
                | (physical_page_offset < page_start)
                | (physical_page_offset >= page_end)
                | (physical_page_offset < 0)
                | (physical_page_offset >= page_ids_len)
            ):
                return
            physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
            if (physical_page < 0) | (physical_page >= num_pages):
                return
            if physical_page_offset >= hp_page_ids_len:
                return
            hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
            if (hp_index < 0) | (hp_index >= num_hp_pages):
                return

            latent_token = seq_idx * gen_len
            tail_byte_offsets = tl.arange(0, K_RESIDUAL_D // 2)
            tail_even_d = V_HEAD_D + tail_byte_offsets * 2
            tail_odd_d = tail_even_d + 1
            tail_even = tl.load(
                latent_cache_ptr + latent_token * (NUM_DIM_BLOCKS * FP4_BLOCK) + tail_even_d,
            ).to(tl.float32)
            tail_odd = tl.load(
                latent_cache_ptr + latent_token * (NUM_DIM_BLOCKS * FP4_BLOCK) + tail_odd_d,
            ).to(tl.float32)
            rope_pair_offsets = tail_byte_offsets
            rotary_offsets = rope_position * (ROPE_DIM * 2) + rope_pair_offsets * 2
            cos = tl.load(rotary_cos_sin_ptr + rotary_offsets).to(tl.float32)
            sin = tl.load(rotary_cos_sin_ptr + rotary_offsets + 1).to(tl.float32)
            tail_even, tail_odd = _fp4_mla_rope_fp32(tail_even, tail_odd, cos, sin)
            tail_even = tail_even.to(tl.bfloat16).to(tl.float32)
            tail_odd = tail_odd.to(tl.bfloat16).to(tl.float32)

            residual_groups: tl.constexpr = K_RESIDUAL_D // FP4_BLOCK
            tail_even_groups = tl.reshape(
                tail_even,
                (residual_groups, FP4_BLOCK // 2),
            )
            tail_odd_groups = tl.reshape(
                tail_odd,
                (residual_groups, FP4_BLOCK // 2),
            )
            token_amax = tl.maximum(
                tl.max(tl.abs(tail_even_groups), axis=1),
                tl.max(tl.abs(tail_odd_groups), axis=1),
            )
            token_scale = tl.where(token_amax > 0.0, token_amax / 6.0, 1.0)
            global_scale = tl.load(global_scale_ptr)
            stored_scale = token_scale * global_scale
            quant_scale = stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
            low = _fp4_e2m1_quantize(tail_even_groups / quant_scale[:, None])
            high = _fp4_e2m1_quantize(tail_odd_groups / quant_scale[:, None])
            packed = low | (high << 4)

            group_offsets = tl.arange(0, residual_groups)
            bytes_in_group = tl.arange(0, FP4_BLOCK // 2)
            packed_cols = (
                V_HEAD_D // 2 + group_offsets[:, None] * (FP4_BLOCK // 2) + bytes_in_group[None, :]
            )
            kv_base = physical_page * kv_s0
            tl.store(
                kv_cache_ptr + kv_base + page_pos * kv_s2 + packed_cols * kv_s4,
                packed,
            )
            k_sf_cols = V_HEAD_D // FP4_BLOCK + group_offsets
            k_sf_offset = _fp4_mla_swizzled_sf_offset(page_pos, k_sf_cols, SF_PER_TOKEN)
            tl.store(
                sf_cache_ptr + physical_page * sf_s0 + k_sf_offset,
                stored_scale,
            )

            main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
            residual_even = tail_even_groups - main_even * quant_scale[:, None]
            residual_odd = tail_odd_groups - main_odd * quant_scale[:, None]
            residual_amax = tl.maximum(
                tl.max(tl.abs(residual_even), axis=1),
                tl.max(tl.abs(residual_odd), axis=1),
            )
            residual_scale = tl.where(residual_amax > 0.0, residual_amax / 6.0, 1.0)
            residual_stored_scale = residual_scale * global_scale
            residual_quant_scale = (
                residual_stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
            )
            residual_low = _fp4_e2m1_quantize(residual_even / residual_quant_scale[:, None])
            residual_high = _fp4_e2m1_quantize(residual_odd / residual_quant_scale[:, None])
            residual_packed = residual_low | (residual_high << 4)
            residual_packed_cols = (
                HEAD_D // 2 + group_offsets[:, None] * (FP4_BLOCK // 2) + bytes_in_group[None, :]
            )
            tl.store(
                kv_cache_ptr + kv_base + page_pos * kv_s2 + residual_packed_cols * kv_s4,
                residual_packed,
            )
            residual_sf_offset = _fp4_mla_swizzled_sf_offset(
                page_pos,
                HEAD_D // FP4_BLOCK + group_offsets,
                SF_PER_TOKEN,
            )
            tl.store(
                sf_cache_ptr + physical_page * sf_s0 + residual_sf_offset,
                residual_stored_scale,
            )

            hp_slot = (position + hp_pos_offset) % HP_POOL_SIZE
            hp_store_base = (
                hp_pool_ptr + hp_index * pool_s0 + local_layer * pool_s1 + hp_slot * HEAD_D
            )
            tl.store(hp_store_base + tail_even_d, tail_even)
            tl.store(hp_store_base + tail_odd_d, tail_odd)
            return

    for tile_iter in tl.static_range(0, MAX_GEN_TILES):
        if FUSE_ROPE_CACHE_STORE:
            tile_idx = tile_iter
        else:
            tile_idx = tl.program_id(1)
        block_base_pos = first_tile_pos + tile_idx * HP_BLOCK
        if block_base_pos >= kv_len:
            return
        if block_base_pos + HP_BLOCK <= first_new_pos:
            return

        page_idx = block_base_pos // page_size
        page_pos = block_base_pos - page_idx * page_size
        physical_page_offset = page_start + page_idx
        if (
            (page_pos < 0)
            | (page_pos >= page_size)
            | (physical_page_offset < page_start)
            | (physical_page_offset >= page_end)
            | (physical_page_offset < 0)
            | (physical_page_offset >= page_ids_len)
        ):
            return
        physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
        if (physical_page < 0) | (physical_page >= num_pages):
            return
        if physical_page_offset >= hp_page_ids_len:
            return
        hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
        if (hp_index < 0) | (hp_index >= num_hp_pages):
            return

        abs_positions = block_base_pos + token_offsets
        valid_tokens = abs_positions < kv_len
        hp_slots = (abs_positions + hp_pos_offset) % HP_POOL_SIZE
        if USE_HELIX_LOCAL_SLOTS:
            from_latent = abs_positions < 0
            new_token_offsets = abs_positions - abs_positions
            for token_idx in tl.static_range(0, MAX_GEN_TILES * HP_BLOCK):
                local_slot = tl.load(
                    helix_local_slots_ptr + seq_idx * gen_len + token_idx,
                    mask=token_idx < gen_len,
                    other=-1,
                )
                is_local_token = abs_positions == local_slot
                from_latent = from_latent | is_local_token
                new_token_offsets = tl.where(is_local_token, token_idx, new_token_offsets)
        else:
            from_latent = abs_positions >= first_new_pos
            new_token_offsets = abs_positions - first_new_pos
        # Linear MTP uses a uniform generation length, so each sequence
        # occupies one contiguous gen_len slice in latent_cache.
        latent_tokens = seq_idx * gen_len + new_token_offsets
        safe_latent_tokens = tl.where(valid_tokens & from_latent, latent_tokens, 0).to(tl.int64)

        if q1_shared_main:
            # Q1 reads each token from exactly one source. Selecting the pointer
            # first removes two masked loads and additions without changing MTP.
            hp_even_ptrs = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_even_d[None, :]
            )
            hp_odd_ptrs = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_odd_d[None, :]
            )
            latent_even_ptrs = (
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_even_d[None, :]
            )
            latent_odd_ptrs = (
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_odd_d[None, :]
            )
            even_values = tl.load(
                tl.where(from_latent[:, None], latent_even_ptrs, hp_even_ptrs),
                mask=valid_tokens[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            odd_values = tl.load(
                tl.where(from_latent[:, None], latent_odd_ptrs, hp_odd_ptrs),
                mask=valid_tokens[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            hp_even = tl.load(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_even_d[None, :],
                mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            hp_odd = tl.load(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_odd_d[None, :],
                mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
            latent_even = tl.load(
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_even_d[None, :],
                mask=valid_tokens[:, None] & from_latent[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            latent_odd = tl.load(
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_odd_d[None, :],
                mask=valid_tokens[:, None] & from_latent[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
            even_values = hp_even + latent_even
            odd_values = hp_odd + latent_odd

        if FUSE_ROPE_CACHE_STORE and not q1_shared_main:
            if dim_block * FP4_BLOCK >= V_HEAD_D:
                positions = (rope_first_new_pos + new_token_offsets).to(tl.int64)
                if USE_HELIX_LOCAL_SLOTS:
                    positions = tl.load(
                        helix_position_offsets_ptr + safe_latent_tokens,
                        mask=valid_tokens & from_latent,
                        other=-1,
                    ).to(tl.int64)
                valid_position = valid_tokens & from_latent & (positions >= 0)
                rope_pair_offsets = (safe_even_d - V_HEAD_D) // 2
                rope_dim_mask = mask_even_d & (even_d >= V_HEAD_D) & (odd_d < V_HEAD_D + ROPE_DIM)
                rotary_offsets = (
                    positions[:, None] * (ROPE_DIM * 2) + rope_pair_offsets[None, :] * 2
                )
                rotary_mask = valid_position[:, None] & rope_dim_mask[None, :]
                cos = tl.load(
                    rotary_cos_sin_ptr + rotary_offsets,
                    mask=rotary_mask,
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    rotary_cos_sin_ptr + rotary_offsets + 1,
                    mask=rotary_mask,
                    other=0.0,
                ).to(tl.float32)
                roped_even, roped_odd = _fp4_mla_rope_fp32(even_values, odd_values, cos, sin)
                roped_even = roped_even.to(tl.bfloat16).to(tl.float32)
                roped_odd = roped_odd.to(tl.bfloat16).to(tl.float32)
                even_values = tl.where(rotary_mask, roped_even, even_values)
                odd_values = tl.where(rotary_mask, roped_odd, odd_values)

        amax_per_token = tl.maximum(
            tl.max(tl.abs(even_values), axis=1),
            tl.max(tl.abs(odd_values), axis=1),
        )
        global_scale = tl.load(global_scale_ptr)
        token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
        if q1_shared_main:
            tile_amax = tl.max(amax_per_token, axis=0)
            tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
            local_scale = tl.full((HP_BLOCK,), tile_scale, tl.float32)
            token_scale_col = page_pos // HP_BLOCK
            sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
            v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
                vsf_s0, tl.int64
            ) + physical_page * tl.cast(vsf_s1, tl.int64)
            tl.store(
                v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
                tile_scale * global_scale,
                mask=mask_all_d & (all_d < V_HEAD_D),
            )
        else:
            shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
            if shared_tile:
                tile_amax = tl.max(amax_per_token, axis=0)
                tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
                local_scale = tl.full((HP_BLOCK,), tile_scale, tl.float32)
                token_scale_col = page_pos // HP_BLOCK
                sf_offsets = _fp4_mla_swizzled_sf_offset(
                    safe_all_d,
                    token_scale_col,
                    SF_PER_PAGE,
                )
                v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
                    vsf_s0, tl.int64
                ) + physical_page * tl.cast(vsf_s1, tl.int64)
                tl.store(
                    v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
                    tile_scale * global_scale,
                    mask=mask_all_d & (all_d < V_HEAD_D),
                )
            else:
                local_scale = token_scale
        stored_scale = local_scale * global_scale

        quant_scale = local_scale
        if STORE_K_RESIDUAL and not q1_shared_main:
            if dim_block * FP4_BLOCK >= HEAD_D - K_RESIDUAL_D:
                quant_scale = stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
        low = _fp4_e2m1_quantize(even_values / quant_scale[:, None])
        high = _fp4_e2m1_quantize(odd_values / quant_scale[:, None])
        packed = low | (high << 4)
        v_dim_packed = packed

        packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
        page_positions = page_pos + token_offsets
        kv_base = physical_page * kv_s0
        tl.store(
            kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
            packed,
            mask=valid_tokens[:, None] & mask_even_d[None, :],
        )

        if WRITE_V_PACKED:
            if q1_shared_main or dim_block * FP4_BLOCK < V_HEAD_D:
                # The canonical cache packs dimensions; PV packs tokens.
                # Transpose the selected V codes in registers.
                v_pairs = tl.reshape(
                    v_dim_packed.T,
                    (FP4_BLOCK // 2, HP_BLOCK // 2, 2),
                )
                even_token_packed, odd_token_packed = tl.split(v_pairs)
                v_low = (even_token_packed & 0x0F) | ((odd_token_packed & 0x0F) << 4)
                v_high = ((even_token_packed >> 4) & 0x0F) | (odd_token_packed & 0xF0)
                # Keep the two dimension parities in the 8x8 layout produced
                # by the token-pair transpose.  Interleaving them into a
                # 16x8 tensor forces another cross-warp layout conversion;
                # two stores address the exact same coalesced rows without
                # changing any packed bytes.
                v_even_rows = (
                    (v_page_offset + physical_page) * V_HEAD_D
                    + dim_block * FP4_BLOCK
                    + tl.arange(0, FP4_BLOCK // 2) * 2
                ).to(tl.int64)
                pair_offsets = tl.arange(0, HP_BLOCK // 2)
                v_cols = page_pos // 2 + pair_offsets
                v_even_offsets = v_even_rows[:, None] * v_packed_s0 + v_cols[None, :] * v_packed_s1
                v_odd_offsets = v_even_offsets + v_packed_s0
                token0_valid = block_base_pos + pair_offsets * 2 < kv_len
                token1_valid = block_base_pos + pair_offsets * 2 + 1 < kv_len
                partial_pair = token0_valid & (~token1_valid)
                old_v_low = tl.load(
                    v_packed_ptr + v_even_offsets,
                    mask=partial_pair[None, :],
                    other=0,
                )
                old_v_high = tl.load(
                    v_packed_ptr + v_odd_offsets,
                    mask=partial_pair[None, :],
                    other=0,
                )
                v_low = tl.where(
                    partial_pair[None, :],
                    (old_v_low & 0xF0) | (v_low & 0x0F),
                    v_low,
                ).to(tl.uint8)
                v_high = tl.where(
                    partial_pair[None, :],
                    (old_v_high & 0xF0) | (v_high & 0x0F),
                    v_high,
                ).to(tl.uint8)
                tl.store(
                    v_packed_ptr + v_even_offsets,
                    v_low,
                    mask=token0_valid[None, :],
                )
                tl.store(
                    v_packed_ptr + v_odd_offsets,
                    v_high,
                    mask=token0_valid[None, :],
                )

        k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
        tl.store(
            sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets,
            stored_scale,
            mask=valid_tokens,
        )

        if STORE_K_RESIDUAL and not q1_shared_main:
            residual_start = HEAD_D - K_RESIDUAL_D
            if dim_block * FP4_BLOCK >= residual_start:
                if MAX_GEN_TILES == 1:
                    main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
                    main_even *= quant_scale[:, None]
                    main_odd *= quant_scale[:, None]
                else:
                    main_even = _fp4_e2m1_to_f32(low) * quant_scale[:, None]
                    main_odd = _fp4_e2m1_to_f32(high) * quant_scale[:, None]
                residual_even = even_values - main_even
                residual_odd = odd_values - main_odd
                residual_amax = tl.maximum(
                    tl.max(tl.abs(residual_even), axis=1),
                    tl.max(tl.abs(residual_odd), axis=1),
                )
                residual_scale = tl.where(residual_amax > 0.0, residual_amax / 6.0, 1.0)
                residual_stored_scale = residual_scale * global_scale
                residual_quant_scale = (
                    residual_stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
                )
                residual_low = _fp4_e2m1_quantize(residual_even / residual_quant_scale[:, None])
                residual_high = _fp4_e2m1_quantize(residual_odd / residual_quant_scale[:, None])
                residual_packed = residual_low | (residual_high << 4)
                residual_group = dim_block - residual_start // FP4_BLOCK
                residual_packed_cols = (
                    HEAD_D // 2 + residual_group * (FP4_BLOCK // 2) + byte_offsets
                )
                tl.store(
                    kv_cache_ptr
                    + kv_base
                    + page_positions[:, None] * kv_s2
                    + residual_packed_cols[None, :] * kv_s4,
                    residual_packed,
                    mask=valid_tokens[:, None] & mask_even_d[None, :],
                )
                residual_sf_offsets = _fp4_mla_swizzled_sf_offset(
                    page_positions,
                    HEAD_D // FP4_BLOCK + residual_group,
                    SF_PER_TOKEN,
                )
                tl.store(
                    sf_cache_ptr + physical_page * sf_s0 + residual_sf_offsets,
                    residual_stored_scale,
                    mask=valid_tokens,
                )

        if FUSE_ROPE_CACHE_STORE:
            hp_store_mask = valid_tokens[:, None] & from_latent[:, None]
            hp_store_base = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
            )
            tl.store(
                hp_store_base + safe_even_d[None, :],
                even_values,
                mask=hp_store_mask & mask_even_d[None, :],
            )
            tl.store(
                hp_store_base + safe_odd_d[None, :],
                odd_values,
                mask=hp_store_mask & mask_odd_d[None, :],
            )
            if tile_iter + 1 < MAX_GEN_TILES:
                # The next tile can wrap onto HP slots read by this tile.
                # Keep each dimension block's read-before-overwrite order.
                tl.debug_barrier()
