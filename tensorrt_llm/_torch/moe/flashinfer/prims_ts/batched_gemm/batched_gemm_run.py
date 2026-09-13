# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation, correctness, and benchmark entry point for BatchedGemm TS.

Modes:
  --validate-only   Pure Python schedule validation (no GPU).
  --reference-check Launch on GPU, compare to PyTorch reference.
  --benchmark       Benchmark mode.
"""

import argparse
from dataclasses import dataclass, replace
import gc
import math
import os

from ..cutlass_dsl import task_scheduling_scope


def _cute_compile_options() -> str:
    return os.environ.get("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2 --enable-tvm-ffi")


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _generate_line_info_enabled() -> bool:
    return _env_flag_enabled("FLASHINFER_PRIMS_TS_DEBUG_CHECKS")


import torch
import cutlass
import cutlass.cute as cute

from .batched_gemm_kernel import (
    build_batched_gemm_task_manager,
    gemm,
)
from .batched_gemm_config import (
    BatchMode,
    RouteImpl,
    TileScheduler,
    ActKind,
    DType,
    SfLayout,
    derive_problem_shape_constants,
    make_config,
    resolve_early_exit_max_token_ctas,
)
from .batched_gemm_quant import (
    kaiming_uniform_tensor as _kaiming_uniform_tensor,
    quantize_block_scaled as _quantize_block_scaled,
    build_sf_buffer as _build_sf_buffer,
    is_e2m1_kind as _is_e2m1_kind,
)

import cutlass.torch as _cutlass_torch


def get_default_stream():
    return _cutlass_torch.default_stream()


DTYPE_NAME_TO_VALUE = {
    "fp32": DType.FP32,
    "fp16": DType.FP16,
    "bf16": DType.BF16,
    "e2m1": DType.E2M1,
    "mxe2m1": DType.MXE2M1,
    "mxe4m3": DType.MXE4M3,
    "e4m3": DType.E4M3,
}


def _per_token_torch_dtype(cfg):
    if cfg.per_token_sf_dtype == int(DType.BF16):
        return torch.bfloat16
    if cfg.per_token_sf_dtype == int(DType.FP16):
        return torch.float16
    return torch.float32


def _per_token_cutlass_dtype(cfg):
    if cfg.per_token_sf_dtype == int(DType.BF16):
        return cutlass.BFloat16
    if cfg.per_token_sf_dtype == int(DType.FP16):
        return cutlass.Float16
    return cutlass.Float32


def _plain_output_torch_dtype(cfg):
    if cfg.dtype_c_kind == int(DType.FP16):
        return torch.float16
    return torch.bfloat16


def _plain_output_cutlass_dtype(cfg):
    if cfg.dtype_c_kind == int(DType.FP16):
        return cutlass.Float16
    return cutlass.BFloat16


def _is_gated_act_kind(act_kind: int) -> bool:
    return act_kind in (
        int(ActKind.SWIGLU),
        int(ActKind.GEGLU),
        int(ActKind.SILU),
        int(ActKind.SITU),
    )


def _make_per_expert_f32_tensor(value, *, default: float, num_experts: int, device):
    if value is None:
        return torch.full((num_experts,), default, dtype=torch.float32, device=device)
    if torch.is_tensor(value):
        if value.ndim != 1:
            raise ValueError("per-expert OA parameter tensors must be 1D")
        if value.numel() < num_experts:
            raise ValueError(
                "per-expert OA parameter tensors must have at least "
                f"{num_experts} elements"
            )
        return value[:num_experts].to(device=device, dtype=torch.float32).contiguous()
    return torch.full((num_experts,), float(value), dtype=torch.float32, device=device)


def _logical_output_shape(cfg, m: int, n: int, is_gated: bool) -> tuple[int, int]:
    if not is_gated:
        return m, n
    if cfg.is_swap_ab:
        return m // 2, n
    return m, n // 2


@dataclass(frozen=True)
class TokenLayout:
    """Generated-style MoE token layout for the batched token dimension."""

    tile_size: int
    cga_tile_size: int
    tokens_per_expert: list[int]
    padded_starts: list[int]
    expanded_to_expert: list[int]
    expanded_to_token: list[int]
    tile_idx: list[int]
    mn_limit: list[int]

    @property
    def total_padded_tokens(self) -> int:
        return self.padded_starts[-1]

    @property
    def num_token_tiles(self) -> int:
        return len(self.tile_idx)


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _make_token_layout(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    tile_size: int,
    cluster_dim_in_token: int,
) -> TokenLayout:
    """Build the MoE expanded/permuted token layout."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")

    cga_tile_size = tile_size * cluster_dim_in_token
    tokens_per_expert = [0] * num_experts
    assignments: list[tuple[int, int, int]] = []
    for token_idx in range(num_tokens):
        for k_idx in range(top_k):
            expert_idx = (token_idx * top_k + k_idx) % num_experts
            local_idx = tokens_per_expert[expert_idx]
            tokens_per_expert[expert_idx] += 1
            assignments.append((expert_idx, local_idx, token_idx))

    padded_starts = [0]
    for token_count in tokens_per_expert:
        padded_starts.append(padded_starts[-1] + _round_up(token_count, cga_tile_size))

    total_padded = padded_starts[-1]
    expanded_to_expert = [-1] * total_padded
    expanded_to_token = [0] * total_padded
    for expert_idx, local_idx, token_idx in assignments:
        expanded_idx = padded_starts[expert_idx] + local_idx
        expanded_to_expert[expanded_idx] = expert_idx
        expanded_to_token[expanded_idx] = token_idx

    num_token_tiles = total_padded // tile_size if total_padded > 0 else 1
    tile_idx = [0] * num_token_tiles
    mn_limit = [0] * num_token_tiles
    for expert_idx, token_count in enumerate(tokens_per_expert):
        padded_start = padded_starts[expert_idx]
        padded_end = padded_starts[expert_idx + 1]
        valid_end = padded_start + token_count
        for tile in range(padded_start // tile_size, padded_end // tile_size):
            tile_idx[tile] = expert_idx
            mn_limit[tile] = valid_end

    return TokenLayout(
        tile_size=tile_size,
        cga_tile_size=cga_tile_size,
        tokens_per_expert=tokens_per_expert,
        padded_starts=padded_starts,
        expanded_to_expert=expanded_to_expert,
        expanded_to_token=expanded_to_token,
        tile_idx=tile_idx,
        mn_limit=mn_limit,
    )


def _launch_metadata_lists(
    cfg,
    token_layout: TokenLayout,
    early_exit_max_token_ctas: int,
):
    """Return token launch metadata padded for persistent early-exit CTAs."""
    launch_token_tiles = token_layout.num_token_tiles
    if cfg.use_early_exit and cfg.is_persistent:
        launch_token_tiles = max(launch_token_tiles, early_exit_max_token_ctas)

    tile_idx = list(token_layout.tile_idx)
    mn_limit = list(token_layout.mn_limit)
    if launch_token_tiles > token_layout.num_token_tiles:
        extra_tiles = launch_token_tiles - token_layout.num_token_tiles
        # Persistent early-exit over-launches token CTAs for CUDA graph reuse.
        # Extra metadata is inactive because the scheduler exits before those
        # CTAs read routing tables.
        tile_idx.extend([0] * extra_tiles)
        mn_limit.extend([0] * extra_tiles)

    route_map = list(token_layout.expanded_to_token)
    min_route_rows = launch_token_tiles * token_layout.tile_size
    if len(route_map) < min_route_rows:
        route_map.extend([0] * (min_route_rows - len(route_map)))

    return tile_idx, mn_limit, route_map


def _normalize_early_exit_launch_extent(
    cfg,
    token_layout: TokenLayout,
    early_exit_max_token_ctas: int,
) -> int:
    """Return an early-exit extent that cannot underlaunch active token tiles."""
    if cfg.use_early_exit:
        return max(int(token_layout.num_token_tiles), int(early_exit_max_token_ctas))
    return int(early_exit_max_token_ctas)


def _runtime_config(cfg, in_hidden: int):
    """Normalize runtime options for a concrete K."""
    if cfg.has_deepseek_fp8:
        if cfg.tile_k != 128:
            raise ValueError(
                "use_deepseek_fp8=1 requires tile_k=128 to match the "
                f"128-K DeepSeek scale-factor blocks, got {cfg.tile_k}"
            )
        if in_hidden % cfg.tile_k != 0:
            raise ValueError(
                "use_deepseek_fp8=1 requires problem_k to be a multiple of "
                f"tile_k={cfg.tile_k}, got problem_k={in_hidden}"
            )
    num_k_tiles = (in_hidden + cfg.tile_k - 1) // cfg.tile_k
    use_unroll_loop_2x_for_mma = cfg.use_unroll_loop_2x_for_mma
    if use_unroll_loop_2x_for_mma and (num_k_tiles < 2 or num_k_tiles % 2 != 0):
        # Keep the generated loop structure identical to Gen's 2x-unrolled
        # steady state. Fall back when concrete K would require a scalar tail.
        use_unroll_loop_2x_for_mma = 0
    runtime_cfg = replace(
        cfg,
        use_unroll_loop_2x_for_mma=use_unroll_loop_2x_for_mma,
    )
    derive_problem_shape_constants(runtime_cfg, problem_mnk=(0, 0, in_hidden))
    return runtime_cfg


def _expand_bf16_activations(
    compact_activations: torch.Tensor,
    token_layout: TokenLayout,
) -> torch.Tensor:
    expanded = torch.zeros(
        (token_layout.total_padded_tokens, compact_activations.shape[1]),
        dtype=compact_activations.dtype,
        device=compact_activations.device,
    )
    for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
        if expert_idx >= 0:
            token_idx = token_layout.expanded_to_token[expanded_idx]
            expanded[expanded_idx, :] = compact_activations[token_idx, :]
    return expanded


def _expand_fp4_activations(
    compact_activations: torch.Tensor,
    token_layout: TokenLayout,
) -> torch.Tensor:
    expanded = torch.zeros(
        (token_layout.total_padded_tokens, compact_activations.shape[1]),
        dtype=compact_activations.dtype,
        device=compact_activations.device,
    )
    for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
        if expert_idx >= 0:
            token_idx = token_layout.expanded_to_token[expanded_idx]
            expanded[expanded_idx, :] = compact_activations[token_idx, :]
    return expanded


def _random_uint8(
    shape: tuple[int, ...],
    *,
    low: int = 0,
    high: int,
    device: str,
) -> torch.Tensor:
    """Create random uint8 storage."""
    return torch.randint(low, high, shape, dtype=torch.int32, device=device).to(
        torch.uint8
    )


def _random_packed_fp4(shape: tuple[int, ...], *, device: str) -> torch.Tensor:
    """Create packed E2M1 data with two independent FP4 nibbles per byte."""
    return _random_uint8(shape, high=256, device=device)


def _is_packed_e2m1_dtype(dtype_kind: int) -> bool:
    return dtype_kind in (int(DType.E2M1), int(DType.MXE2M1))


def _storage_shape_for_dtype(
    shape: tuple[int, ...], dtype_kind: int
) -> tuple[int, ...]:
    if _is_packed_e2m1_dtype(dtype_kind):
        if shape[-1] % 2 != 0:
            raise ValueError(f"Packed E2M1 requires even K dimension, got {shape[-1]}")
        return (*shape[:-1], shape[-1] // 2)
    return shape


def _random_quantized_tensor(
    shape: tuple[int, ...],
    *,
    dtype_kind: int,
    fan_in: int,
    device: str,
) -> torch.Tensor:
    """Create raw storage for generated block-scaled MMA input dtypes."""
    if _is_packed_e2m1_dtype(dtype_kind):
        return _random_packed_fp4(
            _storage_shape_for_dtype(shape, dtype_kind), device=device
        )
    if dtype_kind in (int(DType.MXE4M3), int(DType.E4M3)):
        return _kaiming_uniform_tensor(
            shape,
            fan_in=fan_in,
            dtype=torch.float8_e4m3fn,
            device=device,
        ).view(torch.uint8)
    raise ValueError(f"Unsupported quantized dtype kind: {dtype_kind}")


def _cutlass_data_dtype(dtype_kind: int):
    if dtype_kind == int(DType.BF16):
        return cutlass.BFloat16
    if dtype_kind in (int(DType.E2M1), int(DType.MXE2M1)):
        return cutlass.Float4E2M1FN
    if dtype_kind in (int(DType.MXE4M3), int(DType.E4M3)):
        return cutlass.Float8E4M3FN
    raise ValueError(f"Unsupported dtype kind: {dtype_kind}")


def _sf_ab_cutlass_dtype(cfg):
    if cfg.has_deepseek_fp8:
        return cutlass.Float32
    return cutlass.Float8E8M0FNU if cfg.uses_mx_scale_factors else cutlass.Float8E4M3FN


def _sf_c_cutlass_dtype(cfg):
    if cfg.has_deepseek_fp8_c_scale:
        return cutlass.Float32
    return cutlass.Float8E8M0FNU if cfg.uses_mx_output_quant else cutlass.Float8E4M3FN


def _sf_torch_dtype(cfg):
    return torch.float8_e8m0fnu if cfg.uses_mx_scale_factors else torch.float8_e4m3fn


def _unit_sf_byte(cfg) -> int:
    return 0x7F if cfg.uses_mx_scale_factors else 0x38


def _create_deepseek_fp8_sf_tensors(
    cfg,
    *,
    num_experts: int,
    num_tokens: int,
    out_hidden: int,
    in_hidden: int,
    total_padded_tokens: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create DeepSeek FP8 AB dequant scale tensors.

    Activation scales are K-major: ``sf_activations[k_block, token_row]``.
    Weight scales are per expert, output tile, and 128-K block.
    """
    if not cfg.has_deepseek_fp8:
        raise ValueError("DeepSeek FP8 scale tensors require use_deepseek_fp8=1")

    ds_k_blocks = _round_up(in_hidden, 128) // 128
    if cfg.is_swap_ab:
        ds_weight_tiles = _round_up(out_hidden, cfg.tile_m) // cfg.tile_m
    else:
        ds_weight_tiles = _round_up(out_hidden, cfg.tile_n) // cfg.tile_n
    ds_act_rows = num_tokens if cfg.has_routed_act else total_padded_tokens

    sf_weights = torch.empty(
        (num_experts, ds_weight_tiles, ds_k_blocks),
        dtype=torch.float32,
        device=device,
    ).uniform_(0.75, 1.25)
    sf_activations = torch.empty(
        (ds_k_blocks, ds_act_rows),
        dtype=torch.float32,
        device=device,
    ).uniform_(0.75, 1.25)
    return sf_weights, sf_activations


def _randomize_sf_layout(layout_kind: str) -> bool:
    """Return whether to randomize the requested scale-factor layout.

    Random scale factors are the default. Set
    BATCHED_GEMM_TS_RANDOMIZE_SF=none, 0, or false to force unit scale
    factors for debugging.
    """
    mode = os.environ.get("BATCHED_GEMM_TS_RANDOMIZE_SF", "all").strip().lower()
    if mode in ("0", "false", "none"):
        return False
    if mode in ("1", "true", "all"):
        return True
    if mode not in ("linear", "mma"):
        raise ValueError(
            "BATCHED_GEMM_TS_RANDOMIZE_SF must be one of "
            "'all', 'none', 'linear', or 'mma'; unset defaults to 'all', got "
            f"{mode!r}"
        )
    return mode == layout_kind


def _random_sf_bytes(
    shape: tuple[int, ...],
    *,
    cfg,
    device: str,
    layout_kind: str,
) -> torch.Tensor:
    """Create random, valid, non-negative scale-factor bytes."""
    if not _randomize_sf_layout(layout_kind):
        return torch.full(shape, _unit_sf_byte(cfg), dtype=torch.uint8, device=device)
    if cfg.uses_mx_scale_factors:
        # Draw UE8M0 scale factors from 2^-5 (0x7a) through
        # 2^5 (0x84). torch.randint uses an exclusive upper bound.
        return _random_uint8(shape, low=0x7A, high=0x85, device=device)
    # E4M3FN scale factors are non-negative. Keeping the sign bit clear also
    # excludes negative values; 0x7f is the positive NaN. This is the full
    # valid non-negative E4M3 range. torch.randint uses an exclusive upper bound.
    return _random_uint8(shape, low=0x00, high=0x7F, device=device)


def _fp4_e2m1_values(device: str) -> torch.Tensor:
    """E2M1 decode table matching CUDA's packed nibble representation."""
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _unpack_fp4_e2m1(packed: torch.Tensor, logical_cols: int) -> torch.Tensor:
    """Unpack packed E2M1 bytes to a float32 tensor with ``logical_cols`` cols."""
    fp4_table = _fp4_e2m1_values(packed.device)
    unpacked_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    nibbles = torch.empty(unpacked_shape, dtype=torch.long, device=packed.device)
    nibbles[..., 0::2] = (packed & 0x0F).to(torch.long)
    nibbles[..., 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    return fp4_table[nibbles[..., :logical_cols]]


def _kaiming_quantized_block_scaled_inputs(
    *,
    num_experts: int,
    out_hidden: int,
    in_hidden: int,
    total_tokens: int,
    total_padded_tokens: int,
    sf_k: int,
    weight_dtype_kind: int,
    activation_dtype_kind: int,
    uses_mx: bool,
    sf_vec: int,
    weight_sf_layout: int,
    activation_sf_layout: int,
    has_routed_sfs: bool,
    token_layout,
    cfg,
    device: str,
):
    """Kaiming-init then block-scale-quantize weights and activations for fusedAct.

    Handles any block-scaled input dtype (NVFP4/MXFP4/MXFP8): fp32 Kaiming data,
    scaled into the kernel's accumulator regime, then quantized to packed E2M1 or
    E4M3 with E4M3 or UE8M0 scale factors. Returns ``(weights_storage,
    sf_weights, activation_compact_storage, sf_activations)`` laid out so the
    reference's dequant readers reconstruct them consistently.
    """
    # Regime lift: scale the Kaiming data by sqrt(fan_in) so the accumulator
    # lands in the kernel's clamp-active regime and fused-activation outputs stay
    # representable (e.g. avoid all-zero FP4/MXFP8 C from SF underflow). UE8M0 MX
    # scale factors are restricted to powers of two and cannot carry the scale,
    # so it is applied to the data; for NVFP4 (E4M3 SF) that is equivalent to
    # scaling the stored SF.
    data_scale = math.sqrt(in_hidden)

    def _quant(data, dtype_kind):
        return _quantize_block_scaled(
            data * data_scale,
            dtype_kind=dtype_kind,
            uses_mx=uses_mx,
            sf_vec_size=sf_vec,
        )

    def _storage_cols(dtype_kind):
        return in_hidden // 2 if _is_e2m1_kind(dtype_kind) else in_hidden

    # Weights: fold experts into the row dimension for quantization, matching the
    # dequant reader's folded SF row index (expert * out_hidden + row).
    w_fp32 = _kaiming_uniform_tensor(
        (num_experts * out_hidden, in_hidden),
        fan_in=in_hidden,
        dtype=torch.float32,
        device=device,
    )
    w_storage, w_sf = _quant(w_fp32, weight_dtype_kind)
    weights_torch = w_storage.reshape(
        num_experts, out_hidden, _storage_cols(weight_dtype_kind)
    )
    w_sf_total = _create_sf_tensor_mma_layout(
        num_experts=num_experts,
        mn=out_hidden,
        sf_k=sf_k,
        layout=weight_sf_layout,
        device=device,
        cfg=cfg,
    ).numel()
    sf_weights = _build_sf_buffer(
        w_sf,
        total_bytes=w_sf_total,
        layout=weight_sf_layout,
        data_blk_cols=sf_k,
        device=device,
    )

    # Activations are quantized per (compact) token row.
    a_fp32 = _kaiming_uniform_tensor(
        (total_tokens, in_hidden),
        fan_in=in_hidden,
        dtype=torch.float32,
        device=device,
    )
    activation_compact_torch, a_sf = _quant(a_fp32, activation_dtype_kind)

    if has_routed_sfs:
        # Routed activation SF: linear GMEM layout, indexed by compact token.
        a_sf_total = _create_sf_linear(
            num_tokens=total_tokens,
            sf_k_dim=sf_k,
            device=device,
            cfg=cfg,
        ).numel()
        sf_activations = _build_sf_buffer(
            a_sf,
            total_bytes=a_sf_total,
            layout=int(SfLayout.LINEAR),
            data_blk_cols=_round_up(sf_k, 16),
            device=device,
        )
    else:
        # Non-routed activation SF: generated MMA layout indexed by the expanded
        # token row. Replicate each token's SF across its expanded rows.
        a_sf_total = _create_sf_tensor_mma_layout(
            num_experts=1,
            mn=total_padded_tokens,
            sf_k=sf_k,
            layout=activation_sf_layout,
            device=device,
            cfg=cfg,
        ).numel()
        expanded_to_token = torch.as_tensor(
            token_layout.expanded_to_token, dtype=torch.long, device=device
        )
        expanded_to_expert = torch.as_tensor(
            token_layout.expanded_to_expert, dtype=torch.long, device=device
        )
        exp_rows = (expanded_to_expert >= 0).nonzero().flatten()
        sf_activations = _build_sf_buffer(
            a_sf[expanded_to_token[exp_rows]],
            total_bytes=a_sf_total,
            layout=activation_sf_layout,
            data_blk_cols=sf_k,
            device=device,
            sf_rows=exp_rows,
        )

    return weights_torch, sf_weights, activation_compact_torch, sf_activations


def _sf_offsets(
    *,
    row_idx: int,
    sf_cols: int,
    layout: int,
    num_padded_sf_cols: int | None = None,
    device: str,
) -> torch.Tensor:
    """Scale-factor byte offsets for one data row.

    Computes SF byte offsets for Linear, R8c4 and R128c4 layouts.
    Weight SF tensors use a single physical row coordinate
    with experts folded into the row dimension.
    """
    data_blk_cols = num_padded_sf_cols if num_padded_sf_cols is not None else sf_cols
    kblk = torch.arange(sf_cols, dtype=torch.long, device=device)
    if layout == int(SfLayout.LINEAR):
        return row_idx * data_blk_cols + kblk

    if layout == int(SfLayout.R8c4):
        rows_per_block = 8
        cols_per_block = 4
        bytes_per_block = 32
        sf_row = row_idx % rows_per_block
    elif layout == int(SfLayout.R128c4):
        rows_per_block = 128
        cols_per_block = 4
        bytes_per_block = 512
        sf_row = (row_idx % 32) * 4 + (row_idx % rows_per_block) // 32
    else:
        raise ValueError(f"Unsupported FP4 SF layout for reference: {layout}")

    sf_blk_row = row_idx // rows_per_block
    sf_blk_col = kblk // cols_per_block
    sf_blk_idx = sf_blk_row * (data_blk_cols // cols_per_block) + sf_blk_col
    sf_col = kblk % cols_per_block
    return sf_blk_idx * bytes_per_block + sf_row * cols_per_block + sf_col


def _fp4_sf_values_for_row(
    sf: torch.Tensor,
    *,
    row_idx: int,
    logical_cols: int,
    layout: int,
    sf_vec_size: int = 16,
    sf_torch_dtype=torch.float8_e4m3fn,
    num_padded_sf_cols: int | None = None,
) -> torch.Tensor:
    """Load scale factors for one row of block-scaled input data."""
    sf_cols = _round_up(logical_cols, sf_vec_size) // sf_vec_size
    offsets = _sf_offsets(
        row_idx=row_idx,
        sf_cols=sf_cols,
        layout=layout,
        num_padded_sf_cols=num_padded_sf_cols,
        device=sf.device,
    )
    return sf.view(sf_torch_dtype).float()[offsets]


def _apply_fp4_sf_to_row(
    unpacked_row: torch.Tensor,
    sf: torch.Tensor,
    *,
    row_idx: int,
    layout: int,
    sf_vec_size: int = 16,
    sf_torch_dtype=torch.float8_e4m3fn,
    num_padded_sf_cols: int | None = None,
) -> torch.Tensor:
    """Apply row-local block scale factors to unpacked input values."""
    sf_vals = _fp4_sf_values_for_row(
        sf,
        row_idx=row_idx,
        logical_cols=unpacked_row.shape[-1],
        layout=layout,
        sf_vec_size=sf_vec_size,
        sf_torch_dtype=sf_torch_dtype,
        num_padded_sf_cols=num_padded_sf_cols,
    )
    return (
        unpacked_row * sf_vals.repeat_interleave(sf_vec_size)[: unpacked_row.shape[-1]]
    )


def _dequantize_fp4_tensor(
    packed: torch.Tensor,
    sf: torch.Tensor,
    *,
    logical_cols: int,
    layout: int,
) -> torch.Tensor:
    """Dequantize ``[experts, rows, K/2]`` packed E2M1 data to float32."""
    unpacked = _unpack_fp4_e2m1(packed, logical_cols)
    out = torch.empty(
        (*packed.shape[:-1], logical_cols),
        dtype=torch.float32,
        device=packed.device,
    )
    rows_per_expert = packed.shape[1]
    for expert_idx in range(packed.shape[0]):
        for row_idx in range(packed.shape[1]):
            sf_row_idx = expert_idx * rows_per_expert + row_idx
            out[expert_idx, row_idx, :] = _apply_fp4_sf_to_row(
                unpacked[expert_idx, row_idx, :],
                sf,
                row_idx=sf_row_idx,
                layout=layout,
            )
    return out


def _decode_quantized_storage(
    storage: torch.Tensor,
    *,
    dtype_kind: int,
    logical_cols: int,
) -> torch.Tensor:
    if _is_packed_e2m1_dtype(dtype_kind):
        return _unpack_fp4_e2m1(storage, logical_cols)
    if dtype_kind in (int(DType.MXE4M3), int(DType.E4M3)):
        return storage.view(torch.float8_e4m3fn).float()[..., :logical_cols]
    raise ValueError(f"Unsupported quantized dtype kind: {dtype_kind}")


def compute_reference_dsfp8(
    *,
    cfg,
    weights_ref: torch.Tensor,
    activations_ref: torch.Tensor,
    sf_weights: torch.Tensor,
    sf_activations: torch.Tensor,
    token_layout: TokenLayout,
    in_hidden: int,
    m_dim: int,
    n_dim: int,
) -> torch.Tensor:
    """
    DeepSeek FP8 reference: per-128K FP32 dequant before accumulation.
    For DSFP8, the scale factors are separate FP32 tensors and are applied per K-block.
    The reference implementation needs to loop over expanded token, expert, output tile, and K-block.
    """
    ref_gemm = torch.zeros(
        (m_dim, n_dim), dtype=torch.float32, device=weights_ref.device
    )
    k_block = 128
    num_k_blocks = _round_up(in_hidden, k_block) // k_block

    if cfg.is_swap_ab:
        num_weight_tiles = _round_up(m_dim, cfg.tile_m) // cfg.tile_m
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            act_sf_row = token_idx if cfg.has_routed_act else expanded_idx
            for m_tile in range(num_weight_tiles):
                m0 = m_tile * cfg.tile_m
                m1 = min(m0 + cfg.tile_m, m_dim)
                for kb in range(num_k_blocks):
                    k0 = kb * k_block
                    k1 = min(k0 + k_block, in_hidden)
                    dq_w = sf_weights[expert_idx, m_tile, kb]
                    dq_act = sf_activations[kb, act_sf_row]
                    ref_gemm[m0:m1, expanded_idx] += (
                        weights_ref[expert_idx, m0:m1, k0:k1] * dq_w
                    ) @ (activations_ref[token_idx, k0:k1] * dq_act)
        return ref_gemm

    num_weight_tiles = _round_up(n_dim, cfg.tile_n) // cfg.tile_n
    for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
        if expert_idx < 0:
            continue
        token_idx = token_layout.expanded_to_token[expanded_idx]
        act_sf_row = token_idx if cfg.has_routed_act else expanded_idx
        for n_tile in range(num_weight_tiles):
            n0 = n_tile * cfg.tile_n
            n1 = min(n0 + cfg.tile_n, n_dim)
            for kb in range(num_k_blocks):
                k0 = kb * k_block
                k1 = min(k0 + k_block, in_hidden)
                dq_w = sf_weights[expert_idx, n_tile, kb]
                dq_act = sf_activations[kb, act_sf_row]
                ref_gemm[expanded_idx, n0:n1] += (
                    activations_ref[token_idx, k0:k1] * dq_act
                ) @ (weights_ref[expert_idx, n0:n1, k0:k1] * dq_w).T
    return ref_gemm


def _dequantize_block_scaled_tensor(
    storage: torch.Tensor,
    sf: torch.Tensor,
    *,
    dtype_kind: int,
    logical_cols: int,
    layout: int,
    sf_vec_size: int,
    sf_torch_dtype,
) -> torch.Tensor:
    """Dequantize ``[experts, rows, K-storage]`` block-scaled data to float32."""
    unpacked = _decode_quantized_storage(
        storage,
        dtype_kind=dtype_kind,
        logical_cols=logical_cols,
    )
    out = torch.empty(
        (*unpacked.shape[:-1], logical_cols),
        dtype=torch.float32,
        device=storage.device,
    )
    rows_per_expert = unpacked.shape[1]
    for expert_idx in range(unpacked.shape[0]):
        for row_idx in range(unpacked.shape[1]):
            sf_row_idx = expert_idx * rows_per_expert + row_idx
            out[expert_idx, row_idx, :] = _apply_fp4_sf_to_row(
                unpacked[expert_idx, row_idx, :],
                sf,
                row_idx=sf_row_idx,
                layout=layout,
                sf_vec_size=sf_vec_size,
                sf_torch_dtype=sf_torch_dtype,
            )
    return out


def _dequantize_fp4_activation_row(
    unpacked_compact: torch.Tensor,
    sf_activations: torch.Tensor,
    *,
    token_idx: int,
    expanded_idx: int,
    expert_idx: int,
    cfg,
    activation_sf_layout: int,
    num_experts: int,
    sf_k: int,
) -> torch.Tensor:
    """Dequantize one logical activation row for routed or expanded MoE input."""
    if cfg.has_routed_sfs:
        return _apply_fp4_sf_to_row(
            unpacked_compact[token_idx, :],
            sf_activations,
            row_idx=token_idx,
            layout=int(SfLayout.LINEAR),
            num_padded_sf_cols=_round_up(sf_k, 16),
        )
    return _apply_fp4_sf_to_row(
        unpacked_compact[token_idx, :],
        sf_activations,
        row_idx=expanded_idx,
        layout=activation_sf_layout,
    )


def _dequantize_block_scaled_activation_row(
    compact_storage: torch.Tensor,
    sf_activations: torch.Tensor,
    *,
    dtype_kind: int,
    logical_cols: int,
    token_idx: int,
    expanded_idx: int,
    expert_idx: int,
    cfg,
    activation_sf_layout: int,
    num_experts: int,
    sf_k: int,
) -> torch.Tensor:
    """Dequantize one logical activation row for MX/NVFP4 input dtypes."""
    decoded = _decode_quantized_storage(
        compact_storage,
        dtype_kind=dtype_kind,
        logical_cols=logical_cols,
    )
    if cfg.has_routed_sfs:
        return _apply_fp4_sf_to_row(
            decoded[token_idx, :],
            sf_activations,
            row_idx=token_idx,
            layout=int(SfLayout.LINEAR),
            sf_vec_size=cfg.sf_vec_size,
            sf_torch_dtype=_sf_torch_dtype(cfg),
            num_padded_sf_cols=_round_up(sf_k, 16),
        )
    return _apply_fp4_sf_to_row(
        decoded[token_idx, :],
        sf_activations,
        row_idx=expanded_idx,
        layout=activation_sf_layout,
        sf_vec_size=cfg.sf_vec_size,
        sf_torch_dtype=_sf_torch_dtype(cfg),
    )


def _create_sf_tensor_mma_layout(
    *,
    num_experts: int,
    mn: int,
    sf_k: int,
    layout: int,
    device: str,
    cfg,
) -> torch.Tensor:
    """Create random scale factors in a generated tiled SF layout."""
    outer = mn * num_experts
    if layout == int(SfLayout.R8c4):
        total = math.ceil(outer / 8) * math.ceil(sf_k / 4) * 32
        return _random_sf_bytes((total,), cfg=cfg, device=device, layout_kind="mma")

    atom_m0, atom_m1, atom_k = 32, 4, 4
    m_tiles = math.ceil(outer / (atom_m0 * atom_m1))
    k_tiles = math.ceil(sf_k / atom_k)

    # R128c4 stores 128x4 SF tiles as contiguous 512B blocks. Generated
    # TensorMaps split that as 256x2 for TMA legality.
    shape5d = (atom_m0, atom_m1, m_tiles, atom_k, k_tiles)
    order = (2, 1, 4, 0, 3)
    total = 1
    for dim in order:
        total *= shape5d[dim]
    return _random_sf_bytes((total,), cfg=cfg, device=device, layout_kind="mma")


def _create_sf_linear(
    *,
    num_tokens: int,
    sf_k_dim: int,
    device: str,
    cfg,
) -> torch.Tensor:
    """Create random linear scale factors for routed activation SF gather."""
    sf_k_padded = _round_up(sf_k_dim, 16)
    return _random_sf_bytes(
        (num_tokens * sf_k_padded,),
        cfg=cfg,
        device=device,
        layout_kind="linear",
    )


def _sf_c_numel(output_m: int, n: int, layout: int, sf_block_size: int = 16) -> int:
    """Number of output SF bytes for generated C SF layout."""
    m_group = sf_block_size * 4
    m_groups = _round_up(output_m, m_group) // m_group
    if layout == int(SfLayout.R8c4):
        return _round_up(n, 8) // 8 * m_groups * 32
    if layout == int(SfLayout.R128c4):
        return _round_up(n, 128) // 128 * m_groups * 512
    raise ValueError(f"Unsupported sfLayoutC for quantized output: {layout}")


def _dsfp8_c_numel(output_m: int, n: int, *, is_swap_ab: bool) -> int:
    """Number of FP32 DeepSeek output dq-scale values."""
    if is_swap_ab:
        return _round_up(output_m, 128) // 128 * n
    return _round_up(n, 128) // 128 * output_m


def _dsfp8_c_scale_index_grid(
    *, output_m: int, n: int, is_swap_ab: bool, device: str
) -> torch.Tensor:
    m_idx = torch.arange(output_m, device=device).view(output_m, 1)
    n_idx = torch.arange(n, device=device).view(1, n)
    if is_swap_ab:
        return (m_idx // 128) * n + n_idx
    return (n_idx // 128) * output_m + m_idx


def _reference_dsfp8_c_scales(ref: torch.Tensor, *, is_swap_ab: bool) -> torch.Tensor:
    """Compute DeepSeek FP8 C dequant scales for an FP32 reference."""
    output_m, n = ref.shape
    sf_c = torch.zeros(
        (_dsfp8_c_numel(output_m, n, is_swap_ab=is_swap_ab),),
        dtype=torch.float32,
        device=ref.device,
    )
    if is_swap_ab:
        for m0 in range(0, output_m, 128):
            m1 = min(m0 + 128, output_m)
            block = ref[m0:m1, :].abs().amax(dim=0)
            sf_c[(m0 // 128) * n : (m0 // 128) * n + n] = block / 448.0
    else:
        for n0 in range(0, n, 128):
            n1 = min(n0 + 128, n)
            block = ref[:, n0:n1].abs().amax(dim=1)
            sf_c[(n0 // 128) * output_m : (n0 // 128 + 1) * output_m] = block
            sf_c[(n0 // 128) * output_m : (n0 // 128 + 1) * output_m] /= 448.0
    return sf_c


def _quantized_c_numel(cfg, output_m: int, n: int) -> int:
    elem_count = output_m * n
    if cfg.uses_mxfp8_output_quant:
        return elem_count
    return _round_up(elem_count, 2) // 2


def _sf_c_index_grid(
    *,
    output_m: int,
    n: int,
    sf_layout_c: int,
    sf_block_size: int,
    device: str,
) -> torch.Tensor:
    m_idx = torch.arange(output_m, device=device).view(output_m, 1)
    n_idx = torch.arange(n, device=device).view(1, n)
    m_block = m_idx // sf_block_size
    m_groups = _round_up(output_m, sf_block_size * 4) // (sf_block_size * 4)
    if sf_layout_c == int(SfLayout.R8c4):
        return (
            (n_idx // 8) * (m_groups * 32)
            + (m_block // 4) * 32
            + (n_idx % 8) * 4
            + (m_block % 4)
        )
    if sf_layout_c == int(SfLayout.R128c4):
        return (
            (n_idx // 128) * (m_groups * 512)
            + (m_block // 4) * 512
            + (n_idx % 32) * 16
            + ((n_idx % 128) // 32) * 4
            + (m_block % 4)
        )
    raise ValueError(f"Unsupported sfLayoutC for quantized output: {sf_layout_c}")


def _dequantize_fp4_output(
    c_storage: torch.Tensor,
    sf_c: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
    sf_layout_c: int,
    uses_mxfp4: bool,
) -> torch.Tensor:
    """Decode packed E2M1 C and its NVFP4/MXFP4 SF-C into a logical view."""
    elem_count = output_m * n
    packed = c_storage[: _round_up(elem_count, 2) // 2]
    fp4_table = _fp4_e2m1_values(c_storage.device)
    nibbles = torch.empty(
        (packed.numel() * 2,), dtype=torch.long, device=c_storage.device
    )
    nibbles[0::2] = (packed & 0x0F).to(torch.long)
    nibbles[1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    fp4_vals = fp4_table[nibbles[:elem_count]]

    if is_swap_ab:
        fp4_view = torch.as_strided(fp4_vals, (output_m, n), (1, output_m))
    else:
        fp4_view = fp4_vals.view(output_m, n)

    sf_idx = _sf_c_index_grid(
        output_m=output_m,
        n=n,
        sf_layout_c=sf_layout_c,
        sf_block_size=32 if uses_mxfp4 else 16,
        device=c_storage.device,
    )
    sf_dtype = torch.float8_e8m0fnu if uses_mxfp4 else torch.float8_e4m3fn
    sf_vals = sf_c.view(sf_dtype).float()
    return fp4_view * sf_vals[sf_idx]


def _dequantize_mxfp8_output(
    c_storage: torch.Tensor,
    sf_c: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
    sf_layout_c: int,
) -> torch.Tensor:
    """Decode E4M3 C and UE8M0 SF-C into a float32 logical view."""
    elem_count = output_m * n
    fp8_vals = c_storage[:elem_count].view(torch.float8_e4m3fn).float()
    if is_swap_ab:
        fp8_view = torch.as_strided(fp8_vals, (output_m, n), (1, output_m))
    else:
        fp8_view = fp8_vals.view(output_m, n)

    sf_idx = _sf_c_index_grid(
        output_m=output_m,
        n=n,
        sf_layout_c=sf_layout_c,
        sf_block_size=32,
        device=c_storage.device,
    )
    sf_vals = sf_c.view(torch.float8_e8m0fnu).float()
    return fp8_view * sf_vals[sf_idx]


def _decode_fp8_output(
    c_storage: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
) -> torch.Tensor:
    """Decode plain per-tensor FP8 C storage into a logical float32 view."""
    elem_count = output_m * n
    fp8_vals = c_storage[:elem_count].view(torch.float8_e4m3fn).float()
    if is_swap_ab:
        return torch.as_strided(fp8_vals, (output_m, n), (1, output_m))
    return fp8_vals.view(output_m, n)


def _dequantize_dsfp8_output(
    c_storage: torch.Tensor,
    sf_c: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
) -> torch.Tensor:
    """Decode E4M3 C using DeepSeek FP32 per-128-hidden dequant scales."""
    fp8_view = _decode_fp8_output(
        c_storage,
        output_m=output_m,
        n=n,
        is_swap_ab=is_swap_ab,
    )
    sf_idx = _dsfp8_c_scale_index_grid(
        output_m=output_m,
        n=n,
        is_swap_ab=is_swap_ab,
        device=c_storage.device,
    )
    return fp8_view * sf_c.float()[sf_idx]


# Row shuffle indices that reorder rows within blocks to optimize SMEM
# write patterns in the
# transposed-output (16dp256b) epilogue.
_SHUFFLE_BLOCK_16 = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
_SHUFFLE_BLOCK_32 = [
    0,
    8,
    16,
    24,
    1,
    9,
    17,
    25,
    2,
    10,
    18,
    26,
    3,
    11,
    19,
    27,
    4,
    12,
    20,
    28,
    5,
    13,
    21,
    29,
    6,
    14,
    22,
    30,
    7,
    15,
    23,
    31,
]


def reorder_rows_for_gated_act(tensor: torch.Tensor) -> torch.Tensor:
    """Interleave first half and second half rows for gated activation.

      Original:  [r0, r1, ..., r(M/2-1), r(M/2), ..., r(M-1)]
      Reordered: [r0, r(M/2), r1, r(M/2+1), ..., r(M/2-1), r(M-1)]

    Applied to weights so adjacent output rows in the MMA accumulator
    contain (gate, up) pairs for gated activations.

    Args:
        tensor: (..., M, K) — last two dims are rows × cols. M must be even.
    """
    M = tensor.shape[-2]
    assert M % 2 == 0, f"M must be even for gated act reorder, got {M}"
    half = M // 2
    out = torch.empty_like(tensor)
    for i in range(half):
        out[..., 2 * i, :] = tensor[..., i, :]
        out[..., 2 * i + 1, :] = tensor[..., i + half, :]
    return out


def shuffle_matrix(tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """Reorder rows of a matrix for the transposed-output epilogue.

    Rows are permuted within blocks of 16 or 32 (depending on epilogueTileM)
    to match the SMEM write pattern of the 16dp256b T2R instruction.

    Args:
        tensor: (..., M, K) — last two dims are rows × cols.
        epilogue_tile_m: epilogue tile M dimension (128 → block=32, else 16).

    Returns:
        Shuffled tensor with same shape.
    """
    block_size = 32 if epilogue_tile_m % 128 == 0 else 16
    indices = _SHUFFLE_BLOCK_32 if block_size == 32 else _SHUFFLE_BLOCK_16

    M = tensor.shape[-2]
    if block_size > M:
        # Matrix too small for shuffling — return as-is.
        return tensor.clone()
    out = torch.empty_like(tensor)
    for mi in range(M):
        block_idx = mi // block_size
        src_in_block = mi % block_size
        dst_in_block = indices[src_in_block]
        dst_row = block_idx * block_size + dst_in_block
        out[..., dst_row, :] = tensor[..., mi, :]
    return out


def _block_major_k_weight_tensor(tensor: torch.Tensor, cfg) -> torch.Tensor:
    """Pack per-expert weight storage into TRT-LLM Gen BlockMajorK layout."""
    if not cfg.uses_block_major_k_weight:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(
            "BlockMajorK weight packing expects [experts, Mn, K-storage], "
            f"got shape {tuple(tensor.shape)}"
        )
    from flashinfer.fused_moe import convert_to_block_layout

    packed = []
    for expert_idx in range(tensor.shape[0]):
        expert_matrix = tensor[expert_idx].contiguous().view(torch.uint8)
        packed.append(convert_to_block_layout(expert_matrix, cfg.block_major_k_bytes))
    return torch.stack(packed, dim=0).contiguous()


def _shuffle_bias_m_storage(
    bias_torch: torch.Tensor, epilogue_tile_m: int
) -> torch.Tensor:
    """Store per-expert M bias with the same row order as ``shuffleMatrixA``."""
    block_size = 32 if epilogue_tile_m % 128 == 0 else 16
    if bias_torch.shape[-1] % block_size != 0:
        raise ValueError(
            "BiasType.M shuffled storage requires the M dimension to be a "
            f"multiple of {block_size}, got {bias_torch.shape[-1]}"
        )
    return shuffle_matrix(bias_torch.unsqueeze(-1), epilogue_tile_m).squeeze(-1)


def _kernel_weight_row_source_indices(
    num_rows: int,
    *,
    is_gated: bool,
    epilogue_tile_m: int,
    shuffle_for_swap_ab: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return the original weight row stored at each kernel weight row."""
    row_ids = torch.arange(num_rows, dtype=torch.long, device=device).view(
        1, num_rows, 1
    )
    if is_gated:
        row_ids = reorder_rows_for_gated_act(row_ids)
    if shuffle_for_swap_ab:
        row_ids = shuffle_matrix(row_ids, epilogue_tile_m)
    return row_ids.view(num_rows)


def _permute_mma_sf_rows(
    sf: torch.Tensor,
    *,
    row_source_indices: torch.Tensor,
    logical_cols: int,
    layout: int,
    num_experts: int,
    sf_vec_size: int,
) -> torch.Tensor:
    """Permute tiled MMA scale-factor rows to match a data-row permutation."""
    out = torch.empty_like(sf)
    sf_cols = _round_up(logical_cols, sf_vec_size) // sf_vec_size
    row_sources = [int(v) for v in row_source_indices.detach().cpu().tolist()]
    rows_per_expert = len(row_sources)
    for expert_idx in range(num_experts):
        for dst_row, src_row in enumerate(row_sources):
            src_offsets = _sf_offsets(
                row_idx=expert_idx * rows_per_expert + src_row,
                sf_cols=sf_cols,
                layout=layout,
                device=sf.device,
            )
            dst_offsets = _sf_offsets(
                row_idx=expert_idx * rows_per_expert + dst_row,
                sf_cols=sf_cols,
                layout=layout,
                device=sf.device,
            )
            out[dst_offsets] = sf[src_offsets]
    return out


def _kernel_weight_sf(
    sf_weights: torch.Tensor | None,
    *,
    cfg,
    out_hidden: int,
    in_hidden: int,
    num_experts: int,
    is_gated: bool,
    weight_sf_layout: int | None,
    shuffle_for_swap_ab: bool,
) -> torch.Tensor | None:
    """Return weight SFs reordered to match kernel weight storage."""
    if sf_weights is None or (not is_gated and not shuffle_for_swap_ab):
        return sf_weights
    assert weight_sf_layout is not None
    row_source_indices = _kernel_weight_row_source_indices(
        out_hidden,
        is_gated=is_gated,
        epilogue_tile_m=cfg.tile_m,
        shuffle_for_swap_ab=shuffle_for_swap_ab,
        device=sf_weights.device,
    )
    return _permute_mma_sf_rows(
        sf_weights,
        row_source_indices=row_source_indices,
        logical_cols=in_hidden,
        layout=weight_sf_layout,
        num_experts=num_experts,
        sf_vec_size=cfg.sf_vec_size,
    )


def _parse_overrides(args):
    overrides = {
        "batch_mode": int(
            {"n": BatchMode.BATCH_N, "m": BatchMode.BATCH_M}[args.batch_mode]
        ),
        "route_act": int(
            {
                "false": RouteImpl.NONE,
                "none": RouteImpl.NONE,
                "tma": RouteImpl.TMA,
                "ldgsts": RouteImpl.LDGSTS,
                "ldgplussts": RouteImpl.LDG_PLUS_STS,
            }[args.route_act]
        ),
        "route_sfs_act": int(
            {
                "false": RouteImpl.NONE,
                "none": RouteImpl.NONE,
                "tma": RouteImpl.TMA,
                "ldgsts": RouteImpl.LDGSTS,
                "ldgplussts": RouteImpl.LDG_PLUS_STS,
            }[args.route_sfs_act]
        ),
        "tile_scheduler": int(
            {
                "static": TileScheduler.STATIC,
                "persistent": TileScheduler.PERSISTENT,
            }[args.tile_scheduler]
        ),
        "transpose_mma_output": args.transpose_mma_output,
        "act_kind": int(
            {
                "none": ActKind.NONE,
                "swiglu": ActKind.SWIGLU,
                "geglu": ActKind.GEGLU,
                "relu2": ActKind.RELU2,
                "silu": ActKind.SILU,
                "situ": ActKind.SITU,
            }[args.act_kind]
        ),
        "sf_layout_a": int(
            {
                "8x4": SfLayout.R8c4,
                "r8c4": SfLayout.R8c4,
                "linear": SfLayout.LINEAR,
                "128x4": SfLayout.R128c4,
                "r128c4": SfLayout.R128c4,
            }[args.sf_layout_a]
        ),
        "sf_layout_b": int(
            {
                "8x4": SfLayout.R8c4,
                "r8c4": SfLayout.R8c4,
                "linear": SfLayout.LINEAR,
                "128x4": SfLayout.R128c4,
                "r128c4": SfLayout.R128c4,
            }[args.sf_layout_b]
        ),
        "sf_layout_c": int(
            {
                "8x4": SfLayout.R8c4,
                "r8c4": SfLayout.R8c4,
                "linear": SfLayout.LINEAR,
                "128x4": SfLayout.R128c4,
                "r128c4": SfLayout.R128c4,
            }[args.sf_layout_c]
        ),
        "cluster_m": args.cluster_m,
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "tile_k": args.tile_k,
        "epi_tile_n": args.epi_tile_n,
        "mma_m": args.mma_m,
        "mma_n": args.mma_n,
        "mma_k": args.mma_k,
        "dtype_a": int(DTYPE_NAME_TO_VALUE[args.dtype_a]),
        "dtype_b": int(DTYPE_NAME_TO_VALUE[args.dtype_b]),
        "dtype_c": int(DTYPE_NAME_TO_VALUE[args.dtype_c]),
        "sf_block_size_a": args.sf_block_size_a,
        "sf_block_size_b": args.sf_block_size_b,
        "sf_block_size_c": args.sf_block_size_c,
        "num_stages_a": args.num_stages_a,
        "num_stages_b": args.num_stages_b,
        "num_stages_smem_sfa": args.num_stages_smem_sfa,
        "num_stages_smem_sfb": args.num_stages_smem_sfb,
        "num_stages_tmem_acc": args.num_stages_tmem_acc,
        "num_stages_tmem_sfa": args.num_stages_tmem_sfa,
        "num_stages_tmem_sfb": args.num_stages_tmem_sfb,
        "num_stages_c_smem": args.num_stages_c_smem,
        "use_tma_oob_opt": args.use_tma_oob_opt,
        "use_tma_store": args.use_tma_store,
        "num_load_b_warps": args.num_load_b_warps,
        "num_load_sfb_warps": args.num_load_sfb_warps,
        "epilogue_regs": args.epilogue_regs,
        "mma_regs": args.mma_regs,
        "load_regs": args.load_regs,
        "load_sf_regs": args.load_sf_regs,
        "load_a_regs": args.load_a_regs,
        "load_b_regs": args.load_b_regs,
        "load_sfa_regs": args.load_sfa_regs,
        "load_sfb_regs": args.load_sfb_regs,
        "copy_sf_regs": args.copy_sf_regs,
        "workid_regs": args.workid_regs,
        "padding_regs": args.padding_regs,
        "use_unroll_loop_2x_for_mma": args.use_unroll_loop_2x_for_mma,
        "use_max_tmem_overlap": args.use_max_tmem_overlap,
        "use_early_exit": args.use_early_exit,
        "use_clc_fast_drain": args.use_clc_fast_drain,
        "use_work_throttle": args.use_work_throttle,
        "use_pdl": args.use_pdl,
        "do_pdl_wait_for_num_non_exiting_ctas": (
            args.do_pdl_wait_for_num_non_exiting_ctas
        ),
        "use_global_scales": args.use_global_scales,
        "use_per_token_sf_a": args.use_per_token_sf_a,
        "use_per_token_sf_b": args.use_per_token_sf_b,
        "per_token_sf_dtype": int(DTYPE_NAME_TO_VALUE[args.per_token_sf_dtype]),
        "use_deepseek_fp8": args.use_deepseek_fp8,
        # DeepSeek FP8 will use a single LoadSfAb warp.
        "num_load_sfab_warps": int(args.use_deepseek_fp8 != 0),
        "load_sfab_regs": args.load_sfab_regs,
        "bias_type": args.bias_type,
        "has_gemm1_alpha": args.has_gemm1_alpha,
        "has_gemm1_beta": args.has_gemm1_beta,
        "has_gemm1_clamp_limit": args.has_gemm1_clamp_limit,
    }
    return overrides


def validate_schedule(
    *,
    num_experts,
    num_tokens,
    top_k,
    early_exit_max_token_ctas=0,
    **cfg_overrides,
):
    launch_early_exit_max_token_ctas = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    print(
        f"Validating: num_experts={num_experts}, num_tokens={num_tokens}, "
        f"top_k={top_k}, early_exit_max_token_ctas="
        f"{launch_early_exit_max_token_ctas}, overrides={cfg_overrides}"
    )
    build_batched_gemm_task_manager(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        early_exit_max_token_ctas=launch_early_exit_max_token_ctas,
        **cfg_overrides,
    )
    print("Schedule validation: PASSED")
    return True


def reference_check(
    *,
    num_experts,
    num_tokens,
    top_k,
    problem_n=None,
    problem_k=None,
    seed=42,
    scale_c_value=1.0,
    scale_gate_value=1.0,
    gemm1_alpha_value=None,
    gemm1_beta_value=None,
    gemm1_clamp_limit_value=2.0,
    return_output=False,
    repeat_launches=1,
    output_guard_elements=0,
    early_exit_max_token_ctas=0,
    **cfg_overrides,
):
    """Compile, launch the kernel, compare to a PyTorch reference.

    MoE BatchedGemm semantics:
      weights:     (num_experts, out_hidden, in_hidden)
      activations: (num_tokens, in_hidden), expanded internally by routing
      output:      kernel-logical C[M, N]; swapAB stores that view in M-major
                   physical order, non-swap stores it row-major

    When problem_n / problem_k are None, they default to tile_n / tile_k
    (single-tile test). Set them to realistic values (e.g. 4096) for
    multi-tile benchmarks. ``repeat_launches`` reuses the compiled kernel and
    requires bitwise-stable output and output scale buffers across launches.
    """
    if repeat_launches < 1:
        raise ValueError(f"repeat_launches must be positive, got {repeat_launches}")
    launch_early_exit_max_token_ctas = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    cfg = make_config(**cfg_overrides)

    torch.manual_seed(seed)
    device = "cuda"
    from cutlass.cute.runtime import make_ptr

    # --- Domain-level problem shape ---
    total_tokens = num_tokens
    # swapAB maps FC1 weights to operand A, whose TMA box spans tile_m rows.
    # Keep the default legal for the generated transOut kernels.
    default_out_hidden = cfg.tile_m if cfg.is_swap_ab else cfg.tile_n
    out_hidden = problem_n if problem_n is not None else default_out_hidden
    in_hidden = problem_k if problem_k is not None else cfg.tile_k
    cfg = _runtime_config(cfg, in_hidden)

    if cfg.is_swap_ab and out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    token_tile_size = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
    cluster_dim_in_token = 1 if cfg.is_swap_ab else cfg.cluster_m
    token_layout = _make_token_layout(
        num_tokens=total_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
    )
    launch_early_exit_max_token_ctas = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )
    total_padded_tokens = token_layout.total_padded_tokens

    print(
        f"Reference check: tokens={total_tokens}, padded_tokens={total_padded_tokens}, "
        f"out_hidden={out_hidden}, in_hidden={in_hidden}, experts={num_experts}, "
        f"top_k={top_k}, swap_ab={cfg.is_swap_ab}"
    )
    print(
        f"  tile=({cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}), "
        f"mma=({cfg.mma_m}x{cfg.mma_n}x{cfg.mma_k}), "
        f"stages_a/b={cfg.num_stages_a}/{cfg.num_stages_b}, "
        f"tmem_acc_stages={cfg.num_stages_tmem_acc}"
    )
    print(f"  tokens_per_expert={token_layout.tokens_per_expert}")

    # --- Create domain tensors ---
    # weights:     (num_experts, out_hidden, in_hidden)
    # activations: compact input is (total_tokens, in_hidden). The kernel sees
    # compact input when routed, otherwise expanded/padded token rows.
    if cfg.is_bf16_mma:
        weights_torch = _kaiming_uniform_tensor(
            (num_experts, out_hidden, in_hidden),
            fan_in=in_hidden // 8,
            dtype=torch.bfloat16,
            device=device,
        )
        if os.environ.get("FLASHINFER_PRIMS_TS_BF16_CONSTANT_WEIGHTS") == "1":
            weights_torch.fill_(1.0)
        activation_compact_torch = _kaiming_uniform_tensor(
            (total_tokens, in_hidden),
            # Heruistic to avoid to small values in the output
            fan_in=in_hidden // 8,
            dtype=torch.bfloat16,
            device=device,
        )
        if os.environ.get("FLASHINFER_PRIMS_TS_CASTA_CONSTANT_ACT") == "1":
            activation_compact_torch.fill_(1.0)
        if os.environ.get("FLASHINFER_PRIMS_TS_CASTA_PATTERN_ACT") == "1":
            activation_compact_torch.fill_(0)
            quarter_elems = in_hidden // 4
            for quarter, value in enumerate((1.0, 2.0, 4.0, 6.0)):
                activation_compact_torch[
                    :, quarter * quarter_elems : (quarter + 1) * quarter_elems
                ].fill_(value)
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_bf16_activations(activation_compact_torch, token_layout)
        )
        a_dtype = b_dtype = cutlass.BFloat16
        weight_sf_layout = None
        activation_sf_layout = None
        sf_weights = sf_activations = None
    elif cfg.has_cast_a:
        sf_k = in_hidden // cfg.sf_vec_size
        weight_sf_layout = cfg.sf_layout_a
        if cfg.act_kind != int(ActKind.NONE):
            # Fused-activation: Kaiming-init + quantize the (MXFP4) weights and
            # scale the BF16 activations into the kernel's accumulator regime,
            # mirroring the block-scaled path, instead of random storage + SF.
            data_scale = math.sqrt(in_hidden)
            w_fp32 = (
                _kaiming_uniform_tensor(
                    (num_experts * out_hidden, in_hidden),
                    fan_in=in_hidden,
                    dtype=torch.float32,
                    device=device,
                )
                * data_scale
            )
            w_storage, w_sf = _quantize_block_scaled(
                w_fp32,
                dtype_kind=cfg.dtype_a_kind,
                uses_mx=cfg.uses_mx_scale_factors,
                sf_vec_size=cfg.sf_vec_size,
            )
            weights_torch = w_storage.reshape(num_experts, out_hidden, in_hidden // 2)
            w_sf_total = _create_sf_tensor_mma_layout(
                num_experts=num_experts,
                mn=out_hidden,
                sf_k=sf_k,
                layout=weight_sf_layout,
                device=device,
                cfg=cfg,
            ).numel()
            sf_weights = _build_sf_buffer(
                w_sf,
                total_bytes=w_sf_total,
                layout=weight_sf_layout,
                data_blk_cols=sf_k,
                device=device,
            )
            activation_compact_torch = (
                _kaiming_uniform_tensor(
                    (total_tokens, in_hidden),
                    fan_in=in_hidden,
                    dtype=torch.float32,
                    device=device,
                )
                * data_scale
            ).to(torch.bfloat16)
        else:
            weights_torch = _random_quantized_tensor(
                (num_experts, out_hidden, in_hidden),
                dtype_kind=cfg.dtype_a_kind,
                fan_in=in_hidden,
                device=device,
            )
            if os.environ.get("FLASHINFER_PRIMS_TS_CASTA_PATTERN_WEIGHTS") == "1":
                weights_torch.fill_(0)
                quarter_bytes = in_hidden // 8
                for quarter, value in enumerate((0x22, 0x44, 0x66, 0x77)):
                    weights_torch[
                        :, :, quarter * quarter_bytes : (quarter + 1) * quarter_bytes
                    ].fill_(value)
            elif os.environ.get("FLASHINFER_PRIMS_TS_CASTA_CONSTANT_WEIGHTS") == "1":
                weights_torch.fill_(0x22)
            activation_compact_torch = _kaiming_uniform_tensor(
                (total_tokens, in_hidden),
                fan_in=in_hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            if os.environ.get("FLASHINFER_PRIMS_TS_CASTA_CONSTANT_ACT") == "1":
                activation_compact_torch.fill_(1.0)
            if os.environ.get("FLASHINFER_PRIMS_TS_CASTA_PATTERN_ACT") == "1":
                activation_compact_torch.fill_(0)
                quarter_elems = in_hidden // 4
                for quarter, value in enumerate((1.0, 2.0, 4.0, 6.0)):
                    activation_compact_torch[
                        :, quarter * quarter_elems : (quarter + 1) * quarter_elems
                    ].fill_(value)
            sf_weights = _create_sf_tensor_mma_layout(
                num_experts=num_experts,
                mn=out_hidden,
                sf_k=sf_k,
                layout=weight_sf_layout,
                device=device,
                cfg=cfg,
            )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_bf16_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = cutlass.BFloat16
        activation_sf_layout = None
        sf_activations = None
    elif cfg.is_fp8_mma:
        weights_torch = _random_quantized_tensor(
            (num_experts, out_hidden, in_hidden),
            dtype_kind=cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind,
            fan_in=in_hidden,
            device=device,
        )
        activation_compact_torch = _random_quantized_tensor(
            (total_tokens, in_hidden),
            dtype_kind=cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b_kind)
        weight_sf_layout = None
        activation_sf_layout = None
        sf_weights = sf_activations = None
        if cfg.has_deepseek_fp8:
            sf_weights, sf_activations = _create_deepseek_fp8_sf_tensors(
                cfg,
                num_experts=num_experts,
                num_tokens=total_tokens,
                out_hidden=out_hidden,
                in_hidden=in_hidden,
                total_padded_tokens=total_padded_tokens,
                device=device,
            )
    else:
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b_kind)
        sf_vec_size = cfg.sf_vec_size

        sf_k = in_hidden // sf_vec_size

        # Routed operand's SF uses linear GMEM layout for both TMA gather4 and
        # LDGSTS. The loader converts it to the R128c4 SMEM layout consumed by
        # tcgen05_cp.
        activation_sf_layout = cfg.sf_layout_b if cfg.is_swap_ab else cfg.sf_layout_a
        weight_sf_layout = cfg.sf_layout_a if cfg.is_swap_ab else cfg.sf_layout_b

        if cfg.act_kind != int(ActKind.NONE):
            # Fused-activation runs: Kaiming-init the data in fp32 and then
            # quantize. This keeps GEMM outputs O(1) instead of summing large
            # random products that catastrophically cancel to tiny values
            # straddling the gated-activation clamp -- the cause of sparse ref-check
            # failures on large configs.
            (
                weights_torch,
                sf_weights,
                activation_compact_torch,
                sf_activations,
            ) = _kaiming_quantized_block_scaled_inputs(
                num_experts=num_experts,
                out_hidden=out_hidden,
                in_hidden=in_hidden,
                total_tokens=total_tokens,
                total_padded_tokens=total_padded_tokens,
                sf_k=sf_k,
                weight_dtype_kind=(
                    cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind
                ),
                activation_dtype_kind=(
                    cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind
                ),
                uses_mx=cfg.uses_mx_scale_factors,
                sf_vec=cfg.sf_vec_size,
                weight_sf_layout=weight_sf_layout,
                activation_sf_layout=activation_sf_layout,
                has_routed_sfs=cfg.has_routed_sfs,
                token_layout=token_layout,
                cfg=cfg,
                device=device,
            )
        else:
            weights_torch = _random_quantized_tensor(
                (num_experts, out_hidden, in_hidden),
                dtype_kind=cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind,
                fan_in=in_hidden,
                device=device,
            )
            activation_compact_torch = _random_quantized_tensor(
                (total_tokens, in_hidden),
                dtype_kind=cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind,
                fan_in=in_hidden,
                device=device,
            )
            if cfg.has_routed_sfs:
                # Activation SF = linear, weight SF = R128c4
                sf_activations = _create_sf_linear(
                    num_tokens=total_tokens,
                    sf_k_dim=sf_k,
                    device=device,
                    cfg=cfg,
                )
                sf_weights = _create_sf_tensor_mma_layout(
                    num_experts=num_experts,
                    mn=out_hidden,
                    sf_k=sf_k,
                    layout=weight_sf_layout,
                    device=device,
                    cfg=cfg,
                )
            else:
                sf_weights = _create_sf_tensor_mma_layout(
                    num_experts=num_experts,
                    mn=out_hidden,
                    sf_k=sf_k,
                    layout=weight_sf_layout,
                    device=device,
                    cfg=cfg,
                )
                sf_activations = _create_sf_tensor_mma_layout(
                    num_experts=1,
                    mn=total_padded_tokens,
                    sf_k=sf_k,
                    layout=activation_sf_layout,
                    device=device,
                    cfg=cfg,
                )

        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )

    # --- Map to kernel GEMM dimensions ---
    # Kernel computes C[M,N] = A[M,K] × B[N,K]^T per expert.
    # swapAB:     A=weights(out_hidden×in_hidden), B=activations(tokens×in_hidden)
    #             M=out_hidden, N=tokens, C is M-major
    # non-swapAB: A=activations(tokens×in_hidden), B=weights(out_hidden×in_hidden)
    #             M=tokens, N=out_hidden, C is row-major
    # For gated activations: interleave gate/up weight rows so adjacent accumulator
    # columns contain (gate, up) pairs. The kernel output has out_hidden
    # columns; the epilogue applies the activation to produce out_hidden/2 columns.
    is_gated = _is_gated_act_kind(cfg.act_kind)
    preprocessed_weights = weights_torch
    if is_gated:
        preprocessed_weights = reorder_rows_for_gated_act(weights_torch)

    if cfg.is_swap_ab:
        M, N, K, L = out_hidden, total_padded_tokens, in_hidden, num_experts
        # Shuffle weight rows for the 16x256b T2R epilogue pattern.
        shuffled_weights = shuffle_matrix(preprocessed_weights, cfg.tile_m)
        kernel_a = _block_major_k_weight_tensor(shuffled_weights, cfg)
        kernel_b = activations_torch
        sf_a = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_for_swap_ab=(not cfg.has_deepseek_fp8),
        )
        sf_b = sf_activations
    else:
        M, N, K, L = total_padded_tokens, out_hidden, in_hidden, num_experts
        kernel_a = activations_torch
        kernel_b = _block_major_k_weight_tensor(preprocessed_weights, cfg)
        sf_a = sf_activations
        sf_b = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_for_swap_ab=False,
        )

    logical_output_m, logical_output_n = _logical_output_shape(cfg, M, N, is_gated)
    plain_output_dtype = _plain_output_torch_dtype(cfg)
    if cfg.has_epilogue_quant:
        if output_guard_elements > 0:
            raise ValueError("output_guard_elements is not supported with FP4 C output")
        c_storage = torch.zeros(
            (_quantized_c_numel(cfg, logical_output_m, logical_output_n),),
            dtype=torch.uint8,
            device=device,
        )
        c_torch = None
        sf_c_torch = torch.zeros(
            (
                _sf_c_numel(
                    logical_output_m,
                    logical_output_n,
                    cfg.sf_layout_c,
                    cfg.output_sf_block_size_c,
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
        output_guard_value = None
        c_guard = None
    elif cfg.uses_fp8_output:
        if output_guard_elements > 0:
            raise ValueError("output_guard_elements is not supported with FP8 C output")
        c_storage = torch.zeros(
            (logical_output_m * logical_output_n,),
            dtype=torch.uint8,
            device=device,
        )
        c_torch = None
        if cfg.has_deepseek_fp8_c_scale:
            sf_c_torch = torch.zeros(
                (
                    _dsfp8_c_numel(
                        logical_output_m,
                        logical_output_n,
                        is_swap_ab=cfg.is_swap_ab,
                    ),
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
        output_guard_value = None
        c_guard = None
    elif output_guard_elements > 0:
        output_guard_value = torch.tensor(
            123.0, dtype=plain_output_dtype, device=device
        )
        c_storage = torch.full(
            (logical_output_m * logical_output_n + output_guard_elements,),
            output_guard_value,
            dtype=plain_output_dtype,
            device=device,
        )
        if cfg.is_swap_ab:
            # Generated transOut kernels store C in M-major physical order.
            # Keep a logical [output_M, output_N] view while matching that layout.
            c_torch = torch.as_strided(
                c_storage[: logical_output_m * logical_output_n],
                (logical_output_m, logical_output_n),
                (1, logical_output_m),
            )
        else:
            c_torch = c_storage[: logical_output_m * logical_output_n].view(
                logical_output_m,
                logical_output_n,
            )
        c_torch.zero_()
        c_guard = c_storage[logical_output_m * logical_output_n :]
        sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    else:
        output_guard_value = None
        c_guard = None
        if cfg.is_swap_ab:
            c_storage = torch.zeros(
                (logical_output_m * logical_output_n,),
                dtype=plain_output_dtype,
                device=device,
            )
            c_torch = torch.as_strided(
                c_storage,
                (logical_output_m, logical_output_n),
                (1, logical_output_m),
            )
        else:
            c_torch = torch.zeros(
                (logical_output_m, logical_output_n),
                dtype=plain_output_dtype,
                device=device,
            )
        sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    if cfg.has_bias_m:
        bias_logical_torch = (
            torch.randn((L, M), dtype=torch.float32, device=device) * 0.01
        )
        bias_torch = _shuffle_bias_m_storage(
            bias_logical_torch, cfg.epi_tile_m
        ).contiguous()
    else:
        bias_logical_torch = torch.zeros((L, M), dtype=torch.float32, device=device)
        bias_torch = bias_logical_torch
    scale_c_torch = torch.full((L,), scale_c_value, dtype=torch.float32, device=device)
    scale_gate_torch = torch.full(
        (L,), scale_gate_value, dtype=torch.float32, device=device
    )
    gemm1_alpha_torch = _make_per_expert_f32_tensor(
        gemm1_alpha_value,
        default=1.0,
        num_experts=L,
        device=device,
    )
    gemm1_beta_torch = _make_per_expert_f32_tensor(
        gemm1_beta_value,
        default=1.0 if cfg.act_kind == int(ActKind.SITU) else 0.0,
        num_experts=L,
        device=device,
    )
    gemm1_clamp_limit_torch = _make_per_expert_f32_tensor(
        gemm1_clamp_limit_value if cfg.has_gemm1_clamp_limit else None,
        default=0.0,
        num_experts=L,
        device=device,
    )

    num_token_tiles = token_layout.num_token_tiles
    tile_idx_list, mn_limit_list, route_map_list = _launch_metadata_lists(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )
    tile_idx = torch.tensor(tile_idx_list, dtype=torch.int32, device=device)

    # --- Create typed CuTe pointers ---
    a_dp = make_ptr(
        a_dtype, kernel_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        b_dtype, kernel_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )

    sf_ab_dtype = _sf_ab_cutlass_dtype(cfg)
    sf_c_dtype = _sf_c_cutlass_dtype(cfg)
    if sf_a is not None:
        sfa_dp = make_ptr(
            sf_ab_dtype, sf_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
        )
    else:
        sfa_dp = a_dp  # unused
    if sf_b is not None:
        sfb_dp = make_ptr(
            sf_ab_dtype, sf_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
        )
    else:
        sfb_dp = b_dp  # unused

    if cfg.has_epilogue_quant or cfg.uses_fp8_output:
        c_dtype = (
            cutlass.Float8E4M3FN
            if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
            else cutlass.Float4E2M1FN
        )
        c_dp = make_ptr(
            c_dtype,
            c_storage.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c_dp = make_ptr(
            _plain_output_cutlass_dtype(cfg),
            c_torch.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    sf_c_dp = make_ptr(
        sf_c_dtype,
        sf_c_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_alpha_dp = make_ptr(
        cutlass.Float32,
        gemm1_alpha_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_beta_dp = make_ptr(
        cutlass.Float32,
        gemm1_beta_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_clamp_limit_dp = make_ptr(
        cutlass.Float32,
        gemm1_clamp_limit_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_dtype = _per_token_torch_dtype(cfg)
    per_token_count = max(total_tokens, total_padded_tokens, M, N)
    per_token_sf_a_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    per_token_sf_b_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    if cfg.has_per_token_sf_a:
        per_token_sf_a_torch = (
            torch.empty(per_token_sf_a_torch.shape, dtype=torch.float32, device=device)
            .uniform_(0.75, 1.25)
            .to(per_token_dtype)
        )
    if cfg.has_per_token_sf_b:
        per_token_sf_b_torch = (
            torch.empty(per_token_sf_b_torch.shape, dtype=torch.float32, device=device)
            .uniform_(0.75, 1.25)
            .to(per_token_dtype)
        )
    per_token_cutlass_dtype = _per_token_cutlass_dtype(cfg)
    per_token_sf_a_dp = make_ptr(
        per_token_cutlass_dtype,
        per_token_sf_a_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        per_token_cutlass_dtype,
        per_token_sf_b_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    # Per-token-tile local valid row count. The launch dimensions come from
    # the padded per-expert layout, while TS resources consume local
    # tile limits for TmaOobOpt and epilogue store predication.
    mn_limit = torch.tensor(mn_limit_list, dtype=torch.int32, device=device)

    # Route map and activation pointer for gather (LDGSTS or TMA gather4)
    if cfg.has_routed_act or cfg.has_routed_sfs:
        route_map = torch.tensor(route_map_list, dtype=torch.int32, device=device)
    else:
        route_map = torch.zeros(1, dtype=torch.int32, device=device)

    if cfg.has_routed_act:
        act_torch = activation_compact_torch
    else:
        act_torch = torch.zeros(1, dtype=torch.bfloat16, device=device)

    route_map_dp = make_ptr(
        cutlass.Int32,
        route_map.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas = torch.tensor(
        [num_token_tiles], dtype=torch.int32, device=device
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    total_num_padded_tokens_torch = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=device
    )
    total_num_padded_tokens_dp = make_ptr(
        cutlass.Int32,
        total_num_padded_tokens_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        cutlass.BFloat16,
        act_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    stream = get_default_stream()

    # --- Compile and launch ---
    # Package the launch pointers so the shared compile/launch helpers can be
    # reused (same single ``cute.compile`` path as compile/benchmark).
    ref_io = {
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "total_num_padded_tokens_dp": total_num_padded_tokens_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "shape": (M, N, K, L, total_tokens),
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
    }
    print("Compiling kernel...")
    compiled_fn = _compile_for_launch(ref_io, stream)
    print("Launching kernel...")
    first_output = None
    first_scale = None
    outputs_deterministic = True
    scales_deterministic = True
    for _ in range(repeat_launches):
        compiled_fn(*_launch_arg_tuple(ref_io, stream))
        torch.cuda.synchronize()
        if repeat_launches > 1:
            output_tensor = (
                c_storage
                if (cfg.has_epilogue_quant or cfg.uses_fp8_output)
                else c_torch
            )
            if first_output is None:
                first_output = output_tensor.detach().clone()
                if cfg.has_epilogue_quant or cfg.has_deepseek_fp8_c_scale:
                    first_scale = sf_c_torch.detach().clone()
            else:
                outputs_deterministic &= torch.equal(first_output, output_tensor)
                if first_scale is not None:
                    scales_deterministic &= torch.equal(first_scale, sf_c_torch)
    # Unload the per-check CUDA library at a deterministic point after the
    # launch is synchronized. Letting Python GC do this later can overlap
    # library unload with the next clustered persistent compile/launch.
    del compiled_fn
    gc.collect()

    if cfg.has_epilogue_quant:
        if cfg.uses_mxfp8_output_quant:
            c_logical = _dequantize_mxfp8_output(
                c_storage,
                sf_c_torch,
                output_m=logical_output_m,
                n=logical_output_n,
                is_swap_ab=cfg.is_swap_ab,
                sf_layout_c=cfg.sf_layout_c,
            )
        else:
            c_logical = _dequantize_fp4_output(
                c_storage,
                sf_c_torch,
                output_m=logical_output_m,
                n=logical_output_n,
                is_swap_ab=cfg.is_swap_ab,
                sf_layout_c=cfg.sf_layout_c,
                uses_mxfp4=cfg.uses_mxfp4_output_quant,
            )
    elif cfg.has_deepseek_fp8_c_scale:
        c_logical = _dequantize_dsfp8_output(
            c_storage,
            sf_c_torch,
            output_m=logical_output_m,
            n=logical_output_n,
            is_swap_ab=cfg.is_swap_ab,
        )
    elif cfg.uses_fp8_output:
        c_logical = _decode_fp8_output(
            c_storage,
            output_m=logical_output_m,
            n=logical_output_n,
            is_swap_ab=cfg.is_swap_ab,
        )
    else:
        c_logical = c_torch

    print(f"Output shape: {c_logical.shape}")
    print(
        f"Output sample (first 8 elements): {c_logical[0, : min(8, logical_output_n)]}"
    )
    print(f"Output max: {c_logical.abs().max().item()}")
    if output_guard_elements > 0:
        guard_changed = (c_guard != output_guard_value).any().item()
        print(f"Output guard changed: {guard_changed}")
        if guard_changed:
            if return_output:
                return False, c_logical.detach().clone()
            return False

    # Routed MoE kernels are only required to write rows/columns that map to
    # real tokens. Expert padding remains outside the observable output, so do
    # not treat unspecified padding values as correctness failures.
    active_route_mask = torch.tensor(
        [expert_idx >= 0 for expert_idx in token_layout.expanded_to_expert],
        dtype=torch.bool,
        device=device,
    )

    def _active_routes(output):
        token_dim = 1 if cfg.is_swap_ab else 0
        if output.shape[token_dim] != active_route_mask.numel():
            raise AssertionError(
                "output token extent does not match routed metadata: "
                f"{output.shape[token_dim]} != {active_route_mask.numel()}"
            )
        if cfg.is_swap_ab:
            return output[:, active_route_mask]
        return output[active_route_mask, :]

    # --- PyTorch reference ---
    c_active = _active_routes(c_logical)
    has_nan = torch.isnan(c_active).any().item()
    has_inf = torch.isinf(c_active).any().item()

    def _return(result):
        if return_output:
            return result, c_logical.detach().clone()
        return result

    if not outputs_deterministic or not scales_deterministic:
        print(
            "FAIL: repeated launches are nondeterministic "
            f"(output={outputs_deterministic}, scale={scales_deterministic})"
        )
        return _return(False)

    if has_nan or has_inf:
        print(f"FAIL: output has NaN={has_nan}, Inf={has_inf}")
        bad_mask = torch.isnan(c_active) | torch.isinf(c_active)
        bad_count = bad_mask.sum().item()
        bad_indices = bad_mask.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
        print(f"Bad value count: {bad_count}, first indices: {bad_indices}")
        return _return(False)

    def _per_token_sf_b_ref_index(expanded_idx, token_idx):
        # Routed kernels load per-token SF-B through route_map; non-routed
        # swapAB kernels consume the already-expanded B operand columns.
        return token_idx if cfg.has_routed_act else expanded_idx

    # Reference: C[M,N] = A[e, M-rows, :] @ B[e, :, :]^T per expert.
    # Uses original domain tensors for math; kernel-only row shuffles and gated
    # interleaving are undone by the logical epilogue below.
    if cfg.is_bf16_mma:
        ref_gemm = torch.zeros((M, N), dtype=plain_output_dtype, device=device)
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            if cfg.is_swap_ab:
                ref_gemm[:, expanded_idx] = (
                    weights_torch[expert_idx, :, :].float()
                    @ activation_compact_torch[token_idx, :].float()
                ).to(plain_output_dtype)
            else:
                ref_gemm[expanded_idx, :] = (
                    activation_compact_torch[token_idx, :].float()
                    @ weights_torch[expert_idx, :, :].float().T
                ).to(plain_output_dtype)
    elif cfg.has_cast_a:
        weights_ref = _dequantize_block_scaled_tensor(
            weights_torch,
            sf_weights,
            dtype_kind=cfg.dtype_a_kind,
            logical_cols=in_hidden,
            layout=weight_sf_layout,
            sf_vec_size=cfg.sf_vec_size,
            sf_torch_dtype=_sf_torch_dtype(cfg),
        )
        ref_gemm = torch.zeros((M, N), dtype=torch.float32, device=device)
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            act_row = activation_compact_torch[token_idx, :].float()
            if cfg.is_swap_ab:
                ref_gemm[:, expanded_idx] = weights_ref[expert_idx, :, :] @ act_row
            else:
                ref_gemm[expanded_idx, :] = act_row @ weights_ref[expert_idx, :, :].T
    elif cfg.is_fp8_mma:
        weight_dtype_kind = cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind
        activation_dtype_kind = cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind
        weights_ref = _decode_quantized_storage(
            weights_torch,
            dtype_kind=weight_dtype_kind,
            logical_cols=in_hidden,
        )
        activations_ref = _decode_quantized_storage(
            activation_compact_torch,
            dtype_kind=activation_dtype_kind,
            logical_cols=in_hidden,
        )
        if cfg.has_deepseek_fp8:
            ref_gemm = compute_reference_dsfp8(
                cfg=cfg,
                weights_ref=weights_ref,
                activations_ref=activations_ref,
                sf_weights=sf_weights,
                sf_activations=sf_activations,
                token_layout=token_layout,
                in_hidden=in_hidden,
                m_dim=M,
                n_dim=N,
            )
        else:
            ref_gemm = torch.zeros((M, N), dtype=torch.float32, device=device)
            for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
                if expert_idx < 0:
                    continue
                token_idx = token_layout.expanded_to_token[expanded_idx]
                act_row = activations_ref[token_idx, :]
                if cfg.is_swap_ab:
                    value = weights_ref[expert_idx, :, :] @ act_row
                    if cfg.has_per_token_sf_b:
                        sf_idx = _per_token_sf_b_ref_index(expanded_idx, token_idx)
                        value = value * per_token_sf_b_torch[sf_idx].float()
                    ref_gemm[:, expanded_idx] = value
                else:
                    value = act_row @ weights_ref[expert_idx, :, :].T
                    if cfg.has_per_token_sf_b:
                        value = value * per_token_sf_b_torch[:N].float()
                    ref_gemm[expanded_idx, :] = value
    else:
        weight_dtype_kind = cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind
        activation_dtype_kind = cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind
        weights_ref = _dequantize_block_scaled_tensor(
            weights_torch,
            sf_weights,
            dtype_kind=weight_dtype_kind,
            logical_cols=in_hidden,
            layout=weight_sf_layout,
            sf_vec_size=cfg.sf_vec_size,
            sf_torch_dtype=_sf_torch_dtype(cfg),
        )
        ref_gemm = torch.zeros((M, N), dtype=torch.float32, device=device)
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            act_row = _dequantize_block_scaled_activation_row(
                activation_compact_torch,
                sf_activations,
                dtype_kind=activation_dtype_kind,
                logical_cols=in_hidden,
                token_idx=token_idx,
                expanded_idx=expanded_idx,
                expert_idx=expert_idx,
                cfg=cfg,
                activation_sf_layout=activation_sf_layout,
                num_experts=num_experts,
                sf_k=sf_k,
            )
            if cfg.is_swap_ab:
                value = weights_ref[expert_idx, :, :] @ act_row
                if cfg.has_per_token_sf_b:
                    sf_idx = _per_token_sf_b_ref_index(expanded_idx, token_idx)
                    value = value * per_token_sf_b_torch[sf_idx].float()
                ref_gemm[:, expanded_idx] = value
            else:
                value = act_row @ weights_ref[expert_idx, :, :].T
                if cfg.has_per_token_sf_b:
                    value = value * per_token_sf_b_torch[:N].float()
                ref_gemm[expanded_idx, :] = value

    if cfg.has_per_token_sf_a:
        if cfg.is_swap_ab:
            ref_gemm = ref_gemm * per_token_sf_a_torch[:M].float().view(M, 1)
        else:
            for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
                if expert_idx < 0:
                    continue
                token_idx = token_layout.expanded_to_token[expanded_idx]
                ref_gemm[expanded_idx, :] = (
                    ref_gemm[expanded_idx, :].float()
                    * per_token_sf_a_torch[token_idx].float()
                ).to(ref_gemm.dtype)

    if cfg.has_bias_m:
        if cfg.is_swap_ab:
            for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
                if expert_idx >= 0:
                    ref_gemm[:, expanded_idx] = (
                        ref_gemm[:, expanded_idx].float()
                        + bias_logical_torch[expert_idx, :]
                    ).to(ref_gemm.dtype)
        else:
            for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
                if expert_idx >= 0:
                    ref_gemm[expanded_idx, :] = (
                        ref_gemm[expanded_idx, :].float()
                        + bias_logical_torch[expert_idx, expanded_idx]
                    ).to(ref_gemm.dtype)

    def _expert_scale_like(shape, scale_tensor):
        scale = torch.ones(shape, dtype=torch.float32, device=device)
        if not cfg.uses_global_scales:
            return scale
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            if cfg.is_swap_ab:
                scale[:, expanded_idx] = scale_tensor[expert_idx]
            else:
                scale[expanded_idx, :] = scale_tensor[expert_idx]
        return scale

    def _expert_param_like(shape, param_tensor, *, active: bool, default: float):
        value = torch.full(shape, default, dtype=torch.float32, device=device)
        if not active:
            return value
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            if cfg.is_swap_ab:
                value[:, expanded_idx] = param_tensor[expert_idx]
            else:
                value[expanded_idx, :] = param_tensor[expert_idx]
        return value

    is_geglu = cfg.act_kind == int(ActKind.GEGLU)
    is_situ = cfg.act_kind == int(ActKind.SITU)
    is_gated = _is_gated_act_kind(cfg.act_kind)
    is_relu2 = cfg.act_kind == int(ActKind.RELU2)

    if is_gated:
        # Gated activation: weights are logically [gate-half, up-half].
        def _quick_gelu(x, scale_gate):
            return x * torch.sigmoid(1.702 * x * scale_gate)

        if cfg.is_swap_ab:
            half = M // 2
            gate = ref_gemm[:half, :].float()
            up = ref_gemm[half:, :].float()
        else:
            half = N // 2
            gate = ref_gemm[:, :half].float()
            up = ref_gemm[:, half:].float()

        scale_gate = _expert_scale_like(gate.shape, scale_gate_torch)
        if is_situ:
            if cfg.has_gemm1_clamp_limit:
                clamp_limit = _expert_param_like(
                    gate.shape,
                    gemm1_clamp_limit_torch,
                    active=True,
                    default=0.0,
                )
                gate = torch.minimum(torch.maximum(gate, -clamp_limit), clamp_limit)
                up = torch.minimum(up, clamp_limit)
            x0 = gate * scale_gate
            x1 = up * scale_gate
            alpha = _expert_param_like(
                gate.shape,
                gemm1_alpha_torch,
                active=bool(cfg.has_gemm1_alpha),
                default=1.0,
            )
            beta = _expert_param_like(
                gate.shape,
                gemm1_beta_torch,
                active=bool(cfg.has_gemm1_beta),
                default=1.0,
            )
            activated = (
                beta
                * torch.tanh(x0 / beta)
                * alpha
                * torch.tanh(x1 / alpha)
                * torch.sigmoid(x1)
            )
        elif cfg.has_swiglu_oai_params:
            if cfg.has_gemm1_clamp_limit:
                clamp_limit = _expert_param_like(
                    gate.shape,
                    gemm1_clamp_limit_torch,
                    active=True,
                    default=0.0,
                )
                gate = torch.minimum(torch.maximum(gate, -clamp_limit), clamp_limit)
                up = torch.minimum(up, clamp_limit)
            up_scaled = up * scale_gate
            alpha = _expert_param_like(
                gate.shape,
                gemm1_alpha_torch,
                active=bool(cfg.has_gemm1_alpha),
                default=1.0,
            )
            beta = _expert_param_like(
                gate.shape,
                gemm1_beta_torch,
                active=bool(cfg.has_gemm1_beta),
                default=0.0,
            )
            activated = (gate + beta) * up_scaled * torch.sigmoid(alpha * up_scaled)
        elif is_geglu:
            activated = (gate * scale_gate) * _quick_gelu(up, scale_gate)
        else:
            activated = (gate * scale_gate) * up * torch.sigmoid(up * scale_gate)
        ref_torch = activated * _expert_scale_like(activated.shape, scale_c_torch)
        if cfg.uses_fp8_output:
            ref_torch = ref_torch.to(torch.float8_e4m3fn).float()
        elif not cfg.has_epilogue_quant:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical
    elif is_relu2:
        ref_torch = torch.clamp(ref_gemm.float(), min=0).pow(2)
        ref_torch = ref_torch * _expert_scale_like(ref_torch.shape, scale_c_torch)
        if cfg.uses_fp8_output:
            ref_torch = ref_torch.to(torch.float8_e4m3fn).float()
        elif not cfg.has_epilogue_quant:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical
    else:
        ref_torch = ref_gemm.float() * _expert_scale_like(ref_gemm.shape, scale_c_torch)
        if cfg.uses_fp8_output:
            ref_torch = ref_torch.to(torch.float8_e4m3fn).float()
        elif not cfg.has_epilogue_quant:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical

    if cfg.has_deepseek_fp8_c_scale:
        expected_sf_c = _reference_dsfp8_c_scales(
            ref_torch.float(), is_swap_ab=cfg.is_swap_ab
        )
        sf_c_diff = (sf_c_torch.float() - expected_sf_c).abs()
        sf_c_max_abs = sf_c_diff.max().item()
        sf_c_ref_max = expected_sf_c.abs().max().item()
        sf_c_threshold = 1.0e-2 + 1.0e-2 * sf_c_ref_max
        print(
            "DeepSeek C scale comparison: "
            f"max_abs={sf_c_max_abs:.6f}, ref_max={sf_c_ref_max:.6f}"
        )
        if sf_c_max_abs > sf_c_threshold:
            print(
                "Reference check: FAILED "
                f"(DeepSeek C scale max_abs={sf_c_max_abs:.6f} > {sf_c_threshold:.6f})"
            )
            return _return(False)

    c_compare_active = _active_routes(c_compare)
    ref_active = _active_routes(ref_torch)
    diff = (c_compare_active.float() - ref_active.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_max = ref_active.float().abs().max().item()
    out_max = c_compare_active.float().abs().max().item()
    print(
        f"Reference comparison: max_abs={max_abs:.6f}, "
        f"mean_abs={mean_abs:.6f}, ref_max={ref_max:.6f}"
    )
    if cfg.has_cast_a and cfg.is_swap_ab:
        # Operates on the actual output rows: gated activations halve the logical
        # output (M//2), so index by logical_output_m, not the full M.
        shuffled_ref = shuffle_matrix(ref_torch, cfg.tile_m)
        unshuffle_idx = torch.empty(
            (logical_output_m,), dtype=torch.long, device=device
        )
        block_size = 32 if cfg.tile_m % 128 == 0 else 16
        indices = _SHUFFLE_BLOCK_32 if block_size == 32 else _SHUFFLE_BLOCK_16
        for mi in range(logical_output_m):
            block_idx = mi // block_size
            src_in_block = mi % block_size
            dst_in_block = indices[src_in_block]
            unshuffle_idx[block_idx * block_size + dst_in_block] = mi
        inv_ref = ref_torch[unshuffle_idx, :]
        shuffled_diff = (
            c_compare_active.float() - _active_routes(shuffled_ref).float()
        ).abs()
        inv_diff = (c_compare_active.float() - _active_routes(inv_ref).float()).abs()
        print(
            "CastA row-diagnostic: "
            f"shuffled_max={shuffled_diff.max().item():.6f}, "
            f"inverse_max={inv_diff.max().item():.6f}"
        )

    if cfg.is_bf16_mma:
        threshold = 1e-1
    elif cfg.has_epilogue_quant:
        threshold = 0.75 + 0.35 * ref_max
    else:
        threshold = 0.75 + 0.05 * ref_max

    if out_max == 0.0 and ref_max > 1e-5:
        print(
            "Reference check: FAILED "
            f"(all-zero output with non-zero reference max {ref_max:.6f})"
        )
        return _return(False)

    if max_abs <= threshold:
        print(f"Reference check: PASSED (max_abs={max_abs:.6f} <= {threshold:.6f})")
        return _return(True)
    max_flat = diff.argmax()
    max_coord = torch.unravel_index(max_flat, diff.shape)
    max_coord_host = tuple(int(v.detach().cpu().item()) for v in max_coord)
    print(f"Reference max-diff index: {max_coord_host}")
    print(f"Reference check: FAILED (max_abs={max_abs:.6f} > {threshold:.6f})")
    return _return(False)


def _build_launch_io(
    *,
    num_experts,
    num_tokens,
    top_k,
    problem_n=None,
    problem_k=None,
    seed=42,
    scale_c_value=1.0,
    scale_gate_value=1.0,
    gemm1_alpha_value=None,
    gemm1_beta_value=None,
    gemm1_clamp_limit_value=2.0,
    early_exit_max_token_ctas=0,
    **cfg_overrides,
):
    """Build all device tensors + pointers for a single kernel launch.

    Shared by :func:`benchmark` and the perf-harness interface
    (:func:`compile` / :func:`prepare_tensors` / :func:`run`). Returns a dict
    of device pointers, the ``(M, N, K, L, num_tokens)`` problem shape, the
    resolved config, and a ``_keepalive`` list that pins the backing torch
    tensors alive for as long as the pointers are dereferenced.
    """
    launch_early_exit_max_token_ctas = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    cfg = make_config(**cfg_overrides)

    torch.manual_seed(seed)
    device = "cuda"
    from cutlass.cute.runtime import make_ptr

    # Keep shape semantics aligned with reference_check and generated batch-N
    # kernels: problem_n names the output hidden dimension for both swapAB and
    # non-swap paths.
    default_out_hidden = cfg.tile_m if cfg.is_swap_ab else cfg.tile_n
    out_hidden = problem_n if problem_n is not None else default_out_hidden
    in_hidden = problem_k if problem_k is not None else cfg.tile_k
    cfg = _runtime_config(cfg, in_hidden)
    if cfg.is_swap_ab and out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    token_tile_size = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
    cluster_dim_in_token = 1 if cfg.is_swap_ab else cfg.cluster_m
    token_layout = _make_token_layout(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
    )
    launch_early_exit_max_token_ctas = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )
    total_padded_tokens = token_layout.total_padded_tokens

    # Create domain tensors. Routed kernels consume compact activation rows;
    # non-routed kernels consume the expanded/padded expert-token layout.
    if cfg.is_bf16_mma:
        weights_torch = _kaiming_uniform_tensor(
            (num_experts, out_hidden, in_hidden),
            fan_in=in_hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        activation_compact_torch = _kaiming_uniform_tensor(
            (num_tokens, in_hidden),
            fan_in=in_hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_bf16_activations(activation_compact_torch, token_layout)
        )
        a_dtype = b_dtype = cutlass.BFloat16
        weight_sf_layout = None
        activation_sf_layout = None
        sf_weights = sf_activations = None
    elif cfg.has_cast_a:
        weights_torch = _random_quantized_tensor(
            (num_experts, out_hidden, in_hidden),
            dtype_kind=cfg.dtype_a_kind,
            fan_in=in_hidden,
            device=device,
        )
        activation_compact_torch = _kaiming_uniform_tensor(
            (num_tokens, in_hidden),
            fan_in=in_hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_bf16_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = cutlass.BFloat16
        sf_k = in_hidden // cfg.sf_vec_size
        weight_sf_layout = cfg.sf_layout_a
        sf_weights = _create_sf_tensor_mma_layout(
            num_experts=num_experts,
            mn=out_hidden,
            sf_k=sf_k,
            layout=weight_sf_layout,
            device=device,
            cfg=cfg,
        )
        sf_activations = None
    elif cfg.is_fp8_mma:
        weights_torch = _random_quantized_tensor(
            (num_experts, out_hidden, in_hidden),
            dtype_kind=cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind,
            fan_in=in_hidden,
            device=device,
        )
        activation_compact_torch = _random_quantized_tensor(
            (num_tokens, in_hidden),
            dtype_kind=cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b_kind)
        weight_sf_layout = None
        activation_sf_layout = None
        sf_weights = sf_activations = None
        if cfg.has_deepseek_fp8:
            sf_weights, sf_activations = _create_deepseek_fp8_sf_tensors(
                cfg,
                num_experts=num_experts,
                num_tokens=num_tokens,
                out_hidden=out_hidden,
                in_hidden=in_hidden,
                total_padded_tokens=total_padded_tokens,
                device=device,
            )
    else:
        weights_torch = _random_quantized_tensor(
            (num_experts, out_hidden, in_hidden),
            dtype_kind=cfg.dtype_a_kind if cfg.is_swap_ab else cfg.dtype_b_kind,
            fan_in=in_hidden,
            device=device,
        )
        activation_compact_torch = _random_quantized_tensor(
            (num_tokens, in_hidden),
            dtype_kind=cfg.dtype_b_kind if cfg.is_swap_ab else cfg.dtype_a_kind,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a_kind)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b_kind)
        sf_k = in_hidden // cfg.sf_vec_size
        activation_sf_layout = cfg.sf_layout_b if cfg.is_swap_ab else cfg.sf_layout_a
        weight_sf_layout = cfg.sf_layout_a if cfg.is_swap_ab else cfg.sf_layout_b
        if cfg.has_routed_sfs:
            sf_activations = _create_sf_linear(
                num_tokens=num_tokens,
                sf_k_dim=sf_k,
                device=device,
                cfg=cfg,
            )
            sf_weights = _create_sf_tensor_mma_layout(
                num_experts=num_experts,
                mn=out_hidden,
                sf_k=sf_k,
                layout=weight_sf_layout,
                device=device,
                cfg=cfg,
            )
        else:
            sf_weights = _create_sf_tensor_mma_layout(
                num_experts=num_experts,
                mn=out_hidden,
                sf_k=sf_k,
                layout=weight_sf_layout,
                device=device,
                cfg=cfg,
            )
            sf_activations = _create_sf_tensor_mma_layout(
                num_experts=1,
                mn=total_padded_tokens,
                sf_k=sf_k,
                layout=activation_sf_layout,
                device=device,
                cfg=cfg,
            )

    is_gated = _is_gated_act_kind(cfg.act_kind)
    preprocessed_weights = weights_torch
    if is_gated:
        preprocessed_weights = reorder_rows_for_gated_act(weights_torch)

    if cfg.is_swap_ab:
        M, N, K, L = out_hidden, total_padded_tokens, in_hidden, num_experts
        kernel_a = _block_major_k_weight_tensor(
            shuffle_matrix(preprocessed_weights, cfg.tile_m),
            cfg,
        )
        kernel_b = activations_torch
        sf_a = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_for_swap_ab=(not cfg.has_deepseek_fp8),
        )
        sf_b = sf_activations
    else:
        M, N, K, L = total_padded_tokens, out_hidden, in_hidden, num_experts
        kernel_a = activations_torch
        kernel_b = _block_major_k_weight_tensor(preprocessed_weights, cfg)
        sf_a = sf_activations
        sf_b = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_for_swap_ab=False,
        )

    logical_output_m, logical_output_n = _logical_output_shape(cfg, M, N, is_gated)
    plain_output_dtype = _plain_output_torch_dtype(cfg)
    if cfg.has_epilogue_quant:
        c_storage = torch.zeros(
            (_quantized_c_numel(cfg, logical_output_m, logical_output_n),),
            dtype=torch.uint8,
            device=device,
        )
        c_torch = None
        sf_c_torch = torch.zeros(
            (
                _sf_c_numel(
                    logical_output_m,
                    logical_output_n,
                    cfg.sf_layout_c,
                    cfg.output_sf_block_size_c,
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
    elif cfg.uses_fp8_output:
        c_storage = torch.zeros(
            (logical_output_m * logical_output_n,),
            dtype=torch.uint8,
            device=device,
        )
        c_torch = None
        if cfg.has_deepseek_fp8_c_scale:
            sf_c_torch = torch.zeros(
                (
                    _dsfp8_c_numel(
                        logical_output_m,
                        logical_output_n,
                        is_swap_ab=cfg.is_swap_ab,
                    ),
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    elif cfg.is_swap_ab:
        c_storage = torch.zeros(
            (logical_output_m * logical_output_n,),
            dtype=plain_output_dtype,
            device=device,
        )
        c_torch = torch.as_strided(
            c_storage,
            (logical_output_m, logical_output_n),
            (1, logical_output_m),
        )
        sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    else:
        c_torch = torch.zeros(
            (logical_output_m, logical_output_n),
            dtype=plain_output_dtype,
            device=device,
        )
        sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    tile_idx_list, mn_limit_list, route_map_list = _launch_metadata_lists(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )
    tile_idx = torch.tensor(tile_idx_list, dtype=torch.int32, device=device)
    mn_limit = torch.tensor(mn_limit_list, dtype=torch.int32, device=device)
    if cfg.has_routed_act or cfg.has_routed_sfs:
        route_map = torch.tensor(route_map_list, dtype=torch.int32, device=device)
    else:
        route_map = torch.zeros(1, dtype=torch.int32, device=device)
    act_torch = (
        activation_compact_torch
        if cfg.has_routed_act
        else torch.zeros(1, dtype=torch.bfloat16, device=device)
    )

    a_dp = make_ptr(
        a_dtype, kernel_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        b_dtype, kernel_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    sf_ab_dtype = _sf_ab_cutlass_dtype(cfg)
    if sf_a is not None:
        sfa_dp = make_ptr(
            sf_ab_dtype, sf_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
        )
    else:
        sfa_dp = a_dp
    if sf_b is not None:
        sfb_dp = make_ptr(
            sf_ab_dtype, sf_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
        )
    else:
        sfb_dp = b_dp
    if cfg.has_epilogue_quant or cfg.uses_fp8_output:
        c_dtype = (
            cutlass.Float8E4M3FN
            if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
            else cutlass.Float4E2M1FN
        )
        c0_dp = make_ptr(
            c_dtype,
            c_storage.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c0_dp = make_ptr(
            _plain_output_cutlass_dtype(cfg),
            c_torch.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    sf_c_dp = make_ptr(
        _sf_c_cutlass_dtype(cfg),
        sf_c_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    bias_torch = torch.zeros((L, M), dtype=torch.float32, device=device)
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_torch = torch.full((L,), scale_c_value, dtype=torch.float32, device=device)
    scale_gate_torch = torch.full(
        (L,), scale_gate_value, dtype=torch.float32, device=device
    )
    gemm1_alpha_torch = _make_per_expert_f32_tensor(
        gemm1_alpha_value,
        default=1.0,
        num_experts=L,
        device=device,
    )
    gemm1_beta_torch = _make_per_expert_f32_tensor(
        gemm1_beta_value,
        default=1.0 if cfg.act_kind == int(ActKind.SITU) else 0.0,
        num_experts=L,
        device=device,
    )
    gemm1_clamp_limit_torch = _make_per_expert_f32_tensor(
        gemm1_clamp_limit_value if cfg.has_gemm1_clamp_limit else None,
        default=0.0,
        num_experts=L,
        device=device,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_alpha_dp = make_ptr(
        cutlass.Float32,
        gemm1_alpha_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_beta_dp = make_ptr(
        cutlass.Float32,
        gemm1_beta_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gemm1_clamp_limit_dp = make_ptr(
        cutlass.Float32,
        gemm1_clamp_limit_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_dtype = _per_token_torch_dtype(cfg)
    per_token_count = max(num_tokens, total_padded_tokens, M, N)
    per_token_sf_a_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    per_token_sf_b_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    per_token_cutlass_dtype = _per_token_cutlass_dtype(cfg)
    per_token_sf_a_dp = make_ptr(
        per_token_cutlass_dtype,
        per_token_sf_a_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        per_token_cutlass_dtype,
        per_token_sf_b_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    route_map_dp = make_ptr(
        cutlass.Int32,
        route_map.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas = torch.tensor(
        [token_layout.num_token_tiles], dtype=torch.int32, device=device
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    total_num_padded_tokens_torch = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=device
    )
    total_num_padded_tokens_dp = make_ptr(
        cutlass.Int32,
        total_num_padded_tokens_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        cutlass.BFloat16,
        act_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    # Pin the torch tensors backing the returned device pointers; otherwise they
    # are freed when this frame returns, leaving the pointers dangling. The C
    # buffer is the packed quant/fp8 storage or the bf16 output tensor, selected
    # by the same condition used to build c0_dp above.
    c_backing = (
        c_storage if (cfg.has_epilogue_quant or cfg.uses_fp8_output) else c_torch
    )
    _keepalive = [
        tensor
        for tensor in (
            kernel_a,
            kernel_b,
            sf_a,
            sf_b,
            c_backing,
            sf_c_torch,
            tile_idx,
            mn_limit,
            route_map,
            act_torch,
            bias_torch,
            scale_c_torch,
            scale_gate_torch,
            gemm1_alpha_torch,
            gemm1_beta_torch,
            gemm1_clamp_limit_torch,
            per_token_sf_a_torch,
            per_token_sf_b_torch,
            num_non_exiting_ctas,
            total_num_padded_tokens_torch,
        )
        if tensor is not None
    ]
    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": M,
        "N": N,
        "K": K,
        "L": L,
        "num_tokens": num_tokens,
        "total_padded_tokens": total_padded_tokens,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "logical_output_n": logical_output_n,
        "plain_output_dtype": plain_output_dtype,
        "shape": (M, N, K, L, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "total_num_padded_tokens_dp": total_num_padded_tokens_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "sf_c_torch": sf_c_torch,
        "_keepalive": _keepalive,
    }


# ---------------------------------------------------------------------------
# Perf-harness interface (``@pytest.mark.perf`` profiler sweep).
#
# The profiler imports the test module and calls ``compile`` / ``prepare_tensors``
# / ``run`` (and ``FLOPS_FORMULA`` for the derived TFLOPS table). These thin
# wrappers reuse :func:`_build_launch_io` so the perf path exercises the exact
# tensor + pointer setup as :func:`benchmark`.
# ---------------------------------------------------------------------------


@dataclass
class _BatchedGemmExecutable:
    """A compiled BatchedGemm kernel plus the shape it was compiled for."""

    compiled_fn: object
    shape: tuple
    torch_stream: object = None


def _compile_arg_tuple(io: dict, stream) -> tuple:
    """Positional args for ``cute.compile(gemm, ...)``.

    ``cfg`` is a compile-time constexpr operand and is not repeated at launch.
    ``launch_early_exit_max_token_ctas`` is a runtime launch argument because
    it only controls the over-launched grid size for dynamic-batch reuse.
    """
    return (
        io["a_dp"],
        io["b_dp"],
        io["sfa_dp"],
        io["sfb_dp"],
        io["c0_dp"],
        io["sf_c_dp"],
        io["tile_idx_dp"],
        io["route_map_dp"],
        io["mn_limit_dp"],
        io["num_non_exiting_ctas_dp"],
        io.get("total_num_padded_tokens_dp", io["num_non_exiting_ctas_dp"]),
        io["act_dp"],
        io["per_token_sf_a_dp"],
        io["per_token_sf_b_dp"],
        io["bias_dp"],
        io["scale_c_dp"],
        io["scale_gate_dp"],
        io["gemm1_alpha_dp"],
        io["gemm1_beta_dp"],
        io["gemm1_clamp_limit_dp"],
        io["shape"],
        io["launch_early_exit_max_token_ctas"],
        io["cfg"],
        stream,
    )


def _launch_arg_tuple(io: dict, stream) -> tuple:
    """Positional args for a runtime launch of the compiled kernel.

    Mirrors the ``JitArguments`` order in :func:`benchmark`: the constexpr
    ``cfg`` operand is omitted at launch time.
    """
    return (
        io["a_dp"],
        io["b_dp"],
        io["sfa_dp"],
        io["sfb_dp"],
        io["c0_dp"],
        io["sf_c_dp"],
        io["tile_idx_dp"],
        io["route_map_dp"],
        io["mn_limit_dp"],
        io["num_non_exiting_ctas_dp"],
        io.get("total_num_padded_tokens_dp", io["num_non_exiting_ctas_dp"]),
        io["act_dp"],
        io["per_token_sf_a_dp"],
        io["per_token_sf_b_dp"],
        io["bias_dp"],
        io["scale_c_dp"],
        io["scale_gate_dp"],
        io["gemm1_alpha_dp"],
        io["gemm1_beta_dp"],
        io["gemm1_clamp_limit_dp"],
        io["shape"],
        io["launch_early_exit_max_token_ctas"],
        stream,
    )


def _compile_for_launch(io: dict, stream) -> object:
    """Compile ``gemm`` for one launch IO on ``stream``; returns the compiled fn.

    The single ``cute.compile`` entry point, shared by :func:`compile`,
    :func:`benchmark`, and :func:`reference_check`.
    """
    from cutlass.cutlass_dsl import BaseDSL

    compile_entry = cute.compile
    if _generate_line_info_enabled():
        compile_entry = cute.compile[cute.GenerateLineInfo(True)]

    with task_scheduling_scope(), BaseDSL.enable_pyir():
        compiled = compile_entry(
            gemm,
            *_compile_arg_tuple(io, stream),
            # WAR for CUDA 13.1 compiler redundant ELECT/R2UR around TMA ops.
            # Matches the FMHA TS runner workaround and can be removed after
            # upgrading to a CUDA toolkit with the OCG/NVVM fixes.
            options=_cute_compile_options(),
        )
    return compiled


def compile(**cfg) -> _BatchedGemmExecutable:  # noqa: A001
    """Compile the BatchedGemm kernel for one perf config.

    ``cfg`` carries ``num_experts`` / ``num_tokens`` / ``top_k`` /
    ``problem_n`` / ``problem_k`` plus the ``make_config`` overrides (same keys
    the correctness tests pass to :func:`reference_check`).
    """
    import cuda.bindings.driver as cuda_drv

    io = _build_launch_io(**cfg)
    # A non-default stream keeps CUDA-graph capture in the profiler valid.
    torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(torch_stream.cuda_stream)
    compiled_fn = _compile_for_launch(io, stream)
    return _BatchedGemmExecutable(
        compiled_fn=compiled_fn, shape=io["shape"], torch_stream=torch_stream
    )


def prepare_tensors(**cfg) -> dict:
    """Allocate the device tensors + pointers for one perf config."""
    return _build_launch_io(**cfg)


def run(executable: _BatchedGemmExecutable, tensors: dict, stream) -> None:
    """Launch the compiled kernel on ``stream`` with ``tensors``."""
    if tensors["shape"] != executable.shape:
        raise ValueError(
            f"prepared tensors shape {tensors['shape']} does not match the "
            f"compiled shape {executable.shape}"
        )
    executable.compiled_fn(*_launch_arg_tuple(tensors, stream))


def verify_output(tensors: dict, **cfg) -> None:
    """Check correctness for one perf config via the trusted reference path.

    Delegates to :func:`reference_check` (its own compile + launch + PyTorch
    comparison), which is the authoritative numerics path for this kernel.
    """
    del tensors
    if not reference_check(**cfg):
        raise AssertionError(f"batched_gemm reference check failed for cfg={cfg}")


def FLOPS_FORMULA(
    *,
    num_tokens,
    top_k,
    problem_n=None,
    problem_k=None,
    tile_n=None,
    tile_k=None,
    **_,
) -> float:
    """MoE batched-GEMM FLOPs: each of ``top_k`` routed token copies does an
    ``N x K`` matmul (2 FLOPs per MAC). ``N`` (out_hidden) already spans the
    gate+up halves for gated activations, so this counts the full MMA work.
    """
    n = problem_n if problem_n is not None else tile_n
    k = problem_k if problem_k is not None else tile_k
    if n is None or k is None:
        return float("nan")
    return 2.0 * num_tokens * top_k * n * k


def benchmark(
    *,
    num_experts,
    num_tokens,
    top_k,
    problem_n=None,
    problem_k=None,
    seed=42,
    scale_c_value=1.0,
    scale_gate_value=1.0,
    gemm1_alpha_value=None,
    gemm1_beta_value=None,
    gemm1_clamp_limit_value=2.0,
    warmup_iters=10,
    bench_iters=50,
    num_rotated_buffers=None,
    cuda_profiler_range=False,
    early_exit_max_token_ctas=0,
    use_cuda_graphs=True,
    **cfg_overrides,
):
    """Benchmark the kernel using CUDA graph capture + replay."""
    io = _build_launch_io(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=seed,
        scale_c_value=scale_c_value,
        scale_gate_value=scale_gate_value,
        gemm1_alpha_value=gemm1_alpha_value,
        gemm1_beta_value=gemm1_beta_value,
        gemm1_clamp_limit_value=gemm1_clamp_limit_value,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
        **cfg_overrides,
    )
    cfg = io["cfg"]
    M, N, K, L = io["M"], io["N"], io["K"], io["L"]
    num_tokens = io["num_tokens"]
    total_padded_tokens = io["total_padded_tokens"]
    out_hidden, in_hidden = io["out_hidden"], io["in_hidden"]
    logical_output_m = io["logical_output_m"]
    logical_output_n = io["logical_output_n"]
    plain_output_dtype = io["plain_output_dtype"]
    a_dp = io["a_dp"]
    b_dp = io["b_dp"]
    sfa_dp = io["sfa_dp"]
    sfb_dp = io["sfb_dp"]
    sf_c_dp = io["sf_c_dp"]
    tile_idx_dp = io["tile_idx_dp"]
    route_map_dp = io["route_map_dp"]
    mn_limit_dp = io["mn_limit_dp"]
    num_non_exiting_ctas_dp = io["num_non_exiting_ctas_dp"]
    total_num_padded_tokens_dp = io.get(
        "total_num_padded_tokens_dp", num_non_exiting_ctas_dp
    )
    act_dp = io["act_dp"]
    per_token_sf_a_dp = io["per_token_sf_a_dp"]
    per_token_sf_b_dp = io["per_token_sf_b_dp"]
    bias_dp = io["bias_dp"]
    scale_c_dp = io["scale_c_dp"]
    scale_gate_dp = io["scale_gate_dp"]
    gemm1_alpha_dp = io["gemm1_alpha_dp"]
    gemm1_beta_dp = io["gemm1_beta_dp"]
    gemm1_clamp_limit_dp = io["gemm1_clamp_limit_dp"]
    sf_c_torch = io["sf_c_torch"]
    # Pin the domain tensors alive for the whole benchmark (pointers alias them).
    _io_keepalive = io["_keepalive"]
    # Use a non-default stream for CUDA graph support (same as FMHA benchmark)
    import cuda.bindings.driver as cuda_drv
    import cutlass.cute.testing as testing
    from cutlass.cute.runtime import make_ptr

    _torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(_torch_stream.cuda_stream)

    # Compile with the non-default stream
    print(
        f"Benchmark: tokens={num_tokens}, padded_tokens={total_padded_tokens}, "
        f"experts={num_experts}, top_k={top_k}, M={M}, N={N}, K={K}, L={L}"
    )
    print(
        f"  tile=({cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}), "
        f"scheduler={'persistent' if cfg.is_persistent else 'static'}, "
        f"swap_ab={cfg.is_swap_ab}"
    )
    compiled_fn = _compile_for_launch(io, stream)

    # Build workspace generator for cold-L2 / rotated buffers
    def generate_tensors():
        if cfg.has_epilogue_quant or cfg.uses_fp8_output:
            c_buf = torch.zeros(
                (
                    _quantized_c_numel(cfg, logical_output_m, logical_output_n)
                    if cfg.has_epilogue_quant
                    else logical_output_m * logical_output_n
                ),
                dtype=torch.uint8,
                device="cuda",
            )
            c_dtype = (
                cutlass.Float8E4M3FN
                if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
                else cutlass.Float4E2M1FN
            )
            c_ptr = make_ptr(
                c_dtype,
                c_buf.data_ptr(),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
            sf_c_buf = torch.zeros_like(sf_c_torch)
            sf_c_ptr = make_ptr(
                _sf_c_cutlass_dtype(cfg),
                sf_c_buf.data_ptr(),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
        else:
            c_buf = torch.zeros(
                logical_output_m,
                logical_output_n,
                dtype=plain_output_dtype,
                device="cuda",
            )
            c_ptr = make_ptr(
                _plain_output_cutlass_dtype(cfg),
                c_buf.data_ptr(),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
            sf_c_buf = sf_c_torch
            sf_c_ptr = sf_c_dp
        args = testing.JitArguments(
            a_dp,
            b_dp,
            sfa_dp,
            sfb_dp,
            c_ptr,
            sf_c_ptr,
            tile_idx_dp,
            route_map_dp,
            mn_limit_dp,
            num_non_exiting_ctas_dp,
            total_num_padded_tokens_dp,
            act_dp,
            per_token_sf_a_dp,
            per_token_sf_b_dp,
            bias_dp,
            scale_c_dp,
            scale_gate_dp,
            gemm1_alpha_dp,
            gemm1_beta_dp,
            gemm1_clamp_limit_dp,
            (M, N, K, L, num_tokens),
            io["launch_early_exit_max_token_ctas"],
            stream,
        )
        args.add_to_scope([c_buf, sf_c_buf])
        return args

    # Compute workspace count for cold L2
    if cfg.is_bf16_mma or cfg.has_cast_a:
        ab_elem_bytes = 2
    elif cfg.is_fp8_mma:
        ab_elem_bytes = 1
    else:
        ab_elem_bytes = 0.5
    one_workspace_bytes = (
        int(num_experts * out_hidden * in_hidden * ab_elem_bytes)
        + int(num_tokens * in_hidden * ab_elem_bytes)
        + (
            logical_output_m * logical_output_n
            if (cfg.has_epilogue_quant or cfg.uses_fp8_output)
            else logical_output_m * logical_output_n * 2
        )
        + (
            _sf_c_numel(
                logical_output_m,
                logical_output_n,
                cfg.sf_layout_c,
                cfg.output_sf_block_size_c,
            )
            if cfg.has_epilogue_quant
            else 0
        )
    )
    workspace_count = testing.get_workspace_count(
        one_workspace_bytes,
        warmup_iters,
        bench_iters,
    )
    if num_rotated_buffers is not None:
        # Rotated buffers are a boolean toggle.  Treat 0 as
        # "disabled" and reuse one workspace so benchmark L2-cache behavior can
        # be matched with `-rotateBuffers false`.
        workspace_count = (
            1 if num_rotated_buffers <= 0 else min(workspace_count, num_rotated_buffers)
        )

    if cuda_profiler_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    try:
        exec_time = testing.benchmark(
            compiled_fn,
            workspace_generator=generate_tensors,
            workspace_count=workspace_count,
            stream=stream,
            warmup_iterations=warmup_iters,
            iterations=bench_iters,
            use_cuda_graphs=use_cuda_graphs,
        )
    finally:
        if cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
    # Match the reference-check path: clustered persistent kernels are sensitive
    # to compiled CUDA library lifetime if unload overlaps the next compile or
    # launch in the same process.
    del compiled_fn
    gc.collect()

    print(f"  Average kernel time: {exec_time:.1f} µs ({exec_time / 1000:.4f} ms)")
    return exec_time


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the BatchedGemm TS command-line parser."""
    parser = argparse.ArgumentParser(description="BatchedGemm TS runner")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the schedule only (no GPU).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the timed benchmark (after the correctness check unless --skip-ref-check).",
    )
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip the correctness check (correctness runs by default).",
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Disable CUDA graphs in the benchmark.",
    )
    parser.add_argument(
        "--no-cold-l2",
        action="store_true",
        help="Disable cold-L2 workspace rotation (reuse a single workspace).",
    )

    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)

    parser.add_argument("--batch-mode", choices=["n", "m"], default="n")
    parser.add_argument(
        "--route-act",
        choices=["false", "none", "tma", "ldgsts", "ldgplussts"],
        default="none",
    )
    parser.add_argument(
        "--route-sfs-act",
        choices=["false", "none", "tma", "ldgsts", "ldgplussts"],
        default="none",
    )
    parser.add_argument(
        "--tile-scheduler",
        choices=["static", "persistent"],
        default="static",
    )
    parser.add_argument(
        "--transpose-mma-output",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--act-kind",
        choices=["none", "swiglu", "geglu", "relu2", "silu", "situ"],
        default="none",
    )
    parser.add_argument(
        "--sf-layout-a",
        choices=["8x4", "r8c4", "128x4", "r128c4", "linear"],
        default="128x4",
    )
    parser.add_argument(
        "--sf-layout-b",
        choices=["8x4", "r8c4", "128x4", "r128c4", "linear"],
        default="8x4",
    )
    parser.add_argument(
        "--sf-layout-c",
        choices=["8x4", "r8c4", "128x4", "r128c4", "linear"],
        default="8x4",
    )

    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=8)
    parser.add_argument("--tile-k", type=int, default=512)
    parser.add_argument("--epi-tile-n", type=int, default=8)
    parser.add_argument("--mma-m", type=int, default=128)
    parser.add_argument("--mma-n", type=int, default=8)
    parser.add_argument("--mma-k", type=int, default=64)
    parser.add_argument(
        "--dtype-a",
        choices=sorted(DTYPE_NAME_TO_VALUE.keys()),
        default="e2m1",
    )
    parser.add_argument(
        "--dtype-b",
        choices=sorted(DTYPE_NAME_TO_VALUE.keys()),
        default="e2m1",
    )
    parser.add_argument(
        "--dtype-c",
        choices=sorted(DTYPE_NAME_TO_VALUE.keys()),
        default="bf16",
    )
    parser.add_argument("--sf-block-size-a", type=int, default=0)
    parser.add_argument("--sf-block-size-b", type=int, default=0)
    parser.add_argument("--sf-block-size-c", type=int, default=0)
    parser.add_argument("--num-stages-a", type=int, default=5)
    parser.add_argument("--num-stages-b", type=int, default=5)
    parser.add_argument("--num-stages-smem-sfa", type=int, default=5)
    parser.add_argument("--num-stages-smem-sfb", type=int, default=5)
    parser.add_argument("--num-stages-tmem-acc", type=int, default=1)
    parser.add_argument("--num-stages-tmem-sfa", type=int, default=5)
    parser.add_argument("--num-stages-tmem-sfb", type=int, default=5)
    parser.add_argument("--num-stages-c-smem", type=int, default=2)
    parser.add_argument("--use-tma-oob-opt", type=int, default=1)
    parser.add_argument("--use-tma-store", type=int, default=0)
    parser.add_argument("--num-load-b-warps", type=int, default=1)
    parser.add_argument("--num-load-sfb-warps", type=int, default=1)
    parser.add_argument("--epilogue-regs", type=int, default=160)
    parser.add_argument("--mma-regs", type=int, default=48)
    parser.add_argument("--load-regs", type=int, default=48)
    parser.add_argument("--load-sf-regs", type=int, default=48)
    parser.add_argument("--load-a-regs", type=int, default=0)
    parser.add_argument("--load-b-regs", type=int, default=0)
    parser.add_argument("--load-sfa-regs", type=int, default=0)
    parser.add_argument("--load-sfb-regs", type=int, default=0)
    parser.add_argument("--copy-sf-regs", type=int, default=48)
    parser.add_argument("--workid-regs", type=int, default=48)
    parser.add_argument("--padding-regs", type=int, default=48)

    parser.add_argument(
        "--use-unroll-loop-2x-for-mma",
        type=int,
        default=0,
    )
    parser.add_argument("--use-max-tmem-overlap", type=int, default=0)
    parser.add_argument("--use-early-exit", type=int, default=0)
    parser.add_argument("--use-clc-fast-drain", type=int, default=0)
    parser.add_argument("--use-work-throttle", type=int, default=1)
    parser.add_argument("--use-pdl", type=int, default=0)
    parser.add_argument(
        "--do-pdl-wait-for-num-non-exiting-ctas",
        type=int,
        default=0,
    )
    parser.add_argument("--use-global-scales", type=int, default=1)
    parser.add_argument("--use-per-token-sf-a", type=int, default=0)
    parser.add_argument("--use-per-token-sf-b", type=int, default=0)
    parser.add_argument(
        "--per-token-sf-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument(
        "--early-exit-max-token-ctas",
        type=int,
        default=0,
    )
    parser.add_argument("--use-deepseek-fp8", type=int, default=0)
    parser.add_argument("--load-sfab-regs", type=int, default=48)
    parser.add_argument("--bias-type", type=int, default=0)
    parser.add_argument("--has-gemm1-alpha", type=int, choices=[0, 1], default=0)
    parser.add_argument("--has-gemm1-beta", type=int, choices=[0, 1], default=0)
    parser.add_argument("--has-gemm1-clamp-limit", type=int, choices=[0, 1], default=0)

    parser.add_argument(
        "--problem-n",
        type=int,
        default=None,
        help="Problem N dimension (default: tile_n)",
    )
    parser.add_argument(
        "--problem-k",
        type=int,
        default=None,
        help="Problem K dimension (default: tile_k)",
    )

    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Number of timed benchmark iterations.",
    )
    parser.add_argument("--num-rotated-buffers", type=int, default=60)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-c", type=float, default=1.0)
    parser.add_argument("--scale-gate", type=float, default=1.0)
    parser.add_argument("--gemm1-alpha", type=float, default=1.0)
    parser.add_argument("--gemm1-beta", type=float, default=0.0)
    parser.add_argument("--gemm1-clamp-limit", type=float, default=2.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run schedule validation, correctness, and/or benchmark from CLI args.

    Modes:
      --validate-only : schedule validation only, no GPU.
      (default)       : correctness check on GPU.
      --benchmark     : timed benchmark (preceded by the correctness check
                        unless --skip-ref-check is given).
    """
    args = build_arg_parser().parse_args(argv)
    overrides = _parse_overrides(args)

    if args.validate_only:
        validate_schedule(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            **overrides,
        )
        return 0

    # GPU path: correctness runs by default; --benchmark adds timing.
    if not args.skip_ref_check:
        reference_check(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
            scale_c_value=args.scale_c,
            scale_gate_value=args.scale_gate,
            gemm1_alpha_value=args.gemm1_alpha,
            gemm1_beta_value=args.gemm1_beta,
            gemm1_clamp_limit_value=args.gemm1_clamp_limit,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            **overrides,
        )

    if args.benchmark:
        num_rotated_buffers = 0 if args.no_cold_l2 else args.num_rotated_buffers
        benchmark(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
            scale_c_value=args.scale_c,
            scale_gate_value=args.scale_gate,
            gemm1_alpha_value=args.gemm1_alpha,
            gemm1_beta_value=args.gemm1_beta,
            gemm1_clamp_limit_value=args.gemm1_clamp_limit,
            warmup_iters=args.warmup_iterations,
            bench_iters=args.benchmark_iterations,
            num_rotated_buffers=num_rotated_buffers,
            cuda_profiler_range=args.cuda_profiler_range,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            use_cuda_graphs=not args.no_cuda_graphs,
            **overrides,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
