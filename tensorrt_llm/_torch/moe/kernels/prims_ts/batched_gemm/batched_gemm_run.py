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
from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
import gc
import math
import os
import random

from ..cutlass_dsl import task_scheduling_scope
from ..utils import is_prims_ts_arch_available


def _cute_compile_options() -> str:
    return os.environ.get("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2")


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _binary_cli_flag(value: str) -> int:
    """Parse a CLI boolean while retaining the config's integer ABI."""
    normalized = value.lower()
    if normalized in {"1", "true"}:
        return 1
    if normalized in {"0", "false"}:
        return 0
    raise argparse.ArgumentTypeError("expected one of: true, false, 1, 0")


def _auto_or_float_cli_arg(value: str) -> float | None:
    """Parse ``auto`` as None, otherwise require a floating-point value."""
    if value.lower() == "auto":
        return None
    try:
        return float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected 'auto' or a float") from err


def _routing_cli_arg(value: str) -> str:
    """Validate a token-routing policy while preserving its CLI spelling."""
    try:
        _parse_routing(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    return value


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
    CtaRasterOrder,
    RouteImpl,
    TileScheduler,
    ActKind,
    DType,
    SfLayout,
    resolve_problem_config,
    make_config,
    resolve_early_exit_max_token_ctas,
    shuffled_matrix_row_indices,
    validate_launch_grid,
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


def _torch_supports_cuda_arch(major: int, minor: int) -> bool:
    """Whether this PyTorch build advertises native codegen for an SM target."""
    arch_prefixes = (f"sm_{major}{minor}", f"compute_{major}{minor}")
    return any(arch.startswith(arch_prefixes) for arch in torch.cuda.get_arch_list())


def _quantize_block_scaled_for_reference(
    data: torch.Tensor,
    *,
    dtype_kind: int,
    uses_mx: bool,
    sf_vec_size: int,
):
    """Quantize reference inputs, falling back to CPU for unsupported Rubin.

    PyTorch builds that do not advertise SM107 generate an invalid NVRTC target
    for the ``exp2`` used by MX scale quantization. Keep the fallback limited to
    that combination; kernel compilation and execution remain on the GPU.
    """
    original_device = data.device
    quant_data = data
    if data.is_cuda and uses_mx:
        major, minor = torch.cuda.get_device_capability(data.device)
        if (major, minor) == (10, 7) and not _torch_supports_cuda_arch(major, minor):
            quant_data = data.cpu()

    storage, scales = _quantize_block_scaled(
        quant_data,
        dtype_kind=dtype_kind,
        uses_mx=uses_mx,
        sf_vec_size=sf_vec_size,
    )
    if quant_data.device != original_device:
        storage = storage.to(original_device)
        scales = scales.to(original_device)
    return storage, scales


DTYPE_NAME_TO_VALUE = {
    "fp32": DType.FP32,
    "fp16": DType.FP16,
    "bf16": DType.BF16,
    "e2m1": DType.E2M1,
    "mxe2m1": DType.MXE2M1,
    "mxe4m3": DType.MXE4M3,
    "e4m3": DType.E4M3,
}

DATA_DISTRIBUTIONS = ("default", "allZeros")


def _uses_all_zero_input_data(data_distribution: str) -> bool:
    """Validate the TRTLLM-gen-compatible input distribution setting."""
    if data_distribution not in DATA_DISTRIBUTIONS:
        raise ValueError(
            f"unknown data distribution {data_distribution!r}; "
            f"expected one of {DATA_DISTRIBUTIONS}"
        )
    return data_distribution == "allZeros"


def _zero_input_data_tensors(*tensors: torch.Tensor | None) -> None:
    """Zero numerical kernel inputs without touching routing metadata.

    Callers pass token/weight storage and their scale-factor tensors after
    layout expansion or quantization. Routing maps, tile indices, and token
    limits are deliberately not accepted or modified here.
    """
    for tensor in tensors:
        if tensor is not None:
            tensor.zero_()


def _move_to_cuda_memo(
    memo: dict[int, torch.Tensor], tensor: torch.Tensor | None
) -> torch.Tensor | None:
    """Move a tensor to CUDA once while preserving exact-object aliases."""
    if tensor is None:
        return None
    key = id(tensor)
    moved = memo.get(key)
    if moved is None:
        moved = tensor if tensor.is_cuda else tensor.to("cuda")
        memo[key] = moved
    return moved


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
    if cfg.dtype_c == int(DType.FP16):
        return torch.float16
    return torch.bfloat16


def _plain_output_cutlass_dtype(cfg):
    if cfg.dtype_c == int(DType.FP16):
        return cutlass.Float16
    return cutlass.BFloat16


def _is_gated_act_kind(act_kind: int) -> bool:
    return act_kind in (
        int(ActKind.SWIGLU),
        int(ActKind.GEGLU),
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
        """Number of CTA tiles along the padded token dimension."""
        return self.total_padded_tokens // self.tile_size

    @property
    def num_token_cgas(self) -> int:
        """Number of CGA-scoped routing metadata entries."""
        return len(self.tile_idx)

    @property
    def token_cluster_dim(self) -> int:
        """Number of CTA token tiles represented by one metadata entry."""
        return self.cga_tile_size // self.tile_size


@dataclass(frozen=True)
class ReferenceCheckDiagnostics:
    """Structured numerical diagnostics with bool-compatible pass semantics."""

    passed: bool
    max_abs: float | None = None
    mean_abs: float | None = None
    max_rel: float | None = None
    threshold: float | None = None
    nan_count: int = 0
    inf_count: int = 0
    error: str | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class _CachedTokenLayout:
    """Immutable host-only payload stored by the token-layout cache."""

    cga_tile_size: int
    tokens_per_expert: tuple[int, ...]
    padded_starts: tuple[int, ...]
    expanded_to_expert: tuple[int, ...]
    expanded_to_token: tuple[int, ...]
    tile_idx: tuple[int, ...]
    mn_limit: tuple[int, ...]


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _parse_routing(routing: str) -> tuple[str, float | None]:
    """Validate a routing policy and return its normalized kind and parameter."""
    if routing in ("balanced_round_robin", "uniform"):
        return routing, None
    if routing.startswith("zipf:"):
        alpha_text = routing.removeprefix("zipf:")
        try:
            alpha = float(alpha_text)
        except ValueError as err:
            raise ValueError(
                f"invalid Zipf routing alpha in {routing!r}; expected zipf:<alpha>"
            ) from err
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                f"Zipf routing alpha must be finite and positive, got {alpha_text!r}"
            )
        return "zipf", alpha
    raise ValueError(
        f"unknown routing policy {routing!r}; expected balanced_round_robin, "
        "uniform, or zipf:<alpha>"
    )


def _sample_zipf_experts(
    rng: random.Random,
    *,
    cumulative_weights: tuple[float, ...],
    top_k: int,
    alpha: float,
) -> tuple[int, ...]:
    """Sample distinct Zipf-ranked experts using a deterministic host RNG."""
    num_experts = len(cumulative_weights)
    population = range(num_experts)
    selected: list[int] = []
    selected_set: set[int] = set()
    max_rejection_attempts = max(64, top_k * 32)
    attempts = 0
    while len(selected) < top_k and attempts < max_rejection_attempts:
        draw = rng.random() * cumulative_weights[-1]
        expert_idx = min(bisect_right(cumulative_weights, draw), num_experts - 1)
        attempts += 1
        if expert_idx not in selected_set:
            selected.append(expert_idx)
            selected_set.add(expert_idx)

    # Highly concentrated Zipf distributions can make rejection sampling slow.
    # Finish with exact weighted-without-replacement draws to guarantee progress.
    while len(selected) < top_k:
        remaining = [
            expert_idx for expert_idx in population if expert_idx not in selected_set
        ]
        first_rank = remaining[0] + 1
        remaining_weights = [
            ((expert_idx + 1) / first_rank) ** -alpha for expert_idx in remaining
        ]
        expert_idx = rng.choices(remaining, weights=remaining_weights, k=1)[0]
        selected.append(expert_idx)
        selected_set.add(expert_idx)
    return tuple(selected)


@lru_cache(maxsize=16)
def _make_token_layout_cached(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    tile_size: int,
    cluster_dim_in_token: int,
    routing: str,
    routing_alpha: float | None,
    routing_seed: int,
) -> _CachedTokenLayout:
    """Build and cache immutable host routing metadata (never GPU tensors)."""
    cga_tile_size = tile_size * cluster_dim_in_token
    tokens_per_expert = [0] * num_experts
    assignments: list[tuple[int, int, int]] = []
    rng = random.Random(routing_seed)
    zipf_cumulative_weights: tuple[float, ...] | None = None
    if routing == "zipf":
        assert routing_alpha is not None
        cumulative = 0.0
        cumulative_weights = []
        for expert_idx in range(num_experts):
            cumulative += (expert_idx + 1) ** -routing_alpha
            cumulative_weights.append(cumulative)
        zipf_cumulative_weights = tuple(cumulative_weights)
    for token_idx in range(num_tokens):
        if routing == "balanced_round_robin":
            token_experts = tuple(
                (token_idx * top_k + k_idx) % num_experts for k_idx in range(top_k)
            )
        elif routing == "uniform":
            token_experts = tuple(rng.sample(range(num_experts), top_k))
        else:
            assert (
                routing == "zipf"
                and routing_alpha is not None
                and zipf_cumulative_weights is not None
            )
            token_experts = _sample_zipf_experts(
                rng,
                cumulative_weights=zipf_cumulative_weights,
                top_k=top_k,
                alpha=routing_alpha,
            )
        for expert_idx in token_experts:
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

    num_token_cgas = total_padded // cga_tile_size if total_padded > 0 else 1
    tile_idx = [0] * num_token_cgas
    mn_limit = [0] * num_token_cgas
    for expert_idx, token_count in enumerate(tokens_per_expert):
        padded_start = padded_starts[expert_idx]
        padded_end = padded_starts[expert_idx + 1]
        valid_end = padded_start + token_count
        for cga in range(padded_start // cga_tile_size, padded_end // cga_tile_size):
            tile_idx[cga] = expert_idx
            mn_limit[cga] = valid_end

    return _CachedTokenLayout(
        cga_tile_size=cga_tile_size,
        tokens_per_expert=tuple(tokens_per_expert),
        padded_starts=tuple(padded_starts),
        expanded_to_expert=tuple(expanded_to_expert),
        expanded_to_token=tuple(expanded_to_token),
        tile_idx=tuple(tile_idx),
        mn_limit=tuple(mn_limit),
    )


def _make_token_layout(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    tile_size: int,
    cluster_dim_in_token: int,
    routing: str = "balanced_round_robin",
    routing_seed: int = 42,
) -> TokenLayout:
    """Build the MoE expanded/permuted token layout for a routing policy."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if cluster_dim_in_token <= 0:
        raise ValueError(
            f"cluster_dim_in_token must be positive, got {cluster_dim_in_token}"
        )
    if top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")
    routing_kind, routing_alpha = _parse_routing(routing)
    cached = _make_token_layout_cached(
        num_tokens,
        num_experts,
        top_k,
        tile_size,
        cluster_dim_in_token,
        routing_kind,
        routing_alpha,
        int(routing_seed),
    )
    return TokenLayout(
        tile_size=tile_size,
        cga_tile_size=cached.cga_tile_size,
        tokens_per_expert=list(cached.tokens_per_expert),
        padded_starts=list(cached.padded_starts),
        expanded_to_expert=list(cached.expanded_to_expert),
        expanded_to_token=list(cached.expanded_to_token),
        tile_idx=list(cached.tile_idx),
        mn_limit=list(cached.mn_limit),
    )


def _launch_metadata_lists(
    cfg,
    token_layout: TokenLayout,
    early_exit_max_token_ctas: int,
):
    """Return CGA routing metadata and the CTA-row route map for launch."""
    launch_token_ctas = token_layout.num_token_tiles
    launch_token_cgas = token_layout.num_token_cgas
    if cfg.use_early_exit and cfg.is_persistent:
        launch_token_ctas = max(launch_token_ctas, early_exit_max_token_ctas)
        launch_token_cgas = max(
            launch_token_cgas,
            (launch_token_ctas + token_layout.token_cluster_dim - 1)
            // token_layout.token_cluster_dim,
        )

    tile_idx = list(token_layout.tile_idx)
    mn_limit = list(token_layout.mn_limit)
    if launch_token_cgas > token_layout.num_token_cgas:
        extra_cgas = launch_token_cgas - token_layout.num_token_cgas
        # Persistent early-exit over-launches token CGAs for CUDA graph reuse.
        # Extra metadata is inactive because the scheduler exits before those
        # CGAs read routing tables.
        tile_idx.extend([0] * extra_cgas)
        mn_limit.extend([0] * extra_cgas)

    route_map = list(token_layout.expanded_to_token)
    min_route_rows = launch_token_ctas * token_layout.tile_size
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


def _validate_swap_ab_out_hidden(cfg, out_hidden: int) -> None:
    """Reject swapAB output widths the weight-row tiling cannot serve.

    DeepSeek FP8 predicates a partial last ``tile_m`` tile in the epilogue and
    zero-fills its rows through the TMA map, so it only needs whole scale row
    groups: a multiple of 32, the smallest ``deepseek_fp8_block_size``. The
    other swapAB paths keep the whole-tile requirement.
    """
    if not cfg.is_swap_ab:
        return
    unit = 32 if cfg.use_deepseek_fp8 else cfg.tile_m
    if out_hidden % unit != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of {unit}, got {out_hidden}"
        )


def _runtime_config(cfg, in_hidden: int):
    """Normalize runtime options for a concrete K."""
    if cfg.use_deepseek_fp8:
        block = cfg.deepseek_fp8_block_size
        if cfg.tile_k != 128:
            raise ValueError(
                "use_deepseek_fp8=1 requires tile_k=128 to match the "
                f"DeepSeek scale-factor tiles, got {cfg.tile_k}"
            )
        # A partial last K tile is zero-filled by TMA and its sub-blocks past
        # K are skipped by the epilogue, so K only has to be block-aligned.
        if in_hidden % block != 0:
            raise ValueError(
                "use_deepseek_fp8=1 requires problem_k to be a multiple of "
                f"deepseek_fp8_block_size={block}, got problem_k={in_hidden}"
            )
    return resolve_problem_config(cfg, in_hidden)


def _expand_bf16_activations(
    compact_activations: torch.Tensor,
    token_layout: TokenLayout,
) -> torch.Tensor:
    expanded = torch.zeros(
        (token_layout.total_padded_tokens, compact_activations.shape[1]),
        dtype=compact_activations.dtype,
        device=compact_activations.device,
    )
    expanded_to_expert = torch.as_tensor(
        token_layout.expanded_to_expert,
        dtype=torch.long,
        device=compact_activations.device,
    )
    valid_rows = torch.nonzero(expanded_to_expert >= 0, as_tuple=False).flatten()
    if valid_rows.numel():
        expanded_to_token = torch.as_tensor(
            token_layout.expanded_to_token,
            dtype=torch.long,
            device=compact_activations.device,
        )
        expanded.index_copy_(
            0,
            valid_rows,
            compact_activations.index_select(
                0, expanded_to_token.index_select(0, valid_rows)
            ),
        )
    return expanded


def _expand_fp4_activations(
    compact_activations: torch.Tensor,
    token_layout: TokenLayout,
) -> torch.Tensor:
    return _expand_bf16_activations(compact_activations, token_layout)


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
    if cfg.use_deepseek_fp8:
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
    Weight scales are per expert, output row group, and K block. SwapAB groups
    output rows by ``deepseek_fp8_block_size``; the non-swap path groups them
    by ``tile_n``.
    """
    if not cfg.use_deepseek_fp8:
        raise ValueError("DeepSeek FP8 scale tensors require use_deepseek_fp8=1")

    block = cfg.deepseek_fp8_block_size
    ds_k_blocks = _round_up(in_hidden, block) // block
    if cfg.is_swap_ab:
        ds_weight_tiles = _round_up(out_hidden, block) // block
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


_NVFP4_OUTPUT_MAX = 6.0 * 448.0
_AUTO_SCALE_C_ACT_KINDS = (
    int(ActKind.SWIGLU),
    int(ActKind.GEGLU),
    int(ActKind.RELU2),
)


def _auto_scale_c_applies(cfg) -> bool:
    """Whether the harness can predict this NVFP4 output's magnitude."""
    return (
        cfg.dtype_c == int(DType.E2M1)
        and cfg.uses_global_scales
        and not cfg.is_bf16_mma
        and not cfg.has_cast_a
        and not cfg.is_fp8_mma
        and cfg.act_kind in _AUTO_SCALE_C_ACT_KINDS
        and not cfg.has_gemm1_clamp_limit
    )


def _resolve_scale_c(
    scale_c_value, *, cfg, in_hidden: int, scale_gate_value: float
) -> float:
    """Choose a power-of-two output scale that avoids NVFP4 saturation."""
    if scale_c_value is not None:
        return float(scale_c_value)
    if not _auto_scale_c_applies(cfg):
        return 1.0

    # The fused-activation generator scales Kaiming-uniform inputs by sqrt(K),
    # so each GEMM has variance K/9 and gated products scale as K/9. Keep the
    # predicted peak below 75% of E2M1 max times E4M3 scale-factor max.
    variance = in_hidden / 9.0
    if cfg.act_kind == int(ActKind.RELU2):
        # ReLU2 squares the gate-scaled GEMM output, so the peak is the square
        # of the largest sample; budget a ~7 sigma tail for the ~1e9 outputs
        # a large FC1 check compares.
        predicted_peak = 48.0 * scale_gate_value**2 * variance
    else:
        predicted_peak = 12.0 * abs(scale_gate_value) * variance
    target = 0.75 * _NVFP4_OUTPUT_MAX
    if predicted_peak <= target:
        return 1.0
    return 2.0 ** math.floor(math.log2(target / predicted_peak))


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
        return _quantize_block_scaled_for_reference(
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
        expanded_a_sf = a_sf.view(torch.uint8)[expanded_to_token[exp_rows]].view(
            a_sf.dtype
        )
        sf_activations = _build_sf_buffer(
            expanded_a_sf,
            total_bytes=a_sf_total,
            layout=activation_sf_layout,
            data_blk_cols=sf_k,
            device=device,
            sf_rows=exp_rows,
        )

    return weights_torch, sf_weights, activation_compact_torch, sf_activations


def _sf_offsets(
    *,
    row_idx: int | torch.Tensor,
    sf_cols: int,
    layout: int,
    num_padded_sf_cols: int | None = None,
    device: str | torch.device,
) -> torch.Tensor:
    """Scale-factor byte offsets for one or more data rows.

    Computes SF byte offsets for Linear, R8c4 and R128c4 layouts.
    Weight SF tensors use a single physical row coordinate
    with experts folded into the row dimension.
    """
    data_blk_cols = num_padded_sf_cols if num_padded_sf_cols is not None else sf_cols
    rows = torch.as_tensor(row_idx, dtype=torch.long, device=device)
    scalar_row = rows.ndim == 0
    rows = rows.reshape(-1, 1)
    kblk = torch.arange(sf_cols, dtype=torch.long, device=device).reshape(1, -1)
    if layout == int(SfLayout.LINEAR):
        offsets = rows * data_blk_cols + kblk
        return offsets[0] if scalar_row else offsets

    if layout == int(SfLayout.R8c4):
        rows_per_block = 8
        cols_per_block = 4
        bytes_per_block = 32
        sf_row = rows % rows_per_block
    elif layout == int(SfLayout.R128c4):
        rows_per_block = 128
        cols_per_block = 4
        bytes_per_block = 512
        sf_row = (rows % 32) * 4 + (rows % rows_per_block) // 32
    else:
        raise ValueError(f"Unsupported FP4 SF layout for reference: {layout}")

    sf_blk_row = rows // rows_per_block
    sf_blk_col = kblk // cols_per_block
    sf_blk_idx = sf_blk_row * (data_blk_cols // cols_per_block) + sf_blk_col
    sf_col = kblk % cols_per_block
    offsets = sf_blk_idx * bytes_per_block + sf_row * cols_per_block + sf_col
    return offsets[0] if scalar_row else offsets


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
    return sf.view(sf_torch_dtype)[offsets].float()


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
    DeepSeek FP8 reference: per-K-block FP32 dequant before accumulation.
    For DSFP8, the scale factors are separate FP32 tensors and are applied per K-block.
    The reference implementation needs to loop over expanded token, expert, output tile, and K-block.
    """
    ref_gemm = torch.zeros(
        (m_dim, n_dim), dtype=torch.float32, device=weights_ref.device
    )
    k_block = cfg.deepseek_fp8_block_size
    num_k_blocks = _round_up(in_hidden, k_block) // k_block

    if cfg.is_swap_ab:
        m_block = cfg.deepseek_fp8_block_size
        num_weight_tiles = _round_up(m_dim, m_block) // m_block
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            act_sf_row = token_idx if cfg.has_routed_act else expanded_idx
            for m_tile in range(num_weight_tiles):
                m0 = m_tile * m_block
                m1 = min(m0 + m_block, m_dim)
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


def _dequantize_block_scaled_expert(
    storage: torch.Tensor,
    sf: torch.Tensor,
    *,
    expert_idx: int,
    dtype_kind: int,
    logical_cols: int,
    layout: int,
    sf_vec_size: int,
    sf_torch_dtype,
) -> torch.Tensor:
    """Dequantize one expert's ``[rows, K-storage]`` weights to float32."""
    decoded = _decode_quantized_storage(
        storage[expert_idx],
        dtype_kind=dtype_kind,
        logical_cols=logical_cols,
    )
    rows_per_expert = storage.shape[1]
    sf_cols = _round_up(logical_cols, sf_vec_size) // sf_vec_size
    sf_rows = expert_idx * rows_per_expert + torch.arange(
        rows_per_expert, dtype=torch.long, device=storage.device
    )
    sf_offsets = _sf_offsets(
        row_idx=sf_rows,
        sf_cols=sf_cols,
        layout=layout,
        device=sf.device,
    )
    sf_values = sf.view(sf_torch_dtype)[sf_offsets].float()
    return decoded.mul_(
        sf_values.repeat_interleave(sf_vec_size, dim=-1)[:, :logical_cols]
    )


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
    out = torch.empty(
        (*storage.shape[:-1], logical_cols),
        dtype=torch.float32,
        device=storage.device,
    )
    for expert_idx in range(storage.shape[0]):
        out[expert_idx] = _dequantize_block_scaled_expert(
            storage,
            sf,
            expert_idx=expert_idx,
            dtype_kind=dtype_kind,
            logical_cols=logical_cols,
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
    decoded_compact: torch.Tensor,
    sf_activations: torch.Tensor,
    *,
    token_idx: int,
    expanded_idx: int,
    cfg,
    activation_sf_layout: int,
    sf_k: int,
) -> torch.Tensor:
    """Dequantize one logical activation row for MX/NVFP4 input dtypes."""
    if cfg.has_routed_sfs:
        return _apply_fp4_sf_to_row(
            decoded_compact[token_idx, :],
            sf_activations,
            row_idx=token_idx,
            layout=int(SfLayout.LINEAR),
            sf_vec_size=cfg.input_sf_block_size_a,
            sf_torch_dtype=_sf_torch_dtype(cfg),
            num_padded_sf_cols=_round_up(sf_k, 16),
        )
    return _apply_fp4_sf_to_row(
        decoded_compact[token_idx, :],
        sf_activations,
        row_idx=expanded_idx,
        layout=activation_sf_layout,
        sf_vec_size=cfg.input_sf_block_size_a,
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


def _sf_c_numel(
    output_m: int,
    n: int,
    layout: int,
    sf_block_size: int = 16,
    *,
    is_swap_ab: bool = True,
    pitch: int | None = None,
) -> int:
    """Number of output SF bytes for generated C SF layout."""
    if not is_swap_ab:
        output_m, n = n, output_m
    elif pitch is not None:
        output_m = pitch
    m_group = sf_block_size * 4
    m_groups = _round_up(output_m, m_group) // m_group
    if layout == int(SfLayout.R8c4):
        return _round_up(n, 8) // 8 * m_groups * 32
    if layout == int(SfLayout.R128c4):
        return _round_up(n, 128) // 128 * m_groups * 512
    raise ValueError(f"Unsupported sfLayoutC for quantized output: {layout}")


def _dsfp8_c_numel(output_m: int, n: int, *, is_swap_ab: bool, block: int = 128) -> int:
    """Number of FP32 DeepSeek output dq-scale values."""
    if is_swap_ab:
        return _round_up(output_m, block) // block * n
    return _round_up(n, 128) // 128 * output_m


def _dsfp8_c_scale_index_grid(
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
    device: str,
    scale_stride: int | None = None,
    block: int = 128,
) -> torch.Tensor:
    m_idx = torch.arange(output_m, device=device).view(output_m, 1)
    n_idx = torch.arange(n, device=device).view(1, n)
    if is_swap_ab:
        # The swapAB epilogue strides scale blocks by the routed padded token
        # count, which is narrower than N when the caller declares a wider
        # token capacity.
        return (m_idx // block) * (n if scale_stride is None else scale_stride) + n_idx
    return (n_idx // 128) * output_m + m_idx


def _reference_dsfp8_c_scales(
    ref: torch.Tensor, *, is_swap_ab: bool, block: int = 128
) -> torch.Tensor:
    """Compute DeepSeek FP8 C dequant scales for an FP32 reference."""
    output_m, n = ref.shape
    sf_c = torch.zeros(
        (_dsfp8_c_numel(output_m, n, is_swap_ab=is_swap_ab, block=block),),
        dtype=torch.float32,
        device=ref.device,
    )
    if is_swap_ab:
        for m0 in range(0, output_m, block):
            m1 = min(m0 + block, output_m)
            group = m0 // block
            sf_c[group * n : group * n + n] = ref[m0:m1, :].abs().amax(dim=0) / 448.0
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
    is_swap_ab: bool,
    device: str,
    pitch: int | None = None,
    m_offset: int = 0,
) -> torch.Tensor:
    m_idx = torch.arange(output_m, device=device).view(output_m, 1)
    n_idx = torch.arange(n, device=device).view(1, n)
    if not is_swap_ab:
        if m_offset:
            raise ValueError("SF-C m_offset is supported for swapAB only")
        m_idx, n_idx = n_idx, m_idx
        output_m = n
    else:
        m_idx += int(m_offset)
        if pitch is not None:
            output_m = pitch
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
    pitch: int | None = None,
    sf_c_m_offset: int = 0,
) -> torch.Tensor:
    """Decode packed E2M1 C and its NVFP4/MXFP4 SF-C into a logical view."""
    pitch = output_m if pitch is None else pitch
    elem_count = (n - 1) * pitch + output_m
    packed = c_storage[: _round_up(elem_count, 2) // 2]
    fp4_table = _fp4_e2m1_values(c_storage.device)
    nibbles = torch.empty(
        (packed.numel() * 2,), dtype=torch.long, device=c_storage.device
    )
    nibbles[0::2] = (packed & 0x0F).to(torch.long)
    nibbles[1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    fp4_vals = fp4_table[nibbles[:elem_count]]

    if is_swap_ab:
        fp4_view = torch.as_strided(fp4_vals, (output_m, n), (1, pitch))
    else:
        fp4_view = fp4_vals.view(output_m, n)

    sf_idx = _sf_c_index_grid(
        output_m=output_m,
        n=n,
        sf_layout_c=sf_layout_c,
        sf_block_size=32 if uses_mxfp4 else 16,
        is_swap_ab=is_swap_ab,
        device=c_storage.device,
        pitch=pitch,
        m_offset=sf_c_m_offset,
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
    pitch: int | None = None,
    sf_c_m_offset: int = 0,
) -> torch.Tensor:
    """Decode E4M3 C and UE8M0 SF-C into a float32 logical view."""
    pitch = output_m if pitch is None else pitch
    elem_count = (n - 1) * pitch + output_m
    fp8_vals = c_storage[:elem_count].view(torch.float8_e4m3fn).float()
    if is_swap_ab:
        fp8_view = torch.as_strided(fp8_vals, (output_m, n), (1, pitch))
    else:
        fp8_view = fp8_vals.view(output_m, n)

    sf_idx = _sf_c_index_grid(
        output_m=output_m,
        n=n,
        sf_layout_c=sf_layout_c,
        sf_block_size=32,
        is_swap_ab=is_swap_ab,
        device=c_storage.device,
        pitch=pitch,
        m_offset=sf_c_m_offset,
    )
    sf_vals = sf_c.view(torch.float8_e8m0fnu).float()
    return fp8_view * sf_vals[sf_idx]


def _decode_fp8_output(
    c_storage: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
    pitch: int | None = None,
) -> torch.Tensor:
    """Decode plain per-tensor FP8 C storage into a logical float32 view."""
    pitch = output_m if pitch is None else pitch
    elem_count = (n - 1) * pitch + output_m
    fp8_vals = c_storage[:elem_count].view(torch.float8_e4m3fn).float()
    if is_swap_ab:
        return torch.as_strided(fp8_vals, (output_m, n), (1, pitch))
    return fp8_vals.view(output_m, n)


def _dequantize_dsfp8_output(
    c_storage: torch.Tensor,
    sf_c: torch.Tensor,
    *,
    output_m: int,
    n: int,
    is_swap_ab: bool,
    scale_stride: int | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Decode E4M3 C using DeepSeek FP32 per-block-row dequant scales."""
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
        scale_stride=scale_stride,
        block=block,
    )
    return fp8_view * sf_c.float()[sf_idx]


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
    rows = torch.arange(M, dtype=torch.long, device=tensor.device)
    source_rows = torch.stack((rows[:half], rows[half:]), dim=1).flatten()
    return tensor.index_select(-2, source_rows)


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
    indices = shuffled_matrix_row_indices(epilogue_tile_m)
    block_size = len(indices)

    M = tensor.shape[-2]
    if block_size > M:
        # Matrix too small for shuffling — return as-is.
        return tensor.clone()
    if M % block_size != 0:
        raise ValueError(
            "Shuffled matrix rows must contain whole shuffle blocks, got "
            f"M={M}, block_size={block_size}"
        )
    destination_rows = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    source_rows = torch.empty_like(destination_rows)
    source_rows[destination_rows] = torch.arange(
        block_size, dtype=torch.long, device=tensor.device
    )
    source_rows = (
        source_rows.unsqueeze(0)
        + torch.arange(
            0, M, block_size, dtype=torch.long, device=tensor.device
        ).unsqueeze(1)
    ).flatten()
    return tensor.index_select(-2, source_rows)


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
    shuffle_weight_rows: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return the original weight row stored at each kernel weight row."""
    row_ids = torch.arange(num_rows, dtype=torch.long, device=device).view(
        1, num_rows, 1
    )
    if is_gated:
        row_ids = reorder_rows_for_gated_act(row_ids)
    if shuffle_weight_rows:
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
    out = sf.clone()
    sf_cols = _round_up(logical_cols, sf_vec_size) // sf_vec_size
    row_sources = row_source_indices.to(dtype=torch.long, device=sf.device)
    rows_per_expert = row_sources.numel()
    expert_offsets = (
        torch.arange(num_experts, dtype=torch.long, device=sf.device) * rows_per_expert
    ).unsqueeze(1)
    src_rows = (expert_offsets + row_sources.unsqueeze(0)).flatten()
    dst_rows = (
        expert_offsets
        + torch.arange(rows_per_expert, dtype=torch.long, device=sf.device).unsqueeze(0)
    ).flatten()
    src_offsets = _sf_offsets(
        row_idx=src_rows,
        sf_cols=sf_cols,
        layout=layout,
        device=sf.device,
    )
    dst_offsets = _sf_offsets(
        row_idx=dst_rows,
        sf_cols=sf_cols,
        layout=layout,
        device=sf.device,
    )
    out[dst_offsets.flatten()] = sf[src_offsets.flatten()]
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
    shuffle_weight_rows: bool,
) -> torch.Tensor | None:
    """Return weight SFs reordered to match kernel weight storage."""
    if sf_weights is None or (not is_gated and not shuffle_weight_rows):
        return sf_weights
    assert weight_sf_layout is not None
    row_source_indices = _kernel_weight_row_source_indices(
        out_hidden,
        is_gated=is_gated,
        epilogue_tile_m=cfg.epi_tile_m,
        shuffle_weight_rows=shuffle_weight_rows,
        device=sf_weights.device,
    )
    return _permute_mma_sf_rows(
        sf_weights,
        row_source_indices=row_source_indices,
        logical_cols=in_hidden,
        layout=weight_sf_layout,
        num_experts=num_experts,
        sf_vec_size=cfg.input_sf_block_size_a,
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
            }[args.route_act]
        ),
        "route_sfs_act": int(
            {
                "auto": RouteImpl.AUTO,
                "false": RouteImpl.NONE,
                "none": RouteImpl.NONE,
                "tma": RouteImpl.TMA,
                "ldgsts": RouteImpl.LDGSTS,
            }[args.route_sfs_act]
        ),
        "tile_scheduler": int(
            {
                "static": TileScheduler.STATIC,
                "persistent": TileScheduler.PERSISTENT,
            }[args.tile_scheduler]
        ),
        "cta_raster_order": int(
            {
                "rasterizeAlongM": CtaRasterOrder.ALONG_M,
                "rasterizeAlongN": CtaRasterOrder.ALONG_N,
            }[args.cta_swizzle_type]
        ),
        "transpose_mma_output": args.transpose_mma_output,
        "use_shuffled_matrix": args.use_shuffled_matrix,
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
        "umma_cta_count": args.umma_cta_count,
        "cluster_dim_x": args.cluster_dim_x,
        "cluster_dim_y": args.cluster_dim_y,
        "fallback_cluster_dim_x": args.fallback_cluster_dim_x,
        "fallback_cluster_dim_y": args.fallback_cluster_dim_y,
        "tile_n": args.tile_n,
        "tile_k": args.tile_k,
        "epi_tile_n": args.epi_tile_n,
        "mma_m": args.mma_m,
        "mma_n": args.mma_n,
        "mma_k": args.mma_k,
        "dtype_a": int(DTYPE_NAME_TO_VALUE[args.dtype_a]),
        "dtype_b": int(DTYPE_NAME_TO_VALUE[args.dtype_b]),
        "dtype_c": int(DTYPE_NAME_TO_VALUE[args.dtype_c]),
        "num_stages_a": args.num_stages_a,
        "num_stages_b": args.num_stages_b,
        "num_stages_smem_sfa": args.num_stages_smem_sfa,
        "num_stages_smem_sfb": args.num_stages_smem_sfb,
        "num_stages_tmem_acc": args.num_stages_tmem_acc,
        "num_stages_tmem_sfa": args.num_stages_tmem_sfa,
        "num_stages_tmem_sfb": args.num_stages_tmem_sfb,
        "num_stages_c_smem": args.num_stages_c_smem,
        "tmem_ldst_max_num_regs": args.tmem_ldst_max_num_regs,
        "use_tma_oob_opt": args.use_tma_oob_opt,
        "use_tma_store": args.use_tma_store,
        "num_load_a_warps": args.num_load_a_warps,
        "num_load_b_warps": args.num_load_b_warps,
        "num_load_sfa_warps": args.num_load_sfa_warps,
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
        "fuse_sf_copy_to_mma": args.fuse_sf_copy_to_mma,
        "fuse_operand_sf_loads": args.fuse_operand_sf_loads,
        "use_early_exit": args.use_early_exit,
        "use_clc_fast_drain": args.use_clc_fast_drain,
        "use_work_throttle": args.use_work_throttle,
        "use_pdl": args.use_pdl,
        "skip_early_exit_pdl": args.skip_early_exit_pdl,
        "use_global_scales": args.use_global_scales,
        "use_per_token_sf_a": args.use_per_token_sf_a,
        "use_per_token_sf_b": args.use_per_token_sf_b,
        "per_token_sf_dtype": int(DTYPE_NAME_TO_VALUE[args.per_token_sf_dtype]),
        "use_deepseek_fp8": args.use_deepseek_fp8,
        # DeepSeek FP8 will use a single LoadSfAb warp.
        "load_sfab_regs": args.load_sfab_regs,
        "bias_type": args.bias_type,
        "has_gemm1_alpha": args.has_gemm1_alpha,
        "has_gemm1_beta": args.has_gemm1_beta,
        "has_gemm1_clamp_limit": args.has_gemm1_clamp_limit,
    }
    if args.sm_arch is not None:
        overrides["sm_arch"] = args.sm_arch
    return overrides


def validate_schedule(
    *,
    num_experts,
    num_tokens,
    top_k,
    early_exit_max_token_ctas=0,
    **cfg_overrides,
):
    cfg_overrides = _resolve_runtime_sm_arch(cfg_overrides)
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


_C_STRIDE_SENTINEL_F = 123.0
_C_STRIDE_SENTINEL_U8 = 0xA5

# Poison for the region the epilogue is expected to write. A zero-filled output
# cannot tell "stored a zero" from "never stored", which is how a swapAB
# epilogue that dropped the upper half of every M group went unnoticed: the
# unwritten elements read as zero and the numeric threshold is an absolute
# floor, so a small-magnitude output still compared clean. NaN survives no
# arithmetic, so any element left poisoned fails the existing NaN check.
_C_UNWRITTEN_POISON_F = float("nan")
# E4M3 encodes NaN at 0x7F, so poisoned FP8 storage decodes to NaN too. E2M1
# has no NaN encoding, so FP4 output keeps its zero fill and is not covered.
_C_UNWRITTEN_POISON_E4M3_U8 = 0x7F


def _same_launch_output(first: torch.Tensor, current: torch.Tensor) -> bool:
    """Whether two launches produced the same C buffer.

    Elements the epilogue never stores keep their poison, so plain equality
    would read the unwritten NaNs as a difference and report a deterministic
    kernel as nondeterministic. Treat two NaNs in the same position as equal
    and let the NaN coverage check downstream report them instead.
    """
    if torch.equal(first, current):
        return True
    if not first.dtype.is_floating_point:
        return False
    matches = (first == current) | (torch.isnan(first) & torch.isnan(current))
    return bool(matches.all().item())


def _partition_view(storage, width, cols, pitch):
    """Logical [width, cols] view of one partition inside a pitch-strided buffer."""
    return torch.as_strided(storage, (width, cols), (1, pitch))


def _mixed_cga_wave_counts(
    *,
    logical_work_cgas,
    physical_grid_ctas,
    umma_cta_count,
    preferred_shape,
    fallback_shape,
):
    """Return logical work/grid CGA counts for both physical shapes.

    A wider physical CGA contains multiple independent ``umma_cta_count x 1``
    UMMA atoms.  Convert the logical work into the number of physical CGAs it
    would occupy for each shape, and compare that with an all-preferred or
    all-fallback wave using the actual persistent grid's CTA budget.
    """

    def counts(shape):
        physical_size = math.prod(shape)
        logical_cgas_per_physical_cga = physical_size // umma_cta_count
        work_cgas = (
            logical_work_cgas + logical_cgas_per_physical_cga - 1
        ) // logical_cgas_per_physical_cga
        wave_cgas = physical_grid_ctas // physical_size
        return work_cgas, wave_cgas

    return counts(preferred_shape), counts(fallback_shape)


def _resolve_runtime_sm_arch(cfg_overrides: dict) -> dict:
    """Fill ``sm_arch`` from the active CUDA device when it is not explicit."""
    if "sm_arch" in cfg_overrides or not torch.cuda.is_available():
        return cfg_overrides
    major, minor = torch.cuda.get_device_capability()
    detected = major * 10 + minor
    if detected not in (100, 103, 107):
        return cfg_overrides
    if not is_prims_ts_arch_available(major, minor):
        hint = "; install a Rubin-capable DSL" if (major, minor) == (10, 7) else ""
        raise RuntimeError(
            f"the installed CuTe DSL cannot natively target SM{detected}{hint}"
        )
    resolved = dict(cfg_overrides)
    resolved["sm_arch"] = detected
    return resolved


def reference_check(
    *,
    num_experts,
    num_tokens,
    top_k,
    routing="balanced_round_robin",
    routing_seed=42,
    problem_n=None,
    problem_k=None,
    seed=42,
    data_distribution="default",
    scale_c_value=None,
    scale_gate_value=1.0,
    gemm1_alpha_value=None,
    gemm1_beta_value=None,
    gemm1_clamp_limit_value=2.0,
    return_output=False,
    repeat_launches=1,
    output_guard_elements=0,
    require_mixed_cga_runtime=False,
    c_row_stride=None,
    c_partition_offset=0,
    early_exit_max_token_ctas=0,
    token_capacity=None,
    input_device: str = "cuda",
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

    ``token_capacity`` declares a token extent wider than the routed padded
    count, the way the MoE runner must: its ``routed_token_capacity`` has to be
    a host-known worst-case bound because the real count only exists on the
    device. Token-axis buffers grow to the capacity and the kernel still reads
    the routed count from ``total_num_padded_tokens``, so the padded tail stays
    outside the compared output.

    ``require_mixed_cga_runtime`` additionally requires
    the logical work to exceed one persistent-grid wave for both the preferred
    and fallback physical CGA shapes.

    ``scale_c_value=None`` selects a power-of-two scale for generated fused
    activations with NVFP4 output so the reference stays below the format's
    saturation limit. Pass a float to use an explicit scale.

    """
    if repeat_launches < 1:
        raise ValueError(f"repeat_launches must be positive, got {repeat_launches}")
    cfg_overrides = _resolve_runtime_sm_arch(cfg_overrides)
    launch_early_exit_max_token_ctas = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    all_zero_input_data = _uses_all_zero_input_data(data_distribution)
    cfg = make_config(**cfg_overrides)
    if require_mixed_cga_runtime:
        if cfg.fallback_cluster_shape is None:
            raise ValueError(
                "require_mixed_cga_runtime requires distinct preferred and "
                "fallback cluster shapes"
            )
        if not cfg.is_persistent:
            raise ValueError(
                "require_mixed_cga_runtime requires the persistent tile scheduler"
            )
        cfg = replace(cfg, record_runtime_cga_shapes=1)

    torch.manual_seed(seed)
    device = input_device
    from cutlass.cute.runtime import make_ptr

    # --- Domain-level problem shape ---
    total_tokens = num_tokens
    # swapAB maps FC1 weights to operand A, whose TMA box spans tile_m rows.
    # Keep the default legal for the generated transOut kernels.
    default_out_hidden = cfg.tile_m if cfg.is_swap_ab else cfg.tile_n
    out_hidden = problem_n if problem_n is not None else default_out_hidden
    in_hidden = problem_k if problem_k is not None else cfg.tile_k
    cfg = _runtime_config(cfg, in_hidden)
    scale_c_is_auto = scale_c_value is None
    scale_c_value = _resolve_scale_c(
        scale_c_value,
        cfg=cfg,
        in_hidden=in_hidden,
        scale_gate_value=scale_gate_value,
    )

    _validate_swap_ab_out_hidden(cfg, out_hidden)

    token_tile_size = cfg.token_tile_size
    cluster_dim_in_token = cfg.token_cluster_dim
    token_layout = _make_token_layout(
        num_tokens=total_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
        routing=routing,
        routing_seed=routing_seed,
    )
    total_padded_tokens = token_layout.total_padded_tokens
    if total_padded_tokens == 0:
        raise ValueError(
            "routing produced no assignments for the selected local expert slice"
        )
    declared_padded_tokens = total_padded_tokens
    if token_capacity is not None:
        declared_padded_tokens = int(token_capacity)
        if not cfg.is_swap_ab or not cfg.use_deepseek_fp8:
            raise ValueError(
                "token_capacity currently covers swapAB DeepSeek FP8 only, "
                "whose token-axis buffers are plain (rows, k) and "
                "(k_blocks, rows) tensors that widen directly"
            )
        if declared_padded_tokens < total_padded_tokens:
            raise ValueError(
                f"token_capacity={declared_padded_tokens} is narrower than the "
                f"routed padded token count {total_padded_tokens}"
            )
        if declared_padded_tokens % token_layout.cga_tile_size:
            raise ValueError(
                f"token_capacity={declared_padded_tokens} must be a multiple "
                f"of the CGA token tile {token_layout.cga_tile_size}"
            )
        # Over-launch the token dimension so the routing metadata carries the
        # inactive tail tiles, as it does for an over-declared MoE launch.
        launch_early_exit_max_token_ctas = max(
            int(launch_early_exit_max_token_ctas),
            declared_padded_tokens // token_tile_size,
        )
    launch_early_exit_max_token_ctas = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )

    print(
        f"Reference check: tokens={total_tokens}, padded_tokens={total_padded_tokens}, "
        f"declared_tokens={declared_padded_tokens}, "
        f"out_hidden={out_hidden}, in_hidden={in_hidden}, experts={num_experts}, "
        f"top_k={top_k}, routing={routing}, routing_seed={routing_seed}, "
        f"swap_ab={cfg.is_swap_ab}, dist={data_distribution}, "
        f"scale_c={scale_c_value:g}{' (auto)' if scale_c_is_auto else ''}"
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
        sf_k = in_hidden // cfg.input_sf_block_size_a
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
            w_storage, w_sf = _quantize_block_scaled_for_reference(
                w_fp32,
                dtype_kind=cfg.dtype_a,
                uses_mx=cfg.uses_mx_scale_factors,
                sf_vec_size=cfg.input_sf_block_size_a,
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
                dtype_kind=cfg.dtype_a,
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
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = cutlass.BFloat16
        activation_sf_layout = None
        sf_activations = None
    elif cfg.is_fp8_mma:
        weights_torch = _random_quantized_tensor(
            (num_experts, out_hidden, in_hidden),
            dtype_kind=cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b,
            fan_in=in_hidden,
            device=device,
        )
        activation_compact_torch = _random_quantized_tensor(
            (total_tokens, in_hidden),
            dtype_kind=cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b)
        weight_sf_layout = None
        activation_sf_layout = None
        sf_weights = sf_activations = None
        if cfg.use_deepseek_fp8:
            sf_weights, sf_activations = _create_deepseek_fp8_sf_tensors(
                cfg,
                num_experts=num_experts,
                num_tokens=total_tokens,
                out_hidden=out_hidden,
                in_hidden=in_hidden,
                total_padded_tokens=declared_padded_tokens,
                device=device,
            )
    else:
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b)
        sf_vec_size = cfg.input_sf_block_size_a

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
                weight_dtype_kind=(cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b),
                activation_dtype_kind=(cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a),
                uses_mx=cfg.uses_mx_scale_factors,
                sf_vec=cfg.input_sf_block_size_a,
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
                dtype_kind=cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b,
                fan_in=in_hidden,
                device=device,
            )
            activation_compact_torch = _random_quantized_tensor(
                (total_tokens, in_hidden),
                dtype_kind=cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a,
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

    if all_zero_input_data:
        # Zero both compact and expanded activation storage: routed kernels use
        # the former, while non-routed kernels consume the latter. Applying the
        # distribution before weight/SF preprocessing lets every shuffled or
        # block-major representation inherit the zero values.
        _zero_input_data_tensors(
            weights_torch,
            activation_compact_torch,
            activations_torch,
            sf_weights,
            sf_activations,
        )

    # Keep the reference math and kernel launch on CUDA while allowing the
    # expensive random input construction to run on the host.
    move_memo: dict[int, torch.Tensor] = {}
    weights_torch = _move_to_cuda_memo(move_memo, weights_torch)
    activation_compact_torch = _move_to_cuda_memo(move_memo, activation_compact_torch)
    activations_torch = _move_to_cuda_memo(move_memo, activations_torch)
    sf_weights = _move_to_cuda_memo(move_memo, sf_weights)
    sf_activations = _move_to_cuda_memo(move_memo, sf_activations)
    device = "cuda"

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

    if declared_padded_tokens > total_padded_tokens and not cfg.has_routed_act:
        # Expanded activations are indexed by padded token row, so widening the
        # declared token extent also widens the operand. Tail rows stay zero.
        # Build the wider buffer by hand because F.pad rejects FP8 operands.
        widened_activations = torch.zeros(
            (declared_padded_tokens, activations_torch.shape[1]),
            dtype=activations_torch.dtype,
            device=activations_torch.device,
        )
        widened_activations[: activations_torch.shape[0]] = activations_torch
        activations_torch = widened_activations

    if cfg.is_swap_ab:
        M, N, K, L = out_hidden, declared_padded_tokens, in_hidden, num_experts
        # Shuffle weight rows for the 16x256b T2R epilogue pattern.
        shuffled_weights = shuffle_matrix(preprocessed_weights, cfg.epi_tile_m)
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
            shuffle_weight_rows=(not cfg.use_deepseek_fp8),
        )
        sf_b = sf_activations
    else:
        M, N, K, L = total_padded_tokens, out_hidden, in_hidden, num_experts
        kernel_a = activations_torch
        kernel_weights = preprocessed_weights
        if cfg.uses_shuffled_matrix:
            kernel_weights = shuffle_matrix(kernel_weights, cfg.epi_tile_m)
        kernel_b = _block_major_k_weight_tensor(kernel_weights, cfg)
        sf_a = sf_activations
        sf_b = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_weight_rows=(cfg.uses_shuffled_matrix and not cfg.use_deepseek_fp8),
        )

    logical_output_m, logical_output_n = _logical_output_shape(cfg, M, N, is_gated)
    plain_output_dtype = _plain_output_torch_dtype(cfg)
    c_pitch = int(c_row_stride) if c_row_stride else logical_output_m
    sf_c_m_offset = 0
    if c_row_stride:
        if not cfg.is_swap_ab:
            raise ValueError("c_row_stride reference checks cover swapAB only")
        if c_pitch < logical_output_m:
            raise ValueError(
                f"c_row_stride={c_pitch} is narrower than the output width "
                f"{logical_output_m}"
            )
        if c_pitch % 32:
            raise ValueError(
                f"c_row_stride={c_pitch} must be a multiple of 32 "
                "(the C layout applies cute.assume(stride, 32))"
            )
        if cfg.has_deepseek_fp8_c_scale:
            raise ValueError(
                "c_row_stride reference checks do not cover DeepSeek FP8 C scales"
            )
    if c_partition_offset:
        if not c_row_stride:
            raise ValueError("c_partition_offset requires c_row_stride")
        if c_partition_offset % 32:
            raise ValueError(
                f"c_partition_offset={c_partition_offset} must be a multiple of 32"
            )
        if cfg.has_output_block_scaling:
            if c_partition_offset % cfg.output_sf_block_size_c:
                raise ValueError(
                    f"block-scaled c_partition_offset={c_partition_offset} must be "
                    "a multiple of the SF block width "
                    f"{cfg.output_sf_block_size_c}"
                )
            sf_c_m_offset = c_partition_offset
        if c_partition_offset + logical_output_m > c_pitch:
            raise ValueError(
                f"partition at offset {c_partition_offset} with width "
                f"{logical_output_m} does not fit inside pitch {c_pitch}"
            )
    c_storage_elems = logical_output_n * c_pitch
    c_touched = None
    sf_c_touched = None
    if cfg.has_output_block_scaling:
        if output_guard_elements > 0:
            raise ValueError("output_guard_elements is not supported with FP4 C output")
        c_storage = torch.full(
            (_quantized_c_numel(cfg, c_pitch, logical_output_n),),
            _C_STRIDE_SENTINEL_U8 if c_row_stride else 0,
            dtype=torch.uint8,
            device=device,
        )
        c_torch = None
        sf_c_torch = torch.full(
            (
                _sf_c_numel(
                    logical_output_m,
                    logical_output_n,
                    cfg.sf_layout_c,
                    cfg.output_sf_block_size_c,
                    is_swap_ab=cfg.is_swap_ab,
                    pitch=c_pitch,
                ),
            ),
            _C_STRIDE_SENTINEL_U8 if c_row_stride else 0,
            dtype=torch.uint8,
            device=device,
        )
        if c_row_stride:
            m_idx = torch.arange(logical_output_m, device=device).view(-1, 1)
            n_idx = torch.arange(logical_output_n, device=device).view(1, -1)
            c_elem_idx = n_idx * c_pitch + m_idx
            c_storage_idx = (
                c_elem_idx if cfg.uses_mxfp8_output_quant else c_elem_idx // 2
            )
            c_storage_idx = c_storage_idx + (
                c_partition_offset
                if cfg.uses_mxfp8_output_quant
                else c_partition_offset // 2
            )
            c_touched = torch.zeros_like(c_storage, dtype=torch.bool)
            c_touched[c_storage_idx.reshape(-1)] = True
            c_storage[c_touched] = 0

            sf_idx = _sf_c_index_grid(
                output_m=logical_output_m,
                n=logical_output_n,
                sf_layout_c=cfg.sf_layout_c,
                sf_block_size=cfg.output_sf_block_size_c,
                is_swap_ab=cfg.is_swap_ab,
                device=device,
                pitch=c_pitch,
                m_offset=sf_c_m_offset,
            )
            sf_c_touched = torch.zeros_like(sf_c_torch, dtype=torch.bool)
            sf_c_touched[sf_idx.reshape(-1)] = True
            sf_c_torch[sf_c_touched] = 0
        output_guard_value = None
        c_guard = None
    elif cfg.uses_fp8_output:
        if output_guard_elements > 0:
            raise ValueError("output_guard_elements is not supported with FP8 C output")
        c_storage = torch.full(
            (c_storage_elems,),
            _C_STRIDE_SENTINEL_U8 if c_row_stride else _C_UNWRITTEN_POISON_E4M3_U8,
            dtype=torch.uint8,
            device=device,
        )
        if c_row_stride:
            _partition_view(
                c_storage[c_partition_offset:],
                logical_output_m,
                logical_output_n,
                c_pitch,
            ).fill_(_C_UNWRITTEN_POISON_E4M3_U8)
        c_torch = None
        if cfg.has_deepseek_fp8_c_scale:
            sf_c_torch = torch.zeros(
                (
                    _dsfp8_c_numel(
                        logical_output_m,
                        logical_output_n,
                        is_swap_ab=cfg.is_swap_ab,
                        block=cfg.deepseek_fp8_block_size,
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
        c_torch.fill_(_C_UNWRITTEN_POISON_F)
        c_guard = c_storage[logical_output_m * logical_output_n :]
        sf_c_torch = torch.zeros((1,), dtype=torch.uint8, device=device)
    else:
        output_guard_value = None
        c_guard = None
        if cfg.is_swap_ab:
            c_storage = torch.full(
                (c_storage_elems,),
                _C_STRIDE_SENTINEL_F if c_row_stride else _C_UNWRITTEN_POISON_F,
                dtype=plain_output_dtype,
                device=device,
            )
            c_torch = _partition_view(
                c_storage[c_partition_offset:],
                logical_output_m,
                logical_output_n,
                c_pitch,
            )
            if c_row_stride:
                c_torch.fill_(_C_UNWRITTEN_POISON_F)
        else:
            c_torch = torch.full(
                (logical_output_m, logical_output_n),
                _C_UNWRITTEN_POISON_F,
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
    if all_zero_input_data:
        # TRTLLM-gen's allZeros distribution also clears an enabled bias.
        _zero_input_data_tensors(bias_logical_torch, bias_torch)
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

    if cfg.has_output_block_scaling or cfg.uses_fp8_output:
        c_dtype = (
            cutlass.Float8E4M3FN
            if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
            else cutlass.Float4E2M1FN
        )
        c_dp = make_ptr(
            c_dtype,
            c_storage[
                (
                    c_partition_offset
                    if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
                    else c_partition_offset // 2
                ) :
            ].data_ptr(),
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
    if all_zero_input_data:
        # Dummy scale buffers stay at one; only scale factors enabled by the
        # kernel configuration are part of the input-data distribution.
        _zero_input_data_tensors(
            per_token_sf_a_torch if cfg.has_per_token_sf_a else None,
            per_token_sf_b_torch if cfg.has_per_token_sf_b else None,
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
        [token_layout.num_token_cgas, 0, 0], dtype=torch.int32, device=device
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
        "c_row_stride": c_row_stride,
        "sf_c_m_offset": sf_c_m_offset,
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
                if (cfg.has_output_block_scaling or cfg.uses_fp8_output)
                else c_torch
            )
            if first_output is None:
                first_output = output_tensor.detach().clone()
                if cfg.has_output_block_scaling or cfg.has_deepseek_fp8_c_scale:
                    first_scale = sf_c_torch.detach().clone()
            else:
                outputs_deterministic &= _same_launch_output(
                    first_output, output_tensor
                )
                if first_scale is not None:
                    scales_deterministic &= _same_launch_output(first_scale, sf_c_torch)

    runtime_cga_shapes_ok = True
    if require_mixed_cga_runtime:
        preferred_cgas, fallback_cgas = num_non_exiting_ctas[1:].detach().cpu().tolist()
        print(
            "Runtime physical CGAs: "
            f"preferred={preferred_cgas}, fallback={fallback_cgas}"
        )
        observed_both_shapes = preferred_cgas > 0 and fallback_cgas > 0
        if not observed_both_shapes:
            print("FAIL: launch did not exercise both preferred and fallback CGAs")

        problem_m_tiles = (M + cfg.tile_m - 1) // cfg.tile_m
        problem_n_tiles = (N + cfg.tile_n - 1) // cfg.tile_n
        if cfg.use_early_exit:
            if cfg.is_swap_ab:
                problem_n_tiles = launch_early_exit_max_token_ctas
            else:
                problem_m_tiles = launch_early_exit_max_token_ctas
        logical_work_cgas = (
            (problem_m_tiles + cfg.umma_cta_count - 1) // cfg.umma_cta_count
        ) * problem_n_tiles
        preferred_size = math.prod(cfg.preferred_cluster_shape)
        fallback_size = math.prod(cfg.fallback_cluster_shape)
        physical_grid_ctas = (
            preferred_cgas * preferred_size + fallback_cgas * fallback_size
        )
        (preferred_work, preferred_wave), (fallback_work, fallback_wave) = (
            _mixed_cga_wave_counts(
                logical_work_cgas=logical_work_cgas,
                physical_grid_ctas=physical_grid_ctas,
                umma_cta_count=cfg.umma_cta_count,
                preferred_shape=cfg.preferred_cluster_shape,
                fallback_shape=cfg.fallback_cluster_shape,
            )
        )
        print(
            "Persistent physical-CGA work/wave counts: "
            f"preferred={preferred_work}/{preferred_wave}, "
            f"fallback={fallback_work}/{fallback_wave}"
        )
        has_multiwave_work = (
            preferred_work > preferred_wave and fallback_work > fallback_wave
        )
        if not has_multiwave_work:
            print(
                "FAIL: logical work does not exceed one persistent-grid wave "
                "for both physical CGA shapes"
            )
        runtime_cga_shapes_ok = observed_both_shapes and has_multiwave_work
    # Unload the per-check CUDA library at a deterministic point after the
    # launch is synchronized. Letting Python GC do this later can overlap
    # library unload with the next clustered persistent compile/launch.
    del compiled_fn
    gc.collect()

    if cfg.has_output_block_scaling:
        c_storage_offset = (
            c_partition_offset
            if cfg.uses_mxfp8_output_quant
            else c_partition_offset // 2
        )
        if cfg.uses_mxfp8_output_quant:
            c_logical = _dequantize_mxfp8_output(
                c_storage[c_storage_offset:],
                sf_c_torch,
                output_m=logical_output_m,
                n=logical_output_n,
                is_swap_ab=cfg.is_swap_ab,
                sf_layout_c=cfg.sf_layout_c,
                pitch=c_pitch,
                sf_c_m_offset=sf_c_m_offset,
            )
        else:
            c_logical = _dequantize_fp4_output(
                c_storage[c_storage_offset:],
                sf_c_torch,
                output_m=logical_output_m,
                n=logical_output_n,
                is_swap_ab=cfg.is_swap_ab,
                sf_layout_c=cfg.sf_layout_c,
                uses_mxfp4=cfg.uses_mxfp4_output_quant,
                pitch=c_pitch,
                sf_c_m_offset=sf_c_m_offset,
            )
    elif cfg.has_deepseek_fp8_c_scale:
        c_logical = _dequantize_dsfp8_output(
            c_storage,
            sf_c_torch,
            output_m=logical_output_m,
            n=logical_output_n,
            is_swap_ab=cfg.is_swap_ab,
            scale_stride=total_padded_tokens if cfg.is_swap_ab else None,
            block=cfg.deepseek_fp8_block_size,
        )
    elif cfg.uses_fp8_output:
        c_logical = _decode_fp8_output(
            c_storage[c_partition_offset:],
            output_m=logical_output_m,
            n=logical_output_n,
            is_swap_ab=cfg.is_swap_ab,
            pitch=c_pitch,
        )
    else:
        c_logical = c_torch

    print(f"Output shape: {c_logical.shape}")
    print(
        f"Output sample (first 8 elements): {c_logical[0, : min(8, logical_output_n)]}"
    )
    # C is poisoned with NaN, and the inactive padded-token tail is legitimately
    # never written, so summarize over the elements that carry a stored value.
    c_finite = c_logical[torch.isfinite(c_logical)]
    print(
        f"Output max: {c_finite.abs().max().item() if c_finite.numel() else 'none'}"
        f" (over {c_finite.numel()}/{c_logical.numel()} stored elements)"
    )

    def _return_diagnostics(
        passed: bool,
        *,
        max_abs: float | None = None,
        mean_abs: float | None = None,
        max_rel: float | None = None,
        threshold: float | None = None,
        nan_count: int = 0,
        inf_count: int = 0,
        error: str | None = None,
        reason: str | None = None,
    ):
        diagnostics = ReferenceCheckDiagnostics(
            passed=passed,
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=max_rel,
            threshold=threshold,
            nan_count=nan_count,
            inf_count=inf_count,
            error=error,
            reason=reason,
        )
        if return_output:
            return diagnostics, c_logical.detach().clone()
        return diagnostics

    if c_row_stride:
        if c_touched is None:
            c_touched = torch.zeros_like(c_storage, dtype=torch.bool)
            _partition_view(
                c_touched[c_partition_offset:],
                logical_output_m,
                logical_output_n,
                c_pitch,
            ).fill_(True)
        expect = (
            _C_STRIDE_SENTINEL_U8
            if cfg.has_output_block_scaling or cfg.uses_fp8_output
            else _C_STRIDE_SENTINEL_F
        )
        if (c_storage.reshape(-1)[~c_touched] != expect).any().item():
            print("Strided output wrote outside its partition")
            return _return_diagnostics(
                False, reason="strided output wrote outside its partition"
            )
        if (
            sf_c_touched is not None
            and (sf_c_torch.reshape(-1)[~sf_c_touched] != _C_STRIDE_SENTINEL_U8)
            .any()
            .item()
        ):
            print("Strided SF output wrote outside its partition")
            return _return_diagnostics(
                False, reason="strided scale output wrote outside its partition"
            )

    if output_guard_elements > 0:
        guard_changed = (c_guard != output_guard_value).any().item()
        print(f"Output guard changed: {guard_changed}")
        if guard_changed:
            return _return_diagnostics(False, reason="output guard was modified")

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
        # A declared capacity wider than the routed count leaves a tail the
        # kernel never writes, so drop it before selecting the routed rows.
        if output.shape[token_dim] > active_route_mask.numel():
            output = output.narrow(token_dim, 0, active_route_mask.numel())
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
    nan_count = int(torch.isnan(c_active).sum().item())
    inf_count = int(torch.isinf(c_active).sum().item())
    has_nan = nan_count > 0
    has_inf = inf_count > 0

    if not outputs_deterministic or not scales_deterministic:
        print(
            "FAIL: repeated launches are nondeterministic "
            f"(output={outputs_deterministic}, scale={scales_deterministic})"
        )
        return _return_diagnostics(
            False,
            nan_count=nan_count,
            inf_count=inf_count,
            reason=(
                "repeated launches were nondeterministic "
                f"(output={outputs_deterministic}, scale={scales_deterministic})"
            ),
        )

    if not runtime_cga_shapes_ok:
        return _return_diagnostics(
            False,
            nan_count=nan_count,
            inf_count=inf_count,
            reason="runtime CGA shape validation failed",
        )

    if has_nan or has_inf:
        print(f"FAIL: output has NaN={has_nan}, Inf={has_inf}")
        bad_mask = torch.isnan(c_active) | torch.isinf(c_active)
        bad_count = bad_mask.sum().item()
        bad_indices = bad_mask.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
        print(f"Bad value count: {bad_count}, first indices: {bad_indices}")
        if has_nan:
            # C is poisoned with NaN before the launch, so a NaN here is either
            # a computed NaN or an element the epilogue never stored. Report the
            # M extents affected: a store that drops part of a tile leaves a
            # regular stride of M spans, which a computed NaN would not.
            m_dim = 0 if cfg.is_swap_ab else 1
            nan_m = torch.nonzero(torch.isnan(c_active).any(dim=1 - m_dim)).flatten()
            print(
                f"NaN M extents: {nan_m.numel()}/{c_active.shape[m_dim]}"
                + (
                    f", first={int(nan_m[0])} last={int(nan_m[-1])}"
                    if nan_m.numel()
                    else ""
                )
            )
        return _return_diagnostics(
            False,
            nan_count=nan_count,
            inf_count=inf_count,
            reason="output contains NaN or Inf values",
        )

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
            dtype_kind=cfg.dtype_a,
            logical_cols=in_hidden,
            layout=weight_sf_layout,
            sf_vec_size=cfg.input_sf_block_size_a,
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
        weight_dtype_kind = cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b
        activation_dtype_kind = cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a
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
        if cfg.use_deepseek_fp8:
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
        weight_dtype_kind = cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b
        activation_dtype_kind = cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a
        decoded_activations = _decode_quantized_storage(
            activation_compact_torch,
            dtype_kind=activation_dtype_kind,
            logical_cols=in_hidden,
        )
        ref_gemm = torch.zeros((M, N), dtype=torch.float32, device=device)
        routed_rows_by_expert = {}
        for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
            if expert_idx < 0:
                continue
            token_idx = token_layout.expanded_to_token[expanded_idx]
            routed_rows_by_expert.setdefault(expert_idx, []).append(
                (expanded_idx, token_idx)
            )

        for expert_idx, routed_rows in routed_rows_by_expert.items():
            weights_ref = _dequantize_block_scaled_expert(
                weights_torch,
                sf_weights,
                expert_idx=expert_idx,
                dtype_kind=weight_dtype_kind,
                logical_cols=in_hidden,
                layout=weight_sf_layout,
                sf_vec_size=cfg.input_sf_block_size_a,
                sf_torch_dtype=_sf_torch_dtype(cfg),
            )
            expanded_indices = torch.tensor(
                [expanded_idx for expanded_idx, _ in routed_rows],
                dtype=torch.long,
                device=device,
            )
            activation_rows = torch.stack(
                [
                    _dequantize_block_scaled_activation_row(
                        decoded_activations,
                        sf_activations,
                        token_idx=token_idx,
                        expanded_idx=expanded_idx,
                        cfg=cfg,
                        activation_sf_layout=activation_sf_layout,
                        sf_k=sf_k,
                    )
                    for expanded_idx, token_idx in routed_rows
                ]
            )
            if cfg.is_swap_ab:
                value = weights_ref @ activation_rows.T
                if cfg.has_per_token_sf_b:
                    sf_indices = torch.tensor(
                        [
                            _per_token_sf_b_ref_index(expanded_idx, token_idx)
                            for expanded_idx, token_idx in routed_rows
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    value = value * per_token_sf_b_torch[sf_indices].float().view(1, -1)
                ref_gemm[:, expanded_indices] = value
            else:
                value = activation_rows @ weights_ref.T
                if cfg.has_per_token_sf_b:
                    value = value * per_token_sf_b_torch[:N].float()
                ref_gemm[expanded_indices, :] = value
            del weights_ref

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
    is_silu = cfg.act_kind == int(ActKind.SILU)

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
        elif not cfg.has_output_block_scaling:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical
    elif is_relu2 or is_silu:
        ref_torch = ref_gemm.float()
        if cfg.uses_global_scales:
            ref_torch = ref_torch * _expert_scale_like(
                ref_torch.shape, scale_gate_torch
            )
        if is_relu2:
            ref_torch = torch.clamp(ref_torch, min=0).pow(2)
        else:
            ref_torch = torch.nn.functional.silu(ref_torch)
        ref_torch = ref_torch * _expert_scale_like(ref_torch.shape, scale_c_torch)
        if cfg.uses_fp8_output:
            ref_torch = ref_torch.to(torch.float8_e4m3fn).float()
        elif not cfg.has_output_block_scaling:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical
    else:
        ref_torch = ref_gemm.float() * _expert_scale_like(ref_gemm.shape, scale_c_torch)
        if cfg.uses_fp8_output:
            ref_torch = ref_torch.to(torch.float8_e4m3fn).float()
        elif not cfg.has_output_block_scaling:
            ref_torch = ref_torch.to(plain_output_dtype)
        c_compare = c_logical

    if cfg.has_deepseek_fp8_c_scale:
        # The epilogue packs scale blocks at the routed padded token stride, so
        # a wider declared capacity leaves an unwritten tail past that prefix.
        sf_c_ref_source = ref_torch.float()
        if cfg.is_swap_ab and declared_padded_tokens > total_padded_tokens:
            sf_c_ref_source = sf_c_ref_source[:, :total_padded_tokens]
        expected_sf_c = _reference_dsfp8_c_scales(
            sf_c_ref_source,
            is_swap_ab=cfg.is_swap_ab,
            block=cfg.deepseek_fp8_block_size,
        )
        sf_c_diff = (sf_c_torch.float()[: expected_sf_c.numel()] - expected_sf_c).abs()
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
            return _return_diagnostics(
                False,
                max_abs=sf_c_max_abs,
                threshold=sf_c_threshold,
                nan_count=nan_count,
                inf_count=inf_count,
                reason="DeepSeek output scale comparison failed",
            )

    c_compare_active = _active_routes(c_compare)
    ref_active = _active_routes(ref_torch)
    diff = (c_compare_active.float() - ref_active.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    nonzero_ref = ref_active.float().abs() > 0
    max_rel = None
    if nonzero_ref.any().item():
        max_rel = (
            (diff[nonzero_ref] / ref_active.float().abs()[nonzero_ref]).max().item()
        )
    ref_max = ref_active.float().abs().max().item()
    out_max = c_compare_active.float().abs().max().item()
    print(
        f"Reference comparison: max_abs={max_abs:.6f}, "
        f"mean_abs={mean_abs:.6f}, ref_max={ref_max:.6f}"
    )
    if cfg.has_cast_a and cfg.is_swap_ab:
        # Operates on the actual output rows: gated activations halve the logical
        # output (M//2), so index by logical_output_m, not the full M.
        shuffled_ref = shuffle_matrix(ref_torch, cfg.epi_tile_m)
        unshuffle_idx = torch.empty(
            (logical_output_m,), dtype=torch.long, device=device
        )
        indices = shuffled_matrix_row_indices(cfg.epi_tile_m)
        block_size = len(indices)
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
    elif cfg.has_output_block_scaling:
        threshold = 0.75 + 0.35 * ref_max
    else:
        threshold = 0.75 + 0.05 * ref_max

    if out_max == 0.0 and ref_max > 1e-5:
        print(
            "Reference check: FAILED "
            f"(all-zero output with non-zero reference max {ref_max:.6f})"
        )
        return _return_diagnostics(
            False,
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=max_rel,
            threshold=threshold,
            nan_count=nan_count,
            inf_count=inf_count,
            reason="all-zero output with non-zero reference",
        )

    if max_abs <= threshold:
        print(f"Reference check: PASSED (max_abs={max_abs:.6f} <= {threshold:.6f})")
        return _return_diagnostics(
            True,
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=max_rel,
            threshold=threshold,
            nan_count=nan_count,
            inf_count=inf_count,
        )
    max_flat = diff.argmax()
    max_coord = torch.unravel_index(max_flat, diff.shape)
    max_coord_host = tuple(int(v.detach().cpu().item()) for v in max_coord)
    print(f"Reference max-diff index: {max_coord_host}")
    print(f"Reference check: FAILED (max_abs={max_abs:.6f} > {threshold:.6f})")
    return _return_diagnostics(
        False,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        threshold=threshold,
        nan_count=nan_count,
        inf_count=inf_count,
        reason="maximum absolute error exceeded threshold",
    )


def _build_launch_io(
    *,
    num_experts,
    num_tokens,
    top_k,
    routing="balanced_round_robin",
    routing_seed=42,
    problem_n=None,
    problem_k=None,
    seed=42,
    data_distribution="default",
    scale_c_value=None,
    scale_gate_value=1.0,
    gemm1_alpha_value=None,
    gemm1_beta_value=None,
    gemm1_clamp_limit_value=2.0,
    early_exit_max_token_ctas=0,
    input_device: str = "cuda",
    weight_cache=None,
    **cfg_overrides,
):
    """Build all device tensors + pointers for a single kernel launch.

    Shared by :func:`benchmark` and the perf-harness interface
    (:func:`compile` / :func:`prepare_tensors` / :func:`run`). Returns a dict
    of device pointers, the ``(M, N, K, L, num_tokens)`` problem shape, the
    resolved config, and a ``_keepalive`` list that pins the backing torch
    tensors alive for as long as the pointers are dereferenced.

    """
    cfg_overrides = _resolve_runtime_sm_arch(cfg_overrides)
    launch_early_exit_max_token_ctas = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    all_zero_input_data = _uses_all_zero_input_data(data_distribution)
    cfg = make_config(**cfg_overrides)

    torch.manual_seed(seed)
    device = input_device
    from cutlass.cute.runtime import make_ptr

    # Keep shape semantics aligned with reference_check and generated batch-N
    # kernels: problem_n names the output hidden dimension for both swapAB and
    # non-swap paths.
    default_out_hidden = cfg.tile_m if cfg.is_swap_ab else cfg.tile_n
    out_hidden = problem_n if problem_n is not None else default_out_hidden
    in_hidden = problem_k if problem_k is not None else cfg.tile_k
    cfg = _runtime_config(cfg, in_hidden)
    scale_c_value = _resolve_scale_c(
        scale_c_value,
        cfg=cfg,
        in_hidden=in_hidden,
        scale_gate_value=scale_gate_value,
    )
    _validate_swap_ab_out_hidden(cfg, out_hidden)

    token_tile_size = cfg.token_tile_size
    cluster_dim_in_token = cfg.token_cluster_dim
    token_layout = _make_token_layout(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
        routing=routing,
        routing_seed=routing_seed,
    )
    launch_early_exit_max_token_ctas = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        launch_early_exit_max_token_ctas,
    )
    total_padded_tokens = token_layout.total_padded_tokens
    if total_padded_tokens == 0:
        raise ValueError(
            "routing produced no assignments for the selected local expert slice"
        )

    weight_cache_key = None
    cached_weight_io = None
    if weight_cache is not None and not cfg.use_deepseek_fp8:
        weight_sf_layout_key = (
            None
            if cfg.is_bf16_mma or cfg.is_fp8_mma
            else (cfg.sf_layout_a if cfg.is_swap_ab else cfg.sf_layout_b)
        )
        weight_cache_key = (
            "prims-ts-launch-weight-v1",
            str(input_device),
            int(num_experts),
            int(out_hidden),
            int(in_hidden),
            int(seed),
            str(data_distribution),
            int(cfg.dtype_a),
            int(cfg.dtype_b),
            bool(cfg.is_bf16_mma),
            bool(cfg.has_cast_a),
            bool(cfg.is_fp8_mma),
            bool(cfg.is_swap_ab),
            int(cfg.act_kind),
            int(cfg.epi_tile_m),
            bool(cfg.uses_shuffled_matrix),
            int(getattr(cfg, "weight_layout", 0)),
            bool(cfg.uses_block_major_k_weight),
            int(cfg.block_major_k_bytes),
            weight_sf_layout_key,
            int(cfg.input_sf_block_size_a),
        )
        cached_weight_io = weight_cache.get(weight_cache_key)
        move_to_end = getattr(weight_cache, "move_to_end", None)
        if cached_weight_io is not None and move_to_end is not None:
            move_to_end(weight_cache_key)

    # Create domain tensors. Routed kernels consume compact activation rows;
    # non-routed kernels consume the expanded/padded expert-token layout.
    weights_torch = None
    sf_weights = None
    if cfg.is_bf16_mma:
        if cached_weight_io is None:
            weights_torch = _kaiming_uniform_tensor(
                (num_experts, out_hidden, in_hidden),
                fan_in=in_hidden,
                dtype=torch.bfloat16,
                device=device,
            )
        if weight_cache_key is not None:
            torch.manual_seed(seed + 1)
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
        sf_activations = None
    elif cfg.has_cast_a:
        weight_sf_layout = cfg.sf_layout_a
        sf_k = in_hidden // cfg.input_sf_block_size_a
        if cached_weight_io is None:
            weights_torch = _random_quantized_tensor(
                (num_experts, out_hidden, in_hidden),
                dtype_kind=cfg.dtype_a,
                fan_in=in_hidden,
                device=device,
            )
            if weight_cache_key is not None:
                sf_weights = _create_sf_tensor_mma_layout(
                    num_experts=num_experts,
                    mn=out_hidden,
                    sf_k=sf_k,
                    layout=weight_sf_layout,
                    device=device,
                    cfg=cfg,
                )
        if weight_cache_key is not None:
            torch.manual_seed(seed + 1)
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
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = cutlass.BFloat16
        if sf_weights is None and cached_weight_io is None:
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
        if cached_weight_io is None:
            weights_torch = _random_quantized_tensor(
                (num_experts, out_hidden, in_hidden),
                dtype_kind=cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b,
                fan_in=in_hidden,
                device=device,
            )
        if weight_cache_key is not None:
            torch.manual_seed(seed + 1)
        activation_compact_torch = _random_quantized_tensor(
            (num_tokens, in_hidden),
            dtype_kind=cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b)
        weight_sf_layout = None
        activation_sf_layout = None
        sf_activations = None
        if cfg.use_deepseek_fp8:
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
        sf_k = in_hidden // cfg.input_sf_block_size_a
        activation_sf_layout = cfg.sf_layout_b if cfg.is_swap_ab else cfg.sf_layout_a
        weight_sf_layout = cfg.sf_layout_a if cfg.is_swap_ab else cfg.sf_layout_b
        if cached_weight_io is None:
            weights_torch = _random_quantized_tensor(
                (num_experts, out_hidden, in_hidden),
                dtype_kind=cfg.dtype_a if cfg.is_swap_ab else cfg.dtype_b,
                fan_in=in_hidden,
                device=device,
            )
            if weight_cache_key is not None:
                sf_weights = _create_sf_tensor_mma_layout(
                    num_experts=num_experts,
                    mn=out_hidden,
                    sf_k=sf_k,
                    layout=weight_sf_layout,
                    device=device,
                    cfg=cfg,
                )
        if weight_cache_key is not None:
            torch.manual_seed(seed + 1)
        activation_compact_torch = _random_quantized_tensor(
            (num_tokens, in_hidden),
            dtype_kind=cfg.dtype_b if cfg.is_swap_ab else cfg.dtype_a,
            fan_in=in_hidden,
            device=device,
        )
        activations_torch = (
            activation_compact_torch
            if cfg.has_routed_act
            else _expand_fp4_activations(activation_compact_torch, token_layout)
        )
        a_dtype = _cutlass_data_dtype(cfg.dtype_a)
        b_dtype = _cutlass_data_dtype(cfg.dtype_b)
        if cfg.has_routed_sfs:
            sf_activations = _create_sf_linear(
                num_tokens=num_tokens,
                sf_k_dim=sf_k,
                device=device,
                cfg=cfg,
            )
            if sf_weights is None and cached_weight_io is None:
                sf_weights = _create_sf_tensor_mma_layout(
                    num_experts=num_experts,
                    mn=out_hidden,
                    sf_k=sf_k,
                    layout=weight_sf_layout,
                    device=device,
                    cfg=cfg,
                )
        else:
            if sf_weights is None and cached_weight_io is None:
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

    if all_zero_input_data:
        # This is the perf-path counterpart of reference_check's distribution
        # handling. It covers packed FP4/FP8 and plain BF16 data plus all
        # operand scale factors, while leaving token routing metadata intact.
        _zero_input_data_tensors(
            weights_torch if cached_weight_io is None else None,
            activation_compact_torch,
            activations_torch,
            sf_weights if cached_weight_io is None else None,
            sf_activations,
        )

    is_gated = _is_gated_act_kind(cfg.act_kind)
    if cached_weight_io is None:
        assert weights_torch is not None
        preprocessed_weights = weights_torch
        if is_gated:
            preprocessed_weights = reorder_rows_for_gated_act(weights_torch)
        kernel_weights = preprocessed_weights
        shuffle_weight_rows = cfg.is_swap_ab or cfg.uses_shuffled_matrix
        if shuffle_weight_rows:
            kernel_weights = shuffle_matrix(kernel_weights, cfg.epi_tile_m)
        kernel_weights = _block_major_k_weight_tensor(kernel_weights, cfg)
        kernel_weight_sf = _kernel_weight_sf(
            sf_weights,
            cfg=cfg,
            out_hidden=out_hidden,
            in_hidden=in_hidden,
            num_experts=num_experts,
            is_gated=is_gated,
            weight_sf_layout=weight_sf_layout,
            shuffle_weight_rows=(shuffle_weight_rows and not cfg.use_deepseek_fp8),
        )
    else:
        kernel_weights, kernel_weight_sf = cached_weight_io

    if cfg.is_swap_ab:
        M, N, K, L = out_hidden, total_padded_tokens, in_hidden, num_experts
        kernel_a = kernel_weights
        kernel_b = activations_torch
        sf_a = kernel_weight_sf
        sf_b = sf_activations
    else:
        M, N, K, L = total_padded_tokens, out_hidden, in_hidden, num_experts
        kernel_a = activations_torch
        kernel_b = kernel_weights
        sf_a = sf_activations
        sf_b = kernel_weight_sf

    logical_output_m, logical_output_n = _logical_output_shape(cfg, M, N, is_gated)
    plain_output_dtype = _plain_output_torch_dtype(cfg)
    c_storage = None
    if cfg.has_output_block_scaling:
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
                    is_swap_ab=cfg.is_swap_ab,
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
                        block=cfg.deepseek_fp8_block_size,
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

    bias_torch = torch.zeros((L, M), dtype=torch.float32, device=device)
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
    per_token_dtype = _per_token_torch_dtype(cfg)
    per_token_count = max(num_tokens, total_padded_tokens, M, N)
    per_token_sf_a_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    per_token_sf_b_torch = torch.ones(
        (per_token_count,), dtype=per_token_dtype, device=device
    )
    if all_zero_input_data:
        _zero_input_data_tensors(
            per_token_sf_a_torch if cfg.has_per_token_sf_a else None,
            per_token_sf_b_torch if cfg.has_per_token_sf_b else None,
        )
    num_non_exiting_ctas = torch.tensor(
        [token_layout.num_token_cgas, 0, 0], dtype=torch.int32, device=device
    )
    total_num_padded_tokens_torch = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=device
    )

    # Finish random generation and layout preprocessing on the host, then issue
    # only HtoD copies before the kernel.
    move_memo: dict[int, torch.Tensor] = {}

    def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        original_stride = tensor.stride()
        moved = _move_to_cuda_memo(move_memo, tensor)
        assert moved is not None and moved.is_cuda
        assert moved.stride() == original_stride
        return moved

    kernel_a = _move(kernel_a)
    kernel_b = _move(kernel_b)
    sf_a = _move(sf_a)
    sf_b = _move(sf_b)
    c_storage = _move(c_storage)
    sf_c_torch = _move(sf_c_torch)
    tile_idx = _move(tile_idx)
    mn_limit = _move(mn_limit)
    route_map = _move(route_map)
    act_torch = _move(act_torch)
    bias_torch = _move(bias_torch)
    scale_c_torch = _move(scale_c_torch)
    scale_gate_torch = _move(scale_gate_torch)
    gemm1_alpha_torch = _move(gemm1_alpha_torch)
    gemm1_beta_torch = _move(gemm1_beta_torch)
    gemm1_clamp_limit_torch = _move(gemm1_clamp_limit_torch)
    per_token_sf_a_torch = _move(per_token_sf_a_torch)
    per_token_sf_b_torch = _move(per_token_sf_b_torch)
    num_non_exiting_ctas = _move(num_non_exiting_ctas)
    total_num_padded_tokens_torch = _move(total_num_padded_tokens_torch)
    if c_storage is not None and c_torch is not None:
        c_torch = torch.as_strided(
            c_storage,
            (logical_output_m, logical_output_n),
            (1, logical_output_m),
        )
    else:
        c_torch = _move(c_torch)

    if weight_cache_key is not None and cached_weight_io is None:
        kernel_weights = kernel_a if cfg.is_swap_ab else kernel_b
        kernel_weight_sf = sf_a if cfg.is_swap_ab else sf_b
        weight_cache[weight_cache_key] = (kernel_weights, kernel_weight_sf)
        while len(weight_cache) > 2:
            weight_cache.pop(next(iter(weight_cache)))

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
    if cfg.has_output_block_scaling or cfg.uses_fp8_output:
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
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
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
        c_storage if (cfg.has_output_block_scaling or cfg.uses_fp8_output) else c_torch
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
        "requested_num_tokens": num_tokens,
        "total_padded_tokens": total_padded_tokens,
        "token_layout": token_layout,
        "routing": routing,
        "routing_seed": routing_seed,
        "resolved_scale_c": scale_c_value,
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
        _c_row_stride(io),
        int(io.get("sf_c_m_offset", 0)),
        io["cfg"],
        stream,
    )


def _c_row_stride(io: dict) -> int:
    """Row pitch of the C buffer in elements; defaults to this launch's width."""
    v = io.get("c_row_stride")
    if v:
        return int(v)
    m = int(io["shape"][0])
    cfg = io["cfg"]
    if cfg.is_swap_ab and cfg.has_gated_epilogue:
        return m // 2
    return m


def _launch_arg_tuple(io: dict, stream) -> tuple:
    """Positional args for a runtime launch of the compiled kernel.

    The single launch-argument order, shared by :func:`run` and the
    ``JitArguments`` workspaces in :func:`benchmark`: same as
    :func:`_compile_arg_tuple` minus the constexpr ``cfg`` operand.
    """
    problem_m, problem_n = (int(value) for value in io["shape"][:2])
    validate_launch_grid(
        io["cfg"],
        problem_m=problem_m,
        problem_n=problem_n,
        early_exit_max_token_ctas=int(io["launch_early_exit_max_token_ctas"]),
    )
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
        _c_row_stride(io),
        int(io.get("sf_c_m_offset", 0)),
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
    routed_tokens = num_tokens * top_k
    return 2.0 * routed_tokens * n * k


def benchmark_compiled(
    compiled_fn,
    io: dict,
    stream,
    *,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    num_rotated_buffers: int | None = None,
    cuda_profiler_range: bool = False,
    use_cuda_graphs: bool = True,
    host_input_gen: bool = False,
) -> float:
    """Time an already-compiled batched GEMM against one launch IO.

    The generated workspace set rotates only C and SF-C output buffers. Inputs,
    weights, scale factors, and routing data remain shared and can stay hot in
    cache; the underlying helper flushes L2 once before its replay loop, not
    before every iteration. Compilation is intentionally outside this helper
    so sweep tools can compile once and measure many runtime shapes that share
    the same compile signature.
    """
    import cutlass.cute.testing as testing
    from cutlass.cute.runtime import make_ptr

    cfg = io["cfg"]
    logical_output_m = io["logical_output_m"]
    logical_output_n = io["logical_output_n"]
    plain_output_dtype = io["plain_output_dtype"]
    sf_c_dp = io["sf_c_dp"]
    sf_c_torch = io["sf_c_torch"]
    num_experts = io["L"]
    num_tokens = io["num_tokens"]
    out_hidden, in_hidden = io["out_hidden"], io["in_hidden"]

    # Pin the domain tensors alive for the whole benchmark (pointers alias them).
    _io_keepalive = io["_keepalive"]

    def generate_tensors():
        output_factory = torch.empty if host_input_gen else torch.zeros
        if cfg.has_output_block_scaling or cfg.uses_fp8_output:
            c_buf = output_factory(
                (
                    _quantized_c_numel(cfg, logical_output_m, logical_output_n)
                    if cfg.has_output_block_scaling
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
            if host_input_gen:
                sf_c_buf = torch.empty_like(sf_c_torch)
                sf_c_buf.copy_(sf_c_torch)
            else:
                sf_c_buf = torch.zeros_like(sf_c_torch)
            sf_c_ptr = make_ptr(
                _sf_c_cutlass_dtype(cfg),
                sf_c_buf.data_ptr(),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
        else:
            c_buf = output_factory(
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
            *_launch_arg_tuple({**io, "c0_dp": c_ptr, "sf_c_dp": sf_c_ptr}, stream)
        )
        args.add_to_scope([c_buf, sf_c_buf])
        return args

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
            if (cfg.has_output_block_scaling or cfg.uses_fp8_output)
            else logical_output_m * logical_output_n * 2
        )
        + (
            _sf_c_numel(
                logical_output_m,
                logical_output_n,
                cfg.sf_layout_c,
                cfg.output_sf_block_size_c,
                is_swap_ab=cfg.is_swap_ab,
            )
            if cfg.has_output_block_scaling
            else 0
        )
    )
    workspace_count = testing.get_workspace_count(
        one_workspace_bytes,
        warmup_iters,
        bench_iters,
    )
    if num_rotated_buffers is not None:
        workspace_count = (
            1 if num_rotated_buffers <= 0 else min(workspace_count, num_rotated_buffers)
        )

    if cuda_profiler_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    try:
        return float(
            testing.benchmark(
                compiled_fn,
                workspace_generator=generate_tensors,
                workspace_count=workspace_count,
                stream=stream,
                warmup_iterations=warmup_iters,
                iterations=bench_iters,
                use_cuda_graphs=use_cuda_graphs,
            )
        )
    finally:
        if cuda_profiler_range:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        # Keep aliases live through the final synchronization in the timer.
        del _io_keepalive


def benchmark(
    *,
    num_experts,
    num_tokens,
    top_k,
    routing="balanced_round_robin",
    routing_seed=42,
    problem_n=None,
    problem_k=None,
    seed=42,
    data_distribution="default",
    scale_c_value=None,
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
    input_device: str = "cuda",
    **cfg_overrides,
):
    """Benchmark with CUDA graphs and report logical GEMM throughput.

    Effective bandwidth counts routed activations, the weights of experts that
    receive at least one token, and output once, excluding padding, scale
    factors, and auxiliary tensors.
    """
    io = _build_launch_io(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        routing=routing,
        routing_seed=routing_seed,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=seed,
        data_distribution=data_distribution,
        scale_c_value=scale_c_value,
        scale_gate_value=scale_gate_value,
        gemm1_alpha_value=gemm1_alpha_value,
        gemm1_beta_value=gemm1_beta_value,
        gemm1_clamp_limit_value=gemm1_clamp_limit_value,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
        input_device=input_device,
        **cfg_overrides,
    )
    cfg = io["cfg"]
    M, N, K, L = io["M"], io["N"], io["K"], io["L"]
    num_tokens = io["num_tokens"]
    total_padded_tokens = io["total_padded_tokens"]
    out_hidden, in_hidden = io["out_hidden"], io["in_hidden"]
    # Use a non-default stream for CUDA graph support (same as FMHA benchmark)
    import cuda.bindings.driver as cuda_drv

    _torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(_torch_stream.cuda_stream)

    # Compile with the non-default stream
    print(
        f"Benchmark: tokens={num_tokens}, padded_tokens={total_padded_tokens}, "
        f"experts={num_experts}, top_k={top_k}, M={M}, N={N}, K={K}, L={L}, "
        f"routing={routing}, routing_seed={routing_seed}, "
        f"dist={data_distribution}"
    )
    print(
        f"  tile=({cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}), "
        f"scheduler={'persistent' if cfg.is_persistent else 'static'}, "
        f"swap_ab={cfg.is_swap_ab}"
    )
    compiled_fn = _compile_for_launch(io, stream)
    exec_time = benchmark_compiled(
        compiled_fn,
        io,
        stream,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        num_rotated_buffers=num_rotated_buffers,
        cuda_profiler_range=cuda_profiler_range,
        use_cuda_graphs=use_cuda_graphs,
        host_input_gen=input_device == "cpu",
    )
    # Match the reference-check path: clustered persistent kernels are sensitive
    # to compiled CUDA library lifetime if unload overlaps the next compile or
    # launch in the same process.
    del compiled_fn
    gc.collect()

    print(f"  Average kernel time: {exec_time:.1f} µs ({exec_time / 1000:.4f} ms)")
    routed_tokens = sum(io["token_layout"].tokens_per_expert)
    flops = 2.0 * routed_tokens * out_hidden * in_hidden
    # Only experts selected by this concrete routing layout read weights.
    active_experts = sum(
        token_count > 0 for token_count in io["token_layout"].tokens_per_expert
    )
    activation_bits = cfg.dtype_b_bits if cfg.is_swap_ab else cfg.dtype_a_bits
    weight_bits = cfg.dtype_a_bits if cfg.is_swap_ab else cfg.dtype_b_bits
    output_width = out_hidden // 2 if cfg.has_gated_epilogue else out_hidden
    # Logical A/B reads plus C writes, independent of tile padding and routing
    # reuse. This is an effective rate, not measured device-memory traffic.
    io_bytes = (
        routed_tokens * in_hidden * activation_bits
        + active_experts * out_hidden * in_hidden * weight_bits
        + routed_tokens * output_width * cfg.dtype_c_bits
    ) / 8
    print(f"  GEMM throughput: {flops / (exec_time * 1e6):.2f} TFLOPS")
    print(
        f"  Effective bandwidth (logical A+B+C): "
        f"{io_bytes / (exec_time * 1e3):.2f} GB/s "
        f"({io_bytes / (exec_time * 1e6):.3f} TB/s)"
    )
    return exec_time


def profile(
    *,
    cuda_profiler_range=False,
    input_device: str = "cuda",
    **launch_cfg,
) -> None:
    """Compile and launch the kernel exactly once, with no timing or reference.

    Intended for external profilers. Unlike :func:`benchmark`, this never
    stream-captures the kernel: the stream check in
    ``cutlass.cute.testing.benchmark`` captures a launch it never replays, so it
    is unsuitable when exactly one observable launch is required.
    """
    import cuda.bindings.driver as cuda_drv

    io = _build_launch_io(input_device=input_device, **launch_cfg)
    cfg = io["cfg"]
    print(
        f"Profile: tokens={io['num_tokens']}, "
        f"padded_tokens={io['total_padded_tokens']}, experts={io['L']}, "
        f"M={io['M']}, N={io['N']}, K={io['K']}, "
        f"tile=({cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k})"
    )
    torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(torch_stream.cuda_stream)
    compiled_fn = _compile_for_launch(io, stream)
    launch_args = _launch_arg_tuple(io, stream)
    torch.cuda.synchronize()
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        compiled_fn(*launch_args)
        torch.cuda.synchronize()
    finally:
        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    print("  Launched kernel once.")
    # Clustered persistent kernels are sensitive to compiled-library lifetime.
    del compiled_fn
    gc.collect()


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
        "--profile",
        action="store_true",
        help=(
            "Launch the kernel exactly once with no correctness check, timing, "
            "or stream capture (for external profilers)."
        ),
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
    parser.add_argument(
        "--host-input-gen",
        action="store_true",
        help=(
            "Generate inputs on CPU and copy them to CUDA, avoiding pre-GEMM "
            "device kernels. CPU and CUDA RNG values differ."
        ),
    )

    parser.add_argument(
        "--sm-arch",
        type=int,
        choices=(100, 103, 107),
        default=None,
        help="Target SM architecture (default: the active CUDA device, else 100).",
    )
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--routing",
        type=_routing_cli_arg,
        default="balanced_round_robin",
        help="Token routing policy: balanced_round_robin, uniform, or zipf:<alpha>.",
    )
    parser.add_argument(
        "--routing-seed",
        type=int,
        default=42,
        help="Deterministic host RNG seed for stochastic routing policies.",
    )
    parser.add_argument("--batch-mode", choices=["n", "m"], default="n")
    parser.add_argument(
        "--route-act",
        choices=["false", "none", "tma", "ldgsts"],
        default="none",
    )
    parser.add_argument(
        "--route-sfs-act",
        choices=["auto", "false", "none", "tma", "ldgsts"],
        default="auto",
    )
    parser.add_argument(
        "--tile-scheduler",
        choices=["static", "persistent"],
        default="static",
    )
    parser.add_argument(
        "--cta-swizzle-type",
        choices=["rasterizeAlongM", "rasterizeAlongN"],
        default="rasterizeAlongM",
        help=(
            "Persistent CLC CTA rasterization order; the option name and values "
            "match TRTLLM-gen, but do not enable an additional CTA swizzle."
        ),
    )
    parser.add_argument(
        "--transpose-mma-output",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--use-shuffled-matrix",
        type=_binary_cli_flag,
        choices=[0, 1],
        default=0,
        help=(
            "Use TRTLLM-gen shuffleMatrix weight/SF storage for batch-M. "
            "Accepts true/false or 1/0. Batch-N always uses the shuffled "
            "representation."
        ),
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

    parser.add_argument("--umma-cta-count", type=int, default=1)
    parser.add_argument("--cluster-dim-x", type=int, default=0)
    parser.add_argument("--cluster-dim-y", type=int, default=1)
    parser.add_argument("--fallback-cluster-dim-x", type=int, default=0)
    parser.add_argument("--fallback-cluster-dim-y", type=int, default=1)
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
    parser.add_argument("--num-stages-a", type=int, default=5)
    parser.add_argument("--num-stages-b", type=int, default=5)
    parser.add_argument("--num-stages-smem-sfa", type=int, default=5)
    parser.add_argument("--num-stages-smem-sfb", type=int, default=5)
    parser.add_argument("--num-stages-tmem-acc", type=int, default=1)
    parser.add_argument("--num-stages-tmem-sfa", type=int, default=5)
    parser.add_argument("--num-stages-tmem-sfb", type=int, default=5)
    parser.add_argument("--num-stages-c-smem", type=int, default=2)
    parser.add_argument(
        "--tmem-ldst-max-num-regs",
        type=int,
        choices=(32, 64),
        default=32,
        help="Maximum register width for TMEM load/store instructions.",
    )
    parser.add_argument("--use-tma-oob-opt", type=int, default=1)
    parser.add_argument("--use-tma-store", type=int, default=0)
    parser.add_argument("--num-load-a-warps", type=int, default=0)
    parser.add_argument("--num-load-b-warps", type=int, default=0)
    parser.add_argument("--num-load-sfa-warps", type=int, default=0)
    parser.add_argument("--num-load-sfb-warps", type=int, default=0)
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
    parser.add_argument("--fuse-sf-copy-to-mma", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fuse-operand-sf-loads", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use-early-exit", type=int, default=0)
    parser.add_argument("--use-clc-fast-drain", type=int, default=0)
    parser.add_argument("--use-work-throttle", type=int, default=1)
    parser.add_argument("--use-pdl", type=int, default=0)
    parser.add_argument("--skip-early-exit-pdl", type=int, choices=(0, 1), default=0)
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
    parser.add_argument(
        "--dist",
        choices=DATA_DISTRIBUTIONS,
        default="default",
        help=(
            "Input-data distribution, matching TRTLLM-gen's -dist option. "
            "allZeros clears token data, weight data, enabled bias, and input "
            "scale factors, but preserves routing metadata."
        ),
    )
    parser.add_argument(
        "--scale-c",
        type=_auto_or_float_cli_arg,
        default=None,
        metavar="AUTO_OR_FLOAT",
        help="Output global scale (default: auto for eligible NVFP4 fused activations).",
    )
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
      --profile       : a single untimed launch, no correctness check.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.profile and (args.benchmark or args.validate_only):
        parser.error("--profile cannot be combined with --benchmark or --validate-only")
    overrides = _parse_overrides(args)
    input_device = "cpu" if args.host_input_gen else "cuda"

    if args.validate_only:
        validate_schedule(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            **overrides,
        )
        return 0

    if args.profile:
        profile(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            routing=args.routing,
            routing_seed=args.routing_seed,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
            data_distribution=args.dist,
            scale_c_value=args.scale_c,
            scale_gate_value=args.scale_gate,
            gemm1_alpha_value=args.gemm1_alpha,
            gemm1_beta_value=args.gemm1_beta,
            gemm1_clamp_limit_value=args.gemm1_clamp_limit,
            cuda_profiler_range=args.cuda_profiler_range,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            input_device=input_device,
            **overrides,
        )
        return 0

    # GPU path: correctness runs by default; --benchmark adds timing.
    if not args.skip_ref_check:
        reference_check(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            routing=args.routing,
            routing_seed=args.routing_seed,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
            data_distribution=args.dist,
            scale_c_value=args.scale_c,
            scale_gate_value=args.scale_gate,
            gemm1_alpha_value=args.gemm1_alpha,
            gemm1_beta_value=args.gemm1_beta,
            gemm1_clamp_limit_value=args.gemm1_clamp_limit,
            early_exit_max_token_ctas=args.early_exit_max_token_ctas,
            input_device=input_device,
            **overrides,
        )

    if args.benchmark:
        num_rotated_buffers = 0 if args.no_cold_l2 else args.num_rotated_buffers
        benchmark(
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            routing=args.routing,
            routing_seed=args.routing_seed,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
            data_distribution=args.dist,
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
            input_device=input_device,
            **overrides,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
