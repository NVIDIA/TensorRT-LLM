# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""
Map Prims-TS MoE autotuner tactics to local Prims-TS GEMM configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import math
from typing import Sequence

from flashinfer.tllm_enums import ActivationType, WeightLayout

SUPPORTED_BF16_TILE_N = (8, 16, 32, 64, 128, 192, 256)
SUPPORTED_NVFP4_TILE_N = (8, 16, 32, 64, 128, 256)
SUPPORTED_FP8_TILE_N = (8, 16, 32, 64, 128, 256)
SUPPORTED_MXFP4_MXFP8_TILE_N = (8, 16, 32, 64, 128, 192, 256)
SUPPORTED_MXFP4_BF16_TILE_N = (8, 16, 32, 64, 128)
SUPPORTED_DSFP8_TILE_N = (8, 16, 32, 64, 128)
SUPPORTED_MXFP8_MXFP8_TILE_N = (8, 16, 32, 64, 128, 256)

_ACTIVATION_TO_ACT_KIND = {
    int(ActivationType.Identity): 0,
    int(ActivationType.Swiglu): 1,
    int(ActivationType.Geglu): 2,
    int(ActivationType.Relu2): 3,
    int(ActivationType.Silu): 4,
    int(ActivationType.Situ): 5,
}


from ..batched_gemm.config_types import (
    BatchMode as _BatchMode,
    RouteImpl as _RouteImpl,
    TileScheduler as _TileScheduler,
    ActKind as _ActKind,
    DType as _DType,
    SfLayout as _SfLayout,
    BiasType as _BiasType,
)


@dataclass(frozen=True)
class PrimsTsConfigSpec:
    """Lazy config spec that can materialize a full BatchedGemmConfig."""

    kwargs: dict[str, int]

    def build(self) -> object:
        from ..batched_gemm.batched_gemm_config import make_config
        from ..batched_gemm.batched_gemm_run import (
            _resolve_runtime_sm_arch,
        )

        return make_config(**_resolve_runtime_sm_arch(self.kwargs))


@dataclass(frozen=True)
class PrimsTsMoeGemmConfig:
    """One Prims-TS GEMM config plus the local config row it came from."""

    cfg: object
    prims_ts_gemm_config_index: int


@dataclass(frozen=True)
class PrimsTsGemmPair:
    """FC1/FC2 config pair for one Prims-TS MoE tactic."""

    tile_n: int
    moe_config_index: int
    fc1: PrimsTsMoeGemmConfig
    fc2: PrimsTsMoeGemmConfig


@dataclass(frozen=True)
class _JsonBatchedGemmConfig:
    global_index: int
    raw_index: int
    comment: str
    combo_index: int
    options: dict[str, object]
    is_scheduler_union_variant: bool = False


_SUPPORTED_JSON_OPTION_KEYS = frozenset(
    {
        "act",
        "bias_type",
        "cast_a_regs",
        "cluster_dim_x",
        "cluster_dim_y",
        "cta_raster_order",
        "umma_cta_count",
        "copy_sf_regs",
        "dtype_a",
        "dtype_b",
        "dtype_c",
        "eltwise_act_type",
        "epi_tile_m",
        "epi_tile_n",
        "epilogue_regs",
        "fused_act",
        "fuse_operand_sf_loads",
        "fuse_sf_copy_to_mma",
        "fallback_cluster_dim_x",
        "fallback_cluster_dim_y",
        "load_a_regs",
        "load_b_regs",
        "load_regs",
        "load_sf_regs",
        "load_sfa_regs",
        "load_sfab_regs",
        "load_sfb_regs",
        "mma_k",
        "mma_m",
        "mma_n",
        "mma_regs",
        "num_load_a_warps",
        "num_load_b_warps",
        "num_load_sfa_warps",
        "num_load_sfb_warps",
        "num_stages_a",
        "num_stages_b",
        "num_stages_c_smem",
        "num_stages_smem_sfa",
        "num_stages_smem_sfb",
        "num_stages_tmem_acc",
        "num_stages_tmem_sfa",
        "num_stages_tmem_sfb",
        "num_stages_workid",
        "padding_regs",
        "per_token_sf_dtype",
        "route_act",
        "route_sfs_act",
        "sf_layout_a",
        "sf_layout_b",
        "sf_layout_c",
        "skip_early_exit_pdl",
        "tile_k",
        "tile_n",
        "tile_scheduler",
        "tmem_ldst_max_num_regs",
        "transpose_mma_output",
        "use_clc_fast_drain",
        "use_deepseek_fp8",
        "use_early_exit",
        "use_global_scales",
        "use_max_tmem_overlap",
        "use_pdl",
        "use_per_token_sf_a",
        "use_per_token_sf_b",
        "use_tma_oob_opt",
        "use_tma_store",
        "use_unroll_loop_2x_for_mma",
        "use_work_throttle",
        "weight_layout",
        "workid_regs",
    }
)


def _next_power_of_two(value: float) -> int:
    n = int(math.ceil(value))
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _is_expert_parallel(num_local_experts: int | None, num_experts: int | None) -> bool:
    """True when this rank holds only a slice of the routed experts."""
    if num_local_experts is None or num_experts is None:
        return False
    return int(num_experts) > int(num_local_experts)


def _avg_tokens_per_expert(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    num_experts: int | None,
) -> float:
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    # Tokens pick their top-k out of every expert, so a local expert receives
    # num_tokens * top_k / num_experts rows on average, whatever the EP split.
    selection_experts = num_local_experts if num_experts is None else int(num_experts)
    if selection_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {selection_experts}")
    return float(num_tokens * top_k) / float(selection_experts)


def _center_tile_n(avg_tokens_per_expert: float, supported_tiles: Sequence[int]) -> int:
    """Smallest supported tile that covers the average expert-token count."""
    return min(
        max(_next_power_of_two(avg_tokens_per_expert), supported_tiles[0]),
        supported_tiles[-1],
    )


def _selected_tile_ns(
    *,
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
    num_experts: int | None = None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> tuple[int, ...]:
    if num_tokens is None or top_k is None or num_local_experts is None:
        return (supported_tiles[-1],)
    avg_tokens_per_expert = _avg_tokens_per_expert(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
    )
    center_tile = _center_tile_n(avg_tokens_per_expert, supported_tiles)
    center_idx = supported_tiles.index(center_tile)
    selected_tiles = {center_tile}
    for offset in (1, 2):
        if center_idx + offset < len(supported_tiles):
            selected_tiles.add(supported_tiles[center_idx + offset])
    if center_idx > 0:
        lower_tile = supported_tiles[center_idx - 1]
        selected_tiles.add(lower_tile)
        # Keep the original power-of-two lower neighbor when an intermediate
        # expert-token tile (for example 192) is added below the center. This
        # lets the autotuner consider N=192 for large routes without dropping
        # the N=128 tactics that are important for smaller EP workloads.
        if center_idx > 1 and lower_tile & (lower_tile - 1):
            selected_tiles.add(supported_tiles[center_idx - 2])
    # Very wide token tiles only become plausible once they are at least half
    # full on average. Keep smaller tiles available to cover routing skew.
    selected_tiles = {
        tile
        for tile in selected_tiles
        if tile <= 128 or tile <= 2 * avg_tokens_per_expert
    }
    return tuple(sorted(selected_tiles))


def _fp8_per_tensor_supported_tiles(*, use_per_token_sf_b: bool) -> tuple[int, ...]:
    del use_per_token_sf_b
    return SUPPORTED_FP8_TILE_N


def _default_tile_n(
    *,
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
    num_experts: int | None = None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> int:
    selected = _selected_tile_ns(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=supported_tiles,
    )
    if (
        num_tokens is not None
        and top_k is not None
        and _is_expert_parallel(num_local_experts, num_experts)
    ):
        # Under expert parallelism a local expert sees few rows but its
        # weights are streamed once per token tile, so the untuned default is
        # the smallest tile that covers the average rather than the narrower
        # neighbour, which would read every expert's weights twice.
        avg_tokens_per_expert = _avg_tokens_per_expert(
            num_tokens=num_tokens,
            top_k=top_k,
            num_local_experts=num_local_experts,
            num_experts=num_experts,
        )
        for tile in selected:
            if tile >= avg_tokens_per_expert:
                return tile
        return selected[-1]
    return selected[0]


def _parse_tactic(
    tactic: int | Sequence[int],
    *,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> tuple[int, int]:
    if tactic == -1:
        return (
            _default_tile_n(
                num_tokens=num_tokens,
                top_k=top_k,
                num_local_experts=num_local_experts,
                num_experts=num_experts,
                supported_tiles=supported_tiles,
            ),
            -1,
        )
    if not isinstance(tactic, Sequence) or len(tactic) != 2:
        raise ValueError(f"Expected tactic [tile_N, config_index], got {tactic!r}")
    tile_n = int(tactic[0])
    moe_config_index = int(tactic[1])
    if tile_n == -1 or moe_config_index == -1:
        tile_n = _default_tile_n(
            num_tokens=num_tokens,
            top_k=top_k,
            num_local_experts=num_local_experts,
            num_experts=num_experts,
            supported_tiles=supported_tiles,
        )
        moe_config_index = -1
    return tile_n, moe_config_index


def _activation_act_kind(activation_type: int) -> int:
    act_kind = _ACTIVATION_TO_ACT_KIND.get(int(activation_type))
    if act_kind is None:
        raise ValueError(f"Unsupported Prims-TS activation_type={activation_type!r}")
    return act_kind


def _bias_type(has_bias: bool) -> int:
    return int(_BiasType.M if has_bias else _BiasType.NONE)


def _bias_kwargs(has_bias: bool) -> dict[str, int]:
    return {"bias_type": _bias_type(has_bias)}


def _gemm1_oa_flags(
    *,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
) -> dict[str, int]:
    return {
        "has_gemm1_alpha": int(has_gemm1_alpha),
        "has_gemm1_beta": int(has_gemm1_beta),
        "has_gemm1_clamp_limit": int(has_gemm1_clamp_limit),
    }


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return bool(value)


def _json_dtype_value(value: object | None, *, default: str = "bf16") -> int:
    dtype = default if value is None else str(value).lower()
    if dtype == "fp32":
        return int(_DType.FP32)
    if dtype == "fp16":
        return int(_DType.FP16)
    if dtype == "bf16":
        return int(_DType.BF16)
    if dtype == "e2m1":
        return int(_DType.E2M1)
    if dtype == "mxe2m1":
        return int(_DType.MXE2M1)
    if dtype == "mxe4m3":
        return int(_DType.MXE4M3)
    if dtype == "e4m3":
        return int(_DType.E4M3)
    raise ValueError(f"Unsupported TRT-LLM Gen dtype value: {value!r}")


def _dtype_json_name(dtype: int) -> str:
    dtype = int(dtype)
    if dtype == int(_DType.FP32):
        return "fp32"
    if dtype == int(_DType.FP16):
        return "fp16"
    if dtype == int(_DType.BF16):
        return "bf16"
    if dtype == int(_DType.E2M1):
        return "e2m1"
    if dtype == int(_DType.MXE2M1):
        return "mxe2m1"
    if dtype == int(_DType.MXE4M3):
        return "mxe4m3"
    if dtype == int(_DType.E4M3):
        return "e4m3"
    raise ValueError(f"Unsupported Prims-TS dtype value: {dtype!r}")


def _json_route_value(value: object) -> int:
    if value is False or value is None:
        return int(_RouteImpl.NONE)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"false", "none"}:
            return int(_RouteImpl.NONE)
        if lowered == "tma":
            return int(_RouteImpl.TMA)
        if lowered == "ldgsts":
            return int(_RouteImpl.LDGSTS)
        if lowered == "auto":
            return int(_RouteImpl.AUTO)
    raise ValueError(f"Unsupported TRT-LLM Gen route value: {value!r}")


def _json_scheduler_value(value: object) -> int:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "static":
            return int(_TileScheduler.STATIC)
        if lowered == "persistent":
            return int(_TileScheduler.PERSISTENT)
    raise ValueError(f"Unsupported TRT-LLM Gen scheduler value: {value!r}")


def _json_sf_layout_value(value: object | None, *, default: str = "128x4") -> int:
    layout = default if value is None else str(value)
    lowered = layout.lower()
    if lowered == "8x4":
        return int(_SfLayout.R8c4)
    if lowered == "linear":
        return int(_SfLayout.LINEAR)
    if lowered == "128x4":
        return int(_SfLayout.R128c4)
    raise ValueError(f"Unsupported TRT-LLM Gen scale-factor layout: {value!r}")


def _json_weight_layout_value(
    value: object | None,
    *,
    default: WeightLayout | int = WeightLayout.MajorK,
) -> int:
    if value is None:
        return int(default)
    if isinstance(value, WeightLayout):
        return int(value)
    if isinstance(value, int):
        return int(WeightLayout(value))
    lowered = str(value).replace("_", "").replace("-", "").lower()
    if lowered == "majork":
        return int(WeightLayout.MajorK)
    if lowered == "blockmajork":
        return int(WeightLayout.BlockMajorK)
    if lowered == "majormn":
        return int(WeightLayout.MajorMn)
    raise ValueError(f"Unsupported TRT-LLM Gen weight layout: {value!r}")


def _json_act_kind(options: dict[str, object]) -> int:
    if _bool_value(options.get("fused_act", False)):
        act = str(options.get("act", "none")).lower()
        if act == "swiglu":
            return int(_ActKind.SWIGLU)
        if act == "geglu":
            return int(_ActKind.GEGLU)
    eltwise_act = str(options.get("eltwise_act_type", "none")).lower()
    if eltwise_act == "relu2":
        return int(_ActKind.RELU2)
    if eltwise_act == "silu":
        return int(_ActKind.SILU)
    return int(_ActKind.NONE)


def _resolve_moe_sm_arch(sm_arch: int | None = None) -> int:
    """Resolve the architecture used to select and materialize MoE configs."""
    if sm_arch is None:
        from ..batched_gemm.batched_gemm_run import (
            _resolve_runtime_sm_arch,
        )

        sm_arch = int(_resolve_runtime_sm_arch({}).get("sm_arch", 100))
    if sm_arch not in (100, 103, 107):
        raise ValueError(f"sm_arch must be 100, 103, or 107, got {sm_arch}")
    return sm_arch


def _prims_ts_config_resource(sm_arch: int):
    config_name = (
        "prims_ts_moe_configs_sm107.json"
        if sm_arch == 107
        else "prims_ts_moe_configs.json"
    )
    config = resources.files(__package__).joinpath(config_name)
    if not config.is_file():
        raise FileNotFoundError(
            f"Packaged Prims-TS MoE config JSON is required but was not found: {config}"
        )
    return config


def _resolve_json_template(
    name: str,
    templates: dict[str, dict[str, object]],
    *,
    stack: tuple[str, ...] = (),
) -> dict[str, object]:
    if name in stack:
        raise ValueError(
            f"Prims-TS config template cycle: {' -> '.join(stack + (name,))}"
        )
    raw = templates[name]
    merged: dict[str, object] = {}
    parent = raw.get("_template")
    if parent is not None:
        merged.update(
            _resolve_json_template(str(parent), templates, stack=stack + (name,))
        )
    merged.update({key: value for key, value in raw.items() if key != "_template"})
    return merged


def _validate_json_options(
    cfg: _JsonBatchedGemmConfig, options: dict[str, object]
) -> None:
    unsupported = sorted(set(options) - _SUPPORTED_JSON_OPTION_KEYS)
    if unsupported:
        raise ValueError(
            "Local Prims-TS MoE config contains unsupported option(s) "
            f"{unsupported} in config {cfg.raw_index} ({cfg.comment}) "
            f"combo {cfg.combo_index}. Rename or remove stale TRT-LLM Gen "
            "metadata before using this config."
        )


def _expanded_prims_ts_json_configs(
    sm_arch: int | None = None,
) -> tuple[_JsonBatchedGemmConfig, ...]:
    return _expanded_prims_ts_json_configs_for_arch(_resolve_moe_sm_arch(sm_arch))


@lru_cache(maxsize=None)
def _expanded_prims_ts_json_configs_for_arch(
    sm_arch: int,
) -> tuple[_JsonBatchedGemmConfig, ...]:
    config = _prims_ts_config_resource(sm_arch)
    data = json.loads(config.read_text(encoding="utf-8"))
    declared_archs = tuple(int(value) for value in data.get("sm_archs", ()))
    if sm_arch not in declared_archs:
        raise ValueError(
            f"Prims-TS MoE config {config.name} declares sm_archs={declared_archs}, "
            f"not requested sm_arch={sm_arch}"
        )
    templates = data["templates"]
    expanded: list[_JsonBatchedGemmConfig] = []

    for raw_index, raw_config in enumerate(data["configs"]):
        comment = raw_config.get("_comment", f"config_{raw_index}")
        merged: dict[str, object] = {}
        parent = raw_config.get("_template")
        if parent is not None:
            merged.update(_resolve_json_template(str(parent), templates))
        merged.update(
            {
                key: value
                for key, value in raw_config.items()
                if key not in ("_template", "_comment")
            }
        )

        combos: list[dict[str, object]] = [{}]
        for key, value in merged.items():
            if "," in key:
                keys = tuple(part.strip() for part in key.split(",") if part.strip())
                rows = value if isinstance(value, list) else [value]
                choices = []
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(keys):
                        raise ValueError(
                            f"Prims-TS config {comment}: grouped key {key!r} "
                            f"expects {len(keys)} values, got {row!r}"
                        )
                    choices.append(dict(zip(keys, row, strict=True)))
            elif isinstance(value, list):
                choices = [{key: item} for item in value]
            else:
                choices = [{key: value}]
            combos = [{**base, **choice} for base in combos for choice in choices]

        for combo_index, options in enumerate(combos):
            cfg = _JsonBatchedGemmConfig(
                global_index=len(expanded),
                raw_index=raw_index,
                comment=str(comment),
                combo_index=combo_index,
                options=options,
            )
            _validate_json_options(cfg, options)
            expanded.append(cfg)

    # Keep every existing config and append the persistent fast-drain plus
    # work-throttled counterpart when that exact scheduler variant is absent.
    # Appending after the original expansion preserves every original global
    # config index while making the scheduler-policy configurations available
    # to fresh autotuning.
    seen_options = {
        json.dumps(cfg.options, sort_keys=True, separators=(",", ":"))
        for cfg in expanded
    }
    original_configs = tuple(expanded)
    for cfg in original_configs:
        if str(cfg.options.get("tile_scheduler", "static")).lower() != "persistent":
            continue
        options = {
            **cfg.options,
            "use_clc_fast_drain": True,
            "use_work_throttle": True,
        }
        options_key = json.dumps(options, sort_keys=True, separators=(",", ":"))
        if options_key in seen_options:
            continue
        seen_options.add(options_key)
        variant = _JsonBatchedGemmConfig(
            global_index=len(expanded),
            raw_index=cfg.raw_index,
            comment=f"{cfg.comment} [fast-drain+work-throttle]",
            combo_index=cfg.combo_index,
            options=options,
            is_scheduler_union_variant=True,
        )
        _validate_json_options(variant, options)
        expanded.append(variant)
    return tuple(expanded)


def _expanded_trtllm_gen_json_configs(
    sm_arch: int | None = None,
) -> tuple[_JsonBatchedGemmConfig, ...]:
    """Temporary compatibility alias for tests during the Prims-TS refactor."""
    return _expanded_prims_ts_json_configs(sm_arch)


def _json_config_by_global_index(
    global_index: int, *, sm_arch: int
) -> _JsonBatchedGemmConfig:
    configs = _expanded_prims_ts_json_configs(sm_arch)
    if global_index < 0 or global_index >= len(configs):
        raise ValueError(f"Unknown Prims-TS batched GEMM config index={global_index}")
    return configs[global_index]


def _json_config_matches_moe(
    cfg: _JsonBatchedGemmConfig,
    *,
    tile_n: int,
    activation_type: int,
    fc: str,
    dtype_a: int,
    dtype_b: int,
    dtype_c: int,
    use_deepseek_fp8: bool = False,
    use_per_token_sf_a: bool | None = None,
    use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
) -> bool:
    options = cfg.options
    try:
        json_dtype_a = _json_dtype_value(
            options.get("dtype_a"), default=_dtype_json_name(dtype_a)
        )
        json_dtype_b = _json_dtype_value(
            options.get("dtype_b"), default=_dtype_json_name(dtype_b)
        )
        json_dtype_c = _json_dtype_value(
            options.get("dtype_c"), default=_dtype_json_name(dtype_c)
        )
    except ValueError:
        return False
    if json_dtype_a != int(dtype_a):
        return False
    if json_dtype_b != int(dtype_b):
        return False
    if json_dtype_c != int(dtype_c):
        return False
    if _bool_value(options.get("use_deepseek_fp8", False)) != bool(use_deepseek_fp8):
        return False
    try:
        json_weight_layout = _json_weight_layout_value(
            options.get("weight_layout"),
            default=weight_layout,
        )
    except ValueError:
        return False
    if json_weight_layout != int(weight_layout):
        return False
    transposed = _bool_value(options.get("transpose_mma_output", True))
    if not transposed and (dtype_a != int(_DType.BF16) or dtype_b != int(_DType.BF16)):
        return False
    token_tile = (
        int(options.get("tile_n", -1))
        if transposed
        else 128
        * (
            int(options.get("cluster_dim_x", 0))
            or int(options.get("umma_cta_count", 1))
        )
    )
    if token_tile != int(tile_n):
        return False

    route_active = _json_route_value(options.get("route_act", False)) != int(
        _RouteImpl.NONE
    )
    if route_active != (fc == "fc1"):
        return False

    act_kind = _json_act_kind(options)
    if use_deepseek_fp8:
        if act_kind != int(_ActKind.NONE):
            return False
    elif fc == "fc1":
        expected_act_kind = _activation_act_kind(activation_type)
        # SiTU uses the same gated GEMM geometry as SwiGLU. Existing JSON
        # tactic rows describe geometry rather than the activation formula,
        # so reuse SwiGLU rows and replace the generated ActKind below.
        if expected_act_kind == int(_ActKind.SITU):
            expected_act_kind = int(_ActKind.SWIGLU)
        # SiLU uses the same non-gated GEMM geometry as ReLU2. Reuse those
        # rows and replace the generated ActKind below.
        if expected_act_kind == int(_ActKind.SILU):
            expected_act_kind = int(_ActKind.RELU2)
        if act_kind != expected_act_kind:
            return False
    elif act_kind != int(_ActKind.NONE):
        return False

    json_use_per_token_sf_a = _bool_value(options.get("use_per_token_sf_a", False))
    if use_per_token_sf_a is None:
        if json_use_per_token_sf_a:
            return False
    elif json_use_per_token_sf_a != bool(use_per_token_sf_a):
        return False
    json_use_per_token_sf_b = _bool_value(options.get("use_per_token_sf_b", False))
    if use_per_token_sf_b is None:
        if json_use_per_token_sf_b:
            return False
    elif json_use_per_token_sf_b != bool(use_per_token_sf_b):
        return False
    if (
        bool(use_per_token_sf_a) or bool(use_per_token_sf_b)
    ) and per_token_sf_dtype is not None:
        try:
            json_per_token_sf_dtype = _json_dtype_value(
                options.get("per_token_sf_dtype"), default="bf16"
            )
        except ValueError:
            return False
        if json_per_token_sf_dtype != int(per_token_sf_dtype):
            return False

    if str(options.get("bias_type", "none")).lower() == "mn":
        return False
    return True


@lru_cache(maxsize=None)
def _moe_json_passing_indices(
    *,
    sm_arch: int,
    tile_n: int,
    activation_type: int,
    fc: str,
    dtype_a: int,
    dtype_b: int,
    dtype_c: int,
    use_deepseek_fp8: bool = False,
    use_per_token_sf_a: bool | None = None,
    use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
) -> tuple[int, ...]:
    return tuple(
        cfg.global_index
        for cfg in _expanded_prims_ts_json_configs(sm_arch)
        if _json_config_matches_moe(
            cfg,
            tile_n=tile_n,
            activation_type=activation_type,
            fc=fc,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            use_deepseek_fp8=use_deepseek_fp8,
            use_per_token_sf_a=use_per_token_sf_a,
            use_per_token_sf_b=use_per_token_sf_b,
            per_token_sf_dtype=per_token_sf_dtype,
            weight_layout=weight_layout,
        )
    )


def _same_batch_mode(fc1_index: int, fc2_index: int, *, sm_arch: int) -> bool:
    return _bool_value(
        _json_config_by_global_index(fc1_index, sm_arch=sm_arch).options.get(
            "transpose_mma_output", True
        )
    ) == _bool_value(
        _json_config_by_global_index(fc2_index, sm_arch=sm_arch).options.get(
            "transpose_mma_output", True
        )
    )


def _sorted_moe_json_config_pairs(
    tile_n: int,
    *,
    sm_arch: int,
    activation_type: int,
    dtype_a: int,
    dtype_b: int,
    fc1_dtype_c: int,
    fc2_dtype_c: int,
    dtype_label: str,
    use_deepseek_fp8: bool = False,
    fc1_use_per_token_sf_a: bool | None = None,
    fc2_use_per_token_sf_a: bool | None = None,
    fc1_use_per_token_sf_b: bool | None = None,
    fc2_use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[tuple[int, int]]:
    fc1_indices = _moe_json_passing_indices(
        sm_arch=sm_arch,
        tile_n=tile_n,
        activation_type=activation_type,
        fc="fc1",
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=fc1_dtype_c,
        use_deepseek_fp8=use_deepseek_fp8,
        use_per_token_sf_a=fc1_use_per_token_sf_a,
        use_per_token_sf_b=fc1_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    fc2_indices = _moe_json_passing_indices(
        sm_arch=sm_arch,
        tile_n=tile_n,
        activation_type=activation_type,
        fc="fc2",
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=fc2_dtype_c,
        use_deepseek_fp8=use_deepseek_fp8,
        use_per_token_sf_a=fc2_use_per_token_sf_a,
        use_per_token_sf_b=fc2_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    if not fc1_indices or not fc2_indices:
        raise ValueError(
            "No local Prims-TS batched GEMM JSON configs match "
            f"{dtype_label} tile_N={tile_n}, activation_type={activation_type}, "
            f"fc1_matches={len(fc1_indices)}, fc2_matches={len(fc2_indices)}"
        )
    config_pairs = [
        (fc1_index, fc2_index)
        for fc1_index in fc1_indices
        for fc2_index in fc2_indices
        if _same_batch_mode(fc1_index, fc2_index, sm_arch=sm_arch)
    ]
    if use_deepseek_fp8:
        # Persisted autotuner caches store this pair index. Keep all original
        # no-fast-drain pairs first, with the legacy unthrottled 2x2 Cartesian
        # product at indices 0..3 and the work-throttled pairs at 4..8. Append
        # fast-drain pairs without reinterpreting any existing tactic.
        config_pairs.sort(
            key=lambda pair: (
                int(
                    _bool_value(
                        _json_config_by_global_index(
                            pair[0], sm_arch=sm_arch
                        ).options.get("use_clc_fast_drain", False)
                    )
                    or _bool_value(
                        _json_config_by_global_index(
                            pair[1], sm_arch=sm_arch
                        ).options.get("use_clc_fast_drain", False)
                    )
                ),
                int(
                    _bool_value(
                        _json_config_by_global_index(
                            pair[0], sm_arch=sm_arch
                        ).options.get("use_work_throttle", False)
                    )
                    or _bool_value(
                        _json_config_by_global_index(
                            pair[1], sm_arch=sm_arch
                        ).options.get("use_work_throttle", False)
                    )
                ),
            )
        )
    else:
        # Appended union variants must not reinterpret any persisted baseline
        # pair index. Keep the complete original FC1 x FC2 product first and
        # place every pair involving a new scheduler variant afterward.
        config_pairs.sort(
            key=lambda pair: (
                not _bool_value(
                    _json_config_by_global_index(pair[0], sm_arch=sm_arch).options.get(
                        "transpose_mma_output", True
                    )
                ),
                int(
                    _json_config_by_global_index(
                        pair[0], sm_arch=sm_arch
                    ).is_scheduler_union_variant
                    or _json_config_by_global_index(
                        pair[1], sm_arch=sm_arch
                    ).is_scheduler_union_variant
                ),
            )
        )
    return config_pairs


def _resolve_moe_json_config_pair(
    tile_n: int,
    moe_config_index: int,
    *,
    sm_arch: int,
    dtype_label: str,
    **kwargs,
) -> tuple[_JsonBatchedGemmConfig, _JsonBatchedGemmConfig] | None:
    if moe_config_index < 0:
        return None
    config_pairs = _sorted_moe_json_config_pairs(
        tile_n, sm_arch=sm_arch, dtype_label=dtype_label, **kwargs
    )
    total = len(config_pairs)
    if moe_config_index >= total:
        raise ValueError(
            f"Unsupported MoE config index={moe_config_index}; valid range is "
            f"[0, {total - 1}] for {dtype_label} tile_N={tile_n}"
        )
    fc1_index, fc2_index = config_pairs[moe_config_index]
    return _json_config_by_global_index(
        fc1_index, sm_arch=sm_arch
    ), _json_config_by_global_index(fc2_index, sm_arch=sm_arch)


def _json_config_is_persistent(cfg: _JsonBatchedGemmConfig) -> bool:
    return str(cfg.options.get("tile_scheduler", "static")).lower() == "persistent"


def _default_moe_config_index(
    tile_n: int,
    *,
    sm_arch: int,
    prefer_persistent_scheduler: bool,
    **kwargs,
) -> int:
    """Config pair index the untuned ``[-1, -1]`` tactic resolves to.

    Index 0 unless a persistent-scheduler pair is preferred: the static
    scheduler launches the worst-case token-tile grid, which under expert
    parallelism is num_experts / num_local_experts times the real work.
    """
    if not prefer_persistent_scheduler:
        return 0
    for index, (fc1_index, fc2_index) in enumerate(
        _sorted_moe_json_config_pairs(tile_n, sm_arch=sm_arch, **kwargs)
    ):
        if _json_config_is_persistent(
            _json_config_by_global_index(fc1_index, sm_arch=sm_arch)
        ) and _json_config_is_persistent(
            _json_config_by_global_index(fc2_index, sm_arch=sm_arch)
        ):
            return index
    return 0


def _ensure_json_config_supported_by_ts(cfg: _JsonBatchedGemmConfig) -> None:
    _validate_json_options(cfg, cfg.options)


def _json_config_kwargs(
    cfg: _JsonBatchedGemmConfig,
    *,
    sm_arch: int,
    fc: str,
    activation_type: int,
    has_bias: bool,
    default_dtype_a: int,
    default_dtype_b: int,
    default_dtype_c: int,
    weight_layout: int = int(WeightLayout.MajorK),
    use_per_token_sf_a: bool | None = None,
    use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
) -> dict[str, int]:
    _ensure_json_config_supported_by_ts(cfg)
    options = cfg.options
    if has_bias and not _bool_value(options.get("transpose_mma_output", True)):
        raise ValueError("Prims-TS BF16 BatchM MoE does not support per-expert bias")
    use_max_tmem_overlap = int(_bool_value(options.get("use_max_tmem_overlap", False)))
    tmem_acc_stages = int(options["num_stages_tmem_acc"])
    mma_regs = int(options.get("mma_regs", 48))
    load_regs = int(options.get("load_regs", mma_regs))
    load_sf_regs = int(options.get("load_sf_regs", load_regs))
    copy_sf_regs = int(options.get("copy_sf_regs", load_regs))
    padding_regs = int(options.get("padding_regs", load_regs))
    workid_regs = int(options.get("workid_regs", load_regs))
    epilogue_regs = int(options.get("epilogue_regs", 160))
    load_a_regs = int(options.get("load_a_regs", 0))
    load_b_regs = int(options.get("load_b_regs", 0))
    load_sfa_regs = int(options.get("load_sfa_regs", 0))
    load_sfb_regs = int(options.get("load_sfb_regs", 0))
    json_use_per_token_sf_a = _bool_value(options.get("use_per_token_sf_a", False))
    json_use_per_token_sf_b = _bool_value(options.get("use_per_token_sf_b", False))
    if use_per_token_sf_a is None:
        use_per_token_sf_a = json_use_per_token_sf_a
    use_global_scales = _bool_value(options.get("use_global_scales", False))
    if use_per_token_sf_b is None:
        use_per_token_sf_b = json_use_per_token_sf_b
    batch_mode = int(
        _BatchMode.BATCH_N
        if _bool_value(options.get("transpose_mma_output", True))
        else _BatchMode.BATCH_M
    )
    kwargs = {
        "sm_arch": sm_arch,
        "batch_mode": batch_mode,
        # Prims-TS MoE consumes weights shuffled offline by
        # prepare_trtllm_bf16_weights. Batch-N implies this layout in the
        # kernel; Batch-M needs the explicit opt-in.
        "use_shuffled_matrix": int(batch_mode == int(_BatchMode.BATCH_M)),
        "dtype_a": _json_dtype_value(
            options.get("dtype_a"), default=_dtype_json_name(default_dtype_a)
        ),
        "dtype_b": _json_dtype_value(
            options.get("dtype_b"), default=_dtype_json_name(default_dtype_b)
        ),
        "dtype_c": _json_dtype_value(
            options.get("dtype_c"), default=_dtype_json_name(default_dtype_c)
        ),
        "weight_layout": _json_weight_layout_value(
            options.get("weight_layout"),
            default=weight_layout,
        ),
        "mma_k": int(options.get("mma_k", 32)),
        "route_act": _json_route_value(options.get("route_act", False)),
        "route_sfs_act": _json_route_value(options.get("route_sfs_act", "auto")),
        "tile_scheduler": _json_scheduler_value(options["tile_scheduler"]),
        "act_kind": (
            _activation_act_kind(activation_type)
            if fc == "fc1" and not _bool_value(options.get("use_deepseek_fp8", False))
            else _json_act_kind(options)
        ),
        **_bias_kwargs(has_bias),
        "sf_layout_a": _json_sf_layout_value(options.get("sf_layout_a")),
        "sf_layout_b": _json_sf_layout_value(options.get("sf_layout_b")),
        "sf_layout_c": _json_sf_layout_value(options.get("sf_layout_c"), default="8x4"),
        "umma_cta_count": int(options.get("umma_cta_count", 1)),
        "tile_n": int(options["tile_n"]),
        "tile_k": int(options["tile_k"]),
        "epi_tile_m": int(options.get("epi_tile_m", 128)),
        "epi_tile_n": int(options["epi_tile_n"]),
        "mma_m": int(options["mma_m"]),
        "mma_n": int(options["mma_n"]),
        "num_stages_a": int(options["num_stages_a"]),
        "num_stages_b": int(options["num_stages_b"]),
        "num_stages_smem_sfa": int(options["num_stages_smem_sfa"]),
        "num_stages_smem_sfb": int(options["num_stages_smem_sfb"]),
        "num_stages_tmem_sfa": int(options["num_stages_tmem_sfa"]),
        "num_stages_tmem_sfb": int(options["num_stages_tmem_sfb"]),
        "num_stages_tmem_acc": tmem_acc_stages,
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("use_unroll_loop_2x_for_mma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_tma_oob_opt": int(_bool_value(options.get("use_tma_oob_opt", True))),
        "use_early_exit": int(_bool_value(options.get("use_early_exit", True))),
        "use_clc_fast_drain": int(
            _bool_value(options.get("use_clc_fast_drain", False))
        ),
        "use_tma_store": int(_bool_value(options.get("use_tma_store", True))),
        "use_global_scales": int(use_global_scales),
        "use_work_throttle": int(_bool_value(options.get("use_work_throttle", False))),
        "use_max_tmem_overlap": use_max_tmem_overlap,
        "fuse_sf_copy_to_mma": int(
            _bool_value(options.get("fuse_sf_copy_to_mma", False))
        ),
        "fuse_operand_sf_loads": int(
            _bool_value(options.get("fuse_operand_sf_loads", False))
        ),
        "epilogue_regs": epilogue_regs,
        "mma_regs": mma_regs,
        "load_regs": load_regs,
        "load_sf_regs": load_sf_regs,
        "load_a_regs": load_a_regs,
        "load_b_regs": load_b_regs,
        "load_sfa_regs": load_sfa_regs,
        "load_sfb_regs": load_sfb_regs,
        "cast_a_regs": int(options.get("cast_a_regs", 160)),
        "copy_sf_regs": copy_sf_regs,
        "padding_regs": padding_regs,
        "workid_regs": workid_regs,
    }
    if "num_stages_workid" in options:
        kwargs["num_stages_workid"] = int(options["num_stages_workid"])
    if "num_stages_c_smem" in options:
        kwargs["num_stages_c_smem"] = int(options["num_stages_c_smem"])
    if "cluster_dim_x" in options:
        kwargs["cluster_dim_x"] = int(options["cluster_dim_x"])
    if "cluster_dim_y" in options:
        kwargs["cluster_dim_y"] = int(options["cluster_dim_y"])
    if "fallback_cluster_dim_x" in options:
        kwargs["fallback_cluster_dim_x"] = int(options["fallback_cluster_dim_x"])
    if "fallback_cluster_dim_y" in options:
        kwargs["fallback_cluster_dim_y"] = int(options["fallback_cluster_dim_y"])
    if "cta_raster_order" in options:
        kwargs["cta_raster_order"] = int(options["cta_raster_order"])
    if "tmem_ldst_max_num_regs" in options:
        kwargs["tmem_ldst_max_num_regs"] = int(options["tmem_ldst_max_num_regs"])
    if "use_pdl" in options:
        kwargs["use_pdl"] = int(_bool_value(options["use_pdl"]))
    if "skip_early_exit_pdl" in options:
        if _bool_value(options["skip_early_exit_pdl"]):
            raise ValueError(
                "MoE FC2 enables skip_early_exit_pdl automatically; "
                "JSON rows may only set it to false to opt out"
            )
        kwargs["skip_early_exit_pdl"] = 0
    if _bool_value(options.get("use_deepseek_fp8", False)):
        kwargs["use_deepseek_fp8"] = 1
        kwargs["load_sfab_regs"] = int(options.get("load_sfab_regs", load_sf_regs))
    if use_per_token_sf_a:
        kwargs["use_per_token_sf_a"] = 1
    if use_per_token_sf_b:
        kwargs["use_per_token_sf_b"] = 1
    if use_per_token_sf_a or use_per_token_sf_b:
        kwargs["per_token_sf_dtype"] = int(
            per_token_sf_dtype
            if per_token_sf_dtype is not None
            else _json_dtype_value(options.get("per_token_sf_dtype"), default="bf16")
        )
    if fc == "fc1":
        kwargs.update(
            _gemm1_oa_flags(
                has_gemm1_alpha=has_gemm1_alpha,
                has_gemm1_beta=has_gemm1_beta,
                has_gemm1_clamp_limit=has_gemm1_clamp_limit,
            )
        )
    if "num_load_a_warps" in options:
        kwargs["num_load_a_warps"] = int(options["num_load_a_warps"])
    if "num_load_b_warps" in options:
        kwargs["num_load_b_warps"] = int(options["num_load_b_warps"])
    if "num_load_sfa_warps" in options:
        kwargs["num_load_sfa_warps"] = int(options["num_load_sfa_warps"])
    if "num_load_sfb_warps" in options:
        kwargs["num_load_sfb_warps"] = int(options["num_load_sfb_warps"])
    if (128 * kwargs["umma_cta_count"]) % kwargs["mma_m"] != 0:
        raise ValueError(
            "Prims-TS requires cluster-wide tile_m to be a multiple of mma_m "
            f"for batched GEMM config {cfg.global_index}"
        )
    return kwargs


def _make_config_spec(**kwargs: int) -> PrimsTsConfigSpec:
    return PrimsTsConfigSpec(kwargs=kwargs)


def _pdl_overrides(
    kwargs: dict[str, int],
    *,
    enable_pdl: bool,
    skip_early_exit_pdl: bool,
) -> dict[str, int]:
    skip_early_exit_pdl = bool(
        enable_pdl
        and skip_early_exit_pdl
        and kwargs["use_early_exit"]
        and kwargs.get("skip_early_exit_pdl", 1)
    )
    return {
        "use_pdl": int(bool(enable_pdl)),
        "skip_early_exit_pdl": int(skip_early_exit_pdl),
    }


def apply_weight_l2_policy(pair: PrimsTsGemmPair) -> PrimsTsGemmPair:
    """Enable per-expert weight policy selection for BatchN GEMMs.

    The kernel uses the actual routing span of each expert: one token CTA
    uses evict-first; multiple token CTAs use the default eviction priority.
    Different M tiles access different weight slices and do not count as
    reuse of the same weights.
    """

    def with_policy(gemm: PrimsTsMoeGemmConfig) -> PrimsTsMoeGemmConfig:
        kwargs = dict(gemm.cfg.kwargs)
        if kwargs.get("batch_mode", int(_BatchMode.BATCH_N)) != int(_BatchMode.BATCH_N):
            return gemm
        # Wider token CGAs share weights and routing records across CTAs.
        # Keep their normal policy; the neighbor lookup requires one CTA/record.
        if int(kwargs.get("cluster_dim_y", 1)) != 1:
            return gemm
        kwargs["weight_l2_evict_first"] = 1
        return PrimsTsMoeGemmConfig(
            cfg=PrimsTsConfigSpec(kwargs=kwargs),
            prims_ts_gemm_config_index=gemm.prims_ts_gemm_config_index,
        )

    return PrimsTsGemmPair(
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        fc1=with_policy(pair.fc1),
        fc2=with_policy(pair.fc2),
    )


def _with_pdl_spec(
    spec: PrimsTsConfigSpec,
    *,
    enable_pdl: bool,
    skip_early_exit_pdl: bool,
) -> PrimsTsConfigSpec:
    kwargs = dict(spec.kwargs)
    kwargs.update(
        _pdl_overrides(
            kwargs,
            enable_pdl=enable_pdl,
            skip_early_exit_pdl=skip_early_exit_pdl,
        )
    )
    return PrimsTsConfigSpec(kwargs=kwargs)


def _with_spec_overrides_pair(
    pair: PrimsTsGemmPair, **overrides: int
) -> PrimsTsGemmPair:
    """Return ``pair`` with ``overrides`` applied to both FC1 and FC2 specs."""

    def _apply(gemm: PrimsTsMoeGemmConfig) -> PrimsTsMoeGemmConfig:
        kwargs = dict(gemm.cfg.kwargs)
        kwargs.update(overrides)
        return PrimsTsMoeGemmConfig(
            cfg=PrimsTsConfigSpec(kwargs=kwargs),
            prims_ts_gemm_config_index=gemm.prims_ts_gemm_config_index,
        )

    return PrimsTsGemmPair(
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        fc1=_apply(pair.fc1),
        fc2=_apply(pair.fc2),
    )


def _with_pdl_pair(pair: PrimsTsGemmPair, *, enable_pdl: bool) -> PrimsTsGemmPair:
    return PrimsTsGemmPair(
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        fc1=PrimsTsMoeGemmConfig(
            cfg=_with_pdl_spec(
                pair.fc1.cfg,
                enable_pdl=enable_pdl,
                skip_early_exit_pdl=False,
            ),
            prims_ts_gemm_config_index=pair.fc1.prims_ts_gemm_config_index,
        ),
        fc2=PrimsTsMoeGemmConfig(
            cfg=_with_pdl_spec(
                pair.fc2.cfg,
                enable_pdl=enable_pdl,
                skip_early_exit_pdl=True,
            ),
            prims_ts_gemm_config_index=pair.fc2.prims_ts_gemm_config_index,
        ),
    )


def _make_json_moe_config_pair(
    *,
    sm_arch: int,
    tile_n: int,
    moe_config_index: int,
    activation_type: int,
    dtype_a: int,
    dtype_b: int,
    fc1_dtype_c: int,
    fc2_dtype_c: int,
    dtype_label: str,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    use_deepseek_fp8: bool = False,
    fc1_use_per_token_sf_a: bool | None = None,
    fc2_use_per_token_sf_a: bool | None = None,
    fc1_use_per_token_sf_b: bool | None = None,
    fc2_use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    deepseek_fp8_block_size: int = 128,
) -> PrimsTsGemmPair | None:
    json_pair = _resolve_moe_json_config_pair(
        tile_n,
        moe_config_index,
        sm_arch=sm_arch,
        activation_type=activation_type,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        fc1_dtype_c=fc1_dtype_c,
        fc2_dtype_c=fc2_dtype_c,
        dtype_label=dtype_label,
        use_deepseek_fp8=use_deepseek_fp8,
        fc1_use_per_token_sf_a=fc1_use_per_token_sf_a,
        fc2_use_per_token_sf_a=fc2_use_per_token_sf_a,
        fc1_use_per_token_sf_b=fc1_use_per_token_sf_b,
        fc2_use_per_token_sf_b=fc2_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    if json_pair is None:
        return None

    fc1_json, fc2_json = json_pair
    pair = _with_pdl_pair(
        PrimsTsGemmPair(
            tile_n=tile_n,
            moe_config_index=moe_config_index,
            fc1=PrimsTsMoeGemmConfig(
                cfg=_make_config_spec(
                    **_json_config_kwargs(
                        fc1_json,
                        sm_arch=sm_arch,
                        fc="fc1",
                        activation_type=activation_type,
                        has_bias=fc1_has_bias,
                        default_dtype_a=dtype_a,
                        default_dtype_b=dtype_b,
                        default_dtype_c=fc1_dtype_c,
                        weight_layout=weight_layout,
                        use_per_token_sf_a=fc1_use_per_token_sf_a,
                        use_per_token_sf_b=fc1_use_per_token_sf_b,
                        per_token_sf_dtype=per_token_sf_dtype,
                        has_gemm1_alpha=has_gemm1_alpha,
                        has_gemm1_beta=has_gemm1_beta,
                        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
                    )
                ),
                prims_ts_gemm_config_index=fc1_json.global_index,
            ),
            fc2=PrimsTsMoeGemmConfig(
                cfg=_make_config_spec(
                    **_json_config_kwargs(
                        fc2_json,
                        sm_arch=sm_arch,
                        fc="fc2",
                        activation_type=activation_type,
                        has_bias=fc2_has_bias,
                        default_dtype_a=dtype_a,
                        default_dtype_b=dtype_b,
                        default_dtype_c=fc2_dtype_c,
                        weight_layout=weight_layout,
                        use_per_token_sf_a=fc2_use_per_token_sf_a,
                        use_per_token_sf_b=fc2_use_per_token_sf_b,
                        per_token_sf_dtype=per_token_sf_dtype,
                    )
                ),
                prims_ts_gemm_config_index=fc2_json.global_index,
            ),
        ),
        enable_pdl=enable_pdl,
    )
    if use_deepseek_fp8:
        # The tactic rows are block-agnostic; the scale block comes from the
        # call's scale shapes.
        pair = _with_spec_overrides_pair(
            pair, deepseek_fp8_block_size=deepseek_fp8_block_size
        )
    return pair


def _task_manager_smem_bytes(manager: object) -> int | None:
    allocator = getattr(manager, "_smem_allocator", None)
    if allocator is None:
        allocator = getattr(manager, "smem_allocator", None)
    if allocator is None:
        return None
    return int(allocator.total_smem_bytes) + int(
        getattr(allocator, "barrier_smem_bytes", 0)
    )


_TASK_MANAGER_VALIDATION_CACHE: dict[
    tuple[tuple[tuple[str, int], ...], int, int, bool], Exception | None
] = {}
_VALIDATION_CACHE_MISS = object()


def _validate_config_task_manager(
    kwargs: dict[str, int],
    top_k: int,
    num_experts: int,
) -> None:
    from ..batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    manager = build_batched_gemm_task_manager(
        num_experts=num_experts,
        # Validation does not use the resolved early-exit extent, so use one
        # canonical geometry rather than making validation token-count-specific.
        num_tokens=128,
        top_k=top_k,
        verbose=False,
        **kwargs,
    )
    _task_manager_smem_bytes(manager)


def _ensure_config_task_manager_buildable(
    spec: PrimsTsConfigSpec,
    *,
    dtype_label: str,
    fc: str,
    tile_n: int,
    moe_config_index: int,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_experts: int | None = None,
) -> None:
    validation_top_k = max(1, int(top_k or 1))
    validation_num_experts = max(validation_top_k, int(num_experts or 2))
    from ..batched_gemm.batched_gemm_kernel import (
        _exhaustive_deadlock_race_check_enabled,
    )

    # The validation schedule must not depend on num_tokens. The builder only
    # uses it to compute an early-exit extent whose result is discarded. Keep
    # validation modes in the key so tests and applications can toggle them.
    key = (
        tuple(sorted(spec.kwargs.items())),
        validation_top_k,
        validation_num_experts,
        _exhaustive_deadlock_race_check_enabled(),
    )
    cached = _TASK_MANAGER_VALIDATION_CACHE.get(key, _VALIDATION_CACHE_MISS)
    if cached is not _VALIDATION_CACHE_MISS:
        if cached is not None:
            raise cached.with_traceback(None)
        return

    try:
        _validate_config_task_manager(
            spec.kwargs,
            validation_top_k,
            validation_num_experts,
        )
    except Exception as exc:
        _TASK_MANAGER_VALIDATION_CACHE[key] = exc
        raise
    _TASK_MANAGER_VALIDATION_CACHE[key] = None


def _ensure_pair_buildable(
    pair: PrimsTsGemmPair,
    *,
    dtype_label: str,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_experts: int | None = None,
) -> None:
    _ensure_config_task_manager_buildable(
        pair.fc1.cfg,
        dtype_label=dtype_label,
        fc="FC1",
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
    )
    _ensure_config_task_manager_buildable(
        pair.fc2.cfg,
        dtype_label=dtype_label,
        fc="FC2",
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
    )


def _fallback_tile_ns_for_label(dtype_label: str, tile_n: int) -> tuple[int, ...]:
    supported_by_label = {
        "BF16": SUPPORTED_BF16_TILE_N,
        "NVFP4xNVFP4": SUPPORTED_NVFP4_TILE_N,
        "FP8 per-tensor": SUPPORTED_FP8_TILE_N,
        "MXFP4xMXFP8": SUPPORTED_MXFP4_MXFP8_TILE_N,
        "MXFP4xBF16": SUPPORTED_MXFP4_BF16_TILE_N,
        "MXFP8xMXFP8": SUPPORTED_MXFP8_MXFP8_TILE_N,
        "DeepSeek FP8": SUPPORTED_DSFP8_TILE_N,
    }
    supported_tiles = supported_by_label.get(dtype_label, (tile_n,))
    larger_tiles = tuple(tile for tile in supported_tiles if tile > tile_n)
    smaller_tiles = tuple(tile for tile in supported_tiles if tile < tile_n)
    return (tile_n, *larger_tiles, *smaller_tiles)


def _make_default_json_moe_config_pair(
    *,
    sm_arch: int,
    tile_n: int,
    activation_type: int,
    dtype_a: int,
    dtype_b: int,
    fc1_dtype_c: int,
    fc2_dtype_c: int,
    dtype_label: str,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    use_deepseek_fp8: bool = False,
    fc1_use_per_token_sf_a: bool | None = None,
    fc2_use_per_token_sf_a: bool | None = None,
    fc1_use_per_token_sf_b: bool | None = None,
    fc2_use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_experts: int | None = None,
    deepseek_fp8_block_size: int = 128,
    prefer_persistent_scheduler: bool = False,
) -> PrimsTsGemmPair | None:
    fc1_indices = _moe_json_passing_indices(
        sm_arch=sm_arch,
        tile_n=tile_n,
        activation_type=activation_type,
        fc="fc1",
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=fc1_dtype_c,
        use_deepseek_fp8=use_deepseek_fp8,
        use_per_token_sf_a=fc1_use_per_token_sf_a,
        use_per_token_sf_b=fc1_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    fc2_indices = _moe_json_passing_indices(
        sm_arch=sm_arch,
        tile_n=tile_n,
        activation_type=activation_type,
        fc="fc2",
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=fc2_dtype_c,
        use_deepseek_fp8=use_deepseek_fp8,
        use_per_token_sf_a=fc2_use_per_token_sf_a,
        use_per_token_sf_b=fc2_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    if not fc1_indices or not fc2_indices:
        return None
    moe_config_index = _default_moe_config_index(
        tile_n,
        sm_arch=sm_arch,
        prefer_persistent_scheduler=prefer_persistent_scheduler,
        activation_type=activation_type,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        fc1_dtype_c=fc1_dtype_c,
        fc2_dtype_c=fc2_dtype_c,
        dtype_label=dtype_label,
        use_deepseek_fp8=use_deepseek_fp8,
        fc1_use_per_token_sf_a=fc1_use_per_token_sf_a,
        fc2_use_per_token_sf_a=fc2_use_per_token_sf_a,
        fc1_use_per_token_sf_b=fc1_use_per_token_sf_b,
        fc2_use_per_token_sf_b=fc2_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
    )
    return _make_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        activation_type=activation_type,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        fc1_dtype_c=fc1_dtype_c,
        fc2_dtype_c=fc2_dtype_c,
        dtype_label=dtype_label,
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        use_deepseek_fp8=use_deepseek_fp8,
        fc1_use_per_token_sf_a=fc1_use_per_token_sf_a,
        fc2_use_per_token_sf_a=fc2_use_per_token_sf_a,
        fc1_use_per_token_sf_b=fc1_use_per_token_sf_b,
        fc2_use_per_token_sf_b=fc2_use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
        weight_layout=weight_layout,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        deepseek_fp8_block_size=deepseek_fp8_block_size,
    )


def _required_json_moe_config_pair(
    *,
    sm_arch: int | None = None,
    tile_n: int,
    moe_config_index: int,
    dtype_label: str,
    fallback_tile_ns: Sequence[int] | None = None,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_experts: int | None = None,
    num_global_experts: int | None = None,
    **kwargs,
) -> PrimsTsGemmPair:
    sm_arch = _resolve_moe_sm_arch(sm_arch)
    pair = None
    if moe_config_index == -1:
        # ``num_experts`` is the local expert count here (see the callers);
        # ``num_global_experts`` is the routed total.
        prefer_persistent = _is_expert_parallel(num_experts, num_global_experts)
        if fallback_tile_ns is None:
            fallback_tile_ns = _fallback_tile_ns_for_label(dtype_label, tile_n)
            if prefer_persistent:
                # The default tile was chosen to cover the expected rows per
                # expert; if the menu has no entry for it, try the nearest
                # narrower tiles before the widest or the narrowest ones.
                fallback_tile_ns = (
                    tile_n,
                    *sorted((t for t in fallback_tile_ns if t < tile_n), reverse=True),
                    *sorted(t for t in fallback_tile_ns if t > tile_n),
                )
        for candidate_tile_n in fallback_tile_ns:
            pair = _make_default_json_moe_config_pair(
                sm_arch=sm_arch,
                tile_n=candidate_tile_n,
                dtype_label=dtype_label,
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                prefer_persistent_scheduler=prefer_persistent,
                **kwargs,
            )
            if pair is not None:
                break
    else:
        pair = _make_json_moe_config_pair(
            sm_arch=sm_arch,
            tile_n=tile_n,
            moe_config_index=moe_config_index,
            dtype_label=dtype_label,
            **kwargs,
        )
    if pair is None:
        raise ValueError(
            "No buildable local Prims-TS MoE config found for "
            f"{dtype_label} tile_N={tile_n}, config_index={moe_config_index}"
        )
    _ensure_pair_buildable(
        pair,
        dtype_label=dtype_label,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
    )
    return pair


def _valid_json_moe_tactics(
    *,
    sm_arch: int | None,
    supported_tiles: Sequence[int],
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
    num_experts: int | None,
    activation_type: int,
    dtype_a: int,
    dtype_b: int,
    fc1_dtype_c: int,
    fc2_dtype_c: int,
    dtype_label: str,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    use_deepseek_fp8: bool = False,
    fc1_use_per_token_sf_a: bool | None = None,
    fc2_use_per_token_sf_a: bool | None = None,
    fc1_use_per_token_sf_b: bool | None = None,
    fc2_use_per_token_sf_b: bool | None = None,
    per_token_sf_dtype: int | None = None,
    weight_layout: int = int(WeightLayout.MajorK),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
) -> list[list[int]]:
    sm_arch = _resolve_moe_sm_arch(sm_arch)
    tactics: list[list[int]] = []
    for tile_n in _selected_tile_ns(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=supported_tiles,
    ):
        fc1_indices = _moe_json_passing_indices(
            sm_arch=sm_arch,
            tile_n=tile_n,
            activation_type=activation_type,
            fc="fc1",
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=fc1_dtype_c,
            use_deepseek_fp8=use_deepseek_fp8,
            use_per_token_sf_a=fc1_use_per_token_sf_a,
            use_per_token_sf_b=fc1_use_per_token_sf_b,
            per_token_sf_dtype=per_token_sf_dtype,
            weight_layout=weight_layout,
        )
        fc2_indices = _moe_json_passing_indices(
            sm_arch=sm_arch,
            tile_n=tile_n,
            activation_type=activation_type,
            fc="fc2",
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=fc2_dtype_c,
            use_deepseek_fp8=use_deepseek_fp8,
            use_per_token_sf_a=fc2_use_per_token_sf_a,
            use_per_token_sf_b=fc2_use_per_token_sf_b,
            per_token_sf_dtype=per_token_sf_dtype,
            weight_layout=weight_layout,
        )
        pair_count = sum(
            _same_batch_mode(a, b, sm_arch=sm_arch)
            for a in fc1_indices
            for b in fc2_indices
        )
        for moe_config_index in range(pair_count):
            if dtype_label == "BF16" and (fc1_has_bias or fc2_has_bias):
                pair = _resolve_moe_json_config_pair(
                    tile_n,
                    moe_config_index,
                    sm_arch=sm_arch,
                    activation_type=activation_type,
                    dtype_a=dtype_a,
                    dtype_b=dtype_b,
                    fc1_dtype_c=fc1_dtype_c,
                    fc2_dtype_c=fc2_dtype_c,
                    dtype_label=dtype_label,
                    weight_layout=weight_layout,
                )
                if pair is not None and not _bool_value(
                    pair[0].options.get("transpose_mma_output", True)
                ):
                    continue
            tactics.append([tile_n, moe_config_index])
    return tactics


def valid_prims_ts_bf16_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_BF16_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.BF16),
        dtype_b=int(_DType.BF16),
        fc1_dtype_c=int(_DType.BF16),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="BF16",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_nvfp4_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    use_per_token_sf_b: bool = False,
    per_token_sf_dtype: int = int(_DType.FP32),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_NVFP4_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.E2M1),
        dtype_b=int(_DType.E2M1),
        fc1_dtype_c=int(_DType.E2M1),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="NVFP4xNVFP4",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        fc1_use_per_token_sf_b=use_per_token_sf_b,
        fc2_use_per_token_sf_b=False,
        per_token_sf_dtype=per_token_sf_dtype,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_fp8_per_tensor_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    fc1_use_per_token_sf_a: bool = False,
    fc2_use_per_token_sf_a: bool = False,
    use_per_token_sf_b: bool = False,
    per_token_sf_dtype: int = int(_DType.FP32),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=_fp8_per_tensor_supported_tiles(
            use_per_token_sf_b=use_per_token_sf_b
        ),
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.E4M3),
        dtype_b=int(_DType.E4M3),
        fc1_dtype_c=int(_DType.E4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="FP8 per-tensor",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        fc1_use_per_token_sf_a=fc1_use_per_token_sf_a,
        fc2_use_per_token_sf_a=fc2_use_per_token_sf_a,
        fc1_use_per_token_sf_b=use_per_token_sf_b,
        fc2_use_per_token_sf_b=False,
        per_token_sf_dtype=per_token_sf_dtype,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_mxfp4_mxfp8_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_MXFP4_MXFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE2M1),
        dtype_b=int(_DType.MXE4M3),
        fc1_dtype_c=int(_DType.MXE4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP4xMXFP8",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_mxfp4_bf16_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_MXFP4_BF16_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE2M1),
        dtype_b=int(_DType.BF16),
        fc1_dtype_c=int(_DType.BF16),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP4xBF16",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_mxfp8_mxfp8_moe_tactics(
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_MXFP8_MXFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE4M3),
        dtype_b=int(_DType.MXE4M3),
        fc1_dtype_c=int(_DType.MXE4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP8xMXFP8",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def valid_prims_ts_deepseek_fp8_moe_tactics(
    *,
    sm_arch: int | None = None,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        sm_arch=sm_arch,
        supported_tiles=SUPPORTED_DSFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        activation_type=int(ActivationType.Swiglu),
        dtype_a=int(_DType.E4M3),
        dtype_b=int(_DType.E4M3),
        fc1_dtype_c=int(_DType.E4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="DeepSeek FP8",
        use_deepseek_fp8=True,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_bf16_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for a Prims-TS BF16 MoE tactic."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_BF16_TILE_N,
    )
    if tile_n not in SUPPORTED_BF16_TILE_N:
        raise ValueError(f"Unsupported Prims-TS BF16 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.BF16),
        dtype_b=int(_DType.BF16),
        fc1_dtype_c=int(_DType.BF16),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="BF16",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_nvfp4_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    use_per_token_sf_b: bool = False,
    per_token_sf_dtype: int = int(_DType.FP32),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for a Prims-TS NVFP4xNVFP4 MoE tactic."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_NVFP4_TILE_N,
    )
    if tile_n not in SUPPORTED_NVFP4_TILE_N:
        raise ValueError(f"Unsupported Prims-TS NVFP4 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.E2M1),
        dtype_b=int(_DType.E2M1),
        fc1_dtype_c=int(_DType.E2M1),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="NVFP4xNVFP4",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        fc1_use_per_token_sf_b=use_per_token_sf_b,
        fc2_use_per_token_sf_b=False,
        per_token_sf_dtype=per_token_sf_dtype,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_fp8_per_tensor_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    fc1_use_per_token_sf_a: bool = False,
    fc2_use_per_token_sf_a: bool = False,
    use_per_token_sf_b: bool = False,
    per_token_sf_dtype: int = int(_DType.FP32),
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS FP8 per-tensor MoE."""

    supported_tiles = _fp8_per_tensor_supported_tiles(
        use_per_token_sf_b=use_per_token_sf_b
    )
    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=supported_tiles,
    )
    if tile_n not in supported_tiles:
        raise ValueError(f"Unsupported Prims-TS FP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.E4M3),
        dtype_b=int(_DType.E4M3),
        fc1_dtype_c=int(_DType.E4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="FP8 per-tensor",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        fc1_use_per_token_sf_a=fc1_use_per_token_sf_a,
        fc2_use_per_token_sf_a=fc2_use_per_token_sf_a,
        fc1_use_per_token_sf_b=use_per_token_sf_b,
        fc2_use_per_token_sf_b=False,
        per_token_sf_dtype=per_token_sf_dtype,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_mxfp4_mxfp8_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS MXFP4xMXFP8 MoE."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_MXFP4_MXFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP4_MXFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP4xMXFP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE2M1),
        dtype_b=int(_DType.MXE4M3),
        fc1_dtype_c=int(_DType.MXE4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP4xMXFP8",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_mxfp4_bf16_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS MXFP4xBF16 MoE."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_MXFP4_BF16_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP4_BF16_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP4xBF16 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE2M1),
        dtype_b=int(_DType.BF16),
        fc1_dtype_c=int(_DType.BF16),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP4xBF16",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_mxfp8_mxfp8_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS MXFP8xMXFP8 MoE."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_MXFP8_MXFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP8_MXFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP8xMXFP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=activation_type,
        dtype_a=int(_DType.MXE4M3),
        dtype_b=int(_DType.MXE4M3),
        fc1_dtype_c=int(_DType.MXE4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="MXFP8xMXFP8",
        fc1_has_bias=fc1_has_bias,
        fc2_has_bias=fc2_has_bias,
        has_gemm1_alpha=has_gemm1_alpha,
        has_gemm1_beta=has_gemm1_beta,
        has_gemm1_clamp_limit=has_gemm1_clamp_limit,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
    )


def map_trtllm_deepseek_fp8_moe_tactic(
    tactic: int | Sequence[int],
    *,
    sm_arch: int | None = None,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    num_experts: int | None = None,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
    deepseek_fp8_block_size: int = 128,
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS DeepSeek FP8 MoE.

    ``deepseek_fp8_block_size`` is the side of the square FP32 scale block,
    ``128`` (the DeepSeek checkpoint layout) or ``32`` (vLLM's re-blocked
    TP shards).
    """

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        supported_tiles=SUPPORTED_DSFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_DSFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS DeepSeek FP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        sm_arch=sm_arch,
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
        num_global_experts=num_experts,
        activation_type=int(ActivationType.Swiglu),
        dtype_a=int(_DType.E4M3),
        dtype_b=int(_DType.E4M3),
        fc1_dtype_c=int(_DType.E4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label="DeepSeek FP8",
        use_deepseek_fp8=True,
        enable_pdl=enable_pdl,
        weight_layout=weight_layout,
        deepseek_fp8_block_size=deepseek_fp8_block_size,
    )
