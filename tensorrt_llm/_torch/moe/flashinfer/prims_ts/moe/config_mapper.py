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
from enum import IntEnum
from functools import lru_cache
from importlib import resources
import math
from typing import Sequence

from tensorrt_llm._torch.moe.flashinfer.tllm_enums import ActivationType, WeightLayout

SUPPORTED_BF16_TILE_N = (8, 16, 32, 64, 128, 256)
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


class _BatchMode(IntEnum):
    BATCH_N = 0


class _RouteImpl(IntEnum):
    NONE = 0
    TMA = 1
    LDGSTS = 2
    LDG_PLUS_STS = 3


class _TileScheduler(IntEnum):
    STATIC = 0
    PERSISTENT = 1


class _ActKind(IntEnum):
    NONE = 0
    SWIGLU = 1
    GEGLU = 2
    RELU2 = 3
    SILU = 4
    SITU = 5


class _DType(IntEnum):
    FP32 = 1
    FP16 = 2
    BF16 = 3
    E2M1 = 4
    MXE2M1 = 5
    MXE4M3 = 6
    E4M3 = 7


class _SfLayout(IntEnum):
    R8c4 = 1
    LINEAR = 2
    R128c4 = 3


class _BiasType(IntEnum):
    NONE = 0
    M = 1


@dataclass(frozen=True)
class PrimsTsConfigSpec:
    """Lazy config spec that can materialize a full BatchedGemmConfig."""

    kwargs: dict[str, int]

    def build(self) -> object:
        from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config

        return make_config(**self.kwargs)


@dataclass(frozen=True)
class PrimsTsMoeGemmConfig:
    """One Prims-TS GEMM config plus the local config row it came from."""

    cfg: object
    prims_ts_gemm_config_index: int

    @property
    def trtllm_gemm_config_index(self) -> int:
        """Temporary compatibility alias for tests and callers mid-refactor."""
        return self.prims_ts_gemm_config_index


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
        "cluster_m",
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
        "gather_regs",
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
        "num_epilogue_warps",
        "num_load_a_warps",
        "num_load_b_warps",
        "num_load_sfa_warps",
        "num_load_sfab_warps",
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
        "sf_block_size_a",
        "sf_block_size_b",
        "sf_block_size_c",
        "sf_layout_a",
        "sf_layout_b",
        "sf_layout_c",
        "tile_k",
        "tile_m",
        "tile_n",
        "tile_scheduler",
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
        "use_two_tma_load_warps",
        "use_unroll_loop_2x_for_mma",
        "use_work_throttle",
        "weight_layout",
        "workid_regs",
        "do_pdl_wait_for_num_non_exiting_ctas",
    }
)


def _next_power_of_two(value: float) -> int:
    n = int(math.ceil(value))
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _selected_tile_ns(
    *,
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> tuple[int, ...]:
    if num_tokens is None or top_k is None or num_local_experts is None:
        return (supported_tiles[-1],)
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    avg_tokens_per_expert = float(num_tokens * top_k) / float(num_local_experts)
    center_tile = min(
        max(_next_power_of_two(avg_tokens_per_expert), supported_tiles[0]),
        supported_tiles[-1],
    )
    center_idx = supported_tiles.index(center_tile)
    selected_tiles = {center_tile}
    for offset in (1, 2, 3):
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
    return tuple(sorted(selected_tiles))


def _fp8_per_tensor_supported_tiles(*, use_per_token_sf_b: bool) -> tuple[int, ...]:
    del use_per_token_sf_b
    return SUPPORTED_FP8_TILE_N


def _default_tile_n(
    *,
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> int:
    return _selected_tile_ns(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        supported_tiles=supported_tiles,
    )[0]


def _parse_tactic(
    tactic: int | Sequence[int],
    *,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    supported_tiles: Sequence[int] = SUPPORTED_BF16_TILE_N,
) -> tuple[int, int]:
    if tactic == -1:
        return (
            _default_tile_n(
                num_tokens=num_tokens,
                top_k=top_k,
                num_local_experts=num_local_experts,
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
        if lowered == "ldgplussts":
            return int(_RouteImpl.LDG_PLUS_STS)
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
        if act == "silu":
            return int(_ActKind.SILU)
    eltwise_act = str(options.get("eltwise_act_type", "none")).lower()
    if eltwise_act == "relu2":
        return int(_ActKind.RELU2)
    return int(_ActKind.NONE)


def _activation_json_fused_act(activation_type: int) -> bool:
    return int(activation_type) in (
        int(ActivationType.Swiglu),
        int(ActivationType.Geglu),
        int(ActivationType.Silu),
        int(ActivationType.Situ),
    )


def _activation_json_act(activation_type: int) -> str:
    if int(activation_type) in (
        int(ActivationType.Swiglu),
        int(ActivationType.Situ),
    ):
        return "swiglu"
    if int(activation_type) == int(ActivationType.Geglu):
        return "geglu"
    if int(activation_type) == int(ActivationType.Silu):
        return "silu"
    return "none"


def _activation_json_eltwise_act(activation_type: int) -> str:
    if int(activation_type) == int(ActivationType.Relu2):
        return "relu2"
    return "none"


def _prims_ts_config_resource():
    config = resources.files(__package__).joinpath("prims_ts_moe_configs.json")
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


@lru_cache(maxsize=1)
def _expanded_prims_ts_json_configs() -> tuple[_JsonBatchedGemmConfig, ...]:
    config = _prims_ts_config_resource()
    data = json.loads(config.read_text(encoding="utf-8"))
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


def _expanded_trtllm_gen_json_configs() -> tuple[_JsonBatchedGemmConfig, ...]:
    """Temporary compatibility alias for tests during the Prims-TS refactor."""
    return _expanded_prims_ts_json_configs()


def _json_config_by_global_index(global_index: int) -> _JsonBatchedGemmConfig:
    configs = _expanded_prims_ts_json_configs()
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
    if not _bool_value(options.get("transpose_mma_output", True)):
        return False
    if int(options.get("tile_n", -1)) != int(tile_n):
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
        for cfg in _expanded_prims_ts_json_configs()
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


def _resolve_moe_json_config_pair(
    tile_n: int,
    moe_config_index: int,
    *,
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
) -> tuple[_JsonBatchedGemmConfig, _JsonBatchedGemmConfig] | None:
    if moe_config_index < 0:
        return None
    fc1_indices = _moe_json_passing_indices(
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
        (fc1_index, fc2_index) for fc1_index in fc1_indices for fc2_index in fc2_indices
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
                        _json_config_by_global_index(pair[0]).options.get(
                            "use_clc_fast_drain", False
                        )
                    )
                    or _bool_value(
                        _json_config_by_global_index(pair[1]).options.get(
                            "use_clc_fast_drain", False
                        )
                    )
                ),
                int(
                    _bool_value(
                        _json_config_by_global_index(pair[0]).options.get(
                            "use_work_throttle", False
                        )
                    )
                    or _bool_value(
                        _json_config_by_global_index(pair[1]).options.get(
                            "use_work_throttle", False
                        )
                    )
                ),
            )
        )
    else:
        # Appended union variants must not reinterpret any persisted baseline
        # pair index. Keep the complete original FC1 x FC2 product first and
        # place every pair involving a new scheduler variant afterward.
        config_pairs.sort(
            key=lambda pair: int(
                _json_config_by_global_index(pair[0]).is_scheduler_union_variant
                or _json_config_by_global_index(pair[1]).is_scheduler_union_variant
            )
        )
    total = len(config_pairs)
    if moe_config_index >= total:
        raise ValueError(
            f"Unsupported MoE config index={moe_config_index}; valid range is "
            f"[0, {total - 1}] for {dtype_label} tile_N={tile_n}"
        )
    fc1_index, fc2_index = config_pairs[moe_config_index]
    return _json_config_by_global_index(fc1_index), _json_config_by_global_index(
        fc2_index
    )


def _ensure_json_config_supported_by_ts(cfg: _JsonBatchedGemmConfig) -> None:
    _validate_json_options(cfg, cfg.options)


def _json_config_kwargs(
    cfg: _JsonBatchedGemmConfig,
    *,
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
    use_max_tmem_overlap = int(_bool_value(options.get("use_max_tmem_overlap", False)))
    tmem_acc_stages = 1 if use_max_tmem_overlap else int(options["num_stages_tmem_acc"])
    mma_regs = int(options.get("mma_regs", 48))
    load_regs = int(options.get("load_regs", mma_regs))
    load_sf_regs = int(options.get("load_sf_regs", load_regs))
    copy_sf_regs = int(options.get("copy_sf_regs", load_regs))
    padding_regs = int(options.get("padding_regs", load_regs))
    workid_regs = int(options.get("workid_regs", load_regs))
    gather_regs = int(options.get("gather_regs", load_regs))
    epilogue_regs = int(options.get("epilogue_regs", 160))
    load_a_regs = int(options.get("load_a_regs", load_regs))
    load_b_regs = int(options.get("load_b_regs", load_regs))
    load_sfa_regs = int(options.get("load_sfa_regs", load_sf_regs))
    load_sfb_regs = int(options.get("load_sfb_regs", load_sf_regs))
    json_use_per_token_sf_a = _bool_value(options.get("use_per_token_sf_a", False))
    json_use_per_token_sf_b = _bool_value(options.get("use_per_token_sf_b", False))
    if use_per_token_sf_a is None:
        use_per_token_sf_a = json_use_per_token_sf_a
    use_global_scales = _bool_value(options.get("use_global_scales", False))
    if use_per_token_sf_b is None:
        use_per_token_sf_b = json_use_per_token_sf_b
    kwargs = {
        "batch_mode": int(_BatchMode.BATCH_N),
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
        "sf_bits": 8,
        "sf_block_size_a": int(options.get("sf_block_size_a", 0)),
        "sf_block_size_b": int(options.get("sf_block_size_b", 0)),
        "sf_block_size_c": int(options.get("sf_block_size_c", 0)),
        "mma_k": int(options.get("mma_k", 32)),
        "route_act": _json_route_value(options.get("route_act", False)),
        "route_sfs_act": _json_route_value(options.get("route_sfs_act", False)),
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
        "cluster_m": int(options.get("cluster_m", 1)),
        "tile_m": int(options.get("tile_m", 128)),
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
        "use_two_tma_load_warps": int(
            _bool_value(options.get("use_two_tma_load_warps", True))
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
        "gather_regs": gather_regs,
    }
    if "num_stages_workid" in options:
        kwargs["num_stages_workid"] = int(options["num_stages_workid"])
    if "num_stages_c_smem" in options:
        kwargs["num_stages_c_smem"] = int(options["num_stages_c_smem"])
    if "use_pdl" in options:
        kwargs["use_pdl"] = int(_bool_value(options["use_pdl"]))
    if "do_pdl_wait_for_num_non_exiting_ctas" in options:
        kwargs["do_pdl_wait_for_num_non_exiting_ctas"] = int(
            _bool_value(options["do_pdl_wait_for_num_non_exiting_ctas"])
        )
    if _bool_value(options.get("use_deepseek_fp8", False)):
        kwargs["use_deepseek_fp8"] = 1
        kwargs["num_load_sfab_warps"] = 1
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
    if (kwargs["tile_m"] * kwargs["cluster_m"]) % kwargs["mma_m"] != 0:
        raise ValueError(
            "Prims-TS requires cluster-wide tile_m to be a multiple of mma_m "
            f"for batched GEMM config {cfg.global_index}"
        )
    return kwargs


def _make_config_spec(**kwargs: int) -> PrimsTsConfigSpec:
    return PrimsTsConfigSpec(kwargs=kwargs)


def _pdl_overrides(kwargs: dict[str, int], *, enable_pdl: bool) -> dict[str, int]:
    use_pdl = int(bool(enable_pdl))
    return {
        "use_pdl": use_pdl,
        "do_pdl_wait_for_num_non_exiting_ctas": int(
            bool(use_pdl) and bool(kwargs.get("use_early_exit", 1))
        ),
    }


def _with_pdl_spec(spec: PrimsTsConfigSpec, *, enable_pdl: bool) -> PrimsTsConfigSpec:
    kwargs = dict(spec.kwargs)
    kwargs.update(_pdl_overrides(kwargs, enable_pdl=enable_pdl))
    return PrimsTsConfigSpec(kwargs=kwargs)


def _with_pdl_pair(pair: PrimsTsGemmPair, *, enable_pdl: bool) -> PrimsTsGemmPair:
    return PrimsTsGemmPair(
        tile_n=pair.tile_n,
        moe_config_index=pair.moe_config_index,
        fc1=PrimsTsMoeGemmConfig(
            cfg=_with_pdl_spec(pair.fc1.cfg, enable_pdl=enable_pdl),
            prims_ts_gemm_config_index=pair.fc1.prims_ts_gemm_config_index,
        ),
        fc2=PrimsTsMoeGemmConfig(
            cfg=_with_pdl_spec(pair.fc2.cfg, enable_pdl=enable_pdl),
            prims_ts_gemm_config_index=pair.fc2.prims_ts_gemm_config_index,
        ),
    )


def _make_json_moe_config_pair(
    *,
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
) -> PrimsTsGemmPair | None:
    json_pair = _resolve_moe_json_config_pair(
        tile_n,
        moe_config_index,
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
    return _with_pdl_pair(
        PrimsTsGemmPair(
            tile_n=tile_n,
            moe_config_index=moe_config_index,
            fc1=PrimsTsMoeGemmConfig(
                cfg=_make_config_spec(
                    **_json_config_kwargs(
                        fc1_json,
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


def _task_manager_smem_bytes(manager: object) -> int | None:
    allocator = getattr(manager, "_smem_allocator", None)
    if allocator is None:
        allocator = getattr(manager, "smem_allocator", None)
    if allocator is None:
        return None
    return int(allocator.total_smem_bytes) + int(
        getattr(allocator, "barrier_smem_bytes", 0)
    )


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
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    validation_top_k = max(1, int(top_k or 1))
    validation_num_experts = max(validation_top_k, int(num_experts or 2))
    manager = build_batched_gemm_task_manager(
        num_experts=validation_num_experts,
        num_tokens=max(1, int(num_tokens or 128)),
        top_k=validation_top_k,
        verbose=False,
        **spec.kwargs,
    )
    _task_manager_smem_bytes(manager)


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
) -> PrimsTsGemmPair | None:
    fc1_indices = _moe_json_passing_indices(
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
    return _make_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=0,
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
    )


def _required_json_moe_config_pair(
    *,
    tile_n: int,
    moe_config_index: int,
    dtype_label: str,
    fallback_tile_ns: Sequence[int] | None = None,
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_experts: int | None = None,
    **kwargs,
) -> PrimsTsGemmPair:
    pair = None
    if moe_config_index == -1:
        for candidate_tile_n in fallback_tile_ns or _fallback_tile_ns_for_label(
            dtype_label, tile_n
        ):
            pair = _make_default_json_moe_config_pair(
                tile_n=candidate_tile_n,
                dtype_label=dtype_label,
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                **kwargs,
            )
            if pair is not None:
                break
    else:
        pair = _make_json_moe_config_pair(
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
    supported_tiles: Sequence[int],
    num_tokens: int | None,
    top_k: int | None,
    num_local_experts: int | None,
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
    tactics: list[list[int]] = []
    for tile_n in _selected_tile_ns(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        supported_tiles=supported_tiles,
    ):
        fc1_indices = _moe_json_passing_indices(
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
        for moe_config_index in range(len(fc1_indices) * len(fc2_indices)):
            tactics.append([tile_n, moe_config_index])
    return tactics


def valid_prims_ts_bf16_moe_tactics(
    *,
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        supported_tiles=SUPPORTED_BF16_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_NVFP4_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=_fp8_per_tensor_supported_tiles(
            use_per_token_sf_b=use_per_token_sf_b
        ),
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        supported_tiles=SUPPORTED_MXFP4_MXFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        supported_tiles=SUPPORTED_MXFP4_BF16_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    fc1_has_bias: bool = False,
    fc2_has_bias: bool = False,
    has_gemm1_alpha: bool = False,
    has_gemm1_beta: bool = False,
    has_gemm1_clamp_limit: bool = False,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        supported_tiles=SUPPORTED_MXFP8_MXFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> list[list[int]]:
    return _valid_json_moe_tactics(
        supported_tiles=SUPPORTED_DSFP8_TILE_N,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_BF16_TILE_N,
    )
    if tile_n not in SUPPORTED_BF16_TILE_N:
        raise ValueError(f"Unsupported Prims-TS BF16 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_NVFP4_TILE_N,
    )
    if tile_n not in SUPPORTED_NVFP4_TILE_N:
        raise ValueError(f"Unsupported Prims-TS NVFP4 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=supported_tiles,
    )
    if tile_n not in supported_tiles:
        raise ValueError(f"Unsupported Prims-TS FP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_MXFP4_MXFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP4_MXFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP4xMXFP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_MXFP4_BF16_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP4_BF16_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP4xBF16 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    activation_type: int = int(ActivationType.Swiglu),
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
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
        supported_tiles=SUPPORTED_MXFP8_MXFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_MXFP8_MXFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS MXFP8xMXFP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
    num_tokens: int | None = None,
    top_k: int | None = None,
    num_local_experts: int | None = None,
    enable_pdl: bool = False,
    weight_layout: int = int(WeightLayout.MajorK),
) -> PrimsTsGemmPair:
    """Return Prims-TS FC1/FC2 configs for Prims-TS DeepSeek FP8 MoE."""

    tile_n, moe_config_index = _parse_tactic(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        supported_tiles=SUPPORTED_DSFP8_TILE_N,
    )
    if tile_n not in SUPPORTED_DSFP8_TILE_N:
        raise ValueError(f"Unsupported Prims-TS DeepSeek FP8 tile_N={tile_n}")
    if moe_config_index < -1:
        raise ValueError(f"Unsupported MoE config index={moe_config_index}")

    return _required_json_moe_config_pair(
        tile_n=tile_n,
        moe_config_index=moe_config_index,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_local_experts,
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
