# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Search-space expansion and candidate pruning for bench_moe."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from tensorrt_llm._utils import local_mpi_size
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .backend import MoeBackendType, get_backend_class
from .mapping import _PARALLEL_MODE_LAYOUTS, _resolve_mapping_layout
from .specs import _ALL_BACKENDS, _FORCED_COMM_ENV_VALUES, ConfigSpec, ModelSpec, SearchSpec

_FUSED_COMM_BACKENDS = frozenset({"MEGAMOE_DEEPGEMM"})


def _is_deepep_feasible(num_ranks: int) -> bool:
    """Return True if DeepEP supports the given EP rank count on this node topology.

    Intranode: num_ranks in {2, 4, 8} and num_ranks == local_mpi_size().
    Internode: exactly 8 ranks per node, with 2/4/8/16 RDMA nodes.
    Mirrors the feasibility check in fused_moe_wide_ep.py::select_alltoall_method_type.
    """
    _INTRANODE_RANKS = {2, 4, 8}
    _REQUIRED_LOCAL_SIZE = 8
    _INTERNODE_RDMA_NODES = {2, 4, 8, 16}
    mpi_size = local_mpi_size()
    if num_ranks == mpi_size and num_ranks in _INTRANODE_RANKS:
        return True
    if mpi_size != _REQUIRED_LOCAL_SIZE:
        return False
    return (num_ranks // mpi_size) in _INTERNODE_RDMA_NODES


def _check_backend_can_implement(
    backend_str: str,
    quant_algo: Optional[QuantAlgo],
    dtype_activation: torch.dtype,
    swiglu_gptoss_style: bool,
) -> Tuple[bool, Optional[str]]:
    """Resolve backend_str to its MoE class and forward to can_implement."""
    try:
        backend_cls = get_backend_class(MoeBackendType(backend_str.upper()))
    except (ImportError, KeyError, RuntimeError, ValueError) as exc:
        return False, f"unknown MoE backend {backend_str!r}: {exc}"
    try:
        return backend_cls.can_implement(
            quant_algo=quant_algo,
            dtype_activation=dtype_activation,
            swiglu_gptoss_style=swiglu_gptoss_style,
        )
    except Exception as exc:
        return False, (f"{backend_cls.__name__}.can_implement raised {type(exc).__name__}: {exc}")


def _expand_axis(values: Iterable[Any], default: Any) -> Tuple[Any, ...]:
    out = tuple(values)
    return out if out else (default,)


def _comm_axis_for_backend(backend: Any, comm_methods: Tuple[Any, ...]) -> Tuple[Any, ...]:
    if str(backend).upper() in _FUSED_COMM_BACKENDS:
        return ("NONE",)
    return comm_methods


def _comm_axis_for_parallel_mode(pmode: str, comm_methods: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Collapse comm axis to AUTO for parallel modes without attention DP.

    Non-AUTO forced comm methods require enable_attention_dp=True (see
    is_candidate_valid). TEP and TTP have enable_dp=False, so only AUTO
    is ever valid for them. Generating forced-comm candidates for these
    modes only produces prune rows — handle it at generation time instead.
    CUSTOM mode is passed through unchanged (validated separately).
    """
    layout = _PARALLEL_MODE_LAYOUTS.get(str(pmode).upper())
    if layout is None:
        return comm_methods  # CUSTOM: unknown layout, keep as-is
    if not layout["enable_attention_dp"]:
        return ("AUTO",)
    return comm_methods


def expand_search(
    base_config: ConfigSpec,
    search: SearchSpec,
    world_size: int,
) -> List[ConfigSpec]:
    """Cartesian-product candidate generation, then explicit pruning.

    ``base_config`` carries the *non-search* fields (cuda_graph default,
    combine precision default). Search axes
    explicitly listed on ``search`` override the base values.
    """
    backends = _expand_axis(search.backends, base_config.backend)
    parallel_modes = _expand_axis(search.parallel_modes, base_config.parallel_mode)
    comm_methods = _expand_axis(search.comm_methods, base_config.comm_method)
    cuda_graph_options = _expand_axis(search.cuda_graph_options, base_config.cuda_graph)
    combine_options = _expand_axis(
        search.combine_precision_options, base_config.use_low_precision_moe_combine
    )

    candidates: List[ConfigSpec] = []
    for backend, pmode, cgraph, combine in itertools.product(
        backends, parallel_modes, cuda_graph_options, combine_options
    ):
        effective_comm = _comm_axis_for_backend(backend, comm_methods)
        # For non-fused backends apply parallel-mode comm constraint at
        # generation time so TEP/TTP always get comm=AUTO instead of
        # generating forced-comm candidates that are immediately pruned.
        if effective_comm != ("NONE",):
            effective_comm = _comm_axis_for_parallel_mode(pmode, effective_comm)
        for comm in effective_comm:
            candidate = replace(
                base_config,
                backend=str(backend).upper(),
                parallel_mode=str(pmode).upper(),
                comm_method=str(comm).upper(),
                cuda_graph=bool(cgraph),
                use_low_precision_moe_combine=bool(combine),
            )
            candidates.append(candidate)
    return candidates


def is_candidate_valid(
    config: ConfigSpec,
    model: ModelSpec,
    world_size: int,
    act_dtype: torch.dtype,
) -> Tuple[bool, Optional[str]]:
    """Return ``(ok, reason)`` based on backend / mapping / comm gates."""
    # Backend can_implement gate.
    ok, reason = _check_backend_can_implement(
        config.backend, model.quant_algo_enum, act_dtype, model.swiglu_gptoss_style
    )
    if not ok:
        return False, reason

    # Mapping layout gate.
    try:
        moe_ep, moe_tp, enable_dp = _resolve_mapping_layout(config, world_size)
    except ValueError as exc:
        return False, str(exc)

    # DenseGEMM only supports TP; any EP configuration (TEP, DEP, custom ep>1) is unsupported.
    if config.backend.upper() == "DENSEGEMM" and moe_ep > 1:
        return False, (
            f"DENSEGEMM does not support EP (ep_size={moe_ep}); "
            "use TEP/DEP only with other backends"
        )

    # MegaMoEDeepGemm is EP-only (asserts moe_tp_size == 1 in __init__); DTP/TTP are invalid.
    if config.backend.upper() == "MEGAMOE_DEEPGEMM" and moe_tp > 1:
        return False, (
            f"MEGAMOE_DEEPGEMM does not support MoE-TP (moe_tp_size={moe_tp}); "
            "use DEP/TEP modes only"
        )

    # DENSEGEMM DTP: FC2 kernel requires (intermediate_size / moe_tp_size) % 256 == 0.
    # DENSEGEMM __init__ only checks the full intermediate_size, so a model like
    # DeepSeek V3 (intermediate_size=2048, 2048%256=0) passes __init__ but fails
    # at runtime with moe_tp_size=16 (2048/16=128, 128%256!=0).
    if config.backend.upper() == "DENSEGEMM" and moe_ep == 1 and moe_tp > 1:
        if model.intermediate_size % moe_tp != 0:
            return False, (
                f"DENSEGEMM DTP: intermediate_size={model.intermediate_size} "
                f"not divisible by moe_tp_size={moe_tp}"
            )
        per_tp_k = model.intermediate_size // moe_tp
        _DENSEGEMM_MMA_TILE_K = 256
        if per_tp_k % _DENSEGEMM_MMA_TILE_K != 0:
            return False, (
                f"DENSEGEMM DTP moe_tp_size={moe_tp}: intermediate_size/tp={per_tp_k} "
                f"not aligned to FC2 MMA tile-K={_DENSEGEMM_MMA_TILE_K}"
            )

    # NVFP4 on CuteDSL / TRTLLM-Gen requires the per-partition intermediate size
    # (intermediate_size / moe_tp_size) to be a multiple of the NVFP4 weight
    # alignment (128). Unlike CUTLASS (which pads intermediate_size_per_partition
    # up to 128), these backends use the unpadded logical size when laying out the
    # block-scale tensor and fail during weight load: CUTEDSL raises a reshape
    # RuntimeError (e.g. "shape '[-1, 192, 448]' is invalid for input of size
    # 114688" — 192 padded to 256) and TRTLLM-Gen hits `assert intermediate_size %
    # weight_alignment == 0`. Prune the unsupported combo with a clear reason
    # instead of letting it crash mid-sweep. Example: DeepSeek-V4-Pro
    # (intermediate_size=3072) at moe_tp_size=32 -> 3072/32=96, 96%128!=0.
    if (
        config.backend.upper() in ("CUTEDSL", "TRTLLM")
        and model.quant_algo_enum == QuantAlgo.NVFP4
        and moe_tp > 1
    ):
        _NVFP4_WEIGHT_ALIGNMENT = 128
        if model.intermediate_size % moe_tp != 0:
            return False, (
                f"{config.backend.upper()} NVFP4: intermediate_size="
                f"{model.intermediate_size} not divisible by moe_tp_size={moe_tp}"
            )
        per_tp_k = model.intermediate_size // moe_tp
        if per_tp_k % _NVFP4_WEIGHT_ALIGNMENT != 0:
            return False, (
                f"{config.backend.upper()} NVFP4 moe_tp_size={moe_tp}: "
                f"intermediate_size/tp={per_tp_k} not aligned to NVFP4 weight "
                f"alignment={_NVFP4_WEIGHT_ALIGNMENT} (CUTLASS pads to 128, "
                f"CUTEDSL/TRTLLM do not)"
            )

    # Forced communication on non-DP / MoE-TP paths.
    forced = config.comm_method.upper()
    if forced not in ("AUTO", "NONE"):
        if not enable_dp:
            return False, f"comm_method={forced} requires enable_attention_dp=True"
        if moe_tp != 1 and forced != "ALLGATHER":
            return False, f"comm_method={forced} requires moe_tp_size=1 (got {moe_tp})"
        if world_size == 1:
            return False, f"comm_method={forced} has no effect at world_size=1"
        if forced == "DEEPEP" and not _is_deepep_feasible(moe_ep):
            return False, (
                f"comm_method={forced}: moe_ep_size={moe_ep} not supported by DeepEP topology "
                f"(local_mpi_size={local_mpi_size()}; supported: intranode {{2,4,8}}, "
                f"internode 8-ranks/node x {{2,4,8,16}} nodes)"
            )

    return True, None


def expand_and_prune(
    base_config: ConfigSpec,
    search: SearchSpec,
    model: ModelSpec,
    world_size: int,
    act_dtype: torch.dtype,
    max_configs: Optional[int] = None,
) -> Tuple[List[ConfigSpec], Dict[ConfigSpec, str]]:
    """Expand search and split into ``(valid_candidates, skip_reasons_for_invalid)``.

    ``max_configs`` truncates the *valid* list after pruning; the skipped /
    invalid candidates are reported in full so dashboard rows do not silently
    disappear.
    """
    raw = expand_search(base_config, search, world_size)
    valid: List[ConfigSpec] = []
    skipped: Dict[ConfigSpec, str] = {}
    for cand in raw:
        ok, reason = is_candidate_valid(cand, model, world_size, act_dtype)
        if ok:
            valid.append(cand)
        else:
            skipped[cand] = reason or "invalid"
    if max_configs is not None and max_configs >= 0 and len(valid) > max_configs:
        truncated = valid[max_configs:]
        valid = valid[:max_configs]
        for cand in truncated:
            skipped[cand] = f"truncated by --max_configs={max_configs}"
    return valid, skipped


def _maybe_auto_enable_search_axes(args: argparse.Namespace) -> None:
    """Promote multi-value runtime flags into ``--search`` axes when needed.

    Rules:
      - Passing >1 value to ``--backend``/``--comm_method``/``--parallel_mode``
        (or ``--backend ALL``) implicitly enables the matching ``--search`` axis
        so users do not have to repeat themselves.
      - A single value keeps the prior single-config behavior intact.
      - An explicit ``--search none`` together with a multi-value flag is an
        error -- the conflicting intent is surfaced rather than silently
        overridden.
      - ``--search full`` already covers every axis; the auto-promote logic is
        a no-op in that case.
    """
    provided = set(getattr(args, "_cli_provided", set()))
    current = tuple(args.search) if args.search else ("none",)

    if current == ("full",):
        return

    promote: List[str] = []
    backends = _coerce_str_tuple(getattr(args, "backend", ()))
    if "backend" in provided and (len(backends) > 1 or backends == ("ALL",)):
        promote.append("backend")
    comm_methods = _coerce_str_tuple(getattr(args, "comm_method", ()))
    if "comm_method" in provided and len(comm_methods) > 1:
        promote.append("comm")
    parallel_modes = _coerce_str_tuple(getattr(args, "parallel_mode", ()))
    if "parallel_mode" in provided and len(parallel_modes) > 1:
        promote.append("parallel")

    # Reject CUSTOM in a multi-value parallel sweep -- CUSTOM still needs scalar
    # --moe_ep_size / --moe_tp_size so it cannot be combined with other modes.
    if len(parallel_modes) > 1 and "CUSTOM" in parallel_modes:
        raise ValueError(
            "--parallel_mode CUSTOM must be passed alone; it requires --moe_ep_size "
            f"and --moe_tp_size and cannot be combined with other modes (got {list(parallel_modes)})."
        )

    if not promote:
        return

    if current == ("none",) and "search" in provided:
        raise ValueError(
            "--search none conflicts with a multi-value runtime flag "
            f"(would promote axes {promote}). Drop --search none or pass a single value per axis."
        )

    if current == ("none",):
        new_axes = tuple(promote)
    else:
        merged: List[str] = list(current)
        for axis in promote:
            if axis not in merged:
                merged.append(axis)
        new_axes = tuple(merged)

    args.search = _parse_search_axes(new_axes)


def _coerce_str_tuple(val: Any) -> Tuple[str, ...]:
    """Normalize ``nargs="+"`` argparse fields into an upper-cased str tuple.

    Accepts the post-parse value of ``--backend``/``--comm_method``/
    ``--parallel_mode`` regardless of whether it came in as a single string
    (e.g. from a config file or default) or a list (argparse ``nargs="+"``).
    """
    if val is None:
        return ()
    if isinstance(val, (list, tuple)):
        return tuple(str(v).upper() for v in val if str(v).strip())
    return (str(val).upper(),)


_SEARCH_AXES = ("backend", "comm", "parallel")
_SEARCH_MODES = ("none",) + _SEARCH_AXES + ("full",)


def _normalize_csv_tokens(value: Any) -> List[str]:
    """Normalise a ``nargs='+'`` / comma-separated / scalar input into tokens.

    Splits on commas and whitespace, strips, lowercases, and drops empty
    fragments. Preserves input order so the caller can deduplicate while
    keeping a stable axis order in error messages.
    """
    items = value if isinstance(value, (list, tuple)) else [value]
    return [
        part.strip().lower()
        for item in items
        for part in str(item).replace(",", " ").split()
        if part.strip()
    ]


def _parse_search_axes(value: Any) -> Tuple[str, ...]:
    parts = _normalize_csv_tokens(value)
    if not parts:
        return ("none",)

    out: List[str] = []
    for part in parts:
        if part not in _SEARCH_MODES:
            raise ValueError(f"unknown --search axis {part!r}; valid: {list(_SEARCH_MODES)}")
        if part not in out:
            out.append(part)

    if "none" in out and len(out) > 1:
        raise ValueError("--search none cannot be combined with other axes")
    if "full" in out and len(out) > 1:
        raise ValueError("--search full cannot be combined with other axes")
    return tuple(out)


_DEFAULT_PARALLEL_AXIS_VALUES: Tuple[str, ...] = ("DEP", "TEP", "DTP", "TTP")


def _axis_values_from_args(
    args: argparse.Namespace,
    *,
    cli_dest: str,
    cli_flag_name: str,
    config_key: Optional[str],
    full_set: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Resolve the value set for a search axis.

    Resolution order (highest priority first):
      1. ``args._config_search_axes[config_key]`` if the JSON config provided a list
         and the user did not also pass the corresponding CLI flag.
      2. The CLI flag list if the user explicitly provided it (``ALL`` expands to
         ``full_set`` for the backend axis).
      3. ``full_set`` -- the default when the axis is enabled but no explicit
         subset was given. This replaces the previous footgun where a bare
         ``--search backend`` would silently expand to a single default value.
    """
    config_axes = getattr(args, "_config_search_axes", {}) or {}
    provided = set(getattr(args, "_cli_provided", set()))

    if config_key is not None and config_key in config_axes and cli_dest not in provided:
        return tuple(config_axes[config_key])

    if cli_dest in provided:
        values = _coerce_str_tuple(getattr(args, cli_dest))
        if not values:
            return full_set
        if cli_dest == "backend" and "ALL" in values:
            if len(values) != 1:
                raise ValueError(f"{cli_flag_name} ALL must be passed alone (got {list(values)}).")
            return full_set
        return values

    return full_set


def _resolve_search_from_args(args: argparse.Namespace, base_config: ConfigSpec) -> SearchSpec:
    search_axes = _parse_search_axes(args.search)

    if search_axes == ("none",):
        return SearchSpec(mode="none")

    full_search = search_axes == ("full",)
    enabled_axes = set(_SEARCH_AXES if full_search else search_axes)
    mode = "full" if full_search else ",".join(search_axes)

    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_options: Tuple[bool, ...] = ()

    if "backend" in enabled_axes:
        backends = _axis_values_from_args(
            args,
            cli_dest="backend",
            cli_flag_name="--backend",
            config_key="backend",
            full_set=tuple(_ALL_BACKENDS),
        )
    if "parallel" in enabled_axes:
        parallel_modes = _axis_values_from_args(
            args,
            cli_dest="parallel_mode",
            cli_flag_name="--parallel_mode",
            config_key="parallel_mode",
            full_set=_DEFAULT_PARALLEL_AXIS_VALUES,
        )
    if "comm" in enabled_axes:
        comm_methods = _axis_values_from_args(
            args,
            cli_dest="comm_method",
            cli_flag_name="--comm_method",
            config_key="comm_method",
            full_set=_FORCED_COMM_ENV_VALUES,
        )
        comm_methods = tuple(v for v in comm_methods if str(v).upper() != "AUTO")
        if not comm_methods:
            raise ValueError(
                "--search comm does not include AUTO because AUTO aliases one of the concrete "
                "communication strategies; pass a forced --comm_method value or disable comm search."
            )
    # ``cuda_graph`` and ``combine_precision`` axes are reserved for ``full``;
    # leave them empty by default so the base value is used.
    if full_search:
        cuda_graph_options = (True, False)

    return SearchSpec(
        mode=mode,
        backends=backends,
        parallel_modes=parallel_modes,
        comm_methods=comm_methods,
        cuda_graph_options=cuda_graph_options,
        combine_precision_options=combine_options,
    )


def _parse_analysis(value: Any) -> Tuple[str, ...]:
    parts = _normalize_csv_tokens(value)
    valid = {"none", "kernels"}
    out: List[str] = []
    for p in parts:
        if p not in valid:
            raise ValueError(f"unknown --analysis dimension {p!r}; valid: {sorted(valid)}")
        if p == "none":
            continue
        if p not in out:
            out.append(p)
    return tuple(out)
