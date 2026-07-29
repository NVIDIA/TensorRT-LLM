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

"""CLI and config-file resolution for the MoE microbenchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .backend import MoeBackendType
from .mapping import _resolve_mapping_layout
from .routing import _per_rank_tokens
from .search import (
    _coerce_str_tuple,
    _maybe_auto_enable_search_axes,
    _parse_analysis,
    _parse_search_axes,
    _resolve_search_from_args,
)
from .specs import (
    _ALL_BACKENDS,
    _COMM_METHODS,
    _ROUTING_METHODS,
    BUILT_IN_MODELS,
    ConfigSpec,
    ModelSpec,
    RoutingControlSpec,
    SearchSpec,
    WorkloadSpec,
)


@dataclass(frozen=True)
class _BenchmarkContext:
    model: ModelSpec
    workloads: List[WorkloadSpec]
    base_config: ConfigSpec
    search: SearchSpec
    analysis: Tuple[str, ...]
    act_dtype: torch.dtype
    routing_logits_dtype: torch.dtype


def _resolve_benchmark_context(args: argparse.Namespace) -> _BenchmarkContext:
    analysis = _parse_analysis(args.analysis)
    model = _resolve_model_from_args(args)
    workloads = _resolve_workloads_from_args(args)
    base_config = _resolve_base_config_from_args(args)
    search = _resolve_search_from_args(args, base_config)
    act_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    routing_logits_dtype = (
        torch.float32 if model.routing_method_cls is DeepSeekV3MoeRoutingMethod else act_dtype
    )
    return _BenchmarkContext(
        model=model,
        workloads=workloads,
        base_config=base_config,
        search=search,
        analysis=analysis,
        act_dtype=act_dtype,
        routing_logits_dtype=routing_logits_dtype,
    )


def _build_worker_header(ctx: _BenchmarkContext, launcher: str, world_size: int) -> Dict[str, Any]:
    return {
        "benchmark": "bench_moe",
        "launcher": launcher,
        "model": ctx.model.to_dict(),
        "search": ctx.search.to_dict(),
        "world_size": world_size,
        "analysis": list(ctx.analysis) or ["summary"],
        "workloads": [
            w.to_dict(
                per_rank_num_tokens=_per_rank_tokens(
                    w,
                    world_size,
                    enable_dp=bool(_resolve_mapping_layout(ctx.base_config, world_size)[2]),
                )
            )
            for w in ctx.workloads
        ],
        "base_config": ctx.base_config.to_dict(),
    }


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Search and sweep")
    group.add_argument(
        "--search",
        type=lambda s: str(s).lower(),
        nargs="+",
        default=("none",),
        help=(
            "Expand one or more runtime axes. Examples: --search backend --backend ALL; "
            "--search backend comm; --search full. Comma-separated input is also accepted."
        ),
    )
    group.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Run at most this many valid candidate configs after pruning. Example: --max_configs 32.",
    )
    group.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Stop launching new candidates after this wall-clock budget. Example: --time_budget_minutes 30.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE module microbenchmark (MPI). Times ConfigurableMoE.forward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    launch_group = parser.add_argument_group("Launch")
    launch_group.add_argument(
        "--world_size",
        type=int,
        default=None,
        help=(
            "Number of ranks to run. Under external mpirun/srun, this must match "
            "the external MPI world size; without an external launcher, values >1 "
            "use the self-spawn launcher."
        ),
    )

    model_group = parser.add_argument_group("Model and shape")
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(BUILT_IN_MODELS.keys()),
        help=(
            "Built-in model shape. Examples: deepseek_v3, qwen1.5_moe. "
            "Omit only when passing all custom shape fields below."
        ),
    )
    model_group.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Custom-shape total expert count. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Custom-shape experts selected per token. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Custom-shape hidden size. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="Custom-shape MoE intermediate size. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--n_group",
        type=int,
        default=None,
        help="DeepSeek-style routing group count for custom grouped routing.",
    )
    model_group.add_argument(
        "--topk_group",
        type=int,
        default=None,
        help="DeepSeek-style number of routing groups kept per token.",
    )
    model_group.add_argument(
        "--quant",
        type=lambda s: QuantAlgo[str(s).upper()] if s is not None else None,
        default=None,
        choices=[q.name for q in QuantAlgo],
        help="Quantization algorithm. Example: --quant FP8_BLOCK_SCALES.",
    )
    model_group.add_argument(
        "--routing_method",
        type=lambda s: str(s).upper(),
        default="AUTO",
        choices=sorted(_ROUTING_METHODS) + ["AUTO"],
        help=(
            "Routing method. Defaults to AUTO: built-in models use the spec "
            "default; custom shapes must specify an explicit method."
        ),
    )

    workload_group = parser.add_argument_group("Workload shape")
    workload_group.add_argument(
        "--balanced_total_num_tokens",
        "--num_tokens",
        dest="balanced_total_num_tokens",
        type=int,
        nargs="+",
        required=False,
        help=(
            "Global token counts to sweep. Each value is balanced across ranks, "
            "spreading any remainder one token per leading rank (e.g. world_size=4, "
            "tokens=2 -> [1, 1, 0, 0]). Example: --balanced_total_num_tokens 64 256 1024."
        ),
    )

    routing_group = parser.add_argument_group("Routing control")
    routing_group.add_argument(
        "--routing_mode",
        type=lambda s: str(s).lower(),
        default="native",
        choices=("native", "forced"),
        help=(
            "native: route through production logits kernels (default); "
            "forced: supply top-k ids/scales directly (skips fused scoring)."
        ),
    )
    routing_group.add_argument(
        "--projection_policy",
        type=lambda s: str(s).lower(),
        default="project",
        choices=("project", "reject"),
        help=(
            "project: when native logits cannot exactly realise the plan, run with "
            "the closest legal projection and warn; reject: skip the case instead."
        ),
    )
    routing_group.add_argument(
        "--comm_pattern",
        type=str,
        default="balanced_alltoall",
        help=(
            "Source-to-target slot dispatch pattern. balanced_alltoall builds a balanced "
            "plan by default; random keeps logits uncontrolled when expert_pattern is also random. "
            "Examples: balanced_alltoall, random, "
            "receiver_hotspot,hotness=0.75,rank=0, pair_hotspot,hotness=0.5,src=0,dst=1, "
            "local_only, ring."
        ),
    )
    routing_group.add_argument(
        "--expert_pattern",
        type=str,
        default="balanced",
        help=(
            "Per-target-rank local expert histogram pattern. balanced builds a balanced "
            "plan by default; random keeps logits uncontrolled when comm_pattern is also random. "
            "Examples: balanced, random, "
            "hotspot,hotness=0.5, hotspot,active_experts=2."
        ),
    )
    routing_group.add_argument(
        "--routing_pattern_file",
        type=str,
        default=None,
        help="JSON file that provides both slot_dispatch_matrix and expert_histogram.",
    )
    routing_group.add_argument(
        "--per_rank_num_tokens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit per-rank input token counts. Length must equal world_size "
            "and sum defines the workload total. Mutually exclusive with --balanced_total_num_tokens."
        ),
    )
    routing_group.add_argument(
        "--routing_dump_matrix",
        action="store_true",
        help="Include the full observed slot/token matrix and expert histogram in each result row.",
    )
    routing_group.add_argument(
        "--routing_seed",
        type=int,
        default=0,
        help="Seed for deterministic routing-plan materialisation; independent from --random_seed.",
    )
    routing_group.add_argument(
        "--enable_perfect_router",
        action="store_true",
        help=(
            "Use the lower-level ENABLE_PERFECT_ROUTER path to replace router logits with "
            "load-balanced logits inside the MoE module. Disabled by default so routing-control "
            "patterns use bench_moe's own projected logits."
        ),
    )

    parallel_group = parser.add_argument_group("Parallel layout")
    parallel_group.add_argument(
        "--parallel_mode",
        type=str,
        nargs="+",
        default=("DEP",),
        choices=("DEP", "TEP", "DTP", "TTP", "CUSTOM"),
        help=(
            "Parallel layout(s) to benchmark. Pass multiple values to sweep, e.g. "
            "--parallel_mode DEP TEP. DEP=attention DP + MoE EP; TEP=attention TP + "
            "MoE EP; DTP/TTP use MoE TP; CUSTOM requires --moe_ep_size and "
            "--moe_tp_size and must be passed alone."
        ),
    )
    parallel_group.add_argument(
        "--moe_ep_size",
        type=int,
        default=None,
        help="CUSTOM only: MoE expert-parallel size. Must multiply with --moe_tp_size to world_size.",
    )
    parallel_group.add_argument(
        "--moe_tp_size",
        type=int,
        default=None,
        help="CUSTOM only: MoE tensor-parallel size. Must multiply with --moe_ep_size to world_size.",
    )
    parallel_group.add_argument(
        "--enable_attention_dp",
        action="store_true",
        help="CUSTOM only: enable attention data parallelism for the mapping.",
    )

    runtime_group = parser.add_argument_group("Runtime backend and communication")
    runtime_group.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        nargs="+",
        default=("TRTLLM",),
        choices=_ALL_BACKENDS + ["ALL"],
        help=(
            "MoE backend(s) to benchmark. Pass multiple values to sweep, e.g. "
            "--backend CUTLASS DEEPGEMM. ALL expands to every ConfigurableMoE-eligible "
            "backend and must be passed alone. Passing >1 value (or ALL) implicitly "
            "enables --search backend."
        ),
    )
    runtime_group.add_argument(
        "--comm_method",
        type=lambda s: str(s).upper(),
        nargs="+",
        default=("AUTO",),
        choices=_COMM_METHODS,
        help=(
            "Communication method(s) to benchmark. Pass multiple values to sweep, "
            "e.g. --comm_method NVLINK_ONE_SIDED DEEPEP. AUTO lets TensorRT-LLM "
            "select; other values force a specific path. Passing >1 value implicitly "
            "enables --search comm."
        ),
    )

    timing_group = parser.add_argument_group("Timing")
    timing_group.add_argument(
        "--no_cuda_graph",
        dest="cuda_graph",
        action="store_false",
        default=True,
        help="Disable CUDA-Graph capture and use eager timing.",
    )
    timing_group.add_argument("--warmup", type=int, default=1, help="Warmup iterations per case.")
    timing_group.add_argument("--iters", type=int, default=12, help="Timed iterations per case.")
    timing_group.add_argument(
        "--nsys",
        action="store_true",
        default=False,
        help="Emit an NVTX range + cudaProfilerStart/Stop around the measured MoE forward "
        "(after warmup) so `nsys profile -c cudaProfilerApi` captures only that region. "
        "Disables CUPTI kernel breakdown (conflicts with nsys). Latency measurement is unchanged.",
    )
    timing_group.add_argument(
        "--fast_autotune",
        action="store_true",
        help="Use a short autotune pass for smoke tests; may reduce measurement quality.",
    )
    timing_group.add_argument(
        "--per_candidate_timeout_s",
        type=float,
        default=0.0,
        help=(
            "Hard wall-clock budget per candidate (seconds). If a candidate exceeds "
            "this, a watchdog thread sends SIGKILL to break suspected NCCL deadlocks "
            "or CUDA hangs that ``torch.cuda.synchronize()`` cannot detect. The killed "
            "candidate is missing from the checkpoint JSON, so an outer driver that "
            "restarts with --resume_from will re-attempt it. 0 disables the watchdog."
        ),
    )
    timing_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Activation dtype for synthetic inputs.",
    )
    timing_group.add_argument(
        "--use_low_precision_moe_combine",
        action="store_true",
        help="Use low-precision combine where the selected backend supports it.",
    )
    timing_group.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Seed for synthetic hidden states/router logits; routing-control plans use --routing_seed.",
    )

    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--analysis",
        nargs="+",
        default=("kernels",),
        choices=("none", "kernels"),
        help="Analysis data to collect. Use --analysis none for latency-only output.",
    )

    # ---- Search / sweep ----
    _add_search_arguments(parser)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="JSON config file. CLI flags override matching config-file fields.",
    )
    output_group.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Write the final dashboard JSON report to this path.",
    )
    output_group.add_argument(
        "--analysis_workbook_file",
        type=str,
        default=None,
        help=(
            "Write an Excel workbook with all candidate rows, per-workload sheets, "
            "best configs, and status summaries. Defaults to <output_file>.analysis.xlsx."
        ),
    )
    output_group.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Read an existing JSON report and skip every (workload, config) candidate "
            "whose row is terminal (success/failed, or skipped for a non-upstream reason). "
            "Placeholder rows left behind by a prior crash (skip_reason ending in "
            "'_upstream') are dropped and re-attempted. Combine with --output_file to "
            "write fresh results back into the same file (atomic, checkpointed after "
            "every candidate)."
        ),
    )
    output_group.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help=(
            "Write the --output_file JSON checkpoint after every N freshly completed "
            "candidates. 0 disables incremental checkpointing (only the final JSON is "
            "written). Default 1 trades a small amount of JSON I/O for crash-safety: "
            "no completed candidate is ever lost to a watchdog SIGKILL or sticky-error "
            "exit."
        ),
    )
    args = parser.parse_args()
    provided = set()
    argv = sys.argv[1:]
    for action in parser._actions:
        for option in action.option_strings:
            if option in argv or any(arg.startswith(option + "=") for arg in argv):
                provided.add(action.dest)
    args._cli_provided = provided
    args.search = _parse_search_axes(args.search)
    _maybe_auto_enable_search_axes(args)
    return args


def _resolve_model_from_args(args: argparse.Namespace) -> ModelSpec:
    base = BUILT_IN_MODELS.get(args.model) if args.model is not None else None

    routing = args.routing_method
    if base is None:
        # Custom shape requires explicit fields.
        missing = [
            f
            for f in ("num_experts", "top_k", "hidden_size", "intermediate_size")
            if getattr(args, f) is None
        ]
        if missing:
            raise ValueError("No --model selected; you must also pass: " + ", ".join(missing))
        if routing == "AUTO":
            raise ValueError(
                "Custom shapes (no --model) require an explicit --routing_method; "
                "AUTO has no safe default."
            )
        quant_name = args.quant.name if args.quant is not None else None
        return ModelSpec(
            name="custom",
            num_experts=int(args.num_experts),
            top_k=int(args.top_k),
            hidden_size=int(args.hidden_size),
            intermediate_size=int(args.intermediate_size),
            quant_algo=quant_name,
            routing_method=routing,
            n_group=args.n_group,
            topk_group=args.topk_group,
        )

    # Built-in model with optional per-field overrides.
    if routing == "AUTO":
        routing = base.routing_method

    quant_name: Optional[str]
    if args.quant is not None:
        quant_name = args.quant.name
    else:
        quant_name = base.quant_algo
    return ModelSpec(
        name=base.name,
        num_experts=int(args.num_experts) if args.num_experts is not None else base.num_experts,
        top_k=int(args.top_k) if args.top_k is not None else base.top_k,
        hidden_size=int(args.hidden_size) if args.hidden_size is not None else base.hidden_size,
        intermediate_size=int(args.intermediate_size)
        if args.intermediate_size is not None
        else base.intermediate_size,
        quant_algo=quant_name,
        routing_method=routing,
        n_group=args.n_group if args.n_group is not None else base.n_group,
        topk_group=args.topk_group if args.topk_group is not None else base.topk_group,
        swiglu_alpha=base.swiglu_alpha,
        swiglu_beta=base.swiglu_beta,
        swiglu_limit=base.swiglu_limit,
    )


def _resolve_workloads_from_args(args: argparse.Namespace) -> List[WorkloadSpec]:
    balanced_total_num_tokens = getattr(args, "balanced_total_num_tokens", None)
    if balanced_total_num_tokens is None:
        balanced_total_num_tokens = getattr(args, "num_tokens", None)

    # Resolve RoutingControlSpec from explicit CLI/config fields.
    per_rank_num_tokens: Optional[Tuple[int, ...]] = None
    if getattr(args, "per_rank_num_tokens", None):
        if balanced_total_num_tokens:
            raise ValueError(
                "--balanced_total_num_tokens and --per_rank_num_tokens are mutually exclusive"
            )
        raw = args.per_rank_num_tokens
        if isinstance(raw, str):
            try:
                parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
            except ValueError as exc:
                raise ValueError(f"--per_rank_num_tokens must be integers; got {raw!r}") from exc
        else:
            parts = [int(v) for v in raw]
        per_rank_num_tokens = tuple(parts)
        if any(v < 0 for v in per_rank_num_tokens):
            raise ValueError("--per_rank_num_tokens entries must be >= 0")
        token_values = [sum(per_rank_num_tokens)]
    else:
        if not balanced_total_num_tokens:
            raise ValueError(
                "--balanced_total_num_tokens or --per_rank_num_tokens is required "
                "(or supply via --config_file)"
            )
        token_values = [int(t) for t in balanced_total_num_tokens]
        if any(t < 0 for t in token_values):
            raise ValueError("--balanced_total_num_tokens entries must be >= 0")

    routing_spec = RoutingControlSpec(
        routing_mode=str(getattr(args, "routing_mode", "native")),
        projection_policy=str(getattr(args, "projection_policy", "project")),
        comm_pattern=str(getattr(args, "comm_pattern", "balanced_alltoall")),
        expert_pattern=str(getattr(args, "expert_pattern", "balanced")),
        routing_pattern_file=getattr(args, "routing_pattern_file", None),
        per_rank_num_tokens=per_rank_num_tokens,
        routing_dump_matrix=bool(getattr(args, "routing_dump_matrix", False)),
        seed=int(getattr(args, "routing_seed", 0)),
    )

    return [
        WorkloadSpec(
            num_tokens=int(t),
            routing_control=routing_spec,
        )
        for t in token_values
    ]


def _resolve_base_config_from_args(args: argparse.Namespace) -> ConfigSpec:
    # ``--backend``/``--comm_method``/``--parallel_mode`` are ``nargs="+"`` lists.
    # The base config holds a single placeholder value; ``_resolve_search_from_args``
    # is responsible for expanding the actual sweep set per axis.
    backends_list = _coerce_str_tuple(args.backend)
    comm_list = _coerce_str_tuple(args.comm_method)
    parallel_list = _coerce_str_tuple(args.parallel_mode)

    comm_method = comm_list[0] if comm_list else "AUTO"

    backend = backends_list[0] if backends_list else MoeBackendType.CUTLASS.value
    if backend == "ALL":
        backend = MoeBackendType.CUTLASS.value  # placeholder; overwritten by search expansion

    parallel_mode = parallel_list[0] if parallel_list else "DEP"

    # parallel_mode CUSTOM if explicit EP/TP overrides are present.
    if (args.moe_ep_size is not None or args.moe_tp_size is not None) and parallel_mode in (
        "DEP",
        "TEP",
        "DTP",
        "TTP",
    ):
        # Treat explicit overrides as opting into CUSTOM so output metadata is honest.
        parallel_mode = "CUSTOM"

    if parallel_mode == "CUSTOM" and (args.moe_ep_size is None or args.moe_tp_size is None):
        raise ValueError("--parallel_mode=CUSTOM requires both --moe_ep_size and --moe_tp_size")

    return ConfigSpec(
        backend=backend,
        parallel_mode=parallel_mode,
        moe_ep_size=args.moe_ep_size,
        moe_tp_size=args.moe_tp_size,
        enable_attention_dp=bool(args.enable_attention_dp) if parallel_mode == "CUSTOM" else None,
        comm_method=comm_method,
        cuda_graph=bool(args.cuda_graph),
        use_low_precision_moe_combine=bool(args.use_low_precision_moe_combine),
    )


def _maybe_load_config_file(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay ``--config_file`` JSON onto ``args``; explicit CLI flags win."""
    if not args.config_file:
        args._config_search_axes = {}
        return args
    with open(args.config_file) as f:
        cfg = json.load(f)

    provided = set(getattr(args, "_cli_provided", set()))

    def set_if_unset(dest: str, value: Any) -> None:
        if dest not in provided:
            setattr(args, dest, value)

    if "model" in cfg:
        set_if_unset("model", cfg["model"])
    workload_cfg = cfg.get("workload", {}) or {}
    if "balanced_total_num_tokens" in workload_cfg:
        set_if_unset("balanced_total_num_tokens", list(workload_cfg["balanced_total_num_tokens"]))
    elif "num_tokens" in workload_cfg:
        set_if_unset("balanced_total_num_tokens", list(workload_cfg["num_tokens"]))
    if "per_rank_num_tokens" in workload_cfg:
        prnt = workload_cfg["per_rank_num_tokens"]
        set_if_unset(
            "per_rank_num_tokens",
            [int(v) for v in prnt] if isinstance(prnt, list) else str(prnt),
        )
    routing_cfg = workload_cfg.get("routing_control", {}) or {}
    if "routing_mode" in routing_cfg:
        set_if_unset("routing_mode", str(routing_cfg["routing_mode"]).lower())
    if "projection_policy" in routing_cfg:
        set_if_unset("projection_policy", str(routing_cfg["projection_policy"]).lower())
    if "comm_pattern" in routing_cfg:
        set_if_unset("comm_pattern", str(routing_cfg["comm_pattern"]))
    if "expert_pattern" in routing_cfg:
        set_if_unset("expert_pattern", str(routing_cfg["expert_pattern"]))
    if "routing_pattern_file" in routing_cfg:
        set_if_unset("routing_pattern_file", str(routing_cfg["routing_pattern_file"]))
    if "routing_dump_matrix" in routing_cfg:
        set_if_unset("routing_dump_matrix", bool(routing_cfg["routing_dump_matrix"]))
    if "seed" in routing_cfg:
        set_if_unset("routing_seed", int(routing_cfg["seed"]))
    search_cfg = cfg.get("search", {}) or {}
    unsupported_search_keys = set(search_cfg) - {"backend", "parallel_mode", "comm_method"}
    if unsupported_search_keys:
        raise ValueError(f"unsupported search key(s): {sorted(unsupported_search_keys)}")
    config_search_axes: Dict[str, Tuple[str, ...]] = {}

    def normalize_search_axis(key: str, value: Any) -> Optional[Tuple[str, ...]]:
        """Project a config-file search-axis entry onto ``args`` (list form).

        Returns the tuple of canonical values if it should be recorded as a
        sweep axis in ``_config_search_axes`` (multi-value or backend=ALL).
        Returns ``None`` when the value was instead written to the matching
        ``args.<key>`` scalar-list flag.
        """
        if key in provided:
            return None
        values = (
            tuple(str(v).upper() for v in value)
            if isinstance(value, list)
            else (str(value).upper(),)
        )
        # ``backend=ALL`` is the explicit "expand to every backend" sentinel.
        if key == "backend" and values == ("ALL",):
            set_if_unset("backend", ("ALL",))
            return None
        if len(values) == 1:
            # Single-value config entry behaves like a CLI scalar default.
            set_if_unset(key, values)
            return None
        return values

    if search_cfg:
        for key in ("backend", "parallel_mode", "comm_method"):
            if key in search_cfg:
                axis = normalize_search_axis(key, search_cfg[key])
                if axis:
                    config_search_axes[key] = axis
        if "search" not in provided:
            # Merge config-provided multi-value axes into any axes the CLI
            # auto-promoted (e.g. via multi-value --backend). Without merging,
            # config-driven axes would clobber CLI auto-promoted ones.
            existing = tuple(args.search) if args.search and args.search != ("none",) else ()
            extra: List[str] = []
            if "backend" in search_cfg and "backend" not in provided:
                extra.append("backend")
            if "parallel_mode" in search_cfg and "parallel_mode" not in provided:
                extra.append("parallel")
            if "comm_method" in search_cfg and "comm_method" not in provided:
                extra.append("comm")
            if extra:
                merged: List[str] = [a for a in existing if a not in ("none",)]
                for axis in extra:
                    if axis not in merged:
                        merged.append(axis)
                if merged:
                    args.search = tuple(merged)
    args._config_search_axes = config_search_axes
    if "analysis" in cfg:
        set_if_unset("analysis", cfg["analysis"])
    if "max_configs" in cfg:
        set_if_unset("max_configs", int(cfg["max_configs"]))
    if "time_budget_minutes" in cfg:
        set_if_unset("time_budget_minutes", float(cfg["time_budget_minutes"]))
    if "output_file" in cfg:
        set_if_unset("output_file", cfg["output_file"])
    if "analysis_workbook_file" in cfg:
        set_if_unset("analysis_workbook_file", cfg["analysis_workbook_file"])
    return args
