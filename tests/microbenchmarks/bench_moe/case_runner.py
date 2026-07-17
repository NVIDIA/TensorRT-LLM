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

"""Per-candidate execution and routing-control orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._utils import mpi_allgather

from .build import (
    _backend_name_from_module,
    _build_moe_module,
    _calculate_num_chunks_safe,
    _comm_method_name,
    _scheduler_kind_name,
)
from .mapping import _build_mapping_from_config, _resolve_mapping_layout
from .results import (
    _build_latency_block,
    _build_raw_data_block,
    _classify_bottleneck,
    _gather_kernel_timing_blocks,
    _gather_per_iteration_times,
)
from .routing import (
    RoutingPlan,
    _build_routing_plan,
    _materialize_selected_experts_for_rank,
    _maybe_install_routing_control_patch,
    _observe_routing_metrics,
    _observe_summary,
    _per_rank_tokens,
    _project_router_logits_for_plan,
)
from .specs import (
    _FORCED_COMM_ENV_VALUES,
    ConfigSpec,
    ModelSpec,
    RoutingControlSpec,
    RunResult,
    WorkloadSpec,
)
from .timing import _run_autotune, _time_moe_forward_cuda_graph, _time_moe_forward_eager
from .utils import _InputCache, _make_inputs, _maybe_print_rank0


def _force_comm_env(comm_method: str, prev: Optional[str]) -> None:
    """Push or restore ``TRTLLM_FORCE_COMM_METHOD`` for one case.

    The design notes that env-var forcing is the only available per-case knob
    today; subprocess isolation is the recommended long-term path.
    """
    upper = comm_method.upper()
    if upper in _FORCED_COMM_ENV_VALUES:
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = upper
    else:
        if prev is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev


def _gather_status_per_rank(local_status: str) -> Dict[str, str]:
    payload = mpi_allgather(local_status)
    return {f"rank{i}": s for i, s in enumerate(payload)}


@dataclass(frozen=True)
class _RoutingObservations:
    """Observation matrices and counters fed into ``_build_routing_control_block``.

    These are the four numbers the consumer cares about: the per-slot dispatch
    matrix, per-token dispatch matrix, per-rank expert histogram, and the
    optional ``num_chunks`` counter observed during the run. All come from a
    deterministic re-materialisation of the canonical ``RoutingPlan`` -- see
    ``_build_routing_control_block`` for the contract.
    """

    slot: List[List[int]]
    token: List[List[int]]
    hist: List[List[int]]
    num_chunks: Optional[int] = None


def _build_routing_control_block(
    *,
    spec: RoutingControlSpec,
    plan: RoutingPlan,
    observations: _RoutingObservations,
    routing_path: Optional[str],
    realization_status: str,
    realization_reason: str,
    enable_perfect_router: bool,
    max_num_tokens_per_rank: int,
    warnings: List[str],
    scale_dtype: torch.dtype,
    moe_ep_size: int,
) -> Dict[str, Any]:
    """Compose the ``routing_control`` block for a result row.

    Always includes ``requested`` and an ``actual`` summary; full slot/token
    matrices and histograms are included only when ``spec.routing_dump_matrix``
    is set, to avoid JSON bloat during large sweeps.

    NOTE on ``observations.*`` fields: the dispatch / histogram numbers
    reported here are derived from a deterministic re-materialisation of the
    canonical ``RoutingPlan`` -- they describe *the plan the bench asked the
    kernel to realise*, not what the kernel actually emitted at runtime. The
    ``actual.observation_source`` field documents this. In ``forced`` mode the
    kernel is patched to consume the exact materialised top-k, so plan ==
    kernel output by construction. In ``native`` mode the kernel routes via
    projected logits and may produce slightly different top-k due to fp ties,
    quantisation, or projection-status='projected'; a warning is added so the
    consumer does not over-trust the slot/histogram numbers.
    """
    routing_mode = spec.routing_mode
    dump_full = bool(spec.routing_dump_matrix)
    observed_slot = observations.slot
    observed_token = observations.token
    observed_hist = observations.hist
    num_chunks_observed = observations.num_chunks

    requested_slot = [list(row) for row in plan.dispatch_matrix]
    max_abs, max_rel = _observe_summary(requested_slot, observed_slot)
    row_sums = [sum(row) for row in observed_slot]
    col_sums = [sum(observed_slot[s][d] for s in range(moe_ep_size)) for d in range(moe_ep_size)]
    diag = sum(observed_slot[i][i] for i in range(moe_ep_size)) if moe_ep_size > 0 else 0
    total = sum(row_sums)
    off_diag_ratio = 0.0 if total <= 0 else (1.0 - diag / total)

    flat_hist = [v for row in observed_hist for v in row]
    hist_min = min(flat_hist) if flat_hist else 0
    hist_max = max(flat_hist) if flat_hist else 0
    active_experts = sum(1 for v in flat_hist if v > 0)

    warnings_out = list(warnings)
    if routing_mode != "forced":
        warnings_out.append(
            "observed_* fields are derived from RoutingPlan re-materialisation, "
            "not from the kernel's actual selected_experts output; in native mode "
            "the real top-k may differ from the plan."
        )

    block: Dict[str, Any] = {
        "requested": {
            "routing_mode": spec.routing_mode,
            "projection_policy": spec.projection_policy,
            "comm_pattern": spec.comm_pattern,
            "expert_pattern": spec.expert_pattern,
            "routing_pattern_file": spec.routing_pattern_file,
            "per_rank_num_tokens": list(plan.per_rank_num_tokens),
            "seed": int(spec.seed),
        },
        "actual": {
            "routing_path": routing_path,
            "routing_realization": {
                "status": realization_status,
                "reason": realization_reason,
                "max_abs_slot_error": int(max_abs),
                "max_relative_slot_error": float(max_rel),
            },
            "enable_perfect_router": bool(enable_perfect_router),
            "effective_src_axis": "dp_rank",
            "max_num_tokens_per_rank": int(max_num_tokens_per_rank),
            "num_chunks_observed": int(num_chunks_observed)
            if isinstance(num_chunks_observed, int)
            else None,
            "use_dp_padding": False,
            # "plan_exact": forced mode (kernel is patched to the materialised
            # plan, so observed == plan by construction).
            # "plan_simulation": native mode (numbers are a deterministic
            # re-materialisation of the plan, NOT the kernel's actual top-k).
            "observation_source": ("plan_exact" if routing_mode == "forced" else "plan_simulation"),
            "observed_dispatch_matrix_summary": {
                "row_sums": row_sums,
                "col_sums": col_sums,
                "off_diagonal_ratio": float(off_diag_ratio),
                "max_abs_slot_error": int(max_abs),
                "matrix_dump_path": None,
            },
            "observed_expert_histogram_summary": {
                "min": int(hist_min),
                "max": int(hist_max),
                "active_experts": int(active_experts),
            },
            "selected_scales": {
                "distribution": "uniform",
                "dtype": str(scale_dtype),
                "seed": int(spec.seed),
            },
            "warnings": warnings_out,
        },
    }
    if dump_full:
        block["actual"]["observed_slot_dispatch_matrix"] = [list(r) for r in observed_slot]
        block["actual"]["observed_token_dispatch_matrix"] = [list(r) for r in observed_token]
        block["actual"]["observed_expert_histogram"] = [list(r) for r in observed_hist]
        block["actual"]["requested_slot_dispatch_matrix"] = [list(r) for r in requested_slot]
    return block


@dataclass
class _RoutingInputs:
    """Inputs derived from routing control for one (case, rank).

    ``router_logits`` is the tensor that will be passed to ``moe.forward``; in
    forced mode it is left unchanged because the routing kernels are bypassed.
    ``materialized_ids`` / ``materialized_scales`` are only populated in forced
    mode and are consumed by ``_maybe_install_routing_control_patch``.
    """

    router_logits: torch.Tensor
    materialized_ids: Optional[torch.Tensor] = None
    materialized_scales: Optional[torch.Tensor] = None
    realization_status: str = "exact"
    realization_reason: str = "balanced default"
    routing_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _short_circuit(result: RunResult, status: str, reason: str) -> RunResult:
    """Stamp ``status``/``reason`` on ``result`` and broadcast across ranks.

    Every early-return path in ``_run_one_candidate`` must go through this
    helper so the per-rank status allgather completes on all ranks and the
    output row remains coherent.
    """
    result.status = status
    result.skip_reason = reason
    result.status_per_rank = _gather_status_per_rank(status)
    return result


def _initial_instrumentation(
    analysis: Tuple[str, ...],
    config: ConfigSpec,
    cupti_ctx: Optional[Any],
    nsys: bool = False,
) -> Dict[str, Any]:
    return {
        "level": ",".join(sorted(analysis)) if analysis else "summary",
        "cuda_graph": bool(config.cuda_graph),
        "cupti_available": bool(cupti_ctx is not None and cupti_ctx.ok),
        "nsys_capture": bool(nsys),
        "phase_timing_available": False,
        "kernel_breakdown_available": "kernels" in analysis,
        "autotune_status": "not_run",
        "latency_source": "cuda_event_external" if config.cuda_graph else "cuda_event_eager",
    }


def _pick_enable_perfect_router(
    rc_spec: RoutingControlSpec,
    enable_perfect_router_requested: bool,
) -> bool:
    """Decide whether to enable ``ENABLE_PERFECT_ROUTER`` for one case.

    The perfect router is a lower-level MoE override that replaces the incoming
    router logits inside ``MoE.forward``. Keep it explicit so normal
    routing-control cases use the logits projected by bench_moe itself.
    """
    if not enable_perfect_router_requested:
        return False
    if rc_spec.routing_mode == "forced":
        return False
    return True


def _build_empty_routing_control_block_for_rejection(
    *,
    rc_spec: RoutingControlSpec,
    routing_plan: RoutingPlan,
    model: ModelSpec,
    moe_ep_size: int,
    per_rank: List[int],
    num_chunks: Optional[int],
    rejected_reason: str,
    enable_perfect_router: bool,
    act_dtype: torch.dtype,
) -> Dict[str, Any]:
    experts_per_rank = int(model.num_experts) // int(moe_ep_size)
    ep = int(moe_ep_size)
    empty_observations = _RoutingObservations(
        slot=[[0] * ep for _ in range(ep)],
        token=[[0] * ep for _ in range(ep)],
        hist=[[0] * experts_per_rank for _ in range(ep)],
        num_chunks=num_chunks,
    )
    return _build_routing_control_block(
        spec=rc_spec,
        plan=routing_plan,
        observations=empty_observations,
        routing_path=None,
        realization_status="rejected",
        realization_reason=rejected_reason,
        enable_perfect_router=enable_perfect_router,
        max_num_tokens_per_rank=max(per_rank) if per_rank else 0,
        warnings=[],
        scale_dtype=act_dtype,
        moe_ep_size=ep,
    )


@dataclass
class _RoutingSkip:
    """Encodes a routing-control skip with optional dashboard annotation.

    ``skip_reason`` is the human-readable reason written to the result row.
    ``rejected_reason`` is set only for ``projection_policy=reject`` skips, in
    which case the caller still emits a ``routing_control`` block carrying the
    untruncated projection reason.
    """

    skip_reason: str
    rejected_reason: Optional[str] = None


def _select_routing_inputs(
    *,
    moe,
    model: ModelSpec,
    rc_spec: RoutingControlSpec,
    routing_plan: RoutingPlan,
    rank: int,
    moe_ep_size: int,
    enable_attention_dp: bool,
    base_router_logits: torch.Tensor,
    device: torch.device,
    act_dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
) -> Tuple[Optional[_RoutingInputs], Optional[_RoutingSkip]]:
    """Produce the router_logits / materialised top-k tensors for routing control.

    Returns ``(inputs, None)`` on success or ``(None, skip)`` to abort. The
    skip object distinguishes a plain error (materialise / projection) from a
    ``projection_policy=reject`` rejection that the dashboard wants to see.
    """
    experts_per_rank = int(model.num_experts) // int(moe_ep_size)
    ep_axis_rank = rank if rank < int(moe_ep_size) else (rank % int(moe_ep_size))

    if rc_spec.routing_mode == "forced":
        try:
            ids, scales = _materialize_selected_experts_for_rank(
                routing_plan,
                src_rank=ep_axis_rank,
                top_k=int(model.top_k),
                experts_per_rank=experts_per_rank,
                moe_ep_size=int(moe_ep_size),
                device=device,
                scale_dtype=act_dtype,
            )
        except Exception as exc:
            return None, _RoutingSkip(f"routing materialise error: {type(exc).__name__}: {exc}")
        inner_backend = getattr(moe, "backend", moe)
        routing_path = (
            "supplied_topk_run_moe"
            if isinstance(inner_backend, TRTLLMGenFusedMoE)
            else "supplied_topk_apply"
        )
        return (
            _RoutingInputs(
                router_logits=base_router_logits,
                materialized_ids=ids,
                materialized_scales=scales,
                realization_status="forced_exact",
                realization_reason=(
                    "forced routing_mode: top-k ids and uniform 1/top_k scales materialised "
                    "from RoutingPlan; native fused scoring is intentionally bypassed"
                ),
                routing_path=routing_path,
            ),
            None,
        )

    # Native mode: synthesise router_logits that drive the production routing
    # kernel toward the plan; the path is "logits_native" when the projection
    # is exact and "logits_projected" when the routing method cannot represent
    # the plan exactly.
    try:
        new_logits, projection_status, projection_reason = _project_router_logits_for_plan(
            routing_plan,
            src_rank=ep_axis_rank,
            routing_method=moe.routing_method,
            num_experts=int(model.num_experts),
            top_k=int(model.top_k),
            experts_per_rank=experts_per_rank,
            moe_ep_size=int(moe_ep_size),
            device=device,
            dtype=routing_logits_dtype,
        )
    except Exception as exc:
        return None, _RoutingSkip(f"native logits projection error: {type(exc).__name__}: {exc}")

    # In attention-DP + MoE-TP layouts (DTP / CUSTOM-DP), _project_router_logits
    # returns logits shaped [agg_tokens, E] covering all DP shards aggregated
    # onto ep_axis_rank.  The MoE internally allgathers each rank's local
    # router_logits before routing, so each rank must supply only its local
    # slice [offset_r : offset_r + n_r] of the full projected tensor.
    world_size_inferred = len(routing_plan.per_rank_num_tokens)
    if enable_attention_dp and int(moe_ep_size) < world_size_inferred:
        offset = sum(
            routing_plan.per_rank_num_tokens[s]
            for s in range(world_size_inferred)
            if s % int(moe_ep_size) == ep_axis_rank and s < rank
        )
        local_n = routing_plan.per_rank_num_tokens[rank]
        new_logits = new_logits[offset : offset + local_n]

    if projection_status != "exact" and rc_spec.projection_policy == "reject":
        return None, _RoutingSkip(
            skip_reason=(
                f"routing_realization rejected by projection_policy=reject: {projection_reason}"
            ),
            rejected_reason=projection_reason,
        )

    warnings: List[str] = []
    if projection_status != "exact":
        warnings.append(f"routing_realization={projection_status}: {projection_reason}")

    return (
        _RoutingInputs(
            router_logits=new_logits,
            realization_status=projection_status,
            realization_reason=projection_reason,
            routing_path=("logits_native" if projection_status == "exact" else "logits_projected"),
            warnings=warnings,
        ),
        None,
    )


def _observe_routing_plan(
    *,
    routing_plan: RoutingPlan,
    model: ModelSpec,
    moe_ep_size: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Materialise the plan on every EP source rank and aggregate the totals.

    We re-materialise on rank 0 (CPU) for *every* EP source instead of relying
    on MPI gather: in DTP/TTP modes multiple world ranks share an EP rank, so
    a naive allgather would double-count.
    """
    experts_per_rank = int(model.num_experts) // int(moe_ep_size)
    per_rank_ids: List[Any] = []
    for src in range(int(moe_ep_size)):
        try:
            src_ids, _ = _materialize_selected_experts_for_rank(
                routing_plan,
                src_rank=src,
                top_k=int(model.top_k),
                experts_per_rank=experts_per_rank,
                moe_ep_size=int(moe_ep_size),
                device=torch.device("cpu"),
                scale_dtype=torch.float32,
            )
        except Exception:
            src_ids = torch.empty((0, int(model.top_k)), dtype=torch.int32)
        per_rank_ids.append(src_ids)
    return _observe_routing_metrics(routing_plan, per_rank_ids, experts_per_rank, int(moe_ep_size))


def _finalize_routing_control_block(
    *,
    result: RunResult,
    rc_spec: RoutingControlSpec,
    routing_plan: RoutingPlan,
    routing_inputs: _RoutingInputs,
    model: ModelSpec,
    moe_ep_size: int,
    per_rank: List[int],
    enable_perfect_router: bool,
    act_dtype: torch.dtype,
) -> None:
    observed_slot, observed_token, observed_hist = _observe_routing_plan(
        routing_plan=routing_plan,
        model=model,
        moe_ep_size=int(moe_ep_size),
    )
    observations = _RoutingObservations(
        slot=observed_slot,
        token=observed_token,
        hist=observed_hist,
        num_chunks=result.num_chunks,
    )
    result.routing_control = _build_routing_control_block(
        spec=rc_spec,
        plan=routing_plan,
        observations=observations,
        routing_path=routing_inputs.routing_path,
        realization_status=routing_inputs.realization_status,
        realization_reason=routing_inputs.realization_reason,
        enable_perfect_router=enable_perfect_router,
        max_num_tokens_per_rank=max(per_rank) if per_rank else 0,
        warnings=routing_inputs.warnings,
        scale_dtype=act_dtype,
        moe_ep_size=int(moe_ep_size),
    )


def _resolve_layout_and_plan(
    *,
    result: RunResult,
    model: ModelSpec,
    workload: WorkloadSpec,
    config: ConfigSpec,
    world_size: int,
    rc_spec: RoutingControlSpec,
    rc_active: bool,
) -> Union[RunResult, Tuple[int, List[int], Optional[RoutingPlan]]]:
    """Step 1 of ``_run_one_candidate``: resolve layout, routing plan, per-rank tokens.

    Resolves the EP-axis ``moe_ep_size`` from the candidate config (the
    Mapping object built later will agree), validates that routing-control
    cases satisfy ``moe_ep_size == world_size`` (so the dispatch_matrix axis
    aligns with the world-rank token distribution), and either builds the
    canonical ``RoutingPlan`` from ``rc_spec`` or falls back to a uniform
    per-rank token split.

    Returns either:
    - ``RunResult`` (already short-circuited) on any layout/plan error, or
    - ``(moe_ep_size, per_rank, routing_plan)`` on success.

    Routing-control candidates return a world-axis ``per_rank`` list because
    non-EP layouts are skipped before plan construction. Non-routing-control
    candidates use the regular balanced world-rank split.
    """
    try:
        moe_ep_size, _moe_tp_size, _enable_dp = _resolve_mapping_layout(config, world_size)
    except ValueError as exc:
        return _short_circuit(result, "skipped", str(exc))

    routing_plan: Optional[RoutingPlan] = None
    if rc_active:
        try:
            routing_plan = _build_routing_plan(
                rc_spec,
                num_tokens=int(workload.num_tokens),
                world_size=world_size,
                top_k=int(model.top_k),
                num_experts=int(model.num_experts),
                moe_ep_size=int(moe_ep_size),
                enable_dp=bool(_enable_dp),
            )
        except Exception as exc:
            reason = f"routing plan error: {type(exc).__name__}: {exc}"
            _maybe_print_rank0(f"[bench_moe] {reason}")
            return _short_circuit(result, "skipped", reason)
        per_rank = list(routing_plan.per_rank_num_tokens)
    else:
        per_rank = _per_rank_tokens(workload, world_size, enable_dp=bool(_enable_dp))

    return int(moe_ep_size), per_rank, routing_plan


def _make_candidate_input_seed(
    *,
    random_seed: int,
    rank: int,
    model: ModelSpec,
    workload: WorkloadSpec,
    per_rank: List[int],
    local_num_tokens: int,
) -> int:
    """Build a stable per-workload input seed shared across runtime candidates."""
    payload = {
        "random_seed": int(random_seed),
        "rank": int(rank),
        "model": model.to_dict(),
        "workload": workload.to_dict(per_rank_num_tokens=per_rank),
        "local_num_tokens": int(local_num_tokens),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)


def _run_one_candidate(
    *,
    model: ModelSpec,
    workload: WorkloadSpec,
    config: ConfigSpec,
    world_size: int,
    rank: int,
    device: torch.device,
    act_dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    warmup: int,
    iters: int,
    fast_autotune: bool,
    analysis: Tuple[str, ...],
    cupti_ctx: Optional[Any],
    random_seed: int,
    input_cache: Optional[_InputCache],
    enable_perfect_router_requested: bool,
    nsys: bool = False,
) -> RunResult:
    """Build, autotune, and time one ``ConfigSpec`` candidate.

    Always returns a ``RunResult``; failures are encoded in the ``status`` /
    ``skip_reason`` fields so the caller can write a row even for a failed
    case. Every early-return path goes through ``_short_circuit`` so the
    per-rank status allgather completes on every rank.

    Pipeline:
        Step 1  Resolve EP/TP layout and routing plan (if routing-control)
        Step 2  Build mapping, configure AutoTuner, force comm method
        Step 3  Build the MoE module and validate the actual backend
        Step 4  Synthesise inputs and (if routing-control) pick routing inputs
        Step 5  Run autotune and time the forward (eager or CUDA graph)
        Step 6  Aggregate latency / kernels / bottleneck and routing observation
    """
    result = RunResult(model=model, workload=workload, config=config)
    rc_spec = workload.routing_control
    rc_active = rc_spec.is_active

    # ---- Step 1: layout + routing plan ----------------------------------
    layout = _resolve_layout_and_plan(
        result=result,
        model=model,
        workload=workload,
        config=config,
        world_size=world_size,
        rc_spec=rc_spec,
        rc_active=rc_active,
    )
    if isinstance(layout, RunResult):
        return layout
    moe_ep_size, per_rank, routing_plan = layout

    result.per_rank_num_tokens = list(per_rank)
    local_num_tokens = per_rank[rank] if rank < len(per_rank) else 0
    all_rank_num_tokens = list(per_rank)

    result.instrumentation = _initial_instrumentation(analysis, config, cupti_ctx, nsys)

    # ---- Step 2: mapping + AutoTuner + comm env -------------------------
    try:
        mapping = _build_mapping_from_config(config, world_size)
    except ValueError as exc:
        return _short_circuit(result, "skipped", str(exc))

    result.moe_ep_size = int(mapping.moe_ep_size)
    result.moe_tp_size = int(mapping.moe_tp_size)
    result.enable_attention_dp = bool(mapping.enable_attention_dp)

    # TEP/TTP (no attention DP): no cross-rank dispatch; the scheduler fills
    # all_rank_num_tokens from x.shape[0]. Pass None to follow that path.
    if not mapping.enable_attention_dp:
        all_rank_num_tokens = None

    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()

    prev_force_comm = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
    _force_comm_env(config.comm_method, prev_force_comm)

    enable_perfect_router = _pick_enable_perfect_router(
        rc_spec, bool(enable_perfect_router_requested)
    )

    moe = None
    try:
        # ---- Step 3: build MoE module and validate ----------------------
        try:
            moe, _ = _build_moe_module(
                model=model,
                config=config,
                mapping=mapping,
                moe_backend=config.backend,
                use_cuda_graph=bool(config.cuda_graph),
                # Symmetric-memory comm backends (e.g. NVLINK_ONE_SIDED) size their
                # workspace from max_num_tokens and require every rank to allocate the
                # same size, so use the global per-rank maximum rather than this rank's
                # local token count (which differs under uneven attention-DP shards).
                max_num_tokens=max(int(max(per_rank)) if per_rank else 0, 1),
                use_low_precision_moe_combine=bool(config.use_low_precision_moe_combine),
                enable_perfect_router=enable_perfect_router,
                dtype=act_dtype,
                routing_logits_dtype=routing_logits_dtype,
                device=device,
            )
        except Exception as exc:
            reason = f"build error: {type(exc).__name__}: {exc}"
            _maybe_print_rank0(f"[bench_moe] build failed: {reason}")
            return _short_circuit(result, "failed", reason)

        result.actual_backend = _backend_name_from_module(moe)
        result.scheduler_kind = _scheduler_kind_name(moe)
        result.actual_comm_method = _comm_method_name(moe)
        result.num_chunks = _calculate_num_chunks_safe(moe, all_rank_num_tokens)

        if result.actual_backend != config.backend.upper():
            reason = f"requested backend {config.backend!r} fell back to {result.actual_backend!r}"
            _maybe_print_rank0(f"[bench_moe] {reason}")
            return _short_circuit(result, "skipped", reason)

        # ---- Step 4: synthetic inputs + routing-control routing inputs --
        input_seed = _make_candidate_input_seed(
            random_seed=int(random_seed),
            rank=rank,
            model=model,
            workload=workload,
            per_rank=per_rank,
            local_num_tokens=local_num_tokens,
        )
        result.instrumentation["input_seed"] = int(input_seed)
        x, router_logits = _make_inputs(
            local_num_tokens,
            model.hidden_size,
            model.num_experts,
            act_dtype,
            routing_logits_dtype,
            device,
            seed=input_seed,
            cache=input_cache,
        )

        routing_inputs: Optional[_RoutingInputs] = None
        if rc_active and routing_plan is not None:
            routing_inputs, rc_skip = _select_routing_inputs(
                moe=moe,
                model=model,
                rc_spec=rc_spec,
                routing_plan=routing_plan,
                rank=rank,
                moe_ep_size=int(moe_ep_size),
                enable_attention_dp=bool(result.enable_attention_dp),
                base_router_logits=router_logits,
                device=device,
                act_dtype=act_dtype,
                routing_logits_dtype=routing_logits_dtype,
            )
            if routing_inputs is None:
                assert rc_skip is not None
                _maybe_print_rank0(f"[bench_moe] {rc_skip.skip_reason}")
                # projection_policy=reject: keep a routing_control block on the
                # row so dashboards still see the rejected case.
                if rc_skip.rejected_reason is not None:
                    result.routing_control = _build_empty_routing_control_block_for_rejection(
                        rc_spec=rc_spec,
                        routing_plan=routing_plan,
                        model=model,
                        moe_ep_size=int(moe_ep_size),
                        per_rank=per_rank,
                        num_chunks=result.num_chunks,
                        rejected_reason=rc_skip.rejected_reason,
                        enable_perfect_router=enable_perfect_router,
                        act_dtype=act_dtype,
                    )
                return _short_circuit(result, "skipped", rc_skip.skip_reason)
            router_logits = routing_inputs.router_logits

        materialized_ids = routing_inputs.materialized_ids if routing_inputs else None
        materialized_scales = routing_inputs.materialized_scales if routing_inputs else None

        # ---- Step 5: autotune + timed forward ---------------------------
        with _maybe_install_routing_control_patch(
            moe,
            materialized_ids,
            materialized_scales,
            active=(rc_active and rc_spec.routing_mode == "forced"),
        ):
            try:
                autotune_status = _run_autotune(
                    moe, x, router_logits, all_rank_num_tokens, bool(fast_autotune)
                )
            except Exception as exc:
                autotune_status = f"failed:{type(exc).__name__}: {exc}"
                _maybe_print_rank0(f"[bench_moe] autotune skipped: {type(exc).__name__}: {exc}")
            result.instrumentation["autotune_status"] = autotune_status

            try:
                if config.cuda_graph:
                    fwd_times_ms, detailed_stats = _time_moe_forward_cuda_graph(
                        moe,
                        x,
                        router_logits,
                        all_rank_num_tokens,
                        warmup=int(warmup),
                        iters=int(iters),
                        cupti_ctx=cupti_ctx,
                        nsys=nsys,
                    )
                else:
                    fwd_times_ms, detailed_stats = _time_moe_forward_eager(
                        moe,
                        x,
                        router_logits,
                        all_rank_num_tokens,
                        warmup=int(warmup),
                        iters=int(iters),
                        collect_kernels="kernels" in analysis,
                        nsys=nsys,
                    )
            except Exception as exc:
                reason = f"timed phase error: {type(exc).__name__}: {exc}"
                _maybe_print_rank0(f"[bench_moe] {reason}\n{traceback.format_exc()}")
                return _short_circuit(result, "failed", reason)

        # ---- Step 6: aggregate latency, kernels, routing observation ----
        # The comm factory may swap moe.comm to AllGatherReduceScatter inside
        # dispatch, so refresh ``actual_comm_method`` after the first forward.
        result.actual_comm_method = _comm_method_name(moe)

        per_rank_iters = _gather_per_iteration_times(fwd_times_ms)
        result.latency_ms = _build_latency_block(per_rank_iters)

        if "kernels" in analysis:
            result.kernel_breakdown, raw_kernel_times = _gather_kernel_timing_blocks(detailed_stats)
        else:
            result.kernel_breakdown = {"moe_forward_kernels": [], "other_kernels": []}
            raw_kernel_times = {"moe_forward_kernels": [], "other_kernels": []}
        result.raw_data = _build_raw_data_block(per_rank_iters, raw_kernel_times)

        # Phase markers live in moe_scheduler.py (Phase 5 of the design); not
        # implemented yet, so emit an empty agg/per_rank with a stable shape.
        result.phase_times_ms = {"agg": {}, "per_rank": {}}
        result.overlap = {"overlap_ms": None, "overlap_ratio": None}
        result.bottleneck = _classify_bottleneck(
            result.phase_times_ms["agg"], result.kernel_breakdown, result.latency_ms["score"]
        )

        if rc_active and routing_plan is not None and routing_inputs is not None:
            _finalize_routing_control_block(
                result=result,
                rc_spec=rc_spec,
                routing_plan=routing_plan,
                routing_inputs=routing_inputs,
                model=model,
                moe_ep_size=int(moe_ep_size),
                per_rank=per_rank,
                enable_perfect_router=enable_perfect_router,
                act_dtype=act_dtype,
            )

        result.status_per_rank = _gather_status_per_rank("success")
        return result
    finally:
        # Always free GPU memory and restore the per-case env var so the next
        # candidate runs from a clean state.
        if moe is not None:
            try:
                moe.destroy()
            except Exception:
                pass
        if prev_force_comm is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev_force_comm
