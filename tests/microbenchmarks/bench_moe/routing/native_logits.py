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

"""Native logits projection and supplied-topk forward patches.

When ``RoutingControlSpec.routing_mode == "native"`` the benchmark constructs
synthetic ``router_logits`` that drive the production routing kernels to the
requested plan (and labels the outcome ``"exact"`` / ``"projected"`` /
``"rejected"`` based on the routing method's capability). When
``routing_mode == "forced"`` the routing kernels are bypassed entirely via
the supplied-topk patches in this module.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm.tools.layer_wise_benchmarks.runner import make_forward_impl_check

from .builders import RoutingPlan
from .materialize import _materialize_selected_experts_for_rank

_NATIVE_PROJECTION_CAPABILITIES: Dict[str, str] = {
    "DefaultMoeRoutingMethod": "exact_ids",
    "RenormalizeMoeRoutingMethod": "exact",
    "RenormalizeNaiveMoeRoutingMethod": "exact",
    "SigmoidRenormMoeRoutingMethod": "exact_ids",
    "Llama4RenormalizeMoeRoutingMethod": "top1_exact",
    "MiniMaxM2MoeRoutingMethod": "exact_with_zero_bias",
    "DeepSeekV3MoeRoutingMethod": "projected_or_exact",
    "SparseMixerMoeRoutingMethod": "unsupported",
}


def _classify_native_projection(
    *,
    routing_method,
    ids: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> Tuple[str, str]:
    """Return projection status/reason for a native routing method."""
    method_name = type(routing_method).__name__
    capability = _NATIVE_PROJECTION_CAPABILITIES.get(method_name, "unsupported")
    status = "exact"
    reason = "high/low logits drive top-k to plan"

    if capability == "exact":
        return status, reason
    if capability == "exact_ids":
        return (
            status,
            "expert ids exactly realised; selected_scales follow native routing kernel and are not "
            "matrix-controlled",
        )
    if capability == "top1_exact":
        if top_k > 1:
            return (
                "projected",
                "Llama4 native realisation is only exact for top1; multi-target plans are projected",
            )
        return status, reason
    if capability == "exact_with_zero_bias":
        return status, "MiniMax2 exact realisation assumes zero score-correction bias"
    if capability == "projected_or_exact":
        routing_impl = getattr(routing_method, "routing_impl", None)
        n_group = getattr(routing_impl, "n_group", 1) if routing_impl is not None else 1
        topk_group = getattr(routing_impl, "topk_group", 1) if routing_impl is not None else 1
        if n_group > 1 and topk_group >= 1:
            experts_per_group = num_experts // n_group
            ids_cpu = ids.detach().cpu().tolist()
            for row in ids_cpu:
                groups = {int(eid) // experts_per_group for eid in row}
                if len(groups) > topk_group:
                    return (
                        "projected",
                        f"DeepSeekV3 grouped routing: row needs experts in {len(groups)} groups "
                        f"but topk_group={topk_group}",
                    )
        return status, reason
    if capability == "unsupported":
        return (
            "projected",
            f"{method_name} native logits realisation is unsupported in v1; falling back to high/low logits",
        )
    return "projected", f"{method_name}: unknown capability"


def _project_router_logits_for_plan(
    plan: RoutingPlan,
    src_rank: int,
    routing_method,
    num_experts: int,
    top_k: int,
    experts_per_rank: int,
    moe_ep_size: int,
    device: torch.device,
    dtype: torch.dtype,
    high_logit: float = 10.0,
    low_logit: float = -10.0,
) -> Tuple[torch.Tensor, str, str]:
    """Build router_logits for ``routing_method.apply`` matching the plan.

    Construct logits such that ``routing_method.apply`` yields a top-k matching
    ``plan``'s ``[local_num_tokens, top_k]`` materialised expert ids.

    Returns ``(router_logits, status, reason)`` where ``status`` is one of
    ``"exact"``, ``"projected"``, or ``"rejected"``.
    """
    # Derive the effective token count from the dispatch-matrix row sum.
    # In MoE-TP + attention-DP layouts (DTP / CUSTOM-DP) the row sum equals
    # the aggregated source tokens for the EP rank (which is what the router
    # sees after the in-MoE allgather), while per_rank_num_tokens[src_rank]
    # would only cover one DP shard.
    row_sum = sum(plan.dispatch_matrix[src_rank])
    local_num_tokens = row_sum // max(top_k, 1) if row_sum > 0 else 0
    if local_num_tokens == 0:
        return (
            torch.empty((0, num_experts), dtype=dtype, device=device),
            "exact",
            "no local tokens; trivial logits",
        )

    # Materialise target experts using the canonical plan, then construct
    # logits that drive the routing kernels towards those experts.
    ids, _ = _materialize_selected_experts_for_rank(
        plan,
        src_rank=src_rank,
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        moe_ep_size=moe_ep_size,
        device=device,
        scale_dtype=dtype if dtype.is_floating_point else torch.bfloat16,
    )

    # Base low / high pattern with a small monotone perturbation by k index so
    # that ties inside a row are broken consistently.
    logits = torch.full((local_num_tokens, num_experts), low_logit, dtype=dtype, device=device)
    k_offsets = torch.linspace(0.0, 1.0, steps=top_k, device=device, dtype=dtype) * 0.01
    # Score per slot decreases with k_idx so that top-k tie-breaking yields the
    # same expert ordering when scales are derived from logits.
    score_per_k = high_logit + (k_offsets.flip(dims=(0,)) - 0.005)
    row_idx = (
        torch.arange(local_num_tokens, device=device).unsqueeze(1).expand(local_num_tokens, top_k)
    )
    logits[row_idx, ids.long()] = score_per_k.unsqueeze(0).expand_as(ids).to(dtype)

    status, reason = _classify_native_projection(
        routing_method=routing_method, ids=ids, num_experts=num_experts, top_k=top_k
    )
    return logits, status, reason


def _align_topk_to_batch(
    local: torch.Tensor, scales: torch.Tensor, batch_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align materialised routing tensors to runtime batch rows.

    CUDA graph capture may pad or trim the local batch dimension. Repeating the
    final row is only used for over-allocation and keeps the synthetic routing
    payload well-formed without changing the steady-state path.
    """
    if local.shape[0] == batch_rows:
        return local, scales
    if local.shape[0] >= batch_rows:
        return local[:batch_rows], scales[:batch_rows]
    pad_rows = batch_rows - local.shape[0]
    return (
        torch.cat([local, local[-1:].expand(pad_rows, -1).clone()], dim=0),
        torch.cat([scales, scales[-1:].expand(pad_rows, -1).clone()], dim=0),
    )


def _make_supplied_topk_run_moe(
    moe_module,
    run_moe_orig,
    materialized_ids: torch.Tensor,
    materialized_scales: torch.Tensor,
):
    """Return a ``run_moe`` wrapper that injects pre-materialised top-k tensors.

    Mirrors the ``make_balanced_run_moe`` helper in the layer-wise benchmark
    runner, but feeds the routing-control plan instead of the legacy
    balanced/imbalanced selection helpers.
    """

    def supplied_run_moe(
        x, token_selected_experts, token_final_scales, x_sf, router_logits, do_finalize, moe_output
    ):
        if getattr(moe_module, "_routing_results_replaced_at", None) is not None:
            return run_moe_orig(
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                router_logits,
                do_finalize,
                moe_output,
            )
        local, scales = _align_topk_to_batch(materialized_ids, materialized_scales, x.shape[0])
        local = local.to(device=x.device, dtype=torch.int32)
        scales = scales.to(device=x.device)
        final_hidden_states = run_moe_orig(x, local, scales, x_sf, None, do_finalize, moe_output)
        if not do_finalize:
            final_hidden_states = (
                final_hidden_states[0],
                scales,
                final_hidden_states[2],
            )
        moe_module._routing_results_replaced_at = "make_supplied_topk_run_moe"
        return final_hidden_states

    return supplied_run_moe


def _make_supplied_topk_apply(
    moe_module,
    materialized_ids: torch.Tensor,
    materialized_scales: torch.Tensor,
):
    """Return a ``routing_method.apply`` wrapper returning the plan directly."""

    def supplied_apply(router_logits):
        local = materialized_ids
        scales = materialized_scales
        if router_logits is not None:
            local, scales = _align_topk_to_batch(local, scales, router_logits.shape[0])
        device = router_logits.device if router_logits is not None else local.device
        moe_module._routing_results_replaced_at = "make_supplied_topk_apply"
        return local.to(device=device, dtype=torch.int32), scales.to(device=device)

    return supplied_apply


@contextlib.contextmanager
def _maybe_install_routing_control_patch(
    moe,
    materialized_ids: Optional[torch.Tensor],
    materialized_scales: Optional[torch.Tensor],
    active: bool,
):
    """Install supplied-topk patches when routing control is active in forced mode.

    For non-TRTLLM backends we override ``routing_method.apply`` to return the
    pre-materialised ``(ids, scales)`` pair. For ``TRTLLMGenFusedMoE`` the
    fused TEP path needs ``run_moe`` to be patched as well, mirroring the
    layer-wise benchmark's ``make_balanced_run_moe`` flow.

    ``active=False`` makes this a no-op pass-through so legacy callers keep
    behaving exactly as before.
    """
    if not active or materialized_ids is None or materialized_scales is None:
        yield
        return

    routing_target = moe
    apply_method_orig = routing_target.routing_method.apply
    inner_backend = getattr(moe, "backend", moe)
    run_moe_orig = None
    forward_impl_orig = moe.forward_impl

    try:
        routing_target.routing_method.apply = _make_supplied_topk_apply(
            routing_target, materialized_ids, materialized_scales
        )

        if isinstance(inner_backend, TRTLLMGenFusedMoE):
            run_moe_orig = inner_backend.run_moe
            inner_backend.run_moe = _make_supplied_topk_run_moe(
                inner_backend, run_moe_orig, materialized_ids, materialized_scales
            )

        moe.forward_impl = make_forward_impl_check(moe, forward_impl_orig)
        yield
    finally:
        routing_target.routing_method.apply = apply_method_orig
        if run_moe_orig is not None:
            inner_backend.run_moe = run_moe_orig
        moe.forward_impl = forward_impl_orig
