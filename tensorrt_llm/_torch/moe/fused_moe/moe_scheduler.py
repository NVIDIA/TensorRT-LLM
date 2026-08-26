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

"""MoE forward-execution schedulers.

ConfigurableMoE owns module lifecycle (backend creation, attribute sync,
weight loading delegation, comm strategy lifetime, EPLB init, repeat_idx
advancement, DWDP record). Schedulers own forward-time decisions: padding,
chunking, communication ordering, EPLB hook ordering, and backend
``run_moe`` invocation.

Schedulers are read-mostly with respect to ``ConfigurableMoE``: they may
call ``moe.X`` helpers and read ``moe.<attribute>``, but must NOT write
``moe.repeat_idx`` (advanced by the wrapper) and must only mutate
``moe.comm`` through ``moe.determine_communication_method`` (the documented
AllToAll -> AllGather fallback). See MOE_SCHEDULER_DESIGN.md for the full
contract.

Two schedulers exist today, distinguished by where the cross-rank EP
exchange runs:

- ``ExternalCommMoEScheduler``: comm lives outside the MoE kernel; the
  scheduler issues ``Communication.dispatch`` / ``Communication.combine``
  from the host with per-chunk EPLB hooks and optional multi-stream
  chunk overlap.
- ``FusedCommMoEScheduler``: comm is fused into the backend's fused
  kernel (DeepGEMM ``fp8_fp4_mega_moe``-style "MegaMoE") via NVLink
  SymmBuffer; no host comm, lockstep chunk launches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from tensorrt_llm._torch.route_capture import RouteCapture  # R3

from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.utils import EventType, Fp4QuantizedTensor
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from .communication import DeepEP, DeepEPLowLatency, NcclEP, NVLinkOneSided, NVLinkTwoSided
from .communication.nvlink_two_sided_flashinfer import NVLinkTwoSidedFlashinfer
from .fused_moe_cutlass import raise_moe_lora_multichunk_unsupported
from .impl_contract import MoECommPlan, MoERunContext
from .interface import FORCE_SEPARATED_ROUTING, MoESchedulerKind

__all__ = [
    "MoEScheduler",
    "ExternalCommMoEScheduler",
    "FusedCommMoEScheduler",
    "create_moe_scheduler",
]

if TYPE_CHECKING:
    from .configurable_moe import ConfigurableMoE


class MoEScheduler(ABC):
    """Forward-execution strategy for ConfigurableMoE.

    Stateless w.r.t. model configuration. Holds a back-reference to the
    owning ``ConfigurableMoE`` and reads (but does not write) wrapper
    state. See module docstring for the contract.
    """

    def __init__(self, moe: "ConfigurableMoE") -> None:
        self.moe = moe

    @abstractmethod
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: Optional[List[int]],
        use_dp_padding: Optional[bool],
        input_ids: Optional[torch.Tensor] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor: ...


# ============================================================================
# External-comm scheduler
# ============================================================================


class ExternalCommMoEScheduler(MoEScheduler):
    """External-comm forward path: host-side dispatch/combine + per-chunk EPLB hooks.

    Steps:

    1. Fill ``all_rank_num_tokens`` with local token count when missing.
    2. Apply DP padding metadata when requested.
    3. Compute ``num_chunks`` via ``moe.calculate_num_chunks``.
    4. Validate / fallback comm strategy via
       ``moe.determine_communication_method``.
    5. Dispatch to single- or multi-chunk implementation.
    6. Truncate DP padding from outputs.

    ``repeat_idx`` advancement and DWDP record are owned by
    ``ConfigurableMoE.forward_impl`` after the scheduler returns.

    ``TRTLLM_ENABLE_DUMMY_ALLREDUCE`` is a performance-debug knob that
    injects symmetric synchronization around dispatch/combine. It helps
    separate MoE communication timing from rank skew or load-imbalance
    artifacts when analyzing traces.
    """

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: Optional[List[int]],
        use_dp_padding: Optional[bool],
        input_ids: Optional[torch.Tensor] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        moe = self.moe

        # ========== Step 1: Handle padding ==========
        if all_rank_num_tokens is None:
            all_rank_num_tokens = [x.shape[0]]

        all_rank_max_num_tokens = max(all_rank_num_tokens)

        if use_dp_padding:
            all_rank_num_tokens_padded = [all_rank_max_num_tokens] * len(all_rank_num_tokens)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        # ========== Step 2: Determine communication method ==========
        num_chunks = moe.calculate_num_chunks(all_rank_num_tokens_padded)

        if (
            num_chunks > 1
            and moe.backend.capabilities.supports_moe_lora
            and moe.backend._moe_lora_active(lora_params)
        ):
            raise_moe_lora_multichunk_unsupported(num_chunks)

        # ========== 0-token rank deadlock fix ==========
        # When some ranks have 0 tokens in single-chunk forward with collective comm,
        # those ranks hang in CUDA kernels (e.g. NVFP4 quantize_input with 0-row tensor)
        # before reaching moe.comm.dispatch(), causing NCCL AllGather deadlock on
        # non-zero ranks. Fix: activate DP padding uniformly across all ranks so every
        # rank uses sizes=None (uniform allgather) and pads x/router_logits to max_tokens.
        # Mirrors the empty-chunk substitution in _forward_multiple_chunks (line ~597-620).
        # Existing truncation at Step 4 discards dummy-token outputs automatically.
        # NOTE: kept after the multi-chunk rejection above so unit tests that stub `moe`
        # with a minimal namespace (no `.comm`/`.use_dp`) still exercise that path; only
        # relevant to single-chunk collective comm anyway.
        if (
            moe.comm is not None
            and moe.use_dp
            and all_rank_max_num_tokens > 0
            and not use_dp_padding
            and any(t == 0 for t in all_rank_num_tokens_padded)
        ):
            use_dp_padding = True
            all_rank_num_tokens_padded = [all_rank_max_num_tokens] * len(all_rank_num_tokens)
            local_n = x.shape[0]
            if local_n < all_rank_max_num_tokens:
                pad = all_rank_max_num_tokens - local_n
                x = torch.cat([x, x.new_zeros((pad, x.shape[1]))], dim=0)
                router_logits = torch.cat(
                    [router_logits, router_logits.new_zeros((pad, router_logits.shape[1]))], dim=0
                )

        # May fall back AllToAll -> AllGather; this is the only sanctioned
        # mutation of ``moe.comm`` from a scheduler.
        moe.determine_communication_method(all_rank_num_tokens_padded, num_chunks)

        # ========== Step 3: Execute MoE computation ==========
        if num_chunks == 1:
            outputs = self._forward_single_chunk(
                x,
                router_logits,
                output_dtype,
                all_rank_num_tokens_padded,
                use_dp_padding,
                do_finalize,
                input_ids,
                lora_params=lora_params,
            )
        else:
            outputs = self._forward_multiple_chunks(
                x,
                router_logits,
                num_chunks,
                output_dtype,
                all_rank_num_tokens_padded,
                use_dp_padding,
                do_finalize,
                input_ids,
                lora_params=lora_params,
            )

        # ========== Step 4: Truncate DP padding ==========
        if moe.use_dp and moe.parallel_size > 1:
            outputs = outputs[: all_rank_num_tokens[moe.mapping.tp_rank]]

        return outputs

    # ------------------------------------------------------------------
    # Communication-strategy probes (used by _forward_chunk_impl to gate
    # NVLink-specific EPLB stat-gather paths)
    # ------------------------------------------------------------------
    def _is_using_nvlink_two_sided(self) -> bool:
        return isinstance(self.moe.comm, (NVLinkTwoSided, NVLinkTwoSidedFlashinfer))

    def _is_using_nvlink_one_sided(self) -> bool:
        return isinstance(self.moe.comm, NVLinkOneSided)

    # ------------------------------------------------------------------
    # DeepGemm workspace allocation
    # ------------------------------------------------------------------
    def _prepare_workspace_deepgemm(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens: List[int],
    ) -> Optional[torch.Tensor]:
        """Single-chunk workspace for backends that ask for one; else ``None``.

        Multi-chunk execution uses ``_prepare_workspaces_for_chunk`` instead.
        """
        moe = self.moe
        if not moe.backend.input_requirement.requires_run_moe_workspace:
            return None

        num_rows = x.shape[0]
        if moe.use_dp and moe.comm is not None:
            # Communication path padding: dispatch outputs are
            # ``[num_dp_ranks * max_tokens_per_rank, ...]`` (or expert-major for
            # DeepEPLowLatency). Workspace must cover that footprint.
            if isinstance(moe.comm, DeepEPLowLatency):
                num_rows = moe.num_slots * max(all_rank_num_tokens)
            else:
                num_rows = moe._dp_padded_num_rows(all_rank_num_tokens)

        workspaces = moe.backend.get_workspaces([num_rows])
        return workspaces[0]

    def _prepare_workspaces_for_chunk(
        self,
        all_rank_num_tokens_list: List[Optional[List[int]]],
        chunk_size_list: List[int],
        use_multi_stream: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Multi-chunk workspaces for backends that ask for one; else ``(None, None)``.

        Single-chunk execution uses ``_prepare_workspace_deepgemm`` instead.
        """
        moe = self.moe
        workspace_0 = None
        workspace_1 = None

        if not moe.backend.input_requirement.requires_run_moe_workspace:
            return workspace_0, workspace_1

        # Always need at least workspace_0; reuse chunk_0 size for workspace_1
        # since chunk 0 is always >= subsequent chunks under split_chunk.
        # Mirror ``_prepare_workspace_deepgemm``: DeepEPLowLatency dispatches
        # expert-major outputs sized ``num_slots * max_tokens_per_rank`` per
        # rank (one shard per slot), while other comms produce
        # ``num_dp_ranks * max_tokens_per_rank``. Using the wrong formula
        # under-allocates the workspace for DeepEPLowLatency multi-chunk
        # runs and is caught by ``DeepGemmFusedMoE.run_moe``.
        if moe.use_dp and all_rank_num_tokens_list[0] is not None:
            max_tokens = max(all_rank_num_tokens_list[0])
            if isinstance(moe.comm, DeepEPLowLatency):
                chunk_size_0 = moe.num_slots * max_tokens
            else:
                chunk_size_0 = moe._dp_padded_num_rows(all_rank_num_tokens_list[0])
        else:
            chunk_size_0 = chunk_size_list[0]
        workspace_chunk_sizes = [chunk_size_0]

        if use_multi_stream:
            workspace_chunk_sizes.append(chunk_size_0)

        workspaces = moe.backend.get_workspaces(workspace_chunk_sizes)
        workspace_0 = workspaces[0]
        if use_multi_stream:
            workspace_1 = workspaces[1]

        return workspace_0, workspace_1

    # ------------------------------------------------------------------
    # Single / multi chunk dispatch
    # ------------------------------------------------------------------
    def _forward_single_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool],
        do_finalize: bool = True,
        input_ids: Optional[torch.Tensor] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        moe = self.moe
        is_first_call = moe.repeat_idx == 0
        is_last_call = moe.repeat_idx == moe.repeat_count - 1

        workspace = self._prepare_workspace_deepgemm(x, all_rank_num_tokens)

        return self._forward_chunk_impl(
            x,
            router_logits,
            output_dtype,
            all_rank_num_tokens,
            use_dp_padding,
            is_first_call,
            is_last_call,
            do_finalize,
            workspace=workspace,
            input_ids=input_ids,
            lora_params=lora_params,
        )

    def _forward_chunk_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: bool,
        is_first_call: bool,
        is_last_call: bool,
        do_finalize: bool = True,
        workspace: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Unified per-chunk execution flow for all external-comm backends.

        Flow:
          1. EPLB - Start wait GPU stage (first call only, dynamic only)
          2. Apply routing (only if backend supports routing separation)
          3. EPLB - Update statistics and route (only if EPLB enabled)
          4. Communication prepare phase (NVLINK two-sided only)
          5. Quantization + dispatch (pre/post-quant adaptive ordering)
          6. backend.run_moe
          7. EPLB - Start CPU stage (last call only, dynamic only)
          8. Communication combine
          9. EPLB - Done CPU stage (last call only, dynamic only)
        """
        moe = self.moe

        # ========== Step 1: EPLB - Start wait GPU stage ==========
        moe._load_balancer_start_wait_gpu_stage(is_first_call)

        # ========== Step 2: Apply routing ==========
        # External dispatch (Step 5) sends per-token expert/scale payloads, so
        # routing must be precomputed whenever a comm strategy is active — even
        # for backends whose run_moe can otherwise route internally from
        # router_logits (e.g. MarlinFusedMoE under attention-DP + EP).
        requires_separated_routing = (
            moe.backend._supports_load_balancer()
            or moe.routing_method.requires_separated_routing
            or moe.comm is not None
            or FORCE_SEPARATED_ROUTING
        )
        supports_post_quant = moe.comm is None or moe.comm.supports_post_quant_dispatch()
        used_fused_route_quant = False
        if requires_separated_routing:
            # ``MoEImplBase`` declines a fused route+quant path by default. The
            # conditions are the scheduler's: a fused result skips the separate
            # dispatch and carries no per-token scale to fold a router weight or
            # an EPLB layout into.
            if (
                supports_post_quant
                and not moe._using_load_balancer()
                and not moe.apply_router_weight_on_input
            ):
                fused_result = moe.backend.try_fused_route_quant(x, router_logits)
            else:
                fused_result = None

            if fused_result is None:
                # Separated routing: ConfigurableMoE calls routing_method.
                token_selected_experts, token_final_scales = moe.routing_method.apply(
                    router_logits, input_ids
                )
            else:
                token_selected_experts, token_final_scales, x, x_sf = fused_result
                used_fused_route_quant = True

            token_selected_experts = token_selected_experts.to(torch.int32)
            RouteCapture.capture(moe.layer_idx, token_selected_experts)  # R3 device-buffer capture

            assert token_selected_experts.shape[1] == moe.routing_method.experts_per_token
            assert token_selected_experts.shape == token_final_scales.shape
            assert token_selected_experts.dtype == torch.int32

            # Backends disagree on routing-scale precision, so the requirement
            # names the dtype instead of the backend class.
            scales_dtype = moe.backend.input_requirement.routing_scales_dtype
            if scales_dtype is not None and token_final_scales is not None:
                if scales_dtype == torch.float32:
                    # Asking for float32 is asking for routing's own
                    # full-precision output, so the cast below must be a no-op.
                    # Several routing methods take an ``output_dtype``, and one
                    # configured to a narrower type has already dropped mantissa
                    # bits that widening here cannot recover -- which is what
                    # this check catches. A narrower request (TRTLLM-Gen's
                    # bfloat16) is a deliberate conversion, not a loss.
                    assert token_final_scales.dtype == torch.float32, (
                        f"{type(moe.backend).__name__} requires float32 routing "
                        f"scales, but {type(moe.routing_method).__name__} produced "
                        f"{token_final_scales.dtype}. Casting would widen a value "
                        "that already lost precision."
                    )
                token_final_scales = token_final_scales.to(scales_dtype)

            # apply_router_weight_on_input: fuse top-k weight onto x
            if moe.apply_router_weight_on_input:
                assert x.dtype != torch.float8_e4m3fn, (
                    "Current workaround for apply_router_weight_on_input does not support fp8 input"
                )
                x = x * token_final_scales.to(x.dtype)
                # These strategies need non-None token_final_scales, so feed
                # all-ones after folding the real weights into x.
                if isinstance(moe.comm, (DeepEP, DeepEPLowLatency, NcclEP)):
                    token_final_scales = torch.ones_like(token_final_scales)
                else:
                    token_final_scales = None

        else:
            # Fused routing: backend handles routing internally; EPLB must be off.
            assert not moe._using_load_balancer(), (
                f"EPLB is enabled but backend {moe.backend.__class__.__name__} "
                f"has fused routing (does not support routing separation)"
            )
            token_selected_experts = None
            token_final_scales = None

        # ========== Step 3: EPLB - Update statistics and route ==========
        if moe.layer_load_balancer and token_selected_experts is not None:
            moe._load_balancer_done_wait_gpu_stage(is_first_call)

            # NVLink two-sided / one-sided gather EPLB stats themselves; skip the
            # base helper's own AllReduce in that case (ignore_allreduce=True).
            ignore_allreduce = (
                self._is_using_nvlink_two_sided() or self._is_using_nvlink_one_sided()
            )
            moe._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=ignore_allreduce,
            )

            token_selected_slots = moe._load_balancer_route(token_selected_experts, moe.use_dp)
        else:
            token_selected_slots = token_selected_experts

        if token_selected_slots is not None:
            ExpertStatistic.set_layer(moe.layer_idx)
            ExpertStatistic.maybe_add_info(moe.num_slots, token_selected_slots)
        token_selected_slots = get_calibrator().maybe_collect_or_replay_slots(
            moe.num_slots, token_selected_slots
        )

        # ========== Step 4: Communication prepare phase (NVLINK two-sided only) ==========
        local_statistic_tensor_for_dispatch = None
        eplb_dispatch_kwargs = {}
        should_update_eplb_after_dispatch = False
        if self._is_using_nvlink_two_sided():
            local_statistic_tensor = None
            if is_last_call:
                local_statistic_tensor = moe._load_balancer_get_local_statistic_tensor()

            # prepare_dispatch stores alltoall_info in _dispatch_state and returns gathered_stats
            gathered_stats = moe.comm.prepare_dispatch(
                token_selected_slots, all_rank_num_tokens, local_statistic_tensor
            )

            if gathered_stats is not None:
                gathered_stats = gathered_stats.view((moe.mapping.moe_ep_size, moe.num_experts))
                moe._load_balancer_update_statistic_with_gathered_statistic(gathered_stats)
        # NVLinkOneSided gathers EPLB stats inside dispatch, not prepare_dispatch
        elif self._is_using_nvlink_one_sided():
            if moe.layer_load_balancer and is_last_call:
                local_statistic_tensor_for_dispatch = (
                    moe._load_balancer_get_local_statistic_tensor()
                )
            if local_statistic_tensor_for_dispatch is not None:
                eplb_dispatch_kwargs["eplb_local_stats"] = local_statistic_tensor_for_dispatch
                should_update_eplb_after_dispatch = True

        # ========== Step 5: Quantization + dispatch (pre/post-quant adaptive ordering) ==========
        if moe.comm is not None:
            # Debug: optional dummy AllReduce to break load-balancing artifacts
            if moe.enable_dummy_allreduce:
                moe.dummy_allreduce()

            dispatch_kwargs = dict(eplb_dispatch_kwargs)
            # Only DeepEP.dispatch reads this; every other strategy absorbs it
            # through **kwargs, so the request does not need a comm-side test.
            if moe.backend.input_requirement.requires_sanitized_expert_ids:
                dispatch_kwargs["enable_sanitize_expert_ids"] = True

            if supports_post_quant:
                # Quantize -> Dispatch
                if not used_fused_route_quant:
                    x, x_sf = moe.backend.quantize_input(x)

                # W4AFP8 + DeepEPLowLatency needs pre_quant_scale_1; other strategies
                # absorb the kwarg via **kwargs so unconditional passing is safe.
                if hasattr(moe, "quant_scales") and moe.quant_scales is not None:
                    if hasattr(moe.quant_scales, "pre_quant_scale_1"):
                        dispatch_kwargs["pre_quant_scale"] = moe.quant_scales.pre_quant_scale_1
                x, x_sf, token_selected_slots, token_final_scales = moe.comm.dispatch(
                    hidden_states=x,
                    hidden_states_sf=x_sf,
                    token_selected_slots=token_selected_slots,
                    token_final_scales=token_final_scales,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    **dispatch_kwargs,
                )
                if should_update_eplb_after_dispatch:
                    gathered_stats = moe.comm.get_eplb_gathered_statistics()
                    moe._load_balancer_update_statistic_with_gathered_statistic(gathered_stats)
            else:
                # Dispatch -> Quantize
                x, x_sf, token_selected_slots, token_final_scales = moe.comm.dispatch(
                    hidden_states=x,
                    hidden_states_sf=None,  # not quantized yet
                    token_selected_slots=token_selected_slots,
                    token_final_scales=token_final_scales,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    **dispatch_kwargs,
                )
                x, x_sf = moe.backend.quantize_input(x, post_quant_comm=False)
        else:
            # No comm: just quantize
            if not used_fused_route_quant:
                x, x_sf = moe.backend.quantize_input(x, post_quant_comm=False)

        # ========== Step 6: MoE computation ==========
        # If EPLB is enabled, token_selected_slots is slot ids; otherwise expert ids.
        ctx = self._build_run_context(
            x=x,
            x_sf=x_sf,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            router_logits=router_logits,
            do_finalize=do_finalize,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            lora_params=lora_params,
        )
        final_hidden_states = moe.backend.run_moe(
            ctx,
            workspace=(
                workspace if moe.backend.input_requirement.requires_run_moe_workspace else None
            ),
        )

        # ========== Step 7: EPLB - Start CPU stage ==========
        moe._load_balancer_start_set_cpu_stage(is_last_call)

        # ========== Step 8: Communication combine ==========
        if moe.comm is not None:
            if moe.enable_dummy_allreduce:
                moe.dummy_allreduce()
            all_rank_max_num_tokens = max(all_rank_num_tokens)
            final_hidden_states = moe.comm.combine(
                final_hidden_states,
                all_rank_max_num_tokens=all_rank_max_num_tokens,
            )
        else:
            # Non-comm path: attention TP or single rank; only AllReduce if reduce_results
            if moe.parallel_size > 1 and moe.reduce_results:
                final_hidden_states = moe.all_reduce(final_hidden_states)

        # ========== Step 9: EPLB - Done CPU stage ==========
        moe._load_balancer_done_set_cpu_stage(is_last_call)

        return final_hidden_states

    def _forward_multiple_chunks(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        num_chunks: int,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool],
        do_finalize: bool = True,
        input_ids: Optional[torch.Tensor] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Multiple-chunk path with optional aux-stream overlap."""
        moe = self.moe

        # ========== Chunk preparation ==========
        if moe.use_dp:
            # DP: need all ranks' token counts for reducescatter
            all_rank_chunk_size_list = [
                moe.split_chunk(val, num_chunks) for val in all_rank_num_tokens
            ]
            all_rank_num_tokens_list = [
                [val[idx_chunk] for val in all_rank_chunk_size_list]
                for idx_chunk in range(num_chunks)
            ]
            chunk_size_list = all_rank_chunk_size_list[moe.rank]

            # AllToAll cannot consume an all-zero rank; substitute 1 token.
            if moe.enable_alltoall:
                all_rank_num_tokens_list = [
                    [1 if val == 0 else val for val in val_list]
                    for val_list in all_rank_num_tokens_list
                ]
        else:
            all_rank_num_tokens_list = [None] * num_chunks
            chunk_size_list = moe.split_chunk(x.shape[0], num_chunks)

        x_list = x.split(chunk_size_list)
        router_logits_list = router_logits.split(chunk_size_list)
        input_ids_list = (
            input_ids.split(chunk_size_list) if input_ids is not None else [None] * num_chunks
        )

        use_multi_stream = not moe.enable_alltoall and moe.aux_stream is not None

        # ========== Setup auxiliary stream ==========
        if use_multi_stream:
            moe.event_dict[EventType.Main].record()
            with torch.cuda.stream(moe.aux_stream):
                moe.event_dict[EventType.Main].wait()

        # ========== DeepGemm workspaces ==========
        workspace_0, workspace_1 = self._prepare_workspaces_for_chunk(
            all_rank_num_tokens_list, chunk_size_list, use_multi_stream
        )

        # ========== Empty-chunk substitution (DP only) ==========
        # Host-only bookkeeping, so keep it in Python state. A tensor here
        # costs an allocation per call plus a Tensor.__bool__ dispatch per
        # chunk read below, on a path that runs once per MoE layer per step.
        # It is also a latent hazard: the tensor only lands on the host
        # because no device is requested, and a CUDA one would turn each read
        # into a device-to-host sync that is illegal under CUDA Graph capture.
        chunked_used = [True] * num_chunks
        if moe.use_dp:
            # The split heuristic guarantees chunk 0 has >= 1 token, so it can
            # stand in for any empty chunk on this rank. Without substitution,
            # the per-chunk dispatch would launch with 0-token shape and the
            # peers would see a barrier mismatch.
            assert x_list[0].numel() != 0, "chunk 0 shouldn't be empty"
            x_list = list(x_list)
            router_logits_list = list(router_logits_list)
            input_ids_list = list(input_ids_list)
            for idx_chunk in range(num_chunks):
                _x = x_list[idx_chunk]
                if _x.numel() == 0:
                    chunked_used[idx_chunk] = False
                    x_list[idx_chunk] = x_list[0]
                    router_logits_list[idx_chunk] = router_logits_list[0]
                    input_ids_list[idx_chunk] = input_ids_list[0]
            # Mirror the empty-chunk substitution above into the work list:
            # all_rank_num_tokens_list feeds the varsize collectives, so every
            # rank must patch EVERY empty entry, not just its own -- the size
            # vectors have to be identical on all ranks. all_rank_chunk_size_list
            # is the untouched ground truth used to detect the empty chunks.
            for idx_chunk in range(num_chunks):
                vec = all_rank_num_tokens_list[idx_chunk]
                for j in range(len(vec)):
                    if all_rank_chunk_size_list[j][idx_chunk] == 0:
                        vec[j] = all_rank_chunk_size_list[j][0]
            x_list = tuple(x_list)
            router_logits_list = tuple(router_logits_list)
            input_ids_list = tuple(input_ids_list)

        # ========== Execute chunking with overlap ==========
        outputs_list = []
        for idx_chunk, (x_chunk, router_logits_chunk, input_ids_chunk) in enumerate(
            zip(x_list, router_logits_list, input_ids_list)
        ):
            is_first_call = idx_chunk == 0 and moe.repeat_idx == 0
            is_last_call = idx_chunk == num_chunks - 1 and moe.repeat_idx == moe.repeat_count - 1

            if use_multi_stream:
                # Alternate streams; each chunk fully owns its (forward + reducescatter).
                # Even chunks use aux_stream so chunk 0 is isolated from outer main-stream traffic.
                if idx_chunk % 2 == 0:
                    with torch.cuda.stream(moe.aux_stream):
                        outputs = self._forward_chunk_impl(
                            x_chunk,
                            router_logits_chunk,
                            output_dtype,
                            all_rank_num_tokens_list[idx_chunk],
                            use_dp_padding,
                            is_first_call,
                            is_last_call,
                            do_finalize,
                            workspace=workspace_0,
                            input_ids=input_ids_chunk,
                            lora_params=lora_params,
                        )
                else:
                    outputs = self._forward_chunk_impl(
                        x_chunk,
                        router_logits_chunk,
                        output_dtype,
                        all_rank_num_tokens_list[idx_chunk],
                        use_dp_padding,
                        is_first_call,
                        is_last_call,
                        do_finalize,
                        workspace=workspace_1,
                        input_ids=input_ids_chunk,
                        lora_params=lora_params,
                    )
            else:
                outputs = self._forward_chunk_impl(
                    x_chunk,
                    router_logits_chunk,
                    output_dtype,
                    all_rank_num_tokens_list[idx_chunk],
                    use_dp_padding,
                    is_first_call,
                    is_last_call,
                    do_finalize,
                    workspace=workspace_0,
                    input_ids=input_ids_chunk,
                    lora_params=lora_params,
                )

            if chunked_used[idx_chunk]:
                outputs_list.append(outputs)

        # ========== Wait for auxiliary stream to complete ==========
        if use_multi_stream:
            with torch.cuda.stream(moe.aux_stream):
                moe.event_dict[EventType.MoeChunkingOverlap].record()
            moe.event_dict[EventType.MoeChunkingOverlap].wait()

        outputs = torch.cat(outputs_list)
        return outputs

    # ------------------------------------------------------------------
    # Backend run_moe inputs (external-comm only)
    # ------------------------------------------------------------------
    def _plan_onesided_workspace(
        self,
        all_rank_num_tokens: Optional[List[int]],
        output_dtype: Optional[torch.dtype],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Decide the NVLinkOneSided combine payload buffer for this forward.

        Returns ``(moe_output, payload_in_workspace)``. Both are decided on
        every path, including the ones that opt out, so the flag can never be
        inherited from a previous forward.
        """
        moe = self.moe
        if not isinstance(moe.comm, NVLinkOneSided):
            return None, False

        if not moe.backend.supports_moe_output_in_alltoall_workspace():
            # Backend emits its own output tensor; a workspace buffer would be
            # left unfilled while combine() read from it.
            return None, False

        # None means "no override": the buffer matches the model output dtype.
        workspace_dtype = moe.backend.input_requirement.onesided_workspace_dtype or output_dtype

        assert all_rank_num_tokens is not None, (
            "all_rank_num_tokens must be provided for NVLinkOneSided backend"
        )
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)

        moe_output = moe.comm.get_combine_payload_tensor_in_workspace(
            runtime_max_tokens_per_rank, moe.hidden_size, workspace_dtype
        )
        return moe_output, True

    def _build_run_context(
        self,
        *,
        x: torch.Tensor,
        x_sf: Optional[torch.Tensor],
        token_selected_slots: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        router_logits: Optional[torch.Tensor],
        do_finalize: bool,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: Optional[List[int]],
        lora_params: Optional[Dict],
    ) -> MoERunContext:
        """The single ``run_moe`` argument set, identical for every backend.

        The only per-backend decision left here is dropping ``lora_params``
        for backends that do not fuse routed-expert LoRA: handing them one
        would silently produce un-adapted output.
        """
        moe = self.moe
        return MoERunContext(
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x=x,
            x_sf=x_sf,
            output_dtype=output_dtype,
            do_finalize=do_finalize,
            lora_params=lora_params if moe.backend.capabilities.supports_moe_lora else None,
            router_logits=router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            comm_plan=self._build_comm_plan(all_rank_num_tokens, output_dtype),
        )

    def _build_comm_plan(
        self,
        all_rank_num_tokens: Optional[List[int]],
        output_dtype: Optional[torch.dtype],
    ) -> MoECommPlan:
        """The comm-layer facts for this forward, derived once for every backend.

        Backends read the fields they care about and ignore the rest, so the
        set of facts no longer depends on which class is running.
        """
        moe = self.moe
        # Pre-quant dispatch: SFs arrive swizzled; post-quant dispatch: SFs
        # arrive unswizzled. Backends use this to skip a re-swizzle.
        supports_post_quant = moe.comm is not None and moe.comm.supports_post_quant_dispatch()
        moe_output, payload_in_workspace = self._plan_onesided_workspace(
            all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
        )
        if isinstance(moe.comm, NVLinkOneSided):
            # combine() still reads the flag off the strategy; the plan stays
            # the single place that decides its value.
            moe.comm.payload_in_workspace = payload_in_workspace
        return MoECommPlan(
            input_sf_swizzled=not supports_post_quant,
            enable_alltoall=moe.enable_alltoall,
            moe_output=moe_output,
            payload_in_workspace=payload_in_workspace,
        )


# ============================================================================
# Fused-comm scheduler (MegaMoE-style)
# ============================================================================


class FusedCommMoEScheduler(MoEScheduler):
    """Fused-comm scheduler: backend's fused kernel owns the EP exchange.

    Invariants (see MOE_SCHEDULER_DESIGN.md / mega_moe/CHUNKING_DESIGN.md):

    1. Reject ``Fp4QuantizedTensor`` activation; backend.quantize_input
       owns the BF16 -> FP8 conversion.
    2. Ignore ``use_dp_padding`` (no host-side cross-rank shape alignment).
    3. Use ``mapping.moe_ep_rank`` for local token count, not global rank.
    4. Strip ADP padding before splitting tensors.
    5. ``had_meta=False`` -> pass ``None`` per-chunk so inner falls back to
       ``num_tokens=x.shape[0]`` (avoids IndexError on moe_ep_rank>0).
    6. ``num_chunks = max(real_all_rank_num_tokens)`` (not the generic
       ``calculate_num_chunks``; that one falls back to ``sum()`` for
       ``comm is None`` and would diverge per rank).
    7. Launch every chunk on every EP rank, including zero-token chunks,
       so peers can cross the in-kernel NVLink barrier.
    8. No external Communication.dispatch / Communication.combine.
    9. No multi-stream chunk overlap.

    ``repeat_idx`` advancement is done by ``ConfigurableMoE.forward_impl``
    after this scheduler returns. The scheduler must not rotate
    ``moe.repeat_idx``.
    """

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: Optional[List[int]],
        use_dp_padding: Optional[bool],
        input_ids: Optional[torch.Tensor],
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Sequential multi-chunk path for MegaMoE-style backends.

        Single-chunk case is just ``num_chunks == 1`` -- no separate fast
        path. Invariants enforced here (see class docstring): identical
        ``num_chunks`` per rank computed from ``max()``, ADP padding
        stripped before splitting, zero-token chunks still launch the
        kernel for the cross-rank barrier.
        """
        del use_dp_padding  # MegaMoE has no host-side cross-rank shape alignment.

        # Fused-comm (MegaMoE) backends cannot carry LoRA adapters; routed-expert
        # MoE LoRA is supported only on the CUTLASS backend. Reject rather than
        # silently ignore.
        if lora_params:
            raise NotImplementedError(
                "Routed-expert MoE LoRA is not supported by the fused-comm "
                "(MegaMoE) scheduler; only the CUTLASS backend supports it."
            )

        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "Fused-comm MoE expects BF16 activation; "
                "quantization happens in backend.quantize_input."
            )

        x_real, rl_real, input_ids_real, real_all_rank_num_tokens, ep_rank, had_meta = (
            self._strip_adp_padding(x, router_logits, input_ids, all_rank_num_tokens)
        )
        num_chunks, x_chunks, rl_chunks, all_rank_chunk_size_list = self._compute_chunk_layout(
            x_real, rl_real, real_all_rank_num_tokens, ep_rank
        )
        input_ids_chunks = []
        if input_ids_real is not None:
            chunk_size_list = all_rank_chunk_size_list[ep_rank]
            input_ids_chunks = (
                list(input_ids_real.split(chunk_size_list)) if input_ids_real.numel() > 0 else []
            )
        outputs = self._run_chunks(
            x_chunks,
            rl_chunks,
            input_ids_chunks,
            num_chunks=num_chunks,
            x_real=x_real,
            rl_real=rl_real,
            all_rank_chunk_size_list=all_rank_chunk_size_list,
            had_meta=had_meta,
            output_dtype=output_dtype,
            do_finalize=do_finalize,
        )
        if not outputs:
            cast_dtype = output_dtype if output_dtype is not None else x.dtype
            return x.new_empty((0, x.shape[1]), dtype=cast_dtype)
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def _strip_adp_padding(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        all_rank_num_tokens: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int], int, bool]:
        """Slice ADP padding off ``x`` / ``router_logits`` using moe_ep_rank.

        SymmBuffer exchange is EP-scoped, so we index the per-rank token count
        via ``moe.mapping.moe_ep_rank``, not ``self.rank``. ``had_meta`` lets
        the per-chunk impl fall back to ``num_tokens=x.shape[0]`` (avoids
        ``[len-1 list][moe_ep_rank>0]`` IndexError when no metadata is
        provided, e.g. dummy / single-rank forwards).
        """
        moe = self.moe
        had_meta = all_rank_num_tokens is not None
        if had_meta:
            # Force plain Python int: downstream torch.Tensor.split and range()
            # reject torch 0-d tensor / numpy scalar elements, and the public
            # ``Optional[List[int]]`` type hint is not runtime-enforced.
            real_all_rank_num_tokens = [int(v) for v in all_rank_num_tokens]
            ep_rank = moe.mapping.moe_ep_rank
        else:
            real_all_rank_num_tokens = [int(x.shape[0])]
            ep_rank = 0
        real_local = real_all_rank_num_tokens[ep_rank]
        assert real_local <= x.shape[0], (
            f"real_local ({real_local}) > x.shape[0] ({x.shape[0]}); "
            "all_rank_num_tokens may not be indexed correctly."
        )
        # ADP padding stripped before split, else trailing rows silently
        # drift into chunk-0 or torch.split shape-errors.
        x_real = x[:real_local]
        rl_real = router_logits[:real_local]
        input_ids_real = input_ids[:real_local] if input_ids is not None else None
        return x_real, rl_real, input_ids_real, real_all_rank_num_tokens, ep_rank, had_meta

    def _compute_chunk_layout(
        self,
        x_real: torch.Tensor,
        rl_real: torch.Tensor,
        real_all_rank_num_tokens: List[int],
        ep_rank: int,
    ) -> Tuple[int, List[torch.Tensor], List[torch.Tensor], List[List[int]]]:
        """Compute per-rank/per-chunk shape and the actual tensor splits.

        ``num_chunks`` uses ``max()``, not ``moe.calculate_num_chunks``: the
        latter falls back to ``sum()`` when ``comm is None`` and would diverge
        per rank, breaking the in-kernel cross-rank barrier (class invariant 6).
        ``... else 0`` defends against an empty meta list (caller passing
        ``[]`` instead of ``None``); ``max([])`` would otherwise raise.

        ``all_rank_chunk_size_list[r][c]`` = tokens rank r contributes to
        chunk c. ``split_chunk`` evenly partitions ``v`` into exactly
        ``num_chunks`` pieces (zero-padded when v < num_chunks, including
        v == 0), so every row has the same length and ``chunk_size_list``
        below is this rank's row.
        """
        moe = self.moe
        real_local = real_all_rank_num_tokens[ep_rank]

        max_real = max(real_all_rank_num_tokens) if real_all_rank_num_tokens else 0
        num_chunks = max(
            1,
            (max_real + moe.moe_max_num_tokens - 1) // moe.moe_max_num_tokens,
        )

        all_rank_chunk_size_list = [
            moe.split_chunk(v, num_chunks) for v in real_all_rank_num_tokens
        ]
        chunk_size_list = all_rank_chunk_size_list[ep_rank]
        # ``else []`` shortcut for real_local == 0: equivalent to
        # x_real.split([0]*num_chunks) but skips the no-op torch call. The
        # zero-token fallback in ``_run_chunks`` then fires for every chunk.
        x_chunks = list(x_real.split(chunk_size_list)) if real_local > 0 else []
        rl_chunks = list(rl_real.split(chunk_size_list)) if real_local > 0 else []
        return num_chunks, x_chunks, rl_chunks, all_rank_chunk_size_list

    def _run_chunks(
        self,
        x_chunks: List[torch.Tensor],
        rl_chunks: List[torch.Tensor],
        input_ids_chunks: List[torch.Tensor],
        *,
        num_chunks: int,
        x_real: torch.Tensor,
        rl_real: torch.Tensor,
        all_rank_chunk_size_list: List[List[int]],
        had_meta: bool,
        output_dtype: Optional[torch.dtype],
        do_finalize: bool,
    ) -> List[torch.Tensor]:
        """Drive the per-chunk kernel launches, padding zero-token chunks.

        Stage hooks + AllReduce only fire at the (first|last) chunk of the
        (first|last) repeat, matching the external-comm path. The
        ``idx_chunk >= len(x_chunks)`` branch only triggers when
        ``real_local == 0`` (this rank has no tokens but peers do): class
        invariant 7 says launch every chunk on every EP rank so the in-kernel
        NVLink barrier (SymmBuffer collective) can synchronize.
        """
        moe = self.moe
        outputs: List[torch.Tensor] = []
        for idx_chunk in range(num_chunks):
            is_first_call = idx_chunk == 0 and moe.repeat_idx == 0
            is_last_call = idx_chunk == num_chunks - 1 and moe.repeat_idx == moe.repeat_count - 1

            if idx_chunk < len(x_chunks):
                x_chunk = x_chunks[idx_chunk]
                rl_chunk = rl_chunks[idx_chunk]
                input_ids_chunk = input_ids_chunks[idx_chunk] if input_ids_chunks else None
            else:
                # Shape ``(0, hidden_size)`` keeps dtype/device/column-width
                # intact so routing / quantize / run_moe execute as no-ops
                # without shape errors before reaching the barrier.
                x_chunk = x_real.new_empty((0, x_real.shape[1]))
                rl_chunk = rl_real.new_empty((0, rl_real.shape[1]))
                input_ids_chunk = None

            per_chunk_all_rank = (
                [lst[idx_chunk] for lst in all_rank_chunk_size_list] if had_meta else None
            )

            out_chunk = self._forward_chunk(
                x_chunk,
                rl_chunk,
                output_dtype=output_dtype,
                all_rank_num_tokens=per_chunk_all_rank,
                do_finalize=do_finalize,
                is_first_call=is_first_call,
                is_last_call=is_last_call,
                input_ids=input_ids_chunk,
            )
            outputs.append(out_chunk)
        return outputs

    def _forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: Optional[List[int]],
        do_finalize: bool,
        is_first_call: bool = True,
        is_last_call: bool = True,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a single chunk through the fused-comm backend.

        Inputs are already ADP-stripped by the caller; ``x.shape[0]`` is
        the true unpadded per-rank token count for this chunk.
        ``x.shape[0] == 0`` is valid: the kernel still launches so peers
        can cross ``nvlink_barrier``.

        EPLB hook ordering (matches ``ExternalCommMoEScheduler._forward_chunk_impl``):
        ``start_wait_gpu_stage`` -> routing -> ``done_wait_gpu_stage`` ->
        ``update_statistic(ignore_allreduce=False)`` -> ``route`` ->
        quantize -> ``run_moe`` -> ``start_set_cpu_stage`` ->
        ``done_set_cpu_stage``. ``start/done_set_cpu_stage`` are placed
        AFTER ``run_moe``; otherwise dynamic-EPLB weight migration would
        race with the fused kernel using those weights.
        """
        moe = self.moe
        assert not moe.apply_router_weight_on_input, (
            "Fused-comm MoE does not support apply_router_weight_on_input"
        )
        assert do_finalize, "Fused-comm MoE always finalizes inside the fused kernel"

        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "Fused-comm MoE expects BF16 activation; "
                "quantization happens in backend.quantize_input."
            )
        if output_dtype is None:
            output_dtype = x.dtype

        # Index per moe_ep_rank, not self.rank: SymmBuffer exchange is EP-scoped.
        if all_rank_num_tokens is not None:
            num_tokens = int(all_rank_num_tokens[moe.mapping.moe_ep_rank])
        else:
            num_tokens = x.shape[0]
        assert num_tokens <= x.shape[0], f"num_tokens ({num_tokens}) > x.shape[0] ({x.shape[0]})"

        x_chunk_real = x[:num_tokens]
        router_logits_chunk_real = router_logits[:num_tokens]
        input_ids_chunk_real = input_ids[:num_tokens] if input_ids is not None else None

        # ----- EPLB: drain previous CPU rebalance -----
        # Static EPLB early-returns inside the helper; only the dynamic
        # balancer actually waits.
        moe._load_balancer_start_wait_gpu_stage(is_first_call)

        # ----- routing -----
        # int32 matches the EPLB stats kernel contract used by the external-comm
        # path; the fused-comm backend casts to int64 internally.
        if num_tokens > 0:
            token_selected_experts, token_final_scales = moe.routing_method.apply(
                router_logits_chunk_real, input_ids_chunk_real
            )
            token_selected_experts = token_selected_experts.to(torch.int32)
            RouteCapture.capture(moe.layer_idx, token_selected_experts)  # R3 device-buffer capture
            token_final_scales = token_final_scales.to(torch.float32)
        else:
            device = x.device
            token_selected_experts = torch.empty(
                (0, moe.routing_method.experts_per_token),
                dtype=torch.int32,
                device=device,
            )
            token_final_scales = torch.empty(
                (0, moe.routing_method.experts_per_token),
                dtype=torch.float32,
                device=device,
            )

        # ----- EPLB: update stats + remap expert ids -> slot ids -----
        if moe.layer_load_balancer:
            moe._load_balancer_done_wait_gpu_stage(is_first_call)
            # ignore_allreduce=False: the fused kernel has no side channel
            # for an external stats gather. The base helper runs its own
            # EP-wide AllReduce, gated to is_last_call=True.
            moe._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=False,
            )
            token_selected_slots = moe._load_balancer_route(token_selected_experts, moe.use_dp)
        else:
            token_selected_slots = token_selected_experts

        if token_selected_slots is not None:
            ExpertStatistic.set_layer(moe.layer_idx)
            ExpertStatistic.maybe_add_info(moe.num_slots, token_selected_slots)
        token_selected_slots = get_calibrator().maybe_collect_or_replay_slots(
            moe.num_slots, token_selected_slots
        )

        # ----- quantize / prepare -----
        if getattr(moe.backend, "supports_fused_prepare", lambda: False)():
            # MegaMoE can fuse BF16->MXFP8 quantization with the SymmBuffer
            # topk copies, so keep the original activations and let run_moe
            # prepare its workspace.
            moe_input = x_chunk_real
            x_sf = None
        else:
            # Delegate to ``backend.quantize_input`` so each fused-comm backend
            # owns its own empty-tensor layout. Both MegaMoEDeepGemm and
            # MegaMoECuteDsl short-circuit ``x.shape[0] == 0`` inside their
            # quantize_input contracts.
            moe_input, x_sf = moe.backend.quantize_input(x_chunk_real)

        # CuteDSL needs the scheduler's rank-identical chunk maximum to select
        # one adaptive bucket on every EP rank; using a local token count could
        # diverge and deadlock its in-kernel NVLink barrier.
        set_adaptive = getattr(moe.backend, "set_adaptive_launch_tokens", None)
        if set_adaptive is not None:
            set_adaptive(max(all_rank_num_tokens) if all_rank_num_tokens else None)

        # ----- MoE compute -----
        # ``token_selected_slots`` is in [0, num_slots), matching the kernel's
        # ``num_experts`` template parameter (SymmBuffer / weights sized to
        # num_slots in quantization.py).
        # Fused-comm backends own the EP exchange, so there is no comm plan:
        # nothing outside the fused kernel decided anything about this forward.
        out = moe.backend.run_moe(
            MoERunContext(
                token_selected_experts=token_selected_slots,
                token_final_scales=token_final_scales,
                x=moe_input,
                x_sf=x_sf,
                output_dtype=output_dtype,
            )
        )

        # ----- EPLB: start/done CPU rebalance, AFTER run_moe -----
        # The external-comm path overlaps CPU stage with ``comm.combine``;
        # fused-comm has no external combine, so start_set fires
        # immediately after the fused kernel and done_set drains it. Placing
        # start_set before run_moe would let dynamic-EPLB migration race the
        # kernel.
        moe._load_balancer_start_set_cpu_stage(is_last_call)
        moe._load_balancer_done_set_cpu_stage(is_last_call)

        return out


# ============================================================================
# Factory
# ============================================================================


def create_moe_scheduler(moe: "ConfigurableMoE") -> MoEScheduler:
    """Pick the scheduler matching ``moe.backend.scheduler_kind``."""
    kind = moe.backend.scheduler_kind
    if kind == MoESchedulerKind.FUSED_COMM:
        return FusedCommMoEScheduler(moe)
    if kind == MoESchedulerKind.EXTERNAL_COMM:
        return ExternalCommMoEScheduler(moe)
    raise ValueError(
        f"Unknown MoE scheduler kind {kind!r} on backend "
        f"{type(moe.backend).__name__}. Set ``scheduler_kind`` to one of "
        f"{[k.name for k in MoESchedulerKind]}."
    )
