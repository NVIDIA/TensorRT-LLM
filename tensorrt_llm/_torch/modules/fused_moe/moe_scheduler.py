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

from tensorrt_llm._torch.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.utils import EventType, Fp4QuantizedTensor
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from .communication import DeepEP, DeepEPLowLatency, NVLinkOneSided, NVLinkTwoSided
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepGemmFusedMoE
from .fused_moe_densegemm import DenseGEMMFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .interface import MoESchedulerKind

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
        return isinstance(self.moe.comm, NVLinkTwoSided)

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
        """Single-chunk workspace for DeepGemmFusedMoE; otherwise ``None``.

        Multi-chunk execution uses ``_prepare_workspaces_for_chunk`` instead.
        """
        moe = self.moe
        if not isinstance(moe.backend, DeepGemmFusedMoE):
            return None

        num_rows = x.shape[0]
        if moe.use_dp and moe.comm is not None:
            # Communication path padding: dispatch outputs are
            # ``[ep_size * max_tokens_per_rank, ...]`` (or expert-major for
            # DeepEPLowLatency). Workspace must cover that footprint.
            if isinstance(moe.comm, DeepEPLowLatency):
                num_rows = moe.num_slots * max(all_rank_num_tokens)
            else:
                num_rows = moe.mapping.moe_ep_size * max(all_rank_num_tokens)

        workspaces = moe.backend.get_workspaces([num_rows])
        return workspaces[0]

    def _prepare_workspaces_for_chunk(
        self,
        all_rank_num_tokens_list: List[Optional[List[int]]],
        chunk_size_list: List[int],
        use_multi_stream: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Multi-chunk workspaces for DeepGemmFusedMoE; ``(None, None)`` otherwise.

        Single-chunk execution uses ``_prepare_workspace_deepgemm`` instead.
        """
        moe = self.moe
        workspace_0 = None
        workspace_1 = None

        if not isinstance(moe.backend, DeepGemmFusedMoE):
            return workspace_0, workspace_1

        # Always need at least workspace_0; reuse chunk_0 size for workspace_1
        # since chunk 0 is always >= subsequent chunks under split_chunk.
        # Mirror ``_prepare_workspace_deepgemm``: DeepEPLowLatency dispatches
        # expert-major outputs sized ``num_slots * max_tokens_per_rank`` per
        # rank (one shard per slot), while other comms produce
        # ``ep_size * max_tokens_per_rank``. Using the wrong formula
        # under-allocates the workspace for DeepEPLowLatency multi-chunk
        # runs and is caught by ``DeepGemmFusedMoE.run_moe``.
        if moe.use_dp and all_rank_num_tokens_list[0] is not None:
            max_tokens = max(all_rank_num_tokens_list[0])
            if isinstance(moe.comm, DeepEPLowLatency):
                chunk_size_0 = moe.num_slots * max_tokens
            else:
                chunk_size_0 = moe.mapping.moe_ep_size * max_tokens
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

        # ========== Step 2: Apply routing (only if backend supports load balancer) ==========
        if moe.backend._supports_load_balancer():
            # Separated routing: ConfigurableMoE calls routing_method
            token_selected_experts, token_final_scales = moe.routing_method.apply(router_logits)

            token_selected_experts = token_selected_experts.to(torch.int32)

            assert token_selected_experts.shape[1] == moe.routing_method.experts_per_token
            assert token_selected_experts.shape == token_final_scales.shape
            # CutlassFusedMoE and DenseGEMMFusedMoE expect float32; TRTLLMGen expects bfloat16
            if isinstance(moe.backend, (CutlassFusedMoE, DenseGEMMFusedMoE)):
                assert token_final_scales.dtype == torch.float32
            assert token_selected_experts.dtype == torch.int32

            if token_final_scales is not None and isinstance(moe.backend, TRTLLMGenFusedMoE):
                token_final_scales = token_final_scales.to(torch.bfloat16)

            # apply_router_weight_on_input: fuse top-k weight onto x
            if moe.apply_router_weight_on_input:
                assert x.dtype != torch.float8_e4m3fn, (
                    "Current workaround for apply_router_weight_on_input does not support fp8 input"
                )
                x = x * token_final_scales.to(x.dtype)
                # DeepEP variants need a non-None token_final_scales tensor
                # (they don't tolerate None), so feed all-ones; other strategies
                # accept None and skip the multiply.
                if isinstance(moe.comm, (DeepEP, DeepEPLowLatency)):
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
            supports_post_quant = moe.comm.supports_post_quant_dispatch()

            # Debug: optional dummy AllReduce to break load-balancing artifacts
            if moe.enable_dummy_allreduce:
                moe.dummy_allreduce()

            dispatch_kwargs = dict(eplb_dispatch_kwargs)
            if isinstance(moe.comm, DeepEP) and isinstance(moe.backend, TRTLLMGenFusedMoE):
                dispatch_kwargs["enable_sanitize_expert_ids"] = True

            if supports_post_quant:
                # Quantize -> Dispatch
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
            x, x_sf = moe.backend.quantize_input(x, post_quant_comm=False)

        # ========== Step 6: MoE computation ==========
        # If EPLB is enabled, token_selected_slots is slot ids; otherwise expert ids.
        final_hidden_states = moe.backend.run_moe(
            x=x,
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            **self._get_backend_kwargs(
                router_logits, do_finalize, all_rank_num_tokens, output_dtype, x, workspace
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
        chunked_used = torch.ones(num_chunks, dtype=torch.bool)
        if moe.use_dp:
            # The split heuristic guarantees chunk 0 has >= 1 token, so it can
            # stand in for any empty chunk on this rank. Without substitution,
            # the per-chunk dispatch would launch with 0-token shape and the
            # peers would see a barrier mismatch.
            assert x_list[0].numel() != 0, "chunk 0 shouldn't be empty"
            x_list = list(x_list)
            router_logits_list = list(router_logits_list)
            for idx_chunk in range(num_chunks):
                _x = x_list[idx_chunk]
                if _x.numel() == 0:
                    chunked_used[idx_chunk] = False
                    x_list[idx_chunk] = x_list[0]
                    router_logits_list[idx_chunk] = router_logits_list[0]
                    all_rank_num_tokens_list[idx_chunk][moe.mapping.tp_rank] = (
                        all_rank_num_tokens_list[0][moe.mapping.tp_rank]
                    )
            x_list = tuple(x_list)
            router_logits_list = tuple(router_logits_list)

        # ========== Execute chunking with overlap ==========
        outputs_list = []
        for idx_chunk, (x_chunk, router_logits_chunk) in enumerate(zip(x_list, router_logits_list)):
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
    # Backend run_moe kwargs builder (external-comm only)
    # ------------------------------------------------------------------
    def _get_nvlink_onesided_moe_output(
        self,
        all_rank_num_tokens: Optional[List[int]],
        output_dtype: Optional[torch.dtype],
    ) -> Optional[torch.Tensor]:
        """Workspace-backed output buffer for NVLinkOneSided combine, or None.

        Only meaningful when ``moe.comm`` is NVLinkOneSided AND the backend
        supports payload-in-workspace combine. Returns None for all other
        comm strategies; callers should always set the resulting kwarg
        unconditionally and let backends ignore None.
        """
        moe = self.moe
        if not isinstance(moe.comm, NVLinkOneSided):
            return None

        if not moe.backend.supports_moe_output_in_alltoall_workspace():
            # Backend opts out: keep payload off the workspace path.
            moe.comm.payload_in_workspace = False
            return None

        workspace_dtype = output_dtype
        if isinstance(moe.backend, TRTLLMGenFusedMoE):
            # TRTLLMGen sentinel for unfilled rows; bf16 workspace is the
            # combine reduction precision used by the kernel.
            moe.comm.invalid_token_expert_id = -1
            workspace_dtype = torch.bfloat16

        assert all_rank_num_tokens is not None, (
            "all_rank_num_tokens must be provided for NVLinkOneSided backend"
        )
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)

        moe_output = moe.comm.get_combine_payload_tensor_in_workspace(
            runtime_max_tokens_per_rank, moe.hidden_size, workspace_dtype
        )

        # Toggle on for this forward; combine() reads this flag to decide
        # whether to emit into the workspace tensor.
        moe.comm.payload_in_workspace = True
        return moe_output

    def _get_backend_kwargs(
        self,
        router_logits: Optional[torch.Tensor] = None,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        output_dtype: Optional[torch.dtype] = None,
        x: Optional[torch.Tensor] = None,
        workspace: Optional[dict] = None,
    ) -> Dict:
        """Backend-specific kwargs for ``backend.run_moe`` (external-comm only).

        ``FusedCommMoEScheduler`` constructs its own kwargs and never
        calls this helper, so all branches here are EXTERNAL_COMM backends.

        Backend-specific kwargs:
            - Cutlass: is_sf_swizzled, enable_alltoall, tuner_*, moe_output
            - CuteDSL: enable_alltoall, moe_output, dwdp_weight_view
            - DeepGemm: workspace
            - TRTLLMGen: router_logits, do_finalize, moe_output
        """
        moe = self.moe
        kwargs: Dict = {}

        if moe.backend.__class__ == CutlassFusedMoE:
            # Pre-quant dispatch: SFs arrive swizzled; post-quant dispatch:
            # SFs arrive unswizzled. Backend uses this to skip a re-swizzle.
            supports_post_quant = moe.comm is not None and moe.comm.supports_post_quant_dispatch()
            kwargs["is_sf_swizzled"] = not supports_post_quant
            kwargs["output_dtype"] = output_dtype

            # Tuner sees pre-alltoall token shapes so cached tactics from the
            # warmup (no-alltoall) phase still apply at runtime.
            kwargs["enable_alltoall"] = moe.enable_alltoall
            if moe.enable_alltoall:
                if all_rank_num_tokens is not None:
                    kwargs["tuner_num_tokens"] = sum(all_rank_num_tokens)
                else:
                    kwargs["tuner_num_tokens"] = (
                        x.shape[0] * moe.mapping.tp_size if x is not None else None
                    )
                kwargs["tuner_top_k"] = moe.routing_method.top_k

            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

        elif moe.backend.__class__ == CuteDslFusedMoE:
            kwargs["enable_alltoall"] = moe.enable_alltoall
            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

            if moe.enable_dwdp:
                kwargs["dwdp_weight_view"] = moe.dwdp_manager.build_weight_view(
                    moe.layer_idx, moe.backend
                )

        elif moe.backend.__class__ == DeepGemmFusedMoE:
            if workspace is not None:
                kwargs["workspace"] = workspace

        elif moe.backend.__class__ == TRTLLMGenFusedMoE:
            # When the scheduler precomputes top-k for DP/load-balancer paths,
            # the backend must not route again.  Single-rank TRTLLMGen paths do
            # not get precomputed top-k, so they still need router_logits.
            router_logits_arg = None if moe.backend._supports_load_balancer() else router_logits
            kwargs["router_logits"] = router_logits_arg
            kwargs["do_finalize"] = do_finalize
            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

        return kwargs


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
    ) -> torch.Tensor:
        """Sequential multi-chunk path for MegaMoE-style backends.

        Single-chunk case is just ``num_chunks == 1`` -- no separate fast
        path. Invariants enforced here (see class docstring): identical
        ``num_chunks`` per rank computed from ``max()``, ADP padding
        stripped before splitting, zero-token chunks still launch the
        kernel for the cross-rank barrier.
        """
        del use_dp_padding  # MegaMoE has no host-side cross-rank shape alignment.

        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "Fused-comm MoE expects BF16 activation; "
                "quantization happens in backend.quantize_input."
            )

        x_real, rl_real, real_all_rank_num_tokens, ep_rank, had_meta = self._strip_adp_padding(
            x, router_logits, all_rank_num_tokens
        )
        num_chunks, x_chunks, rl_chunks, all_rank_chunk_size_list = self._compute_chunk_layout(
            x_real, rl_real, real_all_rank_num_tokens, ep_rank
        )
        outputs = self._run_chunks(
            x_chunks,
            rl_chunks,
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
        return torch.cat(outputs, dim=0)

    def _strip_adp_padding(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], int, bool]:
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
        return x_real, rl_real, real_all_rank_num_tokens, ep_rank, had_meta

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
            else:
                # Shape ``(0, hidden_size)`` keeps dtype/device/column-width
                # intact so routing / quantize / run_moe execute as no-ops
                # without shape errors before reaching the barrier.
                x_chunk = x_real.new_empty((0, x_real.shape[1]))
                rl_chunk = rl_real.new_empty((0, rl_real.shape[1]))

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

        # ----- EPLB: drain previous CPU rebalance -----
        # Static EPLB early-returns inside the helper; only the dynamic
        # balancer actually waits.
        moe._load_balancer_start_wait_gpu_stage(is_first_call)

        # ----- routing -----
        # int32 matches the EPLB stats kernel contract used by the external-comm
        # path; the fused-comm backend casts to int64 internally.
        if num_tokens > 0:
            token_selected_experts, token_final_scales = moe.routing_method.apply(
                router_logits_chunk_real
            )
            token_selected_experts = token_selected_experts.to(torch.int32)
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

        # ----- quantize -----
        if num_tokens > 0:
            x_fp8, x_sf = moe.backend.quantize_input(x_chunk_real)
        else:
            device = x.device
            x_fp8 = torch.empty((0, moe.hidden_size), dtype=torch.float8_e4m3fn, device=device)
            # Packed-UE8M0 int32 SF: one int32 per 128 input elements per row,
            # same stride contract as the non-empty runs for run_moe.
            x_sf = torch.empty((0, moe.hidden_size // 128), dtype=torch.int32, device=device)

        # ----- MoE compute -----
        # ``token_selected_slots`` is in [0, num_slots), matching the kernel's
        # ``num_experts`` template parameter (SymmBuffer / weights sized to
        # num_slots in quantization.py).
        out = moe.backend.run_moe(
            x=x_fp8,
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            output_dtype=output_dtype,
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
