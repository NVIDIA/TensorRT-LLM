# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as shared
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import (
    _MEGAMOE_PRIME_LADDER,
    MegaMoECuteDsl,
    MegaMoeCuteDslUnavailable,
    _as_fp8_sf,
    _as_nvfp4,
)
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import FusedCommMoEScheduler
from tensorrt_llm._torch.route_capture import get_active_route_capture
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.logger import logger
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import Mode

from ... import native_ops
from ...kernels.moe.constants import DEFERRED_TOPK_REDUCE_BUCKETS, GEN_TOKEN_BUCKETS
from .megamoe_runner import (
    get_rubin_runner,
    query_rubin_kernel_properties,
    run_rubin_megamoe,
    run_rubin_megamoe_op,
    runner_kernel,
)

_DIRECT_STAGE_MAX_TOKENS = 64
_DIRECT_STAGE_TOPK_SLOTS: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class DirectStage:
    launch_max_T: int
    num_tokens: int
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    activation: torch.Tensor
    activation_sf: torch.Tensor
    quantized: bool = False


class RubinMegaMoE(MegaMoECuteDsl):
    def __init__(self, **kwargs):
        self._direct_stage_enabled = True
        self._wide_buckets = (128, 192)
        self._pending_direct_stage = None
        self._deferred_topk_tokens = None
        self._launch_defer = False
        super().__init__(**kwargs)
        # Match the reference GLM path: fold routing weights into the gated FC1
        # output before its FP4 quantization. Deferred planes are already weighted.
        self.apply_topk_in_fc1 = True

    def _alloc_symm_provider(self):
        """Build-time symmetric provider allocation. See ``create_weights``.

        One provider per adaptive bucket. Each ``get_megamoe_symm_provider``
        call is a COLLECTIVE rendezvous, so the loop must iterate identical
        buckets in identical order on every EP rank (guaranteed: the ladder is
        a pure function of rank-identical inputs); the module-scope cache
        makes only the first layer pay. Raises MegaMoeCuteDslUnavailable when
        no ProcessGroup is available (hard error for multi-rank).
        """
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            get_megamoe_symm_provider,
        )

        if self._ep_pg is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank requires a torch.distributed EP "
                "ProcessGroup. Use Ray / DeviceMesh (mapping.moe_ep_group_pg) "
                "or initialize torch.distributed before model build."
            )
        top_k = self.routing_method.experts_per_token
        full_T = self._maxt_buckets[-1]
        self._profiling_scratch_provider_ref = None
        self._launch_buffers.clear()
        for max_T in self._maxt_buckets:
            (
                shared_workspace_bytes,
                supports_direct_inputs,
            ) = query_rubin_kernel_properties(
                world_size=self.ep_size,
                num_topk=top_k,
                num_experts_per_rank=self.compute_slot_count,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                max_tokens_per_rank=max_T,
                apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
                gate_up_clamp=self.act_clamp,
                swiglu_alpha=(
                    self.act_alpha if self.activation_type is ActivationType.SwigluBias else None
                ),
                swiglu_beta=(
                    self.act_beta if self.activation_type is ActivationType.SwigluBias else None
                ),
                situ_beta=(self.act_alpha if self.activation_type is ActivationType.SiTu else None),
                situ_linear_beta=(
                    self.act_beta if self.activation_type is ActivationType.SiTu else None
                ),
                in_kernel_fc2_reduce=self._use_in_kernel_fc2_reduce(max_T),
                combine_format=self.combine_format,
            )
            if supports_direct_inputs:
                self._genphase_direct_input_buckets.add(max_T)
            provider = get_megamoe_symm_provider(
                process_group=self._ep_pg,
                world_size=self.ep_size,
                rank=self.ep_rank,
                hidden_size=self.hidden_size,
                max_tokens_per_rank=max_T,
                num_topk=top_k,
                output_dtype=self.dtype or torch.bfloat16,
                combine_format=self.combine_format,
                shared_workspace_bytes=shared_workspace_bytes,
            )
            self._symm_providers[max_T] = provider
            # AutoTuner profiling scratch: record only the request kwargs
            # (full bucket only, matching tuning mode's forced top bucket);
            # the multi-GiB allocation itself is deferred to the op's
            # profiling pre-hook via the factory in ``_launch_megamoe_kernel``,
            # so a process that never tunes never pays it.
            if max_T == full_T and self.tactic_autotune:
                self._profiling_scratch_kwargs = dict(
                    process_group=self._ep_pg,
                    world_size=self.ep_size,
                    rank=self.ep_rank,
                    hidden_size=self.hidden_size,
                    max_tokens_per_rank=max_T,
                    num_topk=top_k,
                    output_dtype=self.dtype or torch.bfloat16,
                    combine_format=self.combine_format,
                    shared_workspace_bytes=shared_workspace_bytes,
                )
        if len(self._maxt_buckets) > 1 and self.ep_rank == 0:
            logger.info(
                "[MegaMoECuteDsl] adaptive max_tokens_per_rank buckets=%s "
                "(one symmetric provider + kernel per bucket; per-launch bucket "
                "= smallest >= max(all_rank_num_tokens))",
                self._maxt_buckets,
            )

    def prepare_direct_stage(
        self,
        x: torch.Tensor,
        *,
        num_tokens: int,
        chunk_max_tokens: Optional[int],
    ) -> Optional[DirectStage]:
        """Return the kernel input buffers of this launch, or ``None`` to stage
        through copies (``_stage_inputs``).

        Used by the target-local attention-DP decode for a
        multi-rank launch of 1..bucket BF16 rows into a generation-phase bucket
        of at most ``_DIRECT_STAGE_MAX_TOKENS`` rows, or of the wide
        attention-DP decode's ``_wide_buckets``. ``chunk_max_tokens`` is the lockstep
        ``max(all_rank_num_tokens)`` that ``set_adaptive_launch_tokens`` will
        receive, so the bucket is the one ``_run_moe`` launches. The caller
        writes the routing into ``topk_idx[:num_tokens]`` /
        ``topk_weights[:num_tokens]`` and quantizes ``x`` with
        ``quantize_input_into``.
        """
        if not (self._direct_stage_enabled and self.ep_size > 1) or self.tactic_autotune:
            return None
        if self._symm_provider is None or self._topk_idx_dtype != torch.int32:
            return None
        hidden = self.hidden_size
        if not (
            x.dim() == 2
            and 0 < num_tokens == x.shape[0]
            and x.shape[1] == hidden
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        ):
            return None
        # The quantization writes linear scale-factor rows of hidden // 16
        # bytes; they must have the stride of the kernel's SF rows.
        if megamoe_activation_sf_bytes_per_row(hidden) != hidden // 16:
            return None
        launch_max_T = self._select_launch_max_tokens(chunk_max_tokens)
        if (
            num_tokens > launch_max_T
            or (launch_max_T > _DIRECT_STAGE_MAX_TOKENS and launch_max_T not in self._wide_buckets)
            or launch_max_T not in self._genphase_direct_input_buckets
        ):
            return None
        top_k = self.routing_method.experts_per_token
        slots = self._direct_stage_topk_slots(launch_max_T, top_k, x.device)
        if slots is None:
            return None
        topk_idx_slots, topk_weights_slots = slots
        regions = self._symm_providers[launch_max_T].get_regions()
        return DirectStage(
            launch_max_T=launch_max_T,
            num_tokens=num_tokens,
            topk_idx=topk_idx_slots[num_tokens],
            topk_weights=topk_weights_slots[num_tokens],
            activation=regions.activation,
            activation_sf=regions.activation_sf,
        )

    @staticmethod
    def _direct_stage_topk_slots(
        launch_max_T: int, top_k: int, device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The bucket's per-row-count top-k slots (``_DIRECT_STAGE_TOPK_SLOTS``).

        Created on first use outside CUDA-graph capture (the eager warmup that
        precedes every capture); during a capture that finds none, returns
        ``None`` so the launch stages through copies rather than recording the
        fills into the graph.
        """
        key = (str(device), int(launch_max_T), int(top_k), torch.is_inference_mode_enabled())
        slots = _DIRECT_STAGE_TOPK_SLOTS.get(key)
        if slots is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            shape = (launch_max_T + 1, launch_max_T, top_k)
            slots = (
                torch.full(shape, -1, dtype=torch.int32, device=device),
                torch.zeros(shape, dtype=torch.float32, device=device),
            )
            _DIRECT_STAGE_TOPK_SLOTS[key] = slots
        return slots

    def route_and_quantize_into(self, x, router_logits, routing_method, stage, input_ids=None):
        del input_ids
        impl = getattr(routing_method, "routing_impl", None)
        get_bias = getattr(routing_method, "callable_e_score_correction_bias", None)
        if impl is None or get_bias is None or getattr(self, "fc31_act_scale", None) is not None:
            return None
        bias = get_bias()
        if not (
            impl.is_fused
            and impl.n_group == impl.topk_group == 1
            and 0 < impl.top_k <= 32
            and router_logits.shape == (stage.num_tokens, 256)
            and router_logits.dtype == bias.dtype == torch.float32
            and router_logits.is_contiguous()
            and bias.is_contiguous()
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        ):
            return None
        n = stage.num_tokens
        native_ops.glm_rubin_noaux_tc_fp4_quantize_out(
            router_logits,
            bias,
            x,
            self.fc31_input_scale,
            1,
            1,
            impl.top_k,
            impl.routed_scaling_factor,
            stage.topk_weights[:n],
            stage.topk_idx[:n],
            stage.activation[:n],
            stage.activation_sf[:n],
        )
        return replace(stage, quantized=True)

    def quantize_input_into(self, x, stage):
        n = stage.num_tokens
        if not stage.quantized:
            activation, sf = super().quantize_input(x)
            stage.activation[:n].copy_(activation.view(torch.uint8))
            stage.activation_sf[:n].copy_(sf.view(torch.uint8))
        self._pending_direct_stage = stage
        return stage.activation[:n], stage.activation_sf[:n]

    def defer_next_topk_reduce(self, num_tokens):
        """Return the next matching single-chunk result as a weighted BF16 top-k plane."""
        self._deferred_topk_tokens = int(num_tokens)

    def _run_moe(self, **kwargs):
        stage = self._pending_direct_stage
        self._pending_direct_stage = None
        self._launch_defer = self._deferred_topk_tokens == kwargs["num_tokens"]
        self._deferred_topk_tokens = None
        try:
            if stage is None:
                return super()._run_moe(**kwargs)
            launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
            self._active_launch_max_tokens = None
            if stage.launch_max_T != launch_max_T or stage.num_tokens != kwargs["num_tokens"]:
                raise RuntimeError(
                    "GLM MegaMoE direct stage does not match the lockstep launch bucket"
                )
            if (
                self.ep_size > 1
                and not self.tactic_autotune
                and _MEGAMOE_PRIME_LADDER
                and AutoTuner.get().is_tuning_mode
            ):
                self._prime_ladder(
                    live_T=launch_max_T,
                    weight_view=kwargs["weight_view"],
                    top_k=kwargs["top_k"],
                    hidden=kwargs["hidden"],
                    device=kwargs["device"],
                    output_dtype=kwargs["output_dtype"],
                )
            bufs = self._acquire_buffers(
                top_k=kwargs["top_k"],
                hidden=kwargs["hidden"],
                device=kwargs["device"],
                output_dtype=kwargs["output_dtype"],
                launch_max_T=launch_max_T,
            )
            return self._launch_megamoe_kernel(
                activation=stage.activation,
                activation_sf=stage.activation_sf,
                topk_idx=stage.topk_idx,
                topk_weights=stage.topk_weights,
                weight_view=kwargs["weight_view"],
                combine_output=bufs.combine_output,
                shared_workspace=bufs.shared_workspace,
                world_size=self.ep_size,
                local_rank=self.ep_rank,
                top_k=kwargs["top_k"],
                hidden=kwargs["hidden"],
                peer_offsets=bufs.peer_offsets,
                num_tokens=kwargs["num_tokens"],
                output_dtype=kwargs["output_dtype"],
                launch_max_T=launch_max_T,
                staged_inputs=True,
            )
        finally:
            self._launch_defer = False

    def _launch_megamoe_kernel(self, **kw):
        bucket = kw["launch_max_T"]
        if bucket not in GEN_TOKEN_BUCKETS:
            return super()._launch_megamoe_kernel(**kw)
        if self.tactic_autotune or kw.get("replica_plan") is not None:
            raise ValueError("GLM Rubin MegaMoE uses fixed generation tactics without dynamic EPLB")
        n = kw["num_tokens"]
        in_kernel = self._use_in_kernel_fc2_reduce(bucket)
        deferred = bool(
            self._launch_defer
            and n > 0
            and not in_kernel
            and bucket in DEFERRED_TOPK_REDUCE_BUCKETS
        )
        runner_args = dict(
            world_size=self.ep_size,
            local_rank=self.ep_rank,
            num_topk=kw["top_k"],
            num_experts_per_rank=self.compute_slot_count,
            hidden_size=kw["hidden"],
            intermediate_size_per_partition=int(self.intermediate_size_per_partition),
            expand_intermediate_size_per_partition=int(self.expand_intermediate_size_per_partition),
            max_tokens_per_rank=bucket,
            apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
            gate_up_clamp=self.act_clamp,
            in_kernel_fc2_reduce=in_kernel,
            combine_format=self.combine_format,
            defer_topk_reduce=deferred,
        )
        runner = get_rubin_runner(**runner_args)
        output = kw["combine_output"]
        if in_kernel and n > 0:
            output[:n].zero_()
        if n > 0:
            output = output[:n]
        weights = kw["weight_view"]
        inputs = [
            _as_nvfp4(kw["activation"]),
            _as_fp8_sf(kw["activation_sf"]),
            kw["topk_idx"],
            kw["topk_weights"],
            _as_nvfp4(weights.fc1_weight),
            _as_fp8_sf(weights.fc1_weight_sf),
            _as_nvfp4(weights.fc2_weight),
            _as_fp8_sf(weights.fc2_weight_sf),
            weights.fc31_alpha,
            weights.fc2_alpha,
            weights.fc1_norm_const,
            output,
        ]
        shared.set_active_megamoe_cute_weight_cache(self._cute_weight_cache)
        provider = self._symm_providers.get(bucket) if kw.get("staged_inputs") else None
        shared.set_active_megamoe_cute_buffer_cache(
            provider.cute_views if provider is not None else None
        )
        try:
            launch = run_rubin_megamoe_op if torch.compiler.is_compiling() else run_rubin_megamoe
            launch(
                *inputs,
                shared_workspace=kw["shared_workspace"],
                peer_offsets=list(kw["peer_offsets"]),
                **runner_args,
            )
        finally:
            shared.set_active_megamoe_cute_weight_cache(None)
            shared.set_active_megamoe_cute_buffer_cache(None)
        if n == 0:
            return output.new_empty((0, kw["hidden"]), dtype=kw["output_dtype"])
        if deferred:
            offset, shape = runner_kernel(runner).pre_reduced_activation_plane()
            nbytes = shape[0] * shape[1] * shape[2] * 2
            workspace = kw["shared_workspace"]
            if offset + nbytes > workspace.numel():
                raise RuntimeError("GLM MegaMoE deferred plane exceeds its symmetric workspace")
            return workspace[offset : offset + nbytes].view(torch.bfloat16).view(shape)[:n]
        return output.squeeze(1).to(kw["output_dtype"])


class RubinFusedCommScheduler(FusedCommMoEScheduler):
    def _forward_chunk(
        self,
        x,
        router_logits,
        *,
        output_dtype,
        all_rank_num_tokens,
        do_finalize,
        is_first_call=True,
        is_last_call=True,
        input_ids=None,
        row_offset=0,
        routing_aux=None,
    ):
        moe = self.moe
        n = int(all_rank_num_tokens[moe.mapping.moe_ep_rank]) if all_rank_num_tokens else x.shape[0]
        stage = None
        if (
            n > 0
            and moe.layer_load_balancer is None
            and routing_aux is None
            and get_active_route_capture() is None
            and get_calibrator().mode is Mode.NONE
            and not moe._enable_perfect_router_expert_ids
        ):
            stage = moe.backend.prepare_direct_stage(
                x[:n],
                num_tokens=n,
                chunk_max_tokens=max(all_rank_num_tokens) if all_rank_num_tokens else None,
            )
        if stage is None:
            return super()._forward_chunk(
                x,
                router_logits,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                do_finalize=do_finalize,
                is_first_call=is_first_call,
                is_last_call=is_last_call,
                input_ids=input_ids,
                row_offset=row_offset,
                routing_aux=routing_aux,
            )
        if moe.apply_router_weight_on_input:
            raise ValueError(
                "GLM fused-communication MoE does not support routing weights on the input"
            )
        if not do_finalize:
            raise ValueError("GLM fused-communication MoE requires finalization")
        fused = moe.backend.route_and_quantize_into(
            x[:n], router_logits[:n], moe.routing_method, stage, input_ids
        )
        if fused is None:
            indices, scales = moe.routing_method.apply(
                router_logits[:n], input_ids[:n] if input_ids is not None else None
            )
            stage.topk_idx[:n].copy_(indices)
            stage.topk_weights[:n].copy_(scales)
        else:
            stage = fused
        quantized, sf = moe.backend.quantize_input_into(x[:n], stage)
        moe.backend.set_adaptive_launch_tokens(
            max(all_rank_num_tokens) if all_rank_num_tokens else None
        )
        return moe.backend.run_moe(
            MoERunContext(
                x=quantized,
                x_sf=sf,
                token_selected_experts=stage.topk_idx[:n],
                token_final_scales=stage.topk_weights[:n],
                output_dtype=output_dtype or x.dtype,
            )
        )


def create_rubin_moe(**kwargs):
    quant = kwargs.get("override_quant_config") or kwargs["model_config"].quant_config
    if (
        kwargs["model_config"].moe_backend != "MEGAMOE_CUTEDSL"
        or quant is None
        or not quant.layer_quant_mode.has_nvfp4()
    ):
        return create_moe(**kwargs)
    kwargs.pop("allow_backend_degradation", None)
    moe = ConfigurableMoE(moe_cls=RubinMegaMoE, **kwargs)
    moe.scheduler = RubinFusedCommScheduler(moe)
    return moe
