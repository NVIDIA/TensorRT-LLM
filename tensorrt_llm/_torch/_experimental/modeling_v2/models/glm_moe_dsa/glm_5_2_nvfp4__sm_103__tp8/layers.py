# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decoder layers, the draft layer and the MoE modules of the glm_moe_dsa sm_103 tp8 target.

``adopt`` re-classes a built-in module in place: a ``DeepseekV3DecoderLayer`` as
``GlmDecoderLayer``, the ``DeepseekV3MTP`` draft layer as ``GlmMtpLayer`` and a
``Deepseekv3MoE`` as ``GlmMoE``. It then attaches the fast-path objects the installed plan
enables, in the order the built-in constructor builds the modules they wrap: the attention
modules and the attention tail (``attention.GlmMLA``), the post-attention hop and its
transports, the router GEMM, the post-MoE hops, the attention tail's successor weights and, on
the draft layer, the draft front and the live-row hop. The
classes add methods and class-level defaults only (no parameter, buffer or submodule), so
the module tree, the weight names and the loaded weights stay the built-in model's.

Each forward override runs the target's decode composition when the step is a decode step
(``dispatch.Phase.DECODE``) and the built-in method otherwise, so a prefill or mixed step and
a call outside a step run the built-in model.

Every target of the family runs this composition; the targets differ in the plan they install
(``dispatch.TargetPaths``). A fast path's block runs only when the plan enables that path
(``dispatch.is_enabled``) or when an object only that path attaches is set, so the block of a
path a target does not serve never runs for that target. The routed-MoE paths, for instance,
take tensor-parallel experts: a target whose experts are expert-parallel does not serve them,
and its MoE runs the built-in experts on the router logits the other paths produce.

``adopt_model`` adopts a model's layers and its speculative worker; ``adopted_state`` lists
what the adoption attached, for the rank-uniform check.
"""

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.attention import maybe_slice_for_helix_cp
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, MoEAllReduceParams
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Gate,
    Deepseekv3MoE,
    DeepseekV3MTP,
    GlmMoeDsaForCausalLM,
    moe_reduce_add_shared_output,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import EventType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal.thop import BufferKind

from . import dispatch
from .attention import GlmMLA, reclass
from .spec_worker import GlmMtpWorker


class GlmMoE(Deepseekv3MoE):
    """The MoE of a decoder layer or of the draft layer: the router-logit producers and the
    routed fast paths in front of the built-in experts."""

    # The fast-path objects ``adopt`` and the layer's adoption attach; None keeps the built-in
    # computation.
    router_gemm = None
    _routed_into_hop = None

    @classmethod
    def adopt(cls, moe: Deepseekv3MoE, model_config, layer_idx: int) -> "GlmMoE":
        """Re-class a built-in MoE and attach its router GEMM.

        ``model_config`` and ``layer_idx`` are the ones the built-in constructor received.

        Raises:
            TypeError: if ``moe`` is not exactly the built-in ``Deepseekv3MoE``.
        """
        reclass(moe, Deepseekv3MoE, cls)
        # The router GEMM may replace this layer's gate GEMM; None keeps the native gate.
        if dispatch.is_enabled("router_gemm"):
            moe.router_gemm = dispatch.get_impl("router_gemm", "make_moe_router")(
                gate=moe.gate, layer_idx=layer_idx, model_config=model_config
            )
        return moe

    def compute_routed_output(
        self,
        hidden_states,
        hidden_states_fp4,
        all_rank_num_tokens,
        do_finalize,
        router_logits=None,
        hop_routing_out=None,
    ):
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().compute_routed_output(
                hidden_states=hidden_states,
                hidden_states_fp4=hidden_states_fp4,
                all_rank_num_tokens=all_rank_num_tokens,
                do_finalize=do_finalize,
            )
        # max-throughput
        use_dp_padding = False
        # Add DP padding on SM120 for context comm performance
        # TODO: Move this model-agonostic part to MoE
        if self.use_dp and self.mapping.tp_size > 1 and get_sm_version() == 120:
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0])
            )

        # Router logits come from at most one producer: the post-attention hop (the
        # argument), the router GEMM, or the native gate.
        if router_logits is None and self.router_gemm is not None and not use_dp_padding:
            # Returns None, before launching anything, outside its domain.
            router_logits = self.router_gemm(hidden_states)
        if router_logits is None:
            router_logits = self.gate(hidden_states)

        # The draft layer's BF16 routed experts (the FlashInfer TRT-LLM-Gen BF16 leaf) through
        # the draft routed-MoE kernels on the leaf's own buffers, bit-identical to the native
        # call. Its eligibility (leaf type, routing, shapes, dtypes, device, finalized output)
        # is decided before any device work; None keeps the path below (NVFP4 layers always
        # decline).
        if (
            hidden_states_fp4 is None
            and not self.use_dp
            and not use_dp_padding
            and dispatch.is_enabled("draft_routed_moe")
        ):
            routed_output = dispatch.get_impl("draft_routed_moe", "routed_output")(
                self.experts,
                self.gate.e_score_correction_bias,
                hidden_states,
                router_logits,
                do_finalize,
            )
            if routed_output is not None:
                return routed_output

        # The routed MoE: routing + activation quant + FC1 + FC2 + finalize in one CuTe DSL
        # launch on the native expert buffers. Its eligibility (shapes/dtypes/device/routing/
        # bias) is decided before any device work; None means "run the native path below".
        if (
            hidden_states_fp4 is None
            and not self.use_dp
            and not use_dp_padding
            and dispatch.is_enabled("routed_moe")
        ):
            # With the post-MoE hop as the consumer, the routed MoE writes its per-rank routed
            # result straight into the hop's routed-input buffer (in the hop's mode) and
            # returns [buffer]; None keeps the raw tuple below. The hop is set only when every
            # TP rank enabled it.
            routed_into_hop = getattr(self, "_routed_into_hop", None)
            if (
                routed_into_hop is not None
                and not do_finalize
                and dispatch.is_enabled("routed_moe_into_hop")
            ):
                # The hop-routing variant that also prefetches this layer's shared-expert
                # weights into L2 before its dependency wait, so the shared expert on the
                # auxiliary stream reads them from L2. None (decided before any device work)
                # keeps the hop-routing variant.
                if hop_routing_out is not None and dispatch.is_enabled("shared_expert_prefetch"):
                    routed_output = dispatch.get_impl(
                        "shared_expert_prefetch", "routed_output_into_hop"
                    )(
                        self.experts,
                        self.gate.e_score_correction_bias,
                        hidden_states,
                        router_logits,
                        hop_routing_out,
                        routed_into_hop,
                        self.shared_experts,
                    )
                    if routed_output is not None:
                        return routed_output
                # The routed MoE on the routing the pre-MoE hop computed (set only when the
                # hop-routing hop ran); None keeps the routed MoE, which routes from the same
                # logits to the same experts.
                if hop_routing_out is not None:
                    routed_output = dispatch.get_impl("hop_routing", "routed_output_into_hop")(
                        self.experts,
                        self.gate.e_score_correction_bias,
                        hidden_states,
                        router_logits,
                        hop_routing_out,
                        routed_into_hop,
                    )
                    if routed_output is not None:
                        return routed_output
                routed_output = dispatch.get_impl("routed_moe", "routed_output_into_hop")(
                    self.experts,
                    self.gate.e_score_correction_bias,
                    hidden_states,
                    router_logits,
                    routed_into_hop,
                )
                if routed_output is not None:
                    return routed_output
            if hop_routing_out is not None:
                routed_output = dispatch.get_impl("hop_routing", "routed_output")(
                    self.experts,
                    self.gate.e_score_correction_bias,
                    hidden_states,
                    router_logits,
                    hop_routing_out,
                    do_finalize,
                )
                if routed_output is not None:
                    return routed_output
            routed_output = dispatch.get_impl("routed_moe", "routed_output")(
                self.experts,
                self.gate.e_score_correction_bias,
                hidden_states,
                router_logits,
                do_finalize,
            )
            if routed_output is not None:
                return routed_output

        routed_output = self.experts(
            hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=hidden_states.dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Optional[Fp4QuantizedTensor] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
        router_logits: Optional[torch.Tensor] = None,
        hop_routing_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward(
                hidden_states=hidden_states,
                hidden_states_fp4=hidden_states_fp4,
                all_rank_num_tokens=all_rank_num_tokens,
                final_all_reduce_params=final_all_reduce_params,
                do_finalize=do_finalize,
            )
        if not do_finalize:
            assert not self.use_dp
        # Precomputed logits (and the hop routing) come only from the TP post-attention hop.
        assert router_logits is None or not self.use_dp
        assert hop_routing_out is None or router_logits is not None

        def _compute_shared_output():
            shared_input = (
                hidden_states_fp4
                if (hidden_states_fp4 is not None and self.shared_experts_use_fp4)
                else hidden_states
            )
            shared_output = self.shared_experts(shared_input)
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            routed_output = self.compute_routed_output(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens,
                do_finalize,
                router_logits,
                hop_routing_out,
            )
            return routed_output

        # NOTE: define compiled helpers at module scope to avoid defining decorators inside compiled frames

        if self.shared_experts is not None:
            routed_output, shared_output = maybe_execute_in_parallel(
                _compute_routed_output,
                _compute_shared_output,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                self.aux_stream,
                disable_on_compile=True,
            )
        else:
            # Shared experts have been fused into the routed experts (see post_load_weights);
            # routed_output already contains the shared expert contribution.
            shared_output = None
            routed_output = _compute_routed_output()

        if not do_finalize:
            return [shared_output, *routed_output]
        else:
            if shared_output is None:
                final_hidden_states = routed_output
            elif not isinstance(shared_output, torch.Tensor):
                final_hidden_states = shared_output + routed_output
                if not self.use_dp and self.mapping.tp_size > 1:
                    final_hidden_states = self.allreduce(
                        final_hidden_states, all_reduce_params=final_all_reduce_params
                    )
                return final_hidden_states
            else:
                output_tensor = None
                if not self.use_dp and self.mapping.tp_size > 1:
                    w, actual_kind = torch.ops.trtllm.allocate_output(
                        shared_output, self.allreduce.output_buffer_kind, self.mapping.tp_group
                    )
                    if actual_kind == int(BufferKind.NCCL_WINDOW):
                        output_tensor = w
                if routed_output.dim() == 3:
                    assert shared_output.numel() * self.top_k == routed_output.numel(), (
                        "unmatched tensor shape"
                    )
                    final_hidden_states = moe_reduce_add_shared_output(
                        routed_output, shared_output, out=output_tensor
                    )
                else:
                    assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
                    if output_tensor is not None:
                        final_hidden_states = torch.add(
                            shared_output, routed_output, out=output_tensor
                        )
                    else:
                        # In-place add to avoid allocating a temporary tensor, reducing peak memory
                        final_hidden_states = shared_output.add_(routed_output)

            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states, all_reduce_params=final_all_reduce_params
                )
            return final_hidden_states


class _PostAttentionHopMixin:
    """The post-attention hop of a decoder layer or of the draft layer: when it serves a call
    and how it is called."""

    def _post_attn_router_weight(self) -> Optional[torch.Tensor]:
        gate = getattr(self.mlp, "gate", None)
        return gate.weight if type(gate) is DeepseekV3Gate else None

    def _can_use_post_attn_hop(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        router_weight: Optional[torch.Tensor],
    ) -> bool:
        """Whether the post-attention hop serves this call.

        Decided from shapes/dtypes that are identical on all TP ranks, before
        any device work.
        """
        return (
            self.post_attn_hop is not None
            and self.post_attention_layernorm.nvfp4_scale is None
            and self.post_attn_hop.can_run(
                hidden_states, residual, self.post_attention_layernorm.weight, router_weight
            )
        )

    def _post_attn_hop(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        router_weight: Optional[torch.Tensor],
    ):
        return self.post_attn_hop(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            router_weight,
        )


class _AdoptedLayerMixin:
    """The fast-path state of a decoder layer or of the draft layer, and how it is attached.

    The adoption steps run in the order the built-in constructor builds the modules they
    wrap: the attention modules and the attention tail (the attention is built first), the
    post-attention hop and its transports (built before the MLP), the router GEMM (the MoE's),
    the post-MoE hops (after the MoE) and the attention tail's successor weights (last). They
    are methods of the layer: the deferred-norm hop finds the layer it serves through its
    caller's ``self`` (``kernels/deferred_norm/scope.py``).
    """

    # The fast-path objects the adoption attaches; None keeps the built-in computation.
    post_attn_hop = None
    _into_hop = None
    _hop_routing = None
    post_moe_hop = None
    deferred_norm_hop = None
    # Bound to the next MoE layer by the shell's ``setup_aliases``.
    _expert_prefetch_link = None

    def _adopt_fast_paths(self) -> None:
        """Attach this layer's fast-path objects, in the built-in constructor's order."""
        GlmMLA.adopt(self.self_attn)
        self._adopt_pre_moe()
        if isinstance(self.mlp, Deepseekv3MoE):
            GlmMoE.adopt(self.mlp, self.model_config, self.layer_idx)
        self._adopt_post_moe()
        self._adopt_tail_prefetch()

    def _adopt_pre_moe(self) -> None:
        """The post-attention hop, the into-hop transport and the hop routing."""
        config = self.model_config.pretrained_config
        # The post-attention hop of the pre-MoE fused boundary. It is shared per TP group and
        # created on every rank in layer order; it serves a layer only if that layer's
        # all-reduce uses MNNVL.
        if self.allreduce is not None and dispatch.is_enabled("post_attention_hop"):
            self.post_attn_hop = dispatch.get_impl("post_attention_hop", "make_post_attn_hop")(
                self.mapping,
                config.hidden_size,
                config.torch_dtype,
                native_allreduce=self.allreduce,
            )
        # The attention tail publishes its partial straight into the hop's transport; the hop
        # then reads it from there.
        if (
            self.post_attn_hop is not None
            and dispatch.is_enabled("attention_output_into_hop")
            and getattr(self.self_attn, "_attn_tail", None) is not None
        ):
            self._into_hop = dispatch.get_impl("post_attention_hop", "make_into_hop_transport")(
                self.post_attn_hop
            )
        # The into-hop also computes the native top-8 routing, which the routed MoE consumes
        # instead of routing again.
        if self._into_hop is not None and dispatch.is_enabled("hop_routing"):
            self._hop_routing = dispatch.get_impl("hop_routing", "make_transport")(self._into_hop)

    def _adopt_post_moe(self) -> None:
        """The post-MoE hop, its routed input and the deferred-norm hop (MoE layers only)."""
        config = self.model_config.pretrained_config
        layer_idx = self.layer_idx
        if not (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            return
        # Fused finalize + MNNVL all-reduce + RMSNorm on the raw NVFP4 MoE output.
        # Construction negotiates capability, including the raw-tuple contract the MoE module
        # declares, and allocates its workspace collectively; it stays None (native path)
        # unless every TP rank engages.
        if (
            dispatch.is_enabled("post_moe_hop")
            and self.fusion_config.POST_MOE_FUSION
            and self.is_nvfp4
            and self.model_config.moe_backend == "TRTLLM"
        ):
            hop = dispatch.get_impl("post_moe_hop", "post_moe_hop")(
                mapping=self.mapping,
                hidden_size=config.hidden_size,
                top_k=self.top_k,
                dtype=config.torch_dtype,
                raw_producer=self.mlp.experts,
            )
            self.post_moe_hop = hop if hop.engaged else None
            # The post-MoE hop consumes the routed MoE's result directly (collective enable).
            if (
                self.post_moe_hop is not None
                and dispatch.is_enabled("routed_moe_into_hop")
                and self.post_moe_hop.enable_routed_input()
            ):
                self.mlp._routed_into_hop = self.post_moe_hop

        # The hop defers the next layer's RMSNorm into that layer's projection. Only where the
        # next layer is a target attention layer; construction is collective and the hop
        # stays None unless every TP rank engages.
        if (
            self.post_moe_hop is not None
            and dispatch.is_enabled("deferred_norm")
            and dispatch.is_enabled("absorbed_query")
            and layer_idx + 1 < config.num_hidden_layers
        ):
            # The evict-hints variant is the same deferred hop with another chain schedule
            # (bit-identical packet).
            deferred_norm_hop = dispatch.get_impl(
                "deferred_norm_evict_hints"
                if dispatch.is_enabled("deferred_norm_evict_hints")
                else "deferred_norm",
                "deferred_hop",
            )(self.post_moe_hop)
            self.deferred_norm_hop = deferred_norm_hop if deferred_norm_hop.engaged else None
            # The deferred hop also serves the routed input the routed MoE writes into the
            # post-MoE hop (collective enable; it stays off on every rank unless all of them
            # engage).
            if (
                self.deferred_norm_hop is not None
                and dispatch.is_enabled("routed_moe_into_hop")
                and getattr(self.mlp, "_routed_into_hop", None) is not None
            ):
                self.deferred_norm_hop.enable_routed_input()

    def _adopt_tail_prefetch(self) -> None:
        """The router weight the attention tail stages into L2 (a cache hint)."""
        # The attention tail also stages the router weight, which the op after the tail reads
        # first, into L2 (no tensor changes).
        if (
            dispatch.is_enabled("attention_output_tail_prefetch")
            and isinstance(self.mlp, Deepseekv3MoE)
            and getattr(self.mlp, "gate", None) is not None
            and getattr(self.self_attn, "_attn_tail", None) is not None
        ):
            self.self_attn._attn_tail.set_successor_weights([self.mlp.gate.weight])


class GlmDecoderLayer(_AdoptedLayerMixin, _PostAttentionHopMixin, DeepseekV3DecoderLayer):
    """A decoder layer: the built-in layer, with the target's decode composition."""

    @classmethod
    def adopt(cls, layer: DeepseekV3DecoderLayer) -> "GlmDecoderLayer":
        """Re-class a built-in decoder layer and attach its fast-path objects.

        Raises:
            TypeError: if ``layer``, its attention modules or its MoE is not exactly the
                built-in class.
        """
        reclass(layer, DeepseekV3DecoderLayer, cls)
        layer._adopt_fast_paths()
        return layer

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        return self.forward_decode(
            position_ids, hidden_states, attn_metadata, residual, spec_metadata, **kwargs
        )

    def forward_decode(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer on a decode step."""
        if residual is None:
            residual = hidden_states
            if self.input_layernorm.nvfp4_scale is not None:
                # Layer-0 prologue: the residual-less input_layernorm folds the
                # NVFP4 kv_a_proj input-quant via its attached .nvfp4_scale
                # (RMSNorm.forward), returning an Fp4QuantizedTensor.
                # return_norm_out stashes the BF16 view DSA's pre_indexer_proj
                # needs on that tensor's unquantized_hidden_states.
                hidden_states = self.input_layernorm(hidden_states, return_norm_out=True)[0]
            else:
                hidden_states = self.input_layernorm(hidden_states)
        # Ask the attention tail to publish into the hop transport when this call will take
        # the hop (rank-uniform facts, no device work).
        into_hop_tail = None
        if (
            self._into_hop is not None
            and isinstance(self.mlp, Deepseekv3MoE)
            and self.fusion_config.PRE_MOE_FUSION
            and self.post_attention_layernorm.nvfp4_scale is None
            and self._into_hop.can_run(
                residual, self.post_attention_layernorm.weight, self._post_attn_router_weight()
            )
        ):
            into_hop_tail = self.self_attn._attn_tail
            into_hop_tail.request_into_hop(self._into_hop, residual.shape[0])
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )
        into_hop_published = into_hop_tail is not None and into_hop_tail.take_into_hop_published()
        residual = maybe_slice_for_helix_cp(
            residual, attn_metadata, self.mapping_with_cp, self.layer_idx
        )
        if isinstance(self.mlp, Deepseekv3MoE):
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MOE_FUSION = False
            return self.forward_MoE(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                into_hop_published=into_hop_published,
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MLP_FUSION = False
            assert isinstance(self.mlp, GatedMLP)
            return self.forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
                spec_metadata=spec_metadata,
            )

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        into_hop_published: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward_MoE(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        def _run_MoE(hidden_states, hidden_states_fp4, do_finalize):
            return self.mlp(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(
                    enable_allreduce=not (
                        self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                    )
                ),
                do_finalize=do_finalize,
                router_logits=router_logits,
                hop_routing_out=hop_routing_out,
            )

        router_logits = None
        hop_routing_out = None
        router_weight = self._post_attn_router_weight()
        routing_bias = self.mlp.gate.e_score_correction_bias if router_weight is not None else None
        if (
            into_hop_published
            and self._hop_routing is not None
            and self._hop_routing.can_run(
                residual, self.post_attention_layernorm.weight, router_weight, routing_bias
            )
        ):
            # The into-hop's outputs bit for bit, plus the native top-8 routing of every row
            # (rank-uniform host-only decision).
            (hidden_states, residual, router_logits, hop_routing_out) = self._hop_routing(
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
                router_weight,
                routing_bias,
            )
        elif into_hop_published:
            # The tail already published every rank's partial into the hop transport; same
            # arithmetic and bits as the post-attention hop below.
            hidden_states, residual, router_logits = self._into_hop(
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
                router_weight,
            )
        elif self.fusion_config.PRE_MOE_FUSION and self._can_use_post_attn_hop(
            hidden_states, residual, router_weight
        ):
            # Same normalized hidden and residual bits as the fused all-reduce
            # below. The router logits replace self.gate's: same FP32 GEMM,
            # different summation order, so they are not bit-identical and a
            # row whose 8th/9th expert scores lie within that rounding error
            # can select a different expert set than native.
            hidden_states, residual, router_logits = self._post_attn_hop(
                hidden_states, residual, router_weight
            )
        elif self.fusion_config.PRE_MOE_FUSION:
            # moe_backend can be either CUTLASS or TRTLLM here
            # TODO: unify the two min-latency MoE backends by enabling quant fusion
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Cross-layer expert prefetch: predict the next MoE layer's routed experts from this
        # layer's normalized hidden and prefetch their weights into L2 off the critical path
        # (no tensor the model reads changes).
        expert_prefetch = getattr(self, "_expert_prefetch_link", None)
        if expert_prefetch is not None:
            expert_prefetch.on_hop(hidden_states, router_logits)

        # The post-MoE hop decision depends only on rank-identical state, so every TP rank
        # takes the same path before the MoE runner produces any output.
        use_post_moe_hop = (
            self.post_moe_hop is not None
            and self.fusion_config.POST_MOE_FUSION
            and self.mlp.experts.has_nvfp4
            and self.post_moe_hop.can_run(hidden_states.shape[0])
        )

        # Note: this fusion pattern is only supported for single-node TRTLLM-nvfp4 backend now
        do_finalize = not use_post_moe_hop and (
            self.mapping.is_multi_node()
            or (
                not (
                    self.fusion_config.POST_MOE_FUSION
                    and hidden_states.shape[0] <= self.moe_allreduce.max_token
                    and self.model_config.moe_backend == "TRTLLM"
                    and self.mlp.experts.has_nvfp4
                    and self.is_p2p_supported
                )
            )
        )

        hidden_states = _run_MoE(hidden_states, hidden_states_fp4=None, do_finalize=do_finalize)
        if expert_prefetch is not None:
            expert_prefetch.after_moe()

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                        trigger_completion_at_end=False,
                    ),
                )
            elif use_post_moe_hop and len(hidden_states) == 2:
                # [shared, routed], with routed written into the hop by the routed MoE.
                shared_output, routed_output = hidden_states
                # The deferred-norm hop reads the same routed input with the same collective
                # protocol and returns a deferred-norm packet (rank-identical decision).
                hop = (
                    self.deferred_norm_hop
                    if self.deferred_norm_hop is not None
                    and self.deferred_norm_hop.can_run_routed(residual.shape[0], attn_metadata)
                    else self.post_moe_hop
                )
                hidden_states, residual = hop.hop_routed(
                    routed_output,
                    shared_output,
                    residual,
                    self.next_layer_layernorm.weight,
                    self.next_layer_layernorm.variance_epsilon,
                )
            elif use_post_moe_hop:
                (shared_output, fc2_output, expert_scale_factor, expanded_idx_to_permuted_idx) = (
                    hidden_states
                )
                # The deferred-norm hop: same inputs and collective protocol; returns a
                # deferred-norm packet for the next layer instead of the normalized hidden
                # state (rank-identical decision).
                hop = (
                    self.deferred_norm_hop
                    if self.deferred_norm_hop is not None
                    and self.deferred_norm_hop.can_run(residual.shape[0], attn_metadata)
                    else self.post_moe_hop
                )
                hidden_states, residual = hop(
                    fc2_output,
                    expert_scale_factor,
                    expanded_idx_to_permuted_idx,
                    shared_output,
                    residual,
                    self.next_layer_layernorm.weight,
                    self.next_layer_layernorm.variance_epsilon,
                )
            else:
                assert len(hidden_states) == 4, "hidden_states must have 4 elements"

                shared_output = hidden_states[0]
                fc2_output = hidden_states[1]
                expert_scale_factor = hidden_states[2]
                expanded_idx_to_permuted_idx = hidden_states[3]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=shared_output,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)

        if expert_prefetch is not None:
            expert_prefetch.after_post_moe()
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        if self.fusion_config.PRE_MLP_FUSION:
            if self.mlp.gate_up_proj.has_nvfp4:
                act_fp4, act_sf, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        scale=self.mlp.gate_up_proj.input_scale,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
            else:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ),
                )
        elif self.post_attention_layernorm.nvfp4_scale is not None:
            # Attention-DP path (no allreduce): post_attention_layernorm folds
            # the dense MLP's NVFP4 gate_up_proj input-quant into one kernel via
            # its attached .nvfp4_scale (RMSNorm.forward), mirroring the
            # PRE_MLP_FUSION quant but without the allreduce.
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            # No fusion
            # We need to add twoshot allreduce here to avoid modifying MLA logic
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        final_all_reduce_params = AllReduceParams(
            enable_allreduce=not (self.fusion_config.POST_MLP_FUSION or self.mlp_tp_size == 1)
        )
        mlp_output = None
        if dispatch.is_enabled("dense_mlp"):
            # Eligibility is decided on the host before any device work;
            # None selects the native module below.
            mlp_output = dispatch.get_impl("dense_mlp", "dense_mlp_forward")(
                self.mlp, hidden_states, final_all_reduce_params
            )
        if mlp_output is None:
            mlp_output = self.mlp(hidden_states, final_all_reduce_params=final_all_reduce_params)
        hidden_states = mlp_output

        if self.fusion_config.POST_MLP_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                ),
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual


class GlmMtpLayer(_AdoptedLayerMixin, _PostAttentionHopMixin, DeepseekV3MTP):
    """The draft layer: the built-in MTP layer, with the target's decode composition."""

    # The draft front packs this rank's chunk of the projection input; None keeps the
    # built-in norms and concatenation.
    draft_front = None
    # The live-row hop of the first draft step and the rows the draft loop arms it with.
    draft_live_row_hop = None
    _draft_live_rows = None
    _draft_live_rows_compacted = False

    @classmethod
    def adopt(cls, layer: DeepseekV3MTP) -> "GlmMtpLayer":
        """Re-class the built-in draft layer and attach its fast-path objects.

        Raises:
            TypeError: if ``layer``, its attention modules or its MoE is not exactly the
                built-in class.
        """
        reclass(layer, DeepseekV3MTP, cls)
        layer._adopt_fast_paths()
        layer._adopt_draft()
        return layer

    def _adopt_draft(self) -> None:
        """The draft front and the live-row hop, after the draft layer's own modules."""
        # Normalize and pack only this rank's chunk of the eh_proj input. Non-ADP TP only, and
        # only while enorm/hnorm run the FlashInfer RMSNorm the draft front mirrors.
        if dispatch.is_enabled("draft_front"):
            mapping = self.model_config.mapping
            eligible = not mapping.enable_attention_dp and mapping.tp_size > 1
            reason = "" if eligible else "requires non-ADP tensor parallelism"
            if eligible:
                reason = (
                    dispatch.get_impl("draft_front", "native_norm_reason")(self.enorm, self.hnorm)
                    or ""
                )
                eligible = not reason
            dispatch.log_capability("draft_front", "draft_front", eligible, reason)
            if eligible:
                self.draft_front = dispatch.get_impl("draft_front", "draft_front")

        # On draft step 0 the draft loop arms the layer with the live rows it selected
        # (``arm_live_rows``); the post-attention hop then keeps only those rows, and the MoE
        # and the post-MoE all-reduce run on them.
        if self.post_attn_hop is not None and dispatch.is_enabled("draft_live_rows"):
            self.draft_live_row_hop = dispatch.get_impl("draft_live_rows", "make_live_row_hop")(
                self.post_attn_hop
            )

    def arm_live_rows(self, rows: torch.Tensor) -> None:
        """Arm the next forward with the live rows of draft step 0 (``rows``: int64 row ids)."""
        self._draft_live_rows = rows
        self._draft_live_rows_compacted = False

    def disarm_live_rows(self) -> bool:
        """Clear the arming; returns whether the armed forward kept only the live rows."""
        compacted = bool(self._draft_live_rows_compacted)
        self._draft_live_rows = None
        self._draft_live_rows_compacted = False
        return compacted

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                embed_tokens=embed_tokens,
                attn_metadata=attn_metadata,
                all_rank_num_tokens=all_rank_num_tokens,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        # Split hidden_states columnwise based on TP
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank

        embeds = None
        packed = None
        if self.draft_front is not None:
            # Returns None, before launching anything, outside its support
            # domain; the native path below then reuses the embeddings.
            embeds = embed_tokens(input_ids)
            packed = self.draft_front(
                embeds,
                hidden_states,
                self.enorm.weight,
                self.enorm.variance_epsilon,
                self.hnorm.weight,
                self.hnorm.variance_epsilon,
                tp_rank,
                tp_size,
            )

        if packed is not None:
            hidden_states = packed
        else:

            def norm_embeds():
                return self.enorm(
                    embed_tokens(input_ids) if embeds is None else embeds
                )  # emdedding

            def norm_hidden():
                return self.hnorm(hidden_states)

            inputs_embeds, hidden_states = maybe_execute_in_parallel(
                norm_embeds,
                norm_hidden,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                self.aux_stream,
                disable_on_compile=True,
            )
            hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)

            if tp_size > 1 and not (self.model_config.mapping.enable_attention_dp):
                hidden_states = torch.chunk(hidden_states, tp_size, dim=-1)[tp_rank]
        hidden_states = self.eh_proj(hidden_states)

        # Input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        # MTP Layer Must have sparse MOE
        router_logits = None
        router_weight = self._post_attn_router_weight()
        use_post_attn_hop = self.fusion_config.PRE_MOE_FUSION and self._can_use_post_attn_hop(
            hidden_states, residual, router_weight
        )
        live_rows = self._draft_live_rows
        if (
            use_post_attn_hop
            and live_rows is not None
            and self.draft_live_row_hop is not None
            and self.draft_live_row_hop.can_run(hidden_states, live_rows, router_weight)
        ):
            # The live rows only, each as the post-attention hop below.
            hidden_states, residual, router_logits = self.draft_live_row_hop(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
                router_weight,
                live_rows,
            )
            self._draft_live_rows_compacted = True
            if all_rank_num_tokens is not None:
                all_rank_num_tokens = [hidden_states.shape[0]] * len(all_rank_num_tokens)
        elif use_post_attn_hop:
            # Numerics as in DeepseekV3DecoderLayer.forward_MoE.
            hidden_states, residual, router_logits = self._post_attn_hop(
                hidden_states, residual, router_weight
            )
        elif self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                )
            ),
            router_logits=router_logits,
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.shared_head.norm.weight,
                    eps=self.shared_head.norm.variance_epsilon,
                ),
            )
        else:
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        # It's for 2-model path, capture the hidden states
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)

        return hidden_states


def adopt_model(model: "GlmMoeDsaForCausalLM", plan: dispatch.Plan) -> None:
    """Re-class the built-in decoder layers, the draft layer, their attention modules, their
    MoE modules and the speculative worker, and attach the fast-path objects ``plan`` enables.

    Layer by layer in index order, the decoder layers first and the draft layer last, as the
    built-in model builds them; within a layer in the order its constructor builds the modules
    the objects wrap (this module). The speculative worker comes last, as the built-in model
    builds it after the draft layer; its draft-loop fast paths are decided per forward. The
    fast paths read the installed plan, which is ``plan``.

    Raises:
        TypeError: if a layer, its attention modules, its MoE or the speculative worker is not
            exactly the built-in class.
    """
    core = model.model
    num_layers = core.num_hidden_layers
    for layer in core.layers[:num_layers]:
        GlmDecoderLayer.adopt(layer)
    for layer in core.layers[num_layers:]:
        GlmMtpLayer.adopt(layer)
    if model.spec_worker is not None:
        GlmMtpWorker.adopt(model.spec_worker)


# The fast-path objects a decoder layer (and the draft layer) carries: per submodule (empty for
# the layer itself), the attributes that hold them. An object a target's plan never attaches
# reads None and is left out of the state.
_LAYER_STATE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "",
        (
            "post_attn_hop",
            "_into_hop",
            "_hop_routing",
            "post_moe_hop",
            "deferred_norm_hop",
            "_expert_prefetch_link",
            "draft_front",
            "draft_live_row_hop",
        ),
    ),
    ("self_attn", ("_attn_tail",)),
    ("mlp", ("router_gemm", "_routed_into_hop")),
)
# Attributes of a fast-path object that every rank must agree on; none of them holds a rank,
# a device or an address.
_STATE_FLAGS = ("engaged", "impl", "routed_mode", "routed_engaged")


def adopted_state(model: "GlmMoeDsaForCausalLM") -> dispatch.AdoptedState:
    """The fast-path state attached to the decoder layers, layer by layer.

    Per layer, each attached object as its attribute, its class name and its
    ``_STATE_FLAGS``; ``dispatch.check_rank_uniform_adopted`` compares it across the ranks.
    """
    state = []
    for index, layer in enumerate(model.model.layers):
        attached = []
        for submodule, names in _LAYER_STATE:
            owner = getattr(layer, submodule, None) if submodule else layer
            for name in names:
                obj = getattr(owner, name, None)
                if obj is None:
                    continue
                flags = tuple(
                    f"{flag}={getattr(obj, flag)!r}" for flag in _STATE_FLAGS if hasattr(obj, flag)
                )
                where = f"{submodule}.{name}" if submodule else name
                attached.append((where, type(obj).__name__) + flags)
        state.append((f"model.layers.{index}", tuple(attached)))
    return tuple(state)
