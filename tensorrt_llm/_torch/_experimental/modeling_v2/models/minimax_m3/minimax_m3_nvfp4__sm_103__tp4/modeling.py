# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: MiniMax-M3 NVFP4 / sm_103 / tp4.

MiniMax-M3's language model: 60 layers, hidden 6144, 64 query heads and 4 KV
heads of dim 128. Layers 0-2 are dense attention with a dense SwiGLU-OAI MLP;
layers 3-59 are MSA block-sparse attention (an index branch scores 128-token
blocks and the top 16 are attended) with a MoE of 128 routed experts (top 4)
plus one shared expert. The checkpoint is a modelopt mixed-precision export:
NVFP4 routed experts, MXFP8 projections, bf16 router, embedding and head; the KV
cache is NVFP4 for the sparse layers and FP8 for the dense ones.

``tp4`` is four GB300s, head-split attention (16 query heads and 1 KV head per
rank) and tensor-parallel experts (every expert on every rank, width 768 per
rank), no attention DP. Speculative decoding, when configured, is one-model
Eagle3 with draft length 3: the in-tree BF16 drafter over the full vocabulary
and the BF16 target head, both at the checkpoint's precision; decode steps of
up to 8 requests verify and draft through fused kernels at that precision
(``spec_worker.py``).

**Each step is dispatched once, on the host, from its shape** (``_select_target``):

* ``DecodeTarget`` takes a pure generation step of at most 48 tokens (up to 8
  requests, each one token or a draft verify of 1 + 3 + extra rows). It runs
  the layers with the decode-step kernels of ``layers.py`` and chooses each
  layer boundary's all-reduce by the step's token bucket (``DECODE_PLANS``).
  Every fast path it can take, the rows each covers and what a step that cannot
  take one does (raise under ``modeling_v2='require'``, else log once and fall
  back) are in ``dispatch.py``.
* ``PrefillTarget`` takes every other step (prefill, mixed steps, wider decode
  steps) on the built-in modules' generic path.

**What this target asserts rather than adapts**: SM 10.3, the topology above,
the checkpoint shape and quantization (``routing.py``), the MiniMax-M3 MSA
sparse attention backend with its NVFP4 hybrid KV cache, and text-only input:
the vision tower is not built and a step carrying image input raises.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    ModelingV2Mode,
    resolved_mode,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.mx_gemv import op as mx_gemv
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3ForCausalLM,
    get_text_model_config,
    is_minimax_m3_vl_config,
)
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture_active
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import dispatch
from . import weights as _weights
from .dispatch import AR_PUSH_MODE, EMBED_FIRSTNORM_MAX_TOKENS, DecodePlan, plan_for
from .kernels import m3_mc_allreduce, m3_prequant
from .layers import M3DecoderLayer, M3MoE, M3TextModel, prebuild_index_scorer
from .linears import adopt_m3_linears
from .ops.moe.m3_moe16_engine import m3_moe16_engine
from .ops.norm.m3_embedding_firstnorm import DecodeEmbedding, m3_embedding_firstnorm
from .runtime import make_runtime_hooks
from .spec_worker import M3Eagle3Worker, M3SpecDecForwardMixin

_SM = (10, 3)


class MiniMaxM3Tp4ModelingV2(M3TextModel):
    """The text model of this target: owns the layers and dispatches each step to a target."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._embed_owners: Dict[tuple, DecodeEmbedding] = {}
        self._embed_identity: Optional[tuple] = None
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    def build_targets(self) -> None:
        """Build the two targets, once the weights are real."""
        table = self.embed_tokens.weight
        norm = self.layers[0].input_layernorm
        identity = (
            id(table),
            table.data_ptr(),
            id(norm.weight),
            norm.weight.data_ptr(),
            float(norm.variance_epsilon),
        )
        if self._embed_owners and identity != self._embed_identity:
            raise RuntimeError(
                "the embedding or the first norm moved after the decode embedding owners were built"
            )
        if not self._embed_owners:
            self._embed_owners = {
                (rows, dtype): DecodeEmbedding(table, norm, rows, dtype)
                for rows in range(1, EMBED_FIRSTNORM_MAX_TOKENS + 1)
                for dtype in (torch.int32, torch.int64)
            }
            self._embed_identity = identity
        self.prebuild_kernels()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def decode_query_lens(self) -> List[int]:
        """The query rows per request a decode step can have: 1, and 1 + each draft length speculation runs."""
        lens = {1}
        spec = getattr(self.model_config, "spec_config", None)
        if spec is not None:
            for drafts in (
                getattr(spec, "max_draft_len", None),
                getattr(spec, "max_total_draft_tokens", None),
                *(getattr(spec, "draft_len_schedule", None) or {}).values(),
            ):
                if drafts:
                    lens.add(1 + int(drafts))
        return sorted(lens)

    def prebuild_kernels(self) -> Dict[str, object]:
        """Compile (and load) every kernel a decode forward dispatches to that would otherwise compile on its first
        call: each projection's decode GEMVs (every row count, activation and boundary push), the MoE engines of
        ``MOE_ENGINE`` / ``MOE16_ENGINE``'s token counts, and the index scorer at the decode query lengths. Runs on the
        current (main) stream before the executor's warmup, in the same order on every rank, then synchronizes; after
        it, GEMV compiles are frozen and the forward only looks kernels up (a missing one is ``dispatch.missed``).
        Returns the seconds each part took and the scorer keys."""
        times: Dict[str, object] = {}
        mx_gemv.freeze(False)
        start = time.perf_counter()
        adopt_m3_linears(self)
        times["gemv_s"] = time.perf_counter() - start
        start = time.perf_counter()
        for layer in self.layers:
            moe = layer.block_sparse_moe
            if isinstance(moe, M3MoE):
                moe.prebuild_engines()
        torch.cuda.synchronize()
        times["moe_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["scorer_keys"] = prebuild_index_scorer(
            self.layers, self.decode_query_lens(), self.embed_tokens.weight.device
        )
        torch.cuda.synchronize()
        times["scorer_s"] = time.perf_counter() - start
        mx_gemv.freeze(True)
        logger.info(
            "[M3 prebuild] decode kernels built before warmup: GEMVs %.1f s, MoE engines %.1f s, index scorer "
            "%.1f s (%d variants)"
            % (times["gemv_s"], times["moe_s"], times["scorer_s"], len(times["scorer_keys"]))
        )
        return times

    def _select_target(self, attn_metadata: AttentionMetadata, input_ids: Optional[torch.Tensor]):
        """Which target runs this step: host-side counts only, fixed per CUDA-graph capture."""
        if (
            attn_metadata.num_contexts == 0
            and input_ids is not None
            and input_ids.ndim == 1
            and dispatch.is_decode_step(input_ids.numel())
        ):
            return self._decode
        return self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if kwargs.get("lora_params") is not None:
            raise ValueError("this MiniMax-M3 target serves no LoRA adapters")
        target = (
            self._select_target(attn_metadata, input_ids)
            if inputs_embeds is None
            else self._prefill
        )
        return target.forward(
            attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, **kwargs
        )


class PrefillTarget:
    """The general case: context rows, mixed steps and wide generation steps, on the generic layer path."""

    def __init__(self, core: MiniMaxM3Tp4ModelingV2) -> None:
        self.core = core

    def forward(
        self, attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, **kwargs
    ):
        core = self.core
        hidden_states = core.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in core.layers:
            hidden_states, residual = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        return hidden_states


class DecodeTarget:
    """A pure generation step of at most 48 tokens.

    Per layer: attention (o_proj all-reduce deferred, its rows pushed into the
    boundary all-reduce up to 16 tokens) -> attention boundary (all-reduce +
    residual + post-attention RMSNorm) -> MoE (the 1..8 / 9..13-row engines,
    the first pushing its sum into the boundary all-reduce; the unfinalized MoE
    with the finalize in the all-reduce from 14 rows) or dense MLP -> MoE
    boundary (all-reduce + shared/routed add + residual + the next layer's
    RMSNorm, emitting the next QKV's MXFP8 activation). A layer whose output
    Eagle3 captures runs the generic layer. ``dispatch.py`` lists the row
    ranges and what a covered step that cannot take its path does.
    """

    def __init__(self, core: MiniMaxM3Tp4ModelingV2) -> None:
        self.core = core
        self._moe_push_descs: Dict[tuple, Any] = {}

    def forward(
        self, attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, **kwargs
    ):
        core = self.core
        num_tokens = int(input_ids.numel())
        plan = plan_for(num_tokens)
        hidden_states, residual = self._embed(input_ids, num_tokens)
        for layer in core.layers:
            if residual is None or (
                spec_metadata is not None and spec_metadata.is_layer_capture(layer.layer_idx)
            ):
                hidden_states, residual = layer(
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    residual=residual,
                    spec_metadata=spec_metadata,
                    **kwargs,
                )
            else:
                hidden_states, residual = self._layer(
                    layer, plan, position_ids, hidden_states, residual, attn_metadata, kwargs
                )
        return hidden_states

    def _embed(self, input_ids: torch.Tensor, num_tokens: int):
        """(hidden, residual) of the fused embedding + first RMSNorm; (embedding, None) when it does not run, so the
        first layer runs its own input norm."""
        core = self.core
        if not dispatch.EMBED_FIRSTNORM.covers(num_tokens):
            dispatch.uncovered(
                dispatch.EMBED_FIRSTNORM.path, f"more than {EMBED_FIRSTNORM_MAX_TOKENS} tokens"
            )
            return core.embed_tokens(input_ids), None
        owner = core._embed_owners.get((num_tokens, input_ids.dtype))
        if owner is None or not input_ids.is_contiguous():
            dispatch.missed(
                dispatch.EMBED_FIRSTNORM.path,
                num_tokens,
                f"no embedding owner for {input_ids.dtype} ids (built for int32 / int64 by build_targets)"
                if owner is None
                else "the input ids are not contiguous",
            )
            return core.embed_tokens(input_ids), None
        return m3_embedding_firstnorm(
            input_ids,
            owner.table,
            owner.norm.weight,
            owner.eps,
            owner.raw,
            owner.output,
            owner.use_pdl,
        )

    @staticmethod
    def _bucket_flags(layer: M3DecoderLayer, num_tokens: int, hidden: int):
        """(bucket workspace, its 8-CTA call counters) of the boundary all-reduce for ``num_tokens``, or None."""
        workspace = m3_mc_allreduce.get_workspace(layer.mapping, hidden)
        workspace = None if workspace is None else m3_mc_allreduce._bucket(workspace, num_tokens)
        flags = None if workspace is None else workspace.flags_for(8)
        return None if flags is None else (workspace, flags)

    @staticmethod
    def _no_bucket(path: str, num_tokens: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            dispatch.capture_missed(path, num_tokens, "the multicast all-reduce workspace")
        else:
            dispatch.missed(path, num_tokens, "no multicast all-reduce bucket holds the step")

    def _attn_push_target(self, layer: M3DecoderLayer, num_tokens: int, hidden: int):
        """The o_proj's push tuple into the attention boundary's all-reduce, or None (after a miss).

        None comes only from the token count or from workspace state created collectively in the same eager call on
        every rank, so all ranks agree. (A lone non-pushing rank would still be safe: its all-reduce pushes the same
        bf16 rows into the same set A.)"""
        found = self._bucket_flags(layer, num_tokens, hidden)
        if found is None:
            self._no_bucket(dispatch.ATTN_PUSH.path, num_tokens)
            return None
        workspace, flags = found
        return (
            workspace.mc.data_ptr(),
            flags.data_ptr(),
            workspace.rank,
            workspace.world_size,
            workspace.max_tokens,
            AR_PUSH_MODE,
        )

    def _moe_push_targets(self, layer: M3DecoderLayer, num_tokens: int, hidden: int):
        """(engine push descriptor, shared down-projection push tuple) for the MoE boundary's exchange sets, or None
        (after a miss).

        This decision must match across ranks: prepushed 3 reads set B of every rank, which a non-pushing rank never
        fills. None comes only from the token count or from state created in the same eager call on every rank
        (workspace, flags, the descriptor cache), and the engine it is armed for is agreed at build time
        (``M3MoE.prebuild_engines``), so it does."""
        from .ops.moe.m3_moe_engine import ar_push_desc

        found = self._bucket_flags(layer, num_tokens, hidden)
        if found is None:
            self._no_bucket(dispatch.MOE_PUSH.path, num_tokens)
            return None
        workspace, flags = found
        key = (workspace.mc.data_ptr(), flags.data_ptr())
        desc = self._moe_push_descs.get(key)
        if desc is None:
            if torch.cuda.is_current_stream_capturing():
                dispatch.capture_missed(
                    dispatch.MOE_PUSH.path, num_tokens, "the MoE engine's push descriptor"
                )
                return None
            desc = self._moe_push_descs[key] = ar_push_desc(
                workspace.mc.data_ptr(),
                flags.data_ptr(),
                workspace.rank,
                workspace.world_size,
                workspace.max_tokens,
                flags.device,
            )
        set_b = workspace.mc.data_ptr() + 2 * 4 * workspace.max_tokens * workspace.world_size * (
            hidden // 2
        )
        return desc, (
            set_b,
            flags.data_ptr(),
            workspace.rank,
            workspace.world_size,
            workspace.max_tokens,
            2,
        )

    def _boundary(
        self, layer: M3DecoderLayer, use_mc: bool, x, residual, norm, x2=None, quant=False
    ):
        """All-reduce + (second addend) + residual + RMSNorm: the multicast kernel, else the built-in all-reduce."""
        if use_mc:
            workspace = m3_mc_allreduce.get_workspace(layer.mapping, x.shape[-1])
            fused = (
                None
                if workspace is None
                else m3_mc_allreduce.m3_mc_allreduce_norm(
                    x,
                    residual.contiguous(),
                    norm.weight,
                    norm.variance_epsilon,
                    workspace,
                    use_pdl=get_env_enable_pdl(),
                    x2=x2,
                    quant=quant,
                )
            )
            if fused is not None:
                return fused
            dispatch.missed(
                dispatch.MC_ALLREDUCE.path,
                x.shape[0],
                m3_mc_allreduce.decline_reason(x, residual, workspace, x2=x2),
            )
        if x2 is not None:
            x = x.add_(x2)
        return layer.native_allreduce_norm(x, residual, norm)

    def _prepushed(self, layer, x, residual, norm, quant, prepushed, x2=None):
        workspace = m3_mc_allreduce.get_workspace(layer.mapping, x.shape[-1])
        fused = m3_mc_allreduce.m3_mc_allreduce_norm(
            x,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            workspace,
            use_pdl=get_env_enable_pdl(),
            x2=x2,
            quant=quant,
            prepushed=prepushed,
        )
        if fused is None:
            raise RuntimeError(
                f"layer {layer.layer_idx}: rows were pushed into an all-reduce that did not run"
            )
        return fused

    def _layer(
        self,
        layer: M3DecoderLayer,
        plan: DecodePlan,
        position_ids,
        hidden_states,
        residual,
        attn_metadata,
        kwargs,
    ):
        num_tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[-1]
        # Padded rows: the boundary all-reduce polls and re-arms rows t < M = residual.shape[0], CUDA-graph padding
        # rows included. Every producer pushes all rows of the same tensor: the o_proj / shared down GEMV t < m =
        # x.shape[0] (its row count comes from the host shape, not a live count), the MoE engine every row of the
        # num_tokens it was built for (= num_tokens here). The all-reduce checks x.sizes() == residual.sizes(), so
        # no padded row is left with a stale flag.
        # The GEMV dispatch (and with it every push into a boundary all-reduce) is a decode-graph one.
        decode_graph = flashinfer_mxfp8_decode_graph_capture_active()
        if not decode_graph:
            dispatch.uncovered(
                "decode_graph",
                "an eager decode step: the decode-graph GEMV dispatch and the boundary pushes are off",
            )
        push = plan.attn_mc_ar and decode_graph and dispatch.ATTN_PUSH.covers(num_tokens)
        armed = None
        if push:
            armed = self._attn_push_target(layer, num_tokens, hidden)
            m3_prequant.AR_PUSH.arm(armed)
        hidden_states = layer.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            **kwargs,
        )
        if push:
            m3_prequant.AR_PUSH.target = None
        if push and m3_prequant.AR_PUSH.take_pushed(hidden_states):
            hidden_states, residual = self._prepushed(
                layer, hidden_states, residual, layer.post_attention_layernorm, False, AR_PUSH_MODE
            )
        else:
            if armed is not None:
                dispatch.missed(
                    dispatch.ATTN_PUSH.path,
                    num_tokens,
                    "the o_proj did not push its rows into the armed boundary all-reduce",
                )
            hidden_states, residual = self._boundary(
                layer, plan.attn_mc_ar, hidden_states, residual, layer.post_attention_layernorm
            )

        moe = layer.block_sparse_moe
        if moe is None:
            out = layer.mlp(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
            return self._boundary(layer, plan.attn_mc_ar, out, residual, layer.next_layer_layernorm)

        norm = layer.next_layer_layernorm
        quant = plan.qkv_prequant and layer.next_is_decoder_layer
        path = dispatch.moe_path(num_tokens)
        moe_push = None
        if path == dispatch.MOE_ENGINE.path and moe.shared_experts is not None:
            # The engine itself runs inside the MoE forward; looked up (or found missing) here first so a push is
            # armed only for an engine that will take it.
            engine = moe.engine(num_tokens, shared=True)
            if engine is not None and plan.moe_mc_ar and decode_graph:
                moe_push = self._moe_push_targets(layer, num_tokens, hidden)
                if moe_push is not None:
                    m3_prequant.MOE_PUSH.arm(*moe_push)
        elif path == dispatch.MOE16_ENGINE.path and decode_graph:
            # Router, routed and shared experts in one kernel; its bf16 sum goes to the boundary all-reduce.
            engine = moe.moe16_engine(num_tokens)
            if engine is not None:
                out = m3_moe16_engine(hidden_states.contiguous(), engine)
                return self._boundary(layer, plan.moe_mc_ar, out, residual, norm, quant=quant)
        if (
            path in (dispatch.MOE16_ENGINE.path, dispatch.MC_FINALIZE.path)
            and plan.moe_mc_ar
            and moe.shared_experts is not None
        ):
            if self._bucket_flags(layer, num_tokens, hidden) is None:
                self._no_bucket(dispatch.MC_FINALIZE.path, num_tokens)
            else:
                shared, fc2, scales, perm = moe(
                    hidden_states,
                    attn_metadata,
                    final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                    do_finalize=False,
                )
                fused = m3_mc_allreduce.m3_mc_finalize_allreduce_norm(
                    fc2,
                    perm.view(num_tokens, -1).to(torch.int32),
                    scales.reshape(num_tokens, -1),
                    shared,
                    residual,
                    norm.weight,
                    norm.variance_epsilon,
                    m3_mc_allreduce.get_workspace(layer.mapping, hidden),
                    use_pdl=get_env_enable_pdl(),
                    quant=quant,
                )
                if fused is None:
                    raise RuntimeError(
                        "the MC finalize all-reduce rejected the unfinalized MoE outputs"
                    )
                return fused
        parts = moe(
            hidden_states,
            attn_metadata,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False),
            return_parts=True,
        )
        shared, routed = parts if isinstance(parts, tuple) else (parts, None)
        if moe_push is not None:
            m3_prequant.MOE_PUSH.disarm()
            if routed is None:
                # The engine folded the shared expert in and pushed routed + shared into set A.
                folded_pushed = m3_prequant.MOE_PUSH.engine_pushed == shared.data_ptr()
                m3_prequant.MOE_PUSH.engine_pushed = m3_prequant.MOE_PUSH.shared_pushed = None
                if folded_pushed:
                    return self._prepushed(layer, shared, residual, norm, quant, 2)
            else:
                routed_pushed, shared_pushed = m3_prequant.MOE_PUSH.take_pushed(shared, routed)
                if routed_pushed and shared_pushed:
                    return self._prepushed(layer, shared, residual, norm, quant, 3, x2=routed)
                if routed_pushed or shared_pushed:
                    raise RuntimeError("only one MoE part pushed into the boundary all-reduce")
            dispatch.missed(
                dispatch.MOE_PUSH.path,
                num_tokens,
                "the MoE did not push its output into the armed boundary all-reduce",
            )
        return self._boundary(layer, plan.moe_mc_ar, shared, residual, norm, x2=routed, quant=quant)


class ModelingV2MiniMaxM3Nvfp4Sm103Tp4(M3SpecDecForwardMixin, MiniMaxM3ForCausalLM):
    """MiniMax-M3 NVFP4 / sm_103 / tp4, text only, with one-model Eagle3 or no speculation.

    A subclass of the built-in text model shell for its loader, its KV-cache and
    transceiver preferences and its one-engine speculative plumbing; the text
    model it wraps, the forward and the speculative worker are this target's.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        assert torch.cuda.get_device_capability() == _SM, (
            f"this target is built for sm_{_SM[0]}{_SM[1]}, the device is sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        # Under modeling_v2='require' a covered decode shape that cannot take its fast path raises (dispatch.py).
        dispatch.set_require(resolved_mode(type(self)) is ModelingV2Mode.REQUIRE)
        if is_minimax_m3_vl_config(model_config.pretrained_config):
            model_config = get_text_model_config(model_config)
        if model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4:
            # The one-model Eagle3 layer has no NVFP4 trtllm-gen decode kernel: keep its modules and the shared
            # draft cache on FP8 while the target stays NVFP4.
            model_config.extra_attrs["draft_kv_cache_quant_algo_override"] = QuantAlgo.FP8
        # The one-engine shell's constructor, past the built-in one that builds the built-in text model.
        SpecDecOneEngineForCausalLM.__init__(
            self, MiniMaxM3Tp4ModelingV2(model_config), model_config
        )
        self.runtime_hooks, self._spec_fold = make_runtime_hooks(self)
        if self.spec_worker is not None:
            self.spec_worker = M3Eagle3Worker.adopt(self.spec_worker)
            worker = self.spec_worker
            for layer in self.model.layers:
                layer.capture = worker.capture_hidden_states

    def load_weights(self, weights: Dict, *args, **kwargs) -> None:
        _weights.load(self, weights, *args, **kwargs)

    def setup_aliases(self) -> None:
        """Chain each decoder layer's next_layer_layernorm for the fused boundaries."""
        super().setup_aliases()
        layers = self.model.layers
        for idx, layer in enumerate(layers):
            layer.next_is_decoder_layer = idx < len(layers) - 1

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self.model.build_targets()
        if self.spec_worker is not None:
            # The fused verify's coded target head, from the loaded weights, before the executor's warmup.
            start = time.perf_counter()
            if self.spec_worker.prebuild_target_head(self.lm_head):
                torch.cuda.synchronize()
                logger.info(
                    "[M3 prebuild] coded target head built before warmup: %.1f s"
                    % (time.perf_counter() - start)
                )
            # The fused draft loop's head copy, from the loaded draft weights, before the executor's warmup.
            start = time.perf_counter()
            if self.spec_worker.prebuild_draft_head(self.draft_model):
                torch.cuda.synchronize()
                logger.info(
                    "[M3 prebuild] fused draft head built before warmup: %.1f s"
                    % (time.perf_counter() - start)
                )

    def forward(self, attn_metadata: AttentionMetadata, *args, **kwargs):
        """The fused-verify one-engine forward (``M3SpecDecForwardMixin``), text only."""
        multimodal_params: List[Any] = kwargs.pop("multimodal_params", None) or []
        if any(
            param is not None and getattr(param, "multimodal_data", None)
            for param in multimodal_params
        ):
            raise ValueError(
                "this MiniMax-M3 target is text only; the request carries multimodal input"
            )
        return super().forward(attn_metadata, *args, **kwargs)
