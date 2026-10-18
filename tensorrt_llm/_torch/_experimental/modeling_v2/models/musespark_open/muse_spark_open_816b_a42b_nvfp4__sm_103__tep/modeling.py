# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: muse-spark-open-816b-a42b-nvfp4 / sm_103 / tep8.

Muse Spark Open, NVFP4 export: 62 decoder layers, hidden 8192, 128 query and 16
key-value heads of width 64; every fourth layer counted from the last is a
full-attention NoPE layer and every other layer a 2048-token sliding-window RoPE
layer; the MoE routes each token to 8 of 256 experts at half the residual width.
``tep8`` is the attention heads tensor-parallel and the experts expert-parallel
over eight ranks. Speculation is DFlash, whose drafter is a separate checkpoint,
or none.

The shell is the built-in ``MuseSparkOpenOmniForConditionalGeneration``: the same
module tree, the same weights, the same engine preferences (KV cache manager,
transceiver, model defaults) and the same speculative-decoding path, because the
built-in ``MuseSparkOpenForCausalLM`` it wraps carries the one-engine DFlash
worker and builds the drafter from ``speculative_model``. Construction builds the
built-in model, then adopts its text model as ``MuseSparkOpenTepModelingV2``,
whose forward hands each step to ``PrefillTarget`` or ``DecodeTarget``.

Both targets call the built-in modules, so a step launches the kernels the
built-in implementation launches:

* ``PrefillTarget`` is the general case -- context rows, possibly with
  generation rows beside them -- and runs the built-in text-model forward.
* ``DecodeTarget`` is the specialization, routed to only when a step has no
  context rows: with DFlash, each request's last accepted token and its drafts.
  Its forward is the built-in forward's decode path written out over the
  built-in modules. The prefill sequence-parallel split, which needs context
  rows, is absent. A decode op that replaces a built-in call replaces it here.

Still built-in, and reached from here when a change needs it: the drafter and
the DFlash worker (``llm.draft_model``, ``llm.spec_worker``), the logits
processor and the weight loader (``weights.py``). Entries only this target calls
go under ``ops/<category>/`` beside this file.

The fast paths a model is built with are fixed by its composition
(``TARGET_PATHS``, keyed by the GPU architecture and the tensor-parallel size,
each with the GPU topology it was validated on, for DFlash drafting seven
tokens on an NVFP4 checkpoint; ``dispatch.compose``), decided once when the
shell is constructed: all of them,
or none (``dispatch.NO_PATHS``) when the deployment has no composition, and the
target then runs only the built-in modules as described above. A missing public
precondition of a planned fast path is an error, whatever the mode. Over a
windowed drafter context arena the fast paths written for a full-length arena
are left out of the plan (``dispatch.FULL_CONTEXT_ARENA_PATHS``). The installers
ask the plan per fast path (``Plan.enabled``); each paragraph below names one
fast path of ``dispatch.FAST_PATHS``. Their installers live under
``kernels/<fast path>/`` and their launches under ``ops/<category>/``.

Target attention (installed by the shell's constructor, ``layers.adopt_model``):

* ``preattn_prep`` re-classes each layer's attention as
  ``MuseSparkOpenTepAttention`` and has ``DecodeTarget`` hand it
  ``preattn_prep``, which replaces the built-in pre-attention prep on decode
  steps of at most 8 rows.
* ``swa_paged_decode`` re-classes each layer's attention as
  ``MuseSparkOpenTepDecodeAttention`` and has ``DecodeTarget`` hand it a
  ``DecodeAttentionPlan``: the sliding-window layers run
  ``paged_decode_attention`` (Q in E4M3, FP8 K/V write) in place of the built-in
  KV append and generation FMHA. With ``preattn_prep`` the attention is one
  class carrying both (``MuseSparkOpenTepPrepDecodeAttention``): the prep, then
  the decode attention on the fused QKV the prep produced.
* ``swa_block`` re-classes the attention once more: on a sliding-window layer's
  decode step the block from the projection to the output projection's input
  runs as ``ops/attention/swa_block`` -- the split kernel's cache reads issued
  before its dependency wait, the prep folded into the split and the output
  norm-gate folded into the combine -- with the bits of the
  ``preattn_prep`` + ``swa_paged_decode`` chain.
* ``full_attention_prep_append`` re-classes the full-attention layers' attention
  with ``FullAppendAttention`` in front of the class above (the ``swa_block``
  class acts only on sliding-window layers): their decode prep also writes the
  KV append, the E4M3 query and the FMHA scale words, and the built-in
  generation preprocess launch stands down. With ``dense_gemm_prefetch`` it also
  clears the attention output prefetch that each of those layers' QKV site
  would fork.
* ``full_attention_split_kv`` re-classes them with ``FullAttnSplitKvAttention``
  in front: on the steps the prep append folds, the decode FMHA call launches
  ``ops/attention/full_attention_split_kv``'s page-split partials, and the
  output gate merges them (``ops/attention/combine_norm_gate``) in place of the
  norm-gate. Those layers run the kernel's spill-free build
  (``install_spill_free``).
* ``full_attention_prep_fold`` re-classes them with
  ``FullAttnSplitKvFoldAttention`` in front: on those steps the prep is not
  launched; ``ops/attention/full_attention_prep_fold`` normalizes Q and appends
  the step's K/V inside the attention kernel, launched from ``_attn_impl``
  without the built-in call chain.

Dense projections and prefetch:

* ``dense_gemms`` re-classes every layer's QKV, attention output, pooler and
  unpooler ``Linear`` so that their calls of at most 8 rows run one named
  kernel per shape on every rank. The other fast paths' GEMM calls on those
  modules (the built-in attention's, the pool GEMM, the unpool GEMM handed to
  the boundary) go through the same dispatchers.
* ``dense_gemm_prefetch`` has those dispatchers also prefetch the next
  projection's weight into L2 on a side stream, a fork that both targets join
  by the end of their forward.
* ``next_qkv_prefetch`` registers a named fork in the dense GEMM plan, joined
  by the QKV dispatchers only, and gives every MoE module whose next QKV edge
  ``boundary_early_release`` converted the next layer's QKV weight: the stacked
  expert call forks its L2 prefetch right after the expert launch, and
  ``DecodeTarget`` forks layer 0's before the embedding. No module is
  re-classed.

Hops and boundaries:

* ``post_attention_hop`` re-classes each layer's attention-side residual join
  (``add_attn``, and nothing else) as ``MuseSparkOpenTepPostAttnJoin``. On
  decode steps of at most 8 rows ``DecodeTarget`` arms those joins and has
  every layer call pass ``all_reduce_params=AllReduceParams(enable_allreduce=False)``,
  which the built-in layer hands to its attention: the o_proj partial skips the
  built-in all-reduce and one kernel reduces it, joins it into the residual and
  applies the pre-FFN norm. The hop triggers its dependents early and re-reads
  only the peer slots still stale, all such reads in flight together. The
  decoder layer, its attention and its o_proj keep their classes, so the
  attention paths and ``projection_gathers`` compose with it in either order.
* ``attention_output_into_hop`` changes who publishes the o_proj partial on
  decode steps of 8 rows: the arming also hands every layer's o_proj dispatcher
  (``DenseGemmDispatcher.publish``) this rank's slot of the hop's direct
  transport for the layer's parity (layer ``l``: ``l mod 2``), so the o_proj kernel
  (cuBLAS on sm103, flashinfer's split-K on sm107: ``DirectPublish.PRODUCER_FAMILIES``)
  writes the partial into every rank's copy, and the join runs
  ``post_attn_consumer_hop`` (poll, reset, sum, boundary) instead of the hop.
  Other row counts keep the hop (``DirectPublish``). At those 8 rows o_proj
  forks no prefetch of the pool weight. After the weights load,
  ``verify_direct_publish`` checks the kernel the published o_proj captures; a
  step that raises while armed resets the transport.
* ``projection_gathers`` re-classes each layer's decoder, MoE and post-FFN join
  and has ``DecodeTarget`` pass ``fused_unpool_gather`` / ``fused_pool_gather``,
  which fuse the sharded projections' column gathers into the post-FFN boundary
  kernel and the pre-expert FP4 quantization kernel.
* ``boundary_early_release`` moves two dependent-launch triggers forward: in the
  fused unpool boundary (which then releases the next layer's QKV projection at
  its entry) and in the attention output's norm-gate, which a decode step then
  runs as a PDL copy of the built-in kernel releasing the output projection
  right after its own wait (the attention carries it as one more class in the
  attention class lattice). It drops the dense GEMM prefetch on the edges it
  converts; the layers whose output DFlash captures keep both the exit trigger
  and the prefetch, since a plain capture copy precedes their next QKV.
* ``unpool_into_boundary`` re-classes each fused post-FFN join once more: on
  the 8-row decode graph the unpool GEMM writes its slice straight into a
  multicast transport slot and the join's boundary kernel only polls it
  (``ops/comm/unpool_direct_boundary``); eager steps and other row counts keep
  the tagged inbox of ``projection_gathers``.
* ``capture_fold`` re-classes each decoder layer once more and has
  ``DecodeTarget`` pass ``capture_fold``: on the DFlash capture layers the fused
  post-FFN boundary also writes the layer's BF16 capture into the DFlash
  buffer, and the built-in capture copies of that layer are not launched.

MoE:

* ``moe_prologue`` re-classes the MoE once more: on a decode step of at most 8
  rows its input prep runs the router shard, the pool slice GEMM and the routed
  pool kernel on one stream, and fills the routing memo and the expert
  placement for the experts and the finalize. It also gives those layers a side
  stream (``install_router_fork``): the router shard is forked onto it as a
  plain launch after the attention hop and joined before the routed pool
  kernel, and the pool GEMM follows the hop on the programmatic chain.
* ``pool_into_route`` builds a direct transport for those layers: on the 8-row
  decode graph the pool GEMM writes its slice into the transport through the
  dense GEMM dispatcher, the router shard's publish twin writes its logits into
  it on the router side stream, and ``ops/comm/pool_direct_route`` only polls
  them before routing and quantizing. Its sites are found and compiled once the
  weights are final (``install_loaded_features``, after ``early_expert_ids``).
* ``finalize_transport`` puts a ``FinalizeTransportWorkspace`` in every MoE
  layer's fused finalize attribute, in front of the built-in workspace: calls
  of 1..16 tokens run ``ops/moe/finalize_allreduce_transport``, bitwise the
  built-in kernel's result on a different transport; larger calls run the
  built-in kernel. No module is re-classed.

The expert paths change the expert layers once their weights are final and are
installed by the shell's ``load_weights`` after the built-in loader
(``install_loaded_features``):

* ``replicated_experts`` replicates each rank's ring predecessor's experts and
  patches every layer's experts module and routing method, whatever class the
  MoE module itself has.
* ``persistent_experts`` routes the NVFP4 layers' stacked calls of at most 8
  rows to the persistent expert kernel.
* ``three_window_placement`` stacks a third ring window on them.
* ``bf16_routed_experts`` gives the two BF16 layers' stacked calls unpacked
  routing.
* ``early_expert_ids`` gives the layers that run the prologue's routed kernel
  one early-ids buffer, so the expert kernel starts its weight stream from ids
  the routed kernel publishes before it ends.

``replicated_embedding`` is installed by ``load_weights`` too, after the expert
layers (``install_embedding_features``): it all-gathers the token embedding's
shards into a whole table on every rank and re-classes ``embed_tokens``, so
``DecodeTarget``'s embedding call -- and the drafter's, which shares the
module -- is one row gather.

The modules that install fast paths by patching built-in modules sit under
``kernels/``, as does ``persistent_experts/rows.py``, the (token, slot) -> row
contract between an unfinalized expert call and the MoE finalize;
``ops/<category>/`` holds only ``OpWrapper`` entries.

The drafter paths change the drafter or the DFlash worker and are installed on
them once the drafter's weights have loaded (``drafter.adopt_drafter``), in this
order:

* ``drafter_block_conv`` and ``drafter_block_decode`` on the drafter's modules;
* ``draft_head`` on the DFlash worker and the drafter's logits processor;
* ``drafter_fc_shard``, ``fused_accept`` and ``drafter_fixed_gemms`` on the
  drafter's ``fc`` and GEMMs and on the worker's acceptance -- after
  ``draft_head``, because ``fused_accept`` derives its worker class from
  ``draft_head``'s while ``draft_head`` accepts only the built-in class;
* ``context_kv_fork``, which re-classes the worker on top of those, binds the
  fork point on the LM head's GEMM call and is armed per decode step by
  ``DecodeTarget``;
* ``verify_fold`` and ``path_walk_tail`` on top of that (on a folded step its
  prepare takes the context-K/V fork's side chain);
* ``drafter_hops`` on the drafter's block decode;
* ``drafter_pdl_chain``.

As with the target paths, a drafter path of the plan that cannot be installed
raises (``ValueError``) instead of leaving the built-in path to run under its
name; the installs run from every load hook and are logged only by the call
that changed something.

The executor paths (``deferred_input_prep``, ``sampler_store``) change neither
the model nor the drafter: ``runtime.py`` stages the work between two decode
CUDA graphs, and the shell declares it to the executor as its
``runtime_hooks`` (``tensorrt_llm/_torch/pyexecutor/runtime_hooks.py``), both
the step-input stager and the sampler store writer. The shell's constructor
builds the hooks right after the composition is decided and before the text
model is adopted (``runtime.make_runtime_hooks``) whenever the composition is
not empty; ``runtime`` is imported only then.

The head paths (``split_sampler``, ``lm_head_inbox``) are installed from the
same hooks, after the drafter paths: ``split_sampler/target_head.py``
re-classes the target logits processor and the DFlash worker (as a class
derived from the class ``draft_head`` / ``fused_accept`` left it with) so that
decode steps' head logits stay BF16 and are sampled by
``ops/spec/head_sampler.py``; ``lm_head_inbox`` also lets the LM-head GEMM
write into a multicast inbox that the sampler gathers from (and fires the
context-K/V fork's head fork point right after that GEMM).
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import resolved_mode
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

from . import dispatch as _dispatch
from . import layers as _layers
from . import weights as _weights
from .attention import DecodeAttentionPlan
from .layers import _NO_ALLREDUCE, PostAttnHopCall
from .ops.attention.preattn_prep import PreattnPrep, preattn_prep


class MuseSparkOpenTepModelingV2(MuseSparkOpenModel):
    """The text model: the built-in ``MuseSparkOpenModel``, whose forward picks a target per step.

    Never constructed directly: ``adopt`` re-classes the built-in text model the
    shell has already built, so the module tree, the parameters and their names
    stay the built-in model's. The class adds methods and plain attributes only.
    """

    _prefill: "PrefillTarget"
    _decode: "DecodeTarget"
    _plan: "_dispatch.Plan"
    _decode_attention: Optional[DecodeAttentionPlan]
    _dense_gemm_plan: Optional[object]
    _layer_kwargs: dict
    _post_attn_hop: Optional[PostAttnHopCall]
    _norm_gate_trigger: Optional[str]
    _finalize_transport: Optional[object]
    _swa_block: Optional[object]
    _unpool_direct: Optional[object]
    _pool_direct: Optional[object]
    _qkv_prefetch: Optional[object]

    @classmethod
    def adopt(
        cls, model: MuseSparkOpenModel, plan: "_dispatch.Plan" = _dispatch.NO_PATHS
    ) -> "MuseSparkOpenTepModelingV2":
        """Re-class the built-in text model as this core and build its two targets.

        ``plan`` is the model's composition (``dispatch.compose``); with
        ``NO_PATHS`` the core runs the built-in modules only.

        Raises:
            TypeError: if ``model`` is not exactly the built-in ``MuseSparkOpenModel``.
            ValueError: if a fast path of the plan cannot run on this model.
        """
        if type(model) is not MuseSparkOpenModel:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenModel, got {type(model).__name__}"
            )
        adopted = _layers.adopt_model(model, plan)
        model.__class__ = cls
        model._plan = plan
        model._finalize_transport = adopted.finalize_transport
        model._decode_attention = adopted.decode_attention
        model._dense_gemm_plan = adopted.dense_gemm_plan
        model._layer_kwargs = adopted.layer_kwargs
        model._post_attn_hop = adopted.post_attn_hop
        model._norm_gate_trigger = adopted.norm_gate_trigger
        model._swa_block = adopted.swa_block
        model._unpool_direct = adopted.unpool_direct
        model._pool_direct = adopted.pool_direct
        model._qkv_prefetch = adopted.qkv_prefetch
        model._prefill = PrefillTarget(model)
        model._decode = DecodeTarget(model)
        return model

    def _join_prefetch(self) -> None:
        """Join a dense GEMM prefetch fork still outstanding on the current stream.

        A target calls this at the end of its forward, so a fork made inside a
        CUDA graph capture is joined inside it even when its consumer did not run.
        """
        _join_prefetch(self)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment. A mixed batch -- context and generation
        rows in one step -- goes to prefill, so prefill is the general case and
        decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def _join_prefetch(core) -> None:
    """``core``'s dense GEMM prefetch fork and named forks joined on the current stream, if
    outstanding.

    The plan is read at call time (a core without ``dense_gemms`` has none).
    """
    plan = getattr(core, "_dense_gemm_plan", None)
    if plan is not None:
        if getattr(plan, "named_forks", None):
            plan.join_all()
        else:
            plan.join()


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them.

    Runs the built-in text-model forward, including its prefill sequence-parallel
    split when that is enabled.
    """

    def __init__(self, core: MuseSparkOpenTepModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = MuseSparkOpenModel.forward(
            self.core,
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        _join_prefetch(self.core)
        return hidden_states


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    The built-in forward's decode path over the built-in modules: the token
    embedding, the weightless embedding norm and ``m_emb``, the FP32 residual
    stream, the decoder layers -- each given the next layer's pre-QKV norm so it
    can hand over that layer's attention input -- and the final norm. Every
    keyword the step carries (``spec_metadata`` among them, for the DFlash
    hidden-state capture) reaches every layer, as in the built-in forward.

    With ``preattn_prep`` every layer call also carries ``preattn_prep``, which the
    adopted attention runs in place of its built-in pre-attention prep. With
    ``projection_gathers`` every layer call also carries ``fused_unpool_gather`` and
    ``fused_pool_gather``, which the re-classed decoder layer takes as leave to fuse
    its projection gathers into their consumers. With ``capture_fold`` every layer
    call also carries ``capture_fold``, which the decoder layer takes as leave to
    have its post-FFN boundary write the layer's DFlash capture. With
    ``swa_paged_decode`` every layer call also carries ``decode_attention`` (the
    ``DecodeAttentionPlan``), which the adopted attention takes for its attention
    op. With ``boundary_early_release`` every layer call also carries
    ``norm_gate_pdl`` (a trigger name), which the adopted attention takes as leave
    to run its output gate as the PDL norm-gate kernel. With ``swa_block`` every
    layer call also carries ``swa_block`` (``SwaBlockPlan``), which a
    sliding-window layer's attention takes for its decode attention block.

    With ``post_attention_hop``, a step of at most 8 rows arms every layer's
    attention-side join (``PostAttnHopCall.armed``) for the forward and every
    layer call also carries ``all_reduce_params=AllReduceParams(enable_allreduce=False)``,
    which the built-in layer hands to its attention: the o_proj partial reaches
    the armed join unreduced, and the join runs the fused hop in place of the
    built-in o_proj all-reduce and residual join. With
    ``attention_output_into_hop`` a step of ``DirectPublish.ROWS`` rows also arms
    the direct publication: o_proj writes into the hop's transport and the join
    runs the consumer hop.

    With ``unpool_into_boundary``, a forward of exactly 8 rows captured into a
    CUDA graph arms every fused layer's unpool transport (``DirectUnpool.armed``):
    the unpool GEMM writes into the layer's transport slot and the post-FFN join
    consumes it.

    With ``pool_into_route``, a forward of exactly 8 rows captured into a CUDA
    graph arms every routed layer's pool transport (``DirectPool.armed``): the
    pool GEMM writes into the layer's pool slot, the forked router shard into its
    logit boxes, and the routed pool kernel only polls them.

    With ``next_qkv_prefetch``, the forward first forks the L2 prefetch of layer
    0's QKV weight (``QkvPrefetch.graph_start``); every other layer's is forked by
    the previous layer's expert call.
    """

    def __init__(self, core: MuseSparkOpenTepModelingV2) -> None:
        self.core = core
        self.preattn_prep: Optional[PreattnPrep] = (
            preattn_prep if core._plan.enabled("preattn_prep") else None
        )
        self.layer_kwargs = dict(core._layer_kwargs)
        self.post_attn_hop: Optional[PostAttnHopCall] = getattr(core, "_post_attn_hop", None)
        self.decode_attention: Optional[DecodeAttentionPlan] = getattr(
            core, "_decode_attention", None
        )
        self.norm_gate_trigger: Optional[str] = getattr(core, "_norm_gate_trigger", None)
        self.swa_block = getattr(core, "_swa_block", None)
        self.unpool_direct = getattr(core, "_unpool_direct", None)
        self.pool_direct = getattr(core, "_pool_direct", None)
        self.qkv_prefetch = getattr(core, "_qkv_prefetch", None)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if self.qkv_prefetch is not None:
            # Layer 0's QKV weight, asked of the L2 while the embedding chain runs.
            self.qkv_prefetch.graph_start(
                int((input_ids if input_ids is not None else inputs_embeds).shape[0])
            )
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        embeds = core.embed_norm(inputs_embeds.to(core.dtype))
        if core.m_emb != 1.0:
            embeds = embeds * core.m_emb
        hidden_states = embeds.to(torch.float32)
        if self.preattn_prep is not None:
            kwargs = dict(kwargs, preattn_prep=self.preattn_prep)
        if self.decode_attention is not None:
            kwargs = dict(kwargs, decode_attention=self.decode_attention)
        if self.norm_gate_trigger is not None:
            kwargs = dict(kwargs, norm_gate_pdl=self.norm_gate_trigger)
        if self.swa_block is not None:
            kwargs = dict(kwargs, swa_block=self.swa_block)
        if self.layer_kwargs:
            kwargs = dict(kwargs, **self.layer_kwargs)
        hop = self.post_attn_hop
        hop_armed = hop is not None and hop.takes(hidden_states.shape[0])
        if hop_armed:
            kwargs = dict(kwargs, all_reduce_params=_NO_ALLREDUCE)
        # Arm the DFlash context-K/V fork with this decode step (kernels/context_kv_fork).
        fork = getattr(core, "_context_kv_fork", None)
        if fork is not None:
            fork.arm(kwargs.get("spec_metadata"), attn_metadata)

        direct = self.unpool_direct
        rows = hidden_states.shape[0]
        direct_armed = direct is not None and direct.takes(rows)
        pool = self.pool_direct
        pool_armed = pool is not None and pool.takes(rows)

        layers = core.layers
        precomputed_attn_input = None
        arming = (
            hop.armed(hidden_states.shape[0], hidden_states)
            if hop_armed
            else contextlib.nullcontext()
        )
        with (
            arming,
            direct.armed(rows) if direct_armed else contextlib.nullcontext(),
            pool.armed(rows) if pool_armed else contextlib.nullcontext(),
        ):
            for index, decoder_layer in enumerate(layers):
                next_norm = layers[index + 1].pre_qkv_proj_norm if index + 1 < len(layers) else None
                result = decoder_layer(
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    precomputed_attn_input=precomputed_attn_input,
                    next_layer_norm=next_norm,
                    sp_num_tokens=None,
                    **kwargs,
                )
                if next_norm is None:
                    hidden_states = result
                else:
                    hidden_states, precomputed_attn_input = result
        hidden_states = core.norm(hidden_states.to(core.dtype))
        _join_prefetch(core)
        return hidden_states


#: TEP8: every fast path.
_TEP8_PATHS = _dispatch.FAST_PATH_NAMES
#: TEP4: the fast paths written for any rank count whose output is bitwise the
#: built-in model's, and ``drafter_fc_shard``, whose sharded drafter projection
#: can move a draft token's bf16 last place (the target still verifies every
#: draft; it shards every row count at TP4).
_TEP4_PATHS = frozenset(
    {
        "drafter_fc_shard",
        "preattn_prep",
        "full_attention_prep_append",
        "post_attention_hop",
        "deferred_input_prep",
        "sampler_store",
        "replicated_embedding",
        "fused_accept",
        "verify_fold",
        "draft_head",
        "path_walk_tail",
        "drafter_block_conv",
        "context_kv_fork",
        "split_sampler",
    }
)

#: The compositions, per (SM, tensor-parallel size) (``dispatch.CompositionKey``):
#: the fast paths served and the GPU topology they were validated on, with DFlash
#: drafting a linear chain of ``dispatch.DRAFT_LEN`` tokens on an NVFP4
#: checkpoint; every other deployment runs the built-in modules only.
TARGET_PATHS = {
    (103, 8): _dispatch.TargetPaths(served=_TEP8_PATHS, topology=_dispatch.TWO_TRAYS_OF_4),
    (107, 8): _dispatch.TargetPaths(
        served=_TEP8_PATHS,
        topology=_dispatch.TWO_TRAYS_OF_4,
        preconditions=(_dispatch.SM107_FUSED_KERNELS,),
    ),
    (103, 4): _dispatch.TargetPaths(
        served=_TEP4_PATHS,
        topology=_dispatch.ONE_TRAY_OF_4,
        preconditions=(_dispatch.NO_PREFILL_SEQUENCE_PARALLEL,),
    ),
    (107, 4): _dispatch.TargetPaths(
        served=_TEP4_PATHS,
        topology=_dispatch.ONE_TRAY_OF_4,
        preconditions=(_dispatch.NO_PREFILL_SEQUENCE_PARALLEL, _dispatch.SM107_FUSED_KERNELS),
    ),
}


def device_compute_capability() -> Optional[Tuple[int, int]]:
    """This rank's compute capability, the fact ``route`` routes on; None without a CUDA device.

    Read once, when the shell is constructed; the composition is keyed by it
    (``TARGET_PATHS``), and a deployment without a device has none.
    """
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


class ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep(MuseSparkOpenOmniForConditionalGeneration):
    """Muse-Spark-Open-816B-A42B-NVFP4 on sm_103 at tep8: the built-in model with this target's core."""

    #: The fast paths this model is built with (``dispatch.compose``).
    _plan: "_dispatch.Plan" = _dispatch.NO_PATHS
    #: The parallel mapping the model was built with (the drafter shares it).
    _mapping: object
    #: The executor hooks this model declares (``runtime.make_runtime_hooks``); None declares none.
    _runtime_hooks = None

    @property
    def runtime_hooks(self):
        """The executor hooks this model declares, read by ``runtime_hooks_of``: both parts
        (``runtime.make_runtime_hooks``) under a composition, None under ``NO_PATHS``."""
        return self._runtime_hooks

    def __init__(self, model_config: ModelConfig, *args, **kwargs):
        """Build the built-in model, decide its composition, and adopt its text model.

        Under ``modeling_v2="require"`` a deployment without a composition fails
        before the built-in constructor runs. The preconditions read the built
        model (and the drafter arena its DFlash worker keeps), so the plan is
        decided right after it, before anything is installed
        (``dispatch.compose``, agreed over the TP group).

        Raises:
            ValueError: under ``require`` without a composition; under any mode
                with a precondition of the plan missing (``dispatch.compose``).
        """
        mode = resolved_mode(type(self))
        capability = device_compute_capability()
        _dispatch.require_served(model_config, mode, TARGET_PATHS, capability)
        super().__init__(model_config, *args, **kwargs)
        plan = _dispatch.compose(
            self.llm.model,
            model_config,
            mode,
            TARGET_PATHS,
            capability,
            spec_worker=getattr(self.llm, "spec_worker", None),
        )
        self._plan = plan
        self._mapping = getattr(model_config, "mapping", None)
        if plan and self._runtime_hooks is None:
            # Every composition serves both executor fast paths
            # (dispatch.TargetPaths); NO_PATHS declares no hooks.
            from . import runtime as _runtime

            self._runtime_hooks = _runtime.make_runtime_hooks(self)
        # The built-in constructor returns early when it is re-entered; adopt once.
        if not isinstance(self.llm.model, MuseSparkOpenTepModelingV2):
            MuseSparkOpenTepModelingV2.adopt(self.llm.model, plan)

    def load_weights(self, weights, weight_mapper=None):
        """The built-in load, then the plan's weight-dependent fast paths (the expert
        paths, then ``replicated_embedding``).

        Expert tensors are final once the built-in loader returns, so this is
        the first point ``replicated_experts`` (``ring_buddy.install``, which
        copies them) can run; it is collective, and every rank reaches it before
        any CUDA graph is captured. From the plan decided when the shell was built.
        With ``attention_output_into_hop`` it then checks the kernel the
        published o_proj runs (``verify_direct_publish``, collective, needs the
        loaded weight). Under ``NO_PATHS`` this is the built-in load and nothing
        else.
        """
        result = _weights.load(self, weights, weight_mapper)
        plan = self._plan
        if not plan:
            return result
        _weights.install_loaded_features(self.llm.model, self.model_config.mapping, plan)
        # The token embedding after the experts, so the expert replication's
        # all-gather and this one's never interleave.
        _weights.install_embedding_features(self.llm.model, self.model_config.mapping, plan)
        if plan.enabled("attention_output_into_hop"):
            _layers.verify_direct_publish(self.llm.model)
        return result

    def load_draft_weights(self, *args, **kwargs):
        """The built-in drafter load, then the planned drafter fast paths.

        The drafter's DFlash 2 modules and selector, and the target LM head it
        borrows, are in place only once its checkpoint is loaded, so this is the
        first point a drafter fast path can be installed. The block installs act
        on the drafter's own modules (``drafter_block_conv``,
        ``drafter_block_decode``) and the ``draft_head`` install on the DFlash
        worker and the drafter's logits processor: different objects, so the
        order does not matter, and both are idempotent. Under ``NO_PATHS`` this
        is the built-in load and nothing else.

        Raises:
            ValueError: if a planned drafter fast path cannot be installed
                (``_install_draft_features``).
        """
        result = super().load_draft_weights(*args, **kwargs)
        self._install_draft_features()
        return result

    def post_load_weights(self) -> None:
        """Install the selected drafter features if no ``load_draft_weights`` call has.

        ``LoadFormat.DUMMY`` never calls ``load_draft_weights``: it initializes
        the weights and lets the drafter borrow the target's head directly.
        The loader's post-load walk visits this root module on every load
        path, after the drafter is wired, so a selected feature is installed
        here, or refused (``ValueError``; a DUMMY load never loads the
        drafter's DFlash modules, so a drafter feature refuses there), instead
        of silently missing. The installs are idempotent: after
        ``load_draft_weights`` installed them this changes nothing and logs
        nothing. Under ``NO_PATHS`` this does nothing.
        """
        self._install_draft_features()

    def cache_derived_state(self) -> None:
        """``post_load_weights`` for the loader paths whose walk calls only this stage."""
        self._install_draft_features()

    def _install_draft_features(self) -> None:
        """The drafter fast paths of the plan, in install order (``drafter.adopt_drafter``).

        ``drafter_block_conv`` / ``drafter_block_decode``, ``draft_head``, then
        ``drafter_fc_shard`` / ``fused_accept`` / ``drafter_fixed_gemms``
        (``spec_worker.install``), then ``context_kv_fork``, then
        ``verify_fold`` / ``path_walk_tail`` (deriving from the class
        ``draft_head``, ``fused_accept`` and ``context_kv_fork`` left, whose
        kernels it extends; its drafter-input prep runs the context-K/V fork's
        side chain on a folded step).

        From the plan decided when the shell was built. Each install is
        idempotent, logs only what it changed, raises ``ValueError`` for a
        fast path of the plan it cannot install, and imports its module only
        when the plan has it.
        ``fused_accept`` goes after ``draft_head``: it re-classes the worker as
        a class derived from ``draft_head``'s, and ``draft_head`` refuses a
        worker that is not the built-in class (on a repeated call it sees its
        own class through ``isinstance`` and changes nothing).
        ``verify_fold`` re-classes the worker after ``spec_worker.install`` (its
        class derives from ``fused_accept``'s and ``draft_head``'s).
        ``drafter_hops`` goes after the block installs and ``spec_worker.install``:
        its hops run inside the drafter's block decode.
        ``drafter_pdl_chain`` goes after it: its entry triggers are checked
        against the ``drafter_fixed_gemms`` GEMM methods and the hops.
        The head paths (``split_sampler``, ``lm_head_inbox``) go last: they
        re-class the worker as a class derived from the one the others left, and
        with the context-K/V fork's head fork point the head fires it itself.
        """
        from . import drafter as _drafter

        _drafter.adopt_drafter(self, self._plan)
