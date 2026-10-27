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
  generation rows beside them -- and runs the built-in text-model forward, or,
  with ``context_sequence_parallel``, its sequence-parallel arm on large steps.
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
full-length or paged drafter context the fast paths written for a windowed arena
are left out of the plan (``dispatch.WINDOWED_CONTEXT_ARENA_PATHS``); over a
windowed arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``) the paths that write the
arena by column (``dispatch.FULL_CONTEXT_ARENA_PATHS``: the context-K/V fork and
the verify fold) stay where the composition serves the window compaction too
(sm103 tep4) or on sm107 (``dispatch.WINDOWED_COMMIT_SMS``: their commit compacts
in its own kernel), and are left out otherwise. The installers
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
* ``multi_request_split_kv`` re-classes every layer's attention with
  ``MultiRequestSplitKvAttention`` in front of all of the above: on decode steps
  of two or more sequences (which none of the 8-row paths above take) the
  built-in decode FMHA call launches ``ops/attention/multi_request_split_kv``
  -- the batch's pages split evenly over one CTA per SM, then a combine -- into
  the call's output; steps of one sequence never reach it.
* ``multi_request_prep_append`` re-classes every layer's attention with
  ``MultiRequestPrepAppendAttention`` in front of all of the above: on decode
  steps of two or more sequences (which none of the one-sequence paths above
  take) the prep runs ``ops/attention/multi_request_prep_append``, which also
  writes every sequence's K/V append, the E4M3 query, the FMHA scale words and
  the cumulative query lengths, and the built-in generation preprocess launch
  (decoder-info + append kernels) stands down; steps of one sequence never
  reach it. The decode FMHA that follows is ``multi_request_split_kv``'s where
  that path serves the layer.
* With any of the attention paths above other than ``preattn_prep``
  (``attention.TRTLLM_GEN_DECODE_PATHS``), every layer's attention selects
  flashinfer's trtllm-gen FMHA for its decode steps
  (``attention.select_decode_fmha``): those paths read its KV-pool page bound,
  and the full-attention ones intercept its decode call. On sm_103 the public
  FMHA library list already holds it. On sm_107 the public list leaves it out,
  so a ``DecodeTrtllmGenFmha`` goes in front of each layer's FMHA libraries:
  decode steps run it, while prefill and mixed steps keep the built-in
  selection (the ``thop`` attention). The drafter's attention is not a target
  layer and keeps the built-in selection.
* ``decode_kv_splits`` (tep4 on sm_103 only) re-classes nothing: ``DecodeTarget``
  arms it for a forward of 72..512 rows (9 to 64 requests), and while armed the
  full-attention layers' trtllm-gen decode FMHA calls present flashinfer's
  KV-split selection with the SM count of a grid of about
  ``TARGET_CTAS_PER_SM`` CTAs per SM (``kernels/decode_kv_splits``), so those
  steps keep multi-CTA KV at every batch size. Decode steps of up to 8
  requests and prefill steps run what they ran before. Where
  ``multi_request_split_kv`` serves a full-attention layer, its stand-in sits
  inside this one and launches its own kernel, so this split only acts on the
  calls that path declines.

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
  ``boundary_early_release`` converted the next layer's QKV weight: a wrapper
  around the experts' forward (installed with the expert paths once the weights
  are final) forks its L2 prefetch right after the expert launch, and
  ``DecodeTarget`` forks layer 0's before the embedding. No module is
  re-classed.
* ``batch_dense_gemms`` (tep4 on sm_103 only) re-classes nothing: ``DecodeTarget``
  arms it for a decode forward of 33..512 rows, where the built-in ``Linear``
  no longer asks a low-M dispatcher. While armed, the ``dense_gemms`` projections'
  ``apply_linear`` hands it each call with the built-in GEMM: the QKV site runs
  flashinfer's TGV GEMM at the row counts of its table (same bits as the built-in
  cuBLAS call there); up to 64 rows a sliding-window layer's multi-request decode
  prep issues the L2 prefetch of the layer's o_proj weight; and the
  ``boundary_early_release`` norm-gate runs the PDL copy of the kernel the built-in
  would launch up to 64 and above 128 rows (``kernels/batch_dense_gemms``).

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
* ``multi_request_post_attention_hop`` (sm103 tep4 only) arms the same joins on decode
  steps of 9..1024 rows (two to 128 requests at K = 7) with ``ops/allreduce/post_attn_hop_wide``:
  the o_proj partial skips the built-in MNNVL all-reduce and one kernel reduces it, joins
  it into the residual and applies the pre-FFN norm on the cluster boundary's 9..32-row
  split, on its own transport sized for 512 rows (one-shot below ``TWO_SHOT_ROWS`` rows,
  two-shot with a per-row owner from there, unicast publication) -- a step of 513..1024 rows
  (a generation worker's) on a second, two-shot transport of 1024 rows built on the first such
  step -- launched after the
  o_proj GEMM rather than as its programmatic dependent. Bitwise the built-in
  pair at 9..32 rows; above, each row is what the built-in pair computes for it at 9..32
  rows (the built-in Triton join sums the variances in another order).
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
* ``wide_projection_gathers`` (tep4 on sm_103 only) widens ``projection_gathers`` to
  decode steps of 9..32 rows (two to four requests at K = 7), which ``DecodeTarget``
  arms (``WideProjections.armed``): the pool and unpool projections return the rank's
  slice, the post-FFN boundary and the pre-expert quantization gather it on the path's
  32-row inboxes, and the dense GEMM plan runs its 9..32-row table, so the next QKV is
  flashinfer's split-K, which the boundary's early release lets stream its weight.
  Decode steps of 1..8 rows keep the 8-row inboxes and kernels.
* ``projection_publish`` (tep4 on sm_103 only) takes the deferred pool and unpool
  projections of decode steps of 33..512 rows (5 to 64 requests at K = 7), which
  ``DecodeTarget`` arms (``PublishedProjections.armed``): each rank multiplies its column
  slice with the slot of the site's multicast transport as the GEMM's output, and the
  pre-expert NVFP4 quantization and the post-FFN join poll the published rows (the built-in
  kernels' arithmetic on the gathered rows).

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
* ``router_gemm`` re-classes every MoE layer once more, with ``RouterGemmMoE`` in
  front of the class the paths above left it with, and ``DecodeTarget`` arms it on
  decode steps of 9..512 rows (two or more requests, which the 8-row prologue does
  not take): there the layers' router logits run ``ops/moe/router_gemm``, a BF16x3
  split-K tensor-core GEMM, in place of the built-in TF32x3 router kernel (16..128
  rows) and the TF32 ``Linear`` (above 128 rows). Steps with context rows and
  decode steps of at most 8 or more than 512 rows keep the class behind it. Not
  bitwise the built-in logits: its error is the FP32 accumulation's alone.
* ``wide_moe_allreduce`` (tep4 on sm_103 only) puts a ``WideMoeAllReduce`` in
  every MoE layer's ``all_reduce``: ``DecodeTarget`` arms them for a decode
  forward of 129..1024 rows (above the fused finalize's 128), whose unfused MoE
  output is then all-reduced on the MNNVL two-shot instead of the built-in NCCL
  ring (``kernels/wide_moe_allreduce``); every other call, and every call of a
  prefill or mixed step, runs the built-in NCCL module.
* ``finalize_two_shot`` (tep4 on sm_103 only) puts a ``FinalizeTwoShotWorkspace`` in
  front of the finalize transport in every MoE layer's fused finalize attribute:
  ``DecodeTarget`` arms it for a decode forward of 72..512 rows, whose expert finalize
  and all-reduce then run as one kernel on an owner-per-row two-shot transport
  (``ops/moe/finalize_two_shot``) -- bitwise the built-in fused kernel's result -- in
  place of the one-shot fused kernel (64..128 rows) or the unfused combine, BF16 cast
  and ``wide_moe_allreduce`` (129..512 rows). Every other call is the wrapped
  workspace's or the unfused path's, as before.

Context steps:

* ``context_sequence_parallel`` (tep4 on one tray only, sm_103 and sm_107; row
  thresholds per SM in ``kernels/context_sequence_parallel``) has ``PrefillTarget`` run a
  step with context rows through the built-in decoder layers' sequence-parallel arm
  when it is captured into the breakable prefill CUDA graph with at least
  ``kernels/context_sequence_parallel``'s ``MIN_ROWS`` rows, or runs eagerly with at
  least its ``EAGER_MIN_ROWS_BY_SM`` rows for the SM: each
  rank keeps a quarter of the rows for the residual stream, the norms, the router,
  the pool and unpool projections and the pre-expert norm and NVFP4 quantization,
  while attention and the experts see every row (all-gathers in front of them,
  reduce-scatters of their partial sums). The DFlash capture of those steps is
  gathered once, after the last layer. ``PrefillTarget``'s constructor re-classes
  every layer's MoE once more, last, with a mixin acting on sequence-parallel calls
  only (``moe.ContextSequenceParallelMoE``: it gathers the quantized expert input);
  smaller steps and decode steps run what they ran before.
* ``context_prep`` (on ``preattn_prep``'s attention class) has ``PrefillTarget`` hand
  every layer call ``context_prep``, the ``preattn_prep`` op, which the adopted
  attention runs in place of the built-in pre-attention prep on every row of a step
  with context rows (the gate and query split copies, the two per-head norms, the
  query pre-scale, the RoPE and the concat; the same bits), sequence-parallel or not.
  ``layers.adopt_model`` also has each adopted attention zero the rows of its output
  that a replayed prefill CUDA graph pads the step with (``attention.zero_padded_rows``),
  so what those discarded rows load onto the experts does not follow the captured
  body's allocations.

The expert paths change the expert layers once their weights are final and are
installed by the shell's ``load_weights`` after the built-in loader
(``install_loaded_features``):

* ``replicated_experts`` replicates each rank's ring predecessor's experts and
  patches every layer's experts module and routing method, whatever class the
  MoE module itself has.
* ``persistent_experts`` routes the NVFP4 layers' stacked calls of at most 8
  rows to the persistent expert kernel. Without ``replicated_experts`` (tp 4)
  the stacked calls run over the rank's own expert block (the owner-only window,
  ``ring_buddy.install(windows=1)``), with nothing replicated.
* ``three_window_placement`` stacks a third ring window on them.
* ``half_window_experts`` (tp 4) stacks the ring predecessor's upper expert half on
  them, which the prologue's routed kernel places onto; only up to the
  ``max_batch_size`` of its capacity gate (``dispatch.CAPACITY_GATES``).
* ``bf16_routed_experts`` gives the two BF16 layers' stacked calls unpacked
  routing (on the owner-only window as well).
* ``hot_replicas`` (tep4 on sm_103 only) hooks the NVFP4 layers' expert and routing
  calls after the owner-window install; the copies themselves are gathered on the
  first forward of a deployment whose largest decode step (``max_batch_size`` times
  the verify width) reaches 33 rows (``HotReplicas.decide``), and ``DecodeTarget``
  arms them on decode steps of 33..512 rows: every touched hot expert is placed on
  its owner or its holder and the expert call runs over the ``64 + R`` window.
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

* ``drafter_block_conv`` and ``drafter_block_decode`` on the drafter's modules,
  then ``wide_drafter_window_attention`` (tep4 on sm_103 only) on the block decode
  ``drafter_block_decode`` bound: its block decodes of 33..1024 rows call the
  multi-request window attention;
* ``draft_head`` on the DFlash worker and the drafter's logits processor, then
  ``wide_draft_topk`` (tep4 on sm_103 only) on the draft head's key all-reduce: draft
  steps of 33 or more rows exchange each rank's own top-k keys;
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
* ``drafter_pdl_chain``;
* ``drafter_window_compact`` last, on the worker the others left (and after
  the head paths below): its ``_compact_ctx_window`` -- the windowed drafter
  arena's per-step left shift of each generation slot's window -- runs
  ``ops/spec/ctx_window_compact`` in place, with the built-in's arena bytes.

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
context-K/V fork's head fork point right after that GEMM); ``wide_split_sampler``
(tep4 on sm_103 only) widens the head to decode steps of 129..1024 rows, which the
sampler takes on the shipped sampler's 512-thread block (``target_head.widen``).
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    resolved_max_batch_size,
    resolved_mode,
)
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
    _router_gemm: Optional[object]
    _decode_kv_splits: Optional[object]
    _wide_moe_allreduce: Optional[object]
    _finalize_two_shot: Optional[object]
    _wide_hop: Optional[object]
    _hot_replicas: Optional[object] = None

    @classmethod
    def adopt(
        cls,
        model: MuseSparkOpenModel,
        plan: "_dispatch.Plan" = _dispatch.NO_PATHS,
        compute_capability: Optional[Tuple[int, int]] = None,
    ) -> "MuseSparkOpenTepModelingV2":
        """Re-class the built-in text model as this core and build its two targets.

        ``plan`` is the model's composition (``dispatch.compose``); with
        ``NO_PATHS`` the core runs the built-in modules only. ``compute_capability``
        is the architecture the plan was composed for (``layers.adopt_model``).

        Raises:
            TypeError: if ``model`` is not exactly the built-in ``MuseSparkOpenModel``.
            ValueError: if a fast path of the plan cannot run on this model.
        """
        if type(model) is not MuseSparkOpenModel:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenModel, got {type(model).__name__}"
            )
        adopted = _layers.adopt_model(model, plan, compute_capability)
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
        model._router_gemm = adopted.router_gemm
        model._decode_kv_splits = adopted.decode_kv_splits
        model._wide_moe_allreduce = adopted.wide_moe_allreduce
        model._finalize_two_shot = adopted.finalize_two_shot
        model._wide_hop = adopted.wide_hop
        model._wide_projections = adopted.wide_projections
        model._published_projections = adopted.published_projections
        model._batch_dense_gemms = adopted.batch_dense_gemms
        model._prefill = PrefillTarget(model, _dispatch.sm_of(compute_capability))
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
        hot = self._hot_replicas
        if hot is not None and not hot.decided:
            # The first forward decides whether this deployment allocates the hot
            # expert copies (kernels/replicated_experts/hot_replicas.py).
            hot.decide(attn_metadata, kwargs.get("spec_metadata"))
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

    Runs the built-in text-model forward. With ``context_sequence_parallel``, a
    step that path takes (``ContextSequenceParallel.takes``: context rows, and enough
    rows for a graph-captured or an eager step) runs the built-in decoder layers'
    sequence-parallel arm instead, with the DFlash capture gathered after the last
    layer; the first step grows the MNNVL all-reduce workspaces for the steps the path
    declines (``ContextSequenceParallel.reserve``). With ``context_prep`` every layer
    call also carries ``context_prep``, which the adopted attention runs in place of
    its built-in pre-attention prep.
    """

    #: The op every layer call carries as ``context_prep``; None without the path.
    context_prep: Optional[PreattnPrep] = None

    def __init__(self, core: MuseSparkOpenTepModelingV2, sm: Optional[int] = None) -> None:
        self.core = core
        if core._plan.enabled("context_prep"):
            self.context_prep = preattn_prep
        self.sequence_parallel = None
        if core._plan.enabled("context_sequence_parallel"):
            from .kernels.context_sequence_parallel import forward as _context_sp

            self.sequence_parallel = _context_sp.install(core, sm)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        sequence_parallel = self.sequence_parallel
        if sequence_parallel is not None:
            sequence_parallel.reserve(self.core)
        if self.context_prep is not None:
            kwargs = dict(kwargs, context_prep=self.context_prep)
        source = input_ids if inputs_embeds is None else inputs_embeds
        if (
            sequence_parallel is not None
            and source is not None
            and sequence_parallel.takes(attn_metadata, int(source.shape[0]), kwargs)
        ):
            hidden_states = sequence_parallel.forward(
                self.core,
                attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        else:
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

    With ``multi_request_post_attention_hop``, a step of 9..1024 rows arms the same joins with
    the wide hop (``WideHopCall.armed``) and every layer call carries the same
    ``all_reduce_params``: the joins run ``post_attn_hop_wide`` in place of the
    built-in o_proj all-reduce and residual join.

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

    With ``router_gemm``, a step of 9..512 rows arms the router GEMM
    (``RouterGemmCall.armed``): every MoE layer's router logits run
    ``ops/moe/router_gemm``.

    With ``decode_kv_splits``, a forward of 72..512 rows (9 to 64 requests) is armed
    (``DecodeKvSplits.armed``): the layers' trtllm-gen decode FMHA calls run over the
    path's over-subscribed KV split instead of flashinfer's own.

    With ``wide_moe_allreduce``, a forward of 129..1024 rows (17 to 128 requests) is
    armed (``WideMoeAllReduces.armed``): every layer's unfused MoE output is
    all-reduced on the MNNVL two-shot instead of the built-in NCCL module.

    With ``wide_projection_gathers``, a forward of 9..32 rows (two to four requests) is
    armed (``WideProjections.armed``): the layers' pool and unpool projections return the
    rank's slice, their consumers gather it on the path's 32-row inboxes, and the dense
    GEMM plan runs its 9..32-row table (the QKV projection on split-K).

    With ``projection_publish``, a forward of 33..448 rows (5 to 56 requests) is armed
    (``PublishedProjections.armed``): the layers' pool and unpool slice GEMMs write into the
    sites' multicast transports and their consumers poll the published rows.

    With ``finalize_two_shot``, a forward of 72..512 rows (9 to 64 requests) is armed
    (``FinalizeTwoShotSwitch.armed``): every layer's expert finalize and all-reduce run
    as one two-shot kernel, in place of the one-shot fused kernel (64..128 rows) or the
    unfused combine, cast and all-reduce (129..512 rows).

    With ``hot_replicas`` and its copies allocated, a forward of 33..512 rows (5 to 64
    requests) is armed (``HotReplicas.armed``): every NVFP4 layer places the step's
    touched hot experts on their owners or their holders and runs its experts over
    the ``64 + R`` window.
    With ``batch_dense_gemms``, a forward of 33..512 rows (5 to 64 requests) is armed
    (``BatchDenseGemms.armed``): the QKV projections run the TGV GEMM at its table's
    rows, the sliding-window layers' multi-request preps issue the L2 prefetch of
    their o_proj weight up to 64 rows, and the attention norm-gates run as PDL
    kernels up to 64 and above 128 rows.
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
        self.router_gemm = getattr(core, "_router_gemm", None)
        self.kv_splits = getattr(core, "_decode_kv_splits", None)
        self.wide_moe_allreduce = getattr(core, "_wide_moe_allreduce", None)
        self.finalize_two_shot = getattr(core, "_finalize_two_shot", None)
        self.wide_hop = getattr(core, "_wide_hop", None)
        self.wide_projections = getattr(core, "_wide_projections", None)
        self.published_projections = getattr(core, "_published_projections", None)
        self.batch_dense_gemms = getattr(core, "_batch_dense_gemms", None)

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
        wide_hop = self.wide_hop
        wide_hop_armed = (
            not hop_armed and wide_hop is not None and wide_hop.takes(hidden_states.shape[0])
        )
        if hop_armed or wide_hop_armed:
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
        router = self.router_gemm
        router_armed = router is not None and router.takes(rows)
        kv_splits = self.kv_splits
        kv_splits_armed = kv_splits is not None and kv_splits.takes(rows)
        wide_ar = self.wide_moe_allreduce
        wide_ar_armed = wide_ar is not None and wide_ar.takes(rows)
        wide_proj = self.wide_projections
        wide_proj_armed = wide_proj is not None and wide_proj.takes(rows)
        published = self.published_projections
        published_armed = published is not None and published.takes(rows)
        two_shot = self.finalize_two_shot
        two_shot_armed = two_shot is not None and two_shot.takes(rows)
        # Installed at weight load, after this target was built.
        hot = getattr(core, "_hot_replicas", None)
        hot_armed = hot is not None and hot.takes(rows)
        batch = self.batch_dense_gemms
        batch_armed = batch is not None and batch.takes(rows)

        layers = core.layers
        precomputed_attn_input = None
        if hop_armed:
            arming = hop.armed(hidden_states.shape[0], hidden_states)
        elif wide_hop_armed:
            arming = wide_hop.armed(hidden_states.shape[0])
        else:
            arming = contextlib.nullcontext()
        with (
            arming,
            direct.armed(rows) if direct_armed else contextlib.nullcontext(),
            pool.armed(rows) if pool_armed else contextlib.nullcontext(),
            router.armed() if router_armed else contextlib.nullcontext(),
            kv_splits.armed() if kv_splits_armed else contextlib.nullcontext(),
            wide_ar.armed() if wide_ar_armed else contextlib.nullcontext(),
            wide_proj.armed() if wide_proj_armed else contextlib.nullcontext(),
            published.armed(rows) if published_armed else contextlib.nullcontext(),
            two_shot.armed() if two_shot_armed else contextlib.nullcontext(),
            hot.armed() if hot_armed else contextlib.nullcontext(),
            batch.armed() if batch_armed else contextlib.nullcontext(),
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


#: The fast paths written for one tray of four only: those of decode steps of two to
#: four requests (the wide (16..32-row) TP4 inboxes, the wide post-attention hop
#: transport and the multi-request sliding-window block), and the half expert window
#: (a 64-expert block's upper half replicated on the ring successor), and the TP4
#: gather of the sharded drafter fc over the tray's NVLink.
_TEP4_ONLY_PATHS = frozenset(
    {
        "half_window_experts",
        "drafter_fc_inbox",
        "wide_prologue",
        "wide_post_attention_hop",
        "wide_swa_block",
        "wide_persistent_experts",
        "wide_full_attention",
        "wide_dense_gemms",
    }
)
#: The fast paths validated at a tep4 rank's shape on sm103, which no TEP8 composition
#: serves (sm107 TEP4 adds ``_SM107_TEP4_CONTEXT_PATHS`` of them): the multi-request
#: decode prep append, the windowed drafter arena compaction, the router GEMM of
#: multi-request calls, the multi-request decode FMHA, the full-attention
#: decode FMHA over an over-subscribed KV split, the MoE output all-reduce of 129..1024-row
#: decode steps on the MNNVL two-shot, the post-attention hop of 9..1024-row decode steps
#: on its own transports, the projection gathers and the split-K QKV of 9..32-row decode
#: steps on 32-row inboxes, the pool / unpool slices of 33..448-row decode steps published
#: from their GEMMs, large context steps sequence-parallel over the tray, the expert
#: finalize + two-shot all-reduce of 72..512-row decode steps and the hot NVFP4 expert
#: copies of 33..512-row decode steps, the attention block's QKV TGV GEMM, o_proj L2
#: prefetch and PDL norm-gate of 33..512-row decode steps, the drafter window attention of
#: 33..1024-row block decodes on a multi-request split geometry, the draft head's rank top-k
#: key exchange on draft steps of 33 or more rows, the split verify sampler on
#: 129..1024-row decode steps, and the context steps' pre-attention prep in one launch.
_SM103_TEP4_ONLY_PATHS = frozenset(
    {
        "hot_replicas",
        "multi_request_prep_append",
        "drafter_window_compact",
        "router_gemm",
        "multi_request_split_kv",
        "decode_kv_splits",
        "wide_moe_allreduce",
        "multi_request_post_attention_hop",
        "wide_projection_gathers",
        "projection_publish",
        "context_sequence_parallel",
        "finalize_two_shot",
        "batch_dense_gemms",
        "wide_drafter_window_attention",
        "wide_draft_topk",
        "wide_split_sampler",
        "context_prep",
    }
)
#: TEP8: every fast path but the TP4-only ones of either architecture.
_TEP8_PATHS = _dispatch.FAST_PATH_NAMES - _TEP4_ONLY_PATHS - _SM103_TEP4_ONLY_PATHS
#: The fast paths written for any rank count whose output is bitwise the built-in
#: model's.
_TEP4_BITWISE_PATHS = frozenset(
    {
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
#: TEP4 on either architecture: the bitwise paths and ``drafter_fc_shard``, whose
#: sharded drafter projection can move a draft token's bf16 last place (the target
#: still verifies every draft; it shards every row count at TP4).
_TEP4_PATHS = _TEP4_BITWISE_PATHS | frozenset({"drafter_fc_shard"})
#: TEP4 on sm103 adds the attention paths validated at a tep4 rank's 32 query / 4 KV
#: heads, the dense projections and drafter GEMMs on their tep4 kernel tables, the
#: drafter hops and PDL chain on the tray's MNNVL one-shot, the LM-head inbox on the
#: tray's multicast, the expert paths on the owner-only 64-expert window (no
#: replication), and the sharded-projection gathers, the MoE prologue and the
#: finalize transport on the tray; and the multi-request decode prep append, the
#: windowed drafter arena compaction, the router GEMM of multi-request calls, the
#: multi-request decode FMHA; the full-attention decode FMHA of 72..512-row decode
#: steps over an over-subscribed KV split; the MoE output all-reduce of 129..1024-row
#: decode steps on the tray's MNNVL two-shot; on decode steps of 9..1024 rows, the
#: post-attention hop on its own one-shot transport; large context steps
#: sequence-parallel over the tray; and the context steps' pre-attention prep in one launch.
_SM103_TEP4_PATHS = (
    _TEP4_PATHS
    | _SM103_TEP4_ONLY_PATHS
    | frozenset(
        {
            "swa_paged_decode",
            "swa_block",
            "full_attention_split_kv",
            "full_attention_prep_fold",
            "drafter_block_decode",
            "dense_gemms",
            "drafter_fixed_gemms",
            "drafter_hops",
            "drafter_pdl_chain",
            "lm_head_inbox",
            "persistent_experts",
            "bf16_routed_experts",
            "projection_gathers",
            "boundary_early_release",
            "capture_fold",
            "moe_prologue",
            "finalize_transport",
        }
    )
)
#: The sm103 TEP4 communication paths of decode steps above 32 rows that sm107 TEP4 also
#: serves: the MoE output all-reduce of 129..1024-row steps on the MNNVL two-shot, the
#: post-attention hop of 33..1024-row steps on its own transports (the 8-row hop and its
#: wide TP4 transport keep 1..32 rows), the expert finalize + two-shot all-reduce of
#: 96..512-row steps (``finalize_two_shot.install.MIN_ROWS_BY_SM``) and the pool / unpool
#: slices of 33..448-row steps published from their GEMMs.
_SM107_TEP4_COMM_PATHS = frozenset(
    {
        "wide_moe_allreduce",
        "multi_request_post_attention_hop",
        "finalize_two_shot",
        "projection_publish",
    }
)
#: TEP4 on sm107 adds the attention paths at a tep4 rank's 32 query / 4 KV heads (the
#: split-KV FMHA at half the device's SMs per pair of KV heads), the dense projections
#: and drafter GEMMs on their sm107 tep4 kernel tables, the drafter hops and PDL chain
#: on the tray's MNNVL one-shot, the expert paths on the owner-only 64-expert window
#: (eight weight stages there, ``persistent_experts.OPTIONS_BY_SM``), and the
#: sharded-projection gathers, the MoE prologue and the finalize transport on the tray;
#: the sharded projections and the TP4 local finalize they build on are the built-in
#: model's fused kernels, opted in on sm107 (``SM107_FUSED_KERNELS``). On top of those it
#: serves the single-request decode paths of TEP8 that need no ring replication and no
#: eight-rank kernel: the L2 prefetches of the dense projections' weights (the next
#: layer's QKV from the expert call), the expert kernel starting from the routed
#: kernel's early ids, the attention output and unpool projections writing straight
#: into the hop's and the post-FFN boundary's transports, and the pool projection and
#: router shard writing straight into the routed kernel's transport
#: (``pool_into_route``, on the tray's TP4 layout). Its NVFP4 expert layers also
#: hold their ring predecessor's upper expert half (``half_window_experts``), so the
#: routed kernel balances the selected experts of a step of at most 8 rows over the
#: four ranks; that window costs 45 GiB of KV cache per rank, so a plan carries it
#: only up to ``max_batch_size`` 8 (``dispatch.CAPACITY_GATES``). The sharded drafter fc
#: of up to 128 rows is gathered over the tray's NVLink (``drafter_fc_inbox``). Decode
#: steps of two to four
#: requests (16..32 rows) also take the TP4 sharded MoE prologue and projection gathers
#: on wide inboxes (``wide_prologue``), the post-attention hop on a wide transport
#: (``wide_post_attention_hop``) and the sliding-window attention block of one call for
#: all their sequences (``wide_swa_block``); two-request steps run the persistent expert
#: kernel's 16-column build (``wide_persistent_experts``); the full-attention layers run
#: the folded prep, split-KV partials and merge of all their sequences in one call each
#: (``wide_full_attention``); QKV and the pooler / unpooler slices of 9..32-row steps run
#: flashinfer's split-K (``wide_dense_gemms``).
_SM107_TEP4_PATHS = _TEP4_PATHS | frozenset(
    {
        "dense_gemm_prefetch",
        "next_qkv_prefetch",
        "early_expert_ids",
        "attention_output_into_hop",
        "unpool_into_boundary",
        "pool_into_route",
        "half_window_experts",
        "drafter_fc_inbox",
        "wide_prologue",
        "wide_post_attention_hop",
        "wide_swa_block",
        "wide_persistent_experts",
        "wide_full_attention",
        "dense_gemms",
        "wide_dense_gemms",
        "drafter_fixed_gemms",
        "drafter_hops",
        "drafter_pdl_chain",
        "swa_paged_decode",
        "swa_block",
        "full_attention_split_kv",
        "full_attention_prep_fold",
        "drafter_block_decode",
        "persistent_experts",
        "bf16_routed_experts",
        "projection_gathers",
        "boundary_early_release",
        "capture_fold",
        "moe_prologue",
        "finalize_transport",
    }
)
#: The multi-request attention paths of the sm103 tep4 set that sm107 tep4 serves too: the
#: decode FMHA of steps of five or more requests (``multi_request_split_kv``; steps of two to
#: four requests stay on ``wide_full_attention``).
_SM107_TEP4_ATTENTION_PATHS = frozenset({"multi_request_split_kv"})
_SM107_TEP4_PATHS = _SM107_TEP4_PATHS | _SM107_TEP4_ATTENTION_PATHS
_SM107_TEP4_PATHS |= _SM107_TEP4_COMM_PATHS
#: The GB300 TEP4 GEMM and expert-balance paths also served at sm107 TEP4, tuned for
#: R200: the router GEMM of decode steps of five or more requests (33..512 rows; 16..32
#: rows stay on ``wide_prologue``'s router shard), with one CTA per SM of the device;
#: the attention block's QKV TGV GEMM on the rows of the R200 table and the PDL
#: norm-gates of 33..512-row decode steps (``batch_dense_gemms``); and the hot NVFP4
#: expert copies of 33..512-row decode steps, which yield to ``half_window_experts``
#: wherever its capacity gate keeps it (``dispatch.YIELDS``).
_SM107_TEP4_GEMM_EXPERT_PATHS = frozenset({"router_gemm", "batch_dense_gemms", "hot_replicas"})
_SM107_TEP4_PATHS = _SM107_TEP4_PATHS | _SM107_TEP4_GEMM_EXPERT_PATHS
#: The sm103 tep4 decode-prep and drafter-at-batch paths sm107 TEP4 serves too, above
#: the two-to-four-request band its own paths hold: the multi-request decode prep append
#: on steps of five or more requests (``wide_full_attention`` and ``wide_swa_block`` fold
#: the two-to-four-request steps' append themselves), the drafter window attention of
#: 33..1024-row block decodes on the multi-request split geometry (sized by the device's
#: SM count), the draft head's rank top-k key exchange on draft steps of 33 or more rows,
#: and the split verify sampler on 129..1024-row decode steps.
_SM107_TEP4_DRAFTER_AT_BATCH_PATHS = frozenset(
    {
        "multi_request_prep_append",
        "wide_drafter_window_attention",
        "wide_draft_topk",
        "wide_split_sampler",
    }
)
_SM107_TEP4_PATHS = _SM107_TEP4_PATHS | _SM107_TEP4_DRAFTER_AT_BATCH_PATHS
#: The context-step paths sm107 TEP4 serves from the sm103 TEP4-only set: large context
#: steps sequence-parallel over the tray and the context steps' pre-attention prep in one
#: launch. Neither acts on a decode step, so they share no decode-row band with the
#: other sm107 TEP4 paths.
_SM107_TEP4_CONTEXT_PATHS = frozenset({"context_sequence_parallel", "context_prep"})

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
        served=_SM103_TEP4_PATHS,
        topology=_dispatch.ONE_TRAY_OF_4,
        preconditions=(_dispatch.NO_PREFILL_SEQUENCE_PARALLEL,),
    ),
    (107, 4): _dispatch.TargetPaths(
        served=_SM107_TEP4_PATHS | _SM107_TEP4_CONTEXT_PATHS,
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
            max_batch_size=resolved_max_batch_size(type(self)),
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
            MuseSparkOpenTepModelingV2.adopt(self.llm.model, plan, capability)

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
        The head paths (``split_sampler``, ``lm_head_inbox``) go after those: they
        re-class the worker as a class derived from the one the others left, and
        with the context-K/V fork's head fork point the head fires it itself.
        ``drafter_window_compact`` goes last: it re-classes the worker as a class
        derived from the one every other path left, and on a repeated call sees
        its own class through ``isinstance``.
        """
        from . import drafter as _drafter

        _drafter.adopt_drafter(self, self._plan)
