# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: every target-side feature together, on real weights, in the decode serving shape (TP8).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes and the
checkpoint, under the serving switches, e.g.::

    MUSESPARK_REFERENCE_CKPT=<Muse-Spark-Open-816B-A42B-NVFP4 checkpoint> \\
    MUSESPARK_OPEN_ALLREDUCE=MNNVL MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1 \\
    MUSESPARK_OPEN_SHARDED_PROJECTIONS=1 TRTLLM_LOW_M_GEMM_BACKEND=auto TRTLLM_ENABLE_PDL=1 \\
    srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>

Anywhere else every test skips.

The text model is built over the checkpoint's layers 0, 1, 2, 59, 60 and 61 --
both BF16-expert layers (0 and 61), both NoPE full-attention ends (1 and 61),
four 2048-token sliding-window layers and the real last layer -- loaded with
those layers' real weights, the target LM head's vocab shard beside it. The KV
cache is the serving one: FP8, ``KVCacheManagerV2``, 32-token pages. Each run is
the decode serving shape: one request, prefilled with a prompt longer than the
sliding window (the built-in path in every arm), then decode steps of that one
request with M rows -- its last token and M - 1 drafts, M = 1..8 -- and the
fallback widths 9 and 16, fed real embedding rows, once eagerly and once
replayed from a CUDA graph captured per arm, with the metadata updated in
place before every replay.

One copy of the model per arm. An arm is a row ``(arm, ids, reference arm)`` of the
selected table (``ARMS``). Each arm either must equal its reference arm bit for bit --
every output of every step (prefill, final, both outputs of every layer), the greedy
token of every row, and the KV cache bytes of every layer -- or, without a reference,
is measured against the built-in model: the bit-identical features (``preattn_prep``,
``hop``, ``gathers``, ...) must give the built-in bits alone and together, and must not
change the bits of a non-bit-identical feature they are combined with. The
non-bit-identical features (``ring_buddy``, ``paged_decode``, ``paged_decode_q8``,
``dense_gemms``, ``gemm_prefetch``) are measured against the built-in model alone and
together: their combination ``n_all`` (with the bit-identical ``preattn_prep,hop,gathers``
it is the full target set ``full``) must stay within the sum of their distances alone
from the built-in output (no amplification, ``MARGIN``), every output must be finite,
the combination must be deterministic (``DETERMINISM``: run twice, the same bits), and
``gemm_prefetch`` must equal ``dense_gemms``.
The prefill's final output and the cache it wrote must equal the built-in model's in
every arm: at 2100 + 3M rows no feature acts on it (``dense_gemms`` takes calls of at most
8 rows and ``ring_buddy`` of at most 16, prefill steps included; the short prefill chunks
they do act on are in ``test_modeling_v2_musespark_open_schedule.py``, with the draft head
and ``fc_shard`` in the graph, interleaved graphs and delayed ranks). Its per-layer outputs
are not kept.
Each feature's entry point is counted on every decode forward (on the capture
in graph mode) against the number of call sites it must take at that width
(``ENTRIES``, ``_expected_counts``), so a feature that silently fell back is caught too.
``gemm_prefetch``'s prefetch forks must all be joined inside the forward that made them
(no fork dropped, none pending, every capture succeeds).

``MODELING_V2_COMPOSE_OUT=<dir>`` also writes rank 0's per-arm, per-width result
table there as JSON. ``MODELING_V2_COMPOSE_TIER`` selects the table (default ``tier1``):

* ``tier1`` (``TIER1_ARMS``): the bit-identical and non-bit-identical tier-1 features, alone
  and together, as above.
* ``early_release`` (``EARLY_RELEASE_ARMS``): ``early_release`` with ``gathers`` (whose
  fused unpool boundary's trigger it moves) must give the built-in bits.
* ``tier2`` (``TIER2_ARMS``): ``TIER2_NEW`` (``early_release``, ``persistent_experts``,
  ``stale_poll``) is bit-identical: ``ring_buddy`` + ``persistent_experts`` equals
  ``ring_buddy`` and the tier-2 target set equals the tier-1 one; ``persistent_experts``
  runs on the four NVFP4 layers wherever it acts, and each stale-poll site is installed
  where selected.
* ``tier3`` (``TIER3_ARMS``): ``finalize_transport`` (the finalize on the hop transport) and
  ``moe_prologue`` (the one-chain MoE prologue) are bit-identical: the ``ring_buddy``
  chains with them equal ``ring_buddy`` (``persistent_experts``' slot map read through the
  transport finalize, and fed by the prologue's routing memo and placement), and the
  tier-3 target set equals the tier-2 one. The transport kernel runs on every MoE layer at
  up to 16 rows (counted on its workspace), the prologue's router shard and routed pool op
  on the four NVFP4-expert layers at up to 8 rows, in place of the fused pool gather and
  ``ring_buddy``'s placement kernel (``PROLOGUE_REPLACES``).
* ``swa_block`` (``SWA_BLOCK_ARMS``): the tier-1 target set (``base``) and the tier-2 one
  (``base4``, with ``early_release``'s PDL norm-gate), and the sliding-window block ids on
  top of each, all of which must equal their base bit for bit; ``swa_block_attention``
  replaces ``paged_decode_attention`` at the four sliding-window layers, with
  ``swa_fused_prep`` ``preattn_prep`` runs only at the two full-attention layers, and with
  ``swa_fused_gate`` so does ``early_release``'s norm-gate.
* ``direct_publish`` (``DIRECT_PUBLISH_ARMS``): o_proj publishing into the hop transport at
  8 rows (the consumer hop then runs in place of the hop) must leave the bits of
  ``hop`` + ``dense_gemms``, of the tier-1 target set and of the tier-2 target set
  unchanged.
* ``tier4`` (``TIER4_ARMS``): the tier-4 target paths on this model -- the sliding-window
  block, ``prep_append,prep_append_prefetch_trim`` (the full-attention layers' folded KV
  append and dropped o_proj prefetch) and ``direct_publish`` -- are bit-identical: each
  group on the tier-3 target set equals that set, and so does the tier-4 target set.
  ``capture_fold`` needs a DFlash drafter's capture layers, which this model has none of;
  its identity is its own suites'. The full-attention layers (model layers 1 and 5) of a
  ``prep_append`` arm carry ``FullAppendAttention`` in front of the attention class, run
  the folded prep (``preattn_prep_append``) and skip the built-in preprocess
  (``FoldedPreprocess.outputs``) at up to 8 rows; ``prep_append_prefetch_trim`` drops their
  QKV projections' prefetch.
* ``unpool_direct``, ``unpool_direct_tier3``, ``unpool_direct_tier4``
  (``UNPOOL_DIRECT_ARMS``, ``UNPOOL_DIRECT_TIER3_ARMS``, ``UNPOOL_DIRECT_TIER4_ARMS``): the
  unpool GEMM publishing into the direct boundary transport (the 8-row decode graph) must
  leave the bits of ``gathers,dense_gemms`` and of the tier-2 / tier-3 / tier-4 target set;
  on the 8-row capture the direct boundary takes every fused boundary's call, everywhere
  else the gathers' boundary does. On the tier-4 set o_proj publishes into the hop
  transport and the unpool GEMM into the boundary transport, each site under its own
  producer policy.
* ``qkv_prefetch``, ``qkv_prefetch_tier4`` (``QKV_PREFETCH_ARMS``,
  ``QKV_PREFETCH_TIER4_ARMS``): the next layer's QKV weight L2-prefetched on a named fork
  after each expert call (and layer 0's at the forward's start), joined by that QKV GEMM's
  dispatcher, must leave the bits of the tier-3 / tier-4 target set and of that set with
  ``unpool_direct``. The prefetch runs at the widths whose QKV kernel loads weight k-tiles
  after its wait (split-K, 3..8 rows): once at the start and after each of the five
  converted boundaries' layers' expert calls; in each captured graph there it adds one
  programmatic edge per prefetch (into the QKV GEMM) to the reference arm's prefetch edges,
  and the boundary edges stay five.
* ``early_ids`` (``EARLY_IDS_ARMS``): the early expert ids (``moe_prologue``'s routed kernel
  publishes them, ``persistent_experts``' kernel starts streaming before its wait) are
  bit-identical: the ``moe_prologue`` chain with them equals ``ring_buddy`` and the tier-3
  target set with them equals that set; every such arm's four NVFP4 layers launch the
  expert kernel early (``early_expert_ids.launches_early``) at up to 8 rows.
* ``tier5`` (``TIER5_ARMS``): the bit-identical tier-5 paths (``TIER5_NEW``) on the tier-4
  target set (``full7b_nof``) must equal that set bit for bit, which equals the
  non-bit-identical core. The budgeted paths on it -- ``three_windows`` (the third ring
  window: the expert partials regrouped, as ``ring_buddy`` does), ``split_kv`` (the
  full-attention layers' FMHA as ``full_attention_split_kv`` partials merged in the
  norm-gate) and both (``full7_nof``) -- act on decode steps of up to 8 rows
  (``CHANGES_AT``): there each must equal ``full7b_nof`` upstream of the first layer it
  changes (``UPSTREAM``: layer 0's outputs, the prefill, the KV cache bytes of layers 0 and
  1), every output must be finite, ``full7_nof`` deterministic, and its distance from
  ``full7b_nof`` is recorded; at 9 and 16 rows each must equal ``full7b_nof`` bit for bit.
  ``full7_nof`` is gated against amplification (``COMBINED``). Under ``split_kv`` the two
  full-attention layers run ``full_attention_split_kv`` and ``combine_norm_gate`` (which
  takes their norm-gate) at up to 8 rows, and carry ``FullAttnSplitKvAttention`` in front
  of ``prep_append``'s class.
* ``router_fork``, ``router_fork_tier5`` (``ROUTER_FORK_ARMS``, ``ROUTER_FORK_TIER5_ARMS``):
  ``hop_early_trigger`` and ``router_fork`` (the router shard forked onto a side stream as
  a plain launch, ``router_shard_side`` in place of ``router_shard`` on the four
  NVFP4-expert layers up to 8 rows) are bit-identical: on the tier-4 target set and on
  ``full7b_nof`` (with and without ``three_windows``) each arm equals the set without them
  at every width; every layer of a ``router_fork`` arm holds the side stream.
* ``prep_fold`` (``PREP_FOLD_ARMS``): ``prep_fold`` (the prep-append folded into the
  split-KV kernel, ``full_attention_prep_fold``) is bit-identical, so ``full7_nof`` +
  ``prep_fold`` must equal ``full7_nof`` bit for bit at every width -- every output, the
  greedy tokens and the KV cache bytes of every layer (the step's K/V are appended by the
  attention kernel instead of the prep) -- and ``full7b_split_kv`` + ``prep_fold`` must
  equal ``full7b_split_kv`` likewise. Under ``prep_fold`` the two full-attention layers run
  ``full_attention_prep_fold`` and ``combine_norm_gate`` and neither the prep-append, the
  elided preprocess nor ``full_attention_split_kv`` at up to 8 rows, and carry
  ``FullAttnSplitKvFoldAttention`` in front of ``split_kv``'s class. ``spill_free`` (the
  split-KV kernel, and so the fold build, rescaling O in 16-column chunks: the same values)
  changes no bit either, and its full-attention layers launch the 16-column build
  (``test_spill_free_full_layers_launch_the_spill_free_build``).
* ``pool_direct`` (``POOL_DIRECT_ARMS``): on the 8-row CUDA graphs ``pool_direct`` has the
  pool GEMM and the router shard's publish twin write into the routed pool kernel's direct
  transport (``router_shard_publish`` and ``pool_direct_route`` in place of
  ``router_shard_side`` and ``pool_gather_route`` on the four NVFP4-expert layers); it is
  bit-identical, so each ``pool_direct`` arm equals the same set with ``router_fork`` only,
  at every width, eager and graph.
* ``tier7`` (``TIER7_ARMS``): ``full8n_nof`` (``TIER6N``: ``full7_nof`` + ``router_fork``)
  must equal ``full7_nof`` bit for bit at every width, and ``full9_nof`` (+ ``TIER7_NEW``:
  ``prep_fold,spill_free,pool_direct``) must equal ``full8n_nof``, eager and graph. Its
  full-attention layers carry ``rescale_cols`` 16
  (``test_tier7_full_layers_launch_the_spill_free_build``).

Every arm is a full model copy; the tables are separate because together they do not fit
one GPU.

The per-layer outputs are copied on a side stream forked after each layer and
joined after the model call (``_graph_edges.LayerOutputCopies``), so no kernel of
this test sits between a layer's fused post-FFN boundary and the next layer's QKV
GEMM, where it would turn ``early_release``'s early-launched edge into a full
dependency. Each captured graph's programmatic edges are read back: at 1..8 rows
an arm with the fused unpool boundary must have a programmatic edge out of each of
its five boundaries, any other graph none. ``MODELING_V2_HOOK_STREAM=main`` copies on
the capturing stream instead, the negative control: then no graph has one.
"""

import json
import os
import re
import types

import pytest
import torch
import tqdm

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    weights as target_weights,
)

__extra_import_path__ = ["."]
import _compose_plans as _plans  # noqa: E402
from _graph_edges import (  # noqa: E402
    LayerOutputCopies,
    boundary_edges,
    hook_on_side_stream,
    programmatic_edges,
)

# As tests/unittest/conftest.py does, which the multi-rank runs skip (--noconftest): no
# tqdm monitor thread, which the thread-leak check would report against the first test
# that shows a progress bar.
tqdm.tqdm.monitor_interval = 0

WORLD = 8
#: The checkpoint layers the model is built from, in order.
CKPT_LAYERS = (0, 1, 2, 59, 60, 61)
#: Decode steps after the prefill.
STEPS = 3
TOKENS_PER_BLOCK = 32
#: Decode widths: 1..8, and two the features must leave to the built-in path
#: (except ring_buddy, which takes up to 16 rows).
ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16)
#: Prompt length at width M: past the 2048-token window, so decode attends a full one.
PROMPT_BASE = 2100
#: Real embedding rows the inputs are drawn from.
TOKEN_POOL = 64
VOCAB = 202048
#: The serving switches the features need at construction.
REQUIRED_ENV = {
    "MUSESPARK_OPEN_ALLREDUCE": "MNNVL",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL": "1",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS": "1",
    "TRTLLM_LOW_M_GEMM_BACKEND": "auto",
}
#: (arm, the arm's ids (``_compose_plans.plan_of_ids``), the arm whose bits it must
#: equal); arms whose ids are no plan are left out. None as the selection is the
#: built-in model, not adopted; None as the reference means the
#: arm is only measured against the built-in model (every arm's distance from it is
#: recorded).
TIER1_ARMS = _plans.served_arms(
    (
        ("builtin", None, None),
        ("off", "", "builtin"),
        ("bi", "preattn_prep,hop,gathers", "builtin"),
        ("ring_buddy", "ring_buddy", None),
        ("paged_decode", "paged_decode", None),
        ("paged_decode_q8", "paged_decode_q8", None),
        ("dense_gemms", "dense_gemms", None),
        ("gemm_prefetch", "gemm_prefetch", "dense_gemms"),
        ("n_all", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
        ("full", "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch", "n_all"),
        (
            "full_dense_gemms",
            "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,dense_gemms",
            "full",
        ),
        ("preattn_prep_paged_decode", "preattn_prep,paged_decode", "paged_decode"),
        ("bi_paged_decode_q8", "preattn_prep,hop,gathers,paged_decode_q8", "paged_decode_q8"),
        ("bi_ring_buddy", "preattn_prep,hop,gathers,ring_buddy", "ring_buddy"),
        ("bi_gemm_prefetch", "preattn_prep,hop,gathers,gemm_prefetch", "gemm_prefetch"),
    )
)
#: The early release's table: early_release with gathers alone must not move a bit
#: (early_release needs the fused unpool boundary, whose trigger it moves).
EARLY_RELEASE_ARMS = (
    ("builtin", None, None),
    ("off", "", "builtin"),
    ("gathers_early_release", "gathers,early_release", "builtin"),
    ("dense_gemms", "dense_gemms", None),
)
#: The tier-1 target set, and the tier-2 fast paths on this model.
TIER1_TARGET = "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch"
TIER2_NEW = "early_release,persistent_experts,stale_poll"
#: The tier-2 table: every new path is bit-identical, on ring_buddy's MoE path and on the full set.
TIER2_ARMS = (
    ("builtin", None, None),
    ("ring_buddy", "ring_buddy", None),
    ("ring_buddy_persistent_experts", "ring_buddy,persistent_experts", "ring_buddy"),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full_nof", TIER1_TARGET, "n_nof"),
    ("full4_nof", TIER1_TARGET + "," + TIER2_NEW, "full_nof"),
)
#: The tier-2 and tier-3 target sets: tier 3 adds finalize_transport and moe_prologue.
TIER2_TARGET = TIER1_TARGET + ",early_release,persistent_experts,stale_poll"
TIER3_TARGET = (
    TIER1_TARGET + ",early_release,finalize_transport,persistent_experts,moe_prologue,stale_poll"
)
#: The tier-3 table: finalize_transport and moe_prologue on ring_buddy's MoE path
#: (persistent_experts' slot map read by the transport finalize, and fed by the prologue's
#: routing and placement), and the tier-3 target set against the tier-2 one.
TIER3_ARMS = (
    ("builtin", None, None),
    ("ring_buddy", "ring_buddy", None),
    ("ring_buddy_persistent_experts", "ring_buddy,persistent_experts", "ring_buddy"),
    (
        "ring_buddy_persistent_experts_finalize_transport",
        "ring_buddy,persistent_experts,finalize_transport",
        "ring_buddy",
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue",
        "gathers,ring_buddy,persistent_experts,moe_prologue",
        "ring_buddy",
    ),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full_nof", TIER1_TARGET, "n_nof"),
    ("full4_nof", TIER2_TARGET, "full_nof"),
    ("full5_nof", TIER3_TARGET, "full4_nof"),
)
#: The sliding-window block: the tier-1 and tier-2 target sets, and the block ids on top of
#: each (bitwise).
SWA_BLOCK_BASE = TIER1_TARGET
SWA_BLOCK_BASE4 = TIER1_TARGET + "," + TIER2_NEW
SWA_BLOCK_ARMS = (
    ("builtin", None, None),
    ("base", SWA_BLOCK_BASE, None),
    ("swa_prewait", SWA_BLOCK_BASE + ",swa_prewait", "base"),
    ("swa_fused_prep", SWA_BLOCK_BASE + ",swa_fused_prep", "base"),
    ("swa_fused_gate", SWA_BLOCK_BASE + ",swa_fused_gate", "base"),
    ("swa_block_all", SWA_BLOCK_BASE + ",swa_prewait,swa_fused_prep,swa_fused_gate", "base"),
    ("base4", SWA_BLOCK_BASE4, None),
    ("base4_swa_prewait", SWA_BLOCK_BASE4 + ",swa_prewait", "base4"),
    ("base4_swa_fused_prep", SWA_BLOCK_BASE4 + ",swa_fused_prep", "base4"),
    ("base4_swa_fused_gate", SWA_BLOCK_BASE4 + ",swa_fused_gate", "base4"),
    (
        "base4_swa_block_all",
        SWA_BLOCK_BASE4 + ",swa_prewait,swa_fused_prep,swa_fused_gate",
        "base4",
    ),
)
#: Model layers 1 and 5 are full-attention layers: under swa_fused_prep they alone run
#: preattn_prep, under swa_fused_gate they alone run early_release's norm-gate.
SWA_BLOCK_SLIDING_SITES, SWA_BLOCK_FULL_SITES = 4, 2
#: The direct publish's arms (``direct_publish``): direct_publish must not change a bit of
#: what hop + dense_gemms compute, alone (``hop,dense_gemms``), in the tier-1 target set and
#: with the tier-2 paths on top; at 8 rows o_proj publishes and the consumer hop runs, at
#: every other width the hop runs. ``hop_early_trigger,gemm_prefetch,direct_publish`` takes
#: the early trigger.
DIRECT_PUBLISH_ARMS = (
    ("builtin", None, None),
    ("off", "", "builtin"),
    ("hop_dense_gemms", "hop,dense_gemms", None),
    ("hop_dense_gemms_direct_publish", "hop,dense_gemms,direct_publish", "hop_dense_gemms"),
    (
        "hop_early_trigger_gemm_prefetch_direct_publish",
        "hop_early_trigger,gemm_prefetch,direct_publish",
        "hop_dense_gemms",
    ),
    ("srv", TIER1_TARGET, None),
    ("srv_direct_publish", TIER1_TARGET + ",direct_publish", "srv"),
    ("full4_nof", TIER1_TARGET + "," + TIER2_NEW, None),
    ("full4_nof_direct_publish", TIER1_TARGET + "," + TIER2_NEW + ",direct_publish", "full4_nof"),
)
#: The tier-4 target paths on this model (capture_fold needs DFlash capture layers; the
#: drafter-side ids are not on this model), and the tier-4 target set.
TIER4_SWA_BLOCK = ",swa_prewait,swa_fused_prep,swa_fused_gate"
TIER4_PREP_APPEND = ",prep_append,prep_append_prefetch_trim"
TIER4_TARGET = TIER3_TARGET + TIER4_SWA_BLOCK + TIER4_PREP_APPEND + ",direct_publish"
#: The tier-4 table: each tier-4 group on the tier-3 target set, and the tier-4 target set,
#: each against the arm it must equal bit for bit.
TIER4_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full5_nof", TIER3_TARGET, "n_nof"),
    ("full5_swa_block", TIER3_TARGET + TIER4_SWA_BLOCK, "full5_nof"),
    ("full5_prep_append", TIER3_TARGET + TIER4_PREP_APPEND, "full5_nof"),
    ("full5_direct_publish", TIER3_TARGET + ",direct_publish", "full5_nof"),
    ("full6_nof", TIER4_TARGET, "full5_nof"),
)
#: The router fork's table: hop_early_trigger and router_fork on the tier-4 target set.
TIER4_TARGET_HOP_EARLY_TRIGGER = TIER4_TARGET.replace("hop,", "hop_early_trigger,")
ROUTER_FORK_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full6_hop_early_trigger", TIER4_TARGET_HOP_EARLY_TRIGGER, "full6_nof"),
    ("full6_router_fork", TIER4_TARGET_HOP_EARLY_TRIGGER + ",router_fork", "full6_nof"),
)
#: Model layers 1 and 5 (checkpoint 1 and 61) are the full-attention layers.
FULL_LAYER_INDICES = (1, 5)
#: The direct unpool publish's table: the direct transport added to gathers with
#: dense_gemms and to the tier-2 target set.
UNPOOL_DIRECT_ARMS = (
    ("builtin", None, None),
    ("gathers_dense_gemms", "gathers,dense_gemms", None),
    (
        "gathers_dense_gemms_unpool_direct",
        "gathers,dense_gemms,unpool_direct",
        "gathers_dense_gemms",
    ),
    ("full4_nof", TIER2_TARGET, None),
    ("full4_nof_unpool_direct", TIER2_TARGET + ",unpool_direct", "full4_nof"),
)
#: The QKV prefetch's table: the QKV prefetch added to the tier-3 target set, and to that
#: set with unpool_direct.
QKV_PREFETCH_ARMS = (
    ("builtin", None, None),
    ("full5_nof", TIER3_TARGET, None),
    ("full5_nof_qkv_prefetch", TIER3_TARGET + ",qkv_prefetch", "full5_nof"),
    ("full5_nof_unpool_direct", TIER3_TARGET + ",unpool_direct", None),
    (
        "full5_nof_unpool_direct_qkv_prefetch",
        TIER3_TARGET + ",unpool_direct,qkv_prefetch",
        "full5_nof_unpool_direct",
    ),
)
#: The direct unpool publish on the tier-3 set (moe_prologue and finalize_transport in
#: it), with and without unpool_direct.
UNPOOL_DIRECT_TIER3_ARMS = (
    ("builtin", None, None),
    ("full5_nof", TIER3_TARGET, None),
    ("full5_nof_unpool_direct", TIER3_TARGET + ",unpool_direct", "full5_nof"),
)
#: The direct unpool publish on the tier-4 set (direct_publish in it), with and without
#: unpool_direct. direct_publish publishes o_proj into the hop transport and unpool_direct
#: the unpool GEMM into the boundary transport, each site under its own producer policy
#: (``dense_gemms.claim_publish``).
UNPOOL_DIRECT_TIER4_ARMS = (
    ("builtin", None, None),
    ("full6_nof", TIER4_TARGET, None),
    ("full6_nof_unpool_direct", TIER4_TARGET + ",unpool_direct", "full6_nof"),
)
#: The early expert ids' table: on ring_buddy's MoE path fed by moe_prologue, and on the
#: tier-3 target set.
EARLY_IDS_ARMS = (
    ("builtin", None, None),
    ("ring_buddy", "ring_buddy", None),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue",
        "gathers,ring_buddy,persistent_experts,moe_prologue",
        "ring_buddy",
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids",
        "gathers,ring_buddy,persistent_experts,moe_prologue,early_ids",
        "ring_buddy",
    ),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full5_nof", TIER3_TARGET, "n_nof"),
    ("full5_early_ids", TIER3_TARGET + ",early_ids", "full5_nof"),
)
#: The QKV prefetch on the tier-4 set, with and without unpool_direct.
QKV_PREFETCH_TIER4_ARMS = (
    ("builtin", None, None),
    ("full6_nof", TIER4_TARGET, None),
    ("full6_nof_qkv_prefetch", TIER4_TARGET + ",qkv_prefetch", "full6_nof"),
    ("full6_nof_unpool_direct", TIER4_TARGET + ",unpool_direct", "full6_nof"),
    (
        "full6_nof_unpool_direct_qkv_prefetch",
        TIER4_TARGET + ",unpool_direct,qkv_prefetch",
        "full6_nof_unpool_direct",
    ),
)
#: The bit-identical tier-5 paths on this model, and the tier-4 target set with them
#: (``full7b_nof``).
TIER5_NEW = ",hop_early_trigger,unpool_direct,early_ids,qkv_prefetch,bf16_routing"
TIER5B = TIER4_TARGET + TIER5_NEW
#: The tier-5 table: full7b_nof against the tier-4 target set (bit for bit), and the budgeted
#: paths on full7b_nof, each against full7b_nof where it changes nothing (``CHANGES_AT``).
TIER5_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full7b_nof", TIER5B, "full6_nof"),
    ("full7b_three_windows", TIER5B + ",three_windows", "full7b_nof"),
    ("full7b_split_kv", TIER5B + ",split_kv", "full7b_nof"),
    ("full7_nof", TIER5B + ",three_windows,split_kv", "full7b_nof"),
)
#: The prep fold's table: full7_nof and full7b_split_kv on the tier-5 reference chain
#: (n_nof -> full6_nof -> full7b_nof), and each with prep_fold, which must give its bits.
PREP_FOLD_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full7b_nof", TIER5B, "full6_nof"),
    ("full7b_split_kv", TIER5B + ",split_kv", "full7b_nof"),
    ("full7b_split_kv_prep_fold", TIER5B + ",split_kv,prep_fold", "full7b_split_kv"),
    ("full7_nof", TIER5B + ",three_windows,split_kv", "full7b_nof"),
    ("full7_prep_fold", TIER5B + ",three_windows,split_kv,prep_fold", "full7_nof"),
    ("full7_spill_free", TIER5B + ",three_windows,split_kv,spill_free", "full7_nof"),
    (
        "full7_prep_fold_spill_free",
        TIER5B + ",three_windows,split_kv,prep_fold,spill_free",
        "full7_nof",
    ),
)
#: The direct pool transport's table: router_fork and router_fork + pool_direct on full7b_nof
#: (through the tier-5 chain to the core: each arm against the same set without its newest
#: ids, so the QKV prefetch-edge count is checked against an arm that differs from it in
#: qkv_prefetch alone) and on full7_nof (pool_direct against full7_nof + router_fork).
POOL_DIRECT_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full7b_nof", TIER5B, "full6_nof"),
    ("full7b_router_fork", TIER5B + ",router_fork", "full7b_nof"),
    ("full7b_pool_direct", TIER5B + ",router_fork,pool_direct", "full7b_router_fork"),
    ("full7_router_fork", TIER5B + ",three_windows,split_kv,router_fork", None),
    (
        "full7_pool_direct",
        TIER5B + ",three_windows,split_kv,router_fork,pool_direct",
        "full7_router_fork",
    ),
)
#: Arm -> the widths at which it changes bits by design (a budgeted path): there it must
#: equal its reference arm on the outputs ``UPSTREAM`` names only, and its distance from
#: the reference is recorded; at every other width it must equal it. The three-window
#: placement (moe_prologue's routed kernel) and the split-KV partials (prep_append's folded
#: steps) act at up to 8 rows.
CHANGES_AT = {
    "full7b_three_windows": tuple(range(1, 9)),
    "full7b_split_kv": tuple(range(1, 9)),
    "full7_nof": tuple(range(1, 9)),
}
#: The outputs written before the first layer a budgeted tier-5 path changes (model layer 1:
#: the first NVFP4-expert layer and the first full-attention layer): layer 0's outputs, the
#: prefill, and the KV cache bytes of layers 0 and 1 (layer 1's K/V come from layer 0).
UPSTREAM = re.compile(r"^(decode\d+/layer0\.\d+|cache/layer[01]\.[kv]|prefill.*)$")
#: The router fork on the tier-5 sets: full7b_nof (hop_early_trigger and early_ids in it)
#: with router_fork against full7b_nof, and full7b_nof + three_windows with router_fork
#: against full7b_nof + three_windows (router_fork changes no value, so at every width).
ROUTER_FORK_TIER5_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full7b_nof", TIER5B, "full6_nof"),
    ("full7b_router_fork", TIER5B + ",router_fork", "full7b_nof"),
    ("full7b_three_windows", TIER5B + ",three_windows", "full7b_nof"),
    (
        "full7b_three_windows_router_fork",
        TIER5B + ",three_windows,router_fork",
        "full7b_three_windows",
    ),
)
#: full7_nof + router_fork: the tier-6 target set on this model without ``finalize_trigger``
#: (its drafter ids act on the DFlash drafter, which this model does not have).
TIER6N = TIER5B + ",three_windows,split_kv,router_fork"
#: The tier-7 target paths: ``prep_fold`` (the prep-append folded into the split-KV kernel),
#: ``spill_free`` (the split-KV kernel's spill-free build) and ``pool_direct`` (the pool GEMM
#: and the router's publish twin write into the routed pool kernel's direct transport).
TIER7_NEW = ",prep_fold,spill_free,pool_direct"
#: The tier-7 table: full8n_nof against full7_nof and full9_nof (full8n_nof + the tier-7
#: paths) against full8n_nof at every width, the widths where full7_nof differs from
#: full7b_nof by design included.
TIER7_ARMS = (
    ("builtin", None, None),
    ("n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", None),
    ("full6_nof", TIER4_TARGET, "n_nof"),
    ("full7b_nof", TIER5B, "full6_nof"),
    ("full7_nof", TIER5B + ",three_windows,split_kv", "full7b_nof"),
    ("full8n_nof", TIER6N, "full7_nof"),
    ("full9_nof", TIER6N + TIER7_NEW, "full8n_nof"),
)
_SET = os.environ.get("MODELING_V2_COMPOSE_TIER", "tier1")
ARMS = {
    "tier1": TIER1_ARMS,
    "early_release": EARLY_RELEASE_ARMS,
    "tier2": TIER2_ARMS,
    "tier3": TIER3_ARMS,
    "swa_block": SWA_BLOCK_ARMS,
    "direct_publish": DIRECT_PUBLISH_ARMS,
    "tier4": TIER4_ARMS,
    "unpool_direct": UNPOOL_DIRECT_ARMS,
    "qkv_prefetch": QKV_PREFETCH_ARMS,
    "unpool_direct_tier3": UNPOOL_DIRECT_TIER3_ARMS,
    "unpool_direct_tier4": UNPOOL_DIRECT_TIER4_ARMS,
    "early_ids": EARLY_IDS_ARMS,
    "qkv_prefetch_tier4": QKV_PREFETCH_TIER4_ARMS,
    "tier5": TIER5_ARMS,
    "router_fork": ROUTER_FORK_ARMS,
    "router_fork_tier5": ROUTER_FORK_TIER5_ARMS,
    "prep_fold": PREP_FOLD_ARMS,
    "pool_direct": POOL_DIRECT_ARMS,
    "tier7": TIER7_ARMS,
}[_SET]
#: A combination of non-bit-identical features and the single-feature arms it is made of.
COMBINED = {
    "tier1": {"n_all": ("ring_buddy", "paged_decode_q8", "gemm_prefetch")},
    "tier5": {"full7_nof": ("full7b_three_windows", "full7b_split_kv")},
}.get(_SET, {})
#: Margin of the no-amplification gate: combination <= MARGIN x the sum of the singles.
MARGIN = 1.25
#: Run twice; the second run must give the first one's bits.
DETERMINISM = {
    "tier1": ("full",),
    "early_release": ("gathers_early_release",),
    "tier2": ("full4_nof",),
    "tier3": ("full5_nof",),
    "swa_block": ("swa_block_all", "base4_swa_block_all"),
    "direct_publish": ("full4_nof_direct_publish",),
    "tier4": ("full6_nof",),
    "unpool_direct": ("full4_nof_unpool_direct",),
    "qkv_prefetch": ("full5_nof_unpool_direct_qkv_prefetch",),
    "unpool_direct_tier3": ("full5_nof_unpool_direct",),
    "unpool_direct_tier4": ("full6_nof_unpool_direct",),
    "early_ids": ("full5_early_ids",),
    "qkv_prefetch_tier4": ("full6_nof_unpool_direct_qkv_prefetch",),
    "tier5": ("full7b_nof", "full7_nof"),
    "router_fork": ("full6_router_fork",),
    "router_fork_tier5": ("full7b_router_fork", "full7b_three_windows_router_fork"),
    "prep_fold": ("full7_nof", "full7_prep_fold"),
    "pool_direct": ("full7b_pool_direct", "full7_pool_direct"),
    "tier7": ("full8n_nof", "full9_nof"),
}[_SET]


#: Entry point -> (feature ids that run it, call sites per decode forward, widest width).
#: Model layers 0 and 5 (checkpoint 0 and 61) have BF16 experts, so the fused pool
#: quantization falls back there; the last layer's unpool keeps the built-in join;
#: model layers 1 and 5 are full-attention layers, the other four sliding-window ones.
ENTRIES = {
    "preattn_prep": ({"preattn_prep"}, 6, 8),
    "post_attn_hop": ({"hop", "hop_early_trigger"}, 6, 8),
    "unpool_gather_boundary": ({"gathers"}, 5, 8),
    "pool_gather_fp4quant": ({"gathers"}, 4, 8),
    "paged_decode_attention": ({"paged_decode", "paged_decode_q8"}, 4, 8),
    "dense_gemm": ({"dense_gemms", "gemm_prefetch"}, 24, 8),
    "l2_prefetch": ({"gemm_prefetch"}, 17, 8),
    "ring_buddy_assign": (set(_plans.LOADED), 6, 16),
    "attention_norm_gate": ({"early_release"}, 6, 8),
    # persistent_experts needs ring_buddy; it runs on the four NVFP4 MoE layers (model layers 1-4).
    "persistent_experts": ({"persistent_experts"}, 4, 8),
    "router_shard": ({"moe_prologue"}, 4, 8),
    # router_fork's plain launch of the router shard on its side stream, in place of router_shard.
    "router_shard_side": ({"router_fork"}, 4, 8),
    "pool_gather_route": ({"moe_prologue"}, 4, 8),
    # pool_direct: the publish twin and the direct routed kernel in place of the side launch and
    # the routed pool kernel, on the 8-row capture only (_expected_counts).
    "router_shard_publish": ({"pool_direct"}, 4, 8),
    "pool_direct_route": ({"pool_direct"}, 4, 8),
    # finalize_transport's kernel takes the fused finalize of every MoE layer on calls of 1..16 tokens
    # (counted on the workspace it installs: no module-level entry).
    "finalize_transport": ({"finalize_transport"}, 6, 16),
    "post_attn_consumer_hop": ({"direct_publish"}, 6, 8),
    # prep_append's folded prep and the built-in preprocess it skips, on the two
    # full-attention layers (counted when an arm selects prep_append: ``env``, or the
    # prep-append composition test).
    "preattn_prep_append": ({"prep_append"}, len(FULL_LAYER_INDICES), 8),
    "preprocess_elided": ({"prep_append"}, len(FULL_LAYER_INDICES), 8),
    # unpool_direct: in place of the fused unpool boundary, on the 8-row capture only.
    "unpool_direct_boundary": ({"unpool_direct"}, 5, 8),
    # early_ids: the expert kernel's early launches (``early_expert_ids.launches_early`` true),
    # on the four NVFP4 layers that run moe_prologue's routed kernel into persistent_experts',
    # up to 8 rows.
    "early_expert_ids": ({"early_ids"}, 4, 8),
    # qkv_prefetch's prefetch of the next QKV weight: at the forward's start and after the
    # expert call of each layer whose boundary early_release converted (5), from 3 rows
    # (_expected_counts).
    "qkv_prefetch_prefetch": ({"qkv_prefetch"}, 6, 8),
    # split_kv on the full-attention layers, at up to 8 rows (prep_append's folded steps).
    "full_attention_split_kv": ({"split_kv"}, len(FULL_LAYER_INDICES), 8),
    "combine_norm_gate": ({"split_kv"}, len(FULL_LAYER_INDICES), 8),
    # prep_fold: the prep folded into the split-KV kernel on the same steps (in place of the prep-append,
    # the elided preprocess and full_attention_split_kv).
    "full_attention_prep_fold": ({"prep_fold"}, len(FULL_LAYER_INDICES), 8),
}
#: The entries the prep-fold kernel takes the place of on the steps it serves.
PREP_FOLD_REPLACES = ("preattn_prep_append", "preprocess_elided", "full_attention_split_kv")
#: Whether an arm selects split_kv: its entry points are then counted (``env``).
SPLIT_KV_SELECTED = any("split_kv" in (selection or "") for _, selection, _ in ARMS)
#: Whether an arm selects prep_fold: its kernel's entry point is then counted (``env``).
PREP_FOLD_SELECTED = any("prep_fold" in (selection or "") for _, selection, _ in ARMS)
#: Whether an arm selects prep_append: its entry points are then counted, and every fresh
#: metadata's attention workspace is sized up front (``_metadata``).
PREP_APPEND_ARMS = any("prep_append" in (selection or "") for _, selection, _ in ARMS)
#: moe_prologue's routed pool op takes the place of the fused pool gather and of
#: ring_buddy's placement kernel on the layers it serves (the four NVFP4-expert layers, up
#: to 8 rows).
PROLOGUE_REPLACES = {"pool_gather_fp4quant": 4, "ring_buddy_assign": 4}
#: direct_publish publishes and consumes at exactly this width (``target_layers.DirectPublish.ROWS``).
DIRECT_PUBLISH_ROWS = 8

#: Copy the per-layer outputs on a side stream (``MODELING_V2_HOOK_STREAM``, ``_graph_edges``).
HOOK_SIDE = hook_on_side_stream()
#: Widest decode step the fused unpool boundary (gathers) takes.
UNPOOL_MAX_ROWS = 8

_RESULTS: dict = {}
_COUNTS: dict = {}


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    checkpoint = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")
    if not os.path.isfile(os.path.join(checkpoint, "model.safetensors.index.json")):
        pytest.skip(
            "needs MUSESPARK_REFERENCE_CKPT=<the Muse-Spark-Open-816B-A42B-NVFP4 checkpoint>"
        )
    missing = [f"{k}={v}" for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    if missing:
        pytest.skip(f"needs the serving switches: {' '.join(missing)}")
    return comm, rank, checkpoint


class _Lazy:
    """One checkpoint tensor, read on first use per index and kept on the host for the next copy."""

    def __init__(self, handle, key):
        self._handle = handle
        self._key = key
        self._read = {}

    def get_shape(self):
        return self._handle.get_slice(self._key).get_shape()

    def __getitem__(self, index):
        tag = repr(index)
        if tag not in self._read:
            self._read[tag] = self._handle.get_slice(self._key)[index]
        return self._read[tag].clone()


def _checkpoint_weights(checkpoint: str):
    """Every key of the checkpoint, lazily, renamed to the source spelling; and the rename flag."""
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        rename_open_format_weights,
    )

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}
    weights = {}
    for key, shard in weight_map.items():
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(checkpoint, shard), framework="pt")
        weights[key] = _Lazy(handles[shard], key)
    open_format = rename_open_format_weights(weights)
    return weights, open_format


def _model_config(checkpoint: str, mapping):
    """The checkpoint's model config over ``CKPT_LAYERS``, with the FP8 KV cache."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.quantization.mode import QuantAlgo

    config = ModelConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        mapping=mapping,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
        allreduce_strategy=AllReduceStrategy.NCCL,
        max_num_tokens=8192,
    )
    pretrained = config.pretrained_config
    full = pretrained.num_hidden_layers
    # The per-layer schedule (window, RoPE) is computed for the full model; keep these layers'.
    for name, value in list(vars(pretrained).items()):
        if isinstance(value, list) and len(value) == full:
            setattr(pretrained, name, [value[i] for i in CKPT_LAYERS])
    pretrained.num_hidden_layers = len(CKPT_LAYERS)
    quant = config.quant_config
    # The serving kv_cache_config.dtype fp8.
    quant.kv_cache_quant_algo = QuantAlgo.FP8
    if quant.exclude_modules:
        remapped = []
        for pattern in quant.exclude_modules:
            found = re.search(r"layers\.(\d+)\.", pattern)
            if found is None:
                remapped.append(pattern)
            elif int(found.group(1)) in CKPT_LAYERS:
                index = CKPT_LAYERS.index(int(found.group(1)))
                remapped.append(pattern.replace(found.group(0), f"layers.{index}."))
        quant.exclude_modules = remapped
    return config


class _Counted:
    """A callable that counts its calls into ``_COUNTS[name]`` and calls through."""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        _COUNTS[self.name] = _COUNTS.get(self.name, 0) + 1
        return self.fn(*args, **kwargs)

    def __getattr__(self, name):
        # Everything else (``router_gemv.precompile``, say) is the wrapped object's.
        return getattr(self.fn, name)


def _selection(value):
    return _plans.selection(value)


def _build_arm(e, weights, open_format, selection):
    """One copy of the model: built, adopted under ``selection``, loaded, and post-loaded,
    in the serving order: the shell's ``load_weights`` installs the load-time features
    (``ring_buddy``) right after the built-in load, before the loader's post-load walk."""
    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        WeightAccounting,
        load_layer,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenForCausalLM
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    config = _model_config(e.checkpoint, e.mapping)
    with torch.device("cuda"):
        model = MuseSparkOpenForCausalLM(config).model
    plan = _plans.plan_of_ids(selection)
    if plan is not None:
        # For the device's architecture, as the shell adopts it (the decode FMHA selection).
        modeling.MuseSparkOpenTepModelingV2.adopt(model, plan, modeling.device_compute_capability())
        if model._decode.preattn_prep is not None:
            model._decode.preattn_prep = _Counted("preattn_prep", model._decode.preattn_prep)
        if model._post_attn_hop is not None:
            model._post_attn_hop.op = _Counted("post_attn_hop", model._post_attn_hop.op)
            direct = model._post_attn_hop.direct
            if direct is not None:
                direct.op = _Counted("post_attn_consumer_hop", direct.op)
        transport = getattr(model, "_finalize_transport", None)
        if transport is not None:
            # Every call the workspace routes to the transport kernel launches through _launch.
            transport._launch = _Counted("finalize_transport", transport._launch)
    owner = types.SimpleNamespace(config=config.pretrained_config)
    accounting = WeightAccounting()
    for index, layer in enumerate(model.layers):
        load_layer(owner, layer, CKPT_LAYERS[index], weights, accounting, open_format=open_format)
    with torch.no_grad():
        model.norm.weight.copy_(weights["model.norm.weight"][:].to(model.norm.weight.dtype))
    target_weights.install_loaded_features(model, e.mapping, plan or dispatch.NO_PATHS)
    if "direct_publish" in _selection(selection):
        target_layers.verify_direct_publish(model)
    ModelLoader._walk_full_post_load(model)
    for layer in model.layers:
        layer.register_forward_hook(_copies(e))
    return model, config


def _copies(e) -> LayerOutputCopies:
    """The environment's per-layer output copier (one side stream for every arm)."""
    if getattr(e, "copies", None) is None:
        e.copies = LayerOutputCopies(e.sink, HOOK_SIDE)
    return e.copies


def _expected_boundary_edges(selected: frozenset, rows: int) -> int:
    """Programmatic edges out of the fused unpool boundaries of one captured decode graph."""
    unpool = "gathers" in selected and rows <= UNPOOL_MAX_ROWS
    return len(CKPT_LAYERS) - 1 if HOOK_SIDE and unpool else 0


@pytest.fixture(scope="module")
def env():
    """Every arm's model copy, the LM head shard and the real embedding rows (collective)."""
    comm, rank, checkpoint = _world()
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
        triggers as pdl_triggers,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
        dispatcher as dense_gemms,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.moe_prologue import (  # noqa: E501
        prologue as moe_prologue,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.next_qkv_prefetch import (  # noqa: E501
        prefetch as qkv_prefetch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
        gathers as projection_gathers,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        pool_direct_route as pool_direct_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        unpool_direct_boundary as direct_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        persistent_experts as pe_mod,
    )
    from tensorrt_llm.mapping import Mapping

    e = types.SimpleNamespace(
        comm=comm, rank=rank, checkpoint=checkpoint, sink=[], models={}, stale_poll_on={}
    )
    e.mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
    )
    weights, open_format = _checkpoint_weights(checkpoint)
    # Module-level entry points, counted for every arm (the call sites look them up by name).
    patched = [
        (target_attention, "paged_decode_attention", "paged_decode_attention"),
        (projection_gathers._unpool, "unpool_gather_boundary", "unpool_gather_boundary"),
        (projection_gathers._pool, "pool_gather_rmsnorm_fp4quant", "pool_gather_fp4quant"),
        (dense_gemms, "dense_gemm", "dense_gemm"),
        (dense_gemms, "l2_prefetch", "l2_prefetch"),
        (ring_buddy, "ring_buddy_assign", "ring_buddy_assign"),
        (pdl_triggers._norm_gate, "attention_norm_gate", "attention_norm_gate"),
        (pe_mod, "persistent_experts", "persistent_experts"),
        (moe_prologue._shard, "router_shard", "router_shard"),
        (moe_prologue._shard, "router_shard_side", "router_shard_side"),
        (moe_prologue._route, "pool_gather_route", "pool_gather_route"),
        (moe_prologue._shard, "router_shard_publish", "router_shard_publish"),
        (pool_direct_mod, "pool_direct_route", "pool_direct_route"),
        (direct_mod, "unpool_direct_boundary", "unpool_direct_boundary"),
        (qkv_prefetch, "l2_prefetch", "qkv_prefetch_prefetch"),
    ]
    if os.environ.get("MODELING_V2_COMPOSE_TIER") in (
        "swa_block",
        "tier4",
        "unpool_direct_tier4",
        "qkv_prefetch_tier4",
        "tier5",
        "router_fork",
        "router_fork_tier5",
        "prep_fold",
        "pool_direct",
        "tier7",
    ):
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.swa_block import (  # noqa: E501
            attention as swa_attention_block,
        )

        patched.append((swa_attention_block, "swa_block_attention", "swa_block_attention"))
    if SPLIT_KV_SELECTED:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
            combine_norm_gate,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
            full_attention_split_kv as full_attention_split_kv,
        )

        # Unless the caller counts them already (the split-KV composition test wraps this fixture).
        for module, attr in (
            (full_attention_split_kv, "full_attention_split_kv"),
            (combine_norm_gate, "combine_norm_gate"),
        ):
            if not isinstance(getattr(module, attr), _Counted):
                patched.append((module, attr, attr))
    if PREP_FOLD_SELECTED:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
            full_attention_prep_fold as full_attention_prep_fold,
        )

        patched.append(
            (full_attention_prep_fold, "full_attention_prep_fold", "full_attention_prep_fold")
        )
    folded_outputs = None
    if PREP_APPEND_ARMS:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
            attention as full_attention,
        )

        patched.append((full_attention, "preattn_prep_append", "preattn_prep_append"))
        folded_outputs = full_attention.FoldedPreprocess.outputs

        def counted_outputs(self, args):
            _COUNTS["preprocess_elided"] = _COUNTS.get("preprocess_elided", 0) + 1
            return folded_outputs(self, args)

        full_attention.FoldedPreprocess.outputs = counted_outputs
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    for module, attr, name in patched:
        setattr(module, attr, _Counted(name, getattr(module, attr)))
    # early_ids: count only the expert launches that run early.
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        early_expert_ids as ei_mod,
    )

    launches_early = ei_mod.launches_early

    def counted_launches_early(early, topk_ids):
        ran = launches_early(early, topk_ids)
        if ran:
            _COUNTS["early_expert_ids"] = _COUNTS.get("early_expert_ids", 0) + 1
        return ran

    ei_mod.launches_early = counted_launches_early
    originals.append((ei_mod, "launches_early", launches_early))
    try:
        e.layout = {}
        for arm, selection, _ in ARMS:
            e.models[arm], config = _build_arm(e, weights, open_format, selection)
            e.layout[arm] = [
                (
                    type(layer.self_attn).__name__,
                    type(layer.feed_forward).__name__,
                    type(layer.self_attn.qkv_proj).__name__,
                    hasattr(layer.feed_forward, "_replicated_experts"),
                    type(getattr(layer.feed_forward, "_finalize_ar_workspace", None)).__name__,
                    getattr(layer.feed_forward, "_router_stream", None) is not None,
                )
                for layer in e.models[arm].layers
            ]
            e.stale_poll_on[arm] = _stale_poll_schedules(e.models[arm])
            comm.Barrier()
        e.pretrained = config.pretrained_config
        e.free_gib = torch.cuda.mem_get_info()[0] / 2**30
        embed = weights["model.embed_tokens.weight"]
        generator = torch.Generator().manual_seed(2026)
        tokens = torch.randint(0, embed.get_shape()[0], (TOKEN_POOL,), generator=generator)
        e.table = torch.cat([embed[t : t + 1] for t in tokens.tolist()]).to("cuda", torch.bfloat16)
        shard = VOCAB // WORLD
        e.vocab_begin = rank * shard
        e.head = weights["lm_head.weight"][e.vocab_begin : e.vocab_begin + shard].to(
            "cuda", torch.float32
        )
        torch.cuda.synchronize()
        comm.Barrier()
        yield e
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        for module, attr, original in originals:
            setattr(module, attr, original)
        if folded_outputs is not None:
            full_attention.FoldedPreprocess.outputs = folded_outputs


def _stale_poll_schedules(model) -> dict:
    """Which polling schedule each stale-poll site of ``model`` runs (False: the default)."""
    hop = getattr(model, "_post_attn_hop", None)
    return {
        "hop": bool(hop is not None and getattr(hop.workspace, "stale_poll", False)),
    }


def _kv_cache(e, capacity: int):
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    pages = -(-capacity // TOKENS_PER_BLOCK) + 1
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=pages * TOKENS_PER_BLOCK),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=len(CKPT_LAYERS),
        num_kv_heads=e.pretrained.num_key_value_heads,
        head_dim=e.pretrained.head_dim,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=pages * TOKENS_PER_BLOCK,
        max_batch_size=1,
        mapping=e.mapping,
        dtype=tensorrt_llm.bindings.DataType.FP8,
    )


def _metadata(e, kv, seq_len, cached, prompt_len, num_contexts):
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
        kv_cache_manager=kv,
        request_ids=[0],
        prompt_lens=[prompt_len],
        max_num_requests=1,
        max_num_tokens=8192,
        mapping=e.mapping,
    )
    metadata.prepare()
    if PREP_APPEND_ARMS:
        # A fresh metadata's attention workspace starts empty and is grown by the first
        # attention call, so the first full-attention layer of an eager step would not fold
        # (prep_append needs the workspace's layout): size it up front, as the serving engine's
        # one metadata is sized by the warm-up before any graph is captured.
        for workspace in (metadata.workspace, metadata.cuda_graph_workspace):
            if workspace is not None and workspace.numel() * workspace.element_size() < (1 << 30):
                workspace.resize_((1 << 30) // workspace.element_size())
    return metadata


def _plan(rows: int):
    """Prompt length, prompt token ids and decode token ids (indices into ``table``) at ``rows``."""
    prompt_len = PROMPT_BASE + 3 * rows
    generator = torch.Generator().manual_seed(rows)
    prompt = torch.randint(0, TOKEN_POOL, (prompt_len,), generator=generator)
    decode = [torch.randint(0, TOKEN_POOL, (rows,), generator=generator) for _ in range(STEPS)]
    return prompt_len, prompt, decode


def _collect(out: dict, prefix: str, layer_outputs) -> None:
    for index, layer_out in enumerate(layer_outputs):
        for part, tensor in enumerate(layer_out):
            out[f"{prefix}/layer{index}.{part}"] = tensor.clone()


def _greedy(e, hidden: torch.Tensor) -> torch.Tensor:
    """The greedy token of every row over the full vocabulary (LM-head shards, collective)."""
    logits = hidden.float() @ e.head.t()
    value, index = logits.max(dim=-1)
    gathered = e.comm.allgather((value.cpu(), (index + e.vocab_begin).cpu()))
    values = torch.stack([v for v, _ in gathered])
    indices = torch.stack([i for _, i in gathered])
    best = values.argmax(dim=0)
    return indices.gather(0, best[None]).squeeze(0)


def _cache_bytes(kv, metadata, length: int) -> dict:
    """The FP8 K and V bytes of every layer at positions ``0 .. length - 1``."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.bindings.internal import thop
    from tensorrt_llm.quantization.mode import QuantMode

    out = {}
    positions = torch.arange(length, device="cuda")
    for layer in range(len(CKPT_LAYERS)):
        attention_layer = kv.layer_offsets[layer]
        total = kv.impl.get_page_index_upper_bound(attention_layer, Role.KEY)
        pool, tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            metadata.host_kv_cache_pool_pointers,
            metadata.host_kv_cache_pool_mapping,
            metadata.kv_cache_block_offsets,
            attention_layer,
            kv.num_kv_heads_per_layer[attention_layer],
            TOKENS_PER_BLOCK,
            kv.head_dim,
            2,
            int(total),
            int(QuantMode.FP8_KV_CACHE),
            0,
            1,
            torch.bfloat16,
        )
        pool_u8 = pool.view(torch.uint8)
        for role, name in ((0, "k"), (1, "v")):
            pages = tables[0, role].long()[positions // TOKENS_PER_BLOCK]
            out[f"cache/layer{layer}.{name}"] = pool_u8[
                pages, :, positions % TOKENS_PER_BLOCK, :
            ].clone()
    return out


@torch.inference_mode()
def _run(e, model, rows: int, graph: bool = False):
    """Prefill one request, then ``STEPS`` decode steps of ``rows`` rows; every output, the
    greedy tokens, the cache bytes and the decode-step entry-point counts.

    With ``graph`` the decode steps replay one CUDA graph, captured after an eager
    warm-up as the engine captures its decode graphs, with the attention metadata and
    the inputs updated in place before every replay; the counts are then the capture's.
    """
    from tensorrt_llm._torch.metadata import KVCacheParams

    prompt_len, prompt, decode = _plan(rows)
    capacity = prompt_len + STEPS * rows
    kv = _kv_cache(e, capacity)
    out = {}
    cuda_graph = None
    plan = getattr(model, "_dense_gemm_plan", None)
    try:
        kv.add_dummy_requests([0], [capacity])
        metadata = _metadata(e, kv, prompt_len, 0, prompt_len, num_contexts=1)
        e.sink.clear()
        out["prefill"] = model(
            attn_metadata=metadata,
            position_ids=torch.arange(prompt_len, dtype=torch.int, device="cuda")[None],
            inputs_embeds=e.table[prompt.cuda()],
        ).clone()
        e.copies.join()
        # The prefill's per-layer outputs are not kept (P x 8192 FP32 per layer): its final
        # output and the cache it wrote cover it.
        prefill_cache = _cache_bytes(kv, metadata, prompt_len)
        out.update({f"prefill_{k}": v for k, v in prefill_cache.items()})
        _COUNTS.clear()
        if not graph:
            for step in range(STEPS):
                cached = prompt_len + step * rows
                metadata = _metadata(e, kv, rows, cached, prompt_len, 0)
                e.sink.clear()
                hidden = model(
                    attn_metadata=metadata,
                    position_ids=torch.arange(cached, cached + rows, dtype=torch.int).cuda()[None],
                    inputs_embeds=e.table[decode[step].cuda()],
                )
                e.copies.join()
                out[f"decode{step}"] = hidden.clone()
                _collect(out, f"decode{step}", e.sink)
            counts = dict(_COUNTS)
        else:
            metadata = _metadata(e, kv, rows, prompt_len, prompt_len, 0)
            metadata = metadata.create_cuda_graph_metadata(1, max_draft_tokens=rows - 1)
            metadata.prepare()
            position_ids = torch.arange(prompt_len, prompt_len + rows, dtype=torch.int).cuda()[None]
            embeds = e.table[decode[0].cuda()].clone()

            def forward():
                hidden = model(
                    attn_metadata=metadata, position_ids=position_ids, inputs_embeds=embeds
                )
                e.copies.join()
                return hidden

            forward()  # warm-up: compiles and allocates outside the capture
            torch.cuda.synchronize()
            e.comm.Barrier()
            _COUNTS.clear()
            e.sink.clear()
            # Kept, so its edges can be read before the first replay instantiates it.
            cuda_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(cuda_graph):
                static_out = forward()
            counts = dict(_COUNTS)
            edges = programmatic_edges(cuda_graph)
            counts["boundary_pdl_edges"] = boundary_edges(edges)
            counts["prefetch_pdl_edges"] = sum(
                count for (source, _, _), count in edges.items() if "l2_prefetch" in source
            )
            static_layers = list(e.sink)
            torch.cuda.synchronize()
            e.comm.Barrier()
            for step in range(STEPS):
                cached = prompt_len + step * rows
                metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[cached]
                )
                metadata.prepare()
                position_ids.copy_(torch.arange(cached, cached + rows, dtype=torch.int)[None])
                embeds.copy_(e.table[decode[step].cuda()])
                cuda_graph.replay()
                out[f"decode{step}"] = static_out.clone()
                _collect(out, f"decode{step}", static_layers)
        torch.cuda.synchronize()
        for step in range(STEPS):
            out[f"tokens{step}"] = _greedy(e, out[f"decode{step}"])
        out.update(_cache_bytes(kv, metadata, capacity))
        if plan is not None:
            # gemm_prefetch's fork and the named forks (qkv_prefetch's): none outstanding, none dropped, none
            # skipped (a fork inside a capture before its stream exists).
            forks = list(getattr(plan, "named_forks", {}).values())
            counts["prefetch_pending"] = int(plan.pending) + sum(int(f.pending) for f in forks)
            counts["prefetch_dropped"] = int(plan.dropped) + sum(
                f.dropped + f.skipped for f in forks
            )
    finally:
        e.sink.clear()
        del cuda_graph
        kv.shutdown()
    return out, counts


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    return tensor.view(
        {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.int8}[tensor.element_size()]
    )


def _differences(got: dict, want: dict) -> list:
    """Each output of ``got`` that is not ``want``'s, as bits."""
    found = []
    for name, reference in want.items():
        tensor = got.get(name)
        if tensor is None:
            found.append(f"{name}: missing")
        elif tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            found.append(
                f"{name}: {tuple(tensor.shape)} {tensor.dtype} vs "
                f"{tuple(reference.shape)} {reference.dtype}"
            )
        else:
            differ = int((_bits(tensor) != _bits(reference)).sum())
            if differ:
                delta = (tensor.float() - reference.float()).abs().max().item()
                found.append(f"{name}: {differ}/{tensor.numel()} differ, max |d| {delta:.3e}")
    extra = sorted(set(got) - set(want))
    if extra:
        found.append(f"unexpected outputs {extra[:4]}")
    return found


def _distance(got: dict, want: dict) -> dict:
    """How far a non-bit-identical arm's decode outputs are from the reference's."""
    num = den = 0.0
    final_num = final_den = 0.0
    max_abs = 0.0
    finite = True
    same_tokens = rows = 0
    caches_equal = {}
    for name, reference in want.items():
        if name.startswith("prefill") or name.startswith("cache/"):
            continue
        tensor = got[name]
        if name.startswith("tokens"):
            same_tokens += int((tensor == reference).sum())
            rows += tensor.numel()
            continue
        diff = (tensor.double() - reference.double()).pow(2).sum().item()
        size = reference.double().pow(2).sum().item()
        finite = finite and bool(torch.isfinite(tensor).all())
        num, den = num + diff, den + size
        max_abs = max(max_abs, (tensor.float() - reference.float()).abs().max().item())
        if "/" not in name:
            final_num, final_den = final_num + diff, final_den + size
    for name, reference in want.items():
        if name.startswith("cache/"):
            caches_equal[name] = bool(torch.equal(got[name], reference))
    return {
        "rel_rms": (num / den) ** 0.5 if den else 0.0,
        "final_rel_rms": (final_num / final_den) ** 0.5 if final_den else 0.0,
        "max_abs": max_abs,
        "finite": finite,
        "token_agreement": same_tokens / rows if rows else 1.0,
        "caches_equal": caches_equal,
    }


def _expected_counts(selected: frozenset, rows: int, graph: bool = False) -> dict:
    expected = {}
    for entry, (ids, sites, widest) in ENTRIES.items():
        active = bool(selected & ids) and (widest is None or rows <= widest)
        expected[entry] = sites if active else 0
    if expected.get("unpool_direct_boundary"):
        # unpool_direct arms the direct transport for a forward of 8 rows inside a capture only.
        if graph and rows == 8:
            expected["unpool_gather_boundary"] = 0
        else:
            expected["unpool_direct_boundary"] = 0
    if expected.get("l2_prefetch") and "early_release" in selected:
        # early_release drops gemm_prefetch's prefetch on the edges it converts, at the widths whose consumer
        # loads its weight before its wait (split-K QKV, from 3 rows): the unpooler's of the
        # next QKV weight behind every fused boundary (5 here; no DFlash capture layers in
        # this model).
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
            triggers as pdl_triggers,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
            dispatcher as dense_gemms,
        )

        kernels = dense_gemms.default_kernels()
        if "gathers" in selected and rows in pdl_triggers.weights_before_wait_rows(kernels["qkv"]):
            expected["l2_prefetch"] -= ENTRIES["unpool_gather_boundary"][1]
    if "moe_prologue" in selected and rows <= 8:
        for entry, sites in PROLOGUE_REPLACES.items():
            if expected.get(entry):
                expected[entry] -= sites
    if expected.get("qkv_prefetch_prefetch"):
        # qkv_prefetch prefetches for a QKV kernel that loads weight k-tiles after its wait (split-K).
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
            triggers as pdl_triggers,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
            dispatcher as dense_gemms,
        )

        if rows not in pdl_triggers.weights_before_wait_rows(dense_gemms.default_kernels()["qkv"]):
            expected["qkv_prefetch_prefetch"] = 0
    if "router_fork" in selected and rows <= 8:
        # router_fork launches the router shard on its side stream instead.
        expected["router_shard"] = 0
    if expected.get("pool_direct_route"):
        # pool_direct arms the direct pool transport for a forward of 8 rows inside a capture only.
        if graph and rows == 8:
            expected["router_shard_side"] = 0
            expected["pool_gather_route"] = 0
        else:
            expected["router_shard_publish"] = 0
            expected["pool_direct_route"] = 0
    if selected & _plans.PATH_IDS["swa_block"]:
        # swa_block takes the sliding-window layers' attention; swa_fused_prep also their prep,
        # swa_fused_gate their output gate (early_release's norm-gate kernel then runs only at
        # the full layers).
        fits = rows <= 8
        expected["paged_decode_attention"] = 0
        expected["swa_block_attention"] = SWA_BLOCK_SLIDING_SITES if fits else 0
        if "swa_fused_prep" in selected and fits:
            expected["preattn_prep"] = SWA_BLOCK_FULL_SITES
        if "swa_fused_gate" in selected and fits and expected.get("attention_norm_gate"):
            expected["attention_norm_gate"] = SWA_BLOCK_FULL_SITES
    # prep_append: the full-attention layers run the folded prep instead of preattn_prep's
    # (after swa_fused_prep, which leaves preattn_prep on exactly those layers);
    # prep_append_prefetch_trim (with the prep append on those layers): their QKV
    # projections fork no prefetch.
    full_layers = len(FULL_LAYER_INDICES)
    if "prep_append" in selected and "preattn_prep" in selected and rows <= 8:
        expected["preattn_prep"] -= full_layers
    if (
        "prep_append_prefetch_trim" in selected
        and "prep_append" in selected
        and rows <= 8
        and expected.get("l2_prefetch")
    ):
        expected["l2_prefetch"] -= full_layers
    # direct_publish takes exactly its width: there the consumer hop replaces the hop,
    # o_proj's GEMM runs through dense_gemm with the slot as its output (the producer
    # policy), and forks no gemm_prefetch prefetch of the pool weight (one per layer).
    direct = "direct_publish" in selected and rows == DIRECT_PUBLISH_ROWS
    expected["post_attn_consumer_hop"] = 6 if direct else 0
    if direct:
        expected["post_attn_hop"] = 0
        if expected.get("l2_prefetch"):
            expected["l2_prefetch"] -= len(CKPT_LAYERS)
    # split_kv's combine_norm_gate takes the full-attention layers' norm-gate.
    if expected.get("combine_norm_gate") and expected.get("attention_norm_gate"):
        expected["attention_norm_gate"] -= expected["combine_norm_gate"]
    # The prep-fold kernel takes the prep-append's, the elided preprocess's and full_attention_split_kv's place.
    if expected.get("full_attention_prep_fold"):
        for entry in PREP_FOLD_REPLACES:
            expected[entry] = 0
    return expected


def _compare_arms(e, rows: int, mode: str) -> None:
    """Every arm against its reference arm or the built-in model at ``rows`` (collective)."""
    graph = mode == "graph"
    runs = {}
    report = {}
    for arm, selection, _ in ARMS:
        runs[arm], counts = _run(e, e.models[arm], rows, graph)
        _RESULTS[(mode, arm, rows)] = {"counts": counts}
        e.comm.Barrier()
    again, _ = _run(e, e.models["builtin"], rows, graph)
    report["builtin (second run)"] = _differences(again, runs["builtin"])
    report["builtin (finite)"] = [
        f"{name}: not finite"
        for name, t in runs["builtin"].items()
        if t.is_floating_point() and not torch.isfinite(t).all()
    ]
    for arm in DETERMINISM:
        rerun, _ = _run(e, e.models[arm], rows, graph)
        report[f"{arm} (second run)"] = _differences(rerun, runs[arm])
    builtin_prefill = {k: v for k, v in runs["builtin"].items() if k.startswith("prefill")}
    for arm, selection, reference in ARMS[1:]:
        result = _RESULTS[(mode, arm, rows)]
        # No feature acts on a 2.1k-row prefill: its outputs and cache are the built-in's.
        found = _differences(
            {k: v for k, v in runs[arm].items() if k.startswith("prefill")}, builtin_prefill
        )
        # Every arm's distance from the built-in model is recorded (the table, and the
        # no-amplification gate's singles); an arm with a reference must also equal it.
        result["distance"] = _distance(runs[arm], runs["builtin"])
        if not result["distance"]["finite"]:
            found.append("an output is not finite")
        if reference is not None and rows in CHANGES_AT.get(arm, ()):
            # Bits change by design from model layer 1 on: equal upstream of it, and the
            # distance from the reference is recorded.
            upstream = {k: v for k, v in runs[reference].items() if UPSTREAM.match(k)}
            found += _differences(
                {k: v for k, v in runs[arm].items() if UPSTREAM.match(k)}, upstream
            )
            result["distance_from_reference"] = _distance(runs[arm], runs[reference])
            result["upstream_outputs_compared"] = len(upstream)
        elif reference is not None:
            found += _differences(runs[arm], runs[reference])
        expected = _expected_counts(_selection(selection), rows, graph)
        counts = result["counts"]
        least = STEPS if mode == "eager" else 1
        for entry, sites in expected.items():
            if counts.get(entry, 0) != sites * least:
                found.append(f"{entry} ran {counts.get(entry, 0)} times, expected {sites * least}")
        if counts.get("prefetch_pending") or counts.get("prefetch_dropped"):
            found.append(
                f"prefetch forks (gemm_prefetch, named) pending {counts.get('prefetch_pending')} "
                f"dropped {counts.get('prefetch_dropped')}"
            )
        if graph:
            want = _expected_boundary_edges(_selection(selection), rows)
            if counts.get("boundary_pdl_edges") != want:
                found.append(
                    f"{counts.get('boundary_pdl_edges')} programmatic edges out of the fused "
                    f"boundary in the captured graph, expected {want}"
                )
            if "qkv_prefetch" in _selection(selection) and reference is not None:
                # Each qkv_prefetch prefetch is a programmatic predecessor of its QKV GEMM;
                # nothing else of the reference arm's graph changes (a reference with
                # qkv_prefetch has them).
                base = _RESULTS[(mode, reference, rows)]["counts"].get("prefetch_pdl_edges", 0)
                ref_selection = next(s for a, s, _ in ARMS if a == reference)
                added = (
                    0
                    if "qkv_prefetch" in _selection(ref_selection)
                    else expected["qkv_prefetch_prefetch"]
                )
                want = base + added
                if counts.get("prefetch_pdl_edges") != want:
                    found.append(
                        f"{counts.get('prefetch_pdl_edges')} programmatic edges out of the L2 "
                        f"prefetches, expected {want} (the reference's {base} + qkv_prefetch's)"
                    )
        report[arm] = found
    for combined, singles in COMBINED.items():
        whole = _RESULTS[(mode, combined, rows)]["distance"]["rel_rms"]
        parts = [_RESULTS[(mode, single, rows)]["distance"]["rel_rms"] for single in singles]
        _RESULTS[(mode, combined, rows)]["singles_sum"] = sum(parts)
        if whole > MARGIN * sum(parts):
            report[combined].append(
                f"rel RMS {whole:.3e} > {MARGIN} x the singles' sum {sum(parts):.3e} {parts}"
            )
    everyone = e.comm.allgather({arm: found for arm, found in report.items() if found})
    failed = {
        f"rank {rank} {arm}": found[:6]
        for rank, bad in enumerate(everyone)
        for arm, found in bad.items()
    }
    assert not failed, f"{mode} at {rows} rows: {failed}"


def test_the_arms_have_the_expected_layout(env):
    """Every arm's modules are the classes its selection asks for, on every layer."""
    e = env
    wrong = []
    for arm, selection, _ in ARMS:
        selected = _selection(selection)
        preattn_prep_on, paged_decode_on = (
            "preattn_prep" in selected,
            bool(selected & {"paged_decode", "paged_decode_q8"}),
        )
        attention = {
            (True, True): "MuseSparkOpenTepPrepDecodeAttention",
            (True, False): "MuseSparkOpenTepAttention",
            (False, True): "MuseSparkOpenTepDecodeAttention",
            (False, False): "MuseSparkOpenAttention",
        }[(preattn_prep_on, paged_decode_on)]
        if "early_release" in selected:
            attention = {
                "MuseSparkOpenAttention": "MuseSparkOpenTepNormGateAttention",
                "MuseSparkOpenTepAttention": "MuseSparkOpenTepPrepNormGateAttention",
                "MuseSparkOpenTepDecodeAttention": "MuseSparkOpenTepDecodeNormGateAttention",
                "MuseSparkOpenTepPrepDecodeAttention": (
                    "MuseSparkOpenTepPrepDecodeNormGateAttention"
                ),
            }[attention]
        if selected & _plans.PATH_IDS["swa_block"]:
            attention += "SwaBlock"
        # prep_append re-classes the full-attention layers with FullAppendAttention in front.
        full_attention = attention + ("FullAppend" if "prep_append" in selected else "")
        # split_kv puts FullAttnSplitKvAttention in front of that, prep_fold
        # FullAttnSplitKvFoldAttention in front of split_kv's.
        full_attention += "V4" if "split_kv" in selected else ""
        full_attention += "Fold" if "prep_fold" in selected else ""
        qkv = (
            "MuseSparkOpenTepDenseLinear"
            if selected & {"dense_gemms", "gemm_prefetch"}
            else "Linear"
        )
        ring_buddy_on = bool(selected & _plans.LOADED)
        finalize = (
            "FinalizeTransportWorkspace"
            if "finalize_transport" in selected
            else "FinalizeIpcWorkspace"
        )
        prologue = "moe_prologue" in selected
        forked = "router_fork" in selected
        for index, row in enumerate(e.layout[arm]):
            attn, moe, linear, replicated, workspace, side = row
            want = full_attention if index in FULL_LAYER_INDICES else attention
            if (attn, linear, replicated) != (want, qkv, ring_buddy_on):
                wrong.append(f"{arm} layer {index}: {attn} {linear} replicated={replicated}")
            if workspace != finalize:
                wrong.append(f"{arm} layer {index}: finalize workspace {workspace}")
            if side != forked:
                wrong.append(f"{arm} layer {index}: router_fork side stream {side}")
            if prologue != (moe == "MuseSparkOpenTepPrologueMoE"):
                wrong.append(f"{arm} layer {index}: MoE class {moe}")
        stale_poll_on = {
            "hop": "stale_poll" in selected and bool(selected & {"hop", "hop_early_trigger"})
        }
        if e.stale_poll_on[arm] != stale_poll_on:
            wrong.append(
                f"{arm} polling schedules {e.stale_poll_on[arm]}, expected {stale_poll_on}"
            )
    assert not wrong, wrong


def test_tier7_full_layers_launch_the_spill_free_build(env):
    """``MODELING_V2_COMPOSE_TIER=tier7``: every full-attention layer of an arm with
    ``spill_free`` launches the split-KV kernel (``prep_fold``: its fold build) with
    ``rescale_cols`` 16, every other ``split_kv`` arm's with 64, and an arm with ``pool_direct``
    has a direct pool site on each of the four
    NVFP4-expert layers (the routed layers) and on no other."""
    if _SET != "tier7":
        pytest.skip("runs with MODELING_V2_COMPOSE_TIER=tier7")
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
        full_attention_split_kv as full_attention_split_kv,
    )

    e = env
    wrong = []
    for arm, selection, _ in ARMS:
        selected = _selection(selection)
        model = e.models[arm]
        if "split_kv" in selected:
            want = (
                full_attention_split_kv.SPILL_FREE_RESCALE_COLS
                if "spill_free" in selected
                else full_attention_split_kv.HEAD_DIM
            )
            for index in FULL_LAYER_INDICES:
                have = getattr(model.layers[index].self_attn, "_split_kv_rescale_cols", None)
                if have != want:
                    wrong.append(f"{arm} layer {index}: rescale_cols {have}, want {want}")
        pool = getattr(model, "_pool_direct", None)
        sites = 0 if pool is None else len(pool.sites)
        want_sites = ENTRIES["pool_direct_route"][1] if "pool_direct" in selected else 0
        if sites != want_sites:
            wrong.append(f"{arm}: {sites} direct pool sites, want {want_sites}")
    everyone = e.comm.allgather(wrong)
    assert not any(everyone), everyone


@pytest.mark.parametrize("rows", ROWS)
def test_every_arm_composes_in_eager_decode(env, rows):
    """Eager decode steps: bitwise arms equal their reference, the others stay within the
    no-amplification gate, every selected entry point runs where it must."""
    _compare_arms(env, rows, "eager")


@pytest.mark.parametrize("rows", ROWS)
def test_every_arm_composes_in_cuda_graphs(env, rows):
    """The same with the decode steps replayed from one CUDA graph per arm (the serving mode)."""
    _compare_arms(env, rows, "graph")


def test_spill_free_full_layers_launch_the_spill_free_build(env):
    """Every full-attention layer of an arm with ``spill_free`` launches the split-KV kernel
    (``prep_fold``: its fold build) with ``rescale_cols`` 16 (``SPILL_FREE_RESCALE_COLS``), every
    other ``split_kv`` arm's with 64 (the whole head). Runs where some arm selects
    ``spill_free``."""
    if not any("spill_free" in _selection(selection) for _, selection, _ in ARMS):
        pytest.skip("no arm of this set selects spill_free")
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
        full_attention_split_kv as full_attention_split_kv,
    )

    e = env
    wrong = []
    for arm, selection, _ in ARMS:
        selected = _selection(selection)
        if "split_kv" not in selected:
            continue
        want = (
            full_attention_split_kv.SPILL_FREE_RESCALE_COLS
            if "spill_free" in selected
            else full_attention_split_kv.HEAD_DIM
        )
        for index in FULL_LAYER_INDICES:
            have = getattr(e.models[arm].layers[index].self_attn, "_split_kv_rescale_cols", None)
            if have != want:
                wrong.append(f"{arm} layer {index}: rescale_cols {have}, want {want}")
    everyone = e.comm.allgather(wrong)
    assert not any(everyone), everyone


def test_write_the_result_table(env):
    """Rank 0 writes the per-arm, per-width table (counts, distances) when asked to."""
    e = env
    assert _RESULTS, "the composition tests recorded nothing"
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and e.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        table = {
            f"{mode}:{arm}@{rows}": result for (mode, arm, rows), result in sorted(_RESULTS.items())
        }
        table["layout"] = {arm: layout for arm, layout in e.layout.items()}
        table["stale_poll_schedules"] = e.stale_poll_on
        table["arm_set"] = _SET
        table["hook_stream"] = "side" if HOOK_SIDE else "main"
        table["free_gib_after_build"] = e.free_gib
        with open(os.path.join(out_dir, f"compose_{_SET}.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True, default=str)
