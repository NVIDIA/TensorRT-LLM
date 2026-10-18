# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the composed fast-path sets under a serving-like schedule, on real weights (TP8).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, the target
checkpoint and the DFlash 2 drafter checkpoint, under the serving switches::

    MUSESPARK_REFERENCE_CKPT=<Muse-Spark-Open-816B-A42B-NVFP4 checkpoint> \\
    MUSESPARK_REFERENCE_DRAFTER=<Muse-Spark-Open-DFlash2 checkpoint> \\
    MUSESPARK_OPEN_ALLREDUCE=MNNVL MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1 \\
    MUSESPARK_OPEN_SHARDED_PROJECTIONS=1 TRTLLM_LOW_M_GEMM_BACKEND=auto TRTLLM_ENABLE_PDL=1 \\
    srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>

Anywhere else every test skips.

The model is the combined test's (``test_modeling_v2_musespark_open_decode_forward.py``):
checkpoint layers 0, 1, 2, 59, 60 and 61 with their real weights, the FP8
``KVCacheManagerV2`` with 32-token pages, one request whose decode steps carry M rows,
and the load-time expert placement (``ring_buddy``) installed in the shell's order. On top
of it this test runs what the combined test does not:

* Short prefills. Each width's state first prefills a fresh request of 3, 8 or 13
  tokens on its own KV cache -- the prefill row counts at which the dense GEMMs
  (``dense_gemms`` / ``gemm_prefetch``, at most 8 rows) and ``ring_buddy`` (at most 16)
  act on a prefill step, as on a short suffix chunk after a KV-cache hit; their entry
  points are counted there to show they do -- then the decode request (2100 + 3M tokens).
* The draft head in the graph. One CUDA graph per width M <= 8 holds the target's
  decode forward, then the DFlash 2 draft head on the target's output (built-in, or
  the vocab-sharded ``draft_head`` with its key exchange) and the drafter's ``fc``
  projection of five layers' outputs (built-in, or the sharded ``fc_shard`` with its
  own projection gather), from the drafter checkpoint's selector, ``fc`` and
  ``hidden_norm`` weights.
* A schedule. ``STEPS`` steps in bursts of 8: graphs of different widths replayed
  back to back, one step in seven eager, no host sync inside a burst, new embedding
  rows and anchors every step (device copies); at a burst boundary the host syncs
  and every state that ran moves on by a seeded accepted count, so rejected rows are
  rewritten by later steps.
* Delayed ranks. A device-side spin kernel at ``SITES`` -- before each fused
  collective, on the prepare stream, on the ``gemm_prefetch`` side stream, before
  the head and before the ``fc_shard`` gather -- delays a seeded rank by 30 us, 300 us
  or 2 ms on most steps, and one rank sleeps 4 ms on the host before every fourth step
  (seed 0: no spin kernel in the graph at all).

An arm is a row ``(arm, model, ids, head, reference arm, delay seed)`` of the selected
table. Every output (the short prefill and its cache bytes, the prefill, each step's final
output, layer outputs, head tokens and ``fc`` projection, and every layer's K/V bytes at
the end) of an arm must equal its reference arm's bit for bit on every rank, whatever the
delays: a bit-identical set equals the built-in model, and a target set equals the
non-bit-identical core it is built on (``n_all`` / ``n_nof``:
``ring_buddy,paged_decode_q8,gemm_prefetch``). An arm without a reference is itself one.
A budgeted set (one that changes values by design) is its own reference: its undelayed
run (seed 0) is the reference of its delayed runs, and ``PARTIAL`` names the core it must
still equal on every output outside the regex it is exempt from. A negative seed draws no
delay kernel, but one rank launches every other step late from the host, so the graph
between two hop / finalize calls is the all-PDL one under a late rank. Each feature's
entry points are counted on every capture (``ENTRIES``, ``_expected_counts``);
``gemm_prefetch`` must leave no prefetch fork pending or dropped.

The delay probe replays one M = 8 graph per arm of ``PROBE_ARMS`` while rank 3 spins 2 ms
at one site at a time: a site the arm's graph passes must hold every rank back by it (the
joins and collectives behind it are real graph edges), a site it does not pass must not,
and every replay must give the undelayed replay's bits.

``MODELING_V2_COMPOSE_TIER`` selects the arm table (``ARMS``; default ``tier1``):

* ``tier1`` (``TIER1_ARMS``): the built-in model, the bit-identical set
  ``preattn_prep,hop,gathers`` and the full set ``TIER1_TARGET`` against the core ``n_all``.
* ``tier2`` (``TIER2_ARMS``): ``TIER2_NEW`` (``early_release``, ``persistent_experts``,
  ``stale_poll``) is bit-identical, so the bit-identical set with it equals the built-in
  model and ``TIER1_TARGET`` + ``TIER2_NEW`` equals the core. ``tier2_reseeded``
  (``TIER2R_ARMS``) runs the same sets under other delay seeds.
* ``tier3`` (``TIER3_ARMS``): ``TIER3_NEW`` adds the bit-identical ``finalize_transport``
  and ``moe_prologue``: ``srv_bitid5`` equals the built-in model, the ``ring_buddy`` chains
  with ``persistent_experts`` and ``finalize_transport`` or ``moe_prologue`` equal
  ``ring_buddy`` (a rank delayed before the layer-2 expert call and at the layer-3 pool
  projection, which ``moe_prologue`` puts on the prologue's one stream), and
  ``srv_full5_nof`` equals the core. With ``moe_prologue`` the decode steps of at most
  8 rows do not pass ``pool_prep_L1`` (``prepare_expert_input`` belongs to the two-stream
  prep it replaces).
* ``swa_block`` (``SWA_BLOCK_ARMS``): the tier-1 and tier-2 target sets alone and with the
  sliding-window block ids (all three, ``swa_prewait`` alone, ``swa_fused_gate`` alone),
  each equal to the core. Their graphs must hold the programmatic edges the block is built
  on (``test_the_swa_block_block_is_a_programmatic_chain``). That edge census is what any
  change that adds an in-graph writer of the buffers the split reads before its wait must
  re-run (``swa_attention_block``'s docstring).
* ``tier4`` (``TIER4_ARMS``): the sliding-window block ids, ``prep_append``,
  ``prep_append_prefetch_trim`` and ``direct_publish`` are bit-identical: ``srv_bitid6``
  equals the built-in model and ``srv_full6_nof`` the core. At 8 rows ``direct_publish``'s
  o_proj publishes into the hop transport and the consumer hop replaces the post-attention
  hop (counted; its consecutive calls are in the transitivity census), and layer 2's
  o_proj forks no prefetch, so the probe's ``prefetch_L2`` is not passed there.
* ``unpool_direct``, ``unpool_direct_tier3``, ``unpool_direct_tier4``
  (``UNPOOL_DIRECT_ARMS``, ``UNPOOL_DIRECT_TIER3_ARMS``, ``UNPOOL_DIRECT_TIER4_ARMS``):
  the unpool GEMM publishing into the direct boundary transport on the 8-row graphs,
  between eager steps and graphs of other widths on the gathers' tagged inbox, must leave
  the bits of the bit-identical set with ``gemm_prefetch`` and of the tier-2 / tier-3 /
  tier-4 target set under every delay pattern (a rank late before its unpool GEMM or
  before its boundary). On the tier-4 sets o_proj publishes into the hop transport and
  the unpool GEMM into the boundary transport in the same 8-row graph.
* ``qkv_prefetch``, ``qkv_prefetch_tier4`` (``QKV_PREFETCH_ARMS``,
  ``QKV_PREFETCH_TIER4_ARMS``): the next layer's QKV weight L2-prefetched on a named fork
  after each expert call (layer 0's at the forward's start), joined only by that QKV GEMM's
  dispatcher, must leave the bits of the tier-3 / tier-4 target set and of that set with
  ``unpool_direct``. These tables add the delay site ``qkv_prefetch_L2`` (the prefetch's
  side stream, before layer 2's QKV prefetch), which the probe must find holding every
  rank. At 3..8 rows each prefetch adds one programmatic edge (into its QKV GEMM) over the
  same set's graph without it; a short prefill of 3 or 8 rows forks it after each
  converted layer's experts.
* ``three_windows`` (``TIER3_ARMS`` + ``THREE_WINDOWS_ARMS``): the tier-3 target set and
  the ``moe_prologue`` chain with the third ring window, which regroups the expert
  partials as ``ring_buddy`` does, so these arms are their own references.
* ``early_ids`` (``EARLY_IDS_ARMS``): the early expert ids are bit-identical: the
  ``moe_prologue`` chain with them equals ``ring_buddy`` and the tier-3 target set with them
  equals the core (the expert site delays a rank in front of the layer-2 expert call, which
  ``early_ids`` launches before the routed kernel ends); every such arm's four NVFP4 layers
  launch the expert kernel early at up to 8 rows. ``early_ids_three_windows``
  (``EARLY_IDS_THREE_WINDOWS_ARMS``) carries the three-window placement through the early
  ids: every arm with both equals the ``three_windows`` arm without ``early_ids``.
* ``tier5`` (``TIER5_ARMS``): ``srv_full7b`` (the tier-4 target set with the bit-identical
  ``hop_early_trigger,unpool_direct,early_ids,qkv_prefetch,bf16_routing``) must equal the
  core. ``srv_full7`` = ``srv_full7b`` + ``three_windows,split_kv`` is budgeted (the third
  ring window regroups the expert partials; the split-KV partials replace the
  full-attention layers' FMHA) and must equal the core outside ``TIER5_CHANGED``. Under
  ``split_kv`` every captured graph of up to 8 rows runs ``full_attention_split_kv`` and
  ``combine_norm_gate`` on both full-attention layers; the transitivity census relies on
  ``combine_norm_gate``'s Triton kernel waiting before it exits, and enters
  ``full_attention_split_kv`` from the prep-append, which never triggers early.
  ``split_kv_tier4`` (``SPLIT_KV_TIER4_ARMS``) checks the tier-4 target set + ``split_kv``
  the same way (``SPLIT_KV_CHANGED``).
* ``router_fork``, ``router_fork_tier5`` (``ROUTER_FORK_ARMS``, ``ROUTER_FORK_TIER5_ARMS``):
  ``hop_early_trigger`` and ``router_fork`` (the router shard forked onto a side stream as a
  plain launch, joined before the routed pool kernel) are bit-identical, on the tier-4 sets
  and on ``srv_full7b`` (with and without ``three_windows``). The delay site
  ``router_side_L3`` spins in front of layer 3's router on the side stream: the routed pool
  kernel's wait must hold every rank back by it. The side launches' edges are checked by
  ``test_the_router_fork_side_launches_are_forked_from_the_hop_and_joined_into_the_routed_pool``;
  a side launch is a ``SIDE_LAUNCHES`` kernel of the transitivity census, like the
  ``gemm_prefetch`` prefetch.
* ``prep_fold`` (``PREP_FOLD_ARMS``): ``srv_full7`` with ``prep_fold`` (the prep-append
  folded into the split-KV kernel, ``full_attention_prep_fold``), with ``spill_free`` (the
  fold build rescaling O in 16-column chunks), with both, all bit-identical to
  ``srv_full7``: every run must give ``srv_full7``'s undelayed bits. Under ``prep_fold``
  every captured graph of up to 8 rows runs ``full_attention_prep_fold`` and
  ``combine_norm_gate`` on both full-attention layers and neither the prep-append nor
  ``full_attention_split_kv``; the fold kernel is entered from the QKV projection, which
  triggers early, so the census relies on its own wait before exit (``WAIT_BEFORE_EXIT``,
  audited on its SASS).
* ``pool_direct`` (``POOL_DIRECT_ARMS``): on the 8-row graphs the pool GEMM and the router
  shard's publish twin write into the routed pool kernel's direct transport, which
  ``pool_direct_route`` polls. It is bit-identical: ``srv_full7b`` + ``router_fork`` +
  ``pool_direct`` equals the core, and ``srv_full7`` + ``router_fork`` + ``pool_direct``
  equals ``srv_full7`` + ``router_fork`` undelayed. On the 8-row graphs each side launch is
  the publish twin, joined into ``PoolDirectRouteKernel``; the twin is a ``SIDE_LAUNCHES``
  kernel and the direct routed kernel a ``WAIT_BEFORE_EXIT`` one (its dry-launch exit is in
  ``LAUNCH_UNIFORM_EXITS``).
* ``tier7`` (``TIER7_ARMS``): ``srv_full8n`` (``srv_full7`` + ``router_fork``) and
  ``srv_full9`` (+ ``prep_fold,spill_free,pool_direct``) must give ``srv_full7``'s undelayed
  bits, and ``srv_full9b`` (``srv_full7b`` + ``router_fork`` + ``pool_direct``) the core's.
  The census, the SASS audit (``FullAttnSplitKvFoldKernel`` in its 16-column build,
  ``PoolDirectRouteKernel``) and the all-PDL negative control run with every tier-7 path
  on; the QKV-prefetch edges of ``TIER7_QKV_PREFETCH_ARMS`` are compared against
  ``srv_full9_no_qkv_prefetch``.

``pool_direct`` and ``tier7`` capture their graphs on a stream of their own
(``DEDICATED_CAPTURE_SETS``).

The per-layer outputs are copied on a side stream forked after each layer and
joined after the model call, so no kernel of this test sits between a layer's
fused post-FFN boundary and the next layer's QKV GEMM: a copy there would make that
edge a full dependency, and ``early_release``'s early launch of the QKV GEMM would never
happen. The captured graphs are read back (their programmatic edges,
``cuGraphGetEdges``): every M <= 8 graph of an arm with the fused unpool boundary
must have a programmatic edge out of each of its five boundaries, and no other
graph any. ``MODELING_V2_HOOK_STREAM=main`` copies on the capturing stream instead,
the negative control of that check: no graph may then have such an edge.

Between two consecutive post-attention hop calls (fused finalize calls) of a graph, call
n is complete when call n + 1's wait returns if some path between them has every kernel
entered through a programmatic edge executing ``griddepcontrol.wait`` before it
exits (``_census``). Where no plain launch is left between the two calls -- with
``moe_prologue``, ``persistent_experts`` and ``hop`` on the NVFP4-expert layers -- that
path must run only through kernels of ``WAIT_BEFORE_EXIT``, or kernels entered from one of
``NO_EARLY_TRIGGER`` (which launch only once its thread blocks have exited, after its
wait). With CuTe DSL cubin dumps
(``CUTE_DSL_KEEP=cubin`` / ``CUTE_EXPERIMENTAL_DSL_KEEP=cubin`` and their
``*_DUMP_DIR``) rank 0 also audits each of them as this run compiled it (the dumps,
the Triton cache; cuBLAS's ``nvjet`` from ``MODELING_V2_SASS_AUDIT_EXTRA``, an audit of
its Nsight Compute SASS): some warp leader of CTA 0 executes ``ACQBULK`` on every path
before it can exit, so no launch completes before its predecessor
(``_sass_wait_audit``); a launch-uniform early exit must be a known one
(``LAUNCH_UNIFORM_EXITS``); a ``NO_EARLY_TRIGGER`` kernel has no ``PREEXIT``.

``MODELING_V2_COMPOSE_OUT=<dir>`` also writes rank 0's results there as JSON;
``MODELING_V2_SCHEDULE_STEPS`` sets the schedule length (default 96),
``MODELING_V2_TRACE_RANK`` the delay probe's delayed rank (default 3) and
``MODELING_V2_INPUT_SEED`` the seed of the real embedding rows the inputs are drawn
from (default 2026).
"""

import hashlib
import importlib
import json
import os
import random
import re
import time
import types

import pytest
import torch
import tqdm

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

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
    dedicated_stream,
    edge_table,
    hook_on_side_stream,
    programmatic_edges,
)
from _sass_wait_audit import collect as collect_sass  # noqa: E402

# As tests/unittest/conftest.py does, which the multi-rank runs skip (--noconftest).
tqdm.tqdm.monitor_interval = 0

WORLD = 8
CKPT_LAYERS = (0, 1, 2, 59, 60, 61)
TOKENS_PER_BLOCK = 32
TOKEN_POOL = 64
HIDDEN = 8192
VOCAB, TOP_K, SELECTOR_RANK = 202048, 16, 256
ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16)
#: Widest decode step that runs the draft head (K + 1 at K = 7).
HEAD_MAX_ROWS = 8
PROMPT_BASE = 2100
BURST = 8
#: Target layers whose outputs feed the drafter's ``fc`` projection.
FC_LAYERS = 5
STEPS = int(os.environ.get("MODELING_V2_SCHEDULE_STEPS", "96"))
REQUIRED_ENV = {
    "MUSESPARK_OPEN_ALLREDUCE": "MNNVL",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL": "1",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS": "1",
    "TRTLLM_LOW_M_GEMM_BACKEND": "auto",
}

#: Delay sites, by index into the device-side delay table.
SITES = (
    "hop_L2",  # layer 2 add_attn join (the post-attention hop, or after the built-in o_proj all-reduce)
    "pool_prep_L1",  # layer 1 prepare_expert_input (prepare stream; before the fused pool gather)
    "pool_proj_L3",  # layer 3 pool projection (prepare stream; the gemm_prefetch join is its GEMM)
    "unpool_proj_L3",  # layer 3 unpool projection (main stream)
    "f1_L4",  # layer 4 add_ffn join (the fused unpool boundary)
    "hop_L5",  # last layer's add_attn join
    "experts_L2",  # layer 2 experts call (ring_buddy's stacked call; then the fused finalize AR)
    "attn_L3",  # layer 3 attention op (the paged decode kernels)
    "prefetch_L2",  # gemm_prefetch side stream, before layer 2's o_proj -> pool prefetch
    "head",  # before the draft head (draft_head's key exchange / the built-in all-gather)
    "fc_gather",  # before fc_shard's projection gather
)

#: (arm, model, the arm's ids (``_compose_plans.plan_of_ids``), head, reference arm, delay pattern seed).
#: head: "builtin" = built-in head and fc projection; "draft_head" = the sharded draft head,
#: built-in projection; "draft_head_fc_shard" = the sharded draft head and the sharded fc. Seed
#: 0: no delay kernel in the graph. A negative seed: no delay kernel either, but one rank (drawn
#: per step) launches every other step 4 ms late from the host, so the graph between two hop /
#: finalize calls is the all-PDL one (with moe_prologue) under a late rank. An arm without a
#: reference is one (its outputs are kept).
TIER1_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("builtin_again", "builtin", None, "builtin", "builtin", 2),
    ("srv_bitid", "srv_bitid", "preattn_prep,hop,gathers", "draft_head", "builtin", 1),
    ("srv_bitid_nd", "srv_bitid", "preattn_prep,hop,gathers", "draft_head", "builtin", 0),
    ("n_all", "n_all", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    (
        "srv_full",
        "srv_full",
        "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch",
        "draft_head_fc_shard",
        "n_all",
        1,
    ),
    (
        "srv_full_d2",
        "srv_full",
        "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch",
        "draft_head_fc_shard",
        "n_all",
        2,
    ),
    (
        "srv_full_nd",
        "srv_full",
        "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch",
        "draft_head_fc_shard",
        "n_all",
        0,
    ),
)
#: The tier-2 fast paths on this model, and the tier-1 target set.
TIER2_NEW = "early_release,persistent_experts,stale_poll"
TIER1_TARGET = "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch"
TIER2_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    (
        "srv_bitid4",
        "srv_bitid4",
        "preattn_prep,hop,gathers,early_release,stale_poll",
        "draft_head",
        "builtin",
        1,
    ),
    (
        "srv_bitid4_d2",
        "srv_bitid4",
        "preattn_prep,hop,gathers,early_release,stale_poll",
        "draft_head",
        "builtin",
        2,
    ),
    ("ring_buddy", "ring_buddy", "ring_buddy", "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    (
        "srv_full4_nof",
        "srv_full4_nof",
        TIER1_TARGET + "," + TIER2_NEW,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
)
TIER2R_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 3),
    (
        "srv_bitid4_d6",
        "srv_bitid4",
        "preattn_prep,hop,gathers,early_release,stale_poll",
        "draft_head",
        "builtin",
        6,
    ),
    ("ring_buddy", "ring_buddy", "ring_buddy", "builtin", None, 3),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 3),
    (
        "srv_full4_nof_d8",
        "srv_full4_nof",
        TIER1_TARGET + "," + TIER2_NEW,
        "draft_head_fc_shard",
        "n_nof",
        8,
    ),
)
#: The tier-3 target fast paths on this model (tier 2's with finalize_transport and moe_prologue).
TIER3_NEW = "early_release,finalize_transport,persistent_experts,moe_prologue,stale_poll"
_BITID5 = "preattn_prep,hop,gathers,early_release,finalize_transport,moe_prologue,stale_poll"
_FULL5_NOF = TIER1_TARGET + "," + TIER3_NEW
TIER3_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid5", "srv_bitid5", _BITID5, "draft_head", "builtin", 1),
    ("srv_bitid5_d2", "srv_bitid5", _BITID5, "draft_head", "builtin", 2),
    ("ring_buddy", "ring_buddy", "ring_buddy", "builtin", None, 1),
    (
        "ring_buddy_persistent_experts_finalize_transport",
        "ring_buddy_persistent_experts_finalize_transport",
        "ring_buddy,persistent_experts,finalize_transport",
        "builtin",
        "ring_buddy",
        1,
    ),
    (
        "ring_buddy_persistent_experts_finalize_transport_d2",
        "ring_buddy_persistent_experts_finalize_transport",
        "ring_buddy,persistent_experts,finalize_transport",
        "builtin",
        "ring_buddy",
        2,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue",
        "gathers_ring_buddy_persistent_experts_moe_prologue",
        "gathers,ring_buddy,persistent_experts,moe_prologue",
        "builtin",
        "ring_buddy",
        1,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_d2",
        "gathers_ring_buddy_persistent_experts_moe_prologue",
        "gathers,ring_buddy,persistent_experts,moe_prologue",
        "builtin",
        "ring_buddy",
        2,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    (
        "srv_full4_nof",
        "srv_full4_nof",
        TIER1_TARGET + "," + TIER2_NEW,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full5_nof_d2", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 2),
    ("srv_full5_nof_nd", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 0),
    ("srv_full5_nof_h3", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", -3),
    ("srv_bitid5_h4", "srv_bitid5", _BITID5, "draft_head", "builtin", -4),
)
#: The sliding-window block's arms (``swa_block``): the tier-1 and tier-2 target sets, and the
#: block ids on top (all three, swa_prewait alone, swa_fused_gate alone).
_SWA_BLOCK = ",swa_prewait,swa_fused_prep,swa_fused_gate"
_SRV3_NOF = TIER1_TARGET
_SRV4_NOF = TIER1_TARGET + "," + TIER2_NEW
SWA_BLOCK_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    (
        "srv_bitid4",
        "srv_bitid4",
        "preattn_prep,hop,gathers,early_release,stale_poll",
        "draft_head",
        "builtin",
        1,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv3_nof", "srv3_nof", _SRV3_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv3_nof_swa_block",
        "srv3_nof_swa_block",
        _SRV3_NOF + _SWA_BLOCK,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv3_nof_swa_block_nd",
        "srv3_nof_swa_block",
        _SRV3_NOF + _SWA_BLOCK,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    ("srv4_nof", "srv4_nof", _SRV4_NOF, "draft_head_fc_shard", "n_nof", 1),
    ("srv4_nof_nd", "srv4_nof", _SRV4_NOF, "draft_head_fc_shard", "n_nof", 0),
    (
        "srv4_nof_swa_block",
        "srv4_nof_swa_block",
        _SRV4_NOF + _SWA_BLOCK,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv4_nof_swa_block_d2",
        "srv4_nof_swa_block",
        _SRV4_NOF + _SWA_BLOCK,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv4_nof_swa_block_nd",
        "srv4_nof_swa_block",
        _SRV4_NOF + _SWA_BLOCK,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv4_nof_swa_prewait",
        "srv4_nof_swa_prewait",
        _SRV4_NOF + ",swa_prewait",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv4_nof_swa_prewait_nd",
        "srv4_nof_swa_prewait",
        _SRV4_NOF + ",swa_prewait",
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv4_nof_swa_fused_gate",
        "srv4_nof_swa_fused_gate",
        _SRV4_NOF + ",swa_fused_gate",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
)
#: The tier-4 arms (``tier4``): the tier-4 target fast paths on this model (capture_fold needs
#: DFlash capture layers, the drafter-side ids are not on it) on the tier-3 sets.
_BITID6 = _BITID5 + ",prep_append"
_FULL6_NOF = (
    _FULL5_NOF
    + ",swa_prewait,swa_fused_prep,swa_fused_gate,prep_append,prep_append_prefetch_trim,direct_publish"
)
TIER4_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid6", "srv_bitid6", _BITID6, "draft_head", "builtin", 1),
    ("srv_bitid6_d2", "srv_bitid6", _BITID6, "draft_head", "builtin", 2),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full6_nof_d2", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 2),
    ("srv_full6_nof_nd", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 0),
    ("srv_full6_nof_h3", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", -3),
    ("srv_bitid6_h4", "srv_bitid6", _BITID6, "draft_head", "builtin", -4),
)
#: The direct unpool publish's arms (``unpool_direct``): the bit-identical set with
#: gemm_prefetch (and early_release), and the tier-2 target set, each with and without the
#: direct transport, under three delay patterns.
TIER2_TARGET = TIER1_TARGET + "," + TIER2_NEW
_BI12 = "preattn_prep,hop,gathers,gemm_prefetch,early_release,stale_poll"
_FULL_UNPOOL_DIRECT = TIER2_TARGET + ",unpool_direct"
UNPOOL_DIRECT_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    (
        "srv_bitid4",
        "srv_bitid4",
        "preattn_prep,hop,gathers,early_release,stale_poll",
        "draft_head",
        "builtin",
        1,
    ),
    ("bi12", "bi12", _BI12, "draft_head", None, 1),
    ("bi12_unpool_direct", "bi12_unpool_direct", _BI12 + ",unpool_direct", "draft_head", "bi12", 1),
    (
        "bi12_unpool_direct_d2",
        "bi12_unpool_direct",
        _BI12 + ",unpool_direct",
        "draft_head",
        "bi12",
        2,
    ),
    (
        "bi12_unpool_direct_nd",
        "bi12_unpool_direct",
        _BI12 + ",unpool_direct",
        "draft_head",
        "bi12",
        0,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full4_nof", "srv_full4_nof", TIER2_TARGET, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full4_nof_unpool_direct",
        "srv_full4_nof_unpool_direct",
        _FULL_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full4_nof_unpool_direct_d2",
        "srv_full4_nof_unpool_direct",
        _FULL_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full4_nof_unpool_direct_nd",
        "srv_full4_nof_unpool_direct",
        _FULL_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
)
#: The QKV prefetch's arms (``qkv_prefetch``): the tier-3 target set, and that set with
#: unpool_direct, each with and without the QKV prefetch, under four delay patterns.
_FULL5_UNPOOL_DIRECT = _FULL5_NOF + ",unpool_direct"
QKV_PREFETCH_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full5_nof_qkv_prefetch",
        "srv_full5_nof_qkv_prefetch",
        _FULL5_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full5_nof_qkv_prefetch_d2",
        "srv_full5_nof_qkv_prefetch",
        _FULL5_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full5_nof_qkv_prefetch_nd",
        "srv_full5_nof_qkv_prefetch",
        _FULL5_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full5_nof_qkv_prefetch_h3",
        "srv_full5_nof_qkv_prefetch",
        _FULL5_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
    (
        "srv_full5_unpool_direct",
        "srv_full5_unpool_direct",
        _FULL5_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full5_unpool_direct_qkv_prefetch",
        "srv_full5_unpool_direct_qkv_prefetch",
        _FULL5_UNPOOL_DIRECT + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full5_unpool_direct_qkv_prefetch_d2",
        "srv_full5_unpool_direct_qkv_prefetch",
        _FULL5_UNPOOL_DIRECT + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
)
#: The direct unpool publish on the tier-3 sets (``unpool_direct_tier3``), under in-graph
#: delays (1, 2), none (0) and a rank that launches late from the host on the all-PDL graph (-3).
_BI5P = "preattn_prep,hop,gathers,gemm_prefetch,early_release,finalize_transport,moe_prologue,stale_poll"
_FULL5_UNPOOL_DIRECT = _FULL5_NOF + ",unpool_direct"
UNPOOL_DIRECT_TIER3_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid5", "srv_bitid5", _BITID5, "draft_head", "builtin", 1),
    ("bi5p", "bi5p", _BI5P, "draft_head", None, 1),
    ("bi5p_unpool_direct", "bi5p_unpool_direct", _BI5P + ",unpool_direct", "draft_head", "bi5p", 1),
    (
        "bi5p_unpool_direct_d2",
        "bi5p_unpool_direct",
        _BI5P + ",unpool_direct",
        "draft_head",
        "bi5p",
        2,
    ),
    (
        "bi5p_unpool_direct_nd",
        "bi5p_unpool_direct",
        _BI5P + ",unpool_direct",
        "draft_head",
        "bi5p",
        0,
    ),
    (
        "bi5p_unpool_direct_h3",
        "bi5p_unpool_direct",
        _BI5P + ",unpool_direct",
        "draft_head",
        "bi5p",
        -3,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full5_nof_unpool_direct",
        "srv_full5_nof_unpool_direct",
        _FULL5_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full5_nof_unpool_direct_d2",
        "srv_full5_nof_unpool_direct",
        _FULL5_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full5_nof_unpool_direct_nd",
        "srv_full5_nof_unpool_direct",
        _FULL5_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full5_nof_unpool_direct_h3",
        "srv_full5_nof_unpool_direct",
        _FULL5_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
)
#: The direct unpool publish on the tier-4 sets (``unpool_direct_tier4``): the bit-identical
#: tier-3 set with gemm_prefetch, prep_append and direct_publish, and the tier-4 target set
#: (direct_publish in it), each with and without unpool_direct, under in-graph delays (1, 2),
#: none (0) and a rank that launches late from the host on the all-PDL graph (-3).
_BI6P = _BI5P + ",prep_append,direct_publish"
_FULL6_UNPOOL_DIRECT = _FULL6_NOF + ",unpool_direct"
UNPOOL_DIRECT_TIER4_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid6", "srv_bitid6", _BITID6, "draft_head", "builtin", 1),
    ("bi6p", "bi6p", _BI6P, "draft_head", None, 1),
    ("bi6p_unpool_direct", "bi6p_unpool_direct", _BI6P + ",unpool_direct", "draft_head", "bi6p", 1),
    (
        "bi6p_unpool_direct_d2",
        "bi6p_unpool_direct",
        _BI6P + ",unpool_direct",
        "draft_head",
        "bi6p",
        2,
    ),
    (
        "bi6p_unpool_direct_nd",
        "bi6p_unpool_direct",
        _BI6P + ",unpool_direct",
        "draft_head",
        "bi6p",
        0,
    ),
    (
        "bi6p_unpool_direct_h3",
        "bi6p_unpool_direct",
        _BI6P + ",unpool_direct",
        "draft_head",
        "bi6p",
        -3,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full6_nof_unpool_direct",
        "srv_full6_nof_unpool_direct",
        _FULL6_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full6_nof_unpool_direct_d2",
        "srv_full6_nof_unpool_direct",
        _FULL6_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full6_nof_unpool_direct_nd",
        "srv_full6_nof_unpool_direct",
        _FULL6_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full6_nof_unpool_direct_h3",
        "srv_full6_nof_unpool_direct",
        _FULL6_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
)
#: The third ring window's arms: the tier-3 target set and the moe_prologue chain with it.
_FULL5_THREE_WINDOWS = _FULL5_NOF + ",three_windows"
_CHAIN_THREE_WINDOWS = "gathers,ring_buddy,persistent_experts,moe_prologue,three_windows"
THREE_WINDOWS_ARMS = (
    (
        "srv_full5_three_windows",
        "srv_full5_three_windows",
        _FULL5_THREE_WINDOWS,
        "draft_head_fc_shard",
        None,
        1,
    ),
    (
        "srv_full5_three_windows_d2",
        "srv_full5_three_windows",
        _FULL5_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        2,
    ),
    (
        "srv_full5_three_windows_nd",
        "srv_full5_three_windows",
        _FULL5_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        0,
    ),
    (
        "srv_full5_three_windows_h3",
        "srv_full5_three_windows",
        _FULL5_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        -3,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        _CHAIN_THREE_WINDOWS,
        "builtin",
        None,
        1,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows_d2",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        _CHAIN_THREE_WINDOWS,
        "builtin",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        2,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows_h4",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        _CHAIN_THREE_WINDOWS,
        "builtin",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        -4,
    ),
)
#: The early expert ids' arms (``early_ids``): on ring_buddy's MoE path fed by moe_prologue,
#: and on the tier-3 target set; delayed, not, and host-delayed.
_FULL5_EARLY_IDS = _FULL5_NOF + ",early_ids"
EARLY_IDS_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid5", "srv_bitid5", _BITID5, "draft_head", "builtin", 1),
    ("ring_buddy", "ring_buddy", "ring_buddy", "builtin", None, 1),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids",
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids",
        "gathers,ring_buddy,persistent_experts,moe_prologue,early_ids",
        "builtin",
        "ring_buddy",
        1,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_d2",
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids",
        "gathers,ring_buddy,persistent_experts,moe_prologue,early_ids",
        "builtin",
        "ring_buddy",
        2,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full5_early_ids",
        "srv_full5_early_ids",
        _FULL5_EARLY_IDS,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full5_early_ids_d2",
        "srv_full5_early_ids",
        _FULL5_EARLY_IDS,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full5_early_ids_nd",
        "srv_full5_early_ids",
        _FULL5_EARLY_IDS,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full5_early_ids_h3",
        "srv_full5_early_ids",
        _FULL5_EARLY_IDS,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
)
#: The early expert ids with the third ring window (``early_ids_three_windows``): the early ids
#: publish the three-window placement (one ``_placed_expert`` expression with the ``local_ids``
#: store) and persistent_experts' early start runs the 96-expert window; early_ids changes no
#: value, so every arm with both equals the three_windows arm without early_ids bit for bit,
#: delayed, without a delay kernel, and with a rank launching late from the host on the
#: all-PDL graph.
_FULL5_EARLY_IDS_THREE_WINDOWS = _FULL5_THREE_WINDOWS + ",early_ids"
_CHAIN_EARLY_IDS_THREE_WINDOWS = _CHAIN_THREE_WINDOWS + ",early_ids"
EARLY_IDS_THREE_WINDOWS_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid5", "srv_bitid5", _BITID5, "draft_head", "builtin", 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full5_nof", "srv_full5_nof", _FULL5_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full5_three_windows",
        "srv_full5_three_windows",
        _FULL5_THREE_WINDOWS,
        "draft_head_fc_shard",
        None,
        1,
    ),
    (
        "srv_full5_early_ids_three_windows",
        "srv_full5_early_ids_three_windows",
        _FULL5_EARLY_IDS_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        1,
    ),
    (
        "srv_full5_early_ids_three_windows_d2",
        "srv_full5_early_ids_three_windows",
        _FULL5_EARLY_IDS_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        2,
    ),
    (
        "srv_full5_early_ids_three_windows_nd",
        "srv_full5_early_ids_three_windows",
        _FULL5_EARLY_IDS_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        0,
    ),
    (
        "srv_full5_early_ids_three_windows_h3",
        "srv_full5_early_ids_three_windows",
        _FULL5_EARLY_IDS_THREE_WINDOWS,
        "draft_head_fc_shard",
        "srv_full5_three_windows",
        -3,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        _CHAIN_THREE_WINDOWS,
        "builtin",
        None,
        1,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows",
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows",
        _CHAIN_EARLY_IDS_THREE_WINDOWS,
        "builtin",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        1,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows_d2",
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows",
        _CHAIN_EARLY_IDS_THREE_WINDOWS,
        "builtin",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        2,
    ),
    (
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows_h4",
        "gathers_ring_buddy_persistent_experts_moe_prologue_early_ids_three_windows",
        _CHAIN_EARLY_IDS_THREE_WINDOWS,
        "builtin",
        "gathers_ring_buddy_persistent_experts_moe_prologue_three_windows",
        -4,
    ),
)
#: The QKV prefetch on the tier-4 sets (``qkv_prefetch_tier4``): the tier-4 target set, and
#: that set with unpool_direct, each with and without the QKV prefetch, under four delay patterns.
QKV_PREFETCH_TIER4_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full6_nof_qkv_prefetch",
        "srv_full6_nof_qkv_prefetch",
        _FULL6_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full6_nof_qkv_prefetch_d2",
        "srv_full6_nof_qkv_prefetch",
        _FULL6_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full6_nof_qkv_prefetch_nd",
        "srv_full6_nof_qkv_prefetch",
        _FULL6_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full6_nof_qkv_prefetch_h3",
        "srv_full6_nof_qkv_prefetch",
        _FULL6_NOF + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
    (
        "srv_full6_unpool_direct",
        "srv_full6_unpool_direct",
        _FULL6_UNPOOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full6_unpool_direct_qkv_prefetch",
        "srv_full6_unpool_direct_qkv_prefetch",
        _FULL6_UNPOOL_DIRECT + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full6_unpool_direct_qkv_prefetch_d2",
        "srv_full6_unpool_direct_qkv_prefetch",
        _FULL6_UNPOOL_DIRECT + ",qkv_prefetch",
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
)
#: The tier-5 arms (``tier5``): srv_full7b (the tier-4 target set + the bit-identical tier-5
#: paths) against the non-bit-identical core under every delay pattern, and srv_full7 =
#: srv_full7b + three_windows, split_kv (budgeted), whose undelayed run is the reference of its
#: delayed ones.
_FULL7B = _FULL6_NOF + ",hop_early_trigger,unpool_direct,early_ids,qkv_prefetch,bf16_routing"
_FULL7 = _FULL7B + ",three_windows,split_kv"
TIER5_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid6", "srv_bitid6", _BITID6, "draft_head", "builtin", 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full7b", "srv_full7b", _FULL7B, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full7b_d2", "srv_full7b", _FULL7B, "draft_head_fc_shard", "n_nof", 2),
    ("srv_full7b_nd", "srv_full7b", _FULL7B, "draft_head_fc_shard", "n_nof", 0),
    ("srv_full7b_h3", "srv_full7b", _FULL7B, "draft_head_fc_shard", "n_nof", -3),
    ("srv_full7_nd", "srv_full7", _FULL7, "draft_head_fc_shard", None, 0),
    ("srv_full7", "srv_full7", _FULL7, "draft_head_fc_shard", "srv_full7_nd", 1),
    ("srv_full7_d2", "srv_full7", _FULL7, "draft_head_fc_shard", "srv_full7_nd", 2),
    ("srv_full7_h3", "srv_full7", _FULL7, "draft_head_fc_shard", "srv_full7_nd", -3),
)
#: The outputs the budgeted tier-5 paths change by design (``_record`` / ``_cache_bytes``
#: names): on decode steps of up to 8 rows the final output, the head's tokens and fc
#: projection and the outputs of layers 1-5 (model layer 1 is the first NVFP4-expert layer
#: and the first full-attention layer), and those requests' KV cache bytes of layers 2-5.
#: Layer 0 (BF16 experts, sliding window), the K/V it feeds layer 1, the prefills and every
#: wider step are unchanged.
TIER5_CHANGED = re.compile(
    r"^(s\d+:\w+@[1-8](/(tokens|proj|layer[1-5]\.\d+))?|cache@[1-8]/layer[2-5]\.[kv])$"
)
#: Arm -> (a kept arm it must also equal, the outputs exempt from that comparison): the
#: non-bit-identical core, which srv_full7b equals bit for bit (its own arm here).
PARTIAL = {"srv_full7_nd": ("n_nof", TIER5_CHANGED)}
#: The router fork's arms (``router_fork``): hop_early_trigger and router_fork on the tier-4 sets.
_BITID6_ROUTER_FORK = _BITID6.replace("hop,", "hop_early_trigger,") + ",router_fork"
_FULL6_ROUTER_FORK = _FULL6_NOF.replace("hop,", "hop_early_trigger,") + ",router_fork"
ROUTER_FORK_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    (
        "srv_bitid6_router_fork",
        "srv_bitid6_router_fork",
        _BITID6_ROUTER_FORK,
        "draft_head",
        "builtin",
        1,
    ),
    (
        "srv_bitid6_router_fork_h4",
        "srv_bitid6_router_fork",
        _BITID6_ROUTER_FORK,
        "draft_head",
        "builtin",
        -4,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full6_router_fork",
        "srv_full6_router_fork",
        _FULL6_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full6_router_fork_d2",
        "srv_full6_router_fork",
        _FULL6_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full6_router_fork_nd",
        "srv_full6_router_fork",
        _FULL6_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full6_router_fork_h3",
        "srv_full6_router_fork",
        _FULL6_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
)
#: The router fork on the tier-5 sets (``router_fork_tier5``): srv_full7b (hop_early_trigger
#: and early_ids in it) + router_fork against the core, and srv_full7b + three_windows +
#: router_fork against srv_full7b + three_windows' undelayed run (three_windows is budgeted;
#: router_fork changes no value, so it must give that run's bits), under every delay pattern.
_FULL7B_ROUTER_FORK = _FULL7B + ",router_fork"
_FULL7B_THREE_WINDOWS = _FULL7B + ",three_windows"
ROUTER_FORK_TIER5_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    (
        "srv_bitid6_router_fork",
        "srv_bitid6_router_fork",
        _BITID6_ROUTER_FORK,
        "draft_head",
        "builtin",
        1,
    ),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full7b", "srv_full7b", _FULL7B, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full7b_router_fork",
        "srv_full7b_router_fork",
        _FULL7B_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full7b_router_fork_d2",
        "srv_full7b_router_fork",
        _FULL7B_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full7b_router_fork_nd",
        "srv_full7b_router_fork",
        _FULL7B_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full7b_router_fork_h3",
        "srv_full7b_router_fork",
        _FULL7B_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
    (
        "srv_full7b_three_windows_nd",
        "srv_full7b_three_windows",
        _FULL7B_THREE_WINDOWS,
        "draft_head_fc_shard",
        None,
        0,
    ),
    (
        "srv_full7b_three_windows_router_fork",
        "srv_full7b_three_windows_router_fork",
        _FULL7B_THREE_WINDOWS + ",router_fork",
        "draft_head_fc_shard",
        "srv_full7b_three_windows_nd",
        1,
    ),
    (
        "srv_full7b_three_windows_router_fork_nd",
        "srv_full7b_three_windows_router_fork",
        _FULL7B_THREE_WINDOWS + ",router_fork",
        "draft_head_fc_shard",
        "srv_full7b_three_windows_nd",
        0,
    ),
    (
        "srv_full7b_three_windows_router_fork_h3",
        "srv_full7b_three_windows_router_fork",
        _FULL7B_THREE_WINDOWS + ",router_fork",
        "draft_head_fc_shard",
        "srv_full7b_three_windows_nd",
        -3,
    ),
)
PARTIAL["srv_full7b_three_windows_nd"] = ("n_nof", TIER5_CHANGED)
_FULL7_ROUTER_FORK = _FULL7 + ",router_fork"
#: The direct pool transport's arms (``pool_direct``): router_fork and router_fork +
#: pool_direct on srv_full7b (against the core) and on srv_full7 (pool_direct against
#: srv_full7 + router_fork undelayed), under every delay pattern.
_FULL7B_POOL_DIRECT = _FULL7B_ROUTER_FORK + ",pool_direct"
_FULL7_POOL_DIRECT = _FULL7_ROUTER_FORK + ",pool_direct"
POOL_DIRECT_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    (
        "srv_full7b_router_fork",
        "srv_full7b_router_fork",
        _FULL7B_ROUTER_FORK,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full7b_pool_direct",
        "srv_full7b_pool_direct",
        _FULL7B_POOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        1,
    ),
    (
        "srv_full7b_pool_direct_d2",
        "srv_full7b_pool_direct",
        _FULL7B_POOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        2,
    ),
    (
        "srv_full7b_pool_direct_nd",
        "srv_full7b_pool_direct",
        _FULL7B_POOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        0,
    ),
    (
        "srv_full7b_pool_direct_h3",
        "srv_full7b_pool_direct",
        _FULL7B_POOL_DIRECT,
        "draft_head_fc_shard",
        "n_nof",
        -3,
    ),
    (
        "srv_full7_router_fork_nd",
        "srv_full7_router_fork",
        _FULL7_ROUTER_FORK,
        "draft_head_fc_shard",
        None,
        0,
    ),
    (
        "srv_full7_pool_direct",
        "srv_full7_pool_direct",
        _FULL7_POOL_DIRECT,
        "draft_head_fc_shard",
        "srv_full7_router_fork_nd",
        1,
    ),
    (
        "srv_full7_pool_direct_d2",
        "srv_full7_pool_direct",
        _FULL7_POOL_DIRECT,
        "draft_head_fc_shard",
        "srv_full7_router_fork_nd",
        2,
    ),
    (
        "srv_full7_pool_direct_h3",
        "srv_full7_pool_direct",
        _FULL7_POOL_DIRECT,
        "draft_head_fc_shard",
        "srv_full7_router_fork_nd",
        -3,
    ),
)
PARTIAL["srv_full7_router_fork_nd"] = ("n_nof", TIER5_CHANGED)
#: The tier-7 arms (``tier7``): srv_full8n (``_FULL7_ROUTER_FORK``) and srv_full9 = srv_full8n +
#: prep_fold, spill_free, pool_direct against srv_full7's undelayed run (none of them changes a
#: value: each must give srv_full7's bits everywhere, and srv_full7 equals the core where
#: three_windows / split_kv change nothing, ``PARTIAL``) under every delay pattern; srv_full9b
#: (srv_full7b + router_fork + pool_direct) against the core. srv_full9 without qkv_prefetch is
#: the graph the prefetch -> QKV edge test compares srv_full9's with.
_FULL9 = _FULL7_ROUTER_FORK + ",prep_fold,spill_free,pool_direct"
TIER7_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full9b", "srv_full9b", _FULL7B_POOL_DIRECT, "draft_head_fc_shard", "n_nof", 1),
    ("srv_full9b_h3", "srv_full9b", _FULL7B_POOL_DIRECT, "draft_head_fc_shard", "n_nof", -3),
    ("srv_full7_nd", "srv_full7", _FULL7, "draft_head_fc_shard", None, 0),
    ("srv_full8n_nd", "srv_full8n", _FULL7_ROUTER_FORK, "draft_head_fc_shard", "srv_full7_nd", 0),
    ("srv_full8n", "srv_full8n", _FULL7_ROUTER_FORK, "draft_head_fc_shard", "srv_full7_nd", 1),
    ("srv_full9_nd", "srv_full9", _FULL9, "draft_head_fc_shard", "srv_full7_nd", 0),
    ("srv_full9", "srv_full9", _FULL9, "draft_head_fc_shard", "srv_full7_nd", 1),
    ("srv_full9_d2", "srv_full9", _FULL9, "draft_head_fc_shard", "srv_full7_nd", 2),
    ("srv_full9_d3", "srv_full9", _FULL9, "draft_head_fc_shard", "srv_full7_nd", 3),
    ("srv_full9_h3", "srv_full9", _FULL9, "draft_head_fc_shard", "srv_full7_nd", -3),
    (
        "srv_full9_no_qkv_prefetch",
        "srv_full9_no_qkv_prefetch",
        _FULL9.replace(",qkv_prefetch", ""),
        "draft_head_fc_shard",
        "srv_full7_nd",
        1,
    ),
)
#: Tier-7 arms whose set the QKV-prefetch edge test must find with and without qkv_prefetch.
TIER7_QKV_PREFETCH_ARMS = ("srv_full9",)
_SET = os.environ.get("MODELING_V2_COMPOSE_TIER", "tier1")
if _SET in (
    "qkv_prefetch",
    "qkv_prefetch_tier4",
    "tier5",
    "router_fork_tier5",
    "prep_fold",
    "pool_direct",
    "tier7",
):
    # qkv_prefetch's side stream, before the prefetch of layer 2's QKV weight (after layer 1's experts).
    SITES = SITES + ("qkv_prefetch_L2",)
if _SET in ("router_fork", "router_fork_tier5", "pool_direct", "tier7"):
    # router_fork's side stream, before layer 3's router shard (pool_direct: or its publish twin).
    SITES = SITES + ("router_side_L3",)
#: The split-KV attention on the tier-4 sets (``split_kv_tier4``): the tier-4 target set
#: (bit-identical to the core) and the same set + split_kv (budgeted), whose undelayed run is
#: the reference of its delayed ones.
_FULL6_SPLIT_KV = _FULL6_NOF + ",split_kv"
SPLIT_KV_TIER4_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("srv_bitid6", "srv_bitid6", _BITID6, "draft_head", "builtin", 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full6_nof", "srv_full6_nof", _FULL6_NOF, "draft_head_fc_shard", "n_nof", 1),
    (
        "srv_full6_split_kv_nd",
        "srv_full6_split_kv",
        _FULL6_SPLIT_KV,
        "draft_head_fc_shard",
        None,
        0,
    ),
    (
        "srv_full6_split_kv",
        "srv_full6_split_kv",
        _FULL6_SPLIT_KV,
        "draft_head_fc_shard",
        "srv_full6_split_kv_nd",
        1,
    ),
    (
        "srv_full6_split_kv_d2",
        "srv_full6_split_kv",
        _FULL6_SPLIT_KV,
        "draft_head_fc_shard",
        "srv_full6_split_kv_nd",
        2,
    ),
    (
        "srv_full6_split_kv_h3",
        "srv_full6_split_kv",
        _FULL6_SPLIT_KV,
        "draft_head_fc_shard",
        "srv_full6_split_kv_nd",
        -3,
    ),
)
#: The prep fold's arms (``prep_fold``): srv_full7, and srv_full7 + prep_fold (bit-identical to
#: srv_full7), whose every run under every delay pattern must give srv_full7's undelayed bits.
#: srv_full7 + prep_fold + spill_free (the fold build rescaling O in 16-column chunks, the same
#: values; delay seeds none, 1, 2 and a late rank) and srv_full7 + spill_free (none, 1 and a
#: late rank) must give them too; the census then audits the 16-column fold build next to the
#: whole-head one.
_FULL7_PREP_FOLD = _FULL7 + ",prep_fold"
_FULL7_PREP_FOLD_SPILL_FREE = _FULL7_PREP_FOLD + ",spill_free"
_FULL7_SPILL_FREE = _FULL7 + ",spill_free"
PREP_FOLD_ARMS = (
    ("builtin", "builtin", None, "builtin", None, 1),
    ("n_nof", "n_nof", "ring_buddy,paged_decode_q8,gemm_prefetch", "draft_head_fc_shard", None, 1),
    ("srv_full7_nd", "srv_full7", _FULL7, "draft_head_fc_shard", None, 0),
    ("srv_full7", "srv_full7", _FULL7, "draft_head_fc_shard", "srv_full7_nd", 1),
    (
        "srv_full7_prep_fold_nd",
        "srv_full7_prep_fold",
        _FULL7_PREP_FOLD,
        "draft_head_fc_shard",
        "srv_full7_nd",
        0,
    ),
    (
        "srv_full7_prep_fold",
        "srv_full7_prep_fold",
        _FULL7_PREP_FOLD,
        "draft_head_fc_shard",
        "srv_full7_nd",
        1,
    ),
    (
        "srv_full7_prep_fold_d2",
        "srv_full7_prep_fold",
        _FULL7_PREP_FOLD,
        "draft_head_fc_shard",
        "srv_full7_nd",
        2,
    ),
    (
        "srv_full7_prep_fold_h3",
        "srv_full7_prep_fold",
        _FULL7_PREP_FOLD,
        "draft_head_fc_shard",
        "srv_full7_nd",
        -3,
    ),
    (
        "srv_full7_prep_fold_spill_free_nd",
        "srv_full7_prep_fold_spill_free",
        _FULL7_PREP_FOLD_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        0,
    ),
    (
        "srv_full7_prep_fold_spill_free",
        "srv_full7_prep_fold_spill_free",
        _FULL7_PREP_FOLD_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        1,
    ),
    (
        "srv_full7_prep_fold_spill_free_d2",
        "srv_full7_prep_fold_spill_free",
        _FULL7_PREP_FOLD_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        2,
    ),
    (
        "srv_full7_prep_fold_spill_free_h3",
        "srv_full7_prep_fold_spill_free",
        _FULL7_PREP_FOLD_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        -3,
    ),
    (
        "srv_full7_spill_free_nd",
        "srv_full7_spill_free",
        _FULL7_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        0,
    ),
    (
        "srv_full7_spill_free",
        "srv_full7_spill_free",
        _FULL7_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        1,
    ),
    (
        "srv_full7_spill_free_h3",
        "srv_full7_spill_free",
        _FULL7_SPILL_FREE,
        "draft_head_fc_shard",
        "srv_full7_nd",
        -3,
    ),
)
#: The outputs split_kv changes by design: the same names as ``TIER5_CHANGED`` (model layer 1
#: is the first full-attention layer).
SPLIT_KV_CHANGED = TIER5_CHANGED
PARTIAL["srv_full6_split_kv_nd"] = ("n_nof", SPLIT_KV_CHANGED)
ARMS = {
    "tier1": TIER1_ARMS,
    "tier2": TIER2_ARMS,
    "tier2_reseeded": TIER2R_ARMS,
    "tier3": TIER3_ARMS,
    "swa_block": SWA_BLOCK_ARMS,
    "tier4": TIER4_ARMS,
    "unpool_direct": UNPOOL_DIRECT_ARMS,
    "qkv_prefetch": QKV_PREFETCH_ARMS,
    "unpool_direct_tier3": UNPOOL_DIRECT_TIER3_ARMS,
    "unpool_direct_tier4": UNPOOL_DIRECT_TIER4_ARMS,
    "three_windows": TIER3_ARMS + THREE_WINDOWS_ARMS,
    "early_ids": EARLY_IDS_ARMS,
    "early_ids_three_windows": EARLY_IDS_THREE_WINDOWS_ARMS,
    "qkv_prefetch_tier4": QKV_PREFETCH_TIER4_ARMS,
    "tier5": TIER5_ARMS,
    "split_kv_tier4": SPLIT_KV_TIER4_ARMS,
    "router_fork": ROUTER_FORK_ARMS,
    "router_fork_tier5": ROUTER_FORK_TIER5_ARMS,
    "prep_fold": PREP_FOLD_ARMS,
    "pool_direct": POOL_DIRECT_ARMS,
    "tier7": TIER7_ARMS,
}[_SET]
#: Arms of the delay probe, and its delayed rank and delay.
PROBE_ARMS = {
    "tier1": ("builtin", "srv_bitid", "srv_full"),
    "tier2": ("builtin", "srv_bitid4", "srv_full4_nof"),
    "tier3": ("builtin", "srv_bitid5", "srv_full5_nof"),
    "unpool_direct": ("builtin", "bi12_unpool_direct", "srv_full4_nof_unpool_direct"),
    "qkv_prefetch": (
        "builtin",
        "srv_full5_nof_qkv_prefetch",
        "srv_full5_unpool_direct_qkv_prefetch",
    ),
    "unpool_direct_tier3": ("builtin", "bi5p_unpool_direct", "srv_full5_nof_unpool_direct"),
    "three_windows": ("builtin", "srv_bitid5", "srv_full5_nof", "srv_full5_three_windows"),
    "tier2_reseeded": ("builtin", "srv_bitid4_d6", "srv_full4_nof_d8"),
    "swa_block": ("srv4_nof", "srv4_nof_swa_block"),
    "tier4": ("builtin", "srv_bitid6", "srv_full6_nof"),
    "unpool_direct_tier4": ("builtin", "bi6p_unpool_direct", "srv_full6_nof_unpool_direct"),
    "early_ids": ("builtin", "srv_full5_early_ids"),
    "early_ids_three_windows": (
        "builtin",
        "srv_bitid5",
        "srv_full5_three_windows",
        "srv_full5_early_ids_three_windows",
    ),
    "qkv_prefetch_tier4": (
        "builtin",
        "srv_full6_nof_qkv_prefetch",
        "srv_full6_unpool_direct_qkv_prefetch",
    ),
    "tier5": ("builtin", "srv_bitid6", "srv_full7b", "srv_full7_nd"),
    "prep_fold": ("builtin", "srv_full7_nd", "srv_full7_prep_fold_nd"),
    "split_kv_tier4": ("builtin", "srv_full6_nof", "srv_full6_split_kv_nd"),
    "router_fork": ("builtin", "srv_full6_nof", "srv_full6_router_fork"),
    "router_fork_tier5": (
        "builtin",
        "srv_full7b",
        "srv_full7b_router_fork",
        "srv_full7b_three_windows_router_fork",
    ),
    "pool_direct": ("builtin", "srv_full7b_router_fork", "srv_full7b_pool_direct"),
    "tier7": ("builtin", "srv_full9b", "srv_full8n_nd", "srv_full9_nd"),
}[_SET]
#: The full target set and the target-bit-identical set, whose layout is checked.
LAYOUT_ARMS = {
    "tier1": ("srv_full", "srv_bitid"),
    "tier2": ("srv_full4_nof", "srv_bitid4"),
    "tier3": ("srv_full5_nof", "srv_bitid5"),
    "unpool_direct": ("srv_full4_nof_unpool_direct", "srv_bitid4"),
    "qkv_prefetch": ("srv_full5_unpool_direct_qkv_prefetch", "builtin"),
    "unpool_direct_tier3": ("srv_full5_nof_unpool_direct", "srv_bitid5"),
    "three_windows": ("srv_full5_nof", "srv_bitid5"),
    "tier2_reseeded": ("srv_full4_nof", "srv_bitid4"),
    "swa_block": ("srv4_nof_swa_block", "srv_bitid4"),
    "tier4": ("srv_full6_nof", "srv_bitid6"),
    "unpool_direct_tier4": ("srv_full6_nof_unpool_direct", "srv_bitid6"),
    "early_ids": ("srv_full5_early_ids", "srv_bitid5"),
    "early_ids_three_windows": ("srv_full5_early_ids_three_windows", "srv_bitid5"),
    "qkv_prefetch_tier4": ("srv_full6_unpool_direct_qkv_prefetch", "builtin"),
    "tier5": ("srv_full7", "srv_bitid6"),
    "prep_fold": ("srv_full7_prep_fold", "builtin"),
    "split_kv_tier4": ("srv_full6_split_kv", "srv_bitid6"),
    "router_fork": ("srv_full6_router_fork", "srv_bitid6_router_fork"),
    "router_fork_tier5": ("srv_full7b_router_fork", "srv_bitid6_router_fork"),
    "pool_direct": ("srv_full7_pool_direct", "builtin"),
    "tier7": ("srv_full9", "builtin"),
}[_SET]
#: Model layers 1 and 5 (checkpoint 1 and 61) are the full-attention layers.
FULL_LAYER_INDICES = (1, 5)
PROBE_RANK = int(os.environ.get("MODELING_V2_TRACE_RANK", "3"))
PROBE_NS, PROBE_REPLAYS = 2_000_000, 8
#: Copy the per-layer outputs on a side stream (``MODELING_V2_HOOK_STREAM``, ``_graph_edges``).
HOOK_SIDE = hook_on_side_stream()
#: (arm, rows) -> the programmatic edges of that arm's captured graph at ``rows``.
_PDL_EDGES: dict = {}
_CURRENT_ARM = [None]

#: (arm, rows) -> the edge census of that arm's captured decode graph at that width.
_GRAPH_CENSUS: dict = {}
#: Kernel symbol -> threads per block, from the captured graphs (for the SASS audit).
_BLOCK_THREADS: dict = {}
#: Kernels that open a post-attention hop call (or direct_publish's consumer hop) / a fused
#: finalize call (built-in or finalize_transport's).
_HOP_KERNEL = re.compile(r"PostAttn(Hop|Consumer)")
_FINALIZE_KERNEL = re.compile(r"[Ff]inalize")
#: Kernels (short names, ``_short_name``) whose SASS executes ``griddepcontrol.wait`` on
#: every path to an exit, so that entered through a programmatic edge each completes
#: only after its predecessor: this tree's and flashinfer's CuTe DSL kernels by class
#: (every variant a run compiles is audited; ``direct_publish``'s consumer hop among them), the
#: Triton router shard, ``prep_append``'s prep-append and ``early_release``'s PDL norm-gate (a
#: full-attention layer's chain once ``prep_append_prefetch_trim`` drops its QKV-site prefetch),
#: and cuBLAS's
#: attention output projection kernel by its exact symbol (another cuBLAS build or
#: heuristic choice is a new kernel to audit), and the split-K pair cuBLAS runs for the
#: 1-row BF16 GEMMs that gemm_prefetch would otherwise take (pool slice, QKV, unpool: split-K
#: ``nvjet`` + ``splitKreduce_kernel``; like ``nvjet``, read with Nsight Compute and
#: passed in ``MODELING_V2_SASS_AUDIT_EXTRA``). ``test_the_waits_between_calls_are_audited_
#: kernels`` checks the SASS of the ones on the paths it relies on. unpool_direct's direct
#: boundary takes the fused unpool boundary's place on the 8-row graphs; its variants are
#: audited the same way. ``split_kv``'s ``combine_norm_gate`` takes a full-attention layer's norm-gate (its Triton
#: kernel waits before any read but ``seq_lens``); ``full_attention_split_kv`` before it is entered
#: from the prep-append (``NO_EARLY_TRIGGER``), so it launches only after the prep's blocks
#: exited. It is not listed here: its key-less CTAs may exit before their wait (an exit
#: that depends on the ``rows`` parameter, which the SASS walker cannot bound), so its own
#: SASS does not pass the wait-before-exit audit. ``prep_fold``'s ``full_attention_prep_fold`` kernel
#: (``FullAttnSplitKvFoldKernel``, a class of its own, so its symbol is not the op's build's) is
#: entered from the QKV projection, which triggers early: every one of its threads waits
#: before it exits, and it is listed.
WAIT_BEFORE_EXIT = re.compile(
    r"^(_router_shard_kernel|PoolGatherRouteKernel|PersistentExpertsKernel"
    r"|FinalizeTransportKernel|UnpoolGatherBoundaryKernel|UnpoolDirectBoundaryKernel"
    r"|SplitKDenseGemmKernel|DirectDenseGemmKernel|nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"
    r"|nvjet_sm103_tst_64x8_64x16_4x1_v_bz_splitK_TNT"
    r"|_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif13__nv_bfloat16fS1_Lb0EfS1"
    r"|PostAttnConsumerKernel|_preattn_prep_append_kernel|_attention_norm_gate_pdl_kernel"
    r"|_combine_norm_gate_pdl_kernel|FullAttnSplitKvFoldKernel|PoolDirectRouteKernel)$"
)
#: Kernels (short names) whose SASS never executes ``griddepcontrol.launch_dependents``
#: (``PREEXIT``): a kernel entered through a programmatic edge from one of them launches
#: only after every thread block of it has exited (the implicit trigger, CUDA
#: Programming Guide, "Programmatic Dependent Launch"), so it completes after that
#: predecessor's own wait returned whether or not it waits itself. ``prep_append``'s
#: prep-append: the trtllm-gen decode FMHA after it may exit without
#: waiting (a thread block whose KV range is empty does), and needs no audit. Only an
#: edge out of the source's programmatic port (``PORT_PROGRAMMATIC``, which activates
#: once every block has triggered or exited) carries this; a launch-order port would not.
NO_EARLY_TRIGGER = re.compile(r"^(_preattn_prep_append_kernel)$")
#: ``CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC``: the source port of a programmatic edge that
#: stream capture of a ``cudaLaunchAttributeProgrammaticStreamSerialization`` launch makes.
PORT_PROGRAMMATIC = 1
#: Plain launches that leave a pair of calls "with no plain launch between them":
#: gemm_prefetch's and qkv_prefetch's L2 prefetch forks (side branches; nothing but the join
#: into the next GEMM and the fork stream's next prefetch follows one), and router_fork's
#: router shard (a side branch whose successors are the routed pool kernel, which waits, and
#: the next side launch, by a full edge; ``test_the_router_fork_side_launches_are_forked_from_
#: the_hop_and_joined_into_the_routed_pool`` checks that shape; pool_direct's publish twin
#: takes its place on the 8-row graphs, joined into the direct routed kernel the same way).
SIDE_LAUNCHES = frozenset(
    {
        "_l2_prefetch_kernel",
        "_router_shard_side_kernel",
        "_router_shard_publish_kernel",
    }
)
#: router_fork's side launch of the router shard (``ops/moe/router_shard.router_shard_side``).
ROUTER_FORK_SIDE_KERNEL = "_router_shard_side_kernel"
#: pool_direct's side launch on the 8-row graphs (``ops/moe/router_shard.router_shard_publish``).
POOL_DIRECT_SIDE_KERNEL = "_router_shard_publish_kernel"
#: (arm, rows) -> the edges into and out of each router_fork side launch of that captured graph.
_FORK_EDGES: dict = {}
#: (arm, rows) -> whether some layer's MoE prepare stream (the model's pooled ``aux_stream``) was
#: the capturing stream of that graph: its prepare chain is then in-order work on the forward's
#: stream, and the graph's edges are those of a model without the fork.
_PREPARE_ALIASED: dict = {}
#: The sets whose graphs are captured on a stream of their own (``_graph_edges.dedicated_stream``).
#: ``torch.cuda.graph`` otherwise captures on a pooled stream, which a model built after enough
#: other pooled-stream draws also gets as its MoE prepare stream; that model's prepare chain then
#: runs in order on the forward's stream and its graphs differ from the set's other arms at the
#: two BF16-expert layers. The ``pool_direct`` and ``tier7`` sets are captured there, so that their fork /
#: join edge readings come from graphs without an aliased prepare stream. The other sets keep
#: the pooled stream.
DEDICATED_CAPTURE_SETS = ("pool_direct", "tier7")
_CAPTURE_STREAM: list = []


def _capture_stream():
    """The stream ``_State.capture`` captures on: None (PyTorch's pooled default) but for
    ``DEDICATED_CAPTURE_SETS`` (``MODELING_V2_CAPTURE_STREAM=pooled`` keeps the pooled default there
    too, a diagnostic run)."""
    if (
        _SET not in DEDICATED_CAPTURE_SETS
        or os.environ.get("MODELING_V2_CAPTURE_STREAM") == "pooled"
    ):
        return None
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(dedicated_stream())
    return _CAPTURE_STREAM[0]


def _prepare_stream_aliased(model) -> bool:
    """Whether some layer's MoE prepare stream is the current (capturing) stream."""
    current = torch.cuda.current_stream().cuda_stream
    return any(
        getattr(getattr(layer.feed_forward, "_prepare_stream", None), "cuda_stream", None)
        == current
        for layer in model.layers
    )


def _short_name(raw: str) -> str:
    """A kernel's class name (``...Kernel``) when its symbol carries one, else its symbol's head."""
    camel = re.findall(r"[A-Z][A-Za-z0-9]*Kernel", raw)
    return camel[-1] if camel else raw[:72]


def _graph_nodes_and_edges(graph):
    """``(names, raws, edges)`` of a kept CUDA graph: ``names`` {node: short kernel name or
    <node type>} in node order, ``raws`` {node: full symbol}, ``edges`` [(from, to, programmatic)]
    with ``programmatic`` 0 for a full edge, else the edge's source port (``PORT_PROGRAMMATIC``,
    the launch-order port, or -1 for an edge that names no port)."""
    from cuda.bindings import driver

    g = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(g, 0)
    _, nodes, _ = driver.cuGraphGetNodes(g, int(count))
    names, raws = {}, {}
    for node in nodes:
        key = int(node)
        _, kind = driver.cuGraphNodeGetType(node)
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            names[key] = raws[key] = f"<{kind.name}>"
            continue
        _, params = driver.cuGraphKernelNodeGetParams(node)
        status, raw = driver.cuFuncGetName(params.func)
        if status != driver.CUresult.CUDA_SUCCESS:
            status, raw = driver.cuKernelGetName(params.kern)
        raws[key] = raw.decode() if isinstance(raw, bytes) else str(raw)
        names[key] = _short_name(raws[key])
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(get(g, 0)[-1])
    _, frm, to, data, _ = get(g, total)
    programmatic = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    edges = [
        (int(a), int(b), (int(d.from_port) or -1) if int(d.type) == programmatic else 0)
        for a, b, d in zip(frm, to, data)
    ]
    return names, raws, edges


def _reach(start, step) -> set:
    seen, todo = {start}, [start]
    while todo:
        for nxt in step(todo.pop()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _without_delays(names: dict, raws: dict, edges: list):
    """The graph without the test's delay kernels (``_spin``): each one's predecessors are joined
    to its successors by the edge type into the successor (the successor's launch attribute), as
    the model launches them when no delay kernel sits in between."""
    spins = [n for n in names if raws[n] == "_spin" or raws[n].startswith("_spin_")]
    for node in spins:
        ins = [a for a, b, _ in edges if b == node]
        outs = [(b, prog) for a, b, prog in edges if a == node]
        edges = [x for x in edges if node not in (x[0], x[1])]
        edges += [(a, b, prog) for a in ins for b, prog in outs]
    keep = {n: names[n] for n in names if n not in spins}
    return keep, {n: raws[n] for n in keep}, edges, len(spins)


def _census(names: dict, raws: dict, edges: list) -> dict:
    """The programmatic edges by kernel pair, and per pair of consecutive hop (finalize) calls
    a and b, in the graph without the test's delay kernels: b's own wait returns once its
    predecessors completed; a kernel entered through a full edge started after its predecessor
    completed; one entered through a programmatic edge implies its predecessor's completion
    only if every CTA of it executes ``griddepcontrol.wait`` before it exits. So call a is
    complete when b's wait returns if some path a -> b has every programmatically entered
    kernel waiting before exit. ``waits``: the programmatically entered kernels on the path
    with the fewest of them (``path``: that path, ``P``/``F`` = entered programmatically /
    through a full edge); ``plain``: every kernel entered through a full edge from a node on
    some a -> b path (the plain launches between the two calls)."""

    def after_boundaries(names, raws, edges) -> dict:
        found = {}
        for a, b, prog in edges:
            if "UnpoolGather" in raws[a]:
                key = f"{names[b]}:{'P' if prog else 'F'}"
                found[key] = found.get(key, 0) + 1
        return found

    captured = after_boundaries(names, raws, edges)
    names, raws, edges, spins = _without_delays(names, raws, edges)
    succ, pred = {}, {}
    for a, b, prog in edges:
        succ.setdefault(a, []).append((b, prog))
        pred.setdefault(b, []).append((a, prog))
    pairs = {}
    for a, b, prog in edges:
        if prog:
            pairs[f"{names[a]} -> {names[b]}"] = pairs.get(f"{names[a]} -> {names[b]}", 0) + 1
    order, indegree = [], {n: 0 for n in names}
    for _, b, _ in edges:
        indegree[b] += 1
    ready = [n for n in names if indegree[n] == 0]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for b, _ in succ.get(n, ()):
            indegree[b] -= 1
            if indegree[b] == 0:
                ready.append(b)
    full = sum(1 for _, _, prog in edges if not prog)
    out = {
        "programmatic": pairs,
        "nodes": len(names),
        "edges": len(edges),
        "full_edges": full,
        "delay_kernels_dropped": spins,
        "after_boundary_captured": captured,
        "after_boundary": after_boundaries(names, raws, edges),
    }
    for label, pattern in (("hop", _HOP_KERNEL), ("finalize", _FINALIZE_KERNEL)):
        marks = [n for n in order if pattern.search(raws[n])]
        rows = []
        for a, b in zip(marks, marks[1:]):
            after_a = _reach(a, lambda n: [x for x, _ in succ.get(n, ())])
            if b not in after_a:
                continue
            before_b = _reach(b, lambda n: [x for x, _ in pred.get(n, ())])
            between = after_a & before_b
            plain = {}
            for x in between - {a}:
                if any(not prog and p in between for p, prog in pred.get(x, ())):
                    plain[names[x]] = plain.get(names[x], 0) + 1
            # 0-1 BFS from a inside ``between``: a programmatic edge into a kernel other
            # than b costs one wait.
            cost, back, todo = {a: 0}, {}, [a]
            while todo:
                todo.sort(key=lambda n: cost[n])
                n = todo.pop(0)
                for x, prog in succ.get(n, ()):
                    if x not in between:
                        continue
                    c = cost[n] + (1 if prog and x != b else 0)
                    if c < cost.get(x, 1 << 30):
                        cost[x], back[x] = c, (n, prog)
                        todo.append(x)
            path, n = [], b
            while n != a:
                p, prog = back[n]
                path.append(f"{names[n]}:{'P' if prog else 'F'}")
                n = p
            path.reverse()
            waits = [step.rsplit(":", 1)[0] for step in path[:-1] if step.endswith(":P")]
            # A path whose programmatically entered kernels (b aside) are all audited ones,
            # or are entered from an audited kernel that never triggers early (it waited
            # before its thread blocks exited, and they all exited before this launch).
            seen, todo, via = {a}, [a], {}
            while todo:
                n = todo.pop(0)
                for x, prog in succ.get(n, ()):
                    if x in seen or x not in between:
                        continue
                    if (
                        prog
                        and x != b
                        and not WAIT_BEFORE_EXIT.match(names[x])
                        and not (
                            n != a
                            and prog == PORT_PROGRAMMATIC
                            and NO_EARLY_TRIGGER.match(names[n])
                        )
                    ):
                        continue
                    seen.add(x)
                    via[x] = (n, prog)
                    todo.append(x)
            audited, relied_waits, relied_no_trigger = None, {}, {}
            if b in via:
                audited, n = [], b
                while n != a:
                    p, prog = via[n]
                    audited.append((n, prog))
                    if prog and n != b:
                        if WAIT_BEFORE_EXIT.match(names[n]):
                            relied_waits[raws[n]] = names[n]
                        else:
                            relied_no_trigger[raws[p]] = names[p]
                    n = p
                audited.reverse()
            rows.append(
                {
                    "pair": f"{names[a]} -> {names[b]}",
                    "waits": waits,
                    "path": path,
                    "plain": plain,
                    "all_pdl": set(plain) <= SIDE_LAUNCHES,
                    "unaudited": sorted({w for w in waits if not WAIT_BEFORE_EXIT.match(w)}),
                    "audited_path": None
                    if audited is None
                    else [f"{names[x]}:{'P' if prog else 'F'}" for x, prog in audited],
                    # Symbol -> short name of the kernels the audited path relies on: to
                    # wait before they exit, and (``NO_EARLY_TRIGGER``) never to trigger.
                    "audited_waits": relied_waits,
                    "audited_no_trigger": relied_no_trigger,
                }
            )
        out[label] = rows
    return out


def _fork_edges(names: dict, raws: dict, edges: list) -> dict:
    """Per router_fork side launch (``ROUTER_FORK_SIDE_KERNEL``, or ``POOL_DIRECT_SIDE_KERNEL``) of a captured
    graph without the test's delay kernels: its incoming and outgoing edges as ``kernel:P`` /
    ``kernel:F`` (programmatic / full), counted over the graph's side launches, and the
    side kernels seen."""
    names, raws, edges, _ = _without_delays(names, raws, edges)
    side = [n for n in names if raws[n] in (ROUTER_FORK_SIDE_KERNEL, POOL_DIRECT_SIDE_KERNEL)]
    into, out = {}, {}
    for node in side:
        ins = sorted(f"{names[a]}:{'P' if prog else 'F'}" for a, b, prog in edges if b == node)
        outs = sorted(f"{names[b]}:{'P' if prog else 'F'}" for a, b, prog in edges if a == node)
        into[" ".join(ins)] = into.get(" ".join(ins), 0) + 1
        out[" ".join(outs)] = out.get(" ".join(outs), 0) + 1
    kernels = sorted({raws[n] for n in side})
    return {"side_launches": len(side), "into": into, "out": out, "kernels": kernels}


#: Entry point -> (feature ids that run it, call sites per decode forward, widest width).
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
    # moe_prologue's kernels, on the four NVFP4-expert layers up to 8 rows (the fused pool
    # gather and ring_buddy's placement kernel then do not run there: PROLOGUE_REPLACES).
    "router_shard": ({"moe_prologue"}, 4, 8),
    # router_fork's plain launch of the router shard on its side stream, in place of router_shard.
    "router_shard_side": ({"router_fork"}, 4, 8),
    "pool_gather_route": ({"moe_prologue"}, 4, 8),
    # pool_direct: the publish twin and the direct routed kernel in place of the side launch and
    # the routed pool kernel, on the 8-row captures (_expected_counts).
    "router_shard_publish": ({"pool_direct"}, 4, 8),
    "pool_direct_route": ({"pool_direct"}, 4, 8),
    # finalize_transport's kernel takes the fused finalize of every MoE layer on 1..16-token calls.
    "finalize_transport": ({"finalize_transport"}, 6, 16),
    # swa_block takes the decode attention of the four sliding-window layers (model layers 0, 2, 3
    # and 4; layers 1 and 5 are full-attention layers).
    "swa_block_attention": ({"swa_prewait", "swa_fused_prep", "swa_fused_gate"}, 4, 8),
    # prep_append's folded prep and the built-in preprocess it skips, on the full-attention layers.
    "preattn_prep_append": ({"prep_append"}, len(FULL_LAYER_INDICES), 8),
    "preprocess_elided": ({"prep_append"}, len(FULL_LAYER_INDICES), 8),
    # direct_publish's consumer hop: at exactly DIRECT_PUBLISH_ROWS rows, in place of the hop.
    "post_attn_consumer_hop": ({"direct_publish"}, 6, 8),
    # unpool_direct: in place of the fused unpool boundary, on the 8-row captures (every count
    # here is a capture's).
    "unpool_direct_boundary": ({"unpool_direct"}, 5, 8),
    # early_ids: the expert kernel's early launches, on the four NVFP4 layers up to 8 rows.
    "early_expert_ids": ({"early_ids"}, 4, 8),
    # qkv_prefetch's prefetch of the next QKV weight: at the forward's start and after the
    # expert call of each layer whose boundary early_release converted (5), from 3 rows
    # (_expected_counts).
    "qkv_prefetch_prefetch": ({"qkv_prefetch"}, 6, 8),
    # split_kv on the full-attention layers at up to 8 rows (combine_norm_gate takes their
    # norm-gate).
    "full_attention_split_kv": ({"split_kv"}, len(FULL_LAYER_INDICES), 8),
    "combine_norm_gate": ({"split_kv"}, len(FULL_LAYER_INDICES), 8),
    # prep_fold: the prep folded into the split-KV kernel on the same steps (in place of the prep-append,
    # the elided preprocess and full_attention_split_kv).
    "full_attention_prep_fold": ({"prep_fold"}, len(FULL_LAYER_INDICES), 8),
}
#: The entries the prep-fold kernel takes the place of on the steps it serves.
PREP_FOLD_REPLACES = ("preattn_prep_append", "preprocess_elided", "full_attention_split_kv")
#: direct_publish publishes and consumes at exactly this width (``target_layers.DirectPublish.ROWS``).
DIRECT_PUBLISH_ROWS = 8
PROLOGUE_REPLACES = {"pool_gather_fp4quant": 4, "ring_buddy_assign": 4}
#: The sliding-window block ids, and the full-attention layers' call sites (prep / norm-gate stay there).
SWA_BLOCK_IDS = frozenset({"swa_prewait", "swa_fused_prep", "swa_fused_gate"})
SWA_BLOCK_FULL_SITES = 2

_COUNTS: dict = {}
_KEPT: dict = {}
_RESULTS: dict = {}

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    checkpoint = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")
    if not os.path.isfile(os.path.join(checkpoint, "model.safetensors.index.json")):
        pytest.skip(
            "needs MUSESPARK_REFERENCE_CKPT=<the Muse-Spark-Open-816B-A42B-NVFP4 checkpoint>"
        )
    drafter = os.environ.get("MUSESPARK_REFERENCE_DRAFTER", "")
    if not os.path.isfile(os.path.join(drafter, "model.safetensors")):
        pytest.skip("needs MUSESPARK_REFERENCE_DRAFTER=<the Muse-Spark-Open-DFlash2 checkpoint>")
    missing = [f"{k}={v}" for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    if missing:
        pytest.skip(f"needs the serving switches: {' '.join(missing)}")
    return comm, rank, checkpoint, drafter


class _Counted:
    """A callable that counts its calls into ``_COUNTS[name]`` and calls through."""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        _COUNTS[self.name] = _COUNTS.get(self.name, 0) + 1
        return self.fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.fn, name)


# --------------------------------------------------------------------------- weights / build
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
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        rename_open_format_weights,
    )

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles, weights = {}, {}
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
    for name, value in list(vars(pretrained).items()):
        if isinstance(value, list) and len(value) == full:
            setattr(pretrained, name, [value[i] for i in CKPT_LAYERS])
    pretrained.num_hidden_layers = len(CKPT_LAYERS)
    quant = config.quant_config
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


def _selection(value):
    return _plans.selection(value)


def _build_arm(e, selection):
    """One copy of the model, in the serving order: built, adopted under ``selection``,
    loaded, the load-time features installed (the shell's ``load_weights``), then the
    loader's post-load walk."""
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
        modeling.MuseSparkOpenTepModelingV2.adopt(model, plan)
        if model._decode.preattn_prep is not None:
            model._decode.preattn_prep = _Counted("preattn_prep", model._decode.preattn_prep)
        if model._post_attn_hop is not None:
            model._post_attn_hop.op = _Counted("post_attn_hop", model._post_attn_hop.op)
            direct = getattr(model._post_attn_hop, "direct", None)
            if direct is not None:
                direct.op = _Counted("post_attn_consumer_hop", direct.op)
        transport = getattr(model, "_finalize_transport", None)
        if transport is not None:
            # Every call the workspace routes to the transport kernel launches through _launch.
            transport._launch = _Counted("finalize_transport", transport._launch)
    owner = types.SimpleNamespace(config=config.pretrained_config)
    accounting = WeightAccounting()
    for index, layer in enumerate(model.layers):
        load_layer(
            owner, layer, CKPT_LAYERS[index], e.weights, accounting, open_format=e.open_format
        )
    with torch.no_grad():
        model.norm.weight.copy_(e.weights["model.norm.weight"][:].to(model.norm.weight.dtype))
    target_weights.install_loaded_features(model, e.mapping, plan or dispatch.NO_PATHS)
    if "direct_publish" in _selection(selection):
        # The shell's load_weights: the published o_proj must capture a validated producer.
        target_layers.verify_direct_publish(model)
    ModelLoader._walk_full_post_load(model)
    for layer in model.layers:
        layer.register_forward_hook(e.copies)
    return model


def _stale_poll_schedules(model) -> dict:
    """Which polling schedule each stale-poll site of ``model`` runs (False: the default)."""
    hop = getattr(model, "_post_attn_hop", None)
    return {
        "hop": bool(hop is not None and getattr(hop.workspace, "stale_poll", False)),
    }


def _describe(model):
    """Each layer's module classes and ring-buddy state."""
    rows = []
    for index, layer in enumerate(model.layers):
        moe = layer.feed_forward
        rows.append(
            {
                "layer": index,
                "attention": type(layer.self_attn).__name__,
                "qkv_proj": type(layer.self_attn.qkv_proj).__name__,
                "o_proj": type(layer.self_attn.o_proj).__name__,
                "pooler": type(moe.pooler).__name__,
                "moe": type(moe).__name__,
                "add_attn": type(layer.add_attn).__name__,
                "add_ffn": type(layer.add_ffn).__name__,
                "ring_buddy": hasattr(moe, "_replicated_experts"),
                "finalize": type(getattr(moe, "_finalize_ar_workspace", None)).__name__,
            }
        )
    return rows


# --------------------------------------------------------------------------- delays
def _make_delay(e):
    """A one-thread Triton kernel that spins for ``ns[site]`` nanoseconds (0: returns at once)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _clock(dummy):
        return tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
        )

    @triton.jit
    def _spin(ns_ptr, site):
        ns = tl.load(ns_ptr + site)
        start = _clock(site)
        now = start
        while now - start < ns:
            now = _clock(site)

    ns = torch.zeros(len(SITES), dtype=torch.int64, device="cuda")

    def delay(site):
        if e.delay_on:
            _spin[(1,)](ns, site, num_warps=1)

    e.delay_on = True
    for site in range(len(SITES)):
        delay(site)  # compile outside any capture
    torch.cuda.synchronize()
    return delay, ns


def _install_delays(e, model):
    """The spin kernel before each site of ``model`` (``prefetch_L2`` and the head's are elsewhere)."""
    layers = model.layers
    delay = e.delay

    def wrap_method(obj, name, site):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            delay(site)
            return original(*args, **kwargs)

        setattr(obj, name, wrapped)

    def wrap_projection(moe, pool_site, unpool_site):
        original = moe._replicated_projection

        def wrapped(module, hidden_states, workspace):
            delay(pool_site if module is moe.pooler else unpool_site)
            return original(module, hidden_states, workspace)

        moe._replicated_projection = wrapped

    wrap_method(layers[2].add_attn, "_add_after_rmsnorm_and_norm", SITES.index("hop_L2"))
    wrap_method(layers[1].feed_forward, "prepare_expert_input", SITES.index("pool_prep_L1"))
    wrap_projection(
        layers[3].feed_forward, SITES.index("pool_proj_L3"), SITES.index("unpool_proj_L3")
    )
    wrap_method(layers[4].add_ffn, "_add_after_rmsnorm_and_norm", SITES.index("f1_L4"))
    wrap_method(layers[5].add_attn, "_add_after_rmsnorm_and_norm", SITES.index("hop_L5"))
    wrap_method(layers[2].feed_forward.experts, "forward", SITES.index("experts_L2"))
    wrap_method(layers[3].self_attn, "_attn_impl", SITES.index("attn_L3"))
    # router_fork's side launch of layer 3's router shard reads layer 3's router weight.
    e.router_ptrs.add(layers[3].feed_forward.router.weight.data_ptr())
    # gemm_prefetch's o_proj -> pool prefetch of layer 2 reads the rank's slice of layer 2's pooler.
    moe = layers[2].feed_forward
    columns = moe.pooler.weight.shape[0] // e.mapping.tp_size
    e.prefetch_ptrs.add(moe.pooler.weight[e.mapping.tp_rank * columns].data_ptr())
    # qkv_prefetch's prefetch after layer 1's experts reads layer 2's QKV weight (the rank's shard).
    e.qkv_prefetch_ptrs.add(layers[2].self_attn.qkv_proj.weight.data_ptr())


# --------------------------------------------------------------------------- draft head + fc
class _Head:
    """The DFlash 2 draft head (built-in, or the sharded ``draft_head``) and the drafter ``fc``
    projection (built-in, or ``fc_shard``'s), on the drafter checkpoint's selector, fc and norm
    weights."""

    def __init__(self, e, drafter_dir):
        from safetensors import safe_open

        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
        from tensorrt_llm._torch.models.modeling_dflash import (
            DFlash2CandidateSelector,
            DFlashForCausalLM,
        )
        from tensorrt_llm._torch.modules.embedding import LMHead
        from tensorrt_llm._torch.modules.linear import TensorParallelMode
        from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
        from tensorrt_llm._torch.speculative.dflash import DFlashWorker

        draft_head = importlib.import_module(_PKG + ".kernels.draft_head.head")
        drafter_worker = importlib.import_module(_PKG + ".spec_worker")
        device = f"cuda:{torch.cuda.current_device()}"
        with safe_open(os.path.join(drafter_dir, "model.safetensors"), "pt", device) as f:
            names = list(f.keys())

            def take(key):
                (name,) = [n for n in names if n == key or n.endswith("." + key)]
                return f.get_tensor(name)

            predecessor = take("candidate_selector.predecessor_codebook")
            successor = take("candidate_selector.successor_codebook")
            projection = take("candidate_selector.hidden_projection.weight")
            fc_weight = take("fc.weight")
            norm_weight = take("hidden_norm.weight")
        self.selector = DFlash2CandidateSelector(
            predecessor, successor, projection, TOP_K, VOCAB, SELECTOR_RANK
        ).cuda()
        self.lm_head = LMHead(
            VOCAB,
            HIDDEN,
            dtype=torch.bfloat16,
            mapping=e.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        ).cuda()
        shard = VOCAB // WORLD
        generator = torch.Generator(device="cuda").manual_seed(1234 + e.rank)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                (torch.randn(shard, HIDDEN, generator=generator, device="cuda") * 0.02).bfloat16()
            )

        class _SelectorOwner:
            select_candidate_path = DFlashForCausalLM.select_candidate_path

            def __init__(self, selector):
                self.candidate_selector = selector

        self.owner = _SelectorOwner(self.selector)
        config = types.SimpleNamespace(max_draft_len=7, attention_backend="VANILLA")
        self.builtin_worker = DFlashWorker(config, e.mapping)
        self.draft_head_worker = DFlashWorker(config, e.mapping)
        self.builtin_processor = LogitsProcessor()
        llm = types.SimpleNamespace(
            spec_worker=self.draft_head_worker,
            draft_model=types.SimpleNamespace(
                candidate_selector=self.selector,
                logits_processor=LogitsProcessor(),
                lm_head=self.lm_head,
            ),
        )
        self.draft_head_installed = draft_head.install(llm)
        self.draft_head_processor = llm.draft_model.logits_processor
        self.worker_class = type(self.draft_head_worker).__name__
        self.draft_head = draft_head
        self.selector_draft_tokens = draft_head.selector_draft_tokens
        draft_head.selector_draft_tokens = _Counted(
            "draft_head_selector", self.selector_draft_tokens
        )
        self.metadata = types.SimpleNamespace(
            wants_advanced_draft_sampling=False,
            is_all_greedy_sample=True,
            draft_vocab_size=VOCAB,
            vocab_size=VOCAB,
            use_rejection_sampling=False,
        )
        # The drafter's fc + hidden_norm on a DFlashForCausalLM without its constructor:
        # the built-in project_target_hidden, and fc_shard's sharded one bound by its install.
        fc_model = DFlashForCausalLM.__new__(DFlashForCausalLM)
        torch.nn.Module.__init__(fc_model)
        fc_model.fc = torch.nn.Linear(
            fc_weight.shape[1], fc_weight.shape[0], bias=False, device="cuda", dtype=torch.bfloat16
        )
        fc_model.hidden_norm = torch.nn.RMSNorm(
            norm_weight.shape[0],
            eps=1e-5,
            device="cuda",
            elementwise_affine=True,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            fc_model.fc.weight.copy_(fc_weight)
            fc_model.hidden_norm.weight.copy_(norm_weight)
        self.fc_model = fc_model
        self.fc_shard_reason = drafter_worker.install_sharded_fc(
            fc_model, e.mapping, get_env_enable_pdl()
        )
        state = fc_model.__dict__.get("_fc_shard")
        if state is not None:
            workspace = state.gather
            gather = workspace.gather

            def delayed_gather(local):
                e.delay(SITES.index("fc_gather"))
                _COUNTS["fc_shard_gather"] = _COUNTS.get("fc_shard_gather", 0) + 1
                return gather(local)

            workspace.gather = delayed_gather
        self.builtin_project = DFlashForCausalLM.project_target_hidden

    def restore(self):
        self.draft_head.selector_draft_tokens = self.selector_draft_tokens

    def __call__(self, hidden, anchor, kind, fc_input):
        rows = hidden.shape[0]
        block_hidden = hidden.view(1, rows, HIDDEN)
        if kind in ("draft_head", "draft_head_fc_shard"):
            logits = self.draft_head_processor(hidden, self.lm_head, None, True)
            logits = logits.reshape(1, rows, logits.shape[-1])
            worker = self.draft_head_worker
        else:
            logits = self.builtin_processor(hidden, self.lm_head, None, True).reshape(
                1, rows, VOCAB
            )
            worker = self.builtin_worker
        block = worker._apply_dflash2_selector(
            self.owner, logits, block_hidden, anchor, self.metadata
        )
        tokens = worker.sample_draft_tokens(block, self.metadata, 1, num_contexts=0)
        if kind == "draft_head_fc_shard":
            projection = self.fc_model.project_target_hidden(fc_input)
        else:
            projection = self.builtin_project(self.fc_model, fc_input)
        return tokens, projection


# --------------------------------------------------------------------------- KV / metadata
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
    return metadata


def _cache_bytes(kv, metadata, length: int, prefix: str) -> dict:
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
            out[f"{prefix}/layer{layer}.{name}"] = pool_u8[
                pages, :, positions % TOKENS_PER_BLOCK, :
            ].clone()
    return out


# --------------------------------------------------------------------------- schedule
def _base_schedule(steps: int):
    """(kind, rows, input seed) per step, and per burst the accepted count of every width."""
    rng = random.Random(1235)
    weights = [1, 1, 1, 1, 1, 1, 1, 6, 1, 1]
    schedule = []
    for step in range(steps):
        rows = rng.choices(ROWS, weights=weights)[0]
        kind = "eager" if step % 7 == 3 else "replay"
        schedule.append((kind, rows, 1000 + step))
    bursts = -(-steps // BURST)
    accept = [{m: rng.randint(1, m) for m in ROWS} for _ in range(bursts)]
    return schedule, accept


def _delay_schedule(steps: int, seed: int):
    """Per step: [(rank, site, ns)] spins, and the rank that sleeps on the host (or None)."""
    if seed == 0:
        return [([], None) for _ in range(steps)]
    if seed < 0:
        rng = random.Random(7000 - seed)
        return [([], rng.randrange(WORLD) if step % 2 else None) for step in range(steps)]
    rng = random.Random(7000 + seed)
    out = []
    for step in range(steps):
        delays = []
        if step % 5 != 0:
            for _ in range(rng.choice([1, 1, 2, 3])):
                delays.append(
                    (
                        rng.randrange(WORLD),
                        rng.randrange(len(SITES)),
                        rng.choice([30_000, 300_000, 2_000_000]),
                    )
                )
        host = rng.randrange(WORLD) if step % 4 == 1 else None
        out.append((delays, host))
    return out


def _short_prefill_rows(rows: int) -> int:
    return (3, 8, 13)[rows % 3]


class _State:
    """One width: the short prefill, the decode request's KV cache, metadata, inputs and graph."""

    def __init__(self, e, model, rows, bursts, out):
        self.rows = rows
        self.prompt_len = PROMPT_BASE + 3 * rows
        short = _short_prefill_rows(rows)
        self.capacity = self.prompt_len + (bursts + 2) * rows + TOKENS_PER_BLOCK
        generator = torch.Generator().manual_seed(rows)
        prompt = torch.randint(0, TOKEN_POOL, (self.prompt_len,), generator=generator)
        kv_short = _kv_cache(e, short)
        try:
            kv_short.add_dummy_requests([0], [short])
            metadata = _metadata(e, kv_short, short, 0, short, 1)
            e.sink.clear()
            _COUNTS.clear()
            out[f"short_prefill@{rows}"] = model(
                attn_metadata=metadata,
                position_ids=torch.arange(short, dtype=torch.int, device="cuda")[None],
                inputs_embeds=e.table[prompt[:short].cuda()],
            ).clone()
            e.copies.join()
            self.short_counts = dict(_COUNTS)
            for index, layer_out in enumerate(e.sink):
                for part, tensor in enumerate(layer_out):
                    out[f"short_prefill@{rows}/layer{index}.{part}"] = tensor.clone()
            e.sink.clear()
            out.update(_cache_bytes(kv_short, metadata, short, f"short_cache@{rows}"))
        finally:
            kv_short.shutdown()
        self.kv = _kv_cache(e, self.capacity)
        self.kv.add_dummy_requests([0], [self.capacity])
        metadata = _metadata(e, self.kv, self.prompt_len, 0, self.prompt_len, 1)
        e.sink.clear()
        out[f"prefill@{rows}"] = model(
            attn_metadata=metadata,
            position_ids=torch.arange(self.prompt_len, dtype=torch.int, device="cuda")[None],
            inputs_embeds=e.table[prompt.cuda()],
        ).clone()
        e.copies.join()
        e.sink.clear()
        metadata = _metadata(e, self.kv, rows, self.prompt_len, self.prompt_len, 0)
        metadata = metadata.create_cuda_graph_metadata(1, max_draft_tokens=rows - 1)
        metadata.prepare()
        self.metadata = metadata
        self.cached = self.prompt_len
        self.position_ids = torch.arange(
            self.prompt_len, self.prompt_len + rows, dtype=torch.int
        ).cuda()[None]
        self.embeds = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.anchor = torch.zeros(1, dtype=torch.long, device="cuda")
        self.graph = None
        self.static = None

    def advance(self, accepted: int) -> None:
        """Host side, after a sync: the next decode step starts ``accepted`` tokens later."""
        from tensorrt_llm._torch.metadata import KVCacheParams

        self.cached += accepted
        assert self.cached + self.rows <= self.capacity
        self.metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[self.cached]
        )
        self.metadata.prepare()
        self.position_ids.copy_(
            torch.arange(self.cached, self.cached + self.rows, dtype=torch.int)[None]
        )

    def forward(self, e, model, head_kind):
        e.sink.clear()
        out = model(
            attn_metadata=self.metadata, position_ids=self.position_ids, inputs_embeds=self.embeds
        )
        e.copies.join()
        layers = [list(x) for x in e.sink]
        tokens = projection = None
        if self.rows <= HEAD_MAX_ROWS:
            e.delay(SITES.index("head"))
            fc_input = torch.cat([layers[i][1] for i in range(FC_LAYERS)], dim=-1)
            tokens, projection = e.head(out, self.anchor, head_kind, fc_input)
        e.sink.clear()
        return out, tokens, projection, layers

    def capture(self, e, model, head_kind) -> dict:
        self.forward(e, model, head_kind)  # warm-up: compiles and allocates outside the capture
        torch.cuda.synchronize()
        e.comm.Barrier()
        _COUNTS.clear()
        # Kept, so its edges can be read before the first replay instantiates it.
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(self.graph, stream=_capture_stream()):
            aliased = _prepare_stream_aliased(model)
            self.static = self.forward(e, model, head_kind)
        counts = dict(_COUNTS)
        if _CURRENT_ARM[0] is not None:
            _PREPARE_ALIASED[(_CURRENT_ARM[0], self.rows)] = aliased
            _PDL_EDGES[(_CURRENT_ARM[0], self.rows)] = programmatic_edges(self.graph)
            nodes_and_edges = _graph_nodes_and_edges(self.graph)
            _GRAPH_CENSUS[(_CURRENT_ARM[0], self.rows)] = _census(*nodes_and_edges)
            _FORK_EDGES[(_CURRENT_ARM[0], self.rows)] = _fork_edges(*nodes_and_edges)
            for raw, threads in _block_threads(self.graph).items():
                _BLOCK_THREADS[raw] = max(_BLOCK_THREADS.get(raw, 0), threads)
        plan = getattr(model, "_dense_gemm_plan", None)
        if plan is not None:
            counts.update(_fork_state(plan))
        torch.cuda.synchronize()
        e.comm.Barrier()
        return counts

    def close(self) -> None:
        self.graph = None
        self.static = None
        self.kv.shutdown()


def _fork_state(plan) -> dict:
    """gemm_prefetch's prefetch fork and the named forks (qkv_prefetch's): outstanding, and dropped or skipped
    (a fork inside a capture before its stream exists) so far."""
    forks = list(getattr(plan, "named_forks", {}).values())
    return {
        "prefetch_pending": int(plan.pending) + sum(int(f.pending) for f in forks),
        "prefetch_dropped": int(plan.dropped) + sum(f.dropped + f.skipped for f in forks),
    }


def _record(out, prefix, final, tokens, projection, layers):
    out[prefix] = final.clone()
    if tokens is not None:
        out[prefix + "/tokens"] = tokens.clone()
    if projection is not None:
        out[prefix + "/proj"] = projection.clone()
    for index, layer_out in enumerate(layers):
        for part, tensor in enumerate(layer_out):
            out[f"{prefix}/layer{index}.{part}"] = tensor.clone()


@torch.inference_mode()
def _run_arm(e, model, head_kind, schedule, accept, delays):
    """Every width's state, its capture counts, then the schedule; all outputs and the cache."""
    out, states, counts = {}, {}, {}
    try:
        for rows in ROWS:
            states[rows] = _State(e, model, rows, len(accept), out)
        torch.cuda.synchronize()
        e.comm.Barrier()
        counts["short"] = {rows: state.short_counts for rows, state in states.items()}
        for rows in ROWS:
            counts[rows] = states[rows].capture(e, model, head_kind)
        inputs, anchors = {}, {}
        for step, (_kind, rows, seed) in enumerate(schedule):
            generator = torch.Generator().manual_seed(seed)
            inputs[step] = torch.randint(0, TOKEN_POOL, (rows,), generator=generator).cuda()
            anchors[step] = torch.randint(0, VOCAB, (1,), generator=generator).cuda()
        table = torch.zeros(len(schedule), len(SITES), dtype=torch.int64)
        for step, (spins, _host) in enumerate(delays):
            for rank, site, ns in spins:
                if rank == e.rank:
                    table[step, site] += ns
        table = table.cuda()
        torch.cuda.synchronize()
        e.comm.Barrier()
        ran = set()
        for step, (kind, rows, _seed) in enumerate(schedule):
            if step % BURST == 0:
                torch.cuda.synchronize()
                if step:
                    # A state that ran in the burst wrote [cached, cached + M); it moves on
                    # by the burst's accepted count, so its rejected rows get rewritten.
                    for m in sorted(ran):
                        states[m].advance(accept[step // BURST - 1][m])
                    ran = set()
            ran.add(rows)
            state = states[rows]
            if delays[step][1] == e.rank:
                time.sleep(0.004)
            e.delay_ns.copy_(table[step], non_blocking=True)
            torch.index_select(e.table, 0, inputs[step], out=state.embeds)
            state.anchor.copy_(anchors[step], non_blocking=True)
            if kind == "replay":
                state.graph.replay()
                final, tokens, projection, layers = state.static
            else:
                final, tokens, projection, layers = state.forward(e, model, head_kind)
            _record(out, f"s{step}:{kind}@{rows}", final, tokens, projection, layers)
        torch.cuda.synchronize()
        e.delay_ns.zero_()
        for m in sorted(ran):
            states[m].advance(accept[(len(schedule) - 1) // BURST][m])
        torch.cuda.synchronize()
        for m in ROWS:
            state = states[m]
            out.update(_cache_bytes(state.kv, state.metadata, state.cached, f"cache@{m}"))
        plan = getattr(model, "_dense_gemm_plan", None)
        if plan is not None:
            counts["after"] = dict(
                _fork_state(plan),
                side_stream_is_capture_stream=plan.side_stream_is_capture_stream,
            )
    finally:
        for state in states.values():
            state.close()
        torch.cuda.synchronize()
        e.comm.Barrier()
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
                found.append(f"{name}: {differ}/{tensor.numel()} differ")
    extra = sorted(set(got) - set(want))
    if extra:
        found.append(f"unexpected outputs {extra[:4]}")
    return found


def _short_prefill_problems(selection, short_counts: dict) -> list:
    """Where ``dense_gemms`` / ``gemm_prefetch`` (at most 8 rows) and ``ring_buddy`` (at most 16) did not act on a
    short prefill they take, or acted on one they do not."""
    selected = _selection(selection)
    problems = []
    for rows, counts in short_counts.items():
        short = _short_prefill_rows(rows)
        acting = {
            "dense_gemm": bool(selected & {"dense_gemms", "gemm_prefetch"}) and short <= 8,
            "l2_prefetch": "gemm_prefetch" in selected and short <= 8,
            "ring_buddy_assign": bool(selected & _plans.LOADED) and short <= 16,
            # After each converted layer's experts (no forward-start fork in a prefill).
            "qkv_prefetch_prefetch": "qkv_prefetch" in selected and 3 <= short <= 8,
        }
        for entry, active in acting.items():
            if bool(counts.get(entry, 0)) != active:
                problems.append(
                    f"short prefill of {short} rows: {entry} ran {counts.get(entry, 0)} times"
                )
    return problems


def _expected_counts(selection, rows: int, head_kind: str) -> dict:
    selected = _selection(selection)
    expected = {}
    for entry, (ids, sites, widest) in ENTRIES.items():
        active = bool(selected & ids) and (widest is None or rows <= widest)
        expected[entry] = sites if active else 0
    if expected.get("unpool_direct_boundary"):
        # unpool_direct arms the direct transport for a forward of 8 rows inside a capture only.
        if rows == 8:
            expected["unpool_gather_boundary"] = 0
        else:
            expected["unpool_direct_boundary"] = 0
    if expected.get("l2_prefetch") and "early_release" in selected and "gathers" in selected:
        # early_release drops gemm_prefetch's prefetch of the next QKV weight behind every fused boundary, at
        # the widths whose QKV kernel loads its weight before its wait (split-K). No layer
        # here is a DFlash capture layer: the head reads the layer outputs through this
        # test's hooks.
        dense_gemms = importlib.import_module(_PKG + ".kernels.dense_gemms.dispatcher")
        pdl_triggers = importlib.import_module(_PKG + ".kernels.boundary_early_release.triggers")
        qkv = dense_gemms.default_kernels()["qkv"]
        if rows in pdl_triggers.weights_before_wait_rows(qkv):
            expected["l2_prefetch"] -= ENTRIES["unpool_gather_boundary"][1]
    if "moe_prologue" in selected and rows <= 8:
        for entry, sites in PROLOGUE_REPLACES.items():
            if expected.get(entry):
                expected[entry] -= sites
    if expected.get("qkv_prefetch_prefetch"):
        # qkv_prefetch prefetches for a QKV kernel that loads weight k-tiles after its wait (split-K).
        dense_gemms = importlib.import_module(_PKG + ".kernels.dense_gemms.dispatcher")
        pdl_triggers = importlib.import_module(_PKG + ".kernels.boundary_early_release.triggers")
        if rows not in pdl_triggers.weights_before_wait_rows(dense_gemms.default_kernels()["qkv"]):
            expected["qkv_prefetch_prefetch"] = 0
    if "router_fork" in selected and rows <= 8:
        # router_fork launches the router shard on its side stream instead.
        expected["router_shard"] = 0
    if expected.get("pool_direct_route"):
        # pool_direct arms the direct pool transport for a forward of 8 rows inside a capture only.
        if rows == 8:
            expected["router_shard_side"] = 0
            expected["pool_gather_route"] = 0
        else:
            expected["router_shard_publish"] = 0
            expected["pool_direct_route"] = 0
    if selected & SWA_BLOCK_IDS:
        # swa_block takes the sliding-window layers' attention; swa_fused_prep also their prep,
        # swa_fused_gate their output gate (early_release's norm-gate then runs only at the two
        # full layers).
        fits = rows <= ENTRIES["swa_block_attention"][2]
        expected["paged_decode_attention"] = 0
        if "swa_fused_prep" in selected and fits:
            expected["preattn_prep"] = SWA_BLOCK_FULL_SITES
        if "swa_fused_gate" in selected and fits and expected.get("attention_norm_gate"):
            expected["attention_norm_gate"] = SWA_BLOCK_FULL_SITES
    # prep_append: the full-attention layers run the folded prep instead of preattn_prep's;
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
    # direct_publish at exactly its width: the consumer hop in place of the hop, o_proj's GEMM
    # through dense_gemm with the slot as its output (the producer policy), and no o_proj -> pool
    # prefetch.
    direct = "direct_publish" in selected and rows == DIRECT_PUBLISH_ROWS
    expected["post_attn_consumer_hop"] = ENTRIES["post_attn_consumer_hop"][1] if direct else 0
    if direct:
        expected["post_attn_hop"] = 0
        if expected.get("l2_prefetch"):
            expected["l2_prefetch"] -= len(CKPT_LAYERS)
    if expected.get("combine_norm_gate") and expected.get("attention_norm_gate"):
        expected["attention_norm_gate"] -= expected["combine_norm_gate"]
    if expected.get("full_attention_prep_fold"):
        for entry in PREP_FOLD_REPLACES:
            expected[entry] = 0
    head = rows <= HEAD_MAX_ROWS
    expected["draft_head_selector"] = (
        1 if head and head_kind in ("draft_head", "draft_head_fc_shard") else 0
    )
    expected["fc_shard_gather"] = 1 if head and head_kind == "draft_head_fc_shard" else 0
    return expected


def _assert_on_every_rank(e, found: list, what: str) -> None:
    everyone = e.comm.allgather(found[:6])
    failed = {f"rank {rank}": bad for rank, bad in enumerate(everyone) if bad}
    assert not failed, f"{what}: {failed}"


@pytest.fixture(scope="module")
def env():
    """Every arm's model copy, the draft head, the delay kernel and the embedding rows."""
    comm, rank, checkpoint, drafter_dir = _world()
    from tensorrt_llm.mapping import Mapping

    dense_gemms = importlib.import_module(_PKG + ".kernels.dense_gemms.dispatcher")
    projection_gathers = importlib.import_module(_PKG + ".kernels.projection_gathers.gathers")
    ring_buddy = importlib.import_module(_PKG + ".kernels.replicated_experts.ring_buddy")
    pdl_triggers = importlib.import_module(_PKG + ".kernels.boundary_early_release.triggers")
    persistent_experts = importlib.import_module(_PKG + ".ops.moe.persistent_experts")
    moe_prologue = importlib.import_module(_PKG + ".kernels.moe_prologue.prologue")
    swa_attention_block = importlib.import_module(_PKG + ".kernels.swa_block.attention")
    full_attention = importlib.import_module(_PKG + ".kernels.full_attention_prep_append.attention")
    direct_mod = importlib.import_module(_PKG + ".ops.comm.unpool_direct_boundary")
    pool_direct_mod = importlib.import_module(_PKG + ".ops.comm.pool_direct_route")
    qkv_prefetch = importlib.import_module(_PKG + ".kernels.next_qkv_prefetch.prefetch")
    e = types.SimpleNamespace(
        comm=comm,
        rank=rank,
        checkpoint=checkpoint,
        sink=[],
        prefetch_ptrs=set(),
        qkv_prefetch_ptrs=set(),
        router_ptrs=set(),
    )
    e.copies = LayerOutputCopies(e.sink, HOOK_SIDE)
    e.mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
    )
    e.weights, e.open_format = _checkpoint_weights(checkpoint)
    e.pretrained = _model_config(checkpoint, e.mapping).pretrained_config
    embed = e.weights["model.embed_tokens.weight"]
    generator = torch.Generator().manual_seed(int(os.environ.get("MODELING_V2_INPUT_SEED", "2026")))
    tokens = torch.randint(0, embed.get_shape()[0], (TOKEN_POOL,), generator=generator).tolist()
    e.table = torch.cat([embed[t : t + 1] for t in tokens]).to("cuda", torch.bfloat16)
    patched = [
        (modeling, "paged_decode_attention", "paged_decode_attention"),
        (projection_gathers._unpool, "unpool_gather_boundary", "unpool_gather_boundary"),
        (projection_gathers._pool, "pool_gather_rmsnorm_fp4quant", "pool_gather_fp4quant"),
        (dense_gemms, "dense_gemm", "dense_gemm"),
        (dense_gemms, "l2_prefetch", "l2_prefetch"),
        (ring_buddy, "ring_buddy_assign", "ring_buddy_assign"),
        (pdl_triggers._norm_gate, "attention_norm_gate", "attention_norm_gate"),
        (persistent_experts, "persistent_experts", "persistent_experts"),
        (moe_prologue._shard, "router_shard", "router_shard"),
        (moe_prologue._shard, "router_shard_side", "router_shard_side"),
        (moe_prologue._route, "pool_gather_route", "pool_gather_route"),
        (moe_prologue._shard, "router_shard_publish", "router_shard_publish"),
        (pool_direct_mod, "pool_direct_route", "pool_direct_route"),
        (swa_attention_block, "swa_block_attention", "swa_block_attention"),
        (full_attention, "preattn_prep_append", "preattn_prep_append"),
        (direct_mod, "unpool_direct_boundary", "unpool_direct_boundary"),
        (qkv_prefetch, "l2_prefetch", "qkv_prefetch_prefetch"),
    ]
    if any("split_kv" in (a[2] or "") for a in ARMS):
        fa4_mod = importlib.import_module(_PKG + ".ops.attention.full_attention_split_kv")
        cng_mod = importlib.import_module(_PKG + ".ops.attention.combine_norm_gate")
        patched += [
            (fa4_mod, "full_attention_split_kv", "full_attention_split_kv"),
            (cng_mod, "combine_norm_gate", "combine_norm_gate"),
        ]
    if any("prep_fold" in (a[2] or "") for a in ARMS):
        fold_mod = importlib.import_module(_PKG + ".ops.attention.full_attention_prep_fold")
        patched.append((fold_mod, "full_attention_prep_fold", "full_attention_prep_fold"))
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    folded_outputs = full_attention.FoldedPreprocess.outputs

    def counted_outputs(self, args):
        # prep_append skips the built-in preprocess of a full-attention layer through this.
        _COUNTS["preprocess_elided"] = _COUNTS.get("preprocess_elided", 0) + 1
        return folded_outputs(self, args)

    e.head = None
    # early_ids: count only the expert launches that run early.
    ei_mod = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
        "muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.early_expert_ids"
    )
    launches_early = ei_mod.launches_early

    def counted_launches_early(early, topk_ids):
        ran = launches_early(early, topk_ids)
        if ran:
            _COUNTS["early_expert_ids"] = _COUNTS.get("early_expert_ids", 0) + 1
        return ran

    originals.append((ei_mod, "launches_early", launches_early))
    try:
        full_attention.FoldedPreprocess.outputs = counted_outputs
        ei_mod.launches_early = counted_launches_early
        for module, attr, name in patched:
            setattr(module, attr, _Counted(name, getattr(module, attr)))
        e.delay, e.delay_ns = _make_delay(e)
        counted_prefetch = dense_gemms.l2_prefetch

        def delayed_prefetch(weight, sink):
            if weight.data_ptr() in e.prefetch_ptrs:
                e.delay(SITES.index("prefetch_L2"))
            return counted_prefetch(weight, sink)

        dense_gemms.l2_prefetch = delayed_prefetch
        if "qkv_prefetch_L2" in SITES:
            counted_qkv = qkv_prefetch.l2_prefetch

            def delayed_qkv(counted):
                def launch(weight, *args):
                    if weight.data_ptr() in e.qkv_prefetch_ptrs:
                        e.delay(SITES.index("qkv_prefetch_L2"))
                    return counted(weight, *args)

                return launch

            qkv_prefetch.l2_prefetch = delayed_qkv(counted_qkv)
        if "router_side_L3" in SITES:

            def delayed_router(counted):
                def launch(hidden_states, weight, *args):
                    # On router_fork's side stream (the caller's current stream), in front of
                    # the router (the side launch, or pool_direct's publish twin).
                    if weight.data_ptr() in e.router_ptrs:
                        e.delay(SITES.index("router_side_L3"))
                    return counted(hidden_states, weight, *args)

                return launch

            moe_prologue._shard.router_shard_side = delayed_router(
                moe_prologue._shard.router_shard_side
            )
            moe_prologue._shard.router_shard_publish = delayed_router(
                moe_prologue._shard.router_shard_publish
            )
        e.models, e.layout, e.stale_poll_on = {}, {}, {}
        for _arm, key, selection, _head, _reference, _seed in ARMS:
            if key in e.models:
                continue
            e.models[key] = _build_arm(e, selection)
            _keep_finalize_cubins(f"build-{key}")
            e.layout[key] = _describe(e.models[key])
            e.stale_poll_on[key] = _stale_poll_schedules(e.models[key])
            _install_delays(e, e.models[key])
            comm.Barrier()
        e.free_gib = torch.cuda.mem_get_info()[0] / 2**30
        e.head = _Head(e, drafter_dir)
        e.schedule, e.accept = _base_schedule(STEPS)
        torch.cuda.synchronize()
        comm.Barrier()
        yield e
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        if e.head is not None:
            e.head.restore()
        for module, attr, original in originals:
            setattr(module, attr, original)
        full_attention.FoldedPreprocess.outputs = folded_outputs


def test_the_served_sets_have_the_expected_layout(env):
    """The full target set re-classes every layer as its features ask, ring_buddy is on
    every MoE layer, and the head runs the sharded draft head's worker with the sharded fc."""
    e = env
    wrong = []
    full, bitid = LAYOUT_ARMS
    early_release_on = (
        "NormGate"
        if _SET
        in (
            "tier2",
            "tier2_reseeded",
            "tier3",
            "swa_block",
            "tier4",
            "unpool_direct",
            "qkv_prefetch",
            "unpool_direct_tier3",
            "unpool_direct_tier4",
            "qkv_prefetch_tier4",
            "early_ids",
            "three_windows",
            "early_ids_three_windows",
            "tier5",
            "split_kv_tier4",
            "router_fork",
            "router_fork_tier5",
            "prep_fold",
            "pool_direct",
            "tier7",
        )
        else ""
    )
    swa_block_on = (
        "SwaBlock"
        if _SET
        in (
            "swa_block",
            "tier4",
            "unpool_direct_tier4",
            "qkv_prefetch_tier4",
            "tier5",
            "split_kv_tier4",
            "router_fork",
            "router_fork_tier5",
            "prep_fold",
            "pool_direct",
            "tier7",
        )
        else ""
    )
    # The tier-4 target set has prep_append: FullAppendAttention in front on the full-attention layers.
    prep_append_on = (
        "FullAppend"
        if _SET
        in (
            "tier4",
            "unpool_direct_tier4",
            "qkv_prefetch_tier4",
            "tier5",
            "split_kv_tier4",
            "router_fork",
            "router_fork_tier5",
            "prep_fold",
            "pool_direct",
            "tier7",
        )
        else ""
    )
    # split_kv puts FullAttnSplitKvAttention in front of prep_append's class there.
    full_selection = _selection(next(a[2] for a in ARMS if a[1] == full))
    v4 = "V4" if "split_kv" in full_selection else ""
    # prep_fold puts FullAttnSplitKvFoldAttention in front of split_kv's class there.
    v4 += "Fold" if "prep_fold" in full_selection else ""
    for row in e.layout[full]:
        tail = prep_append_on + v4 if row["layer"] in FULL_LAYER_INDICES else ""
        expected = {
            "attention": f"MuseSparkOpenTepPrepDecode{early_release_on}Attention{swa_block_on}{tail}",
            "qkv_proj": "MuseSparkOpenTepDenseLinear",
            "o_proj": "MuseSparkOpenTepDenseLinear",
            "pooler": "MuseSparkOpenTepDenseLinear",
            "ring_buddy": True,
        }
        for name, value in expected.items():
            if row[name] != value:
                wrong.append(f"{full} layer {row['layer']} {name}: {row[name]}")
    for row in e.layout[bitid]:
        if row["qkv_proj"] != "Linear" or row["ring_buddy"]:
            wrong.append(f"{bitid} layer {row['layer']}: {row}")
    if _SET in (
        "unpool_direct",
        "unpool_direct_tier3",
        "unpool_direct_tier4",
        "qkv_prefetch_tier4",
        "tier5",
        "router_fork_tier5",
        "prep_fold",
        "pool_direct",
        "tier7",
    ):
        # unpool_direct re-classes every fused post-FFN join; the last layer keeps the built-in one.
        joins = [row["add_ffn"] for row in e.layout[full]]
        want = ["MuseSparkOpenTepDirectGatedResidual"] * (len(joins) - 1)
        if joins[:-1] != want or joins[-1] == want[0]:
            wrong.append(f"{full} post-FFN joins: {joins}")
    for _arm, key, selection, _head, _reference, _seed in ARMS:
        selected = _selection(selection)
        stale_poll_on = {
            "hop": "stale_poll" in selected and bool(selected & {"hop", "hop_early_trigger"})
        }
        if e.stale_poll_on[key] != stale_poll_on:
            wrong.append(
                f"{key} polling schedules {e.stale_poll_on[key]}, expected {stale_poll_on}"
            )
        finalize = (
            "FinalizeTransportWorkspace"
            if "finalize_transport" in selected
            else "FinalizeIpcWorkspace"
        )
        prologue = "moe_prologue" in selected
        for row in e.layout[key]:
            if row["finalize"] != finalize:
                wrong.append(f"{key} layer {row['layer']} finalize workspace {row['finalize']}")
            if prologue != (row["moe"] == "MuseSparkOpenTepPrologueMoE"):
                wrong.append(f"{key} layer {row['layer']} MoE class {row['moe']}")
    if not e.head.draft_head_installed or e.head.worker_class != "ShardedDraftHeadDFlashWorker":
        wrong.append(
            f"draft head: installed={e.head.draft_head_installed} worker={e.head.worker_class}"
        )
    if e.head.fc_shard_reason is not None:
        wrong.append(f"fc_shard not installed: {e.head.fc_shard_reason}")
    _assert_on_every_rank(e, wrong, "layout")


@pytest.mark.parametrize("arm", [arm[0] for arm in ARMS])
def test_every_arm_equals_its_reference_under_the_schedule(env, arm):
    """Every output of the schedule (and of the short prefills) equals the reference arm's
    bit for bit, whatever the delays; every capture counts each entry point exactly."""
    e = env
    _, key, selection, head_kind, reference, seed = next(a for a in ARMS if a[0] == arm)
    if reference is not None and reference not in _KEPT:
        pytest.fail(f"the reference arm {reference} did not run")
    e.delay_on = seed > 0
    try:
        _CURRENT_ARM[0] = arm
        out, counts = _run_arm(
            e,
            e.models[key],
            head_kind,
            e.schedule,
            e.accept,
            _delay_schedule(len(e.schedule), seed),
        )
    finally:
        e.delay_on = True
        _CURRENT_ARM[0] = None
        _keep_finalize_cubins(arm)
    found = []
    if reference is None:
        _KEPT[arm] = out
        found += [
            f"{name}: not finite"
            for name, tensor in out.items()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all()
        ]
    else:
        found += _differences(out, _KEPT[reference])
    if arm in PARTIAL:
        other, changed = PARTIAL[arm]
        if other not in _KEPT:
            found.append(f"{other} did not run before {arm}")
        else:
            keep = {k: v for k, v in _KEPT[other].items() if not changed.match(k)}
            found += _differences({k: v for k, v in out.items() if not changed.match(k)}, keep)
    for rows in ROWS:
        expected = _expected_counts(selection, rows, head_kind)
        have = {k: counts[rows].get(k, 0) for k in expected}
        if have != expected:
            found.append(f"capture counts at {rows} rows: {have}, expected {expected}")
        if counts[rows].get("prefetch_pending") or counts[rows].get("prefetch_dropped"):
            found.append(f"gemm_prefetch forks left at {rows} rows: {counts[rows]}")
    found += _short_prefill_problems(selection, counts["short"])
    after = counts.get("after")
    if after and (after["prefetch_pending"] or after["prefetch_dropped"]):
        found.append(f"gemm_prefetch forks left after the schedule: {after}")
    _RESULTS[arm] = {
        "selection": selection,
        "head": head_kind,
        "reference": reference,
        "delay_seed": seed,
        "outputs": len(out),
        "problems": found[:8],
        "counts_at_8": counts[8],
        "counts_at_16": counts[16],
        "short_prefill_counts": counts["short"],
        # gemm_prefetch's prefetch stream vs the capturing stream (recorded: an alias only serializes).
        "gemm_prefetch_side_stream_is_capture_stream": (counts.get("after") or {}).get(
            "side_stream_is_capture_stream"
        ),
    }
    _assert_on_every_rank(e, found, f"{arm} vs {reference}")


def test_a_delayed_rank_holds_every_rank_at_every_site_it_passes(env):
    """Rank 3 spins 2 ms at one site at a time inside an M = 8 graph: every rank's replay
    takes 2 ms longer where the arm passes the site (the collective or join behind it is
    a real graph edge), and not where it does not; the bits never change."""
    e = env
    g = torch.Generator().manual_seed(4242)
    inputs = torch.randint(0, TOKEN_POOL, (8,), generator=g).cuda()
    anchor = torch.randint(0, VOCAB, (1,), generator=g).cuda()
    cases = [("none", None)] + [(name, index) for index, name in enumerate(SITES)]
    found, table = [], {}
    with torch.inference_mode():
        for arm in PROBE_ARMS:
            _, key, selection, head_kind, _reference, _seed = next(a for a in ARMS if a[0] == arm)
            state = _State(e, e.models[key], 8, 1, {})
            try:
                state.capture(e, e.models[key], head_kind)
                torch.index_select(e.table, 0, inputs, out=state.embeds)
                state.anchor.copy_(anchor)
                medians, base, same = {}, None, True
                for name, site in cases:
                    e.delay_ns.zero_()
                    if site is not None and e.rank == PROBE_RANK:
                        e.delay_ns[site] = PROBE_NS
                    times = []
                    for _ in range(PROBE_REPLAYS):
                        torch.cuda.synchronize()
                        e.comm.Barrier()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        state.graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                        final, tokens, projection, layers = state.static
                        snap = [final.clone(), tokens.clone(), projection.clone()]
                        snap += [t.clone() for x in layers for t in x]
                        if base is None:
                            base = snap
                        else:
                            same &= all(torch.equal(_bits(x), _bits(y)) for x, y in zip(snap, base))
                    times.sort()
                    medians[name] = times[len(times) // 2]
                e.delay_ns.zero_()
            finally:
                state.close()
            table[arm] = medians
            if not same:
                found.append(f"{arm}: a delayed replay changed the bits")
            selected = _selection(selection)
            for name, _site in cases[1:]:
                passes = {
                    # direct_publish (this probe's 8 rows are its width) drops o_proj's pool prefetch.
                    "prefetch_L2": "gemm_prefetch" in selected and "direct_publish" not in selected,
                    "qkv_prefetch_L2": "qkv_prefetch" in selected,
                    "fc_gather": head_kind == "draft_head_fc_shard",
                    # moe_prologue's one-chain prep (8 rows here) does not call prepare_expert_input.
                    "pool_prep_L1": "moe_prologue" not in selected,
                    # Only router_fork launches layer 3's router shard on a side stream.
                    "router_side_L3": "router_fork" in selected,
                }.get(name, True)
                added = medians[name] - medians["none"]
                if passes and added < 0.75 * PROBE_NS / 1e6:
                    found.append(f"{arm} {name}: +{added:.3f} ms, the delay did not hold this rank")
                if not passes and added > 0.25 * PROBE_NS / 1e6:
                    found.append(f"{arm} {name}: +{added:.3f} ms at a site the graph never passes")
    _RESULTS["probe"] = table
    _assert_on_every_rank(e, found, "delay probe")


def test_every_fused_boundary_is_the_programmatic_predecessor_of_the_next_qkv(env):
    """Every captured M <= 8 graph of an arm with the fused unpool boundary has a
    programmatic edge out of each of its boundaries (the next layer's QKV GEMM is its
    PDL dependent, which ``early_release``'s entry trigger launches early); no other graph has one.
    With ``MODELING_V2_HOOK_STREAM=main`` (the negative control) no graph has one at all."""
    e = env
    found, table = [], {}
    for (arm, rows), edges in sorted(_PDL_EDGES.items()):
        have = boundary_edges(edges)
        table[f"{arm}@{rows}"] = {
            "boundary_out": have,
            "total": sum(edges.values()),
            "edges": edge_table(edges),
        }
        selection = next(a[2] for a in ARMS if a[0] == arm)
        unpool = "gathers" in _selection(selection)
        sites, widest = ENTRIES["unpool_gather_boundary"][1:]
        want = sites if HOOK_SIDE and unpool and rows <= widest else 0
        if have != want:
            found.append(
                f"{arm}@{rows}: {have} programmatic edges out of the boundary, want {want}"
            )
    widths = {}
    for arm, rows in _PDL_EDGES:
        widths.setdefault(arm, set()).add(rows)
    if not widths:
        found.append("no captured graph was read")
    found += [
        f"{arm}: graphs read at {sorted(have)}" for arm, have in widths.items() if have != set(ROWS)
    ]
    _RESULTS["pdl_edges"] = table
    _RESULTS["hook_stream"] = "side" if HOOK_SIDE else "main"
    _assert_on_every_rank(e, found, "boundary -> QKV edges")


def test_each_qkv_prefetch_prefetch_is_a_programmatic_predecessor_of_a_qkv(env):
    """``MODELING_V2_COMPOSE_TIER=qkv_prefetch``, ``qkv_prefetch_tier4`` or ``tier7``: each
    captured graph of an arm with ``qkv_prefetch`` has, at the widths it prefetches at, one more
    programmatic edge out of an L2 prefetch kernel per prefetch (into its QKV GEMM) than the
    graph of the same set without ``qkv_prefetch``, and the same number at the other widths. The
    tier-7 table carries the set without ``qkv_prefetch`` for ``TIER7_QKV_PREFETCH_ARMS`` only;
    its other ``qkv_prefetch`` arms have no such set and are not compared."""
    if _SET not in ("qkv_prefetch", "qkv_prefetch_tier4", "tier7"):
        pytest.skip("runs with MODELING_V2_COMPOSE_TIER=qkv_prefetch, qkv_prefetch_tier4 or tier7")
    e = env
    found, table = [], {}

    def prefetch_edges(edges):
        return sum(count for (source, _, _), count in edges.items() if "l2_prefetch" in source)

    by_selection = {a[2]: a[0] for a in ARMS}
    for (arm, rows), edges in sorted(_PDL_EDGES.items()):
        selection = next(a[2] for a in ARMS if a[0] == arm)
        if "qkv_prefetch" not in _selection(selection):
            continue
        if _SET == "tier7" and selection.replace(",qkv_prefetch", "") not in by_selection:
            continue
        base_arm = by_selection[selection.replace(",qkv_prefetch", "")]
        base = _PDL_EDGES.get((base_arm, rows))
        if base is None:
            found.append(f"{arm}@{rows}: no graph of {base_arm} was read")
            continue
        have, want = prefetch_edges(edges), prefetch_edges(base)
        want += _expected_counts(selection, rows, "builtin")["qkv_prefetch_prefetch"]
        table[f"{arm}@{rows}"] = {"prefetch_out": have, "base": base_arm, "want": want}
        if have != want:
            found.append(f"{arm}@{rows}: {have} programmatic edges out of a prefetch, want {want}")
    if not table:
        found.append("no qkv_prefetch graph was read")
    if _SET == "tier7":
        compared = {key.rsplit("@", 1)[0] for key in table}
        found += [
            f"{arm}: no graph compared" for arm in TIER7_QKV_PREFETCH_ARMS if arm not in compared
        ]
    _RESULTS["qkv_prefetch_prefetch_edges"] = table
    _assert_on_every_rank(e, found, "qkv_prefetch prefetch -> QKV edges")


def test_record_the_transitivity_census(env):
    """Per captured graph: the programmatic edges, and for consecutive post-attention hop calls and
    consecutive fused finalize calls the kernels whose wait-before-exit carries "call n is
    complete when call n + 1's wait returns" on the path that needs fewest of them, and the
    plain launches between the two calls (``_census``). Recorded; every rank's graph must
    agree."""
    e = env
    table = {f"{arm}@{rows}": census for (arm, rows), census in sorted(_GRAPH_CENSUS.items())}
    summary = {
        key: {
            label: [{"waits": row["waits"], "plain": sorted(row["plain"])} for row in census[label]]
            for label in ("hop", "finalize")
        }
        for key, census in table.items()
    }
    gathered = e.comm.allgather(json.dumps(summary, sort_keys=True))
    found = [] if all(g == gathered[0] for g in gathered) else ["the ranks' graphs differ"]
    _RESULTS["graph_census"] = table
    _RESULTS["transitivity"] = summary
    _assert_on_every_rank(e, found, "transitivity census")


def test_the_router_fork_side_launches_are_forked_from_the_hop_and_joined_into_the_routed_pool(env):
    """In every captured graph of an arm with ``router_fork`` at most 8 rows wide, each of the
    four NVFP4-expert layers has one side launch of the router shard, entered by exactly one
    full edge from that layer's attention hop (the post-attention hop, or ``direct_publish``'s
    consumer hop) and
    left by exactly one programmatic edge, into the routed pool kernel (whose wait the
    logits are read after); the only other edges are full ones between consecutive side
    launches (the side stream's order: none into the first, none out of the last, one
    into each other); wider graphs and arms without ``router_fork`` have none. With
    ``pool_direct`` the 8-row graphs' side launches are its publish twin
    (``POOL_DIRECT_SIDE_KERNEL``), joined into the
    direct routed kernel (``PoolDirectRouteKernel``) the same way. The edges are read
    with the test's delay kernels taken out (each one's predecessors joined to its
    successors by the successor's edge type)."""
    e = env
    found, table = [], {}
    sites, widest = ENTRIES["router_shard_side"][1:]
    for (arm, rows), record in sorted(_FORK_EDGES.items()):
        table[f"{arm}@{rows}"] = record
        selected = _selection(next(a[2] for a in ARMS if a[0] == arm))
        want = sites if "router_fork" in selected and rows <= widest else 0
        if record["side_launches"] != want:
            found.append(f"{arm}@{rows}: {record['side_launches']} side launches, want {want}")
        if not want:
            continue
        direct = "pool_direct" in selected and rows == 8
        side_kernel = POOL_DIRECT_SIDE_KERNEL if direct else ROUTER_FORK_SIDE_KERNEL
        joined = "PoolDirectRouteKernel:P" if direct else "PoolGatherRouteKernel:P"
        if record["kernels"] != [side_kernel]:
            found.append(f"{arm}@{rows}: side kernels {record['kernels']}, want {side_kernel}")
        chained = f"{side_kernel}:F"
        ends = {"into": 0, "out": 0}
        for way in ("into", "out"):
            for edges, count in record[way].items():
                parts = edges.split()
                main = [p for p in parts if p != chained]
                if len(main) == len(parts):
                    ends[way] += count
                if way == "into":
                    fits = len(main) == 1 and main[0].endswith(":F") and _HOP_KERNEL.search(main[0])
                else:
                    fits = main == [joined]
                if not fits or len(parts) - len(main) > 1:
                    found.append(f"{arm}@{rows}: {count} side launches {way} {edges!r}")
        if ends != {"into": 1, "out": 1}:
            found.append(f"{arm}@{rows}: the side launches are not one chain: ends {ends}")
    router_fork_sets = ("router_fork", "router_fork_tier5", "pool_direct", "tier7")
    if _SET in router_fork_sets and not any(r["side_launches"] for r in _FORK_EDGES.values()):
        found.append("no captured graph had a router_fork side launch")
    if _SET in ("pool_direct", "tier7") and not any(
        r["kernels"] == [POOL_DIRECT_SIDE_KERNEL] for r in _FORK_EDGES.values()
    ):
        found.append("no captured graph had a pool_direct publish twin")
    _RESULTS["router_fork_fork_edges"] = table
    _assert_on_every_rank(e, found, "router_fork fork / join edges")


def test_the_swa_block_block_is_a_programmatic_chain(env):
    """At every sliding-window layer of each sliding-window block arm's M <= 8 graphs without
    delay kernels (seed 0): one programmatic edge into ``_swa_split_kernel`` (from the QKV GEMM,
    or from prep without ``swa_fused_prep``; never from prep under ``swa_fused_prep``), one split
    -> combine and one out of the combine (to o_proj, or to ``early_release``'s norm-gate). Arms
    without a block id have no block kernel in their graphs."""
    e = env
    found, table = [], {}
    sites, widest = ENTRIES["swa_block_attention"][1:]
    for (arm, rows), edges in sorted(_PDL_EDGES.items()):
        _, _key, selection, _head, _reference, seed = next(a for a in ARMS if a[0] == arm)
        selected = _selection(selection)
        into = {}
        out = {}
        chained = 0
        for (source, target, _port), count in edges.items():
            if "_swa_split_kernel" in target:
                into[source] = into.get(source, 0) + count
            if "_swa_combine_kernel" in source:
                out[target] = out.get(target, 0) + count
            if "_swa_split_kernel" in source and "_swa_combine_kernel" in target:
                chained += count
        table[f"{arm}@{rows}"] = {
            "into_split": into,
            "split_to_combine": chained,
            "out_of_combine": out,
        }
        if not selected & SWA_BLOCK_IDS:
            if into or chained or out:
                found.append(f"{arm}@{rows}: swa_block kernels in a graph without a swa_block id")
            continue
        if seed != 0 or rows > widest:
            continue
        from_prep = sum(n for source, n in into.items() if "_preattn_prep_kernel" in source)
        if sum(into.values()) != sites:
            found.append(f"{arm}@{rows}: programmatic edges into the split {into}, want {sites}")
        if "swa_fused_prep" in selected and from_prep:
            found.append(f"{arm}@{rows}: prep still feeds the split under swa_fused_prep: {into}")
        if "swa_fused_prep" not in selected and from_prep != sites:
            found.append(
                f"{arm}@{rows}: the split is not prep's dependent without swa_fused_prep: {into}"
            )
        if chained != sites:
            found.append(f"{arm}@{rows}: {chained} split -> combine edges, want {sites}")
        if sum(out.values()) != sites:
            found.append(f"{arm}@{rows}: programmatic edges out of the combine {out}, want {sites}")
    if _SET in (
        "swa_block",
        "tier4",
        "unpool_direct_tier4",
        "qkv_prefetch_tier4",
        "tier5",
        "split_kv_tier4",
        "router_fork",
        "router_fork_tier5",
        "prep_fold",
        "pool_direct",
        "tier7",
    ) and not any(key.endswith("_nd@8") for key in table):
        found.append("no seed-0 swa_block graph was read")
    _RESULTS["swa_block_edges"] = table
    _assert_on_every_rank(e, found, "swa_block programmatic edges")


def _cubin_dump_dirs() -> list:
    """The CuTe DSL dump directories (the DSL and its experimental variant, which flashinfer's
    split-K GEMM is compiled with) this run keeps cubins in."""
    dirs = []
    for prefix in ("CUTE_DSL", "CUTE_EXPERIMENTAL_DSL"):
        keep = os.environ.get(f"{prefix}_KEEP", "").lower()
        keeps = "cubin" in keep or "all" in keep or os.environ.get(f"{prefix}_KEEP_CUBIN") == "1"
        directory = os.environ.get(f"{prefix}_DUMP_DIR")
        if keeps and directory and os.path.isdir(directory):
            dirs.append(directory)
    return dirs


_FINALIZE_CUBINS_SEEN: set = set()


def _keep_finalize_cubins(arm: str) -> None:
    """Copy each new ``FinalizeTransportKernel`` binary of the dump directories to
    ``<dump>/by-arm/<arm>-<sha256 prefix>.cubin``. The DSL names a dump after the kernel only,
    so the trigger variants (the late one, ``finalize_trigger``'s entry one) overwrite one another;
    the copies keep every variant a model build (the workspace compiles at install) or an arm
    compiled for the SASS checks that run later."""
    for root in _cubin_dump_dirs():
        for directory, _, files in os.walk(root):
            if os.path.basename(directory) == "by-arm":
                continue
            for name in files:
                if not name.endswith(".cubin") or "FinalizeTransportKern" not in name:
                    continue
                path = os.path.join(directory, name)
                with open(path, "rb") as f:
                    blob = f.read()
                digest = hashlib.sha256(blob).hexdigest()
                if digest in _FINALIZE_CUBINS_SEEN:
                    continue
                _FINALIZE_CUBINS_SEEN.add(digest)
                os.makedirs(os.path.join(root, "by-arm"), exist_ok=True)
                with open(os.path.join(root, "by-arm", f"{arm}-{digest[:12]}.cubin"), "wb") as f:
                    f.write(blob)


def _block_threads(graph) -> dict:
    """{kernel symbol: threads per block} of a kept graph's kernel nodes (the largest seen)."""
    from cuda.bindings import driver

    g = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(g, 0)
    _, nodes, _ = driver.cuGraphGetNodes(g, int(count))
    out = {}
    for node in nodes:
        _, kind = driver.cuGraphNodeGetType(node)
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        _, params = driver.cuGraphKernelNodeGetParams(node)
        status, raw = driver.cuFuncGetName(params.func)
        if status != driver.CUresult.CUDA_SUCCESS:
            status, raw = driver.cuKernelGetName(params.kern)
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        threads = int(params.blockDimX) * int(params.blockDimY) * int(params.blockDimZ)
        out[raw] = max(out.get(raw, 0), threads)
    return out


#: Kernels with an exit before their wait that every CTA of a launch takes or none does (its
#: guard reads only constant bank 0), the constants it reads, and why no launch in a decode
#: graph takes it.
LAUNCH_UNIFORM_EXITS = {
    "FinalizeTransportKernel": {
        "c[0x0][0x414]": "the dry flag (the kernel's last parameter): FinalizeTransportWorkspace "
        "launches dry = 1 once, eagerly, while it is built; finalize() always passes dry = 0",
    },
    "PostAttnConsumerKernel": {
        "c[0x0][0x3c8]": "the dry flag (the kernel's last parameter): PostAttnDirectWorkspace "
        "launches dry = 1 once per compiled variant, eagerly; launch() always passes dry = 0",
    },
    "UnpoolDirectBoundaryKernel": {
        "c[0x0][0x3dc]": "the dry flag (the kernel's last parameter): UnpoolDirectWorkspace "
        "launches dry = 1 once per variant, eagerly, when it builds it (kernel()); the op "
        "always passes dry = 0",
    },
    "PoolDirectRouteKernel": {
        "c[0x0][0x420]": "the dry flag (the kernel's last parameter): PoolDirectWorkspace "
        "launches dry = 1 once per variant, eagerly, when it builds it (kernel()); "
        "pool_direct_route always passes dry = 0",
    },
}


def test_the_waits_between_calls_are_audited_kernels(env):
    """Every pair of consecutive hop (finalize) calls with no plain launch between them
    (``SIDE_LAUNCHES`` aside) has a path between the two calls whose programmatically
    entered kernels all match ``WAIT_BEFORE_EXIT`` or are entered from a kernel of
    ``NO_EARLY_TRIGGER``; an arm with ``hop``, ``persistent_experts`` and ``moe_prologue``
    has such pairs in every M <= 8 graph, and with ``MODELING_V2_HOOK_STREAM=main`` (the
    negative control: the hook's copies are plain launches between the calls) no graph has
    any. With CuTe DSL cubin dumps, rank 0 audits every
    kernel those paths rely on, as this run compiled it (the dumps, its Triton cache; for a
    kernel in neither, the audits in ``MODELING_V2_SASS_AUDIT_EXTRA``, e.g. cuBLAS's ``nvjet``
    read with Nsight Compute): a warp leader of CTA 0 executes ``ACQBULK`` on every path
    before it can exit, any launch-uniform early exit is one of ``LAUNCH_UNIFORM_EXITS``,
    and a ``NO_EARLY_TRIGGER`` kernel has no ``PREEXIT`` in any binary of it."""
    e = env
    found, pairs, relied, no_trigger = [], {}, {}, {}
    for (arm, rows), census in sorted(_GRAPH_CENSUS.items()):
        selection = _selection(next(a[2] for a in ARMS if a[0] == arm))
        for label in ("hop", "finalize"):
            rows_all_pdl = [row for row in census[label] if row["all_pdl"]]
            pairs[f"{arm}@{rows} {label}"] = len(rows_all_pdl)
            if not HOOK_SIDE:
                # The negative control: the hook's plain copies on the capturing stream
                # sit between every two calls.
                if rows_all_pdl:
                    found.append(
                        f"{arm}@{rows} {label}: {len(rows_all_pdl)} pairs without plain "
                        "launches although the hook copies on the capturing stream"
                    )
            elif (
                {"hop", "persistent_experts", "moe_prologue"} <= selection
                and rows <= 8
                and not rows_all_pdl
            ):
                found.append(f"{arm}@{rows} {label}: no pair without plain launches")
            for row in rows_all_pdl:
                if row["audited_path"] is None:
                    found.append(
                        f"{arm}@{rows} {label} {row['pair']}: no path through audited kernels; "
                        f"unaudited waits {row['unaudited']}"
                    )
                relied.update(row["audited_waits"])
                no_trigger.update(row["audited_no_trigger"])
    _RESULTS["all_pdl_pairs"] = pairs
    _RESULTS["audited_kernels_relied_on"] = sorted(set(relied.values()))
    _RESULTS["no_trigger_kernels_relied_on"] = sorted(set(no_trigger.values()))
    _assert_on_every_rank(e, found, "paths through audited kernels")

    dumps = _cubin_dump_dirs()
    if not all(e.comm.allgather(bool(dumps))):
        _RESULTS["sass_wait_audit"] = "not run: no CuTe DSL cubin dump on every rank"
        pytest.skip("the SASS audit needs CUTE_DSL_KEEP=cubin and CUTE_DSL_DUMP_DIR")
    found, table = [], {}
    if e.rank == 0 and (relied or no_trigger):
        triton_cache = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
        wanted = {**no_trigger, **relied}
        audits = collect_sass(wanted, dumps + [triton_cache], threads=_BLOCK_THREADS)
        extra_path = os.environ.get("MODELING_V2_SASS_AUDIT_EXTRA")
        extra = {}
        if extra_path:
            with open(extra_path) as f:
                extra = json.load(f)
        for raw, short in sorted(wanted.items(), key=lambda item: item[1]):
            results = audits.get(raw, [])
            if not results and raw in extra:
                results = [(r["source"], r) for r in extra[raw]]
            if not results:
                found.append(
                    f"{short}: no SASS in the cubin dumps, the Triton cache or {extra_path}"
                )
            allowed = LAUNCH_UNIFORM_EXITS.get(short, {})
            for source, result in results:
                if raw in relied and not result["ok"]:
                    found.append(f"{short} ({source}): {result}")
                if raw in no_trigger and result["preexit"]:
                    found.append(f"{short} ({source}): {result['preexit']} PREEXIT, relied on none")
                for early in result["launch_uniform_exits"]:
                    if raw in relied and not set(early["reads"]) <= set(allowed):
                        found.append(f"{short} ({source}): launch-uniform exit {early}")
            table[raw] = {
                "kernel": short,
                "relied_on": [
                    what
                    for what, kernels in (("wait_before_exit", relied), ("no_trigger", no_trigger))
                    if raw in kernels
                ],
                "block_threads": _BLOCK_THREADS.get(raw),
                "binaries": [{"source": s, **r} for s, r in results],
            }
    _RESULTS["sass_wait_audit"] = table
    _assert_on_every_rank(e, found, "SASS wait before exit")


def test_write_the_result_table(env):
    """Rank 0 writes the per-arm results and the probe table when asked to."""
    e = env
    assert _RESULTS, "the schedule tests recorded nothing"
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and e.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        report = {
            "steps": len(e.schedule),
            "bursts": len(e.accept),
            "schedule_kinds": {
                kind: sum(1 for s in e.schedule if s[0] == kind) for kind in ("replay", "eager")
            },
            "schedule_rows": {str(m): sum(1 for s in e.schedule if s[1] == m) for m in ROWS},
            "set": _SET,
            "layout": e.layout,
            "stale_poll_schedules": e.stale_poll_on,
            "arm_set": _SET,
            "hook_stream": "side" if HOOK_SIDE else "main",
            "transitivity": _RESULTS.get("transitivity"),
            "head": {
                "draft_head_installed": e.head.draft_head_installed,
                "worker": e.head.worker_class,
                "fc_shard": e.head.fc_shard_reason,
            },
            "free_gib_after_build": e.free_gib,
            "prepare_stream_aliased": {
                f"{arm}@{rows}": aliased
                for (arm, rows), aliased in sorted(_PREPARE_ALIASED.items())
            },
            "capture_stream": "dedicated" if _capture_stream() is not None else "pooled default",
            "results": _RESULTS,
        }
        with open(os.path.join(out_dir, f"compose_{_SET}_schedule.json"), "w") as f:
            json.dump(report, f, indent=1, sort_keys=True, default=str)
