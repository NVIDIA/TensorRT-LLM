# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which fast paths a model of this target is built with, decided once at construction.

``FAST_PATHS`` names every lever of the target -- each fast path, and the refinements
and installs registered beside them -- and the levers each one builds on
(``depends_on``). A composition (``TargetPaths``) is the set of levers served on one
GPU architecture at one tensor-parallel size (its ``CompositionKey``), closed under
``depends_on``, with the GPU topology it was validated on; it applies when the
drafter is DFlash drafting a linear chain of ``DRAFT_LEN`` tokens on an NVFP4
checkpoint. Those facts are structural: a deployment that misses one has no
composition, and runs the built-in forward under ``modeling_v2="auto"``
(``NO_PATHS``) or raises under ``require``.

``compose`` resolves the composition's lever table (``levers.resolve``): the
structural rules of the deployment's drafter arena and facts
(``structural_rules``), then each lever's knob and deployment rule. The lever
table is the only environment ``compose`` reads (``levers.environment``). It then
checks the public preconditions the planned fast paths are written against
(``PRECONDITIONS``, and the composition's own), once, agreed over the TP group
together with the arena and the lever environment. A missing precondition, a lever
error or a disagreement, on any rank, raises ``ValueError`` on every rank, under
``auto`` as under ``require``: a composition is never served in part because of a
missing setting. The model's installers ask the resulting ``Plan`` per lever
(``Plan.enabled``, ``Plan.param``) and report what they installed and what a step
declined (``Plan.installed``, ``Plan.declined``).
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.logger import logger

from . import levers as _levers

#: The draft tokens per step of the one speculative-decoding mode the compositions serve.
DRAFT_LEN = _levers.DRAFT_LEN

#: The log tag of this family.
TAG = "[musespark_open]"

#: Every fast path launches its kernels with programmatic dependent launch:
#: ``TRTLLM_ENABLE_PDL`` on is a precondition of every composition (``PRECONDITIONS``).
LAUNCH_PDL = True


@dataclasses.dataclass(frozen=True)
class FastPath:
    """One lever of this target: what it changes, the levers it builds on, and when it installs.

    ``hook`` names the shell hook that installs it: ``construct`` (the shell's
    constructor, on the built model), ``load`` (after the target weights load),
    ``draft`` (after the drafter loads) or ``executor`` (the executor hooks).
    """

    name: str
    summary: str
    hook: str
    depends_on: Tuple[str, ...] = ()


#: Every fast path of this target, grouped by what it changes.
FAST_PATHS: Tuple[FastPath, ...] = (
    # Target attention.
    FastPath("preattn_prep", "decode pre-attention prep in one launch per layer", "construct"),
    FastPath(
        "swa_paged_decode",
        "sliding-window decode attention with the FP8 K/V write, Q in E4M3",
        "construct",
    ),
    FastPath(
        "swa_block",
        "sliding-window decode attention block with the prep and the output norm-gate fused",
        "construct",
        ("preattn_prep", "swa_paged_decode"),
    ),
    FastPath(
        "wide_swa_block",
        "sliding-window decode attention block of 2..4-request decode steps in one call",
        "construct",
        ("swa_block",),
    ),
    FastPath(
        "full_attention_prep_append",
        "full-attention layers' K/V append folded into the decode prep",
        "construct",
        ("preattn_prep",),
    ),
    FastPath(
        "full_attention_split_kv",
        "full-attention decode FMHA split over every SM, its spill-free build",
        "construct",
        ("full_attention_prep_append",),
    ),
    FastPath(
        "full_attention_prep_fold",
        "full-attention decode prep folded into the split-KV FMHA",
        "construct",
        ("full_attention_split_kv",),
    ),
    FastPath(
        "wide_full_attention",
        "full-attention folded prep, split-KV partials and merge of 2..4-request decode steps",
        "construct",
        ("full_attention_split_kv",),
    ),
    FastPath(
        "multi_request_split_kv",
        "decode FMHA of steps with two or more requests as balanced page-split partials",
        "construct",
    ),
    FastPath(
        "multi_request_prep_append",
        "every layer's K/V append folded into the decode prep of steps with two or more requests",
        "construct",
        ("preattn_prep",),
    ),
    FastPath(
        "mixed_step_gen_rows",
        "full-attention generation rows of mixed steps with 9..64 generation requests on "
        "multi_request_split_kv, context rows on the thop op",
        "construct",
        ("multi_request_split_kv",),
    ),
    # Dense projections and prefetch.
    FastPath("dense_gemms", "dense projections on one named kernel per shape", "construct"),
    FastPath(
        "wide_dense_gemms",
        "dense projections of 9..32-row decode steps on flashinfer's split-K at sm107 tep4",
        "construct",
        ("dense_gemms",),
    ),
    FastPath(
        "dense_gemm_prefetch",
        "L2 prefetch of the next dense projection's weight",
        "construct",
        ("dense_gemms",),
    ),
    FastPath(
        "next_qkv_prefetch",
        "next layer's QKV weight prefetched from the expert call",
        "construct",
        ("dense_gemms", "boundary_early_release"),
    ),
    FastPath(
        "batch_dense_gemms",
        "33..512-row decode steps: QKV on the TGV GEMM at 144..256 rows, o_proj weight L2 "
        "prefetch and PDL attention norm-gate in their row bands",
        "construct",
        ("dense_gemms", "boundary_early_release"),
    ),
    # Hops and boundaries.
    FastPath(
        "post_attention_hop",
        "attention-side all-reduce, residual join and pre-FFN norm in one launch",
        "construct",
    ),
    FastPath(
        "wide_post_attention_hop",
        "post-attention hop of 9..32-row decode steps on a wide TP4 transport",
        "construct",
        ("post_attention_hop",),
    ),
    FastPath(
        "attention_output_into_hop",
        "attention output projection writing into the hop's transport",
        "construct",
        ("post_attention_hop", "dense_gemms"),
    ),
    FastPath(
        "multi_request_post_attention_hop",
        "the post-attention hop on 9..1024-row decode steps (several requests)",
        "construct",
        ("post_attention_hop",),
    ),
    FastPath(
        "projection_gathers",
        "projection column gathers fused into their consumers",
        "construct",
    ),
    FastPath(
        "boundary_early_release",
        "post-FFN boundary and norm-gate releasing the next GEMM early",
        "construct",
        ("projection_gathers",),
    ),
    FastPath(
        "unpool_into_boundary",
        "unpool projection writing into the post-FFN boundary's transport",
        "construct",
        ("projection_gathers", "dense_gemms"),
    ),
    FastPath(
        "capture_fold",
        "DFlash capture written by the post-FFN boundary",
        "construct",
        ("projection_gathers",),
    ),
    FastPath(
        "projection_publish",
        "sharded pool/unpool GEMMs publishing into multicast transports their consumers poll, "
        "33..512-row decode steps",
        "construct",
        ("projection_gathers",),
    ),
    # MoE.
    FastPath(
        "moe_prologue",
        "MoE input prep as one chain with the router shard on a side stream",
        "construct",
        ("projection_gathers", "post_attention_hop"),
    ),
    FastPath(
        "pool_into_route",
        "pool projection and router shard writing into the routed kernel's transport",
        "construct",
        ("moe_prologue", "dense_gemms"),
    ),
    FastPath(
        "wide_prologue",
        "MoE prologue and projection gathers of 16..32-row decode steps on wide TP4 inboxes",
        "construct",
        ("moe_prologue",),
    ),
    FastPath(
        "finalize_transport",
        "expert finalize all-reduce on the fused hop's transport",
        "construct",
    ),
    FastPath(
        "wide_finalize_transport",
        "expert finalize all-reduce of 17..32-token calls on a 32-token TP4 transport layout",
        "construct",
        ("finalize_transport",),
    ),
    FastPath(
        "router_gemm",
        "router logits of 9..512-row calls as a BF16x3 split-K tensor-core GEMM",
        "construct",
    ),
    FastPath(
        "wide_moe_allreduce",
        "MoE output all-reduce of 129..1024-row decode steps on the MNNVL two-shot",
        "construct",
    ),
    FastPath(
        "finalize_two_shot",
        "expert finalize and all-reduce of 64..512-row decode steps in one two-shot kernel",
        "construct",
        ("finalize_transport",),
    ),
    # Context steps.
    FastPath(
        "context_sequence_parallel",
        "large context steps' norms, residual joins, router and pool/unpool on a row shard",
        "construct",
    ),
    FastPath(
        "context_prep",
        "context steps' pre-attention prep in one launch per layer, padded graph rows zeroed",
        "construct",
        ("preattn_prep",),
    ),
    FastPath("replicated_experts", "ring-buddy expert replication, balanced", "load"),
    # persistent_experts and bf16_routed_experts run over the ring-buddy stack with
    # replicated_experts, else over the rank's own expert block (the owner-only window).
    FastPath("persistent_experts", "NVFP4 experts as one persistent launch", "load"),
    FastPath(
        "wide_persistent_experts",
        "NVFP4 experts of 9..16-row decode steps on the persistent kernel's wide build",
        "load",
        ("persistent_experts",),
    ),
    FastPath(
        "half_window_experts",
        "the ring predecessor's upper expert half on each TP4 rank, balanced per step",
        "load",
        ("persistent_experts", "moe_prologue"),
    ),
    FastPath(
        "three_window_placement",
        "a third ring window for the replicated experts",
        "load",
        ("replicated_experts", "persistent_experts", "moe_prologue"),
    ),
    FastPath("bf16_routed_experts", "BF16 experts given unpacked routing", "load"),
    FastPath(
        "hot_replicas",
        "hot NVFP4 expert copies placed per step on 33..512-row decode steps",
        "load",
    ),
    FastPath(
        "early_expert_ids",
        "persistent expert kernel starting from early-published ids",
        "load",
        ("persistent_experts", "moe_prologue"),
    ),
    FastPath("replicated_embedding", "token embedding gathered from a replicated table", "load"),
    FastPath(
        "model_boundary",
        "decode input boundary (gather, embed norm, residual, layer 0 pre-QKV norm) in one launch; "
        "last-layer join storing BF16",
        "load",
        ("replicated_embedding",),
    ),
    # Drafter.
    FastPath("drafter_block_conv", "drafter block convolutions as one fused kernel", "draft"),
    FastPath(
        "drafter_block_decode",
        "drafter block decode over the window's keys with the K/V append fused",
        "draft",
    ),
    FastPath(
        "wide_drafter_window_attention",
        "drafter window attention of 33..1024-row block decodes on a multi-request split geometry",
        "draft",
        ("drafter_block_decode",),
    ),
    FastPath("draft_head", "drafter LM head on vocabulary shards", "draft"),
    FastPath(
        "wide_draft_topk",
        "draft-head key exchange of draft steps of 33 or more rows: each rank's own top-k",
        "draft",
        ("draft_head",),
    ),
    FastPath("drafter_fc_shard", "drafter fc projection column-sharded", "draft"),
    FastPath(
        "drafter_fc_inbox",
        "TP4 sharded drafter fc gathered over the tray's NVLink",
        "draft",
        ("drafter_fc_shard",),
    ),
    FastPath("fused_accept", "acceptance and forced acceptance in one launch", "draft"),
    FastPath("drafter_fixed_gemms", "drafter GEMMs on fixed low-M tactics", "draft"),
    FastPath("context_kv_fork", "drafter context-K/V store on a side stream", "draft"),
    FastPath(
        "verify_fold",
        "step bookkeeping folded into the acceptance kernel",
        "draft",
        ("fused_accept",),
    ),
    FastPath(
        "paged_context_commit",
        "the context-K/V fork's commit over a paged drafter context in one launch",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "paged_ctx_block_tables",
        "the context-K/V fork's decode of a paged drafter context's page table in two launches",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "paged_ctx_kv_store",
        "a paged drafter context's K/V store of every drafter layer in one launch",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "rejection_fold",
        "a rejection-sampled step's bookkeeping folded into one launch after the rejection sampler",
        "draft",
        ("verify_fold",),
    ),
    FastPath(
        "path_walk_tail",
        "step bookkeeping folded into the draft-head walk",
        "draft",
        ("draft_head",),
    ),
    FastPath(
        "draft_head_sampled",
        "sampled drafts (rejection sampling) on the draft head's lattice, walked in one launch",
        "draft",
        ("draft_head",),
    ),
    FastPath(
        "drafter_hops",
        "drafter all-reduce hops, releasing dependents early",
        "draft",
        ("drafter_block_decode",),
    ),
    FastPath(
        "wide_drafter_hops",
        "the drafter all-reduce hops of 9..56-row (sm107: 9..64-row) block decodes on a wide "
        "one-shot transport",
        "draft",
        ("drafter_hops",),
    ),
    FastPath(
        "drafter_pdl_chain",
        "drafter and draft head launches releasing their GEMMs early",
        "draft",
        (
            "drafter_block_conv",
            "drafter_block_decode",
            "drafter_fixed_gemms",
            "drafter_hops",
            "draft_head",
        ),
    ),
    FastPath("split_sampler", "target head verify sampler split over two kernels", "draft"),
    FastPath(
        "wide_split_sampler",
        "the split verify sampler on 129..1024-row decode steps (the shipped 512-thread block)",
        "draft",
        ("split_sampler",),
    ),
    FastPath(
        "lm_head_inbox",
        "LM-head GEMM into a multicast inbox the sampler gathers",
        "draft",
        ("split_sampler",),
    ),
    FastPath(
        "vocab_parallel_head",
        "the verify head of 16..1024-row decode steps on each rank's own vocabulary shard (no logits all-gather)",
        "draft",
        ("split_sampler",),
    ),
    # Executor.
    FastPath("deferred_input_prep", "a decode step's input writes as one launch", "executor"),
    FastPath("sampler_store", "the sampler's store update and snapshots as one launch", "executor"),
    # Levers on the paths above (``LEVER_NAMES``): each changes how the path it builds on
    # runs, or is an install the plan carries; their rules are ``levers.RULES``.
    FastPath(
        "packed_prep_append",
        "multi-request decode prep-append calls of MIN_ROWS rows or more on the packed kernel",
        "construct",
        ("multi_request_prep_append",),
    ),
    FastPath(
        "windowed_commit_parallel",
        "windowed context-K/V commits of MIN_REQUESTS requests or more as a parallel zeroing "
        "and a tuned compaction",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "context_fp8_dense",
        "the dense projections of context calls of MIN_ROWS rows or more on the Rubin MXFP8 "
        "GEMM (lossy)",
        "load",
    ),
    FastPath(
        "context_fp8_attn_gather",
        "the sequence-parallel attention input all-gathered in MXFP8 (lossy)",
        "load",
        ("context_fp8_dense", "context_sequence_parallel"),
    ),
    FastPath(
        "wide_direct_transports",
        "the pool and unpool direct transports of 16..32-row captured decode steps",
        "construct",
        ("wide_prologue", "pool_into_route", "unpool_into_boundary"),
    ),
    FastPath(
        "wide_router_per_row",
        "the router shard of 16..32-row decode steps with each row's one-request logits",
        "construct",
        ("wide_prologue",),
    ),
    FastPath(
        "wide_oproj_publish",
        "o_proj of 16..32-row decode steps publishing into the hop's 32-row direct transport",
        "construct",
        ("attention_output_into_hop", "wide_post_attention_hop"),
    ),
    FastPath(
        "graph_start_qkv_prefetch",
        "layer 0's QKV weight prefetch forked at the start of a decode forward",
        "construct",
        ("next_qkv_prefetch",),
    ),
    FastPath(
        "step_loop_on_execution_stream",
        "the executor's overlap loop issuing the sample and prepare launches on the execution "
        "stream",
        "executor",
        ("deferred_input_prep",),
    ),
    FastPath(
        "incremental_block_offsets",
        "the staged block-offset copy of only the columns whose page index changed",
        "executor",
        ("deferred_input_prep",),
    ),
    FastPath(
        "drafter_kv_side_chain",
        "the context-K/V side chain with its prologue and arena store fused (dense arena or pages)",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "drafter_fc_early_fork",
        "the side chain's local fc GEMM forked after the last DFlash capture layer",
        "draft",
        ("drafter_kv_side_chain", "drafter_fc_shard"),
    ),
    FastPath(
        "drafter_preamble_from_commit",
        "the drafter forward's integer preamble written by the folded step's commit (dense arena or pages)",
        "draft",
        ("verify_fold", "context_kv_fork"),
    ),
    FastPath(
        "head_sampler_coop_fold",
        "the head sampler's kernel A with the soft cap computed by the whole CTA, head calls "
        "of MAX_ROWS rows or fewer",
        "draft",
        ("split_sampler",),
    ),
    FastPath(
        "head_multimem_gather",
        "the 8-row head logits all-gathered over NVLink multicast (MULTIMEM)",
        "draft",
        ("split_sampler",),
    ),
    FastPath(
        "prefill_cutedsl_experts",
        "the NVFP4 experts of 4096..8192-row calls on the Rubin CuTe DSL GEMMs",
        "load",
    ),
    # The Vera Rubin low-latency levers (sm107 compositions, ``levers.RULES``).
    FastPath(
        "splitk_pool_split8",
        "the pool GEMM of 9..32-row TP4 decode calls summing K in 8 slices, with the split-K "
        "kernel's reduce-scatter epilogue",
        "construct",
        ("wide_dense_gemms",),
    ),
    FastPath(
        "splitk_frontier_tables",
        "the TP4 dense projections of 1..32-row decode calls on the low-latency split-K and "
        "direct tactic tables",
        "construct",
        ("wide_dense_gemms",),
    ),
    FastPath(
        "tep8_wide64_splitk",
        "the TP8 QKV and o_proj of 33..64-row decode calls on split-K",
        "construct",
        ("wide_dense_gemms",),
    ),
    FastPath(
        "tep8_publish_splitk",
        "the TP8 pool and unpool slices of 33..64-row decode calls published from split-K",
        "construct",
        ("projection_publish",),
    ),
    FastPath(
        "tep8_drafter_wide_gemms",
        "the TP8 drafter projections of 2..4-request blocks on split-K",
        "draft",
        ("drafter_fixed_gemms",),
    ),
    FastPath(
        "tep8_fc_shard_wide",
        "the TP8 sharded drafter fc on decode steps of up to 64 rows, its gather on 128 programs",
        "draft",
        ("drafter_fc_shard",),
    ),
    FastPath(
        "weight_prefetch_after_join",
        "dense weight prefetches joined after their consumer, the QKV prefetch dropped",
        "construct",
        ("dense_gemm_prefetch",),
    ),
    FastPath(
        "next_qkv_prefetch_skip",
        "no next-layer QKV prefetch on decode steps of at most 8 rows",
        "construct",
        ("next_qkv_prefetch",),
    ),
    FastPath(
        "swa_one_cluster_block",
        "the sliding-window decode attention block of one to eight requests as one cluster launch",
        "construct",
        ("swa_block",),
    ),
    FastPath(
        "swa_prep_split",
        "the sliding-window block's prep split off steps of two or more requests whose split "
        "grid spans several waves",
        "construct",
        ("swa_block",),
    ),
    FastPath(
        "swa_eight_requests",
        "the sliding-window block's merged split kernel: TP8 decode steps of up to eight requests, "
        "and the one-cluster block's fallback",
        "construct",
        ("wide_swa_block",),
    ),
    FastPath(
        "full_attention_fold_multi",
        "the full-attention decode prep folded into the split-KV FMHA on steps of two to four "
        "requests",
        "construct",
        ("full_attention_prep_fold",),
    ),
    FastPath(
        "split_kv_deep_rings",
        "the split-KV decode FMHA kernels on a 20-stage TMA ring",
        "construct",
        ("full_attention_split_kv", "multi_request_split_kv"),
    ),
    FastPath(
        "bf16_half_window",
        "the BF16 expert layers on the half expert window",
        "load",
        ("half_window_experts",),
    ),
    FastPath(
        "prerouted_experts",
        "the NVFP4 experts on trtllm::fp4_block_scale_moe_prerouted, from the placement "
        "kernels' routing tables",
        "load",
        ("persistent_experts",),
    ),
    FastPath(
        "tep8_ring_buddy_wide",
        "the TP8 ring-buddy expert placement on decode steps of up to 64 rows",
        "load",
        ("replicated_experts",),
    ),
    FastPath(
        "tep8_ring_buddy_pdl",
        "the TP8 ring-buddy table writer launched with programmatic dependent launch",
        "load",
        ("replicated_experts",),
    ),
    FastPath(
        "tep8_tile_placement",
        "the TP8 replicated experts placed by tile cost on decode steps of 41 or more rows",
        "load",
        ("replicated_experts",),
    ),
    FastPath(
        "tep8_router_push",
        "the TP8 router columns and their all-gather in one kernel, routing folded into its "
        "epilogue, on 33..128-row steps",
        "construct",
        ("router_gemm",),
    ),
    FastPath(
        "tep8_router_prewait",
        "the TP8 router GEMM loading its weight before its dependency wait",
        "construct",
        ("router_gemm",),
    ),
    FastPath(
        "finalize_entry_trigger",
        "the expert finalize's dependents released at its entry",
        "construct",
        ("finalize_transport",),
    ),
    FastPath(
        "tep8_finalize_transport_64",
        "the TP8 expert finalize of 17..64-token calls on a 64-token transport",
        "construct",
        ("finalize_transport",),
    ),
    FastPath(
        "hop_entry_trigger",
        "the 8-row post-attention hop kernels' dependents released at their entry",
        "construct",
        ("post_attention_hop",),
    ),
    FastPath(
        "tep8_early_multicast_hop",
        "the TP8 post-attention hop of 9..64-row steps launched early, gathered over multicast, "
        "two rows per CTA from 41 rows",
        "construct",
        ("multi_request_post_attention_hop",),
    ),
    FastPath(
        "tep8_direct_hop",
        "the TP8 o_proj of 33..64-row steps scattering its partial into the owners' hop slots",
        "construct",
        ("multi_request_post_attention_hop", "tep8_wide64_splitk"),
    ),
    FastPath(
        "tep8_publish_join",
        "the TP8 post-FFN join launched with programmatic dependent launch as one cluster",
        "construct",
        ("projection_publish",),
    ),
    FastPath(
        "tep8_pool_programmatic_fork",
        "the TP8 pool slice GEMM forked programmatically from its producer",
        "construct",
        ("projection_publish",),
    ),
    FastPath(
        "tep8_wide_projection_gathers",
        "the TP8 pool and unpool gathers of 9..32-row steps fused into their consumers",
        "construct",
        ("projection_gathers",),
    ),
    FastPath(
        "step_commit_v2",
        "the step commit's descriptors read as 64-byte rows, the host splitting each piece",
        "executor",
        ("deferred_input_prep",),
    ),
    FastPath(
        "wide_norm_gate_pdl",
        "the post-FFN norm-gates of 33..128-row decode steps launched with programmatic dependent "
        "launch",
        "construct",
        ("boundary_early_release",),
    ),
    FastPath(
        "drafter_kv_glue",
        "the context-K/V side chain's index math and arena stores in two launches (worker-owned "
        "arena)",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "drafter_kv_fork_before_head",
        "the context-K/V side chain forked before the LM-head GEMM",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "context_sp_extra_parts",
        "large context steps' router input and expert all-gather on the row shard too",
        "construct",
        ("context_sequence_parallel",),
    ),
    FastPath(
        "step_window_compact",
        "windowed drafter arena compaction in place on the steps the context-K/V fork leaves",
        "draft",
    ),
    FastPath(
        "fmk_context_fmha",
        "the context rows of mixed steps on the multi-instance paged context FMHA",
        "construct",
        ("mixed_step_gen_rows",),
    ),
    FastPath(
        "context_kv_late_join",
        "the context-K/V side chain joined right before the drafter's first block attention",
        "draft",
        ("context_kv_fork",),
    ),
    FastPath(
        "hop_cluster_split",
        "a row of the wide post-attention hop's calls of ROWS rows on CLUSTER CTAs",
        "construct",
        ("multi_request_post_attention_hop",),
    ),
    FastPath(
        "drafter_fc_inbox_wide",
        "the drafter fc inbox gather sized for MAX_ROWS rows, on PROGRAMS programs",
        "draft",
        ("drafter_fc_inbox",),
    ),
    FastPath(
        "l061_tile32_pin",
        "the 32-token tile pinned for layer 0 / 61's owner-window expert calls of 256..384 rows",
        "load",
        ("hot_replicas",),
    ),
    FastPath(
        "router_staged_epilogue",
        "the router GEMM's partial tiles written through shared memory",
        "construct",
        ("router_gemm",),
    ),
    FastPath(
        "decode_qkv_mxfp8",
        "the QKV projections of published decode steps on the MXFP8 GEMM (lossy)",
        "load",
        ("context_fp8_dense", "projection_publish", "batch_dense_gemms"),
    ),
    FastPath(
        "split_kv_deep_ring",
        "sm107's E4M3-query multi-request split-KV decode on a 20-page ring",
        "construct",
        ("multi_request_split_kv",),
    ),
    FastPath(
        "context_prep_rows_vec",
        "sm107 context steps' pre-attention prep on the vectorized rows kernel",
        "construct",
        ("context_prep",),
    ),
    FastPath(
        "context_embedding_replicated",
        "sm107 context steps' token embedding gathered from the replicated table",
        "load",
        ("replicated_embedding",),
    ),
    FastPath(
        "context_sp_router_input",
        "sm107 sequence-parallel steps' router FP32 input written by the residual join",
        "construct",
        ("context_sequence_parallel",),
    ),
    FastPath(
        "swa_blocks_per_program",
        "the sliding-window split programs of 2..4-sequence decode steps over as many key "
        "blocks each as the step has sequences",
        "construct",
        ("wide_swa_block",),
    ),
    FastPath(
        "wide_qkv_prefetch",
        "the next QKV weight's L2 prefetch of 9..32-row decode steps forked after the experts",
        "construct",
        ("next_qkv_prefetch", "wide_dense_gemms"),
    ),
    FastPath(
        "wide_capture_fold",
        "the DFlash capture of 16..32-row decode steps written by the boundary on the wide "
        "direct transport",
        "construct",
        ("capture_fold", "wide_direct_transports"),
    ),
    # The single-request decode schedule of sm107 TEP8 (``SM107_TEP8_ONLY``).
    FastPath(
        "megakernel",
        "decoder layers 1 to 60 of a one-sequence 8-row decode step as one launch per layer "
        "(the served build, its numerics-budget levers below)",
        "load",
        ("replicated_experts", "three_window_placement"),
    ),
    FastPath(
        "megakernel_one_exchange_boundaries",
        "the megakernel's MoE boundaries with one exchange, the second norm's statistic from "
        "the expanded square of the unrounded residual",
        "load",
    ),
    FastPath(
        "megakernel_two_term_pool",
        "the megakernel's pool and router GEMVs on the two BF16 terms of the gated residual, "
        "combined in FP32",
        "load",
    ),
    FastPath(
        "megakernel_router_by_linearity",
        "the megakernel's router logits formed from the o_proj partials in FP32",
        "load",
    ),
    FastPath(
        "megakernel_zero_exchange_boundary",
        "the megakernel's post-attention boundary without an exchange (no build of the served "
        "megakernel carries it)",
        "load",
    ),
    FastPath(
        "layer0_qkv_table",
        "layer 0's QKV rows of an 8-row decode step gathered from a per-token table, the "
        "residual's input chain forked beside the gather",
        "load",
        ("replicated_embedding", "dense_gemms"),
    ),
    FastPath(
        "edge_bf16_moe_front",
        "the BF16 expert layers' MoE front in the prologue and the pool transport",
        "construct",
        ("moe_prologue", "pool_into_route", "bf16_routed_experts"),
    ),
    FastPath(
        "last_layer_boundary",
        "the last layer's unpool, gated residual and final norm as one launch; the identity "
        "row select before the LM head skipped",
        "load",
        ("projection_gathers",),
    ),
    FastPath(
        "context_kv_fork_last_layer",
        "the context-K/V side chain forked on the last decoder layer after its experts",
        "draft",
        ("drafter_fc_early_fork",),
    ),
    FastPath(
        "distributed_head_sampler",
        "the 8-row verify head on an interleaved vocabulary copy and one-shot gathered logits, "
        "its verify fold released early",
        "draft",
        ("split_sampler",),
    ),
    FastPath(
        "drafter_preamble_late_join",
        "with the commit-written preamble, the context-K/V side chain joined inside the "
        "drafter forward before layer 0's QK norm and RoPE",
        "draft",
        ("drafter_preamble_from_commit", "drafter_block_decode"),
    ),
    FastPath(
        "drafter_folds",
        "drafter elementwise steps folded into its GEMVs, its layer 0 from a per-token table",
        "draft",
        (
            "drafter_block_conv",
            "drafter_block_decode",
            "drafter_fixed_gemms",
            "drafter_pdl_chain",
            "draft_head",
            "replicated_embedding",
        ),
    ),
    FastPath(
        "draft_head_tail",
        "the draft head's key selection, merge and path walk of a one-request draft step as "
        "three launches",
        "draft",
        ("drafter_folds", "path_walk_tail"),
    ),
    FastPath(
        "step_loop_inline_commits",
        "the one-stream step loop's commits with their descriptors in the kernel parameters, "
        "and the overlap offsets as one launch",
        "executor",
        (
            "step_loop_on_execution_stream",
            "incremental_block_offsets",
            "sampler_store",
            "layer0_qkv_table",
        ),
    ),
    FastPath(
        "rejection_sampler",
        "rejection sampling of a one-request decode step fused into the draft-head tail's walk "
        "and the distributed verify sampler, when the speculative config asks for it",
        "draft",
        ("distributed_head_sampler", "drafter_folds", "verify_fold", "path_walk_tail"),
    ),
)

#: The levers of the single-request decode schedule: built for sm107 at TP8 with EP8 (the
#: megakernel's eight-rank kernels and the compute_107a builds), and served by that
#: composition only.
SM107_TEP8_ONLY: FrozenSet[str] = frozenset(
    {
        "megakernel",
        "megakernel_one_exchange_boundaries",
        "megakernel_two_term_pool",
        "megakernel_router_by_linearity",
        "megakernel_zero_exchange_boundary",
        "layer0_qkv_table",
        "edge_bf16_moe_front",
        "last_layer_boundary",
        "context_kv_fork_last_layer",
        "distributed_head_sampler",
        "drafter_preamble_late_join",
        "drafter_folds",
        "draft_head_tail",
        "step_loop_inline_commits",
        "rejection_sampler",
    }
)

#: The name of every fast path.
FAST_PATH_NAMES: FrozenSet[str] = frozenset(path.name for path in FAST_PATHS)

#: The levers registered beside the fast paths (the last group of ``FAST_PATHS``).
LEVER_NAMES: FrozenSet[str] = frozenset(
    {
        "packed_prep_append",
        "windowed_commit_parallel",
        "context_fp8_dense",
        "context_fp8_attn_gather",
        "wide_direct_transports",
        "wide_router_per_row",
        "wide_oproj_publish",
        "graph_start_qkv_prefetch",
        "step_loop_on_execution_stream",
        "incremental_block_offsets",
        "drafter_kv_side_chain",
        "drafter_fc_early_fork",
        "drafter_preamble_from_commit",
        "head_sampler_coop_fold",
        "head_multimem_gather",
        "prefill_cutedsl_experts",
        "splitk_pool_split8",
        "splitk_frontier_tables",
        "tep8_wide64_splitk",
        "tep8_publish_splitk",
        "tep8_drafter_wide_gemms",
        "tep8_fc_shard_wide",
        "weight_prefetch_after_join",
        "next_qkv_prefetch_skip",
        "swa_one_cluster_block",
        "swa_prep_split",
        "swa_eight_requests",
        "full_attention_fold_multi",
        "split_kv_deep_rings",
        "bf16_half_window",
        "prerouted_experts",
        "tep8_ring_buddy_wide",
        "tep8_ring_buddy_pdl",
        "tep8_tile_placement",
        "tep8_router_push",
        "tep8_router_prewait",
        "finalize_entry_trigger",
        "tep8_finalize_transport_64",
        "hop_entry_trigger",
        "tep8_early_multicast_hop",
        "tep8_direct_hop",
        "tep8_publish_join",
        "tep8_pool_programmatic_fork",
        "tep8_wide_projection_gathers",
        "step_commit_v2",
        "wide_norm_gate_pdl",
        "drafter_kv_glue",
        "drafter_kv_fork_before_head",
        "context_sp_extra_parts",
        "step_window_compact",
        "fmk_context_fmha",
        "context_kv_late_join",
        "hop_cluster_split",
        "drafter_fc_inbox_wide",
        "l061_tile32_pin",
        "router_staged_epilogue",
        "decode_qkv_mxfp8",
        "split_kv_deep_ring",
        "context_prep_rows_vec",
        "context_embedding_replicated",
        "context_sp_router_input",
        "swa_blocks_per_program",
        "wide_qkv_prefetch",
        "wide_capture_fold",
    }
    | SM107_TEP8_ONLY
)
#: The fast paths a composition is built from; a composition lists its levers explicitly.
BASE_PATH_NAMES: FrozenSet[str] = FAST_PATH_NAMES - LEVER_NAMES

_BY_NAME: Dict[str, FastPath] = {path.name: path for path in FAST_PATHS}

#: Every lever and the levers it builds on, in ``FAST_PATHS`` order (``levers.resolve``).
DEPENDS_ON: Mapping[str, Tuple[str, ...]] = {path.name: path.depends_on for path in FAST_PATHS}


def closure(names: Iterable[str]) -> FrozenSet[str]:
    """``names`` and every fast path they build on, transitively.

    Raises:
        KeyError: for a name that is not a fast path.
    """
    seen = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(_BY_NAME[name].depends_on)
    return frozenset(seen)


#: GPUs per node: one compute tray of an NVL72 rack, whose GPUs all share one NVLink domain.
GPUS_PER_TRAY = 4


def tp_group_nodes(mapping: Any) -> int:
    """How many nodes ``mapping``'s TP group spans (the built-in model's own count)."""
    return len({rank // mapping.gpus_per_node for rank in mapping.tp_group})


def describe_topology(mapping: Any) -> str:
    """The topology facts ``Topology.holds`` reads, in words."""
    if mapping is None:
        return "no mapping"
    return (
        f"{mapping.gpus_per_node} GPUs per node, the TP group over "
        f"{tp_group_nodes(mapping)} node(s)"
    )


@dataclasses.dataclass(frozen=True)
class Topology:
    """Where the ranks of a tensor-parallel group sit: ``nodes`` nodes of four GPUs.

    The nodes are compute trays of one NVLink domain: the MNNVL allocations the
    fast paths need (``PRECONDITIONS``) hold only there.
    """

    description: str
    nodes: int

    def holds(self, mapping: Any) -> bool:
        """Whether ``mapping``'s TP group spans exactly ``nodes`` nodes of four GPUs."""
        return (
            getattr(mapping, "gpus_per_node", None) == GPUS_PER_TRAY
            and tp_group_nodes(mapping) == self.nodes
        )


#: The TP group on one node of four GPUs.
ONE_TRAY_OF_4 = Topology("one node of 4 in one NVLink domain", 1)
#: The TP group over two nodes of four GPUs in one NVLink domain.
TWO_TRAYS_OF_4 = Topology("two nodes of 4 in one NVLink domain", 2)


#: What a composition is keyed by: the GPU architecture as an SM number
#: (``sm_of``: 107 for sm_107) and the tensor-parallel size.
CompositionKey = Tuple[int, int]


def sm_of(compute_capability: Optional[Tuple[int, int]]) -> Optional[int]:
    """The SM number of ``compute_capability`` (``(10, 7)`` -> 107), or None without one."""
    if compute_capability is None:
        return None
    major, minor = compute_capability
    return 10 * int(major) + int(minor)


@dataclasses.dataclass(frozen=True)
class TargetPaths:
    """A composition: the fast paths served at one ``CompositionKey``, closed under
    ``depends_on``, the topology they were validated on, and the preconditions of
    the composition beyond its paths' own (``PRECONDITIONS``).

    Raises:
        ValueError: at construction, for a name that is not a fast path or a fast
            path whose dependency is not served.
    """

    served: FrozenSet[str]
    topology: Topology
    preconditions: Tuple["Precondition", ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.served) - FAST_PATH_NAMES)
        if unknown:
            raise ValueError(f"{TAG} not fast paths: {unknown}")
        missing = sorted(closure(self.served) - set(self.served))
        if missing:
            raise ValueError(f"{TAG} a served fast path builds on unserved {missing}")


#: Steps between two log lines of the lever decline counts (``Plan.step``).
DECLINE_REPORT_STEPS = 10_000


@dataclasses.dataclass(frozen=True)
class Plan:
    """The levers a model is built with: a whole composition's table, or none (``NO_PATHS``).

    Closed under ``depends_on`` like a composition (``plan_of`` closes a set). ``levers``
    is the table ``compose`` resolved (None for ``NO_PATHS`` and for a plan built from
    names); ``rank0`` says whether this rank logs the per-server lines.

    Raises:
        ValueError: at construction, for a name that is not a fast path or a fast
            path whose dependency is not in the plan.
    """

    paths: FrozenSet[str] = frozenset()
    levers: Optional[_levers.LeverTable] = dataclasses.field(default=None, compare=False)
    rank0: bool = dataclasses.field(default=True, compare=False)
    #: Lever -> (installed, why not), as the installers reported (``installed``).
    reports: Dict[str, Tuple[bool, str]] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    #: (lever, reason) -> steps declined (``declined``).
    declines: Dict[Tuple[str, str], int] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    #: Steps counted by ``step``, in a one-element list so the frozen plan can count.
    clock: List[int] = dataclasses.field(default_factory=lambda: [0], compare=False, repr=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.paths) - FAST_PATH_NAMES)
        if unknown:
            raise ValueError(f"{TAG} not fast paths: {unknown}")
        missing = sorted(closure(self.paths) - set(self.paths))
        if missing:
            raise ValueError(f"{TAG} a fast path of the plan builds on {missing}, not in it")

    def enabled(self, path: str) -> bool:
        """Whether ``path`` is one of the plan's fast paths.

        Raises:
            KeyError: for a name that is not a fast path.
        """
        if path not in FAST_PATH_NAMES:
            raise KeyError(f"{TAG} not a fast path: {path!r}")
        return path in self.paths

    def param(self, name: str, key: str) -> Any:
        """Lever ``name``'s parameter ``key``: as resolved, else its declared default.

        Raises:
            KeyError: for a name that is not a fast path, or a parameter ``name`` does
                not declare.
        """
        if name not in FAST_PATH_NAMES:
            raise KeyError(f"{TAG} not a fast path: {name!r}")
        if self.levers is not None:
            return self.levers.param(name, key)
        return _levers.default_param(name, key)

    def installed(self, name: str, ok: bool, why: str = "") -> None:
        """An installer's report on lever ``name``: installed, or refused for ``why``.

        The installer passes an outcome agreed over the TP group, so every rank reports
        the same. A refusal of a lever its knob set ``on`` raises; any other refusal is
        a warning, and the installed table (``report``) shows it.

        Raises:
            ValueError: for a refusal of a lever set ``on`` by its knob.
        """
        self.reports[name] = (bool(ok), why)
        if ok:
            return
        decision = self.levers.decisions.get(name) if self.levers is not None else None
        message = f"{TAG} lever {name} is not installed: {why}"
        if decision is not None and decision.source == "knob":
            raise ValueError(f"{message} ({_levers.knob_name(name)}=on)")
        if self.rank0:
            logger.warning(f"{message}; the model runs without it")

    def declined(self, name: str, reason: str, shape: str = "") -> None:
        """A step lever ``name`` left to the path below it, for ``reason``.

        Counted per (lever, reason); the first of each is logged as a warning with the
        step's ``shape``, and ``step`` logs the counts.
        """
        key = (name, reason)
        count = self.declines.get(key, 0) + 1
        self.declines[key] = count
        if count == 1 and self.rank0:
            at = f" ({shape})" if shape else ""
            logger.warning(f"{TAG} lever {name} declined a step{at}: {reason}")

    def step(self) -> None:
        """Count one executor step; every ``DECLINE_REPORT_STEPS`` log the decline counts."""
        self.clock[0] += 1
        if self.clock[0] % DECLINE_REPORT_STEPS == 0 and self.declines and self.rank0:
            counts = {f"{name}: {reason}": count for (name, reason), count in self.declines.items()}
            logger.info(f"{TAG} lever declines after {self.clock[0]} steps: {counts}")

    def report(self) -> str:
        """The installed lever table as one JSON line (``levers.LeverTable.as_json``)."""
        if self.levers is None:
            return "{}"
        return self.levers.as_json({name: ok for name, (ok, _) in self.reports.items()})

    def __bool__(self) -> bool:
        return bool(self.paths)

    def describe(self) -> str:
        """The plan's fast paths in ``FAST_PATHS`` order, for the log line."""
        names = [path.name for path in FAST_PATHS if path.name in self.paths]
        return f"{len(names)} fast paths: {', '.join(names)}" if names else "no fast path"


#: The plan of a deployment with no composition, or one that misses a precondition.
NO_PATHS = Plan()


def plan_of(*names: str) -> Plan:
    """The plan of ``names`` and every fast path they build on."""
    return Plan(closure(names))


def dflash_linear(spec_config: Any) -> bool:
    """Whether ``spec_config`` is DFlash drafting a linear chain of ``DRAFT_LEN`` tokens.

    That is the drafter every composition's drafter, worker and head paths are
    written for: one draft per position (no tree) and a fixed draft length (no
    schedule).
    """
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

    if not isinstance(spec_config, DFlashDecodingConfig):
        return False
    mode = getattr(spec_config, "spec_dec_mode", None)
    return (
        mode is not None
        and mode.is_dflash()
        and getattr(spec_config, "max_draft_len", None) == DRAFT_LEN
        and getattr(spec_config, "max_total_draft_tokens", None) == DRAFT_LEN
        and getattr(spec_config, "draft_len_schedule", None) is None
    )


def describe_speculation(spec_config: Any) -> str:
    """The speculative-decoding facts ``dflash_linear`` reads, in words."""
    if spec_config is None:
        return "off"
    mode = getattr(spec_config, "spec_dec_mode", None)
    return (
        f"{getattr(mode, 'name', type(spec_config).__name__)} "
        f"max_draft_len={getattr(spec_config, 'max_draft_len', None)} "
        f"max_total_draft_tokens={getattr(spec_config, 'max_total_draft_tokens', None)} "
        f"draft_len_schedule={getattr(spec_config, 'draft_len_schedule', None)}"
    )


def describe_quantization(model_config: Any) -> str:
    """The checkpoint quantization ``nvfp4_checkpoint`` reads, in words."""
    quant_config = getattr(model_config, "quant_config", None)
    algo = getattr(quant_config, "quant_algo", None)
    return str(getattr(algo, "name", algo))


def nvfp4_checkpoint(model_config: Any) -> bool:
    """Whether the checkpoint is NVFP4 (``hf_quant_config.json`` ``quant_algo``)."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant_config = getattr(model_config, "quant_config", None)
    return getattr(quant_config, "quant_algo", None) == QuantAlgo.NVFP4


def served_compositions(compositions: Mapping[CompositionKey, TargetPaths]) -> str:
    """The compositions this target serves, in words."""
    keys = (
        ", ".join(
            f"sm{sm} tp {tp} on {compositions[sm, tp].topology.description}"
            for sm, tp in sorted(compositions)
        )
        or "none"
    )
    return (
        f"{keys}, each with DFlash drafting a linear chain of {DRAFT_LEN} tokens "
        "(max_draft_len = max_total_draft_tokens = "
        f"{DRAFT_LEN}, no draft_len_schedule) and an NVFP4 checkpoint"
    )


def composition(
    model_config: Any,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
) -> Optional[TargetPaths]:
    """The composition this deployment is served with, or None. Pure: reads its arguments only.

    Every fact read here is structural: the GPU architecture
    (``compute_capability``, read once from the device by the caller, as
    ``route`` gets it), the tensor-parallel size, where its ranks sit
    (``Topology``), the speculative-decoding mode and the checkpoint layout. An
    architecture without an entry has no composition.
    """
    mapping = getattr(model_config, "mapping", None)
    if (
        mapping is None
        or not dflash_linear(getattr(model_config, "spec_config", None))
        or not nvfp4_checkpoint(model_config)
    ):
        return None
    served = compositions.get((sm_of(compute_capability), int(mapping.tp_size)))
    if served is None or not served.topology.holds(mapping):
        return None
    return served


def unserved_reason(
    model_config: Any,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
) -> str:
    """Why ``composition`` found none for this deployment, naming every fact it read."""
    mapping = getattr(model_config, "mapping", None)
    tp = getattr(mapping, "tp_size", None)
    spec = describe_speculation(getattr(model_config, "spec_config", None))
    arch = "no CUDA device" if compute_capability is None else f"sm{sm_of(compute_capability)}"
    return (
        f"{arch}, tp {tp}, {describe_topology(mapping)}, speculative decoding {spec}, checkpoint "
        f"quantization {describe_quantization(model_config)}; the fast paths are served at "
        + served_compositions(compositions)
    )


def require_served(
    model_config: Any,
    mode: ModelingV2Mode,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
) -> None:
    """Raise under ``require`` when the deployment has no composition; before any construction.

    Raises:
        ValueError: under ``modeling_v2="require"`` when ``composition`` is None.
    """
    if (
        mode is ModelingV2Mode.REQUIRE
        and composition(model_config, compositions, compute_capability) is None
    ):
        raise ValueError(
            f"modeling_v2='require': {TAG} serves no fast path for this deployment: "
            + unserved_reason(model_config, compositions, compute_capability)
        )


def _low_m_gemm_active(text_model, model_config) -> bool:
    from tensorrt_llm._torch.modules import low_m_gemm

    return bool(low_m_gemm.LOW_M_GEMM_ACTIVE)


def _pdl_enabled(text_model, model_config) -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return bool(get_env_enable_pdl())


def _builtin_switch(name: str, on: bool) -> Callable[[Any, Any], bool]:
    def holds(text_model, model_config) -> bool:
        from tensorrt_llm._torch.models import modeling_musespark_open as builtin

        return bool(getattr(builtin, name)()) == on

    return holds


def _mnnvl_o_proj_all_reduce(text_model, model_config) -> bool:
    from tensorrt_llm.functional import AllReduceStrategy

    for layer in text_model.layers:
        all_reduce = getattr(getattr(layer.self_attn, "o_proj", None), "all_reduce", None)
        if (
            all_reduce is None
            or getattr(all_reduce, "strategy", None) != AllReduceStrategy.MNNVL
            or getattr(all_reduce, "mnnvl_allreduce", None) is None
        ):
            return False
    return True


def _mnnvl_fp32_all_reduce(text_model, model_config) -> bool:
    """Whether an explicitly requested FP32 MNNVL all-reduce is built on this TP group.

    The check ``AllReduce(strategy=MNNVL)`` makes before it builds one; allocates nothing.
    """
    import torch

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    mapping = getattr(model_config, "mapping", None)
    return mapping is not None and bool(
        MNNVLAllReduce.is_mnnvl(mapping, torch.float32, explicitly_requested=True)
    )


def _mnnvl_bf16_all_reduce(text_model, model_config) -> bool:
    """Whether an explicitly requested BF16 MNNVL all-reduce is built on this TP group.

    The drafter hops' row projections need one; on one tray the hops' install
    requests it by name. Allocates nothing.
    """
    import torch

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    mapping = getattr(model_config, "mapping", None)
    return mapping is not None and bool(
        MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True)
    )


def _fabric_finalize(text_model, model_config) -> bool:
    workspace = getattr(text_model, "_finalize_ar_workspace", None)
    return workspace is not None and all(
        getattr(layer.feed_forward, "_finalize_ar_workspace", None) is workspace
        for layer in text_model.layers
    )


def _sharded_projections(text_model, model_config) -> bool:
    return all(
        getattr(layer.feed_forward, "_pool_gather_workspace", None) is not None
        and getattr(layer.feed_forward, "_unpool_gather_workspace", None) is not None
        for layer in text_model.layers
    )


def _builtin_prep_kernels(text_model, model_config) -> bool:
    from .attention import _builtin_prep_kernel_mismatch

    return _builtin_prep_kernel_mismatch() is None


def _default_qkv_weight_layout(text_model, model_config) -> bool:
    """The built-in attention keeps the checkpoint's QKV weight layout on every layer."""
    return all(
        not getattr(layer.self_attn, "_qkv_weight_layout", False)
        and getattr(layer.self_attn, "_qkv_layout_inverse", None) is None
        for layer in text_model.layers
    )


def _no_prefill_sequence_parallel(text_model, model_config) -> bool:
    return not getattr(text_model, "_prefill_sp_enabled", False)


def _cute_dsl_kernels(*checks: Tuple[str, str]) -> Callable[[Any, Any], bool]:
    """Every ``(module, name)``: the op module's CuTe-DSL kernels can be built here.

    ``name`` is a module-level flag (``HAS_CUTE_DSL``) or an ``available()`` function.
    """

    def holds(text_model, model_config) -> bool:
        for module_name, name in checks:
            module = importlib.import_module(f"{__package__}.{module_name}")
            value = getattr(module, name)
            if not (value() if callable(value) else value):
                return False
        return True

    return holds


def _head_sampler_toolchain(text_model, model_config) -> bool:
    """``ninja`` and ``nvcc``, which torch's extension loader builds the head sampler with."""
    from torch.utils import cpp_extension

    home = cpp_extension.CUDA_HOME
    return bool(
        cpp_extension.is_ninja_available()
        and home is not None
        and (pathlib.Path(home) / "bin" / "nvcc").is_file()
    )


@dataclasses.dataclass(frozen=True)
class Precondition:
    """A public setting the fast paths are written against, and how a built model shows it.

    ``holds(text_model, model_config)`` reads the built text model and its
    config, or the public helper the built-in model reads the setting through;
    never the environment. ``needed_by`` names the fast paths written against
    it; empty means every composition.
    """

    requirement: str
    holds: Callable[[Any, Any], bool]
    needed_by: Tuple[str, ...] = ()

    def needed(self, paths: FrozenSet[str]) -> bool:
        """Whether a model built with ``paths`` needs this precondition."""
        return not self.needed_by or bool(set(self.needed_by) & set(paths))


_PREP = ("preattn_prep", "full_attention_prep_append")
_MOE_PREP = ("moe_prologue", "projection_gathers", "pool_into_route")

#: Every public precondition of a fast path, in the order they are reported.
PRECONDITIONS: Tuple[Precondition, ...] = (
    Precondition("programmatic dependent launch (TRTLLM_ENABLE_PDL=1)", _pdl_enabled),
    Precondition(
        "the low-M GEMM dispatcher (TRTLLM_LOW_M_GEMM_BACKEND=auto)",
        _low_m_gemm_active,
        ("dense_gemms", "lm_head_inbox", "drafter_pdl_chain", "distributed_head_sampler"),
    ),
    Precondition(
        "the MNNVL one-shot o_proj all-reduce on every layer (MUSESPARK_OPEN_ALLREDUCE=MNNVL "
        "on an MNNVL-capable TP group)",
        _mnnvl_o_proj_all_reduce,
        ("post_attention_hop",),
    ),
    Precondition(
        "an MNNVL FP32 all-reduce on the TP group (the draft head's key exchange)",
        _mnnvl_fp32_all_reduce,
        ("draft_head",),
    ),
    Precondition(
        "an MNNVL BF16 all-reduce on the TP group (the drafter hops' row projections)",
        _mnnvl_bf16_all_reduce,
        ("drafter_hops",),
    ),
    Precondition(
        "the fused finalize all-reduce workspace (MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1 at "
        "TP8; built in on one tray at TP4)",
        _fabric_finalize,
        ("finalize_transport", "finalize_two_shot"),
    ),
    Precondition(
        "the sharded pool/unpool projections (MUSESPARK_OPEN_SHARDED_PROJECTIONS=1)",
        _sharded_projections,
        ("projection_gathers", "dense_gemm_prefetch"),
    ),
    Precondition(
        "the fused residual join (MUSESPARK_OPEN_FUSED_RESIDUAL on, its default)",
        _builtin_switch("_fused_residual_enabled", True),
        ("post_attention_hop",),
    ),
    Precondition(
        "the routing memo (MUSESPARK_OPEN_ROUTING_MEMO on, its default)",
        _builtin_switch("_routing_memo_enabled", True),
        _MOE_PREP,
    ),
    Precondition(
        "the fused pre-expert quantization (MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT on, its default)",
        _builtin_switch("_fused_pre_expert_quant_enabled", True),
        _MOE_PREP,
    ),
    Precondition(
        "the unfused pre-attention prep (MUSESPARK_OPEN_FUSED_PREATTN off, its default)",
        _builtin_switch("_fused_preattention_enabled", False),
        _PREP,
    ),
    Precondition(
        "the BF16 pre-QKV norm input (MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL off, its default)",
        _builtin_switch("_preqkv_norm_fp32_residual_enabled", False),
        ("post_attention_hop",),
    ),
    Precondition(
        "the built-in prep on flashinfer's CuTe-DSL RMSNorm and cos/sin-cache RoPE "
        "(flashinfer available, FLASHINFER_USE_CUDA_NORM unset)",
        _builtin_prep_kernels,
        _PREP,
    ),
    Precondition(
        "the checkpoint's QKV weight layout (MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT off, its default)",
        _default_qkv_weight_layout,
        _PREP,
    ),
    Precondition(
        "the CuTe DSL for the post-attention hop",
        _cute_dsl_kernels(("ops.allreduce.post_attn_hop", "HAS_CUTE_DSL")),
        ("post_attention_hop",),
    ),
    Precondition(
        "the CuTe DSL for the multi-request post-attention hop",
        _cute_dsl_kernels(("ops.allreduce.post_attn_hop_wide", "HAS_CUTE_DSL")),
        ("multi_request_post_attention_hop",),
    ),
    Precondition(
        "the CuTe DSL for the drafter hops",
        _cute_dsl_kernels(("ops.allreduce.drafter_hop", "HAS_CUTE_DSL")),
        ("drafter_hops",),
    ),
    Precondition(
        "the CuTe DSL and the built-in finalize module for the finalize transport",
        _cute_dsl_kernels(("ops.moe.finalize_allreduce_transport", "HAS_CUTE_DSL")),
        ("finalize_transport",),
    ),
    Precondition(
        "the CuTe DSL and the built-in finalize module for the two-shot finalize",
        _cute_dsl_kernels(("ops.moe.finalize_two_shot", "HAS_CUTE_DSL")),
        ("finalize_two_shot",),
    ),
    Precondition(
        "the CuTe DSL, the built-in cluster boundary and flashinfer's RMSNormFP4QuantKernel "
        "for the projection gathers",
        _cute_dsl_kernels(
            ("ops.comm.unpool_gather_boundary", "available"),
            ("ops.comm.pool_gather_fp4quant", "available"),
        ),
        ("projection_gathers",),
    ),
    Precondition(
        "the CuTe DSL for the MoE prologue's routed gather",
        _cute_dsl_kernels(("ops.comm.pool_gather_route", "available")),
        ("moe_prologue",),
    ),
    Precondition(
        "the CuTe DSL for the pool transport into the route",
        _cute_dsl_kernels(("ops.comm.pool_direct_route", "available")),
        ("pool_into_route",),
    ),
    Precondition(
        "the CuTe DSL for the unpool transport into the boundary",
        _cute_dsl_kernels(("ops.comm.unpool_direct_boundary", "available")),
        ("unpool_into_boundary",),
    ),
    Precondition(
        "ninja and nvcc for the head sampler's build (torch.utils.cpp_extension)",
        _head_sampler_toolchain,
        ("split_sampler", "draft_head_sampled"),
    ),
)

#: The built-in prefill sequence parallelism, a single-tray TP4 switch, left off.
NO_PREFILL_SEQUENCE_PARALLEL = Precondition(
    "no prefill sequence parallelism (MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL unset)",
    _no_prefill_sequence_parallel,
)

#: The built-in model's fused kernels (its cluster residual boundary, fused finalize
#: all-reduce and sharded projections), opted in on sm107, which the compositions build on
#: and whose arithmetic their bitwise paths reproduce.
SM107_FUSED_KERNELS = Precondition(
    "the built-in fused kernels on sm107 (MUSESPARK_OPEN_SM107_FUSED_KERNELS=1)",
    _builtin_switch("_sm107_fused_kernels_enabled", True),
)


def preconditions(paths: FrozenSet[str], served: TargetPaths) -> Tuple[Precondition, ...]:
    """The preconditions of a model built with ``paths`` from composition ``served``."""
    return tuple(p for p in PRECONDITIONS if p.needed(paths)) + tuple(served.preconditions)


def missing_preconditions(
    text_model: Any, model_config: Any, required: Iterable[Precondition]
) -> List[str]:
    """The requirements of ``required`` this rank's built model does not meet. Changes nothing."""
    return [p.requirement for p in required if not p.holds(text_model, model_config)]


#: The fast paths that compact a windowed drafter context arena on the steps the windowed
#: commit does not take (the built-in prepare of a mixed step's generation rows). The paths
#: that write the arena by token column -- the context-K/V fork's side chain and commit, and
#: the verify fold's written-out context store -- address each slot's window from its base
#: (``_ctx_base``) over a windowed arena, and their commit compacts in its own kernel
#: (``ops/spec/context_kv_commit_windowed.py``), so they compose over either arena. (The
#: drafter block decode reads only the physical length the built-in prepare hands it, so it
#: composes over either arena too.) These paths leave a plan over a full-length arena, which
#: never compacts.
WINDOWED_STEP_PATHS: Tuple[str, ...] = ("step_window_compact",)


def dependents(names: Iterable[str]) -> FrozenSet[str]:
    """``names`` and every fast path that builds on one of them, transitively."""
    names = frozenset(names)
    return frozenset(path for path in FAST_PATH_NAMES if closure((path,)) & names)


#: The levers whose store is the dense full-length arena's or the KV cache manager's pages,
#: and so not a windowed arena's: the context-K/V side chain with its fused store, its early fc
#: fork, and the folded commit that also writes the drafter's preamble.
UNWINDOWED_ARENA_LEVERS: Tuple[str, ...] = (
    "drafter_kv_side_chain",
    "drafter_fc_early_fork",
    "drafter_preamble_from_commit",
)


@dataclasses.dataclass(frozen=True)
class StructuralRules:
    """What a deployment's drafter arena, architecture and facts exclude or require.

    ``exclusions`` maps each served lever a rule leaves out to the reason; ``requires``
    adds the levers a served lever needs on this deployment (``levers.resolve``).
    """

    exclusions: Mapping[str, str]
    requires: Mapping[str, Tuple[str, ...]]


def structural_rules(served: FrozenSet[str], deployment: "_levers.Deployment") -> StructuralRules:
    """The structural rules of ``deployment`` for the levers of ``served``. Pure.

    * An arena that is not windowed leaves the per-step compaction (``WINDOWED_STEP_PATHS``)
      out.
    * ``UNWINDOWED_ARENA_LEVERS`` need a dense full-length arena or pages.
    * ``windowed_commit_parallel`` runs inside the windowed commit, so it needs a windowed
      arena.
    * A paged arena leaves out ``drafter_kv_glue``, whose store writes a worker-owned arena.
    * ``TRTLLM_KV_DEFER_CLOSE=1`` leaves out ``step_loop_on_execution_stream``, which is
      not validated with the deferred close.
    * A drafter that declares a logit transform, or one whose transform is not known,
      leaves out ``draft_head_tail``: its key selection and walk rank the untransformed
      logits, so they would not reproduce the transformed selector (``draft_head``'s
      canonical keys and transformed unary values).
    * A speculative config that does not ask for rejection sampling, or whose request
      forced acceptance left inert, or one not known, leaves out ``rejection_sampler``.
    """
    arena = deployment.arena
    exclusions: Dict[str, str] = {}
    requires: Dict[str, Tuple[str, ...]] = {}
    if arena is not _levers.Arena.WINDOWED:
        for name in WINDOWED_STEP_PATHS:
            if name in served:
                exclusions[name] = f"the drafter context arena is {arena.value}, not windowed"
    if arena is _levers.Arena.WINDOWED:
        for name in UNWINDOWED_ARENA_LEVERS:
            if name in served:
                exclusions[name] = (
                    "it needs a dense full-length drafter context arena or pages, not a "
                    f"{arena.value} one"
                )
    if "windowed_commit_parallel" in served and arena is not _levers.Arena.WINDOWED:
        exclusions["windowed_commit_parallel"] = (
            "it needs a windowed drafter context arena, whose commit compacts in its own "
            f"kernel; this is a {arena.value} arena"
        )
    if arena is _levers.Arena.PAGED and "drafter_kv_glue" in served:
        exclusions["drafter_kv_glue"] = (
            "a paged drafter context has no worker-owned arena for its store"
        )
    if deployment.facts.kv_defer_close and "step_loop_on_execution_stream" in served:
        exclusions["step_loop_on_execution_stream"] = (
            f"{_levers.KV_DEFER_CLOSE_ENV}=1: the deferred close is not validated with it"
        )
    if "draft_head_tail" in served and deployment.draft_logit_transform is not False:
        exclusions["draft_head_tail"] = (
            "the drafter declares a logit transform, which the tail's key selection and walk "
            "do not apply"
            if deployment.draft_logit_transform
            else "the drafter's logit transform is not known"
        )
    if "rejection_sampler" in served and deployment.rejection_sampling is not True:
        exclusions["rejection_sampler"] = (
            "the speculative config does not ask for rejection sampling, or forced acceptance "
            "leaves it inert"
            if deployment.rejection_sampling is False
            else "whether the speculative config asks for rejection sampling is not known"
        )
    return StructuralRules(exclusions, requires)


def rejection_sampling_of(model_config) -> Optional[bool]:
    """Whether ``model_config``'s speculative config asks for rejection sampling, or None
    without one.

    Under forced acceptance (``rs_common.forced_mode``) it is False whatever the config says:
    the fused path never runs under a forced count. Where the forced-acceptance policy
    applies (``rs_common.policy_applies``), ``routing.route`` has already switched the
    engine's flag off; elsewhere the engine's own rejection sampling runs.
    """
    spec = getattr(model_config, "spec_config", None)
    if spec is None:
        return None
    if not getattr(spec, "use_rejection_sampling", False):
        return False
    from .kernels.rs_sampler import rs_common as _rs_common

    return _rs_common.forced_mode() != "inert"


@dataclasses.dataclass(frozen=True)
class _EngineFacts:
    """The engine facts ``compose`` falls back to without ``_router_index.DeploymentFacts``."""

    max_batch_size: Optional[int]
    max_num_tokens: Optional[int] = None
    disaggregated: Optional[bool] = None


#: The prefix of the drafter arena item of the TP agreement.
_ARENA_TAG = "drafter context arena: "
#: The prefix of a rank's lever error carried through the TP agreement.
_LEVER_ERROR = "lever table: "


def compose(
    text_model: Any,
    model_config: Any,
    mode: ModelingV2Mode,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    spec_worker: Any = None,
    max_batch_size: Optional[int] = None,
    *,
    draft_model: Any = None,
    facts: Any = None,
    snapshot: Optional[Mapping[str, str]] = None,
) -> Plan:
    """The plan this model is built with: its composition's lever table, or ``NO_PATHS``.

    Call after the built-in constructor and before anything is installed. A
    deployment without a composition (``composition``, structural) gets ``NO_PATHS``
    under ``auto`` and raises under ``require``; its lever environment is still
    validated (``levers.no_composition_notes``). Otherwise:

    1. the drafter context arena (``levers.arena_of``, the worker's own predicate over
       ``spec_worker`` and ``draft_model``) and the ``levers.Deployment``
       (``levers.deployment_of``; ``facts`` are the engine's
       ``_router_index.DeploymentFacts``, else ``max_batch_size`` alone);
    2. the lever table (``levers.resolve`` under ``structural_rules``), whose levers on
       are the plan;
    3. the preconditions of those levers, the arena, the lever environment and any
       rank's lever error, agreed over the TP group in one collective: every rank
       returns the same plan or raises the same error.

    The lever table is logged once, on TP rank 0. ``snapshot`` stands in for the
    process's environment snapshot (``levers.environment``).

    Raises:
        ValueError: under ``modeling_v2="require"`` when the deployment has no
            composition; under any mode for a lever error (``levers.resolve``), a
            precondition missing on some rank, or ranks that disagree on the drafter
            arena or the lever environment (on every rank alike).
    """
    snapshot = _levers.environment() if snapshot is None else snapshot
    served = composition(model_config, compositions, compute_capability)
    mapping = getattr(model_config, "mapping", None)
    if served is None:
        require_served(model_config, mode, compositions, compute_capability)
        notes = _levers.no_composition_notes(snapshot, DEPENDS_ON)
        reason = unserved_reason(model_config, compositions, compute_capability)
        logger.info(
            f"{TAG} no fast path for this deployment ({reason}); running the built-in forward"
        )
        for note in notes:
            logger.info(f"{TAG} {note}")
        return NO_PATHS
    from tensorrt_llm._torch.speculative.dflash import _PAGED_ATTENTION_BACKENDS

    arena = _levers.arena_of(spec_worker, draft_model, _PAGED_ATTENTION_BACKENDS)
    engine = facts if facts is not None else _EngineFacts(max_batch_size)
    table: Optional[_levers.LeverTable] = None
    local: List[str] = []
    try:
        # Every grammar error first, the role hint's included, before the role is derived.
        _levers.parse_knobs(snapshot, DEPENDS_ON)
        deployment = _levers.deployment_of(
            model_config,
            compute_capability,
            arena,
            engine,
            snapshot,
            draft_logit_transform=_levers.draft_logit_transform_of(draft_model),
            rejection_sampling=rejection_sampling_of(model_config),
        )
        rules = structural_rules(served.served, deployment)
        table = _levers.resolve(
            deployment, served.served, rules.exclusions, snapshot, DEPENDS_ON, rules.requires
        )
    except ValueError as error:
        local.append(f"{_LEVER_ERROR}{error}")
    if table is not None:
        local += missing_preconditions(
            text_model, model_config, preconditions(table.paths(), served)
        )
    tags = [f"{_ARENA_TAG}{arena.value}", *_levers.tagged(snapshot)]
    agreed = agreed_union(mapping, local + tags)
    own = set(tags)
    tagged = (_ARENA_TAG, _levers.AGREEMENT_TAG)
    split = [item for item in agreed if item.startswith(tagged) and item not in own]
    errors = [item[len(_LEVER_ERROR) :] for item in agreed if item.startswith(_LEVER_ERROR)]
    missing = [item for item in agreed if not item.startswith(tagged + (_LEVER_ERROR,))]
    composed = (
        f"{TAG} sm{sm_of(compute_capability)} tep{mapping.tp_size} on {served.topology.description}"
    )
    if split:
        raise ValueError(
            f"{composed}: the TP ranks disagree on the drafter context arena or the lever "
            "environment: " + "; ".join(split)
        )
    if errors:
        raise ValueError(f"{composed}: " + "; ".join(errors))
    if missing:
        raise ValueError(f"{composed}: the fast paths need " + "; ".join(missing))
    rank0 = int(getattr(mapping, "tp_rank", 0) or 0) == 0
    plan = Plan(table.paths(), levers=table, rank0=rank0)
    logger.info(f"{composed} with a {arena.value} drafter context arena: {plan.describe()}")
    if rank0:
        for line in table.describe():
            logger.info(f"{TAG} {line}")
        for warning in table.warnings:
            logger.warning(f"{TAG} lever warning: {warning}")
    return plan


def table_for(
    compositions: Mapping[CompositionKey, TargetPaths],
    sm: int,
    tp: int,
    max_batch_size: Optional[int],
    role: str,
    arena: str,
    knobs: Optional[Mapping[str, str]] = None,
    facts: Optional["_levers.Facts"] = None,
    max_num_tokens: Optional[int] = None,
    role_source: Optional[str] = None,
    draft_logit_transform: Optional[bool] = None,
    rejection_sampling: Optional[bool] = None,
) -> "_levers.LeverTable":
    """The lever table ``compose`` resolves for a deployment, without a model or a device.

    For tests and for reviewing a deployment's levers. ``role`` and ``arena`` are ``levers.Role`` /
    ``levers.Arena`` values; ``knobs`` the lever environment (names under
    ``levers.PREFIX``); ``facts`` the shared-path facts; ``role_source`` defaults to what
    a served deployment of that role reports (agg servers derive their role, the others
    take the hint); ``draft_logit_transform`` whether the drafter declares a logit transform
    (None: not known); ``rejection_sampling`` whether the speculative config asks for
    rejection sampling once forced acceptance has applied (None: not known).

    Raises:
        KeyError: for a key no composition serves.
        ValueError: as ``levers.resolve``.
    """
    served = compositions[(sm, tp)].served
    role = _levers.Role(role)
    if role_source is None:
        role_source = {
            _levers.Role.AGG: "no cache transceiver",
            _levers.Role.UNKNOWN: _levers.DISAGGREGATED_UNHINTED,
        }.get(role, "hint")
    deployment = _levers.Deployment(
        sm=sm,
        tp=tp,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        role=role,
        role_source=role_source,
        arena=_levers.Arena(arena),
        facts=facts or _levers.Facts(),
        draft_logit_transform=draft_logit_transform,
        rejection_sampling=rejection_sampling,
    )
    rules = structural_rules(served, deployment)
    return _levers.resolve(
        deployment, served, rules.exclusions, dict(knobs or {}), DEPENDS_ON, rules.requires
    )


# --------------------------------------------------------------------------
# Install refusals agreed over the TP group
# --------------------------------------------------------------------------
_T = TypeVar("_T")


def group_size(mapping) -> int:
    """The TP group size of ``mapping`` (1 for None or an object without ``tp_size``)."""
    return int(getattr(mapping, "tp_size", 1) or 1) if mapping is not None else 1


def agreed_refusal(mapping, local: Optional[str], facts: Optional[Any] = None) -> Optional[str]:
    """``local`` (this rank's reason to refuse, or None) as every rank of ``mapping``'s TP group agrees.

    Collective over the TP group when it has more than one rank: every rank of the
    group must call it at the same point, whatever its own reason. A group of one
    returns ``local``. Returns None when no rank refuses; the reason itself when
    every rank gives the same one; otherwise the distinct reasons, followed by the
    TP ranks that gave one. Changes nothing.

    ``facts`` (optional, a small picklable value with ``==``): what every rank must
    hold alike, gathered in the same collective. Ranks whose ``facts`` differ are a
    refusal on every rank, naming each rank's value.
    """
    if group_size(mapping) <= 1:
        return local
    if facts is None:
        everyone = Distributed.get(mapping).tp_allgather(local, small_payload=True)
    else:
        gathered = Distributed.get(mapping).tp_allgather((local, facts), small_payload=True)
        everyone = [why for why, _ in gathered]
        values = [value for _, value in gathered]
        if any(value != values[0] for value in values):
            by_rank = ", ".join(f"rank {rank}: {value!r}" for rank, value in enumerate(values))
            differ = f"the TP ranks disagree on {by_rank}"
            reasons = sorted({why for why in everyone if why is not None})
            return "; ".join(reasons + [differ])
    refused = [(rank, why) for rank, why in enumerate(everyone) if why is not None]
    if not refused:
        return None
    reasons = sorted({why for _, why in refused})
    if len(refused) == len(everyone) and len(reasons) == 1:
        return reasons[0]
    ranks = [rank for rank, _ in refused]
    return f"{'; '.join(reasons)} (on TP ranks {ranks} of {len(everyone)})"


def agreed_install(
    mapping,
    install: Callable[[], _T],
    refusals: Tuple[Type[Exception], ...] = (ValueError,),
) -> _T:
    """``install()``'s result, or its refusal raised on every rank of ``mapping``'s TP group.

    For an installer that runs no collective itself: a ``refusals`` exception it
    raises on this rank is held, every rank's outcome is agreed (``agreed_refusal``,
    collective when the group has more than one rank, so every rank calls this at
    the same point), and every rank raises the agreed reason as a ``ValueError``
    when any rank refused. An exception outside ``refusals`` propagates unagreed.
    """
    result = None
    held: Optional[Exception] = None
    try:
        result = install()
    except refusals as error:
        held = error
    reason = agreed_refusal(mapping, None if held is None else (str(held) or type(held).__name__))
    if reason is not None:
        raise ValueError(reason) from held
    return result


def agreed_union(mapping, local: Sequence[str]) -> List[str]:
    """Every item some rank of ``mapping``'s TP group lists, the same list on every rank.

    Collective over the TP group when it has more than one rank: every rank of the
    group must call it at the same point, whatever it lists. An item only some
    ranks list names those ranks. Changes nothing.
    """
    if group_size(mapping) <= 1:
        return list(dict.fromkeys(local))
    everyone = Distributed.get(mapping).tp_allgather(list(local), small_payload=True)
    ranks: dict = {}
    for rank, items in enumerate(everyone):
        for item in items:
            ranks.setdefault(item, []).append(rank)
    return [
        item if len(seen) == len(everyone) else f"{item} (on TP ranks {seen} of {len(everyone)})"
        for item, seen in ranks.items()
    ]
