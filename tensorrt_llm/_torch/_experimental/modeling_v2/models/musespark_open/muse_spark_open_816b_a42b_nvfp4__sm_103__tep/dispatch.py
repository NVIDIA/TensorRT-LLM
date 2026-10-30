# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which fast paths a model of this target is built with, decided once at construction.

``FAST_PATHS`` names every fast path of the target and the fast paths each one
builds on (``depends_on``). A composition (``TargetPaths``) is the set of fast
paths served on one GPU architecture at one tensor-parallel size (its
``CompositionKey``), closed under ``depends_on``, with the GPU topology it was
validated on; it applies when the drafter is DFlash drafting a linear chain of
``DRAFT_LEN`` tokens on an NVFP4 checkpoint. Those facts are structural: a
deployment that misses one has no composition, and runs the built-in forward
under ``modeling_v2="auto"`` (``NO_PATHS``) or raises under ``require``. Nothing
reads the environment to pick or trim a composition.

``compose`` then checks the public preconditions the planned fast paths are
written against (``PRECONDITIONS``, and the composition's own), once, agreed
over the TP group. A missing one, on any rank, raises ``ValueError`` naming each
one, under ``auto`` as under ``require``: a composition is never served in part
and never dropped because of a setting. The model's installers ask the
resulting ``Plan`` per fast path (``Plan.enabled``).
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

#: The draft tokens per step of the one speculative-decoding mode the compositions serve.
DRAFT_LEN = 7

#: The log tag of this family.
TAG = "[musespark_open]"

#: Every fast path launches its kernels with programmatic dependent launch:
#: ``TRTLLM_ENABLE_PDL`` on is a precondition of every composition (``PRECONDITIONS``).
LAUNCH_PDL = True


@dataclasses.dataclass(frozen=True)
class FastPath:
    """One fast path: what it replaces, the fast paths it builds on, and when it installs.

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
        "decode_kv_splits",
        "multi-request decode FMHA over an over-subscribed KV split, 72..512 rows",
        "construct",
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
        "wide_projection_gathers",
        "sharded pool/unpool with fused gathers and the next QKV on split-K, 9..32-row decode steps",
        "construct",
        ("projection_gathers", "dense_gemms"),
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
        "drafter_window_compact",
        "windowed drafter arena compaction in place, one read and write per window",
        "draft",
    ),
    FastPath(
        "verify_fold",
        "step bookkeeping folded into the acceptance kernel",
        "draft",
        ("fused_accept",),
    ),
    FastPath(
        "path_walk_tail",
        "step bookkeeping folded into the draft-head walk",
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
)

#: The name of every fast path.
FAST_PATH_NAMES: FrozenSet[str] = frozenset(path.name for path in FAST_PATHS)

_BY_NAME: Dict[str, FastPath] = {path.name: path for path in FAST_PATHS}


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
#: (``sm_of``: 103 for sm_103) and the tensor-parallel size.
CompositionKey = Tuple[int, int]


def sm_of(compute_capability: Optional[Tuple[int, int]]) -> Optional[int]:
    """The SM number of ``compute_capability`` (``(10, 3)`` -> 103), or None without one."""
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


@dataclasses.dataclass(frozen=True)
class Plan:
    """The fast paths a model is built with: a whole composition, or none (``NO_PATHS``).

    Closed under ``depends_on`` like a composition (``plan_of`` closes a set).

    Raises:
        ValueError: at construction, for a name that is not a fast path or a fast
            path whose dependency is not in the plan.
    """

    paths: FrozenSet[str] = frozenset()

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
        ("dense_gemms", "lm_head_inbox", "drafter_pdl_chain"),
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
        ("split_sampler",),
    ),
)

#: The built-in prefill sequence parallelism, a single-tray TP4 switch, left off.
NO_PREFILL_SEQUENCE_PARALLEL = Precondition(
    "no prefill sequence parallelism (MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL unset)",
    _no_prefill_sequence_parallel,
)

#: On sm107, the built-in model's SM103-measured fused kernels (its cluster residual
#: boundary, fused finalize all-reduce and sharded projections), which the sm107
#: compositions build on and whose arithmetic their bitwise paths reproduce.
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


#: The fast paths that write the drafter's context-K/V arena by token column: the
#: context-K/V fork's side chain and commit, and the verify fold's written-out
#: context store. Over a windowed arena they address each slot's window from its
#: base (``_ctx_base``). Where the composition serves this target's compaction
#: (``WINDOWED_CONTEXT_ARENA_PATHS``) they end with it; on an architecture of
#: ``WINDOWED_COMMIT_SMS`` their commit compacts in its own kernel
#: (``ops/spec/context_kv_commit_windowed.py``); elsewhere a windowed arena leaves
#: them out. (The drafter block decode reads only the physical length the built-in
#: prepare hands it, so it composes over either arena.)
FULL_CONTEXT_ARENA_PATHS: Tuple[str, ...] = (
    "context_kv_fork",
    "verify_fold",
)
#: The fast paths that act only on a windowed drafter context arena: the per-step window
#: compaction. None of them is composed over a full-length arena, which never compacts.
WINDOWED_CONTEXT_ARENA_PATHS: Tuple[str, ...] = ("drafter_window_compact",)
#: The architectures (SM numbers) whose compositions serve ``FULL_CONTEXT_ARENA_PATHS``
#: over a windowed arena without this target's compaction, on the windowed commit that
#: compacts in its own kernel.
WINDOWED_COMMIT_SMS: FrozenSet[int] = frozenset({107})


def dependents(names: Iterable[str]) -> FrozenSet[str]:
    """``names`` and every fast path that builds on one of them, transitively."""
    names = frozenset(names)
    return frozenset(path for path in FAST_PATH_NAMES if closure((path,)) & names)


def windowed_context_arena(spec_worker: Any) -> bool:
    """Whether the DFlash worker keeps only a window of each slot's context K/V.

    Read off the worker the built-in model built (``DFlashWorker.windowed_ctx_arena``).
    """
    return bool(getattr(spec_worker, "windowed_ctx_arena", False))


_WINDOWED_ARENA = "a windowed drafter context arena"


@dataclasses.dataclass(frozen=True)
class CapacityGate:
    """A served fast path left out above a request capacity, because of what it costs.

    ``path`` (and every fast path built on it) is in the plan only when the
    engine's ``max_batch_size`` is at most ``most_requests``; an unknown
    ``max_batch_size`` leaves it out. ``cost`` says in words what the path takes
    from the deployment.
    """

    path: str
    most_requests: int
    cost: str


#: The fast paths a plan carries only at a few concurrent requests.
#: ``half_window_experts`` stacks a second half expert window on every TP4 rank,
#: which takes 45 GiB per rank from the KV cache; it is measured ahead of the
#: owner-only window through 8 concurrent requests, and beyond that the
#: prefix cache's capacity is worth more.
CAPACITY_GATES: Tuple[CapacityGate, ...] = (
    CapacityGate(
        "half_window_experts",
        8,
        "a second half expert window per TP4 rank, 45 GiB of KV cache",
    ),
)


def capacity_gated(
    paths: FrozenSet[str], max_batch_size: Optional[int]
) -> Tuple[FrozenSet[str], List[CapacityGate]]:
    """``paths`` without the gates closed at ``max_batch_size``, and the gates that closed.

    A gate closes when its path is in ``paths`` and ``max_batch_size`` is None or
    above its ``most_requests``; the path and its dependents leave the set, which
    stays closed under ``depends_on``. Changes nothing.
    """
    closed = [
        gate
        for gate in CAPACITY_GATES
        if gate.path in paths and (max_batch_size is None or max_batch_size > gate.most_requests)
    ]
    if not closed:
        return paths, closed
    return paths - dependents(gate.path for gate in closed), closed


@dataclasses.dataclass(frozen=True)
class Yield:
    """A served fast path left out of a plan that carries another one doing its job.

    ``path`` (and every fast path built on it) leaves the plan when ``to`` is in it,
    after the capacity gates. ``why`` says in words what the two share.
    """

    path: str
    to: str
    why: str


#: The fast paths a plan leaves to another one it carries. ``hot_replicas`` and
#: ``half_window_experts`` both stack extra NVFP4 experts on every TP4 rank and place
#: each step's experts over them; the half window is measured ahead through 8
#: concurrent requests (its capacity gate), so a plan carrying it leaves the hot copies
#: out and the two never both hold memory or both arm.
YIELDS: Tuple[Yield, ...] = (
    Yield(
        "hot_replicas",
        "half_window_experts",
        "both stack extra NVFP4 experts per TP4 rank and place each step's experts over them",
    ),
)


def yielded(paths: FrozenSet[str]) -> Tuple[FrozenSet[str], List[Yield]]:
    """``paths`` without those that yield to another path in it, and the yields applied.

    Closed under ``depends_on`` like ``capacity_gated``. Changes nothing.
    """
    applied = [rule for rule in YIELDS if rule.path in paths and rule.to in paths]
    if not applied:
        return paths, applied
    return paths - dependents(rule.path for rule in applied), applied


def compose(
    text_model: Any,
    model_config: Any,
    mode: ModelingV2Mode,
    compositions: Mapping[CompositionKey, TargetPaths],
    compute_capability: Optional[Tuple[int, int]],
    spec_worker: Any = None,
    max_batch_size: Optional[int] = None,
) -> Plan:
    """The plan this model is built with: its composition, or ``NO_PATHS``.

    Call after the built-in constructor and before anything is installed. A
    deployment without a composition (``composition``, structural) gets
    ``NO_PATHS`` under ``auto`` and raises under ``require``. Otherwise the plan
    is the composition's fast paths, less those written for a windowed arena when
    the worker's arena is not windowed, and, over a windowed arena, less the
    arena-writing paths (``FULL_CONTEXT_ARENA_PATHS``) unless the composition also
    serves the window compaction or the architecture is one of
    ``WINDOWED_COMMIT_SMS`` (whose windowed commit addresses the arena by its
    physical columns and compacts itself; structural as well), less the paths a
    capacity gate closes at the engine's ``max_batch_size`` (``CAPACITY_GATES``;
    every rank reads the same engine argument), less the paths that yield to another
    path still in the plan (``YIELDS``), and the preconditions of those
    paths and the arena kind are checked once, agreed over the TP group
    (collective; every rank returns the same plan or raises the same error).

    Raises:
        ValueError: under ``modeling_v2="require"`` when the deployment has no
            composition; under any mode when a precondition is missing on some
            rank, or the ranks disagree on the drafter arena (on every rank alike).
    """
    served = composition(model_config, compositions, compute_capability)
    mapping = getattr(model_config, "mapping", None)
    if served is None:
        require_served(model_config, mode, compositions, compute_capability)
        reason = unserved_reason(model_config, compositions, compute_capability)
        logger.info(
            f"{TAG} no fast path for this deployment ({reason}); running the built-in forward"
        )
        return NO_PATHS
    from tensorrt_llm._torch.speculative.dflash import _PAGED_ATTENTION_BACKENDS

    # The worker pages the drafter's context on these backends, and windows only a dense arena.
    paged = getattr(spec_worker, "_dflash_attention_backend", None) in _PAGED_ATTENTION_BACKENDS
    windowed = windowed_context_arena(spec_worker) and not paged
    paths, gated = capacity_gated(frozenset(served.served), max_batch_size)
    paths, yields = yielded(paths)
    if windowed:
        if not paths & frozenset(WINDOWED_CONTEXT_ARENA_PATHS) and (
            sm_of(compute_capability) not in WINDOWED_COMMIT_SMS
        ):
            paths = paths - dependents(FULL_CONTEXT_ARENA_PATHS)
    else:
        paths = paths - dependents(WINDOWED_CONTEXT_ARENA_PATHS)
    local = missing_preconditions(text_model, model_config, preconditions(paths, served))
    agreed = agreed_union(mapping, local + ([_WINDOWED_ARENA] if windowed else []))
    missing = [item for item in agreed if not item.startswith(_WINDOWED_ARENA)]
    split = [
        item for item in agreed if item.startswith(_WINDOWED_ARENA) and item != _WINDOWED_ARENA
    ]
    if split:
        missing.append(f"the same drafter context arena on every rank ({split[0]})")
    composed = (
        f"{TAG} sm{sm_of(compute_capability)} tep{mapping.tp_size} on {served.topology.description}"
    )
    if missing:
        raise ValueError(f"{composed}: the fast paths need " + "; ".join(missing))
    plan = Plan(paths)
    if windowed:
        composed += f" with {_WINDOWED_ARENA}"
    if paged:
        composed += " with a paged drafter context"
    for gate in gated:
        composed += (
            f" without {gate.path} (max_batch_size {max_batch_size} above {gate.most_requests};"
            f" it costs {gate.cost})"
        )
    for rule in yields:
        composed += f" without {rule.path} (it yields to {rule.to}: {rule.why})"
    logger.info(f"{composed}: {plan.describe()}")
    return plan


# --------------------------------------------------------------------------
# Install refusals agreed over the TP group
# --------------------------------------------------------------------------
_T = TypeVar("_T")


def group_size(mapping) -> int:
    """The TP group size of ``mapping`` (1 for None or an object without ``tp_size``)."""
    return int(getattr(mapping, "tp_size", 1) or 1) if mapping is not None else 1


def agreed_refusal(mapping, local: Optional[str]) -> Optional[str]:
    """``local`` (this rank's reason to refuse, or None) as every rank of ``mapping``'s TP group agrees.

    Collective over the TP group when it has more than one rank: every rank of the
    group must call it at the same point, whatever its own reason. A group of one
    returns ``local``. Returns None when no rank refuses; the reason itself when
    every rank gives the same one; otherwise the distinct reasons, followed by the
    TP ranks that gave one. Changes nothing.
    """
    if group_size(mapping) <= 1:
        return local
    everyone = Distributed.get(mapping).tp_allgather(local, small_payload=True)
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
