# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``hot_replicas``: copies of each layer's hot NVFP4 experts, placed per step (tep4).

Under tep4 expert parallelism rank ``r`` owns experts ``[64 r, 64 r + 64)``. On a
decode step of 5 to 64 requests every rank streams the weights of the experts its
rows touch, and the finalize all-reduce waits for the rank that touches the most:
about four experts above the mean in every layer, at every batch size.

``hot_replicas`` gives every rank ``R`` more experts per NVFP4 layer: copies of other
ranks' hot experts, chosen offline from a routing census (``hot_census``: an
immutable table per layer, every hot expert with exactly one copy on one other rank).
The copies are gathered from their owners once, bitwise their bytes, and stacked
after the rank's own block: ``[own 64 | copies in table order]``. The built-in
parameters are re-bound as views of the stacks' first 64 experts, so the own block
is stored once.

On a decode step of ``MIN_ROWS..MAX_ROWS`` rows ``DecodeTarget`` arms the path
(``armed``); every MoE layer's routing call then also runs
``ops/moe/hot_replica_assign``: from the step's top-k ids and the table it places
every touched hot expert on its owner or its holder (an integer greedy rule, the same
on every rank) and rewrites the ids into this rank's stacked id space. The expert call
runs over the ``64 + R`` window with those ids; its unfinalized output feeds the
built-in finalize unchanged, since the placement partitions the (token, slot) pairs
across the ranks. Prefill, mixed steps and decode steps outside the band run the
built-in (or the owner-window) expert call over the own-block views, unchanged.

Rounding-different from the built-in path: the finalize rounds each rank's partial
sum to BF16 before adding the ranks, and the placement changes which expert rows
share a partial. Expert selection and every expert row are unchanged.

Agreement premise: no rank tells another where an expert runs; each derives the
placement from its own copy of the step's top-k ids, so the ids must be the same
bytes on every rank. They are, because every input of the routing is replicated
bitwise: the routing kernel is deterministic on equal logits; the router GEMM is a
fixed-split reduction over the replicated router weight; its input, the pre-FFN norm
of the residual stream, is identical on every rank because every cross-rank sum that
writes the residual stream (the post-attention hop, the MoE finalize all-reduce and
its two-shot forms) adds the ranks' partials in rank order, and the replicated
pooler / unpooler projections run the same GEMM on the same bytes on every rank. A
disagreement would not mis-route one near-tie slot, as it would in the built-in
expert-parallel call, but run a whole hot expert on two ranks or on none for every
token of that step and layer; a change that makes any of those sums or projections
rank-dependent breaks this premise.

Allocation: only a deployment that serves the band allocates. The model does not
know its ``max_batch_size`` at load, but the eager metadata of its first forward does
(``max_num_requests``, and the speculative metadata's verify width). So the hooks are
installed at load and inert; the first forward decides (``decide``): when the
deployment's largest decode step -- ``max_batch_size`` times the rows a request adds
to a decode step (``decode_rows_per_request``) -- reaches ``MIN_ROWS``
(``replicas_for``) and that forward is eager, every rank gathers its copies
(collective over the TP group) before any CUDA graph is captured; otherwise nothing
is allocated and the path stays off for the life of the engine. A first forward
under graph capture also leaves it off. When the KV-cache estimation runs, the first
forward is its profiling pass, so the copies are charged to the pool like weights;
when it is skipped, the pool is sized before any forward and the copies come out of
the memory the pool leaves free -- a deployment that keeps that headroom lowers its
``free_gpu_memory_fraction`` by the copies' bytes. When less than
``MIN_FREE_AFTER_BYTES`` would remain free on some rank, every rank declines (a
warning) and the path stays off; so does a deployment whose stacks fail to allocate
on some rank part-way, which every rank learns before that layer's first all-gather.
The engine cannot tell whether a disaggregated worker will serve context or
generation requests (the role is decided per request), so a context-only worker
whose ``max_batch_size`` reaches the band also allocates copies, which it never arms.

A later weight load (``install`` again) re-gathers the copies of an allocated
deployment from the reloaded owners, in place; the hooks stay as they are.

The first forward is also the autotuner's warmup pass when the engine runs it (its
profiling forwards are the first forwards when the estimation is skipped): there
the first layer's expert call additionally runs the ``64 + R`` window once in the
tuning context, so that window's tactics are profiled like the 64-expert window's, and
then gives each of its profiles the 64-expert window's tactic where that is valid.
"""

from __future__ import annotations

import contextlib
import types
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ... import dispatch as _tp_agreement
from ...ops.moe.hot_replica_assign import (
    BLOCK,
    EXPERTS,
    MAX_ROWS,
    MIN_ROWS,
    RANKS,
    build_table,
    hot_replica_assign,
    pack_table,
)
from . import hot_census as _census
from . import ring_buddy as _ring_buddy

#: Copies per rank by the rows of the deployment's largest decode step: the last entry
#: whose bound those rows reach applies; below the first bound nothing is allocated.
REPLICAS_BY_DECODE_ROWS: Tuple[Tuple[int, int], ...] = ((MIN_ROWS, 4),)

#: Device memory that must stay free once the copies are allocated (activations,
#: CUDA graphs and workspaces the KV pool left room for).
MIN_FREE_AFTER_BYTES = 12 << 30

#: The per-expert tensors of the native NVFP4 leaf, stacked in this order.
STACKED = _ring_buddy.STACKED["nvfp4"]


def replicas_for(max_decode_rows: int) -> int:
    """Copies per rank for a deployment whose largest decode step has ``max_decode_rows`` rows."""
    replicas = 0
    for bound, count in REPLICAS_BY_DECODE_ROWS:
        if max_decode_rows >= bound:
            replicas = count
    return replicas


def decode_rows_per_request(spec_metadata) -> int:
    """Rows one request adds to a decode step: its verify width under speculative decoding.

    A linear draft verifies ``max_draft_len + 1`` tokens per request, a draft tree
    ``max_total_draft_tokens + 1``; without speculative metadata a request adds one row.
    """
    if spec_metadata is None:
        return 1
    if getattr(spec_metadata, "is_spec_dec_tree", False):
        drafts = getattr(spec_metadata, "max_total_draft_tokens", 0)
    else:
        drafts = getattr(spec_metadata, "max_draft_len", 0)
    return int(drafts or 0) + 1


def eligible(moe, mapping: Mapping) -> Optional[str]:
    """Why this MoE layer takes no copies, or None if it does (the tep4 NVFP4 leaf)."""
    if getattr(mapping, "tp_size", None) != RANKS:
        return "the mapping is not tp4/ep4"
    reason = _ring_buddy.eligible(moe, mapping)
    if reason is not None:
        return reason
    if _ring_buddy.expert_format(moe) != "nvfp4":
        return "the experts are not the native NVFP4 leaf"
    return None


class _Layer:
    """One NVFP4 MoE layer's copies, census table and step placement."""

    def __init__(self, index: int, moe, rank: int):
        self.index = index
        self.moe = moe
        self.rank = rank
        self.stacks: Optional[Dict[str, torch.Tensor]] = None
        # The layer's census table (build_table) and its kernel arguments for this rank.
        self.table: Optional[List[int]] = None
        self.packed: Optional[Tuple[int, ...]] = None
        self.replicas = 0
        self.call = None
        # (router_logits, its version, local_ids, weights_bf16, stats) of the current call.
        self.placement = None
        self.last_stats = None

    def current(self, router_logits: torch.Tensor):
        placement = self.placement
        if (
            placement is None
            or placement[0] is not router_logits
            or placement[1] != _ring_buddy._version(router_logits)
        ):
            return None
        return placement


class HotReplicas:
    """The switch ``DecodeTarget`` arms and the layers' copies.

    Off except inside ``armed``, which ``DecodeTarget`` enters for one decode forward of
    ``takes`` rows. Host state only: under a CUDA graph it is a per-capture constant.
    """

    def __init__(self, layers: List[_Layer], mapping: Mapping):
        self.layers = layers
        self.mapping = mapping
        self.active = False
        self.decided = False
        self.replicas = 0
        self.tuned = False
        self.declined = 0
        self.tables = None

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows is armed (only once copies exist)."""
        return self.replicas > 0 and MIN_ROWS <= rows <= MAX_ROWS

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Place the layers' experts over their copies for one decode forward."""
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def decide(self, attn_metadata, spec_metadata=None) -> None:
        """Allocate the copies on the first forward if the deployment serves the band.

        ``spec_metadata`` is that forward's speculative metadata (None without
        speculative decoding), which gives the rows a request adds to a decode step.
        Collective over the TP group when it allocates; every rank reaches its first
        forward with the same metadata, so every rank takes the same branch.

        Raises:
            ValueError: on every rank alike, for a refusal on some rank.
        """
        if self.decided:
            return
        self.decided = True
        if torch.cuda.is_current_stream_capturing():
            logger.warning(
                "[musespark_open] hot_replicas: the first forward is under CUDA graph capture; "
                "no copies are allocated and the path stays off"
            )
            return
        max_batch = int(getattr(attn_metadata, "max_num_requests", 0) or 0)
        per_request = decode_rows_per_request(spec_metadata)
        replicas = replicas_for(max_batch * per_request)
        if replicas == 0 or not self.layers:
            logger.info(
                f"[musespark_open] hot_replicas: max_batch_size {max_batch} at {per_request} "
                f"rows per request never reaches {MIN_ROWS} decode rows; no copies allocated"
            )
            return
        tables = _census.TABLES.get(replicas)
        missing = [
            layer.index for layer in self.layers if tables is None or layer.index not in tables
        ]
        if missing:
            # The same shipped tables on every rank: no agreement needed.
            raise ValueError(
                f"[musespark_open] hot_replicas: the census has no {replicas}-copy table for "
                f"layers {missing}"
            )
        needed = sum(_expert_bytes(layer.moe) for layer in self.layers) * replicas
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        short = None
        if free - needed < MIN_FREE_AFTER_BYTES:
            short = (
                f"{needed / 2**30:.2f} GiB of copies would leave {(free - needed) / 2**30:.2f} GiB "
                f"free, below {MIN_FREE_AFTER_BYTES / 2**30:.0f} GiB"
            )
        short = _tp_agreement.agreed_refusal(self.mapping, short)
        if short is not None:
            logger.warning(
                f"[musespark_open] hot_replicas: {short}; no copies allocated and the path stays "
                f"off (lower free_gpu_memory_fraction by the copies' bytes to serve it)"
            )
            return
        with torch.inference_mode(False), torch.no_grad():
            self._allocate(replicas, tables, free)

    def _allocate(self, replicas: int, tables, free_before: int) -> None:
        failed = None
        for position, layer in enumerate(self.layers):
            entries = tables[layer.index]
            words = build_table(entries, replicas)
            mine = words[2 * RANKS * replicas + layer.rank * replicas :][:replicas]
            allocated = torch.cuda.memory_allocated()
            # Every rank allocates the layer's stacks before the layer's first all-gather
            # and learns whether every other rank could: a rank that cannot must not leave
            # the others waiting in a gather it never joins.
            stacks, short = None, None
            try:
                stacks = _new_stacks(layer.moe, replicas)
            except torch.cuda.OutOfMemoryError as error:
                short = f"layer {layer.index}: {str(error).splitlines()[0]}"
            short = _tp_agreement.agreed_refusal(self.mapping, short)
            if short is not None:
                del stacks
                torch.cuda.empty_cache()
                for done in self.layers[:position]:
                    done.call = None
                logger.warning(
                    f"[musespark_open] hot_replicas: the copies' stacks could not be allocated "
                    f"({short}); the path stays off, and the {position} layers stacked so far "
                    "keep their stacks unused"
                )
                return
            layer.stacks = _stack_copies(
                layer.moe, self.mapping, entries, mine, replicas, into=stacks
            )
            # The layer's old own-block storage is now unreferenced; return it to the device
            # so the next layer's (larger) stacks need not wait for a failed cudaMalloc.
            torch.cuda.empty_cache()
            layer.table = words
            layer.packed = pack_table(words, layer.rank, replicas)
            layer.replicas = replicas
            if position == 0:
                # The own block must be stored once: the re-bound parameters' old storage
                # is freed only if nothing else still references it.
                grew = torch.cuda.memory_allocated() - allocated
                expected = _expert_bytes(layer.moe) * replicas
                kept = None
                if grew > expected + (64 << 20):
                    kept = (
                        f"layer {layer.index} grew by {grew / 2**30:.2f} GiB for "
                        f"{expected / 2**30:.2f} GiB of copies: the own block's old storage is "
                        "still referenced"
                    )
                kept = _tp_agreement.agreed_refusal(self.mapping, kept)
                if kept is not None:
                    raise ValueError(f"[musespark_open] hot_replicas: {kept}")
            if failed is None:
                try:
                    backend = _ring_buddy._backend(layer.moe)
                    layer.call = _ring_buddy._nvfp4_call(backend, layer.moe.experts.routing_method)
                except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
                    failed = f"layer {layer.index}: {error!r}"
        failed = _tp_agreement.agreed_refusal(
            self.mapping,
            None
            if failed is None
            else f"[musespark_open] hot_replicas could not install: {failed}",
        )
        if failed is not None:
            raise ValueError(failed)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        added = free_before - torch.cuda.mem_get_info()[0]
        self.replicas = replicas
        self.tables = tables
        logger.info(
            f"[musespark_open] hot_replicas: {replicas} hot NVFP4 expert copies per rank on "
            f"{len(self.layers)} MoE layers (census {_census.SOURCE}); per rank "
            f"+{added / 2**30:.2f} GiB of device memory, "
            f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free"
        )

    def refresh(self) -> None:
        """Re-gather an allocated deployment's copies after a weight load, in place.

        Collective over the TP group when copies exist (every rank holds the same
        ``replicas``); nothing otherwise.
        """
        if self.replicas == 0:
            return
        with torch.inference_mode(False), torch.no_grad():
            for layer in self.layers:
                entries = self.tables[layer.index]
                words = layer.table
                mine = words[2 * RANKS * layer.replicas + layer.rank * layer.replicas :]
                _stack_copies(
                    layer.moe,
                    self.mapping,
                    entries,
                    mine[: layer.replicas],
                    layer.replicas,
                    into=layer.stacks,
                    refresh=True,
                )
                layer.placement = None


def _expert_bytes(moe) -> int:
    backend = _ring_buddy._backend(moe)
    return sum(
        getattr(backend, name).data[0].numel() * getattr(backend, name).data.element_size()
        for name in STACKED
    )


def _new_stacks(moe, replicas: int) -> Dict[str, torch.Tensor]:
    """Uninitialized ``[64 + replicas, ...]`` stacks for the layer's per-expert tensors."""
    backend = _ring_buddy._backend(moe)
    backend.quant_method.transform_weights(backend)
    stacks = {}
    for name in STACKED:
        own = getattr(backend, name).data
        stacks[name] = torch.empty(
            (BLOCK + replicas, *own.shape[1:]), dtype=own.dtype, device=own.device
        )
    return stacks


def _stack_copies(
    moe, mapping: Mapping, entries, mine: List[int], replicas: int, into, refresh: bool = False
) -> Dict[str, torch.Tensor]:
    """Fill the layer's ``[own block | copies]`` stacks ``into``; the parameters become views
    of the own block.

    ``into`` holds a stack per per-expert tensor (``_new_stacks``, or with ``refresh``
    the layer's existing stacks after a weight reload, refreshed in place). Collective
    over the TP group. Round ``k`` all-gathers every rank's ``k``-th hot expert of its
    own block (by table order) and each holder keeps the ones it holds. The owner-only
    window's stacks (``ring_buddy.install(windows=1)``) are re-pointed at the same views,
    so no copy of the own block survives.
    """
    backend = _ring_buddy._backend(moe)
    backend.quant_method.transform_weights(backend)
    rank = mapping.tp_rank
    outgoing = [[int(e) for e, _ in entries if int(e) // BLOCK == r] for r in range(RANKS)]
    rounds = max(len(out) for out in outgoing)
    owner_window = getattr(moe, "_replicated_experts", None)
    stacks = {}
    for name in STACKED:
        param = getattr(backend, name)
        own = param.data
        stack = into[name]
        if own.shape != stack[:BLOCK].shape or own.dtype != stack.dtype:
            what = "cannot refresh in place" if refresh else "the stack does not fit"
            raise RuntimeError(
                f"hot_replicas: {name} is now {tuple(own.shape)} {own.dtype}, its stack's "
                f"own block {tuple(stack[:BLOCK].shape)} {stack.dtype}; {what}"
            )
        if own.data_ptr() != stack.data_ptr():
            stack[:BLOCK].copy_(own)
            param.data = stack[:BLOCK]
        del own
        if owner_window is not None and owner_window.windows == 1 and name in owner_window.stacks:
            owner_window.stacks[name] = stack[:BLOCK]
        # Every byte of the gathers comes from the stack's own block, the parameter's bytes.
        flat = stack.reshape(BLOCK + replicas, -1).view(torch.uint8)
        for k in range(rounds):
            mine_out = outgoing[rank]
            if k < len(mine_out):
                local = mine_out[k] - BLOCK * rank
                part = flat[local : local + 1]
            else:
                part = torch.zeros_like(flat[:1])
            gathered = allgather(part.contiguous(), mapping, dim=0)
            for src in range(RANKS):
                if k < len(outgoing[src]) and outgoing[src][k] in mine:
                    flat[BLOCK + mine.index(outgoing[src][k])].copy_(gathered[src])
            del gathered, part
        stacks[name] = stack
    return stacks


def _patch_routing(layer: _Layer, switch: HotReplicas) -> None:
    routing = layer.moe.experts.routing_method
    builtin_apply = routing.apply
    builtin_clear = routing.clear_routing_memo

    def apply(self, router_logits, input_ids=None):
        selected, weights = builtin_apply(router_logits, input_ids)
        if (
            switch.active
            and layer.table is not None
            and layer.current(router_logits) is None
            and router_logits.is_cuda
            and MIN_ROWS <= router_logits.shape[0] <= MAX_ROWS
            and selected.dtype == torch.int32
            and weights.dtype == torch.float32
            and selected.is_contiguous()
            and weights.is_contiguous()
            and not torch.compiler.is_compiling()
        ):
            local_ids, weights_bf16, stats = hot_replica_assign(
                selected, weights, layer.packed, layer.rank, layer.replicas
            )
            layer.placement = (
                router_logits,
                _ring_buddy._version(router_logits),
                local_ids,
                weights_bf16,
                stats,
            )
            layer.last_stats = stats
        return selected, weights

    def clear_routing_memo(self):
        layer.placement = None
        builtin_clear()

    routing.apply = types.MethodType(apply, routing)
    routing.clear_routing_memo = types.MethodType(clear_routing_memo, routing)


def _owner_ids(selected: torch.Tensor, rank: int, replicas: int) -> torch.Tensor:
    """Every expert on its owner, in the ``64 + replicas`` window's id space (for tuning only)."""
    window = BLOCK + replicas
    return torch.where(
        selected // BLOCK == rank, selected - BLOCK * rank, window + selected % (EXPERTS - window)
    ).to(torch.int32)


def _patch_experts(layer: _Layer, switch: HotReplicas) -> None:
    experts = layer.moe.experts
    previous_forward = experts.forward
    routing = experts.routing_method

    def forward(self, x, router_logits, *args, do_finalize=True, **kwargs):
        rows = x.shape[0]
        if switch.active and not do_finalize and not args and MIN_ROWS <= rows <= MAX_ROWS:
            routing.apply(router_logits)
            placement = layer.current(router_logits)
            if placement is not None:
                outputs = layer.call(layer.stacks, x, placement[2], placement[3])
                if outputs is not None:
                    return outputs
            if switch.declined == 0:
                logger.warning(
                    f"[musespark_open] hot_replicas: an armed {rows}-row call of layer "
                    f"{layer.index} took the built-in expert call"
                )
            switch.declined += 1
        elif layer.call is not None and not switch.tuned and not do_finalize and not args:
            # The owner window's call first, so its tactics are in the cache the wide
            # window's are aligned to.
            outputs = previous_forward(x, router_logits, do_finalize=do_finalize, **kwargs)
            _tune_window(layer, switch, x, router_logits)
            return outputs
        return previous_forward(x, router_logits, *args, do_finalize=do_finalize, **kwargs)

    experts.forward = types.MethodType(forward, experts)


def _tune_window(layer: _Layer, switch: HotReplicas, x, router_logits) -> None:
    """Under the autotuner's tuning context, run the ``64 + R`` window once so its tactics
    are profiled (one layer suffices: every layer shares the shapes), then give each of
    its profiles the owner window's tactic (``_owner_window_tactics``). A call the stacked
    call does not take (a BF16 expert input) leaves it for the next tuning forward."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    if not AutoTuner.get().is_tuning_mode or torch.cuda.is_current_stream_capturing():
        return
    selected, weights = layer.moe.experts.routing_method.apply(router_logits)
    ids = _owner_ids(selected, layer.rank, layer.replicas)
    if layer.call(layer.stacks, x, ids, weights.to(torch.bfloat16)) is None:
        return
    switch.tuned = True
    window = BLOCK + layer.replicas
    aligned, profiled = _owner_window_tactics(window)
    message = (
        f"[musespark_open] hot_replicas: profiled the {window}-expert window "
        f"on a {x.shape[0] if hasattr(x, 'shape') else '?'}-row call; {aligned} of its "
        f"{profiled} profiles take the {BLOCK}-expert window's tactic"
    )
    if aligned == 0:
        # Every profile kept its own tactic: the wide window may run another tile than the
        # owner window on the same rows (or the autotuner's cache keys were not recognized).
        logger.warning(
            f"{message}; the {window}-expert window keeps its separately profiled tactics"
        )
    else:
        logger.info(message)


_EXPERT_OP = "trtllm::fp4_block_scale_moe_runner"
_HIDDEN_STATES = 2


def _valid_tactics(top_k: int, hidden: int, intermediate: int, window: int, rows: int):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    act_type = 0
    runner = FP4BlockScaleMoERunner.runner_dict.get((act_type,))
    if runner is None:
        runner = torch.classes.trtllm.FP4BlockScaleMoERunner(act_type)
    return runner.get_valid_configs(top_k, hidden, intermediate, window, rows)


def _owner_window_tactics(window: int) -> Tuple[int, int]:
    """Give every autotuner profile of the ``window``-expert NVFP4 expert call the tactic
    the 64-expert (owner) window holds for the same profile, where it is valid for
    ``window`` experts; returns ``(profiles holding the owner's tactic, profiles of the window)``.

    The two windows run the same rows with ``R`` experts more or less on a rank, and the
    owner window's tactics are those every deployment already serves its decode steps
    with; a separately profiled wide window can settle on a different tile (profiles
    of random routing, measured at another time) that is slower on the served routing.
    Only the SwiGLU expert (act type 0) is aligned, the one ``hot_replicas`` serves.
    """
    from tensorrt_llm._torch.autotuner import AutoTuner

    cache = AutoTuner.get().profiling_cache.cache
    aligned = 0
    profiled = 0
    for key, entry in list(cache.items()):
        op, runner_name, unique_id, shapes = key
        if op != _EXPERT_OP:
            continue
        fields = unique_id.strip("()").split(",")
        if len(fields) != 4:
            continue
        top_k, intermediate, local, act_type = (int(v) for v in fields)
        if local != window or act_type != 0:
            continue
        profiled += 1
        owner_shapes = tuple(
            (BLOCK, *shape[1:])
            if index != _HIDDEN_STATES and len(shape) > 0 and shape[0] == window
            else shape
            for index, shape in enumerate(shapes)
        )
        owner_id = str((top_k, intermediate, BLOCK, act_type))
        owner = cache.get((op, runner_name, owner_id, owner_shapes))
        if owner is None:
            continue
        tactic = owner[1]
        if not isinstance(tactic, (list, tuple)):
            continue
        rows, hidden_bytes = shapes[_HIDDEN_STATES][0], shapes[_HIDDEN_STATES][1]
        if list(tactic) not in [
            list(t) for t in _valid_tactics(top_k, 2 * hidden_bytes, intermediate, window, rows)
        ]:
            continue
        cache[key] = (entry[0], tactic, entry[2])
        aligned += 1
    return aligned, profiled


def install(text_model, mapping: Mapping) -> Optional[HotReplicas]:
    """Hook every eligible NVFP4 MoE layer of ``text_model``; returns the switch (or None).

    Allocates nothing (``HotReplicas.decide`` does, on the first forward). Must run after
    the owner-window install (``ring_buddy.install(windows=1)``), whose expert and routing
    hooks it wraps, and before any forward. Called again (a weight reload), it refreshes
    the copies an allocated deployment holds (``HotReplicas.refresh``) and hooks nothing.
    """
    switch = getattr(text_model, "_hot_replicas", None)
    if switch is not None:
        switch.refresh()
        return switch
    layers = []
    skipped = {}
    for index, decoder_layer in enumerate(text_model.layers):
        moe = decoder_layer.feed_forward
        reason = eligible(moe, mapping)
        if reason is not None:
            skipped[index] = reason
            continue
        layers.append(_Layer(index, moe, mapping.tp_rank))
    if not layers:
        logger.warning(
            f"[musespark_open] hot_replicas: no eligible layer ({sorted(set(skipped.values()))})"
        )
        return None
    switch = HotReplicas(layers, mapping)
    for layer in layers:
        _patch_routing(layer, switch)
        _patch_experts(layer, switch)
    text_model._hot_replicas = switch
    logger.info(
        f"[musespark_open] hot_replicas: hooks on {len(layers)} NVFP4 MoE layers (copies allocated "
        f"on the first forward of a deployment whose decode steps reach "
        f"{REPLICAS_BY_DECODE_ROWS[0][0]} rows); built-in path kept for layers {sorted(skipped)}"
    )
    return switch
