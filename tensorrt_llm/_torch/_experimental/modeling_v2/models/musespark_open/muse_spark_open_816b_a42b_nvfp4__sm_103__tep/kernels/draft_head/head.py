# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``draft_head``: the DFlash 2 draft head on vocab shards.

The drafter borrows the target's LM head, which is column-parallel with
``gather_output``: every draft step all-gathers ``[K, vocab]`` logits, runs
``torch.topk`` over the full vocabulary for the selector's candidates, and
after the selector's walk fills a ``[K, vocab]`` tensor with ``-inf``,
scatters the walked scores into it and takes the greedy ``argmax`` over it.
Only the top ``selector_top_k`` candidates per position ever matter.

With this feature the draft step runs, per rank:

1. the same shard GEMM the LM head runs (``lm_head.apply_linear``), without
   the all-gather;
2. ``draft_shard_topk``: top-k keys per 4096-column chunk of the shard;
3. an exchange of those keys only (a few KB instead of the logits): the
   MNNVL one-shot all-reduce in fp32, each rank's keys as exact 16-bit
   integer halves in its own slot and +0.0 elsewhere, or an NCCL all-gather
   of the int32 keys where that all-reduce is unavailable (an exchange that
   carries a key selection of its own, ``wide_draft_topk``'s, takes the rows
   it admits instead of this one);
4. ``draft_topk_merge``: ``torch.topk``'s full-vocab result rebuilt exactly
   (selection, tie order) plus the selector's einsum operands;
5. the built-in ``hidden_projection`` and the built-in einsum, unchanged;
6. ``draft_path_walk``: the walk and the greedy draft tokens.

The draft tokens are bitwise the built-in ones. Batches that sample drafts
(rejection sampling of a non-greedy batch) or map draft vocab ids need the
full-vocab logits; for those the worker all-gathers the shard logits itself
and runs the built-in selector, which is the built-in computation.

Installed from the shell's ``load_draft_weights`` (the drafter's selector and
the borrowed head exist only after the drafter loads), or from the shell's
post-load hooks on load paths that never call it (``LoadFormat.DUMMY``, which
never loads the drafter, so there the install refuses: no DFlash 2 drafter):
the built-in DFlash worker is re-classed in place to
``ShardedDraftHeadDFlashWorker`` and the drafter's logits processor is
wrapped. Nothing else changes.

The merge reproduces the tie order of the ``torch.topk`` build it was written
against, which torch does not document. Before installing, ``tie_order_reason``
runs ``torch.topk`` and the sharded path on tie-heavy logits of the deployed
shape and refuses the install if they disagree.
"""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from ...ops.spec.draft_path_walk import DraftPathWalk
from ...ops.spec.draft_shard_topk import CHUNK, NUM_WARPS, DraftShardTopK
from ...ops.spec.draft_topk_merge import SORT_SLOTS, DraftTopKMerge

draft_shard_topk = DraftShardTopK()
draft_topk_merge = DraftTopKMerge()
draft_path_walk = DraftPathWalk()

#: Attribute of the zero-copy block-logits stand-in that carries the draft tokens.
TOKENS_ATTR = "_draft_head_tokens"

#: Largest shard the 16-bit column field of a shard key holds.
MAX_SHARD = 65536

#: Rows of the install-time tie-order probe. ``torch.topk`` over a vocabulary
#: this wide takes its multi-block path for up to 20 rows, as every draft step
#: of this target does (one request, K + 1 = 8 block positions).
PROBE_ROWS = 8

#: Seed of the probe's logits, drawn on the host so no device RNG state moves.
PROBE_SEED = 1234


def gathers_column_shards(lm_head: nn.Module) -> bool:
    """Whether ``lm_head`` is a column-parallel head whose forward is shard GEMM + all-gather.

    Only even shards in rank order qualify: global vocab id = rank * shard +
    column. ``gather_output_sizes`` is set exactly when the shards are uneven.
    """
    return (
        getattr(lm_head, "tp_mode", None) == TensorParallelMode.COLUMN
        and bool(getattr(lm_head, "gather_output", False))
        and getattr(lm_head, "tp_size", 1) > 1
        and getattr(lm_head, "padding_size", 0) == 0
        and getattr(lm_head, "gather_output_sizes", None) is None
        and getattr(lm_head, "lora", None) is None
        and not getattr(lm_head, "enable_lm_head_tp_in_adp", False)
        and getattr(getattr(lm_head, "weight", None), "dtype", None) == torch.bfloat16
    )


def key_allreduce(mapping) -> nn.Module | None:
    """An MNNVL one-shot fp32 all-reduce for the key exchange, or None.

    The keys travel as exact 16-bit integer halves in fp32, each rank's in its
    own slot and +0.0 elsewhere, so the sum is the concatenation. Over the
    fabric this is a one-shot kernel, cheaper than NCCL's all-gather of the
    same bytes. None when the MNNVL all-reduce cannot be built, which
    ``install`` refuses. Collective
    to construct -- every rank of ``mapping`` must call it.
    """
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.float32)
    return allreduce if allreduce.mnnvl_allreduce is not None else None


def selector_draft_tokens(
    local_logits: torch.Tensor,
    block_hidden_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    selector: nn.Module,
    mapping,
    allreduce: nn.Module | None = None,
    walk=None,
):
    """Greedy DFlash 2 draft tokens ``[B, K]`` int32 from this rank's ``[B, K, shard]`` logits.

    ``allreduce`` (from ``key_allreduce``) exchanges the keys as fp32 halves;
    without it they are all-gathered as int32. An ``allreduce`` with a
    ``takes(rows)`` method (``wide_draft_topk``'s ``RankTopKExchange``) selects
    and packs the keys itself (``keys``) and names the merge's launch
    (``merge_launch``) on the steps it takes. ``walk`` stands in for
    ``draft_path_walk`` (same arguments; its result is returned as is):
    the ``path_walk_tail`` fast path passes a walk that also writes the step's tail.
    """
    requests, positions, shard = local_logits.shape
    local = local_logits.reshape(requests * positions, shard)
    merge_launch = {}
    if allreduce is not None:
        takes = getattr(allreduce, "takes", None)
        if takes is not None and takes(local.shape[0]):
            packed = allreduce.keys(local, selector.top_k, mapping.tp_rank, mapping.tp_size)
            merge_launch = allreduce.merge_launch(local.shape[0])
        else:
            packed = draft_shard_topk(
                local, selector.top_k, CHUNK, NUM_WARPS, mapping.tp_rank, mapping.tp_size
            )
        gathered = allreduce(packed)
    else:
        gathered = allgather(
            draft_shard_topk(local, selector.top_k, CHUNK, NUM_WARPS), mapping, dim=0
        )
    # The built-in selector's projection call, operand for operand.
    gate = selector.hidden_projection(
        block_hidden_states.to(selector.hidden_projection.weight.dtype)
    )
    ids, unary, predecessors, successors = draft_topk_merge(
        gathered,
        shard,
        requests,
        positions,
        anchor_tokens.reshape(-1),
        gate,
        selector.predecessor_codebook,
        selector.successor_codebook,
        selector.top_k,
        mapping.tp_size,
        **merge_launch,
    )
    # dflash2_score_edges' einsum on tensors of its shapes and strides: the same kernel.
    edges = torch.einsum("blpr,blcr->blpc", predecessors, successors)
    return (draft_path_walk if walk is None else walk)(edges, unary, ids)


class ShardDraftLogitsProcessor(nn.Module):
    """The drafter's logits processor, returning the LM head's bf16 shard un-gathered.

    Only the call the DFlash worker makes (``return_context_logits=True`` on
    a head ``gathers_column_shards`` admits) is changed; anything else goes to
    the wrapped processor.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        if return_context_logits and gathers_column_shards(lm_head):
            fork = self.__dict__.get("_selector_fork")
            if fork is not None:
                # The selector fork of the PDL chain (drafter_pdl_chain/chain.py): the
                # selector's projection of these rows on a side stream, under the LM-head GEMM.
                fork.launch(hidden_states)
            return lm_head.apply_linear(hidden_states, lm_head.bias)
        return self.inner(hidden_states, lm_head, attn_metadata, return_context_logits)


class ShardedDraftHeadDFlashWorker(DFlashWorker):
    """The built-in DFlash worker with the draft head on vocab shards (fast path ``draft_head``)."""

    def _draft_slot_ids(self, draft_model, num_gens, block_size, num_draft_tokens):
        """The built-in slot ids, built once per shape outside capture and reused.

        Five ``arange``/arithmetic launches per step become none. A tensor
        first requested during a capture is not kept: it would live in that
        graph's pool.
        """
        cache = self.__dict__.setdefault("_draft_head_slot_ids", {})
        key = (num_gens, block_size, num_draft_tokens)
        ids = cache.get(key)
        if ids is None:
            ids = super()._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
            if not torch.cuda.is_current_stream_capturing():
                cache[key] = ids
        if self.__dict__.get("_pdl_head"):
            return self._pdl_head_rows(key, ids)
        return ids

    def _pdl_head_rows(self, key, ids):
        """The PDL head (``drafter_pdl_chain/chain.py``): a contiguous one-request slot range as a row view."""
        from ..drafter_pdl_chain import chain as _drafter_pdl

        ranges = self.__dict__.setdefault("_pdl_head_ranges", {})
        if key not in ranges:
            if torch.cuda.is_current_stream_capturing():
                return ids
            ranges[key] = _drafter_pdl.draft_row_range(*key, ids)
        rows = ranges[key]
        return ids if rows is None else _drafter_pdl.DraftRowRange(ids, *rows)

    def _shard_selector_applies(self, gen_logits, spec_metadata, full_vocab) -> bool:
        mapping = self.mapping
        return (
            gen_logits.dim() == 3
            and gen_logits.dtype == torch.bfloat16
            and gen_logits.stride(-1) == 1
            and gen_logits.shape[-1] * mapping.tp_size == full_vocab
            and not spec_metadata.wants_advanced_draft_sampling
            and self._d2t is None
        )

    def _apply_dflash2_selector(
        self, draft_model, gen_logits, block_hidden_states, anchor_tokens, spec_metadata
    ):
        selector = draft_model.candidate_selector
        full_vocab = selector.predecessor_codebook.shape[0]
        if gen_logits.shape[-1] == full_vocab:
            return super()._apply_dflash2_selector(
                draft_model, gen_logits, block_hidden_states, anchor_tokens, spec_metadata
            )
        if self._shard_selector_applies(gen_logits, spec_metadata, full_vocab):
            tokens = selector_draft_tokens(
                gen_logits,
                block_hidden_states,
                anchor_tokens,
                selector,
                self.mapping,
                self.__dict__.get("_draft_head_allreduce"),
            )
            stand_in = tokens.unsqueeze(-1).expand(*tokens.shape, full_vocab)
            setattr(stand_in, TOKENS_ATTR, tokens)
            return stand_in
        # The built-in path: the head's own all-gather and .float(), then the built-in selector.
        full_logits = allgather(gen_logits, self.mapping, dim=-1).float()
        return super()._apply_dflash2_selector(
            draft_model, full_logits, block_hidden_states, anchor_tokens, spec_metadata
        )

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        tokens = getattr(logits, TOKENS_ATTR, None)
        if tokens is not None:
            return tokens
        return super().sample_draft_tokens(logits, spec_metadata, batch_size, **kwargs)


def unsupported_reason(worker, draft_model) -> str | None:
    """Why the draft head cannot run on this drafter, or None."""
    if type(worker) not in (DFlashWorker, ShardedDraftHeadDFlashWorker):
        return f"the spec worker is {type(worker).__name__}, not the built-in DFlashWorker"
    selector = getattr(draft_model, "candidate_selector", None)
    if selector is None:
        return "the drafter has no DFlash 2 candidate selector"
    processor = getattr(draft_model, "logits_processor", None)
    if type(processor) not in (LogitsProcessor, ShardDraftLogitsProcessor):
        return f"the drafter's logits processor is {type(processor).__name__}"
    lm_head = getattr(draft_model, "lm_head", None)
    if not gathers_column_shards(lm_head):
        return "the draft LM head is not a gathered bf16 column-parallel head"
    mapping = worker.mapping
    if mapping.enable_attention_dp:
        return "attention DP"
    full_vocab = selector.predecessor_codebook.shape[0]
    if lm_head.out_features * mapping.tp_size != full_vocab:
        return f"shard {lm_head.out_features} x tp {mapping.tp_size} != selector vocab {full_vocab}"
    if lm_head.out_features > MAX_SHARD:
        return f"shard {lm_head.out_features} exceeds {MAX_SHARD} columns"
    top_k = selector.top_k
    if top_k > SORT_SLOTS or top_k & (top_k - 1):
        return f"selector_top_k {top_k} is not a power of two <= {SORT_SLOTS}"
    for name in ("predecessor_codebook", "successor_codebook"):
        if getattr(selector, name).dtype != torch.bfloat16:
            return f"selector {name} is not bf16"
    if selector.hidden_projection.weight.dtype != torch.bfloat16:
        return "selector hidden_projection is not bf16"
    return None


def tie_order_probe_logits(vocab: int, shard: int, rows: int = PROBE_ROWS) -> torch.Tensor:
    """Host bf16 ``[rows, vocab]`` logits whose top-k order is decided by ties.

    Rows, in turn: all +0.0; +0.0 and -0.0 mixed (equal under ``==``, ordered
    by torch's radix key); value grids of 4 and 40 levels (thousands of ties
    at the maximum); five distinct values above a tied grid (both groups of
    ``gatherTopK``'s arrangement); twenty +inf; three NaNs and nine equal
    values above a tied grid; a tie group on every shard's first and last
    columns below three larger values. ``shard`` places the last row's ties.
    """
    g = torch.Generator().manual_seed(PROBE_SEED)

    def grid(levels):
        return torch.randint(0, levels, (vocab,), generator=g).float() * 0.25 - 3.0

    def place(row, values):
        columns = torch.randperm(vocab, generator=g)[: len(values)]
        row[columns] = torch.tensor(values)
        return row

    signed_zeros = torch.where(torch.rand(vocab, generator=g) < 0.5, 0.0, -0.0)
    edge_row = grid(4)
    edge_row[[c for r in range(vocab // shard) for c in (r * shard, (r + 1) * shard - 1)]] = 7.0
    edge_row[[shard // 2, shard + shard // 2, vocab - shard // 2]] = 7.5
    probe = [
        torch.zeros(vocab),
        signed_zeros,
        grid(4),
        grid(40),
        place(grid(8), [10.0, 11.0, 12.0, 13.0, 14.0]),
        place(grid(17), [float("inf")] * 20),
        place(grid(8), [float("nan")] * 3 + [5.0] * 9),
        edge_row,
    ]
    return torch.stack([probe[r % len(probe)] for r in range(rows)]).to(torch.bfloat16)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of fp32 tensors, NaN payloads aside."""
    nan = a.isnan()
    if not torch.equal(nan, b.isnan()):
        return False
    return torch.equal(
        torch.where(nan, 0.0, a).view(torch.int32), torch.where(nan, 0.0, b).view(torch.int32)
    )


def tie_order_reason(selector: nn.Module, num_ranks: int, topk=torch.topk) -> str | None:
    """Why the sharded candidates would differ from this torch's ``torch.topk``, or None.

    The merge copies the tie order of ``torch.topk`` (selection, arrangement
    and bitonic network), which torch does not document. This runs ``topk``
    as the built-in selector does -- ``[1, rows, vocab]`` fp32 -- on
    ``tie_order_probe_logits``, and the sharded path on the same logits split
    into ``num_ranks`` shards with the keys exchanged in both layouts
    (concatenated int32 keys and summed fp32 halves, which is what the
    all-gather and the all-reduce return), and compares ids and values bit for
    bit. ``topk`` is a parameter so a test can stand in another order.
    """
    predecessor_codebook = selector.predecessor_codebook
    vocab, rank_dim = predecessor_codebook.shape
    shard = vocab // num_ranks
    top_k = selector.top_k
    device = predecessor_codebook.device
    with torch.cuda.device(device):
        full = tie_order_probe_logits(vocab, shard).to(device)
        rows = full.shape[0]
        ref_values, ref_ids = topk(full.float().view(1, rows, vocab), top_k, dim=-1)
        shards = [full[:, r * shard : (r + 1) * shard].contiguous() for r in range(num_ranks)]
        layouts = {
            "int32 all-gather": torch.cat(
                [draft_shard_topk(s, top_k, CHUNK, NUM_WARPS) for s in shards], dim=0
            ),
            "fp32 all-reduce": torch.stack(
                [
                    draft_shard_topk(s, top_k, CHUNK, NUM_WARPS, r, num_ranks)
                    for r, s in enumerate(shards)
                ]
            ).sum(dim=0),
        }
        gate = torch.zeros(1, rows, rank_dim, dtype=torch.bfloat16, device=device)
        anchor = torch.zeros(1, dtype=torch.int64, device=device)
        for layout, keys in layouts.items():
            ids, values, _, _ = draft_topk_merge(
                keys,
                shard,
                1,
                rows,
                anchor,
                gate,
                predecessor_codebook,
                selector.successor_codebook,
                top_k,
                num_ranks,
            )
            if not (torch.equal(ids, ref_ids) and _same_bits(values, ref_values)):
                differ = (ids != ref_ids).any(dim=-1).view(-1).nonzero().view(-1).tolist()
                return (
                    f"torch {torch.__version__} orders top-k ties differently from the "
                    f"merge ({layout} keys; probe rows {differ} differ)"
                )
    return None


def installed(llm: nn.Module) -> bool:
    """Whether ``install`` has put the draft head on ``llm``'s DFlash worker."""
    return isinstance(getattr(llm, "spec_worker", None), ShardedDraftHeadDFlashWorker)


def install(llm: nn.Module, make_allreduce=key_allreduce) -> bool:
    """Install the draft head on ``llm``'s DFlash worker and drafter; return True.

    ``make_allreduce(mapping)`` builds the key exchange (collective: every rank
    installs at the same point of the load). A second call on an installed
    ``llm`` changes nothing.

    Raises:
        ValueError: when there is no DFlash 2 drafter (no spec worker, or a
            drafter without a candidate selector), or its head layout is not one
            the feature supports, or this torch's ``torch.topk`` orders ties
            differently from the merge (``tie_order_reason``) -- running the
            built-in path, or a path whose drafts differ, under the fast
            path's name would misreport what runs.
    """
    if installed(llm):
        return True
    worker = getattr(llm, "spec_worker", None)
    draft_model = getattr(llm, "draft_model", None)
    if worker is None or getattr(draft_model, "candidate_selector", None) is None:
        raise ValueError(
            "[musespark_open] draft_head cannot run here: there is no DFlash 2 drafter"
        )
    reason = unsupported_reason(worker, draft_model) or tie_order_reason(
        draft_model.candidate_selector, worker.mapping.tp_size
    )
    if reason is not None:
        raise ValueError(f"[musespark_open] draft_head cannot run here: {reason}")
    worker.__class__ = ShardedDraftHeadDFlashWorker
    if type(draft_model.logits_processor) is LogitsProcessor:
        draft_model.logits_processor = ShardDraftLogitsProcessor(draft_model.logits_processor)
    if "_draft_head_allreduce" not in worker.__dict__:
        allreduce = make_allreduce(worker.mapping)
        if allreduce is None:
            # The composition's precondition checked the group takes an MNNVL all-reduce.
            raise ValueError(
                "the vocab-sharded draft head: the MNNVL FP32 all-reduce of the key exchange "
                "could not be built on this TP group"
            )
        # Not registered as a submodule: it owns no parameters, only a fabric workspace.
        worker.__dict__["_draft_head_allreduce"] = allreduce
    exchange = "MNNVL all-reduce" if worker.__dict__["_draft_head_allreduce"] else "NCCL all-gather"
    logger.info(
        "[musespark_open] draft_head: DFlash 2 draft head on vocab shards "
        f"(shard {draft_model.lm_head.out_features}, tp {worker.mapping.tp_size}, "
        f"top_k {draft_model.candidate_selector.top_k}, keys by {exchange}, "
        f"tie order checked against torch {torch.__version__})"
    )
    return True
