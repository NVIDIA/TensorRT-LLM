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

A drafter that declares a logit transform (``DFlashForCausalLM.
apply_draft_logit_transform``: a multiplier and a tanh softcap, in fp32) is
transformed after the top-k instead of before it (``logit_transform.py``):
stages 2 to 4 run on the bf16 shard with canonical keys, which rank as the
transformed logits do, and the merge reads each candidate's transformed value
from a table the drafter's own method computed. The built-in worker calls the
transform right after the drafter's logits processor; the install puts a
``_DeferredTransform`` on the drafter instance in its place, which hands back
the very shard tensor ``ShardDraftLogitsProcessor`` just returned, untouched,
and records it. ``ShardedDraftHeadDFlashWorker._refine_block_logits``, the
next worker method the built-in forward calls on those logits, keeps them in
bf16 only when this step takes the sharded selector, and applies the
drafter's transform otherwise, so every other path (sampled drafts, d2t,
another layout, any other tensor) gets the built-in transform bit for bit.
With a transform installed, bf16 draft logits are untransformed ones: the
transform always returns fp32. The tables hold only while the drafter's
``logit_multiplier`` and ``logit_softcap`` are the ones they were built from;
a step on which they differ takes the built-in path with the drafter's
current transform. Each reason a step with a transform installed takes the
built-in path is logged once, as a warning (``_warn_builtin``).

Installed from the shell's ``load_draft_weights`` (the drafter's selector and
the borrowed head exist only after the drafter loads), or from the shell's
post-load hooks on load paths that never call it (``LoadFormat.DUMMY``, which
never loads the drafter, so there the install refuses: no DFlash 2 drafter):
the built-in DFlash worker is re-classed in place to
``ShardedDraftHeadDFlashWorker``, the drafter's logits processor is wrapped
and, with a logit transform, the drafter instance gets the deferring
transform. Nothing else changes.

The merge reproduces the tie order of the ``torch.topk`` build it was written
against, which torch does not document. Before installing, ``tie_order_reason``
runs ``torch.topk`` and the sharded path on tie-heavy logits of the deployed
shape (with a transform, also on logits whose transformed values tie) and
refuses the install if they disagree.
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from ...ops.spec.draft_path_walk import DraftPathWalk
from ...ops.spec.draft_shard_topk import CHUNK, FINITE_KEYS, NUM_WARPS, DraftShardTopK
from ...ops.spec.draft_topk_merge import SORT_SLOTS, DraftTopKMerge
from . import logit_transform as _logit_transform
from .logit_transform import DraftLogitTransform

draft_shard_topk = DraftShardTopK()
draft_topk_merge = DraftTopKMerge()
draft_path_walk = DraftPathWalk()

#: Attribute of the zero-copy block-logits stand-in that carries the draft tokens.
TOKENS_ATTR = "_draft_head_tokens"

#: Worker attribute holding the drafter's ``DraftLogitTransform`` (absent without a transform).
TRANSFORM_ATTR = "_draft_head_transform"
#: Worker attribute: this step's logits were kept untransformed for the sharded selector.
OWED_ATTR = "_draft_head_owed"
#: Processor attribute: a weak reference to the shard tensor its last call returned.
SHARD_ATTR = "_draft_head_shard"
#: Drafter attribute: ``(data_ptr, numel)`` of the logits the deferring transform handed back.
DEFERRED_ATTR = "_draft_head_deferred"
#: Drafter attribute: why the deferring transform's last call applied the transform in full.
DECLINED_ATTR = "_draft_head_declined"
#: The drafter method the built-in worker calls on the draft logits.
TRANSFORM_METHOD = "apply_draft_logit_transform"

#: Why a step takes the built-in path: the drafter's transform attributes moved off the tables.
STALE_TABLES = "the drafter's logit_multiplier or logit_softcap differs from the install's"
#: Why a step takes the built-in path: its bf16 logits are not the shard the transform deferred.
NOT_DEFERRED = "the draft logits are not the shard the deferring transform handed back"

#: Largest shard the 16-bit column field of a shard key holds.
MAX_SHARD = 65536

#: Rows of the install-time tie-order probe. ``torch.topk`` over a vocabulary
#: this wide takes its multi-block path for up to 20 rows, as every draft step
#: of this target does (one request, K + 1 = 8 block positions).
PROBE_ROWS = 8

#: Rows of the install-time probe of many requests (24 requests of K = 7
#: positions), run with a drafter logit transform.
WIDE_PROBE_ROWS = 168

#: Seed of the probe's logits, drawn on the host so no device RNG state moves.
PROBE_SEED = 1234


class DraftTransformStateError(RuntimeError):
    """A step kept its draft logits untransformed and no selector applied the transform."""


def _warn_builtin(reason: str) -> None:
    """Warn, once per ``reason``, that a draft step with a transform installed takes the built-in path."""
    logger.warning_once(
        "[musespark_open] draft_head: a draft step takes the built-in path, with the drafter's "
        f"logit transform applied before the top-k ({reason})",
        key=f"musespark_open.draft_head.builtin.{reason}",
    )


def _builtin_transform(draft_model, logits: torch.Tensor) -> torch.Tensor:
    """The drafter's class ``apply_draft_logit_transform`` on ``logits``, with its current attributes.

    The call the built-in worker makes on a drafter without the deferring
    transform: fp32 out when the drafter declares a transform, ``logits``
    itself when it declares none.
    """
    return getattr(type(draft_model), TRANSFORM_METHOD)(draft_model, logits)


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


def canon_kwargs(transform: Optional[DraftLogitTransform]) -> dict:
    """``draft_shard_topk`` / ``draft_rank_topk`` keywords for ``transform`` (none without one)."""
    if transform is None:
        return {}
    return {"canon": transform.canon, "canon_band": transform.band}


def values_kwargs(transform: Optional[DraftLogitTransform]) -> dict:
    """``draft_topk_merge`` keywords for ``transform`` (none without one)."""
    return {} if transform is None else {"values": transform.values}


def selector_draft_tokens(
    local_logits: torch.Tensor,
    block_hidden_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    selector: nn.Module,
    mapping,
    allreduce: nn.Module | None = None,
    walk=None,
    *,
    transform: Optional[DraftLogitTransform],
):
    """Greedy DFlash 2 draft tokens ``[B, K]`` int32 from this rank's ``[B, K, shard]`` logits.

    ``allreduce`` (from ``key_allreduce``) exchanges the keys as fp32 halves;
    without it they are all-gathered as int32. An ``allreduce`` with a
    ``takes(rows)`` method (``wide_draft_topk``'s ``RankTopKExchange``) selects
    and packs the keys itself (``keys``) and names the merge's launch
    (``merge_launch``) on the steps it takes. ``walk`` stands in for
    ``draft_path_walk`` (same arguments; its result is returned as is):
    the ``path_walk_tail`` fast path passes a walk that also writes the step's tail.

    ``transform`` is the drafter's logit transform when ``local_logits`` are
    the untransformed bf16 shard of a drafter that declares one, else None:
    the keys are then canonical and the unary values the transformed ones, so
    the tokens are the built-in ones on the transformed logits. Keyword-only
    and required, so no call site drops it by accident.
    """
    requests, positions, shard = local_logits.shape
    local = local_logits.reshape(requests * positions, shard)
    canon = canon_kwargs(transform)
    merge_launch = {}
    if allreduce is not None:
        takes = getattr(allreduce, "takes", None)
        if takes is not None and takes(local.shape[0]):
            packed = allreduce.keys(
                local, selector.top_k, mapping.tp_rank, mapping.tp_size, **canon
            )
            merge_launch = allreduce.merge_launch(local.shape[0])
        else:
            packed = draft_shard_topk(
                local, selector.top_k, CHUNK, NUM_WARPS, mapping.tp_rank, mapping.tp_size, **canon
            )
        gathered = allreduce(packed)
    else:
        gathered = allgather(
            draft_shard_topk(local, selector.top_k, CHUNK, NUM_WARPS, **canon), mapping, dim=0
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
        **values_kwargs(transform),
    )
    # dflash2_score_edges' einsum on tensors of its shapes and strides: the same kernel.
    edges = torch.einsum("blpr,blcr->blpc", predecessors, successors)
    return (draft_path_walk if walk is None else walk)(edges, unary, ids)


class ShardDraftLogitsProcessor(nn.Module):
    """The drafter's logits processor, returning the LM head's bf16 shard un-gathered.

    Only the call the DFlash worker makes (``return_context_logits=True`` on
    a head ``gathers_column_shards`` admits) is changed; anything else goes to
    the wrapped processor. Each call records a weak reference to the shard it
    returned (``SHARD_ATTR``; cleared by every other call), which the
    deferring drafter transform matches.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        self.__dict__.pop(SHARD_ATTR, None)
        if return_context_logits and gathers_column_shards(lm_head):
            fork = self.__dict__.get("_selector_fork")
            if fork is not None:
                # The selector fork of the PDL chain (drafter_pdl_chain/chain.py): the
                # selector's projection of these rows on a side stream, under the LM-head GEMM.
                fork.launch(hidden_states)
            out = lm_head.apply_linear(hidden_states, lm_head.bias)
            self.__dict__[SHARD_ATTR] = weakref.ref(out)
            return out
        return self.inner(hidden_states, lm_head, attn_metadata, return_context_logits)


class _DeferredTransform:
    """The drafter's ``apply_draft_logit_transform`` with the draft head's transform installed.

    Set on the drafter instance (``__dict__``), where it shadows the class
    method. A call on the shard tensor the drafter's ``ShardDraftLogitsProcessor``
    has just returned, while the drafter still declares the transform the
    tables hold and the worker class still routes those logits through
    ``ShardedDraftHeadDFlashWorker._refine_block_logits`` (``_defers_transform``),
    returns that tensor untouched and records its ``(data_ptr, numel)`` on the
    drafter (``DEFERRED_ATTR``). Every other call returns the drafter's own
    transform with its current attributes, the built-in computation, records
    why on the drafter (``DECLINED_ATTR``) and warns once per reason
    (``_warn_builtin``). Holds weak references only.
    """

    def __init__(self, worker, draft_model):
        self._worker = weakref.ref(worker)
        self._model = weakref.ref(draft_model)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        draft_model = self._model()
        worker = self._worker()
        if draft_model is None:
            raise DraftTransformStateError("[musespark_open] draft_head: the drafter is gone")
        transform = None if worker is None else worker.__dict__.get(TRANSFORM_ATTR)
        draft_model.__dict__[DEFERRED_ATTR] = None
        draft_model.__dict__[DECLINED_ATTR] = None
        # The processor in place now, not the one of the install: a wrapper put
        # around it later records nothing, and the transform then runs here.
        processor = getattr(draft_model, "logits_processor", None)
        shard = None
        if isinstance(processor, ShardDraftLogitsProcessor):
            shard = processor.__dict__.pop(SHARD_ATTR, None)
        if transform is None:
            return _builtin_transform(draft_model, logits)
        if not isinstance(processor, ShardDraftLogitsProcessor):
            reason = f"the drafter's logits processor is {type(processor).__name__}"
        elif shard is None or shard() is not logits:
            reason = "the transformed tensor is not the shard the logits processor returned"
        elif logits.dtype != torch.bfloat16:
            reason = f"the drafter's logits processor returned {logits.dtype} logits"
        elif not transform.matches(draft_model):
            reason = STALE_TABLES
        elif not _defers_transform(type(worker)):
            reason = (
                f"the worker class {type(worker).__name__} overrides the draft head's "
                "refinement or selector"
            )
        else:
            draft_model.__dict__[DEFERRED_ATTR] = (logits.data_ptr(), logits.numel())
            return logits
        draft_model.__dict__[DECLINED_ATTR] = reason
        _warn_builtin(reason)
        return _builtin_transform(draft_model, logits)


def _owner(cls: type, name: str, after: Optional[type] = None) -> Optional[type]:
    """The class of ``cls``'s MRO (past ``after``, when given) that defines ``name``."""
    mro = cls.__mro__
    if after is not None:
        mro = mro[mro.index(after) + 1 :]
    return next((klass for klass in mro if name in vars(klass)), None)


#: ``_defers_transform`` by worker class.
_DEFERS: dict[type, bool] = {}


def _defers_transform(cls: type) -> bool:
    """Whether a worker of class ``cls`` takes deferred draft logits as this module expects.

    ``ShardedDraftHeadDFlashWorker._refine_block_logits`` must be the one the
    built-in forward calls, followed in the MRO by the built-in identity, and the
    ``_apply_dflash2_selector`` it reaches must be one that applies the
    transform after the top-k (a class marked ``_draft_transform_aware``).
    """
    known = _DEFERS.get(cls)
    if known is None:
        known = False
        if issubclass(cls, ShardedDraftHeadDFlashWorker):
            name = "_refine_block_logits"
            selector = _owner(cls, "_apply_dflash2_selector")
            known = (
                _owner(cls, name) is ShardedDraftHeadDFlashWorker
                and _owner(cls, name, after=ShardedDraftHeadDFlashWorker) is DFlashWorker
                and selector is not None
                and bool(vars(selector).get("_draft_transform_aware", False))
            )
        _DEFERS[cls] = known
    return known


class ShardedDraftHeadDFlashWorker(DFlashWorker):
    """The built-in DFlash worker with the draft head on vocab shards (fast path ``draft_head``)."""

    #: Its ``_apply_dflash2_selector`` applies a deferred drafter transform (``_defers_transform``).
    _draft_transform_aware = True

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

    def _shard_reason(self, gen_logits, spec_metadata, full_vocab) -> Optional[str]:
        """Why the sharded selector cannot take ``gen_logits`` on this step, or None."""
        if gen_logits.shape[-1] == full_vocab:
            return "full-vocab draft logits"
        if gen_logits.dtype != torch.bfloat16:
            return f"{gen_logits.dtype} draft logits"
        if (
            gen_logits.dim() != 3
            or gen_logits.stride(-1) != 1
            or gen_logits.shape[-1] * self.mapping.tp_size != full_vocab
        ):
            return "draft logits outside the [B, K, shard] layout"
        if spec_metadata.wants_advanced_draft_sampling:
            return "sampled drafts"
        if self._d2t is not None:
            return "a d2t draft vocab map"
        return None

    def _shard_selector_applies(self, gen_logits, spec_metadata, full_vocab) -> bool:
        return self._shard_reason(gen_logits, spec_metadata, full_vocab) is None

    def _draft_head_takes(self, draft_model, gen_logits, spec_metadata, full_vocab) -> bool:
        """Whether the sharded selector drafts this step.

        ``_shard_selector_applies``, and, with a transform installed, tables
        that still match ``draft_model``. A step it turns down with a transform
        installed takes the built-in path, where bf16 logits get
        ``_draft_head_transformed``, and warns once per reason.
        """
        reason = self._shard_reason(gen_logits, spec_metadata, full_vocab)
        transform = self.__dict__.get(TRANSFORM_ATTR)
        if transform is None:
            return reason is None
        if reason is None and not transform.matches(draft_model):
            reason = STALE_TABLES
        if reason is None:
            return True
        if gen_logits.dtype == torch.bfloat16:
            # Logits transformed upstream (fp32) were warned for where that happened.
            _warn_builtin(reason)
        return False

    def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
        """The drafter's logit transform, unless this step's sharded selector applies it.

        The deferring transform handed back the processor's bf16 shard
        untransformed (``DEFERRED_ATTR``). These logits stay bf16 only when they
        are that tensor (the reshape keeps its address and size), this step
        takes the sharded selector and the drafter still declares the
        transform the tables hold; the selector then owes the transform
        (``OWED_ATTR``). Any other bf16 logits get the drafter's current
        transform here, with a one-time warning per reason. Then the next
        class's refinement, the built-in identity.
        """
        deferred = draft_model.__dict__.pop(DEFERRED_ATTR, None)
        declined = draft_model.__dict__.pop(DECLINED_ATTR, None)
        self.__dict__[OWED_ATTR] = False
        transform = self.__dict__.get(TRANSFORM_ATTR)
        if transform is not None and gen_logits.dtype == torch.bfloat16:
            full_vocab = draft_model.candidate_selector.predecessor_codebook.shape[0]
            # A declined call has applied the drafter's current transform already,
            # which keeps bf16 logits only when that transform is the identity.
            reason = declined
            if reason is None and deferred != (gen_logits.data_ptr(), gen_logits.numel()):
                reason = NOT_DEFERRED
            if reason is None:
                reason = self._shard_reason(gen_logits, spec_metadata, full_vocab)
            if reason is None and not transform.matches(draft_model):
                reason = STALE_TABLES
            if reason is None:
                self.__dict__[OWED_ATTR] = True
            else:
                _warn_builtin(reason)
                gen_logits = self._draft_head_transformed(draft_model, gen_logits)
        return super()._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)

    def _draft_head_transform_for(self, draft_model, gen_logits) -> Optional[DraftLogitTransform]:
        """The transform the sharded selector applies to ``gen_logits`` (bf16 means untransformed).

        For a step ``_draft_head_takes`` admits, which checked that the tables
        still match ``draft_model``.
        """
        self.__dict__[OWED_ATTR] = False
        transform = self.__dict__.get(TRANSFORM_ATTR)
        if transform is None or gen_logits.dtype != torch.bfloat16:
            return None
        if not transform.matches(draft_model):
            raise DraftTransformStateError(
                "[musespark_open] draft_head: the sharded selector was asked for tables that no "
                f"longer hold ({STALE_TABLES})"
            )
        return transform

    def _draft_head_transformed(self, draft_model, gen_logits) -> torch.Tensor:
        """``gen_logits`` with the drafter's current transform applied once, for a built-in path.

        With a transform installed, bf16 logits are untransformed or were
        transformed by an identity: both get the drafter's class method with
        its current attributes, which is what the built-in worker computes.
        """
        self.__dict__[OWED_ATTR] = False
        if self.__dict__.get(TRANSFORM_ATTR) is None or gen_logits.dtype != torch.bfloat16:
            return gen_logits
        return _builtin_transform(draft_model, gen_logits)

    def _apply_dflash2_selector(
        self, draft_model, gen_logits, block_hidden_states, anchor_tokens, spec_metadata
    ):
        selector = draft_model.candidate_selector
        full_vocab = selector.predecessor_codebook.shape[0]
        takes = self._draft_head_takes(draft_model, gen_logits, spec_metadata, full_vocab)
        if gen_logits.shape[-1] == full_vocab:
            return super()._apply_dflash2_selector(
                draft_model,
                self._draft_head_transformed(draft_model, gen_logits),
                block_hidden_states,
                anchor_tokens,
                spec_metadata,
            )
        if takes:
            tokens = selector_draft_tokens(
                gen_logits,
                block_hidden_states,
                anchor_tokens,
                selector,
                self.mapping,
                self.__dict__.get("_draft_head_allreduce"),
                transform=self._draft_head_transform_for(draft_model, gen_logits),
            )
            stand_in = tokens.unsqueeze(-1).expand(*tokens.shape, full_vocab)
            setattr(stand_in, TOKENS_ATTR, tokens)
            return stand_in
        # The built-in path: the transform, the head's own all-gather and .float(),
        # then the built-in selector.
        full_logits = allgather(
            self._draft_head_transformed(draft_model, gen_logits), self.mapping, dim=-1
        ).float()
        return super()._apply_dflash2_selector(
            draft_model, full_logits, block_hidden_states, anchor_tokens, spec_metadata
        )

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        if self.__dict__.get(OWED_ATTR):
            self.__dict__[OWED_ATTR] = False
            raise DraftTransformStateError(
                "[musespark_open] draft_head: this step kept its draft logits untransformed for "
                "the sharded selector, but no selector applied the drafter's logit transform"
            )
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
    shadow = getattr(draft_model, "__dict__", {}).get(TRANSFORM_METHOD)
    if shadow is not None and not isinstance(shadow, _DeferredTransform):
        return f"the drafter's {TRANSFORM_METHOD} is replaced on the instance"
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


def transform_probe_logits(
    vocab: int, shard: int, transform: DraftLogitTransform, rows: int = PROBE_ROWS
) -> torch.Tensor:
    """Host bf16 ``[rows, vocab]`` logits whose top-k under ``transform`` is decided by its ties.

    Built from the transform's own tie groups (``logit_transform.tie_groups``),
    so no threshold is assumed: the largest group above +0.0 (``A < B`` its
    smallest and largest finite members) and the largest below -0.0 (``C <
    D``). Rows, in turn: ten ``A`` at low ids and ten ``B`` at high ids above a
    small grid; the same reversed; a row of the lower group's members with ten
    ``C`` low and ten ``D`` high; five +inf among fifteen finite members of the
    upper group; consecutive keys across the upper edge ``hi`` of the identity
    band; members of the upper group on every shard's first and last columns;
    a row of upper-group members only; consecutive keys across the lower edge
    ``lo`` above lower-group members. A transform without ties gets the
    largest finite values in their place, which tie nowhere.
    """
    g = torch.Generator().manual_seed(PROBE_SEED + 1)
    lowest, highest = FINITE_KEYS
    groups = _logit_transform.tie_groups(transform)

    def largest(candidates, fallback):
        return max(candidates, key=lambda grp: grp[1] - grp[0], default=fallback)

    upper = largest([grp for grp in groups if grp[0] >= 0x8000], (highest - 255, highest))
    lower = largest([grp for grp in groups if grp[1] <= 0x7FFF], (lowest, lowest + 255))

    def finite(group):
        return torch.arange(max(group[0], lowest + 1), min(group[1], highest - 1) + 1)

    def members(group, n, with_inf=False):
        keys = torch.arange(group[0], group[1] + 1) if with_inf else finite(group)
        return keys[torch.randint(0, keys.numel(), (n,), generator=g)]

    def grid(levels):
        return (torch.randint(0, levels, (vocab,), generator=g).float() * 0.25 - 3.0).to(
            torch.bfloat16
        )

    def place(row, columns, keys):
        row[columns] = _logit_transform.bf16_of_key16(keys)
        return row

    def ends(group):
        keys = finite(group)
        return int(keys[0]), int(keys[-1])

    low = torch.randperm(vocab // 4, generator=g)[:10]
    high = vocab - 1 - torch.randperm(vocab // 4, generator=g)[:10]
    a, b = ends(upper)
    c, d = ends(lower)
    ten = torch.ones(10, dtype=torch.int64)
    top = min(transform.hi, highest)
    bottom = max(transform.lo, lowest)
    upper_ladder = torch.arange(max(top - 23, 0x8000), min(top + 8, highest) + 1)
    lower_ladder = torch.arange(max(bottom - 8, lowest), min(bottom + 23, 0x7FFF) + 1)
    edges = torch.tensor(
        [c for r in range(vocab // shard) for c in (r * shard, (r + 1) * shard - 1)]
    )
    mixed = torch.randperm(vocab, generator=g)[:20]
    probe = [
        place(place(grid(8), low, a * ten), high, b * ten),
        place(place(grid(8), low, b * ten), high, a * ten),
        place(
            place(place(grid(8), torch.arange(vocab), members(lower, vocab, True)), low, c * ten),
            high,
            d * ten,
        ),
        place(place(grid(8), mixed[:5], torch.full((5,), highest)), mixed[5:], members(upper, 15)),
        place(grid(8), torch.randperm(vocab, generator=g)[: upper_ladder.numel()], upper_ladder),
        place(grid(4), edges, members(upper, edges.numel())),
        place(grid(8), torch.arange(vocab), members(upper, vocab, True)),
        place(
            place(grid(8), torch.arange(vocab), members(lower, vocab, True)),
            torch.randperm(vocab, generator=g)[: lower_ladder.numel()],
            lower_ladder,
        ),
    ]
    return torch.stack([probe[r % len(probe)] for r in range(rows)])


def tie_order_probes(
    vocab: int, shard: int, transform: Optional[DraftLogitTransform], device
) -> list[torch.Tensor]:
    """The install's probe logits on ``device``, bf16 ``[rows, vocab]`` each.

    ``tie_order_probe_logits``; with a transform also ``transform_probe_logits``
    and, at ``WIDE_PROBE_ROWS`` rows, both repeated.
    """
    probes = [tie_order_probe_logits(vocab, shard).to(device)]
    if transform is not None:
        ties = transform_probe_logits(vocab, shard, transform).to(device)
        both = torch.cat([probes[0], ties])
        repeats = -(-WIDE_PROBE_ROWS // both.shape[0])
        probes += [ties, both.repeat(repeats, 1)[:WIDE_PROBE_ROWS].contiguous()]
    return probes


def probe_reference(full: torch.Tensor, top_k: int, transform, topk=torch.topk):
    """``topk`` as the built-in selector runs it on ``full`` (bf16 ``[rows, vocab]``).

    On ``[1, rows, vocab]`` fp32: ``full.float()``, or the drafter's transform of it.
    """
    rows, vocab = full.shape
    logits = full.float() if transform is None else transform.real(full)
    return topk(logits.view(1, rows, vocab), top_k, dim=-1)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of fp32 tensors, NaN payloads aside."""
    return _logit_transform.same_bits(a, b)


def tie_order_reason(
    selector: nn.Module,
    num_ranks: int,
    topk=torch.topk,
    transform: Optional[DraftLogitTransform] = None,
) -> str | None:
    """Why the sharded candidates would differ from this torch's ``torch.topk``, or None.

    The merge copies the tie order of ``torch.topk`` (selection, arrangement
    and bitonic network), which torch does not document. This runs ``topk``
    as the built-in selector does -- ``[1, rows, vocab]`` fp32 -- on
    ``tie_order_probes``, and the sharded path on the same logits split
    into ``num_ranks`` shards with the keys exchanged in both layouts
    (concatenated int32 keys and summed fp32 halves, which is what the
    all-gather and the all-reduce return), and compares ids and values bit for
    bit. With ``transform`` the reference transforms the logits first and the
    sharded path runs on canonical keys with the merge's values table. ``topk``
    is a parameter so a test can stand in another order.
    """
    predecessor_codebook = selector.predecessor_codebook
    vocab, rank_dim = predecessor_codebook.shape
    shard = vocab // num_ranks
    top_k = selector.top_k
    device = predecessor_codebook.device
    canon = canon_kwargs(transform)
    with torch.cuda.device(device):
        for probe, full in enumerate(tie_order_probes(vocab, shard, transform, device)):
            rows = full.shape[0]
            ref_values, ref_ids = probe_reference(full, top_k, transform, topk)
            shards = [full[:, r * shard : (r + 1) * shard].contiguous() for r in range(num_ranks)]
            layouts = {
                "int32 all-gather": torch.cat(
                    [draft_shard_topk(s, top_k, CHUNK, NUM_WARPS, **canon) for s in shards], dim=0
                ),
                "fp32 all-reduce": torch.stack(
                    [
                        draft_shard_topk(s, top_k, CHUNK, NUM_WARPS, r, num_ranks, **canon)
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
                    **values_kwargs(transform),
                )
                if not (torch.equal(ids, ref_ids) and _same_bits(values, ref_values)):
                    differ = (ids != ref_ids).any(dim=-1).view(-1).nonzero().view(-1).tolist()
                    return (
                        f"torch {torch.__version__} orders top-k ties differently from the "
                        f"merge ({layout} keys; probe {probe} rows {differ} differ)"
                    )
    return None


def agreed_reason(mapping, reason: str | None, transform: Optional[DraftLogitTransform]):
    """``reason`` as every rank of ``mapping``'s TP group agrees, tables compared.

    Collective over the TP group (``dispatch.agreed_union`` and
    ``dispatch.agreed_refusal``): every rank calls it at the same point of the
    install, whatever its own reason. Ranks whose transform tables differ
    (``logit_transform.digest``) refuse, since each rank's merge reads its own
    tables and differing ones would draft different tokens per rank.
    """
    from ... import dispatch as _tp_agreement

    mine = _logit_transform.digest(transform)
    seen = _tp_agreement.agreed_union(mapping, [mine])
    if reason is None and seen != [mine]:
        reason = f"the TP ranks differ in their drafter logit transform ({'; '.join(seen)})"
    return _tp_agreement.agreed_refusal(mapping, reason)


def agreed_refusal(mapping, reason: str | None) -> str | None:
    """``reason`` as every rank of ``mapping``'s TP group agrees (``dispatch.agreed_refusal``).

    Collective over the TP group: every rank calls it at the same point of the
    install, whatever its own reason.
    """
    from ... import dispatch as _tp_agreement

    return _tp_agreement.agreed_refusal(mapping, reason)


def installed(llm: nn.Module) -> bool:
    """Whether ``install`` has put the draft head on ``llm``'s DFlash worker."""
    return isinstance(getattr(llm, "spec_worker", None), ShardedDraftHeadDFlashWorker)


def install(llm: nn.Module, make_allreduce=key_allreduce) -> bool:
    """Install the draft head on ``llm``'s DFlash worker and drafter; return True.

    ``make_allreduce(mapping)`` builds the key exchange (collective: every rank
    installs at the same point of the load). A second call on an installed
    ``llm`` changes nothing. Everything is built and checked before anything
    changes, and each refusal is agreed over the TP group (``agreed_reason``
    for the checks, ``agreed_refusal`` for the key exchange ``make_allreduce``
    built), so a refusing install leaves the worker and the drafter as they
    were on every rank.

    Raises:
        ValueError: when there is no DFlash 2 drafter (no spec worker, or a
            drafter without a candidate selector), or its head layout is not one
            the feature supports, or its logit transform is not one the draft
            head can apply after the top-k (``logit_transform.unsupported_reason``),
            or this torch's ``torch.topk`` orders ties differently from the merge
            (``tie_order_reason``), or the key exchange's MNNVL all-reduce could
            not be built on some rank -- running the built-in path, or a path
            whose drafts differ, under the fast path's name would misreport what
            runs.
    """
    if installed(llm):
        return True
    worker = getattr(llm, "spec_worker", None)
    draft_model = getattr(llm, "draft_model", None)
    if worker is None or getattr(draft_model, "candidate_selector", None) is None:
        raise ValueError(
            "[musespark_open] draft_head cannot run here: there is no DFlash 2 drafter"
        )
    selector = draft_model.candidate_selector
    transform = None
    reason = unsupported_reason(worker, draft_model)
    if reason is None:
        transform = _logit_transform.build(draft_model, selector.predecessor_codebook.device)
        if transform is not None:
            reason = _logit_transform.unsupported_reason(
                transform, draft_model.lm_head.out_features
            )
    if reason is None:
        reason = tie_order_reason(selector, worker.mapping.tp_size, transform=transform)
    reason = agreed_reason(worker.mapping, reason, transform)
    if reason is not None:
        raise ValueError(f"[musespark_open] draft_head cannot run here: {reason}")
    if "_draft_head_allreduce" in worker.__dict__:
        allreduce = worker.__dict__["_draft_head_allreduce"]
    else:
        allreduce = make_allreduce(worker.mapping)
        if allreduce is None:
            # The composition's precondition checked the group takes an MNNVL all-reduce.
            reason = "the MNNVL FP32 all-reduce of the key exchange could not be built"
    # Building the all-reduce can fail on some ranks only; every rank refuses alike.
    reason = agreed_refusal(worker.mapping, reason)
    if reason is not None:
        raise ValueError(f"[musespark_open] draft_head cannot run here: {reason}")
    worker.__class__ = ShardedDraftHeadDFlashWorker
    if type(draft_model.logits_processor) is LogitsProcessor:
        draft_model.logits_processor = ShardDraftLogitsProcessor(draft_model.logits_processor)
    # Not registered as a submodule: it owns no parameters, only a fabric workspace.
    worker.__dict__["_draft_head_allreduce"] = allreduce
    if transform is not None:
        # Tables built outside capture, on the drafter's device; plain attributes, not buffers.
        worker.__dict__[TRANSFORM_ATTR] = transform
        draft_model.__dict__[TRANSFORM_METHOD] = _DeferredTransform(worker, draft_model)
    exchange = "MNNVL all-reduce" if allreduce else "NCCL all-gather"
    applied = (
        ""
        if transform is None
        else (
            f", drafter logit transform applied after the top-k (canonical keys outside "
            f"[{transform.lo:#06x}, {transform.hi:#06x}])"
        )
    )
    logger.info(
        "[musespark_open] draft_head: DFlash 2 draft head on vocab shards "
        f"(shard {draft_model.lm_head.out_features}, tp {worker.mapping.tp_size}, "
        f"top_k {draft_model.candidate_selector.top_k}, keys by {exchange}, "
        f"tie order checked against torch {torch.__version__}{applied})"
    )
    return True
