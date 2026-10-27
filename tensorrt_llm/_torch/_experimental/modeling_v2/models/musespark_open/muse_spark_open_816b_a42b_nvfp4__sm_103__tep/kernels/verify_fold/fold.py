# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast paths ``verify_fold`` and ``path_walk_tail``: the DFlash step's integer bookkeeping folded into its kernels.

A decode step of the one-engine DFlash worker (``DFlashWorker._forward_impl``)
spends about 25 launches on a handful of integers around two kernels this
target already owns:

* after the acceptance kernel (``dflash_accept``, ``fused_accept``), 18 launches that
  derive from the accepted counts and the worker's buffers: the saved copy of
  ``_seq_lens_cuda``, the KV-length rewind amount and ``kv_lens_cuda += 1``,
  the bonus token, the context lengths, the drafter's query and context
  positions, the ``[bonus, mask]`` ids it embeds;
* after the draft head's path walk (``draft_path_walk``, ``draft_head``), 7 launches:
  the KV-length rewind and the next step's input tokens.

``verify_fold`` replaces ``dflash_accept`` by ``dflash_verify_fold``, which computes
the counts as ``dflash_accept`` does and, in the same program after each count
exists, writes every one of those values; the worker methods that would have
computed them take them from it. ``path_walk_tail`` replaces ``draft_path_walk`` by
``draft_path_walk_tail``, which also writes the rewind and the next input
tokens. Neither moves any work ahead of the value it depends on: in particular
no accepted count is read before the acceptance kernel has produced it, under
forced acceptance or not. Every tensor the built-in path writes is written
with the same bits (graph inputs of the next step -- ``kv_lens_cuda``, the
forced-acceptance counter -- in place, at the same addresses); the
temporaries live at other addresses of the same graph pool.

With ``replicated_embedding`` (``replicated_embedding/embedding.py``) also selected, the drafter's
noise block (the bonus token's row, then the mask token's) is one gather from
the replicated table instead of the embedding call, an ``expand(...).clone()``
and a row copy.

Scope: generation-only steps (every decode step the target captures),
captured or eager, but not the eager forwards the CUDA-graph runner makes
before a capture (there the built-in clones ``kv_lens_cuda`` before the
``+= 1``, so the fold's in-place bump would land on the wrong tensor); those,
and every step with context rows, run the built-in methods.

The worker is re-classed once more, after ``draft_head`` and the drafter worker
fast paths (``spec_worker.py``): one combined class on every rank. The built-in methods
it writes out are ``DFlashWorker.prepare_1st_drafter_inputs``,
``DFlashWorker._prepare_attn_metadata_for_dflash``,
``DFlashWorker._prepare_kv_for_draft_forward`` and
``AttentionMetadata.prepare_for_spec_dec`` (``verify_fold``); the walk's tail hands back
``DFlashWorker._apply_kv_rewind_after_draft`` and
``SpecWorkerBase._prepare_next_new_tokens`` (``path_walk_tail``); the per-step state
follows the call order of ``DFlashWorker._forward_impl``.

``prepare_1st_drafter_inputs`` is written out because its bonus part is
inline; it stores and commits this step's context with the worker's own stages
(``_store_step_context``, ``_commit_context``). With the context-K/V fork
(``context_kv_fork/fork.py``, installed before this module, so its class sits
behind this one) a folded step whose store the fork's side chain ran skips the
store: ``_fold_forked_inputs`` launches the fork's commit right away -- the fold
kernel was the main stream's last read of ``_ctx_len`` -- builds the noise block
meanwhile and joins the side stream before returning; a step this class does
not fold reaches the fork's own prepare through ``super()``.

Once the fold has run in a step it has applied ``kv_lens_cuda += 1`` and owes
the rewind, so the methods that consume it (``FOLD_CONSUMERS``) never hand the
step back to their built-in versions, which would apply both again: each one
checks that it was given this step's attention metadata and the fold's own
output tensors, and raises ``FoldStateError`` otherwise; a step whose fold was
not taken by all of them raises too. A worker class composed on top of this
one must pass the accepted tokens and counts through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.logger import logger

from ... import spec_worker as _drafter_worker
from ...ops.embedding.replicated_gather import replicated_embedding_gather
from ...ops.spec.dflash_accept import FORCE_FRACTIONAL
from ...ops.spec.dflash_verify_fold import MAX_REQUESTS, VerifyFold, dflash_verify_fold
from ...ops.spec.draft_path_walk_tail import draft_path_walk_tail
from ..draft_head import head as _draft_head
from ..replicated_embedding import embedding as _replicated_embedding

#: The drafter block width the fold writes positions for.
MAX_BLOCK = 64

#: Worker methods this module overrides; a worker class other than the
#: built-in, ``draft_head``'s and ``fused_accept``'s that overrides one of them is refused.
OVERRIDDEN = (
    "_forward_impl",
    "sample_and_accept_draft_tokens",
    "_sample_and_accept_draft_tokens_base",
    "_prepare_attn_metadata_for_dflash",
    "_prepare_kv_for_draft_forward",
    "prepare_1st_drafter_inputs",
    "_apply_dflash2_selector",
    "_apply_kv_rewind_after_draft",
    "_prepare_next_new_tokens",
)

#: The worker methods that take a step's fold in place of their built-in work;
#: once the fold has run, each of them must take it.
FOLD_CONSUMERS = (
    "_prepare_attn_metadata_for_dflash",
    "_prepare_kv_for_draft_forward",
    "prepare_1st_drafter_inputs",
)


class FoldStateError(RuntimeError):
    """A step's fold (or tail walk) has applied state that a built-in method would apply again."""


@dataclass
class _Tail:
    """What ``draft_path_walk_tail`` wrote this step."""

    tokens: torch.Tensor
    next_tokens: torch.Tensor
    rewound: bool
    batch_indices: torch.Tensor


@dataclass
class _Step:
    """The per-forward state the overridden methods hand to each other."""

    attn_metadata: object
    spec_metadata: object
    draft_model: object
    warmup: bool
    fold: Optional[VerifyFold] = None
    #: The tensor the fold copied as ``_seq_lens_cuda``'s saved copy.
    seq_lens_source: Optional[torch.Tensor] = None
    accepted: Optional[torch.Tensor] = None
    num_accepted: Optional[torch.Tensor] = None
    tail: Optional[_Tail] = None
    #: The ``FOLD_CONSUMERS`` that have taken ``fold``.
    consumed: set = field(default_factory=set)


_FORWARD_ARGS = (
    "input_ids",
    "position_ids",
    "hidden_states",
    "logits",
    "attn_metadata",
    "spec_metadata",
    "draft_model",
)


def _forward_arg(name: str, args: tuple, kwargs: dict):
    if name in kwargs:
        return kwargs[name]
    index = _FORWARD_ARGS.index(name)
    return args[index] if index < len(args) else None


class SpecLoopFoldDFlashWorker(DFlashWorker):
    """The DFlash worker with the step's bookkeeping folded into its kernels (``verify_fold``, ``path_walk_tail``).

    Never constructed: ``install`` re-classes the worker as a class derived
    from this one and the worker's current class, and sets which folds run
    (``_fold_verify``: ``verify_fold``; ``_fold_path_walk_tail``: ``path_walk_tail``).
    """

    _fold_verify: bool = False
    _fold_path_walk_tail: bool = False

    # ------------------------------------------------------------------
    # The per-step state
    # ------------------------------------------------------------------
    def _forward_impl(self, *args, **kwargs):
        spec_metadata = _forward_arg("spec_metadata", args, kwargs)
        step = _Step(
            attn_metadata=_forward_arg("attn_metadata", args, kwargs),
            spec_metadata=spec_metadata,
            draft_model=_forward_arg("draft_model", args, kwargs),
            warmup=bool(getattr(spec_metadata, "is_cuda_graph", False))
            and not torch.cuda.is_current_stream_capturing(),
        )
        self.__dict__["_fold_step"] = step
        try:
            outputs = super()._forward_impl(*args, **kwargs)
        finally:
            self.__dict__["_fold_step"] = None
        if step.fold is not None and step.consumed != set(FOLD_CONSUMERS):
            missing = ", ".join(name for name in FOLD_CONSUMERS if name not in step.consumed)
            raise FoldStateError(
                f"[musespark_open] verify_fold: the step's fold was not taken by {missing}; the step "
                "does not follow DFlashWorker._forward_impl's order that the fold relies on"
            )
        return outputs

    def _fold_current(self) -> Optional[_Step]:
        step = self.__dict__.get("_fold_step")
        return None if step is None or step.warmup else step

    def _fold_folded(self) -> Optional[_Step]:
        """The current step if its fold has run (``kv_lens_cuda`` bumped, rewind owed), else None."""
        step = self._fold_current()
        return None if step is None or step.fold is None else step

    @staticmethod
    def _fold_refuse(method: str, got: str) -> FoldStateError:
        return FoldStateError(
            f"[musespark_open] verify_fold: {method} got {got} after this step's fold had run; the fold "
            "has already applied kv_lens_cuda += 1 and owes the rewind, so the built-in method "
            "cannot take the step over (it would apply both again). A worker class composed with "
            "the folds must hand the fold's accepted tokens and counts and the step's attention metadata "
            "through unchanged"
        )

    def _fold_mask_token_id(self, draft_model) -> int:
        """The built-in ``prepare_1st_drafter_inputs``' mask-token resolution, cached the same way."""
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        return self._resolved_mask_token_id

    # ------------------------------------------------------------------
    # verify_fold: acceptance + bookkeeping in one launch
    # ------------------------------------------------------------------
    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        accepted, num_accepted = super().sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )
        step = self._fold_current()
        if step is not None:
            step.accepted, step.num_accepted = accepted, num_accepted
        return accepted, num_accepted

    def _verify_fold_applies(
        self, step, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ) -> bool:
        if step is None or not self._fold_verify:
            return False
        attn_metadata = step.attn_metadata
        num_gens = batch_size - num_contexts
        seq_lens = getattr(attn_metadata, "_seq_lens_cuda", None)
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        return (
            num_contexts == 0
            and spec_metadata is not None
            and not spec_metadata.is_ragged_verify
            and logits.is_cuda
            and draft_tokens.dim() == 2
            and draft_tokens.shape[0] == num_gens
            and 1 <= num_gens <= MAX_REQUESTS
            and 1 <= draft_tokens.shape[1] <= 63
            and draft_tokens.stride(-1) == 1
            and not draft_tokens.dtype.is_floating_point
            and logits.shape[0] == num_gens * (draft_tokens.shape[1] + 1)
            and getattr(attn_metadata, "num_contexts", None) == 0
            and type(attn_metadata).prepare_for_spec_dec is AttentionMetadata.prepare_for_spec_dec
            and self._batch_to_slot is not None
            and self._ctx_len is not None
            and self._compute_block_size is not None
            and 1 <= self._compute_block_size <= MAX_BLOCK
            and isinstance(seq_lens, torch.Tensor)
            and seq_lens.is_cuda
            and seq_lens.is_contiguous()
            and (
                not hasattr(attn_metadata, "kv_lens_cuda")
                or (
                    isinstance(kv_lens, torch.Tensor)
                    and kv_lens.dtype == torch.int32
                    and kv_lens.is_contiguous()
                    and kv_lens.numel() >= num_gens
                )
            )
        )

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        step = self._fold_current()
        if not self._verify_fold_applies(
            step, logits, draft_tokens, num_contexts, batch_size, spec_metadata
        ):
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        attn_metadata = step.attn_metadata
        runtime_draft_len = draft_tokens.shape[1]
        # fused_accept's statements up to its dflash_accept call.
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        mode, base_total, max_total, frac = self._force_plan(runtime_draft_len, spec_metadata)
        pool = counter = None
        if mode == FORCE_FRACTIONAL:
            self._ensure_force_accept_rng_state(logits.device)
            pool, counter = self._force_accept_rng_pool, self._force_accept_rng_counter
        seq_lens = attn_metadata._seq_lens_cuda
        bumps_kv_lens = hasattr(attn_metadata, "kv_lens_cuda")
        fold = dflash_verify_fold(
            target_tokens.reshape(-1),
            draft_tokens,
            mode,
            base_total,
            max_total,
            frac,
            pool,
            counter,
            self._batch_to_slot,
            self._ctx_len,
            self._max_ctx,
            self._compute_block_size,
            int(self._fold_mask_token_id(step.draft_model)),
            attn_metadata.kv_lens_cuda if bumps_kv_lens else None,
            seq_lens,
        )
        step.fold, step.seq_lens_source = fold, seq_lens
        if bumps_kv_lens:
            # The fold has applied kv_lens_cuda[0:batch_size] += 1: the rewind is
            # owed from here on (the built-in sets these with its += 1), so a
            # forward that fails before the draft still gets it from
            # _ensure_spec_dec_state_restored.
            self._kv_rewind_amount = fold.rewind
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
        return fold.accepted, fold.num_accepted

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        method = "_prepare_attn_metadata_for_dflash"
        step = self._fold_folded()
        if step is None:
            return super()._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        if attn_metadata is not step.attn_metadata:
            raise self._fold_refuse(method, "another attention metadata than the step's")
        if attn_metadata._seq_lens_cuda is not step.seq_lens_source:
            raise self._fold_refuse(method, "a _seq_lens_cuda other than the one the fold copied")
        # The built-in's branch outside the eager warmup,
        # prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda"), with the clone of
        # _seq_lens_cuda the fold already took.
        attn_metadata.prepare_for_spec_dec("_seq_lens")
        attn_metadata._saved_tensors["_seq_lens_cuda"] = attn_metadata._seq_lens_cuda
        attn_metadata._seq_lens_cuda = step.fold.seq_lens_copy
        step.consumed.add(method)

    def _prepare_kv_for_draft_forward(
        self, attn_metadata, num_accepted_tokens, num_contexts, batch_size
    ):
        method = "_prepare_kv_for_draft_forward"
        step = self._fold_folded()
        if step is None:
            return super()._prepare_kv_for_draft_forward(
                attn_metadata, num_accepted_tokens, num_contexts, batch_size
            )
        fold = step.fold
        if attn_metadata is not step.attn_metadata:
            raise self._fold_refuse(method, "another attention metadata than the step's")
        if num_accepted_tokens is not fold.num_accepted:
            raise self._fold_refuse(method, "accepted counts other than the fold's tensor")
        if (num_contexts, batch_size) != (0, fold.accepted.shape[0]):
            raise self._fold_refuse(
                method,
                f"rows [{num_contexts}, {batch_size}), not the fold's [0, {fold.accepted.shape[0]})",
            )
        if hasattr(attn_metadata, "kv_lens_cuda"):
            # kv_lens_cuda[num_contexts:batch_size] += 1 and the rewind state were
            # applied with the fold (_sample_and_accept_draft_tokens_base).
            attn_metadata.update_for_spec_dec()
        step.consumed.add(method)

    def _fold_noise_block(self, embed_tokens, fold: VerifyFold, num_gens: int, block_size: int):
        """The drafter's ``[g, block, hidden]`` noise embedding: the bonus row, then mask rows."""
        table = _replicated_embedding.table_of(embed_tokens)
        if table is not None and fold.noise_ids.numel() <= _replicated_embedding.MAX_ROWS:
            # replicated_embedding: the built-in's embedding call + expand().clone() + row copy, in one gather.
            return replicated_embedding_gather(table, fold.noise_ids)
        combined_embed = embed_tokens(fold.embed_ids)
        embed_bonus = combined_embed[:num_gens]
        embed_mask = combined_embed[num_gens]
        noise_embed_2d = embed_mask.expand(num_gens, block_size, -1).clone()
        noise_embed_2d[:, 0, :] = embed_bonus
        return noise_embed_2d

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens=0,
    ):
        """The built-in method on a step whose fold has run, from the fold; otherwise the built-in."""
        method = "prepare_1st_drafter_inputs"
        step = self._fold_folded()
        if step is None:
            return super().prepare_1st_drafter_inputs(
                input_ids,
                position_ids,
                hidden_states,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                total_target_tokens,
            )
        fold = step.fold
        if attn_metadata is not step.attn_metadata:
            raise self._fold_refuse(method, "another attention metadata than the step's")
        if accepted_tokens is not fold.accepted:
            raise self._fold_refuse(method, "accepted tokens other than the fold's tensor")
        if num_accepted_tokens is not fold.num_accepted:
            raise self._fold_refuse(method, "accepted counts other than the fold's tensor")
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts
        if num_contexts != 0 or num_gens != fold.accepted.shape[0]:
            raise self._fold_refuse(
                method,
                f"{num_contexts} context and {num_gens} generation rows, not the fold's "
                f"0 and {fold.accepted.shape[0]}",
            )
        step.consumed.add(method)
        self._fold_mask_token_id(draft_model)
        embed_tokens = draft_model.draft_model_full.model.embed_tokens

        gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
        total_tokens_per_req = self._draft_tokens_per_req  # K+1
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        has_target_features = (
            captured_hs is not None
            and hasattr(draft_model, "fc")
            and hasattr(draft_model, "hidden_norm")
        )
        block_size = self._compute_block_size
        slots = self._batch_to_slot[num_contexts : num_contexts + num_gens]
        # From the fold: gen_rows_out, bonus, the positions and offsets.
        gen_rows_out = fold.gen_rows
        bonus = fold.bonus
        fork = self.__dict__.get("_kv_fork")
        launched = fork.take() if fork is not None else None
        if launched is not None:
            return self._fold_forked_inputs(
                fork, launched, fold, embed_tokens, attn_metadata, spec_metadata, slots,
                gen_num_accepted, total_tokens_per_req, block_size, has_target_features,
            )  # fmt: skip
        noise_embed_2d = self._fold_noise_block(embed_tokens, fold, num_gens, block_size)

        if has_target_features:
            # Target now processes exactly K+1 tokens per gen req, so the
            # captured slice is already the full set we need to project.
            gen_hs = captured_hs[: num_gens * total_tokens_per_req]
            num_accepted_long = fold.num_accepted_long
            self._store_step_context(
                draft_model, gen_hs, slots, gen_rows_out, self._ctx_len[slots], num_accepted_long
            )
            num_ctx_per_req_t = self._commit_context(slots, num_accepted_long, gen_rows_out)
        else:
            num_ctx_per_req_t = self._drafter_context_lengths(slots, gen_rows_out)

        attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        return self._drafter_inputs(
            noise_embed_2d, fold.query_positions, num_ctx_per_req_t, slots, gen_rows_out, bonus
        )

    def _fold_forked_inputs(
        self,
        fork,
        launched,
        fold: VerifyFold,
        embed_tokens,
        attn_metadata,
        spec_metadata,
        slots,
        gen_num_accepted,
        total_tokens_per_req,
        block_size,
        has_target_features,
    ):
        """The drafter inputs of a folded step whose context store the context-K/V fork's side chain ran.

        ``context_kv_fork`` forked the store of this step's ``K + 1`` captured rows
        onto its side stream, unmasked, from the context lengths before this step.
        The fold kernel was the main stream's last read of ``_ctx_len`` in this step
        (the query and context positions are its outputs), so the fork's commit is
        launched first, the noise block runs on the main stream meanwhile, and the
        main stream joins the side stream before returning, then settles a windowed
        arena that ``drafter_window_compact`` compacts (``_settle_context``).
        """
        num_gens = fold.accepted.shape[0]
        if (
            launched.num_gens != num_gens
            or launched.k != spec_metadata.runtime_draft_len
            or self._ctx_block_tables is not None
            or not has_target_features
        ):
            raise FoldStateError(
                "[musespark_open] verify_fold + context_kv_fork: the context-K/V side chain was launched for "
                f"{launched.num_gens} requests and K {launched.k}, the folded step has "
                f"{num_gens} and K {spec_metadata.runtime_draft_len}"
                + ("; the arena is paged" if self._ctx_block_tables is not None else "")
                + ("; the step has no target features" if not has_target_features else "")
            )
        num_ctx = fork.commit(launched, slots, gen_num_accepted)
        noise_embed_2d = self._fold_noise_block(embed_tokens, fold, num_gens, block_size)
        fork.join(launched)
        if num_ctx is None:
            num_ctx = self._settle_context(slots, fold.gen_rows)
        attn_metadata._seq_lens_cuda[0:num_gens] = total_tokens_per_req
        attn_metadata._seq_lens[0:num_gens] = total_tokens_per_req
        return self._drafter_inputs(
            noise_embed_2d, fold.query_positions, num_ctx, slots, fold.gen_rows, fold.bonus
        )

    # ------------------------------------------------------------------
    # path_walk_tail: the step's tail in the path walk
    # ------------------------------------------------------------------
    def _path_walk_tail_args(self, step, gen_logits, spec_metadata):
        """``(kv_lens, rewind, batch_indices)`` for the tail walk, or None when it does not apply."""
        if step is None or not self._fold_path_walk_tail or step.accepted is None:
            return None
        attn_metadata = step.attn_metadata
        requests, positions = gen_logits.shape[0], gen_logits.shape[1]
        accepted, num_accepted = step.accepted, step.num_accepted
        batch_indices = getattr(spec_metadata, "batch_indices_cuda", None)
        if not (
            getattr(attn_metadata, "num_contexts", None) == 0
            and getattr(attn_metadata, "num_seqs", None) == requests
            and isinstance(batch_indices, torch.Tensor)
            and batch_indices.dtype == torch.int32
            and batch_indices.numel() >= requests
            and accepted.dtype == torch.int32
            and accepted.dim() == 2
            and tuple(accepted.shape) == (requests, positions + 1)
            and accepted.stride(-1) == 1
            and num_accepted.dtype == torch.int32
            and num_accepted.numel() == requests
        ):
            return None
        # The built-in _apply_kv_rewind_after_draft's condition outside the warmup.
        rewinds = (
            self._kv_rewind_amount is not None
            and hasattr(attn_metadata, "kv_lens_cuda")
            and self._kv_rewind_pending
        )
        if rewinds:
            kv_lens, rewind = attn_metadata.kv_lens_cuda, self._kv_rewind_amount
            if not (
                self._kv_rewind_nc == 0
                and self._kv_rewind_bs == requests
                and isinstance(kv_lens, torch.Tensor)
                and kv_lens.dtype == torch.int32
                and kv_lens.is_contiguous()
                and kv_lens.numel() >= requests
                and rewind.dtype == torch.int32
                and rewind.numel() == requests
                and rewind.is_contiguous()
            ):
                return None
            return kv_lens, rewind, batch_indices
        return None, None, batch_indices

    def _apply_dflash2_selector(
        self, draft_model, gen_logits, block_hidden_states, anchor_tokens, spec_metadata
    ):
        step = self._fold_current()
        selector = draft_model.candidate_selector
        full_vocab = selector.predecessor_codebook.shape[0]
        args = None
        if (
            self._fold_path_walk_tail
            and gen_logits.shape[-1] != full_vocab
            and self._shard_selector_applies(gen_logits, spec_metadata, full_vocab)
        ):
            args = self._path_walk_tail_args(step, gen_logits, spec_metadata)
        if args is None:
            return super()._apply_dflash2_selector(
                draft_model, gen_logits, block_hidden_states, anchor_tokens, spec_metadata
            )
        kv_lens, rewind, batch_indices = args
        accepted, num_accepted = step.accepted, step.num_accepted

        def walk(edges, unary, ids):
            return draft_path_walk_tail(
                edges, unary, ids, accepted, num_accepted, batch_indices, kv_lens, rewind
            )

        tokens, next_tokens = _draft_head.selector_draft_tokens(
            gen_logits,
            block_hidden_states,
            anchor_tokens,
            selector,
            self.mapping,
            self.__dict__.get("_draft_head_allreduce"),
            walk=walk,
        )
        step.tail = _Tail(tokens, next_tokens, kv_lens is not None, batch_indices)
        if kv_lens is not None:
            # The walk has rewound kv_lens_cuda: nothing is owed any more, also
            # not to _ensure_spec_dec_state_restored if the step fails from here.
            self._kv_rewind_pending = False
        stand_in = tokens.unsqueeze(-1).expand(*tokens.shape, full_vocab)
        setattr(stand_in, _draft_head.TOKENS_ATTR, tokens)
        return stand_in

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        step = self._fold_current()
        if step is not None and step.tail is not None and step.tail.rewound:
            if attn_metadata is not step.attn_metadata:
                # The built-in would subtract the rewind a second time.
                raise FoldStateError(
                    "[musespark_open] path_walk_tail: _apply_kv_rewind_after_draft got another attention "
                    "metadata than the step's after the draft-head walk had applied the rewind"
                )
            # Applied by the tail walk, which ran after every reader of the
            # pre-rewind kv_lens_cuda in this step.
            self._kv_rewind_pending = False
            return
        return super()._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

    # A tail that does not match the arguments falls back to the built-in, which
    # recomputes the next input tokens from them: a pure function, no state to
    # apply twice.
    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        step = self._fold_current()
        tail = None if step is None else step.tail
        if (
            tail is not None
            and next_draft_tokens is tail.tokens
            and accepted_tokens is step.accepted
            and num_accepted_tokens is step.num_accepted
            and batch_indices_cuda is tail.batch_indices
            and batch_size == tail.tokens.shape[0]
        ):
            return tail.next_tokens
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )


#: ``SpecLoopFoldDFlashWorker`` combined with another feature's worker class, by that class.
_COMPOSED_WORKERS: dict[type, type] = {}


def _is_context_kv_fork(owner: type) -> bool:
    """Whether ``owner`` is the context-K/V fork's ``ContextKvForkDFlashWorker``.

    Admitted below this class in the MRO: its ``sample_and_accept_draft_tokens``
    fires the verify fork point and calls the next body; its
    ``prepare_1st_drafter_inputs`` runs only on steps this class's prepare
    hands to ``super()`` (no fold), and on a folded step this class's prepare
    takes the fork's side chain itself (``_fold_forked_inputs``).
    """
    if owner.__name__ != "ContextKvForkDFlashWorker":
        return False
    from ..context_kv_fork import fork as context_kv_fork

    return owner is context_kv_fork.ContextKvForkDFlashWorker


def unsupported_reason(
    spec_worker, verify_fold: bool = True, path_walk_tail: bool = True
) -> Optional[str]:
    """Why the folds asked for cannot be installed on ``spec_worker``, or None."""
    if not isinstance(spec_worker, DFlashWorker):
        return f"the spec worker is {type(spec_worker).__name__}, not a DFlashWorker"
    if verify_fold and not isinstance(spec_worker, _drafter_worker.FusedAcceptDFlashWorker):
        return "verify_fold extends the fused acceptance (fused_accept), and the worker does not run it"
    if path_walk_tail and not isinstance(spec_worker, _draft_head.ShardedDraftHeadDFlashWorker):
        return "path_walk_tail extends the draft-head path walk (draft_head), and the worker does not run it"
    cls = type(spec_worker)
    allowed = {
        DFlashWorker,
        SpecWorkerBase,
        SpecLoopFoldDFlashWorker,
        _drafter_worker.FusedAcceptDFlashWorker,
        _draft_head.ShardedDraftHeadDFlashWorker,
    }
    for name in OVERRIDDEN:
        owner = next((klass for klass in cls.__mro__ if name in vars(klass)), None)
        if owner is not None and owner not in allowed and not _is_context_kv_fork(owner):
            return f"the spec worker class {cls.__name__} overrides {name} in {owner.__name__}"
        if name in vars(spec_worker):
            return f"{name} is already replaced on this worker instance"
    return None


def install(spec_worker, verify_fold: bool, path_walk_tail: bool) -> bool:
    """Re-class the DFlash worker for the verify fold (``verify_fold``) and the path-walk tail
    (``path_walk_tail``) asked for; return whether anything changed.

    Idempotent; the shell calls it from every drafter load hook, after
    ``draft_head.install`` and ``drafter_worker.install``.

    Raises:
        ValueError: if a fold asked for cannot be installed (``unsupported_reason``).
    """
    selected = [
        name
        for name, on in (("path_walk_tail", path_walk_tail), ("verify_fold", verify_fold))
        if on
    ]
    if not selected:
        return False
    if spec_worker is None:
        raise ValueError("[musespark_open] the DFlash step folds cannot run here: no spec worker")
    reason = unsupported_reason(spec_worker, verify_fold, path_walk_tail)
    if reason is not None:
        raise ValueError(f"[musespark_open] {','.join(selected)} cannot run here: {reason}")
    if isinstance(spec_worker, SpecLoopFoldDFlashWorker):
        changed = (spec_worker._fold_verify, spec_worker._fold_path_walk_tail) != (
            verify_fold,
            path_walk_tail,
        )
        spec_worker._fold_verify = verify_fold
        spec_worker._fold_path_walk_tail = path_walk_tail
        return changed
    cls = type(spec_worker)
    composed = _COMPOSED_WORKERS.get(cls)
    if composed is None:
        composed = type(
            f"SpecLoopFold{cls.__name__}",
            (SpecLoopFoldDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` with ``SpecLoopFoldDFlashWorker``'s folds.",
            },
        )
        _COMPOSED_WORKERS[cls] = composed
    spec_worker.__class__ = composed
    spec_worker._fold_verify = verify_fold
    spec_worker._fold_path_walk_tail = path_walk_tail
    sites = [
        name
        for on, name in (
            (verify_fold, "the acceptance kernel"),
            (path_walk_tail, "the draft-head walk"),
        )
        if on
    ]
    logger.info(
        f"[musespark_open] {','.join(selected)}: DFlash step bookkeeping folded into "
        f"{' and '.join(sites)} (worker class {type(spec_worker).__name__})"
    )
    return True
