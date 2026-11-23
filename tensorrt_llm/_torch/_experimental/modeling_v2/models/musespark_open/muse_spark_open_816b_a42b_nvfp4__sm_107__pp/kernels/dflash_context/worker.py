# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The last stage's DFlash worker on a context server: it stores the drafter context and never drafts.

A disaggregated context server serves context steps only. On those the built-in
``DFlashWorker`` samples the first token and writes the drafter context (fc +
hidden_norm over the captures, the fused drafter K/V with k_norm and RoPE, the paged append
into the draft layers, ``DFlashWorker._store_prefill_context``), and runs no drafter
forward. The engine still runs steps with generation rows on the server itself (the
CUDA-graph warm-up and the capture of a decode graph); there the built-in worker would run
the drafter forward, which reads the target's embedding (absent on the last stage) and the
drafter's attention, MLP and selector weights (freed by ``slim``).

``ContextServerDFlashWorker`` keeps the built-in step for steps without generation rows and
for a zero draft length, and gives a step with generation rows the placeholder a context
row gets: the target's verification of the step's tokens (the first token of each row),
zero draft tokens and one-hot draft-probability rows, with no drafter forward and no
drafter context write. ``install`` re-classes the stage's worker in place.

A step with generation rows writes no drafter context for any of its rows, its context
rows included: their draft-layer pages keep what the pool held. A disaggregated context
server schedules context-only requests, so only the engine's own steps (warm-up and graph
capture, whose requests are dummies) have generation rows. A request served in full on the
context server, co-scheduled with another one's generation step, would reach the
generation server with a stale drafter context: its tokens stay exact, its acceptance
drops.

A context step stores the drafter context with ``store.store_runs``: the built-in's
per-request bookkeeping and K/V ops, and one paged store launch for every request of the
step in place of the built-in's per-request, per-layer copies and appends (the same bytes in
the pages). A worker class that overrides the built-in store (``store_override``), or a
worker or drafter ``store.refusal`` refuses, keeps the built-in store.

With a drafter context window (``carry``: the step's plan keeps the rows of positions at or
after each request's ``draft_context_keep_from``), the drafter context write stores the kept
rows only (``_store_window_context``): the same slot bookkeeping as the built-in write on the
KV cache manager's paged draft layers, then fc + hidden_norm over the carried rows' captures
(the kept rows and the plan's lead rows), and the fused drafter K/V with k_norm and RoPE over
each request's carried rows and the paged store at the kept positions: one
``store.store_runs`` for the step's kept rows, or, where the built-in store is kept (stream
capture, ``store_override``, ``store.refusal``), per request the built-in calls on fewer rows.
The lead rows make every GEMM of the write run over as many rows as the whole write's or over
at least ``window.MIN_GEMM_ROWS`` rows, so each kept row comes out bit for bit the whole write's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from . import store as _store


class ContextServerDFlashWorker:
    """The context-server step of a DFlash worker; combined with the worker's class by
    ``worker_class`` (this class first)."""

    #: The stage's ``carry.CaptureCarry`` (``install``); None without one.
    _capture_carry: Optional[Any] = None

    def _store_window_context(
        self, draft_model, spec_metadata, attn_metadata, plan, captures, one_launch: bool = False
    ):
        """Store the drafter context of the kept rows of a step (``plan``, ``captures``).

        The slot bookkeeping is the built-in write's on a paged arena: a chunk that does not
        start where its request's last one ended takes a fresh slot, a reused prefix starts
        the slot at the chunk's first position, and an overflow drops the request's context.
        Each request's kept rows are written at their positions; the dropped leading rows
        advance the slot's length without a write. ``captures`` holds the plan's carried rows:
        each request's lead rows run through the K/V GEMM with its kept rows and are not
        stored. With ``one_launch`` (the caller checked ``store.refusal``) every request's kept
        rows go to ``store.store_runs`` as one run, else each request's carried rows go through
        ``precompute_context_kv`` and its kept rows through ``_store_context_kv_paged``.

        Raises:
            RuntimeError: when the drafter context is not paged (the only arena a window is
                planned for: the KV cache manager's draft layers).
        """
        if attn_metadata.num_ctx_tokens == 0:
            return
        if not self._ctx_paged:
            raise RuntimeError(
                "the drafter context window is written into paged draft layers only; this "
                "drafter keeps its context in a dense arena"
            )
        bound = self._ctx_block_tables is not None
        num_contexts = attn_metadata.num_contexts
        if len(plan.lengths) != num_contexts:
            raise RuntimeError(
                f"the drafter context window planned {len(plan.lengths)} context requests for "
                f"a step of {num_contexts}"
            )
        projected = draft_model.project_target_hidden(captures) if captures is not None else None
        request_ids = spec_metadata.request_ids[:num_contexts]
        ctx_alloc = self._ctx_alloc_tokens(request_ids) if bound else None
        compute_dtype = _store.compute_dtype(draft_model, self._ctx_kv_buf[0].dtype)
        updates: Dict[int, int] = {}
        runs: List[_store.Run] = []
        rows = zip(request_ids, plan.lengths, plan.starts, plan.skips, plan.leads, plan.offsets())
        for i, (req_id, length, first_pos, skip, lead, offset) in enumerate(rows):
            prev_pos = self._req_ctx_pos.get(req_id)
            if self._assign_slot(req_id, reset=prev_pos != first_pos, updates=updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                continue
            slot = self._req_to_slot[req_id]
            cur = updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                cur = first_pos
            if cur + length > cap:
                logger.warning(
                    f"DFlash: ctx overflow on slot {slot} ({cur} + {length} > {cap}); "
                    "skipping its context store, so this request drafts nothing."
                )
                self._req_to_slot.pop(req_id, None)
                self._req_ctx_pos.pop(req_id, None)
                updates[slot] = 0
                self._free_slots.append(slot)
                continue
            end = cur + length
            self._req_ctx_pos[req_id] = first_pos + length
            if length == 0:
                continue
            updates[slot] = end
            kept = length - skip
            if kept == 0:
                continue
            # Manager block tables are keyed by batch position, the private arena's by slot.
            row = i if bound else slot
            if one_launch:
                runs.append(
                    _store.Run(offset=offset, rows=kept, table_row=row, first=cur + skip, lead=lead)
                )
                continue
            chunk = projected[offset : offset + lead + kept].to(compute_dtype)
            positions = torch.arange(cur + skip - lead, end, dtype=torch.long, device=chunk.device)
            chunk_k, chunk_v = draft_model.precompute_context_kv(chunk, positions)
            self._store_context_kv_paged(
                chunk_k[lead:],
                chunk_v[lead:],
                torch.full((kept,), row, dtype=torch.long, device=chunk.device),
                positions[lead:],
            )
        if runs:
            _store.store_runs(self, draft_model, projected, runs)
        self._write_ctx_len(updates, {})

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        if num_gens == 0 or spec_metadata.runtime_draft_len == 0:
            return super()._forward_impl(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager,
            )
        return self._forward_without_draft(logits, attn_metadata, spec_metadata)

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata,
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """The drafter context write of a context step, with one paged store launch for the
        step's requests.

        A step received with a drafter context window (the stage's ``carry``) writes its kept
        rows only (``_store_window_context``); any other step writes every context row.
        Stream capture keeps the built-in calls: the store's table is host data of the step.
        So does a worker class that overrides the built-in store (``store_override``), or a
        ``store.refusal``. Each outcome (this store, or the reason for the built-in calls) is
        logged once per worker.
        """
        carry = self._capture_carry
        window = None
        if (
            carry is not None
            and hasattr(draft_model, "fc")
            and hasattr(draft_model, "hidden_norm")
            and getattr(spec_metadata, "captured_hidden_states", None) is not None
        ):
            window = carry.window_capture(spec_metadata.captured_hidden_states)
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            reason = "stream capture"
        else:
            reason = store_override(type(self)) or _store.refusal(self, draft_model)
        logged = getattr(self, "_page_store_logged", None)
        if logged is None:
            logged = self._page_store_logged = set()
        if reason not in logged:
            logged.add(reason)
            if reason is None:
                logger.info("DFlash context server: one-launch drafter context store")
            else:
                logger.info(f"DFlash context server: built-in drafter context store ({reason})")
        if window is not None:
            return self._store_window_context(
                draft_model, spec_metadata, attn_metadata, *window, one_launch=reason is None
            )
        if reason is not None:
            return super()._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return
        ctx_proj = draft_model.project_target_hidden(captured_hs[:num_ctx_tokens])
        runs, ctx_len_updates = self._context_runs(spec_metadata, attn_metadata, position_ids)
        _store.store_runs(self, draft_model, ctx_proj, runs)
        self._write_ctx_len(ctx_len_updates, {})

    def _context_runs(
        self, spec_metadata, attn_metadata, position_ids: torch.Tensor
    ) -> Tuple[List["_store.Run"], Dict[int, int]]:
        """The built-in store's per-request bookkeeping on a paged arena: the rows it stores
        and the slots' new lengths, with the same slot, position and length updates."""
        host_pos = spec_metadata.host_position_ids
        ctx_len_updates: Dict[int, int] = {}
        runs: List[_store.Run] = []
        offset = 0
        num_contexts = attn_metadata.num_contexts
        manager_tables = self._ctx_block_tables is not None
        ctx_alloc = (
            self._ctx_alloc_tokens(spec_metadata.request_ids[:num_contexts])
            if manager_tables
            else None
        )
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            if slen == 0:
                first_pos = 0
            elif host_pos is not None:
                first_pos = int(host_pos[offset])
            else:
                first_pos = int(position_ids[offset].item())
            # A chunk continues the stored context only when it starts where it ended.
            reset = self._req_ctx_pos.get(req_id) != first_pos
            if self._assign_slot(req_id, reset=reset, updates=ctx_len_updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                offset += slen
                continue
            slot = self._req_to_slot[req_id]
            cur = ctx_len_updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                # Reuse skipped [0, first_pos): its drafter K/V are in the pool already.
                cur = first_pos
            if cur + slen > cap:
                logger.warning(
                    f"DFlash: ctx overflow on slot {slot} ({cur} + {slen} > {cap}); skipping "
                    "its context store, so this request drafts nothing."
                )
                self._req_to_slot.pop(req_id, None)
                self._req_ctx_pos.pop(req_id, None)
                ctx_len_updates[slot] = 0
                self._free_slots.append(slot)
                offset += slen
                continue
            self._req_ctx_pos[req_id] = first_pos + slen
            if slen > 0:
                ctx_len_updates[slot] = cur + slen
                runs.append(
                    _store.Run(
                        offset=offset,
                        rows=slen,
                        table_row=i if manager_tables else slot,
                        first=cur,
                    )
                )
            offset += slen
        return runs, ctx_len_updates

    def _forward_without_draft(self, logits, attn_metadata, spec_metadata) -> Dict[str, Any]:
        """A step with generation rows: verify, then a context row's placeholder drafts."""
        batch_size = attn_metadata.num_seqs
        draft_len = spec_metadata.runtime_draft_len
        self._execute_guided_decoder_if_present(logits)
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )
        self.write_context_onehot_draft_probs(spec_metadata, batch_size, 0, draft_len)
        next_draft_tokens = torch.zeros(
            (batch_size, draft_len), dtype=torch.int32, device=logits.device
        )
        self._rollback_guided_decoder_after_verify(num_accepted_tokens)
        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )
        return {
            "logits": logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }


#: The built-in store methods ``ContextServerDFlashWorker._store_prefill_context`` stands in for.
_STORE_METHODS = ("_store_prefill_context", "_store_context_kv_paged")


def store_override(cls: type, worker_base: Optional[type] = None) -> Optional[str]:
    """Why a ``cls`` worker keeps the built-in store: a class of its chain other than
    ``ContextServerDFlashWorker`` overrides one of the built-in store methods (None when none
    does). ``worker_base`` is the built-in ``DFlashWorker`` (imported when None)."""
    if worker_base is None:
        from tensorrt_llm._torch.speculative.dflash import DFlashWorker as worker_base
    for name in _STORE_METHODS:
        owner = next(
            (
                klass
                for klass in cls.__mro__
                if klass is not ContextServerDFlashWorker and name in vars(klass)
            ),
            None,
        )
        if owner is not worker_base:
            holder = "no class" if owner is None else owner.__name__
            return f"{holder} defines {name}"
    return None


#: ``ContextServerDFlashWorker`` combined with a worker class, by that class.
_COMPOSED: Dict[type, type] = {}


def worker_class(cls: type) -> type:
    """The class of a ``cls`` worker on a context server: ``ContextServerDFlashWorker``
    first, then ``cls``. One class per ``cls``."""
    if issubclass(cls, ContextServerDFlashWorker):
        return cls
    composed = _COMPOSED.get(cls)
    if composed is None:
        composed = type(
            f"ContextServer{cls.__name__}",
            (ContextServerDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` on a context server: no draft step.",
            },
        )
        _COMPOSED[cls] = composed
    return composed


def refusal(spec_worker: Any, worker_base: Optional[type] = None) -> Optional[str]:
    """Why ``spec_worker`` cannot be re-classed (None when it can); changes nothing.

    ``worker_base`` is the built-in ``DFlashWorker`` (imported when None).
    """
    if worker_base is None:
        from tensorrt_llm._torch.speculative.dflash import DFlashWorker as worker_base
    if spec_worker is None:
        return "the last stage has no speculative-decoding worker"
    if not isinstance(spec_worker, worker_base):
        return f"the spec worker is {type(spec_worker).__name__}, not a {worker_base.__name__}"
    return None


def install(spec_worker: Any, capture_carry: Optional[Any] = None) -> None:
    """Re-class ``spec_worker`` (a ``DFlashWorker``) as its context-server class, which reads
    the drafter context window of ``capture_carry`` (the last stage's ``CaptureCarry``).
    Idempotent.

    Call after ``refusal`` returned None.
    """
    spec_worker.__class__ = worker_class(type(spec_worker))
    spec_worker._capture_carry = capture_carry
