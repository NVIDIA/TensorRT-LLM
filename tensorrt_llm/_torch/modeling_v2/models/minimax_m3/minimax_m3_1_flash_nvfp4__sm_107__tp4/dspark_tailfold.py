# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILFOLD: three bitwise folds of the TP4 batch-1 DSpark step tail, each with its own kill switch.

``M3DSparkWorker`` (dspark.py) wires them; every one leaves every byte the step writes as it was. All three serve only
the steps TAIL_FORK forks (``M3DSparkWorker._tf_arm``: decode-only, not ragged, n x (K + 1) rows within FC_SHARD's
MAX_ROWS = 8, so batch 1 at K7, the context bound to the draft pool); every other step runs the stock code.

TAILGLUE_A (``TRTLLM_M31_DISABLE_TAILGLUE_A=1``; needs the C++ op ``minimax_m31_tail_glue_a``)
    After the fork joins, the DRAFTER_CTX_KV_STORE_FUSED context store (``dspark_tailfork.ctx_store_launch``), the
    DRAFTER_CTX_ADVANCE_FUSED advance + read view (``ops/dspark/dspark_ctx_advance``, two launches with ADV_PAR) and,
    inside the drafter forward, the DRAFTER_FORWARD_INDICES_FUSED forward indices
    (``ops/dspark/dspark_forward_indices``) run as one launch (``kernels/m3_tail_glue_op``: the C++ op, bit for bit the
    Triton kernel ``kernels/m3_tail_glue`` that the tests keep as the oracle). The store hook defers its kernel
    (``TailGlue.store``), the advance hook launches it with the advance, read view and forward indices
    (``TailGlue.advance``), and the forward-indices hook returns those outputs when it is called for exactly the tensors
    they were computed from (``TailGlue.indices``), else the stock op. Nothing reads the cache, the context lengths or
    the read table between the deferred store and the launch (the stock order is store, advance, read view, then the
    drafter), and a store deferred but not launched is launched before the drafter inputs return (``flush``). The read
    view takes each request's context length before the advance from the store's first position of the request: the
    fused launch runs only when those positions are the ones this step's DRAFTER_INPUTS_FUSED launch wrote
    (``ctx_len[slot] + j``) from the very slots and context lengths the advance gets. A fused launch the op rejects
    leaves this rank on the stock ops for good (a warning; no collective is involved, and the stock ops write the same
    bytes); a build without the op never defers a store.

MEMCPY86K (``TRTLLM_M31_DISABLE_MEMCPY86K=1``)
    The drafter forward's first layer copies the noise block into its residual stream (``residual =
    hidden_states.clone()``, modeling_dflash.py): an 86,016-byte graph memcpy node at K7 b1 ([1, 7, 6144] bf16) right
    before the first RMSNorm. The DRAFTER_INPUTS_FUSED drafter-inputs launch writes that block into a fresh tensor that
    only the drafter forward reads, so on a forked step the worker names it to the drafter
    (``DFlashForCausalLM._dflash_owned_noise``) and the residual stream starts in it: layer 0 reads the block only
    through its input norm, before the residual's first write, so every value read is the copy's.

MEMCPY4B (default OFF: ``TRTLLM_M31_ENABLE_MEMCPY4B=1`` opts in; ``TRTLLM_M31_DISABLE_MEMCPY4B=1`` still wins)
    DFlash's ``_prepare_attn_metadata_for_dflash`` swaps in a copy of ``_seq_lens_cuda`` (``prepare_for_spec_dec``:
    a 4-byte graph memcpy node at b1) between the forced acceptance and the KV rewind, on the sampling chain, the
    critical one. On a forked step the same copy of the same tensor is taken first on the fork's stream (nothing writes
    ``_seq_lens_cuda`` during the forward: the setter's upload precedes the replay) and swapped in at the stock point,
    after an event wait on that copy, so every later reader on the main stream sees the copy's bytes. The saved-state
    dict ends as the stock one (the original host and device tensors, in the stock order). Off by default: the copy's
    event adds a full cross-branch edge into the sampling chain's KV rewind node (a PDL kernel), which costs the step
    more than the copy it moves; off, the stock prepare runs on every step.

Each item's engagement line is logged once per rank, the first time the item is taken while a CUDA graph is captured
(the replayed path; the eager warmup before the capture does not count). TAILGLUE_A and MEMCPY86K also log, once per
rank each, the two steps of theirs that line does not cover: the first forked step that runs eagerly after that
captured take (a step no captured graph served, e.g. a first decode step after a disaggregated KV-transfer admission
that does not replay; "... at an eager step after the capture: taken" or "...: stock path (<why>)"), and the first
forked step captured with the stock path ("... stock path while a CUDA graph is captured (<why>)"). A replayed step
runs no host code, so it logs nothing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .dspark_tailfork import StoreOperands, ctx_store_operands, view_key
from .kernels import m3_tail_glue_op
from .m31_switches import m31_disabled, m31_enabled

# Each item's switch, read once at import (the same on every rank and every step). TAILGLUE_A and MEMCPY86K are on by
# default; MEMCPY4B is off by default (opt in with TRTLLM_M31_ENABLE_MEMCPY4B=1; its kill switch still wins).
TAILGLUE_A = not m31_disabled("TAILGLUE_A")
MEMCPY86K = not m31_disabled("MEMCPY86K")
MEMCPY4B = m31_enabled("MEMCPY4B") and not m31_disabled("MEMCPY4B")

TAILGLUE_A_MAX_REQUESTS = (
    8  # capacity: fixed: the fork's batch-1 rows; MAX_FAST_BATCH bounds the drafter's batch
)


def _rank(worker) -> Optional[int]:
    return getattr(getattr(worker, "mapping", None), "tp_rank", None)


def capturing() -> bool:
    """Whether the current stream is capturing a CUDA graph (the engagement lines report the captured path)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ------------------------------------------------------------------------------------------------------- TAILGLUE_A


class StepInputs:
    """One DRAFTER_INPUTS_FUSED drafter-inputs call of this step: its outputs (bonus, query positions, ctx positions,
    noise block)
    and the slots and context lengths it read."""

    __slots__ = ("outs", "slots", "ctx_len")

    def __init__(self, outs: tuple, slots, ctx_len) -> None:
        self.outs = outs
        self.slots = slots
        self.ctx_len = ctx_len


class _Indices:
    """The forward indices one fused launch wrote, with the operands they were computed from."""

    __slots__ = ("num_key", "rows", "qpos", "block", "width", "page", "outs")


class TailGlue:
    """TAILGLUE_A's per-worker state: the deferred store, the forward indices of the last fused launch, this step's
    DRAFTER_INPUTS_FUSED call. ``stock_advance`` / ``stock_indices`` are the hooks the worker would install without
    it."""

    def __init__(self, stock_advance, stock_indices, rank=None) -> None:
        self.stock_advance = stock_advance
        self.stock_indices = stock_indices
        self.rank = rank
        self.pending: Optional[StoreOperands] = None
        self.idx: Optional[_Indices] = None
        self.inputs: Optional[StepInputs] = None  # this step's DRAFTER_INPUTS_FUSED call
        self.fused = 0  # fused launches (host count, also counted while captured)
        self.handed = 0  # forward indices handed back from a fused launch
        self.declined = 0  # advance calls that took the stock path with a deferred store
        self.logged = False
        self.logged_handed = False
        self.logged_eager = (
            False  # the first deferred store's step run eagerly after the captured take (logged)
        )
        self.logged_stock_capt = (
            False  # the first deferred store not fused while a graph is captured (logged)
        )
        self.last_decline: Optional[str] = None
        # why a fused launch failed or cannot run (then every call takes the stock ops)
        self.broken: Optional[str] = (
            None if m3_tail_glue_op.available() else "this build has no minimax_m31_tail_glue_a"
        )

    # The DRAFTER_CTX_KV_STORE_FUSED store hook's kernel call (TAIL_FORK_KV's forked branch): ctx_store_launch's
    # contract.
    def store(self, prep, draft_model, projected, positions, num_accepted, request_rows, cols, block_table, kv_buf,
              tokens_per_request, *, capacity_counts=None, max_col=None):  # fmt: skip
        """``ctx_store_launch`` with the kernel deferred to the advance call when the store derives its columns on a
        bound pool (the layout the fused launch reproduces); any other layout launches now."""
        ops = ctx_store_operands(
            prep,
            draft_model,
            projected,
            positions,
            num_accepted,
            request_rows,
            cols,
            block_table,
            kv_buf,
            tokens_per_request,
            capacity_counts=capacity_counts,
            max_col=max_col,
        )
        if not isinstance(ops, StoreOperands):
            return ops
        self.flush()  # an earlier deferred store of this step keeps its place before this one
        if ops.meta["COL_FROM_POS"] and ops.meta["BOUND"] and self.broken is None:
            self.pending = ops
        else:
            ops.launch()
        return True

    def flush(self) -> bool:
        """Launch a deferred store now (the stock kernel); whether one was pending."""
        p, self.pending = self.pending, None
        if p is None:
            return False
        p.launch()
        return True

    def _unfit(self, p: StoreOperands, slots, rows, num_accepted, ctx_len, block_counts, base_pages, tables,
               tables_read, page_size) -> Optional[str]:  # fmt: skip
        """Why the fused launch does not reproduce this advance after the store ``p`` (None: it does). Host state and
        shapes only (fixed per captured graph). The advance's own operand rules come first (advance_ctx_and_read_view
        declines the same layouts)."""
        if self.broken is not None:
            return f"the fused launch failed earlier ({self.broken})"
        n = slots.shape[0] if slots.dim() == 1 else -1
        if not (
            n >= 1
            and rows.shape == (n,)
            and num_accepted.shape == (n,)
            and num_accepted.dtype in (torch.int32, torch.int64)
            and ctx_len.dtype == torch.int64
            and block_counts.dtype == torch.int64
            and base_pages.dtype == torch.int64
            and tables.dtype == torch.int32
            and tables_read.dtype == torch.int32
            and tables.shape == tables_read.shape
            and tables.stride(1) == 1
            and tables_read.stride(1) == 1
            and page_size >= 1
        ):
            return "an advance layout the stock op declines or takes on its own"
        if n > TAILGLUE_A_MAX_REQUESTS:
            return f"{n} requests (more than {TAILGLUE_A_MAX_REQUESTS})"
        if not (
            p.num_accepted is num_accepted
            and p.request_rows is rows
            and p.counts is block_counts
            and p.block_table is tables
        ):
            return "the advance is not over the deferred store's requests, counts and table"
        if not (
            slots.dtype == torch.int64
            and slots.is_contiguous()
            and rows.dtype == torch.int64
            and rows.is_contiguous()
            and num_accepted.is_contiguous()
            and ctx_len.is_contiguous()
            and block_counts.is_contiguous()
            and base_pages.is_contiguous()
        ):
            return "non-int64 or non-contiguous request operands"
        kp1 = p.meta["KP1"]
        pos = p.positions
        if not (
            pos.dtype == torch.int64
            and pos.is_contiguous()
            and pos.shape == (n * kp1,)
            and p.grid[0] == n * kp1
            and p.kv_flat.shape[0] == n * kp1
        ):
            return "store rows or positions are not [n x (K + 1)] (int64 positions)"
        rec = self.inputs
        if rec is None or len(rec.outs) < 3:
            return "no DRAFTER_INPUTS_FUSED call this step"
        # The read view takes ctx_len[slot] before the advance from positions[r * (K + 1)]: exact only when the store's
        # positions are the ones DRAFTER_INPUTS_FUSED wrote (ctx_len[slot] + j), from the slots and context lengths of
        # this advance.
        if not (rec.slots is slots and rec.ctx_len is ctx_len):
            return (
                "the advance's slots or context lengths are not the ones DRAFTER_INPUTS_FUSED read"
            )
        q, cpos = rec.outs[1], rec.outs[2]
        if not (
            isinstance(cpos, torch.Tensor)
            and cpos.dim() == 2
            and cpos.shape == (n, kp1)
            and view_key(cpos.reshape(-1)) == view_key(pos)
        ):
            return "the store's positions are not DRAFTER_INPUTS_FUSED's context positions"
        if (
            not isinstance(q, torch.Tensor)
            or q.dim() != 2
            or q.shape[0] != n
            or q.dtype not in (torch.int32, torch.int64)
        ):
            return "no DRAFTER_INPUTS_FUSED query positions for these requests"
        return None

    # The DRAFTER_CTX_ADVANCE_FUSED advance hook (dflash.enable_fused_ctx_advance's contract).
    def advance(self, slots, rows, num_accepted, ctx_len, max_ctx, dummy_slot, block_counts, base_pages, tables,
                tables_read, page_size, block_size):  # fmt: skip
        # The deferred store stays pending until the fused launch has been issued: on any exception the drafter
        # inputs' flush launches it (the stock order, store then advance).
        p = self.pending
        why = (
            "no deferred store"
            if p is None
            else self._unfit(
                p,
                slots,
                rows,
                num_accepted,
                ctx_len,
                block_counts,
                base_pages,
                tables,
                tables_read,
                page_size,
            )
        )
        if why is None:
            q = self.inputs.outs[1]
            iblock = q.shape[1]
            try:
                out, outs = m3_tail_glue_op.launch(
                    p, slots, ctx_len, max_ctx, dummy_slot, base_pages, tables_read, page_size, block_size, q,
                    iblock, page_size,
                )  # fmt: skip
            # The op rejected an operand or the launch failed: nothing was enqueued.
            except RuntimeError as err:
                self.broken = f"{type(err).__name__}: {err}"[:240]
                why = "the fused launch failed"
                logger.warning(
                    f"M3.1 DSpark TAILGLUE_A: the fused launch failed ({self.broken}); this rank runs the stock store, "
                    f"advance and forward indices from now on (rank {self.rank})"
                )
        if why is not None:
            self.pending = None
            if p is not None:
                p.launch()  # the stock order: the store, then the advance
                self.declined += 1
                self.last_decline = why
                self._note(why)
            return self.stock_advance(
                slots, rows, num_accepted, ctx_len, max_ctx, dummy_slot, block_counts, base_pages, tables,
                tables_read, page_size, block_size,
            )  # fmt: skip
        self.pending = None
        c = _Indices()
        c.num_key = view_key(out)
        c.rows = rows
        c.qpos = q
        c.block = iblock
        c.width = tables.shape[1]
        c.page = int(page_size)
        c.outs = outs
        self.idx = c
        self.fused += 1
        if not self.logged and capturing():
            self.logged = True
            logger.info(
                f"[M3 dispatch] TAILGLUE_A taken: ctx store + context advance + read view + forward indices in one "
                f"launch ({rows.shape[0]} requests, width {tables.shape[1]}) on rank {self.rank}"
            )
        else:
            self._note(None)
        return out

    def _note(self, why: Optional[str]) -> None:
        """The two lines beside the captured take's (once per rank each) for a step with a deferred store: the first
        one run eagerly after the captured take, and the first one captured with the stock path. ``why`` is the
        reason the fused launch did not run, None when it ran."""
        capt = capturing()
        tally = f"fused {self.fused}, stock {self.declined}"
        if capt and why is not None and not self.logged_stock_capt:
            self.logged_stock_capt = True
            logger.info(
                f"[M3 dispatch] TAILGLUE_A stock path while a CUDA graph is captured ({why}; {tally}) on rank "
                f"{self.rank}"
            )
        elif not capt and self.logged and not self.logged_eager:
            self.logged_eager = True
            what = "taken" if why is None else f"stock path ({why})"
            logger.info(
                f"[M3 dispatch] TAILGLUE_A at an eager step after the capture: {what} ({tally}) on rank {self.rank}"
            )

    # The DRAFTER_FORWARD_INDICES_FUSED forward-indices hook (dflash.enable_fused_forward_indices' contract).
    def indices(self, num_ctx, batch_idx, query_positions, block_size, pages_per_slot, page_size):
        c, self.idx = self.idx, None
        if (
            c is not None
            and view_key(num_ctx) == c.num_key
            and batch_idx is c.rows
            and query_positions is c.qpos
            and int(block_size) == c.block
            and int(pages_per_slot) == c.width
            and int(page_size) == c.page
        ):
            self.handed += 1
            if not self.logged_handed and capturing():
                self.logged_handed = True
                logger.info(
                    f"[M3 dispatch] TAILGLUE_A indices handed back: the drafter's forward indices come from the fused "
                    f"launch, no forward-indices launch ({int(num_ctx.shape[0])} requests) on rank {self.rank}"
                )
            return c.outs
        return self.stock_indices(
            num_ctx, batch_idx, query_positions, block_size, pages_per_slot, page_size
        )

    def end_step(self) -> None:
        self.flush()
        self.idx = None
        self.inputs = None


def recording_inputs(worker, inputs_fn):
    """The DRAFTER_INPUTS_FUSED hook ``inputs_fn`` with this step's call kept on ``worker`` (a ``StepInputs``:
    TAILGLUE_A's query and context positions with the slots and context lengths they came from, MEMCPY86K's noise
    block), None when it declined."""

    def hook(*args, **kwargs):
        out = inputs_fn(*args, **kwargs)
        rec = None
        if out is not None:
            # The hook's contract: (embed_tokens, accepted, num_accepted, slots, ctx_len, max_ctx, ...).
            slots = args[3] if len(args) > 3 else kwargs.get("slots")
            ctx_len = args[4] if len(args) > 4 else kwargs.get("ctx_len")
            rec = StepInputs(tuple(out), slots, ctx_len)
        worker._tfold_inputs = rec
        tg = getattr(worker, "_tg", None)
        if tg is not None:
            tg.inputs = rec
        return out

    hook.inner = inputs_fn
    return hook


# -------------------------------------------------------------------------------------------------------- MEMCPY86K


def drafter_owns_noise_hook(draft_model) -> bool:
    """Whether the drafter's forward takes a worker-owned noise block (the modeling_dflash hook is present)."""
    return draft_model is not None and hasattr(draft_model, "_dflash_owned_noise")


def own_noise(draft_model, inputs, recorded: Optional[StepInputs]) -> bool:
    """MEMCPY86K: name the step's noise block to the drafter for its next forward when it is exactly the tensor this
    step's DRAFTER_INPUTS_FUSED launch wrote (fresh, read by nothing but that forward). Whether it was named."""
    if not isinstance(inputs, dict) or recorded is None or not drafter_owns_noise_hook(draft_model):
        return False
    noise = inputs.get("noise_embedding")
    if noise is None or len(recorded.outs) < 4 or noise is not recorded.outs[3]:
        return False
    draft_model._dflash_owned_noise = noise
    draft_model._dflash_owned_noise_taken = False
    return True


# --------------------------------------------------------------------------------------------------------- MEMCPY4B


class SeqLensSnapshot:
    """The fork's copy of ``_seq_lens_cuda`` (taken first on the fork's stream) and the event recorded after it."""

    __slots__ = ("attn", "src", "copy", "event")


def stock_two_field_prepare(spec_metadata) -> bool:
    """Whether DFlash's _prepare_attn_metadata_for_dflash takes its two-field branch (("_seq_lens", "_seq_lens_cuda")):
    every step but the CUDA-graph warmup, which also saves kv_lens_cuda."""
    return not (
        getattr(spec_metadata, "is_cuda_graph", False)
        and not torch.cuda.is_current_stream_capturing()
    )


def _stock_prepare(attn_metadata) -> bool:
    """Whether the metadata's prepare_for_spec_dec is the base class's (save the original, swap in a clone)."""
    fn = getattr(type(attn_metadata), "prepare_for_spec_dec", None)
    return getattr(fn, "__qualname__", "") == "AttentionMetadata.prepare_for_spec_dec"


def take_seq_lens_snapshot(attn_metadata, spec_metadata) -> Optional[SeqLensSnapshot]:
    """On the fork's stream (the current one), the copy prepare_for_spec_dec would take of ``_seq_lens_cuda`` at this
    step's two-field prepare, and an event after it; None where the stock prepare must run."""
    if attn_metadata is None or not stock_two_field_prepare(spec_metadata):
        return None
    src = getattr(attn_metadata, "_seq_lens_cuda", None)
    if not isinstance(src, torch.Tensor) or not src.is_cuda or not _stock_prepare(attn_metadata):
        return None
    snap = SeqLensSnapshot()
    snap.attn = attn_metadata
    snap.src = src
    snap.copy = src.clone()
    snap.event = torch.cuda.Event()
    snap.event.record()
    return snap


def install_seq_lens_snapshot(
    snap: Optional[SeqLensSnapshot], attn_metadata, spec_metadata
) -> bool:
    """The stock two-field prepare_for_spec_dec with the fork's copy swapped in for the device field (after an event
    wait on it); False (nothing done) where the stock prepare must run."""
    if (
        snap is None
        or snap.attn is not attn_metadata
        or getattr(attn_metadata, "_seq_lens_cuda", None) is not snap.src
        or not stock_two_field_prepare(spec_metadata)
        or getattr(attn_metadata, "_saved_tensors", None)
        or not _stock_prepare(attn_metadata)
    ):
        return False
    main = torch.cuda.current_stream()
    main.wait_event(snap.event)
    attn_metadata.prepare_for_spec_dec("_seq_lens")
    attn_metadata._saved_tensors["_seq_lens_cuda"] = snap.src
    attn_metadata._seq_lens_cuda = snap.copy
    if not torch.cuda.is_current_stream_capturing():
        snap.copy.record_stream(main)
    return True


def _memcpy4b_off_reason() -> str:
    """Why MEMCPY4B's switch leaves it off in this process ('' when the switch has it on)."""
    if m31_disabled("MEMCPY4B"):
        return " (TRTLLM_M31_DISABLE_MEMCPY4B set)"
    if not m31_enabled("MEMCPY4B"):
        return " (default off: opt in with TRTLLM_M31_ENABLE_MEMCPY4B=1)"
    return ""


def tailfold_setup_line(worker, glue: bool, owned: bool, fork4b: bool) -> Tuple[str, str, str]:
    """The three set-up lines (one per item) for this rank, in the worker's set-up line form '<item> on|off on rank N'
    (as DRAFTER_CTX_KV_STORE_FUSED..E, TAILBK_F and TAILGLUE_B); MEMCPY4B's off line ends with why its switch has it
    off."""
    rank = _rank(worker)
    return (
        f"M3.1 DSpark TAILGLUE_A: ctx store + context advance + read view + forward indices in one launch "
        f"{'on' if glue else 'off'} on rank {rank}",
        f"M3.1 DSpark MEMCPY86K: drafter residual stream starts in the DRAFTER_INPUTS_FUSED noise block "
        f"{'on' if owned else 'off'} on rank {rank}",
        f"M3.1 DSpark MEMCPY4B: seq_lens copy on the fork's stream {'on' if fork4b else 'off'} on rank {rank}"
        f"{'' if fork4b else _memcpy4b_off_reason()}",
    )


__all__ = [
    "MEMCPY4B",
    "MEMCPY86K",
    "TAILGLUE_A",
    "SeqLensSnapshot",
    "StepInputs",
    "TailGlue",
    "capturing",
    "drafter_owns_noise_hook",
    "install_seq_lens_snapshot",
    "own_noise",
    "recording_inputs",
    "stock_two_field_prepare",
    "tailfold_setup_line",
    "take_seq_lens_snapshot",
]
