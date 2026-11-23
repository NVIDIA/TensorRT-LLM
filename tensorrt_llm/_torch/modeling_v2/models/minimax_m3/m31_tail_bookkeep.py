# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared one-request tail bookkeeping of the DSpark worker.

TP8 retains its specialized FOLD_F3 producer; this mixin changes no TP8 path.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN
from tensorrt_llm.logger import logger

# ``SpecMetadata.tail_bookkeep``: the per-token buffers a single decode request's rows are read from, with the dtype
# ``populate_sampling_params_for_one_model`` / ``_populate_request_rng_state`` allocate them with.
_TAIL_BOOKKEEP_ROW_BUFFERS = (
    ("temperatures", torch.float32),
    ("top_ks", torch.int32),
    ("top_ps", torch.float32),
    ("min_ps", torch.float32),
    ("seeds", torch.int64),
    ("offsets", torch.int64),
)
# The tail bookkeeping items that already logged their first use.
_TAIL_BOOKKEEP_LOGGED: set = set()
_TAIL_BOOKKEEP_USES: dict = {}


def tail_bookkeep_log_once(item: str, message: str) -> None:
    """Log a tail bookkeeping item's engagement line once per process, and its use count at every power of two.

    Host only, so a captured step logs it while it is captured: the count is the item's host executions (eager steps
    and graph captures), not graph replays.
    """
    uses = _TAIL_BOOKKEEP_USES[item] = _TAIL_BOOKKEEP_USES.get(item, 0) + 1
    if item not in _TAIL_BOOKKEEP_LOGGED:
        _TAIL_BOOKKEEP_LOGGED.add(item)
        logger.info(f"[M3 host] tail bookkeeping {item}: {message}")
    elif uses & (uses - 1) == 0:
        logger.info(f"[M3 host] tail bookkeeping {item}: used {uses} times")


def tail_bookkeep_draft_len(spec_metadata, num_contexts: int, batch_size: Optional[int]) -> int:
    """The draft length when one request owns the per-token rows, else 0.

    ``K`` when this step's one request owns rows ``[0, K]`` of the per-token sampling and Philox buffers, and the
    metadata enables tail bookkeeping (``spec_metadata.tail_bookkeep``; 0 when the runtime does not set it).

    ``SpecMetadata._scan_one_model_sampling`` lays the per-token buffers out as ``[ctx: 1 row, gen: 1 +
    runtime_draft_len rows]`` per request, and ``populate_sampling_params_for_one_model`` /
    ``_populate_request_rng_state`` fill a generation request's rows with its per-request values: the same normalized
    temperature / top-k / top-p / min-p as ``request_*``, its seed, and offsets ``request_offsets + r *
    _RNG_SLOT_SPAN`` for row ``r``. With tail bookkeeping on, a decode-only step of one request (no tree, no ragged
    verification, the static draft length) can read its draft block's parameters and Philox state from those rows
    instead of expanding the per-request buffers.

    Host state and shapes only (no device read), so it is fixed per captured graph.
    """
    if not getattr(spec_metadata, "tail_bookkeep", False):
        return 0
    if num_contexts != 0 or batch_size != 1:
        return 0
    if (
        spec_metadata.is_spec_dec_tree
        or spec_metadata.is_spec_dec_dynamic_tree
        or spec_metadata.is_ragged_verify
    ):
        return 0
    draft_len = spec_metadata.max_draft_len
    if draft_len < 1 or spec_metadata.runtime_draft_len != draft_len:
        return 0
    for name, dtype in _TAIL_BOOKKEEP_ROW_BUFFERS:
        buf = getattr(spec_metadata, name, None)
        if (
            not isinstance(buf, torch.Tensor)
            or buf.dim() != 1
            or buf.dtype != dtype
            or buf.numel() < draft_len + 1
            or not buf.is_contiguous()
        ):
            return 0
    for name in ("request_seeds", "request_offsets"):
        buf = getattr(spec_metadata, name, None)
        if (
            not isinstance(buf, torch.Tensor)
            or buf.dim() != 1
            or buf.dtype != torch.int64
            or buf.numel() < 1
        ):
            return 0
    return draft_len


class TailBookkeepMixin:
    """Reuse the exact per-token RNG-window guards and their one-forward lifetime at TP2/TP4."""

    _tailbk_win = None

    def _forward_impl(self, *args, **kwargs):
        """The DSpark Markov / DFlash step; the per-token offsets window lives for one step only (set by its accept
        call, read by its draft block), so a step whose accept did not stash one recomputes it."""
        self._tailbk_win = None
        try:
            return super()._forward_impl(*args, **kwargs)
        finally:
            self._tailbk_win = None

    def _rng_state_per_request(self, spec_metadata, start=0, end=None, repeat=1, *, slot):
        """TAIL_BOOKKEEP (per-token RNG window): at one decode request, the accept and draft-block Philox (seed, offset)
        rows read from the per-token buffers (one int64 add per step instead of eight launches), else the in-tree
        expansion.

        ``_populate_request_rng_state`` writes row r (0..K) of the request's per-token window as
        ``seeds[r] = seed`` and ``offsets[r] = o + r * SPAN`` (o = ``request_offsets[0]``; the host sum fits int64,
        or its upload would have raised), uploaded with ``request_seeds`` / ``request_offsets`` on every sampled step.
        The in-tree rows are ``(o + slot * SPAN) + k * SPAN``: draft row k at slot K + 1 is ``offsets[k] + (K + 1) *
        SPAN`` and the accept row at slot 2K + 1 is ``offsets[K] + (K + 1) * SPAN``. Integer addition is associative
        and commutative mod 2^64, so the window ``offsets[0:K + 1] + (K + 1) * SPAN`` holds exactly the in-tree
        offsets, even where they wrap, and the seeds are the request's seed by construction. The returned tensors are
        contiguous int64 views with one entry per row, as the in-tree ones. Preconditions are host state and shapes
        (``tail_bookkeep_draft_len``), so the choice is fixed per captured graph."""
        state = self._tailbk_rng_window(spec_metadata, start, end, repeat, slot)
        if state is not None:
            return state
        return super()._rng_state_per_request(spec_metadata, start, end, repeat, slot=slot)

    def _tailbk_rng_window(self, spec_metadata, start, end, repeat: int, slot: int):
        """The per-token window's (seed, offset) rows, or None (the in-tree expansion) outside its preconditions."""
        k = tail_bookkeep_draft_len(spec_metadata, start, end)
        if k == 0:
            return None
        first = spec_metadata.rng_draft_slot(0)
        if first != k + 1 or spec_metadata.rng_accept_slot != first + k:
            return None
        accept = repeat == 1 and slot == spec_metadata.rng_accept_slot
        block = repeat == k and slot == first
        if not (accept or block):
            return None
        stash = self._tailbk_win
        if block and stash is not None and stash[0] is spec_metadata:
            win = stash[1]
        else:
            win = spec_metadata.offsets[: k + 1] + first * _RNG_SLOT_SPAN
        tail_bookkeep_log_once(
            "R1", "Philox seeds / offsets from the per-token window (one add per step)"
        )
        if accept:
            self._tailbk_win = (spec_metadata, win)
            return spec_metadata.request_seeds[0:1], win[k : k + 1]
        self._tailbk_win = None
        return spec_metadata.seeds[:k], win[:k]

    def _zero_padding_rows(
        self,
        logits,
        spec_metadata,
        num_contexts: int,
        batch_size: int,
        rows_per_request: int,
        *,
        packed_verify: bool = False,
    ):
        """TAILBK Z1: no padding-row zeroing while a batch-1 decode graph is captured, else the in-tree zeroing.

        Why the in-tree result is a bitwise copy of ``logits`` on every replay of such a graph:
        (i) a batch-1 graph replays exactly one real request. ``CUDAGraphRunner._get_padded_batch`` only appends
            dummies after the batch's own requests, up to the graph's size, and the executor forwards no empty batch
            without attention DP (``_can_queue``: batch_size > 0), so a size-1 graph never holds a dummy. Under
            attention DP an empty rank does run a dummy, so attention DP is excluded.
        (ii) a real request's slot is its ``py_seq_slot`` in ``[0, slot_capacity)`` and the padding marker is
            ``dummy_slot_row == slot_capacity`` (``prepare_rejection_sampling_buffers``), so the padding mask is
            all-false and the in-tree ``masked_fill`` copies ``logits`` unchanged.
        (iii) every consumer is out-of-place (the non-fused probs: the softmax / top-k mask ops allocate their outputs,
            top-p reads probs; this target installs no softmax override), so returning ``logits`` itself instead of
            that copy exposes no write to the caller's verify logits. The fused modes are not audited and keep the
            copy.
        (iv) the consumers see the layout the copy would have: the skip needs contiguous logits at a 256-byte aligned
            address (the copy is a fresh contiguous allocation, which the caching allocator aligns at least that much;
            the verify logits here are the A1 all-gather's or the NCCL path's fresh output), so a consumer that
            dispatches on layout takes the same path.
        Eager steps (the graph warmup's dummy included) keep the in-tree zeroing; so do padded graphs (2..8), which
        can hold dummies, and ragged verification (its layout check). Host state only, fixed per captured graph."""
        if self._tailbk_skips_padding_rows(logits, spec_metadata, num_contexts, batch_size):
            tail_bookkeep_log_once("Z1", "padding-row zeroing skipped in captured batch-1 graphs")
            return logits
        return super()._zero_padding_rows(
            logits,
            spec_metadata,
            num_contexts,
            batch_size,
            rows_per_request,
            packed_verify=packed_verify,
        )

    def _tailbk_skips_padding_rows(
        self, logits, spec_metadata, num_contexts: int, batch_size: int
    ) -> bool:
        """Z1's exact condition: tail bookkeeping on, one generation request and no context rows (num_gens == 1),
        no ragged verification, a non-fused sampling mode, no attention DP, contiguous 256-byte aligned logits and a
        capturing stream."""
        if not getattr(spec_metadata, "tail_bookkeep", False):
            return False
        if not logits.is_contiguous() or logits.data_ptr() % 256 != 0:
            return False
        if num_contexts != 0 or batch_size != 1 or getattr(spec_metadata, "is_ragged_verify", True):
            return False
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        if mode is None or mode.is_fused:
            return False
        mapping = getattr(self, "mapping", None)
        if mapping is None or getattr(mapping, "enable_attention_dp", True):
            return False
        return torch.cuda.is_current_stream_capturing()
