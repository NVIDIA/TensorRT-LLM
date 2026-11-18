# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in compact store of the rejection-sampling draft distributions.

One-model rejection sampling keeps every slot's draft distributions in a dense
``draft_probs [slots + 1, max_draft_len, vocab]`` FP32 store that lives for the
whole process, and gathers the generation requests' rows into a second dense
``[requests, draft_len, vocab]`` tensor at every verify. With a DFlash 2
drafter (candidate selector) every row the store ever holds has at most
``selector_top_k`` nonzero entries: the sampled path writes the walk's
``top_k`` candidate probabilities over a zeroed row, and the context and
padding placeholders write a one-hot row at token 0.

``TRTLLM_RS_COMPACT_DRAFT_PROBS=1`` (default off) keeps those rows as
``LANES`` (token id, probability) records per slot and position instead
(``CompactDraftProbs``, under 1 MB at 97 slots) and rebuilds the dense rows of
the verified requests only at the verify (``DeferredDraftProbs``): a zero fill
and one scatter of the recorded values. The rebuilt rows hold the dense store's
values bit for bit (the same FP32 words at the same columns, zeros elsewhere),
so FlashInfer's chain speculative sampling reads the same inputs and accepts
the same tokens: the output is bitwise unchanged. Lanes beyond a record's real entries repeat
its first lane, so the scatter writes the same word twice at most and stays
deterministic.

When the target logits handed to the verify are private to it (a materializer
marked them with ``PRIVATE_VERIFY_LOGITS_ATTR``: nothing reads them once the
target probabilities exist), the rebuilt rows are written into the logits'
own storage after the target probabilities are computed, so the verify adds no
``[requests, draft_len, vocab]`` allocation to the step's CUDA-graph pool.

The knob is read per process and is meant to be set on the generation server
only. It applies to one-model DFlash with a draft vocabulary equal to the
target's; anything else keeps the dense store (``compact_store_refusal``).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

#: The environment knob (``1`` / ``on`` / ``true`` / ``yes`` turn it on; default off).
ENV = "TRTLLM_RS_COMPACT_DRAFT_PROBS"
#: Records per slot and position. A writer with more entries per row raises.
LANES = 64
#: Attribute a materializer sets on FP32 target logits nothing reads after the verify's
#: target probabilities: the verify may then rebuild the draft rows in their storage.
PRIVATE_VERIFY_LOGITS_ATTR = "_trtllm_rs_private_verify_logits"

_TRUE = frozenset({"1", "on", "true", "yes"})


def compact_draft_probs_requested() -> bool:
    """Whether ``TRTLLM_RS_COMPACT_DRAFT_PROBS`` asks for the compact store (default off)."""
    return os.environ.get(ENV, "0").strip().lower() in _TRUE


def compact_store_refusal(spec_dec_mode, vocab_size: int, draft_vocab_size: int) -> Optional[str]:
    """Why the compact store cannot replace the dense one for this metadata, or None."""
    is_dflash = getattr(spec_dec_mode, "is_dflash", None)
    if is_dflash is None or not is_dflash():
        return "the speculative mode is not one-model DFlash"
    if vocab_size <= 0:
        return "the vocabulary size is unknown"
    if draft_vocab_size not in (0, vocab_size):
        return f"the draft vocabulary ({draft_vocab_size}) differs from the target's ({vocab_size})"
    return None


def mark_private_verify_logits(logits: torch.Tensor) -> torch.Tensor:
    """Mark ``logits`` as private to the verify that receives them; returns ``logits``."""
    setattr(logits, PRIVATE_VERIFY_LOGITS_ATTR, True)
    return logits


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def private_verify_scratch(logits: torch.Tensor, *live: torch.Tensor) -> Optional[torch.Tensor]:
    """``logits`` when the verify may overwrite it, else None.

    Requires the private mark, contiguous FP32 storage, and no storage shared with
    any tensor in ``live`` (the target probabilities the verify still reads).
    Host-side checks only: CUDA-graph capture safe.
    """
    if not getattr(logits, PRIVATE_VERIFY_LOGITS_ATTR, False):
        return None
    if logits.dtype != torch.float32 or not logits.is_contiguous():
        return None
    if any(_same_storage(logits, t) for t in live):
        return None
    return logits


class CompactDraftProbs:
    """Slot-indexed draft distributions as ``LANES`` (token id, probability) records.

    ``ids`` / ``vals``: ``[rows, max_draft_len, lanes]`` int64 / FP32, indexed by the
    request's ``py_seq_slot`` like the dense store. A record stands for the dense row
    that is zero over ``[0, vocab)`` except ``row[ids[l]] = vals[l]``.
    """

    def __init__(
        self, num_rows: int, max_draft_len: int, vocab_size: int, device, lanes: int = LANES
    ):
        if num_rows <= 0 or max_draft_len <= 0 or vocab_size <= 0 or lanes <= 0:
            raise ValueError(
                f"compact draft probs: bad shape rows={num_rows}, "
                f"draft_len={max_draft_len}, vocab={vocab_size}, lanes={lanes}"
            )
        self.vocab_size = int(vocab_size)
        self.lanes = int(lanes)
        # Zero records stand for zero rows (the dense store starts uninitialized; no
        # live request reads a row before a writer filled it).
        self.ids = torch.zeros((num_rows, max_draft_len, lanes), dtype=torch.int64, device=device)
        self.vals = torch.zeros(
            (num_rows, max_draft_len, lanes), dtype=torch.float32, device=device
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """The dense store's shape this store stands in for."""
        return (self.ids.shape[0], self.ids.shape[1], self.vocab_size)

    @property
    def device(self) -> torch.device:
        return self.ids.device

    def nbytes(self) -> int:
        return (
            self.ids.numel() * self.ids.element_size()
            + self.vals.numel() * self.vals.element_size()
        )

    def _check_vocab(self, vocab: int, what: str) -> None:
        if vocab != self.vocab_size:
            raise ValueError(
                f"[{ENV}] {what} writes {vocab}-wide rows; the compact store "
                f"holds {self.vocab_size}-wide rows"
            )

    def write_rows(
        self, slots: torch.Tensor, candidate_ids: torch.Tensor, proposal: torch.Tensor, vocab: int
    ) -> None:
        """The dense store's ``rows[slots, :n] = 0; rows[slots, t, ids[b, t]] = proposal[b, t]``.

        ``candidate_ids`` / ``proposal``: ``[B, n, k]`` with ``k <= lanes`` distinct ids
        per row (``torch.topk`` indices). Capture safe (index_put with device slots).
        """
        self._check_vocab(vocab, "a sampled draft")
        if candidate_ids.shape != proposal.shape or candidate_ids.dim() != 3:
            raise ValueError(
                f"[{ENV}] candidate ids {tuple(candidate_ids.shape)} and proposal "
                f"{tuple(proposal.shape)} must be the same [B, n, k]"
            )
        requests, steps, k = candidate_ids.shape
        if k > self.lanes:
            raise ValueError(f"[{ENV}] {k} entries per row exceed the store's {self.lanes} lanes")
        if steps > self.ids.shape[1]:
            raise ValueError(
                f"[{ENV}] {steps} draft positions exceed the store's {self.ids.shape[1]}"
            )
        ids = candidate_ids.to(torch.int64)
        vals = proposal.to(torch.float32)
        if k < self.lanes:
            pad = (requests, steps, self.lanes - k)
            ids = torch.cat([ids, ids[..., :1].expand(pad)], dim=-1)
            vals = torch.cat([vals, vals[..., :1].expand(pad)], dim=-1)
        self.ids[slots, :steps] = ids
        self.vals[slots, :steps] = vals

    def write_onehot_index_fill(self, slots: torch.Tensor, draft_len: int, vocab: int) -> None:
        """One-hot rows at token 0 for ``slots`` over ``[:draft_len]``, by ``index_fill_``.

        ``index_fill_`` over basic-slice views, as the dense context writer does: an
        advanced-index assignment with device slot ids would synchronize the host.
        """
        self._check_vocab(vocab, "a one-hot placeholder")
        self.ids[:, :draft_len].index_fill_(0, slots, 0)
        self.vals[:, :draft_len].index_fill_(0, slots, 1.0)

    def write_onehot(self, slots: torch.Tensor, draft_len: int, vocab: int) -> None:
        """One-hot rows at token 0 for ``slots`` over ``[:draft_len]`` (advanced index)."""
        self._check_vocab(vocab, "a one-hot placeholder")
        self.ids[slots, :draft_len] = 0
        self.vals[slots, :draft_len] = 1.0

    def densify(
        self, slot_ids: torch.Tensor, draft_len: int, vocab: int, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The dense store's ``rows[slot_ids, :draft_len, :vocab]`` as ``[n, draft_len, vocab]`` FP32.

        ``out`` (contiguous FP32 of that shape) receives the rows; else a new tensor.
        """
        self._check_vocab(vocab, "the verify")
        shape = (slot_ids.shape[0], draft_len, vocab)
        if out is None:
            out = torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            if tuple(out.shape) != shape or out.dtype != torch.float32 or not out.is_contiguous():
                raise ValueError(f"[{ENV}] densify out must be contiguous FP32 {shape}")
            out.zero_()
        out.scatter_(2, self.ids[slot_ids, :draft_len], self.vals[slot_ids, :draft_len])
        return out


class DeferredDraftProbs:
    """The verified requests' dense draft rows, built when the verify asks for them.

    Stands in for the gathered ``draft_probs[slot_ids, :draft_len, :vocab]`` tensor in
    the verify's shape checks (``shape``, ``dtype``); ``materialize`` returns the rows.
    """

    dtype = torch.float32

    def __init__(
        self, store: CompactDraftProbs, slot_ids: torch.Tensor, draft_len: int, vocab: int
    ):
        self.store = store
        self.slot_ids = slot_ids
        self.draft_len = int(draft_len)
        self.vocab = int(vocab)

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.slot_ids.shape[0], self.draft_len, self.vocab))

    def dim(self) -> int:
        return 3

    def materialize(self, scratch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The dense rows, in ``scratch``'s storage when it is a large enough FP32 buffer."""
        out = None
        numel = self.shape.numel()
        if (
            scratch is not None
            and scratch.dtype == torch.float32
            and scratch.is_contiguous()
            and scratch.numel() >= numel
            and scratch.device == self.store.device
        ):
            out = scratch.reshape(-1)[:numel].view(self.shape)
        return self.store.densify(self.slot_ids, self.draft_len, self.vocab, out=out)
