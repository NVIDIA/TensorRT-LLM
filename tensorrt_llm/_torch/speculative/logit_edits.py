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
"""Embedding bias and bad words for one-model speculative decoding.

``TorchSampler`` applies both on the host-prepared logits of one step
(``apply_embedding_bias`` and ``TokenBanHandler``). The one-model worker samples
and verifies inside a forward that is replayed as a CUDA graph, so here each
request's edits live in device tables indexed by sequence slot, at addresses
that never move, and every decision is taken on the device:

* **Embedding bias** is stored sparsely: the token ids with a nonzero bias and
  their values. A row of logits adds the entries of the slot that owns it.
* **Bad words** are stored as suffix rules ``(prefix, token)``: ``token`` is
  banned on a row whose sequence ends with ``prefix``. A single-token word has
  an empty prefix and is banned on every row.
* A rule's prefix is matched against the row's own sequence, which the host does
  not know: with the overlap scheduler it lags the device by a whole step, and a
  verified position follows draft tokens that only the device holds. Each slot
  therefore keeps the last tokens of its sequence in ``tail``, advanced on the
  device by the tokens every step accepts, and position ``p`` of a step sees the
  last tokens of ``tail ++ drafts[:p]``.

The tables have fixed capacities (:data:`MAX_BIAS_TOKENS`, :data:`MAX_BAD_WORDS`,
:data:`MAX_BAD_WORD_LEN`); a request that exceeds one is rejected at admission
(:func:`rejection_reason`).

Slots that hold no edits keep their neutral rows, so the pass changes nothing
for them, and it only reads and writes ``[rows, capacity]`` tensors: its cost
does not grow with the vocabulary.
"""

import weakref
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

__all__ = [
    "MAX_BAD_WORDS",
    "MAX_BAD_WORD_LEN",
    "MAX_BIAS_TOKENS",
    "LogitEditState",
    "advance_tails",
    "apply_bad_words",
    "apply_embedding_bias",
    "rejection_reason",
    "request_has_logit_edits",
]

# Tokens one request may bias (nonzero entries of its embedding_bias).
MAX_BIAS_TOKENS = 1024
# Bad words one request may carry, and the tokens one bad word may span.
MAX_BAD_WORDS = 256
MAX_BAD_WORD_LEN = 16

# Tokens of a slot's sequence that the longest rule prefix can read.
_TAIL_LEN = MAX_BAD_WORD_LEN - 1
# Marks the prefix positions a rule does not constrain.
_ANY_TOKEN = -1
# Fills the tail of a sequence shorter than it; equals no token id and no rule entry.
_NO_TOKEN = -2


def _nonzero_bias_count(embedding_bias: Optional[torch.Tensor]) -> int:
    if embedding_bias is None:
        return 0
    return int(torch.count_nonzero(embedding_bias))


def rejection_reason(
    embedding_bias: Optional[torch.Tensor],
    bad_words: Optional[Sequence[Sequence[int]]],
) -> Optional[str]:
    """Why a request's edits do not fit the per-slot tables, or None when they do.

    Args:
        embedding_bias: the request's ``[vocab_size]`` bias, or None.
        bad_words: the request's bad words as token sequences, or None.
    """
    num_biased = _nonzero_bias_count(embedding_bias)
    if num_biased > MAX_BIAS_TOKENS:
        return (
            f"embedding_bias sets {num_biased} tokens, but one-model speculative decoding "
            f"supports at most {MAX_BIAS_TOKENS} biased tokens per request. Bias fewer "
            "tokens, or disable speculative decoding."
        )
    words = [word for word in bad_words or () if len(word) > 0]
    if len(words) > MAX_BAD_WORDS:
        return (
            f"The request carries {len(words)} bad words, but one-model speculative decoding "
            f"supports at most {MAX_BAD_WORDS} per request. Pass fewer bad words, or disable "
            "speculative decoding."
        )
    longest = max((len(word) for word in words), default=0)
    if longest > MAX_BAD_WORD_LEN:
        return (
            f"A bad word spans {longest} tokens, but one-model speculative decoding supports "
            f"bad words of at most {MAX_BAD_WORD_LEN} tokens. Shorten it, or disable "
            "speculative decoding."
        )
    return None


def request_has_logit_edits(request) -> bool:
    """Whether ``request`` carries an embedding bias or bad words."""
    return getattr(request, "py_embedding_bias", None) is not None or bool(
        getattr(request, "py_bad_words", None)
    )


def _owner_ref(request) -> Callable[[], object]:
    """A reference to ``request`` that does not keep it alive when it can be weak."""
    try:
        return weakref.ref(request)
    except TypeError:
        return lambda: request


@dataclass(kw_only=True)
class LogitEditState:
    """Device tables of the per-request logits edits, one row per sequence slot.

    ``rows`` is the slot pool plus one scratch row: CUDA-graph padding requests
    (``py_seq_slot is None``) are routed to ``dummy_row``, whose tables stay
    neutral.

    The neutral values are a zero bias on token 0 and rules that are not valid,
    so a slot that was never filled leaves its logits untouched.
    """

    vocab_size: int
    dummy_row: int
    bias_ids: torch.Tensor
    """int64 [rows, MAX_BIAS_TOKENS]; token ids, 0 where unused."""
    bias_values: torch.Tensor
    """float32 [rows, MAX_BIAS_TOKENS]; their biases, 0.0 where unused."""
    ban_prefixes: torch.Tensor
    """int32 [rows, MAX_BAD_WORDS, _TAIL_LEN]; each rule's prefix, right-aligned after
    ``_ANY_TOKEN``."""
    ban_tokens: torch.Tensor
    """int64 [rows, MAX_BAD_WORDS]; the token each rule bans, 0 where unused."""
    ban_valid: torch.Tensor
    """bool [rows, MAX_BAD_WORDS]; which rules exist."""
    tail: torch.Tensor
    """int32 [rows, _TAIL_LEN]; the last tokens of each slot's sequence, right-aligned
    after ``_NO_TOKEN``. Maintained for every slot, read only by rules with a prefix."""
    tail_offsets: torch.Tensor
    """int64 [_TAIL_LEN]; arange, for the tail windows."""

    # Host bookkeeping. The request object that last took each slot is compared by
    # identity, not by request id: warmup dummy requests reuse the ids real requests get.
    _slot_owner: list = field(default_factory=list, repr=False)
    # Whether a slot's device rows differ from the neutral ones.
    _slot_has_bias: list = field(default_factory=list, repr=False)
    _slot_has_bans: list = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls, *, slot_capacity: int, vocab_size: int, device: torch.device | str = "cuda"
    ) -> "LogitEditState":
        """Allocate every table up front: a captured CUDA graph records their addresses."""
        rows = slot_capacity + 1
        return cls(
            vocab_size=vocab_size,
            dummy_row=slot_capacity,
            bias_ids=torch.zeros((rows, MAX_BIAS_TOKENS), dtype=torch.int64, device=device),
            bias_values=torch.zeros((rows, MAX_BIAS_TOKENS), dtype=torch.float32, device=device),
            ban_prefixes=torch.full(
                (rows, MAX_BAD_WORDS, _TAIL_LEN), _ANY_TOKEN, dtype=torch.int32, device=device
            ),
            ban_tokens=torch.zeros((rows, MAX_BAD_WORDS), dtype=torch.int64, device=device),
            ban_valid=torch.zeros((rows, MAX_BAD_WORDS), dtype=torch.bool, device=device),
            tail=torch.full((rows, _TAIL_LEN), _NO_TOKEN, dtype=torch.int32, device=device),
            tail_offsets=torch.arange(_TAIL_LEN, dtype=torch.int64, device=device),
            _slot_owner=[None] * rows,
            _slot_has_bias=[False] * rows,
            _slot_has_bans=[False] * rows,
        )

    def _bias_entries(self, request) -> tuple[list[int], list[float]]:
        """The nonzero entries of the request's bias that address a token of the vocabulary."""
        bias = getattr(request, "py_embedding_bias", None)
        if bias is None:
            return [], []
        bias = bias.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        ids = torch.nonzero(bias, as_tuple=False).reshape(-1)
        if ids.numel() and int(ids[-1]) >= self.vocab_size:
            logger.warning_once(
                f"embedding_bias has {bias.numel()} entries but the vocabulary has "
                f"{self.vocab_size} tokens; the entries past the vocabulary are ignored.",
                key="spec_embedding_bias_past_vocab",
            )
            ids = ids[ids < self.vocab_size]
        # Admission rejects a request with more entries (see rejection_reason).
        assert ids.numel() <= MAX_BIAS_TOKENS, "embedding_bias exceeds the per-slot bias table"
        return ids.tolist(), bias[ids].tolist()

    def _ban_rules(self, request) -> tuple[list[list[int]], list[int]]:
        """The request's bad words as ``(prefixes, tokens)`` rows of the rule table."""
        prefixes: list[list[int]] = []
        tokens: list[int] = []
        for word in getattr(request, "py_bad_words", None) or ():
            # A token outside the vocabulary can never be sampled.
            if len(word) == 0 or not 0 <= word[-1] < self.vocab_size:
                continue
            prefix = list(word[:-1])
            # Admission rejects a request with longer or more words (see rejection_reason).
            assert len(prefix) <= _TAIL_LEN, "a bad word exceeds the per-slot rule table"
            prefixes.append([_ANY_TOKEN] * (_TAIL_LEN - len(prefix)) + prefix)
            tokens.append(word[-1])
        assert len(tokens) <= MAX_BAD_WORDS, "bad words exceed the per-slot rule table"
        return prefixes, tokens

    def observe(self, requests: Sequence) -> None:
        """Load the edits of every request that newly holds its slot, before the step's forward.

        A slot is (re)loaded when its occupant changes and when its occupant samples its first
        token: a request prefilling in chunks occupies its slot for several steps, and each of
        those advances the slot's tail by a token that is then discarded.

        The tail starts as the last tokens the request holds: its prompt, plus whatever it
        already generated elsewhere (a disaggregated context worker, a resumed request).
        """
        pin = prefer_pinned()
        bias_slots: list[int] = []
        bias_ids: list[list[int]] = []
        bias_values: list[list[float]] = []
        ban_slots: list[int] = []
        ban_prefixes: list[list[list[int]]] = []
        ban_tokens: list[list[int]] = []
        ban_valid: list[list[bool]] = []
        tail_slots: list[int] = []
        tails: list[list[int]] = []
        # (slot, owner, has_bias, has_bans) of the slots that change occupant.
        loaded: list[tuple[int, Callable[[], object], bool, bool]] = []

        for request in requests:
            slot = getattr(request, "py_seq_slot", None)
            # Live rows only: the last row is the scratch row of padding requests.
            if slot is None or not 0 <= slot < self.dummy_row:
                continue
            owner = self._slot_owner[slot]
            is_owner = owner is not None and owner() is request
            starts_sampling = getattr(request, "is_context_init_state", False) and getattr(
                request, "is_last_context_chunk", True
            )
            if is_owner and not starts_sampling:
                continue

            if not is_owner:
                ids, values = self._bias_entries(request)
                if ids or self._slot_has_bias[slot]:
                    pad = MAX_BIAS_TOKENS - len(ids)
                    bias_slots.append(slot)
                    bias_ids.append(ids + [0] * pad)
                    bias_values.append(values + [0.0] * pad)
                prefixes, tokens = self._ban_rules(request)
                if tokens or self._slot_has_bans[slot]:
                    pad = MAX_BAD_WORDS - len(tokens)
                    ban_slots.append(slot)
                    ban_prefixes.append(prefixes + [[_ANY_TOKEN] * _TAIL_LEN] * pad)
                    ban_tokens.append(tokens + [0] * pad)
                    ban_valid.append([True] * len(tokens) + [False] * pad)
                loaded.append((slot, _owner_ref(request), bool(ids), bool(tokens)))

            # Only rules with a prefix read the tail; skip the token-history copy otherwise.
            if any(len(word) > 1 for word in getattr(request, "py_bad_words", None) or ()):
                history = request.get_tokens(0)[-_TAIL_LEN:]
                tail_slots.append(slot)
                tails.append([_NO_TOKEN] * (_TAIL_LEN - len(history)) + list(history))

        def upload(values: list, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(values, dtype=dtype, pin_memory=pin).to(
                self.tail.device, non_blocking=True
            )

        # A slot's bookkeeping follows its device rows. Until the uploads below are through,
        # the slot counts as unowned and holding edits, so that if one of them raises (the
        # step fails, the executor keeps serving) the next occupant reloads every row.
        for slot, _, _, _ in loaded:
            self._slot_owner[slot] = None
            self._slot_has_bias[slot] = True
            self._slot_has_bans[slot] = True

        if bias_slots:
            slots = upload(bias_slots, torch.int64)
            self.bias_ids.index_copy_(0, slots, upload(bias_ids, torch.int64))
            self.bias_values.index_copy_(0, slots, upload(bias_values, torch.float32))
        if ban_slots:
            slots = upload(ban_slots, torch.int64)
            self.ban_prefixes.index_copy_(0, slots, upload(ban_prefixes, torch.int32))
            self.ban_tokens.index_copy_(0, slots, upload(ban_tokens, torch.int64))
            self.ban_valid.index_copy_(0, slots, upload(ban_valid, torch.bool))
        if tail_slots:
            self.tail.index_copy_(0, upload(tail_slots, torch.int64), upload(tails, torch.int32))

        for slot, owner, has_bias, has_bans in loaded:
            self._slot_owner[slot] = owner
            self._slot_has_bias[slot] = has_bias
            self._slot_has_bans[slot] = has_bans


def apply_embedding_bias(
    logits: torch.Tensor, state: LogitEditState, row_slots: torch.Tensor
) -> None:
    """Add each row's embedding bias to ``logits`` in place.

    Device-only and shape-static: safe to capture in a CUDA graph.

    Args:
        logits: float32 ``[T, vocab_size]`` target logits.
        state: the per-slot tables.
        row_slots: int64 ``[T]`` slot row that owns each logits row; rows of no live
            request point at ``state.dummy_row``.
    """
    # Unused entries add 0.0 to token 0, so repeated ids only ever repeat a no-op.
    logits.scatter_add_(1, state.bias_ids[row_slots], state.bias_values[row_slots])


def apply_bad_words(
    logits: torch.Tensor,
    state: LogitEditState,
    row_slots: torch.Tensor,
    intra_step_tokens: torch.Tensor,
    intra_step_valid: torch.Tensor,
) -> None:
    """Set the logits of each row's banned tokens to ``-inf`` in place.

    Device-only and shape-static: safe to capture in a CUDA graph.

    Args:
        logits: float32 ``[T, vocab_size]`` target logits.
        state: the per-slot tables.
        row_slots: int64 ``[T]`` slot row that owns each logits row; rows of no live
            request point at ``state.dummy_row``.
        intra_step_tokens: ``[T, draft_len]``; column ``j`` is the token the row's own
            request drafted at position ``j`` of this step.
        intra_step_valid: bool ``[T, draft_len]``; true for the draft positions that precede
            the row's own, i.e. the tokens between the slot's tail and the row.
    """
    # The row's sequence ends with its slot's tail followed by the drafts before the row.
    position = intra_step_valid.sum(dim=1, dtype=torch.int64)
    history = torch.cat([state.tail[row_slots], intra_step_tokens.to(torch.int32)], dim=1)
    context = history.gather(1, position.unsqueeze(1) + state.tail_offsets)

    prefixes = state.ban_prefixes[row_slots]
    matched = ((prefixes == context.unsqueeze(1)) | (prefixes == _ANY_TOKEN)).all(dim=-1)
    matched &= state.ban_valid[row_slots]

    # A minimum rather than an addition: several rules may ban one token, and a rule that did
    # not match must leave its (possibly shared) token as it is.
    inf = torch.full((), float("inf"), dtype=logits.dtype, device=logits.device)
    logits.scatter_reduce_(
        1, state.ban_tokens[row_slots], torch.where(matched, -inf, inf), reduce="amin"
    )


def advance_tails(
    state: LogitEditState,
    slot_rows: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Append the tokens each request accepted in this step to its slot's tail.

    Device-only and shape-static: safe to capture in a CUDA graph.

    Args:
        state: the per-slot tables.
        slot_rows: int64 ``[B]`` slot row per request; padding requests point at
            ``state.dummy_row``.
        accepted_tokens: ``[B, max_accepted]`` (or ``[B]``) accepted token ids.
        num_accepted_tokens: ``[B]`` how many of each row's entries are real.
    """
    if accepted_tokens.dim() == 1:
        accepted_tokens = accepted_tokens.unsqueeze(1)
    accepted = num_accepted_tokens.to(torch.int64).clamp(0, accepted_tokens.shape[1])
    # New tail = the last tokens of (old tail ++ the accepted tokens).
    history = torch.cat([state.tail[slot_rows], accepted_tokens.to(torch.int32)], dim=1)
    state.tail.index_copy_(
        0, slot_rows, history.gather(1, accepted.unsqueeze(1) + state.tail_offsets)
    )
