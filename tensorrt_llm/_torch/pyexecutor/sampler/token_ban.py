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
"""Token-ban handling for the TorchSampler.

A *token ban* masks specific logits to ``-inf`` so certain tokens cannot be
sampled. Several features reduce to this: bad words, no-repeat n-gram, and the
min-length EOS suppression.

All bans are accumulated on the host into a single :class:`TokenBans` and then
applied to the logits in one batched step per category. Two handler variants
exist, selected once at construction from whether the overlap scheduler is
enabled:

* :class:`SynchronousTokenBanHandler` -- overlap disabled. The host token
  history is always complete, so every ban is unconditional.
* :class:`OverlappedTokenBanHandler` -- overlap enabled. The host history may
  lag the device state by one token, so a batch mixes "fresh" requests (history
  complete) with "stale" ones (missing the previous step's token, still on the
  device); the latter produce conditional bans resolved on the device at apply
  time without a device-to-host sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from tensorrt_llm._utils import prefer_pinned

from ..llm_request import LlmRequest

# A suffix-ban rule ``(prefix, col)``: ban token ``col`` when the sequence ends
# with ``prefix`` (an empty prefix bans unconditionally).
BanRule = tuple[list[int], int]
RuleGenerator = Callable[[int, LlmRequest, Callable[[], list[int]]], Iterable[BanRule]]


def _memoizing_tokens_getter(request: LlmRequest, beam_idx: int) -> Callable[[], list[int]]:
    """Return a thunk for ``request.get_tokens(beam_idx)`` that fetches once.

    ``get_tokens`` copies the whole token history through the bindings (~40us
    at 2048 tokens), so rule generators receive this thunk and only pay for
    the copy if they actually need the history.
    """
    tokens: list[int] | None = None

    def get() -> list[int]:
        nonlocal tokens
        if tokens is None:
            tokens = request.get_tokens(beam_idx)
        return tokens

    return get


@dataclass(kw_only=True)
class TokenBans:
    """Accumulated token bans, applied by ``TokenBanHandler.apply_ban_list``.

    Host-side accumulator; nothing here lives on the GPU. ``apply_ban_list``
    turns the populated lists into tensors and writes ``-inf`` into the logits,
    one batched transfer and ``index_put_`` per category.
    """

    # Unconditional bans: logits[row, col] = -inf (fresh requests + min_length).
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    # Conditional bans (overlap/stale): ban only where the device-side previous
    # token new_tokens_cuda[0, slot, 0] == expected.
    cond_rows: list[int] = field(default_factory=list)
    cond_cols: list[int] = field(default_factory=list)
    cond_slots: list[int] = field(default_factory=list)
    cond_expected: list[int] = field(default_factory=list)
    # Conditional special case: the banned column IS the device token itself
    # (overlap + no_repeat_ngram n == 1). Only OverlappedTokenBanHandler fills
    # these.
    dev_rows: list[int] = field(default_factory=list)
    dev_slots: list[int] = field(default_factory=list)


@dataclass
class _NgramIndexCache:
    """Incrementally maintained n-gram index for a single request (single beam).

    A request's ngram size is fixed for its lifetime, so the index is only ever
    grown, never rebuilt for a different ``n``; the size is therefore not stored.
    """

    tokens: list[int]
    """Host-side mirror of the request's token history."""
    indexed_up_to: int
    """How far into ``tokens`` the indices have been built, so each window is
    indexed exactly once."""
    fresh_idx: defaultdict[tuple[int, ...], set[int]]
    """Maps each ``(n - 1)``-gram to the tokens that followed it."""
    stale_idx: defaultdict[tuple[int, ...], set[tuple[int, int]]]
    """Populated only for ``n > 2``: maps each ``(n - 2)``-gram to its
    ``(next, next-but-one)`` token pairs."""


class TokenBanHandler(ABC):
    """Base class for token-ban handling: state + shared collect/apply helpers.

    Subclasses implement ``generate_ban_list`` (build a :class:`TokenBans`) and
    ``apply_ban_list`` (write it into the logits). Shared feature logic (bad
    words, no-repeat ngram, min-length EOS suppression, the suffix matcher, and
    the unconditional flush) lives here.
    """

    def __init__(self) -> None:
        # Per-request incrementally maintained n-gram indices, keyed weakly on
        # the request so entries are reclaimed when the request is freed.
        self._ngram_index_caches: WeakKeyDictionary[LlmRequest, _NgramIndexCache] = (
            WeakKeyDictionary()
        )

    # ---- entry points (subclass-specific) --------------------------------

    @abstractmethod
    def generate_ban_list(
        self,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        ngram_sizes: list[int | None],
        *,
        stale_by_one: list[bool] | None = None,
    ) -> TokenBans:
        """Collect all enabled feature bans into a fresh :class:`TokenBans`.

        ``stale_by_one`` is the per-request overlap-scheduler flag; it is only
        meaningful for :class:`OverlappedTokenBanHandler` and ignored otherwise.
        """

    @abstractmethod
    def apply_ban_list(
        self,
        logits: torch.Tensor,
        bans: TokenBans,
        *,
        new_tokens_cuda: torch.Tensor | None = None,
    ) -> None:
        """Write the accumulated bans into ``logits`` in-place."""

    # ---- feature rule sources (shared) -----------------------------------

    def _add_bad_words_bans(
        self,
        bans: TokenBans,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        *,
        stale_by_one: list[bool] | None,
    ) -> None:
        """Add "bad words" bans.

        A single-token word is banned unconditionally; a multi-token word
        ``[t0, ..., t_{k-1}]`` bans its final token ``t_{k-1}`` only when the
        ``k-1`` most recently generated tokens match the prefix
        ``[t0, ..., t_{k-2}]``.
        """

        def fresh_rules(
            index: int, r: LlmRequest, get_context: Callable[[], list[int]]
        ) -> list[BanRule]:
            # A bad word is the suffix rule "after word[:-1], ban word[-1]";
            # matching is left to the driver.
            return [
                (word[:-1], word[-1]) for word in getattr(r, "py_bad_words", None) or () if word
            ]

        def stale_rules(
            index: int, r: LlmRequest, get_context: Callable[[], list[int]]
        ) -> list[BanRule]:
            # Stale contract: check the first k - 2 prefix tokens here; the
            # last one is compared against the device token d at flush time.
            rules: list[BanRule] = []
            context: list[int] | None = None
            for word in getattr(r, "py_bad_words", None) or ():
                k = len(word)
                if k == 0:
                    continue
                if k == 1:
                    rules.append(([], word[0]))
                    continue
                if context is None:
                    context = get_context()
                # True sequence length is len(context) + 1; need >= k - 1.
                if len(context) < k - 2:
                    continue
                if k > 2 and context[-(k - 2) :] != word[: k - 2]:
                    continue
                rules.append((word[:-1], word[-1]))
            return rules

        self._add_suffix_rule_bans(
            bans,
            requests,
            num_steps,
            num_beams,
            active=lambda index, r: bool(getattr(r, "py_bad_words", None)),
            fresh_rules=fresh_rules,
            stale_rules=stale_rules,
            stale_by_one=stale_by_one,
        )

    def _add_min_length_bans(
        self,
        bans: TokenBans,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
    ) -> None:
        """Add bans that suppress EOS until a request reaches its min length.

        For a request with ``py_min_length``, the end-of-sequence token is
        banned on every step whose generated length is still below the minimum.
        This is an unconditional ban that does not depend on the token history,
        so it has no stale-host variant and is identical under both handlers.
        """
        current_offset = 0
        for index, r in enumerate(requests):
            # Advance the offset before any guard below can skip the request:
            # every request occupies its logits rows, penalized or not.
            request_offset = current_offset
            current_offset += num_steps[index] * num_beams[index]

            if not r.py_min_length:
                continue
            # Use the original end_id (before ignore_eos override) so we
            # suppress the real EOS token, not token -1.
            end_id = getattr(r, "py_original_end_id", r.py_end_id)
            if end_id is None or end_id <= -1:
                continue

            for beam_idx in range(num_beams[index]):
                for step in range(num_steps[index]):
                    if (r.get_num_tokens(beam_idx) - r.py_orig_prompt_len) + step < r.py_min_length[
                        0
                    ]:
                        bans.rows.append(request_offset + num_steps[index] * beam_idx + step)
                        bans.cols.append(end_id)
                    else:
                        break

    def _add_no_repeat_ngram_bans(
        self,
        bans: TokenBans,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        ngram_sizes: list[int | None],
        *,
        stale_by_one: list[bool] | None,
    ) -> None:
        """Add bans for tokens that would repeat an existing n-gram.

        For no-repeat-ngram size ``n``, the last ``n - 1`` tokens of the
        sequence (prompt + generated) form the current prefix; the final token
        of every existing n-gram whose first ``n - 1`` tokens equal that prefix
        is masked to ``-inf`` (same semantics as the C++ ``banRepeatNgram``
        kernel and HF's ``NoRepeatNGramLogitsProcessor``). ``n == 1`` bans every
        token already present. The restriction is per-beam.

        On the stale-host overlap path the rules cover the true sequence
        ``tokens + [d]``: the window ending at ``d`` reduces to a host-known
        column, and ``n == 1`` additionally bans ``d`` itself (``stale_extra``).

        ``ngram_sizes`` holds the effective per-request size (None or a
        non-positive value disables the restriction).
        """

        def fresh_rules(
            index: int, r: LlmRequest, get_context: Callable[[], list[int]]
        ) -> list[BanRule]:
            # Every existing n-gram is the suffix rule "after its first n - 1
            # tokens, ban its last token".
            n = ngram_sizes[index]
            if not n:  # None or non-positive: restriction disabled
                return []
            if num_beams[index] == 1:
                # Single beam: O(1) index lookup, rules come out pre-matched.
                tokens, fresh_idx, _ = self._extend_ngram_index(r, n)
                if len(tokens) < n:
                    return []
                key = tuple(tokens[-(n - 1) :]) if n > 1 else ()
                return [([], col) for col in fresh_idx.get(key, ())]
            # Multi-beam: histories diverge, so scan this beam's context.
            context = get_context()
            length = len(context)
            if length < n:
                return []
            if n == 1:
                return [([], col) for col in set(context)]
            return [(context[i : i + n - 1], context[i + n - 1]) for i in range(length - n + 1)]

        def stale_rules(
            index: int, r: LlmRequest, get_context: Callable[[], list[int]]
        ) -> list[BanRule]:
            # Stale contract: rules are already host-matched; only the device
            # comparison of d against prefix[-1] remains.
            n = ngram_sizes[index]
            if not n:  # None or non-positive: restriction disabled
                return []
            tokens, fresh_idx, stale_idx = self._extend_ngram_index(r, n)
            if n == 1:
                # Host tokens; d itself is banned via stale_extra below.
                return [([], col) for col in fresh_idx.get((), ())]
            if n == 2:
                # Empty (n - 2)-gram head: every indexed bigram is a candidate;
                # reconstruct the pairs from fresh_idx.
                pairs: Iterable[tuple[int, int]] = (
                    (key[0], col) for key, cols in fresh_idx.items() for col in cols
                )
            else:
                pairs = stale_idx.get(tuple(tokens[-(n - 2) :]), ())
            rules: list[BanRule] = [([expected], col) for expected, col in pairs]
            # The window ending at d bans d itself; the match condition forces
            # d == tokens[-1] (host-known). Check its host-side part here.
            if len(tokens) >= n - 1 and (n == 2 or tokens[-(n - 1) : -1] == tokens[-(n - 2) :]):
                rules.append(([tokens[-1]], tokens[-1]))
            return rules

        def stale_extra(bans: TokenBans, request_offset: int, index: int, r: LlmRequest) -> None:
            if ngram_sizes[index] == 1:
                # n == 1 also bans the device-only token, via a gathered column.
                assert r.py_seq_slot is not None
                bans.dev_rows.append(request_offset)
                bans.dev_slots.append(r.py_seq_slot)

        self._add_suffix_rule_bans(
            bans,
            requests,
            num_steps,
            num_beams,
            active=lambda index, r: bool(ngram_sizes[index]),
            fresh_rules=fresh_rules,
            stale_rules=stale_rules,
            stale_extra=stale_extra,
            stale_by_one=stale_by_one,
        )

    def _extend_ngram_index(
        self, request: LlmRequest, n: int
    ) -> tuple[
        list[int], dict[tuple[int, ...], set[int]], dict[tuple[int, ...], set[tuple[int, int]]]
    ]:
        """Incrementally maintained n-gram index for a single-beam request.

        Returns ``(tokens, fresh_idx, stale_idx)``: a host-side mirror of the
        token history, a map from each ``(n - 1)``-gram to the tokens that
        followed it, and (for ``n > 2`` only) a map from each ``(n - 2)``-gram
        to its ``(next, next-but-one)`` token pairs. The mirror grows via the
        scalar ``get_num_tokens`` / ``get_last_tokens`` accessors — a full
        ``get_tokens()`` copy costs ~40us at 2048 tokens — and each window is
        indexed once, so the steady-state per-step cost is O(new tokens).
        Rebuilt when the history shrinks (speculative rollback); single-beam
        only.
        """
        num_tokens = request.get_num_tokens(0)
        cache = self._ngram_index_caches.get(request)
        # A request's ngram size is fixed for its lifetime and the cache is
        # keyed on the request, so the cached index is always for this ``n``.
        if cache is not None and len(cache.tokens) <= num_tokens:
            tokens, indexed_up_to = cache.tokens, cache.indexed_up_to
            fresh_idx, stale_idx = cache.fresh_idx, cache.stale_idx
            if len(tokens) == num_tokens - 1:
                # Common decode step: exactly one new token since last call.
                tokens.append(request.get_last_tokens(0))
            elif len(tokens) < num_tokens:
                # Several tokens landed at once (e.g. accepted draft tokens).
                tokens = request.get_tokens(0)
        else:
            # First use or shrunken history (speculative rollback).
            tokens, indexed_up_to = request.get_tokens(0), 0
            fresh_idx, stale_idx = defaultdict(set), defaultdict(set)
        end = max(len(tokens) - n + 1, 0)
        for i in range(indexed_up_to, end):
            # defaultdict(set) creates the entry on first access, avoiding an
            # empty set constructed (and discarded) on every already-seen key.
            fresh_idx[tuple(tokens[i : i + n - 1])].add(tokens[i + n - 1])
            if n > 2:
                stale_idx[tuple(tokens[i : i + n - 2])].add((tokens[i + n - 2], tokens[i + n - 1]))
        self._ngram_index_caches[request] = _NgramIndexCache(
            tokens=tokens, indexed_up_to=end, fresh_idx=fresh_idx, stale_idx=stale_idx
        )
        return tokens, fresh_idx, stale_idx

    # ---- shared suffix matcher + packed-row bookkeeping ------------------

    def _add_suffix_rule_bans(
        self,
        bans: TokenBans,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        *,
        active: Callable[[int, LlmRequest], bool],
        fresh_rules: RuleGenerator,
        stale_rules: RuleGenerator,
        stale_extra: Callable[[TokenBans, int, int, LlmRequest], None] | None = None,
        stale_by_one: list[bool] | None = None,
    ) -> None:
        """Shared matcher for suffix-rule token-ban features.

        A rule ``(prefix, col)`` bans token ``col`` when the sequence (prompt +
        generated) ends with ``prefix``; an empty prefix bans unconditionally.
        Both bad words and no-repeat ngram reduce to sets of such rules; this
        driver owns the packed-row bookkeeping and the suffix matching.

        Stale-host overlap path: with the overlap scheduler, ``sample_async``
        for step ``i`` runs before ``update_requests`` for step ``i - 1``, so
        the host token list misses exactly the previous step's token ``d``,
        which still lives device-side in ``new_tokens_cuda[0, seq_slot, 0]``.
        For requests flagged in ``stale_by_one``, a rule's suffix match against
        the true sequence ``context + [d]`` is split into a host-side check of
        ``prefix[:-1]`` and a device-side comparison of ``d`` against
        ``prefix[-1]``, resolved at flush time without any device-to-host
        synchronization. This path only supports ``num_steps == 1`` and
        ``num_beams == 1`` (no speculation, no beam search).

        Args:
            bans: Accumulator the collected bans are appended to; row indices
                refer to the flattened ``[total_rows, vocab]`` logits layout
                (``num_steps * num_beams`` consecutive rows per request, in
                beam-major / step-minor order).
            requests: The requests, aligned with the packed logits rows.
            num_steps: Number of steps per request.
            num_beams: Number of beams per request.
            active: ``(index, request) -> bool``; requests where this is False
                are skipped entirely.
            fresh_rules: ``(index, request, get_context) -> iterable`` of
                ``(prefix, col)`` rules; the driver matches them against the
                beam context. ``get_context()`` fetches (and memoizes) the
                beam's token history — a full copy through the bindings — so
                generators should call it only when needed.
            stale_rules: Like ``fresh_rules``, but for the true sequence
                ``context + [d]`` and already host-matched by the generator:
                the driver only compares ``d`` against ``prefix[-1]`` on the
                device (empty prefix = unconditional). Prefixes and cols must
                consist of host-known tokens.
            stale_extra: Optional hook ``(bans, request_offset, index, request)``
                invoked once per stale request for bans the rule form cannot
                express (e.g. banning the device token itself).
            stale_by_one: Per-request flag; True when the host token list lags
                the device state by exactly one token (overlap scheduler).
                These bans need ``new_tokens_cuda`` when flushed.
        """
        current_offset = 0
        for index, r in enumerate(requests):
            request_offset = current_offset
            # Advance to the next request's rows before any early continue.
            current_offset += num_steps[index] * num_beams[index]

            if not active(index, r):
                continue

            if stale_by_one is not None and stale_by_one[index]:
                assert num_steps[index] == 1 and num_beams[index] == 1, (
                    "stale-host token-ban path only supports a single step and beam"
                )
                assert r.py_seq_slot is not None
                # Rules arrive host-matched; the device comparison of d against
                # prefix[-1] is resolved at flush time.
                seen_uncond: set[int] = set()
                seen_cond: set[tuple[int, int]] = set()
                for prefix, col in stale_rules(index, r, _memoizing_tokens_getter(r, 0)):
                    if not prefix:
                        if col not in seen_uncond:
                            seen_uncond.add(col)
                            bans.rows.append(request_offset)
                            bans.cols.append(col)
                        continue
                    key = (prefix[-1], col)
                    if key in seen_cond:
                        continue
                    seen_cond.add(key)
                    bans.cond_rows.append(request_offset)
                    bans.cond_cols.append(col)
                    bans.cond_slots.append(r.py_seq_slot)
                    bans.cond_expected.append(prefix[-1])
                if stale_extra is not None:
                    stale_extra(bans, request_offset, index, r)
                continue

            for beam_idx in range(num_beams[index]):
                # The context (prompt + generated) is fetched at most once,
                # and only if a rule with a non-empty prefix needs it.
                get_context = _memoizing_tokens_getter(r, beam_idx)
                seen_cols: set[int] = set()
                for prefix, col in fresh_rules(index, r, get_context):
                    if col in seen_cols:
                        continue
                    k = len(prefix)
                    if k > 0:
                        context = get_context()
                        if len(context) < k or context[-k:] != prefix:
                            continue
                    seen_cols.add(col)
                    # Apply to every step row of this beam. With speculation
                    # the banned set for later steps is approximated from the
                    # host context.
                    for step in range(num_steps[index]):
                        bans.rows.append(request_offset + num_steps[index] * beam_idx + step)
                        bans.cols.append(col)

    # ---- shared unconditional apply (both variants) ----------------------

    @staticmethod
    @torch.inference_mode()
    def _apply_unconditional_bans(logits: torch.Tensor, bans: TokenBans) -> None:
        """Write the unconditional bans (``bans.rows`` / ``bans.cols``) to -inf.

        Rows and cols are packed into one ``[2, N]`` tensor so the host-to-device
        copy is a single transfer; the two index rows are split on the device.
        """
        if not bans.rows:
            return
        neg_inf = torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device)
        rowcol_idx = torch.tensor(
            [bans.rows, bans.cols], dtype=torch.long, pin_memory=prefer_pinned()
        ).to(logits.device, non_blocking=True)
        logits.index_put_((rowcol_idx[0], rowcol_idx[1]), neg_inf, accumulate=False)


class SynchronousTokenBanHandler(TokenBanHandler):
    """Token bans without the overlap scheduler: the host history is complete,
    so every request is "fresh" and all bans are unconditional."""

    def generate_ban_list(
        self,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        ngram_sizes: list[int | None],
        *,
        stale_by_one: list[bool] | None = None,
    ) -> TokenBans:
        assert stale_by_one is None, "synchronous handler has no stale requests"
        bans = TokenBans()
        if any(getattr(r, "py_min_length", None) for r in requests):
            self._add_min_length_bans(bans, requests, num_steps, num_beams)
        if any(getattr(r, "py_bad_words", None) for r in requests):
            self._add_bad_words_bans(bans, requests, num_steps, num_beams, stale_by_one=None)
        if any(size is not None for size in ngram_sizes):
            self._add_no_repeat_ngram_bans(
                bans, requests, num_steps, num_beams, ngram_sizes, stale_by_one=None
            )
        # Without overlap there are no stale requests, hence no conditional bans.
        assert not bans.cond_rows and not bans.dev_rows
        return bans

    def apply_ban_list(
        self,
        logits: torch.Tensor,
        bans: TokenBans,
        *,
        new_tokens_cuda: torch.Tensor | None = None,
    ) -> None:
        self._apply_unconditional_bans(logits, bans)


class OverlappedTokenBanHandler(TokenBanHandler):
    """Token bans with the overlap scheduler: the host history can lag the
    device by one token, so a batch mixes fresh and stale requests. Stale
    requests produce conditional bans resolved on the device at apply time.

    The per-request ``stale_by_one`` flags are computed by the sampler (which
    owns the pending-step counter and the draft-batch / step / beam-width
    context) and passed into ``generate_ban_list``.
    """

    def generate_ban_list(
        self,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
        ngram_sizes: list[int | None],
        *,
        stale_by_one: list[bool] | None = None,
    ) -> TokenBans:
        bans = TokenBans()
        if any(getattr(r, "py_min_length", None) for r in requests):
            # min_length has no stale variant; identical under both handlers.
            self._add_min_length_bans(bans, requests, num_steps, num_beams)
        if any(getattr(r, "py_bad_words", None) for r in requests):
            self._add_bad_words_bans(
                bans, requests, num_steps, num_beams, stale_by_one=stale_by_one
            )
        if any(size is not None for size in ngram_sizes):
            self._add_no_repeat_ngram_bans(
                bans, requests, num_steps, num_beams, ngram_sizes, stale_by_one=stale_by_one
            )
        return bans

    def apply_ban_list(
        self,
        logits: torch.Tensor,
        bans: TokenBans,
        *,
        new_tokens_cuda: torch.Tensor | None = None,
    ) -> None:
        self._apply_unconditional_bans(logits, bans)  # fresh requests (+ min_length)
        self._apply_conditional_bans(logits, bans, new_tokens_cuda)  # stale requests

    @staticmethod
    @torch.inference_mode()
    def _apply_conditional_bans(
        logits: torch.Tensor,
        bans: TokenBans,
        new_tokens_cuda: torch.Tensor | None,
    ) -> None:
        """Write the overlap/stale bans, resolving the device-token comparison.

        ``dev_*``: the banned column is the device token ``d`` itself.
        ``cond_*``: ban only where ``d == expected``; the additive
        ``torch.where`` update keeps the op shape-static (boolean-mask indexing
        would force a device-to-host sync).
        """
        neg_inf = torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device)
        device = logits.device

        if bans.dev_rows:
            assert new_tokens_cuda is not None
            dev_row_idx = torch.tensor(
                bans.dev_rows, dtype=torch.long, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            dev_slot_idx = torch.tensor(
                bans.dev_slots, dtype=torch.long, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            prev_tokens = new_tokens_cuda[0].index_select(0, dev_slot_idx)[:, 0].to(torch.long)
            logits.index_put_((dev_row_idx, prev_tokens), neg_inf, accumulate=False)

        if bans.cond_rows:
            assert new_tokens_cuda is not None
            cond_row_idx = torch.tensor(
                bans.cond_rows, dtype=torch.long, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            cond_col_idx = torch.tensor(
                bans.cond_cols, dtype=torch.long, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            slot_idx = torch.tensor(
                bans.cond_slots, dtype=torch.long, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            expected = torch.tensor(
                bans.cond_expected, dtype=new_tokens_cuda.dtype, pin_memory=prefer_pinned()
            ).to(device, non_blocking=True)
            # Previous step's token per request (single step, single beam).
            prev_tokens = new_tokens_cuda[0].index_select(0, slot_idx)[:, 0]
            penalty = torch.where(
                prev_tokens == expected,
                neg_inf,
                torch.zeros((), dtype=logits.dtype, device=device),
            )
            logits.index_put_((cond_row_idx, cond_col_idx), penalty, accumulate=True)
