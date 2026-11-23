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
"""The reuse guard of a window-bounded drafter context (``draft_context_window.py``), on CPU.

The guard module and the prefix hasher it uses are loaded from their files under a test
package, with the logger stubbed, so the tests need neither the bindings nor a GPU. They pin:

* the mark filter: every added key is found again, and the filter never forgets one;
* the claim limit in the cases a multi-turn context server meets: a new prompt, an exact
  repeat, a turn that extends the previous one or resumes within the rewind, a re-rendered turn
  that starts before what the previous prompt wrote (capped), a short new suffix behind a
  shared interior prefix (capped), a match that ends inside a block of a sibling prefix
  (capped), and another reuse scope (not capped);
* soundness over random request streams against a model of the paged cache (the first page
  committed for a block stays; a match may end inside any committed block of the same prefix):
  with the guard, no request's drafter window inherits a row nobody wrote; without it, some do;
* the keys a request object holds (``owner``): over random streams of reservations, resumes and
  other requests' claims in between, every limit, every stat and the filter's bytes after every
  call equal the guard's without them; a request hashes its prompt once; keys are never served
  to another request, another prompt length, scope or guard; a repeated note adds only the
  blocks below what the request marked, and marks everything again after a ``clear``;
* the held keys keep no reference to the guard (a copied request rehashes and never carries the
  mark filter); an uncounted owner call (the store prefetch) seeds the keys and changes no count;
  random streams that add prefetches, agreement drops and re-reservations, reverted resumes and
  ``clear``s between notes, on a cache geometry and prompt mix like a served agentic workload,
  agree with the guard without owners in every limit, count and filter byte;
* the opt-in knob ``TLLM_KV_DRAFT_CONTEXT_HELD_KEYS``.
"""

import copy
import gc
import importlib
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[5]
_KV_CACHE = _ROOT / "tensorrt_llm/_torch/pyexecutor/kv_cache"
_PACKAGE = "_draft_context_window_under_test"


class _Log:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.lines.append(str(message))

    warning = debug = error = info


LOG = _Log()


def _load():
    saved = {name: sys.modules.get(name) for name in ("tensorrt_llm.logger",)}
    sys.modules["tensorrt_llm.logger"] = types.SimpleNamespace(logger=LOG)
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_KV_CACHE)]
    sys.modules[_PACKAGE] = package
    try:
        return importlib.import_module(f"{_PACKAGE}.draft_context_window")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


GUARD = _load()

TPB = 4
WINDOW = 8
REWIND = 4
KEEP = WINDOW + REWIND + TPB  # what KVCacheManagerV2.enable_draft_context_window derives
SEED = 12345


def _guard(**kwargs) -> "GUARD.DraftContextWindowGuard":
    return GUARD.DraftContextWindowGuard(
        TPB, WINDOW, KEEP, filter_bits=1 << 16, log_every=0, **kwargs
    )


def _tokens(*parts) -> np.ndarray:
    out: List[int] = []
    for part in parts:
        out.extend(part)
    return np.asarray(out, dtype=np.int32)


def _run(guard, tokens, claim: int, seed: int = SEED) -> Optional[int]:
    """One request: its claim limit, then the claim it makes (``min(claim, limit)``) noted."""
    limit = guard.claim_limit(tokens, len(tokens), seed)
    made = claim if limit is None else min(claim, limit)
    guard.note_claim(tokens, len(tokens), seed, made)
    return limit


# --------------------------------------------------------------------------
# The mark filter
# --------------------------------------------------------------------------
def test_mark_filter_finds_every_added_key():
    marks = GUARD.MarkFilter(1 << 12, 3)
    rng = np.random.default_rng(0)
    added = rng.integers(0, 2**63, size=500, dtype=np.uint64)
    marks.add(added)
    assert marks.contains(added).all()
    others = rng.integers(0, 2**63, size=2000, dtype=np.uint64)
    # A false positive only caps a safe claim; at this load it stays rare.
    assert marks.contains(others).mean() < 0.1
    assert 0 < marks.fill() < 1
    marks.clear()
    assert not marks.contains(added).any() and marks.fill() == 0


def test_mark_filter_refuses_a_bad_geometry():
    with pytest.raises(ValueError):
        GUARD.MarkFilter(1000, 3)
    with pytest.raises(ValueError):
        GUARD.MarkFilter(1 << 10, 0)


def test_guard_refuses_a_keep_below_the_window():
    with pytest.raises(ValueError, match="keep_tokens must cover"):
        GUARD.DraftContextWindowGuard(TPB, WINDOW, WINDOW - 1)


# --------------------------------------------------------------------------
# Claim limits
# --------------------------------------------------------------------------
SYSTEM = list(range(100, 140))  # 40 tokens: 10 blocks
TURN_A = list(range(500, 520))  # A = SYSTEM + TURN_A, 60 tokens


def test_a_new_prompt_is_not_capped_and_marks_what_it_leaves_unwritten():
    guard = _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    assert _run(guard, tokens, claim=0) is None
    # Rows [0, 60 - 16) are unwritten: blocks 0..10 (block 10 holds 40..43 < 44).
    assert guard.stats["marked_blocks"] == 11


def test_an_exact_repeat_reuses_up_to_the_last_token():
    guard = _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    _run(guard, tokens, claim=0)
    # Its window [52, 60) lies in the written rows [44, 60).
    assert guard.claim_limit(tokens, len(tokens), SEED) is None


@pytest.mark.parametrize("extra", [1, 5, 40])
def test_a_turn_that_extends_the_previous_prompt_is_not_capped(extra):
    guard = _guard()
    a = _tokens(SYSTEM, TURN_A)
    _run(guard, a, claim=0)
    b = _tokens(SYSTEM, TURN_A, range(900, 900 + extra))
    assert guard.claim_limit(b, len(b), SEED) is None


def test_a_turn_resuming_within_the_rewind_with_a_longer_prompt_is_not_capped():
    guard = _guard()
    a = _tokens(SYSTEM, TURN_A)
    _run(guard, a, claim=0)
    # Re-renders A's last 4 tokens (the rewind) and adds 6: prompt 62 >= 60.
    b = _tokens(SYSTEM, TURN_A[:-4], range(900, 906))
    assert guard.claim_limit(b, len(b), SEED) is None


def test_a_turn_that_starts_before_the_previous_watermark_is_capped():
    """B shares A's first 46 tokens, then re-renders and ends at 50, before A's end (60). B's
    window [42, 50) reaches the unwritten rows [42, 44) of A's block 10 (A wrote [44, 60)),
    which B would inherit through A's prefix."""
    guard = _guard()
    a = _tokens(SYSTEM, TURN_A)
    _run(guard, a, claim=0)
    b = _tokens(SYSTEM, TURN_A[:6], range(900, 904))  # shares A through 46, prompt 50
    limit = guard.claim_limit(b, len(b), SEED)
    assert limit == 40  # the start of block 10, the first window block under a marked prefix
    # B then computes from 40, writes [40, 50) except below 50 - 16 = 34: all of it. A later C
    # that continues B with a short suffix reads only written rows: not capped.
    guard.note_claim(b, len(b), SEED, limit)
    c = _tokens(SYSTEM, TURN_A[:6], range(900, 904), range(950, 953))
    assert guard.claim_limit(c, len(c), SEED) is None


def test_a_continuation_of_a_reused_endpoint_inherits_the_watermark():
    """B continues A's endpoint with a prompt shorter than A's (it drops A's tail and adds
    little): B computes from its match, so the rows below B's match are A's. C then continues
    B with a short suffix; its window reaches below what A wrote and must be capped."""
    guard = _guard()
    long_turn = list(range(500, 540))  # A: 80 tokens, writes [64, 80)
    a = _tokens(SYSTEM, long_turn)
    _run(guard, a, claim=0)
    b = _tokens(SYSTEM, long_turn[:24], range(900, 902))  # prompt 66, window [58, 66)
    assert guard.claim_limit(b, len(b), SEED) == 56  # block 14 (56..59) holds unwritten 56..63
    guard.note_claim(b, len(b), SEED, 56)
    c = _tokens(SYSTEM, long_turn[:24], range(900, 902), range(950, 952))
    # C's window [60, 68): every block written by B (from 56) or by A at A's tokens? Block 15
    # (60..63) is B's own computation, under B's prefix (= A's), which A marked: capped there.
    assert guard.claim_limit(c, len(c), SEED) == 60


def test_a_short_suffix_behind_a_shared_interior_prefix_is_capped():
    guard = _guard()
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    # Another conversation with the same system prompt and a 3-token message: its window
    # [35, 43) reaches the system prompt's rows, which A left unwritten.
    other = _tokens(SYSTEM, [7, 8, 9])
    assert guard.claim_limit(other, len(other), SEED) == 32
    # A window that starts in the shared prefix's last block is capped at that block, which a
    # match diverging there could take from A; it reuses at most up to it anyway.
    edge = _tokens(SYSTEM, range(7, 7 + WINDOW))
    assert guard.claim_limit(edge, len(edge), SEED) == 40
    # With a message long enough the window clears the shared prefix: not capped.
    longer = _tokens(SYSTEM, range(7, 7 + WINDOW + TPB))
    assert guard.claim_limit(longer, len(longer), SEED) is None


def test_an_uncounted_claim_limit_is_the_counted_one_and_leaves_the_stats():
    """A prefetch asks the limit the claim will get without counting a claim."""
    guard = _guard()
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    other = _tokens(SYSTEM, [7, 8, 9])
    longer = _tokens(SYSTEM, range(7, 7 + WINDOW + TPB))
    before = dict(guard.stats)
    assert guard.claim_limit(other, len(other), SEED, count=False) == 32
    assert guard.claim_limit(longer, len(longer), SEED, count=False) is None
    # Every count but the prefix hashes (the work done, uncounted calls included) is left.
    assert guard.stats == dict(before, prefix_hashes=before["prefix_hashes"] + 2)
    assert guard.claim_limit(other, len(other), SEED) == 32
    assert guard.stats["claims"] == before["claims"] + 1
    assert guard.stats["capped"] == before["capped"] + 1


def test_a_match_ending_inside_a_sibling_block_is_capped():
    """Marks are keyed by the prefix before a block: a request whose tokens diverge from A's
    inside block 10 may land on A's page for block 10 (a partial match), so it is capped
    there even though its own block 10 was never committed."""
    guard = _guard()
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    sibling = _tokens(SYSTEM, TURN_A[:2], [1, 2, 3, 4, 5])  # diverges at 42, prompt 47
    assert guard.claim_limit(sibling, len(sibling), SEED) == 36


def test_a_prompt_that_reuses_nothing_still_marks_its_shared_prefix():
    """A prompt whose claim is forced to zero (a multimodal one) commits its text prefix's
    blocks all the same; a later prompt with that prefix and a short suffix is capped."""
    guard = _guard()
    tokens = _tokens(SYSTEM, [7] * 6, TURN_A)
    guard.note_claim(tokens, len(tokens), SEED, 0)
    other = _tokens(SYSTEM, [8, 9, 10])
    assert guard.claim_limit(other, len(other), SEED) == 32


def test_another_reuse_scope_is_not_capped():
    guard = _guard()
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0, seed=SEED)
    other = _tokens(SYSTEM, [7, 8, 9])
    assert guard.claim_limit(other, len(other), SEED + 1) is None


def test_clear_forgets_every_mark():
    guard = _guard()
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    other = _tokens(SYSTEM, [7, 8, 9])
    assert guard.claim_limit(other, len(other), SEED) is not None
    guard.clear()
    assert guard.claim_limit(other, len(other), SEED) is None


def test_log_stats_names_the_counts():
    guard = _guard(name=" rank=0")
    _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    guard.log_stats()
    assert "[draft-context-window] rank=0" in LOG.lines[-1]
    assert f"keep={KEEP}" in LOG.lines[-1] and "marked_blocks=11" in LOG.lines[-1]
    assert "skipped_marks=0" in LOG.lines[-1] and "prefix_hashes=2" in LOG.lines[-1]
    assert "filter_fill" not in LOG.lines[-1]
    guard.log_stats(with_fill=True)
    assert "filter_fill=0.0" in LOG.lines[-1]


def test_the_periodic_stats_line_does_not_scan_the_filter(monkeypatch):
    """The periodic line runs on the scheduling path: it must not scan the whole filter."""
    guard = GUARD.DraftContextWindowGuard(TPB, WINDOW, KEEP, filter_bits=1 << 16, log_every=2)

    def scan():
        raise AssertionError("the periodic stats line scanned the mark filter")

    monkeypatch.setattr(guard._marks, "fill", scan)
    lines = len(LOG.lines)
    for _ in range(4):
        _run(guard, _tokens(SYSTEM, TURN_A), claim=0)
    assert len(LOG.lines) == lines + 2


# --------------------------------------------------------------------------
# Keys held by the request object
# --------------------------------------------------------------------------
class _Owner:
    """A request object: the guard keeps its keys on it."""


_COUNTS = ("claims", "capped", "capped_window_tokens", "notes", "marked_blocks")


def _reserve_then_resume(guard, owner, tokens, seed, claims, between=()):
    """A stage agreement's first chunk: a reservation (limit, then the claim noted), the
    ``between`` callables (other requests' claims), then the resume with the resumed claim
    (``claims[1]``, capped by the resume's limit) noted. Returns the two limits."""
    prompt = len(tokens)
    limits = []
    for step, claim in enumerate(claims):
        limit = guard.claim_limit(tokens, prompt, seed, owner=owner)
        limits.append(limit)
        made = claim if limit is None else min(claim, limit)
        guard.note_claim(tokens, prompt, seed, made, owner=owner)
        if step == 0:
            for call in between:
                call()
    guard.release(owner)
    return limits


def _streams_agree(seed: int) -> Tuple[int, int]:
    """Replays one stream of reservations and resumes, with other requests' claims between a
    reservation and its resume, on a guard with owners and one without. Returns the two
    guards' prefix hash counts; asserts every limit, count and filter byte agrees."""
    rng = random.Random(seed)
    owned, plain = _guard(), _guard()
    pending: List[Tuple[_Owner, np.ndarray, int, int]] = []
    for tokens in _stream(rng, 300):
        array = np.asarray(tokens, dtype=np.int32)
        prompt = len(array)
        scope = SEED + rng.randrange(2)
        roll = rng.random()
        if roll < 0.1:
            owned.clear()
            plain.clear()
        first_claim = rng.randrange(prompt + 1)
        # The resumed claim moves either way: a shorter settled cursor or a longer match.
        second_claim = max(0, min(prompt, first_claim + rng.randrange(-3 * TPB, 3 * TPB)))
        owner = _Owner()
        # Reservation on both guards.
        a = owned.claim_limit(array, prompt, scope, owner=owner)
        b = plain.claim_limit(array, prompt, scope)
        assert a == b
        made = first_claim if a is None else min(first_claim, a)
        owned.note_claim(array, prompt, scope, made, owner=owner)
        plain.note_claim(array, prompt, scope, made)
        assert np.array_equal(owned._marks._bits, plain._marks._bits)
        pending.append((owner, array, scope, second_claim))
        # Resume some pending requests, in any order, while others are still reserved.
        while pending and rng.random() < 0.6:
            owner, array, scope, claim = pending.pop(rng.randrange(len(pending)))
            prompt = len(array)
            a = owned.claim_limit(array, prompt, scope, owner=owner)
            b = plain.claim_limit(array, prompt, scope)
            assert a == b
            made = claim if a is None else min(claim, a)
            owned.note_claim(array, prompt, scope, made, owner=owner)
            plain.note_claim(array, prompt, scope, made)
            owned.release(owner)
            assert getattr(owner, GUARD.REQUEST_KEYS_ATTR) is None
            assert np.array_equal(owned._marks._bits, plain._marks._bits)
        for name in _COUNTS:
            assert owned.stats[name] == plain.stats[name], name
    return owned.stats["prefix_hashes"], plain.stats["prefix_hashes"]


@pytest.mark.parametrize("seed", range(8))
def test_held_keys_give_the_same_limits_counts_and_marks(seed):
    owned, plain = _streams_agree(seed)
    assert owned < plain


def test_a_reserved_and_resumed_request_hashes_its_prompt_once():
    guard = _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    _reserve_then_resume(guard, _Owner(), tokens, SEED, claims=(0, 0))
    assert guard.stats["claims"] == 2 and guard.stats["notes"] == 2
    assert guard.stats["prefix_hashes"] == 1
    plain = _guard()
    for _ in range(2):
        _run(plain, tokens, claim=0)
    assert plain.stats["prefix_hashes"] == 4
    assert np.array_equal(guard._marks._bits, plain._marks._bits)


def test_held_keys_are_never_served_to_another_request():
    """Two prompts of one length and scope that share their first and last blocks and differ in
    between: each request object gets its own prompt's keys."""
    guard = _guard()
    middle_a, middle_b = list(range(700, 712)), list(range(800, 812))
    a = _tokens(SYSTEM, middle_a, TURN_A[:8])
    b = _tokens(SYSTEM, middle_b, TURN_A[:8])
    # A long prompt behind b's prefix leaves b's prefix unwritten: b's claim is capped, a's not.
    behind_b = _tokens(SYSTEM, middle_b, TURN_A[:8], range(900, 940))
    guard.note_claim(behind_b, len(behind_b), SEED, 0)
    plain = _guard()
    plain.note_claim(behind_b, len(behind_b), SEED, 0)
    owner_a, owner_b = _Owner(), _Owner()
    assert guard.claim_limit(a, len(a), SEED, owner=owner_a) == plain.claim_limit(a, len(a), SEED)
    assert guard.claim_limit(b, len(b), SEED, owner=owner_b) == plain.claim_limit(b, len(b), SEED)
    assert plain.claim_limit(a, len(a), SEED) != plain.claim_limit(b, len(b), SEED)
    # Each owner keeps its own keys, which equal a fresh hash of its own prompt.
    for owner, tokens in ((owner_a, a), (owner_b, b)):
        held = getattr(owner, GUARD.REQUEST_KEYS_ATTR)
        fresh = guard._hasher.block_keys(tokens, (len(tokens) - 1) // TPB, SEED)
        assert np.array_equal(held.keys, fresh)


def test_held_keys_of_another_prompt_scope_or_guard_are_hashed_again():
    guard, other_guard = _guard(), _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    owner = _Owner()
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    first = getattr(owner, GUARD.REQUEST_KEYS_ATTR)
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR) is first
    assert guard.stats["prefix_hashes"] == 1
    shorter = tokens[:-TPB]
    guard.claim_limit(shorter, len(shorter), SEED, owner=owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR).prompt_len == len(shorter)
    guard.claim_limit(shorter, len(shorter), SEED + 1, owner=owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR).seed == SEED + 1
    assert guard.stats["prefix_hashes"] == 3
    other_guard.claim_limit(shorter, len(shorter), SEED + 1, owner=owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR).token is other_guard._keys_token
    assert other_guard.stats["prefix_hashes"] == 1


@pytest.mark.parametrize(
    "starts",
    [(0, 0), (0, 2 * TPB), (2 * TPB, 0), (2 * TPB, 2 * TPB + 3), (3 * TPB, TPB - 1), (60, 0)],
)
def test_a_repeated_note_adds_only_the_blocks_below_what_was_marked(starts):
    tokens = _tokens(SYSTEM, TURN_A)  # 60 tokens: notes mark blocks [start // 4, 11)
    guard, plain = _guard(), _guard()
    owner = _Owner()
    for start in starts:
        guard.note_claim(tokens, len(tokens), SEED, start, owner=owner)
        plain.note_claim(tokens, len(tokens), SEED, start)
        assert np.array_equal(guard._marks._bits, plain._marks._bits)
    first, second = (min(start // TPB, 11) for start in starts)
    assert guard.stats["marked_blocks"] == plain.stats["marked_blocks"]
    assert guard.stats["skipped_marks"] == (11 - max(second, first) if first < 11 else 0)
    assert guard._marks.num_added == plain._marks.num_added - guard.stats["skipped_marks"]


@pytest.mark.parametrize("empty_start", [-5, -12])
def test_a_note_that_marks_nothing_records_no_marked_range(empty_start):
    """A note whose range is empty (a start below zero while the kept tail covers the whole
    prompt) counts and hashes nothing, as without an owner; later notes agree too."""
    short = _tokens(range(12))  # fewer tokens than KEEP: the note's stop is below zero
    tokens = _tokens(SYSTEM, TURN_A)
    guard, plain = _guard(), _guard()
    owner = _Owner()
    guard.note_claim(short, len(short), SEED, empty_start, owner=owner)
    plain.note_claim(short, len(short), SEED, empty_start)
    assert guard.stats["marked_blocks"] == plain.stats["marked_blocks"] == 0
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR, None) is None
    guard.note_claim(tokens, len(tokens), SEED, 0, owner=owner)
    plain.note_claim(tokens, len(tokens), SEED, 0)
    for name in _COUNTS:
        assert guard.stats[name] == plain.stats[name], name
    assert guard.stats["skipped_marks"] == 0
    assert np.array_equal(guard._marks._bits, plain._marks._bits)


def test_a_note_after_clear_marks_every_block_again():
    tokens = _tokens(SYSTEM, TURN_A)
    guard = _guard()
    owner = _Owner()
    guard.note_claim(tokens, len(tokens), SEED, 0, owner=owner)
    guard.clear()
    other = _tokens(SYSTEM, [7, 8, 9])
    assert guard.claim_limit(other, len(other), SEED) is None
    guard.note_claim(tokens, len(tokens), SEED, 0, owner=owner)
    assert guard.stats["skipped_marks"] == 0
    assert guard.claim_limit(other, len(other), SEED) == 32


def test_release_drops_the_held_keys():
    guard = _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    owner = _Owner()
    guard.release(owner)  # nothing held: no attribute needed
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR) is not None
    guard.release(owner)
    assert getattr(owner, GUARD.REQUEST_KEYS_ATTR) is None
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    assert guard.stats["prefix_hashes"] == 2


def test_a_multimodal_note_without_a_limit_holds_keys_too():
    """A caller that notes without a claim limit (a multimodal prompt) still hashes once."""
    guard = _guard()
    tokens = _tokens(SYSTEM, [7] * 6, TURN_A)
    owner = _Owner()
    guard.note_claim(tokens, len(tokens), SEED, 0, owner=owner)
    guard.note_claim(tokens, len(tokens), SEED, 0, owner=owner)
    assert guard.stats["prefix_hashes"] == 1
    other = _tokens(SYSTEM, [8, 9, 10])
    assert guard.claim_limit(other, len(other), SEED) == 32


@pytest.mark.parametrize("seed", range(6))
def test_no_drafter_window_inherits_an_unwritten_row_with_held_keys(seed):
    assert _violations(seed, guarded=True, owned=True) == 0


def test_held_keys_keep_no_reference_to_the_guard():
    guard = _guard()
    tokens = _tokens(SYSTEM, TURN_A)
    owner = _Owner()
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    held = getattr(owner, GUARD.REQUEST_KEYS_ATTR)
    assert all(referent is not guard for referent in gc.get_referents(held))
    assert not hasattr(held, "guard")
    assert held.token is guard._keys_token


@pytest.mark.parametrize("duplicate", ["deepcopy", "pickle"])
def test_a_copied_request_hashes_again_and_carries_no_filter(duplicate):
    """A request copied after a claim (a child request copies its ``py_*`` attributes with
    ``deepcopy``) holds keys of a token no guard has: they are never served, and the copy is
    small (the mark filter does not come along)."""
    guard = GUARD.DraftContextWindowGuard(TPB, WINDOW, KEEP, filter_bits=1 << 22, log_every=0)
    tokens = _tokens(SYSTEM, TURN_A)
    owner = _Owner()
    guard.claim_limit(tokens, len(tokens), SEED, owner=owner)
    if duplicate == "deepcopy":
        twin = copy.deepcopy(owner)
    else:
        twin = pickle.loads(pickle.dumps(owner))
    assert len(pickle.dumps(owner)) < guard._marks._bits.nbytes // 64
    assert getattr(twin, GUARD.REQUEST_KEYS_ATTR).token is not guard._keys_token
    assert guard.stats["prefix_hashes"] == 1
    plain = _guard()
    assert guard.claim_limit(tokens, len(tokens), SEED, owner=twin) == plain.claim_limit(
        tokens, len(tokens), SEED
    )
    assert guard.stats["prefix_hashes"] == 2
    assert getattr(twin, GUARD.REQUEST_KEYS_ATTR).token is guard._keys_token


def test_an_uncounted_owner_call_seeds_the_keys_and_counts_nothing():
    """The store prefetch's ``count=False`` call with an owner: no claim, cap or window count
    moves, the filter is untouched, the reservation and resume after it are served from the
    keys it made (one hash for the three calls), and their limits equal a bare guard's."""
    tokens = _tokens(SYSTEM, TURN_A)
    behind = _tokens(SYSTEM, TURN_A, range(900, 940))
    guard, plain = _guard(), _guard()
    for each in (guard, plain):
        each.note_claim(behind, len(behind), SEED, 0)
    before = dict(guard.stats)
    bits = guard._marks._bits.copy()
    owner = _Owner()
    prefetched = guard.claim_limit(tokens, len(tokens), SEED, count=False, owner=owner)
    assert prefetched == plain.claim_limit(tokens, len(tokens), SEED, count=False)
    assert prefetched is not None  # a counted claim here would count a cap
    for name in ("claims", "capped", "capped_window_tokens", "notes", "marked_blocks"):
        assert guard.stats[name] == before[name], name
    assert guard.stats["prefix_hashes"] == before["prefix_hashes"] + 1
    assert np.array_equal(guard._marks._bits, bits)
    limits = _reserve_then_resume(guard, owner, tokens, SEED, claims=(0, 0))
    plain_limits = [_run(plain, tokens, claim=0) for _ in range(2)]
    assert limits == plain_limits
    assert guard.stats["prefix_hashes"] == before["prefix_hashes"] + 1
    for name in _COUNTS:
        assert guard.stats[name] == plain.stats[name], name
    assert np.array_equal(guard._marks._bits, plain._marks._bits)


class _Twin:
    """The same calls on a guard with owners and on one without; every result, count and filter
    byte is compared after each call."""

    def __init__(self, make):
        self.owned, self.plain = make(), make()

    def _check(self):
        for name in _COUNTS:
            assert self.owned.stats[name] == self.plain.stats[name], name
        assert np.array_equal(self.owned._marks._bits, self.plain._marks._bits)

    def limit(self, owner, tokens, seed, count=True):
        a = self.owned.claim_limit(tokens, len(tokens), seed, count=count, owner=owner)
        b = self.plain.claim_limit(tokens, len(tokens), seed, count=count)
        assert a == b
        self._check()
        return a

    def note(self, owner, tokens, seed, start):
        self.owned.note_claim(tokens, len(tokens), seed, start, owner=owner)
        self.plain.note_claim(tokens, len(tokens), seed, start)
        self._check()

    def claim(self, owner, tokens, seed, want):
        limit = self.limit(owner, tokens, seed)
        made = want if limit is None else min(want, limit)
        self.note(owner, tokens, seed, made)
        return made

    def clear(self):
        self.owned.clear()
        self.plain.clear()
        self._check()


def _first_chunk_life(twin, rng, owner, tokens, seed, depth, prefetch):
    """One first chunk's guard calls in the PP context executor, in a random legal order: the
    store prefetch (uncounted) at arrival, the stage agreement's reservation, possibly an
    agreement drop and a re-reservation (the keys are kept), the resume (its claim moves either
    way) and the release, possibly a reverted resume claimed again after the release, and
    ``clear``s in between. Returns the number of resumes."""
    tpb = twin.owned.tokens_per_block
    if prefetch:
        twin.limit(owner, tokens, seed, count=False)
    for _ in range(1 + (rng.random() < 0.2)):
        twin.claim(owner, tokens, seed, depth)
        if rng.random() < 0.05:
            twin.clear()
    resumes = 1 + (rng.random() < 0.15)
    for _ in range(resumes):
        moved = depth + rng.randrange(-3 * tpb, 3 * tpb)
        twin.claim(owner, tokens, seed, max(0, min(len(tokens), moved)))
        twin.owned.release(owner)
        assert getattr(owner, GUARD.REQUEST_KEYS_ATTR) is None
    return resumes


@pytest.mark.parametrize("seed", range(8))
def test_prefetch_drop_and_revert_streams_agree_with_the_bare_guard(seed):
    rng = random.Random(seed)
    twin = _Twin(_guard)
    for tokens in _stream(rng, 300):
        array = np.asarray(tokens, dtype=np.int32)
        scope = SEED + rng.randrange(2)
        depth = rng.randrange(len(array) + 1)
        _first_chunk_life(twin, rng, _Owner(), array, scope, depth, rng.random() < 0.8)
    assert twin.owned.stats["prefix_hashes"] < twin.plain.stats["prefix_hashes"]


def _agentic_stream(rng: random.Random, sessions: int, turns: int, tpb: int):
    """Prompts shaped like a served agentic workload, at a served block size: sessions share a
    few long system prefixes, each turn extends the session's previous prompt by a tool or user
    span (most not block aligned), and some turns fork a subagent chain off the parent's
    prompt. Yields prompts in an interleaved arrival order."""

    def span(blocks):
        return [rng.randrange(1, 50_000) for _ in range(blocks * tpb + rng.randrange(tpb))]

    systems = [span(rng.randrange(100, 300)) for _ in range(3)]
    chains = {session: list(rng.choice(systems)) for session in range(sessions)}
    order = sorted(
        ((session, turn) for session in range(sessions) for turn in range(turns)),
        key=lambda item: (item[1], rng.random()),
    )
    for session, _ in order:
        chains[session] = chains[session] + span(rng.randrange(1, 60))
        yield chains[session]
        if rng.random() < 0.15:
            yield chains[session] + span(1)


def _cached_depth(seen: set, keys: np.ndarray, tpb: int) -> int:
    """The reusable prefix of a prompt (whole blocks) under the blocks committed so far."""
    depth = 0
    for key in keys.tolist():
        if key not in seen:
            break
        depth += 1
    return depth * tpb


@pytest.mark.parametrize("seed", range(3))
def test_agentic_streams_at_the_served_geometry_agree_with_the_bare_guard(seed):
    """The served block size, window and keep (32, 2048, 2048 + 512 + 32) on long multi-turn
    prompts with shared system prefixes and subagent forks; claims reach as deep as the blocks
    committed so far. Every limit, count and filter byte agrees with the bare guard, each
    first chunk hashes its prompt once per resume (the prefetch's keys serve the reservations
    and the first resume; a reverted resume hashes again), and some claims are capped."""
    rng = random.Random(seed)
    tpb, read, keep = 32, 2048, 2048 + 512 + 32

    def make():
        return GUARD.DraftContextWindowGuard(tpb, read, keep, filter_bits=1 << 20, log_every=0)

    twin = _Twin(make)
    hasher = twin.owned._hasher
    seen: set = set()
    for tokens in _agentic_stream(rng, sessions=12, turns=6, tpb=tpb):
        array = np.asarray(tokens, dtype=np.int32)
        keys = hasher.block_keys(array, len(array) // tpb, SEED)
        depth = min(_cached_depth(seen, keys, tpb), len(array) - 1)
        before = twin.owned.stats["prefix_hashes"]
        resumes = _first_chunk_life(twin, rng, _Owner(), array, SEED, depth, prefetch=True)
        assert twin.owned.stats["prefix_hashes"] - before == resumes
        seen.update(keys.tolist())
    assert twin.owned.stats["capped"] > 0
    assert twin.owned.stats["prefix_hashes"] * 2 < twin.plain.stats["prefix_hashes"]


def test_held_keys_knob():
    assert GUARD.HELD_KEYS_ENV == "TLLM_KV_DRAFT_CONTEXT_HELD_KEYS"
    assert GUARD.draft_context_held_keys_enabled({}) is False
    assert GUARD.draft_context_held_keys_enabled({GUARD.HELD_KEYS_ENV: "0"}) is False
    assert GUARD.draft_context_held_keys_enabled({GUARD.HELD_KEYS_ENV: " 1 "}) is True


# --------------------------------------------------------------------------
# Soundness against a model of the paged cache
# --------------------------------------------------------------------------
class _PagedCache:
    """Committed pages by block content (the tokens through the block's end, partial last
    blocks included); the first page committed for a block stays. Each page knows which of
    its rows hold a written drafter context row."""

    def __init__(self) -> None:
        self.pages: Dict[Tuple[int, ...], List[bool]] = {}

    def match(self, tokens: Tuple[int, ...], rng: random.Random) -> Tuple[int, List[bool]]:
        """The longest reusable prefix (at most ``len - 1``) and the written flags of the rows
        it inherits, landing a partial last block on a random committed page that shares it."""
        limit = len(tokens) - 1
        depth, flags = 0, []
        while depth + TPB <= limit and tokens[: depth + TPB] in self.pages:
            flags.extend(self.pages[tokens[: depth + TPB]])
            depth += TPB
        candidates = []
        for key, page in self.pages.items():
            if not depth < len(key) <= depth + TPB or key[:depth] != tokens[:depth]:
                continue
            shared = 0
            for ours, theirs in zip(tokens[depth:limit], key[depth:]):
                if ours != theirs:
                    break
                shared += 1
            if shared:
                candidates.append((shared, page))
        if candidates:
            shared, page = rng.choice(candidates)
            flags.extend(page[:shared])
            depth += shared
        return depth, flags

    def commit(self, tokens: Tuple[int, ...], written: List[bool]) -> None:
        for start in range(0, len(tokens), TPB):
            end = min(start + TPB, len(tokens))
            self.pages.setdefault(tokens[:end], written[start:end])


def _stream(rng: random.Random, count: int) -> List[Tuple[int, ...]]:
    """Requests of multi-turn conversations behind two shared system prompts: extensions,
    re-renders of the last few tokens, forks and exact repeats."""
    systems = [tuple(range(1000, 1013)), tuple(range(2000, 2018))]
    conversations: List[Tuple[int, ...]] = []
    out = []
    for _ in range(count):
        roll = rng.random()
        if not conversations or roll < 0.2:
            prompt = rng.choice(systems) + tuple(
                rng.randrange(50) for _ in range(rng.randrange(1, 12))
            )
            conversations.append(prompt)
        else:
            index = rng.randrange(len(conversations))
            base = conversations[index]
            if roll < 0.55:  # the next turn
                prompt = base + tuple(rng.randrange(50) for _ in range(rng.randrange(1, 14)))
            elif roll < 0.8:  # re-render the tail, then add a little
                cut = rng.randrange(1, 12)
                prompt = base[: max(1, len(base) - cut)] + tuple(
                    rng.randrange(50) for _ in range(rng.randrange(0, 6))
                )
            elif roll < 0.9:  # an exact repeat
                prompt = base
            else:  # a fork: a new branch from somewhere inside
                cut = rng.randrange(1, max(2, len(base)))
                prompt = base[:cut] + tuple(rng.randrange(50) for _ in range(rng.randrange(1, 10)))
            if roll < 0.8:
                conversations[index] = prompt
            else:
                conversations.append(prompt)
        out.append(prompt)
    return out


def _violations(seed: int, guarded: bool, owned: bool = False) -> int:
    rng = random.Random(seed)
    cache = _PagedCache()
    guard = _guard()
    bad = 0
    for tokens in _stream(rng, 400):
        prompt = len(tokens)
        array = np.asarray(tokens, dtype=np.int32)
        owner = _Owner() if owned else None
        limit = guard.claim_limit(array, prompt, SEED, owner=owner) if guarded else None
        depth, flags = cache.match(tokens, rng)
        if limit is not None and depth > limit:
            # A capped claim is a shorter match: the cache prunes to whole blocks below it.
            depth, flags = limit, flags[:limit]
        window = range(max(0, prompt - WINDOW), depth)
        if any(not flags[row] for row in window):
            bad += 1
        guard.note_claim(array, prompt, SEED, depth, owner=owner)
        written = flags[:depth] + [row >= prompt - KEEP for row in range(depth, prompt)]
        cache.commit(tokens, written)
    return bad


@pytest.mark.parametrize("seed", range(12))
def test_no_drafter_window_inherits_an_unwritten_row(seed):
    assert _violations(seed, guarded=True) == 0


def test_the_unguarded_cache_serves_unwritten_rows():
    """The stream is adversarial enough to matter: without the guard, windows inherit
    unwritten rows."""
    assert sum(_violations(seed, guarded=False) for seed in range(12)) > 0
