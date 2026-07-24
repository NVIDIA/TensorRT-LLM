# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Unit tests for the token-ban handlers (bad words, no-repeat ngram,
min-length EOS suppression) in
tensorrt_llm/_torch/pyexecutor/sampler/token_ban.py."""

from typing import cast

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler.token_ban import (
    OverlappedTokenBanHandler,
    SynchronousTokenBanHandler,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestApplyBadWords:
    """Unit tests for the bad-words path of TokenBanHandler.

    Single-token words are banned unconditionally; multi-token words ban their
    final token only when the token suffix (prompt + generated) matches the
    word prefix.
    """

    VOCAB = 16

    class MockLlmRequest:
        """Minimal stub exposing the attributes the bad-words path reads."""

        def __init__(self, tokens, *, bad_words=None, prompt_len=0, seq_slot=None):
            # get_tokens(beam) returns the full token sequence (prompt +
            # generated); py_orig_prompt_len marks where generation starts.
            self.py_orig_prompt_len = prompt_len
            self._tokens = list(tokens)
            self.py_bad_words = bad_words
            self.py_seq_slot = seq_slot

        def get_tokens(self, beam_idx):
            return self._tokens

    def _run(self, requests, num_steps, num_beams):
        total_rows = sum(s * b for s, b in zip(num_steps, num_beams))
        logits = torch.zeros(total_rows, self.VOCAB, device="cuda")
        ngram_sizes: list[int | None] = [None] * len(requests)
        handler = SynchronousTokenBanHandler()
        bans = handler.generate_ban_list(
            cast(list[LlmRequest], requests), num_steps, num_beams, ngram_sizes
        )
        handler.apply_ban_list(logits, bans)
        return logits

    @staticmethod
    def _banned_cols(logits_row):
        return set(torch.nonzero(torch.isinf(logits_row)).flatten().tolist())

    def test_single_token_unconditional(self):
        req = self.MockLlmRequest(tokens=[3, 4], bad_words=[[7]])
        logits = self._run([req], num_steps=[1], num_beams=[1])
        assert self._banned_cols(logits[0]) == {7}

    def test_multi_token_prefix_hit(self):
        # tokens end with [9]; word [9, 2] -> ban token 2.
        req = self.MockLlmRequest(tokens=[1, 9], bad_words=[[9, 2]])
        logits = self._run([req], num_steps=[1], num_beams=[1])
        assert self._banned_cols(logits[0]) == {2}

    def test_multi_token_prefix_miss(self):
        # tokens end with [3]; word [9, 2] prefix [9] does not match.
        req = self.MockLlmRequest(tokens=[1, 3], bad_words=[[9, 2]])
        logits = self._run([req], num_steps=[1], num_beams=[1])
        assert self._banned_cols(logits[0]) == set()

    def test_multi_token_prefix_in_prompt(self):
        # The prefix [9] lies in the prompt (nothing generated yet); the word
        # [9, 2] must still ban token 2.
        req = self.MockLlmRequest(tokens=[1, 9], bad_words=[[9, 2]], prompt_len=2)
        logits = self._run([req], num_steps=[1], num_beams=[1])
        assert self._banned_cols(logits[0]) == {2}

    # --- Overlap-scheduler (stale-host) path -------------------------------
    #
    # With the overlap scheduler the host token list lags the device state by
    # one token; the newest token is read from new_tokens_cuda[0, seq_slot, 0]
    # on the GPU. These tests drive the overlap handler with stale_by_one set.

    NUM_SLOTS = 4

    def _new_tokens_cuda(self, slot_tokens):
        buf = torch.full((1, self.NUM_SLOTS, 1), -1, dtype=torch.int32, device="cuda")
        for slot, tok in slot_tokens.items():
            buf[0, slot, 0] = tok
        return buf

    def _run_stale(self, requests, stale_by_one, slot_tokens):
        total_rows = len(requests)
        logits = torch.zeros(total_rows, self.VOCAB, device="cuda")
        ngram_sizes: list[int | None] = [None] * len(requests)
        num_steps = [1] * len(requests)
        num_beams = [1] * len(requests)
        handler = OverlappedTokenBanHandler()
        bans = handler.generate_ban_list(
            cast(list[LlmRequest], requests),
            num_steps,
            num_beams,
            ngram_sizes,
            stale_by_one=stale_by_one,
        )
        handler.apply_ban_list(logits, bans, new_tokens_cuda=self._new_tokens_cuda(slot_tokens))
        return logits

    def test_stale_two_token_device_hit(self):
        # Host context [1]; device holds the pending token 9; word [9, 2]
        # completes its prefix on the device side -> ban token 2.
        req = self.MockLlmRequest(tokens=[1], bad_words=[[9, 2]], seq_slot=0)
        logits = self._run_stale([req], [True], {0: 9})
        assert self._banned_cols(logits[0]) == {2}

    def test_stale_two_token_device_miss(self):
        # Device token is 3, not the required prefix 9 -> nothing banned.
        req = self.MockLlmRequest(tokens=[1], bad_words=[[9, 2]], seq_slot=0)
        logits = self._run_stale([req], [True], {0: 3})
        assert self._banned_cols(logits[0]) == set()

    def test_stale_three_token_host_and_device_hit(self):
        # Word [5, 9, 2]: host suffix must be [5], device token must be 9.
        req = self.MockLlmRequest(tokens=[1, 5], bad_words=[[5, 9, 2]], seq_slot=1)
        logits = self._run_stale([req], [True], {1: 9})
        assert self._banned_cols(logits[0]) == {2}

    def test_stale_three_token_host_miss(self):
        # Host suffix [3] does not match the word prefix [5]; the device token
        # matching is irrelevant -> nothing banned.
        req = self.MockLlmRequest(tokens=[1, 3], bad_words=[[5, 9, 2]], seq_slot=1)
        logits = self._run_stale([req], [True], {1: 9})
        assert self._banned_cols(logits[0]) == set()

    def test_stale_single_token_unconditional(self):
        # Single-token words are banned regardless of the device token.
        req = self.MockLlmRequest(tokens=[1], bad_words=[[7]], seq_slot=0)
        logits = self._run_stale([req], [True], {0: 3})
        assert self._banned_cols(logits[0]) == {7}

    def test_stale_and_fresh_requests_mixed(self):
        # Request 0 (stale) matches via the device token; request 1 (fresh)
        # matches via the host context, as on the regular path.
        stale_req = self.MockLlmRequest(tokens=[1], bad_words=[[9, 2]], seq_slot=0)
        fresh_req = self.MockLlmRequest(tokens=[1, 9], bad_words=[[9, 4]], seq_slot=1)
        logits = self._run_stale([stale_req, fresh_req], [True, False], {0: 9})
        assert self._banned_cols(logits[0]) == {2}
        assert self._banned_cols(logits[1]) == {4}

    def test_stale_conditional_and_unconditional_same_cell(self):
        # An unconditional single-token ban and a device-conditional ban on the
        # same logit must combine to -inf (no NaN from -inf + -inf).
        req = self.MockLlmRequest(tokens=[1], bad_words=[[2], [9, 2]], seq_slot=0)
        logits = self._run_stale([req], [True], {0: 9})
        assert self._banned_cols(logits[0]) == {2}
        assert not torch.isnan(logits).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestApplyNoRepeatNgram:
    """Unit tests for the no-repeat-ngram path of TokenBanHandler.

    With ngram size n, the last n-1 tokens of the sequence form the current
    prefix; the final token of every existing n-gram whose first n-1 tokens
    match that prefix is banned, so no n-gram is generated twice.
    """

    VOCAB = 16

    class MockLlmRequest:
        """Minimal stub exposing the attributes the no-repeat-ngram path reads."""

        def __init__(self, tokens, *, seq_slot=None):
            self._tokens = list(tokens)
            self.py_seq_slot = seq_slot

        def get_tokens(self, beam_idx):
            # Copy, like the real binding: the sampler's mirror must not alias.
            return list(self._tokens)

        def get_num_tokens(self, beam_idx):
            return len(self._tokens)

        def get_last_tokens(self, beam_idx):
            return self._tokens[-1]

    class MultiBeamMockLlmRequest:
        """Stub with a distinct token history per beam for multi-beam tests."""

        def __init__(self, beam_tokens, *, seq_slot=None):
            self._beam_tokens = [list(t) for t in beam_tokens]
            self.py_seq_slot = seq_slot

        def get_tokens(self, beam_idx):
            return list(self._beam_tokens[beam_idx])

        def get_num_tokens(self, beam_idx):
            return len(self._beam_tokens[beam_idx])

        def get_last_tokens(self, beam_idx):
            return self._beam_tokens[beam_idx][-1]

    @staticmethod
    def _distinct_logits(total_rows, vocab):
        # Distinct non-zero values per cell, so a comparison against the input
        # can detect any logit the sampler touched but should not have.
        return (torch.arange(total_rows * vocab, device="cuda", dtype=torch.float) + 1.0).reshape(
            total_rows, vocab
        )

    def _run(self, requests, ngram_sizes, num_steps=None, num_beams=None, handler=None):
        num_steps = num_steps or [1] * len(requests)
        num_beams = num_beams or [1] * len(requests)
        total_rows = sum(s * b for s, b in zip(num_steps, num_beams, strict=True))
        logits = self._distinct_logits(total_rows, self.VOCAB)
        before = logits.clone()
        handler = handler or SynchronousTokenBanHandler()
        bans = handler.generate_ban_list(
            cast(list[LlmRequest], requests), num_steps, num_beams, ngram_sizes
        )
        handler.apply_ban_list(logits, bans)
        self._assert_only_banned_changed(logits, before)
        return logits

    @staticmethod
    def _banned_cols(logits_row):
        return set(torch.nonzero(torch.isinf(logits_row)).flatten().tolist())

    @classmethod
    def _assert_only_banned_changed(cls, logits, before):
        # Every non-banned cell must still hold its original (non-zero) value;
        # banned cells become -inf.
        banned = torch.isinf(logits)
        assert torch.equal(logits[~banned], before[~banned]), (
            "a logit outside the banned set was modified"
        )

    def test_bigram_repeat_banned(self):
        # Sequence [1, 2, 3, 1, 2] with n=2: prefix [2]; existing bigram
        # (2, 3) -> ban 3. The trailing (1, 2) is the prefix itself.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2])
        logits = self._run([req], [2])
        assert self._banned_cols(logits[0]) == {3}

    def test_trigram_repeat_banned(self):
        # Sequence [1, 2, 3, 4, 1, 2] with n=3: prefix [1, 2]; existing
        # trigram (1, 2, 3) -> ban 3.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 4, 1, 2])
        logits = self._run([req], [3])
        assert self._banned_cols(logits[0]) == {3}

    def test_multiple_matches_all_banned(self):
        # Prefix [2] occurs twice with different continuations -> ban both.
        req = self.MockLlmRequest(tokens=[2, 3, 2, 5, 2])
        logits = self._run([req], [2])
        assert self._banned_cols(logits[0]) == {3, 5}

    def test_no_match_nothing_banned(self):
        req = self.MockLlmRequest(tokens=[1, 2, 3, 4])
        logits = self._run([req], [3])
        assert self._banned_cols(logits[0]) == set()

    def test_sequence_shorter_than_ngram(self):
        req = self.MockLlmRequest(tokens=[1, 2])
        logits = self._run([req], [3])
        assert self._banned_cols(logits[0]) == set()

    def test_unigram_bans_all_seen_tokens(self):
        req = self.MockLlmRequest(tokens=[4, 7, 4])
        logits = self._run([req], [1])
        assert self._banned_cols(logits[0]) == {4, 7}

    @pytest.mark.parametrize("disabled_size", [None, 0])
    def test_disabled_request_untouched(self, disabled_size):
        # Both None and 0 disable the restriction for that request.
        active = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2])
        disabled = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2])
        logits = self._run([active, disabled], [2, disabled_size])
        assert self._banned_cols(logits[0]) == {3}
        assert self._banned_cols(logits[1]) == set()

    def test_incremental_cache_updates_with_growing_context(self):
        # The per-request index must extend as the token history grows.
        # Reuse one sampler so the incremental cache carries across calls.
        handler = SynchronousTokenBanHandler()
        req = self.MockLlmRequest(tokens=[1, 2, 3])
        logits = self._run([req], [2], handler=handler)
        assert self._banned_cols(logits[0]) == set()
        req._tokens.extend([1, 2])  # now [1, 2, 3, 1, 2]: bigram (2, 3) -> ban 3
        logits = self._run([req], [2], handler=handler)
        assert self._banned_cols(logits[0]) == {3}

    def test_cache_rebuilt_on_history_rollback(self):
        # A shrunken history (speculative rollback) must invalidate the cache.
        # Reuse one sampler so the stale cache would be hit if not rebuilt.
        handler = SynchronousTokenBanHandler()
        req = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2])
        logits = self._run([req], [2], handler=handler)
        assert self._banned_cols(logits[0]) == {3}
        req._tokens[:] = [4, 5]
        logits = self._run([req], [2], handler=handler)
        assert self._banned_cols(logits[0]) == set()

    def test_multi_step_rows_all_banned(self):
        # Speculative steps share the (host-approximated) banned set.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2])
        logits = self._run([req], [2], num_steps=[2])
        assert self._banned_cols(logits[0]) == {3}
        assert self._banned_cols(logits[1]) == {3}

    def test_multi_beam_uses_per_beam_history(self):
        # Beam histories diverge, so each beam's row is banned from its own
        # n-grams; the shared per-request cache must not be used here.
        req = self.MultiBeamMockLlmRequest(
            beam_tokens=[
                [1, 2, 3, 1, 2],  # beam 0: bigram (2, 3) -> ban 3
                [4, 5, 6, 4, 5],  # beam 1: bigram (5, 6) -> ban 6
            ]
        )
        # A beam-major / step-minor row layout with num_beams=2, num_steps=1
        # yields one row per beam.
        logits = self._run([req], [2], num_beams=[2])
        assert self._banned_cols(logits[0]) == {3}
        assert self._banned_cols(logits[1]) == {6}

    # --- Overlap-scheduler (stale-host) path -------------------------------
    #
    # The host token list lags the device state by one token, read from
    # new_tokens_cuda[0, seq_slot, 0] on the GPU.

    NUM_SLOTS = 4

    def _new_tokens_cuda(self, slot_tokens):
        buf = torch.full((1, self.NUM_SLOTS, 1), -1, dtype=torch.int32, device="cuda")
        for slot, tok in slot_tokens.items():
            buf[0, slot, 0] = tok
        return buf

    def _run_stale(self, requests, ngram_sizes, stale_by_one, slot_tokens, handler=None):
        total_rows = len(requests)
        logits = self._distinct_logits(total_rows, self.VOCAB)
        before = logits.clone()
        handler = handler or OverlappedTokenBanHandler()
        num_steps = [1] * len(requests)
        num_beams = [1] * len(requests)
        bans = handler.generate_ban_list(
            cast(list[LlmRequest], requests),
            num_steps,
            num_beams,
            ngram_sizes,
            stale_by_one=stale_by_one,
        )
        handler.apply_ban_list(logits, bans, new_tokens_cuda=self._new_tokens_cuda(slot_tokens))
        self._assert_only_banned_changed(logits, before)
        return logits

    def test_stale_bigram_device_hit(self):
        # True sequence [1, 2, 3, 1] + device 2 == [1, 2, 3, 1, 2], n=2:
        # window (2, 3) matches the prefix [d=2] -> ban 3.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 1], seq_slot=0)
        logits = self._run_stale([req], [2], [True], {0: 2})
        assert self._banned_cols(logits[0]) == {3}

    def test_stale_bigram_device_miss(self):
        # Device token 5 never occurred before -> nothing banned.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 1], seq_slot=0)
        logits = self._run_stale([req], [2], [True], {0: 5})
        assert self._banned_cols(logits[0]) == set()

    def test_stale_trigram_host_and_device_hit(self):
        # True sequence [1, 2, 3, 4, 1] + device 2, n=3: prefix [1, 2];
        # window (1, 2, 3) -> ban 3.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 4, 1], seq_slot=1)
        logits = self._run_stale([req], [3], [True], {1: 2})
        assert self._banned_cols(logits[0]) == {3}

    def test_stale_trigram_host_miss(self):
        # Prefix head is [4]; no earlier window starts with 4 followed by the
        # device token -> nothing banned.
        req = self.MockLlmRequest(tokens=[1, 2, 3, 4], seq_slot=1)
        logits = self._run_stale([req], [3], [True], {1: 5})
        assert self._banned_cols(logits[0]) == set()

    def test_stale_window_ending_at_device_token(self):
        # True sequence [7, 7] + device 7, n=2: window (7, 7) ends at the
        # device token; the match forces d == context[-1] -> ban 7.
        req = self.MockLlmRequest(tokens=[7, 7], seq_slot=0)
        logits = self._run_stale([req], [2], [True], {0: 7})
        assert self._banned_cols(logits[0]) == {7}

    def test_stale_unigram_bans_host_and_device_tokens(self):
        req = self.MockLlmRequest(tokens=[4, 7], seq_slot=2)
        logits = self._run_stale([req], [1], [True], {2: 9})
        assert self._banned_cols(logits[0]) == {4, 7, 9}

    def test_stale_and_fresh_requests_mixed(self):
        stale_req = self.MockLlmRequest(tokens=[1, 2, 3, 1], seq_slot=0)
        fresh_req = self.MockLlmRequest(tokens=[1, 2, 3, 1, 2], seq_slot=1)
        logits = self._run_stale([stale_req, fresh_req], [2, 2], [True, False], {0: 2})
        assert self._banned_cols(logits[0]) == {3}
        assert self._banned_cols(logits[1]) == {3}

    def test_stale_no_nan_on_duplicate_bans(self):
        # Duplicate bans on one cell must combine to -inf, not NaN: bigrams
        # starting with 2 continue with 3 (twice) and with 2 (window at d).
        req = self.MockLlmRequest(tokens=[2, 3, 2, 3, 2], seq_slot=0)
        logits = self._run_stale([req], [2], [True], {0: 2})
        assert self._banned_cols(logits[0]) == {2, 3}
        assert not torch.isnan(logits).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestAddMinLengthBans:
    """Unit tests for the min-length EOS-suppression path of TokenBanHandler.

    EOS (the request's original end_id) is banned on every step whose generated
    length is still below py_min_length; it is an unconditional ban with no
    stale-host variant.
    """

    VOCAB = 16
    END_ID = 5

    class MockLlmRequest:
        """Minimal stub exposing the attributes the min-length path reads."""

        def __init__(self, num_tokens, *, min_length=None, prompt_len=0, end_id=5):
            self._num_tokens = num_tokens  # total tokens (prompt + generated)
            self.py_min_length = [min_length] if min_length is not None else None
            self.py_orig_prompt_len = prompt_len
            self.py_original_end_id = end_id
            self.py_end_id = end_id

        def get_num_tokens(self, beam_idx):
            return self._num_tokens

    @staticmethod
    def _distinct_logits(total_rows, vocab):
        return (torch.arange(total_rows * vocab, device="cuda", dtype=torch.float) + 1.0).reshape(
            total_rows, vocab
        )

    @staticmethod
    def _banned_cols(logits_row):
        return set(torch.nonzero(torch.isinf(logits_row)).flatten().tolist())

    def _run(self, requests, num_steps=None, num_beams=None):
        num_steps = num_steps or [1] * len(requests)
        num_beams = num_beams or [1] * len(requests)
        total_rows = sum(s * b for s, b in zip(num_steps, num_beams, strict=True))
        logits = self._distinct_logits(total_rows, self.VOCAB)
        before = logits.clone()
        ngram_sizes: list[int | None] = [None] * len(requests)
        handler = SynchronousTokenBanHandler()
        bans = handler.generate_ban_list(
            cast(list[LlmRequest], requests), num_steps, num_beams, ngram_sizes
        )
        handler.apply_ban_list(logits, bans)
        # Only banned cells (-inf) changed; everything else keeps its value.
        banned = torch.isinf(logits)
        assert torch.equal(logits[~banned], before[~banned])
        return logits

    def test_below_min_length_bans_eos(self):
        # 2 generated tokens (num_tokens 2, prompt 0), min_length 5 -> ban EOS.
        req = self.MockLlmRequest(num_tokens=2, min_length=5)
        logits = self._run([req])
        assert self._banned_cols(logits[0]) == {self.END_ID}

    def test_at_min_length_no_ban(self):
        # 5 generated tokens, min_length 5 -> already satisfied, nothing banned.
        req = self.MockLlmRequest(num_tokens=5, min_length=5)
        logits = self._run([req])
        assert self._banned_cols(logits[0]) == set()

    def test_generated_length_excludes_prompt(self):
        # num_tokens 7 with prompt_len 4 -> only 3 generated < min_length 5 -> ban.
        req = self.MockLlmRequest(num_tokens=7, min_length=5, prompt_len=4)
        logits = self._run([req])
        assert self._banned_cols(logits[0]) == {self.END_ID}

    def test_no_min_length_untouched(self):
        active = self.MockLlmRequest(num_tokens=1, min_length=5)
        disabled = self.MockLlmRequest(num_tokens=1, min_length=None)
        logits = self._run([active, disabled])
        assert self._banned_cols(logits[0]) == {self.END_ID}
        assert self._banned_cols(logits[1]) == set()

    def test_invalid_end_id_skipped(self):
        # end_id <= -1 (e.g. ignore_eos) -> nothing to suppress.
        req = self.MockLlmRequest(num_tokens=1, min_length=5, end_id=-1)
        logits = self._run([req])
        assert self._banned_cols(logits[0]) == set()

    def test_multi_step_stops_at_min_length(self):
        # 3 generated, min_length 5, 3 speculative steps: gen+step < 5 for
        # steps 0,1 (3,4) but not step 2 (5) -> ban only rows 0 and 1.
        req = self.MockLlmRequest(num_tokens=3, min_length=5)
        logits = self._run([req], num_steps=[3])
        assert self._banned_cols(logits[0]) == {self.END_ID}
        assert self._banned_cols(logits[1]) == {self.END_ID}
        assert self._banned_cols(logits[2]) == set()
