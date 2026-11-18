# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for tensorrt_llm.inputs.prefix_token_cache.

The cache splices a cached prefix's token ids with a freshly tokenized tail, so
the property that matters is that its output is identical to tokenizing the
whole prompt. The stub tokenizers below have context-sensitive boundaries, so
the seam-straddling case a char-level stub could never expose is exercised.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest

from tensorrt_llm.inputs import prefix_token_cache
from tensorrt_llm.inputs.prefix_token_cache import (
    PrefixTokenCache,
    PrefixTokenCacheConfig,
    _common_prefix_len,
    create_prefix_token_cache,
)
from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _tok_id(token: str) -> int:
    """Stable across processes, unlike hash() on str."""
    return int.from_bytes(token.encode(), "big") % 100_000 + 1


class _MergeTokenizer:
    """Char-level tokenizer with two-char merges.

    Merges make tokenization context-sensitive at a split point: a tail
    beginning mid-merge tokenizes differently than the same characters do
    inside the whole string. That is the BPE property the cache must survive.
    """

    is_fast = True
    MERGES = frozenset({"ab", "cd", "th", "he", "in", "er"})

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets, i = [], [], 0
        while i < len(text):
            pair = text[i : i + 2]
            token, width = (pair, 2) if pair in self.MERGES else (text[i], 1)
            ids.append(_tok_id(token))
            offsets.append((i, i + width))
            i += width
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self(text)["input_ids"]


class _UnsyncableTokenizer:
    """Pathological: every id encodes the token's absolute position.

    A tail tokenized in isolation restarts its position counter, so its ids can
    never equal the cached prefix's ids over the resync span. The cache must
    detect this and tokenize the prompt in full.
    """

    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = [], []
        for i in range(0, len(text), 2):
            token = text[i : i + 2]
            ids.append(_tok_id(token) + 1000 * len(ids))
            offsets.append((i, i + len(token)))
        return {"input_ids": ids, "offset_mapping": offsets}


class _ExpandingTokenizer:
    """Char-level, except that "ﬀ" becomes two equal tokens with the same offsets.

    As with a byte-fallback character or an NFKC ligature, a split can land on
    the second token of the character, and a long run of them makes the tail
    agree with the cached ids however it is misaligned.
    """

    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = [], []
        for i, char in enumerate(text):
            for _ in range(2 if char == "ﬀ" else 1):
                ids.append(_tok_id(char))
                offsets.append((i, i + 1))
        return {"input_ids": ids, "offset_mapping": offsets}


class _PrependingTokenizer:
    """Prepends a space to every text, as a ``Prepend("▁")`` normalizer does.

    Runs of spaces become tokens of up to four spaces, counted from the start
    of the run, so a tail split inside a run is shifted by the prepended space
    and still agrees with the cached ids until the end of the run.
    """

    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        normalized = " " + text
        ids, offsets, i = [], [], 0
        while i < len(normalized):
            j = i + 1
            if normalized[i] == " ":
                while j < len(normalized) and j - i < 4 and normalized[j] == " ":
                    j += 1
            ids.append(_tok_id(normalized[i:j]))
            # The prepended space maps to the start of the text.
            offsets.append((max(i - 1, 0), j - 1))
            i = j
        return {"input_ids": ids, "offset_mapping": offsets}


class _RejectingTokenizer(_MergeTokenizer):
    """Rejects a lone surrogate, as HF tokenizers do."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        if "\ud800" in text:
            raise TypeError("cannot tokenize a lone surrogate")
        return super().__call__(text, **kwargs)


_MIN_CHARS = 16
# A shared opening longer than min_chars, so growing prompts land in the same
# bucket and the prefix lookup can find them.
_PREAMBLE = "the cabinet had inner thread. " * 4


def _cache(**overrides: int) -> PrefixTokenCache:
    config = dict(max_entries=64, overlap=4, resync=2, min_chars=_MIN_CHARS, checkpoint_tokens=8)
    return PrefixTokenCache(PrefixTokenCacheConfig(**{**config, **overrides}))


def _turns(n: int = 12, preamble: str = _PREAMBLE) -> list[str]:
    """Prompts that grow by a small delta, as multi-turn serving does."""
    text = preamble
    out = []
    for i in range(n):
        # Includes non-ASCII so character offsets are exercised too.
        text += f"turn {i}: thé cd inner answer aber {i} thereabouts. "
        out.append(text)
    return out


def _assert_checkpoints_are_valid(cache: PrefixTokenCache, tokenizer: Any) -> None:
    """Checkpoint i is the first token of the character holding token i * step."""
    step = cache._config.checkpoint_tokens
    for entry in cache._entries.values():
        enc = tokenizer(entry.text)
        assert entry.ids.tolist() == enc["input_ids"]
        starts = [start for start, _ in enc["offset_mapping"]]
        tokens = entry.checkpoint_token_indices.tolist()
        chars = entry.checkpoint_chars.tolist()
        assert len(tokens) == len(chars) == -(-len(starts) // step)
        assert tokens == sorted(tokens) and chars == sorted(chars)
        for i, (token, char) in enumerate(zip(tokens, chars)):
            assert token <= i * step
            assert starts[token] == starts[i * step] == char
            assert token == 0 or starts[token - 1] != char


def _conversations(n: int) -> list[str]:
    """First turns of ``n`` unrelated conversations."""
    return [_turns(1, preamble=f"conversation {i:03d}: {_PREAMBLE}")[0] for i in range(n)]


def test_multi_turn_growth_is_identical_to_full_tokenization() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    for prompt in _turns():
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    # The whole point: later turns must actually reuse a cached prefix.
    assert cache.hits > 0
    assert cache.resync_failures == 0
    # Each turn replaces the entry it extended, so the conversation is one entry.
    assert len(cache._entries) == 1


def test_interleaved_unrelated_prompts_are_correct() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    unrelated = "a wholly different opening that shares no prefix at all. " * 3
    for prompt in _turns(6):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
        assert cache.encode(tokenizer, unrelated) == tokenizer(unrelated)["input_ids"]


def test_resync_failure_falls_back_to_full_tokenization() -> None:
    tokenizer, cache = _UnsyncableTokenizer(), _cache()
    for prompt in _turns():
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    # The fallback must have been exercised, else this proves nothing.
    assert cache.resync_failures > 0


@pytest.mark.parametrize("tokenizer", [_MergeTokenizer(), _UnsyncableTokenizer()])
def test_rewritten_history_is_identical_to_full_tokenization(tokenizer: Any) -> None:
    """A turn that rewrites the end of the history splices from the common prefix."""
    cache = _cache()
    turns = _turns(8)
    for i, prompt in enumerate(turns):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
        # Trim the last message and append a different one, as agents do.
        rewritten = prompt[: len(prompt) - 20] + f"rewritten tail {i} there. "
        hits, failures = cache.hits, cache.resync_failures
        assert cache.encode(tokenizer, rewritten) == tokenizer(rewritten)["input_ids"]
        if isinstance(tokenizer, _MergeTokenizer):
            assert cache.hits == hits + 1
        else:
            # Spliced from the history and rejected at the seam.
            assert cache.resync_failures == failures + 1
    if isinstance(tokenizer, _MergeTokenizer):
        assert cache.divergent_hits >= len(turns)
        # The rewritten branch does not evict the history it diverged from.
        assert any(entry.text == turns[-1] for entry in cache._entries.values())
    else:
        assert cache.divergent_hits == 0
    assert not cache.disabled
    _assert_checkpoints_are_valid(cache, tokenizer)


def test_a_deep_edit_splices_from_a_late_checkpoint() -> None:
    """A long history edited deep inside is spliced across several compare blocks."""
    tokenizer, cache = _MergeTokenizer(), _cache()
    history = _turns(400)[-1][:10_000]
    assert cache.encode(tokenizer, history) == tokenizer(history)["input_ids"]
    edited = history[:7000] + "#" + history[7001:]
    found = cache._lookup(edited)
    assert found is not None and not found.extends
    # One checkpoint interval (8 tokens of at most 2 chars) before the edit, at most.
    assert 7000 - 3 * 2 * 8 <= found.split_char < 7000
    assert cache.encode(tokenizer, edited) == tokenizer(edited)["input_ids"]
    assert cache.divergent_hits == 1 and cache.resync_failures == 0
    assert not cache.disabled
    _assert_checkpoints_are_valid(cache, tokenizer)


def test_a_rewritten_turn_splices_from_its_history_not_a_shorter_shared_opening() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    opening = _PREAMBLE
    history = _turns(20)[-1]
    # Another conversation that so far holds only the shared opening.
    for prompt in (history, opening):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    rewritten = history[:-20] + "a rewritten ending. "
    # The rewritten turn extends the opening's entry too.
    assert any(entry.text == opening for entry in cache._entries.values())
    found = cache._lookup(rewritten)
    assert found is not None and found.entry.text == history and not found.extends
    divergent_hits = cache.divergent_hits
    assert cache.encode(tokenizer, rewritten) == tokenizer(rewritten)["input_ids"]
    assert cache.divergent_hits == divergent_hits + 1


def test_a_tail_shifted_inside_a_run_of_spaces_is_rejected() -> None:
    """A shift the resync span cannot see is caught at the next checkpoint."""
    tokenizer, cache = _PrependingTokenizer(), _cache()
    run_end = len(_PREAMBLE) + 400
    history = _PREAMBLE + " " * 400 + "end of the padded tool output. " * 4
    for kept in range(12):
        cache = _cache()
        assert cache.encode(tokenizer, history) == tokenizer(history)["input_ids"]
        # Diverge just after the run, so the split lands inside it.
        rewritten = history[: run_end + kept] + "#rewritten"
        found = cache._lookup(rewritten)
        assert found is not None and run_end - 100 < found.split_char < run_end
        assert cache.encode(tokenizer, rewritten) == tokenizer(rewritten)["input_ids"]
        assert cache.resync_failures == 1
        assert not cache.disabled


def test_a_tokenizer_that_shifts_every_tail_gets_no_cache() -> None:
    assert create_prefix_token_cache(_PrependingTokenizer()) is None
    for tokenizer in (_MergeTokenizer(), _ExpandingTokenizer(), _UnsyncableTokenizer()):
        assert create_prefix_token_cache(tokenizer) is not None


@pytest.mark.parametrize("mismatch", [0, 1, 4095, 4096, 4097, 6000, 8191, None])
def test_common_prefix_len_finds_the_first_mismatch(mismatch: int | None) -> None:
    a = "".join(chr(97 + (i * 7) % 26) for i in range(8192))
    b = a if mismatch is None else a[:mismatch] + "#" + a[mismatch + 1 :]
    expected = len(a) if mismatch is None else mismatch
    assert _common_prefix_len(a, b) == _common_prefix_len(b, a) == expected


@pytest.mark.parametrize("n", [0, 4095, 4096, 4097, 8192])
def test_common_prefix_len_of_a_prefix_is_its_length(n: int) -> None:
    a = "".join(chr(97 + (i * 7) % 26) for i in range(8192))
    assert _common_prefix_len(a, a[:n]) == _common_prefix_len(a[:n], a) == n


def test_a_split_is_moved_to_the_first_token_of_a_multi_token_character() -> None:
    tokenizer, cache = _ExpandingTokenizer(), _cache()
    text = _PREAMBLE
    for i in range(12):
        # The odd character shifts every later split onto the other token of a pair.
        text += "ﬀ" * 40 + "." * (i % 2)
        assert cache.encode(tokenizer, text) == tokenizer(text)["input_ids"]
        rewritten = text[:-10] + "rewritten. "
        assert cache.encode(tokenizer, rewritten) == tokenizer(rewritten)["input_ids"]
    # The splits are usable, not rejected at the seam.
    assert cache.resync_failures == 0
    assert cache.divergent_hits > 0
    assert not cache.disabled
    _assert_checkpoints_are_valid(cache, tokenizer)


def test_short_prompts_are_tokenized_but_not_cached() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = "the cab"
    assert len(prompt) < _MIN_CHARS
    assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert not cache._entries
    assert cache.hits == cache.misses == 0


def test_repeated_prompt_is_not_stored_twice() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = _turns(1)[0]
    cache.encode(tokenizer, prompt)
    cache.encode(tokenizer, prompt)
    assert len(cache._entries) == 1
    assert cache.hits == 1


def test_conversations_sharing_a_long_preamble_find_their_own_entry() -> None:
    """The probe must tell apart entries the bucket key and last char cannot.

    Agentic prompts share a system preamble far longer than min_chars and end
    with the same chat-template suffix.
    """
    tokenizer, cache = _MergeTokenizer(), _cache()
    preamble = "shared system prompt and tool schema. " * 40
    suffix = "<|assistant|>\n"
    turns = {i: preamble + f"user {i} says hi.{suffix}" for i in range(16)}
    for prompt in turns.values():
        cache.encode(tokenizer, prompt)
    assert len(cache._buckets) == 1
    hits_before = cache.hits
    for i, prompt in turns.items():
        # The next turn keeps the previous prompt, generation suffix included.
        extended = prompt + f"reply to {i}.<|user|>\nmore from {i}.{suffix}"
        found = cache._lookup(extended)
        assert found is not None and found[1].text == prompt
        assert cache.encode(tokenizer, extended) == tokenizer(extended)["input_ids"]
    assert cache.hits - hits_before == 16 and cache.resync_failures == 0


def test_rewritten_end_of_cached_prompt_reuses_its_prefix() -> None:
    """A chat template may replace the previous turn's generation prefix."""
    tokenizer, cache = _MergeTokenizer(), _cache()
    turn = _turns(1)[0]
    previous = turn + "<ge>"
    current = turn + "<gex reply text>"
    assert not current.startswith(previous)
    assert cache.encode(tokenizer, previous) == tokenizer(previous)["input_ids"]
    assert cache.encode(tokenizer, current) == tokenizer(current)["input_ids"]
    assert cache.hits == 1 and cache.resync_failures == 0


def test_eviction_bounds_entry_count() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache(max_entries=8)
    for prompt in _conversations(40):
        cache.encode(tokenizer, prompt)
    assert len(cache._entries) == 8
    assert sum(len(b) for b in cache._buckets.values()) == 8


def test_eviction_bounds_total_chars() -> None:
    prompts = _conversations(40)
    budget = len(prompts[0]) * 3
    tokenizer, cache = _MergeTokenizer(), _cache(max_total_chars=budget)
    for prompt in prompts:
        cache.encode(tokenizer, prompt)
        assert cache._total_chars <= budget
    assert len(cache._entries) == 3


def test_eviction_is_least_recently_used() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache(max_entries=2)
    hot, cold, unrelated = _conversations(3)
    cache.encode(tokenizer, hot)
    cache.encode(tokenizer, cold)
    # Touch the older entry, then insert a third: the untouched one must go.
    assert cache.encode(tokenizer, hot) == tokenizer(hot)["input_ids"]
    cache.encode(tokenizer, unrelated)
    texts = {entry.text for entry in cache._entries.values()}
    assert hot in texts
    assert cold not in texts


def test_concurrent_encode_is_correct() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompts = _turns(16)
    expected = {p: tokenizer(p)["input_ids"] for p in prompts}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for prompt in prompts:
                assert cache.encode(tokenizer, prompt) == expected[prompt]
        except BaseException as exc:  # surface in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_hit_rate_is_logged_periodically() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompts = _turns(8)
    with (
        patch.object(prefix_token_cache, "LOG_EVERY_REQUESTS", 4),
        patch.object(prefix_token_cache.logger, "info") as info,
    ):
        for prompt in prompts:
            cache.encode(tokenizer, prompt)
    assert info.call_count == 2
    assert "hits (" in info.call_args.args[0]
    assert "resync fallbacks" in cache.summary()


def test_error_disables_cache_and_still_returns_ids() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = _turns(1)[0]
    with patch.object(PrefixTokenCache, "_encode", side_effect=RuntimeError("boom")):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert cache.disabled
    assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert not cache._entries


def test_a_prompt_the_tokenizer_rejects_fails_without_disabling_the_cache() -> None:
    tokenizer, cache = _RejectingTokenizer(), _cache()
    turns = _turns(3)
    cache.encode(tokenizer, turns[0])
    with pytest.raises(TypeError):
        cache.encode(tokenizer, turns[1] + "\ud800")
    assert not cache.disabled
    assert cache.encode(tokenizer, turns[2]) == tokenizer(turns[2])["input_ids"]
    assert cache.hits == 1


def test_checkpoints_must_be_wider_than_the_resync_span() -> None:
    with pytest.raises(ValueError, match="checkpoint_tokens"):
        PrefixTokenCacheConfig(resync=32, checkpoint_tokens=32)


def test_config_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", "7")
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", "100")
    config = PrefixTokenCacheConfig.from_env()
    assert config.max_entries == 7
    assert config.min_chars == 100
    assert config.overlap == PrefixTokenCacheConfig.overlap


@pytest.mark.parametrize("value", ["abc", "-1", "0", ""])
def test_config_from_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", value)
    with pytest.raises(ValueError, match="TLLM_PREFIX_TOKEN_CACHE_ENTRIES"):
        PrefixTokenCacheConfig.from_env()


def test_create_returns_none_for_slow_tokenizer() -> None:
    class Slow:
        is_fast = False

    assert create_prefix_token_cache(Slow()) is None
    assert create_prefix_token_cache(None) is None


def test_create_raises_on_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", "many")
    with pytest.raises(ValueError):
        create_prefix_token_cache(_MergeTokenizer())


# --- DefaultInputProcessor integration --------------------------------------


def _processor(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> DefaultInputProcessor:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", str(_MIN_CHARS))
    return DefaultInputProcessor(None, None, _MergeTokenizer(), enable_tokenization_cache=enabled)


def test_processor_uses_cache_for_eligible_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch)
    cache = processor._prefix_token_cache
    assert cache is not None
    params = SamplingParams(add_special_tokens=False)
    for prompt in _turns(4):
        ids, extra = processor({"prompt": prompt}, params)
        assert ids == _MergeTokenizer()(prompt)["input_ids"]
        assert extra is None
    assert cache.hits > 0


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(add_special_tokens=True),
        SamplingParams(add_special_tokens=False, truncate_prompt_tokens=8),
    ],
)
def test_processor_bypasses_cache_when_arguments_alter_tokenization(
    monkeypatch: pytest.MonkeyPatch, params: SamplingParams
) -> None:
    processor = _processor(monkeypatch)
    cache = processor._prefix_token_cache
    prompt = _turns(1)[0]
    ids, _ = processor({"prompt": prompt}, params)
    assert ids == _MergeTokenizer().encode(prompt)
    assert cache.hits == cache.misses == 0


def test_processor_bypasses_cache_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, enabled=False)
    assert processor._prefix_token_cache is None
    prompt = _turns(1)[0]
    ids, _ = processor({"prompt": prompt}, SamplingParams(add_special_tokens=False))
    assert ids == _MergeTokenizer().encode(prompt)


# --- Qwen2-VL family (Qwen3-VL, Qwen3.5) text-only path ----------------------


class _NoSpecialTokensMergeTokenizer(_MergeTokenizer):
    def __init__(self, num_special: int = 0) -> None:
        self.num_special = num_special

    def num_special_tokens_to_add(self) -> int:
        return self.num_special


def _qwen2vl_processor(monkeypatch: pytest.MonkeyPatch, tokenizer: Any):
    from types import SimpleNamespace

    from tensorrt_llm._torch.models import modeling_qwen2vl

    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", str(_MIN_CHARS))
    monkeypatch.setattr(
        modeling_qwen2vl.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object()
    )
    config = SimpleNamespace(torch_dtype=None, vision_config=SimpleNamespace(temporal_patch_size=2))
    return modeling_qwen2vl.Qwen2VLInputProcessorBase(
        "unused", config, tokenizer, enable_tokenization_cache=True
    )


# Default SamplingParams add special tokens, a no-op for this tokenizer.
@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_qwen2vl_processor_text_prompts_use_the_cache(
    monkeypatch: pytest.MonkeyPatch, add_special_tokens: bool
) -> None:
    processor = _qwen2vl_processor(monkeypatch, _NoSpecialTokensMergeTokenizer())
    assert type(processor).supports_tokenization_cache
    cache = processor._prefix_token_cache
    assert cache is not None
    params = SamplingParams() if add_special_tokens else SamplingParams(add_special_tokens=False)
    for prompt in _turns(4):
        ids, extra = processor.call_with_text_prompt({"prompt": prompt}, params)
        assert ids == _MergeTokenizer()(prompt)["input_ids"]
        assert extra is None
    assert cache.hits > 0


def test_qwen2vl_processor_skips_the_cache_for_special_token_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _qwen2vl_processor(monkeypatch, _NoSpecialTokensMergeTokenizer(num_special=1))
    assert processor._prefix_token_cache is None
