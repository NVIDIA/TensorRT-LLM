# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the engine reused from its KV cache, read from its responses, and a probe that asks it to.

Every response of the PyTorch executor carries ``cached_tokens``: how many of
the request's prompt tokens the engine resumed from the KV cache rather than
computed. A hybrid model's manager resumes a prompt only where it holds a
snapshot of the recurrent state, so for such a model a request that resumed
at token ``p`` restored the recurrent-state snapshot taken at ``p``. The
responses come from rank 0's executor at every world size, so the reading is
the same under ``trtllm-llmapi-launch``.

:class:`ReuseLedger` keeps one entry per request, and :func:`emit` prints one
``MV2FLOW_REUSE`` record (:data:`.workload_io.REUSE_MARKER`) after the workload:

* ``requests``: requests read; ``reported``: how many of their responses
  carried ``cached_tokens`` at all. A tree whose responses do not carry it
  reports 0, and nothing is judged from the record.
* ``hits`` and ``reused_tokens``: the requests that resumed from the cache, and
  the prompt tokens they did not compute. ``resumed_at``: the positions they
  resumed at, the most common first, with how many resumed at each.
* ``min_resume``: the shortest prefix this engine resumes from
  (:func:`min_resume`); ``resumable``: the requests whose prompt shared more
  than ``min_resume`` tokens with an earlier request's, which an engine that
  reuses could have resumed. ``concurrency``: the most requests the workload
  had in flight at once, so a gate can tell whether two requests that share a
  prefix ran one after the other.
* ``probe``: the reuse probe's outcome, when the workload asked for one
  (:func:`run_probe`).

Standard library only at import time; ``tensorrt_llm`` is imported only to pin
the probe to one attention-DP rank.
"""

from __future__ import annotations

import bisect
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402 -- imported by path, see the package docstring
import workload_io as wio  # noqa: E402

#: ``KvCacheConfig.tokens_per_block``'s default.
DEFAULT_TOKENS_PER_BLOCK = 32
#: How many resume positions a record lists, the most common first.
RESUMED_AT_SHOWN = 8

#: Prose the probe's prompt is cut from: tokens a tokenizer makes of plain
#: text, never a special or an image-placeholder token.
PROBE_TEXT = (
    "The river ran slowly past the mill, and the miller counted the sacks of grain "
    "that the farmers had brought in from the valley. Each sack was weighed twice, "
    "once when it arrived and once before it was ground, and the two numbers were "
    "written side by side in a ledger that had been kept in the same hand for years."
)


def cached_tokens(output: Any) -> int | None:
    """The prompt tokens one request resumed from the cache; ``None`` when its output does not say."""
    value = getattr(output, "cached_tokens", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def prompt_ids(output: Any) -> tuple[Any, ...] | None:
    """One request's prompt token ids, as a tuple; ``None`` when its output does not carry them."""
    ids = getattr(output, "prompt_token_ids", None)
    if ids is None:
        return None
    try:
        return tuple(ids)
    except TypeError:
        return None


def common_prefix(a: Sequence[Any], b: Sequence[Any]) -> int:
    """How many leading tokens ``a`` and ``b`` share."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def shared_prefixes(prompts: Sequence[tuple[Any, ...] | None]) -> list[int | None]:
    """For each prompt, the longest prefix it shares with any earlier prompt of ``prompts``.

    The earlier prompts are kept sorted, so the longest match is the prompt's
    neighbour on one side or the other of where it would be inserted.
    """
    earlier: list[tuple[Any, ...]] = []
    shared: list[int | None] = []
    for prompt in prompts:
        if prompt is None:
            shared.append(None)
            continue
        at = bisect.bisect_left(earlier, prompt)
        neighbours = earlier[max(0, at - 1) : at + 1]
        shared.append(max((common_prefix(prompt, n) for n in neighbours), default=0))
        earlier.insert(at, prompt)
    return shared


def min_resume(policy: Mapping[str, Any] | None, tokens_per_block: int) -> int | None:
    """The shortest prefix an engine with this snapshot policy resumes a prompt from.

    With a snapshot policy, the first point where every prompt long enough
    gets a snapshot: the smallest offset from the start, or the interval.
    Without one, one KV block. A policy of offsets from the end alone puts
    each prompt's snapshots at distances from its own end, so no one length
    answers: ``None``.

    For a model without recurrent state a policy changes nothing and the
    engine resumes at any block. The policy's first point is then larger
    than the engine's own granularity, which counts fewer requests resumable,
    never more.
    """
    policy = policy or {}
    points = [int(o) for o in policy.get("additional_snapshot_offsets_from_start") or () if o > 0]
    interval = int(policy.get("periodic_snapshot_interval") or 0)
    if interval > 0:
        points.append(interval)
    if points:
        return min(points)
    if any(int(o) > 0 for o in policy.get("additional_snapshot_offsets_from_end") or ()):
        return None
    return max(1, int(tokens_per_block))


def probe_tokens(policy: Mapping[str, Any] | None, tokens_per_block: int) -> int:
    """The probe's prompt length: two KV blocks past the first point it can resume at.

    For a policy of offsets from the end alone, the smallest offset: an
    identical second prompt can then resume two blocks in.
    """
    block = max(1, int(tokens_per_block))
    first = min_resume(policy, block)
    if first is None:
        first = min(
            int(o)
            for o in (policy or {}).get("additional_snapshot_offsets_from_end") or ()
            if o > 0
        )
    return first + 2 * block


#: ``(snapshot policy, tokens per block)``: how finely an engine resumes a prompt.
Granularity = Tuple[Optional[Dict[str, Any]], int]


def engine_granularity(llm: Any) -> Granularity:
    """``(snapshot policy, tokens per block)`` of ``llm``'s engine, read while it is up."""
    kv = getattr(getattr(llm, "args", None), "kv_cache_config", None)
    policy = None
    state = getattr(kv, "mamba_state_config", None)
    if state is not None:
        try:
            policy = engine_facts.snapshot_policy(state)
        except (AttributeError, TypeError, ValueError):
            policy = None
    block = engine_facts.kv_capacity(llm).get("tokensPerBlock") or getattr(
        kv, "tokens_per_block", None
    )
    try:
        block = int(block) if block else DEFAULT_TOKENS_PER_BLOCK
    except (TypeError, ValueError):
        block = DEFAULT_TOKENS_PER_BLOCK
    return policy, max(1, block)


def max_batch_size(llm: Any) -> int | None:
    """The most requests ``llm``'s engine runs at once, from its args; ``None`` when unknown."""
    value = getattr(getattr(llm, "args", None), "max_batch_size", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class ReuseLedger:
    """One entry per request of a workload: its prompt token ids and the tokens it resumed.

    :meth:`add` reads a finished request's output at once. :meth:`track` keeps
    an output to read when the record is made, for a harness that hands each
    request over as it is submitted and never sees it finish.
    """

    def __init__(self) -> None:
        self._read: list[tuple[tuple[Any, ...] | None, int | None]] = []
        self._tracked: list[Any] = []

    def add(self, output: Any) -> None:
        self._read.append((prompt_ids(output), cached_tokens(output)))

    def track(self, output: Any) -> None:
        self._tracked.append(output)

    def entries(self) -> list[tuple[tuple[Any, ...] | None, int | None]]:
        """Every request, in the order it was added or tracked."""
        return [*self._read, *((prompt_ids(o), cached_tokens(o)) for o in self._tracked)]

    def summary(
        self,
        *,
        min_resume: int | None,
        concurrency: int | None,
        probe: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The ``MV2FLOW_REUSE`` payload; see the module docstring."""
        entries = self.entries()
        reported = [tokens for _, tokens in entries if tokens is not None]
        hits = [tokens for tokens in reported if tokens > 0]
        shared = shared_prefixes([prompt for prompt, _ in entries])
        resumable = None
        if min_resume is not None and any(s is not None for s in shared):
            resumable = sum(1 for s in shared if s is not None and s > min_resume)
        return {
            "requests": len(entries),
            "reported": len(reported),
            "hits": len(hits),
            "reused_tokens": sum(hits),
            "resumed_at": {
                str(position): count
                for position, count in Counter(hits).most_common(RESUMED_AT_SHOWN)
            },
            "min_resume": min_resume,
            "resumable": resumable,
            "concurrency": concurrency,
            "probe": dict(probe) if probe else None,
        }


def _finished(handle: Any) -> Any:
    """The final output of one submitted request."""
    result = getattr(handle, "result", None)
    if callable(result):
        return result()
    last = None
    for last in handle:
        pass
    return last


def _pinned_to_rank0(attention_dp: bool) -> dict[str, Any]:
    """``generate_async`` keyword arguments that keep a request on attention-DP rank 0.

    Each attention-DP rank holds its own KV cache, so the probe's two
    requests must be served by the same one.
    """
    if not attention_dp:
        return {}
    try:
        from tensorrt_llm.scheduling_params import SchedulingParams
    except ImportError:
        return {}
    return {"scheduling_params": SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)}


def probe_prompt(tokenizer: Any, tokens: int) -> list[int]:
    """``tokens`` prompt token ids cut from :data:`PROBE_TEXT`, repeated as often as it takes.

    Without a tokenizer, ids from the low, ordinary range of a vocabulary.
    """
    encode = getattr(tokenizer, "encode", None)
    ids: list[int] = []
    if callable(encode):
        try:
            ids = [int(t) for t in encode(PROBE_TEXT, add_special_tokens=False)]
        except TypeError:
            ids = [int(t) for t in encode(PROBE_TEXT)]
    if not ids:
        ids = [1000 + n for n in range(97)]
    return (ids * (tokens // len(ids) + 1))[:tokens]


def run_probe(
    llm: Any,
    params: Any,
    tokens: int,
    *,
    max_seq_len: int | None = None,
    max_prompt: int | None = None,
    attention_dp: bool = False,
) -> dict[str, Any]:
    """Send one ``tokens``-token prompt twice, the second after the first finished.

    ``params`` are the sampling parameters both use (one generated token is
    enough). The outcome: ``tokens``, ``first`` and ``reused``, the cached
    tokens of each response. An engine that reuses resumes the second at
    :func:`min_resume` or later; one that does not, at 0. ``skipped`` says why
    the probe did not run: the prompt and its token do not fit
    ``max_seq_len``, or the prompt is longer than ``max_prompt`` (an engine
    without chunked prefill takes no prompt over ``max_num_tokens``).
    """
    record: dict[str, Any] = {"tokens": int(tokens)}
    if max_seq_len and tokens + 1 > int(max_seq_len):
        record["skipped"] = f"{tokens} prompt tokens do not fit max_seq_len {max_seq_len}"
        return record
    if max_prompt and tokens > int(max_prompt):
        record["skipped"] = (
            f"{tokens} prompt tokens exceed max_num_tokens {max_prompt} without chunked prefill"
        )
        return record
    prompt = {"prompt_token_ids": probe_prompt(getattr(llm, "tokenizer", None), tokens)}
    pinned = _pinned_to_rank0(attention_dp)
    first = _finished(llm.generate_async(prompt, params, **pinned))
    second = _finished(llm.generate_async(prompt, params, **pinned))
    record.update(first=cached_tokens(first), reused=cached_tokens(second))
    return record


def emit(
    ledger: ReuseLedger,
    granularity: Granularity,
    *,
    concurrency: int | None,
    probe: Mapping[str, Any] | None = None,
) -> None:
    """Print the ``MV2FLOW_REUSE`` record of ``ledger``.

    ``granularity`` is the engine's (:func:`engine_granularity`), read while
    it was up. A record that cannot be assembled is printed as
    ``{"error": ...}``: reading what the engine reused is an observation of a
    workload that has already run, and its failure must not fail the run.
    """
    try:
        policy, block = granularity
        record = ledger.summary(
            min_resume=min_resume(policy, block), concurrency=concurrency, probe=probe
        )
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        record = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    wio.emit(wio.REUSE_MARKER, record)


__all__ = [
    "DEFAULT_TOKENS_PER_BLOCK",
    "Granularity",
    "PROBE_TEXT",
    "RESUMED_AT_SHOWN",
    "ReuseLedger",
    "cached_tokens",
    "common_prefix",
    "emit",
    "engine_granularity",
    "max_batch_size",
    "min_resume",
    "probe_prompt",
    "probe_tokens",
    "prompt_ids",
    "run_probe",
    "shared_prefixes",
]
