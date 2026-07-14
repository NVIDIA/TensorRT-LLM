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
"""Regression test for NVBug 6025177 (CVE-2026-24205).

Cross-request KV cache contamination: with chunked prefill + block reuse
enabled, cancelling a request mid-prefill caused its half-populated KV cache
blocks to be stored in the reuse pool. Subsequent requests that hit those
blocks via prefix caching would return content belonging to the cancelled
request. Fixed in PR #12763.

Strategy
--------
Run repeated rounds of concurrent chat-completion requests against a
``trtllm-serve`` instance configured with the minimum trigger conditions
(chunked prefill + block reuse + per-request prompt larger than
``max_num_tokens`` so each request's prefill spans multiple iterations).

In each round we cancel ~25% of the requests mid-prefill via
``asyncio.Task.cancel`` (which propagates an HTTP disconnect to the
server).

Two probe styles exercise the same failure mode (inspired by the
external number-set probe package: static shared system prefill, large
repeated per-request payload, structured machine-readable answers):

1. **Tag echo** — each request asks the model to echo a unique
   ``TAG_<10 digits>`` from a pool of 100. Contamination if another pool
   tag appears in the completion.
2. **Number set** — each user message repeats one large integer many
   times; the model must reply with ``ALL_SAME:<int>`` or ``UNIQUE:...``
   per the number-set probe rules. Contamination follows the same
   detection logic as ``run_number_set_probe.py`` (wrong pool integer,
   ``UNIQUE`` listing pool integers, or pool integers in freeform text).

Hallucinated matches against the fixed pools are negligible at 10-digit
tags and 10^10-scale integers with pool size 100.
"""

import asyncio
import os
import random
import re

import pytest
from utils.util import skip_num_gpus_less_than, skip_pre_blackwell

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

NUM_DISTINCT_TAGS = 100
TAG_DIGITS = 10
RNG_SEED = 0xC0DECAFE

# Number-set probe: pool range and repeats (see number-set-probe-package).
NUM_POOL_SIZE = 100
NUMBER_MIN = 10_000_000
NUMBER_MAX = 9_999_999_999
REPEATED_NUMBER_COUNT_MIN = 1000
REPEATED_NUMBER_COUNT_MAX = 3000

# Long static prefix — maximises prefix-cache hits across all requests so
# the bug's "shared prefix + per-request suffix" pattern is reproduced.
_FILLER_LINE = "Maintain precision and echo the tag faithfully without modification. "
SYSTEM_PROMPT = (
    "You are a precise echo service. The user will give you a single tag "
    "of the form TAG_<digits>. Reply with exactly that tag on its own "
    "line, with no other text. " + _FILLER_LINE * 60
)

# Shared static system text for number-set probe (same idea as
# system_prompt_1k_tokens.txt in number-set-probe-package, inlined for CI).
_NUMBER_SYSTEM_UNIT = (
    "You are a deterministic number comparison assistant. "
    "Follow the user's instructions exactly. "
    "Reply using only the requested output format. "
)
NUMBER_SYSTEM_PROMPT = _NUMBER_SYSTEM_UNIT * 80

TAG_RE = re.compile(r"TAG_(\d{%d})" % TAG_DIGITS)
_ALL_SAME_RE = re.compile(r"ALL_SAME\s*:\s*(-?\d+)", re.IGNORECASE)
_UNIQUE_RE = re.compile(r"UNIQUE\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _normalize_model_text(text: str) -> str:
    """Strip fences and leading chatter so format checks work on chatty models."""
    normalized = text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    return normalized


def _parse_integers(text: str) -> list[int]:
    """Extract integers from model output (number-set probe convention)."""
    return [int(m) for m in re.findall(r"-?\d+", text)]


def _evaluate_number_set_response(
    number_value: int,
    response_content: str | None,
    number_pool: set[int],
) -> tuple[str, str | None, list[int]]:
    """Classify a completion using the same rules as number-set-probe-package.

    Returns ``(verdict, reason, matched_pool_numbers)`` where ``verdict`` is
    one of ``valid``, ``hallucination_ok``, ``weak_invalid``, ``cross_contamination``.
    """
    if response_content is None:
        return ("weak_invalid", "empty_response", [])

    normalized = _normalize_model_text(response_content)

    all_same = _ALL_SAME_RE.search(normalized)
    if all_same:
        common = int(all_same.group(1))
        if common == number_value:
            return ("valid", None, [])
        if common in number_pool:
            return ("cross_contamination", "wrong_common_number", [common])
        return ("hallucination_ok", None, [])

    unique = _UNIQUE_RE.search(normalized)
    if unique:
        unique_values = sorted(set(_parse_integers(unique.group(1))))
        foreign = [
            value for value in unique_values if value in number_pool and value != number_value
        ]
        if foreign:
            return ("cross_contamination", "unique_numbers_intersect_pool", foreign)
        return ("hallucination_ok", None, [])

    parsed = _parse_integers(normalized)
    foreign = [value for value in parsed if value in number_pool and value != number_value]
    if foreign:
        return ("cross_contamination", "freeform_numbers_intersect_pool", foreign)
    if number_value in parsed:
        return ("valid", None, [])
    return ("weak_invalid", "unexpected_format", [])


def _make_number_pool(pool_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    values: set[int] = set()
    while len(values) < pool_size:
        values.add(rng.randint(NUMBER_MIN, NUMBER_MAX))
    return list(values)


@pytest.fixture(scope="module")
def model_name():
    """Local checkpoint under ``LLM_MODELS_ROOT`` (moonshotai/Kimi-K2.5 NVFP4).

    Same layout as integration tests: ``Kimi-K2.5-NVFP4`` directory.
    """
    return "Kimi-K2.5-NVFP4"


@pytest.fixture(scope="module")
def server(model_name: str):
    """``trtllm-serve`` configured with the minimum settings to trigger
    NVBug 6025177: chunked prefill on (so prefill spans multiple steps)
    and block reuse on (so the contamination surfaces via prefix cache).
    """
    model_path = get_model_path(model_name)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Kimi K2.5 FP4 weights not found at {model_path}; "
            "set LLM_MODELS_ROOT to the parent of Kimi-K2.5-NVFP4"
        )

    extra_config = {
        "enable_chunked_prefill": True,
        "kv_cache_config": {
            "enable_block_reuse": True,
            "free_gpu_memory_fraction": 0.5,
        },
        "max_num_tokens": 512,
        "max_batch_size": 16,
    }
    args = [
        "--backend",
        "pytorch",
        "--max_seq_len",
        "32768",
        "--tp_size",
        "4",
        "--trust_remote_code",
    ]
    with RemoteOpenAIServer(model_path, args, extra_config=extra_config) as remote_server:
        yield remote_server


def _make_messages(tag: str) -> list:
    """Build a chat-completion message list whose user content is large
    enough that prefill spans multiple chunked-prefill iterations (i.e.
    user-content tokens > ``max_num_tokens`` configured on the server).
    """
    # ~3K filler tokens => 6+ chunks of 512 tokens each, so cancellation
    # mid-prefill has a wide window to land in.
    payload = " ".join(["number"] * 3000)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"My tag is {tag}. The list is: {payload}. Echo my tag now."),
        },
    ]


def _make_number_set_messages(number_value: int, repeated_count: int) -> list:
    """User prompt: one integer repeated many times (number-set probe style).

    Maximizes shared prefix across requests; only the repeated integer and
    count vary per request.
    """
    series = ", ".join([str(number_value)] * repeated_count)
    user_content = (
        "I'm going to provide a series of numbers. Tell me if all numbers "
        "are the same. If yes then what is the common number. If not then "
        "give me all the unique numbers.\n"
        "Return only the machine-readable answer using ALL_SAME:<number> or "
        "UNIQUE:<comma-separated unique numbers>. Do not add extra text.\n\n"
        f"In the series:\n{series}"
    )
    return [
        {"role": "system", "content": NUMBER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _send_one(
    client,
    model: str,
    messages: list,
    cancel_after_s,
    *,
    max_tokens: int = 64,
):
    """Send one request. If ``cancel_after_s`` is not None, cancel it
    after that many seconds via ``asyncio.Task.cancel`` — which closes the
    underlying HTTPX connection; ``trtllm-serve`` treats this as a client
    disconnect and aborts the in-flight request.
    """
    if cancel_after_s is None:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    task = asyncio.create_task(
        client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    )
    await asyncio.sleep(cancel_after_s)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return None
    return None


@skip_pre_blackwell
@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
async def test_no_kv_cache_contamination_under_cancel(server: RemoteOpenAIServer, model_name: str):
    """Regression for NVBug 6025177.

    Any single in-pool tag mismatch across all rounds is treated as a
    KV-cache contamination event and fails the test.
    """
    rng = random.Random(RNG_SEED)
    pool = [f"TAG_{rng.randrange(10**TAG_DIGITS):0{TAG_DIGITS}d}" for _ in range(NUM_DISTINCT_TAGS)]
    pool_set = set(pool)

    NUM_ROUNDS = 10
    BATCH = 16
    CANCEL_RATIO = 0.25
    # Tuned to land inside chunked prefill; long user payloads keep prefill
    # busy for tens of ms+ on typical Blackwell setups with Kimi K2.5.
    CANCEL_AFTER_S = 0.05

    client = server.get_async_client(timeout=60.0)

    contamination = []
    completed_count = 0
    correct_echo_count = 0

    for round_idx in range(NUM_ROUNDS):
        sent = [rng.choice(pool) for _ in range(BATCH)]
        cancel_indices = set(rng.sample(range(BATCH), int(BATCH * CANCEL_RATIO)))

        results = await asyncio.gather(
            *[
                _send_one(
                    client,
                    model_name,
                    _make_messages(tag),
                    CANCEL_AFTER_S if i in cancel_indices else None,
                )
                for i, tag in enumerate(sent)
            ],
            return_exceptions=True,
        )

        for i, (sent_tag, body) in enumerate(zip(sent, results)):
            if i in cancel_indices:
                continue
            if isinstance(body, BaseException):
                raise body
            assert isinstance(body, str), (
                f"round {round_idx} request {i}: expected completion, got {body!r}"
            )
            completed_count += 1
            if sent_tag in body:
                correct_echo_count += 1
            for digits in TAG_RE.findall(body):
                observed = f"TAG_{digits}"
                if observed == sent_tag:
                    continue
                if observed in pool_set:
                    contamination.append(
                        {
                            "round": round_idx,
                            "sent": sent_tag,
                            "observed": observed,
                            "body": body[:200],
                        }
                    )

    # Sanity: the probe is only meaningful if the model is actually echoing
    # tags. If it's not, we'd never see contamination either — and the test
    # would silently pass while measuring nothing.
    assert completed_count > 0, (
        "no requests completed; the cancellation rate is too aggressive "
        "or the server failed to respond"
    )
    assert correct_echo_count >= max(1, completed_count // 4), (
        f"model echoed the sent tag in only {correct_echo_count}/"
        f"{completed_count} completed responses — probe is not reliable "
        "on this setup; the test cannot detect contamination"
    )

    assert not contamination, (
        "KV cache contamination detected (NVBug 6025177 regression):\n"
        + "\n".join(repr(c) for c in contamination[:5])
        + f"\n  ... ({len(contamination)} total events)"
    )


@skip_pre_blackwell
@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
async def test_no_kv_cache_contamination_number_set_probe(
    server: RemoteOpenAIServer, model_name: str
):
    """Second regression probe for NVBug 6025177 (number-set package style).

    Uses ``ALL_SAME`` / ``UNIQUE`` machine-readable answers and the same
    contamination classification as ``run_number_set_probe.py``.
    """
    rng = random.Random(RNG_SEED + 1)
    number_pool_list = _make_number_pool(NUM_POOL_SIZE, RNG_SEED + 12345)
    number_pool = set(number_pool_list)

    NUM_ROUNDS = 10
    BATCH = 16
    CANCEL_RATIO = 0.25
    CANCEL_AFTER_S = 0.05

    client = server.get_async_client(timeout=60.0)

    contamination = []
    completed_count = 0
    valid_or_ok_count = 0

    for round_idx in range(NUM_ROUNDS):
        sent_values = [rng.choice(number_pool_list) for _ in range(BATCH)]
        cancel_indices = set(rng.sample(range(BATCH), int(BATCH * CANCEL_RATIO)))

        tasks = []
        for i, n in enumerate(sent_values):
            rep = rng.randint(REPEATED_NUMBER_COUNT_MIN, REPEATED_NUMBER_COUNT_MAX)
            messages = _make_number_set_messages(n, rep)
            tasks.append(
                _send_one(
                    client,
                    model_name,
                    messages,
                    CANCEL_AFTER_S if i in cancel_indices else None,
                    max_tokens=128,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (sent_value, body) in enumerate(zip(sent_values, results)):
            if i in cancel_indices:
                continue
            if isinstance(body, BaseException):
                raise body
            assert isinstance(body, str), (
                f"round {round_idx} request {i}: expected completion, got {body!r}"
            )
            completed_count += 1
            verdict, reason, matched = _evaluate_number_set_response(sent_value, body, number_pool)
            if verdict in ("valid", "hallucination_ok"):
                valid_or_ok_count += 1
            if verdict == "cross_contamination":
                contamination.append(
                    {
                        "round": round_idx,
                        "sent": sent_value,
                        "reason": reason,
                        "matched": matched,
                        "body": body[:400],
                    }
                )

    assert completed_count > 0, (
        "no requests completed; the cancellation rate is too aggressive "
        "or the server failed to respond"
    )
    assert valid_or_ok_count >= max(1, completed_count // 4), (
        f"model returned valid ALL_SAME/UNIQUE semantics in only "
        f"{valid_or_ok_count}/{completed_count} completed responses — "
        "number-set probe is not reliable on this setup"
    )

    assert not contamination, (
        "KV cache contamination detected (number-set probe, NVBug 6025177):\n"
        + "\n".join(repr(c) for c in contamination[:5])
        + f"\n  ... ({len(contamination)} total events)"
    )


def test_parse_integers_all_same_line():
    assert _parse_integers("ALL_SAME:482937154321") == [482937154321]


def test_evaluate_number_set_all_same_valid():
    pool = {10, 20}
    verdict, reason, matched = _evaluate_number_set_response(42, "ALL_SAME:42", pool)
    assert verdict == "valid"
    assert reason is None
    assert matched == []


def test_evaluate_number_set_all_same_wrong_common():
    pool = {10, 99}
    verdict, reason, matched = _evaluate_number_set_response(10, "ALL_SAME:99", pool)
    assert verdict == "cross_contamination"
    assert reason == "wrong_common_number"
    assert matched == [99]


def test_evaluate_number_set_all_same_wrong_not_in_pool():
    pool = {10, 99}
    verdict, reason, matched = _evaluate_number_set_response(10, "ALL_SAME:999888777", pool)
    assert verdict == "hallucination_ok"
    assert reason is None
    assert matched == []


def test_evaluate_number_set_unique_intersects_pool():
    pool = {100, 200, 300}
    verdict, reason, matched = _evaluate_number_set_response(100, "UNIQUE:100,200", pool)
    assert verdict == "cross_contamination"
    assert reason == "unique_numbers_intersect_pool"
    assert set(matched) <= pool


def test_evaluate_number_set_unique_hallucination_ok():
    pool = {100, 200}
    verdict, reason, matched = _evaluate_number_set_response(100, "UNIQUE:999888777", pool)
    assert verdict == "hallucination_ok"
    assert matched == []


def test_evaluate_number_set_freeform_pool_digits():
    pool = {111, 222}
    verdict, reason, matched = _evaluate_number_set_response(222, "The value might be 111", pool)
    assert verdict == "cross_contamination"
    assert matched == [111]


def test_evaluate_number_set_weak_invalid_empty():
    verdict, reason, matched = _evaluate_number_set_response(1, None, {1})
    assert verdict == "weak_invalid"
    assert reason == "empty_response"


def test_evaluate_number_set_all_same_markdown_fence():
    pool = {42, 99}
    verdict, reason, matched = _evaluate_number_set_response(
        42, "Here is the result:\n```\nALL_SAME:42\n```", pool
    )
    assert verdict == "valid"
    assert reason is None
    assert matched == []


def test_evaluate_number_set_freeform_correct_number_only():
    pool = {111, 222}
    verdict, reason, matched = _evaluate_number_set_response(222, "The common value is 222.", pool)
    assert verdict == "valid"
    assert reason is None
    assert matched == []
