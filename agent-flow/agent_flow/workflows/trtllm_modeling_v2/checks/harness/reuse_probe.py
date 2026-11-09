# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public cached_tokens evidence from a controlled, sequential repeat-prompt probe.

No prompt ledger or inference about what an arbitrary evaluation should reuse.
Requested snapshot settings size the probe; they are not attested engine facts.
A missing capacity, missing tokenizer, unsuitable budget or unsupported rank
pinning is UNKNOWN, never a negative cache observation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402

PROBE_TEXT = (
    "The river ran slowly past the mill, and the miller counted the sacks of grain "
    "that the farmers had brought in from the valley. Each sack was weighed twice, "
    "once when it arrived and once before it was ground, and the two numbers were "
    "written side by side in a ledger that had been kept in the same hand for years."
)


def cached_tokens(output: Any) -> int | None:
    """The public response count, or UNKNOWN when unavailable or invalid."""
    value = getattr(output, "cached_tokens", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def probe_prompt(tokenizer: Any, tokens: int) -> list[int]:
    """Ordinary tokenizer-produced text tokens, never arbitrary vocabulary ids."""
    ids = list(tokenizer.encode(PROBE_TEXT, add_special_tokens=False))
    if not ids:
        raise ValueError("the tokenizer encoded the reuse probe as no tokens")
    return (ids * (tokens // len(ids) + 1))[:tokens]


def probe(llm: Any, params: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Repeat one prompt after completion on the same cache, with a unique cache salt."""
    record: dict[str, Any] = {"status": "UNKNOWN"}
    try:
        capacity = engine_facts.kv_capacity(llm)
        block = capacity.get("tokensPerBlock")
        if not isinstance(block, int) or block <= 0:
            return {**record, "skipped": "public KV capacity did not report tokensPerBlock"}
        kv = requested.get("kv_cache_config") or {}
        state = kv.get("mamba_state_config") or {}
        points = [
            int(x) for x in state.get("additional_snapshot_offsets_from_start") or () if int(x) > 0
        ]
        interval = int(
            state.get("periodic_snapshot_interval", kv.get("mamba_state_cache_interval", 0)) or 0
        )
        if interval > 0:
            points.append(interval)
        if not points:
            points = [
                int(x)
                for x in state.get("additional_snapshot_offsets_from_end") or ()
                if int(x) > 0
            ]
        tokens = (min(points) if points else block) + 2 * block
        record["tokens"] = tokens
        seq = requested.get("max_seq_len")
        budget = requested.get("max_num_tokens")
        if (seq and tokens + 1 > seq) or (
            budget and not requested.get("enable_chunked_prefill") and tokens > budget
        ):
            return {**record, "skipped": "repeat prompt does not fit declared request limits"}
        if capacity.get("maxNumTokens", 0) < tokens + 1:
            return {**record, "skipped": "repeat prompt does not fit public KV capacity"}
        import copy

        params = copy.deepcopy(params)
        params.n = 1
        params.use_beam_search = False
        params.temperature = 0.0
        params.top_k = 1
        params.max_tokens = 1
        pinned = {}
        if requested.get("enable_attention_dp"):
            from tensorrt_llm.scheduling_params import SchedulingParams

            pinned["scheduling_params"] = SchedulingParams(
                attention_dp_rank=0, attention_dp_relax=False
            )
        prompt = {"prompt_token_ids": probe_prompt(llm.tokenizer, tokens)}
        # Isolate this experiment from the preceding workload's cache entries.
        import uuid

        pinned["cache_salt"] = f"mv2flow-reuse-{uuid.uuid4().hex}"
        first = llm.generate_async(prompt, params, **pinned).result()
        second = llm.generate_async(prompt, params, **pinned).result()
        record.update(first=cached_tokens(first), reused=cached_tokens(second))
        if record["reused"] is not None:
            record["status"] = "OBSERVED"
        return record
    except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as exc:
        return {**record, "error": f"{type(exc).__name__}: {exc}"[:300]}
