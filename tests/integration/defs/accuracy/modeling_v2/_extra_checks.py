# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extra checks every modeling_v2 target may name: judgments of what its datasets did.

Each public function here is an extra check as ``_e2e.py`` defines one: ``f(result)``,
given the case's result (``_e2e.CaseResult``) and nothing else, failing by raising
``AssertionError`` and passing by returning what it saw. A spec names one by its name::

    extra_checks: [sparse_topk_exceeded]

A target that needs a check of its own writes it in its own ``extra_checks.py``, beside
its generated test. A function whose name starts with ``_`` is a helper, not a check.
Only the standard library is imported here, so the checks run on recorded outputs
without a GPU.
"""

from __future__ import annotations

from typing import Any


def _sparse_topk(result: Any) -> int | None:
    """The TopK budget of the engine's top-k sparse attention; ``None`` when it has none.

    The engine's ``sparse_attention_config.index_topk`` when its options set one, else
    the checkpoint's ``index_topk``, as the engine reads them.
    """
    sparse = result.options.get("sparse_attention_config")
    if isinstance(sparse, dict) and sparse.get("index_topk") is not None:
        return int(sparse["index_topk"])
    config = result.model_config()
    for block in (config, config.get("text_config") or {}):
        if isinstance(block, dict) and block.get("index_topk") is not None:
            return int(block["index_topk"])
    return None


def sparse_topk_exceeded(result: Any) -> str:
    """Some request outgrew the TopK budget of top-k sparse attention, so its selection ran.

    While every request of a batch is within the budget, the engine skips the indexer and
    selects every key: the dense path. A score measured that way says nothing of the
    sparse one; one request longer than the budget, its prompt and its output together,
    shows the selection ran.

    Returns:
        The longest request against the budget, and how many requests were seen.
    """
    topk = _sparse_topk(result)
    assert topk is not None, (
        "no TopK budget: the engine's options set no sparse_attention_config.index_topk and "
        "the checkpoint's config.json has no index_topk"
    )
    lengths = result.lengths()
    assert lengths, (
        f"no request's length was seen, so whether one outgrew the TopK budget {topk} is unknown"
    )
    longest = max(lengths, key=lambda n: n.total)
    assert longest.total > topk, (
        f"no request outgrew the TopK budget {topk}: the longest was {longest.total} tokens "
        f"(prompt {longest.prompt}) of {len(lengths)}"
    )
    return (
        f"the longest request {longest.total} tokens (prompt {longest.prompt}) > TopK {topk}, "
        f"of {len(lengths)}"
    )
