# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval at length on Mistral-Large-4-1T-A52B-NVFP4's sm_103 dep4 target, at the checkpoint's
own context.

The engine is the one the core unit serves: per rank max_seq_len 1048576, prefill in chunks of
max_num_tokens 32768 (chunked prefill on), sixteen sequences, the fp8_ds_mla latent KV cache
at kv_cache_config's default dtype, block reuse off, 0.85 of the free memory to the KV cache,
CUDA graphs over batches 1, 2, 4, 8 and 16, padded; attention data parallel over four ranks,
the routed experts expert parallel. A passkey needle
is hidden in plain filler text (`test_modeling_v2_long_context.py`'s prompt) at an early, a
middle and a late depth of prompts of about 32768, 131072, 524288 and 1040000 tokens, and
greedy decoding must recall every key. Text only: token ids through the LLM API, no chat
template.

One request at a time: one rank holds it and prefills it chunk by chunk, every chunk after its
first over the cached prefix the earlier ones wrote, while the other three ranks step beside it
on the engine's dummy rows (each step's experts gather every rank's rows). The rank is read back
from the iteration statistics, which under attention DP carry one record per rank. The time to
the first token, as the client sees it streaming, is printed per request with the rank and its
context steps.

Runs on four GB300s with the ranks already in place: pytest under `trtllm-llmapi-launch`, as
the whole-model accuracy class runs, with the checkpoint under `LLM_MODELS_ROOT` as
`Mistral-Large-4-1T-A52B-NVFP4` (or `mistral-large-4-nvfp4`) in its Mistral-native
format, which the engine loads with checkpoint_format "mistral_large_4". Skipped where the
checkpoint is absent, below four GPUs, or off sm_103.
"""

from __future__ import annotations

import collections
import time

import pytest
from test_common.llm_data import llm_models_root
from utils.util import skip_num_gpus_less_than

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs import TokensPrompt
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from . import test_modeling_v2_long_context as long_context

pytestmark = [
    skip_num_gpus_less_than(4),
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    # One engine serves the module: the LLM API starts its result-dispatch thread on the first
    # request, inside the first test, and stops it when the fixture shuts the engine down.
    pytest.mark.threadleak(enabled=False),
]

#: dep4's four attention-DP ranks.
RANKS = 4
MAX_SEQ_LEN = 1048576
MAX_NUM_TOKENS = 32768
# The core unit's serving engine (its LLM-API options), per rank.
ENGINE = dict(
    tensor_parallel_size=RANKS,
    moe_expert_parallel_size=RANKS,
    enable_attention_dp=True,
    max_batch_size=16,
    max_num_tokens=MAX_NUM_TOKENS,
    max_seq_len=MAX_SEQ_LEN,
    enable_chunked_prefill=True,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
# Prompt lengths, and the needle's depth in each, early to late.
LENGTHS = (32768, 131072, 524288, 1040000)
DEPTHS = (0.1, 0.5, 0.9)
# A prompt "about" its length when it lands within one percent of it.
FILL = 0.99
ANSWER = SamplingParams(max_tokens=12, temperature=0.0)
#: Every request's key, distinct, fixed by its length and depth.
KEYS = dict(
    zip(
        [(n, d) for n in LENGTHS for d in DEPTHS],
        long_context._keys(len(LENGTHS) * len(DEPTHS), seed=MAX_SEQ_LEN),
    )
)


def _rank_steps(llm: LLM) -> dict[int, list[tuple[int, int, int, int]]]:
    """Since the last call, per attention-DP rank, every iteration it recorded:
    `(iteration, context requests, context tokens, generation requests)`, in order."""
    by_rank: dict[int, list[tuple[int, int, int, int]]] = collections.defaultdict(list)
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        by_rank[int(record.get("attentionDpRank", 0))].append(
            (
                int(record["iter"]),
                ifb.get("numContextRequests", 0),
                ifb.get("numCtxTokens", 0),
                ifb.get("numGenRequests", 0),
            )
        )
    return {rank: sorted(steps) for rank, steps in by_rank.items()}


@pytest.fixture(scope="module")
def llm():
    path = long_context._checkpoint_dir()
    if path is None:
        pytest.skip(
            f"checkpoint {long_context.CHECKPOINTS} is not under LLM_MODELS_ROOT "
            f"({llm_models_root()})"
        )
    with LLM(
        str(path),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        **ENGINE,
    ) as engine:
        print(f"RETRIEVAL kv_cache_capacity {engine.get_kv_cache_capacity()}", flush=True)
        _rank_steps(engine)
        yield engine


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("tokens", LENGTHS)
def test_passkey_at_length(llm, tokens, depth):
    """A passkey at `depth` of a prompt of about `tokens` tokens, prefilled in 32768-token
    chunks by one rank while the others step beside it, recalled by greedy decoding."""
    key = KEYS[(tokens, depth)]
    ids = long_context._passkey_ids(llm.tokenizer, tokens, depth, key)
    assert FILL * tokens <= len(ids) <= tokens, (tokens, len(ids))

    start = time.monotonic()
    ttft = None
    for out in llm.generate_async(TokensPrompt(prompt_token_ids=ids), ANSWER, streaming=True):
        if ttft is None and out.outputs[0].token_ids:
            ttft = time.monotonic() - start
    total = time.monotonic() - start
    answer = out.outputs[0].text
    steps = _rank_steps(llm)

    held = {
        rank: [(it, t) for it, n, t, _ in s if n]
        for rank, s in steps.items()
        if any(n for _, n, _, _ in s)
    }
    print(
        f"RETRIEVAL tokens={len(ids)} depth={depth} key={key} answer={answer!r} "
        f"ttft_s={ttft:.1f} total_s={total:.1f} rank={list(held)}",
        flush=True,
    )
    for rank in sorted(steps):
        ctx = [(it, t) for it, n, t, _ in steps[rank] if n]
        print(
            f"RETRIEVAL tokens={len(ids)} depth={depth} rank {rank}: {len(steps[rank])} "
            f"iterations recorded (iterations {steps[rank][0][0]}..{steps[rank][-1][0]}), "
            f"{len(ctx)} with context rows {ctx}",
            flush=True,
        )
    # One rank prefilled the whole prompt, in chunks of at most max_num_tokens.
    assert len(held) == 1, held
    ((rank, chunks),) = held.items()
    assert sum(t for _, t in chunks) == len(ids), (rank, chunks)
    assert all(t <= MAX_NUM_TOKENS for _, t in chunks), (rank, chunks)
    assert str(key) in answer, (len(ids), depth, key, answer)
