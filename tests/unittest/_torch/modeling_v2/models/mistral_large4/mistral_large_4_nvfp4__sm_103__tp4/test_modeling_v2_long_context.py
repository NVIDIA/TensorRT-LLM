# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The long-context witness of Mistral-Large-4-1T-A52B-NVFP4 on its sm_103 tp4 target, at tp4
(four ranks) and at tp8 (eight ranks over two nodes of four).

GSM8K's prompts never pass the indexer's `index_topk` (2048 positions). Below it the engine
skips the indexer and sparse MLA attends over the causal window, so the accuracy gate never
runs the indexer's selection, nor SSMax, LeRoPE and YaRN at long positions, nor the FlashMLA
sparse kernels on a selection past it (each called on the rank's heads -- 16 at tp4, 8 at tp8 --
first in the 64-head query they take), nor the routed experts at the largest step. Here the
whole model runs under its serving configuration -- max_seq_len = max_num_tokens = 32768, chunked
prefill off, the latent KV cache at its default (the fp8_ds_mla pool), CUDA graphs over batches
1, 2, 4, 8 and 16, padded -- with the routed experts tensor parallel (every rank holds all 128
experts, each cut to its slice of the 6144-wide intermediate: 1536 at tp4, 768 at tp8), and
greedy decoding must recall a passkey hidden in plain filler text:

* in prompts of about 4K, 8K, 16K and 31K tokens, at three depths each, one prompt at a time:
  each prompt is one context step of one request (read back from the engine's iteration
  statistics), then decodes in the batch-1 graph;
* in sixteen prompts whose lengths sum to nearly 32768 tokens, six of them past `index_topk`
  and ten within it, admitted together: they are one context step of sixteen requests (read
  back the same way) holding the whole token budget, then decode in the batch-16 graph.

Every rank sees every token of a step, so the ~31K prompt and the sixteen-prompt step each run
every rank's experts over the whole step: up to 32768 rows, 131072 routed slots, over 128 local
experts of width 1536 at tp4 and 768 at tp8.

Text only: no image, no chat template. Two engine options beyond the serving ones make the
steps observable and do not change what a step computes: iteration statistics, and a request
queue that waits up to `BATCH_WAIT_MS` for a full batch so the sixteen prompts are admitted
in one step rather than split by their arrival.

Runs with the ranks already in place, on four GB300s at tp4 or on eight over two nodes of four
at tp8: pytest under `trtllm-llmapi-launch`, as the whole-model accuracy class runs, with the
checkpoint under `LLM_MODELS_ROOT` as `Mistral-Large-4-1T-A52B-NVFP4` (or
`mistral-large-4-nvfp4`) as published, Mistral-native (params.json, tekken.json and the
consolidated safetensors), which the engine loads with checkpoint_format "mistral_large_4". One
scheduler serves all the ranks, so every step below is one step on every rank. Skipped where the
checkpoint is absent or off sm_103; each world size is skipped below its own ranks (four GPUs at
tp4, eight ranks at tp8) and inside a launch of another world size.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from test_common.llm_data import llm_models_root
from utils.util import skip_num_gpus_less_than

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs import TokensPrompt
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size


def _ranks() -> int:
    """The ranks this process can serve: the MPI world a launcher started (eight over two nodes
    of four, where each node sees four devices), or, unlaunched, the devices the LLM API spawns
    its workers on -- as the integration conftest's `skip_less_mpi_world_size` counts."""
    world = get_mpi_world_size()
    return torch.cuda.device_count() if world == 1 else world


def _launched_elsewhere(world: int) -> bool:
    """A launcher started an MPI world of another size: an engine of `world` ranks does not
    start inside it."""
    return get_mpi_world_size() not in (1, world)


#: The engine's world sizes, each gated on its own ranks: tp4 on four GPUs, tp8 on eight ranks
#: over two nodes of four.
WORLDS = [
    pytest.param(
        4,
        marks=[
            skip_num_gpus_less_than(4),
            pytest.mark.skipif(
                _launched_elsewhere(4), reason="tp4 needs a launch of four ranks or none"
            ),
        ],
        id="tp4",
    ),
    pytest.param(
        8,
        marks=[
            pytest.mark.skipif(_ranks() < 8, reason="tp8 needs eight ranks"),
            pytest.mark.skipif(
                _launched_elsewhere(8), reason="tp8 needs a launch of eight ranks or none"
            ),
        ],
        id="tp8",
    ),
]

pytestmark = [
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    # One engine serves the module: the LLM API starts its result-dispatch thread on the first
    # request, inside the first test, and stops it when the fixture shuts the engine down.
    pytest.mark.threadleak(enabled=False),
]

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The serving configuration: tp4 or tp8 (attention and the routed experts tensor parallel over all
# the ranks, the experts not expert-parallel), sixteen sequences, unchunked steps of up to 32768
# tokens.
MAX_NUM_TOKENS = 32768
ENGINE = dict(
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
    max_batch_size=16,
    max_num_tokens=MAX_NUM_TOKENS,
    max_seq_len=MAX_NUM_TOKENS,
    enable_chunked_prefill=False,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
# How long the request queue holds an iteration for more requests while fewer than
# max_batch_size are waiting; every iteration pays it, decode steps included. The sixteen
# prompts are submitted together as token ids and arrive within milliseconds of each other.
BATCH_WAIT_MS = 1000

INDEX_TOPK = 2048
# Prompts of about 4K, 8K, 16K and 31K tokens, a passkey at three depths of each.
LENGTHS = (4096, 8192, 16384, 31744)
DEPTHS = (0.1, 0.5, 0.9)
# Sixteen prompts filling one 32768-token context step: six past index_topk, ten within it.
# Each is built to at most its length and within one filler sentence of it, so the step holds
# between 98% of the budget and 32700 tokens.
BATCH_PAST_TOPK = (8500, 4000, 3000, 2600, 2300, 2100)
BATCH_WITHIN_TOPK = (1900, 1700, 1500, 1300, 1100, 900, 700, 500, 350, 250)
BATCH_LENGTHS = BATCH_PAST_TOPK + BATCH_WITHIN_TOPK
BATCH_FILL = 0.98
BATCH_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)

HEAD = (
    "There is an important piece of information hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about it.\n\n"
)
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)
QUESTION = "\nWhat is the pass key? The pass key is"
# A five-digit key and its closing punctuation fit with room to spare.
ANSWER = SamplingParams(max_tokens=12, temperature=0.0)


def _checkpoint_dir() -> Path | None:
    """The checkpoint's directory, found by its Mistral-native config, params.json."""
    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        if (Path(root) / name / "params.json").is_file():
            return Path(root) / name
    return None


def _prompt(reps: int, depth: float, key: int) -> str:
    before = round(reps * depth)
    needle = f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
    return HEAD + FILLER * before + needle + FILLER * (reps - before) + QUESTION


def _passkey_ids(tokenizer, tokens: int, depth: float, key: int) -> list[int]:
    """The prompt's token ids, BOS included as the LLM API's own tokenization adds it, at most
    `tokens` long and within one filler sentence of it."""
    unit = len(tokenizer.encode(FILLER * 64)) / 64
    bare = len(tokenizer.encode(_prompt(0, depth, key)))
    reps = max(0, int((tokens - bare) / unit))
    ids = tokenizer.encode(_prompt(reps, depth, key))
    while len(ids) > tokens and reps > 0:
        reps -= 1
        ids = tokenizer.encode(_prompt(reps, depth, key))
    return ids


def _keys(count: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(10000, 100000), count)


def _context_steps(llm: LLM) -> list[tuple[int, int]]:
    """(context requests, context tokens) of each iteration since the last call that ran any
    context rows."""
    steps = []
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append((ifb["numContextRequests"], ifb["numCtxTokens"]))
    return steps


@pytest.fixture(scope="module", params=WORLDS)
def llm(request):
    world = request.param
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({llm_models_root()})")
    with LLM(
        str(path),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        batch_wait_timeout_ms=BATCH_WAIT_MS,
        tensor_parallel_size=world,
        moe_tensor_parallel_size=world,
        **ENGINE,
    ) as engine:
        capacity = engine.get_kv_cache_capacity()
        print(f"LONGCTX kv_cache_capacity {capacity}", flush=True)
        _context_steps(engine)
        yield engine


@pytest.mark.parametrize("tokens", LENGTHS)
def test_passkey_alone_in_one_context_step(llm, tokens):
    """A passkey at three depths of a prompt of about `tokens` tokens, each prompt one context
    step of one request, recalled by greedy decoding."""
    misses = []
    for depth, key in zip(DEPTHS, _keys(len(DEPTHS), seed=tokens)):
        ids = _passkey_ids(llm.tokenizer, tokens, depth, key)
        assert INDEX_TOPK < len(ids) <= tokens, (tokens, len(ids))
        answer = llm.generate(TokensPrompt(prompt_token_ids=ids), ANSWER).outputs[0].text
        steps = _context_steps(llm)
        print(
            f"LONGCTX alone tokens={len(ids)} depth={depth} key={key} answer={answer!r} "
            f"context_steps={steps}",
            flush=True,
        )
        assert steps == [(1, len(ids))], (len(ids), depth, steps)
        if str(key) not in answer:
            misses.append((len(ids), depth, key, answer))
    assert not misses, misses


def test_sixteen_prompts_fill_one_context_step(llm):
    """Sixteen passkey prompts admitted together are one context step of sixteen requests
    holding nearly the whole 32768-token budget, and every one recalls its key."""
    keys = _keys(len(BATCH_LENGTHS), seed=MAX_NUM_TOKENS)
    prompts = [
        _passkey_ids(llm.tokenizer, tokens, BATCH_DEPTHS[i % len(BATCH_DEPTHS)], key)
        for i, (tokens, key) in enumerate(zip(BATCH_LENGTHS, keys))
    ]
    lengths = [len(ids) for ids in prompts]
    total = sum(lengths)
    assert sum(n > INDEX_TOPK for n in lengths) == len(BATCH_PAST_TOPK), lengths
    assert BATCH_FILL * MAX_NUM_TOKENS <= total <= MAX_NUM_TOKENS, (total, lengths)

    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], ANSWER)
    steps = _context_steps(llm)
    answers = [out.outputs[0].text for out in outputs]
    for n, key, answer in zip(lengths, keys, answers):
        print(f"LONGCTX batch tokens={n} key={key} answer={answer!r}", flush=True)
    print(f"LONGCTX batch total={total} context_steps={steps}", flush=True)

    assert steps == [(len(prompts), total)], (total, steps)
    misses = [(n, key, a) for n, key, a in zip(lengths, keys, answers) if str(key) not in a]
    assert not misses, misses
