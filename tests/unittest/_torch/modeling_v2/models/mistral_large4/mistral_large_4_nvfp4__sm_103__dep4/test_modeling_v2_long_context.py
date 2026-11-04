# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The long-context witness of Mistral-Large-4-1T-A52B-NVFP4 on its sm_103 dep4 target.

GSM8K's prompts never pass the indexer's `index_topk` (2048 positions). Below it the engine
skips the indexer and sparse MLA attends over the causal window, so the accuracy gate never
runs the indexer's selection, nor SSMax, LeRoPE and YaRN at long positions. Here the whole
model runs under its serving config -- per rank, max_seq_len = max_num_tokens = 32768,
chunked prefill off, the fp8_ds_mla latent KV cache at kv_cache_config's default dtype, CUDA
graphs over batches 1, 2, 4, 8 and 16, padded -- attention data parallel over the ranks of the
world, four on one node or eight on two nodes of four (dep8), and greedy decoding must recall a
passkey hidden in plain filler text:

* in prompts of about 4K, 16K and 31K tokens, one prompt at a time: one rank prefills it in one
  context step while the other ranks step beside it on the engine's one-token dummy, so the
  experts gather the world size times the prompt's rows -- up to 4 x 31K in one call per layer
  at four ranks, up to 8 x 31K in calls of 32768 rows at eight;
* in one ~31K-token prompt per rank admitted together (read back from each rank's own
  iteration statistics), so every rank prefills a ~31K prompt in the same step and the experts
  see all of those prefills at once -- about 127K gathered rows at four ranks, about 254K
  through every expert call of every MoE layer at eight -- then every rank decodes.

Text only: no image, no chat template. Two engine options beyond the serving ones make the
steps observable and do not change what a step computes: iteration statistics, which under
attention DP carry one record per rank, and a request queue that waits up to `BATCH_WAIT_MS`
for a full batch so the prompts are admitted in one step rather than split by their arrival.

Runs with the ranks already in place: pytest under `trtllm-llmapi-launch`, as the whole-model
accuracy class runs, with the checkpoint under `LLM_MODELS_ROOT` as
`Mistral-Large-4-1T-A52B-NVFP4` (or `mistral-large-4-nvfp4`) as published,
Mistral-native (params.json, tekken.json and the consolidated safetensors, which the engine loads
with checkpoint_format "mistral_large_4") -- on four GB300s for the four-rank engine, on eight
over two nodes of four for the eight-rank one. Each world runs only in a launch of its own size
(or unlaunched): the four-rank engine where four GPUs are visible, the eight-rank one where eight
ranks are, counted on the launched MPI world rather than the devices a node shows (each node sees
four). Skipped where the checkpoint is absent, below the world's ranks, or off sm_103.
"""

from __future__ import annotations

import collections
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


def _launched_as(world: int):
    """An engine of `world` ranks starts in no launch of another size."""
    launched = get_mpi_world_size()
    return pytest.mark.skipif(
        launched not in (1, world),
        reason=f"the {world}-rank engine does not start in a launch of {launched} ranks",
    )


#: The worlds the target serves, each gated on its own ranks: four ranks on one node, eight
#: over two nodes of four.
WORLDS = [
    pytest.param(4, marks=[skip_num_gpus_less_than(4), _launched_as(4)], id="world4"),
    pytest.param(
        8,
        marks=[
            pytest.mark.skipif(_ranks() < 8, reason="the dep8 configuration needs eight ranks"),
            _launched_as(8),
        ],
        id="world8",
    ),
]

pytestmark = [
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    # One engine serves each world: the LLM API starts its result-dispatch thread on the first
    # request, inside the first test, and stops it when the fixture shuts the engine down.
    pytest.mark.threadleak(enabled=False),
]

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The serving config: dep4 (dep8 at eight ranks), sixteen sequences and unchunked steps of up
# to 32768 tokens per rank.
MAX_NUM_TOKENS = 32768


def _engine(world: int) -> dict:
    return dict(
        tensor_parallel_size=world,
        moe_expert_parallel_size=world,
        enable_attention_dp=True,
        max_batch_size=16,
        max_num_tokens=MAX_NUM_TOKENS,
        max_seq_len=MAX_NUM_TOKENS,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
    )


# How long the request queue holds an iteration for more requests while fewer than a full
# batch are waiting; every iteration pays it, decode steps included. The prompts admitted
# together are submitted as token ids and arrive within milliseconds of each other.
BATCH_WAIT_MS = 1000

INDEX_TOPK = 2048
# Prompts of about 4K, 16K and 31K tokens, a passkey in the middle of each.
LENGTHS = (4096, 16384, 31744)
# One ~31K prompt per rank, the needle at a different depth in each.
TOGETHER = 31744
TOGETHER_DEPTHS = {
    4: (0.1, 0.4, 0.6, 0.9),
    8: (0.05, 0.15, 0.3, 0.4, 0.55, 0.7, 0.8, 0.95),
}
# A prompt "about 31K" when it lands within 3% of TOGETHER.
TOGETHER_FILL = 0.97

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


def _context_steps(llm: LLM) -> dict[int, dict[int, tuple[int, int]]]:
    """Since the last call, every iteration that ran context rows on some rank:
    `{iteration: {attention-DP rank: (context requests, context tokens)}}`, one entry per rank
    whose own record that iteration carries context rows."""
    steps: dict[int, dict[int, tuple[int, int]]] = collections.defaultdict(dict)
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            rank = int(record.get("attentionDpRank", 0))
            steps[int(record["iter"])][rank] = (ifb["numContextRequests"], ifb["numCtxTokens"])
    return dict(steps)


@pytest.fixture(scope="module", params=WORLDS)
def world(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def llm(world):
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
        **_engine(world),
    ) as engine:
        capacity = engine.get_kv_cache_capacity()
        print(f"LONGCTX world {world} kv_cache_capacity {capacity}", flush=True)
        _context_steps(engine)
        yield engine


@pytest.mark.parametrize("tokens", LENGTHS)
def test_passkey_alone_in_one_context_step(llm, tokens):
    """A passkey in the middle of a prompt of about `tokens` tokens, prefilled by one rank in
    one context step while the others step on their dummies, recalled by greedy decoding."""
    (key,) = _keys(1, seed=tokens)
    ids = _passkey_ids(llm.tokenizer, tokens, 0.5, key)
    assert INDEX_TOPK < len(ids) <= tokens, (tokens, len(ids))
    answer = llm.generate(TokensPrompt(prompt_token_ids=ids), ANSWER).outputs[0].text
    steps = _context_steps(llm)
    print(
        f"LONGCTX alone tokens={len(ids)} key={key} answer={answer!r} context_steps={steps}",
        flush=True,
    )
    assert [list(by_rank.values()) for by_rank in steps.values()] == [[(1, len(ids))]], steps
    assert str(key) in answer, (len(ids), key, answer)


def test_31k_prompts_one_per_rank_in_one_step(llm, world):
    """One ~31K-token passkey prompt per rank -- four at four ranks, eight at eight -- admitted
    together: each rank prefills one of them in the same context step, so the experts see every
    rank's ~31K prefill at once, and every one recalls its key."""
    keys = _keys(world, seed=TOGETHER)
    prompts = [
        _passkey_ids(llm.tokenizer, TOGETHER, depth, key)
        for depth, key in zip(TOGETHER_DEPTHS[world], keys)
    ]
    lengths = [len(ids) for ids in prompts]
    assert len(lengths) == world
    assert all(TOGETHER_FILL * TOGETHER <= n <= TOGETHER for n in lengths), lengths

    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], ANSWER)
    steps = _context_steps(llm)
    answers = [out.outputs[0].text for out in outputs]
    for n, key, answer in zip(lengths, keys, answers):
        print(f"LONGCTX together tokens={n} key={key} answer={answer!r}", flush=True)
    print(f"LONGCTX together context_steps={steps}", flush=True)

    # One iteration, every rank one ~31K prompt: the per-rank token counts of that step.
    assert len(steps) == 1, steps
    ((iteration, by_rank),) = steps.items()
    assert sorted(by_rank) == list(range(world)), by_rank
    assert all(requests == 1 for requests, _ in by_rank.values()), by_rank
    assert sorted(tokens for _, tokens in by_rank.values()) == sorted(lengths), (by_rank, lengths)
    print(
        f"LONGCTX together iteration {iteration}: per-rank context tokens "
        f"{[by_rank[r][1] for r in range(world)]}",
        flush=True,
    )
    misses = [(n, key, a) for n, key, a in zip(lengths, keys, answers) if str(key) not in a]
    assert not misses, misses
