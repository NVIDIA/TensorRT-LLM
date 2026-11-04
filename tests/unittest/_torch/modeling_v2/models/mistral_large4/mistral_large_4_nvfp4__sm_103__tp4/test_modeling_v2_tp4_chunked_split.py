# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The indexer's query split under chunked prefill, end to end, on Mistral-Large-4-1T-A52B-NVFP4's
sm_103 tp4 target at the 1M config, at tp4 (four ranks) and at tp8 (eight ranks over two nodes of
four).

GSM8K's prompts stay within the indexer's `index_topk` (2048 positions), where the engine skips
the context indexer: the accuracy gate never runs the split of a step's context queries over the
TP ranks, nor the gather of their selections. Here the whole model runs under its serving
configuration -- max_seq_len 1048576, chunked prefill in steps of max_num_tokens 32768, sixteen
sequences, the fp8_ds_mla latent cache at the default KV dtype, CUDA graphs over batches 1, 2, 4,
8 and 16, padded -- and two passkey prompts submitted together in one generate call are
scheduled, first come first served, as two context steps:

* all of prompt A (20480 tokens) and the first 12288 tokens of prompt B, 32768 context rows, B's
  chunk first (the scheduler puts a step's last chunks after the others): the engine splits them
  evenly over the ranks, 8192 rows a rank at tp4 (rank 1's share runs from B's chunk into A, one
  indexer chunk per request) and 4096 at tp8 (B's chunk is exactly ranks 0-2's shares, A ranks
  3-7's); each rank selects for its own rows and every full indexer layer gathers every rank's
  share;
* the rest of B, 9003 tokens over its 12288-token cached prefix, with A's first decode row behind
  them: an uneven split of a chunk over a cached prefix in a mixed step -- (2250, 2251, 2251,
  2251) at tp4, (1125, 1125, 1126, 1125, 1125, 1126, 1125, 1126) at tp8.

A's passkey sits at 0.7 of its filler, in rank 3's share of the first step at tp4 and rank 6's at
tp8; B's at 0.3, in its first chunk (rank 0's share at tp4, rank 1's at tp8), so B's question rows
reach it only through the cached prefix. Greedy decoding must recall both. Every key count stays
at or below 32768. The step composition is read back from the engine's iteration statistics, and
each step's shares at the world are printed beside it.

Text only: no image, no chat template. Two engine options beyond the serving ones make the steps
observable and do not change what a step computes: iteration statistics, and a request queue that
waits up to `BATCH_WAIT_MS` for a full batch so the two prompts are admitted in one step rather
than split by their arrival.

Runs with the ranks already in place, on four GB300s at tp4 or on eight over two nodes of four
at tp8: pytest under `trtllm-llmapi-launch`, as the whole-model accuracy class runs, with the
checkpoint under `LLM_MODELS_ROOT` as `Mistral-Large-4-1T-A52B-NVFP4` (or
`mistral-large-4-nvfp4`) in its Mistral-native format, which the engine loads with
checkpoint_format "mistral_large_4". Skipped where the checkpoint is absent or off sm_103; each
world size is skipped below its own ranks (four GPUs at tp4, eight ranks at tp8) and inside a
launch of another world size.
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
# the ranks, the experts not expert-parallel), sixteen sequences, the checkpoint's own context
# prefilled in chunked steps of up to 32768 tokens.
MAX_NUM_TOKENS = 32768
ENGINE = dict(
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
    max_batch_size=16,
    max_num_tokens=MAX_NUM_TOKENS,
    max_seq_len=1048576,
    enable_chunked_prefill=True,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
# How long the request queue holds an iteration for more requests while fewer than
# max_batch_size are waiting; every iteration pays it, decode steps included. The two prompts
# are submitted together as token ids and arrive within milliseconds of each other.
BATCH_WAIT_MS = 1000

# The engine's defaults: the split from this many context rows in a step, and the KV page, which
# is also the unit a chunk is cut down to when the step's budget trims it.
Q_SPLIT_THRESHOLD = 8192
TOKENS_PER_BLOCK = 32

# Prompt A fills its first step whole; the rest of the step's budget is B's first chunk, and the
# rest of B is the next step's one chunk, behind which A decodes.
A_TOKENS = 20480
B_TOKENS = 21291
B_FIRST_CHUNK = MAX_NUM_TOKENS - A_TOKENS
B_LAST_CHUNK = B_TOKENS - B_FIRST_CHUNK
A_DEPTH, B_DEPTH = 0.7, 0.3

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


def _split(rows: int, world: int) -> list[int]:
    """The engine's per-rank shares of a step's context rows over `world` ranks, in rank order:
    rank r takes `[rows * r // world, rows * (r + 1) // world)`."""
    return [rows * (r + 1) // world - rows * r // world for r in range(world)]


def _passkey_ids(tokenizer, tokens: int, depth: float, key: int) -> list[int]:
    """Exactly `tokens` ids: BOS, then the head, the filler with the needle at `depth` of it, and
    the question, each part tokenized on its own and the filler cut at the token that gives the
    length."""

    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text)
        return ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids

    head, question = encode(HEAD), encode(QUESTION)
    needle = encode(f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n")
    filler = encode(FILLER * (tokens // 8))
    room = tokens - 1 - len(head) - len(needle) - len(question)
    assert 0 < room <= len(filler), (tokens, room, len(filler))
    before = int(room * depth)
    ids = [tokenizer.bos_token_id, *head, *filler[:before], *needle]
    ids += [*filler[: room - before], *question]
    assert len(ids) == tokens, (len(ids), tokens)
    return ids


def _steps(llm: LLM) -> list[tuple[int, int, int]]:
    """(context requests, context tokens, generation requests) of each iteration since the last
    call that ran any context rows."""
    steps = []
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append(
                (ifb["numContextRequests"], ifb["numCtxTokens"], ifb.get("numGenRequests", 0))
            )
    return steps


@pytest.fixture(scope="module", params=WORLDS)
def world(request) -> int:
    """The engine's world size: four ranks at tp4, eight at tp8."""
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
        tensor_parallel_size=world,
        moe_tensor_parallel_size=world,
        **ENGINE,
    ) as engine:
        print(f"SPLIT tp{world} kv_cache_capacity {engine.get_kv_cache_capacity()}", flush=True)
        _steps(engine)
        yield engine


def test_two_prompts_split_over_the_ranks(llm, world):
    """Two passkey prompts in one generate call: a full 32768-row step split evenly over the
    world's ranks, then a chunk over a cached prefix split unevenly with a decode row behind it;
    both keys are recalled."""
    assert A_TOKENS % TOKENS_PER_BLOCK == 0 and B_FIRST_CHUNK % TOKENS_PER_BLOCK == 0
    assert _split(MAX_NUM_TOKENS, world) == [MAX_NUM_TOKENS // world] * world
    assert B_LAST_CHUNK >= Q_SPLIT_THRESHOLD and B_LAST_CHUNK % world != 0
    keys = random.Random(MAX_NUM_TOKENS + 1).sample(range(10000, 100000), 2)
    prompts = [
        _passkey_ids(llm.tokenizer, A_TOKENS, A_DEPTH, keys[0]),
        _passkey_ids(llm.tokenizer, B_TOKENS, B_DEPTH, keys[1]),
    ]
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], ANSWER)
    steps = _steps(llm)
    answers = [out.outputs[0].text for out in outputs]
    for name, ids, key, answer in zip("AB", prompts, keys, answers):
        print(f"SPLIT prompt {name} tokens={len(ids)} key={key} answer={answer!r}", flush=True)
    print(
        f"SPLIT context_steps={steps} (context requests, context tokens, generation requests); "
        f"q_sizes at tp{world} {_split(MAX_NUM_TOKENS, world)} then {_split(B_LAST_CHUNK, world)}",
        flush=True,
    )

    assert steps == [(2, MAX_NUM_TOKENS, 0), (1, B_LAST_CHUNK, 1)], steps
    misses = [(n, key, a) for n, key, a in zip("AB", keys, answers) if str(key) not in a]
    assert not misses, misses
