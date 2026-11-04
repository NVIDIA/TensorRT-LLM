# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval at length on Mistral-Large-4-1T-A52B-NVFP4's sm_103 tp4 target at the 1M config, at
tp4 (four ranks) and at tp8 (eight ranks over two nodes of four).

The checkpoint's own context is 1048576 positions (YaRN 512 x 2048, LeRoPE, SSMax). Here the whole
model runs under its serving configuration -- max_seq_len 1048576, chunked prefill in steps of
max_num_tokens 32768, sixteen sequences, the fp8_ds_mla latent cache at the default KV dtype, CUDA
graphs over batches 1, 2, 4, 8 and 16, padded -- and greedy decoding must recall a passkey hidden in
plain filler text, in prompts of 32768, 131072, 524288 and 1040384 tokens, at an early, a middle
and a late depth of each, one request at a time.

Each prompt is prefilled in ceil(tokens / 32768) context steps of one request, read back from the
engine's iteration statistics: every chunk after the first runs over a cached prefix, its 32768
queries split over the TP ranks (8192 a rank at tp4, 4096 at tp8, every full indexer layer
gathering the shares), the indexer selecting over the cached and the new keys and sparse MLA
attending over the selected latent rows of every earlier chunk and its own. While a step's
context KV (the request's cached and new rows) is at most 2**19 rows, the target's held workspace
at this configuration (`_ctx_workspace_rows`), those rows are dequantized out of the pool into it
for the sparse prefill kernel; the 524288-token prompt's last chunk reaches exactly 2**19 and
stays there. Past it -- the 1040384-token prompt's last 16 chunks -- every context row attends
straight off the fp8 pool as a one-row sequence of the sparse decode kernel, in sub-batches of
8192 rows: 1040384 is 31 full chunks and a last one of three full sub-batches, the sub-batch
size the decode kernel's 1M cells certify. That last chunk reads keys to 1040384 and positions
to 1040383. Then the answer decodes in the batch-1 graph over the whole prompt. Each prompt's
time to first token, its context steps and the KV pool blocks it held are printed.

Text only: no image, no chat template. Iteration statistics are on to read the steps back; they
do not change what a step computes.

Runs with the ranks already in place, on four GB300s at tp4 or on eight over two nodes of four
at tp8: pytest under `trtllm-llmapi-launch`, as the whole-model accuracy class runs, with the
checkpoint under `LLM_MODELS_ROOT` as `Mistral-Large-4-1T-A52B-NVFP4` (or
`mistral-large-4-nvfp4`) in its Mistral-native format, which the engine loads with
checkpoint_format "mistral_large_4". Skipped where the checkpoint is absent or off sm_103; each
world size is skipped below its own ranks (four GPUs at tp4, eight ranks at tp8) and inside a
launch of another world size.
"""

from __future__ import annotations

import math
import random
import time
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
    # The first test of each world also loads its engine (about 10 minutes); a 1040000-token
    # prompt took 105 s to its first token on four GB300s.
    pytest.mark.timeout(1800),
]

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The serving configuration (the core unit's LLM-API options): tp4 or tp8 (attention and the routed
# experts tensor parallel over all the ranks, the experts not expert-parallel), sixteen
# sequences, the checkpoint's own context prefilled in chunked steps of up to 32768 tokens.
MAX_NUM_TOKENS = 32768
MAX_SEQ_LEN = 1048576
ENGINE = dict(
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
    max_batch_size=16,
    max_num_tokens=MAX_NUM_TOKENS,
    max_seq_len=MAX_SEQ_LEN,
    enable_chunked_prefill=True,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
TOKENS_PER_BLOCK = 32

# The longest: 31 chunks of 32768 and a last one of 3 x 8192, so that every decode-path sub-batch
# is a full 8192 rows (1040000 would end on one of 7808).
LENGTHS = (32768, 131072, 524288, 1040384)
DEPTHS = (0.1, 0.5, 0.9)

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


def _passkey_ids(tokenizer, tokens: int, depth: float, key: int) -> list[int]:
    """Exactly `tokens` ids: BOS, then the head, the filler with the needle at `depth` of it, and
    the question, each part tokenized on its own and the filler cut at the token that gives the
    length."""

    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text)
        return ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids

    head, question = encode(HEAD), encode(QUESTION)
    needle = encode(f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n")
    # A long run of the filler tokenizes a little shorter per sentence than a short one.
    per_filler = len(encode(FILLER * 64)) / 64
    filler = encode(FILLER * math.ceil(1.05 * tokens / per_filler))
    room = tokens - 1 - len(head) - len(needle) - len(question)
    assert 0 < room <= len(filler), (tokens, room, len(filler))
    before = int(room * depth)
    ids = [tokenizer.bos_token_id, *head, *filler[:before], *needle]
    ids += [*filler[: room - before], *question]
    assert len(ids) == tokens, (len(ids), tokens)
    return ids


def _stats(llm: LLM) -> tuple[list[tuple[int, int]], int]:
    """Since the last call: (context requests, context tokens) of each iteration that ran context
    rows, and the most KV pool blocks in use at any iteration."""
    steps, used = [], 0
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append((ifb["numContextRequests"], ifb["numCtxTokens"]))
        used = max(used, (record.get("kvCacheStats") or {}).get("usedNumBlocks", 0))
    return steps, used


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
        tensor_parallel_size=world,
        moe_tensor_parallel_size=world,
        **ENGINE,
    ) as engine:
        print(f"RETRIEVAL tp{world} kv_cache_capacity {engine.get_kv_cache_capacity()}", flush=True)
        _stats(engine)
        yield engine


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("tokens", LENGTHS)
def test_passkey_at_length(llm, tokens, depth):
    """A passkey at `depth` of a `tokens`-token prompt, prefilled in 32768-token chunks, recalled
    by greedy decoding."""
    key = random.Random(tokens + int(depth * 10)).randrange(10000, 100000)
    ids = _passkey_ids(llm.tokenizer, tokens, depth, key)

    started_at = time.strftime("%H:%M:%S", time.gmtime())
    print(f"RETRIEVAL start tokens={len(ids)} depth={depth} at {started_at} UTC")
    started = time.monotonic()
    ttft, text = None, ""
    for output in llm.generate_async(TokensPrompt(prompt_token_ids=ids), ANSWER, streaming=True):
        if ttft is None and output.outputs[0].token_ids:
            ttft = time.monotonic() - started
        text = output.outputs[0].text
    total = time.monotonic() - started
    steps, used_blocks = _stats(llm)
    print(
        f"RETRIEVAL tokens={len(ids)} depth={depth} key={key} answer={text!r} "
        f"ttft_s={ttft:.2f} total_s={total:.2f} context_steps={len(steps)} "
        f"chunks={sorted(set(t for _, t in steps))} kv_blocks_used_max={used_blocks} "
        f"({used_blocks * TOKENS_PER_BLOCK} tokens)",
        flush=True,
    )

    assert all(n == 1 and t <= MAX_NUM_TOKENS for n, t in steps), steps
    assert sum(t for _, t in steps) == len(ids), (len(ids), steps)
    assert len(steps) == math.ceil(len(ids) / MAX_NUM_TOKENS), (len(ids), steps)
    assert str(key) in text, (len(ids), depth, key, text)
