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
"""Bitwise gates of DeepSeek-V4.1's encoder SWA bounded replay on the modeling_v2 target.

On a prefix-cache hit of ``reused`` tokens, encoder replay re-forwards the last
``R = min(128, reused)`` cached tokens (they rebuild the sliding windows the prefix
cache does not keep) and charges them to the step's token budget. So the hit's
context chunks start R positions earlier than the same hit's would without
replay. Chunked prefill is not invariant to where chunks start: cuBLAS picks its
kernel by row count, so a row's bits depend on the row count of its step. A hit
with replay is therefore not bitwise equal to the same hit without it once the
prompt spans more than one chunk. What replay does guarantee, and what these
cases gate:

* a hit of at most 128 tokens replays from position 0, so it runs exactly the
  cold run's chunks and its tokens and logprobs are bitwise equal to the cold
  run's. The producer's first step has as many rows as the cold run's first
  chunk, so the cached rows were computed at the same row count;
* two producers of one prefix give bitwise-equal hits;
* the same cold request twice is bitwise equal.

The engine is multi-chunk (``max_num_tokens`` 2048 against a ~4.6K-token prompt),
greedy, without speculation (so generation logprobs are returned), with both
replay switches unset: the cache manager resolves them on at tep4 and dep4. At
dep4 every request is pinned to attention-DP rank 0, since a prefix hit needs the
producer's KV on the same rank. ``TRTLLM_MODELING_V2=require`` and the step
contract check travel on the ``LLM`` (``modeling_v2_env``), as in the accuracy
file; on a host where the LLM API cannot spawn its workers, launch under
``mpirun -n 4 --oversubscribe trtllm-llmapi-launch pytest ...``.
"""

import os

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DECODER_REPLAY_ENV,
    DEEPSEEK_V41_ENCODER_REPLAY_ENV,
)
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SchedulingParams

from .modeling_v2_env import modeling_v2_llm_args
from .test_modeling_v2_deepseek_v41 import _MODEL_PATH, skip_no_checkpoint, skip_not_sm103

_MEMBERS = {
    "tep4": dict(tensor_parallel_size=4, moe_expert_parallel_size=4),
    "dep4": dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True),
}
_BUDGET = 2048
_HIT = 120  # a hit of at most one window: replay from position 0
_TAIL = 200


def _prompt(tokenizer):
    """A ~4.6K-token passkey haystack: three chunks at a 2048-token budget."""
    filler = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    )
    text = (
        filler * 120
        + "The pass key is 73520. Remember it. 73520 is the pass key. "
        + filler * 120
        + "What is the pass key? The pass key is"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) > 2 * _BUDGET, len(ids)
    return ids


def _tail(n: int, seed: int):
    return [1000 + (seed * 7919 + i * 104729) % 59000 for i in range(n)]


@skip_not_sm103
@skip_no_checkpoint
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_encoder_replay_bitwise(member, monkeypatch):
    for name in (DEEPSEEK_V41_ENCODER_REPLAY_ENV, DEEPSEEK_V41_DECODER_REPLAY_ENV):
        if os.environ.get(name) is not None:
            pytest.skip(
                f"{name} is set; these cases gate the replays as the cache manager resolves them"
            )
    with LLM(
        _MODEL_PATH,
        max_seq_len=16384,
        max_batch_size=8,
        max_num_tokens=_BUDGET,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, enable_block_reuse=True),
        **_MEMBERS[member],
        **modeling_v2_llm_args("require", monkeypatch, **{MODELING_V2_VALIDATE_ENV: "1"}),
    ) as llm:
        sp = SamplingParams(max_tokens=16, temperature=0.0, logprobs=5)
        pin = {}
        if member == "dep4":
            pin["scheduling_params"] = [
                SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
            ]

        def run(ids, salt):
            out = llm.generate([ids], sp, cache_salt=[salt], **pin)[0]
            o = out.outputs[0]
            logprobs = [{t: v.logprob for t, v in step.items()} for step in o.logprobs]
            return list(o.token_ids), logprobs, out.cached_tokens

        ids = _prompt(llm.tokenizer)
        cold = run(ids, "cold")
        assert cold[:2] == run(ids, "cold-again")[:2], "the same cold request twice differs"

        # A hit of _HIT tokens. The producer's first step has _BUDGET rows, as the cold
        # run's first chunk does, so its cached rows were computed at the same row count.
        run(ids[:_HIT] + _tail(_BUDGET - _HIT, 1), "short")
        short = run(ids, "short")
        assert short[:2] == cold[:2], (
            "a hit replayed from position 0 runs the cold run's chunks and must match it bitwise"
        )

        # Two producers of the prompt's first N - 60 tokens (a hit inside its last
        # window), with different tails of one length: bitwise-equal hits.
        k = len(ids) - 60
        run(ids[:k] + _tail(_TAIL, 2), "x")
        cross = run(ids, "x")
        run(ids[:k] + _tail(_TAIL, 3), "y")
        cross2 = run(ids, "y")
        assert cross[2] > 0 and cross2[2] > 0, (cross[2], cross2[2])
        assert cross[:2] == cross2[:2], "two producers of one prefix gave different hits"
