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
"""One-model EAGLE-3 speculative decoding for Qwen2-class models.

``test_qwen2_is_a_one_model_speculative_target`` runs anywhere. The end-to-end test
needs a Qwen2-architecture checkpoint and a matching EAGLE-3 draft head, given by
``TLLM_QWEN2_EAGLE3_TARGET`` and ``TLLM_QWEN2_EAGLE3_DRAFT`` (for example Qwen2.5-7B-Instruct
and an EAGLE-3 head trained for it); it skips otherwise.
"""

import contextlib
import os

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen import Qwen2ForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

PROMPTS = [
    "The history of the printing press began in the fifteenth century when Johannes Gutenberg",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "In a small village by the sea, an old fisherman woke before dawn and",
]


def test_qwen2_is_a_one_model_speculative_target():
    """Without the speculative base class the draft model and spec worker are never
    created and the model fails with ``no attribute 'draft_config'``."""
    assert issubclass(Qwen2ForCausalLM, SpecDecOneEngineForCausalLM)


@contextlib.contextmanager
def _accepted_drafts():
    """Counts the draft tokens the speculative worker accepts.

    ``LLM.get_stats`` returns nothing for this configuration, so the executor's per-iteration
    stats are read as they are produced; the executor must run in this process
    (``TLLM_WORKER_USE_SINGLE_PROCESS=1``).
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    counts = {"accepted": 0, "requests": 0}
    original = PyExecutor._update_iter_stats

    def update(self, stats, *args, **kwargs):
        result = original(self, stats, *args, **kwargs)
        spec = getattr(stats, "specdec_stats", None)
        if spec is not None and spec.num_requests_with_draft_tokens > 0:
            counts["accepted"] += spec.num_accepted_tokens
            counts["requests"] += spec.num_requests_with_draft_tokens
        return result

    PyExecutor._update_iter_stats = update
    try:
        yield counts
    finally:
        PyExecutor._update_iter_stats = original


def _paths():
    target = os.environ.get("TLLM_QWEN2_EAGLE3_TARGET")
    draft = os.environ.get("TLLM_QWEN2_EAGLE3_DRAFT")
    if not (target and draft and os.path.isdir(target) and os.path.isdir(draft)):
        return None
    return target, draft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_paths() is None, reason="Qwen2 target and EAGLE-3 draft not provided")
@pytest.mark.threadleak(enabled=False)
def test_qwen2_eagle3_one_model_generates_and_accepts_draft_tokens(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

    target, draft = _paths()
    max_tokens = 48
    tokenizer = AutoTokenizer.from_pretrained(target)
    prompt_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    sampling = SamplingParams(
        temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=True, end_id=-1
    )

    def generate(speculative_config):
        llm = LLM(
            model=target,
            max_batch_size=len(PROMPTS),
            max_num_tokens=2048,
            max_seq_len=512,
            speculative_config=speculative_config,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
            cuda_graph_config=CudaGraphConfig(max_batch_size=len(PROMPTS)),
            skip_tokenizer_init=True,
            enable_iter_perf_stats=True,
        )
        with _accepted_drafts() as counts, llm:
            tokens = [o.outputs[0].token_ids for o in llm.generate(prompt_ids, sampling)]
        return tokens, counts["accepted"]

    reference, _ = generate(None)
    speculative, accepted = generate(Eagle3DecodingConfig(max_draft_len=3, speculative_model=draft))
    assert accepted > 0, "the speculative worker accepted no draft token"

    assert [len(t) for t in speculative] == [max_tokens] * len(PROMPTS)
    # Speculative decoding verifies a different token count per step, so bf16 rounding
    # can flip a near-tied token late in a generation; the opening tokens must agree.
    for ref, spec in zip(reference, speculative):
        assert ref[:8] == spec[:8], (ref[:8], spec[:8])
