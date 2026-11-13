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
"""cuda_graph_config.sampling_variants on one-engine Eagle3: only the listed graphs, with their fallbacks."""

from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig
from tensorrt_llm.logger import logger

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "What is the capital of Australia?",
]
GREEDY = SamplingParams(max_tokens=10, temperature=0)
SAMPLED = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=7)


def _graph_config(variants):
    return CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True, sampling_variants=variants)


def _run(cuda_graph_config, *calls):
    """Graph keys, each call's outputs and graph replays, eager-variant notices and vocab size."""
    models_path = llm_models_root()
    notices = set()
    warning_once = logger.warning_once

    def record_warning_once(*msg, key):
        if "runs eagerly" in " ".join(map(str, msg)):
            notices.add(key)
        return warning_once(*msg, key=key)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        patch.setattr(logger, "warning_once", record_warning_once)
        with LLM(
            model=f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct",
            speculative_config=Eagle3DecodingConfig(
                max_draft_len=3, speculative_model=f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
            ),
            max_batch_size=4,
            cuda_graph_config=cuda_graph_config,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.5),
        ) as llm:
            runner = llm._executor.engine.model_engine.cuda_graph_runner
            keys = set(runner.graphs)
            replays = 0
            replay = runner.replay

            def counting_replay(*args, **kwargs):
                nonlocal replays
                replays += 1
                return replay(*args, **kwargs)

            runner.replay = counting_replay
            results = []
            for prompts, params in calls:
                start = replays
                outputs = [output.outputs[0] for output in llm.generate(prompts, params)]
                results.append((outputs, replays - start))
            vocab_size = len(llm.tokenizer.tokenizer)
    return SimpleNamespace(keys=keys, calls=results, notices=notices, vocab_size=vocab_size)


@pytest.mark.high_cuda_memory
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_variants_capture_only_the_listed_graphs():
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")
    greedy = _run(_graph_config(["greedy"]), (PROMPTS, GREEDY), (PROMPTS[:1], SAMPLED))
    advanced = _run(_graph_config(["advanced"]), (PROMPTS, GREEDY))
    (greedy_outputs, greedy_replays), ((sampled,), sampled_replays) = greedy.calls
    ((advanced_outputs, advanced_replays),) = advanced.calls

    # Each engine captures one variant of the same graph shapes.
    assert greedy.keys and all(key.is_all_greedy_sample for key in greedy.keys)
    assert not any(key.is_all_greedy_sample for key in advanced.keys)
    assert {key._replace(is_all_greedy_sample=True) for key in advanced.keys} == greedy.keys

    # Without greedy graphs, greedy requests replay the advanced ones and still decode greedily.
    assert advanced_replays > 0 and advanced.notices == set()
    assert [output.token_ids for output in advanced_outputs] == [
        output.token_ids for output in greedy_outputs
    ]

    # Without advanced graphs, the sampled request runs eagerly under the once-per-variant warning.
    assert greedy_replays > 0 and sampled_replays == 0
    assert greedy.notices == {"uncaptured_sampling_variant_advanced"}
    assert 0 < len(sampled.token_ids) <= SAMPLED.max_tokens and sampled.text
    assert all(0 <= token < greedy.vocab_size for token in sampled.token_ids)
