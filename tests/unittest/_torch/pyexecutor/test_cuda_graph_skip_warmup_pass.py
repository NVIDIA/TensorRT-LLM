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
"""TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS=1 on a real model: same tokens, one capture pass, safe fallback."""

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "Machine learning is",
    "The future of AI is",
    "What is the capital of Australia?",
    "Explain in one sentence why the sky is blue.",
    "Who wrote the book 'Pride and Prejudice'?",
    "List three U.S. national holidays.",
]


def _generate(skip_warmup_pass: bool, move_workspace: bool = False):
    """Return the greedy tokens, the graph keys and each capture call's warm-up-only flag per pass.
    ``move_workspace`` regrows the shared workspace before the second single-pass capture."""
    passes: list[list[bool]] = []
    moved: list[bool] = []
    warmup_and_capture = DecoderCudaGraphs.warmup_and_capture
    run_cuda_graph_warmup = DecoderCudaGraphs.run_cuda_graph_warmup
    capture = CUDAGraphRunner.capture

    def record_warmup_and_capture(self, resource_manager, **kwargs):
        passes.append([])
        if kwargs.get("allow_single_pass") and move_workspace:
            moved.append(False)
        return warmup_and_capture(self, resource_manager, **kwargs)

    def record_pass(self, resource_manager, **kwargs):
        passes[-1].append(self.cuda_graph_runner.is_warmup_only)
        return run_cuda_graph_warmup(self, resource_manager, **kwargs)

    def capture_and_move_workspace(self, key, *args, **kwargs):
        if moved == [False] and not self.is_warmup_only and self.graphs:
            captured = next(iter(self.graph_metadata.values()))["attn_metadata"]
            workspace = captured.cuda_graph_workspace
            workspace.resize_(workspace.numel() + (1 << 20))
            moved[0] = True
        return capture(self, key, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        if skip_warmup_pass:
            patch.setenv("TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS", "1")
        else:
            patch.delenv("TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS", raising=False)
        patch.setattr(DecoderCudaGraphs, "warmup_and_capture", record_warmup_and_capture)
        patch.setattr(DecoderCudaGraphs, "run_cuda_graph_warmup", record_pass)
        patch.setattr(CUDAGraphRunner, "capture", capture_and_move_workspace)
        with LLM(
            model=str(llm_models_root() / "llama-models-v2" / "TinyLlama-1.1B-Chat-v1.0"),
            max_batch_size=8,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
        ) as llm:
            # Requests finish one by one, so the decode batch replays every graph size.
            outputs = llm.generate(
                PROMPTS,
                [
                    SamplingParams(max_tokens=8 + 3 * i, temperature=0, ignore_eos=True)
                    for i in range(len(PROMPTS))
                ],
            )
            keys = set(llm._executor.engine.model_engine.cuda_graph_runner.graphs)
    assert moved == ([True] if move_workspace else [])
    return [list(output.outputs[0].token_ids) for output in outputs], keys, passes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_final_capture_skips_warmup_pass_and_falls_back_when_a_buffer_moves():
    tokens, keys, passes = _generate(skip_warmup_pass=False)
    assert passes[-1] == [True, False]

    single_tokens, single_keys, single_passes = _generate(skip_warmup_pass=True)
    # Only the final executor skips the warm-up-only pass; a KV-cache profiling pass keeps both.
    assert single_passes[-1] == [False]
    assert single_passes[:-1] == passes[:-1]
    assert single_keys == keys
    assert single_tokens == tokens

    fallback_tokens, fallback_keys, fallback_passes = _generate(
        skip_warmup_pass=True, move_workspace=True
    )
    assert fallback_passes[-1] == [False, True, False]
    assert fallback_keys == keys
    assert fallback_tokens == tokens
