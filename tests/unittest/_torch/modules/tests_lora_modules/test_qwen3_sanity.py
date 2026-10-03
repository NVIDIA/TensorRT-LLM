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
"""Sanity tests for Qwen3 LoRA support (dense and MoE)."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from _torch.modules.tests_lora_modules.lora_sanity_utils import (
    ATTN_LORA_MODULES,
    ATTN_TRTLLM_MODULES,
    MLP_LORA_MODULES,
    MLP_TRTLLM_MODULES,
    assert_lora_changes_output,
    assert_outputs_match,
    create_lora_adapter,
    run_lora_test,
)
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_80gb

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.peft.lora import layer as lora_layer_module
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import KvCacheConfig


def _run_mixed_lora_cuda_graph_test(
    model_path, target_modules, trtllm_modules, kv_cache_config: KvCacheConfig
):
    """Verify base rows remain unchanged when sharing a LoRA-specialized graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_dir = create_lora_adapter(
            os.path.join(tmpdir, "lora"),
            model_path,
            target_modules,
        )
        lora_config = LoraConfig(
            lora_dir=[lora_dir],
            lora_target_modules=trtllm_modules,
            max_lora_rank=16,
            max_loras=2,
            cuda_graph_specialize_lora=True,
        )
        with LLM(
            model=model_path,
            backend="pytorch",
            lora_config=lora_config,
            kv_cache_config=kv_cache_config,
            tensor_parallel_size=1,
            max_batch_size=4,
            max_num_tokens=256,
        ) as llm:
            # Prevent adapter EOS from shrinking the mixed batch and changing
            # BF16 GEMM rounding relative to the all-base reference.
            sampling = SamplingParams(max_tokens=20, temperature=0.0, logprobs=0, ignore_eos=True)
            prompts = [
                "The capital of France is",
                "The capital of France is",
                "Hello, how are you",
                "Hello, how are you",
            ]
            lora_request = LoRARequest("test-lora", 0, lora_dir)
            mixed_outputs = llm.generate(
                prompts,
                sampling,
                lora_request=[lora_request, None, lora_request, None],
            )
            base_outputs = llm.generate(prompts, sampling)

        # Keep both calls on full prefill: partial reuse can select a different
        # attention kernel, whose numerical drift can fail this strict token and
        # logprob comparison even when LoRA correctly leaves base rows unchanged.
        assert all(output.cached_tokens == 0 for output in mixed_outputs)
        assert all(output.cached_tokens == 0 for output in base_outputs)
        for index in (1, 3):
            assert_outputs_match(mixed_outputs[index], base_outputs[index])
        assert_lora_changes_output(
            [mixed_outputs[index] for index in (0, 2)],
            [base_outputs[index] for index in (0, 2)],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LoRA overlap requires CUDA streams.")
def test_lora_forward_with_base_executes_overlap(monkeypatch):
    """Verify that the LoRA overlap path uses its auxiliary CUDA stream."""
    lora_layer = LoraLayer([LoraModuleType.ATTENTION_Q], [2])
    layer_key = lora_layer_module.CudaGraphLoraParams.LoraLayerKey(
        layer_idx=0, module_ids=tuple(lora_layer.lora_module_types)
    )
    aux_stream = torch.cuda.Stream()
    parallel_streams = []
    original_parallel_executor = lora_layer_module.maybe_execute_in_parallel

    def record_parallel_executor(*args, **kwargs):
        parallel_streams.append(args[4])
        return original_parallel_executor(*args, **kwargs)

    monkeypatch.setattr(lora_layer_module, "maybe_execute_in_parallel", record_parallel_executor)
    monkeypatch.setattr(LoraLayer, "forward", lambda self, x, *_: torch.ones_like(x))

    x = torch.ones((2, 2), device="cuda")
    lora_params = {
        "cuda_graph_params": SimpleNamespace(layer_info={layer_key: object()}),
        "lora_aux_stream": aux_stream,
    }
    with with_multi_stream(True):
        output = LoraLayer.forward_with_base(lambda: x.clone(), (lora_layer,), x, lora_params, 0)

    assert lora_layer._par_events is not None
    assert len(parallel_streams) == 1
    assert parallel_streams[0] is aux_stream
    torch.testing.assert_close(output, 2 * x)


class TestQwen3LoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"
        self.kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=True)
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_qwen3_bf16_lora(self):
        run_lora_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            kv_cache_config=self.kv_cache_config,
        )

    def test_qwen3_fp8_lora(self):
        run_lora_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            dtype=torch.float8_e4m3fn,
            kv_cache_config=self.kv_cache_config,
        )

    def test_qwen3_bf16_lora_overlap(self):
        run_lora_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            overlap=True,
            kv_cache_config=self.kv_cache_config,
        )

    def test_qwen3_fp8_lora_overlap(self):
        run_lora_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            dtype=torch.float8_e4m3fn,
            overlap=True,
            kv_cache_config=self.kv_cache_config,
        )

    def test_qwen3_bf16_lora_cuda_graph_specialization(self):
        run_lora_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            specialize_cuda_graph=True,
            kv_cache_config=self.kv_cache_config,
        )

    def test_qwen3_bf16_lora_cuda_graph_specialization_mixed_batch(self):
        # Keep these short prompts on the same context attention path so the
        # strict comparison measures LoRA isolation without cold/warm drift.
        _run_mixed_lora_cuda_graph_test(
            self.model_path,
            {**ATTN_LORA_MODULES, **MLP_LORA_MODULES},
            ATTN_TRTLLM_MODULES + MLP_TRTLLM_MODULES,
            kv_cache_config=self.kv_cache_config.model_copy(update={"enable_partial_reuse": False}),
        )


@skip_gpu_memory_less_than_80gb
class TestQwen3MoELoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_qwen3_moe_bf16_lora(self):
        run_lora_test(
            self.model_path,
            ATTN_LORA_MODULES,
            ATTN_TRTLLM_MODULES,
        )

    def test_qwen3_moe_fp8_lora(self):
        run_lora_test(
            self.model_path,
            ATTN_LORA_MODULES,
            ATTN_TRTLLM_MODULES,
            dtype=torch.float8_e4m3fn,
        )
