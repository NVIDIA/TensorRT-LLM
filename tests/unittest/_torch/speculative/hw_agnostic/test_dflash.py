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

import os
import sys
import unittest
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import get_spec_metadata
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "The future of AI is",
]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_dflash_metadata_preserves_default_seq_slot_pool_in_graph_copy():
    metadata = DFlashSpecMetadata(
        max_draft_len=4,
        max_total_draft_tokens=4,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=5,
    )

    graph_metadata = metadata.create_cuda_graph_metadata(max_batch_size=2)

    assert metadata.num_seq_slots == 5
    assert graph_metadata.max_num_requests == 2
    assert graph_metadata.num_seq_slots == 5


def test_dflash_graph_bucket_uses_full_seq_slot_pool():
    """A small graph bucket must not shrink the persistent context pool."""
    num_seq_slots = 5
    spec_config = DFlashDecodingConfig(
        max_draft_len=4,
        target_layer_ids=[0],
    )
    metadata = get_spec_metadata(
        spec_config,
        SimpleNamespace(hidden_size=4, torch_dtype=torch.bfloat16, vocab_size=32),
        max_num_requests=num_seq_slots,
        max_num_tokens=8,
        num_seq_slots=num_seq_slots,
    ).create_cuda_graph_metadata(max_batch_size=2)

    class DraftModel:
        block_size = 5
        config = SimpleNamespace(max_position_embeddings=8)
        fc = SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16))
        hidden_norm = object()
        _num_attn_layers = 1
        _num_kv_heads = 2
        _head_dim = 4

        def _build_fused_kv_buffers(self):
            pass

        def project_target_hidden(self, hidden_states):
            return hidden_states

        def precompute_context_kv(self, hidden_states, position_ids):
            shape = (hidden_states.shape[0], 1, 2, 4)
            return (
                torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
            )

    worker = DFlashWorker(spec_config, Mapping())
    draft_model = DraftModel()
    attn_metadata = SimpleNamespace(
        max_seq_len=8,
        num_ctx_tokens=1,
        num_contexts=1,
        _seq_lens=[1],
    )
    worker._lazy_init_ctx_buffers(draft_model, metadata, attn_metadata)

    assert metadata.max_num_requests == 2 < metadata.num_seq_slots
    num_slots = num_seq_slots + 1
    assert worker._ctx_k_buf.shape[0] == num_slots
    assert worker._ctx_v_buf.shape == worker._ctx_k_buf.shape
    assert worker._ctx_len.shape[0] == num_slots
    assert worker._batch_to_slot.shape == (num_seq_slots,)
    assert worker._dummy_slot == num_seq_slots
    assert list(worker._free_slots) == list(range(num_seq_slots))

    metadata.request_ids = [42]
    worker._store_prefill_context(
        draft_model,
        metadata,
        attn_metadata,
        torch.tensor([0], device="cuda"),
        total_target_tokens=1,
    )
    live_slot = worker._req_to_slot[42]
    assert live_slot != worker._dummy_slot
    assert worker._dummy_slot not in worker._free_slots
    assert list(worker._free_slots) == [1, 2, 3, 4]


def _make_llm_config(
    target_model_dir: str,
    dflash_model_dir: str,
    disable_overlap_scheduler: bool,
    max_draft_len: int = 4,
    max_batch_size: int = 4,
):
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2048)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)
    spec_config = DFlashDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=dflash_model_dir,
    )
    return dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=False,
        speculative_config=spec_config,
    )


def _run_and_check(llm_config: dict):
    llm = LLM(**llm_config)
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=256, temperature=0))
    llm.shutdown()

    # Simple check for reasonable output.
    # Acceptance length checks are flaky since there are only 3 test prompts
    assert len(outputs) == len(PROMPTS)
    for out in outputs:
        assert len(out.outputs[0].token_ids) > 0
        assert out.outputs[0].text.strip()


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_dflash_qwen3_8b(disable_overlap_scheduler: bool):
    """Test DFlash with Qwen3-8B BF16: CUDA graphs, padding, and draft acceptance."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    llm_config = _make_llm_config(
        target_model_dir=f"{models_path}/Qwen3/Qwen3-8B",
        dflash_model_dir=f"{models_path}/Qwen3-8B-DFlash-b16",
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    _run_and_check(llm_config)


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_dflash_qwen3_5_4b(disable_overlap_scheduler: bool):
    """Test DFlash with Qwen3.5-4B BF16: CUDA graphs, padding, and draft acceptance."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    llm_config = _make_llm_config(
        target_model_dir=f"{models_path}/Qwen3.5-4B",
        dflash_model_dir=f"{models_path}/Qwen3.5-4B-DFlash",
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    _run_and_check(llm_config)


@pytest.mark.high_cuda_memory
def test_dflash_qwen3_8b_rejection():
    """DFlash with rejection sampling on: the block-capture rejection path
    (draft-prob scatter -> fail-closed guard -> rejection acceptance) runs
    end-to-end with non-greedy sampling and produces coherent output."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    llm_config = _make_llm_config(
        target_model_dir=f"{models_path}/Qwen3/Qwen3-8B",
        dflash_model_dir=f"{models_path}/Qwen3-8B-DFlash-b16",
        disable_overlap_scheduler=True,
    )
    llm_config["speculative_config"].use_rejection_sampling = True

    llm = LLM(**llm_config)
    # Non-greedy so rejection sampling actually engages (all-greedy bypasses it).
    outputs = llm.generate(
        PROMPTS, SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, top_k=50, seed=1234)
    )
    llm.shutdown()

    assert len(outputs) == len(PROMPTS)
    for out in outputs:
        assert len(out.outputs[0].token_ids) > 0
        assert out.outputs[0].text.strip()


if __name__ == "__main__":
    unittest.main()
