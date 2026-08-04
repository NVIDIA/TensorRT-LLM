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
"""GPU unit tests for DFlash worker backend-specific cache setup.

The tests exercise framework-side TRTLLM-Gen cache plumbing without loading a
draft model. The full generated-FMHA path is covered by the Qwen3.6 NVFP4
DFlash accuracy test in ``integration/defs/accuracy/test_llm_api_pytorch.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlash worker allocates CUDA context-cache buffers"
)


class _FakeDraftModel:
    block_size = 8
    config = SimpleNamespace(max_position_embeddings=128)

    def __init__(self, attention_backend="TRTLLM"):
        self.dflash_attention_backend = attention_backend
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._num_attn_layers = 0
        self._num_heads = 0
        self._num_kv_heads = 0
        self._head_dim = 0

    def _build_fused_kv_buffers(self):
        self._num_attn_layers = 2
        self._num_heads = 8
        self._num_kv_heads = 1
        self._head_dim = 64

    def _get_attention_mask_args(self, layer_idx):
        return True, (-1, -1)


def test_trtllm_backend_builds_private_paged_context_cache(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM")
    worker = DFlashWorker(config, Mapping())
    draft_model = _FakeDraftModel()
    worker.set_draft_model(draft_model)
    spec_metadata = SimpleNamespace(max_num_requests=2)
    attn_metadata = SimpleNamespace(max_seq_len=64)

    worker._lazy_init_ctx_buffers(draft_model, spec_metadata, attn_metadata)

    # max_ctx=64 plus an 8-token draft block requires three 32-token pages
    # per slot. Two request slots plus one scratch slot therefore use 9 pages.
    assert worker._ctx_pages_per_slot == 3
    assert worker._ctx_kv_buf.shape == (2, 9, 2, 1, 32, 64)
    assert worker._ctx_k_buf is None
    assert worker._ctx_v_buf is None
    assert worker._ctx_page_table.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
