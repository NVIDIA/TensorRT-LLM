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

    def validate_block_attention_windows(self):
        # Every layer is causal and unwindowed, so the TRTLLM check passes.
        pass


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


def test_generation_step_keeps_the_dummy_slot_context_empty():
    """Padding and ADP dummy rows must not accumulate drafter context.

    They all map to the shared dummy slot, which no request ever frees. If the
    generation step advanced its length like a real slot's, the dummies' drafter
    context would grow every step until it reached ``_max_ctx``.
    """
    max_draft_len = 3
    num_slots = 3  # two request slots plus the dummy slot
    hidden, num_layers, num_kv_heads, head_dim = 8, 1, 1, 4
    max_ctx = 64
    config = DFlashDecodingConfig(
        max_draft_len=max_draft_len, attention_backend="TRTLLM", mask_token_id=0
    )
    worker = DFlashWorker(config, Mapping())
    worker._dummy_slot = num_slots - 1
    worker._max_ctx = max_ctx
    worker._compute_block_size = max_draft_len + 1
    worker._ctx_len = torch.tensor([5, 0, 0], dtype=torch.long, device="cuda")
    # Gen request 0 owns slot 0; gen request 1 is padding on the dummy slot.
    worker._batch_to_slot = torch.tensor([0, worker._dummy_slot], dtype=torch.long, device="cuda")
    buf_shape = (num_slots, num_layers, max_ctx + max_draft_len + 1, num_kv_heads, head_dim)
    worker._ctx_k_buf = torch.zeros(buf_shape, device="cuda")
    worker._ctx_v_buf = torch.zeros(buf_shape, device="cuda")

    num_gens = 2
    tokens_per_req = max_draft_len + 1
    num_target_tokens = num_gens * tokens_per_req

    def precompute_context_kv(projected, positions):
        rows = projected.shape[0]
        k = torch.ones(rows, num_layers, num_kv_heads, head_dim, device="cuda")
        return k, k.clone()

    embed_tokens = torch.nn.Embedding(16, hidden).cuda()
    draft_model = SimpleNamespace(
        fc=object(),
        hidden_norm=object(),
        project_target_hidden=lambda hs: hs,
        precompute_context_kv=precompute_context_kv,
        draft_model_full=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens)),
    )
    captured = torch.randn(num_target_tokens, hidden, device="cuda")
    spec_metadata = SimpleNamespace(
        hidden_size=hidden,
        runtime_draft_len=max_draft_len,
        get_hidden_states=lambda num_tokens: captured[:num_tokens],
    )
    attn_metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=num_gens,
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.zeros(num_gens, dtype=torch.int32, device="cuda"),
        _seq_lens=torch.zeros(num_gens, dtype=torch.int32),
    )
    accepted_tokens = torch.arange(num_gens * tokens_per_req, dtype=torch.long, device="cuda").view(
        num_gens, tokens_per_req
    )
    num_accepted_tokens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")

    for step in range(2):
        inputs = worker.prepare_1st_drafter_inputs(
            input_ids=torch.zeros(num_target_tokens, dtype=torch.long, device="cuda"),
            position_ids=torch.zeros(num_target_tokens, dtype=torch.long, device="cuda"),
            hidden_states=torch.zeros(num_target_tokens, hidden, device="cuda"),
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            total_target_tokens=num_target_tokens,
        )
        real_len = 5 + 2 * (step + 1)
        assert worker._ctx_len.tolist() == [real_len, 0, 0]
        assert inputs["num_ctx_per_req"].tolist() == [real_len, 0]
