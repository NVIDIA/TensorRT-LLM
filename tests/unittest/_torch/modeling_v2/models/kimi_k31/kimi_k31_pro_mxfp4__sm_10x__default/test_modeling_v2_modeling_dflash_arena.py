# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter arena of the default target's DFlash worker after KV cache estimation (one GPU).

KV cache estimation runs a probe manager whose small quota clamps the served length, then the real
manager: the drafter arena (and the drafter's position ceiling) is rebuilt for the real one, and
kept while the manager stays.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31DFlashWorker,
)
from tensorrt_llm._torch.speculative import dflash
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping


class _FakeMLADrafter:
    """What the target's DFlash worker reads off the DFlash2 MLA drafter to size and address its
    arena."""

    config = SimpleNamespace(max_position_embeddings=1 << 20)
    dflash_attention_backend = "VANILLA"
    _ctx_ring_rows = None

    def __init__(self, window, block_size):
        self.block_size = block_size
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._ctx_ring_window = window
        self._num_attn_layers = 2
        self._num_heads = 4
        self._num_kv_heads = 1
        self._head_dim = 16
        self._kv_factor = 1

    def _build_fused_kv_buffers(self):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the worker allocates CUDA buffers")
def test_arena_follows_the_served_length_after_kv_cache_estimation():
    """The arena follows the probe's then the real manager's served length, and stays."""
    worker = K31DFlashWorker(
        DFlashDecodingConfig(max_draft_len=4, attention_backend="VANILLA"), Mapping()
    )
    draft = _FakeMLADrafter(64, 5)
    worker.set_draft_model(draft)
    spec_metadata = SimpleNamespace(max_num_requests=2, num_seq_slots=2)
    probe = SimpleNamespace(kv_cache_manager=SimpleNamespace(max_seq_len=15869))
    served = SimpleNamespace(kv_cache_manager=SimpleNamespace(max_seq_len=32779))
    worker._lazy_init_ctx_buffers(draft, spec_metadata, probe)
    assert worker._max_ctx == 15869
    worker._lazy_init_ctx_buffers(draft, spec_metadata, served)
    assert worker._max_ctx == 32779
    assert draft._runtime_position_ceiling == dflash.dflash_position_ceiling(
        32779, worker._compute_block_size, worker.max_draft_len
    )
    arena = worker._ctx_kv_buf
    worker._lazy_init_ctx_buffers(draft, spec_metadata, served)
    assert worker._ctx_kv_buf is arena
