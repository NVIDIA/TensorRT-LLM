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
"""The Kimi K3 DFlash worker sizes its drafter context slots from the executor's seq-slot pool.

The slots are allocated once, on the first forward, and reused for every later batch shape. That
first forward can run on CUDA graph metadata, whose ``max_num_requests`` is the captured bucket,
so the worker takes the slot count from ``num_seq_slots`` instead.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (
    K3DFlashWorker,
)
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the worker allocates CUDA context-cache buffers"
)

NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM = 2, 1, 64
MAX_SEQ_LEN, BLOCK = 64, 8


class _FakeDrafter:
    block_size = BLOCK
    config = SimpleNamespace(max_position_embeddings=128)
    dflash_attention_backend = "VANILLA"

    def _build_fused_kv_buffers(self):
        self._num_attn_layers = NUM_LAYERS
        self._num_kv_heads = NUM_KV_HEADS
        self._head_dim = HEAD_DIM


@pytest.mark.parametrize("num_seq_slots", [2, 4], ids=["pool-eq-batch", "disagg-adp-2x-pool"])
def test_ctx_cache_spans_the_full_seq_slot_pool_on_a_graph_bucket(num_seq_slots):
    worker = K3DFlashWorker(
        DFlashDecodingConfig(max_draft_len=BLOCK - 1, attention_backend="VANILLA"), Mapping()
    )
    graph_bucket = SimpleNamespace(max_num_requests=1, num_seq_slots=num_seq_slots)

    worker._lazy_init_ctx_buffers(
        _FakeDrafter(), graph_bucket, SimpleNamespace(max_seq_len=MAX_SEQ_LEN)
    )

    # num_seq_slots request slots plus one scratch slot for padding dummies.
    assert worker._dummy_slot == num_seq_slots
    assert list(worker._free_slots) == list(range(num_seq_slots))
    assert worker._ctx_len.shape == (num_seq_slots + 1,)
    assert worker._batch_to_slot_buffer.tensor.shape == (num_seq_slots,)
    assert worker._ctx_k_buf.shape == (
        num_seq_slots + 1,
        NUM_LAYERS,
        MAX_SEQ_LEN + BLOCK,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
