# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tensorrt_llm._torch.attention_backend.fmha.prims_ts_block_sparse import (
    PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY,
    PrimsTSBlockSparseFmha,
    PrimsTSBlockSparseRuntime,
)
from tensorrt_llm._torch.attention_backend.sparse.block_sparse import BlockSparseParams
from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as visual_trtllm
from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline


class _FakeBaseTrtllmAttentionMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepare_calls = 0
        self.seq_lens = None
        self.num_contexts = None
        self.max_seq_len = None
        self.request_ids = None

    def prepare(self):
        self.prepare_calls += 1


class _FakeBlockSparseAttention:
    def __init__(self, layer_idx: int, attention_metadata_state: dict[str, object]):
        self.layer_idx = layer_idx
        self.num_heads = 4
        self.num_kv_heads = 4
        self.head_dim = 128
        self.sparse_params = BlockSparseParams(q_block_size=64, kv_block_size=64)
        self.attention_metadata_state = attention_metadata_state

        self.q_scaling = 1.0
        self.quant_mode = 0
        self.predicted_tokens_per_seq = 1
        self.position_embedding_type = 0
        self.attention_chunk_size = 0
        self.is_mla_enable = False


def test_trtllm_attention_metadata_caches_distinct_seq_lens(monkeypatch):
    monkeypatch.setattr(
        visual_trtllm,
        "BaseTrtllmAttentionMetadata",
        _FakeBaseTrtllmAttentionMetadata,
    )
    attention_metadata_state = {}
    metadata = visual_trtllm.TrtllmAttentionMetadata(
        attention_metadata_state=attention_metadata_state,
    )

    first_seq_lens = torch.tensor([64], dtype=torch.int32)
    first_metadata = metadata.prepare(batch_size=1, seq_lens=first_seq_lens)
    first_seq_lens.fill_(999)

    second_metadata = metadata.prepare(batch_size=1, seq_lens=torch.tensor([96], dtype=torch.int32))
    first_metadata_again = metadata.prepare(
        batch_size=1,
        seq_lens=torch.tensor([64], dtype=torch.int32),
    )

    assert first_metadata is first_metadata_again
    assert first_metadata is not second_metadata
    assert first_metadata.prepare_calls == 1
    assert second_metadata.prepare_calls == 1

    assert torch.equal(first_metadata.seq_lens, torch.tensor([64], dtype=torch.int32))
    assert torch.equal(second_metadata.seq_lens, torch.tensor([96], dtype=torch.int32))


def test_trtllm_block_sparse_layers_share_component_scoped_runtime():
    attention_metadata_state = create_attention_metadata_state()
    owners = [
        _FakeBlockSparseAttention(layer_idx, attention_metadata_state) for layer_idx in range(2)
    ]
    block_sparse_fmhas = [PrimsTSBlockSparseFmha(owner) for owner in owners]

    assert isinstance(block_sparse_fmhas[0]._runtime, PrimsTSBlockSparseRuntime)
    assert block_sparse_fmhas[0]._runtime is block_sparse_fmhas[1]._runtime


def test_pipeline_cleanup_releases_graph_captured_attention_state_in_order():
    cleanup_order = []
    graph_runner = Mock()
    graph_runner.clear.side_effect = lambda: cleanup_order.append("graphs")
    builder = Mock()
    builder.clear.side_effect = lambda: cleanup_order.append("vsa_builder")
    metadata_cache = Mock()
    metadata_cache.clear.side_effect = lambda: cleanup_order.append("metadata_cache")
    runtime = Mock()
    runtime.clear.side_effect = lambda: cleanup_order.append("runtime")
    pipeline = object.__new__(WanPipeline)
    pipeline._profiler = Mock()
    pipeline._cuda_graph_runners = {"transformer": graph_runner}
    pipeline._vsa_metadata_builder = builder
    pipeline.pipeline_config = SimpleNamespace(
        model_configs={
            "transformer": SimpleNamespace(
                attention_metadata_state={
                    "metadata_cache": metadata_cache,
                    PRIMS_TS_BLOCK_SPARSE_RUNTIME_STATE_KEY: runtime,
                }
            )
        }
    )

    pipeline.cleanup()

    assert cleanup_order[0] == "graphs"
    assert set(cleanup_order[1:]) == {"metadata_cache", "runtime", "vsa_builder"}
    graph_runner.clear.assert_called_once_with()
    builder.clear.assert_called_once_with()
    metadata_cache.clear.assert_called_once_with()
    runtime.clear.assert_called_once_with()
