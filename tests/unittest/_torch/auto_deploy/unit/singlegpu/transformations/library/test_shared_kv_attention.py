# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.torch_backend_attention import (
    TorchBackendAttention,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import InsertCachedAttentionConfig
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import _InsertCachedOperator


class _TinySharedKVModule(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], 2, 4)
        regular = torch.ops.auto_deploy.torch_attention(
            qkv,
            qkv,
            qkv,
            None,
            0.0,
            True,
            1.0,
            None,
            None,
            None,
            "bsnd",
            0,
        )
        shared = torch.ops.auto_deploy.torch_attention_shared_kv(
            qkv,
            qkv,
            qkv,
            None,
            0.0,
            True,
            1.0,
            None,
            None,
            None,
            "bsnd",
            1,
            0,
        )
        return regular + shared


def test_shared_kv_transform_aliases_source_cache_placeholders():
    module = _TinySharedKVModule().eval()
    gm = torch_export_to_gm(module, (torch.randn(1, 4, 8),))

    cm = CachedSequenceInterface(
        max_seq_len=16,
        max_batch_size=2,
        max_num_tokens=16,
        device="cpu",
    )
    transform = _InsertCachedOperator(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="torch")
    )
    gm, info = transform._apply(gm, cm, factory=None, shared_config=SharedConfig())

    assert info.num_matches == 2

    placeholder_names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
    assert placeholder_names.count("k_cache_0") == 1
    assert placeholder_names.count("v_cache_0") == 1
    assert "k_cache_1" not in placeholder_names
    assert "v_cache_1" not in placeholder_names

    cached_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    regular_node = next(
        node
        for node in cached_nodes
        if node.target == torch.ops.auto_deploy.torch_cached_attention_with_cache.default
    )
    shared_node = next(
        node
        for node in cached_nodes
        if node.target == torch.ops.auto_deploy.torch_cached_shared_kv_attention_with_cache.default
    )

    assert regular_node.args[8] is shared_node.args[8]
    assert regular_node.args[9] is shared_node.args[9]


def test_shared_kv_cached_attention_reads_without_writing():
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    dummy_k = torch.full((1, 1, 2, 2), 123.0, dtype=torch.float32)
    dummy_v = torch.full((1, 1, 2, 2), -456.0, dtype=torch.float32)

    k_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]], [[0.25, 0.0], [0.0, 0.25]]]],
        dtype=torch.float32,
    )
    v_cache = torch.tensor(
        [[[[10.0, 1.0], [2.0, 20.0]], [[30.0, 3.0], [4.0, 40.0]], [[50.0, 5.0], [6.0, 60.0]]]],
        dtype=torch.float32,
    )
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()

    output = torch.ops.auto_deploy.torch_cached_shared_kv_attention_with_cache(
        q,
        dummy_k,
        dummy_v,
        torch.tensor([0, 0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        k_cache,
        v_cache,
        1.0,
        None,
        None,
        None,
    )

    assert torch.equal(k_cache, k_cache_before)
    assert torch.equal(v_cache, v_cache_before)

    k_for_attn = k_cache_before[0, :3].transpose(0, 1)
    v_for_attn = v_cache_before[0, :3].transpose(0, 1)
    logits = torch.matmul(q[0, 0].unsqueeze(1), k_for_attn.transpose(-2, -1))
    weights = torch.softmax(logits, dim=-1)
    expected = torch.matmul(weights, v_for_attn).squeeze(1).unsqueeze(0).unsqueeze(0)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_torch_backend_attention_metadata_for_shared_kv_node():
    module = _TinySharedKVModule().eval()
    gm = torch_export_to_gm(module, (torch.randn(1, 4, 8),))
    source_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    regular = next(node for node in source_nodes if node.target == torch.ops.auto_deploy.torch_attention.default)
    shared = next(
        node for node in source_nodes if node.target == torch.ops.auto_deploy.torch_attention_shared_kv.default
    )

    assert TorchBackendAttention.get_layer_idx(regular) == 0
    assert TorchBackendAttention.get_layer_idx(shared) == 1
    assert TorchBackendAttention.get_shared_kv_source_layer_idx(regular) is None
    assert TorchBackendAttention.get_shared_kv_source_layer_idx(shared) == 0
