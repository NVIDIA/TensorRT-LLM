# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.compile.piecewise_utils import is_dynamic_cached_op
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.flashinfer_attention import (
    FlashInferAttention,
)
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


def _context_meta(seq_len: int):
    return (
        torch.tensor([1, 0, 0], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
    )


def _decode_meta(input_pos: int):
    return (
        torch.tensor([0, 0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([input_pos], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
    )


def _manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int | None = None,
) -> torch.Tensor:
    batch, seq_len_q, num_heads, _ = q.shape
    _, seq_len_k, num_kv_heads, _ = k.shape
    if num_heads != num_kv_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1))
    causal_mask = torch.triu(
        torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device),
        diagonal=seq_len_k - seq_len_q + 1,
    )
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    if sliding_window is not None:
        query_positions = torch.arange(seq_len_k - seq_len_q, seq_len_k, device=scores.device)
        key_positions = torch.arange(seq_len_k, device=scores.device)
        pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        sliding_window_mask = (pos_diff < 0) | (pos_diff >= sliding_window)
        scores = scores.masked_fill(sliding_window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v_t).transpose(1, 2)


def _make_layer_inputs(offset: float, seq_len: int, decode: bool = False):
    base_q = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]] if decode else
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, -0.5]],
            [[0.25, 0.75], [0.75, 0.25]],
        ],
        dtype=torch.float32,
    )
    base_k = torch.tensor(
        [[[1.0, 0.0]]] if decode else
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]],
        dtype=torch.float32,
    )
    base_v = torch.tensor(
        [[[10.0, 1.0]]] if decode else
        [[[10.0, 1.0]], [[2.0, 20.0]], [[30.0, 3.0]]],
        dtype=torch.float32,
    )
    q = (base_q + offset).unsqueeze(0)
    k = (base_k + offset).unsqueeze(0)
    v = (base_v + offset * 10.0).unsqueeze(0)
    assert q.shape[1] == seq_len
    return q, k, v


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


def test_flashinfer_backend_attention_metadata_for_shared_kv_node():
    module = _TinySharedKVModule().eval()
    gm = torch_export_to_gm(module, (torch.randn(1, 4, 8),))
    source_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    regular = next(node for node in source_nodes if node.target == torch.ops.auto_deploy.torch_attention.default)
    shared = next(
        node for node in source_nodes if node.target == torch.ops.auto_deploy.torch_attention_shared_kv.default
    )

    assert FlashInferAttention.get_layer_idx(regular) == 0
    assert FlashInferAttention.get_layer_idx(shared) == 1
    assert FlashInferAttention.get_shared_kv_source_layer_idx(regular) is None
    assert FlashInferAttention.get_shared_kv_source_layer_idx(shared) == 0
    assert (
        FlashInferAttention.get_cached_attention_op_for_source_node(regular)
        == torch.ops.auto_deploy.flashinfer_attention_mha_with_cache.default
    )
    assert (
        FlashInferAttention.get_cached_attention_op_for_source_node(shared)
        == torch.ops.auto_deploy.flashinfer_attention_shared_kv_mha_with_cache.default
    )


def test_shared_kv_transform_aliases_source_cache_placeholders_for_flashinfer():
    module = _TinySharedKVModule().eval()
    gm = torch_export_to_gm(module, (torch.randn(1, 4, 8),))

    cm = CachedSequenceInterface(
        max_seq_len=16,
        max_batch_size=2,
        max_num_tokens=16,
        device="cpu",
    )
    transform = _InsertCachedOperator(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="flashinfer")
    )
    gm, info = transform._apply(gm, cm, factory=None, shared_config=SharedConfig())

    assert info.num_matches == 2

    placeholder_names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
    assert placeholder_names.count("kv_cache_0") == 1
    assert "kv_cache_1" not in placeholder_names

    cached_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    regular_node = next(
        node
        for node in cached_nodes
        if node.target == torch.ops.auto_deploy.flashinfer_attention_mha_with_cache.default
    )
    shared_node = next(
        node
        for node in cached_nodes
        if node.target == torch.ops.auto_deploy.flashinfer_attention_shared_kv_mha_with_cache.default
    )

    assert regular_node.args[11] is shared_node.args[11]


def test_flashinfer_shared_kv_cached_attention_is_dynamic_for_piecewise():
    shared_op_name = torch.ops.auto_deploy.flashinfer_attention_shared_kv_mha_with_cache.default.name()

    class _FakeNode:
        op = "call_function"

        def __init__(self, target):
            self.target = target

    assert "flashinfer_attention_shared_kv_mha_with_cache" in shared_op_name
    assert is_dynamic_cached_op(_FakeNode(torch.ops.auto_deploy.flashinfer_attention_shared_kv_mha_with_cache.default))


@torch.no_grad()
def test_flashinfer_shared_kv_cached_attention_reads_aliased_cache_without_writing():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    head_dim = 64
    q = torch.zeros((1, 1, 1, head_dim), dtype=torch.float16, device=device)
    q[0, 0, 0, 0] = 1.0
    dummy_k = torch.full((1, 1, 1, head_dim), 9.0, dtype=torch.float16, device=device)
    dummy_v = torch.full((1, 1, 1, head_dim), 7.0, dtype=torch.float16, device=device)

    owner_k = torch.zeros((1, 3, 1, head_dim), dtype=torch.float16, device=device)
    owner_k[0, 0, 0, 0] = 1.0
    owner_k[0, 1, 0, 1] = 1.0
    owner_k[0, 2, 0, 0] = 1.0
    owner_k[0, 2, 0, 1] = 1.0
    owner_v = torch.zeros((1, 3, 1, head_dim), dtype=torch.float16, device=device)
    owner_v[0, 0, 0, 0] = 10.0
    owner_v[0, 1, 0, 1] = 20.0
    owner_v[0, 2, 0, 0] = 30.0
    owner_v[0, 2, 0, 1] = 3.0
    kv_cache = torch.zeros((1, 2, 1, 32, head_dim), dtype=torch.float16, device=device)
    kv_cache[0, 0, 0, :3, :] = owner_k[0, :, 0, :]
    kv_cache[0, 1, 0, :3, :] = owner_v[0, :, 0, :]
    kv_cache_before = kv_cache.clone()

    batch_info_host = torch.tensor([0, 0, 1], dtype=torch.int32, device="cpu")
    cu_seqlen_host = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    cu_num_pages = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_num_pages_host = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    cache_loc = torch.tensor([0], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([3], dtype=torch.int32, device=device)
    last_page_len_host = torch.tensor([3], dtype=torch.int32, device="cpu")
    seq_len_with_cache_host = torch.tensor([3], dtype=torch.int32, device="cpu")
    batch_indices = torch.zeros(1, dtype=torch.int32, device=device)
    positions = torch.zeros(1, dtype=torch.int32, device=device)

    output = torch.ops.auto_deploy.flashinfer_attention_shared_kv_mha_with_cache(
        q,
        dummy_k,
        dummy_v,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        batch_indices,
        positions,
        kv_cache,
        1.0,
        None,
        1.0,
        1.0,
    )

    torch.testing.assert_close(kv_cache, kv_cache_before, rtol=0.0, atol=0.0)

    expected = _manual_attention(q.float(), owner_k.float(), owner_v.float()).to(output.dtype)
    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_shared_kv_six_layer_stack_matches_reference_for_prefill_and_decode():
    layer_sources = {4: 2, 5: 3}
    sliding_layers = {2, 4}
    prefill_len = 3
    decode_pos = prefill_len
    owner_caches = {
        layer_idx: (
            torch.zeros(1, 8, 1, 2, dtype=torch.float32),
            torch.zeros(1, 8, 1, 2, dtype=torch.float32),
        )
        for layer_idx in range(4)
    }
    owner_history = {}

    for layer_idx in range(6):
        q_prefill, k_prefill, v_prefill = _make_layer_inputs(offset=float(layer_idx), seq_len=prefill_len)
        batch_info_host, seq_len, input_pos, slot_idx, cu_seqlen = _context_meta(prefill_len)
        sliding_window = 2 if layer_idx in sliding_layers else None

        if layer_idx in layer_sources:
            source_idx = layer_sources[layer_idx]
            k_cache, v_cache = owner_caches[source_idx]
            output_prefill = torch.ops.auto_deploy.torch_cached_shared_kv_attention_with_cache(
                q_prefill,
                k_prefill,
                v_prefill,
                batch_info_host,
                seq_len,
                input_pos,
                slot_idx,
                cu_seqlen,
                k_cache,
                v_cache,
                1.0,
                None,
                sliding_window,
                None,
            )
            expected_prefill = _manual_attention(
                q_prefill,
                owner_history[source_idx]["k_prefill"],
                owner_history[source_idx]["v_prefill"],
                sliding_window=sliding_window,
            )
        else:
            k_cache, v_cache = owner_caches[layer_idx]
            output_prefill = torch.ops.auto_deploy.torch_cached_attention_with_cache(
                q_prefill,
                k_prefill,
                v_prefill,
                batch_info_host,
                seq_len,
                input_pos,
                slot_idx,
                cu_seqlen,
                k_cache,
                v_cache,
                1.0,
                None,
                sliding_window,
                None,
            )
            expected_prefill = _manual_attention(
                q_prefill,
                k_prefill,
                v_prefill,
                sliding_window=sliding_window,
            )
            owner_history[layer_idx] = {
                "k_prefill": k_prefill.clone(),
                "v_prefill": v_prefill.clone(),
            }
            torch.testing.assert_close(k_cache[0, :prefill_len], k_prefill[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(v_cache[0, :prefill_len], v_prefill[0], rtol=0.0, atol=0.0)

        torch.testing.assert_close(output_prefill, expected_prefill, rtol=1e-5, atol=1e-5)

    for layer_idx in range(6):
        q_decode, k_decode, v_decode = _make_layer_inputs(
            offset=100.0 + float(layer_idx), seq_len=1, decode=True
        )
        batch_info_host, seq_len, input_pos, slot_idx, cu_seqlen = _decode_meta(decode_pos)
        sliding_window = 2 if layer_idx in sliding_layers else None

        if layer_idx in layer_sources:
            source_idx = layer_sources[layer_idx]
            k_cache, v_cache = owner_caches[source_idx]
            k_cache_before = k_cache.clone()
            v_cache_before = v_cache.clone()
            output_decode = torch.ops.auto_deploy.torch_cached_shared_kv_attention_with_cache(
                q_decode,
                k_decode,
                v_decode,
                batch_info_host,
                seq_len,
                input_pos,
                slot_idx,
                cu_seqlen,
                k_cache,
                v_cache,
                1.0,
                None,
                sliding_window,
                None,
            )
            torch.testing.assert_close(k_cache, k_cache_before, rtol=0.0, atol=0.0)
            torch.testing.assert_close(v_cache, v_cache_before, rtol=0.0, atol=0.0)
            expected_k = owner_history[source_idx]["k_full"]
            expected_v = owner_history[source_idx]["v_full"]
        else:
            k_cache, v_cache = owner_caches[layer_idx]
            output_decode = torch.ops.auto_deploy.torch_cached_attention_with_cache(
                q_decode,
                k_decode,
                v_decode,
                batch_info_host,
                seq_len,
                input_pos,
                slot_idx,
                cu_seqlen,
                k_cache,
                v_cache,
                1.0,
                None,
                sliding_window,
                None,
            )
            expected_k = torch.cat([owner_history[layer_idx]["k_prefill"], k_decode], dim=1)
            expected_v = torch.cat([owner_history[layer_idx]["v_prefill"], v_decode], dim=1)
            owner_history[layer_idx]["k_full"] = expected_k
            owner_history[layer_idx]["v_full"] = expected_v
            torch.testing.assert_close(k_cache[0, : decode_pos + 1], expected_k[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(v_cache[0, : decode_pos + 1], expected_v[0], rtol=0.0, atol=0.0)

        expected_decode = _manual_attention(
            q_decode,
            expected_k,
            expected_v,
            sliding_window=sliding_window,
        )
        torch.testing.assert_close(output_decode, expected_decode, rtol=1e-5, atol=1e-5)
