# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kimi K3-specific Mamba metadata."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_attention_metadata(context_lengths: list[int]) -> SimpleNamespace:
    seq_lens = torch.tensor(context_lengths or [1], dtype=torch.int)
    num_contexts = len(context_lengths)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=sum(context_lengths),
        kv_cache_manager=None,
        request_ids=None,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.zeros(num_contexts, dtype=torch.int),
        ),
    )


@pytest.mark.parametrize("lengths", [[128, 256], [1, 64, 65, 129], [17, 1, 17, 16], [300], []])
def test_prepare_packed_kda_metadata(lengths: list[int]) -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    slots = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device="cuda")
    metadata.state_indices.copy_(slots)
    metadata.prepare(_make_attention_metadata(lengths))
    assert metadata.kda_max_seqlen == max(lengths, default=0)
    torch.testing.assert_close(
        metadata.kda_seq_order,
        torch.tensor(
            sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)),
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        metadata.kda_cu_chunks,
        torch.tensor(
            list(accumulate(((length + 15) // 16 for length in lengths), initial=0)),
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(metadata.state_indices, slots)
    if lengths:
        expected = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="cuda"
        )
        torch.testing.assert_close(metadata.query_start_loc_long, expected)
        torch.testing.assert_close(metadata.query_start_loc, expected.int())
        assert metadata.query_start_loc_long.is_contiguous()
        assert metadata.query_start_loc.is_contiguous()
        torch.testing.assert_close(metadata.kda_cu_seqlens, expected)
        torch.testing.assert_close(metadata.kda_cu_seqlens_cpu, expected.cpu())
    else:
        assert metadata.kda_cu_seqlens is None
        assert metadata.kda_cu_seqlens_cpu is None


def test_prepare_refreshes_offsets_without_changing_storage() -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    metadata.prepare(_make_attention_metadata([17, 65]))
    ptrs = (
        metadata.query_start_loc.data_ptr(),
        metadata.query_start_loc_long.data_ptr(),
        metadata.state_indices.data_ptr(),
        metadata.kda_seq_order.data_ptr(),
        metadata.kda_cu_chunks.data_ptr(),
    )
    metadata.prepare(_make_attention_metadata([64, 18]))
    assert metadata.kda_max_seqlen == 64
    expected = torch.tensor([0, 64, 82], dtype=torch.int64, device="cuda")
    torch.testing.assert_close(metadata.query_start_loc_long, expected)
    torch.testing.assert_close(metadata.query_start_loc, expected.int())
    assert ptrs == (
        metadata.query_start_loc.data_ptr(),
        metadata.query_start_loc_long.data_ptr(),
        metadata.state_indices.data_ptr(),
        metadata.kda_seq_order.data_ptr(),
        metadata.kda_cu_chunks.data_ptr(),
    )
    metadata.prepare(_make_attention_metadata([]))
    assert metadata.kda_max_seqlen == 0


def test_prepare_schedule_excludes_decode_requests() -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    batch = _make_attention_metadata([17, 65])
    batch.seq_lens = torch.tensor([17, 65, 1, 1], dtype=torch.int)
    batch.seq_lens_cuda = batch.seq_lens.cuda()
    metadata.prepare(batch)
    torch.testing.assert_close(
        metadata.kda_seq_order, torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        metadata.kda_cu_chunks, torch.tensor([0, 2, 7], dtype=torch.int32, device="cuda")
    )


@pytest.mark.parametrize("heads", [6, 96])
@torch.inference_mode()
def test_module_with_prepared_schedule_matches_device_generation(heads: int) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("FlashInfer KDA requires SM100/SM103/SM107")
    pytest.importorskip("fla")
    from kda_module_test_utils import clone_pool, make_attention, make_batch

    module = make_attention(heads, hidden_size=768, bf16_weights=True)
    lengths = [17, 65]
    batch = make_batch(heads, lengths, hidden_size=768, initial="mixed", padded=True)
    prepared = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    prepared.prepare(_make_attention_metadata(lengths))
    prepared.state_indices[: batch.n].copy_(batch.indices)
    prepared.has_initial_states[: batch.n].copy_(batch.metadata.has_initial_states)
    prepared.use_initial_states = batch.metadata.use_initial_states
    states = [clone_pool(batch.state), clone_pool(batch.state)]
    convs = [batch.conv.clone(), batch.conv.clone()]
    outputs = [
        module.forward_prefill(
            batch.hidden, batch.cu, metadata, batch.n, conv, state, batch.indices
        )
        for metadata, conv, state in zip((batch.metadata, prepared), convs, states)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    torch.testing.assert_close(convs[0], convs[1], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["engine", "decomp", "bt16", "auto", "small-bh"])
@pytest.mark.parametrize("heads", [6, 96])
@torch.inference_mode()
def test_prepared_metadata_matches_device_generation_across_layers_and_replays(
    mode: str, heads: int
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("FlashInfer KDA requires SM100/SM103/SM107")
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops import kda_prefill

    torch.manual_seed(1901)
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    metadata.prepare(_make_attention_metadata([1, 15, 47, 65]))
    offsets = metadata.query_start_loc_long
    slots = torch.tensor([4, 1, 3, 2], dtype=torch.int32, device="cuda")
    q, k, v, g = [
        torch.randn(1, 128, heads, 128, dtype=torch.bfloat16, device="cuda") * 0.2 for _ in range(4)
    ]
    beta = torch.randn(1, 128, heads, dtype=torch.bfloat16, device="cuda")
    a_log = torch.linspace(-2, 1, heads, device="cuda")
    bias = torch.linspace(-1, 1, heads * 128, device="cuda")
    row_size = heads * 128 * 128
    initial = torch.randn(6, row_size + 128, dtype=torch.bfloat16, device="cuda") * 0.1
    initial[slots[::2].long(), :row_size] = 0
    # Two independent layer pools consume the same prepared scheduling buffers.
    backing = [initial.clone(), initial.clone()]
    states = [tensor[:, :row_size].view(6, heads, 128, 128) for tensor in backing]
    pointers = (
        offsets.data_ptr(),
        metadata.kda_seq_order.data_ptr(),
        metadata.kda_cu_chunks.data_ptr(),
    )

    def run(prepared: bool) -> list[torch.Tensor]:
        return [
            kda_prefill(
                q,
                k,
                v,
                g,
                beta,
                state,
                slots,
                offsets,
                a_log,
                bias,
                128**-0.5,
                -5.0,
                mode,
                128,
                metadata.kda_seq_order if prepared else None,
                metadata.kda_cu_chunks if prepared else None,
            )
            for state in states
        ]

    run(True)
    run(False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    # Fixed token/sequence capacity, but different ordering and chunk counts.
    for lengths in ([1, 15, 47, 65], [17, 15, 64, 32], [62, 1, 1, 64]):
        metadata.prepare(_make_attention_metadata(lengths))
        assert pointers == (
            metadata.query_start_loc_long.data_ptr(),
            metadata.kda_seq_order.data_ptr(),
            metadata.kda_cu_chunks.data_ptr(),
        )
        immutable = [
            q,
            k,
            v,
            g,
            beta,
            offsets,
            slots,
            a_log,
            bias,
            metadata.kda_seq_order,
            metadata.kda_cu_chunks,
        ]
        snapshots = [tensor.clone() for tensor in immutable]
        for tensor in backing:
            tensor.copy_(initial)
        expected = run(False)
        expected_states = [tensor.clone() for tensor in backing]
        for replay in (False, True):
            for tensor in backing:
                tensor.copy_(initial)
            if replay:
                graph.replay()
                actual = captured
            else:
                actual = run(True)
            for output, reference, tensor, reference_state in zip(
                actual, expected, backing, expected_states
            ):
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                torch.testing.assert_close(tensor, reference_state, rtol=0, atol=0)
                torch.testing.assert_close(tensor[[0, 5]], initial[[0, 5]], rtol=0, atol=0)
                torch.testing.assert_close(
                    tensor[:, row_size:], initial[:, row_size:], rtol=0, atol=0
                )
            for tensor, snapshot in zip(immutable, snapshots):
                torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


def test_k3_prepare_materializes_stable_aligned_generation_indices() -> None:
    class KdaCacheManager:
        use_kda_replay_update = True

        def __init__(self) -> None:
            self._state_indices_buffer = DeviceCopyBuffer(
                torch.tensor([9, 4, 7], dtype=torch.int32, device="cuda")
            )

        def get_state_indices(self, request_ids: list[int], is_padding: list[bool]) -> torch.Tensor:
            return self._state_indices_buffer.tensor[: len(request_ids)]

    manager = KdaCacheManager()
    metadata = KimiK3MambaMetadata(max_batch_size=3, chunk_size=8, max_num_tokens=8)
    seq_lens = torch.tensor([2, 1, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=1,
        num_ctx_tokens=2,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int),
        ),
    )

    metadata.prepare(attn_metadata)

    assert metadata.state_indices[1:].data_ptr() % 16 != 0
    assert metadata.generation_state_indices is not None
    assert metadata.generation_state_indices.data_ptr() % 16 == 0
    aligned_data_ptr = metadata.generation_state_indices.data_ptr()
    torch.testing.assert_close(
        metadata.generation_state_indices,
        torch.tensor([4, 7], dtype=torch.int32, device="cuda"),
    )

    metadata.prepare(attn_metadata)

    assert metadata.generation_state_indices is not None
    assert metadata.generation_state_indices.data_ptr() == aligned_data_ptr

    attn_metadata.num_contexts = 0
    attn_metadata.num_ctx_tokens = 0
    metadata.prepare(attn_metadata)

    assert metadata.generation_state_indices is None
