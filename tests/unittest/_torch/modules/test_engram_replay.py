# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded seed semantics with a deterministic compressed-tokenizer mapping."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider


def provider(layer_ids=(0,)):
    # Non-identity compression plus a DEAD token catch raw-ID and padding seeds.
    table = torch.tensor([0, 7, 2, -1, 5, 11, 1, 9, 4, 8], dtype=torch.int64)
    constants = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(lookup_table=table),
        vocab_size_across_layers={layer: [[101], [103], [107]] for layer in layer_ids},
        layer_multipliers={
            layer: torch.tensor([3, 5, 7, 11], dtype=torch.int64) for layer in layer_ids
        },
        pad_id=0,
    )
    config = EngramConfig(layer_ids=list(layer_ids), max_ngram_size=4, n_head_per_ngram=1)
    with patch(
        "tensorrt_llm._torch.modules.engram.engram.NgramHashMapping", return_value=constants
    ):
        return EngramHashProvider(config)


def hashes(p, ids, start, request_id=7):
    lengths = torch.tensor([ids.numel()], dtype=torch.int32)
    return p.compute_hashes(
        ids,
        position_ids=torch.arange(start, start + ids.numel(), device=ids.device),
        seq_lens_host=lengths,
        request_ids=[request_id],
        max_seq_len=64,
    )[0].clone()


@pytest.mark.parametrize("start", [0, 1, 2, 3, 17])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recovery_input_seed_matches_full_history_with_dead_tokens(start, device):
    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] * 4, device=device)
    expected = hashes(provider(), ids, 0)[start:]
    p = provider()
    # Force a row recycle and fill unrelated content before recovery.
    hashes(p, ids.flip(0), 0, request_id=91)
    tokens = ids.tolist()
    request = SimpleNamespace(
        py_request_id=7,
        is_dummy=False,
        context_current_position=start + 1 if start else 0,
        py_ced_replay=EncoderReplay(7, 1, start + 1, start) if start else None,
        get_tokens_range=Mock(side_effect=lambda beam, begin, end: tokens[begin:end]),
    )
    generation = SimpleNamespace(
        py_request_id=8,
        is_dummy=False,
        get_tokens_range=Mock(side_effect=AssertionError("stale CPU generation tokens")),
    )
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(max_seq_len=64),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            engram_hash_provider=p,
            use_engram=True,
            embed_tokens=SimpleNamespace(weight=ids),
        )
    )
    batch = SimpleNamespace(
        context_requests=[request],
        generation_requests=[generation],
        all_requests=lambda: [request, generation],
    )
    with patch.object(p, "seed_context_history", wraps=p.seed_context_history) as seed:
        DeepseekV41ForCausalLM.prepare_request_inputs(model, batch, metadata)
        assert seed.call_args.args == ([7, 8], {7: (start, tokens[max(0, start - 3) : start])})
    generation.get_tokens_range.assert_not_called()
    request.get_tokens_range.assert_called_once_with(0, max(0, start - 3), start)
    torch.testing.assert_close(hashes(p, ids[start:], start), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seeding_whole_mixed_batch_preserves_each_request(device):
    p = provider()
    a = torch.tensor([1, 2, 4, 5, 6, 7], device=device)
    b = torch.tensor([8, 7, 6, 3, 2, 1], device=device)
    reference = torch.cat((hashes(provider(), a, 0)[4:], hashes(provider(), b, 0)[4:]))
    p.seed_context_history(
        [7, 8],
        {7: (4, a[1:4].tolist()), 8: (4, b[1:4].tolist())},
        max_seq_len=64,
        device=a.device,
    )
    result = p.compute_hashes(
        torch.cat((a[4:], b[4:])),
        position_ids=torch.tensor([4, 5, 4, 5], device=device),
        seq_lens_host=torch.tensor([2, 2]),
        request_ids=[7, 8],
        max_seq_len=64,
    )[0]
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_padded_graph_refresh_preserves_history_and_request_order():
    p = provider((0, 1))
    dummy = torch.zeros(4, dtype=torch.long, device="cuda")
    p.compute_hashes(dummy)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        p.compute_hashes(dummy)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = {layer: value.clone() for layer, value in p.compute_hashes(dummy).items()}
    captured_ptrs = {layer: value.data_ptr() for layer, value in p._cached_hashes.items()}
    histories = {7: [], 8: []}
    for request_ids, tokens in (([7, 8], [1, 2]), ([8], [4]), ([8, 7], [5, 6])):
        positions, expected = [], []
        for rid, token in zip(request_ids, tokens):
            positions.append(len(histories[rid]))
            histories[rid].append(token)
            full = torch.tensor(histories[rid], device="cuda")
            expected.append(hashes(provider(), full, 0)[-1])
        refreshed = p.refresh_captured_hashes(
            torch.tensor(tokens, device="cuda"),
            position_ids=torch.tensor(positions, device="cuda"),
            request_ids=request_ids,
            seq_lens_host=torch.ones(len(tokens), dtype=torch.int32),
            max_seq_len=64,
            padded_num_tokens=4,
        )
        assert {layer: value.data_ptr() for layer, value in refreshed.items()} == captured_ptrs
        graph.replay()
        for output in outputs.values():
            torch.testing.assert_close(output[: len(tokens)], torch.stack(expected), rtol=0, atol=0)
            assert torch.count_nonzero(output[len(tokens) :]).item() == 0
        assert set(p._history_row_of) == {7, 8}
