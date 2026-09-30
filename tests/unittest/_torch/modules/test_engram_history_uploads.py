# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async Engram history restores request lookback without cross-request reuse."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram import engram as engram_module

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    ),
]
_MULTIPLIERS = [17, 31, 47, 61]
_MODULI = [[101, 103], [107, 109], [113, 127]]


@pytest.fixture
def provider(monkeypatch):
    config = EngramConfig(layer_ids=[1], max_ngram_size=4, n_head_per_ngram=2)
    mapping = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(lookup_table=torch.arange(256) // 2),
        pad_id=0,
        layer_multipliers={1: torch.tensor(_MULTIPLIERS)},
        vocab_size_across_layers={1: _MODULI},
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    return EngramHashProvider(config)


def _reference_shifts(history, position):
    result, blocked = [], False
    for offset in range(4):
        value = history.get(position - offset, -1)
        blocked = blocked or value < 0
        result.append(0 if blocked else value)
    return result


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("masked", [False, True])
def test_engram_history_uploads_refresh_recycled_rows(provider, device, masked):
    device = torch.device(device)
    stream = torch.cuda.Stream() if device.type == "cuda" else None
    provider._ensure_on_device(device)
    observed = []
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        if stream is not None:
            stream.wait_stream(torch.cuda.default_stream())
            torch.cuda._sleep(2_000_000)
        for step, count in enumerate((1, 3, 2, 1, 3, 2)):
            request_ids = [100 * (step + 1) + index for index in range(count)]
            prefixes, masks = {}, {}
            lengths, query_positions, query_values, expected = [], [], [], []
            for index, request_id in enumerate(request_ids):
                start = (step + index) % 5
                tokens = [20 + step * 5 + index + offset for offset in range(min(start, 3))]
                prefixes[request_id] = (start, tokens)
                if masked and index % 2 == 0:
                    masks[request_id] = [offset != 1 for offset in range(len(tokens))]
                mask = masks.get(request_id, [True] * len(tokens))
                history = {
                    position: token // 2 if text else -1
                    for position, token, text in zip(
                        range(start - len(tokens), start), tokens, mask, strict=True
                    )
                }
                length = 1 + index % 2
                lengths.append(length)
                for offset in range(length):
                    position, compressed = start + offset, (100 + step * 7 + index + offset) // 2
                    query_positions.append(position)
                    query_values.append(compressed)
                    history[position] = compressed
                    expected.append(_reference_shifts(history, position))
            if step % 2:
                for request_id, (start, tokens) in prefixes.items():
                    provider.queue_history_seed(
                        request_id, start - len(tokens), tokens, masks.get(request_id)
                    )
            else:
                provider.seed_context_history(
                    request_ids,
                    prefixes,
                    max_seq_len=16,
                    device=device,
                    token_masks=masks if masked else None,
                )
            actual = provider._lookback_from_history(
                torch.tensor(query_values, dtype=torch.int64, device=device),
                torch.tensor(query_positions, dtype=torch.int64, device=device),
                torch.tensor(lengths, dtype=torch.int32, device="cpu"),
                request_ids,
                16,
            )
            observed.append((torch.stack(actual, dim=1).clone(), expected))
            assert not provider._pending_history_seeds
            assert set(request_ids).issubset(provider._history_row_of)
            assert len(set(provider._history_row_of.values())) == len(provider._history_row_of)
    if stream is not None:
        stream.synchronize()
    for actual, expected in observed:
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected, dtype=torch.int64), atol=0, rtol=0
        )
    assert provider._history.shape[0] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_engram_history_uploads_refresh_captured_hashes(provider):
    device = torch.device("cuda")
    input_ids = torch.tensor([90, 92], dtype=torch.int64, device=device)
    positions = torch.tensor([3, 3], dtype=torch.int64, device=device)
    lengths = torch.tensor([1, 1], dtype=torch.int32, device="cpu")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        provider.seed_context_history(
            [41, 97], {41: (3, [20, 22, 24]), 97: (3, [30, 32, 34])}, max_seq_len=16, device=device
        )
        cached = provider.compute_hashes(
            input_ids,
            position_ids=positions,
            request_ids=[41, 97],
            seq_lens_host=lengths,
            max_seq_len=16,
        )[1]
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = provider.compute_hashes(
            input_ids,
            position_ids=positions,
            request_ids=[41, 97],
            seq_lens_host=lengths,
            max_seq_len=16,
        )[1].clone()
    captured_pointer = cached.data_ptr()
    observed = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        for step in range(6):
            request_ids = [100 + 2 * step, 101 + 2 * step]
            histories = []
            for index, request_id in enumerate(request_ids):
                raw = [10 + step * 7 + index + offset for offset in range(3)]
                mask = [True, step % 2 == 0, True] if index == 0 else None
                provider.queue_history_seed(request_id, 0, raw, mask)
                histories.append(
                    {
                        position: token // 2 if mask is None or mask[position] else -1
                        for position, token in enumerate(raw)
                    }
                )
            input_ids.fill_(100 + step)
            refreshed = provider.refresh_captured_hashes(
                input_ids,
                position_ids=positions,
                request_ids=request_ids,
                seq_lens_host=lengths,
                max_seq_len=16,
            )[1]
            assert refreshed.data_ptr() == captured_pointer
            graph.replay()
            expected = []
            for history in histories:
                history[3] = (100 + step) // 2
                shifts = _reference_shifts(history, 3)
                row, mixed = [], shifts[0] * _MULTIPLIERS[0]
                for order in range(1, 4):
                    mixed ^= shifts[order] * _MULTIPLIERS[order]
                    row.extend(mixed % modulus for modulus in _MODULI[order - 1])
                expected.append(row)
            observed.append((actual.clone(), expected))
    stream.synchronize()
    for actual, expected in observed:
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=0
        )
