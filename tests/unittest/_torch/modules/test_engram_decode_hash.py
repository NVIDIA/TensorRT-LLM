# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode hashing preserves history, integer arithmetic, and graph storage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram import engram as engram_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def providers(monkeypatch):
    def create(ngram=4, heads=3, layers=(1, 15), pad=0):
        # Large odd multipliers exercise exact int64 arithmetic rather than
        # float approximations. Negative padding also exercises signed modulo.
        bound = torch.iinfo(torch.int64).max // 256
        mapping = SimpleNamespace(
            compressed_tokenizer=SimpleNamespace(lookup_table=torch.arange(256) // 2),
            pad_id=pad,
            layer_multipliers={
                layer: torch.tensor([bound - 2 * (layer + i) for i in range(ngram)])
                for layer in layers
            },
            vocab_size_across_layers={
                layer: [
                    [101 + 2 * (order * heads + head) for head in range(heads)]
                    for order in range(ngram - 1)
                ]
                for layer in layers
            },
        )
        monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
        config = EngramConfig(layer_ids=list(layers), max_ngram_size=ngram, n_head_per_ngram=heads)
        candidate, reference = EngramHashProvider(config), EngramHashProvider(config)
        monkeypatch.setattr(reference, "_compute_decode_hashes", lambda *args: None)
        return candidate, reference

    return create


def _arguments(ids, positions, request_ids, *, dtype=torch.int64, padded=None, mask=None):
    return dict(
        input_ids=torch.tensor(ids, dtype=dtype, device="cuda"),
        position_ids=torch.tensor(positions, dtype=dtype, device="cuda"),
        request_ids=request_ids,
        seq_lens_host=torch.ones(len(request_ids), dtype=torch.int32, device="cpu"),
        max_seq_len=64,
        token_mask=None if mask is None else torch.tensor(mask, dtype=torch.bool, device="cuda"),
        padded_num_tokens=padded,
    )


def _assert_hashes(actual, expected):
    assert actual.keys() == expected.keys()
    for layer in actual:
        torch.testing.assert_close(actual[layer], expected[layer], atol=0, rtol=0)


@pytest.mark.parametrize("ngram,heads,layers", [(2, 1, (1,)), (3, 8, (1, 15)), (4, 3, (1, 8, 15))])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pad,masked", [(0, False), (-7, True)])
def test_engram_decode_hash_matches_general_history(
    providers, monkeypatch, ngram, heads, layers, dtype, pad, masked
):
    candidate, reference = providers(ngram, heads, layers, pad)
    # Prove this parity comparison exercises the fused implementation.
    monkeypatch.setattr(candidate, "_lookback_from_history", Mock(side_effect=AssertionError))
    positions = {}
    cohorts = (
        [41, 97, 111],
        [111, 41, 97],
        [111, 23],
        [23, 111],
        [201, 202, 203, 204, 205],
        [201, 205],
        [301, 302],
        [301, 302],
    )
    observed = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step, request_ids in enumerate(cohorts):
            for index, request_id in enumerate(request_ids):
                if request_id not in positions or step == 7:
                    # Re-seeding retained IDs may reuse the identical physical
                    # row map, but must still clear and restore their history.
                    positions[request_id] = ngram - 1
                    raw = [-5, 47 + step, 300][: ngram - 1]
                    seed_mask = [index % 2 == 0] + [True] * (len(raw) - 1) if masked else None
                    for provider in (candidate, reference):
                        provider.queue_history_seed(request_id, 0, raw, seed_mask)
            ids = [
                -9 if index == 0 else 270 if index == 1 else step * 17 + index
                for index in range(len(request_ids))
            ]
            mask = (
                [not (step % 3 == 1 and index == 0) for index in range(len(ids))]
                if masked
                else None
            )
            kwargs = _arguments(
                ids,
                [positions[rid] for rid in request_ids],
                list(request_ids),
                dtype=dtype,
                padded=len(ids) + 3,
                mask=mask,
            )
            actual, expected = (
                candidate.compute_hashes(**kwargs),
                reference.compute_hashes(**kwargs),
            )
            observed.append(
                (
                    {layer: value.clone() for layer, value in actual.items()},
                    {layer: value.clone() for layer, value in expected.items()},
                )
            )
            assert not candidate._pending_history_seeds
            assert candidate._history_row_of == reference._history_row_of
            for request_id in request_ids:
                positions[request_id] += 1
            if step == 1:
                for provider in (candidate, reference):
                    provider.release_request_state(97)
            if step == 5:
                for provider in (candidate, reference):
                    provider.release_request_state(201)
                    provider.release_request_state(205)
    stream.synchronize()
    for actual, expected in observed:
        _assert_hashes(actual, expected)
    torch.testing.assert_close(candidate._history, reference._history, atol=0, rtol=0)


@pytest.mark.parametrize("masked", [False, True])
def test_engram_decode_hash_refreshes_padded_graph_buffers(providers, masked):
    candidate, reference = providers()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    initial = _arguments([10, 20, 30, 40], [0] * 4, [1, 2, 3, 4], padded=4)
    with torch.cuda.stream(stream):
        cached = candidate.compute_hashes(**initial)
    stream.synchronize()
    pointers = {layer: value.data_ptr() for layer, value in cached.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = {
            layer: value.clone() for layer, value in candidate.compute_hashes(**initial).items()
        }
    observed = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        for step, count in enumerate((1, 3, 2, 4, 1, 3)):
            request_ids = list(range(100 * (step + 1), 100 * (step + 1) + count))
            for provider in (candidate, reference):
                for request_id in request_ids:
                    provider.queue_history_seed(
                        request_id, 0, [11, 12, 13], [True, not masked, True]
                    )
            kwargs = _arguments(
                [30 + step + index for index in range(count)],
                [3] * count,
                request_ids,
                padded=4,
                mask=[index % 2 == 0 for index in range(count)] if masked else None,
            )
            refreshed = candidate.refresh_captured_hashes(**kwargs)
            assert {layer: value.data_ptr() for layer, value in refreshed.items()} == pointers
            expected = reference.compute_hashes(**kwargs)
            graph.replay()
            observed.append(
                (
                    {layer: value.clone() for layer, value in captured.items()},
                    {layer: value.clone() for layer, value in expected.items()},
                )
            )
    stream.synchronize()
    for actual, expected in observed:
        _assert_hashes(actual, expected)


def test_engram_decode_row_maps_are_immutable_and_stream_local(providers):
    provider, _ = providers()
    provider._ensure_on_device(torch.device("cuda"))
    provider._reserve_history_rows([1, 2], 64, torch.device("cuda"))
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    observed, pointers = [], []
    for stream in streams:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(2_000_000)
            first = provider._decode_history_rows([0, 1], (1, 1), torch.device("cuda"))
            assert provider._decode_history_rows([0, 1], (1, 1), torch.device("cuda")) is first
            pointers.append(first.data_ptr())
            observed.append((first.clone(), [0, 1]))
            for index in range(12):
                rows = [index, index + 1]
                row_map = provider._decode_history_rows(rows, (1, 1), torch.device("cuda"))
                observed.append((row_map.clone(), rows))
                assert provider._decode_row_map[1] is row_map
            # Replacing the cached map must not rewrite pending device data.
            observed.append((first.clone(), [0, 1]))
    assert pointers[0] != pointers[1]
    for stream in streams:
        stream.synchronize()
    for actual, expected in observed:
        torch.testing.assert_close(actual.cpu(), torch.tensor(expected), atol=0, rtol=0)


@pytest.mark.parametrize(
    "case",
    [
        "packed",
        "missing_positions",
        "missing_ids",
        "missing_lengths",
        "device_lengths",
        "short_lengths",
        "float_lengths",
        "strided_tokens",
        "strided_positions",
        "float_tokens",
        "missing_ceiling",
        "position_count",
        "strided_mask",
        "cpu",
    ],
)
def test_engram_decode_hash_rejects_unsupported_geometry(providers, case):
    provider, _ = providers()
    kwargs = _arguments([10, 20], [0, 0], [1, 2])
    if case == "packed":
        kwargs["seq_lens_host"] = torch.tensor([2, 0], dtype=torch.int32)
    elif case == "missing_positions":
        kwargs["position_ids"] = None
    elif case == "missing_ids":
        kwargs["request_ids"] = None
    elif case == "missing_lengths":
        kwargs["seq_lens_host"] = None
    elif case == "device_lengths":
        kwargs["seq_lens_host"] = kwargs["seq_lens_host"].cuda()
    elif case == "short_lengths":
        kwargs["seq_lens_host"] = kwargs["seq_lens_host"][:1]
    elif case == "float_lengths":
        kwargs["seq_lens_host"] = kwargs["seq_lens_host"].float()
    elif case == "strided_tokens":
        kwargs["input_ids"] = torch.arange(4, device="cuda")[::2]
    elif case == "strided_positions":
        kwargs["position_ids"] = torch.arange(4, device="cuda")[::2]
    elif case == "float_tokens":
        kwargs["input_ids"] = kwargs["input_ids"].float()
    elif case == "missing_ceiling":
        kwargs["max_seq_len"] = None
    elif case == "position_count":
        kwargs["position_ids"] = kwargs["position_ids"][:1]
    elif case == "strided_mask":
        kwargs["token_mask"] = torch.ones(4, dtype=torch.bool, device="cuda")[::2]
    else:
        kwargs["input_ids"] = kwargs["input_ids"].cpu()
    assert provider._compute_decode_hashes(**kwargs) is None
    assert provider._history is None
    assert provider._decode_row_map is None


@pytest.mark.parametrize(
    "case", ["mask_dtype", "mask_shape", "negative_padding", "float_padding", "seed_too_long"]
)
def test_engram_decode_hash_preserves_validation(providers, case):
    provider, _ = providers()
    kwargs = _arguments([10, 20], [3, 3], [1, 2])
    if case == "mask_dtype":
        kwargs["token_mask"] = torch.ones(2, dtype=torch.int32, device="cuda")
    elif case == "mask_shape":
        kwargs["token_mask"] = torch.ones(1, dtype=torch.bool, device="cuda")
    elif case == "negative_padding":
        kwargs["padded_num_tokens"] = 1
    elif case == "float_padding":
        kwargs["padded_num_tokens"] = 2.0
    else:
        provider.queue_history_seed(1, 64, [10])
    with pytest.raises(ValueError):
        provider.compute_hashes(**kwargs)


def test_engram_decode_hash_keeps_packed_prefill_path(providers, monkeypatch):
    candidate, reference = providers()
    kwargs = _arguments([10, 11, 20, 21], [0, 1, 0, 1], [1, 2], padded=8)
    kwargs["seq_lens_host"] = torch.tensor([2, 2], dtype=torch.int32)
    original = candidate._lookback_from_history
    spy = Mock(wraps=original)
    monkeypatch.setattr(candidate, "_lookback_from_history", spy)
    _assert_hashes(candidate.compute_hashes(**kwargs), reference.compute_hashes(**kwargs))
    assert spy.call_count == 1
    assert candidate._decode_row_map is None


@pytest.mark.parametrize("position", [-64, -1, 0, 1, 63])
def test_engram_decode_hash_preserves_history_indexing_and_int32_storage(providers, position):
    candidate, reference = providers(pad=-7)
    for provider in (candidate, reference):
        provider._lookup_table[10] = 2**31 + 3
        provider._lookup_table[20] = 2**32 - 1
    kwargs = _arguments([10, 20], [position, position], [1, 2])
    _assert_hashes(candidate.compute_hashes(**kwargs), reference.compute_hashes(**kwargs))
    torch.testing.assert_close(candidate._history, reference._history, atol=0, rtol=0)


def test_engram_decode_hash_row_map_hit_still_consumes_seed(providers):
    candidate, reference = providers()
    kwargs = _arguments([40], [3], [41])
    for provider in (candidate, reference):
        provider.queue_history_seed(41, 0, [10, 20, 30])
    first = {layer: value.clone() for layer, value in candidate.compute_hashes(**kwargs).items()}
    reference.compute_hashes(**kwargs)
    row_map = candidate._decode_row_map[1]
    for provider in (candidate, reference):
        provider.queue_history_seed(41, 0, [51, 61, 71], [True, False, True])
    actual = candidate.compute_hashes(**kwargs)
    expected = reference.compute_hashes(**kwargs)
    assert candidate._decode_row_map[1] is row_map
    assert not candidate._pending_history_seeds
    _assert_hashes(actual, expected)
    assert any(not torch.equal(first[layer], value) for layer, value in actual.items())
    torch.testing.assert_close(candidate._history, reference._history, atol=0, rtol=0)


def test_engram_decode_hash_declines_history_on_another_device(providers):
    provider, _ = providers()
    provider.seed_context_history(
        [41], {41: (3, [10, 20, 30])}, max_seq_len=64, device=torch.device("cpu")
    )
    history = provider._history
    ownership = dict(provider._history_row_of)
    device = torch.device("cuda", torch.cuda.current_device())
    provider._ensure_on_device(device)
    assert provider._lookup_table.device == device
    assert all(value.device == device for value in provider._multipliers.values())
    assert history.device.type == "cpu"
    assert not provider._cached_hashes_store
    assert provider._compute_decode_hashes(**_arguments([40], [3], [41])) is None
    assert provider._history is history
    assert provider._history_row_of == ownership
    assert provider._decode_row_map is None


def test_engram_decode_hash_declines_cached_outputs_on_another_device(providers):
    provider, _ = providers()
    cached = provider.compute_hashes(torch.tensor([40], dtype=torch.int64, device="cpu"))
    pointers = {layer: value.data_ptr() for layer, value in cached.items()}
    assert all(value.device.type == "cpu" for value in cached.values())
    assert provider._history is None
    device = torch.device("cuda", torch.cuda.current_device())
    provider._ensure_on_device(device)
    assert provider._lookup_table.device == device
    assert provider._compute_decode_hashes(**_arguments([40], [3], [41])) is None
    assert provider._history is None
    assert provider._decode_row_map is None
    assert {layer: value.data_ptr() for layer, value in cached.items()} == pointers
    assert all(value.device.type == "cpu" for value in cached.values())


def _hashes_from_shared_history(provider, request_ids, positions, padded):
    """Integer oracle over the resolved scatter state, including alias winners."""
    history = provider._history.cpu().tolist()
    result = {}
    for layer in provider.config.layer_ids:
        multipliers = provider._multipliers[layer].cpu().tolist()
        rows = []
        for request_id, position in zip(request_ids, positions, strict=True):
            resident = history[provider._history_row_of[request_id]]
            window, blocked = [], False
            for offset in range(provider.config.max_ngram_size):
                value = resident[max(position - offset, 0)]
                blocked = blocked or position < offset or value == provider._DEAD
                window.append(provider._pad_id if blocked else value)
            hashes = []
            for order in range(2, provider.config.max_ngram_size + 1):
                mixed = 0
                for offset in range(order):
                    product = window[offset] * multipliers[offset]
                    mixed ^= (product + 2**63) % 2**64 - 2**63
                hashes.extend(
                    mixed % modulus for modulus in provider._moduli_ints[layer][order - 2]
                )
            rows.append(hashes)
        width = (provider.config.max_ngram_size - 1) * provider.config.n_head_per_ngram
        rows.extend([[0] * width for _ in range(padded - len(rows))])
        result[layer] = torch.tensor(rows, dtype=torch.int64)
    return result


@pytest.mark.parametrize("real_requests", [0, 9, 11, 12])
@pytest.mark.parametrize("masked", [False, True])
def test_engram_decode_aliases_preserve_general_hashes(
    providers, monkeypatch, real_requests, masked
):
    candidate, reference = providers(pad=-7)
    monkeypatch.setattr(candidate, "_lookback_from_history", Mock(side_effect=AssertionError))
    dummy_id = (1 << 64) - 1
    request_ids = list(range(real_requests)) + [dummy_id] * (16 - real_requests)
    for provider in (candidate, reference):
        for request_id in range(real_requests):
            provider.queue_history_seed(request_id, 0, [10, 20, 30], [True, not masked, True])
    kwargs = _arguments(
        list(range(90, 90 + real_requests)) + [71] * (16 - real_requests),
        [3] * 16,
        request_ids,
        padded=16,
        mask=[index % 2 == 0 for index in range(real_requests)]
        + [not masked] * (16 - real_requests)
        if masked
        else None,
    )
    actual, expected = candidate.compute_hashes(**kwargs), reference.compute_hashes(**kwargs)
    _assert_hashes(actual, expected)
    torch.testing.assert_close(candidate._history, reference._history, atol=0, rtol=0)
    assert not candidate._pending_history_seeds
    assert candidate._history_row_of == reference._history_row_of
    assert candidate._decode_row_map is not None


@pytest.mark.parametrize("real_requests", [0, 9, 11, 12])
def test_engram_decode_alias_hashes_read_resolved_scatter(providers, monkeypatch, real_requests):
    provider, _ = providers(pad=-7)
    monkeypatch.setattr(provider, "_lookback_from_history", Mock(side_effect=AssertionError))
    dummy_id = (1 << 64) - 1
    request_ids = list(range(real_requests)) + [dummy_id] * (16 - real_requests)
    unique_ids = list(dict.fromkeys(request_ids))
    provider.seed_context_history(
        unique_ids,
        {request_id: (3, [31, 41, 51]) for request_id in unique_ids},
        max_seq_len=64,
        device=torch.device("cuda"),
    )
    # Some aliases write equal positions with different tokens/masks; others
    # publish a preceding cell read by another alias in this same forward.
    # Equal-index Torch writes have no promised winner, so compare against an
    # independent integer oracle over the exact history that scatter produced.
    aliases = 16 - real_requests
    positions = [3 + index % 2 for index in range(real_requests)] + [3, 4, 4, 5, 3, 5, -1, 6] * 2
    positions = positions[:16]
    tokens = list(range(70, 70 + real_requests)) + [11 + 17 * index for index in range(aliases)]
    masks = [index % 3 != 1 for index in range(16)]
    launch = Mock(wraps=engram_module._decode_hash_kernel.run)
    monkeypatch.setattr(engram_module._decode_hash_kernel, "run", launch)
    actual = provider.compute_hashes(
        **_arguments(tokens, positions, request_ids, padded=32, mask=masks)
    )
    expected = _hashes_from_shared_history(provider, request_ids, positions, 32)
    for layer, hashes in actual.items():
        torch.testing.assert_close(hashes.cpu(), expected[layer], atol=0, rtol=0)
    assert len(launch.call_args_list) == len(provider.config.layer_ids)
    assert all(
        call.kwargs["READ_CURRENT_HISTORY"] and not call.kwargs["WRITE_HISTORY"]
        for call in launch.call_args_list
    )


def test_engram_decode_aliases_refresh_all_dummy_capture(providers, monkeypatch):
    candidate, reference = providers()
    monkeypatch.setattr(candidate, "_lookback_from_history", Mock(side_effect=AssertionError))
    dummy_id = (1 << 64) - 1
    warmup = _arguments([71] * 16, [0] * 16, [dummy_id] * 16, padded=16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cached = candidate.compute_hashes(**warmup)
        reference.compute_hashes(**warmup)
    stream.synchronize()
    pointers = {layer: hashes.data_ptr() for layer, hashes in cached.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = {
            layer: hashes.clone() for layer, hashes in candidate.compute_hashes(**warmup).items()
        }
    observed = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        for step, count in enumerate((9, 11, 12, 0, 9)):
            real_ids = list(range(100 * (step + 1), 100 * (step + 1) + count))
            real_ids.reverse()
            for provider in (candidate, reference):
                for request_id in real_ids:
                    provider.queue_history_seed(
                        request_id, step, [31, 41, 51], [True, step % 2 == 0, True]
                    )
            kwargs = _arguments(
                [81 + step + index for index in range(count)] + [71] * (16 - count),
                [3 + step] * count + [0] * (16 - count),
                real_ids + [dummy_id] * (16 - count),
                padded=16,
            )
            actual = candidate.refresh_captured_hashes(**kwargs)
            expected = reference.compute_hashes(**kwargs)
            assert {layer: value.data_ptr() for layer, value in actual.items()} == pointers
            graph.replay()
            observed.append(
                (
                    {layer: value.clone() for layer, value in captured.items()},
                    {layer: value.clone() for layer, value in expected.items()},
                )
            )
            assert not candidate._pending_history_seeds
    stream.synchronize()
    for actual, expected in observed:
        _assert_hashes(actual, expected)
    torch.testing.assert_close(candidate._history, reference._history, atol=0, rtol=0)
