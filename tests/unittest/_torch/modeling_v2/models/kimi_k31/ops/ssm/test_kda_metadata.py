# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Kimi K3.1 KDA metadata (``ops/ssm/kda_metadata.KdaMetadata``).

The packed-prefill schedule, the stable storage CUDA graphs replay, the decode exclusion and the 16-byte aligned
generation state indices, plus an equivalence check against the engine's Kimi K3 Mamba metadata on random batches.
"""

import random
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_metadata import (
    KdaMetadata,
)

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


class _ReplayCacheManager:
    """The two members the metadata reads from the KDA cache manager: the replay switch and the state slots.

    One fixed slot buffer for the manager's lifetime, as the engine's manager keeps it: the metadata requires a stable
    slot pointer across prepares (CUDA graphs replay it). ``set_slots`` writes a batch's slots into it.
    """

    use_kda_replay_update = True

    def __init__(self, capacity: int) -> None:
        self._state_indices_buffer = DeviceCopyBuffer(
            torch.zeros(capacity, dtype=torch.int32, device="cuda")
        )

    def set_slots(self, slots: list[int]) -> None:
        self._state_indices_buffer.tensor[: len(slots)].copy_(
            torch.tensor(slots, dtype=torch.int32, device="cuda")
        )

    def get_state_indices(self, request_ids: list[int], is_padding: list[bool]) -> torch.Tensor:
        return self._state_indices_buffer.tensor[: len(request_ids)]


def _make_mixed_batch(context_lengths: list[int], num_generation: int, manager) -> SimpleNamespace:
    seq_lens = torch.tensor(context_lengths + [1] * num_generation, dtype=torch.int)
    num_contexts = len(context_lengths)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=sum(context_lengths),
        kv_cache_manager=manager,
        request_ids=list(range(100, 100 + num_contexts + num_generation)),
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.zeros(num_contexts, dtype=torch.int),
        ),
    )


@pytest.mark.parametrize("lengths", [[128, 256], [1, 64, 65, 129], [17, 1, 17, 16], [300], []])
def test_prepare_packed_kda_metadata(lengths: list[int]) -> None:
    metadata = KdaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
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


def test_prepare_refreshes_offsets_without_changing_storage() -> None:
    metadata = KdaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
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
    metadata = KdaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
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


def test_prepare_materializes_stable_aligned_generation_indices() -> None:
    manager = _ReplayCacheManager(3)
    manager.set_slots([9, 4, 7])
    metadata = KdaMetadata(max_batch_size=3, chunk_size=8, max_num_tokens=8)
    attn_metadata = _make_mixed_batch([2], 2, manager)

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


def _assert_same_tensor(got: torch.Tensor, value: torch.Tensor, name: str) -> None:
    assert isinstance(got, torch.Tensor), name
    assert (got.shape, got.dtype, got.device, got.stride()) == (
        value.shape,
        value.dtype,
        value.device,
        value.stride(),
    ), name
    # The aligned generation slots: the same 16-byte alignment as the engine's copy.
    assert got.data_ptr() % 16 == value.data_ptr() % 16, name
    torch.testing.assert_close(got, value, rtol=0, atol=0, msg=name)


def _assert_same_state(family: KdaMetadata, engine) -> None:
    assert type(family).prefill_chunk_alignment == type(engine).prefill_chunk_alignment == 16
    assert list(vars(family)) == list(vars(engine))
    for name, value in vars(engine).items():
        got = getattr(family, name)
        if isinstance(value, torch.Tensor):
            _assert_same_tensor(got, value, name)
        elif value is None or isinstance(value, (bool, int, float, str)):
            assert got == value, name
        else:
            # Buffer holders (the state-slot DeviceCopyBuffer): the same type holding the same tensor.
            assert type(got) is type(value), name
            if isinstance(getattr(value, "tensor", None), torch.Tensor):
                _assert_same_tensor(got.tensor, value.tensor, name)


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_engine_kimi_k3_mamba_metadata(seed: int) -> None:
    """The family class prepares the same fields, buffers and alignment as the engine class."""
    from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

    rng = random.Random(seed)
    max_batch_size = rng.choice([3, 4, 8, 16])
    family = KdaMetadata(max_batch_size=max_batch_size, chunk_size=128, max_num_tokens=4096)
    engine = KimiK3MambaMetadata(max_batch_size=max_batch_size, chunk_size=128, max_num_tokens=4096)
    # Both classes allocate the aligned generation-slot backing uninitialized and write only its prefix; a sentinel
    # makes the untouched tail compare equal, so the whole backing buffer stays part of the comparison.
    for metadata in (family, engine):
        metadata._generation_state_indices.fill_(-1)
    _assert_same_state(family, engine)
    # One manager (or none) for all of a seed's batches, as an engine keeps one: the slot pointer stays stable.
    manager = _ReplayCacheManager(max_batch_size) if rng.random() < 0.5 else None
    for _ in range(4):
        num_contexts = rng.randint(0, max_batch_size)
        num_generation = rng.randint(0 if num_contexts else 1, max_batch_size - num_contexts)
        lengths = [rng.randint(1, 4096 // max_batch_size) for _ in range(num_contexts)]
        slots = rng.sample(range(4 * max_batch_size), num_contexts + num_generation)
        if manager is not None:
            manager.set_slots(slots)
        for metadata in (family, engine):
            metadata.prepare(_make_mixed_batch(lengths, num_generation, manager))
        torch.cuda.synchronize()
        _assert_same_state(family, engine)
