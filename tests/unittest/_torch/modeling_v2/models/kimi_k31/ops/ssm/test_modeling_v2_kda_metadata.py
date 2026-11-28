# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Kimi K3.1 KDA metadata (``ops/ssm/kda_metadata.KdaMetadata``).

An equivalence check against the engine's Kimi K3 Mamba metadata on random mixed batches: the same fields, buffers
and 16-byte aligned generation state indices.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_metadata import KdaMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


@pytest.mark.parametrize("seed", range(1))
def test_matches_the_engine_kimi_k3_mamba_metadata(seed: int) -> None:
    """The family class prepares the same fields, buffers and alignment as the engine class."""
    from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

    rng = random.Random(seed)
    max_batch_size = rng.choice([3, 4, 8, 16])
    family = KdaMetadata(max_batch_size=max_batch_size, chunk_size=128, max_num_tokens=4096)
    engine = KimiK3MambaMetadata(max_batch_size=max_batch_size, chunk_size=128, max_num_tokens=4096)
    # Both classes allocate the aligned generation-slot backings uninitialized and write only their prefixes; a
    # sentinel makes the untouched tails compare equal, so the whole backing buffers stay part of the comparison.
    for metadata in (family, engine):
        metadata._generation_state_indices.fill_(-1)
        metadata._generation_replay_state_indices.fill_(-1)
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
