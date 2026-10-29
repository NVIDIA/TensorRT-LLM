# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host planning preserves Helix page ownership and generation metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only


def _metadata(cp_size: int, rank: int, page: int) -> SimpleNamespace:
    lengths = [1, 65, 2, 63, 0, 3, 67, 65]
    return SimpleNamespace(
        num_contexts=len(lengths),
        num_ctx_tokens=sum(lengths),
        seq_lens=torch.tensor(lengths + [1, 2], dtype=torch.int32),
        seq_lens_cuda=torch.empty(0),  # Exercise host planning and a CPU destination.
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=page,
            mapping=Mapping(
                world_size=cp_size,
                rank=rank,
                cp_size=cp_size,
                cp_config={"cp_type": CpType.HELIX},
            ),
        ),
        helix_context_local_slots=None,
        helix_position_offsets=torch.tensor([1234567, 7654321]),
        helix_is_inactive_rank=torch.tensor([True, False]),
    )


@pytest.mark.parametrize("cp_size", [2, 4, 8, 32])
@pytest.mark.parametrize("page", [16, 64, 128])
def test_helix_context_slots(cp_size: int, page: int) -> None:
    prefixes = [0, page - 1, page, page + 1, 2 * page - 1, 2 * page, 2 * page + 1, 2047]
    ownership = []
    for rank in range(cp_size):
        metadata = _metadata(cp_size, rank, page)
        positions = metadata.helix_position_offsets.clone()
        inactive = metadata.helix_is_inactive_rank.clone()
        TrtllmAttentionMetadata._prepare_helix_context_slots(
            metadata, torch.tensor(prefixes + [5, 7], dtype=torch.int32)
        )
        expected = []
        for prefix, length in zip(prefixes, metadata.seq_lens[: metadata.num_contexts].tolist()):
            # Enumerate the complete owned history, independently of the
            # vectorized slot formula, then select only the current chunk.
            owned = [g for g in range(prefix + length) if (g // page) % cp_size == rank]
            local = {position: slot for slot, position in enumerate(owned)}
            expected.extend(local.get(g, -1) for g in range(prefix, prefix + length))
        torch.testing.assert_close(
            metadata.helix_context_local_slots, torch.tensor(expected, dtype=torch.int32)
        )
        torch.testing.assert_close(metadata.helix_position_offsets, positions)
        torch.testing.assert_close(metadata.helix_is_inactive_rank, inactive)
        ownership.append(metadata.helix_context_local_slots >= 0)
    # Each token is persisted exactly once across the CP group.
    torch.testing.assert_close(
        torch.stack(ownership).sum(dim=0), torch.ones(len(expected), dtype=torch.int64)
    )


@pytest.mark.parametrize("phase", ["decode", "no_cache", "no_manager", "legacy_manager", "no_cp"])
def test_helix_context_slots_reset(phase: str) -> None:
    metadata = _metadata(2, 0, 64)
    metadata.helix_context_local_slots = torch.tensor([123], dtype=torch.int32)
    cached = torch.zeros(metadata.num_contexts + 2, dtype=torch.int32)
    if phase == "decode":
        metadata.num_contexts = 0
    elif phase == "no_cache":
        cached = None
    elif phase == "no_manager":
        metadata.kv_cache_manager = None
    elif phase == "legacy_manager":
        metadata.kv_cache_manager = SimpleNamespace(tokens_per_block=64)
    else:
        metadata.kv_cache_manager.mapping = Mapping()
    TrtllmAttentionMetadata._prepare_helix_context_slots(metadata, cached)
    assert metadata.helix_context_local_slots is None
