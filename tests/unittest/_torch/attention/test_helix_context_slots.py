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
        enable_context_mla_with_cached_kv=True,
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


@pytest.mark.parametrize(
    "phase", ["decode", "no_cache", "no_manager", "legacy_manager", "no_cp", "disabled"]
)
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
    elif phase == "disabled":
        metadata.enable_context_mla_with_cached_kv = False
    else:
        metadata.kv_cache_manager.mapping = Mapping()
    TrtllmAttentionMetadata._prepare_helix_context_slots(metadata, cached)
    assert metadata.helix_context_local_slots is None


@pytest.mark.parametrize("phase", ["decode", "no_cp", "disabled"])
def test_helix_context_slots_metadata_reuse(phase: str) -> None:
    """The host planner clears and recomputes slots on the same metadata object."""
    cp_size, rank, page = 2, 1, 16
    metadata = _metadata(cp_size, rank, page)
    mapping = metadata.kv_cache_manager.mapping
    positions = metadata.helix_position_offsets.clone()
    inactive = metadata.helix_is_inactive_rank.clone()

    def check_slots(cached: torch.Tensor) -> None:
        TrtllmAttentionMetadata._prepare_helix_context_slots(metadata, cached)
        expected = []
        for prefix, length in zip(
            cached[: metadata.num_contexts].tolist(),
            metadata.seq_lens[: metadata.num_contexts].tolist(),
        ):
            owned = [g for g in range(prefix + length) if (g // page) % cp_size == rank]
            local = {position: slot for slot, position in enumerate(owned)}
            expected.extend(local.get(g, -1) for g in range(prefix, prefix + length))
        torch.testing.assert_close(
            metadata.helix_context_local_slots, torch.tensor(expected, dtype=torch.int32)
        )
        torch.testing.assert_close(metadata.helix_position_offsets, positions)
        torch.testing.assert_close(metadata.helix_is_inactive_rank, inactive)

    cached = torch.tensor([0] * metadata.num_contexts + [5, 7], dtype=torch.int32)
    check_slots(cached)

    if phase == "decode":
        metadata.num_contexts = 0
        metadata.num_ctx_tokens = 0
        metadata.seq_lens = torch.tensor([1, 2], dtype=torch.int32)
        cached = cached[-2:]
    elif phase == "no_cp":
        metadata.kv_cache_manager.mapping = Mapping()
    else:
        metadata.enable_context_mla_with_cached_kv = False
    TrtllmAttentionMetadata._prepare_helix_context_slots(metadata, cached)
    assert metadata.helix_context_local_slots is None
    torch.testing.assert_close(metadata.helix_position_offsets, positions)
    torch.testing.assert_close(metadata.helix_is_inactive_rank, inactive)

    lengths = [3, 0, 19]
    metadata.num_contexts = len(lengths)
    metadata.num_ctx_tokens = sum(lengths)
    metadata.seq_lens = torch.tensor(lengths + [1, 2], dtype=torch.int32)
    metadata.kv_cache_manager.mapping = mapping
    metadata.enable_context_mla_with_cached_kv = True
    check_slots(torch.tensor([15, 16, 33, 5, 7], dtype=torch.int32))


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_helix_mla_chunk_plans(cp_size: int) -> None:
    page = 64
    lengths = torch.tensor([[192, 1, 0, 64], [64, 0, 65, 0]], dtype=torch.int64)
    offsets = torch.cat((torch.zeros((1, 4), dtype=torch.int64), lengths.cumsum(0))) + torch.tensor(
        [0, 63, 64, 129]
    )
    metadata = SimpleNamespace(
        helix_context_local_slots=object(),
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=page,
            mapping=SimpleNamespace(cp_size=cp_size, cp_rank=cp_size - 1),
        ),
        host_chunked_global_offset=offsets,
        host_chunked_seq_len=lengths,
        num_contexts=4,
        chunked_loop_num=2,
        seq_lens_cuda=torch.empty(0),
    )
    TrtllmAttentionMetadata._prepare_helix_mla_chunks(metadata)
    assert len(metadata.helix_mla_chunk_plans) == 2
    for loop, plan in enumerate(metadata.helix_mla_chunk_plans):
        expected_cu, expected_starts = [], []
        for rank in range(cp_size):
            cu, starts = [0], []
            for start, length in zip(offsets[loop].tolist(), lengths[loop].tolist()):
                starts.append(sum(g // page % cp_size == rank for g in range(start)))
                cu.append(
                    cu[-1] + sum(g // page % cp_size == rank for g in range(start, start + length))
                )
            expected_cu.append(cu)
            expected_starts.append(starts)
        cu = torch.tensor(expected_cu, dtype=torch.int64)
        sizes = cu[:, -1]
        torch.testing.assert_close(plan.cu_seq_lens, cu)
        torch.testing.assert_close(
            plan.local_offsets, torch.tensor(expected_starts, dtype=torch.int64)
        )
        torch.testing.assert_close(
            plan.rank_offsets, torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
        )
        assert plan.rank_sizes == sizes.tolist()
        assert plan.max_local_seq_len == (cu[-1, 1:] - cu[-1, :-1]).max().item()
    metadata.helix_context_local_slots = None
    TrtllmAttentionMetadata._prepare_helix_mla_chunks(metadata)
    assert metadata.helix_mla_chunk_plans is None
