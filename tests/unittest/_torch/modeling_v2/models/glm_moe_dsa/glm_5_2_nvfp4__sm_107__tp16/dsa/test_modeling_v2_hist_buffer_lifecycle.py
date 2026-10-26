# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Histogram scratch storage across profiling/final KV managers and CUDA graph replay.

Only the DecodeTarget state consumed by the metadata path is constructed: no model weights or
MNNVL collective workspace. The shard metadata and schedule execute their real CUDA kernels.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.modeling import (  # noqa: E501
    DecodeTarget,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa._hist_topk import (  # noqa: E501
    MAX_SHARD,
    shard_capacity,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa.hist_shard_meta import (  # noqa: E501
    hist_shard_meta,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa.paged_mqa_logits_schedule import (  # noqa: E501
    paged_mqa_logits_schedule,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 histogram metadata kernels require sm_107"
)

_TP = 16
_FINAL_MAX_SEQ_LEN = 1_048_596


def _target() -> DecodeTarget:
    target = DecodeTarget.__new__(DecodeTarget)
    target.mapping = SimpleNamespace(tp_size=_TP, tp_rank=_TP - 1)
    target._unit_scale = torch.ones(1, device="cuda")
    target._hist_workspace = object()
    target._hist_buffers = None
    return target


def _metadata(max_seq_len: int, kv_len: int, tokens_per_block: int = 64) -> SimpleNamespace:
    pages = -(-max_seq_len // tokens_per_block)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            max_seq_len=max_seq_len, tokens_per_block=tokens_per_block
        ),
        num_seqs=1,
        gen_indexer_kv_lens_cuda_runtime=torch.tensor([kv_len], dtype=torch.int32, device="cuda"),
        indexer_k_cache_block_offsets=(
            torch.arange(pages, dtype=torch.int32, device="cuda") + 1000
        ).view(1, -1),
    )
    # The stub carries the DSA extension state itself.
    metadata.ext = lambda cls: metadata
    return metadata


def _assert_metadata(target: DecodeTarget, metadata: SimpleNamespace) -> None:
    buffers = target._hist_buffers
    manager = metadata.kv_cache_manager
    required = shard_capacity(manager.max_seq_len, _TP)
    assert buffers["capacity"] >= required
    assert buffers["logits"].shape[1] >= required
    assert buffers["shard_block_table"].numel() * manager.tokens_per_block >= required
    expected_ctx, expected_table = hist_shard_meta.reference(
        metadata.gen_indexer_kv_lens_cuda_runtime,
        metadata.indexer_k_cache_block_offsets[0],
        buffers["shard_ctx"],
        buffers["shard_block_table"],
        _TP,
        target.mapping.tp_rank,
        manager.tokens_per_block,
    )
    expected_schedule = paged_mqa_logits_schedule.reference(
        expected_ctx, manager.tokens_per_block, buffers["sms"], buffers["schedule"]
    )
    assert torch.equal(buffers["shard_ctx"], expected_ctx)
    assert torch.equal(buffers["shard_block_table"], expected_table)
    assert torch.equal(buffers["schedule"], expected_schedule)


def test_histogram_buffers_grow_after_profiling_manager_teardown() -> None:
    """One model serves a short profiling cache and then the final long-context cache."""
    target = _target()
    profiling = _metadata(4096, 4096)
    assert target._shard_indexer_keys(profiling)
    _assert_metadata(target, profiling)
    old_buffers = target._hist_buffers
    del profiling

    final = _metadata(_FINAL_MAX_SEQ_LEN, _FINAL_MAX_SEQ_LEN)
    assert target._shard_indexer_keys(final)
    assert target._hist_buffers is not old_buffers
    _assert_metadata(target, final)


@pytest.mark.parametrize("tokens_per_block", [32, 128])
def test_histogram_buffers_follow_replacement_manager_page_size(tokens_per_block: int) -> None:
    """Equal token capacity with a different page layout needs a compatible shard table."""
    target = _target()
    metadata = _metadata(65_536, 60_003)
    assert target._shard_indexer_keys(metadata)
    old_buffers = target._hist_buffers

    replacement = _metadata(65_536, 60_003, tokens_per_block)
    assert target._shard_indexer_keys(replacement)
    assert target._hist_buffers is not old_buffers
    _assert_metadata(target, replacement)


def test_histogram_buffers_reject_unsupported_replacement_capacity() -> None:
    """A cached small buffer cannot make an oversized manager eligible for sharding."""
    target = _target()
    metadata = _metadata(4096, 2048)
    assert target._shard_indexer_keys(metadata)
    buffers = target._hist_buffers
    oversized = _metadata(MAX_SHARD * _TP + 1, 2048)
    assert not target._shard_indexer_keys(oversized)
    assert target._hist_buffers is buffers
    assert target._shard_indexer_keys(metadata)
    _assert_metadata(target, metadata)


def test_histogram_buffers_remain_stable_for_final_cuda_graph() -> None:
    """Compatible managers reuse storage, and replay consumes updated device metadata."""
    target = _target()
    assert target._shard_indexer_keys(_metadata(4096, 4096))
    final = _metadata(_FINAL_MAX_SEQ_LEN, 530_000)
    assert target._shard_indexer_keys(final)
    buffers = target._hist_buffers
    pointers = {key: value.data_ptr() for key, value in buffers.items() if torch.is_tensor(value)}

    for max_seq_len in (_FINAL_MAX_SEQ_LEN, 65_536):
        replacement = _metadata(max_seq_len, 32_003)
        assert target._shard_indexer_keys(replacement)
        assert target._hist_buffers is buffers
        _assert_metadata(target, replacement)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert target._shard_indexer_keys(final)
    for kv_len in (2048, 530_000, _FINAL_MAX_SEQ_LEN):
        final.gen_indexer_kv_lens_cuda_runtime.fill_(kv_len)
        final.indexer_k_cache_block_offsets.add_(17)
        buffers["shard_ctx"].fill_(-7)
        buffers["shard_block_table"].fill_(-7)
        buffers["schedule"].fill_(-7)
        graph.replay()
        assert target._hist_buffers is buffers
        assert all(buffers[key].data_ptr() == pointer for key, pointer in pointers.items())
        _assert_metadata(target, final)
