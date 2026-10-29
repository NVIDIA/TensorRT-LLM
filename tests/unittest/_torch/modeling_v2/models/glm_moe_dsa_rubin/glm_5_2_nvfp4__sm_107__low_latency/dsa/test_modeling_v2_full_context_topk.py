# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal full-context TopK, padded row pitches, and persistent graph workspace."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency._full_context import (  # noqa: E501
    FullContextIndexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa._full_context_topk import (  # noqa: E501
    full_context_topk,
    streaming_plan,
    workspace_bytes,
)
from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_topk_decode_self_sampling_host import (
    route_streaming,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(get_sm_version() != 107, reason="Rubin GLM full-context indexer")


def test_tp8_full_context_eligibility_uses_query_window_and_live_metadata() -> None:
    """TP8 admits B1/Q8 but rejects B2/Q8 and unsupported query windows."""
    capacity = 8192
    decode = SimpleNamespace(mapping=SimpleNamespace(tp_size=8))
    indexer = FullContextIndexer(decode, capacity, torch.device("cuda"))
    manager = SimpleNamespace(tokens_per_block=64, use_fp4=False, max_seq_len=capacity)
    for batch, next_n, supported in (
        (1, 1, True),
        (2, 1, True),
        (1, 4, True),
        (2, 4, True),
        (2, 8, False),
        (1, 2, False),
        (1, 6, False),
        (1, 9, False),
        (1, 8, True),
    ):
        rows = batch * next_n
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            num_seqs=batch,
            num_tokens=rows,
            is_ragged_verify=False,
            use_expanded_buffers_for_mtp=False,
            expand_for_dsl=False,
            gen_indexer_kv_lens_cuda_runtime=torch.full(
                (batch,), 5000, dtype=torch.int32, device="cuda"
            ),
            indexer_k_cache_block_offsets=torch.zeros(
                (batch, capacity // 64), dtype=torch.int32, device="cuda"
            ),
            shared_topk_indices=torch.empty((rows, 2048), dtype=torch.int32, device="cuda"),
            scheduler_metadata_buffer=torch.empty(1, dtype=torch.int32, device="cuda"),
            get_indexer_max_seq_len=lambda: manager.max_seq_len,
        )
        assert indexer.supports(metadata, rows) is supported, (batch, next_n)

    # An eligible window must still have complete live metadata and fit the allocated arena.
    metadata.num_tokens = 7
    assert not indexer.supports(metadata, 8)
    metadata.num_tokens = 8
    metadata.shared_topk_indices = metadata.shared_topk_indices[:7]
    assert not indexer.supports(metadata, 8)
    metadata.shared_topk_indices = indexer.topk[:8]
    manager.max_seq_len = capacity + 1
    assert not indexer.supports(metadata, 8)


def _assert_selection(logits, lens, got, next_n):
    for row in range(logits.shape[0]):
        length = min(int(lens[row // next_n]) - next_n + row % next_n + 1, logits.shape[1])
        selected = got[row]
        if length <= 2048:
            assert torch.equal(
                selected[:length], torch.arange(length, device=got.device, dtype=torch.int32)
            )
            assert (selected[length:] == -1).all()
            continue
        assert ((selected >= 0) & (selected < length)).all()
        assert selected.unique().numel() == 2048
        expected_values = logits[row, :length].topk(2048).values
        actual_values = logits[row, selected.long()].sort(descending=True).values
        assert torch.equal(actual_values, expected_values)


@pytest.mark.parametrize(
    "rows,next_n,capacity",
    [
        (1, 1, 1048590),
        (3, 1, 65550),
        (6, 6, 65550),
        (8, 8, 1048590),
        (12, 6, 65550),
        (18, 6, 1048590),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_causal_selection_graph_and_workspace_reuse(rows, next_n, capacity, deterministic):
    generator = torch.Generator(device="cuda").manual_seed(901 + rows)
    pitch = (capacity + 255) // 256 * 256
    arena = torch.randn(rows, pitch, device="cuda", generator=generator)
    logits = arena[:, :capacity]
    arena[:, capacity:].fill_(1e20)
    batch = rows // next_n
    lens = torch.tensor(
        [max(next_n, 3000 + 137 * i) for i in range(batch)], device="cuda", dtype=torch.int32
    )
    indices = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(workspace_bytes() // 4, device="cuda", dtype=torch.int32)

    def run():
        return full_context_topk(
            logits,
            lens,
            indices,
            workspace,
            next_n=next_n,
            max_seq_len=capacity,
            sm_reserve=72,
            deterministic=deterministic,
        )

    run()
    torch.cuda.synchronize()
    _assert_selection(logits, lens, indices, next_n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for length in (next_n, 2048, min(capacity, 33000), capacity):
        lens.fill_(length)
        # Many ties: every selected value must still equal the independent sorted top values.
        arena[:, :capacity].copy_(
            torch.randint(0, 4096, (rows, capacity), device="cuda", generator=generator).float()
        )
        graph.replay()
        torch.cuda.synchronize()
        _assert_selection(logits, lens, indices, next_n)
        if deterministic:
            previous = indices.clone()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(previous, indices)


def test_streaming_plan_preserves_shared_abi_and_caps_all_derived_fields():
    for rows in (1, 3, 6, 8, 12, 18, 24):
        for n in (129, 2048, 3000, 65550, 1048590):
            tpl, splits, scap, cmp, _ = streaming_plan(rows, n, 184, 0)
            baseline = route_streaming(
                rows, max(n, 2049), (n + 255) // 256 * 256, 2048, force_main=True
            )
            assert tpl == baseline["tpl"]
            assert splits == baseline["rt"]["R"]
            assert scap == baseline["rt"]["SCAP_"] and cmp == baseline["rt"]["CMP_"]
            for reserve in (36, 72):
                tpl, splits, _, _, _ = streaming_plan(rows, n, 184, reserve)
                assert rows * splits <= 184 - reserve
                assert tpl[5] == (splits > 1)
