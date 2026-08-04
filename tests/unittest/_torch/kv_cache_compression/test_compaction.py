# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical KV-cache compaction: packed moves, protected tails, SWA windows,
and draft co-compaction, checked byte-exactly against torch oracles."""

from types import SimpleNamespace

import pytest
import torch
from conftest import build_compaction as _build_compaction
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import make_ramp_pools as _make_ramp_pools
from conftest import run_compaction as _run_compaction
from conftest import set_protected_tails as _set_protected_tails

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="KV-cache compaction kernels require SM100",
)


def _logical_view(pool: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Gather one request's pages into [K/V, head, token, dim] order."""
    num_kv_heads = int(pool.shape[2])
    head_dim = int(pool.shape[4])
    return pool.index_select(0, pages).permute(1, 2, 0, 3, 4).reshape(2, num_kv_heads, -1, head_dim)


@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
def test_eager_compaction_preserves_exact_selected_bytes_and_tail(eviction_mode):
    # Supported bf16 geometry; kept ordinals span all three pages so moves
    # cross page boundaries.
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    num_layers = 2
    num_kv_heads = 2
    # Mixed prompt lengths prove per-request destination rebasing.
    prompt_lens = [2, 5]
    decode_keep_count = 4
    seq_len = 80
    tokens_per_block = 32
    pages_per_request = 3
    head_dim = 64
    protected_tails = [2, 1]
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    initial_pools = _make_ramp_pools(num_layers, device=device)
    pools = [pool.clone() for pool in initial_pools]

    # Decode-only kept ordinals holding absolute positions.
    union_decode = torch.tensor(
        [[16, 32, 56, 72], [24, 40, 48, 64]], dtype=torch.int64, device=device
    )
    if eviction_mode == "union":
        keep = union_decode
        selection_rows = 1
    else:
        selection_rows = num_kv_heads if eviction_mode == "per_head" else num_layers * num_kv_heads
        keep = torch.empty(
            request_count,
            selection_rows,
            decode_keep_count,
            dtype=torch.int64,
            device=device,
        )
        for request in range(request_count):
            for row in range(selection_rows):
                keep[request, row] = torch.tensor(
                    sorted(
                        {
                            prompt_lens[request] + ((request + row + offset * 2) % 8) * 8
                            for offset in range(decode_keep_count)
                        }
                    ),
                    dtype=torch.int64,
                    device=device,
                )

    compaction = _build_compaction(
        eviction_mode=eviction_mode,
        layer_pools=pools,
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(page_tables.unsqueeze(0)),
        prompt_offsets=torch.tensor(prompt_lens, dtype=torch.int32, device=device),
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    # Production settles the kept ordinals into the contract's decision
    # rows; with pre-settled ordinals the pack launch inside compact() is
    # its exact analog.
    _run_compaction(compaction)
    torch.cuda.synchronize(device)

    for layer, (before_pool, after_pool) in enumerate(zip(initial_pools, pools)):
        for request in range(request_count):
            prompt_len = prompt_lens[request]
            pages = page_tables[request].to(torch.long)
            before = (
                before_pool[pages]
                .permute(1, 2, 0, 3, 4)
                .reshape(2, num_kv_heads, pages_per_request * tokens_per_block, head_dim)
            )
            after = after_pool[pages].permute(1, 2, 0, 3, 4).reshape_as(before)
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            for head in range(num_kv_heads):
                if eviction_mode == "union":
                    selected = keep[request]
                elif eviction_mode == "per_head":
                    selected = keep[request, head]
                else:
                    selected = keep[request, layer * num_kv_heads + head]
                tail = torch.arange(
                    seq_len,
                    seq_len + protected_tails[request],
                    dtype=torch.int64,
                    device=device,
                )
                source = torch.cat((selected, tail))
                destination = torch.arange(
                    prompt_len,
                    prompt_len + source.numel(),
                    dtype=torch.int64,
                    device=device,
                )
                assert torch.equal(
                    after[:, head].index_select(1, destination),
                    before[:, head].index_select(1, source),
                )


@requires_sm100
def test_eager_compaction_rebases_masked_swa_window_and_tail():
    # Supported bf16 geometry; dense and SWA moves stay page-crossing.
    device = torch.device("cuda", torch.cuda.current_device())
    dense_tables = torch.tensor([[2, 0, 1], [5, 3, 4]], dtype=torch.int32, device=device)
    swa_tables = torch.tensor([[1, 2, 0], [4, 5, 3]], dtype=torch.int32, device=device)
    initial_pools = _make_ramp_pools(2, num_kv_heads=1, device=device)
    pools = [pool.clone() for pool in initial_pools]
    # Decode-only kept ordinals holding absolute positions past the prompt.
    keep = torch.tensor(
        [[16, 32, 40, 56], [16, 24, 40, 48]],
        dtype=torch.int64,
        device=device,
    )
    valid_seq_lens = torch.tensor([64, 56], dtype=torch.int32, device=device)
    protected_tails = [2, 1]
    compaction = _build_compaction(
        layer_pools=pools,
        dense_layers=[0],
        swa_layers=[1],
        layer_group_representative={0: 0},
        # Dense layer 0 stages in plane 0, the SWA layer in its own plane 1.
        layer_pool_ids=[0, 1],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=valid_seq_lens,
        kv_block_offsets=_encode_block_offsets(torch.stack((dense_tables, swa_tables))),
        prompt_offsets=torch.tensor([2, 2], dtype=torch.int32, device=device),
        swa_window=2,
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    _run_compaction(compaction)
    torch.cuda.synchronize(device)

    for request, (valid_seq_len, tail_length) in enumerate(
        zip(valid_seq_lens.tolist(), protected_tails)
    ):
        dense_pages = dense_tables[request].to(torch.long)
        swa_pages = swa_tables[request].to(torch.long)
        dense_before = initial_pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 64)
        dense_after = pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape_as(dense_before)
        swa_before = initial_pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 64)
        swa_after = pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape_as(swa_before)
        tail = torch.arange(
            valid_seq_len,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        dense_source = torch.cat((keep[request], tail))
        dense_destination = torch.arange(
            2, 2 + dense_source.numel(), dtype=torch.int64, device=device
        )
        swa_source = torch.arange(
            valid_seq_len - 2,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        swa_destination = torch.arange(4, 4 + swa_source.numel(), dtype=torch.int64, device=device)
        assert torch.equal(dense_after[:, :, :2], dense_before[:, :, :2])
        assert torch.equal(swa_after[:, :, :2], swa_before[:, :, :2])
        assert torch.equal(
            dense_after.index_select(2, dense_destination),
            dense_before.index_select(2, dense_source),
        )
        assert torch.equal(
            swa_after.index_select(2, swa_destination),
            swa_before.index_select(2, swa_source),
        )


def _launched_draft_compaction(draft_protected_tails):
    """Target and draft pools with distinct head counts (supported bf16
    geometry, mod-251 ramp payload), compacted in one round."""
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    prompt_len = 2
    target_protected_tails = [2, 1]
    valid_seq_lens = [10, 9]

    target_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    draft_tables = torch.tensor([[1, 0, 2], [5, 4, 3]], dtype=torch.int32, device=device)
    target_pools = _make_ramp_pools(2, num_kv_heads=2, device=device)
    draft_pool = _make_ramp_pools(1, num_kv_heads=4, base=149, device=device)[0]
    assert target_pools[0].shape[2] != draft_pool.shape[2]
    initial_target = [pool.clone() for pool in target_pools]
    initial_draft = draft_pool.clone()

    keep = torch.tensor([[2, 4, 7, 9], [3, 5, 6, 8]], dtype=torch.int64, device=device)

    compaction = _build_compaction(
        layer_pools=target_pools,
        layer_pool_ids=[0, 0],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor(valid_seq_lens, dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(target_tables),
        prompt_offsets=torch.full((request_count,), prompt_len, dtype=torch.int32, device=device),
        protected_tail_capacity=max(target_protected_tails),
        draft_layer_pools=[draft_pool],
        draft_layers=[0],
        draft_layer_group_representative={0: 0},
        draft_layer_pool_ids=[0],
        draft_protected_tail_capacity=max(draft_protected_tails),
        draft_kv_block_offsets=_encode_block_offsets(draft_tables),
    )
    _set_protected_tails(compaction, target_protected_tails, draft_protected_tails)
    _run_compaction(compaction)
    torch.cuda.synchronize(device)

    return SimpleNamespace(
        device=device,
        request_count=request_count,
        prompt_len=prompt_len,
        keep=keep,
        valid_seq_lens=valid_seq_lens,
        target_protected_tails=target_protected_tails,
        draft_protected_tails=draft_protected_tails,
        target_tables=target_tables,
        draft_tables=draft_tables,
        target_pools=target_pools,
        draft_pool=draft_pool,
        initial_target=initial_target,
        initial_draft=initial_draft,
        compaction=compaction,
    )


def test_draft_moves_and_pack_match_keep_broadcast_and_tail_oracle():
    # Ragged draft tails [1, 2] against target tails [2, 1]: one request's
    # draft tail below and one above its target, subsuming the uniform row.
    built = _launched_draft_compaction(draft_protected_tails=[1, 2])
    device = built.device
    prompt_len = built.prompt_len

    expected_offsets = [0]
    for request in range(built.request_count):
        valid = built.valid_seq_lens[request]
        # Target dense layers compact the union keep set plus the target tail.
        target_pages = built.target_tables[request].to(torch.long)
        target_tail = torch.arange(
            valid,
            valid + built.target_protected_tails[request],
            dtype=torch.int64,
            device=device,
        )
        target_source = torch.cat((built.keep[request], target_tail))
        target_destination = torch.arange(
            prompt_len,
            prompt_len + target_source.numel(),
            dtype=torch.int64,
            device=device,
        )
        for before_pool, after_pool in zip(built.initial_target, built.target_pools):
            before = _logical_view(before_pool, target_pages)
            after = _logical_view(after_pool, target_pages)
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            assert torch.equal(
                after.index_select(2, target_destination),
                before.index_select(2, target_source),
            )

        # Same kept ordinals through the draft's OWN table/heads/tail.
        draft_pages = built.draft_tables[request].to(torch.long)
        draft_tail = torch.arange(
            valid,
            valid + built.draft_protected_tails[request],
            dtype=torch.int64,
            device=device,
        )
        draft_source = torch.cat((built.keep[request], draft_tail))
        draft_destination = torch.arange(
            prompt_len,
            prompt_len + draft_source.numel(),
            dtype=torch.int64,
            device=device,
        )
        before = _logical_view(built.initial_draft, draft_pages)
        after = _logical_view(built.draft_pool, draft_pages)
        assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
        for head in range(int(built.draft_pool.shape[2])):
            assert torch.equal(
                after[:, head].index_select(1, draft_destination),
                before[:, head].index_select(1, draft_source),
            )

        expected_offsets.append(expected_offsets[-1] + int(draft_source.numel()))

    # The test-owned draft move-offset row must match the broadcast-plus-tail
    # oracle; the packed move sources themselves are covered byte-exactly by
    # the pool assertions above (the ramp payload makes every wrong move land
    # on different bytes) and by the pack-kernel oracle suite.
    assert built.compaction["draft_move_offsets"].cpu().tolist() == expected_offsets
