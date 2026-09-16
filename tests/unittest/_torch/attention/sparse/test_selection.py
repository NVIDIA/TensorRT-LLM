# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection/layout contracts with synthetic host KV and GPU entry mappings."""

from dataclasses import replace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.selection import DSASelectionPolicy
from tensorrt_llm._torch.attention.backends.sparse.kv_layout import (
    EntryComponent,
    EntryLayout,
    EntryResolution,
    GpuCacheView,
    HostStorageView,
    resolve_entries,
)
from tensorrt_llm._torch.attention.backends.sparse.selection import (
    SelectedEntries,
    SelectionContext,
    SelectionPolicy,
)


def _layout(layer_id: int = 7, life_cycle_id: int = 3) -> EntryLayout:
    # Four native entries per page. KV and scales share one pool; V uses another.
    return EntryLayout(
        layer_id,
        life_cycle_id,
        2,
        4,
        (64, 32),
        (
            EntryComponent("kv", 0, 0, 8, 4),
            EntryComponent("scale", 0, 32, 1, 1),
            EntryComponent("v", 1, 0, 4, 2),
        ),
    )


def _outputs(shape: tuple[int, int], components: int = 3) -> EntryResolution:
    return EntryResolution(
        torch.empty(shape, dtype=torch.bool, device="cuda"),
        torch.empty(shape, dtype=torch.bool, device="cuda"),
        torch.empty((*shape, components), dtype=torch.int64, device="cuda"),
        torch.empty(shape, dtype=torch.int32, device="cuda"),
    )


def _inputs(
    positions: list[list[int]],
) -> tuple[SelectedEntries, HostStorageView, GpuCacheView, EntryResolution]:
    device = "cuda"
    ids = torch.tensor([100, 200], dtype=torch.int64, device=device)
    context = SelectionContext(
        ids,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([7, 6], dtype=torch.int32, device=device),
        7,
        3,
    )
    selection = DSASelectionPolicy().select(
        torch.tensor(positions, dtype=torch.int32, device=device), context
    )
    host = HostStorageView(
        _layout(),
        ids.clone(),
        torch.tensor([[1, 3], [2, 0]], dtype=torch.int64, device=device),
        torch.tensor([[4, 2], [4, 2]], dtype=torch.int32, device=device),
        (256, 128),
    )
    gpu = GpuCacheView(7, 3, ids.clone(), torch.full((2, 8), -1, dtype=torch.int32, device=device))
    return selection, host, gpu, _outputs(selection.positions.shape)


def test_dsa_borrows_positions_and_keeps_identity_order_and_mask() -> None:
    positions = torch.tensor([[4, 1, 4, -1]], dtype=torch.int32)
    context = SelectionContext(
        torch.tensor([123]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        8,
        2,
    )
    mask = torch.tensor([[True, False, True, True]])
    policy: SelectionPolicy = DSASelectionPolicy()
    result = policy.select(positions, context, mask)
    assert result.positions is positions
    assert result.context is context
    assert result.valid_mask is mask
    # IndexShare shares positions, but never substitutes the KV layer's identity.
    next_layer = policy.select(positions, replace(context, layer_id=9, life_cycle_id=4))
    assert next_layer.positions is positions
    assert (next_layer.context.layer_id, next_layer.context.life_cycle_id) == (9, 4)
    assert positions.tolist() == [[4, 1, 4, -1]]


@pytest.mark.parametrize(
    "changes",
    [
        {"entries_per_page": 0},
        {"layer_id": -1},
        {"pool_group_index": -1},
        {"pool_slot_bytes": (16, 32)},
        {"components": ()},
        {"components": (EntryComponent("bad", 2, 0, 4, 4),)},
        {"components": (EntryComponent("same", 0, 0, 4, 4), EntryComponent("same", 1, 0, 4, 4))},
    ],
)
def test_rejects_invalid_layout(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(_layout(), **changes)


@pytest.mark.parametrize(
    "values", [("", 0, 0, 4, 4), ("kv", -1, 0, 4, 4), ("kv", 0, -1, 4, 4), ("kv", 0, 0, 2, 4)]
)
def test_rejects_invalid_component(values: tuple) -> None:
    with pytest.raises(ValueError):
        EntryComponent(*values)


@pytest.mark.parametrize(
    "invalid", ["positions_dtype", "query_count", "mask_shape", "request_dtype", "length_shape"]
)
def test_rejects_invalid_selection_metadata(invalid: str) -> None:
    ctx = SelectionContext(
        torch.tensor([100]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        7,
        3,
    )
    positions = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(ValueError):
        if invalid == "positions_dtype":
            DSASelectionPolicy().select(positions.long(), ctx)
        elif invalid == "query_count":
            DSASelectionPolicy().select(positions.expand(2, -1), ctx)
        elif invalid == "mask_shape":
            DSASelectionPolicy().select(positions, ctx, torch.ones((1, 3), dtype=torch.bool))
        elif invalid == "request_dtype":
            replace(ctx, request_ids=ctx.request_ids.int())
        else:
            replace(ctx, valid_lengths=torch.ones(2, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_only_selection_partial_page_scales_and_duplicate_order() -> None:
    selection, host, gpu, out = _inputs([[3, 4, 4, 5, 6, -1, 8], [0, 4, 5, 6, 1, -1, 0]])
    # Only a single entry has a ready GPU copy; its page neighbours do not.
    gpu.entry_indices[0, 4] = 17
    pools = (
        torch.arange(256, dtype=torch.uint8, pin_memory=True),
        torch.arange(128, dtype=torch.uint8, pin_memory=True),
    )
    original_pools = tuple(pool.clone() for pool in pools)
    original_positions = selection.positions.clone()
    original_slots = host.page_slots.clone()
    original_gpu = gpu.entry_indices.clone()
    resolve_entries(selection, host, gpu, out)
    assert out.valid.cpu().tolist() == [
        [True, True, True, True, True, False, False],
        [True, True, True, False, True, False, True],
    ]
    assert out.host_valid.cpu().tolist() == [
        [True, True, True, True, False, False, False],
        [True, True, True, False, True, False, True],
    ]
    assert out.gpu_indices.cpu().tolist() == [[-1, 17, 17, -1, -1, -1, -1], [-1] * 7]
    expected = [
        [
            [88, 99, 44],
            [192, 224, 96],
            [192, 224, 96],
            [200, 225, 100],
            [-1] * 3,
            [-1] * 3,
            [-1] * 3,
        ],
        [
            [128, 160, 64],
            [0, 32, 0],
            [8, 33, 4],
            [-1] * 3,
            [136, 161, 68],
            [-1] * 3,
            [128, 160, 64],
        ],
    ]
    assert out.host_offsets.cpu().tolist() == expected
    # Read actual synthetic host bytes by the returned offsets. No GPU page
    # index is needed; scales and KV stay available after any selected hit.
    for offsets in expected[0][:4]:
        for component, offset in zip(host.layout.components, offsets):
            copied = pools[component.pool_index][offset : offset + component.size].clone()
            assert copied.tolist() == list(range(offset, offset + component.size))
    for pool, original in zip(pools, original_pools):
        torch.testing.assert_close(pool, original)
    torch.testing.assert_close(selection.positions, original_positions)
    torch.testing.assert_close(host.page_slots, original_slots)
    torch.testing.assert_close(gpu.entry_indices, original_gpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_entries_coalesced_layer_offsets_and_causal_bounds() -> None:
    selection, host, gpu, out = _inputs([[3, 4, 5], [0, 1, 2]])
    # A 16-token block compressed by 4 has four entries; no second division by
    # four in the resolver. Layer B occupies the second half of each pool slot.
    layout = EntryLayout(
        9,
        5,
        2,
        4,
        (128, 64),
        (
            EntryComponent("kv", 0, 64, 8, 4),
            EntryComponent("scale", 0, 96, 1, 1),
            EntryComponent("v", 1, 32, 4, 2),
        ),
    )
    context = replace(selection.context, layer_id=9, life_cycle_id=5)
    context.request_indices.copy_(torch.tensor([0, 0], device="cuda", dtype=torch.int32))
    context.valid_lengths.copy_(torch.tensor([6, 2], device="cuda", dtype=torch.int32))
    selection = replace(selection, context=context)
    host = replace(host, layout=layout, pool_bytes=(512, 256))
    gpu = replace(gpu, layer_id=9, life_cycle_id=5)
    resolve_entries(selection, host, gpu, out)
    assert out.host_offsets.cpu().tolist() == [
        [[216, 227, 108], [448, 480, 224], [456, 481, 228]],
        [[192, 224, 96], [200, 225, 100], [-1, -1, -1]],
    ]
    assert out.valid.cpu().tolist() == [[True] * 3, [True, True, False]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_padding_invalid_slots_masks_and_stale_request_ids() -> None:
    selection, host, gpu, out = _inputs([[0, 1, 4, -2147483648], [0, 1, 4, 5]])
    gpu.entry_indices.fill_(9)
    host.page_slots[0, 0] = torch.iinfo(torch.int64).max
    host.page_slots[0, 1] = -1
    mask = torch.tensor([[True, False, True, True], [True, True, False, True]], device="cuda")
    selection = replace(selection, valid_mask=mask)
    host.request_ids[1] = 999
    gpu.request_ids[1] = 999
    resolve_entries(selection, host, gpu, out)
    assert out.valid.cpu().tolist() == [[True, False, True, False], [True, True, False, True]]
    assert not out.host_valid.any().item()
    assert out.gpu_indices.cpu().tolist() == [[9, -1, 9, -1], [-1] * 4]
    selection.context.request_indices.copy_(torch.tensor([-1, 2], device="cuda", dtype=torch.int32))
    resolve_entries(selection, host, gpu, out)
    assert not out.valid.any().item()
    assert (out.host_offsets == -1).all().item()
    assert (out.gpu_indices == -1).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(0, 4), (2, 0), (2, 4)])
def test_empty_selection_and_empty_storage(shape: tuple[int, int]) -> None:
    selection, host, gpu, _ = _inputs([[0], [0]])
    ctx = replace(
        selection.context,
        request_indices=selection.context.request_indices[: shape[0]],
        valid_lengths=selection.context.valid_lengths[: shape[0]],
    )
    selection = replace(
        selection, context=ctx, positions=torch.zeros(shape, device="cuda", dtype=torch.int32)
    )
    host = replace(host, page_slots=host.page_slots[:, :0], valid_entries=host.valid_entries[:, :0])
    gpu = replace(gpu, entry_indices=gpu.entry_indices[:, :0])
    out = _outputs(shape)
    resolve_entries(selection, host, gpu, out)
    assert not out.host_valid.any().item()
    assert (out.host_offsets == -1).all().item()
    assert (out.gpu_indices == -1).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strided_topk_and_changed_graph_inputs() -> None:
    selection, host, gpu, out = _inputs([[0, 99, 4, 99], [1, 99, 5, 99]])
    selection = replace(selection, positions=selection.positions[:, ::2])
    out = _outputs((2, 2))
    resolve_entries(selection, host, gpu, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        resolve_entries(selection, host, gpu, out)
    output_address = out.host_offsets.data_ptr()
    graph.replay()
    assert out.host_offsets[0, 0].cpu().tolist() == [64, 96, 32]
    # Same addresses, new request identity and positions. Stale maps must miss.
    selection.positions.copy_(torch.tensor([[5, -1], [4, 0]], device="cuda", dtype=torch.int32))
    selection.context.request_ids[0] = 300
    graph.replay()
    assert out.valid.cpu().tolist() == [[True, False], [True, True]]
    assert out.host_valid.cpu().tolist() == [[False, False], [True, True]]
    host.request_ids[0] = 300
    gpu.request_ids[0] = 300
    gpu.entry_indices[0, 5] = 21
    graph.replay()
    assert out.host_offsets[0, 0].cpu().tolist() == [200, 225, 100]
    assert out.gpu_indices.cpu().tolist() == [[21, -1], [-1, -1]]
    assert out.host_offsets.data_ptr() == output_address


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("wrong", ["layer", "lifecycle", "gpu_layer", "output_shape"])
def test_rejects_cross_layer_resolution_and_wrong_outputs(wrong: str) -> None:
    selection, host, gpu, out = _inputs([[0], [1]])
    if wrong == "layer":
        selection = replace(selection, context=replace(selection.context, layer_id=8))
    elif wrong == "lifecycle":
        selection = replace(selection, context=replace(selection.context, life_cycle_id=8))
    elif wrong == "gpu_layer":
        gpu = replace(gpu, layer_id=8)
    else:
        out = _outputs((2, 2))
    with pytest.raises(ValueError):
        resolve_entries(selection, host, gpu, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resolution_across_topk_tiles_and_pool_capacity() -> None:
    positions = [[i % 10 - 1 for i in range(257)], [8 - i % 10 for i in range(257)]]
    selection, host, gpu, out = _inputs(positions)
    gpu.entry_indices[:, 4] = 23
    # The second pool cannot hold slot 3. A partial component set is not usable.
    host = replace(host, pool_bytes=(256, 96))
    resolve_entries(selection, host, gpu, out)
    assert out.valid.cpu().tolist() == [
        [0 <= p < limit for p in row] for row, limit in zip(positions, [7, 6])
    ]
    assert out.host_valid.cpu().tolist() == [
        [0 <= p < limit for p in row] for row, limit in zip(positions, [4, 6])
    ]
    assert out.gpu_indices.cpu().tolist() == [
        [23 if p == 4 else -1 for p in row] for row in positions
    ]
    offsets = out.host_offsets.cpu()
    assert (offsets[0, torch.tensor([p >= 4 or p < 0 for p in positions[0]])] == -1).all()
    assert offsets[1, 256].tolist() == [144, 162, 72]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_active_requests() -> None:
    selection, host, gpu, out = _inputs([[0, -1], [1, 4]])
    context = replace(selection.context, request_ids=selection.context.request_ids[:0])
    selection = replace(selection, context=context)
    host = replace(
        host,
        request_ids=host.request_ids[:0],
        page_slots=host.page_slots[:0],
        valid_entries=host.valid_entries[:0],
    )
    gpu = replace(gpu, request_ids=gpu.request_ids[:0], entry_indices=gpu.entry_indices[:0])
    resolve_entries(selection, host, gpu, out)
    assert not out.valid.any().item()
    assert not out.host_valid.any().item()
    assert (out.host_offsets == -1).all().item()
    assert (out.gpu_indices == -1).all().item()
