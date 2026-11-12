# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB transport: block layout, peer copies and migration."""

import ctypes
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import EpMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import transport as tr
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
    LayerBlockLayout,
    PeerSlotTransport,
    SlotPlane,
    find_backends,
    migrate_backend_weights,
    peer_copy_gap,
)

pytestmark = pytest.mark.cpu_only

EP_SIZE, EP_RANK, SLOTS = 4, 1, 3


def _plane(name: str, slot_bytes: int, alignment: int = 1) -> SlotPlane:
    return SlotPlane(name, (SLOTS, slot_bytes), (slot_bytes, 1), torch.uint8, alignment)


PLANES = (_plane("w13", 24, 16), _plane("w13_sf", 6, 4), _plane("w2", 12, 16), _plane("alpha", 4))
STREAM = object()
CPU = torch.device("cpu")


class _Comm:
    """The EP group: allgather ``n`` answers ``answers[n](value)``, else ``[value] * EP_SIZE``."""

    Get_rank = staticmethod(lambda: EP_RANK)
    Get_size = staticmethod(lambda: EP_SIZE)

    def __init__(self, answers=None) -> None:
        self.answers = answers or {}
        self.calls = 0

    def allgather(self, value: object) -> list:
        self.calls += 1
        answer = self.answers.get(self.calls)
        return answer(value) if answer else [value] * EP_SIZE


def _memory(comm: _Comm) -> type:
    """An MNNVL memory class over ``comm`` whose blocks are rows of one CPU tensor."""

    class Memory:
        blocks: list[torch.Tensor] = []
        mapped = 0  # blocks not yet collected: a real block unmaps its memory when collected

        @classmethod
        def get_comm(cls, mapping: object) -> _Comm:
            return comm

        def __init__(self, mapping: object, nbytes: int) -> None:
            self.block = torch.zeros(EP_SIZE, nbytes, dtype=torch.uint8)
            Memory.blocks.append(self.block)
            Memory.mapped += 1

        def __del__(self) -> None:
            Memory.mapped -= 1

        def as_torch_strided_tensor(self, dtype: torch.dtype) -> torch.Tensor:
            return self.block.view(dtype)

    return Memory


class _Batches:
    """Copies a batch with ``memmove``; logs the batches and the event records in order."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.log: list[str] = []

    def __call__(self, dst, src, sizes, stream) -> None:
        self.calls.append((list(dst), list(src), list(sizes), stream))
        self.log.append("batch")
        for to, frm, count in zip(dst, src, sizes):
            ctypes.memmove(to, frm, count)

    def event(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(record=lambda stream: self.log.append((name, stream)))


def _transport(comm=None, batches=None) -> PeerSlotTransport:
    memory = _memory(comm or _Comm())
    return PeerSlotTransport(None, memory=memory, copy_batch=batches or _Batches())


def _layout(planes=PLANES) -> LayerBlockLayout:
    return LayerBlockLayout(planes, SLOTS, min_alignment=8)


# ------------------------------------------------------------------------------------------------
# Block layout
# ------------------------------------------------------------------------------------------------


def test_layout_aligns_every_plane_and_keeps_them_apart():
    layout = _layout()
    assert layout.offsets == {"w13": 0, "w13_sf": 72, "w2": 96, "alpha": 136}
    assert layout.nbytes == 152
    assert layout.expert_bytes == 24 + 6 + 12 + 4
    spans = sorted(
        (layout.offsets[p.name], layout.offsets[p.name] + SLOTS * p.slot_bytes) for p in PLANES
    )
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))


# ------------------------------------------------------------------------------------------------
# Peer copies
# ------------------------------------------------------------------------------------------------


def test_peer_table_views_every_rank_block():
    transport = _transport()
    transport.add_layer(0, _layout())  # the returned block is not kept: the transport keeps it
    gc.collect()
    assert transport._memory.mapped == 1
    assert transport.expert_bytes(0) == sum(plane.slot_bytes for plane in PLANES)
    local = transport.local_planes(0)
    assert (transport.ep_rank, transport.ep_size) == (EP_RANK, EP_SIZE)
    (block,) = transport._memory.blocks
    for plane in PLANES:
        rows = transport._plane_rows[0][plane.name]
        assert rows.shape == (EP_SIZE, SLOTS, plane.slot_bytes)
        assert local[plane.name].data_ptr() == rows[EP_RANK].data_ptr()
        for rank in range(EP_SIZE):
            for slot in range(SLOTS):
                start = block[rank].data_ptr() + _layout().offsets[plane.name]
                assert rows[rank, slot].data_ptr() == start + slot * plane.slot_bytes


def test_a_copy_moves_every_plane_row_in_one_batch_between_its_events_and_nothing_else():
    batches = _Batches()
    transport = _transport(batches=batches)
    for layer in range(2):
        transport.add_layer(layer, _layout())
    blocks = transport._memory.blocks
    for index, block in enumerate(blocks):
        block.copy_((torch.arange(block.numel()).view_as(block) * (index + 3)) % 251)
    before = [block.clone() for block in blocks]
    events = (batches.event("start"), batches.event("end"))
    transport.copy([(1, 2, 3, 0), (0, 1, 0, 2)], STREAM, *events)
    assert batches.log == [("start", STREAM), "batch", ("end", STREAM)]
    ((dst, src, sizes, stream),) = batches.calls
    assert (len(dst), stream) == (2 * len(PLANES), STREAM)
    expected = [block.clone() for block in before]
    for item, first in (((1, 2, 3, 0), 0), ((0, 1, 0, 2), 4)):
        layer, dst_slot, src_rank, src_slot = item
        for index, plane in enumerate(PLANES):
            rows = transport._plane_rows[layer][plane.name]
            assert dst[first + index] == rows[EP_RANK, dst_slot].data_ptr()
            assert src[first + index] == rows[src_rank, src_slot].data_ptr()
            assert sizes[first + index] == plane.slot_bytes
            start = _layout().offsets[plane.name]
            size = plane.slot_bytes
            target = slice(start + dst_slot * size, start + (dst_slot + 1) * size)
            source = slice(start + src_slot * size, start + (src_slot + 1) * size)
            expected[layer][EP_RANK, target] = before[layer][src_rank, source]
    for got, want in zip(blocks, expected):
        assert torch.equal(got, want)
    transport.copy([], STREAM, *events)  # nothing to copy: no batch, the events still recorded
    assert len(batches.calls) == 1 and batches.log[3:] == [("start", STREAM), ("end", STREAM)]


def test_a_rank_with_another_block_layout_is_named():
    short = _layout(PLANES[:2] + PLANES[3:]).signature()
    transport = _transport(_Comm(answers={1: lambda value: [value, value, short, value]}))
    with pytest.raises(ValueError, match=r"ranks \[2\] lay out the block differently"):
        transport.add_layer(0, _layout())
    assert not transport._memory.blocks


def test_ep_mnnvl_memory_groups_ranks_by_ep_rank_with_its_own_state():
    assert issubclass(EpMnnvlMemory, MnnvlMemory)
    assert EpMnnvlMemory.allocated_map is not MnnvlMemory.allocated_map
    pg = object()
    fixed = dict(cp_rank=0, cp_size=1, moe_tp_size=2, moe_cluster_size=1, moe_cluster_rank=0)

    def mapping(pp_rank, moe_tp_rank, moe_ep_rank):
        ranks = dict(pp_rank=pp_rank, moe_tp_rank=moe_tp_rank, moe_ep_rank=moe_ep_rank)
        return SimpleNamespace(**fixed, **ranks, moe_ep_group_pg=pg)

    assert EpMnnvlMemory.comm_process_group(mapping(0, 0, 0)) is pg
    ranks = [(0, 0, 0), (0, 0, 3), (0, 1, 3), (1, 0, 3)]
    keys = {key: EpMnnvlMemory.comm_split_color_key(mapping(*key)) for key in ranks}
    assert keys[(0, 0, 0)][0] == keys[(0, 0, 3)][0]
    assert len({color for color, _ in keys.values()}) == 3
    assert [key for _, key in keys.values()] == [0, 3, 3, 3]


# ------------------------------------------------------------------------------------------------
# Weight migration
# ------------------------------------------------------------------------------------------------


class _Backend(torch.nn.Module):
    """A backend with three slot planes over numpy storage; its quant scales hold one of them.

    ``shared`` makes the ``sf`` plane a view into a larger storage, which cannot move whole.
    """

    capabilities = SimpleNamespace(supports_adaptive_eplb=True)
    slot_id_dtype = torch.int32

    def __init__(self, layer: int, handle: object, shared: bool = False) -> None:
        super().__init__()
        self.layer_idx = layer
        self.layer_load_balancer = handle
        self.arrays = {
            "w": np.arange(SLOTS * 2 * 8, dtype=np.uint8).reshape(SLOTS, 2, 8) + layer,
            "sf": np.arange(SLOTS * 5, dtype=np.uint8).reshape(SLOTS, 5) * (layer + 3),
            "alpha": np.linspace(1.0, 2.0, SLOTS, dtype=np.float32) + layer,
        }
        self.values = {name: torch.from_numpy(a.copy()) for name, a in self.arrays.items()}
        for name, array in self.arrays.items():
            tensor = torch.from_numpy(array)
            if shared and name == "sf":
                tensor = torch.zeros(SLOTS + 1, 5, dtype=torch.uint8)[1:]
            self.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
        self.quant_scales = (self.alpha,)
        self.views: dict = {}

    def slot_weight_planes(self) -> list[tuple[str, int]]:
        return [("w", 16), ("sf", 16), ("alpha", 16)]

    def check_weights_movable(self) -> None:
        if self.views:
            raise RuntimeError(f"MoE layer {self.layer_idx} has launched its kernel")

    def release_arrays(self) -> list:
        refs = [weakref.ref(array) for array in self.arrays.values()]
        self.arrays = {}
        return refs


def _backends(layers=(0, 1), shared=False):
    return {layer: _Backend(layer, object(), shared) for layer in layers}


def _pointers(backends) -> dict:
    return {
        (layer, name): param.data_ptr()
        for layer, module in backends.items()
        for name, param in module.named_parameters()
    }


def test_parameters_take_their_block_rows_and_keep_their_identity():
    backends = _backends()
    params = {
        (layer, name): param
        for layer, module in backends.items()
        for name, param in module.named_parameters()
    }
    transport = _transport()
    assert migrate_backend_weights(backends, transport, CPU) == 2 * SLOTS * (16 + 5 + 4)
    blocks = transport._memory.blocks
    assert len(blocks) == 2
    for layer, module in backends.items():
        offsets = {}
        cursor = 0
        for name, _ in module.slot_weight_planes():
            param = module.get_parameter(name)
            cursor = (cursor + 511) // 512 * 512
            offsets[name] = cursor
            cursor += param.numel() * param.element_size()
        for name, param in module.named_parameters():
            assert param is params[(layer, name)]
            assert param.data_ptr() == blocks[layer][EP_RANK].data_ptr() + offsets[name]
            assert torch.equal(param.data, module.values[name])
        assert module.quant_scales[0] is module.alpha
        assert module.quant_scales[0].data_ptr() == module.alpha.data_ptr()
        assert module.adaptive_eplb_block.block is blocks[layer]
    # The module keeps the block its parameters point into after the transport is gone.
    held = [weakref.ref(module.adaptive_eplb_block) for module in backends.values()]
    del transport
    gc.collect()
    assert all(ref() is not None for ref in held)
    # The original storage is released; the block once its module is gone too.
    refs = [ref for module in backends.values() for ref in module.release_arrays()]
    gc.collect()
    assert all(ref() is None for ref in refs)
    del backends, module
    gc.collect()
    assert all(ref() is None for ref in held)


@pytest.mark.parametrize(
    "answers,shared,ran,error",
    [
        # A plane that does not own its storage: every rank stops before any block.
        ({}, True, None, r"checking the backend planes: rank 0: .*'sf'"),
        # A peer that cannot check stops this rank too.
        ({1: lambda v: [v, "MoE layer 0: no planes", v, v]}, False, None, "rank 1: MoE layer 0"),
        # A module that ran a forward keeps views of its planes.
        ({}, False, 1, "MoE layer 1 has launched its kernel"),
    ],
)
def test_a_failure_on_any_rank_stops_every_rank_and_moves_nothing(answers, shared, ran, error):
    backends = _backends(shared=shared)
    if ran is not None:
        backends[ran].views["view"] = object()
    before = _pointers(backends)
    transport = _transport(_Comm(answers=answers))
    with pytest.raises(RuntimeError, match=error):
        migrate_backend_weights(backends, transport, CPU)
    assert not transport._memory.blocks and _pointers(backends) == before


def test_find_backends_takes_the_capable_backend_of_every_layer(monkeypatch):
    monkeypatch.setattr(tr, "MoEImplBase", torch.nn.Module)
    handles = [SimpleNamespace(layer_idx=index) for index in range(3)]
    model = torch.nn.Module()
    for index in (0, 1):
        wrapper = torch.nn.Module()
        wrapper.layer_load_balancer = handles[index]
        wrapper.capabilities = SimpleNamespace(supports_adaptive_eplb=False)
        wrapper.backend = _Backend(index, handles[index])
        model.add_module(f"moe{index}", wrapper)
    with pytest.raises(ValueError, match=r"balanced layers \[2\] have no MoE backend"):
        find_backends(model, handles)
    assert find_backends(model, handles[:2]) == {0: model.moe0.backend, 1: model.moe1.backend}
    model.add_module("twin", _Backend(1, handles[1]))
    with pytest.raises(ValueError, match="layer 1 has two backend modules"):
        find_backends(model, handles[:2])


def test_the_copies_need_the_ep_ranks_in_one_nvlink_domain(monkeypatch):
    """EP ranks on one node always reach each other; across nodes only within an MNNVL domain."""
    asked = []
    monkeypatch.setattr(MnnvlMemory, "supports_mnnvl", lambda: asked.append(True) or False)
    monkeypatch.setattr(tr, "_batch_attributes", lambda: None)
    one_node = SimpleNamespace(gpus_per_node=4, moe_ep_group=[4, 5, 6, 7])
    two_nodes = SimpleNamespace(gpus_per_node=4, moe_ep_group=list(range(8)))
    assert peer_copy_gap(one_node) is None and not asked
    assert peer_copy_gap(two_nodes) == "the EP ranks span 2 nodes outside one MNNVL domain"
    monkeypatch.setattr(MnnvlMemory, "supports_mnnvl", lambda: True)
    assert peer_copy_gap(two_nodes) is None

    def no_batches():
        raise RuntimeError("adaptive EPLB copies experts with cudaMemcpyBatchAsync")

    monkeypatch.setattr(tr, "_batch_attributes", no_batches)
    assert "cudaMemcpyBatchAsync" in peer_copy_gap(one_node)
