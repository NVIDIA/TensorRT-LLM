# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``layout_mapping`` against a byte oracle: each rank's page holds, at every byte of every tensor,
the id of the global element there; after every source rank of an instance sends what
``map_buffers`` gives it, every byte a destination rank holds of every buffer both transfer has
arrived exactly once, holding its own global element, and only from source ranks it pairs with."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import fan, map_buffers, merge_runs
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    Axis,
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    Progression,
    RankTopology,
    Tensor,
    head_major,
    held,
    replicated,
    sectioned,
    token_major,
)

pytestmark = pytest.mark.cpu_only

HEAD = 16  # bytes of one head of one page
TOKENS = 4


@dataclass(frozen=True)
class Instance:
    """An instance: its parallel sizes and the global layers of each pipeline stage."""

    tp: int = 1
    cp: int = 1
    attention_dp: bool = False
    stages: tuple[tuple[int, ...], ...] = ((0, 1),)

    def ranks(self, dp: int) -> list[RankTopology]:
        """The ranks serving the requests of attention data-parallel rank ``dp``."""
        tps = [dp] if self.attention_dp else range(self.tp)
        return [
            RankTopology(
                tp_size=self.tp,
                tp_rank=t,
                pp_size=len(self.stages),
                pp_rank=p,
                cp_size=self.cp,
                cp_rank=c,
                dp_size=self.tp if self.attention_dp else 1,
                dp_rank=t if self.attention_dp else 0,
                attention_dp=self.attention_dp,
            )
            for p in range(len(self.stages))
            for t in tps
            for c in range(self.cp)
        ]


# A buffer spec: (role, topology -> tensors, kind of its layer group).
Spec = tuple[str, Callable[[RankTopology], tuple], str]


def _tp(topology: RankTopology) -> int:
    return 1 if topology.attention_dp else topology.tp_size


def kv(total: int, layout: str = "hnd") -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        heads = -(-total // _tp(topology))
        if layout == "nhd":
            return token_major(TOKENS * heads * HEAD, TOKENS, heads, total)
        return head_major(heads * HEAD, heads, total)

    return tensors


def state_heads(total: int) -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        ranks = _tp(topology) * topology.cp_size
        return head_major(total // ranks * HEAD, total // ranks, total, ("tp", "cp"))

    return tensors


def conv(sections: tuple[int, ...]) -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        ranks = _tp(topology) * topology.cp_size
        return sectioned([s // ranks for s in sections], ranks, ("tp", "cp"))

    return tensors


def whole(size: int) -> Callable[[RankTopology], tuple]:
    return lambda topology: replicated(size)


def layout(topology: RankTopology, layers: tuple[int, ...], specs: list[Spec]) -> PageLayout:
    """One rank's page: each kind's layers in a pool group of their own, buffers back to back."""
    kinds = sorted({kind for _, _, kind in specs})
    buffers, widths = [], {}
    for g, kind in enumerate(kinds):
        offset = 0
        for layer in layers:
            for role, make, k in specs:
                if k != kind:
                    continue
                tensors = make(topology)
                size = max(
                    t.offset + 1 + sum((len(held(a, topology)) - 1) * a.stride for a in t.axes)
                    for t in tensors
                )
                buffers.append(BufferDesc(layer, role, g, 0, offset, size, tensors))
                offset += size
        widths[g] = (offset,)
    groups = tuple(
        LayerGroupDesc(kind, None, 0, g, layers) for g, kind in enumerate(kinds) if widths[g][0]
    )
    return PageLayout(TOKENS, widths, groups, tuple(buffers), topology)


@dataclass
class Oracle:
    """Global element ids, allotted per (layer, role, tensor)."""

    bases: dict = field(default_factory=dict)
    next_id: int = 0

    def base(self, key: tuple, tensor: Tensor) -> int:
        if key not in self.bases:
            self.bases[key] = self.next_id
            self.next_id += int(np.prod([a.extent for a in tensor.axes]))
        return self.bases[key]

    def elements(self, page: PageLayout, buffer: BufferDesc, tensor: Tensor):
        """(byte offset in the pool slot, global element id) of each byte of the tensor's piece."""
        base = self.base((buffer.layer, buffer.role, tensor.name), tensor)
        holds = [list(held(a, page.topology)) for a in tensor.axes]
        extents = [a.extent for a in tensor.axes]
        for local in itertools.product(*(range(len(h)) for h in holds)):
            offset = buffer.slot_offset + tensor.offset
            linear = 0
            for a, h, k, extent in zip(tensor.axes, holds, local, extents):
                offset += k * a.stride
                linear = linear * extent + h[k]
            yield offset, base + linear

    def memory(self, page: PageLayout) -> dict[int, np.ndarray]:
        mem = {g: np.full(w[0], -1, dtype=np.int64) for g, w in page.pool_slot_bytes.items()}
        for b in page.buffers:
            for t in b.tensors:
                for offset, element in self.elements(page, b, t):
                    mem[b.pool_group][offset] = element
        return mem


def check(
    src: Instance, dst: Instance, specs: list[Spec], *, src_dp: int = 0, in_fan: bool = True
) -> None:
    """Every destination rank of every data-parallel group gets each byte it holds exactly once,
    holding the right element, and, under contiguous splits (``in_fan``), only from source ranks
    of its fan."""
    oracle = Oracle()
    sources = [(t, layout(t, src.stages[t.pp_rank], specs)) for t in src.ranks(src_dp)]
    src_memory = {id(page): oracle.memory(page) for _, page in sources}
    for dst_dp in range(dst.tp if dst.attention_dp else 1):
        for dst_topology in dst.ranks(dst_dp):
            page = layout(dst_topology, dst.stages[dst_topology.pp_rank], specs)
            got = {g: np.full(w[0], -1, dtype=np.int64) for g, w in page.pool_slot_bytes.items()}
            writes = {g: np.zeros(w[0], dtype=np.int64) for g, w in page.pool_slot_bytes.items()}
            for src_topology, src_page in sources:
                maps = map_buffers(src_page, page)
                if maps and in_fan:
                    tp_fan = fan(
                        _tp(src_topology),
                        _tp(dst_topology),
                        0 if dst_topology.attention_dp else dst_topology.tp_rank,
                    )
                    cp_fan = fan(src_topology.cp_size, dst_topology.cp_size, dst_topology.cp_rank)
                    src_tp = 0 if src_topology.attention_dp else src_topology.tp_rank
                    assert src_tp in tp_fan and src_topology.cp_rank in cp_fan, (
                        src_topology,
                        dst_topology,
                    )
                memory = src_memory[id(src_page)]
                for m in maps:
                    for s, d, n in zip(m.src_offsets, m.dst_offsets, m.sizes):
                        got[m.dst.pool_group][d : d + n] = memory[m.src.pool_group][s : s + n]
                        writes[m.dst.pool_group][d : d + n] += 1
            src_layers = {layer for stage in src.stages for layer in stage}
            for b in page.buffers:
                if b.layer not in src_layers:
                    continue
                for t in b.tensors:
                    for offset, element in oracle.elements(page, b, t):
                        assert writes[b.pool_group][offset] == 1, (
                            f"{dst_topology} buffer ({b.layer}, {b.role}) tensor {t.name!r} byte "
                            f"{offset}: {writes[b.pool_group][offset]} writes"
                        )
                        assert got[b.pool_group][offset] == element, (
                            f"{dst_topology} buffer ({b.layer}, {b.role}) byte {offset}: element "
                            f"{got[b.pool_group][offset]}, not {element}"
                        )


TPS = (1, 2, 4, 8)


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 6])
@pytest.mark.parametrize("kv_layout", ["hnd", "nhd"])
@pytest.mark.parametrize("src_tp,dst_tp", list(itertools.product(TPS, TPS)))
def test_attention_heads_reshard_over_tensor_ranks(heads, kv_layout, src_tp, dst_tp):
    """Heads split unevenly (6 over 4 or 8 ranks) reshard too, though a destination may then take
    a head from a source rank outside its fan."""
    specs = [("key", kv(heads, kv_layout), "attention"), ("index", whole(24), "attention")]
    aligned = all(heads % tp == 0 or tp % heads == 0 for tp in (src_tp, dst_tp))
    check(Instance(tp=src_tp), Instance(tp=dst_tp), specs, in_fan=aligned)


@pytest.mark.parametrize("src_adp,dst_adp", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("src_tp,dst_tp", [(2, 4), (4, 2), (4, 4), (1, 4)])
def test_attention_data_parallel_holds_whole_heads(src_adp, dst_adp, src_tp, dst_tp):
    specs = [("key", kv(8), "attention"), ("index", whole(24), "attention")]
    for src_dp in range(src_tp if src_adp else 1):
        check(
            Instance(tp=src_tp, attention_dp=src_adp),
            Instance(tp=dst_tp, attention_dp=dst_adp),
            specs,
            src_dp=src_dp,
        )


@pytest.mark.parametrize("src_tp,dst_tp", list(itertools.product((1, 2, 4), (1, 2, 4))))
@pytest.mark.parametrize("dst_cp", [1, 2])
def test_recurrent_state_splits_over_tensor_times_context_ranks(src_tp, dst_tp, dst_cp):
    specs = [
        ("ssm", state_heads(8), "state"),
        ("conv", conv((16, 8, 8)), "state"),
        ("ple", whole(12), "state"),
        ("key", kv(4), "attention"),
    ]
    check(Instance(tp=src_tp), Instance(tp=dst_tp, cp=dst_cp), specs)


@pytest.mark.parametrize(
    "src_stages,dst_stages",
    [
        (((0, 1, 2, 3),), ((0, 1), (2, 3))),
        (((0, 1), (2, 3)), ((0,), (1, 2), (3,))),
        (((0, 1, 2), (3,)), ((0, 1, 2, 3),)),
    ],
)
@pytest.mark.parametrize("src_tp,dst_tp", [(1, 2), (2, 1), (2, 2)])
def test_pipeline_stages_pair_by_layer(src_stages, dst_stages, src_tp, dst_tp):
    specs = [("key", kv(4), "attention"), ("ssm", state_heads(4), "state")]
    check(Instance(tp=src_tp, stages=src_stages), Instance(tp=dst_tp, stages=dst_stages), specs)


def test_a_two_dimensional_split_reshards_both_axes():
    """Heads over tensor ranks and tokens over context ranks, which no single split expresses."""

    def tensors(topology):
        heads = 4 // topology.tp_size
        tokens = 8 // topology.cp_size
        return (
            Tensor(
                (
                    Axis("token", 8, heads * HEAD, ("cp",), tokens),
                    Axis("head", 4, HEAD, ("tp",), heads),
                    Axis("byte", HEAD, 1),
                )
            ),
        )

    specs = [("key", tensors, "attention")]
    for src, dst in [((1, 1), (2, 2)), ((2, 2), (1, 1)), ((2, 1), (1, 2)), ((4, 2), (2, 4))]:
        check(Instance(tp=src[0], cp=src[1]), Instance(tp=dst[0], cp=dst[1]), specs)


def test_a_cyclic_split_hands_each_rank_every_nth_index():
    def tensors(topology):
        return (Tensor((Axis("row", 12, HEAD, ("tp",), cyclic=True), Axis("byte", HEAD, 1))),)

    specs = [("rows", tensors, "attention")]
    for src_tp, dst_tp in [(1, 3), (3, 1), (2, 4), (4, 2), (3, 2)]:
        check(Instance(tp=src_tp), Instance(tp=dst_tp), specs, in_fan=False)


def test_runs_merge_where_both_sides_continue():
    src = np.array([0, 8, 32, 40], dtype=np.int64)
    dst = np.array([100, 108, 200, 240], dtype=np.int64)
    sizes = np.array([8, 8, 8, 8], dtype=np.int64)
    merged = merge_runs(src, dst, sizes)
    assert [list(a) for a in merged] == [[0, 32, 40], [100, 200, 240], [16, 8, 8]]


def test_a_whole_page_moves_as_one_run_when_layouts_match():
    specs = [("key", kv(4), "attention"), ("value", kv(4), "attention")]
    topology = Instance(tp=2).ranks(0)[1]
    page = layout(topology, (0, 1), specs)
    maps = map_buffers(page, page)
    src = np.concatenate([m.src_offsets for m in maps])
    sizes = np.concatenate([m.sizes for m in maps])
    assert len(src) == len(maps) and int(sizes.sum()) == page.pool_slot_bytes[0][0]


def test_progressions_intersect_by_their_common_residues():
    a, b = Progression(1, 4, 10), Progression(3, 6, 10)  # 1, 5, 9, ...; 3, 9, 15, ...
    assert a.intersect(b) == Progression(9, 12, 3)  # 9, 21, 33
    assert Progression(0, 2, 5).intersect(Progression(1, 2, 5)) is None
    assert Progression(4, 1, 2).intersect(Progression(0, 1, 4)) is None


def test_mismatched_tensors_are_refused():
    topology = Instance().ranks(0)[0]
    a = layout(topology, (0,), [("key", kv(4), "attention")])
    b = layout(topology, (0,), [("key", kv(8), "attention")])
    with pytest.raises(ValueError, match="extent"):
        map_buffers(a, b)
