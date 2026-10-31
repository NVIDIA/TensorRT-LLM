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
"""Attached dense DWDP beside the expert prefetch on real streams, events and copies.

A real ``DenseDWDP`` runtime, with staging slots from ``allocate_slot_ring`` and a launcher that
copies each layer's source bytes into the layer's slots, and a real ``DWDPWeightManager``, over a
stand-in for ``WeightBuffer`` whose prefetch slots hold the remote experts, share one copy stream.
The attached ``DenseDwdpManager`` holds both.  A harness drives them the way the executor loop
(``DwdpManager.start_forward``), ``KimiLinearModel`` (H0-H3) and ``ConfigurableMoE`` do, with a
read kernel in every read window: the attention weights between H1 and H2, the MoE-dense weights
before the expert bind and after the expert finish, the experts between the bind and the finish.
Delaying the reads exposes a missing WAR wait (a copy overwrites a slot that is still being read),
delaying the copies a missing RAW wait.  Every read is checked against the bytes of its layer.
"""

import itertools
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    DenseMode,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
NUM_EXPERTS = 4  # experts per rank
DELAY_CYCLES = 400_000
ABORT_POINTS = ("H1", "H2", "finish", "H3")


class _ExpertSlots:
    """Stand-in for ``WeightBuffer`` of DWDP rank 0 of two: rows [0, NUM_EXPERTS) of a layer's
    full tensor play the (unused) local experts, the other rows are its slot's remote experts."""

    local_start = 0
    local_end = NUM_EXPERTS

    def __init__(self, layers: List[int], num_slots: int, dim: int) -> None:
        self.device_id = torch.cuda.current_device()
        self.num_slots = num_slots
        self._layers = layers
        self.slots = [
            torch.full((2 * NUM_EXPERTS, dim), -1.0, dtype=torch.float64, device="cuda")
            for _ in range(num_slots)
        ]

    def buffer_index_for_layer(self, layer: int) -> int:
        return self._layers.index(layer) % self.num_slots

    def get_full_tensor(self, layer: int, name: str) -> torch.Tensor:
        return self.slots[self.buffer_index_for_layer(layer)]

    def get_remote_slices(self, layer: int, name: str) -> List[Tuple[torch.Tensor, int, int]]:
        full = self.get_full_tensor(layer, name)
        return [(full[NUM_EXPERTS:], NUM_EXPERTS, full.shape[0])]


class _Backend(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.empty(0, dtype=torch.float64, device="cuda"), requires_grad=False
        )


class _Abort(Exception):
    pass


class _Harness:
    """Attached dense DWDP and expert DWDP of ``layers``, driven in Kimi-K3's program order.

    Args:
        layers: Covered layers, which are also the MoE layers.
        segments: Enabled dense segments.
        num_slots: Expert prefetch slots.
        dense_bytes: Bytes of each layer's weights per dense segment.
        dim: Columns of each expert row.
        copy_delay: Cycles each copy waits on the copy stream before it starts.
    """

    def __init__(
        self,
        layers: List[int],
        segments: Tuple[SegmentKind, ...],
        num_slots: int,
        dense_bytes: int,
        dim: int,
        copy_delay: int = 0,
    ) -> None:
        self.layers = layers
        self.segments = segments
        self.dense_bytes = dense_bytes
        self.copy_delay = copy_delay
        self.copies: List[Tuple[str, int]] = []
        self.dense_src: Dict[Tuple[SegmentKind, int], torch.Tensor] = {
            (kind, layer): torch.randint(0, 256, (dense_bytes,), dtype=torch.uint8, device="cuda")
            for kind in segments
            for layer in layers
        }
        self.expert_src: Dict[int, torch.Tensor] = {
            layer: torch.randint(0, 64, (NUM_EXPERTS, dim), device="cuda").to(torch.float64)
            for layer in layers
        }
        self.buffer = _ExpertSlots(layers, num_slots, dim)
        self.wm = DWDPWeightManager(
            weight_buffer=self.buffer,
            peer_views={(1, layer, "w"): self.expert_src[layer] for layer in layers},
            peer_ranges=[(0, NUM_EXPERTS), (NUM_EXPERTS, 2 * NUM_EXPERTS)],
            moe_layer_indices=layers,
            weight_names=["w"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        slots = {
            kind: allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], dense_bytes, 1), "cuda")
            for kind in segments
        }
        self.dense = DenseDWDP(
            layers, segments, slots, self.wm.copy_stream, prefetch_next_pass=True
        )
        self.dense.set_launcher(self._launch_dense)
        self.dense.commit(None)
        self.dense_manager = DenseDwdpManager(DenseMode.ATTACHED)
        self.dense_manager.attach(self.dense, expert=self.wm)
        self.mgr = object.__new__(DwdpManager)
        self.mgr._weight_manager = self.wm
        self.mgr._dense_manager = self.dense_manager
        self.backend = _Backend()
        self.compute_stream = torch.cuda.Stream()
        self.results: List[Tuple[str, int, torch.Tensor]] = []

    def _launch_dense(self, layer: int) -> None:
        """The dense launcher: copies ``L(layer)`` into its slots on the current (copy) stream."""
        if self.copy_delay:
            torch.cuda._sleep(self.copy_delay)
        self.copies.append(("L", layer))
        for kind in self.segments:
            slot = self.dense.slots[kind][self.dense.slot_index(kind, layer)]
            slot.blob_view(self.dense_bytes).copy_(self.dense_src[(kind, layer)])

    def _read(self, what: str, layer: int, delay: int) -> None:
        """A read kernel in the read window, after ``delay`` cycles of other work."""
        if delay:
            torch.cuda._sleep(delay)
        if what == "E":
            value = self.backend.w[NUM_EXPERTS:].clone()
        else:
            kind = SegmentKind(what)
            slot = self.dense.slots[kind][self.dense.slot_index(kind, layer)]
            value = slot.blob_view(self.dense_bytes).clone()
        self.results.append((what, layer, value))

    def run_pass(
        self,
        read_delay: int = 0,
        start: bool = True,
        stop: Optional[Tuple[str, int]] = None,
    ) -> None:
        """One executor step; ``stop=(point, layer)`` raises ``_Abort`` right before that point."""
        if start:
            self.mgr.start_forward()
        self.compute_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.compute_stream):
            self.dense_manager.ensure_primed()  # H0
            for layer in self.layers:
                point = stop[0] if stop is not None and stop[1] == layer else None
                if point == "H1":
                    raise _Abort()
                self.dense.wait_layer(layer)  # H1
                self._read(ATTN.value, layer, read_delay)
                if point == "H2":
                    raise _Abort()
                self.dense.attention_done(layer)  # H2
                if MOE in self.segments:
                    self._read(MOE.value, layer, read_delay)
                self.mgr.wait_and_bind(self.backend, layer)
                self._read("E", layer, read_delay)
                if point == "finish":
                    raise _Abort()
                self.mgr.finish_layer(layer)
                if MOE in self.segments:
                    self._read(MOE.value, layer, read_delay)
                if point == "H3":
                    raise _Abort()
                self.dense.layer_done(layer)  # H3
        torch.cuda.current_stream().wait_stream(self.compute_stream)

    def mismatches(self) -> List[Tuple[str, int]]:
        """Reads that saw other bytes than their layer's; clears the reads."""
        torch.cuda.synchronize()
        bad = []
        for what, layer, value in self.results:
            if what == "E":
                expected = self.expert_src[layer]
            else:
                expected = self.dense_src[(SegmentKind(what), layer)]
            if not torch.equal(value, expected):
                bad.append((what, layer))
        self.results.clear()
        return bad


def _drive(h: _Harness, read_delay: int) -> None:
    """Warmup, steps with idle gaps, every abort point with its recovery, an expert update."""
    layers = h.layers
    # A forward the executor does not start: H0 primes, the end of the pass primes the next one.
    h.run_pass(read_delay, start=False)
    assert h.mismatches() == []
    assert h.dense.primed
    for step, delay in enumerate([read_delay, 0, read_delay]):
        del h.copies[:]
        h.run_pass(delay)
        assert h.mismatches() == [], f"step {step}"
        # Every layer's dense weights are gathered exactly once per pass.
        assert sorted(h.copies) == sorted(("L", layer) for layer in layers), f"step {step}"
        time.sleep(0.01)
        torch.cuda._sleep(DELAY_CYCLES)
    # Aborts: the next pass recovers, with the executor's start or, as in warmup, from H0 alone,
    # while the reads the aborted pass enqueued may still run.
    stops = itertools.product(ABORT_POINTS, (layers[0], layers[len(layers) // 2], layers[-1]))
    for index, stop in enumerate(stops):
        with pytest.raises(_Abort):
            h.run_pass(read_delay, stop=stop)
        h.run_pass(read_delay, start=index % 2 == 0)
        assert h.mismatches() == [], stop
    # New expert weights: the copies issued ahead hold the old ones until invalidate_prefetch.
    torch.cuda.synchronize()
    for tensor in h.expert_src.values():
        tensor.add_(1000.0)
    h.mgr.invalidate_prefetch()
    h.run_pass(read_delay)
    h.run_pass(read_delay)
    assert h.mismatches() == []
    assert h.dense.demand_fetches == 0 and h.dense.war_fences == 0


@pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
@pytest.mark.parametrize("num_layers", [4, 5])
@pytest.mark.parametrize("num_slots", [2, 3])
@pytest.mark.parametrize("slow", ["read", "copy"])
def test_dense_and_expert_reads_see_their_layers(
    segments: tuple[SegmentKind, ...], num_layers: int, num_slots: int, slow: str
) -> None:
    layers = [2 * i + 1 for i in range(num_layers)]  # non-contiguous decoder indices
    h = _Harness(
        layers,
        segments,
        num_slots,
        dense_bytes=1 << 16,
        dim=1024,
        copy_delay=DELAY_CYCLES if slow == "copy" else 0,
    )
    _drive(h, DELAY_CYCLES if slow == "read" else 0)


@pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
def test_kimi_k3_pass_boundary(segments: tuple[SegmentKind, ...]) -> None:
    """92 layers, two expert slots: the next pass's first layers are gathered at its end."""
    layers = list(range(1, 93))
    h = _Harness(layers, segments, 2, dense_bytes=1 << 12, dim=64, copy_delay=DELAY_CYCLES // 8)
    _drive(h, DELAY_CYCLES // 8)


def test_release_right_after_a_pass() -> None:
    """The end of a pass leaves the next pass's first layers in flight; release waits for them."""
    h = _Harness([1, 2, 3, 4, 5], BOTH, 2, dense_bytes=1 << 16, dim=256, copy_delay=DELAY_CYCLES)
    h.run_pass(DELAY_CYCLES)
    assert h.dense.primed and h.copies[-2:] == [("L", 1), ("L", 2)]
    h.dense.release()
    assert h.mismatches() == []
    assert h.dense.slots == {}


def test_missing_dense_consume_record_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sensitivity check: without H2's consume record the fill overwrites attention slots in use."""
    original = DenseDWDP._close_window

    def without_attention_record(
        self: DenseDWDP, kind: SegmentKind, slot: int, layer: int, hook: str
    ) -> None:
        if kind is ATTN:
            self._consumed[kind][slot] = True
            return
        original(self, kind, slot, layer, hook)

    monkeypatch.setattr(DenseDWDP, "_close_window", without_attention_record)
    h = _Harness([1, 2, 3, 4, 5], BOTH, 2, dense_bytes=1 << 16, dim=256)
    for _ in range(3):
        h.run_pass(DELAY_CYCLES)
    assert any(what == ATTN.value for what, _ in h.mismatches())
