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
"""Cross-step DWDP prefetch on one GPU: real streams, events and copies.

A one-GPU stand-in for ``WeightBuffer`` backs every layer's remote experts with its slot's
storage, and the peer's experts are plain tensors.  Each "MoE" multiplies activations of a
per-pass token count with the bound remote experts after a delay; delaying the MoE exposes a
missing WAR wait (the next copy overwrites the slot first), delaying the copies exposes a
missing RAW wait.  Every output is checked against the source weights of its pass.
"""

import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_REMOTE = 8  # remote experts per layer (owned by peer 1)
DIM = 4096
DTYPE = torch.float64
DELAY_CYCLES = 400_000


class _SlotBuffer:
    """Stand-in for ``WeightBuffer``: rows [NUM_REMOTE, 2 * NUM_REMOTE) of a layer's full tensor
    are its slot's remote experts; rows [0, NUM_REMOTE) play the (unused) local experts."""

    local_start = 0
    local_end = NUM_REMOTE

    def __init__(self, layers: List[int], num_slots: int) -> None:
        self.device_id = torch.cuda.current_device()
        self.num_slots = num_slots
        self._layers = layers
        self.slots = [
            torch.full((2 * NUM_REMOTE, DIM), -1.0, dtype=DTYPE, device="cuda")
            for _ in range(num_slots)
        ]

    def buffer_index_for_layer(self, layer: int) -> int:
        return self._layers.index(layer) % self.num_slots

    def get_full_tensor(self, layer: int, name: str) -> torch.Tensor:
        return self.slots[self.buffer_index_for_layer(layer)]

    def get_remote_slices(self, layer: int, name: str) -> List[Tuple[torch.Tensor, int, int]]:
        return [(self.get_full_tensor(layer, name)[NUM_REMOTE:], NUM_REMOTE, 2 * NUM_REMOTE)]

    def release(self) -> None:
        """Nothing is mapped; the slot tensors free themselves."""


class _Backend(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(0, dtype=DTYPE, device="cuda"), requires_grad=False)


class _AbortPass(Exception):
    pass


class _Harness:
    """Drives the facade the way the executor loop and ``ConfigurableMoE`` do."""

    def __init__(self, layers: List[int], num_slots: int, monkeypatch) -> None:
        monkeypatch.delenv("TRTLLM_DWDP_PREFETCH_GATHER", raising=False)
        self.layers = layers
        self.buffer = _SlotBuffer(layers, num_slots)
        self.src: Dict[int, torch.Tensor] = {}
        for layer in layers:
            self.src[layer] = torch.randint(
                0, 64, (NUM_REMOTE, DIM), device="cuda", dtype=torch.int32
            ).to(DTYPE)
        self.wm = DWDPWeightManager(
            weight_buffer=self.buffer,
            peer_views={(1, layer, "w"): self.src[layer] for layer in layers},
            peer_ranges=[(0, NUM_REMOTE), (NUM_REMOTE, 2 * NUM_REMOTE)],
            moe_layer_indices=layers,
            weight_names=["w"],
            dwdp_rank=0,
            dwdp_size=2,
            cross_step_prefetch=True,
        )
        self.mgr = object.__new__(DwdpManager)
        self.mgr._weight_manager = self.wm
        self.mgr._cross_step_prefetch = True
        self.backend = _Backend()
        self.compute_stream = torch.cuda.Stream()
        self.copies: List[int] = []
        self.results: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def snapshot(self) -> Dict[int, torch.Tensor]:
        return {layer: t.clone() for layer, t in self.src.items()}

    def run_pass(
        self,
        tokens: int,
        moe_delay: int,
        expected: Dict[int, torch.Tensor],
        start: bool = True,
        abort_after_bind: Optional[int] = None,
    ) -> None:
        x = torch.randint(0, 4, (tokens, DIM), device="cuda", dtype=torch.int32).to(DTYPE)
        if start:
            self.mgr.prefetch_first_layers()
        self.compute_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.compute_stream):
            for layer in self.layers:
                self.mgr.wait_and_bind(self.backend, layer)
                if moe_delay:
                    torch.cuda._sleep(moe_delay)
                out = x @ self.backend.w[NUM_REMOTE:].T
                self.results.append((layer, x, out, expected[layer]))
                if layer == abort_after_bind:
                    raise _AbortPass()
                self.mgr.record_compute_and_prefetch_next(layer)
        torch.cuda.current_stream().wait_stream(self.compute_stream)

    def mismatches(self) -> List[int]:
        torch.cuda.synchronize()
        bad = [
            layer for layer, x, out, weights in self.results if not torch.equal(out, x @ weights.T)
        ]
        self.results.clear()
        return bad


def _logged_copy(delay: int, log: List[int]):
    """Per-slice copy that first stalls the copy stream for ``delay`` cycles and logs the layer."""
    original = DWDPWeightManager._prefetch_layer_per_slice

    def copy(self, layer_idx):
        if delay:
            torch.cuda._sleep(delay)
        log.append(layer_idx)
        original(self, layer_idx)

    return copy


@pytest.mark.parametrize("num_layers", [4, 5])
@pytest.mark.parametrize("num_slots", [2, 3])
@pytest.mark.parametrize("slow", ["moe", "copy"])
def test_bound_weights_across_passes(monkeypatch, num_layers, num_slots, slow):
    layers = [2 * i + 1 for i in range(num_layers)]  # non-contiguous decoder indices
    h = _Harness(layers, num_slots, monkeypatch)
    moe_delay = DELAY_CYCLES if slow == "moe" else 0
    copy_delay = DELAY_CYCLES if slow == "copy" else 0
    weights = h.snapshot()
    copy = _logged_copy(copy_delay, h.copies)
    with patch.object(DWDPWeightManager, "_prefetch_layer_per_slice", copy):
        # Warmup forward that never calls prefetch_first_layers.
        h.run_pass(5, moe_delay, weights, start=False)
        assert h.mismatches() == []
        assert h.copies == layers + layers[: min(num_slots, num_layers)]

        # Steps with different token counts, idle gaps and unrelated GPU work in between.
        for step, tokens in enumerate([16, 3, 64, 1]):
            del h.copies[:]
            h.run_pass(tokens, moe_delay, weights)
            assert h.mismatches() == [], f"step {step}"
            # The first layers came from the previous pass; this pass prefetches the next one's.
            assert sorted(h.copies) == sorted(layers), f"step {step}"
            time.sleep(0.01)
            torch.cuda._sleep(DELAY_CYCLES)

        # A pass that raises inside MoE(layer 3): the next pass restarts the sequence.
        abort_layer = layers[2]
        with pytest.raises(_AbortPass):
            h.run_pass(8, moe_delay, weights, abort_after_bind=abort_layer)
        torch.cuda.current_stream().wait_stream(h.compute_stream)
        assert h.mismatches() == []
        h.run_pass(8, moe_delay, weights)
        assert h.mismatches() == []

        # Weight update between steps (e.g. update_weights): the prefetches issued by the last
        # pass hold the old weights until invalidate_prefetch drops them.
        torch.cuda.synchronize()
        for t in h.src.values():
            t.add_(1000.0)
        weights = h.snapshot()
        h.mgr.invalidate_prefetch()
        h.run_pass(7, moe_delay, weights)
        assert h.mismatches() == []
        h.run_pass(2, moe_delay, weights)
        assert h.mismatches() == []


def test_without_invalidate_first_layers_are_stale(monkeypatch):
    """The prefetch issued at the end of a pass holds the weights of that moment."""
    layers = [1, 2, 3, 4]
    h = _Harness(layers, 2, monkeypatch)
    weights = h.snapshot()
    h.run_pass(4, 0, weights)
    assert h.mismatches() == []
    torch.cuda.synchronize()
    for t in h.src.values():
        t.add_(1000.0)
    h.run_pass(4, 0, h.snapshot())
    assert h.mismatches() == layers[:2]


def test_missing_war_wait_is_detected(monkeypatch):
    """Sensitivity check: without the consume events the next copies overwrite slots in use."""
    layers = [1, 2, 3, 4, 5, 6]
    h = _Harness(layers, 2, monkeypatch)
    weights = h.snapshot()

    def finish_without_consume(self, layer_idx):
        order = self._order_of(layer_idx)
        self._enqueue_scheduled(self._schedule.consume(order), abandoned_read=False)

    with patch.object(DWDPWeightManager, "finish_layer", finish_without_consume):
        for _ in range(3):
            h.run_pass(4, DELAY_CYCLES, weights)
    assert h.mismatches() != []
