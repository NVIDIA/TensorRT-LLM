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
"""DWDP expert prefetch on real streams, events and copies.

A stand-in for ``WeightBuffer`` backs every layer's remote experts with its slot's storage, and
the peers' experts are plain tensors (on this GPU, or on the other GPUs of the node for the
``trtllm::dwdp_fabric_gather`` kernel).  Each "MoE" multiplies activations of a per-pass token
count with the bound remote experts after a delay; delaying the MoE exposes a missing WAR wait
(the next copy overwrites the slot first), delaying the copies exposes a missing RAW wait.  Every
output is checked against the source weights of its pass.
"""

import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_EXPERTS = 8  # experts per rank
DIM = 4096
DTYPE = torch.float64
DELAY_CYCLES = 400_000


class _SlotBuffer:
    """Stand-in for ``WeightBuffer`` of DWDP rank 0: rows [0, NUM_EXPERTS) of a layer's full
    tensor play the (unused) local experts, the other rows are its slot's remote experts."""

    local_start = 0
    local_end = NUM_EXPERTS

    def __init__(self, layers: List[int], num_slots: int, num_ranks: int) -> None:
        self.device_id = torch.cuda.current_device()
        self.num_slots = num_slots
        self._layers = layers
        self.slots = [
            torch.full((num_ranks * NUM_EXPERTS, DIM), -1.0, dtype=DTYPE, device="cuda")
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
        self.w = torch.nn.Parameter(torch.empty(0, dtype=DTYPE, device="cuda"), requires_grad=False)


class _AbortPass(Exception):
    pass


class _Harness:
    """Drives the facade the way the executor loop and ``ConfigurableMoE`` do."""

    def __init__(
        self, layers: List[int], num_slots: int, peer_devices: Optional[List[int]] = None
    ) -> None:
        self.layers = layers
        peers = [torch.cuda.current_device()] if peer_devices is None else peer_devices
        self.num_ranks = len(peers) + 1
        self.buffer = _SlotBuffer(layers, num_slots, self.num_ranks)
        # src[layer][r - 1]: the experts of DWDP rank r (r >= 1) on its device.
        self.src: Dict[int, List[torch.Tensor]] = {}
        for layer in layers:
            self.src[layer] = [
                torch.randint(0, 64, (NUM_EXPERTS, DIM), device=f"cuda:{dev}", dtype=torch.int32)
                .to(DTYPE)
                .contiguous()
                for dev in peers
            ]
        peer_views = {
            (rank, layer, "w"): self.src[layer][rank - 1]
            for layer in layers
            for rank in range(1, self.num_ranks)
        }
        self.wm = DWDPWeightManager(
            weight_buffer=self.buffer,
            peer_views=peer_views,
            peer_ranges=[
                (rank * NUM_EXPERTS, (rank + 1) * NUM_EXPERTS) for rank in range(self.num_ranks)
            ],
            moe_layer_indices=layers,
            weight_names=["w"],
            dwdp_rank=0,
            dwdp_size=self.num_ranks,
        )
        self.mgr = object.__new__(DwdpManager)
        self.mgr._weight_manager = self.wm
        self.backend = _Backend()
        self.compute_stream = torch.cuda.Stream()
        self.copies: List[int] = []
        self.results: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def snapshot(self) -> Dict[int, torch.Tensor]:
        """Remote experts of every layer as rank 0 must see them, on this GPU."""
        return {
            layer: torch.cat([t.to(torch.cuda.current_device()) for t in tensors])
            for layer, tensors in self.src.items()
        }

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
            self.mgr.start_forward()
        self.compute_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.compute_stream):
            for layer in self.layers:
                self.mgr.wait_and_bind(self.backend, layer)
                if moe_delay:
                    torch.cuda._sleep(moe_delay)
                out = x @ self.backend.w[NUM_EXPERTS:].T
                self.results.append((layer, x, out, expected[layer]))
                if layer == abort_after_bind:
                    raise _AbortPass()
                self.mgr.finish_layer(layer)
        torch.cuda.current_stream().wait_stream(self.compute_stream)

    def mismatches(self) -> List[int]:
        torch.cuda.synchronize()
        bad = [
            layer for layer, x, out, weights in self.results if not torch.equal(out, x @ weights.T)
        ]
        self.results.clear()
        return bad


def _logged(method_name: str, delay: int, log: List[int]):
    """Wrap a copy method: first stall the copy stream for ``delay`` cycles, log the layer."""
    original = getattr(DWDPWeightManager, method_name)

    def copy(self, layer_idx):
        if delay:
            torch.cuda._sleep(delay)
        log.append(layer_idx)
        original(self, layer_idx)

    return copy


def _drive(h: _Harness, moe_delay: int, num_slots: int) -> None:
    """Warmup pass, steps of varying size with idle gaps, an aborted pass and a weight update."""
    layers = h.layers
    weights = h.snapshot()
    # Warmup forward that never calls start_forward; its last layers already gather the
    # first layers of the next pass.
    h.run_pass(5, moe_delay, weights, start=False)
    assert h.mismatches() == []
    assert h.copies[: len(layers)] == layers
    assert h.copies[len(layers) :] == layers[: min(num_slots, len(layers))]

    # Steps with different token counts, idle gaps and unrelated GPU work in between.
    for step, tokens in enumerate([16, 3, 64, 1]):
        del h.copies[:]
        h.run_pass(tokens, moe_delay, weights)
        assert h.mismatches() == [], f"step {step}"
        # Every layer is gathered exactly once per pass.
        assert sorted(h.copies) == sorted(layers), f"step {step}"
        time.sleep(0.01)
        torch.cuda._sleep(DELAY_CYCLES)

    # A pass that raises inside MoE(layer 3): the next pass restarts the sequence.
    with pytest.raises(_AbortPass):
        h.run_pass(8, moe_delay, weights, abort_after_bind=layers[2])
    torch.cuda.current_stream().wait_stream(h.compute_stream)
    assert h.mismatches() == []
    h.run_pass(8, moe_delay, weights)
    assert h.mismatches() == []

    # Weight update between steps (e.g. update_weights): the copies issued by the last pass hold
    # the old weights until invalidate_prefetch drops them.
    torch.cuda.synchronize()
    for tensors in h.src.values():
        for t in tensors:
            t.add_(1000.0)
    weights = h.snapshot()
    h.mgr.invalidate_prefetch()
    h.run_pass(7, moe_delay, weights)
    assert h.mismatches() == []
    h.run_pass(2, moe_delay, weights)
    assert h.mismatches() == []


@pytest.mark.parametrize("num_layers", [4, 5])
@pytest.mark.parametrize("num_slots", [2, 3])
@pytest.mark.parametrize("slow", ["moe", "copy"])
def test_bound_weights_across_passes(monkeypatch, num_layers, num_slots, slow):
    monkeypatch.delenv("TRTLLM_DWDP_PREFETCH_GATHER", raising=False)
    layers = [2 * i + 1 for i in range(num_layers)]  # non-contiguous decoder indices
    h = _Harness(layers, num_slots)
    moe_delay = DELAY_CYCLES if slow == "moe" else 0
    copy_delay = DELAY_CYCLES if slow == "copy" else 0
    copy = _logged("_prefetch_layer_per_slice", copy_delay, h.copies)
    with patch.object(DWDPWeightManager, "_prefetch_layer_per_slice", copy):
        _drive(h, moe_delay, num_slots)


def test_without_invalidate_first_layers_are_stale(monkeypatch):
    """The prefetch issued at the end of a pass holds the weights of that moment."""
    monkeypatch.delenv("TRTLLM_DWDP_PREFETCH_GATHER", raising=False)
    layers = [1, 2, 3, 4]
    h = _Harness(layers, 2)
    weights = h.snapshot()
    h.run_pass(4, 0, weights)
    assert h.mismatches() == []
    torch.cuda.synchronize()
    for tensors in h.src.values():
        for t in tensors:
            t.add_(1000.0)
    h.run_pass(4, 0, h.snapshot())
    assert h.mismatches() == layers[:2]


def test_missing_war_wait_is_detected(monkeypatch):
    """Sensitivity check: without the consume events the next copies overwrite slots in use."""
    monkeypatch.delenv("TRTLLM_DWDP_PREFETCH_GATHER", raising=False)
    layers = [1, 2, 3, 4, 5, 6]
    h = _Harness(layers, 2)
    weights = h.snapshot()
    original = DWDPWeightManager._apply

    def without_consume(self, step):
        original(self, step._replace(release=None))

    with patch.object(DWDPWeightManager, "_apply", without_consume):
        for _ in range(3):
            h.run_pass(4, DELAY_CYCLES, weights)
    assert h.mismatches() != []


def _enable_peer_access(devices: List[int]) -> None:
    from cuda.bindings import runtime as cudart

    current = torch.cuda.current_device()
    for dev in devices:
        if not torch.cuda.can_device_access_peer(current, dev):
            pytest.skip(f"GPU {current} cannot access GPU {dev}")
        (err,) = cudart.cudaDeviceEnablePeerAccess(dev, 0)
        if err not in (
            cudart.cudaError_t.cudaSuccess,
            cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
        ):
            pytest.skip(f"cudaDeviceEnablePeerAccess({dev}) failed: {err}")


@pytest.mark.parametrize("num_slots", [2, 3])
@pytest.mark.parametrize("slow", ["moe", "copy"])
def test_fabric_gather_from_peer_gpus(monkeypatch, num_slots, slow):
    """The production transport: the SM gather kernel pulls the remote experts from peer GPUs."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_GATHER", "1")
    # The gather sets these for the CuTeDSL kernels unless pinned; pin them to keep the env clean.
    monkeypatch.setenv("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", "clc_dynamic")
    monkeypatch.setenv("TLLM_ROUTING_DISABLE_COOP_LAUNCH", "1")
    if fabric_gather.gather_op() is None:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered")
    torch.cuda.set_device(0)
    peers = list(range(1, min(torch.cuda.device_count(), 4)))
    _enable_peer_access(peers)
    layers = [2 * i + 1 for i in range(5)]
    h = _Harness(layers, num_slots, peer_devices=peers)
    assert h.wm._use_gather
    moe_delay = DELAY_CYCLES if slow == "moe" else 0
    copy = _logged("_prefetch_layer_gather", DELAY_CYCLES if slow == "copy" else 0, h.copies)
    with patch.object(DWDPWeightManager, "_prefetch_layer_gather", copy):
        _drive(h, moe_delay, num_slots)
