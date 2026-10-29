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
"""One-GPU check of the DWDP expert prefetch protocol on real CUDA streams.

The real ``DWDPWeightManager`` (its per-slice and SM-gather transports) and the real
``DwdpManager`` protocol methods drive two-slot buffers that a small stand-in weight buffer
keeps in ordinary device memory; the "peers" are per-layer source tensors on the same GPU.
Each MoE layer's "kernel" clones the bound full tensor and multiplies it on the compute stream.
Copies or compute are slowed with spin kernels at random, and the host never synchronizes
between forwards, so a missing WAR or RAW dependency, or a skipped copy whose layer is not in
its slot, shows up as another layer's bytes. Every protocol (two-layer, cross-forward wrap,
legacy one-layer) must reproduce the replicated reference bit for bit.
"""

import os
import random
from typing import Dict, List, Tuple
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LAYERS = [1, 2, 4, 5, 7, 8]  # six MoE layers: slots 0, 1, 0, 1, 0, 1
NUM_EXPERTS = 8
ROWS, COLS = 256, 512  # 256 KiB of bf16 per expert, 2 MiB per layer
FORWARDS = 4
SPIN_CYCLES = 2_000_000  # GPU clock cycles of the spin kernel


class _DeviceWeightBuffer:
    """The ``WeightBuffer`` calls the weight manager makes, over two plain device slots."""

    def __init__(self, layers: List[int], device: torch.device) -> None:
        self.device_id = device.index
        self.local_start = 0
        self.local_end = NUM_EXPERTS // 2
        self._order = {layer: pos for pos, layer in enumerate(layers)}
        self.slots = [
            torch.full((NUM_EXPERTS, ROWS, COLS), float("nan"), dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]

    def buffer_index_for_layer(self, layer_idx: int) -> int:
        return self._order[layer_idx] % 2

    def get_remote_slices(self, layer_idx: int, name: str) -> List[Tuple[torch.Tensor, int, int]]:
        slot = self.slots[self.buffer_index_for_layer(layer_idx)]
        return [(slot, 0, NUM_EXPERTS)]  # every expert is copied, the local half included

    def get_full_tensor(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.slots[self.buffer_index_for_layer(layer_idx)]


class _Backend:
    """The MoE backend attribute ``wait_and_bind`` rebinds."""

    def __init__(self) -> None:
        self.w = torch.nn.Parameter(torch.empty(0), requires_grad=False)


def _sources(device: torch.device) -> Dict[int, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(1234)
    return {
        layer: torch.randn((NUM_EXPERTS, ROWS, COLS), generator=gen).to(torch.bfloat16).to(device)
        for layer in LAYERS
    }


def _activations(device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(99)
    return torch.randn((16, COLS), generator=gen).to(torch.bfloat16).to(device)


def _run(protocol: str, transport: str, delays: str, seed: int) -> Tuple[list, list, int]:
    """Run FORWARDS forwards; return (per-layer clones, per-layer products, copies enqueued)."""
    device = torch.device("cuda", torch.cuda.current_device())
    sources = _sources(device)
    half = NUM_EXPERTS // 2
    peer_views = {}
    for layer, src in sources.items():
        peer_views[(0, layer, "w")] = src[:half]
        peer_views[(1, layer, "w")] = src[half:]
    weight_buffer = _DeviceWeightBuffer(LAYERS, device)
    rng = random.Random(seed)
    copies = []

    real_slice = DWDPWeightManager._prefetch_layer_per_slice
    real_gather = DWDPWeightManager._prefetch_layer_gather

    def spin(kind: str) -> None:
        if delays == kind or (delays == "random" and rng.random() < 0.5):
            torch.cuda._sleep(rng.randint(SPIN_CYCLES // 4, SPIN_CYCLES))

    def slow_slice(self, layer_idx):
        copies.append(layer_idx)
        spin("copy")  # runs on the copy stream: prefetch_layer holds it current
        real_slice(self, layer_idx)

    def slow_gather(self, layer_idx):
        copies.append(layer_idx)
        spin("copy")
        real_gather(self, layer_idx)

    env = {"TRTLLM_DWDP_PREFETCH_GATHER": "1" if transport == "gather" else "0"}
    if transport == "gather":
        env["TRTLLM_DWDP_GATHER_KERNEL"] = "tma"
    with (
        patch.dict(os.environ, env),
        patch.object(DWDPWeightManager, "_prefetch_layer_per_slice", slow_slice),
        patch.object(DWDPWeightManager, "_prefetch_layer_gather", slow_gather),
    ):
        weight_manager = DWDPWeightManager(
            weight_buffer=weight_buffer,
            peer_views=peer_views,
            peer_ranges=[(0, half), (half, NUM_EXPERTS)],
            moe_layer_indices=LAYERS,
            weight_names=["w"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        if transport == "gather" and not weight_manager.use_gather:
            pytest.skip("trtllm::dwdp_fabric_gather is not registered in this build")
        manager = DwdpManager.__new__(DwdpManager)  # the runtime methods only, no MPI group
        manager._weight_manager = weight_manager
        manager._dense = None
        manager._legacy_prefetch = protocol == "legacy"
        manager._wrap_prefetch = protocol == "wrap"
        manager._expert_forward_open = False

        backend = _Backend()
        x = _activations(device)
        clones, products = [], []
        for _ in range(FORWARDS):  # no host synchronization between forwards
            manager.prefetch_first_layers()
            for layer in LAYERS:
                manager.wait_and_bind(backend, layer)
                spin("compute")
                bound = backend.w.data
                clones.append(bound.clone())
                products.append(x @ bound.reshape(-1, COLS).t())
                manager.record_compute_and_prefetch_next(layer)
        torch.cuda.synchronize()
        weight_manager._released = True  # no VMM resources behind the stand-in buffer
    return clones, products, len(copies)


@pytest.mark.parametrize("delays", ["copy", "compute", "random"])
@pytest.mark.parametrize("transport", ["per_slice", "gather"])
@pytest.mark.parametrize("protocol", ["two_layer", "wrap", "legacy"])
def test_prefetch_protocol_is_bitwise_exact(protocol: str, transport: str, delays: str) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    sources = _sources(device)
    x = _activations(device)
    clones, products, copies = _run(protocol, transport, delays, seed=7)

    for i, (clone, product) in enumerate(zip(clones, products)):
        layer = LAYERS[i % len(LAYERS)]
        where = f"forward {i // len(LAYERS)} layer {layer}"
        # What the layer's kernel read is the layer's weights, byte for byte ...
        assert torch.equal(clone, sources[layer]), where
        # ... and its product equals the replicated reference computed on the default stream.
        assert torch.equal(product, x @ sources[layer].reshape(-1, COLS).t()), where

    if protocol == "wrap":
        # Each forward gathers each MoE layer once; the last one also gathers the next
        # forward's first two layers.
        assert copies == FORWARDS * len(LAYERS) + 2
    else:
        # The legacy protocol no longer gathers the second MoE layer twice.
        assert copies == FORWARDS * len(LAYERS)


def test_gather_transport_matches_the_copy_engine_transport() -> None:
    _, products_slice, _ = _run("wrap", "per_slice", "none", seed=0)
    _, products_gather, _ = _run("wrap", "gather", "none", seed=0)
    for got, want in zip(products_gather, products_slice):
        assert torch.equal(got, want)


def test_fabric_gather_is_enabled_reads_env() -> None:
    with patch.dict(os.environ, {"TRTLLM_DWDP_PREFETCH_GATHER": "0"}):
        assert not fabric_gather.is_enabled()
