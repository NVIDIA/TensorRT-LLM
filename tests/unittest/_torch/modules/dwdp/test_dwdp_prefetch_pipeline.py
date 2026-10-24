# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU-mocked tests for the DWDP double-buffer event protocol."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager

pytestmark = pytest.mark.cpu_only


class TestDWDPPrefetchPipeline:
    def setup_method(self):
        self.weight_buffer = MagicMock()
        self.weight_buffer.device_id = 0
        self.weight_buffer.local_start = 0
        self.weight_buffer.local_end = 2
        self.weight_buffer.buffer_index_for_layer.side_effect = {
            3: 0,
            5: 1,
            7: 0,
        }.__getitem__

        self.copy_stream = MagicMock(name="copy_stream")
        self.compute_stream = MagicMock(name="compute_stream")
        self.events = [
            MagicMock(name="prefetch_0"),
            MagicMock(name="prefetch_1"),
            MagicMock(name="consume_0"),
            MagicMock(name="consume_1"),
        ]
        self.patches = [
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Stream",
                return_value=self.copy_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Event",
                side_effect=self.events,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.current_stream",
                return_value=self.compute_stream,
            ),
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            # The per-slice transport, whatever TRTLLM_DWDP_PREFETCH_GATHER the environment holds.
            patch(
                "tensorrt_llm._torch.modules.dwdp.weight_manager.fabric_gather.is_enabled",
                return_value=False,
            ),
        ]
        for cuda_patch in self.patches:
            cuda_patch.start()

        self.manager = DWDPWeightManager(
            weight_buffer=self.weight_buffer,
            peer_views={},
            peer_ranges=[(0, 2), (2, 4)],
            moe_layer_indices=[3, 5, 7],
            weight_names=["gate_up_proj", "down_proj"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        for event in self.events:
            event.reset_mock()

    def teardown_method(self):
        self.manager._released = True
        for cuda_patch in reversed(self.patches):
            cuda_patch.stop()

    def test_wait_and_bind_waits_for_copy_without_signalling_consumption(self):
        full_tensors = {
            "gate_up_proj": torch.tensor([1.0]),
            "down_proj": torch.tensor([2.0]),
        }
        self.weight_buffer.get_full_tensor.side_effect = lambda _layer_idx, name: full_tensors[name]
        backend = nn.Module()
        backend.gate_up_proj = nn.Parameter(torch.empty(1), requires_grad=False)
        backend.down_proj = nn.Parameter(torch.empty(1), requires_grad=False)

        self.manager.wait_and_bind(backend, 3)

        self.compute_stream.wait_event.assert_called_once_with(self.events[0])
        self.events[2].record.assert_not_called()
        assert backend.gate_up_proj.data.data_ptr() == full_tensors["gate_up_proj"].data_ptr()
        assert backend.down_proj.data.data_ptr() == full_tensors["down_proj"].data_ptr()

    def test_record_compute_signals_the_current_slot(self):
        self.manager.record_compute(3)

        self.events[2].record.assert_called_once_with(self.compute_stream)
        self.events[3].record.assert_not_called()

    def test_record_other_slot_consumed_signals_the_other_slot(self):
        # TRTLLM_DWDP_LEGACY_PREFETCH: layer 3 uses slot 0, so slot 1 is released.
        self.manager.record_other_slot_consumed(3)

        self.events[3].record.assert_called_once_with(self.compute_stream)
        self.events[2].record.assert_not_called()

    def test_prefetch_layer_skips_a_resident_layer(self):
        self.manager.prefetch_layer(3)
        self.manager.prefetch_layer(3)

        # One WAR wait and one RAW record: the second call found layer 3 in slot 0.
        self.copy_stream.wait_event.assert_called_once_with(self.events[2])
        self.events[0].record.assert_called_once_with(self.copy_stream)
        assert self.manager.resident_layer(0) == 3
        assert self.manager.resident_layer(1) is None

    def test_prefetch_layer_replaces_another_layer_in_the_slot(self):
        self.manager.prefetch_layer(3)
        self.manager.prefetch_layer(7)  # same slot as layer 3

        assert self.copy_stream.wait_event.call_count == 2
        assert self.events[0].record.call_count == 2
        assert self.manager.resident_layer(0) == 7

    def test_failed_enqueue_is_not_trusted(self):
        with patch.object(
            DWDPWeightManager, "_prefetch_layer_per_slice", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                self.manager.prefetch_layer(3)
        assert self.manager.resident_layer(0) is None

        self.manager.prefetch_layer(3)
        assert self.manager.resident_layer(0) == 3
        self.events[0].record.assert_called_once_with(self.copy_stream)


# ----------------------------------------------------------------------------------------------
# Protocol check: the real DWDPWeightManager and DwdpManager protocol on simulated CUDA streams.
# ----------------------------------------------------------------------------------------------


class _SimEvent:
    """A CUDA event: ``record`` captures the stream's enqueue position; a wait captures the latest."""

    def __init__(self, sim: "_StreamSim") -> None:
        self.sim = sim
        self.point = None  # (stream, ops enqueued before the record) of the latest record

    def record(self, stream=None) -> None:
        stream = stream if stream is not None else self.sim.current
        self.point = (stream, len(self.sim.queues[stream]))


class _SimStream:
    def __init__(self, sim: "_StreamSim", name: str) -> None:
        self.sim = sim
        self.name = name

    def wait_event(self, event: _SimEvent) -> None:
        self.sim.enqueue(self, ("wait", event.point))


class _StreamSim:
    """In-order streams with CUDA event semantics, run in a random or greedy interleaving.

    Ops are ("wait", point), ("copy", slot, layer) and ("compute", slot, layer). A copy is a
    WAR violation if a compute enqueued before it (host order) still has to read the layer the
    slot holds; a compute is a RAW violation if its slot does not hold its layer.
    """

    def __init__(self) -> None:
        self.queues = {}
        self.done = {}
        self.seq = 0
        self.current = None

    def stream(self, name: str) -> _SimStream:
        stream = _SimStream(self, name)
        self.queues[stream] = []
        self.done[stream] = 0
        return stream

    def enqueue(self, stream: _SimStream, op: tuple) -> None:
        self.seq += 1
        self.queues[stream].append((self.seq, op))

    def _ready(self, stream: _SimStream) -> bool:
        queue, i = self.queues[stream], self.done[stream]
        if i >= len(queue):
            return False
        op = queue[i][1]
        if op[0] != "wait" or op[1] is None:
            return True
        source, count = op[1]
        return self.done[source] >= count

    def run(self, pick) -> list:
        """Execute every op; returns the violations found."""
        slots = {}
        violations = []
        while any(self.done[s] < len(q) for s, q in self.queues.items()):
            ready = [s for s in self.queues if self._ready(s)]
            assert ready, "deadlock: every stream waits"
            stream = pick(ready)
            seq, op = self.queues[stream][self.done[stream]]
            self.done[stream] += 1
            if op[0] == "copy":
                _, slot, layer = op
                held = slots.get(slot)
                for other, queue in self.queues.items():
                    for other_seq, other_op in queue[self.done[other] :]:
                        if (
                            other_op[0] == "compute"
                            and other_op[1] == slot
                            and other_op[2] == held
                            and other_seq < seq
                            and held != layer
                        ):
                            violations.append(f"WAR: copy {layer} over {held} in slot {slot}")
                slots[slot] = layer
            elif op[0] == "compute":
                _, slot, layer = op
                if slots.get(slot) != layer:
                    violations.append(f"RAW: compute {layer} read {slots.get(slot)} in slot {slot}")
        return violations


def _run_protocol(layers, protocol, forwards, pick_seed, abort=None):
    """Drive ``forwards`` forwards through the real manager methods; return (copies, violations).

    ``abort=(forward, position, phase)`` stops that forward at its MoE ``position``: "bind" after
    ``wait_and_bind``, "kernel" after the MoE kernel is enqueued (before ``record_compute``), as an
    exception in the model would. Its kernels' results are discarded, but they still read the slots.
    """
    import random

    from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

    sim = _StreamSim()
    copy_stream = sim.stream("copy")
    compute_stream = sim.stream("compute")
    sim.current = compute_stream
    slot_of = {layer: pos % 2 for pos, layer in enumerate(layers)}
    weight_buffer = MagicMock()
    weight_buffer.device_id = 0
    weight_buffer.local_start = 0
    weight_buffer.local_end = 2
    weight_buffer.buffer_index_for_layer.side_effect = slot_of.__getitem__
    copies = []

    def fake_copy(self, layer_idx):
        copies.append(layer_idx)
        sim.enqueue(copy_stream, ("copy", slot_of[layer_idx], layer_idx))

    patches = [
        patch(
            "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Stream",
            return_value=copy_stream,
        ),
        patch(
            "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.Event",
            side_effect=lambda *a, **k: _SimEvent(sim),
        ),
        patch(
            "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.current_stream",
            return_value=compute_stream,
        ),
        patch(
            "tensorrt_llm._torch.modules.dwdp.weight_manager.torch.cuda.stream",
            return_value=nullcontext(),
        ),
        patch.object(DWDPWeightManager, "_prefetch_layer_per_slice", fake_copy),
        # The per-slice transport, whatever TRTLLM_DWDP_PREFETCH_GATHER the environment holds.
        patch(
            "tensorrt_llm._torch.modules.dwdp.weight_manager.fabric_gather.is_enabled",
            return_value=False,
        ),
    ]
    for p in patches:
        p.start()
    try:
        weight_manager = DWDPWeightManager(
            weight_buffer=weight_buffer,
            peer_views={},
            peer_ranges=[(0, 2), (2, 4)],
            moe_layer_indices=list(layers),
            weight_names=["gate_up_proj"],
            dwdp_rank=0,
            dwdp_size=2,
        )
        manager = DwdpManager.__new__(DwdpManager)  # the runtime methods only, no MPI group
        manager._weight_manager = weight_manager
        manager._dense = None
        manager._legacy_prefetch = protocol == "legacy"
        manager._wrap_prefetch = protocol == "wrap"
        manager._expert_forward_open = False
        per_forward = []
        for forward in range(forwards):
            start = len(copies)
            manager.prefetch_first_layers()
            for position, layer in enumerate(layers):
                stop = abort is not None and abort[:2] == (forward, position)
                manager.wait_and_bind(MagicMock(), layer)
                if stop and abort[2] == "bind":
                    break
                sim.enqueue(compute_stream, ("compute", slot_of[layer], layer))
                if stop:
                    break
                manager.record_compute_and_prefetch_next(layer)
            per_forward.append(copies[start:])
        weight_manager._released = True
    finally:
        for p in reversed(patches):
            p.stop()

    rng = random.Random(pick_seed)
    if pick_seed == -1:  # copies run as early as possible
        pick = lambda ready: copy_stream if copy_stream in ready else ready[0]  # noqa: E731
    elif pick_seed == -2:  # compute runs as early as possible
        pick = lambda ready: compute_stream if compute_stream in ready else ready[0]  # noqa: E731
    else:
        pick = rng.choice
    return per_forward, sim.run(pick)


_LAYER_SETS = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9], [3, 5, 7, 9, 11], [2, 3, 5, 8, 13, 21]]


@pytest.mark.parametrize("layers", _LAYER_SETS, ids=lambda layer_set: f"{len(layer_set)}moe")
@pytest.mark.parametrize("protocol", ["two_layer", "wrap", "legacy"])
def test_protocol_has_no_hazard_in_any_interleaving(layers, protocol):
    for seed in [-1, -2, *range(40)]:
        _, violations = _run_protocol(layers, protocol, forwards=3, pick_seed=seed)
        assert violations == [], f"seed {seed}: {violations}"


@pytest.mark.parametrize("protocol", ["two_layer", "wrap", "legacy"])
def test_protocol_gathers_each_layer_once_per_forward(protocol):
    layers = [3, 5, 7, 9]
    per_forward, _ = _run_protocol(layers, protocol, forwards=3, pick_seed=0)
    if protocol == "wrap":
        # The first forward also gathers the next forward's first two layers; later forwards
        # find them resident, so every forward gathers each MoE layer once.
        assert per_forward == [[3, 5, 7, 9, 3, 5], [7, 9, 3, 5], [7, 9, 3, 5]]
    else:
        # The legacy protocol used to gather layer 5 twice: [3, 5, 5, 7, 9].
        assert per_forward == [[3, 5, 7, 9]] * 3


@pytest.mark.parametrize("phase", ["bind", "kernel"])
@pytest.mark.parametrize("protocol", ["two_layer", "wrap", "legacy"])
def test_aborted_forward_leaves_no_hazard(protocol, phase):
    # A forward that stops between its first and last MoE layer leaves a kernel whose slot was
    # never released; the next priming records a WAR barrier on both slots before copying.
    for layers in ([3, 5], [3, 5, 7], [3, 5, 7, 9], [2, 3, 5, 8, 13, 21]):
        for position in range(len(layers)):
            for seed in [-1, -2, *range(12)]:
                _, violations = _run_protocol(
                    layers, protocol, forwards=3, pick_seed=seed, abort=(0, position, phase)
                )
                assert violations == [], f"{layers} abort at {position}, seed {seed}: {violations}"


def test_wrap_needs_an_even_moe_layer_count():
    # No wrap with 3 MoE layers (slots 0, 1, 0). Layer 5 is the only slot-1 layer, so it stays
    # resident and the next forward's priming skips it.
    per_forward, _ = _run_protocol([3, 5, 7], "wrap", forwards=2, pick_seed=0)
    assert per_forward == [[3, 5, 7], [3, 7]]
