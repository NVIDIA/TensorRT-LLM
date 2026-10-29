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
"""CPU tests of the dense DWDP setup.

Every rank of an ``N``-rank DWDP group is a thread that runs ``setup_dense_dwdp`` on its own copy
of a Kimi-K3-like toy model.  The threads gather over ``StoreDwdpGroup`` on one in-process
``HashStore`` with every wait bounded, so a rank that skips or adds a collective fails the test
instead of hanging it.  Fakes stand in for the device: ``FakeOps`` for the stream and event seam,
``FakeStore`` keeps one rank's rows in host memory and exports their addresses (``LoopbackStore``
runs the real loopback store instead), ``FakeLauncher`` runs the real copy plans as host
memmoves, so each rank's slots are reassembled from every rank's rows the way the gather
reassembles them, and ``FakeMemInfo`` models a caching allocator in front of the device.  Every
fake appends what setup asked of it to its rank's log, so the tests check the order of the steps
as well as their outcome.  Faults are injected in each phase on each rank.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import re
import sys
import threading
import traceback
import weakref
from collections import Counter
from dataclasses import dataclass, replace
from typing import NamedTuple

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import repoint as dense_repoint
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout, row_pieces
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.setup import (
    SETUP_HEADROOM_BYTES,
    PhaseReport,
    dense_roots_available,
    setup_dense_dwdp,
)
from tensorrt_llm._torch.modules.dwdp.dense.store import DenseShardStore, LoopbackShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    EMPTY_TAG,
    RING_DEPTH,
    ROW_ALIGN,
    CopyMode,
    DenseProtocolError,
    DenseRepointError,
    DenseSetupError,
    SegmentKind,
)
from tensorrt_llm._torch.modules.dwdp.rendezvous import StoreDwdpGroup
from tensorrt_llm._torch.modules.dwdp.weight_manager import DWDPWeightManager

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

#: Seconds one collective waits for the slowest rank; a rank that skips a collective times out.
COLLECTIVE_TIMEOUT_S = 20
#: Seconds a test waits for its rank threads.
JOIN_TIMEOUT_S = 90

#: Free device memory every rank reports, and what setup frees by its commit.
PLENTY_FREE_BYTES = 64 << 30
FREED_BYTES = 53_070_000_000
TOTAL_BYTES = 200 << 30
#: Free blocks a rank's caching allocator holds before setup, in the tests that model them.
CACHED_BYTES = 1 << 30

# ---------------------------------------------------------------------------
# Toy Kimi-K3 decoder (patterns of test_dense_layout.py, smaller)
# ---------------------------------------------------------------------------

#: The model tables.
ATTR_ALLOW = frozenset(
    {
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)
EXCLUDE = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)

#: Toy threshold: the moved toy storages are 4-8 KiB, the replicated ones at most 1 KiB.
TOY_MIN_BYTES = 4096
HIDDEN = 64
ROWS = 32
F_ROWS = 24
B_ROWS = 2
PAD_ROWS = 6
#: Decoder layer kinds: layer 0 stays replicated, layer 3 is an MLA layer.
LAYER_KINDS = ("dense", "kda", "kda", "mla", "kda", "kda")
COVERED = (1, 2, 3, 4, 5)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
FP32 = torch.float32

#: Paths, relative to a covered decoder layer, that move: every alias of a moved storage,
#: including excluded names, the empty view and the stride-0 view.
KDA_MOVED = (
    "linear_attn._bfa_proj_weight",
    "linear_attn._bfa_views[0]",
    "linear_attn._bfa_views[1]",
    "linear_attn._gate_b_t",
    "linear_attn._o_dense",
    "linear_attn.q_proj.weight",
    "linear_attn.k_proj.weight",
    "linear_attn.v_proj.weight",
    "linear_attn.g_proj.weight",
    "linear_attn.qkvg_proj.weight",
    "linear_attn.qkvg_proj.weight_scale",
    "linear_attn.o_proj.weight",
    "linear_attn.o_proj.weight_scale",
    "linear_attn.f_a_proj.weight",
    "linear_attn.b_proj.weight",
    "linear_attn.f_b_proj.weight",
)
MLA_MOVED = (
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.kv_b_proj.weight_scale",
    "self_attn.o_proj.weight",
    "self_attn.k_b_proj_trans",
    "self_attn.k_b_proj_trans_dequant",
    "self_attn.v_b_proj_scale",
)
MOE_MOVED = (
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.shared_experts.gate_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_up_proj.weight_scale",
    "block_sparse_moe.shared_experts.down_proj.weight",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
)
#: Path prefixes, relative to a decoder layer, of each segment's roots.
SEGMENT_ROOTS = {
    ATTN: ("linear_attn.", "self_attn."),
    MOE: tuple(
        f"block_sparse_moe.{name}."
        for name in (
            "gate",
            "shared_experts",
            "routed_expert_down_proj",
            "routed_expert_up_proj",
            "routed_expert_norm",
        )
    ),
}
#: The last MoE-dense anchor of the last covered layer; no other binding views its storage.
LAST_MOE_ANCHOR = f"layers.{COVERED[-1]}.block_sparse_moe.routed_expert_up_proj.weight"
#: The module whose ``weight`` buffer the ``lose_buffer`` fault deletes from the last layer.
LOST_MODULE = "block_sparse_moe.routed_expert_up_proj"


def _ints(generator, shape, dtype):
    """Small integers in [-3, 3], exactly representable in every moved dtype."""
    return torch.randint(-3, 4, shape, generator=generator).to(FP32).to(dtype)


def _bytes(generator, shape):
    return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)


def _parameter(tensor):
    return nn.Parameter(tensor, requires_grad=False)


class ToyFp8Linear(nn.Module):
    """Stands in for ``_Fp8BlockScaleWeightReadLinear``: weight and scale buffers."""

    def __init__(self, weight, scale):
        super().__init__()
        self._weights_transformed = True
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)


class ToyLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = _parameter(weight)


class ToyNorm(nn.Module):
    def __init__(self, generator, size):
        super().__init__()
        self.weight = _parameter(_ints(generator, (size,), BF16))


class ToyKDA(nn.Module):
    """KDA mixer after load: fused FP8 qkvg with row-view buffers, fused BF16 [f_a | b | pad]."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.A_log = _parameter(_ints(g, (16,), FP32))
        fused = _ints(g, (4 * ROWS, HIDDEN), FP8)
        for index, name in enumerate(("q_proj", "k_proj", "v_proj", "g_proj")):
            rows = fused[index * ROWS : (index + 1) * ROWS]
            setattr(self, name, ToyFp8Linear(rows, _bytes(g, (ROWS, 8))))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, 2 * ROWS), FP8), _bytes(g, (HIDDEN, 64)))
        self.f_a_proj = ToyLinear(_ints(g, (F_ROWS, HIDDEN), BF16))
        self.b_proj = ToyLinear(_ints(g, (B_ROWS, HIDDEN), BF16))
        self.f_b_proj = ToyLinear(_ints(g, (HIDDEN, 32), BF16))
        self.o_norm = ToyNorm(g, HIDDEN)
        # What the K3 loader's finalize step leaves behind.
        self.qkvg_proj = ToyFp8Linear(fused, _bytes(g, (4 * ROWS, 32)))
        pad = torch.zeros(PAD_ROWS, HIDDEN, dtype=BF16)
        bfa = torch.cat([self.f_a_proj.weight.data, self.b_proj.weight.data, pad])
        self.f_a_proj.weight.data = bfa[:F_ROWS]
        self.b_proj.weight.data = bfa[F_ROWS : F_ROWS + B_ROWS]
        self._bfa_proj_weight = bfa
        self._bfa_views = [bfa[:1].expand(4, -1), bfa[bfa.shape[0] :]]
        self._gate_b_t = self.f_b_proj.weight.data.view(2, HIDDEN // 2, 32).transpose(1, 2)
        self._mtp_conv_weights = tuple(_ints(g, (4, HIDDEN), BF16) for _ in range(3))
        self._packed_conv_weight = _ints(g, (3, HIDDEN), BF16)
        self._A_log_f32 = self.A_log.detach().float()
        self._o_dense = self.o_proj.weight[8:24]


class ToyAttentionBackend:
    """Not an ``nn.Module``, so the walk never enters it (like MLA's ``mha`` / ``mqa``)."""

    def __init__(self, generator):
        self.workspace = _ints(generator, (4096,), BF16)


class ToyMLA(nn.Module):
    """MLA attention after load, with the BF16 absorb copy and excluded scratch tensors."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.kv_a_proj_with_mqa = ToyFp8Linear(_ints(g, (64, HIDDEN), FP8), _bytes(g, (64, 16)))
        self.q_b_proj = ToyFp8Linear(_ints(g, (64, HIDDEN), FP8), _bytes(g, (64, 16)))
        self.kv_b_proj = ToyFp8Linear(_ints(g, (64, HIDDEN), FP8), _bytes(g, (64, 64)))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, 64), FP8), _bytes(g, (HIDDEN, 16)))
        self.k_b_proj_trans = _parameter(_ints(g, (4, 32, 32), FP8))
        self.k_b_proj_trans_dequant = _parameter(_ints(g, (4, 32, 32), BF16))
        self.v_b_proj_scale = _parameter(self.kv_b_proj.weight_scale[:16])
        self._fused_q_placeholder = _ints(g, (64, HIDDEN), BF16)
        self.softmax_stats_tensor = torch.zeros(1024, dtype=FP32)
        self.mha = ToyAttentionBackend(g)


class ToyGate(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.weight = _parameter(_ints(generator, (16, HIDDEN), FP32))
        self.e_score_correction_bias = _parameter(_ints(generator, (16,), FP32))


class ToySharedMLP(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate_up_proj = ToyFp8Linear(_ints(g, (2 * HIDDEN, HIDDEN), FP8), _bytes(g, (128, 32)))
        self.down_proj = ToyFp8Linear(_ints(g, (HIDDEN, HIDDEN), FP8), _bytes(g, (HIDDEN, 8)))


class ToyMoE(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate = ToyGate(g)
        self.shared_experts = ToySharedMLP(g)
        self.routed_expert_down_proj = ToyFp8Linear(_ints(g, (64, HIDDEN), FP8), _bytes(g, (64, 8)))
        self.routed_expert_up_proj = ToyFp8Linear(
            _ints(g, (HIDDEN, 64), FP8), _bytes(g, (HIDDEN, 8))
        )
        self.routed_expert_norm = ToyNorm(g, 64)
        # Expert DWDP owns the routed experts; they are never a dense root.
        self.routed_experts = ToyLinear(_ints(g, (2, HIDDEN, HIDDEN), BF16))


class ToyDecoderLayer(nn.Module):
    """A decoder layer with the model seam: roots plus the two tables."""

    dense_dwdp_attr_allow = ATTR_ALLOW
    dense_dwdp_exclude = EXCLUDE

    def __init__(self, kind, generator):
        super().__init__()
        self.is_moe = kind != "dense"
        self.is_kda = kind != "mla"
        self.input_layernorm = ToyNorm(generator, HIDDEN)
        if self.is_kda:
            self.linear_attn = ToyKDA(generator)
        else:
            self.self_attn = ToyMLA(generator)
        if self.is_moe:
            self.block_sparse_moe = ToyMoE(generator)
        else:
            self.mlp = ToySharedMLP(generator)

    def dense_dwdp_roots(self, segment):
        if not self.is_moe:
            return []
        if segment == "attn":
            return (
                [("linear_attn", self.linear_attn)]
                if self.is_kda
                else [("self_attn", self.self_attn)]
            )
        return [
            (prefix.rstrip("."), self.get_submodule(prefix.rstrip(".")))
            for prefix in SEGMENT_ROOTS[MOE]
        ]


class ToyDecoder(nn.Module):
    def __init__(self, seed, kinds):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(ToyDecoderLayer(kind, generator) for kind in kinds)


class ToyCausalLM(nn.Module):
    """Like the deployed VL wrapper: ``model.model`` is the decoder that holds ``layers``."""

    def __init__(self, seed=0, kinds=LAYER_KINDS):
        super().__init__()
        self.model = ToyDecoder(seed, kinds)


class PlainCausalLM(nn.Module):
    """A model whose decoder layers have no dense roots (DeepSeek-like)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(3))


def expected_moved(kinds=LAYER_KINDS):
    moved = set()
    for layer, kind in enumerate(kinds):
        if kind != "dense":
            attention = KDA_MOVED if kind == "kda" else MLA_MOVED
            moved.update(f"layers.{layer}.{path}" for path in attention + MOE_MOVED)
    return moved


def toy_layout(num_ranks):
    """The layout every rank of an ``num_ranks``-rank group collects from the toy model."""
    return collect_dense_layout(
        list(enumerate(ToyCausalLM().model.layers)),
        DenseConfig(enabled=True, min_bytes=TOY_MIN_BYTES),
        num_ranks=num_ranks,
        device=torch.device("cpu"),
        attr_allow=ATTR_ALLOW,
        exclude=EXCLUDE,
    )


# ---------------------------------------------------------------------------
# Module tensor snapshots
# ---------------------------------------------------------------------------


def module_tensors(decoder):
    """Every tensor the decoder layers hold, by ``layers.<l>.<path>``."""
    found = {}
    for index, layer in enumerate(decoder.layers):
        for path, module in layer.named_modules(prefix=f"layers.{index}"):
            for name, value in [*module._parameters.items(), *module._buffers.items()]:
                if value is not None:
                    found[f"{path}.{name}"] = value
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    found[f"{path}.{name}"] = value
                elif (
                    isinstance(value, (tuple, list))
                    and value
                    and all(isinstance(item, torch.Tensor) for item in value)
                ):
                    found.update({f"{path}.{name}[{i}]": item for i, item in enumerate(value)})
    return found


def placement(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone()


def segment_of(path):
    """``(layer, segment)`` of a snapshot path; the segment is None outside the dense roots."""
    _, layer, rest = path.split(".", 2)
    for kind, prefixes in SEGMENT_ROOTS.items():
        if rest.startswith(prefixes):
            return int(layer), kind
    return int(layer), None


class Snapshot:
    """A decoder's tensors before setup: placements, bytes, storage weakrefs, Parameter objects.

    It holds no strong reference to an original storage, so the originals can die at commit.
    """

    def __init__(self, decoder):
        self.placements = {}
        self.values = {}
        self.storages = {}
        self.parameters = {}
        for path, tensor in module_tensors(decoder).items():
            self.placements[path] = placement(tensor)
            self.values[path] = raw_bytes(tensor)
            self.storages[path] = weakref.ref(tensor.untyped_storage())
            if isinstance(tensor, nn.Parameter):
                self.parameters[path] = tensor

    def changed(self, decoder):
        """Paths whose tensor is gone, moved, changed shape or changed bytes since the snapshot."""
        now = module_tensors(decoder)
        assert now.keys() <= self.placements.keys()
        return [
            path
            for path in self.placements
            if path not in now
            or placement(now[path]) != self.placements[path]
            or not torch.equal(raw_bytes(now[path]), self.values[path])
        ]

    def assert_restored(self, decoder):
        """Every tensor is back: same storage, view, dtype and bytes; same Parameter objects."""
        assert self.changed(decoder) == []
        now = module_tensors(decoder)
        assert [path for path, param in self.parameters.items() if now[path] is not param] == []

    def assert_values(self, decoder, layer, kind):
        """The tensors of ``(layer, kind)`` read the snapshot's bytes through the modules."""
        now = module_tensors(decoder)
        paths = [path for path in now if segment_of(path) == (layer, kind)]
        assert paths
        wrong = [path for path in paths if not torch.equal(raw_bytes(now[path]), self.values[path])]
        assert wrong == [], f"layer {layer} {kind.value} reads other bytes through {wrong[:4]}"


def slot_storage_ptrs(runtime):
    return {
        slot.storage.untyped_storage().data_ptr()
        for ring in runtime.slots.values()
        for slot in ring
    }


def moved_paths(decoder, runtime):
    """Paths whose tensor is now a view of one of the runtime's slots."""
    slots = slot_storage_ptrs(runtime)
    return {
        path
        for path, tensor in module_tensors(decoder).items()
        if tensor.untyped_storage().data_ptr() in slots
    }


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.waits = 0

    def wait_event(self, event):
        self.waits += 1

    def __repr__(self):
        return f"<stream {self.name}>"


class FakeEvent:
    """An event that logs every record into its rank's log."""

    def __init__(self, log):
        self.log = log
        self.records = []

    def record(self, stream=None):
        self.log.append("record")
        self.records.append(stream)


class FakeOps:
    """``DeviceOps`` on the host: a compute and a copy stream; records and synchronizes logged."""

    def __init__(self, log):
        self.log = log
        self.compute = FakeStream("compute")
        self.copy = FakeStream("copy")
        self.current = self.compute

    def new_event(self):
        return FakeEvent(self.log)

    def current_stream(self):
        return self.current

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def synchronize(self):
        self.log.append("sync")

    def is_current_stream_capturing(self):
        return False

    def do_multi_stream(self):
        return False

    def nvtx_range(self, message):
        return contextlib.nullcontext()


class FakeWeightManager:
    """The W1 properties of ``DWDPWeightManager`` that dense setup reads."""

    def __init__(self, copy_stream, *, use_gather=False, use_batched_prefetch=False):
        self._copy_stream = copy_stream
        self._use_gather = use_gather
        self._use_batched_prefetch = use_batched_prefetch

    @property
    def copy_stream(self):
        return self._copy_stream

    @property
    def use_gather(self):
        return self._use_gather

    @property
    def use_batched_prefetch(self):
        return self._use_batched_prefetch


class FakeMemInfo:
    """``mem_get_info`` of a rank's device, with a caching allocator in front of it.

    Before setup the allocator caches ``spec.cached_bytes`` of free blocks, which the device
    counts as used; the first ``empty_cache`` returns them.  An ``empty_cache`` after the moved
    originals died also returns ``FREED_BYTES``: the moved bytes net of the store and the slots.
    """

    def __init__(self, rank):
        self.rank = rank
        self.calls = 0

    def __call__(self):
        rank = self.rank
        self.calls += 1
        rank.log.append("mem_get_info")
        free = rank.spec.free_bytes
        if rank.empty_cache_states:
            free += rank.spec.cached_bytes
        if any(state.dead_paths for state in rank.empty_cache_states):
            free += FREED_BYTES
        return free, TOTAL_BYTES


class HostMemory:
    """The host buffers standing in for store chunks and slots, shared by every rank of a test.

    ``FakeStore`` chunks stay allocated until the test ends, so a rank can never read a peer's
    released store; slots are tracked by weak reference.  Every fake copy is checked against both
    before it touches memory.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._kept = []
        self.chunks = []
        self.slots = []

    def add_chunk(self, chunk):
        """Accept ``chunk`` as a copy source and keep it alive until the test ends."""
        with self._lock:
            self._kept.append(chunk)
        self.add_chunk_range(chunk.data_ptr(), chunk.data_ptr() + chunk.numel())

    def add_chunk_range(self, start, end):
        """Accept ``[start, end)`` as a copy source; its owner keeps it alive."""
        with self._lock:
            self.chunks.append((start, end))

    def add_slot(self, storage):
        start = storage.data_ptr()
        entry = (
            threading.current_thread(),
            start,
            start + storage.numel(),
            weakref.ref(storage),
            weakref.ref(storage.untyped_storage()),
        )
        with self._lock:
            self.slots.append(entry)

    def slots_of(self, thread):
        with self._lock:
            entries = [entry for entry in self.slots if entry[0] is thread]
        return [tensor for tensor in (entry[3]() for entry in entries) if tensor is not None]

    def live_slots(self, thread=None):
        """Owners of slot storages that are still alive; only ``thread``'s if given."""
        with self._lock:
            return [
                entry[0].name
                for entry in self.slots
                if entry[4]() is not None and (thread is None or entry[0] is thread)
            ]

    def check_copy(self, dst, src, nbytes):
        with self._lock:
            chunks = list(self.chunks)
            slots = [(start, end) for _, start, end, _, storage in self.slots if storage()]
        if not any(start <= src and src + nbytes <= end for start, end in chunks):
            raise AssertionError(f"copy source {src:#x}+{nbytes} is outside every store chunk")
        if not any(start <= dst and dst + nbytes <= end for start, end in slots):
            raise AssertionError(f"copy destination {dst:#x}+{nbytes} is outside every slot")


def _called_from_finalizer():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "__del__":
            return True
        frame = frame.f_back
    return False


#: The store calls setup makes, in this order and once each (``DenseShardStore``).
STORE_STEPS = ("allocate", "write", "export", "import")


class FakeStore(DenseShardStore):
    """One rank's rows in host memory, read by the other rank threads at the exported addresses.

    It keeps its own state rather than the base constructor's and enforces the base class's call
    order: each step once, in ``STORE_STEPS`` order, none after ``release``.  ``release`` notes
    whether a finalizer called it, so a test can tell setup's explicit release from garbage
    collection.
    """

    def __init__(self, layout, rank, device, *, owner):
        self.layout = layout
        self.rank = rank
        self.device = device
        self.owner = owner
        self.calls = Counter()
        self.finalizer_releases = 0
        self.chunks = []
        self.bases = None

    def _step(self, name):
        self.owner.log.append(name)
        self.calls[name] += 1
        done = tuple(step for step in STORE_STEPS if self.calls[step])
        if self.calls["release"] or self.calls[name] > 1 or done != STORE_STEPS[: len(done)]:
            raise RuntimeError(f"store {name} out of order: {dict(self.calls)}")
        if self.owner.spec.fail == name:
            raise RuntimeError(f"injected: store {name} failed")
        if self.owner.spec.interrupt == name:
            raise KeyboardInterrupt(f"injected: store {name} interrupted")

    def allocate(self):
        self._step("allocate")
        for chunk in self.layout.chunks:
            raw = torch.zeros(chunk.handle_bytes + ROW_ALIGN, dtype=torch.uint8)
            pad = (-raw.data_ptr()) % ROW_ALIGN
            local = raw[pad : pad + chunk.handle_bytes]
            self.owner.memory.add_chunk(local)
            self.chunks.append(local)

    def write_local_rows(self):
        self._step("write")
        for segment in self.layout.segments.values():
            chunk = self.chunks[segment.chunk]
            for piece in row_pieces(segment, self.rank, num_ranks=self.layout.num_ranks):
                source = segment.anchors[piece.anchor_index].source
                start = segment.chunk_offset + piece.row_offset
                chunk[start : start + piece.nbytes].copy_(
                    source[piece.anchor_offset : piece.anchor_offset + piece.nbytes]
                )

    def export(self):
        self._step("export")
        return [chunk.data_ptr().to_bytes(8, "little") for chunk in self.chunks]

    def import_peers(self, payloads):
        self._step("import")
        bases = tuple(
            tuple(int.from_bytes(handle, "little") for handle in payload) for payload in payloads
        )
        if bases[self.rank] != tuple(chunk.data_ptr() for chunk in self.chunks):
            raise RuntimeError("collective #2 returned another rank's payload for this rank")
        self.bases = bases

    def chunk_bases(self):
        if self.bases is None:
            raise RuntimeError("chunk_bases before import_peers")
        return self.bases

    def release(self):
        self.owner.log.append("release")
        self.calls["release"] += 1
        if _called_from_finalizer():
            self.finalizer_releases += 1
        self.chunks = []
        self.bases = None


class LoopbackStore(LoopbackShardStore):
    """The real loopback store, logged like ``FakeStore``; its chunks become copy sources."""

    def __init__(self, layout, rank, device, *, owner):
        super().__init__(layout, rank, device)
        self.layout = layout
        self.owner = owner
        self.calls = Counter()
        self.finalizer_releases = 0

    def _log(self, name):
        self.owner.log.append(name)
        self.calls[name] += 1

    def allocate(self):
        self._log("allocate")
        super().allocate()

    def write_local_rows(self):
        self._log("write")
        super().write_local_rows()

    def export(self):
        self._log("export")
        return super().export()

    def import_peers(self, payloads):
        self._log("import")
        super().import_peers(payloads)
        for bases in self.chunk_bases():
            for chunk, base in zip(self.layout.chunks, bases):
                self.owner.memory.add_chunk_range(base, base + chunk.handle_bytes)

    def release(self):
        self._log("release")
        if _called_from_finalizer():
            self.finalizer_releases += 1
        super().release()


class FakeLauncher:
    """Runs a rank's copy plans as host memmoves, with the faults of the rank's ``RankSpec``.

    Round 0 is Phase V (P4), round 1 the P5 self-check, later rounds the committed runtime.
    """

    def __init__(self, plans, rank):
        self.plans = dict(plans)
        self.rank = rank
        self.streams = []
        self.releases = 0

    def __call__(self, layer):
        rank = self.rank
        round_index = len(self.streams) // len(self.plans)
        self.streams.append(rank.ops.current)
        rank.log.append(f"launch L{layer}")
        spec = rank.spec
        fault = (spec.fail, round_index)
        if fault in {("p4_raise", 0), ("p5_raise", 1)}:
            raise RuntimeError(f"injected: the copy of layer {layer} failed")
        for dst, src, nbytes in self.plans[layer].descriptors:
            rank.memory.check_copy(dst, src, nbytes)
            ctypes.memmove(dst, src, nbytes)
        corrupt = spec.corrupt
        if corrupt is not None and (corrupt.phase - 4, corrupt.layer) == (round_index, layer):
            byte = ctypes.c_uint8.from_address(rank.corrupt_address)
            byte.value ^= 0xFF
        if fault == ("lose_buffer", 1) and layer == COVERED[-1]:
            del rank.decoder.layers[layer].get_submodule(LOST_MODULE)._buffers["weight"]

    def release(self):
        self.releases += 1


class CountingComm:
    """The rank's ``GroupComm``: a ``StoreDwdpGroup`` whose collectives are logged and recorded.

    ``spec.comm_rank`` overrides the rank the group reports; at collective ``spec.malformed`` the
    rank sends a ``dict`` instead of its report.
    """

    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.sent = []
        self.received = []

    def Get_rank(self):
        override = self.rank.spec.comm_rank
        return self.group.Get_rank() if override is None else override

    def Get_size(self):
        return self.group.Get_size()

    def allgather(self, obj):
        number = len(self.sent) + 1
        self.rank.log.append(f"#{number}")
        if number == self.rank.spec.malformed:
            obj = {"ok": True}
        self.sent.append(obj)
        gathered = self.group.allgather(obj)
        self.received.append(gathered)
        return gathered


class CaptureLogger:
    """Stands in for the setup and repoint modules' ``logger``; keeps each message's thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.records = []

    def _add(self, level, message):
        with self._lock:
            self.records.append((threading.current_thread(), level, message))

    def debug(self, message):
        self._add("debug", message)

    def info(self, message):
        self._add("info", message)

    def warning(self, message):
        self._add("warning", message)

    def error(self, message):
        self._add("error", message)


# ---------------------------------------------------------------------------
# Rank threads
# ---------------------------------------------------------------------------


class Corruption(NamedTuple):
    """A slot byte a rank's launcher flips right after it copies ``layer`` in ``phase`` (4 or 5).

    ``anchor`` indexes the anchors of ``(layer, segment)`` and ``byte`` the anchor's bytes;
    negative values count from the end.
    """

    phase: int
    layer: int
    segment: SegmentKind
    anchor: int
    byte: int


def corruption_address(layout, slot_bases, corruption):
    """Address of the slot byte ``corruption`` flips."""
    anchor = layout.segment(corruption.layer, corruption.segment).anchors[corruption.anchor]
    ring = slot_bases[corruption.segment]
    base = ring[layout.slot_index(corruption.segment, corruption.layer)]
    return base + anchor.blob_offset + corruption.byte % anchor.nbytes


@dataclass(frozen=True)
class RankSpec:
    """One rank's switches, model, device memory and injected faults.

    ``fail`` names a fault: ``store`` (store constructor), ``allocate``, ``write``, ``export``,
    ``import`` (store steps), ``runtime`` (no copy stream, so the runtime constructor raises),
    ``launcher`` (launcher factory), ``p4_raise`` / ``p5_raise`` (a copy raises), ``lose_buffer``
    (the P5 copy of the last layer deletes the ``weight`` buffer of ``LOST_MODULE``, so the
    self-check and then the rollback fail).  ``interrupt`` names a store step that raises
    ``KeyboardInterrupt``.  ``corrupt`` flips one slot byte.  ``perturb`` adds 1 to the first
    element of a decoder parameter.  ``comm_rank`` overrides the rank the group reports;
    ``malformed`` is the collective at which the rank sends a ``dict`` instead of its report.
    ``loopback`` runs the real ``LoopbackShardStore`` instead of ``FakeStore``.
    """

    requested: bool = True
    required: bool = True
    strict: bool = False
    debug: int = 0
    seed: int = 0
    kinds: tuple[str, ...] = LAYER_KINDS
    roots: bool = True
    moe_layers: tuple[int, ...] = COVERED
    free_bytes: int = PLENTY_FREE_BYTES
    cached_bytes: int = 0
    fail: str | None = None
    interrupt: str | None = None
    corrupt: Corruption | None = None
    perturb: str | None = None
    comm_rank: int | None = None
    malformed: int | None = None
    loopback: bool = False
    use_gather: bool = False
    use_batched: bool = False


@dataclass(frozen=True)
class EmptyCacheState:
    """What a rank looked like when setup called ``torch.cuda.empty_cache``."""

    model_active: bool
    dead_paths: frozenset[str]
    live_slots: int
    slots_filled: bool
    stores_released: bool


class Rank:
    """Everything one rank thread owns: model, fakes, seams, its log and the outcome of setup."""

    def __init__(self, index, spec, memory):
        self.index = index
        self.spec = spec
        self.memory = memory
        self.log = []
        self.model = ToyCausalLM(spec.seed, spec.kinds) if spec.roots else PlainCausalLM()
        self.decoder = self.model.model
        if spec.perturb is not None:
            with torch.no_grad():
                self.decoder.get_parameter(spec.perturb).data.view(-1)[0] += 1
        self.snapshot = Snapshot(self.decoder)
        self.ops = FakeOps(self.log)
        self.weight_manager = FakeWeightManager(
            None if spec.fail == "runtime" else self.ops.copy,
            use_gather=spec.use_gather,
            use_batched_prefetch=spec.use_batched,
        )
        self.mem_get_info = FakeMemInfo(self)
        self.config = DenseConfig(
            enabled=spec.requested,
            required=spec.required,
            min_bytes=TOY_MIN_BYTES,
            strict=spec.strict,
            debug=spec.debug,
        )
        self.stores = []
        self.launchers = []
        self.factory_streams = []
        self.empty_cache_states = []
        self.corrupt_address = None
        self.thread = None
        self.comm = None
        self.result = None
        self.error = None
        self.trace = ""

    def store_factory(self, layout, rank, device):
        if self.spec.fail == "store":
            raise RuntimeError("injected: the store constructor failed")
        store_type = LoopbackStore if self.spec.loopback else FakeStore
        store = store_type(layout, rank, device, owner=self)
        self.stores.append(store)
        return store

    def launcher_factory(self, plans, weight_manager, device):
        assert weight_manager is self.weight_manager and device == torch.device("cpu")
        self.log.append("factory")
        self.factory_streams.append(self.ops.current)
        if self.spec.fail == "launcher":
            raise RuntimeError("injected: building the copy launcher failed")
        launcher = FakeLauncher(plans, self)
        self.launchers.append(launcher)
        return launcher

    def on_empty_cache(self):
        thread = threading.current_thread()
        self.log.append("empty_cache")
        slots = self.memory.slots_of(thread)
        fill = self.config.slot_fill_byte
        self.empty_cache_states.append(
            EmptyCacheState(
                model_active=getattr(self.model, "_dense_dwdp_active", False),
                dead_paths=frozenset(
                    path for path, ref in self.snapshot.storages.items() if ref() is None
                ),
                live_slots=len(self.memory.live_slots(thread)),
                slots_filled=bool(slots) and all(bool((slot == fill).all()) for slot in slots),
                stores_released=all(store.calls["release"] for store in self.stores),
            )
        )

    def run(self, store, size, overrides):
        self.comm = CountingComm(
            StoreDwdpGroup(
                global_rank=self.index,
                dwdp_size=size,
                master_addr="unused",
                timeout_s=COLLECTIVE_TIMEOUT_S,
                store=store,
            ),
            self,
        )
        seams = {
            "config": self.config,
            "store_factory": self.store_factory,
            "ops": self.ops,
            "device": torch.device("cpu"),
            "launcher_factory": self.launcher_factory,
            "mem_get_info": self.mem_get_info,
            **overrides,
        }
        try:
            self.result = setup_dense_dwdp(
                self.model,
                self.weight_manager,
                self.comm,
                dwdp_rank=self.index,
                dwdp_size=size,
                device_id=0,
                moe_layers=list(self.spec.moe_layers),
                **seams,
            )
        except (Exception, KeyboardInterrupt) as exc:  # the test inspects every rank's outcome
            self.error = (type(exc), str(exc))
            self.trace = traceback.format_exc()


class Group:
    """``N`` rank threads that run dense setup together over one ``HashStore``."""

    def __init__(self, specs, memory, capture, hash_store_type):
        self.ranks = [Rank(index, spec, memory) for index, spec in enumerate(specs)]
        self.memory = memory
        self.capture = capture
        self._hash_store_type = hash_store_type

    def run(self, **overrides):
        store = self._hash_store_type()
        size = len(self.ranks)
        for rank in self.ranks:
            rank.thread = threading.Thread(
                target=rank.run,
                args=(store, size, overrides),
                name=f"dense-setup-rank{rank.index}",
                daemon=True,
            )
        for rank in self.ranks:
            rank.thread.start()
        for rank in self.ranks:
            rank.thread.join(JOIN_TIMEOUT_S)
        stuck = [rank.index for rank in self.ranks if rank.thread.is_alive()]
        assert stuck == [], f"dense setup did not finish on ranks {stuck}"
        return self

    def rank_on(self, thread):
        return next((rank for rank in self.ranks if rank.thread is thread), None)

    def messages(self, rank, level):
        return [
            message
            for thread, kind, message in self.capture.records
            if thread is rank.thread and kind == level
        ]

    def collective_counts(self):
        return [len(rank.comm.sent) for rank in self.ranks]

    def assert_collectives(self, count):
        """Every rank made ``count`` collectives, all of ``PhaseReport``s, and gathered the same."""
        assert self.collective_counts() == [count] * len(self.ranks)
        for rank in self.ranks:
            assert all(isinstance(report, PhaseReport) for report in rank.comm.sent)
            assert rank.comm.received == self.ranks[0].comm.received
        return self.ranks[0].comm.received

    def assert_all_failed(self, pattern):
        for rank in self.ranks:
            assert rank.error is not None, f"rank {rank.index} returned {rank.result!r}"
            error_type, message = rank.error
            assert error_type is DenseSetupError, rank.trace
            assert re.search(pattern, message), message

    def assert_all_none(self):
        for rank in self.ranks:
            assert rank.error is None, rank.trace
            assert rank.result is None

    def assert_released(self):
        """Every resource setup made was released exactly once, explicitly, and the slots died."""
        gc.collect()
        for rank in self.ranks:
            for store in rank.stores:
                assert store.calls["release"] == 1, (rank.index, store.calls)
                assert store.finalizer_releases == 0, rank.index
            assert [launcher.releases for launcher in rank.launchers] == [1] * len(rank.launchers)
        assert self.memory.live_slots() == []

    def assert_cache_emptied_after_release(self):
        """Abandoning ended with ``empty_cache`` once the store was released and the slots died."""
        for rank in self.ranks:
            assert rank.log[-1] == "empty_cache", rank.log
            state = rank.empty_cache_states[-1]
            assert state.stores_released and state.live_slots == 0, (rank.index, state)

    def assert_restored(self):
        for rank in self.ranks:
            rank.snapshot.assert_restored(rank.decoder)
            assert not hasattr(rank.model, "_dense_dwdp_active")
            assert not hasattr(rank.decoder, "_dense_dwdp_active")

    def committed(self):
        for rank in self.ranks:
            assert rank.error is None, rank.trace
            assert isinstance(rank.result, DenseDWDP)
        return [rank.result for rank in self.ranks]

    def release_runtimes(self):
        for rank in self.ranks:
            if isinstance(rank.result, DenseDWDP):
                rank.result.release()


@pytest.fixture
def make_group(monkeypatch):
    """Builds rank groups; routes setup's slots, plans, cache release and logs to the test."""
    torch_distributed = pytest.importorskip("torch.distributed")
    memory = HostMemory()
    capture = CaptureLogger()
    groups = []
    allocate_slot_ring = dense_setup.allocate_slot_ring
    build_copy_plans = dense_setup.build_copy_plans

    def current_rank():
        thread = threading.current_thread()
        for group in groups:
            rank = group.rank_on(thread)
            if rank is not None:
                return rank
        return None

    def recording_allocate_slot_ring(spec, device):
        ring = allocate_slot_ring(spec, device)
        for slot in ring:
            memory.add_slot(slot.storage)
        rank = current_rank()
        if rank is not None:
            rank.log.append(f"slots {spec.segment.value}")
        return ring

    def recording_build_copy_plans(layout, *, rank, chunk_bases, slot_bases):
        owner = current_rank()
        if owner is not None:
            owner.log.append("plans")
            if owner.spec.corrupt is not None:
                owner.corrupt_address = corruption_address(layout, slot_bases, owner.spec.corrupt)
        return build_copy_plans(layout, rank=rank, chunk_bases=chunk_bases, slot_bases=slot_bases)

    def recording_empty_cache():
        rank = current_rank()
        if rank is not None:
            rank.on_empty_cache()

    monkeypatch.setattr(dense_setup, "allocate_slot_ring", recording_allocate_slot_ring)
    monkeypatch.setattr(dense_setup, "build_copy_plans", recording_build_copy_plans)
    monkeypatch.setattr(dense_setup, "logger", capture)
    monkeypatch.setattr(dense_repoint, "logger", capture)
    monkeypatch.setattr(torch.cuda, "empty_cache", recording_empty_cache)

    def make(specs):
        group = Group(specs, memory, capture, torch_distributed.HashStore)
        groups.append(group)
        return group

    yield make
    for group in groups:
        group.release_runtimes()


def specs_for(num_ranks, **fields):
    return [RankSpec(**fields) for _ in range(num_ranks)]


def assert_setup_order(rank, kinds=(ATTN, MOE)):
    """``rank`` ran setup's steps up to the commit in the order described below.

    P1 reads the free memory after emptying the allocator cache.  P2 allocates the store and the
    slots (in either order), writes the rows, synchronizes, constructs the runtime (one consume
    record per slot) and only then exports.  P3 imports, builds the plans and then the launcher,
    and synchronizes before the barrier.  P4 and P5 synchronize after every layer's copy.  P6
    empties the cache before it measures the free memory; then the runtime records the consume
    events again.
    """
    records = ["record"] * sum(RING_DEPTH[kind] for kind in kinds)
    fetches = [step for layer in COVERED for step in (f"launch L{layer}", "sync")]
    allocations = ["allocate", *(f"slots {kind.value}" for kind in kinds)]
    log = rank.log
    assert log[:3] == ["empty_cache", "mem_get_info", "#1"], log
    assert sorted(log[3 : 3 + len(allocations)]) == sorted(allocations), log
    assert log[3 + len(allocations) :] == [
        "write",
        "sync",
        *records,
        "export",
        "#2",
        "import",
        "plans",
        "factory",
        "sync",
        "#3",
        *fetches,
        "#4",
        *fetches,
        "#5",
        "empty_cache",
        "mem_get_info",
        *records,
    ], log


def drive_forward(rank):
    """Prime and run the covered layers through the hooks, as ``DwdpManager`` and the model do.

    Each segment's tensors are read through the modules inside the segment's read window; tensors
    of a segment that is not enabled are the originals and read the same bytes anywhere.
    """
    runtime = rank.result
    for layer in runtime.layers[:2]:
        runtime.prefetch(layer)
    runtime.mark_primed()
    for layer in runtime.layers:
        runtime.wait_layer(layer)
        rank.snapshot.assert_values(rank.decoder, layer, ATTN)
        runtime.attention_done(layer)
        rank.snapshot.assert_values(rank.decoder, layer, MOE)
        runtime.layer_done(layer)
    assert (runtime.demand_fetches, runtime.war_fences) == (0, 0)


def clear_dense_env(monkeypatch):
    """Unset every ``TRTLLM_DWDP_DENSE*`` variable, so a test sets exactly the ones it reads."""
    for name in [name for name in os.environ if name.startswith("TRTLLM_DWDP_DENSE")]:
        monkeypatch.delenv(name)


COMMIT_LINE = re.compile(
    r"^\[DWDP Dense\] committed: moved=\d+\.\d\dGB store=\d+\.\d\dGB slots=\d+\.\d\dGB "
    r"freed\(mem_get_info\)=(?P<freed>-?\d+\.\d\d)GB "
    r"freed\(memory_allocated\)=-?\d+\.\d\dGB leaks=(?P<leaks>\d+) "
    r"layers=(?P<layers>\d+) segments=(?P<segments>\S+) setup=\d+\.\ds$"
)

# ---------------------------------------------------------------------------
# W1 and the model probe
# ---------------------------------------------------------------------------


def test_weight_manager_exposes_its_copy_stream_and_transfer_mode():
    manager = DWDPWeightManager.__new__(DWDPWeightManager)
    stream = object()
    manager._copy_stream = stream
    manager._use_gather = True
    manager._use_batched_prefetch = False
    assert manager.copy_stream is stream
    assert manager.use_gather is True
    assert manager.use_batched_prefetch is False
    for name in ("copy_stream", "use_gather", "use_batched_prefetch"):
        assert isinstance(getattr(DWDPWeightManager, name), property)
        with pytest.raises(AttributeError):
            setattr(manager, name, None)


def test_dense_roots_are_found_through_the_decoder():
    assert dense_roots_available(ToyCausalLM())
    assert dense_roots_available(ToyCausalLM().model)
    assert not dense_roots_available(PlainCausalLM())
    assert not dense_roots_available(nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_ranks", [2, 4])
def test_setup_commits_on_every_rank(make_group, num_ranks):
    group = make_group(specs_for(num_ranks, debug=1, cached_bytes=CACHED_BYTES)).run()
    group.assert_collectives(5)
    runtimes = group.committed()
    gc.collect()
    for rank, runtime in zip(group.ranks, runtimes):
        assert runtime.layers == COVERED and runtime.segments == (ATTN, MOE)
        assert not runtime.primed and not runtime.forward_open
        for kind, ring in runtime.slots.items():
            assert [runtime.slot_tag(kind, slot.index) for slot in ring] == [EMPTY_TAG] * len(ring)
            assert all(bool((slot.storage == 0xFF).all()) for slot in ring)
        assert rank.model._dense_dwdp_active is True
        assert rank.decoder._dense_dwdp_active is True

        moved = moved_paths(rank.decoder, runtime)
        assert moved == expected_moved()
        dead = {path for path, ref in rank.snapshot.storages.items() if ref() is None}
        assert moved <= dead
        kept = [
            path
            for path, tensor in module_tensors(rank.decoder).items()
            if path not in moved and placement(tensor) != rank.snapshot.placements[path]
        ]
        assert kept == []
        tensors = module_tensors(rank.decoder)
        assert all(tensors[path] is param for path, param in rank.snapshot.parameters.items())

        # The copies ran with the copy stream current; the launcher was built under it.
        assert rank.factory_streams == [rank.ops.copy]
        (launcher,) = rank.launchers
        assert len(launcher.streams) == 2 * len(COVERED)
        assert all(stream is rank.ops.copy for stream in launcher.streams)

        # P1 measures with the allocator cache emptied.  P6 drops the originals, empties the
        # cache and measures again; then the runtime fills the slots and the model is flagged.
        before, after = rank.empty_cache_states
        assert not before.dead_paths and before.live_slots == 0
        assert moved <= after.dead_paths and after.live_slots == sum(RING_DEPTH.values())
        assert not after.model_active and not after.slots_filled and not after.stores_released

        (line,) = [m for m in group.messages(rank, "info") if "committed:" in m]
        fields = COMMIT_LINE.match(line)
        assert fields is not None, line
        assert fields["freed"] == f"{FREED_BYTES / 1e9:.2f}"
        assert (fields["leaks"], fields["layers"], fields["segments"]) == ("0", "5", "attn,moe")

    for rank in group.ranks:
        drive_forward(rank)
    for rank, runtime in zip(group.ranks, runtimes):
        runtime.release()
        (store,) = rank.stores
        assert store.calls["release"] == 1 and store.finalizer_releases == 0
        tensors = module_tensors(rank.decoder)
        assert all(tensors[path].numel() == 0 for path in expected_moved())
        assert rank.model._dense_dwdp_active is True


@pytest.mark.parametrize("loopback", [False, True], ids=["fake-store", "loopback-store"])
@pytest.mark.parametrize("num_ranks", [2, 4])
def test_setup_runs_its_steps_in_the_design_order(make_group, num_ranks, loopback):
    group = make_group(specs_for(num_ranks, loopback=loopback)).run()
    group.assert_collectives(5)
    group.committed()
    for rank in group.ranks:
        assert_setup_order(rank)
        assert [type(store) for store in rank.stores] == [LoopbackStore if loopback else FakeStore]
        drive_forward(rank)


def test_switches_come_from_the_environment_without_a_config(make_group, monkeypatch):
    clear_dense_env(monkeypatch)
    monkeypatch.setenv("TRTLLM_DWDP_DENSE", "1")
    monkeypatch.setenv("TRTLLM_DWDP_DENSE_SEGMENTS", "attn")
    monkeypatch.setenv("TRTLLM_DWDP_DENSE_MIN_BYTES", str(TOY_MIN_BYTES))
    group = make_group(specs_for(2)).run(config=None)
    group.assert_collectives(5)
    for rank, runtime in zip(group.ranks, group.committed()):
        assert runtime.segments == (ATTN,)
        assert_setup_order(rank, kinds=(ATTN,))
        # Without TRTLLM_DWDP_DENSE_DEBUG the commit fills the slots with zeros.
        assert all(bool((slot.storage == 0).all()) for slot in runtime.slots[ATTN])
        assert moved_paths(rank.decoder, runtime) == {
            path for path in expected_moved() if segment_of(path)[1] is ATTN
        }
        drive_forward(rank)


@pytest.mark.parametrize(
    "use_gather,use_batched,mode",
    [
        (True, False, CopyMode.GATHER),
        (True, True, CopyMode.GATHER),
        (False, True, CopyMode.BATCHED),
        (False, False, CopyMode.PER_SLICE),
    ],
)
def test_the_default_launcher_follows_the_expert_transfer(
    make_group, monkeypatch, use_gather, use_batched, mode
):
    built = []

    class RecordingLauncher(FakeLauncher):
        def __init__(self, plans, copy_mode, *, device):
            rank = group.rank_on(threading.current_thread())
            built.append((rank.index, copy_mode, device, rank.ops.current is rank.ops.copy))
            super().__init__(plans, rank)

    monkeypatch.setattr(dense_setup, "DenseCopyLauncher", RecordingLauncher)
    group = make_group(specs_for(2, use_gather=use_gather, use_batched=use_batched))
    group.run(launcher_factory=None)
    group.committed()
    cpu = torch.device("cpu")
    assert sorted(built, key=lambda entry: entry[0]) == [(0, mode, cpu, True), (1, mode, cpu, True)]


def test_strict_reaches_the_committed_runtime(make_group):
    group = make_group(specs_for(2, strict=True)).run()
    group.assert_collectives(5)
    for runtime in group.committed():
        # Nothing primed the first layer, so its wait needs a demand fetch, which STRICT forbids.
        with pytest.raises(DenseProtocolError, match="STRICT forbids demand fetches"):
            runtime.wait_layer(runtime.first_layer)


@pytest.mark.parametrize("debug", [0, 1, 2])
def test_only_debug_2_arms_the_slot_watchdog(make_group, debug):
    """``DEBUG=2`` builds the slot watchdog in P2 over the P1 checksums; ``DEBUG=1`` only fills."""
    group = make_group(specs_for(2, debug=debug)).run()
    group.assert_collectives(5)
    for rank, runtime in zip(group.ranks, group.committed()):
        fill = 0xFF if debug else 0
        assert all(
            bool((slot.storage == fill).all()) for ring in runtime.slots.values() for slot in ring
        )
        drive_forward(rank)
        checks = len(COVERED) * len(runtime.segments) if debug == 2 else 0
        # Every H1 checked the slots it waited for, and the copies brought the P1 bytes.
        assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (checks, 0)


@pytest.mark.parametrize(
    "strict,debug", [(False, 0), (False, 1), (True, 0)], ids=["warn", "warn-debug", "strict"]
)
def test_a_leaked_original_is_reported_after_the_commit_point(make_group, strict, debug):
    group = make_group([RankSpec(), RankSpec(strict=strict, debug=debug)])
    leaker = group.ranks[1]
    # A holder the bindings do not know about keeps one original alive.
    hidden = leaker.decoder.layers[1].linear_attn.o_proj.weight
    group.run()
    group.assert_collectives(5)
    assert isinstance(group.ranks[0].result, DenseDWDP)
    (warning,) = [m for m in group.messages(leaker, "warning") if "still alive after commit" in m]
    assert "storage of layers.1.linear_attn._o_dense " in warning
    # TRTLLM_DWDP_DENSE_DEBUG >= 1 also describes the holders (a scan of every live object).
    assert ("; holders: " in warning) == (debug >= 1)
    if not strict:
        group.committed()
        (line,) = [m for m in group.messages(leaker, "info") if "committed:" in m]
        assert COMMIT_LINE.match(line)["leaks"] == "1"
        return
    # Only the leaking rank fails, after the commit point: its originals are gone, so its
    # bindings are detached and its store released, and the model is flagged.
    error_type, message = leaker.error
    assert error_type is DenseRepointError and "still alive after commit" in message
    assert leaker.result is None
    tensors = module_tensors(leaker.decoder)
    assert all(tensors[path].numel() == 0 for path in expected_moved())
    (store,) = leaker.stores
    assert store.calls["release"] == 1 and store.finalizer_releases == 0
    assert leaker.launchers[0].releases == 1
    assert leaker.model._dense_dwdp_active is True
    assert torch.equal(
        raw_bytes(hidden), leaker.snapshot.values["layers.1.linear_attn.o_proj.weight"]
    )


# ---------------------------------------------------------------------------
# Dense off, configuration mismatches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_ranks", [2, 4])
@pytest.mark.parametrize("roots", [True, False], ids=["k3", "no-roots"])
def test_dense_off_everywhere_returns_after_one_collective(make_group, num_ranks, roots):
    group = make_group(specs_for(num_ranks, requested=False, roots=roots)).run()
    (reports,) = group.assert_collectives(1)
    assert all(report.ok and not report.data.requested for report in reports)
    group.assert_all_none()
    group.assert_restored()
    # A rank that did not ask for dense DWDP does nothing but the first collective.
    assert all(rank.log == ["#1"] and rank.stores == [] for rank in group.ranks)


def test_a_bad_segment_list_cannot_break_dense_off(make_group, monkeypatch):
    clear_dense_env(monkeypatch)
    monkeypatch.setenv("TRTLLM_DWDP_DENSE_SEGMENTS", "bogus")
    group = make_group(specs_for(2)).run(config=None)
    group.assert_collectives(1)
    group.assert_all_none()


@pytest.mark.parametrize("num_ranks,off", [(2, 0), (2, 1), (4, 2)])
def test_ranks_that_disagree_on_the_dense_switch_all_raise(make_group, num_ranks, off):
    specs = specs_for(num_ranks, required=False)
    specs[off] = replace(specs[off], requested=False)
    group = make_group(specs).run()
    group.assert_collectives(1)
    group.assert_all_failed(r"TRTLLM_DWDP_DENSE differs across the DWDP group")
    group.assert_restored()
    assert all(rank.stores == [] for rank in group.ranks)


def test_an_unreadable_config_fails_safe_on_every_rank(make_group, monkeypatch):
    clear_dense_env(monkeypatch)
    monkeypatch.setenv("TRTLLM_DWDP_DENSE", "1")
    monkeypatch.setenv("TRTLLM_DWDP_DENSE_REQUIRED", "0")
    monkeypatch.setenv("TRTLLM_DWDP_DENSE_SEGMENTS", "moe")
    group = make_group(specs_for(2)).run(config=None)
    group.assert_collectives(1)
    group.assert_all_failed(r"collective #1 .*TRTLLM_DWDP_DENSE_SEGMENTS")
    group.assert_restored()


def test_a_model_without_roots_cannot_turn_dense_on(make_group):
    group = make_group(specs_for(2, roots=False)).run()
    group.assert_collectives(1)
    group.assert_all_failed(
        r"ranks \[0, 1\]: DenseLayoutError: .*does not provide dense_dwdp_roots"
    )


def test_covered_layers_must_be_the_expert_layers(make_group):
    group = make_group(specs_for(2, moe_layers=(1, 2, 3, 4))).run()
    group.assert_collectives(1)
    group.assert_all_failed(r"only dense \[5\], only expert \[\]")
    group.assert_restored()


def test_the_group_must_report_the_rank_setup_runs_as(make_group):
    specs = specs_for(2)
    specs[1] = replace(specs[1], comm_rank=0)
    group = make_group(specs).run()
    group.assert_collectives(1)
    group.assert_all_failed(
        r"collective #1 .*: rank 1: DenseSetupError: the DWDP group reports rank 0 of 2, but "
        r"dense setup runs as rank 1 of 2"
    )
    group.assert_restored()


def test_every_decoder_layer_must_carry_the_same_tables(make_group):
    group = make_group(specs_for(2))
    for rank in group.ranks:
        rank.decoder.layers[2].dense_dwdp_attr_allow = ATTR_ALLOW - {"_gate_b_t"}
    group.run()
    group.assert_collectives(1)
    group.assert_all_failed(
        r"ranks \[0, 1\]: DenseLayoutError: the decoder layers carry 2 different "
        r"dense_dwdp_attr_allow / dense_dwdp_exclude tables \(layers \[0, 1, 3, 4, 5\]; "
        r"layers \[2\]\)"
    )
    group.assert_restored()


@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
def test_different_layouts_abandon_at_the_first_collective(make_group, required):
    specs = specs_for(4, required=required)
    specs[2] = replace(specs[2], kinds=("dense", "kda", "kda", "kda", "kda", "kda"))
    group = make_group(specs).run()
    group.assert_collectives(1)
    pattern = r"collective #1 .*the dense layouts differ across ranks .* on rank 2\)"
    if required:
        group.assert_all_failed(pattern)
    else:
        group.assert_all_none()
        for rank in group.ranks:
            (message,) = group.messages(rank, "error")
            assert re.search(pattern, message), message
    group.assert_restored()


@pytest.mark.parametrize(
    "change,findings",
    [
        ({"seed": 1}, r"layers\.1 attn: ranks \[0, 2, 3\] vs rank 1; layers\.1 moe: "),
        (
            {"perturb": "layers.4.block_sparse_moe.gate.weight"},
            r"layers\.4 moe: ranks \[0, 2, 3\] vs rank 1$",
        ),
    ],
    ids=["every-segment", "one-moe-weight"],
)
def test_different_weights_abandon_at_the_first_collective(make_group, change, findings):
    specs = specs_for(4)
    specs[1] = replace(specs[1], **change)
    group = make_group(specs).run()
    group.assert_collectives(1)
    group.assert_all_failed(rf"the dense weights differ across ranks: {findings}")
    group.assert_restored()


@pytest.mark.parametrize("short", [1, 0], ids=["one-byte-short", "exactly-enough"])
def test_the_memory_precheck_needs_the_store_the_slots_and_the_headroom(make_group, short):
    layout = toy_layout(2)
    needed = layout.store_bytes + layout.slot_bytes + SETUP_HEADROOM_BYTES
    # The allocator of rank 1 caches CACHED_BYTES, which P1 counts once it emptied the cache.
    tight = RankSpec(free_bytes=needed - short - CACHED_BYTES, cached_bytes=CACHED_BYTES)
    group = make_group([RankSpec(), tight]).run()
    if not short:
        group.assert_collectives(5)
        group.committed()
        return
    group.assert_collectives(1)
    group.assert_all_failed(
        r"collective #1 .*: rank 1: DenseSetupError: \d+\.\d\d GB of device memory is free, but "
        r"the store .* need \d+\.\d\d GB"
    )
    group.assert_restored()


def test_one_rank_requiring_dense_makes_every_rank_raise(make_group):
    group = make_group([RankSpec(required=False), RankSpec(required=True, fail="allocate")])
    group.run()
    group.assert_collectives(2)
    group.assert_all_failed(
        r"collective #2 .*rank 1: RuntimeError: injected: store allocate failed"
    )
    group.assert_restored()
    group.assert_released()


# ---------------------------------------------------------------------------
# Failures in every phase on every rank
# ---------------------------------------------------------------------------

#: The fault each phase injects on the failing rank, and what that rank's report says.
PHASE_FAULTS = {
    1: ({"free_bytes": 1 << 20}, r"DenseSetupError: 0\.00 GB of device memory is free"),
    2: ({"fail": "allocate"}, r"RuntimeError: injected: store allocate failed"),
    3: ({"fail": "import"}, r"RuntimeError: injected: store import failed"),
    4: (
        {"corrupt": Corruption(4, COVERED[-1], MOE, -1, -1)},
        rf"Phase V: 1 gathered anchors differ from their originals: {re.escape(LAST_MOE_ANCHOR)}",
    ),
    5: (
        {"corrupt": Corruption(5, COVERED[-1], MOE, -1, -1)},
        r"P5 self-check: 1 module tensors are not their slot views or differ from their P1 "
        rf"checksums after repointing: {re.escape(LAST_MOE_ANCHOR)}",
    ),
}


@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
@pytest.mark.parametrize("num_ranks,failing", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("phase", sorted(PHASE_FAULTS))
def test_a_failure_in_any_phase_on_any_rank_abandons_everywhere(
    make_group, phase, num_ranks, failing, required
):
    fault, report_pattern = PHASE_FAULTS[phase]
    specs = specs_for(num_ranks, required=required)
    specs[failing] = replace(specs[failing], **fault)
    group = make_group(specs).run()

    gathered = group.assert_collectives(phase)
    assert [report.ok for report in gathered[-1]] == [rank != failing for rank in range(num_ranks)]
    if phase >= 4:
        # The failing rank names exactly the corrupted tensor; the others found nothing.
        assert [report.data for report in gathered[-1]] == [
            [LAST_MOE_ANCHOR] if rank == failing else [] for rank in range(num_ranks)
        ]
    pattern = rf"abandoned at collective #{phase} .*: rank {failing}: {report_pattern}"
    if required:
        group.assert_all_failed(pattern)
    else:
        group.assert_all_none()
        for rank in group.ranks:
            (message,) = group.messages(rank, "error")
            assert re.search(pattern, message), message
    group.assert_restored()
    group.assert_released()
    group.assert_cache_emptied_after_release()
    assert all(len(rank.stores) == (phase >= 2) for rank in group.ranks)
    expected_launchers = [
        int(phase >= 4 or (phase == 3 and rank != failing)) for rank in range(num_ranks)
    ]
    assert [len(rank.launchers) for rank in group.ranks] == expected_launchers


#: Slot bytes ``(layer, segment, anchor, byte)`` to corrupt, with the paths Phase V reports (the
#: anchor) and the P5 self-check reports (every binding that views the byte).
CORRUPTIONS = {
    "last-byte-of-the-last-moe-anchor": (
        (COVERED[-1], MOE, -1, -1),
        [LAST_MOE_ANCHOR],
        [LAST_MOE_ANCHOR],
    ),
    "first-byte-of-an-aliased-mla-scale": (
        (3, ATTN, 2, 0),
        ["layers.3.self_attn.v_b_proj_scale"],
        ["layers.3.self_attn.v_b_proj_scale", "layers.3.self_attn.kv_b_proj.weight_scale"],
    ),
}


@pytest.mark.parametrize("phase", [4, 5])
@pytest.mark.parametrize("target", sorted(CORRUPTIONS))
def test_phase_v_and_the_self_check_name_every_corrupted_tensor(make_group, target, phase):
    where, anchors, bindings = CORRUPTIONS[target]
    specs = specs_for(2, required=False)
    specs[0] = replace(specs[0], corrupt=Corruption(phase, *where))
    group = make_group(specs).run()
    gathered = group.assert_collectives(phase)
    assert [report.data for report in gathered[-1]] == [anchors if phase == 4 else bindings, []]
    group.assert_all_none()
    group.assert_restored()
    group.assert_released()


@pytest.mark.parametrize(
    "fail,phase,report_pattern",
    [
        ("store", 2, r"RuntimeError: injected: the store constructor failed"),
        ("write", 2, r"RuntimeError: injected: store write failed"),
        ("runtime", 2, r"ValueError: dense DWDP needs the expert DWDP copy stream"),
        ("export", 2, r"RuntimeError: injected: store export failed"),
        ("launcher", 3, r"RuntimeError: injected: building the copy launcher failed"),
        ("p4_raise", 4, r"RuntimeError: injected: the copy of layer 1 failed"),
        ("p5_raise", 5, r"RuntimeError: injected: the copy of layer 1 failed"),
    ],
)
def test_whoever_owns_a_resource_releases_it_once(make_group, fail, phase, report_pattern):
    group = make_group([RankSpec(required=False), RankSpec(required=False, fail=fail)]).run()
    group.assert_collectives(phase)
    group.assert_all_none()
    (message,) = group.messages(group.ranks[1], "error")
    assert re.search(rf"collective #{phase} .*: rank 1: {report_pattern}", message), message
    group.assert_restored()
    group.assert_released()
    group.assert_cache_emptied_after_release()


def test_a_rank_that_cannot_restore_its_weights_raises_alone(make_group):
    group = make_group([RankSpec(required=False), RankSpec(required=False, fail="lose_buffer")])
    group.run()
    gathered = group.assert_collectives(5)
    assert [report.ok for report in gathered[-1]] == [True, False]
    intact, broken = group.ranks
    # Every rank abandons at the commit point; the rank whose rollback succeeded follows
    # TRTLLM_DWDP_DENSE_REQUIRED=0, the other raises because its modules are not intact.
    assert intact.error is None and intact.result is None, intact.trace
    intact.snapshot.assert_restored(intact.decoder)
    error_type, message = broken.error
    assert error_type is DenseSetupError, broken.trace
    assert re.search(
        r"abandoned at collective #5 .*: rank 1: DenseRepointError: .*no longer holds "
        rf"{LOST_MODULE}\.weight .*; this rank could not restore its original dense weights",
        message,
    ), message
    assert any("restoring the module tensors failed" in m for m in group.messages(broken, "error"))
    lost = f"layers.{COVERED[-1]}.{LOST_MODULE}.weight"
    assert broken.snapshot.changed(broken.decoder) == [lost]
    group.assert_released()
    group.assert_cache_emptied_after_release()


def test_a_base_exception_escapes_setup_after_the_release(make_group):
    group = make_group(specs_for(2, interrupt="write")).run()
    # The interrupt is not a report: no further collective, nothing abandoned.
    assert group.collective_counts() == [1, 1]
    for rank in group.ranks:
        assert rank.error == (KeyboardInterrupt, "injected: store write interrupted"), rank.trace
        assert rank.log[-2:] == ["write", "release"]
    group.assert_restored()
    group.assert_released()


def test_a_malformed_report_counts_as_a_failed_rank(make_group):
    specs = specs_for(2, required=False)
    specs[1] = replace(specs[1], malformed=3)
    group = make_group(specs).run()
    assert group.collective_counts() == [3, 3]
    group.assert_all_none()
    for rank in group.ranks:
        (message,) = group.messages(rank, "error")
        assert re.search(
            r"collective #3 .*: rank 1: sent a dict instead of a PhaseReport", message
        ), message
    group.assert_restored()
    group.assert_released()
    group.assert_cache_emptied_after_release()
