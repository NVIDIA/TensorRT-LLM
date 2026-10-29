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
"""CPU tests of repointing dense bindings into staging slots.

The K3-like toy decoder layers come from ``test_dense_layout.py``.  CPU slot rings stand in for
the staging slots, and a "gather" of ``L(l)`` copies the layer's blobs (as every rank's store rows
reassemble them) into the layer's slots, exactly as the runtime leaves them before ``l`` runs.
"""

import gc
import inspect
import weakref
from collections import namedtuple

import pytest
import torch
from test_dense_layout import (
    BF16,
    COVERED,
    F_ROWS,
    FP8,
    HIDDEN,
    ROWS,
    ToyModel,
    collect,
    make_config,
    storage_ptr,
    toy_input,
)
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import layout as dense_layout
from tensorrt_llm._torch.modules.dwdp.dense import repoint as dense_repoint
from tensorrt_llm._torch.modules.dwdp.dense.layout import compute_checksums, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.repoint import (
    BindingState,
    DenseBindings,
    install_binding,
    read_binding,
    slot_view,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    ROW_ALIGN,
    AnchorRecord,
    BindingKind,
    DenseRepointError,
    SegmentKind,
    SlotBuffer,
    SlotSpec,
    TensorBinding,
    allocate_slot_ring,
)

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Offset of the misaligned slot allocations from a ``ROW_ALIGN`` boundary.
MISALIGNMENT = 192


def misaligned_ring(spec):
    """Slots whose allocations start ``MISALIGNMENT`` bytes past a 4 KiB boundary.

    CPU allocations of slot size tend to be page aligned, which would leave ``pad == 0``; these
    slots need ``pad = ROW_ALIGN - MISALIGNMENT``, so every view must honour it.
    """
    ring = []
    for index in range(spec.depth):
        raw = bytearray(spec.alloc_bytes + ROW_ALIGN)
        shift = (MISALIGNMENT - torch.frombuffer(raw, dtype=torch.uint8).data_ptr()) % ROW_ALIGN
        storage = torch.frombuffer(raw, dtype=torch.uint8, offset=shift, count=spec.alloc_bytes)
        assert storage.storage_offset() == 0
        assert storage.data_ptr() % ROW_ALIGN == MISALIGNMENT
        ring.append(SlotBuffer(spec.segment, index, storage, ROW_ALIGN - MISALIGNMENT))
    return tuple(ring)


def setup_bindings(model, config=None, *, allocate=misaligned_ring):
    """P1 (collect, checksums) and the slot rings, as setup prepares them before P5."""
    layout = collect(model, config)
    compute_checksums(layout)
    slots = {kind: allocate(layout.slot_specs[kind]) for kind in layout.segment_kinds}
    return layout, slots, DenseBindings(layout, slots)


def allocated_ring(spec):
    """The production allocation (``types.allocate_slot_ring``) on the CPU."""
    return allocate_slot_ring(spec, "cpu")


def store_blobs(layout):
    """Every rank's rows of every ``(layer, segment)``, reassembled; outlives the originals."""
    blobs = {}
    for key, segment in layout.segments.items():
        blob = torch.empty(layout.num_ranks * segment.row_bytes, dtype=torch.uint8)
        write_blob(segment, blob, num_ranks=layout.num_ranks)
        blobs[key] = blob
    return blobs


def slot_of(layout, slots, layer, kind):
    return slots[kind][layout.slot_index(kind, layer)]


def gather(layout, slots, blobs, layer):
    """Leave ``L(layer)`` in its slots, as the runtime's gather does."""
    for segment in layout.layer_segments(layer):
        blob = blobs[segment.key]
        slot_of(layout, slots, layer, segment.segment).blob_view(blob.numel()).copy_(blob)


def layer_outputs(model, x, *, before_layer=None):
    outputs = []
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            if before_layer is not None:
                before_layer(index)
            x = layer(x)
            outputs.append(x)
    return outputs


def dense_outputs(model, layout, slots, blobs, x):
    def before_layer(index):
        if layout.covers(index):
            gather(layout, slots, blobs, index)

    return layer_outputs(model, x, before_layer=before_layer)


def exposed_state(model):
    """What every module exposes: object identities and storage placement, no tensors kept."""

    def describe(tensor):
        if tensor is None:
            return None
        return (
            id(tensor),
            type(tensor).__name__,
            tensor.dtype,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.stride(),
        )

    state = {}
    for path, module in model.named_modules():
        for name, param in module._parameters.items():
            state[(path, "param", name)] = describe(param)
        for name, buffer in module._buffers.items():
            state[(path, "buffer", name)] = describe(buffer)
        state[(path, "persistent")] = tuple(sorted(module._non_persistent_buffers_set))
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                state[(path, "attr", name)] = describe(value)
            elif isinstance(value, (tuple, list)) and value:
                if all(isinstance(item, torch.Tensor) for item in value):
                    state[(path, "seq", name)] = (type(value), tuple(map(describe, value)))
    return state


def all_bindings(layout):
    for anchor in layout.anchors():
        for binding in anchor.bindings:
            yield anchor, binding


def bindings_of(layout, layer):
    return [
        binding
        for segment in layout.layer_segments(layer)
        for anchor in segment.anchors
        for binding in anchor.bindings
    ]


# ---------------------------------------------------------------------------
# Repoint, self-check, rollback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("allocate", [misaligned_ring, allocated_ring])
def test_forward_is_bit_identical_after_repointing_to_slots(allocate):
    model = ToyModel()
    x = toy_input()
    reference = layer_outputs(model, x)
    layout, slots, bindings = setup_bindings(model, allocate=allocate)
    blobs = store_blobs(layout)

    bindings.repoint()
    assert bindings.state is BindingState.REPOINTED
    for anchor, binding in all_bindings(layout):
        slot = slot_of(layout, slots, anchor.layer, anchor.segment)
        assert storage_ptr(read_binding(binding)) == storage_ptr(slot.storage)

    # Two forwards: the rings wrap from layer 5 back to layer 1.
    for _ in range(2):
        outputs = dense_outputs(model, layout, slots, blobs, x)
        for got, want in zip(outputs, reference, strict=True):
            assert torch.equal(got, want)

    for layer in COVERED:
        gather(layout, slots, blobs, layer)
        assert bindings.verify_layer(layer) == []

    assert bindings.commit(strict=True) == []
    outputs = dense_outputs(model, layout, slots, blobs, x)
    for got, want in zip(outputs, reference, strict=True):
        assert torch.equal(got, want)


def test_aliases_keep_their_offsets():
    model = ToyModel()
    layout, slots, bindings = setup_bindings(model, make_config(min_bytes=0))
    before = {
        binding: (tuple(binding.size), binding.stride, binding.byte_offset)
        for _, binding in all_bindings(layout)
    }
    bindings.repoint()

    for anchor, binding in all_bindings(layout):
        slot = slot_of(layout, slots, anchor.layer, anchor.segment)
        tensor = read_binding(binding)
        assert slot.pad == ROW_ALIGN - MISALIGNMENT and slot.base % ROW_ALIGN == 0
        # Placement as storage plus offset: data_ptr() is 0 for the empty view.
        assert storage_ptr(tensor) == storage_ptr(slot.storage)
        start = tensor.storage_offset() * tensor.element_size()
        assert start == slot.pad + anchor.blob_offset + binding.byte_offset
        assert (tuple(tensor.shape), tensor.stride(), binding.byte_offset) == before[binding]
        assert (slot.storage.data_ptr() + start) % binding.dtype.itemsize == 0

    kda = model.layers[1].linear_attn
    fused = kda.qkvg_proj.weight.data_ptr()
    rows = [getattr(kda, f"{name}_proj").weight.data_ptr() - fused for name in "qkvg"]
    assert rows == [0, ROWS * HIDDEN, 2 * ROWS * HIDDEN, 3 * ROWS * HIDDEN]
    bfa = kda._bfa_proj_weight
    assert kda.f_a_proj.weight.data_ptr() == bfa.data_ptr()
    assert kda.b_proj.weight.data_ptr() == bfa.data_ptr() + F_ROWS * HIDDEN * 2
    stride_0, empty = kda._bfa_views
    assert (stride_0.data_ptr(), stride_0.stride()) == (bfa.data_ptr(), (0, 1))
    assert empty.numel() == 0 and empty.storage_offset() == bfa.storage_offset() + bfa.numel()
    assert kda._gate_b_t.data_ptr() == kda.f_b_proj.weight.data_ptr()
    assert kda._gate_b_t.stride() == (HIDDEN // 2 * F_ROWS, 1, F_ROWS)
    assert kda._o_dense.data_ptr() == kda.o_proj.weight.data_ptr() + 64 * HIDDEN
    assert kda._A_log_f32.data_ptr() == kda.A_log.data_ptr()
    assert kda._dt_bias_f32.data_ptr() == kda.dt_bias.data_ptr()
    mla = model.layers[3].self_attn
    assert mla.v_b_proj_scale.data_ptr() == mla.kv_b_proj.weight_scale.data_ptr()

    # Layers sharing a ring slot see the same addresses: A slot 0 holds layers 1, 3 and 5, D slot
    # 0 holds layers 1 and 4.
    attn_slot = slot_of(layout, slots, 1, ATTN)
    for layer in (1, 3, 5):
        assert slot_of(layout, slots, layer, ATTN) is attn_slot
    assert model.layers[5].linear_attn.qkvg_proj.weight.data_ptr() == fused
    assert storage_ptr(mla.o_proj.weight) == storage_ptr(attn_slot.storage)
    gate = model.layers[1].block_sparse_moe.gate.weight
    assert model.layers[4].block_sparse_moe.gate.weight.data_ptr() == gate.data_ptr()


def test_parameter_identity_and_buffer_registration_are_preserved():
    model = ToyModel()
    parameters = {name: id(param) for name, param in model.named_parameters()}
    buffers = [name for name, _ in model.named_buffers()]
    persistent = {path: set(m._non_persistent_buffers_set) for path, m in model.named_modules()}
    state_keys = list(model.state_dict())
    layout, slots, bindings = setup_bindings(model)

    bindings.repoint()
    assert {name: id(param) for name, param in model.named_parameters()} == parameters
    assert all(type(param) is nn.Parameter for param in model.parameters())
    assert [name for name, _ in model.named_buffers()] == buffers
    assert {path: set(m._non_persistent_buffers_set) for path, m in model.named_modules()} == (
        persistent
    )
    assert list(model.state_dict()) == state_keys

    kda = model.layers[2].linear_attn
    attn_slot = slot_of(layout, slots, 2, ATTN)
    assert kda.qkvg_proj._buffers["weight"] is kda.qkvg_proj.weight
    assert storage_ptr(kda.qkvg_proj.weight) == storage_ptr(attn_slot.storage)
    assert storage_ptr(kda.f_a_proj.weight) == storage_ptr(attn_slot.storage)
    assert not kda.f_a_proj.weight.requires_grad
    assert "_bfa_proj_weight" in vars(kda) and "_bfa_proj_weight" not in kda._buffers
    for container, kind in ((kda._mtp_conv_weights, tuple), (kda._bfa_views, list)):
        assert type(container) is kind
        assert all(storage_ptr(t) == storage_ptr(attn_slot.storage) for t in container)


def test_rollback_restores_the_original_state():
    model = ToyModel()
    x = toy_input()
    reference = layer_outputs(model, x)
    original = exposed_state(model)
    layout, slots, bindings = setup_bindings(model)
    blobs = store_blobs(layout)

    bindings.rollback()  # a no-op before repointing
    assert bindings.state is BindingState.PENDING
    bindings.repoint()
    assert exposed_state(model) != original
    bindings.rollback()
    assert bindings.state is BindingState.ROLLED_BACK
    assert exposed_state(model) == original
    for got, want in zip(layer_outputs(model, x), reference, strict=True):
        assert torch.equal(got, want)
    bindings.rollback()  # a no-op when rolled back
    assert exposed_state(model) == original

    # Rolled back -> repointed again works, and so does a second rollback.
    bindings.repoint()
    for got, want in zip(dense_outputs(model, layout, slots, blobs, x), reference, strict=True):
        assert torch.equal(got, want)
    bindings.rollback()
    assert exposed_state(model) == original


def test_verify_layer_reads_through_the_module_attributes():
    model = ToyModel()
    reference = layer_outputs(model, toy_input())
    layout, slots, bindings = setup_bindings(model)
    blobs = store_blobs(layout)
    bindings.repoint()
    empty = [binding for binding in bindings_of(layout, 1) if binding.span_bytes == 0]
    assert [binding.qualified_name for binding in empty] == ["linear_attn._bfa_views[1]"]
    reads = sorted(
        f"layers.1.{binding.qualified_name}"
        for binding in bindings_of(layout, 1)
        if binding.span_bytes
    )

    # A slot 0 holds L(3)'s attention bytes and D slot 0 nothing yet: every binding that reads
    # bytes mismatches.
    gather(layout, slots, blobs, 3)
    assert sorted(bindings.verify_layer(1)) == reads
    with torch.no_grad():
        assert not torch.equal(model.layers[1](reference[0]), reference[1])

    # Every binding matches, including the empty view and the strided ones.
    gather(layout, slots, blobs, 1)
    assert bindings.verify_layer(1) == []
    with torch.no_grad():
        assert torch.equal(model.layers[1](reference[0]), reference[1])

    # An empty view reads no bytes, so only its placement can be wrong.
    kda = model.layers[1].linear_attn
    views = kda._bfa_views
    object.__setattr__(kda, "_bfa_views", [views[0], kda._bfa_proj_weight[3:3]])
    assert bindings.verify_layer(1) == ["layers.1.linear_attn._bfa_views[1]"]
    object.__setattr__(kda, "_bfa_views", views)
    assert bindings.verify_layer(1) == []

    # One corrupted slot byte is attributed to every binding that reads it.
    f_b = next(b for b in bindings_of(layout, 1) if b.qualified_name.endswith("f_b_proj.weight"))
    f_b_anchor = next(a for a in layout.segment(1, ATTN).anchors if f_b in a.bindings)
    slot_of(layout, slots, 1, ATTN).blob_view(layout.segment(1, ATTN).blob_bytes)[
        f_b_anchor.blob_offset + 100
    ] ^= 0x01
    assert bindings.verify_layer(1) == [
        "layers.1.linear_attn._gate_b_t",
        "layers.1.linear_attn.f_b_proj.weight",
    ]

    # A binding that no longer views its slot fails even with the right bytes.
    gather(layout, slots, blobs, 1)
    object.__setattr__(kda, "_bfa_proj_weight", kda._bfa_proj_weight.clone())
    assert bindings.verify_layer(1) == ["layers.1.linear_attn._bfa_proj_weight"]

    with pytest.raises(DenseRepointError, match="not covered"):
        bindings.verify_layer(0)


def test_failed_repoint_restores_what_it_installed():
    model = ToyModel()
    layout, slots, bindings = setup_bindings(model)
    del model.layers[4].block_sparse_moe.shared_experts.down_proj._buffers["weight_scale"]
    before = exposed_state(model)
    with pytest.raises(DenseRepointError, match="no longer holds"):
        bindings.repoint()
    assert bindings.state is BindingState.PENDING
    assert exposed_state(model) == before

    # Slots too small for the layout fail before anything is installed.
    model = ToyModel()
    layout = collect(model)
    compute_checksums(layout)
    small = {
        kind: allocate_slot_ring(SlotSpec(kind, layout.slot_specs[kind].depth, ROW_ALIGN, 2), "cpu")
        for kind in layout.segment_kinds
    }
    bindings = DenseBindings(layout, small)
    before = exposed_state(model)
    with pytest.raises(DenseRepointError, match="leaves"):
        bindings.repoint()
    assert bindings.state is BindingState.PENDING
    assert exposed_state(model) == before


# ---------------------------------------------------------------------------
# Commit: dropping the originals and the release check
# ---------------------------------------------------------------------------


def test_commit_releases_the_old_storages():
    model = ToyModel()
    kda = model.layers[1].linear_attn
    stays = [
        weakref.ref(tensor.untyped_storage())
        for tensor in (
            kda.A_log,
            kda._packed_conv_weight,
            model.layers[3].self_attn._fused_q_placeholder,
            model.layers[1].block_sparse_moe.routed_expert_norm.weight,
            model.layers[0].linear_attn.qkvg_proj.weight,
        )
    ]
    layout, slots, bindings = setup_bindings(model)
    moved = [weakref.ref(anchor.source.untyped_storage()) for anchor in layout.anchors()]
    assert len(moved) == 5 * 7 + 4 * 13 + 10

    bindings.repoint()
    assert all(ref() is not None for ref in moved)  # alive until the commit point
    assert bindings.commit(strict=True) == []
    assert bindings.state is BindingState.COMMITTED
    assert all(ref() is None for ref in moved)
    assert all(ref() is not None for ref in stays)
    for anchor in layout.anchors():
        assert anchor.source is None
        assert all(binding.original is None for binding in anchor.bindings)
    # The checksums stay for the runtime's diagnostics.
    assert all(binding.checksum is not None for _, binding in all_bindings(layout))


def test_commit_collects_garbage_cycles_before_reporting():
    """An original that only unreachable garbage still holds is not a leak."""
    model = ToyModel()
    layout, slots, bindings = setup_bindings(model)
    moved = [weakref.ref(anchor.source.untyped_storage()) for anchor in layout.anchors()]
    enabled = gc.isenabled()
    gc.disable()  # only commit's own gc.collect() may free the cycle
    try:
        cycle = [model.layers[2].linear_attn.f_b_proj.weight.data]
        cycle.append(cycle)
        del cycle
        bindings.repoint()
        assert bindings.commit(strict=True) == []
    finally:
        if enabled:
            gc.enable()
    assert all(ref() is None for ref in moved)


class _CaptureLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize("strict, describe", [(False, False), (False, True), (True, True)])
def test_hidden_holder_is_reported_with_its_path(monkeypatch, strict, describe):
    capture = _CaptureLogger()
    monkeypatch.setattr(dense_repoint, "logger", capture)
    model = ToyModel()
    kda = model.layers[2].linear_attn
    hidden = {"stash": kda.f_b_proj.weight.data}
    mla = model.layers[3].self_attn
    mla.mha.cached_absorb = mla.k_b_proj_trans.data  # a non-module object the walk never enters
    layout, slots, bindings = setup_bindings(model)
    survivor = weakref.ref(hidden["stash"].untyped_storage())
    bindings.repoint()

    # An anchor is named after its first binding; the report lists all of them.
    expected = ["layers.2.linear_attn._gate_b_t", "layers.3.self_attn.k_b_proj_trans"]
    if strict:
        with pytest.raises(DenseRepointError, match="still alive after commit") as error:
            bindings.commit(strict=True, describe_referrers=describe)
        assert all(path in str(error.value) for path in expected)
    else:
        assert bindings.commit(strict=False, describe_referrers=describe) == expected
    assert bindings.state is BindingState.COMMITTED
    assert len(capture.warnings) == 2
    for path, warning in zip(expected, capture.warnings, strict=True):
        assert f"storage of {path} " in warning
    assert "bindings (linear_attn._gate_b_t, linear_attn.f_b_proj.weight)" in capture.warnings[0]
    assert ("dict key 'stash'" in capture.warnings[0]) is describe
    assert ("holders:" in capture.warnings[1]) is describe
    # Leaking costs memory only: the module reads the slot, the holder keeps an old copy.
    assert survivor() is not None
    assert storage_ptr(kda.f_b_proj.weight) == storage_ptr(slot_of(layout, slots, 2, ATTN).storage)
    assert storage_ptr(hidden["stash"]) != storage_ptr(kda.f_b_proj.weight)


@pytest.mark.parametrize("strict", [False, True], ids=["warn", "strict"])
@pytest.mark.parametrize("describe", [False, True], ids=["plain", "describe"])
def test_a_leak_in_every_layer_names_a_few_storages_and_counts_the_rest(
    monkeypatch, strict, describe
):
    """Every original survives: a few warnings (and heap scans), one summary, every path returned.

    The holder description scans every live object, so only the first ``_MAX_LISTED`` leaks get
    one; the rest are counted, with their bytes and layers, in one summary warning.
    """
    capture = _CaptureLogger()
    monkeypatch.setattr(dense_repoint, "logger", capture)
    described = []
    describe_holders = dense_repoint._describe_holders

    def counting_describe(storage_ref):
        described.append(storage_ref)
        return describe_holders(storage_ref)

    monkeypatch.setattr(dense_repoint, "_describe_holders", counting_describe)
    model = ToyModel()
    layout, slots, bindings = setup_bindings(model)
    anchors = [
        anchor
        for layer in layout.layers
        for segment in layout.layer_segments(layer)
        for anchor in segment.anchors
    ]
    limit = dense_repoint._MAX_LISTED
    assert len(anchors) > limit + 1
    held = [anchor.source for anchor in anchors]  # a holder outside the bindings, for every one
    bindings.repoint()

    paths = [anchor.path for anchor in anchors]
    if strict:
        with pytest.raises(DenseRepointError, match="still alive after commit") as error:
            bindings.commit(strict=True, describe_referrers=describe)
        assert f"and {len(anchors) - limit} more" in str(error.value)
    else:
        assert bindings.commit(strict=False, describe_referrers=describe) == paths
    assert bindings.state is BindingState.COMMITTED
    assert len(described) == (limit if describe else 0)
    assert len(capture.warnings) == limit + 1
    for path, warning in zip(paths, capture.warnings[:limit]):
        assert f"storage of {path} " in warning
        assert ("; holders: " in warning) is describe
    rest = anchors[limit:]
    summary = capture.warnings[-1]
    assert f"{len(rest)} more moved storages ({sum(a.nbytes for a in rest)} bytes)" in summary
    total = sum(anchor.nbytes for anchor in anchors)
    assert f"{len(anchors)} storages, {total} bytes in all" in summary
    layers = dense_repoint._layer_ranges(sorted({anchor.layer for anchor in rest}))
    kinds = ",".join(kind.value for kind in SegmentKind if kind in {a.segment for a in rest})
    assert f"in segments {kinds} of layers {layers}: " in summary
    assert all(path not in summary for path in paths)
    del held


def test_layer_ranges_compress_runs():
    assert dense_repoint._layer_ranges([]) == ""
    assert dense_repoint._layer_ranges([4]) == "4"
    assert dense_repoint._layer_ranges([1, 2, 3, 5, 7, 8]) == "1..3, 5, 7..8"


def test_resize_is_never_called(monkeypatch):
    calls = []

    def forbidden(name):
        def record(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called")

        return record

    for owner, name in (
        (torch.UntypedStorage, "resize_"),
        (torch.TypedStorage, "resize_"),
        (torch.Tensor, "resize_"),
        (torch.Tensor, "resize_as_"),
    ):
        monkeypatch.setattr(owner, name, forbidden(f"{owner.__name__}.{name}"))

    model = ToyModel()
    x = toy_input()
    reference = layer_outputs(model, x)
    layout, slots, bindings = setup_bindings(model)
    blobs = store_blobs(layout)
    bindings.repoint()
    for layer in COVERED:
        gather(layout, slots, blobs, layer)
        assert bindings.verify_layer(layer) == []
    bindings.rollback()
    bindings.repoint()
    assert bindings.commit(strict=True) == []
    for got, want in zip(dense_outputs(model, layout, slots, blobs, x), reference, strict=True):
        assert torch.equal(got, want)
    bindings.detach()
    assert calls == []
    for module in (dense_layout, dense_repoint):
        assert "resize_" not in inspect.getsource(module)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_detach_points_every_binding_at_an_empty_tensor():
    model = ToyModel()
    original = exposed_state(model)
    parameters = {name: id(param) for name, param in model.named_parameters()}
    layout, slots, bindings = setup_bindings(model)

    bindings.detach()  # PENDING: nothing to do
    assert bindings.state is BindingState.PENDING and exposed_state(model) == original
    bindings.repoint()
    bindings.detach()  # REPOINTED: the originals are alive, so roll back instead
    assert bindings.state is BindingState.ROLLED_BACK and exposed_state(model) == original
    bindings.detach()  # ROLLED_BACK: nothing to do
    assert exposed_state(model) == original

    bindings.repoint()
    assert bindings.commit(strict=True) == []
    slot_storages = [
        weakref.ref(slot.storage.untyped_storage()) for ring in slots.values() for slot in ring
    ]
    bindings.detach()
    assert bindings.state is BindingState.DETACHED
    for _, binding in all_bindings(layout):
        tensor = read_binding(binding)
        assert tensor.numel() == 0 and tensor.dtype == binding.dtype
        assert tensor.device == torch.device("cpu")
    assert {name: id(param) for name, param in model.named_parameters()} == parameters
    kda = model.layers[1].linear_attn
    assert "weight" in kda.qkvg_proj._buffers and kda.qkvg_proj.weight.numel() == 0
    assert type(kda._mtp_conv_weights) is tuple and len(kda._mtp_conv_weights) == 3
    assert type(kda._bfa_views) is list and len(kda._bfa_views) == 2
    assert kda.A_log.numel() == 16  # below the threshold: never repointed
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model.layers[1](toy_input())
    bindings.detach()  # idempotent
    assert bindings.state is BindingState.DETACHED

    # No module view keeps a slot alive: dropping the rings frees them.
    del bindings, slots
    gc.collect()
    assert all(ref() is None for ref in slot_storages)


def committed_bindings(model):
    layout, slots, bindings = setup_bindings(model, allocate=allocated_ring)
    bindings.repoint()
    assert bindings.commit(strict=True) == []
    return layout, slots, bindings


def test_detach_counts_bindings_of_collected_modules_as_detached():
    """A model collected before teardown: no module can read its bindings, so detach succeeds."""
    model = ToyModel()
    layout, slots, bindings = committed_bindings(model)
    model_ref = weakref.ref(model)
    del model
    gc.collect()
    assert model_ref() is None
    assert all(binding.owner_ref() is None for _, binding in all_bindings(layout))
    slot_storages = [
        weakref.ref(slot.storage.untyped_storage()) for ring in slots.values() for slot in ring
    ]

    bindings.detach()

    assert bindings.state is BindingState.DETACHED
    bindings.detach()  # idempotent
    del bindings, slots
    gc.collect()
    assert all(ref() is None for ref in slot_storages)


def test_detach_raises_only_for_live_owners_that_lost_their_binding():
    """Collected modules are skipped, live ones detached; a live owner without its binding fails.

    The failure lists that binding alone, after every other binding was attempted.
    """
    model = ToyModel()
    layout, slots, bindings = committed_bindings(model)
    replaced = weakref.ref(model.layers[3])
    model.layers[3] = nn.Identity()  # the MLA layer's modules are collected
    gc.collect()
    assert replaced() is None
    del model.layers[2].linear_attn._bfa_proj_weight

    with pytest.raises(
        DenseRepointError,
        match=r"^detach could not reach every binding: layers\.2\.linear_attn\._bfa_proj_weight: "
        r"ToyKDA at linear_attn no longer holds linear_attn\._bfa_proj_weight as a attr$",
    ):
        bindings.detach()

    assert bindings.state is BindingState.DETACHED
    collected = 0
    for anchor, binding in all_bindings(layout):
        if anchor.layer == 3:
            assert binding.owner_ref() is None
            collected += 1
        elif binding.name != "_bfa_proj_weight":
            tensor = read_binding(binding)
            assert tensor.numel() == 0 and tensor.dtype == binding.dtype, binding.qualified_name
    assert collected > 0
    assert not hasattr(model.layers[2].linear_attn, "_bfa_proj_weight")
    bindings.detach()  # idempotent: nothing is attempted again


def test_state_machine_rejects_out_of_order_calls():
    model = ToyModel()
    layout, slots, bindings = setup_bindings(model)
    with pytest.raises(DenseRepointError, match="verify_layer"):
        bindings.verify_layer(1)
    with pytest.raises(DenseRepointError, match="commit"):
        bindings.commit(strict=False)
    bindings.repoint()
    with pytest.raises(DenseRepointError, match="repoint"):
        bindings.repoint()
    bindings.commit(strict=False)
    for call in (
        bindings.repoint,
        bindings.rollback,
        lambda: bindings.verify_layer(1),
        lambda: bindings.commit(strict=False),
    ):
        with pytest.raises(DenseRepointError, match="committed"):
            call()
    bindings.detach()
    with pytest.raises(DenseRepointError, match="detached"):
        bindings.rollback()

    # A layout whose originals were dropped cannot be bound again.
    with pytest.raises(DenseRepointError, match="already dropped"):
        DenseBindings(layout, slots)


def test_bindings_check_the_rings():
    model = ToyModel()
    layout = collect(model)
    rings = {kind: allocate_slot_ring(layout.slot_specs[kind], "cpu") for kind in (ATTN, MOE)}
    with pytest.raises(ValueError, match="moe ring needs 3 slots"):
        DenseBindings(layout, {ATTN: rings[ATTN]})
    with pytest.raises(ValueError, match="attn ring needs 2 slots"):
        DenseBindings(layout, {ATTN: rings[ATTN][:1], MOE: rings[MOE]})
    with pytest.raises(ValueError, match="slot 0 of the attn ring"):
        DenseBindings(layout, {ATTN: rings[ATTN][::-1], MOE: rings[MOE]})
    # Plain segment names work as ring keys.
    assert DenseBindings(layout, {"attn": rings[ATTN], "moe": rings[MOE]}).state is (
        BindingState.PENDING
    )
    with pytest.raises(DenseRepointError, match="verify_layer"):
        DenseBindings(layout, rings).verify_layer(1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
        self.register_buffer("scale", torch.arange(4, dtype=torch.int32), persistent=False)
        self.pair = [torch.zeros(4, dtype=BF16), torch.ones(4, dtype=BF16)]
        self.named = _Pair(torch.zeros(2, dtype=FP8), torch.ones(2, dtype=FP8))


_Pair = namedtuple("_Pair", ["first", "second"])


def _binding_for(owner, kind, name, index, tensor):
    return TensorBinding(
        module_path="holder",
        kind=kind,
        name=name,
        index=index,
        dtype=tensor.dtype,
        size=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        byte_offset=tensor.storage_offset() * tensor.element_size(),
        owner_ref=weakref.ref(owner),
        original=tensor,
    )


def test_install_and_read_every_binding_kind():
    holder = _Holder()
    weight = holder.weight
    cases = [
        (BindingKind.PARAMETER, "weight", None, holder.weight.data),
        (BindingKind.BUFFER, "scale", None, holder.scale),
        (BindingKind.SEQUENCE_ITEM, "pair", 1, holder.pair[1]),
        (BindingKind.SEQUENCE_ITEM, "named", 0, holder.named.first),
    ]
    for kind, name, index, tensor in cases:
        binding = _binding_for(holder, kind, name, index, tensor)
        replacement = (tensor.float() + 1).to(tensor.dtype)
        install_binding(binding, replacement)
        assert torch.equal(read_binding(binding).view(torch.uint8), replacement.view(torch.uint8))
    assert holder.weight is weight and type(holder.weight) is nn.Parameter
    assert "scale" in holder._buffers and "scale" in holder._non_persistent_buffers_set
    assert type(holder.pair) is list and holder.pair[0].dtype == BF16
    assert type(holder.named) is _Pair
    assert torch.equal(holder.named.second.float(), torch.ones(2))

    holder._bfa_like = torch.zeros(6, dtype=BF16)
    attribute = _binding_for(holder, BindingKind.ATTRIBUTE, "_bfa_like", None, holder._bfa_like)
    install_binding(attribute, torch.ones(6, dtype=BF16))
    assert torch.equal(holder._bfa_like, torch.ones(6, dtype=BF16))

    with pytest.raises(DenseRepointError, match="cannot install a torch.float32"):
        install_binding(attribute, torch.ones(6))
    del holder._bfa_like
    with pytest.raises(DenseRepointError, match="no longer holds"):
        install_binding(attribute, torch.ones(6, dtype=BF16))
    with pytest.raises(DenseRepointError, match="no longer holds"):
        read_binding(attribute)
    del holder, weight
    gc.collect()
    with pytest.raises(DenseRepointError, match="garbage collected"):
        read_binding(attribute)


def test_slot_view_formula_and_bounds():
    storage = torch.zeros(8 * ROW_ALIGN, dtype=torch.uint8)
    slot = SlotBuffer(ATTN, 0, storage, (-storage.data_ptr()) % ROW_ALIGN)
    owner = nn.Module()
    source = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    view = source[8:12, 4:]
    binding = _binding_for(owner, BindingKind.ATTRIBUTE, "w", None, view)
    anchor = AnchorRecord(
        layer=1,
        segment=ATTN,
        index=0,
        nbytes=source.untyped_storage().nbytes(),
        blob_offset=ROW_ALIGN,
        bindings=(binding,),
    )
    raw = torch.empty(0, dtype=torch.uint8).set_(source.untyped_storage())
    slot.blob_view(ROW_ALIGN + raw.numel())[ROW_ALIGN:].copy_(raw)

    repointed = slot_view(binding, anchor, slot)
    assert torch.equal(repointed, view)
    assert repointed.stride() == view.stride()
    assert repointed.data_ptr() == slot.base + ROW_ALIGN + view.storage_offset() * 4
    assert storage_ptr(repointed) == storage_ptr(storage)

    misaligned = _binding_for(owner, BindingKind.ATTRIBUTE, "w", None, view)
    misaligned.byte_offset += 2
    with pytest.raises(DenseRepointError, match="not a multiple of its itemsize"):
        slot_view(misaligned, anchor, slot)
    beyond = _binding_for(owner, BindingKind.ATTRIBUTE, "w", None, source)
    beyond.byte_offset = 4
    with pytest.raises(DenseRepointError, match="leaves"):
        slot_view(beyond, anchor, slot)
    tight = SlotBuffer(ATTN, 0, storage[: slot.pad + ROW_ALIGN + 1024], slot.pad)
    with pytest.raises(DenseRepointError, match="leaves"):
        slot_view(binding, anchor, tight)
    shaped = SlotBuffer(ATTN, 0, storage.view(8, ROW_ALIGN), slot.pad)
    with pytest.raises(DenseRepointError, match="flat uint8"):
        slot_view(binding, anchor, shaped)
