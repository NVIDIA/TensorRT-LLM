# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA checkpoint layout, MR8 flat views, seed / commit and slot indices on CPU tensors."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SOURCE = (
    Path(__file__).resolve().parents[7]
    / "tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl"
    / "ops/cache/k3_ull_kda_checkpoints.py"
)
_spec = importlib.util.spec_from_file_location("_k3_ull_kda_checkpoints_test", _SOURCE)
ck = importlib.util.module_from_spec(_spec)
# dataclasses resolve the defining module through sys.modules while the module executes.
sys.modules[_spec.name] = ck
_spec.loader.exec_module(ck)
pytestmark = pytest.mark.cpu_only

HEADS, DIM, NUM_SPEC, WINDOW = 2, 8, 3, 3
SLOTS = 5
INT32_MAX = 2**31 - 1


def _layout(state_dtype: torch.dtype, heads: int = HEADS, dim: int = DIM):
    return ck.kda_checkpoint_layout(NUM_SPEC, heads, dim, WINDOW, state_dtype, torch.bfloat16)


def _slot_buffer(payload: int, slot_stride: int, misalign: int = 0, slots: int = SLOTS):
    """``[slots, payload]`` uint8 rows ``slot_stride`` bytes apart, starting ``misalign`` bytes
    past a 256-byte boundary; the bytes between payloads are filled with 0xAB."""
    backing = torch.full((slots * slot_stride + 512 + misalign,), 0xAB, dtype=torch.uint8)
    start = (-backing.data_ptr()) % 256 + misalign
    rows = backing[start : start + slots * slot_stride].view(slots, slot_stride)
    rows[:, :payload].zero_()
    return backing, rows, rows[:, :payload]


def _expected_unit(slot_stride: int, address: int, state_bytes: int) -> int:
    unit = 256
    while unit > 16 and (slot_stride % unit or address % unit or state_bytes % unit):
        unit //= 2
    return unit


def _pools(layout, generator: torch.Generator):
    h, v, k = layout.state_shape
    ssm = torch.randn(SLOTS, h, v, k, generator=generator, dtype=torch.float32)
    conv = torch.randn(SLOTS, layout.conv_dim, WINDOW, generator=generator).to(torch.bfloat16)
    return ssm, conv


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_layout_bytes(state_dtype: torch.dtype) -> None:
    layout = _layout(state_dtype)
    state_bytes = HEADS * DIM * DIM * state_dtype.itemsize
    conv_bytes = 3 * HEADS * DIM * (WINDOW + NUM_SPEC) * 2
    assert layout.num_checkpoints == 1 + NUM_SPEC
    assert layout.state_bytes == state_bytes
    assert layout.conv_offset == 256 + (1 + NUM_SPEC) * state_bytes
    assert layout.nbytes % 256 == 0
    assert layout.conv_offset + conv_bytes <= layout.nbytes < layout.conv_offset + conv_bytes + 256
    assert ck.estimate_checkpoint_bytes_per_slot(layout, 3) == 3 * layout.nbytes + 256
    assert ck.estimate_checkpoint_bytes_per_slot(layout, 0) == 0


@pytest.mark.parametrize(
    ("state_dtype", "nbytes"), [(torch.bfloat16, 1_619_200), (torch.float32, 3_192_064)]
)
def test_layout_bytes_of_the_deployment(state_dtype: torch.dtype, nbytes: int) -> None:
    """Helix CP16 KDA: 6 local heads of 128, q = 1 + 7, 4-tap convolution."""
    layout = ck.kda_checkpoint_layout(7, 6, 128, 3, state_dtype, torch.bfloat16)
    assert layout.nbytes == nbytes
    assert layout.conv_dim == 2304 and layout.conv_width == 10


@pytest.mark.parametrize(
    ("state_dtype", "conv_dtype"),
    [(torch.float16, torch.bfloat16), (torch.float32, torch.float32)],
)
def test_layout_rejects_unsupported_dtypes(state_dtype, conv_dtype) -> None:
    with pytest.raises(TypeError):
        ck.kda_checkpoint_layout(NUM_SPEC, HEADS, DIM, WINDOW, state_dtype, conv_dtype)


# (slot stride beyond the payload, misalignment of the buffer)
_PADDINGS = [(0, 0), (64, 0), (16, 0), (256, 32), (512, 16)]


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("extra", "misalign"), _PADDINGS)
def test_views_alias_one_storage(state_dtype: torch.dtype, extra: int, misalign: int) -> None:
    layout = _layout(state_dtype)
    stride = layout.nbytes + extra
    _, _, raw = _slot_buffer(layout.nbytes, stride, misalign)
    views = ck.kda_checkpoint_views(raw, layout)
    unit = _expected_unit(stride, raw.data_ptr(), layout.state_bytes)
    assert views.unit_bytes == unit
    assert (views.slot_units, views.state_units, views.header_units) == (
        stride // unit,
        layout.state_bytes // unit,
        256 // unit,
    )
    storage = raw.untyped_storage().data_ptr()
    for view in (views.ckpt, views.vconv, views.state_flat, views.conv_flat):
        assert view.untyped_storage().data_ptr() == storage
    assert views.ckpt.shape == (SLOTS, 1 + NUM_SPEC, HEADS, DIM, DIM)
    assert views.ckpt.dtype == views.state_flat.dtype == state_dtype
    assert views.vconv.shape == (SLOTS, 3 * HEADS * DIM, WINDOW + NUM_SPEC)
    assert views.vconv.dtype == views.conv_flat.dtype == torch.bfloat16
    # Row r of either flat view starts r * unit bytes after the buffer.
    assert views.state_flat.data_ptr() == raw.data_ptr()
    assert views.state_flat.stride(0) * state_dtype.itemsize == unit
    assert views.conv_flat.stride(0) * 2 == unit
    assert views.conv_flat.data_ptr() == raw.data_ptr() + layout.conv_offset - 256
    last = views.state_flat[-1]
    assert last.data_ptr() + layout.state_bytes == raw[-1, layout.conv_offset :].data_ptr()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("extra", "misalign"), _PADDINGS)
def test_indices_address_the_checkpoints(
    state_dtype: torch.dtype, extra: int, misalign: int
) -> None:
    layout = _layout(state_dtype)
    backing, _, raw = _slot_buffer(layout.nbytes, layout.nbytes + extra, misalign)
    views = ck.kda_checkpoint_views(raw, layout)
    for s in range(SLOTS):
        for t in range(1 + NUM_SPEC):
            views.ckpt[s, t].fill_(s * 100 + t + 0.5)
        views.vconv[s].copy_(
            torch.arange(views.vconv[s].numel()).view_as(views.vconv[s]).to(torch.bfloat16) + s
        )
    padding_before = backing.clone()
    drafts = torch.tensor([3, 0, 1, 2, 0], dtype=torch.int32)
    state_indices = torch.tensor([4, 0, 2], dtype=torch.int32)
    rows = ck.checkpoint_position_rows(views, torch.device("cpu"))
    out_slots = torch.full((3, 1 + NUM_SPEC), -7, dtype=torch.int32)
    out_accepted = torch.full((3,), -7, dtype=torch.int32)
    slots, accepted = ck.checkpoint_indices(
        state_indices, drafts, rows, views.slot_units, out_slots, out_accepted
    )
    assert slots is out_slots and accepted is out_accepted
    assert slots.dtype == accepted.dtype == torch.int32
    assert accepted.tolist() == [1, 4, 2]
    assert slots.min() > 0
    for b, s in enumerate(state_indices.tolist()):
        for t in range(1 + NUM_SPEC):
            expected = views.header_units + s * views.slot_units + t * views.state_units
            assert slots[b, t] == expected
            assert torch.equal(views.state_flat[slots[b, t]], views.ckpt[s, t])
        assert torch.equal(views.conv_flat[slots[b, 0]], views.vconv[s])
    # Writes through the flat views land in the typed views and nowhere else.
    for b, s in enumerate(state_indices.tolist()):
        for t in range(1 + NUM_SPEC):
            views.state_flat[slots[b, t]].fill_(-(s * 10 + t))
        views.conv_flat[slots[b, 0]].fill_(-s)
    for s in state_indices.tolist():
        for t in range(1 + NUM_SPEC):
            assert torch.all(views.ckpt[s, t] == -(s * 10 + t))
        assert torch.all(views.vconv[s] == -s)
    changed = (backing != padding_before).nonzero().flatten()
    start = raw.data_ptr() - backing.data_ptr()
    for offset in (changed - start).tolist():
        slot, byte = divmod(offset, raw.stride(0))
        assert (
            slot in state_indices.tolist() and 256 <= byte < layout.conv_offset + layout.conv_bytes
        )
    fresh = ck.checkpoint_indices(state_indices, drafts, rows, views.slot_units)
    assert torch.equal(fresh[0], slots) and torch.equal(fresh[1], accepted)


def test_indices_reject_wrong_output_buffers() -> None:
    layout = _layout(torch.float32)
    _, _, raw = _slot_buffer(layout.nbytes, layout.nbytes)
    views = ck.kda_checkpoint_views(raw, layout)
    rows = ck.checkpoint_position_rows(views, torch.device("cpu"))
    drafts = torch.zeros(SLOTS, dtype=torch.int32)
    indices = torch.tensor([1, 2])
    with pytest.raises(ValueError):
        ck.checkpoint_indices(indices, drafts, rows, views.slot_units, torch.empty(2, 1 + NUM_SPEC))
    with pytest.raises(ValueError):
        ck.checkpoint_indices(
            indices,
            drafts,
            rows,
            views.slot_units,
            torch.empty(2, NUM_SPEC, dtype=torch.int32),
        )
    with pytest.raises(ValueError):
        ck.checkpoint_indices(
            indices,
            drafts,
            rows,
            views.slot_units,
            None,
            torch.empty(3, dtype=torch.int32),
        )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("extra", "misalign"), [(0, 0), (64, 16)])
def test_seed_and_commit(state_dtype: torch.dtype, extra: int, misalign: int) -> None:
    layout = _layout(state_dtype)
    backing, _, raw = _slot_buffer(layout.nbytes, layout.nbytes + extra, misalign)
    views = ck.kda_checkpoint_views(raw, layout)
    generator = torch.Generator().manual_seed(0)
    ssm, conv = _pools(layout, generator)
    seeded = torch.tensor([3, 1], dtype=torch.int64)
    before = views.ckpt.clone(), views.vconv.clone()
    ck.seed_kda_checkpoints(views.ckpt, views.vconv, ssm, conv, seeded)
    for s in range(SLOTS):
        if s in seeded.tolist():
            assert torch.equal(views.ckpt[s, 0], ssm[s].to(state_dtype))
            assert torch.equal(views.ckpt[s, 1:], before[0][s, 1:])
            assert torch.equal(views.vconv[s, :, :WINDOW], conv[s])
            assert torch.equal(views.vconv[s, :, WINDOW:], before[1][s, :, WINDOW:])
        else:
            assert torch.equal(views.ckpt[s], before[0][s])
            assert torch.equal(views.vconv[s], before[1][s])
    ck.seed_kda_checkpoints(views.ckpt, views.vconv, ssm, conv, seeded[:0])

    # A verify wrote every position; commit the accepted one of each slot.
    views.ckpt.copy_(torch.randn(views.ckpt.shape, generator=generator).to(state_dtype))
    views.vconv.copy_(torch.randn(views.vconv.shape, generator=generator).to(torch.bfloat16))
    drafts = torch.tensor([0, 3, 1, 2, 0], dtype=torch.int32)
    committed = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    ssm_before, conv_before = ssm.clone(), conv.clone()
    ck.commit_kda_checkpoints(views.ckpt, views.vconv, ssm, conv, committed, drafts)
    for s in range(SLOTS):
        if s in committed.tolist():
            a = int(drafts[s])
            assert torch.equal(ssm[s], views.ckpt[s, a].float())
            assert torch.equal(conv[s], views.vconv[s, :, a : a + WINDOW])
        else:
            assert torch.equal(ssm[s], ssm_before[s])
            assert torch.equal(conv[s], conv_before[s])
    # Commit then seed is the identity on checkpoint 0 of the accepted position.
    ck.seed_kda_checkpoints(views.ckpt, views.vconv, ssm, conv, committed)
    for s in committed.tolist():
        a = int(drafts[s])
        assert torch.equal(views.ckpt[s, 0], ssm[s].to(state_dtype))
        assert torch.equal(views.vconv[s, :, :WINDOW], conv[s])
        assert torch.equal(ssm[s], views.ckpt[s, 0].float())
        assert a >= 0
    # Bytes between the slots' payloads are untouched.
    stride = raw.stride(0)
    start = raw.data_ptr() - backing.data_ptr()
    tail = backing[start : start + SLOTS * stride].view(SLOTS, stride)[:, layout.nbytes :]
    assert torch.all(tail == 0xAB)


def test_unit_follows_the_state_size() -> None:
    layout = ck.kda_checkpoint_layout(NUM_SPEC, 1, 4, WINDOW, torch.bfloat16, torch.bfloat16)
    assert layout.state_bytes == 32
    _, _, raw = _slot_buffer(layout.nbytes, layout.nbytes)
    views = ck.kda_checkpoint_views(raw, layout)
    assert views.unit_bytes == 32 and views.header_units == 8


def test_views_reject_bad_buffers() -> None:
    layout = _layout(torch.float32)
    _, _, raw = _slot_buffer(layout.nbytes, layout.nbytes + 256, misalign=8)
    with pytest.raises(ValueError):
        ck.kda_checkpoint_views(raw, layout)
    _, _, short = _slot_buffer(layout.nbytes - 256, layout.nbytes)
    with pytest.raises(ValueError):
        ck.kda_checkpoint_views(short, layout)
    with pytest.raises(ValueError):
        ck.kda_checkpoint_views(torch.zeros(SLOTS, layout.nbytes // 4, dtype=torch.int32), layout)
    with pytest.raises(ValueError):
        ck.kda_checkpoint_views(torch.zeros(SLOTS * layout.nbytes, dtype=torch.uint8), layout)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_int32_bound(state_dtype: torch.dtype) -> None:
    """The last flat row of the largest addressable pool is INT32_MAX at most; one more slot is
    rejected before any view is built."""
    layout = _layout(state_dtype)
    unit = _expected_unit(layout.nbytes, 0, layout.state_bytes)
    slot_units, state_units, header = layout.nbytes // unit, layout.state_bytes // unit, 256 // unit
    tail = header + NUM_SPEC * state_units + 1
    max_slots = (INT32_MAX - tail) // slot_units + 1
    fits = torch.empty((max_slots, layout.nbytes), dtype=torch.uint8, device="meta")
    views = ck.kda_checkpoint_views(fits, layout)
    assert views.unit_bytes == unit
    assert views.state_flat.shape[0] - 1 <= INT32_MAX
    assert views.state_flat.shape[0] == (max_slots - 1) * slot_units + tail
    over = torch.empty((max_slots + 1, layout.nbytes), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="int32"):
        ck.kda_checkpoint_views(over, layout)


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    ),
]


def _coalesced_layers(num_layers, layout, ssm_dtype, device, generator):
    """Slots coalesced as the V2 manager lays them out: every layer's SSM state, then conv
    state, then ``kda_ckpt`` payload, then the 256-byte count; random bytes everywhere, and
    random values in the live pools, zeros around the slots. Returns the backing bytes, the
    slots' bytes and the per-layer views."""
    h, v, k = layout.state_shape
    ssm_bytes = h * v * k * ssm_dtype.itemsize
    conv_bytes = layout.conv_dim * WINDOW * 2
    slot_bytes = num_layers * (ssm_bytes + conv_bytes + layout.nbytes) + 256
    backing = torch.zeros(SLOTS * slot_bytes + 512, dtype=torch.uint8, device=device)
    start = (-backing.data_ptr()) % 256
    region = backing[start : start + SLOTS * slot_bytes]
    region.copy_(torch.randint(0, 256, region.shape, dtype=torch.uint8, generator=generator))
    slots = region.view(SLOTS, slot_bytes)

    def per_layer(offset, nbytes, dtype, shape):
        return [
            slots[:, offset + i * nbytes : offset + (i + 1) * nbytes]
            .view(dtype)
            .unflatten(1, shape)
            for i in range(num_layers)
        ]

    ssm = per_layer(0, ssm_bytes, ssm_dtype, (h, v, k))
    conv = per_layer(num_layers * ssm_bytes, conv_bytes, torch.bfloat16, (layout.conv_dim, WINDOW))
    raw = per_layer(
        num_layers * (ssm_bytes + conv_bytes), layout.nbytes, torch.uint8, (layout.nbytes,)
    )
    for tensor in (*ssm, *conv):
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(device, tensor.dtype))
    return (backing, region), ssm, conv, [ck.kda_checkpoint_views(r, layout) for r in raw]


def _seed_stacked(ssm, conv, views, slots):
    window = conv[0].shape[-1]
    stacked = [
        ck.stack_layer_views(group)
        for group in (
            [view.ckpt[:, 0] for view in views],
            [view.vconv[..., :window] for view in views],
            ssm,
            conv,
        )
    ]
    assert all(s is not None for s in stacked)
    ck.seed_kda_checkpoints_stacked(*stacked, slots)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("state_dtype", "ssm_dtype", "num_layers", "seeded"),
    [
        pytest.param(torch.float32, torch.float32, 1, [0], id="fp32-one-slot"),
        pytest.param(torch.bfloat16, torch.bfloat16, 1, [0], id="bf16-one-slot"),
        pytest.param(torch.bfloat16, torch.float32, 1, [0], id="cast-one-slot"),
        pytest.param(torch.float32, torch.float32, 69, [4, 0, 2], id="fp32-sparse-slots"),
        pytest.param(torch.bfloat16, torch.bfloat16, 69, [4, 0, 2], id="bf16-sparse-slots"),
        pytest.param(torch.bfloat16, torch.float32, 69, [4, 0, 2], id="cast-sparse-slots"),
        pytest.param(torch.bfloat16, torch.float32, 3, [3, 1], id="cast-intermediate-layers"),
        # An empty index returns before the dtype-dependent copies.
        pytest.param(torch.float32, torch.float32, 3, [], id="empty-slots"),
    ],
)
def test_stacked_seed_is_the_layer_loop_bitwise(device, state_dtype, ssm_dtype, num_layers, seeded):
    """Seeding all layers at once writes the same bytes as seed_kda_checkpoints layer by layer,
    over the whole pool (every other byte of the slots and the padding included)."""
    layout = _layout(state_dtype)
    # Two pools with the same bytes; one is seeded layer by layer, the other at once.
    loop = _coalesced_layers(
        num_layers, layout, ssm_dtype, device, torch.Generator().manual_seed(7)
    )
    batched = _coalesced_layers(
        num_layers, layout, ssm_dtype, device, torch.Generator().manual_seed(7)
    )
    (loop_backing, loop_slots), (batched_backing, batched_slots) = loop[0], batched[0]
    assert torch.equal(loop_slots, batched_slots)
    before = loop_slots.clone()
    slots = torch.tensor(seeded, dtype=torch.int64, device=device)

    _, ssm, conv, views = loop
    for i in range(num_layers):
        ck.seed_kda_checkpoints(views[i].ckpt, views[i].vconv, ssm[i], conv[i], slots)
    _seed_stacked(*batched[1:], slots)

    assert torch.equal(loop_slots, batched_slots)
    assert torch.equal(loop_slots, before) == (not seeded)
    # Nothing outside the slots is written (the bytes around them stay 0).
    for backing, region in ((loop_backing, loop_slots), (batched_backing, batched_slots)):
        assert backing.sum(dtype=torch.int64) == region.sum(dtype=torch.int64)


def test_stack_layer_views_rejects_uneven_layers() -> None:
    backing = torch.zeros(4096)
    even = [backing[i * 100 : i * 100 + 64].view(8, 8) for i in range(3)]
    stacked = ck.stack_layer_views(even)
    assert stacked.shape == (3, 8, 8) and stacked.stride() == (100, 8, 1)
    for i, view in enumerate(even):
        assert stacked[i].data_ptr() == view.data_ptr() and torch.equal(stacked[i], view)
    assert ck.stack_layer_views(even[:1]).shape == (1, 8, 8)
    assert ck.stack_layer_views([]) is None
    uneven = [backing[o : o + 64].view(8, 8) for o in (0, 100, 300)]
    assert ck.stack_layer_views(uneven) is None
    descending = list(reversed(even))
    assert ck.stack_layer_views(descending) is None
    strided = [even[0], backing[100:228].view(8, 16)[:, :8], even[2]]
    assert ck.stack_layer_views(strided) is None
    other_storage = [even[0], torch.zeros(8, 8), even[2]]
    assert ck.stack_layer_views(other_storage) is None
    other_dtype = [view.view(torch.int32) for view in even[:2]] + [even[2]]
    assert ck.stack_layer_views(other_dtype) is None


def _aten_ops(fn) -> list[str]:
    from torch.utils._python_dispatch import TorchDispatchMode

    calls = []

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            calls.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    with Count():
        fn()
    return calls


# Ops that only build views (no kernel launch on a GPU).
_VIEW_OPS = {"select", "slice", "as_strided", "view", "unsqueeze", "expand", "alias", "detach"}


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_stacked_seed_ops_do_not_grow_with_layers(state_dtype) -> None:
    """The stacked seed runs one gather and one scatter per buffer kind whatever the layer count;
    the layer loop runs 4 data ops per layer (the 276 seed kernels of an admission on the 69
    local KDA layers of the deployment)."""
    layout = _layout(state_dtype)
    slots = torch.tensor([3], dtype=torch.int64)

    def counts(num_layers):
        _, ssm, conv, views = _coalesced_layers(
            num_layers, layout, state_dtype, "cpu", torch.Generator().manual_seed(0)
        )

        def loop():
            for i in range(num_layers):
                ck.seed_kda_checkpoints(views[i].ckpt, views[i].vconv, ssm[i], conv[i], slots)

        def stacked():
            ck.seed_kda_checkpoints_stacked(*stacked_views, slots)

        window = conv[0].shape[-1]
        stacked_views = [
            ck.stack_layer_views(g)
            for g in (
                [v.ckpt[:, 0] for v in views],
                [v.vconv[..., :window] for v in views],
                ssm,
                conv,
            )
        ]
        data = lambda ops: [op for op in ops if op not in _VIEW_OPS]  # noqa: E731
        return data(_aten_ops(loop)), data(_aten_ops(stacked))

    loop2, stacked2 = counts(2)
    loop69, stacked69 = counts(69)
    assert stacked2 == stacked69 == ["index_select", "index_copy_", "index_select", "index_copy_"]
    assert len(loop69) == 69 * len(loop2) // 2 == 69 * 4
