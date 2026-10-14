# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_mtp_reuse_topk catalog entry.

Captured rows are built the way `qsa_topk_select` emits them: `min(K, visible)`
distinct complete groups in an arbitrary order as a prefix, then `-1`. Inputs
sit in the engine's buffers -- one slot of a `[slots, max_num_sequences, K]`
shared buffer holding stale rows elsewhere, and row slices of the
`max_num_tokens`-tall request map, visible-count and Top-K buffers. Every gate
is bitwise.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_mtp_reuse_topk import (  # noqa: E501
    qsa_mtp_reuse_topk as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_mtp_reuse_topk requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: these are group ids, and one wrong entry attends to a different
    group. There is no rounding for a tolerance to absorb."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


DEV = "cuda"
CR = 4
K = 512
#: The engine's `max_num_tokens`: height of the request-map and Top-K buffers.
ENGINE_MAX_NUM_TOKENS = 8192
ENGINE_MAX_SEQ_LEN = 8192
#: Captured positions for the first requests of a "mixed" batch: empty (0
#: groups), 1, 100, 255, 511 and 512 groups, rows that gain a group and rows
#: that do not, up to the last position the 8192-token envelope allows.
ANCHORS = (2, 3, 402, 2046, 2050, 8188, 2051, 2047, 1022, 4095, 5998, 7001, 37, 2044, 3001, 6143)


def _captured_row(visible: int, block_topk: int, gen: torch.Generator) -> torch.Tensor:
    """A legal capture: `min(K, visible)` distinct groups below `visible`, then -1."""
    row = torch.full((block_topk,), -1, dtype=torch.int32)
    if visible > 0:
        take = min(visible, block_topk)
        row[:take] = torch.randperm(visible, generator=gen)[:take].to(torch.int32)
    return row


def _requests(contexts, drafted, gen) -> list[tuple[int, int]]:
    """(captured_visible, current_visible) per request."""
    kind = contexts[0]
    if kind == "gaps":
        return list(contexts[1])
    if kind == "explicit":
        positions = list(contexts[1])
    else:
        count = contexts[1]
        low, high = {"mixed": (0, 8188), "short": (0, 2046), "full": (2047, 8188)}[kind]
        fixed = {
            "mixed": ANCHORS,
            "short": (2, 2046, 2045, 2044, 2043, 402, 3),
            "full": (2047, 2050, 8188, 2051),
        }[kind]
        positions = list(fixed[:count])
        positions += torch.randint(low, high + 1, (count - len(positions),), generator=gen).tolist()
    out = []
    for q, position in enumerate(positions):
        d = drafted[q % len(drafted)]
        position = min(position, ENGINE_MAX_SEQ_LEN - 1 - d)
        out.append(((position + 1) // CR, (position + d + 1) // CR))
    return out


def _build(spec, seed: int, requests: int | None = None) -> dict:
    """Turn a cell spec into the op's keyword arguments, plus the buffers behind them.

    Returns `dict(kwargs=..., buffers=...)`; `buffers` holds the full tensors the
    arguments are views of, so a test can check that nothing outside the view moved.
    """
    gen = torch.Generator().manual_seed(seed)
    block_topk = spec["block_topk"]
    contexts = spec["contexts"]
    if requests is not None:
        contexts = (contexts[0], requests)
    pairs = _requests(contexts, spec["drafted"], gen)
    num_requests = len(pairs)
    per_request = spec["rows_per_request"]
    rows = num_requests * per_request
    num_slots, slot = spec["slots"]
    shared_rows = max(spec["shared_rows"], num_requests)
    meta_dtype = spec["metadata_dtype"]

    # Stale rows everywhere: other layers' slots and requests past this batch.
    shared = torch.randint(-1, 3000, (num_slots, shared_rows, block_topk), generator=gen)
    shared = shared.to(torch.int32)
    shared_vis = torch.randint(0, 3000, (num_slots, shared_rows), generator=gen).to(meta_dtype)
    for q, (captured, _) in enumerate(pairs):
        shared[slot, q] = _captured_row(captured, block_topk, gen)
        shared_vis[slot, q] = captured

    order = list(range(num_requests))
    if spec["request_map"] == "permuted":
        order = torch.randperm(num_requests, generator=gen).tolist()
    request, visible = [], []
    for q in order:
        captured, current = pairs[q]
        for j in range(per_request):
            request.append(q)
            # A later row of the same request sees at least as many groups.
            visible.append(current + j // CR)
    request = torch.tensor(request, dtype=meta_dtype).reshape(rows)
    visible = torch.tensor(visible, dtype=meta_dtype).reshape(rows)

    if spec["layout"] == "engine":
        height = max(ENGINE_MAX_NUM_TOKENS, rows + 37)
        tall_out = torch.randint(-1, 3000, (height, block_topk), generator=gen).to(torch.int32)
        tall_req = torch.randint(0, shared_rows, (height,), generator=gen).to(meta_dtype)
        tall_vis = torch.randint(0, 3000, (height,), generator=gen).to(meta_dtype)
        tall_req[:rows], tall_vis[:rows] = request, visible
        buffers = dict(
            shared=shared.to(DEV),
            shared_vis=shared_vis.to(DEV),
            request=tall_req.to(DEV),
            visible=tall_vis.to(DEV),
            output=tall_out.to(DEV),
        )
    else:
        assert spec["layout"] == "dense"
        buffers = dict(
            shared=shared[slot].clone().to(DEV),
            shared_vis=shared_vis[slot].clone().to(DEV),
            request=request.to(DEV),
            visible=visible.to(DEV),
            output=torch.full((rows, block_topk), -7, dtype=torch.int32, device=DEV),
        )
        slot = None
    kwargs = dict(
        shared_indices=buffers["shared"] if slot is None else buffers["shared"][slot],
        shared_visible_blocks=buffers["shared_vis"]
        if slot is None
        else buffers["shared_vis"][slot],
        request_indices=buffers["request"][:rows],
        visible_blocks=buffers["visible"][:rows],
        output=buffers["output"][:rows],
    )
    return dict(kwargs=kwargs, buffers=buffers, rows=rows)


def _assert_legal_selection(out: torch.Tensor, kwargs: dict) -> None:
    """What the producer's contract buys: no duplicate group, none past the horizon."""
    if out.shape[0] == 0:
        return
    ids = out.long()
    ordered = torch.sort(ids, dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()), (
        "a group id appears twice in a row"
    )
    captured = kwargs["shared_visible_blocks"].long()[kwargs["request_indices"].long()]
    horizon = torch.maximum(captured, kwargs["visible_blocks"].long())
    assert bool((ids < horizon[:, None]).all()), "a group id reaches past the row's horizon"
    assert bool((ids >= -1).all())


def _check_cell(spec, seed: int, requests: int | None = None) -> torch.Tensor:
    built = _build(spec, seed, requests)
    kwargs, buffers, rows = built["kwargs"], built["buffers"], built["rows"]
    before = {name: tensor.clone() for name, tensor in buffers.items()}
    out = op(**kwargs)
    torch.cuda.synchronize()
    assert out.data_ptr() == kwargs["output"].data_ptr(), "the op returns the buffer it wrote"
    assert out.dtype == torch.int32 and tuple(out.shape) == (rows, spec["block_topk"])
    _compare(out, op.reference(**kwargs))
    _assert_legal_selection(out, kwargs)
    for name, tensor in buffers.items():
        if name == "output":
            assert torch.equal(tensor[rows:], before[name][rows:]), "rows past the view moved"
        else:
            assert torch.equal(tensor, before[name]), f"{name} was written"
    return out


def _check_graph(spec, seed: int) -> None:
    """At each request count: eager, capture, then replay on new data."""
    for requests in spec["graph_batches"]:
        first = _build(spec, seed + requests, requests)
        kwargs = first["kwargs"]
        eager = op(**kwargs).clone()
        _compare(eager, op.reference(**kwargs))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(**kwargs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), f"{requests} requests: replay != eager"

        # The mixed batch's first requests sit at fixed positions, so the new
        # data comes from full rows: every request's captured row changes.
        refill = {**spec, "contexts": ("full", requests)}
        fresh = _build(refill, seed + 1000 + requests)["kwargs"]
        for name, tensor in kwargs.items():
            if name != "output":
                tensor.copy_(fresh[name])
        graph.replay()
        torch.cuda.synchronize()
        expected = op.reference(**kwargs)
        assert not torch.equal(expected, eager), f"{requests} requests: new data changes nothing"
        _compare(captured, expected)
        rerun = op(**{**kwargs, "output": torch.empty_like(kwargs["output"])})
        assert torch.equal(captured, rerun), f"{requests} requests: replay != eager on new data"
        del graph


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode())
    if cell.spec["graph_batches"]:
        _check_graph(cell.spec, seed)
    else:
        _check_cell(cell.spec, seed)


def _spec_of(prefix: str) -> dict:
    (cell,) = [c for c in op.CELLS if c.why.startswith(prefix)]
    return cell.spec


def test_boundary_rows_take_the_documented_branch() -> None:
    """The boundary cell's rows land where the contract says, read off the output."""
    spec = _spec_of("8 requests, one per boundary")
    built = _build(spec, seed=5)
    kwargs = built["kwargs"]
    out = op(**kwargs).cpu()
    shared = kwargs["shared_indices"].cpu()
    # (captured, current) per request: (0,1) (1,1) (100,101) (511,512) (512,513)
    # (2047,2048) (513,513) (512,512)
    assert out[0, 0] == 0 and bool((out[0, 1:] == -1).all()), "empty row: group 0 at slot 0"
    assert torch.equal(out[1], shared[1]), "no new group: a plain copy"
    assert torch.equal(out[2, :100], shared[2, :100]) and out[2, 100] == 100
    assert bool((out[2, 101:] == -1).all()), "short row: appended right after the prefix"
    assert torch.equal(out[3, :511], shared[3, :511]) and out[3, 511] == 511, "fills slot 511"
    for row, group in ((4, 512), (5, 2047)):
        assert torch.equal(out[row, :511], shared[row, :511]) and out[row, 511] == group, (
            "full row: the new group displaces slot 511"
        )
    assert torch.equal(out[6], shared[6]) and torch.equal(out[7], shared[7])


def test_worked_example_small_geometry() -> None:
    """Five rows written out by hand at K = 8, through a permuted request map."""
    shared = torch.tensor(
        [
            [5, 2, 7, 0, -1, -1, -1, -1],  # 4 of 8 groups, 2 more completed
            [3, 1, 6, 4, 0, 2, 7, 5],  # full, 1 more: displaces slot 7
            [-1] * 8,  # nothing captured yet
            [1, 0, 2, -1, -1, -1, -1, -1],  # nothing new: a copy
            [4, 1, 0, 3, 2, 6, 5, -1],  # 7 ids + 2 new: displaces slot 6 too
        ],
        dtype=torch.int32,
        device=DEV,
    )
    captured = torch.tensor([8, 8, 0, 3, 7], dtype=torch.int32, device=DEV)
    request = torch.tensor([1, 0, 4, 2, 3], dtype=torch.int32, device=DEV)
    current = torch.tensor([9, 10, 9, 1, 3], dtype=torch.int32, device=DEV)
    kwargs = dict(
        shared_indices=shared,
        shared_visible_blocks=captured,
        request_indices=request,
        visible_blocks=current,
        output=torch.full((5, 8), 99, dtype=torch.int32, device=DEV),
    )
    out = op(**kwargs)
    _compare(out, op.reference(**kwargs))
    assert out.cpu().tolist() == [
        [3, 1, 6, 4, 0, 2, 7, 8],
        [5, 2, 7, 0, 8, 9, -1, -1],
        [4, 1, 0, 3, 2, 6, 7, 8],
        [0, -1, -1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1, -1, -1],
    ]


def test_unpacked_prefix_overwrites_a_valid_id() -> None:
    """The append slot is the count of valid ids, so a -1 hole costs a group, silently.

    Prefix packing is a value property, so no metadata guard can check it.
    """
    shared = torch.tensor([[3, -1, 1, -1, -1, -1, -1, -1]], dtype=torch.int32, device=DEV)
    kwargs = dict(
        shared_indices=shared,
        shared_visible_blocks=torch.tensor([4], dtype=torch.int32, device=DEV),
        request_indices=torch.tensor([0], dtype=torch.int32, device=DEV),
        visible_blocks=torch.tensor([5], dtype=torch.int32, device=DEV),
        output=torch.empty((1, 8), dtype=torch.int32, device=DEV),
    )
    out = op(**kwargs)
    _compare(out, op.reference(**kwargs))
    assert out.cpu().tolist() == [[3, -1, 4, -1, -1, -1, -1, -1]]


def _guard_case(name: str):
    """Legal arguments, then the one layout `is_valid` refuses.

    Returns (intended, kwargs): `intended` is the reference on the legal
    arguments -- what a caller passing the bad layout meant to get.
    """
    spec = _spec_of("permuted request map")
    kwargs = _build({**spec, "slots": (3, 2)}, seed=13, requests=6)["kwargs"]
    intended = op.reference(**kwargs)
    rows = kwargs["output"].shape[0]

    def strided(tensor):
        wide = torch.zeros((tensor.numel() * 2,), dtype=tensor.dtype, device=DEV)
        wide[::2] = tensor.reshape(-1)
        return wide[::2]

    if name == "output column slice":
        kwargs["output"] = torch.zeros((rows, 2 * K), dtype=torch.int32, device=DEV)[:, :K]
    elif name == "shared_indices column slice":
        wide = torch.full((kwargs["shared_indices"].shape[0], 2 * K), -1, dtype=torch.int32)
        wide = wide.to(DEV)
        wide[:, :K] = kwargs["shared_indices"]
        kwargs["shared_indices"] = wide[:, :K]
    elif name == "shared_indices 3-D, middle extent == block_topk":
        # Slot 1 meant; the 3-D buffer passes the width check and reads slot 0.
        stack = torch.full((2, 8, 8), -1, dtype=torch.int32, device=DEV)
        stack[1, :, :2] = torch.tensor([5, 6], dtype=torch.int32, device=DEV)
        intended_kwargs = dict(
            shared_indices=stack[1],
            shared_visible_blocks=torch.full((8,), 7, dtype=torch.int32, device=DEV),
            request_indices=torch.arange(4, dtype=torch.int32, device=DEV),
            visible_blocks=torch.full((4,), 8, dtype=torch.int32, device=DEV),
            output=torch.empty((4, 8), dtype=torch.int32, device=DEV),
        )
        intended = op.reference(**intended_kwargs)
        kwargs = {**intended_kwargs, "shared_indices": stack}
    elif name == "shared_visible_blocks 2-D [slots, seqs]":
        whole = torch.zeros((3, kwargs["shared_visible_blocks"].numel()), dtype=torch.int32)
        whole = whole.to(DEV)
        whole[2] = kwargs["shared_visible_blocks"]
        kwargs["shared_visible_blocks"] = whole
    else:
        field = name.split()[1]
        kwargs[field] = strided(kwargs[field])
    return intended, kwargs


GUARDS = (
    ("output column slice", "output must be contiguous"),
    ("shared_indices column slice", "shared_indices must be one slot's"),
    ("shared_indices 3-D, middle extent == block_topk", "shared_indices must be one slot's"),
    ("shared_visible_blocks 2-D [slots, seqs]", "shared_visible_blocks must be one slot's"),
    ("strided request_indices", "request_indices must be contiguous"),
    ("strided visible_blocks", "visible_blocks must be contiguous"),
    ("strided shared_visible_blocks", "shared_visible_blocks must be one slot's"),
)


@pytest.mark.parametrize("name, message", GUARDS, ids=[g[0] for g in GUARDS])
def test_guard_refuses_a_silently_wrong_layout(name: str, message: str) -> None:
    """Unguarded the op returns a plausible wrong answer; `is_valid` refuses it."""
    intended, kwargs = _guard_case(name)
    out = op(**kwargs)
    torch.cuda.synchronize()
    assert not torch.equal(out, intended), f"{name}: the op got it right after all"
    _, kwargs = _guard_case(name)
    with validating(op), pytest.raises(ValueError, match=message):
        op(**kwargs)


def test_op_rejects_malformed_arguments() -> None:
    """The callable's own loud errors: why `is_valid` does not restate them."""
    kwargs = _build(_spec_of("dense standalone"), seed=17)["kwargs"]
    cases = [
        ("width does not match", dict(shared_indices=kwargs["shared_indices"][:, : K // 2])),
        (
            "one request and bound per query row",
            dict(request_indices=kwargs["request_indices"][:3]),
        ),
        ("one request and bound per query row", dict(visible_blocks=kwargs["visible_blocks"][:3])),
        ("requires CUDA tensors", dict(shared_indices=kwargs["shared_indices"].cpu())),
        ("requires CUDA tensors", dict(output=kwargs["output"].cpu())),
    ]
    for message, override in cases:
        with pytest.raises(ValueError, match=message):
            op(**{**kwargs, **override})


def test_discriminates_plausible_wrong_variants() -> None:
    """The bitwise gate separates this contract from the ones a rewrite would produce."""
    spec = _spec_of("8 requests, one per boundary")
    kwargs = _build(spec, seed=19)["kwargs"]
    out = op(**kwargs).cpu()
    rows = kwargs["request_indices"].long().cpu()
    shared = kwargs["shared_indices"].cpu()[rows]
    captured = kwargs["shared_visible_blocks"].long().cpu()[rows]
    current = kwargs["visible_blocks"].long().cpu()

    never_displace = shared.clone()  # appends only where there is room
    always_last = shared.clone()  # always writes the last `fresh` slots
    for r in range(out.shape[0]):
        fresh = int((current[r] - captured[r]).clamp(0, K))
        filled = int((shared[r] >= 0).sum())
        for j in range(fresh):
            if filled + j < K:
                never_displace[r, filled + j] = captured[r] + j
            always_last[r, K - fresh + j] = captured[r] + j
    for wrong in (never_displace, always_last, shared):
        assert not torch.equal(out, wrong.to(torch.int32))
