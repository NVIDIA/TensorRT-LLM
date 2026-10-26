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
    if kind == "explicit":
        positions = list(contexts[1])
    else:
        count = contexts[1]
        low, high = {"mixed": (0, 8188), "full": (2047, 8188)}[kind]
        fixed = {"mixed": ANCHORS, "full": (2047, 2050, 8188, 2051)}[kind]
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

    request, visible = [], []
    for q in range(num_requests):
        captured, current = pairs[q]
        for j in range(per_request):
            request.append(q)
            # A later row of the same request sees at least as many groups.
            visible.append(current + j // CR)
    request = torch.tensor(request, dtype=meta_dtype).reshape(rows)
    visible = torch.tensor(visible, dtype=meta_dtype).reshape(rows)

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
    kwargs = dict(
        shared_indices=buffers["shared"][slot],
        shared_visible_blocks=buffers["shared_vis"][slot],
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


def _guard_case():
    """Legal arguments, then the whole `[slots, seqs]` visible-count buffer passed where
    one slot's view belongs.

    Returns (intended, kwargs): `intended` is the reference on the legal
    arguments -- what a caller passing the bad layout meant to get.
    """
    spec = _spec_of("decode MTP")
    kwargs = _build({**spec, "slots": (3, 2)}, seed=13, requests=6)["kwargs"]
    intended = op.reference(**kwargs)
    whole = torch.zeros((3, kwargs["shared_visible_blocks"].numel()), dtype=torch.int32)
    whole = whole.to(DEV)
    whole[2] = kwargs["shared_visible_blocks"]
    kwargs["shared_visible_blocks"] = whole
    return intended, kwargs


def test_guard_refuses_a_whole_multi_slot_count_buffer() -> None:
    """Unguarded the op reads slot 0's counts and returns a plausible wrong answer;
    `is_valid` refuses it."""
    intended, kwargs = _guard_case()
    out = op(**kwargs)
    torch.cuda.synchronize()
    assert not torch.equal(out, intended), "the op got it right after all"
    _, kwargs = _guard_case()
    with (
        validating(op),
        pytest.raises(ValueError, match="shared_visible_blocks must be one slot's"),
    ):
        op(**kwargs)
