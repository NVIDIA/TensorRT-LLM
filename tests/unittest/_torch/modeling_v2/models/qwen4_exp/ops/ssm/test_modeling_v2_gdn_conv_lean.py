# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_conv_lean entries (`conv_lean`, `promote_x`). Beyond the cells: the
lean pair against the stock pair (`causal_conv1d_triton_update` with snapshots, then the
runtime's `promote_conv_and_advance_replay_triton`) -- the conv output, the promoted windows of
every layer and the bookkeeping bitwise -- eagerly and under CUDA graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_triton_update import (  # noqa: E501
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_conv_lean import (
    GDN_LAYERS,
    WIDTH,
    gdn_conv_lean,
    gdn_promote_x,
    precompile,
)
from tensorrt_llm._torch.modules.fla.replay_promotion import promote_conv_and_advance_replay_triton

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_conv_lean requires a CUDA device"

PAD = -1


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)


def _slots(batch: int, pool_slots: int) -> torch.Tensor:
    """`batch` distinct slots, the top slot and slot 0 first, then a stride-7 walk."""
    rest = [(i * 7 + 3) % (pool_slots - 2) + 1 for i in range(max(batch - 2, 0))]
    slots = ([pool_slots - 1, 0] + rest)[:batch]
    assert len(set(slots)) == batch
    return torch.tensor(slots, dtype=torch.int32, device="cuda")


def _lean_args(spec, seed):
    """`conv_lean`'s positional arguments for one spec: x the q|k|v columns of a `[n * T,
    stride]` projection, the pool the last layer's view of a `[slots, layers, C, 3]` page,
    `xsave` a layer's view of `[2, max_batch, T, C]`, `xrows` reversed (not the batch row)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c, n, t = spec["dim"], spec["batch"], spec["seqlen"]
    layers = spec.get("pool_layers", GDN_LAYERS)
    buf = _randn(g, n * t, spec["x_row_stride"])
    page = _randn(g, spec["pool_slots"], layers, c, WIDTH - 1)
    slots = _slots(n, spec["pool_slots"])
    if "pad" in spec:
        slots[spec["pad"]] = PAD
    xsave = torch.zeros(2, n + 3, t, c, dtype=torch.bfloat16, device="cuda")
    xrows = torch.arange(n, dtype=torch.int32, device="cuda").flip(0) + 2
    args = [
        buf[:, :c],
        _randn(g, c, WIDTH),
        page[:, layers - 1],
        slots,
        xrows,
        xsave[1],
        t,
        PAD,
    ]
    kw = dict(wait=spec.get("wait", True), release=spec.get("release", True))
    return args, kw, dict(buf=buf, page=page, xsave=xsave)


@pytest.mark.parametrize("cell", gdn_conv_lean.CELLS, ids=[c.why[:44] for c in gdn_conv_lean.CELLS])
def test_conv_lean_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(gdn_conv_lean)
    args, kw, extras = _lean_args(cell.spec, seed=zlib.crc32(cell.why.encode()))
    page, buf, xsave = extras["page"].clone(), extras["buf"].clone(), extras["xsave"]
    ref_out, ref_saved = gdn_conv_lean.reference(*args, **kw)
    before = xsave.clone()
    out = gdn_conv_lean(*args, **kw)
    torch.cuda.synchronize()
    written = ~torch.isnan(ref_out)
    # Against the reference at torch's default band, as the stock conv's own test: the kernel's
    # fp32 `exp` is one bf16 ulp off torch's in a handful of elements (1-7 of 0.08-2.6M on GB300).
    torch.testing.assert_close(out[written], ref_out[written])
    # Bitwise the stock launch (`causal_conv1d_triton_update` with snapshots) on the same inputs:
    # the op's claim, and the conv output a lean step hands the GDN.
    x, weight, conv_state, slots, _, _, seqlen, pad = args
    n, c = x.shape[0] // seqlen, weight.shape[0]
    snaps = torch.zeros(n, seqlen, c, WIDTH - 1, dtype=torch.bfloat16, device="cuda")
    stock = causal_conv1d_triton_update(
        x.reshape(n, seqlen, c).transpose(1, 2), conv_state.clone(), weight, None, "silu", slots,
        pad, None, snaps, torch.arange(n, dtype=torch.int32, device="cuda"), True,
    ).transpose(1, 2).reshape(n * seqlen, c)  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(out[written], stock[written]), (
        "the conv output must be bitwise the stock conv's"
    )
    assert torch.equal(xsave[1], ref_saved), "the saved rows must be the raw rows"
    assert torch.equal(xsave[0], before[0]), "another layer's xsave written"
    assert torch.equal(extras["page"], page), "the pool is read-only"
    assert torch.equal(extras["buf"], buf), "x is read-only"


def _promote_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    c, layers, n, t = spec["dim"], spec["layers"], spec["batch"], spec["seqlen"]
    history = spec.get("history", 16)
    page = _randn(g, spec["pool_slots"], layers, c, WIDTH - 1)
    stack = page.permute(1, 0, 2, 3)
    xsave = _randn(g, layers, n + 2, t, c)
    slots = _slots(n, spec["pool_slots"])
    if "pad" in spec:
        slots[spec["pad"]] = PAD
    src = torch.arange(n, dtype=torch.int32, device="cuda").flip(0) + 1
    acc = torch.tensor([1 + i % t for i in range(n)], dtype=torch.int32, device="cuda")
    dummy = torch.zeros(n, dtype=torch.bool, device="cuda")
    if "dummy" in spec:
        dummy[spec["dummy"]] = True
    replay_rows = spec["pool_slots"] + 7
    # The V2 manager packs both bookkeeping words of a replay slot side by side.
    meta = torch.randint(0, history + 1, (replay_rows, 2), generator=g, device="cuda").int()
    meta[:, 1] %= 2
    replay = None
    if spec.get("replay"):
        replay = torch.randperm(replay_rows, generator=g, device="cuda")[:n].int()
    args = [stack, xsave, src, slots, acc, dummy, meta[:, 0], meta[:, 1], replay, t, history, PAD]
    return args, dict(page=page, meta=meta)


@pytest.mark.parametrize("cell", gdn_promote_x.CELLS, ids=[c.why[:44] for c in gdn_promote_x.CELLS])
def test_promote_x_cells(cell, certifying) -> None:  # noqa: F811
    certifying(gdn_promote_x)
    args, extras = _promote_args(cell.spec, seed=zlib.crc32(cell.why.encode()))
    ref_pool, ref_pnat, ref_cbi = gdn_promote_x.reference(*args)
    gdn_promote_x(*args)
    torch.cuda.synchronize()
    assert torch.equal(args[0], ref_pool), "the promoted windows must be bitwise"
    assert torch.equal(args[6], ref_pnat) and torch.equal(args[7], ref_cbi), "bookkeeping"


@pytest.mark.parametrize("batch", [2, 5, 12, 24])
@pytest.mark.parametrize("graph", [False, True])
def test_lean_pair_is_bitwise_the_stock_pair(batch, graph) -> None:
    """Two layers, separate replay slots, every accepted count: `conv_lean` + `promote_x` leave
    the conv output, both layers' pool windows and the bookkeeping bitwise where the stock conv
    with snapshots + the runtime's one-launch promotion leaves them; under `graph` both pairs
    are captured once and replayed over fresh inputs (the order a decode graph runs them in)."""
    g = torch.Generator(device="cuda").manual_seed(batch)
    c, t, layers, slots_n, history = 10240, 4, 2, 64, 16
    stride = 16480
    page = _randn(g, slots_n, layers, c, WIDTH - 1)
    weights = [_randn(g, c, WIDTH) for _ in range(layers)]
    slots = _slots(batch, slots_n)
    rows = torch.arange(batch, dtype=torch.int32, device="cuda")
    replay = torch.randperm(slots_n + 5, generator=g, device="cuda")[:batch].int()
    dummy = torch.zeros(batch, dtype=torch.bool, device="cuda")
    dummy[batch - 1] = True
    acc = torch.empty(batch, dtype=torch.int32, device="cuda")
    buf = torch.empty(batch * t, stride, dtype=torch.bfloat16, device="cuda")
    precompile(c, t, history, stride, torch.device("cuda"))

    def lean_state():
        return dict(
            page=page.clone(),
            xsave=torch.zeros(layers, batch, t, c, dtype=torch.bfloat16, device="cuda"),
            meta=torch.zeros(slots_n + 5, 2, dtype=torch.int32, device="cuda"),
        )

    def stock_state():
        return dict(
            page=page.clone(),
            snaps=torch.zeros(layers, batch, t, c, WIDTH - 1, dtype=torch.bfloat16, device="cuda"),
            meta=torch.zeros(slots_n + 5, 2, dtype=torch.int32, device="cuda"),
        )

    lean, stock = lean_state(), stock_state()

    def run_lean():
        outs = []
        for i in range(layers):
            outs.append(
                gdn_conv_lean(
                    buf[:, :c],
                    weights[i],
                    lean["page"][:, i],
                    slots,
                    rows,
                    lean["xsave"][i],
                    t,
                    PAD,
                )
            )
        gdn_promote_x(
            lean["page"].permute(1, 0, 2, 3), lean["xsave"], rows, slots, acc, dummy,
            lean["meta"][:, 0], lean["meta"][:, 1], replay, t, history, PAD,
        )  # fmt: skip
        return outs

    def run_stock():
        outs = []
        for i in range(layers):
            x3 = buf[:, :c].reshape(batch, t, c).transpose(1, 2)
            o = causal_conv1d_triton_update(
                x3, stock["page"][:, i], weights[i], None, "silu", slots, PAD, None,
                stock["snaps"][i], rows, True,
            )  # fmt: skip
            outs.append(o.transpose(1, 2).reshape(batch * t, c))
        promote_conv_and_advance_replay_triton(
            stock["page"].permute(1, 0, 2, 3), stock["snaps"], rows, acc, slots,
            num_accepted_tokens=acc, replay_pnat=stock["meta"][:, 0],
            replay_cache_buf_idx=stock["meta"][:, 1], dummy_request_mask=dummy,
            replay_step_width=t, replay_history_size=history, replay_indices=replay,
            position_source_is_token_count=True,
        )  # fmt: skip
        return outs

    def fresh_inputs(step):
        buf.copy_(_randn(g, batch * t, stride))
        acc.copy_(torch.tensor([1 + (b + step) % t for b in range(batch)], device="cuda"))

    fresh_inputs(0)
    if graph:
        # Compile every kernel eagerly (on states thrown away below), then capture once from
        # the same starting state: a capture runs nothing.
        run_lean(), run_stock()
        torch.cuda.synchronize()
        lean.update(lean_state())
        stock.update(stock_state())
        lean_graph, stock_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(lean_graph):
            lean_outs = run_lean()
        with torch.cuda.graph(stock_graph):
            stock_outs = run_stock()
    for step in range(6):
        fresh_inputs(step)
        if graph:
            lean_graph.replay()
            stock_graph.replay()
        else:
            lean_outs, stock_outs = run_lean(), run_stock()
        torch.cuda.synchronize()
        for i in range(layers):
            assert torch.equal(lean_outs[i], stock_outs[i]), (step, i, "conv output")
        assert torch.equal(lean["page"], stock["page"]), (step, "promoted windows")
        assert torch.equal(lean["meta"], stock["meta"]), (step, "bookkeeping")
