# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the 2..16-row hyper-connection site (`qwen4_exp_hc_decode_rows_first` +
`qwen4_exp_hc_decode_rows_mix`) against the 1-row GEMV kernels applied row by row, over the
three site kinds the forward sends (after a MoE block: combine + injection; MoE site: injection
+ NVFP4; norm-only). `hc_out` must be bitwise; `mixed` within 1e-2 of the reference's largest
magnitude (1-2 bf16 ulp: the projections take bf16 operands on tensor cores); the injection
weights within one bf16 ulp; the NVFP4 codes may differ where `mixed` does."""

import importlib

import pytest
import torch

P = "tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc."
rows_mod = importlib.import_module(P + "qwen4_exp_hc_decode_rows_tc")
first_op = rows_mod.qwen4_exp_hc_decode_rows_first
mix_op = rows_mod.qwen4_exp_hc_decode_rows_mix
first1 = importlib.import_module(
    P + "qwen4_exp_hc_decode_first_cute"
).qwen4_exp_hc_decode_first_cute
mix1 = importlib.import_module(P + "qwen4_exp_hc_decode_mix_fp4_cute")

assert torch.cuda.is_available(), "the multi-row hyper-connection site requires a CUDA device"

DEV = "cuda"
HIDDEN, C, RANK = 2560, 4, 320
WIDTH = HIDDEN * C
EPS = 1e-6


def _operands(rows, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    bf = torch.bfloat16
    rnd = lambda *s, scale=1.0: (torch.randn(*s, generator=g, device=DEV) * scale).to(bf)  # noqa: E731
    return dict(
        first=rnd(384, WIDTH, scale=0.01).t(),
        up=rnd(WIDTH, RANK, scale=0.05).t(),
        norm=rnd(WIDTH, scale=0.1),
        hc=rnd(rows, WIDTH, scale=0.5),
        block=rnd(rows, HIDDEN, scale=0.5),
        shared=rnd(rows, HIDDEN, scale=0.5),
        gate=rnd(rows, 1),
        inject=(torch.rand(rows, C, generator=g, device=DEV) * 2).to(bf),
        gscale=torch.tensor([448.0 * 6 / 4.0], device=DEV),
    )


def _site(o, site, rows_path: bool):
    """The site through either path: `(hc_out, mixed, inject_out, xq)`."""
    blk, inj = (None, None) if site == "norm" else (o["block"], o["inject"])
    kw = dict(shared=o["shared"], gate_logits=o["gate"]) if site == "combine" else {}
    n_cols = RANK if site == "norm" else RANK + C
    gs = o["gscale"] if site == "inject" else None
    if rows_path:
        hc, low, inj_out, rstd = first_op(
            o["hc"], blk, inj, o["norm"], o["first"], C, n_cols, EPS, **kw
        )
        mixed, xq, _ = mix_op(low, rstd, o["up"], hc, o["norm"], C, gs)
        return hc, mixed, inj_out, xq
    out = []
    for r in range(o["hc"].shape[0]):
        one = lambda t: None if t is None else t[r : r + 1]  # noqa: E731
        h, part, ss = first1(
            one(o["hc"]),
            one(blk),
            one(inj),
            o["norm"],
            o["first"],
            C,
            n_cols,
            **{k: one(v) for k, v in kw.items()},
        )
        if gs is not None:
            m, i, q, _ = mix1.qwen4_exp_hc_decode_mix_fp4_cute(
                part, ss, o["up"], h, o["norm"], EPS, C, True, gs
            )
        else:
            m, i = mix1.qwen4_exp_hc_decode_mix_cute(
                part, ss, o["up"], h, o["norm"], EPS, C, n_cols > RANK
            )
            q = None
        out.append((h, m, i, q))
    cat = lambda k: None if out[0][k] is None else torch.cat([x[k] for x in out])  # noqa: E731
    return cat(0), cat(1), cat(2), cat(3)


def _check(cell):
    rows, site = cell.spec["rows"], cell.spec["site"]
    o = _operands(rows, 1000 + rows)
    got = _site(o, site, True)
    want = _site(o, site, False)
    assert torch.equal(got[0], want[0]), "hc_out bitwise"
    rel = (got[1].float() - want[1].float()).abs().max() / want[1].float().abs().max()
    assert rel <= 1e-2, f"mixed {rel:.2e} of the reference's max"
    if want[2] is not None:
        ulp = (got[2].float() - want[2].float()).abs() / want[2].float().abs().clamp_min(1e-6)
        assert ulp.max() <= 2.0**-7 * 1.01, "injection weights within one bf16 ulp"
    if want[3] is not None:
        assert (got[3] != want[3]).float().mean() < 0.01, "NVFP4 codes"


@pytest.mark.parametrize("cell", first_op.CELLS, ids=[c.why[:44] for c in first_op.CELLS])
def test_first_cells(cell) -> None:
    _check(cell)


@pytest.mark.parametrize("cell", mix_op.CELLS, ids=[c.why[:44] for c in mix_op.CELLS])
def test_mix_cells(cell) -> None:
    _check(cell)


def test_replay_matches_eager() -> None:
    o = _operands(8, 77)
    _site(o, "inject", True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _site(o, "inject", True)
    fresh = _operands(8, 78)
    o["hc"].copy_(fresh["hc"])
    o["block"].copy_(fresh["block"])
    graph.replay()
    torch.cuda.synchronize()
    eager = _site(o, "inject", True)
    assert torch.equal(captured[1], eager[1]) and torch.equal(captured[0], eager[0])
