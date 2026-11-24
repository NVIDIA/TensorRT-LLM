# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's GEMM pins on one sm_107 GPU: every pinned site keeps today's bytes at every row count a
helix8 generation step runs.

A helix8 generation rank's steps run 6 rows per request (k5: the request's token and five drafts),
6 to 144 rows (b1 to b24). For each site of ``_GEMM_PIN_TABLE`` and each of those row counts,
``_gemm_pinned`` with the committed pins runs eagerly twice (the first call of a pinned row count
checks the pin on the device, the second runs it), then a CUDA graph is captured on one set of
inputs and replayed on two others; every output equals the site's call with no pin
(``_GEMM_PINS`` empty) on the same inputs, word for word. At the pinned row counts the check
passes on this device and the captured call runs the entry. The site groups ship off; each case
pins its site as the table does (``_GEMM_PINS`` of that site's rows).

The A-class pins (``_GEMM_PIN_TABLE_A``, another summation order) pass their band check on this
device, stay within 2^-6 of today's largest magnitude on every input, repeat their own words, and a
captured call replays the eager pinned call's words on new inputs. Each site's call is the one its
``_gemm_pinned`` call in modeling.py passes as ``today`` (``test_rubin_helix8_gemm_pins.py`` checks
the wiring); the operands are seeded at the site's shape, and the bytes compared are the outputs.
"""

from __future__ import annotations

import importlib
import zlib

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_HELIX8 = _PKG + ".kimi_k31_pro_nvfp4__sm_107__helix8.modeling"
# A helix8 generation step's rows: 6 per request (k5), b1 to b24.
_ROWS = tuple(range(6, 145, 6))
#: Each pinned site: (N, K) at helix8's share and the call it replaces (``uv``: W_UV over the 96
#: heads as strided batches, into the value heads' rows).
_SITES = {
    "moe_latent_up": (7168, 3584, "mm"),
    "moe_latent_down": (3584, 7168, "mm"),
    "moe_router": (896, 7168, "router"),
    "mla_qkv_a": (2112, 7168, "mm"),
    "mla_wuv": (128, 512, "uv"),
    "mla_o": (7168, 1536, "linear"),
    "mla_idx_wq_b": (4096, 1536, "mm"),
    "dense_gu": (8448, 7168, "mm"),
    "dense_dn": (7168, 4224, "mm"),
    "draft_q_b": (18432, 1536, "linear"),
    "draft_o": (7168, 1536, "linear"),
    "draft_gate_up": (3584, 7168, "linear"),
    "draft_down": (7168, 1792, "linear"),
    "draft_attn_conv_proj": (1792, 7168, "linear"),
    "draft_mlp_conv_proj": (1792, 7168, "linear"),
}
_HEADS = 96

_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="helix8's kernels run on sm_107",
)


class _Site:
    """One site's operands at ``rows`` rows (weights seeded once, rows reloaded per input) and its
    call through the module's ``_gemm_pinned``, as modeling.py makes it."""

    def __init__(self, mod, site: str, rows: int):
        self.mod, self.site, self.rows = mod, site, rows
        n, k, self.kind = _SITES[site]
        gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(site.encode()))
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        if self.kind == "uv":
            kv_b = torch.randn(_HEADS * n, k, generator=gen, device="cuda") * k**-0.5
            self.b = kv_b.to(torch.bfloat16).view(_HEADS, n, k).transpose(1, 2)
            self.x = torch.empty(rows, _HEADS, k, **bf16)
            self.out = torch.empty(rows, _HEADS, n, **bf16)
        else:
            self.w = (torch.randn(n, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
            self.b = self.w.t()
            self.x = torch.empty(rows, k, **bf16)

    def inputs(self, seed: int) -> torch.Tensor:
        """A unit normal at a scale per row of 2^-6..2^6, a zero row and a tiny row."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(self.x.shape, generator=gen, device="cuda")
        exponent = (torch.arange(self.rows, device="cuda") * seed % 13 - 6).float()
        x *= torch.pow(2.0, exponent).view(-1, *[1] * (x.dim() - 1))
        x[seed % self.rows] = 0.0
        x[(seed + 3) % self.rows] *= 2.0**-120
        return x.to(torch.bfloat16)

    def today(self) -> torch.Tensor:
        mod, x = self.mod, self.x
        if self.kind == "router":
            return mod.dsv3_router_gemm_op(x, self.b, None, torch.float32)
        if self.kind == "linear":
            return mod.rubin_linear(x, self.w, False)
        if self.kind == "uv":
            return mod.bmm_out(x.transpose(0, 1), self.b, self.out.transpose(0, 1))
        return mod.cublas_mm(x, self.b)

    def run(self) -> torch.Tensor:
        if self.kind == "uv":
            a, heads = self.x.transpose(0, 1), self.out.transpose(0, 1)
            return self.mod._gemm_pinned(self.site, a, self.b, self.today, out=heads)
        dtype = torch.float32 if self.kind == "router" else torch.bfloat16
        return self.mod._gemm_pinned(self.site, self.x, self.b, self.today, out_dtype=dtype)


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32).clone()


@_ON_SM107
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_pinned_site_keeps_todays_bytes_eager_and_on_graph_replays(site, rows, monkeypatch):
    mod = importlib.import_module(_HELIX8)
    pinned = rows in mod._GEMM_PIN_TABLE[site]
    s = _Site(mod, site, rows)
    inputs = [s.inputs(seed) for seed in (1, 2, 3)]
    # Today's call on every input: no pin.
    with monkeypatch.context() as m:
        m.setattr(mod, "_GEMM_PINS", {})
        want = []
        for x in inputs:
            s.x.copy_(x)
            want.append(_words(s.run()))
    monkeypatch.setattr(
        mod, "_GEMM_PINS", {(site, r): c for r, c in mod._GEMM_PIN_TABLE[site].items()}
    )
    monkeypatch.setattr(mod, "_GEMM_PINS_A_CLASS", frozenset())
    monkeypatch.setattr(mod, "_GEMM_PIN_CHECKED", {})
    calls = []
    entry = mod.rubin_lt_gemm

    def counted(*args, **kwargs):
        calls.append(args[3])
        return entry(*args, **kwargs)

    monkeypatch.setattr(mod, "rubin_lt_gemm", counted)
    for x, w in zip(inputs[1:], want[1:]):
        s.x.copy_(x)
        assert torch.equal(_words(s.run()), w), f"{site} at {rows} rows, eager"
    if pinned:
        assert mod._GEMM_PIN_CHECKED == {(site, rows): True}, (
            f"{site}'s pin at {rows} rows failed its check on this device"
        )
    s.x.copy_(inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = s.run()
    torch.cuda.synchronize()
    for x, w in zip(inputs[1:], want[1:]):
        s.x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_words(out), w), f"{site} at {rows} rows, graph replay"
    # the check's call, the second eager call and the capture
    assert len(calls) == (3 if pinned else 0), calls


#: ``_GEMM_PIN_TABLE_A``'s (site, rows), checked against the table in each case.
_A_CLASS = (
    ("dense_gu", 18),
    ("dense_gu", 24),
    ("draft_gate_up", 12),
    ("draft_gate_up", 54),
    ("draft_gate_up", 60),
    ("draft_gate_up", 66),
    ("draft_gate_up", 72),
    ("mla_idx_wq_b", 12),
    ("mla_qkv_a", 12),
    ("moe_latent_down", 12),
    ("moe_latent_down", 54),
    ("moe_latent_down", 60),
    ("moe_latent_down", 66),
    ("moe_latent_down", 72),
    ("moe_latent_up", 12),
)


@_ON_SM107
@pytest.mark.parametrize("site, rows", _A_CLASS)
def test_an_a_class_pin_stays_in_band_and_replays_its_own_words(site, rows, monkeypatch):
    mod = importlib.import_module(_HELIX8)
    table = sorted((t, r) for t, pins in mod._GEMM_PIN_TABLE_A.items() for r in pins)
    assert table == sorted(_A_CLASS), "the A-class cases are not the table's"
    s = _Site(mod, site, rows)
    inputs = [s.inputs(seed) for seed in (1, 2, 3)]
    with monkeypatch.context() as m:
        m.setattr(mod, "_GEMM_PINS", {})
        want = []
        for x in inputs:
            s.x.copy_(x)
            want.append(s.run().float().clone())
    monkeypatch.setattr(mod, "_GEMM_PINS", {(site, rows): mod._GEMM_PIN_TABLE_A[site][rows]})
    monkeypatch.setattr(mod, "_GEMM_PINS_A_CLASS", frozenset({(site, rows)}))
    monkeypatch.setattr(mod, "_GEMM_PIN_CHECKED", {})
    s.x.copy_(inputs[0])
    s.run()  # the check: today's output, the pin compared in its band
    assert mod._GEMM_PIN_CHECKED == {(site, rows): True}, f"{site} at {rows}: band check failed"
    pinned = []
    for x, w in zip(inputs, want):
        s.x.copy_(x)
        got = s.run().float().clone()
        again = s.run().float().clone()
        assert torch.equal(got, again), f"{site} at {rows} rows: a repeat differs"
        assert torch.isfinite(got).all()
        bound = 2.0**-6 * w.abs().max()
        assert (got - w).abs().max() <= bound, f"{site} at {rows} rows: outside 2^-6 of today"
        pinned.append(got)
    s.x.copy_(inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = s.run()
    torch.cuda.synchronize()
    for x, w in zip(inputs[1:], pinned[1:]):
        s.x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.float(), w), f"{site} at {rows} rows, graph replay"
