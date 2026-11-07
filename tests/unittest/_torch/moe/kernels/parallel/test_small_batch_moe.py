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
"""Small-batch NVFP4 routed-experts kernel (``trtllm::cute_dsl_nvfp4_small_batch_moe_rubin``).

Against a float64 reference built from the dequantized operands:

* FC1 + SiTU + NVFP4 requantization + FC2 + combine: the output matches the routing-weighted
  down-projections of the reference intermediate after the same NVFP4 requantization (a few
  elements may round to the neighbouring FP4 value).
* The red combine (the default) is within the BF16 accumulation bound of the slice combine.
* The slice combine does not depend on the schedule (bitwise for other persistent CTA counts),
  and the token-indexed layout gives the permuted layout's bits.
* Every output element is written and the counters are zero after every launch, also over CUDA
  graph replays.
"""

import pytest
import torch
from _torch.helpers import (
    SMALL_BATCH_MOE_SITU_BETA,
    SMALL_BATCH_MOE_SITU_LINEAR_BETA,
    run_small_batch_moe,
    small_batch_moe_operands,
)

from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._utils import get_sm_version

SF_VEC = 16

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the kernel targets the SM100 family (dispatched on SM107)",
)

SHAPES = {
    # hidden, intermediate per rank, experts, local experts, local offset, top-k
    "k3_ep16": (3584, 3072, 896, 56, 392, 16),
    "k3_tp4ep4": (3584, 768, 896, 224, 224, 16),
    "small": (1024, 256, 64, 16, 0, 8),
}


def _fp4_to_f64(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    p = packed.view(torch.uint8).long()
    return torch.stack([lut[p & 0xF], lut[p >> 4]], dim=-1).flatten(-2)


def _e4m3_to_f64(sf: torch.Tensor) -> torch.Tensor:
    return sf.view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)


def _situ(up, gate):
    beta, linear_beta = SMALL_BATCH_MOE_SITU_BETA, SMALL_BATCH_MOE_SITU_LINEAR_BETA
    g = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    u = linear_beta * torch.tanh(up / linear_beta)
    return g * u


def _setup(shape, num_tokens, seed, local_tokens=True):
    hidden, inter_size, num_experts, num_local, offset, top_k = SHAPES[shape]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=gen)
    scales, ids = logits.topk(top_k, dim=-1)
    ids = ids.to(torch.int32)
    scales = scales.softmax(dim=-1).to(torch.float32)
    if local_tokens:
        ids[0, 0] = offset  # at least one token on this rank
    else:
        ids.fill_((offset + num_local) % num_experts)  # every token on another rank
    return small_batch_moe_operands(
        ids, scales, hidden, inter_size, num_experts, num_local, offset, gen
    )


def _counters():
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_small_batch_moe_custom_op import (
        new_small_batch_moe_counters,
    )

    return new_small_batch_moe_counters(torch.device("cuda"))


def _rows(d):
    """(tile, permuted row) of every expanded slot served on this rank."""
    num_tiles = int(d["nnet"].item())
    limits = d["t2lim"][:num_tiles].tolist()
    return [(t, r) for t in range(num_tiles) for r in range(t * 128, limits[t])]


def _dequant_w(w, sf, e, rows, cols):
    lin = unswizzle_sf(sf[e].contiguous().flatten(), rows, cols, SF_VEC).view(rows, cols // SF_VEC)
    return _fp4_to_f64(w[e]) * _e4m3_to_f64(lin).repeat_interleave(SF_VEC, dim=1)


def _reference(d, num_tokens):
    """float64 FC1 + SiTU per served slot, and the routed output of those intermediates."""
    hidden, inter_size, top_k = d["hidden"], d["inter_size"], d["top_k"]
    rows = _rows(d)
    row_idx = torch.tensor([r for _, r in rows], device="cuda")
    tiles = torch.tensor([t for t, _ in rows], device="cuda")
    expanded = d["p2e"][row_idx].long()
    tokens, slots = expanded // top_k, expanded % top_k
    experts = d["t2g"][tiles].long()
    x_deq = _fp4_to_f64(d["x"]) * _e4m3_to_f64(d["x_sf"]).repeat_interleave(SF_VEC, dim=1)
    h_ref = torch.empty(len(rows), inter_size, dtype=torch.float64, device="cuda")
    for e in experts.unique().tolist():
        sel = (experts == e).nonzero().flatten()
        w1 = _dequant_w(d["w1"], d["w1_sf"], e, 2 * inter_size, hidden)
        acc = (x_deq[tokens[sel]] @ w1.t()) * float(d["alpha1"][e])
        acc = acc.view(len(sel), -1, 2, 16)  # [gate16, up16] row groups
        h_ref[sel] = _situ(acc[:, :, 1], acc[:, :, 0]).reshape(len(sel), -1)
    return rows, tokens, slots, experts, h_ref


def _nvfp4_roundtrip(h, global_sf):
    """``h`` quantized like the kernel's FC1 epilogue (16-element blocks, E4M3 scales of
    ``amax / 6 * global_sf``), dequantized in the scaled domain the FC2 alphas expect."""
    grid = torch.tensor(_E2M1[:8], dtype=torch.float64, device=h.device)
    blocks = h.view(h.size(0), -1, SF_VEC) * global_sf
    amax = blocks.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0).float().to(torch.float8_e4m3fn).double()
    v = torch.where(sf > 0, blocks / sf, torch.zeros_like(blocks)).clamp(-6.0, 6.0)
    q = grid[(v.abs()[..., None] - grid).abs().argmin(-1)] * v.sign()
    return (q * sf).view_as(h)


def _routed_sum(d, num_tokens, h, tokens, slots, experts):
    """float64 routed output of the intermediates ``h`` and the magnitude sum of its terms."""
    hidden, inter_size = d["hidden"], d["inter_size"]
    partial = torch.empty(len(tokens), hidden, dtype=torch.float64, device="cuda")
    for e in experts.unique().tolist():
        sel = (experts == e).nonzero().flatten()
        w2 = _dequant_w(d["w2"], d["w2_sf"], e, hidden, inter_size)
        partial[sel] = h[sel] @ w2.t() * float(d["alpha2"][e])
    terms = partial * d["scales"][tokens, slots].double()[:, None]
    ref = torch.zeros(num_tokens, hidden, dtype=torch.float64, device="cuda")
    mag = torch.zeros_like(ref)
    ref.index_add_(0, tokens, terms)
    mag.index_add_(0, tokens, terms.abs())
    return ref, mag


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_tokens", [1, 5, 8, 13, 16, 27, 32])
def test_small_batch_moe_matches_reference(shape, num_tokens):
    """Permuted layout, slice combine: FC1 and the output against the float64 reference, schedule
    independence, CUDA graph replays; the default red combine within the BF16 accumulation bound."""
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_small_batch_moe_custom_op import (
        small_batch_moe_supports,
    )

    d = _setup(shape, num_tokens, 1234 + num_tokens)
    hidden, inter_size = d["hidden"], d["inter_size"]
    assert small_batch_moe_supports(inter_size, hidden)
    counters = _counters()
    out = torch.full((num_tokens, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    run_small_batch_moe(d, out, counters, token_indexed=False, deterministic=True)
    torch.cuda.synchronize()
    assert int(counters.abs().sum().item()) == 0, "counters not reset"
    assert bool(torch.isfinite(out.float()).all()), "output not fully written"

    rows, tokens, slots, experts, h_ref = _reference(d, num_tokens)
    h_q = _nvfp4_roundtrip(h_ref, float(d["global_sf"][0]))
    assert float((h_q != 0).float().mean()) > 0.5, "intermediate mostly zero: scales too small"
    ref, mag = _routed_sum(d, num_tokens, h_q, tokens, slots, experts)
    assert float((ref[tokens.unique()] != 0).float().mean()) > 0.9, "reference mostly zero"
    rel = float((out.double() - ref).norm() / ref.norm())
    assert rel < 0.03, f"relative error vs the float64 reference {rel:.3e}"

    # schedule independence: other persistent CTA counts give the same bits
    for ctas in (1, 37):
        out2 = torch.full_like(out, float("nan"))
        run_small_batch_moe(
            d, out2, counters, token_indexed=False, deterministic=True, num_ctas=ctas
        )
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), out2.view(torch.int16)), f"ctas={ctas}"
        assert int(counters.abs().sum().item()) == 0

    # CUDA graph replays
    out_g = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_small_batch_moe(d, out_g, counters, token_indexed=False, deterministic=True)
    for _ in range(3):
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), out_g.view(torch.int16)), "graph replay"
        assert int(counters.abs().sum().item()) == 0, "counters not reset after a replay"

    # red combine: the slice-combine result within the BF16 accumulation bound of its terms
    out_r = torch.full_like(out, float("nan"))
    run_small_batch_moe(d, out_r, counters, token_indexed=False, deterministic=False)
    torch.cuda.synchronize()
    assert int(counters.abs().sum().item()) == 0
    bound = d["top_k"] * 2.0**-8 * mag + 2.0**-8 * out.double().abs() + 1e-30
    assert bool(((out_r.double() - out.double()).abs() <= bound).all()), "red combine"


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_tokens", [1, 8, 13, 32])
def test_small_batch_moe_token_indexed_matches_permuted(shape, num_tokens):
    """The token-indexed layout computes the permuted layout's FP32 sums: with the slice combine
    the bits are equal while both schedules split the tiles into the same slices (at most 16
    tiles), else within the BF16 rounding of the other slice order; for any CTA count and over
    graph replays."""
    d = _setup(shape, num_tokens, 2468 + num_tokens)
    counters = _counters()
    ref = torch.full((num_tokens, d["hidden"]), float("nan"), dtype=torch.bfloat16, device="cuda")
    run_small_batch_moe(d, ref, counters, token_indexed=False, deterministic=True)
    same_slices = int(d["nnet"].item()) <= 16
    for ctas in (0, 1, 37):
        out = torch.full_like(ref, float("nan"))
        run_small_batch_moe(d, out, counters, token_indexed=True, deterministic=True, num_ctas=ctas)
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, f"counters not reset (ctas={ctas})"
        if same_slices:
            assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), f"ctas={ctas}"
        else:
            torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-7, atol=1e-6)
    out_g = torch.empty_like(ref)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_small_batch_moe(d, out_g, counters, token_indexed=True, deterministic=False)
    for _ in range(2):
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0
        a, b = out_g.float(), ref.float()
        assert float((a - b).norm() / b.norm().clamp_min(1e-30)) < 1e-2, "red combine replay"


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_tokens", [1, 8, 13, 32])
def test_small_batch_moe_published_routing_matches_tables(shape, num_tokens):
    """Tables built in the kernel from the published top-k are moe_sort's: bitwise the
    token-indexed launch on the tables with the slice combine, for any CTA count; the counters
    (the routing word too) are zero after every launch, also over graph replays."""
    d = _setup(shape, num_tokens, 1357 + num_tokens)
    counters = _counters()
    ref = torch.full((num_tokens, d["hidden"]), float("nan"), dtype=torch.bfloat16, device="cuda")
    run_small_batch_moe(d, ref, counters, token_indexed=True, deterministic=True)
    for ctas in (0, 1, 37):
        out = torch.full_like(ref, float("nan"))
        run_small_batch_moe(
            d, out, counters, token_indexed=True, deterministic=True, num_ctas=ctas, published=True
        )
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, f"counters not reset (ctas={ctas})"
        assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), f"ctas={ctas}"
    out_g = torch.empty_like(ref)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_small_batch_moe(
            d, out_g, counters, token_indexed=True, deterministic=True, published=True
        )
    for _ in range(2):
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0
        assert torch.equal(ref.view(torch.int16), out_g.view(torch.int16)), "graph replay"


@pytest.mark.parametrize("token_indexed", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_small_batch_moe_no_local_expert(token_indexed, deterministic):
    """No token on this rank: the kernel zeroes the output."""
    d = _setup("k3_ep16", 8, 7, local_tokens=False)
    assert int(d["nnet"].item()) == 0
    counters = _counters()
    out = torch.full((8, d["hidden"]), 3.0, dtype=torch.bfloat16, device="cuda")
    run_small_batch_moe(d, out, counters, token_indexed=token_indexed, deterministic=deterministic)
    torch.cuda.synchronize()
    assert int(out.abs().sum().item()) == 0
    assert int(counters.abs().sum().item()) == 0


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32])
@pytest.mark.parametrize("layout", ["permuted", "token_indexed", "published"])
def test_small_batch_moe_die_major_matches_default(shape, num_tokens, layout, monkeypatch):
    """Die-major weights with per-domain claiming: the same units and arithmetic as the default
    layout, so with the slice combine the output is bitwise the default's, for any CTA count and
    over graph replays; counters reset. ``published``: the tables come from the published top-k
    and the first claim precedes them. On a device without two locality domains the weights
    stay in default memory and a synthetic SM -> domain table drives the claiming."""
    from tensorrt_llm._torch.locality_domain import placement
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_small_batch_moe_custom_op import (
        die_major_layout,
    )

    token_indexed, published = layout != "permuted", layout == "published"
    d = _setup(shape, num_tokens, 8642 + num_tokens)
    counters = _counters()
    ref = torch.full((num_tokens, d["hidden"]), float("nan"), dtype=torch.bfloat16, device="cuda")
    run_small_batch_moe(d, ref, counters, token_indexed=token_indexed, deterministic=True)

    two_domains = is_locality_domain_enabled()
    if not two_domains:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        domains = tuple((sm >> 1) & 1 for sm in range(num_sms))
        monkeypatch.setattr(placement, "sm_locality_domains", lambda device_index: domains)
    dm = dict(d)
    for key in ("w1", "w1_sf", "w2", "w2_sf"):
        layout = die_major_layout(d[key])
        if two_domains:
            layout = placement.split_domain_copy(layout, layout.numel() // 2)
        dm[key] = layout.view(d[key].shape)

    for ctas in (0, 1, 37):
        out = torch.full_like(ref, float("nan"))
        run_small_batch_moe(
            dm,
            out,
            counters,
            token_indexed=token_indexed,
            deterministic=True,
            num_ctas=ctas,
            die_major=True,
            published=published,
        )
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0, f"counters not reset (ctas={ctas})"
        assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), f"bits (ctas={ctas})"
    out_g = torch.empty_like(ref)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_small_batch_moe(
            dm,
            out_g,
            counters,
            token_indexed=token_indexed,
            deterministic=True,
            die_major=True,
            published=published,
        )
    for _ in range(2):
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int16), out_g.view(torch.int16)), "graph replay"
        assert int(counters.abs().sum().item()) == 0
    out_r = torch.full_like(ref, float("nan"))
    run_small_batch_moe(
        dm,
        out_r,
        counters,
        token_indexed=token_indexed,
        deterministic=False,
        die_major=True,
        published=published,
    )
    torch.cuda.synchronize()
    assert int(counters.abs().sum().item()) == 0
    a, b = out_r.float(), ref.float()
    assert float((a - b).norm() / b.norm().clamp_min(1e-30)) < 1e-2, "red combine"
