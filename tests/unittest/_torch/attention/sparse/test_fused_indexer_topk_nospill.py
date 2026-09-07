# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused DSv4 decode indexer + top-K (no logits materialization) vs a pure
PyTorch reference.

Covers the failure modes found while auditing the kernel against real
captures: SIGNED per-head weights (negative scores must order correctly),
the all-negative adversarial case (K-th value == 0.0 with a massive
zero-tie storm), scattered page tables, and per-row ragged context
lengths. Correctness is value-set exactness of the selected scores under
(atol=1e-2, rtol=1e-3) — the kernel emits fp16-rounded values — plus
index uniqueness/range checks.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import fused_indexer_topk_nospill
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="requires Blackwell sm_100/sm_103",
)

PAGE = 32
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _quant_fp4(x: torch.Tensor, gran: int = 32):
    """[M, 128] f32 -> packed [M, 64] u8 + scale words [M] i32 (4x ue8m0)."""
    m, n = x.shape
    xv = x.view(m, n // gran, gran)
    amax = xv.abs().amax(dim=2).clamp_min(1e-4)
    sf = amax / 6.0
    bits = sf.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    sf = (exp << 23).view(torch.float32)
    xs = xv / sf.unsqueeze(2)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(xs.abs().clamp_max(6.0), bounds).to(torch.uint8)
    code = (code | (((xs < 0) & (code != 0)).to(torch.uint8) << 3)).view(m, n)
    packed = (code[:, 0::2] & 0x0F) | ((code[:, 1::2] & 0x0F) << 4)
    sf_words = exp.to(torch.uint8).contiguous().view(torch.int32).reshape(-1)
    return packed.contiguous(), sf_words


def _dequant_fp4(packed: torch.Tensor, sf_words: torch.Tensor, gran: int = 32):
    lut = torch.tensor(FP4_LUT, device=packed.device, dtype=torch.float32)
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    v = lut[codes & 0x07]
    v = torch.where((codes & 0x08) != 0, -v, v)
    exp = sf_words.view(torch.int32).reshape(-1, 1).view(torch.uint8)
    sf = (exp.to(torch.int32) << 23).view(torch.float32)
    g = torch.arange(v.shape[-1], device=packed.device) // gran
    return v * sf[:, g]


def _build_inputs(batch, n_comp, k_top, weight_mode, seed, device, pattern="random"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    maxb = n_comp // PAGE
    nb_total = batch * maxb
    q = torch.randn((batch * 64, 128), generator=g, device=device)
    q_packed, q_sf = _quant_fp4(q)
    kv = torch.randn((nb_total * PAGE, 128), generator=g, device=device)
    if pattern != "random":
        # adversarial score shapes for the safe-line filter: all keys share one
        # direction u so the score is a scalar function of the per-token scale
        u = torch.randn((128,), generator=g, device=device)
        u = u / u.norm()
        t = torch.arange(nb_total * PAGE, device=device, dtype=torch.float32)
        if pattern == "monotone":  # scores grow along the row: every tile beats the line
            c = 0.5 + 2.5 * (t % (n_comp * 1.0)) / n_comp
        elif pattern == "allequal":  # one fp16 tie class over the whole row
            c = torch.full_like(t, 2.0)
        else:  # "giantbin": 70% of the row shares one high value
            c = torch.where(
                torch.rand(t.shape, generator=g, device=device) < 0.7,
                torch.full_like(t, 3.0),
                torch.rand(t.shape, generator=g, device=device),
            )
        kv = c.unsqueeze(1) * u.unsqueeze(0) * 4.0
    kv_packed, kv_sf = _quant_fp4(kv)
    # planar production page: 2048 B data plane then 128 B scale plane
    flat = torch.empty((nb_total, PAGE * 68), device=device, dtype=torch.uint8)
    flat[:, : PAGE * 64] = kv_packed.view(nb_total, PAGE * 64)
    flat[:, PAGE * 64 :] = (
        kv_sf.reshape(nb_total, PAGE, 1)
        .view(torch.int32)
        .view(torch.uint8)
        .reshape(nb_total, PAGE * 4)
    )
    kv_cache = flat.view(nb_total, PAGE, 1, 68)
    lens = torch.randint(
        max((3 * n_comp) // 4, k_top),
        n_comp + 1,
        (batch,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    block_table = (
        torch.randperm(nb_total, generator=g, device=device)
        .to(torch.int32)
        .view(batch, maxb)
        .contiguous()
    )
    weights = torch.randn((batch, 64), generator=g, device=device)
    if weight_mode == "nonneg":
        weights = weights.abs()
    elif weight_mode == "allneg":
        weights = -weights.abs()
    return {
        "q_fp4": q_packed.view(batch, 1, 64, 64),
        "sf_q": q_sf.view(batch, 1, 64),
        "kv_cache": kv_cache,
        "weights": weights,
        "context_lens": lens,
        "block_table": block_table,
    }


def _reference(inp, k_top):
    kvf = inp["kv_cache"].reshape(inp["kv_cache"].shape[0], -1)
    kvp = kvf[:, : PAGE * 64].reshape(-1, 64)
    kvs = kvf[:, PAGE * 64 :].contiguous().view(torch.int32).reshape(-1)
    k = _dequant_fp4(kvp, kvs).reshape(-1, PAGE, 128)
    batch = inp["q_fp4"].shape[0]
    q = _dequant_fp4(
        inp["q_fp4"][:, 0].reshape(batch * 64, 64), inp["sf_q"][:, 0].reshape(batch * 64)
    ).view(batch, 64, 128)
    vals = []
    for i in range(batch):
        length = int(inp["context_lens"][i])
        nb = (length + PAGE - 1) // PAGE
        kx = k[inp["block_table"][i, :nb].long()].reshape(nb * PAGE, 128)
        s = torch.relu(q[i] @ kx.t())
        s = (s * inp["weights"][i].unsqueeze(1)).sum(dim=0)
        s[length:] = float("-inf")
        vals.append(torch.topk(s, k_top).values)
    return torch.stack(vals)


@skip_not_sm100
@pytest.mark.parametrize("batch", [2, 16])
@pytest.mark.parametrize("n_comp", [8192, 16384])
@pytest.mark.parametrize("k_top", [512, 1024])
@pytest.mark.parametrize("weight_mode", ["signed", "nonneg", "allneg"])
def test_fused_indexer_topk_nospill(batch, n_comp, k_top, weight_mode):
    device = torch.device("cuda")
    inp = _build_inputs(batch, n_comp, k_top, weight_mode, seed=1234, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: max value err {float(dv.max()):.4f}"
        )


@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize("k_top", [512, 1024])
def test_fused_indexer_topk_nospill_cuda_graph(batch, k_top):
    """The op must be CUDA-graph capturable: capture one launch after a
    warm-up (compile happens on first call and must stay outside capture),
    replay, and require the replay's VALUE SET to match eager. Indices may
    legally differ inside boundary-tie classes (atomic claim order), so the
    contract is value-set equality plus index validity, same as the base
    test. Measured motivation: graph replay removes ~6-7us of host/launch
    overhead per call at small batch (B<=32)."""
    # Exercise the registered custom op when the full package is
    # importable (CI); fall back to the kernel entry point under the
    # stub-injected local runner -- the captured/replayed launch is the
    # same either way.
    try:
        from tensorrt_llm._torch.attention.backends.sparse.dsa import (  # noqa: F401,E501
            custom_ops as _dsa_custom_ops,
        )

        _use_op = True
    except Exception:
        _use_op = False

    device = torch.device("cuda")
    n_comp = 8192
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=77, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)

    def call():
        if _use_op:
            torch.ops.trtllm.dsa_fused_indexer_topk_decode(
                inp["q_fp4"],
                inp["sf_q"],
                inp["kv_cache"],
                inp["weights"],
                inp["context_lens"],
                inp["block_table"],
                indices,
                values,
            )
        else:
            fused_indexer_topk_nospill.run(
                inp["q_fp4"],
                inp["sf_q"],
                inp["kv_cache"],
                inp["weights"],
                inp["context_lens"],
                inp["block_table"],
                None,
                indices,
                values,
            )

    call()  # warm-up: compile + autotune outside capture
    torch.cuda.synchronize()
    eager_vals = values.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    indices.fill_(-3)
    values.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices after replay"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(eager_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: replay value set diverged, max err {float(dv.max()):.4f}"
        )


@skip_not_sm100
@pytest.mark.parametrize("batch", [1, 4, 16, 36, 70])
@pytest.mark.parametrize("n_comp", [16384, 65536])
@pytest.mark.parametrize("k_top", [1024])
def test_fused_indexer_topk_nospill_split(batch, n_comp, k_top, monkeypatch):
    # cluster-free row split forced on short rows: S = SMs / batch co-resident CTAs per
    # row merge through the GMEM workspace (ragged tile counts, non-power-of-two S)
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", "1")
    device = torch.device("cuda")
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=31, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: max value err {float(dv.max()):.4f}"
        )
    _check_fp32_boundary(inp, indices, k_top)


@skip_not_sm100
def test_fused_indexer_topk_nospill_split_cuda_graph(monkeypatch):
    # the split's workspace must be clean at every replay: the last CTA of a row re-zeroes
    # it inside the launch, so three replays after capture must reproduce eager
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", "1")
    device = torch.device("cuda")
    batch, n_comp, k_top = 4, 65536, 1024
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=93, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)

    def call():
        fused_indexer_topk_nospill.run(
            inp["q_fp4"],
            inp["sf_q"],
            inp["kv_cache"],
            inp["weights"],
            inp["context_lens"],
            inp["block_table"],
            None,
            indices,
            values,
        )

    call()
    torch.cuda.synchronize()
    eager_vals = values.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        indices.fill_(-3)
        values.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for i in range(batch):
            row = indices[i]
            assert int(row.min()) >= 0
            assert row.unique().numel() == k_top, "duplicate indices after replay"
            got, _ = torch.sort(values[i], descending=True)
            want, _ = torch.sort(eager_vals[i], descending=True)
            assert bool(((got - want).abs() <= 1e-2 + 1e-3 * want.abs()).all()), (
                f"row {i}: replay value set diverged"
            )
    # eager launches and replays share the workspace: interleaving must stay clean
    call()
    torch.cuda.synchronize()
    indices.fill_(-3)
    values.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for i in range(batch):
        assert indices[i].unique().numel() == k_top
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(eager_vals[i], descending=True)
        assert bool(((got - want).abs() <= 1e-2 + 1e-3 * want.abs()).all())


@skip_not_sm100
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
@pytest.mark.parametrize("n_comp", [65536, 131072, 262144])
@pytest.mark.parametrize("k_top", [512, 1024])
def test_fused_indexer_topk_nospill_long_context(batch, n_comp, k_top):
    # Rows longer than one CTA's key budget are split across a cluster whose
    # size follows the row length (up to 16 CTAs = 1M-token context at
    # compress ratio 4); each CTA stages only its own tiles' page-table slice.
    device = torch.device("cuda")
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=4321, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: max value err {float(dv.max()):.4f}"
        )


@skip_not_sm100
@pytest.mark.parametrize("pattern", ["random", "monotone", "allequal", "giantbin"])
@pytest.mark.parametrize("batch", [2, 16, 64, 80])
@pytest.mark.parametrize("k_top", [512, 1024])
def test_fused_indexer_topk_nospill_filtered(pattern, batch, k_top, monkeypatch):
    # Force the safe-line filter path on short rows: an 8-tile dense prefix,
    # everything after it filtered/compacted. Adversarial score shapes make
    # every chunk trigger a compaction (monotone) or a single giant tie class.
    # batch 80 runs one CTA per row (CS=1), so the local and final K-th bins coincide.
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_NDENSE", "8")
    device = torch.device("cuda")
    weight_mode = "nonneg" if pattern != "random" else "signed"
    inp = _build_inputs(batch, 16384, k_top, weight_mode, seed=99, device=device, pattern=pattern)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: max value err {float(dv.max()):.4f}"
        )


def _reference_indices(inp, k_top):
    """fp32 reference top-K indices and full score rows (same math as _reference)."""
    kvf = inp["kv_cache"].reshape(inp["kv_cache"].shape[0], -1)
    kvp = kvf[:, : PAGE * 64].reshape(-1, 64)
    kvs = kvf[:, PAGE * 64 :].contiguous().view(torch.int32).reshape(-1)
    k = _dequant_fp4(kvp, kvs).reshape(-1, PAGE, 128)
    batch = inp["q_fp4"].shape[0]
    q = _dequant_fp4(
        inp["q_fp4"][:, 0].reshape(batch * 64, 64), inp["sf_q"][:, 0].reshape(batch * 64)
    ).view(batch, 64, 128)
    rows = []
    for i in range(batch):
        length = int(inp["context_lens"][i])
        nb = (length + PAGE - 1) // PAGE
        kx = k[inp["block_table"][i, :nb].long()].reshape(nb * PAGE, 128)
        s = torch.relu(q[i] @ kx.t())
        s = (s * inp["weights"][i].unsqueeze(1)).sum(dim=0)
        s[length:] = float("-inf")
        rows.append(s)
    return rows


@skip_not_sm100
@pytest.mark.parametrize("batch", [2, 16])
@pytest.mark.parametrize("n_comp", [8192, 16384])
@pytest.mark.parametrize("k_top", [512, 1024])
def test_fused_indexer_topk_nospill_fp32_boundary(batch, n_comp, k_top, monkeypatch):
    # fp32 boundary refinement (default on): the selected INDEX set equals the fp32
    # top-K set; members may differ only inside a genuine fp32 tie class at the
    # boundary (scores equal to within 1e-6 relative).
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_FP32_EXACT", "1")
    device = torch.device("cuda")
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=777, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    _check_fp32_boundary(inp, indices, k_top)


def _check_fp32_boundary(inp, indices, k_top, tie_cap=2048):
    scores = _reference_indices(inp, k_top)
    for i in range(indices.shape[0]):
        s = scores[i]
        ref = torch.topk(s, k_top).indices
        kth = s[ref[-1]]
        got = indices[i].long()
        assert got.unique().numel() == k_top
        if int((s.half() == kth.half()).sum()) > tie_cap:
            # a boundary tie class above the cap keeps the fp16 fill: exact at fp16 granularity
            assert int((s[got].half() < kth.half()).sum()) == 0, (
                f"row {i}: token below the fp16 K-th value"
            )
            gh, _ = torch.sort(s[got].half(), descending=True)
            rh, _ = torch.sort(s[ref].half(), descending=True)
            assert bool((gh == rh).all()), f"row {i}: fp16 value multiset differs"
            continue
        # every selected token must score >= the fp32 K-th value (within 1e-6 rel)
        tol = 1e-6 * kth.abs() + 1e-6
        below = (s[got] < kth - tol).sum().item()
        assert below == 0, f"row {i}: {below} selected tokens fall below the fp32 K-th value"
        # and the sorted score multiset must match the reference to fp32 tolerance
        gs, _ = torch.sort(s[got], descending=True)
        rs, _ = torch.sort(s[ref], descending=True)
        assert bool(((gs - rs).abs() <= 1e-6 * rs.abs() + 1e-6).all()), (
            f"row {i}: fp32 score multiset differs"
        )


@pytest.mark.parametrize("pattern", ["random", "monotone", "giantbin"])
@pytest.mark.parametrize("k_top", [512, 1024])
def test_fused_indexer_topk_nospill_fp32_boundary_filtered(pattern, k_top, monkeypatch):
    # fp32 exactness on filtered rows at CS=1 (batch 80, 8-tile dense prefix):
    # compaction must keep every member of a tie class the refinement rescores
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_NDENSE", "8")
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_FP32_EXACT", "1")
    device = torch.device("cuda")
    batch = 80
    weight_mode = "nonneg" if pattern != "random" else "signed"
    inp = _build_inputs(batch, 16384, k_top, weight_mode, seed=4242, device=device, pattern=pattern)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    _check_fp32_boundary(inp, indices, k_top)


@skip_not_sm100
def test_fused_indexer_topk_nospill_routing(monkeypatch):
    # measured on B200 (148 SMs): clusters of 8/16 fit one wave only up to 64 CTAs, the
    # GMEM split wins at every row length once 8 rows share the chip
    if torch.cuda.get_device_properties(0).multi_processor_count != 148:
        pytest.skip("routing table measured for 148 SMs")
    monkeypatch.delenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", raising=False)
    fused_indexer_topk_nospill._cfg.clear()
    cases = {  # (batch, block-table width in 32-token pages) -> (cluster size, split)
        (1, 256): (16, 0),
        (4, 256): (16, 0),
        (8, 256): (8, 0),
        (8, 2048): (18, 1),
        (16, 256): (4, 0),
        (4, 2048): (16, 0),
        (32, 256): (4, 0),
        (64, 256): (2, 0),
        (128, 256): (1, 0),
        (1, 4096): (148, 1),
        (4, 4096): (37, 1),
        (32, 4096): (4, 0),
        (60, 4096): (2, 0),
        # off the power-of-two grid the split places 1.2x+ the CTAs: 22 -> 6x22, 36 -> 4x36, 70 -> 2x70
        (22, 4096): (6, 1),
        (36, 4096): (4, 1),
        (70, 4096): (2, 1),
        (25, 2048): (5, 1),
        (22, 256): (4, 0),
    }
    for (batch, maxb), (cs, gm) in cases.items():
        key, _, _ = fused_indexer_topk_nospill._config(batch, maxb, 4096, 1024)
        assert (key[6], key[12]) == (cs, gm), (batch, maxb, key[6], key[12])


@skip_not_sm100
@pytest.mark.parametrize("batch", [4, 8, 16])
@pytest.mark.parametrize("split", ["auto", "1"])
def test_fused_indexer_topk_nospill_mixed_lengths(batch, split, monkeypatch):
    # rows of very different lengths in one launch: the split shares the CTAs out by length
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", split)
    device = torch.device("cuda")
    n_comp, k_top = 65536, 1024
    inp = _build_inputs(batch, n_comp, k_top, "signed", seed=2718, device=device)
    lens = torch.tensor(
        [max(k_top, n_comp >> (i % 4)) - 37 * i for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    inp["context_lens"] = lens
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0 and int(row.max()) < int(lens[i])
        assert row.unique().numel() == k_top
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        assert bool(((got - want).abs() <= 1e-2 + 1e-3 * want.abs()).all()), f"row {i}"
