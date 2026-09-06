# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DeepSeek V3.2 (FP8) decode indexer + top-K vs a pure PyTorch reference.

Pages of 32 tokens hold 128 e4m3 bytes per token followed by one fp32 scale per
token; the query is plain e4m3. Correctness is value-set exactness of the selected
scores (the kernel emits fp16-rounded values) plus index uniqueness/range checks,
for signed / non-negative / all-negative head weights, scattered page tables and
ragged context lengths, on the single-CTA, cluster and GMEM-split paths.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import fused_indexer_topk_nospill_fp8
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="requires Blackwell sm_100/sm_103",
)

PAGE = 32
ROWB = 128  # fp8 bytes per token row
PGB = PAGE * (ROWB + 4)


def _to_fp8_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def _from_fp8_bytes(u: torch.Tensor) -> torch.Tensor:
    return u.view(torch.float8_e4m3fn).float()


def _build_inputs(batch, n_comp, k_top, weight_mode, seed, device, pattern="random"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    maxb = n_comp // PAGE
    nb_total = batch * maxb
    q = torch.randn((batch * 64, 128), generator=g, device=device)
    kv = torch.randn((nb_total * PAGE, 128), generator=g, device=device)
    scale = 0.5 + torch.rand((nb_total * PAGE,), generator=g, device=device)
    if pattern != "random":
        # adversarial score shapes for the safe-line filter: every key shares one direction
        # u, so the score is a scalar function of the per-token magnitude c
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
        scale = torch.ones_like(scale)
    flat = torch.empty((nb_total, PGB), device=device, dtype=torch.uint8)
    flat[:, : PAGE * ROWB] = _to_fp8_bytes(kv).view(nb_total, PAGE * ROWB)
    flat[:, PAGE * ROWB :] = (
        scale.view(nb_total, PAGE).view(torch.uint8).reshape(nb_total, PAGE * 4)
    )
    kv_cache = flat.view(nb_total, PAGE, 1, ROWB + 4)
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
        "q_fp8": _to_fp8_bytes(q).view(batch, 1, 64, ROWB),
        "kv_cache": kv_cache,
        "weights": weights,
        "context_lens": lens,
        "block_table": block_table,
    }


def _reference(inp, k_top):
    kvf = inp["kv_cache"].reshape(inp["kv_cache"].shape[0], -1)
    k = _from_fp8_bytes(kvf[:, : PAGE * ROWB].reshape(-1, ROWB)).reshape(-1, PAGE, 128)
    scale = kvf[:, PAGE * ROWB :].contiguous().view(torch.float32).reshape(-1, PAGE)
    batch = inp["q_fp8"].shape[0]
    q = _from_fp8_bytes(inp["q_fp8"][:, 0].reshape(batch * 64, ROWB)).view(batch, 64, 128)
    vals = []
    for i in range(batch):
        length = int(inp["context_lens"][i])
        nb = (length + PAGE - 1) // PAGE
        pages = inp["block_table"][i, :nb].long()
        kx = k[pages].reshape(nb * PAGE, 128)
        s = torch.relu(q[i] @ kx.t())
        s = (s * inp["weights"][i].unsqueeze(1)).sum(dim=0) * scale[pages].reshape(-1)
        s[length:] = float("-inf")
        vals.append(torch.topk(s, k_top).values)
    return torch.stack(vals)


def _reference_scores(inp):
    kvf = inp["kv_cache"].reshape(inp["kv_cache"].shape[0], -1)
    k = _from_fp8_bytes(kvf[:, : PAGE * ROWB].reshape(-1, ROWB)).reshape(-1, PAGE, 128)
    scale = kvf[:, PAGE * ROWB :].contiguous().view(torch.float32).reshape(-1, PAGE)
    batch = inp["q_fp8"].shape[0]
    q = _from_fp8_bytes(inp["q_fp8"][:, 0].reshape(batch * 64, ROWB)).view(batch, 64, 128)
    out = []
    for i in range(batch):
        length = int(inp["context_lens"][i])
        nb = (length + PAGE - 1) // PAGE
        pages = inp["block_table"][i, :nb].long()
        kx = k[pages].reshape(nb * PAGE, 128)
        sc = torch.relu(q[i] @ kx.t())
        sc = (sc * inp["weights"][i].unsqueeze(1)).sum(dim=0) * scale[pages].reshape(-1)
        sc[length:] = float("-inf")
        out.append(sc)
    return out


def _check_fp32_boundary(inp, indices, k_top, tie_cap=2048):
    scores = _reference_scores(inp)
    for i in range(indices.shape[0]):
        sc = scores[i]
        ref = torch.topk(sc, k_top).indices
        kth = sc[ref[-1]]
        got = indices[i].long()
        assert got.unique().numel() == k_top
        if int((sc.half() == kth.half()).sum()) > tie_cap:
            assert int((sc[got].half() < kth.half()).sum()) == 0, (
                f"row {i}: token below the fp16 K-th value"
            )
            gh, _ = torch.sort(sc[got].half(), descending=True)
            rh, _ = torch.sort(sc[ref].half(), descending=True)
            assert bool((gh == rh).all()), f"row {i}: fp16 value multiset differs"
            continue
        tol = 1e-6 * kth.abs() + 1e-6
        assert int((sc[got] < kth - tol).sum()) == 0, (
            f"row {i}: selected token below the fp32 K-th value"
        )
        gs, _ = torch.sort(sc[got], descending=True)
        rs, _ = torch.sort(sc[ref], descending=True)
        assert bool(((gs - rs).abs() <= 1e-6 * rs.abs() + 1e-6).all()), (
            f"row {i}: fp32 score multiset differs"
        )


def _run(inp, k_top):
    batch = inp["q_fp8"].shape[0]
    device = inp["q_fp8"].device
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill_fp8.run(
        inp["q_fp8"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    return indices, values


def _check(inp, indices, values, k_top):
    ref_vals = _reference(inp, k_top)
    for i in range(indices.shape[0]):
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
@pytest.mark.parametrize("batch", [2, 16])
@pytest.mark.parametrize("n_comp", [8192, 16384])
@pytest.mark.parametrize("k_top", [1024, 2048])
@pytest.mark.parametrize("weight_mode", ["signed", "nonneg", "allneg"])
def test_fused_indexer_topk_nospill_fp8(batch, n_comp, k_top, weight_mode):
    inp = _build_inputs(batch, n_comp, k_top, weight_mode, seed=1234, device=torch.device("cuda"))
    indices, values = _run(inp, k_top)
    _check(inp, indices, values, k_top)


@skip_not_sm100
@pytest.mark.parametrize("batch", [1, 4, 16, 36, 70])
@pytest.mark.parametrize("n_comp", [16384, 65536])
def test_fused_indexer_topk_nospill_fp8_split(batch, n_comp, monkeypatch):
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", "1")
    inp = _build_inputs(batch, n_comp, 1024, "signed", seed=31, device=torch.device("cuda"))
    indices, values = _run(inp, 1024)
    _check(inp, indices, values, 1024)


@skip_not_sm100
def test_fused_indexer_topk_nospill_fp8_filtered(monkeypatch):
    # long rows: the dense prefix ends and the safe-line filter carries the rest
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_NDENSE", "8")
    inp = _build_inputs(4, 65536, 1024, "signed", seed=77, device=torch.device("cuda"))
    indices, values = _run(inp, 1024)
    _check(inp, indices, values, 1024)


@skip_not_sm100
def test_fused_indexer_topk_nospill_fp8_cuda_graph():
    device = torch.device("cuda")
    batch, k_top = 4, 1024
    inp = _build_inputs(batch, 16384, k_top, "signed", seed=93, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)

    def call():
        fused_indexer_topk_nospill_fp8.run(
            inp["q_fp8"],
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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        indices.fill_(-3)
        values.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(inp, indices, values, k_top)


@skip_not_sm100
@pytest.mark.parametrize("pattern", ["random", "monotone", "allequal", "giantbin"])
@pytest.mark.parametrize("batch", [2, 16, 80])
def test_fused_indexer_topk_nospill_fp8_patterns(pattern, batch, monkeypatch):
    # forced safe-line filter (8-tile dense prefix) under adversarial score shapes; batch 80
    # runs one CTA per row so the local and final K-th bins coincide
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_NDENSE", "8")
    weight_mode = "nonneg" if pattern != "random" else "signed"
    inp = _build_inputs(
        batch, 16384, 2048, weight_mode, seed=99, device=torch.device("cuda"), pattern=pattern
    )
    indices, values = _run(inp, 2048)
    _check(inp, indices, values, 2048)
    _check_fp32_boundary(inp, indices, 2048)


@skip_not_sm100
@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize("split", ["auto", "1"])
def test_fused_indexer_topk_nospill_fp8_mixed_lengths(batch, split, monkeypatch):
    monkeypatch.setenv("TRTLLM_FUSED_TOPK_GMEM_SPLIT", split)
    device = torch.device("cuda")
    inp = _build_inputs(batch, 65536, 1024, "signed", seed=2718, device=device)
    inp["context_lens"] = torch.tensor(
        [max(1024, 65536 >> (i % 4)) - 37 * i for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    indices, values = _run(inp, 1024)
    _check(inp, indices, values, 1024)
