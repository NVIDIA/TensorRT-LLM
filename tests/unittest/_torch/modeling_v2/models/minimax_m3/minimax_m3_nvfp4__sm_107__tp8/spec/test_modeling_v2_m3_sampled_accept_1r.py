# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SA-1R (ops/m3_sampled_accept.ONE_ROUND_K) on trtllm::minimax_m3_sm107tp8_sampled_accept, world 1, the TP8 shard.

The one-round variants (one_round_k 9: exact top-9 list; 82: pooled top-8 + bound) must return the two-round
kernel's (one_round_k 0) sampled tokens, accepted tokens, accepted counts, gather ids and extra-attempt count BIT FOR
BIT on the same Philox stream, over random AgentX-like rows (several temperatures / top-p, broad and heavy-tailed
heads) and over rows built so the decision sits within ulps of top_p * Z (the margin fallback) or a slice holds more
heavy tokens than its list (the incomplete-list fallback), eager and under CUDA-graph replay. The round-0 decided
counter must engage on the AgentX-like rows and stay below the row count on the boundary rows (fallbacks fire).
Ported from MoE's prototype test (moe/sa1r/6c753632ec, exact on 39,968 rows, 788162).
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

VS, D = 25008, 3
KX = (9, 82)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 tp8 sampled accept runs on SM 10.7", allow_module_level=True)


class Runner:
    """One variant with its own workspace / counters (call tag sequence) and outputs."""

    def __init__(self, k, rows):
        self.k, self.rows = k, rows
        words = int(torch.ops.trtllm.minimax_m3_sm107tp8_sampled_accept_ws_words(1, k))
        self.ws = torch.zeros(words, dtype=torch.int64, device="cuda")
        self.counters = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.tokens = torch.zeros(48, dtype=torch.int32, device="cuda")
        self.extra = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.decided = torch.zeros(1, dtype=torch.int32, device="cuda")
        nreq = rows // (D + 1)
        self.acc = torch.zeros((nreq, D + 1), dtype=torch.int32, device="cuda")
        self.nacc = torch.zeros(nreq, dtype=torch.int32, device="cuda")
        self.gids = torch.zeros(nreq, dtype=torch.int64, device="cuda")

    def __call__(self, lg, temps, tps, seeds, offs, draft, bidx):
        torch.ops.trtllm.minimax_m3_sm107tp8_sampled_accept(
            lg, 0, temps, tps, seeds, offs, self.ws, self.ws.data_ptr(), self.counters, self.tokens, 0, 1, draft,
            bidx, self.acc, self.nacc, self.gids, 0, None, None, self.extra, False, True, False, self.k,
            self.decided)  # fmt: skip

    def out(self):
        return (self.tokens[: self.rows].clone(), self.acc.clone(), self.nacc.clone(), self.gids.clone(),
                self.extra.clone())  # fmt: skip


def _params(rows, temp, top_p, seed0):
    temps = torch.full((rows,), temp, dtype=torch.float32, device="cuda")
    tps = (
        (top_p if torch.is_tensor(top_p) else torch.full((rows,), top_p))
        .float()
        .cuda()
        .contiguous()
    )
    seeds = torch.arange(rows, dtype=torch.int64, device="cuda") * 7919 + seed0
    offs = torch.zeros(rows, dtype=torch.int64, device="cuda")
    return temps, tps, seeds, offs


def _sets(n, rows, g, kind):
    out = []
    for _ in range(n):
        lg = torch.randn(rows, VS, device="cuda", generator=g) * (1.0 if kind == "flat" else 2.0)
        if kind == "lm":  # a few heavy tokens over a broad tail
            hot = torch.randint(0, VS, (rows, 8), device="cuda", generator=g)
            lg.scatter_(1, hot, 12.0 + 3.0 * torch.rand(rows, 8, device="cuda", generator=g))
        elif kind == "tail":  # 64 near-equal top tokens: lists overflow
            hot = torch.randint(0, VS, (rows, 64), device="cuda", generator=g)
            lg.scatter_(1, hot, 9.0 + 0.5 * torch.rand(rows, 64, device="cuda", generator=g))
        out.append(lg.to(torch.bfloat16).contiguous())
    return out


def _draft_from(lg, g):
    nreq = lg.shape[0] // (D + 1)
    am = lg.float().argmax(1).view(nreq, D + 1)[:, :D].int()
    rnd = torch.randint(0, VS, (nreq, D), device="cuda", generator=g, dtype=torch.int32)
    return torch.where(torch.rand(nreq, D, device="cuda", generator=g) < 0.7, am, rnd).contiguous()


def _compare(ref, got, what):
    for n, x, y in zip(
        ("tokens", "acc_tokens", "num_acc", "gather_ids", "extra_attempts"), ref, got
    ):
        assert torch.equal(x, y), f"{what}: {n} differs"


CASES = [(1.0, 0.95, "lm"), (0.7, 0.8, "lm"), (1.3, 0.5, "lm"), (1.0, 1.0, "lm"), (1.0, 0.95, "flat"),
         (1.0, 0.95, "tail"), (0.6, 0.99, "tail")]  # fmt: skip


def test_one_round_is_two_round_bitwise():
    g = torch.Generator(device="cuda").manual_seed(2)
    total = 0
    decided = {k: 0 for k in KX}
    for rows, calls in ((32, 20), (4, 60)):
        for ci, (t, p, kind) in enumerate(CASES):
            runners = {k: Runner(k, rows) for k in (0, *KX)}
            temps, tps, seeds, offs = _params(rows, t, p, 1000 + ci)
            sets = _sets(4, rows, g, kind)
            bidx = torch.arange(rows // (D + 1), dtype=torch.int32, device="cuda")
            for i in range(calls):
                lg = sets[i % 4]
                draft = _draft_from(lg, g)
                offs.fill_(i)
                for r in runners.values():
                    r(lg, temps, tps, seeds, offs, draft, bidx)
                for k in KX:
                    _compare(
                        runners[0].out(), runners[k].out(), f"K={k} rows={rows} case={ci} call={i}"
                    )
                total += rows
            if kind == "lm" and p < 1.0:
                for k in KX:
                    decided[k] += int(runners[k].decided.item())
            assert int(runners[0].decided.item()) == 0
    assert total >= 5000
    for k in KX:
        assert decided[k] > 0, (
            f"K={k}: no row decided in round 0 on the AgentX-like rows (SA-1R not engaged)"
        )


def _boundary_rows(rows, n_heavy_same_slice):
    """Heavy tokens heavier than a 50-token candidate group; top_p swept across ulps of S / Z (margin fallback), or
    n heavy tokens in one slice (more than its list: incomplete fallback)."""
    lg = torch.full((rows, VS), -40.0, device="cuda")
    sl = ((VS + 7) // 8 + 7) // 8 * 8
    heavy = list(range(5, 5 + n_heavy_same_slice)) if n_heavy_same_slice else [5, 3 * sl + 11]
    grp = [2 * sl + 3 + 37 * j for j in range(25)] + [5 * sl + 7 + 41 * j for j in range(25)]
    lg[:, heavy] = 5.0
    lg[:, grp] = 4.0
    lg = lg.to(torch.bfloat16).contiguous()
    w = torch.exp(lg[0].double() - lg[0].double().max())
    crit = float(w[heavy].sum()) / float(w.sum())
    ks = torch.arange(rows, dtype=torch.float64) - rows // 2
    return lg, (crit * (1.0 + ks * 2.0**-24)).float()


@pytest.mark.parametrize("graph", [False, True])
def test_one_round_boundary_and_incomplete(graph):
    rows = 32
    g = torch.Generator(device="cuda").manual_seed(3)
    seen = {k: [0, 0] for k in KX}  # decided rows, rows
    for n_same in (0, 12):
        lg, tps_rows = _boundary_rows(rows, n_same)
        for shift in range(-4, 4):
            tps = (tps_rows.double() * (1.0 + shift * 32 * 2.0**-24)).float()
            runners = {k: Runner(k, rows) for k in (0, *KX)}
            temps, tp_t, seeds, offs = _params(rows, 1.0, tps, 5000 + shift)
            bidx = torch.arange(rows // (D + 1), dtype=torch.int32, device="cuda")
            draft = _draft_from(lg, g)
            graphs = {}
            if graph:
                for r in runners.values():
                    r(lg, temps, tp_t, seeds, offs, draft, bidx)
                torch.cuda.synchronize()
                for k, r in runners.items():
                    graphs[k] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[k]):
                        r(lg, temps, tp_t, seeds, offs, draft, bidx)
                for r in runners.values():
                    r.decided.zero_()
            for i in range(12):
                offs.fill_(i)
                for k, r in runners.items():
                    if graph:
                        graphs[k].replay()
                    else:
                        r(lg, temps, tp_t, seeds, offs, draft, bidx)
                for k in KX:
                    _compare(runners[0].out(), runners[k].out(),
                             f"boundary n_same={n_same} shift={shift} call={i} K={k}")  # fmt: skip
            for k in KX:
                seen[k][0] += int(runners[k].decided.item())
                seen[k][1] += 12 * rows
    for k in KX:
        assert 0 < seen[k][0] < seen[k][1], (
            f"K={k}: decided {seen[k]}: the boundary sweep missed both sides"
        )
