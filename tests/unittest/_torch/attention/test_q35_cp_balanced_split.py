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
"""Balanced split of the Qwen3.5 CP-lite fused partial decode attention: the plans (host mirror invariants) and the
kernel output vs the FP32 reference. Batches above 32 take the wide-plan kernel variant and above 64 the two-slot
kernel (a CTA may compute two requests); the quad plan (the one-slot plan for 65..128 requests) runs through
plan="quad".
"""

import random

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import q35_cp_fused as F

CP, HQ, D, PAGE = 4, 4, 256, 64


def _tiles(L: int, rank: int) -> int:
    n_pages = (L + PAGE - 1) // PAGE
    return ((n_pages - rank + CP - 1) // CP + 1) // 2


def _feasible(t: list, T: int, n_ctas: int) -> bool:
    return sum(max(-(-x // T), 1) for x in t) <= n_ctas


@pytest.mark.parametrize("n_ctas", [152, 33])
@pytest.mark.parametrize("seed", range(3))
def test_balanced_plan_invariants(n_ctas: int, seed: int) -> None:
    rnd = random.Random(seed)
    for batch in (1, 2, 5, 16, 32, 33, 48, 64, 65, 96, 128):
        if batch >= n_ctas:
            continue
        lens = [
            rnd.choice([rnd.randrange(1, 300), rnd.randrange(1, 260_000)]) for _ in range(batch)
        ]
        for rank in range(CP):
            T, c, ctas = F.balanced_split_plan(lens, rank, CP, n_ctas)
            t = [_tiles(L, rank) for L in lens]
            assert len(ctas) == n_ctas and sum(c) == n_ctas and min(c) >= 1
            # T is the smallest feasible largest block; every block is within it
            assert _feasible(t, T, n_ctas) and (T == 1 or not _feasible(t, T - 1, n_ctas))
            assert max(n for _, _, _, n in ctas) <= max(T, 0)
            # per request: consecutive CTAs, blocks tile [0, t_r) in order, the first min(c_r, t_r) CTAs have work
            i = 0
            for r in range(batch):
                nxt = 0
                for j in range(c[r]):
                    rr, jj, k_lo, n_t = ctas[i]
                    assert (rr, jj) == (r, j) and k_lo == nxt
                    assert (n_t > 0) == (j < min(c[r], t[r]))
                    nxt += n_t
                    i += 1
                assert nxt == t[r]


def _check_two_slot_plan(lens: list, rank: int, n_ctas: int):
    """Invariants of the host mirror of the two-slot plan; returns (T, the largest tile count of a CTA)."""
    t = [_tiles(L, rank) for L in lens]
    T, slots = F.balanced_split_plan_two(lens, rank, CP, n_ctas)
    # T is the smallest tile count with sum max(t_r, T) <= n_ctas * T
    assert sum(max(x, T) for x in t) <= n_ctas * T and (
        T == 1 or sum(max(x, T - 1) for x in t) > n_ctas * (T - 1)
    )
    seen = {r: [] for r in range(len(lens))}
    max_cta = 0
    for i, (s0, s1) in enumerate(slots):
        if s0 is not None and s1 is not None:
            assert s0[0] < s1[0]
        load = sum(s[5] for s in (s0, s1) if s is not None)
        assert load <= T, (i, load, T)
        max_cta = max(max_cta, load)
        for s in (s0, s1):
            if s is not None:
                seen[s[0]].append((i, s))
    first = 0
    for r in range(len(lens)):
        cs = seen[r]
        c = cs[0][1][2]
        # consecutive CTAs, workspace slots in request order, blocks tile [0, t_r) in order
        assert len(cs) == c and [i for i, _ in cs] == list(range(cs[0][0], cs[0][0] + c))
        nxt = 0
        for j, (_, (rr, jj, cc, s0, k_lo, n_t, nv)) in enumerate(cs):
            assert (rr, jj, cc, s0, k_lo) == (r, j, c, first, nxt)
            assert nv == (c if t[r] else 0) and (n_t > 0 or t[r] == 0)
            nxt += n_t
        assert nxt == t[r] and (t[r] > 0 or c == 1)
        first += c
    assert first <= 2 * n_ctas
    return T, max_cta


@pytest.mark.parametrize("n_ctas", [152, 144, 133])
@pytest.mark.parametrize("seed", range(3))
def test_two_slot_plan_invariants(n_ctas: int, seed: int) -> None:
    rnd = random.Random(seed)
    for batch in (1, 2, 5, 16, 33, 64, 65, 76, 96, 101, 128):
        dists = (
            lambda: rnd.choice([rnd.randrange(1, 300), rnd.randrange(1, 260_000)]),
            lambda: rnd.choice([1, 7, 64, 65, 200, 40_000]),
            lambda: 32768,
        )
        for dist in dists:
            lens = [dist() for _ in range(batch)]
            for rank in range(CP):
                _check_two_slot_plan(lens, rank, n_ctas)


def test_two_slot_plan_uniform_long() -> None:
    # past ~76 uniform requests the one-slot plans hold a whole request on one CTA; the two-slot plan stays at
    # ceil(total / CTAs) (152 CTAs, rank 0)
    for batch, L, T_two, T_one in (
        (96, 65536, 81, 128),
        (128, 32768, 54, 64),
        (96, 131072, 162, 256),
    ):
        lens = [L] * batch
        T, max_cta = _check_two_slot_plan(lens, 0, 152)
        assert T == T_two and max_cta == T_two
        assert max(n for _, _, _, n in F.balanced_split_plan(lens, 0, CP, 152)[2]) == T_one


def _gpu_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._utils import is_sm_100f

    return IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()


def _make(lens, q_len, seed):
    b = len(lens)
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_pages = max((max(lens) + PAGE - 1) // PAGE, CP)  # the reference needs cp_rank < pages
    nblocks = b * 2 * max_pages + 8
    kv = (torch.randn(nblocks, 1, PAGE, D, device="cuda", generator=g) * 0.6).to(
        torch.float8_e4m3fn
    )
    tables = torch.randperm(nblocks, device="cuda", generator=g)[: b * 2 * max_pages].to(
        torch.int32
    )
    tables = tables.view(b, 2, max_pages).contiguous()
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q_all = (
        torch.randn(b * q_len, CP, HQ, D, device="cuda", generator=g)
        .to(torch.float8_e4m3fn)
        .permute(1, 0, 2, 3)
    )
    return kv, tables, seq, q_all


@pytest.fixture
def max_batch_128(monkeypatch):
    """Admit up to 128 requests (the wide-plan and two-slot kernels); the device state is rebuilt for it."""
    monkeypatch.setattr(F, "MAX_BATCH", 128)
    monkeypatch.setattr(F, "MAX_BATCH_Q1", 128)
    monkeypatch.setattr(F, "_STATES", {})


@pytest.mark.skipif(not _gpu_ok(), reason="the CuTe DSL kernel needs an SM100-family GPU")
@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize(
    "lens",
    [
        [7, 9000, 33000, 64, 130, 21000, 1, 4000],
        [
            3000,
            120000,
            250000,
            64000,
            90000,
            2000,
            180000,
            40000,
            70000,
            110000,
            5000,
            60000,
            150000,
            30000,
            100000,
            8000,
        ],
        # wide plan: the longest request in the upper half (it takes the spare CTAs), tiny and empty-prefix requests
        [1 + (i * 7919) % 50000 for i in range(40)] + [120000],
        [7, 64, 130] + [2000 + (i * 104729) % 90000 for i in range(61)],
        # two-slot kernel: the longest request last, 128 requests with tiny and empty-prefix ones, uniform long
        # contexts (the two-slot plan's reason: no request held to one CTA)
        [1 + (i * 7919) % 50000 for i in range(100)] + [120000],
        [7, 64, 130] + [2000 + (i * 104729) % 90000 for i in range(125)],
        [65536] * 76,
        [65536] * 128,
        [131072] * 96,
    ],
)
def test_balanced_kernel_vs_reference(q_len: int, lens: list, max_batch_128) -> None:
    _kernel_vs_reference(q_len, lens, None)


@pytest.mark.skipif(not _gpu_ok(), reason="the CuTe DSL kernel needs an SM100-family GPU")
@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize(
    "lens,plan",
    [
        ([1 + (i * 7919) % 50000 for i in range(100)] + [120000], "quad"),
        ([200000], "two"),
        ([7, 9000, 33000, 64, 130, 21000, 1, 4000], "two"),
        ([1 + (i * 7919) % 50000 for i in range(40)] + [120000], "two"),
    ],
)
def test_plan_override_vs_reference(q_len: int, lens: list, plan: str, max_batch_128) -> None:
    # the quad plan above 64 requests, the two-slot plan on the one-slot plans' batches
    _kernel_vs_reference(q_len, lens, plan)


def _kernel_vs_reference(q_len: int, lens: list, plan) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.q35_cp import q35_cp_fused_partial_ref

    lens = [max(L, q_len) for L in lens]
    b = len(lens)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if not F.q35_cp_fused_supported(q_len, CP, HQ, b, sms):
        pytest.skip("shape outside the fused CP partial kernel's limits")
    kv, tables, seq, q_all = _make(lens, q_len, b * 10 + q_len)
    for rank in range(CP):
        o = torch.full((CP, b * q_len, HQ, D), float("nan"), dtype=torch.bfloat16, device="cuda")
        lse = torch.full((CP, b * q_len, HQ), float("nan"), dtype=torch.float32, device="cuda")
        F.q35_cp_fused_partial(
            q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, o, lse, plan=plan
        )
        ref_o, ref_lse = torch.empty_like(o), torch.empty_like(lse)
        q35_cp_fused_partial_ref(
            q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, ref_o, ref_lse
        )
        torch.cuda.synchronize()
        # FP8 P / FP16 partials: max error ~5 % of the row max, mean ~1e-3
        of, rf = o.float(), ref_o.float()
        assert not torch.isnan(of).any() and not torch.isnan(lse).any()
        rel = ((of - rf).abs() / rf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)).amax().item()
        assert rel < 0.07, (rank, rel)
        assert (of - rf).abs().mean().item() < 2e-3
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin) and torch.equal(
            torch.isneginf(lse), torch.isneginf(ref_lse)
        )
        if fin.any():
            assert (lse[fin] - ref_lse[fin]).abs().amax().item() < 0.05


@pytest.mark.skipif(not _gpu_ok(), reason="the CuTe DSL kernel needs an SM100-family GPU")
@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize("b", [24, 33, 64])
def test_quad_plan_bitwise_below_65(q_len: int, b: int, max_batch_128) -> None:
    # the quad-plan variant computes the same plan as the lane / wide variants, so forcing it on their batches must
    # leave every output bit unchanged
    lens = [max(q_len, (i * 7919 + 13) % 70000) for i in range(b)]
    lens[b // 2] = 150000
    kv, tables, seq, q_all = _make(lens, q_len, b * 100 + q_len)
    for rank in range(CP):
        outs = []
        for plan in (None, "quad"):
            o = torch.zeros(CP, b * q_len, HQ, D, dtype=torch.bfloat16, device="cuda")
            lse = torch.zeros(CP, b * q_len, HQ, dtype=torch.float32, device="cuda")
            F.q35_cp_fused_partial(
                q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, o, lse, plan=plan
            )
            outs.append((o, lse))
        torch.cuda.synchronize()
        (o1, l1), (o2, l2) = outs
        assert torch.equal(o1.view(torch.int16), o2.view(torch.int16)), rank
        assert torch.equal(l1.view(torch.int32), l2.view(torch.int32)), rank
