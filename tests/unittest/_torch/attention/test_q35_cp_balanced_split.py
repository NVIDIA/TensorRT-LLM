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
"""Balanced split of the Qwen3.5 CP-lite fused partial decode attention: the plan (host mirror invariants) and the
kernel output vs the FP32 reference."""

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
    for batch in (1, 2, 5, 16, 32):
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
    ],
)
def test_balanced_kernel_vs_reference(q_len: int, lens: list) -> None:
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
        F.q35_cp_fused_partial(q_all, kv, tables, seq, q_len, rank, CP, 1.0 / 16, 1.0, o, lse)
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
