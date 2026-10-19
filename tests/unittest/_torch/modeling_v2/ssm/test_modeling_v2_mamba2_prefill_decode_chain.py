# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: a Mamba2 layer's prefill state carried through decode.

The two SSM entries meet at the pool: `mamba_chunk_scan_combined` writes each
row's final state into its slot, `selective_state_update` reads it back and
advances it one token per step. Each entry is certified against its own
reference; this checks the hand-off -- layout, slot, dtype, the gather /
scatter glue of the prefill entry's `note` -- by running N prefill tokens and
then K decode steps, and holding every decode output and the final pool rows
to the fp32 recurrence over all N + K tokens (the prefill entry's `reference`
on the concatenated rows), for an fp16 and an fp32 pool.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    CHUNK_SIZE,
    DEP4_NGROUPS,
    DEP4_NHEADS,
    DSTATE,
    HEADDIM,
    NGROUPS,
    NHEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    mamba_chunk_scan_combined as prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.selective_state_update import (  # noqa: E501
    selective_state_update as decode,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

assert torch.cuda.is_available(), "the Mamba2 chain requires a CUDA device"

DEV = "cuda"
U = 2.0**-8  # one bf16 unit, the `_validating.ULP` convention
DT_LIMIT = (0.0, float("inf"))
#: Prefill rows: off the chunk grid, one longer than several chunks, carried and fresh.
LENGTHS = (300, 1000, 37, 128)
INITIAL = (True, False, True, False)


def _bands(pool_dtype: torch.dtype, steps: int) -> dict:
    """(element, relative RMS) bands in U, against the elementwise magnitude.

    fp32 pool: the prefill state enters decode at most 1 U of its magnitude off
    (the prefill entry's state band, measured 1.02 at most) and decays; a
    decode step is fp32 throughout but for the kernel's approximate softplus
    and exp (about 1e-4 of the state per step, forgotten at the head's own
    decay rate); each y is then rounded to bf16 by both sides. y: 4 U and
    1 U RMS (measured at most 1.86 and 0.33); the final state 2 U and 0.5 U
    (measured 0.65 and 0.10).

    fp16 pool: every write rounds the state to nearest fp16 -- once after the
    prefill, once per decode step -- each at most 2^-11 = U/8 of the state's
    magnitude, and nothing averages them: a slow-decay head whose per-step
    increment is below half an fp16 ulp does not move at all. So the bound
    grows linearly, (K + 1) U/8 over the fp32 one on the element. The RMS is
    gated on the same slope at an eighth of it, (K + 1) U/64, which measured
    growth stays under by 2x or more: y 0.51 U at K = 64 and 0.91 at 256, the
    final state 0.60 and 2.11 (and 3.8 at K = 512 -- the drift an fp16 pool
    costs long generations; the fp32 pool stays flat, 0.05). States below
    fp16's normal range (2^-14) carry an absolute error of up to half a
    subnormal spacing per write, allowed as (K + 1) * 2^-25.
    """
    if pool_dtype == torch.float32:
        return dict(y=(4.0, 1.0), state=(2.0, 0.5), floor=0.0)
    slope = (steps + 1) / 8
    return dict(
        y=(4.0 + slope, 1.0 + slope / 8),
        state=(2.0 + slope, 0.5 + slope / 8),
        floor=(steps + 1) * 2.0**-25,
    )


def _gate(got, want, mag, band, floor, what):
    diff = (got.float() - want.float()).abs()
    allowed = band[0] * U * mag.float() + floor
    worst = (diff / allowed.clamp_min(1e-30)).max().item()
    assert bool((diff <= allowed).all()), f"{what}: element error {worst:.3f} of the band"
    rel_rms = (diff.pow(2).mean().sqrt() / want.float().pow(2).mean().sqrt()).item() / U
    assert rel_rms <= band[1], f"{what}: relative RMS {rel_rms:.3f} U exceeds {band[1]:.3f} U"
    return worst, rel_rms


def _metadata(lengths, initial) -> Mamba2Metadata:
    n = len(lengths)
    mm = Mamba2Metadata(max_batch_size=n, chunk_size=CHUNK_SIZE)
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    mm.prepare(
        SimpleNamespace(
            seq_lens=seq_lens,
            num_contexts=n,
            seq_lens_cuda=seq_lens.to(DEV),
            num_ctx_tokens=sum(lengths),
            kv_cache_manager=None,
            request_ids=None,
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=[17 if carried else 0 for carried in initial]
            ),
        )
    )
    return mm


#: (nheads, ngroups): tep4's per-rank slice and dep4's whole mixer.
GEOMETRIES = {"tep4": (NHEADS, NGROUPS), "dep4": (DEP4_NHEADS, DEP4_NGROUPS)}


@pytest.mark.parametrize("steps", [64, 256])
@pytest.mark.parametrize("pool_dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_prefill_then_decode_follows_the_fp32_recurrence(geometry, pool_dtype, steps) -> None:
    gen = torch.Generator(device=DEV).manual_seed(41 + steps)
    n = len(LENGTHS)
    nheads, ngroups = GEOMETRIES[geometry]

    def uniform(lo, hi):
        return torch.empty(nheads, device=DEV).uniform_(lo, hi, generator=gen)

    A = -torch.exp(uniform(-3.2, 6.1))
    dt_bias = uniform(-8.1, -1.3).to(torch.bfloat16)
    D = uniform(-1.13, 3.94).to(torch.bfloat16)

    # Every row's tokens, prefill then decode, packed row after row: the
    # recurrence's view. The kernels see the prefill part packed by row and
    # each decode step as one token per row.
    xw, bw = nheads * HEADDIM, ngroups * DSTATE
    width = xw + 2 * bw
    rows = [length + steps for length in LENGTHS]
    total = sum(rows)
    conv = F.silu(torch.randn(total, width, device=DEV, generator=gen) * 2).to(torch.bfloat16)
    dt_all = torch.randn(total, nheads, device=DEV, generator=gen).to(torch.bfloat16)
    starts = [sum(rows[:s]) for s in range(n)]
    prefill_rows = torch.cat(
        [torch.arange(st, st + length) for st, length in zip(starts, LENGTHS)]
    ).to(DEV)

    def step_rows(k):
        return torch.tensor([st + length + k for st, length in zip(starts, LENGTHS)], device=DEV)

    # The pool: the layer's sub-page of a two-layer V2-style slot.
    base = torch.empty(n + 5, 2, nheads, HEADDIM, DSTATE, device=DEV, dtype=pool_dtype)
    base.fill_(1.5)
    pool = base[:, 1]
    pool.copy_(torch.randn(pool.shape, device=DEV, generator=gen) * 3)
    slots = torch.randperm(n + 5, generator=torch.Generator().manual_seed(steps))[:n]
    slots = slots.to(torch.int32).to(DEV)
    before = base.clone()

    # Prefill, as the target does it.
    mm = _metadata(LENGTHS, INITIAL)
    init = pool[slots].to(torch.float32)
    init.masked_fill_(~mm.has_initial_states[:n].view(n, 1, 1, 1), 0)
    tp = sum(LENGTHS)
    xbc = conv[prefill_rows]
    out = torch.empty(1, tp, nheads, HEADDIM, dtype=torch.bfloat16, device=DEV)
    final = prefill(
        xbc[:, :xw].view(1, tp, nheads, HEADDIM),
        dt_all[prefill_rows][None],
        A,
        xbc[:, xw : xw + bw].view(1, tp, ngroups, DSTATE),
        xbc[:, xw + bw :].view(1, tp, ngroups, DSTATE),
        CHUNK_SIZE,
        D,
        None,
        dt_bias,
        init,
        mm.seq_idx,
        mm.chunk_indices,
        mm.chunk_offsets,
        mm.cu_seqlens[: n + 1],
        True,
        DT_LIMIT,
        out,
        False,
        True,
        torch.float32,
    )
    pool[slots] = final.to(pool_dtype)

    # Decode, as the target does it.
    A_b = A[:, None, None].expand(nheads, HEADDIM, DSTATE)
    D_b = D[:, None].expand(nheads, HEADDIM)
    dt_bias_b = dt_bias[:, None].expand(nheads, HEADDIM)
    ys = []
    for k in range(steps):
        tok = conv[step_rows(k)]
        y = torch.empty(n, nheads, HEADDIM, dtype=torch.bfloat16, device=DEV)
        decode(
            pool,
            tok[:, :xw].view(n, nheads, HEADDIM),
            dt_all[step_rows(k)][:, :, None].expand(n, nheads, HEADDIM),
            A_b,
            tok[:, xw : xw + bw].view(n, ngroups, DSTATE),
            tok[:, xw + bw :].view(n, ngroups, DSTATE),
            D_b,
            None,
            dt_bias_b,
            True,
            slots,
            -1,
            y,
        )
        ys.append(y)

    # The fp32 recurrence over all N + K tokens of every row.
    cu = torch.tensor([0] + [sum(rows[: s + 1]) for s in range(n)], dtype=torch.int32, device=DEV)

    def recurrence(absolute: bool):
        f = torch.abs if absolute else (lambda t: t)
        return prefill.reference(
            f(conv[:, :xw]).view(1, total, nheads, HEADDIM),
            dt_all[None],
            A,
            f(conv[:, xw : xw + bw]).view(1, total, ngroups, DSTATE),
            f(conv[:, xw + bw :]).view(1, total, ngroups, DSTATE),
            CHUNK_SIZE,
            f(D),
            None,
            dt_bias,
            f(init),
            None,
            None,
            None,
            cu,
            True,
            DT_LIMIT,
            torch.empty((), dtype=torch.bfloat16 if not absolute else torch.float32),
            False,
            True,
            torch.float32,
        )

    ref_y, ref_final = recurrence(absolute=False)
    mag_y, mag_final = recurrence(absolute=True)
    bands = _bands(pool_dtype, steps)
    got_y = torch.stack(ys)  # [K, n, H, P]
    rows_y = torch.stack([step_rows(k) for k in range(steps)])  # [K, n]
    worst_y, rms_y = _gate(
        got_y, ref_y[0][rows_y], mag_y[0][rows_y], bands["y"], 0.0, "decode outputs"
    )
    worst_s, rms_s = _gate(
        pool[slots.long()],
        ref_final,
        mag_final,
        bands["state"],
        bands["floor"],
        "final pool rows",
    )
    print(
        f"pool {pool_dtype}, K={steps}: y element {worst_y:.3f} of band, RMS {rms_y:.3f} U; "
        f"state element {worst_s:.3f} of band, RMS {rms_s:.3f} U"
    )
    # Every other slot, and the other layer's sub-page of every slot, is untouched.
    expect = before.clone()
    expect[slots.long(), 1] = base[slots.long(), 1]
    assert torch.equal(base, expect)
