# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_compress entry: ratio-2 pair pooling with state kept across steps.

Each cell replays a schedule of steps (one chunk per request per step) through the op and through
the reference on separate state pools. Band: the op pools in fp32, the reference in float64 over
the stacked pair; a two-term softmax-weighted mean of unit-scale values differs by a few fp32
ulp, so the latents are gated at rtol/atol 1e-5. The state pools hold copies of the inputs, so
they must match exactly after every step.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compress import (  # noqa: E501
    csa2_compress_pairs,
)


def _schedule(reqs):
    """-> per step, per request (start, size)."""
    steps = max(len(r) for r in reqs)
    out, starts = [], [0] * len(reqs)
    for s in range(steps):
        step = []
        for i, r in enumerate(reqs):
            if s < len(r):
                step.append((i, starts[i], r[s]))
                starts[i] += r[s]
        out.append(step)
    return out


def _step_inputs(step, kv, sc):
    pos, rid, cs, ce, k, g = [], [], [], [], [], []
    for r, a, c in step:
        pos += list(range(a, a + c))
        rid += [r] * c
        cs += [a] * c
        ce += [a + c] * c
        k.append(kv[r][a : a + c])
        g.append(sc[r][a : a + c])
    t = lambda v, dt: torch.tensor(v, dtype=dt, device=DEV)  # noqa: E731
    return (
        torch.cat(k),
        torch.cat(g),
        t(pos, torch.int32),
        t(rid, torch.int64),
        t(cs, torch.int32),
        t(ce, torch.int32),
    )


def _state(nreq, d):
    pages = 3 * nreq + 1
    pools = [torch.zeros(pages, 128, d, device=DEV) for _ in range(2)]
    table = torch.full((nreq, 3), -1, dtype=torch.int32, device=DEV)
    ids = torch.arange(pages - 1, 0, -1, dtype=torch.int32, device=DEV)  # shuffled, page 0 unused
    for r in range(nreq):
        table[r] = ids[3 * r : 3 * r + 3]
    return pools, table


@pytest.mark.parametrize(
    "cell", csa2_compress_pairs.CELLS, ids=[c.why[:44] for c in csa2_compress_pairs.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_compress_pairs)
    d, reqs = cell.spec["d"], cell.spec["reqs"]
    gen = gen_for(cell)
    kv = [torch.randn(sum(r), d, generator=gen, device=DEV) for r in reqs]
    sc = [torch.randn(sum(r), d, generator=gen, device=DEV) for r in reqs]
    (pk, ps), table = _state(len(reqs), d)
    (rk, rs) = (pk.clone(), ps.clone())
    for step in _schedule(reqs):
        k, g, pos, rid, cs, ce = _step_inputs(step, kv, sc)
        got = csa2_compress_pairs(k, g, pos, rid, cs, ce, pk, ps, table, table)
        ref = csa2_compress_pairs.reference(k, g, pos, rid, cs, ce, rk, rs, table, table)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
        assert torch.equal(pk, rk) and torch.equal(ps, rs)


def _ref_groups(kv, score):
    """The checkpoint Compressor on a whole sequence at once (start_pos == 0 branch)."""
    n = kv.shape[0] // 2 * 2
    return (kv[:n].unflatten(0, (-1, 2)) * score[:n].unflatten(0, (-1, 2)).softmax(dim=1)).sum(
        dim=1
    )


@pytest.mark.parametrize("chunks", [[11], [1] * 11, [3, 1, 4, 1, 2], [5, 5, 1], [2, 2, 2, 2, 2, 1]])
def test_chunked_equals_whole_sequence(chunks):
    torch.manual_seed(0)
    total, d = sum(chunks), 16
    kv, score = torch.randn(total, d, device=DEV), torch.randn(total, d, device=DEV)
    ref = _ref_groups(kv, score)
    (pk, ps), table = _state(1, d)
    s = 0
    with validating(csa2_compress_pairs):
        for c in chunks:
            _, _, pos, rid, cs, ce = _step_inputs([(0, s, c)], [kv], [score])
            lat = csa2_compress_pairs(
                kv[s : s + c], score[s : s + c], pos, rid, cs, ce, pk, ps, table, table
            )
            for t in range(c):
                p = s + t
                if p % 2 == 1:  # this token completes pair p // 2
                    assert torch.allclose(lat[t], ref[p // 2], atol=1e-5), (chunks, p)
                else:  # an even token whose partner is in the step gets the completer's latent
                    if t + 1 < c:
                        assert torch.allclose(lat[t], lat[t + 1], atol=1e-6)
            s += c


def test_rejections():
    (pk, ps), table = _state(1, 4)
    kv = torch.zeros(2, 4, device=DEV)
    pos = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    rid = torch.zeros(2, dtype=torch.long, device=DEV)
    cs, ce = (
        torch.zeros(2, dtype=torch.int32, device=DEV),
        torch.full((2,), 2, dtype=torch.int32, device=DEV),
    )
    args = (kv, kv, pos, rid, cs, ce, pk, ps, table, table)
    csa2_compress_pairs.is_valid(*args)
    with pytest.raises(AssertionError, match="fp32"):
        csa2_compress_pairs.is_valid(kv.bfloat16(), kv, *args[2:])
    with pytest.raises(
        AssertionError, match="adjacent"
    ):  # a token of the request missing in between
        csa2_compress_pairs.is_valid(
            kv,
            kv,
            torch.tensor([0, 2], dtype=torch.int32, device=DEV),
            rid,
            cs,
            ce + 1,
            pk,
            ps,
            table,
            table,
        )
    with pytest.raises(AssertionError, match="cs, ce"):
        csa2_compress_pairs.is_valid(kv, kv, pos, rid, cs, ce - 1, pk, ps, table, table)
    # interleaved request rows: no two neighbours share a request, yet each row's partner is another request's row
    kv4, z4 = torch.zeros(4, 4, device=DEV), torch.zeros(4, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="one run"):
        csa2_compress_pairs.is_valid(
            kv4,
            kv4,
            torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEV),
            torch.tensor([0, 1, 0, 1], device=DEV),
            z4,
            z4 + 2,
            pk,
            ps,
            table,
            table,
        )


def _rollback_schedule():
    """(start, rows, accepted rows) of a speculative request: a 5-token prefill, then verify steps of 4 rows (1 + 3
    drafts) of which `accepted` are kept; the next step starts right after them, so it often starts at an odd position
    inside the previous step, whose rows past the accepted ones were rejected drafts."""
    steps, start = [(0, 5, 5)], 5
    for a in (1, 2, 1, 4, 3, 1, 2, 2, 1, 3):
        steps.append((start, 4, a))
        start += a
    return steps


def test_speculative_rollback_reads_the_accepted_partner():
    """A step that restarts at an odd position pools it with the accepted token before it, written by the previous
    step (not with the page's old content or a rejected draft): every completed pair of accepted tokens equals the
    whole-sequence pooling of the true values."""
    torch.manual_seed(1)
    d = 16
    steps = _rollback_schedule()
    total = steps[-1][0] + 4
    kv, score = torch.randn(total, d, device=DEV), torch.randn(total, d, device=DEV)
    ref = _ref_groups(kv, score)
    (pk, ps), table = _state(1, d)
    pk.normal_(), ps.normal_()  # a reused page: whatever an earlier request left there
    checked = 0
    for start, rows, accepted in steps:
        k, g = kv[start : start + rows].clone(), score[start : start + rows].clone()
        k[accepted:], g[accepted:] = (
            torch.randn_like(k[accepted:]),
            torch.randn_like(g[accepted:]),
        )  # rejected drafts
        _, _, pos, rid, cs, ce = _step_inputs([(0, start, rows)], [kv], [score])
        lat = csa2_compress_pairs(k, g, pos, rid, cs, ce, pk, ps, table, table)
        for t in range(accepted):
            p = start + t
            if p % 2 == 1:
                torch.testing.assert_close(lat[t], ref[p // 2], rtol=1e-5, atol=1e-5)
                checked += 1
    assert checked >= 8
