# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-aware work plan of the locality MoE engine variant (TP8, 1..8 decode tokens). Pure Python (no torch).

The stock engine (TP8 kernels/m3_moe) streams each distinct-expert group's FC1 stages in a slot-major item order
(item i -> unit ug = i // S1, slot = ug // U1, unit c = ug % U1) and gives every routed CTA one contiguous item range
per group plus a fixed range of the 192 FC2 32-row output blocks. Each expert's FC1 rows [p * I, (p + 1) * I) (units
c in [p * U1H, (p + 1) * U1H)) and FC2 output rows [p * H / 2, (p + 1) * H / 2) (blocks [p * NB2H, (p + 1) * NB2H)) are
the half placed in locality domain p (shard-major weights; a contiguous per-expert byte range of the stock tensor,
proved by rowrange_check_purepy_v1 at I = 384 / 768 / 1536).

This variant keeps the kernel's range format but numbers a group's FC1 stages DOMAIN-MAJOR with a fixed per-half
stride of 4 slots (padded): padded unit pu -> half h = pu // (4 U1H), slot = (pu % (4 U1H)) // U1H,
c = h U1H + pu % U1H, original group-local unit ug = slot U1 + c. A group of n_p slots occupies, in half h, the padded
items [h * 4 U1H S1, h * 4 U1H S1 + n_p U1H S1); domain h's routed CTAs (logical [h * PD, (h + 1) * PD)) split exactly
that range with the stock balancing, and FC2 block b is on logical CTA b (one block each when PD = NB2H). Logical CTA
indices come from the static %smid table (LocalityMap.logical_map), so a CTA's work is in its own die.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

HIDDEN = 6144
TOP_K = 4
R1 = 32  # FC1 unit rows
R2 = 32  # FC2 block rows
NB2 = HIDDEN // R2  # 192 FC2 output blocks
NB2H = NB2 // 2  # 96 per half
K1_TILES = HIDDEN // 256
C1 = 4
S1 = K1_TILES // C1  # 6 stages per FC1 unit


def geometry(inter: int) -> Dict[str, int]:
    """Per-TP constants (inter = I per rank: 384 TP8, 768 TP4)."""
    two_i = 2 * inter
    u1 = two_i // R1
    assert u1 % 2 == 0
    item_bytes = (R1 * (HIDDEN // 2 + HIDDEN // 16)) // S1
    row2_bytes = inter // 2 + inter // 16
    return {
        "I": inter,
        "U1": u1,  # FC1 units per expert (24 TP8, 48 TP4)
        "U1H": u1 // 2,  # per half
        "HALF_ITEMS": 4 * (u1 // 2) * S1,  # padded items per half (4 slots)
        "ITEM_BYTES": item_bytes,
        "BLK_BYTES": R2 * row2_bytes,  # one FC2 block, one slot
    }


def remap_unit(pu: int, g: Dict[str, int]) -> Tuple[int, int, int]:
    """Padded domain-major unit pu -> (half, slot, c); the kernel's _seg returns ug = slot * U1 + c."""
    u1h = g["U1H"]
    h = pu // (4 * u1h)
    r = pu - h * 4 * u1h
    slot = r // u1h
    c = h * u1h + (r - slot * u1h)
    return h, slot, c


def _fine_ranges(weights: Sequence[float], total: int, halfplan: bool = True) -> List[int]:
    """The stock _fine_ranges (TP8 m3_moe op._fine_ranges), unchanged."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
    half = S1 // 2
    if halfplan and n and max(weights) == min(weights) and total % half == 0 and total <= half * n:
        for c in range(n):
            acc += weights[c]
            e = total if c == n - 1 else min(total, max(s, half * int(acc / half + 0.5)))
            items[c] = e - s
            s = e
        assert s == total
        return items
    for c in range(n):
        acc += weights[c]
        e = total if c == n - 1 else min(total, max(s, int(acc + 0.5)))
        if s % S1 and e < total:
            e = max(e, min(total, (s // S1 + 1) * S1))
        items[c] = e - s
        s = e
    assert s == total
    return items


def _balance(total: int, fc2: Sequence[int], g: Dict[str, int], fine: bool = True) -> List[int]:
    """The stock per-group item balancing (TP8 m3_moe op.make_phase_plan) over len(fc2) CTAs, unchanged."""
    n = len(fc2)
    ib = g["ITEM_BYTES"]
    items = [0] * n
    if total >= S1 * n:

        def counts_for(limit):
            return [max(S1, (limit - f) // ib) for f in fc2]

        lo, hi = 0, max(fc2) + total * ib
        while lo < hi:
            mid = (lo + hi) // 2
            if sum(counts_for(mid)) >= total:
                hi = mid
            else:
                lo = mid + 1
        items = counts_for(lo)
        extra = sum(items) - total
        while extra > 0:
            c = min(
                (c for c in range(n) if items[c] > S1), key=lambda c: (-(fc2[c] + items[c] * ib), c)
            )
            items[c] -= 1
            extra -= 1
    elif fine:
        lo_f, hi_f = 0.0, float(max(fc2) + total * ib)
        for _ in range(100):
            mid = 0.5 * (lo_f + hi_f)
            if sum(max(0.0, (mid - f) / ib) for f in fc2) >= total:
                hi_f = mid
            else:
                lo_f = mid
        wts = [max(0.0, (hi_f - f) / ib) for f in fc2]
        sw = sum(wts)
        items = _fine_ranges([w * total / sw for w in wts], total)
    else:
        chosen = sorted(sorted(range(n), key=lambda c: (fc2[c], c))[: total // S1])
        for c in chosen:
            items[c] = S1
    assert sum(items) == total
    return items


def make_domain_phase_plan(
    m_max: int,
    inter: int = 384,
    per_domain: int = NB2H,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = True,
    fine: bool = True,
):
    """Domain-aware plan, same format as the stock make_phase_plan: plan[g][c][p] = i0 | i1 << 16 (padded
    domain-major item range of group p), plan[g][c][m_max] = b0 | nb << 16; returns (plan, nb_cap). Routed CTAs are
    2 * per_domain logical indices; logical c is in domain c // per_domain."""
    g_ = geometry(inter)
    ncta = 2 * per_domain
    assert NB2 % ncta == 0 or ncta == NB2, "one FC2 block per routed CTA (SH20 layout) expected"
    gcap = TOP_K * m_max
    nbs = [NB2 // ncta + (1 if c < NB2 % ncta else 0) for c in range(ncta)]
    b0s = [sum(nbs[:c]) for c in range(ncta)]
    for c in range(ncta):  # FC2 blocks stay in the CTA's own half
        assert all(((b0s[c] + k) // NB2H) == c // per_domain for k in range(nbs[c])), (
            c,
            b0s[c],
            nbs[c],
        )
    plan = [[[0] * (m_max + 1) for _ in range(ncta)] for _ in range(gcap + 1)]
    blk = g_["BLK_BYTES"]
    u1h = g_["U1H"]
    for gg in range(TOP_K, gcap + 1):
        ng = (gg + 3) // 4
        for c in range(ncta):
            plan[gg][c][m_max] = b0s[c] | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, gg - 4 * p)
            fc2 = [0] * ncta
            for c in range(ncta):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, gg - 4 * (p - lag)) * blk
                if p == ng - 1 and not (early0 and p == 0 and m_max >= 2):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, gg - 4 * q) * blk)
            for h in range(2):
                ctas = list(range(h * per_domain, (h + 1) * per_domain))
                total = n_p * u1h * S1
                items = _balance(total, [fc2[c] for c in ctas], g_, fine)
                i0 = h * g_["HALF_ITEMS"]
                for k, c in enumerate(ctas):
                    if items[k]:
                        plan[gg][c][p] = i0 | ((i0 + items[k]) << 16)
                    i0 += items[k]
                assert i0 == h * g_["HALF_ITEMS"] + total
    return plan, max(nbs)


def validate_plan(plan, m_max: int, inter: int = 384, per_domain: int = NB2H) -> List[str]:
    """Problems with a domain plan (empty = OK): every original unit of every group gets exactly its S1 stages;
    every CTA's range lies in its own domain's half; at most one CTA boundary strictly inside a unit; FC2 blocks
    cover 0..NB2-1 exactly once within their halves; group 0's ranges are identical for every G (early0)."""
    g_ = geometry(inter)
    errs = []
    ncta = 2 * per_domain
    gcap = TOP_K * m_max
    for gg in range(TOP_K, gcap + 1):
        ng = (gg + 3) // 4
        blocks = []
        for c in range(ncta):
            w = plan[gg][c][m_max]
            b0, nb = w & 0xFFFF, w >> 16
            blocks += list(range(b0, b0 + nb))
            if any(b // NB2H != c // per_domain for b in range(b0, b0 + nb)):
                errs.append(f"G{gg} CTA{c}: FC2 block outside its half")
        if sorted(blocks) != list(range(NB2)):
            errs.append(f"G{gg}: FC2 blocks not an exact cover")
        for p in range(ng):
            n_p = min(4, gg - 4 * p)
            cover: Dict[int, int] = {}
            bounds: Dict[int, int] = {}
            for c in range(ncta):
                w = plan[gg][c][p]
                i0, i1 = w & 0xFFFF, w >> 16
                if i1 <= i0:
                    continue
                h = c // per_domain
                lo = h * g_["HALF_ITEMS"]
                if not (lo <= i0 and i1 <= lo + n_p * g_["U1H"] * S1):
                    errs.append(
                        f"G{gg} p{p} CTA{c}: range [{i0},{i1}) outside half {h}'s valid items"
                    )
                for i in range(i0, i1):
                    hh, slot, cc = remap_unit(i // S1, g_)
                    if hh != h or slot >= n_p:
                        errs.append(f"G{gg} p{p} CTA{c}: item {i} -> half {hh} slot {slot}")
                    ug = slot * g_["U1"] + cc
                    cover[ug] = cover.get(ug, 0) + 1
                for edge in (i0, i1):
                    if edge % S1:
                        bounds[edge // S1] = bounds.get(edge // S1, 0) + 1
            want = {s * g_["U1"] + c_: S1 for s in range(n_p) for c_ in range(g_["U1"])}
            if cover != want:
                errs.append(
                    f"G{gg} p{p}: unit stage coverage differs ({len(cover)} vs {len(want)} units)"
                )
            if any(v > 2 for v in bounds.values()):
                errs.append(f"G{gg} p{p}: a unit has more than one CTA boundary inside it")
    g0 = {tuple(plan[gg][c][0] for c in range(ncta)) for gg in range(TOP_K, gcap + 1)}
    if m_max >= 2 and len(g0) != 1:
        errs.append("group 0's ranges differ across G (early0 broken)")
    return errs


def stock_phase_plan(
    m_max: int,
    inter: int = 384,
    num_ctas: int = 2 * NB2H,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = True,
    fine: bool = True,
):
    """The stock make_phase_plan (TP8 m3_moe op.py) in plain lists: contiguous slot-major items over num_ctas."""
    g_ = geometry(inter)
    gcap = TOP_K * m_max
    nbs = [NB2 // num_ctas + (1 if c < NB2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    plan = [[[0] * (m_max + 1) for _ in range(num_ctas)] for _ in range(gcap + 1)]
    blk = g_["BLK_BYTES"]
    for gg in range(TOP_K, gcap + 1):
        ng = (gg + 3) // 4
        for c in range(num_ctas):
            plan[gg][c][m_max] = b0s[c] | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, gg - 4 * p)
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, gg - 4 * (p - lag)) * blk
                if p == ng - 1 and not (early0 and p == 0 and m_max >= 2):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, gg - 4 * q) * blk)
            items = _balance(n_p * g_["U1"] * S1, fc2, g_, fine)
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[gg][c][p] = i0 | ((i0 + items[c]) << 16)
                i0 += items[c]
    return plan, max(nbs)


def split_structure(plan, m_max: int, inter: int = 384, domain: bool = True, ncta: int = 2 * NB2H):
    """{(G, group p, original unit ug): (stage of the CTA boundary inside the unit or None, head CTA, tail CTA)}.
    The kernel sums a split unit as head part (stages [0, s)) + tail part ([s, S1)) on two CTAs; an unsplit unit is
    one accumulation. Two plans give bit-identical FC1 results when every unit has the same split stage."""
    g_ = geometry(inter)
    out = {}
    for gg in range(TOP_K, TOP_K * m_max + 1):
        ng = (gg + 3) // 4
        for p in range(ng):
            parts: Dict[int, List[Tuple[int, int, int]]] = {}
            for c in range(ncta):
                w = plan[gg][c][p]
                i0, i1 = w & 0xFFFF, w >> 16
                for i in range(i0, i1):
                    u = i // S1
                    if domain:
                        _, slot, cc = remap_unit(u, g_)
                        ug = slot * g_["U1"] + cc
                    else:
                        ug = u
                    parts.setdefault(ug, []).append((i % S1, c))
            for ug, st in parts.items():
                st.sort()
                ctas = []
                for j, c in st:
                    if not ctas or ctas[-1][1] != c:
                        ctas.append((j, c))
                assert len(ctas) <= 2, (gg, p, ug, ctas)
                out[(gg, p, ug)] = (ctas[1][0] if len(ctas) == 2 else None, ctas[0][1], ctas[-1][1])
    return out


def logical_map(
    sm_domain: Sequence[int], per_domain: int, extra: int
) -> Tuple[List[int], List[int]]:
    """Mirror of LocalityMap.logical_map: sm_domain[s] = 0 / 1 / -1 (remainder), SMs in id order."""
    dom = [[s for s, d in enumerate(sm_domain) if d == k] for k in range(2)]
    remainder = [s for s, d in enumerate(sm_domain) if d not in (0, 1)]
    n = len(sm_domain)
    assert 2 * per_domain + extra == n and all(len(d) >= per_domain for d in dom)
    lmap = [-1] * n
    for k in range(2):
        for i, s in enumerate(dom[k][:per_domain]):
            lmap[s] = k * per_domain + i
    left = [dom[k][per_domain:] for k in range(2)]
    rest = [x for pair in zip(*left) for x in pair] + remainder
    ext = []
    for i, s in enumerate(rest):
        lmap[s] = 2 * per_domain + i
        ext.append(sm_domain[s])
    assert sorted(lmap) == list(range(n))
    return lmap, ext
