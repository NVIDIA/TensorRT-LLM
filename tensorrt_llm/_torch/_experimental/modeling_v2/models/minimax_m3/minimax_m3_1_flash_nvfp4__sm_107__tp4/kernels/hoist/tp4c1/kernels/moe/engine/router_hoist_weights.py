# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router hoist weights and layouts: the router folded with the post norm, the route-line / counter layouts, and the
shared-expert pretile."""

from __future__ import annotations

import argparse
import hashlib
from types import SimpleNamespace

import torch

H = 6144
E = 128
TOP_K = 4
ROUTE_SCALE = 2.0
SUM_EPS = 1e-20
EPS = 1e-6
NPART = 2  # router partial parts per rank: 0 = a (folded o_proj), 1 = r (residual half)
EMPTY = (
    -2147483648
)  # 0x80000000: the v3 Lamport sentinel word (fp32 -0.0, canonicalised away by the pusher)
ROUTE_LINE_U64 = 16  # one 128-byte line per (replica, token); words 0..3 carry the top-4
RCTR_STRIDE = 32  # int32 words per row counter (own 128-byte line)


# ----------------------------------------------------------------------------- MXFP8 scale layout (128x4 swizzle)
def sf128x4_index(n: torch.Tensor, kb: torch.Tensor, kb_total: int) -> torch.Tensor:
    """Byte index of scale (row n, 32-column block kb) in the 128x4 swizzled layout (lb_dev.cuh mxfp8_8's formula)."""
    kt = (kb_total + 3) // 4
    return (n // 128) * kt * 512 + (kb // 4) * 512 + (n % 32) * 16 + ((n % 128) // 32) * 4 + kb % 4


def swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """[N, KB] uint8 -> flat swizzled uint8 (rows padded to 128, KB to a multiple of 4)."""
    n, kb = sf.shape
    npad, kbpad = (n + 127) // 128 * 128, (kb + 3) // 4 * 4
    out = torch.zeros(npad * kbpad, dtype=torch.uint8, device=sf.device)
    nn, kk = torch.meshgrid(
        torch.arange(n, device=sf.device), torch.arange(kb, device=sf.device), indexing="ij"
    )
    out[sf128x4_index(nn, kk, kb).reshape(-1)] = sf.reshape(-1)
    return out


def unswizzle_128x4(swz: torch.Tensor, n: int, kb: int) -> torch.Tensor:
    flat = swz.reshape(-1).view(torch.uint8)
    nn, kk = torch.meshgrid(
        torch.arange(n, device=flat.device), torch.arange(kb, device=flat.device), indexing="ij"
    )
    return flat[sf128x4_index(nn, kk, kb)].reshape(n, kb)


def check_swizzle_against(swz_fn, n: int = 256, k: int = 4096) -> bool:
    """In-container check that vLLM's swizzle_mxfp8_scale (weights.py _swz) is this 128x4 layout: three byte-plane
    probes of index + 1 (the swizzle is a permutation plus zero padding)."""
    kb = k // 32
    idx = torch.arange(n * kb, dtype=torch.int64).reshape(n, kb) + 1
    got = torch.zeros(n * kb, dtype=torch.int64)
    for plane in range(3):
        probe = ((idx >> (8 * plane)) & 0xFF).to(torch.uint8)
        s = swz_fn(probe, n, k).reshape(-1).cpu().to(torch.int64)
        u = unswizzle_128x4(s.to(torch.uint8), n, kb).reshape(-1).to(torch.int64)
        got |= u << (8 * plane)
    return bool(torch.equal(got, idx.reshape(-1)))


def e8m0_scale(codes_u8: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes_u8.to(torch.float64) - 127.0)


def mxfp8_dequant(w_e4m3: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    """[N, K] e4m3 x UE8M0 per 32 columns -> fp64 (exact)."""
    n, k = w_e4m3.shape
    return w_e4m3.to(torch.float64) * e8m0_scale(sf_u8[:, : k // 32]).repeat_interleave(32, dim=1)


def mxfp8_quant_rows(x: torch.Tensor):
    """Row-wise MXFP8 (mxfp8_8 semantics: code = e8m0 roundup of amax/448, x * 1/2^code -> e4m3 RN sat) for tests.
    Returns (e4m3 [T, K], codes uint8 [T, K/32])."""
    t, k = x.shape
    xb = x.float().reshape(t, k // 32, 32)
    amax = xb.abs().amax(-1)
    e = torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=2.0**-127)))
    code = torch.clamp(e + 127, 0, 254).to(torch.uint8)
    sc = torch.where(amax > 0, torch.exp2(-(code.float() - 127.0)), torch.zeros_like(amax))
    q = (xb * sc.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(t, k)
    return q, code


# ----------------------------------------------------------------------------- folding
def fold_router(router: torch.Tensor, post_norm: torch.Tensor) -> torch.Tensor:
    """W' = W_r diag(1 + g): (1 + g) in fp32 as norm8's add_f(1, w), then one fp32 product."""
    return router.float() * (1.0 + post_norm.float()).unsqueeze(0)


def fold_oproj(wp: torch.Tensor, o_deq: torch.Tensor) -> torch.Tensor:
    """W'' = W' deq(W_o_k), fp64 accumulate, one rounding to fp32. o_deq: [H, Ko]."""
    return (wp.double() @ o_deq.double()).float()


def r_cols(rank: int, world: int) -> tuple[int, int]:
    w = H // world
    return rank * w, (rank + 1) * w


# ----------------------------------------------------------------------------- per-CTA job table + packed blob
def job_list(ko: int, rank: int, world: int, r_split: int = 1):
    """(expert, part, c0, ncols) jobs of one rank: one a-job per expert over [0, Ko), r_split r-jobs per expert over
    its column half (c0 in residual coordinates)."""
    c0r, c1r = r_cols(rank, world)
    step = (c1r - c0r) // r_split
    jobs = [(e, 0, 0, ko) for e in range(E)]
    for s in range(r_split):
        jobs += [(e, 1 + s, c0r + s * step, step) for e in range(E)]
    return jobs


def assign_jobs(jobs, grid: int):
    """Deterministic LPT: largest job first onto the least-loaded CTA (ties -> lower CTA)."""
    load = [0] * grid
    per = [[] for _ in range(grid)]
    for j in sorted(jobs, key=lambda x: (-x[3], x[1], x[0])):
        c = min(range(grid), key=lambda i: (load[i], i))
        per[c].append(j)
        load[c] += j[3]
    return per, max(load)


def build_jobs(
    wp: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    world: int,
    grid: int,
    fmt: str = "fp32",
    r_split: int = 1,
):
    """Packed weights + table for the o_proj router role.

    blob: per CTA one 128-byte-aligned contiguous span (one cp.async.bulk pre-wait), jobs back to back. fmt fp32:
    ncols floats; bf16: ncols bf16; bf16x2: hi row then lo row (lo = bf16(w - hi)), 2 * ncols bf16.
    table int32 [grid, maxj, 4]: (expert | part << 8, c0, ncols, element offset into blob), -1 rows unused.
    span int32 [grid, 2]: (element offset, elements) of the CTA's blob span."""
    ko = w2.shape[1]
    per, mx = assign_jobs(job_list(ko, rank, world, r_split), grid)
    maxj = max(len(p) for p in per)
    el = (
        1 if fmt == "fp32" else 2
    )  # 2-byte elements need 64-element (128 B) alignment instead of 32
    align = 32 * el
    mult = 2 if fmt == "bf16x2" else 1
    table = torch.full((grid, maxj, 4), -1, dtype=torch.int32)
    span = torch.zeros(grid, 2, dtype=torch.int32)
    chunks, off = [], 0
    for c, jobs in enumerate(per):
        start = off
        for i, (e, part, c0, n) in enumerate(jobs):
            src = w2[e, c0 : c0 + n] if part == 0 else wp[e, c0 : c0 + n]
            if fmt == "fp32":
                v = src.float()
            elif fmt == "bf16":
                v = src.to(torch.bfloat16)
            elif fmt == "bf16x2":
                hi = src.to(torch.bfloat16)
                v = torch.cat([hi, (src.float() - hi.float()).to(torch.bfloat16)])
            else:
                raise ValueError(fmt)
            table[c, i] = torch.tensor([e | (part << 8), c0, n, off], dtype=torch.int32)
            chunks.append(v.cpu())
            off += n * mult
        pad = (-(off - start)) % align
        if pad:
            chunks.append(torch.zeros(pad, dtype=chunks[-1].dtype if chunks else torch.float32))
            off += pad
        span[c] = torch.tensor([start, off - start], dtype=torch.int32)
    dt = torch.float32 if fmt == "fp32" else torch.bfloat16
    blob = torch.cat([ch.to(dt) for ch in chunks]) if chunks else torch.zeros(0, dtype=dt)
    return SimpleNamespace(
        blob=blob,
        table=table,
        span=span,
        fmt=fmt,
        maxj=maxj,
        max_cols=mx,
        r_split=r_split,
        npart=1 + r_split,
    )


def blob_weights(jb, cta: int, slot: int) -> torch.Tensor:
    e_part, c0, n, off = (int(v) for v in jb.table[cta, slot])
    if jb.fmt == "bf16x2":
        return jb.blob[off : off + n].double() + jb.blob[off + n : off + 2 * n].double()
    return jb.blob[off : off + n].double()


def emulate_partials(
    jb, a_deq: torch.Tensor, r: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """Walk the table as the kernel does (fp64 dots, rounded once): P [T, npart, E] fp32 of this rank."""
    t = a_deq.shape[0]
    out = torch.zeros(t, jb.npart, E, dtype=torch.float64)
    for c in range(jb.table.shape[0]):
        for s in range(jb.maxj):
            e_part, c0, n, off = (int(v) for v in jb.table[c, s])
            if e_part < 0:
                continue
            e, part = e_part & 0xFF, e_part >> 8
            x = a_deq[:, c0 : c0 + n] if part == 0 else r[:, c0 : c0 + n]
            out[:, part, e] += x.double() @ blob_weights(jb, c, s)
    return out.float()


# ----------------------------------------------------------------------------- one rank's hoist weights
def build_rank(
    router,
    bias,
    post_norm,
    o_w_e4m3,
    o_sf_u8,
    rank: int,
    world: int,
    grid: int = 192,
    fmt: str = "fp32",
    r_split: int = 1,
):
    """o_w_e4m3 [H, Ko] e4m3, o_sf_u8 [H, Ko/32] raw (unswizzled) UE8M0 of this rank's o_proj slice."""
    wp = fold_router(router, post_norm)
    w2 = fold_oproj(wp, mxfp8_dequant(o_w_e4m3, o_sf_u8))
    jb = build_jobs(wp.cpu(), w2.cpu(), rank, world, grid, fmt, r_split)
    h = hashlib.sha256(jb.blob.view(torch.uint8).numpy().tobytes()).hexdigest()[:12]
    return SimpleNamespace(
        wp=wp,
        w2=w2,
        jobs=jb,
        bias=bias.float(),
        rank=rank,
        world=world,
        grid=grid,
        r_cols=r_cols(rank, world),
        sha=h,
    )


def build_from_blk(
    blk, rank: int, world: int, grid: int = 192, fmt: str = "fp32", r_split: int = 1, swz_fn=None
):
    """From an M3-mini blk: blk.aw.o (MXFP8, scales swizzled by weights._swz), blk.aw.post_norm, blk.mw.router/bias.
    swz_fn: weights._swz, to prove the 128x4 inverse before trusting it (recommended once per process)."""
    if swz_fn is not None and not check_swizzle_against(swz_fn):
        raise RuntimeError(
            "hoist: vLLM's MXFP8 scale swizzle is not the 128x4 layout unswizzle_128x4 inverts"
        )
    o = blk.aw.o
    sf = unswizzle_128x4(o.s.view(torch.uint8), o.n, o.k // 32)
    return build_rank(
        blk.mw.router, blk.mw.bias, blk.aw.post_norm, o.w, sf, rank, world, grid, fmt, r_split
    )


def canon_word(x: torch.Tensor) -> torch.Tensor:
    """fp32 bits as pushed: -0.0 (the sentinel 0x80000000) becomes +0.0."""
    b = x.float().view(torch.int32)
    return torch.where(b == EMPTY, torch.zeros_like(b), b)


def bf16_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF


def pack_route(tag: int, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """u64 (as int64) per (token, k): tag << 32 | id << 16 | bf16(w) bits."""
    assert 0 < tag < (1 << 31)
    return (int(tag) << 32) | (ids.to(torch.int64) << 16) | bf16_bits(w)


def unpack_route(words: torch.Tensor):
    tag = (words >> 32) & 0xFFFFFFFF
    ids = (words >> 16) & 0xFFFF
    wb = (words & 0xFFFF).to(torch.int32) << 16
    return tag, ids, wb.view(torch.float32)


# ----------------------------------------------------------------------------- references
def f32_sum_rank_order(p: torch.Tensor) -> torch.Tensor:
    """p [world, npart, T, E] fp32 -> ((p[0,0] + p[0,1]) + (p[1,0] + p[1,1])) ...: the post kernel's fixed order."""
    s = None
    for r in range(p.shape[0]):
        pr = p[r, 0]
        for j in range(1, p.shape[1]):
            pr = pr + p[r, j]
        s = pr if s is None else s + pr
    return s


def hoisted_logits(parts: list, rstd: torch.Tensor) -> torch.Tensor:
    """parts: per source rank [T, npart, E] fp32 (canonicalised words); rstd [T] fp32."""
    p = torch.stack([canon_word(x).view(torch.float32).transpose(0, 1) for x in parts])
    return f32_sum_rank_order(p) * rstd.float().unsqueeze(1)


def stock_boundary(p_list: list, r: torch.Tensor, post_norm: torch.Tensor, eps: float = EPS):
    """lbv3 / stock: h = bf16(bf16(sum_r p_r) + r); rstd = rsqrt(sum h^2 / H + eps); normed = bf16(h rstd (1 + g)).
    Reduction trees are approximated in fp64 (the kernel's sq order differs in the last bits)."""
    s = p_list[0].float()
    for p in p_list[1:]:
        s = s + p.float()
    h = (s.to(torch.bfloat16).float() + r.float()).to(torch.bfloat16)
    sq = (h.double() ** 2).sum(-1).float()
    rstd = torch.rsqrt(sq / H + eps)
    normed = ((h.float() * rstd.unsqueeze(1)) * (1.0 + post_norm.float())).to(torch.bfloat16)
    return h, rstd, normed


def stock_logits(normed: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    return (normed.double() @ router.double().t()).float()


def route_minimax2(logits: torch.Tensor, bias: torch.Tensor):
    """The engine's MiniMax2 routing, operation for operation (m3_moe_kernel.py routing warps): key = sigmoid + bias
    with sigmoid = fma(0.5, tanh(0.5 x), 0.5); top-4 by key, ties to the lower id; sk = key - bias; total =
    (sk0 + sk2) + (sk1 + sk3) (the 32-lane butterfly over 4 live lanes); w = bf16(sk * 2 / (total + 1e-20)).
    Returns ids [T, 4] int64 (rank order), w [T, 4] fp32 (bf16 values), keys [T, E]."""
    x = logits.float()
    sig = (0.5 * torch.tanh(0.5 * x) + 0.5).float()
    keys = sig + bias.float().unsqueeze(0)
    ids = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    sk = keys.gather(1, ids) - bias.float()[ids]
    total = (sk[:, 0] + sk[:, 2]) + (sk[:, 1] + sk[:, 3])
    w = ((sk * ROUTE_SCALE) / (total + SUM_EPS).unsqueeze(1)).to(torch.bfloat16).float()
    return ids, w, keys


def flip_report(ids_ref: torch.Tensor, ids_got: torch.Tensor, keys_ref: torch.Tensor) -> dict:
    """Set flips per token (4 - |intersection|) plus the reference's 4th-vs-5th key margin (how close a token was)."""
    a, b = ids_ref.sort(-1).values, ids_got.sort(-1).values
    inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1)
    flips = TOP_K - inter
    ks = keys_ref.sort(-1, descending=True).values
    margin = (ks[:, TOP_K - 1] - ks[:, TOP_K]).double()
    return {
        "tokens": int(ids_ref.shape[0]),
        "tokens_flipped": int((flips > 0).sum()),
        "slot_flips": int(flips.sum()),
        "margin_min": float(margin.min()),
        "margin_p01": float(margin.quantile(0.01)),
        "margin_flipped_max": float(margin[flips > 0].max()) if (flips > 0).any() else None,
    }


# ----------------------------------------------------------------------------- selftest (CPU)
def _rand_case(t: int, world: int, ko: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    router = torch.randn(E, H, generator=g) * 0.02
    bias = torch.randn(E, generator=g) * 0.05
    post_norm = (torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
    r = (torch.randn(t, H, generator=g) * 2.0).to(torch.bfloat16)
    ows, osfs, aqs, asfs = [], [], [], []
    for _ in range(world):
        wf = torch.randn(H, ko, generator=g) * 0.02
        wq, wsf = mxfp8_quant_rows(wf)
        ows.append(wq)
        osfs.append(wsf)
        a = torch.randn(t, ko, generator=g)
        aq, asf = mxfp8_quant_rows(a)
        aqs.append(aq)
        asfs.append(asf)
    return router, bias, post_norm, r, ows, osfs, aqs, asfs


def selftest(t: int = 16, world: int = 2, ko: int = 4096, grid: int = 192, seed: int = 0) -> dict:
    router, bias, post_norm, r, ows, osfs, aqs, asfs = _rand_case(t, world, ko, seed)
    rep = {}
    # 1. swizzle round trip
    sf = osfs[0]
    rep["swizzle_roundtrip"] = bool(torch.equal(unswizzle_128x4(swizzle_128x4(sf), *sf.shape), sf))
    rep["swizzle_probe_self"] = check_swizzle_against(lambda p, n, k: swizzle_128x4(p), 256, 4096)
    # 2. per-rank fold + packing; stock partials p_k = bf16(deq(W_o_k) deq(a_k))
    ranks, p_list, p_exact, parts = [], [], [], []
    for k in range(world):
        hw = build_rank(router, bias, post_norm, ows[k], osfs[k], k, world, grid)
        a_deq = mxfp8_dequant(aqs[k], asfs[k])
        p_exact.append(a_deq @ mxfp8_dequant(ows[k], osfs[k]).t())
        p_list.append(p_exact[-1].to(torch.bfloat16))
        pk = emulate_partials(hw.jobs, a_deq, r, k, world)
        c0, c1 = hw.r_cols
        direct_a = (a_deq @ hw.w2.double().t()).float()
        direct_r = (r[:, c0:c1].double() @ hw.wp[:, c0:c1].double().t()).float()
        rep[f"pack_a_r{k}_maxabs"] = float((pk[:, 0] - direct_a).abs().max())
        rep[f"pack_r_r{k}_maxabs"] = float((pk[:, 1] - direct_r).abs().max())
        cover = torch.zeros(hw.jobs.npart, E, dtype=torch.int64)
        for row in hw.jobs.table.reshape(-1, 4):
            if int(row[0]) >= 0:
                cover[int(row[0]) >> 8, int(row[0]) & 0xFF] += 1
        rep[f"cover_r{k}"] = bool((cover == 1).all())
        rep[f"max_cols_r{k}"] = hw.jobs.max_cols
        ranks.append(hw)
        parts.append(pk)
    # 3. stock vs hoisted logits and routing
    h, rstd, normed = stock_boundary(p_list, r, post_norm)
    lg_stock = stock_logits(normed, router)
    lg_hoist = hoisted_logits(parts, rstd)
    # ideal: no bf16 rounding anywhere (fp64 h, rstd, x); both arms are judged against it
    h64 = sum(p_exact) + r.double()
    rstd64 = torch.rsqrt((h64**2).sum(-1) / H + EPS)
    lg_ideal = (h64 * rstd64.unsqueeze(1)) @ ranks[0].wp.double().t()
    scale = lg_stock.abs().max().item()
    rep["logit_scale"] = scale
    rep["hoist_vs_stock_maxabs"] = float((lg_hoist - lg_stock).abs().max())
    rep["stock_vs_ideal_maxabs"] = float((lg_stock.double() - lg_ideal).abs().max())
    rep["hoist_vs_ideal_maxabs"] = float((lg_hoist.double() - lg_ideal).abs().max())
    ids_s, w_s, keys_s = route_minimax2(lg_stock, bias)
    ids_h, w_h, _ = route_minimax2(lg_hoist, bias)
    rep["flips_hoist_vs_stock"] = flip_report(ids_s, ids_h, keys_s)
    # 4. cross-rank identity: both ranks sum the same pushed words in rank order -> bitwise equal logits
    rep["cross_rank_bitwise"] = bool(
        torch.equal(
            hoisted_logits(parts, rstd).view(torch.int32),
            hoisted_logits([p.clone() for p in parts], rstd).view(torch.int32),
        )
    )
    # 5. route words
    words = pack_route(7, ids_h, w_h)
    tg, ii, ww = unpack_route(words)
    rep["route_pack_roundtrip"] = bool(
        (tg == 7).all() and torch.equal(ii, ids_h) and torch.equal(ww, w_h)
    )
    # 6. negative control: fold without (1 + g) must be caught by the same instrument
    bad = [
        emulate_partials(
            build_jobs(
                router.float(),
                (router.double() @ mxfp8_dequant(ows[k], osfs[k])).float(),
                k,
                world,
                grid,
            ),
            mxfp8_dequant(aqs[k], asfs[k]),
            r,
            k,
            world,
        )
        for k in range(world)
    ]
    lg_bad = hoisted_logits(bad, rstd)
    rep["negctl_maxabs"] = float((lg_bad - lg_stock).abs().max())
    rep["negctl_flips"] = flip_report(ids_s, route_minimax2(lg_bad, bias)[0], keys_s)
    tol = rep[
        "stock_vs_ideal_maxabs"
    ]  # the hoist may not be further from the ideal than stock's own rounding
    rep["pass"] = bool(
        rep["swizzle_roundtrip"]
        and rep["swizzle_probe_self"]
        and rep["route_pack_roundtrip"]
        and rep["cross_rank_bitwise"]
        and all(rep[f"cover_r{k}"] for k in range(world))
        and all(
            rep[f"pack_a_r{k}_maxabs"] <= 1e-6 * scale
            and rep[f"pack_r_r{k}_maxabs"] <= 1e-6 * scale
            for k in range(world)
        )
        and rep["hoist_vs_ideal_maxabs"] <= tol
        and rep["negctl_maxabs"] > 10 * tol
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        import json

        rep = selftest(t=a.rows, seed=a.seed)
        print(json.dumps(rep, indent=1))
        raise SystemExit(0 if rep["pass"] else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
