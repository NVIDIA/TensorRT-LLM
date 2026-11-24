# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 o_proj GEMV with the router role fused in (TP4 shapes: Ko 2048, world 4)."""

from __future__ import annotations

import atexit
import hashlib
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob

NR = 16  # router CTAs
RE = 8  # experts per router CTA
E = 128
KO = 2048
HK = 1536
HKP = 2048  # padded r-part columns
NPART = 1
H = 6144
THR = 256
_st = {}
_TIMERS = []
_DBG = []


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _dump():
    if _DBG:
        for li, d in _DBG:
            v = d.cpu().tolist()
            if v[0]:
                print(
                    f"H9_RA_R8 r{_rank()} L{li}: oep/rxc mismatch recorded (tag {v[1]}, rxc {v[2]})",
                    flush=True,
                )
    if not _TIMERS:
        return
    import json

    try:
        rec = [
            {
                "layer": li,
                "t": t,
                "router": rs.view(NR, 8).cpu().tolist(),
                "gemv": g.view(-1, 8).cpu().tolist(),
            }
            for li, t, rs, g in _TIMERS
        ]
        with open(os.path.join(_K.get("MINI_OUT", "."), f"ra_r{_rank()}.json"), "w") as f:
            json.dump(rec, f)
    except Exception as ex:  # noqa: BLE001
        print(f"H9_RA timers dump failed: {ex!r}", flush=True)


atexit.register(_dump)


def blobs(router, post_norm, o_w_e4m3, o_sf_u8, rank: int, world: int = 4):
    """(blob_a, blob_r, W', W''_k). blob_a [r][e][m][tid][4] = W''_k[8r + e, 16 tid + 4 m + j] (2 MB, chunk-major:
    conflict-free LDS.128); blob_r [r][e][m][tid][4] = W'[8r + e, c0 + 1024 m + 4 tid + j] (1.5 MB, coalesced LDG.128).
    o_sf_u8: the raw (unswizzled) UE8M0 [H, KO / 32] of this rank's o_proj slice."""
    from ..engine import router_hoist_weights as hw

    wp = hw.fold_router(router, post_norm)
    w2 = hw.fold_oproj(wp, hw.mxfp8_dequant(o_w_e4m3, o_sf_u8))
    assert wp.shape == (E, H) and w2.shape == (E, KO), (wp.shape, w2.shape)
    a = (
        w2.reshape(NR, RE, THR, KO // 1024, 4).permute(0, 1, 3, 2, 4).contiguous()
    )  # [r][e][m][tid][4] = W''[., 8 tid + 4 m + j]
    c0, c1 = hw.r_cols(rank, world)
    assert c1 - c0 == HK
    rp = torch.zeros(E, HKP, dtype=wp.dtype)
    rp[:, :HK] = wp[:, c0:c1]
    r = rp.reshape(NR, RE, HKP // 1024, THR, 4).contiguous()
    return a, r, wp, w2


def emulate(blob_a, blob_r, xq_f, sc, res, rank: int, t: int):
    """CPU model of one rank's router CTAs with the kernel's own flat indexing (fp64: layout check, not rounding).
    xq_f [t][KO] e4m3 values, sc [t][KO / 32] scale factors, res [t][H]. Returns (z_a, z_r) [t][E]."""
    ba, br = blob_a.reshape(-1).double(), blob_r.reshape(-1).double()
    za = torch.zeros(t, E, dtype=torch.float64)
    zr = torch.zeros(t, E, dtype=torch.float64)
    tid = torch.arange(THR)
    for r in range(NR):
        for e in range(RE):
            for m in range(HKP // 1024):
                w = br[
                    (((r * RE + e) * (HKP // 1024) + m) * THR * 4)
                    + (tid[:, None] * 4 + torch.arange(4))
                ]
                off = m * 1024 + 4 * tid[:, None] + torch.arange(4)
                off = torch.where(
                    off < HK, off, 4 * tid[:, None] + torch.arange(4)
                )  # the kernel's clamped load
                col = rank * HK + off
                for tt in range(t):
                    zr[tt, r * RE + e] += (w * res[tt].double()[col]).sum()
            for m in range(KO // 1024):
                w = ba[
                    (((r * RE + e) * (KO // 1024) + m) * THR * 4)
                    + (tid[:, None] * 4 + torch.arange(4))
                ]
                col = 8 * tid[:, None] + 4 * m + torch.arange(4)
                for tt in range(t):
                    x = xq_f[tt].double()[col] * sc[tt].double()[tid // 4][:, None]
                    za[tt, r * RE + e] += (w * x).sum()
    return za, zr


def selftest(seed: int = 0, t: int = 4) -> dict:
    """CPU: blob layouts + kernel indexing vs W''_k a + W'[:, H_k] r, and the rank-sum identity vs stock W_r x /
    rstd."""
    from ..engine import router_hoist_weights as hw

    g = torch.Generator().manual_seed(seed)
    router = torch.randn(E, H, generator=g) * 0.02
    post_norm = torch.randn(H, generator=g) * 0.1
    res = torch.randn(t, H, generator=g).bfloat16().float()
    out = {}
    zs = []
    W = 4
    for rank in range(W):
        ow = (torch.randn(H, KO, generator=g) * 0.5).to(torch.float8_e4m3fn)
        osf = torch.randint(120, 125, (H, KO // 32), generator=g, dtype=torch.uint8)
        a, r, wp, w2 = blobs(router, post_norm, ow, osf, rank, W)
        xq, code = hw.mxfp8_quant_rows(torch.randn(t, KO, generator=g))
        xs = hw.e8m0_scale(code)
        za, zr = emulate(a, r, xq.float(), xs, res, rank, t)
        ref_a = (w2.double() @ (xq.double() * xs.repeat_interleave(32, 1)).T).T
        c0, c1 = hw.r_cols(rank, W)
        ref_r = (wp[:, c0:c1].double() @ res[:, c0:c1].double().T).T
        out[f"a{rank}"] = float((za - ref_a).abs().max() / ref_a.abs().max())
        out[f"r{rank}"] = float((zr - ref_r).abs().max() / ref_r.abs().max())
        p = (ow.double() * hw.e8m0_scale(osf).repeat_interleave(32, 1)) @ (
            xq.double() * xs.repeat_interleave(32, 1)
        ).T
        zs.append((za + zr, p.T))
    za_good = za
    h = res.double() + sum(z[1] for z in zs)
    stock = (wp.double() @ h.T).T  # W' h = W_r diag(1+g) h (rstd factors out)
    out["sum"] = float((sum(z[0] for z in zs) - stock).abs().max() / stock.abs().max())
    out["ok"] = (
        all(v < 1e-12 for k, v in out.items() if k != "sum") and out["sum"] < 1e-6
    )  # sum: W'' fp32 rounding
    # negative control: a transposed a-blob must fail
    za_bad, _ = emulate(a.flip(3).contiguous(), r, xq.float(), xs, res, W - 1, t)
    out["neg_red"] = bool((za_bad - za_good).abs().max() > 1e-6)
    return out


def _ck():
    hang = _K.get("H9_RH_HANG_DBG", "0") == "1"
    return int(knob("router_rx_check", "2" if hang else "1"))


def _blob_dev(blk, rank):
    from ..engine import router_hoist_weights as hw

    key = (id(blk.mw), id(blk.aw), rank)
    hit = _st.get(key)
    if hit is not None:
        return hit
    o = blk.aw.o
    sf = hw.unswizzle_128x4(o.s.view(torch.uint8), o.n, o.k // 32)
    a, r, _, _ = blobs(blk.mw.router, blk.aw.post_norm, o.w, sf, rank, 4)
    a, r = a.to(blk.normed2.device), r.to(blk.normed2.device)
    sha = hashlib.sha256(a.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
    _st[key] = (a, r, sha)
    return _st[key]


def make_oproj_ra4(blk):
    from ...comm.allreduce_norm_quant import router_exchange as lbh
    from ...common import allreduce_push_workspace as k2c
    from ..engine import router_hoist_host as hoist
    from .fused_o_proj import host as gemv

    gemv.use_world(4)

    lin, t = blk.aw.o, blk.t
    assert hasattr(blk, "h10_oq"), (
        "decode must be an MXFP8-output decode (blk.h10_oq / h10_osf), built first"
    )
    assert lin.w.shape == (H, KO), lin.w.shape
    base = (
        2,
        8,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    )  # TP4: mxpfx at split 2 (K 2048: 8 tiles per CTA, ring 8 = 128 KB)
    ck = _ck()
    timed = knob("router_timed", "0") == "1"
    cfg = (base[:11] + (1,) + base[12:] if timed else base) + (NR, ck)
    from ...comm.allreduce_norm_quant import route_warp_tp4_host as prr4

    ptrig = prr4.ptrig_on()
    if ptrig:  # DEV: trigger_early survives the push mode (host._resolve); the post waits at its end (PW 1)
        cfg = cfg[:10] + (1,) + cfg[11:]
    arspec = k2c._prx_spec(t)
    assert arspec[5] == 4 and arspec[3] == 4, arspec
    rank = arspec[2]
    slot = lbh.rx_slot(blk, NPART)
    assert slot.maxt == arspec[4] and slot.world == 4 and slot.npart == NPART, (slot.maxt, arspec)
    assert t <= 8
    a, r, sha = _blob_dev(blk, rank)
    dev = blk.normed2.device
    oep = torch.zeros(NR * 32, dtype=torch.int32, device=dev)
    lbh.register_counter(oep)
    dbg = torch.zeros(4, dtype=torch.int32, device=dev)
    rstamp = torch.zeros(NR * 8, dtype=torch.int64, device=dev)
    gt = (
        torch.zeros(lin.w.shape[0] // 128 * 4 * 8, dtype=torch.int64, device=dev) if timed else None
    )
    res = blk.residual_mid
    assert res.dtype == torch.bfloat16 and res.shape[-1] == H and res.is_contiguous()
    xq, xsf = blk.h10_oq[:t], blk.h10_osf
    assert xq.is_contiguous() and xq.shape[1] == KO and xq.data_ptr() % 16 == 0
    tab = torch.tensor(
        [
            a.data_ptr(),
            r.data_ptr(),
            res.data_ptr(),
            xq.data_ptr(),
            xsf.data_ptr(),
            slot.bases.data_ptr(),
            oep.data_ptr(),
            slot.rxc.data_ptr(),
            dbg.data_ptr(),
            rstamp.data_ptr(),
        ],
        dtype=torch.int64,
        device=dev,
    )
    if timed:
        _TIMERS.append((blk.layer_idx, t, rstamp, gt))
    if ck == 2:
        _DBG.append((blk.layer_idx, dbg))
    w = SimpleNamespace(
        blob=a,
        jobs=r,
        span=torch.zeros(1, dtype=torch.int32, device=dev),
        sha=sha,
        fmt="ra4-fp32",
        npart=NPART,
        maxj=0,
    )
    # hoist's pairing contract (o_proj rh <-> post_rh on this blk) and mini_rh's RH_HASH record
    blk.h9_oproj_rh = SimpleNamespace(
        solo=False,
        oep=oep,
        slot=slot,
        w=w,
        blob=w.blob,
        jobs=w.jobs,
        span=w.span,
        rh_w=0,
        rh_r=1,
        rh_neg=0,
        rh_ck=1,
        rh_ckdbg=int(ck == 2),
        rxc=slot.rxc,
        l_cfg={"ra": NR},
        ra_tab=tab,
        hold=(dbg, rstamp, gt),
    )
    # shape check without the push (the ra kernel compiles on fn's first eager call, as h12r_slot / lr)
    assert gemv.m3_mx_gemv(xq, xsf, lin.w, lin.s, cfg=base, use_pdl=True) is not None
    ptrs = (res.data_ptr(), xq.data_ptr(), xsf.data_ptr())
    boundary = hoist._boundary_claim()
    st = {"n": 0}
    if _rank() == 0:
        print(
            f"TP4_RA4_OPROJ L{blk.layer_idx} t{t} cfg {list(cfg)} blob {sha} slot {slot.bases_host[0]:#x}",
            flush=True,
        )
        if ptrig:
            print(
                f"M3LL_DEV_OPROJ_PTRIG=1: lr4 o_proj L{blk.layer_idx} ptrig 1 (dependents launch after the wait: "
                f"GEMV warp 1 + router CTAs)",
                flush=True,
            )

    def fn():
        hoist._st["rh_building"] = False
        if (blk.residual_mid.data_ptr(), blk.h10_oq.data_ptr(), blk.h10_osf.data_ptr()) != ptrs:
            raise RuntimeError("lr4 o_proj: residual_mid / h10_oq / h10_osf moved after the build")
        if st["n"] == 0:
            if not getattr(blk, "h9_post_rh", False):
                raise RuntimeError(
                    f"LROUTER3 L{blk.layer_idx}: o_proj without make_post_rh(_x4) on its layer"
                )
            if boundary:
                lbh.rx_reset_all()
        st["n"] += 1
        gemv.m3_mx_gemv(
            xq,
            xsf,
            lin.w,
            lin.s,
            cfg=cfg,
            use_pdl=True,
            ar=arspec,
            timers=gt,
            ra_tab=tab.data_ptr(),
        )

    return fn


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print(json.dumps({f"t{t}": selftest(0, t) for t in (1, 4)}))
