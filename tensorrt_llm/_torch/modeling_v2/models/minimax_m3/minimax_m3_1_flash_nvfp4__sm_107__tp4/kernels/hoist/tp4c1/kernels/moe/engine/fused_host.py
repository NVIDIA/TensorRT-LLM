# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds one fused engine state per layer (make_k3) and keeps the push registry the next layer's front reads."""

import json
import os

import torch

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob

_st = {}


def _g3():
    from ...comm.allreduce_norm_quant import lamport_allreduce_host as g3

    return g3


def _bucket(t):
    return 8 if t <= 8 else 16 if t <= 16 else 48


def _push_reg():
    return _st.setdefault("pushed", {})


def _dump_tdbg():
    import atexit

    if _st.get("dump"):
        return
    _st["dump"] = True

    def dump():
        out = _K.get("MINI_OUT")
        if not out or not _K.get("M3_TIMING"):
            return
        res = {}
        for i, st in enumerate(_st.get("states", [])):
            td = st.tdbg.cpu()
            n = st.num_ctas
            t0 = td[:n, 0][td[:n, 0] > 0].min().item()
            row = {}
            for sid in range(64):
                off = 0 if sid in (26, 27, 28, 30, 31) else t0  # durations / counts, not timestamps
                v = sorted(td[c, sid].item() - off for c in range(n) if td[c, sid].item() > 0)
                if v and sid != 63:
                    row[sid] = (round(v[len(v) // 2] / 1e3, 2), round(v[-1] / 1e3, 2))
            res[f"layer{i}"] = row
        rk = _K.get("RANK", "0")
        with open(os.path.join(out, f"h4-tdbg-r{rk}.json"), "w") as f:
            json.dump(res, f, indent=1)

    atexit.register(dump)


V5 = {"sf2_w6": 1, "b2buf": 1, "stages": 16, "smem_slack": 3584}


def make_k3(blk, cfg_over=None, reg="blks", *, op, sharded=False, shared_ctas_le4=None):
    mw, t = blk.mw, blk.t
    if t > 8:
        raise ValueError("H4 make_k3: rows <= 8")
    _K.setdefault("M3_G1_ITP", str(mw.inter))
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    tp2 = mw.inter > 768
    big = t > 4
    if shared_ctas_le4 is not None and not big:
        nsh = shared_ctas_le4
    else:
        nsh = int(
            knob(
                "fused_shared_ctas_5_8_rows" if big else "fused_shared_ctas_le4_rows",
                "48" if tp2 else "24",
            )
        )
    ncta = sms - nsh
    push = knob("fused_push", "1") == "1"
    extra = json.loads(
        knob(
            "fused_engine_cfg_5_8_rows" if big else "fused_engine_cfg_le4_rows",
            '{"early0": 0}' if (big and tp2) else "{}",
        )
    )
    if cfg_over:
        extra = dict(extra, **cfg_over)
    if tp2 and not sharded:
        extra.setdefault(
            "sfa_bulk", 1
        )  # FC1 scales pre-spliced: one bulk copy per stage (-1 us/layer, bit-exact)
    extra.setdefault(
        "roll_fc2", 1
    )  # FC2 k-tile loops rolled: i-cache hot (C1 -2.2, C2 -4.1 us/layer, bit-exact)
    arp = None
    ar_v3 = None
    if push:
        g3 = _g3()
        ext, wss = g3._pr(1)
        arp, peers, ar_v3 = op.push_params_v3(ext.push_params_bytes(wss[_bucket(t)]), dev)
        blk.h4_peers = peers  # keep the peer-pointer array alive
    arin = knob("fused_allreduce_in") == "1" and t % 4 == 0
    desc = None
    if arin:
        import struct

        g3 = _g3()
        ext, wss = g3._pr(1)
        b = bytes(ext.push_params_bytes(wss[_bucket(t)]).cpu().numpy().tobytes())
        peers = struct.unpack_from("<8Q", b, 0)
        flags = struct.unpack_from("<Q", b, 64)[0]
        rank, world, maxt, cluster, ndst = struct.unpack_from("<5i", b, 72)
        if ndst != world:
            raise ValueError("H4 AR_IN reads the local unicast buffer (ndst == world)")
        aux = torch.zeros(
            64, dtype=torch.int32, device=dev
        )  # counter at 0, xready at +128 B (own line)
        psq = torch.zeros(
            ncta * t + 4, dtype=torch.float32, device=dev
        )  # per-CTA row sum-of-squares partials
        eps_bits = struct.unpack("<i", struct.pack("<f", g3.EPS))[0]
        desc = torch.tensor(
            [
                peers[rank],
                flags,
                rank,
                maxt,
                blk.residual_mid.data_ptr(),
                blk.residual_out.data_ptr(),
                blk.aw.post_norm.data_ptr(),
                eps_bits,
                aux.data_ptr(),
                aux.data_ptr() + 128,
                psq.data_ptr(),
            ],
            dtype=torch.int64,
            device=dev,
        )
        blk.h4_arin = (desc, aux, psq)
        extra = dict(extra, ar_in=1, ar_in_w=world, ar_in_cl=cluster)
    st = op.M3MoeState(
        dev,
        num_tokens=t,
        num_ctas=ncta,
        shared=True,
        n_shared=nsh,
        extra_cfg=extra or None,
        ar_v3=ar_v3,
        sharded=sharded,
        sharded_scales_full=sharded,
    )
    sw = (mw.shared_gu.w, mw.shared_gu.s, mw.shared_dn.w, mw.shared_dn.s)
    sc = (
        mw.g1_alphas,
        mw.g1_scale_c,
        mw.gemm1_alpha,
        mw.gemm1_beta,
        mw.gemm1_clamp_limit,
        mw.g2_alphas,
        mw.bias,
    )
    kw = dict(router_weight=mw.router, fc31_input_scale=mw.a1_gscale.reshape(1), shared_weights=sw)
    if sharded:
        from . import expert_shards as g1e

        sh = g1e._shards(mw)
        lay = st.layer(*sh[0], *sc, shard1=sh[1], **kw)
    else:
        lay = st.layer(*mw.experts, *sc, **kw)
    blk.h4_state = st
    _st.setdefault("states", []).append(st)
    _dump_tdbg()
    lay(
        blk.normed2, out=blk.ffn_out
    )  # compile outside capture (no push, no AR_IN: the slots stay untouched)
    torch.cuda.synchronize()
    # push only when a later boundary_in claims these rows (the last layer has no consumer: unconsumed pushes would
    # race the next step's layer-0 boundary on the same slots); read at capture time, after every layer is built
    base = torch.zeros(8, dtype=torch.int64, device=dev)
    if desc is not None:
        base[6] = desc.data_ptr()
        if arp is not None:
            arp[6] = desc.data_ptr()
    pfq = bool(
        extra.get("pf_next")
    )  # H4_CFG {"pf_next":1,"pf_where":1}: shared CTAs prefetch the next layer's QKV
    pf2 = bool(extra.get("pf_fc2")) and not sharded
    if pf2:  # arp[7]: the FC2 weight base for the engine's routing-time L2 prefetch
        base[7] = mw.experts[2].data_ptr()
        if arp is not None:
            arp[7] = mw.experts[2].data_ptr()
    blk.h4_arp0 = base
    hold = {"arp": base if (desc is not None or pf2 or pfq) else None}
    blks = _st.setdefault(reg, [])
    blks.append(blk)
    pfs = {}

    def pf_init():
        # the next layer's QKV weight + scales (K1 reads them first after this engine): table for arp[6] / arp[7]
        if pfs or not pfq:
            return
        i = next(k for k, b in enumerate(blks) if b is blk)
        nx = blks[(i + 1) % len(blks)].aw.qkv
        segs = [nx.w, nx.s]
        tb = torch.tensor(
            [v for t_ in segs for v in (t_.data_ptr(), t_.numel() * t_.element_size())],
            dtype=torch.int64,
            device=dev,
        )
        pfs["tb"] = tb
        for a_ in (base, arp):
            if a_ is not None:
                a_[6] = tb.data_ptr()
                a_[7] = len(segs)

    if push:
        _push_reg()[id(blk.ffn_out)] = (hold, arp, blk.ffn_out)  # the tensor ref pins the id

    def fn():
        if pfq and not pfs:
            pf_init()  # first (eager) call: every layer is built by then
        lay(blk.normed2, out=blk.ffn_out, arp=hold["arp"])

    return fn
