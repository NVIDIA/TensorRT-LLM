# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 routing-warp post with mode Q's per-token NVFP4 tail (route_warp_tp4_fc1_per_token_ext.cu), the matching TP4
engine factory, and the mode-Q weight preparation (FC1 per token amax / 2688, FC2 fixed 16)."""

from __future__ import annotations

import hashlib
import logging
import os

import torch
import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ....toolchain.paths import csrc_dir

_log = logging.getLogger(__name__)

POST_DIR = csrc_dir("comm", "allreduce_norm_quant")
HDRS = (
    "route_warp_tp4_fc1_per_token_ext.cu",
    "route_warp_tp4_fc1_per_token.cuh",
    "allreduce_norm_quant.cuh",
    "lamport_exchange.cuh",
)
QAT_MAX = 6.0 * 448.0
NSH = 48  # shared-expert CTAs of the TP4 engine
_st = {}


def src_hash() -> str:
    h = hashlib.sha1()
    for f in HDRS:
        with open(os.path.join(POST_DIR, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def install():
    """Build the pt-t4 post ext once (rank 0 first); make it lbh's and prr4's ext; tag-driven kwargs per blk."""
    from . import route_warp_tp4_host as prr4
    from . import router_exchange as lbh

    if "ext" in _st:
        assert lbh._st.get("ext") is _st["ext"] and prr4._st.get("ext") is _st["ext"], (
            "pt4: lbh / prr4 ext replaced"
        )
        return
    if "ext" in lbh._st or "hh" in prr4._st:
        raise RuntimeError(
            "pt4: lbh / prr4 ext loaded before pt4.install (rx workspaces would live in the other ext)"
        )
    from torch.utils.cpp_extension import load

    hh = src_hash()
    bd = os.path.join(_K.get("TMPDIR", "/tmp"), "pt4post-build-" + hh)
    os.makedirs(bd, exist_ok=True)

    def _load():
        return load(
            name="pt4_post_" + hh,
            sources=[os.path.join(POST_DIR, "route_warp_tp4_fc1_per_token_ext.cu")],
            build_directory=bd,
            verbose=False,
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-gencode=arch=compute_107a,code=sm_107a",
                f"-DH9_HDR_HASH=0x{hh}",
            ],
            extra_cflags=["-O3", "-std=c++20"],
        )

    if dist.is_initialized():
        if dist.get_rank() == 0:
            e = _load()
        dist.barrier()
        if dist.get_rank() != 0:
            e = _load()
    else:
        e = _load()
    rx2 = int(_K.get("PRR_RX2", "1"))
    pw = (
        prr4.ptrig_on()
    )  # DEV M3LL_DEV_OPROJ_PTRIG: the post waits at its end (PW 1) while the o_proj runs ptrig 1
    kw0 = lbh._kw

    def _kw(blk):
        kw = kw0(blk)
        if getattr(blk, "h9_prr", False):
            kw.update(prr=1, rx2=rx2)
            if pw:
                kw.update(pwait=1)
        if getattr(blk, "h9_pt", False):
            b = blk.h9_x4
            assert kw.get("x4_q") is not None and kw.get("prr") == 1, (
                "pt4: the pt post needs X4 + prr"
            )
            kw.update(pt=1, x4_pts=b.pts)
        return kw

    lbh._kw = _kw
    _st["ext"], _st["hh"] = e, hh
    lbh._st["ext"] = e
    prr4._st["ext"], prr4._st["hh"] = e, hh
    if not dist.is_initialized() or dist.get_rank() == 0:
        _log.info(
            "route_warp_tp4_fc1_per_token: router_exchange ext = pt4_post_%s (%s)", hh, POST_DIR
        )
        if pw:
            print(
                f"M3LL_DEV_OPROJ_PTRIG=1: lr4 pt4 post lbv3rp PW=1 (griddepcontrol.wait at CTA end; ext pt4_post_{hh})",
                flush=True,
            )


def _pt_bufs(blk):
    """blk's X4 buffers with one extra scale row: sf_full [X4R + 1][384], row X4R = fp32 s_t [t] (pts)."""
    from . import router_exchange as lbh

    b = getattr(blk, "h9_x4", None)
    if b is None:
        b = blk.h9_x4 = lbh.X4Bufs(blk)
    if not hasattr(b, "pts"):
        assert b.rows == max(blk.t, 16) and not b.post, (b.rows, blk.t)
        b.sf_full = torch.zeros(b.rows + 1, b.sf.shape[1], dtype=torch.uint8, device="cuda")
        b.sf = b.sf_full[: b.rows]
        b.pts = b.sf_full[b.rows].view(torch.float32)
    return b


def make_oproj(blk):
    """oproj: the TP4 fused o_proj with router CTAs, after this install."""
    install()
    from ...moe.router import fused_o_proj_tp4_host as ra4

    return ra4.make_oproj_ra4(blk)


def make_post_pt(blk):
    """boundary_post: the prr4 rh post with the per-token X4 tail (X4 2) and s_t into row X4R."""
    install()
    from ...moe.engine import post_nvfp4_handoff as b1x4

    b = _pt_bufs(blk)
    blk.h9_prr = True
    blk.h9_pt = True
    fn = b1x4.make_post_rh_x4(blk)
    if _K.get("PT4_POSTCHK") != "1":
        return fn
    done = []

    def fn_chk():
        fn()
        if not done and not torch.cuda.is_current_stream_capturing():
            done.append(1)
            postchk(blk, b)

    return fn_chk


def postchk(blk, b):
    """PT4_POSTCHK=1: x4 q / sf / s_t vs the torch row-quant emulation on the post's own normed2 (one-shot, eager), with
    two red controls on the same comparator (s_t x (1 + 2^-8); static-scale-1 x4 bytes) that must mismatch."""
    torch.cuda.synchronize()
    t = blk.t
    q, sf, st = qat_row_quant(blk.normed2[:t])
    gq, gsf, gst = b.q[:t], b.sf[:t], b.pts[:t]
    nq = int((gq != q).sum())
    nsf = int((gsf != sf.view(torch.uint8)).sum())
    nst = int((gst != st).sum())
    red_st = int((gst != st * (1 + 2**-8)).sum())
    import vllm._custom_ops as ops

    q1, _ = ops.scaled_fp4_quant(
        blk.normed2[:t].contiguous(), torch.ones(1, device="cuda"), is_sf_swizzled_layout=False
    )
    red_q = int((gq != q1.view(torch.uint8).reshape(t, -1)).sum())
    reds = red_st == t and red_q > 0
    st_ulp = float(((gst - st).abs() / (st.abs() * 2**-8)).max())
    verdict = (
        "PASS"
        if nq == nsf == nst == 0
        else "NEAR"
        if st_ulp <= 1 and nq + nsf <= 0.001 * gq.numel()
        else "FAIL"
    )
    ok = reds and verdict != "FAIL"
    rank = int(_K.get("RANK", "0"))
    _log.warning(
        f"PT4 POSTCHK r{rank} L{blk.layer_idx}: q mism {nq}/{gq.numel()} sf mism {nsf}/{gsf.numel()} "
        f"s_t mism {nst}/{t} (s_t {gst.tolist()} ref {st.tolist()}); red s_t*(1+2^-8) mism {red_st}/{t}, "
        f"red static-x4 q mism {red_q} -> {verdict if ok else 'FAIL'} (s_t max {st_ulp:.2f} bf16 ulp; "
        f"reds {'ok' if reds else 'VACUOUS'})"
    )


def _sht_pt_build():
    from ...moe.engine import router_hoist_host as hoist

    def b(blk, reg, cfg, **route):
        if int(cfg.get("ext_route", 0)):
            red = int(_K.get("PT4_RED", "0"))
            cfg = dict(cfg, sh_trig=1, pt5=1, **({"pt5_red": red} if red else {}))
        return hoist._c1_build(
            blk, reg, cfg, **route
        )  # resolved at call: moe4._patched's TP4 cfg merge applies

    return b


def _moe_pt(blk):
    from ...moe.engine import router_hoist_host as hoist
    from . import router_exchange as lbh

    if getattr(blk.mw, "pt5_fc2", None) != 16.0:
        raise RuntimeError(
            "pt4: the Q engine needs the FC2 scale-16 weight patch (pt4mini / pt4srv prep)"
        )
    b = blk.h9_x4
    if not getattr(b, "post", False) or not hasattr(b, "pts"):
        raise RuntimeError(
            f"pt4 L{blk.layer_idx}: the pt engine needs make_post_pt as this layer's boundary_post"
        )
    seen = []
    build = _sht_pt_build()

    def build_x4(blk_, reg, cfg, **route):
        if int(cfg.get("ext_route", 0)) != 1:
            return build(blk_, reg, cfg, **route)
        cfg = dict(cfg, b1_x4=1)
        on_layer = route.pop("on_layer", None)
        patched = []

        def observe(st):
            run = type(st).run

            def run_obs(layer, hidden, *a, **kw):
                lay = getattr(layer, "_lay", layer)
                pick = lambda k: kw[k] if kw.get(k) is not None else getattr(lay, k, None)  # noqa: E731
                seen.append(
                    dict(state=st, x4=pick("x4"), x4sf=pick("x4sf"), sfsc=lay.fc31_input_scale)
                )
                return run(st, layer, hidden, *a, **kw)

            st.run = run_obs
            patched.append(st)
            if on_layer is not None:
                on_layer(st)

        try:
            return build(blk_, reg, cfg, x4=b.q, x4sf=b.sf_full, on_layer=observe, **route)
        finally:
            for st in patched:
                st.__dict__.pop("run", None)

    fn = hoist._ext(blk, "extx4pt4", build_x4, hoist._state_c1)
    st = hoist._state_c1(blk)
    mod = getattr(st, "mod", None)
    if (
        not seen
        or not getattr(st, "b1_x4", 0)
        or not int(getattr(mod, "PT5", 0))
        or not int(getattr(mod, "SH_TRIG", 0))
    ):
        raise RuntimeError(
            f"pt4 L{blk.layer_idx}: engine not built with b1_x4 + pt5 + sh_trig (seen {len(seen)}, "
            f"PT5 {getattr(mod, 'PT5', None)}): the engine module lacks PT5 / SH_TRIG"
        )
    if not int(getattr(mod, "B2IMG", 0)):
        raise RuntimeError(
            f"pt4 L{blk.layer_idx}: engine built without b2img (TP4_MOE_OVER not merged)"
        )
    s = seen[-1]
    if s["x4sf"] is None or s["x4sf"].data_ptr() != b.sf_full.data_ptr():
        raise RuntimeError(f"pt4 L{blk.layer_idx}: engine x4sf is not this blk's sf_full")
    lbh.check_x4_pair(blk, True, s["x4"], s["x4sf"], s["sfsc"])
    return fn


def make_moe_pt(blk):
    """moe_experts: the TP4 engine with NSH shared-expert CTAs and engine cfg pt5=1 (reads this post's s_t)."""
    from ...moe.engine import fused_tp4_host as moe4

    with moe4._patched(), moe4._nsh(NSH):
        return _moe_pt(blk)


def prep_mw(mw):
    """Decision 3 on a chain mw (idempotent): c = g1_scale_c x 16, fc2_alpha = g2_alphas / 16 (exact powers of two).
    Refuses non-placeholder routed FC1 input scales and an a2 != 1 checkpoint."""
    if getattr(mw, "pt5_fc2", None) == 16.0:
        return mw
    assert float(mw.a1_gscale.reshape(-1)[0]) == 1.0, (
        "FC2-16 patch assumes the M3.1 1.0 placeholder input scales"
    )
    assert torch.equal(mw.g1_scale_c, mw.g1_alphas), (
        "FC2-16 patch assumes the w2 input scale a2 == 1 (c = g1a / a2)"
    )
    mw.g1_scale_c = (mw.g1_scale_c * 16.0).contiguous()
    mw.g2_alphas = (mw.g2_alphas / 16.0).contiguous()
    mw.pt5_fc2 = 16.0
    return mw


_CONST = {}


def _consts(dev):
    k = str(dev)
    if k not in _CONST:
        _CONST[k] = (
            torch.tensor(QAT_MAX, device=dev),
            torch.tensor(6.0, device=dev),
            torch.ones((), device=dev),
        )
    return _CONST[k]


def qat_row_quant(x):
    """Torch reference of the per-token NVFP4 row quantizer (IEEE divisions where the kernel uses div.full /
    rcp.approx): bf16 [t, H] -> (u8 [t, H/2], e4m3 [t, H/16], fp32 s_t [t])."""
    cmax, c6, one = _consts(x.device)
    x = x.float()
    t, h = x.shape
    amax = x.abs().amax(-1).clamp_min(1e-12)
    s_t = torch.div(amax, cmax).bfloat16().float()
    enc = torch.div(torch.full_like(amax, QAT_MAX), amax)
    xn = (x * enc[:, None]).bfloat16().float()
    blocks = xn.view(t, h // 16, 16)
    sixth = torch.div(one, c6)
    sf = (blocks.abs().amax(-1) * sixth).to(torch.float8_e4m3fn)
    inv = torch.div(torch.ones_like(sf, dtype=torch.float32), sf.float())
    sc = blocks * inv.masked_fill(sf.float() == 0, 0.0)[..., None]
    mag = sc.abs()
    code = sum(
        ((mag > th) if strict else (mag >= th)).to(torch.uint8)
        for th, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    )
    code = code | (torch.signbit(sc).to(torch.uint8) << 3)
    code = code.view(t, h // 2, 2)
    return code[..., 0] | (code[..., 1] << 4), sf.view(t, -1), s_t
