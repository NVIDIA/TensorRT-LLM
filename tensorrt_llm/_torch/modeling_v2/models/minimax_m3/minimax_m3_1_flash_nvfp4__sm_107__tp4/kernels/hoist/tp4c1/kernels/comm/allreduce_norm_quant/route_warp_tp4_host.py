# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 routing-warp post (route_warp_tp4_ext.cu; world 4, one rx part per rank): install() makes its extension
router_exchange's; posts on a blk with h9_prr launch the routing-warp kernel."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile

import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ....toolchain.paths import csrc_dir

_log = logging.getLogger(__name__)

_HERE = csrc_dir("comm", "allreduce_norm_quant")
SRC = os.path.join(_HERE, "route_warp_tp4_ext.cu")
HDRS = (
    "route_warp_tp4_ext.cu",
    "route_warp_tp4.cuh",
    "allreduce_norm_quant.cuh",
    "lamport_exchange.cuh",
)
_st = {}


def src_hash() -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for f in HDRS:
        with open(os.path.join(_HERE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


# DEV (opt-in): M3LL_DEV_OPROJ_PTRIG=1 moves the lr4 o_proj's dependent launch from after its push to after its wait
# (GEMV warp 1 and the router CTAs, ptrig 1) and makes this post wait at the end of every CTA (PW 1), so the post is a
# waiting, polling consumer (rule 6 after_wait, assumption A6). Both halves or neither: one flag.
PTRIG_ENV = "M3LL_DEV_OPROJ_PTRIG"


def ptrig_on() -> bool:
    v = os.environ.get(PTRIG_ENV, "")
    if v not in ("", "0", "1"):
        raise ValueError(f"{PTRIG_ENV}={v!r}: expected 0 or 1")
    return v == "1"


def install():
    """Build this copy once (rank 0 first), make it lbh's ext, and route prr=1 into the launches of tagged blks."""
    from . import router_exchange as lbh

    if "hh" in _st:
        assert lbh._st.get("ext") is _st["ext"], (
            "route_warp_tp4: router_exchange's ext was replaced after install"
        )
        return
    if "ext" in lbh._st:
        raise RuntimeError(
            "route_warp_tp4: router_exchange's ext was loaded before the install (rx workspaces would live in the "
            "other ext)"
        )
    from torch.utils.cpp_extension import load

    hh = src_hash()
    bd = os.path.join(_K.get("TMPDIR", tempfile.gettempdir()), "h9prr4-build-" + hh)
    os.makedirs(bd, exist_ok=True)

    def _load():
        return load(
            name="h9_prr4_" + hh,
            sources=[SRC],
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
    pw = ptrig_on()
    kw0 = lbh._kw

    def _kw(blk):
        kw = kw0(blk)
        if getattr(blk, "h9_prr", False):
            kw.update(prr=1, rx2=rx2)
            if pw:
                kw.update(pwait=1)
        return kw

    lbh._kw = _kw
    _st["ext"], _st["hh"] = e, hh
    lbh._st["ext"] = e
    if not dist.is_initialized() or dist.get_rank() == 0:
        _log.info("route_warp_tp4: router_exchange ext = h9_prr4_%s rx2=%d", hh, rx2)
        if pw:
            print(
                f"M3LL_DEV_OPROJ_PTRIG=1: lr4 post lbv3rp PW=1 (griddepcontrol.wait at CTA end; ext h9_prr4_{hh})",
                flush=True,
            )


def make_oproj_ra4_p(blk):
    install()
    from ...moe.router import fused_o_proj_tp4_host as ra4

    return ra4.make_oproj_ra4(blk)


def make_post_rh_x4_p(blk):
    install()
    from ...moe.engine import post_nvfp4_handoff as b1x4

    blk.h9_prr = True
    return b1x4.make_post_rh_x4(blk)
