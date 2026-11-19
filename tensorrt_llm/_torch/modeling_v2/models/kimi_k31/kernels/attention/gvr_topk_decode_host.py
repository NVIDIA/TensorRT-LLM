# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-sampling GVR top-K decode — host side (dispatch, workspace, entry).

Companion to ``gvr_topk_decode.py`` (the device module).
Three sections:

1. dispatch — the CUDA host dispatch as a pure function
   ``route(b, n, npad, k, num_sms=148, sm_version=100)``;
2. workspace — one zero-initialised per-device slab (20,973,568 B) via the
   torch caching allocator, with keep-alive + double-checked locking;
3. operator entry — ``run_varlen`` (decode rows: per-request ``kv_lens``
   read on-device, per-row MTP offsets — sync-free and CUDA-graph-replay
   safe with growing KV) and ``run_prefill`` (prefill rows), with input
   hardening.
"""

import math
import operator
import threading

import torch

_dev_mod = None


def _device():
    """Lazy import of the merged device module (first routed shape compiles;
    a broken/absent device module only fails when actually reached)."""
    global _dev_mod
    if _dev_mod is None:
        try:
            from . import gvr_topk_decode as _m  # in-tree
        except ImportError:  # standalone dir
            import gvr_topk_decode as _m
        _dev_mod = _m
    return _dev_mod


# ===========================================================================
# ==== dispatch =============================================================
# ===========================================================================
"""Pure-Python mirror of the GVR CUDA host dispatch (gvr_topk_launch).

route(b, n, npad, k, num_sms, sm_version) is a PURE function of its arguments -- no env
knobs, no GPU, stdlib only.  It returns the kernel family, its compile-time
template tuple, the runtime scalar pack `rt`, grid/cluster/block geometry,
smem size, and whether the family needs the workspace.

rt carries the FULL runtime scalar list each kernel receives, in signature
order, always starting with (n, npad, k).

Dead ABI-parity args: gvr_main's `int SCAP_, int CMP_` params are NEVER read
by the kernel body -- it recomputes them as constexprs that mirror the host
formulas bit-identically.  They are kept in rt purely for ABI parity.
gvr_clus's SCAP/CMP are LIVE runtime args.  `aim` and `SFAC` are host-side
intermediates only (never cross the ABI), so they do not appear in rt.

C-semantics notes encoded here:
  * every `/` on ints is C truncating division -> Python `//` (all operands
    are non-negative on every reachable path);
  * `sel = (long long)SFAC * n / aim` and the TGT/TGT2 products are 64-bit in C;
    Python ints are exact, so `//` reproduces them;
  * `int r = (int)(0.5 + sqrt((double)(6LL*n)))` truncates toward zero after
    the +0.5 -> `int(0.5 + math.sqrt(float(6*n)))`;
  * `IMGW = (n + 3) & ~3` four-element float4 round-up;
  * the reg-block CMP (possibly widened to n by DEGE) is scoped to the
    register-resident block; the streaming path re-derives its own CMP.
"""


# ---- dispatch constants (must match the device kernels) ---------------------
NB = 1024  # register-path histogram bins
SNB = 256  # streaming-path bin count
CMPC = 4096  # crossing-bin slots per CTA, clustered register path
BLKC = 1024  # CTA size of the clustered register path


def route(
    b: int,
    n: int,
    npad: int,
    k: int,
    num_sms: int = 148,
    sm_version: int = 100,
) -> dict[str, object]:
    """Select a kernel and launch resources from the shape and device properties."""
    if b < 1:
        raise RuntimeError(f"route requires b >= 1, got {b}")
    if num_sms < 1:
        raise RuntimeError(f"route requires num_sms >= 1, got {num_sms}")
    # Only the register-resident rungs scale with the available SM count. The
    # streaming and clustered-register paths below retain their independently
    # tuned 148/296 constants.
    wide = b <= num_sms

    # ======================= register-resident block ========================
    n4 = n >> 2
    CMP = n if n < 2560 else 2560
    QC = 160
    CURE = not (n < 2 * k and b > num_sms)
    if not wide and k == 2048 and sm_version in (100, 103) and 512 < n4 <= 1024:
        CURE = True
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
    if DEGE and CMP < n:
        CMP = n
    long_register = (
        not wide
        and sm_version in (100, 103)
        and n4 > 2048
        and ((k == 2048 and n4 <= 3072) or (512 < k <= 1024 and n4 <= 5 * 512))
    )
    short_wide = wide and k == 2048 and sm_version in (100, 103) and 512 < n4 <= 1024
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 2048 and not wide)) else NB
    if long_register or short_wide:
        NBSEL = NB
    IMGOFF = NBSEL

    def _reg(BLK, VPT, MINB, NBH):
        # Deterministic emitters pack one bit per scalar into a 32-bit mask.
        if VPT > 8:
            BLK *= 2
            VPT = (VPT + 1) // 2
        # DEG wins over the CUR flag; DEG forces KPT=1, else KPT ladder 1/2/4.
        if DEGE:
            tpl = (BLK, VPT, MINB, 1, CURE, True, False, NBH)
        else:
            kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else 4)
            tpl = (BLK, VPT, MINB, kpt, CURE, False, False, NBH)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "CMP": CMP,
                "IMGOFF": NBH,
                "QC": QC,
            },
            "grid": (b, 1),
            "cluster": 1,
            "block": BLK,
            "smem": (NBH + 2 * CMP) * 4,
            "ws": False,
        }

    IMGW = (n + 3) & ~3
    smi = (NBSEL + (2 * CMP if 2 * CMP > IMGW else IMGW)) * 4
    IMGE = wide and (not DEGE) and k <= 1024

    if n4 <= 256:
        return _reg(256, 1, 8, NB)
    if n4 <= 512:
        return _reg(512, 1, 4, NB)
    if n4 <= 1024:
        if wide:
            if short_wide:
                return _reg(512, 2, 1, NB)
            if IMGE:
                # regimg launch: gvr_topk_reg<1024,1,2,1,true,false,true,2048>
                return {
                    "kernel": "regimg",
                    "tpl": (1024, 1, 2, 1, True, False, True, 2 * NB),
                    "rt": {
                        "n": n,
                        "npad": npad,
                        "k": k,  # full ABI
                        "CMP": CMP,
                        "IMGOFF": IMGOFF,
                        "QC": QC,
                    },
                    "grid": (b, 1),
                    "cluster": 1,
                    "block": 1024,
                    "smem": smi,
                    "ws": False,
                }
            return _reg(1024, 1, 1, 2 * NB)
        if sm_version in (100, 103) and k == 2048 and n4 <= 768:
            return _reg(256, 3, 4, NB)
        return _reg(512, 2, 4, NB)

    if k == 1024 and b in (512, 1024) and 1024 < n4 <= 1536 and sm_version in (100, 103):
        return _reg(256, (n4 + 255) // 256, 4, NB)

    # Seven/eight resident fragments use rematerialized bins in the varlen launcher.
    if k == 1024 and b in (512, 1024) and 1536 < n4 <= 2048 and sm_version in (100, 103):
        return _reg(256, (n4 + 255) // 256, 4, NB)

    if k == 512 and sm_version in (100, 103):
        if b > 6 * num_sms and 1024 < n4 <= 2560:
            if n4 <= 2048:
                return _reg(512, (n4 + 511) // 512, 4, NB)
            return _reg(256, (n4 + 255) // 256, 4, NB)
        if num_sms < b <= 2 * num_sms and 2048 < n4 <= 3072:
            return _reg(512, (n4 + 511) // 512, 2, NB)

    # Wide K=2048 rows that fit one wave of register CTAs (<= 4 streaming splits).
    k2048_reg_rung = False
    if wide and k == 2048 and sm_version in (100, 103) and n4 <= 8192:
        streaming_splits = min(148 // b, (n4 + 1023) // 1024) if b <= 32 else 1
        k2048_reg_rung = streaming_splits <= 4 or (b <= 16 and n == 32768)

    # ---- clustered register-resident path ----
    # K=2048 runs with two hint slots per thread (tuple slot KH); the one-wave rung keeps its rows.
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and (k <= BLKC or (k == 2 * BLKC and not k2048_reg_rung)):
        av = 148 // (b if b > 0 else 1)  # truncating
        amax = 1
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            # cs=8 co-residency veto: an 8-CTA cluster with b > 15 exceeds
            # GPC packing; such shapes fall through to the streaming path.
            for v in (1, 2, 4):
                c = 1  # 64-bit product in C
                while c * BLKC * v < n4:
                    c <<= 1
                if c == 8 and b > 15:  # the veto
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            smc = (3 * NB + 2 * CMPC) * 4
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs, k),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": smc,
                "ws": False,
            }

    # 4K < n <= 8K: select register geometry from the row count and K.
    if n4 <= 2048:
        if wide:
            if k == 2048 and sm_version in (100, 103) and n4 <= 1536:
                return _reg(512, 3, 1, 2 * NB)
            return _reg(1024, 2, 1, 2 * NB)
        sm103_large_batch_k2048 = (
            sm_version in (100, 103) and num_sms == 148 and 512 <= b <= 1024 and k == 2048
        )
        if b <= 2 * num_sms or sm103_large_batch_k2048:
            if n4 <= 1536:
                if sm_version in (100, 103) and k == 2048 and n4 <= 1280 and b > 2 * num_sms:
                    sparse_final_slot = n4 <= 4 * 256 + 128
                    if b <= 4 * num_sms or (b > 6 * num_sms and sparse_final_slot):
                        return _reg(256, 5, 4, NB)
                    if b > 5 * num_sms:
                        return _reg(256, 5, 3, NB)
                return _reg(512, 3, 2, NB)
            return _reg(512, 4, 2, NB)

    if n4 <= 4096 and wide:
        vpt = 3 if n4 <= 3072 else 4
        return _reg(1024, vpt, 1, 2 * NB)

    if long_register:
        return _reg(512, 5 if n4 <= 2560 else 6, 2, NB)

    if k2048_reg_rung:
        return _reg(1024, (n4 + 1023) // 1024, 1, 2 * NB)

    if n <= 12288:
        return _reg(512, (n4 + 511) // 512, 2, NBSEL)

    # ====================== streaming / collect path ========================
    R = 1
    if b <= 32:
        r1 = 148 // b
        if r1 < 1:
            r1 = 1
        r2 = ((n >> 2) + 1023) // 1024
        if r2 < 1:
            r2 = 1
        R = r1 if r1 < r2 else r2
        if R < 1:
            R = 1
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:  # shallow R=2 split
        R = 2

    useclus = False
    if 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        # gvr_clus cs=8 hits the same GPC packing wall as the clustered
        # register path; same veto, same b > 15 threshold.
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True

    big = b * R <= 148
    SCAP = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    CMP = (4096 if k > 1024 else 2048) if big else 1024

    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    q = 6 * n  # 6LL * n
    r = int(0.5 + math.sqrt(float(q)))  # C cast trunc
    if r > aim:
        aim = r
    SFAC = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (SCAP >> 1):
        aim = SCAP >> 1
    if aim < k:
        aim = k

    n4s = n >> 2
    SMP, SS2, TGT, TGT2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= SCAP and n > 2 * k
    if (n > SCAP or small_dense) and n4s >= 4:  # PAIR sample
        sel = SFAC * n // aim  # 64-bit
        if sel < 256:
            sel = 256
        if sel > n // 2:
            sel = n // 2
        pairs = sel >> 3
        if pairs < 1:
            pairs = 1
        half = n4s >> 1
        if half < 1:
            half = 1
        if pairs > half:
            pairs = half
        SS2 = half // pairs
        if SS2 < 1:
            SS2 = 1
        SMP = half // SS2
        if SMP < 1:
            SMP = 1
        TGT = (aim * (SMP * 8)) // n  # 64-bit
        if TGT < 1:
            TGT = 1
        TGT2 = (k * (SMP * 8)) // n  # 64-bit
        if TGT2 < 1:
            TGT2 = 1
    Q = (n4s + R - 1) // R

    if useclus:
        if n > SCAP and n4s >= 4:  # QUAD override
            sel = SFAC * n // aim
            if sel < 256:
                sel = 256
            if sel > n // 2:
                sel = n // 2
            quads = sel >> 4
            if quads < 1:
                quads = 1
            quarter = n4s >> 2
            if quarter < 1:
                quarter = 1
            if quads > quarter:
                quads = quarter
            SS2 = quarter // quads
            if SS2 < 1:
                SS2 = 1
            SMP = quarter // SS2
            if SMP < 1:
                SMP = 1
            TGT = (aim * (SMP * 16)) // n
            if TGT < 1:
                TGT = 1
            TGT2 = (k * (SMP * 16)) // n
            if TGT2 < 1:
                TGT2 = 1
        smc = SNB * 8 + (SCAP + 4) * 8 + CMP * 8
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        CS = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, U, 1, SNB, CS),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # ABI (live)
                "SCAP": SCAP,
                "CMP": CMP,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (CS, b),
            "cluster": CS,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }

    smem_main = (SCAP + 4) * (8 if (R > 1 or b <= 296) else 4) + (CMP + 1) * 8

    def _main(BLK, MINB, U, SPLIT):
        # KPT ladder 1/2/4/8; grid = (R, b).
        kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else (4 if k <= 4 * BLK else 8))
        # TSH-floor staging gate.  The CUDA form is a grid-uniform RUNTIME
        # gate (gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768); here it
        # is a compile-time key -- per-launch semantics are identical
        # because the gate is uniform over the grid.
        tshg = bool(SPLIT) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (BLK, U, MINB, SNB, kpt, SPLIT, tshg),
            # SCAP_/CMP_ are dead ABI-parity args: gvr_main never reads them
            # (it recomputes them as constexprs).
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "SCAP_": SCAP,
                "CMP_": CMP,
                "R": R,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": BLK,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, U, R > 1)  # SPLIT iff R>1
    if b <= 296:
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)


def route_streaming(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """route() restricted to its STREAMING half (main / clus) — the varlen
    capture policy: per-row kernels must be picked from the families that are
    correct for ANY row length, so the register-resident specialists are
    skipped even when the envelope n would normally land on them.  Where
    route() itself lands on main/clus this is IDENTICAL to route().
    force_main additionally skips the clus rounding, so the raw
    min(r1, r2) R matches the CUDA else-branch exactly."""
    if b < 1:
        raise RuntimeError(f"route_streaming requires b >= 1, got {b}")
    R = 1
    if b <= 32 or (force_main and k > 1024 and b <= _VARLEN_SPLIT_MAX_ROWS):
        r1 = max(148 // b, 1)
        r2 = max(((n >> 2) + 1023) // 1024, 1)
        R = max(min(r1, r2), 1)
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:
        R = 2
    useclus = False
    if not force_main and 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True
    big = b * R <= 148
    scap = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    cmp_ = (4096 if k > 1024 else 2048) if big else 1024
    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    r_ = int(0.5 + math.sqrt(float(6 * n)))
    if r_ > aim:
        aim = r_
    sfac = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (scap >> 1):
        aim = scap >> 1
    if aim < k:
        aim = k
    n4s = n >> 2
    smp, ss2, tgt, tgt2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= scap and n > 2 * k
    if (n > scap or small_dense) and n4s >= 4:
        sel = min(max(sfac * n // aim, 256), n // 2)
        pairs = min(max(sel >> 3, 1), max(n4s >> 1, 1))
        half = max(n4s >> 1, 1)
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if useclus:
        if n > scap and n4s >= 4:
            sel = min(max(sfac * n // aim, 256), n // 2)
            quads = min(max(sel >> 4, 1), max(n4s >> 2, 1))
            quarter = max(n4s >> 2, 1)
            ss2 = max(quarter // quads, 1)
            smp = max(quarter // ss2, 1)
            tgt = max((aim * (smp * 16)) // n, 1)
            tgt2 = max((k * (smp * 16)) // n, 1)
        smc = SNB * 8 + (scap + 4) * 8 + cmp_ * 8
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        cs = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, u_, 1, SNB, cs),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP": scap,
                "CMP": cmp_,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (cs, b),
            "cluster": cs,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }
    smem_main = (scap + 4) * (8 if (R > 1 or b <= 296) else 4) + (cmp_ + 1) * 8

    def _main(blk_, minb_, u_, split_):
        kpt = 1 if k <= blk_ else (2 if k <= 2 * blk_ else (4 if k <= 4 * blk_ else 8))
        tshg = bool(split_) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (blk_, u_, minb_, SNB, kpt, split_, tshg),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP_": scap,
                "CMP_": cmp_,
                "R": R,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": blk_,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, u_, R > 1)
    if b <= 296:
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)


# (rows, npad, k, n_env, next_n, cr, packed device profile) -> compiled launcher
_VARLEN_CACHE = {}

# ---- prefill launcher cache ------------------------------------------------
# Prefill forces R==1 (route_streaming gives R>1 only for b<=74). The compiled
# launcher depends only on the row tier, k and the envelope bucket — never on the
# exact row count or npad — so the cache stays bounded on a long-running server.
_PREFILL_CACHE = {}
_PREFILL_ROW_SLAB = 32768  # gridDim.y <= 65535; slab so keys stay bounded
_PREFILL_TIER_ROWS = (75, 149, 297)  # (rows<=148, 149..296, >296) band reps
# The tier-0 plan is the BLK=1024 non-split slab, whose compile-time pair-sample
# gate is n > 16384, so under an envelope <= 16384 every row runs the unsampled
# path. The tier-1 BLK=512 plan samples above its own gate (4096 for k <= 1024,
# 8192 for k > 1024); <= 148-row launches take it when the envelope is above
# that gate by a margin (just above the gate the freshly sampling BLK=512 plan
# is slower than the unsampled BLK=1024 plan for b >= 32) and at most 16384
# (above that the BLK=1024 plan samples too and is the better slab).
_PREFILL_T1_MARGIN = 256
_PREFILL_T1_MAX = 16384


def _prefill_scpb_tier1(k: int) -> int:
    return 8192 if k > 1024 else 4096


def _prefill_tier(rows: int, n_env: int, k: int) -> int:
    tier = 0 if rows <= 148 else 1 if rows <= 296 else 2
    if tier == 0 and _prefill_scpb_tier1(k) + _PREFILL_T1_MARGIN < n_env <= _PREFILL_T1_MAX:
        tier = 1
    return tier


def _prefill_bucket(n_env: int) -> int:
    # pow2-quantize the envelope so a growing envelope reuses one plan; cap at
    # 32768 because U=8 for every n>=32768 on the tier-0 arm.
    return min(1 << max(int(n_env) - 1, 1).bit_length(), 32768)


def _prefill_cache_key(tier: int, k: int, n_bucket: int):
    # tiers 1/2 fix U, so the bucket does not change their engine — collapse it
    # to one key so warmup covers them with a single launch.
    return (tier, k, n_bucket if tier == 0 else 0)


def _prefill_launcher(tier: int, k: int, n_bucket: int) -> tuple:
    """Prefill plan + compiled launcher: ``_varlen_launcher``'s main branch with
    r_const=1, split=False and the prefill compile flag. SCAP_/CMP_ are envelope
    upper bounds; npad is filled per call in ``run_prefill``."""
    key = _prefill_cache_key(tier, k, n_bucket)
    hit = _PREFILL_CACHE.get(key)
    if hit is not None:
        return hit
    b_route = _PREFILL_TIER_ROWS[tier]
    n_route = max(n_bucket, k + 1)
    plan = route_streaming(b_route, n_route, n_route, k, force_main=True)
    if plan["kernel"] != "main":
        raise RuntimeError(f"prefill route did not land on gvr_main: {plan['kernel']}")
    rt = plan["rt"]
    if rt["R"] != 1:
        raise RuntimeError(f"prefill requires R==1 (got {rt['R']})")
    tpl = tuple(plan["tpl"])
    dev = _device()
    fn = dev.get_compiled(tpl[:6] + (False,) + (1, 0, 1), hint_free=True, prefill=True, fused_k=k)
    big = tier == 0
    # r_const==1 branch of the _varlen_launcher tuning scalars
    aim_base = (
        (4 * k if k >= 1024 else 2 * k) if big else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = 64 if k >= 1024 else 32
    amin = (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    tail = (aim_base, sfac, amin, sd_en, 0)  # tsh_en=0 (split=False)
    lc = ("main", fn, (rt["SCAP_"], rt["CMP_"]), tail)
    _PREFILL_CACHE[key] = lc
    return lc


#: (smallest R, largest U) pairs of _varlen_split_tile, first match wins: R >= 10 CTAs per row take a 2-vector
#: tile, 2 .. 9 CTAs per row a 4-vector tile.
_VARLEN_TILE_CAPS = ((10, 2), (2, 4))
#: route_streaming(force_main=True) splits the rows of launches with K > 1024 over R = min(148 // rows, n / 4096)
#: CTAs up to this many rows (3 or 4 CTAs per row past 32 rows); larger launches keep one CTA per row. At 2 CTAs
#: per row (50-74 rows) the rows of some layers take a 100-145 us pass in the model (TEP16 at 64 rows: 15 % of the
#: launches), which costs more than the split saves on the others.
_VARLEN_SPLIT_MAX_ROWS = 49


def _varlen_split_tile(tpl: tuple, r_const: int) -> tuple:
    """U (float4 vectors per thread per tile) of the deterministic SPLIT main family under varlen.

    route_streaming sizes U for rows of the envelope's length (the server's max_seq_len, e.g. 1M), but each of
    the R CTAs of a row only streams n / R of it and decode rows are mostly far shorter than the envelope.
    _VARLEN_TILE_CAPS holds the GB300 choice: at K = 2048 over 64k-300k scores per row a 2-vector tile is as
    fast or faster with R >= 10 (8 rows: 64k 12.7 -> 11.1 us, 120k 12.9 -> 11.1, 170k 12.6 -> 12.0, 300k 14.6
    -> 14.7; 10 rows: 12.3 -> 11.1, 12.0 -> 11.7, 12.0 -> 11.9, 15.3 -> 15.0) and a 4-vector tile with R in
    2 .. 9 (16 rows: 13.6 -> 11.5, 13.3 -> 11.6, 13.5 -> 12.9, 16.4 -> 16.1; 24 rows: 12.7 -> 11.4, 12.9 -> 12.7,
    13.4 -> 13.3, 16.9 -> 17.0; 48 rows at R = 3: 120k 14.5 -> 14.2, 170k 15.4 -> 15.2). The deterministic
    emission order depends on neither U nor R (the merge CTA ends each row in ascending order), so the rows are
    the same bits.
    """
    u = tpl[1]
    for r_min, u_max in _VARLEN_TILE_CAPS:
        if r_const >= r_min:
            u = min(u, u_max)
            break
    return tpl[:1] + (u,) + tpl[2:]


def enable_early_release() -> None:
    """Turn on EARLY RELEASE for this process's deterministic varlen remap launches: they release their dependent
    grid right after their own grid-dependency wait (the device module's ``enable_early_release``; call before warmup
    and CUDA graph capture)."""
    _device().enable_early_release()


_SPLIT_LAUNCH_CACHE: dict = {}


def varlen_launch_is_split(
    num_rows: int, npad: int, k: int, n_env: int, num_sms: int, sm_version: int
) -> bool:
    """Whether the deterministic ``run_varlen`` launch of this shape is the main family with ``R > 1`` CTAs per
    row (SPLIT). Its rows end in ascending index order (the merge CTA's bitmap / bitonic canonicalizer; rows with
    ``n <= k`` are the identity), so with ``remap`` it writes each row's pool rows in ascending local order.
    Mirrors ``_varlen_launcher``'s routing; cached per shape."""
    key = (num_rows, npad, k, n_env, num_sms, sm_version)
    hit = _SPLIT_LAUNCH_CACHE.get(key)
    if hit is None:
        n_route = max(min(n_env, npad), k + 1)
        plan_free = route(num_rows, n_route, npad, k, num_sms, sm_version)
        hit = plan_free["kernel"] not in ("reg_clus", "reg", "regimg", "clus") and (
            route_streaming(num_rows, n_route, npad, k, force_main=True)["rt"]["R"] > 1
        )
        _SPLIT_LAUNCH_CACHE[key] = hit
    return hit


def _varlen_launcher(
    num_rows: int,
    npad: int,
    k: int,
    n_env: int,
    next_n: int,
    cr: int,
    num_sms: int = 148,
    sm_version: int = 100,
    remap: bool = False,
) -> tuple:
    """Capture-time varlen plan + compiled launcher.  The gvr_main port is
    the universally correct fallback; specialist family tiers below.  Every
    choice here is a function of capture-stable quantities only — mirroring
    the in-tree runner's pick_tuning(graph_capture=...) discipline.
    ``remap``: the main family is compiled with the paged-pool remap epilogue (run_varlen ``remap``);
    the other families are not, and run_varlen then reports that it left the indices local."""
    profile = _pack_device_profile(num_sms, sm_version)
    key = (num_rows, npad, k, n_env, next_n, cr, profile) + (("remap",) if remap else ())
    hit = _VARLEN_CACHE.get(key)
    if hit is not None:
        return hit
    # Two envelopes: the kernel gets the PHYSICAL bound (never past the row
    # stride, so a row whose kv length exceeds the logits width clamps and
    # takes the short path instead of reading into the next row); the router
    # gets the k+1 floor it needs to pick a non-degenerate family.
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    # ---- route() parity, family tier 1: clustered register-resident --------
    # Admit reg_clus exactly where the free route picks it; its whole
    # admission window (n4 <= 32768) fits capture-frozen envelopes. The
    # choice is a pure function of this cache key, so CUDA-graph replay
    # safety is unchanged; per-row n / short-row handling lives in-kernel.
    plan_free = route(num_rows, n_route, npad, k, num_sms, sm_version)
    if plan_free["kernel"] == "reg_clus":
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
        )
        lc = ("reg_clus", fn, n_kernel)
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 2: register-resident (+img flavor) ----
    # Same admission rule as tier 1: exactly where the free route picks
    # reg/regimg (the whole small/mid-N band across all row counts). CMP/QC/
    # smem are envelope-derived launch constants -- in-kernel they are pure
    # capacity clamps (CMP), a fast-path threshold (QC) and the launch smem
    # size, all safe upper bounds for every per-row n <= envelope; per-row n
    # / short-row handling lives in-kernel.
    if plan_free["kernel"] in ("reg", "regimg"):
        # Launch smem: the plan's window, plus 4 * k bytes on these shapes (the launch smem, and so
        # the CTA occupancy, the register rungs run at).
        smem_tail = num_rows > num_sms and not (
            k <= 512 and n_kernel <= 2 * k and num_rows <= 2 * num_sms
        )
        if k == 2048 and n_kernel > k and num_rows <= num_sms and sm_version in (100, 103):
            smem_tail = True
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (
                n_kernel,
                rt_f["CMP"],
                rt_f["QC"],
                dev.STATIC_BYTES + plan_free["smem"] + (4 * k if smem_tail else 0),
            ),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 3: cluster split (clus) ---------------
    # Same admission rule: exactly where the free route picks clus (the
    # large-N mid-rows band). SCAP/CMP are launch-stable (pure functions of
    # rows/CS/k — never of n) so the envelope values are the per-row values;
    # the sampling-ladder scalars (SMP/TGT/Q/SS2/TGT2) are dead launch slots,
    # re-derived per row in-kernel by the route_dynamic clus mirror.
    # Per-row n / short-row handling in-kernel.
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            det=True,
        )
        lc = (
            "clus",
            fn,
            (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    plan = route_streaming(num_rows, n_route, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])  # (BLK, U, MINB, SNB, KPT, SPLIT, TSHG)
    if tpl[5]:
        tpl = _varlen_split_tile(tpl, plan["rt"]["R"])
    if k == 512 and num_rows > 2 * num_sms and sm_version in (100, 103):
        n4 = n_route >> 2
        if 2048 < n4 <= 3072 or (num_rows <= 6 * num_sms and 1536 < n4 <= 2048):
            tpl = (256, 4, 4, tpl[3], 2, False, False)
    rt = plan["rt"]
    r_const = rt["R"]
    # TSHG (tpl[6]) is dead under varlen (the ctor compiles the TSH
    # machinery in whenever SPLIT); normalize it out of the compile key so
    # row counts differing only in that slot share one engine
    fn = dev.get_compiled(
        tpl[:6] + (False,) + (next_n, cr_shift, r_const),
        hint_free=True,
        fused_k=k,
        remap=remap,
    )
    big = num_rows * r_const <= 148
    aim_base = (
        ((4 * k if k >= 1024 else 2 * k) if r_const == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = (
        (32 if r_const == 2 else (48 if k > 1024 else 16))
        if r_const > 1
        else (64 if k >= 1024 else 32)
    )
    amin = 3 * k if r_const == 2 else (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    # TSH-floor staging: gate on SPLIT and K only. Gating additionally on
    # num_rows > 15 would strand small batches in SPLIT-main without the
    # staged floor (a distribution-dependent tail regression); the kernel
    # gates TSH per row at runtime anyway.
    tsh_en = 1 if (tpl[5] and k <= 1024) else 0
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE[key] = lc
    return lc


# ===========================================================================
# ==== workspace ============================================================
# ===========================================================================
"""Per-device workspace slab for the multi-CTA SPLIT path.

Semantics:
  * ONE zero-initialised slab workspace per device, lazily allocated through
    the torch caching allocator;
  * keep-alive store: module dict `_ws_keep` (tensor refcount = keep-alive);
  * double-checked locking: lock-free hot-path load (a GIL-atomic dict get
    plays an acquire load), slow path re-checks under a mutex;
  * device index bounds `0 <= d < GVR_MAX_DEV` -- checked BEFORE the
    CUDA-ness of the tensor.

Size: WS_BYTES = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B.

Kernel-facing view: the compiled main-family signature takes the workspace
as a 1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16
-- torch caching-allocator bases are 256B-aligned so the slab always
satisfies it).
"""


# workspace geometry constants -- must match the device kernels
GVR_MAX_DEV = 64
_MAXC = 160
_GCAP = 16384
_GVR_WS_BUF_OFF = 2048
WS_BYTES = _GVR_WS_BUF_OFF + _MAXC * _GCAP * 8  # 20,973,568
assert WS_BYTES == 20_973_568

_mu = threading.Lock()  # slow-path mutex
_ws_keep = {}  # device index -> keep-alive int32 view


def default_workspace(ref: torch.Tensor) -> torch.Tensor:
    """Per-device cached workspace slab.

    Returns the kernel-facing 1-D int32 view (zero-initialised on first use;
    the kernel restores the zeros it consumes, so one zeroing suffices for
    the lifetime of the cache entry)."""
    d = ref.get_device()
    if not (0 <= d < GVR_MAX_DEV):
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_keep.get(d)  # hot path: one (GIL-atomic) load
    if ws is not None:
        return ws
    with _mu:  # slow path: double-checked
        ws = _ws_keep.get(d)
        if ws is not None:
            return ws
        # lazy zeros via the torch caching allocator, viewed int32 for the
        # DSL launch signature.
        buf = torch.zeros(WS_BYTES, dtype=torch.uint8, device=ref.device)
        ws = buf.view(torch.int32)
        _ws_keep[d] = ws  # keep-alive (ws_keep[d] = tensor)
        return ws


# ===========================================================================
# ==== operator entry =======================================================
# ===========================================================================
"""Operator entry: input hardening and dispatch of the varlen decode and prefill launches.

The device module is imported LAZILY (first shape that routes to it), so a
missing/broken module only fails when actually reached.
"""


_DEVICE_PROFILE = {}
_DEVICE_PROFILE_SHIFT = 16
_DEVICE_NUM_SMS_MASK = (1 << _DEVICE_PROFILE_SHIFT) - 1


# hot-path local bindings: each torch.<attr> lookup costs ~0.1 us and the
# validation battery runs on EVERY call
_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_is_capturing = torch.cuda.is_current_stream_capturing
_index = operator.index
_ws_hot = _ws_keep  # shared dict object (hot-path load)
_GVR_MAX_DEV = GVR_MAX_DEV
_get_device_properties = torch.cuda.get_device_properties


def _pack_device_profile(num_sms: int, sm_version: int) -> int:
    """Pack route topology into one cache-key integer."""
    if not 1 <= num_sms <= _DEVICE_NUM_SMS_MASK:
        raise RuntimeError(f"invalid SM count for device profile: {num_sms}")
    if sm_version < 1:
        raise RuntimeError(f"invalid SM version for device profile: {sm_version}")
    return (sm_version << _DEVICE_PROFILE_SHIFT) | num_sms


def _unpack_device_profile(profile: int) -> tuple[int, int]:
    """Return ``(SM count, SM version)`` from a packed cache key."""
    return profile & _DEVICE_NUM_SMS_MASK, profile >> _DEVICE_PROFILE_SHIFT


def _device_profile_key(dev_index: int) -> int:
    """Return the packed route profile, cached per CUDA device."""
    profile = _DEVICE_PROFILE.get(dev_index)
    if profile is None:
        properties = _get_device_properties(dev_index)
        num_sms = int(properties.multi_processor_count)
        sm_version = int(properties.major) * 10 + int(properties.minor)
        try:
            profile = _pack_device_profile(num_sms, sm_version)
        except RuntimeError as error:
            raise RuntimeError(f"device {dev_index} reports an invalid profile: {error}") from error
        _DEVICE_PROFILE[dev_index] = profile
    return profile


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------


def run_varlen(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    max_seq_len: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    remap: tuple | None = None,
) -> bool:
    """Run hint-free self-sampling Top-K with per-request device KV lengths.

    ``remap = (block_table, block_size, stride_factor, layer_offset)`` (``compress_ratio`` 1): where the main family
    serves the launch, its writer CTAs also map every selected index to its paged-pool index in place, the integers
    of ``convertReqIndexToGlobal`` with ``req_id[r] = r // next_n`` (``block_table`` int32 ``[batch, max_blocks]``,
    rows contiguous, 4-byte aligned). Returns whether the indices are pool indices on return (False: local indices,
    the caller converts them).

    The kernels are deterministic: a fixed selected set (ties broken by the smaller index) and a
    fixed per-route output order.

    Row semantics (mirror of ``heuristicTopKDecode.cu`` and the in-tree
    ``cute_dsl_gvr_topk_decode`` runner):

      ``num_rows = logits.shape[0]``, ``batch = num_rows // next_n``;
      ``kv_lens`` int32 ``[batch]`` — per-request TOTAL cache length in
      UNCOMPRESSED token space (dsa.py ``metadata.kv_lens_cuda_runtime``,
      not new-token seq_lens); row ``r`` uses
      ``n_r = (kv_lens[r // next_n] - next_n + (r % next_n) + 1) //
      compress_ratio`` valid entries (cr 1 = DSv3.2, 4 = DSv4 Flash/Pro);
      the bracket is derived from the current row itself (register families
      use resident scores; streaming families do not consume a temporal
      hint on the accept path); ``k`` comes from
      ``indices.shape[1]``;
      per-row ``n_r <= k`` takes the short path (identity + ``-1`` tail).

    The per-row in-kernel engine launches once for the whole batch. Each CTA
    reads its row's kv_len on device and re-derives the sampling ladder (route_dynamic
    formula mirror); the envelope is ``max_seq_len`` (a capture-stable engine
    constant, e.g. dsa.py's ``indexer_max_seq_len``), so the call performs NO
    host reads.

    KNOWN LIMITATION: on rows containing NaN logits the selected index SET
    can differ from ``heuristicTopKDecode.cu`` (both kernels order NaNs
    implementation-specifically). Finite inputs — including +/-inf and
    denormals — are tie-aware exact.

    CONTRACT: correct and dispatched for any ``num_rows``
    (BS 1..1024+ x next_n) and any envelope up to 1M kv tokens.  Family
    selection (streaming main / clustered register-resident) is a pure
    function of the capture-stable launcher key.
    """
    if logits.dtype is not torch.float32:
        raise RuntimeError(f"logits must be float32 (got {logits.dtype})")
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda):
        raise RuntimeError("kv_lens must be a CUDA tensor")
    if kv_lens.dtype is not _I32:
        raise RuntimeError("kv_lens must be int32")
    if kv_lens.dim() != 1:
        raise RuntimeError("kv_lens must be 1-D")
    nn = _index(next_n)
    cr = _index(compress_ratio)
    if nn < 1:
        raise RuntimeError(f"next_n must be >= 1, got {nn}")
    if cr not in (1, 4):
        raise RuntimeError(f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4), got {cr}")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return remap is not None
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_hot.get(d)
    if ws is None:
        ws = default_workspace(logits)

    # ---- per-row in-kernel engine (gvr_main varlen port) ----------------
    # Full validation battery (the batch-dim check is CRITICAL: the kernel grid comes from
    # logits.shape[0], so a short indices tensor would be written
    # out of bounds).
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    k = indices.shape[1]
    if not (indices.is_contiguous() and kv_lens.is_contiguous()):
        raise RuntimeError("indices/kv_lens must be contiguous")
    # logits: accept row-major views with a wider row stride (the DSL
    # paged-MQA logits arena is 256-aligned and column-sliced — a legal
    # NON-contiguous view). The kernel only needs (base, row stride):
    # widen back to a compact [rows, stride] view over the same storage;
    # the tail columns are never classified (per-row n gates all reads).
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // logits.element_size() < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 3:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    cshift = 0 if cr == 1 else 2
    n_env = min(max(int(max_seq_len) >> cshift, 1), npad)
    if remap is not None:
        if cr != 1:
            raise RuntimeError("remap needs compress_ratio 1")
        block_table = remap[0]
        if (
            not block_table.is_cuda
            or block_table.dtype is not _I32
            or block_table.dim() != 2
            or block_table.stride(1) != 1
            or block_table.stride(0) != block_table.shape[1]
            or block_table.shape[0] < num_rows // nn
            or block_table.data_ptr() & 3
        ):
            raise RuntimeError(
                "remap block_table must be a row-major int32 [batch, blocks] CUDA tensor"
            )
    profile = _device_profile_key(d)
    key = (num_rows, npad, k, n_env, nn, cr, profile) + (("remap",) if remap is not None else ())
    lc = _VARLEN_CACHE.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        num_sms, sm_version = _unpack_device_profile(profile)
        lc = _varlen_launcher(
            num_rows,
            npad,
            k,
            n_env,
            nn,
            cr,
            num_sms,
            sm_version,
            remap=remap is not None,
        )
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    # Hint-free engines do not read the compiled kernel's pre_idx ABI slot.
    pre_arg = idx
    if lc[0] == "reg_clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_envelope)
        lc[1](lg, pre_arg, kv_lens, idx, lc[2])
    elif lc[0] == "reg":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, CMP, QC, smem)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    elif lc[0] == "clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, npad, k,
        #                SCAP, CMP, dead DYN x5)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    else:
        _, fn, pre, tail = lc
        if remap is not None:
            # REMAP: block table in the (hint-free, unread) pre_idx slot; stride factor / layer offset x block size /
            # blocks per request / block size in the dead varlen SMP / TGT / Q / TGT2 slots.
            block_table, block_size, stride_factor, layer_offset = remap
            pre_arg = block_table
            pre = pre[:6] + (
                int(stride_factor),
                int(layer_offset) * int(block_size),
                int(block_table.shape[1]),
                pre[9],
                int(block_size),
            )
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
        if remap is not None:
            return True
    return False


def run_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    max_row_len: int,
    tier_rows: int | None = None,
) -> None:
    """Hint-free self-sampling Top-K for prefill: row ``r`` selects the Top-K of
    ``logits[r, ks:ke]`` (compressed columns) into the local frame (column - ks)
    with a -1 pad; ``nv <= k`` rows get the identity, as ``indexer_topk_prefill``.
    No device reads, never compiles under capture; trusts 0 <= ks <= ke <= shape[1].
    The selection is deterministic, as in ``run_varlen``; ``max_row_len`` is the
    envelope of the row windows.

    The engine plan (row tier) follows the launch's row count, and the order a row's
    indices come out in depends on the plan. ``tier_rows`` picks the plan as if the
    launch had that many rows, so a slice of a larger tile gets the tile's per-row
    output bit for bit (every plan is correct for any row count)."""
    if logits.dtype is not _F32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); the kernel has no bf16/fp16 path"
        )
    for _nm, _t in (("row_starts", row_starts), ("row_ends", row_ends)):
        if not (isinstance(_t, _TENSOR) and _t.is_cuda):
            raise RuntimeError(f"{_nm} must be a CUDA tensor")
        if _t.dtype is not _I32:
            raise RuntimeError(f"{_nm} must be int32")
        if _t.dim() != 1:
            raise RuntimeError(f"{_nm} must be 1-D")
        if not _t.is_contiguous():
            raise RuntimeError(f"{_nm} must be contiguous")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if row_starts.shape[0] != num_rows or row_ends.shape[0] != num_rows:
        raise RuntimeError(
            f"row_starts/row_ends length must equal logits.shape[0]={num_rows}, "
            f"got {row_starts.shape[0]}/{row_ends.shape[0]}"
        )
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(f"indices must be [num_rows={num_rows}, k], got {tuple(indices.shape)}")
    if not indices.is_contiguous():
        raise RuntimeError("indices must be contiguous")
    k = indices.shape[1]
    if k < 4 or (k & 3):
        raise RuntimeError(f"index_topk must be a multiple of 4 and >= 4, got {k}")
    if indices.data_ptr() & 15:
        raise RuntimeError("indices base must be 16-byte aligned")
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    # key on stride(0) for every row count: DeepGEMM prefill rows are 1024B-aligned
    # with slack, and the varlen 1-row shape[1] rule would reject odd-width tiles.
    npad = logits.stride(0)
    if npad & 3:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    lg = logits
    if logits.shape[1] != npad:
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    ws = _ws_hot.get(d)
    if ws is None:
        ws = default_workspace(logits)
    n_env = min(max(_index(max_row_len), 1), npad)
    n_bucket = _prefill_bucket(n_env)
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        r1 = min(r0 + _PREFILL_ROW_SLAB, num_rows)
        tier = _prefill_tier(tier_rows if tier_rows is not None else r1 - r0, n_env, k)
        lc = _PREFILL_CACHE.get(_prefill_cache_key(tier, k, n_bucket))
        if lc is None:
            if _is_capturing():
                raise RuntimeError(
                    "prefill launcher not compiled for this shape — warm up "
                    "before CUDA graph capture"
                )
            lc = _prefill_launcher(tier, k, n_bucket)
        _, fn, (scap, cmp_), tail = lc
        # varlen main ABI: pre_idx slot = row_ends, kv_lens slot = row_starts;
        # only npad / k / SCAP_ / CMP_ matter (R=1), the other scalars are dead.
        pre = (0, npad, k, scap, cmp_, 1, 0, 0, 0, 0, 0)
        fn(lg[r0:r1], row_ends[r0:r1], indices[r0:r1], ws, *pre, row_starts[r0:r1], *tail)


__all__ = [
    "route",
    "run_varlen",
    "run_prefill",
    "warmup_prefill",
    "WS_BYTES",
    "default_workspace",
]


_PREFILL_WARMUP_DONE: set = set()
_PREFILL_WARMUP_LOCK = threading.Lock()
#: Launch row counts the warmup evaluates the prefill tier at.
_PREFILL_WARMUP_ROWS = (1, 149, 297)


def warmup_prefill(top_k: int, max_cols: int) -> None:
    """Compile every prefill engine ``run_prefill`` can request before serving:
    the tier-0 arm per pow2 envelope bucket where a <= 148-row launch keeps it
    (``_prefill_tier`` is evaluated at both edges of every bucket), tiers 1/2
    one launch each. ``max_cols`` is the compressed max column count;
    idempotent per done-key."""
    dev = torch.cuda.current_device()
    k = int(top_k)
    max_cols = int(max_cols)
    lo = _prefill_bucket(k + 1)
    hi = _prefill_bucket(max_cols)
    buckets = []
    b = lo
    while b <= hi:
        buckets.append(b)
        b <<= 1
    if not buckets:
        buckets = [hi]
    keys = {}  # cache_key -> (tier, bucket, envelope) representative for the launch
    for rows in _PREFILL_WARMUP_ROWS:
        for bk in buckets:
            for n_env in (bk // 2 + 1, bk):  # the tier can change inside a bucket
                tier = _prefill_tier(rows, n_env, k)
                keys.setdefault(_prefill_cache_key(tier, k, bk), (tier, bk, n_env))
    done_key = (dev, k, max_cols)
    with _PREFILL_WARMUP_LOCK:
        if done_key in _PREFILL_WARMUP_DONE:
            return
    for tier, bk, n_env in keys.values():
        rows = _PREFILL_TIER_ROWS[tier]
        stride = (bk + 256 + 255) // 256 * 256
        logits = torch.zeros((rows, stride), dtype=torch.float32, device=dev)
        ks = torch.zeros((rows,), dtype=torch.int32, device=dev)
        ke = torch.full((rows,), n_env, dtype=torch.int32, device=dev)
        out = torch.empty((rows, k), dtype=torch.int32, device=dev)
        run_prefill(logits[:, :bk], ks, ke, out, max_row_len=n_env)
        del logits, ks, ke, out
    torch.cuda.synchronize()
    with _PREFILL_WARMUP_LOCK:
        _PREFILL_WARMUP_DONE.add(done_key)
