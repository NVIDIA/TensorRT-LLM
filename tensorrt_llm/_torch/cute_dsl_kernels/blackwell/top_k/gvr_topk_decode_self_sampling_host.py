# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

Companion to ``gvr_topk_decode_self_sampling.py`` (the device module).
Three sections:

1. dispatch — the CUDA host dispatch as a pure function
   ``route(b, n, npad, k, num_sms=148, sm_version=100)``;
2. workspace — one zero-initialised per-device slab (20,973,568 B) via the
   torch caching allocator, with keep-alive + double-checked locking;
3. operator entry — ``run(logits, pre_idx, n_valid, indices)`` /
   ``run_ws(..., workspace)`` DPS forms with input hardening and a
   bind-once launch cache keyed on shape plus a packed device route profile.

OPERATOR CONTRACT (batch-uniform entries): ``n_valid`` is one host python
int for the whole batch — every row shares the same valid prefix, in
COMPRESSED index space (the caller applies any ``compressRatio`` division).
``pre_idx`` is consumed as-is — raw prev-step top-K indices, uniformly for
DSv3.2 / DSv4 Flash / Pro. The +1 temporal shift ``heuristicTopKDecode.cu``
applies for cr==1 is deliberately dropped: hints only steer the sampling
ladder (exactness never depends on them), and raw prev-step hints overlap
the current top-K at least as well as +1-shifted ones on real decode data,
so one offset-free hint convention serves all three models. The production
per-row contract (per-request ``kv_lens`` read on-device, per-row MTP
offsets — sync-free and CUDA-graph-replay safe with growing KV) is
implemented by ``run_varlen``, which is the entry the opt-in DSA dispatch
seam calls. The batch-uniform ``run``/``run_ws`` entries keep the simpler
contract (one host-side ``n_valid`` for the whole batch), are exercised for
unit tests and benchmarking only, and must not be substituted for
``run_varlen`` under continuous batching, MTP (``next_n > 1``), or
CUDA-graph capture.
"""

import math
import operator
import threading
from collections.abc import Sequence

import torch

_dev_mod = None


def _device():
    """Lazy import of the merged device module (first routed shape compiles;
    a broken/absent device module only fails when actually reached)."""
    global _dev_mod
    if _dev_mod is None:
        try:
            from . import gvr_topk_decode_self_sampling as _m  # in-tree
        except ImportError:  # standalone dir
            import gvr_topk_decode_self_sampling as _m
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
QUADC = 96  # crossing-bin O(mc^2) rank gate (streaming/reg paths, every register plan)
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
    """Mirror of the CUDA gvr_topk_launch dispatch. Pure. See module doc.

    Deviations from the CUDA reference (self-sampling only): the 4K < n <= 8K
    register rungs, QC = QUADC for every register plan, NB bins for the
    not-wide register plans up to n4 <= 2048."""
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
    # QC = QUADC for every register plan: the O(mc^2) rank loop is taken only
    # for mc <= QC, so a wider gate lets one row whose first-k bracket misses
    # the k-th value stall the whole one-wave launch. Rows with mc <= QUADC
    # take the same branch under either gate.
    QC = QUADC
    CURE = not (n < 2 * k and b > num_sms)
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
    if DEGE and CMP < n:
        CMP = n
    # NB bins for the not-wide register plans up to n4 <= 2048 (incl. the
    # (512, 4, 2) rung below) so IMGOFF == NBH holds at every reg site.
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 2048 and not wide)) else NB
    IMGOFF = NBSEL
    smem_reg = (NBSEL + 2 * CMP) * 4

    def _reg(BLK, VPT, MINB, NBH):
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
                "IMGOFF": IMGOFF,
                "QC": QC,
            },
            "grid": (b, 1),
            "cluster": 1,
            "block": BLK,
            "smem": smem_reg,
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
            return _reg(1024, 1, 2, 2 * NB)
        return _reg(512, 2, 4, NB)

    # ---- clustered register-resident path ----
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and k <= BLKC:
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
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": smc,
                "ws": False,
            }

    # 4K < n <= 8K (n4 in (1024, 2048]): one-wave register rungs. Wide rows:
    # BLK=1024 x VPT=2 covers n4 <= 2048 exactly (the VPT=4 plan below carries
    # two empty float4 slots per thread). num_sms < b <= 2*num_sms: two
    # BLK=512 CTAs per SM form one wave; taller batches keep the main slab.
    if n4 <= 2048:
        if wide:
            return _reg(1024, 2, 1, 2 * NB)
        # On SM103 with 148 SMs, the existing register plan remains faster
        # across the measured b=512..1024, k=2048 range. Keep the exception
        # architecture- and shape-exact: smaller K and nearby lower batches
        # have different winners, while SM100 retains its original streaming
        # route byte-for-byte.
        sm103_large_batch_k2048 = (
            sm_version == 103 and num_sms == 148 and 512 <= b <= 1024 and k == 2048
        )
        if b <= 2 * num_sms or sm103_large_batch_k2048:
            return _reg(512, 4, 2, NB)

    if n4 <= 4096 and wide:
        return _reg(1024, 4, 1, 2 * NB)

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


if __name__ == "__main__":
    smoke = [
        # (b, n, npad, k)                          expected family
        (64, 1024, 1024, 512),  # reg   n4<=256 rung (DEG: n<=3k)
        (64, 2048, 2048, 512),  # reg   n4<=512 rung
        (1024, 4096, 4096, 1024),  # reg   n4<=1024, b>148 -> (512,2,4)
        (64, 4096, 4096, 512),  # regimg wide !DEGE k<=1024
        (64, 4096, 4096, 1024),  # reg   wide but DEGE (n<=4k+64)
        (8, 65536, 65536, 1024),  # reg_clus (vsel=2, cs=8; b<=15 no veto)
        (16, 131072, 131072, 512),  # main  cs=8 veto fall-through -> SPLIT slab, tshg=True
        (64, 8192, 8192, 512),  # reg   wide 4K<n<=8K rung (1024,2,1)
        (256, 8192, 8192, 1024),  # reg   148<b<=296, 4K<n<=8K rung (512,4,2)
        (64, 16384, 16384, 1024),  # reg   wide 4k fallback (1024,4,1)
        (64, 262144, 262144, 1024),  # clus  R=2 shallow cluster split
        (1, 1048576, 1048576, 1024),  # main  deep slab SPLIT R=148
        (20, 262144, 262144, 2048),  # main  k>1024 split (no useclus)
        (512, 131072, 131072, 1024),  # main  b>296 BLK=256
        (512, 6144, 6144, 2048),  # main  small_dense sample gate (b > 296)
        (256, 262144, 262144, 2048),  # main  KBIG-domain (k>1024), BLK=512 KPT=4
    ]
    for shp in smoke:
        print(shp, "->", route(*shp))


# ---------------------------------------------------------------------------
# two-time-scale dispatch split (per-row varlen / CUDA-graph groundwork)
# ---------------------------------------------------------------------------
# route(b, n, npad, k, num_sms, sm_version) factored into
#   route_static(...) — everything frozen per launch:
#       family, compile tuple, grid, cluster, block, and the rt scalars that
#       change only at discrete n-thresholds;
#   route_dynamic(static, n)     — the n-continuous scalars a per-row kernel
#       recomputes from its own row length (the device code will mirror these
#       formulas): n, CMP (reg families), the sampling ladder
#       SMP/TGT/SS2/TGT2/Q (streaming families), and the reg-family smem
#       footprint.
# INVARIANT: merging route_dynamic back into route_static reproduces
# route() EXACTLY for every n. The policy of which n to freeze the static
# half at (e.g. max_seq_len) is a perf-only choice — the factorization
# itself is lossless.

_DYN_RT = {
    "reg": ("n", "CMP"),
    "regimg": ("n", "CMP"),
    "reg_clus": ("n",),
    "clus": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
    "main": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
}
_DYN_SMEM = ("reg", "regimg")  # smem depends on CMP/IMGW -> recomputed per n


def route_static(
    b: int,
    n: int,
    npad: int,
    k: int,
    num_sms: int = 148,
    sm_version: int = 100,
) -> dict[str, object]:
    """route() with the n-continuous fields redacted (see _DYN_RT/_DYN_SMEM).
    Constant on maximal n-intervals ("bands"); every redacted field is
    reconstructible from (static, n) by route_dynamic."""
    plan = route(b, n, npad, k, num_sms, sm_version)
    st = {key: (dict(val) if isinstance(val, dict) else val) for key, val in plan.items()}
    for f in _DYN_RT[st["kernel"]]:
        st["rt"].pop(f)
    if st["kernel"] in _DYN_SMEM:
        st.pop("smem")
    return st


def route_dynamic(static: dict[str, object], n: int) -> tuple[dict[str, object], int]:
    """Recompute the redacted n-continuous scalars from (static, n).
    Returns (rt_updates, smem). Must stay equivalent to route(); the
    device-side per-row engine mirrors exactly these formulas."""
    fam = static["kernel"]
    k = static["rt"]["k"]
    if fam in ("reg", "regimg"):
        dege = static["tpl"][5]
        cmp_ = n if dege else (n if n < 2560 else 2560)
        nbsel = static["rt"]["IMGOFF"]
        if fam == "regimg":
            imgw = (n + 3) & ~3
            smem = (nbsel + (2 * cmp_ if 2 * cmp_ > imgw else imgw)) * 4
        else:
            smem = (nbsel + 2 * cmp_) * 4
        return {"n": n, "CMP": cmp_}, smem
    if fam == "reg_clus":
        return {"n": n}, static["smem"]

    # streaming families (main / clus): the sampling-ladder scalars
    b = static["grid"][1]
    if fam == "clus":
        R = static["cluster"]
        scap = static["rt"]["SCAP"]
    else:
        R = static["rt"]["R"]
        scap = static["rt"]["SCAP_"]
    big = b * R <= 148
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
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        pairs = max(sel >> 3, 1)
        half = max(n4s >> 1, 1)
        pairs = half if pairs > half else pairs
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if fam == "clus" and n > scap and n4s >= 4:
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        quads = max(sel >> 4, 1)
        quarter = max(n4s >> 2, 1)
        quads = quarter if quads > quarter else quads
        ss2 = max(quarter // quads, 1)
        smp = max(quarter // ss2, 1)
        tgt = max((aim * (smp * 16)) // n, 1)
        tgt2 = max((k * (smp * 16)) // n, 1)
    return (
        {"n": n, "SMP": smp, "TGT": tgt, "Q": q_, "SS2": ss2, "TGT2": tgt2},
        static["smem"],
    )


def route_split(
    b: int,
    n: int,
    npad: int,
    k: int,
    num_sms: int = 148,
    sm_version: int = 100,
) -> dict[str, object]:
    """route_static + route_dynamic recombined — must equal route() exactly
    (the factorization fuzz in the unit tests asserts this)."""
    st = route_static(b, n, npad, k, num_sms, sm_version)
    dyn, smem = route_dynamic(st, n)
    plan = {key: (dict(val) if isinstance(val, dict) else val) for key, val in st.items()}
    plan["rt"].update(dyn)
    plan["smem"] = smem
    return plan


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
    if b <= 32:
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


def _bf16_halve_u(plan: dict) -> dict:
    """bf16 streaming arms read 8-element 16B vectors, so one vector covers
    what two fp32 float4s did: halve the per-thread vector batch U (tuple slot
    1) to keep per-tile ELEMENT coverage — and the 32-bit classify mask —
    identical while halving the load-instruction count."""
    tpl = list(plan["tpl"])
    tpl[1] = max(int(tpl[1]) // 2, 1)
    plan["tpl"] = tuple(tpl)
    return plan


def _gvr_main_gate(blk: int, kpt: int, split: bool = True) -> tuple[int, int]:
    """(SCPB, CMPB) constexpr mirror of GvrMainKernel.__init__.
    Kept here so a route rung can check what it does to the
    in-kernel degeneracy gate before changing BLK/KPT."""
    kbig = kpt >= 2 and kpt * blk >= 2048
    scpb = (8192 if split else 16384) if blk >= 1024 else 8192 if kbig else 4096
    cmpb = (4096 if kbig else 2048) if blk >= 1024 else 1024
    return (scpb, cmpb)


def _degen_gate_ok(blk_old: int, kpt_old: int, blk_new: int, kpt_new: int) -> bool:
    """True iff moving (blk_old, kpt_old) -> (blk_new, kpt_new) does not shrink
    either half of the degeneracy gate."""
    s_o, c_o = _gvr_main_gate(blk_old, kpt_old)
    s_n, c_n = _gvr_main_gate(blk_new, kpt_new)
    return s_n >= s_o and c_n >= c_o


def _route_bf16(
    b: int, n: int, npad: int, k: int, num_sms: int = 148, sm_version: int = 100
) -> dict[str, object]:
    """bf16 dispatch table. route() is a pure function of shape, so the fp32
    table is the correct starting point; bf16-specific re-tunes (16B-vector U
    halving, register-family capacity fitting, bin-count halving) are applied
    to the returned copy only."""
    plan = route(b, n, npad, k, num_sms, sm_version)
    if plan["kernel"] in ("reg", "regimg") and k == 512 and (n <= 1280) and (b > 148):
        plan["tpl"] = (256, 1, 8, 1, True, True, False, 512)
        plan["block"] = 256
    pk16 = npad <= 65536 and n <= 65536 and (b > 15)
    if plan["kernel"] == "main" and (not plan["tpl"][5]) and (b > 148):
        n4q = n >> 2
        if 1024 < n4q <= 1152:
            CMP = n if n < 2560 else 2560
            QC = QUADC
            CURE = not (n < 2 * k and b > 148)
            DEGE = n <= 3 * k or n <= 4 * k + 64
            if DEGE and CMP < n:
                CMP = n
            vptx = 4
            while vptx > 1 and (512 * (vptx // 2) >= n4q or 512 * (vptx // 2) * 4 + 512 >= n):
                vptx //= 2
            if DEGE:
                tpl = (512, vptx, 4, 1, CURE, True, False, 512)
            else:
                kpt = 1 if k <= 512 else 2
                tpl = (512, vptx, 4, kpt, CURE, False, False, 512)
            if b > 296 and 2 * vptx <= 4 and (k <= 1024):
                tpl = (256, 2 * vptx, 8) + tuple(tpl[3:])
            return {
                "kernel": "reg",
                "tpl": tpl,
                "rt": {"n": n, "npad": npad, "k": k, "CMP": CMP, "IMGOFF": 2 * NB, "QC": QC},
                "grid": (b, 1),
                "cluster": 1,
                "block": tpl[0],
                "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
                "pk16": pk16,
                "ws": False,
            }
    if (
        plan["kernel"] == "main"
        and b <= 148
        and (1024 * 4 * 4 + 1024 >= n)
        and (n >> 2 > 1152)
        and (not (b <= 2 and k > BLKC and (2 * BLKC * 8 + BLKC >= n)))
    ):
        n4q = n >> 2
        CMP = n if n < 2560 else 2560
        QC = QUADC
        CURE = not (n < 2 * k and b > 148)
        DEGE = n <= 3 * k or n <= 4 * k + 64
        if DEGE and CMP < n:
            CMP = n
        kpt = 1 if k <= 1024 else 2
        vptx = 4
        while vptx > 1 and (1024 * (vptx // 2) >= n4q or 1024 * (vptx // 2) * 4 + 1024 >= n):
            vptx //= 2
        tpl = (1024, vptx, 1, kpt, CURE, DEGE, False, 1024)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {"n": n, "npad": npad, "k": k, "CMP": CMP, "IMGOFF": 2 * NB, "QC": QC},
            "grid": (b, 1),
            "cluster": 1,
            "block": tpl[0],
            "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
            "pk16": pk16,
            "ws": False,
        }
    if plan["kernel"] == "main" and plan["tpl"][5] and (b <= 32) and (k <= 2 * BLKC):
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while amax << 1 <= av and amax < 16:
            amax <<= 1
        vsel = 0
        cs = 0
        g4_skip = False
        if amax >= 2 and (not g4_skip):
            for v in (1, 2):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:
                    continue
                if c == 16 and (b < 2 or b > 8 or v < 2 or (k < 1024)):
                    continue
                if c >= 16 and n >= 131072 and (b >= 8):
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": (3 * NB + 2 * CMPC) * 4,
                "ws": False,
            }
    if plan["kernel"] in ("main", "clus"):
        plan = _bf16_halve_u(plan)
    elif plan["kernel"] == "reg_clus":
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while amax << 1 <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            for v in (1, 2, 4):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            plan["tpl"] = (BLKC, vsel, cs)
            plan["grid"] = (cs, b)
            plan["cluster"] = cs
        else:
            plan = _bf16_halve_u(plan)
    elif plan["kernel"] in ("reg", "regimg"):
        tpl = list(plan["tpl"])
        n4r = n >> 2
        while tpl[1] > 1 and (
            tpl[0] * (tpl[1] // 2) >= n4r or tpl[0] * (tpl[1] // 2) * 4 + tpl[0] >= n
        ):
            tpl[1] //= 2
        target_nbh = 512 if tpl[0] == 512 else 1024
        if tpl[7] > target_nbh:
            tpl[7] = target_nbh
        if b > 296 and k <= 1024 and (tpl[0] == 512) and (tpl[2] == 4) and (tpl[1] <= 2):
            tpl[0] = 256
            tpl[1] *= 2
            tpl[2] = 8
            if tpl[1] <= 2 and k > 512:
                tpl[7] = 256
            plan["block"] = 256
        plan["tpl"] = tuple(tpl)
        if pk16 and plan["kernel"] == "reg":
            plan["smem"] = (tpl[7] + plan["rt"]["CMP"]) * 4
            plan["pk16"] = True
        else:
            plan["smem"] = (tpl[7] + 2 * plan["rt"]["CMP"]) * 4
    return plan


def _route_streaming_bf16(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """bf16 twin of route_streaming (see _route_bf16).

    Split-aware vector-width pick: the fp32 table derives the split count R so
    each CTA's chunk (~n/R) fills one fp32 tile of BLK*U*4 elements. When
    U == 1 the 16-byte bf16 tile cannot shrink with U and spans BLK*8 -- a
    ~BLK*4 chunk then idles half the threads and halves the outstanding-load
    parallelism exactly in the latency-bound deep-split regime. Route those
    plans to the 8-byte-vector engine (fp32 tile geometry, all threads
    active); keep the 16-byte engine when the chunk actually fills >= 3/4 of
    the wider tile (fewer load instructions at full thread activity)."""
    plan = route_streaming(b, n, npad, k, force_main=force_main)
    if plan["kernel"] in ("main", "clus"):
        tpl = plan["tpl"]
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (int(tpl[0]) == 1024)
            and (int(tpl[1]) == 1)
            and (n >= 65536)
            and (k <= 2048)
        ):
            r512 = ((n >> 2) + 511) // 512
            kpt = 1 if k <= 512 else 2 if k <= 1024 else 4
            g1_ok = k < 1024 or _degen_gate_ok(int(tpl[0]), int(tpl[4]), 512, kpt)
            if r512 > int(plan["rt"]["R"]) and b * r512 <= 148 and g1_ok:
                tpl = (512, 1, 1, tpl[3], kpt, True, tpl[6])
                plan["tpl"] = tpl
                plan["grid"] = (r512, b)
                plan["block"] = 512
                plan["rt"]["R"] = r512
                plan["rt"]["Q"] = ((n >> 2) + r512 - 1) // r512
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (b <= 32)
            and (k <= 1024)
            and (int(tpl[1]) == 1)
            and ("rt" in plan)
            and (int(plan["rt"].get("R", 1)) > 1)
            and (b * int(plan["rt"].get("R", 1)) >= 148)
            and ((n + int(plan["rt"]["R"]) - 1) // int(plan["rt"]["R"]) < 6144)
        ):
            r1_8 = max(148 // b, 1)
            r2_8 = max(((n >> 3) + 1023) // 1024, 1)
            r_n8 = max(min(r1_8, r2_8), 1)
            if r_n8 > 1 and r_n8 != int(plan["rt"]["R"]):
                q8 = ((n >> 3) + r_n8 - 1) // r_n8
                per8 = q8 >> 10
                u8 = 8 if per8 >= 8 else 4 if per8 >= 4 else 2 if per8 >= 2 else 1
                if u8 > 4:
                    u8 = 4
                kpt8 = 1 if k <= 1024 else 2 if k <= 2048 else 4 if k <= 4096 else 8
                tshg8 = b > 15 and k <= 1024 and (n >> 2 <= 32768)
                plan["tpl"] = (1024, u8, 1, tpl[3], kpt8, True, tshg8)
                plan["rt"]["R"] = r_n8
                plan["rt"]["Q"] = ((n >> 2) + r_n8 - 1) // r_n8
                plan["grid"] = (r_n8, b)
                return plan
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (int(tpl[1]) == 1)
            and (n >= 16384)
            and ("rt" in plan)
            and (int(plan["rt"].get("R", 1)) > 1)
            and (b * int(plan["rt"].get("R", 1)) < 148)
        ):
            r_const = int(plan["rt"]["R"])
            chunk = (n + r_const - 1) // r_const
            if chunk < int(tpl[0]) * 8 * 3 // 4:
                plan["vec4"] = True
                return plan
        if plan["kernel"] == "main" and (not tpl[5]) and (int(tpl[1]) == 4):
            blk = int(tpl[0])
            rolls_full = -(-n // (blk * 4 * 8))
            rolls_half = -(-n // (blk * 2 * 8))
            if rolls_full < rolls_half:
                return plan
        if plan["kernel"] == "main" and tpl[5] and (int(tpl[1]) == 4) and ("rt" in plan):
            r_c = int(plan["rt"].get("R", 1))
            chunk = (n + r_c - 1) // r_c
            rolls_full_s = -(-chunk // (int(tpl[0]) * 4 * 8))
            rolls_half_s = -(-chunk // (int(tpl[0]) * 2 * 8))
            if rolls_full_s < rolls_half_s:
                return plan
        plan = _bf16_halve_u(plan)
        tpl = plan["tpl"]
        if plan["kernel"] == "main" and tpl[5] and (int(tpl[0]) == 1024) and (int(tpl[1]) < 4):
            r_const = int(plan["rt"]["R"])
            r_pow2 = 1 << r_const.bit_length() - 1
            u_const = int(tpl[1])
            u_pow2 = min(2 * u_const, 4)
            chunk_now = (n + r_const - 1) // r_const
            chunk_pow2 = (n + r_pow2 - 1) // r_pow2
            rolls_now = (chunk_now + 1024 * u_const * 8 - 1) // (1024 * u_const * 8)
            rolls_pow2 = (chunk_pow2 + 1024 * u_pow2 * 8 - 1) // (1024 * u_pow2 * 8)
            if r_pow2 < r_const and b * r_pow2 >= 128 and (rolls_pow2 < rolls_now):
                plan["tpl"] = (tpl[0], u_pow2) + tuple(tpl[2:])
                plan["rt"]["R"] = r_pow2
                plan["rt"]["Q"] = ((n >> 2) + r_pow2 - 1) // r_pow2
                plan["grid"] = (r_pow2, b)
    return plan


# Dtype-isolated launch caches preserve the original FP32 key and hot lookup.
# BF16 additionally keys by device because its cold compilation uses that device.
_VARLEN_CACHE = {}
_VARLEN_CACHE_BF16: dict[tuple[int, ...], tuple] = {}
_BF16_COMPILE_LOCK = threading.RLock()

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


# FP32 register-resident prefill windows share the decode kernel template.
_PREFILL_REG_CACHE = {}


def _prefill_reg_route(
    rows: int, k: int, n_hint: int, num_sms: int = 148, sm_version: int = 100
) -> dict | None:
    """Pure SM100 plan using a trusted maximum window length in compressed columns."""
    if sm_version != 100 or k not in (512, 1024, 2048) or n_hint <= k or n_hint > 8192:
        return None
    if n_hint <= 2048:
        blk, vpt, minb = 512, 1, 4
    elif n_hint <= 4096:
        blk, vpt, minb = 512, 2, 4
    elif k == 2048:
        blk, vpt, minb = 512, 4, 2
    else:
        return None
    n = blk * vpt * 4
    cmp_ = min(n, 2560)
    cure = not (n < 2 * k and rows > num_sms)
    dege = n <= 4 * k + 64
    if dege:
        cmp_ = n
    nbsel = 2 * NB if n // 4 > 512 and rows <= num_sms else NB
    kpt = 1 if dege or k <= blk else 2 if k <= 2 * blk else 4
    tpl = (blk, vpt, minb, kpt, cure, dege, False, nbsel)
    # Retain the existing QC=QUADC for the window specialization.
    return {
        "kernel": "reg",
        "tpl": tpl,
        "rt": {"CMP": cmp_, "QC": QUADC},
        "smem": (nbsel + 2 * cmp_) * 4,
        "capacity": n,
    }


def _prefill_reg_key(plan: dict, k: int, device_index: int) -> tuple:
    """Key every constexpr/resource dimension and isolate per-device launchers."""
    return (device_index, tuple(plan["tpl"]), k, plan["rt"]["CMP"], plan["rt"]["QC"], plan["smem"])


def _prefill_reg_launcher(plan: dict, k: int, device_index: int) -> tuple:
    """Compile a window-local FP32 register specialization with TRT start/end ABI."""
    key = _prefill_reg_key(plan, k, device_index)
    hit = _PREFILL_REG_CACHE.get(key)
    if hit is not None:
        return hit
    dev = _device()
    with torch.cuda.device(device_index):
        fn = dev.get_compiled__reg(tuple(plan["tpl"]), varlen=True, hint_free=True, prefill=True)
    lc = (fn, (plan["rt"]["CMP"], plan["rt"]["QC"], dev.STATIC_BYTES + plan["smem"]))
    _PREFILL_REG_CACHE[key] = lc
    return lc


def _prefill_window_bound(max_row_len: int | None, width: int) -> int:
    """Share the existing host-bound clamping across both prefill entry points."""
    if max_row_len is None:
        return max(width, 1)
    return max(min(_index(max_row_len), width), 1)


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
    fn = dev.get_compiled(tpl[:6] + (False,) + (1, 0, 1), hint_free=True, prefill=True)
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


def _varlen_launcher(
    num_rows: int,
    npad: int,
    k: int,
    n_env: int,
    next_n: int,
    cr: int,
    num_sms: int = 148,
    sm_version: int = 100,
    *,
    dtype: torch.dtype = torch.float32,
    device_index: int | None = None,
) -> tuple:
    """Build a capture-stable launcher from the shared dtype kernel templates.

    FP32 keeps its original routes, tuning scalars and cache key. BF16 uses
    the same family selection and ABI assembly with its packed-load routes,
    refinement parameters and near-K complement specialization. All choices
    depend only on the capture-stable geometry and dtype.
    """
    if dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"decode launcher requires float32 or bfloat16, got {dtype}")
    bf16 = dtype is torch.bfloat16
    profile = _pack_device_profile(num_sms, sm_version)
    if bf16:
        if device_index is None:
            raise RuntimeError("BF16 launcher requires a device index")
        key = (num_rows, npad, k, n_env, next_n, cr, device_index, profile)
        cache = _VARLEN_CACHE_BF16
    else:
        key = (num_rows, npad, k, n_env, next_n, cr, profile)
        cache = _VARLEN_CACHE
    hit = cache.get(key)
    if hit is not None:
        return hit
    # The device envelope never exceeds the physical row stride. The router
    # needs a k+1 floor to select a family even when every row is short.
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    dtype_options = {"dtype": "bf16"} if bf16 else {}
    if bf16 and k in (512, 1024, 2048) and 0 < n_kernel - k <= 3:
        fn = dev.get_compiled_complement(k, n_kernel, next_n, cr_shift, **dtype_options)
        lc = ("complement", fn, n_kernel)
        cache[key] = lc
        return lc
    route_fn = _route_bf16 if bf16 else route
    plan_free = route_fn(num_rows, n_route, npad, k, num_sms, sm_version)
    if plan_free["kernel"] == "reg_clus":
        cluster_options = dict(dtype_options)
        if bf16:
            cluster_options["nbh"] = (
                512 if k > 1024 or (k in (512, 1024) and n_env >= 65536) else 1024
            )
            cluster_options["quadc"] = (
                96 if k > 1024 and num_rows > 1 and plan_free["tpl"][2] >= 8 else 384
            )
            if (
                num_rows == 1
                and next_n == 1
                and cr == 1
                and k == 2048
                and 65536 <= n_kernel <= 132096
                and tuple(plan_free["tpl"]) in ((1024, 1, 8), (1024, 2, 8))
            ):
                cluster_options["hybrid"] = True
                cluster_options["oneq_enabled"] = True
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            **cluster_options,
        )
        lc = ("reg_clus", fn, n_kernel)
    elif plan_free["kernel"] in ("reg", "regimg"):
        reg_options = dict(dtype_options)
        if bf16:
            reg_options["pk16"] = bool(plan_free.get("pk16", False))
            if k == 2048 and num_rows <= 148 and n_kernel > 2048:
                reg_options["binproof"] = True
                if plan_free["kernel"] == "reg" and plan_free["tpl"][4]:
                    tpl = list(plan_free["tpl"])
                    old_bins = tpl[7]
                    if old_bins < 1024:
                        tpl[7] = 1024
                        plan_free["tpl"] = tuple(tpl)
                        plan_free["smem"] += 4 * (1024 - old_bins)
            if n_kernel <= 2048:
                reg_options["packed_prefetch"] = False
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            **reg_options,
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (n_kernel, rt_f["CMP"], rt_f["QC"], dev.STATIC_BYTES + plan_free["smem"]),
        )
    elif plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            **dtype_options,
        )
        lc = ("clus", fn, (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0))
    else:
        streaming_fn = _route_streaming_bf16 if bf16 else route_streaming
        plan = streaming_fn(num_rows, n_route, npad, k, force_main=True)
        tpl = tuple(plan["tpl"])
        rt = plan["rt"]
        r_const = rt["R"]
        main_options = dict(dtype_options)
        if bf16:
            if plan.get("vec4"):
                main_options["vector_elems"] = 4
            else:
                main_options["v16"] = not tpl[5] and int(tpl[0]) < 512 and npad <= 65536
                main_options["dense"] = k == 2048
        # TSHG is dead under varlen; normalize it out of the compile key.
        fn = dev.get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const),
            hint_free=True,
            **main_options,
        )
        big = num_rows * r_const <= 148
        if bf16:
            if r_const > 2 and k > 1024:
                split_aim = 13 * k // 8 if num_rows > 8 else 7 * k // 4
                split_sfac = 48
            else:
                split_aim = 2 * k if k > 1024 or r_const == 2 else 7 * k // 2
                split_sfac = 32 if r_const == 2 else 48 if k > 1024 else 16
            unsplit_aim = 3 * k // 2 if k >= 1024 and (k == 1024 or n_env >= 131072) else 2 * k
            amin = 3 * k if r_const == 2 else split_aim
            tsh_en = 1 if tpl[5] else 0
        else:
            split_aim = 2 * k
            split_sfac = 32 if r_const == 2 else 48 if k > 1024 else 16
            unsplit_aim = 4 * k if k >= 1024 else 2 * k
            amin = 3 * k if r_const == 2 else 7 * k // 2
            tsh_en = 1 if tpl[5] and k <= 1024 else 0
        aim_base = (
            (unsplit_aim if r_const == 1 else split_aim)
            if big
            else 11 * k // 8
            if k >= 1024
            else 3 * k // 2
        )
        sfac = split_sfac if r_const > 1 else 64 if k >= 1024 else 32
        sd_en = 1 if k > 1024 and not big else 0
        pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
        tail = (aim_base, sfac, amin, sd_en, tsh_en)
        lc = ("main", fn, pre, tail)
    cache[key] = lc
    return lc


def route_bands(
    b: int,
    npad: int,
    k: int,
    n_lo: int | None = None,
    n_hi: int | None = None,
    num_sms: int = 148,
    sm_version: int = 100,
) -> list[tuple[int, int, dict[str, object]]]:
    """Enumerate maximal n-intervals on which route_static is constant.
    Dense O(n_hi - n_lo) scan of the pure host dispatch — an offline /
    engine-init tool (seconds for the 262144-token envelope), NOT a hot
    path. Returns [(n_lo, n_hi, static_plan), ...]."""
    lo = k + 1 if n_lo is None else max(n_lo, k + 1)
    hi = npad if n_hi is None else min(n_hi, npad)
    bands = []
    cur_key, cur_lo, cur_plan = None, lo, None
    for n in range(lo, hi + 1):
        st = route_static(b, n, npad, k, num_sms, sm_version)
        key = repr(st)
        if key != cur_key:
            if cur_key is not None:
                bands.append((cur_lo, n - 1, cur_plan))
            cur_key, cur_lo, cur_plan = key, n, st
    if cur_key is not None:
        bands.append((cur_lo, hi, cur_plan))
    return bands


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
    CUDA-ness of the tensor (run() resolves the default workspace before the
    input checks, so a CPU logits tensor dies here with "device index out of
    range: -1").

Concurrent STREAMS on one device that may both take the multi-CTA SPLIT path
must pass their own workspace via run_ws().

Size: workspace_bytes() = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B.

Kernel-facing view: the compiled main-family signature takes the workspace
as a 1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16
-- torch caching-allocator bases are 256B-aligned so the default slab always
satisfies it).  `kernel_view()` reproduces raw `workspace.data_ptr()`
semantics for arbitrary user tensors by aliasing the underlying storage at
the tensor's byte offset.
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


def workspace_bytes() -> int:
    """Workspace bytes required by the multi-CTA SPLIT path."""
    return WS_BYTES


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


def validate_run_ws(workspace: torch.Tensor, logits: torch.Tensor) -> None:
    """run_ws() workspace hardening, in a fixed predicate order:
    CUDA + same device as logits; numel*element_size >= workspace_bytes();
    base 16-byte aligned (the DSL workspace fake declares assumed_align=16)."""
    if not (workspace.is_cuda and workspace.get_device() == logits.get_device()):
        raise RuntimeError("workspace must be a CUDA tensor on the same device")
    if workspace.numel() * workspace.element_size() < WS_BYTES:
        raise RuntimeError(f"workspace too small: need {WS_BYTES} bytes")
    if workspace.data_ptr() & 15:
        raise RuntimeError("workspace must be 16-byte aligned")


def kernel_view(workspace: torch.Tensor) -> torch.Tensor:
    """Raw-pointer view of a user workspace tensor: alias the first WS_BYTES
    bytes at the tensor's data_ptr() as int32[WS_BYTES/4], ignoring
    dtype/shape.

    NOTE: the DSL-side fake tensor declares assumed_align=16, matching the
    validate_run_ws base-alignment check, so misaligned workspaces fail on
    the host with a clear message instead of at DSL conversion."""
    if (
        workspace.dtype is torch.int32
        and workspace.dim() == 1
        and workspace.is_contiguous()
        and workspace.storage_offset() == 0
        and workspace.numel() == WS_BYTES // 4
    ):
        return workspace  # already the canonical view
    off_bytes = workspace.storage_offset() * workspace.element_size()
    if off_bytes & 3:
        # unreachable past the 8B-alignment check for allocator-backed
        # storages; kept as a hard error rather than silent misalias.
        raise RuntimeError("workspace storage offset must be 4-byte aligned")
    t = torch.empty(0, dtype=torch.int32, device=workspace.device)
    t.set_(workspace.untyped_storage(), off_bytes // 4, (WS_BYTES // 4,))
    return t


def _reset_for_tests() -> None:
    """Drop cached slabs (tests only; NOT part of the C contract)."""
    with _mu:
        _ws_keep.clear()


# ===========================================================================
# ==== operator entry =======================================================
# ===========================================================================
"""Operator entry: input hardening, dispatch, and bind-once launch cache.

Hardening checks run in a fixed order with fixed predicates:
  1. all three tensors CUDA
  2. dtypes: logits f32, pre_idx i32, indices i32
  3. all 2-D
  4. all contiguous
  5. n_valid unwrap: python-int fast path (strict integral cast); Tensor
     path checks torch.cuda.is_current_stream_capturing() FIRST and fails
     loudly, else .item() (the D2H sync)
  6. b/npad from logits, k = pre_idx.size(1)
  7. b == 0 -> early no-op
  8. npad % 4 == 0 (float4 row loads)
  9. logits base 16-byte aligned
 10. pre_idx/indices batch dims match
 11. indices width >= k
 12. n_valid >= 0
 13. n = min(nv, npad) clamped in unbounded ints BEFORE any narrowing

Dispatch: route(b, n, npad, k, num_sms, sm_version) -> compile cache keyed on
(kernel family, constexpr tuple) in the device module -> bind-once launch
cache keyed on shape plus a packed device route profile: caches the compiled
callable + the prebuilt runtime-scalar arg pack as plain Python ints (never pre-wrapped
cutlass.Int32 -- the FFI per-argument cost is paid every call regardless;
pre-binding removes only route()/marshal-prep work).

Error contract: launch failures surface as exceptions WITH
(b, n, npad, k) context.

The device module is imported LAZILY (first shape that routes to it), so a
missing/broken module only fails when actually reached, with (b, n, npad, k)
context.  The per-family compiled ABIs are documented at each launcher
builder in _build_launcher; only the main family takes the workspace.
"""


# shape key (b, n, npad, k, packed device profile) -> (fn, args tuple, needs_ws)
_LAUNCH_CACHE = {}
_DUMMY_KV = {}
_DEVICE_PROFILE = {}
_DEVICE_PROFILE_SHIFT = 16
_DEVICE_NUM_SMS_MASK = (1 << _DEVICE_PROFILE_SHIFT) - 1


def _dummy_kv(dev_index, device):
    """Cached 1-element int32 tensor per device — the dead kv_lens slot of
    the extended gvr_main ABI in legacy (batch-uniform) mode."""
    t = _DUMMY_KV.get(dev_index)
    if t is None:
        t = torch.zeros(1, dtype=_I32, device=device)
        _DUMMY_KV[dev_index] = t
    return t


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


def _device_num_sms(dev_index: int) -> int:
    """Return the cached CUDA runtime-reported SM count."""
    return _unpack_device_profile(_device_profile_key(dev_index))[0]


# ---------------------------------------------------------------------------
# per-family launcher builders (cold path: once per distinct shape key)
# ---------------------------------------------------------------------------
def _build_launcher(b, n, npad, k, num_sms, sm_version=100):
    rd = route(b, n, npad, k, num_sms, sm_version)
    fam = rd["kernel"]
    tpl = tuple(rd["tpl"])
    rt = rd["rt"]
    if fam in ("reg", "regimg"):
        dev = _device()
        raw = dev.get_compiled__reg(tpl)

        # compiled ABI: (logits, pre_idx, kv_lens, out, n, CMP, QC,
        # smem_total) -- kv_lens is the dead varlen slot in batch-uniform
        # mode (cached dummy tensor)
        def fn(lg, pi, o, *a, _raw=raw):
            _raw(lg, pi, _dummy_kv(lg.get_device(), lg.device), o, *a)

        args = (rt["n"], rt["CMP"], rt["QC"], dev.STATIC_BYTES + rd["smem"])
        return (fn, args, False)
    if fam == "main":
        dev = _device()
        raw = dev.get_compiled(tpl)

        # compiled ABI: (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_,
        #                R, SMP, TGT, Q, SS2, TGT2,
        #                kv_lens, aim_base, sfac, amin, sd_en, tsh_en)
        # [SCAP_/CMP_ dead, ABI parity; the trailing varlen block is dead in
        #  legacy mode — a cached dummy kv_lens tensor + five zeros]
        def fn(lg, pi, o, w, *a, _raw=raw):
            _raw(lg, pi, o, w, *a, _dummy_kv(lg.get_device(), lg.device), 0, 0, 0, 0, 0)

        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP_"],
            rt["CMP_"],
            rt["R"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )
        return (fn, args, True)
    if fam == "clus":
        dev = _device()
        # compile key carries the smem-extent scalars (scap/cmp_); compiled
        # ABI: (logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP,
        #       TGT, Q, SS2, TGT2) -- NO workspace; kv_lens is the dead
        # varlen slot in batch-uniform mode (cached dummy tensor)
        fn = dev.get_compiled__clus(tpl, scap=rt["SCAP"], cmp_=rt["CMP"])
        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP"],
            rt["CMP"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )

        def _call(lg, pi, idx, _fn=fn, _args=args):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, *_args)

        return (_call, (), False)
    if fam == "reg_clus":
        dev = _device()
        # compiled ABI: (logits, pre_idx, kv_lens, out, n) -- kv_lens is the
        # dead varlen slot in batch-uniform mode (cached dummy tensor);
        # smem/k derived in-module
        fn = dev.get_compiled__regclus(tpl)
        n_arg = rt["n"]

        def _call(lg, pi, idx, _fn=fn, _n=n_arg):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, _n)

        return (_call, (), False)
    # unreachable: route() only emits the five families above
    raise RuntimeError(f"unknown dispatch family {fam!r}")


# ---------------------------------------------------------------------------
# shared implementation of the batch-uniform entries
# ---------------------------------------------------------------------------
def _run_impl(logits, pre_idx, n_valid, indices, ws, values=None):
    if not (logits.is_cuda and pre_idx.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if logits.dtype is not _F32:
        raise RuntimeError("logits must be float32")
    if pre_idx.dtype is not _I32:
        raise RuntimeError("pre_idx must be int32")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    lsh, psh, ish = logits.shape, pre_idx.shape, indices.shape
    if not (len(lsh) == 2 and len(psh) == 2 and len(ish) == 2):
        raise RuntimeError("logits/pre_idx/indices must be 2-D")
    if not (logits.is_contiguous() and pre_idx.is_contiguous() and indices.is_contiguous()):
        raise RuntimeError("tensors must be contiguous")

    # n_valid unwrap: tensor path = D2H sync, illegal under CUDA graph
    # capture -- fail loudly instead of crashing the capture.
    if isinstance(n_valid, _TENSOR):
        if _is_capturing():
            raise RuntimeError(
                "tensor n_valid requires a D2H sync, illegal under CUDA "
                "graph capture — pass n_valid as a python int"
            )
        nv = int(n_valid.item())
    else:
        # strict integral cast (rejects floats/strings)
        nv = _index(n_valid)

    b, npad = lsh
    k = psh[1]
    if b == 0:  # empty batch: no-op
        return
    if npad & 3:
        raise RuntimeError(f"npad (logits stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError(
            "logits base must be 16-byte aligned (storage-offset views break the float4 row loads)"
        )
    if psh[0] != b or ish[0] != b:
        raise RuntimeError(f"batch dims must match: logits {b} pre_idx {psh[0]} indices {ish[0]}")
    if ish[1] < k:
        raise RuntimeError(f"indices width {ish[1]} < k={k} (k is pre_idx.size(1))")
    if nv < 0:
        raise RuntimeError(f"n_valid must be non-negative, got {nv}")
    # clamp BEFORE any narrowing (python ints are unbounded, so min() is the
    # exact 64-bit clamp)
    n = nv if nv < npad else npad

    # CUDA out-indexing mirror: every kernel derives O = out + row*k --
    # flat PACKED rows, ignoring the actual indices width.  The DSL kernels
    # index out[row, :] with the tensor's own row stride, so a wider
    # `indices` must be re-viewed packed (pure view, no copy; contiguity
    # already checked).
    if ish[1] != k:
        indices = indices.reshape(-1)[: b * k].view(b, k)

    # ---- optional values output (production parity, default OFF) ------------
    # dsa.py allocates the values scratch only for the non-CuTeDSL path, so
    # values stay opt-in. The indices are exact, so a gather epilogue
    # reproduces the in-kernel writeback bit-for-bit; the constexpr in-kernel
    # form rides the CUDA-graph per-row rewrite.
    if values is not None:
        if not values.is_cuda:
            raise RuntimeError("values must be CUDA")
        if values.dtype is not _F32:
            raise RuntimeError("values must be float32")
        vsh = values.shape
        if len(vsh) != 2 or not values.is_contiguous():
            raise RuntimeError("values must be 2-D contiguous")
        if vsh[0] != b:
            raise RuntimeError(f"batch dims must match: logits {b} values {vsh[0]}")
        if vsh[1] < k:
            raise RuntimeError(f"values width {vsh[1]} < k={k}")
        if vsh[1] != k:
            values = values.reshape(-1)[: b * k].view(b, k)

    # ---- n <= k short path (heuristicTopKDecode.cu parity) ------------------
    # Every valid position is in the top-K: emit identity indices and pad the
    # tail with -1 (the production pad convention; downstream treats -1 as
    # invalid). Order is contract-irrelevant — exactness is tie-interchangeable
    # SET semantics. Torch-op path for now; the CUDA-graph-safe per-row rewrite
    # moves this branch in-kernel (it cannot fall back per row inside a graph).
    if n <= k:
        if n > 0:
            indices[:, :n] = torch.arange(n, dtype=_I32, device=indices.device)
            if values is not None:
                values[:, :n] = logits[:, :n]
        if n < k:
            indices[:, n:] = -1
            if values is not None:
                values[:, n:] = torch.finfo(_F32).min  # -FLT_MAX pad
        return

    dev_index = logits.get_device()
    profile = _device_profile_key(dev_index)
    key = (b, n, npad, k, profile)
    lc = _LAUNCH_CACHE.get(key)
    if lc is None:
        num_sms, sm_version = _unpack_device_profile(profile)
        lc = _build_launcher(b, n, npad, k, num_sms, sm_version)
        _LAUNCH_CACHE[key] = lc
    fn, args, needs_ws = lc
    try:
        if needs_ws:
            fn(logits, pre_idx, indices, ws, *args)
        else:
            fn(logits, pre_idx, indices, *args)
    except Exception as e:
        raise RuntimeError(f"gvr_topk launch failed (b={b} n={n} npad={npad} k={k}): {e}") from e
    if values is not None:
        # same epilogue as run_varlen: a (never-expected) negative index
        # degrades to -FLT_MAX instead of a context-poisoning device assert
        idx64 = indices.to(torch.int64)
        values.copy_(logits.gather(1, idx64.clamp_min(0)))
        values.masked_fill_(indices < 0, torch.finfo(_F32).min)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def run(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen`` (per-request
    device kv_lens; this entry assumes one batch-uniform host ``n_valid``,
    which real serving batches do not satisfy).

    Fast 4-arg form.  ``values`` (optional DPS output, default None = OFF)
    mirrors the production values writeback; see _run_impl.
    The default per-device slab workspace is resolved FIRST (a CPU logits
    tensor therefore dies with 'device index out of range').
    Hot path inlines the device check + atomic load + cache hit; the slow
    path allocates under the workspace lock."""
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:  # checked on EVERY call
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_hot.get(d)
    if ws is None:
        ws = default_workspace(logits)
    _run_impl(logits, pre_idx, n_valid, indices, ws, values)


def run_ws(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen(workspace=...)``.

    Explicit-workspace form for multi-stream callers."""
    validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices, kernel_view(workspace), values)


def run_varlen(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Run hint-free self-sampling Top-K with per-request device KV lengths.

    Row semantics (mirror of ``heuristicTopKDecode.cu`` and the in-tree
    ``cute_dsl_gvr_topk_decode`` runner):

      ``num_rows = logits.shape[0]``, ``batch = num_rows // next_n``;
      ``kv_lens`` int32 ``[batch]`` — per-request TOTAL cache length in
      UNCOMPRESSED token space (dsa.py ``metadata.kv_lens_cuda_runtime``,
      not new-token seq_lens); row ``r`` uses
      ``n_r = (kv_lens[r // next_n] - next_n + (r % next_n) + 1) //
      compress_ratio`` valid entries (cr 1 = DSv3.2, 4 = DSv4 Flash/Pro);
      the bracket is derived from the current row itself (register families:
      min/max fold of the first k row values; streaming families do not
      consume a temporal hint on the accept path); ``k`` comes from
      ``indices.shape[1]``;
      per-row ``n_r <= k`` takes the short path (identity + ``-1`` tail).

    The per-row in-kernel engine launches once for the whole batch. Each CTA
    reads its row's kv_len on device and re-derives the sampling ladder (route_dynamic
    formula mirror), so with ``max_seq_len`` given (a capture-stable engine
    constant, e.g. dsa.py's ``indexer_max_seq_len``) the call performs NO
    host reads.  Without ``max_seq_len`` the envelope comes from ONE
    ``kv_lens.max()`` host read (documented sync, refused under capture).

    Both dtypes use the shared kernel templates with dtype-specific tuning. BF16 logits
    require a row stride divisible by eight and a 16-byte aligned base; optional
    values remain FP32. Input conversion, if needed, belongs to the caller.

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
        if logits.dtype is torch.bfloat16:
            return _run_varlen_bf16(
                logits,
                kv_lens,
                indices,
                next_n=next_n,
                compress_ratio=compress_ratio,
                values=values,
                max_seq_len=max_seq_len,
                workspace=workspace,
            )
        raise RuntimeError(f"logits must be float32 or bfloat16 (got {logits.dtype})")
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
        return
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        # multi-stream escape hatch (run_ws parity): concurrent varlen
        # launches on one device must not share the SPLIT publish slab
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)

    # ---- per-row in-kernel engine (gvr_main varlen port) ----------------
    # Full validation battery (the engine bypasses _run_impl — every
    # check the batch-uniform path enforces is replayed here; the
    # batch-dim check is CRITICAL: the kernel grid comes from
    # logits.shape[0], so a short indices/values tensor would be written
    # out of bounds).
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if logits.dtype is not _F32 or indices.dtype is not _I32:
        raise RuntimeError("logits must be float32; indices must be int32")
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
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 3:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    if values is not None:
        if not values.is_cuda or values.dtype is not _F32:
            raise RuntimeError("values must be CUDA float32")
        if (
            len(values.shape) != 2
            or values.shape[0] != num_rows
            or values.shape[1] < k
            or not values.is_contiguous()
        ):
            raise RuntimeError(
                f"values must be contiguous [num_rows={num_rows}, >=k], got {tuple(values.shape)}"
            )
    cshift = 0 if cr == 1 else 2
    if max_seq_len is not None:
        n_env = int(max_seq_len) >> cshift
    else:
        if _is_capturing():
            raise RuntimeError(
                "run_varlen without max_seq_len reads kv_lens.max() on "
                "host — pass max_seq_len (a capture-stable engine "
                "constant) under CUDA graph capture"
            )
        n_env = int(kv_lens.max().item()) >> cshift
        # eager mode: quantize the data-dependent envelope up to the next
        # power of two so a growing decode does not recompile at every
        # R increment (bounded plans, bounded _VARLEN_CACHE)
        n_env = 1 << max(n_env - 1, 1).bit_length()
    n_env = min(max(n_env, 1), npad)
    profile = _device_profile_key(d)
    key = (num_rows, npad, k, n_env, nn, cr, profile)
    lc = _VARLEN_CACHE.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        num_sms, sm_version = _unpack_device_profile(profile)
        lc = _varlen_launcher(num_rows, npad, k, n_env, nn, cr, num_sms, sm_version)
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
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
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
    if vals is not None:
        idx64 = idx.to(torch.int64)
        vals.copy_(lg.gather(1, idx64.clamp_min(0)))
        vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
    return


def _run_varlen_bf16(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """BF16 entry specialization preserving the FP32 hot path in ``run_varlen``.

    Same per-row varlen contract;
    bfloat16 logits read directly by the device code and widened only in
    registers. Optional values are CUDA float32, with the same flattened
    output view and -FLT_MAX padding as FP32 decode. Supplying
    max_seq_len avoids device-to-host reads; use the exact uncompressed
    envelope, not the padded storage width, to preserve dispatch."""
    if logits.dtype is not torch.bfloat16:
        raise RuntimeError(f"logits must be bfloat16 (got {logits.dtype}); fp32 -> run_varlen")
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
        return
    if not logits.is_cuda:
        raise RuntimeError("logits must be CUDA")
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    if kv_lens.device != logits.device or indices.device != logits.device:
        raise RuntimeError("logits, kv_lens and indices must be on the same CUDA device")
    if values is not None and values.device != logits.device:
        raise RuntimeError("values must be on the same CUDA device as logits")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)
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
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 2 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 7:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 8, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    if values is not None:
        if not values.is_cuda or values.dtype is not _F32:
            raise RuntimeError("values must be CUDA float32")
        if (
            len(values.shape) != 2
            or values.shape[0] != num_rows
            or values.shape[1] < k
            or (not values.is_contiguous())
        ):
            raise RuntimeError(
                f"values must be contiguous [num_rows={num_rows}, >=k], got {tuple(values.shape)}"
            )
    cshift = 0 if cr == 1 else 2
    if max_seq_len is not None:
        n_env = int(max_seq_len) >> cshift
    else:
        if _is_capturing():
            raise RuntimeError(
                "run_varlen_bf16 without max_seq_len reads kv_lens.max() on host; "
                "pass max_seq_len (a capture-stable engine constant)"
            )
        n_env = int(kv_lens.max().item()) >> cshift
        n_env = 1 << max(n_env - 1, 1).bit_length()
    n_env = min(max(n_env, 1), npad)
    profile = _device_profile_key(d)
    key = (num_rows, npad, k, n_env, nn, cr, d, profile)
    lc = _VARLEN_CACHE_BF16.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        # Serialize cold compilation across BF16 families. Shared-memory layouts
        # are instance constants; warmed launches never take the lock.
        with _BF16_COMPILE_LOCK, torch.cuda.device(d):
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
                dtype=torch.bfloat16,
                device_index=d,
            )
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
    pre_arg = idx
    if lc[0] in ("reg_clus", "complement"):
        lc[1](lg, pre_arg, kv_lens, idx, lc[2])
    elif lc[0] == "reg":
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    elif lc[0] == "clus":
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    else:
        _, fn, pre, tail = lc
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
    if vals is not None:
        idx64 = idx.to(torch.int64)
        vals.copy_(lg.gather(1, idx64.clamp_min(0)))
        vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
    return


def run_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    max_row_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Hint-free self-sampling Top-K for prefill: row ``r`` selects the Top-K of
    ``logits[r, ks:ke]`` (compressed columns) into the local frame (column - ks)
    with a -1 pad; ``nv <= k`` rows get the identity, as ``indexer_topk_prefill``.
    No device reads, never compiles under capture; trusts 0 <= ks <= ke <= shape[1].

    ``max_row_len`` may be None (streaming Main) or an integer upper
    bound on every ``row_ends - row_starts``. A provided bound enables the
    register-resident window path on supported shapes. The caller must keep
    this bound valid across CUDA graph replays; no device-to-host length read
    is performed. Zero is valid for an all-empty batch. The existing clamping
    to [1, shape[1]] is retained. Omitting it preserves the streaming fallback. Output indices
    remain window-local and all short-row padding is -1.
    """
    if logits.dtype is not _F32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); bf16/fp16 paths "
            "are a follow-up — see the PR roadmap"
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
    if any(t.device != logits.device for t in (row_starts, row_ends, indices)):
        raise RuntimeError("prefill tensors must share the logits device")
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    lg = logits
    if logits.shape[1] != npad:
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if workspace is not None:
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)
    n_env = _prefill_window_bound(max_row_len, logits.shape[1])
    num_sms, sm_version = _unpack_device_profile(_device_profile_key(d))
    n_bucket = _prefill_bucket(n_env)
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        r1 = min(r0 + _PREFILL_ROW_SLAB, num_rows)
        plan = (
            _prefill_reg_route(r1 - r0, k, n_env, num_sms, sm_version)
            if max_row_len is not None
            else None
        )
        if plan is not None:
            lc_reg = _PREFILL_REG_CACHE.get(_prefill_reg_key(plan, k, d))
            if lc_reg is None:
                if _is_capturing():
                    raise RuntimeError("register prefill launcher not warmed before capture")
                lc_reg = _prefill_reg_launcher(plan, k, d)
            fn_reg, args_reg = lc_reg
            fn_reg(
                lg[r0:r1],
                row_starts[r0:r1],
                row_ends[r0:r1],
                indices[r0:r1],
                logits.shape[1],
                *args_reg,
            )
            continue
        tier = _prefill_tier(r1 - r0, n_env, k)
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
    return


def prefill_ready(
    logits: torch.Tensor, indices: torch.Tensor, max_row_len: int | None = None
) -> bool:
    """True iff ``run_prefill(logits, ..., indices, max_row_len)`` would launch
    without compiling — the same (tier, k, envelope bucket) keys it looks up, so
    a caller can route around the engine under CUDA graph capture. Host-only.
    Pass the same ``max_row_len`` as the ``run_prefill`` call (the envelope
    bucket, and with it the tier, is derived from it)."""
    num_rows = logits.shape[0]
    if num_rows == 0:
        return True
    k = indices.shape[1]
    n_env = _prefill_window_bound(max_row_len, logits.shape[1])
    d = logits.get_device()
    num_sms, sm_version = _unpack_device_profile(_device_profile_key(d))
    n_bucket = _prefill_bucket(n_env)
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        rows = min(r0 + _PREFILL_ROW_SLAB, num_rows) - r0
        plan = (
            _prefill_reg_route(rows, k, n_env, num_sms, sm_version)
            if max_row_len is not None
            else None
        )
        if plan is not None:
            if _prefill_reg_key(plan, k, d) not in _PREFILL_REG_CACHE:
                return False
            continue
        tier = _prefill_tier(rows, n_env, k)
        if _prefill_cache_key(tier, k, n_bucket) not in _PREFILL_CACHE:
            return False
    return True


__all__ = [
    "route",
    "route_static",
    "route_dynamic",
    "route_split",
    "route_bands",
    "run",
    "run_ws",
    "run_varlen",
    "run_prefill",
    "prefill_ready",
    "warmup_varlen",
    "warmup_prefill",
    "workspace_bytes",
    "WS_BYTES",
    "default_workspace",
    "validate_run_ws",
    "kernel_view",
]


# --------------------------------------------------------------------------
# warmup: pre-compile the varlen engine for an engine envelope so no live
# request pays the first-touch DSL JIT (mirrors warmup_heuristic_topk_decode
# and warmup_cute_dsl_radix_topk). Idempotent per (device, geometry) key.
# CUDA-graph capture warmup naturally compiles the captured batch sizes;
# this covers the eager/first-touch path (num_rows defaults to (1,)).
_VARLEN_WARMUP_DONE: set = set()
_VARLEN_WARMUP_DONE_BF16: set = set()
_VARLEN_WARMUP_LOCK = threading.Lock()


def warmup_varlen(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int = 1,
    next_n: int = 1,
    num_rows_list: Sequence[int] = (1,),
    row_stride: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """TESTING/INIT ONLY — compile the varlen engine's envelope tuples.

    One tiny real launch per requested ``num_rows`` (compile keys do not
    depend on tensor contents). Uses the current CUDA device. The done-key
    is recorded only after every launch succeeds, so a failed or interrupted
    warmup is retried on the next call instead of short-circuiting to an
    uncompiled engine.

    ``row_stride`` must be the logits row stride the serving producer will
    emit: the launcher key includes it, so a warmup at a different stride
    compiles a variant dispatch never looks up. Callers that know the
    producer layout (e.g. the DSL paged-MQA arena's 256-element rounding)
    must pass it; the 64-element default only matches producers that round
    the same way.

    """
    if dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"decode warmup requires float32 or bfloat16, got {dtype}")
    bf16 = dtype is torch.bfloat16
    dev = torch.cuda.current_device()
    profile = _device_profile_key(dev)
    num_sms, sm_version = _unpack_device_profile(profile)
    if bf16:
        k = operator.index(top_k)
        nn = operator.index(next_n)
        cr = operator.index(compress_ratio)
        envelope = operator.index(max_seq_len)
        if nn < 1 or cr not in (1, 4):
            raise RuntimeError("next_n must be positive and compress_ratio must be 1 or 4")
        if k < 1 or envelope < 0:
            raise RuntimeError("top_k must be positive and max_seq_len must be nonnegative")
        req_rows = sorted({max(operator.index(r) // nn * nn, nn) for r in num_rows_list})
        if not req_rows:
            return
        n_env = max(envelope // cr, 1)
        npad = (n_env + 63) // 64 * 64 if row_stride is None else operator.index(row_stride)
        if npad < n_env or npad % 8:
            raise RuntimeError(f"row_stride must be a multiple of 8 >= {n_env}, got {npad}")
        rows_list = []
        warmup_keys = []
        with _VARLEN_WARMUP_LOCK:
            for rows in req_rows:
                key = (rows, npad, k, n_env, nn, cr, dev, profile)
                if key in _VARLEN_WARMUP_DONE_BF16 and key in _VARLEN_CACHE_BF16 and dev in _ws_hot:
                    continue
                rows_list.append(rows)
                warmup_keys.append(key)
        bands_done = not rows_list
    else:
        nn = max(1, int(next_n))
        # round each request down to a next_n multiple (min next_n) and dedup
        req_rows = sorted({max(int(r) - int(r) % nn, nn) for r in num_rows_list})
        if not req_rows:
            return
        # BAND-AWARE enumeration: the engine compile key depends on the plan's
        # constexpr tuple (+ r_const family axis), NOT on the exact row count, so
        # warming ONE representative row per distinct engine key covers every row
        # count up to the largest request. Representatives are the first row of
        # each band, which keeps the warmup allocation bounded (~a few hundred
        # rows) even when CUDA-graph batch lists reach thousands of rows.
        n_env_c = max(1, int(max_seq_len) // int(compress_ratio))
        npad_c = (n_env_c + 63) // 64 * 64 if row_stride is None else int(row_stride)
        seen_keys = set()
        rows_list = []
        r = nn
        r_max = req_rows[-1]
        while r <= r_max:
            plan_free = route(
                r,
                max(min(n_env_c, npad_c), int(top_k) + 1),
                npad_c,
                int(top_k),
                num_sms,
                sm_version,
            )
            if plan_free["kernel"] == "reg_clus":
                ekey = ("reg_clus", tuple(plan_free["tpl"]))
            elif plan_free["kernel"] in ("reg", "regimg"):
                ekey = ("reg", tuple(plan_free["tpl"]))
            else:
                p = route_streaming(
                    r,
                    max(min(n_env_c, npad_c), int(top_k) + 1),
                    npad_c,
                    int(top_k),
                    force_main=True,
                )
                ekey = ("main", tuple(p["tpl"][:6]), p["rt"]["R"])
            if ekey not in seen_keys:
                seen_keys.add(ekey)
                rows_list.append(r)
            r += nn
        if not rows_list:
            return
        n_env = max(1, int(max_seq_len) // int(compress_ratio))
        if row_stride is None:
            npad = (n_env + 63) // 64 * 64
        else:
            npad = int(row_stride)
            if npad < n_env or npad % 4:
                raise RuntimeError(
                    f"row_stride must be a float4-multiple >= n_env={n_env}, got {row_stride}"
                )
        key = (
            dev,
            int(top_k),
            int(max_seq_len),
            int(compress_ratio),
            nn,
            tuple(rows_list),
            npad,
            profile,
        )
        # The done key covers the GPU band launches only (one per engine compile
        # key). The exact-row launcher population below is keyed by the requested
        # row counts, which the band key does not see, so it always runs: a later
        # call with a new row count inside an already-warmed band must still
        # create that row count's entry, or capture at it raises not-compiled.
        with _VARLEN_WARMUP_LOCK:
            bands_done = key in _VARLEN_WARMUP_DONE
    if not bands_done:
        rows_max = rows_list[-1]
        # Both dtypes share one arena; shorter geometries use contiguous views.
        logits = torch.zeros((rows_max, npad), dtype=dtype, device=dev)
        kv_lens = torch.full((rows_max // nn,), int(max_seq_len), dtype=torch.int32, device=dev)
        out = torch.empty((rows_max, int(top_k)), dtype=torch.int32, device=dev)
        for rows in rows_list:
            batch = rows // nn
            run_varlen(
                logits[:rows],
                kv_lens[:batch],
                out[:rows],
                next_n=nn,
                compress_ratio=int(compress_ratio),
                max_seq_len=int(max_seq_len),
            )
        del logits, kv_lens, out
        torch.cuda.synchronize()
    if bf16:
        with _VARLEN_WARMUP_LOCK:
            _VARLEN_WARMUP_DONE_BF16.update(warmup_keys)
        return
    # Band launches compile every FP32 engine. Populate exact requested rows
    # without additional launches so subsequent capture finds each cache key.
    n_env_l = min(max(int(max_seq_len) >> (0 if int(compress_ratio) == 1 else 2), 1), npad)
    for r in req_rows:
        _varlen_launcher(
            r,
            npad,
            int(top_k),
            n_env_l,
            nn,
            int(compress_ratio),
            num_sms,
            sm_version,
        )
    if not bands_done:
        with _VARLEN_WARMUP_LOCK:
            _VARLEN_WARMUP_DONE.add(key)


_PREFILL_WARMUP_DONE: set = set()
_PREFILL_WARMUP_LOCK = threading.Lock()


def warmup_prefill(
    top_k: int,
    max_cols: int,
    num_rows_list: Sequence[int] = (1, 149, 297),
    row_stride: int | None = None,
) -> None:
    """Compile every prefill engine ``run_prefill`` can request before serving:
    the tier-0 arm per pow2 envelope bucket where a <= 148-row launch keeps it
    (``_prefill_tier`` is evaluated at both edges of every bucket), tiers 1/2
    one launch each, plus each supported register-window plan for the requested
    row-count bands. ``max_cols`` is the compressed max column count;
    idempotent per done-key."""
    dev = torch.cuda.current_device()
    k = int(top_k)
    max_cols = int(max_cols)
    lo = _prefill_bucket(1)
    hi = _prefill_bucket(max_cols)
    buckets = []
    b = lo
    while b <= hi:
        buckets.append(b)
        b <<= 1
    if not buckets:
        buckets = [hi]
    keys = {}  # cache_key -> (tier, bucket, envelope) representative for the launch
    warmup_slab_rows = set()
    for row_count in num_rows_list:
        for row_start in range(0, int(row_count), _PREFILL_ROW_SLAB):
            warmup_slab_rows.add(min(_PREFILL_ROW_SLAB, int(row_count) - row_start))
    for rows in sorted(warmup_slab_rows):
        for bk in buckets:
            for n_env in (bk // 2 + 1, bk):  # the tier can change inside a bucket
                tier = _prefill_tier(int(rows), n_env, k)
                keys.setdefault(_prefill_cache_key(tier, k, bk), (tier, bk, n_env))
    done_key = (dev, k, max_cols, tuple(sorted(int(r) for r in num_rows_list)), row_stride)
    with _PREFILL_WARMUP_LOCK:
        if done_key in _PREFILL_WARMUP_DONE:
            return
    for tier, bk, n_env in keys.values():
        rows = _PREFILL_TIER_ROWS[tier]
        stride = row_stride if row_stride is not None else ((bk + 256 + 255) // 256 * 256)
        if stride < bk or stride % 4:
            stride = (max(stride, bk) + 256 + 255) // 256 * 256
        logits = torch.zeros((rows, stride), dtype=torch.float32, device=dev)
        ks = torch.zeros((rows,), dtype=torch.int32, device=dev)
        ke = torch.full((rows,), n_env, dtype=torch.int32, device=dev)
        out = torch.empty((rows, k), dtype=torch.int32, device=dev)
        _, fn, (scap, cmp_), tail = _prefill_launcher(tier, k, bk)
        ws = kernel_view(default_workspace(logits))
        pre = (0, stride, k, scap, cmp_, 1, 0, 0, 0, 0, 0)
        fn(logits, ke, out, ws, *pre, ks, *tail)
        del logits, ks, ke, out
    num_sms, sm_version = _unpack_device_profile(_device_profile_key(dev))
    reg_plans = {}
    slab_rows = set()
    for row_count in num_rows_list:
        for row_start in range(0, int(row_count), _PREFILL_ROW_SLAB):
            slab_rows.add(min(_PREFILL_ROW_SLAB, int(row_count) - row_start))
    for row_count in sorted(slab_rows):
        for bound in (2048, 4096, 8192):
            hint = min(bound, max_cols)
            plan = _prefill_reg_route(int(row_count), k, hint, num_sms, sm_version)
            if plan is not None:
                key = _prefill_reg_key(plan, k, dev)
                # Only the <= num_sms band affects the row-dependent template
                # options. Launch the smallest representative of that band;
                # large requested slabs must not inflate warmup allocations.
                representative_rows = 1 if row_count <= num_sms else num_sms + 1
                reg_plans.setdefault(key, (plan, representative_rows, hint))
    for plan, rows, hint in reg_plans.values():
        width = plan["capacity"] + 4
        stride = max(width, row_stride or 0)
        stride = (stride + 3) // 4 * 4
        logits = torch.zeros((rows, stride), dtype=torch.float32, device=dev)
        ks = torch.ones((rows,), dtype=torch.int32, device=dev)
        ke = torch.full((rows,), hint + 1, dtype=torch.int32, device=dev)
        out = torch.empty((rows, k), dtype=torch.int32, device=dev)
        fn, args_reg = _prefill_reg_launcher(plan, k, dev)
        fn(logits, ks, ke, out, width, *args_reg)
        del logits, ks, ke, out
    torch.cuda.synchronize()
    with _PREFILL_WARMUP_LOCK:
        _PREFILL_WARMUP_DONE.add(done_key)
