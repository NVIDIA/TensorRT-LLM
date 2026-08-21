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

Companion to ``gvr_topk_decode_self_sampling.py`` (the merged device
module). Three sections, each a rename-only merge of the per-family source
(fork branch ``GVR-selfsampling-CuTeDSL``):

1. dispatch — bit-exact transcription of the CUDA host dispatch, a pure
   function ``route(b, n, npad, k)`` (cross-checked against an independent
   second transcription by a 1,159,168-case boundary+fuzz sweep);
2. workspace — one zero-initialised per-device slab (20,973,568 B) via the
   torch caching allocator, keep-alive + double-checked locking semantics
   mirrored from the CUDA binding;
3. operator entry — ``run(logits, pre_idx, n_valid, indices)`` /
   ``run_ws(..., workspace)`` DPS forms with the CUDA binding's hardening
   battery, BIND-ONCE launch cache keyed on ``(b, n, npad, k)``.

OPERATOR CONTRACT (standalone; not wired into the decode path): ``n_valid``
is one host python int for the whole batch — every row shares the same
valid prefix, in COMPRESSED index space (the caller applies any
``compressRatio`` division). ``pre_idx`` is consumed AS-IS — raw prev-step
top-K indices, uniformly for DSv3.2 / DSv4 Flash / Pro. This deliberately
drops the +1 temporal shift ``heuristicTopKDecode.cu`` applies for cr==1:
hints only steer the sampling ladder (exactness never depends on them), and
on real V3.2 decode captures raw prev-step hints land on MORE of the current
top-K than +1-shifted ones (mean overlap 0.773 vs 0.536 across 15 cells x 14
consecutive step-pairs, the gap widening with ISL), so one offset-free hint
convention serves all three models. The production per-row contract
(per-request ``kv_lens`` read on-device, per-row MTP offsets — sync-free and
CUDA-graph-replay safe with growing KV) is implemented by ``run_varlen``,
which is the entry the opt-in DSA dispatch seam calls. The batch-uniform
``run``/``run_ws`` entries keep the original standalone contract (one
host-side ``n_valid`` for the whole batch), are exercised for unit tests and
benchmarking only, and must not be substituted for the tiered path under
continuous batching, MTP (``next_n > 1``), or CUDA-graph capture.
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
# ==== dispatch (ct_dispatch.py) ============================================
# ===========================================================================
"""Pure-Python transcription of the frozen GVR CUDA dispatch (gvr_topk_launch).

Source of truth: ../src_cuda/kernel.cu (3197 lines).  route(b, n, npad, k) is a
PURE function of its four ints -- no env knobs, no GPU, stdlib only.

Branch map (kernel.cu line citations):
  constants          NB L16, QUADC L21, SNB L170, CMPC L2372, BLKC L2374
  reg-block prologue L2757-2822: wide=(b<=148) L2757; n4=n>>2 L2759;
                     CMP=min(n,2560) L2764; QC=(b>148?1024:QUADC) L2768;
                     CURE L2775; DEGE L2788; DEG widens CMP to n L2791;
                     NBSEL L2820; IMGOFF=NBSEL L2821; smem=(NBSEL+2*CMP)*4 L2822
  LAUNCH_REG2/DEG/REG macros L2823-2847 (KPT ladder 1/2/4; DEG forces KPT=1,
                     CUR=CURE both places); IMGW/smi/IMGE L2852-2854;
                     LAUNCH_REGIMG L2861-2863 -> gvr_topk_reg<...,KPT=1,CUR=true,
                     DEG=false,IMG=true,NBH=2*NB> via launch_regimg L2672-2686
  n4 rungs           n4<=256 L2864; n4<=512 L2865; n4<=1024 wide/img/else L2866-2884
  clustered reg path L2897-2940: gate n4>4096 && n4<=8*BLKC*4 && k<=BLKC L2897;
                     av/amax L2898-2899; two-pass cs=8 co-residency veto
                     (pass==0 && c==8 && b>15 -> skip) L2917-2923; 64-bit product
                     (long long)c*BLKC*v < n4 L2919; smc=(3*NB+2*CMPC)*4 L2926;
                     grid dim3(cs,b) L2666
  wide 4k fallback   n4<=4096 && wide -> LAUNCH_REG(1024,4,1,2*NB) L2945-2947
  streaming R        L2959-2975 (b<=32: R=min(148/b, ((n>>2)+1023)/1024));
                     r11 shallow split b<=74 && n4>=16384 && k<=1024 -> R=2 L2985;
                     cluster clamp R->pow2, useclus, only if 2<=R<=8 && k<=1024 L2994
  big/SCAP/CMP       big=(b*R<=148) L2995; SCAP L3009-3010; CMP L3011
  aim                L3039-3040; sqrt floor r=int(0.5+sqrt(6LL*n)) L3041-3042;
                     SFAC L3072-3073; amin L3079-3080; clamps L3081-3082
  sample geometry    small_dense gate L3091 ((k>1024)&&!big&&n<=SCAP&&n>2*k);
                     PAIR form (sel>>3, half=n4s>>1, SMP*8) L3092-3109;
                     clus QUAD override (sel>>4, quarter=n4s>>2, SMP*16),
                     gated n>SCAP only, L3115-3128
  Q                  Q=(n4s+R-1)/R L3110
  clus launch        smc=SNB*8+(SCAP+4)*8+CMP*8 L3130; U ladder per=Q>>10
                     L3132-3142; CS=R in {2,4,8} L3143-3145; grid dim3(CS,b) L2704
  main launch        smem=(SCAP+4)*((R>1||b<=296)?8:4)+(CMP+1)*8 L3149;
                     KPT ladder 1/2/4/8 L3150-3169; big: per=Q>>10 U ladder,
                     SPLIT=(R>1), grid dim3(R,b) L3173-3185 + L2750;
                     b<=296 -> (512,2,8,false) L3193; else (256,4,8,false) L3194

rt carries the FULL runtime scalar list each kernel receives, in signature
order, always starting with (n, npad, k) -- every launch site passes them
(L2666-2667 reg_clus, L2684-2685 regimg, L2704-2705 clus, L2726-2727 reg,
L2750-2751 main).  [dispatch x-check 2026-08-13: rt previously omitted the
leading n/npad/k; fixed for full-ABI parity with the independent spec
transcription.]

Dead ABI-parity args: gvr_main's 7th/8th params are declared `int SCAP_, int CMP_`
(kernel.cu L381) and are NEVER read by the kernel body -- it recomputes SCPB/CMPB
as constexprs of (BLK, SPLIT, KBIG) (L413-424) that mirror the host formulas
bit-identically.  They are kept in rt under their source names 'SCAP_'/'CMP_'
purely for ABI parity.  gvr_clus's SCAP/CMP (L1798) are LIVE runtime args.
`aim` and `SFAC` are host-side intermediates only (never cross the ABI), so they
do not appear in rt.

C-semantics notes encoded here:
  * every `/` on ints is C truncating division -> Python `//` (all operands
    are non-negative on every reachable path);
  * `sel = (long long)SFAC * n / aim` and the TGT/TGT2 products are 64-bit in C;
    Python ints are exact, so `//` reproduces them;
  * `int r = (int)(0.5 + sqrt((double)(6LL*n)))` truncates toward zero after
    the +0.5 -> `int(0.5 + math.sqrt(float(6*n)))`;
  * `IMGW = (n + 3) & ~3` four-element float4 round-up;
  * the reg-block CMP (possibly widened to n by DEGE) is scoped to the braces
    at L2758-2949; the streaming path re-derives its own CMP.
"""


# ---- constants lifted from kernel.cu ---------------------------------------
NB = 1024  # L16   register-path histogram bins
QUADC = 96  # L21   crossing-bin O(mc^2) rank gate (streaming/reg paths)
SNB = 256  # L170  streaming-path bin count
CMPC = 4096  # L2372 crossing-bin slots per CTA, clustered register path
BLKC = 1024  # L2374 CTA size of the clustered register path


def route(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """Mirror of gvr_topk_launch (kernel.cu L2754-3197). Pure. See module doc."""
    if b < 1:
        raise RuntimeError(f"route requires b >= 1, got {b}")
    wide = b <= 148  # L2757

    # ================= register-resident block (L2758-2949) =================
    n4 = n >> 2  # L2759
    CMP = n if n < 2560 else 2560  # L2764
    QC = 1024 if b > 148 else QUADC  # L2768
    CURE = not (n < 2 * k and b > 148)  # L2775
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)  # L2788
    if DEGE and CMP < n:  # L2791
        CMP = n
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 1024 and not wide)) else NB  # L2820
    IMGOFF = NBSEL  # L2821
    smem_reg = (NBSEL + 2 * CMP) * 4  # L2822

    def _reg(BLK, VPT, MINB, NBH):
        # LAUNCH_REG (L2844-2847): DEG wins, else CUR flag; KPT ladder L2823-2834.
        if DEGE:
            tpl = (BLK, VPT, MINB, 1, CURE, True, False, NBH)  # LAUNCH_DEG L2836-2843
        else:
            kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else 4)
            tpl = (BLK, VPT, MINB, kpt, CURE, False, False, NBH)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # L2726-2727 full ABI
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

    IMGW = (n + 3) & ~3  # L2852
    smi = (NBSEL + (2 * CMP if 2 * CMP > IMGW else IMGW)) * 4  # L2853
    IMGE = wide and (not DEGE) and k <= 1024  # L2854

    if n4 <= 256:  # L2864
        return _reg(256, 1, 8, NB)
    if n4 <= 512:  # L2865
        return _reg(512, 1, 4, NB)
    if n4 <= 1024:  # L2866-2884
        if wide:
            if IMGE:  # LAUNCH_REGIMG(1024,1,2) L2872
                # launch_regimg<1024,1,2,NBV=2*NB,KPTV=1> -> gvr_topk_reg
                # <1024,1,2,1,true,false,true,2048>  (L2672-2686)
                return {
                    "kernel": "regimg",
                    "tpl": (1024, 1, 2, 1, True, False, True, 2 * NB),
                    "rt": {
                        "n": n,
                        "npad": npad,
                        "k": k,  # L2684-2685 full ABI
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
            return _reg(1024, 1, 2, 2 * NB)  # L2872 else-arm
        return _reg(512, 2, 4, NB)  # L2883

    # ---- clustered register-resident path (L2897-2940) ----
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and k <= BLKC:  # L2897
        av = 148 // (b if b > 0 else 1)  # L2898 truncating
        amax = 1  # L2899
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:  # L2901
            # knife5 (layer 9): UNCONDITIONAL cs=8 co-residency veto --
            # the L2w pass-1 rescue is deleted; 512k b>15 falls through to
            # streaming, made retry-safe by TSH-floor staging (S1) and the
            # gvr_clus veto (S2).
            for v in (1, 2, 4):
                c = 1  # 64-bit product
                while c * BLKC * v < n4:
                    c <<= 1
                if c == 8 and b > 15:  # THE VETO
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:  # L2925
            smc = (3 * NB + 2 * CMPC) * 4  # L2926
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only, L2666-2667
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": smc,
                "ws": False,
            }

    if n4 <= 4096 and wide:  # L2945-2947
        return _reg(1024, 4, 1, 2 * NB)

    # ================= streaming / collect path (L2950-3196) =================
    R = 1  # L2959
    if b <= 32:  # L2960-2975
        r1 = 148 // b
        if r1 < 1:
            r1 = 1
        r2 = ((n >> 2) + 1023) // 1024  # L2972
        if r2 < 1:
            r2 = 1
        R = r1 if r1 < r2 else r2
        if R < 1:
            R = 1
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:  # L2985 r11 split
        R = 2

    useclus = False  # L2993-2994
    if 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        # knife5 (layer 8): gvr_clus cs=8 hits the same GPC packing wall as
        # the clustered register path; same veto, same b>15 threshold.
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True

    big = b * R <= 148  # L2995
    SCAP = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)  # L3009-3010
    CMP = (4096 if k > 1024 else 2048) if big else 1024  # L3011

    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )  # L3039-3040
    q = 6 * n  # L3041: 6LL * n
    r = int(0.5 + math.sqrt(float(q)))  # L3041 C cast trunc
    if r > aim:  # L3042
        aim = r
    SFAC = (
        (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    )  # L3072-3073
    amin = 3 * k if R == 2 else (7 * k) // 2  # L3079
    if R > 1 and aim < amin:  # L3080
        aim = amin
    if aim > (SCAP >> 1):  # L3081
        aim = SCAP >> 1
    if aim < k:  # L3082
        aim = k

    n4s = n >> 2  # L3084
    SMP, SS2, TGT, TGT2 = 0, 1, 0, 0  # L3085
    small_dense = (k > 1024) and (not big) and n <= SCAP and n > 2 * k  # L3091
    if (n > SCAP or small_dense) and n4s >= 4:  # L3092: PAIR sample
        sel = SFAC * n // aim  # L3095 64-bit
        if sel < 256:  # L3096
            sel = 256
        if sel > n // 2:  # L3097
            sel = n // 2
        pairs = sel >> 3  # L3098
        if pairs < 1:
            pairs = 1
        half = n4s >> 1  # L3099
        if half < 1:
            half = 1
        if pairs > half:  # L3100
            pairs = half
        SS2 = half // pairs  # L3101
        if SS2 < 1:
            SS2 = 1
        SMP = half // SS2  # L3102
        if SMP < 1:
            SMP = 1
        TGT = (aim * (SMP * 8)) // n  # L3103 64-bit
        if TGT < 1:  # L3104
            TGT = 1
        TGT2 = (k * (SMP * 8)) // n  # L3107 64-bit
        if TGT2 < 1:  # L3108
            TGT2 = 1
    Q = (n4s + R - 1) // R  # L3110

    if useclus:  # L3111-3147
        if n > SCAP and n4s >= 4:  # L3115: QUAD override
            sel = SFAC * n // aim  # L3116
            if sel < 256:
                sel = 256
            if sel > n // 2:
                sel = n // 2
            quads = sel >> 4  # L3119
            if quads < 1:
                quads = 1
            quarter = n4s >> 2  # L3120
            if quarter < 1:
                quarter = 1
            if quads > quarter:  # L3121
                quads = quarter
            SS2 = quarter // quads  # L3122
            if SS2 < 1:
                SS2 = 1
            SMP = quarter // SS2  # L3123
            if SMP < 1:
                SMP = 1
            TGT = (aim * (SMP * 16)) // n  # L3124
            if TGT < 1:
                TGT = 1
            TGT2 = (k * (SMP * 16)) // n  # L3126
            if TGT2 < 1:
                TGT2 = 1
        smc = SNB * 8 + (SCAP + 4) * 8 + CMP * 8  # L3130
        per = Q >> 10  # L3131
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))  # L3134-3141
        CS = 2 if R == 2 else (4 if R == 4 else 8)  # L3143-3145
        return {
            "kernel": "clus",
            "tpl": (1024, U, 1, SNB, CS),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # L2704-2705 ABI (live)
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

    smem_main = (SCAP + 4) * (8 if (R > 1 or b <= 296) else 4) + (CMP + 1) * 8  # L3149

    def _main(BLK, MINB, U, SPLIT):
        # LAUNCH_MAIN KPT ladder 1/2/4/8 (L3150-3169); grid dim3(gx=R, gy=b) L2750.
        kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else (4 if k <= 4 * BLK else 8))
        # knife5 (layer 7) TSH-floor staging gate.  CUDA form: grid-uniform
        # RUNTIME gate gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768 with
        # a dual scan-instantiation branch.  Here: compile-time key -- the
        # ungated variant IS the pre-knife5 kernel; per-launch semantics are
        # identical because the gate is uniform over the grid.
        tshg = bool(SPLIT) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (BLK, U, MINB, SNB, kpt, SPLIT, tshg),
            # SCAP_/CMP_ are DEAD ABI-parity args: gvr_main (L381) never reads
            # them, it uses constexpr SCPB/CMPB (L413-424).  Kept for ABI parity.
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # L2750-2751 full ABI
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

    if big:  # L3173-3185
        per = Q >> 10  # L3174
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, U, R > 1)  # SPLIT iff R>1
    if b <= 296:  # L3193
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)  # L3194


if __name__ == "__main__":
    smoke = [
        # (b, n, npad, k)                          expected family
        (64, 1024, 1024, 512),  # reg   n4<=256 rung (DEG: n<=3k)
        (64, 2048, 2048, 512),  # reg   n4<=512 rung
        (1024, 4096, 4096, 1024),  # reg   n4<=1024, b>148 -> (512,2,4)
        (64, 4096, 4096, 512),  # regimg wide !DEGE k<=1024
        (64, 4096, 4096, 1024),  # reg   wide but DEGE (n<=4k+64)
        (8, 65536, 65536, 1024),  # reg_clus (vsel=2, cs=8; b<=15 no veto)
        (16, 131072, 131072, 512),  # knife5: veto fall-through -> SPLIT slab, tshg=True
        (64, 16384, 16384, 1024),  # reg   wide 4k fallback (1024,4,1)
        (64, 262144, 262144, 1024),  # clus  r11 R=2 shallow cluster split
        (1, 1048576, 1048576, 1024),  # main  deep slab SPLIT R=148
        (20, 262144, 262144, 2048),  # main  k>1024 split (no useclus)
        (512, 131072, 131072, 1024),  # main  b>296 BLK=256
        (256, 6144, 6144, 2048),  # main  small_dense sample gate
        (256, 262144, 262144, 2048),  # main  v32 KBIG-domain, BLK=512 KPT=4
    ]
    for shp in smoke:
        print(shp, "->", route(*shp))


# ---------------------------------------------------------------------------
# two-time-scale dispatch split (per-row varlen / CUDA-graph groundwork)
# ---------------------------------------------------------------------------
# route(b, n, npad, k) factored into
#   route_static(b, n, npad, k)  — everything that must be frozen per launch:
#       family, compile tuple, grid, cluster, block, and the rt scalars that
#       change only at discrete n-thresholds;
#   route_dynamic(static, n)     — the n-continuous scalars a per-row kernel
#       recomputes from its own row length (the device code will mirror these
#       formulas): n, CMP (reg families), the sampling ladder
#       SMP/TGT/SS2/TGT2/Q (streaming families), and the reg-family smem
#       footprint.
# INVARIANT (fuzz-verified): merging route_dynamic back into route_static
# reproduces route() EXACTLY for every n. The capture-time policy of which n
# to freeze the static half at (e.g. max_seq_len) is a later, perf-only
# choice — this split only proves the factorization is lossless.

_DYN_RT = {
    "reg": ("n", "CMP"),
    "regimg": ("n", "CMP"),
    "reg_clus": ("n",),
    "clus": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
    "main": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
}
_DYN_SMEM = ("reg", "regimg")  # smem depends on CMP/IMGW -> recomputed per n


def route_static(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """route() with the n-continuous fields redacted (see _DYN_RT/_DYN_SMEM).
    Constant on maximal n-intervals ("bands"); every redacted field is
    reconstructible from (static, n) by route_dynamic."""
    plan = route(b, n, npad, k)
    st = {key: (dict(val) if isinstance(val, dict) else val) for key, val in plan.items()}
    for f in _DYN_RT[st["kernel"]]:
        st["rt"].pop(f)
    if st["kernel"] in _DYN_SMEM:
        st.pop("smem")
    return st


def route_dynamic(static: dict[str, object], n: int) -> tuple[dict[str, object], int]:
    """Recompute the redacted n-continuous scalars from (static, n).
    Returns (rt_updates, smem). Transcribed independently from route() —
    the factorization fuzz is the equivalence proof, and the device-side
    per-row engine mirrors exactly these formulas."""
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


def route_split(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """route_static + route_dynamic recombined — must equal route() exactly
    (the factorization fuzz in the unit tests asserts this)."""
    st = route_static(b, n, npad, k)
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
    route() itself lands on main/clus this is IDENTICAL to route() (fuzz:
    110,003/110,003 agreement).  force_main additionally skips the clus
    rounding (v1 varlen engine ships the gvr_main port first; the raw
    min(r1, r2) R then matches the CUDA else-branch exactly)."""
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


_VARLEN_CACHE = {}


def _varlen_launcher(num_rows, npad, k, n_env, next_n, cr):
    """Capture-time varlen plan + compiled launcher (v1 = the gvr_main port,
    universally correct across the envelope; specialist tiers below).  Every
    choice here is a function of capture-stable quantities only — mirroring
    the in-tree runner's pick_tuning(graph_capture=...) discipline."""
    key = (num_rows, npad, k, n_env, next_n, cr)
    hit = _VARLEN_CACHE.get(key)
    if hit is not None:
        return hit
    n_eff = max(min(n_env, npad), k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    # ---- route() parity, family tier 1: clustered register-resident --------
    # Admit reg_clus exactly where the free route picks it (886-real-capture
    # grid: this family recovers the deep-layer distribution tail and the
    # large-N small-rows band; its whole admission window n4 <= 32768 fits
    # capture-frozen envelopes). The choice is a pure function of this cache
    # key, so CUDA-graph replay safety is unchanged; per-row n / short-row
    # handling lives in-kernel (GvrMainKernel varlen discipline).
    plan_free = route(num_rows, n_eff, npad, k)
    if plan_free["kernel"] == "reg_clus":
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]), varlen=True, next_n=next_n, cr_shift=cr_shift
        )
        lc = ("reg_clus", fn, n_eff)
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 2: register-resident (+img flavor) ----
    # Same admission rule as tier 1: exactly where the free route picks
    # reg/regimg (the whole small/mid-N band across all row counts). CMP/QC/
    # smem are envelope-derived launch constants -- in-kernel they are pure
    # capacity clamps (CMP), a fast-path threshold (QC) and the launch smem
    # size, all safe upper bounds for every per-row n <= envelope; per-row n
    # / short-row handling lives in-kernel (GvrRegClusKernel discipline).
    if plan_free["kernel"] in ("reg", "regimg"):
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]), varlen=True, next_n=next_n, cr_shift=cr_shift
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (rt_f["n"], rt_f["CMP"], rt_f["QC"], dev.STATIC_BYTES + plan_free["smem"]),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 3: cluster split (clus) ---------------
    # Same admission rule: exactly where the free route picks clus (the
    # large-N mid-rows band). SCAP/CMP are launch-stable (pure functions of
    # rows/CS/k — never of n) so the envelope values are the per-row values;
    # the sampling-ladder scalars (SMP/TGT/Q/SS2/TGT2) are dead launch slots,
    # re-derived per row in-kernel by the route_dynamic clus mirror
    # (GvrMainKernel discipline). Per-row n / short-row handling in-kernel.
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
        )
        lc = (
            "clus",
            fn,
            (n_eff, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    plan = route_streaming(num_rows, n_eff, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])  # (BLK, U, MINB, SNB, KPT, SPLIT, TSHG)
    rt = plan["rt"]
    r_const = rt["R"]
    # TSHG (tpl[6]) is dead under varlen (the ctor compiles the TSH
    # machinery in whenever SPLIT); normalize it out of the compile key so
    # row counts differing only in that slot share one engine
    fn = dev.get_compiled(tpl[:6] + (False,) + (next_n, cr_shift, r_const))
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
    # TSH-floor staging: gate on SPLIT and K only. The old num_rows > 15
    # condition stranded small batches (rows <= 15) in SPLIT-main without
    # the staged floor — a distribution-dependent 6x tail on real deep-layer
    # captures (v4_pro_512k L46/L52, n4 = 32768, rows 1-8: 142-151 us vs
    # 25 us with staging; healthy layers and n4 > 32768 rows unaffected —
    # the kernel gates TSH per row at runtime anyway).
    tsh_en = 1 if (tpl[5] and k <= 1024) else 0
    pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE[key] = lc
    return lc


def route_bands(
    b: int, npad: int, k: int, n_lo: int | None = None, n_hi: int | None = None
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
        st = route_static(b, n, npad, k)
        key = repr(st)
        if key != cur_key:
            if cur_key is not None:
                bands.append((cur_lo, n - 1, cur_plan))
            cur_key, cur_lo, cur_plan = key, n, st
    if cur_key is not None:
        bands.append((cur_lo, hi, cur_plan))
    return bands


# ===========================================================================
# ==== workspace (ct_workspace.py) ==========================================
# ===========================================================================
"""op46 workspace mirror of src_cuda/main.cpp B2 (L15-37) + run_ws checks
(L107-114) and kernel.h workspace_bytes contract.

B2 semantics mirrored exactly:
  * ONE zero-initialised slab workspace per device, lazily allocated through
    the torch caching allocator (main.cpp:32-33 `at::zeros(..., kByte)`);
  * keep-alive store (`ws_keep[GVR_MAX_DEV]`) -> module dict `_ws_keep`
    (tensor refcount = keep-alive, same as the C static array);
  * double-checked locking: lock-free hot-path load (a GIL-atomic dict get
    plays the `std::memory_order_acquire` load, main.cpp:26-27), slow path
    re-checks under a mutex (main.cpp:28-31);
  * device index bounds `0 <= d < GVR_MAX_DEV` (main.cpp:24-25) -- checked
    BEFORE the CUDA-ness of the tensor, exactly like the C binding (run()
    resolves the default workspace before run_impl's B1 checks, so a CPU
    logits tensor dies here with "device index out of range: -1").

Concurrent STREAMS on one device that may both take the multi-CTA SPLIT path
must pass their own workspace via run_ws() (main.cpp:16-17).

Size: gvr_topk_workspace_bytes() = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B (kernel.cu L44-46).

Kernel-facing view: ct_main's compiled signature takes the workspace as a
1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16 --
torch caching-allocator bases are 256B-aligned so the default slab always
satisfies it).  `kernel_view()` reproduces the C binding's raw
`workspace.data_ptr()` semantics for arbitrary user tensors by aliasing the
underlying storage at the tensor's byte offset.
"""


GVR_MAX_DEV = 64  # kernel.cu L19 / main.cpp:19
_MAXC = 160  # kernel.cu L17
_GCAP = 16384  # kernel.cu L18
_GVR_WS_BUF_OFF = 2048  # kernel.cu L43
WS_BYTES = _GVR_WS_BUF_OFF + _MAXC * _GCAP * 8  # 20,973,568 (kernel.cu L44-46)
assert WS_BYTES == 20_973_568

_mu = threading.Lock()  # main.cpp:28 slow-path mutex
_ws_keep = {}  # device index -> keep-alive int32 view


def workspace_bytes() -> int:
    """kernel.h:12 gvr_topk_workspace_bytes()."""
    return WS_BYTES


def default_workspace(ref: torch.Tensor) -> torch.Tensor:
    """main.cpp:23-37 default_workspace(ref) -> per-device cached slab.

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
        # lazy zeros via the torch caching allocator (at::zeros kByte,
        # main.cpp:32-33), viewed int32 for the DSL launch signature.
        buf = torch.zeros(WS_BYTES, dtype=torch.uint8, device=ref.device)
        ws = buf.view(torch.int32)
        _ws_keep[d] = ws  # keep-alive (ws_keep[d] = tensor)
        return ws


def validate_run_ws(workspace: torch.Tensor, logits: torch.Tensor) -> None:
    """main.cpp:107-114 run_ws() workspace hardening, same predicate order:
    CUDA + same device as logits; numel*element_size >= workspace_bytes();
    base 8-byte aligned."""
    if not (workspace.is_cuda and workspace.get_device() == logits.get_device()):
        raise RuntimeError("workspace must be a CUDA tensor on the same device")
    if workspace.numel() * workspace.element_size() < WS_BYTES:
        raise RuntimeError(f"workspace too small: need {WS_BYTES} bytes")
    if workspace.data_ptr() & 7:
        raise RuntimeError("workspace must be 8-byte aligned")


def kernel_view(workspace: torch.Tensor) -> torch.Tensor:
    """Raw-pointer semantics of the C binding (main.cpp:115 passes
    workspace.data_ptr() and nothing else): alias the first WS_BYTES bytes at
    the tensor's data_ptr() as int32[WS_BYTES/4], ignoring dtype/shape.

    NOTE: the DSL-side fake tensor declares assumed_align=16; a workspace at
    8-but-not-16-byte alignment passes the C-contract check above but is
    rejected by the DSL at conversion -- surfaced as a launch failure with
    shape context by ct_op (documented in notes/ct_op_NOTES.md)."""
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
# ==== operator entry (ct_op.py) ============================================
# ===========================================================================
"""op46 operator entry: CuTeDSL mirror of src_cuda/main.cpp run()/run_ws()/
workspace_bytes() (spec section 1).

B1 hardening checks run in the SAME ORDER with the SAME PREDICATES as
main.cpp:43-88 (run_impl):
  1. all three tensors CUDA (main.cpp:43-44)
  2. dtypes: logits f32, pre_idx i32, indices i32 (45-47)
  3. all 2-D (48-49)
  4. all contiguous (50-51)
  5. n_valid unwrap (57-67): python-int fast path (strict integral cast, like
     pybind cast<int64_t>); Tensor path checks
     torch.cuda.is_current_stream_capturing() FIRST and fails loudly (B1d),
     else .item() (the D2H sync)
  6. b/npad from logits, k = pre_idx.size(1) (68-70)
  7. b == 0 -> early no-op (71, B1f)
  8. npad % 4 == 0 (74-75, B1e float4 row loads)
  9. logits base 16-byte aligned (76-78)
 10. pre_idx/indices batch dims match (79-81)
 11. indices width >= k (84-85)
 12. n_valid >= 0 (86)
 13. n = min(nv, npad) clamped in unbounded ints BEFORE any narrowing (88)

Dispatch: ct_dispatch.route(b, n, npad, k) -> compile-cache keyed on
(kernel family, constexpr tuple) inside each family module -> BIND-ONCE
launch cache keyed on the shape key (b, n, npad, k): caches the compiled
callable + the prebuilt runtime-scalar arg pack as plain Python ints (probe
P12: plain ints, never pre-wrapped cutlass.Int32; pre-binding removes only
route()/marshal-prep work -- the tvm-ffi per-argument cost is paid every
call).  Hot enqueue target ~3-6 us (P12 arg-width tax); measured numbers in
notes/ct_op_NOTES.md.

Error contract (spec 1.4): launch failures surface as exceptions WITH
(b, n, npad, k) context, mirroring main.cpp:94-95.

All four family modules are imported LAZILY (first shape that routes to
them), so a missing/broken sibling only fails when actually routed to, with
(b, n, npad, k) context.  Wired compiled ABIs (verified against each
module's __call__ signature):
  ct_reg     (logits, pre_idx, out, n, CMP, QC, smem_bytes)
  ct_main    (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_, R, SMP,
              TGT, Q, SS2, TGT2)             [only family taking workspace]
  ct_clus    (logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q,
              SS2, TGT2)                      [get_compiled keyed +scap/cmp_]
  ct_regclus (logits, pre_idx, out, n)
"""


# shape key (b, n, npad, k) -> (fn, args tuple of python ints, needs_ws)
_LAUNCH_CACHE = {}
_DUMMY_KV = {}


def _dummy_kv(dev_index, device):
    """Cached 1-element int32 tensor per device — the dead kv_lens slot of
    the extended gvr_main ABI in legacy (batch-uniform) mode."""
    t = _DUMMY_KV.get(dev_index)
    if t is None:
        t = torch.zeros(1, dtype=_I32, device=device)
        _DUMMY_KV[dev_index] = t
    return t


# hot-path local bindings (each torch.<attr> lookup costs ~0.1 us; the B1
# battery runs on EVERY call — mirror of main.cpp's "sub-100ns predicted
# branches" intent within Python's reach; measured in notes/ct_op_NOTES.md)
_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_is_capturing = torch.cuda.is_current_stream_capturing
_index = operator.index
_ws_hot = _ws_keep  # shared dict object (hot-path load)
_GVR_MAX_DEV = GVR_MAX_DEV


# ---------------------------------------------------------------------------
# per-family launcher builders (cold path: once per distinct shape key)
# ---------------------------------------------------------------------------
def _build_launcher(b, n, npad, k):
    rd = route(b, n, npad, k)
    fam = rd["kernel"]
    tpl = tuple(rd["tpl"])
    rt = rd["rt"]
    if fam in ("reg", "regimg"):
        dev = _device()
        raw = dev.get_compiled__reg(tpl)

        # compiled ABI: (logits, pre_idx, kv_lens, out, n, CMP, QC,
        # smem_total) -- kv_lens is the dead varlen slot in batch-uniform
        # mode (dummy, gvr_main/reg_clus precedent)
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
        #       TGT, Q, SS2, TGT2) -- NO workspace (spec §4c); kv_lens is the
        # dead varlen slot in batch-uniform mode (dummy, reg_clus precedent)
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
        # dead varlen slot in batch-uniform mode (dummy, gvr_main precedent);
        # smem/k derived in-module
        fn = dev.get_compiled__regclus(tpl)
        n_arg = rt["n"]

        def _call(lg, pi, idx, _fn=fn, _n=n_arg):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, _n)

        return (_call, (), False)
    # unreachable: route() only emits the five families above
    raise RuntimeError(f"unknown dispatch family {fam!r}")


# ---------------------------------------------------------------------------
# run_impl mirror (main.cpp:39-96)
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

    # n_valid unwrap (main.cpp:57-67): tensor path = D2H sync, illegal under
    # CUDA graph capture -- fail loudly instead of crashing the capture (B1d).
    if isinstance(n_valid, _TENSOR):
        if _is_capturing():
            raise RuntimeError(
                "tensor n_valid requires a D2H sync, illegal under CUDA "
                "graph capture — pass n_valid as a python int"
            )
        nv = int(n_valid.item())
    else:
        # strict integral cast (pybind cast<int64_t> rejects floats/strings)
        nv = _index(n_valid)

    b, npad = lsh
    k = psh[1]
    if b == 0:  # empty batch: no-op (main.cpp:71, B1f)
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
    # clamp BEFORE any narrowing (main.cpp:87-88; python ints are unbounded,
    # so min() is the exact 64-bit clamp)
    n = nv if nv < npad else npad

    # CUDA out-indexing mirror: every kernel derives O = out + row*k
    # (kernel.cu L475/L1309 etc.) -- flat PACKED rows, ignoring the actual
    # indices width.  The DSL kernels index out[row, :] with the tensor's own
    # row stride, so a wider `indices` must be re-viewed packed (pure view,
    # no copy; contiguity already checked).
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

    # ---- n <= k short path (heuristicTopKDecode.cu:72-84) -------------------
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

    key = (b, n, npad, k)
    lc = _LAUNCH_CACHE.get(key)
    if lc is None:
        lc = _build_launcher(b, n, npad, k)
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
# exports (main.cpp:98-124)
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

    Fast 4-arg form: signature-identical to the original candidate.
    ``values`` (optional DPS output, default None = OFF) mirrors the
    production values writeback; see _run_impl.
    Default per-device slab workspace resolved FIRST (main.cpp:99-102 --
    a CPU logits tensor therefore dies with 'device index out of range').
    Hot path inlines the C binding's check + atomic-load + cache-hit
    (main.cpp:24-27); the slow path allocates under ct_workspace's lock."""
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:  # main.cpp:25, EVERY call
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

    Explicit-workspace form for multi-stream callers (main.cpp:105-116)."""
    validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices, kernel_view(workspace), values)


def run_varlen(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    engine: str = "auto",
    workspace: torch.Tensor | None = None,
) -> None:
    """Production-contract varlen entry (per-row device kv_lens).

    Row semantics (mirror of ``heuristicTopKDecode.cu`` and the in-tree
    ``cute_dsl_gvr_topk_decode`` runner):

      ``num_rows = logits.shape[0]``, ``batch = num_rows // next_n``;
      ``kv_lens`` int32 ``[batch]`` — per-request TOTAL cache length in
      UNCOMPRESSED token space (dsa.py ``metadata.kv_lens_cuda_runtime``,
      not new-token seq_lens); row ``r`` uses
      ``n_r = (kv_lens[r // next_n] - next_n + (r % next_n) + 1) //
      compress_ratio`` valid entries (cr 1 = DSv3.2, 4 = DSv4 Flash/Pro);
      ``pre_idx`` ``[batch, k]`` is REQUEST-level raw prev-step top-K,
      shared by a request's ``next_n`` rows (offset-free hint contract);
      per-row ``n_r <= k`` takes the short path (identity + ``-1`` tail).

    ENGINES: ``engine="auto"`` (default) launches the per-row IN-KERNEL
    gvr_main varlen port — ONE launch for the whole batch; each CTA reads its
    row's kv_len on device and re-derives the sampling ladder (route_dynamic
    formula mirror), so with ``max_seq_len`` given (a capture-stable engine
    constant, e.g. dsa.py's ``indexer_max_seq_len``) the call performs NO
    host reads.  Without ``max_seq_len`` the envelope comes from ONE
    ``kv_lens.max()`` host read (documented sync, refused under capture).
    ``engine="reference"`` keeps the b=1 host-loop reference implementation —
    the differential oracle the in-kernel engine is validated against.

    KNOWN LIMITATION: on rows containing NaN logits the selected index SET
    can differ from ``heuristicTopKDecode.cu`` (both kernels order NaNs
    implementation-specifically; inherited from the translation campaign's
    probe battery). Finite inputs — including +/-inf and denormals — are
    tie-aware exact.

    FULL-RANGE PRODUCTION CONTRACT: correct and dispatched for any
    ``num_rows`` (BS 1..1024+ x next_n) and any envelope up to 1M kv tokens.
    Family selection (streaming main / clustered register-resident) is a
    pure function of the capture-stable launcher key.
    """
    if logits.dtype is not torch.float32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); bf16/fp16 paths "
            "are a follow-up — see the PR roadmap"
        )
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
    if len(pre_idx.shape) != 2 or pre_idx.shape[0] != batch:
        raise RuntimeError(
            f"pre_idx must be [batch={batch}, k] REQUEST-level, got {tuple(pre_idx.shape)}"
        )
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

    if engine == "auto":
        # ---- per-row in-kernel engine (gvr_main varlen port) ----------------
        # Full B1-style validation battery (the engine bypasses _run_impl —
        # every check the legacy path enforces is replayed here; the batch-dim
        # check is CRITICAL: the kernel grid comes from logits.shape[0], so a
        # short indices/values tensor would be written out of bounds).
        if not (logits.is_cuda and pre_idx.is_cuda and indices.is_cuda):
            raise RuntimeError("all tensors must be CUDA")
        if logits.dtype is not _F32 or pre_idx.dtype is not _I32 or indices.dtype is not _I32:
            raise RuntimeError("logits must be float32; pre_idx/indices int32")
        if len(indices.shape) != 2 or indices.shape[0] != num_rows:
            raise RuntimeError(
                f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
            )
        k = pre_idx.shape[1]
        if indices.shape[1] < k:
            raise RuntimeError(f"indices width {indices.shape[1]} < k={k}")
        if not (pre_idx.is_contiguous() and indices.is_contiguous() and kv_lens.is_contiguous()):
            raise RuntimeError("pre_idx/indices/kv_lens must be contiguous")
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
                    f"values must be contiguous [num_rows={num_rows}, >=k], "
                    f"got {tuple(values.shape)}"
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
        key = (num_rows, npad, k, n_env, nn, cr)
        lc = _VARLEN_CACHE.get(key)
        if lc is None:
            if _is_capturing():
                raise RuntimeError(
                    "varlen launcher not compiled for this shape — warm up "
                    "before CUDA graph capture"
                )
            lc = _varlen_launcher(num_rows, npad, k, n_env, nn, cr)
        idx = indices
        if idx.shape[1] != k:
            idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
        vals = values
        if vals is not None and vals.shape[1] != k:
            vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
        if lc[0] == "reg_clus":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_envelope)
            lc[1](lg, pre_idx, kv_lens, idx, lc[2])
        elif lc[0] == "reg":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, CMP, QC, smem)
            lc[1](lg, pre_idx, kv_lens, idx, *lc[2])
        elif lc[0] == "clus":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, npad, k,
            #                SCAP, CMP, dead DYN x5)
            lc[1](lg, pre_idx, kv_lens, idx, *lc[2])
        else:
            _, fn, pre, tail = lc
            fn(lg, pre_idx, idx, ws, *pre, kv_lens, *tail)
        if vals is not None:
            idx64 = idx.to(torch.int64)
            vals.copy_(lg.gather(1, idx64.clamp_min(0)))
            vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
        return
    if engine != "reference":
        raise RuntimeError(f"engine must be 'auto' or 'reference', got {engine!r}")

    # ---- reference engine (differential oracle): b=1 host loop --------------
    if _is_capturing():
        raise RuntimeError(
            "run_varlen reference engine reads kv_lens on host, illegal under CUDA graph capture"
        )
    # match the engine's flat-packed output convention for wider-than-k
    # buffers (pack ONCE from the tensor base, then slice per row)
    k = pre_idx.shape[1]
    idx = indices
    if len(idx.shape) != 2 or idx.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
    kl = kv_lens.tolist()  # the ONE documented D2H sync of this engine
    for r in range(num_rows):
        # production graph slots can carry kv_len < next_n (padded / evicted
        # requests): clamp to the empty row, emitting all -1 — the same
        # contract the in-kernel engine implements
        actual = max(kl[r // nn] - nn + (r % nn) + 1, 0)
        req = r // nn
        _run_impl(
            logits[r : r + 1],
            pre_idx[req : req + 1],
            actual // cr,
            idx[r : r + 1],
            ws,
            None if vals is None else vals[r : r + 1],
        )


__all__ = [
    "route",
    "route_static",
    "route_dynamic",
    "route_split",
    "route_bands",
    "run",
    "run_ws",
    "run_varlen",
    "warmup_varlen",
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
_VARLEN_WARMUP_LOCK = threading.Lock()


def warmup_varlen(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int = 1,
    next_n: int = 1,
    num_rows_list: Sequence[int] = (1,),
    row_stride: int | None = None,
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
    dev = torch.cuda.current_device()
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
        plan_free = route(r, max(min(n_env_c, npad_c), int(top_k) + 1), npad_c, int(top_k))
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
    key = (dev, int(top_k), int(max_seq_len), int(compress_ratio), nn, tuple(rows_list), npad)
    with _VARLEN_WARMUP_LOCK:
        if key in _VARLEN_WARMUP_DONE:
            return
    rows_max = rows_list[-1]
    # one allocation at the largest geometry; smaller row counts run on
    # contiguous prefix views (compile keys depend on shapes only)
    logits = torch.zeros((rows_max, npad), dtype=torch.float32, device=dev)
    kv_lens = torch.full((rows_max // nn,), int(max_seq_len), dtype=torch.int32, device=dev)
    pre_idx = torch.zeros((rows_max // nn, int(top_k)), dtype=torch.int32, device=dev)
    out = torch.empty((rows_max, int(top_k)), dtype=torch.int32, device=dev)
    for rows in rows_list:
        batch = rows // nn
        run_varlen(
            logits[:rows],
            pre_idx[:batch],
            kv_lens[:batch],
            out[:rows],
            next_n=nn,
            compress_ratio=int(compress_ratio),
            max_seq_len=int(max_seq_len),
        )
    del logits, kv_lens, pre_idx, out
    torch.cuda.synchronize()
    # band launches compiled every ENGINE; now populate the per-row-count
    # LAUNCHER cache entries for the exact requested row counts (pure host
    # work, zero allocation/launch — engines hit the compile cache), so a
    # CUDA-graph capture at any requested geometry finds its key immediately.
    n_env_l = min(max(int(max_seq_len) >> (0 if int(compress_ratio) == 1 else 2), 1), npad)
    for r in req_rows:
        _varlen_launcher(r, npad, int(top_k), n_env_l, nn, int(compress_ratio))
    with _VARLEN_WARMUP_LOCK:
        _VARLEN_WARMUP_DONE.add(key)
