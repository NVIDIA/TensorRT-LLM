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
convention serves all three models. The production decode engine
instead reads per-request ``seq_lens`` on-device with per-row MTP offsets
(sync-free, CUDA-graph-replay safe with growing KV); adopting that per-row
contract inside these kernels is tracked follow-up work. Until then this
module is exercised standalone (unit tests / benchmarking) and must not be
substituted for the tiered path under continuous batching, MTP
(``next_n > 1``), or CUDA-graph capture.
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


def route(b, n, npad, k):
    """Mirror of gvr_topk_launch (kernel.cu L2754-3197). Pure. See module doc."""
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
        fn = dev.get_compiled__reg(tpl)
        # compiled ABI: (logits, pre_idx, out, n, CMP, QC, smem_total)
        args = (rt["n"], rt["CMP"], rt["QC"], dev.STATIC_BYTES + rd["smem"])
        return (fn, args, False)
    if fam == "main":
        dev = _device()
        fn = dev.get_compiled(tpl)
        # compiled ABI: (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_,
        #                R, SMP, TGT, Q, SS2, TGT2)  [SCAP_/CMP_ dead, ABI parity]
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
        # ABI: (logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q,
        #       SS2, TGT2) -- NO workspace (spec §4c)
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
        return (fn, args, False)
    if fam == "reg_clus":
        dev = _device()
        # compiled ABI: (logits, pre_idx, out, n) -- smem/k derived in-module
        fn = dev.get_compiled__regclus(tpl)
        return (fn, (rt["n"],), False)
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
    try:
        if lc is None:
            lc = _build_launcher(b, n, npad, k)
            _LAUNCH_CACHE[key] = lc
        fn, args, needs_ws = lc
        if needs_ws:
            fn(logits, pre_idx, indices, ws, *args)
        else:
            fn(logits, pre_idx, indices, *args)
    except Exception as e:
        raise RuntimeError(f"gvr_topk launch failed (b={b} n={n} npad={npad} k={k}): {e}") from e
    if values is not None:
        values.copy_(logits.gather(1, indices.to(torch.int64)))


# ---------------------------------------------------------------------------
# exports (main.cpp:98-124)
# ---------------------------------------------------------------------------
def run(logits, pre_idx, n_valid, indices, values=None):
    """Fast 4-arg form: signature-identical to the original candidate.
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


def run_ws(logits, pre_idx, n_valid, indices, workspace, values=None):
    """Explicit-workspace form for multi-stream callers (main.cpp:105-116)."""
    validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices, kernel_view(workspace), values)


__all__ = [
    "route",
    "run",
    "run_ws",
    "workspace_bytes",
    "WS_BYTES",
    "default_workspace",
    "validate_run_ws",
    "kernel_view",
]
