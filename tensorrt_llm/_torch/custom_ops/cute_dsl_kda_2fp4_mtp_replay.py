# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 2FP4 KDA MTP replay with fused Conv4, QK normalization and gated RMSNorm."""

# Fused Conv4/SiLU + QK norm + native-NVFP4 KDA MTP replay + RMSNorm/sigmoid gate.
# One CTA (128 threads) per (head, batch-row).  Both checkpoint projections (state x K-operand
# and state x Q-operand, all T tokens at once) run as a single tcgen05.mma kind::mxf4nvf4
# block16 GEMM: A = primary checkpoint (E2M1, UE4M3 block-16 scales from the pool),
# B = per-token quantized [k_t; q_t] rows (E2M1 + UE4M3 block-16 scales), D = FP32 in TMEM.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir import ir as mlir_ir
from cutlass._mlir.dialects import nvvm as nvvm_raw
from cutlass.cutlass_dsl import T as MT
from cutlass.experimental import primitives as prims

KD = 128  # K = V = 128
HROWS = 16  # history window W
STATE_BYTES = 18448
NT = 128  # threads per CTA
ASTR = 136  # padded bf16 row stride for dot-product operands (272 B, conflict-free)
LOG2E = 1.4426950408889634
QSCALE = 128.0**-0.5
INV2688 = 1.0 / (6 * 448)
GMEM = cutlass.AddressSpace.gmem
TMEM_COLS = 32  # [0,16) accumulator, [16,24) SFA (2 k-blocks x 4 cols), [24,32) SFB


def _exp_tl(x):
    # Triton tl.exp on fp32: ex2.approx(x * log2(e))
    return cute.exp2(x * Float32(LOG2E), approx=True)


def _bf(x):
    return Float32(BFloat16(x))


# Bit-exact replica of the reference's libdevice SiLU: x * rcp_rn(1 + __nv_expf(-x)), with the
# same ftz flags and no mul/add contraction (the reference is compiled with fp fusion disabled).
_SILU_PTX = """{
.reg .b32 xn, xt, xj, xs, xf, xe, xd, xr;
neg.f32 xn, {$r0};
fma.rn.ftz.f32 xt, xn, 0f3BBB989D, 0f3F000000;
cvt.ftz.sat.f32.f32 xt, xt;
fma.rm.ftz.f32 xj, xt, 0f437C0000, 0f4B400001;
add.rn.f32 xs, xj, 0fCB40007F;
neg.f32 xs, xs;
fma.rn.ftz.f32 xf, xn, 0f3FB8AA3B, xs;
fma.rn.ftz.f32 xf, xn, 0f32A57060, xf;
shl.b32 xj, xj, 23;
ex2.approx.ftz.f32 xe, xf;
mul.rn.f32 xe, xe, xj;
add.rn.f32 xd, xe, 0f3F800000;
rcp.rn.ftz.f32 xr, xd;
mul.rn.f32 {$w0}, xr, {$r0};
}"""


def _silu_ref(x):
    return cute.arch.inline_ptx(_SILU_PTX, write_only_types=[Float32], read_only_args=[x])


def _mul_rn(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[Float32(a), Float32(b)],
    )


def _add_rn(a, b):
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[Float32(a), Float32(b)],
    )


def _mul_qscale_rn(a):
    # a * fp32(128**-0.5) with explicit .rn (immediates must live in the asm string)
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, 0f3DB504F3;",
        write_only_types=[Float32],
        read_only_args=[Float32(a)],
    )


def _div_mk(x, d, r):
    # x / d correctly rounded (== div.rn) for normal-range operands, given r = RN(1/d):
    # Markstein correction q' = RN(q + r * RN(x - q*d)) with the residual exact via FMA.
    q = x * r
    e = cute.fma(-q, d, x)
    return cute.fma(e, r, q)


def _add1_rn(a):
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, 0f3F800000;",
        write_only_types=[Float32],
        read_only_args=[Float32(a)],
    )


@cute.kernel
def _replay_kernel(
    p_raw: Int64,
    p_conv: Int64,
    p_cw: Int64,
    p_g: Int64,
    p_beta: Int64,
    p_z: Int64,
    p_nw: Int64,
    p_pool: Int64,
    p_slots: Int64,
    p_dummy: Int64,
    p_smap: Int64,
    p_hk: Int64,
    p_hu: Int64,
    p_hg: Int64,
    p_hlen: Int64,
    p_a_log: Int64,
    p_dtb: Int64,
    p_out: Int64,
    p_cand: Int64,
    p_oldlen: Int64,
    candidate_bs: Int64,
    lower: Float32,
    eps: Float32,
    H: cutlass.Constexpr[int],
    raw_bs: cutlass.Constexpr[int],
    raw_rs: cutlass.Constexpr[int],
    conv_ss: cutlass.Constexpr[int],
    g_bs: cutlass.Constexpr[int],
    g_rs: cutlass.Constexpr[int],
    beta_bs: cutlass.Constexpr[int],
    beta_rs: cutlass.Constexpr[int],
    z_bs: cutlass.Constexpr[int],
    z_rs: cutlass.Constexpr[int],
    pool_ss: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    BOUNDED: cutlass.Constexpr[int],
    POOL_VEC: cutlass.Constexpr[int],
    CAP: cutlass.Constexpr[int],
):
    N2 = 2 * T
    MMA_N: cutlass.Constexpr[int] = 8 if cutlass.const_expr(T <= 4) else 16
    NROWS = HROWS + T
    tid, _, _ = cute.arch.thread_idx()
    h, b, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    HK = H * KD
    C3 = HK * 3

    # ---------------- shared memory ----------------
    sA = cutlass.Array(Uint8, (8192,), space=cutlass.AddressSpace.smem, alignment=1024)
    sB = cutlass.Array(Uint32, (MMA_N * 16,), space=cutlass.AddressSpace.smem, alignment=1024)
    # staged history K rows (stride ASTR); turned in place into the adjusted-key rows of the
    # coefficient products, followed by T adjusted new-key rows
    s_uh = cutlass.Array(BFloat16, (HROWS * KD,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gh = cutlass.Array(Float32, (HROWS * KD,), space=cutlass.AddressSpace.smem, alignment=128)
    s_A = cutlass.Array(BFloat16, (NROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_R = cutlass.Array(BFloat16, (N2 * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_coef = cutlass.Array(Float32, (HROWS * MMA_N,), space=cutlass.AddressSpace.smem, alignment=16)
    s_coefn = cutlass.Array(Float32, (T * MMA_N,), space=cutlass.AddressSpace.smem, alignment=16)
    # SFB staging in the UTCCP 32x128b atom image: [k-block][row L][4 words], word 0 = 4 scales
    s_sfb = cutlass.Array(Uint32, (2 * 128,), space=cutlass.AddressSpace.smem, alignment=128)
    s_red1 = cutlass.Array(Float32, (N2 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red2 = cutlass.Array(Float32, (4 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_den = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_rden = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_outer = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red3 = cutlass.Array(Float32, (4 * 8,), space=cutlass.AddressSpace.smem, alignment=16)
    s_tmem = cutlass.Array(Int32, (1,), space=cutlass.AddressSpace.smem, alignment=16)
    s_mbar = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)

    out_ptr = cutlass.inttoptr(p_out, GMEM, BFloat16)
    cand_ptr = cutlass.inttoptr(p_cand, GMEM, BFloat16)
    oldlen_ptr = cutlass.inttoptr(p_oldlen, GMEM, Int32)
    dummy_ptr = cutlass.inttoptr(p_dummy, GMEM, Uint8)
    b64 = Int64(b)
    h64 = Int64(h)
    hoff = h * KD + tid

    is_dummy = (dummy_ptr + b64).load()
    if is_dummy != Uint8(0):
        # Dummy row: exact zeros for every destination element, nothing else touched.
        o_row = out_ptr + ((b64 * T * H + h64) * KD + tid)
        c_row = cand_ptr + (b64 * candidate_bs + hoff)
        for t in cutlass.range_constexpr(T):
            o_row.store(BFloat16(0.0))
            for sec in cutlass.range_constexpr(3):
                (c_row + sec * HK).store(BFloat16(0.0))
            o_row = o_row + HK
            c_row = c_row + C3
        if h == 0:
            if tid == 0:
                (oldlen_ptr + b64).store(Int32(0))
    else:
        slots_ptr = cutlass.inttoptr(p_slots, GMEM, Int32)
        slot = (slots_ptr + b64).load()
        # ---- slot-independent loads ----
        raw_ptr = cutlass.inttoptr(p_raw, GMEM, BFloat16)
        z_ptr = cutlass.inttoptr(p_z, GMEM, BFloat16)
        g_ptr = cutlass.inttoptr(p_g, GMEM, BFloat16)
        beta_ptr = cutlass.inttoptr(p_beta, GMEM, BFloat16)
        cw_ptr = cutlass.inttoptr(p_cw, GMEM, BFloat16)
        xs = []
        x_row0 = raw_ptr + (b64 * raw_bs + hoff)
        for sec in cutlass.range_constexpr(3):
            x_row = x_row0 + sec * HK
            for t in cutlass.range_constexpr(T):
                xs.append(x_row.load())
                x_row = x_row + raw_rs
        zs = []
        gs = []
        betas = []
        z_row = z_ptr + (b64 * z_bs + hoff)
        g_row = g_ptr + (b64 * g_bs + hoff)
        b_row = beta_ptr + (b64 * beta_bs + h64)
        for t in cutlass.range_constexpr(T):
            zs.append(z_row.load())
            gs.append(g_row.load())
            betas.append(b_row.load())
            z_row = z_row + z_rs
            g_row = g_row + g_rs
            b_row = b_row + beta_rs
        cws = []
        for sec in cutlass.range_constexpr(3):
            cws.append((cw_ptr + Int64(sec * HK + hoff) * 4).load(alignment=8, count=4))
        dtb = (cutlass.inttoptr(p_dtb, GMEM, Float32) + hoff).load()
        a_log = (cutlass.inttoptr(p_a_log, GMEM, Float32) + h).load()
        nw = (cutlass.inttoptr(p_nw, GMEM, BFloat16) + tid).load()

        s_sfb[tid] = Uint32(0)
        s_sfb[tid + NT] = Uint32(0)
        # ---- TMEM allocation + mbarrier init ----
        if warp == 0:
            prims.tcgen05_alloc(s_tmem, TMEM_COLS, group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if tid == 0:
            prims.mbarrier_init(s_mbar, 1)
            prims.fence_mbarrier_init()

        # ---- slot-dependent loads ----
        slot64 = Int64(slot)
        live = (cutlass.inttoptr(p_smap, GMEM, Int32) + slot64).load()
        pool_row = p_pool + slot64 * pool_ss + h64 * STATE_BYTES
        pool_u8 = cutlass.inttoptr(pool_row, GMEM, Uint8)
        # state primary values -> SW64 K-major smem tile (8 rows x 64B atoms, chunk ^= (row>>1)&3)
        sfa = []
        if cutlass.const_expr(POOL_VEC != 0):
            # 16B-aligned slot stride: async 16B copies + vector scale loads
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                r = q // 4
                c = q % 4
                dst = r * 64 + ((c ^ ((r >> 1) & 3)) * 16)
                prims.cp_async_shared_global(
                    sA.data_ptr() + dst, pool_u8 + (r * 64 + c * 16), 16, "cg"
                )
            for j in cutlass.range_constexpr(4):
                m = lane + 32 * j
                v = cutlass.inttoptr(pool_row + 16384 + m * 8, GMEM, Uint32).load(
                    alignment=8, count=2
                )
                sfa.append(v)
            outer_s = cutlass.inttoptr(pool_row + 18432, GMEM, Float32).load()
        else:
            # arbitrary (unaligned) slot stride: byte-granular loads, same smem/TMEM images
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                r = q // 4
                c = q % 4
                dst = r * 64 + ((c ^ ((r >> 1) & 3)) * 16)
                bv = (pool_u8 + (r * 64 + c * 16)).load(alignment=1, count=16)
                (sA.data_ptr() + dst).store(bv, alignment=16)
            for j in cutlass.range_constexpr(4):
                m = lane + 32 * j
                bv = (pool_u8 + (16384 + m * 8)).load(alignment=1, count=8)
                sfa.append(bv.bitcast(Uint32))
            outer_s = (pool_u8 + 18432).load(alignment=1, count=4).bitcast(Float32)[0]
        conv_ptr = cutlass.inttoptr(p_conv, GMEM, BFloat16)
        cs = []
        for sec in cutlass.range_constexpr(3):
            cbase = slot64 * conv_ss + Int64(sec * HK + hoff) * 3
            for p in cutlass.range_constexpr(3):
                cs.append((conv_ptr + (cbase + p)).load())

        # ---- live-dependent loads: history rows (zero-filled past length) ----
        live64 = Int64(live)
        length = (cutlass.inttoptr(p_hlen, GMEM, Int32) + live64).load()
        hrow = (live64 * H + h64) * CAP  # row index of (live, h, 0)
        hk_ptr = cutlass.inttoptr(p_hk, GMEM, BFloat16)
        hu_ptr = cutlass.inttoptr(p_hu, GMEM, BFloat16)
        hg_ptr = cutlass.inttoptr(p_hg, GMEM, Float32)
        hk_base = hk_ptr + hrow * KD
        hu_base = hu_ptr + hrow * KD
        hg_base = hg_ptr + hrow * KD
        gdx, gdy, _ = cute.arch.grid_dim()
        if gdx * gdy <= 1024:
            # small launch: fetch the whole 16-row window without waiting for history_len
            for k in cutlass.range_constexpr(2):
                q = tid + NT * k
                r = q // 16
                prims.cp_async_shared_global(
                    s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg"
                )
                prims.cp_async_shared_global(s_uh.data_ptr() + q * 8, hu_base + q * 8, 16, "cg")
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                prims.cp_async_shared_global(s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg")
        else:
            # large launch: only the live rows (bandwidth bound)
            for k in cutlass.range_constexpr(2):
                q = tid + NT * k
                r = q // 16
                nbytes = Int32(16) if r < length else Int32(0)
                prims.cp_async_shared_global(
                    s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg", cp_size=nbytes
                )
                prims.cp_async_shared_global(
                    s_uh.data_ptr() + q * 8, hu_base + q * 8, 16, "cg", cp_size=nbytes
                )
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                r = q // 32
                nbytes = Int32(16) if r < length else Int32(0)
                prims.cp_async_shared_global(
                    s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg", cp_size=nbytes
                )
        prims.cp_async_commit_group()

        # ---- gates (Triton-pinned elementary ops); only raw_g / dt_bias / A_log needed ----
        rate = _exp_tl(a_log)
        gates = []
        for t in cutlass.range_constexpr(T):
            shifted = Float32(gs[t]) + dtb
            if cutlass.const_expr(BOUNDED != 0):
                gates.append(
                    _mul_rn(lower, Float32(1.0) / _add1_rn(_exp_tl(_mul_rn(shifted, -rate))))
                )
            else:
                spv = cute.log(Float32(1.0) + _exp_tl(shifted))
                sp = spv if shifted <= Float32(20.0) else shifted
                gates.append(-rate * sp)

        # ---- Conv4 + SiLU (thread = key/value channel) ----
        qv = []
        kv = []
        vv = []
        for sec in cutlass.range_constexpr(3):
            wv = cws[sec]
            for t in cutlass.range_constexpr(T):
                acc = Float32(0.0)
                for tap in cutlass.range_constexpr(4):
                    pos = t + tap
                    if cutlass.const_expr(pos < 3):
                        xv = Float32(cs[sec * 3 + pos])
                    else:
                        xv = Float32(xs[sec * T + pos - 3])
                    acc = cute.fma(xv, Float32(wv[tap]), acc)
                if cutlass.const_expr(sec == 2):
                    sv = _bf(
                        acc * cute.rcp(Float32(1.0) + cute.exp(-acc, fastmath=True), approx=True)
                    )
                else:
                    sv = _bf(_silu_ref(acc))
                if cutlass.const_expr(sec == 0):
                    qv.append(sv)
                elif cutlass.const_expr(sec == 1):
                    kv.append(sv)
                else:
                    vv.append(sv)

        cw_row = cand_ptr + (b64 * candidate_bs + hoff)
        for t in cutlass.range_constexpr(T):
            for sec in cutlass.range_constexpr(3):
                (cw_row + sec * HK).store(xs[sec * T + t])
            cw_row = cw_row + C3

        # ---- reduction 1: sum of squares of q and k per token ----
        sq = []
        for t in cutlass.range_constexpr(T):
            sq.append(_mul_rn(qv[t], qv[t]))
        for t in cutlass.range_constexpr(T):
            sq.append(_mul_rn(kv[t], kv[t]))
        # Bit-exact replica of the reference tl.sum order (Triton layout: 16 lanes x 8 consecutive
        # keys per row): in-thread ((e0+e2)+(e4+e6))+((e1+e3)+(e5+e7)), then lane butterfly
        # xor 8,4,2,1 over the 16 key-groups.  Here thread = key, so the in-thread part is a
        # xor 2,4,1 butterfly inside 8-lane groups and the group butterfly runs from smem.
        for i in cutlass.range_constexpr(N2):
            x = sq[i]
            x = x + cute.arch.shuffle_sync_bfly(x, offset=2)
            x = x + cute.arch.shuffle_sync_bfly(x, offset=4)
            x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
            sq[i] = x
        if lane % 8 == 0:
            for i in cutlass.range_constexpr(N2):
                s_red1[i * 16 + warp * 4 + lane // 8] = sq[i]
        prims.barrier_cta_sync(0)
        if tid < N2:
            tgv = (s_red1.data_ptr() + tid * 16).load(alignment=16, count=16)
            ug = []
            for m in cutlass.range_constexpr(8):
                ug.append(tgv[m] + tgv[m + 8])
            vg = []
            for m in cutlass.range_constexpr(4):
                vg.append(ug[m] + ug[m + 4])
            tot = (vg[0] + vg[2]) + (vg[1] + vg[3])
            dden = cute.sqrt(tot + Float32(1e-6))
            s_den[tid] = dden
            s_rden[tid] = Float32(1.0) / dden
        prims.barrier_cta_sync(0)
        # SFA (checkpoint block-16 scales) -> TMEM cols 16..23; each warp replicates into its
        # sub-partition: col 16 + 4*kb + j holds rows 32*j + lane, k-blocks 4kb..4kb+3
        tmem_base = s_tmem[0]
        sfa_vals = []
        for kb in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(4):
                sfa_vals.append(Int32(sfa[j][kb]))
        sfa_vec = cutlass.Vector.from_elements(tuple(sfa_vals), Int32)
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(tmem_base + ((warp * 32) << 16) + 16, 6, Int32), sfa_vec
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        qn = []
        kn = []
        for t in cutlass.range_constexpr(T):
            # denominators sqrt(sum+eps) >= 1e-3 and conv outputs are normal-range: the
            # Markstein form equals div.rn bit-for-bit here
            qn.append(_bf(_div_mk(_mul_qscale_rn(qv[t]), s_den[t], s_rden[t])))
            kn.append(_bf(_div_mk(kv[t], s_den[T + t], s_rden[T + t])))

        # ---- history staged ----
        prims.cp_async_wait_group(0)
        prims.barrier_cta_sync(0)
        aidx = (length - 1) if length > 0 else Int32(0)
        anchor_ld = s_gh[aidx * KD + tid]
        anchor = anchor_ld if length > 0 else Float32(0.0)

        # ---- cumulative G (Triton-pinned elementary ops) ----
        cum = []
        prev = anchor
        for t in cutlass.range_constexpr(T):
            cur = _add_rn(prev, gates[t])
            cum.append(cur)
            prev = cur

        # ---- per-token native operand values ----
        # history adjusted keys first: they read s_gh, whose columns are then reused for pv
        a_hist = s_A.data_ptr() + tid
        g_hist = s_gh.data_ptr() + tid
        for r in cutlass.range(length, unroll=1):
            khv = Float32(a_hist.load())
            ghv = g_hist.load()
            a_hist.store(BFloat16(khv * cute.exp(anchor - ghv, fastmath=True)))
            a_hist = a_hist + ASTR
            g_hist = g_hist + KD
        for t in cutlass.range_constexpr(T):
            ecum = _exp_tl(cum[t])
            s_gh[(2 * t) * KD + tid] = _mul_rn(kn[t], ecum)
            s_gh[(2 * t + 1) * KD + tid] = _mul_rn(qn[t], ecum)
        red2 = []
        for t in cutlass.range_constexpr(T):
            x = kn[t] * qn[t]
            for o in cutlass.range_constexpr(5):
                x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> o)
            red2.append(x)
        if lane == 0:
            for t in cutlass.range_constexpr(T):
                s_red2[warp * 8 + t] = red2[t]

        # relative pair (B operand of history/new-token coefficient products) and
        # adjusted new keys; adjusted history keys.
        for t in cutlass.range_constexpr(T):
            er = cute.exp(cum[t] - anchor, fastmath=True)
            s_R[(2 * t) * ASTR + tid] = BFloat16(kn[t] * er)
            s_R[(2 * t + 1) * ASTR + tid] = BFloat16(qn[t] * er)
            ea = cute.exp(anchor - cum[t], fastmath=True)
            s_A[(HROWS + t) * ASTR + tid] = BFloat16(kn[t] * ea)

        # ---- persistent writes that need no further math ----
        hk_w = hk_ptr + ((hrow + length) * KD + tid)
        hg_w = hg_ptr + ((hrow + length) * KD + tid)
        for t in cutlass.range_constexpr(T):
            (hk_w + t * KD).store(BFloat16(kn[t]))
            (hg_w + t * KD).store(cum[t])

        prims.barrier_cta_sync(0)
        # ---- quantize operand rows to E2M1 + UE4M3 block-16 scales ----
        kqs = []
        for t in cutlass.range_constexpr(T):
            kqs.append(s_red2[t] + s_red2[8 + t] + s_red2[16 + t] + s_red2[24 + t])
        e4m3 = mlir_ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type)
        e2m1 = mlir_ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
        # one thread per (operand row n = 2t + p, 16-key block): no shuffles, one scale per block
        qrow = tid // 8
        qblk = tid % 8
        qphys = qrow * 64 + (((qblk // 2) ^ ((qrow >> 1) & 3)) * 16) + (qblk % 2) * 8
        if qrow < N2:
            # Each token occupies one 16-lane half-warp (two operand rows x eight blocks).
            # On odd-T tail warps, exclude the inactive upper half from synchronizing shuffles.
            if cutlass.const_expr(T % 2 == 1):
                qmask = Int32(0x0000FFFF) if warp == T // 2 else Int32(-1)
            else:
                qmask = Int32(-1)
            qbase = s_gh.data_ptr() + (qrow * KD + qblk * 16)
            qv4 = []
            for i in cutlass.range_constexpr(4):
                qv4.append((qbase + 4 * i).load(alignment=16, count=4))
            vals = []
            for i in cutlass.range_constexpr(16):
                vals.append(qv4[i // 4][i % 4])
            bmx = cute.arch.fmax(vals[0], -vals[0])
            for i in cutlass.range_constexpr(1, 16):
                bmx = cute.arch.fmax(bmx, cute.arch.fmax(vals[i], -vals[i]))
            tmx = bmx
            for o in cutlass.range_constexpr(4):
                tmx = cute.arch.fmax(
                    tmx, cute.arch.shuffle_sync_bfly(tmx, offset=8 >> o, mask=qmask)
                )
            outer = tmx * Float32(INV2688)
            safe = Float32(1.0) if outer == Float32(0.0) else outer
            den = Float32(6.0) * safe
            sc = cute.arch.fmin(Float32(448.0), bmx / den)
            w16 = prims.dialect.convert_f32x2_to_f8x2(
                MT.i16(),
                Float32(0.0),
                sc,
                e4m3,
                rnd=nvvm_raw.FPRoundingMode.RN,
                sat=nvvm_raw.SaturationMode.SATFINITE,
            )
            sbyte = Int32(cutlass.Int16(w16)) & Int32(0xFF)
            sfw = sbyte | (cute.arch.shuffle_sync_down(sbyte, offset=1, mask=qmask) << 8)
            sfw = sfw | (cute.arch.shuffle_sync_down(sfw, offset=2, mask=qmask) << 16)
            if tid % 16 == 0:
                s_outer[qrow >> 1] = outer
            if qblk % 4 == 0:
                s_sfb[(qblk // 4) * 128 + qrow * 4] = Uint32(sfw)
            ex = sbyte >> 3
            mn = sbyte & 7
            nrm_bits = ((ex + 120) << 23) | (mn << 20)
            scf_n = nrm_bits.bitcast(Float32)
            scf_s = Float32(mn) * Float32(0.001953125)
            scf = scf_n if ex > 0 else scf_s
            expd = scf * outer
            if expd >= Float32(7.888609052210118e-31):
                # normal-range scale (>= 2^-100): one correctly rounded reciprocal shared by the
                # 16 elements, Markstein-corrected quotients (bit-identical to div.rn)
                rexp = Float32(1.0) / expd
                words = []
                for wi in cutlass.range_constexpr(2):
                    qword = Int32(0)
                    for bi in cutlass.range_constexpr(4):
                        e0 = 8 * wi + 2 * bi
                        n0 = _div_mk(vals[e0], expd, rexp)
                        n1 = _div_mk(vals[e0 + 1], expd, rexp)
                        c8 = prims.dialect.convert_f32x2_to_f4x2(n1, n0, e2m1)
                        qword = qword | ((Int32(cutlass.Int8(c8)) & Int32(0xFF)) << (8 * bi))
                    words.append(qword)
                sB[qphys // 4] = Uint32(words[0])
                sB[qphys // 4 + 1] = Uint32(words[1])
            else:
                zexp = expd == Float32(0.0)
                words2 = []
                for wi in cutlass.range_constexpr(2):
                    qword2 = Int32(0)
                    for bi in cutlass.range_constexpr(4):
                        e0 = 8 * wi + 2 * bi
                        n0 = Float32(0.0) if zexp else vals[e0] / expd
                        n1 = Float32(0.0) if zexp else vals[e0 + 1] / expd
                        c8 = prims.dialect.convert_f32x2_to_f4x2(n1, n0, e2m1)
                        qword2 = qword2 | ((Int32(cutlass.Int8(c8)) & Int32(0xFF)) << (8 * bi))
                    words2.append(qword2)
                sB[qphys // 4] = Uint32(words2[0])
                sB[qphys // 4 + 1] = Uint32(words2[1])
        else:
            # Only descriptor-visible padding rows need packed zeros.  SFB was cleared in full
            # before this phase, so its padding scales remain exact zero without extra stores.
            if qrow < MMA_N:
                sB[qphys // 4] = Uint32(0)
                sB[qphys // 4 + 1] = Uint32(0)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)

        # ---- native NVFP4 MMA: D[128 x 16] = S(E2M1,UE4M3) x [k_t;q_t](E2M1,UE4M3)^T ----
        if warp == 0:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,
                n_dim=MMA_N,
                m_dim=128,
                sparsity_version=0,
            )
            desc_a = prims.Tcgen05SmemDesc.build(
                sA,
                leading_byte_offset=16,
                stride_byte_offset=512,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
            desc_b = prims.Tcgen05SmemDesc.build(
                sB,
                leading_byte_offset=16,
                stride_byte_offset=512,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
            d_ptr = cutlass.inttoptr(tmem_base, 6, Float32)
            desc_sf = prims.Tcgen05SmemDesc.build(
                s_sfb,
                leading_byte_offset=16,
                stride_byte_offset=128,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
            sf_shape, sf_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            if prims.elect_sync():
                for kb in cutlass.range_constexpr(2):
                    prims.tcgen05_cp(
                        sf_shape,
                        cutlass.inttoptr(tmem_base + 24 + 4 * kb, 6, Int32),
                        desc_sf + 32 * kb,
                        group=prims.CTAGroup.CTA_1,
                        multicast=sf_mc,
                    )
                for kb in cutlass.range_constexpr(2):
                    sa_ptr = cutlass.inttoptr(tmem_base + 16 + 4 * kb, 6, Int32)
                    sb_ptr = cutlass.inttoptr(tmem_base + 24 + 4 * kb, 6, Int32)
                    prims.tcgen05_mma_block_scale(
                        prims.MMABlockScaleKind.MXF4NVF4,
                        prims.CTAGroup.CTA_1,
                        d_ptr,
                        desc_a + 2 * kb,
                        desc_b + 2 * kb,
                        idesc,
                        Boolean(kb != 0),
                        scale_a=sa_ptr,
                        scale_b=sb_ptr,
                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                    )
                prims.tcgen05_commit(s_mbar, group=prims.CTAGroup.CTA_1)

        # ---- coefficient products on warp tensor cores (bf16 x bf16 -> fp32) ----
        # C[32 x 16] = A[32 x 128] R[16 x 128]^T; rows 0..15 history (bf16-rounded result),
        # rows 16..16+T-1 new tokens (fp32).  Warp w: m-tile w // 2, n-tile w % 2, 8 k16 steps.
        mt = warp // 2
        nt = warp % 2
        if cutlass.const_expr(MMA_N == 8):
            coefficient_active = nt == 0
        else:
            coefficient_active = True
        if coefficient_active:
            # For N8, nt==1 would duplicate/clamp B rows and discard every result.  Keep every
            # fragment/address/accumulator local to this dynamic branch so its inactive warps do
            # not create a type-unstable SSA join.
            gid = lane // 4
            tig = lane % 4
            ra0 = 16 * mt + gid
            ra1 = ra0 + 8
            if cutlass.const_expr(NROWS < 32):
                ra0 = ra0 if ra0 < NROWS else Int32(NROWS - 1)
                ra1 = ra1 if ra1 < NROWS else Int32(NROWS - 1)
            rb = 8 * nt + gid
            if cutlass.const_expr(N2 < 16):
                rb = rb if rb < N2 else Int32(N2 - 1)
            pa0 = s_A.data_ptr() + (ra0 * ASTR + 2 * tig)
            pa1 = s_A.data_ptr() + (ra1 * ASTR + 2 * tig)
            pb = s_R.data_ptr() + (rb * ASTR + 2 * tig)
            d0 = Float32(0.0)
            d1 = Float32(0.0)
            d2 = Float32(0.0)
            d3 = Float32(0.0)
            for ks in cutlass.range_constexpr(KD // 16):
                fa0 = (pa0 + 16 * ks).load(alignment=4, count=2).bitcast(Int32)[0]
                fa1 = (pa1 + 16 * ks).load(alignment=4, count=2).bitcast(Int32)[0]
                fa2 = (pa0 + (16 * ks + 8)).load(alignment=4, count=2).bitcast(Int32)[0]
                fa3 = (pa1 + (16 * ks + 8)).load(alignment=4, count=2).bitcast(Int32)[0]
                fb0 = (pb + 16 * ks).load(alignment=4, count=2).bitcast(Int32)[0]
                fb1 = (pb + (16 * ks + 8)).load(alignment=4, count=2).bitcast(Int32)[0]
                if cutlass.const_expr(ks == 0):
                    # zero accumulator supplied inside the asm (immediates cannot be asm operands)
                    d0, d1, d2, d3 = cute.arch.inline_ptx(
                        "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                        "{zz, zz, zz, zz};\n}",
                        write_only_types=[Float32, Float32, Float32, Float32],
                        read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                    )
                else:
                    d0, d1, d2, d3 = cute.arch.inline_ptx(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                        "{{$r6}, {$r7}, {$r8}, {$r9}};",
                        write_only_types=[Float32, Float32, Float32, Float32],
                        read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1, d0, d1, d2, d3],
                    )
            cn = 8 * nt + 2 * tig
            if cutlass.const_expr(HROWS == 16):
                if mt == 0:
                    if cn < N2:
                        s_coef[gid * MMA_N + cn] = _bf(d0)
                        s_coef[gid * MMA_N + cn + 1] = _bf(d1)
                        s_coef[(gid + 8) * MMA_N + cn] = _bf(d2)
                        s_coef[(gid + 8) * MMA_N + cn + 1] = _bf(d3)
                else:
                    if cn < N2:
                        if gid < T:
                            s_coefn[gid * MMA_N + cn] = d0
                            s_coefn[gid * MMA_N + cn + 1] = d1
                        if gid + 8 < T:
                            s_coefn[(gid + 8) * MMA_N + cn] = d2
                            s_coefn[(gid + 8) * MMA_N + cn + 1] = d3
        prims.barrier_cta_sync(0)

        # ---- epilogue: thread = value row i ----
        while not prims.mbarrier_try_wait_parity(s_mbar, 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc_ptr = cutlass.inttoptr(tmem_base + ((warp * 32) << 16), 6, Float32)
        accv = prims.tcgen05_ld("32x32b", acc_ptr, num=MMA_N)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        pred = []
        for n in cutlass.range_constexpr(N2):
            pred.append(accv[n] * outer_s * s_outer[n // 2])
        predf = cutlass.Array(Float32, (N2,), space=cutlass.AddressSpace.rmem)
        for n in cutlass.range_constexpr(N2):
            predf[n] = pred[n]
        u_hist = s_uh.data_ptr() + tid
        c_hist = s_coef.data_ptr()
        for r in cutlass.range(length, unroll=1):
            ur = Float32(u_hist.load())
            cv = c_hist.load(alignment=16, count=MMA_N)
            for n in cutlass.range_constexpr(N2):
                predf[n] = cute.fma(ur, cv[n], predf[n])
            u_hist = u_hist + KD
            c_hist = c_hist + MMA_N
        for n in cutlass.range_constexpr(N2):
            pred[n] = predf[n]
        us = []
        cores = []
        for t in cutlass.range_constexpr(T):
            pk = pred[2 * t]
            pq = pred[2 * t + 1]
            for s in cutlass.range_constexpr(t):
                pk = cute.fma(us[s], s_coefn[s * MMA_N + 2 * t], pk)
                pq = cute.fma(us[s], s_coefn[s * MMA_N + 2 * t + 1], pq)
            bsig = cute.rcp(Float32(1.0) + cute.exp(-Float32(betas[t]), fastmath=True), approx=True)
            u = bsig * (vv[t] - pk)
            us.append(u)
            cores.append(_bf(pq + u * kqs[t]))
        ssq = []
        for t in cutlass.range_constexpr(T):
            x = cores[t] * cores[t]
            for o in cutlass.range_constexpr(5):
                x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> o)
            ssq.append(x)
        if lane == 0:
            for t in cutlass.range_constexpr(T):
                s_red3[warp * 8 + t] = ssq[t]
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp == 0:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_base, 6, Float32), TMEM_COLS, group=prims.CTAGroup.CTA_1
            )
        nwf = Float32(nw)
        hu_w = hu_ptr + ((hrow + length) * KD + tid)
        o_row2 = out_ptr + ((b64 * T * H + h64) * KD + tid)
        for t in cutlass.range_constexpr(T):
            tot = s_red3[t] + s_red3[8 + t] + s_red3[16 + t] + s_red3[24 + t]
            rs = cute.rsqrt(tot * Float32(1.0 / 128.0) + eps)
            zf = Float32(zs[t])
            sg = cute.rcp(Float32(1.0) + cute.exp(-zf, fastmath=True), approx=True)
            o = cores[t] * rs * nwf * sg
            (o_row2 + t * HK).store(BFloat16(o))
            (hu_w + t * KD).store(BFloat16(us[t]))
        if h == 0:
            if tid == 0:
                (oldlen_ptr + b64).store(length)


@cute.jit
def _replay_launch(
    p_raw: Int64,
    p_conv: Int64,
    p_cw: Int64,
    p_g: Int64,
    p_beta: Int64,
    p_z: Int64,
    p_nw: Int64,
    p_pool: Int64,
    p_slots: Int64,
    p_dummy: Int64,
    p_smap: Int64,
    p_hk: Int64,
    p_hu: Int64,
    p_hg: Int64,
    p_hlen: Int64,
    p_a_log: Int64,
    p_dtb: Int64,
    p_out: Int64,
    p_cand: Int64,
    p_oldlen: Int64,
    B: Int32,
    candidate_bs: Int64,
    lower: Float32,
    eps: Float32,
    stream: cuda.CUstream,
    H: cutlass.Constexpr[int],
    raw_bs: cutlass.Constexpr[int],
    raw_rs: cutlass.Constexpr[int],
    conv_ss: cutlass.Constexpr[int],
    g_bs: cutlass.Constexpr[int],
    g_rs: cutlass.Constexpr[int],
    beta_bs: cutlass.Constexpr[int],
    beta_rs: cutlass.Constexpr[int],
    z_bs: cutlass.Constexpr[int],
    z_rs: cutlass.Constexpr[int],
    pool_ss: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    BOUNDED: cutlass.Constexpr[int],
    POOL_VEC: cutlass.Constexpr[int],
    CAP: cutlass.Constexpr[int],
):
    _replay_kernel(
        p_raw,
        p_conv,
        p_cw,
        p_g,
        p_beta,
        p_z,
        p_nw,
        p_pool,
        p_slots,
        p_dummy,
        p_smap,
        p_hk,
        p_hu,
        p_hg,
        p_hlen,
        p_a_log,
        p_dtb,
        p_out,
        p_cand,
        p_oldlen,
        candidate_bs,
        lower,
        eps,
        H,
        raw_bs,
        raw_rs,
        conv_ss,
        g_bs,
        g_rs,
        beta_bs,
        beta_rs,
        z_bs,
        z_rs,
        pool_ss,
        T,
        BOUNDED,
        POOL_VEC,
        CAP,
    ).launch(
        grid=(H, B, 1),
        block=(NT, 1, 1),
        stream=stream,
        min_blocks_per_mp=5,
        preferred_smem_carveout=100,
    )


_compiled = {}


def _get_compiled(
    T,
    bounded,
    pool_vec,
    H,
    CAP,
    raw_bs,
    raw_rs,
    conv_ss,
    g_bs,
    g_rs,
    beta_bs,
    beta_rs,
    z_bs,
    z_rs,
    pool_ss,
):
    # Specialize only stable model topology and layout metadata.  Request shape B,
    # accepted lengths, slots, masks and every tensor value remain runtime inputs.
    key = (
        torch.cuda.current_device(),
        T,
        bounded,
        pool_vec,
        H,
        CAP,
        raw_bs,
        raw_rs,
        conv_ss,
        g_bs,
        g_rs,
        beta_bs,
        beta_rs,
        z_bs,
        z_rs,
        pool_ss,
    )
    fn = _compiled.get(key)
    if fn is None:
        z64 = Int64(0)
        # Runtime ABI: 20 pointers, B, candidate stride, lower, eps and stream. Every argument
        # following stream is a compile-time topology/layout constant.
        args = [z64] * 20 + [Int32(1), Int64(0), Float32(0.0), Float32(0.0)]
        fn = cute.compile(
            _replay_launch,
            *args,
            cuda.CUstream(0),
            H,
            raw_bs,
            raw_rs,
            conv_ss,
            g_bs,
            g_rs,
            beta_bs,
            beta_rs,
            z_bs,
            z_rs,
            pool_ss,
            T,
            bounded,
            pool_vec,
            CAP,
        )
        _compiled[key] = fn
    return fn


@torch.no_grad()
def fused_replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    slot_map: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    out: torch.Tensor,
    old_lengths: torch.Tensor,
    *,
    candidate_x: torch.Tensor,
    tokens_per_step: int = 1,
) -> None:
    """Write [B*T,H,128] BF16 output and candidate histories in one launch.

    Checkpoint/Conv slot strides and projection row strides are preserved.
    Contiguous K/U/G histories retain their configured capacity when T shrinks.
    Candidate Conv rows use the configured batch stride, not the current T.
    Neither the checkpoints nor the live history lengths are advanced here.
    """
    batch, heads = slots.numel(), a_log.numel()
    tokens, capacity = tokens_per_step, hk.shape[2]
    if not batch:
        return
    if not 1 <= tokens <= 8 or not HROWS + tokens <= capacity <= HROWS + 8:
        raise ValueError("2FP4 MTP replay requires 1..8 tokens and sufficient history capacity")
    if any(not tensor.is_contiguous() for tensor in (hk, hu, hg)):
        raise ValueError("2FP4 MTP histories require contiguous per-layer K/U/G storage")
    if candidate_x.stride(1) != 3 * heads * KD or candidate_x.stride(2) != 1:
        raise ValueError("Candidate Conv tokens require dense channel rows")
    if conv_state.stride(1) != 3 or conv_state.stride(2) != 1:
        raise ValueError("Conv4 state requires dense inner dimensions")
    raw_x = x.view(batch, tokens, 3 * heads * KD)
    raw_g = gate.view(batch, tokens, heads, KD)
    raw_beta = beta.view(batch, tokens, heads)
    output_gate = z.view(batch, tokens, heads, KD)
    pool_vec = int(pool.stride(0) % 16 == 0 and pool.data_ptr() % 16 == 0)
    strides = (
        raw_x.stride(0),
        raw_x.stride(1),
        conv_state.stride(0),
        raw_g.stride(0),
        raw_g.stride(1),
        raw_beta.stride(0),
        raw_beta.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        pool.stride(0),
    )
    fn = _get_compiled(tokens, int(lower_bound is not None), pool_vec, heads, capacity, *strides)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        raw_x.data_ptr(),
        conv_state.data_ptr(),
        conv_weight.data_ptr(),
        raw_g.data_ptr(),
        raw_beta.data_ptr(),
        output_gate.data_ptr(),
        norm_weight.data_ptr(),
        pool.data_ptr(),
        slots.data_ptr(),
        dummy.data_ptr(),
        slot_map.data_ptr(),
        hk.data_ptr(),
        hu.data_ptr(),
        hg.data_ptr(),
        lengths.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        out.data_ptr(),
        candidate_x.data_ptr(),
        old_lengths.data_ptr(),
        batch,
        candidate_x.stride(0),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(norm_eps),
        stream,
    )
