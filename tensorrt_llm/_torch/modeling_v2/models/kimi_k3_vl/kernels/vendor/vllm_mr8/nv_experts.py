# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed experts for MoE TP16 at 8 token rows on the vLLM MR8 Rubin engine.

``RubinNvExpertsKernel`` is vLLM MR8's ``RubinExpertsKernel`` (``experts_rubin.py``, vendored under
``kimi_k3_vl/k3_vllm/vllm_mr8``) with the checkpoint's NVFP4 experts in place of MXFP4 x
MXFP8. What stays: the static byte partition over the grid (FC1(A) / FC1(B) stages split by K with
the peer-slot exchange of FP32 partial sums, FC2 jobs split evenly with the FC2(B) dynamic pool),
the warp roles, the polled route mailbox, the call-parity hidden images and their re-arm, the
staged-row combine and its finalize into the WaveMoeTail latent mailbox. What changes:

* MMA: ``tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`` (E2M1 x E2M1, UE4M3 scales per 16
  elements), K64 per MMA: four per FC1 stage (K256), three per FC2 tile (K192). Every K64 MMA reads
  one TMEM scale chunk (4 columns, ``tcgen05.cp.32x128b.warpx4`` of a 512 B atom); scale bytes per
  weight stage 2048 (FC1) / 1536 (FC2) instead of 1024.
* Weights (``weights.pack_routed_experts_tp16``): w13 [E, 384, 1792] u8 (up / gate rows interleaved
  in the epilogue's lane order), w2 [E, 3584, 96] u8 (the K 192 slice, rows in the staged-row column
  order); scales pre-arranged per stage in the UMMA 128x4 atom order; per-expert ``alpha`` (FC1,
  FC2) = activation global dequant scale x weight global dequant scale.
* Latent image (written by the routing kernel, ``image_chunk_nv``): NVFP4 [14 stages][8 tokens]
  [128 B, 128B-swizzled] then the scale atoms [56][8 tokens][16 B] (bytes 0-3), 21504 bytes.
* Hidden: the FC1 owner applies ``alpha1``, SiTU, then quantizes each warp's 16 channels of a token
  as one NVFP4 block (global ``h_gsf``) into the compact image [8 tokens][128 B swizzled] + scales
  [8 tokens][16 B]. The images are armed with 0x88 bytes: a written FP4 byte never holds a -0
  nibble (canonicalized to +0) and a scale byte is never negative.
* FC2 applies ``alpha2`` to the accumulator before staging the BF16 rows.

The kernel arguments keep vLLM's positions; the descriptor-table position (unused with grid maps)
carries ``alphas`` (FP32 [E + 1, 2]: alpha1 / alpha2 per expert, then (h_gsf, 0)).
Only the Rubin configuration is supported: packed, grid maps, planned image, lean route, one box
per stage; FC2 die matching (``dies``) is not.

``RubinNvGroupedExpertsKernel`` does the same for vLLM's ``RubinGroupedExpertsKernel``
(``experts_rubin64.py``), the engine vLLM runs TP16 verify steps of 9-64 token rows on: each
expert's tokens in units of <= 8 rows, the activation slices gathered per weight stage from the
NVFP4 grouped plan (``plan_chunk_nv``: [64 rows][1792 B], then [64][56][16 B] scale cells), the
hidden images in the workspace's grouped half. Same weights, alphas and workspace.
"""

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int8, Int32, Int64, Uint8, Uint32, Uint64

from .specialized_decode.cutedsl.common.primitives import (  # noqa: E402
    alloc,
    asm,
    bits_float,
    commit,
    copy_evict,
    copy_g2s,
    fence_tmem,
    float_bits,
    free_tmem,
    load_tma3,
    load_volatile,
    load_words,
    named_barrier,
    redux,
    release_tmem,
    shared_address,
    situ,
    sleep_ns,
    smem_desc,
    smem_words,
    store_shared_words,
    store_words,
    thread,
    unpack_bf16,
    view,
)
from .specialized_decode.cutedsl.common.tensormap import encode_map  # noqa: E402
from .specialized_decode.cutedsl.moe.experts import claim  # noqa: E402
from .specialized_decode.cutedsl.moe.experts_rubin import (  # noqa: E402
    ROUTE_WORDS,
    ROUTED,
    TMEM_DONE,
    RubinExpertsKernel,
    claim_last,
    fence_proxy_global,
    load_acquire,
    select_u32,
    stamp_at,
    store_shared_if,
)
from .specialized_decode.cutedsl.moe.experts_rubin64 import RubinGroupedExpertsKernel  # noqa: E402

shuffle = cute.arch.shuffle_sync

# Instruction descriptor of every MMA: A / B E2M1 (MXF4 format 1, bits 7 / 10), N 8 (bits 17-22),
# UE4M3 scales (bit 23 = 0), M 128 (bits 24-28), scale chunk ids 0, K64 (bit 31 = 0).
NV_IDESC = (1 << 7) | (1 << 10) | (1 << 17) | (8 << 24)
# Latent image (the routing kernel's plan for these experts): NVFP4 [14][8][128 B swizzled], then
# the scale atoms [56 K64 blocks][8 tokens][16 B] (bytes 0-3: the four 16-element blocks' UE4M3).
IMAGE_STAGE = 1024
IMAGE_XSF = 14 * IMAGE_STAGE
IMAGE_ATOMS = 56
IMAGE_BYTES = IMAGE_XSF + IMAGE_ATOMS * 128
# Grouped plan (rows > 8, ``RubinNvGroupedExpertsKernel``; the routing kernel writes it): NVFP4 rows
# [64][1792 B] (element 2 i in the low nibble, unswizzled), then the scale cells [64][56 K64 blocks]
# [16 B] (bytes 0-3: the four 16-element blocks' UE4M3).
PLAN_ROWS = 64
PLAN_ROW_BYTES = 1792
PLAN_SCALE_ROW = IMAGE_ATOMS * 16
PLAN_XSF = PLAN_ROWS * PLAN_ROW_BYTES
PLAN_BYTES = PLAN_XSF + PLAN_ROWS * PLAN_SCALE_ROW
# Weights per expert: w13 384 rows (3 tiles x 128) x 1792 B, w2 3584 rows x 96 B; scale bytes.
W13_ROWS = 384
FC1_SCALES = 2048  # one stage: 128 rows x 16 UE4M3 (4 atoms)
FC2_SCALES = 1536  # one tile: 128 rows x 12 UE4M3 (3 atoms)
FC2_BOX = 128 * 96
W13_SCALE_BYTES = 3 * 14 * FC1_SCALES
W2_SCALE_BYTES = 28 * FC2_SCALES
# Hidden image of one unit in its workspace block (``h_compact_bytes`` = vLLM's 2176 per unit, so
# every workspace offset, the route mailbox included, stays vLLM's): FP4 [8 tokens][128 B
# swizzled] (K 0-191 written), then [8 tokens][16 B] scales (bytes 0-11).
H_IMAGE = 1024
H_UNIT_BYTES = H_IMAGE + 128
H_SMEM_SCALES = 3 * 128 + 384  # three atoms at a 128 B pitch + the last copy's 384 B overhang
ARM_BYTE = 0x88
ARM_WORD = 0x88888888
# Named barrier of the 256 epilogue threads (the alpha table is written).
ALPHAS_READY = 9


@asm(Uint32, pure=True)
def e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7, scale):
    """Eight FP32 values times ``scale`` (``mul.rn``) as packed E2M1 (element 0 lowest) with every
    -0 nibble made +0 (0x8 never appears: the hidden / latent images' arm byte is 0x88)."""
    body = "".join(f"mul.rn.f32 t{i}, ${i + 1}, $9; " for i in range(8))
    body += "".join(f"cvt.rn.satfinite.e2m1x2.f32 b{j}, t{2 * j + 1}, t{2 * j}; " for j in range(4))
    body += "mov.b32 w, {b0, b1, b2, b3}; "
    body += "and.b32 m, w, 0x77777777; add.u32 m, m, 0x77777777; and.b32 m, m, 0x88888888; "
    body += "or.b32 m, m, 0x77777777; and.b32 $0, w, m; "
    code = "{ .reg .f32 t<8>; .reg .b8 b<4>; .reg .b32 w, m; " + body + "}"
    return code, "=r," + ",".join(["f"] * 9), (v0, v1, v2, v3, v4, v5, v6, v7, Float32(scale))


@asm(Uint32, pure=True)
def e2m1_nibble(value, scale):
    """E2M1 of ``value * scale`` in bits 0-3 (-0 made +0)."""
    code = "{ .reg .pred p; .reg .f32 t; .reg .b8 b, z; .reg .b32 w; mul.rn.f32 t, $1, $2; "
    code += "cvt.rn.satfinite.e2m1x2.f32 b, 0f00000000, t; "
    code += "cvt.rn.satfinite.e2m1x2.f32 z, 0f00000000, 0f00000000; "
    code += "mov.b32 w, {b, z, z, z}; and.b32 w, w, 15; setp.eq.u32 p, w, 8; "
    code += "selp.u32 $0, 0, w, p; }"
    return code, "=r,f,f", (Float32(value), Float32(scale))


@asm((Uint32, Float32), pure=True)
def nv_scale(amax, gsf):
    """NVFP4 block scale of a 16-element block with absolute maximum ``amax`` under the global
    quantization scale ``gsf``: (UE4M3 byte of amax / 6 * gsf, the values' multiplier gsf / that
    scale, capped at FLT_MAX) -- the regular quantize kernels' arithmetic."""
    code = "{ .reg .f32 v, s, r; .reg .b16 p, hl, hh; .reg .b32 w; "
    code += "mul.f32 v, $2, 0f3E2AAAAB; mul.f32 v, v, $3; "
    code += "cvt.rn.satfinite.e4m3x2.f32 p, 0f00000000, v; cvt.u32.u16 $0, p; "
    code += "cvt.rn.f16x2.e4m3x2 w, p; mov.b32 {hl, hh}, w; cvt.f32.f16 s, hl; "
    code += "rcp.approx.ftz.f32 r, s; mul.f32 r, r, $3; min.f32 $1, r, 0f7F7FFFFF; }"
    return code, "=r,=f,f,f", (Float32(amax), Float32(gsf))


@cute.jit
def has_arm_byte(word):
    """A byte of ``word`` still holds the 0x88 arm pattern."""
    value = Uint32(word) ^ Uint32(ARM_WORD)
    return (value - Uint32(16843009)) & ~value & Uint32(2155905152) != Uint32(0)


@cute.jit
def pack_nibbles(lo, hi):
    """Two words of four nibble bytes (each 0-15, element 0 lowest) as one word of 8 nibbles."""
    a = (lo & Uint32(15)) | (lo >> 4 & Uint32(240)) | (lo >> 8 & Uint32(3840))
    a = a | (lo >> 12 & Uint32(61440))
    b = (hi & Uint32(15)) | (hi >> 4 & Uint32(240)) | (hi >> 8 & Uint32(3840))
    b = b | (hi >> 12 & Uint32(61440))
    return a | (b << 16)


@cute.jit
def image_chunk_nv(plan, token, chunk, words, gsf):
    """8-element ``chunk`` (four BF16x2 ``words``) of ``token`` as NVFP4 into the latent image:
    lanes 2k, 2k + 1 hold one 16-element block, lanes 8m..8m+7 one K64 scale atom word (the
    caller's chunks follow the lanes: chunk % 8 == lane % 8). ``gsf``: the FC1 activation global
    quantization scale (1 / the experts' input scale)."""
    word, packed = nv_chunk(words, gsf)
    stage, unit = (chunk >> 5, (chunk & 31) >> 2)
    offset = stage * IMAGE_STAGE + token * 128 + ((unit ^ token) & 7) * 16 + (chunk & 3) * 4
    store_words(plan.iterator + offset, (word,))
    if chunk % 8 == 0:
        dst = plan.iterator + IMAGE_XSF + chunk // 8 * 128 + token * 16
        store_words(dst, (packed, Uint32(0), Uint32(0), Uint32(0)))


@cute.jit
def plan_chunk_nv(plan, token, chunk, words, gsf):
    """``image_chunk_nv`` into the grouped plan (``PLAN_BYTES``; token < 64): 4 bytes at
    ``token * 1792 + chunk * 4``, the K64 block's scale word at ``PLAN_XSF + token * 896 +
    chunk // 8 * 16`` (same lane rules)."""
    word, packed = nv_chunk(words, gsf)
    store_words(plan.iterator + token * PLAN_ROW_BYTES + chunk * 4, (word,))
    if chunk % 8 == 0:
        dst = plan.iterator + PLAN_XSF + token * PLAN_SCALE_ROW + chunk // 8 * 16
        store_words(dst, (packed, Uint32(0), Uint32(0), Uint32(0)))


@cute.jit
def nv_chunk(words, gsf):
    """A lane's 8-element chunk (four BF16x2 ``words``; lanes 2k, 2k + 1 one 16-element block) as
    packed E2M1 under the block's UE4M3 scale, and the scale word of the K64 block whose chunks
    lanes 8m..8m+7 hold (valid on lane 8m)."""
    lane, values = (thread()[2], cute.make_rmem_tensor(8, Float32))
    for i in cutlass.range_constexpr(4):
        pair = unpack_bf16(words[i])
        values[2 * i], values[2 * i + 1] = (pair[0], pair[1])
    top = abs(values[0])
    for i in cutlass.range_constexpr(1, 8):
        top = cute.arch.fmax(top, abs(values[i]))
    top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, 1))
    sf, scale = nv_scale(top, gsf)
    word = e2m1x8(*[values[i] for i in range(8)], scale)
    packed = Uint32(sf)
    for s in cutlass.range_constexpr(1, 4):
        packed |= shuffle(Uint32(sf), lane + 2 * s) << 8 * s
    return word, packed


class ImagePlanNvKernel:
    """Standalone NVFP4 latent-image writer (benchmarks, tests): one CTA per token row of the BF16
    latent [batch, 3584]; args (x, plan, gsf, stream)."""

    def __init__(self, batch):
        self.batch = batch

    @cute.jit
    def __call__(self, x, plan, gsf: Float32, stream):
        self.kernel(x, plan, gsf).launch(
            grid=(self.batch, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, x, plan, gsf: Float32):
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:
                words = load_words(x.iterator + token * 3584 + chunk * 8, count=4, mutable=True)
                image_chunk_nv(plan, token, chunk, words, gsf)
        cute.arch.griddepcontrol_launch_dependents()


class PlanNvKernel:
    """Standalone grouped-plan writer (benchmarks, tests): one CTA per token row of the BF16 latent
    [batch <= 64, 3584]; args (x, plan, gsf, stream)."""

    def __init__(self, batch):
        assert batch <= PLAN_ROWS
        self.batch = batch

    @cute.jit
    def __call__(self, x, plan, gsf: Float32, stream):
        self.kernel(x, plan, gsf).launch(
            grid=(self.batch, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, x, plan, gsf: Float32):
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:
                words = load_words(x.iterator + token * 3584 + chunk * 8, count=4, mutable=True)
                plan_chunk_nv(plan, token, chunk, words, gsf)
        cute.arch.griddepcontrol_launch_dependents()


@asm()
def stage_nv(mma, weight_scales, activation_scales, release, done, *more, kind, early=False):
    """One NVFP4 stage. ``kind`` "fc1": four K64 MMAs over a K256 weight stage (the first
    accumulates iff ``release[0]``); "fc2": three K64 MMAs over a K192 tile into a fresh
    accumulator (``release[0]``: the previous unit's hidden buffer, committed first). ``mma``:
    (TMEM accumulator, weight desc, activation desc); scales: (TMEM column, desc[, activation copy
    flag]); each K64 MMA's scale chunk is 4 TMEM columns / one 512 B atom (weights: atoms at a 512 B
    pitch, activations: at a 128 B pitch, rows 0-7 used). ``release[1]``: the weight slot's empty
    barrier (``early``: released once the MMAs read A); ``done``: the accumulator's full barrier
    (0: none); ``more``: one more barrier released with the slot."""
    steps = 3 if kind == "fc2" else 4
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    free = arrive
    if early:
        free = "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        free += ".sync_restrict::shared::read::mma::a.b64"
    mma_op = "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
    code = [
        "{ .reg .pred e,p0,pt,pc,pa,ph; .reg .b32 idesc;",
        f".reg .b64 xa<{steps}>,xb<{steps}>,sa<{steps}>,sb<{steps}>;",
        f".reg .b32 ta<{steps}>,tb<{steps}>;",
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;",
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e;",
        f"mov.b32 idesc,{NV_IDESC:#x};",
    ]
    if kind == "fc1":
        code.append("setp.ne.b32 p0,$8,0;")
    else:
        code.append("setp.eq.b32 p0,0,1; setp.ne.b32 ph,$8,0; and.pred ph,ph,e;")
        code.append(f"@ph {arrive} [$8];")
    for k in range(steps):
        code.append(f"add.s64 sa{k},$4,{32 * k}; add.s64 sb{k},$6,{8 * k};")
        code.append(f"add.u32 ta{k},$3,{4 * k}; add.u32 tb{k},$5,{4 * k};")
        code.append(f"add.s64 xa{k},$1,{2 * k}; add.s64 xb{k},$2,{2 * k};")
    code += [f"@pc {cp} [tb{k}],sb{k};" for k in range(steps)]
    code += [f"@e {cp} [ta{k}],sa{k};" for k in range(steps)]
    for k in range(steps):
        code.append(
            f"@e {mma_op} [$0],xa{k},xb{k},idesc,[ta{k}],[tb{k}],{'p0' if k == 0 else 'pt'};"
        )
    code += [f"@e {free} [$9];"] + ([f"@e {free} [$11];"] if more else [])
    code += [f"@pa {arrive} [$10];", "}"]
    kinds = "rllrlrlrrrr" + "r" * len(more)
    operands = (*mma, *weight_scales, *activation_scales, *release, done, *more)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return "\n".join(code), ",".join(kinds) + ",~{memory}", values


class RubinNvExpertsKernel(RubinExpertsKernel):
    """``RubinExpertsKernel`` on NVFP4 weights and activations (see the module docstring).
    Arguments: (plan, ids, alphas, w13 scales, w2 scales, workspace, output, batch, SiTU
    scalars (beta x linear beta, 1 / beta, 1 / linear beta), [publish box], w13 address, w2
    address, stream). ``publish``: False (BF16 rows [8 x 16, 3584] per (token, top-k slot), no
    routing weights), True (``output`` is the routing weights; the latent mailbox box
    (multicast, parity, rank) follows the scalars) or "push" (vLLM's)."""

    def __init__(
        self, experts, grid, slots, *, publish=False, polled=False, static_groups=False,
        fast_exchange=False, stamp=0, grain=1, pool=1, stages=6, early=False,
    ):  # fmt: skip
        super().__init__(
            experts, grid, slots, stages=stages, early=early, stamp=stamp, grain=grain, pool=pool,
            planned=True, publish=publish, grid_maps=True, halves=True, late_push=True,
            lean_route=True, polled=polled, packed=True, static_groups=static_groups,
            fast_exchange=fast_exchange,
        )  # fmt: skip
        assert self.packed and self.parts == 1 and not self.dies
        self.w13_rows = W13_ROWS
        self.w13_scale_bytes, self.w2_scale_bytes = (W13_SCALE_BYTES, W2_SCALE_BYTES)
        assert self.h_compact_bytes >= H_UNIT_BYTES
        stages = self.stages
        # Shared memory: weight slots (16 KiB) [stages], their scales (2 KiB) [stages], the latent
        # image (14 KiB), two hidden images (1 KiB), the latent scale atoms, two hidden scale
        # images, vLLM's misc block, the alpha table [128 units][2] + h_gsf.
        self.slot_bytes = 16384
        self.off_sfa = stages * self.slot_bytes
        self.off_x = self.off_sfa + stages * FC1_SCALES
        self.off_h = self.off_x + 14 * IMAGE_STAGE
        self.off_xsf = self.off_h + 2 * H_IMAGE
        self.off_hsf = self.off_xsf + IMAGE_ATOMS * 128 + 384
        self.off_misc = self.off_hsf + 2 * H_SMEM_SCALES
        self.off_alpha = self.off_misc + 5120
        for offset in (self.off_x, self.off_h):  # 128B-swizzled operands
            assert offset % 1024 == 0
        assert self.off_misc <= 262144 - 1024
        self.smem_bytes = max(self.off_alpha + 1040, 212096)
        self.off_queue = self.off_misc + 4608
        self.off_meta = self.off_misc + 4224
        self.off_done = self.off_misc + 4224 + max(96, 16 * stages)
        assert self.off_done + 4 <= self.off_queue
        # The routing dedupe's first-occurrence table [experts] Int32 lies in the weight ring
        # (written before any weight load; warp 0 fences it before its producer lane issues).
        self.off_table = 0
        assert 4 * experts <= stages * self.slot_bytes
        # TMEM columns: accumulators 0-127, weight scales 128 + 16 slot, hidden scales
        # 128 + 16 stages + 12 buffer, latent scales 256 + 16 stage.
        self.col_wsf, self.col_hsf = (128, 128 + 16 * stages)
        self.col_xsf = 256
        assert self.col_hsf + 24 <= 256 and self.col_xsf + 16 * 14 <= 512

    # ----------------------------------------------------------------------------------------
    # routing (before the dependency wait)
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def route_init(self, sm, counters, generations=None):
        """``RubinExpertsKernel.route_init`` with the first-occurrence table in the weight ring."""
        tid, _, _, cta = thread()
        table = view(sm.iterator, self.off_table, Int32, self.experts)
        for i in range(tid, self.experts, 384):
            table[i] = Int32(2147483647)
        slot_words = view(sm.iterator, self.off_misc + 2176, Uint32, 256)
        if tid < 256:
            slot_words[tid] = Uint32(4294967295)
        if tid == 256:
            parity = view(sm.iterator, self.off_misc + 864, Int32, 1)
            parity[0] = counters[cta]
        if cutlass.const_expr(generations is not None):
            own = view(sm.iterator, self.off_misc + 4880, Int32, 1)
            if tid == 257:
                own[0] = generations[cta]
        cute.arch.sync_threads()

    @cute.jit
    def route_warp(self, sm, ids, batch):
        """``RubinExpertsKernel.route_warp`` on the ring's table; the warp then fences its generic
        accesses of the ring before the producer lane's first TMA into it."""
        lane = thread()[2]
        misc = self.off_misc
        table = view(sm.iterator, self.off_table, Int32, self.experts)
        expert, uof = (view(sm.iterator, misc + o, Int32, 128) for o in (1140, 1652))
        slots = view(sm.iterator, misc + 2176, Int8, 1024)
        base = 4 * lane
        valid = base < batch * 16
        mine = cute.make_rmem_tensor(4, Int32)
        for j in cutlass.range_constexpr(4):
            mine[j] = Int32(-1)
        if cutlass.const_expr(self.polled):  # Lamport: all ones until written
            ready = False
            while not ready:
                words = load_words(ids + base, count=4, scope="gpu")
                for j in cutlass.range_constexpr(4):
                    mine[j] = Int32(words[j])
                armed = (mine[0] == -1) | (mine[1] == -1)
                armed = armed | (mine[2] == -1) | (mine[3] == -1)
                ready = cute.arch.vote_all_sync((not valid) | (not armed))
                if not ready:
                    sleep_ns(32)
            if not valid:
                for j in cutlass.range_constexpr(4):
                    mine[j] = Int32(-1)
        elif valid:
            words = load_words(ids.iterator + base, count=4, mutable=True)
            for j in cutlass.range_constexpr(4):
                mine[j] = Int32(words[j])
        if valid:
            for j in cutlass.range_constexpr(4):
                cute.arch.atomic_min(table.iterator + mine[j], base + j, sem="relaxed", scope="cta")
        cute.arch.sync_warp()
        firstk = cute.make_rmem_tensor(4, Int32)
        rank, distinct = (Int32(0), Int32(0))
        below = (Uint32(1) << lane) - 1
        for j in cutlass.range_constexpr(4):
            firstk[j] = Int32(-1)
            if valid:
                firstk[j] = table[mine[j]]
            first = valid & (firstk[j] == base + j)
            ballot = Uint32(cute.arch.vote_ballot_sync(first))
            rank += Int32(cute.arch.popc(ballot & below))
            distinct += Int32(cute.arch.popc(ballot))
        for j in cutlass.range_constexpr(4):
            if valid & (firstk[j] == base + j):
                uof[base + j] = rank
                expert[rank] = mine[j]
                rank += 1
        cute.arch.sync_warp()
        if valid:
            for j in cutlass.range_constexpr(4):
                pair = base + j
                slots[uof[firstk[j]] * 8 + pair // 16] = Int8(pair % 16)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        return distinct

    @cute.jit
    def fill_alphas(self, sm, alphas, distinct):
        """Epilogue warps after the routing barrier: (alpha1, alpha2) of every distinct expert into
        the alpha table in unit order, and h_gsf after them; then their barrier."""
        tid = thread()[0] - 128
        experts = view(sm.iterator, self.off_misc + 1140, Int32, 128)
        table = view(sm.iterator, self.off_alpha, Float32, 258)
        for r in range(tid, distinct, 256):
            e = experts[r]
            table[2 * r] = alphas[2 * e]
            table[2 * r + 1] = alphas[2 * e + 1]
        if tid == 0:
            table[256] = alphas[2 * self.experts]
        named_barrier(ALPHAS_READY, 256)

    # ----------------------------------------------------------------------------------------
    # producer
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def load_image(self, sm, plan, cta):
        """One lane: the whole NVFP4 latent image in four bulk copies on xready[0], from stage
        ``cta`` % 14 on."""
        barrier = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 1).iterator
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, IMAGE_BYTES)
        head = cta % 14
        for part in cutlass.range_constexpr(2):
            dst, src = (sm.iterator + self.off_x, plan.iterator)
            width = IMAGE_STAGE
            if cutlass.const_expr(part == 1):
                dst, src = (sm.iterator + self.off_xsf, plan.iterator + IMAGE_XSF)
                width = 4 * 128
            cut = head * width
            copy_g2s(dst + cut, src + cut, 14 * width - cut, barrier)
            if head > 0:
                copy_g2s(dst, src, cut, barrier)

    @cute.jit
    def load_plan(self, sm, plan, shift):
        """Lanes 0-13: latent stage (``shift`` + lane) % 14's image and its four scale atoms."""
        lane = thread()[2]
        if lane < 14:
            stage = shift + lane
            if stage >= 14:
                stage -= 14
            ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
            barrier = ready.iterator + stage
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, IMAGE_STAGE + 512)
            image = sm.iterator + self.off_x + stage * IMAGE_STAGE
            copy_g2s(image, plan.iterator + stage * IMAGE_STAGE, IMAGE_STAGE, barrier)
            scales = sm.iterator + self.off_xsf + stage * 512
            copy_g2s(scales, plan.iterator + IMAGE_XSF + stage * 512, 512, barrier)

    @cute.jit
    def produce16(self, sm, maps, w13_scales, w2_scales, ranges, experts, workspace):
        """``RubinExpertsKernel.produce16`` with the NVFP4 byte counts and scale offsets."""
        stages, (w13, w2) = (self.stages, maps)
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            r, index = (tile % 3, self.first_unit(ranges, 0, group))
            expert = experts[index]
            for s in cutlass.range(lo, hi, unroll=1):
                row = expert * W13_ROWS + r * 128
                offset = Int64(expert) * W13_SCALE_BYTES + r * (14 * FC1_SCALES)
                offset += ks * FC1_SCALES
                tma = (w13, w13_scales, row, ks, offset)
                slot, phase, issued = self.issue_nv(ring, slot, phase, issued, None, tma, False)
                ks += 1
                if ks == 14:
                    ks, tile, r = (Int32(0), tile + 1, r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        expert = experts[index]
        pool = workspace.iterator + self.ws_xflag
        units = ranges[1, 1] // 42
        pooled = units * self.pool // 4 if self.grain else Int32(0)
        grain = max(self.grain, 1)
        base, grains = (28 * (ranges[1, 0] + units - pooled), 28 * pooled // grain)
        pending = claim(pool, 1)  # in flight during the static FC2 jobs
        lo, hi, step = (ranges[0, 4], ranges[0, 5], Int32(0))
        pushed, current, expert = (Int32(0), Int32(-1), Int32(0))
        active = True
        while active:
            for j in cutlass.range(lo, hi, unroll=1):
                unit, tile = (j // 28, j % 28)
                if unit != current:
                    pushed = self.push_unit(sm, pushed, unit)
                    current, expert = (unit, experts[unit])
                row = expert * 3584 + tile * 128
                offset = Int64(expert) * W2_SCALE_BYTES + tile * FC2_SCALES
                tma = (w2, w2_scales, row, Int32(0), offset)
                meta = (Int32(1), unit, tile, Int32(0))
                slot, phase, issued = self.issue_nv(ring, slot, phase, issued, meta, tma, True)
            if step == 0:
                lo, hi = (ranges[1, 4], ranges[1, 5])
            else:
                index = Int32(pending + 1)
                if index < grains:
                    pending = claim(pool, 1)
                    lo = base + index * grain
                    hi = lo + grain
                else:
                    active = False
                    store_shared_words(sm.iterator + self.off_done, (Uint32(1),))
            step += 1
        if issued >= stages:
            self.sync(barriers.iterator + stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot)
        self.push_unit(sm, pushed, Int32(-1))
        if cutlass.const_expr(self.polled):
            xcount = view(sm.iterator, self.off_misc + self.bar_xcount, Uint64, 1)
            self.sync(xcount.iterator, 0)
        if Int32(claim_last(pool + 4)) == self.grid - 2:
            store_words(pool, (Uint32(4294967295), Uint32(4294967295)))
            if cutlass.const_expr(self.polled):  # every CTA has read the mailbox
                mailbox = workspace.iterator + self.ws_route
                for i in cutlass.range_constexpr(ROUTE_WORDS // 4):
                    store_words(mailbox + 16 * i, (Uint32(4294967295),) * 4)

    @cute.jit
    def issue_nv(self, ring, slot, phase, issued, meta, tma, fc2: cutlass.Constexpr):
        """One weight stage into ``slot``: wait for it, publish the metadata (FC2), expect and
        issue the weight box (FC1 16 KiB, FC2 12 KiB) and its scale bytes (2048 / 1536)."""
        sm, barriers, metadata = ring[:3]
        desc, scales, row, chunk, offset = tma
        stages = self.stages
        if issued >= stages:
            self.sync(barriers.iterator + stages + slot, phase ^ 1)
        if cutlass.const_expr(meta is not None):
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = meta[i]
        full = barriers.iterator + slot
        size: cutlass.Constexpr = FC2_SCALES if fc2 else FC1_SCALES
        box: cutlass.Constexpr = FC2_BOX if fc2 else 16384
        cute.arch.mbarrier_arrive_and_expect_tx(full, box + size)
        load_tma3_nv(sm.iterator + slot * self.slot_bytes, desc, full, row, chunk)
        dst = sm.iterator + self.off_sfa + slot * FC1_SCALES
        copy_evict(dst, scales.iterator + offset, size, full)
        slot += 1
        if slot == stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    # ----------------------------------------------------------------------------------------
    # MMA warp
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def consume(self, sm, tmem, trace, ranges):
        """``RubinExpertsKernel.consume`` on NVFP4 stages (``stage_nv``)."""
        lane, stages = (thread()[2], self.stages)
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty, self.bar_xready)
        full, empty, accfull, accempty, hfull, hempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (stages, stages, 16, 16, 2, 2, 14))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        slot16 = self.slot_bytes // 16
        a0, asf0, x0, xsf0, h0, hsf0 = (
            smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets)
        )
        xmask, ended = (Uint32(0), False)
        slot, uses, acc = (Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        phase = Int32(0)
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            for s in cutlass.range(lo, hi, unroll=1):
                first, last = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                self.sync(full.iterator + slot, phase, True)
                if cutlass.const_expr(self.stamp > 0):
                    if (lane == 0) & (uses + (slot + phase) == 0):
                        stamp_at(trace, 7)
                weights = (tmem + self.col_wsf + slot * 16, asf0 + slot * (FC1_SCALES // 16))
                if first:
                    self.acquire(accempty, acc, uses)
                copy_scales = Uint32(0)
                if xmask >> ks & 1 == 0:
                    self.sync(xready.iterator, 0, True)  # the whole image (xearly)
                    xmask |= Uint32(1) << ks
                    copy_scales = Uint32(1)
                completion = Uint32(0)
                if last:
                    completion = shared_address(accfull.iterator + acc)
                released = shared_address(empty.iterator + slot)
                stage_nv(
                    (tmem + acc * 8, a0 + slot * slot16, x0 + ks * (IMAGE_STAGE // 16)),
                    weights,
                    (tmem + self.col_xsf + ks * 16, xsf0 + ks * 32, copy_scales),
                    (Uint32(not first), released),
                    completion,
                    kind="fc1",
                    early=self.early,
                )
                cute.arch.sync_warp()
                if cutlass.const_expr(self.stamp > 0):
                    if (lane == 0) & (uses + (slot + phase) == 0):
                        stamp_at(trace, 5)
                if last:
                    c = Int32(first) | Int32(last) << 1
                    self.signal(accmeta, accfull, acc, (Int32(0), tile, ks, c))
                    uses += 1
                    acc = uses % 16
                slot += 1
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, tile = (Int32(0), tile + 1)
        while not ended:
            self.sync(full.iterator + slot, phase, True)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            if kind == 2:
                if (lane == 0) & (current >= 0) & (hbuffer < 2):
                    commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            else:
                weights = (tmem + self.col_wsf + slot * 16, asf0 + slot * (FC1_SCALES // 16))
                copy_scales, release = (Uint32(0), Uint32(0))
                if a != current:
                    if (current >= 0) & (hbuffer < 2):
                        release = shared_address(hempty.iterator + hbuffer)
                    current = a
                    ordinal += 1
                    hbuffer = ordinal & 1
                    self.sync(hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
                    copy_scales = Uint32(1)
                hd = h0 + hbuffer * (H_IMAGE // 16)
                hs = hsf0 + hbuffer * (H_SMEM_SCALES // 16)
                sfbc = tmem + self.col_hsf + hbuffer * 12
                self.acquire(accempty, acc, uses)
                stage_nv(
                    (tmem + acc * 8, a0 + slot * slot16, hd),
                    weights,
                    (sfbc, hs, copy_scales),
                    (release, shared_address(empty.iterator + slot)),
                    shared_address(accfull.iterator + acc),
                    kind="fc2",
                    early=self.early,
                )
                cute.arch.sync_warp()
                self.signal(accmeta, accfull, acc, (kind, a, b, c))
                uses += 1
                acc = uses % 16
                slot += 1
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)

    # ----------------------------------------------------------------------------------------
    # hidden images
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        """FC1 owner: alpha1, SiTU on the up lanes (gate from lane ^ 8), and each warp's 16 hidden
        channels of a token as one NVFP4 block (``redux.sync.max`` amax, UE4M3 scale under
        h_gsf), staged as nibble bytes and stored by lanes 0-7 (one token each) as 8 bytes of the
        unit's image plus the scale byte. Hidden channel of up lane l (l & 8 == 0) of warp ``ew``
        in tile chunk ``c``: 64 c + 16 ew + 2 (l % 8) + l // 16."""
        unit = tile // self.tiles
        table = view(sm.iterator, self.off_alpha, Float32, 258)
        self.hidden_nv(sm, workspace, values, tile, parity, scalars, table[2 * unit], table[256])

    @cute.jit
    def hidden_nv(self, sm, workspace, values, tile, parity, scalars, alpha, gsf):
        """``store_hidden`` with the unit's FC1 ``alpha`` and the hidden quantization scale ``gsf``."""
        _, warp, lane, _ = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        unit, chunk = (tile // self.tiles, tile % self.tiles)
        up = Uint32(lane & 8 == 0)
        h = cute.make_rmem_tensor(self.width, Float32)
        for t in cutlass.range_constexpr(self.width):
            v = values[t] * alpha
            gate = cute.arch.shuffle_sync_bfly(v, 8)
            g = float_bits(situ(gate, v, *scalars))
            h[t] = bits_float(select_u32(up, g, 0))
        maxima = cute.make_rmem_tensor(self.width, Float32)
        for t in cutlass.range_constexpr(self.width):
            bits = Uint32(float_bits(abs(h[t])))
            bits = select_u32(Uint32(bits > Uint32(2139095040)), 0, bits)
            maxima[t] = bits_float(redux(bits, "max"))
        stage_base = self.off_misc + 3200 + (eg * 4 + ew) * 128
        nibbles = cute.make_rmem_tensor(self.width, Uint32)
        scale_token = Uint32(0)
        for t in cutlass.range_constexpr(self.width):
            sf, multiplier = nv_scale(maxima[t], gsf)
            scale_token = select_u32(Uint32(lane == t), sf, scale_token)
            nibbles[t] = e2m1_nibble(h[t], multiplier)
        for t in cutlass.range_constexpr(self.width):
            dst = sm.iterator + stage_base + t * 16 + 2 * (lane % 8) + lane // 16
            store_shared_if(dst, nibbles[t], up)
        cute.arch.sync_warp()
        if lane < 8:
            lo, hi = (Uint32(0), Uint32(0))
            if lane < self.width:
                staged = smem_words(sm.iterator + stage_base + lane * 16)
                lo, hi = (pack_nibbles(staged[0], staged[1]), pack_nibbles(staged[2], staged[3]))
            image = self.image(workspace, parity, unit)
            unit16 = 2 * chunk + ew // 2
            offset = lane * 128 + ((unit16 ^ lane) & 7) * 16 + (ew & 1) * 8
            store_words(image + offset, (lo, hi))
            byte = view(image, 0, Uint8, H_UNIT_BYTES)
            byte[H_IMAGE + lane * 16 + 4 * chunk + ew] = Uint8(scale_token)
        cute.arch.sync_warp()

    @cute.jit
    def poll_hidden(self, sm, source, buffer: Int32):
        """The unit's NVFP4 hidden image until no written byte (K 0-191, 12 scale bytes per token)
        holds the arm byte, then into hidden buffer ``buffer`` (image and three scale atoms)."""
        lane = thread()[2]
        values, scale = (cute.make_rmem_tensor((2, 4), Uint32), Uint32(0))
        ready = False
        while not ready:
            bad = False
            for v in cutlass.range_constexpr(2):
                index = lane + 32 * v
                words = load_volatile(source + index * 16, count=4)
                for w in cutlass.range_constexpr(4):
                    values[v, w] = words[w]
                token, position = (index >> 3, index & 7)
                if (position ^ token) < 6:
                    for w in cutlass.range_constexpr(4):
                        bad |= has_arm_byte(words[w])
            scale = load_volatile(source + H_IMAGE + lane * 4)[0]
            if lane & 3 < 3:
                bad |= has_arm_byte(scale)
            ready = cute.arch.vote_all_sync(not bad)
        image = sm.iterator + self.off_h + buffer * H_IMAGE
        for v in cutlass.range_constexpr(2):
            words = tuple(values[v, w] for w in range(4))
            store_shared_words(image + (lane + 32 * v) * 16, words)
        word = cute.arch.shuffle_sync(scale, 4 * (lane & 7) + (lane >> 3))
        if lane < 24:
            sf = sm.iterator + self.off_hsf + buffer * H_SMEM_SCALES
            store_shared_words(sf + (lane >> 3) * 128 + (lane & 7) * 16, (word,) * 4)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.jit
    def rearm(self, workspace, call_count):
        """Re-arm the other parity's hidden images (the previous call's) to 0x88."""
        _, _, lane, cta = thread()
        per: cutlass.Constexpr = H_UNIT_BYTES // 16
        items: cutlass.Constexpr = 128 * per
        lo, hi = (cta * items // self.grid, (cta + 1) * items // self.grid)
        other = self.image(workspace, call_count & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            unit, word = (i // per, i % per)
            store_words(
                other + Int64(unit) * self.h_compact_bytes + word * 16, (Uint32(ARM_WORD),) * 4
            )

    # ----------------------------------------------------------------------------------------
    # epilogue
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """alpha2 of the unit's expert, then ``RubinExpertsKernel.fc2``'s staged rows."""
        table = view(sm.iterator, self.off_alpha, Float32, 258)
        alpha = table[2 * unit + 1]
        for t in cutlass.range_constexpr(self.width):
            values[t] = values[t] * alpha
        RubinExpertsKernel.fc2(self, sm, workspace, output, values, unit, tile, ranges)

    # ----------------------------------------------------------------------------------------
    # launch
    # ----------------------------------------------------------------------------------------

    def encode_maps(self, w13, w2):
        """w13 [E x 384 rows, 1792 B] as 128 B x 128-row boxes of 14 chunks; w2 [E x 3584 rows,
        96 B] as one 96 B x 128-row box (both 128B-swizzled, plain bytes)."""
        types_, swizzles = (cuda.CUtensorMapDataType, cuda.CUtensorMapSwizzle)
        promotion = cuda.CUtensorMapL2promotion
        u8 = types_.CU_TENSOR_MAP_DATA_TYPE_UINT8
        swizzle = swizzles.CU_TENSOR_MAP_SWIZZLE_128B
        rows = (self.experts * W13_ROWS, self.experts * 3584)
        return (
            encode_map(
                w13, u8, (128, rows[0], 14), (1792, 128), (128, 128, 1), swizzle,
                promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            ),
            encode_map(
                w2, u8, (96, rows[1], 1), (96, 96 * rows[1]), (96, 128, 1), swizzle,
                promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            ),
        )  # fmt: skip

    @cute.jit
    def body(self, args, maps):
        """``RubinExpertsKernel.body`` (Rubin configuration) with the alpha table."""
        x, ids, alphas, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        counters = view(workspace.iterator, self.ws_counter, Int32, self.slots)
        generations = None
        if cutlass.const_expr(self.fast_exchange):
            generations = view(workspace.iterator, self.ws_generation, Int32, self.slots)
        self.route_init(sm, counters, generations)
        source = ids
        if cutlass.const_expr(self.polled):
            source = view(workspace.iterator, self.ws_route, Int32, 129).iterator
        else:
            cute.arch.griddepcontrol_wait()
        trace = workspace.iterator + self.stamp + cta * 128
        if cutlass.const_expr(self.stamp > 0):
            if tid == 0:
                stamp_at(trace, 4)
        if cutlass.const_expr(self.polled):
            if tid == 96:
                ready = False
                while not ready:
                    ready = load_acquire(source + 128) == Uint32(2 * self.capacity - 1)
                    if not ready:
                        sleep_ns(64)
                xcount = view(sm.iterator, self.off_misc + self.bar_xcount, Uint64, 1)
                cute.arch.mbarrier_arrive(xcount.iterator)
                fence_proxy_global()
        if tid == 96:
            self.load_image(sm, x, cta)
        shared_distinct = view(sm.iterator, self.off_misc + 860, Int32, 1)
        distinct = Int32(0)
        if warp == 0:
            distinct = self.route_warp(sm, source, batch)
            if lane == 0:
                shared_distinct[0] = distinct
            named_barrier(ROUTED, 384, arrive=True)
        else:
            named_barrier(ROUTED, 384)
            distinct = shared_distinct[0]
        call_count = view(sm.iterator, self.off_misc + 864, Int32, 1)[0]
        experts = view(sm.iterator, self.off_misc + 1140, Int32, 128)
        own = call_count
        if cutlass.const_expr(self.fast_exchange):
            own = view(sm.iterator, self.off_misc + 4880, Int32, 1)[0]
        if cutlass.const_expr(self.stamp > 0):
            if tid == 0:
                stamp_at(trace, 0)
        ranges = self.partition(distinct, cta)
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                self.produce16(*weights, workspace)
                counters[cta] = call_count + 1
                if cta < self.slots - self.grid:
                    counters[self.grid + cta] = call_count + 1
                if cutlass.const_expr(self.fast_exchange):
                    generations[cta] = own + 1
                    if cta < self.slots - self.grid:
                        generations[self.grid + cta] = own + 1
        else:
            if warp == 1:
                self.consume(sm, tmem, trace, ranges)
                fence_tmem(before=True)
                named_barrier(TMEM_DONE, 320, arrive=True)
            elif warp == 2:
                self.load_hidden16(sm, workspace, own & 1)
            elif warp == 3:
                self.rearm(workspace, own)
                if cutlass.const_expr(self.fast_exchange):
                    self.rearm_peers(workspace, own & 1 ^ 1, cta)
                if cutlass.const_expr(self.polled):
                    cute.arch.griddepcontrol_wait()  # the shared rows
                if cutlass.const_expr(self.publish == "push" and self.late_push):
                    ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
                    self.sync(ready.iterator, 0)
                    self.push_shared(scalars[3:9], batch, lane, cta, 32)
                named_barrier(TMEM_DONE, 320)
                fence_tmem()
                free_tmem(tmem, 512)
            else:
                self.fill_alphas(sm, alphas, distinct)
                self.epilogue(sm, workspace, output, tmem, ranges, batch, own & 1, scalars)
        if cutlass.const_expr(self.stamp > 0):
            if (lane == 0) & ((warp < 2) | (warp == 4)):
                stamp_at(trace, 1 + warp - 2 * Int32(warp == 4))


class RubinNvGroupedExpertsKernel(RubinGroupedExpertsKernel):
    """vLLM MR8's ``RubinGroupedExpertsKernel`` (``experts_rubin64.py``: TP16, 9-64 token rows) on
    the NVFP4 weights and activations of ``RubinNvExpertsKernel``. What stays: the grouping (each
    expert's tokens in units of <= 8 rows, units in expert order), the static FC1(A) / FC1(B) K-split
    partition with partial sums through the workspace, FC2 jobs split evenly with the claimed FC2(B)
    tail pool, the per-slot activation slices gathered from the plan by warp 3, the hidden images in
    the grouped half of the workspace (alternating on the grouped call generation), the lean staged
    rows and the finalize. What changes is what ``RubinNvExpertsKernel`` changes: ``mxf4nvf4`` K64
    MMAs (four per FC1 stage, three per FC2 job), 2 KiB / 1.5 KiB weight scales per stage, the
    NVFP4 grouped plan (``PLAN_BYTES``; slices of 8 rows x 128 B and four scale atoms per stage), the
    hidden quantized to NVFP4 after alpha1 and SiTU (0x88-armed images), alpha2 on FC2.
    Arguments: (plan, ids, alphas, w13 scales, w2 scales, workspace, output, batch, SiTU scalars,
    [publish box], w13 address, w2 address, stream), the weights as ``RubinNvExpertsKernel``'s.
    FC2 die matching (``dies``) is not supported."""

    def __init__(
        self, experts, grid, slots, *, publish=False, static_groups=False, pool=(12, 6), stages=7,
        trace=0,
    ):  # fmt: skip
        super().__init__(
            experts, grid, slots, publish=publish, stages=stages, trace=trace, pool=pool, dies=(),
            static_groups=static_groups,
        )  # fmt: skip
        self.w13_rows = W13_ROWS
        self.w13_scale_bytes, self.w2_scale_bytes = (W13_SCALE_BYTES, W2_SCALE_BYTES)
        self.w2_tile_scale_bytes = FC2_SCALES
        assert self.h_compact_bytes >= H_UNIT_BYTES and self.slot_bytes == 16384
        # Shared memory: weight slots (16 KiB) [stages], their scales (2 KiB) [stages], the slots'
        # activation slices (1 KiB, 128B-swizzled) [stages], two hidden images (1 KiB), the slices'
        # scale atoms (4 x 128 B) [stages] + the last copy's overhang, two hidden scale images,
        # vLLM's misc block, then (vLLM's) the unit tables, the grouping's scratch, the FC2
        # transpose buffers, and the alpha table [experts + 1][2].
        self.off_sfa = stages * self.slot_bytes
        self.off_x = self.off_sfa + stages * FC1_SCALES
        self.off_h = self.off_x + stages * IMAGE_STAGE
        self.off_xsf = self.off_h + 2 * H_IMAGE
        self.off_hsf = self.off_xsf + stages * 512 + 384
        self.off_misc = self.off_hsf + 2 * H_SMEM_SCALES
        for offset in (self.off_x, self.off_h):  # 128B-swizzled operands
            assert offset % 1024 == 0
        assert self.off_misc <= 262144 - 1024
        self.off_tables = (self.off_misc + 5120 + 127) // 128 * 128
        self.off_masks = self.off_tables + 4096 + 16384
        self.off_bases = self.off_masks + 8 * experts
        self.off_totals = self.off_bases + 4 * experts
        self.off_pairs = self.off_totals + 64
        end = self.off_pairs + 4096
        self.off_fc2s = (end + 127) // 128 * 128
        self.off_alpha = self.off_fc2s + 8 * 512
        end = self.off_alpha + 8 * (experts + 1)
        assert end <= 227 * 1024
        self.smem_bytes = max(end, 212096)
        self.off_meta = self.off_misc + 2560
        self.off_queue = self.off_misc + 4608
        # TMEM columns: accumulators 0-127, weight scales 128 + 16 slot, activation scales
        # 256 + 16 slot, hidden scales 256 + 16 stages + 12 buffer.
        self.col_wsf, self.col_xsf = (128, 256)
        self.col_hsf = 256 + 16 * stages
        assert 128 + 16 * stages <= 256 and self.col_hsf + 24 <= 512

    encode_maps = RubinNvExpertsKernel.encode_maps
    poll_hidden = RubinNvExpertsKernel.poll_hidden
    hidden_nv = RubinNvExpertsKernel.hidden_nv

    # ----------------------------------------------------------------------------------------
    # producer and gather warp
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def fetch(self, sm, desc, scales, full, slot, row, chunk, offset, size, more=None):
        """One packed weight box and its ``size`` scale bytes (2 KiB pitch) into ``slot``."""
        dst = sm.iterator + slot * self.slot_bytes
        load_tma3(dst, desc, full.iterator + slot, row, chunk)
        dst = sm.iterator + self.off_sfa + slot * FC1_SCALES
        copy_evict(dst, scales.iterator + offset, size, full.iterator + slot)

    @cute.jit
    def fc2_job(self, ring, state, maps, experts, ranges, j, pool: cutlass.Constexpr):
        """``RubinGroupedExpertsKernel.fc2_job`` with the NVFP4 tile's scale bytes."""
        slot, phase, issued, pushed, current = state
        unit, job = (j // self.jobs, j % self.jobs)
        if unit != current:
            pushed = self.push_unit(ring[0], pushed, unit)
            current = unit
        expert = experts[unit]
        row = expert * 3584 + job * 128
        offset = Int64(expert) * W2_SCALE_BYTES + job * FC2_SCALES
        tma = (maps[1], ring[3], row, Int32(0), offset, Int32(FC2_SCALES))
        meta = (Int32(1), unit, job, Int32(0)) if pool else None
        slot, phase, issued = self.issue(ring, slot, phase, issued, meta, tma, True)
        return slot, phase, issued, pushed, current

    @cute.jit
    def stream(self, sm, maps, w13_scales, w2_scales, ranges, experts, steal=None):
        """``RubinGroupedExpertsKernel.stream`` with the NVFP4 stage's scale bytes."""
        w13, w2 = maps
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * self.stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            r, index = (tile % 3, self.first_unit(ranges, 0, group))
            expert = experts[index]
            for _ in cutlass.range(lo, hi, unroll=1):
                row = expert * W13_ROWS + r * 128
                offset = Int64(expert) * W13_SCALE_BYTES + r * (14 * FC1_SCALES)
                offset += ks * FC1_SCALES
                tma = (w13, w13_scales, row, ks, offset, Int32(FC1_SCALES))
                slot, phase, issued = self.issue(ring, slot, phase, issued, None, tma, False)
                ks += 1
                if ks == 14:
                    ks, tile, r = (Int32(0), tile + 1, r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        expert = experts[index]
        state = (slot, phase, issued, Int32(0), Int32(-1))
        work = (ring, state, maps, experts, ranges, *steal)
        slot, phase, issued = self.fc2_pool(*work)
        if issued >= self.stages:
            self.sync(barriers.iterator + self.stages + slot, phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + slot, arrive_count=33)

    @cute.jit
    def plan_sources(self, rows, index, sources):
        """Byte offsets in the plan of this lane's two data rows (rows lane // 8 and lane // 8 + 4
        of unit ``index``) and of its scale row (row lane % 8); padded rows repeat the unit's
        first."""
        lane = thread()[2]
        for j in cutlass.range_constexpr(3):
            row = (lane >> 3) + 4 * j if j < 2 else lane & 7
            pair = Int32(rows[index * 8 + row])
            if pair < 0:
                pair = Int32(rows[index * 8])
            sources[j] = (pair >> 4) * (PLAN_ROW_BYTES if j < 2 else PLAN_SCALE_ROW)

    @cute.jit
    def gather_x(self, sm, plan, ranges, batch):
        """Warp 3: each FC1 stage's activation slice (the unit's 8 rows x K256: two 16 B copies per
        lane) and its four scale atoms (one 16 B cell per lane) from the plan into the stage's slot
        (cp.async on the slot's full barrier)."""
        lane, ws = (thread()[2], plan.iterator)
        stages, rows = (self.stages, self.tables(sm)[1])
        full, empty = (
            view(sm.iterator, self.off_misc + o, Uint64, stages)
            for o in (self.bar_full, self.bar_empty)
        )
        chunk, low = (lane & 7, lane >> 3)
        image = [
            self.off_x + (low + 4 * j) * 128 + ((chunk ^ (low + 4 * j)) & 7) * 16 for j in range(2)
        ]
        scale = self.off_xsf + low * 128 + chunk * 16
        sources = cute.make_rmem_tensor(3, Int32)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, r = (lo % 14, lo // 14 % 3)
            index = self.first_unit(ranges, 0, group)
            if lo < hi:
                self.plan_sources(rows, index, sources)
            for s in cutlass.range(lo, hi, unroll=1):
                if issued >= stages:
                    self.sync(empty.iterator + slot, phase ^ 1)
                for j in cutlass.range_constexpr(2):
                    src = sources[j] + ks * 128 + chunk * 16
                    copy_g2s(sm.iterator + image[j] + slot * IMAGE_STAGE, ws + src)
                src = PLAN_XSF + sources[2] + (4 * ks + low) * 16
                copy_g2s(sm.iterator + scale + slot * 512, ws + src)
                cute.arch.cp_async_mbarrier_arrive_noinc(full.iterator + slot)
                slot, issued = (slot + 1, issued + 1)
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, r = (Int32(0), r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        if s + 1 < hi:
                            self.plan_sources(rows, index, sources)

    # ----------------------------------------------------------------------------------------
    # MMA warp
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def consume(self, sm, tmem, ranges):
        """``RubinGroupedExpertsKernel.consume`` on NVFP4 stages (``stage_nv``): FC1 stages read
        the slot's weights and activation slice (both scale sets copied per stage)."""
        lane, stages = (thread()[2], self.stages)
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty)
        full, empty, accfull, accempty, hfull, hempty = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (stages, stages, 16, 16, 2, 2))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        a0, asf0, x0, xsf0, h0, hsf0 = (
            smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets)
        )
        slot16 = self.slot_bytes // 16
        uses, acc, count = (Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        for g in cutlass.range_constexpr(2):  # FC1(A), FC1(B): the producer's order
            lo, hi = (ranges[g, 2], ranges[g, 3])
            ks = lo % 14
            for s in cutlass.range(lo, hi, unroll=1):
                slot = count % stages
                self.sync(full.iterator + slot, count // stages & 1, True)
                weights = (tmem + self.col_wsf + slot * 16, asf0 + slot * (FC1_SCALES // 16))
                released = shared_address(empty.iterator + slot)
                first, last = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                if first:
                    self.acquire(accempty, acc, uses)
                # The slot's own activation slice (its cp.async completed full[slot]).
                cute.arch.fence_proxy("async.shared", space="cta")
                completion = Uint32(0)
                if last:
                    completion = shared_address(accfull.iterator + acc)
                stage_nv(
                    (tmem + acc * 8, a0 + slot * slot16, x0 + slot * (IMAGE_STAGE // 16)),
                    weights,
                    (tmem + self.col_xsf + slot * 16, xsf0 + slot * 32, Uint32(1)),
                    (Uint32(not first), released),
                    completion,
                    kind="fc1",
                )
                cute.arch.sync_warp()
                if last:  # the commit signals the epilogue
                    uses += 1
                    acc = uses % 16
                count += 1
                ks += 1
                if ks == 14:
                    ks = Int32(0)
        bars = (full, empty, accempty, accfull, hfull, hempty)
        mma = (tmem, a0, asf0, h0, hsf0, bars)
        slot, phase = (count % stages, count // stages & 1)
        state = (slot, phase, uses, acc, current, ordinal, hbuffer, Int32(0))
        zone = self.zone_jobs(ranges)
        for g in cutlass.range_constexpr(2):
            end = ranges[g, 5] - (zone if g == 1 else 0)
            for j in cutlass.range(ranges[g, 4], end, unroll=1):
                self.sync(full.iterator + state[0], state[1], True)
                state = self.fc2_mma(state, mma, j // self.jobs)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        ended = False
        while not ended:
            self.sync(full.iterator + state[0], state[1], True)
            kind, a, b = (meta[4 * state[0] + i] for i in range(3))
            if kind != 1:  # the end marker
                ended = True
            else:
                state = self.fc2_mma(state, mma, a, (accmeta, pmeta, b))
        slot, phase, uses, acc, current, ordinal, hbuffer, _ = state
        for _ in cutlass.range(2, unroll=1):
            self.acquire(accempty, acc, uses)
            self.signal(accmeta, pmeta, acc, (Int32(2), Int32(0), Int32(0), Int32(0)))
            uses += 1
            acc = uses % 16
        if (lane == 0) & (current >= 0) & (hbuffer < 2):
            commit(hempty.iterator + hbuffer)

    @cute.jit
    def fc2_mma(self, state, mma, unit, pool=None):
        """``RubinGroupedExpertsKernel.fc2_mma`` on an NVFP4 stage (three K64 MMAs)."""
        slot, phase, uses, acc, current, ordinal, hbuffer, ahead = state
        tmem, a0, asf0, h0, hsf0, bars = mma
        full, empty, accempty, accfull, hfull, hempty = bars
        stages = self.stages
        weights = (tmem + self.col_wsf + slot * 16, asf0 + slot * (FC1_SCALES // 16))
        released = shared_address(empty.iterator + slot)
        copy_scales, release = (Uint32(0), Uint32(0))
        if unit != current:
            if (current >= 0) & (hbuffer < 2):
                release = shared_address(hempty.iterator + hbuffer)
            current = unit
            ordinal += 1
            hbuffer = ordinal & 1
            self.sync(hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
            copy_scales = Uint32(1)
        hd = h0 + hbuffer * (H_IMAGE // 16)
        hs = hsf0 + hbuffer * (H_SMEM_SCALES // 16)
        sfbc = tmem + self.col_hsf + hbuffer * 12
        if cutlass.const_expr(pool is None):
            if ahead == 0:
                for i in cutlass.range_constexpr(6, 8):
                    self.acquire(accempty, (uses + i) % 16, uses + i)
                ahead = Int32(8)
            ahead -= 1
        else:
            self.acquire(accempty, acc, uses)
        slot16 = self.slot_bytes // 16
        stage_nv(
            (tmem + acc * 8, a0 + slot * slot16, hd),
            weights,
            (sfbc, hs, copy_scales),
            (release, released),
            shared_address(accfull.iterator + acc),
            kind="fc2",
        )
        cute.arch.sync_warp()
        if cutlass.const_expr(pool is not None):
            accmeta, pmeta, tile = pool
            self.signal(accmeta, pmeta, acc, (Int32(1), unit, tile, Int32(0)))
        uses += 1
        acc = uses % 16
        slot += 1
        if slot == stages:
            slot, phase = (Int32(0), phase ^ 1)
        return (slot, phase, uses, acc, current, ordinal, hbuffer, ahead)

    # ----------------------------------------------------------------------------------------
    # epilogue and hidden images
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def fill_alphas(self, sm, alphas):
        """Epilogue warps: the whole alpha table (per expert, then h_gsf) into shared memory, then
        their barrier."""
        tid = thread()[0] - 128
        count: cutlass.Constexpr = 2 * (self.experts + 1)
        table = view(sm.iterator, self.off_alpha, Float32, count)
        for i in range(tid, count, 256):
            table[i] = alphas[i]
        named_barrier(ALPHAS_READY, 256)

    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        """``RubinNvExpertsKernel.store_hidden`` with the alpha of the unit's expert."""
        expert = self.tables(sm)[0][tile // self.tiles]
        table = view(sm.iterator, self.off_alpha, Float32, 2 * (self.experts + 1))
        alpha, gsf = (table[2 * expert], table[2 * self.experts])
        self.hidden_nv(sm, workspace, values, tile, parity, scalars, alpha, gsf)

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """alpha2 of the unit's expert, then ``RubinGroupedExpertsKernel.fc2``'s staged rows."""
        expert = self.tables(sm)[0][unit]
        table = view(sm.iterator, self.off_alpha, Float32, 2 * (self.experts + 1))
        alpha = table[2 * expert + 1]
        for t in cutlass.range_constexpr(self.width):
            values[t] = values[t] * alpha
        RubinGroupedExpertsKernel.fc2(self, sm, workspace, output, values, unit, tile, ranges)

    @cute.jit
    def rearm(self, workspace, call_count):
        """Re-arm the hidden images the other parity's last grouped call used to 0x88, and that
        parity's FC2(B) pool counters to -1."""
        _, _, lane, cta = thread()
        per: cutlass.Constexpr = H_UNIT_BYTES // 16
        last = view(workspace.iterator, self.ws_calls + 4 * self.slots, Int32, 2)
        used = max(last[call_count & 1 ^ 1], 0) * per
        lo, hi = (cta * used // self.grid, (cta + 1) * used // self.grid)
        other = self.image(workspace, call_count & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            unit, word = (i // per, i % per)
            store_words(
                other + Int64(unit) * self.h_compact_bytes + word * 16, (Uint32(ARM_WORD),) * 4
            )
        base = self.ws_steal + 4 * self.slots * (call_count & 1 ^ 1)
        counters = view(workspace.iterator, base, Int32, self.slots)
        if lane == 0:
            counters[cta] = Int32(-1)
            if cta < self.slots - self.grid:
                counters[self.grid + cta] = Int32(-1)

    # ----------------------------------------------------------------------------------------
    # launch
    # ----------------------------------------------------------------------------------------

    @cute.jit
    def body(self, args, maps):
        """``RubinGroupedExpertsKernel.body`` with the alpha table (``args[2]``)."""
        x, ids, alphas, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.griddepcontrol_wait()
        counters = view(workspace.iterator, self.ws_counter, Int32, self.slots)
        routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        self.group(sm, ids, workspace, batch)
        distinct, call_count, generation = (routed[0], routed[1], routed[2])
        experts = self.tables(sm)[0]
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        ranges = self.partition(distinct, cta)
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                self.stream(*weights, (workspace, generation & 1))
        else:
            if warp == 1:
                self.consume(sm, tmem, ranges)
            elif warp == 2:
                self.load_hidden(sm, workspace, ranges, generation & 1)
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            elif warp == 3:
                self.gather_x(sm, x, ranges, batch)
                self.rearm(workspace, generation)
                if lane == 0:
                    self.claims(sm, workspace, ranges, generation & 1)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:])
            else:
                if cutlass.const_expr(self.publish == "push"):  # before any accumulator
                    self.push_shared(scalars[3:9], batch, tid - 128, cta, 256)
                self.fill_alphas(sm, alphas)
                self.epilogue(sm, workspace, output, tmem, ranges, batch, generation & 1, scalars)
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            counters[cta] = call_count + 1
            if cta < self.slots - self.grid:
                counters[self.grid + cta] = call_count + 1
            grouped = view(workspace.iterator, self.ws_calls, Int32, self.slots + 2)
            grouped[cta] = generation + 1
            if cta < self.slots - self.grid:
                grouped[self.grid + cta] = generation + 1
            if cta == 0:
                grouped[self.slots + (generation & 1)] = distinct


@asm()
def load_tma3_nv(dst, descriptor, barrier, row, chunk):
    """``primitives.load_tma3`` (evict-first)."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes.L2::cache_hint [$0],[$1,{0,$3,$4}],[$2],pol; }"
    return code, "r,l,r,r,r", (dst, descriptor, barrier, Int32(row), Int32(chunk))
