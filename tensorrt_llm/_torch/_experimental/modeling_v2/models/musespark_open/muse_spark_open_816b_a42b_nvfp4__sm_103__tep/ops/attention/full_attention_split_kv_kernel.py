# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel behind ``full_attention_split_kv``: page-split decode attention partials (tcgen05, sm_103).

Only ``full_attention_split_kv`` imports this module, and only when the op is called, so nothing here
loads with the target. The op module states the contract, the ownership and progress argument
and the PDL memory-order record; the comments here say what each role does.

Shape per rank (``kv_heads``, one build per tensor-parallel size of the target): 2 KV heads and
16 query heads at tep8, 4 KV heads and 32 query heads at tep4, all of width 64; M <= 8 rows (the
decode step and its drafts, causal among themselves); FP8 E4M3 paged K/V ``[blocks, kv_heads,
32, 64]`` with separate K and V page ids.

A CTA serves one pair of KV heads (``blockIdx.y``: the pair, heads ``2 pair`` and ``2 pair + 1``
and their 16 query heads) over a contiguous range of the sequence's 32-key pages (an even split
of the pages computed on the device from ``seq_lens``, so one compiled kernel and one captured
grid serve every context); the grid is ``splits x kv_heads / 2``. Everything a CTA computes is
the tep8 build's computation on its pair's slice of Q, of the pool and of the partials. The 128
query rows of a CTA are the 64 rows of each KV head of its pair
(row = head * 64 + token * 8 + q-head-in-group); Q' is block-diagonal over the two 64-wide head
slices of the 128-wide K dimension, so S = Q' K'^T computes every row against its own head's
keys in one M=128 MMA per 128-key tile, and O = P V' gives each row its own head's 64 output
columns. Two softmax warpgroups take alternating 128-key tiles with their own running max/sum and TMEM O
accumulator, then merge in the CTA into one partial: normalized O in FP16 and a log2-sum-exp,
stored per (token, query head) row and split. There is no combine here: the consumer (the
attention norm-gate, ``combine_norm_gate``) merges the splits.

Before its dependency wait a CTA reads only data that no kernel of the step writes: the
sequence length, the KV scale and its page ids (staged in shared memory with one coalesced
pass). Every page, Q and every store come after the wait. (``prefetch``, which the op does not
build, also streams the pages that hold no key of the step's own rows before the wait.)

Every tensor is built here from a raw pointer and explicit extents and strides, so the compiled
kernel never infers a layout from the first call's tensors.

``FullAttnSplitKvFoldKernel`` (the ``full_attention_prep_fold`` op) is the same kernel with
the folded prep (``preattn_prep_append``) taken in: ``qkv`` is the raw QKV projection row (``[q | gate]`` per head
then K and V, or the permuted ``[q | k | v | gate]``), the softmax threads apply the prep's
weightless RMSNorm and query pre-scale to their Q dims themselves (the prep's operation order,
one PTX operation per step), and the two warps the op's build leaves idle append the step's K/V
into the FP8 cache, in the CTA whose pages hold them, before that CTA's loader streams those
pages. Every one of its threads executes ``griddepcontrol.wait`` before it exits. Its
fold-only code sits under ``const_expr`` in the shared roles, so the op's build is unchanged.
"""

import copy
import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op

F8 = cutlass.Float8E4M3FN
BF16 = cutlass.BFloat16
F16 = cutlass.Float16

#: Tokens per KV page, head width, KV heads and query heads per KV head of one CTA (one pair
#: of KV heads: all of one tep8 rank's, half of one tep4 rank's).
PAGE = 32
D = 64
HKV = 2
G = 8
HQ = HKV * G
#: KV heads of one rank the kernel is built for: tep8 (2: one pair) and tep4 (4: two pairs).
RANK_KV_HEADS = (2, 4)
#: Width of the tep8 fused-QKV operand row: 16 query heads, then 2 K heads, then 2 V heads.
QKV_WIDTH = (HQ + 2 * HKV) * D
#: Width of the tep8 raw QKV projection row the fold build reads: 16 query heads and their 16
#: gate heads, 2 K heads, 2 V heads.
RAW_WIDTH = 2 * (HQ + HKV) * D
#: The fold build's append threads (warps 10 and 11): arrivals on its ``app`` barrier.
APPEND_THREADS = 64
#: Query rows the row map is written for: one accepted token and up to 7 drafts.
MAX_ROWS = 8
PAGES_PER_TILE = 4
TILE_KEYS = PAGE * PAGES_PER_TILE
ROWS = HKV * D  # 128 query rows per CTA (64 per KV head)
#: Page ids one CTA stages in shared memory (per K and per V table) per staging pass; a build
#: stages ``max_cta_pages``, a multiple of it (the op's builds: one pass at tep8, two at tep4).
MAX_CTA_PAGES = 256
#: O columns the rescale may hold in registers at once (``rescale_cols``; D = the whole head).
RESCALE_COLS = (16, 32, D)
#: softmax scale 1/sqrt(64) in log2 units.
SM_SCALE_LOG2 = math.log2(math.e) / math.sqrt(D)
#: ``CacheHintSm90::EVICT_FIRST``: the TMA loads' L2 eviction priority (each K/V byte is read
#: once per step, so it should not displace the weights the next kernels read).
EVICT_FIRST_POLICY = 0x12F0000000000000

try:  # tcgen05.fence::{before,after}_thread_sync
    from cutlass.experimental.primitives.nvvm_wrapper import tcgen05_fence as _tcgen05_fence
except ImportError:  # pragma: no cover - older DSL
    _tcgen05_fence = None


def _fence_before():
    if _tcgen05_fence is not None:
        _tcgen05_fence("before_thread_sync")


def _fence_after():
    if _tcgen05_fence is not None:
        _tcgen05_fence("after_thread_sync")


def _asm(ret, args, text, cons, loc, ip):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def pack_e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """Four f32 -> four E4M3 (RN, satfinite) in one 32-bit word, a in the low byte."""
    return Int32(
        _asm(
            T.i32(),
            [Float32(a), Float32(b), Float32(c), Float32(d)],
            "{\n .reg .b16 lo, hi;\n"
            " cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            " cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
            " mov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def split_hi(x, *, loc=None, ip=None):
    """Veltkamp split: the nearest value with 4 significant bits (an E4M3 normal number).

    PTX with explicit .rn operations, so no reassociation can fold c - (c - x) into x.
    """
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "{\n .reg .f32 c, t;\n mul.rn.f32 c, $1, 0f49800008;\n"
            " sub.rn.f32 t, c, $1;\n sub.rn.f32 $0, c, t;\n}",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fmax3(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b), Float32(c)],
            "max.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def log2_approx(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(x)], "lg2.approx.ftz.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def rcp_approx(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(x)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc, ip))


# The fold build's norm arithmetic, one PTX operation each, so nothing is reassociated or
# contracted differently from ``preattn_prep`` (whose BF16 squares are exact in fp32).
@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [Float32(a), Float32(b)], "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def rsqrt_approx_ftz(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(x)], "rsqrt.approx.ftz.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def round_bf16(x, *, loc=None, ip=None):
    """x rounded to BF16 (round to nearest even), returned as fp32."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "{\n .reg .b16 h;\n cvt.rn.bf16.f32 h, $1;\n cvt.f32.bf16 $0, h;\n}",
            "=f,f",
            loc,
            ip,
        )
    )


def _private_mma(mma):
    """A copy of a tiled MMA whose ACCUMULATE field can be set without touching the original.

    ``TiledMma.set`` rebinds the IR value held by the atom's trait; giving the copy its own trait
    keeps a value defined inside one control-flow region from leaking into another.
    """
    m = copy.copy(mma)
    m._trait = copy.copy(mma._trait)
    return m


def _ab_layout(tiled_mma, tiler, operand, kind, stages):
    if operand == "A":
        shape = tiled_mma.partition_shape_A(cute.dice(tiler, (1, None, 1)))
        is_k = tiled_mma.op.a_major_mode == OperandMajorMode.K
    else:
        shape = tiled_mma.partition_shape_B(cute.dice(tiler, (None, 1, 1)))
        is_k = tiled_mma.op.b_major_mode == OperandMajorMode.K
    atom = sm100_utils.make_smem_layout_atom(kind, F8)
    shape = cute.append(shape, stages)
    order = (1, 2, 3) if is_k else (2, 1, 3)
    return tcgen05.tile_to_mma_shape(atom, shape, order=order)


class FullAttnSplitKvKernel:
    """Page-split partials of the full-attention decode, one CTA per split, both KV heads.

    Q = Q_hi + Q_lo, two E4M3 terms (BF16-class Q); P is one E4M3 term. ``enable_pdl``
    compiles the dependency wait and the trigger (and nothing global is touched before the wait
    except the immutable reads listed in the op's record). ``prefetch`` lets the loader start the
    TMA ring on immutable pages before the wait (the op does not build it: it does not shorten
    the critical path); ``evict_first`` sets the TMA loads' L2 policy. ``loader_wait=False`` is a
    negative control for the PDL tests only: the loader never waits, so it can stream the step's
    own pages before the predecessor wrote them (the op never builds it).
    ``k_tile`` (the op's build) scores a 128-key tile with one N=128 MMA per Q term and K block
    over a ``[head][128 keys][64 B]`` K slot filled by per-head 2 KB boxes; without it, one N=32
    MMA set per page over 4 KB K boxes. ``p_split`` (the op's build) releases P of a tile in
    two halves of 64 keys (``p_full``, then ``p_full2``), so the P.V of pages 0-1 runs while the
    warpgroup converts pages 2-3, and sums the row after the release. Neither changes a value:
    ``k_tile=False`` and ``p_split=False`` are slower, the op builds neither, and the op test
    checks that each gives the op's build's bits.
    ``rescale_cols`` is how many O columns the (rare) rescale of a warpgroup's accumulator holds
    in registers at once. At 64 (the whole head, the default build) those 64 registers are live
    next to the tile's 128 S values under the softmax warps' 208-register cap, and ptxas spills
    part of S to local memory on every tile; at 16 (the spill-free build) or 32 the rescale loads,
    scales and stores O one chunk at a time and the softmax warps do not spill. The same
    multiply per element, so the same values.
    """

    def __init__(
        self,
        enable_pdl: bool,
        prefetch: bool = True,
        evict_first: bool = True,
        k_tile: bool = True,
        stages: int = 16,
        rescale_log2: float = 4.0,
        p_pre_log2: float = 4.0,
        regs_softmax: int = 208,
        regs_other: int = 64,
        min_pages: int = 4,
        loader_wait: bool = True,
        p_split: bool = True,
        rescale_cols: int = D,
        kv_heads: int = HKV,
        max_cta_pages: int = MAX_CTA_PAGES,
    ):
        self.enable_pdl = enable_pdl
        self.prefetch = prefetch and enable_pdl
        self.loader_wait = loader_wait
        self.evict_first = evict_first
        self.k_tile = k_tile
        assert stages % PAGES_PER_TILE == 0, stages
        self.stages = stages
        self.tau = rescale_log2
        self.pre = p_pre_log2
        self.nq = 2
        self.mma_warp = 8
        self.load_warp = 9
        self.threads = 32 * 12  # the remaining warps idle (they complete the third warpgroup)
        self.tmem_warps = 9  # the softmax warps and the MMA warp
        self.regs_softmax = regs_softmax
        self.regs_other = regs_other
        self.min_pages = min_pages
        self.tmem_cols = 512
        # P of a tile goes to TMEM in two halves of 64 keys, each released on its own
        # barrier, so the P.V of pages 0-1 runs while the warpgroup still converts pages 2-3;
        # the row sum follows the release. Same values, same order.
        self.p_split = p_split
        assert rescale_cols in RESCALE_COLS, rescale_cols
        self.rescale_cols = rescale_cols
        # ``FullAttnSplitKvFoldKernel`` sets these: ``fold = (eps, q_scale, permuted)`` and the raw
        # projection's row width. None / the fused operand's width: the op's build.
        self.fold = None
        self.app_wait = True
        # The rank's KV heads (their pairs on grid y) and query heads, its operand widths, and
        # the page ids a CTA stages (whole passes of MAX_CTA_PAGES).
        assert kv_heads in RANK_KV_HEADS, kv_heads
        assert max_cta_pages % MAX_CTA_PAGES == 0, max_cta_pages
        self.kv_heads = kv_heads
        self.pairs = kv_heads // HKV
        self.hq = kv_heads * G
        self.max_cta_pages = max_cta_pages
        self.qkv_width = (self.hq + 2 * kv_heads) * D
        self.raw_width = 2 * (self.hq + kv_heads) * D

    def pair_index(self):
        """This CTA's pair of KV heads: ``blockIdx.y``, or the constant 0 of a one-pair build
        (tep8), whose index arithmetic then folds to the one-pair kernel's."""
        if self.pairs == 1:
            return 0
        _, pair, _ = cute.arch.block_idx()
        return pair

    @cute.jit
    def __call__(
        self,
        qkv_ptr: cute.Pointer,  # BF16 [rows, qkv_width], row stride qkv_stride
        pool_ptr: cute.Pointer,  # F8 [num_blocks, kv_heads, 32, 64], contiguous
        table_ptr: cute.Pointer,  # Int32 [2, table_width], row stride table_stride
        seq_lens_ptr: cute.Pointer,  # Int32 [1]
        kv_scale_ptr: cute.Pointer,  # Float32 [1] (kv_scale_orig_quant)
        part_o_ptr: cute.Pointer,  # F16 [hq * MAX_ROWS rows, num_splits, 64]
        part_lse_ptr: cute.Pointer,  # Float32 [hq * MAX_ROWS rows, num_splits]
        rows: Int32,
        qkv_stride: Int32,
        num_blocks: Int32,
        table_width: Int32,
        table_stride: Int32,
        num_splits: Int32,
        stream,
    ):
        qkv = cute.make_tensor(
            qkv_ptr, cute.make_layout((rows, self.qkv_width), stride=(qkv_stride, 1))
        )
        NKV = self.kv_heads
        pool = cute.make_tensor(
            pool_ptr,
            cute.make_layout((num_blocks, NKV, PAGE, D), stride=(NKV * PAGE * D, PAGE * D, D, 1)),
        )
        table = cute.make_tensor(
            table_ptr, cute.make_layout((2, table_width), stride=(table_stride, 1))
        )
        seq_lens = cute.make_tensor(seq_lens_ptr, cute.make_layout(1))
        kv_scale = cute.make_tensor(kv_scale_ptr, cute.make_layout(1))
        part_o = cute.make_tensor(part_o_ptr, cute.make_layout(self.hq * MAX_ROWS * num_splits * D))
        part_lse = cute.make_tensor(part_lse_ptr, cute.make_layout(self.hq * MAX_ROWS * num_splits))

        one = tcgen05.CtaGroup.ONE
        K = OperandMajorMode.K
        MN = OperandMajorMode.MN
        qk_mma = sm100_utils.make_trivial_tiled_mma(F8, F8, K, K, Float32, one, (128, PAGE))
        qk_mma_tile = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, K, Float32, one, (128, TILE_KEYS)
        )
        pv_mma = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, MN, Float32, one, (128, ROWS), tcgen05.OperandSource.TMEM
        )
        qk_tiler = (128, PAGE, ROWS)
        pv_tiler = (128, ROWS, PAGE)
        kinds = sm100_utils.SmemLayoutAtomKind
        sQ_layout = _ab_layout(qk_mma, qk_tiler, "A", kinds.K_SW64, self.nq)
        # One head's 32 x 64 K box: the TMA destination when the QK spans a tile.
        qk_mma_box = sm100_utils.make_trivial_tiled_mma(F8, F8, K, K, Float32, one, (128, PAGE))
        box_tiler = (128, PAGE, D)
        if cutlass.const_expr(self.k_tile):
            # K of a 128-key tile as one N=128 operand: per tile slot [head][128 keys][64 B] (a
            # uniform 512 B step between 8-key groups), written as 2 KB boxes, one per page and
            # head, at slot * 16 KB + head * 8 KB + page * 2 KB. Both views cover the same bytes.
            sK_layout = _ab_layout(
                qk_mma_tile, (128, TILE_KEYS, ROWS), "B", kinds.K_SW64,
                self.stages // PAGES_PER_TILE,
            )  # fmt: skip
            sKb_layout = _ab_layout(qk_mma_box, box_tiler, "B", kinds.K_SW64, HKV * self.stages)
        else:
            sK_layout = _ab_layout(qk_mma, qk_tiler, "B", kinds.K_SW64, self.stages)
            sKb_layout = sK_layout
        sV_layout = _ab_layout(pv_mma, pv_tiler, "B", kinds.MN_SW64, self.stages)
        # P's layout as the P.V A operand (P itself is in TMEM).
        p_layout = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, F8, 1)

        # K pool as the QK B operand: (keys 32, dims (64, head 2), pair + pairs * block)
        mK = cute.make_tensor(
            pool.iterator,
            cute.make_layout(
                (PAGE, (D, HKV), self.pairs * num_blocks),
                stride=(D, (1, PAGE * D), HKV * PAGE * D),
            ),
        )
        # V pool as the PV B operand: (dims (64, head 2), keys 32, pair + pairs * block)
        mV = cute.make_tensor(
            pool.iterator,
            cute.make_layout(
                ((D, HKV), PAGE, self.pairs * num_blocks),
                stride=((1, PAGE * D), D, HKV * PAGE * D),
            ),
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(one)
        sV_one = cute.select(sV_layout, mode=[0, 1, 2])
        if cutlass.const_expr(self.k_tile):
            assert cute.cosize(sKb_layout) == cute.cosize(sK_layout)
            # K pool as one head's B operand: (keys 32, dims 64, head + kv_heads * block)
            mKb = cute.make_tensor(
                pool.iterator,
                cute.make_layout((PAGE, D, NKV * num_blocks), stride=(D, 1, PAGE * D)),
            )
            sKb_one = cute.select(sKb_layout, mode=[0, 1, 2])
            tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_op, mKb, sKb_one, box_tiler, qk_mma_box, (1, 1, 1)
            )
            k_bytes = HKV * cute.size_in_bytes(F8, sKb_one)
        else:
            sK_one = cute.select(sK_layout, mode=[0, 1, 2])
            tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_op, mK, sK_one, qk_tiler, qk_mma, (1, 1, 1)
            )
            k_bytes = cute.size_in_bytes(F8, sK_one)
        tma_atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mV, sV_one, pv_tiler, pv_mma, (1, 1, 1)
        )
        self.page_bytes = k_bytes + cute.size_in_bytes(F8, sV_one)

        stages = self.stages
        max_cta_pages = self.max_cta_pages

        @cute.struct
        class SharedStorage:
            kv_full: cute.struct.MemRange[cutlass.Int64, stages]
            kv_empty: cute.struct.MemRange[cutlass.Int64, stages]
            s_full: cute.struct.MemRange[cutlass.Int64, 2]
            p_full: cute.struct.MemRange[cutlass.Int64, 2]
            p_full2: cute.struct.MemRange[cutlass.Int64, 2]
            o_done: cute.struct.MemRange[cutlass.Int64, 2]
            # No role of the op's build uses these; the fold build's ``app`` is the first.
            unused_bars: cute.struct.MemRange[cutlass.Int64, 2]
            misc: cute.struct.MemRange[cutlass.Int64, 2]  # q_full
            tmem_holding_buf: Int32
            sml: cute.struct.MemRange[Float32, 2 * ROWS * 2]
            ids: cute.struct.MemRange[Int32, 2 * max_cta_pages]
            sQ: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sQ_layout)], 1024]
            # No role of the op's build uses it; the fold build exchanges its Q half-sums here.
            unused: cute.struct.Align[cute.struct.MemRange[F8, 1024], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sV_layout)], 1024]

        self.shared_storage = SharedStorage

        args = (
            qk_mma,
            qk_mma_tile,
            qk_mma_box,
            pv_mma,
            tma_atom_k,
            tma_k,
            tma_atom_v,
            tma_v,
            qkv,
            table,
            seq_lens,
            kv_scale,
            part_o,
            part_lse,
            rows,
            sQ_layout,
            sK_layout,
            sKb_layout,
            sV_layout,
            p_layout,
        )
        if cutlass.const_expr(self.fold is not None):
            # The fold build's kernel also takes the pool (the append's generic stores).
            args = args + (pool,)
        self.kernel(*args).launch(
            grid=(num_splits, self.pairs, 1),
            block=[self.threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_tile: cute.TiledMma,
        qk_mma_box: cute.TiledMma,
        pv_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        tma_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_v: cute.Tensor,
        qkv: cute.Tensor,
        table: cute.Tensor,
        seq_lens: cute.Tensor,
        kv_scale: cute.Tensor,
        part_o: cute.Tensor,
        part_lse: cute.Tensor,
        rows: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKb_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
    ):
        self.body(
            qk_mma,
            qk_mma_tile,
            qk_mma_box,
            pv_mma,
            tma_atom_k,
            tma_k,
            tma_atom_v,
            tma_v,
            qkv,
            table,
            seq_lens,
            kv_scale,
            part_o,
            part_lse,
            rows,
            sQ_layout,
            sK_layout,
            sKb_layout,
            sV_layout,
            p_layout,
            None,
        )

    @cute.jit
    def body(
        self,
        qk_mma,
        qk_mma_tile,
        qk_mma_box,
        pv_mma,
        tma_atom_k,
        tma_k,
        tma_atom_v,
        tma_v,
        qkv,
        table,
        seq_lens,
        kv_scale,
        part_o,
        part_lse,
        rows,
        sQ_layout,
        sK_layout,
        sKb_layout,
        sV_layout,
        p_layout,
        pool,
    ):
        """The kernel's body, shared by the op's build (``pool`` None) and the fold build."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        split, _, _ = cute.arch.block_idx()
        pair = self.pair_index()
        nsplit, _, _ = cute.arch.grid_dim()

        if warp == self.load_warp:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        kv_full = storage.kv_full.data_ptr()
        kv_empty = storage.kv_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_full = storage.p_full.data_ptr()
        p_full2 = storage.p_full2.data_ptr()
        o_done = storage.o_done.data_ptr()
        unused_bars = storage.unused_bars.data_ptr()
        q_full = storage.misc.data_ptr()

        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(kv_full + i, 1)
                    cute.arch.mbarrier_init(kv_empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(s_full + i, 1)
                    cute.arch.mbarrier_init(p_full + i, 128)
                    cute.arch.mbarrier_init(p_full2 + i, 128)
                    cute.arch.mbarrier_init(o_done + i, 1)
                    if cutlass.const_expr(self.fold is not None and i == 0):
                        # The fold build's ``app``: the step's K/V are in the cache.
                        cute.arch.mbarrier_init(unused_bars + i, APPEND_THREADS)
                    else:
                        cute.arch.mbarrier_init(unused_bars + i, 128)
                cute.arch.mbarrier_init(q_full, 256)
            cute.arch.mbarrier_init_fence()

        # The TMEM users: the softmax warps and the MMA warp.
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * self.tmem_warps)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=self.mma_warp,
        )

        # Before the dependency wait: the sequence length and the KV scale. No kernel of the
        # step writes either (the op's PDL record), so every warp reads them now.
        L = seq_lens[0]
        scale = kv_scale[0]
        cute.arch.barrier()  # barrier inits visible to all warps

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        # ---- problem geometry (identical in every warp) ----
        M = rows
        past = L - M
        n_pages = (L + PAGE - 1) // PAGE
        eff = cutlass.min(nsplit, (n_pages + self.min_pages - 1) // self.min_pages)
        p_lo = split * n_pages // eff
        p_hi = (split + 1) * n_pages // eff
        key_lo = p_lo * PAGE
        key_hi = cutlass.min(p_hi * PAGE, L)
        n_tiles = (p_hi - p_lo + PAGES_PER_TILE - 1) // PAGES_PER_TILE
        last_page_tail = (key_hi == L) & ((L % PAGE) != 0) & (p_hi > p_lo)
        last_idx = (L - 1) // PAGE - p_lo  # page index (within this split) of the last key
        vfix_tile = last_idx // PAGES_PER_TILE
        dequant = Float32(1.0) / scale

        bars = (kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2)
        geo = (L, M, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx,
               vfix_tile, split, nsplit)  # fmt: skip
        # Everything derived from the tiled MMAs is built here, before any role region: the
        # MMA role mutates the tiled MMAs (ACCUMULATE field) and later regions must not use
        # values defined inside it.
        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma.get_slice(0)
        qk_tile_thr = qk_mma_tile.get_slice(0)
        if cutlass.const_expr(self.k_tile):
            sKb = storage.sK.get_tensor(sKb_layout.outer, swizzle=sKb_layout.inner)
            gK = cute.flat_divide(tma_k, (PAGE, D))
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sKb, 0, 3),
                cute.group_modes(qk_mma_box.get_slice(0).partition_B(gK), 0, 3),
            )
        else:
            gK = cute.flat_divide(tma_k, (PAGE, ROWS))
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(qk_thr.partition_B(gK), 0, 3),
            )
        gV = cute.flat_divide(tma_v, (ROWS, PAGE))
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(pv_thr.partition_B(gV), 0, 3),
        )
        tOrV = pv_thr.make_fragment_B(sV)
        o_acc_layout = pv_thr.make_fragment_C(pv_thr.partition_shape_C((128, ROWS))).layout
        s_tile_layout = qk_tile_thr.make_fragment_C(
            qk_tile_thr.partition_shape_C((128, TILE_KEYS))
        ).layout
        if cutlass.const_expr(self.k_tile):
            # QK per tile: M128 x N128 x K128 per Q term, B = the tile slot of K.
            qk_issue = qk_mma_tile
            tSrQ = qk_tile_thr.make_fragment_A(sQ)
            tSrK = qk_tile_thr.make_fragment_B(sK)
            s_acc_layout = s_tile_layout
        else:
            # QK per page: M128 x N32 x K128 per Q term, B = the page's stage.
            qk_issue = qk_mma
            tSrQ = qk_thr.make_fragment_A(sQ)
            tSrK = qk_thr.make_fragment_B(sK)
            s_acc_layout = qk_thr.make_fragment_C(qk_thr.partition_shape_C((128, PAGE))).layout
        tScS = qk_tile_thr.partition_C(cute.make_identity_tensor((128, TILE_KEYS)))
        tOcO = pv_thr.partition_C(cute.make_identity_tensor((128, ROWS)))
        frags = (tSrQ, tSrK, tOrV, s_acc_layout, o_acc_layout, p_layout)
        ids = cute.make_tensor(
            storage.ids.data_ptr(),
            cute.make_layout((2, self.max_cta_pages), stride=(self.max_cta_pages, 1)),
        )

        if split < eff:
            # setmaxnreg sits inside each role's branch so ptxas allocates per role.
            if warp == self.load_warp:
                cute.arch.setmaxregister_decrease(self.regs_other)
                if cutlass.const_expr(self.fold is not None):
                    self.load_role(
                        tma_atom_k, tKsK, tKgK, tma_atom_v, tVsV, tVgV, table, ids, bars, geo,
                        unused_bars,
                    )  # fmt: skip
                else:
                    self.load_role(
                        tma_atom_k, tKsK, tKgK, tma_atom_v, tVsV, tVgV, table, ids, bars, geo
                    )
            if warp >= self.load_warp + 1:
                cute.arch.setmaxregister_decrease(self.regs_other)
                if cutlass.const_expr(self.fold is not None):
                    self.fold_side_role(qkv, pool, table, kv_scale, unused_bars, geo, tidx)
            # TMEM release: the MMA warp and the softmax warps meet (tmem_bar) before the free.
            if warp == self.mma_warp:
                cute.arch.setmaxregister_decrease(self.regs_other)
                self.mma_role(qk_issue, pv_mma, frags, tmem, storage.sV.data_ptr(), bars, geo)
                tmem_bar.arrive_and_wait()
                tmem_end = tmem.retrieve_ptr(Float32)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_end)
            if warp < self.mma_warp:
                cute.arch.setmaxregister_increase(self.regs_softmax)
                if cutlass.const_expr(self.fold is not None):
                    self.softmax_role(
                        qkv,
                        s_tile_layout,
                        tScS,
                        o_acc_layout,
                        tOcO,
                        storage.sQ.data_ptr(),
                        storage.sml.data_ptr(),
                        tmem,
                        part_o,
                        part_lse,
                        dequant,
                        tidx,
                        warp,
                        bars,
                        geo,
                        storage.unused.data_ptr(),
                    )
                else:
                    self.softmax_role(
                        qkv,
                        s_tile_layout,
                        tScS,
                        o_acc_layout,
                        tOcO,
                        storage.sQ.data_ptr(),
                        storage.sml.data_ptr(),
                        tmem,
                        part_o,
                        part_lse,
                        dequant,
                        tidx,
                        warp,
                        bars,
                        geo,
                    )
                tmem_bar.arrive_and_wait()
        if split >= eff:
            # A split with no keys: its rows' log2-sum-exp is -inf (weight 0 in the combine);
            # thread i writes the row of token i // 16 and query head 16 pair + i % 16.
            # The store waits for the predecessor like every other global write here.
            if tidx < HQ * M:
                if cutlass.const_expr(self.enable_pdl):
                    cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.pairs == 1):
                    row_r = tidx
                else:
                    row_r = (tidx // HQ) * self.hq + pair * HQ + tidx % HQ
                part_lse[row_r * nsplit + split] = -Float32.inf
        if cutlass.const_expr(self.fold is not None and self.enable_pdl):
            # The fold build waits before it exits, on every path of every thread: its
            # predecessor is the QKV projection, which releases it early, and kernels after
            # it rely on its completion implying the projection's. Every role that reads or
            # writes global memory has waited by now, so this returns at once for them; the
            # threads that would otherwise exit without a wait (the MMA warp, the side warps
            # of a CTA that appends nothing, the key-less CTAs' threads past the stores) hold
            # nothing another role needs.
            cute.arch.griddepcontrol_wait()

    # ------------------------------------------------------------------ load warp
    @cute.jit
    def load_role(
        self, tma_atom_k, tKsK, tKgK, tma_atom_v, tVsV, tVgV, table, ids, bars, geo, app=None
    ):
        S = self.stages
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2 = bars
        (L, M, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx,
         vfix_tile, split_, nsplit_) = geo  # fmt: skip
        lane = cute.arch.lane_idx()
        pair = self.pair_index()
        n_mine = p_hi - p_lo
        # This CTA's page ids, in coalesced passes of MAX_CTA_PAGES (per pass all loads first,
        # then the stores): the block table of the sequence is not written by any kernel of the
        # step.
        nj = MAX_CTA_PAGES // 32
        for ps in cutlass.range_constexpr(self.max_cta_pages // MAX_CTA_PAGES):
            rk = cute.make_rmem_tensor((nj,), Int32)
            rv = cute.make_rmem_tensor((nj,), Int32)
            for j in cutlass.range_constexpr(nj):
                src = cutlass.min(p_lo + ps * MAX_CTA_PAGES + j * 32 + lane, p_hi - 1)
                rk[j] = table[0, src]
                rv[j] = table[1, src]
            for j in cutlass.range_constexpr(nj):
                i = ps * MAX_CTA_PAGES + j * 32 + lane
                if i < n_mine:
                    ids[0, i] = rk[j]
                    ids[1, i] = rv[j]
        cute.arch.sync_warp()
        # Pages from this one on may hold a key of the step's own rows, which the step's prep
        # writes: they are loaded only after the dependency wait.
        first_new = past // PAGE
        policy = None
        if cutlass.const_expr(self.evict_first):
            policy = cutlass.Int64(EVICT_FIRST_POLICY)
        if cutlass.const_expr(self.enable_pdl and not self.prefetch and self.loader_wait):
            cute.arch.griddepcontrol_wait()
        for t in cutlass.range(n_tiles, unroll=1):
            for p in cutlass.range_constexpr(PAGES_PER_TILE):
                idx = t * PAGES_PER_TILE + p
                st = idx % S
                cute.arch.mbarrier_wait(kv_empty + st, (idx // S + 1) % 2)
                page = p_lo + idx
                if cutlass.const_expr(self.prefetch and self.loader_wait):
                    if page >= first_new:
                        cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.fold is not None and self.app_wait):
                    # The fold build: a page that holds a key of the step's rows is loaded only
                    # after this CTA's side warps appended those keys (their generic stores,
                    # then ``fence.proxy.async.global``, then their arrival on ``app``). Only a
                    # page of this CTA's range waits: the padding slots of its last tile
                    # (page >= p_hi) load nothing, and a CTA whose range ends before the step's
                    # first page has side warps that never arrive.
                    if (page >= first_new) & (page < p_hi):
                        cute.arch.mbarrier_wait(app, 0)
                valid = page < p_hi
                safe = cutlass.min(idx, n_mine - 1)
                kid = cutlass.select_(valid, ids[0, safe], Int32(-1))
                vid = cutlass.select_(valid, ids[1, safe], Int32(-1))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(kv_full + st, self.page_bytes)
                # This CTA's pair of the page's K and V heads: pool heads 2 pair, 2 pair + 1.
                kpair = self.pairs * kid + pair
                vpair = self.pairs * vid + pair
                if cutlass.const_expr(self.k_tile):
                    # Head h of this page: box (slot * 2 + h) * 4 + page-in-tile of the K slots.
                    box = (st // PAGES_PER_TILE) * (HKV * PAGES_PER_TILE) + st % PAGES_PER_TILE
                    for h in cutlass.range_constexpr(HKV):
                        if cutlass.const_expr(self.evict_first):
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, 0, 0, HKV * kpair + h],
                                tKsK[None, box + h * PAGES_PER_TILE],
                                tma_bar_ptr=kv_full + st,
                                cache_policy=policy,
                            )
                        else:
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, 0, 0, HKV * kpair + h],
                                tKsK[None, box + h * PAGES_PER_TILE],
                                tma_bar_ptr=kv_full + st,
                            )
                elif cutlass.const_expr(self.evict_first):
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, 0, 0, kpair],
                        tKsK[None, st],
                        tma_bar_ptr=kv_full + st,
                        cache_policy=policy,
                    )
                else:
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, 0, 0, kpair],
                        tKsK[None, st],
                        tma_bar_ptr=kv_full + st,
                    )
                if cutlass.const_expr(self.evict_first):
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, 0, 0, vpair],
                        tVsV[None, st],
                        tma_bar_ptr=kv_full + st,
                        cache_policy=policy,
                    )
                else:
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, 0, 0, vpair],
                        tVsV[None, st],
                        tma_bar_ptr=kv_full + st,
                    )

    # ------------------------------------------------------------------ MMA warp
    @cute.jit
    def qk_tile(self, qk_mma_in, tmem_ptr, buf, tSrQ, tSrK, kv_full, t, s_acc_layout):
        S = self.stages
        # A private copy: setting ACCUMULATE must not leak this region's value to the caller.
        qk_mma = _private_mma(qk_mma_in)
        if cutlass.const_expr(self.k_tile):
            # The tile's 4 pages, then one N=128 MMA per Q term and 32-wide K block.
            for p in cutlass.range_constexpr(PAGES_PER_TILE):
                idx = t * PAGES_PER_TILE + p
                cute.arch.mbarrier_wait(kv_full + idx % S, (idx // S) % 2)
            slot = t % (S // PAGES_PER_TILE)
            acc = cute.make_tensor(tmem_ptr + buf * 128, s_acc_layout)
            for q in cutlass.range_constexpr(self.nq):
                for k in cutlass.range_constexpr(4):
                    qk_mma.set(tcgen05.Field.ACCUMULATE, (q + k) != 0)
                    cute.gemm(qk_mma, acc, tSrQ[None, None, k, q], tSrK[None, None, k, slot], acc)
        else:
            for p in cutlass.range_constexpr(PAGES_PER_TILE):
                idx = t * PAGES_PER_TILE + p
                st = idx % S
                cute.arch.mbarrier_wait(kv_full + st, (idx // S) % 2)
                acc = cute.make_tensor(tmem_ptr + buf * 128 + PAGE * p, s_acc_layout)
                for q in cutlass.range_constexpr(self.nq):
                    for k in cutlass.range_constexpr(4):
                        qk_mma.set(tcgen05.Field.ACCUMULATE, (q + k) != 0)
                        cute.gemm(qk_mma, acc, tSrQ[None, None, k, q], tSrK[None, None, k, st], acc)

    @cute.jit
    def pv_tile(
        self, pv_mma_in, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, p0=0, p1=PAGES_PER_TILE
    ):
        S = self.stages
        pv_mma = _private_mma(pv_mma_in)
        tOtO = cute.make_tensor(tmem_ptr + 256 + buf * 128, o_acc_layout)
        for p in cutlass.range_constexpr(p0, p1):
            st = (t * PAGES_PER_TILE + p) % S
            a = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + buf * 128 + 8 * p, dtype=F8), tOrP0.layout
            )
            pv_mma.set(tcgen05.Field.ACCUMULATE, (t >= 2) | (p != 0))
            cute.gemm(pv_mma, tOtO, a[None, None, 0], tOrV[None, None, 0, st], tOtO)

    @cute.jit
    def mma_role(self, qk_mma, pv_mma, frags, tmem, sv_ptr, bars, geo):
        S = self.stages
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2 = bars
        (L, M, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx,
         vfix_tile, split_, nsplit_) = geo  # fmt: skip
        tSrQ, tSrK, tOrV, s_acc_layout, o_acc_layout, p_layout = frags
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tP = cute.make_tensor(tmem_ptr, p_layout.outer)
        tOrP0 = pv_mma.get_slice(0).make_fragment_A(tP)[None, None, None, 0]

        cute.arch.mbarrier_wait(q_full, 0)
        _fence_after()
        if n_tiles > 0:
            self.qk_tile(qk_mma, tmem_ptr, 0, tSrQ, tSrK, kv_full, Int32(0), s_acc_layout)
            with cute.arch.elect_one():
                tcgen05.commit(s_full)
        if n_tiles > 1:
            self.qk_tile(qk_mma, tmem_ptr, 1, tSrQ, tSrK, kv_full, Int32(1), s_acc_layout)
            with cute.arch.elect_one():
                tcgen05.commit(s_full + 1)
        for t in cutlass.range(n_tiles, unroll=1):
            buf = t % 2
            cute.arch.mbarrier_wait(p_full + buf, (t // 2) % 2)
            if last_page_tail & (t == vfix_tile):
                # V rows at or past L in the last page may hold any bytes (an E4M3 NaN times
                # P = 0 is NaN): zero them. This warp waited for that page's load in QK(t), and
                # the stage cannot be refilled before PV(t)'s kv_empty commit below.
                st = last_idx % S
                r0 = L % PAGE
                lane = cute.arch.lane_idx()
                sv_i32 = cute.make_tensor(
                    cute.recast_ptr(sv_ptr, dtype=Int32), cute.make_layout(S * 1024)
                )
                n_words = (PAGE - r0) * (D // 4)  # per head
                for i in cutlass.range(lane, 2 * n_words, 32, unroll=1):
                    sv_i32[st * 1024 + (i // n_words) * 512 + r0 * 16 + i % n_words] = Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
            _fence_after()
            if cutlass.const_expr(self.p_split):
                # Pages 0-1 as soon as their P is in TMEM, pages 2-3 after the second half.
                self.pv_tile(pv_mma, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, 0, 2)
                cute.arch.mbarrier_wait(p_full2 + buf, (t // 2) % 2)
                _fence_after()
                self.pv_tile(pv_mma, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, 2, 4)
            else:
                self.pv_tile(pv_mma, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t)
            with cute.arch.elect_one():
                tcgen05.commit(o_done + buf)
                for p in cutlass.range_constexpr(PAGES_PER_TILE):
                    tcgen05.commit(kv_empty + (t * PAGES_PER_TILE + p) % S)
            if t + 2 < n_tiles:
                self.qk_tile(qk_mma, tmem_ptr, buf, tSrQ, tSrK, kv_full, t + 2, s_acc_layout)
                with cute.arch.elect_one():
                    tcgen05.commit(s_full + buf)

    # ------------------------------------------------------------------ fold build: append
    @cute.jit
    def fold_side_role(self, qkv, pool, table, kv_scale, app, geo, tidx):
        """Warps 10-11 of a CTA whose page range holds a key of the step's rows: write the step's
        K' (``preattn_prep``'s weightless RMSNorm, BF16) and V into the paged FP8 cache as
        ``preattn_prep_append`` does, then release this CTA's loader for those pages (``app``).

        These warps run at ``regs_other`` registers: the work streams (one load pass of 16
        elements per thread, no per-item arrays).
        """
        (L, M, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx,
         vfix_tile, split_, nsplit_) = geo  # fmt: skip
        eps_c, qscale_c, permuted_c = self.fold
        first_new = past // PAGE
        if p_hi > first_new:
            if cutlass.const_expr(self.enable_pdl):
                # The projection is the predecessor's output; the cache pages are read by this
                # CTA's loader only after ``app``.
                cute.arch.griddepcontrol_wait()
            # 64 threads = 16 (row, KV head) items x 4 quarters; quarter m holds elements
            # 16m .. 16m + 15 of the head = the prep's lanes 2m and 2m + 1, of K and of V. The K
            # norm follows the prep's order: in-lane sequential adds, lane bit 0 locally, bits 1
            # and 2 by butterflies over the item's four threads.
            pair = self.pair_index()
            t = tidx - 32 * (self.load_warp + 1)
            item = t // 4
            qm = t % 4
            row = item // 2
            # The rank's KV head: head item % 2 of this CTA's pair.
            kvh = pair * HKV + item % 2
            row_c = cutlass.min(row, M - 1)
            pos = past + row_c
            pg = pos // PAGE
            act = (row < M) & (pg >= p_lo) & (pg < p_hi)
            if cutlass.const_expr(permuted_c):
                base_kv = self.hq * D
            else:
                base_kv = 2 * self.hq * D
            src_k = cute.assume(row_c * qkv.stride[0] + base_kv + kvh * D + 16 * qm, divby=8)
            src_v = cute.assume(src_k + self.kv_heads * D, divby=8)
            rk = cute.make_rmem_tensor((16,), BF16)
            rk.store(cute.make_tensor(qkv.iterator + src_k, cute.make_layout(16)).load())
            rv = cute.make_rmem_tensor((16,), BF16)
            rv.store(cute.make_tensor(qkv.iterator + src_v, cute.make_layout(16)).load())
            fk = cute.make_rmem_tensor((16,), Float32)
            fk.store(rk.load().to(Float32))
            sums = []
            for ln in cutlass.range_constexpr(2):
                acc = add_rn(mul_rn(fk[8 * ln], fk[8 * ln]), mul_rn(fk[8 * ln + 1], fk[8 * ln + 1]))
                for e in cutlass.range_constexpr(2, 8):
                    acc = add_rn(acc, mul_rn(fk[8 * ln + e], fk[8 * ln + e]))
                sums.append(acc)
            u = add_rn(sums[0], sums[1])
            v = add_rn(u, cute.arch.shuffle_sync_bfly(u, 1))
            total = add_rn(v, cute.arch.shuffle_sync_bfly(v, 2))
            inv = rsqrt_approx_ftz(add_rn(mul_rn(total, Float32(1.0 / D)), Float32(eps_c)))
            sc = round_bf16(kv_scale[0])
            if act:
                fv = cute.make_rmem_tensor((16,), Float32)
                fv.store(rv.load().to(Float32))
                for kind in cutlass.range_constexpr(2):
                    page_id = table[kind, pg]
                    w0 = cutlass.Int64(page_id) * (self.kv_heads * PAGE * D // 4) + cutlass.Int64(
                        (kvh * PAGE + pos % PAGE) * (D // 4) + 4 * qm
                    )
                    w4 = cute.make_rmem_tensor((4,), Int32)
                    for j in cutlass.range_constexpr(4):
                        z = []
                        for e in cutlass.range_constexpr(4):
                            if cutlass.const_expr(kind == 0):
                                y = round_bf16(mul_rn(fk[4 * j + e], inv))
                            else:
                                y = fv[4 * j + e]
                            # _to_cache_fp8: bf16(x * bf16(s)), then E4M3 RN satfinite.
                            z.append(round_bf16(mul_rn(y, sc)))
                        w4[j] = pack_e4m3x4(z[0], z[1], z[2], z[3])
                    cute.make_tensor(
                        cute.recast_ptr(pool.iterator, dtype=Int32) + w0, cute.make_layout(4)
                    ).store(w4.load())
            # The generic stores above before the loader's TMA (async proxy) reads of them.
            cute.arch.fence_proxy("async.global")
            cute.arch.mbarrier_arrive(app)

    # ------------------------------------------------------------------ softmax warpgroups
    @cute.jit
    def build_q(self, rqb, sq_ptr, r, h, hq, row_valid):
        """Split this thread's 32 dims of row r into Q_hi + Q_lo, block-diagonal Q' in smem."""
        sq_i32 = cute.make_tensor(
            cute.recast_ptr(sq_ptr, dtype=Int32), cute.make_layout(self.nq * 4096)
        )
        rq = cute.make_rmem_tensor((D // 2,), Float32)
        rq.store(rqb.load().to(Float32))
        rw = cute.make_rmem_tensor((self.nq, 8), Int32)
        zf = Float32(0.0)
        for e4 in cutlass.range_constexpr(8):
            e = 4 * e4
            v0 = cutlass.select_(row_valid, rq[e], zf)
            v1 = cutlass.select_(row_valid, rq[e + 1], zf)
            v2 = cutlass.select_(row_valid, rq[e + 2], zf)
            v3 = cutlass.select_(row_valid, rq[e + 3], zf)
            h0, h1, h2, h3 = split_hi(v0), split_hi(v1), split_hi(v2), split_hi(v3)
            rw[0, e4] = pack_e4m3x4(h0, h1, h2, h3)
            rw[1, e4] = pack_e4m3x4(v0 - h0, v1 - h1, v2 - h2, v3 - h3)
        zero4 = cute.make_rmem_tensor((4,), Int32)
        zero4.fill(0)
        for cl in cutlass.range_constexpr(2):  # this thread's two 16-byte chunks of the row slice
            cs = (2 * hq + cl) ^ ((r >> 1) & 3)
            for q in cutlass.range_constexpr(self.nq):
                base = cute.assume(q * 4096 + r * 16 + cs * 4, divby=4)
                own = cute.make_tensor(sq_i32.iterator + base + h * 2048, cute.make_layout(4))
                oth = cute.make_tensor(sq_i32.iterator + base + (1 - h) * 2048, cute.make_layout(4))
                vals = cute.make_rmem_tensor((4,), Int32)
                for w in cutlass.range_constexpr(4):
                    vals[w] = rw[q, 4 * cl + w]
                own.store(vals.load())
                oth.store(zero4.load())

    @cute.jit
    def softmax_role(
        self,
        qkv,
        s_tile_layout,
        tScS,
        o_acc_layout,
        tOcO,
        sq_ptr,
        sml_ptr,
        tmem,
        part_o,
        part_lse,
        dequant,
        tidx,
        warp,
        bars,
        geo,
        xch_ptr=None,
    ):
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2 = bars
        (L, M, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx,
         vfix_tile, split, nsplit) = geo  # fmt: skip
        wg = warp // 4  # 0 or 1: takes tiles wg, wg + 2, ...
        pair = self.pair_index()
        r = tidx % 128  # row 0..127 = TMEM lane
        h = r // D
        ri = r % D
        tok = ri // G
        gq = ri % G
        # The row's query head among the rank's: query gq of head h of this CTA's pair.
        qh = pair * HQ + h * G + gq
        row_valid = tok < M
        qpos = past + tok

        # Q is the predecessor's output: read only after the dependency wait.
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        tok_c = cutlass.select_(row_valid, tok, 0)
        if cutlass.const_expr(self.fold is None):
            qoff = cute.assume(tok_c * qkv.stride[0] + qh * D + wg * (D // 2), divby=8)
            rqb = cute.make_rmem_tensor((D // 2,), BF16)
            rqb.store(cute.make_tensor(qkv.iterator + qoff, cute.make_layout(D // 2)).load())
        else:
            # The fold build: this thread's 32 dims of its (token, q-head) row from the raw
            # projection (head stride 128 in checkpoint order, 64 permuted), normalized as
            # ``preattn_prep`` normalizes them (``_inverse_rms``, ``_normalize``). Lane
            # l = 4a + 2b + c of the prep holds elements 8l .. 8l + 7; this thread holds a = wg:
            # in-lane sequential adds of exact squares, then lane bits 0 and 1 here, then bit 2
            # by adding the other warpgroup's half through shared memory (one add, commutative).
            eps_c, qscale_c, permuted_c = self.fold
            if cutlass.const_expr(permuted_c):
                hq_stride = D
            else:
                hq_stride = 2 * D
            qoff = cute.assume(tok_c * qkv.stride[0] + qh * hq_stride + wg * (D // 2), divby=8)
            rx = cute.make_rmem_tensor((D // 2,), BF16)
            rx.store(cute.make_tensor(qkv.iterator + qoff, cute.make_layout(D // 2)).load())
            xf = cute.make_rmem_tensor((D // 2,), Float32)
            xf.store(rx.load().to(Float32))
            lane_sums = []
            for ln in cutlass.range_constexpr(4):
                acc = add_rn(mul_rn(xf[8 * ln], xf[8 * ln]), mul_rn(xf[8 * ln + 1], xf[8 * ln + 1]))
                for e in cutlass.range_constexpr(2, 8):
                    acc = add_rn(acc, mul_rn(xf[8 * ln + e], xf[8 * ln + e]))
                lane_sums.append(acc)
            half = add_rn(add_rn(lane_sums[0], lane_sums[1]), add_rn(lane_sums[2], lane_sums[3]))
            # The exchange buffer (1 KB = 2 x 128 fp32) is used by nothing else in the CTA.
            xch = cute.make_tensor(
                cute.recast_ptr(xch_ptr, dtype=Float32),
                cute.make_layout((2, ROWS), stride=(ROWS, 1)),
            )
            xch[wg, r] = half
            pipeline.NamedBarrier(barrier_id=3, num_threads=256).arrive_and_wait()
            total = add_rn(xch[0, r], xch[1, r])
            inv = rsqrt_approx_ftz(add_rn(mul_rn(total, Float32(1.0 / D)), Float32(eps_c)))
            rqf = cute.make_rmem_tensor((D // 2,), Float32)
            for e in cutlass.range_constexpr(D // 2):
                if cutlass.const_expr(qscale_c != 1.0):
                    rqf[e] = mul_rn(round_bf16(mul_rn(xf[e], inv)), Float32(qscale_c))
                else:
                    rqf[e] = mul_rn(xf[e], inv)
            rqb = cute.make_rmem_tensor((D // 2,), BF16)
            rqb.store(rqf.load().to(BF16))
        self.build_q(rqb, sq_ptr, r, h, wg, row_valid)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(q_full)

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        s_base = tmem_ptr + wg * 128
        # S tile (128 x 128 fp32): one row per thread
        tStS = cute.make_tensor(s_base, s_tile_layout)
        ld_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32), tStS
        )
        thr_ld_s = ld_s.get_slice(r)
        tLcS = thr_ld_s.partition_D(tScS)
        tLtS = thr_ld_s.partition_S(tStS)
        rS = cute.make_rmem_tensor(tLcS.shape, Float32)

        # P (FP8, 4 per 32-bit column) in the S buffer: 32 columns
        tStP = cute.make_tensor(
            s_base, cute.composition(s_tile_layout, cute.make_layout((128, 32)))
        )
        st_p = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), Float32), tStP
        )
        thr_st_p = st_p.get_slice(r)
        tPtP = thr_st_p.partition_D(tStP)
        tScP = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((128, 32)))
        )
        tPcP = thr_st_p.partition_S(tScP)
        rP = cute.make_rmem_tensor(tPcP.shape, Float32)
        rP_i32 = cute.recast_tensor(rP, Int32)
        # The two 16-column halves of P (keys 0-63 and 64-127 of the tile).
        tStPh = cute.make_tensor(
            s_base, cute.composition(s_tile_layout, cute.make_layout((128, 16)))
        )
        st_ph = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32),
            tStPh,
        )
        thr_st_ph = st_ph.get_slice(r)
        tPtP0 = thr_st_ph.partition_D(tStPh)
        tPtP1 = thr_st_ph.partition_D(
            cute.make_tensor(
                s_base + 16, cute.composition(s_tile_layout, cute.make_layout((128, 16)))
            )
        )
        tScPh = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((128, 16)))
        )
        tPcPh = thr_st_ph.partition_S(tScPh)
        rPh0 = cute.make_rmem_tensor(tPcPh.shape, Float32)
        rPh0_i32 = cute.recast_tensor(rPh0, Int32)
        rPh1 = cute.make_rmem_tensor(tPcPh.shape, Float32)
        rPh1_i32 = cute.recast_tensor(rPh1, Int32)

        RC = self.rescale_cols
        if cutlass.const_expr(RC == D):
            # O: this warpgroup's accumulator, this row's head half (64 columns)
            o_half_layout = cute.composition(o_acc_layout, cute.make_layout((128, D)))
            tOtO_ref = cute.make_tensor(tmem_ptr + 256, o_half_layout)
            ld_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32),
                tOtO_ref,
            )
            st_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), Float32),
                tOtO_ref,
            )
            thr_ld_o = ld_o.get_slice(r)
            thr_st_o = st_o.get_slice(r)
            tOtO_h = cute.make_tensor(tmem_ptr + 256 + wg * 128 + h * D, o_half_layout)
            tLtO = thr_ld_o.partition_S(tOtO_h)
            tStO = thr_st_o.partition_D(tOtO_h)
            tOcO_h = cute.make_tensor(
                tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, D)))
            )
            tLcO = thr_ld_o.partition_D(tOcO_h)
            rO = cute.make_rmem_tensor(tLcO.shape, Float32)
        else:
            # O of this row's head in chunks of RC columns: RC registers instead of 64.
            o_chunk_layout = cute.composition(o_acc_layout, cute.make_layout((128, RC)))
            tOtO_ref = cute.make_tensor(tmem_ptr + 256, o_chunk_layout)
            ld_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(RC)), Float32),
                tOtO_ref,
            )
            st_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(RC)), Float32),
                tOtO_ref,
            )
            thr_ld_o = ld_o.get_slice(r)
            thr_st_o = st_o.get_slice(r)
            o_col0 = 256 + wg * 128 + h * D
            tOcO_h = cute.make_tensor(
                tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, RC)))
            )
            tLcO = thr_ld_o.partition_D(tOcO_h)
            rO = cute.make_rmem_tensor(tLcO.shape, Float32)

        sc = Float32(SM_SCALE_LOG2) * dequant
        neg_inf = -Float32.inf
        m = neg_inf
        l = Float32(0.0)  # noqa: E741 - the softmax denominator
        lim = cutlass.min(key_hi - 1, qpos)
        my_tiles = (n_tiles - wg + 1) // 2
        for u in cutlass.range(my_tiles, unroll=1):
            t = wg + 2 * u
            cute.arch.mbarrier_wait(s_full + wg, u % 2)
            _fence_after()
            cute.copy(ld_s, tLtS, rS)
            cute.arch.fence_view_async_tmem_load()
            key0 = key_lo + t * TILE_KEYS
            if key0 + TILE_KEYS - 1 > lim:
                for j in cutlass.range_constexpr(TILE_KEYS):
                    rS[j] = cutlass.select_(key0 + j <= lim, rS[j], neg_inf)
            mx0 = fmax3(rS[0], rS[1], rS[2])
            mx1 = fmax3(rS[3], rS[4], rS[5])
            mx2 = fmax3(rS[6], rS[7], rS[8])
            mx3 = fmax3(rS[9], rS[10], rS[11])
            for j in cutlass.range_constexpr(12, TILE_KEYS - 4, 8):
                mx0 = fmax3(mx0, rS[j], rS[j + 1])
                mx1 = fmax3(mx1, rS[j + 2], rS[j + 3])
                mx2 = fmax3(mx2, rS[j + 4], rS[j + 5])
                mx3 = fmax3(mx3, rS[j + 6], rS[j + 7])
            mx0 = fmax3(mx0, rS[TILE_KEYS - 4], rS[TILE_KEYS - 3])
            mx1 = fmax3(mx1, rS[TILE_KEYS - 2], rS[TILE_KEYS - 1])
            tmax = fmax3(fmax3(mx0, mx1, mx2), mx3, m)
            m_new = tmax
            need = (m_new - m) * sc > Float32(self.tau)
            do_rescale = need & (u > 0) & (m != neg_inf)
            # PV of this warpgroup's previous tile completed before S(t) was committed.
            if cute.arch.vote_any_sync(do_rescale):
                factor = cutlass.select_(
                    do_rescale,
                    cute.math.exp2((m - m_new) * sc, fastmath=True),
                    Float32(1.0),
                )
                if cutlass.const_expr(RC == D):
                    cute.copy(ld_o, tLtO, rO)
                    cute.arch.fence_view_async_tmem_load()
                    for j in cutlass.range_constexpr(0, D, 2):
                        rO[j], rO[j + 1] = cute.arch.mul_packed_f32x2(
                            (rO[j], rO[j + 1]), (factor, factor)
                        )
                    cute.copy(st_o, rO, tStO)
                else:
                    # Load, scale and store one RC-column chunk at a time: the same multiply
                    # per element, so the same values.
                    for c in cutlass.range_constexpr(D // RC):
                        tOc = cute.make_tensor(tmem_ptr + o_col0 + c * RC, o_chunk_layout)
                        cute.copy(ld_o, thr_ld_o.partition_S(tOc), rO)
                        cute.arch.fence_view_async_tmem_load()
                        for j in cutlass.range_constexpr(0, RC, 2):
                            rO[j], rO[j + 1] = cute.arch.mul_packed_f32x2(
                                (rO[j], rO[j + 1]), (factor, factor)
                            )
                        cute.copy(st_o, rO, thr_st_o.partition_D(tOc))
                cute.arch.fence_view_async_tmem_store()
                l = l * factor  # noqa: E741
            m = cutlass.select_(need, m_new, m)
            m_safe = cutlass.select_(m == neg_inf, Float32(0.0), m)
            bias = Float32(self.pre) - m_safe * sc
            if cutlass.const_expr(self.p_split):
                for hh in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(64 * hh, 64 * hh + 64, 2):
                        rS[j], rS[j + 1] = cute.arch.fma_packed_f32x2(
                            (rS[j], rS[j + 1]), (sc, sc), (bias, bias)
                        )
                        rS[j] = cute.math.exp2(rS[j], fastmath=True)
                        rS[j + 1] = cute.math.exp2(rS[j + 1], fastmath=True)
                    if cutlass.const_expr(hh == 0):
                        for w in cutlass.range_constexpr(16):
                            rPh0_i32[w] = pack_e4m3x4(
                                rS[4 * w], rS[4 * w + 1], rS[4 * w + 2], rS[4 * w + 3]
                            )
                        cute.copy(st_ph, rPh0, tPtP0)
                        cute.arch.fence_view_async_tmem_store()
                        _fence_before()
                        cute.arch.mbarrier_arrive(p_full + wg)
                    else:
                        for w in cutlass.range_constexpr(16):
                            rPh1_i32[w] = pack_e4m3x4(
                                rS[64 + 4 * w],
                                rS[64 + 4 * w + 1],
                                rS[64 + 4 * w + 2],
                                rS[64 + 4 * w + 3],
                            )
                        cute.copy(st_ph, rPh1, tPtP1)
                        cute.arch.fence_view_async_tmem_store()
                        _fence_before()
                        cute.arch.mbarrier_arrive(p_full2 + wg)
            else:
                for j in cutlass.range_constexpr(0, TILE_KEYS, 2):
                    rS[j], rS[j + 1] = cute.arch.fma_packed_f32x2(
                        (rS[j], rS[j + 1]), (sc, sc), (bias, bias)
                    )
                    rS[j] = cute.math.exp2(rS[j], fastmath=True)
                    rS[j + 1] = cute.math.exp2(rS[j + 1], fastmath=True)
            acc0 = (Float32(0.0), Float32(0.0))
            acc1 = (Float32(0.0), Float32(0.0))
            for j in cutlass.range_constexpr(0, TILE_KEYS, 4):
                acc0 = cute.arch.add_packed_f32x2(acc0, (rS[j], rS[j + 1]))
                acc1 = cute.arch.add_packed_f32x2(acc1, (rS[j + 2], rS[j + 3]))
            l = l + ((acc0[0] + acc0[1]) + (acc1[0] + acc1[1]))  # noqa: E741
            if cutlass.const_expr(not self.p_split):
                for w in cutlass.range_constexpr(32):
                    rP_i32[w] = pack_e4m3x4(rS[4 * w], rS[4 * w + 1], rS[4 * w + 2], rS[4 * w + 3])
            if cutlass.const_expr(not self.p_split):
                cute.copy(st_p, rP, tPtP)
                cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(not self.p_split):
                _fence_before()
                cute.arch.mbarrier_arrive(p_full + wg)

        # ---- epilogue: merge the two warpgroups' (O, m, l) of this row into one partial ----
        if cutlass.const_expr(self.enable_pdl):
            # The dependent grid may become resident from here on; it waits for this grid's
            # completion before it reads anything this grid writes.
            cute.arch.griddepcontrol_launch_dependents()
        bar2 = pipeline.NamedBarrier(barrier_id=2, num_threads=256)
        sml = cute.make_tensor(sml_ptr, cute.make_layout((2, ROWS, 2), stride=(2 * ROWS, 2, 1)))
        sml[wg, r, 0] = m
        sml[wg, r, 1] = l
        bar2.arrive_and_wait()
        n0 = (n_tiles + 1) // 2  # tiles of warpgroup 0
        n1 = n_tiles // 2
        my_n = cutlass.select_(wg == 0, n0, n1)
        ot_n = cutlass.select_(wg == 0, n1, n0)
        # Own last PV first: it completes after the other warpgroup's PV of the previous tile,
        # so the second wait below is never more than one phase ahead.
        if my_n > 0:
            cute.arch.mbarrier_wait(o_done + wg, (my_n - 1) % 2)
        if ot_n > 0:
            cute.arch.mbarrier_wait(o_done + (1 - wg), (ot_n - 1) % 2)
        _fence_after()
        # This thread's 32 output columns [32 wg, 32 wg + 32) of its head, from both accumulators.
        o_q_layout = cute.composition(o_acc_layout, cute.make_layout((128, D // 2)))
        tOq_ref = cute.make_tensor(tmem_ptr + 256, o_q_layout)
        ld_q = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32),
            tOq_ref,
        )
        thr_ld_q = ld_q.get_slice(r)
        tOcq = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, D // 2)))
        )
        tLcq = thr_ld_q.partition_D(tOcq)
        rO0 = cute.make_rmem_tensor(tLcq.shape, Float32)
        rO1 = cute.make_rmem_tensor(tLcq.shape, Float32)
        cute.copy(
            ld_q,
            thr_ld_q.partition_S(
                cute.make_tensor(tmem_ptr + 256 + h * D + wg * (D // 2), o_q_layout)
            ),
            rO0,
        )
        cute.copy(
            ld_q,
            thr_ld_q.partition_S(
                cute.make_tensor(tmem_ptr + 384 + h * D + wg * (D // 2), o_q_layout)
            ),
            rO1,
        )
        cute.arch.fence_view_async_tmem_load()
        m0 = sml[0, r, 0]
        l0 = sml[0, r, 1]
        m1 = sml[1, r, 0]
        l1 = sml[1, r, 1]
        mx = cutlass.max(m0, m1)
        mx_safe = cutlass.select_(mx == neg_inf, Float32(0.0), mx)
        w0 = cutlass.select_(
            (n0 > 0) & (m0 != neg_inf),
            cute.math.exp2((m0 - mx_safe) * sc, fastmath=True),
            Float32(0.0),
        )
        w1 = cutlass.select_(
            (n1 > 0) & (m1 != neg_inf),
            cute.math.exp2((m1 - mx_safe) * sc, fastmath=True),
            Float32(0.0),
        )
        lsum = w0 * l0 + w1 * l1
        inv = cutlass.select_(lsum > Float32(0.0), rcp_approx(lsum), Float32(0.0)) * dequant
        a0 = w0 * inv
        a1 = w1 * inv
        res = cute.make_rmem_tensor((D // 2,), Float32)
        for j in cutlass.range_constexpr(D // 2):
            x0 = cutlass.select_(n0 > 0, rO0[j], Float32(0.0))
            x1 = cutlass.select_(n1 > 0, rO1[j], Float32(0.0))
            res[j] = a0 * x0 + a1 * x1
        # Output row of (token, query head): token * hq + the rank's query head.
        R = tok * self.hq + qh
        if row_valid:
            poff = cute.assume((R * nsplit + split) * D + wg * (D // 2), divby=8)
            cute.make_tensor(part_o.iterator + poff, cute.make_layout(D // 2)).store(
                res.load().to(F16)
            )
            if wg == 0:
                part_lse[R * nsplit + split] = cutlass.select_(
                    lsum > Float32(0.0),
                    mx_safe * sc - Float32(self.pre) + log2_approx(lsum),
                    neg_inf,
                )
        _fence_before()


class FullAttnSplitKvFoldKernel(FullAttnSplitKvKernel):
    """``FullAttnSplitKvKernel`` with the folded prep taken in (the ``full_attention_prep_fold`` op).

    ``qkv`` is the raw QKV projection ``[rows, 2 (hq + kv_heads) 64]`` (2304 at tep8, 4608 at
    tep4; checkpoint order, or with ``permuted``
    the ``[q | k | v | gate]`` layout). The softmax threads normalize their Q dims as
    ``preattn_prep`` does (weightless RMSNorm with ``eps``, BF16, then ``q_scale``, BF16) before
    the unchanged ``Q_hi + Q_lo`` build; warps 10-11 of the CTA whose range holds a page of the
    step's rows write the step's K' (the same norm, BF16) and V into the FP8 cache
    (``e4m3_rn_satfinite(bf16(x * bf16(s)))``, ``preattn_prep_append``'s bytes) and release
    that CTA's loader for those pages through the ``app`` barrier. Every thread waits for the
    predecessor before it exits. The prep's E4M3 query, fused operand, gate copy and FMHA scale
    words are not written (nothing on this path reads them).

    A class of its own so that its kernel symbol (``...FullAttnV4FoldKernel_object_at...``) is
    not the op's build's: it is audited as a kernel that waits before it exits, which the op's
    build is not. ``app_wait=False`` (the loader does not wait for the append) is a negative
    control for the tests only; ``rescale_log2`` other than 4 (the base class's rescale
    threshold) exists for the tests only too.
    """

    def __init__(
        self,
        enable_pdl: bool,
        eps: float,
        q_scale: float,
        permuted: bool,
        evict_first: bool = True,
        min_pages: int = 4,
        loader_wait: bool = True,
        app_wait: bool = True,
        rescale_cols: int = D,
        rescale_log2: float = 4.0,
        kv_heads: int = HKV,
        max_cta_pages: int = MAX_CTA_PAGES,
    ):
        super().__init__(
            enable_pdl=enable_pdl,
            prefetch=False,
            evict_first=evict_first,
            min_pages=min_pages,
            loader_wait=loader_wait,
            rescale_cols=rescale_cols,
            rescale_log2=rescale_log2,
            kv_heads=kv_heads,
            max_cta_pages=max_cta_pages,
        )
        self.fold = (float(eps), float(q_scale), bool(permuted))
        self.app_wait = app_wait
        self.qkv_width = self.raw_width

    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_tile: cute.TiledMma,
        qk_mma_box: cute.TiledMma,
        pv_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        tma_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_v: cute.Tensor,
        qkv: cute.Tensor,
        table: cute.Tensor,
        seq_lens: cute.Tensor,
        kv_scale: cute.Tensor,
        part_o: cute.Tensor,
        part_lse: cute.Tensor,
        rows: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKb_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        pool: cute.Tensor,
    ):
        self.body(
            qk_mma,
            qk_mma_tile,
            qk_mma_box,
            pv_mma,
            tma_atom_k,
            tma_k,
            tma_atom_v,
            tma_v,
            qkv,
            table,
            seq_lens,
            kv_scale,
            part_o,
            part_lse,
            rows,
            sQ_layout,
            sK_layout,
            sKb_layout,
            sV_layout,
            p_layout,
            pool,
        )
