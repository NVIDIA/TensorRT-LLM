# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small-batch NVFP4 routed experts: FC1 + SiTU + NVFP4 requantization + FC2 + combine for at
most 32 tokens in one persistent, dynamically scheduled grid (SM100 family; dispatched on Rubin
by ``TrtllmCutedslFusedFc12Nvfp4Impl``).

It reads the fused FC12 kernel's inputs: the NVFP4 tokens with linear scales, the Rubin FC1
weight layout (``[gate16, up16]`` row groups), the FC2 weights and the tile-128 ``moe_sort``
tables, and writes every element of the routed output.

* Work list: every FC1 unit (one 128-row W1 block of one tile, tile-major), then every FC2 task
  (one 128-channel m-tile of the output over one slice of consecutive tiles, slice-major). A tile
  is one local expert and its <= 32 tokens. The slice count is
  ``min(max_slices, ceil(nt / slice_tiles))`` for the ``nt`` tiles of the step, computed on the
  device. CTAs claim units from a global counter.
* FC2's token operand (the FC1 output rows of the tile and their block scales) is streamed per
  256-wide k-tile next to the weight stage it multiplies: every ring stage carries the weights
  (128 x 256 NVFP4 + scales) and a token part (``n_tok`` x 256 NVFP4 + one 2 KB scale block), so
  experts with the whole intermediate (expert parallelism) are served. The weight TMA warp streams
  the weights; the token warp waits until the tile's FC1 units are stored, then fills the token
  part of the same stages (one full barrier for both parts).
* Combine: the warpgroup that owns a task accumulates ``alpha2[e] * scale[token, slot] * acc`` of
  its tiles per token in FP32. Red combine (``deterministic_combine=False``): every task adds its
  BF16-rounded per-token sums into the output (zeroed by the kernel first) with vector reds, in an
  unspecified order. Slice combine (``deterministic_combine=True``): every task stores one FP32
  partial per (m-tile, slice) plus a token mask; the task that completes the m-tile's count sums
  the partials in slice order and writes the BF16 rows once, so the output does not depend on the
  schedule.
* Token-indexed layout (``token_indexed``, for MoE-TP slices of the intermediate): row
  ``tile * 128 + j`` of the FC1 output is token j's intermediate for the tile's expert, for every
  token of the step, so FC2's accumulator column j is token j in every tile and the FC2 epilogue
  sums ``scale[tile, j] * acc[:, j]`` in registers, the per-tile scales coming from a shared-memory
  slot table built once (expanded slot -> tile, ``alpha2 x weight``). Same FP32 operations in the
  same order as the permuted layout, so the slice combine gives the same bits.
* Dependencies: after the stores of an FC1 unit the epilogue adds 1 to the tile's counter
  (GPU-scope release); the token warp acquires ``n1 / 128`` (plus the zeroed output slices with
  the red combine) before loading a tile's rows. Every FC2 task adds 1 more per tile after its
  wait; the add that completes the count stores zero, and the CTA that exits last resets the work
  and exit counters, so the counter buffer is zero after every launch. FC1 units have no
  dependency and every CTA runs its FC1 units before its FC2 tasks, so the grid cannot deadlock
  with any number of resident CTAs.
* Die-major weights (``die_major``, for two locality domains): every expert's first half of W1
  row blocks and of W2 output m-tiles lives in the first half of the weight tensors (domain 0),
  the second halves in the second half (domain 1), as tensors ``[2 L, rows / 2, cols]`` (slot e =
  first half of expert e, slot L + e = second half). FC1 units and FC2 tasks are claimed per
  domain from four counters (own domain's FC1, the other's, own FC2, the other's), so both domains
  hold the same work for any routing and every weight load is domain-local until the steals. The
  units and their arithmetic are the default ones (same bits).
* Warp roles: epilogue warpgroups (warpgroup 0 also converts the token scales and runs every FC1
  epilogue; FC2 tasks alternate between the warpgroups), the MMA warp, the weight TMA warp, the
  token warp.
* Published routing (``published_routing``, token-indexed layout): instead of the ``moe_sort``
  tables the kernel takes the step's top-k ids / weights, which the routing kernel before it
  publishes token by token (stores, then a GPU-scope release add to the counter word
  ``CTR_ROUTE``). The CTAs poll that word, build this rank's tables in shared memory (the active
  local experts in ascending order are the tiles, ``moe_sort``'s tile order at <= 32 tokens) and
  start the weight stream without waiting for the routing grid; the token warp and the epilogue
  wait for it before they read the tokens or write the output. Same tables, same arithmetic.
* Output push (``output_push_world``): the task that combines an m-tile (slice combine), or the
  task whose arrival completes the m-tile after its reds (red combine: it reads the m-tile's rows
  back), also stores the BF16 rows into slot ``(token, rank)`` of every rank's current buffer of
  a one-shot MNNVL all-reduce (``mnnvl_pushed_allreduce.push_slot``, ``-0`` sent as ``+0``), for
  a consumer that finishes that all-reduce. Without a local expert the token warps push zero rows.
* PDL: barrier setup and TMEM allocation run before ``griddepcontrol.wait``; the dependents are
  released right after it, with published routing right after the tables (every consumer of the
  output waits for the whole grid).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm, math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from ....kernels.blackwell import mnnvl_pushed_allreduce as pushed_ar
from .utils import fmin, griddepcontrol_launch_dependents, griddepcontrol_wait

#: L2 eviction hint for the streamed expert weights (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: Weight rows per MMA (UMMA M).
_MMA_M = 128
#: K extent of one pipeline stage (four 64-wide block-scaled MMAs).
_KBLK = 256
_SF_VEC = 16
#: Bytes of one 128-row x 256-K block of scale factors (four 512-byte 32x4x4 atoms).
_SF_KBLK_BYTES = 2048
#: Token tiles (UMMA N).
SUPPORTED_N_TOK = (8, 16, 32)
#: Largest token count of the kernel.
MAX_TOKENS = SUPPORTED_N_TOK[-1]
#: FP32 padding per staged accumulator row (bank spread of the transposed reads).
_SACC_PAD = 4
#: Named barrier of epilogue warpgroup ``wg``: ``_EPI_BARRIER_BASE + wg``.
_EPI_BARRIER_BASE = 1
#: int32 words of the counter buffer: [0] work (die-major: FC1 units of domain 0), [1] exited CTAs,
#: die-major claiming: [CTR_DIE1] FC1 units of domain 1, [CTR_FC2] / [CTR_FC2_DIE1] FC2 tasks of
#: domain 0 / 1 (own 128-byte lines); [CTR_MTILE_BASE + c] FC2 arrivals of m-tile c,
#: [CTR_TILE_BASE + t] FC1 units stored / FC2 tasks started of tile t.
CTR_DIE1 = 32
CTR_FC2 = 48
CTR_FC2_DIE1 = 56
#: Published routing: tokens whose top-k the routing kernel has published (release adds; the CTA
#: that exits last resets it), on a 128-byte line of its own: the CTAs poll it while others claim
#: work.
CTR_ROUTE = 64
CTR_MTILE_BASE = 96
MAX_MTILES = 64
CTR_TILE_BASE = CTR_MTILE_BASE + MAX_MTILES
#: Bytes of the weight part of one ring stage (128 x 256 NVFP4).
_SLOT_BYTES = _MMA_M * _KBLK // 2
#: Largest slice count (the partial workspace is sized for it).
MAX_SLICES = 32
#: Published routing: words of the shared-memory active-expert mask, one bit per local expert.
_ROUTE_MASK_WORDS = 28
#: Published routing: most local experts (bits of the active-expert mask).
MAX_PUBLISHED_LOCAL_EXPERTS = 32 * _ROUTE_MASK_WORDS
#: Slices whose partial loads the slice combine keeps in flight at once.
_COMBINE_CHUNK = 4
#: Red combine: slices of the output zeroed by the kernel (the CTAs that run units
#: 0 .. _ZERO_SLICES - 1); every tile counter counts them before FC2 may add into the output.
_ZERO_SLICES = 4


def n_tok_for(num_tokens: int) -> int:
    """Smallest token tile holding ``num_tokens`` tokens."""
    for n_tok in SUPPORTED_N_TOK:
        if num_tokens <= n_tok:
            return n_tok
    raise ValueError(f"{num_tokens} tokens exceed the small-batch MoE kernel ({MAX_TOKENS})")


def counter_words(max_tiles: int) -> int:
    """int32 words of the (zero-initialized) counter buffer for up to ``max_tiles`` tiles."""
    return CTR_TILE_BASE + max_tiles


def workspace_words(n2: int, n_tok: int, max_slices: int) -> int:
    """32-bit words of the slice combine's workspace: token masks, then FP32 partials of every
    (m-tile, slice); nothing in it needs initialization."""
    mtiles = n2 // _MMA_M
    return mtiles * max_slices * (1 + n_tok * _MMA_M)


def _smem_capacity() -> int:
    """Shared memory per CTA of the current device."""
    major, minor = torch.cuda.get_device_capability()
    return utils.get_smem_capacity_in_bytes(f"sm_{major}{minor}")


def _align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ldg_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _stg_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_v4_zero(addr, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {0, 0, 0, 0};",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32_pred(addr, value, pred, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
            _iv(pred, loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.global.f32 [$0], $1; }",
        "l,f,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_cg_f32_pred(addr, pred, *, loc=None, ip=None):
    """``[addr]`` (L2) if ``pred`` != 0, else 0.0; no branch, so a run of them is in flight at once."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; @p ld.global.cg.f32 $0, [$1]; }",
            "=f,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg_cg_v4(addr, *, loc=None, ip=None):
    """16 bytes at ``addr`` read from L2 (other CTAs' reds), as four 32-bit words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _stg_bf16_from_f32(addr, value, *, loc=None, ip=None):
    """Round ``value`` to BF16 (nearest even) and store the 2 bytes."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_v8_bf16_from_f32_smem(dst_addr, src_smem_addr, *, loc=None, ip=None):
    """``dst[0:8] += bf16(src[0:8])``: 8 FP32 read from shared memory, rounded to BF16, one
    vector red."""
    _asm(
        None,
        [Int64(dst_addr).ir_value(loc=loc, ip=ip), _iv(src_smem_addr, loc=loc, ip=ip)],
        "{\n.reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;\n.reg .b32 a, b, c, d;\n"
        "ld.shared.v4.f32 {f0, f1, f2, f3}, [$1];\n"
        "ld.shared.v4.f32 {f4, f5, f6, f7}, [$1+16];\n"
        "cvt.rn.bf16x2.f32 a, f1, f0;\ncvt.rn.bf16x2.f32 b, f3, f2;\n"
        "cvt.rn.bf16x2.f32 c, f5, f4;\ncvt.rn.bf16x2.f32 d, f7, f6;\n"
        "red.global.add.noftz.v4.bf16x2 [$0], {a, b, c, d};\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _signal_add_release(addr, *, loc=None, ip=None):
    """GPU-scope release add of 1: every write that happens-before it (in particular the CTA's
    writes ordered by a preceding ``bar.sync``) is visible to an acquire that observes it."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], 1;",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_or(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_bits(lo, hi, *, loc=None, ip=None):
    """Two FP32 values rounded to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bit(n, *, loc=None, ip=None):
    """``1 << n`` for ``0 <= n < 32``."""
    return Int32(
        _asm(T.i32(), [_iv(n, loc=loc, ip=ip)], "shl.b32 $0, 1, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _atom_add_relaxed(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atom_add_acq_rel(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst_smem, src_gmem, nbytes, mbar, *, loc=None, ip=None):
    """``nbytes`` contiguous bytes global -> shared, crediting ``mbar``."""
    _asm(
        None,
        [
            _iv(dst_smem, loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ballot(pred, *, loc=None, ip=None):
    """Warp ballot of ``pred != 0`` (every lane of the warp executes it)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(pred, loc=loc, ip=ip)],
            "{ .reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smid(*, loc=None, ip=None):
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc=loc, ip=ip))


def pack_sm_domain_words(domains: tuple) -> tuple:
    """Bit ``s % 32`` of word ``s // 32`` is the locality domain of SM ``s`` (signed 32-bit)."""
    words = [0] * (-(-len(domains) // 32))
    for sm, domain in enumerate(domains):
        if domain == 1:
            words[sm // 32] |= 1 << (sm % 32)
    return tuple(w - (1 << 32) if w >= (1 << 31) else w for w in words)


def _s2t_copy_and_partition(sSF: cute.Tensor, tSF: cute.Tensor, sf_dtype):
    """tcgen05.cp (SMEM -> TMEM) partitions of a scale-factor operand."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype)
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


class RubinSmallBatchFusedMoeKernel:
    """FC1 + SiTU + NVFP4 quantization + FC2 + combine for at most 32 tokens.

    :param n_tok: Token tile (MMA N) of both GEMMs: 8, 16 or 32.
    :param n1: Rows of one expert's W1 (2 x intermediate, ``[gate16, up16]`` row groups).
    :param k1: Hidden size (W1 K, W2 rows, output columns).
    :param topk: Experts per token.
    :param situ_beta: SiTU gate-side constant.
    :param situ_linear_beta: SiTU up-side constant.
    :param token_indexed: Token-indexed FC1 output layout.
    :param claim_free: Ring stages a CTA needs free before it claims its next FC2 task (0: one unit
        ahead).
    :param max_slices: Largest number of FC2 slices.
    :param max_ab_stage: Cap on the ring depth (the rest of shared memory sizes it).
    :param deterministic_combine: Slice combine instead of the BF16 vector reds.
    :param num_acc: Accumulator stages in TMEM per epilogue warpgroup.
    :param num_epi_wg: Epilogue warpgroups (1 or 2).
    :param slice_tiles: Target tiles per FC2 slice.
    :param queue_depth: Entries of the shared-memory unit queue.
    :param use_pdl: Launch with programmatic dependent launch.
    :param die_major: Die-major expert weights, work claimed per locality domain.
    :param sm_domain_words: :func:`pack_sm_domain_words` of the device (with ``die_major``).
    :param published_routing: Tables built in the kernel from the top-k the routing kernel
        publishes (token-indexed layout only).
    :param output_push_world: Also push the output rows into a one-shot all-reduce over that many
        ranks (either combine; 0: off).
    """

    @staticmethod
    def supports(n_tok: int, n1: int, k1: int) -> bool:
        # FC1 runs in 128-row W1 blocks; FC2's K (the intermediate n1 / 2) needs whole 64-wide
        # scale atoms.
        return (
            n_tok in SUPPORTED_N_TOK
            and n1 % _MMA_M == 0
            and (n1 // 2) % 64 == 0
            and k1 % _KBLK == 0
            and k1 // _MMA_M <= MAX_MTILES
        )

    @staticmethod
    def h_sf_tile_bytes(n1: int) -> int:
        """Bytes of one tile's FC1-output scales (128 rows; the intermediate padded to a multiple
        of 256, so a half last k-tile of FC2 reads zeros there: the buffer's padding must be zero)."""
        return _MMA_M * ((n1 // 2 + _KBLK - 1) // _KBLK * _KBLK) // _SF_VEC

    def __init__(
        self,
        n_tok: int,
        n1: int,
        k1: int,
        topk: int,
        situ_beta: float,
        situ_linear_beta: float,
        token_indexed: bool,
        claim_free: int,
        max_slices: int,
        max_ab_stage: int,
        deterministic_combine: bool = False,
        num_acc: int = 2,
        num_epi_wg: int = 2,
        slice_tiles: int = 1,
        queue_depth: int = 16,
        use_pdl: bool = True,
        die_major: bool = False,
        sm_domain_words: tuple = (),
        published_routing: bool = False,
        output_push_world: int = 0,
    ):
        assert self.supports(n_tok, n1, k1), (n_tok, n1, k1)
        self.output_push_world = output_push_world
        assert not published_routing or token_indexed, "published routing is token-indexed"
        self.published_routing = published_routing
        assert not die_major or sm_domain_words, "die-major claiming needs the SM table"
        self.die_major = die_major
        self.sm_domain_words = tuple(sm_domain_words)
        assert num_epi_wg in (1, 2) and num_acc >= 2
        assert 1 <= max_slices <= MAX_SLICES and slice_tiles >= 1
        assert situ_beta > 0 and situ_linear_beta > 0
        self.token_indexed = token_indexed
        # Claim policy: 0 = one unit ahead (the next unit is claimed when the current one starts
        # loading); n > 0 = once the FC1 units are all claimed, the next unit is claimed only when
        # n ring stages are free for it (FC1 units are still claimed right after the current one).
        self.claim_free_req = int(claim_free)
        self.combine_red = not deterministic_combine
        self.ready_target = n1 // _MMA_M + (_ZERO_SLICES if self.combine_red else 0)
        self.n_tok = n_tok
        self.n1, self.k1 = n1, k1
        self.n2, self.k2 = k1, n1 // 2
        self.topk = topk
        self.nkb1, self.nblk1 = k1 // _KBLK, n1 // _MMA_M
        # FC2's K (the intermediate) may end in a half k-tile: the TMA zero-fills the weights,
        # their scales and the rows past K, the FC1-output scales are padded.
        self.nkb2, self.nblk2 = (self.k2 + _KBLK - 1) // _KBLK, self.n2 // _MMA_M
        self.h_tile_sf_bytes = self.h_sf_tile_bytes(n1)
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)
        self.num_epi_wg = num_epi_wg
        self.num_acc = num_acc
        self.nacc_total = num_acc * num_epi_wg
        self.slice_tiles = slice_tiles
        self.max_slices = max_slices
        self.queue_depth = queue_depth
        self.use_pdl = use_pdl
        self.mma_tiler = (_MMA_M, n_tok, _KBLK)
        self.acc_cols = n_tok * self.nacc_total
        self.sfa_cols = 16
        self.sfb_cols = 16
        cols = self.acc_cols + self.sfa_cols + self.sfb_cols
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2
        self.a_stage_bytes = _SLOT_BYTES + _SF_KBLK_BYTES
        self.b_stage_bytes = n_tok * _KBLK // 2
        self.fc2_stage_bytes = self.a_stage_bytes + self.b_stage_bytes + _SF_KBLK_BYTES
        self.x_bytes = self.nkb1 * self.b_stage_bytes
        self.sacc_row_floats = _MMA_M + _SACC_PAD
        self.acc_buf_bytes = n_tok * self.sacc_row_floats * 4
        # slice combine workspace geometry (32-bit words): masks [mtile][max_slices], then
        # partials [mtile][max_slices][n_tok][128]
        self.mask_word_base = 0
        self.part_word_base = self.nblk2 * max_slices

        # token-indexed slot table: one entry per expanded slot (token x top-k) of the step: its
        # tile (-1: another rank's expert) and its FP32 scale; per epilogue warpgroup two buffers
        # of a tile's per-token scales and token masks.
        self.n_x = n_tok * topk
        self.n_sv = 2 * num_epi_wg * n_tok
        # Per-stage bytes: weights + their scales + FC2 token rows + their scale block.
        per_stage = _SLOT_BYTES + _SF_KBLK_BYTES + self.b_stage_bytes + _SF_KBLK_BYTES + 16
        fixed = (
            self.x_bytes
            + self.nkb1 * _SF_KBLK_BYTES
            + (self.acc_buf_bytes if num_epi_wg > 1 else 0)
            + self.acc_buf_bytes
            + 2 * num_epi_wg * n_tok * 4
            + 2 * queue_depth * 4
            + 8 * self.n_x
            + 4 * self.n_sv
            + 8 * num_epi_wg
            + 64
            + 4 * 1024  # barriers, TMEM holder, flags and alignment slack
        )
        # published routing: tile -> expert table (at most one tile per expanded slot), the
        # active-expert mask with its word prefix counts, the tile count
        self.route_t2e = self.n_x if published_routing else 4
        if published_routing:
            fixed += 4 * self.route_t2e + 8 * _ROUTE_MASK_WORDS + 64
        self.num_ab_stage = min(max_ab_stage, (_smem_capacity() - fixed) // per_stage)
        assert self.num_ab_stage >= 2, "ring too small"
        self.claim_free = min(self.claim_free_req, self.num_ab_stage)
        if self.die_major:
            assert self.nblk1 % 2 == 0 and self.nblk2 % 2 == 0, "die-major halves need even blocks"
        S = self.num_ab_stage
        # Region after the weight ring: [B2 stages | x] (token-operand view), then
        # [SFB2 stages | x scales] (token-scale view), then warpgroup 1's token accumulator.
        self.off_b2 = 0
        self.off_x = S * self.b_stage_bytes
        self.off_sfb2 = _align(self.off_x + self.x_bytes, 1024)
        self.off_xsf = self.off_sfb2 + S * _SF_KBLK_BYTES
        self.off_acc1 = _align(self.off_xsf + self.nkb1 * _SF_KBLK_BYTES, 128)
        end = self.off_acc1 + (self.acc_buf_bytes if num_epi_wg > 1 else 0)
        self.region_bytes = _align(end, 1024)
        self.epi_warps = 4 * num_epi_wg
        self.mma_warp = self.epi_warps
        self.tma_warp = self.epi_warps + 1
        self.tok_warp = self.epi_warps + 2
        self.threads = 32 * (self.epi_warps + 3)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w1: cute.Tensor,
        w1_sf: cute.Tensor,
        h: cute.Tensor,
        w2: cute.Tensor,
        w2_sf: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha1: cute.Tensor,
        alpha2: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        token_final_scales: cute.Tensor,
        ctr_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        orig_m: Int32,
        expert_offset: Int32,
        push_mc: Int64,
        push_flags: Int64,
        push_rank: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.ab_dtype = w1.element_type
        self.sf_dtype = w1_sf.element_type
        major = utils.LayoutEnum.from_tensor(w1).mma_major_mode()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            major,
            major,
            self.sf_dtype,
            _SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        S = self.num_ab_stage
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, S)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, _SF_VEC, S)
        # Token-operand views: stages [0, S) are the ring's FC2 token parts, [S, S + nkb1) the
        # step's tokens (FC1); the same for the scales.
        self.b_view_stages = S + self.nkb1
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.b_view_stages
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, _SF_VEC, self.b_view_stages
        )
        w1_sf = cute.make_tensor(
            w1_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w1.shape, _SF_VEC)
        )
        w2_sf = cute.make_tensor(
            w2_sf.iterator, blockscaled_utils.tile_atom_to_shape_SF(w2.shape, _SF_VEC)
        )
        cl = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_a1, tA1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w1, a_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_sfa1, tSFA1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w1_sf, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        tma_a2, tA2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2, a_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_sfa2, tSFA2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2_sf, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        tma_b1, tB1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_one, self.mma_tiler, tiled_mma, cl.shape
        )
        tma_b2, tB2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, h, b_one, self.mma_tiler, tiled_mma, cl.shape
        )

        NACC, NWG, Q = self.nacc_total, self.num_epi_wg, self.queue_depth

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, S]
            a_empty: cute.struct.MemRange[cutlass.Int64, S]
            acc_full: cute.struct.MemRange[cutlass.Int64, NACC]
            acc_empty: cute.struct.MemRange[cutlass.Int64, NACC]
            uq_full: cute.struct.MemRange[cutlass.Int64, Q]
            uq_empty: cute.struct.MemRange[cutlass.Int64, Q]
            tok_full: cutlass.Int64
            sfb_full: cutlass.Int64
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sUnits: cute.struct.Align[cute.struct.MemRange[Int32, Q], 16]
            sStages: cute.struct.Align[cute.struct.MemRange[Int32, Q], 16]
            sFlag: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sRegion: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.region_bytes], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_layout)], 1024
            ]
            sAcc: cute.struct.Align[
                cute.struct.MemRange[Float32, self.n_tok * self.sacc_row_floats], 16
            ]
            sTok: cute.struct.Align[cute.struct.MemRange[Int32, NWG * self.n_tok], 16]
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, NWG * self.n_tok], 16]
            sTl: cute.struct.Align[cute.struct.MemRange[Int32, self.n_x], 16]
            sSc: cute.struct.Align[cute.struct.MemRange[Float32, self.n_x], 16]
            sSv: cute.struct.Align[cute.struct.MemRange[Float32, self.n_sv], 16]
            sSvMask: cute.struct.Align[cute.struct.MemRange[Int32, 2 * NWG], 16]
            sT2e: cute.struct.Align[cute.struct.MemRange[Int32, self.route_t2e], 16]
            sAm: cute.struct.Align[cute.struct.MemRange[Int32, _ROUTE_MASK_WORDS], 16]
            sApre: cute.struct.Align[cute.struct.MemRange[Int32, _ROUTE_MASK_WORDS], 16]
            sNt: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_a1,
            tA1,
            tma_sfa1,
            tSFA1,
            tma_a2,
            tA2,
            tma_sfa2,
            tSFA2,
            tma_b1,
            tB1,
            tma_b2,
            tB2,
            x_sf_ptr,
            h_ptr,
            h_sf_ptr,
            out_ptr,
            alpha1,
            alpha2,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            expanded_idx_to_permuted_idx,
            num_non_exiting_tiles,
            global_sf,
            token_final_scales,
            ctr_ptr,
            ws_ptr,
            orig_m,
            expert_offset,
            push_mc,
            push_flags,
            push_rank,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            cl,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    def _b_view(self, st, layout):
        base = st.sRegion.data_ptr() + self.off_b2
        return cute.make_tensor(
            cute.recast_ptr(base, layout.inner, dtype=self.ab_dtype), layout.outer
        )

    def _sfb_view(self, st, layout):
        base = st.sRegion.data_ptr() + self.off_sfb2
        return cute.make_tensor(cute.recast_ptr(base, dtype=self.sf_dtype), layout)

    def _region_addr(self, st, off: int):
        return (st.sRegion.data_ptr() + off).toint()

    @cute.jit
    def _wg_barrier(self, wg: Int32):
        for i in cutlass.range_constexpr(self.num_epi_wg):
            if wg == i:
                cute.arch.barrier(barrier_id=_EPI_BARRIER_BASE + i, number_of_threads=128)

    @cute.jit
    def _fc2_task(self, v: Int32, nt: Int32, ns: Int32):
        """FC2 task ``v`` (index past the FC1 units): (m-tile, combine slice, tiles [t0, t1))."""
        s = v // self.nblk2
        c = v - s * self.nblk2
        t0, t1 = self._slice_bounds(s, nt, ns)
        return c, s, t0, t1

    @cute.jit
    def _slice_bounds(self, s: Int32, nt: Int32, ns: Int32):
        """Tiles [t0, t1) of slice ``s`` of ``ns`` over ``nt`` tiles."""
        return s * nt // ns, (s + 1) * nt // ns

    # ------------------------------------------------------------------ unit queue
    @cute.jit
    def _q_write(self, st, j: Int32, u: Int32, g: Int32):
        """Publish the CTA's ``j``-th unit (-1: no more units) and its first ring index."""
        Q = self.queue_depth
        q = j % Q
        if j >= Q:
            cute.arch.mbarrier_wait(st.uq_empty.data_ptr() + q, ((j // Q) & 1) ^ 1)
        units = st.sUnits.get_tensor(cute.make_layout(Q))
        stages = st.sStages.get_tensor(cute.make_layout(Q))
        with cute.arch.elect_one():
            units[q] = u
            stages[q] = g
            cute.arch.mbarrier_arrive(st.uq_full.data_ptr() + q)

    @cute.jit
    def _q_read(self, st, j: Int32):
        """The CTA's ``j``-th unit and its first ring index (whole warp)."""
        Q = self.queue_depth
        q = j % Q
        cute.arch.mbarrier_wait(st.uq_full.data_ptr() + q, (j // Q) & 1)
        units = st.sUnits.get_tensor(cute.make_layout(Q))
        stages = st.sStages.get_tensor(cute.make_layout(Q))
        u = units[q]
        g = stages[q]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.uq_empty.data_ptr() + q)
        return u, g

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        x_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha1: cute.Tensor,
        alpha2: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        global_sf: cute.Tensor,
        token_final_scales: cute.Tensor,
        ctr_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        orig_m: Int32,
        expert_offset: Int32,
        push_mc: Int64,
        push_flags: Int64,
        push_rank: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        cl: cute.Layout,
    ):
        S = self.num_ab_stage
        NACC, Q = self.nacc_total, self.queue_depth
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas_, _, _ = cute.arch.grid_dim()
        nctas = Int32(nctas_)

        st = utils.SmemAllocator().allocate(self.shared_storage)
        if warp_idx == self.tma_warp:
            for atom in (tma_a1, tma_sfa1, tma_a2, tma_sfa2, tma_b1, tma_b2):
                cpasync.prefetch_descriptor(atom)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(st.a_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.a_empty.data_ptr() + i, 1)
                for i in cutlass.range_constexpr(NACC):
                    cute.arch.mbarrier_init(st.acc_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.acc_empty.data_ptr() + i, 4)
                for i in cutlass.range_constexpr(Q):
                    cute.arch.mbarrier_init(st.uq_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.uq_empty.data_ptr() + i, self.epi_warps + 2)
                cute.arch.mbarrier_init(st.tok_full, 1)
                cute.arch.mbarrier_init(st.sfb_full, 4)
                cute.arch.mbarrier_init(st.tmem_ready, 1)
                cute.arch.mbarrier_init(st.tmem_free, self.epi_warps)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if warp_idx == self.mma_warp:
            cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holding_buf)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(st.tmem_ready)
        if cutlass.const_expr(self.use_pdl and not self.published_routing):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        # The CTA's first claim after the grid dependency: the previous launch on this counter
        # buffer may still run until then (die-major: claimed by the weight producer). With
        # published routing the routing kernel's producer waited for every earlier grid, so the
        # claim (die-major: on the own domain's FC1 counter, resolved by the weight producer once
        # the tile count is known) runs before the tables.
        first = Int32(0)
        if cutlass.const_expr(not self.die_major or self.published_routing):
            if warp_idx == self.tma_warp:
                if cute.arch.lane_idx() == 0:
                    word = Int64(0)
                    if cutlass.const_expr(self.die_major):
                        if self._sm_domain(_smid()) == 1:
                            word = Int64(CTR_DIE1)
                    first = _atom_add_relaxed(ctr_ptr.toint() + word * 4, 1)

        t2e_pre = Int32(0)
        nt = Int32(0)
        if cutlass.const_expr(self.published_routing):
            self._published_tables(
                st, alpha2, expanded_idx_to_permuted_idx, ctr_ptr, orig_m, expert_offset, tidx
            )
            if cutlass.const_expr(self.use_pdl):
                # every earlier grid completed before the routing kernel published (its producer
                # waited for them); every consumer of the output waits for this whole grid
                griddepcontrol_launch_dependents()
            nt = st.sNt.get_tensor(cute.make_layout(4))[0]
            if warp_idx == self.tma_warp:
                lane = cute.arch.lane_idx()
                t2e_pre = st.sT2e.get_tensor(cute.make_layout(self.route_t2e))[
                    cutlass.min(lane, self.route_t2e - 1)
                ]
        else:
            if warp_idx == self.tma_warp:
                lane = cute.arch.lane_idx()
                t2e_pre = tile_idx_to_expert_idx[
                    cutlass.min(lane, cute.size(tile_idx_to_expert_idx) - 1)
                ]
            nt = Int32(num_non_exiting_tiles[0])
        w1 = nt * self.nblk1
        ns = cutlass.min(Int32(self.max_slices), (nt + (self.slice_tiles - 1)) // self.slice_tiles)
        total = w1 + ns * self.nblk2
        thr_mma = tiled_mma.get_slice(0)

        if warp_idx == self.tma_warp:
            self._weight_producer(
                st,
                thr_mma,
                cl,
                a_layout,
                sfa_layout,
                tma_a1,
                mA1,
                tma_sfa1,
                mSFA1,
                tma_a2,
                mA2,
                tma_sfa2,
                mSFA2,
                tile_idx_to_expert_idx,
                ctr_ptr,
                first,
                t2e_pre,
                w1,
                total,
                nctas,
                nt,
                ns,
            )
        elif warp_idx == self.tok_warp:
            self._token_producer(
                st,
                thr_mma,
                cl,
                b_layout,
                tma_b1,
                mB1,
                tma_b2,
                mB2,
                h_sf_ptr,
                out_ptr,
                ctr_ptr,
                orig_m,
                nt,
                ns,
                w1,
                bid,
                nctas,
                (push_mc, push_flags, push_rank),
            )
        elif warp_idx == self.mma_warp:
            self._mma(st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, w1, nt, ns)
        else:
            wg = tidx // 128
            et = tidx % 128
            cute.arch.mbarrier_wait(st.tmem_ready, 0)
            acc_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
            )
            if cutlass.const_expr(self.published_routing and self.use_pdl):
                # the token scales and the output come with the routing grid's end
                griddepcontrol_wait()
            if wg == 0:
                if w1 > 0:
                    self._token_scales_to_smem(st, x_sf_ptr, orig_m, et)
            if cutlass.const_expr(self.published_routing):
                self._published_slots(
                    st,
                    alpha2,
                    expanded_idx_to_permuted_idx,
                    token_final_scales,
                    orig_m,
                    expert_offset,
                    tidx,
                )
            elif cutlass.const_expr(self.token_indexed):
                self._build_slot_table(
                    st,
                    alpha2,
                    tile_idx_to_expert_idx,
                    expanded_idx_to_permuted_idx,
                    token_final_scales,
                    orig_m,
                    tidx,
                )
            self._epilogue(
                st,
                tiled_mma,
                acc_ptr,
                h_ptr,
                h_sf_ptr,
                out_ptr,
                alpha1,
                alpha2,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                global_sf,
                token_final_scales,
                ctr_ptr,
                ws_ptr,
                orig_m,
                w1,
                nt,
                ns,
                et,
                wg,
                (push_mc, push_flags, push_rank),
            )
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(st.tmem_free)

    # ------------------------------------------------------------------ producers
    @cute.jit
    def _reserve_stage(self, a_empty, g):
        """Wait until ring stage ``g`` may be overwritten."""
        S = self.num_ab_stage
        stage = g % S
        if g >= S:
            cute.arch.mbarrier_wait(a_empty + stage, ((g // S) & 1) ^ 1)
        return stage

    @cute.jit
    def _load_stage(
        self,
        tma_a,
        tma_sfa,
        tAsA,
        tAgA,
        tAsSFA,
        tAgSFA,
        blk,
        kb,
        expert,
        a_full,
        a_empty,
        g,
        tx_bytes: cutlass.Constexpr,
    ):
        stage = self._reserve_stage(a_empty, g)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, tx_bytes)
        # the weights are read once per step: L2 evict-first
        cute.copy(
            tma_a,
            tAgA[(None, blk, kb, expert)],
            tAsA[(None, stage)],
            tma_bar_ptr=a_full + stage,
            cache_policy=Int64(_EVICT_FIRST),
        )
        cute.copy(
            tma_sfa,
            tAgSFA[(None, blk, kb, expert)],
            tAsSFA[(None, stage)],
            tma_bar_ptr=a_full + stage,
            cache_policy=Int64(_EVICT_FIRST),
        )
        return g + 1

    @cute.jit
    def _sm_domain(self, smid: Int32) -> Int32:
        """Locality domain of the SM with ``%smid`` ``smid`` (0 beyond the table)."""
        word = Int32(0)
        for i in cutlass.range_constexpr(len(self.sm_domain_words)):
            if (smid >> 5) == i:
                word = Int32(self.sm_domain_words[i])
        return (word >> (smid & 31)) & 1

    @cute.jit
    def _dm_claim(self, ctr, own_die: Int32, phase: Int32, w1: Int32, nt: Int32, ns: Int32):
        """Die-major claiming (whole warp; lane 0 claims): phase 0 / 1 = FC1 units of the own / the
        other domain (domain d: row blocks [d nblk1 / 2, (d + 1) nblk1 / 2) of every tile,
        tile-major), phase 2 / 3 = FC2 tasks of the own / the other domain (m-tiles
        [d nblk2 / 2, ...) of every slice, slice-major), 4 = done. Returns the unit in the default
        numbering (an FC1 unit ``tile * nblk1 + blk`` or ``w1 +`` the FC2 task) and the phase."""
        lane = cute.arch.lane_idx()
        h1 = self.nblk1 // 2
        h2 = self.nblk2 // 2
        n1d = nt * h1
        n2d = ns * h2
        idx = Int32(0)
        ph = phase
        if lane == 0:
            searching = Int32(1)
            while searching != 0:
                if ph >= 4:
                    idx = w1 + 2 * n2d  # past the last unit
                    searching = Int32(0)
                else:
                    die = own_die
                    if (ph & 1) == 1:
                        die = 1 - own_die
                    if ph < 2:
                        word = Int64(0)
                        if die == 1:
                            word = Int64(CTR_DIE1)
                        i = _atom_add_relaxed(ctr + word * 4, 1)
                        if i < n1d:
                            tile = i // h1
                            idx = tile * self.nblk1 + die * h1 + (i - tile * h1)
                            searching = Int32(0)
                        else:
                            ph = ph + 1
                    else:
                        word = Int64(CTR_FC2)
                        if die == 1:
                            word = Int64(CTR_FC2_DIE1)
                        i = _atom_add_relaxed(ctr + word * 4, 1)
                        if i < n2d:
                            s2 = i // h2
                            idx = w1 + s2 * self.nblk2 + die * h2 + (i - s2 * h2)
                            searching = Int32(0)
                        else:
                            ph = ph + 1
        idx = cute.arch.shuffle_sync(idx, 0)
        ph = cute.arch.shuffle_sync(ph, 0)
        return idx, ph

    @cute.jit
    def _expert_of(self, st, tile_idx_to_expert_idx, t2e_pre: Int32, tile: Int32):
        """Expert of ``tile`` (warp-uniform): from the prefetched lanes for the first 32 tiles
        (published routing: the shared-memory table)."""
        expert = Int32(0)
        if tile < 32:
            expert = cute.arch.shuffle_sync(t2e_pre, tile)
        else:
            if cutlass.const_expr(self.published_routing):
                expert = st.sT2e.get_tensor(cute.make_layout(self.route_t2e))[tile]
            else:
                expert = tile_idx_to_expert_idx[tile]
        return expert

    @cute.jit
    def _weight_producer(
        self,
        st,
        thr_mma,
        cl,
        a_layout,
        sfa_layout,
        tma_a1,
        mA1,
        tma_sfa1,
        mSFA1,
        tma_a2,
        mA2,
        tma_sfa2,
        mSFA2,
        tile_idx_to_expert_idx,
        ctr_ptr,
        first: Int32,
        t2e_pre: Int32,
        w1: Int32,
        total: Int32,
        nctas: Int32,
        nt: Int32,
        ns: Int32,
    ):
        """Fetches the CTA's units from the work counter (one ahead, or with ``claim_free`` once
        enough ring stages are free), publishes each with its first ring index and streams its
        weight stages; the CTA that exits last resets the work and exit counters."""
        a_full, a_empty = st.a_full.data_ptr(), st.a_empty.data_ptr()
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        a_cta = cute.make_layout(cute.slice_(cl, (0, 0, None, 0)).shape)
        tiler = cute.slice_(self.mma_tiler, (None, 0, None))
        parts = []
        for mA, mSFA, tma_a, tma_sfa in (
            (mA1, mSFA1, tma_a1, tma_sfa1),
            (mA2, mSFA2, tma_a2, tma_sfa2),
        ):
            tCgA = thr_mma.partition_A(cute.local_tile(mA, tiler, (None, None, None)))
            tCgSFA = thr_mma.partition_A(cute.local_tile(mSFA, tiler, (None, None, None)))
            tAsA, tAgA = cpasync.tma_partition(
                tma_a, 0, a_cta, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_sfa, 0, a_cta, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
            )
            parts.append((tAsA, tAgA, cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)))
        tAsA1, tAgA1, tAsSFA1, tAgSFA1 = parts[0]
        tAsA2, tAgA2, tAsSFA2, tAgSFA2 = parts[1]
        lane = cute.arch.lane_idx()
        ctr = ctr_ptr.toint()
        cur = cute.arch.shuffle_sync(first, 0)
        own_die = Int32(0)
        queue = Int32(0)
        nloc = Int32(0)
        if cutlass.const_expr(self.die_major):
            nloc = Int32(cute.size(mA1, mode=[2])) // 2
            own_die = self._sm_domain(_smid())
            if cutlass.const_expr(self.published_routing):
                # the claim before the tables: an own-domain FC1 unit, else on to the other's
                h1 = self.nblk1 // 2
                if cur < nt * h1:
                    tile0 = cur // h1
                    cur = tile0 * self.nblk1 + own_die * h1 + (cur - tile0 * h1)
                else:
                    cur, queue = self._dm_claim(ctr, own_die, Int32(1), w1, nt, ns)
            else:
                cur, queue = self._dm_claim(ctr, own_die, queue, w1, nt, ns)
        g = Int32(0)
        j = Int32(0)
        while cur < total:
            nxt = Int32(0)
            if cutlass.const_expr(self.claim_free == 0):
                if cutlass.const_expr(self.die_major):
                    nxt, queue = self._dm_claim(ctr, own_die, queue, w1, nt, ns)
                else:
                    if lane == 0:
                        nxt = _atom_add_relaxed(ctr, 1)
            self._q_write(st, j, cur, g)
            if cur < w1:
                tile = cur // self.nblk1
                blk = cur - tile * self.nblk1
                expert = self._expert_of(st, tile_idx_to_expert_idx, t2e_pre, tile)
                tblk = blk
                texp = expert
                if cutlass.const_expr(self.die_major):
                    # die-major slot of this row block: second half -> slot L + e
                    if blk >= self.nblk1 // 2:
                        tblk = blk - self.nblk1 // 2
                        texp = expert + nloc
                for kb in cutlass.range(self.nkb1, unroll=1):
                    g = self._load_stage(
                        tma_a1,
                        tma_sfa1,
                        tAsA1,
                        tAgA1,
                        tAsSFA1,
                        tAgSFA1,
                        tblk,
                        kb,
                        texp,
                        a_full,
                        a_empty,
                        g,
                        self.a_stage_bytes,
                    )
            else:
                c, _s, t0, t1 = self._fc2_task(cur - w1, nt, ns)
                tc = c
                eoff = Int32(0)
                if cutlass.const_expr(self.die_major):
                    if c >= self.nblk2 // 2:
                        tc = c - self.nblk2 // 2
                        eoff = nloc
                for t in cutlass.range(t0, t1, unroll=1):
                    expert = self._expert_of(st, tile_idx_to_expert_idx, t2e_pre, t) + eoff
                    for kb in cutlass.range(self.nkb2, unroll=1):
                        g = self._load_stage(
                            tma_a2,
                            tma_sfa2,
                            tAsA2,
                            tAgA2,
                            tAsSFA2,
                            tAgSFA2,
                            tc,
                            kb,
                            expert,
                            a_full,
                            a_empty,
                            g,
                            self.fc2_stage_bytes,
                        )
            if cutlass.const_expr(self.claim_free > 0):
                # while FC1 units are left (they never wait on other CTAs) the next unit is claimed
                # right away; FC2 tasks only once claim_free ring stages are free for them
                peek = Int32(0)
                if lane == 0:
                    if cutlass.const_expr(self.die_major):
                        # FC1 work is left for this CTA until it is past both domains' FC1 queues
                        if queue >= 2:
                            peek = w1
                    else:
                        peek = _ld_relaxed_u32(ctr)
                peek = cute.arch.shuffle_sync(peek, 0)
                gw = g + (self.claim_free - 1)
                if peek >= w1:
                    if gw >= self.num_ab_stage:
                        cute.arch.mbarrier_wait(
                            a_empty + gw % self.num_ab_stage, ((gw // self.num_ab_stage) & 1) ^ 1
                        )
                if cutlass.const_expr(self.die_major):
                    nxt, queue = self._dm_claim(ctr, own_die, queue, w1, nt, ns)
                else:
                    if lane == 0:
                        nxt = _atom_add_relaxed(ctr, 1)
            cur = cute.arch.shuffle_sync(nxt, 0)
            j += 1
        self._q_write(st, j, Int32(-1), g)
        if lane == 0:
            # every fetch of this CTA happens before its release; the last CTA's acquire sees all
            if _atom_add_acq_rel(ctr + 4, 1) == nctas - 1:
                _st_relaxed_u32(ctr, 0)
                _st_relaxed_u32(ctr + 4, 0)
                # a routing kernel may have published into it for a launch with tables too
                _st_relaxed_u32(ctr + CTR_ROUTE * 4, 0)
                if cutlass.const_expr(self.die_major):
                    _st_relaxed_u32(ctr + CTR_DIE1 * 4, 0)
                    _st_relaxed_u32(ctr + CTR_FC2 * 4, 0)
                    _st_relaxed_u32(ctr + CTR_FC2_DIE1 * 4, 0)

    @cute.jit
    def _token_producer(
        self,
        st,
        thr_mma,
        cl,
        b_layout,
        tma_b1,
        mB1,
        tma_b2,
        mB2,
        h_sf_ptr,
        out_ptr,
        ctr_ptr,
        orig_m: Int32,
        nt: Int32,
        ns: Int32,
        w1: Int32,
        bid,
        nctas: Int32,
        push,
    ):
        """The step's tokens (FC1's token operand) once; then, for every tile of every FC2 task,
        once FC1 has stored all of the tile's rows, its rows and scales of each k-tile into the
        token part of the ring stage that holds the matching weights."""
        S = self.num_ab_stage
        b_cta = cute.make_layout(cute.slice_(cl, (0, None, 0, 0)).shape)
        tiler = cute.slice_(self.mma_tiler, (0, None, None))
        sB = self._b_view(st, b_layout)
        tCgB1 = thr_mma.partition_B(cute.local_tile(mB1, tiler, (None, None, None)))
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_b1, 0, b_cta, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB1, 0, 3)
        )
        tCgB2 = thr_mma.partition_B(cute.local_tile(mB2, tiler, (None, None, None)))
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_b2, 0, b_cta, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB2, 0, 3)
        )
        if cutlass.const_expr(self.published_routing and self.use_pdl):
            # the tokens (and the output this warp zeroes) come with the routing grid's end
            griddepcontrol_wait()
        if w1 > 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(st.tok_full, self.x_bytes)
            for kb in cutlass.range(self.nkb1, unroll=1):
                cute.copy(
                    tma_b1,
                    tBgB1[(None, 0, kb, 0)],
                    tBsB1[(None, S + kb)],
                    tma_bar_ptr=st.tok_full,
                )
        a_full, a_empty = st.a_full.data_ptr(), st.a_empty.data_ptr()
        sfb2_base = self._region_addr(st, self.off_sfb2)
        h_sf_base = h_sf_ptr.toint()
        tile_sf_bytes = self.h_tile_sf_bytes
        ctr = ctr_ptr.toint()
        lane = cute.arch.lane_idx()
        out_bytes = orig_m * (self.n2 * 2)
        if nt == 0:
            # no local expert: no FC2 task writes the output, CTA 0 zeroes it
            if bid == 0:
                out_base = out_ptr.toint()
                for off in cutlass.range(lane * 16, out_bytes, 32 * 16, unroll=1):
                    _stg_v4_zero(out_base + Int64(off))
            if cutlass.const_expr(self.output_push_world > 0):
                # No FC2 task pushes the output either, so every CTA pushes zero rows: the
                # consumer polls every rank's slot of every token.
                self._push_zero_rows(push, orig_m, Int32(bid) * 32 + lane, nctas * 32)

        slice_bytes = (out_bytes + _ZERO_SLICES * 16 - 1) // (_ZERO_SLICES * 16) * 16
        e = Int32(0)
        u, g = self._q_read(st, e)
        while u >= 0:
            if cutlass.const_expr(self.combine_red):
                if u < _ZERO_SLICES:
                    # zero slice u of the output, then count it on every tile: FC2 tasks of a tile
                    # add into the output only after all slices (the warp barrier orders every
                    # lane's stores before the releases)
                    lo = u * slice_bytes
                    hi = cutlass.min(lo + slice_bytes, out_bytes)
                    for off in cutlass.range(lo + lane * 16, hi, 32 * 16, unroll=1):
                        _stg_v4_zero(out_ptr.toint() + Int64(off))
                    cute.arch.sync_warp()
                    for t in cutlass.range(lane, nt, 32, unroll=1):
                        _red_add_release(ctr + Int64(CTR_TILE_BASE + t) * 4, 1)
            if u >= w1:
                _c, _s, t0, t1 = self._fc2_task(u - w1, nt, ns)
                for t in cutlass.range(t0, t1, unroll=1):
                    ready_addr = ctr + Int64(CTR_TILE_BASE + t) * 4
                    ready = _ld_acquire_u32(ready_addr)
                    while ready < self.ready_target:
                        ready = _ld_acquire_u32(ready_addr)
                    cute.arch.fence_proxy("async.global")
                    h_sf_src = h_sf_base + Int64(t) * tile_sf_bytes
                    for kb in cutlass.range(self.nkb2, unroll=1):
                        # the weight producer expects this part's bytes on the same barrier; the
                        # stage is free once the MMAs of its previous use are done
                        stage = self._reserve_stage(a_empty, g)
                        cute.copy(
                            tma_b2,
                            tBgB2[(None, t * (_MMA_M // self.n_tok), kb, 0)],
                            tBsB2[(None, stage)],
                            tma_bar_ptr=a_full + stage,
                        )
                        with cute.arch.elect_one():
                            _bulk_g2s(
                                sfb2_base + stage * _SF_KBLK_BYTES,
                                h_sf_src + kb * _SF_KBLK_BYTES,
                                _SF_KBLK_BYTES,
                                (a_full + stage).toint(),
                            )
                        g += 1
                    # counter reset: every FC2 task of the tile adds 1 after its wait; the add
                    # that completes the count zeroes it
                    if lane == 0:
                        old = _atom_add_relaxed(ready_addr, 1)
                        if old + 1 == self.ready_target + self.nblk2:
                            _st_relaxed_u32(ready_addr, 0)
            e += 1
            u, g = self._q_read(st, e)

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def _mma(
        self,
        st,
        tiled_mma,
        a_layout,
        b_layout,
        sfa_layout,
        sfb_layout,
        w1: Int32,
        nt: Int32,
        ns: Int32,
    ):
        S = self.num_ab_stage
        NA, NWG = self.num_acc, self.num_epi_wg
        a_full, a_empty = st.a_full.data_ptr(), st.a_empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, self.nacc_total)).layout
        )
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, _SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols + self.sfa_cols, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, _SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            ),
        )
        sSFA = st.sSFA.get_tensor(sfa_layout)
        s2t_sfa, tCsSFA, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA, self.sf_dtype)
        s2t_sfb, tCsSFB, tCtSFB_s2t = _s2t_copy_and_partition(
            self._sfb_view(st, sfb_layout), tCtSFB, self.sf_dtype
        )
        tCrA = tiled_mma.make_fragment_A(st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner))
        tCrB = tiled_mma.make_fragment_B(self._b_view(st, b_layout))
        nkp = cute.size(tCrA, mode=[2])

        g = Int32(0)
        jw0 = Int32(0)
        jw1 = Int32(0)
        f2 = Int32(0)
        first1 = Int32(1)
        e = Int32(0)
        u, _g0 = self._q_read(st, e)
        while u >= 0:
            # FC1 unit: one tile block over nkb1 k-tiles, epilogue on warpgroup 0; FC2 task: the
            # tiles of its slice, each over nkb2 k-tiles, epilogue on the task's warpgroup
            owner = Int32(0)
            ntl = Int32(1)
            nst = Int32(self.nkb1)
            fc2 = u >= w1
            if fc2:
                owner = f2 % NWG
                _c, _s, t0, t1 = self._fc2_task(u - w1, nt, ns)
                ntl = t1 - t0
                nst = Int32(self.nkb2)
                f2 += 1
            elif first1 > 0:
                first1 = Int32(0)
                cute.arch.mbarrier_wait(st.tok_full, 0)
                cute.arch.mbarrier_wait(st.sfb_full, 0)
            jj = jw0
            if owner != 0:
                jj = jw1
            for _t in cutlass.range(ntl, unroll=1):
                acc_stage = owner * NA + jj % NA
                if jj >= NA:
                    cute.arch.mbarrier_wait(acc_empty + acc_stage, ((jj // NA) & 1) ^ 1)
                tAcc = tAcc_base[(None, None, None, acc_stage)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range(nst, unroll=1):
                    stage = g % S
                    cute.arch.mbarrier_wait(a_full + stage, (g // S) & 1)
                    bidx = stage
                    if u < w1:
                        bidx = S + kb
                    cute.copy(s2t_sfa, tCsSFA[(None, None, None, None, stage)], tCtSFA_s2t)
                    cute.copy(s2t_sfb, tCsSFB[(None, None, None, None, bidx)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(nkp):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tAcc[(None, 0, 0)],
                            tCrA[(None, 0, kp, stage)],
                            tCrB[(None, 0, kp, bidx)],
                            tAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(a_empty + stage)
                    g += 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + acc_stage)
                jj += 1
            if owner == 0:
                jw0 = jj
            else:
                jw1 = jj
            e += 1
            u, _g0 = self._q_read(st, e)
        cute.arch.mbarrier_wait(st.tmem_free, 0)
        cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols)

    # ------------------------------------------------------------------ epilogues
    @cute.jit
    def _token_scales_to_smem(self, st, x_sf_ptr, orig_m: Int32, tidx: Int32):
        """Token scales, linear [orig_m, k1 / 16], into the MMA's 128x4-blocked SMEM layout (rows
        past orig_m zero), every load in flight before the first store."""
        n_tok = self.n_tok
        nwords = self.nkb1 * 4 * n_tok
        per_thread = (nwords + 127) // 128
        sf_row_bytes = self.k1 // _SF_VEC
        x_sf_base = x_sf_ptr.toint()
        sfb_base = self._region_addr(st, self.off_xsf)
        vals = cute.make_rmem_tensor((per_thread,), Int32)
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * 128
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = cutlass.min(rest // 4, self.nkb1 - 1)
            rr = cutlass.min(r, orig_m - 1)
            vals[t] = _ldg_u32(x_sf_base + Int64(rr * sf_row_bytes + kb * 16 + atom * 4))
        for t in cutlass.range_constexpr(per_thread):
            i = tidx + t * 128
            r = i % n_tok
            rest = i // n_tok
            atom = rest % 4
            kb = rest // 4
            if i < nwords:
                v = vals[t]
                if r >= orig_m:
                    v = Int32(0)
                _sts_u32(
                    sfb_base + kb * _SF_KBLK_BYTES + atom * 512 + (r % 32) * 16 + (r // 32) * 4,
                    v,
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.sfb_full)

    @cute.jit
    def _build_slot_table(
        self,
        st,
        alpha2,
        tile_idx_to_expert_idx,
        expanded_idx_to_permuted_idx,
        token_final_scales,
        orig_m: Int32,
        tidx: Int32,
    ):
        """Token-indexed layout: for every expanded slot ``x = token * topk + k`` of the step, its
        tile (``expanded_idx_to_permuted_idx[x] // 128``, -1 for another rank's expert or a
        padding token) and its scale ``alpha2[expert] * token_final_scales[token, k]`` (the
        factor the permuted epilogue applies), by all epilogue threads; independent loads, one
        round."""
        nthr = 128 * self.num_epi_wg
        tl_t = st.sTl.get_tensor(cute.make_layout(self.n_x))
        sc_t = st.sSc.get_tensor(cute.make_layout(self.n_x))
        for it in cutlass.range_constexpr((self.n_x + nthr - 1) // nthr):
            x = tidx + it * nthr
            if x < self.n_x:
                t = Int32(-1)
                sc = Float32(0.0)
                if x < orig_m * self.topk:
                    p = Int32(expanded_idx_to_permuted_idx[x])
                    if p >= 0:
                        t = p // _MMA_M
                        token = x // self.topk
                        slot_k = x - token * self.topk
                        sc = alpha2[tile_idx_to_expert_idx[t]] * token_final_scales[(token, slot_k)]
                tl_t[x] = t
                sc_t[x] = sc
        cute.arch.barrier(barrier_id=_EPI_BARRIER_BASE + self.num_epi_wg, number_of_threads=nthr)

    @cute.jit
    def _published_tables(
        self, st, alpha2, ids_t, ctr_ptr, orig_m: Int32, expert_offset: Int32, tidx: Int32
    ):
        """Published routing: once the routing kernel has published every token's top-k
        (``CTR_ROUTE``), this rank's active-expert mask with its word prefix counts, the tile ->
        expert table (the active local experts in ascending order) and the tile count into shared
        memory, by the epilogue warps; every warp of the CTA waits at the closing barrier."""
        nthr = 128 * self.num_epi_wg
        bar = _EPI_BARRIER_BASE + self.num_epi_wg
        if tidx < nthr:
            ids_base = ids_t.iterator.toint()
            num_local = Int32(cute.size(alpha2))
            am_base = st.sAm.data_ptr().toint()
            am = st.sAm.get_tensor(cute.make_layout(_ROUTE_MASK_WORDS))
            apre = st.sApre.get_tensor(cute.make_layout(_ROUTE_MASK_WORDS))
            t2e = st.sT2e.get_tensor(cute.make_layout(self.route_t2e))
            ntt = st.sNt.get_tensor(cute.make_layout(4))
            if tidx < _ROUTE_MASK_WORDS:
                am[tidx] = Int32(0)
            if tidx == 0:
                # relaxed polls, one acquire once complete
                addr = ctr_ptr.toint() + Int64(CTR_ROUTE * 4)
                done = _ld_relaxed_u32(addr)
                while done < orig_m:
                    done = _ld_relaxed_u32(addr)
                done = _ld_acquire_u32(addr)
            cute.arch.barrier(barrier_id=bar, number_of_threads=nthr)
            for x in cutlass.range(tidx, orig_m * self.topk, nthr, unroll=1):
                local = _ld_relaxed_u32(ids_base + Int64(x) * 4) - expert_offset
                if local >= 0 and local < num_local:
                    _red_shared_or(am_base + (local >> 5) * 4, _bit(local & 31))
            cute.arch.barrier(barrier_id=bar, number_of_threads=nthr)
            nwords = (num_local + 31) >> 5
            if tidx < 32:
                pre = Int32(0)
                tot = Int32(0)
                for w in cutlass.range(nwords, unroll=1):
                    c = cute.arch.popc(am[w])
                    if w < tidx:
                        pre = pre + c
                    tot = tot + c
                if tidx < nwords:
                    apre[tidx] = pre
                if tidx == 0:
                    ntt[0] = tot
            cute.arch.barrier(barrier_id=bar, number_of_threads=nthr)
            for e in cutlass.range(tidx, num_local, nthr, unroll=1):
                word = am[e >> 5]
                bit = _bit(e & 31)
                if (word & bit) != 0:
                    t2e[apre[e >> 5] + cute.arch.popc(word & (bit - 1))] = e
        cute.arch.barrier()

    @cute.jit
    def _published_slots(
        self,
        st,
        alpha2,
        ids_t,
        vals_t,
        orig_m: Int32,
        expert_offset: Int32,
        tidx: Int32,
    ):
        """Published routing: the token-indexed slot table (expanded slot -> tile, ``alpha2 x
        weight``, as :meth:`_build_slot_table`) from the published top-k and the mask of
        :meth:`_published_tables`, by the epilogue warps."""
        nthr = 128 * self.num_epi_wg
        ids_base = ids_t.iterator.toint()
        vals_base = vals_t.iterator.toint()
        num_local = Int32(cute.size(alpha2))
        am = st.sAm.get_tensor(cute.make_layout(_ROUTE_MASK_WORDS))
        apre = st.sApre.get_tensor(cute.make_layout(_ROUTE_MASK_WORDS))
        tl_t = st.sTl.get_tensor(cute.make_layout(self.n_x))
        sc_t = st.sSc.get_tensor(cute.make_layout(self.n_x))
        for x in cutlass.range(tidx, self.n_x, nthr, unroll=1):
            t = Int32(-1)
            sc = Float32(0.0)
            if x < orig_m * self.topk:
                local = _ld_relaxed_u32(ids_base + Int64(x) * 4) - expert_offset
                if local >= 0 and local < num_local:
                    word = am[local >> 5]
                    t = apre[local >> 5] + cute.arch.popc(word & (_bit(local & 31) - 1))
                    sc = alpha2[local] * _ld_relaxed_f32(vals_base + Int64(x) * 4)
            tl_t[x] = t
            sc_t[x] = sc
        cute.arch.barrier(barrier_id=_EPI_BARRIER_BASE + self.num_epi_wg, number_of_threads=nthr)

    @cute.jit
    def _tile_scales(self, st, t: Int32, sbuf: Int32, et: Int32, wg: Int32):
        """Token-indexed layout: token j's scale for tile ``t`` (0 if the tile's expert does not
        serve it) into buffer ``sbuf`` of the warpgroup (thread j scans token j's top-k slots) and
        the tile's token mask (warp ballot); the caller's warpgroup barrier publishes them."""
        tl_t = st.sTl.get_tensor(cute.make_layout(self.n_x))
        sc_t = st.sSc.get_tensor(cute.make_layout(self.n_x))
        sv_t = st.sSv.get_tensor(
            cute.make_layout(
                (self.n_tok, 2, self.num_epi_wg), stride=(1, self.n_tok, 2 * self.n_tok)
            )
        )
        svm_t = st.sSvMask.get_tensor(cute.make_layout((2, self.num_epi_wg), stride=(1, 2)))
        hit = Int32(0)
        if et < self.n_tok:
            sv = Float32(0.0)
            for k in cutlass.range_constexpr(self.topk):
                x = et * self.topk + k
                if tl_t[x] == t:
                    sv = sc_t[x]
                    hit = Int32(1)
            sv_t[et, sbuf, wg] = sv
        if et < 32:
            bits = _ballot(hit)
            if et == 0:
                svm_t[sbuf, wg] = bits

    @cute.jit
    def _situ(self, up, gate, alpha_val, out):
        """SiTU of 16 (up, gate) pairs, ``beta tanh(g / beta) sigmoid(g) lbeta tanh(u / lbeta)``
        with ``tanh(v) = 2 sigmoid(2 v) - 1``, in packed f32x2."""
        beta = self.situ_beta
        linear_beta = self.situ_linear_beta
        LOG2_E = Float32(1.4426950408889634)
        neg_log2e_pair = (-LOG2_E, -LOG2_E)
        one_pair = (Float32(1.0), Float32(1.0))
        inv_2beta = Float32(2.0 / beta)
        two_beta = Float32(2.0 * beta)
        neg_beta = Float32(-beta)
        inv_2lbeta = Float32(2.0 / linear_beta)
        two_lbeta = Float32(2.0 * linear_beta)
        neg_lbeta = Float32(-linear_beta)

        def _sigmoid(p0, p1):
            neg = cute.arch.mul_packed_f32x2((p0, p1), neg_log2e_pair)
            e = (cute.math.exp2(neg[0], fastmath=True), cute.math.exp2(neg[1], fastmath=True))
            d = cute.arch.add_packed_f32x2(e, one_pair)
            return (cute.arch.rcp_approx(d[0]), cute.arch.rcp_approx(d[1]))

        alpha_pair = (Float32(alpha_val), Float32(alpha_val))
        for i in cutlass.range_constexpr(0, 16, 2):
            gg = cute.arch.mul_packed_f32x2((gate[i], gate[i + 1]), alpha_pair)
            uu = cute.arch.mul_packed_f32x2((up[i], up[i + 1]), alpha_pair)
            sigmoid_g = _sigmoid(gg[0], gg[1])
            gs = _sigmoid(*cute.arch.mul_packed_f32x2(gg, (inv_2beta, inv_2beta)))
            tanh_g = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(gs, (two_beta, two_beta)), (neg_beta, neg_beta)
            )
            us = _sigmoid(*cute.arch.mul_packed_f32x2(uu, (inv_2lbeta, inv_2lbeta)))
            tanh_u = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(us, (two_lbeta, two_lbeta)), (neg_lbeta, neg_lbeta)
            )
            situ_gate = cute.arch.mul_packed_f32x2(tanh_g, sigmoid_g)
            out_pair = cute.arch.mul_packed_f32x2(situ_gate, tanh_u)
            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

    @cute.jit
    def _epilogue(
        self,
        st,
        tiled_mma,
        acc_ptr,
        h_ptr,
        h_sf_ptr,
        out_ptr,
        alpha1,
        alpha2,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        global_sf,
        token_final_scales,
        ctr_ptr,
        ws_ptr,
        orig_m: Int32,
        w1: Int32,
        nt: Int32,
        ns: Int32,
        et: Int32,
        wg: Int32,
        push,
    ):
        """Every epilogue warp walks the CTA's units. FC1 units (warpgroup 0): the accumulator
        into a token-major staging buffer, SiTU and NVFP4 quantization of 16 outputs of one row
        per thread, stores, tile counter release. FC2 tasks (alternating warpgroups): per tile,
        FP32 ``scale * acc`` added into the warpgroup's token-major accumulator; then the task's
        BF16-rounded sums added into the output (red combine), or the touched tokens' rows as the
        task's partial, the m-tile's arrival, and on the last arrival the slice-ordered sum into
        the output (slice combine)."""
        NA, NWG = self.num_acc, self.num_epi_wg
        n_tok = self.n_tok
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, self.nacc_total)).layout
        )
        copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n_tok)), Float32)
        tAcc_epi = cute.flat_divide(tAcc_base[((None, None), 0, 0, None)], (_MMA_M, n_tok))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(et)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((_MMA_M, n_tok)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()

        # FC1 epilogue state
        sacc_base = st.sAcc.data_ptr().toint()
        col_bytes = self.sacc_row_floats * 4
        h_base = h_ptr.toint()
        h_sf_base = h_sf_ptr.toint()
        h_row_bytes = self.k2 // 2
        tile_sf_bytes = self.h_tile_sf_bytes
        r = et >> 2
        q = et & 3
        norm_const = global_sf[0]
        rcp_limit = Float32(1.0 / 6.0)
        fp32_max = Float32(3.40282346638528859812e38)
        up = cute.make_rmem_tensor((16,), Float32)
        gate = cute.make_rmem_tensor((16,), Float32)
        tCompute = cute.make_rmem_tensor((16,), Float32)
        rSF = cute.make_rmem_tensor((2,), self.sf_dtype)
        rQ = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
        rPv = cute.make_rmem_tensor((2,), Float32)
        ctr = ctr_ptr.toint()

        # FC2 epilogue state: the warpgroup's token-major FP32 accumulator ([token][channel],
        # padded rows; warpgroup 0 reuses the FC1 staging buffer, whose last use precedes it)
        tacc_base = sacc_base
        if cutlass.const_expr(NWG > 1):
            if wg == 1:
                tacc_base = self._region_addr(st, self.off_acc1)
        stok = st.sTok.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        sscale = st.sScale.get_tensor(cute.make_layout((n_tok, NWG), stride=(1, n_tok)))
        sflag = st.sFlag.get_tensor(cute.make_layout(4))
        ws = ws_ptr.toint()
        out_base = out_ptr.toint()
        out_row_bytes = self.n2 * 2
        if cutlass.const_expr(self.output_push_world > 0):
            # read once, past the grid dependency (as every read of the state must be)
            push = self._push_target(push)
        racc = cute.make_rmem_tensor((n_tok,), Float32)
        # token-indexed layout: the FC2 task's per-token sums of this thread's channel (rAcc's
        # columns), the per-tile scale buffers of the warpgroup and the buffer of the next tile
        rtok = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        sv_t = st.sSv.get_tensor(cute.make_layout((n_tok, 2, NWG), stride=(1, n_tok, 2 * n_tok)))
        svm_t = st.sSvMask.get_tensor(cute.make_layout((2, NWG), stride=(1, 2)))
        sbuf = Int32(0)

        jj = Int32(0)
        f2 = Int32(0)
        e = Int32(0)
        u, _g = self._q_read(st, e)
        while u >= 0:
            if u < w1:
                if wg == 0:
                    tile = u // self.nblk1
                    blk = u - tile * self.nblk1
                    alpha_val = Float32(0.0)
                    if cutlass.const_expr(self.published_routing):
                        alpha_val = alpha1[
                            st.sT2e.get_tensor(cute.make_layout(self.route_t2e))[tile]
                        ]
                    else:
                        alpha_val = alpha1[tile_idx_to_expert_idx[tile]]
                    tok = Int32(0)
                    n_valid = Int32(n_tok)
                    if cutlass.const_expr(self.token_indexed):
                        # row j of the tile = token j, for every token of the step
                        tok = r
                    else:
                        n_valid = cutlass.min(
                            cutlass.max(tile_idx_to_mn_limit[tile] - tile * _MMA_M, Int32(0)),
                            Int32(n_tok),
                        )
                        if r < n_valid:
                            tok = permuted_idx_to_expanded_idx[tile * _MMA_M + r] // self.topk
                    acc_stage = jj % NA
                    cute.arch.mbarrier_wait(acc_full + acc_stage, (jj // NA) & 1)
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                    for i in cutlass.range_constexpr(cute.size(rAcc)):
                        row, col = tTR_cAcc[i]
                        # W1 rows 32 q + i (gate) and 32 q + 16 + i (up) of the block go to
                        # staging rows 64 + 16 (q ^ 1) + i and 16 q + i: a warp's stores hit 32
                        # distinct banks and the reads below are two-way instead of four-way.
                        g = ((row >> 4) & 1) ^ 1
                        srow = (((row >> 5) ^ g) << 4) + (row & 15) + (g << 6)
                        _sts_f32(sacc_base + col * col_bytes + srow * 4, rAcc[i])
                    self._wg_barrier(Int32(0))
                    if r < n_valid:
                        # outputs 16 q .. 16 q + 15 of the block
                        src = sacc_base + tok * col_bytes
                        src_up = src + q * 64
                        src_gate = src + 256 + (q ^ 1) * 64
                        for vv in cutlass.range_constexpr(4):
                            g0, g1, g2, g3 = _lds_v4_f32(src_gate + vv * 16)
                            u0, u1, u2, u3 = _lds_v4_f32(src_up + vv * 16)
                            up[4 * vv] = u0
                            up[4 * vv + 1] = u1
                            up[4 * vv + 2] = u2
                            up[4 * vv + 3] = u3
                            gate[4 * vv] = g0
                            gate[4 * vv + 1] = g1
                            gate[4 * vv + 2] = g2
                            gate[4 * vv + 3] = g3
                        self._situ(up, gate, alpha_val, tCompute)
                        frg = cute.logical_divide(tCompute, cute.make_layout(_SF_VEC))
                        acc_frg = frg.load()
                        abs_ir = math.absf(acc_frg.ir_value())
                        abs_frg = type(acc_frg)(abs_ir, acc_frg.shape, acc_frg.dtype)
                        amax = abs_frg[None, 0].reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                        pv = cute.arch.mul_packed_f32x2(
                            (amax, Float32(0.0)), (rcp_limit, rcp_limit)
                        )
                        pv = cute.arch.mul_packed_f32x2((pv[0], pv[1]), (norm_const, norm_const))
                        rPv[0] = pv[0]
                        rPv[1] = pv[1]
                        rSF.store(rPv.load().to(self.sf_dtype))
                        sf_f32 = rSF.load().to(Float32)
                        acc_scale = cute.arch.mul_packed_f32x2(
                            (cute.arch.rcp_approx(sf_f32[0]), cute.arch.rcp_approx(sf_f32[1])),
                            (norm_const, norm_const),
                        )
                        scale0 = fmin(acc_scale[0], fp32_max, nan=True)
                        for ei in cutlass.range_constexpr(0, 16, 2):
                            tCompute[ei], tCompute[ei + 1] = cute.arch.mul_packed_f32x2(
                                (tCompute[ei], tCompute[ei + 1]), (scale0, scale0)
                            )
                        rQ.store(tCompute.load().to(cutlass.Float4E2M1FN))
                        rQw = cute.recast_tensor(rQ, Int32)
                        rSFb = cute.recast_tensor(rSF, cutlass.Uint8)
                        grow = tile * _MMA_M + r
                        _stg_v2_b32(
                            h_base + Int64(grow) * h_row_bytes + blk * 32 + q * 8, rQw[0], rQw[1]
                        )
                        _stg_u8(
                            h_sf_base
                            + Int64(tile) * tile_sf_bytes
                            + blk * 512
                            + (r % 32) * 16
                            + (r // 32) * 4
                            + q,
                            Int32(rSFb[0]),
                        )
                    self._wg_barrier(Int32(0))
                    # the barrier orders every thread's stores of this unit before the release
                    if et == 0:
                        _signal_add_release(ctr + Int64(CTR_TILE_BASE + tile) * 4)
                    jj += 1
            else:
                owner = f2 % NWG
                f2 += 1
                if owner == wg:
                    c, s, t0, t1 = self._fc2_task(u - w1, nt, ns)
                    mask = Int32(0)
                    if cutlass.const_expr(self.token_indexed):
                        # accumulator column j = token j in every tile: per-token FP32 sums in
                        # registers (tile order), then this thread's channel of every token row
                        # into the warpgroup's token-major staging buffer for the combine
                        for i in cutlass.range_constexpr(cute.size(rAcc)):
                            rtok[i] = Float32(0.0)
                        for t in cutlass.range(t0, t1, unroll=1):
                            self._tile_scales(st, t, sbuf, et, wg)
                            self._wg_barrier(wg)
                            acc_stage = wg * NA + jj % NA
                            cute.arch.mbarrier_wait(acc_full + acc_stage, (jj // NA) & 1)
                            cute.copy(
                                tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc
                            )
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                            for i in cutlass.range_constexpr(cute.size(rAcc)):
                                row, col = tTR_cAcc[i]
                                rtok[i] = rtok[i] + rAcc[i] * sv_t[col, sbuf, wg]
                            mask = mask | svm_t[sbuf, wg]
                            sbuf = 1 - sbuf
                            jj += 1
                        for i in cutlass.range_constexpr(cute.size(rAcc)):
                            row, col = tTR_cAcc[i]
                            _sts_f32(tacc_base + col * col_bytes + row * 4, rtok[i])
                    else:
                        # this thread's channel column of the token accumulator
                        for tk in cutlass.range_constexpr(n_tok):
                            _sts_f32(tacc_base + tk * col_bytes + et * 4, Float32(0.0))
                        for t in cutlass.range(t0, t1, unroll=1):
                            n_valid = cutlass.min(
                                cutlass.max(tile_idx_to_mn_limit[t] - t * _MMA_M, Int32(0)),
                                Int32(n_tok),
                            )
                            scale = Float32(0.0)
                            token = Int32(0)
                            if et < n_valid:
                                expanded = permuted_idx_to_expanded_idx[t * _MMA_M + et]
                                token = expanded // self.topk
                                slot_k = expanded - token * self.topk
                                scale = (
                                    alpha2[tile_idx_to_expert_idx[t]]
                                    * token_final_scales[(token, slot_k)]
                                )
                            # the previous tile's accumulation still reads the previous metadata
                            self._wg_barrier(wg)
                            if et < n_valid:
                                stok[et, wg] = token
                                sscale[et, wg] = scale
                            self._wg_barrier(wg)
                            acc_stage = wg * NA + jj % NA
                            cute.arch.mbarrier_wait(acc_full + acc_stage, (jj // NA) & 1)
                            cute.copy(
                                tiled_t2r, tTR_tAcc[(None, None, None, 0, 0, acc_stage)], rAcc
                            )
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                            for i in cutlass.range_constexpr(cute.size(rAcc)):
                                row, col = tTR_cAcc[i]
                                if col < n_valid:
                                    addr = tacc_base + stok[col, wg] * col_bytes + row * 4
                                    _sts_f32(addr, _lds_f32(addr) + rAcc[i] * sscale[col, wg])
                            for i in cutlass.range(n_valid, unroll=1):
                                mask = mask | (Int32(1) << stok[i, wg])
                            jj += 1
                    if cutlass.const_expr(self.combine_red):
                        # BF16-rounded per-token sums of the task added into the output: 16
                        # threads per touched token row, 8 channels (16 bytes) each
                        self._wg_barrier(wg)
                        ch8 = (et % 16) * 8
                        for rb in cutlass.range_constexpr(0, n_tok, 8):
                            tk = rb + et // 16
                            if ((mask >> tk) & 1) != 0:
                                _red_add_v8_bf16_from_f32_smem(
                                    out_base + Int64(tk) * out_row_bytes + (c * _MMA_M + ch8) * 2,
                                    tacc_base + tk * col_bytes + ch8 * 4,
                                )
                        # the next task's accumulator reset must not overtake these reads
                        self._wg_barrier(wg)
                        if cutlass.const_expr(self.output_push_world > 0):
                            self._red_push(
                                ctr, out_base, out_row_bytes, sflag, c, ns, orig_m, et, wg, push
                            )
                    else:
                        self._fc2_combine(
                            ws,
                            ctr,
                            out_base,
                            out_row_bytes,
                            tacc_base,
                            col_bytes,
                            sflag,
                            racc,
                            mask,
                            c,
                            s,
                            ns,
                            orig_m,
                            et,
                            wg,
                            push,
                        )
            e += 1
            u, _g = self._q_read(st, e)

    @cute.jit
    def _red_push(
        self,
        ctr,
        out_base,
        out_row_bytes,
        sflag,
        c,
        ns,
        orig_m,
        et,
        wg,
        push,
    ):
        """Red combine with the output push: the task's arrival on its m-tile (acq_rel, after the
        warpgroup barrier that follows its reds); the task that completes the m-tile's ``ns``
        arrivals reads the m-tile's BF16 output rows of every token (every task's reds precede
        its arrival) and stores them into the token's slot of the all-reduce, as
        :meth:`_push_rows` does for the slice combine (``push``: :meth:`_push_target`)."""
        if et == 0:
            old = _atom_add_acq_rel(ctr + Int64(CTR_MTILE_BASE + c) * 4, 1)
            flag = Int32(0)
            if old == ns - 1:
                flag = Int32(1)
            sflag[wg] = flag
        self._wg_barrier(wg)
        if sflag[wg] != 0:
            push_mc, push_rank, cur, nbytes = push
            # 16 threads per token row, 8 channels (16 bytes) each, as the reds
            ch8 = (et % 16) * 8
            for rb in cutlass.range_constexpr(0, self.n_tok, 8):
                tk = rb + et // 16
                if tk < orig_m:
                    w = _ldg_cg_v4(out_base + Int64(tk) * out_row_bytes + (c * _MMA_M + ch8) * 2)
                    dst = pushed_ar.push_slot(
                        push_mc,
                        push_mc,
                        cur,
                        nbytes,
                        tk,
                        push_rank,
                        c * _MMA_M + ch8,
                        self.output_push_world,
                        self.n2,
                        True,
                    )
                    pushed_ar.push_v4(dst, w[0], w[1], w[2], w[3], True)
            if et == 0:
                _st_relaxed_u32(ctr + Int64(CTR_MTILE_BASE + c) * 4, 0)
        # sflag is reused by the next task's arrival
        self._wg_barrier(wg)

    @cute.jit
    def _fc2_combine(
        self,
        ws,
        ctr,
        out_base,
        out_row_bytes,
        tacc_base,
        col_bytes,
        sflag,
        racc,
        mask,
        c,
        s,
        ns,
        orig_m,
        et,
        wg,
        push,
    ):
        """Slice combine: the task's partial (rows of the tokens it touched, each thread its
        channel) and the m-tile's arrival; the last of the m-tile's ``ns`` tasks sums the slices in
        slice order and writes the BF16 output rows of every token."""
        n_tok = self.n_tok
        pslot = c * self.max_slices + s
        part = ws + (Int64(self.part_word_base) + Int64(pslot) * (n_tok * _MMA_M)) * 4
        for tk in cutlass.range_constexpr(n_tok):
            _stg_f32_pred(
                part + (tk * _MMA_M + et) * 4,
                _lds_f32(tacc_base + tk * col_bytes + et * 4),
                (mask >> tk) & 1,
            )
        if et == 0:
            _st_relaxed_u32(ws + Int64(self.mask_word_base + pslot) * 4, mask)
        self._wg_barrier(wg)
        # arrival (release of every thread's stores, ordered by the barrier); the last of the
        # m-tile's ns tasks combines
        if et == 0:
            old = _atom_add_acq_rel(ctr + Int64(CTR_MTILE_BASE + c) * 4, 1)
            flag = Int32(0)
            if old == ns - 1:
                flag = Int32(1)
            sflag[wg] = flag
        self._wg_barrier(wg)
        if sflag[wg] != 0:
            for tk in cutlass.range_constexpr(n_tok):
                racc[tk] = Float32(0.0)
            # slices in order, _COMBINE_CHUNK at a time with every load of the chunk in flight
            # (predicated on the slice's token mask, 0.0 where absent)
            rv = cute.make_rmem_tensor((_COMBINE_CHUNK * n_tok,), Float32)
            # lane l holds slice l's token mask (ns <= max_slices <= 32)
            lane = cute.arch.lane_idx()
            my_mask = Int32(0)
            if lane < ns:
                my_mask = _ld_relaxed_u32(
                    ws + Int64(self.mask_word_base + c * self.max_slices + lane) * 4
                )
            for s0 in cutlass.range(0, ns, _COMBINE_CHUNK, unroll=1):
                for q in cutlass.range_constexpr(_COMBINE_CHUNK):
                    sq = cutlass.min(s0 + q, ns - 1)
                    ps2 = c * self.max_slices + sq
                    m2 = cute.arch.shuffle_sync(my_mask, sq)
                    if s0 + q >= ns:
                        m2 = Int32(0)
                    p2 = ws + (Int64(self.part_word_base) + Int64(ps2) * (n_tok * _MMA_M)) * 4
                    for tk in cutlass.range_constexpr(n_tok):
                        rv[q * n_tok + tk] = _ldg_cg_f32_pred(
                            p2 + (tk * _MMA_M + et) * 4, (m2 >> tk) & 1
                        )
                for q in cutlass.range_constexpr(_COMBINE_CHUNK):
                    for tk in cutlass.range_constexpr(n_tok):
                        racc[tk] = racc[tk] + rv[q * n_tok + tk]
            for tk in cutlass.range_constexpr(n_tok):
                if tk < orig_m:
                    _stg_bf16_from_f32(
                        out_base + Int64(tk) * out_row_bytes + (c * _MMA_M + et) * 2,
                        racc[tk],
                    )
            if cutlass.const_expr(self.output_push_world > 0):
                self._push_rows(racc, push, c, orig_m, et)
            if et == 0:
                _st_relaxed_u32(ctr + Int64(CTR_MTILE_BASE + c) * 4, 0)

    @cute.jit
    def _push_target(self, push):
        """``(push_mc, push_rank, cur, nbytes)`` of the output push: the multicast base, this
        rank, and the all-reduce call's current buffer and bytes per buffer
        (:func:`mnnvl_pushed_allreduce.lamport_state`, valid past the grid dependency)."""
        push_mc, push_flags, push_rank = push
        cur, _, nbytes, _, _, _, _, _ = pushed_ar.lamport_state(push_flags)
        return push_mc, push_rank, cur, nbytes

    @cute.jit
    def _push_rows(self, racc, push, c, orig_m: Int32, et: Int32):
        """Output push: lane groups of eight gather their eight channels of every token (BF16)
        and the group's first lane stores them into the token's slot of the all-reduce
        (``push``: :meth:`_push_target`)."""
        push_mc, push_rank, cur, nbytes = push
        lane = cute.arch.lane_idx()
        group = lane - (lane & 7)
        for tk in cutlass.range_constexpr(self.n_tok):
            vals = [cute.arch.shuffle_sync(racc[tk], group + i) for i in range(8)]
            if (lane & 7) == 0 and tk < orig_m:
                words = [_bf16x2_bits(vals[2 * i], vals[2 * i + 1]) for i in range(4)]
                dst = pushed_ar.push_slot(
                    push_mc,
                    push_mc,
                    cur,
                    nbytes,
                    Int32(tk),
                    push_rank,
                    c * _MMA_M + et,
                    self.output_push_world,
                    self.n2,
                    True,
                )
                pushed_ar.push_v4(dst, words[0], words[1], words[2], words[3], True)

    @cute.jit
    def _push_zero_rows(self, push, orig_m: Int32, gtid: Int32, nthreads: Int32):
        """Output push without a local expert: the vectors :meth:`_push_rows` stores for an
        all-zero output (``+0`` words), grid-strided over the token warps."""
        push_mc, push_rank, cur, nbytes = self._push_target(push)
        vecs = self.n2 // 8
        zero = Int32(0)
        for i in cutlass.range(gtid, orig_m * vecs, nthreads, unroll=1):
            tk = i // vecs
            dst = pushed_ar.push_slot(
                push_mc,
                push_mc,
                cur,
                nbytes,
                tk,
                push_rank,
                (i - tk * vecs) * 8,
                self.output_push_world,
                self.n2,
                True,
            )
            pushed_ar.push_v4(dst, zero, zero, zero, zero, True)

    # ------------------------------------------------------------------ wrapper
    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        w1_ptr: cute.Pointer,
        x_sf_ptr: cute.Pointer,
        w1_sf_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_sf_ptr: cute.Pointer,
        alpha1_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        alpha2_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        expanded_idx_to_permuted_idx_ptr: cute.Pointer,
        ctr_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        orig_m: Int32,
        m: Int32,
        num_local: Int32,
        expert_offset: Int32,
        push_mc: Int64,
        push_flags: Int64,
        push_rank: Int32,
        num_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """``x`` [orig_m, k1] tokens and ``x_sf`` their linear scales; ``h`` / ``h_sf`` the FC1
        output buffers ([m, k2] rows, 128x4-blocked scales); ``out`` the BF16 [orig_m, k1] output
        (fully written); ``ctr`` int32 [counter_words(m // 128)] zeros; ``ws``
        [workspace_words(...)] scratch of the slice combine; ``expanded_idx_to_permuted_idx``
        [orig_m * topk] (read with the token-indexed layout only; published routing: the top-k ids,
        ``token_final_scales`` the weights, ``expert_offset`` this rank's first expert). Die-major
        weights are ``[2 L, rows / 2, cols]`` (``num_local`` is L)."""
        n1, k1, n2, k2 = self.n1, self.k1, self.n2, self.k2
        L = num_local
        num_tiles = m // _MMA_M

        def kmajor(ptr, rows, cols, batch):
            return cute.make_tensor(
                ptr, layout=cute.make_ordered_layout((rows, cols, batch), order=(1, 0, 2))
            )

        def blocked_sf(ptr, rows, cols, batch):
            return cute.make_tensor(
                ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, rows // 128, 4, cols // (_SF_VEC * 4), batch), order=(2, 1, 4, 0, 3, 5)
                ),
            )

        wn1, wn2, wl = n1, n2, L
        if cutlass.const_expr(self.die_major):
            wn1, wn2, wl = n1 // 2, n2 // 2, 2 * L
        self(
            kmajor(x_ptr, orig_m, k1, 1),
            kmajor(w1_ptr, wn1, k1, wl),
            blocked_sf(w1_sf_ptr, wn1, k1, wl),
            kmajor(h_ptr, m, k2, 1),
            kmajor(w2_ptr, wn2, k2, wl),
            blocked_sf(w2_sf_ptr, wn2, k2, wl),
            x_sf_ptr,
            h_ptr,
            h_sf_ptr,
            out_ptr,
            cute.make_tensor(alpha1_ptr, layout=cute.make_layout((L,))),
            cute.make_tensor(alpha2_ptr, layout=cute.make_layout((L,))),
            cute.make_tensor(tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))),
            cute.make_tensor(tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))),
            cute.make_tensor(permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))),
            cute.make_tensor(
                expanded_idx_to_permuted_idx_ptr, layout=cute.make_layout((orig_m * self.topk,))
            ),
            cute.make_tensor(num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))),
            cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,))),
            cute.make_tensor(
                token_final_scales_ptr,
                layout=cute.make_ordered_layout((orig_m, self.topk), order=(1, 0)),
            ),
            ctr_ptr,
            ws_ptr,
            orig_m,
            expert_offset,
            push_mc,
            push_flags,
            push_rank,
            num_ctas,
            stream,
        )
