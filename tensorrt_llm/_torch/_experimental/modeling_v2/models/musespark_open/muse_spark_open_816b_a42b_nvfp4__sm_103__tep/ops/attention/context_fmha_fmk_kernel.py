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
#
# Portions of this file are derived from the CUTLASS CuTe DSL Blackwell FMHA example
# (examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py, CUTLASS v4.8.0):
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Paged FP8 context FMHA for sm_103a and sm_107 with several independent softmax streams per SMSP.

One rank's full-attention context call: ``total_q`` new query rows of ``batch`` requests packed
back to back (``cu_q``), ``num_heads`` query heads and ``num_kv_heads`` KV heads of width 64, E4M3
Q ``[total_q, heads, 64]``, E4M3 K / V in the paged pool ``[blocks, kv_heads, cache_page, 64]``
(``cache_page`` 32 or 64 tokens) addressed through a ``[batch, 2, width]`` block-index table
(``table[b, 0, p]`` is the block holding keys ``cache_page p .. cache_page p + cache_page - 1`` of
request ``b``, ``table[b, 1, p]`` the block of their values), BF16 O
``[total_q, heads, 64]``. Request ``b`` has ``kv_lens[b]`` keys; its query rows are its last
``q_len(b)`` positions, so row ``r`` sees keys ``0 .. kv_lens[b] - q_len(b) + r``.

Organisation. One CTA per SM runs ``num_inst`` query instances at once. Instance ``j`` is 128
query rows of one head with its own S (``tile_n`` columns) and O (64 columns) in TMEM and its own
softmax warpgroup; the instances of a CTA are the heads ``num_inst * hg + j`` of one KV group, so
every K / V tile in shared memory feeds all of them. With four instances, every SMSP holds four
independent softmax warps (one per instance), which keeps the MUFU pipe fed while any one of them
waits for its next S tile or for a TMEM load.

Work. The kernel executes an explicit list of work units, eight int32 each, ``(b, m, hg, t0, t1,
mode, slot, 0)``: request ``b``, query rows ``128 m .. 128 m + 127`` of the request, heads
``num_inst * hg + j``, key tiles ``[t0, t1)`` of ``tile_n`` keys. CTA ``k`` runs units
``cta_off[k] .. cta_off[k + 1] - 1`` in order (a host planner balances them and may cut a unit's
key range into several units). ``mode`` 0 writes BF16 O (and, if built with ``write_stats``, the
natural-log row max and the row sum, as the served kernel's softmax statistics); ``mode`` 1 writes
``scale_output * O / l`` in fp32 and ``(m * scale_log2, l)`` of every row to workspace slot ``slot`` for a merge
kernel. A cut never falls inside a unit's masked (diagonal) tiles, so every partial but the last
is mask-free.

Roles (warp ids): warps ``4 j .. 4 j + 3`` softmax of instance ``j`` (thread ``t`` of the
warpgroup owns TMEM lane / query row ``t``); warp ``4 num_inst`` MMA issue and TMEM allocation;
warp ``4 num_inst + 1`` TMA load; the last two warps idle (they complete the warpgroup that
lowers its register budget).

Ownership and progress (cuFlow record, per CTA; all waits are on this CTA's own barriers, every
role walks the same unit list in the same order with the same per-unit counts):

* Q ring (``3 num_inst`` slots of 128 x 64 B, ``q_full`` / ``q_empty``): the load warp fills the
  ``num_inst`` slots of a unit after ``q_empty``, the first unit's before its loop and unit
  ``u + 1``'s at the start of unit ``u``; the MMA warp reads them in every QK of the unit and
  releases them with a ``tcgen05.commit`` after the unit's last QK. Unit ``u + 1`` reuses the
  slots of unit ``u - 2``, which the MMA warp released before it took unit ``u - 1``'s K / V
  (the load warp's earlier copies), so the wait cannot block on the MMA warp's current unit.
* K / V ring (``kv_stage`` slots of ``tile_n`` x 64 B, ``kv_full`` / ``kv_empty``): per unit the
  load warp fills K(t0), then per tile t: V(t), K(t + 1) with one S stage, or K(t + 1), V(t) with
  two; each tile is ``tile_n / 32`` page boxes of 32 tokens (2 KiB) under one ``expect_tx``; a
  page past the request's last page gets a box index past the pool and is zero-filled by the TMA
  unit. The MMA warp consumes in the same order (K(t0) at the unit prologue). K(t) is released
  after the QK of every instance, V(t) after the P.V of every instance (commits).
* V rows past ``kv_lens[b]`` in a request's last tile may hold any bytes (an E4M3 NaN times P = 0
  is NaN in O): the MMA warp zeroes them in shared memory after that slot's ``kv_full`` and
  before its first P.V, then ``fence.proxy.async.shared::cta`` and a warp sync; the slot cannot be
  refilled before its ``kv_empty`` commit, which follows those P.Vs.
* TMEM: instance ``j`` owns ``s_stages * tile_n + 64`` columns from ``j * inst_cols``: S buffers
  ``0 .. s_stages - 1`` of ``tile_n`` columns, then O_j (64). Every S tile of a CTA has a running
  index n (the MMA warp and the softmax warps count the same tiles of the same units); tile n goes
  to buffer ``n % s_stages`` and its P_j (E4M3, ``tile_n / 4`` columns) is stored over the start
  of that buffer. The tcgen05 MMAs of one issuing thread execute in issue order (the in-place P of
  the CUTLASS example relies on the same). QK_j(n + s_stages) overwrites the buffer of P_j(n); it
  is issued after P.V_j(n) (one stage: right after it; two stages: in the next step, which the MMA
  warp reaches only after P.V_j(n)), so no S write overlaps a pending P read; P.V_j(n) is issued
  only after ``p_full[j]`` for P_j(n), which softmax ``j`` arrives after ``tcgen05.wait::st``.
  With two stages QK_j(n + 1) is issued before P.V_j(n) and computes while softmax ``j`` works
  on S_j(n).
* ``s_full[j, b]`` (MMA -> softmax ``j``, by ``tcgen05.commit``) / ``s_empty[j, b]`` (128
  arrivals): once per S tile of buffer b on both sides, parities from the running index. Softmax
  ``j`` arrives on ``s_empty`` once its TMEM loads of the tile completed; the MMA warp waits for it
  before the next QK into that buffer.
* ``p_full[j]`` (128 arrivals) / ``p_empty[j]`` (commit after P.V_j): ``t1 - t0`` uses per unit
  on both sides. Softmax ``j`` waits ``p_empty[j]`` at every step but a unit's first, for
  P.V_j of the previous step, before it may rescale O_j (P.V_j of this step is not issued before
  this step's ``p_full``), and once after a unit's last step (the drain) before its epilogue reads
  O_j. The next unit's first P.V_j (accumulate off) overwrites O_j only after ``p_full[j]`` of that
  unit's first P, which follows the epilogue's ``tcgen05.wait::ld``.
* TMEM (512 columns) is allocated by the MMA warp and freed by it after a named barrier that every
  softmax warp reaches after its last epilogue read; the last MMAs are P.Vs whose completion the
  softmax warps observed in their drains.
* Progress. The load warp waits only on empties released by the MMA warp; the MMA warp waits on
  ``q_full`` / ``kv_full`` (load warp), ``p_full[j]`` and ``s_empty[j, b]`` (softmax ``j``);
  softmax ``j`` waits on ``s_full[j, b]`` and ``p_empty[j]`` (MMA warp). The MMA warp's wait for
  P_j(n) is satisfiable: S_j(n) was committed before it (prologue or an earlier QK), and the
  softmax's ``p_empty`` waits are for P.Vs issued before (n - 1, or the unit's last). The MMA
  warp's ``s_empty`` wait before QK_j(n + s_stages) is for the load of S_j(n), which softmax ``j``
  performs before it arrives P_j(n). No cross-CTA wait exists (partials are merged by a separate
  launch).
* Programmatic dependent launch. Every warp executes ``griddepcontrol.wait`` before it reads any
  input; the MMA warp issues ``griddepcontrol.launch_dependents`` after its unit loop (a dependent
  grid's own wait returns only after this grid completed and its memory is visible).

Numerics: S = Q K^T in fp32 on the tensor core; online softmax in log2 units with the running
maximum raised when a tile exceeds it by CR_HEADROOM_LOG2 (``cond_rescale``; every increase
without it); P = 2^(s * scale_log2 - max * scale_log2 + 8) rounded to E4M3 (three of every four
exponentials as f16x2 MUFUs with ``f16_exp``); the row sum of P in registers (f32; the f16x2
pairs' in f16x2 per step); O accumulated in fp32 on the tensor core and multiplied by
``scale_output / row_sum``. The scales are read at run time from two device scalars,
``scale_log2`` and ``scale_output`` (the context preprocess writes them: its BMM1 scale's second
entry and its BMM2 scale), so a CUDA-graph replay sees their current values.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

F8 = cutlass.Float8E4M3FN
BF16 = cutlass.BFloat16

#: Tokens per TMA box (a cache block of 64 tokens is read as two boxes), head width, query rows per
#: instance.
PAGE = 32
#: Tokens per KV cache block the kernel reads.
CACHE_PAGES = (32, 64)
D = 64
M_TILE = 128
#: int32 fields per work unit.
UNIT_FIELDS = 8
#: Page ids one load-warp lane holds per batch (one per lane).
IDS_PER_BATCH = 32
#: P = 2^(s - max + P_PRESCALE_LOG2) uses E4M3's range (max 448) with the maximum raised on every
#: increase: floor(log2(448)).
P_PRESCALE_LOG2 = 8.0
#: Conditional rescale: log2 headroom above the prescaled P maximum 2^8 (E4M3 max 448 = 2^8.8).
CR_HEADROOM_LOG2 = 0.75
#: Exponentials of ``f16_exp`` builds, per group of four P elements, cycling through the pattern:
#: ``m`` one f16x2 pair (one half-precision MUFU) and one f32 pair, ``h`` two f16x2 pairs.
F16_EXP_PATTERN = "mh"
#: Diagnostic trace (``trace=True`` builds): int64 timestamps per work unit, then per-step cycle
#: counts of the first unit of CTA 0 (see ``PagedContextFmhaFmk``).
TRACE_FIELDS = 24
TRACE_STEPS = 64
TRACE_STEP_WORDS = 16 * TRACE_STEPS


@dsl_user_op
def _cvt_f32x4_to_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Int32:
    """Four fp32 values rounded (satfinite) to E4M3 and packed into one 32-bit word, a lowest."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
            "{\n\t.reg .b16 lo, hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}\n",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ex2_f16x2(a, b, *, loc=None, ip=None) -> Int32:
    """2^a and 2^b as one packed f16x2 word (a in the low half): the f32 pair is rounded to f16x2
    and one half-precision MUFU ``ex2`` serves both."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
            "{\n\t.reg .b32 h;\n\tcvt.rn.f16x2.f32 h, $2, $1;\n\tex2.approx.f16x2 $0, h;\n\t}\n",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _hadd2(a, b, *, loc=None, ip=None) -> Int32:
    """Packed f16x2 add."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _f16x2_sum_f32(h, *, loc=None, ip=None) -> Float32:
    """Both halves of a packed f16x2 word added in f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(h).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 lo, hi;\n\t.reg .f32 f0, f1;\n\tmov.b32 {lo, hi}, $1;\n\t"
            "cvt.f32.f16 f0, lo;\n\tcvt.f32.f16 f1, hi;\n\tadd.f32 $0, f0, f1;\n\t}\n",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_e4m3x4_f16x2(h01, h23, *, loc=None, ip=None) -> Int32:
    """Two packed f16x2 words rounded (satfinite) to four E4M3 in one word, h01's low half lowest."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip) for x in (h01, h23)],
            "{\n\t.reg .b16 lo, hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 lo, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 hi, $2;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}\n",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_e4m3x4_f16x2_f32x2(h01, c, d, *, loc=None, ip=None) -> Int32:
    """One packed f16x2 word (lowest two bytes) and two f32 values (c below d) rounded
    (satfinite) to four E4M3 in one word."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(h01).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
                Float32(d).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b16 lo, hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 lo, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $3, $2;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}\n",
            "=r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class PagedContextFmhaFmk:
    """Compiled per (heads, KV heads, instances, key tile, options); sizes and pointers are runtime
    arguments.

    :param num_heads: query heads of the rank.
    :param num_kv_heads: KV heads of the rank.
    :param num_inst: query instances per CTA (heads of one KV group run together); 2 or 4.
    :param tile_n: keys per tile (32, 64 or 128); ``num_inst * (s_stages * tile_n + 64) <= 512``
        TMEM columns.
    :param s_stages: S buffers per instance (1 or 2). With 2, QK_j(t + 1) is issued before P.V_j(t)
        and computes while softmax ``j`` works on S_j(t).
    :param cache_page: tokens per KV cache block, 32 or 64; ``tile_n`` is a whole number of blocks
        or a part of one (tile 32 on 64-token blocks reads one of the block's two boxes).
    :param kv_stage: K / V ring slots (one key tile of K or V each).
    :param write_stats: mode-0 units also write the natural-log row max and the row sum.
    :param ablate_mufu: diagnostic build, wrong values: exp2 becomes a multiply (no MUFU), to time
        everything else.
    :param cond_rescale: keep the running maximum (and so skip the rescale of O and the row sum)
        until a tile's maximum exceeds it by ``CR_HEADROOM_LOG2`` log2 units: P stays below 2^8.75
        (E4M3 maximum 448) and the final division by the row sum cancels the reference. A warp
        takes the rescale path only when one of its rows needs it.
    :param f16_exp: the exponentials of three of every four f32 pairs of P as f16x2 MUFUs (one
        half-precision ``ex2`` per pair: the argument is rounded to f16, at most 2^-8 absolute for
        arguments above -16, i.e. 0.3% on a P that is then rounded to E4M3's 3 mantissa bits). The
        row sum of those pairs is added in f16x2 (at most 16 terms of at most 448 per accumulator
        and step) and joins the f32 sum once per step.
    :param trace: diagnostic build: ``stats_ptr`` is an int64 buffer of ``num_units *
        TRACE_FIELDS + TRACE_STEP_WORDS`` words. Unit ``u`` gets ``%globaltimer`` values at
        ``u * TRACE_FIELDS + f``: f 0-3 MMA warp (unit start, Q ready, prologue QKs issued, unit
        loop done), f 4-7 load warp (unit start, Q copies issued, page ids read, last K / V copy
        issued), f 8 + j / 12 + j / 16 + j / 20 + j softmax ``j`` row 0 (unit start, unmasked
        steps done, masked steps done, epilogue done). The first unit of CTA 0 also gets
        ``%clock64`` per step ``s < TRACE_STEPS`` at ``num_units * TRACE_FIELDS + (4 g + j) *
        TRACE_STEPS + s``: g 0 softmax ``j`` S ready, g 1 softmax ``j`` P arrived, g 2 MMA saw
        P_j, g 3 MMA committed S_j.
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        *,
        num_inst: int = 4,
        tile_n: int = 64,
        kv_stage: int = 12,
        s_stages: int = 1,
        write_stats: bool = False,
        ablate_mufu: bool = False,
        trace: bool = False,
        cache_page: int = PAGE,
        cond_rescale: bool = True,
        f16_exp: bool = True,
    ):
        assert num_heads % num_kv_heads == 0
        assert (num_heads // num_kv_heads) % num_inst == 0, "instances must share one KV head"
        assert num_inst in (1, 2, 4) and tile_n in (32, 64, 128) and s_stages in (1, 2)
        assert num_inst * (s_stages * tile_n + D) <= 512, "TMEM columns"
        assert not (trace and write_stats), "the trace build uses the stats buffer"
        assert cache_page in CACHE_PAGES and tile_n % cache_page == 0, "a key tile is whole blocks"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.h_r = num_heads // num_kv_heads
        self.ni = num_inst
        self.tile_n = tile_n
        self.cache_page = cache_page
        self.cond_rescale = cond_rescale
        self.f16_exp = f16_exp
        # TMA boxes per key tile and per cache block; blocks per key tile.
        self.pages_per_tile = tile_n // PAGE
        self.boxes_per_block = cache_page // PAGE
        # Key tiles per batch of IDS_PER_BATCH block ids (a tile of 32 keys is half a 64-token block).
        self.tiles_per_batch = IDS_PER_BATCH * self.boxes_per_block // self.pages_per_tile
        self.slot_words = tile_n * D // 4
        self.row_words = D // 4
        self.kv_stage = kv_stage
        self.s_stages = s_stages
        self.q_stage = 3 * num_inst
        self.write_stats = write_stats
        self.ablate_mufu = ablate_mufu
        self.trace = trace
        self.qk_mma_tiler = (M_TILE, tile_n, D)
        self.pv_mma_tiler = (M_TILE, D, tile_n)
        self.inst_cols = s_stages * tile_n + D

        self.mma_warp_id = 4 * num_inst
        self.load_warp_id = 4 * num_inst + 1
        self.num_warps = 4 * num_inst + 4
        self.threads_per_cta = 32 * self.num_warps
        launch_regs = min(256, (65536 // self.threads_per_cta) // 8 * 8)
        self.num_regs_other = 32
        softmax_regs = (launch_regs * self.threads_per_cta - 128 * self.num_regs_other) // (
            128 * num_inst
        )
        self.num_regs_softmax = min(248, softmax_regs // 8 * 8)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * (4 * num_inst + 1)
        )
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (4 * num_inst + 1)
        )
        self.buffer_align_bytes = 1024

    # ------------------------------------------------------------------ host side
    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        pool_ptr: cute.Pointer,
        table_ptr: cute.Pointer,
        cu_q_ptr: cute.Pointer,
        kv_lens_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        units_ptr: cute.Pointer,
        cta_off_ptr: cute.Pointer,
        part_o_ptr: cute.Pointer,
        part_ml_ptr: cute.Pointer,
        stats_ptr: cute.Pointer,
        scale_qk_ptr: cute.Pointer,
        scale_o_ptr: cute.Pointer,
        total_q: Int32,
        batch: Int32,
        num_blocks: Int32,
        table_stride_b: Int32,
        table_stride_kv: Int32,
        num_units: Int32,
        num_ctas: Int32,
        stream,
    ):
        h_q, h_k = self.num_heads, self.num_kv_heads
        q = cute.make_tensor(q_ptr, cute.make_layout((total_q, D, h_q), stride=(h_q * D, 1, D)))
        o = cute.make_tensor(o_ptr, cute.make_layout((total_q, D, h_q), stride=(h_q * D, 1, D)))
        # The pool as boxes of 32 tokens of one KV head: (token 32, dim 64, box); a block of 64
        # tokens is two consecutive boxes.
        kv_boxes = cute.make_tensor(
            pool_ptr,
            cute.make_layout(
                (PAGE, D, h_k * num_blocks * self.boxes_per_block), stride=(D, 1, PAGE * D)
            ),
        )
        table = cute.make_tensor(table_ptr, cute.make_layout(batch * table_stride_b))
        cu_q = cute.make_tensor(cu_q_ptr, cute.make_layout(batch + 1))
        kv_lens = cute.make_tensor(kv_lens_ptr, cute.make_layout(batch))
        units = cute.make_tensor(units_ptr, cute.make_layout(num_units * UNIT_FIELDS))
        cta_off = cute.make_tensor(cta_off_ptr, cute.make_layout(num_ctas + 1))
        scale_qk = cute.make_tensor(scale_qk_ptr, cute.make_layout(1))
        scale_o = cute.make_tensor(scale_o_ptr, cute.make_layout(1))

        cta_group = tcgen05.CtaGroup.ONE
        K = OperandMajorMode.K
        MN = OperandMajorMode.MN
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, K, Float32, cta_group, self.qk_mma_tiler[:2]
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, MN, Float32, cta_group, self.pv_mma_tiler[:2], tcgen05.OperandSource.TMEM
        )
        # The K operand of one page (N = 32): the shape of one TMA box in shared memory.
        qk_box_mma = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, K, Float32, cta_group, (M_TILE, PAGE)
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (qk_tiled_mma.thr_id.shape,)
        )

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, F8, self.q_stage
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_mma_tiler, F8, self.kv_stage
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, F8, 1
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, F8, self.kv_stage
        )
        box_shape = cute.dice((M_TILE, PAGE, D), (None, 1, 1))
        box_shape = qk_box_mma.partition_shape_B(box_shape)
        box_atom = sm100_utils.make_smem_layout_atom(sm100_utils.SmemLayoutAtomKind.K_SW64, F8)
        box_smem_layout_staged = tcgen05.tile_to_mma_shape(
            box_atom, cute.append(box_shape, self.pages_per_tile * self.kv_stage), order=(1, 2, 3)
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, q, q_smem_layout, self.qk_mma_tiler, qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )  # fmt: skip
        box_smem_layout = cute.select(box_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kv, tma_tensor_kv = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, kv_boxes, box_smem_layout, (M_TILE, PAGE, D), qk_box_mma,
            self.cluster_layout_vmnk.shape,
        )  # fmt: skip
        self.q_bytes = cute.size_in_bytes(F8, q_smem_layout)
        self.kv_bytes = self.pages_per_tile * cute.size_in_bytes(F8, box_smem_layout)

        ni = self.ni

        @cute.struct
        class SharedStorage:
            q_full: cute.struct.MemRange[Int64, self.q_stage]
            q_empty: cute.struct.MemRange[Int64, self.q_stage]
            kv_full: cute.struct.MemRange[Int64, self.kv_stage]
            kv_empty: cute.struct.MemRange[Int64, self.kv_stage]
            s_full: cute.struct.MemRange[Int64, ni * self.s_stages]
            s_empty: cute.struct.MemRange[Int64, ni * self.s_stages]
            p_full: cute.struct.MemRange[Int64, ni]
            p_empty: cute.struct.MemRange[Int64, ni]
            tmem_holding_buf: Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            qk_box_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_kv,
            tma_tensor_kv,
            o,
            table,
            cu_q,
            kv_lens,
            units,
            cta_off,
            part_o_ptr,
            part_ml_ptr,
            stats_ptr,
            scale_qk,
            scale_o,
            table_stride_b,
            table_stride_kv,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            box_smem_layout_staged,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------ device side
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        qk_box_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_atom_kv: cute.CopyAtom,
        mKV: cute.Tensor,
        mO: cute.Tensor,
        table: cute.Tensor,
        cu_q: cute.Tensor,
        kv_lens: cute.Tensor,
        units: cute.Tensor,
        cta_off: cute.Tensor,
        part_o_ptr: cute.Pointer,
        part_ml_ptr: cute.Pointer,
        stats_ptr: cute.Pointer,
        scale_qk: cute.Tensor,
        scale_o: cute.Tensor,
        table_stride_b: Int32,
        table_stride_kv: Int32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        box_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ni = self.ni

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_kv)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_full = storage.q_full.data_ptr()
        q_empty = storage.q_empty.data_ptr()
        kv_full = storage.kv_full.data_ptr()
        kv_empty = storage.kv_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        s_empty = storage.s_empty.data_ptr()
        p_full = storage.p_full.data_ptr()
        p_empty = storage.p_empty.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_init(q_full + i, 1)
                    cute.arch.mbarrier_init(q_empty + i, 1)
                for i in cutlass.range_constexpr(self.kv_stage):
                    cute.arch.mbarrier_init(kv_full + i, 1)
                    cute.arch.mbarrier_init(kv_empty + i, 1)
                for i in cutlass.range_constexpr(ni * self.s_stages):
                    cute.arch.mbarrier_init(s_full + i, 1)
                    cute.arch.mbarrier_init(s_empty + i, 128)
                for i in cutlass.range_constexpr(ni):
                    cute.arch.mbarrier_init(p_full + i, 128)
                    cute.arch.mbarrier_init(p_empty + i, 1)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )
        pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # V and the page boxes are views of the same ring bytes (see the module docstring).
        sV = storage.sK.get_tensor(v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner)
        sBox = storage.sK.get_tensor(
            box_smem_layout_staged.outer, swizzle=box_smem_layout_staged.inner
        )
        ring_words = cute.make_tensor(
            cute.recast_ptr(storage.sK.data_ptr(), dtype=Int32),
            cute.make_layout(self.kv_stage * self.slot_words),
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        gBox = cute.flat_divide(mKV, (PAGE, D))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_kv,
            0,
            cute.make_layout(1),
            cute.group_modes(sBox, 0, 3),
            cute.group_modes(qk_box_mma.get_slice(0).partition_B(gBox), 0, 3),
        )

        # Every role reads the unit list, cu_q / kv_lens (and the load warp Q, the table and the
        # pool) from here on, so every warp waits for the grids this launch depends on.
        cute.arch.griddepcontrol_wait()
        pipeline_init_wait()

        u_begin = cta_off[bidx]
        u_end = cta_off[bidx + 1]
        trace = None
        step_base = Int32(0)
        if cutlass.const_expr(self.trace):
            step_base = cute.size(units) // UNIT_FIELDS * TRACE_FIELDS
            trace = cute.make_tensor(
                cute.recast_ptr(stats_ptr, dtype=Int64),
                cute.make_layout(step_base + TRACE_STEP_WORDS),
            )

        # //////////////////////////////////////////////////////////////// IDLE
        if warp_idx >= self.load_warp_id + 1:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # //////////////////////////////////////////////////////////////// LOAD
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            lane = cute.arch.lane_idx()
            oob_box = cute.size(mKV, mode=[2])
            tpb = self.tiles_per_batch
            q_base = Int32(0)
            q_ph = Int32(1)
            kv_idx = Int32(0)
            kv_ph = Int32(1)
            q_ring = (tma_atom_q, mQ, sQ, qk_thr_mma, units, cu_q, q_full, q_empty)
            if u_begin < u_end:
                q_base, q_ph = self.load_q(q_ring, u_begin, q_base, q_ph)
            for u in cutlass.range(u_begin, u_end, 1, unroll=1):
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 4] = cute.arch.globaltimer()
                b = units[u * UNIT_FIELDS + 0]
                hg = units[u * UNIT_FIELDS + 2]
                t0 = units[u * UNIT_FIELDS + 3]
                t1 = units[u * UNIT_FIELDS + 4]
                kv_len = kv_lens[b]
                num_pages = (kv_len + self.cache_page - 1) // self.cache_page
                kvh = (hg * ni) // self.h_r
                # Q of the next unit (this unit's was issued one unit earlier).
                if u + 1 < u_end:
                    q_base, q_ph = self.load_q(q_ring, u + 1, q_base, q_ph)
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 5] = cute.arch.globaltimer()
                # Page ids of key tiles [tpb * bj, tpb * (bj + 1)): one K id and one V id per lane;
                # the next batch is read one batch ahead of its use.
                bj = t0 // tpb
                kid, vid = self.load_page_ids(
                    table, b, bj, num_pages, lane, table_stride_b, table_stride_kv
                )
                kid_n, vid_n = self.load_page_ids(
                    table, b, bj + 1, num_pages, lane, table_stride_b, table_stride_kv
                )
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        # (the ids enter the value so the stamp follows the loads)
                        trace[u * TRACE_FIELDS + 6] = (
                            Int64(kid + kid_n) * Int64(0) + cute.arch.globaltimer()
                        )
                kv_ring = (tma_atom_kv, tBgB, tBsB, kvh, oob_box, kv_full, kv_empty)
                # K(t0), then per tile t: V(t), K(t + 1) (one S stage) or K(t + 1), V(t) (two).
                kv_idx, kv_ph = self.load_tile(kv_ring, kid, t0, kv_idx, kv_ph)
                for t in cutlass.range(t0, t1, 1, unroll=1):
                    if t // tpb != bj:
                        bj = t // tpb
                        kid = kid_n
                        vid = vid_n
                        kid_n, vid_n = self.load_page_ids(
                            table, b, bj + 1, num_pages, lane, table_stride_b, table_stride_kv
                        )
                    kid_next = cutlass.select_((t + 1) // tpb == bj, kid, kid_n)
                    if cutlass.const_expr(self.s_stages == 2):
                        if t + 1 < t1:
                            kv_idx, kv_ph = self.load_tile(kv_ring, kid_next, t + 1, kv_idx, kv_ph)
                        kv_idx, kv_ph = self.load_tile(kv_ring, vid, t, kv_idx, kv_ph)
                    else:
                        kv_idx, kv_ph = self.load_tile(kv_ring, vid, t, kv_idx, kv_ph)
                        if t + 1 < t1:
                            kv_idx, kv_ph = self.load_tile(kv_ring, kid_next, t + 1, kv_idx, kv_ph)
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 7] = cute.arch.globaltimer()

        # //////////////////////////////////////////////////////////////// MMA
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tStS, tOtO, tOrP = self.make_tmem_tensors(
                qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            lane = cute.arch.lane_idx()
            q_base = Int32(0)
            q_ph = Int32(0)
            kv_idx = Int32(0)
            kv_ph = Int32(0)
            p_ph = Int32(0)
            # S tiles of this CTA's earlier units (the softmax warps count the same).
            n0 = Int32(0)
            for u in cutlass.range(u_begin, u_end, 1, unroll=1):
                rec = (bidx == 0) & (u == u_begin)
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 0] = cute.arch.globaltimer()
                b = units[u * UNIT_FIELDS + 0]
                t0 = units[u * UNIT_FIELDS + 3]
                t1 = units[u * UNIT_FIELDS + 4]
                kv_len = kv_lens[b]
                last_tile = (kv_len - 1) // self.tile_n
                for j in cutlass.range_constexpr(ni):
                    cute.arch.mbarrier_wait(q_full + q_base + j, q_ph)
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 1] = cute.arch.globaltimer()
                # Prologue: S_j(t0) for every instance.
                cute.arch.mbarrier_wait(kv_full + kv_idx, kv_ph)
                k_slot = kv_idx
                kv_idx, kv_ph = self.ring_advance(kv_idx, kv_ph, self.kv_stage)
                for j in cutlass.range_constexpr(ni):
                    self.issue_qk(qk_tiled_mma, tStS[j], tSrQ[None, None, None, q_base + j],
                                  tSrK[None, None, None, k_slot], s_full, s_empty, j, n0)  # fmt: skip
                self.commit(kv_empty + k_slot)
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 2] = cute.arch.globaltimer()
                for t in cutlass.range(t0, t1, 1, unroll=1):
                    n = n0 + t - t0
                    has_next = t + 1 < t1
                    k_slot = kv_idx
                    v_slot = kv_idx
                    if cutlass.const_expr(self.s_stages == 2):
                        if has_next:
                            cute.arch.mbarrier_wait(kv_full + kv_idx, kv_ph)
                            kv_idx, kv_ph = self.ring_advance(kv_idx, kv_ph, self.kv_stage)
                        v_slot = kv_idx
                        cute.arch.mbarrier_wait(kv_full + kv_idx, kv_ph)
                        kv_idx, kv_ph = self.ring_advance(kv_idx, kv_ph, self.kv_stage)
                    else:
                        cute.arch.mbarrier_wait(kv_full + kv_idx, kv_ph)
                        kv_idx, kv_ph = self.ring_advance(kv_idx, kv_ph, self.kv_stage)
                        k_slot = kv_idx
                        if has_next:
                            cute.arch.mbarrier_wait(kv_full + kv_idx, kv_ph)
                            kv_idx, kv_ph = self.ring_advance(kv_idx, kv_ph, self.kv_stage)
                    if t == last_tile:
                        # Zero the V rows past the request's last key (any bytes, see the record).
                        rows_valid = kv_len - t * self.tile_n
                        if rows_valid < self.tile_n:
                            n_words = (self.tile_n - rows_valid) * self.row_words
                            base = v_slot * self.slot_words + rows_valid * self.row_words
                            for w in cutlass.range(lane, n_words, 32, unroll=1):
                                ring_words[base + w] = Int32(0)
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.sync_warp()
                    for j in cutlass.range_constexpr(ni):
                        if cutlass.const_expr(self.s_stages == 2):
                            if has_next:
                                self.issue_qk(qk_tiled_mma, tStS[j],
                                              tSrQ[None, None, None, q_base + j],
                                              tSrK[None, None, None, k_slot],
                                              s_full, s_empty, j, n + 1)  # fmt: skip
                        cute.arch.mbarrier_wait(p_full + j, p_ph)
                        if cutlass.const_expr(self.trace):
                            if rec & (lane == 0) & (t - t0 < TRACE_STEPS):
                                trace[step_base + (8 + j) * TRACE_STEPS + t - t0] = (
                                    cute.arch.clock64()
                                )
                        self.issue_pv(pv_tiled_mma, tOtO[j], tOrP[j],
                                      tOrV[None, None, None, v_slot], t > t0, n)  # fmt: skip
                        self.commit(p_empty + j)
                        if cutlass.const_expr(self.s_stages == 1):
                            if has_next:
                                self.issue_qk(qk_tiled_mma, tStS[j],
                                              tSrQ[None, None, None, q_base + j],
                                              tSrK[None, None, None, k_slot],
                                              s_full, s_empty, j, n + 1, True)  # fmt: skip
                        if cutlass.const_expr(self.trace):
                            if rec & (lane == 0) & (t - t0 < TRACE_STEPS):
                                trace[step_base + (12 + j) * TRACE_STEPS + t - t0] = (
                                    cute.arch.clock64()
                                )
                    p_ph ^= 1
                    self.commit(kv_empty + v_slot)
                    if has_next:
                        self.commit(kv_empty + k_slot)
                n0 += t1 - t0
                if cutlass.const_expr(self.trace):
                    if lane == 0:
                        trace[u * TRACE_FIELDS + 3] = cute.arch.globaltimer()
                for j in cutlass.range_constexpr(ni):
                    self.commit(q_empty + q_base + j)
                q_base += ni
                if q_base == self.q_stage:
                    q_base = Int32(0)
                    q_ph ^= 1
            cute.arch.griddepcontrol_launch_dependents()
            tmem.relinquish_alloc_permit()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # //////////////////////////////////////////////////////////////// SOFTMAX
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.softmax(
                warp_idx // 4,
                tidx % 128,
                tmem_ptr,
                qk_thr_mma,
                pv_thr_mma,
                (s_full, s_empty, p_full, p_empty),
                (units, u_begin, u_end, cu_q, kv_lens, mO, part_o_ptr, part_ml_ptr, stats_ptr),
                (scale_qk, scale_o),
                (trace, step_base, bidx),
            )
            self.tmem_dealloc_barrier.arrive()
        return

    # ------------------------------------------------------------------ helpers
    @cute.jit
    def ring_advance(self, idx, ph, stages: int):
        idx += 1
        if idx == stages:
            idx = Int32(0)
            ph ^= 1
        return idx, ph

    @cute.jit
    def commit(self, mbar_ptr):
        with cute.arch.elect_one():
            tcgen05.commit(mbar_ptr)

    @cute.jit
    def load_q(self, q_ring, u, q_base, q_ph):
        """Copy the Q tiles of unit ``u`` (one per instance) into Q slots ``q_base ..``."""
        tma_atom_q, mQ, sQ, qk_thr_mma, units, cu_q, q_full, q_empty = q_ring
        ni = self.ni
        b = units[u * UNIT_FIELDS + 0]
        m = units[u * UNIT_FIELDS + 1]
        hg = units[u * UNIT_FIELDS + 2]
        q0 = cu_q[b]
        mQ_ = cute.domain_offset((q0 + m * M_TILE, 0, 0), mQ)
        gQ = cute.flat_divide(mQ_, (M_TILE, D))
        tSgQ = qk_thr_mma.partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )
        for j in cutlass.range_constexpr(ni):
            cute.arch.mbarrier_wait(q_empty + q_base + j, q_ph)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_full + q_base + j, self.q_bytes)
            cute.copy(
                tma_atom_q,
                tQgQ[None, 0, 0, hg * ni + j],
                tQsQ[None, q_base + j],
                tma_bar_ptr=q_full + q_base + j,
            )
        q_base += ni
        if q_base == self.q_stage:
            q_base = Int32(0)
            q_ph ^= 1
        return q_base, q_ph

    @cute.jit
    def load_tile(self, kv_ring, ids, x, kv_idx, kv_ph):
        """Copy key tile ``x`` (K or V, by ``ids``) of the unit into ring slot ``kv_idx``."""
        tma_atom_kv, tBgB, tBsB, kvh, oob_box, kv_full, kv_empty = kv_ring
        cute.arch.mbarrier_wait(kv_empty + kv_idx, kv_ph)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(kv_full + kv_idx, self.kv_bytes)
        for p in cutlass.range_constexpr(self.pages_per_tile):
            g = (x % self.tiles_per_batch) * self.pages_per_tile + p  # box index in the batch
            blk = g // self.boxes_per_block
            half = g % self.boxes_per_block
            pid = cute.arch.shuffle_sync(ids, blk)
            box = cutlass.select_(
                pid >= 0, (pid * self.num_kv_heads + kvh) * self.boxes_per_block + half, oob_box
            )
            cute.copy(
                tma_atom_kv,
                tBgB[None, 0, 0, box],
                tBsB[None, kv_idx * self.pages_per_tile + p],
                tma_bar_ptr=kv_full + kv_idx,
            )
        return self.ring_advance(kv_idx, kv_ph, self.kv_stage)

    @cute.jit
    def issue_qk(self, tiled_mma, tStS_j, tSrQ, tSrK, s_full, s_empty, j: cutlass.Constexpr, n,
                 skip_empty: cutlass.Constexpr = False):  # fmt: skip
        """S tile ``n`` of instance ``j`` into its S buffer ``n % s_stages``, once softmax ``j``
        has read that buffer's previous tile; commits ``s_full``. ``skip_empty``: the caller has
        waited ``p_full`` of ``j``, which its softmax warps arrive after their ``s_empty``."""
        sb = n % self.s_stages
        idx = j * self.s_stages + sb
        if cutlass.const_expr(not skip_empty):
            cute.arch.mbarrier_wait(s_empty + idx, ((n // self.s_stages) & 1) ^ 1)
        tS = cute.make_tensor(tStS_j.iterator + sb * self.tile_n, tStS_j.layout)
        self.gemm_qk(tiled_mma, tS, tSrQ, tSrK)
        self.commit(s_full + idx)

    @cute.jit
    def issue_pv(self, tiled_mma, tOtO_j, tOrP_j, tOrV, accumulate, n):
        """O_j (+)= P_j(n) V, P_j(n) in S buffer ``n % s_stages``."""
        if cutlass.const_expr(self.s_stages == 2):
            if n % 2 == 0:
                self.gemm_pv(tiled_mma, tOtO_j, tOrP_j[0], tOrV, accumulate)
            else:
                self.gemm_pv(tiled_mma, tOtO_j, tOrP_j[1], tOrV, accumulate)
        else:
            self.gemm_pv(tiled_mma, tOtO_j, tOrP_j[0], tOrV, accumulate)

    @cute.jit
    def load_page_ids(self, table, b, bj, num_pages, lane, stride_b, stride_kv):
        """Lane ``l``: the K and V blocks of cache block ``32 bj + l`` of request ``b``, -1 past its
        last block."""
        page = bj * IDS_PER_BATCH + lane
        kid = Int32(-1)
        vid = Int32(-1)
        if page < num_pages:
            kid = table[b * stride_b + page]
            vid = table[b * stride_b + stride_kv + page]
        return kid, vid

    def make_tmem_tensors(self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr):
        """Per instance: S buffer 0 (fp32, tile_n columns; buffer 1 follows it), O (fp32, 64
        columns, after the S buffers), P per S buffer (E4M3 over its start)."""
        tStS_fake = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        )
        tOtO_fake = pv_thr_mma.make_fragment_C(
            pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        )
        tP = cute.make_tensor(tmem_ptr, p_tmem_layout_staged.outer)
        tOrP_fake = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        tStS, tOtO, tOrP = [], [], []
        for j in range(self.ni):
            base = j * self.inst_cols
            tStS.append(cute.make_tensor(tmem_ptr + base, tStS_fake.layout))
            tOtO.append(
                cute.make_tensor(tmem_ptr + base + self.s_stages * self.tile_n, tOtO_fake.layout)
            )
            tOrP.append(
                [
                    cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + base + sb * self.tile_n, dtype=tOrP_fake.dtype),
                        tOrP_fake.layout,
                    )
                    for sb in range(self.s_stages)
                ]
            )
        return tStS, tOtO, tOrP

    def fresh_qk_mma(self):
        """A new QK tiled MMA: its runtime fields (accumulate) start from the atom's value in the
        region where it is built, so no update from another region is used."""
        return sm100_utils.make_trivial_tiled_mma(
            F8, F8, OperandMajorMode.K, OperandMajorMode.K, Float32, tcgen05.CtaGroup.ONE,
            self.qk_mma_tiler[:2],
        )  # fmt: skip

    def fresh_pv_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            F8, F8, OperandMajorMode.K, OperandMajorMode.MN, Float32, tcgen05.CtaGroup.ONE,
            self.pv_mma_tiler[:2], tcgen05.OperandSource.TMEM,
        )  # fmt: skip

    @cute.jit
    def gemm_qk(self, tiled_mma, tStS, tSrQ, tSrK):
        tiled_mma = self.fresh_qk_mma()
        for k in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, k != 0)
            cute.gemm(tiled_mma, tStS, tSrQ[None, None, k], tSrK[None, None, k], tStS)

    @cute.jit
    def gemm_pv(self, tiled_mma, tOtO, tOrP, tOrV, accumulate):
        tiled_mma = self.fresh_pv_mma()
        for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate if k == 0 else True)
            cute.gemm(tiled_mma, tOtO, tOrP[None, None, k], tOrV[None, None, k], tOtO)

    @cute.jit
    def ex2(self, x):
        if cutlass.const_expr(self.ablate_mufu):
            return x * Float32(0.5)
        return cute.math.exp2(x, fastmath=True)

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, inst, row, tmem_ptr, qk_thr_mma, pv_thr_mma, bars, work, scales, tr):
        """Softmax warpgroup of instance ``inst``; thread ``row`` owns TMEM lane / query row
        ``row`` of every unit."""
        s_full, s_empty, p_full, p_empty = bars
        units, u_begin, u_end, cu_q, kv_lens, mO, part_o_ptr, part_ml_ptr, stats_ptr = work
        s_full_j = s_full + inst * self.s_stages
        s_empty_j = s_empty + inst * self.s_stages
        p_full_j = p_full + inst
        p_empty_j = p_empty + inst
        n = self.tile_n
        base_col = inst * self.inst_cols
        tStS_fake = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        )
        tStS = cute.make_tensor(tmem_ptr + base_col, tStS_fake.layout)
        tOtO_fake = pv_thr_mma.make_fragment_C(
            pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        )
        tOtO = cute.make_tensor(tmem_ptr + base_col + self.s_stages * n, tOtO_fake.layout)
        # S loads with the per-32-column max (tcgen05.ld.red.max).
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.LdRed32x32bOp(
                tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
            ),
            Float32,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_tmem_load = tiled_tmem_load.get_slice(row)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStS)
        cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tScS = qk_thr_mma.partition_C(cS)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        # P (E4M3, tile_n / 4 columns) over the start of S.
        p_cols = n // 4
        tStS_P = cute.make_tensor(
            tStS.iterator, cute.composition(tStS.layout, cute.make_layout((M_TILE, p_cols)))
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p_cols)), Float32
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(row)
        tTMEM_STOREtP = thr_tmem_store.partition_D(tStS_P)
        tScS_P = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((M_TILE, p_cols)))
        )
        tTMEM_STOREcP = thr_tmem_store.partition_S(tScS_P)
        n_chunks = cute.size(tTMEM_LOADtS, mode=[1])

        scale = scales[0][0]
        scale_out = scales[1][0]
        # Conditional rescale: the raw-score distance a tile maximum must exceed the reference by.
        thr_raw = Float32(0.0)
        if cutlass.const_expr(self.cond_rescale):
            thr_raw = Float32(CR_HEADROOM_LOG2) / scale
        trace, step_base, bidx = tr
        # S tiles of this CTA's earlier units (the MMA warp counts the same); P.V completions
        # waited so far (parity).
        n0 = Int32(0)
        pe_ph = Int32(0)
        for u in cutlass.range(u_begin, u_end, 1, unroll=1):
            rec = (bidx == 0) & (u == u_begin) & (row == 0)
            if cutlass.const_expr(self.trace):
                if row == 0:
                    trace[u * TRACE_FIELDS + 8 + inst] = cute.arch.globaltimer()
            b = units[u * UNIT_FIELDS + 0]
            m = units[u * UNIT_FIELDS + 1]
            hg = units[u * UNIT_FIELDS + 2]
            t0 = units[u * UNIT_FIELDS + 3]
            t1 = units[u * UNIT_FIELDS + 4]
            mode = units[u * UNIT_FIELDS + 5]
            slot = units[u * UNIT_FIELDS + 6]
            q0 = cu_q[b]
            q_len = cu_q[b + 1] - q0
            kv_len = kv_lens[b]
            off = kv_len - q_len
            r_req = m * M_TILE + row
            # Row r keeps keys k <= min(off + r, kv_len - 1); tiles from first_masked on need it.
            key_limit = cutlass.min(off + r_req, kv_len - 1)
            first_masked = (off + m * M_TILE + 1) // n
            t_mid = cutlass.max(t0, cutlass.min(t1, first_masked))
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            neg_c = Float32(0.0)
            for t in cutlass.range(t0, t_mid, 1, unroll=1):
                row_max, row_sum, pe_ph, neg_c = self.softmax_step(
                    False, t == t0, t, key_limit, row_max, row_sum, n0 + t - t0, pe_ph, (scale, thr_raw), neg_c,
                    (s_full_j, s_empty_j, p_full_j, p_empty_j),
                    (tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADcS, tiled_tmem_store,
                     tTMEM_STOREtP, tTMEM_STOREcP, n_chunks, tOtO, pv_thr_mma, row),
                    (trace, step_base, rec, t - t0, inst),
                )  # fmt: skip
            if cutlass.const_expr(self.trace):
                if row == 0:
                    trace[u * TRACE_FIELDS + 12 + inst] = cute.arch.globaltimer()
            for t in cutlass.range(t_mid, t1, 1, unroll=1):
                row_max, row_sum, pe_ph, neg_c = self.softmax_step(
                    True, t == t0, t, key_limit, row_max, row_sum, n0 + t - t0, pe_ph, (scale, thr_raw), neg_c,
                    (s_full_j, s_empty_j, p_full_j, p_empty_j),
                    (tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADcS, tiled_tmem_store,
                     tTMEM_STOREtP, tTMEM_STOREcP, n_chunks, tOtO, pv_thr_mma, row),
                    (trace, step_base, rec, t - t0, inst),
                )  # fmt: skip
            if cutlass.const_expr(self.trace):
                if row == 0:
                    trace[u * TRACE_FIELDS + 16 + inst] = cute.arch.globaltimer()
            # Drain: the unit's last P.V of this instance has completed.
            cute.arch.mbarrier_wait(p_empty_j, pe_ph)
            pe_ph ^= 1
            n0 += t1 - t0
            head = hg * self.ni + inst
            r_local = row
            row_ok = m * M_TILE + r_local < q_len
            inv = Float32(0.0)
            if row_sum > Float32(0.0):
                inv = Float32(1.0) / row_sum
            if mode == 0:
                o_row = mO.iterator + ((q0 + m * M_TILE + r_local) * self.num_heads + head) * D
                self.store_o(pv_thr_mma, tOtO, o_row, scale_out * inv, row_ok, row)
                if cutlass.const_expr(self.write_stats):
                    if row_ok:
                        st = cute.make_tensor(
                            stats_ptr + ((q0 + m * M_TILE + r_local) * self.num_heads + head) * 2,
                            cute.make_layout(2),
                        )
                        st[0] = row_max * scale * Float32(math.log(2.0))
                        st[1] = row_sum * Float32(2.0**-P_PRESCALE_LOG2)
            else:
                prow = (slot * self.ni + inst) * M_TILE + r_local
                self.store_partial(pv_thr_mma, tOtO, part_o_ptr + prow * D, scale_out * inv, row)
                ml = cute.make_tensor(part_ml_ptr + prow * 2, cute.make_layout(2))
                ml[0] = row_max * scale
                ml[1] = row_sum
            if cutlass.const_expr(self.trace):
                if row == 0:
                    trace[u * TRACE_FIELDS + 20 + inst] = cute.arch.globaltimer()

    @cute.jit
    def softmax_step(
        self,
        masked: cutlass.Constexpr,
        is_first,
        t,
        key_limit,
        row_max,
        row_sum,
        n_step,
        pe_ph,
        scale_thr,
        neg_c,
        bars,
        atoms,
        tr,
    ):
        s_full_j, s_empty_j, p_full_j, p_empty_j = bars
        trace, step_base, rec, step, inst = tr
        (tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADcS, tiled_tmem_store, tTMEM_STOREtP,
         tTMEM_STOREcP, n_chunks, tOtO, pv_thr_mma, row) = atoms  # fmt: skip
        n = self.tile_n
        sb = n_step % self.s_stages
        cute.arch.mbarrier_wait(s_full_j + sb, (n_step // self.s_stages) & 1)
        if cutlass.const_expr(self.trace):
            if rec & (step < TRACE_STEPS):
                trace[step_base + inst * TRACE_STEPS + step] = cute.arch.clock64()
        rS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, Float32)
        rMax = cute.make_rmem_tensor(cute.make_layout((1, n_chunks)), Float32)
        tS = cute.make_tensor(tTMEM_LOADtS.iterator + sb * n, tTMEM_LOADtS.layout)
        for c in cutlass.range_constexpr(n_chunks):
            cute.copy_atom_call(
                tiled_tmem_load, tS[None, c, 0, 0], (rS[None, c, 0, 0], rMax[None, c])
            )
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(s_empty_j + sb)
        tile_max = -Float32.inf
        if cutlass.const_expr(masked):
            # Register i of a 32x32b TMEM load holds column i of this thread's row.
            lim = key_limit - t * n
            for i in cutlass.range_constexpr(cute.size(rS)):
                rS[i] = cutlass.select_(lim < i, -Float32.inf, rS[i])
            tile_max = rS.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        else:
            tile_max = rMax.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        old_max = row_max
        scale, thr_raw = scale_thr
        factor = Float32(1.0)
        any_need = True
        if cutlass.const_expr(self.cond_rescale):
            # Conditional rescale: the reference maximum (and with it the factor 2^(scale (old -
            # new)), 8 - max * scale and the rescale of O and the row sum) changes only when a row
            # of the warp has a tile maximum more than thr_raw above it.
            need = tile_max > row_max + thr_raw
            any_need = cute.arch.vote_ballot_sync(need) != 0
            new_max = row_max
            neg = neg_c
            rs_in = row_sum
            if any_need:
                new_max = cutlass.select_(need, tile_max, row_max)
                if cutlass.const_expr(masked):
                    if new_max == -Float32.inf:
                        new_max = Float32(0.0)
                factor = cute.math.exp2(scale * (old_max - new_max), fastmath=True)
                neg = (Float32(0.0) - new_max) * scale + Float32(P_PRESCALE_LOG2)
                rs_in = row_sum * factor
            sum2 = (rs_in, Float32(0.0))
        else:
            new_max = cute.arch.fmax(tile_max, row_max)
            if new_max == -Float32.inf:
                new_max = Float32(0.0)
            neg = (Float32(0.0) - new_max) * scale + Float32(P_PRESCALE_LOG2)
            factor = cute.math.exp2(scale * (old_max - new_max), fastmath=True)
            half = factor * Float32(0.5)
            sum2 = (row_sum * half, row_sum * half)
        rP = cute.make_rmem_tensor(tTMEM_STOREcP.shape, Float32)
        rP_i32 = cute.recast_tensor(rP, Int32)
        sum_h = Float32(0.0)
        if cutlass.const_expr(self.f16_exp):
            acc_a = Int32(0)
            acc_b = Int32(0)
            for i in cutlass.range_constexpr(0, cute.size(rS), 4):
                mode = F16_EXP_PATTERN[(i // 4) % len(F16_EXP_PATTERN)]
                a0, a1 = cute.arch.fma_packed_f32x2((rS[i], rS[i + 1]), (scale, scale), (neg, neg))
                a2, a3 = cute.arch.fma_packed_f32x2(
                    (rS[i + 2], rS[i + 3]), (scale, scale), (neg, neg)
                )
                if cutlass.const_expr(mode == "h"):
                    h01 = _ex2_f16x2(a0, a1)
                    h23 = _ex2_f16x2(a2, a3)
                    acc_a = _hadd2(acc_a, h01)
                    acc_b = _hadd2(acc_b, h23)
                    rP_i32[i // 4] = _cvt_e4m3x4_f16x2(h01, h23)
                else:
                    h01 = _ex2_f16x2(a0, a1)
                    e2 = self.ex2(a2)
                    e3 = self.ex2(a3)
                    acc_a = _hadd2(acc_a, h01)
                    sum2 = cute.arch.add_packed_f32x2(sum2, (e2, e3))
                    rP_i32[i // 4] = _cvt_e4m3x4_f16x2_f32x2(h01, e2, e3)
            sum_h = _f16x2_sum_f32(_hadd2(acc_a, acc_b))
        else:
            for i in cutlass.range_constexpr(0, cute.size(rS), 4):
                a0, a1 = cute.arch.fma_packed_f32x2((rS[i], rS[i + 1]), (scale, scale), (neg, neg))
                a2, a3 = cute.arch.fma_packed_f32x2(
                    (rS[i + 2], rS[i + 3]), (scale, scale), (neg, neg)
                )
                e0 = self.ex2(a0)
                e1 = self.ex2(a1)
                e2 = self.ex2(a2)
                e3 = self.ex2(a3)
                sum2 = cute.arch.add_packed_f32x2(sum2, (e0, e1))
                sum2 = cute.arch.add_packed_f32x2(sum2, (e2, e3))
                rP_i32[i // 4] = _cvt_f32x4_to_e4m3x4(e0, e1, e2, e3)
        if not is_first:
            # P.V of the previous step has completed: O is stable until this step's P arrives.
            cute.arch.mbarrier_wait(p_empty_j, pe_ph)
            pe_ph ^= 1
            if cutlass.const_expr(self.cond_rescale):
                if any_need:
                    self.rescale_o(pv_thr_mma, tOtO, factor, row)
            else:
                if cute.arch.vote_ballot_sync(old_max != new_max) != 0:
                    self.rescale_o(pv_thr_mma, tOtO, factor, row)
        tP = cute.make_tensor(tTMEM_STOREtP.iterator + sb * n, tTMEM_STOREtP.layout)
        cute.copy(tiled_tmem_store, rP, tP)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(p_full_j)
        if cutlass.const_expr(self.trace):
            if rec & (step < TRACE_STEPS):
                trace[step_base + (4 + inst) * TRACE_STEPS + step] = cute.arch.clock64()
        return new_max, sum2[0] + sum2[1] + sum_h, pe_ph, neg

    @cute.jit
    def rescale_o(self, thr_mma, tOtO, scale, row):
        """O *= scale for this thread's row (16 columns at a time through registers)."""
        corr = 16
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr)), Float32
        )
        st_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr)), Float32
        )
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((M_TILE, corr)))
        )
        tOcO_i = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((M_TILE, corr)))
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtO_i)
        tiled_st = tcgen05.make_tmem_copy(st_atom, tOtO_i)
        thr_ld = tiled_ld.get_slice(row)
        thr_st = tiled_st.get_slice(row)
        tLDtO = thr_ld.partition_S(tOtO_i)
        tLDcO = thr_ld.partition_D(tOcO_i)
        tSTtO = thr_st.partition_D(tOtO_i)
        for i in cutlass.range_constexpr(0, D // corr):
            r = cute.make_rmem_tensor(tLDcO.shape, Float32)
            cute.copy(tiled_ld, cute.make_tensor(tLDtO.iterator + i * corr, tLDtO.layout), r)
            for k in cutlass.range_constexpr(0, cute.size(r), 2):
                r[k], r[k + 1] = cute.arch.mul_packed_f32x2((r[k], r[k + 1]), (scale, scale))
            cute.copy(tiled_st, r, cute.make_tensor(tSTtO.iterator + i * corr, tSTtO.layout))

    @cute.jit
    def store_o(self, thr_mma, tOtO, dst_ptr, scale, row_ok, row):
        """O * scale to BF16 into this thread's 64-element output row at ``dst_ptr`` (16 columns
        per TMEM load, 32-byte row pieces)."""
        corr = 16
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr)), Float32
        )
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((M_TILE, corr)))
        )
        tOcO_i = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((M_TILE, corr)))
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtO_i)
        thr_ld = tiled_ld.get_slice(row)
        tLDtO = thr_ld.partition_S(tOtO_i)
        tLDcO = thr_ld.partition_D(tOcO_i)
        for i in cutlass.range_constexpr(0, D // corr):
            r = cute.make_rmem_tensor(tLDcO.shape, Float32)
            cute.copy(tiled_ld, cute.make_tensor(tLDtO.iterator + i * corr, tLDtO.layout), r)
            for k in cutlass.range_constexpr(0, cute.size(r), 2):
                r[k], r[k + 1] = cute.arch.mul_packed_f32x2((r[k], r[k + 1]), (scale, scale))
            rb = cute.make_rmem_tensor(cute.make_layout(corr), BF16)
            rb.store(r.load().to(BF16))
            if row_ok:
                # A row starts at a multiple of 64 elements (128 B), a piece at 32 B.
                dst = cute.make_tensor((dst_ptr + i * corr).align(32), cute.make_layout(corr))
                cute.autovec_copy(rb, dst)
        # Every O read of this unit is complete before this thread's next P commit.
        cute.arch.fence_view_async_tmem_load()

    @cute.jit
    def store_partial(self, thr_mma, tOtO, dst_ptr, scale, row):
        """O * scale in fp32 to this thread's 64-float workspace row at ``dst_ptr``."""
        corr = 16
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr)), Float32
        )
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((M_TILE, corr)))
        )
        tOcO_i = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((M_TILE, corr)))
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtO_i)
        thr_ld = tiled_ld.get_slice(row)
        tLDtO = thr_ld.partition_S(tOtO_i)
        tLDcO = thr_ld.partition_D(tOcO_i)
        for i in cutlass.range_constexpr(0, D // corr):
            r = cute.make_rmem_tensor(tLDcO.shape, Float32)
            cute.copy(tiled_ld, cute.make_tensor(tLDtO.iterator + i * corr, tLDtO.layout), r)
            for k in cutlass.range_constexpr(0, cute.size(r), 2):
                r[k], r[k + 1] = cute.arch.mul_packed_f32x2((r[k], r[k + 1]), (scale, scale))
            dst = cute.make_tensor(dst_ptr + i * corr, cute.make_layout(r.shape))
            cute.autovec_copy(r, dst)
        cute.arch.fence_view_async_tmem_load()
