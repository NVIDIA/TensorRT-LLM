# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformed NVFP4 K/V resource backed by tensor memory."""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...task_compat import consumer_work_stage
from flashinfer.cute_dsl.availability import is_rubin_cute_dsl_available
from ...placeholder_helpers import _placeholder_local_array
from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import KV_KIND_V
from .helpers_common import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
)
from .smem_resources import SmemKvResource, SmemKvTileResource


# Probed once at import: the converters below are traced inside task bodies,
# where CuTe DSL 4.9 rejects calls to memoized functions.
_RUBIN_CUTE_DSL_AVAILABLE = is_rubin_cute_dsl_available()


def _has_packed_e4m3_e2m1_mul() -> bool:
    """Whether the compile target has the packed E4M3 x E2M1 multiply.

    PTX 9.4 (CUDA 13.4) adds it for SM100 and SM103; ptxas rejects it for
    SM107, which takes the f16x2 path instead.
    """
    if not cutlass.target_version(min_version="13.4"):
        return False
    if not _RUBIN_CUTE_DSL_AVAILABLE:
        return True
    arch = BaseDSL._get_dsl().get_arch_enum()
    return not (Arch.sm_107 <= arch <= Arch.sm_107f)


@cute.jit
def _mul_e2m1x4_e4m3x4(packed_fp4: Int32, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scale factors to four unpacked E2M1 values.

    Each FP4 value occupies the low nibble of one byte in the 32-bit input;
    byte i scales by byte i of ``packed_sf`` and the products round to E4M3.
    Both paths ignore the high nibble of every input byte: the packed
    ``e2m1p4`` operand defines it as padding (checked exhaustively on SM103
    for all byte and scale values), and the f16x2 fallback masks it off.
    Callers may therefore pass packed payload bytes, or the same word shifted
    right by four, without masking.
    """
    if cutlass.const_expr(_has_packed_e4m3_e2m1_mul()):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        # Keep each byte's low nibble, then OR with a four-bit shift to pack
        # adjacent values into bytes 0 and 2 without scalar extraction.
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23, unused0, unused1;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 lo, tmp, h_01, h_23, sf_h_01, sf_h_23;
                and.b32 lo, {$r0}, 0x0F0F0F0F;
                shr.u32 tmp, lo, 4;
                or.b32 tmp, tmp, lo;
                mov.b32 {fp4_01, unused0, fp4_23, unused1}, tmp;
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


def has_direct_bf16_fp4_fp8_cvt() -> bool:
    """Whether PTX converts E2M1x2 and E4M3x2 directly to BF16x2 (CUDA 13.2+)."""
    return cutlass.target_version(min_version="13.2")


@cute.jit
def e4m3x2_to_bf16x2(packed_pair: Int32) -> Int32:
    """Convert the low two E4M3 bytes of a word to BF16x2 exactly."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
            "cvt.rn.bf16x2.e4m3x2 {$w0}, lo; }",
            write_only_types=[Int32],
            read_only_args=[packed_pair],
        )
    return cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi, h0, h1, b0, b1; .reg .b32 half2; "
        "mov.b32 {lo, hi}, {$r0}; cvt.rn.f16x2.e4m3x2 half2, lo; "
        "mov.b32 {h0, h1}, half2; cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[packed_pair],
    )


@cute.jit
def broadcast_byte(word: Int32, byte_idx: Int32) -> Int32:
    """Replicate byte ``byte_idx`` (runtime, 0..3) of ``word`` into all bytes."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[word, byte_idx * Int32(0x1111)],
    )


@cute.jit
def broadcast_byte_const(word: Int32, byte_idx: Constexpr[int]) -> Int32:
    """``broadcast_byte`` with a compile-time byte, as an immediate selector."""
    imm = ("0x0000", "0x1111", "0x2222", "0x3333")[byte_idx]
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r0}}, {imm};",
        write_only_types=[Int32],
        read_only_args=[word],
    )


@cute.jit
def _mul_e2m1x4_e4m3x4_bx(
    packed_fp4: Int32, packed_sf: Int32, sf_byte_idx: Constexpr[int]
) -> Int32:
    """Apply one selected E4M3 scale byte to four unpacked E2M1 values."""
    return _mul_e2m1x4_e4m3x4(packed_fp4, broadcast_byte_const(packed_sf, sf_byte_idx))


@cute.jit
def scale_e2m1x4_unpacked_to_bf16x2x2(
    unpacked_fp4: Int32, sf_lo: Int32, sf_hi: Int32
) -> tuple[Int32, Int32]:
    """Scale four unpacked E2M1 values into two exact BF16x2 words.

    ``unpacked_fp4`` holds one E2M1 value in the low nibble of each byte (the
    ``B4X16_P64`` LDSM output). Values 0/1 use ``sf_lo`` and values 2/3 use
    ``sf_hi``. Every E2M1 x E4M3 product has at most five significant bits,
    so the BF16 multiply is exact; nothing is rounded through E4M3.
    """
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b32 x, y;
                .reg .b8 b0, b1, b2, b3;
                and.b32 x, {$r0}, 0x0F0F0F0F;
                shr.u32 y, x, 4;
                or.b32 x, x, y;
                mov.b32 {b0, b1, b2, b3}, x;
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b2;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            }""",
            write_only_types=[Int32, Int32],
            read_only_args=[unpacked_fp4, sf_lo, sf_hi],
        )
    return cute.arch.inline_ptx(
        """{
            .reg .b32 x, y, h01, h23;
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 a0, a1, a2, a3, c0, c1, c2, c3;
            and.b32 x, {$r0}, 0x0F0F0F0F;
            shr.u32 y, x, 4;
            or.b32 x, x, y;
            mov.b32 {b0, b1, b2, b3}, x;
            cvt.rn.f16x2.e2m1x2 h01, b0;
            cvt.rn.f16x2.e2m1x2 h23, b2;
            mov.b32 {a0, a1}, h01;
            mov.b32 {a2, a3}, h23;
            cvt.rn.bf16.f16 c0, a0;
            cvt.rn.bf16.f16 c1, a1;
            cvt.rn.bf16.f16 c2, a2;
            cvt.rn.bf16.f16 c3, a3;
            mov.b32 {$w0}, {c0, c1};
            mov.b32 {$w1}, {c2, c3};
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
        }""",
        write_only_types=[Int32, Int32],
        read_only_args=[unpacked_fp4, sf_lo, sf_hi],
    )


@cute.jit
def scale_e2m1x8_packed_to_bf16x2x4(packed: Int32, sf_pair: Int32):
    """Scale eight packed E2M1 values (two per byte) into four exact BF16x2."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b8 b0, b1, b2, b3;
                mov.b32 {b0, b1, b2, b3}, {$r0};
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b1;
                cvt.rn.bf16x2.e2m1x2 {$w2}, b2;
                cvt.rn.bf16x2.e2m1x2 {$w3}, b3;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r1};
                mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
                mul.rn.bf16x2 {$w3}, {$w3}, {$r1};
            }""",
            write_only_types=[Int32, Int32, Int32, Int32],
            read_only_args=[packed, sf_pair],
        )
    lo = scale_e2m1x4_unpacked_to_bf16x2x2(packed & Int32(0x0F0F0F0F), sf_pair, sf_pair)
    hi = scale_e2m1x4_unpacked_to_bf16x2x2(
        (packed >> Int32(4)) & Int32(0x0F0F0F0F), sf_pair, sf_pair
    )
    # Unpacked order is (b0lo, b1lo), (b2lo, b3lo); restore (lo, hi) per byte.
    return (
        byte_permute(lo[0], hi[0], 0x5410),
        byte_permute(lo[0], hi[0], 0x7632),
        byte_permute(lo[1], hi[1], 0x5410),
        byte_permute(lo[1], hi[1], 0x7632),
    )


@cute.jit
def scale_e2m1_token_quad_to_bf16x2x4(word: Int32, sf01: Int32, sf23: Int32):
    """Scale a byte column of four tokens into token-pair BF16x2 words.

    Byte i of ``word`` is token i, holding head-dim rows d (low nibble) and
    d+1 (high nibble). Returns (row d: tokens 0/1, row d: tokens 2/3,
    row d+1: tokens 0/1, row d+1: tokens 2/3); ``sf01``/``sf23`` are the
    per-token BF16 scale pairs. Two byte-selects pair the tokens' nibbles of
    each row so one conversion yields a token pair; products are exact.
    """
    return cute.arch.inline_ptx(
        """{
            .reg .b32 s4, s8, lo, hi;
            .reg .b8 l0, l1, l2, l3, h0, h1, h2, h3;
            shr.u32 s4, {$r0}, 4;
            shr.u32 s8, {$r0}, 8;
            lop3.b32 lo, {$r0}, s4, 0x0F0F0F0F, 0xE4;
            lop3.b32 hi, s4, s8, 0x0F0F0F0F, 0xE4;
            mov.b32 {l0, l1, l2, l3}, lo;
            mov.b32 {h0, h1, h2, h3}, hi;
            cvt.rn.bf16x2.e2m1x2 {$w0}, l0;
            cvt.rn.bf16x2.e2m1x2 {$w1}, l2;
            cvt.rn.bf16x2.e2m1x2 {$w2}, h0;
            cvt.rn.bf16x2.e2m1x2 {$w3}, h2;
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
            mul.rn.bf16x2 {$w3}, {$w3}, {$r2};
        }""",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[word, sf01, sf23],
    )


@cute.jit
def byte_permute(a: Int32, b: Int32, selector: Constexpr[int]) -> Int32:
    """Pick four bytes of the pair (``a`` = bytes 0-3, ``b`` = bytes 4-7).

    Nibble i of ``selector`` (least significant first) names the source byte
    of result byte i, as in PTX ``prmt.b32``. For example ``0x5140`` packs
    a0, b0, a1, b1: it interleaves the even/odd E4M3 products of two words.
    """
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r1}}, {selector:#x};",
        write_only_types=[Int32],
        read_only_args=[a, b],
    )


@cute.jit
def byte_permute_runtime(a: Int32, b: Int32, selector: Int32) -> Int32:
    """``byte_permute`` with a runtime ``selector``."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Int32],
        read_only_args=[a, b, selector],
    )


# The shared-memory helpers below take 32-bit shared-window byte addresses.
# Transform loops compute per-token row addresses arithmetically (fragment
# slots, token rows, swizzled segments), so raw vector loads and stores avoid
# building a CuTe tensor view per row inside the rolled loops.


@cute.jit
def ld_shared_v4_b32(addr: Int32):
    """Load four 32-bit words from a 16-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def ld_shared_v2_b32(addr: Int32):
    """Load two 32-bit words from an 8-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def ld_shared_b32(addr: Int32) -> Int32:
    """Load one 32-bit word from a 4-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


@cute.jit
def st_shared_v4_b32(addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32) -> None:
    """Store four 32-bit words to a 16-byte-aligned shared address."""
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, w0, w1, w2, w3],
    )


@cute.jit
def ldsm_m16n16_trans_b8(addr: Int32):
    """Transposing 16x16-byte LDSM: thread t gets byte columns t/4 and t/4+8
    for rows 4*(t%4) .. 4*(t%4)+3 (rows are supplied by threads 0..15)."""
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


@dataclass(kw_only=True)
class TmemTransformedKvResource(DecodeGenResourceBase):
    """Pipeline transformed NVFP4 K/V through TMEM.

    The transform warp groups dequantize a raw NVFP4 stage from
    :class:`SmemKvResource` (an unpacking-TMA stage with separate scale
    tensors, or packed interleaved rows) into E4M3 or BF16 operands, by Q's
    compute dtype. K is stored token-major and V is transposed, so both BMMs
    consume the transformed tile as operand A.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed K consumed by QK MMA.",
        ),
        (
            "v_desc_0_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 0.",
        ),
        (
            "v_desc_1_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 1.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    src_smem_kv: Constexpr[SmemKvResource | SmemKvTileResource] = None
    _alloc: Constexpr[TmemAllocation | None] = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return no data allocation; the pipeline barriers still use SMEM."""
        return []

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the staged transformed K/V TMEM ring."""
        if self._alloc is None:
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=(
                    self.cfg.tmem_transformed_kv_stage_cols
                    * self.pipeline_config.num_stages
                ),
            )
        return [self._alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize the stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize one work tile's stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _source_stage(self) -> tuple[cutlass.Array, cutlass.Array]:
        """Return the raw stage's NVFP4 payload and scale arrays.

        Separate scale tensors stage an unpacking-TMA payload plus a scale
        ring; interleaved rows keep packed payload and scales in one stage.
        """
        cfg = self.cfg
        src_stage_idx = consumer_work_stage(self.src_smem_kv)
        src_byte_base = self.src_smem_kv._smem_base_kv.subview(
            src_stage_idx * cfg.smem_kv_tile_elements
        )
        src_bytes = cutlass.Array(
            src_byte_base.data_ptr(),
            dtype=Uint8,
            shape=(cfg.smem_kv_tile_elements,),
            addrspace=3,
        )
        if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
            return src_bytes, None
        return src_bytes, self.src_smem_kv._sf_stage_base(src_stage_idx)

    @cute.jit
    def interleaved_row_offset(self, token: Int32) -> Int32:
        """Byte offset of tile token ``token``'s interleaved row in a raw stage."""
        return self.cfg.nvfp4_interleaved_row_offset(token)

    @cute.jit
    def store_k_interleaved(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_chunk: Constexpr[int],
        num_chunks: Constexpr[int],
    ) -> None:
        """Convert one D128 slice of packed interleaved K rows into TMEM.

        Each thread owns one token row, which is also its TMEM lane, so packed
        bytes convert in place without unpacking loads: byte-selected E2M1x2
        pairs scale exactly to BF16x2 for BF16 Q, or round through E4M3 for
        FP8 Q. Each row stores sixteen columns at a time.
        """
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        token = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        src_bytes, _ = self._source_stage()
        row_addr = Int32(src_bytes.data_ptr().toint()) + self.interleaved_row_offset(
            token
        )
        payload_addr = row_addr + Int32(
            head_dim_stage_idx * (cfg.head_dim_kv_stage // 2)
        )
        scale_addr = (
            row_addr
            + Int32(cfg.headdim // 2)
            + Int32(head_dim_stage_idx * (cfg.head_dim_kv_stage // 16))
        )
        sf_lo_word, sf_hi_word = ld_shared_v2_b32(scale_addr)
        stage_addr = self._destination_stage_addr(stage_info)
        cols_per_chunk = 32 * cfg.qk_dtype_bytes // 4
        # Each 16-byte chunk holds 32 values: two scale blocks, 16 BF16 or
        # 8 FP8 columns.
        # FP8 output is short enough to unroll, overlapping each chunk's load
        # and conversion latency; BF16's longer body stays rolled (I-cache).
        for chunk_offset in cutlass.range(
            num_chunks, unroll=num_chunks if cfg.qk_dtype_bytes == 1 else 1
        ):
            chunk = Int32(first_chunk) + chunk_offset
            words = ld_shared_v4_b32(payload_addr + chunk * Int32(16))
            sf_word = cutlass.select_(chunk >= Int32(2), sf_hi_word, sf_lo_word)
            block_base = (chunk & Int32(1)) * Int32(2)
            dst_regs = []
            # Words 0/1 and 2/3 each share one 16-value scale block.
            block_scales = (
                broadcast_byte(sf_word, block_base),
                broadcast_byte(sf_word, block_base + Int32(1)),
            )
            if cutlass.const_expr(cfg.qk_dtype_bytes == 2):
                block_scales = (
                    e4m3x2_to_bf16x2(block_scales[0]),
                    e4m3x2_to_bf16x2(block_scales[1]),
                )
            for word_idx in cutlass.range_constexpr(4):
                sf_scalar = block_scales[word_idx // 2]
                if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                    # Even and odd nibbles scale separately (E4M3 rounding),
                    # then interleave back to natural order.
                    even = _mul_e2m1x4_e4m3x4(words[word_idx], sf_scalar)
                    odd = _mul_e2m1x4_e4m3x4(words[word_idx] >> Int32(4), sf_scalar)
                    dst_regs.append(byte_permute(even, odd, 0x5140))
                    dst_regs.append(byte_permute(even, odd, 0x7362))
                else:
                    dst_regs.extend(
                        scale_e2m1x8_packed_to_bf16x2x4(words[word_idx], sf_scalar)
                    )
            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(stage_addr + chunk * Int32(cols_per_chunk), Int32),
                cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
            )

    @cute.jit
    def store_v_interleaved(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert and transpose one D128 slice of packed interleaved V rows.

        Each warp owns 32 head-dimension rows (16 packed bytes per token). A
        transposing LDSM gives thread t byte column t/4 (rows 2c, 2c+1) and
        t/4+8 for four tokens. Instead of shuffling bytes into identity lane
        order, every 16-lane block stores row 2l at lane l and row 2l+1 at
        lane l+8 (l < 8): the low and high nibbles of the thread's own byte.
        Rows stay inside their 16-row scale block, PV carries the order into
        O, and the correction tail restores it once per output tile.
        """
        cfg = self.cfg
        assert has_direct_bf16_fp4_fp8_cvt() or cfg.pv_dtype_bytes == 1
        task_cache = _decode_gen_task_cache(stage_info)
        thread = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp = thread >> Int32(5)
        lane = thread & Int32(31)
        src_bytes, _ = self._source_stage()
        stage_base = Int32(src_bytes.data_ptr().toint())
        column_offset = Int32(
            head_dim_stage_idx * (cfg.head_dim_kv_stage // 2)
        ) + warp * Int32(16)
        # The 16 scale bytes of 16 rows form a 16x16 byte tile. Its transposed
        # LDSM gives byte column (block) b of four tokens to threads 4b..4b+3;
        # this warp's blocks are 2*warp and 2*warp+1 of the D128 slice.
        scale_src_a = warp * Int32(8) + (lane & Int32(3))
        scale_src_b = scale_src_a + Int32(4)
        stage_addr = self._destination_stage_addr(stage_info)
        cols_per_slice = 16 * cfg.pv_dtype_bytes // 4
        for slice_offset in cutlass.range(num_slices, unroll=num_slices):
            slice_idx = Int32(first_slice) + slice_offset
            slice_token = slice_idx * Int32(16)
            ldsm_row = self.interleaved_row_offset(slice_token + (lane & Int32(15)))
            payload = ldsm_m16n16_trans_b8(stage_base + ldsm_row + column_offset)
            scale_lo, scale_hi = ldsm_m16n16_trans_b8(
                stage_base + ldsm_row + Int32(cfg.headdim // 2)
            )
            # Scales of this thread's four tokens for its two blocks.
            scale_cols = scale_lo if head_dim_stage_idx == 0 else scale_hi
            block_scales = (
                cute.arch.shuffle_sync(scale_cols, scale_src_a),
                cute.arch.shuffle_sync(scale_cols, scale_src_b),
            )
            tmem_addr = stage_addr + slice_idx * Int32(cols_per_slice)
            for block in cutlass.range_constexpr(2):
                word = payload[block]
                dst_regs = []
                if cutlass.const_expr(cfg.pv_dtype_bytes == 1):
                    # Byte i is token i, matching the per-token scale word.
                    dst_regs.append(_mul_e2m1x4_e4m3x4(word, block_scales[block]))
                    dst_regs.append(
                        _mul_e2m1x4_e4m3x4(word >> Int32(4), block_scales[block])
                    )
                else:
                    dst_regs.extend(
                        scale_e2m1_token_quad_to_bf16x2x4(
                            word,
                            e4m3x2_to_bf16x2(block_scales[block]),
                            e4m3x2_to_bf16x2(block_scales[block] >> Int32(16)),
                        )
                    )
                prims.tcgen05_st(
                    "16x256b" if cfg.pv_dtype_bytes == 2 else "16x128b",
                    prims.make_tmem_ptr(
                        tmem_addr + (Int32(block * 16) << Int32(16)), Int32
                    ),
                    cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                )

    @cute.jit
    def _destination_stage_addr(self, stage_info: StageInfo) -> Int32:
        """Return the raw TMEM address of one transformed pipeline stage."""
        task_cache = _decode_gen_task_cache(stage_info)
        return (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_info.stage_idx * Int32(self.cfg.tmem_transformed_kv_stage_cols)
        )

    @cute.jit
    def _store_k(self, stage_info: StageInfo, head_dim_stage_idx: Int32 | int) -> None:
        """Dequantize K and store token-major TMEM.

        Separate-scale stages use unpacking LDSM; interleaved rows convert
        packed bytes in place (see ``store_k_interleaved``).

        Each transform warp group covers all 128 TMEM lanes. With two groups,
        each converts half of the stage's eight 16-column slices.
        """
        if cutlass.const_expr(self.cfg.use_nvfp4_interleaved_kv):
            chunks = self.cfg.head_dim_kv_stage // 32
            if cutlass.const_expr(self.num_transform_groups() == 1):
                self.store_k_interleaved(stage_info, head_dim_stage_idx, 0, chunks)
            elif self.transform_group_idx() == Int32(0):
                self.store_k_interleaved(stage_info, head_dim_stage_idx, 0, chunks // 2)
            else:
                self.store_k_interleaved(
                    stage_info, head_dim_stage_idx, chunks // 2, chunks // 2
                )
        elif cutlass.const_expr(self.num_transform_groups() == 1):
            self.store_k_slices(stage_info, head_dim_stage_idx, 0, 8)
        else:
            if self.transform_group_idx() == Int32(0):
                self.store_k_slices(stage_info, head_dim_stage_idx, 0, 4)
            else:
                self.store_k_slices(stage_info, head_dim_stage_idx, 4, 4)
        cute.arch.fence_view_async_tmem_store()

    def num_transform_groups(self) -> int:
        """Return the number of complete warp groups in the transform task."""
        assert self.cfg.transform_kv_num_warps % 4 == 0
        return self.cfg.transform_kv_num_warps // 4

    @cute.jit
    def transform_group_idx(self) -> Int32:
        """Return this warp's warp-group rank inside the transform task."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        return (warp_idx - Int32(self.cfg.transform_kv_warp_idx)) // Int32(4)

    @cute.jit
    def ldsm_k_pair(self, src_bytes, thread_row, swizzle_mask, first) -> list:
        """Issue the four unpacking LDSMs of K slices ``first`` and ``first+1``.

        Returns eight words: (slice, row block, matrix) in row-major order.
        Bounded converter grants (``uses_x4_nvfp4_transform_loads``) issue one
        x4 LDSM per slice instead: lane L addresses row ``warp*32 + L`` and
        matrix g holds 8-row group g, which is the same (row block, matrix)
        order.
        """
        head_dim_kv_stage = self.cfg.head_dim_kv_stage
        regs = []
        for col_offset in cutlass.range_constexpr(2):
            if cutlass.const_expr(self.cfg.uses_x4_nvfp4_transform_loads):
                src_ptr = src_bytes.data_ptr() + (
                    thread_row * Int32(head_dim_kv_stage)
                    + ((Int32(col_offset * 16) + first * Int32(16)) ^ swizzle_mask)
                )
                unpacked = prims.ldmatrix(
                    src_ptr,
                    4,
                    prims.MMALayout.ROW,
                    shape=prims.LoadShape.M8N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                for group in cutlass.range_constexpr(4):
                    regs.append(Int32(unpacked[group]))
            else:
                for row_block in cutlass.range_constexpr(2):
                    local_offset = Int32(
                        row_block * 16 * head_dim_kv_stage + col_offset * 16
                    ) + first * Int32(16)
                    src_ptr = src_bytes.data_ptr() + (
                        thread_row * Int32(head_dim_kv_stage)
                        + (local_offset ^ swizzle_mask)
                    )
                    unpacked = prims.ldmatrix(
                        src_ptr,
                        2,
                        prims.MMALayout.ROW,
                        shape=prims.LoadShape.M8N16,
                        src_format=prims.LoadSrcFormat.B4X16_P64,
                    )
                    regs.append(Int32(unpacked[0]))
                    regs.append(Int32(unpacked[1]))
        return regs

    @cute.jit
    def store_k_slices(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Int32 | int,
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert K column slices ``[first_slice, first_slice + num_slices)``."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        # The cached index counts from the task's first warp group; each
        # transform group covers its own 128 TMEM lanes.
        transform_thread_idx = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_qwords = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int64,
            shape=(cfg.smem_kv_sf_tile_bytes // 8,),
            addrspace=3,
        )

        # Each lane owns four K rows. Staged heads load the complete D/16-byte
        # SF row, then select the qword matching the active 128-column slice.
        sf_rows = []
        sf_row_base = warp_idx * Int32(32) + lane_idx // Int32(4)
        for row_group in cutlass.range_constexpr(4):
            sf_row = sf_row_base + Int32(row_group * 8)
            sf_qword = src_sf_qwords[
                cfg.smem_kv_sf_row_offset(sf_row) // Int32(8)
                + Int32(head_dim_stage_idx)
            ]
            sf_rows.append((Int32(sf_qword), Int32(sf_qword >> Int64(32))))

        thread_row = warp_idx * Int32(32) + lane_idx
        swizzle_mask = (thread_row & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        # Sixteen K values form one column slice and share one scale byte.
        # FP8 packs them in four TMEM columns; BF16 packs them in eight.
        # The loop is rolled on the QToken route, where the body is replicated
        # at every schedule site, and fully unrolled elsewhere
        # (cfg.rolls_tmem_kv_transform). Each iteration issues four LDSMs first.
        cols_per_slice = 16 * cfg.qk_dtype_bytes // 4
        num_pairs = num_slices // 2
        unroll = 1 if cfg.rolls_tmem_kv_transform else num_pairs
        # Software-pipeline the loop: issue the next pair's LDSMs before
        # converting the current pair so their latency overlaps the ALU work.
        # The final iteration reloads its own (still resident) pair.
        loaded = self.ldsm_k_pair(
            src_bytes, thread_row, swizzle_mask, Int32(first_slice)
        )
        for pair in cutlass.range(num_pairs, unroll=unroll):
            first = Int32(first_slice) + pair * Int32(2)
            current = loaded
            next_first = cutlass.select_(
                pair + Int32(1) < Int32(num_pairs), first + Int32(2), first
            )
            loaded = self.ldsm_k_pair(src_bytes, thread_row, swizzle_mask, next_first)
            for col_offset in cutlass.range_constexpr(2):
                col_stage = first + Int32(col_offset)
                use_hi_word = col_stage >= Int32(4)
                sf_byte_idx = col_stage & Int32(3)
                tmem_addr = stage_addr + col_stage * Int32(cols_per_slice)
                for row_block in cutlass.range_constexpr(2):
                    reg_idx = (col_offset * 2 + row_block) * 2
                    unpacked = (current[reg_idx], current[reg_idx + 1])
                    dst_regs = []
                    for half in cutlass.range_constexpr(2):
                        sf_lo_word, sf_hi_word = sf_rows[row_block * 2 + half]
                        sf_scalar = broadcast_byte(
                            cutlass.select_(use_hi_word, sf_hi_word, sf_lo_word),
                            sf_byte_idx,
                        )
                        if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                            dst_regs.append(
                                _mul_e2m1x4_e4m3x4(unpacked[half], sf_scalar)
                            )
                        else:
                            sf_pair = e4m3x2_to_bf16x2(sf_scalar)
                            dst_regs.extend(
                                scale_e2m1x4_unpacked_to_bf16x2x2(
                                    unpacked[half], sf_pair, sf_pair
                                )
                            )
                    lane_offset = Int32(row_block * 16) << Int32(16)
                    if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                        prims.tcgen05_st(
                            "16x128b",
                            prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                            cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                        )
                    else:
                        # 16x256b: registers 0/1 hold lane t/4 and 2/3 hold
                        # lane t/4 + 8, at BF16 columns 2*(t%4) and 2*(t%4)+1.
                        prims.tcgen05_st(
                            "16x256b",
                            prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                            cutlass.Vector.from_elements(
                                (dst_regs[0], dst_regs[1], dst_regs[2], dst_regs[3]),
                                dtype=Int32,
                            ),
                        )

    @cute.jit
    def _store_v(self, stage_info: StageInfo, head_dim_stage_idx: Int32 | int) -> None:
        """Dequantize and transpose V with LDSM into TMEM operand A.

        With two transform warp groups, each converts half of the stage's
        eight 16-token slices.
        """
        if cutlass.const_expr(self.cfg.use_nvfp4_interleaved_kv):
            slices = self.cfg.tile_size_kv // 16
            if cutlass.const_expr(self.num_transform_groups() == 1):
                self.store_v_interleaved(stage_info, head_dim_stage_idx, 0, slices)
            elif self.transform_group_idx() == Int32(0):
                self.store_v_interleaved(stage_info, head_dim_stage_idx, 0, slices // 2)
            else:
                self.store_v_interleaved(
                    stage_info, head_dim_stage_idx, slices // 2, slices // 2
                )
        elif cutlass.const_expr(self.num_transform_groups() == 1):
            self.store_v_slices(stage_info, head_dim_stage_idx, 0, 8)
        else:
            if self.transform_group_idx() == Int32(0):
                self.store_v_slices(stage_info, head_dim_stage_idx, 0, 4)
            else:
                self.store_v_slices(stage_info, head_dim_stage_idx, 4, 4)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_v_slices(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Int32 | int,
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert V token slices ``[first_slice, first_slice + num_slices)``."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        # The cached index counts from the task's first warp group; each
        # transform group covers its own 128 TMEM lanes.
        transform_thread_idx = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_words = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_kv_sf_tile_bytes // 4,),
            addrspace=3,
        )

        swizzle_mask = (lane_idx & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        sf_head_stage_offset = head_dim_stage_idx * (cfg.head_dim_kv_stage // 16)
        # Sixteen tokens form one slice. Each unpacked word holds four
        # consecutive tokens of one head-dimension row, scaled per token. Each
        # two-slice iteration issues its LDSMs and scale loads first; the loop
        # unrolls like store_k_slices.
        cols_per_slice = 16 * cfg.pv_dtype_bytes // 4
        # Scale rows of a 16-token slice start at a fixed stride because page
        # fragments (four tokens or more) divide the slice. Hoist the address
        # math: a rolled loop must not touch the Python config object.
        assert 16 % cfg.num_tokens_per_page == 0 or cfg.num_tokens_per_page % 16 == 0
        sf_slice_stride = cfg.smem_kv_sf_row_offset(16)
        lane_sf_byte_offset = (
            cfg.smem_kv_sf_row_offset((lane_idx % Int32(4)) * Int32(4))
            + Int32(sf_head_stage_offset * 4)
            + warp_idx * Int32(8)
        )
        head_dim_kv_stage = cfg.head_dim_kv_stage
        swizzles_v_scales = cfg.swizzles_v_scale_factors
        x4_loads = cfg.uses_x4_nvfp4_transform_loads
        num_pairs = num_slices // 2
        unroll = 1 if cfg.rolls_tmem_kv_transform else num_pairs
        for pair in cutlass.range(num_pairs, unroll=unroll):
            first = Int32(first_slice) + pair * Int32(2)
            unpacked_tiles = []
            sf_words = []
            for slice_offset in cutlass.range_constexpr(2):
                token_stage = first + Int32(slice_offset)
                # V scale factors use TRT-LLM's 4-token interleaved layout, so
                # the two packed scale vectors are adjacent 32-bit words.
                sf_byte_offset = (
                    token_stage * Int32(sf_slice_stride) + lane_sf_byte_offset
                )
                if cutlass.const_expr(swizzles_v_scales):
                    # 128 B swizzle: XOR the 16 B chunk with the row index mod
                    # 8. Both words share one chunk, so they stay adjacent.
                    sf_byte_offset = sf_byte_offset ^ (
                        ((sf_byte_offset >> Int32(7)) & Int32(7)) << Int32(4)
                    )
                sf_word_idx = sf_byte_offset // Int32(4)
                sf_words.append(
                    (src_sf_words[sf_word_idx], src_sf_words[sf_word_idx + Int32(1)])
                )
                if cutlass.const_expr(x4_loads):
                    # One transposing x4 LDSM covers this warp's two 16-column
                    # blocks (lanes 16-31 address the second); matrices 0-1
                    # belong to the first block and 2-3 to the second.
                    local_offset = (
                        token_stage * Int32(16 * head_dim_kv_stage)
                        + warp_idx * Int32(32)
                        + (lane_idx >> Int32(4)) * Int32(16)
                    )
                    src_ptr = src_bytes.data_ptr() + (
                        (lane_idx & Int32(15)) * Int32(head_dim_kv_stage)
                        + (local_offset ^ swizzle_mask)
                    )
                    unpacked = prims.ldmatrix(
                        src_ptr,
                        4,
                        prims.MMALayout.COL,
                        shape=prims.LoadShape.M16N16,
                        src_format=prims.LoadSrcFormat.B4X16_P64,
                    )
                    unpacked_tiles.append((unpacked[0], unpacked[1]))
                    unpacked_tiles.append((unpacked[2], unpacked[3]))
                else:
                    for col_block in cutlass.range_constexpr(2):
                        local_offset = (
                            token_stage * Int32(16 * head_dim_kv_stage)
                            + Int32(col_block * 16)
                            + warp_idx * Int32(32)
                        )
                        src_ptr = src_bytes.data_ptr() + (
                            lane_idx * Int32(head_dim_kv_stage)
                            + (local_offset ^ swizzle_mask)
                        )
                        unpacked_tiles.append(
                            prims.ldmatrix(
                                src_ptr,
                                2,
                                prims.MMALayout.COL,
                                shape=prims.LoadShape.M16N16,
                                src_format=prims.LoadSrcFormat.B4X16_P64,
                            )
                        )
            for slice_offset in cutlass.range_constexpr(2):
                tmem_addr = stage_addr + (first + Int32(slice_offset)) * Int32(
                    cols_per_slice
                )
                self.store_v_slice_blocks(
                    tmem_addr,
                    unpacked_tiles[slice_offset * 2 : slice_offset * 2 + 2],
                    sf_words[slice_offset],
                )

    @cute.jit
    def store_v_slice_blocks(self, tmem_addr, unpacked_tiles, sf_words) -> None:
        """Scale and store both 16-row head-dimension blocks of one V slice.

        Called inside a rolled runtime loop: do not bind Python-only values
        such as the config to local names here.
        """
        for col_block in cutlass.range_constexpr(2):
            unpacked = unpacked_tiles[col_block]
            packed_sf = sf_words[col_block]
            dst_regs = []
            if cutlass.const_expr(self.cfg.pv_dtype_bytes == 1):
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[0], packed_sf))
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[1], packed_sf))
            else:
                sf_lo = e4m3x2_to_bf16x2(packed_sf)
                sf_hi = e4m3x2_to_bf16x2(packed_sf >> Int32(16))
                for half in cutlass.range_constexpr(2):
                    dst_regs.extend(
                        scale_e2m1x4_unpacked_to_bf16x2x2(unpacked[half], sf_lo, sf_hi)
                    )
            lane_offset = Int32(col_block * 16) << Int32(16)
            if cutlass.const_expr(self.cfg.pv_dtype_bytes == 1):
                prims.tcgen05_st(
                    "16x128b",
                    prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                    cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                )
            else:
                prims.tcgen05_st(
                    "16x256b",
                    prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                    cutlass.Vector.from_elements(
                        (dst_regs[0], dst_regs[1], dst_regs[2], dst_regs[3]),
                        dtype=Int32,
                    ),
                )

    @producer_work
    @cute.jit
    def transform_kv_loop_stage(self, stage_info: StageInfo) -> None:
        """Transform one raw stage of a steady-state K/V loop step.

        Each one-instance loop step converts the next K tile and then the
        current V tile, ``num_head_dim_stages_kv`` raw stages each, so loop
        iteration ``n`` holds V when bit ``log2(stages)`` of ``n`` is set and
        head-dimension stage ``n % stages``. Selecting both at run time keeps
        one K and one V body in the hot loop.
        """
        stages = self.cfg.num_head_dim_stages_kv
        assert stages & (stages - 1) == 0, "stage decoding uses bit masks"
        item = Int32(stage_info.loop_offset) - Int32(stage_info.loop_start)
        head_dim_stage_idx = item & Int32(stages - 1)
        if (item & Int32(stages)) == Int32(0):
            self._store_k(stage_info, head_dim_stage_idx)
        else:
            self._store_v(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_k0_loop_stage(self, stage_info: StageInfo) -> None:
        """Transform head-dimension stage ``n % stages`` of a dual-ring K tile.

        The dual K converter's steady state runs once per raw stage; loop
        iteration ``n`` selects the stage at run time so one K body serves
        every stage.
        """
        stages = self.cfg.num_head_dim_stages_kv
        assert stages & (stages - 1) == 0, "stage decoding uses bit masks"
        item = Int32(stage_info.loop_offset) - Int32(stage_info.loop_start)
        self._store_k(stage_info, item & Int32(stages - 1))

    @producer_work
    @cute.jit
    def transform_v0_loop_stage(self, stage_info: StageInfo) -> None:
        """Transform head-dimension stage ``n % stages`` of a dual-ring V tile.

        The V counterpart of :meth:`transform_k0_loop_stage`.
        """
        stages = self.cfg.num_head_dim_stages_kv
        assert stages & (stages - 1) == 0, "stage decoding uses bit masks"
        item = Int32(stage_info.loop_offset) - Int32(stage_info.loop_start)
        self._store_v(stage_info, item & Int32(stages - 1))

    @producer_work
    @cute.jit
    def transform_k0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 0 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_k1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 1 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 0 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 1 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 1."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_0_slot)
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_1_slot)
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 1."""
        return self._destination_stage_addr(stage_info)


@dataclass(kw_only=True)
class SmemNvfp4RawKvResource(SmemKvResource):
    """Raw NVFP4 ring of the Keeps profiles fed by the NVFP4 transform.

    Each stage holds one KV128 tile of whole D256 rows: interleaved rows
    (payload, then scales) in page-fragment slots, or 128-byte SW128 payload
    rows whose scale rows sit in the ring's scale stage
    (``FmhaDecodeConfig.uses_sw128_nvfp4_raw_payload``). ``payload_row``,
    ``payload_chunk`` and ``scale_words`` address either layout.

    TransformKvTask converts each stage into the FP8 or BF16 K/V rings. With
    register capture (the BF16 single-raw-stage profile), it first copies its
    share of the tile into registers (``capture_tile``) and releases the
    stage before waiting for the BF16 destination, so the loader refills it
    while the transform writes. Each thread then holds its (token,
    64-channel slab) units of the tile: eight payload words and one word of
    four block scales each.
    """

    # FmhaDecodeConfig.uses_nvfp4_raw_register_capture of the profile; this
    # ring's own cfg is the raw NVFP4 view, which cannot tell.
    uses_register_capture: Constexpr[bool] = False
    captured_words: cutlass.Array = None
    captured_scales: cutlass.Array = None

    @property
    def captured_units(self) -> int:
        """(token, 64-channel slab) units per thread across the whole tile."""
        cfg = self.cfg
        return (cfg.headdim // 64 * cfg.tile_size_kv) // (
            cfg.transform_kv_num_warps * 32
        )

    def _init_placeholder_state(self) -> None:
        """Add the capture registers to the raw ring's placeholder state."""
        super()._init_placeholder_state()
        if self.uses_register_capture:
            self.captured_words = _placeholder_local_array(
                Int32, 8 * self.captured_units
            )
            self.captured_scales = _placeholder_local_array(Int32, self.captured_units)

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the raw ring, then allocate the capture registers."""
        task_locals = SmemKvResource._create_initial_task_locals(self, context)
        if cutlass.const_expr(self.uses_register_capture):
            self.captured_words = cutlass.Array(
                Int32, 8 * self.captured_units, space=cutlass.AddressSpace.rmem
            )
            self.captured_scales = cutlass.Array(
                Int32, self.captured_units, space=cutlass.AddressSpace.rmem
            )
        return task_locals

    @cute.jit
    def payload_row(self, stage_idx: Int32, token: Int32) -> Int32:
        """Shared address of tile token ``token``'s row in raw stage ``stage_idx``."""
        cfg = self.cfg
        stage_base = Int32(self._smem_base_kv.data_ptr().toint()) + (
            stage_idx * Int32(cfg.smem_kv_tile_elements)
        )
        if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
            return stage_base + cfg.nvfp4_interleaved_row_offset(token)
        return stage_base + token * Int32(cfg.headdim // 2)

    @cute.jit
    def payload_chunk(self, row: Int32, token: Int32, byte_offset: Int32) -> Int32:
        """Address of the 16-byte payload chunk at ``byte_offset`` of a row.

        ``row`` is ``payload_row`` of ``token``; ``byte_offset`` is a multiple
        of 16 (32 E2M1 channels). A SW128 row stores its chunk ``c`` at chunk
        ``c ^ (token % 8)``.
        """
        if cutlass.const_expr(self.cfg.uses_sw128_nvfp4_raw_payload):
            return row + (((byte_offset >> Int32(4)) ^ (token & Int32(7))) << Int32(4))
        return row + byte_offset

    @cute.jit
    def scale_words(
        self,
        stage_idx: Int32,
        row: Int32,
        token: Int32,
        first_block: Int32,
        num_words: Constexpr[int],
        kv_kind: Constexpr[int],
    ):
        """Load ``num_words`` (1 or 2) words of ``token``'s E4M3 block scales.

        Word ``j`` packs the scales of 16-channel blocks ``first_block + 4j``
        to ``first_block + 4j + 3``, lowest byte first; ``first_block`` is a
        multiple of 4 and ``row`` is ``payload_row`` of ``token``. Interleaved
        rows hold the scales after the payload. The scale stage holds K rows
        token-major and V rows in TRT-LLM's 4-token interleaved order (byte
        ``4 * block + token % 4`` of each 4-token group, see
        ``SmemTransformedKvResource._load_nvfp4_scale_byte``); one 16-byte
        load and three byte permutes then gather a V word.
        """
        assert num_words in (1, 2)
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
            addr = row + Int32(cfg.headdim // 2) + first_block
        else:
            sf_stage = Int32(self._smem_base_kv_sf.data_ptr().toint()) + (
                stage_idx * Int32(cfg.smem_kv_sf_tile_bytes)
            )
            addr = sf_stage + cfg.smem_kv_sf_row_offset(token) + first_block
            if cutlass.const_expr(kv_kind == KV_KIND_V):
                group = sf_stage + cfg.smem_kv_sf_row_offset(token & Int32(-4))
                # prmt selector picking byte token % 4 of each source word.
                token_byte = token & Int32(3)
                selector = token_byte * Int32(0x11) + Int32(0x40)
                words = []
                for word_idx in cutlass.range_constexpr(num_words):
                    blocks = ld_shared_v4_b32(
                        group + (first_block + Int32(4 * word_idx)) * Int32(4)
                    )
                    lo = byte_permute_runtime(blocks[0], blocks[1], selector)
                    hi = byte_permute_runtime(blocks[2], blocks[3], selector)
                    words.append(byte_permute(lo, hi, 0x5410))
                return tuple(words)
        if cutlass.const_expr(num_words == 1):
            return (ld_shared_b32(addr),)
        return ld_shared_v2_b32(addr)

    @cute.jit
    def bf16_unit(self, rep: Constexpr[int]):
        """``(token, slab, half)`` of this thread's BF16 transform unit ``rep``.

        A unit is one (token, 64-channel slab) row piece: 32 payload bytes and
        one word of four block scales. Each thread converts the
        ``captured_units`` units of one token. With separate scale tensors
        (SW128 payload rows), lane pairs share a token and ``half`` = lane % 2
        takes slabs ``2 * half + rep``; each eight-lane quarter warp then
        covers one four-token scale fragment, and walking the two 16-byte
        payload chunks in opposite orders per half (chunk ``c ^ half``) keeps
        the payload loads, scale loads and SW128 stores free of bank
        conflicts. Interleaved 144-byte rows keep 32 consecutive tokens per
        warp and slab, whose row stride already spreads their payload loads.
        """
        cfg = self.cfg
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.transform_kv_warp_idx
        )
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(cfg.uses_sw128_nvfp4_raw_payload):
            assert cfg.transform_kv_num_warps * 16 == cfg.tile_size_kv
            half = lane & Int32(1)
            token = warp * Int32(16) + (lane >> Int32(1))
            return token, half * Int32(2) + Int32(rep), half
        unit = warp * Int32(32) + lane + Int32(rep * cfg.transform_kv_num_warps * 32)
        return unit % Int32(cfg.tile_size_kv), unit // Int32(cfg.tile_size_kv), Int32(0)

    @cute.jit
    def bf16_unit_scale_words(
        self, stage_idx: Int32, token: Int32, half: Int32, kv_kind: Constexpr[int]
    ):
        """Block-scale words of this thread's units, one per ``bf16_unit`` rep.

        Separate K scales are token-major, so a token's two words (slabs
        ``2 * half`` and ``2 * half + 1``) are one 8-byte load.
        """
        cfg = self.cfg
        row = self.payload_row(stage_idx, token)
        if cutlass.const_expr(
            cfg.uses_sw128_nvfp4_raw_payload and kv_kind != KV_KIND_V
        ):
            sf_stage = Int32(self._smem_base_kv_sf.data_ptr().toint()) + (
                stage_idx * Int32(cfg.smem_kv_sf_tile_bytes)
            )
            return ld_shared_v2_b32(
                sf_stage + cfg.smem_kv_sf_row_offset(token) + half * Int32(8)
            )
        words = []
        for rep in cutlass.range_constexpr(self.captured_units):
            _, slab, _ = self.bf16_unit(rep)
            words.append(
                self.scale_words(stage_idx, row, token, slab * Int32(4), 1, kv_kind)[0]
            )
        return tuple(words)

    @consumer_work
    @cute.jit
    def capture_tile(self, stage_info: StageInfo, *, kv_kind: Constexpr[int]) -> None:
        """Copy this thread's units of the raw ``kv_kind`` tile into registers.

        Every raw SMEM read finishes here, so the stage can be released
        before the transform waits for its destination; the registers stay
        owned by the transform until its last destination commit. Unit
        ``idx`` is ``bf16_unit(idx)``, and capture slot ``c`` of a unit holds
        the payload chunk ``SmemKvFromNvfp4Resource.bf16_from_raw`` converts
        at its step ``c``.
        """
        assert self.uses_register_capture
        token, _, half = self.bf16_unit(0)
        row = self.payload_row(stage_info.stage_idx, token)
        scales = self.bf16_unit_scale_words(stage_info.stage_idx, token, half, kv_kind)
        for idx in cutlass.range_constexpr(self.captured_units):
            _, slab, _ = self.bf16_unit(idx)
            for chunk in cutlass.range_constexpr(2):
                words = ld_shared_v4_b32(
                    self.payload_chunk(
                        row,
                        token,
                        ((Int32(chunk) ^ half) << Int32(4)) + slab * Int32(32),
                    )
                )
                for word_idx in cutlass.range_constexpr(4):
                    self.captured_words[8 * idx + 4 * chunk + word_idx] = words[
                        word_idx
                    ]
            self.captured_scales[idx] = scales[idx]


@dataclass(kw_only=True)
class SmemKvFromNvfp4Resource(SmemKvTileResource):
    """FP8 or BF16 K or V tile ring written by TransformKvTask from NVFP4.

    MMA consumes it exactly like a TMA-loaded ``SmemKvTileResource`` of the
    matching KeepsMmaAb profile: 128-row slabs of 128-byte SW128 rows,
    token-major for both K (K-major B) and V (MN-major B), so neither operand
    needs a transpose. A D256 stage holds two 128-channel FP8 slabs or four
    64-channel BF16 slabs, so one raw tile fills one stage of either dtype.
    ``raw_src`` is the raw NVFP4 ring whose current consumer stage the
    transform converts.
    """

    raw_src: Constexpr[SmemKvResource | None] = None

    @producer_work
    @cute.jit
    def transform_from_raw(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Convert the raw ring's current tile into this ring's stage.

        ``head_dim_stage_idx`` selects a BF16 head-dim stage; the D256
        Keeps profiles have one stage for either dtype.
        """
        if cutlass.const_expr(self.cfg.k_dtype_bytes == 2):
            self.bf16_from_raw(stage_info, head_dim_stage_idx)
        else:
            assert head_dim_stage_idx == 0
            self.fp8_from_raw(stage_info)

    @cute.jit
    def bf16_from_raw(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Convert the raw tile's D256 head-dim stage to BF16.

        Each thread owns the (token, 64-channel slab) units of
        ``SmemNvfp4RawKvResource.bf16_unit``: 32 payload bytes and their four
        block scales become one 128-byte SW128 row per unit. E2M1 x E4M3
        products are exact in BF16, so this matches the dequantized cache.
        With register capture the bytes come from ``raw_src``'s registers
        (see ``SmemNvfp4RawKvResource.capture_tile``) instead of its stage.
        """
        cfg = self.cfg
        assert cfg.k_dtype_bytes == 2 and cfg.v_dtype_bytes == 2
        assert cfg.headdim == 256 and cfg.head_dim_kv_stage == 256
        assert head_dim_stage_idx == 0
        assert cfg.tile_size_kv == 128
        raw = self.raw_src
        assert (
            raw.captured_units * cfg.transform_kv_num_warps * 32 == 4 * cfg.tile_size_kv
        )
        dst_base = Int32(self._stage_base(stage_info).data_ptr().toint())
        slab_bytes = cfg.tile_size_kv * 128
        src_stage = consumer_work_stage(raw)
        captured = raw.uses_register_capture
        token, _, half = raw.bf16_unit(0)
        if cutlass.const_expr(not captured):
            row = raw.payload_row(src_stage, token)
            sf_words = raw.bf16_unit_scale_words(src_stage, token, half, self.kv_kind)
        # SW128: 16-byte segment s of row r lands at segment s ^ (r % 8).
        swizzle = token % Int32(8)
        # Step ``chunk`` converts payload chunk ``chunk ^ half`` (see
        # bf16_unit); swapping the scale word's 16-bit halves for half 1
        # pairs that chunk with its two block scales.
        sf_selector = Int32(0x3210) ^ (half * Int32(0x2222))
        for idx in cutlass.range_constexpr(raw.captured_units):
            _, slab, _ = raw.bf16_unit(idx)
            if cutlass.const_expr(captured):
                sf_word = raw.captured_scales[idx]
            else:
                sf_word = sf_words[idx]
            sf_word = byte_permute_runtime(sf_word, sf_word, sf_selector)
            # One BF16x2 pair per 16-channel scale block, in step order.
            sf_pairs = (
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 0)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 1)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 2)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 3)),
            )
            dst_row = dst_base + slab * Int32(slab_bytes) + token * Int32(128)
            for chunk in cutlass.range_constexpr(2):
                chunk_idx = Int32(chunk) ^ half
                if cutlass.const_expr(captured):
                    base = 8 * idx + 4 * chunk
                    words = (
                        raw.captured_words[base],
                        raw.captured_words[base + 1],
                        raw.captured_words[base + 2],
                        raw.captured_words[base + 3],
                    )
                else:
                    words = ld_shared_v4_b32(
                        raw.payload_chunk(
                            row, token, (chunk_idx << Int32(4)) + slab * Int32(32)
                        )
                    )
                for word_idx in cutlass.range_constexpr(4):
                    # Eight values per word, sixteen per scale block; each
                    # word becomes one 16-byte BF16 segment in natural order.
                    out = scale_e2m1x8_packed_to_bf16x2x4(
                        words[word_idx], sf_pairs[(4 * chunk + word_idx) // 2]
                    )
                    seg_idx = (chunk_idx << Int32(2)) + Int32(word_idx)
                    st_shared_v4_b32(
                        dst_row + ((seg_idx ^ swizzle) << Int32(4)),
                        out[0],
                        out[1],
                        out[2],
                        out[3],
                    )
        # Generic-proxy SMEM writes must be visible to the async-proxy MMA.
        cute.arch.fence_view_async_shared()

    @cute.jit
    def fp8_from_raw(self, stage_info: StageInfo) -> None:
        """Convert the raw ring's current tile into this ring's E4M3 stage.

        Each thread owns (token, head-dim half) rows: 64 payload bytes and
        their eight block scales become one 128-byte SW128 row. Even and odd
        nibbles are scaled separately (rounding through E4M3, as the FP8-MMA
        variant does everywhere) and interleaved back to natural order.
        """
        cfg = self.cfg
        assert cfg.k_dtype_bytes == 1 and cfg.v_dtype_bytes == 1
        assert cfg.headdim == 256 and cfg.head_dim_kv_stage == 256
        assert cfg.tile_size_kv == 128
        num_threads = cfg.transform_kv_num_warps * 32
        units = 2 * cfg.tile_size_kv
        assert units % num_threads == 0
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.transform_kv_warp_idx
        )
        thread = warp * Int32(32) + cute.arch.lane_idx()
        dst_base = Int32(self._stage_base(stage_info).data_ptr().toint())
        # One head-dim half: 128 values = 64 payload bytes and 8 scale bytes;
        # its FP8 slab holds tile_size_kv rows of 128 bytes.
        half_payload_bytes = cfg.headdim // 4
        half_scale_blocks = cfg.headdim // 32
        slab_bytes = cfg.tile_size_kv * 128
        raw = self.raw_src
        src_stage = consumer_work_stage(raw)
        for rep in cutlass.range_constexpr(units // num_threads):
            unit = thread + Int32(rep * num_threads)
            token = unit % Int32(cfg.tile_size_kv)
            half = unit // Int32(cfg.tile_size_kv)
            row = raw.payload_row(src_stage, token)
            sf_lo, sf_hi = raw.scale_words(
                src_stage, row, token, half * Int32(half_scale_blocks), 2, self.kv_kind
            )
            dst_row = dst_base + half * Int32(slab_bytes) + token * Int32(128)
            # SW128: 16-byte segment s of row r lands at segment s ^ (r % 8).
            swizzle = token % Int32(8)
            # Each 16-byte payload chunk holds 32 values: two scale blocks.
            for chunk in cutlass.range_constexpr(4):
                words = ld_shared_v4_b32(
                    raw.payload_chunk(
                        row, token, half * Int32(half_payload_bytes) + Int32(chunk * 16)
                    )
                )
                sf_word = sf_lo if chunk < 2 else sf_hi
                scales = (
                    broadcast_byte_const(sf_word, (2 * chunk) % 4),
                    broadcast_byte_const(sf_word, (2 * chunk + 1) % 4),
                )
                out = []
                for word_idx in cutlass.range_constexpr(4):
                    sf = scales[word_idx // 2]
                    # Low nibbles are the even head-dim values, high nibbles
                    # the odd ones; interleave the two product words back.
                    even = _mul_e2m1x4_e4m3x4(words[word_idx], sf)
                    odd = _mul_e2m1x4_e4m3x4(words[word_idx] >> Int32(4), sf)
                    out.append(byte_permute(even, odd, 0x5140))
                    out.append(byte_permute(even, odd, 0x7362))
                for seg in cutlass.range_constexpr(2):
                    seg_idx = Int32(chunk * 2 + seg)
                    st_shared_v4_b32(
                        dst_row + ((seg_idx ^ swizzle) << Int32(4)),
                        out[4 * seg],
                        out[4 * seg + 1],
                        out[4 * seg + 2],
                        out[4 * seg + 3],
                    )
        # Generic-proxy SMEM writes must be visible to the async-proxy MMA.
        cute.arch.fence_view_async_shared()
