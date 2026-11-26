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

"""PrimTS attention output written as the grouped MXFP8 O-A projection's input.

The kernel's final output stage calls ``MXFP8OutputQuant.store`` with FP32 values instead of
storing BF16. It writes what ``projection.inverse_rope_mxfp8`` makes of the BF16
output, byte for byte: E4M3 ``[groups, T, D]`` and per-group CUTLASS-swizzled UE8M0 scales
with rows padded to 128, ``D = (heads / groups) * 512``.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

__all__ = ["MXFP8OutputQuant", "mxfp8_output_args"]

_HEAD_DIM = 512
_ROPE_START = 448
_BLOCK = 32
# The kernel's generated code depends on this file; persistent caches key on its bytes.
_SOURCE_KEY = sha256(Path(__file__).read_bytes()).hexdigest()


@cutlass.dsl_user_op
def _max_abs(m, x, *, loc=None, ip=None):
    """``max(m, |x|)`` without flushing subnormals, as Triton's ``tl.max(tl.abs(x))``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (m, x)],
            "{\n  .reg .f32 a;\n  abs.f32 a, $2;\n  max.f32 $0, $1, a;\n}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _asm_u32(ptx: str, constraints: str, operands, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip) for v in operands],
            ptx,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16, RN, packed with ``lo`` in the low half."""
    return _asm_u32(
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        [Float32(hi), Float32(lo)],
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _abs_bf16x2(x, *, loc=None, ip=None):
    """Elementwise BF16 absolute value; NaN stays NaN and subnormals are kept."""
    return _asm_u32("abs.bf16x2 $0, $1;", "=r,r", [Uint32(x)], loc=loc, ip=ip)


@cutlass.dsl_user_op
def _max_bf16x2(x, y, *, loc=None, ip=None):
    """Elementwise BF16 maximum; a NaN loses to a number, as ``max.f32`` does."""
    return _asm_u32("max.bf16x2 $0, $1, $2;", "=r,r,r", [Uint32(x), Uint32(y)], loc=loc, ip=ip)


@cutlass.dsl_user_op
def _max_magnitude_bf16x2(x, y, *, loc=None, ip=None):
    """Elementwise BF16 ``max(|x|, |y|)`` with the XOR of the input signs as its sign.

    The magnitudes are those of ``max.bf16x2`` applied to ``abs.bf16x2`` of each input,
    including a NaN losing to a number; only the sign differs, and callers drop it.
    """
    return _asm_u32(
        "max.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", [Uint32(x), Uint32(y)], loc=loc, ip=ip
    )


@cutlass.dsl_user_op
def _mul_bf16x2(x, y, *, loc=None, ip=None):
    """Elementwise BF16 product, RN, subnormals kept."""
    return _asm_u32("mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", [Uint32(x), Uint32(y)], loc=loc, ip=ip)


@cutlass.dsl_user_op
def _e4m3x4(first, second, *, loc=None, ip=None):
    """Four E4M3 bytes, RN and saturating, of two packed BF16 pairs (low element first)."""
    return _asm_u32(
        "{\n  .reg .b16 lo, hi;\n"
        "  cvt.rn.satfinite.e4m3x2.bf16x2 lo, $1;\n"
        "  cvt.rn.satfinite.e4m3x2.bf16x2 hi, $2;\n"
        "  mov.b32 $0, {lo, hi};\n}",
        "=r,r,r",
        [Uint32(first), Uint32(second)],
        loc=loc,
        ip=ip,
    )


@cute.jit
def _low(word: Uint32) -> Float32:
    """The low BF16 of a packed pair, widened exactly."""
    return (word << Uint32(16)).bitcast(Float32)


@cute.jit
def _high(word: Uint32) -> Float32:
    """The high BF16 of a packed pair, widened exactly."""
    return (word & Uint32(0xFFFF0000)).bitcast(Float32)


@dataclass(frozen=True)
class MXFP8OutputQuant:
    """Final-output stage of the PrimTS kernel that emits the grouped MXFP8 O-A input.

    Runtime arguments are ``(data, positions, cos_sin, scales)``: flat E4M3 storage of
    ``[groups, T, D]``, INT32 ``[T]`` token positions, FP32 ``[P, 64]`` cos then sin per
    position and an INT32 view whose base address is that of the swizzled UE8M0 scale
    bytes (``mxfp8_output_args`` builds them). Rows are ``token * heads + head``.

    Each head channel is rounded to BF16 as the kernel's BF16 output would be. Channels
    448..511 then rotate back from the GPT-J position, in FP32, and round to BF16 again.
    Each 1x32 block is quantized as ``mxfp8_quantize`` does.
    """

    num_heads: int
    num_groups: int
    # Read by the native compile path, which shapes placeholder arguments from them.
    block_size: int = _BLOCK
    ue8m0: bool = True
    source_key: str = _SOURCE_KEY

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.num_groups <= 0 or self.num_heads % self.num_groups:
            raise ValueError("MXFP8 output groups must divide the attention heads")
        if self.block_size != _BLOCK or not self.ue8m0:
            raise ValueError("MXFP8 output stores 1x32 UE8M0 blocks")

    # Shared bytes per warp for one token's inverse-RoPE table: 32 cos, then 32 sin.
    rotary_scratch_bytes: int = 256
    # ``store`` writes each lane's 128 E4M3 channels into its 128-byte row of a TMA SW128
    # staging box and takes ``rotary_scratch``, so the WS TMA output path may use it.
    stages_ws_sw128_rows: ClassVar[bool] = True

    @cute.jit
    def prefetch(self, args, row, phase: cutlass.Constexpr[int], state=None):
        """Load a row's inverse-RoPE inputs ahead of ``store`` without waiting on them.

        Phase 0, when the row's work starts, returns its token position. Phase 1, when
        its output is final, takes that position and returns this lane's cosine and
        sine (pair ``lane`` of the token's table); ``stage_rotary`` publishes them.
        """
        _, positions, cos_sin, _ = args
        if cutlass.const_expr(phase == 0):
            return Int32(positions[Int64(row) // self.num_heads])
        lane = cute.arch.thread_idx()[0] % 32
        # Row-major [P, 64] table: address it directly, not by a flat tensor index.
        trig = cos_sin.iterator.raw_ptr() + Int64(state) * 64 + lane
        return Float32(trig.load()), Float32((trig + 32).load())

    @cute.jit
    def stage_rotary(self, staged, scratch):
        """Write the warp's staged cos/sin pairs to its ``rotary_scratch_bytes`` slot."""
        lane = cute.arch.thread_idx()[0] % 32
        cos, sin = staged
        cutlass.inttoptr(scratch + lane * 4, mem_space=3, dtype=Float32).store(cos)
        cutlass.inttoptr(scratch + 128 + lane * 4, mem_space=3, dtype=Float32).store(sin)
        cute.arch.sync_warp()

    @cute.jit
    def store(
        self,
        args,
        values,
        row,
        column,
        lane_stride: cutlass.Constexpr[int] = 1,
        rotary: cutlass.Constexpr[bool] = True,
        shared_output=None,
        rotary_scratch=None,
        scale_out: cutlass.Constexpr[bool] = False,
    ):
        """Quantize lane-local runs of whole 1x32 blocks of one head's final output.

        The E4M3 bytes go to global memory, or with ``shared_output`` to the calling
        lane's 128-byte row of a TMA SW128 staging box: ``shared_output`` is the box's
        1 KiB-aligned shared address, the lane index is the row and ``column % 128`` the
        byte. The caller stores the box. Block scales always go to global memory.
        With ``rotary_scratch`` (a slot ``stage_rotary`` wrote) the inverse-RoPE
        table is read from shared memory instead of the global position table.
        With ``scale_out`` a single block's UE8M0 scale is returned instead of
        stored; ``store_scales`` then stores four consecutive blocks' at once.
        """
        return self.store_words(
            args,
            self.round_words(values),
            row,
            column,
            lane_stride=lane_stride,
            rotary=rotary,
            shared_output=shared_output,
            rotary_scratch=rotary_scratch,
            scale_out=scale_out,
        )

    @cute.jit
    def round_words(self, values):
        """BF16-rounded channels as packed pairs, low channel in the low half.

        These are the kernel's BF16 output values. A caller can hold them instead of the
        FP32 accumulator and pass them to ``store_words`` later.
        """
        count = cutlass.const_expr(values.shape[0])
        words = cutlass.Array(Uint32, count // 2, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(count // 2):
            words[i] = _bf16x2(values[2 * i], values[2 * i + 1])
        return words

    @cute.jit
    def store_words(
        self,
        args,
        words,
        row,
        column,
        lane_stride: cutlass.Constexpr[int] = 1,
        rotary: cutlass.Constexpr[bool] = True,
        shared_output=None,
        rotary_scratch=None,
        scale_out: cutlass.Constexpr[bool] = False,
    ):
        """``store`` of ``round_words`` output; ``words`` is updated in place by the rotation."""
        count = cutlass.const_expr(words.shape[0] * 2)
        if cutlass.const_expr(count % _BLOCK != 0):
            raise NotImplementedError("CSA2 MXFP8 output takes lane-local whole 1x32 blocks")
        data, positions, cos_sin, scales = args
        heads_per_group = cutlass.const_expr(self.num_heads // self.num_groups)
        blocks_per_row = cutlass.const_expr(heads_per_group * _HEAD_DIM // _BLOCK)
        token = Int32(Int64(row) // self.num_heads)
        head = Int32(Int64(row) % self.num_heads)
        group = head // heads_per_group
        head_in_group = head % heads_per_group
        tokens = Int32(Int64(cute.size(data)) // (self.num_heads * _HEAD_DIM))
        scale_rows = (tokens + 127) // 128 * 128
        data_base = Int64((group * tokens + token) * heads_per_group + head_in_group) * _HEAD_DIM
        # CUTLASS 128x4 scale atom: 32 rows x 16 B, four 32-row slices, 4 K blocks each.
        scale_base = (
            Int64(group) * Int64(scale_rows) * blocks_per_row
            + Int64(token // 128) * (blocks_per_row * 128)
            + Int64(token % 32) * 16
            + Int64((token % 128) // 32) * 4
        )
        if cutlass.const_expr(rotary):
            if column + count - 1 >= _ROPE_START:
                position = Int64(0)
                if cutlass.const_expr(rotary_scratch is None):
                    position = Int64(positions[token])
                for start in cutlass.range_constexpr(0, count, _BLOCK):
                    if column + start >= _ROPE_START:
                        pairs = cutlass.const_expr(_BLOCK // 2)
                        pair = (column + start - _ROPE_START) // 2
                        if cutlass.const_expr(rotary_scratch is None):
                            trig = cos_sin.iterator.raw_ptr() + position * 64 + pair
                            cos = trig.load(count=pairs, alignment=16)
                            sin = (trig + 32).load(count=pairs, alignment=16)
                        else:
                            staged = cutlass.inttoptr(
                                rotary_scratch + pair * 4, mem_space=3, dtype=Float32
                            )
                            cos = staged.load(count=pairs, alignment=16)
                            sin = (staged + 32).load(count=pairs, alignment=16)
                        for j in cutlass.range_constexpr(pairs):
                            pair = words[start // 2 + j]
                            even, odd = _low(pair), _high(pair)
                            # As Triton contracts even*cos + odd*sin and odd*cos - even*sin.
                            sine = cute.arch.mul_packed_f32x2(
                                (odd, even), (sin[j], sin[j]), ftz=False, rnd="rn"
                            )
                            rotated = cute.arch.fma_packed_f32x2(
                                (even, odd),
                                (cos[j], cos[j]),
                                (sine[0], -sine[1]),
                                ftz=False,
                                rnd="rn",
                            )
                            words[start // 2 + j] = _bf16x2(rotated[0], rotated[1])
        for start in cutlass.range_constexpr(0, count, _BLOCK):
            # A max tree over the packed BF16 magnitudes: the block's values are exactly
            # BF16, so this equals the FP32 max of their magnitudes. Each level takes the
            # magnitudes of its inputs itself; the final _max_abs drops the XORed sign.
            level = [words[start // 2 + j] for j in range(_BLOCK // 2)]
            # 16 pairs -> 1 in four levels, unrolled at trace time.
            for depth in cutlass.range_constexpr(4):
                width = (_BLOCK // 2) >> (depth + 1)
                level = [
                    _max_magnitude_bf16x2(level[2 * k], level[2 * k + 1]) for k in range(width)
                ]
            amax = _max_abs(_max_abs(Float32(0.0), _low(level[0])), _high(level[0]))
            scale = cute.arch.mul_packed_f32x2(
                (amax, amax), (Float32(1.0 / 448.0), Float32(1.0 / 448.0)), ftz=False, rnd="rn"
            )[0]
            bits = scale.bitcast(Uint32)
            exponent = (bits >> 23) & Uint32(255)
            if (bits & Uint32(0x007FFFFF)) != Uint32(0):
                exponent = exponent + Uint32(1)
            # scale <= 2**-127; amax >= 0 makes the bit pattern ordered.
            if bits <= Uint32(0x00400000):
                exponent = Uint32(0)
            if exponent > Uint32(254):
                exponent = Uint32(254)
            reciprocal = ((Uint32(254) - exponent) << 23).bitcast(Float32)
            if amax == Float32(0.0):
                reciprocal = Float32(0.0)
            block = (column + start) // _BLOCK
            kblock = head_in_group * (_HEAD_DIM // _BLOCK) + block
            scale_ptr = cutlass.inttoptr(
                (
                    scales.iterator.raw_ptr().toint(Int64)
                    + scale_base
                    + Int64(kblock // 4) * 512
                    + Int64(kblock % 4)
                ),
                mem_space=1,
                dtype=cutlass.Uint8,
            )
            if cutlass.const_expr(not scale_out):
                scale_ptr.store(exponent.to(cutlass.Uint8))
            packed = cutlass.Array(Int32, _BLOCK // 4, space=cutlass.AddressSpace.rmem)
            # The power-of-two reciprocal scales BF16 values exactly (a product
            # below 2**-126 rounds to E4M3 zero either way), so BF16 products
            # convert to the same E4M3 bytes as the FP32 ones.
            half = reciprocal.bitcast(Uint32) >> Uint32(16)
            reciprocal2 = half | (half << Uint32(16))
            for w in cutlass.range_constexpr(_BLOCK // 4):
                first, second = words[start // 2 + 2 * w], words[start // 2 + 2 * w + 1]
                packed[w] = _e4m3x4(
                    _mul_bf16x2(first, reciprocal2), _mul_bf16x2(second, reciprocal2)
                ).bitcast(Int32)
            if cutlass.const_expr(shared_output is not None):
                lane = cute.arch.thread_idx()[0] % 32
                for half in cutlass.range_constexpr(2):
                    offset = lane * 128 + ((column + start + 16 * half) & 127)
                    # TMA 128B swizzle XORs byte bits [4:7] with [7:10].
                    swizzled = offset ^ ((offset >> 3) & 0x70)
                    staged = cutlass.inttoptr(shared_output + swizzled, mem_space=3, dtype=Int32)
                    staged.store(packed.load(4 * half, 4), alignment=16)
            else:
                destination = cutlass.inttoptr(
                    data.iterator.raw_ptr().toint(Int64) + data_base + column + start,
                    mem_space=1,
                    dtype=Int32,
                )
                destination.store(packed.load(0, 4), alignment=16)
                (destination + 4).store(packed.load(4, 4), alignment=16)
        if cutlass.const_expr(scale_out):
            assert count == _BLOCK, "scale_out returns one block's scale"
            return exponent

    @cute.jit
    def store_scales(self, args, row, column, word):
        """Store the UE8M0 scales of the four blocks from ``column`` (a multiple of 128).

        ``word`` holds them low byte first; the 128x4 layout keeps them adjacent.
        """
        data, _, _, scales = args
        heads_per_group = cutlass.const_expr(self.num_heads // self.num_groups)
        blocks_per_row = cutlass.const_expr(heads_per_group * _HEAD_DIM // _BLOCK)
        token = Int32(Int64(row) // self.num_heads)
        head = Int32(Int64(row) % self.num_heads)
        group = head // heads_per_group
        head_in_group = head % heads_per_group
        tokens = Int32(Int64(cute.size(data)) // (self.num_heads * _HEAD_DIM))
        scale_rows = (tokens + 127) // 128 * 128
        kblock = head_in_group * (_HEAD_DIM // _BLOCK) + column // _BLOCK
        address = (
            scales.iterator.raw_ptr().toint(Int64)
            + Int64(group) * Int64(scale_rows) * blocks_per_row
            + Int64(token // 128) * (blocks_per_row * 128)
            + Int64(token % 32) * 16
            + Int64((token % 128) // 32) * 4
            + Int64(kblock // 4) * 512
        )
        cutlass.inttoptr(address, mem_space=1, dtype=Uint32).store(Uint32(word))


def mxfp8_output_args(
    data: torch.Tensor,
    scales: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind one launch's MXFP8 output after checking it; zero the padded scale rows.

    ``data`` is contiguous E4M3 ``[groups, T, D]``, ``scales`` contiguous UINT8
    ``groups * pad128(T) * D / 32`` swizzled bytes, ``positions`` contiguous INT32 ``[T]``
    and ``cos_sin`` contiguous FP32 with 64 values per position. The kernel writes the
    scales of real rows only; rows from ``T`` to the next multiple of 128 are zeroed here,
    as ``inverse_rope_mxfp8`` writes them.
    """
    if data.ndim != 3:
        raise ValueError("CSA2 MXFP8 output data must be E4M3 [groups, T, D]")
    groups, rows, depth = data.shape
    scale_rows = (rows + 127) // 128 * 128
    if (
        data.dtype != torch.float8_e4m3fn
        or not data.is_contiguous()
        or data.data_ptr() % 32
        or groups * depth != heads * _HEAD_DIM
        or depth % _HEAD_DIM
        or scales.dtype != torch.uint8
        or not scales.is_contiguous()
        or scales.data_ptr() % 16
        or scales.numel() != groups * scale_rows * depth // _BLOCK
        or positions.dtype != torch.int32
        or positions.shape != (rows,)
        or not positions.is_contiguous()
        or cos_sin.dtype != torch.float32
        or not cos_sin.is_contiguous()
        or cos_sin.numel() % 64
        or cos_sin.data_ptr() % 16
        or len({t.device for t in (data, scales, positions, cos_sin)}) != 1
    ):
        raise ValueError("CSA2 MXFP8 output got an unsupported buffer")
    if rows % 128:
        scales.view(groups, scale_rows // 128, -1)[:, -1].zero_()
    # The native placeholder is an INT32 [T, groups, columns] view with padded row-major
    # columns; only its base address is read.
    padded = (rows + 3) // 4 * 4
    columns = depth // _BLOCK // 4
    view = scales.view(torch.int32).as_strided(
        (rows, groups, columns), (1, columns * padded, padded)
    )
    return data.view(-1), positions, cos_sin.view(-1, 64), view
