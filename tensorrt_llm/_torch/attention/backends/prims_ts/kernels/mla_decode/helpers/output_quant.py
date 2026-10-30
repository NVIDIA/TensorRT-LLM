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

"""Final sparse MLA output: conjugate RoPE and block E4M3 quantization."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.experimental import primitives as prims

from .math import mul_packed_f32x2, fma_packed_f32x2
from .ops import pack_float4_to_fp8_e4m3


@cutlass.dsl_user_op
def _max_abs3(x, y, z, *, loc=None, ip=None):
    """SM100 max-of-magnitudes with native floating-point operand constraints."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z)],
            "max.abs.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dataclass(frozen=True)
class MlaOutputQuant:
    """Static geometry; runtime args are (flat O, positions, cos/sin, scales).

    O is E4M3 stored group-major, physically [G, T, (H/G) * 512] (T is derived
    from its extent); positions are INT32; cos/sin is FP32; scales are a
    token-major logical view of column-major FP32 or packed INT32 storage.
    Rows are token * H + head and columns are channels within the D512 head.
    """

    block_size: int
    ue8m0: bool
    num_heads: int
    num_groups: int

    @cute.jit
    def rotate(self, args, values, row, column):
        """Rotate a mutable final-output fragment in place."""
        _, positions, cos_sin, _ = args
        token = Int64(row) // self.num_heads
        count = cutlass.const_expr(values.shape[0])
        fragment = values

        # The cache contains forward sin; conjugation changes its sign here.
        if column + count - 1 >= 448:
            position = positions[token]
            # Load independent coefficients together before consuming them.
            pairs = cutlass.const_expr(min(32, count // 2))
            for i in cutlass.range_constexpr(0, count, pairs * 2):
                d = column + i
                if d >= 448:
                    # Callers provide aligned contiguous channel vectors.
                    base = (
                        cos_sin.iterator.raw_ptr()
                        + Int64(position) * 64
                        + (d - 448) // 2
                    )
                    alignment = cutlass.const_expr(min(16, pairs * 4))
                    cos = base.load(count=pairs, alignment=alignment)
                    sin = (base + 32).load(count=pairs, alignment=alignment)
                    for j in cutlass.range_constexpr(pairs):
                        x, y = fragment[i + 2 * j], fragment[i + 2 * j + 1]
                        sine = mul_packed_f32x2((y, x), (sin[j], sin[j]), ftz=True)
                        rotated = fma_packed_f32x2(
                            (x, y), (cos[j], cos[j]), (sine[0], -sine[1]), ftz=True
                        )
                        fragment[i + 2 * j] = rotated[0]
                        fragment[i + 2 * j + 1] = rotated[1]

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
    ):
        """Finalize contiguous FP32 fragments after all attention reductions.

        Blocks fit in one thread or one warp; ``lane_stride=16`` joins the
        two lanes owning the halves of a keep-MMA head. Nonrotary D stages
        remove the rotation at compile time, limiting epilogue code size.
        ``shared_output`` is the optional 2CTA FP8 output stage: two swizzled
        64-row by 128-byte boxes, consumed by its asynchronous TMA-store task.
        """
        if cutlass.const_expr(rotary):
            # Some epilogues pass an immutable register vector. Two-CTA
            # epilogues rotate their mutable accumulator before this store.
            count = cutlass.const_expr(values.shape[0])
            fragment = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
            for i in cutlass.range_constexpr(count):
                fragment[i] = values[i]
            self.rotate(args, fragment, row, column)
        else:
            fragment = values
        out, _, _, scales = args
        token = Int64(row) // self.num_heads
        head = Int64(row) % self.num_heads
        # Physical [G, T, (H/G) * 512] output; the 512-channel head rows of one
        # (group, token) pair are contiguous. Planning bounds T * H below 2**31,
        # so the row index fits Int32.
        heads_per_group = cutlass.const_expr(self.num_heads // self.num_groups)
        total_tokens = Int32(Int64(cute.size(out)) // (self.num_heads * 512))
        head32 = Int32(head)
        if cutlass.const_expr(heads_per_group & (heads_per_group - 1) == 0):
            shift = cutlass.const_expr(heads_per_group.bit_length() - 1)
            group = head32 >> shift
            head_in_group = head32 & Int32(heads_per_group - 1)
        else:
            group = head32 // heads_per_group
            head_in_group = head32 % heads_per_group
        out_row = Int64(
            (group * total_tokens + Int32(token)) * heads_per_group + head_in_group
        )
        lane = cute.arch.thread_idx()[0] % 32
        count = cutlass.const_expr(fragment.shape[0])
        per_block = cutlass.const_expr(min(count, self.block_size))
        lanes = cutlass.const_expr(self.block_size // per_block)
        # RoPE branches can leave partners at different program counters.
        # Name every lane in the quant block instead of sampling activemask.
        group_base = lane % lane_stride + lane // (lanes * lane_stride) * (
            lanes * lane_stride
        )
        group_bits = cutlass.const_expr(
            sum(1 << (i * lane_stride) for i in range(lanes))
        )
        group_mask = Uint32(group_bits) << Uint32(group_base)
        for start in cutlass.range_constexpr(0, count, per_block):
            # Two independent max3 chains avoid a long scalar amax dependency.
            m0, m1 = Float32(1e-4), Float32(1e-4)
            for i in cutlass.range_constexpr(0, per_block, 4):
                m0 = _max_abs3(m0, fragment[start + i], fragment[start + i + 1])
                m1 = _max_abs3(m1, fragment[start + i + 2], fragment[start + i + 3])
            amax = _max_abs3(m0, m1, Float32(1e-4))
            for step in cutlass.range_constexpr(lanes.bit_length() - 1):
                peer = prims.shfl_sync(
                    thread_mask=group_mask,
                    val=amax,
                    offset=(1 << step) * lane_stride,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
                amax = cute.math.max(amax, peer)
            scale = amax * Float32(1.0 / 448.0)
            if cutlass.const_expr(self.ue8m0):
                exponent = (scale.bitcast(Uint32) + Uint32(0x007FFFFF)) >> 23
                scale = (exponent << 23).bitcast(Float32)
            if cutlass.const_expr(self.ue8m0):
                inverse_bits = (Uint32(254) - exponent) << 23
                # UE8M0 scales are powers of two; keep reciprocal(inf) == 0.
                inv_scale = (
                    Float32(0) if exponent == 255 else inverse_bits.bitcast(Float32)
                )
            else:
                inv_scale = cute.math.rcp(scale, approx=True)
            block = (column + start) // self.block_size
            # The validated [G, columns, pad4(T)] backing is contiguous in
            # head/block order, so group decomposition is unnecessary.
            if lane // lane_stride % lanes == 0:
                if cutlass.const_expr(self.ue8m0):
                    word = head * (512 // self.block_size // 4) + block // 4
                    byte = block % 4
                    offset = token + word * Int64(scales.stride[2])
                    byte_ptr = cutlass.inttoptr(
                        (scales.iterator.raw_ptr() + offset).toint(Int64) + byte,
                        mem_space=1,
                        dtype=cutlass.Uint8,
                    )
                    byte_ptr.store(exponent.to(cutlass.Uint8))
                else:
                    offset = token + (head * (512 // self.block_size) + block) * Int64(
                        scales.stride[2]
                    )
                    (scales.iterator.raw_ptr() + offset).store(scale)

            words = cutlass.const_expr(min(8, per_block // 4))
            for i in cutlass.range_constexpr(0, per_block, 4 * words):
                v = start + i
                packed = cutlass.Array(Int32, words, space=cutlass.AddressSpace.rmem)
                for w in cutlass.range_constexpr(words):
                    j = v + 4 * w
                    lo = mul_packed_f32x2(
                        (fragment[j], fragment[j + 1]), (inv_scale, inv_scale), ftz=True
                    )
                    hi = mul_packed_f32x2(
                        (fragment[j + 2], fragment[j + 3]),
                        (inv_scale, inv_scale),
                        ftz=True,
                    )
                    packed[w] = pack_float4_to_fp8_e4m3(lo[0], lo[1], hi[0], hi[1])
                if cutlass.const_expr(shared_output is not None):
                    for part in cutlass.range_constexpr(0, words, 4):
                        col = Int32(column + v + part * 4) & Int32(255)
                        smem_offset = (
                            (col // 128) * 8192 + (Int32(head) & 63) * 128 + (col & 127)
                        )
                        # TMA 128B swizzle XORs byte bits [4:7] with [7:10].
                        # The 1024B-aligned base leaves these swizzle bits unchanged.
                        swizzled = smem_offset ^ ((smem_offset >> 3) & 0x70)
                        destination = cutlass.inttoptr(
                            shared_output + swizzled, mem_space=3, dtype=Int32
                        )
                        destination.store(packed.load(part, 4), alignment=16)
                else:
                    ptr = cutlass.inttoptr(
                        (out.iterator.raw_ptr() + out_row * 512 + column + v).toint(
                            Int64
                        ),
                        mem_space=1,
                        dtype=Int32,
                    )
                    if cutlass.const_expr(words == 8):
                        # Native allocations support a 256-bit store. Retain the
                        # documented 16-byte alignment for caller-owned buffers.
                        if (ptr.toint(Int64) & Int64(31)) == 0:
                            ptr.store(packed.load(0, 8), alignment=32)
                        else:
                            ptr.store(packed.load(0, 4), alignment=16)
                            (ptr + 4).store(packed.load(4, 4), alignment=16)
                    else:
                        ptr.store(packed.load(0, words), alignment=4 * words)
