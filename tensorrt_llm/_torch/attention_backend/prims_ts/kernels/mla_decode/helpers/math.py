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

"""Math, dtype, and atomic helper functions for MLA decode TS examples."""

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass.experimental import primitives as prims

# Softmax converts exp2 inputs with log2(e) and initializes masked scores to
# negative Float32 max.
LOG2_E = 1.4426950408889634074
NEG_FLT_MAX = -3.4028235e38

# E4M3FN finite maximum used when clamping FP8 output conversion.
FP8_E4M3_MAX = 448.0

fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")

fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd="rn")
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd="rn")
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd="rn")


def ceil_div(a, b):
    """Return integer ceil(a / b) for positive integer-like values."""
    return (a + b - 1) // b


def qkv_dtype(cfg):
    """Return the shared Q/K/V element dtype for MLA decode examples."""
    if cfg.is_fp8_qkv():
        return cutlass.Float8E4M3FN
    return cutlass.BFloat16


def output_dtype(cfg):
    """Return the output element dtype for MLA decode examples."""
    if getattr(cfg, "use_fp8_output", 0) == 1:
        return cutlass.Float8E4M3FN
    return cutlass.BFloat16


def partial_output_dtype(cfg):
    """Return the split-KV partial-O dtype used before the final reduction."""
    if getattr(cfg, "num_ctas_per_seq_kv", 1) > 1:
        return cutlass.BFloat16
    return output_dtype(cfg)


def qkv_smem_swizzle(cfg):
    """Return the SMEM swizzle used by staged Q/K/V tensors."""
    if cfg.is_fp8_qkv() and cfg.head_dim_per_stage_kv == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def qkv_smem_swizzle_for_head_dim(cfg, head_dim: int):
    """Return the staged Q/K/V SMEM swizzle for one head-dimension width."""
    if cfg.is_fp8_qkv() and head_dim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def mma_kind_for_qkv(cfg):
    """Return the tcgen05 MMA kind for QK and PV operations."""
    if cfg.is_fp8_qkv():
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def mma_k_step_for_qkv(cfg) -> int:
    """Return the K step consumed by one tcgen05 MMA instruction."""
    return 32 if cfg.is_fp8_qkv() else 16


def qkv_major_k_stride_bytes_for(cfg, head_dim: int) -> int:
    """Return the shared-memory major-K stride for one Q/K/V head dimension."""
    num_smem_cols = 128 // cfg.qkv_dtype_bytes
    rows_per_smem_row = max(1, num_smem_cols // head_dim)
    if rows_per_smem_row == 1:
        rows_per_swizzle_block = 8
    elif rows_per_smem_row == 2:
        rows_per_swizzle_block = 4
    elif rows_per_smem_row == 4:
        rows_per_swizzle_block = 2
    else:
        rows_per_swizzle_block = 1
    return 128 * rows_per_swizzle_block


def qk_desc_layout_for_head_dim(cfg, head_dim: int):
    """Return the UMMA descriptor layout for a QK operand head dimension."""
    if cfg.is_fp8_qkv() and head_dim == 64:
        return 4
    return 2


def qk_desc_layout(cfg):
    """Return the UMMA descriptor layout for the latent QK operand."""
    return qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def qk_desc_leading_byte_offset_for_head_dim(cfg, tile_rows: int, head_dim: int) -> int:
    """Return the descriptor leading byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return tile_rows * head_dim * cfg.qkv_dtype_bytes
    return 16


def qk_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for latent QK."""
    return qk_desc_leading_byte_offset_for_head_dim(
        cfg, cfg.mma_qk_tiler[0] // cfg.num_mma_ctas, cfg.mma_qk_tiler_k
    )


def qk_desc_stride_byte_offset_for_head_dim(cfg, head_dim: int) -> int:
    """Return the descriptor stride byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return qkv_major_k_stride_bytes_for(cfg, head_dim)
    return 1024


def qk_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for latent QK."""
    return qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def p_desc_layout(cfg):
    """Return the UMMA descriptor layout for P in SMEM."""
    del cfg
    return 4


def p_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for P in SMEM."""
    del cfg
    return 16


def p_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for P in SMEM."""
    del cfg
    return 512


def neg_max_f32():
    """Return the sentinel negative max value used by online softmax."""
    return Float32(NEG_FLT_MAX)


@cute.jit
def pack_float2_to_bf16(v0: Float32, v1: Float32):
    """Pack two Float32 values into one Int32 containing two BF16 lanes."""
    return (
        cutlass.Vector.from_elements((v0, v1), Float32)
        .to(cutlass.BFloat16)
        .bitcast(Int32)[0]
    )


@cute.jit
def float_to_u32_for_atomic_max(val: Float32):
    """Encode a Float32 value so unsigned atomic max preserves float ordering."""
    bits = prims.mov_b32(val, target_type=Int32)
    mask = (bits >> Int32(31)) | Int32(0x80000000)
    encoded = bits ^ mask
    return prims.mov_b32(encoded, target_type=Uint32)


@cute.jit
def u32_to_float_for_atomic_max(val: Uint32):
    """Decode an unsigned atomic-max ordered value back into Float32."""
    encoded = prims.mov_b32(val, target_type=Int32)
    mask = (~(encoded >> Int32(31))) | Int32(0x80000000)
    bits = encoded ^ mask
    return prims.mov_b32(bits, target_type=Float32)


@cute.jit
def smem_atomic_max_u32(ptr, val: Uint32):
    """Apply a CTA-scoped unsigned max atomic to shared memory."""
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        ptr,
        val,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


@cute.jit
def init_softmax_scratch_u32(scratch, warp_grp_thread_idx, num_entries: int):
    """Initialize encoded softmax scratch entries to the negative max sentinel."""
    encoded = float_to_u32_for_atomic_max(neg_max_f32())
    for scratch_idx in cutlass.range(
        warp_grp_thread_idx, Int32(num_entries), Int32(128), unroll=1
    ):
        scratch[scratch_idx] = encoded
