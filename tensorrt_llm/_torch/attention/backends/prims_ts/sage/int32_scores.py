"""INT32 scores on a float bias: 8-bit integer QK MMA that the softmax reads as FP32.

INT8 Q/K accumulate onto the bit pattern of ``1.5 * 2**23``. Integers in
``[2**23, 2**24)`` are FP32 numbers with unit spacing and every D = 128 INT8
dot product has ``|score| <= 2**21``, so the INT32 accumulator is the exact
FP32 number ``12582912.0 + score`` and the softmax reads it without a
conversion, removing the bias where it applies its scales.

One ``kind::f16`` MMA step writes the bias before the INT8 K steps: both
operands read one BF16 tile whose K = 16 rows repeat ``[1024, 1024, 1024,
0]``, so the twelve products ``2**20`` sum exactly to ``1.5 * 2**23``. The
tile is unswizzled K-major 32-byte rows with 128-byte LBO and 256-byte SBO
core-matrix strides. The decode and context kernels share the tile and the
seeding step through the helpers below.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.experimental import primitives as prims

from ..kernels.tcgen05_compat import tcgen05_mma_ws

INT32_SCORE_BIAS = 12582912.0
INT32_SCORE_SEED_MMA_K = 16
INT32_SCORE_SEED_TILE_WORDS = (0x44804480, 0x00004480)
INT32_SCORE_SEED_TILE_LBO = 128
INT32_SCORE_SEED_TILE_SBO = 256


def sage_score_seed_tile_bytes(mma_m: int, mma_n: int) -> int:
    """Return the byte size of the seed MMA's operand tile.

    A reads the first M rows and B the first N rows of the same tile.
    """
    return max(mma_m, mma_n) * INT32_SCORE_SEED_MMA_K * 2


@cute.jit
def sage_score_seed_words(smem_base, offset: int, num_bytes: int) -> cutlass.Array:
    """Return the seed MMA's operand tile as a 32-bit SMEM array."""
    return cutlass.Array(
        smem_base.data_ptr() + offset,
        dtype=Uint32,
        shape=(num_bytes // 4,),
        addrspace=3,
    )


@cute.jit
def fill_sage_score_seed_tile(
    words: cutlass.Array, num_words: int, thread_idx: Int32, num_threads: int
) -> None:
    """Fill the seed tile with ``num_threads`` threads, one word pair each per round.

    The fence makes the generic-proxy stores visible to the tensor core.
    """
    even_word, odd_word = INT32_SCORE_SEED_TILE_WORDS
    num_pairs = num_words // 2
    for round_idx in cutlass.range_constexpr(
        (num_pairs + num_threads - 1) // num_threads
    ):
        pair_idx = thread_idx + Int32(round_idx * num_threads)
        if pair_idx < Int32(num_pairs):
            words[pair_idx * 2] = Uint32(even_word)
            words[pair_idx * 2 + 1] = Uint32(odd_word)
    cute.arch.fence_view_async_shared()


@cute.jit
def issue_sage_score_seed_mma(
    tmem_s,
    words: cutlass.Array,
    mma_m: int,
    mma_n: int,
    *,
    ws: cutlass.Constexpr[bool] = False,
) -> None:
    """Write ``INT32_SCORE_BIAS`` into an S stage with one BF16 MMA step.

    The elected thread issues the step; it reads only the constant tile and no
    accumulator, so the INT8 K steps issued afterwards accumulate onto the
    bias bit pattern in order.
    """
    if prims.elect_sync():
        tile_desc = prims.Tcgen05SmemDesc.build(
            words,
            leading_byte_offset=INT32_SCORE_SEED_TILE_LBO,
            stride_byte_offset=INT32_SCORE_SEED_TILE_SBO,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=BFloat16,
            b_dtype=BFloat16,
            n_dim=mma_n,
            m_dim=mma_m,
        )
        if cutlass.const_expr(ws):
            tcgen05_mma_ws(
                prims.Tcgen05MMAKind.F16, tmem_s, tile_desc, tile_desc, idesc, False
            )
        else:
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                tmem_s,
                tile_desc,
                tile_desc,
                idesc,
                False,
            )
