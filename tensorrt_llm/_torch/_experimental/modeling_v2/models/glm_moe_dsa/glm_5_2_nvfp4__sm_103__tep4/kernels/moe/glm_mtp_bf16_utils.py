# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and helpers for GLM-5.2's **MTP-layer** MoE, which is BF16.

This is deliberately *not* `glm_nvfp4_utils`. The main GLM-5.2 layers are NVFP4 and go
through the CUTEDSL backend; the MTP layer is left unquantized in the checkpoint
(`modeling_glm.py::_get_decoder_layer_quant_config`), so it falls back to the CUTLASS
grouped-GEMM path and runs in BF16 end to end.

The shapes and routing are identical -- `Glm4MTP` extends `Glm4DecoderLayer`, so
hidden_size 6144, moe_intermediate_size 2048, 256 experts, top-k 8 -- but the data path
is not. Everything the NVFP4 config carries about scale factors (`sf_dtype`,
`sf_vec_size`, the 512-byte atom geometry, `compute_sf_rest`) has no meaning here and is
absent on purpose: importing it would invite code that indexes scale factors that do not
exist.

Two constants differ from the NVFP4 kernels in ways that are easy to carry over by
mistake:

`mma_inst_k` is 16, not 64. The NVFP4 block-scaled atom issues 64 K-elements per
instruction; the F16/BF16 atom issues 16. The K tile is four instructions either way, so
the tile is 64 elements here against 256 there.

`gate_up_layout` is two contiguous halves, not a 64-column interleave. The NVFP4 path
reorders `w3_w1_weight` so up and gate alternate every 64 columns, and
`moe_gather_grouped_gemm_act_fusion.py` hardcodes that. The BF16 path does not reorder:
`quantization.py::load_expert_w3_w1_weight` writes `w3` (up) into the first half and `w1`
(gate) into the second, and `doActivationKernel` reads the gated operand at
`+ inter_size`. So the epilogue pairs column `j` with column `j + inter_size`.
"""

from __future__ import annotations

import cutlass

# ---------------------------------------------------------------------------
# Data types. BF16 end to end -- no scale factors anywhere.
# ---------------------------------------------------------------------------
a_dtype = cutlass.BFloat16
b_dtype = cutlass.BFloat16
c_dtype = cutlass.BFloat16

# ---------------------------------------------------------------------------
# MMA geometry for the F16/BF16 tcgen05 atom.
# ---------------------------------------------------------------------------
# `Tcgen05InstrDesc.build(c_dtype=Float32, n_dim=N, m_dim=M)` with
# `tcgen05_mma(Tcgen05MMAKind.F16, ...)`; each instruction consumes 16 K-elements.
mma_inst_k = 16
mma_insts_per_ktile = 4
mma_k_granule = mma_inst_k * mma_insts_per_ktile  # 64


def compute_ktile_config(mma_tiler_k: int, mma_tiler_n: int) -> dict:
    """SMEM/TMA/descriptor geometry for one K tile, for BF16 operands.

    The governing constraint is the `s128b` swizzle: the TMA bounding box's inner
    dimension may be at most 128 B, so one issue can carry at most `128 * 8 / width`
    K-elements. That is 64 at BF16 and 256 at NVFP4 -- the same 128 B either way. A K
    tile larger than that is fetched in `num_k_slices` issues, and the two SMEM
    descriptor `leading_byte_offset` values change accordingly, because consecutive
    slices are no longer 16 B apart in SMEM but a whole operand row apart.

    At `mma_tiler_k = 64` this degenerates to one slice and the descriptor constants
    coincide, byte for byte, with the ones the NVFP4 kernels run at their 256-element
    tile. That is not a coincidence to lean on, it is the reason 64 is the default:
    the K tile was chosen so the row width matches.
    """
    assert mma_tiler_k % mma_k_granule == 0, (
        f"mma_tiler_k must be a multiple of {mma_k_granule}, got {mma_tiler_k}"
    )

    smem_row_bytes = 128
    k_chunks_per_slice = 8  # 128 B / 16 B
    swizzle_mask = 7
    mma_layout = 2  # SWIZZLE_128B
    mma_stride = 1024  # 8 rows x 128 B

    # Elements that fit in one 128 B TMA box. Setting this to mma_tiler_k regardless --
    # which is correct for the default tile and only for it -- makes the descriptor
    # builder reject the K=128 tile outright rather than slicing it.
    max_box_k = smem_row_bytes * 8 // a_dtype.width
    tma_b_box_k = min(mma_tiler_k, max_box_k)
    slice_row_bytes = tma_b_box_k * a_dtype.width // 8
    ktile_bytes = mma_tiler_k * a_dtype.width // 8
    num_k_slices = max(1, ktile_bytes // slice_row_bytes)

    if num_k_slices > 1:
        mma_leading_a = 128 * smem_row_bytes
        mma_leading_b = mma_tiler_n * smem_row_bytes
    else:
        mma_leading_a = 16
        mma_leading_b = 16

    return {
        "smem_row_bytes": smem_row_bytes,
        "slice_row_bytes": slice_row_bytes,
        "num_k_slices": num_k_slices,
        "tma_b_box_k": tma_b_box_k,
        "mma_layout": mma_layout,
        "mma_stride": mma_stride,
        "mma_leading_a": mma_leading_a,
        "mma_leading_b": mma_leading_b,
        "k_chunks_per_slice": k_chunks_per_slice,
        "swizzle_mask": swizzle_mask,
    }
