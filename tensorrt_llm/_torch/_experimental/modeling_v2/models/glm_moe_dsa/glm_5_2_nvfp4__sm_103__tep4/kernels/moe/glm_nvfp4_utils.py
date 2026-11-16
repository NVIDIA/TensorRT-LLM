# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 NVFP4 host/kernel helpers: the single place GLM's constants are defined.

Why this file exists
--------------------
`moe_utils` is the GPT-OSS configuration: `a_dtype = Float8E4M3FN`,
`sf_dtype = Float8E8M0FNU`, `sf_vec_size = 32`, `torch_sf_dtype = float8_e8m0fnu`.
Several of its functions read those module-level names directly, so importing them
into a file that overrides the same names in *its own* namespace silently keeps the
GPT-OSS behaviour: the override has no effect on the imported function.

For example, moe_utils' `compute_sf_rest` computes the SF atom's K span as
`num_k_per_sf_atom * sf_vec_size` = 4 * 32 = 128 instead of GLM's 4 * 16 = 64. The
scale-factor TMEM addressing is then off by 2x and the output is NaN.
moe_utils' `compute_ktile_config` likewise derives K slicing from its 8-bit `a_dtype`.

This module reimplements both functions with GLM's values: `compute_sf_rest` and
`compute_ktile_config`. gg1/gg2 take them and the constants below from this module,
not from moe_utils.

GLM-5.2 configuration is fixed by the checkpoint, not chosen:
`hf_quant_config.json` says quant_algo=NVFP4 / group_size=16, and the weight_scale
tensors are F8_E4M3 (weight_scale_2, the global dequant scale, is F32 -> `alpha`).
"""

from __future__ import annotations

import cutlass

# ---------------------------------------------------------------------------
# GLM-5.2 NVFP4 configuration
# ---------------------------------------------------------------------------
a_dtype = cutlass.Float4E2M1FN
b_dtype = cutlass.Float4E2M1FN
# Model/TMA storage is byte-addressed FP4x2; MMA geometry remains scalar FP4.
fp4x2_dtype = cutlass.Float4E2M1FNx2
fp4_lanes_per_storage = fp4x2_dtype.width // a_dtype.width
sf_dtype = cutlass.Float8E4M3FN
sf_vec_size = 16  # NVFP4: one scale per 16 elements (MX uses 32)

# Scale-factor atom geometry: (m0=32, m1=4, k=4) = 512 bytes.
# Independent of sf_vec_size -- in the atom layout
# ((32,4),(sf_vec_size,4)):((16,4),(0,1)) the sf_vec_size mode has stride 0, so it
# contributes no storage. Verified against the layout comment in moe_utils.
num_m0_per_sf_atom = 32
num_m1_per_sf_atom = 4
num_k_per_sf_atom = 4
num_elts_atom_sf_e8 = num_m0_per_sf_atom * num_m1_per_sf_atom * num_k_per_sf_atom  # 512
num_elts_atom_sf_fp16 = num_elts_atom_sf_e8 // 2
num_tmem_cols_per_sf_atom = 4

# One SMEM row is one 128-byte core matrix (the unit the MMA state machine reads).
smem_row_bytes = 128


def compute_sf_rest(tiler_mnk: list[int], inst_mnk: list[int]) -> tuple[dict[str, int], int, int]:
    """SF atom counts per tile and TMEM columns per MMA kblock, for NVFP4.

    Identical in structure to moe_utils', but `sf_atom_k` uses GLM's sf_vec_size:

        GPT-OSS: 4 * 32 = 128 K elements per atom
        GLM:     4 * 16 =  64 K elements per atom

    Using the wrong one misprograms the scale-factor TMEM addressing; the kernel
    then reads E4M3 bytes from the wrong offsets and the output goes all-NaN.
    """
    sf_atom_mn = num_m0_per_sf_atom * num_m1_per_sf_atom  # 128
    sf_atom_k = num_k_per_sf_atom * sf_vec_size  # 64 for NVFP4

    assert tiler_mnk[2] % sf_atom_k == 0, (
        f"K tile ({tiler_mnk[2]}) must be divisible by SF atom K ({sf_atom_k})"
    )
    assert tiler_mnk[0] % sf_atom_mn == 0, (
        f"M tile ({tiler_mnk[0]}) must be divisible by SF atom MN ({sf_atom_mn})"
    )
    assert tiler_mnk[1] % sf_atom_mn == 0 or sf_atom_mn % tiler_mnk[1] == 0, (
        f"N tile ({tiler_mnk[1]}) must divide or be divisible by SF atom MN"
    )
    assert inst_mnk[0] % sf_atom_mn == 0, (
        f"M inst ({inst_mnk[0]}) must be divisible by SF atom MN ({sf_atom_mn})"
    )
    assert inst_mnk[1] % sf_atom_mn == 0 or sf_atom_mn % inst_mnk[1] == 0, (
        f"N inst ({inst_mnk[1]}) must divide or be divisible by SF atom MN"
    )
    assert tiler_mnk[2] % inst_mnk[2] == 0, (
        f"K tile ({tiler_mnk[2]}) must be divisible by K inst ({inst_mnk[2]})"
    )

    tile_rest = {
        "k": max(1, tiler_mnk[2] // sf_atom_k),
        "m": max(1, tiler_mnk[0] // sf_atom_mn),
        "n": max(1, tiler_mnk[1] // sf_atom_mn),
    }
    rest_k = max(1, inst_mnk[2] // sf_atom_k)
    tmem_cols_per_kblock_sfa = (
        num_tmem_cols_per_sf_atom * max(1, inst_mnk[0] // sf_atom_mn) * rest_k
    )
    tmem_cols_per_kblock_sfb = (
        num_tmem_cols_per_sf_atom * max(1, inst_mnk[1] // sf_atom_mn) * rest_k
    )
    return tile_rest, tmem_cols_per_kblock_sfa, tmem_cols_per_kblock_sfb


def compute_ktile_config(mma_tiler_k: int) -> dict:
    """SMEM/MMA geometry for one full-K NVFP4 TMA tile.

    Packed FP4x2 descriptors cover the complete logical K tile in one transaction;
    no source-level slice geometry is needed.

    The descriptor fields themselves are unchanged, and match the values the NVFP4
    example hardcodes (leading 16 / stride 1024 / layout 2 = SWIZZLE_128B):
      - stride_byte_offset: bytes to step one core matrix along y (A's M / B's N),
        = 8 rows x 128 B.
      - leading_byte_offset: bytes to step past one full swizzle period (8 core
        matrices). Within a period the hardware advances by a fixed 16 B, so this
        only matters when a K-tile spans more than one period.
    """
    assert mma_tiler_k in (128, 256), f"mma_tiler_k must be 128 or 256, got {mma_tiler_k}"

    swizzle_mask = 7
    mma_layout = 2  # SWIZZLE_128B
    mma_stride = 1024  # 8 rows x 128 B
    return {
        "smem_row_bytes": smem_row_bytes,
        "mma_layout": mma_layout,
        "mma_stride": mma_stride,
        "mma_leading": 16,
        "swizzle_mask": swizzle_mask,
    }
