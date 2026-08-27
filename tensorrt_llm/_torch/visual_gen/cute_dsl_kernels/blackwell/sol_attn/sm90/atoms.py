"""Hopper MMA atoms used by Sol-Attn."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from ._compat import sm90_utils


def make_pv_mma(tile_m: int = 64, tile_v: int = 128) -> cute.TiledMma:
    return sm90_utils.make_tiled_mma(
        cutlass.BFloat16,
        "K",
        "MN",
        tile_v,
        source="RS",
        atom_layout_mnk=(tile_m // 64, 1, 1),
        b_dtype=cutlass.BFloat16,
        acc_dtype=Float32,
    )


__all__ = ["make_pv_mma"]
