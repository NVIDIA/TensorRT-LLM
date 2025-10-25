import itertools
from dataclasses import dataclass
from functools import partial
from typing import List, Literal, Optional


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    pingpong: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8


def get_all_configs(
    device_capacity: Literal[9, 10] = 9,
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
    # tune_raster_order=True,
) -> List[GemmConfig]:
    assert device_capacity in [9, 10]
    if device_capacity == 9:
        tile_n_vals = [128, 144, 160, 176, 192, 208]
        tile_mn_coop_vals = [(256, tile_n) for tile_n in tile_n_vals] + [
            (128, 224),
            (128, 256),
            # (192, 256),  # Getting IOT instruction (core dumped) in the bwd
        ]
        tile_mn_pingpong_vals = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
        if epilogue in ["gated"]:
            tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if n % 32 == 0 and m != 192]
            tile_mn_pingpong_vals = [(m, n) for m, n in tile_mn_pingpong_vals if n % 32 == 0]
        elif epilogue in ["lse"]:
            tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if m != 192]
        tile_mn_vals = []
        if tune_coop:
            tile_mn_vals += [(m, n, False) for m, n in tile_mn_coop_vals]
        tile_mn_vals += [(m, n, True) for m, n in tile_mn_pingpong_vals]
        cluster = [(1, 2), (2, 1)]
        # cluster = [(1, 1), (1, 2), (2, 1)]
        if epilogue in ["lse"]:
            cluster = [(1, 2), (2, 1)]
        swap_ab_vals = [False, True]
        if epilogue in ["lse", "gated"]:
            swap_ab_vals = [False]
        # raster_swizzle = (
        #     [(0, 1)]
        #     if not tune_raster_order
        #     else [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (2, 8)]
        # )
        return [
            GemmConfig(
                tile_m=tile_m,
                tile_n=tile_n,
                pingpong=pingpong,
                cluster_m=cluster_m,
                cluster_n=cluster_n,
                swap_ab=swap_ab,
                # raster_order=raster_order,
                # max_swizzle_size=max_swizzle_size,
            )
            for (tile_m, tile_n, pingpong), (
                cluster_m,
                cluster_n,
            ), swap_ab in itertools.product(
                tile_mn_vals,
                cluster,
                swap_ab_vals,
                # raster_swizzle,
            )
        ]
    elif device_capacity == 10:
        tile_n_vals = [128, 160, 192, 224, 256]
        tile_mn_cluster_vals = (
            [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
            # + [(128, tile_n, (2, 1)) for tile_n in tile_n_64_vals]
            + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
            + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
        )
        swap_ab_vals = [False, True]
        if epilogue in ["lse", "gated"]:
            swap_ab_vals = [False]
        max_swizzle_size_vals = [4, 8, 16]
        GemmConfigCls = partial(GemmConfig, pingpong=False)  # There's no pingpong on Sm100
        return [
            GemmConfigCls(
                tile_m=m,
                tile_n=n,
                cluster_m=cm,
                cluster_n=cn,
                swap_ab=sab,
                max_swizzle_size=ms,
            )
            for (m, n, (cm, cn)), sab, ms in itertools.product(
                tile_mn_cluster_vals, swap_ab_vals, max_swizzle_size_vals
            )
        ]
