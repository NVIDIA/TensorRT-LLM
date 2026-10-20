# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compile-time search space for the dense PrimsTS GEMM.

Each tactic is one compiled module. The profiled list is capped so a single
M bucket does not compile the full product of cluster, tile, stages, CLC,
epilogue warps, and TMA store.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .config import PrimsTsGemmConfig

# Kernel pipeline defaults. Kept here so tactic enumeration does not import
# the CuTe module. _load_kernel applies the same numbers.
_DEFAULT_AB_STAGES = 6
_FUSED_QKNORM_AB_STAGES = 4
_NVFP4_AB_STAGES = 5
_STAGE_CHOICES = (5, 6)
_CLUSTERS = ((2, 2), (4, 4))
MAX_PROFILED_TACTICS = 1024

# cluster_m, cluster_n, tile_n, tile_k, tmem_overlap, ab_stages, mma_k,
# use_clc, epilogue_warps, use_tma_store
Tactic = tuple[int, int, int, int, bool, int, int, bool, int, bool]


def derived_ab_stages(
    operand_format: str,
    epilogue: str,
    mma_k: int,
    tile_n: int,
    tile_k: int,
) -> int:
    """Return the stage count the kernel uses when ``ab_stages`` is unset."""
    if epilogue == "qkv_qknorm_rope":
        stages = _FUSED_QKNORM_AB_STAGES
    elif operand_format == "nvfp4_e2m1":
        stages = _NVFP4_AB_STAGES
    else:
        stages = _DEFAULT_AB_STAGES
    if mma_k == 96 and tile_k == 768:
        return 1 if tile_n == 256 else 2
    return stages


def stage_candidates(derived: int) -> tuple[int, ...]:
    """Derived count plus the nearest smaller and larger known counts."""
    if derived not in _STAGE_CHOICES:
        return (derived,)
    index = _STAGE_CHOICES.index(derived)
    choices = [derived]
    if index > 0:
        choices.append(_STAGE_CHOICES[index - 1])
    if index + 1 < len(_STAGE_CHOICES):
        choices.append(_STAGE_CHOICES[index + 1])
    return tuple(choices)


def fallback_cluster(cluster: tuple[int, int, int]) -> Optional[tuple[int, int, int]]:
    """Smaller cluster used when the preferred shape does not fit.

    ``(2, 1, 1)`` has no smaller fallback. Any wider cluster uses it, and
    that fallback divides every preferred shape we search.
    """
    cluster_m, cluster_n, _cluster_k = cluster
    if cluster_m > 2 or cluster_n > 1:
        return (2, 1, 1)
    return None


def _derived_warps(epilogue: str, overlap: bool) -> int:
    if epilogue != "linear" or overlap:
        return 8
    return 4


def default_tactic(
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
) -> Tactic:
    del arch, output_format
    tile_k = 256 if operand_format == "nvfp4_e2m1" else 128
    tile_n = 256
    mma_k = 64
    overlap = False
    return (
        2,
        2,
        tile_n,
        tile_k,
        overlap,
        derived_ab_stages(operand_format, epilogue, mma_k, tile_n, tile_k),
        mma_k,
        True,
        _derived_warps(epilogue, overlap),
        False,
    )


def tactic_is_legal(
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
    tactic: Tactic,
) -> bool:
    (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    ) = tactic
    if cluster_m not in (2, 4) or cluster_n not in (1, 2, 4):
        return False
    if (
        not isinstance(use_clc, bool)
        or not isinstance(overlap, bool)
        or not isinstance(use_tma_store, bool)
    ):
        return False
    # FP4 output stores are packed and direct. TMEM overlap cannot share the
    # accumulator with a TMA output buffer.
    if use_tma_store and (output_format == "nvfp4_e2m1" or overlap):
        return False
    if warps not in (4, 8) or mma_k not in (64, 96):
        return False
    if epilogue != "linear" and warps != 8:
        return False
    if operand_format == "nvfp4_e2m1":
        if tile_n not in (128, 256):
            return False
        # 768 still launches from a pinned config. The search does not try it.
        if mma_k == 96:
            if arch != 103 or tile_k != 256:
                return False
        elif tile_k not in (256, 512):
            return False
    elif operand_format == "fp8_e4m3":
        if tile_n not in (32, 64, 128, 256) or tile_k not in (128, 256) or mma_k != 64:
            return False
    else:
        return False
    if overlap and not (
        operand_format == "nvfp4_e2m1"
        and (
            (epilogue == "linear" and output_format == "bf16")
            or (epilogue == "swiglu" and output_format in ("bf16", "nvfp4_e2m1"))
            or (
                epilogue == "qkv_qknorm_rope"
                and output_format == "bf16"
                and mma_k == 64
            )
        )
        and tile_n == 256
        and tile_k == 256
        and warps == 8
    ):
        return False
    # QKV's derived depth is 4, which is outside _STAGE_CHOICES. The candidate
    # list is the limit; an earlier membership test would drop that epilogue.
    derived = derived_ab_stages(operand_format, epilogue, mma_k, tile_n, tile_k)
    return stages in stage_candidates(derived)


def _tile_ns() -> tuple[int, ...]:
    return (256,)


def _mma_ks(arch: int, operand_format: str) -> tuple[int, ...]:
    if arch == 103 and operand_format == "nvfp4_e2m1":
        return (64, 96)
    return (64,)


def _tile_ks(operand_format: str, mma_k: int) -> tuple[int, ...]:
    if operand_format == "nvfp4_e2m1":
        return (256,)
    return (128,)


def _warps() -> tuple[int, ...]:
    return (4, 8)


def _replace(tactic: Tactic, **updates: object) -> Tactic:
    (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    ) = tactic
    values = {
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "overlap": overlap,
        "stages": stages,
        "mma_k": mma_k,
        "use_clc": use_clc,
        "warps": warps,
        "use_tma_store": use_tma_store,
    }
    values.update(updates)
    return (
        int(values["cluster_m"]),
        int(values["cluster_n"]),
        int(values["tile_n"]),
        int(values["tile_k"]),
        bool(values["overlap"]),
        int(values["stages"]),
        int(values["mma_k"]),
        bool(values["use_clc"]),
        int(values["warps"]),
        bool(values["use_tma_store"]),
    )


def _with_derived_stages(
    operand_format: str,
    epilogue: str,
    tactic: Tactic,
) -> Tactic:
    (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        _stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    ) = tactic
    stages = derived_ab_stages(operand_format, epilogue, mma_k, tile_n, tile_k)
    return (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    )


def _neighbors(
    operand_format: str,
    epilogue: str,
    base: Tactic,
) -> Iterator[Tactic]:
    """One-axis steps around the default, so the cap still covers every axis."""
    for cluster_m, cluster_n in _CLUSTERS:
        if (cluster_m, cluster_n) != (base[0], base[1]):
            yield _replace(base, cluster_m=cluster_m, cluster_n=cluster_n)
    for tile_n in _tile_ns():
        if tile_n != base[2]:
            yield _with_derived_stages(
                operand_format, epilogue, _replace(base, tile_n=tile_n, overlap=False)
            )
    for mma_k in (64, 96):
        if mma_k == base[6]:
            continue
        for tile_k in _tile_ks(operand_format, mma_k):
            candidate = _with_derived_stages(
                operand_format,
                epilogue,
                _replace(base, tile_k=tile_k, mma_k=mma_k, overlap=False),
            )
            yield candidate
    for tile_k in _tile_ks(operand_format, base[6]):
        if tile_k != base[3]:
            yield _with_derived_stages(
                operand_format, epilogue, _replace(base, tile_k=tile_k, overlap=False)
            )
    derived = derived_ab_stages(operand_format, epilogue, base[6], base[2], base[3])
    for stages in stage_candidates(derived):
        if stages != base[5]:
            yield _replace(base, stages=stages)
    yield _replace(base, use_clc=not base[7])
    for warps in _warps():
        if warps != base[8]:
            yield _replace(base, warps=warps, overlap=False)
    if epilogue in ("linear", "swiglu") or (
        epilogue == "qkv_qknorm_rope" and base[6] == 64
    ):
        yield _replace(
            base, overlap=True, warps=8, tile_n=256, tile_k=base[3], use_tma_store=False
        )
    yield _replace(base, use_tma_store=not base[9], overlap=False)


def _product(
    arch: int,
    operand_format: str,
    epilogue: str,
) -> Iterator[Tactic]:
    for cluster_m, cluster_n in _CLUSTERS:
        for tile_n in _tile_ns():
            for mma_k in _mma_ks(arch, operand_format):
                for tile_k in _tile_ks(operand_format, mma_k):
                    derived = derived_ab_stages(
                        operand_format, epilogue, mma_k, tile_n, tile_k
                    )
                    for stages in stage_candidates(derived):
                        # for use_clc in (True, False):
                        for use_clc in (True,):
                            for warps in _warps():
                                for overlap in (False, True):
                                    for use_tma_store in (False, True):
                                        yield (
                                            cluster_m,
                                            cluster_n,
                                            tile_n,
                                            tile_k,
                                            overlap,
                                            stages,
                                            mma_k,
                                            use_clc,
                                            warps,
                                            use_tma_store,
                                        )


def legal_tactics(
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
) -> list[Tactic]:
    """Legal tactics, default first, at most ``MAX_PROFILED_TACTICS``."""
    default = default_tactic(arch, operand_format, output_format, epilogue)
    selected = [default]
    seen = {default}

    def add(tactic: Tactic) -> None:
        if len(selected) >= MAX_PROFILED_TACTICS or tactic in seen:
            return
        if tactic_is_legal(arch, operand_format, output_format, epilogue, tactic):
            selected.append(tactic)
            seen.add(tactic)

    for tactic in _neighbors(operand_format, epilogue, default):
        add(tactic)
    for tactic in _product(arch, operand_format, epilogue):
        if len(selected) >= MAX_PROFILED_TACTICS:
            break
        add(tactic)
    return selected


def config_from_tactic(
    *,
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
    has_bias: bool,
    head_dim: Optional[int],
    is_neox: Optional[bool],
    has_qkv_scale: bool,
    tactic: Tactic,
) -> PrimsTsGemmConfig:
    """Build a config. Default warp and stage counts stay ``None``."""
    (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    ) = tactic
    derived_stages = derived_ab_stages(operand_format, epilogue, mma_k, tile_n, tile_k)
    derived_warps = _derived_warps(epilogue, overlap)
    return PrimsTsGemmConfig(
        arch,
        operand_format,  # type: ignore[arg-type]
        output_format,  # type: ignore[arg-type]
        epilogue,  # type: ignore[arg-type]
        has_bias,
        head_dim,
        is_neox,
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_shape=(cluster_m, cluster_n, 1),
        scheduler="clc_dynamic" if use_clc else "static",
        has_qkv_scale=has_qkv_scale,
        nvfp4_mma_k=mma_k,
        epilogue_warps=None if warps == derived_warps else warps,
        tmem_overlap=overlap,
        ab_stages=None if stages == derived_stages else stages,
        use_tma_store=use_tma_store,
    )


def fallback_config(
    *,
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
    has_bias: bool,
    head_dim: Optional[int],
    is_neox: Optional[bool],
    has_qkv_scale: bool,
) -> PrimsTsGemmConfig:
    """Today's default: tile N 256, CLC on, overlap off, ``mma_k`` 64."""
    return config_from_tactic(
        arch=arch,
        operand_format=operand_format,
        output_format=output_format,
        epilogue=epilogue,
        has_bias=has_bias,
        head_dim=head_dim,
        is_neox=is_neox,
        has_qkv_scale=has_qkv_scale,
        tactic=default_tactic(arch, operand_format, output_format, epilogue),
    )
