# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter around NVIDIA Matmul Heuristics (nvMatmulHeuristics) for the CuTe DSL
GEMM autotuner.

The autotuner enumerates every valid CuTe DSL tactic and then compiles +
benchmarks all of them, which dominates autotuning wall-time. This module uses
nvMatmulHeuristics' analytical performance model to rank the candidate
(CTA-tile, cluster) configurations so the runner can keep only the top-K before
profiling.

Everything here is best-effort and opt-in: if the library is missing, the
problem is unsupported, or the query fails, the public helpers return an empty
ranking so callers fall back to the full tactic list and never regress.
"""

import os
import re
from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple

from tensorrt_llm.logger import logger

try:
    import nvMatmulHeuristics as _nvmmh

    IS_NVMMH_AVAILABLE = True
except ImportError:
    _nvmmh = None
    IS_NVMMH_AVAILABLE = False

# NVFP4 dense GEMM precision/layout for nvMatmulHeuristics, fixed to the
# CuteDSLNVFP4BlackwellRunner kernel: A/B are Float4E2M1FN (fp4), the output is
# BFloat16, and A/B are K-major (see the can_implement call in
# cute_dsl_custom_ops.py). "F4F4T" is the cuBLAS-style token (fp4 A, fp4 B,
# bf16 output; bf16->T) and TN_ROW_MAJOR is the K-major A/B layout enum name on
# NvMatmulHeuristicsMatmulLayout. The whole path is best-effort: if the token is
# ever rejected by the installed wheel, rank_configs degrades to an empty
# ranking and the caller falls back to the full tactic sweep.
NVFP4_PRECISION = "F4F4T"
NVFP4_LAYOUT = "TN_ROW_MAJOR"

CtaCluster = Tuple[Tuple[int, int], Tuple[int, int]]

# The knobs a caller may let nvMatmulHeuristics drive (instead of sweeping the
# runner's candidate list). A field named here is supplied by the model; a field
# not named is swept. "tile" and "cluster" are a coupled pair (the 2-SM encoding
# maps the joint (cta, cluster)), so selecting either drives both jointly.
# "swizzle" (swizzle_factor) and "cta_order" (rasterization order) are per-knob.
NVMMH_OPTIONAL_FIELDS = frozenset({"tile", "cluster", "swizzle", "cta_order"})

# Accepted spellings that normalize onto a canonical field name.
_NVMMH_FIELD_ALIASES = {"cta_tile": "tile", "mma_tiler": "tile"}


class HeuristicConfig(NamedTuple):
    """One nvMatmulHeuristics candidate, reduced to the knobs we can map to a
    CuTe DSL NVFP4 tactic. ``swizzle_factor``/``cta_order`` default to the
    kernel's neutral values (no swizzle, M-major raster) when absent."""

    cta: Tuple[int, int]
    cluster: Tuple[int, int]
    swizzle_factor: int = 1
    cta_order: int = 0


def nvmmh_fields() -> set:
    """Which knobs nvMatmulHeuristics drives (vs. the runner sweeping them).

    Read from TRTLLM_CUTEDSL_NVMMH_FIELDS (comma-separated). Recognized tokens
    are ``NVMMH_OPTIONAL_FIELDS`` (plus the ``cta_tile``/``mma_tiler`` aliases
    for "tile"); unknown tokens are warned about once and ignored. Because the
    2-SM encoding maps the joint (cta, cluster), "tile" and "cluster" are
    coupled -- naming either drives both. Default: "tile,cluster" (today's
    behavior: heuristic tile+cluster, swept swizzle/cta_order).
    """
    raw = os.environ.get("TRTLLM_CUTEDSL_NVMMH_FIELDS", "tile,cluster")
    fields = set()
    for tok in (f.strip().lower() for f in raw.split(",")):
        if not tok:
            continue
        canonical = _NVMMH_FIELD_ALIASES.get(tok, tok)
        if canonical not in NVMMH_OPTIONAL_FIELDS:
            logger.warning_once(
                f"[nvMatmulHeuristics] Ignoring unknown field '{tok}' in "
                f"TRTLLM_CUTEDSL_NVMMH_FIELDS. Recognized: "
                f"{sorted(NVMMH_OPTIONAL_FIELDS)} (aliases: "
                f"{sorted(_NVMMH_FIELD_ALIASES)}).",
                key=f"nvmmh_unknown_field_{tok}",
            )
            continue
        fields.add(canonical)
    # tile/cluster are a coupled pair: naming either drives both.
    if fields & {"tile", "cluster"}:
        fields.update({"tile", "cluster"})
    return fields


def nvmmh_enabled() -> bool:
    """Master opt-in switch for heuristic tactic pruning."""
    return IS_NVMMH_AVAILABLE and os.environ.get("TRTLLM_CUTEDSL_NVMMH_ENABLE", "0") == "1"


def nvmmh_enabled_for_nvfp4() -> bool:
    """Whether heuristic pruning applies to the NVFP4 dense GEMM. Gated solely
    by the master switch (TRTLLM_CUTEDSL_NVMMH_ENABLE)."""
    return nvmmh_enabled()


def nvmmh_max_tactics() -> int:
    """Number of heuristic-ranked (tile, cluster) configs to keep per problem.

    Mirrors TorchInductor's nvgemm_max_profiling_configs. Both use_prefetch
    variants of a kept config are profiled on top of this (prefetch is not
    modeled by the heuristic), so the final tactic count is up to ~2x this.
    """
    try:
        return max(1, int(os.environ.get("TRTLLM_CUTEDSL_NVMMH_MAX_TACTICS", "5")))
    except ValueError:
        return 5


@lru_cache(maxsize=None)
def _get_interface(precision: str):
    """Build and cache an interface for a given precision (None on failure)."""
    if not IS_NVMMH_AVAILABLE:
        return None
    try:
        return _nvmmh.NvMatmulHeuristicsInterface(
            _nvmmh.NvMatmulHeuristicsTarget.CUTLASS3,
            precision=precision,
            flags=_nvmmh.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
        )
    except Exception as e:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning_once(
            f"[nvMatmulHeuristics] Failed to build interface for "
            f"precision={precision}: {e}. Falling back to full tactic list.",
            key="nvmmh_interface_init_failure",
        )
        return None


def _get_layout(name: str):
    try:
        return getattr(_nvmmh.NvMatmulHeuristicsMatmulLayout, name)
    except AttributeError:
        logger.warning_once(
            f"[nvMatmulHeuristics] Unknown layout '{name}'. Falling back to full tactic list.",
            key="nvmmh_unknown_layout",
        )
        return None


def _as_int_pair(values) -> Optional[Tuple[int, int]]:
    try:
        nums = [int(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(nums) < 2:
        return None
    return (nums[0], nums[1])


def _extract_int(config, kernel, dict_key: str, attr: str, str_re: str, default: int) -> int:
    """Read one scalar field, tolerant of dict / object / printable-string."""
    if isinstance(config, dict) and dict_key in config:
        try:
            return int(config[dict_key])
        except (TypeError, ValueError):
            pass
    if hasattr(kernel, attr):
        try:
            return int(getattr(kernel, attr))
        except (TypeError, ValueError):
            pass
    m = re.search(str_re, kernel if isinstance(kernel, str) else str(kernel))
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            pass
    return default


def _extract_config(config) -> Optional[HeuristicConfig]:
    """Pull (cta, cluster, swizzle_factor, cta_order) out of one heuristic
    config. Tolerant of the several shapes the config may take across versions:
    a dict with flat fields, a dict whose ``kernel`` is an object with
    attributes, or a dict whose ``kernel`` is a printable string.
    """
    kernel = config.get("kernel") if isinstance(config, dict) else config
    cta = cluster = None

    # Form 1: flat dict fields (DKG getEx-style wrapper).
    if isinstance(config, dict) and "cta_tile_m" in config:
        cta = _as_int_pair((config["cta_tile_m"], config["cta_tile_n"]))
        cluster = _as_int_pair((config.get("cluster_m", 1), config.get("cluster_n", 1)))

    # Form 2: kernel object with attributes (public wheel's GemmConfig uses
    # cluster_m/cluster_n; the DKG internal wrapper used cga_m/cga_n).
    if (cta is None or cluster is None) and hasattr(kernel, "cta_tile_m"):
        cta = _as_int_pair((kernel.cta_tile_m, kernel.cta_tile_n))
        cluster_m = getattr(kernel, "cluster_m", getattr(kernel, "cga_m", 1))
        cluster_n = getattr(kernel, "cluster_n", getattr(kernel, "cga_n", 1))
        cluster = _as_int_pair((cluster_m, cluster_n))

    # Form 3: printable string like "... cta(128 16 128) ... cluster(2 1) ...".
    if cta is None or cluster is None:
        text = kernel if isinstance(kernel, str) else str(kernel)
        cta_m = re.search(r"cta\(\s*([\d\s]+?)\)", text)
        cluster_m = re.search(r"cluster\(\s*([\d\s]+?)\)", text)
        if cta_m:
            cta = _as_int_pair(cta_m.group(1).split())
            cluster = _as_int_pair(cluster_m.group(1).split()) if cluster_m else (1, 1)

    if cta is None or cluster is None:
        return None

    # swizzle_factor -> "swizz(N)"; cta_order -> "ctaOrder(N)". Default to the
    # kernel's neutral values (no swizzle, M-major raster) when the field is
    # absent so an unselected knob never changes behavior.
    swizzle = _extract_int(
        config, kernel, "swizzle_factor", "swizzle_factor", r"swizz\(\s*(\d+)\)", 1
    )
    cta_order = _extract_int(config, kernel, "cta_order", "cta_order", r"ctaOrder\(\s*(\d+)\)", 0)
    return HeuristicConfig(cta, cluster, max(1, swizzle), cta_order)


def rank_configs(m: int, n: int, k: int, precision: str, count: int) -> List[HeuristicConfig]:
    """Return up to ``count`` HeuristicConfig entries ranked best-first.

    Returns an empty list if heuristics are unavailable or the query fails, so
    callers can fall back to their full tactic list.
    """
    interface = _get_interface(precision)
    if interface is None:
        return []
    layout = _get_layout(NVFP4_LAYOUT)
    if layout is None:
        return []
    try:
        # hw=None targets the current GPU.
        try:
            interface.loadInternalDiscoverySet(layout, None)
        except Exception:  # noqa: BLE001 - discovery data is optional
            pass
        configs = interface.get_with_mnk(int(m), int(n), int(k), layout, int(count), None)
    except Exception as e:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning_once(
            f"[nvMatmulHeuristics] Query failed for "
            f"precision={precision}, mnk=({m},{n},{k}): {e}. "
            f"Falling back to full tactic list.",
            key="nvmmh_query_failure",
        )
        return []

    if not configs:
        return []

    # Sort by estimated runtime when present; the API may already be ranked.
    def _runtime(c):
        return c.get("runtime", float("inf")) if isinstance(c, dict) else float("inf")

    ordered = sorted(configs, key=_runtime)
    ranked: List[HeuristicConfig] = []
    for c in ordered:
        cfg = _extract_config(c)
        if cfg is not None and cfg not in ranked:
            ranked.append(cfg)
    return ranked


def rank_tile_cluster_configs(
    m: int, n: int, k: int, precision: str, count: int
) -> List[CtaCluster]:
    """Back-compat helper: (cta, cluster) pairs only, ranked best-first."""
    return [(c.cta, c.cluster) for c in rank_configs(m, n, k, precision, count)]
