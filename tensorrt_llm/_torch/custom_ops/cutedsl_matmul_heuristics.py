# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nvMatmulHeuristics (nvMMH) guidance for the CuTe DSL GEMM/BMM autotuner.

The autotuner normally compiles and benchmarks every tactic a CuTe DSL GEMM or
BMM runner enumerates. When ``autotuner_nvmmh_config`` installs an
:class:`~tensorrt_llm._torch.autotuner.NvMMHConfig` on the AutoTuner, runners
instead build their tactics from the kernel configurations nvMMH's analytical
model ranks highest for the problem. Every field nvMMH models (CTA tile,
cluster shape, split-K, rasterization order, swizzle) is taken as returned;
runners only sweep knobs nvMMH does not model.

Each runner describes what its kernel can execute through a validity check, so
nvMMH ranks implementable configurations only. Both the 0.1.0.x wheels (the
``nvidia-matmul-heuristics`` requirement) and newer libheuristics builds are
supported. If the library is missing or a query fails, :func:`query` returns
an empty list and the runner falls back to its full tactic space.
"""

import ctypes
import importlib.metadata
import inspect
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Hashable, List, NamedTuple, Optional, Tuple

from tensorrt_llm.logger import logger

try:
    import nvMatmulHeuristics as _nvmmh

    IS_NVMMH_AVAILABLE = True
except (ImportError, OSError):
    _nvmmh = None
    IS_NVMMH_AVAILABLE = False

# nvMMH precision strings name one data type per character (T = BF16,
# S = FP32, H = FP16, Q = FP8 E4M3, O = FP4 E2M1). Three characters are
# (A/B, compute, C/D); five are (A, B, C, compute, D).
BF16_PRECISION = "TST"
BF16_FP32_OUTPUT_PRECISION = "TSS"
BF16_FP8_OUTPUT_PRECISION = "TTQSQ"
FP8_PRECISION = "QQTST"
FP8_FP16_OUTPUT_PRECISION = "QQHSH"
FP4_PRECISION = "OOTST"

# Layouts of the CuTe DSL GEMMs: A [M, K] and B [N, K] are K-major ("TN") and
# C [M, N] is row-major. Swapping A and B computes C^T, a column-major [N, M].
LAYOUT_ROW_MAJOR_C = "TN_ROW_MAJOR"
LAYOUT_COL_MAJOR_C = "TN_COL_MAJOR"

# nvMMH backend split-K kinds, by name so callers need not import nvMMH.
SPLIT_K_IN_PLACE = "IN_PLACE"
SPLIT_K_OUT_OF_PLACE = "OUT_OF_PLACE"

# Configurations requested from nvMMH per kept tactic: several returned
# configurations differ only in fields a CuTe DSL kernel does not expose
# (e.g. the warp tile) and collapse onto one tactic.
QUERY_OVERSAMPLING = 8
MIN_QUERY_COUNT = 32


class KernelConfig(NamedTuple):
    """One nvMMH kernel configuration (a CUTLASS3 ``nvmmhKernelConfiguration``).

    ``cta`` is the per-CTA (M, N, K) tile. On Blackwell-class GPUs nvMMH runs
    2-CTA MMA instructions whenever ``cluster[0]`` is even (see
    :func:`pairs_ctas`). ``cta_order`` 0 rasterizes along M and 1 along N
    (see :func:`cta_order_to_raster`).
    ``score`` is the value nvMMH ranks configurations by (lower is better),
    comparable across problems that compute the same product (see
    :func:`query`).
    """

    cta: Tuple[int, int, int]
    cluster: Tuple[int, int]
    split_k: int
    swizzle: int
    cta_order: int
    stages: int
    warp: Tuple[int, int, int]
    instr: Tuple[int, int, int]
    score: float


@dataclass(frozen=True)
class Problem:
    """A GEMM/BMM problem in nvMMH terms (after any A/B swap)."""

    m: int
    n: int
    k: int
    batch: int
    precision: str
    layout: str = LAYOUT_ROW_MAJOR_C


@dataclass(frozen=True)
class Backend:
    """CUTLASS3 backend properties describing a CuTe DSL kernel family.

    Properties left as None keep nvMMH's CUTLASS3 defaults.
    """

    split_k_kind: Optional[str] = None
    supports_col_major_raster: Optional[bool] = None
    supports_odd_cluster_n: Optional[bool] = None
    # The kernel launches a 2-CTA cluster above two CTAs with a 2x1 fallback
    # cluster (preferred-cluster launch).
    supports_fallback_cluster: Optional[bool] = None
    cta_tile_m_div: Optional[int] = None
    cta_tile_n_div: Optional[int] = None


@lru_cache(maxsize=1)
def library_version() -> str:
    """Installed nvMMH distribution version (part of cache keys)."""
    try:
        return importlib.metadata.version("nvidia-matmul-heuristics")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(_nvmmh, "__version__", "unknown"))


def cta_order_to_raster(cta_order: int) -> str:
    """nvMMH CUTLASS3 CTA order -> CuTe DSL raster ("m": M-fastest).

    0 rasterizes along M and 1 along N, as in CUTLASS's own nvMMH provider for
    the same CUTLASS3 target (``raster_order``):
    https://github.com/NVIDIA/cutlass/blob/d4bbf728ca52ce23a549a299e229666e734ffb7c/python/cutlass_library/heuristics_provider.py#L222
    """
    return "n" if int(cta_order) else "m"


@lru_cache(maxsize=1)
def _pairs_only_cluster_m_2() -> bool:
    """0.1.0.x PyPI releases model 2-CTA MMA for cluster M == 2 only.

    An installation without version metadata is a 0.1.0.x release if its
    interface lacks the per-backend silicon-metric API newer builds add.
    """
    version = library_version()
    if version == "unknown":
        interface = _interface_class()
        old = not (
            hasattr(interface, "estimateSiliconMetricEx")
            and hasattr(interface, "setBackendPropertyValue")
        )
        logger.warning_once(
            "nvMatmulHeuristics reports no version; assuming "
            f"{'a 0.1.0.x release' if old else 'a libheuristics build'} "
            "from its API.",
            key="nvmmh_unknown_version",
        )
        return old
    match = re.fullmatch(r"0\.1\.0\.(\d+)", version)
    return match is not None and int(match.group(1)) < 10000


def pairs_ctas(config: KernelConfig) -> bool:
    """Whether nvMMH models this config with 2-CTA (CTA pair) MMA.

    nvMMH's CUTLASS3 target pairs CTAs along an even cluster M extent on
    Blackwell-class GPUs (only cluster M == 2 before libheuristics 0.2), so a
    CuTe DSL kernel realizes the config with ``use_2cta_instrs`` and an MMA
    tile of twice the per-CTA M.
    """
    cluster_m = int(config.cluster[0])
    if _pairs_only_cluster_m_2():
        return cluster_m == 2
    return cluster_m % 2 == 0


def _interface_class():
    """The interface class that manages discovery sets implicitly.

    0.1.0.x names it NvMatmulHeuristicsInterfaceEx; newer wheels moved the
    implicit discovery handling to NvMatmulHeuristicsInterface.
    """
    interface = _nvmmh.NvMatmulHeuristicsInterface
    if hasattr(interface, "_ensure_discovery_loaded"):
        return interface
    return _nvmmh.NvMatmulHeuristicsInterfaceEx


_interfaces: Dict[Tuple[int, str], object] = {}
_interfaces_lock = threading.Lock()


def _get_interface(precision: str):
    """One interface per (thread, precision); nvMMH handles are not shared."""
    key = (threading.get_ident(), precision)
    with _interfaces_lock:
        interface = _interfaces.get(key)
    if interface is not None:
        return interface
    cls = _interface_class()
    kwargs = dict(flags=_nvmmh.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING)
    if "precision" in inspect.signature(cls.__init__).parameters:
        kwargs["precision"] = precision
    interface = cls(_nvmmh.NvMatmulHeuristicsTarget.CUTLASS3, **kwargs)
    with _interfaces_lock:
        _interfaces[key] = interface
    return interface


def _make_problem(interface, problem: Problem):
    layout = getattr(_nvmmh.NvMatmulHeuristicsMatmulLayout, problem.layout)
    make = interface.makeNvMatmulHeuristicsProblem
    if "mnk" in inspect.signature(make).parameters:
        return make((problem.m, problem.n, problem.k), layout, batch_size=problem.batch)
    return make(problem.m, problem.n, problem.k, layout, batch_size=problem.batch)


def _set_property(interface, backend, name: str, value: int) -> None:
    prop = getattr(_nvmmh.NvMatmulHeuristicsBackendProperty, name)
    setter = getattr(interface, "setBackendPropertyValue", None)
    if setter is not None:
        setter(backend, prop, int(value))
        return
    # 0.1.0.x takes a raw value buffer.
    buffer = ctypes.c_int32(int(value))
    interface.setBackendValueProperty(backend, prop, ctypes.byref(buffer), ctypes.sizeof(buffer))


def _set_validity_check(interface, backend, accept: Callable[[object], bool]) -> None:
    """Register ``accept(raw_config) -> bool`` as the additional validity check."""
    kind = _nvmmh.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK
    setter = getattr(interface, "backendSetCallbackProperty", None)
    if setter is not None:
        setter(backend, kind, lambda config: int(bool(accept(config))))
        return
    # 0.1.0.x passes a configuration pointer and an unused problem pointer.
    interface.setBackendCallbackProperty(
        backend, kind, lambda config_ptr, _problem_ptr: int(bool(accept(config_ptr.contents)))
    )


def _configure_backend(interface, backend, spec: Backend) -> None:
    if spec.split_k_kind is not None:
        _set_property(
            interface,
            backend,
            "SPLIT_K_KIND",
            int(getattr(_nvmmh.NvMatmulHeuristicsSplitKKind, spec.split_k_kind)),
        )
    for name, value in (
        ("HAS_COL_MAJOR_RASTER", spec.supports_col_major_raster),
        ("SUPPORTS_ODD_CLUSTER_N", spec.supports_odd_cluster_n),
        ("SUPPORTS_FALLBACK_CLUSTER", spec.supports_fallback_cluster),
        ("CTA_TILE_M_DIV_REQUIREMENT", spec.cta_tile_m_div),
        ("CTA_TILE_N_DIV_REQUIREMENT", spec.cta_tile_n_div),
    ):
        if value is not None:
            _set_property(interface, backend, name, int(value))


def _to_kernel_config(raw, score: float) -> KernelConfig:
    return KernelConfig(
        cta=tuple(int(v) for v in raw.cta),
        cluster=(max(1, int(raw.cluster[0])), max(1, int(raw.cluster[1]))),
        split_k=max(1, int(raw.splitK)),
        swizzle=max(1, int(raw.gridSwizzle)),
        cta_order=int(raw.ctaOrder),
        stages=int(raw.loadStages),
        warp=tuple(int(v) for v in raw.warp),
        instr=tuple(int(v) for v in raw.instr),
        score=float(score),
    )


def _ranking_score(interface, backend, nv_problem, raw, precision: str, fallback: float) -> float:
    """The score nvMMH ranks ``raw`` by, with this backend's properties.

    Libraries with ``RANKING_SCORE`` report it directly; older libheuristics
    builds approximate it with ``RUNTIME_RELATIVE_FAST_S`` (their ranking
    metric without the Rubin L2 simulation). 0.1.0.x has no per-backend
    estimate: ``fallback`` is the wrapper's ``runtime`` field, a default-backend
    absolute runtime.
    """
    estimate = getattr(interface, "estimateSiliconMetricEx", None)
    if estimate is None or not hasattr(interface, "setBackendPropertyValue"):
        return fallback
    metrics = _nvmmh.NvMatmulHeuristicsSiliconMetric
    metric = getattr(metrics, "RANKING_SCORE", metrics.RUNTIME_RELATIVE_FAST_S)
    try:
        value = estimate(precision, backend, nv_problem, raw, int(metric))
    except Exception:  # noqa: BLE001 - only orders merged rankings
        return fallback
    return float(value) if value and value > 0 else fallback


def _query_uncached(
    problem: Problem, count: int, backend_spec: Backend, is_valid: Callable[[KernelConfig], bool]
) -> List[KernelConfig]:
    interface = _get_interface(problem.precision)
    backend = interface.createBackend(_nvmmh.NvMatmulHeuristicsTarget.CUTLASS3)
    try:
        _configure_backend(interface, backend, backend_spec)
        _set_validity_check(interface, backend, lambda raw: is_valid(_to_kernel_config(raw, 0.0)))
        nv_problem = _make_problem(interface, problem)
        configs = interface.getEx(nv_problem, int(count), backend, problem.precision)
        ranked = []
        for entry in configs:
            raw = entry["nvmmhKernelConfiguration"]
            score = _ranking_score(
                interface, backend, nv_problem, raw, problem.precision, entry.get("runtime", 0.0)
            )
            config = _to_kernel_config(raw, score)
            # Wrappers that drop the callback's verdict still return only
            # nvMMH-ranked configs; re-check them here.
            if is_valid(config):
                ranked.append(config)
    finally:
        interface.destroyBackend(backend)
    return ranked


_query_cache: Dict[Hashable, List[KernelConfig]] = {}
_query_cache_lock = threading.Lock()


def query(
    problem: Problem,
    count: int,
    is_valid: Callable[[KernelConfig], bool],
    backend: Backend = Backend(),
    cache_key: Optional[Hashable] = None,
) -> List[KernelConfig]:
    """Return up to ``count`` valid nvMMH configs in nvMMH's ranking order.

    nvMMH's own order is kept. Each config carries its ranking ``score``, so
    rankings of several problems computing the same product (e.g. both A/B
    orientations) can be merged consistently with that order.
    ``cache_key`` identifies the validity check; results are memoized per
    (problem, count, backend, cache_key) when it is given. Failures are
    logged once and yield an empty list.
    """
    if not IS_NVMMH_AVAILABLE:
        return []
    key = None
    if cache_key is not None:
        key = (problem, int(count), backend, cache_key)
        with _query_cache_lock:
            cached = _query_cache.get(key)
        if cached is not None:
            return list(cached)
    try:
        ranked = _query_uncached(problem, count, backend, is_valid)
    except Exception as e:  # noqa: BLE001 - guidance must never break tuning
        logger.warning_once(
            f"[nvMatmulHeuristics] Query failed for {problem} with nvMMH "
            f"{library_version()}: {e}. Falling back to the full tactic list.",
            key="nvmmh_query_failure",
        )
        return []
    if key is not None:
        with _query_cache_lock:
            _query_cache[key] = list(ranked)
    return ranked


def query_count(max_tactics: int) -> int:
    """How many configs to request so ``max_tactics`` distinct tactics remain."""
    return max(MIN_QUERY_COUNT, QUERY_OVERSAMPLING * int(max_tactics))


def select_tactics(ranked: List[Tuple[KernelConfig, List[tuple]]], max_tactics: int) -> List[tuple]:
    """Expand the top ``max_tactics`` nvMMH configs into runner tactics.

    ``ranked`` pairs each config (nvMMH order) with the tactics realizing it:
    one per combination of the knobs nvMMH does not model. Configs that map
    onto an already kept realization (e.g. differing only in warp tile) do not
    count against ``max_tactics``.
    """
    selected: List[tuple] = []
    seen = set()
    kept = 0
    for _, tactics in ranked:
        fresh = [t for t in tactics if t not in seen]
        if not fresh:
            continue
        selected.extend(fresh)
        seen.update(fresh)
        kept += 1
        if kept >= max_tactics:
            break
    return selected
