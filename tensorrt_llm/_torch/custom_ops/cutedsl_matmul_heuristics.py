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

import ctypes
import importlib.metadata
import re
from functools import lru_cache
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

from tensorrt_llm.logger import logger

try:
    import nvMatmulHeuristics as _nvmmh

    IS_NVMMH_AVAILABLE = True
except (ImportError, OSError):
    _nvmmh = None
    IS_NVMMH_AVAILABLE = False


@lru_cache(maxsize=1)
def nvmmh_library_version() -> str:
    """Return the installed nvMatmulHeuristics distribution version.

    Rankings change between wheels, so callers include this in persistent
    tactic-search cache keys.
    """
    try:
        return importlib.metadata.version("nvidia-matmul-heuristics")
    except importlib.metadata.PackageNotFoundError:
        return getattr(_nvmmh, "__version__", "unknown")


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

# Dense BF16 GEMM uses BF16 A/B with FP32 accumulation. The output may be BF16
# or FP32, which is the third character in nvMatmulHeuristics' three-character
# precision form. The runner stores A as [M, K] and weight as [N, K] while
# consuming weight^T, matching the TN row-major layout used by the NVFP4 path.
BF16_PRECISION = "TST"
BF16_FP32_OUTPUT_PRECISION = "TSS"
BF16_LAYOUT = "TN_ROW_MAJOR"

# Dense FP8 GEMM uses E4M3 A/B with FP32 accumulation and BF16 or FP16 output.
# nvMatmulHeuristics uses its five-character FP8 form: A, B, C, compute, D.
# The installed Rubin wheel accepts both tokens below. A/B are K-major, matching
# the physical [M, K] activation and [N, K] weight tensors.
FP8_BF16_OUTPUT_PRECISION = "QQTST"
FP8_FP16_OUTPUT_PRECISION = "QQHSH"
FP8_LAYOUT = "TN_ROW_MAJOR"

# Rubin's native split-K scheduler currently requires more than 20 K tiles per
# split. That lower bound hides useful split factors for the relatively small-K
# CuTe DSL kernels used by TRT-LLM. Keep the Rubin-specific override here so
# every split-capable CuTe DSL runner applies the same policy. This is an
# admission rule only: CUPTI still chooses the fastest admitted factor.
SM107_NVMMH_MIN_K_TILES_PER_SPLIT = 6


def is_sm107_nvmmh_split_k_eligible(k: int, cta_k: int, split_k: int) -> bool:
    """Whether an SM107 split-K candidate clears TRT-LLM's K-tile bound.

    This mirrors nvMatmulHeuristics' strict ``tiles_per_split > lower_bound``
    gate while replacing its hard-coded lower bound for SM107. Split-K 1 is
    always retained so the heuristic path cannot remove the unsplit fallback.
    """
    k, cta_k, split_k = int(k), int(cta_k), int(split_k)
    if k <= 0 or cta_k <= 0 or split_k <= 0:
        return False
    if split_k == 1:
        return True
    tiles_per_split = (k + cta_k * split_k - 1) // (cta_k * split_k)
    return tiles_per_split > SM107_NVMMH_MIN_K_TILES_PER_SPLIT


# Canonical query mode used by kernels that reduce split-K partials directly
# into the destination tensor.  Kept as an adapter-owned string so callers do
# not need to import nvMatmulHeuristics enums (or even have the wheel installed).
IN_PLACE_SPLIT_K_KIND = "in_place"

_BF16_KNOWN_GOOD_PREFERRED_TACTIC = (
    "preferred_cluster",
    True,
    (256, 256),
    (4, 2),
    (2, 1),
    0,
)

CtaCluster = Tuple[Tuple[int, int], Tuple[int, int]]


class HeuristicConfig(NamedTuple):
    """One model candidate reduced to knobs understood by CuTe DSL runners.

    Optional fields use neutral defaults when absent. ``split_k`` is appended
    to preserve positional compatibility with the original four-field tuple.
    """

    cta: Tuple[int, int]
    cluster: Tuple[int, int]
    swizzle_factor: int = 1
    cta_order: int = 0
    split_k: int = 1


def nvmmh_cta_order_to_cute_raster(cta_order: int) -> str:
    """Translate CUTLASS3 nvMMH CTA order to CuTe's raster name.

    Order 0 is ``AlongM`` and order 1 is ``AlongN``, as in CUTLASS's nvMMH
    provider (https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/python/cutlass_library/heuristics_provider.py#L222)
    and nvMMH's own CUTLASS swizzler model. Keep this conversion at the
    adapter boundary because the public nvMMH header's row/column wording
    differs from this CUTLASS3 mapping.
    """
    if cta_order == 0:
        return "m"
    if cta_order == 1:
        return "n"
    raise ValueError(f"Unsupported nvMMH CTA order {cta_order!r}; expected 0 or 1")


def cute_raster_to_nvmmh_cta_order(raster_along: str) -> int:
    """Translate CuTe's M/N-fast raster name to CUTLASS3 nvMMH CTA order."""
    if raster_along == "m":
        return 0
    if raster_along == "n":
        return 1
    raise ValueError(f"Unsupported CuTe raster {raster_along!r}; expected 'm' or 'n'")


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


def _extract_split_k(config, kernel) -> int:
    """Read split-K across the public, raw-ctypes, and printable API forms."""
    candidates = []
    if isinstance(config, dict):
        candidates.extend(config.get(key) for key in ("split_k", "splitK", "split_k_slices"))
    for attr in ("split_k", "splitK", "split_k_slices"):
        if hasattr(kernel, attr):
            candidates.append(getattr(kernel, attr))
    if isinstance(config, dict):
        raw = config.get("nvmmhKernelConfiguration")
        if raw is not None and hasattr(raw, "splitK"):
            candidates.append(raw.splitK)
    for value in candidates:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value >= 1:
            return value

    text = kernel if isinstance(kernel, str) else str(kernel)
    match = re.search(r"\bsplitK\(\s*(\d+)\s*\)", text)
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            pass
    return 1


def _extract_config(config) -> Optional[HeuristicConfig]:
    """Extract runner-visible fields from one heuristic configuration.

    Tolerates flat dictionaries, public ``GemmConfig`` objects, raw ctypes
    wrappers, and printable kernel strings used across package versions.
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
    # kernel's neutral values (no swizzle, along-M raster) when the field is
    # absent so an unselected knob never changes behavior.
    swizzle = _extract_int(
        config, kernel, "swizzle_factor", "swizzle_factor", r"swizz\(\s*(\d+)\)", 1
    )
    cta_order = _extract_int(config, kernel, "cta_order", "cta_order", r"ctaOrder\(\s*(\d+)\)", 0)
    split_k = _extract_split_k(config, kernel)
    return HeuristicConfig(cta, cluster, max(1, swizzle), cta_order, max(1, split_k))


def _set_tile_validity_check(interface, backend, tiles: frozenset) -> None:
    """Restrict a backend's search to the per-CTA tiles a runner can launch.

    NVMMH calls the check for every candidate it would otherwise rank. Without
    it the model can spend its whole ranking on tiles the CuTe DSL kernel does
    not implement (e.g. CTA_M=64 on block-scaled Rubin GEMMs), and the caller
    then falls back to the full sweep. Clusters stay unrestricted: runners
    either materialize model clusters or substitute legal ones per tile.
    """

    def admit(config) -> int:
        return int((int(config.cta[0]), int(config.cta[1])) in tiles)

    def admit_pointer(config_ptr, _problem_ptr) -> int:
        return admit(config_ptr.contents)

    set_callback = getattr(interface, "backendSetCallbackProperty", None)
    callback = admit
    if set_callback is None:
        # nvMatmulHeuristics 0.1.0.27 spells it setBackendCallbackProperty and
        # passes a configuration pointer plus an unused problem pointer.
        set_callback = getattr(interface, "setBackendCallbackProperty", None)
        callback = admit_pointer
    if set_callback is None:
        raise AttributeError("nvMatmulHeuristics interface cannot set a validity callback")
    set_callback(
        backend,
        _nvmmh.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
        callback,
    )


def rank_configs(
    m: int,
    n: int,
    k: int,
    precision: str,
    count: int,
    layout_name: str = NVFP4_LAYOUT,
    split_k_kind: Optional[str] = None,
    batch_size: int = 1,
    tiles: Optional[Iterable[Tuple[int, int]]] = None,
) -> List[HeuristicConfig]:
    """Return up to ``count`` HeuristicConfig entries ranked best-first.

    Each entry retains its own model-recommended split-K value. Returns an
    empty list if heuristics are unavailable or the query fails, so callers
    can fall back to their full tactic list. In-place queries preserve the
    backend's native split-K search range.

    ``tiles`` lists the per-CTA (M, N) tiles the caller can launch. When
    given, the model only ranks configurations with one of those tiles.
    """
    tiles = None if tiles is None else frozenset((int(m_), int(n_)) for m_, n_ in tiles)
    if tiles is not None and not tiles:
        return []
    interface = _get_interface(precision)
    if interface is None:
        return []
    layout = _get_layout(layout_name)
    if layout is None:
        return []
    try:
        # hw=None targets the current GPU.
        try:
            interface.loadInternalDiscoverySet(layout, None)
        except Exception as e:  # noqa: BLE001 - discovery data is optional
            logger.debug(f"[nvMatmulHeuristics] loadInternalDiscoverySet skipped: {e}")

        def _method(*names):
            """Resolve a supported method name across NVMMH Python API versions."""
            for name in names:
                method = getattr(interface, name, None)
                if method is not None:
                    return method
            raise AttributeError(f"nvMatmulHeuristics interface has none of {names!r}")

        if split_k_kind is None and batch_size == 1 and tiles is None:
            get_with_mnk = getattr(interface, "get_with_mnk", None)
            if get_with_mnk is not None:
                configs = get_with_mnk(int(m), int(n), int(k), layout, int(count), None)
            else:
                configs = _method("getWithMNK")((int(m), int(n), int(k)), layout, int(count))
        elif split_k_kind is None:
            backend = _method("createBackend", "create_backend")(
                _nvmmh.NvMatmulHeuristicsTarget.CUTLASS3
            )
            try:
                if tiles is not None:
                    _set_tile_validity_check(interface, backend, tiles)
                make_problem = _method(
                    "makeNvMatmulHeuristicsProblem", "make_nv_matmul_heuristics_problem"
                )
                try:
                    problem = make_problem(
                        (int(m), int(n), int(k)), layout, batch_size=int(batch_size)
                    )
                except TypeError:
                    problem = make_problem(
                        int(m), int(n), int(k), layout, batch_size=int(batch_size)
                    )
                configs = _method("getEx", "get_ex")(problem, int(count), backend)
            finally:
                _method("destroyBackend", "destroy_backend")(backend)
        else:
            if split_k_kind != IN_PLACE_SPLIT_K_KIND:
                raise ValueError(f"Unsupported nvMatmulHeuristics split-K kind {split_k_kind!r}")

            backend = None
            try:
                create_backend = _method("createBackend", "create_backend")
                backend = create_backend(_nvmmh.NvMatmulHeuristicsTarget.CUTLASS3)
                property_name = _nvmmh.NvMatmulHeuristicsBackendProperty.SPLIT_K_KIND
                property_value = int(_nvmmh.NvMatmulHeuristicsSplitKKind.IN_PLACE)
                set_backend_property = getattr(
                    interface,
                    "setBackendPropertyValue",
                    getattr(interface, "set_backend_property_value", None),
                )
                if set_backend_property is not None:
                    # Newer wrappers accept the scalar value directly.
                    set_backend_property(backend, property_name, property_value)
                else:
                    # The wheel bundled in the Rubin container exposes the
                    # original raw-buffer InterfaceEx spelling.
                    set_backend_property = _method(
                        "setBackendValueProperty", "set_backend_value_property"
                    )
                    property_buffer = ctypes.c_int32(property_value)
                    set_backend_property(
                        backend,
                        property_name,
                        ctypes.byref(property_buffer),
                        ctypes.sizeof(property_buffer),
                    )
                if tiles is not None:
                    _set_tile_validity_check(interface, backend, tiles)
                make_problem = _method(
                    "makeNvMatmulHeuristicsProblem", "make_nv_matmul_heuristics_problem"
                )
                try:
                    problem = make_problem(
                        (int(m), int(n), int(k)), layout, batch_size=int(batch_size)
                    )
                except TypeError:
                    # Older wheels take M, N, and K as separate arguments.
                    problem = make_problem(
                        int(m), int(n), int(k), layout, batch_size=int(batch_size)
                    )
                get_ex = _method("getEx", "get_ex")
                configs = get_ex(problem, int(count), backend)
            finally:
                if backend is not None:
                    _method("destroyBackend", "destroy_backend")(backend)
    except Exception as e:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning_once(
            f"[nvMatmulHeuristics] Query failed for "
            f"precision={precision}, mnk=({m},{n},{k}), batch_size={batch_size}, "
            f"tiles={None if tiles is None else sorted(tiles)}, "
            f"split_k_kind={split_k_kind}: {e}. "
            f"Falling back to full tactic list.",
            key="nvmmh_query_failure",
        )
        return []

    if not configs:
        return []

    # Sort by estimated runtime when present; the API may already be ranked.
    def _runtime(c):
        return c.get("runtime", float("inf")) if isinstance(c, dict) else float("inf")

    # getEx already returns a custom backend's ranking.  Its convenience
    # ``runtime`` value may be estimated with the interface's default backend,
    # so sorting it again could reorder the IN_PLACE split-K recommendation
    # using OUT_OF_PLACE costs.
    # A tile restriction only filters candidates, so non-split queries keep the
    # same runtime ordering with or without it.
    ordered = list(configs) if split_k_kind is not None else sorted(configs, key=_runtime)
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


def _fp8_config_signature(config: HeuristicConfig, fields: set) -> tuple:
    """Map a model config to fields actionable by dense FP8 runners."""
    signature = []
    if fields & {"tile", "cluster"}:
        signature.extend((tuple(config.cta), tuple(config.cluster)))
    if "cta_order" in fields:
        signature.append(int(config.cta_order))
    return tuple(signature)


def filter_fp8_tactics(
    tactics: List[tuple],
    ranked_configs: List[HeuristicConfig],
    fields: set,
    max_tactics: int,
    tactic_signature: Callable[[tuple, set], Optional[tuple]],
    sweep_cluster_n: bool = False,
    local_cluster_sweep: bool = False,
) -> List[tuple]:
    """Keep FP8 tactics matching the top model-visible field signatures.

    Instruction shape, B-reuse, TMA-store mode, and other runner-local fields
    stay swept because ``tactic_signature`` deliberately omits them. Rubin may
    additionally sweep cluster-N within each selected (CTA, cluster-M) family.
    ``local_cluster_sweep`` also keeps every cluster of the rank-1 tile and the
    small (<= 2 CTA) clusters of the other selected tiles; see
    ``_widened_cluster``.
    An empty query or intersection returns the original validated tactic list.
    """
    if not tactics or not ranked_configs or not fields:
        return tactics

    valid_signatures = {
        signature
        for tactic in tactics
        if (signature := tactic_signature(tactic, fields)) is not None
    }
    kept_signatures = []
    seen = set()
    for config in ranked_configs:
        signature = _fp8_config_signature(config, fields)
        if signature in valid_signatures and signature not in seen:
            kept_signatures.append(signature)
            seen.add(signature)
            if len(kept_signatures) >= max(1, int(max_tactics)):
                break
    if not kept_signatures:
        return tactics

    widen = (
        _widened_cluster(kept_signatures)
        if local_cluster_sweep and fields & {"tile", "cluster"}
        else (lambda signature: False)
    )
    if sweep_cluster_n and fields & {"tile", "cluster"}:

        def _cluster_n_family(signature):
            """Ignore the locally swept cluster-N dimension in the matching key."""
            cta, cluster, *remaining = signature
            return (cta, int(cluster[0]), *remaining)

        kept_families = {_cluster_n_family(signature) for signature in kept_signatures}
        selected = []
        for tactic in tactics:
            signature = tactic_signature(tactic, fields)
            if signature is not None and (
                _cluster_n_family(signature) in kept_families or widen(signature)
            ):
                selected.append(tactic)
    else:
        kept = set(kept_signatures)
        selected = []
        for tactic in tactics:
            signature = tactic_signature(tactic, fields)
            if signature is not None and (signature in kept or widen(signature)):
                selected.append(tactic)
    return selected if selected else tactics


def _widened_cluster(ranked_signatures: List[tuple]) -> Callable[[tuple], bool]:
    """Admit locally swept clusters around model-ranked (tile, cluster) keys.

    On Rubin the model ranks tiles well but over-values large clusters (4x1,
    4x2, 2x4), while the measured winner often uses the same tile with a
    1x1/1x2/2x1 cluster. Keep every cluster of the rank-1 tile and the
    small clusters of the other ranked tiles. Keys are (tile, cluster, *rest);
    ``rest`` (e.g. a model CTA order) must match a ranked key of that tile.
    """
    if not ranked_signatures:
        return lambda signature: False
    rank1_tile = ranked_signatures[0][0]
    ranked_rest = {(signature[0], tuple(signature[2:])) for signature in ranked_signatures}

    def admit(signature: tuple) -> bool:
        cta, cluster, *rest = signature
        if (cta, tuple(rest)) not in ranked_rest:
            return False
        return cta == rank1_tile or int(cluster[0]) * int(cluster[1]) <= 2

    return admit


def expand_bf16_tactics_with_heuristic_splits(
    tactics: List[tuple], ranked_configs: List[HeuristicConfig]
) -> List[tuple]:
    """Make model-recommended split counts available for per-config matching.

    The runner has already validated each base tactic's tile and cluster. Split
    count does not participate in ``can_implement``. The caller must ensure
    its output shape supports direct reduction before expanding the validated
    templates. Every positive model-recommended split is retained, including
    values absent from the baseline sweep.
    """
    recommended = {int(config.split_k) for config in ranked_configs if int(config.split_k) > 1}
    if not recommended:
        return tactics

    expanded = list(tactics)
    seen = set(expanded)
    templates = []
    seen_templates = set()
    for tactic in tactics:
        if not isinstance(tactic, tuple) or len(tactic) < 6 or tactic[0] != "base":
            continue
        template = (tactic[1], tuple(tactic[2]), tuple(tactic[3]), tactic[4])
        if template not in seen_templates:
            templates.append(template)
            seen_templates.add(template)
    for use_2cta, mma_tiler_mn, cluster_shape_mn, max_num_ab_stage in templates:
        for split_k in sorted(recommended):
            tactic = (
                "base",
                use_2cta,
                mma_tiler_mn,
                cluster_shape_mn,
                max_num_ab_stage,
                split_k,
            )
            if tactic not in seen:
                expanded.append(tactic)
                seen.add(tactic)
    return expanded


def _bf16_actionable_fields(fields: set, split_k: int) -> set:
    """Return model fields the selected BF16 kernel path can control.

    Split-K>1 uses the reduction path's legacy six-field tactic and fixed
    scheduler defaults, so swizzle and CTA order are not actionable there.
    Split-K itself, when selected, remains model-controlled; tile/cluster
    fields also remain part of the match when requested.
    """
    if split_k > 1:
        return fields - {"swizzle", "cta_order"}
    return fields


def _bf16_config_signature(config: HeuristicConfig, fields: set) -> tuple:
    """Map a model config into the BF16 runner's selected-field key."""
    # The model's split recommendation only narrows the actionable fields when
    # split-K was selected. Honoring it otherwise strips swizzle/cta_order from
    # the key and matches the split-K tactics instead of the scheduler sweep the
    # caller asked the model to rank.
    config_split_k = max(1, int(config.split_k)) if "split_k" in fields else 1
    fields = _bf16_actionable_fields(fields, config_split_k)
    signature = []
    if fields & {"tile", "cluster"}:
        cta_m, cta_n = (int(v) for v in config.cta)
        cluster_m, cluster_n = (int(v) for v in config.cluster)
        # The model reports a per-CTA tile. Both MMA instruction modes remain
        # local choices for this tile and cluster.
        signature.extend(((cta_m, cta_n), (cluster_m, cluster_n)))
    if "split_k" in fields:
        signature.append(max(1, int(config.split_k)))
    if "swizzle" in fields:
        signature.append(max(1, int(config.swizzle_factor)))
    if "cta_order" in fields:
        signature.append(int(config.cta_order))
    return tuple(signature)


def _bf16_tactic_signature(tactic: tuple, fields: set) -> Optional[tuple]:
    """Map a base BF16 tactic into the same selected-field key as the model."""
    if not isinstance(tactic, tuple) or len(tactic) < 6 or tactic[0] != "base":
        return None
    split_k = max(1, int(tactic[5]))
    fields = _bf16_actionable_fields(fields, split_k)
    signature = []
    if fields & {"tile", "cluster"}:
        cta_group = 2 if tactic[1] else 1
        mma_tiler_m, mma_tiler_n = tactic[2]
        if mma_tiler_m % cta_group:
            return None
        signature.extend(((mma_tiler_m // cta_group, mma_tiler_n), tuple(tactic[3])))
    if "split_k" in fields:
        signature.append(split_k)
    if "swizzle" in fields:
        # Split-K tactics use the kernel scheduler defaults and therefore keep
        # their legacy six-field form.
        signature.append(max(1, int(tactic[7])) if len(tactic) > 7 else 1)
    if "cta_order" in fields:
        raster_along = tactic[6] if len(tactic) > 6 else "m"
        signature.append(cute_raster_to_nvmmh_cta_order(raster_along))
    # An empty key carries no information, so it would match every config the
    # model returns. Report the tactic as unmappable instead: a split-K tactic
    # has no actionable scheduler knob, and callers already sweep unmatched
    # tactics via the fallback path.
    return tuple(signature) if signature else None


def _bf16_structural_tactic_signature(tactic: tuple, fields: set) -> Optional[tuple]:
    """Return the model key after removing directly-applied scheduler knobs."""
    if not isinstance(tactic, tuple) or len(tactic) < 6 or tactic[0] != "base":
        return None
    split_k = max(1, int(tactic[5]))
    actionable_fields = _bf16_actionable_fields(fields, split_k)
    structural_fields = actionable_fields - {"swizzle", "cta_order"}
    if not structural_fields:
        # Scheduler-only guidance applies to the non-split base path. Split-K
        # kernels have fixed scheduler defaults and must stay unannotated.
        return () if split_k == 1 and actionable_fields else None
    return _bf16_tactic_signature(tactic, structural_fields)


def _bf16_structural_config_signature(config: HeuristicConfig, fields: set) -> tuple:
    """Return the config key used before scheduler values are materialized."""
    config_split_k = max(1, int(config.split_k)) if "split_k" in fields else 1
    actionable_fields = _bf16_actionable_fields(fields, config_split_k)
    structural_fields = actionable_fields - {"swizzle", "cta_order"}
    return _bf16_config_signature(config, structural_fields)


def _bf16_tile_family(signature: tuple) -> tuple:
    """Drop the cluster from a tile/cluster-led BF16 key; keep the rest."""
    return (signature[0],) + tuple(signature[2:])


def _apply_bf16_scheduler_config(tactic: tuple, config: HeuristicConfig, fields: set) -> tuple:
    """Copy selected scheduler values from one NVMMH config to a base tactic."""
    split_k = max(1, int(tactic[5]))
    actionable_fields = _bf16_actionable_fields(fields, split_k)
    scheduler_fields = actionable_fields & {"swizzle", "cta_order"}
    if not scheduler_fields:
        return tactic
    raster_order = (
        nvmmh_cta_order_to_cute_raster(config.cta_order) if "cta_order" in scheduler_fields else "m"
    )
    swizzle_size = max(1, int(config.swizzle_factor)) if "swizzle" in scheduler_fields else 1
    return tactic[:6] + (raster_order, swizzle_size, True)


def filter_bf16_tactics(
    tactics: List[tuple],
    ranked_configs: List[HeuristicConfig],
    fields: set,
    max_tactics: int,
    include_known_good: bool = True,
    fallback_tactics: Optional[List[tuple]] = None,
    local_cluster_sweep: bool = False,
) -> List[tuple]:
    """Return top model-matched BF16 tactics with direct scheduler annotation.

    ``local_cluster_sweep`` (Rubin) also keeps every legal cluster of the
    rank-1 matched tile and the small (<= 2 CTA) clusters of the other
    matched tiles; see ``_widened_cluster``.

    Only base tactics participate in model matching. Valid preferred/fallback
    cluster schedulers cannot be represented directly by the model's one cluster
    field, so only the validated q_b known-good tactic is supplemented when its
    implicit split-K=1 agrees with a matched model configuration. Each model
    configuration keeps its own split count during matching. An empty
    intersection returns ``fallback_tactics`` (normally the exact list produced
    by ``get_valid_tactics``).
    """
    fallback = tactics if fallback_tactics is None else fallback_tactics
    if not tactics or not ranked_configs or not fields:
        return fallback
    valid_signatures = {
        signature
        for tactic in tactics
        if (signature := _bf16_structural_tactic_signature(tactic, fields)) is not None
    }
    # The model sometimes ranks a supported tile with a cluster the kernel
    # cannot launch (SM107 rejects CTA_N=256 with cluster_n=2). Dropping that
    # config discards the tile entirely, even when it is the fastest one, so
    # keep the tile and let the autotuner pick among its legal clusters. The
    # substituted family counts as one model key.
    substitute_cluster = bool(fields & {"tile", "cluster"})
    signatures_by_tile = {}
    if substitute_cluster:
        for signature in valid_signatures:
            signatures_by_tile.setdefault(_bf16_tile_family(signature), set()).add(signature)
    config_by_signature = {}
    model_keys = 0
    for config in ranked_configs:
        signature = _bf16_structural_config_signature(config, fields)
        if signature in valid_signatures:
            matched = [signature]
        elif substitute_cluster:
            matched = sorted(signatures_by_tile.get(_bf16_tile_family(signature), ()))
        else:
            matched = []
        matched = [s for s in matched if s not in config_by_signature]
        if not matched:
            continue
        for s in matched:
            config_by_signature[s] = config
        model_keys += 1
        if model_keys >= max(1, int(max_tactics)):
            break
    if not config_by_signature:
        return fallback
    if local_cluster_sweep and substitute_cluster:
        # Insertion order follows the model ranking, so the first key is the
        # rank-1 matched tile family.
        ranked_signatures = list(config_by_signature)
        config_by_family = {}
        for signature, config in config_by_signature.items():
            config_by_family.setdefault(_bf16_tile_family(signature), config)
        widen = _widened_cluster(ranked_signatures)
        for signature in sorted(valid_signatures):
            if signature not in config_by_signature and widen(signature):
                config_by_signature[signature] = config_by_family[_bf16_tile_family(signature)]

    allow_known_good_supplement = include_known_good and (
        "split_k" not in fields
        or any(int(config.split_k) == 1 for config in config_by_signature.values())
    )
    selected = []
    for tactic in tactics:
        signature = _bf16_structural_tactic_signature(tactic, fields)
        config = config_by_signature.get(signature)
        if config is None:
            continue
        selected.append(_apply_bf16_scheduler_config(tactic, config, fields))
    if (
        allow_known_good_supplement
        and _BF16_KNOWN_GOOD_PREFERRED_TACTIC in tactics
        and _BF16_KNOWN_GOOD_PREFERRED_TACTIC not in selected
    ):
        selected.append(_BF16_KNOWN_GOOD_PREFERRED_TACTIC)
    return selected if selected else fallback
