# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuteDSL MegaMoE NVFP4 custom op + TunableRunner.

Wraps the ported ``Sm100MegaMoEKernel`` (see
``tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/``) into the
standard TRT-LLM CuteDSL op pattern used by
``cute_dsl_custom_ops.py``:

* :class:`Sm100MegaMoENvfp4Runner` is the :class:`TunableRunner` that
  owns the kernel compile cache, the candidate-tactic enumeration, and
  the per-launch ``cute.compile`` + invocation. Tactic representation
  is a tuple of JSON-friendly primitives so ``eval(repr(tactic))``
  round-trips (required by the autotuner cache).
* ``trtllm::cute_dsl_megamoe_nvfp4_blackwell`` is the registered torch
  custom op that the ``MegaMoECuteDsl`` backend calls from
  ``run_moe``. It runs ``AutoTuner.choose_one`` once per call to pick
  the best tactic and forwards to the runner.

The backend never instantiates :class:`Sm100MegaMoENvfp4Runner`
directly; this mirrors how ``CuteDslFusedMoE`` only consumes
``torch.ops.trtllm.cute_dsl_nvfp4_*`` and never reaches into
``cute_dsl_custom_ops.py`` for its inner runners. Keeping the boundary
here lets us evolve the tactic enumeration / compile cache without
touching the MoE backend.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Any, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version
from ...math_utils import ceil_div, pad_up
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2


def _import_megamoe_kernel():
    """Import the in-tree MegaMoE kernel class + combine-format enum (lazy so
    non-SM100 / no-cutlass-dsl environments can still import this module)."""
    from ..cute_dsl_kernels.mega_moe_nvfp4 import import_kernel
    from ..cute_dsl_kernels.mega_moe_nvfp4.token_comm import CombineFormat

    return import_kernel(), CombineFormat


__all__ = [
    "IS_MEGAMOE_OP_AVAILABLE",
    "MEGAMOE_OP_UNAVAILABLE_REASON",
    "default_megamoe_tactic",
    "enumerate_megamoe_candidate_tactics",
    "megamoe_activation_sf_bytes_per_row",
    "validate_megamoe_tactic",
]

# Set to ``True`` if every symbol the op registration needs imports
# cleanly. ``False`` keeps the op unregistered so callers fall back via
# the factory instead of crashing the whole ``custom_ops`` package
# import. ``IS_CUTLASS_DSL_AVAILABLE`` only probes ``cutlass.cute``;
# half-installed or older cutlass-dsl wheels can still expose
# ``cutlass.cute`` but miss ``cutlass.torch`` / ``cutlass._mlir`` /
# ``cute_nvgpu`` / the symm-memory adapter symbols this op needs.
IS_MEGAMOE_OP_AVAILABLE: bool = False
MEGAMOE_OP_UNAVAILABLE_REASON: Optional[str] = None

# Process-global set of (local_ws_ptr, shared_ws_ptr, world_size) keys whose
# in-kernel barrier (counter, signal) has already been fenced-reset. Module-global
# (not per-runner) because the runner is rebuilt every op call; the fence must
# fire ONCE per key. See the fence logic in forward().
_MEGAMOE_FENCED_KEYS: set = set()


# ---------------------------------------------------------------------------
# Tactic representation (v3: 8-tuple perf knobs)
# ---------------------------------------------------------------------------
#
# A tactic is a tuple of JSON-friendly primitives (lists / ints / bools /
# strings / nested tuples) so it round-trips through ``json.dumps``/
# ``json.loads`` *and* ``eval(repr(tactic))`` -- both are required by
# ``TunableRunner`` cache serialization. Order matches the kernel
# constructor kwargs.
#
#   (mma_tiler_mnk,       # list[int] of length 3 (M decides use_2cta)
#    cluster_shape_mnk,   # list[int] of length 3
#    group_hint,          # int, TUNED (>=512); NOT max_active_clusters
#    load_balance_mode,   # str: "static" | "atomic_counter"
#    token_back_mode,     # "epi_warps" | "standalone_warps" | "reuse_dispatch_warps"
#    use_bulk_fc2_store,  # bool (bulk store valid only with epi_warps)
#    flag_batch,          # int >= 1 (standalone_warps requires == 1)
#    epi_flag_batch)      # (int, int) fc1/fc2 done-counter publish batch
#
# Derived / out-of-tuple:
#   use_2cta_instrs      = (mma_tiler_mnk[0] == 256)  -- derived in _build_kernel.
#   in_kernel_fc2_reduce -- backend functional config (default False), carried
#                           in the runner ``unique_id``, NOT a perf-tuning knob.
#
# Tuple wrapping makes the tactic hashable, which AutoTuner needs for the
# tactics cache. Lists / nested tuples inside the tuple are reconstructed
# from JSON intact.

_TACTIC_LEN = 8

# Kernel-side upper bound on the done-counter publish batch. ``flag_batch`` is
# hard-checked ``[1, 32]`` in ``TokenInPullTokenBackPush.__init__``
# (token_comm.py) and the ``epi_flag_batch`` entries are silently clamped to
# ``[1, 32]`` in ``SwapABSwigluFp4Epilogue.__init__`` (epilogue_refactor.py).
# Reject ``> 32`` here so a hand-supplied / JSON-cached tactic fails fast with a
# clear message instead of crashing at kernel build (flag_batch) or silently
# running at 32 (epi_flag_batch). The curated/full enumeration tops out at
# flag_batch=16 and epi_flag_batch=(2, 4), so legal candidates are unaffected.
_FLAG_BATCH_MAX = 32


def _unpack_tactic(tactic: Tuple) -> Tuple:
    """Return the tactic's 8 fields in canonical order, the single source of
    truth for the field layout.

    Every consumer (validate / kernel build / cache key / compile / log)
    unpacks through this helper so the field order is defined once. A 9th
    tactic field is then a one-line change here plus the consumers that
    actually use it, instead of editing five copy-pasted positional unpacks
    that must stay in lockstep. Plain positional unpack (no validation);
    callers that need legality go through :func:`validate_megamoe_tactic`.
    """
    (
        mma_tiler,
        cluster_shape,
        group_hint,
        load_balance_mode,
        token_back_mode,
        use_bulk_fc2_store,
        flag_batch,
        epi_flag_batch,
    ) = tactic
    return (
        mma_tiler,
        cluster_shape,
        group_hint,
        load_balance_mode,
        token_back_mode,
        use_bulk_fc2_store,
        flag_batch,
        epi_flag_batch,
    )


def default_megamoe_tactic(num_tokens: int) -> Tuple:
    """Token-aware fallback tactic (autotune disabled / cache miss / tactic=-1).

    Picks one of three token regimes. This is NOT a tactic the autotuner
    profiles; it is the deterministic fallback when no tuned tactic is
    available. The autotuner, when enabled, profiles the curated candidate
    list per token bucket instead.
    """
    if num_tokens <= 1024:
        # decode winner: N128 only helps epi_warps + bulk.
        return ([256, 128, 256], [2, 1, 1], 512, "static", "epi_warps", True, 1, (1, 1))
    if num_tokens <= 8192:
        # transition/mid: robust N256 geometry, flag_batch=4 as mid default.
        return ([256, 256, 256], [2, 1, 1], 512, "static", "epi_warps", True, 4, (1, 1))
    # prefill: reuse_dispatch_warps forces non-bulk fc2 store; atomic_counter
    # only helps the very large tail (>=16384).
    return (
        [256, 256, 256],
        [2, 1, 1],
        512,
        "atomic_counter" if num_tokens >= 16384 else "static",
        "reuse_dispatch_warps",
        False,
        8,
        (2, 4),
    )


# ---------------------------------------------------------------------------
# Curated tuning space.
#
# ``MEGAMOE_CUTEDSL_TUNING_FULL=0`` (default): curated reduced space.
# ``MEGAMOE_CUTEDSL_TUNING_FULL=1``: full cartesian product (exhaustive search).
# Both share the same legality filter (``validate_megamoe_tactic`` +
# token_back/store binding + N128-only-epi rule).
# ---------------------------------------------------------------------------

# token_back_mode -> the only legal use_bulk_fc2_store (bulk binds to epi_warps;
# reuse/standalone force non-bulk).
_TOKEN_BACK_STORE_BINDING: dict = {
    "epi_warps": True,
    "reuse_dispatch_warps": False,
    "standalone_warps": False,
}

# (mma_tiler_mnk, cluster_shape_mnk) geometries. use_2cta is derived (M==256).
# BEST=0 reduced set: N256 + N128, cluster_m2 only (drop N64, M128/1-CTA,
# cluster_m4). BEST=1 full set adds M128/1-CTA + cluster_m4 + N64.
_GEOMETRIES_REDUCED: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((256, 256, 256), (2, 1, 1)),
    ((256, 128, 256), (2, 1, 1)),
)
# The 15 legal geometries: M128/1-CTA with
# cluster_m in {1,2,4}, and M256/2-CTA with cluster_m in {2,4} (use_2cta even
# divisibility), each crossed with N in {64,128,256}.
# Defined as ``_GEOMETRIES_REDUCED`` plus the extra geometries so the reduced
# set stays a guaranteed subset of the full set (no manual drift between them).
_GEOMETRIES_FULL: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    _GEOMETRIES_REDUCED
    + (
        ((128, 64, 256), (1, 1, 1)),
        ((128, 128, 256), (1, 1, 1)),
        ((128, 256, 256), (1, 1, 1)),
        ((128, 64, 256), (2, 1, 1)),
        ((128, 128, 256), (2, 1, 1)),
        ((128, 256, 256), (2, 1, 1)),
        ((128, 64, 256), (4, 1, 1)),
        ((128, 128, 256), (4, 1, 1)),
        ((128, 256, 256), (4, 1, 1)),
        ((256, 64, 256), (2, 1, 1)),
        ((256, 64, 256), (4, 1, 1)),
        ((256, 128, 256), (4, 1, 1)),
        ((256, 256, 256), (4, 1, 1)),
    )
)

# token_back modes scanned. BEST=0 drops standalone (never a sole winner).
_TOKEN_BACK_MODES_REDUCED: Tuple[str, ...] = ("epi_warps", "reuse_dispatch_warps")
_TOKEN_BACK_MODES_FULL: Tuple[str, ...] = (
    "epi_warps",
    "reuse_dispatch_warps",
    "standalone_warps",
)

# Load-balance modes supported by the integrated fused FC12 path (see
# ImplDesc.__post_init__ in fc1_fc2_fuse_sched.py).
# ``clc`` is intentionally excluded -- it routes through a separate
# scheduler class not wired through the fused FC12 kernel here.
_LOAD_BALANCE_MODE_CANDIDATES: Tuple[str, ...] = ("static", "atomic_counter")

# The non-bulk standalone tactic: quantized combine (fp8/fp4) and form-B reject
# the bulk fc2 store at kernel construction, so this is the deterministic
# default (and sizing-probe) tactic for those modes.
_MEGAMOE_NONBULK_STANDALONE_TACTIC: Tuple = (
    [256, 256, 256],
    [2, 1, 1],
    512,
    "static",
    "standalone_warps",
    False,
    1,
    (2, 4),
)

# group_hint scan: ~512 saturates, larger is better, omission -> ~74 (worst).
# BEST=0 scans {512, 1024}; BEST=1 adds 256 (validated but low-value).
_GROUP_HINT_REDUCED: Tuple[int, ...] = (512, 1024)
_GROUP_HINT_FULL: Tuple[int, ...] = (256, 512, 1024)

# flag_batch scan. BEST=0 {1,4,8}; BEST=1 {1,4,8,16}.
_FLAG_BATCH_REDUCED: Tuple[int, ...] = (1, 4, 8)
_FLAG_BATCH_FULL: Tuple[int, ...] = (1, 4, 8, 16)

# epi_flag_batch is token-bucket dependent (NOT a free scan axis):
# decode/transition use (1, 1); prefill (>8192) uses (2, 4). One value per
# token bucket; see ``_epi_flag_batch_for_tokens``.
_EPI_FLAG_BATCH_SMALL: Tuple[int, int] = (1, 1)
_EPI_FLAG_BATCH_LARGE: Tuple[int, int] = (2, 4)
_EPI_FLAG_BATCH_TOKEN_THRESHOLD = 8192


def _epi_flag_batch_for_tokens(num_tokens: int) -> Tuple[int, int]:
    """Token-bucket epi_flag_batch: ``<=8192 -> (1,1)``, ``>8192 -> (2,4)``."""
    if num_tokens > _EPI_FLAG_BATCH_TOKEN_THRESHOLD:
        return _EPI_FLAG_BATCH_LARGE
    return _EPI_FLAG_BATCH_SMALL


# Kernel-construction knobs locked until the backend owns the corresponding
# runtime buffer / scheduler contracts.
_LOCKED_KERNEL_KWARGS = {
    "force_static_sched": True,
    "clc_bundle_size": None,
    "num_sched_stages": None,
}


def _is_pow2_in_range(val: int, lo: int, hi: int) -> bool:
    return lo <= val <= hi and (val & (val - 1)) == 0


def megamoe_activation_sf_bytes_per_row(hidden_size: int) -> int:
    """Return the per-row byte width the MegaMoE kernel expects for the
    activation SF tensor.

    The kernel reads ``ceil(hidden / (4 * sf_vec_size=64)) * 4`` FP8
    bytes per token (see ``sf_uint32_per_token`` in megamoe_kernel.py
    -- one uint32 packs 4 FP8 scales, each covering 16 NVFP4 elements,
    so each uint32 covers 64 elements). For hidden sizes that are
    multiples of 32 but not 64 (e.g. 1568, 1632, 2080) the naive
    ``hidden // 16`` byte-row width is short by 2 bytes; the kernel's
    TMA load would then either read uninitialized bytes or trip the
    last-row stride check. Always use this helper when allocating /
    sizing activation SF tensors at the backend boundary.
    """
    if hidden_size <= 0 or hidden_size % 32 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 32, got {hidden_size}")
    # ceil(hidden / 16) rounded up to multiples of 4 FP8 columns
    # (= `pad_up(ceil_div(hidden_size, scaling_vector_size=16), 4)`), matching
    # the kernel's TMA load width and the ``can_implement`` hidden_size
    # alignment rule.
    return pad_up(ceil_div(hidden_size, 16), 4)


def validate_megamoe_tactic(tactic: Tuple) -> None:
    """Validate a tactic tuple against the kernel-side constraints
    (see the tactic-representation comment block above). Raises
    ``ValueError`` with a clear message on failure; the caller
    (``get_valid_tactics`` / ``forward``) catches and filters.
    """
    from ..cute_dsl_kernels.mega_moe_nvfp4 import (
        Nvfp4BlockSize,
        SupportedMmaTileM,
        SupportedMmaTileN,
    )

    if (not isinstance(tactic, tuple)) or len(tactic) != _TACTIC_LEN:
        raise ValueError(
            f"MegaMoE tactic must be an {_TACTIC_LEN}-tuple, got "
            f"{type(tactic).__name__} len={len(tactic) if isinstance(tactic, tuple) else 'NA'}={tactic!r}"
        )
    (
        mma_tiler,
        cluster_shape,
        group_hint,
        load_balance_mode,
        token_back_mode,
        use_bulk_fc2_store,
        flag_batch,
        epi_flag_batch,
    ) = _unpack_tactic(tactic)

    if (not isinstance(mma_tiler, (list, tuple))) or len(mma_tiler) != 3:
        raise ValueError(f"mma_tiler_mnk must be a 3-tuple/list, got {mma_tiler!r}")
    if mma_tiler[0] not in SupportedMmaTileM:
        raise ValueError(
            f"mma_tiler_mnk[0]={mma_tiler[0]} not in SupportedMmaTileM={SupportedMmaTileM}"
        )
    if mma_tiler[1] not in SupportedMmaTileN:
        raise ValueError(
            f"mma_tiler_mnk[1]={mma_tiler[1]} not in SupportedMmaTileN={SupportedMmaTileN}"
        )
    if mma_tiler[2] % (Nvfp4BlockSize * 4) != 0:
        raise ValueError(
            f"mma_tiler_mnk[2]={mma_tiler[2]} must be a multiple of "
            f"{Nvfp4BlockSize * 4} (= sf_vec_size * 4); see kernel_fc12 "
            f"_validate_mma_*."
        )

    # ``use_2cta_instrs`` is DERIVED from M (not a tactic field): M==256 -> 2cta.
    use_2cta = mma_tiler[0] == 256

    if (not isinstance(cluster_shape, (list, tuple))) or len(cluster_shape) != 3:
        raise ValueError(f"cluster_shape_mnk must be a 3-tuple/list, got {cluster_shape!r}")
    if cluster_shape[2] != 1:
        raise ValueError(f"cluster_shape_mnk[2] (L axis) must be 1, got {cluster_shape[2]}")
    if cluster_shape[1] != 1:
        raise ValueError(
            f"cluster_shape_mnk[1] (N axis) must be 1 for the integrated "
            f"fused FC12 path; got {cluster_shape[1]}."
        )
    for axis, val in zip(("M", "N"), (cluster_shape[0], cluster_shape[1])):
        if not _is_pow2_in_range(val, 1, 4):
            raise ValueError(
                f"cluster_shape_mnk {axis}-axis must be a power of two in [1, 4], got {val}."
            )
    if cluster_shape[0] * cluster_shape[1] > 16:
        raise ValueError(
            f"cluster_shape_mnk[M]*cluster_shape_mnk[N] must be <= 16, got "
            f"{cluster_shape[0] * cluster_shape[1]}."
        )

    if cluster_shape[0] % (2 if use_2cta else 1) != 0:
        raise ValueError(
            f"cluster_shape_mnk[0] ({cluster_shape[0]}) must be divisible by "
            f"{(2 if use_2cta else 1)} when use_2cta_instrs={use_2cta} "
            f"(derived from mma_tiler_mnk[0]={mma_tiler[0]})."
        )

    if (not isinstance(group_hint, int)) or isinstance(group_hint, bool) or group_hint <= 0:
        raise ValueError(f"group_hint must be a positive int, got {group_hint!r}.")
    if group_hint < 512:
        logger.warning_once(
            f"[MegaMoE] group_hint={group_hint} < 512; group_hint saturates "
            f"around 512 and smaller values are strictly worse.",
            key="megamoe_group_hint_lt_512",
        )

    if load_balance_mode not in {"static", "atomic_counter"}:
        raise ValueError(
            f"load_balance_mode must be 'static' or 'atomic_counter', got {load_balance_mode!r}."
        )

    if token_back_mode not in {"epi_warps", "standalone_warps", "reuse_dispatch_warps"}:
        raise ValueError(
            f"token_back_mode must be one of 'epi_warps' / 'standalone_warps' / "
            f"'reuse_dispatch_warps', got {token_back_mode!r}."
        )

    if not isinstance(use_bulk_fc2_store, bool):
        raise ValueError(f"use_bulk_fc2_store must be bool, got {use_bulk_fc2_store!r}.")
    # bulk fc2 store is only valid under epi_warps (the non-epi modes stage fc2
    # output locally -> forced non-bulk).
    if use_bulk_fc2_store and token_back_mode != "epi_warps":  # nosec B105
        raise ValueError(
            f"use_bulk_fc2_store=True requires token_back_mode='epi_warps', got "
            f"{token_back_mode!r} (non-epi token-back forces non-bulk fc2 store)."
        )
    # N128 only pays off under epi_warps + bulk; other token-back modes
    # with N128 are not recommended.
    if mma_tiler[1] == 128 and token_back_mode != "epi_warps":  # nosec B105
        raise ValueError(
            f"mma_tiler_mnk[1]=128 (N128) is only recommended with "
            f"token_back_mode='epi_warps'; got {token_back_mode!r}."
        )

    if (
        (not isinstance(flag_batch, int))
        or isinstance(flag_batch, bool)
        or not (1 <= flag_batch <= _FLAG_BATCH_MAX)
    ):
        raise ValueError(
            f"flag_batch must be an int in [1, {_FLAG_BATCH_MAX}] (kernel "
            f"TokenInPullTokenBackPush hard limit), got {flag_batch!r}."
        )
    # standalone token-back warps publish one slot at a time.
    if token_back_mode == "standalone_warps" and flag_batch != 1:  # nosec B105
        raise ValueError(
            f"token_back_mode='standalone_warps' requires flag_batch == 1, got {flag_batch}."
        )

    if (not isinstance(epi_flag_batch, (tuple, list))) or len(epi_flag_batch) != 2:
        raise ValueError(f"epi_flag_batch must be a 2-tuple (fc1, fc2), got {epi_flag_batch!r}.")
    for v in epi_flag_batch:
        # The epilogue silently clamps each entry to [1, 32]; reject out-of-range
        # here so the requested value never diverges from what runs, and so two
        # tactics differing only above 32 do not produce distinct cache keys for
        # identical kernel behavior.
        if (not isinstance(v, int)) or isinstance(v, bool) or not (1 <= v <= _FLAG_BATCH_MAX):
            raise ValueError(
                f"epi_flag_batch entries must be ints in [1, {_FLAG_BATCH_MAX}] "
                f"(epilogue clamp range), got {epi_flag_batch!r}."
            )


def enumerate_megamoe_candidate_tactics(num_tokens: int) -> List[Tuple]:
    """Return the curated candidate tactic list for the current token bucket.

    ``MEGAMOE_CUTEDSL_TUNING_FULL`` selects the search space size (0/unset ->
    curated reduced set ~36 per bucket; 1 -> full exhaustive cartesian product).
    Both build the full cartesian product over the selected dimension ranges and
    then drop illegal combinations through the SAME ``validate_megamoe_tactic``
    filter (geometry legality, bulk<=>epi binding, standalone=>flag_batch==1,
    N128-only-epi).

    ``epi_flag_batch`` is token-bucket dependent (one value per bucket, NOT a
    free scan axis): ``num_tokens <= 8192 -> (1, 1)``, ``> 8192 -> (2, 4)``.
    """
    # 0 / unset -> curated reduced space; 1 -> full exhaustive space.
    full = os.environ.get("MEGAMOE_CUTEDSL_TUNING_FULL", "0") == "1"
    geometries = _GEOMETRIES_FULL if full else _GEOMETRIES_REDUCED
    token_back_modes = _TOKEN_BACK_MODES_FULL if full else _TOKEN_BACK_MODES_REDUCED
    group_hints = _GROUP_HINT_FULL if full else _GROUP_HINT_REDUCED
    flag_batches = _FLAG_BATCH_FULL if full else _FLAG_BATCH_REDUCED
    epi_flag_batch = _epi_flag_batch_for_tokens(num_tokens)

    candidates: List[Tuple] = []
    for mma_tiler, cluster_shape in geometries:
        for token_back_mode in token_back_modes:
            use_bulk = _TOKEN_BACK_STORE_BINDING[token_back_mode]
            for load_balance_mode in _LOAD_BALANCE_MODE_CANDIDATES:
                for group_hint in group_hints:
                    for flag_batch in flag_batches:
                        tactic = (
                            list(mma_tiler),
                            list(cluster_shape),
                            int(group_hint),
                            load_balance_mode,
                            token_back_mode,
                            use_bulk,
                            int(flag_batch),
                            tuple(epi_flag_batch),
                        )
                        try:
                            validate_megamoe_tactic(tactic)
                        except ValueError as e:
                            logger.debug(f"[MegaMoE] dropping candidate tactic {tactic!r}: {e}")
                            continue
                        candidates.append(tactic)
    return candidates


if IS_CUTLASS_DSL_AVAILABLE:
    # Stricter than ``IS_CUTLASS_DSL_AVAILABLE``: the op needs
    # ``cutlass.torch.from_dlpack``, ``cutlass._mlir`` adapters for the
    # ``SymBufferHost`` struct, ``cute_nvgpu.tcgen05`` for the MMA
    # atoms, and the ported MegaMoE NVFP4 kernel package. A half-
    # installed or older cutlass-dsl wheel exposes ``cutlass.cute`` but
    # is missing one of the above; we catch every such failure so the
    # rest of the ``custom_ops`` package still imports and only this
    # op is unregistered.
    try:
        import cutlass
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as torch_symm_mem
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401

        try:
            from cuda.bindings import driver as cuda
        except ImportError:
            from cuda import cuda

        # ``Nvfp4BlockSize`` is probe-only at module load to fail fast
        # when the kernel package is partially installed; it is consumed
        # by the lazy import inside ``validate_megamoe_tactic``.
        from ..cute_dsl_kernels.mega_moe_nvfp4 import (
            Nvfp4BlockSize,  # noqa: F401
            SfPaddingBlock,
        )

        IS_MEGAMOE_OP_AVAILABLE = True
    except Exception as _megamoe_import_err:  # pragma: no cover - env-specific
        MEGAMOE_OP_UNAVAILABLE_REASON = (
            f"MegaMoE CuteDSL op registration probe failed with "
            f"{type(_megamoe_import_err).__name__}: {_megamoe_import_err}"
        )
        logger.info(
            "MegaMoE CuteDSL op skipped: %s. Backend ``MegaMoECuteDsl`` "
            "stays uninstalled; ``torch.ops.trtllm."
            "cute_dsl_megamoe_nvfp4_blackwell`` is not registered. The "
            "factory falls back to CutlassFusedMoE.",
            _megamoe_import_err,
        )
        IS_MEGAMOE_OP_AVAILABLE = False


if IS_MEGAMOE_OP_AVAILABLE:
    # ----- Local workspace cache --------------------------------------------
    #
    # ``local_workspace`` is per-rank CUDA-only and sized by
    # ``kernel.get_workspace_sizes()``. It is shape-stable across forward
    # calls (size only depends on kernel construction kwargs), so we cache
    # it per static shape so multiple MoE layers / chunks amortize the
    # allocation. ``shared_workspace`` lives in the symmetric heap for
    # multi-rank and is supplied by the caller (the MegaMoECuteDsl backend's
    # MegaMoeSymmMemProvider carves it out of the rendezvous'd buffer).
    _MEGAMOE_LOCAL_WORKSPACE_CACHE: dict = {}
    # Cache keys whose workspace was allocated while the AutoTuner was sweeping
    # candidate tactics (``is_tuning_mode``). The tuner profiles ~N candidates
    # per shape, each with a DISTINCT cache_key (the workspace SIZE varies with
    # the tactic -- see ``_tactic_cache_key`` use below) and a workspace that is
    # multi-GiB at high ``max_tokens_per_rank``. Persisting every candidate
    # would accumulate N x local_bytes and OOM. These profiling workspaces are
    # throwaway -- only the winning tactic is used at runtime -- so we free a
    # candidate's workspace as soon as the next candidate is allocated.
    _MEGAMOE_TUNING_WORKSPACE_KEYS: set = set()

    @functools.lru_cache(maxsize=1)
    def _cute_launch_helpers():
        """Import the cutlass-dsl launch surface ONCE per process and build the
        tensor/pointer converters (lazy: the module must import without
        cutlass-dsl / a GPU; hoisted out of forward() so the per-call eager
        launch does not rebuild closures or re-run imports)."""
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as cutlass_utils
        from cutlass.cute.typing import AddressSpace

        from ..cute_dsl_kernels.mega_moe_nvfp4.sym_buffer import SymBufferHost

        def to_cute(t, assumed_align=16):
            ct = cutlass_torch.from_dlpack(t, assumed_align=assumed_align)
            return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

        def to_cute_ptr(t, assumed_align=16):
            return cute.runtime.make_ptr(
                cutlass.Uint8, t.data_ptr(), AddressSpace.gmem, assumed_align=assumed_align
            )

        return to_cute, to_cute_ptr, SymBufferHost, cute, cutlass_utils

    def _get_or_alloc_local_workspace(
        kernel, cache_key: Tuple, device: torch.device
    ) -> torch.Tensor:
        cached = _MEGAMOE_LOCAL_WORKSPACE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # A cache MISS means a new (tactic, shape). Release any workspace held
        # for a PREVIOUSLY-profiled autotune candidate: it belongs to an
        # already-timed tactic that is never relaunched (MegaMoE profiles
        # eagerly -- ``get_tuning_config`` sets ``use_cuda_graph=False`` -- so
        # no captured graph holds the buffer). This bounds the autotune sweep
        # to ONE live local workspace instead of one-per-candidate. Within a
        # single candidate's multi-launch profiling the key HITS the cache, so
        # the buffer is never freed mid-candidate; outside tuning the set is
        # empty, so runtime caching keeps one workspace per MoE shape.
        if _MEGAMOE_TUNING_WORKSPACE_KEYS:
            for stale_key in _MEGAMOE_TUNING_WORKSPACE_KEYS:
                stale_ws = _MEGAMOE_LOCAL_WORKSPACE_CACHE.pop(stale_key, None)
                if stale_ws is not None:
                    # The freed local block can be handed back by the allocator
                    # for a same-size next candidate. ``_MEGAMOE_FENCED_KEYS`` is
                    # keyed on the raw local ptr, so drop the stale key here or a
                    # reused ptr ABA-skips its barrier-reset fence and hangs the
                    # dispatch barrier mid-sweep.
                    _stale_ptr = int(stale_ws.data_ptr())
                    _MEGAMOE_FENCED_KEYS.difference_update(
                        {k for k in _MEGAMOE_FENCED_KEYS if k[0] == _stale_ptr}
                    )
            _MEGAMOE_TUNING_WORKSPACE_KEYS.clear()
        local_bytes, _ = kernel.get_workspace_sizes()
        # The local workspace embeds Int32 atomic counters whose spin_wait expects
        # v >= a positive threshold; a stray negative byte hangs the kernel at 100%
        # SM. But ONLY the counters need zeroing -- the data/scratch regions (the
        # multi-GiB bulk at high max_tokens_per_rank) are overwritten every launch.
        # The kernel exposes exactly the byte ranges to zero, so allocate
        # uninitialised and zero just those (a full ``torch.zeros`` over local_bytes
        # is a huge wasted memset):
        #  - the front counter prefix ``require_zero_workspace_leading_bytes[0]``
        #    (``tail_reset_counters`` bulk-zeros it each launch; first launch needs it);
        #  - the persisted ``nvlink_barrier_counter`` -- deliberately OUTSIDE that
        #    prefix so the tail never resets it (the barrier rides it across launches),
        #    hence only the FIRST launch relies on this caller zero.
        local_workspace = torch.empty(local_bytes, dtype=torch.uint8, device=device)
        _lead = int(kernel.require_zero_workspace_leading_bytes[0])
        local_workspace[:_lead].zero_()
        _bc_off = int(kernel._local_offsets["nvlink_barrier_counter"])
        _bc_n = int(kernel._local_region_by_name["nvlink_barrier_counter"].nbytes)
        local_workspace[_bc_off : _bc_off + _bc_n].zero_()
        _MEGAMOE_LOCAL_WORKSPACE_CACHE[cache_key] = local_workspace
        if AutoTuner.get().is_tuning_mode:
            _MEGAMOE_TUNING_WORKSPACE_KEYS.add(cache_key)
        return local_workspace

    # ----- Symmetric-memory provider (NVSHMEM-equivalent) -------------------
    #
    # PyTorch's ``torch.distributed._symmetric_memory`` is an NVSHMEM-equivalent
    # symmetric-heap provider built on cuMem APIs. It exposes per-rank buffer
    # pointers (``handle.buffer_ptrs``) which we use to populate the
    # ``SymBufferHost(base_addr, offsets, rank_idx, num_max_ranks)`` payload
    # the MegaMoE kernel expects.
    #
    # We allocate ONE large symmetric buffer per (group, layout_key) and
    # carve out fixed-size regions for activation / activation_sf /
    # topk_weights / combine_output / shared_workspace. All ranks have
    # identical region offsets inside the buffer, so peer_offsets =
    # [handle.buffer_ptrs[r] - local_base for r in range(world_size)]
    # correctly maps any region's local pointer to its peer counterpart.
    #
    # Cache lives at module scope so multiple MoE layers with the same
    # (group, layout) share the same allocation (mirrors
    # ``_MEGA_MOE_SYMM_BUFFER_CACHE`` in ``mega_moe_deepgemm.py``).
    _MEGAMOE_SYMM_PROVIDER_CACHE: dict = {}

    @dataclasses.dataclass
    class MegaMoeSymmRegions:
        """User-domain symmetric tensors carved out of a single rendezvous'd
        buffer. All views share the same underlying allocation, so they
        share ``peer_offsets``.

        ``base_buf`` is kept alive for the lifetime of the provider; views
        only stay valid while the provider is in the cache.
        """

        base_buf: torch.Tensor
        activation: torch.Tensor  # (max_T, hidden // 2) uint8 (NVFP4 packed)
        # Row stride MUST equal kernel's ``sf_uint32_per_token * 4 ==
        # ceil(hidden / 64) * 4 == megamoe_activation_sf_bytes_per_row(hidden)``;
        # ``hidden // 16`` is short by 2 bytes for hidden % 64 != 0
        # (1568, 1632, 2080, ...) and triggers a host copy_ shape
        # mismatch in the backend's ``_stage_inputs``. See dispatch
        # kernel sf_addr formula in dispatch_kernel.py.
        activation_sf: torch.Tensor  # (max_T, sf_bytes_per_row) uint8 (FP8 SF)
        topk_weights: torch.Tensor  # (max_T, num_topk) float32
        # (max_T, combine_k, hidden): combine_k = num_topk (form-A) or 1 (form-B).
        combine_output: torch.Tensor
        shared_workspace: torch.Tensor  # (shared_ws_bytes,) uint8
        peer_offsets: List[int]  # symmetric peer-pointer deltas
        rank: int
        world_size: int

    class MegaMoeSymmMemProvider:
        """Allocate + carve symmetric-memory regions for MegaMoE multi-rank
        execution.

        The provider is bound to a ProcessGroup (the EP sub-world the
        kernel exchanges over) and a layout key (hidden, num_topk,
        max_tokens_per_rank, output_dtype, shared_workspace_bytes). It
        survives across MoE layers with identical layouts so the
        expensive ``torch_symm_mem.rendezvous`` collective only runs
        once per build.

        ``MegaMoECuteDsl.create_weights`` constructs the provider via
        :func:`get_megamoe_symm_provider` at build time, so every EP
        rank crosses the (collective) ``torch_symm_mem.rendezvous``
        in lockstep before any forward call. ``run_moe`` only consumes
        the cached :class:`MegaMoeSymmRegions`; doing the rendezvous
        at forward time would risk deadlocking under PP / layer-skip
        and is forbidden by the design contract.
        """

        # Alignment in bytes between region boundaries. 128 B keeps each
        # region aligned to the TMA load requirement; matches the
        # alignment used for blocked_scale / FP8 SF inside the kernel.
        _REGION_ALIGN = 128

        def __init__(
            self,
            *,
            process_group,
            world_size: int,
            rank: int,
            hidden_size: int,
            max_tokens_per_rank: int,
            num_topk: int,
            output_dtype: torch.dtype,
            shared_workspace_bytes: int,
        ) -> None:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("MegaMoeSymmMemProvider requires torch.distributed initialized.")
            if process_group is None:
                raise ValueError(
                    "MegaMoeSymmMemProvider requires a non-None process_group "
                    "(MegaMoECuteDsl resolves this from mapping.moe_ep_group_pg)."
                )
            if not hasattr(process_group, "group_name"):
                raise RuntimeError(
                    "MegaMoeSymmMemProvider requires a torch.distributed "
                    "ProcessGroup with a stable group_name. Use Ray / DeviceMesh "
                    "or pass a group created with dist.new_group(...)."
                )

            self.process_group = process_group
            self.group_name = str(process_group.group_name)
            self.world_size = int(world_size)
            self.rank = int(rank)
            self.hidden_size = int(hidden_size)
            self.max_tokens_per_rank = int(max_tokens_per_rank)
            self.num_topk = int(num_topk)
            self.output_dtype = output_dtype
            # The kernel returns a unified (T, hidden) output (top-k reduced
            # in-kernel for form-B, by the TopkReduce tail kernel for form-A),
            # so the symmetric combine region is (max_T, 1, hidden) and the op
            # aliases it as the 2D output_activation. combine_k MUST match the
            # kernel ``combine_output`` shape -- a mismatch silently corrupts
            # the symmetric combine region.
            self.combine_k = 1

            # Region byte sizes (worst case across launches; staging
            # writes only the live ``T`` rows). NVFP4 packs 2 elems / byte
            # along K so activation rows are hidden // 2 bytes. The SF
            # row width matches the kernel's TMA load expectation
            # ``round_up(ceil(hidden / 16), 4)`` -- naive ``hidden // 16``
            # under-allocates by 2 bytes when ``hidden % 64 != 0`` (e.g.
            # 1568, 1632, 2080).
            act_bytes_per_row = hidden_size // 2
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden_size)
            topkw_bytes_per_row = num_topk * 4  # float32
            combine_bytes_per_row = self.combine_k * hidden_size * output_dtype.itemsize

            act_region = pad_up(max_tokens_per_rank * act_bytes_per_row, self._REGION_ALIGN)
            sf_region = pad_up(max_tokens_per_rank * sf_bytes_per_row, self._REGION_ALIGN)
            topkw_region = pad_up(max_tokens_per_rank * topkw_bytes_per_row, self._REGION_ALIGN)
            combine_region = pad_up(max_tokens_per_rank * combine_bytes_per_row, self._REGION_ALIGN)
            shared_region = pad_up(shared_workspace_bytes, self._REGION_ALIGN)

            self._region_offsets: dict = {}
            self._region_sizes: dict = {}
            cursor = 0
            for name, region in (
                ("activation", act_region),
                ("activation_sf", sf_region),
                ("topk_weights", topkw_region),
                ("combine_output", combine_region),
                ("shared_workspace", shared_region),
            ):
                self._region_offsets[name] = cursor
                self._region_sizes[name] = region
                cursor += region
            total_bytes = cursor

            # Enable symm mem on the group exactly once (idempotent).
            torch_symm_mem.enable_symm_mem_for_group(self.group_name)

            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._buf = torch_symm_mem.empty(total_bytes, device=device, dtype=torch.uint8)
            # Zero the symmetric buffer once at construction so peers
            # read deterministic 0 in the TMA OOB-fill region (the
            # padded per-row tail). Without this, peers see whatever
            # cuMem mapped, producing silently non-deterministic runs.
            self._buf.zero_()
            # Collective: every rank in the group must call rendezvous
            # in lockstep. Safe at construction because the backend
            # resolves the EP PG before the first run_moe.
            self._handle = torch_symm_mem.rendezvous(self._buf, self.group_name)
            local_base = int(self._buf.data_ptr())
            self.peer_offsets: List[int] = []
            for r in range(self.world_size):
                peer_ptr = int(self._handle.buffer_ptrs[r])
                self.peer_offsets.append(peer_ptr - local_base)

            logger.debug(
                "[MegaMoeSymmMemProvider] group=%s rank=%d/%d total_bytes=%d "
                "(activation=%d sf=%d topk_weights=%d combine=%d shared=%d)",
                self.group_name,
                self.rank,
                self.world_size,
                total_bytes,
                act_region,
                sf_region,
                topkw_region,
                combine_region,
                shared_region,
            )

        def _region_view(
            self, name: str, shape: Tuple[int, ...], dtype: torch.dtype
        ) -> torch.Tensor:
            offset = self._region_offsets[name]
            numel = 1
            for d in shape:
                numel *= d
            byte_len = numel * dtype.itemsize
            byte_view = self._buf[offset : offset + byte_len]
            return byte_view.view(dtype).view(shape)

        def get_regions(self) -> MegaMoeSymmRegions:
            hidden = self.hidden_size
            max_t = self.max_tokens_per_rank
            top_k = self.num_topk
            combine_k = self.combine_k
            # ``sf_bytes_per_row`` MUST match the byte width used at
            # allocation time (``__init__`` above) and at the backend's
            # ``quantize_input`` output: kernel reads
            # ``ceil(hidden / 64) * 4`` FP8 bytes per token via the
            # ``sf_addr`` formula in dispatch_kernel.py. ``hidden // 16``
            # under-allocates by 2 bytes when hidden % 64 != 0.
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden)
            return MegaMoeSymmRegions(
                base_buf=self._buf,
                activation=self._region_view("activation", (max_t, hidden // 2), torch.uint8),
                activation_sf=self._region_view(
                    "activation_sf", (max_t, sf_bytes_per_row), torch.uint8
                ),
                topk_weights=self._region_view("topk_weights", (max_t, top_k), torch.float32),
                combine_output=self._region_view(
                    "combine_output", (max_t, combine_k, hidden), self.output_dtype
                ),
                shared_workspace=self._region_view(
                    "shared_workspace", (self._region_sizes["shared_workspace"],), torch.uint8
                ),
                peer_offsets=list(self.peer_offsets),
                rank=self.rank,
                world_size=self.world_size,
            )

    def get_megamoe_symm_provider(
        *,
        process_group,
        world_size: int,
        rank: int,
        hidden_size: int,
        max_tokens_per_rank: int,
        num_topk: int,
        output_dtype: torch.dtype,
        shared_workspace_bytes: int,
    ) -> MegaMoeSymmMemProvider:
        """Return a cached provider for (group, layout). The cache is
        keyed on the group ``group_name`` plus every layout knob that
        affects allocation size so two MoE layers with the same shape
        share the same symmetric buffer.

        First call from each rank performs the (collective)
        ``torch_symm_mem.rendezvous``; subsequent calls return the
        cached provider with no further collectives.
        """
        if not hasattr(process_group, "group_name"):
            raise RuntimeError(
                "process_group must expose .group_name (ProcessGroup created "
                "via Ray DeviceMesh or dist.new_group). Use ConfigurableMoE "
                "with mapping.moe_ep_group_pg."
            )
        cache_key = (
            str(process_group.group_name),
            int(hidden_size),
            int(max_tokens_per_rank),
            int(num_topk),
            str(output_dtype),
            int(shared_workspace_bytes),
        )
        cached = _MEGAMOE_SYMM_PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        provider = MegaMoeSymmMemProvider(
            process_group=process_group,
            world_size=world_size,
            rank=rank,
            hidden_size=hidden_size,
            max_tokens_per_rank=max_tokens_per_rank,
            num_topk=num_topk,
            output_dtype=output_dtype,
            shared_workspace_bytes=shared_workspace_bytes,
        )
        _MEGAMOE_SYMM_PROVIDER_CACHE[cache_key] = provider
        return provider

    # ---- AutoTuner profiling scratch (symmetric, transient) ---------------
    # The fused kernel uses a SINGLE ``peer_rank_ptr_mapper`` (one
    # ``peer_offsets``) for every cross-rank access, so ALL cross-rank tensors
    # -- activation / activation_sf / topk_weights / combine_output AND the
    # shared_workspace -- must live in one symmetric allocation at a consistent
    # per-rank offset. During the real run the backend supplies the staging
    # symmetric buffer; but the AutoTuner regenerates the data inputs as fresh
    # NON-symmetric tensors for each profile, which makes the peer mapping point
    # outside any symmetric region (illegal memory access, multi-rank only).
    # We therefore profile against a SEPARATE symmetric scratch buffer (a full
    # MegaMoeSymmMemProvider with its own peer_offsets) so the real staging
    # buffer is never corrupted. The scratch is shared across MoE layers with
    # the same layout (one buffer per shape) and released after warmup via
    # ``release_megamoe_profiling_scratch()``.
    _MEGAMOE_PROFILING_SCRATCH_CACHE: dict = {}
    _ACTIVE_MEGAMOE_PROFILING_SCRATCH = None

    def get_megamoe_profiling_scratch(
        *,
        process_group,
        world_size: int,
        rank: int,
        hidden_size: int,
        max_tokens_per_rank: int,
        num_topk: int,
        output_dtype: torch.dtype,
        shared_workspace_bytes: int,
    ):
        """Return a cached, transient symmetric scratch provider used ONLY for
        AutoTuner profiling. ``None`` for single-rank (no cross-rank exchange).
        Shares one buffer across layers with the same layout; freed by
        :func:`release_megamoe_profiling_scratch`.
        """
        if int(world_size) <= 1:
            return None
        if not hasattr(process_group, "group_name"):
            raise RuntimeError(
                "get_megamoe_profiling_scratch requires a ProcessGroup with "
                ".group_name (mapping.moe_ep_group_pg)."
            )
        cache_key = (
            str(process_group.group_name),
            int(hidden_size),
            int(max_tokens_per_rank),
            int(num_topk),
            str(output_dtype),
            int(shared_workspace_bytes),
        )
        cached = _MEGAMOE_PROFILING_SCRATCH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        provider = MegaMoeSymmMemProvider(
            process_group=process_group,
            world_size=world_size,
            rank=rank,
            hidden_size=hidden_size,
            max_tokens_per_rank=max_tokens_per_rank,
            num_topk=num_topk,
            output_dtype=output_dtype,
            shared_workspace_bytes=shared_workspace_bytes,
        )
        _MEGAMOE_PROFILING_SCRATCH_CACHE[cache_key] = provider
        return provider

    def set_active_megamoe_profiling_scratch(regions) -> None:
        """Set (``None`` clears) the :class:`MegaMoeSymmRegions` the MegaMoE op
        uses for AutoTuner profiling on the current call. The backend sets this
        around the op invocation; the op consumes it inside ``choose_one``.
        """
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH = regions

    def release_megamoe_profiling_scratch() -> None:
        """Free all profiling-scratch symmetric buffers. Call after warmup once
        tuning is done; profiling re-allocates lazily if it runs again.
        """
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH = None
        _MEGAMOE_PROFILING_SCRATCH_CACHE.clear()

    def query_megamoe_shared_workspace_bytes(
        *,
        world_size: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        tactic: Optional[Tuple] = None,
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        in_kernel_fc2_reduce: bool = False,
        combine_format: str = "bf16",
    ) -> int:
        """Probe ``Sm100MegaMoEKernel.get_workspace_sizes()`` for the
        shared workspace byte count. The SHARED workspace size is
        invariant across all candidate tactics and across the codegen-time
        graph/clamp modes (its regions depend only on world_size /
        num_experts_per_rank / num_topk / max_tokens_per_rank -- see
        _build_shared_region_specs in megamoe_kernel.py), so we use the
        default 8-tuple tactic for the probe. ``apply_topk_in_fc1`` /
        ``gate_up_clamp`` / ``in_kernel_fc2_reduce`` are still threaded so the
        probe kernel ctor signature is satisfied and matches the real build
        (``in_kernel_fc2_reduce`` does not change the SHARED size, only the
        LOCAL workspace, but the ctor requires it).
        """

        if tactic is None:
            # Any valid tactic works: the shared workspace size is invariant
            # across tactics, so reuse the token-aware fallback for sizing.
            # EXCEPTION: quantized combine (fp8/fp4) rejects bulk fc2 store at
            # kernel construction, and default_megamoe_tactic(0) is bulk -> use
            # the non-bulk standalone tactic so the (tactic-invariant) sizing
            # probe can build the kernel.
            if combine_format != "bf16":
                tactic = _MEGAMOE_NONBULK_STANDALONE_TACTIC
            else:
                tactic = default_megamoe_tactic(0)
        (
            mma_tiler,
            cluster_shape,
            group_hint,
            load_balance_mode,
            token_back_mode,
            use_bulk_fc2_store,
            flag_batch,
            epi_flag_batch,
        ) = _unpack_tactic(tactic)
        mma_tiler = tuple(mma_tiler)
        common = dict(
            mma_tiler_mnk=mma_tiler,
            cluster_shape_mnk=tuple(cluster_shape),
            use_2cta_instrs=bool(mma_tiler[0] == 256),
            group_hint=int(group_hint),
            token_padding_block=64,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=str(load_balance_mode),
            static_expert_shape=(
                num_experts_per_rank,
                expand_intermediate_size_per_partition,
                hidden_size,
            ),
            world_size=int(world_size),
            num_topk=int(num_topk),
            max_tokens_per_rank=int(max_tokens_per_rank),
            hidden=int(hidden_size),
            fc2_output_dtype=cutlass.BFloat16,
            in_kernel_fc2_reduce=bool(in_kernel_fc2_reduce),
            token_back_mode=str(token_back_mode),
            non_ubulk_fc2_store=(not bool(use_bulk_fc2_store)),
            flag_batch=int(flag_batch),
            epi_flag_batch=tuple(epi_flag_batch),
            apply_topk_in_fc1=bool(apply_topk_in_fc1),
            gate_up_clamp=(None if gate_up_clamp is None else float(gate_up_clamp)),
            **_LOCKED_KERNEL_KWARGS,
        )
        # The probe MUST be the SAME kernel that actually runs (same combine_format),
        # else the provider carves an undersized shared region and the kernel writes
        # the combine staging out of bounds (single-rank IMA; at EP>1 the faulting
        # rank dies and peers spin on the dispatch barrier -> looks like a hang).
        kernel_cls, CombineFormat = _import_megamoe_kernel()
        probe = kernel_cls(combine_format=CombineFormat.parse(combine_format), **common)
        _, shared_bytes = probe.get_workspace_sizes()
        return int(shared_bytes)

    class Sm100MegaMoENvfp4Runner(TunableRunner):
        """TunableRunner for the ported MegaMoE CuteDSL NVFP4 kernel.

        Owns a process-global ``kernel_cache`` keyed on the full
        ``(static_shape + tactic)`` tuple so multiple MoE layers with
        identical shapes amortize the (expensive) ``cute.compile``.
        ``get_valid_tactics`` enumerates the upstream-tested geometries
        and validates each against the kernel-side constraints.
        """

        # Module-scope compile cache shared by every runner instance.
        kernel_cache: dict = {}

        # Module-scope tuning-config cache keyed on ``unique_id()``. The op
        # rebuilds a runner per call, so an instance-level cache would never
        # hit; keeping it at class scope amortizes the config build across
        # calls (mirrors the ``tuning_config_cache`` of the CuteDSL
        # grouped-gemm runners in ``cute_dsl_custom_ops.py``).
        tuning_config_cache: dict = {}

        def __init__(
            self,
            *,
            world_size: int,
            local_rank: int,
            num_topk: int,
            num_experts_per_rank: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            expand_intermediate_size_per_partition: int,
            max_tokens_per_rank: int,
            output_dtype: torch.dtype,
            apply_topk_in_fc1: bool = True,
            gate_up_clamp: Optional[float] = None,
            in_kernel_fc2_reduce: bool = False,
            combine_format: str = "bf16",
        ) -> None:
            super().__init__()
            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"Sm100MegaMoENvfp4Runner requires SM 100 (B200) or SM 103 "
                    f"(B300); got SM {sm_version}."
                )
            if num_experts_per_rank <= 0:
                raise ValueError(
                    f"num_experts_per_rank must be positive, got {num_experts_per_rank}"
                )
            if max_tokens_per_rank <= 0:
                raise ValueError(f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}")
            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"Sm100MegaMoENvfp4Runner only supports bfloat16 output; got {output_dtype}"
                )
            self.world_size = int(world_size)
            self.local_rank = int(local_rank)
            self.num_topk = int(num_topk)
            self.num_experts_per_rank = int(num_experts_per_rank)
            self.hidden_size = int(hidden_size)
            self.intermediate_size_per_partition = int(intermediate_size_per_partition)
            self.expand_intermediate_size_per_partition = int(
                expand_intermediate_size_per_partition
            )
            self.max_tokens_per_rank = int(max_tokens_per_rank)
            self.output_dtype = output_dtype
            # Codegen-time graph/clamp/output modes. They change the generated
            # kernel (and form-B changes the combine_output buffer shape), so
            # they are part of ``unique_id`` (and therefore the compile-cache +
            # workspace-cache key) -- never per-call runtime kwargs.
            #
            # ``in_kernel_fc2_reduce`` is a backend FUNCTIONAL config (form-A vs
            # form-B output), NOT a perf-tuning tactic field; it lives here so
            # form-A and form-B get independent compile / workspace caches.
            self.apply_topk_in_fc1 = bool(apply_topk_in_fc1)
            self.gate_up_clamp = None if gate_up_clamp is None else float(gate_up_clamp)
            # Per-call AutoTuner profiling scratch (symmetric ``MegaMoeSymmRegions``)
            # set by the op around ``choose_one`` so the profiling pre-hook routes
            # the cross-rank inputs through a symmetric scratch buffer instead of
            # the AutoTuner's fresh non-symmetric tensors. ``None`` outside tuning
            # and for the single-rank degenerate path.
            self._profiling_scratch = None
            self.in_kernel_fc2_reduce = bool(in_kernel_fc2_reduce)
            # combine wire format: bf16 / 32e4m3xe8m0 (fp8) / 16e2m1xbf16 (fp4);
            # quantized requires separate-reduce + changes shared_workspace size,
            # so it is part of unique_id (compile + workspace cache key).
            self.combine_format = str(combine_format)

        def unique_id(self):
            # local_rank is intentionally excluded from the key. The kernel
            # requires every EP rank to run the SAME tactic (shared NVLink
            # dispatch barrier + peer-pointer layout), and the MERGE tuning
            # strategy merges per-shape timings by cache key across ranks --
            # so the key MUST be identical on every rank for the all-gather
            # merge to converge them onto one global-best tactic. Including
            # local_rank would give each rank a distinct key, leaving every
            # rank on its own best and breaking the same-tactic requirement.
            # world_size stays so single-rank (degenerate) and multi-rank
            # tuning never share entries.
            return (
                self.world_size,
                self.num_topk,
                self.num_experts_per_rank,
                self.hidden_size,
                self.intermediate_size_per_partition,
                self.expand_intermediate_size_per_partition,
                self.max_tokens_per_rank,
                str(self.output_dtype),
                self.apply_topk_in_fc1,
                self.gate_up_clamp,
                self.in_kernel_fc2_reduce,
                self.combine_format,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple]:
            del profile, kwargs
            # The activation token axis (inputs[0].shape[0]) drives the
            # token-bucket-dependent ``epi_flag_batch`` choice; pass it through
            # so the autotuner profiles ``(2, 4)`` for >8192 buckets and
            # ``(1, 1)`` otherwise.
            num_tokens = int(inputs[0].shape[0])
            candidates = enumerate_megamoe_candidate_tactics(num_tokens)
            # in-kernel-reduce (form-B) collapses the K routes UNSUMMED under a
            # bulk store (silent wrong output), and fp4 combine ("16e2m1...") cannot
            # scalar-deref sub-byte data in the UBLK smem scratch (the kernel
            # raises). Both force non-bulk. fp8 combine ("32e4m3xe8m0") is byte-legal
            # for the UBLK store, so fp8 bulk is left tunable here.
            # ``_unpack_tactic(t)[5]`` is ``use_bulk_fc2_store``.
            if self.in_kernel_fc2_reduce or self.combine_format.startswith("16e2m1"):
                candidates = [t for t in candidates if not _unpack_tactic(t)[5]]
            return candidates

        def _autotuner_inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            """Sanitize ONLY the autotuner-regenerated fake inputs.

            AutoTuner rebuilds fake tensors for the dynamic/constraint inputs
            (activation 0, activation_sf 1, topk_idx 2, topk_weights 3,
            combine_output 11) and passes the static inputs (the caller's real
            weights/scales at indices 4-10) through by reference. We fix only
            the regenerated tensors and leave 4-10 untouched: filling them
            would clobber the real per-expert weight SF / alphas and make every
            post-tuning forward emit all-zero output.

            - topk_idx: fresh random ints can be out of range and index OOB
              into the SMEM histogram / peer-pointer table, so rewrite to a
              valid round-robin.
            - activation_sf / topk_weights: fill with 1.0 (NOT 0) so the kernel
              does real, NaN-free work; SF==0 degenerates the GEMMs and gives
              unrealistic tactic timing.
            """
            inputs = list(inputs)
            total_experts = self.num_experts_per_rank * self.world_size
            if total_experts <= 0:
                return inputs

            # topk_idx (inputs[2]) is consumed LOCALLY (not peer-mapped), so it
            # stays a regenerated tensor; just make the fake ids a valid
            # round-robin so they never index OOB into the expert tables.
            topk_idx = inputs[2]
            if isinstance(topk_idx, torch.Tensor) and topk_idx.dim() == 2:
                T, K = topk_idx.shape
                valid = (
                    torch.arange(
                        T * K,
                        dtype=topk_idx.dtype,
                        device=topk_idx.device,
                    )
                    % total_experts
                ).view(T, K)
                topk_idx.copy_(valid)

            scratch = self._profiling_scratch
            if scratch is not None:
                # Multi-rank profiling: route the CROSS-RANK inputs through the
                # symmetric scratch buffer (sliced to the regenerated token
                # count) so the kernel's single peer_rank_ptr_mapper resolves to
                # a valid symmetric region. The AutoTuner's fresh non-symmetric
                # tensors are discarded for these indices; the staging buffer is
                # never touched. Fake values mirror the legacy sanitizer
                # (round-robin topk above, SF/weights == 1.0) so the GEMM does
                # NaN-free, representative work.
                m = int(inputs[0].shape[0])
                act = scratch.activation[:m]
                act.copy_(inputs[0].view(torch.uint8))
                inputs[0] = act.view(inputs[0].dtype)

                sf = scratch.activation_sf[:m]
                sf.fill_(0x38)  # raw byte == FP8 1.0
                inputs[1] = sf if inputs[1].dtype == torch.uint8 else sf.view(inputs[1].dtype)

                w = scratch.topk_weights[:m]
                w.fill_(1.0)
                inputs[3] = w

                comb = scratch.combine_output[:m]
                comb.zero_()  # form-B accumulates onto live rows; form-A overwrites
                inputs[11] = comb
                return inputs

            # Single-rank / scratch-disabled (legacy path; no symmetric
            # requirement): sanitize the regenerated tensors in place.
            # inputs[1] is already float8_e4m3fn, where fill_(1.0) writes the
            # FP8 value 1.0 (byte 0x38); the uint8 fallback writes that raw byte
            # directly (do NOT fill_(0x38) on the FP8 view -- that is 56.0).
            activation_sf = inputs[1]
            if isinstance(activation_sf, torch.Tensor):
                if activation_sf.dtype == torch.float8_e4m3fn:
                    activation_sf.fill_(1.0)  # FP8 1.0 (exact, byte 0x38)
                else:
                    activation_sf.view(torch.uint8).fill_(0x38)  # raw byte = FP8 1.0
            topk_weights = inputs[3]
            if isinstance(topk_weights, torch.Tensor):
                topk_weights.fill_(1.0)

            return inputs

        def get_tuning_config(self) -> TuningConfig:
            """Tuning config: only the activation token-axis is dynamic.

            Constraints chain activation_sf / topk_idx / topk_weights /
            combine_output to the activation token count, so the
            autotuner does not double-enumerate tile sizes for
            independent token axes.

            Cached at class scope keyed on ``unique_id()``: every field below
            is constant except ``inputs_pre_hook`` and ``tune_max_num_tokens``.
            ``inputs_pre_hook`` is a bound method that only reads
            ``num_experts_per_rank`` / ``world_size`` and ``tune_max_num_tokens``
            is ``max_tokens_per_rank`` -- all three are part of ``unique_id()``,
            so two runners that map to the same key share a functionally
            identical config. (Mirrors the ``tuning_config_cache`` pattern of
            the CuteDSL grouped-gemm runners.)
            """
            key = self.unique_id()
            cached = self.__class__.tuning_config_cache.get(key)
            if cached is not None:
                return cached

            # Constraints reuse the runner's own shape-derivation rules
            # (the activation token count drives every other tensor's
            # leading axis). We pass shape-derivation lambdas that pull
            # the runtime ``num_tokens`` from input[0].
            def _num_tokens(shapes: List[torch.Size]) -> int:
                return shapes[0][0]

            config = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(1, 0, _num_tokens),  # activation_sf
                    ConstraintSpec(2, 0, _num_tokens),  # topk_idx
                    ConstraintSpec(3, 0, _num_tokens),  # topk_weights
                    # combine_output moved from idx 8 -> 11 after inserting
                    # fc1_alpha(8) / fc2_alpha(9) / fc1_norm_const(10).
                    ConstraintSpec(11, 0, _num_tokens),  # combine_output
                ),
                inputs_pre_hook=self._autotuner_inputs_pre_hook,
                # MUST stay False for multi-rank MegaMoE. ``use_cold_l2_cache``
                # makes the AutoTuner ``clone()`` the profiling inputs into extra
                # buffers for L2-cold rotation. The cross-rank inputs
                # (activation / activation_sf / topk_weights / combine_output)
                # are routed by ``_autotuner_inputs_pre_hook`` to a SYMMETRIC
                # scratch; the clones are ordinary NON-symmetric tensors, so the
                # kernel's peer-pointer mapping (base = clone ptr + the scratch's
                # peer_offsets) resolves outside any symmetric region and a
                # dispatch/combine peer access lands out of bounds (CUDA illegal
                # memory access during autotuning, e.g. e4_k4 at the m=128
                # profiling bucket). Single-batch profiling keeps every launch on
                # the real symmetric scratch.
                use_cold_l2_cache=False,
                # Pin the num_tokens bucket ladder to the per-rank token
                # ceiling instead of letting it depend on whatever
                # num_tokens the autotuner warmup forward happened to feed
                # (which is clamped by available KV cache and may be a
                # non-power-of-2 / lower value). Mirrors trtllm-gen's
                # ``tune_max_num_tokens=self.max_num_tokens`` (the runner-level
                # name for that ceiling here is ``max_tokens_per_rank``).
                tune_max_num_tokens=self.max_tokens_per_rank,
                # CUDA Graph capture cannot reproduce MegaMoE's runtime
                # peer-pointer table / dispatch-counter view and would
                # spin inside the captured barrier when the autotuner's
                # L2-cache buffers rotate. Plain repeat-loop profiling
                # is correct and only marginally slower.
                use_cuda_graph=False,
                # MegaMoE fuses cross-rank dispatch + FC1 + FC2 + combine into
                # one kernel: ranks couple at the dispatch count-exchange and
                # combine NVLink barriers, and FC1 token tiles depend on peers'
                # pull progress. Faithful per-tactic timing therefore requires
                # ALL EP ranks to run the SAME tactic in lockstep and time it
                # together. MERGE does exactly that: it does NOT split the
                # candidate list across ranks (every rank profiles the full
                # list, same tactic per step), inserts a tp_barrier before each
                # measurement, and disables the short-profile heuristic so every
                # rank issues an identical number of launches (otherwise the
                # all-rank NVLink barriers would desync/hang). It then all-gathers
                # the per-shape timings so every rank converges onto one
                # global-best tactic -- which the kernel requires, since all EP
                # ranks must run the same compiled tactic for the NVLink dispatch
                # barrier and peer-pointer mapping to line up. PARALLEL (used by
                # plain multi-rank CuteDSL GEMMs that have no in-kernel cross-rank
                # coupling) would split tactics across ranks, contaminating the
                # comm-coupled timing and risking barrier desync, so it is NOT
                # used here.
                distributed_tuning_strategy=DistributedTuningStrategy.MERGE,
            )
            self.__class__.tuning_config_cache[key] = config
            return config

        def _build_kernel(self, tactic: Tuple):
            (
                mma_tiler,
                cluster_shape,
                group_hint,
                load_balance_mode,
                token_back_mode,
                use_bulk_fc2_store,
                flag_batch,
                epi_flag_batch,
            ) = _unpack_tactic(tactic)
            common = dict(
                mma_tiler_mnk=tuple(mma_tiler),
                cluster_shape_mnk=tuple(cluster_shape),
                # use_2cta_instrs is DERIVED from M, not a tactic field.
                use_2cta_instrs=bool(mma_tiler[0] == 256),
                group_hint=int(group_hint),
                token_padding_block=64,
                sf_padding_block=SfPaddingBlock,
                load_balance_mode=str(load_balance_mode),
                static_expert_shape=(
                    self.num_experts_per_rank,
                    self.expand_intermediate_size_per_partition,
                    self.hidden_size,
                ),
                world_size=self.world_size,
                num_topk=self.num_topk,
                max_tokens_per_rank=self.max_tokens_per_rank,
                hidden=self.hidden_size,
                fc2_output_dtype=cutlass.BFloat16,
                # in_kernel_fc2_reduce is a backend functional config (form-A/B),
                # carried in unique_id, NOT a tactic field.
                in_kernel_fc2_reduce=self.in_kernel_fc2_reduce,
                token_back_mode=str(token_back_mode),
                non_ubulk_fc2_store=(not bool(use_bulk_fc2_store)),
                flag_batch=int(flag_batch),
                epi_flag_batch=tuple(epi_flag_batch),
                apply_topk_in_fc1=self.apply_topk_in_fc1,
                gate_up_clamp=self.gate_up_clamp,
                **_LOCKED_KERNEL_KWARGS,
            )
            kernel_cls, CombineFormat = _import_megamoe_kernel()
            return kernel_cls(combine_format=CombineFormat.parse(self.combine_format), **common)

        def _tactic_cache_key(self, tactic: Tuple) -> Tuple:
            """Hashable cache key over ``unique_id()`` + the FULL 8-tuple.

            Both the compile cache and the local-workspace cache MUST key on
            the full tactic: the local workspace size varies with the tactic
            (token_back_mode != 'epi_warps' adds a large fc2_output_workspace +
            fc2_done_counter; atomic_counter adds token_back_schedule_counter),
            so an under-keyed workspace cache would reuse an epi-sized buffer
            for a reuse/standalone tactic -> OOB / 100% SM hang. ``unique_id()``
            already carries apply_topk_in_fc1 / gate_up_clamp / in_kernel_fc2_reduce
            (form-A and form-B never share a cache entry).
            """
            (
                mma_tiler,
                cluster_shape,
                group_hint,
                load_balance_mode,
                token_back_mode,
                use_bulk_fc2_store,
                flag_batch,
                epi_flag_batch,
            ) = _unpack_tactic(tactic)
            return (
                self.unique_id(),
                tuple(mma_tiler),
                tuple(cluster_shape),
                int(group_hint),
                str(load_balance_mode),
                str(token_back_mode),
                bool(use_bulk_fc2_store),
                int(flag_batch),
                tuple(epi_flag_batch),
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            *,
            tactic: Any = -1,
            peer_offsets: Optional[List[int]] = None,
            shared_workspace: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> None:
            del kwargs
            # Resolve fallback tactic. ``inputs`` is in hand here, so the
            # autotune-disabled / tactic=-1 fallback selects the token-aware
            # ``default_megamoe_tactic(num_tokens)`` directly (3-regime).
            if tactic == -1 or tactic is None:
                num_tokens = int(inputs[0].shape[0])
                # Non-bulk is a HARD requirement for (a) in-kernel reduce (form-B /
                # REDG: the fc2 epilogue selects bulk-store BEFORE in-kernel-reduce,
                # so a bulk tactic collapses the K routes UNSUMMED -> silent wrong
                # output) and (b) fp4 combine ("16e2m1...": the UBLK path cannot
                # scalar-deref sub-byte data -> kernel raises). fp8 bulk is legal,
                # but this non-autotune default keeps ALL quantized combine
                # non-bulk as a deterministic fallback; the autotuner
                # (get_valid_tactics) may recover fp8 bulk.
                if self.in_kernel_fc2_reduce or self.combine_format != "bf16":
                    tactic_t = _MEGAMOE_NONBULK_STANDALONE_TACTIC
                else:
                    tactic_t = default_megamoe_tactic(num_tokens)
            elif isinstance(tactic, list):
                tactic_t = tuple(tactic)
            else:
                tactic_t = tactic
            validate_megamoe_tactic(tactic_t)

            (
                activation,
                activation_sf,
                topk_idx,
                topk_weights,
                fc1_weight,
                fc1_weight_sf,
                fc2_weight,
                fc2_weight_sf,
                fc1_alpha,
                fc2_alpha,
                fc1_norm_const,
                combine_output,
            ) = inputs[:12]
            assert peer_offsets is not None, (
                "Sm100MegaMoENvfp4Runner.forward requires peer_offsets kwarg "
                "(length = world_size); single-rank degenerate mode passes "
                "(0,) * world_size."
            )
            assert len(peer_offsets) == self.world_size, (
                f"peer_offsets length {len(peer_offsets)} != world_size {self.world_size}"
            )

            kernel = self._build_kernel(tactic_t)

            # form-B (in_kernel_fc2_reduce) accumulates the top-k reduction into
            # ``combine_output`` via in-kernel atomic / cp.reduce on top of the
            # existing cell, so the combine buffer's live rows MUST start at 0.
            # The caller (MegaMoECuteDsl backend) zeros only ``[:num_tokens]``
            # before invoking this op -- we do NOT full-``max_T`` zero here
            # because the op has no live token count (the activation is always
            # padded to ``max_T``, so ``inputs[0].shape[0]`` is not the live
            # count) and a full zero is wasted work for small decode batches.
            # form-A overwrites cells and needs no per-launch zero.

            # ``local_workspace`` is per-rank private; cached across calls. The
            # cache key is the FULL 8-tuple because the local workspace SIZE
            # varies with the tactic (token_back_mode / load_balance add large
            # regions); an under-keyed cache would reuse a wrong-sized buffer.
            local_workspace = _get_or_alloc_local_workspace(
                kernel,
                cache_key=self._tactic_cache_key(tactic_t),
                device=activation.device,
            )
            # ``shared_workspace`` is peer-mapped (symmetric heap) for
            # multi-rank or local CUDA for the single-rank degenerate
            # path. The MegaMoECuteDsl backend supplies it; for the rare
            # call-site that omits it (legacy / tests) we fall back to a
            # local CUDA tensor sized by the kernel.
            if shared_workspace is None:
                if self.world_size > 1:
                    raise RuntimeError(
                        f"Sm100MegaMoENvfp4Runner: multi-rank "
                        f"(world_size={self.world_size}) requires the caller "
                        f"to supply a symmetric-memory shared_workspace via "
                        f"MegaMoeSymmMemProvider; got None."
                    )
                _, shared_bytes = kernel.get_workspace_sizes()
                shared_workspace = torch.empty(
                    shared_bytes, dtype=torch.uint8, device=activation.device
                )
            # Workspaces embed atomic counters / signals that must start at 0.
            #
            # Single-rank (degenerate, no peer access): a full per-launch zero
            # of both workspaces is safe and cheap.
            #
            # Multi-rank EP: ``shared_workspace`` is peer-mapped and is already
            # zeroed once under lockstep at provider rendezvous
            # (``MegaMoeSymmMemProvider``: ``_buf.zero_()`` before
            # ``symm_mem.rendezvous``). Re-zeroing it here per launch RACES a
            # peer rank's in-kernel dispatch barrier write into this rank's
            # ``nvlink_barrier_signal``: a fast peer ``red_add(+1)``s our slot,
            # then our late ``zero_()`` wipes it, so the barrier never reaches
            # ``world_size`` and the whole grid deadlocks. The symmetric
            # workspace's peer-written count regions (expert_recv_count[_sum])
            # are instead reset device-side by the kernel's
            # ``tail_reset_shared_counters``, ``nvlink_barrier_signal``
            # self-primes (phase-flip), and ``src_token_topk_idx`` is
            # overwritten by dispatch each launch -- so the shared workspace
            # needs no per-launch host zero at all.
            #
            # In-kernel NVLink-barrier phase coherence. The kernel's
            # sense-reversing dispatch barrier pairs a PERSISTED rank-local
            # nvlink_barrier_counter (outside the device tail-reset prefix)
            # with a PERSISTED shared nvlink_barrier_signal. Two regimes:
            #  * SAME tactic (steady state): the pair rides across launches;
            #    the kernel tail-resets its other counters device-side, so no
            #    host work at all.
            #  * TACTIC CHANGE (autotune candidate boundary, profiling->real
            #    transition): local counter and shared signal can be phase-
            #    DECOUPLED -> the next dispatch barrier spins forever. Reset
            #    both to phase 0 under a cross-rank fence, ONCE per tactic.
            # The fence does a host sync + collective barrier: ILLEGAL under
            # CUDA-graph capture (the eager warmup pre-pass fences first, so
            # capture skips it) and valid only for pure DEP (world == EP group,
            # see the guard below).
            _capturing = torch.cuda.is_current_stream_capturing()
            # Fence ONCE per (local_ws, shared_ws, world) key via the
            # module-global ``_MEGAMOE_FENCED_KEYS`` (the runner is rebuilt
            # every op call, so per-runner state cannot carry this). Each
            # autotune candidate has a distinct local_ws and re-fences once on
            # its warmup launch (never in a timed window); captured forwards
            # reuse an already-fenced workspace and skip.
            _fkey = (
                int(local_workspace.data_ptr()),
                int(shared_workspace.data_ptr()),
                int(self.world_size),
            )
            if (_fkey not in _MEGAMOE_FENCED_KEYS) and not _capturing:
                import torch.distributed as _dist

                _sig_off = int(kernel._shared_offsets["nvlink_barrier_signal"])
                _sig_n = int(kernel._shared_region_by_name["nvlink_barrier_signal"].nbytes)
                _bc_off = int(kernel._local_offsets["nvlink_barrier_counter"])
                _bc_n = int(kernel._local_region_by_name["nvlink_barrier_counter"].nbytes)
                _have_dist = _dist.is_available() and _dist.is_initialized()
                # The fence uses the default (WORLD) process group -- correct ONLY
                # for pure DEP (world == EP group == moe_ep_size). Under TP x EP / PP
                # the WORLD barrier spans ranks that never reach this MoE fence ->
                # deadlock; fail loud here instead of hanging. (Threading the EP
                # subgroup is the general fix.)
                if _have_dist and _dist.get_world_size() != int(self.world_size):
                    # Not an assert: this guard must survive ``python -O`` -- a
                    # stripped check would let the WORLD barrier deadlock under
                    # TP x EP / PP instead of failing loud.
                    raise RuntimeError(
                        "MegaMoE-CuteDSL barrier-reset fence uses the WORLD process "
                        "group, valid only for pure DEP (world == EP). Detected WORLD="
                        f"{_dist.get_world_size()} != EP={self.world_size} (TP x EP / "
                        "PP) -- fence must run on the EP subgroup."
                    )
                torch.cuda.current_stream().synchronize()
                if _have_dist:
                    _dist.barrier()
                # Reset BOTH sides of the persisted dispatch barrier to a common
                # phase 0. The local ``nvlink_barrier_counter`` is NOT always fresh
                # here: at the autotune profiling->real transition the winning
                # tactic's local is cache-reused with an ALREADY-ADVANCED counter
                # (profiling runs on a separate symmetric scratch, so the real launch
                # has a new shared_ptr -> a new fkey fires the fence on the advanced
                # local). Zero the tiny counter slice (NOT the multi-GiB workspace)
                # so it matches the freshly-zeroed signal; the bulk data stays uninit.
                local_workspace[_bc_off : _bc_off + _bc_n].zero_()
                shared_workspace[_sig_off : _sig_off + _sig_n].zero_()
                # The zeros above are stream-async. The post-zero dist.barrier()
                # is NOT guaranteed on-device-ordered after them, so once a peer
                # clears the barrier and launches, its in-kernel NVLink signal
                # WRITE can be CLOBBERED by our still-pending zero -> our dispatch
                # barrier spins forever. Sync so the zeros LAND before the
                # cross-rank barrier releases peers to launch signal-writing
                # kernels. Fires once per workspace, so it is not a hot-path cost.
                torch.cuda.current_stream().synchronize()
                if _have_dist:
                    _dist.barrier()
                _MEGAMOE_FENCED_KEYS.add(_fkey)
            # else: SAME tactic -- the kernel's device-side tail self-reset suffices.

            # === cute.compile (JIT) launch -- kernel-team mega_runner ABI ===
            # Inputs -> cute.Tensor (from_dlpack + mark_layout_dynamic); the opaque
            # uint8 workspaces -> cute.Pointer (make_ptr), NOT cute.Tensor: a
            # cute.Tensor's 32-bit memref shape field overflows once the internal
            # combine staging pushes shared_workspace past 2 GiB (the kernel
            # addresses workspaces by raw base + Int64 byte offset). SymBufferHost
            # carries the peer mapper.
            _to_cute, _to_cute_ptr, SymBufferHost, cute, cutlass_utils = _cute_launch_helpers()

            # ``combine_output`` is (max_T, 1, hidden) (provider combine_k=1); the
            # unified 2D ``output_activation`` (max_T, hidden) is a free reshape
            # over the same storage. Weights present K stride-1 via a transpose
            # VIEW (DLPack carries the strides); do NOT ``.contiguous()``.
            output_activation = combine_output.reshape(combine_output.shape[0], self.hidden_size)
            runtime_kwargs = dict(
                activation=_to_cute(activation),
                activation_sf=_to_cute(activation_sf),
                topk_idx=_to_cute(topk_idx),
                topk_weights=_to_cute(topk_weights),
                fc1_weight=_to_cute(fc1_weight.transpose(1, 2)),
                fc1_weight_sf=_to_cute(fc1_weight_sf),
                fc2_weight=_to_cute(fc2_weight.transpose(1, 2)),
                fc2_weight_sf=_to_cute(fc2_weight_sf),
                fc1_alpha=_to_cute(fc1_alpha, assumed_align=4),
                fc2_alpha=_to_cute(fc2_alpha, assumed_align=4),
                fc1_norm_const=_to_cute(fc1_norm_const, assumed_align=4),
                output_activation=_to_cute(output_activation),
                local_workspace=_to_cute_ptr(local_workspace),
                shared_workspace=_to_cute_ptr(shared_workspace),
                peer_rank_ptr_mapper_host=SymBufferHost(
                    base_addr=int(activation.data_ptr()),
                    offsets=tuple(int(off) for off in peer_offsets),
                    rank_idx=int(self.local_rank),
                    num_max_ranks=int(self.world_size),
                ),
                stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
            cache_key = self._tactic_cache_key(tactic_t)
            compiled = self.__class__.kernel_cache.get(cache_key)
            if compiled is None:
                # ``max_active_clusters`` is a compile-time Constexpr (baked in,
                # omitted at launch); cluster_size = cluster_shape M*N.
                _cluster = _unpack_tactic(tactic_t)[1]
                _max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
                    int(_cluster[0]) * int(_cluster[1])
                )
                compiled = cute.compile(
                    kernel, max_active_clusters=_max_active_clusters, **runtime_kwargs
                )
                self.__class__.kernel_cache[cache_key] = compiled
            compiled(**runtime_kwargs)
            return combine_output

    # ----- torch op ---------------------------------------------------------

    @torch.library.custom_op(
        "trtllm::cute_dsl_megamoe_nvfp4_blackwell",
        mutates_args=("combine_output", "shared_workspace"),
        device_types="cuda",
    )
    def cute_dsl_megamoe_nvfp4_blackwell(
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        peer_offsets: List[int],
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        in_kernel_fc2_reduce: bool = False,
        combine_format: str = "bf16",
        num_tokens: int = -1,
    ) -> None:
        """Run the fused MegaMoE CuteDSL NVFP4 kernel.

        Inputs are pre-staged by the caller (the ``MegaMoECuteDsl``
        backend in ``mega_moe_cute_dsl.py``). The op runs AutoTuner once
        per call to pick the best tactic for the current shape and
        invokes the runner.

        ``shared_workspace`` MUST be a symmetric-heap tensor for
        ``world_size > 1`` (use :class:`MegaMoeSymmMemProvider`); a
        local CUDA tensor is acceptable for the single-rank degenerate
        path. ``combine_output`` is mutated in place; the op does not
        return it because torch custom_op forbids the return value from
        aliasing any mutated input.

        ``in_kernel_fc2_reduce`` selects the output reduction form:
        ``False`` (default, form-A) keeps ``combine_output`` at
        ``(T, num_topk, hidden)`` and the caller does the host-side
        ``.sum(dim=1)``; ``True`` (form-B) uses ``(T, 1, hidden)`` (zeroed
        per launch) and the kernel folds the top-k reduction in-kernel, so
        the output is NON-deterministic (in-kernel float accumulation order).
        ``token_back_mode`` / ``use_bulk_fc2_store`` / ``flag_batch`` /
        ``epi_flag_batch`` are perf knobs selected by the autotuner tactic,
        not op arguments.
        """
        sm_version = get_sm_version()
        if sm_version not in (100, 103):
            raise RuntimeError(
                f"cute_dsl_megamoe_nvfp4_blackwell requires SM 100 (B200) or "
                f"SM 103 (B300); got SM {sm_version}."
            )

        # Live-token trim: the caller stages ``combine_output`` at the full
        # bucket size (activation is padded to ``max_tokens_per_rank``), but
        # the TopkReduce pass sizes its grid and bounds-guard from THIS
        # tensor's dim0. Slicing to the live token count skips the dead
        # reduce work on rows [num_tokens, max_T) -- worth ~5-8us/layer at
        # decode. ``num_tokens < 0`` (or an oversized value, e.g. from the
        # autotuner's regenerated profiling tensors) keeps the full bucket.
        # ``num_tokens == 0`` (a zero-token rank in a lockstep chunk, launched
        # anyway so peers can cross the in-kernel NVLink barrier) ALSO keeps
        # the full bucket: a 0-row slice would make TopkReduce compute and
        # launch a zero grid.
        if num_tokens > 0:
            combine_output = combine_output[: min(num_tokens, combine_output.shape[0])]

        runner = Sm100MegaMoENvfp4Runner(
            world_size=world_size,
            local_rank=local_rank,
            num_topk=num_topk,
            num_experts_per_rank=num_experts_per_rank,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            expand_intermediate_size_per_partition=expand_intermediate_size_per_partition,
            max_tokens_per_rank=max_tokens_per_rank,
            output_dtype=combine_output.dtype,
            apply_topk_in_fc1=apply_topk_in_fc1,
            gate_up_clamp=gate_up_clamp,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            combine_format=combine_format,
        )
        inputs = [
            activation,
            activation_sf,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            combine_output,
        ]
        tuner = AutoTuner.get()
        if os.environ.get("MEGAMOE_AUTOTUNE", "0") != "1":
            # Default: BYPASS the AutoTuner candidate sweep and use the token-bucket
            # heuristic tactic (one forward, no tactic changes). The sweep IS
            # supported via the tactic-change barrier fence in ``forward`` (reset
            # (counter, signal) to a common phase under a cross-rank barrier on every
            # tactic boundary) -- enable it with ``MEGAMOE_AUTOTUNE=1`` to pick the
            # best tactic per shape. Kept OFF by default as the conservative path.
            best_tactic = -1
        else:
            # AutoTuner profiling must use a SYMMETRIC buffer for the cross-rank
            # inputs. The backend hands off a transient symmetric scratch (its own
            # peer_offsets + shared_workspace); the profiling pre-hook routes the
            # cross-rank inputs through it so the real staging buffer is never
            # corrupted. Single-rank / no-scratch falls back to the staging buffer.
            prof_scratch = _ACTIVE_MEGAMOE_PROFILING_SCRATCH if world_size > 1 else None
            if prof_scratch is not None:
                runner._profiling_scratch = prof_scratch
                prof_peer_offsets = list(prof_scratch.peer_offsets)
                prof_shared_workspace = prof_scratch.shared_workspace
            else:
                runner._profiling_scratch = None
                prof_peer_offsets = peer_offsets
                prof_shared_workspace = shared_workspace
            try:
                _, best_tactic = tuner.choose_one(
                    "trtllm::cute_dsl_megamoe_nvfp4_blackwell",
                    [runner],
                    runner.get_tuning_config(),
                    inputs,
                    peer_offsets=prof_peer_offsets,
                    shared_workspace=prof_shared_workspace,
                )
            finally:
                runner._profiling_scratch = None
        # Real run always uses the caller's staging buffer + peer_offsets.
        runner(
            inputs,
            tactic=best_tactic,
            peer_offsets=peer_offsets,
            shared_workspace=shared_workspace,
        )

    @torch.library.register_fake("trtllm::cute_dsl_megamoe_nvfp4_blackwell")
    def _(
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        peer_offsets: List[int],
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        in_kernel_fc2_reduce: bool = False,
        combine_format: str = "bf16",
        num_tokens: int = -1,
    ) -> None:
        return None
