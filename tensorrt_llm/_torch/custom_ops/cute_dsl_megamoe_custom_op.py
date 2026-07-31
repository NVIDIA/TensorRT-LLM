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
  ``run_moe``. ``tactic_autotune=True`` (bench-only, enabled via the
  ``MEGAMOE_TACTIC_AUTOTUNE=1`` env var read by the backend) runs
  ``AutoTuner.choose_one`` per call; the default bypasses the AutoTuner
  and uses the deterministic token-bucket heuristic tactic.

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
import weakref
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
    """Lazy import so non-SM100 / no-cutlass-dsl envs can still import this module."""
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

# (local_ws_ptr, shared_ws_ptr, world_size) keys whose in-kernel barrier
# (counter, signal) was fenced-reset. Module-global because the runner is
# rebuilt every op call; the fence must fire ONCE per key (see forward()).
_MEGAMOE_FENCED_KEYS: set = set()

# fkey -> (weakref(local UntypedStorage), weakref(shared UntypedStorage))
# recorded when the fence ran. Fence keys are RAW pointers, so a new
# allocation on a recycled VA (ABA) is only detectable via these storage refs
# (identity-stable, unlike the provider's throwaway per-call views).
_MEGAMOE_FENCED_KEY_WS_REFS: dict = {}


def _megamoe_prune_fenced_keys(dead_keys) -> None:
    """Drop ``dead_keys`` from ``_MEGAMOE_FENCED_KEYS`` and their recorded
    workspace storage weakrefs (single helper so the two cannot drift)."""
    _MEGAMOE_FENCED_KEYS.difference_update(dead_keys)
    for _key in dead_keys:
        _MEGAMOE_FENCED_KEY_WS_REFS.pop(_key, None)


def _megamoe_fence_key_live(fkey: Tuple[int, int, int]) -> bool:
    """True iff ``fkey`` was fenced AND both recorded workspace storages are
    still alive.

    A dead weakref means this pointer is a NEW allocation on a recycled VA:
    skipping its fence would pair a virgin barrier side with the other side's
    persisted phase (dispatch barrier spins forever). A new workspace
    re-fences on EVERY rank (dead-ref prune or plain set miss), so the
    collective fence cannot desync. Host-only; capture-safe.
    """
    if fkey not in _MEGAMOE_FENCED_KEYS:
        return False
    refs = _MEGAMOE_FENCED_KEY_WS_REFS.get(fkey)
    if refs is not None and refs[0]() is not None and refs[1]() is not None:
        return True
    _megamoe_prune_fenced_keys({fkey})
    return False


# Shared-workspace pointers whose pairings were replayed under CUDA-graph
# capture. A LATER eager fence must never zero such a shared side: captured
# pairings cannot re-fence -> phase-decoupled barrier -> silent all-rank hang.
_MEGAMOE_CAPTURED_SHARED_PTRS: set = set()

# True once ANY MegaMoE launch was captured into a CUDA graph. Before that
# the whole local-workspace cache is safely evictable; afterwards only
# tuning-latched entries may go. See ``release_megamoe_profiling_scratch``.
_MEGAMOE_GRAPH_CAPTURE_SEEN: bool = False


# ---------------------------------------------------------------------------
# Tactic representation (v3: 8-tuple perf knobs)
# ---------------------------------------------------------------------------
#
# A tactic is a tuple of JSON-friendly primitives (lists / ints / bools /
# strings / nested tuples) so it round-trips through ``json.dumps`` *and*
# ``eval(repr(tactic))`` (required by ``TunableRunner`` cache serialization).
# Order matches the kernel constructor kwargs.
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
#   use_2cta_instrs      = (mma_tiler_mnk[0] == 256), derived in _build_kernel.
#   in_kernel_fc2_reduce -- functional config in runner unique_id, NOT a perf knob.
#
# Tuple wrapping makes the tactic hashable, which AutoTuner needs for the
# tactics cache.

_TACTIC_LEN = 8

# Kernel-side ceiling: ``flag_batch`` is hard-checked ``[1, 32]`` and
# ``epi_flag_batch`` entries are SILENTLY clamped to ``[1, 32]``; reject > 32
# here so a hand-supplied tactic fails fast instead of silently running at 32.
_FLAG_BATCH_MAX = 32


def _unpack_tactic(tactic: Tuple) -> Tuple:
    """Return the tactic's 8 fields in canonical order -- the single source of
    truth for the field layout; every consumer unpacks through here. Plain
    positional unpack, no validation (see :func:`validate_megamoe_tactic`)."""
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
    """Deterministic token-bucket fallback tactic (autotune disabled /
    cache miss / tactic=-1); never profiled by the autotuner."""
    if num_tokens <= 1024:
        # decode winner: N128 only helps epi_warps + bulk.
        return ([256, 128, 256], [2, 1, 1], 512, "static", "epi_warps", True, 1, (1, 1))
    if num_tokens <= 8192:
        return ([256, 256, 256], [2, 1, 1], 512, "static", "epi_warps", True, 4, (1, 1))
    # prefill: atomic_counter only helps the very large tail (>=16384).
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
# Curated tuning space (~36 candidates per token bucket), filtered through
# ``validate_megamoe_tactic``. MUST be identical on every EP rank: the MERGE
# tuning sweep runs the candidate list in lockstep across ranks, so there is
# deliberately NO runtime knob selecting a different space.
# ---------------------------------------------------------------------------

# token_back_mode -> the only legal use_bulk_fc2_store (bulk binds to epi_warps).
_TOKEN_BACK_STORE_BINDING: dict = {
    "epi_warps": True,
    "reuse_dispatch_warps": False,
    "standalone_warps": False,
}

# (mma_tiler_mnk, cluster_shape_mnk) geometries. use_2cta is derived (M==256).
# N64, M128/1-CTA and cluster_m4 never won a bucket upstream.
_GEOMETRIES: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((256, 256, 256), (2, 1, 1)),
    ((256, 128, 256), (2, 1, 1)),
)

# token_back modes scanned (standalone_warps dropped: never a sole winner).
_TOKEN_BACK_MODES: Tuple[str, ...] = ("epi_warps", "reuse_dispatch_warps")

# Load-balance modes supported by the integrated fused FC12 path (see
# ImplDesc.__post_init__ in fc1_fc2_fuse_sched.py).
# ``clc`` is intentionally excluded -- it routes through a separate
# scheduler class not wired through the fused FC12 kernel here.
_LOAD_BALANCE_MODE_CANDIDATES: Tuple[str, ...] = ("static", "atomic_counter")

# Quantized combine (fp8/fp4) and form-B reject the bulk fc2 store at kernel
# construction; this is their deterministic default and sizing-probe tactic.
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

# group_hint saturates ~512; omission (-> ~74) is worst.
_GROUP_HINTS: Tuple[int, ...] = (512, 1024)

_FLAG_BATCHES: Tuple[int, ...] = (1, 4, 8)

# epi_flag_batch is fixed per token bucket, NOT a free scan axis; see
# ``_epi_flag_batch_for_tokens``.
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

    if (not isinstance(group_hint, int)) or isinstance(group_hint, bool) or group_hint < 512:
        raise ValueError(f"group_hint must be an int >= 512, got {group_hint!r}.")

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
    if use_bulk_fc2_store and token_back_mode != "epi_warps":  # nosec B105
        raise ValueError(
            f"use_bulk_fc2_store=True requires token_back_mode='epi_warps', got "
            f"{token_back_mode!r} (non-epi token-back forces non-bulk fc2 store)."
        )
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
    if token_back_mode == "standalone_warps" and flag_batch != 1:  # nosec B105
        raise ValueError(
            f"token_back_mode='standalone_warps' requires flag_batch == 1, got {flag_batch}."
        )

    if (not isinstance(epi_flag_batch, (tuple, list))) or len(epi_flag_batch) != 2:
        raise ValueError(f"epi_flag_batch must be a 2-tuple (fc1, fc2), got {epi_flag_batch!r}.")
    for v in epi_flag_batch:
        if (not isinstance(v, int)) or isinstance(v, bool) or not (1 <= v <= _FLAG_BATCH_MAX):
            raise ValueError(
                f"epi_flag_batch entries must be ints in [1, {_FLAG_BATCH_MAX}] "
                f"(epilogue clamp range), got {epi_flag_batch!r}."
            )


def enumerate_megamoe_candidate_tactics(num_tokens: int) -> List[Tuple]:
    """Return the curated candidate tactic list for the current token bucket.

    Cartesian product over the curated dimension ranges, filtered through
    ``validate_megamoe_tactic``. ``epi_flag_batch`` is fixed per token
    bucket (see ``_epi_flag_batch_for_tokens``).
    """
    epi_flag_batch = _epi_flag_batch_for_tokens(num_tokens)

    candidates: List[Tuple] = []
    for mma_tiler, cluster_shape in _GEOMETRIES:
        for token_back_mode in _TOKEN_BACK_MODES:
            use_bulk = _TOKEN_BACK_STORE_BINDING[token_back_mode]
            for load_balance_mode in _LOAD_BALANCE_MODE_CANDIDATES:
                for group_hint in _GROUP_HINTS:
                    for flag_batch in _FLAG_BATCHES:
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
    # Cache keys allocated during the autotune sweep. Each candidate has a
    # DISTINCT key (workspace SIZE varies with the tactic) and a multi-GiB
    # workspace; persisting all would OOM, so each candidate's workspace is
    # freed when the next one allocates (only the winner is reused).
    _MEGAMOE_TUNING_WORKSPACE_KEYS: set = set()

    @functools.lru_cache(maxsize=1)
    def _cute_launch_helpers():
        """Import the cutlass-dsl launch surface once per process and build the
        tensor/pointer converters (hoisted out of forward(); lazy so the
        module imports without cutlass-dsl / a GPU)."""
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

    def _evict_latched_tuning_workspaces() -> None:
        """Pop every tuning-latched local workspace and prune its fence keys.

        Fence keys hold the raw local ptr; if the allocator hands the freed
        block to the next candidate, a stale key would ABA-skip the
        barrier-reset fence and hang the dispatch barrier mid-sweep.
        """
        for stale_key in _MEGAMOE_TUNING_WORKSPACE_KEYS:
            stale_ws = _MEGAMOE_LOCAL_WORKSPACE_CACHE.pop(stale_key, None)
            if stale_ws is not None:
                _stale_ptr = int(stale_ws.data_ptr())
                _megamoe_prune_fenced_keys({k for k in _MEGAMOE_FENCED_KEYS if k[0] == _stale_ptr})
        _MEGAMOE_TUNING_WORKSPACE_KEYS.clear()

    def _get_or_alloc_local_workspace(
        kernel, cache_key: Tuple, device: torch.device, latch_in_tuning_mode: bool = True
    ) -> torch.Tensor:
        cached = _MEGAMOE_LOCAL_WORKSPACE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Cache MISS = new (tactic, shape): release any workspace latched for
        # a previously-profiled candidate (never relaunched; profiling is
        # eager, so no captured graph holds the buffer). Within one
        # candidate's multi-launch profiling the key HITs, so nothing is
        # freed mid-candidate; outside tuning the latch set is empty.
        if _MEGAMOE_TUNING_WORKSPACE_KEYS:
            _evict_latched_tuning_workspaces()
        local_bytes, _ = kernel.get_workspace_sizes()
        # Only the atomic counters need zeroing (a stray negative byte hangs
        # spin_wait at 100% SM); the multi-GiB bulk is overwritten every launch
        # (full ``torch.zeros`` = huge wasted memset). Zero the leading counter
        # prefix plus the persisted ``nvlink_barrier_counter``, deliberately
        # OUTSIDE that prefix so the device tail-reset never touches it.
        local_workspace = torch.empty(local_bytes, dtype=torch.uint8, device=device)
        _lead = int(kernel.require_zero_workspace_leading_bytes[0])
        local_workspace[:_lead].zero_()
        _bc_off = int(kernel._local_offsets["nvlink_barrier_counter"])
        _bc_n = int(kernel._local_region_by_name["nvlink_barrier_counter"].nbytes)
        local_workspace[_bc_off : _bc_off + _bc_n].zero_()
        _MEGAMOE_LOCAL_WORKSPACE_CACHE[cache_key] = local_workspace
        # ``latch_in_tuning_mode=False`` (tactic_autotune opted OUT): launches
        # inside a global autotune() are real fallback-tactic runs, not sweep
        # candidates, so their workspaces must persist.
        if latch_in_tuning_mode and AutoTuner.get().is_tuning_mode:
            _MEGAMOE_TUNING_WORKSPACE_KEYS.add(cache_key)
        return local_workspace

    # ----- Symmetric-memory provider (NVSHMEM-equivalent) -------------------
    #
    # PyTorch's ``torch.distributed._symmetric_memory`` is an NVSHMEM-equivalent
    # symmetric-heap provider built on cuMem APIs. It exposes per-rank buffer
    # pointers (``handle.buffer_ptrs``) which we use to populate the
    # ``SymBufferHost(offsets, rank_idx, num_max_ranks)`` payload
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
        # (max_T, 1, hidden): both reduction forms collapse the top-k axis in-op.
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
            # Kernel output is unified (T, hidden), so the symmetric combine
            # region is (max_T, 1, hidden). combine_k MUST match the kernel
            # ``combine_output`` shape or the region is silently corrupted.
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
        combine_format: str,
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
            str(combine_format),
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
    # The kernel's SINGLE ``peer_rank_ptr_mapper`` requires ALL cross-rank
    # tensors to live in one symmetric allocation at consistent offsets; the
    # AutoTuner's regenerated NON-symmetric inputs would peer-map outside any
    # symmetric region (multi-rank IMA). So profiling runs on a SEPARATE
    # symmetric scratch (never the real staging buffer), shared across
    # same-layout layers and freed by ``release_megamoe_profiling_scratch()``.
    _MEGAMOE_PROFILING_SCRATCH_CACHE: dict = {}
    _ACTIVE_MEGAMOE_PROFILING_SCRATCH = None
    # Deferred variant: zero-arg callable returning MegaMoeSymmRegions, set
    # when the scratch is not allocated. The profiling pre-hook calls it only
    # when REAL profiling launches, so a tuning-mode cache HIT never pays the
    # multi-GiB allocation.
    _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY = None

    def get_megamoe_profiling_scratch(
        *,
        process_group,
        world_size: int,
        rank: int,
        hidden_size: int,
        max_tokens_per_rank: int,
        num_topk: int,
        output_dtype: torch.dtype,
        combine_format: str,
        shared_workspace_bytes: int,
    ):
        """Return a cached, transient symmetric scratch provider used ONLY for
        AutoTuner profiling; ``None`` for single-rank."""
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
            str(combine_format),
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
        """Set (``None`` clears) the :class:`MegaMoeSymmRegions` used for
        AutoTuner profiling on the current call (set by the backend around
        the op invocation; consumed inside ``choose_one``)."""
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH = regions

    def set_active_megamoe_profiling_scratch_factory(factory) -> None:
        """Set (``None`` clears) a zero-arg factory returning the profiling
        scratch regions; used instead of the eager setter when the scratch is
        not allocated. See ``_ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY``."""
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY = factory

    def release_megamoe_profiling_scratch() -> None:
        """Free all profiling-scratch symmetric buffers (call after warmup);
        profiling re-allocates lazily if it runs again."""
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH = None
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY = None
        # Prune fence keys pairing with the dying scratch shared buffers: a
        # future allocation on the recycled VA must re-fence (ABA).
        _dead_shared = {
            int(_p.get_regions().shared_workspace.data_ptr())
            for _p in _MEGAMOE_PROFILING_SCRATCH_CACHE.values()
        }
        if _dead_shared:
            _megamoe_prune_fenced_keys({k for k in _MEGAMOE_FENCED_KEYS if k[1] in _dead_shared})
        _MEGAMOE_PROFILING_SCRATCH_CACHE.clear()
        # EVICT stale local workspaces -- do not merely clear the latch: the
        # latch may hold the LAST LOSER, and a losing default's multi-GiB
        # workspace would shrink the KV-cache budget for the process lifetime.
        # Before the FIRST capture nothing holds raw pointers, so the whole
        # cache is evictable; after a capture only latched entries go.
        if not _MEGAMOE_GRAPH_CAPTURE_SEEN:
            for _key, _ws in list(_MEGAMOE_LOCAL_WORKSPACE_CACHE.items()):
                _ptr = int(_ws.data_ptr())
                _megamoe_prune_fenced_keys({k for k in _MEGAMOE_FENCED_KEYS if k[0] == _ptr})
                del _MEGAMOE_LOCAL_WORKSPACE_CACHE[_key]
            _MEGAMOE_TUNING_WORKSPACE_KEYS.clear()
        else:
            _evict_latched_tuning_workspaces()

    def reset_megamoe_workspace_state() -> None:
        """TEST/BENCH ONLY: drop ALL process-global MegaMoE workspace state.

        PRECONDITION: every CUDA graph that replayed a MegaMoE launch is
        destroyed and nothing will launch on the dropped workspaces. Fence /
        captured-ptr sets are cleared TOGETHER with the buffer caches (stale
        entries on recycled addresses would ABA-skip the fence or false-positive
        the capture guard). Must run in lockstep on ALL EP ranks.
        """
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH
        global _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY
        global _MEGAMOE_GRAPH_CAPTURE_SEEN
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH = None
        _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY = None
        _MEGAMOE_PROFILING_SCRATCH_CACHE.clear()
        _MEGAMOE_SYMM_PROVIDER_CACHE.clear()
        _MEGAMOE_LOCAL_WORKSPACE_CACHE.clear()
        _MEGAMOE_TUNING_WORKSPACE_KEYS.clear()
        _MEGAMOE_FENCED_KEYS.clear()
        _MEGAMOE_FENCED_KEY_WS_REFS.clear()
        _MEGAMOE_CAPTURED_SHARED_PTRS.clear()
        _MEGAMOE_GRAPH_CAPTURE_SEEN = False

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
        default 8-tuple tactic for the probe; the remaining kwargs are
        threaded only to satisfy the kernel ctor and match the real build.
        """

        if tactic is None:
            # Sizing is tactic-invariant, but quantized combine (fp8/fp4)
            # rejects the bulk fc2 store at kernel construction, so those
            # modes must probe with the non-bulk standalone tactic.
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
        # The probe MUST build the SAME kernel that runs (same combine_format):
        # otherwise the provider carves an undersized shared region and the
        # combine staging writes OOB (single-rank IMA; EP>1 looks like a hang).
        kernel_cls, CombineFormat = _import_megamoe_kernel()
        probe = kernel_cls(combine_format=CombineFormat.parse(combine_format), **common)
        _, shared_bytes = probe.get_workspace_sizes()
        return int(shared_bytes)

    def _megamoe_autotune_num_tokens(shapes: List[torch.Size]) -> int:
        """Shape-derivation rule: the activation token count (input[0] dim0)
        drives every other tensor's leading axis.

        Module-scope on purpose: ConstraintSpec hashes include the callable
        and the AutoTuner lru_caches on it; a per-call closure would defeat
        memoization and leak cache entries."""
        return shapes[0][0]

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
            tactic_autotune: bool = False,
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
            # Codegen-time modes: they change the generated kernel, so they are
            # part of ``unique_id`` (compile + workspace cache key), never
            # per-call runtime kwargs.
            self.apply_topk_in_fc1 = bool(apply_topk_in_fc1)
            self.gate_up_clamp = None if gate_up_clamp is None else float(gate_up_clamp)
            # Symmetric profiling scratch, set by the op around ``choose_one``
            # so the pre-hook routes cross-rank inputs through symmetric
            # memory; None outside tuning and for single-rank.
            self._profiling_scratch = None
            # Deferred scratch factory; the pre-hook materializes it on the
            # first REAL profiling launch (tuning-mode cache HITs never allocate).
            self._profiling_scratch_factory = None
            self.in_kernel_fc2_reduce = bool(in_kernel_fc2_reduce)
            # combine wire format: bf16 / 32e4m3xe8m0 (fp8) / 16e2m1xbf16 (fp4);
            # quantized changes the shared_workspace size, so part of unique_id.
            self.combine_format = str(combine_format)
            # Tactic-autotune opt-in (default OFF). Does NOT change the
            # generated kernel, so deliberately EXCLUDED from unique_id /
            # _tactic_cache_key: opted-in and opted-out runs share caches.
            self.tactic_autotune = bool(tactic_autotune)

        def unique_id(self):
            # local_rank is intentionally excluded: every EP rank must run the
            # SAME tactic and MERGE merges timings by cache key across ranks,
            # so the key MUST be rank-identical. world_size stays so
            # single-rank and multi-rank never share entries.
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
            num_tokens = int(inputs[0].shape[0])
            candidates = enumerate_megamoe_candidate_tactics(num_tokens)
            # Non-bulk is HARD for form-B (a bulk store collapses the K routes
            # UNSUMMED -> silent wrong output) and fp4 combine (UBLK cannot
            # scalar-deref sub-byte data -> kernel raises); fp8 bulk is legal
            # and stays tunable. ``_unpack_tactic(t)[5]`` is use_bulk_fc2_store.
            if self.in_kernel_fc2_reduce or self.combine_format.startswith("16e2m1"):
                candidates = [t for t in candidates if not _unpack_tactic(t)[5]]
            return candidates

        def _autotuner_inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            """Sanitize ONLY the autotuner-regenerated fake inputs.

            AutoTuner rebuilds inputs 0-3 and 11; the static inputs (4-10) are
            the caller's REAL weights/scales, passed by reference -- filling
            those would clobber them. topk_idx becomes a valid round-robin
            (random ints index OOB); SF / weights are filled with 1.0, NOT 0
            (SF==0 degenerates the GEMMs and skews tactic timing).
            """
            # Runs ONLY on a real profiling MISS (a tuning-mode cache HIT
            # early-returns in choose_one), so materializing the DEFERRED
            # scratch here keeps cache-hitting forwards allocation-free. The
            # factory's collective rendezvous is safe: MERGE lockstep tuning
            # gets every EP rank to this pre-hook before any coupled launch.
            if self._profiling_scratch is None and self._profiling_scratch_factory is not None:
                self._profiling_scratch = self._profiling_scratch_factory()
            if self._profiling_scratch is None and self.world_size > 1:
                # Multi-rank profiling without symmetric scratch would peer-map
                # non-symmetric tensors: cross-rank IMA. Fail loud.
                raise RuntimeError(
                    "MegaMoE-CuteDSL autotune profiling requires the "
                    "symmetric profiling scratch on multi-rank, but none is "
                    "active (and no scratch factory produced one)."
                )
            inputs = list(inputs)
            total_experts = self.num_experts_per_rank * self.world_size
            if total_experts <= 0:
                return inputs

            # topk_idx is consumed LOCALLY (not peer-mapped), so the
            # regenerated tensor stays; just make the fake ids in-range.
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
                # Multi-rank: route the CROSS-RANK inputs through the symmetric
                # scratch (sliced to the regenerated token count) so the peer
                # mapper resolves inside a symmetric region; staging untouched.
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

            # Single-rank / scratch-disabled: sanitize in place. fill_(1.0) on
            # the FP8 view writes byte 0x38; on a uint8 view write the raw
            # byte. Do NOT fill_(0x38) on the FP8 view -- that is 56.0.
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

            Rebuilt on every call, NEVER cached across runners:
            ``inputs_pre_hook`` is bound to THIS runner's ``_profiling_scratch``;
            a cached hook bound to a dead runner would mix non-symmetric
            profiling tensors with the live runner's peer offsets (IMA + hang).
            """

            config = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(1, 0, _megamoe_autotune_num_tokens),  # activation_sf
                    ConstraintSpec(2, 0, _megamoe_autotune_num_tokens),  # topk_idx
                    ConstraintSpec(3, 0, _megamoe_autotune_num_tokens),  # topk_weights
                    # combine_output moved from idx 8 -> 11 after inserting
                    # fc1_alpha(8) / fc2_alpha(9) / fc1_norm_const(10).
                    ConstraintSpec(11, 0, _megamoe_autotune_num_tokens),  # combine_output
                ),
                inputs_pre_hook=self._autotuner_inputs_pre_hook,
                # MUST stay False for multi-rank: cold-L2 rotation clone()s the
                # profiling inputs into NON-symmetric buffers, so the kernel's
                # peer mapping (clone ptr + scratch peer_offsets) resolves
                # outside any symmetric region -> cross-rank IMA while tuning.
                use_cold_l2_cache=False,
                # Pin the bucket ladder to the per-rank token ceiling instead
                # of whatever (KV-clamped, possibly non-pow2) num_tokens the
                # warmup forward fed. Mirrors trtllm-gen's tune_max_num_tokens.
                tune_max_num_tokens=self.max_tokens_per_rank,
                # CUDA Graph capture cannot reproduce MegaMoE's runtime
                # peer-pointer table / dispatch-counter view and would
                # spin inside the captured barrier when the autotuner's
                # L2-cache buffers rotate. Plain repeat-loop profiling
                # is correct and only marginally slower.
                use_cuda_graph=False,
                # Ranks couple inside the fused kernel, so every EP rank must
                # profile the SAME tactic in lockstep with identical launch
                # counts; MERGE does that and all-gathers timings so all ranks
                # converge on one global-best tactic. PARALLEL would split
                # tactics across ranks -> contaminated timing + barrier desync.
                distributed_tuning_strategy=DistributedTuningStrategy.MERGE,
            )
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
            the full tactic: the local workspace SIZE varies with it (non-epi
            token_back / atomic_counter add large regions), so an under-keyed
            cache would reuse a wrong-sized buffer -> OOB / 100% SM hang.
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
            if tactic == -1 or tactic is None:
                num_tokens = int(inputs[0].shape[0])
                # Form-B / quantized combine require non-bulk (see
                # get_valid_tactics); this deterministic fallback keeps ALL
                # quantized combine non-bulk (the tuner may recover fp8 bulk).
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
            if self._profiling_scratch is not None:
                # Profiling launch: with the DEFERRED factory, choose_one's
                # kwargs were bound to the STAGING buffers before the pre-hook
                # materialized the scratch -- rebind both cross-rank args here.
                # No-op for an eager scratch; never fires on the real run.
                peer_offsets = list(self._profiling_scratch.peer_offsets)
                shared_workspace = self._profiling_scratch.shared_workspace
            assert peer_offsets is not None, (
                "Sm100MegaMoENvfp4Runner.forward requires peer_offsets kwarg "
                "(length = world_size); single-rank degenerate mode passes "
                "(0,) * world_size."
            )
            assert len(peer_offsets) == self.world_size, (
                f"peer_offsets length {len(peer_offsets)} != world_size {self.world_size}"
            )

            kernel = self._build_kernel(tactic_t)

            # form-B accumulates the top-k reduction into ``combine_output``,
            # so its live rows MUST start at 0; the CALLER zeros [:num_tokens]
            # (the op has no live token count -- activation is padded to max_T
            # -- and a full zero wastes decode time). form-A overwrites.

            # Cached per FULL tactic: local workspace SIZE is tactic-dependent
            # (see _tactic_cache_key).
            local_workspace = _get_or_alloc_local_workspace(
                kernel,
                cache_key=self._tactic_cache_key(tactic_t),
                device=activation.device,
                latch_in_tuning_mode=self.tactic_autotune,
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
            # ``world_size`` and the whole grid deadlocks. Peer-written count
            # regions are reset device-side, the signal self-primes
            # (phase-flip) -- no per-launch host zero needed.
            #
            # Phase coherence: the sense-reversing dispatch barrier pairs a
            # PERSISTED local nvlink_barrier_counter with a PERSISTED shared
            # nvlink_barrier_signal; on a TACTIC CHANGE (autotune boundary,
            # profiling->real) the pair can be phase-DECOUPLED -> the next
            # barrier spins forever. So reset both to phase 0 under a
            # cross-rank fence ONCE per tactic. The fence (host sync +
            # collective barrier) is ILLEGAL under capture (eager warmup
            # fences first) and valid only for pure DEP (guard below).
            _capturing = torch.cuda.is_current_stream_capturing()
            # Fence ONCE per (local_ws, shared_ws, world) key, tracked
            # module-globally. Keys are raw pointers, so "already fenced" also
            # requires the STORAGES to be alive (ABA; _megamoe_fence_key_live).
            _fkey = (
                int(local_workspace.data_ptr()),
                int(shared_workspace.data_ptr()),
                int(self.world_size),
            )
            _fkey_live = _megamoe_fence_key_live(_fkey)
            if _capturing:
                if not _fkey_live:
                    # Eager pre-passes fence every pairing a captured forward
                    # uses; reaching capture unfenced means two runners share a
                    # workspace side. Fail loud (replay would hang silently).
                    raise RuntimeError(
                        "MegaMoE-CuteDSL: CUDA-graph capture reached an "
                        "unfenced (local, shared) workspace pairing; the "
                        "barrier-reset fence cannot run during capture."
                    )
                # A later EAGER fence must never zero this captured shared side.
                _MEGAMOE_CAPTURED_SHARED_PTRS.add(_fkey[1])
                global _MEGAMOE_GRAPH_CAPTURE_SEEN
                _MEGAMOE_GRAPH_CAPTURE_SEEN = True
            if (not _fkey_live) and not _capturing:
                import torch.distributed as _dist

                _sig_off = int(kernel._shared_offsets["nvlink_barrier_signal"])
                _sig_n = int(kernel._shared_region_by_name["nvlink_barrier_signal"].nbytes)
                _bc_off = int(kernel._local_offsets["nvlink_barrier_counter"])
                _bc_n = int(kernel._local_region_by_name["nvlink_barrier_counter"].nbytes)
                _have_dist = _dist.is_available() and _dist.is_initialized()
                # The fence barriers the default (WORLD) group -- correct ONLY
                # for pure DEP. Under TP x EP / PP the WORLD barrier spans ranks
                # that never reach this fence -> deadlock; fail loud instead.
                # (Threading the EP subgroup is the general fix.)
                if _have_dist and _dist.get_world_size() != int(self.world_size):
                    # Not an assert: the guard must survive ``python -O``.
                    raise RuntimeError(
                        "MegaMoE-CuteDSL barrier-reset fence uses the WORLD process "
                        "group, valid only for pure DEP (world == EP). Detected WORLD="
                        f"{_dist.get_world_size()} != EP={self.world_size} (TP x EP / "
                        "PP) -- fence must run on the EP subgroup."
                    )
                # Never re-zero a shared side baked into a captured graph
                # (captured pairings cannot re-fence -> silent hang).
                if _fkey[1] in _MEGAMOE_CAPTURED_SHARED_PTRS:
                    raise RuntimeError(
                        "MegaMoE-CuteDSL: eager barrier-reset fence would "
                        "zero a shared workspace already used by a captured "
                        "CUDA graph; the captured pairings cannot re-fence."
                    )
                torch.cuda.current_stream().synchronize()
                if _have_dist:
                    _dist.barrier()
                # Reset BOTH sides to phase 0. The local counter can be
                # ALREADY-ADVANCED here (at the profiling->real transition the
                # winning tactic's cached local ran under the scratch's shared
                # ptr; the real launch's new fkey re-fences it). Zero only the
                # tiny counter slice; the multi-GiB bulk stays uninitialized.
                local_workspace[_bc_off : _bc_off + _bc_n].zero_()
                shared_workspace[_sig_off : _sig_off + _sig_n].zero_()
                # The zeros are stream-async and dist.barrier() is NOT
                # device-ordered after them: a released peer's in-kernel signal
                # WRITE could be CLOBBERED by our still-pending zero (barrier
                # spins forever). Sync so the zeros LAND before peers launch.
                torch.cuda.current_stream().synchronize()
                if _have_dist:
                    _dist.barrier()
                # Both sides were just re-zeroed, so every OTHER fenced pairing
                # sharing either side is now phase-stale (the autotune
                # real->scratch->real ABA); drop them so they re-fence.
                # Invariant: fkey in set => neither side touched under any
                # other pairing since its fence.
                _megamoe_prune_fenced_keys(
                    {k for k in _MEGAMOE_FENCED_KEYS if k[0] == _fkey[0] or k[1] == _fkey[1]}
                )
                _MEGAMOE_FENCED_KEYS.add(_fkey)
                # Record the fenced ALLOCATIONS for recycled-VA (ABA) detection.
                _MEGAMOE_FENCED_KEY_WS_REFS[_fkey] = (
                    weakref.ref(local_workspace.untyped_storage()),
                    weakref.ref(shared_workspace.untyped_storage()),
                )
            # else: SAME tactic -- the kernel's device-side tail self-reset suffices.

            # cute.compile (JIT) launch. The
            # uint8 workspaces MUST be cute.Pointer, NOT cute.Tensor: the
            # 32-bit memref shape field overflows once shared_workspace passes
            # 2 GiB (the kernel addresses by raw base + Int64 byte offset).
            _to_cute, _to_cute_ptr, SymBufferHost, cute, cutlass_utils = _cute_launch_helpers()

            # combine_output (max_T, 1, hidden) reshapes freely to 2D. Weights
            # present K stride-1 via a transpose VIEW (DLPack carries the
            # strides); do NOT ``.contiguous()``.
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
        tactic_autotune: bool = False,
        num_tokens: int = -1,
    ) -> None:
        """Run the fused MegaMoE CuteDSL NVFP4 kernel.

        Inputs are pre-staged by the caller (the ``MegaMoECuteDsl``
        backend in ``mega_moe_cute_dsl.py``). ``tactic_autotune=True``
        (``MEGAMOE_TACTIC_AUTOTUNE=1`` bench opt-in) picks the tactic via
        AutoTuner per call; the default runs the deterministic token-bucket
        heuristic (``default_megamoe_tactic``).

        ``shared_workspace`` MUST be a symmetric-heap tensor for
        ``world_size > 1`` (use :class:`MegaMoeSymmMemProvider`); a
        local CUDA tensor is acceptable for the single-rank degenerate
        path. ``combine_output`` is mutated in place; the op does not
        return it because torch custom_op forbids the return value from
        aliasing any mutated input.

        ``in_kernel_fc2_reduce`` selects the reduction form: ``False``
        (form-A) runs the deterministic standalone TopkReduce; ``True``
        (form-B) folds top-k into the kernel and is NON-deterministic (float
        accumulation order). Both write ``combine_output`` with shape
        ``(T, 1, hidden)``. Perf knobs come from the tactic, not op
        arguments.
        """
        sm_version = get_sm_version()
        if sm_version not in (100, 103):
            raise RuntimeError(
                f"cute_dsl_megamoe_nvfp4_blackwell requires SM 100 (B200) or "
                f"SM 103 (B300); got SM {sm_version}."
            )

        # Live-token trim: TopkReduce sizes its grid from THIS tensor's dim0,
        # so slicing to the live count skips dead reduce rows (~5-8us/layer at
        # decode). num_tokens < 0 or oversized keeps the full bucket; so does
        # num_tokens == 0 (a zero-token rank still launches so peers can cross
        # the NVLink barrier -- a 0-row slice would build a zero grid).
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
            tactic_autotune=tactic_autotune,
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
        if not tactic_autotune:
            # Opt-OUT (default; serving never opts in): skip the AutoTuner.
            # ``choose_one`` is NOT a safe no-op -- in tuning mode a cache miss
            # materializes the multi-GiB scratch and MERGE-sweeps ~36
            # candidates. tactic=-1 guarantees the deterministic heuristic
            # with no tuning collectives, even inside a global autotune().
            runner(
                inputs,
                tactic=-1,
                peer_offsets=peer_offsets,
                shared_workspace=shared_workspace,
            )
            return
        tuner = AutoTuner.get()
        # Opt-IN: in tuning mode the MERGE lockstep sweep runs (made safe by
        # the tactic-change fence in forward); outside it choose_one is a pure
        # cache lookup (tuned tactic, or -1 -> deterministic heuristic).
        # Profiling must use SYMMETRIC cross-rank buffers (the transient
        # scratch); only single-rank may fall back to the staging buffer.
        prof_scratch = _ACTIVE_MEGAMOE_PROFILING_SCRATCH if world_size > 1 else None
        prof_factory = _ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY if world_size > 1 else None
        if (
            prof_scratch is None
            and prof_factory is None
            and world_size > 1
            and tuner.is_tuning_mode
        ):
            # Not a fallback: profiling would peer-map non-symmetric tensors
            # (cross-rank IMA). The backend hands either a live scratch or a
            # DEFERRED factory; reaching here means both were bypassed.
            raise RuntimeError(
                "MegaMoE-CuteDSL autotune profiling requires the "
                "symmetric profiling scratch (or a scratch factory) on "
                "multi-rank, but neither is active."
            )
        if prof_scratch is not None:
            runner._profiling_scratch = prof_scratch
            prof_peer_offsets = list(prof_scratch.peer_offsets)
            prof_shared_workspace = prof_scratch.shared_workspace
        else:
            # DEFERRED path: the pre-hook materializes the scratch only if
            # real profiling launches; ``forward`` then rebinds these staging
            # values from it, and on a cache hit choose_one never launches --
            # so they are never consumed for profiling.
            runner._profiling_scratch = None
            runner._profiling_scratch_factory = prof_factory
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
            runner._profiling_scratch_factory = None
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
        tactic_autotune: bool = False,
        num_tokens: int = -1,
    ) -> None:
        return None
