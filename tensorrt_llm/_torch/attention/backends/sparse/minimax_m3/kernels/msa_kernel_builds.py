# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fmha_sm100's kernels, built ahead through ``KernelBuilder``.

fmha_sm100 compiles one kernel per variant (dtype, query tile, single warp group, sparse
mode, page size, split KV, pack factor) on first use, one variant at a time per process.
Its plan picks the variant from the longest query segment and the split count; the call
site fixes the rest. These two are its build dimensions (``_CONFIG``, ``_PROXY_CONFIG``).

``install`` wraps the package's ``fmha_sm100_plan`` and ``fmha_sm100``. On a run in
KernelBuilder's build mode, the plan and the run are replayed for one query segment of each
shape the dimensions sweep from that call's and no earlier run of its call signature swept,
with the package's variant lookup recording the variant it is asked for instead of compiling
it. The variants found build on the rank's worker processes (:func:`build_variant`), in the
directory and under the file lock the package's own lookup uses, and the run goes on once they
are built: its lookups load them. The plan, split-KV reduction and sparse top-k modules, and the
k2q CSR extension the sparse prefill loads, need nothing from a call: they build with the
variants of the first run in build mode.

In a KernelBuilder rehearsal the builds are queued and the run is itself such a replay: it
compiles and launches nothing, and its outputs hold what a run over no keys gives, zeros and a
maximum score of -inf.
"""

from __future__ import annotations

import functools
import os
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.kernel_builder import (
    BuildConfig,
    BuildStrategy,
    BuildTask,
    DynamicSpec,
    KernelBuilder,
)
from tensorrt_llm.logger import logger

_OP = "fmha_sm100"


def _query_len_buckets(longest: int) -> Tuple[int, ...]:
    """Query segment lengths up to the longest a call has: each length up to 129 crosses a
    boundary of the pack factor, the single warp group or the 128-row query tile; 256 takes
    the 256-row tile."""
    return tuple(range(1, min(longest, 129) + 1)) + ((256,) if longest >= 256 else ())


def _cache_scope() -> str:
    """Where the package's builds land: the ranks whose lookups read the same directory on this
    host split the builds."""
    return f"{os.uname().nodename}:{os.path.realpath(str(_package.jit._jit_base()))}"


# A variant or module takes 35-60 s to compile on GB300.
_BUILD_SECONDS = 40.0
_QUERY_LEN = DynamicSpec("qo_len", _query_len_buckets)
_SHARED = dict(
    strategy=BuildStrategy.SHARED, share_scope=_cache_scope, build_seconds=_BUILD_SECONDS
)
# A call's query segment lengths.
_CONFIG = BuildConfig((_QUERY_LEN,), **_SHARED)
# The MSA indexer's maximum-score pass also chooses whether to split KV, per batch
# (msa_indexer._proxy_kv_splits); a variant depends only on whether it splits.
_PROXY_CONFIG = BuildConfig((_QUERY_LEN, DynamicSpec("num_kv_splits", (1, 2))), **_SHARED)
# The k2q CSR extension builds into PyTorch's extension directory, under PyTorch's own lock.
_K2Q_CONFIG = BuildConfig(build_seconds=_BUILD_SECONDS)
# Plan options a replay does not pass through: it plans its own lengths and offsets, and a
# swept split count.
_PLAN_SKIP = ("qo_offset", "device", "stream", "num_kv_splits")
# Run options that do not choose the variant.
_RUN_SKIP = ("sm_scale", "q_scale", "k_scale", "v_scale", "o_scale", "check_input_valid")
# Run options a replay replaces: its query is its own, and its outputs are discarded.
_RUN_REPLACED = ("q_offset_override", "out", "max_score")
_PLAN_ARGS = "_tllm_kernel_builder_plan_args"
_INSTALLED = "_tllm_kernel_builder_installed"
# The modules every MSA model loads: (name, library, the package's compile function).
_MODULES = (
    ("plan", "fmha_sm100_plan.so", "_do_compile_plan"),
    ("reduction", "fmha_sm100_reduction.so", "_do_compile_reduction"),
    ("sparse_topk", "sparse_topk_select.so", "_do_compile_sparse_topk"),
)
# The package module that compiles the sparse prefill's k2q CSR extension on import.
_K2Q_CSR_MODULE = "src.sm100.build_k2q_csr"

# Where the worker processes find the build functions below.
_HERE = __name__
_replay = threading.local()  # .found: the variants a replay on this thread asked for
_package: Optional["_Package"] = None  # set by install
_install_lock = threading.Lock()


class _Stub:
    """A variant or the split-KV reduction module whose calls do nothing, for a replay."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


def _tag(value: Any) -> Any:
    """What an argument contributes to a call signature: a tensor its dtype and rank."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value.dtype, value.dim())
    if isinstance(value, (list, tuple)):
        return tuple(_tag(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return ("unhashable", type(value).__name__)
    return value


class _Package:
    """The fmha_sm100 entry points ``install`` wraps, and the originals they call."""

    def __init__(self, module: types.ModuleType, api: types.ModuleType, jit: types.ModuleType):
        self.jit = jit
        self.manager = jit._variant_manager
        self.variant_key = jit._variant_key_from_runtime
        self.sparse_prefill_min_len = api._prefill_qlen_threshold(True)
        # Checked here so that a package without them is left alone.
        for name in ("_compile_only", "_load_templates", "_loaded", "_ensure_all_module"):
            getattr(self.manager, name)
        for name in ("_built_dir", "_jit_base", "_acquire_file_lock", "_release_file_lock"):
            getattr(jit, name)
        for _, _, compile_fn in _MODULES:
            getattr(jit, compile_fn)
        self.plan_fn, self.run_fn = module.fmha_sm100_plan, module.fmha_sm100
        self.variant_fn, self.reduction_fn = api.get_fmha_variant, api.get_reduction_module
        self.looked_up: set = set()
        self.modules_listed = False

    def plan(self, qo_segment_lens, kv_segment_lens, *args, **kwargs):
        plan_info = self.plan_fn(qo_segment_lens, kv_segment_lens, *args, **kwargs)
        options = {key: value for key, value in kwargs.items() if key not in _PLAN_SKIP}
        num_kv_splits = kwargs.get("num_kv_splits", 1)
        for sub_plan in plan_info[3:]:
            if isinstance(sub_plan, dict):
                sub_plan[_PLAN_ARGS] = (args, options, qo_segment_lens, num_kv_splits)
        return plan_info

    def _module_builds(self) -> List[Tuple[str, BuildTask, BuildConfig]]:
        """The builds of the modules, the first time they are asked for."""
        if self.modules_listed:
            return []
        self.modules_listed = True
        builds = [
            (name, BuildTask(f"{_HERE}:build_module", (name, library, compile_fn)), _CONFIG)
            for name, library, compile_fn in _MODULES
        ]
        builds.append((_K2Q_CSR_MODULE, BuildTask(f"{_HERE}:import_k2q_csr"), _K2Q_CONFIG))
        return builds

    def run(self, q, k, v, plan_info, *args, **kwargs):
        builder = KernelBuilder.get()
        if (
            builder.active()
            and not args
            and isinstance(plan_info, tuple)
            and not torch.cuda.is_current_stream_capturing()
        ):
            planned = next(
                (p[_PLAN_ARGS] for p in plan_info[3:] if isinstance(p, dict) and _PLAN_ARGS in p),
                None,
            )
            if planned is not None:
                plan_args, plan_options, qo_segment_lens, num_kv_splits = planned
                run_options = {key: v_ for key, v_ in kwargs.items() if key not in _RUN_SKIP}
                proxy = run_options.get("output_o") is False and plan_options.get("output_maxscore")
                config = _PROXY_CONFIG if proxy else _CONFIG
                signature = (
                    tuple(_tag(arg) for arg in plan_args),
                    tuple(sorted((key, _tag(value)) for key, value in plan_options.items())),
                    q.dtype,
                    tuple(q.shape[1:]),
                    k.dtype,
                    tuple(k.shape[1:]),
                    tuple(sorted((key, _tag(value)) for key, value in run_options.items())),
                    None if proxy else num_kv_splits,
                )
                lens = torch.as_tensor(qo_segment_lens)
                observed = {"qo_len": max(int(lens.max()), 1) if lens.numel() else 1}
                if proxy:
                    observed["num_kv_splits"] = num_kv_splits
                builder.prebuild(
                    _OP,
                    signature,
                    config,
                    observed,
                    functools.partial(
                        self._list, q, k, v, plan_args, plan_options, num_kv_splits, run_options
                    ),
                )
        if builder.rehearsing():
            return self._rehearse(q, k, v, plan_info, *args, **kwargs)
        return self.run_fn(q, k, v, plan_info, *args, **kwargs)

    def _rehearse(self, q, k, v, plan_info, *args, **kwargs):
        """The package's run with its variant and split-KV reduction stubbed; its outputs are set
        to what a run over no keys gives: zeros and a maximum score of -inf."""
        _replay.found = {}
        try:
            result = self.run_fn(q, k, v, plan_info, *args, **kwargs)
        finally:
            _replay.found = None
        out, max_score = result if isinstance(result, tuple) else (result, None)
        for tensor, value in (
            (out, 0.0),
            (kwargs.get("out"), 0.0),
            (max_score, float("-inf")),
            (kwargs.get("max_score"), float("-inf")),
        ):
            if isinstance(tensor, torch.Tensor):
                tensor.fill_(value)
        return result

    def _list(
        self, q, k, v, plan_args, plan_options, num_kv_splits, run_options, shapes
    ) -> List[tuple]:
        """Replay the plan and the run for one query segment of each shape; return the
        builds of the variants asked for, after those of the modules the first time.

        A shape the package rejects is skipped: a call of that shape would fail too.
        """
        options = {key: value for key, value in run_options.items() if key not in _RUN_REPLACED}
        sparse = plan_options.get("kv_block_num", -1) > 0
        sparse_mode = plan_options.get("sparse_kernel_mode", "auto")
        found: Dict[str, dict] = {}
        _replay.found = found
        try:
            for shape in shapes:
                qo_len = shape["qo_len"]
                if sparse and (
                    sparse_mode == "prefill"
                    or (sparse_mode == "auto" and qo_len > self.sparse_prefill_min_len)
                ):
                    # The sparse prefill path, whose CuTe DSL kernels are not variants.
                    continue
                lens = torch.tensor([qo_len], dtype=torch.int32)
                q_replay = torch.empty((qo_len, *q.shape[1:]), dtype=q.dtype, device=q.device)
                splits = shape.get("num_kv_splits", num_kv_splits)
                try:
                    plan_info = self.plan_fn(
                        lens, lens.clone(), *plan_args, num_kv_splits=splits, **plan_options
                    )
                    self.run_fn(q_replay, k, v, plan_info, **options)
                except (ValueError, RuntimeError, AssertionError) as e:
                    logger.debug(f"[KernelBuilder] {_OP}: no variant for {shape}: {e}")
        finally:
            _replay.found = None
        # Variants in the package's bundled module need no build.
        self.manager._ensure_all_module()
        bundled = self.manager._all_module
        return self._module_builds() + [
            (name, BuildTask(f"{_HERE}:build_variant", (name, params)))
            for name, params in found.items()
            if bundled is None or not hasattr(bundled, f"run_{name}")
        ]

    def get_variant(self, *args):
        found: Optional[Dict[str, dict]] = getattr(_replay, "found", None)
        if found is not None:
            name, params = self.variant_key(*args)
            found.setdefault(name, params)
            return _Stub()
        if args in self.looked_up:
            return self.variant_fn(*args)
        try:
            name = self.variant_key(*args)[0]
        except ValueError:
            return self.variant_fn(*args)
        missing = (
            name not in self.manager._loaded
            and not (self.jit._built_dir(name, f"{name}.so") / f"{name}.so").exists()
        )
        t0 = time.perf_counter()
        variant = self.variant_fn(*args)
        seconds = time.perf_counter() - t0
        self.looked_up.add(args)
        # A variant loaded from the package's bundle or a finished build takes milliseconds.
        if missing and seconds >= 1.0:
            KernelBuilder.get().op_compiled(_OP, name, seconds)
        return variant

    def get_reduction_module(self):
        if getattr(_replay, "found", None) is not None:
            return _Stub()
        return self.reduction_fn()


def _require_package():
    """In a worker process: the fmha_sm100 package, imported as this rank imports it."""
    from .msa_utils import require_msa_module

    require_msa_module()
    from fmha_sm100 import jit

    return jit


def build_variant(name: str, params: dict) -> bool:
    """In a worker process: compile the variant ``name`` into the package's directory under
    the file lock its own lookup takes; False when it was built already."""
    jit = _require_package()
    manager = jit._variant_manager
    cache_dir = jit._built_dir(name, f"{name}.so")
    fd = jit._acquire_file_lock(cache_dir / ".compile.lock")
    try:
        if (cache_dir / f"{name}.so").exists():
            return False
        manager._load_templates()
        manager._compile_only(name, params, cache_dir)
        return True
    finally:
        jit._release_file_lock(fd)


def build_module(name: str, library: str, compile_fn: str) -> bool:
    """In a worker process: compile the package's module ``name`` (``library``) with its
    compile function ``compile_fn``; False when it was built already."""
    jit = _require_package()
    cache_dir = jit._built_dir(name, library)
    fd = jit._acquire_file_lock(cache_dir.parent / f"{name}.lock")
    try:
        if (cache_dir / library).exists():
            return False
        getattr(jit, compile_fn)(cache_dir)
        return True
    finally:
        jit._release_file_lock(fd)


def import_k2q_csr() -> bool:
    """In a worker process: the k2q CSR extension compiles when its module is first imported,
    into PyTorch's extension directory, where this rank's import finds it built."""
    import importlib

    _require_package()
    t0 = time.perf_counter()
    importlib.import_module(_K2Q_CSR_MODULE)
    # Loading a built extension takes milliseconds.
    return time.perf_counter() - t0 >= 1.0


def _route(module: Any, api: Any, package: _Package) -> None:
    """Point the package's entry points at ``package``'s wrappers."""
    global _package
    _package = package
    module.fmha_sm100_plan = package.plan
    module.fmha_sm100 = package.run
    api.get_fmha_variant = package.get_variant
    api.get_reduction_module = package.get_reduction_module


def install(module: types.ModuleType) -> None:
    """Route ``module``'s plan, run and variant lookup through ``KernelBuilder``; once."""
    if getattr(module, _INSTALLED, False):
        return
    with _install_lock:
        if getattr(module, _INSTALLED, False):
            return
        if KernelBuilder.get().enabled:
            try:
                from fmha_sm100 import api, jit

                package = _Package(module, api, jit)
            except Exception as e:  # noqa: BLE001 - the kernels stay on the package's own path
                logger.warning(
                    f"[KernelBuilder] fmha_sm100's kernels are left to their first use: "
                    f"{type(e).__name__}: {e}"
                )
            else:
                _route(module, api, package)
        setattr(module, _INSTALLED, True)
