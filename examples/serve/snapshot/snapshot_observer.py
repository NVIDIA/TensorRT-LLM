# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment-only observers and a pre-admission native MPI capture boundary."""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
import weakref
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

_COUNTS: dict[str, int] = {}
_LOCK = threading.Lock()
_RANK: int | None = None
_ATTEMPT: str | None = None
_CONTROL = Path("/snapshot-control")
_PURPOSES: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()


def write_json(path: Path, value: Any) -> None:
    """Publish an observation atomically without keeping checkpointed file offsets.

    Args:
        path: Destination in this experiment's control mount.
        value: JSON-serializable observation.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value: Any) -> str:
    """Hash a canonical observation.

    Args:
        value: JSON-serializable observation.

    Returns:
        SHA-256 digest.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _observe(owner: type, name: str, event: str, *, static: bool = False) -> None:
    """Count successful calls to an existing implementation, preserving its return.

    Args:
        owner: Class containing the observed implementation.
        name: Existing attribute to wrap.
        event: Counter name; a method entry is not a count of transformed tensors.
        static: Whether to retain static-method binding.
    """
    original = getattr(owner, name)

    @functools.wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        """Call the original implementation and record successful completion."""
        result = original(*args, **kwargs)
        with _LOCK:
            _COUNTS[event] = _COUNTS.get(event, 0) + 1
            if _RANK is not None:
                write_json(
                    _CONTROL / f"counters-{_RANK}.json",
                    {"attempt": _ATTEMPT, "rank": _RANK, "counts": dict(_COUNTS)},
                )
        return result

    setattr(owner, name, staticmethod(observed) if static else observed)


def _kv_state(manager: Any, *, materialized: bool = True) -> dict[str, Any]:
    """Inspect native V2 backing and logical owners without clearing them.

    Args:
        manager: A final, quiescent KVCacheManagerV2.
        materialized: Whether native backing is available for tensor views.

    Returns:
        Per-tier medium bytes, pool geometry and internal-cache ownership.

    Raises:
        RuntimeError: If live user KV or an unsupported manager is present.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    if not isinstance(manager, KVCacheManagerV2):
        raise RuntimeError("The native experiment requires KVCacheManagerV2")
    owners = []
    for request_id, cache in sorted(manager.kv_cache_map.items()):
        excluded = manager.impl.is_stats_excluded(request_id)
        if not excluded:
            raise RuntimeError(f"Live user KV at capture boundary: {request_id}")
        owners.append(
            {
                "id": request_id,
                "stats_excluded": excluded,
                "committed_tokens": cache.num_committed_tokens,
            }
        )
    tiers = []
    for level, tier in enumerate(manager.impl.cache_tier_list):
        tiers.append(
            {
                "level": level,
                "tier": str(tier),
                "quota": manager.impl.get_quota(level),
                "bytes": {
                    name: manager.impl.get_num_bytes(level, medium)
                    for name, medium in (
                        ("gpu", CacheTier.GPU_MEM),
                        ("host", CacheTier.HOST_MEM),
                        ("disk", CacheTier.DISK),
                    )
                },
            }
        )
    pools = []
    for layer in sorted(manager.layer_offsets):
        if not materialized:
            from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

            local_layer = manager.layer_offsets[layer]
            pools.append(
                {
                    "layer": layer,
                    "address": manager.impl.get_mem_pool_base_address(
                        local_layer, "key", PageIndexMode.SHARED
                    ),
                }
            )
            continue
        tensor = manager.get_buffers(layer)
        if tensor is None:
            raise RuntimeError("The native profile requires dense attention KV buffers")
        pools.append(
            {
                "layer": layer,
                "address": tensor.data_ptr(),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
        )
    return {"owners": owners, "tiers": tiers, "pools": pools}


def require_compacted(before: dict[str, Any], prepared: dict[str, Any]) -> None:
    """Require real backing release, not only a successful release API call.

    Args:
        before: KV inventory before native release.
        prepared: KV inventory after native release.

    Raises:
        RuntimeError: If the GPU-only profile does not release all KV backing.
    """
    if sum(tier["bytes"]["gpu"] for tier in before["tiers"]) <= 0:
        raise RuntimeError("Compact baseline has no GPU KV backing")
    if any(sum(tier["bytes"].values()) != 0 for tier in prepared["tiers"]):
        raise RuntimeError("Compact preparation retained KV backing")


def _state(executor: Any) -> dict[str, Any]:
    """Collect final graph, weight and KV evidence at a parked boundary.

    Args:
        executor: Native final PyExecutor before its worker loop starts.

    Returns:
        Content/layout evidence and separately measured observation overhead.

    Raises:
        RuntimeError: If weights or final CUDA graphs cannot be fully inspected.
    """
    import torch

    from tensorrt_llm._torch.weight_sharing.weight_manifest import build_weight_manifest

    started = time.monotonic()
    model = executor.model_engine.model
    manifest = build_weight_manifest(model)
    if manifest.skipped:
        raise RuntimeError("Weight manifest contains unobserved tensors")
    addresses = {
        name: {"address": tensor.data_ptr(), "storage": tensor.untyped_storage().data_ptr()}
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
    }
    runner = executor.model_engine.cuda_graph_runner
    if runner is None or not runner.graphs:
        raise RuntimeError("No final decoder CUDA graphs were captured")
    torch.cuda.synchronize()
    buffers = executor.model_engine._runner._buffers
    input_storage = {}
    for field in fields(buffers):
        tensor = getattr(buffers, field.name)
        if tensor is not None:
            input_storage[field.name] = {
                "address": tensor.data_ptr(),
                "storage": tensor.untyped_storage().data_ptr(),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "pinned": tensor.is_pinned(),
            }
    weights = manifest.to_dict()
    context = weights.pop("context")
    return {
        "weights": weights,
        "weight_observation": context,
        "weight_addresses": addresses,
        "graph_keys": sorted(repr(key) for key in runner.graphs),
        "graph_runner": type(runner).__qualname__,
        "runner_input_storage": input_storage,
        "kv": _kv_state(executor.kv_cache_manager),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "observation_seconds": time.monotonic() - started,
    }


def _require_same_state(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Require preserved content, graph-visible addresses and logical KV ownership.

    Args:
        before: Final resident state before applying the capture policy.
        after: State after process restore and optional compact reconstruction.

    Raises:
        RuntimeError: If a required invariant differs.
    """
    for field in (
        "weights",
        "weight_addresses",
        "graph_keys",
        "graph_runner",
        "runner_input_storage",
    ):
        if before[field] != after[field]:
            raise RuntimeError(f"Restored {field} differs from the captured worker")
    for field in ("owners", "pools", "tiers"):
        if before["kv"][field] != after["kv"][field]:
            raise RuntimeError(f"Reconstructed KV {field} differs from the pre-policy state")


def _park(executor: Any) -> None:
    """Park the final native MPI cohort before it can accept any request.

    Args:
        executor: Final PyExecutor whose original start_worker has not run.

    Raises:
        RuntimeError: If the profile, cohort or restored state is invalid.
        TimeoutError: If no restore signal arrives before the experiment deadline.
    """
    global _RANK, _ATTEMPT
    import torch

    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.executor.worker_process_monitor import capture_worker_process_identity
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

    config = json.loads((_CONTROL / "experiment.json").read_text())
    _RANK = executor.dist.rank
    _ATTEMPT = config["attempt"]
    args = executor.llm_args
    if (
        args.torch_compile_config is not None
        or args.allreduce_strategy != "NCCL"
        or not args.disable_overlap_scheduler
        or args.num_postprocess_workers != 0
        or args.num_serve_frontends != 1
        or args.enable_resource_governor
        or executor._openengine_discovery
        or os.environ.get("TLLM_NCCL_SYMMETRIC_ZERO_COPY") != "0"
        or os.environ.get("TRTLLM_PROXY_BATCH_REQUESTS") != "0"
    ):
        raise RuntimeError("Resolved runtime contains an excluded owner or execution mode")
    if (
        executor.worker_started
        or executor.active_requests
        or executor.waiting_queue
        or executor.previous_batch is not None
        or executor.kv_cache_transceiver is not None
        or executor.kv_connector_manager is not None
        or executor.draft_model_engine is not None
        or executor.dist.pp_size != 1
        or executor.dist.cp_size != 1
    ):
        raise RuntimeError("Only a clean, dense, non-disaggregated TP startup cohort is supported")
    if mpi_comm().Get_size() != config["tp"]:
        raise RuntimeError("Native MPI world size differs from the profile")
    if _PURPOSES.get(executor.model_engine) != "final_executor":
        raise RuntimeError("Refusing to capture a profiling executor")
    if (
        executor.kv_cache_manager._sharing is not None
        or len(executor.kv_cache_manager.impl.cache_tier_list) != 1
    ):
        raise RuntimeError("The first profile excludes external lenders and HOST/DISK tiers")
    tags = [ExecutorMemoryType.KV_CACHE]
    if config["policy"] == "compact":
        if not hasattr(executor, "validate_sleep_config"):
            raise RuntimeError(
                "Compact experiments require a matching #21221-compatible Python/native build"
            )
        executor.validate_sleep_tags(tags)
        executor.validate_sleep_config(tags)
    cohort = mpi_comm().allgather((_RANK, _ATTEMPT, config["profile"], config["policy"]))
    if sorted(cohort) != [
        (rank, _ATTEMPT, config["profile"], config["policy"]) for rank in range(config["tp"])
    ]:
        raise RuntimeError("Capture preflight cohort does not agree")
    torch.cuda.synchronize()
    before = _state(executor)
    if any(_COUNTS.get(key, 0) == 0 for key in ("model_load", "post_load_walk", "graph_capture")):
        raise RuntimeError("Missing positive load/post-load/graph-capture observer evidence")
    saved_counts = dict(_COUNTS)
    identity = capture_worker_process_identity(_RANK)._asdict()
    policy_seconds = 0.0
    compact = config["mode"] in ("capture", "native_cycle") and config["policy"] == "compact"
    if compact:
        from tensorrt_llm._torch.virtual_memory import release_with_tag

        started = time.monotonic()
        release_with_tag("kv_cache", resource=executor)
        torch.cuda.synchronize()
        policy_seconds = time.monotonic() - started
    prepared = _kv_state(
        executor.kv_cache_manager,
        materialized=not compact,
    )
    if compact:
        require_compacted(before["kv"], prepared)
    record = {
        "attempt": _ATTEMPT,
        "template": config["template"],
        "profile": config["profile"],
        "rank": _RANK,
        "identity": identity,
        "counts": saved_counts,
        "before_policy": before,
        "prepared_kv": prepared,
        "policy_seconds": policy_seconds,
        "resolved_config": args.model_dump(mode="json"),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "device": torch.cuda.get_device_name(),
            "visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
    }
    cohort = mpi_comm().allgather((_RANK, _ATTEMPT, config["profile"]))
    if sorted(cohort) != [(rank, _ATTEMPT, config["profile"]) for rank in range(config["tp"])]:
        raise RuntimeError("Capture cohort does not agree on the attempt and profile")
    write_json(_CONTROL / f"capture-{_RANK}.json", record)
    if config["mode"] == "cold":
        write_json(_CONTROL / f"runtime-{_RANK}.json", record)
        return
    if config["mode"] == "native_cycle":
        started = time.monotonic()
        if compact:
            from tensorrt_llm._torch.virtual_memory import materialize_with_tag

            materialize_with_tag("kv_cache", resource=executor)
        torch.cuda.synchronize()
        reconstruction = time.monotonic() - started
        after = _state(executor)
        _require_same_state(before, after)
        if _COUNTS != saved_counts:
            raise RuntimeError("Native cycle repeated startup work")
        write_json(
            _CONTROL / f"runtime-{_RANK}.json",
            {
                **record,
                "state": after,
                "reconstruction_seconds": reconstruction,
                "process_restore": "NOT_RUN",
            },
        )
        return
    deadline = time.monotonic() + config["timeout_seconds"]
    while not (_CONTROL / "restore-complete").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Host capture/restore did not complete")
        time.sleep(0.05)
    restored = json.loads((_CONTROL / "experiment.json").read_text())
    if (
        restored["mode"] != "restore"
        or restored["attempt"] == _ATTEMPT
        or any(restored[key] != config[key] for key in ("template", "profile", "policy", "tp"))
    ):
        raise RuntimeError("Restore lacks a fresh attempt bound to this captured profile")
    _ATTEMPT = restored["attempt"]
    torch.cuda.synchronize()
    write_json(_CONTROL / f"memory-{_RANK}.json", {"attempt": _ATTEMPT, "rank": _RANK})
    started = time.monotonic()
    if config["policy"] == "compact":
        from tensorrt_llm._torch.virtual_memory import materialize_with_tag

        materialize_with_tag("kv_cache", resource=executor)
    torch.cuda.synchronize()
    reconstruction = time.monotonic() - started
    after = _state(executor)
    _require_same_state(before, after)
    if _COUNTS != saved_counts:
        raise RuntimeError("Load, post-load or CUDA graph work reran before runtime validation")
    cohort = mpi_comm().allgather((_RANK, _ATTEMPT, config["profile"]))
    if sorted(cohort) != [(rank, _ATTEMPT, config["profile"]) for rank in range(config["tp"])]:
        raise RuntimeError("Restored MPI cohort disagrees on readiness")
    write_json(
        _CONTROL / f"runtime-{_RANK}.json",
        {
            "attempt": _ATTEMPT,
            "template": config["template"],
            "rank": _RANK,
            "identity": capture_worker_process_identity(_RANK)._asdict(),
            "counts": dict(_COUNTS),
            "state": after,
            "reconstruction_seconds": reconstruction,
        },
    )


def _install_model_observers(module: ModuleType) -> None:
    """Wrap the model loader after its normal import and worker initialization.

    Args:
        module: Fully initialized native model-loader module.
    """
    import torch

    ModelLoader = module.ModelLoader
    _observe(ModelLoader, "load", "model_load")
    _observe(ModelLoader, "reload", "model_reload")
    _observe(ModelLoader, "_call_load_weights", "weight_population")
    _observe(ModelLoader, "_walk_full_post_load", "post_load_walk", static=True)
    _observe(ModelLoader, "_walk_transform", "staged_transform_walk", static=True)
    _observe(torch.cuda.CUDAGraph, "capture_end", "graph_capture")
    _observe(torch.cuda.CUDAGraph, "replay", "graph_replay")


def _install_engine_observers(module: ModuleType) -> None:
    """Retain the creator's warmup designation across execution-service rebinding.

    Args:
        module: Normally imported model-engine module.
    """
    original = module.ModelEngine.set_warmup_purpose

    @functools.wraps(original)
    def set_purpose(engine: Any, purpose: str) -> None:
        """Observe the completed designation without relying on a temporary timer."""
        original(engine, purpose)
        _PURPOSES[engine] = purpose

    module.ModelEngine.set_warmup_purpose = set_purpose


def _install_executor_observers(module: ModuleType) -> None:
    """Wrap the final worker boundary after normal executor module import.

    Args:
        module: Fully initialized native executor module.
    """
    PyExecutor = module.PyExecutor
    original: Callable[..., Any] = PyExecutor.start_worker

    @functools.wraps(original)
    def start_worker(executor: Any) -> Any:
        """Enter the experimental boundary before starting native worker threads."""
        purpose = _PURPOSES.get(executor.model_engine)
        if purpose == "memory_profiling":
            return original(executor)
        if purpose != "final_executor":
            raise RuntimeError("Cannot identify the final executor capture boundary")
        _park(executor)
        return original(executor)

    PyExecutor.start_worker = start_worker


class _ObserverLoader(importlib.abc.Loader):
    """Delegate normal module loading, then install only its relevant observers."""

    def __init__(
        self, loader: importlib.abc.Loader, callback: Callable[[ModuleType], None]
    ) -> None:
        """Retain the original loader and post-import callback."""
        self.loader = loader
        self.callback = callback

    def create_module(self, spec: Any) -> ModuleType | None:
        """Delegate module creation without importing any runtime dependency."""
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        """Execute the original module before modifying its completed classes."""
        self.loader.exec_module(module)
        self.callback(module)


class _ObserverFinder(importlib.abc.MetaPathFinder):
    """Observe existing imports without moving Torch/MPI initialization."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        """Wrap only the exact loader/executor modules when Python requests them."""
        callbacks = {
            "tensorrt_llm._torch.pyexecutor.model_loader": _install_model_observers,
            "tensorrt_llm._torch.pyexecutor.model_engine": _install_engine_observers,
            "tensorrt_llm._torch.pyexecutor.py_executor": _install_executor_observers,
        }
        if fullname not in callbacks:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot instrument {fullname}")
        spec.loader = _ObserverLoader(spec.loader, callbacks[fullname])
        return spec


if os.environ.get("TRTLLM_SNAPSHOT_EXPERIMENT") == "1":
    sys.meta_path.insert(0, _ObserverFinder())
