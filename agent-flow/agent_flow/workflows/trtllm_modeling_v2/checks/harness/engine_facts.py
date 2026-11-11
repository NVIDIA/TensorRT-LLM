# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public engine capacity, plus a deliberately isolated upstream worker observation.

Only get_kv_cache_capacity is a public resolved-engine API. Complete worker
configuration has no public reporting API: the single rankhook calls
read_worker after setup and labels its records upstream-only. LLM.args and
requested constructor options are never presented as actual configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import workload_io as wio  # noqa: E402 -- imported by path, see the package docstring

_MIB = float(1 << 20)


def _plain(value: Any) -> Any:
    """``value`` as JSON takes it: an enum by its value, anything else not plain as text."""
    value = getattr(value, "value", value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _dtype(value: Any) -> str | None:
    """A dtype by its name: ``torch.float32`` and ``"torch.float32"`` are ``float32``."""
    value = _plain(value)
    return None if value is None else str(value).replace("torch.", "")


def _read(target: dict[str, Any], errors: dict[str, str], name: str, read: Callable[[], Any]):
    """``target[name] = read()``, or the reason it could not be read in ``errors``."""
    try:
        target[name] = read()
    # The objects read here are the engine's internals, which differ from one
    # TensorRT-LLM tree to the next; a field that cannot be read is reported
    # as such and must never stop the gate's own workload.
    except Exception as exc:  # noqa: BLE001
        errors[name] = f"{type(exc).__name__}: {exc}"[:200]


def snapshot_policy(state: Any) -> dict[str, Any]:
    """A ``MambaStateConfig`` as the three settings a snapshot policy is made of."""
    return {
        "periodic_snapshot_interval": int(state.periodic_snapshot_interval or 0),
        "additional_snapshot_offsets_from_start": [
            int(o) for o in state.additional_snapshot_offsets_from_start or ()
        ],
        "additional_snapshot_offsets_from_end": [
            int(o) for o in state.additional_snapshot_offsets_from_end or ()
        ],
    }


def _cuda_graph(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "batch_sizes": sorted(int(b) for b in config.batch_sizes or ()),
        "enable_padding": bool(config.enable_padding),
    }


def _speculative(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "algorithm": str(_plain(config.decoding_type)),
        "max_draft_len": int(config.max_draft_len),
    }


def _expert_parallel(parallel: Any) -> int:
    """The MoE expert-parallel width the engine's ``Mapping`` resolves."""
    try:
        return int(parallel.to_mapping().moe_ep_size)
    except (AttributeError, TypeError, ValueError, AssertionError):
        pass
    ep, tp = int(parallel.moe_ep_size), int(parallel.moe_tp_size)
    if ep > 0:
        return ep
    if tp == -1:
        # Both unset: `Mapping` puts every expert on every rank.
        return 1
    raise ValueError(f"moe_ep_size {ep} with moe_tp_size {tp} is resolved only by Mapping")


def read_args(args: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """``(config, errors)``: the attested fields of one ``TorchLlmArgs``."""
    config: dict[str, Any] = {}
    errors: dict[str, str] = {}
    fields: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("modeling_v2", lambda: _plain(args.modeling_v2)),
        ("block_reuse", lambda: bool(args.kv_cache_config.enable_block_reuse)),
        ("mamba_snapshot", lambda: snapshot_policy(args.kv_cache_config.mamba_state_config)),
        ("kv_cache_dtype", lambda: _dtype(args.kv_cache_config.dtype)),
        ("mamba_ssm_cache_dtype", lambda: _dtype(args.kv_cache_config.mamba_ssm_cache_dtype)),
        ("cuda_graph", lambda: _cuda_graph(args.cuda_graph_config)),
        ("speculative", lambda: _speculative(args.speculative_config)),
        ("max_num_tokens", lambda: _plain(args.max_num_tokens)),
        ("max_batch_size", lambda: _plain(args.max_batch_size)),
        ("max_seq_len", lambda: _plain(args.max_seq_len)),
        ("chunked_prefill", lambda: bool(args.enable_chunked_prefill)),
        ("overlap_scheduler", lambda: not bool(args.disable_overlap_scheduler)),
        ("tp", lambda: int(args.parallel_config.tp_size)),
        ("pp", lambda: int(args.parallel_config.pp_size)),
        ("ep", lambda: _expert_parallel(args.parallel_config)),
        ("attention_dp", lambda: bool(args.parallel_config.enable_attention_dp)),
    )
    for name, read in fields:
        _read(config, errors, name, read)
    return config, errors


def read_worker(worker: Any) -> dict[str, Any]:
    """What one executor worker runs, read as its ``setup_engine`` returns."""
    config, errors = read_args(worker.llm_args)
    engine = getattr(worker, "engine", None)
    model_engine = getattr(engine, "model_engine", None)
    kv = getattr(engine, "kv_cache_manager", None)

    def quant() -> Any:
        return model_engine.model.model_config.quant_config

    if kv is not None and hasattr(kv, "enable_block_reuse"):
        _read(config, errors, "block_reuse", lambda: bool(kv.enable_block_reuse))
    _read(config, errors, "kv_cache_quant", lambda: _plain(quant().kv_cache_quant_algo))
    _read(
        config,
        errors,
        "mamba_ssm_cache_dtype_resolved",
        lambda: _dtype(quant().mamba_ssm_cache_dtype),
    )
    _read(
        config,
        errors,
        "cuda_graph_captured",
        lambda: [int(b) for b in model_engine._cuda_graph_batch_sizes],
    )

    pools: dict[str, Any] = {}
    _read(
        pools,
        errors,
        "kv_budget_bytes",
        lambda: _plain(worker.llm_args.kv_cache_config.max_gpu_total_bytes),
    )
    if kv is not None:

        def kv_tokens() -> int:
            stats = kv.get_kv_cache_stats()
            per_block = int(getattr(stats, "tokens_per_block", 0) or kv.tokens_per_block)
            return int(stats.max_num_blocks) * per_block

        _read(pools, errors, "kv_max_tokens", kv_tokens)

    memory: dict[str, Any] = {}

    def free_mib() -> dict[str, float]:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {"free_mib": round(free / _MIB, 1), "total_mib": round(total / _MIB, 1)}

    _read(memory, errors, "device", free_mib)
    return {
        "rank": _plain(getattr(worker, "rank", None)),
        "config": config,
        "pools": pools,
        "memory": memory.get("device"),
        "errors": errors,
    }


def kv_capacity(llm: Any) -> dict[str, Any]:
    """Rank 0's primary KV-cache pool, as its executor reports it; empty when it cannot.

    ``LLM.get_kv_cache_capacity`` (``{"maxNumBlocks", "tokensPerBlock",
    "maxNumTokens"}``) asks rank 0's executor, in this process or over the
    proxy's RPC channel, so it answers at every world size.
    """
    read = getattr(llm, "get_kv_cache_capacity", None)
    if not callable(read):
        return {}
    try:
        capacity = read()
    # Asking a live engine is an observation; see `_read`.
    except Exception:  # noqa: BLE001
        return {}
    return dict(capacity) if isinstance(capacity, Mapping) else {}


def visible_devices(environ: Mapping[str, str], world_size: int) -> list[str] | None:
    """The devices this engine's ranks use on this node; ``None`` for the first ``world_size``."""
    text = environ.get("CUDA_VISIBLE_DEVICES")
    if text is None:
        return None
    entries = [e.strip() for e in text.split(",") if e.strip()]
    return entries[: max(1, world_size)]


def device_memory(world_size: int, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Free and total memory, in MiB, of each device this engine uses here, through NVML."""
    environ = os.environ if environ is None else environ
    try:
        import pynvml
    except ImportError as exc:
        return {"source": None, "devices": [], "error": f"pynvml: {exc}"}
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        return {"source": None, "devices": [], "error": f"nvmlInit: {exc}"}
    devices: list[dict[str, Any]] = []
    try:
        entries = visible_devices(environ, world_size)
        if entries is None:
            count = min(max(1, world_size), pynvml.nvmlDeviceGetCount())
            entries = [str(i) for i in range(count)]
        for entry in entries:
            if entry.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(entry))
            else:
                handle = pynvml.nvmlDeviceGetHandleByUUID(entry)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                {
                    "device": entry,
                    "free_mib": round(info.free / _MIB, 1),
                    "total_mib": round(info.total / _MIB, 1),
                }
            )
    except pynvml.NVMLError as exc:
        return {"source": "nvml", "devices": devices, "error": str(exc)}
    finally:
        pynvml.nvmlShutdown()
    return {"source": "nvml", "devices": devices}


def emit(llm: Any, *, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Report public capacity; actual configuration is UNKNOWN without rank records."""
    capacity = kv_capacity(llm)
    world = int(requested.get("tensor_parallel_size", 1)) * int(
        requested.get("pipeline_parallel_size", 1)
    )
    record = {
        "config": {},
        "sources": {},
        "requested": dict(requested),
        "pools": {"kv_max_tokens": capacity.get("maxNumTokens")},
        "public": {"kv_cache_capacity": capacity},
        "memory": device_memory(world),
        "note": "resolved configuration requires the upstream-only rank hook; public LLM has no full worker report",
    }
    wio.emit(wio.ENGINE_MARKER, record)
    return record


def worker_rank(worker: Any) -> int | None:
    """The MPI rank of ``worker``: its own ``rank``, else the launcher's rank variables."""
    rank = getattr(worker, "rank", None)
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        text = os.environ.get(name, "")
        if text.isdigit():
            return int(text)
    return None


def emit_rank(worker: Any) -> None:
    """Print the ``MV2FLOW_RANK_ENGINE`` record of one rank's worker, as its engine is up.

    A worker that cannot be read prints ``{"rank", "error"}``, which the gate
    reports as a rank it could not attest.
    """
    try:
        record = read_worker(worker)
        record["rank"] = worker_rank(worker)
        record["source"] = "upstream-only: BaseWorker.setup_engine"
    except Exception as exc:  # noqa: BLE001 -- an observation; see `emit`
        record = {"rank": worker_rank(worker), "error": f"{type(exc).__name__}: {exc}"[:500]}
    wio.emit(wio.RANK_ENGINE_MARKER, record)
