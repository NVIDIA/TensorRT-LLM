# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The warnings a spec gets when it is written and when a run starts; none stops anything."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import deployments as deployments_mod
from .deployments import Deployment

#: The share of a device's memory an engine's weights may take: the rest holds the KV
#: cache, activations and CUDA graphs.
ENGINE_WEIGHT_FRACTION = 0.75
#: Where a config states its multi-token-prediction layers.
DRAFT_LAYER_KEYS = ("num_nextn_predict_layers", "num_mtp_modules", "mtp_num_hidden_layers")
#: What marks a config as a hybrid model whose layers keep recurrent state: a key that
#: configures it, or a per-layer type that names it.
_RECURRENT_KEYS = ("linear_attn_config", "linear_num_value_heads", "mamba_ssm_cfg", "ssm_cfg")
_RECURRENT_WORDS = ("mamba", "linear", "ssm", "delta", "recurrent")


def checkpoint_weight_bytes(checkpoint: str | Path | None) -> int | None:
    """The checkpoint's weights in bytes: its safetensors index's total, else its files' sizes."""
    root = Path(str(checkpoint or ""))
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        total = int((index.get("metadata") or {}).get("total_size") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        total = 0
    try:
        return total or sum(f.stat().st_size for f in root.glob("*.safetensors")) or None
    except OSError:
        return None


def unfit(task_data: Mapping[str, Any], deployment: Deployment, memory_mib: float | None) -> str:
    """Why ``deployment``'s narrowest server cannot hold the weights on such devices; else ``""``."""
    weights = checkpoint_weight_bytes(task_data.get("checkpoint_path"))
    ranks = min(server.ranks for server in deployment.servers.values())
    room = ENGINE_WEIGHT_FRACTION * (memory_mib or 0) * 2**20 * ranks
    if not weights or not memory_mib or weights <= room:
        return ""
    return (
        f"`{deployment.name}` cannot hold the weights: {weights / 2**30:.0f} GiB on {ranks} "
        f"device(s) of {memory_mib / 1024:.0f} GiB is more than {ENGINE_WEIGHT_FRACTION:.0%} of "
        f"them, which leaves the KV cache and CUDA graphs no room; give it more ranks"
    )


def cgroup_memory_limit(proc_cgroup: str | None = None, root: Path = Path("/sys/fs/cgroup")):
    """The tightest memory limit on this process's own cgroup and its ancestors, in bytes.

    A container or a Slurm step sets the limit on a cgroup of its own, not on the mount's
    root: each path of ``/proc/self/cgroup`` (v2 ``0::<path>``, v1 with ``memory`` among its
    controllers) is walked up to its mount. ``None`` when nothing limits it.
    """
    if proc_cgroup is None:
        try:
            proc_cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        except OSError:
            proc_cgroup = ""
    files = [root / "memory.max", root / "memory" / "memory.limit_in_bytes"]
    for line in proc_cgroup.splitlines():
        controllers, _, rel = line.partition(":")[2].partition(":")
        if controllers and "memory" not in controllers.split(","):
            continue
        base, name = (
            (root / "memory", "memory.limit_in_bytes") if controllers else (root, "memory.max")
        )
        here = base / rel.strip().lstrip("/")
        while here != base.parent:
            files.append(here / name)
            here = here.parent
    limits = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:  # "max", or v1's huge number, is no limit
            limits.append(int(text))
    return min(limits) if limits else None


def checkpoint_notes(task_data: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Where the checkpoint's ``config.json`` cannot meet a deployment as stated, one note each.

    A server's ranks split the attention heads, the KV heads and the experts; a deployment
    that drafts has draft layers; one that takes images has a tower; a quantized format
    runs on its architecture; block reuse on a hybrid checkpoint needs a snapshot policy.
    Notes, never a stop: the engine's own load decides, and a boot shows it.
    """
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    experts = next(
        (
            config[k]
            for k in ("num_experts", "n_routed_experts", "num_local_experts")
            if config.get(k)
        ),
        None,
    )
    drafts = any(config.get(k) for k in DRAFT_LAYER_KEYS) or any(
        str(k).startswith("dspark") for k in config
    )
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else {}
    layers = [str(t).lower() for block in (config, text) for t in block.get("layer_types") or ()]
    hybrid = any(block.get(k) for block in (config, text) for k in _RECURRENT_KEYS) or any(
        word in layer for layer in layers for word in _RECURRENT_WORDS
    )
    said: list[str] = []
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            who = deployment.name if role == "server" else f"{deployment.name}.{role}"
            options, kv = server.options, server.options.get("kv_cache_config") or {}
            ranks = int(options.get("tensor_parallel_size") or 1)
            split = int(options.get("moe_expert_parallel_size") or 1)
            # Attention DP replicates attention; without it the ranks split its heads.
            if ranks > 1 and not options.get("enable_attention_dp"):
                if heads and heads % ranks:
                    said.append(f"{who}: {heads} attention heads do not split {ranks} ways")
                if kv_heads and kv_heads % ranks and ranks % kv_heads:
                    said.append(f"{who}: {kv_heads} KV heads do not split {ranks} ways")
            if split > 1 and experts and experts % split:
                said.append(f"{who}: {experts} experts do not split {split} ways")
            drafter = str((options.get("speculative_config") or {}).get("decoding_type") or "")
            if drafter.lower() in ("mtp", "dspark") and not drafts:
                said.append(
                    f"{who}: it drafts with {drafter}, and config.json states no draft layers"
                )
            snapshots = kv.get("mamba_state_cache_interval") or any(
                (kv.get("mamba_state_config") or {}).values()
            )
            if hybrid and kv.get("enable_block_reuse") and not snapshots:
                said.append(
                    f"{who}: block reuse on a hybrid checkpoint with no Mamba state snapshot policy: "
                    f"the engine turns reuse off, and a boot's repeated prompt then resumes nothing; "
                    f"set `kv_cache_config.mamba_state_config.periodic_snapshot_interval`"
                )
        if "image" in deployment.inputs and not {"vision_config", "audio_config"} & set(config):
            said.append(f"{deployment.name}: it takes images, and the checkpoint has no tower")
    arch = str(task_data.get("arch", ""))[3:]
    algo = str((config.get("quantization_config") or {}).get("quant_algo") or "").lower()
    if "fp4" in algo and arch.isdigit() and int(arch) < 100:
        said.append(f"the checkpoint's {algo} weights need sm_100 or later")
    return [f"checkpoint: {line}" for line in dict.fromkeys(said)]


def spec_warnings(task_data: Mapping[str, Any]) -> list[str]:
    """What the spec is likely to cost or miss, one note each; never a stop.

    The first deployment holds the weights on too few devices
    (``environment.gpu_memory_gib`` says how much one has) or has no anchor (``onboarded``
    can then only mean it boots); a deployment is scored without CUDA graphs; a declared
    anchor is 1 or less on the percent scale; engines size their KV host tier past this
    process's memory limit.
    """
    from ..checks import anchors  # read here: the anchors read the deployments of this package

    try:
        found = deployments_mod.resolve(task_data)
    except Exception:  # noqa: BLE001 - advice never stops what it advises on
        return []
    said: list[str] = []
    core = found[0]
    memory = (task_data.get("environment") or {}).get("gpu_memory_gib")
    narrow = unfit(task_data, core, memory * 1024 if isinstance(memory, (int, float)) else None)
    if narrow:
        said.append(f"first deployment too narrow: {narrow}")
    if not core.benchmarks:
        said.append(f"no accuracy anchor: `{core.name}` names no benchmark: nothing judges it")
    for deployment in found:
        for benchmark in deployment.benchmarks:
            who = f"`{deployment.name}` on `{benchmark.name}`"
            anchor = anchors.resolve(task_data, deployment, benchmark)
            if deployment is core and anchor.kind == anchors.NONE:
                said.append(
                    f"no accuracy anchor: {who} has {anchor.source}, so the run cannot judge its "
                    f"accuracy; measure one with an implementation that already serves the "
                    f"checkpoint and write the row, or declare `anchor` with its `source`"
                )
            if (
                benchmark.anchor is not None
                and benchmark.anchor <= 1
                and benchmark.unit == "percent"
            ):
                said.append(
                    f"anchor under one percent: {who} declares {benchmark.anchor:g} with `unit: "
                    f"percent`; if the benchmark reports a fraction, state `unit: fraction`, or "
                    f"every score passes"
                )
    servers = [(d, s.options) for d in found for s in d.servers.values()]
    eager = [d.name for d, o in servers if d.benchmarks and o.get("cuda_graph_config", 0) is None]
    if eager:
        said.append(
            f"accuracy without CUDA graphs: {', '.join(dict.fromkeys(eager))} scored with "
            f"`cuda_graph_config: null`: a full benchmark takes 10-50x longer, and what is scored "
            f"is not the captured path the target serves"
        )
    options = [(d.name, o.get("kv_cache_config") or {}) for d, o in servers]
    unbounded = list(dict.fromkeys(name for name, kv in options if "host_cache_size" not in kv))
    limit = cgroup_memory_limit()
    try:
        free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        free = None
    if unbounded and limit is not None and free is not None and limit < free:
        said.append(
            f"host memory: {', '.join(unbounded)}: set no `kv_cache_config.host_cache_size`. The "
            f"engine then sizes a KV host tier from the node's free memory ({free / 2**30:.0f} "
            f"GiB), while this process may use {limit / 2**30:.0f} GiB (its cgroup's limit): "
            f"every engine a check or a worker starts pins that much host memory, and a few at "
            f"once are OOM-killed. Set `host_cache_size` (bytes, e.g. 8589934592) in the options."
        )
    return said
