# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feasibility checks: can this checkpoint, on this device, meet this request (INTAKE.md section 6).

Each check reads the request and the checkpoint's ``config.json`` and says
``pass``, ``warn`` or ``block``. A ``block`` is an open question: the task
cannot be confirmed while one stands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    GOAL_ADD_FEATURES,
    GOAL_ONBOARD,
    GOAL_TUNE,
    STATUS_OUT,
    STATUS_REQUIRED,
    feature_entry,
    feature_status,
    validate_task,
)

#: Speculative algorithms whose draft model is a checkpoint of its own.
DRAFTER_ALGOS: tuple[str, ...] = ("eagle3", "dflash")

PASS, WARN, BLOCK = "pass", "warn", "block"
_ARCH = re.compile(r"^sm_(\d+)$")


@dataclass(frozen=True)
class Finding:
    level: str
    area: str
    detail: str
    fix: str = ""

    def render(self) -> str:
        fix = f" -- {self.fix}" if self.fix else ""
        return f"[{self.level}] {self.area}: {self.detail}{fix}"


def read_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """The checkpoint's ``config.json``, its ``text_config`` lifted to the top; None if unreadable."""
    path = Path(str(checkpoint)) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # A multimodal checkpoint keeps the language model's shape under text_config.
    if isinstance(data.get("text_config"), dict):
        data = {**data["text_config"], **{k: v for k, v in data.items() if k != "text_config"}}
    return data


def _config(task: Mapping[str, Any]) -> dict[str, Any] | None:
    return read_config((task.get("model") or {}).get("checkpoint", ""))


#: Where a config states its multi-token-prediction layers, as ``onboard/survey.py`` reads them.
DRAFT_LAYER_KEYS: tuple[str, ...] = (
    "num_nextn_predict_layers",
    "num_mtp_modules",
    "mtp_num_hidden_layers",
)


def draft_layers(config: Mapping[str, Any]) -> int:
    """How many draft (MTP) layers the checkpoint carries; 0 for none."""
    for key in DRAFT_LAYER_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _goals(task: Mapping[str, Any]) -> list[Finding]:
    goals = set(task.get("goals") or ())
    out = []
    if not goals:
        out.append(Finding(BLOCK, "goals", "no goal stated", "onboard, add_features and/or tune"))
    if {GOAL_ONBOARD, GOAL_ADD_FEATURES} <= goals:
        out.append(
            Finding(
                BLOCK,
                "goals",
                "onboard and add_features together",
                "onboard builds the target; list the features under it",
            )
        )
    if goals and GOAL_ONBOARD not in goals and not (task.get("model") or {}).get("base"):
        out.append(Finding(BLOCK, "goals", "no onboard goal and no model.base to start from"))
    if GOAL_TUNE in goals and not (task.get("perf") or task.get("workload")):
        out.append(Finding(BLOCK, "goals", "tune needs a workload or perf.configs"))
    return out


def _speculative(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    status = feature_status(task, "speculative")
    if status in (None, STATUS_OUT):
        if draft_layers(config) and status is None:
            return [
                Finding(
                    BLOCK,
                    "speculative",
                    "the checkpoint carries draft layers the request does not mention",
                    "say whether speculative decoding is required or out_of_scope",
                )
            ]
        return []
    algo = feature_entry(task["features"]["speculative"]).get("params", {}).get("algo")
    dspark = any(k.startswith("dspark") for k in config)
    if algo == "mtp" and dspark:
        return [
            Finding(
                BLOCK,
                "speculative",
                "the checkpoint's draft stages are a DSpark drafter (dspark_* keys)",
                "use algo: dspark",
            )
        ]
    if algo in ("mtp", "dspark") and not draft_layers(config):
        return [
            Finding(
                BLOCK,
                "speculative",
                f"{algo} requested; config states no draft layers ({', '.join(DRAFT_LAYER_KEYS)})",
            )
        ]
    block = config.get("dspark_block_size")
    draft_len = feature_entry(task["features"]["speculative"]).get("params", {}).get("draft_len")
    if algo == "dspark" and block and draft_len != block:
        return [
            Finding(
                BLOCK,
                "speculative",
                f"the DSpark drafter proposes blocks of {block} tokens; draft_len is {draft_len}",
                f"set features.speculative.params.draft_len: {block}",
            )
        ]
    if algo in DRAFTER_ALGOS:
        drafter = feature_entry(task["features"]["speculative"]).get("params", {}).get("drafter")
        if not drafter:
            return [
                Finding(
                    BLOCK,
                    "speculative",
                    f"{algo} drafts with a checkpoint of its own, and none is named",
                    "set features.speculative.params.drafter (drafter=<dir> in --goal)",
                )
            ]
        if not (Path(str(drafter)) / "config.json").is_file():
            return [
                Finding(BLOCK, "speculative", f"the {algo} drafter {drafter} has no config.json")
            ]
        return [Finding(PASS, "speculative", f"{algo} drafter {drafter}")]
    return [Finding(PASS, "speculative", f"{algo} weights present")]


def _multimodal(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    status = feature_status(task, "multimodal")
    towers = [k for k in ("vision_config", "audio_config") if k in config]
    if status == STATUS_REQUIRED and not towers:
        return [
            Finding(
                BLOCK, "multimodal", "required, but the checkpoint has no vision or audio tower"
            )
        ]
    if status is None and towers:
        return [
            Finding(
                BLOCK,
                "multimodal",
                f"the checkpoint has {', '.join(towers)} the request does not mention",
                "say whether multimodal is required or out_of_scope",
            )
        ]
    return []


def _divisibility(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    out = []
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    experts = (
        config.get("num_experts")
        or config.get("n_routed_experts")
        or config.get("num_local_experts")
    )
    device = task.get("device") or {}
    per_node = device.get("gpus_per_node")
    capacity = per_node * (device.get("nodes") or 1) if isinstance(per_node, int) else None
    for p in task.get("parallelism") or ():
        kind, world = p["kind"], p["world"]
        name = f"{kind}{world}"
        if capacity is not None and world > capacity:
            out.append(
                Finding(BLOCK, "parallelism", f"{name}: {world} GPUs, the device has {capacity}")
            )
        # Attention DP replicates attention; tp and tep split its heads.
        if kind in ("tp", "tep") and heads and heads % world:
            out.append(
                Finding(BLOCK, "parallelism", f"{name}: {heads} heads do not split {world} ways")
            )
        if kind in ("tp", "tep") and kv_heads and kv_heads % world and world % kv_heads:
            out.append(
                Finding(
                    BLOCK, "parallelism", f"{name}: {kv_heads} KV heads do not split {world} ways"
                )
            )
        if kind in ("tep", "dep"):
            if not experts:
                out.append(
                    Finding(BLOCK, "parallelism", f"{name}: expert parallelism on a dense model")
                )
            elif experts % world:
                out.append(
                    Finding(
                        BLOCK, "parallelism", f"{name}: {experts} experts do not split {world} ways"
                    )
                )
    return out


def _length(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    limit = config.get("max_position_embeddings")
    if not limit:
        return [Finding(WARN, "length", "config.json states no max_position_embeddings")]
    scaling = config.get("rope_scaling") or {}
    factor = float(scaling.get("factor", 1.0)) if isinstance(scaling, Mapping) else 1.0
    reach = int(limit * max(factor, 1.0)) if scaling else int(limit)
    out = []
    workload = task.get("workload") or {}
    wanted = workload.get("max_seq_len") or (workload.get("isl", 0) + workload.get("osl", 0))
    if wanted and wanted > reach:
        out.append(Finding(BLOCK, "length", f"isl + osl = {wanted} exceeds the model's {reach}"))
    if not wanted:
        out.append(
            Finding(
                WARN,
                "length",
                f"no workload: the engine would size its cache for {limit} positions",
                "state workload.isl/osl or max_seq_len",
            )
        )
    long = (task.get("features") or {}).get("long_context")
    if long and feature_entry(long).get("status") != STATUS_OUT:
        length = (feature_entry(long).get("params") or {}).get("length", 0)
        if length > reach:
            out.append(Finding(BLOCK, "long_context", f"{length} exceeds the model's {reach}"))
    return out


def _feature_members(task: Mapping[str, Any]) -> list[Finding]:
    """Every member a feature is narrowed to is one of the run's parallel configurations."""
    configs = {f"{p['kind']}{p['world']}" for p in task.get("parallelism") or ()} or {"tp1"}
    out = []
    for name, raw in (task.get("features") or {}).items():
        if name == "custom":
            continue
        entry = feature_entry(raw)
        members = entry.get("members")
        if members is None or entry.get("status") == STATUS_OUT:
            continue
        if not isinstance(members, list) or not members:
            out.append(Finding(BLOCK, name, "'members' must be a non-empty list"))
            continue
        absent = [str(m) for m in members if str(m) not in configs]
        if absent:
            out.append(
                Finding(
                    BLOCK,
                    name,
                    f"members {', '.join(absent)} are not among the run's "
                    f"{', '.join(sorted(configs))}",
                    "add them to parallelism, or narrow the feature to members the run has",
                )
            )
    return out


def _reuse_snapshots(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    """Block reuse on a hybrid checkpoint needs a Mamba state snapshot policy."""
    from ...run.preflight import hybrid_marker

    status = feature_status(task, "prefix_reuse")
    if status in (None, STATUS_OUT):
        return []
    marker = hybrid_marker(config)
    if marker is None:
        return []
    params = feature_entry(task["features"]["prefix_reuse"]).get("params") or {}
    if params.get("snapshot_interval"):
        return [
            Finding(
                PASS,
                "prefix_reuse",
                f"hybrid checkpoint ({marker}): state snapshots every "
                f"{params['snapshot_interval']} tokens",
            )
        ]
    return [
        Finding(
            BLOCK,
            "prefix_reuse",
            f"a hybrid checkpoint ({marker}) with block reuse and no Mamba state snapshot "
            f"policy: the engine turns reuse off, and every gate fails on its warning",
            "set features.prefix_reuse.params.snapshot_interval (snapshot=<tokens> in --goal), "
            "or make prefix_reuse out_of_scope",
        )
    ]


def _quant_arch(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[Finding]:
    match = _ARCH.match(str((task.get("device") or {}).get("arch", "")))
    algo = str(((config.get("quantization_config") or {}).get("quant_algo") or "")).lower()
    if match and "fp4" in algo and int(match.group(1)) < 100:
        return [Finding(BLOCK, "quantization", f"{algo} needs sm_100 or later")]
    return []


def _perf_configs(task: Mapping[str, Any]) -> list[Finding]:
    out = []
    configs = {f"{p['kind']}{p['world']}" for p in task.get("parallelism") or ()} or {"tp1"}
    for entry in (task.get("perf") or {}).get("configs") or ():
        parallel = entry.get("parallel")
        if parallel and parallel not in configs:
            out.append(
                Finding(
                    BLOCK,
                    "perf",
                    f"config {entry.get('name')!r} runs at {parallel}, which is not among the "
                    f"run's {', '.join(sorted(configs))}",
                )
            )
        for name, on in (entry.get("features") or {}).items():
            if on in (True, "on") and feature_status(task, name) in (None, STATUS_OUT):
                out.append(
                    Finding(
                        BLOCK,
                        "perf",
                        f"config {entry.get('name')!r} turns on {name}, which is not a capability",
                    )
                )
    return out


def run_checks(task: Mapping[str, Any]) -> list[Finding]:
    """Every finding for ``task``.

    With shape errors, those and the open questions only: no other check is
    meaningful on a malformed request, and an open question often says what
    the shape is missing.
    """
    errors = validate_task(task)
    questions = task.get("open_questions") if isinstance(task, Mapping) else None
    asked = (
        [Finding(BLOCK, "open question", str(q)) for q in questions or ()]
        if isinstance(questions, list)
        else []
    )
    if errors:
        return [Finding(BLOCK, "schema", e) for e in errors] + asked
    findings = _goals(task) + asked
    config = _config(task)
    if config is None:
        findings.append(
            Finding(BLOCK, "checkpoint", "model.checkpoint has no readable config.json")
        )
    else:
        findings.append(
            Finding(PASS, "checkpoint", f"architecture {(config.get('architectures') or ['?'])[0]}")
        )
        for check in (
            _speculative,
            _multimodal,
            _divisibility,
            _length,
            _quant_arch,
            _reuse_snapshots,
        ):
            findings += check(task, config)
    findings += _feature_members(task)
    findings += _perf_configs(task)
    return findings


def blocking(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == BLOCK]


def render_findings(findings: list[Finding]) -> str:
    """Every finding on its own line, blocks first."""
    order = {BLOCK: 0, WARN: 1, PASS: 2}
    return "\n".join(f.render() for f in sorted(findings, key=lambda f: order.get(f.level, 3)))


__all__ = [
    "BLOCK",
    "DRAFTER_ALGOS",
    "DRAFT_LAYER_KEYS",
    "Finding",
    "PASS",
    "WARN",
    "blocking",
    "draft_layers",
    "read_config",
    "render_findings",
    "run_checks",
]
