# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``perf:`` block of ``task.yaml``: its keys and their validation."""

from __future__ import annotations

from typing import Any, Mapping

PERF_KEYS = {
    "enabled", "model", "target", "identity_configs", "workload", "reference", "spools", "dev_spools",
    "env_script", "noise_repeats", "campaign", "source_embargo", "sister_engines", "wall",
    "configs", "goal", "sol",
}  # fmt: skip
SOL_KEYS = {"enabled", "gpu"}
GOAL_KEYS = {"ratio_to_reference", "pct_of_sol"}


def validate_perf_block(data: Mapping[str, Any]) -> list[str]:
    """Closed keys, batched errors -- the task schema's style."""
    perf = data.get("perf")
    if perf is None:
        return []
    if not isinstance(perf, Mapping):
        return ["perf: must be a mapping"]
    errors = [f"perf.{k}: unknown key" for k in perf if k not in PERF_KEYS]
    workload = perf.get("workload") or {}
    for key in ("isl", "osl"):
        if key in workload and (not isinstance(workload[key], int) or workload[key] <= 0):
            errors.append(f"perf.workload.{key}: a positive integer")
    con = workload.get("concurrency", [1])
    if not (isinstance(con, list) and con and all(isinstance(c, int) and c > 0 for c in con)):
        errors.append("perf.workload.concurrency: a non-empty list of positive integers")
    if perf.get("reference", "stock") not in ("stock", "none"):
        errors.append("perf.reference: stock | none")
    for key in ("identity_configs", "spools"):
        if key in perf and not isinstance(perf[key], list):
            errors.append(f"perf.{key}: a list")
    errors += _validate_configs(perf.get("configs"))
    errors += _validate_goal(perf)
    sol = perf.get("sol")
    if sol is not None:
        if not isinstance(sol, Mapping):
            errors.append("perf.sol: a mapping {enabled, gpu}")
        else:
            errors += [f"perf.sol.{k}: unknown key" for k in sol if k not in SOL_KEYS]
            if not isinstance(sol.get("enabled", True), bool):
                errors.append("perf.sol.enabled: true or false")
            if "gpu" in sol and not isinstance(sol["gpu"], str):
                errors.append("perf.sol.gpu: the part name the SOL calculator prices, e.g. gb200")
    campaign = perf.get("campaign") or {}
    if campaign and not isinstance(campaign, Mapping):
        errors.append("perf.campaign: a mapping")
    elif campaign:
        errors += validate_campaign_accuracy(campaign.get("accuracy"))
        errors += _validate_max_hours(campaign.get("max_hours"))
        tolerance = campaign.get("ttft_tolerance", 0.1)
        if tolerance is not None and (
            isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0
        ):
            errors.append(
                "perf.campaign.ttft_tolerance: how much slower a first token may come at "
                "concurrency 1, as a fraction (0.1 is 10%); null judges none"
            )
    return errors


def campaign_max_hours(campaign: Mapping[str, Any]) -> float | None:
    """``perf.campaign.max_hours``: the campaign's wall budget in hours; None for no budget."""
    hours = campaign.get("max_hours")
    return None if hours is None else float(hours)


def _validate_max_hours(hours: Any) -> list[str]:
    if hours is None:
        return []
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not hours > 0:
        return [
            "perf.campaign.max_hours: a positive number of hours, e.g. 8; omit it for no budget"
        ]
    return []


#: What ``perf.campaign.accuracy`` may say: which unit and members the keep gate runs at, or
#: that there is none. Everything else about the keep gate is the contract's.
CAMPAIGN_ACCURACY_KEYS = {"gate", "unit", "members"}
#: Keys the keep gate does not take, and the contract key that states each.
CONTRACT_ACCURACY_KEYS = {
    "anchor": "accuracy.reference_yaml, model_key and discriminators",
    "task": "accuracy.benchmark",
    "eval_args": "accuracy.benchmark_args",
}


def keep_gate_off(block: Mapping[str, Any] | None) -> bool:
    """``perf.campaign.accuracy.gate: off``: keeps are made without the accuracy gate.

    YAML reads a bare ``off`` as ``false``; both spellings mean off.
    """
    return isinstance(block, Mapping) and block.get("gate", "on") in (False, "off")


def validate_campaign_accuracy(block: Any) -> list[str]:
    """``perf.campaign.accuracy``: ``{unit, members}`` for the keep gate, or ``{gate: off}``."""
    if block is None:
        return []
    if not isinstance(block, Mapping):
        return ["perf.campaign.accuracy: a mapping {unit, members} or {gate: off}"]
    errors = []
    for key in block:
        if key in CONTRACT_ACCURACY_KEYS:
            errors.append(
                f"perf.campaign.accuracy.{key}: the keep gate is the contract's accuracy gate and "
                f"reads {CONTRACT_ACCURACY_KEYS[key]}; remove this key"
            )
        elif key not in CAMPAIGN_ACCURACY_KEYS:
            errors.append(f"perf.campaign.accuracy.{key}: unknown key")
    if block.get("gate", "on") not in (True, False, "on", "off"):
        errors.append("perf.campaign.accuracy.gate: on | off")
    if "unit" in block and not (isinstance(block["unit"], str) and block["unit"]):
        errors.append("perf.campaign.accuracy.unit: an assembly unit or variant name, e.g. core")
    members = block.get("members")
    if "members" in block and not (
        isinstance(members, list) and members and all(isinstance(m, str) and m for m in members)
    ):
        errors.append(
            "perf.campaign.accuracy.members: a non-empty list of members, e.g. [tp1, dep4]"
        )
    if keep_gate_off(block) and ("unit" in block or "members" in block):
        errors.append(
            "perf.campaign.accuracy: `gate: off` runs no unit or member; drop one or the other"
        )
    return errors


def _validate_goal(perf: Mapping[str, Any]) -> list[str]:
    """``perf.goal``: a ratio to the stock reference, a fraction of SOL, or both."""
    goal = perf.get("goal")
    if goal is None:
        return []
    if not isinstance(goal, Mapping) or not goal or set(goal) - GOAL_KEYS:
        return ["perf.goal: a mapping with ratio_to_reference and/or pct_of_sol"]
    errors = []
    ratio = goal.get("ratio_to_reference")
    if "ratio_to_reference" in goal:
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            errors.append("perf.goal.ratio_to_reference: a positive number")
        if perf.get("reference", "stock") != "stock":
            errors.append("perf.goal: needs reference: stock, the measurement it is a ratio of")
    if "pct_of_sol" in goal:
        pct = goal["pct_of_sol"]
        if not isinstance(pct, (int, float)) or not 0 < pct <= 1:
            errors.append("perf.goal.pct_of_sol: a fraction in (0, 1], e.g. 0.7")
        if not (perf.get("sol") or {}).get("enabled", True):
            errors.append(
                "perf.goal.pct_of_sol: needs perf.sol enabled, the ceilings it is a fraction of"
            )
    return errors


CONFIG_KEYS = {"name", "workload", "identity_configs", "llm_api_options", "constraint"}
CONSTRAINT_KEYS = {"tpot_ms_max", "ttft_ms_max"}


def _validate_configs(configs: Any) -> list[str]:
    """``perf.configs``: named workloads, each measured for every candidate."""
    if configs is None:
        return []
    if not isinstance(configs, list) or not configs:
        return ["perf.configs: a non-empty list of {name, workload, ...}"]
    errors, names = [], []
    for i, entry in enumerate(configs):
        where = f"perf.configs[{i}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{where}: a mapping")
            continue
        errors += [f"{where}.{k}: unknown key" for k in entry if k not in CONFIG_KEYS]
        name = entry.get("name")
        if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
            errors.append(f"{where}.name: a name of letters, digits, '-' and '_'")
        names.append(name)
        workload = entry.get("workload") or {}
        for key in ("isl", "osl"):
            if not isinstance(workload.get(key), int) or workload[key] <= 0:
                errors.append(f"{where}.workload.{key}: a positive integer")
        con = workload.get("concurrency")
        if not (isinstance(con, list) and con and all(isinstance(c, int) and c > 0 for c in con)):
            errors.append(f"{where}.workload.concurrency: a non-empty list of positive integers")
        constraint = entry.get("constraint") or {}
        errors += [
            f"{where}.constraint.{k}: one of {sorted(CONSTRAINT_KEYS)}"
            for k in constraint
            if k not in CONSTRAINT_KEYS
        ]
    if len(set(names)) != len(names):
        errors.append("perf.configs: names must be unique")
    return errors


__all__ = [
    "CAMPAIGN_ACCURACY_KEYS",
    "PERF_KEYS",
    "campaign_max_hours",
    "keep_gate_off",
    "validate_campaign_accuracy",
    "validate_perf_block",
]
