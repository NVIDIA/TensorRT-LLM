# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``perf:`` block of ``task.yaml``: its keys and their validation."""

from __future__ import annotations

from typing import Any, Mapping

from .sol_consensus import METHODS

PERF_KEYS = {
    "enabled", "model", "target", "identity_configs", "workload", "spools", "dev_spools",
    "env_script", "campaign", "source_embargo", "sister_engines", "wall", "configs", "goal", "sol",
}  # fmt: skip
#: Keys the campaign no longer reads: it starts once onboarding is complete and measures one
#: baseline, with no stock reference beside it and no repeat of it.
REMOVED_PERF_KEYS = {
    "reference": "no stock reference is measured; a goal is a ratio to the baseline",
    "noise_repeats": "the baseline is measured once",
}
SOL_KEYS = {"enabled", "gpu", "methods", "tolerance"}
GOAL_KEYS = {"ratio_to_baseline", "pct_of_sol"}
#: ``perf.campaign``'s projection and checkpoint thresholds, each a fraction (0.05 is 5 %).
CAMPAIGN_FRACTIONS = ("min_projected_gain", "micro_noise", "module_tolerance", "checkpoint_gain")
#: ``perf.campaign.logits``: the first-token check at each checkpoint (``tune/logits_check.py``).
LOGITS_KEYS = {"prompts", "top_k", "max_kl", "min_top1", "tie_margin"}


def validate_perf_block(data: Mapping[str, Any]) -> list[str]:
    """Closed keys, batched errors -- the task schema's style."""
    perf = data.get("perf")
    if perf is None:
        return []
    if not isinstance(perf, Mapping):
        return ["perf: must be a mapping"]
    errors = [
        f"perf.{k}: removed: {REMOVED_PERF_KEYS[k]}"
        if k in REMOVED_PERF_KEYS
        else f"perf.{k}: unknown key"
        for k in perf
        if k not in PERF_KEYS
    ]
    workload = perf.get("workload") or {}
    for key in ("isl", "osl"):
        if key in workload and (not isinstance(workload[key], int) or workload[key] <= 0):
            errors.append(f"perf.workload.{key}: a positive integer")
    con = workload.get("concurrency", [1])
    if not (isinstance(con, list) and con and all(isinstance(c, int) and c > 0 for c in con)):
        errors.append("perf.workload.concurrency: a non-empty list of positive integers")
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
                errors.append("perf.sol.gpu: a hardware label only; capacities need public sources")
            methods = sol.get("methods", list(METHODS))
            if (
                not isinstance(methods, list)
                or not methods
                or len(set(map(str, methods))) != len(methods)
                or any(m not in METHODS for m in methods)
            ):
                errors.append(f"perf.sol.methods: distinct analyses among {list(METHODS)}")
            if not _fraction(sol.get("tolerance", 0.15), 1.0) or not sol.get("tolerance", 0.15):
                errors.append(
                    "perf.sol.tolerance: how far analyses may differ and agree, in (0, 1]"
                )
    campaign = perf.get("campaign") or {}
    if campaign and not isinstance(campaign, Mapping):
        errors.append("perf.campaign: a mapping")
    elif campaign:
        errors += validate_campaign_accuracy(campaign.get("accuracy"))
        errors += _validate_max_hours(campaign.get("max_hours"))
        errors += _validate_checkpoints(campaign)
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


def _fraction(value: Any, top: float = float("inf")) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= top


def _validate_checkpoints(campaign: Mapping[str, Any]) -> list[str]:
    """The projection's thresholds, how often a checkpoint comes, and its first-token check."""
    errors = [
        f"perf.campaign.{key}: a fraction >= 0 (0.05 is 5%)"
        for key in CAMPAIGN_FRACTIONS
        if key in campaign and not _fraction(campaign[key])
    ]
    keeps = campaign.get("checkpoint_keeps", 3)
    if isinstance(keeps, bool) or not isinstance(keeps, int) or keeps < 1:
        errors.append("perf.campaign.checkpoint_keeps: a positive number of keeps")
    logits = campaign.get("logits")
    if logits is None:
        return errors
    if not isinstance(logits, Mapping):
        return errors + [f"perf.campaign.logits: a mapping of {sorted(LOGITS_KEYS)}"]
    errors += [f"perf.campaign.logits.{k}: unknown key" for k in logits if k not in LOGITS_KEYS]
    prompts = logits.get("prompts", ["x"])
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p.strip() for p in prompts)
    ):
        errors.append("perf.campaign.logits.prompts: a non-empty list of prompts")
    top_k = logits.get("top_k", 20)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        errors.append("perf.campaign.logits.top_k: a positive integer")
    for key, top in (("max_kl", float("inf")), ("min_top1", 1.0), ("tie_margin", float("inf"))):
        if key in logits and not _fraction(logits[key], top):
            errors.append(f"perf.campaign.logits.{key}: a number in [0, {top:g}]")
    return errors


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
    """``perf.goal``: a ratio to the baseline, a fraction of SOL, or both."""
    goal = perf.get("goal")
    if goal is None:
        return []
    if not isinstance(goal, Mapping) or not goal or set(goal) - GOAL_KEYS:
        return ["perf.goal: a mapping with ratio_to_baseline and/or pct_of_sol"]
    errors = []
    ratio = goal.get("ratio_to_baseline")
    if "ratio_to_baseline" in goal and (not isinstance(ratio, (int, float)) or ratio <= 0):
        errors.append("perf.goal.ratio_to_baseline: a positive number")
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
    """``perf.configs``: named workloads covered by serving validation."""
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
