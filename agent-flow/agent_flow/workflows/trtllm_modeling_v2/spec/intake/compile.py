# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``task.yaml`` + ``site.yaml`` -> the resolved spec the pipeline runs (INTAKE.md section 8).

Deterministic: the same two files always compile to the same spec. Requirements
the pipeline cannot enforce yet are compiled into nothing and returned as
warnings, so the user sees which of their requirements no check will hold.

Each assembly unit gets an LLM-API options file (:attr:`Compiled.llm_api_options`,
written by :func:`compile_files` under :data:`LLM_API_DIRNAME`): ``trtllm-eval``
reads no runtime flags, so without one the accuracy gate measures the engine's
defaults rather than the configuration the other gates run, and a speculative
unit's accuracy is measured with no draft at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..task_schema import load_and_validate_task_yaml
from .schema import (
    GOAL_ADD_FEATURES,
    GOAL_ONBOARD,
    GOAL_TUNE,
    INCREMENT_FEATURES,
    PARALLEL_SEGMENT,
    STATUS_NICE,
    STATUS_OUT,
    UNGATED_FEATURES,
    IntakeError,
    feature_entry,
    validate_site,
    validate_task,
)

#: Core feature -> the runtime switch it sets.
_RUNTIME_SWITCH = {
    "cuda_graph": "cuda_graph",
    "overlap_scheduler": "overlap_scheduler",
    "chunked_prefill": "chunked_prefill",
    "prefix_reuse": "kv_cache_reuse",
}
REFERENCES_DIR = "tests/integration/defs/accuracy/references"
#: Where :func:`compile_files` writes each unit's LLM-API options, beside the spec.
LLM_API_DIRNAME = "llm_api"

#: Speculative algorithm -> its reference rows' ``spec_dec_algo`` and the LLM
#: API's ``speculative_config.decoding_type`` (the two spell them alike).
SPEC_DEC_ALGO = {"mtp": "MTP", "eagle3": "Eagle3", "dflash": "DFlash", "dspark": "DSpark"}
#: Weight formats a reference row states no ``quant_algo`` for.
_UNQUANTIZED = frozenset({"", "none", "auto", "bf16", "bfloat16", "fp16", "float16", "fp32"})
#: KV-cache dtype -> the reference rows' ``kv_cache_quant_algo``.
_KV_QUANT_ALGO = {"fp8": "FP8", "nvfp4": "NVFP4"}


@dataclass
class Compiled:
    spec: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    #: Assembly unit -> the LLM-API options its gates run under.
    llm_api_options: dict[str, dict[str, Any]] = field(default_factory=dict)


def unit_name(name: str, raw: Any) -> str | None:
    """The assembly unit an intake feature builds, or None for a runtime switch or a note."""
    entry = feature_entry(raw)
    if entry.get("status") == STATUS_OUT:
        return None
    if name == "speculative":
        return str((entry.get("params") or {}).get("algo") or "") or None
    return INCREMENT_FEATURES.get(name)


def _parallel(task: Mapping[str, Any]) -> dict[str, Any]:
    """The parallel segment and the member set, per unit where a feature narrows it."""
    configs = [f"{p['kind']}{p['world']}" for p in task.get("parallelism") or []]
    if not configs:
        return {"parallel": "tp1"}
    if len(configs) == 1:
        return {"parallel": configs[0]}
    narrowed: dict[str, list[str]] = {}
    for name, raw in (task.get("features") or {}).items():
        unit = unit_name(name, raw) if name != "custom" else None
        members = feature_entry(raw).get("members") if unit else None
        if unit and isinstance(members, list) and members:
            narrowed[unit] = [str(m) for m in members]
    if not narrowed:
        return {"parallel": "unified", "members": configs}
    return {"parallel": "unified", "members": {"default": configs, **narrowed}}


def _params(task: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = (task.get("features") or {}).get(name)
    if raw is None or feature_entry(raw).get("status") == STATUS_OUT:
        return {}
    return dict(feature_entry(raw).get("params") or {})


def _discriminators(task: Mapping[str, Any], built: list[str]) -> dict[str, Any] | None:
    """The reference row each unit anchors to: the weight and KV formats, and the speculator.

    A reference row matches only its exact discriminator set, so a run that
    names none anchors a quantized checkpoint to the bf16 row. Every unit from
    the speculative one on runs with its draft, so they anchor to its row too.
    """
    explicit = (task.get("accuracy") or {}).get("discriminators")
    if explicit is not None:
        return dict(explicit)
    base: dict[str, Any] = {}
    weights = str(((task.get("model") or {}).get("quantization") or {}).get("weights") or "")
    if weights.strip().lower() not in _UNQUANTIZED:
        base["quant_algo"] = weights.strip().upper()
    kv = str(_params(task, "kv_cache").get("dtype") or "").lower()
    if kv in _KV_QUANT_ALGO:
        base["kv_cache_quant_algo"] = _KV_QUANT_ALGO[kv]
    per_unit: dict[str, dict[str, Any]] = {}
    algo = None
    for unit in built:
        algo = SPEC_DEC_ALGO.get(unit, algo)
        if algo:
            per_unit[unit] = {**base, "spec_dec_algo": algo}
    if per_unit:
        return {"default": base, **per_unit}
    return base or None


def _llm_api_options(
    task: Mapping[str, Any], built: list[str], runtime: Mapping[str, Any], max_batch_size: Any
) -> dict[str, dict[str, Any]]:
    """Each unit's LLM-API options: the runtime switches, the KV cache, and its speculator.

    The capture list is stated here and nowhere in the runtime flags, so the
    quickstart gates derive theirs from it and both launch paths capture one
    list. A unit carries the speculator of every feature up to it, as its
    target implements them all.
    """
    kv_params = _params(task, "kv_cache")
    kv: dict[str, Any] = {}
    if str(kv_params.get("dtype") or "").lower() in _KV_QUANT_ALGO:
        kv["dtype"] = str(kv_params["dtype"]).lower()
    if kv_params.get("memory_fraction"):
        kv["free_gpu_memory_fraction"] = float(kv_params["memory_fraction"])
    if "kv_cache_reuse" in runtime:
        kv["enable_block_reuse"] = bool(runtime["kv_cache_reuse"])
    snapshot = _params(task, "prefix_reuse").get("snapshot_interval")
    if snapshot and runtime.get("kv_cache_reuse", True):
        kv["mamba_state_config"] = {"periodic_snapshot_interval": int(snapshot)}
    base: dict[str, Any] = {}
    if kv:
        base["kv_cache_config"] = kv
    if "chunked_prefill" in runtime:
        base["enable_chunked_prefill"] = bool(runtime["chunked_prefill"])
    chunk = _params(task, "chunked_prefill").get("chunk_tokens")
    if chunk:
        base["max_num_tokens"] = int(chunk)
    if runtime.get("overlap_scheduler") is False:
        base["disable_overlap_scheduler"] = True
    if runtime.get("cuda_graph"):
        graphs: dict[str, Any] = {"enable_padding": True}
        if max_batch_size:
            graphs["max_batch_size"] = int(max_batch_size)
        base["cuda_graph_config"] = graphs
    elif "cuda_graph" in runtime:
        base["cuda_graph_config"] = None

    speculative = _params(task, "speculative")
    options = {"core": dict(base)}
    draft: dict[str, Any] | None = None
    for unit in built:
        if unit in SPEC_DEC_ALGO:
            draft = {"decoding_type": SPEC_DEC_ALGO[unit]}
            draft["max_draft_len"] = int(speculative.get("draft_len") or 1)
            if speculative.get("drafter"):
                draft["speculative_model"] = str(speculative["drafter"])
        options[unit] = {**base, **({"speculative_config": dict(draft)} if draft else {})}
    return options


def _features(task: Mapping[str, Any], warnings: list[str]) -> tuple[list[str], dict, str]:
    """The pipeline's feature list, the runtime switches, and the scope."""
    features = task.get("features") or {}
    built: list[str] = []
    runtime: dict[str, Any] = {}
    scope = "full"
    for name, raw in features.items():
        if name == "custom":
            for item in raw or ():
                warnings.append(
                    f"custom feature {item.get('name', '?')!r}: no gate checks it; it reaches "
                    "the agents as a note"
                )
            continue
        entry = feature_entry(raw)
        status = entry.get("status")
        if status == STATUS_NICE:
            warnings.append(
                f"{name}: nice_to_have is built as required (the pipeline has no optional units)"
            )
        if name in _RUNTIME_SWITCH:
            runtime[_RUNTIME_SWITCH[name]] = status != STATUS_OUT
        elif name == "speculative" and status != STATUS_OUT:
            built.append(entry["params"]["algo"])
        elif name in INCREMENT_FEATURES:
            if status != STATUS_OUT:
                built.append(INCREMENT_FEATURES[name])
            elif name == "multimodal":
                scope = "text_only"
        elif name in UNGATED_FEATURES and status != STATUS_OUT:
            warnings.append(f"{name}: recorded; no gate checks it yet")
    if features.get("kv_cache"):
        dtype = (feature_entry(features["kv_cache"]).get("params") or {}).get("dtype")
        if dtype:
            runtime["kv_cache_dtype"] = dtype
    return built, runtime, scope


#: Feature parameters the compiled spec carries; any other is recorded and warned.
_COMPILED_PARAMS: dict[str, tuple[str, ...]] = {
    "kv_cache": ("dtype", "memory_fraction"),
    "chunked_prefill": ("chunk_tokens",),
    "prefix_reuse": ("snapshot_interval",),
    "speculative": ("algo", "draft_len", "drafter"),
}


def _unconsumed(task: Mapping[str, Any], warnings: list[str]) -> None:
    """Warn of each requirement the request states and the compiled spec does not carry."""
    for name, raw in (task.get("features") or {}).items():
        if name == "custom" or name in UNGATED_FEATURES:
            continue
        entry = feature_entry(raw)
        if entry.get("status") == STATUS_OUT:
            continue
        for key in sorted(set(entry.get("params") or {}) - set(_COMPILED_PARAMS.get(name, ()))):
            warnings.append(f"features.{name}.params.{key}: recorded; nothing compiles it yet")
        if entry.get("accept"):
            warnings.append(f"features.{name}.accept: recorded; the pipeline's gates decide")
        if entry.get("members") and unit_name(name, raw) is None:
            warnings.append(
                f"features.{name}.members: {name} is a runtime switch of every member, not "
                "a unit of its own, so it cannot be narrowed"
            )
    accuracy = task.get("accuracy") or {}
    if accuracy.get("reference"):
        warnings.append(
            "accuracy.reference: recorded; the anchor is the reference_yaml row, which a "
            "measurement must be written into first"
        )
    for key in sorted(set(accuracy.get("protocol") or {}) - {"chat_template"}):
        warnings.append(
            f"accuracy.protocol.{key}: recorded; only chat_template compiles "
            "(to --apply_chat_template)"
        )
    perf = task.get("perf") or {}
    if perf.get("metric"):
        warnings.append("perf.metric: recorded; the keep rule judges both headline metrics")
    if perf.get("baseline") not in (None, "stock"):
        warnings.append(
            f"perf.baseline {perf['baseline']!r}: the tune stage measures against the stock "
            "path or nothing; it compiles to reference: none"
        )
    if task.get("deliverable"):
        warnings.append("deliverable: recorded; delivery is the branch the run commits to")
    model = task.get("model") or {}
    for key in ("in_tree", "reference_impl"):
        if model.get(key):
            warnings.append(f"model.{key}: recorded; say how the agents use it in `notes`")


def compile_task(task: Mapping[str, Any], site: Mapping[str, Any]) -> Compiled:
    """The resolved spec for ``task`` on ``site``. Raises :class:`IntakeError` on a bad shape."""
    errors = validate_task(task) + validate_site(site)
    if errors:
        raise IntakeError("\n".join(errors))
    warnings: list[str] = []
    goals = list(task["goals"])
    model = task["model"]
    workload = task.get("workload") or {}
    if GOAL_ADD_FEATURES in goals:
        warnings.append(
            "add_features: the pipeline has no entry from an existing target; the run "
            "rebuilds core with the new features"
        )
    if GOAL_ONBOARD not in goals and GOAL_ADD_FEATURES not in goals and GOAL_TUNE in goals:
        warnings.append("tune only: run it with `trtllm-modeling-v2 tune` on model.base")

    built, runtime, scope = _features(task, warnings)
    _unconsumed(task, warnings)
    extra_flags: list[str] = []
    kv_dtype = runtime.pop("kv_cache_dtype", None)
    concurrency = workload.get("concurrency") or []
    max_seq_len = workload.get("max_seq_len") or (
        workload["isl"] + workload["osl"] if "isl" in workload and "osl" in workload else None
    )
    max_batch_size = workload.get("max_batch_size") or (max(concurrency) if concurrency else None)
    if max_seq_len:
        extra_flags.append(f"--max_seq_len={max_seq_len}")
    if max_batch_size:
        extra_flags.append(f"--max_batch_size={max_batch_size}")
    if kv_dtype:
        extra_flags.append(f"--kv_cache_dtype={kv_dtype}")
    chunk = _params(task, "chunked_prefill").get("chunk_tokens")
    if chunk:
        extra_flags.append(f"--max_num_tokens={chunk}")
    fraction = _params(task, "kv_cache").get("memory_fraction")
    if fraction:
        extra_flags.append(f"--kv_cache_fraction={fraction}")

    accuracy_in = task.get("accuracy") or {}
    benchmark = accuracy_in.get("benchmark", "gsm8k")
    accuracy: dict[str, Any] = {
        "benchmark": benchmark,
        "reference_yaml": accuracy_in.get("reference_yaml") or f"{REFERENCES_DIR}/{benchmark}.yaml",
    }
    if accuracy_in.get("model_key"):
        accuracy["model_key"] = accuracy_in["model_key"]
    if (accuracy_in.get("protocol") or {}).get("chat_template"):
        accuracy["benchmark_args"] = ["--apply_chat_template"]
    if max_seq_len:
        accuracy["max_seq_len"] = max_seq_len
    if max_batch_size:
        accuracy["max_batch_size"] = max_batch_size
    discriminators = _discriminators(task, built)
    if discriminators:
        accuracy["discriminators"] = discriminators
    for key in ("boot", "witness"):
        if accuracy_in.get(key):
            accuracy[key] = dict(accuracy_in[key])

    spec: dict[str, Any] = {
        "checkpoint_path": model["checkpoint"],
        "trtllm_repo_path": site["trtllm_repo_path"],
        "arch": task["device"]["arch"],
        **_parallel(task),
        "features": built,
        "scope": scope,
        "runtime": {**runtime, "extra_flags": extra_flags},
        "environment": dict(site["environment"]),
        "accuracy": accuracy,
    }
    if model.get("target"):
        spec["target"] = dict(model["target"])
    constraints = task.get("constraints") or {}
    if constraints.get("embargo"):
        spec["source_embargo"] = list(constraints["embargo"])
    commit_policy = dict(site.get("commit_policy") or {})
    if constraints.get("base_ref"):
        commit_policy["base_ref"] = constraints["base_ref"]
    if commit_policy:
        spec["commit_policy"] = commit_policy
    if constraints.get("budget"):
        warnings.append("constraints.budget: recorded; no run-wide budget enforces it yet")
    if site.get("agents"):
        spec["agents"] = dict(site["agents"])
    if task.get("notes"):
        spec["notes"] = str(task["notes"])
    options = _llm_api_options(task, built, spec["runtime"], max_batch_size)
    if GOAL_TUNE in goals:
        draft = options[built[-1]].get("speculative_config") if built else None
        spec["perf"] = _perf(task, site, warnings, draft)
    return Compiled(spec=spec, warnings=warnings, llm_api_options=options)


def member_options(member: str) -> dict[str, Any]:
    """The LLM-API parallelism one member (``tp4``, ``tep4``, ``dep4``) serves at."""
    match = PARALLEL_SEGMENT.match(str(member))
    if match is None:
        return {}
    kind, world = match.group(1), int(match.group(2))
    options: dict[str, Any] = {"tensor_parallel_size": world}
    if kind in ("tep", "dep"):
        options["moe_expert_parallel_size"] = world
    if kind == "dep":
        options["enable_attention_dp"] = True
    return options


def _perf(
    task: Mapping[str, Any],
    site: Mapping[str, Any],
    warnings: list[str],
    draft: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The tune stage's block: each config at its member's parallelism, with the speculator.

    The tune stage serves each config under the options it derives from the
    workload, overlaid with the config's ``llm_api_options``; the parallelism
    and the draft are stated there, or it measures TP1 without a draft.
    ``draft`` is the last speculative unit's ``speculative_config``. Without
    ``perf.configs``, one config per member of the run at the run's workload.
    """
    perf_in = task.get("perf") or {}
    site_perf = site.get("perf") or {}
    perf: dict[str, Any] = {
        "enabled": True,
        "reference": "stock" if perf_in.get("baseline", "stock") == "stock" else "none",
        **{k: v for k, v in site_perf.items()},
    }
    configs = []
    for entry in perf_in.get("configs") or []:
        config: dict[str, Any] = {"name": entry["name"], "workload": dict(entry["workload"])}
        if entry.get("constraint"):
            config["constraint"] = dict(entry["constraint"])
        options = member_options(entry["parallel"]) if entry.get("parallel") else {}
        for name, on in (entry.get("features") or {}).items():
            if name == "speculative":
                if on in (True, "on") and draft:
                    options["speculative_config"] = dict(draft)
            else:
                warnings.append(
                    f"perf config {entry['name']!r}: `features.{name}` is recorded; the tune "
                    f"stage serves {name} as the workload's derived options set it"
                )
        options.update(entry.get("llm_api_options") or {})
        if options:
            config["llm_api_options"] = options
        configs.append(config)
    if not configs:
        workload = task.get("workload") or {}
        shared = {
            "isl": workload.get("isl", 1024),
            "osl": workload.get("osl", 1024),
            "concurrency": workload.get("concurrency") or [1],
        }
        members = [f"{p['kind']}{p['world']}" for p in task.get("parallelism") or []] or ["tp1"]
        for member in members:
            options = member_options(member)
            if draft:
                options["speculative_config"] = dict(draft)
            configs.append({"name": member, "workload": dict(shared), "llm_api_options": options})
    perf["configs"] = configs
    goal = perf_in.get("goal")
    if isinstance(goal, Mapping):
        compiled = {}
        if goal.get("ratio_to_baseline"):
            compiled["ratio_to_reference"] = goal["ratio_to_baseline"]
        if goal.get("pct_of_sol"):
            compiled["pct_of_sol"] = goal["pct_of_sol"]
        if compiled:
            perf["goal"] = compiled
    return perf


def compile_files(task_path: Path, site_path: Path, out: Path) -> Compiled:
    """Compile two files into ``out`` and validate it with the pipeline's own schema.

    Each unit's LLM-API options are written under ``<out's directory>/llm_api/``
    and named in the spec's ``accuracy.llm_api_options``.
    """
    task = yaml.safe_load(Path(task_path).read_text(encoding="utf-8"))
    site = yaml.safe_load(Path(site_path).read_text(encoding="utf-8"))
    compiled = compile_task(task, site)
    out.parent.mkdir(parents=True, exist_ok=True)
    options_dir = out.parent.resolve() / LLM_API_DIRNAME
    options_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for unit, options in compiled.llm_api_options.items():
        path = options_dir / f"{unit}.yaml"
        path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
        paths[unit] = str(path)
    if paths:
        compiled.spec["accuracy"]["llm_api_options"] = paths
    out.write_text(yaml.safe_dump(compiled.spec, sort_keys=False), encoding="utf-8")
    load_and_validate_task_yaml(out)
    return compiled


__all__ = [
    "LLM_API_DIRNAME",
    "SPEC_DEC_ALGO",
    "Compiled",
    "compile_files",
    "compile_task",
    "member_options",
    "unit_name",
]
