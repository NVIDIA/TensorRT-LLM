# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast path: the run's spec derived in seconds from the checkpoint, ``site.yaml`` and a goal.

``--checkpoint``, ``--site`` and ``--goal`` write ``task.derived.yaml`` into the
workspace (the spec ``--task`` would take) beside ``derive.report.md``, which
says every value the goal did not state and where it came from. What cannot be
derived is an open question: the run stops before it starts and names each, and
the operator answers it in the spec. A feature the checkpoint carries (draft
layers, a vision or audio tower) is built unless the goal puts it out of scope;
anything else is built only when the goal names it. The goal is one line of
words, separated by spaces, commas or semicolons; :func:`parse_goal` defines
the accepted features, members, profiles and key=value settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..operators import tree_anchors
from .schema_keys import reject_unknown_keys
from .task_schema import (
    DEFAULT_PERF_CONCURRENCY,
    DISAGG_FEATURE,
    DISAGG_KEYS,
    FEATURE_FAMILIES,
    PARALLEL_PATTERN,
    PROFILE_CERTIFIED,
    PROFILE_SMOKE,
    PROFILE_STANDARD,
    PROFILES,
    SPECULATORS,
    parse_parallel_segment,
)

#: The derived spec and its report, in the workspace.
SPEC_FILENAME = "task.derived.yaml"
REPORT_FILENAME = "derive.report.md"
#: Where each unit's LLM-API options are written, beside the spec.
LLM_API_DIRNAME = "llm_api"

#: Members when neither the goal nor the site names any.
DEFAULT_MEMBERS: tuple[str, ...] = ("tp1",)
#: The workload when neither the goal nor the site states one. Stated rather
#: than left to the engine, which would size its cache for the checkpoint's
#: whole ``max_position_embeddings``.
DEFAULT_WORKLOAD: dict[str, int] = {"isl": 4096, "osl": 1024, "max_batch_size": 64}
#: Tokens between Mamba state snapshots on a hybrid checkpoint with block
#: reuse: without a snapshot policy the engine turns reuse off.
DEFAULT_SNAPSHOT_INTERVAL = 2048
#: The boot gate's prompts when the site declares none: one whose answer any
#: working target states, checked, and one open continuation.
DEFAULT_BOOT: dict[str, Any] = {
    "prompts": [
        {"prompt": "The capital of France is", "keywords_any": ["Paris"]},
        "In one sentence, photosynthesis is",
    ],
    "max_tokens": 32,
}
DEFAULT_BENCHMARK = "gsm8k"

#: Goal word -> the runtime switch ``-<word>`` turns off.
RUNTIME_WORDS: dict[str, str] = {
    "cuda_graph": "cuda_graph",
    "overlap_scheduler": "overlap_scheduler",
    "chunked_prefill": "chunked_prefill",
    "prefix_reuse": "kv_cache_reuse",
}
#: The units a goal may name: every feature of the spec.
UNIT_WORDS: frozenset[str] = frozenset(FEATURE_FAMILIES)
#: ``<feature>:<depth>`` -> the profile the feature's checks follow.
DEPTH_WORDS: dict[str, str] = {
    "smoke": PROFILE_SMOKE,
    "standard": PROFILE_STANDARD,
    "full": PROFILE_CERTIFIED,
}
#: Speculative algorithm -> its reference rows' ``spec_dec_algo`` and the LLM
#: API's ``speculative_config.decoding_type`` (the two spell them alike).
SPEC_DEC_ALGO = {"mtp": "MTP", "eagle3": "Eagle3", "dflash": "DFlash", "dspark": "DSpark"}
#: Where a config states its multi-token-prediction layers.
DRAFT_LAYER_KEYS: tuple[str, ...] = (
    "num_nextn_predict_layers",
    "num_mtp_modules",
    "mtp_num_hidden_layers",
)
WORKLOAD_KEYS = ("isl", "osl", "max_seq_len", "max_batch_size", "concurrency")
SITE_KEYS = (
    "trtllm_repo_path",
    "environment",
    "agents",
    "perf",
    "commit_policy",
    # The cluster's devices, and what a derived spec takes when its goal is silent.
    "device",
    "defaults",
)
SITE_DEVICE_KEYS = ("arch", "gpus_per_node", "nodes")
SITE_PERF_KEYS = ("spools", "dev_spools", "env_script")
SITE_DEFAULTS_KEYS = ("members", "workload", "accuracy", "boot", "disagg", "profile")
#: ``reference`` (how the anchor was measured) is the operator's record; nothing reads it.
SITE_ACCURACY_KEYS = ("benchmark", "protocol", "reference", "reference_yaml", "model_key")

#: ``key=value`` goal words -> (what they set, type).
_PARAM_WORDS: dict[str, tuple[str, type]] = {
    "draft_len": ("draft_len", int),
    "drafter": ("drafter", str),
    "kv": ("kv_dtype", str),
    "memory": ("kv_fraction", float),
    "chunk": ("chunk_tokens", int),
    "snapshot": ("snapshot_interval", int),
}
_ACCURACY_WORDS = {
    "benchmark": "benchmark",
    "model_key": "model_key",
    "reference": "reference_yaml",
}
_KV_DTYPES = ("fp8", "nvfp4")
#: KV dtypes a goal may name for the checkpoint's own KV format.
_KV_AUTO = frozenset({"auto", "bf16", "bfloat16"})
#: Weight formats a reference row states no ``quant_algo`` for.
_UNQUANTIZED = frozenset({"", "none", "auto", "bf16", "bfloat16", "fp16", "float16", "fp32"})
_REPO_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class DeriveError(ValueError):
    """A goal or a ``site.yaml`` the spec cannot be derived from, every problem listed."""


@dataclass
class Goal:
    """What a ``--goal`` line asks for."""

    text: str = ""
    #: Unit -> whether the goal builds it (True) or puts it out of scope (False).
    units: dict[str, bool] = field(default_factory=dict)
    #: Unit -> the members the goal narrows it to.
    unit_members: dict[str, list[str]] = field(default_factory=dict)
    #: Unit -> the profile the goal gives it (``multimodal:smoke``).
    unit_profiles: dict[str, str] = field(default_factory=dict)
    #: The goal asks for speculative decoding by the checkpoint's own drafter (True),
    #: or for none at all (False).
    speculative: bool | None = None
    #: Runtime switch -> off (``-cuda_graph``).
    switches_off: set[str] = field(default_factory=set)
    members: list[str] = field(default_factory=list)
    workload: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, str] = field(default_factory=dict)
    arch: str | None = None
    tune: bool = False


def parse_goal(text: str | None) -> Goal:
    """Read a goal line. Raises :class:`DeriveError` naming every unknown word."""
    goal = Goal(text=" ".join((text or "").split()))
    errors: list[str] = []
    for word in re.split(r"[\s,;]+", goal.text):
        if not word:
            continue
        if "=" in word:
            _key_value(goal, word, errors)
            continue
        lowered = word.lower()
        if lowered == "tune":
            goal.tune = True
            continue
        if PARALLEL_PATTERN.match(lowered):
            goal.members.append(lowered)
            continue
        if lowered == "text_only":
            goal.units["multimodal"] = False
            continue
        out = lowered.startswith(("-", "no-", "no_"))
        lowered, _, depth = lowered.partition(":")
        name, _, members = (
            lowered.lstrip("-").removeprefix("no-").removeprefix("no_").partition("@")
        )
        if depth and (out or name not in UNIT_WORDS or depth not in DEPTH_WORDS):
            errors.append(
                f"goal: {word!r}: a depth follows a feature and is one of {sorted(DEPTH_WORDS)}"
            )
            continue
        if depth:
            goal.unit_profiles[name] = DEPTH_WORDS[depth]
        if name in ("spec", "speculative"):
            goal.speculative = not out
        elif name in RUNTIME_WORDS and out:
            goal.switches_off.add(RUNTIME_WORDS[name])
        elif name in UNIT_WORDS:
            goal.units[name] = not out
            if members and not out:
                listed = [m for m in members.split("/") if m]
                if not listed or not all(PARALLEL_PATTERN.match(m) for m in listed):
                    errors.append(
                        f"goal: {word!r} names no members of the form tp<N>, tep<N>, dep<N>"
                    )
                else:
                    goal.unit_members[name] = listed
        else:
            errors.append(f"goal: {word!r} is not a feature, a member or key=value")
    if errors:
        raise DeriveError("\n".join(errors))
    return goal


def _key_value(goal: Goal, word: str, errors: list[str]) -> None:
    key, _, value = word.partition("=")
    key = key.lower()
    try:
        if key == "concurrency":
            goal.workload[key] = [int(v) for v in value.split("/") if v]
        elif key in WORKLOAD_KEYS:
            goal.workload[key] = int(value)
        elif key == "arch":
            goal.arch = value.lower()
        elif key in _ACCURACY_WORDS:
            goal.accuracy[_ACCURACY_WORDS[key]] = value
        elif key == "kv":
            dtypes = ["auto" if v in _KV_AUTO else v for v in value.lower().split("/") if v]
            wrong = [v for v in dtypes if v not in ("auto", *_KV_DTYPES)]
            if wrong or not dtypes:
                errors.append(f"goal: kv={value}: each of {['auto', 'bf16', *_KV_DTYPES]}")
            elif "kv_dtype" in goal.params:
                errors.append(
                    f"goal: {word!r}: kv= is named once, its dtypes in one word "
                    "(kv=bf16/fp8/nvfp4): the first is every gate's, each other a variant"
                )
            else:
                goal.params["kv_dtype"] = dtypes[0]
                goal.params["kv_variants"] = list(
                    dict.fromkeys(d for d in dtypes[1:] if d != dtypes[0])
                )
        elif key in _PARAM_WORDS:
            param, kind = _PARAM_WORDS[key]
            goal.params[param] = kind(value)
        else:
            errors.append(f"goal: {key!r} in {word!r} is not a key the goal takes")
    except ValueError:
        errors.append(f"goal: {word!r} needs a number")


def read_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """The checkpoint's ``config.json``, its ``text_config`` lifted to the top; None if unreadable."""
    try:
        data = json.loads((Path(str(checkpoint)) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # A multimodal checkpoint keeps the language model's shape under text_config.
    if isinstance(data.get("text_config"), dict):
        data = {**data["text_config"], **{k: v for k, v in data.items() if k != "text_config"}}
    return data


def draft_layers(config: Mapping[str, Any]) -> int:
    """How many draft (MTP) layers the checkpoint carries; 0 for none."""
    for key in DRAFT_LAYER_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


@dataclass(frozen=True)
class CheckpointFacts:
    """What a derivation reads from a checkpoint directory."""

    config: dict[str, Any]
    architecture: str
    #: The weight format and the KV-cache format the checkpoint was quantized to.
    quant_algo: str | None
    kv_cache_quant_algo: str | None
    #: A quantization the checkpoint declares without a readable ``quant_algo``.
    unread_quantization: str | None
    #: ``mtp`` or ``dspark`` for draft layers in the checkpoint, and their depth.
    speculator: str | None
    draft_len: int | None
    towers: tuple[str, ...]
    #: What marks the checkpoint hybrid (``spec/derive.hybrid_marker``), or None.
    hybrid: str | None
    #: The ``<org>/<model>`` name the config states, the reference rows' key.
    name: str | None


def inspect_checkpoint(path: Path) -> CheckpointFacts:
    """Read ``path``'s config files. Raises :class:`DeriveError` without a ``config.json``."""
    config = read_config(path)
    if config is None:
        raise DeriveError(f"{path}: no readable config.json; this is not a HuggingFace checkpoint")
    quant = config.get("quantization_config")
    quant = dict(quant) if isinstance(quant, Mapping) else {}
    try:
        hf_quant = json.loads((Path(path) / "hf_quant_config.json").read_text(encoding="utf-8"))
        quant = {**dict(hf_quant.get("quantization") or {}), **quant}
    except (OSError, ValueError, AttributeError):
        pass
    quant_algo = quant.get("quant_algo")
    method = str(quant.get("quant_method") or "").lower()
    if not quant_algo and method == "fp8":
        quant_algo = "FP8_BLOCK_SCALES" if quant.get("weight_block_size") else "FP8"
    speculator, draft_len = None, None
    if any(str(k).startswith("dspark") for k in config):
        speculator, draft_len = "dspark", config.get("dspark_block_size")
    elif draft_layers(config):
        speculator, draft_len = "mtp", draft_layers(config)
    name = config.get("_name_or_path")
    return CheckpointFacts(
        config=config,
        architecture=str((config.get("architectures") or ["?"])[0]),
        quant_algo=str(quant_algo) if quant_algo else None,
        kv_cache_quant_algo=str(quant.get("kv_cache_quant_algo") or "") or None,
        unread_quantization=(method or "a quantization_config without quant_algo")
        if quant and not quant_algo
        else None,
        speculator=speculator,
        draft_len=int(draft_len) if isinstance(draft_len, int) and draft_len > 0 else None,
        towers=tuple(k for k in ("vision_config", "audio_config") if k in config),
        hybrid=hybrid_marker(config),
        name=str(name) if isinstance(name, str) and _REPO_NAME.match(name) else None,
    )


def validate_site(site: Any) -> list[str]:
    """Every problem with the shape of a ``site.yaml``."""
    if not isinstance(site, Mapping):
        return ["site.yaml must be a mapping"]
    errors: list[str] = []
    reject_unknown_keys(site, SITE_KEYS, errors, where="site.yaml")
    for key in ("trtllm_repo_path", "environment"):
        if not site.get(key):
            errors.append(f"site.yaml: '{key}' is required")
    for key, keys in (("device", SITE_DEVICE_KEYS), ("perf", SITE_PERF_KEYS)):
        block = site.get(key)
        if block is not None and not isinstance(block, Mapping):
            errors.append(f"site.yaml: '{key}' must be a mapping")
        elif block:
            reject_unknown_keys(block, keys, errors, where=f"site.yaml {key}")
    defaults = site.get("defaults")
    if defaults is None:
        return errors
    if not isinstance(defaults, Mapping):
        return [*errors, "site.yaml: 'defaults' must be a mapping"]
    reject_unknown_keys(defaults, SITE_DEFAULTS_KEYS, errors, where="site.yaml defaults")
    members = defaults.get("members")
    if members is not None and (
        not isinstance(members, list)
        or not members
        or not all(isinstance(m, str) and PARALLEL_PATTERN.match(m) for m in members)
    ):
        errors.append(
            "site.yaml: 'defaults.members' must be a non-empty list of tp<N>, tep<N>, dep<N>"
        )
    for key, keys in (
        ("workload", WORKLOAD_KEYS),
        ("accuracy", SITE_ACCURACY_KEYS),
        ("boot", None),
        ("disagg", DISAGG_KEYS),
    ):
        block = defaults.get(key)
        if block is not None and not isinstance(block, Mapping):
            errors.append(f"site.yaml: 'defaults.{key}' must be a mapping")
        elif block and keys:
            reject_unknown_keys(block, keys, errors, where=f"site.yaml defaults.{key}")
    if defaults.get("profile") is not None and defaults["profile"] not in PROFILES:
        errors.append(f"site.yaml: 'defaults.profile' must be one of {list(PROFILES)}")
    return errors


@dataclass
class Derivation:
    """A derived spec, what it holds that the goal did not state, and what is still open."""

    spec: dict[str, Any]
    #: Unit -> the LLM-API options its gates run under, written beside the spec.
    llm_api_options: dict[str, dict[str, Any]]
    #: ``(field, value, source)`` for every value read from the checkpoint or the
    #: site, or defaulted.
    unstated: list[tuple[str, Any, str]] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    #: The checkpoint, site and goal it was derived from, and when.
    derived: dict[str, str] = field(default_factory=dict)


def derive_spec(
    checkpoint: Path, site: Mapping[str, Any], goal: Goal, *, site_path: Path | None = None
) -> Derivation:
    """The spec ``goal`` asks of ``checkpoint`` on ``site``; see the module docstring."""
    errors = validate_site(site)
    if errors:
        raise DeriveError("\n".join(errors))
    checkpoint = Path(checkpoint).resolve()
    facts = inspect_checkpoint(checkpoint)
    defaults = dict(site.get("defaults") or {})
    unstated: list[tuple[str, Any, str]] = []
    questions: list[str] = []

    def note(where: str, value: Any, source: str) -> Any:
        unstated.append((where, value, source))
        return value

    arch = goal.arch or (site.get("device") or {}).get("arch")
    if arch and not goal.arch:
        note("arch", arch, "site")
    if not arch:
        questions.append(
            "arch: name the GPU architecture (arch=sm_100 in --goal, or `device: {arch: ...}` "
            "in site.yaml); receipts are keyed by it and it is never guessed"
        )
    if facts.unread_quantization:
        questions.append(
            f"the checkpoint is quantized ({facts.unread_quantization}) and states no "
            "quant_algo: set accuracy.discriminators to the reference row's (quant_algo, "
            "kv_cache_quant_algo), so the anchor is not the unquantized model's"
        )

    # The members.
    if goal.members:
        members = list(dict.fromkeys(goal.members))
    elif defaults.get("members"):
        members = note("members", list(dict.fromkeys(defaults["members"])), "site")
    else:
        members = note("members", list(DEFAULT_MEMBERS), "default")

    # The workload.
    workload = dict(defaults.get("workload") or {})
    source = "site" if workload else "default"
    if not workload:
        workload = dict(DEFAULT_WORKLOAD)
        reach = facts.config.get("max_position_embeddings")
        if isinstance(reach, int) and workload["isl"] + workload["osl"] > reach:
            # A default must fit the model; a stated workload that does not is a block.
            workload["osl"] = min(workload["osl"], reach // 4)
            workload["isl"] = reach - workload["osl"]
    for key, value in workload.items():
        if key not in goal.workload:
            note(f"workload.{key}", value, source)
    workload.update(goal.workload)
    if goal.tune and not workload.get("concurrency"):
        workload["concurrency"] = note(
            "workload.concurrency", list(DEFAULT_PERF_CONCURRENCY), "default"
        )
    concurrency = workload.get("concurrency") or []
    max_seq_len = workload.get("max_seq_len") or (
        workload["isl"] + workload["osl"] if "isl" in workload and "osl" in workload else None
    )
    max_batch_size = workload.get("max_batch_size") or (max(concurrency) if concurrency else None)
    if concurrency and "max_batch_size" not in goal.workload:
        # A default batch below the goal's concurrency would queue the perf points it names.
        max_batch_size = max(int(max_batch_size or 0), max(concurrency))

    # The units: the checkpoint's own drafter and towers unless the goal says otherwise. A
    # goal that names its features asks for those: what else the checkpoint carries is left out.
    built: list[str] = []
    listed = any(goal.units.values()) or goal.speculative is True
    speculator = next((u for u, on in goal.units.items() if on and u in SPECULATORS), None)
    if (
        speculator is None
        and goal.speculative is not False
        and not any(goal.units.get(u) is False for u in SPECULATORS)
    ):
        if facts.speculator and listed and not goal.speculative:
            note("features (left out)", facts.speculator, "checkpoint carries it, goal names none")
        elif facts.speculator:
            speculator = note("features (speculator)", facts.speculator, "checkpoint")
        elif goal.speculative:
            questions.append(
                "speculative: the goal asks for speculative decoding and the checkpoint carries "
                "no draft layers; name the algorithm (mtp, eagle3, dflash or dspark)"
            )
    if speculator:
        built.append(speculator)
        if "draft_len" not in goal.params and facts.draft_len and speculator == facts.speculator:
            goal.params["draft_len"] = note("draft_len", facts.draft_len, "checkpoint")
    multimodal = goal.units.get("multimodal")
    if multimodal is None and facts.towers and listed:
        multimodal = False
        note("features (left out)", "multimodal", "checkpoint carries it, goal names none")
    elif multimodal is None and facts.towers:
        multimodal = note("features (multimodal)", True, "checkpoint")
    if multimodal:
        built.append("multimodal")
    built += [
        u for u, on in goal.units.items() if on and u not in SPECULATORS and u != "multimodal"
    ]

    # The runtime: every switch on unless the goal turns it off.
    runtime: dict[str, Any] = {
        name: name not in goal.switches_off for name in RUNTIME_WORDS.values()
    }
    reuse = runtime["kv_cache_reuse"]
    snapshot = goal.params.get("snapshot_interval")
    if facts.hybrid and reuse and not snapshot:
        snapshot = note("snapshot_interval", DEFAULT_SNAPSHOT_INTERVAL, "default")
    kv_dtype = str(goal.params.get("kv_dtype") or "").lower()
    native_kv = str(facts.kv_cache_quant_algo or "").lower()
    if kv_dtype in ("", "auto") and native_kv in _KV_DTYPES:
        kv_dtype = note("kv_cache dtype", native_kv, "checkpoint")
    flags = []
    if max_seq_len:
        flags.append(f"--max_seq_len={max_seq_len}")
    if max_batch_size:
        flags.append(f"--max_batch_size={max_batch_size}")
    if kv_dtype in _KV_DTYPES:
        flags.append(f"--kv_cache_dtype={kv_dtype}")
    if goal.params.get("chunk_tokens"):
        flags.append(f"--max_num_tokens={goal.params['chunk_tokens']}")
    if goal.params.get("kv_fraction"):
        flags.append(f"--kv_cache_fraction={goal.params['kv_fraction']}")
    runtime["extra_flags"] = flags

    # The accuracy anchor.
    site_accuracy = dict(defaults.get("accuracy") or {})
    for key, value in site_accuracy.items():
        if key not in goal.accuracy:
            note(f"accuracy.{key}", value, "site")
    anchor = {**site_accuracy, **goal.accuracy}
    benchmark = anchor.get("benchmark") or note("accuracy.benchmark", DEFAULT_BENCHMARK, "default")
    accuracy: dict[str, Any] = {
        "benchmark": benchmark,
        "reference_yaml": anchor.get("reference_yaml")
        or f"{_tree_path(site, 'accuracy_references')}/{benchmark}.yaml",
    }
    model_key = anchor.get("model_key") or (
        note("accuracy.model_key", facts.name, "checkpoint") if facts.name else None
    )
    if model_key:
        accuracy["model_key"] = model_key
    else:
        questions.append(
            "accuracy.model_key: name the model's HuggingFace id (`org/model`, model_key= in "
            "--goal): the reference rows are keyed by it, and config.json names none (the "
            "fallback, the checkpoint path's last two components, is no model's name)"
        )
    if (anchor.get("protocol") or {}).get("chat_template"):
        accuracy["benchmark_args"] = ["--apply_chat_template"]
    if max_seq_len:
        accuracy["max_seq_len"] = max_seq_len
    if max_batch_size:
        accuracy["max_batch_size"] = max_batch_size
    discriminators = _discriminators(facts, kv_dtype, built)
    if discriminators:
        accuracy["discriminators"] = discriminators
    if defaults.get("boot"):
        accuracy["boot"] = note("accuracy.boot", dict(defaults["boot"]), "site")
    else:
        accuracy["boot"] = note("accuracy.boot", json.loads(json.dumps(DEFAULT_BOOT)), "default")
    if DISAGG_FEATURE in built:
        accuracy["disagg"] = _disagg(defaults, facts, site, note, questions)

    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": site["trtllm_repo_path"],
        **({"arch": arch} if arch else {}),
        **_parallel(members, goal.unit_members, built),
        "features": built,
        "scope": "text_only" if multimodal is False else "full",
        "runtime": runtime,
        "environment": dict(site["environment"]),
        "accuracy": accuracy,
    }
    kv_variants = {
        f"kv_{dtype}": {
            "unit": "core",
            "gates": ["accuracy"],
            "llm_api_options": {"kv_cache_config": {"dtype": dtype}},
            "note": f"KV cache stored as {dtype}, judged against the unit's own score",
        }
        for dtype in goal.params.get("kv_variants") or ()
    }
    if kv_variants:
        spec["variants"] = kv_variants
    if site.get("commit_policy"):
        spec["commit_policy"] = dict(site["commit_policy"])
    if site.get("agents"):
        spec["agents"] = dict(site["agents"])
    if defaults.get("profile"):
        spec["profile"] = note("profile", str(defaults["profile"]), "site")
    chosen = {u: p for u, p in goal.unit_profiles.items() if u in built}
    if chosen:
        spec["profile"] = {"default": spec.get("profile") or PROFILE_STANDARD, **chosen}
    layout = spec["members"] if isinstance(spec.get("members"), dict) else {}
    everywhere = list(dict.fromkeys(m for listed in layout.values() for m in listed))
    left_out = [
        f"{unit}@{member}"
        for unit in ["core", *built]
        for member in everywhere
        if member not in layout.get(unit, layout.get("default", ()))
    ]
    if left_out:
        # Where the goal gates a unit on fewer members, the rest is its choice, not a gap.
        spec["coverage"] = {"accept": left_out}
    options = _llm_api_options(runtime, built, kv_dtype, goal.params, snapshot, max_batch_size)
    if goal.tune:
        draft = options[built[-1]].get("speculative_config") if built else None
        spec["perf"] = _perf(site, workload, members, draft)
        adp = [c["name"] for c in spec["perf"]["configs"] if c["name"].startswith("dep")]
        if adp:
            note(
                f"perf ({', '.join(adp)})",
                "the high concurrency point x attention-DP ranks",
                "default",
            )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    derived = {
        "checkpoint": str(checkpoint),
        "architecture": facts.architecture,
        "site": str(Path(site_path).resolve()) if site_path else "",
        "goal": goal.text,
        "at": stamp,
    }
    return Derivation(spec, options, unstated, questions, derived)


def _parallel(
    members: list[str], narrowed: Mapping[str, list[str]], built: list[str]
) -> dict[str, Any]:
    """The parallel segment and the member set, per unit where the goal narrows one."""
    if len(members) == 1:
        return {"parallel": members[0]}
    per_unit = {unit: list(narrowed[unit]) for unit in built if unit in narrowed}
    if not per_unit:
        return {"parallel": "unified", "members": list(members)}
    return {"parallel": "unified", "members": {"default": list(members), **per_unit}}


def _discriminators(
    facts: CheckpointFacts, kv_dtype: str, built: list[str]
) -> dict[str, Any] | None:
    """The reference row each unit anchors to: the weight and KV formats, and the speculator.

    A reference row matches only its exact discriminator set, so a run that
    names none anchors a quantized checkpoint to the bf16 row. Every unit from
    the speculative one on runs with its draft, so they anchor to its row too.
    """
    base: dict[str, Any] = {}
    weights = str(facts.quant_algo or "")
    if weights.strip().lower() not in _UNQUANTIZED:
        base["quant_algo"] = weights.strip().upper()
    if kv_dtype in _KV_DTYPES:
        base["kv_cache_quant_algo"] = kv_dtype.upper()
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
    runtime: Mapping[str, Any],
    built: list[str],
    kv_dtype: str,
    params: Mapping[str, Any],
    snapshot: int | None,
    max_batch_size: Any,
) -> dict[str, dict[str, Any]]:
    """Each unit's LLM-API options: the runtime switches, the KV cache, and its speculator.

    Every check receives these public constructor options, with runtime switches
    resolved on the controller. State the capture list here once. A unit carries
    the speculator of every feature up to it.
    """
    kv: dict[str, Any] = {}
    if kv_dtype in _KV_DTYPES:
        kv["dtype"] = kv_dtype
    if params.get("kv_fraction"):
        kv["free_gpu_memory_fraction"] = float(params["kv_fraction"])
    kv["enable_block_reuse"] = bool(runtime["kv_cache_reuse"])
    if snapshot and runtime["kv_cache_reuse"]:
        kv["mamba_state_config"] = {"periodic_snapshot_interval": int(snapshot)}
    base: dict[str, Any] = {
        "kv_cache_config": kv,
        "enable_chunked_prefill": bool(runtime["chunked_prefill"]),
    }
    if params.get("chunk_tokens"):
        base["max_num_tokens"] = int(params["chunk_tokens"])
    if not runtime["overlap_scheduler"]:
        base["disable_overlap_scheduler"] = True
    if runtime["cuda_graph"]:
        graphs: dict[str, Any] = {"enable_padding": True}
        if max_batch_size:
            graphs["max_batch_size"] = int(max_batch_size)
        base["cuda_graph_config"] = graphs
    else:
        base["cuda_graph_config"] = None
    options = {"core": dict(base)}
    draft: dict[str, Any] | None = None
    for unit in built:
        if unit in SPEC_DEC_ALGO:
            draft = {
                "decoding_type": SPEC_DEC_ALGO[unit],
                "max_draft_len": int(params.get("draft_len") or 1),
            }
            if params.get("drafter"):
                draft["speculative_model"] = str(params["drafter"])
        options[unit] = {**base, **({"speculative_config": dict(draft)} if draft else {})}
    return options


def _tree_path(site: Mapping[str, Any], key: str) -> str:
    """Where anchor ``key`` is in the site's checkout (``operators/tree_anchors.py``), else its hint."""
    found = tree_anchors.resolve(site.get("trtllm_repo_path"), key)
    return found.paths[0] if found.found else tree_anchors.ANCHORS[key].hint


def _disagg(
    defaults: Mapping[str, Any],
    facts: CheckpointFacts,
    site: Mapping[str, Any],
    note,
    questions: list[str],
) -> dict[str, Any]:
    """The disaggregated-serving test the run writes: the site's settings, and its node id.

    The node id defaults to a class named for the architecture, in the file whose
    tests share the tree's disaggregated launcher (``launch_disaggregated_llm``).
    """
    block = dict(defaults.get("disagg") or {})
    if block:
        note("accuracy.disagg", dict(block), "site")
    if block.get("test"):
        return block
    launcher = tree_anchors.resolve(site.get("trtllm_repo_path"), "disagg_launcher")
    if launcher.found:
        name = re.sub(r"\W|For(?:CausalLM|ConditionalGeneration)$", "", facts.architecture)
        name = name or "Model"
        block["test"] = note(
            "accuracy.disagg.test",
            f"{launcher.paths[0]}::Test{name}::test_auto_dtype",
            "default",
        )
    else:
        questions.append(
            "accuracy.disagg.test: name the node id of the disaggregated-serving accuracy test "
            "the run will write, beside the tree's tests that serve through "
            "`trtllm-serve disaggregated`"
        )
    return block


def member_options(member: str) -> dict[str, Any]:
    """The LLM-API parallelism one member (``tp4``, ``tep4``, ``dep4``) serves at."""
    match = PARALLEL_PATTERN.match(str(member))
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
    site: Mapping[str, Any],
    workload: Mapping[str, Any],
    members: list[str],
    draft: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The tune stage's block: one config per member at the run's workload, with the speculator."""
    shared = {
        "isl": workload.get("isl", 1024),
        "osl": workload.get("osl", 1024),
        "concurrency": workload.get("concurrency") or list(DEFAULT_PERF_CONCURRENCY),
    }
    configs = []
    for member in members:
        options = member_options(member)
        if draft:
            options["speculative_config"] = dict(draft)
        load = dict(shared)
        kind, ranks = parse_parallel_segment(member)
        if kind == "dep":
            # Attention DP serves throughput: only its high point, each rank at that load.
            load["concurrency"] = [max(shared["concurrency"]) * (ranks or 1)]
        configs.append({"name": member, "workload": load, "llm_api_options": options})
    return {
        "enabled": True,
        **dict(site.get("perf") or {}),
        "configs": configs,
    }


def render_report(derivation: Derivation, spec_path: Path) -> str:
    """``derive.report.md``: what was read, what was defaulted, and what is still open."""
    derived = derivation.derived
    lines = [
        "# The derived spec",
        "",
        f"Derived from `{derived.get('checkpoint')}` ({derived.get('architecture')}), the site "
        f"`{derived.get('site') or '(unnamed)'}` and the goal `{derived.get('goal') or '(none)'}` "
        f"at {derived.get('at')}, into `{spec_path}`. Edit it and run the same command to change "
        "it; `--clean` derives it again.",
        "",
        "## What the goal did not state",
        "",
        "| field | from | value |",
        "|---|---|---|",
    ]
    for where, value, source in derivation.unstated:
        shown = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        lines.append(f"| `{where}` | {source} | `{shown}` |")
    lines += ["", f"## Open questions ({len(derivation.questions)})", ""]
    lines += [f"- {q}" for q in derivation.questions] or ["None."]
    return "\n".join(lines) + "\n"


def write_derivation(workspace: Path, derivation: Derivation) -> Path:
    """Write the spec, each unit's LLM-API options and the report into ``workspace``; the spec's path."""
    workspace.mkdir(parents=True, exist_ok=True)
    options_dir = workspace.resolve() / LLM_API_DIRNAME
    options_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for unit, options in derivation.llm_api_options.items():
        path = options_dir / f"{unit}.yaml"
        path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
        paths[unit] = str(path)
    spec = dict(derivation.spec)
    spec["accuracy"] = {**spec["accuracy"], "llm_api_options": paths}
    path = workspace / SPEC_FILENAME
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (workspace / REPORT_FILENAME).write_text(render_report(derivation, path), encoding="utf-8")
    return path


#: What ``is_hybrid_linear`` (``tensorrt_llm/_torch/pyexecutor/config_utils.py``)
#: names by ``model_type``, and by ``architectures[0]``.
_HYBRID_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
_HYBRID_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)
#: Words in a per-layer type that name a layer keeping recurrent state:
#: ``mamba``, ``linear_attention``, a gated delta rule (``*_delta*``).
_RECURRENT_LAYER_WORDS = ("mamba", "linear", "ssm", "delta", "recurrent")
_LAYER_TYPE_KEYS = ("layer_types", "layers_block_type")


def hybrid_marker(config: Mapping[str, Any]) -> str | None:
    """What marks a checkpoint's ``config.json`` as a hybrid linear-attention model; ``None``.

    Mirrors the engine's ``is_hybrid_linear`` without importing TensorRT-LLM;
    a model new to the engine is caught by its layer list instead (a
    ``layer_types`` or ``layers_block_type`` entry naming a recurrent layer).
    Read on the config and on its ``text_config``.
    """
    for scope, block in (("", config), ("text_config.", config.get("text_config"))):
        if not isinstance(block, Mapping):
            continue
        if block.get("hybrid_override_pattern"):
            return f"`{scope}hybrid_override_pattern`"
        model_type = block.get("model_type")
        if model_type in _HYBRID_MODEL_TYPES or (
            model_type == "kimi_linear" and block.get("linear_attn_config") is not None
        ):
            return f"`{scope}model_type: {model_type}`"
        architectures = block.get("architectures") or ()
        if architectures and architectures[0] in _HYBRID_ARCHITECTURES:
            return f"`{scope}architectures: {architectures[0]}`"
        for key in _LAYER_TYPE_KEYS:
            entries = block.get(key)
            if not isinstance(entries, (list, tuple)):
                continue
            recurrent = sorted(
                {
                    str(e)
                    for e in entries
                    if any(w in str(e).lower() for w in _RECURRENT_LAYER_WORDS)
                }
            )
            if recurrent:
                return f"`{scope}{key}` with {', '.join(recurrent)}"
    return None


__all__ = [
    "hybrid_marker",
    "DEFAULT_BOOT",
    "DEFAULT_MEMBERS",
    "DEFAULT_SNAPSHOT_INTERVAL",
    "DEFAULT_WORKLOAD",
    "DRAFT_LAYER_KEYS",
    "LLM_API_DIRNAME",
    "REPORT_FILENAME",
    "SPEC_DEC_ALGO",
    "SPEC_FILENAME",
    "CheckpointFacts",
    "DeriveError",
    "Derivation",
    "Goal",
    "derive_spec",
    "draft_layers",
    "inspect_checkpoint",
    "member_options",
    "parse_goal",
    "read_config",
    "render_report",
    "validate_site",
    "write_derivation",
]
