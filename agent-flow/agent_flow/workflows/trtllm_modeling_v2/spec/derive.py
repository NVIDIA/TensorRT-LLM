# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast path: the run's spec derived in seconds from the checkpoint, ``site.yaml`` and a goal.

``--checkpoint``, ``--site`` and ``--goal`` write ``task.derived.yaml`` into the
workspace (the spec ``--task`` would take) beside ``derive.report.md``, which
says every value the goal did not state and where it came from. What cannot be
derived is an open question: the run stops before it starts and names each, and
the operator answers it in the spec. A feature the checkpoint carries (draft
layers, a vision or audio tower) is built unless the goal puts it out of scope;
anything else is built only when the goal names it.

The core configuration is what every unit is gated under: CUDA graphs and the
overlap scheduler on, the KV cache in the checkpoint's own format, block reuse
and chunked prefill off. A non-core runtime feature the goal names
(``chunked_prefill``, ``prefix_reuse``) and each extra KV dtype (``kv=``) is one
deployment of its own beside core, judged against core's own score where no
reference row fits: each non-core feature on its own, never a cross product. The
spec states its deployments natively: each one's options are written to
``engines/<name>.yaml`` beside it (``spec/shorthand.py`` does the arithmetic). The goal is one line of
words, separated by spaces, commas or semicolons; :func:`parse_goal` defines
the accepted features, members and key=value settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import deployments as deployments_mod
from . import shorthand
from .shorthand import DISAGG, DISAGG_KEYS, DRAFTERS, FEATURES, MEMBER
from .task_schema import ACCURACY_FIELD, reject_unknown_keys

#: The derived spec and its report, in the workspace.
SPEC_FILENAME = "task.derived.yaml"
REPORT_FILENAME = "derive.report.md"
#: Where each deployment's native options are written, beside the spec.
ENGINES_DIRNAME = "engines"
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

#: Goal word -> its runtime switch: ``-<word>`` turns a core one off; naming a non-core
#: one (:data:`NON_CORE_SWITCHES`) adds its configuration variant.
RUNTIME_WORDS: dict[str, str] = {
    "cuda_graph": "cuda_graph",
    "overlap_scheduler": "overlap_scheduler",
    "chunked_prefill": "chunked_prefill",
    "prefix_reuse": "kv_cache_reuse",
}
#: The runtime switches of the core configuration: on unless the goal turns them off.
CORE_SWITCHES: frozenset[str] = frozenset({"cuda_graph", "overlap_scheduler"})
#: Non-core runtime switch -> the configuration variant that turns it on: off in core.
NON_CORE_SWITCHES: dict[str, str] = {"chunked_prefill": "chunked", "kv_cache_reuse": "reuse"}
#: Tokens per prefill chunk in the ``chunked`` variant when the goal names none: below the
#: benchmark's few-shot prompts, so their prefill is split.
DEFAULT_CHUNK_TOKENS = 512
#: The units a goal may name: every feature of the spec.
UNIT_WORDS: frozenset[str] = frozenset(FEATURES)
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
    "commit_policy",
    # The cluster's devices, and what a derived spec takes when its goal is silent.
    "device",
    "defaults",
)
SITE_DEVICE_KEYS = ("arch", "gpus_per_node", "nodes")
SITE_DEFAULTS_KEYS = ("members", "workload", "accuracy", "boot", "disagg")
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
    #: The member the goal serves disagg at (``disagg@tep4``); the first, unsaid.
    disagg_member: str | None = None
    #: The goal asks for speculative decoding by the checkpoint's own drafter (True),
    #: or for none at all (False).
    speculative: bool | None = None
    #: Runtime switch -> off (``-cuda_graph``).
    switches_off: set[str] = field(default_factory=set)
    #: Non-core runtime switches the goal names (``chunked_prefill``): one variant each.
    switches_on: set[str] = field(default_factory=set)
    members: list[str] = field(default_factory=list)
    workload: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, str] = field(default_factory=dict)
    arch: str | None = None


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
        if MEMBER.match(lowered):
            goal.members.append(lowered)
            continue
        if lowered == "text_only":
            goal.units["multimodal"] = False
            continue
        out = lowered.startswith(("-", "no-", "no_"))
        name, _, member = lowered.lstrip("-").removeprefix("no-").removeprefix("no_").partition("@")
        if name in ("spec", "speculative"):
            goal.speculative = not out
        elif name in RUNTIME_WORDS and out:
            goal.switches_off.add(RUNTIME_WORDS[name])
        elif name in RUNTIME_WORDS:
            # A core switch is on already; a non-core one is tested as its own variant.
            if RUNTIME_WORDS[name] in NON_CORE_SWITCHES:
                goal.switches_on.add(RUNTIME_WORDS[name])
        elif name in UNIT_WORDS:
            goal.units[name] = not out
            if member and (out or name != DISAGG or not MEMBER.match(member)):
                errors.append(
                    f"goal: {word!r}: only disagg takes a member of its own (disagg@tep4); "
                    f"every other feature is served at the first member"
                )
            elif member:
                goal.disagg_member = member
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
    for key, keys in (("device", SITE_DEVICE_KEYS),):
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
        or not all(isinstance(m, str) and MEMBER.match(m) for m in members)
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
        if key not in goal.workload and key != "max_batch_size":
            note(f"workload.{key}", value, source)
    workload.update(goal.workload)
    concurrency = workload.get("concurrency") or []
    max_seq_len = workload.get("max_seq_len") or (
        workload["isl"] + workload["osl"] if "isl" in workload and "osl" in workload else None
    )
    max_batch_size = workload.get("max_batch_size") or (max(concurrency) if concurrency else None)
    if concurrency and "max_batch_size" not in goal.workload:
        # A default batch below the goal's concurrency would queue the requests it names.
        max_batch_size = max(int(max_batch_size or 0), max(concurrency))
    if max_batch_size and "max_batch_size" not in goal.workload:
        # The value the spec carries, not the default it was raised from.
        note(
            "workload.max_batch_size",
            max_batch_size,
            source if max_batch_size == workload.get("max_batch_size") else "concurrency",
        )

    # The units: the checkpoint's own drafter and towers unless the goal says otherwise. A
    # goal that names its features asks for those: what else the checkpoint carries is left out.
    built: list[str] = []
    listed = any(goal.units.values()) or goal.speculative is True
    speculator = next((u for u, on in goal.units.items() if on and u in DRAFTERS), None)
    if (
        speculator is None
        and goal.speculative is not False
        and not any(goal.units.get(u) is False for u in DRAFTERS)
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
    built += [u for u, on in goal.units.items() if on and u not in DRAFTERS and u != "multimodal"]

    # The runtime: the core switches on unless the goal turns them off, the rest off.
    runtime: dict[str, Any] = {
        name: name in CORE_SWITCHES and name not in goal.switches_off
        for name in RUNTIME_WORDS.values()
    }
    for switch in sorted(set(NON_CORE_SWITCHES) - goal.switches_on - goal.switches_off):
        note(f"runtime.{switch}", False, "core configuration: a non-core feature")
    reuse = "kv_cache_reuse" in goal.switches_on
    chunked = "chunked_prefill" in goal.switches_on
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
    if goal.params.get("chunk_tokens") and not chunked:
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
    # Unsaid, the reference rows are the tree's own file for the benchmark.
    accuracy: dict[str, Any] = {"benchmark": benchmark}
    if anchor.get("reference_yaml"):
        accuracy["reference_yaml"] = anchor["reference_yaml"]
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
    if DISAGG in built and defaults.get("disagg"):
        accuracy[DISAGG] = note("accuracy.disagg", dict(defaults["disagg"]), "site")
    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": site["trtllm_repo_path"],
        **({"arch": arch} if arch else {}),
        "members": {"default": members, DISAGG: [goal.disagg_member]}
        if DISAGG in built and goal.disagg_member
        else members,
        "features": built,
        "scope": "text_only" if multimodal is False else "full",
        "runtime": runtime,
        "environment": dict(site["environment"]),
        "accuracy": accuracy,
    }
    kv_variants = {
        f"kv_{dtype}": {
            "unit": "core",
            "llm_api_options": {"kv_cache_config": {"dtype": dtype}},
            "note": f"KV cache stored as {dtype}, judged against the unit's own score",
        }
        for dtype in goal.params.get("kv_variants") or ()
    }
    for switch in sorted(goal.switches_on):
        name = NON_CORE_SWITCHES[switch]
        if switch == "chunked_prefill":
            tokens = int(goal.params.get("chunk_tokens") or DEFAULT_CHUNK_TOKENS)
            overlay: dict[str, Any] = {"enable_chunked_prefill": True, "max_num_tokens": tokens}
            said = f"chunked prefill on, in chunks of {tokens} tokens"
        else:
            cache: dict[str, Any] = {"enable_block_reuse": True}
            if snapshot:
                cache["mamba_state_config"] = {"periodic_snapshot_interval": int(snapshot)}
            overlay = {"kv_cache_config": cache}
            said = "KV-cache block reuse on"
        kv_variants[name] = {
            "unit": "core",
            "runtime": {switch: True},
            "llm_api_options": overlay,
            "note": f"{said}, judged against the unit's own score",
        }
    if kv_variants:
        spec["variants"] = kv_variants
    if site.get("commit_policy"):
        spec["commit_policy"] = dict(site["commit_policy"])
    if site.get("agents"):
        spec["agents"] = dict(site["agents"])
    # The chunk size is the chunked variant's, not core's.
    params = {k: v for k, v in goal.params.items() if not (chunked and k == "chunk_tokens")}
    options = _llm_api_options(runtime, built, kv_dtype, params, snapshot, max_batch_size)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    derived = {
        "checkpoint": str(checkpoint),
        "architecture": facts.architecture,
        "site": str(Path(site_path).resolve()) if site_path else "",
        "goal": goal.text,
        "at": stamp,
    }
    return Derivation(spec, options, unstated, questions, derived)


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
        algo = DRAFTERS.get(unit, algo)
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
        if unit in DRAFTERS:
            draft = {
                "decoding_type": DRAFTERS[unit],
                "max_draft_len": int(params.get("draft_len") or 1),
            }
            if params.get("drafter"):
                draft["speculative_model"] = str(params["drafter"])
        options[unit] = {**base, **({"speculative_config": dict(draft)} if draft else {})}
    return options


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
    """Write the spec, each deployment's options and the report into ``workspace``; the spec's path.

    The derivation is expanded here (``spec/shorthand.py``) into the deployments the spec
    states: the options go to ``engines/<name>.yaml`` and the spec names them by path.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    asked = dict(derivation.spec)
    asked[ACCURACY_FIELD] = {**asked[ACCURACY_FIELD], "llm_api_options": derivation.llm_api_options}
    spec = {k: v for k, v in asked.items() if k not in (*shorthand.SHORTHAND_KEYS, "parallel")}
    kept = {k: v for k, v in asked[ACCURACY_FIELD].items() if k not in shorthand.ACCURACY_KEYS}
    # The spec's own row tags are core's; a deployment that adds one states it on its benchmark.
    tags = shorthand.per(kept, "discriminators", "core")
    kept.pop("discriminators", None)
    if tags:
        kept["discriminators"] = dict(tags)
    spec[ACCURACY_FIELD] = kept
    default = [{"name": kept.get("benchmark") or deployments_mod.DEFAULT_BENCHMARK}]
    stated = []
    for deployment in shorthand.expand(asked):
        paths = deployments_mod.write_options(deployment, workspace.resolve() / ENGINES_DIRNAME)
        entry = deployment.as_dict()
        for role, path in paths.items():
            holder = entry if role == "server" else entry[role]
            holder["options"] = f"{ENGINES_DIRNAME}/{path.name}"
        # A benchmark states what differs from the spec's `accuracy` block, and no more.
        for benchmark in entry.get("benchmarks") or ():
            for key in [k for k, v in benchmark.items() if k != "name" and kept.get(k) == v]:
                del benchmark[key]
        if entry.get("benchmarks") == default:
            del entry["benchmarks"]
        stated.append(entry)
    spec[deployments_mod.DEPLOYMENTS_FIELD] = stated
    path = workspace / SPEC_FILENAME
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (workspace / REPORT_FILENAME).write_text(render_report(derivation, path), encoding="utf-8")
    return path


#: Checkpoint structure that explicitly describes recurrent attention/state. Keep model names
#: out of this hint: an unknown architecture can use the same checkpoint schema.
_RECURRENT_CONFIG_KEYS = (
    "linear_attn_config",
    "linear_attention_config",
    "linear_num_value_heads",
    "mamba_ssm_cfg",
    "ssm_cfg",
    "full_attention_interval",
)
#: Words in a per-layer type that name a layer keeping recurrent state:
#: ``mamba``, ``linear_attention``, a gated delta rule (``*_delta*``).
_RECURRENT_LAYER_WORDS = ("mamba", "linear", "ssm", "delta", "recurrent")
_LAYER_TYPE_KEYS = ("layer_types", "layers_block_type")


def hybrid_marker(config: Mapping[str, Any]) -> str | None:
    """What marks a checkpoint's ``config.json`` as a hybrid linear-attention model; ``None``.

    This is a structural intake hint, not the engine's architecture registry. It reads
    explicit recurrent configuration or layer descriptions on the config and its
    ``text_config``; runtime evidence decides whether recurrent state is actually used.
    """
    for scope, block in (("", config), ("text_config.", config.get("text_config"))):
        if not isinstance(block, Mapping):
            continue
        if block.get("hybrid_override_pattern"):
            return f"`{scope}hybrid_override_pattern`"
        for key in _RECURRENT_CONFIG_KEYS:
            if block.get(key):
                return f"`{scope}{key}`"
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


USAGE = """usage: trtllm-modeling-v2 intake derive --checkpoint DIR (--site FILE | --repo DIR) --out DIR [--goal "..."]

Derive a spec from the checkpoint, the goal and an optional site.yaml; write it, each
deployment's options (engines/<name>.yaml) and derive.report.md into --out, print what it
is likely to cost or miss, and stop: no run, no GPU. spec/INTAKE.md in the workflow's
package is the procedure. `trtllm-modeling-v2 --task SPEC --workspace W --dry-run`
validates a spec.
"""


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 intake derive ...``: 0 written, 2 a goal or site it cannot derive from."""
    import argparse
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    if argv[0] == "lint":
        print(
            "`intake lint` is replaced by `trtllm-modeling-v2 --task <spec> --workspace <W> "
            "--dry-run`, which validates the spec and journals its warnings; `intake derive` "
            "prints the same warnings",
            file=sys.stderr,
        )
        return 2
    if argv[0] != "derive":
        print(USAGE, file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 intake derive",
        description="Derive a spec and stop: no run, no GPU. The derived spec is a starting "
        "point to edit, not a launch.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal", default=None, help="Features, members and workload in one line.")
    parser.add_argument("--site", type=Path, default=None, help="The cluster's site.yaml, if any.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="The TensorRT-LLM checkout, when there is no --site.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into.")
    args = parser.parse_args(argv[1:])
    if args.site is not None:
        site = yaml.safe_load(args.site.read_text(encoding="utf-8")) or {}
    elif args.repo is not None:
        site = {"trtllm_repo_path": str(args.repo.resolve()), "environment": {"kind": "local"}}
    else:
        parser.error("give --site, or --repo when the cluster has no site.yaml")
    try:
        derivation = derive_spec(args.checkpoint, site, parse_goal(args.goal), site_path=args.site)
    except DeriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.site is None:
        derivation.questions.append(
            "environment: no --site was given, so the spec carries `environment: {kind: local}`; "
            "replace it with the cluster's (its interpreter, image, mounts)"
        )
    path = write_derivation(args.out, derivation)
    print(f"→ {path}\n→ {args.out / REPORT_FILENAME}")
    from .advice import spec_warnings

    written = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deployments_mod.resolve_paths(written, path.resolve().parent)
    warnings = spec_warnings(written)
    for warning in warnings:
        print(f"  warning: {warning}")
    for question in derivation.questions:
        print(f"  open: {question}")
    return 0


__all__ = [
    "ENGINES_DIRNAME",
    "REPORT_FILENAME",
    "SPEC_FILENAME",
    "DeriveError",
    "derive_spec",
    "draft_layers",
    "hybrid_marker",
    "main",
    "parse_goal",
    "read_config",
    "write_derivation",
]
