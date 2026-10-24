# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast path: a ``task.yaml`` derived in seconds from the checkpoint, ``site.yaml`` and a goal.

``trtllm-modeling-v2 --checkpoint <hf_dir> --site site.yaml --workspace W [--goal ...]``
reads the checkpoint's ``config.json`` (and ``hf_quant_config.json``), takes the
device, the default members and workload, the accuracy anchor, the boot prompts
and the witness declarations from ``site.yaml``, and the features, members and
workload from the goal, and writes the request to ``<workspace>/task.intake.yaml``
with every value's provenance, beside ``intake.report.md``. The run starts from it
at once; its intake checks run in phase 0, beside the survey and the substrate
gate (``run/stages/preflight.py``). What cannot be derived is an open question,
which those checks report as a block.

A feature the checkpoint carries (draft layers, a vision or audio tower) is
built unless the goal puts it out of scope; anything else is built only when
the goal names it. The goal is one line of words, separated by spaces, commas
or semicolons:

====================================  =============================================
``mtp`` ``eagle3`` ``dflash``         speculative decoding with that algorithm
``dspark``
``disagg`` ``multimodal`` ...         a feature, by its INTAKE.md section 5 name
``<feature>@tp1/tep4``                the feature, gated on those members only
``-<feature>`` ``no-<feature>``       out of scope; ``text_only`` is ``-multimodal``
``tp1`` ``tep4`` ``dep4`` ...         the members (default: the site's, else tp1)
``isl=`` ``osl=`` ``max_seq_len=``    the workload; ``concurrency=1/16/64``
``max_batch_size=``
``tune``                              tune the target once it is delivered
``arch=sm_100``                       the device (default: the site's)
``benchmark=`` ``model_key=``         the accuracy anchor
``reference=<yaml>``
``draft_len=`` ``drafter=<dir>``      speculative decoding's depth and drafter
``kv=fp8`` ``memory=0.6``             the KV cache's dtype and memory fraction
``chunk=<tokens>``                    the prefill chunk (``max_num_tokens``)
``snapshot=<tokens>``                 a hybrid checkpoint's state snapshot interval
====================================  =============================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .checks import draft_layers, read_config
from .compile import unit_name
from .freeze import confirmation_status
from .schema import (
    CORE_FEATURES,
    GOAL_ONBOARD,
    GOAL_TUNE,
    INTAKE_VERSION,
    KNOWN_FEATURES,
    PARALLEL_SEGMENT,
    SPECULATIVE_ALGOS,
    STATUS_OUT,
    STATUS_REQUIRED,
    WORKLOAD_KEYS,
    IntakeError,
    validate_site,
)

#: The derived request and its report, in the workspace.
REQUEST_FILENAME = "task.intake.yaml"
REPORT_FILENAME = "intake.report.md"

#: Provenance of a value: stated in the goal, read from the checkpoint or the
#: site, or a default of this module.
USER, INFERRED, DEFAULT = "user", "inferred", "default"

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

#: Pipeline feature -> the witness keys it cannot judge without, which nothing
#: can derive from the checkpoint. A replay witness's ``window`` and prompt
#: length are derived from ``sliding_window`` (:func:`_witness`). The
#: disaggregated witness needs none: its leg B is the flow's own
#: (``checks/harness/disagg_leg_b.py``) unless the site names a
#: ``contrast_command``, which then needs its ``contrast_env``.
WITNESS_REQUIRED: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("contrast_env",),
    "encoder_replay": ("contrast_env",),
    "kv_quant": ("expected_bytes_per_token", "contrast_env"),
    "host_offload": ("contrast_env",),
    "disagg": (),
}
#: A replay witness's long prompt (or shared prefix), in multiples of the window.
REPLAY_PROMPT_WINDOWS = 4

#: Goal word -> intake feature, for the words that are not a feature's own name.
_FEATURE_WORDS = {"disagg": "disaggregated", "spec": "speculative"}
#: ``key=value`` goal words -> (feature, parameter, type).
_PARAM_WORDS: dict[str, tuple[str, str, type]] = {
    "draft_len": ("speculative", "draft_len", int),
    "drafter": ("speculative", "drafter", str),
    "kv": ("kv_cache", "dtype", str),
    "memory": ("kv_cache", "memory_fraction", float),
    "chunk": ("chunked_prefill", "chunk_tokens", int),
    "snapshot": ("prefix_reuse", "snapshot_interval", int),
}
_ACCURACY_WORDS = {
    "benchmark": "benchmark",
    "model_key": "model_key",
    "reference": "reference_yaml",
}
_KV_DTYPES = ("auto", "fp8", "nvfp4")
_REPO_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
#: The core switches every derived request turns on unless the goal turns one off.
_CORE_SWITCHES = tuple(name for name in CORE_FEATURES if name != "kv_cache")


@dataclass
class Goal:
    """What a ``--goal`` line asks for."""

    text: str = ""
    #: Intake feature -> status, for every feature the goal names.
    features: dict[str, str] = field(default_factory=dict)
    #: The speculative algorithm the goal names.
    algo: str | None = None
    #: Intake feature -> the members the goal narrows it to.
    feature_members: dict[str, list[str]] = field(default_factory=dict)
    members: list[str] = field(default_factory=list)
    workload: dict[str, Any] = field(default_factory=dict)
    #: Intake feature -> parameters.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    accuracy: dict[str, str] = field(default_factory=dict)
    arch: str | None = None
    tune: bool = False


def _feature(word: str) -> tuple[str, str | None] | None:
    """``(intake feature, speculative algorithm)`` a goal word names, or None."""
    if word in SPECULATIVE_ALGOS:
        return "speculative", word
    name = _FEATURE_WORDS.get(word, word)
    if name in KNOWN_FEATURES and name != "custom":
        return name, None
    return None


def parse_goal(text: str | None) -> Goal:
    """Read a goal line (module docstring). Raises :class:`IntakeError` naming every unknown word."""
    goal = Goal(text=" ".join((text or "").split()))
    errors: list[str] = []
    for word in re.split(r"[\s,;]+", goal.text):
        if not word:
            continue
        if "=" in word:
            _key_value(goal, word, errors)
            continue
        lowered = word.lower()
        if lowered == GOAL_TUNE:
            goal.tune = True
            continue
        if PARALLEL_SEGMENT.match(lowered):
            goal.members.append(lowered)
            continue
        if lowered == "text_only":
            goal.features["multimodal"] = STATUS_OUT
            continue
        negated = lowered.startswith(("-", "no-", "no_"))
        name, _, members = (
            lowered.lstrip("-").removeprefix("no-").removeprefix("no_").partition("@")
        )
        found = _feature(name)
        if found is None:
            errors.append(f"goal: {word!r} is not a feature, a member or key=value")
            continue
        feature, algo = found
        goal.features[feature] = STATUS_OUT if negated else STATUS_REQUIRED
        if algo and not negated:
            goal.algo = algo
        if members:
            listed = [m for m in members.split("/") if m]
            bad = [m for m in listed if not PARALLEL_SEGMENT.match(m)]
            if bad or not listed:
                errors.append(f"goal: {word!r} names no members of the form tp<N>, tep<N>, dep<N>")
            else:
                goal.feature_members[feature] = listed
    if errors:
        raise IntakeError("\n".join(errors))
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
        elif key in _PARAM_WORDS:
            feature, param, kind = _PARAM_WORDS[key]
            parsed = kind(value.lower() if kind is str and key == "kv" else value)
            if key == "kv" and parsed not in _KV_DTYPES:
                errors.append(f"goal: kv={value} is not one of {list(_KV_DTYPES)}")
                return
            goal.params.setdefault(feature, {})[param] = parsed
        else:
            errors.append(f"goal: {key!r} in {word!r} is not a key the goal takes")
    except ValueError:
        errors.append(f"goal: {word!r} needs a number")


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
    moe: bool
    #: ``mtp`` or ``dspark`` for draft layers in the checkpoint, and their depth.
    speculator: str | None
    draft_len: int | None
    towers: tuple[str, ...]
    #: What marks the checkpoint hybrid (``run/preflight.hybrid_marker``), or None.
    hybrid: str | None
    #: The ``<org>/<model>`` name the config states, the reference rows' key.
    name: str | None


def inspect_checkpoint(path: Path) -> CheckpointFacts:
    """Read ``path``'s config files. Raises :class:`IntakeError` without a ``config.json``."""
    from ...run.preflight import hybrid_marker

    config = read_config(path)
    if config is None:
        raise IntakeError(f"{path}: no readable config.json; this is not a HuggingFace checkpoint")
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
    unread = None
    if quant and not quant_algo:
        unread = method or "a quantization_config without quant_algo"
    speculator = None
    draft_len = None
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
        unread_quantization=unread,
        moe=bool(
            config.get("num_experts")
            or config.get("n_routed_experts")
            or config.get("num_local_experts")
        ),
        speculator=speculator,
        draft_len=int(draft_len) if isinstance(draft_len, int) and draft_len > 0 else None,
        towers=tuple(k for k in ("vision_config", "audio_config") if k in config),
        hybrid=hybrid_marker(config),
        name=str(name) if isinstance(name, str) and _REPO_NAME.match(name) else None,
    )


@dataclass
class Derivation:
    """A derived request, and what it holds that the goal did not state."""

    task: dict[str, Any]
    #: ``(field, value, source)`` for every value read from the checkpoint or
    #: the site, or defaulted; ``source`` is ``checkpoint``, ``site`` or ``default``.
    unstated: list[tuple[str, Any, str]] = field(default_factory=list)


class _Builder:
    """Records each value's provenance as the request is assembled."""

    def __init__(self) -> None:
        self.provenance: dict[str, str] = {}
        self.unstated: list[tuple[str, Any, str]] = []
        self.questions: list[str] = []

    def note(self, where: str, value: Any, source: str) -> Any:
        self.provenance[where] = (
            USER if source == USER else (DEFAULT if source == DEFAULT else INFERRED)
        )
        if source != USER:
            self.unstated.append((where, value, source))
        return value


def derive_task(
    checkpoint: Path,
    site: Mapping[str, Any],
    goal: Goal,
    *,
    site_path: Path | None = None,
    now: datetime | None = None,
) -> Derivation:
    """The request ``goal`` asks of ``checkpoint`` on ``site``; see the module docstring."""
    errors = validate_site(site)
    if errors:
        raise IntakeError("\n".join(errors))
    checkpoint = Path(checkpoint).resolve()
    facts = inspect_checkpoint(checkpoint)
    defaults = dict(site.get("defaults") or {})
    b = _Builder()

    model: dict[str, Any] = {"checkpoint": str(checkpoint)}
    model["architecture"] = b.note("model.architecture", facts.architecture, "checkpoint")
    model["kind"] = b.note("model.kind", "moe" if facts.moe else "dense", "checkpoint")
    if facts.quant_algo:
        quantization = {"weights": facts.quant_algo}
        if facts.kv_cache_quant_algo:
            quantization["kv_cache"] = facts.kv_cache_quant_algo
        model["quantization"] = b.note("model.quantization", quantization, "checkpoint")
    elif facts.unread_quantization:
        b.questions.append(
            f"the checkpoint is quantized ({facts.unread_quantization}) and states no "
            f"quant_algo: set accuracy.discriminators to the reference row's (quant_algo, "
            f"kv_cache_quant_algo), so the anchor is not the unquantized model's"
        )

    task: dict[str, Any] = {
        "version": INTAKE_VERSION,
        "goals": [GOAL_ONBOARD, *([GOAL_TUNE] if goal.tune else [])],
        "model": model,
        "device": _device(goal, site, b),
        "parallelism": _parallelism(goal, defaults, b),
        "workload": _workload(goal, defaults, facts, b),
        "features": _features(goal, facts, b),
        "accuracy": {},
    }
    task["accuracy"] = _accuracy(goal, defaults, facts, task["features"], b)
    task["provenance"] = dict(sorted(b.provenance.items()))
    task["open_questions"] = list(b.questions)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    task["derived"] = {
        "checkpoint": str(checkpoint),
        "site": str(Path(site_path).resolve()) if site_path else "",
        "goal": goal.text,
        "at": stamp,
    }
    return Derivation(task=task, unstated=b.unstated)


def _device(goal: Goal, site: Mapping[str, Any], b: _Builder) -> dict[str, Any]:
    site_device = dict(site.get("device") or {})
    device: dict[str, Any] = {}
    if goal.arch:
        device["arch"] = b.note("device.arch", goal.arch, USER)
    elif site_device.get("arch"):
        device["arch"] = b.note("device.arch", site_device["arch"], "site")
    else:
        b.questions.append(
            "device.arch: name the GPU architecture (arch=sm_100 in --goal, or `device: "
            "{arch: ...}` in site.yaml); receipts are keyed by it and it is never guessed"
        )
    for key in ("gpus_per_node", "nodes"):
        if site_device.get(key):
            device[key] = b.note(f"device.{key}", site_device[key], "site")
    return device


def _parallelism(goal: Goal, defaults: Mapping[str, Any], b: _Builder) -> list[dict[str, Any]]:
    if goal.members:
        members, source = goal.members, USER
    elif defaults.get("members"):
        members, source = list(defaults["members"]), "site"
    else:
        members, source = list(DEFAULT_MEMBERS), DEFAULT
    b.note("parallelism", list(dict.fromkeys(members)), source)
    out = []
    for member in dict.fromkeys(members):
        match = PARALLEL_SEGMENT.match(member)
        out.append({"kind": match.group(1), "world": int(match.group(2))})
    return out


def _workload(
    goal: Goal, defaults: Mapping[str, Any], facts: CheckpointFacts, b: _Builder
) -> dict[str, Any]:
    site_workload = dict(defaults.get("workload") or {})
    if site_workload:
        workload, source = site_workload, "site"
    else:
        workload, source = dict(DEFAULT_WORKLOAD), DEFAULT
        reach = facts.config.get("max_position_embeddings")
        if isinstance(reach, int) and workload["isl"] + workload["osl"] > reach:
            # A default must fit the model; a stated workload that does not is a block.
            workload["osl"] = min(workload["osl"], reach // 4)
            workload["isl"] = reach - workload["osl"]
    for key, value in workload.items():
        if key not in goal.workload:
            b.note(f"workload.{key}", value, source)
    for key, value in goal.workload.items():
        workload[key] = b.note(f"workload.{key}", value, USER)
    return workload


def _features(goal: Goal, facts: CheckpointFacts, b: _Builder) -> dict[str, Any]:
    features: dict[str, Any] = {}

    def entry(name: str, status: str, source: str) -> dict[str, Any]:
        value: dict[str, Any] = {"status": b.note(f"features.{name}", status, source)}
        params = dict(goal.params.get(name) or {})
        for key in params:
            b.note(f"features.{name}.params.{key}", params[key], USER)
        if params and status != STATUS_OUT:
            value["params"] = params
        if goal.feature_members.get(name) and status != STATUS_OUT:
            value["members"] = b.note(f"features.{name}.members", goal.feature_members[name], USER)
        features[name] = value
        return value

    for name in _CORE_SWITCHES:
        status = goal.features.get(name)
        entry(name, status or STATUS_REQUIRED, USER if status else DEFAULT)
    reuse = features["prefix_reuse"]
    if facts.hybrid and reuse["status"] != STATUS_OUT:
        params = reuse.setdefault("params", {})
        if "snapshot_interval" not in params:
            params["snapshot_interval"] = b.note(
                "features.prefix_reuse.params.snapshot_interval", DEFAULT_SNAPSHOT_INTERVAL, DEFAULT
            )

    kv_dtype = (goal.params.get("kv_cache") or {}).get("dtype")
    native_kv = str(facts.kv_cache_quant_algo or "").lower()
    if goal.features.get("kv_cache") or kv_dtype or native_kv in _KV_DTYPES:
        stated = goal.features.get("kv_cache") or kv_dtype
        kv = entry(
            "kv_cache",
            goal.features.get("kv_cache") or STATUS_REQUIRED,
            USER if stated else "checkpoint",
        )
        if not kv_dtype and native_kv in _KV_DTYPES and kv["status"] != STATUS_OUT:
            kv.setdefault("params", {})["dtype"] = b.note(
                "features.kv_cache.params.dtype", native_kv, "checkpoint"
            )

    spec_status = goal.features.get("speculative")
    if spec_status == STATUS_OUT:
        entry("speculative", STATUS_OUT, USER)
    elif goal.algo or facts.speculator:
        algo = goal.algo or facts.speculator
        source = USER if goal.algo else "checkpoint"
        spec = entry("speculative", STATUS_REQUIRED, source)
        params = spec.setdefault("params", {})
        params["algo"] = b.note("features.speculative.params.algo", algo, source)
        if "draft_len" not in params and facts.draft_len and algo == facts.speculator:
            params["draft_len"] = b.note(
                "features.speculative.params.draft_len", facts.draft_len, "checkpoint"
            )
    elif spec_status:
        b.questions.append(
            "speculative: the goal asks for speculative decoding and the checkpoint carries "
            "no draft layers; name the algorithm (mtp, eagle3, dflash or dspark)"
        )

    if "multimodal" in goal.features:
        entry("multimodal", goal.features["multimodal"], USER)
    elif facts.towers:
        entry("multimodal", STATUS_REQUIRED, "checkpoint")

    handled = {*_CORE_SWITCHES, "kv_cache", "speculative", "multimodal"}
    for name, status in goal.features.items():
        if name not in handled:
            entry(name, status, USER)
    return features


def _accuracy(
    goal: Goal,
    defaults: Mapping[str, Any],
    facts: CheckpointFacts,
    features: Mapping[str, Any],
    b: _Builder,
) -> dict[str, Any]:
    site_accuracy = dict(defaults.get("accuracy") or {})
    accuracy: dict[str, Any] = {}
    for key, value in site_accuracy.items():
        accuracy[key] = b.note(f"accuracy.{key}", value, "site")
    for key, value in goal.accuracy.items():
        accuracy[key] = b.note(f"accuracy.{key}", value, USER)
    if "benchmark" not in accuracy:
        accuracy["benchmark"] = b.note("accuracy.benchmark", DEFAULT_BENCHMARK, DEFAULT)
    if "model_key" not in accuracy and facts.name:
        accuracy["model_key"] = b.note("accuracy.model_key", facts.name, "checkpoint")
    if defaults.get("boot"):
        accuracy["boot"] = b.note("accuracy.boot", dict(defaults["boot"]), "site")
    else:
        accuracy["boot"] = b.note("accuracy.boot", json.loads(json.dumps(DEFAULT_BOOT)), DEFAULT)
    witness = _witness(defaults, facts, features, b)
    if witness:
        accuracy["witness"] = witness
    return accuracy


def _witness(
    defaults: Mapping[str, Any], facts: CheckpointFacts, features: Mapping[str, Any], b: _Builder
) -> dict[str, Any]:
    """Each built feature's witness declaration: the site's, the window, and questions for the rest."""
    site_witness = dict(defaults.get("witness") or {})
    window = facts.config.get("sliding_window")
    out: dict[str, Any] = {}
    for name, raw in features.items():
        unit = unit_name(name, raw)
        if unit not in WITNESS_REQUIRED:
            continue
        block = dict(site_witness.get(unit) or {})
        if block:
            b.note(f"accuracy.witness.{unit}", dict(block), "site")
        if unit in ("decoder_replay", "encoder_replay") and isinstance(window, int) and window > 0:
            length_key = "prompt_tokens" if unit == "decoder_replay" else "prefix_tokens"
            if "window" not in block:
                block["window"] = b.note(f"accuracy.witness.{unit}.window", window, "checkpoint")
            if length_key not in block:
                block[length_key] = b.note(
                    f"accuracy.witness.{unit}.{length_key}",
                    REPLAY_PROMPT_WINDOWS * int(block["window"]),
                    DEFAULT,
                )
        missing = [key for key in WITNESS_REQUIRED[unit] if not block.get(key)]
        if unit in ("decoder_replay", "encoder_replay") and "window" not in block:
            missing.append("window")
        if unit == "disagg" and block.get("contrast_command") and not block.get("contrast_env"):
            missing.append("contrast_env")
        if missing:
            b.questions.append(
                f"accuracy.witness.{unit}: declare {', '.join(missing)} (site.yaml "
                f"`defaults.witness.{unit}`, or in this file): without them the {unit} "
                f"witness cannot tell a working feature from an absent one"
            )
        if block or unit == "disagg":
            out[unit] = block
    return out


def resume_problem(request: Mapping[str, Any], *, checkpoint: Path, goal: str | None) -> str | None:
    """Why the request already in a workspace is not the one this command asks for, or None.

    A request is re-derived only on ``--clean``: the one on disk may carry the
    operator's edits, and its run may have started from it.
    """
    derived = request.get("derived") if isinstance(request, Mapping) else None
    if not isinstance(derived, Mapping):
        return "the workspace's request was not derived by --checkpoint; run it with --task"
    if Path(str(derived.get("checkpoint") or "")).resolve() != Path(checkpoint).resolve():
        return (
            f"the workspace's request was derived from {derived.get('checkpoint')}, not "
            f"{checkpoint}: use another --workspace, or --clean to derive a new one"
        )
    if goal is not None and " ".join(goal.split()) != str(derived.get("goal") or ""):
        return (
            f"the workspace's request was derived for the goal {derived.get('goal')!r}: edit "
            f"the request, or --clean to derive it again for {goal!r}"
        )
    return None


def render_report(derivation: Derivation, request_path: Path, findings: list | None = None) -> str:
    """``intake.report.md``: what was read, what was defaulted, and what is still open."""
    task = derivation.task
    derived = task.get("derived") or {}
    lines = [
        "# Intake: the derived request",
        "",
        f"Derived from `{derived.get('checkpoint')}` ({task['model'].get('architecture')}), "
        f"the site `{derived.get('site') or '(unnamed)'}` and the goal "
        f"`{derived.get('goal') or '(none)'}` at {derived.get('at')}.",
        "",
        f"The run starts from `{request_path}` at once, and its intake checks run in "
        "phase 0, beside the survey and the substrate gate; a block stops the run there "
        "with every block listed. Edit the request and re-run the same command to change "
        "it; once the run is past preflight, an edit runs only when confirmed.",
        "",
        f"Confirmation: {confirmation_status(task, None)}. "
        f"`trtllm-modeling-v2 intake confirm {request_path} --by NAME` records it, or "
        "`--confirm` on the run's command.",
        "",
        "## What the goal did not state",
        "",
        "| field | from | value |",
        "|---|---|---|",
    ]
    for where, value, source in derivation.unstated:
        shown = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        lines.append(f"| `{where}` | {source} | `{shown}` |")
    questions = task.get("open_questions") or []
    lines += ["", f"## Open questions ({len(questions)})", ""]
    lines += [f"- {q}" for q in questions] or ["None."]
    if findings:
        lines += ["", "## Checks", "", "```"] + [f.render() for f in findings] + ["```"]
    return "\n".join(lines) + "\n"


def write_request(workspace: Path, derivation: Derivation, findings: list | None = None) -> Path:
    """Write the request and its report into ``workspace``; returns the request's path."""
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / REQUEST_FILENAME
    path.write_text(yaml.safe_dump(derivation.task, sort_keys=False), encoding="utf-8")
    (workspace / REPORT_FILENAME).write_text(
        render_report(derivation, path, findings), encoding="utf-8"
    )
    return path


__all__ = [
    "DEFAULT_BOOT",
    "DEFAULT_MEMBERS",
    "DEFAULT_SNAPSHOT_INTERVAL",
    "DEFAULT_WORKLOAD",
    "REPORT_FILENAME",
    "REQUEST_FILENAME",
    "WITNESS_REQUIRED",
    "CheckpointFacts",
    "Derivation",
    "Goal",
    "derive_task",
    "inspect_checkpoint",
    "parse_goal",
    "render_report",
    "resume_problem",
    "write_request",
]
