# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast path: the run's spec derived in seconds from the checkpoint, a site file and a goal.

``intake derive --checkpoint DIR (--site FILE | --repo DIR) --out DIR [--goal "..."]``
writes ``task.derived.yaml``, whose deployments name their options by path
(``engines/<name>.yaml``), prints what it left for the operator to answer and the spec's
warnings (``spec/advice.py``), and stops: no run, no GPU.

The goal is one line of words. Members (``tp1 tep4 dep16``): one deployment each, the first
is core's. Features (``mtp``, ``multimodal``, ``disagg`` or ``disagg@<member>``): one more
deployment each; ``-<feature>`` leaves out one the checkpoint carries, and a goal that names
any feature gets those and no other. ``kv=<dtype>[/<dtype>...]``: the first is the format
every deployment stores its cache in, each other one a deployment of its own. ``arch=``,
``isl=``, ``osl=``, ``max_seq_len=``, ``max_batch_size=``, ``benchmark=``, ``model_key=``
and ``reference=`` state the rest. Unsaid: the site's defaults, then one ``tp1``, the
checkpoint's own drafter and towers, 4096 tokens in and 1024 out at batch 64. What each
deployment's options come to is the shorthand's arithmetic (``spec/shorthand.py``).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import deployments as deployments_mod
from . import shorthand
from .advice import DRAFT_LAYER_KEYS
from .shorthand import DISAGG, FEATURES, MEMBER
from .task_schema import ACCURACY_FIELD

SPEC_FILENAME = "task.derived.yaml"
#: Where each deployment's native options are written, beside the spec.
ENGINES_DIRNAME = "engines"
#: Stated rather than left to the engine, which would size its cache for the checkpoint's
#: whole ``max_position_embeddings``.
DEFAULT_WORKLOAD: dict[str, int] = {"isl": 4096, "osl": 1024, "max_batch_size": 64}
#: ``key=value`` words: the workload's, and the ``accuracy`` keys the others set.
_WORKLOAD = ("isl", "osl", "max_seq_len", "max_batch_size")
_ACCURACY = {"benchmark": "benchmark", "model_key": "model_key", "reference": "reference_yaml"}
#: KV dtypes a goal may name for the checkpoint's own format, which no option states.
_KV_OWN = ("auto", "bf16", "bfloat16")
#: Weight formats a reference row states no ``quant_algo`` for.
_UNQUANTIZED = ("", "none", "auto", "bf16", "bfloat16", "fp16", "float16", "fp32")
USAGE = """usage: trtllm-modeling-v2 intake derive --checkpoint DIR (--site FILE | --repo DIR) --out DIR [--goal "..."]

Derive a spec from the checkpoint, the goal and an optional site.yaml; write it and each
deployment's options (engines/<name>.yaml) into --out, print what it left open and what it
is likely to cost or miss, and stop: no run, no GPU. spec/INTAKE.md in the workflow's
package is the procedure. `trtllm-modeling-v2 --task SPEC --workspace W --dry-run`
validates a spec.
"""


class DeriveError(ValueError):
    """The goal or the site cannot be derived from, or the derived spec leaves questions open."""


@dataclass
class Goal:
    """What a goal line states; everything else comes from the site, then the defaults."""

    members: list[str] = field(default_factory=list)
    #: A feature named (``True``) or left out with ``-`` (``False``).
    features: dict[str, bool] = field(default_factory=dict)
    disagg_member: str = ""
    #: ``arch``, the workload's keys, the ``accuracy`` keys, and ``kv`` as a list of dtypes.
    stated: dict[str, Any] = field(default_factory=dict)


def parse_goal(text: str | None) -> Goal:
    """Read a goal line (the module docstring says its words); every bad word in one error."""
    goal, bad = Goal(), []
    for word in re.split(r"[\s,;]+", (text or "").strip()):
        key, stated, value = word.partition("=")
        name, _, member = word.lstrip("-").partition("@")
        if not word:
            continue
        if MEMBER.match(word):
            goal.members.append(word)
        elif name in FEATURES and (not member or (name == DISAGG and MEMBER.match(member))):
            goal.features[name] = not word.startswith("-")
            goal.disagg_member = member or goal.disagg_member
        elif member:
            bad.append(f"{word!r}: only disagg takes a member of its own (`disagg@tep4`)")
        elif key == "kv" and value and "kv" not in goal.stated:
            goal.stated["kv"] = [dtype.lower() for dtype in value.split("/")]
        elif key in _WORKLOAD and value.isdigit():
            goal.stated[key] = int(value)
        elif stated and value and key in ("arch", *_ACCURACY):
            goal.stated[_ACCURACY.get(key, key)] = value
        else:
            said = "kv= is named once" if key == "kv" else "is not a member, a feature or a setting"
            bad.append(f"{word!r} {said}")
    if bad:
        raise DeriveError("the goal cannot be read:\n  - " + "\n  - ".join(bad))
    return goal


@dataclass
class Derivation:
    """A derived spec, in the shorthand its deployments are expanded from, and what it left open."""

    spec: dict[str, Any]
    questions: list[str] = field(default_factory=list)


def derive_spec(
    checkpoint: Path, site: Mapping[str, Any], goal: Goal, *, site_path: Path | None = None
) -> Derivation:
    """The spec ``goal`` asks of ``checkpoint`` on ``site``; see the module docstring."""
    if not isinstance(site.get("trtllm_repo_path"), str) or not isinstance(
        site.get("environment"), Mapping
    ):
        raise DeriveError(
            f"{site_path or 'the site'} must state `trtllm_repo_path` and `environment`"
        )
    checkpoint = Path(checkpoint).resolve()
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DeriveError(f"{checkpoint}/config.json cannot be read: {exc}") from exc
    defaults, questions = dict(site.get("defaults") or {}), []
    arch = goal.stated.get("arch") or (site.get("device") or {}).get("arch")
    if not arch:
        questions.append(
            "arch: name the GPU architecture (arch=sm_103 in --goal, or the site's `device.arch`)"
        )
    members = goal.members or list(defaults.get("members") or ("tp1",))
    # The checkpoint's own drafter and towers, unless the goal names its features or drops them.
    carried = [
        *(["mtp"] if any(config.get(k) for k in DRAFT_LAYER_KEYS) else []),
        *(["multimodal"] if {"vision_config", "audio_config"} & set(config) else []),
    ]
    named = [name for name, on in goal.features.items() if on]
    features = named or [name for name in carried if goal.features.get(name) is not False]
    workload = {**DEFAULT_WORKLOAD, **(defaults.get("workload") or {})}
    reach = config.get("max_position_embeddings")
    if isinstance(reach, int) and workload["isl"] + workload["osl"] > reach:
        # A default must fit the model; a stated workload that does not is the operator's.
        workload["osl"] = min(workload["osl"], reach // 4)
        workload["isl"] = reach - workload["osl"]
    workload.update({k: v for k, v in goal.stated.items() if k in _WORKLOAD})
    accuracy = {
        k: v for k, v in (defaults.get("accuracy") or {}).items() if k in _ACCURACY.values()
    }
    accuracy.update({k: v for k, v in goal.stated.items() if k in _ACCURACY.values()})
    name = str(config.get("_name_or_path") or "")
    if not accuracy.get("model_key") and re.match(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", name):
        accuracy["model_key"] = name
    if not accuracy.get("model_key"):
        questions.append(
            "accuracy.model_key: the checkpoint names no `<org>/<model>`; state the key of its "
            "reference row, or drop `accuracy` to onboard on boot only"
        )
    # A reference row matches only its exact tags: a quantized checkpoint is not the bf16 row's.
    quant = dict(config.get("quantization_config") or {})
    try:
        stated = json.loads((checkpoint / "hf_quant_config.json").read_text(encoding="utf-8"))
        quant = {**(stated.get("quantization") or {}), **quant}
    except (OSError, ValueError, AttributeError):
        pass
    algo = str(quant.get("quant_algo") or "")
    if not algo and str(quant.get("quant_method") or "").lower() == "fp8":
        algo = "FP8_BLOCK_SCALES" if quant.get("weight_block_size") else "FP8"
    if algo.lower() not in _UNQUANTIZED:
        accuracy["discriminators"] = {"quant_algo": algo.upper()}
    elif quant and not algo:
        questions.append(
            "accuracy.discriminators: the checkpoint is quantized and states no quant_algo; "
            "state its reference row's tags, so the anchor is not the unquantized model's"
        )
    accuracy["max_seq_len"] = workload.get("max_seq_len") or workload["isl"] + workload["osl"]
    accuracy["max_batch_size"] = workload["max_batch_size"]
    if defaults.get(DISAGG) and DISAGG in features:
        accuracy[DISAGG] = dict(defaults[DISAGG])
    layers = max((int(config.get(k) or 0) for k in DRAFT_LAYER_KEYS), default=0)
    if "mtp" in features and layers > 1:
        accuracy["llm_api_options"] = {"mtp": {"speculative_config": {"max_draft_len": layers}}}
    own, *others = goal.stated.get("kv") or ["auto"]
    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": site["trtllm_repo_path"],
        **({"arch": arch} if arch else {}),
        "members": {"default": members, DISAGG: [goal.disagg_member]}
        if goal.disagg_member
        else members,
        "features": features,
        "scope": "text_only"
        if "multimodal" in carried and "multimodal" not in features
        else "full",
        "runtime": {} if own in _KV_OWN else {"extra_flags": [f"--kv_cache_dtype={own}"]},
        "environment": dict(site["environment"]),
        ACCURACY_FIELD: accuracy,
        "variants": {
            f"kv_{dtype}": {"llm_api_options": {"kv_cache_config": {"dtype": dtype}}}
            for dtype in others
            if dtype not in _KV_OWN
        },
        **{key: dict(site[key]) for key in ("agents", "commit_policy") if site.get(key)},
    }
    errors: list[str] = []
    shorthand.validate(spec, errors)
    if errors:
        raise DeriveError(
            "the goal asks for a spec that does not validate:\n  - " + "\n  - ".join(errors)
        )
    return Derivation(spec, questions)


def write_derivation(workspace: Path, derivation: Derivation) -> Path:
    """Write the spec and each deployment's options into ``workspace``; the spec's path.

    The shorthand is expanded here into the deployments the spec states: the options go to
    ``engines/<name>.yaml`` and the spec names them by path.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    asked = derivation.spec
    spec = {k: v for k, v in asked.items() if k not in shorthand.SHORTHAND_KEYS}
    kept = {k: v for k, v in asked[ACCURACY_FIELD].items() if k not in shorthand.ACCURACY_KEYS}
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
    return path


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 intake derive ...``: 0 written, 2 a goal or site it cannot derive from."""
    import argparse

    from .advice import spec_warnings

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    if argv[0] != "derive":
        replaced = "`intake lint` is replaced by `trtllm-modeling-v2 --task <spec> --workspace <W> --dry-run`"
        print(replaced if argv[0] == "lint" else USAGE, file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(prog="trtllm-modeling-v2 intake derive")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal", default=None, help="Members, features and settings in one line.")
    parser.add_argument("--site", type=Path, default=None, help="The cluster's site.yaml, if any.")
    parser.add_argument("--repo", type=Path, default=None, help="The checkout, with no --site.")
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into.")
    args = parser.parse_args(argv[1:])
    if args.site is None and args.repo is None:
        parser.error("give --site, or --repo when the cluster has no site.yaml")
    local = {
        "trtllm_repo_path": str((args.repo or Path()).resolve()),
        "environment": {"kind": "local"},
    }
    site = yaml.safe_load(args.site.read_text(encoding="utf-8")) if args.site else local
    try:
        derivation = derive_spec(
            args.checkpoint, site or {}, parse_goal(args.goal), site_path=args.site
        )
    except DeriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.site is None:
        derivation.questions.append(
            "environment: no --site was given; replace `{kind: local}` with the cluster's"
        )
    path = write_derivation(args.out, derivation)
    print(f"→ {path}")
    written = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deployments_mod.resolve_paths(written, path.resolve().parent)
    for line in [f"warning: {w}" for w in spec_warnings(written)] + [
        f"open: {q}" for q in derivation.questions
    ]:
        print(f"  {line}")
    return 0
