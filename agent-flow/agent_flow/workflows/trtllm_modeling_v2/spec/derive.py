# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``intake derive``: the smallest spec a checkpoint and a site support, written in seconds.

``intake derive --checkpoint DIR (--site FILE | --repo DIR) --out DIR`` writes
``task.derived.yaml``: the checkpoint, the site's checkout, environment and agents, and
one deployment, the first, whose options it writes beside the spec
(``engines/<name>.yaml``). It prints what it left open (``open:``) and the spec's warnings
(``spec/advice.py``), and stops: no run, no GPU. Every further deployment is the
operator's to add to the spec.

The first deployment is the site's first default member, else ``tp1``, serving text. Its
options are the shorthand's arithmetic for that member (``spec/shorthand.py``) at the
site's default workload, else 4096 tokens in and 1024 out at batch 64, fitted to what the
model reaches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import deployments as deployments_mod
from . import shorthand
from .advice import spec_warnings
from .task_schema import ACCURACY_FIELD

SPEC_FILENAME = "task.derived.yaml"
#: Where a deployment's native options are written, beside the spec.
ENGINES_DIRNAME = "engines"
#: Stated rather than left to the engine, which would size its cache for the checkpoint's
#: whole ``max_position_embeddings``.
DEFAULT_WORKLOAD: dict[str, int] = {"isl": 4096, "osl": 1024, "max_batch_size": 64}
#: Weight formats a reference row states no ``quant_algo`` for.
_UNQUANTIZED = ("", "none", "auto", "bf16", "bfloat16", "fp16", "float16", "fp32")
USAGE = """usage: trtllm-modeling-v2 intake derive --checkpoint DIR (--site FILE | --repo DIR) --out DIR

Write the smallest spec the checkpoint and the site support into --out: its first
deployment and that deployment's options (engines/<name>.yaml). Print what it left open
and what the spec is likely to cost or miss, and stop: no run, no GPU. Add deployments by
editing the spec; `trtllm-modeling-v2 --task SPEC --workspace W --dry-run` validates it.
"""


class DeriveError(ValueError):
    """The checkpoint or the site cannot be derived from."""


@dataclass
class Derivation:
    """A derived spec, still without its deployment: that one's name and options, and what is open."""

    spec: dict[str, Any]
    deployment: str
    options: dict[str, Any]
    questions: list[str] = field(default_factory=list)


def derive_spec(
    checkpoint: Path, site: Mapping[str, Any], *, site_path: Path | None = None
) -> Derivation:
    """The smallest spec ``checkpoint`` and ``site`` support; see the module docstring."""
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
    arch = (site.get("device") or {}).get("arch")
    if not arch:
        questions.append("arch: name the GPU architecture (sm_100 is B200, sm_103 is GB300)")
    if not (site.get("agents") or {}).get("defaults", {}).get("model"):
        questions.append("agents.defaults.model: name the model the run's roles work on")
    member = str((defaults.get("members") or ["tp1"])[0])
    workload = {**DEFAULT_WORKLOAD, **(defaults.get("workload") or {})}
    reach = config.get("max_position_embeddings")
    if isinstance(reach, int) and workload["isl"] + workload["osl"] > reach:
        # A default must fit the model.
        workload["osl"] = min(workload["osl"], reach // 4)
        workload["isl"] = reach - workload["osl"]
    accuracy = dict(defaults.get("accuracy") or {})
    name = str(config.get("_name_or_path") or "")
    if not accuracy.get("model_key") and re.match(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", name):
        accuracy["model_key"] = name
    if not accuracy.get("model_key"):
        questions.append(
            "accuracy.model_key: the checkpoint names no `<org>/<model>`; state the key of its "
            "reference row, or leave `accuracy` out to onboard on boot only"
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
    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": site["trtllm_repo_path"],
        **({"arch": arch} if arch else {}),
        "scope": "text_only" if {"vision_config", "audio_config"} & set(config) else "full",
        "environment": dict(site["environment"]),
        **({ACCURACY_FIELD: accuracy} if accuracy.get("model_key") else {}),
        **{key: dict(site[key]) for key in ("agents", "commit_policy") if site.get(key)},
    }
    # The first deployment's options: the shorthand's arithmetic for one member, no feature.
    limits = {
        "max_seq_len": workload.get("max_seq_len") or workload["isl"] + workload["osl"],
        "max_batch_size": workload["max_batch_size"],
    }
    asked = {**spec, "members": [member], ACCURACY_FIELD: {**accuracy, **limits}}
    errors: list[str] = []
    shorthand.validate(asked, errors)
    if errors:
        raise DeriveError(
            "the site's defaults ask for a spec that does not validate:\n  - "
            + "\n  - ".join(errors)
        )
    options = dict(shorthand.expand(asked)[0].server.options)
    return Derivation(spec, member, options, questions)


def write_derivation(out: Path, derivation: Derivation) -> Path:
    """Write the spec and its deployment's options into ``out``; the spec's path."""
    engines = out / ENGINES_DIRNAME
    engines.mkdir(parents=True, exist_ok=True)
    options = f"{ENGINES_DIRNAME}/{derivation.deployment}.yaml"
    (out / options).write_text(
        yaml.safe_dump(derivation.options, sort_keys=False), encoding="utf-8"
    )
    entry = {"name": derivation.deployment, "options": options}
    spec = {**derivation.spec, deployments_mod.DEPLOYMENTS_FIELD: [entry]}
    path = out / SPEC_FILENAME
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 intake derive ...``: 0 written, 2 a checkpoint or site it cannot read."""
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
    parser.add_argument("--site", type=Path, default=None, help="The cluster's site.yaml, if any.")
    parser.add_argument("--repo", type=Path, default=None, help="The checkout, with no --site.")
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into.")
    parser.add_argument("--goal", default=None, help="Ignored, with a notice.")
    args = parser.parse_args(argv[1:])
    if args.site is None and args.repo is None:
        parser.error("give --site, or --repo when the cluster has no site.yaml")
    if args.goal is not None:
        print(
            "note: --goal is ignored: derive writes the first deployment only; add what else "
            "the run should bring up by editing the spec's `deployments`",
            file=sys.stderr,
        )
    local = {
        "trtllm_repo_path": str((args.repo or Path()).resolve()),
        "environment": {"kind": "local"},
    }
    site = yaml.safe_load(args.site.read_text(encoding="utf-8")) if args.site else local
    try:
        derivation = derive_spec(args.checkpoint, site or {}, site_path=args.site)
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
    lines = [f"warning: {w}" for w in spec_warnings(written)]
    for line in lines + [f"open: {q}" for q in derivation.questions]:
        print(f"  {line}")
    return 0
