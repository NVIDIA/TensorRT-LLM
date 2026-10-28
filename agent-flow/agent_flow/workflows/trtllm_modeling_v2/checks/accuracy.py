# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy evaluation, reference lookup and cross-member findings."""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.compute.environment import RunEnvironment
from ..infra.files import write_text_atomic
from ..spec.task_schema import resolve_per_unit, resolve_samples
from . import accuracy_policy, check_config, feature_evidence
from .check_config import check_configuration, engine_declaration, resolve_llm_api_options
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .gate_records import KIND_FINDING, GateRecord, GateRecords, open_findings
from .runtime_evidence import EVAL_NOT_ATTESTED, attest_engine, engine_failure, engine_records
from .verdict import UNKNOWN_TREE_STATE, GateResult, inspect_tree, lost_session, verdict_factory

ACCURACY_TIMEOUT_S = 4 * 60 * 60


# ``lm-eval <task> <filter> accuracy: 90.67`` and
# ``lm-eval <task> average accuracy: 90.67`` — the two shapes
# ``tensorrt_llm/evaluate/lm_eval.py`` emits at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")


# ``MMLU weighted average accuracy: 74.98 (14042)`` — the MMLU path has its
# own reporting because it does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")


def resolve_discriminators(
    accuracy: Mapping[str, Any], unit: str | None
) -> Mapping[str, Any] | None:
    """Pick the accuracy-reference discriminators for ``unit``; ``None`` when there are none.

    Accepts either a flat mapping (the whole run) or a mapping from unit
    name to mapping, with ``default`` as the fallback key.
    """
    block = accuracy.get("discriminators")
    if not isinstance(block, dict):
        return None
    # Per-unit form is recognised by its values being mappings; a flat form
    # like {"quant_algo": "NVFP4"} has scalar values.
    if any(isinstance(v, dict) for v in block.values()):
        chosen = block.get(unit)
        if chosen is None:
            chosen = block.get("default")
        return chosen if isinstance(chosen, dict) else None
    return block


def eval_command(
    task_data: Mapping[str, Any],
    config: Mapping[str, Any],
    benchmark: str,
    *,
    workdir: str | Path,
    unit: str | None = None,
    member: str | None = None,
) -> list[str]:
    """Public trtllm-eval CLI using the same resolved configuration the verdict attests."""
    accuracy = dict(task_data.get("accuracy") or {})
    options = dict(config["llm"])
    options.pop("model")
    target = gate_workdir(workdir) / f"eval.{unit or 'unit'}.{member or 'member'}.yaml"
    write_text_atomic(target, yaml.safe_dump(options, sort_keys=True), mode=0o600)
    parts = [
        "trtllm-eval",
        shlex.quote(f"--model={task_data['checkpoint_path']}"),
        shlex.quote(f"--config={target}"),
        benchmark,
    ]
    if accuracy.get("dataset_path"):
        parts.append(shlex.quote(f"--dataset_path={accuracy['dataset_path']}"))
    parts.extend(
        shlex.quote(str(arg)) for arg in resolve_per_unit(accuracy, "benchmark_args", unit) or ()
    )
    return parts


def eval_harness_command(environment: RunEnvironment, parts: Sequence[str]) -> str:
    """Launch the installed public CLI; the one upstream rank hook observes workers."""
    environment = replace(
        environment, env={**environment.env, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    )
    return environment.command(" ".join(parts))


def parse_eval_score(stdout: str) -> float | None:
    """Extract the final accuracy from a ``trtllm-eval`` run's output.

    MMLU prints its own weighted average, everything else comes through
    lm-eval. The **last** match wins: a long run also prints partial-score
    estimates.
    """
    for pattern in (_MMLU_SCORE, _LM_EVAL_SCORE):
        matches = pattern.findall(stdout)
        if matches:
            return float(matches[-1])
    return None


def load_reference_score(
    reference_yaml: str | Path,
    model_key: str,
    *,
    discriminators: Mapping[str, Any] | None = None,
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    An entry matches when its discriminators (``quant_algo``,
    ``kv_cache_quant_algo``, ``spec_dec_algo``) are exactly the ones asked
    for, no more, no less. Raises ``KeyError`` when no entry, or more than
    one, matches: the anchor is never invented.
    """
    path = Path(reference_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"accuracy reference file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if model_key not in data:
        raise KeyError(
            f"{path} has no entry for {model_key!r}. The anchor comes from a written "
            f"source or from the user — it is never invented. Add the entry (citing "
            f"where the score came from) before the accuracy gate can run."
        )

    wanted = {k: v for k, v in (discriminators or {}).items() if v is not None}
    entries = data[model_key] or []
    matches = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        present = {k: v for k, v in entry.items() if k != "accuracy"}
        if present == wanted:
            matches.append(entry)

    if not matches:
        available = [
            {k: v for k, v in e.items() if k != "accuracy"} for e in entries if isinstance(e, dict)
        ]
        raise KeyError(
            f"{path}:{model_key} has no entry matching {wanted or '{}'}. Available "
            f"discriminator sets: {available}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"{path}:{model_key} has {len(matches)} entries matching {wanted or '{}'}; "
            f"the anchor must be unambiguous"
        )
    return float(matches[0]["accuracy"])


def resolve_floor(
    accuracy: Mapping[str, Any],
    unit: str | None,
    benchmark: str,
    reference: float,
    output: str = "",
) -> tuple[accuracy_policy.Floor | None, str]:
    """The floor ``reference`` gets under ``accuracy.floor``, and where its count came from."""
    policy = accuracy_policy.resolve_floor_policy(accuracy)
    samples, source = resolve_samples(accuracy, unit, benchmark, output)
    return accuracy_policy.floor_for(policy, reference, samples), source


def run_accuracy_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    reference: float,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int = ACCURACY_TIMEOUT_S,
    unit: str | None = None,
    member: str | None = None,
) -> GateResult:
    """Run ``trtllm-eval`` and compare against the anchor. One-sided.

    The floor is ``reference - k * SE`` by default (:func:`resolve_floor`), a
    fixed ``reference - tolerance`` under ``accuracy.floor.rule: tolerance``.
    The engine is held to the unit's options file (:func:`engine_declaration`).
    """
    accuracy = dict(task_data.get("accuracy") or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    tolerance = float(accuracy.get("tolerance", 0.0))
    env = with_mode(environment, MODE_REQUIRE)
    _verdict = verdict_factory(inspect_tree(task_data))
    planned, _ = resolve_floor(accuracy, unit, benchmark, reference)

    def floor_fields(floor: accuracy_policy.Floor | None) -> dict[str, Any]:
        if floor is None:
            return {}
        return {
            "floor": floor.value,
            "se": floor.se,
            "samples": floor.samples,
            "floor_rule": floor.describe(),
        }

    name = f"accuracy[{benchmark}]"
    common = dict(name=name, member=member, reference=reference, tolerance=tolerance)
    config = check_configuration(
        task_data, environment, unit=unit, features=None, max_tokens=64, member=member
    )
    config["llm"]["modeling_v2"] = MODE_REQUIRE
    for key in ("max_seq_len", "max_batch_size"):
        if accuracy.get(key) is not None:
            config["llm"][key] = accuracy[key]
    parts = eval_command(task_data, config, benchmark, unit=unit, member=member, workdir=workdir)
    command = eval_harness_command(env, parts)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(command=command, duration_s=elapsed, stdout_tail=tail, **common)

    if result.infrastructure:
        return _verdict(**lost_session(result), **ran, **floor_fields(planned))
    if result.timed_out:
        return _verdict(
            passed=False, detail=f"timed out after {timeout}s", **ran, **floor_fields(planned)
        )
    if not result.ok:
        return _verdict(
            passed=False,
            detail=f"trtllm-eval exited {result.returncode}",
            **ran,
            **floor_fields(planned),
        )
    options = resolve_llm_api_options(accuracy, unit)
    declared = engine_declaration(config, mode=MODE_REQUIRE)
    asks = bool(declared.get("block_reuse"))
    attested = attest_engine(combined, declared, absent=EVAL_NOT_ATTESTED)
    ran["engine"] = engine_records((attested,))
    failure = engine_failure(combined, attested, options=options, asks=asks)
    if failure:
        return _verdict(passed=False, detail=failure, **ran, **floor_fields(planned))
    unobserved = feature_evidence.unobserved(attested, check_config.subject_fields(task_data, unit))
    if unobserved:
        return _verdict(
            passed=False,
            detail=feature_evidence.unobserved_detail(unobserved),
            **ran,
            **floor_fields(planned),
        )

    measured = parse_eval_score(combined)
    if measured is None:
        return _verdict(
            passed=False,
            detail="trtllm-eval exited 0 but printed no score this module recognizes; "
            "a gate with no number is not a pass",
            **ran,
            **floor_fields(planned),
        )

    floor, source = resolve_floor(accuracy, unit, benchmark, reference, combined)
    if floor is None:
        return _verdict(
            passed=False,
            detail=f"measured {measured:.4f}, but the statistical floor needs the sample "
            f"count and there is {source}. A floor that cannot be computed is not a pass",
            measured=measured,
            **ran,
        )
    passed = measured >= floor.value
    return _verdict(
        passed=passed,
        detail=(
            f"measured {measured:.4f} {'>=' if passed else '<'} {floor.value:.4f} "
            f"({floor.describe()}; count from {source}); score validation only; "
            + (
                f"engine observability UNKNOWN for {', '.join(attested.not_attested)}; "
                if attested.not_attested
                else ""
            )
            + "eval cache reuse was not probed (zero hits are not a failure)"
        ),
        measured=measured,
        **ran,
        **floor_fields(floor),
    )


def check_systematic_shift(
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> accuracy_policy.ShiftFinding | None:
    """The cross-member check over one unit's accuracy verdicts (:func:`.accuracy_policy.systematic_shift`).

    Verdicts without a measurement or an SE are left out.
    """
    policy = accuracy_policy.resolve_floor_policy(accuracy)
    readings = {
        str(r.member): (float(r.measured), float(r.reference), float(r.se))
        for r in results
        if r.measured is not None and r.reference is not None and r.se is not None
    }
    return accuracy_policy.systematic_shift(
        unit,
        readings,
        min_members=policy.shift_min_members,
        acknowledged=policy.acknowledged,
    )


#: The key a systematic-shift finding is recorded under, per unit.
SHIFT_FINDING = "systematic_shift"


def note_systematic_shift(
    records: GateRecords,
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> accuracy_policy.ShiftFinding | None:
    """Run :func:`check_systematic_shift` and record its outcome either way.

    A unit with no shift records a resolved entry, so a finding raised by an
    earlier turn of the same unit stops blocking once a later turn is clean.
    """
    finding = check_systematic_shift(results, accuracy, unit)
    tree = results[-1].tree_state if results else UNKNOWN_TREE_STATE
    records.append(
        GateRecord(
            kind=KIND_FINDING,
            key={"finding": SHIFT_FINDING, "unit": str(unit)},
            tree=tree,
            passed=finding is None or bool(finding.acknowledged),
            values={
                "detail": finding.describe() if finding else "no systematic shift",
                "members": {
                    str(r.member): {"measured": r.measured, "reference": r.reference, "se": r.se}
                    for r in results
                },
            },
        )
    )
    return finding


def blocking_findings(records: GateRecords, accuracy: Mapping[str, Any]) -> list[GateRecord]:
    """Findings that refuse delivery under the CURRENT ``accuracy.floor.acknowledge``."""
    return open_findings(records, accuracy_policy.resolve_floor_policy(accuracy).acknowledged)


def resolve_reference(
    task_data: Mapping[str, Any],
    *,
    model_key: str,
    discriminators: Mapping[str, Any] | None = None,
    unit: str | None = None,
) -> float:
    """Look up this run's anchor, resolving the reference path against the repo.

    ``discriminators`` overrides ``accuracy.discriminators`` for ``unit``.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    reference_yaml = Path(str(resolve_per_unit(accuracy, "reference_yaml", unit)))
    if not reference_yaml.is_absolute():
        reference_yaml = Path(str(task_data["trtllm_repo_path"])) / reference_yaml
    if discriminators is None:
        discriminators = resolve_discriminators(accuracy, unit)
    return load_reference_score(reference_yaml, model_key, discriminators=discriminators)


def resolve_model_key(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """The HuggingFace repo name this unit's anchor row is keyed by.

    An explicit ``accuracy.model_key`` when the spec states one (per unit,
    :func:`~.spec.task_schema.resolve_per_unit`), otherwise the last two components of the
    checkpoint path (``<org>/<model>``). This is the one implementation: the
    workflow's ``_model_key`` delegates here, so preflight's anchor check and
    the gate resolve the same key.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    explicit = resolve_per_unit(accuracy, "model_key", unit)
    if explicit:
        return str(explicit)
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


__all__ = [
    "resolve_model_key",
    "ACCURACY_TIMEOUT_S",
    "SHIFT_FINDING",
    "blocking_findings",
    "check_systematic_shift",
    "eval_command",
    "eval_harness_command",
    "load_reference_score",
    "note_systematic_shift",
    "parse_eval_score",
    "resolve_discriminators",
    "resolve_floor",
    "resolve_reference",
    "run_accuracy_gate",
]
