# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The envelope boot: the target served at the limits its runtime block declares.

Every other gate drives a handful of short prompts, so a target is never run
near the limits it claims to serve: ``max_batch_size`` requests in flight,
prompts near ``max_seq_len``, prefill chunks of ``max_num_tokens``, and every
CUDA-graph capture batch replayed. Failures that only appear there -- an
index that overflows int32, a buffer sized for the gate's batch, a top-k
above a kernel's limit -- crash or corrupt silently in production.

Two legs, one engine configuration, one tuning cache (:func:`gates.pin_tuning`):

- **reference**: a few short probe prompts, one at a time;
- **envelope**: the same probes, plus long prompts of exactly
  ``max_seq_len - max_tokens - 1`` token ids (capped at ``max_num_tokens``
  when chunked prefill is off), plus short fillers with staggered
  ``max_tokens`` up to ``max_batch_size`` requests in flight. The fillers
  finish at different steps, so the live batch shrinks through every size
  below the maximum and each captured graph is replayed.

After its requests each leg's driver sends the reuse probe, one prompt twice
(``checks/harness/engine_reuse.py``): it attests block reuse at every member,
and after the envelope leg it shows the engine still resumes from its KV cache
once it has served at the limits.

It passes when both legs exit cleanly, both engines ran the declared
configuration (:func:`gates.attest_engine`), neither engine of a unit that asks
for block reuse computed the probe's second prompt from the start
(:func:`gates.reuse_shortfall`), every long prompt reached its length, and each
probe's first ``agree_tokens`` token ids are the same in both legs. It does not
detect a small numeric drift near a limit; the kernels' own boundary tests
cover that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.environment import RunEnvironment
from .gates import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    EngineAttestation,
    GateResult,
    ShellRunner,
    _default_shell_runner,
    _verdict_factory,
    _with_mode,
    attest_engine,
    engine_declaration,
    engine_driver_command,
    engine_fields,
    gate_workdir,
    inspect_tree,
    parse_driver_records,
    pin_tuning,
    quickstart_words,
    resolve_llm_api_options,
    reuse_disabled_detail,
    reuse_shortfall,
    with_kv_cache_config,
    write_driver_spec,
)

GATE_NAME = "envelope"

#: The quickstart's own defaults, for a limit the argv does not state.
QUICKSTART_MAX_BATCH_SIZE = 2048
QUICKSTART_MAX_NUM_TOKENS = 8192

DEFAULT_MAX_TOKENS = 16
DEFAULT_AGREE_TOKENS = 4
DEFAULT_LONG_PROMPTS = 2
DEFAULT_PROBES: tuple[str, ...] = (
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "Two plus two equals",
)

#: Token ids the long prompts cycle through. Low ids exist in every
#: vocabulary of the size these models have; the text they spell is
#: irrelevant, only the length is.
_LONG_PROMPT_IDS = tuple(range(1000, 1997))


@dataclass(frozen=True)
class Envelope:
    """The limits a target is claimed to serve, read from the gate argv."""

    max_batch_size: int
    max_num_tokens: int
    max_seq_len: int | None
    chunked_prefill: bool
    cuda_graph: bool
    capture_batches: tuple[int, ...]

    def long_prompt_tokens(self, max_tokens: int) -> int:
        """The prompt length that puts a request at the sequence limit."""
        limit = int(self.max_seq_len or 0) - max_tokens - 1
        if not self.chunked_prefill:
            limit = min(limit, self.max_num_tokens - 1)
        return max(limit, 0)

    def describe(self) -> str:
        graphs = (
            f"cuda graph batches {list(self.capture_batches) or 'default'}"
            if self.cuda_graph
            else "no cuda graph"
        )
        chunked = "chunked" if self.chunked_prefill else "unchunked"
        return (
            f"max_batch_size {self.max_batch_size}, max_seq_len {self.max_seq_len}, "
            f"max_num_tokens {self.max_num_tokens} ({chunked}), {graphs}"
        )


def _flag_values(words: Sequence[str], flag: str) -> list[str]:
    """Every value given to ``flag`` in ``words``, in either spelling; the last one wins."""
    values: list[str] = []
    for index, word in enumerate(words):
        if word == flag:
            values = []
            for value in words[index + 1 :]:
                if value.startswith("--"):
                    break
                values.append(value)
        elif word.startswith(f"{flag}="):
            values = [word.split("=", 1)[1]]
    return values


def resolve_envelope(words: Sequence[str]) -> Envelope:
    """The envelope the quickstart argv ``words`` configures."""

    def single(flag: str, default: int | None) -> int | None:
        values = _flag_values(words, flag)
        return int(values[0]) if values else default

    return Envelope(
        max_batch_size=int(single("--max_batch_size", QUICKSTART_MAX_BATCH_SIZE)),
        max_num_tokens=int(single("--max_num_tokens", QUICKSTART_MAX_NUM_TOKENS)),
        max_seq_len=single("--max_seq_len", None),
        chunked_prefill="--enable_chunked_prefill" in words,
        cuda_graph="--use_cuda_graph" in words,
        capture_batches=tuple(int(v) for v in _flag_values(words, "--cuda_graph_batch_sizes")),
    )


@dataclass(frozen=True)
class EnvelopeSpec:
    """``accuracy.envelope``: how the envelope boot drives the target."""

    enabled: bool = True
    max_tokens: int = DEFAULT_MAX_TOKENS
    agree_tokens: int = DEFAULT_AGREE_TOKENS
    long_prompts: int = DEFAULT_LONG_PROMPTS
    probes: tuple[str, ...] = DEFAULT_PROBES
    timeout: int = 2 * BOOT_TIMEOUT_S

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any]) -> "EnvelopeSpec":
        block = dict((task_data.get("accuracy") or {}).get("envelope") or {})
        default = cls()
        return cls(
            enabled=bool(block.get("enabled", default.enabled)),
            max_tokens=int(block.get("max_tokens", default.max_tokens)),
            agree_tokens=int(block.get("agree_tokens", default.agree_tokens)),
            long_prompts=int(block.get("long_prompts", default.long_prompts)),
            probes=tuple(str(p) for p in block.get("probes") or default.probes),
            timeout=int(block.get("timeout", default.timeout)),
        )


def envelope_requests(envelope: Envelope, spec: EnvelopeSpec) -> list[dict[str, Any]]:
    """The envelope leg's requests: long prompts, then probes, then staggered fillers."""
    requests: list[dict[str, Any]] = []
    length = envelope.long_prompt_tokens(spec.max_tokens)
    for n in range(spec.long_prompts if length else 0):
        ids = [_LONG_PROMPT_IDS[(n + i) % len(_LONG_PROMPT_IDS)] for i in range(length)]
        requests.append({"id": f"long{n}", "token_ids": ids, "max_tokens": spec.max_tokens})
    for n, prompt in enumerate(spec.probes):
        requests.append({"id": f"probe{n}", "prompt": prompt, "max_tokens": spec.max_tokens})
    fillers = max(envelope.max_batch_size - len(requests), 0)
    for n in range(fillers):
        requests.append(
            {
                "id": f"fill{n}",
                "prompt": spec.probes[n % len(spec.probes)],
                # Staggered, so the live batch drains through every size.
                "max_tokens": 1 + n % spec.max_tokens,
            }
        )
    return requests


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run_envelope_boot(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    features: Any | None = None,
    unit: str | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
    spec: EnvelopeSpec | None = None,
) -> GateResult:
    """Serve the target at its declared limits and compare probes with a batch-1 run."""
    spec = spec or EnvelopeSpec.from_task(task_data)
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=GATE_NAME, member=member, passed=False, detail=state.block_detail())

    argv = quickstart_words(
        task_data,
        environment,
        unit=unit,
        features=task_data.get("features") if features is None else features,
        max_tokens=spec.max_tokens,
        member=member,
    )
    envelope = resolve_envelope(argv)
    if envelope.max_seq_len is None:
        return _verdict(
            name=GATE_NAME,
            member=member,
            passed=False,
            detail="the runtime block declares no --max_seq_len, so the envelope is the "
            "checkpoint's own maximum and no gate can reach it; declare it in "
            "runtime.extra_flags",
        )

    directory = gate_workdir(workdir, GATE_NAME, task_data, environment=environment)
    tag = f"{unit or 'unit'}.{member or 'member'}"
    env = with_kv_cache_config(_with_mode(environment, MODE_REQUIRE), task_data, unit)
    env = replace(env, env={**env.env, **pin_tuning(directory / tag)})

    probes = [
        {"id": f"probe{n}", "prompt": p, "max_tokens": spec.max_tokens}
        for n, p in enumerate(spec.probes)
    ]
    loaded = envelope_requests(envelope, spec)
    legs = {
        "reference": (probes, 1),
        "envelope": (loaded, max(envelope.max_batch_size, 1)),
    }
    outputs: dict[str, list[dict[str, Any]]] = {}
    commands: list[str] = []
    attested: list[EngineAttestation] = []
    declared = engine_declaration(task_data, unit, argv, mode=MODE_REQUIRE)
    started = time.monotonic()
    tail = ""
    for leg, (requests, concurrency) in legs.items():
        path = write_driver_spec(
            directory / f"envelope.{tag}.{leg}.json",
            task_data=task_data,
            argv=argv,
            requests=requests,
            max_tokens=spec.max_tokens,
            concurrency=concurrency,
            reuse_probe=True,
        )
        command = engine_driver_command(env, path)
        commands.append(command)
        result = runner(command, None, spec.timeout)
        combined = result.stdout + result.stderr
        tail = "\n".join(combined.splitlines()[-40:])
        ran = dict(
            name=GATE_NAME,
            member=member,
            command=" && ".join(commands),
            duration_s=time.monotonic() - started,
            stdout_tail=tail,
        )
        if result.infrastructure:
            return _verdict(
                passed=False,
                detail=f"did not run to a verdict: {result.infrastructure}",
                infrastructure=result.infrastructure,
                **ran,
            )
        if result.timed_out:
            return _verdict(
                passed=False, detail=f"the {leg} leg timed out after {spec.timeout}s", **ran
            )
        if not result.ok:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg exited {result.returncode} at {envelope.describe()}",
                **ran,
            )
        attested.append(attest_engine(combined, declared, leg=leg))
        ran.update(engine_fields(attested, task_data, unit))
        reuse_off = reuse_disabled_detail(
            combined, options=resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
        )
        if reuse_off:
            return _verdict(passed=False, detail=f"the {leg} leg: {reuse_off}", **ran)
        shortfall = reuse_shortfall(attested[-1], asks=bool(declared.get("block_reuse")))
        if shortfall:
            return _verdict(passed=False, detail=shortfall, **ran)
        if attested[-1].mismatches:
            return _verdict(passed=False, detail=attested[-1].difference(), **ran)
        records = {str(r.get("id")): r for r in parse_driver_records(combined)}
        missing = [str(r["id"]) for r in requests if str(r["id"]) not in records]
        if missing:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg returned no record for {len(missing)} of "
                f"{len(requests)} request(s) ({', '.join(missing[:5])})",
                **ran,
            )
        outputs[leg] = [records[str(r["id"])] for r in requests]

    ran = dict(
        name=GATE_NAME,
        member=member,
        command=" && ".join(commands),
        duration_s=time.monotonic() - started,
        stdout_tail=tail,
        **engine_fields(attested, task_data, unit),
    )
    target_length = envelope.long_prompt_tokens(spec.max_tokens)
    for record in outputs["envelope"]:
        if not str(record.get("id", "")).startswith("long"):
            continue
        if record.get("prompt_tokens") not in (None, target_length):
            return _verdict(
                passed=False,
                detail=f"{record['id']} reached {record.get('prompt_tokens')} prompt tokens, "
                f"not the {target_length} the envelope needs",
                **ran,
            )
        if not record.get("token_ids"):
            return _verdict(
                passed=False, detail=f"{record['id']} generated nothing at the limit", **ran
            )

    loaded_probes = {str(r["id"]): r for r in outputs["envelope"]}
    disagree = []
    for reference in outputs["reference"]:
        alone = [int(t) for t in reference.get("token_ids") or ()]
        under_load = [int(t) for t in loaded_probes[str(reference["id"])].get("token_ids") or ()]
        need = min(spec.agree_tokens, len(alone), len(under_load))
        if common_prefix(alone, under_load) < need or not alone:
            disagree.append(
                f"{reference['id']}: alone {alone[: spec.agree_tokens]} vs under load "
                f"{under_load[: spec.agree_tokens]}"
            )
    if disagree:
        return _verdict(
            passed=False,
            detail=f"{len(disagree)} probe(s) changed under load at {envelope.describe()}: "
            + "; ".join(disagree),
            **ran,
        )
    longs = sum(1 for r in loaded if str(r["id"]).startswith("long"))
    return _verdict(
        passed=True,
        detail=f"served {len(loaded)} requests at {envelope.describe()}, "
        f"{longs} of {target_length} prompt tokens; {len(probes)} probe(s) agree with "
        f"batch 1 on their first {spec.agree_tokens} tokens",
        **ran,
    )


__all__ = [
    "Envelope",
    "EnvelopeSpec",
    "GATE_NAME",
    "common_prefix",
    "envelope_requests",
    "resolve_envelope",
    "run_envelope_boot",
]
