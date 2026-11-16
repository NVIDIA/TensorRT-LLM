# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployments: what a run brings up, stated as TensorRT-LLM itself takes it.

A deployment is one served configuration of the checkpoint: the options
``trtllm-serve --config`` takes. The first deployment defines ``onboarded``. A
disaggregated deployment is context and generation servers (``ctx``, ``gen``), each role
with its own options and its own count of ``instances``::

    deployments:
      - name: dep16
        options: engines/dep16.yaml      # a native options file, or the options as a mapping
        timeout: 5400
        probes: [reuse, chunk]           # what its session also proves; see PROBES
        benchmarks:
          - name: gsm8k                  # judged against the reference row of accuracy.model_key
          - {name: aa_lcr, anchor: 0.62, source: "<where it was measured>", unit: fraction}
          - {name: mmlu, through: eval}  # trtllm-eval in a load of its own
      - name: pd
        env: {UCX_TLS: "^ib"}
        ctx: {options: engines/ctx.yaml}
        gen: {options: engines/gen.yaml, instances: 2}
        relative_to: dep16               # judged against dep16's own score where no row fits

No deployment states how it is launched: its world size is what the engine enforces for
its options (:func:`ranks_of`), and the pool places it. The spec's top-level ``accuracy``
block gives every benchmark its defaults (the reference rows' model, file and tags, the
dataset's place, the sample count, how a score is read); a ``benchmarks`` entry overrides
them for itself. A deployment that names no ``benchmarks`` takes ``accuracy.benchmark``
(``gsm8k`` when the block names the reference rows' ``model_key`` and no benchmark), and
none when the spec says neither: it is then judged by its session's other checks alone.

:func:`resolve` is the one way anything below the spec loader learns what the run deploys.
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .schema_keys import reject_unknown_keys

DEPLOYMENTS_FIELD = "deployments"
ACCURACY_FIELD = "accuracy"
AGGREGATED = "aggregated"
DISAGGREGATED = "disaggregated"
#: The role of an aggregated deployment's one server.
SERVER = "server"
#: The roles of a disaggregated deployment, in the order their servers start.
ROLES: tuple[str, ...] = ("ctx", "gen")
#: What a deployment may be sent: every deployment takes text.
INPUTS: tuple[str, ...] = ("text", "image")
#: What a session proves beside its boot, when the deployment asks. ``reuse``: a prompt
#: sent again resumes from the cache. ``chunk``: a prompt past ``max_num_tokens`` is
#: served. An image prompt follows ``inputs``. Unstated, they follow the options that
#: turn them on.
PROBES: tuple[str, ...] = ("reuse", "chunk")
#: Probes that change the engine's state (its cache): never beside scoring.
MUTATING_PROBES: tuple[str, ...] = ("reuse",)
#: The option each side of a disaggregated deployment must state: what moves the cache
#: from ``ctx`` to ``gen``. ``DEFAULT`` leaves the backend to the engine.
TRANSCEIVER = "cache_transceiver_config"
DEFAULT_BENCHMARK = "gsm8k"
#: How a benchmark is scored: through the session's endpoint, or by ``trtllm-eval`` in a
#: load of its own.
THROUGH_ENDPOINT = "endpoint"
THROUGH_EVAL = "eval"
THROUGH: tuple[str, ...] = (THROUGH_ENDPOINT, THROUGH_EVAL)

#: What one role of a disaggregated deployment states.
ROLE_KEYS: tuple[str, ...] = ("options", "instances")
DEPLOYMENT_KEYS: tuple[str, ...] = (
    "name",
    "options",
    "env",
    "timeout",
    "relative_to",
    "tests",
    "inputs",
    "probes",
    "benchmarks",
    *ROLES,
)
#: What the spec's ``accuracy`` block gives every benchmark and a ``benchmarks`` entry may
#: state for itself: the reference rows' model, file and tags, where the dataset is, the
#: sample count behind the standard error, and how the score is read from the output.
BENCHMARK_DEFAULTS: tuple[str, ...] = (
    "model_key",
    "reference_yaml",
    "discriminators",
    "dataset_path",
    "samples",
    "score_pattern",
)
BENCHMARK_KEYS = (
    "name",
    "args",
    "anchor",
    "source",
    "unit",
    "env",
    "through",
    *BENCHMARK_DEFAULTS,
)
#: The scales a benchmark's scores and anchor may be on.
UNITS: tuple[str, ...] = ("percent", "fraction")

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
#: Names a per-deployment entry of an ``accuracy`` block could be mistaken for.
RESERVED_NAMES: tuple[str, ...] = (
    "default",
    "prompts",
    "prompts_file",
    "keywords",
    "max_tokens",
    "timeout",
    "regression",
    "dataset",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: The spec keys that said what runs before ``deployments`` did, and what replaced each.
REPLACED_KEYS: dict[str, str] = {
    "parallel": "each deployment's options state its parallel sizes; `target.parallel` names "
    "the target's directory",
    "members": "one deployment per configuration, each with its own options file",
    "features": "a deployment whose options turn the feature on (a drafter in "
    "`speculative_config`, `inputs: [text, image]`, `ctx` and `gen` for disaggregation)",
    "runtime": "the switches are native options of each deployment's options file",
    "variants": "a deployment of its own, with `relative_to` naming the one it is judged against",
    "profile": "a deployment owes what it states: its benchmarks, probes and tests",
    "coverage": "a configuration the run does not owe is a deployment the spec does not list",
    "scope": "a deployment takes images when its `inputs` say so",
}
#: The ``accuracy`` keys the older forms read, and what replaced each.
REPLACED_ACCURACY_KEYS: dict[str, str] = {
    "llm_api_options": "each deployment's `options`",
    "benchmark_args": "a benchmark's own `args`",
    "benchmark_members": "the deployments that list the benchmark",
    "anchor": "a benchmark's `anchor` with its `source`, or the deployment's `relative_to`",
    "disagg": "a deployment with `ctx` and `gen`; a native test it names goes in its `tests`",
    "serve": "a deployment's session sends the API prompts itself",
    "max_seq_len": "`max_seq_len` in the deployment's options",
    "max_batch_size": "`max_batch_size` in the deployment's options",
    "envelope": "nothing: the boundary phases left the flow with the `envelope` probe",
    "reference_logits": "nothing: a benchmark against its anchor is the one judge of the model",
}


#: The engine option that states the Modeling V2 requirement, and the flow's default
#: for it: a server that fell back to the built-in model is not what an onboarding measures.
REQUIREMENT_OPTION = "modeling_v2"
REQUIRE = "require"


def requirement(options: Mapping[str, Any] | None) -> str:
    """The Modeling V2 requirement a server is held to: what its options state, else ``require``.

    A spec that states another value (``off`` for a control on the built-in path,
    ``auto``) is used as stated, in the options the server reads, its launch and its
    attestation: a requirement that was mistaken may be changed, and every line judged
    under it says so (:func:`requirement_said`). YAML reads a bare ``off`` as false.
    """
    stated = (options or {}).get(REQUIREMENT_OPTION)
    if stated is None:
        return REQUIRE
    return "off" if stated is False else str(stated)


def requirement_said(deployment: "Deployment") -> str:
    """How a line says the requirement ``deployment`` was judged under; empty for the default."""
    stated = {role: requirement(server.options) for role, server in deployment.servers.items()}
    if set(stated.values()) == {REQUIRE}:
        return ""
    if len(set(stated.values())) == 1:
        what = f"`{REQUIREMENT_OPTION}: {next(iter(stated.values()))}`"
    else:
        what = ", ".join(
            f"`{REQUIREMENT_OPTION}: {mode}` for `{role}`" for role, mode in stated.items()
        )
    return f"judged under {what}, as the spec states it (the flow's default is `{REQUIRE}`)"


def ranks_of(options: Mapping[str, Any]) -> int:
    """The ranks an engine of ``options`` runs: its tensor, pipeline and context parallel sizes, multiplied.

    What the engine itself enforces as its world size. Expert parallelism shares those
    ranks and never adds one.
    """
    sizes = ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size")
    return math.prod(max(1, int(options.get(size) or 1)) for size in sizes)


@dataclass(frozen=True)
class Benchmark:
    """One benchmark a deployment is scored on, with everything its judging reads.

    ``name`` is a ``trtllm-eval`` subcommand and ``args`` its own options. ``anchor``,
    with its ``source``, is a declared score on the scale the benchmark reports; without
    one the reference row of ``model_key`` in ``reference_yaml`` (the tree's file for the
    benchmark, unsaid) whose tags are ``discriminators`` is the anchor, where one fits.
    ``discriminators`` unsaid (``None``) are the tags the deployment's options imply.
    """

    name: str
    args: tuple[str, ...] = ()
    anchor: float | None = None
    source: str = ""
    #: The scale its scores and its anchor are on: stated, never read off a score's size.
    unit: str = "percent"
    env: Mapping[str, str] = field(default_factory=dict)
    #: ``endpoint``: scored in the session's load. ``eval``: ``trtllm-eval``, its own load.
    through: str = THROUGH_ENDPOINT
    model_key: str | None = None
    reference_yaml: str | None = None
    discriminators: Mapping[str, Any] | None = None
    dataset_path: str | None = None
    samples: int | None = None
    score_pattern: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.args:
            out["args"] = list(self.args)
        if self.anchor is not None:
            out.update(anchor=self.anchor, source=self.source)
        if self.unit != UNITS[0]:
            out["unit"] = self.unit
        if self.env:
            out["env"] = dict(self.env)
        if self.through != THROUGH_ENDPOINT:
            out["through"] = self.through
        for key in BENCHMARK_DEFAULTS:
            value = getattr(self, key)
            if value is not None:
                out[key] = dict(value) if isinstance(value, Mapping) else value
        return out


@dataclass(frozen=True)
class Server:
    """One role of a deployment: the native options of its engine, and how many of it run."""

    options: Mapping[str, Any]
    #: The file the options were read from; empty when the spec states them inline.
    options_path: str = ""
    #: How many servers of this role the deployment runs (one for an aggregated deployment).
    instances: int = 1

    @property
    def ranks(self) -> int:
        """The devices one server of this role takes: one for each rank of its engine."""
        return ranks_of(self.options)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"options": self.options_path or dict(self.options)}
        if self.instances != 1:
            out["instances"] = self.instances
        return out


@dataclass(frozen=True)
class Instance:
    """One server a session starts: a role, which of its instances, and its engine."""

    role: str
    index: int
    server: Server

    @property
    def name(self) -> str:
        """``server`` for an aggregated deployment's; ``ctx0``, ``gen1`` for a role's."""
        return SERVER if self.role == SERVER else f"{self.role}{self.index}"


@dataclass(frozen=True)
class Deployment:
    """One deployment of the spec, in the form every check and the run read."""

    name: str
    #: The first deployment: its session is what ``onboarded`` means.
    first: bool = False
    #: The one server of an aggregated deployment; ``None`` for a disaggregated one.
    server: Server | None = None
    ctx: Server | None = None
    gen: Server | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: int | None = None
    #: The deployment whose own score this one is judged against, where no reference row fits.
    relative_to: str | None = None
    #: Native pytest node ids that also judge it (a transfer test, a feature's own test).
    tests: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ("text",)
    #: The probes its session runs; ``None``: those its options turn on (:meth:`asked`).
    probes: tuple[str, ...] | None = None
    benchmarks: tuple[Benchmark, ...] = ()

    @property
    def kind(self) -> str:
        return AGGREGATED if self.server is not None else DISAGGREGATED

    @property
    def servers(self) -> dict[str, Server]:
        """Its roles: ``{"server": ...}``, or ``{"ctx": ..., "gen": ...}``."""
        if self.server is not None:
            return {SERVER: self.server}
        return {role: getattr(self, role) for role in ROLES}

    def instances(self) -> Iterator[Instance]:
        """Every server its session starts, in the order they start."""
        for role, server in self.servers.items():
            for index in range(server.instances):
                yield Instance(role, index, server)

    @property
    def ranks(self) -> int:
        """The devices one session of it holds: every instance of every role, together."""
        return sum(instance.server.ranks for instance in self.instances())

    @property
    def shape(self) -> str:
        """Its one-word shape, with the devices it holds: ``tp2``, ``tep4``, ``dep16``, ``pd3``.

        The label the run's records keep a deployment's checks under. ``tp``: tensor
        parallel; ``tep``: with expert parallel; ``dep``: attention data parallel as
        well; ``pd``: a disaggregated deployment, every role instance counted.
        """
        if self.server is None:
            return f"pd{self.ranks}"
        options = self.server.options
        if options.get("enable_attention_dp"):
            return f"dep{self.ranks}"
        return f"{'tep' if int(options.get('moe_expert_parallel_size') or 1) > 1 else 'tp'}{self.ranks}"

    def asked(self) -> tuple[str, ...]:
        """The probes its session runs: those it states, else those its options turn on."""
        if self.probes is not None:
            return self.probes
        found = []
        every = [server.options for server in self.servers.values()]
        if any((o.get("kv_cache_config") or {}).get("enable_block_reuse") for o in every):
            found.append("reuse")
        if any(o.get("enable_chunked_prefill") for o in every):
            found.append("chunk")
        return tuple(found)

    @property
    def drafts(self) -> bool:
        """Whether its generating engine drafts tokens a target verifies."""
        options = (self.gen or self.server).options
        return bool((options.get("speculative_config") or {}).get("decoding_type"))

    def benchmark(self, name: str) -> Benchmark | None:
        return next((b for b in self.benchmarks if b.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        """The deployment as a spec states one, with only what it sets."""
        out: dict[str, Any] = {"name": self.name}
        if self.server is not None:
            out.update(self.server.as_dict())
        else:
            out.update({role: server.as_dict() for role, server in self.servers.items()})
        for key in ("env", "timeout", "relative_to"):
            value = getattr(self, key)
            if value:
                out[key] = dict(value) if isinstance(value, Mapping) else value
        if self.tests:
            out["tests"] = list(self.tests)
        if self.inputs != ("text",):
            out["inputs"] = list(self.inputs)
        if self.probes is not None:
            out["probes"] = list(self.probes)
        if self.benchmarks:
            out["benchmarks"] = [benchmark.as_dict() for benchmark in self.benchmarks]
        return out


# --------------------------------------------------------------------- a stated block


def load_options(source: Any) -> dict[str, Any]:
    """Native options from a path or an inline mapping. Raises ``ValueError`` saying what is wrong."""
    if isinstance(source, Mapping):
        return dict(source)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("must be the path of an options file, or the options as a mapping")
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"does not exist: {source}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot be read as YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"must hold a mapping of LLM-API options: {source}")
    return dict(loaded)


def resolve_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make each options path written beside the spec absolute, in place."""
    block = data.get(DEPLOYMENTS_FIELD)
    for entry in block if isinstance(block, list) else ():
        if not isinstance(entry, dict):
            continue
        for holder in (entry, *(entry.get(role) for role in ROLES)):
            if isinstance(holder, dict) and isinstance(holder.get("options"), str):
                path = Path(holder["options"])
                if holder["options"].strip() and not path.is_absolute():
                    holder["options"] = str((spec_dir / path).resolve())


def split_args(value: Any) -> tuple[str, ...]:
    """A benchmark's options: a list of strings, or one string split as a shell would.

    Raises ``ValueError`` for a string a shell could not split (an unclosed quote).
    """
    if isinstance(value, str):
        return tuple(shlex.split(value))
    return tuple(str(arg) for arg in value or ())


def validate_env(env: Any, errors: list[str], where: str) -> None:
    if env is None:
        return
    if not isinstance(env, Mapping):
        errors.append(f"'{where}' must be a mapping from variable to value")
        return
    for name, value in env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            errors.append(f"'{where}' names {name!r}, which is not a shell variable name")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            errors.append(f"'{where}.{name}' must be a scalar the shell can carry")


def validate_judging(block: Mapping[str, Any], errors: list[str], where: str) -> None:
    """The keys of :data:`BENCHMARK_DEFAULTS`, in ``accuracy`` or in a ``benchmarks`` entry."""
    for key in ("model_key", "reference_yaml", "dataset_path"):
        value = block.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"'{where}.{key}' must be a non-empty string")
    tags = block.get("discriminators")
    if tags is not None and not (
        isinstance(tags, Mapping)
        and all(isinstance(v, (str, int, float, bool)) for v in tags.values())
    ):
        errors.append(f"'{where}.discriminators' must map a reference row's tag to a scalar")
    samples = block.get("samples")
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int) or samples < 1
    ):
        errors.append(f"'{where}.samples' must be an integer >= 1")
    pattern = block.get("score_pattern")
    if pattern is not None:
        try:
            groups = re.compile(pattern).groups if isinstance(pattern, str) else 0
        except re.error as exc:
            errors.append(f"'{where}.score_pattern' is not a regular expression: {exc}")
        else:
            if groups < 1:
                errors.append(
                    f"'{where}.score_pattern' needs a group around the score, "
                    f"e.g. `accuracy: ([0-9.]+)`"
                )


def transceiver_problem(options: Mapping[str, Any]) -> str | None:
    """Why a disaggregated side with ``options`` cannot answer; ``None`` when it can."""
    block = options.get(TRANSCEIVER)
    if isinstance(block, Mapping) and block.get("backend"):
        return None
    return (
        f"options {TRANSCEIVER}: a disaggregated server refuses its first request without a "
        f"backend to move the cache over; fix: `{TRANSCEIVER}: {{backend: DEFAULT}}`"
    )


def _validate_options(
    holder: Mapping[str, Any], errors: list[str], where: str, side: bool = False
) -> None:
    """``options`` (required) of an aggregated deployment or of one role: they must resolve."""
    if holder.get("options") is None:
        errors.append(
            f"missing {where}.options: a native LLM-API options file; "
            f"fix: `options: engines/<name>.yaml`"
        )
        return
    try:
        options = load_options(holder["options"])
        ranks_of(options)
    except (TypeError, ValueError) as exc:
        errors.append(f"'{where}.options' {exc}")
        return
    if side and transceiver_problem(options):
        errors.append(f"missing {where} {transceiver_problem(options)}")


def _validate_benchmarks(block: Any, errors: list[str], where: str) -> None:
    if block is None:
        return
    if not isinstance(block, list):
        errors.append(f"'{where}' must be a list of benchmarks, each a mapping with a `name`")
        return
    seen: set[str] = set()
    for index, entry in enumerate(block):
        at = f"{where}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"'{at}' must be a mapping with a `name`")
            continue
        reject_unknown_keys(entry, BENCHMARK_KEYS, errors, where=at)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"'{at}.name' must be a benchmark `trtllm-eval` has, e.g. gsm8k")
        elif name in seen:
            errors.append(f"'{at}' repeats the benchmark {name!r}")
        else:
            seen.add(name)
        args = entry.get("args")
        if args is not None and not (
            isinstance(args, str)
            or (isinstance(args, list) and all(isinstance(a, str) for a in args))
        ):
            errors.append(f"'{at}.args' must be a string or a list of strings")
        elif isinstance(args, str):
            try:
                split_args(args)
            except ValueError as exc:
                errors.append(f"'{at}.args' cannot be split as a shell would: {exc}")
        anchor, source = entry.get("anchor"), entry.get("source")
        if anchor is not None:
            number = isinstance(anchor, (int, float)) and not isinstance(anchor, bool)
            if not number or not math.isfinite(anchor) or anchor < 0:
                errors.append(
                    f"'{at}.anchor' must be a finite score >= 0, on the benchmark's own scale"
                )
            if not isinstance(source, str) or not source.strip():
                errors.append(
                    f"'{at}.source' is required with an anchor: where and how it was measured"
                )
        elif source is not None:
            errors.append(f"'{at}.source' says where an `anchor` comes from, and there is none")
        if entry.get("unit") is not None and entry["unit"] not in UNITS:
            errors.append(f"'{at}.unit' is the scale of its scores: one of {list(UNITS)}")
        if entry.get("through") is not None and entry["through"] not in THROUGH:
            errors.append(f"'{at}.through' is how it is scored: one of {list(THROUGH)}")
        validate_env(entry.get("env"), errors, f"{at}.env")
        validate_judging(entry, errors, at)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def validate(block: Any, errors: list[str]) -> None:
    """Every problem with a spec's ``deployments`` block, batched into ``errors``."""
    where = DEPLOYMENTS_FIELD
    if not isinstance(block, list) or not block:
        errors.append(f"'{where}' must be a non-empty list; its first entry defines `onboarded`")
        return
    names: list[str] = []
    for index, entry in enumerate(block):
        at = f"{where}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"'{at}' must be a mapping")
            continue
        reject_unknown_keys(entry, DEPLOYMENT_KEYS, errors, where=at)
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            errors.append(f"'{at}.name' must be lowercase letters, digits and '_', e.g. dep16")
        else:
            at = f"{where}.{name}"
            if name in names:
                errors.append(f"'{at}' is named twice; each deployment needs its own name")
            if name in RESERVED_NAMES:
                errors.append(
                    f"'{at}': `{name}` is a key of the `accuracy` block; name the deployment "
                    f"otherwise"
                )
        sides = [role for role in ROLES if entry.get(role) is not None]
        if sides:
            if len(sides) != len(ROLES):
                errors.append(f"'{at}' has `{sides[0]}` alone; disaggregated needs `ctx` and `gen`")
            if entry.get("options") is not None:
                errors.append(
                    f"'{at}.options' sits beside `ctx` and `gen`: each role states its own"
                )
            for role in sides:
                if not isinstance(entry[role], Mapping):
                    errors.append(f"'{at}.{role}' must be a mapping with `options`")
                    continue
                reject_unknown_keys(entry[role], ROLE_KEYS, errors, where=f"{at}.{role}")
                _validate_options(entry[role], errors, f"{at}.{role}", side=True)
                count = entry[role].get("instances")
                if count is not None and not _positive_int(count):
                    errors.append(f"'{at}.{role}.instances' must be an integer >= 1")
        else:
            _validate_options(entry, errors, at)
        validate_env(entry.get("env"), errors, f"{at}.env")
        timeout = entry.get("timeout")
        if timeout is not None and not _positive_int(timeout):
            errors.append(f"'{at}.timeout' must be an integer number of seconds >= 1")
        relative_to = entry.get("relative_to")
        if relative_to is not None and relative_to not in names:
            # The second run of a model meets this: the deployment it was judged against
            # was the first run's. So the message says what to state instead.
            errors.append(
                f"'{at}.relative_to' is {relative_to!r}, which names no deployment before "
                "this one. A deployment of the run this spec builds on (`onboarded: <ref>`) "
                "is not measured here: state the score that run measured as the benchmark's "
                "`anchor`, with its `source` (`benchmarks: [{name: <benchmark>, anchor: "
                '<the score>, source: "<that run\'s report>"}]`)'
            )
        tests = entry.get("tests")
        if tests is not None and not (
            isinstance(tests, list)
            and all(isinstance(t, str) and "::" in t and not Path(t).is_absolute() for t in tests)
        ):
            errors.append(
                f"'{at}.tests' must be a list of repository-relative pytest node ids "
                f"(`tests/.../test_x.py::TestY::test_z`)"
            )
        inputs = entry.get("inputs")
        if inputs is not None and not (
            isinstance(inputs, list)
            and all(isinstance(i, str) and i in INPUTS for i in inputs)
            and "text" in inputs
            and len(set(inputs)) == len(inputs)
        ):
            errors.append(f"'{at}.inputs' must be `text` and any of {list(INPUTS[1:])}, each once")
        probes = entry.get("probes")
        if probes is not None and not (
            isinstance(probes, list)
            and all(isinstance(p, str) and p in PROBES for p in probes)
            and len(set(probes)) == len(probes)
        ):
            errors.append(f"'{at}.probes' must be a list of {list(PROBES)}, each once")
        _validate_benchmarks(entry.get("benchmarks"), errors, f"{at}.benchmarks")
        if isinstance(name, str):
            names.append(name)


def benchmark_of(entry: Mapping[str, Any], defaults: Mapping[str, Any]) -> Benchmark:
    """One benchmark from a ``benchmarks`` entry, over the ``accuracy`` block's defaults."""

    def stated(key: str) -> Any:
        return entry[key] if entry.get(key) is not None else defaults.get(key)

    tags = stated("discriminators")
    return Benchmark(
        name=str(entry["name"]),
        args=split_args(entry.get("args")),
        anchor=float(entry["anchor"]) if entry.get("anchor") is not None else None,
        source=str(entry.get("source") or ""),
        unit=str(entry.get("unit") or UNITS[0]),
        env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
        through=str(entry.get("through") or THROUGH_ENDPOINT),
        model_key=stated("model_key"),
        reference_yaml=stated("reference_yaml"),
        discriminators=dict(tags) if isinstance(tags, Mapping) else None,
        dataset_path=stated("dataset_path"),
        samples=stated("samples"),
        score_pattern=stated("score_pattern"),
    )


def _server(holder: Mapping[str, Any]) -> Server:
    source = holder.get("options")
    return Server(
        options=load_options(source),
        options_path=source if isinstance(source, str) else "",
        instances=int(holder.get("instances") or 1),
    )


def resolve(data: Mapping[str, Any]) -> tuple[Deployment, ...]:
    """The deployments of a validated spec, the first one first."""
    accuracy = data.get(ACCURACY_FIELD)
    defaults = accuracy if isinstance(accuracy, Mapping) else {}
    judged = bool(defaults.get("benchmark") or defaults.get("model_key"))
    default = [{"name": str(defaults.get("benchmark") or DEFAULT_BENCHMARK)}] if judged else []
    out = []
    for index, entry in enumerate(data.get(DEPLOYMENTS_FIELD) or ()):
        disaggregated = entry.get(ROLES[0]) is not None
        listed = entry.get("benchmarks")
        probes = entry.get("probes")
        out.append(
            Deployment(
                name=str(entry["name"]),
                first=index == 0,
                server=None if disaggregated else _server(entry),
                ctx=_server(entry["ctx"]) if disaggregated else None,
                gen=_server(entry["gen"]) if disaggregated else None,
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                timeout=entry.get("timeout"),
                relative_to=entry.get("relative_to"),
                tests=tuple(entry.get("tests") or ()),
                inputs=tuple(entry.get("inputs") or ("text",)),
                probes=None if probes is None else tuple(probes),
                benchmarks=tuple(
                    benchmark_of(b, defaults) for b in (default if listed is None else listed)
                ),
            )
        )
    return tuple(out)


def named(data: Mapping[str, Any], name: str | None = None) -> Deployment:
    """The deployment called ``name``; the first when unnamed. ``KeyError`` naming the ones there are."""
    found = resolve(data)
    if not found:
        raise KeyError("this spec states no deployment")
    if name in (None, ""):
        return found[0]
    for deployment in found:
        if deployment.name == name:
            return deployment
    raise KeyError(
        f"{name!r} names no deployment of this spec ({', '.join(d.name for d in found)})"
    )


def describe(found: tuple[Deployment, ...], anchors: Mapping[str, str] | None = None) -> str:
    """One line for each deployment: its name, kind, servers, and what judges it.

    ``anchors`` says where each benchmark's anchor comes from, by ``<deployment>:<benchmark>``.
    """
    lines = []
    for deployment in found:
        shapes = []
        for role, server in deployment.servers.items():
            shape = f"{server.ranks} rank(s)"
            if server.options.get("enable_attention_dp"):
                shape += ", attention DP"
            if server.instances > 1:
                shape = f"{server.instances} x {shape}"
            shapes.append(shape if role == SERVER else f"{role} {shape}")
        judged = [f"probe {probe}" for probe in deployment.asked()]
        for benchmark in deployment.benchmarks:
            source = (anchors or {}).get(f"{deployment.name}:{benchmark.name}")
            how = "" if benchmark.through == THROUGH_ENDPOINT else ", trtllm-eval in its own load"
            judged.append(f"{benchmark.name}{how}" + (f" ({source})" if source else ""))
        judged += [f"test {test}" for test in deployment.tests]
        lines.append(
            f"- `{deployment.name}`{' (first)' if deployment.first else ''}: {deployment.kind}, "
            f"{'; '.join(shapes)}; one session: attestation, boot and API prompts"
            + (f", {', '.join(judged)}" if judged else "")
        )
    return "\n".join(lines)


__all__ = [
    "AGGREGATED",
    "BENCHMARK_DEFAULTS",
    "DEPLOYMENTS_FIELD",
    "DISAGGREGATED",
    "MUTATING_PROBES",
    "PROBES",
    "REQUIRE",
    "REQUIREMENT_OPTION",
    "REPLACED_ACCURACY_KEYS",
    "REPLACED_KEYS",
    "ROLES",
    "SERVER",
    "THROUGH_ENDPOINT",
    "THROUGH_EVAL",
    "Benchmark",
    "Deployment",
    "Instance",
    "Server",
    "benchmark_of",
    "describe",
    "load_options",
    "named",
    "ranks_of",
    "requirement",
    "requirement_said",
    "resolve",
    "resolve_paths",
    "split_args",
    "transceiver_problem",
    "validate",
    "validate_env",
    "validate_judging",
]
