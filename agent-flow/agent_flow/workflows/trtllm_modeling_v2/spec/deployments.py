# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployments: what a run brings up, stated as TensorRT-LLM itself takes it.

A deployment is one served configuration of the checkpoint: the options
``trtllm-serve --config`` takes, behind an optional launch prefix. The first
deployment is core's: it defines ``onboarded``. A disaggregated deployment is a
context and a generation server (``ctx``, ``gen``), each with its own options,
prefix and stop command::

    deployments:
      - name: dep16
        options: engines/dep16.yaml      # a native options file, or the options as a mapping
        launch: "<site prefix>"          # empty: this host
        stop: "<site command>"           # optional: ends what `launch` started
        timeout: 5400
        benchmarks:
          - name: gsm8k                  # judged against the reference row of accuracy.model_key
          - {name: aa_lcr, anchor: 0.62, source: "<where it was measured>"}
      - name: pd
        env: {UCX_TLS: "tcp,self,sm,cuda_copy,cuda_ipc"}
        ctx: {options: engines/ctx.yaml, launch: "<nodes 1 to 4>"}
        gen: {options: engines/gen.yaml, launch: "<nodes 5 to 8>"}
        relative_to: dep16               # judged against dep16's own score where no row fits

The spec's top-level ``accuracy`` block gives every benchmark its defaults (the
reference rows' model, file and tags, the dataset's place, the sample count, how a
score is read); a ``benchmarks`` entry overrides them for itself. A deployment that
names no ``benchmarks`` takes ``accuracy.benchmark`` (``gsm8k`` unsaid) when the spec
has an ``accuracy`` block, and none without one.

:func:`resolve` is the one way anything below the spec loader learns what the run
deploys: it returns these types whichever way the spec was written, the shorthand keys
included (``spec/shorthand.py``).
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .task_schema import ACCURACY_FIELD, reject_unknown_keys

DEPLOYMENTS_FIELD = "deployments"
AGGREGATED = "aggregated"
DISAGGREGATED = "disaggregated"
#: The servers of a disaggregated deployment, in the order they start.
ROLES: tuple[str, ...] = ("ctx", "gen")
#: What a deployment may be sent: every deployment takes text.
INPUTS: tuple[str, ...] = ("text", "image")
DEFAULT_BENCHMARK = "gsm8k"

#: What one server states: on an aggregated deployment itself, or inside ``ctx`` and ``gen``.
SERVER_KEYS: tuple[str, ...] = ("options", "launch", "stop")
DEPLOYMENT_KEYS: tuple[str, ...] = (
    "name",
    *SERVER_KEYS,
    "env",
    "timeout",
    "relative_to",
    "tests",
    "inputs",
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
BENCHMARK_KEYS: tuple[str, ...] = ("name", "args", "anchor", "source", "env", *BENCHMARK_DEFAULTS)
#: The ``accuracy`` keys a spec that states ``deployments`` keeps.
STATED_ACCURACY_KEYS: tuple[str, ...] = (
    "benchmark",
    *BENCHMARK_DEFAULTS,
    "boot",
    "serve",
    "timeout",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
    env: Mapping[str, str] = field(default_factory=dict)
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
        if self.env:
            out["env"] = dict(self.env)
        for key in BENCHMARK_DEFAULTS:
            value = getattr(self, key)
            if value is not None:
                out[key] = dict(value) if isinstance(value, Mapping) else value
        return out


@dataclass(frozen=True)
class Server:
    """One ``trtllm-serve``: its native options, and the prefix it starts behind."""

    options: Mapping[str, Any]
    #: The file the options were read from; empty when the spec states them inline or
    #: they were expanded from shorthand.
    options_path: str = ""
    #: The site's launch prefix; empty means the orchestrator's own host.
    launch: str = ""
    #: The site's command that ends what ``launch`` started; empty: a signal is enough.
    stop: str = ""

    @property
    def ranks(self) -> int:
        """How many devices the server takes: its options' tensor-parallel size."""
        return max(1, int(self.options.get("tensor_parallel_size") or 1))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"options": self.options_path or dict(self.options)}
        for key in ("launch", "stop"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


@dataclass(frozen=True)
class Deployment:
    """One deployment of the spec, in the form every check and the run read."""

    name: str
    #: The first deployment: its boot and accuracy are what ``onboarded`` means.
    core: bool = False
    #: The one server of an aggregated deployment; ``None`` for a disaggregated one.
    server: Server | None = None
    ctx: Server | None = None
    gen: Server | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: int | None = None
    #: The deployment whose own score this one is judged against, where no reference row fits.
    relative_to: str | None = None
    #: Native pytest node ids that also judge it (a drafter's acceptance, a disaggregated test).
    tests: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ("text",)
    benchmarks: tuple[Benchmark, ...] = ()

    @property
    def kind(self) -> str:
        return AGGREGATED if self.server is not None else DISAGGREGATED

    @property
    def servers(self) -> dict[str, Server]:
        """Its servers by role: ``{"server": ...}``, or ``{"ctx": ..., "gen": ...}``."""
        if self.server is not None:
            return {"server": self.server}
        return {role: getattr(self, role) for role in ROLES}

    @property
    def ranks(self) -> int:
        """The devices one load of it takes: every server's, together."""
        return sum(server.ranks for server in self.servers.values())

    @property
    def prefixed(self) -> bool:
        """Whether any of its servers starts behind a launch prefix, off this host."""
        return any(server.launch for server in self.servers.values())

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
        if self.benchmarks:
            out["benchmarks"] = [benchmark.as_dict() for benchmark in self.benchmarks]
        return out


def is_stated(data: Mapping[str, Any]) -> bool:
    """Whether the spec states its deployments itself, not through the shorthand keys."""
    return data.get(DEPLOYMENTS_FIELD) is not None


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


def _validate_server(holder: Mapping[str, Any], errors: list[str], where: str) -> None:
    """``options`` (required), ``launch`` and ``stop`` of an aggregated deployment or of one side."""
    if holder.get("options") is None:
        errors.append(
            f"missing {where}.options: a native LLM-API options file; "
            f"fix: `options: engines/<name>.yaml`"
        )
    else:
        try:
            load_options(holder["options"])
        except ValueError as exc:
            errors.append(f"'{where}.options' {exc}")
    launch = holder.get("launch")
    if launch is not None and not isinstance(launch, str):
        errors.append(f"'{where}.launch' must be a string: the site's launch prefix")
    stop = holder.get("stop")
    if stop is not None and (not isinstance(stop, str) or not stop.strip()):
        errors.append(f"'{where}.stop' must be a command, as a string")
    elif stop and not launch:
        errors.append(f"'{where}.stop' ends what `launch` started, and there is no `launch`")


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
        validate_env(entry.get("env"), errors, f"{at}.env")
        validate_judging(entry, errors, at)


def validate(block: Any, errors: list[str]) -> None:
    """Every problem with a spec's ``deployments`` block, batched into ``errors``."""
    where = DEPLOYMENTS_FIELD
    if not isinstance(block, list) or not block:
        errors.append(f"'{where}' must be a non-empty list; its first entry is core's")
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
            if name == "core" and index:
                errors.append(f"'{at}': `core` is how the first deployment is addressed")
        sides = [role for role in ROLES if entry.get(role) is not None]
        if sides:
            if len(sides) != len(ROLES):
                errors.append(f"'{at}' has `{sides[0]}` alone; disaggregated needs `ctx` and `gen`")
            for key in SERVER_KEYS:
                if entry.get(key) is not None:
                    errors.append(
                        f"'{at}.{key}' sits beside `ctx` and `gen`: each side states its own"
                    )
            for role in sides:
                if not isinstance(entry[role], Mapping):
                    errors.append(f"'{at}.{role}' must be a mapping with `options`")
                    continue
                reject_unknown_keys(entry[role], SERVER_KEYS, errors, where=f"{at}.{role}")
                _validate_server(entry[role], errors, f"{at}.{role}")
        else:
            _validate_server(entry, errors, at)
        validate_env(entry.get("env"), errors, f"{at}.env")
        timeout = entry.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1
        ):
            errors.append(f"'{at}.timeout' must be an integer number of seconds >= 1")
        relative_to = entry.get("relative_to")
        if relative_to is not None and relative_to not in names:
            errors.append(
                f"'{at}.relative_to' is {relative_to!r}, which names no deployment before this one"
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
        env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
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
        launch=str(holder.get("launch") or ""),
        stop=str(holder.get("stop") or ""),
    )


def _stated(data: Mapping[str, Any]) -> tuple[Deployment, ...]:
    """The deployments of a validated spec that states them."""
    accuracy = data.get(ACCURACY_FIELD)
    judged = isinstance(accuracy, Mapping)
    defaults = accuracy if judged else {}
    default = [{"name": str(defaults.get("benchmark") or DEFAULT_BENCHMARK)}] if judged else []
    out = []
    for index, entry in enumerate(data[DEPLOYMENTS_FIELD]):
        disaggregated = entry.get(ROLES[0]) is not None
        listed = entry.get("benchmarks")
        out.append(
            Deployment(
                name=str(entry["name"]),
                core=index == 0,
                server=None if disaggregated else _server(entry),
                ctx=_server(entry["ctx"]) if disaggregated else None,
                gen=_server(entry["gen"]) if disaggregated else None,
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                timeout=entry.get("timeout"),
                relative_to=entry.get("relative_to"),
                tests=tuple(entry.get("tests") or ()),
                inputs=tuple(entry.get("inputs") or ("text",)),
                benchmarks=tuple(
                    benchmark_of(b, defaults) for b in (default if listed is None else listed)
                ),
            )
        )
    return tuple(out)


def resolve(data: Mapping[str, Any]) -> tuple[Deployment, ...]:
    """The spec's deployments, core's first, whichever way a validated spec states them."""
    if is_stated(data):
        return _stated(data)
    # Imported here: that module builds this one's types.
    from .shorthand import expand

    return expand(data)


def named(data: Mapping[str, Any], name: str | None = None) -> Deployment:
    """The deployment called ``name``; core's when unnamed. ``KeyError`` naming the ones there are."""
    found = resolve(data)
    if name in (None, "", "core"):
        return found[0]
    for deployment in found:
        if deployment.name == name:
            return deployment
    raise KeyError(
        f"{name!r} names no deployment of this spec ({', '.join(d.name for d in found)})"
    )


def write_options(
    deployment: Deployment, directory: str | Path, extra: Mapping[str, Any] | None = None
) -> dict[str, Path]:
    """Write each server's options where ``trtllm-serve --config`` reads them; the files by role.

    ``<name>.yaml`` for an aggregated deployment, ``<name>.ctx.yaml`` and
    ``<name>.gen.yaml`` for a disaggregated one. ``extra`` is laid over each server's
    options: what the caller's check sets on every engine it starts.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, server in deployment.servers.items():
        suffix = "" if role == "server" else f".{role}"
        path = root / f"{deployment.name}{suffix}.yaml"
        written = {**server.options, **(extra or {})}
        path.write_text(yaml.safe_dump(written, sort_keys=False), encoding="utf-8")
        paths[role] = path
    return paths


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
            if server.launch:
                shape += ", behind a launch prefix"
            shapes.append(shape if role == "server" else f"{role} {shape}")
        judged = []
        for benchmark in deployment.benchmarks:
            source = (anchors or {}).get(f"{deployment.name}:{benchmark.name}")
            judged.append(f"{benchmark.name} ({source})" if source else benchmark.name)
        judged += [f"test {test}" for test in deployment.tests]
        lines.append(
            f"- `{deployment.name}`{' (core)' if deployment.core else ''}: {deployment.kind}, "
            f"{'; '.join(shapes)}; judged by its boot"
            + (f", {', '.join(judged)}" if judged else "")
        )
    return "\n".join(lines)


__all__ = [
    "AGGREGATED",
    "BENCHMARK_DEFAULTS",
    "DEFAULT_BENCHMARK",
    "DEPLOYMENTS_FIELD",
    "DISAGGREGATED",
    "INPUTS",
    "ROLES",
    "STATED_ACCURACY_KEYS",
    "Benchmark",
    "Deployment",
    "Server",
    "benchmark_of",
    "describe",
    "is_stated",
    "load_options",
    "named",
    "resolve",
    "resolve_paths",
    "split_args",
    "validate",
    "validate_env",
    "validate_judging",
    "write_options",
]
