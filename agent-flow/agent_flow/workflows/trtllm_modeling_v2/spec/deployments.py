# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployments: what a run brings up, stated as TensorRT-LLM itself takes it.

A deployment is one served configuration of the checkpoint: the options
``trtllm-serve --config`` takes, behind an optional launch prefix. The first
deployment is core's: it defines ``onboarded``. A disaggregated deployment is a
context and a generation server (``ctx``, ``gen``), each with its own options
and prefix::

    deployments:
      - name: dep16
        options: engines/dep16.yaml      # a native options file, or the options as a mapping
        launch: "<site prefix>"          # empty: this host
        timeout: 5400
        benchmarks:
          - name: gsm8k                  # judged against the reference row of accuracy.model_key
          - {name: aa_lcr, anchor: 62.0, source: "<where it was measured>"}
      - name: pd
        env: {UCX_TLS: "tcp,self,sm,cuda_copy,cuda_ipc"}
        ctx: {options: engines/ctx.yaml, launch: "<nodes 1 to 4>"}
        gen: {options: engines/gen.yaml, launch: "<nodes 5 to 8>"}
        relative_to: dep16               # judged against dep16's own score

The spec's older keys (``parallel``, ``members``, ``features``, ``variants``,
``runtime``, ``accuracy.llm_api_options``) are shorthand for the same thing.
:func:`resolve` reads either form: it expands the shorthand so that each unit
and each variant at each of its members is one deployment (:func:`name_for`),
whose options are what the spec's switches, flags and options file come to.
Nothing downstream needs to know which form the spec was written in.
"""

from __future__ import annotations

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

DEPLOYMENT_KEYS: tuple[str, ...] = (
    "name",
    "options",
    "launch",
    "stop",
    "env",
    "timeout",
    "relative_to",
    "tests",
    "inputs",
    "benchmarks",
    *ROLES,
)
#: What one server states: on an aggregated deployment itself, or inside ``ctx`` and ``gen``.
#: ``stop`` is the site's command that ends what ``launch`` started, where a signal to the
#: launcher is not enough.
SERVER_KEYS: tuple[str, ...] = ("options", "launch", "stop")
BENCHMARK_KEYS: tuple[str, ...] = ("name", "args", "anchor", "source", "env")
#: Shorthand a spec that states ``deployments`` cannot also state: each says what runs.
SHORTHAND_KEYS: tuple[str, ...] = (
    "members",
    "features",
    "variants",
    "runtime",
    "scope",
    "profile",
    "coverage",
)
#: The ``accuracy`` keys a spec that states ``deployments`` keeps, the defaults of every
#: deployment: the reference rows' model, file and tags, the benchmark of a deployment that
#: names none, the dataset's place and the boot workload. A deployment's ``benchmarks``
#: override them per benchmark; an engine setting is in its options.
NATIVE_ACCURACY_KEYS: tuple[str, ...] = (
    "model_key",
    "benchmark",
    "reference_yaml",
    "dataset_path",
    "discriminators",
    "boot",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Benchmark:
    """One benchmark a deployment is scored on: a ``trtllm-eval`` subcommand and its options.

    Without ``anchor`` it is judged against the tree's reference row for
    ``accuracy.model_key``, or the deployment's ``relative_to``; with neither, its score
    is reported and not judged. ``anchor`` is on the scale the benchmark reports.
    """

    name: str
    args: tuple[str, ...] = ()
    anchor: float | None = None
    source: str = ""
    env: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.args:
            out["args"] = list(self.args)
        if self.anchor is not None:
            out.update(anchor=self.anchor, source=self.source)
        if self.env:
            out["env"] = dict(self.env)
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

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"options": self.options_path or dict(self.options)}
        for key in ("launch", "stop"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


@dataclass(frozen=True)
class Deployment:
    """One deployment of the spec, in the form every later step reads."""

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
    #: Request settings a shorthand spec's runtime flags state (``--temperature``, ...) and
    #: whether its prompts go through the chat template. A deployment a spec states itself
    #: has neither: its checks sample greedily, as the check's own protocol says.
    sampling: Mapping[str, Any] = field(default_factory=dict)
    chat_template: bool = False

    @property
    def kind(self) -> str:
        return AGGREGATED if self.server is not None else DISAGGREGATED

    @property
    def servers(self) -> dict[str, Server]:
        """Its servers by role: ``{"server": ...}``, or ``{"ctx": ..., "gen": ...}``."""
        if self.server is not None:
            return {"server": self.server}
        return {role: getattr(self, role) for role in ROLES}

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


def is_native(data: Mapping[str, Any]) -> bool:
    """Whether the spec states its deployments itself, not through the shorthand keys."""
    return data.get(DEPLOYMENTS_FIELD) is not None


def name_for(unit: str, member: str) -> str:
    """The deployment a shorthand unit or variant is at ``member``; core's is the member itself."""
    return member if unit == "core" else f"{unit}@{member}"


# --------------------------------------------------------------------- a stated block


def _load_options(source: Any) -> dict[str, Any]:
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
    if not isinstance(block, list):
        return
    for entry in block:
        if not isinstance(entry, dict):
            continue
        for holder in (entry, *(entry.get(role) for role in ROLES)):
            if isinstance(holder, dict) and isinstance(holder.get("options"), str):
                path = Path(holder["options"])
                if holder["options"].strip() and not path.is_absolute():
                    holder["options"] = str((spec_dir / path).resolve())


def _validate_env(env: Any, errors: list[str], where: str) -> None:
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


def _validate_server(holder: Mapping[str, Any], errors: list[str], where: str) -> None:
    """``options`` (required), ``launch`` and ``stop`` of an aggregated deployment or of one side."""
    if holder.get("options") is None:
        errors.append(f"'{where}.options' is required: a native LLM-API options file")
    else:
        try:
            _load_options(holder["options"])
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
        anchor, source = entry.get("anchor"), entry.get("source")
        if anchor is not None:
            if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or anchor < 0:
                errors.append(f"'{at}.anchor' must be a score >= 0, on the benchmark's own scale")
            if not isinstance(source, str) or not source.strip():
                errors.append(
                    f"'{at}.source' is required with an anchor: where and how it was measured"
                )
        elif source is not None:
            errors.append(f"'{at}.source' says where an `anchor` comes from, and there is none")
        _validate_env(entry.get("env"), errors, f"{at}.env")


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
        _validate_env(entry.get("env"), errors, f"{at}.env")
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
            and "text" in inputs
            and len(set(inputs)) == len(inputs)
            and all(i in INPUTS for i in inputs)
        ):
            errors.append(f"'{at}.inputs' must be `text` and any of {list(INPUTS[1:])}, each once")
        _validate_benchmarks(entry.get("benchmarks"), errors, f"{at}.benchmarks")
        if isinstance(name, str):
            names.append(name)


def validate_native_spec(data: Mapping[str, Any], errors: list[str]) -> None:
    """What a spec that states ``deployments`` may and may not say beside them."""
    validate(data.get(DEPLOYMENTS_FIELD), errors)
    stated = [key for key in SHORTHAND_KEYS if data.get(key)]
    if stated:
        errors.append(
            f"{', '.join(repr(key) for key in stated)} beside '{DEPLOYMENTS_FIELD}': these keys "
            f"are shorthand for deployments, and this spec states its deployments itself. Put "
            f"what they ask for in a deployment's options, `inputs`, `benchmarks` or `tests`"
        )
    accuracy = data.get(ACCURACY_FIELD)
    if accuracy is None:
        return
    if not isinstance(accuracy, Mapping):
        errors.append(f"'{ACCURACY_FIELD}' must be a mapping, got {type(accuracy).__name__}")
        return
    extra = sorted(set(accuracy) - set(NATIVE_ACCURACY_KEYS))
    if extra:
        errors.append(
            f"'{ACCURACY_FIELD}' carries {extra} beside '{DEPLOYMENTS_FIELD}': with deployments "
            f"it keeps {list(NATIVE_ACCURACY_KEYS)} only, the defaults of every deployment; a "
            f"benchmark's options and its anchor are a deployment's `benchmarks`, and an engine "
            f"setting is in its options"
        )
    for key in ("benchmark", "reference_yaml", "dataset_path"):
        value = accuracy.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"'{ACCURACY_FIELD}.{key}' must be a non-empty string")
    tags = accuracy.get("discriminators")
    if tags is not None and not (
        isinstance(tags, Mapping)
        and all(isinstance(v, (str, int, float, bool)) for v in tags.values())
    ):
        errors.append(
            f"'{ACCURACY_FIELD}.discriminators' must map a reference row's tag to a scalar"
        )
    key = accuracy.get("model_key")
    if key is not None and (not isinstance(key, str) or not key.strip()):
        errors.append(f"'{ACCURACY_FIELD}.model_key' must be the model's id, `org/model`")


def _server(holder: Mapping[str, Any]) -> Server:
    source = holder.get("options")
    return Server(
        options=_load_options(source),
        options_path=source if isinstance(source, str) else "",
        launch=str(holder.get("launch") or ""),
        stop=str(holder.get("stop") or ""),
    )


def _args(value: Any) -> tuple[str, ...]:
    """A benchmark's options: a list of strings, or one string split as a shell would."""
    if isinstance(value, str):
        return tuple(shlex.split(value))
    return tuple(str(arg) for arg in value or ())


def _stated(data: Mapping[str, Any]) -> tuple[Deployment, ...]:
    """The deployments of a validated spec that states them.

    A deployment that names no ``benchmarks`` takes the spec's: ``accuracy.benchmark``
    (``gsm8k`` unsaid) when the spec has an ``accuracy`` block, none without one.
    """
    accuracy = data.get(ACCURACY_FIELD)
    default = (
        [{"name": str(accuracy.get("benchmark") or DEFAULT_BENCHMARK)}]
        if isinstance(accuracy, Mapping)
        else []
    )
    out = []
    for index, entry in enumerate(data[DEPLOYMENTS_FIELD]):
        disaggregated = entry.get(ROLES[0]) is not None
        stated = entry.get("benchmarks")
        benchmarks = tuple(
            Benchmark(
                name=str(b["name"]),
                args=_args(b.get("args")),
                anchor=float(b["anchor"]) if b.get("anchor") is not None else None,
                source=str(b.get("source") or ""),
                env={str(k): str(v) for k, v in (b.get("env") or {}).items()},
            )
            for b in (default if stated is None else stated)
        )
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
                benchmarks=benchmarks,
            )
        )
    return tuple(out)


def resolve(data: Mapping[str, Any], world_size: int | None = None) -> tuple[Deployment, ...]:
    """The spec's deployments, core's first, whichever way a validated spec states them.

    ``world_size`` is the world of a shorthand spec that names no member with one;
    unsaid, it is the widest member's. Raises ``ValueError`` where the shorthand's
    switches, flags and options file ask for two values of one setting.
    """
    if is_native(data):
        return _stated(data)
    # Imported here: that module builds this one's types.
    from .shorthand import expand

    return expand(data, world_size)


def write_options(deployment: Deployment, directory: str | Path) -> dict[str, Path]:
    """Write each server's options where ``trtllm-serve --config`` reads them; the files by role.

    ``<name>.yaml`` for an aggregated deployment, ``<name>.ctx.yaml`` and
    ``<name>.gen.yaml`` for a disaggregated one.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, server in deployment.servers.items():
        suffix = "" if role == "server" else f".{role}"
        path = root / f"{deployment.name}{suffix}.yaml"
        path.write_text(yaml.safe_dump(dict(server.options), sort_keys=False), encoding="utf-8")
        paths[role] = path
    return paths


def describe(found: tuple[Deployment, ...]) -> str:
    """One line for each deployment: its name, its servers' topology and what judges it."""
    lines = []
    for deployment in found:
        shapes = []
        for role, server in deployment.servers.items():
            options = server.options
            ranks = options.get("tensor_parallel_size", 1)
            shape = f"{ranks} rank(s)"
            if options.get("enable_attention_dp"):
                shape += ", attention DP"
            if server.launch:
                shape += ", behind a launch prefix"
            shapes.append(shape if role == "server" else f"{role} {shape}")
        judged = [b.name for b in deployment.benchmarks] + [f"test {t}" for t in deployment.tests]
        if deployment.relative_to:
            judged.append(f"relative to {deployment.relative_to}")
        lines.append(
            f"- `{deployment.name}`{' (core)' if deployment.core else ''}: {deployment.kind}, "
            f"{'; '.join(shapes)}; judged by its boot"
            + (f", {', '.join(judged)}" if judged else "")
        )
    return "\n".join(lines)


__all__ = [
    "AGGREGATED",
    "DEPLOYMENTS_FIELD",
    "DISAGGREGATED",
    "INPUTS",
    "ROLES",
    "Benchmark",
    "Deployment",
    "Server",
    "describe",
    "is_native",
    "name_for",
    "resolve",
    "resolve_paths",
    "validate",
    "validate_native_spec",
    "write_options",
]
