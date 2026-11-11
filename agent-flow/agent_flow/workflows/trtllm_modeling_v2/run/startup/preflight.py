# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: what stops a run before its first turn, and what it records for the run.

Hard blockers, and only these: the checkpoint is readable; the tree imports; the first
deployment's devices exist and hold its weights; its temporary directory is writable; a
benchmark it is scored on has its dataset; every agent answers (``agent_probe.py``). The
tree, the architecture, the device memory, the fit and the temporary directory are probed
where the first deployment's engine will run: behind its launch prefix
(``environment.launch`` when it states none), on this host with neither (:func:`probe`).

Only the first deployment can stop the run. A later one that does not fit, or a benchmark
of one that lacks its dataset or an anchor, is recorded "not attempted" with its reason,
for the pipeline to skip and the report to list. A checkpoint config the tree cannot load
is recorded too and stops nothing: a task makes it load first, judged by the same probe.
Both are in ``preflight.json`` in the workspace. Everything else preflight sees is a note.
The anchors are frozen here, once per workspace: a later start adds only what is absent.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ...checks import anchors as anchors_mod
from ...checks import datasets
from ...checks.check_runtime import MODE_REQUIRE, with_mode
from ...infra.compute.environment import (
    RunEnvironment,
    env_assignments,
    resolve_environment,
    single_task_scrub,
)
from ...operators.coverage import MODELING_V2_PACKAGE
from ...spec import deployments as deployments_mod
from ...spec.advice import checkpoint_notes, spec_warnings, unfit
from ...spec.deployments import Deployment
from ...spec.task_schema import (
    DEFAULT_PARALLEL,
    derive_checkpoint_segment,
    derive_family,
    derive_slug,
    routed_family,
)

#: In the workspace: the "not attempted" records, the config record and the notes.
PREFLIGHT_FILENAME = "preflight.json"
#: A device is free when it is this close to empty: an idle card still shows a little in use.
MIN_FREE_MEMORY_FRACTION = 0.90
PROBE_TIMEOUT_S = 900
PROBE_MARKER = "MV2FLOW_PREFLIGHT "
#: Runs under the run's interpreter, in the tree, where an engine will: ``<checkpoint>``.
PROBE_SOURCE = r"""
import json, os, socket, subprocess, sys, tempfile
out = {"host": socket.gethostname()}
def said(exc):
    return "%s: %s" % (type(exc).__name__, exc)
try:
    import tensorrt_llm
    try:
        from tensorrt_llm._torch.model_config import ModelConfig
        ModelConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
    except Exception as exc:
        out["config_error"] = said(exc)
except Exception as exc:
    out["tree_error"] = said(exc)
try:
    with tempfile.NamedTemporaryFile(dir=os.environ.get("TMPDIR") or None):
        pass
except Exception as exc:
    out["tmp_error"] = said(exc)
try:
    query = ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,compute_cap",
             "--format=csv,noheader,nounits"]
    rows = subprocess.run(query, capture_output=True, text=True, timeout=120)
    out["devices"] = [[c.strip() for c in r.split(",")] for r in rows.stdout.splitlines() if r.strip()]
    if rows.returncode:
        out["devices_error"] = rows.stderr.strip()[-300:]
except Exception as exc:
    out["devices_error"] = said(exc)
print("MV2FLOW_PREFLIGHT " + json.dumps(out), flush=True)
"""


class PreflightError(RuntimeError):
    """A hard blocker: the run ends here, before its first turn."""


@dataclass
class PreflightResult:
    """The run's resolved identity and launch environment, and what preflight recorded."""

    slug: str
    branch: str
    environment: RunEnvironment
    devices: tuple[int, ...] = ()
    world_size: int = 1
    family: str = ""
    target_rel: str = ""
    checkpoint_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: A later deployment (``check`` empty) or one check of one the run does not attempt.
    not_attempted: list[dict[str, str]] = field(default_factory=list)
    #: Why the tree cannot load the checkpoint's config; ``""`` when it loads or was not asked.
    config_error: str = ""
    #: What each benchmark of each deployment is judged against, as frozen in the workspace.
    anchors: dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        """A summary for the journal and the console."""
        lines = [
            f"slug:        {self.slug}",
            f"target:      {self.target_rel}",
            f"devices:     {', '.join(map(str, self.devices)) or '(none claimed here)'}",
            f"prefix:      {self.environment.prefix or '(none)'}",
            f"interpreter: {shlex.quote(self.environment.python)}",
        ]
        if self.config_error:
            lines.append(f"config:      the tree cannot load it yet: {self.config_error}")
        skipped = [f"{r['deployment']} {r['check']}: {r['reason']}" for r in self.not_attempted]
        lines += [f"not attempted: {line}" for line in skipped]
        return "\n".join(lines + [f"note: {note}" for note in self.notes])


def _prefix(task_data: Mapping[str, Any], deployment: Deployment) -> str:
    """The prefix ``deployment``'s first server starts behind; the spec's own when it has none."""
    launch = next(iter(deployment.servers.values())).launch.strip()
    return launch or str((task_data.get("environment") or {}).get("launch") or "").strip()


def probe(task_data: Mapping[str, Any], deployment: Deployment, runner: Any) -> dict[str, Any]:
    """What is seen where ``deployment``'s engine will run, from every host that answered.

    ``devices`` is ``(host, index, free MiB, total MiB, arch)`` of each device seen;
    ``tree_error``, ``config_error`` and ``tmp_error`` are the first a host reported;
    ``error`` says that nothing answered.
    """
    environment = resolve_environment(task_data)
    env = {**environment.env, **deployment.env}
    environment = with_mode(replace(environment, env=env), MODE_REQUIRE)
    argv = ["-c", PROBE_SOURCE, str(task_data["checkpoint_path"])]
    launch = _prefix(task_data, deployment)
    command = environment.python_command(argv, single_process=True)
    if launch:
        words = " ".join(shlex.quote(word) for word in (environment.python, *argv))
        inner = f"{single_task_scrub(list(env), rank=False)}; {launch} {words}"
        command = f"{' '.join(env_assignments(environment.env))} bash -c {shlex.quote(inner)}"
    repo = shlex.quote(str(task_data["trtllm_repo_path"]))
    result = runner(f"cd {repo} && {command}", None, PROBE_TIMEOUT_S)
    output = f"{result.stdout}\n{result.stderr}"
    records = []
    for line in output.splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                records.append(json.loads(line[len(PROBE_MARKER) :]))
            except ValueError:
                continue  # a line a rank tore: no answer of its own
    found: dict[str, Any] = {"devices": []}
    if not records:
        tail = "\n".join(output.strip().splitlines()[-8:])
        found["error"] = f"it exited {result.returncode} with no answer; its last lines:\n{tail}"
    for record in records:
        host = record.get("host")
        for key in ("tree_error", "config_error", "tmp_error", "devices_error"):
            if record.get(key):
                found.setdefault(key, f"{record[key]} (on {host})")
        for index, free, total, cap in record.get("devices") or ():
            arch = "sm_" + cap.replace(".", "") if "." in cap else f"sm_{cap}0"
            if (device := (host, int(index), int(free), int(total), arch)) not in found["devices"]:
                found["devices"].append(device)
    return found


def resolve_target_identity(task_data: Mapping[str, Any], config: Mapping[str, Any]) -> tuple:
    """``(family, target_rel)`` of the run's target.

    The family is the one the spec states, else the one whose ``routing.py`` in the checkout
    routes the checkpoint's ``architectures[0]``, else that name's own.
    """
    override = dict(task_data.get("target") or {})
    architectures = config.get("architectures") or []
    if not architectures and not override.get("family"):
        raise PreflightError(
            "the checkpoint's config.json declares no `architectures`, so the modeling_v2 "
            "family cannot be derived. Set `target.family` in task.yaml."
        )
    family = (
        override.get("family")
        or routed_family(task_data["trtllm_repo_path"], str(architectures[0]))
        or derive_family(str(architectures[0]))
    )
    segment = override.get("checkpoint_segment") or derive_checkpoint_segment(
        task_data["checkpoint_path"]
    )
    name = f"{segment}__{task_data['arch']}__{task_data.get('parallel') or DEFAULT_PARALLEL}"
    return family, f"{MODELING_V2_PACKAGE}/models/{family}/{name}"


def _claim(task_data, core: Deployment, need: int, runner: Any, result: PreflightResult) -> tuple:
    """Hold the first deployment to what its probe sees; ``(a device's MiB, this host's free ones)``."""
    seen, arch = probe(task_data, core, runner), str(task_data["arch"])
    prefix = _prefix(task_data, core)
    where = f"where `{core.name}` runs ({f'behind `{prefix}`' if prefix else 'on this host'})"
    blockers = (
        ("error", "the probe did not answer"),
        ("tree_error", "the tree does not import"),
        ("tmp_error", "the temporary directory is not writable"),
    )
    for key, what in blockers:
        if seen.get(key):
            raise PreflightError(f"{what} {where}: {seen[key]}")
    devices = [d for d in seen["devices"] if d[4] == arch]
    if not devices:
        others = sorted({d[4] for d in seen["devices"]}) or seen.get("devices_error") or "none"
        raise PreflightError(f"`{core.name}` needs {arch} devices; {where} the probe saw {others}")
    idle = [d for d in devices if d[2] >= MIN_FREE_MEMORY_FRACTION * d[3]]
    busy = [f"device {d[1]} on {d[0]} is in use: {d[2]} of {d[3]} MiB free" for d in devices]
    result.notes += [line for line, d in zip(busy, devices) if d not in idle]
    if len(idle) < need:
        raise PreflightError(
            f"`{core.name}` needs {need} free {arch} device(s) on this host; {len(idle)} of "
            f"{len(devices)} are free:\n  - " + "\n  - ".join(result.notes or ["(none seen)"])
        )
    memory = min(d[3] for d in devices)
    if unfit(task_data, core, memory):
        raise PreflightError(unfit(task_data, core, memory))
    result.config_error = seen.get("config_error", "")
    # The devices seen are this host's only when the first deployment starts here.
    return memory, None if prefix else [d[1] for d in idle]


def run_preflight(
    task_data: Mapping[str, Any],
    *,
    runner: Any,
    claim: bool = True,
    workspace: str | Path | None = None,
) -> PreflightResult:
    """Hold the first deployment to the hard blockers and resolve the run's identity.

    ``runner`` runs the probe's command line. The devices claimed are this host's free
    ones, as many as the widest deployment the run attempts here takes. Without ``claim``
    (a dry run) nothing is probed and they are taken to be the first ones. With
    ``workspace`` the anchors are frozen there, adding only what it does not hold yet.
    """
    checkpoint = Path(str(task_data["checkpoint_path"]))
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"a JSON object is expected, not {type(config).__name__}")
    except (OSError, ValueError) as exc:
        why = f"{type(exc).__name__}: {exc}"
        raise PreflightError(f"the checkpoint is not readable: {checkpoint}/config.json ({why})")
    found = deployments_mod.resolve(task_data)
    core = found[0]
    # What runs on this host: a server behind a launch prefix places its own ranks.
    here = {d.name: [s.ranks for s in d.servers.values() if not s.launch.strip()] for d in found}
    widest = sum(here[core.name])
    slug = derive_slug(task_data)
    result = PreflightResult(slug, f"trtllm-modeling-v2/{slug}", resolve_environment(task_data))
    result.family, result.target_rel = resolve_target_identity(task_data, config)
    result.checkpoint_config = config
    if workspace is not None:
        result.anchors = anchors_mod.freeze(workspace, task_data)
    memory, free = None, None
    if claim:
        memory, free = _claim(task_data, core, widest, runner, result)
    else:
        result.notes.append("dry run: nothing was probed; a real start probes where it runs")
    env = dict(result.environment.env)
    for deployment in found:
        same = _prefix(task_data, deployment) == _prefix(task_data, core)
        why = unfit(task_data, deployment, memory if same else None)
        wants = sum(here[deployment.name])
        if not why and free is not None and wants > len(free):
            why = f"`{deployment.name}` needs {wants} free devices on this host; it has {len(free)}"
        if why and deployment is not core:
            result.not_attempted.append(dict(deployment=deployment.name, check="", reason=why))
            continue
        widest = max(widest, wants)
        result.world_size = max([result.world_size, *here[deployment.name]])
        for benchmark in deployment.benchmarks:
            name = benchmark.name
            path, said = datasets.find(name, benchmark.dataset_path, {**env, **deployment.env})
            anchor = result.anchors.get(f"{deployment.name}:{name}")
            anchor = anchor or anchors_mod.resolve(task_data, deployment, benchmark)
            reason = said if said and not path else ""
            if reason and deployment is core:
                raise PreflightError(f"`{core.name}` is scored on `{name}`: {said}")
            reason = reason or (anchor.source if anchor.kind == anchors_mod.NONE else "")
            if reason and deployment is not core:
                record = dict(deployment=deployment.name, check=f"accuracy[{name}]", reason=reason)
                result.not_attempted.append(record)
    result.devices = tuple(free[:widest] if free is not None else range(widest))
    if claim and free is None and widest:
        result.notes.append(
            f"devices: the first {widest} of this host are taken for the deployments that "
            f"start here, unprobed: `{core.name}` was probed behind its prefix"
        )
    result.notes += checkpoint_notes(task_data, config) + spec_warnings(task_data)
    result.environment = resolve_environment(
        task_data, devices=result.devices, world_size=result.world_size
    )
    return result
