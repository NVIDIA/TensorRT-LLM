# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The geometry probe: the serving geometry, measured before any kernel is certified.

A kernel certified against an estimated shape -- a pool size, a page view,
the chunks a snapshot policy cuts, the CUDA-graph batches captured, a
speculation depth -- is certified again once a gate shows the real one. So in
phase 0, beside the survey, the probe boots the engine once per width the
spec declares and records what it serves (``checks/harness/geometry_probe.py``)
in ``runtime_facts.yaml`` (:func:`~.runtime_facts.record_geometry`), whose
block every prompt carries: the planner's, the survey's, every smith's.

The legs (:func:`plan_legs`): the narrowest member, the widest member that
is not attention-DP, and the widest attention-DP member when one is
declared, each under core's runtime block; and, when a feature turns
speculation on, the narrowest member again under that unit's, whose capture
list and verify width follow the draft depth.

The path: the stock modeling path (``off``) before a target exists, or the
target (``require``) once it boots. A new architecture has no stock path;
each leg that says so boots ``environment.substrate_checkpoint`` instead,
when the spec declares one, and keeps only what the run's configuration
decides (:data:`CONFIGURATION_NUMBERS`, :data:`CONFIGURATION_PAGE`): the
capture list, the limits, the page size, the widths. Its pools and chunks are
another model's and are left out.

Once the core boots on the target at a width, the target's own geometry
there is measured once more (:func:`reprobe_target`, ``require`` path), and
the values it measured that no earlier probe had are one planner event that
names the entries certified so far (:func:`recertification_note`): the
stand-in or the stock path is an estimate, and re-certification is asked for
once, as a batch.

A probe that fails is never a stop: the run goes on, on estimates, and the
block says which legs were not measured. A resumed run whose file holds a
probe of the same inputs (:func:`inputs_digest`) does not probe again.

The probe waits for a substrate gate running beside it, so the two never
share a device, and runs its legs one after another unless a scheduler hands
each job its own devices. A smith dispatch waits for a probe still running,
at most :data:`WAIT_S` (:func:`await_probe`), so the first certification
reads measured shapes.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from agent_flow.console import print_message

from ..checks import gates as gates_mod
from ..checks.harness import workload_io
from ..infra.environment import RunEnvironment, scheduler_owns_devices
from ..spec.task_schema import assembly_units, members_for, parse_parallel_segment
from . import progress as progress_mod
from . import runtime_facts

#: The probe, run as a file under the measured tree's interpreter.
GEOMETRY_PROBE = Path(__file__).resolve().parents[1] / "checks" / "harness" / "geometry_probe.py"
#: Its record's marker; see the harness.
GEOMETRY_MARKER = "MV2FLOW_GEOMETRY"
#: The tokens a probe prompt generates (the harness's ``PROBE_MAX_TOKENS``).
PROBE_MAX_TOKENS = 2
#: How long one leg may take, in seconds: one cold boot and a few prompts.
PROBE_TIMEOUT_S = 1800
#: How long a smith dispatch waits for a probe still running, in seconds.
WAIT_S = 15 * 60
#: Prompt lengths every leg probes: under a page, over the common prefill chunks.
DEFAULT_PROBE_LENGTHS: tuple[int, ...] = (257, 2049, 8193)
#: The most prompt lengths a leg probes.
MAX_PROBE_LENGTHS = 8
#: Where the probe's specs are written, under the workspace.
WORKDIR = Path("gates") / "geometry"

STATUS_MEASURED = runtime_facts.MEASURED
#: The stock path does not know the checkpoint's architecture.
STATUS_UNSUPPORTED = "unsupported"
STATUS_FAILED = "failed"
SOURCE_CHECKPOINT = "checkpoint"
SOURCE_PROXY = runtime_facts.SOURCE_PROXY

#: A leg's engine numbers a stand-in checkpoint measures for this run's configuration too.
CONFIGURATION_NUMBERS: tuple[str, ...] = (
    "cuda_graph_batches",
    "cuda_graph_padded",
    "speculative",
    "tp",
    "ep",
    "attention_dp",
    "max_batch_size",
    "max_num_tokens",
)
#: A leg's page view a stand-in checkpoint measures for this run's configuration too.
CONFIGURATION_PAGE: tuple[str, ...] = (
    "tokens_per_block",
    "max_seq_len",
    "max_num_tokens",
    "max_batch_size",
    "chunked_prefill",
    "max_blocks_per_seq",
)

#: Narrowest first among members of one width: the general path.
_NARROW_ORDER = {"tp": 0, "tep": 1, "dep": 2}


@dataclass(frozen=True)
class ProbeLeg:
    """One engine boot: a member, under a unit's runtime block and with its features."""

    member: str
    unit: str = "core"
    features: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.unit}@{self.member}"


def _features_through(units: Sequence[str], unit: str) -> tuple[str, ...]:
    """The features a target implements once ``unit`` is assembled (core is none)."""
    if unit not in units:
        return ()
    return tuple(units[1 : units.index(unit) + 1])


def plan_legs(task_data: Mapping[str, Any], world_size: int) -> list[ProbeLeg]:
    """The legs one probe boots; see the module docstring."""
    units = assembly_units(task_data)
    members: list[str] = []
    for unit in units:
        for member in members_for(task_data, unit, world_size):
            if member not in members:
                members.append(member)
    sized = []
    for member in members:
        kind, size = parse_parallel_segment(member)
        sized.append((member, kind, size if size is not None else world_size))
    if not sized:
        return []
    narrowest = min(sized, key=lambda m: (m[2], _NARROW_ORDER.get(m[1], 3), m[0]))[0]
    plain = [m for m in sized if m[1] != "dep"] or sized
    widest = max(plain, key=lambda m: (m[2], m[1] == "tep", m[0]))[0]
    chosen = [narrowest, widest]
    dp = [m for m in sized if m[1] == "dep"]
    if dp:
        chosen.append(max(dp, key=lambda m: (m[2], m[0]))[0])
    legs = [ProbeLeg(member) for member in dict.fromkeys(chosen)]
    for unit in units[1:]:
        if gates_mod.speculation_flags(_features_through(units, unit)):
            legs.append(ProbeLeg(narrowest, unit=unit, features=_features_through(units, unit)))
            break
    return legs


def _declared_lengths(witness: Any) -> list[int]:
    """Every ``prompt_tokens`` length under the spec's witness block."""
    found: list[int] = []
    if isinstance(witness, Mapping):
        for key, value in witness.items():
            if key == "prompt_tokens" and isinstance(value, list):
                found += [int(v) for v in value if isinstance(v, int) and not isinstance(v, bool)]
            else:
                found += _declared_lengths(value)
    return found


def probe_lengths(task_data: Mapping[str, Any], unit: str, argv: Sequence[str] = ()) -> list[int]:
    """The prompt lengths a leg probes: where the token budget and the snapshot policy cut.

    The defaults, one past the prefill budget and past twice it, one past the
    periodic snapshot interval and past two and a half of it, one past each
    snapshot offset from the start, and the witness's declared prompt lengths.
    The harness drops a length the engine cannot hold.
    """
    lengths = set(DEFAULT_PROBE_LENGTHS)
    budget = gates_mod._flag_value(argv, "--max_num_tokens")
    options = gates_mod.load_llm_api_options(
        gates_mod.resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    )
    budget = budget or options.get("max_num_tokens")
    try:
        budget = int(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    if budget:
        lengths |= {budget + 1, 2 * budget + 1}
    policy = gates_mod.resolve_kv_cache_config(task_data, unit).get("mamba_state_config") or {}
    if isinstance(policy, Mapping):
        interval = policy.get("periodic_snapshot_interval")
        if isinstance(interval, int) and interval > 0:
            lengths |= {interval + 1, (5 * interval) // 2}
        for offset in policy.get("additional_snapshot_offsets_from_start") or ():
            if isinstance(offset, int) and offset > 0:
                lengths.add(offset + 1)
    lengths |= set(_declared_lengths((task_data.get("accuracy") or {}).get("witness")))
    return sorted(n for n in lengths if n > 0)[:MAX_PROBE_LENGTHS]


def declared_verify_depths(task_data: Mapping[str, Any]) -> dict[str, int]:
    """The draft depth of each unit and configuration variant that runs speculation."""
    units = assembly_units(task_data)
    try:
        configurations = gates_mod._launch_configurations(task_data)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    depths: dict[str, int] = {}
    for name, unit, options, _runtime, _source in configurations:
        if not gates_mod.speculation_flags(_features_through(units, unit)):
            continue
        spec = options.get("speculative_config") if isinstance(options, Mapping) else None
        depth = 1
        for key in ("max_draft_len", "num_nextn_predict_layers", "block_size"):
            value = (spec or {}).get(key) if isinstance(spec, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool):
                depth = max(1, value)
                break
        depths[name] = depth
    return depths


def leg_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    leg: ProbeLeg,
    *,
    mode: str,
    workdir: Path,
) -> tuple[str, list[str]]:
    """The leg's launch and its quickstart argv; its spec is written under ``workdir``."""
    env = gates_mod.with_kv_cache_config(
        gates_mod._with_mode(environment, mode), task_data, leg.unit
    )
    argv = gates_mod.quickstart_words(
        task_data,
        environment,
        unit=leg.unit,
        features=list(leg.features),
        max_tokens=PROBE_MAX_TOKENS,
        member=leg.member,
    )
    quickstart = gates_mod.quickstart_script(task_data)
    spec = {
        "quickstart": str(quickstart),
        "quickstart_argv": argv,
        "chunk_probes": probe_lengths(task_data, leg.unit, argv),
    }
    path = Path(workdir) / f"geometry.{leg.unit}.{leg.member}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    words = " ".join(shlex.quote(a) for a in argv)
    command = env.python_command(
        f"{shlex.quote(str(GEOMETRY_PROBE))} --spec {shlex.quote(str(path))} "
        f"--quickstart {shlex.quote(str(quickstart))} -- {words}"
    )
    return command, argv


def _configuration_only(record: dict[str, Any]) -> dict[str, Any]:
    """A stand-in checkpoint's leg less what its own model decides."""
    record["numbers"] = {
        k: v for k, v in (record.get("numbers") or {}).items() if k in CONFIGURATION_NUMBERS
    }
    record["page"] = {
        k: v for k, v in (record.get("page") or {}).items() if k in CONFIGURATION_PAGE
    }
    record.pop("chunks", None)
    return record


def run_leg(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    runner: Callable[..., Any],
    leg: ProbeLeg,
    *,
    mode: str,
    workdir: Path,
    timeout: int = PROBE_TIMEOUT_S,
    source: str = SOURCE_CHECKPOINT,
) -> dict[str, Any]:
    """Boot one leg and read what it measured, as a ``geometry`` leg of ``runtime_facts.yaml``."""
    command, argv = leg_command(task_data, environment, leg, mode=mode, workdir=workdir)
    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = (result.stdout or "") + (result.stderr or "")
    record: dict[str, Any] = {
        "unit": leg.unit,
        "member": leg.member,
        "source": source,
        "duration_s": round(time.monotonic() - started, 1),
    }
    if source == SOURCE_PROXY:
        record["checkpoint"] = str(task_data.get("checkpoint_path"))
    tail = "\n".join(combined.splitlines()[-3:])
    if result.infrastructure or result.timed_out or not result.ok:
        detail = (
            f"did not run to an answer: {result.infrastructure}"
            if result.infrastructure
            else "timed out"
            if result.timed_out
            else f"exited {result.returncode}"
        )
        unsupported = gates_mod.reports_unsupported_architecture(detail, combined)
        record["status"] = STATUS_UNSUPPORTED if unsupported else STATUS_FAILED
        record["detail"] = (
            "the stock path does not know this checkpoint's architecture"
            if unsupported
            else f"{detail}: {tail}"[:400]
        )
        return record
    declared = gates_mod.engine_declaration(task_data, leg.unit, argv, mode=mode)
    attested = gates_mod.attest_engine(combined, declared)
    numbers = runtime_facts.engine_numbers(attested.record())
    found = workload_io.parse_marked(combined, GEOMETRY_MARKER)
    geometry = found[-1] if found else {}
    page = geometry.get("page") if isinstance(geometry.get("page"), Mapping) else {}
    record.update(numbers=numbers, page=dict(page))
    chunks = geometry.get("chunks")
    if isinstance(chunks, Mapping) and chunks:
        record["chunks"] = {str(k): list(v) for k, v in chunks.items() if isinstance(v, list)}
    errors = dict(geometry.get("errors") or {}) if isinstance(geometry, Mapping) else {}
    if geometry.get("error"):
        errors["geometry"] = str(geometry["error"])
    if attested.mismatches:
        errors["declared"] = attested.difference()[:400]
    if errors:
        record["errors"] = {str(k): str(v)[:300] for k, v in errors.items()}
    if not numbers and not page:
        record["status"] = STATUS_FAILED
        record["detail"] = f"the probe printed no engine or geometry record: {tail}"[:400]
        return record
    record["status"] = STATUS_MEASURED
    return _configuration_only(record) if source == SOURCE_PROXY else record


def inputs_digest(
    task_data: Mapping[str, Any], mode: str, legs: Sequence[ProbeLeg], marker: str = ""
) -> str:
    """What a probe's measurement stands for: everything its legs' launches are built from.

    The checkpoint, the stand-in, the widths, each leg's runtime block, options
    file (by content) and handed-over KV-cache keys, the interpreter and the
    checkout, and for the target path the target's marker. Not the stock
    path's tree: a probe measures the engine's configuration, which the
    runtime block decides.
    """
    accuracy = task_data.get("accuracy") or {}
    parts: dict[str, Any] = {
        "mode": mode,
        "checkpoint": task_data.get("checkpoint_path"),
        "stand_in": (task_data.get("environment") or {}).get("substrate_checkpoint"),
        "python": (task_data.get("environment") or {}).get("python"),
        "checkout": task_data.get("trtllm_repo_path"),
        "marker": marker if mode == gates_mod.MODE_REQUIRE else "",
        "legs": [],
    }
    for leg in legs:
        path = gates_mod.resolve_llm_api_options(accuracy, leg.unit)
        text = None
        if path and Path(str(path)).is_file():
            text = Path(str(path)).read_text(encoding="utf-8", errors="replace")
        parts["legs"].append(
            {
                "leg": leg.key,
                "features": list(leg.features),
                "runtime": gates_mod.resolve_runtime(task_data, leg.unit),
                "options": [path, text],
            }
        )
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def recorded(workspace: Path, digest: str) -> bool:
    """Whether the file holds a probe of these inputs that measured at least one leg."""
    section = runtime_facts.geometry(workspace)
    legs = section.get("legs") if isinstance(section.get("legs"), Mapping) else {}
    return section.get("inputs") == digest and any(
        isinstance(leg, Mapping) and leg.get("status") == STATUS_MEASURED for leg in legs.values()
    )


def probe_geometry(
    task_data: Mapping[str, Any],
    *,
    workspace: Path,
    env_for: Callable[[str], RunEnvironment],
    runner_for: Callable[[str], Callable[..., Any]],
    world_size: int,
    mode: str = gates_mod.MODE_OFF,
    legs: Sequence[ProbeLeg] | None = None,
    marker: str = "",
    tree: str = "",
    timeout: int = PROBE_TIMEOUT_S,
    merge: bool = False,
) -> dict[str, Any]:
    """Boot each leg, record what they measured in ``runtime_facts.yaml``, and return the section.

    ``env_for`` and ``runner_for`` give a member's environment at its world
    size and the runner its gates use. A leg on the stock path that does not
    know the architecture boots the stand-in checkpoint, when the spec
    declares one.
    """
    legs = list(legs) if legs is not None else plan_legs(task_data, world_size)
    workdir = Path(workspace) / WORKDIR
    stand_in = str((task_data.get("environment") or {}).get("substrate_checkpoint") or "").strip()

    def one(leg: ProbeLeg) -> dict[str, Any]:
        env, runner = env_for(leg.member), runner_for(leg.member)
        common = dict(mode=mode, timeout=timeout)
        record = run_leg(task_data, env, runner, leg, workdir=workdir, **common)
        if record["status"] != STATUS_UNSUPPORTED or not stand_in or mode != gates_mod.MODE_OFF:
            return record
        proxy = run_leg(
            {**dict(task_data), "checkpoint_path": stand_in},
            env,
            runner,
            leg,
            workdir=workdir / "proxy",
            source=SOURCE_PROXY,
            **common,
        )
        return proxy

    kind = str((task_data.get("environment") or {}).get("kind", "local"))
    parallel = scheduler_owns_devices(kind) and len(legs) > 1
    if parallel:
        with ThreadPoolExecutor(max_workers=len(legs), thread_name_prefix="mv2-geometry") as pool:
            records = list(pool.map(one, legs))
    else:
        records = [one(leg) for leg in legs]
    if merge:
        for record in records:
            record.setdefault("path", mode)
    return runtime_facts.record_geometry(
        workspace,
        legs=records,
        inputs=inputs_digest(task_data, mode, legs, marker),
        mode=mode,
        tree=tree,
        marker=marker,
        verify_depths=declared_verify_depths(task_data),
        merge=merge,
    )


def summary(section: Mapping[str, Any]) -> str:
    """One line: how many legs were measured, and why each other one was not."""
    legs = section.get("legs") if isinstance(section.get("legs"), Mapping) else {}
    measured = [k for k, v in legs.items() if v.get("status") == STATUS_MEASURED]
    others = [
        f"{k}: {v.get('detail') or v.get('status')}" for k, v in legs.items() if k not in measured
    ]
    stand_in = [k for k in measured if legs[k].get("source") == SOURCE_PROXY]
    line = f"{len(measured)} of {len(legs)} leg(s) measured"
    if stand_in:
        line += f" ({', '.join(stand_in)} on the stand-in checkpoint)"
    return line + (f"; not measured: {'; '.join(others)}" if others else "")


class GeometryProbe:
    """A probe running on a thread of its own; :meth:`wait` blocks until it is done."""

    def __init__(self, run: Callable[[], dict[str, Any]], *, name: str = "mv2-geometry") -> None:
        self.section: dict[str, Any] = {}
        self.error = ""
        self._run = run
        self._done = threading.Event()
        self._waited = False
        self._thread = threading.Thread(target=self._body, name=name, daemon=True)

    def _body(self) -> None:
        try:
            self.section = self._run() or {}
        # The run waits on this answer, whatever the probe hit; it is never a stop.
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self._done.set()

    def start(self) -> "GeometryProbe":
        self._thread.start()
        return self

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Whether the probe is done, after waiting up to ``timeout`` seconds."""
        return self._done.wait(timeout)


def start_probe(stage: Any, *, mode: str = gates_mod.MODE_OFF) -> GeometryProbe | None:
    """Start the probe beside the run's first turns; ``None`` when a recorded one stands.

    ``stage`` is the workflow: its spec, workspace and journal, its
    environment at a member's width (``_member_env``) and the runner its
    gates use there (``_member_runner``). The probe first waits for a
    substrate gate running beside it, and is not run when that gate failed.
    It is kept as ``stage._geometry_probe`` for :func:`await_probe`.
    """
    task_data = stage.task_data
    world_size = stage._gate_env().world_size
    legs = plan_legs(task_data, world_size)
    marker_now = getattr(stage, "_marker_now", None)
    marker = marker_now() if callable(marker_now) and mode == gates_mod.MODE_REQUIRE else ""
    digest = inputs_digest(task_data, mode, legs, marker)
    if not legs:
        return None
    if recorded(stage.workspace, digest):
        stage.journal.event(
            "geometry probe not run",
            f"`{runtime_facts.facts_path(stage.workspace)}` holds a probe of the same inputs",
        )
        return None
    repo = str(task_data.get("trtllm_repo_path") or "")
    tree = gates_mod.describe_tree_state(repo) if repo and mode == gates_mod.MODE_REQUIRE else ""

    def run() -> dict[str, Any]:
        await_substrate = getattr(stage, "_await_substrate", None)
        if callable(await_substrate):
            await_substrate()
        if getattr(stage, "_substrate_failure", ""):
            stage.journal.event("geometry probe not run", "the substrate gate failed")
            return {}
        section = probe_geometry(
            task_data,
            workspace=stage.workspace,
            env_for=stage._member_env,
            runner_for=stage._member_runner,
            world_size=world_size,
            mode=mode,
            legs=legs,
            marker=marker,
            tree=tree,
        )
        line = summary(section)
        stage.journal.event(
            "geometry probe", f"{line}; `{runtime_facts.facts_path(stage.workspace)}`"
        )
        print_message(f"[bold cyan]◆ geometry probe[/bold cyan] {line}", None)
        return section

    probe = GeometryProbe(run).start()
    stage._geometry_probe = probe
    stage.journal.event(
        "geometry probe started", ", ".join(f"{leg.key}" for leg in legs) + f" ({mode} path)"
    )
    return probe


def recertification_note(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    members: Sequence[str],
    landed: Sequence[str],
) -> str:
    """The one planner event a probe of the target makes: what it measured that nothing was certified against.

    ``before`` is the geometry the certifications so far were made against
    (the stock path's, or a stand-in's), ``after`` the section with the
    target's legs at ``members``, and ``landed`` the entries certified so far.
    ``""`` when the target measured no value the earlier probe had not.
    """
    old = runtime_facts.geometry_values(before)
    new = runtime_facts.geometry_values(after)
    added = {
        name: [v for v in values if v not in old.get(name, ())] for name, values in new.items()
    }
    added = {name: values for name, values in added.items() if values}
    if not added:
        return ""
    values = "; ".join(
        f"{name} {values}" + (f" (the earlier probe: {old[name]})" if old.get(name) else "")
        for name, values in added.items()
    )
    entries = ", ".join(f"`{e}`" for e in landed) or "none yet"
    return (
        f"the target's own geometry at {', '.join(members)}, measured once its core booted "
        f"there, has values no certification was made against: {values}. Entries certified so "
        f"far: {entries}. Re-certify, as one batch, those whose cells take these values: a "
        f"`verify` task per entry, or one for all, its ```cells block the values above "
        f"(`runtime_facts.yaml`'s geometry block has them)"
    )


def _landed_entries(stage: Any) -> list[str]:
    """The entries smiths landed so far, from ``progress.yaml``."""
    path = getattr(stage, "progress_path", None)
    if path is None:
        return []
    rows = progress_mod.find_entries(path, stage=progress_mod.BUILD_STAGE, agent="smith")
    return sorted(
        {str(r.get("entry")) for r in rows if r.get("decision") == "LANDED" and r.get("entry")}
    )


def reprobe_target(stage: Any, member: str, notify: Callable[[str], None]) -> bool:
    """Probe the target's own geometry at ``member`` once its core boots there; ``False`` if done.

    Once per member: the first core boot at a width measures what the
    target serves there (``require`` path), beside the earlier probe's legs
    of the other widths (``merge``). Members whose core boots while a probe
    runs wait for it, and go in the next one together. When a probe ends,
    ``notify`` gets its :func:`recertification_note`, one event per probe.
    """
    lock = stage.__dict__.setdefault("_reprobe_lock", threading.Lock())
    with lock:
        done = stage.__dict__.setdefault("_reprobed", set())
        pending = stage.__dict__.setdefault("_reprobe_pending", set())
        if member in done or member in pending:
            return False
        pending.add(member)
        running = stage.__dict__.get("_reprobe")
        if isinstance(running, GeometryProbe) and not running.done:
            return True
    _start_reprobe(stage, notify)
    return True


def _start_reprobe(stage: Any, notify: Callable[[str], None]) -> None:
    lock = stage.__dict__["_reprobe_lock"]
    with lock:
        members = sorted(stage._reprobe_pending)
        stage._reprobe_pending.clear()
        stage._reprobed.update(members)
    task_data = stage.task_data
    world_size = stage._gate_env().world_size
    legs = [leg for leg in plan_legs(task_data, world_size) if leg.member in members]
    if not legs:
        return
    before = runtime_facts.geometry(stage.workspace)
    marker_now = getattr(stage, "_marker_now", None)
    marker = marker_now() if callable(marker_now) else ""
    repo = str(task_data.get("trtllm_repo_path") or "")
    tree = gates_mod.describe_tree_state(repo) if repo else ""

    def run() -> dict[str, Any]:
        try:
            section = probe_geometry(
                task_data,
                workspace=stage.workspace,
                env_for=stage._member_env,
                runner_for=stage._member_runner,
                world_size=world_size,
                mode=gates_mod.MODE_REQUIRE,
                legs=legs,
                marker=marker,
                tree=tree,
                merge=True,
            )
            stage.journal.event(
                "geometry probe of the target", f"{', '.join(members)}: {summary(section)}"
            )
            note = recertification_note(before, section, members, _landed_entries(stage))
            if note:
                notify(note)
            return section
        finally:
            with lock:
                again = bool(stage._reprobe_pending)
            if again:
                _start_reprobe(stage, notify)

    stage._reprobe = GeometryProbe(run, name="mv2-geometry-target").start()
    stage.journal.event(
        "geometry probe of the target started",
        ", ".join(leg.key for leg in legs) + " (require path)",
    )


def await_probe(stage: Any, timeout: float = WAIT_S) -> None:
    """Wait for the stage's probe while it runs, at most ``timeout`` seconds; journal the wait once."""
    probe = getattr(stage, "_geometry_probe", None)
    if not isinstance(probe, GeometryProbe) or probe.done:
        return
    started = time.monotonic()
    done = probe.wait(timeout)
    if probe._waited:
        return
    probe._waited = True
    waited = time.monotonic() - started
    stage.journal.event(
        "smith dispatch waited for the geometry probe",
        f"{waited:.0f}s"
        + ("" if done else f"; still running after {timeout:.0f}s, dispatched on what is known"),
    )


__all__ = [
    "CONFIGURATION_NUMBERS",
    "CONFIGURATION_PAGE",
    "DEFAULT_PROBE_LENGTHS",
    "GEOMETRY_MARKER",
    "GEOMETRY_PROBE",
    "GeometryProbe",
    "PROBE_TIMEOUT_S",
    "ProbeLeg",
    "STATUS_FAILED",
    "STATUS_MEASURED",
    "STATUS_UNSUPPORTED",
    "WAIT_S",
    "await_probe",
    "declared_verify_depths",
    "inputs_digest",
    "leg_command",
    "plan_legs",
    "probe_geometry",
    "probe_lengths",
    "recertification_note",
    "recorded",
    "reprobe_target",
    "run_leg",
    "start_probe",
    "summary",
]
