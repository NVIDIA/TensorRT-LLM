"""SOL tracks: optimizing the two halves of a disaggregated deployment apart.

An end-to-end disagg campaign (see :mod:`.disagg`) measures the whole
cluster at once. That is the most expensive measurement the flow can
make, and most of it is wasted: the knobs an optimizer can reach live
inside one role at a time, and a full e2e allocation prices in both.

The decomposition ``bench-trtllm-disagg`` is built around splits the
measurement in two, and each half is an *aggregate-shaped* campaign:

- **ctx** — ``trtllm-bench throughput`` at ``output_length: 1``. No
  disaggregation at all: one server, prefill only. Its metric is
  ``avg_request_throughput_req_s``.
- **gen** — a real 1-ctx-1-gen deployment, read only on the generation
  worker's decode iterations. Its metric is ``throughput_per_user``.

So neither track needs new tuning machinery, no stage changes, and no
new gate: N frontier points become ``benchmark.concurrency``, and
omitting ``optimize.max_regression_pct`` already means *any* point can
veto an attempt.

**What this module is, and is not.** It does not drive the benchmark
repo's scripts, and it does not re-implement their sweep expansion. That
repo ships ``bench-disagg`` and states plainly that the CLI is the only
supported agent-facing interface, with the scripts as internal backends
— so a campaign points at one orchestration ``sweep.yaml`` and
:mod:`.bench_cli` asks ``sweep plan`` what that expands to. Everything
this file used to compute by parsing YAML — the operating points, the
sequence lengths, the case names — now arrives already expanded, from
the same code that will run them. An expansion computed twice is one
that eventually disagrees with itself, and the disagreement would
surface as a campaign quoting operating points it never measured.

What remains is the problem :mod:`.disagg` names: **two files describe
one run.** The sweep owns the measurement conditions; ``task.yaml`` is
what the orchestrator reads to build every agent's prompt. When they
disagree nothing raises — the prompts simply quote numbers the run never
measured. So the reconciliation rule is unchanged: a condition the user
did not write is **filled**, one they wrote that **disagrees** is an
**error** naming both values.

One trap survives the move, because it is a property of the harness
rather than of the file it was read from: a sweep row's concurrency is
**per generation server**. The client is driven at ``concurrency *
gen_num`` and the harness names its result directory after that product,
while the case name keeps the listed value because that is its address.
``task.yaml``'s ``concurrency`` means what an aggregate campaign means
by it, so the product is what belongs there. See
:func:`.bench_cli.operating_point`.

Two pieces of glue remain, one per direction, and both are code rather
than prompt for the same reason: skipping either does not fail, it
produces a plausible wrong answer. :func:`apply_overlay` carries the
campaign's tuning file *into* the sweep before a submit — forget it and
the run measures the previous attempt's configuration under the new
attempt's name. :func:`collect` carries the score back *out* — forget it
and a measurement that succeeded is indistinguishable, to every later
stage, from one that never ran.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

# The full dotted name, not `from ... import bench_cli`: the dashboard
# loads these modules by path with `agent_flow` itself unimportable, and
# only the submodule form resolves from the sys.modules cache.
from agent_flow.workflows.perf_optimize.bench_cli import (
    BenchCliError,
    frontier_show,
    operating_point,
    operating_points,
    status,
)

SOL_TRACK_FIELD = "sol_track"
TRACK_KEY = "track"
SWEEP_KEY = "sweep"
WORKSPACE_KEY = "workspace"
CTX_JSON_KEY = "ctx_json"

CTX_TRACK = "ctx"
GEN_TRACK = "gen"
TRACKS: tuple[str, ...] = (CTX_TRACK, GEN_TRACK)

#: The metric each track is scored on. Both are higher-is-better, so
#: neither needs the ``_ms`` suffix rule in ``_normalized_gain_pct``.
#: These are the names ``task.yaml`` carries; ``frontier show`` spells
#: the gen one ``tps_per_user``, and :mod:`.bench_cli` is where the two
#: vocabularies meet.
TRACK_METRICS: dict[str, str] = {
    CTX_TRACK: "avg_request_throughput_req_s",
    GEN_TRACK: "throughput_per_user",
}

#: What a frontier snapshot's ``points[].metrics`` calls the gen metric.
#: Only the gen track has one: ``frontier build`` selects ``stage == GEN``
#: and refuses outright when a workspace holds none, because a snapshot
#: *is* the rate-matched generation curve. The ctx side enters it as the
#: anchor, never as a point — so there is no snapshot key to read a ctx
#: campaign's score out of. See :func:`collect`.
SNAPSHOT_METRICS: dict[str, str] = {GEN_TRACK: "tps_per_user"}

#: Written per operating point under the stage's result directory, so a
#: measurement made by ``bench-disagg`` is discoverable by every part of
#: perf-optimize that globs for result JSONs — the baseline gate first.
SOL_RESULT_NAME = "sol_result.json"

#: Worker keys the sweep row fixes, and which the tuning overlay
#: therefore may not name. Each is a coordinate of the operating point:
#: a gen row is ``[ctx_num, gen_num, tp_size, batch, max_num_tokens,
#: attention_dp, gpu_mem_frac, mtp, eplb, concurrency]``, and a ctx
#: benchmark entry fixes ``max_batch`` and ``tp_size`` the same way.
#:
#: These are the *generated config's* spellings, because that is what the
#: overlay is deep-merged onto — not the sweep row's. Ported from the
#: script-driven implementation, where a dry run caught an override
#: taking ``tensor_parallel_size`` from 4 to 8 and the job from two nodes
#: to three: the run would have succeeded and the number would have
#: looked plausible, with the node count in a log line as the only trace,
#: while the comparison against a baseline taken at the old point was
#: void rather than merely weaker.
FROZEN_WORKER_KEYS = frozenset(
    {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "moe_expert_parallel_size",
        "enable_attention_dp",
        "max_batch_size",
        "max_num_tokens",
    }
)

#: Same reason as :data:`.disagg.DISAGG_PROFILE_METHODS` — the harness
#: only knows how to wrap workers in nsys.
SOL_PROFILE_METHODS: tuple[str, ...] = ("nsys",)

#: The key a sweep stage uses for the overlay deep-merged onto every
#: generated worker config. This is the campaign's tuning surface: it
#: changes what the workers run without changing which points are
#: measured, so case names stay put and ``frontier compare`` can align an
#: attempt against the baseline it is judged against.
TRACK_OVERLAY_KEYS: dict[str, str] = {
    CTX_TRACK: "ctx_extra_llm_api",
    GEN_TRACK: "gen_extra_llm_api",
}


class SolTrackError(ValueError):
    """The track block, or the sweep it names, is unusable."""


# --------------------------------------------------------------- the block


def has_sol_track(data: Mapping[str, Any]) -> bool:
    """Whether the spec enables a SOL track campaign."""
    return SOL_TRACK_FIELD in data


def sol_track_block(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    block = data.get(SOL_TRACK_FIELD)
    return block if isinstance(block, Mapping) else None


def _text(data: Mapping[str, Any], key: str) -> str | None:
    block = sol_track_block(data)
    if block is None:
        return None
    value = block.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def track_name(data: Mapping[str, Any]) -> str | None:
    """Which half this campaign optimizes, or ``None`` if unstated."""
    return _text(data, TRACK_KEY)


def workspace_name(data: Mapping[str, Any]) -> str | None:
    """The ``bench-disagg`` workspace this campaign measures into."""
    return _text(data, WORKSPACE_KEY)


def sweep_path(data: Mapping[str, Any]) -> Path | None:
    """The orchestration ``sweep.yaml``: stages, server config, options."""
    value = _text(data, SWEEP_KEY)
    return None if value is None else Path(value)


def ctx_json_path(data: Mapping[str, Any]) -> Path | None:
    """An existing CTX anchor this campaign builds its frontier against.

    A **gen** campaign cannot be scored without one. ``tps_per_user`` is a
    purely generation-side quantity, but the only thing that reports it is
    ``frontier build``, which rate-matches the whole curve and therefore
    needs the context request rate; with neither a measured ctx stage nor
    this, it raises ``ANCHOR_MISSING`` — *after* the gen jobs have run.

    Pointing at an anchor somebody else measured is legitimate and often
    right: a gen campaign freezes the ctx side by construction, so the
    anchor is a constant. It is not free of consequence, though — the
    build folds the anchor's digest into the snapshot's ``view_id``, so
    ``frontier compare`` will report a curve built against a different
    anchor as not ``comparable``.
    """
    value = _text(data, CTX_JSON_KEY)
    return None if value is None else Path(value)


def anchor_isl(anchor: Path) -> int | None:
    """The input length the ctx anchor was measured at, if it says.

    ``get_ctx_throughput.py`` writes a list of rows, each carrying the
    ``isl`` its measurement ran at alongside the throughput. One anchor
    file can hold several rows (one per MTP size); they come from one
    sweep, so the first that states an ``isl`` is the file's.
    """
    try:
        rows = json.loads(anchor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        return None
    for row in rows:
        value = row.get("isl") if isinstance(row, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def require_matching_anchor(anchor: Path, plan: Mapping[str, Any]) -> str | None:
    """Refuse a borrowed anchor that measured a different request shape.

    The frontier metric divides a decode rate by a prefill rate. Feed it a
    ctx anchor taken at one input length and a gen curve taken at another
    and it still returns a number, on a curve that still looks like a
    frontier, describing a deployment whose two roles were never serving
    the same traffic. Nothing about the output looks wrong — which is what
    makes it the one failure the rest of this module's care would not
    survive.

    Only the input length is compared, and only when the anchor states
    one. A ctx measurement runs at ``osl: 1`` by construction, so an
    output length would never match and comparing it would reject every
    anchor; the dataset path differs legitimately too, since the tracks
    read the ``_for_bench`` and ``_for_serve`` variants of one corpus.
    Returns a note when the anchor is silent, because "not checked" and
    "checked and matched" must not read alike.
    """
    workload = plan.get("workload")
    expected = (workload or {}).get("isl") if isinstance(workload, Mapping) else None
    measured = anchor_isl(anchor)
    if measured is None:
        return (
            f"{anchor} states no 'isl', so nothing verified that it measured this "
            f"campaign's requests — the frontier would rate-match against it either way"
        )
    if isinstance(expected, int) and not isinstance(expected, bool) and measured != expected:
        raise SolTrackError(
            f"the ctx anchor {anchor} was measured at isl {measured}, but this "
            f"campaign's sweep runs at isl {expected}. The frontier divides a decode "
            f"rate by a prefill rate, so it would still produce a number and a curve "
            f"— describing a deployment whose two roles never served the same traffic. "
            f"Point at an anchor measured on this workload, or enable the sweep's ctx "
            f"stage so the campaign measures its own."
        )
    return None


def load_sweep(path: Path) -> dict[str, Any]:
    """Read the orchestration file, or raise with a legible message."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - message path
        raise SolTrackError(f"could not read sol_track sweep {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SolTrackError(
            f"sol_track sweep {path} must be a YAML mapping, got {type(data).__name__}"
        )
    return dict(data)


#: What a stage config needs before a source change can reach the workers.
#: Either names something for the harness to install: a repo it builds in
#: the job, or a wheel someone already built.
BUILD_SOURCE_KEYS = ("trtllm_repo", "trtllm_wheel_path")


def build_source(stage: Mapping[str, Any]) -> dict[str, Any] | None:
    """The stage's ``trtllm_install`` block, if it names one."""
    install = stage.get("trtllm_install")
    if not isinstance(install, Mapping):
        return None
    if not any(str(install.get(key) or "").strip() for key in BUILD_SOURCE_KEYS):
        return None
    return dict(install)


def require_build_source(task_data: Mapping[str, Any], approaches: Any) -> None:
    """Refuse a source-changing campaign whose sweep installs nothing.

    ``bench-disagg`` runs whatever the image already has unless the stage
    config's ``trtllm_install`` names a repo to build or a wheel to
    install. So with ``approach: code`` and no such block, the optimizer
    edits ``trtllm_repo_path``, the job measures the image, the number
    comes back at the baseline, and the evaluator rejects an untested
    change as "no gain" — having spent a full allocation to learn nothing
    about it.

    Nothing downstream can catch that: the run succeeds, the number is
    plausible, and the only trace is that ``code_id`` did not move. So it
    is refused here, before an agent exists.
    """
    wanted = [a for a in (approaches or []) if a == "code"]
    if not wanted:
        return
    track = track_name(task_data)
    sweep_file = sweep_path(task_data)
    if track not in TRACKS or sweep_file is None or not sweep_file.is_file():
        return
    stage_config = stage_config_path(load_sweep(sweep_file), str(track), sweep_file)
    if stage_config is None or not stage_config.is_file():
        return
    if build_source(load_sweep(stage_config)) is None:
        raise SolTrackError(
            f"'optimize.approaches' includes 'code', but {stage_config} names no "
            f"'trtllm_install' with {' or '.join(BUILD_SOURCE_KEYS)}. The harness "
            f"runs whatever the image ships unless one is given, so a source change "
            f"would never reach the workers: the measurement would come back at the "
            f"baseline and the change would be rejected as 'no gain' without having "
            f"been tested. Add to the stage config:\n"
            f"    trtllm_install:\n"
            f"      trtllm_repo: <this campaign's trtllm_repo_path>\n"
            f"      build_wheel: true\n"
            f"or drop 'code' from approaches."
        )


def stage_config_path(sweep: Mapping[str, Any], track: str, sweep_file: Path) -> Path | None:
    """The stage config for ``track``, resolved relative to the sweep file.

    The sweep's own convention: "Paths resolve relative to THIS file."
    """
    stages = sweep.get("stages")
    stage = stages.get(track) if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping):
        return None
    value = stage.get("config")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    return path if path.is_absolute() else (sweep_file.parent / path)


def sweep_accept_rate(sweep: Mapping[str, Any]) -> str | None:
    """The frozen acceptance length from the sweep's ``options``.

    It lives there rather than in ``task.yaml`` because it belongs to the
    measurement, not to the campaign: ``frontier build`` requires it and
    refuses to infer one, on the grounds that a wrong value tilts the
    whole curve with no symptom.
    """
    options = sweep.get("options")
    if not isinstance(options, Mapping):
        return None
    value = options.get("accept_rate")
    return value.strip() if isinstance(value, str) and value.strip() else None


# --------------------------------------------------------------- the copy


#: Where a campaign keeps the sweep it actually measured.
WORKSPACE_SWEEP_DIR = "sweep"


def adopt_sweep(task_data: dict[str, Any], workspace: Path) -> Path | None:
    """Copy the sweep into the workspace and point the spec at the copy.

    ``apply_overlay`` writes this campaign's tuning into a sweep stage
    config before every submit. Writing that into the file the user named
    is a mistake with a long tail: nothing restores it, so the *next*
    campaign seeds its tuning file from the *previous* one's accepted
    overlay and calls the result a baseline -- measured with an
    optimization applied, reported as the sweep's own. The rewrite also
    drops every comment in that file, and any key the user had under the
    overlay key.

    The disagg campaign already answered this: it synthesizes a fresh
    config into its own artifact directory rather than editing the harness
    config it was handed. The only difference here is that a sweep is a
    *directory* -- a stage config resolves its siblings relative to
    itself, and the harness imports the model's ``gen_worker_config.py``
    from beside it -- so the unit of copy is the directory, not the file.

    Copying also makes the campaign self-describing: ``<workspace>/sweep/``
    is what this run measured, not whatever the original has become since.

    Returns the copied sweep file, or ``None`` when there is nothing to
    adopt. Idempotent by design: an existing copy is left alone, so a
    resumed run keeps measuring what it started with, and ``--clean``
    (which removes the workspace) is what starts over from the original.
    """
    source = sweep_path(task_data)
    if source is None or not source.is_file():
        return None
    destination = workspace / WORKSPACE_SWEEP_DIR
    target = destination / source.name
    if not destination.exists():
        shutil.copytree(
            source.parent,
            destination,
            # Build artifacts of the harness' own import of the model's
            # config generator, and of previous runs. Copying them wastes
            # space and, for `__pycache__`, risks a stale module.
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )
    if not target.is_file():
        raise SolTrackError(
            f"the sweep copy at {destination} has no {source.name}; remove it and "
            f"re-run, or pass --clean to rebuild the workspace from {source}"
        )
    block = dict(task_data.get(SOL_TRACK_FIELD) or {})
    block[SWEEP_KEY] = str(target)
    block["adopted_from"] = str(source)
    task_data[SOL_TRACK_FIELD] = block
    return target


# --------------------------------------------------------------- the tuning seed


def tuning_seed_yaml(data: Mapping[str, Any]) -> str:
    """The live tuning file's seed: this track's worker overlay.

    Deliberately **one role's** overlay, and deliberately a *partial* one.
    ``bench-disagg`` deep-merges it onto the worker config the sweep row
    generated, so the row keeps the knobs that define the operating point
    — parallel sizes, batch, token limits — and the optimizer edits only
    what it is actually tuning. The previous shape, a whole copy of the
    role's worker config, put the frozen topology inside the file the
    optimizer edits, which is an invitation to change it by accident.

    Empty is the normal starting point: most sweeps carry no overlay.
    """
    track = track_name(data)
    if track not in TRACKS:
        raise SolTrackError(f"unknown sol_track track {track!r}, expected one of {list(TRACKS)}")
    path = sweep_path(data)
    if path is None:
        raise SolTrackError(f"'{SOL_TRACK_FIELD}.{SWEEP_KEY}' is required")
    sweep = load_sweep(path)
    stage_config = stage_config_path(sweep, track, path)
    overlay: Any = None
    if stage_config is not None and stage_config.is_file():
        overlay = load_sweep(stage_config).get(TRACK_OVERLAY_KEYS[track])
    if overlay is None:
        overlay = {}
    if not isinstance(overlay, Mapping):
        raise SolTrackError(
            f"'{TRACK_OVERLAY_KEYS[track]}' in {stage_config} must be a mapping, "
            f"got {type(overlay).__name__}"
        )
    return yaml.safe_dump(dict(overlay), sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------- reconciliation


def _reconcile(
    task_data: dict[str, Any],
    expected: Mapping[str, Any],
    why: Mapping[str, str],
    user_set: set[str],
) -> list[str]:
    """Fill what the user left out, error on what they wrote differently.

    The same rule :mod:`.disagg` states and for the same reason: a
    ``task.yaml`` whose stated operating point is quietly replaced is a
    file you cannot read, and "my setting did nothing" is the failure
    this avoids.
    """
    benchmark = dict(task_data.get("benchmark") or {})
    notes: list[str] = []
    conflicts: list[str] = []
    for key, value in expected.items():
        if key in user_set and benchmark.get(key) != value:
            conflicts.append(
                f"'benchmark.{key}' is {benchmark.get(key)!r} but the sweep gives "
                f"{value!r} ({why[key]})"
            )
        elif key not in user_set:
            benchmark[key] = value
            notes.append(f"benchmark.{key}={value!r} from the sweep ({why[key]})")
    if conflicts:
        bullet = "\n  - "
        raise SolTrackError(
            f"task.yaml contradicts the sweep, which owns the measurement "
            f"conditions:{bullet}{bullet.join(conflicts)}{bullet}"
            f"remove these keys to take the sweep's values."
        )
    task_data["benchmark"] = benchmark
    return notes


def apply_plan(task_data: dict[str, Any], plan: Mapping[str, Any], user_set: set[str]) -> list[str]:
    """Reconcile ``task_data`` against a ``sweep plan`` envelope, in place.

    The plan is the authority because it is the same expansion that will
    run: its ``workload`` block names the corpus the client will read, and
    its ``cases`` are the points that will be measured, already multiplied
    out from whatever the sweep rows listed.

    **The plan's ``isl`` is not the workload's input length.** It is a
    sizing bound and a client argument — ``submit.py`` spends it on
    ``ctx_max_seq_len = isl + offset``, ``gen_max_seq_len = isl + osl +
    offset``, ``benchmark.input_length`` and the job name — while the
    requests themselves come from ``workload.dataset``, a corpus with its
    own length distribution. The checked-in reference sweep for this
    campaign pairs ``isl: 200000`` with a ``...-coding-c190000-...``
    dataset, and both numbers are right: one bounds the KV allocation, the
    other describes the traffic.

    So a dataset run is reconciled as a dataset run. Copying ``isl`` into
    ``benchmark.random_input_len`` asserted two things that are not true —
    that the dataset is synthetic (``dataset_name`` defaults to
    ``random``) and that every request is exactly that long — and every
    prompt, the analyzer's reasoning and the report then quoted them.
    """
    track = track_name(task_data)
    if track not in TRACKS:
        raise SolTrackError(f"unknown sol_track track {track!r}, expected one of {list(TRACKS)}")

    stage_cases = [case for case in plan.get("cases") or [] if case.get("stage") == track]
    if not stage_cases:
        raise SolTrackError(
            f"the sweep plans no '{track}' cases. Enable that stage in the sweep's "
            f"`stages:` block, or point this campaign at the track the sweep measures."
        )
    points = operating_points({"cases": stage_cases})
    if not points:
        raise SolTrackError(f"no '{track}' case in the plan carries a usable concurrency")

    workload = plan.get("workload")
    workload = workload if isinstance(workload, Mapping) else {}
    expected: dict[str, Any] = {"concurrency": points[0] if len(points) == 1 else points}
    why = {
        "concurrency": (
            "each planned case's max_batch, the in-flight request count of a prefill-only run"
            if track == CTX_TRACK
            else "each planned case's concurrency x its gen_num"
        )
    }
    dataset = workload.get("dataset")
    dataset = dataset.strip() if isinstance(dataset, str) and dataset.strip() else None
    isl, osl = workload.get("isl"), workload.get("osl")
    if dataset is not None:
        expected["dataset_name"] = Path(dataset).name
        expected["dataset_path"] = dataset
        why["dataset_name"] = "the plan's workload.dataset — the corpus the client reads"
        why["dataset_path"] = "the plan's workload.dataset, resolved by the harness"
    else:
        # No corpus named, so the lengths really are the request shape.
        # Every sweep this has been run against names one; the branch is
        # here because a template that leaves the dataset per-row exists.
        if isinstance(isl, int) and not isinstance(isl, bool):
            expected["random_input_len"] = isl
            why["random_input_len"] = "the plan's workload.isl, with no dataset to read from"
        if isinstance(osl, int) and not isinstance(osl, bool):
            expected["random_output_len"] = osl
            why["random_output_len"] = "the plan's workload.osl, with no dataset to read from"

    notes = _reconcile(task_data, expected, why, user_set)

    if dataset is not None:
        # The defaults block seeds `random_input_len` / `random_output_len`
        # for every campaign, and on a dataset run they describe nothing —
        # keeping them is how "1024" or an isl ends up quoted as this
        # campaign's input length. Dropped rather than corrected, because
        # the honest value is a distribution the sweep does not carry.
        # A user who wrote one keeps it: `_reconcile` already refuses a
        # value that contradicts the sweep, and this is not that.
        benchmark = task_data["benchmark"]
        for key in ("random_input_len", "random_output_len"):
            if key not in user_set and benchmark.pop(key, None) is not None:
                notes.append(f"benchmark.{key} dropped: this campaign reads {dataset}")
        notes.append(
            f"workload.isl={isl!r} / osl={osl!r} are the harness' sequence-length "
            f"bounds (ctx_max_seq_len = isl + offset, gen_max_seq_len = isl + osl + "
            f"offset) and the client's input_length — NOT the corpus' measured "
            f"lengths. Quote the dataset, not these."
        )

    # Checked here rather than beside the other anchor rules because it
    # needs the plan: the sweep's own isl is what the anchor has to match,
    # and the plan is where that arrives already resolved.
    anchor = ctx_json_path(task_data)
    if anchor is not None and anchor.is_file():
        unverified = require_matching_anchor(anchor, plan)
        notes.append(unverified or f"ctx anchor {anchor} measured this campaign's isl")

    # TODO: derive the nsys window from the baseline instead of taking it
    # from the spec. ``profile.nsys_iter_range`` is a dead field on both
    # SOL tracks -- two real campaigns each asked for 100-150 and each
    # captured something else (the gen harness fired 200-250, the ctx one
    # 30-50), and on the ctx run the whole benchmark was only 100
    # iterations, so the requested window could not have caught anything
    # even if it had been honoured. A window is only meaningful relative
    # to how many iterations the run actually has, which the benchmarker
    # measures; a number typed in before the baseline exists is a guess
    # that nothing reconciles. Until then the field is left alone rather
    # than quietly rewritten, so at least the spec and the trace disagree
    # visibly. (Same shape as the disagg block's open A3.)
    profile = dict(task_data.get("profile") or {})
    dropped = [m for m in (profile.get("methods") or []) if m not in SOL_PROFILE_METHODS]
    profile["methods"] = list(SOL_PROFILE_METHODS)
    profile.pop("kernel_coverage", None)
    task_data["profile"] = profile
    if dropped:
        notes.append(
            f"profile.methods {dropped} dropped: the benchmark harness only wraps "
            f"workers in nsys (no torch-profiler env var, no ncu path)"
        )

    if task_data.pop("accuracy", None) is not None:
        # The same call disagg makes (`disagg.py:257`). QA is told to run the
        # accuracy `command` verbatim against the live server -- and this
        # campaign's own prompt section opens by saying it never launches
        # one. Left in place, the agent either invents a server or invents a
        # score.
        notes.append(
            "accuracy block ignored: a SOL track never stands up a server for you to "
            "query, and the harness has no accuracy pass of its own on either track"
        )

    optimize = task_data.get("optimize")
    optimize = dict(optimize) if isinstance(optimize, Mapping) else {}
    if optimize.get("target_metric") is None:
        optimize["target_metric"] = TRACK_METRICS[track]
        notes.append(
            f"optimize.target_metric={TRACK_METRICS[track]!r}, what the {track} track is scored on"
        )
    task_data["optimize"] = optimize

    # Recorded so "which code were these points planned against" is
    # answerable from the resolved spec, not only from the workspace.
    notes.append(f"code_id at plan time: {plan.get('code_id')!r}")
    return notes


# --------------------------------------------------------------- the overlay


def apply_overlay(task_data: Mapping[str, Any], tuning: Path) -> Path:
    """Put the live tuning file where ``bench-disagg`` will read it.

    The one piece of glue that stays ours. ``bench-disagg`` has no notion
    of "this campaign's live tuning file" — it reads the stage config it
    is handed — while perf-optimize's whole diff / revert / accepted-
    snapshot machinery is built around exactly one file the optimizer
    edits. So the overlay is copied from that file into the stage config's
    role key before every submit.

    Doing it here rather than in a prompt is the same argument as
    everything else this campaign moved into code: forgetting it does not
    fail, it measures the previous attempt's configuration and books the
    result against the new one. It is also what makes the tuning edit show
    up in ``code_id``, since the CLI digests that key into the code
    fingerprint — which is what keeps the case names, and therefore
    ``frontier compare``'s alignment, unchanged.

    Returns the stage config it wrote.
    """
    track = track_name(task_data)
    if track not in TRACKS:
        raise SolTrackError(f"unknown sol_track track {track!r}, expected one of {list(TRACKS)}")
    sweep_file = sweep_path(task_data)
    if sweep_file is None:
        raise SolTrackError(f"'{SOL_TRACK_FIELD}.{SWEEP_KEY}' is required")
    stage_config = stage_config_path(load_sweep(sweep_file), track, sweep_file)
    if stage_config is None or not stage_config.is_file():
        raise SolTrackError(f"the sweep names no readable '{track}' stage config: {stage_config}")
    overlay = yaml.safe_load(tuning.read_text(encoding="utf-8"))
    if overlay is None:
        overlay = {}
    if not isinstance(overlay, Mapping):
        raise SolTrackError(
            f"{tuning} must be a mapping — it is deep-merged onto the worker config "
            f"the sweep row generates — got {type(overlay).__name__}"
        )
    overreach = sorted(FROZEN_WORKER_KEYS.intersection(overlay))
    if overreach:
        raise SolTrackError(
            f"{tuning} names {overreach}, which the sweep row fixes for this "
            f"campaign. Those keys ARE the operating point, so changing one measures "
            f"a different point — a different world size, a different batch — against "
            f"a baseline taken at the old one. The comparison would be void rather "
            f"than merely weaker, and nothing downstream would notice: the run "
            f"succeeds and the number is plausible. Move the point by editing the "
            f"sweep row if that is really what you mean, which starts a new campaign."
        )
    stage = load_sweep(stage_config)
    key = TRACK_OVERLAY_KEYS[track]
    if overlay:
        stage[key] = dict(overlay)
    else:
        stage.pop(key, None)
    stage_config.write_text(
        yaml.safe_dump(stage, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    return stage_config


# --------------------------------------------------------------- the score


def target_metric(task_data: Mapping[str, Any]) -> str:
    """What this campaign is scored on, and therefore what to write.

    Not ``TRACK_METRICS[track]``. ``apply_plan`` only *defaults* the target
    metric, so an owner who names another one keeps it — and every later
    stage, the baseline gate first, then looks up that name. Writing the
    track's own spelling instead produces a file the gate cannot see, and
    the gate's message says the stage measured nothing, which is the one
    thing it certainly did.
    """
    optimize = task_data.get("optimize")
    named = optimize.get("target_metric") if isinstance(optimize, Mapping) else None
    if isinstance(named, str) and named.strip():
        return named.strip()
    track = track_name(task_data)
    if track not in TRACKS:
        raise SolTrackError(f"unknown sol_track track {track!r}, expected one of {list(TRACKS)}")
    return TRACK_METRICS[track]


def elasticity(
    metrics: Mapping[str, Any], config: Mapping[str, Any], concurrency: int
) -> float | None:
    """How much of a gen improvement survives into the e2e frontier.

    The two numbers a gen campaign holds are not the same objective. The
    gate scores ``tps_per_user``, which is anchor-free; the deployment is
    judged on ``tps_per_gpu``, which is not::

        tps_per_gpu = tps_per_user * concurrency / (ctx_gpus * ctx_per_gen + ep_rank)

    and ``ctx_per_gen`` rises with ``tps_per_user`` — a faster generation
    side consumes prefill faster, so it needs proportionally more context
    GPUs behind it. Differentiating, a 1 % gain in the gate's metric is
    worth ``ep_rank / denominator`` per cent at the frontier, which
    approaches zero as the context side becomes the wall.

    On this branch's own campaign that ratio is **0.97 at concurrency 1**
    and **0.70 at 32** — so the same measured +1 % means materially
    different things at the two ends of one curve, and nothing in the
    report said so. Recorded rather than applied: the gate stays the
    track's own metric, and the evaluator is handed the exchange rate.

    The denominator is recovered from the snapshot rather than assumed:
    ``ctx_gpus`` is not in a gen case's config, but the ratio of the two
    metrics is exactly it.
    """
    per_user = metrics.get("tps_per_user")
    per_gpu = metrics.get("tps_per_gpu")
    gen_num = config.get("gen_num", 1)
    tp_size = config.get("tp_size")
    for value in (per_user, per_gpu, tp_size):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            return None
    if not isinstance(gen_num, int) or isinstance(gen_num, bool) or gen_num < 1:
        gen_num = 1
    denominator = per_user * concurrency / per_gpu
    gen_gpus = tp_size * gen_num
    if denominator < gen_gpus:
        # The identity says the denominator is `ctx_gpus * ctx_per_gen +
        # gen_gpus` with both terms positive, so this cannot happen unless
        # the shape changed. Silence beats a ratio above 1, which would read
        # as "a gen gain is worth MORE at the frontier".
        return None
    return gen_gpus / denominator


def _write_result(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _place(written: dict[int, Path], cases: dict[int, str], concurrency: int, case: str) -> None:
    """Refuse two cases at one operating point rather than overwrite one."""
    if concurrency in written:
        raise SolTrackError(
            f"cases {cases[concurrency]!r} and {case!r} both measure {concurrency} "
            f"requests in flight, so they cannot both be "
            f"'concurrency_{concurrency}'. A SOL-track campaign's operating points "
            f"have to be addressable by concurrency alone, the way an aggregate "
            f"campaign's are. Split the sweep so each point sits at its own "
            f"concurrency -- for a ctx sweep that usually means one campaign per "
            f"`mtp_range` entry, since those share a `max_batch`."
        )


def collect(task_data: Mapping[str, Any], into: Path, *, snapshot: str = "latest") -> list[Path]:
    """Write this attempt's score where perf-optimize looks for a result.

    The other half of :func:`apply_overlay`, and the same argument. The
    overlay is how a campaign's tuning reaches ``bench-disagg``; this is
    how ``bench-disagg``'s answer comes back. Between them the CLI owns
    the measurement entirely and this workflow owns none of it.

    It exists because the two sides name the same quantity differently and
    keep it in different places. Every stage of perf-optimize — starting
    with the baseline gate, which stops a campaign that measured nothing
    rather than replaying a broken config once per stage — reads
    ``optimize.target_metric`` out of a JSON under ``concurrency_<c>/``.
    Without this the flow's own gate rejects a measurement that succeeded,
    which is the worst of the failures available: it looks exactly like
    the failure it was built to catch.

    The two tracks are read from different places, because the CLI reports
    them from different places:

    - **gen** — from the frontier snapshot, whose ``tps_per_user`` becomes
      ``throughput_per_user``. The snapshot is *checked against this
      attempt* first (see :func:`_require_attempt_snapshot`).
    - **ctx** — from ``sweep status``, whose ``result`` is the very
      ``run_*.json`` the backend validated the case on. No snapshot exists
      to read: ``frontier build`` selects ``stage == GEN`` and refuses a
      workspace holding none, taking the ctx side as its anchor.

    Returns the files written, one per operating point.
    """
    track = track_name(task_data)
    if track not in TRACKS:
        raise SolTrackError(f"unknown sol_track track {track!r}, expected one of {list(TRACKS)}")
    workspace = workspace_name(task_data)
    if workspace is None:
        raise SolTrackError(f"'{SOL_TRACK_FIELD}.{WORKSPACE_KEY}' is required")
    metric = target_metric(task_data)
    if track == GEN_TRACK:
        return _collect_gen(task_data, workspace, into, metric, snapshot)
    return _collect_ctx(workspace, into, metric)


def _require_attempt_snapshot(view: Mapping[str, Any], task_data: Mapping[str, Any]) -> None:
    """Refuse a snapshot that was not built from this attempt's overlay.

    ``--snapshot latest`` resolves whatever complete snapshot the
    workspace holds, and nothing about that says it is *this* attempt's.
    An agent that skipped the build, or whose build failed, still gets a
    successful collect writing the PREVIOUS attempt's numbers into this
    attempt's directory. The evaluator then measures a delta of zero and
    rejects a change that was never run — spending the attempt budget to
    learn nothing, and recording a verdict about code nobody benchmarked.

    The check costs no extra call: a snapshot carries the
    ``worker_overrides`` each of its measurements ran under, and the live
    tuning file is the overlay this attempt asked for. If they differ, the
    snapshot predates the overlay.
    """
    track = track_name(task_data)
    tuning = _live_overlay(task_data)
    if tuning is None:
        return
    key = TRACK_OVERLAY_KEYS[str(track)]
    detail = (view.get("code") or {}).get("detail") or {}
    seen = [dict((entry.get("worker_overrides") or {}).get(key) or {}) for entry in detail.values()]
    if seen and not any(overlay == tuning for overlay in seen):
        raise SolTrackError(
            f"snapshot {view.get('snapshot_id')!r} was built from measurements whose "
            f"'{key}' is {seen!r}, but this attempt's tuning file asks for {tuning!r}. "
            f"The snapshot predates the overlay, so scoring it would book the "
            f"previous attempt's numbers against this one. Run `bench-disagg "
            f"frontier build` after the submit that carried this overlay."
        )


def _live_overlay(task_data: Mapping[str, Any]) -> dict[str, Any] | None:
    """This attempt's overlay, read back off the stage config it was written to."""
    track = track_name(task_data)
    sweep_file = sweep_path(task_data)
    if track not in TRACKS or sweep_file is None or not sweep_file.is_file():
        return None
    stage_config = stage_config_path(load_sweep(sweep_file), str(track), sweep_file)
    if stage_config is None or not stage_config.is_file():
        return None
    value = load_sweep(stage_config).get(TRACK_OVERLAY_KEYS[str(track)])
    return dict(value) if isinstance(value, Mapping) else {}


def _collect_gen(
    task_data: Mapping[str, Any], workspace: str, into: Path, metric: str, snapshot: str
) -> list[Path]:
    view = frontier_show(workspace, snapshot)
    _require_attempt_snapshot(view, task_data)
    code = view.get("code") or {}
    written: dict[int, Path] = {}
    cases: dict[int, str] = {}
    skipped: list[str] = []

    for point in view.get("points") or []:
        case = str(point.get("case") or "?")
        metrics = point.get("metrics")
        metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
        config = dict(point.get("config") or {})
        value = metrics.get(SNAPSHOT_METRICS[GEN_TRACK])
        concurrency = operating_point(config)
        if concurrency is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            skipped.append(case)
            continue
        _place(written, cases, concurrency, case)
        written[concurrency] = _write_result(
            into / f"concurrency_{concurrency}" / SOL_RESULT_NAME,
            {
                # First, and under the name `optimize.target_metric`
                # carries: this key is the whole point of the file.
                metric: float(value),
                "concurrency": concurrency,
                "case": case,
                "on_frontier": point.get("frontier"),
                "samples": point.get("samples"),
                "config_id": point.get("config_id"),
                "code_id": point.get("code_id"),
                "snapshot_id": view.get("snapshot_id"),
                # `best` and `latest` are different questions, and a number
                # read without knowing which was asked is not comparable to
                # the one beside it.
                "select": view.get("select"),
                # The prompt tells the evaluator a `code.mixed` curve is not
                # evidence. Recorded rather than refused: the mixing is a
                # property of the CURVE, while this point's own `code_id` is
                # definite and already checked -- so the fact belongs in the
                # file, and the judgement belongs to the reader.
                "code_mixed": code.get("mixed"),
                "code_ids": code.get("ids"),
                # What a per-cent here is worth at the frontier.
                "frontier_elasticity": elasticity(metrics, config, concurrency),
                "snapshot_metrics": metrics,
            },
        )
        cases[concurrency] = case

    if not written:
        raise SolTrackError(
            f"snapshot {view.get('snapshot_id')!r} of workspace {workspace!r} scored no "
            f"usable point" + (f" (skipped: {', '.join(skipped)})" if skipped else "") + ". "
            f"`bench-disagg frontier show --workspace {workspace}` is the snapshot as "
            f"built; `bench-disagg sweep status --workspace {workspace} --cases` says "
            f"whether the cases behind it measured."
        )
    return [written[key] for key in sorted(written)]


#: Where a validated ctx measurement keeps its number. This is the key the
#: backend itself requires before it will call a ctx case successful, so a
#: case reported as `success` is a case that has it.
CTX_RESULT_PATH = ("performance", "request_throughput_req_s")


def _collect_ctx(workspace: str, into: Path, metric: str) -> list[Path]:
    written: dict[int, Path] = {}
    cases: dict[int, str] = {}
    skipped: list[str] = []

    for case in status(workspace).get("cases") or []:
        name = str(case.get("case") or "?")
        if case.get("stage") != CTX_TRACK:
            continue
        result = case.get("result")
        concurrency = operating_point(case.get("config") or {})
        value = _read_ctx_result(result) if isinstance(result, str) else None
        if concurrency is None or value is None:
            skipped.append(f"{name} ({case.get('state')})")
            continue
        _place(written, cases, concurrency, name)
        written[concurrency] = _write_result(
            into / f"concurrency_{concurrency}" / SOL_RESULT_NAME,
            {
                metric: value,
                "concurrency": concurrency,
                "case": name,
                "config": dict(case.get("config") or {}),
                "state": case.get("state"),
                "result": result,
                "workdir": case.get("workdir"),
            },
        )
        cases[concurrency] = name

    if not written:
        raise SolTrackError(
            f"workspace {workspace!r} holds no validated ctx measurement to score"
            + (f" (skipped: {'; '.join(skipped)})" if skipped else "")
            + f". `bench-disagg sweep status --workspace {workspace} --cases` reports "
            f"whether the jobs are queued, failed, or produced artifacts that did not "
            f"validate."
        )
    return [written[key] for key in sorted(written)]


def _read_ctx_result(path: str) -> float | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in CTX_RESULT_PATH:
        payload = payload.get(key) if isinstance(payload, Mapping) else None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin entry
    import argparse

    parser = argparse.ArgumentParser(description=apply_overlay.__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, help="the perf-optimize workspace")
    parser.add_argument(
        "--collect",
        metavar="RESULT_DIR",
        default=None,
        help="instead of applying the overlay, write the latest frontier "
        "snapshot's score into RESULT_DIR/concurrency_<c>/" + SOL_RESULT_NAME,
    )
    parser.add_argument(
        "--snapshot",
        default="latest",
        help="with --collect: a snapshot id, 'latest', or 'baseline' (default: latest)",
    )
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    task_data = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
    if args.collect:
        results = collect(task_data, Path(args.collect), snapshot=args.snapshot)
        print(json.dumps({"ok": True, "results": [str(path) for path in results]}))
        return 0
    written = apply_overlay(task_data, workspace / "tuning" / "extra_llm_api_options.yaml")
    print(json.dumps({"ok": True, "stage_config": str(written)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    import sys

    try:
        sys.exit(_main())
    except (SolTrackError, BenchCliError) as exc:
        import sys as _sys

        # `--collect` shells out, so the CLI's own failures surface here
        # too. Carrying `error.code` through means the caller can still
        # branch on the taxonomy rather than on message text — NO_DATA
        # ("nothing measured yet") and ANCHOR_MISSING ("build it against a
        # ctx anchor") ask for different next moves, and a traceback that
        # collapsed them would send a reader to the wrong one.
        report: dict[str, Any] = {"ok": False, "error": str(exc)}
        code = getattr(exc, "code", None)
        if code:
            report["code"] = code
        print(json.dumps(report), file=_sys.stderr)
        _sys.exit(1)
