"""Staged disagg optimization: fix the operating point, then optimize at it.

A SOL-track campaign (see :mod:`.sol_track`) optimizes at an operating point
it is *handed*. That is the right division of labour, and it has one failure
mode: nothing checks where the point came from. The two campaigns this module
was written for inherited row 8 of a checked-in sweep — a row of unknown
provenance — and spent seven hours and thirteen submissions improving a point
nobody had measured. A gain at the wrong point is worth less than moving to
the right one, and neither campaign could have told the difference.

So this module is the layer above a campaign: it **establishes the point**,
and only then starts the campaigns that optimize at it.

**It is not a merge layer.** The two tracks stay what they are — two
independent campaigns, two workspaces, two checkouts, two branches, each
scored on its own metric, neither able to see the other. Nothing here
computes an end-to-end number, and that is a scope decision rather than an
oversight: see :data:`NO_JOIN`.

The staging, in dependency order:

1. **Model facts** — weight bytes, expert count, layers, read from the
   checkpoint. Free, and the input to every shape decision below.
2. **Shape space** — which (parallelism, world size) combinations are even
   legal for this model on this GPU. Free.
3. **Max-batch probe** — for each shape, the largest batch that fits. This
   cannot be computed, only measured: it is where the memory wall is.
4. **Concurrency sweep** — each shape at its *measured* batch, across its
   concurrency ladder, scored **anchor-free** (:func:`.bench_cli.gen_only_points`).
5. **Selection** — pick the point the campaigns will freeze on.

Steps 1–4 are `create-sweep`'s Phases 0–3, minus the frontier build. This
module does not reimplement them; it reads what they wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from agent_flow.workflows.perf_optimize.bench_cli import (
    GEN_ONLY_CSV,
    BenchCliError,
    ctx_cases,
    gen_cases,
    gen_only_points,
)
from agent_flow.workflows.perf_optimize.sweep_design import designer_instruction, load_yaml

DISAGG_SOL_FIELD = "disagg_sol"
TRACKS_KEY = "tracks"
DESIGN_KEY = "design"
DESIGN_DIR_KEY = "design_dir"
PREFER_KEY = "prefer"
DESIGN_SWEEP_KEY = "design_sweep"

#: What `create-sweep` calls its state file. Read, never written: this module
#: is a consumer of that skill's output, not a reimplementation of it.
DESIGN_STATE = "state.json"

#: The two halves. Ordered ctx-first because that is the order the harness
#: itself uses when both are present, not because one gates the other — under
#: this module's scope neither does.
CTX_TRACK = "ctx"
GEN_TRACK = "gen"
TRACKS: tuple[str, ...] = (CTX_TRACK, GEN_TRACK)

#: Why the selection frontier is not the deployment frontier.
#:
#: Stated in the selection result, and required to survive into any report
#: quoting it. `output_tps_per_gen_gpu` divides throughput by the GENERATION
#: GPUs alone. The deployment number divides by the whole rate-matched pair::
#:
#:     output_tput_per_gpu = output_throughput / (ctx_gpus * ctx_per_gen + gen_gpus)
#:
#: so a large-expert-parallel shape can lead this ranking while trailing on
#: deployment cost, because the context GPUs it drags behind it are not in the
#: denominator. Recording the reason rather than the omission, because "no
#: end-to-end view was taken" and "the end-to-end view was flat" are different
#: findings that a missing field cannot distinguish.
NO_JOIN = (
    "selection ranked on output_tps_per_gen_gpu, which divides by the GENERATION "
    "GPUs alone. This is NOT the deployment frontier: output_tput_per_gpu divides "
    "by ctx_gpus * ctx_per_gen + gen_gpus, and no context measurement was rate-"
    "matched against these points. A shape that drags more context GPUs behind it "
    "therefore ranks better here than it would deploy. Absent, not flat -- do not "
    "quote any number selected this way as an end-to-end result."
)

#: What a one-point campaign cannot see, recorded with the point it froze.
#:
#: The campaign gates on a single operating point, so a change is judged only
#: where it was measured. Nothing here is wrong with that — it is the cheapest
#: honest gate — but it does mean a change that helps at the frozen point and
#: hurts elsewhere on the curve is indistinguishable from one that helps
#: everywhere. That is not hypothetical: the campaign this module supersedes
#: measured `opt-006` at **+1.52 % on one point and -1.85 % on another**, and
#: was only saved from accepting it by having frozen both. With one point, the
#: same attempt is an accept.
#:
#: Stated rather than guarded, because which points a deployment cares about
#: is the same exogenous question :data:`PREFERENCES` answers, and a guard
#: here would be this module inventing an answer to it.
ONE_POINT = (
    "one operating point was frozen, so every gain and regression this campaign "
    "reports is measured only there. A change that helps at this point and hurts "
    "elsewhere on the curve is indistinguishable here from one that helps "
    "everywhere -- the rest of the curve was not re-measured after any attempt. "
    "Unobserved, not unchanged."
)

#: What the selection is allowed to optimize for. Exogenous on purpose: a
#: frontier states the trade-off and cannot state which end of it the
#: deployment is bought for, so the campaign's owner says.
PREFER_INTERACTIVE = "interactive"  # max tokens/s/user
PREFER_THROUGHPUT = "throughput"  # max tokens/s/gen-GPU
PREFERENCES: tuple[str, ...] = (PREFER_INTERACTIVE, PREFER_THROUGHPUT)

#: The axes, in the names `gen_only_perf.csv` carries them through
#: :func:`.bench_cli.gen_only_points`.
X_AXIS = "throughput_per_user"
Y_AXIS = "output_tps_per_gen_gpu"


class DisaggSolError(ValueError):
    """The staged-disagg block, or the design it points at, is unusable."""


# ------------------------------------------------------------------ the block


def has_disagg_sol(data: Mapping[str, Any]) -> bool:
    """Whether this spec is a staged two-track campaign."""
    return DISAGG_SOL_FIELD in data


def _block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    block = data.get(DISAGG_SOL_FIELD)
    if not isinstance(block, Mapping):
        raise DisaggSolError(f"'{DISAGG_SOL_FIELD}' must be a mapping")
    return block


def tracks(data: Mapping[str, Any]) -> list[str]:
    """The halves this spec will optimize, validated.

    Defaults to both. A single-track list is legal and means "run one
    campaign, but still fix the point first" — the staging is the point of
    this module, not the plurality.
    """
    listed = _block(data).get(TRACKS_KEY, list(TRACKS))
    if isinstance(listed, str):
        listed = [listed]
    if not isinstance(listed, Sequence) or not listed:
        raise DisaggSolError(f"'{DISAGG_SOL_FIELD}.{TRACKS_KEY}' must be a non-empty list")
    unknown = [t for t in listed if t not in TRACKS]
    if unknown:
        raise DisaggSolError(
            f"'{DISAGG_SOL_FIELD}.{TRACKS_KEY}' names {unknown}, expected any of {list(TRACKS)}"
        )
    seen: list[str] = []
    for track in listed:
        if track not in seen:
            seen.append(track)
    return seen


def design_dir(data: Mapping[str, Any]) -> Path:
    """Where `create-sweep` left the design this campaign freezes on."""
    value = _block(data).get(DESIGN_KEY)
    value = value if isinstance(value, Mapping) else {}
    stated = value.get(DESIGN_DIR_KEY)
    if not isinstance(stated, str) or not stated.strip():
        raise DisaggSolError(
            f"'{DISAGG_SOL_FIELD}.{DESIGN_KEY}.{DESIGN_DIR_KEY}' is required: the "
            f"`sweep_design/` directory `create-sweep` wrote, which is where the "
            f"measured max batch and the measured concurrency ladders live. A "
            f"campaign that cannot name one is inheriting its operating point."
        )
    return Path(stated.strip())


def preference(data: Mapping[str, Any]) -> str:
    """Which end of the frontier this deployment is bought for.

    No default. A frontier is a trade-off between tokens/s/user and
    tokens/s/GPU; which end matters is a property of the service being run,
    not of the measurement, and a campaign that guesses it optimizes toward
    an operating point nobody asked for. The two campaigns this module
    replaces froze concurrency 1 *and* 32 — opposite ends of one curve —
    without ever saying which one the deployment was for.
    """
    value = _block(data).get(DESIGN_KEY)
    value = value if isinstance(value, Mapping) else {}
    stated = value.get(PREFER_KEY)
    if stated not in PREFERENCES:
        raise DisaggSolError(
            f"'{DISAGG_SOL_FIELD}.{DESIGN_KEY}.{PREFER_KEY}' must be one of "
            f"{list(PREFERENCES)}. The measured curve states the trade-off between "
            f"{X_AXIS} and {Y_AXIS}; it cannot state which end this deployment is "
            f"bought for, so nothing here will infer it."
        )
    return str(stated)


# ---------------------------------------------------------------- the design


def design_state(directory: Path) -> dict[str, Any]:
    """`create-sweep`'s own record of how far it got, read not written."""
    path = Path(directory) / DESIGN_STATE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DisaggSolError(
            f"no {DESIGN_STATE} under {directory}: `create-sweep` records its phase "
            f"and artefact paths there after every phase, so its absence means the "
            f"design was never run — or was run somewhere else."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DisaggSolError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DisaggSolError(f"{path} must be a JSON object, got {type(data).__name__}")
    return dict(data)


def sweep_points(directory: Path) -> list[dict[str, Any]]:
    """Every measured point the design left, across every shape.

    One `gen_only_perf.csv` per run directory, and one run directory per
    shape — so the union is the measured space, at each shape's own measured
    batch rather than at a batch anybody typed in.

    Anchor-free by construction: this reads what `get_gen_only_perf` wrote,
    never a frontier. See :data:`NO_JOIN` for what that costs.
    """
    root = Path(directory)
    csvs = sorted(root.rglob(GEN_ONLY_CSV))
    if not csvs:
        raise DisaggSolError(
            f"no {GEN_ONLY_CSV} under {root}: the design's concurrency sweep has "
            f"not been scored. Run `python -m "
            f"ibc_trtllm_harness.process_data.get_gen_only_perf -i <run_dir>` for "
            f"each shape's run directory, then select again."
        )
    points: list[dict[str, Any]] = []
    for path in csvs:
        try:
            found = gen_only_points(path.parent)
        except BenchCliError as exc:  # pragma: no cover - message path
            raise DisaggSolError(str(exc)) from exc
        for point in found:
            metrics = dict(point.get("metrics") or {})
            points.append(
                {
                    "shape": point.get("name"),
                    "concurrency": point.get("concurrency"),
                    X_AXIS: metrics.get(X_AXIS),
                    Y_AXIS: metrics.get(Y_AXIS),
                    "source_csv": point.get("source_csv"),
                }
            )
    usable = [p for p in points if _usable(p)]
    if not usable:
        raise DisaggSolError(
            f"the {len(csvs)} {GEN_ONLY_CSV} under {root} carry no point with both "
            f"{X_AXIS} and {Y_AXIS}. The extractor drops a case whose iteration log "
            f"never reached steady state, so this means the cases ran but did not settle."
        )
    return usable


def _usable(point: Mapping[str, Any]) -> bool:
    if not isinstance(point.get("concurrency"), int):
        return False
    return all(
        isinstance(point.get(axis), (int, float)) and not isinstance(point.get(axis), bool)
        for axis in (X_AXIS, Y_AXIS)
    )


def pareto_front(points: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated points, both axes higher-is-better.

    A point is dropped only when another is at least as good on *both* axes
    and strictly better on one. Ties are kept: two shapes that measure the
    same pair are two real options, and picking between them is a decision
    this function is not entitled to make.
    """
    listed = [dict(p) for p in points]
    front: list[dict[str, Any]] = []
    for candidate in listed:
        dominated = any(
            other is not candidate
            and other[X_AXIS] >= candidate[X_AXIS]
            and other[Y_AXIS] >= candidate[Y_AXIS]
            and (other[X_AXIS] > candidate[X_AXIS] or other[Y_AXIS] > candidate[Y_AXIS])
            for other in listed
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda p: (-p[X_AXIS], -p[Y_AXIS]))


def select_point(
    points: Iterable[Mapping[str, Any]], *, prefer: str, incumbent: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The point the campaigns will freeze on, and why.

    ``incumbent`` is the point a campaign would otherwise have inherited —
    the checked-in row. It is not used to choose; it is used to *report*,
    because "the selection agrees with what we were already running" and
    "the selection moved us" are the two answers this whole staging exists
    to distinguish, and only one of them makes the previous campaigns'
    measurements still meaningful.
    """
    if prefer not in PREFERENCES:
        raise DisaggSolError(f"unknown preference {prefer!r}, expected one of {list(PREFERENCES)}")
    front = pareto_front(points)
    if not front:
        raise DisaggSolError("no measured point survived the Pareto filter")
    axis = X_AXIS if prefer == PREFER_INTERACTIVE else Y_AXIS
    chosen = max(front, key=lambda p: p[axis])
    result = {
        "shape": chosen["shape"],
        "concurrency": chosen["concurrency"],
        X_AXIS: chosen[X_AXIS],
        Y_AXIS: chosen[Y_AXIS],
        "prefer": prefer,
        "ranked_on": axis,
        "source_csv": chosen.get("source_csv"),
        "pareto_size": len(front),
        "measured_points": len(list(points)),
        # Not a caveat in prose somewhere: the reason travels with the number.
        "e2e_view_absent": NO_JOIN,
        "off_point_effects_unobserved": ONE_POINT,
    }
    if incumbent is not None:
        result["incumbent"] = dict(incumbent)
        result["moved"] = not _same_point(chosen, incumbent)
        result["incumbent_on_pareto"] = any(_same_point(p, incumbent) for p in front)
    return result


def _same_point(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (a.get("shape"), a.get("concurrency")) == (b.get("shape"), b.get("concurrency"))


# ------------------------------------------------------------- the ctx half


#: How a ctx candidate is ranked, and the one place this module's two halves
#: are asymmetric on purpose.
#:
#: The gen side is ranked on a curve, because a decode point trades
#: tokens/s/user against tokens/s/GPU and neither dominates. The ctx side has
#: no such trade: a prefill worker either serves more requests per GPU or it
#: does not, so the objective is scalar and there is nothing to prefer between.
#: The harness' own anchor picker uses exactly this, for exactly this reason —
#: `select_agentx_ctx_anchor.py` ranks on ``req_s_per_gpu`` because the ctx
#: term of the deployment denominator is ``ctx_gpus * ctx_per_gen``, which a
#: higher request rate per GPU shrinks on both factors at once.
#:
#: Note this needs no rate match: it is entirely inside the ctx measurement.
#: Choosing the ctx point is therefore possible without an end-to-end view,
#: which is why this half can be *measured* rather than computed even under
#: this module's no-join scope.
CTX_METRIC = "avg_request_throughput_req_s"
CTX_RANK = "req_s_per_ctx_gpu"

#: Where a validated ctx measurement keeps its number — the field the harness
#: itself requires before it will call a ctx case successful. Duplicated from
#: :mod:`.sol_track` rather than imported, because the two read it for
#: different purposes and a shared constant would make one module's change
#: silently the other's.
CTX_RESULT_PATH = ("performance", "request_throughput_req_s")


def _ctx_case_facts(case_dir: str) -> dict[str, Any] | None:
    """``ctx_{isl}_{osl}_ratio{r}_{batch}_{mnt}_{dep|tep}{tp}_MTP{n}_test{k}``.

    Read positionally, like :func:`.sol_track._ctx_concurrency`: a name this
    workflow cannot parse must stop the selection rather than let it pick
    whichever number happened to match. The GPU count is ``tp`` — a ctx-only
    run is a single aggregate worker, so its world size *is* its GPU count.
    """
    parts = case_dir.split("_")
    if len(parts) < 7 or parts[0] != "ctx":
        return None
    if not (parts[1].isdigit() and parts[4].isdigit()):
        return None
    world = parts[6]
    kind = world[:3]
    if kind not in ("dep", "tep") or not world[3:].isdigit():
        return None
    return {
        "case": case_dir,
        "isl": int(parts[1]),
        "max_batch": int(parts[4]),
        "adp": kind == "dep",
        "ctx_gpus": int(world[3:]),
    }


def ctx_points(directory: Path) -> list[dict[str, Any]]:
    """Every scored ctx candidate the design left.

    One ``run_*.json`` per case, whose
    ``performance.request_throughput_req_s`` is the field the harness
    validated the case on — so a case that appears here measured, and one
    that failed does not appear at all.
    """
    found: list[dict[str, Any]] = []
    for result in sorted(Path(directory).rglob("run_*.json")):
        if result.name.endswith("_timing.json"):
            continue
        facts = _ctx_case_facts(result.parent.name)
        if facts is None:
            continue
        value = _read_ctx_json(result)
        if value is None or facts["ctx_gpus"] <= 0:
            continue
        found.append(
            {
                **facts,
                CTX_METRIC: value,
                CTX_RANK: value / facts["ctx_gpus"],
                "source_run_json": str(result),
            }
        )
    if not found:
        raise DisaggSolError(
            f"no scored ctx case under {directory}. A ctx candidate is a "
            f"`run_*.json` whose {'.'.join(CTX_RESULT_PATH)} the harness "
            f"validated; none was found, so there is nothing to choose between "
            f"and the ctx half would fall back to a computed point."
        )
    return found


def _read_ctx_json(path: Path) -> float | None:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in CTX_RESULT_PATH:
        payload = payload.get(key) if isinstance(payload, Mapping) else None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


#: What makes two ctx measurements the same candidate.
#:
#: Deliberately not the case name. A ctx sweep repeats each case (``rounds``)
#: and expands over ``mtp_range``, so one configuration arrives as several
#: differently-named directories — twelve of them, on the run this was
#: checked against. MTP is not part of a ctx operating point at all: the
#: generation sweep's ``ctx_config`` block has no mtp field, and the measured
#: spread across mtp variants (2.4 %) sat inside the spread across repeats of
#: one variant (4.0 %).
#:
#: Ranking those twelve as twelve candidates does two wrong things: it picks
#: the luckiest repeat rather than the best configuration, and it reports
#: having *moved* when it only moved between repeats of the incumbent.
CTX_CONFIG_KEYS = ("ctx_gpus", "max_batch", "adp")


def _config_key(point: Mapping[str, Any]) -> tuple:
    facts = point
    if not all(key in point for key in CTX_CONFIG_KEYS):
        parsed = _ctx_case_facts(str(point.get("case") or ""))
        if parsed is None:
            return ("?", point.get("case"))
        facts = parsed
    return tuple(facts[key] for key in CTX_CONFIG_KEYS)


def select_ctx_point(
    points: Iterable[Mapping[str, Any]], *, incumbent: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The most GPU-efficient measured prefill *configuration*.

    Scalar, so there is no ``prefer`` here and no Pareto front: unlike the
    decode curve, a ctx candidate that serves more requests per GPU is better
    in every way a deployment cares about. Ties break toward the smaller GPU
    count, because two configurations at one efficiency are not equal — the
    smaller one leaves the rest of the node to the generation side.

    Repeats of one configuration are averaged rather than competed. Taking
    the maximum over repeats would rank configurations by which one drew the
    kindest sample, and the measured spread here — 4.0 % across twelve
    repeats of a single configuration — is wider than the difference between
    configurations that this selection is meant to resolve.
    """
    listed = [dict(p) for p in points]
    if not listed:
        raise DisaggSolError("no measured ctx candidate to choose between")

    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for point in listed:
        grouped.setdefault(_config_key(point), []).append(point)

    candidates = []
    for key, repeats in grouped.items():
        ranks = [r[CTX_RANK] for r in repeats]
        values = [r[CTX_METRIC] for r in repeats]
        first = repeats[0]
        candidates.append(
            {
                "config": key,
                "ctx_gpus": first["ctx_gpus"],
                "max_batch": first["max_batch"],
                "adp": first["adp"],
                CTX_METRIC: sum(values) / len(values),
                CTX_RANK: sum(ranks) / len(ranks),
                "repeats": len(repeats),
                "spread_pct": (max(ranks) - min(ranks)) / (sum(ranks) / len(ranks)) * 100.0,
                "cases": [r.get("case") for r in repeats],
            }
        )

    chosen = max(candidates, key=lambda c: (c[CTX_RANK], -c["ctx_gpus"]))
    result = {
        "ctx_gpus": chosen["ctx_gpus"],
        "max_batch": chosen["max_batch"],
        "adp": chosen["adp"],
        CTX_METRIC: chosen[CTX_METRIC],
        CTX_RANK: chosen[CTX_RANK],
        "ranked_on": f"mean {CTX_RANK} over {chosen['repeats']} repeat(s)",
        "repeats": chosen["repeats"],
        "spread_pct": chosen["spread_pct"],
        "cases": chosen["cases"],
        "candidates": len(candidates),
        "measurements": len(listed),
    }
    if incumbent is not None:
        result["incumbent"] = dict(incumbent)
        # By configuration, never by case name: a different repeat of the
        # incumbent is not a move, and reporting it as one gives the wrong
        # answer to the question this whole staging exists to ask.
        result["moved"] = chosen["config"] != _config_key(incumbent)
    return result


# --------------------------------------------------------------- the campaigns


#: A design is reusable once its concurrency sweep has been scored. Checked
#: on the artefacts rather than on ``state.json``'s ``phase`` string: the
#: phase is what the design *says* it reached, the CSVs are what it left, and
#: a resumed or interrupted design can have the first without the second.
def established(directory: Path, track: str = GEN_TRACK) -> bool:
    """Whether this design can be selected from for ``track``.

    Per track, because the two halves are established by different artefacts
    and — under this module's no-join scope — neither waits on the other. A
    design whose prefill candidates have been measured can start the ctx
    campaign while its generation sweep is still running; requiring both
    would idle a half that is ready on a half that is not, for a dependency
    the scope does not have.

    The reason this is a question at all: fixing the operating point costs
    roughly an order of magnitude more than the campaigns it enables — six
    shapes probed and swept, against one baseline each. Paying that per
    campaign would be absurd; paying it once per (model, cluster, workload)
    and reusing it is the only shape in which the staging is affordable.
    """
    root = Path(directory)
    if track == CTX_TRACK:
        return any(
            path.name.startswith("run_") and not path.name.endswith("_timing.json")
            for path in root.rglob("run_*.json")
        )
    return bool(sorted(root.rglob(GEN_ONLY_CSV)))


#: Why a design may be somewhere other than where it was asked for, and what
#: is done about it.
#:
#: The design agent is given a directory and told to resume from its
#: ``state.json``. When the named directory does not exist and a sibling one
#: does — carrying a state file whose recorded scope matches the instruction —
#: resuming that sibling is the right engineering call: it saved roughly
#: twenty node-hours of re-probing on the run this was written after. Refusing
#: it would forbid a correct decision.
#:
#: But the supervisor checks the directory it *asked* for, so the correct
#: decision surfaced as "the design was never established" and the run
#: stopped. Following the agent silently would be worse: every later artefact
#: would cite a design directory the spec never named.
#:
#: So it is followed and recorded. The redirection travels in the run record
#: and in every campaign's provenance, for the same reason the missing
#: end-to-end view does: a reader must not have to already know.
DESIGN_REDIRECTED = (
    "the design agent was given {requested} and established the design in {actual} "
    "instead. That is legitimate -- a design is resumed from its state.json, and "
    "resuming an existing one avoids re-measuring what it already measured -- but "
    "it means every artefact below was selected from a directory this spec did not "
    "name. Point the spec at {actual} to make the next run's request and result "
    "agree."
)


def resolve_design_dir(requested: Path, wanted: Sequence[str]) -> tuple[Path, str | None]:
    """Where the design actually is, and a note if that is not where it was asked.

    Looked for among the requested directory's siblings, which is where an
    agent resuming an existing design would find one: the model directory
    holds ``sweep_design`` and whatever variants a spec has named. The search
    is deliberately narrow -- one level, and only directories that both carry
    a ``state.json`` and are established for every wanted track -- because a
    supervisor that hunts the filesystem for something that looks like a
    design will eventually find one that is not.

    Ambiguity is refused rather than resolved. Two established designs beside
    each other is a question about which measurement the campaigns should
    freeze on, and this is not the layer that answers it.
    """
    requested = Path(requested)
    if all(established(requested, track) for track in wanted):
        return requested, None

    parent = requested.parent
    if not parent.is_dir():
        return requested, None
    found = [
        candidate
        for candidate in sorted(parent.iterdir())
        if candidate.is_dir()
        and candidate != requested
        and (candidate / DESIGN_STATE).is_file()
        and all(established(candidate, track) for track in wanted)
    ]
    if not found:
        return requested, None
    if len(found) > 1:
        raise DisaggSolError(
            f"{requested} is not established, and {len(found)} sibling directories "
            f"are: {[str(f) for f in found]}. Which measurement the campaigns "
            f"freeze on is not something this layer may pick -- name one in "
            f"'{DISAGG_SOL_FIELD}.{DESIGN_KEY}.{DESIGN_DIR_KEY}'."
        )
    return found[0], DESIGN_REDIRECTED.format(requested=requested, actual=found[0])


def campaign_workspace(root: Path, track: str, label: str) -> Path:
    """Each half gets its own, and the name says which half it is.

    Not cosmetic: the workspace is what a campaign's branch is named after,
    and the branch is what claims the checkout. Two campaigns sharing either
    would have each reset the other's worktree mid-flight.
    """
    return Path(root) / f"ws-{label}-{track}"


def campaign_spec(
    base: Mapping[str, Any],
    *,
    track: str,
    sweep: Path,
    repo: Path,
    point: Mapping[str, Any],
    design: Path,
) -> dict[str, Any]:
    """One half's `task.yaml`, as a single-track SOL campaign.

    The output is an ordinary :mod:`.sol_track` spec — the same shape the two
    campaigns this module supersedes were written by hand. That is the point:
    nothing below this layer changes, so everything already true of a
    single-track campaign stays true, and this layer is additive rather than a
    rewrite.

    What it adds is **provenance**. A campaign started from here records the
    design it froze on and the point it was given, so a report can answer
    "where did this operating point come from" with a path instead of a
    shrug. The two campaigns that motivated this module could not.

    No ``ctx_json``: this module does not build an end-to-end view (see
    :data:`NO_JOIN`), so the gen half is scored anchor-free and says so.
    """
    if track not in TRACKS:
        raise DisaggSolError(f"unknown track {track!r}, expected one of {list(TRACKS)}")
    spec: dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key in ("checkpoint_path", "optimize", "profile")
    }
    spec["trtllm_repo_path"] = str(repo)
    spec["sol_track"] = {
        "track": track,
        "sweep": str(sweep),
        # Recorded, not acted on: this layer chose the point, and the
        # campaign must be able to say so without reading this module.
        "point_provenance": {
            "design_dir": str(design),
            "selected": {k: point.get(k) for k in ("shape", "concurrency", "prefer", "ranked_on")},
            "e2e_view_absent": point.get("e2e_view_absent", NO_JOIN),
        },
    }
    return spec


def verify_sweep_matches_point(sweep: Path, track: str, point: Mapping[str, Any]) -> None:
    """Refuse a sweep that does not contain the point this layer selected.

    Without this the selection is a **report**, not a decision. The chosen
    point is written into every campaign's ``point_provenance``, and the
    campaign runs whatever its sweep says — so a sweep that names a different
    row produces a record claiming an operating point the run never used.
    That is worse than not selecting at all: the whole reason this layer
    exists is to make a campaign's starting point traceable, and an
    untraceable point is at least honest about being untraceable.

    Checked rather than generated. Writing the sweep here would put this
    module back in the business of authoring configs it does not own, which
    is the mistake :mod:`.sweep_design` was just unwound for. The sweep stays
    something a person or the design skill wrote; this refuses the pairing.
    """
    try:
        config = load_yaml(Path(sweep))
    except Exception as exc:  # noqa: BLE001 - re-raised with the pairing named
        raise DisaggSolError(f"could not read the {track} sweep {sweep}: {exc}") from exc

    if track == GEN_TRACK:
        wanted = (point.get("shape"), point.get("concurrency"))
        if wanted[0] is None:
            return  # a ctx-only selection: nothing to check the gen sweep against
        try:
            available = {
                (case["name"], (case["config"] or {}).get("concurrency"))
                for case in gen_cases(config)
            }
        except BenchCliError as exc:  # pragma: no cover - message path
            raise DisaggSolError(str(exc)) from exc
        if wanted not in available:
            raise DisaggSolError(
                f"the gen sweep {sweep} does not contain the selected point "
                f"{wanted[0]} @ concurrency {wanted[1]}. It expands to "
                f"{sorted(available)}. The campaign would run one of those while "
                f"its point_provenance claimed the selected one -- a record that "
                f"says the run used an operating point it did not."
            )
        return

    wanted_ctx = (point.get("ctx_gpus"), point.get("max_batch"))
    if wanted_ctx[0] is None:
        return
    available_ctx = {
        ((case["config"] or {}).get("tp_size"), (case["config"] or {}).get("max_batch"))
        for case in ctx_cases(config)
    }
    if wanted_ctx not in available_ctx:
        raise DisaggSolError(
            f"the ctx sweep {sweep} does not contain the selected point "
            f"tp_size {wanted_ctx[0]} @ max_batch {wanted_ctx[1]}. It expands to "
            f"{sorted(available_ctx)}. The campaign would run one of those while "
            f"its point_provenance claimed the selected one."
        )


#: Where a derived sweep may be written, and the whole of the rule.
#:
#: Inside the campaign's own workspace, never anywhere else. The check is
#: blunt on purpose: an earlier module in this package took an output path as
#: a free parameter and would have overwritten a curated config that other
#: people maintain, which is unrecoverable in a way that a wrong measurement
#: is not. A derivation that cannot escape the workspace cannot make that
#: mistake, whatever it is asked to write.
DERIVED_SWEEP_NAME = "sweep-at-selected-point.yaml"


def derive_sweep_at_point(
    design_sweep: Path, track: str, point: Mapping[str, Any], *, into: Path
) -> Path:
    """Cut the design's sweep down to the one row the campaign will freeze on.

    The last link in the chain, and it exists because of a shape problem
    rather than a preference. The design sweep is the whole measured space —
    six shapes, each with its own concurrency ladder — and a campaign cannot
    run it: two shapes measured at one concurrency both land in
    ``concurrency_<c>``, which :func:`.sol_track._place` refuses, because a
    campaign's operating points have to be addressable by concurrency alone.
    So the campaign needs exactly one row, and which row is not known until
    the design has been measured and selected from.

    Derived rather than authored: every field except the row filter comes
    from the design's own sweep, so the campaign measures the configuration
    the selection was made on and not a hand-typed approximation of it. The
    two campaigns this layer replaces differed from their own recorded point
    in three fields at once, and nothing noticed.

    Written into the campaign's workspace and nowhere else — see
    :data:`DERIVED_SWEEP_NAME`.
    """
    into = Path(into).resolve()
    out = into / DERIVED_SWEEP_NAME
    config = load_yaml(Path(design_sweep))

    if track == GEN_TRACK:
        wanted = (point.get("shape"), point.get("concurrency"))
        kept = [row for row in (config.get("gen_configs") or []) if _gen_row_matches(row, wanted)]
        if len(kept) != 1:
            raise DisaggSolError(
                f"{design_sweep} has {len(kept)} rows matching the selected point "
                f"{wanted[0]} @ concurrency {wanted[1]}; a campaign needs exactly "
                f"one. The design sweep and the measured space have diverged."
            )
        row = list(kept[0])
        row[9] = str(wanted[1])  # this point only, not the row's whole ladder
        config["gen_configs"] = [row]
    else:
        wanted_ctx = (point.get("ctx_gpus"), point.get("max_batch"))
        kept_ctx = [
            {**entry, "tp_size": [wanted_ctx[0]], "max_batch": [wanted_ctx[1]]}
            for entry in (config.get("benchmarks") or [])
            if isinstance(entry, Mapping)
            and wanted_ctx[0] in (entry.get("tp_size") or [])
            and wanted_ctx[1] in (entry.get("max_batch") or [])
        ]
        if len(kept_ctx) != 1:
            raise DisaggSolError(
                f"{design_sweep} has {len(kept_ctx)} benchmark entries matching the "
                f"selected ctx point tp_size {wanted_ctx[0]} @ max_batch "
                f"{wanted_ctx[1]}; a campaign needs exactly one."
            )
        config["benchmarks"] = kept_ctx
        config.pop("gpu_overrides", None)

    config["_derived_from"] = {
        "design_sweep": str(design_sweep),
        "selected": {k: point.get(k) for k in ("shape", "concurrency", "ctx_gpus", "max_batch")},
        "why": (
            "the design sweep is the whole measured space; a campaign freezes one "
            "row of it, because two shapes at one concurrency cannot both be "
            "'concurrency_<c>'"
        ),
    }
    into.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _gen_row_matches(row: Any, wanted: tuple) -> bool:
    if not isinstance(row, (list, tuple)) or len(row) < 10:
        return False
    tp, adp, mtp, eplb = row[2], row[5], row[7], row[8]
    name = f"{'dep' if adp else 'tep'}_{tp}_eplb{eplb}_mtp{mtp}"
    if name != wanted[0]:
        return False
    return any(
        token.strip().isdigit() and int(token) == wanted[1] for token in str(row[9]).split(",")
    )


class CampaignLaunch:
    """One campaign, described completely enough to be checked before it runs.

    Separated from the spawning so the *decision* — which track, which
    sweep, which checkout, which workspace — is testable without starting a
    process. The spawn itself is four lines and has no judgement in it; this
    is where the judgement is.
    """

    def __init__(self, track: str, spec: Mapping[str, Any], workspace: Path, task_path: Path):
        self.track = track
        self.spec = dict(spec)
        self.workspace = Path(workspace)
        self.task_path = Path(task_path)

    @property
    def argv(self) -> list[str]:
        return [
            "perf-optimize",
            "--task",
            str(self.task_path),
            "--workspace",
            str(self.workspace),
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CampaignLaunch(track={self.track!r}, workspace={self.workspace})"


def launch_plan(
    base: Mapping[str, Any],
    *,
    sweeps: Mapping[str, Path],
    repos: Mapping[str, Path],
    workspace_root: Path,
    label: str,
    point: Mapping[str, Any],
    design: Path,
    design_sweep: Path | None = None,
) -> list[CampaignLaunch]:
    """What this spec would start, in full, before anything starts.

    Every campaign is checked for the two things that made the hand-launched
    pair safe and which nothing else enforces: **its own checkout** and **its
    own workspace**. Two campaigns pointed at one checkout review each
    other's worktree; two pointed at one workspace overwrite each other's
    results. Both failures produce plausible numbers, which is why they are
    refused here rather than left to the launcher's care.
    """
    wanted = tracks(base)
    missing = [t for t in wanted if t not in sweeps and design_sweep is None]
    if missing:
        raise DisaggSolError(
            f"track(s) {missing} have neither a sweep of their own nor a "
            f"'{DESIGN_KEY}.{DESIGN_SWEEP_KEY}' to cut one from. A campaign has to "
            f"freeze exactly one row, and which row is not known until the design "
            f"has been measured -- so either name the design's sweep and let it be "
            f"derived, or name a sweep that already contains the selected point."
        )
    missing = [t for t in wanted if t not in repos]
    if missing:
        raise DisaggSolError(f"no trtllm_repo_path given for track(s) {missing}")

    seen_repos: dict[str, str] = {}
    launches: list[CampaignLaunch] = []
    for track in wanted:
        repo = Path(repos[track]).resolve()
        if str(repo) in seen_repos:
            raise DisaggSolError(
                f"tracks {seen_repos[str(repo)]!r} and {track!r} both name the checkout "
                f"{repo}. A campaign resets the checkout it is given, so two of them "
                f"sharing one would each revert the other's work mid-flight — and the "
                f"measurement that followed would be of neither's code."
            )
        seen_repos[str(repo)] = track
        workspace = campaign_workspace(workspace_root, track, label)
        if track in sweeps:
            # A sweep the caller chose: check it, never rewrite it.
            sweep = Path(sweeps[track])
            verify_sweep_matches_point(sweep, track, point)
        else:
            sweep = derive_sweep_at_point(Path(design_sweep), track, point, into=workspace)
        launches.append(
            CampaignLaunch(
                track=track,
                spec=campaign_spec(
                    base,
                    track=track,
                    sweep=Path(sweeps[track]),
                    repo=repo,
                    point=point,
                    design=design,
                ),
                workspace=workspace,
                task_path=workspace / "task.yaml",
            )
        )
    return launches


# --------------------------------------------------------------- the run


#: Where the supervisor records what it selected and started.
RUN_RECORD = "disagg_sol_run.json"


def supervise(
    base: Mapping[str, Any],
    *,
    sweeps: Mapping[str, Path],
    repos: Mapping[str, Path],
    workspace_root: Path,
    label: str,
    incumbent: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    designer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fix the point from an established design, then start each half at it.

    The design is an **input**, not something this produces: establishing it
    costs roughly an order of magnitude more than the campaigns it enables,
    so it is run once per (model, cluster, workload) and reused. A spec whose
    design has not been scored is refused here rather than silently falling
    back to whatever operating point the sweeps happen to carry — falling
    back is exactly the behaviour this layer exists to remove.

    Returns the record it writes: what was selected, against what incumbent,
    and what was started. ``dry_run`` stops after the record, which is the
    same code path minus the processes.
    """
    from agent_flow.workflows.perf_optimize import spawn

    design = design_dir(base)
    prefer = preference(base)
    wanted = tracks(base)

    # Resolved BEFORE the designer is considered, not after. A design costs
    # about an order of magnitude more than the campaigns it enables, so
    # paying for one that already exists next door is the expensive half of
    # this mistake -- and the reason the agent resumed a neighbour in the
    # first place was to avoid exactly that.
    design, redirect = resolve_design_dir(design, wanted)
    unready = [t for t in wanted if not established(design, t)]
    if unready and designer is not None:
        # The design is normally an input -- it is reused across campaigns and
        # costs an order of magnitude more than any of them. But "an input"
        # degenerated into "nobody ran it" once already, which is how the two
        # campaigns this layer replaces came to inherit an unmeasured row. So
        # when a designer is available the run establishes what it needs and
        # then re-checks: not a fallback, a first step.
        designer(
            designer_instruction(
                model_dir=Path(design).parent.name, design_dir=design, tracks=unready
            )
        )
        # ...and read back again after it runs, because the agent resumes
        # from a state file and may have established the design somewhere
        # other than where it was sent.
        design, after = resolve_design_dir(design, wanted)
        redirect = after or redirect
        unready = [t for t in wanted if not established(design, t)]
    if unready:
        what = {
            CTX_TRACK: "no scored ctx case (a `run_*.json` the harness validated)",
            GEN_TRACK: f"no scored concurrency sweep (a `{GEN_ONLY_CSV}`)",
        }
        raise DisaggSolError(
            f"{design} has "
            + "; ".join(f"{what[t]} for track '{t}'" for t in unready)
            + f". There is no measured space to choose those halves' operating "
            f"points from. Establish the design first — it is reused across "
            f"campaigns, so this is paid once — or narrow "
            f"'{DISAGG_SOL_FIELD}.{TRACKS_KEY}' to the halves it already covers."
        )

    record: dict[str, Any] = {
        "design_dir": str(design),
        **({"design_dir_redirected": redirect} if redirect else {}),
        "design_state": design_state(design).get("phase"),
        "tracks": wanted,
        "label": label,
    }

    if GEN_TRACK in wanted:
        record["gen_point"] = select_point(sweep_points(design), prefer=prefer, incumbent=incumbent)
    if CTX_TRACK in wanted:
        # No `prefer`: the ctx objective is scalar. See `select_ctx_point`.
        record["ctx_point"] = select_ctx_point(ctx_points(design), incumbent=incumbent)

    point = record.get("gen_point") or record.get("ctx_point") or {}
    block = _block(base).get(DESIGN_KEY)
    block = block if isinstance(block, Mapping) else {}
    stated_sweep = block.get(DESIGN_SWEEP_KEY)
    launches = launch_plan(
        base,
        sweeps=sweeps,
        repos=repos,
        workspace_root=workspace_root,
        label=label,
        point=point,
        design=design,
        design_sweep=Path(stated_sweep) if stated_sweep else None,
    )
    record["campaigns"] = [
        {"track": run.track, "workspace": str(run.workspace), "argv": run.argv} for run in launches
    ]

    Path(workspace_root).mkdir(parents=True, exist_ok=True)
    (Path(workspace_root) / RUN_RECORD).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if dry_run:
        record["started"] = False
        return record

    started = spawn.start_all(launches)
    record["started"] = True
    record["exit_status"] = spawn.wait_all(started)
    (Path(workspace_root) / RUN_RECORD).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record
