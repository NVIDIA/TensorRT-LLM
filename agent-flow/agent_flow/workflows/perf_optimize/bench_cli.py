"""Reading the benchmark suite's configs and results, without driving it.

A SOL-track campaign measures through `ibc-bench`, the CLI the
`ibc-trtllm-harness` package ships. The agent drives it — `submit sweep`,
`jobs check`, `process ctx`, `process frontier` are Bash commands in the
track's prompt section, run by the role that needs them, the same way an
aggregate campaign runs `trtllm-serve` and `benchmark_serving`.

Python's part is smaller than it looks, and deliberately so: it reads two
kinds of file the harness owns, and runs nothing.

- **The sweep** — at schema validation, before any agent exists, so
  `task.yaml` can be reconciled against the run that will happen. The
  expansion is computed here rather than asked of the CLI because
  `ibc-bench` has no read-only "what would this submit" command:
  `submit sweep --dry-run` answers, but by writing a case directory per
  case, which is not something schema validation may do. The rule it
  applies is the harness's own and is narrow — a `gen_configs` row is
  `[ctx, gen, tp, batch, max_num_tokens, attention_dp, gpu_mem_frac, mtp,
  eplb, concurrency_list]` and expands over that last field; a ctx
  `benchmarks` entry expands over `max_batch` × `tp_size` × `mtp_range`.
  The agent's own `--dry-run` step re-derives it against the harness
  before an allocation is spent, so a drift shows up as a refusal rather
  than as a campaign quoting points it never measured.

- **The frontier CSV** — what `ibc-bench process frontier` writes. Its
  columns are already the names this workflow scores on:
  `throughput_per_user` needs no translation, and `output_tput_per_gpu`
  and `ctx_gen_inst_ratio_round_float` carry the frontier the gate's
  metric is not.

- **The gen-only CSV** — what `get_gen_only_perf` writes, for a gen
  campaign that declares no ctx anchor. The gate's metric is
  ``accept_rate / avg_iteration_time``, which has no context term in it;
  the frontier is the only thing that needs the anchor. Reading the score
  out of the frontier therefore invented a dependency the measurement
  does not have — a gen campaign had to wait for somebody's ctx run
  before it could score a decode change that ctx cannot affect. Both
  readers are kept because they answer different questions, and which one
  applies is declared in `task.yaml` rather than guessed from the
  directory.

What this module does **not** do is reconstruct the identity model the
previous CLI carried (`config_id` / `code_id` / per-measurement records).
Nothing here can tell an attempt from a repeat of its baseline, so the
prompts make that an obligation the evaluator discharges by reading the
case's materialized config — stated as such, rather than implied.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The console script `ibc-trtllm-harness` installs. Named for the prompt
#: sections and error messages; nothing here executes it.
IBC_BENCH = "ibc-bench"

#: Where `process frontier` leaves the scored points. One per (mtp tag,
#: variant), so a run dir can hold several; the reader takes them all.
FRONTIER_CSV_GLOB = "*_frontier_*.csv"

#: The columns this workflow reads. The rest of the row is carried
#: through untouched, because a number a reader can trace beats a number
#: that arrived alone.
CASE_KEY = ("name", "concurrency")
GEN_METRIC = "throughput_per_user"

#: Where `get_gen_only_perf` leaves the anchor-free scores — one file per
#: run dir, written beside the cases it read.
GEN_ONLY_CSV = "gen_only_perf.csv"

#: What that file calls :data:`GEN_METRIC`. The same quantity by the same
#: formula (`accept_rate / avg_iteration_time`); only the column name
#: differs, so the rename happens here and nowhere else.
GEN_ONLY_METRIC = "tps_per_user"

#: How the harness exposes the gen-only extractor. It has no `ibc-bench`
#: subcommand as of v0.5.7 — `process frontier` is the only scored gen
#: path on the CLI — so the prompt runs the module directly. Named here so
#: the error messages and the prompt cannot drift apart.
GEN_ONLY_MODULE = "ibc_trtllm_harness.process_data.get_gen_only_perf"


class BenchCliError(RuntimeError):
    """A sweep or a result set could not be read as the harness writes it."""


def _int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # NaN is not a measurement


def _bool(value: Any) -> bool:
    """A pandas-written boolean cell. Anything unrecognised is ``False``.

    ``get_gen_only_perf`` writes Python bools through ``DataFrame.to_csv``,
    so the cell is the literal ``True``/``False``. Only this column feeds
    the ``dep``/``tep`` half of a case name, and a name that silently
    changed shape would be worse than one that is visibly wrong, so the
    accepted spellings are enumerated rather than inferred from
    truthiness — ``"False"`` is a non-empty string.
    """
    return str(value).strip().lower() in {"true", "1", "yes"}


# ------------------------------------------------------------------ the sweep


def _one(values: Iterable[Any]) -> Any:
    """The single distinct value, or ``None`` if there is not exactly one.

    Disagreement is not resolved to a first or a maximum: two rows at two
    input lengths are two workloads, and a campaign that quoted one of
    them would describe half its own cases wrongly with nothing raising.
    """
    seen = [value for value in values if value is not None]
    unique = {str(value): value for value in seen}
    return next(iter(unique.values())) if len(unique) == 1 else None


def workload(sweep: Mapping[str, Any]) -> dict[str, Any]:
    """What the run will serve, as the sweep states it.

    The two sweep kinds spell this differently and the difference is not
    cosmetic. A **gen** sweep is one deployment, so the model and the
    corpus are top-level and ``isl``/``osl`` with them. A **ctx** sweep is
    a list of prefill benchmarks: the model sits under ``model:`` and each
    ``benchmarks`` entry carries its own lengths, because sweeping the
    input length is the normal thing to do there. Read only the gen shape
    and a ctx campaign resolves to a workload of ``None``s -- which does
    not fail, it simply reconciles nothing, and `task.yaml` keeps the
    defaults block's ``random_input_len: 1024`` as this campaign's stated
    input length whatever the sweep measures.

    So the ctx lengths are taken from the ``benchmarks`` entries, and only
    when they agree. A campaign is frozen to one operating point, so they
    do; a sweep spanning two input lengths has no single workload to
    state, and saying nothing is the honest answer there.

    ``isl`` is deliberately not read as the request length on the gen
    side. The harness spends it on ``max_seq_len`` and the client's
    ``input_length``; the requests come from ``dataset_file``, a corpus
    with its own distribution. The checked-in 8k sweep pairs ``isl: 8192``
    with a ``...-8192-1024-200000-...`` corpus and both are right. On a
    ctx sweep the two coincide -- a prefill-only run at ``osl: 1`` reads a
    corpus generated for that length -- but they are still read from where
    each sweep puts them, not assumed equal.
    """
    model = sweep.get("model")
    model = model if isinstance(model, Mapping) else {}
    entries = [entry for entry in sweep.get("benchmarks") or [] if isinstance(entry, Mapping)]
    return {
        "model": sweep.get("model_id") or model.get("model_card"),
        "model_path": sweep.get("model_path") or model.get("model_path"),
        "dataset": sweep.get("dataset_file") or model.get("dataset_file"),
        "precision": sweep.get("precision"),
        "isl": sweep.get("isl")
        if sweep.get("isl") is not None
        else _one(e.get("isl") for e in entries),
        "osl": sweep.get("osl")
        if sweep.get("osl") is not None
        else _one(e.get("osl") for e in entries),
        "benchmark_client": sweep.get("benchmark_client"),
    }


def gen_cases(sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every gen case a `gen_configs` sweep expands to.

    The row order is the harness': `[ctx_num, gen_num, tp_size, batch,
    max_num_tokens, attention_dp, gpu_memory_fraction, mtp, eplb,
    concurrency_list]`. The case name mirrors what the postprocessor's
    `name` column carries — `{dep|tep}_{tp}_eplb{N}_mtp{M}` — so a point
    read back out of the CSV addresses the row that produced it.
    """
    cases: list[dict[str, Any]] = []
    for row in sweep.get("gen_configs") or []:
        if isinstance(row, Mapping):  # the dict form the harness also accepts
            row = [
                row.get(k)
                for k in (
                    "ctx_num",
                    "gen_num",
                    "gen_tp_size",
                    "gen_batch_size",
                    "gen_max_num_tokens",
                    "gen_enable_attention_dp",
                    "gen_gpu_memory_fraction",
                    "gen_mtp_size",
                    "gen_eplb_num_slots",
                    "gen_concurrency_list",
                )
            ]
        if not isinstance(row, (list, tuple)) or len(row) < 10:
            continue
        ctx_num, gen_num, tp, batch, mnt, adp, gmf, mtp, eplb, concurrencies = row[:10]
        shape = f"{'dep' if adp else 'tep'}_{tp}_eplb{eplb}_mtp{mtp}"
        for token in str(concurrencies).split(","):
            concurrency = _int(token)
            if concurrency is None:
                continue
            cases.append(
                {
                    "case": f"{shape}_conc{concurrency}",
                    "name": shape,
                    "stage": "gen",
                    "config": {
                        "ctx_num": _int(ctx_num),
                        "gen_num": _int(gen_num),
                        "tp_size": _int(tp),
                        "batch_size": _int(batch),
                        "max_num_tokens": _int(mnt),
                        "attention_dp": bool(adp),
                        "gpu_memory_fraction": _float(gmf),
                        "mtp_size": _int(mtp),
                        "eplb_num_slots": _int(eplb),
                        "concurrency": concurrency,
                    },
                }
            )
    return cases


def ctx_cases(sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every ctx case a `benchmarks` block expands to.

    A ctx entry has no concurrency: `max_batch` is the in-flight request
    count for a prefill-only run, which is what this workflow means by
    concurrency everywhere else.
    """
    cases: list[dict[str, Any]] = []
    for entry in sweep.get("benchmarks") or []:
        if not isinstance(entry, Mapping):
            continue
        isl, osl = entry.get("isl"), entry.get("osl", 1)
        for batch in entry.get("max_batch") or []:
            for tp in entry.get("tp_size") or []:
                for ratio in entry.get("ratio") or [None]:
                    for mtp in entry.get("mtp_range") or [0]:
                        cases.append(
                            {
                                "case": f"ctx-isl{isl}_osl{osl}_b{batch}_tp{tp}_mtp{mtp}",
                                "stage": "ctx",
                                "config": {
                                    "isl": _int(isl),
                                    "osl": _int(osl),
                                    "max_batch": _int(batch),
                                    "tp_size": _int(tp),
                                    "ratio": ratio,
                                    "mtp": _int(mtp),
                                },
                            }
                        )
    return cases


def plan(sweep: Mapping[str, Any]) -> dict[str, Any]:
    """The sweep as a workload plus the cases it expands to."""
    return {"workload": workload(sweep), "cases": gen_cases(sweep) + ctx_cases(sweep)}


def operating_point(config: Mapping[str, Any]) -> int | None:
    """Total requests in flight at one case, or ``None`` if unstated.

    Not ``concurrency`` verbatim on a gen case: the sweep row's value is
    **per generation server**, the client is driven at ``concurrency *
    gen_num``, and the harness names the result directory after that
    product. A ctx case has no concurrency at all — ``max_batch`` is the
    in-flight count for a prefill-only run.
    """
    listed = config.get("concurrency")
    if not isinstance(listed, int) or isinstance(listed, bool):
        batch = config.get("max_batch")
        return batch if isinstance(batch, int) and not isinstance(batch, bool) else None
    gen_num = config.get("gen_num", 1)
    if not isinstance(gen_num, int) or isinstance(gen_num, bool) or gen_num < 1:
        gen_num = 1
    return listed * gen_num


def operating_points(plan_data: Mapping[str, Any]) -> list[int]:
    """The concurrency axis a `task.yaml` should carry, from a plan."""
    points = (operating_point(case.get("config") or {}) for case in plan_data.get("cases") or [])
    return sorted({point for point in points if point is not None})


# ---------------------------------------------------------------- the results


def frontier_points(run_dir: Path) -> list[dict[str, Any]]:
    """Every scored point `process frontier` wrote under ``run_dir``.

    Keyed by ``(name, concurrency)`` — the shape the row was measured at
    and the point on its curve — because that pair is what survives a code
    change, and therefore what an attempt and its baseline have in common.
    """
    found: dict[tuple[str, int], dict[str, Any]] = {}
    csvs = sorted(Path(run_dir).rglob(FRONTIER_CSV_GLOB))
    if not csvs:
        raise BenchCliError(
            f"no {FRONTIER_CSV_GLOB} under {run_dir}: `{IBC_BENCH} process frontier` "
            f"has not run, or ran without producing a curve. `{IBC_BENCH} jobs check "
            f"-f {run_dir}/job_status.csv --summary` says whether the cases behind it "
            f"measured."
        )
    for path in csvs:
        try:
            rows: Iterable[dict[str, str]] = list(csv.DictReader(path.open(encoding="utf-8")))
        except OSError as exc:  # pragma: no cover - message path
            raise BenchCliError(f"could not read {path}: {exc}") from exc
        for row in rows:
            name = (row.get("name") or "").strip()
            concurrency = _int(row.get("concurrency"))
            value = _float(row.get(GEN_METRIC))
            if not name or concurrency is None or value is None:
                continue
            found[(name, concurrency)] = {
                "case": f"{name}_conc{concurrency}",
                "name": name,
                "concurrency": concurrency,
                "metrics": {
                    GEN_METRIC: value,
                    "output_tput_per_gpu": _float(row.get("output_tput_per_gpu")),
                    "ctx_gen_inst_ratio": _float(row.get("ctx_gen_inst_ratio_round_float")),
                    "ctx_request_rate": _float(row.get("ctx_request_rate")),
                },
                "gpus": {
                    "ctx": _float(row.get("ctx_gpus_round")),
                    "gen": _float(row.get("gen_num_round")),
                    "total": _float(row.get("total_gpus_round")),
                },
                "source_csv": str(path),
            }
    if not found:
        raise BenchCliError(
            f"the frontier CSVs under {run_dir} carry no row with a name, a "
            f"concurrency and a {GEN_METRIC}. Check that the postprocessor scored "
            f"the cases rather than only listing them."
        )
    return [found[key] for key in sorted(found)]


def gen_only_points(run_dir: Path) -> list[dict[str, Any]]:
    """Every scored point `get_gen_only_perf` wrote under ``run_dir``.

    The same shape :func:`frontier_points` returns, keyed the same way, so
    the rest of the workflow cannot tell which reader produced a point —
    except by what is missing, which is the whole e2e half:
    ``output_tput_per_gpu``, the ctx:gen ratio and the GPU counts all
    divide by a context rate this campaign never measured.

    They are **absent rather than zero**, and the caller is expected to
    say so in the result it writes. A frontier column filled with a
    plausible default is the failure this track is built to refuse: the
    curve still plots, the deployment it describes never existed.

    ``output_tps_per_gen_gpu`` is carried but deliberately not renamed to
    ``output_tput_per_gpu``. They differ in the denominator — this one
    divides by the generation GPUs alone, the frontier's by the whole
    rate-matched deployment — so the two are never the same number, and
    the smaller denominator makes this the flattering one.
    """
    path = Path(run_dir) / GEN_ONLY_CSV
    if not path.is_file():
        raise BenchCliError(
            f"no {GEN_ONLY_CSV} in {run_dir}: this campaign declares no ctx anchor, "
            f"so its score comes from the anchor-free extractor. Run "
            f"`python -m {GEN_ONLY_MODULE} -i {run_dir}` after the sweep, then "
            f"collect again. `{IBC_BENCH} jobs check -f {run_dir}/job_status.csv "
            f"--summary` says whether the cases behind it measured."
        )
    try:
        rows: Iterable[dict[str, str]] = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError as exc:  # pragma: no cover - message path
        raise BenchCliError(f"could not read {path}: {exc}") from exc

    found: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        concurrency = _int(row.get("concurrency"))
        value = _float(row.get(GEN_ONLY_METRIC))
        tp, mtp, eplb = _int(row.get("tp")), _int(row.get("mtp")), _int(row.get("eplb"))
        if concurrency is None or value is None or tp is None:
            continue
        # Rebuilt from the columns rather than parsed out of `config`, so
        # it matches `gen_cases()`'s shape and the frontier CSV's `name`
        # exactly. A point read through either reader then addresses the
        # same planned row.
        name = f"{'dep' if _bool(row.get('adp')) else 'tep'}_{tp}_eplb{eplb}_mtp{mtp}"
        found[(name, concurrency)] = {
            "case": f"{name}_conc{concurrency}",
            "name": name,
            "concurrency": concurrency,
            "metrics": {
                GEN_METRIC: value,
                "output_tput": _float(row.get("output_tput")),
                "output_tps_per_gen_gpu": _float(row.get("output_tps_per_gen_gpu")),
                "avg_itertime_ms": _float(row.get("avg_itertime_ms")),
                "num_iters": _int(row.get("num_iters")),
            },
            "source_csv": str(path),
            "harness_config": (row.get("config") or "").strip() or None,
        }
    if not found:
        raise BenchCliError(
            f"{path} carries no row with a concurrency, a tp and a {GEN_ONLY_METRIC}. "
            f"The extractor drops a case whose iteration log never reached steady "
            f"state, so an empty file means the cases ran but did not settle."
        )
    return [found[key] for key in sorted(found)]
