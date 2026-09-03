# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate ``time_breakdown`` per-request lifecycle metrics for perf-sanity upload.

Reads the per-request ``perf_metrics-*.jsonl`` files produced by a disagg run (or a merged
JSONL) and reduces them to ``mean`` / ``median`` / ``p75`` / ``p99`` per metric, in
milliseconds, ready to be uploaded to OpenSearch as ``d_tb_<metric>_<stat>``.

Five metric groups, matching ``tensorrt_llm/serve/scripts/time_breakdown/README.md``:

===  ======================================  =====================================
#    group                                   supported in
===  ======================================  =====================================
1    Context/Prefill stage (per request)      ``ctx_only``, ``e2e``
2    Per-chunk, prefill (per chunk)           ``ctx_only``, ``e2e``
3    Per-step, generation (per step)          ``gen_only``, ``e2e``
4    Generation/Decode stage (per request)    ``gen_only``, ``e2e``
5    Disaggregation server (per request)      ``gen_only``, ``e2e``
===  ======================================  =====================================

Aggregated (non-disagg) cases are not supported at all. Every metric key is always present in
the returned dict; a group that the mode does not support is reported as ``0.0`` so the
OpenSearch document has a stable schema across modes.

Non-chunked prefill is treated as a single chunk, so group 2 is always populated for
``ctx_only``/``e2e`` -- the per-chunk numbers then simply describe the whole prefill.

Two properties of the data are relied on, both measured rather than assumed (see
``docs`` in ``compute_time_breakdown_metrics`` for the verification identities):

**Role is not in the filename.** Every worker writes ``perf_metrics-server-<host>-<pid>-*``
because ``openai_server.py`` falls back to ``"server"`` when ``server_role is None``. Files are
therefore classified by *content*: ``ctx_chunk_metrics`` => context worker, ``step_metrics``
=> generation worker. Do not use ``kv_cache_transfer_start`` -- the context worker records it
too, as the send side.

**Per-chunk / per-step timestamps use a different clock base than the request timestamps.**
``ctx_chunk_metrics`` / ``step_metrics`` timestamps come from a per-worker-process monotonic
clock whose origin differs from the ``timing_metrics`` base by a constant offset (measured on a
9-node GB300 run: ctx ``+294065.985 s``; the four gen workers ``+0.0003``, ``+377680.565``,
``+9.971``, ``+0.993 s`` -- each constant to ~10 us across 80 requests). Intra-instance spans
are offset-invariant, but the *first* chunk's / *first* step's preprocessing is anchored at
``first_scheduled_time`` and crosses the boundary. That offset is estimated per worker file and
removed; uncorrected, one worker's first-step preprocessing would read as ``+377680 s``.

**The writers are still running when the client exits.** Reading early truncates the
population *silently* -- every field is still populated and the row still uploads. Only the
generation workers have a completion sentinel, so ``wait_for_perf_metrics_files`` supplies the
missing gate for the context workers and the disaggregated server (size stability plus a
record census against the client's request count), and ``_read_jsonl`` skips a partial final
line rather than discarding the file it appears in.
"""

import argparse
import glob
import json
import math
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STATS = ("mean", "median", "p75", "p99")
METRIC_PREFIX = "d_tb_"

# --- group 1: context/prefill stage, one value per request -------------------------------
CTX_STAGE_SPANS = (
    ("ctx_preprocessing", "server_arrival_time", "arrival_time"),
    ("ctx_queue", "arrival_time", "first_scheduled_time"),
    ("ctx_processing", "first_scheduled_time", "first_token_time"),
    ("ctx_postprocessing", "first_token_time", "server_first_token_time"),
)

# --- group 4: generation/decode stage, one value per request -----------------------------
# ``gen_queue`` is the README-canonical span and *contains* the KV-cache transfer. The three
# sub-spans are the finer decomposition; they tile ``gen_queue`` exactly.
GEN_STAGE_SPANS = (
    ("gen_preprocessing", "server_arrival_time", "arrival_time"),
    ("gen_queue", "arrival_time", "first_scheduled_time"),
    ("gen_postprocessing", "first_scheduled_time", "server_first_token_time"),
    ("gen_queue_wait", "arrival_time", "kv_cache_transfer_start"),
    ("gen_kv_transfer", "kv_cache_transfer_start", "kv_cache_transfer_end"),
    ("gen_post_transfer", "kv_cache_transfer_end", "first_scheduled_time"),
)

# --- group 5: disagg server, one value per request ---------------------------------------
# Cross-role spans: (start side, start field, end side, end field). "disagg" means the field
# lives on the combined record itself.
DISAGG_STAGE_SPANS = (
    ("disagg_preprocessing", "disagg", "disagg_server_arrival_time", "ctx", "server_arrival_time"),
    ("disagg_relay", "ctx", "server_first_token_time", "gen", "server_arrival_time"),
    (
        "disagg_postprocessing",
        "gen",
        "server_first_token_time",
        "disagg",
        "disagg_server_first_token_time",
    ),
)

# --- groups 2 and 3: per-instance spans, many values per request -------------------------
# ``preprocessing`` is special: instance N starts at instance N-1's anchor, and instance 0
# starts at ``first_scheduled_time`` (which is why the clock offset matters).
_INSTANCE_SPANS = (
    ("forward", "forward_start_time", "forward_end_time"),
    ("update", "forward_end_time", "sample_start_time"),
    ("sample", "sample_start_time", "sample_end_time"),
    ("postprocessing", "sample_end_time", "token_time"),
)
# GPU fields are already in milliseconds (CUDA-event deltas), so they are not scaled.
_INSTANCE_GPU = (("gpu_forward", "gpu_forward_time"), ("gpu_sample", "gpu_sample_time"))

CHUNK_METRICS = (
    ("chunk_preprocessing",)
    + tuple(f"chunk_{n}" for n, _, _ in _INSTANCE_SPANS)
    + tuple(f"chunk_{n}" for n, _ in _INSTANCE_GPU)
)
STEP_METRICS = (
    ("step_preprocessing",)
    + tuple(f"step_{n}" for n, _, _ in _INSTANCE_SPANS)
    + tuple(f"step_{n}" for n, _ in _INSTANCE_GPU)
)

GROUP_METRICS: Dict[int, Tuple[str, ...]] = {
    1: tuple(n for n, _, _ in CTX_STAGE_SPANS),
    2: CHUNK_METRICS,
    3: STEP_METRICS,
    4: tuple(n for n, _, _ in GEN_STAGE_SPANS),
    5: tuple(n for n, *_ in DISAGG_STAGE_SPANS),
}

# Which groups each benchmark mode can produce. Aggregated cases support nothing.
MODE_GROUPS: Dict[str, Tuple[int, ...]] = {
    "ctx_only": (1, 2),
    "gen_only": (3, 4, 5),
    "e2e": (1, 2, 3, 4, 5),
}

ALL_METRICS: Tuple[str, ...] = tuple(m for g in sorted(GROUP_METRICS) for m in GROUP_METRICS[g])

# A per-instance "preprocessing" whose magnitude exceeds this is taken as evidence that the
# clock-base offset could not be removed, and is discarded rather than averaged in. Real
# values are sub-millisecond to seconds; a failed correction is tens of thousands of seconds.
_MAX_PLAUSIBLE_PREPROC_MS = 60_000.0
# Minimum records needed before a per-worker clock offset is trusted.
_MIN_OFFSET_SAMPLES = 3


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (same convention as ``numpy.percentile``)."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def _summarize(values: Iterable[float]) -> Optional[Dict[str, float]]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals.sort()
    return {
        "mean": float(statistics.fmean(vals)),
        "median": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "p99": _percentile(vals, 0.99),
    }


def _ts(container: Optional[Dict[str, Any]], field: str) -> Optional[float]:
    """Read a timestamp, mapping the tool's 'missing' encodings to ``None``.

    ``0`` and ``NaN`` both mean "endpoint not recorded" in this data, and
    ``TimingMetric.calculate_duration`` treats them as such. Returning ``None`` keeps them out
    of the aggregate instead of contributing a bogus zero-width or huge span.
    """
    if not container:
        return None
    val = container.get(field)
    if val is None or not isinstance(val, (int, float)):
        return None
    val = float(val)
    if val == 0.0 or math.isnan(val):
        return None
    return val


def _span_ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _timing(node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return ((node or {}).get("perf_metrics") or {}).get("timing_metrics") or {}


def _breakdown(node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (node or {}).get("time_breakdown_metrics") or {}


class _RecordView:
    """Uniform view over the three record shapes this tool may be handed.

    * **merged** -- ``ctx_perf_metrics`` / ``gen_perf_metrics`` / ``disagg_*`` (output of
      ``merge_disagg_perf_metrics.py``); carries request timestamps *and* chunk/step detail.
    * **disagg combined** -- same top-level shape, written by the disagg server itself;
      carries request timestamps, and chunk/step detail only for fields the header transport
      carries.
    * **plain worker** -- ``perf_metrics`` / ``time_breakdown_metrics`` at top level, one role
      only. Used directly for ``ctx_only`` (which runs the ctx worker in aggregated mode with
      no disagg server at all) and as the chunk/step source for ``e2e`` / ``gen_only``.
    """

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.is_combined = "ctx_perf_metrics" in raw or "gen_perf_metrics" in raw

    @property
    def ctx(self) -> Optional[Dict[str, Any]]:
        return self.raw.get("ctx_perf_metrics") if self.is_combined else self.raw

    @property
    def gen(self) -> Optional[Dict[str, Any]]:
        return self.raw.get("gen_perf_metrics") if self.is_combined else self.raw


def _classify(path: str, records: List[Dict[str, Any]]) -> str:
    """Return one of ``combined``, ``ctx_worker``, ``gen_worker``, ``empty``.

    Content-based on purpose: the filename's ``<kind>`` field is ``server`` for *every*
    worker, so it cannot distinguish ctx from gen.
    """
    if not records:
        return "empty"
    if any("ctx_perf_metrics" in r or "gen_perf_metrics" in r for r in records[:200]):
        return "combined"
    ctx_hits = sum(1 for r in records[:200] if "ctx_chunk_metrics" in _breakdown(r))
    gen_hits = sum(1 for r in records[:200] if "step_metrics" in _breakdown(r))
    if ctx_hits > gen_hits:
        return "ctx_worker"
    if gen_hits > ctx_hits:
        return "gen_worker"
    # No structured detail at all (num_postprocess_workers > 0 drops it). Fall back to the
    # only role-exclusive request field: kv_cache_transfer_end is written by gen only.
    if any(_ts(_timing(r), "kv_cache_transfer_end") for r in records[:200]):
        return "gen_worker"
    return "ctx_worker"


def _read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Parse a JSONL file, skipping unparsable lines. Returns ``(records, skipped)``.

    A malformed line is expected rather than exceptional: the client and every worker
    append to these files while the run is live, so the final line can be a partial
    write at the moment the aggregator reads (and, for a worker killed mid-flush, can
    stay partial forever). Dropping the whole file on one bad line is the worst possible
    response -- it zeroes the cross-role group and silently reroutes the per-request
    groups to same-role fallbacks, which still upload plausible-looking values. Skip the
    line and count it instead; the count is surfaced as a warning by the caller. This
    mirrors ``benchmark_serving._read_new_perf_metrics``, which also skips.
    """
    out: List[Dict[str, Any]] = []
    skipped = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(record, dict):
                out.append(record)
            else:
                skipped += 1
    return out, skipped


def _request_window(raw: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """``(first arrival, last completion)`` for one record, on that record's own clock.

    Only ever compared against other records from the *same* file, so no cross-worker
    clock correction is needed (or valid).
    """
    view = _RecordView(raw)
    arrivals = [
        ts
        for ts in (_ts(_timing(view.ctx), "arrival_time"), _ts(_timing(view.gen), "arrival_time"))
        if ts is not None
    ]
    ends = [
        ts
        for ts in (
            _ts(_timing(view.gen), "last_token_time"),
            _ts(_timing(view.ctx), "last_token_time"),
        )
        if ts is not None
    ]
    return (min(arrivals) if arrivals else None, max(ends) if ends else None)


def _drop_warmup_record(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Remove ``benchmark_serving``'s warmup record. Returns ``(kept, dropped_or_None)``.

    perf-sanity omits ``--no-test-input`` for the modes in its ``WARMUP_BENCHMARK_MODES``
    (``e2e`` and ``ctx_only`` -- the same two the time_breakdown modifier supports), so the
    client issues one un-measured request before the measured window: its "initial single
    prompt test run". ``benchmark_serving`` awaits that request, checks it for success and
    then discards it, so it is absent from ``completed`` and from every ``d_*`` client
    metric on the row -- but the servers still append a perf-metrics record for it, which
    is exactly why the client computes ``expected_count = completed + 1``. Left in, that
    one cold request would make ``d_tb_*`` the only family on the row computed over a
    different population than the rest.

    It is dropped per file because only the ctx worker and the gen worker that actually
    served it hold a record for it; the run's other workers must be left alone. The test
    is isolation, which is the one property no measured request has: the warmup request
    completes before the measured window opens, whereas measured requests arrive in a
    burst at the lane's concurrency and therefore always overlap. A file whose earliest
    record is not isolated is returned unchanged.

    Only the mean is materially at risk (a percentile over thousands of requests cannot be
    moved by one sample), and no ``d_tb_*`` metric is regression-gated, so this correction
    buys accuracy in the diagnostics rather than protecting a build.
    """
    if len(records) < 2:
        # A lone record cannot be shown to be isolated, and dropping it would leave the
        # file empty -- indistinguishable from a run that measured nothing.
        return records, None
    dated = [
        (window, raw)
        for window, raw in ((_request_window(r), r) for r in records)
        if window[0] is not None
    ]
    if len(dated) < 2:
        return records, None
    dated.sort(key=lambda item: item[0][0])
    (_, first_end), first_rec = dated[0]
    second_start = dated[1][0][0]
    if first_end is None or first_end >= second_start:
        return records, None
    return [r for r in records if r is not first_rec], first_rec


def _estimate_clock_offset(
    records: List[Dict[str, Any]], instances_key: str, reference_field: str
) -> Optional[float]:
    """Estimate ``instance clock base - timing_metrics clock base``, in seconds.

    The last instance's ``token_time`` and the request's ``reference_field`` denote the same
    physical moment, so their difference is the constant offset between the two bases. The
    median over requests is used so a single malformed record cannot move it.
    """
    samples = []
    for raw in records:
        view = _RecordView(raw)
        node = view.ctx if instances_key == "ctx_chunk_metrics" else view.gen
        instances = _breakdown(node).get(instances_key) or []
        ref = _ts(_timing(node), reference_field)
        if not instances or ref is None:
            continue
        last = _ts(instances[-1], "token_time")
        if last is not None:
            samples.append(last - ref)
    if len(samples) < _MIN_OFFSET_SAMPLES:
        return None
    return statistics.median(samples)


def _worker_key(raw: Dict[str, Any], role: str, fallback: str) -> str:
    """Identify the worker *process* a record's instance array came from.

    The clock-base offset is per process, so records must be grouped by process before the
    offset is estimated. A combined/merged record names its workers explicitly
    (``ctx_server`` / ``gen_server``), which matters because a single merged file mixes every
    worker together -- estimating one offset across N workers corrupts the first-instance
    preprocessing for N-1 of them. A plain worker file is already one process, so the file
    path is the key.
    """
    return str(raw.get(f"{role}_server") or fallback)


def _collect_instance_metrics(
    records: List[Dict[str, Any]],
    instances_key: str,
    reference_field: str,
    name_prefix: str,
    sink: Dict[str, List[float]],
    warnings: List[str],
    source: str,
) -> int:
    """Accumulate group 2 (chunks) or group 3 (steps), grouping records per worker process.

    All spans except the first instance's preprocessing are differences *within* the instance
    array and so are invariant to the clock base. The first instance's preprocessing is
    ``first_scheduled_time -> forward_start_time``, which crosses bases and needs the offset --
    estimated separately for each worker process present in ``records``.
    """
    role = "ctx" if instances_key == "ctx_chunk_metrics" else "gen"
    by_worker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in records:
        by_worker[_worker_key(raw, role, source)].append(raw)

    offsets: Dict[str, Optional[float]] = {}
    for key, group in by_worker.items():
        offsets[key] = _estimate_clock_offset(group, instances_key, reference_field)
        if offsets[key] is None:
            warnings.append(
                f"{key}: could not estimate the {name_prefix} clock-base offset "
                f"(<{_MIN_OFFSET_SAMPLES} usable records); first-instance "
                f"{name_prefix}_preprocessing is excluded"
            )
    if len(by_worker) > 1:
        spread = [o for o in offsets.values() if o is not None]
        if spread and max(spread) - min(spread) > 1.0:
            warnings.append(
                f"{source}: {len(by_worker)} {role} worker processes with clock-base offsets "
                f"spanning {max(spread) - min(spread):.3f}s; corrected per worker"
            )

    discarded = 0
    n_instances = 0
    for raw in records:
        offset = offsets[_worker_key(raw, role, source)]
        view = _RecordView(raw)
        node = view.ctx if instances_key == "ctx_chunk_metrics" else view.gen
        instances = _breakdown(node).get(instances_key) or []
        if not instances:
            continue
        n_instances += len(instances)
        anchor = _ts(_timing(node), "first_scheduled_time")
        for idx, inst in enumerate(instances):
            if idx == 0:
                if offset is not None and anchor is not None:
                    start = _ts(inst, "forward_start_time")
                    if start is not None:
                        val = (start - offset - anchor) * 1000.0
                        if abs(val) <= _MAX_PLAUSIBLE_PREPROC_MS:
                            sink[f"{name_prefix}_preprocessing"].append(val)
                        else:
                            discarded += 1
            else:
                val = _span_ms(
                    _ts(instances[idx - 1], "token_time"), _ts(inst, "forward_start_time")
                )
                if val is not None:
                    sink[f"{name_prefix}_preprocessing"].append(val)
            for name, start_f, end_f in _INSTANCE_SPANS:
                val = _span_ms(_ts(inst, start_f), _ts(inst, end_f))
                if val is not None:
                    sink[f"{name_prefix}_{name}"].append(val)
            for name, field in _INSTANCE_GPU:
                val = inst.get(field)
                if isinstance(val, (int, float)) and not math.isnan(float(val)):
                    sink[f"{name_prefix}_{name}"].append(float(val))
    if discarded:
        warnings.append(
            f"{source}: discarded {discarded} first-{name_prefix} preprocessing value(s) "
            f"exceeding {_MAX_PLAUSIBLE_PREPROC_MS:.0f} ms -- clock-base offset looks wrong"
        )
    return n_instances


def compute_time_breakdown_metrics(
    paths: Sequence[str],
    benchmark_mode: str,
    drop_warmup_request: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Reduce per-request JSONLs to ``{d_tb_<metric>_<stat>: value_ms}``.

    Args:
        paths: ``perf_metrics-*.jsonl`` files -- any mix of the disagg combined file, a merged
            file, and per-worker files. Multiple context and multiple generation workers are
            expected and handled: each file is reduced independently (which is what makes the
            per-worker clock-offset correction correct), and the resulting per-request /
            per-instance samples are pooled into one case-level distribution.
        benchmark_mode: ``ctx_only``, ``gen_only`` or ``e2e``. Anything else yields all-zero
            metrics, since aggregated cases do not support the time_breakdown tool.
        drop_warmup_request: set when the client ran with a warmup request (perf-sanity omits
            ``--no-test-input`` for ``e2e`` and ``ctx_only``). Discards that request's record
            so the breakdown covers the same population as every other metric on the row; see
            :func:`_drop_warmup_record`. Reported as ``info["warmup_dropped"]``.

    Returns:
        ``(metrics, info)``. ``metrics`` always has exactly ``len(ALL_METRICS) * 4`` keys;
        unsupported groups and groups with no usable sample are ``0.0``. ``info`` carries
        sample counts, per-file classification, warnings, and the verification identities.

    Verification identities held by construction; the caller may assert them:
        * groups 1+4+5 minus the ``gen_queue`` sub-spans tile the disagg-observed TTFT exactly;
        * the three ``gen_queue`` sub-spans sum to ``gen_queue`` exactly;
        * the five per-step spans tile the inter-token period exactly (so under an enabled
          overlap scheduler ``step_preprocessing`` is legitimately negative);
        * the five per-chunk spans sum to ``ctx_processing``.
    """
    groups = MODE_GROUPS.get(benchmark_mode, ())
    per_request: Dict[str, List[float]] = defaultdict(list)
    per_instance: Dict[str, List[float]] = defaultdict(list)
    warnings: List[str] = []
    classified: Dict[str, str] = {}
    counts: Dict[str, int] = defaultdict(int)
    warmup_dropped: Dict[str, int] = {}

    combined: List[Dict[str, Any]] = []
    ctx_workers: List[Tuple[str, List[Dict[str, Any]]]] = []
    gen_workers: List[Tuple[str, List[Dict[str, Any]]]] = []

    skipped_lines: Dict[str, int] = {}
    for path in paths:
        try:
            records, skipped = _read_jsonl(path)
        except OSError as exc:
            warnings.append(f"{os.path.basename(path)}: unreadable ({exc})")
            continue
        if skipped:
            skipped_lines[os.path.basename(path)] = skipped
            warnings.append(
                f"{os.path.basename(path)}: skipped {skipped} unparsable line(s) "
                f"(kept {len(records)}); a truncated final line is the usual cause"
            )
        if drop_warmup_request:
            records, dropped = _drop_warmup_record(records)
            if dropped is not None:
                warmup_dropped[os.path.basename(path)] = 1
        kind = _classify(path, records)
        classified[os.path.basename(path)] = f"{kind} (n={len(records)})"
        if kind == "combined":
            combined.extend(records)
        elif kind == "ctx_worker":
            ctx_workers.append((path, records))
        elif kind == "gen_worker":
            gen_workers.append((path, records))

    # ---- groups 1, 4, 5: one value per request --------------------------------------
    # Prefer the combined record: group 5 spans are cross-role and need the join. Without
    # it each stage falls back to the workers *of its own role* -- never to the other
    # role's. _RecordView aliases both .ctx and .gen to the raw record for a single-role
    # worker file, so driving group 4 off ctx workers would compute gen_preprocessing /
    # gen_queue / gen_postprocessing from the context worker's timestamps and upload
    # plausible millisecond values for the wrong phase. Resolving per role instead makes
    # a missing combined file cost the affected group its samples (a visible zero) rather
    # than silently mislabelling another role's.
    ctx_stage_records = combined or [r for _, rs in ctx_workers for r in rs]
    gen_stage_records = combined or [r for _, rs in gen_workers for r in rs]
    if 1 in groups:
        for raw in ctx_stage_records:
            ctm = _timing(_RecordView(raw).ctx)
            for name, start_f, end_f in CTX_STAGE_SPANS:
                val = _span_ms(_ts(ctm, start_f), _ts(ctm, end_f))
                if val is not None:
                    per_request[name].append(val)
    if 4 in groups:
        for raw in gen_stage_records:
            gtm = _timing(_RecordView(raw).gen)
            for name, start_f, end_f in GEN_STAGE_SPANS:
                val = _span_ms(_ts(gtm, start_f), _ts(gtm, end_f))
                if val is not None:
                    per_request[name].append(val)
    if 5 in groups:
        # Cross-role by construction, so only the combined record can carry it.
        for raw in combined:
            view = _RecordView(raw)
            if not view.is_combined:
                continue
            sides = {"ctx": _timing(view.ctx), "gen": _timing(view.gen), "disagg": raw}
            for name, s_side, s_field, e_side, e_field in DISAGG_STAGE_SPANS:
                val = _span_ms(_ts(sides[s_side], s_field), _ts(sides[e_side], e_field))
                if val is not None:
                    per_request[name].append(val)
    counts["ctx_stage_records"] = len(ctx_stage_records)
    counts["gen_stage_records"] = len(gen_stage_records)

    # ---- group 2: per-chunk, from every context worker ------------------------------
    if 2 in groups:
        sources = ctx_workers or ([("<combined>", combined)] if combined else [])
        for path, records in sources:
            counts["chunks"] += _collect_instance_metrics(
                records,
                "ctx_chunk_metrics",
                "first_token_time",
                "chunk",
                per_instance,
                warnings,
                os.path.basename(path),
            )
        counts["ctx_workers"] = len(sources)

    # ---- group 3: per-step, from every generation worker ----------------------------
    if 3 in groups:
        sources = gen_workers or ([("<combined>", combined)] if combined else [])
        for path, records in sources:
            counts["steps"] += _collect_instance_metrics(
                records,
                "step_metrics",
                "last_token_time",
                "step",
                per_instance,
                warnings,
                os.path.basename(path),
            )
        counts["gen_workers"] = len(sources)

    # ---- reduce; always emit every key so the OpenSearch schema is mode-stable ------
    metrics: Dict[str, float] = {}
    missing: List[str] = []
    for group, names in sorted(GROUP_METRICS.items()):
        pool = per_instance if group in (2, 3) else per_request
        for name in names:
            summary = _summarize(pool.get(name, [])) if group in groups else None
            if summary is None:
                if group in groups:
                    missing.append(name)
                summary = {stat: 0.0 for stat in STATS}
            for stat in STATS:
                metrics[f"{METRIC_PREFIX}{name}_{stat}"] = summary[stat]
    if missing:
        warnings.append(
            "supported by mode but no usable sample (reported as 0.0): "
            + ", ".join(sorted(missing))
        )

    info = {
        "benchmark_mode": benchmark_mode,
        "groups": list(groups),
        "files": classified,
        "counts": dict(counts),
        # Per file, so an unexpected pattern is visible: with a warmup request exactly one
        # ctx worker and one gen worker (plus the disagg server) should report a drop.
        "warmup_dropped": warmup_dropped,
        # Per file, non-empty only when a line failed to parse (usually a partial write).
        "skipped_lines": skipped_lines,
        "sample_counts": {
            name: len(
                (per_instance if name in CHUNK_METRICS + STEP_METRICS else per_request).get(
                    name, []
                )
            )
            for name in ALL_METRICS
        },
        "warnings": warnings,
    }
    return metrics, info


def discover_perf_metrics_files(output_dir: str) -> List[str]:
    """Find the run's per-request JSONLs, newest-last, under ``output_dir``.

    Looks in ``output_dir`` and a ``perf_metrics/`` subdirectory, which is where
    ``perf_metrics_output_dir`` puts them.
    """
    patterns = (
        os.path.join(output_dir, "perf_metrics-*.jsonl"),
        os.path.join(output_dir, "perf_metrics", "perf_metrics-*.jsonl"),
    )
    found: List[str] = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern)))
    # Deduplicate while preserving order, and drop empties so _classify never sees them.
    seen = set()
    result = []
    for path in found:
        real = os.path.realpath(path)
        if real not in seen and os.path.getsize(path) > 0:
            seen.add(real)
            result.append(path)
    return result


# Bounds for wait_for_perf_metrics_files. The gate exists to cover the *tail* of the
# writers' drain, not a hang: PerfMetricsJsonlWriter drains its queue continuously in a
# background thread (batch 64, no timer), so at the moment the client exits only the last
# handful of records are still in flight. Seconds is the right order of magnitude; the
# timeout is a backstop for a worker that is wedged, and expiring it is a warning rather
# than an error because reading a nearly-complete file still yields usable statistics.
COMPLETION_STABLE_SECONDS = 3.0
COMPLETION_TIMEOUT_SECONDS = 60.0
COMPLETION_POLL_SECONDS = 0.5


def _count_lines(path: str) -> int:
    """Number of newline-terminated lines in ``path``.

    Deliberately counts newlines, not records: a final line without a trailing newline is
    a partial write, and excluding it is exactly the semantics the completion check wants.
    """
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return total
            total += chunk.count(b"\n")


def wait_for_perf_metrics_files(
    output_dir: str,
    expected_requests: Optional[int] = None,
    stable_seconds: float = COMPLETION_STABLE_SECONDS,
    timeout_seconds: float = COMPLETION_TIMEOUT_SECONDS,
    poll_seconds: float = COMPLETION_POLL_SECONDS,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> Tuple[List[str], Dict[str, Any]]:
    """Wait for the perf_metrics JSONLs to stop growing, then return them.

    Positive completion gate for the aggregation. The harness has no completion signal for
    the context workers or the disaggregated server -- unlike the generation workers, whose
    ``gen_server_{i}.done`` sentinels the device-step-time path already waits on -- so
    without this the aggregator races the writers' tail flush and silently reduces a
    truncated population. Nothing downstream can notice: every metric is still populated
    and the row uploads green.

    Poll the discovered set's total byte size until it holds still for ``stable_seconds``
    (new files appearing counts as growth), bounded by ``timeout_seconds``. Then, if
    ``expected_requests`` is known, compare it against the largest file's complete-line
    count -- the disagg server and the aggregated server each write one record per request,
    so that file is the run's census -- and report a shortfall.

    ``sleep`` / ``monotonic`` are injected so the unit tests can drive this without wall
    time.

    Returns ``(paths, info)``; ``info`` carries ``stable`` (bool), ``waited_seconds``,
    ``total_bytes``, ``line_counts`` and ``warnings``.
    """
    warnings: List[str] = []
    deadline = monotonic() + timeout_seconds

    def snapshot() -> Tuple[List[str], Tuple[Tuple[str, Optional[int]], ...]]:
        paths = discover_perf_metrics_files(output_dir)
        sizes = []
        for path in paths:
            try:
                sizes.append((os.path.basename(path), os.path.getsize(path)))
            except OSError:
                # Raced with a rename/removal; ``None`` differs from any size, so the
                # next poll sees a change and the stability window restarts.
                sizes.append((os.path.basename(path), None))
        return paths, tuple(sizes)

    started = monotonic()
    paths, fingerprint = snapshot()
    if not paths:
        # Nothing was ever created, so there is nothing to wait for: the workers write
        # their first record long before the client exits. The caller reports the empty
        # discovery itself.
        return [], {
            "stable": True,
            "waited_seconds": 0.0,
            "total_bytes": 0,
            "line_counts": {},
            "expected_requests": expected_requests,
            "warnings": warnings,
        }
    unchanged_since = monotonic()
    stable = False
    while True:
        now = monotonic()
        if now - unchanged_since >= stable_seconds:
            stable = True
            break
        if now >= deadline:
            warnings.append(
                f"perf_metrics files under {output_dir} were still growing after "
                f"{timeout_seconds:.0f}s; aggregating what is on disk"
            )
            break
        sleep(min(poll_seconds, max(0.0, deadline - now)))
        paths, new_fingerprint = snapshot()
        if new_fingerprint != fingerprint:
            fingerprint = new_fingerprint
            unchanged_since = monotonic()

    line_counts: Dict[str, int] = {}
    for path in paths:
        try:
            line_counts[os.path.basename(path)] = _count_lines(path)
        except OSError as exc:
            warnings.append(f"{os.path.basename(path)}: could not count lines ({exc})")

    if expected_requests and line_counts:
        census = max(line_counts.values())
        if census < expected_requests:
            warnings.append(
                f"the largest perf_metrics file holds {census} complete record(s) but the "
                f"client issued {expected_requests} request(s); the breakdown covers a "
                "subset of the run"
            )

    info = {
        "stable": stable,
        "waited_seconds": monotonic() - started,
        "total_bytes": sum(size for _, size in fingerprint if size is not None),
        "line_counts": line_counts,
        "expected_requests": expected_requests,
        "warnings": warnings,
    }
    return paths, info


def format_metric_log_lines(metrics: Dict[str, float]) -> List[str]:
    """Render metrics as ``Time Breakdown <metric> <stat> (ms): <value>`` log lines.

    The harness re-parses these out of the benchmark log, the same way the ``gen_only``
    device-step-time statistics are transported.
    """
    lines = []
    for name in ALL_METRICS:
        for stat in STATS:
            key = f"{METRIC_PREFIX}{name}_{stat}"
            if key in metrics:
                lines.append(f"Time Breakdown {name} {stat} (ms): {metrics[key]:.6f}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(MODE_GROUPS) + ["aggr"],
        help="benchmark mode; 'aggr' yields all zeros (unsupported)",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="JSONL",
        help="a perf_metrics/merged JSONL; repeatable for multiple workers",
    )
    parser.add_argument("--output-dir", help="discover perf_metrics-*.jsonl under this dir")
    parser.add_argument("--json-out", help="write the metric dict here")
    parser.add_argument(
        "--log-lines",
        action="store_true",
        help="print 'Time Breakdown ...' lines for the harness to re-parse",
    )
    parser.add_argument(
        "--drop-warmup-request",
        action="store_true",
        help="discard the client's un-measured warmup request (perf-sanity runs one for "
        "e2e and ctx_only); pass this to match what the harness uploads",
    )
    args = parser.parse_args()

    paths = list(args.input)
    if args.output_dir:
        paths.extend(discover_perf_metrics_files(args.output_dir))
    if not paths:
        parser.error("no input: pass --input and/or --output-dir")

    metrics, info = compute_time_breakdown_metrics(
        paths, args.mode, drop_warmup_request=args.drop_warmup_request
    )

    if args.log_lines:
        for line in format_metric_log_lines(metrics):
            print(line)
    else:
        print(f"mode={info['benchmark_mode']} groups={info['groups']} counts={info['counts']}")
        for name, kind in sorted(info["files"].items()):
            dropped = " (-1 warmup)" if info["warmup_dropped"].get(name) else ""
            print(f"  {name}: {kind}{dropped}")
        header = f"{'metric':24s}" + "".join(f"{s:>12s}" for s in STATS) + f"{'n':>9s}"
        print(header)
        print("-" * len(header))
        for group, names in sorted(GROUP_METRICS.items()):
            print(f"-- group {group} --")
            for name in names:
                vals = "".join(f"{metrics[f'{METRIC_PREFIX}{name}_{s}']:12.3f}" for s in STATS)
                print(f"{name:24s}{vals}{info['sample_counts'][name]:9d}")
    for warning in info["warnings"]:
        print(f"WARNING: {warning}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({"metrics": metrics, "info": info}, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
