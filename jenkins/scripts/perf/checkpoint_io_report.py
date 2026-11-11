#!/usr/bin/env python3
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
"""Report on the checkpoint I/O startup experiment of perf-sanity CI.

Perf-sanity CI uploads one primary row per server startup to the perf-sanity OpenSearch project.
In post-merge pipelines the root build number assigns the checkpoint I/O policy: even builds load
with `native` and odd builds with `auto`, which selects rank-striped read-ahead for eligible loads
(see `docs/source/features/checkpoint-loading.md`). This script reads the rows back and compares
the two arms for every test case, GPU type and server role:

* effective policy: startups that activated rank-striped read-ahead against native controls,
  leaving out `auto` fallbacks and startups whose policy is unknown;
* intent to treat: startups of builds assigned `auto` against startups of builds assigned
  `native`, with `auto` fallbacks counted in the `auto` arm.

Every treatment startup is paired with the nearest control startup of the same test case, GPU
type, role, Jenkins pipeline, branch and runtime image at most `--pair-window` root builds away,
so both sides of a pair ran nearly the same code. The report gives the median paired change with a bootstrap confidence
interval, unpaired medians and percentiles, roll-ups per GPU type and model, and the reasons for
`auto` fallbacks. Rows recorded before the runtime image was uploaded get the image that
`jenkins/current_image_tags.properties` names at their commit.

Timings are rank 0's of each server and the maximum over the servers of a role. CI does not
record whether the checkpoint was already in the page cache, so pairs are also split by whether
their startups were the first load of the model in the job, a heuristic for cold and warm loads.

Example:
    export OPEN_SEARCH_DB_BASE_URL=<OpenSearch gateway of CI>
    python3 jenkins/scripts/perf/checkpoint_io_report.py --out-dir checkpoint_io_report
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_VERSION = "checkpoint-io-v4-postmerge-50-auto-50-native"
POSTMERGE_ASSIGNMENT = "postmerge_build_number"
RANK_STRIPED_ACTIVATED = "rank_striped_activated"
NATIVE_CONTROL = "native_control"
AUTO_FALLBACK = "auto_fallback"
UNKNOWN = "unknown"
IMAGE_TAGS_PATH = "jenkins/current_image_tags.properties"
SBSA_GPU_TYPES = frozenset({"gb200", "gb300", "gh200"})
_DISAGGREGATED_TEST_CASE = re.compile(r"^[a-z0-9_]+-[a-z0-9]+_(?P<model>.+?)_con\d+_")
_BUILD_URL_PIPELINE = re.compile(r"/job/LLM/job/(?P<pipeline>[^/]+)/")

ROLE_PREFIXES = {"server": "", "ctx": "ctx_", "gen": "gen_"}
"""Server roles of a startup row and the prefix of their fields."""

METRICS = (
    "checkpoint_pipeline_seconds",
    "total_model_loading_seconds",
    "checkpoint_preparation_seconds",
    "weight_population_seconds",
    "checkpoint_finalization_seconds",
)
"""Loader timings of a role, without the `d_` type and role prefixes."""

ROW_FIELDS = (
    "ts_created",
    "s_commit",
    "s_branch",
    "s_build_url",
    "s_gpu_type",
    "s_stage_name",
    "s_host_node_name",
    "s_test_case_name",
    "s_model_name",
    "s_runtime_image",
    "s_startup_observation_id",
    "s_checkpoint_io_experiment_version",
    "s_checkpoint_io_experiment_assigned_arm",
    "s_checkpoint_io_experiment_assignment_source",
    "l_checkpoint_io_experiment_root_build_number",
) + tuple(
    field
    for prefix in ROLE_PREFIXES.values()
    for field in (
        *(f"d_{prefix}{metric}" for metric in METRICS),
        f"s_{prefix}checkpoint_io_experiment_classification",
        f"s_{prefix}checkpoint_io_fallback_category",
        f"s_{prefix}checkpoint_io_fallback_reason",
        f"b_{prefix}startup_metrics_collection_complete",
        f"b_{prefix}startup_metrics_timing_complete",
    )
)
"""Row fields the report reads."""

Row = dict[str, Any]
ImageLookup = Callable[[str, str], str]


@dataclass(frozen=True)
class Observation:
    """One server role of one CI startup."""

    startup_id: str
    role: str
    test_case: str
    model: str
    gpu_type: str
    stage: str
    host: str
    pipeline: str
    branch: str
    commit: str
    build_url: str
    root_build: int
    created_ms: int
    assignment_source: str
    assigned_arm: str
    classification: str
    fallback_category: str
    fallback_reason: str
    timing_complete: bool
    metrics: dict[str, float]
    image: str = UNKNOWN
    image_source: str = UNKNOWN
    repeat_load: bool = False

    @property
    def post_merge(self) -> bool:
        """Whether the build number assigned the arm (the experiment's controlled comparison)."""
        return self.assignment_source == POSTMERGE_ASSIGNMENT


@dataclass(frozen=True)
class Comparison:
    """Two arms to compare, each defined by a predicate on observations."""

    name: str
    title: str
    treatment: str
    control: str
    is_treatment: Callable[[Observation], bool]
    is_control: Callable[[Observation], bool]


COMPARISONS = (
    Comparison(
        name="effective",
        title="Effective policy",
        treatment="RANK-STRIPED activated",
        control="NATIVE control",
        is_treatment=lambda obs: obs.classification == RANK_STRIPED_ACTIVATED,
        is_control=lambda obs: obs.classification == NATIVE_CONTROL,
    ),
    Comparison(
        name="intent_to_treat",
        title="Intent to treat",
        treatment="assigned auto",
        control="assigned native",
        is_treatment=lambda obs: obs.assigned_arm == "auto",
        is_control=lambda obs: obs.assigned_arm == "native",
    ),
)


@dataclass(frozen=True)
class Pair:
    """A treatment startup and the control startup it was matched with."""

    comparison: str
    treatment: Observation
    control: Observation
    log_ratio: float


@dataclass(frozen=True)
class CaseSummary:
    """The comparison of one test case, GPU type and role, over all its runtime images."""

    comparison: str
    test_case: str
    gpu_type: str
    role: str
    model: str
    n_images: int
    n_treatment: int
    n_control: int
    n_pairs: int
    median_log_ratio: float
    change: float
    ci_low: float
    ci_high: float
    share_treatment_faster: float
    treatment_median: float
    treatment_p10: float
    treatment_p90: float
    control_median: float
    control_p10: float
    control_p90: float


@dataclass(frozen=True)
class Rollup:
    """Test summaries of one comparison combined over a group of tests."""

    comparison: str
    group: str
    n_tests: int
    n_pairs: int
    geomean_change: float
    n_faster: int
    n_slower: int


@dataclass(frozen=True)
class LoadSplit:
    """Pairs of one comparison whose startups were first or repeated loads of their model."""

    comparison: str
    load: str
    n_pairs: int
    change: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class FallbackSummary:
    """The `auto` startups of one test case, GPU type and role that fell back to native."""

    gpu_type: str
    test_case: str
    role: str
    n_auto: int
    n_activated: int
    n_fallback: int
    n_post_merge_fallback: int
    categories: str
    top_reason: str
    example_reason: str


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _text(value: object, default: str = UNKNOWN) -> str:
    return value if isinstance(value, str) and value else default


def _model_from_test_case(test_case: str) -> str:
    """Return the model of a disaggregated test case such as `e2e-gb300_glm-5-fp4_8k1k_con64_...`."""
    match = _DISAGGREGATED_TEST_CASE.match(test_case)
    if match is None:
        return UNKNOWN
    return re.sub(r"_(\d+k\d+k|agentx)$", "", match.group("model"))


def _pipeline_from_url(build_url: str) -> str:
    """Return the Jenkins job of a build URL; root build numbers count within it."""
    match = _BUILD_URL_PIPELINE.search(build_url)
    return match.group("pipeline") if match else UNKNOWN


def _image_reference(value: object) -> str:
    image = _text(value)
    return image.split("@", 1)[0] if image != UNKNOWN else UNKNOWN


def observations_from_row(row: Row) -> list[Observation]:
    """Split one primary startup row into one observation per server role with timings."""
    build = row.get("l_checkpoint_io_experiment_root_build_number")
    root_build = build if isinstance(build, int) and not isinstance(build, bool) else -1
    image = _image_reference(row.get("s_runtime_image"))
    test_case = _text(row.get("s_test_case_name"))
    build_url = _text(row.get("s_build_url"), "")
    observations = []
    for role, prefix in ROLE_PREFIXES.items():
        metrics = {}
        for metric in METRICS:
            value = _number(row.get(f"d_{prefix}{metric}"))
            if value is not None:
                metrics[metric] = value
        if not metrics:
            continue
        observations.append(
            Observation(
                startup_id=_text(row.get("s_startup_observation_id")),
                role=role,
                test_case=test_case,
                model=_text(row.get("s_model_name"), _model_from_test_case(test_case)),
                gpu_type=_text(row.get("s_gpu_type")).lower(),
                stage=_text(row.get("s_stage_name")),
                host=_text(row.get("s_host_node_name")),
                pipeline=_pipeline_from_url(build_url),
                branch=_text(row.get("s_branch")),
                commit=_text(row.get("s_commit"), ""),
                build_url=build_url,
                root_build=root_build,
                created_ms=int(_number(row.get("ts_created")) or 0),
                assignment_source=_text(row.get("s_checkpoint_io_experiment_assignment_source")),
                assigned_arm=_text(row.get("s_checkpoint_io_experiment_assigned_arm")),
                classification=_text(row.get(f"s_{prefix}checkpoint_io_experiment_classification")),
                fallback_category=_text(row.get(f"s_{prefix}checkpoint_io_fallback_category")),
                fallback_reason=_text(row.get(f"s_{prefix}checkpoint_io_fallback_reason")),
                timing_complete=(
                    row.get(f"b_{prefix}startup_metrics_collection_complete") is True
                    and row.get(f"b_{prefix}startup_metrics_timing_complete") is not False
                ),
                metrics=metrics,
                image=image,
                image_source="row" if image != UNKNOWN else UNKNOWN,
            )
        )
    return observations


def _parse_properties(text: str) -> dict[str, str]:
    properties = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, value = stripped.partition("=")
            properties[key.strip()] = value.strip()
    return properties


def git_image_lookup(repo_root: Path) -> ImageLookup:
    """Return a lookup of the CI image that `jenkins/current_image_tags.properties` names.

    Args:
        repo_root: A git checkout that contains the commits of the rows.

    Returns:
        A function of a commit and an architecture (`sbsa` or `x86_64`) that returns the image
        reference, or `unknown` if the commit or the file is not in the checkout.
    """
    cache: dict[str, dict[str, str]] = {}

    def lookup(commit: str, architecture: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            return UNKNOWN
        if commit not in cache:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "show", f"{commit}:{IMAGE_TAGS_PATH}"],
                capture_output=True,
                text=True,
                check=False,
            )
            cache[commit] = _parse_properties(result.stdout) if result.returncode == 0 else {}
        key = "LLM_SBSA_DOCKER_IMAGE" if architecture == "sbsa" else "LLM_DOCKER_IMAGE"
        return cache[commit].get(key) or UNKNOWN

    return lookup


def _architecture(observation: Observation) -> str:
    url = observation.build_url.upper()
    if "SBSA" in url or "AARCH64" in url:
        return "sbsa"
    if "X86_64" in url:
        return "x86_64"
    return "sbsa" if observation.gpu_type in SBSA_GPU_TYPES else "x86_64"


def resolve_images(observations: Iterable[Observation], lookup: ImageLookup) -> list[Observation]:
    """Fill in the runtime image of post-merge observations whose row does not record one."""
    resolved = []
    for observation in observations:
        image = UNKNOWN
        if observation.image == UNKNOWN and observation.post_merge and observation.commit:
            image = lookup(observation.commit, _architecture(observation))
        resolved.append(
            observation
            if image == UNKNOWN
            else replace(observation, image=image, image_source="commit")
        )
    return resolved


def mark_repeat_loads(observations: Sequence[Observation]) -> list[Observation]:
    """Mark startups whose model an earlier startup of the same job loaded on the same host.

    A job is a root build of a Jenkins pipeline and a stage. Pre-merge rows and rows without a
    model are never marked.
    """

    def job_of(observation: Observation) -> tuple[str, int, str, str, str]:
        return (
            observation.pipeline,
            observation.root_build,
            observation.stage,
            observation.host,
            observation.model,
        )

    first_startup: dict[tuple[str, int, str, str, str], str] = {}
    for observation in sorted(observations, key=lambda obs: obs.created_ms):
        if observation.root_build >= 0 and observation.model != UNKNOWN:
            first_startup.setdefault(job_of(observation), observation.startup_id)
    return [
        replace(
            observation,
            repeat_load=first_startup.get(job_of(observation), observation.startup_id)
            != observation.startup_id,
        )
        for observation in observations
    ]


def pair_observations(
    treatment: Sequence[Observation], control: Sequence[Observation], window: int
) -> list[tuple[Observation, Observation]]:
    """Match startups of the two arms one-to-one, nearest root builds first.

    Args:
        treatment: Treatment startups that share a test case, GPU type, role, pipeline, branch
            and image.
        control: Control startups with the same test case, GPU type, role, pipeline, branch and
            image.
        window: The largest root-build distance allowed within a pair.

    Returns:
        Disjoint (treatment, control) pairs.
    """
    candidates = sorted(
        (abs(t.root_build - c.root_build), abs(t.created_ms - c.created_ms), i, j)
        for i, t in enumerate(treatment)
        for j, c in enumerate(control)
        if abs(t.root_build - c.root_build) <= window
    )
    used_treatment: set[int] = set()
    used_control: set[int] = set()
    pairs = []
    for _, _, i, j in candidates:
        if i not in used_treatment and j not in used_control:
            used_treatment.add(i)
            used_control.add(j)
            pairs.append((treatment[i], control[j]))
    return pairs


def percentile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated `q`-th percentile (0 to 100), or NaN for no values."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q / 100
    lower = ordered[math.floor(position)]
    upper = ordered[math.ceil(position)]
    return lower + (upper - lower) * (position - math.floor(position))


def bootstrap_median_interval(
    values: Sequence[float], iterations: int, rng: random.Random
) -> tuple[float, float]:
    """Return the 95% percentile-bootstrap interval of the median, or NaNs for under 2 values."""
    if len(values) < 2 or iterations < 1:
        return math.nan, math.nan
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(iterations)]
    return percentile(medians, 2.5), percentile(medians, 97.5)


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan


def compare(
    observations: Iterable[Observation],
    comparison: Comparison,
    metric: str,
    window: int,
    iterations: int,
    seed: int,
) -> tuple[list[CaseSummary], list[Pair]]:
    """Pair and summarize the post-merge startups of one comparison.

    Args:
        observations: Observations of all roles, with images resolved.
        comparison: The two arms to compare.
        metric: The timing to compare, one of `METRICS`.
        window: The largest root-build distance allowed within a pair.
        iterations: Bootstrap resamples per confidence interval.
        seed: Seed of the bootstrap; each test case gets its own generator.

    Returns:
        One summary per test case, GPU type and role with at least one startup in each arm, and
        all pairs.
    """
    arms: dict[tuple[str, ...], tuple[list[Observation], list[Observation]]] = defaultdict(
        lambda: ([], [])
    )
    for observation in observations:
        if not (observation.post_merge and observation.timing_complete):
            continue
        if observation.metrics.get(metric, 0.0) <= 0:
            continue
        test = (observation.test_case, observation.gpu_type, observation.role)
        key = (*test, observation.pipeline, observation.branch, observation.image)
        if comparison.is_treatment(observation):
            arms[key][0].append(observation)
        elif comparison.is_control(observation):
            arms[key][1].append(observation)

    pairs_by_test: dict[tuple[str, str, str], list[Pair]] = defaultdict(list)
    arms_by_test: dict[tuple[str, str, str], tuple[list[Observation], list[Observation]]] = (
        defaultdict(lambda: ([], []))
    )
    for key in sorted(arms):
        treatment, control = arms[key]
        test = key[:3]  # test case, GPU type, role
        arms_by_test[test][0].extend(treatment)
        arms_by_test[test][1].extend(control)
        for treated, untreated in pair_observations(treatment, control, window):
            log_ratio = math.log(treated.metrics[metric] / untreated.metrics[metric])
            pairs_by_test[test].append(Pair(comparison.name, treated, untreated, log_ratio))

    summaries = []
    for test in sorted(arms_by_test):
        treatment, control = arms_by_test[test]
        if not treatment or not control:
            continue
        pairs = pairs_by_test[test]
        ratios = [pair.log_ratio for pair in pairs]
        rng = random.Random(f"{seed}:{comparison.name}:{':'.join(test)}")
        low, high = bootstrap_median_interval(ratios, iterations, rng)
        median_ratio = _median(ratios)
        treatment_values = [obs.metrics[metric] for obs in treatment]
        control_values = [obs.metrics[metric] for obs in control]
        summaries.append(
            CaseSummary(
                comparison=comparison.name,
                test_case=test[0],
                gpu_type=test[1],
                role=test[2],
                model=Counter(obs.model for obs in treatment + control).most_common(1)[0][0],
                n_images=len({pair.treatment.image for pair in pairs}),
                n_treatment=len(treatment),
                n_control=len(control),
                n_pairs=len(pairs),
                median_log_ratio=median_ratio,
                change=math.expm1(median_ratio),
                ci_low=math.expm1(low),
                ci_high=math.expm1(high),
                share_treatment_faster=(
                    sum(ratio < 0 for ratio in ratios) / len(ratios) if ratios else math.nan
                ),
                treatment_median=_median(treatment_values),
                treatment_p10=percentile(treatment_values, 10),
                treatment_p90=percentile(treatment_values, 90),
                control_median=_median(control_values),
                control_p10=percentile(control_values, 10),
                control_p90=percentile(control_values, 90),
            )
        )
    return summaries, [pair for test in sorted(pairs_by_test) for pair in pairs_by_test[test]]


def roll_up(
    summaries: Iterable[CaseSummary], group_of: Callable[[CaseSummary], str], min_pairs: int
) -> list[Rollup]:
    """Combine test summaries with at least `min_pairs` pairs into groups.

    The change of a group is the geometric mean of its tests' median paired changes. A test
    counts as faster or slower when its confidence interval excludes zero.
    """
    groups: dict[tuple[str, str], list[CaseSummary]] = defaultdict(list)
    for summary in summaries:
        if summary.n_pairs >= min_pairs:
            groups[(summary.comparison, group_of(summary))].append(summary)
    return [
        Rollup(
            comparison=comparison,
            group=group,
            n_tests=len(members),
            n_pairs=sum(member.n_pairs for member in members),
            geomean_change=math.expm1(statistics.fmean(m.median_log_ratio for m in members)),
            n_faster=sum(member.ci_high < 0 for member in members),
            n_slower=sum(member.ci_low > 0 for member in members),
        )
        for (comparison, group), members in sorted(groups.items())
    ]


def split_by_load(pairs: Iterable[Pair], iterations: int, seed: int) -> list[LoadSplit]:
    """Pool the pairs of each comparison by whether their startups were repeated loads."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pair in pairs:
        repeats = (pair.treatment.repeat_load, pair.control.repeat_load)
        load = "first" if not any(repeats) else "repeat" if all(repeats) else "mixed"
        groups[(pair.comparison, load)].append(pair.log_ratio)
    splits = []
    for (comparison, load), ratios in sorted(groups.items()):
        rng = random.Random(f"{seed}:{comparison}:{load}")
        low, high = bootstrap_median_interval(ratios, iterations, rng)
        splits.append(
            LoadSplit(
                comparison=comparison,
                load=load,
                n_pairs=len(ratios),
                change=math.expm1(_median(ratios)),
                ci_low=math.expm1(low),
                ci_high=math.expm1(high),
            )
        )
    return splits


def _reason_pattern(reason: str) -> str:
    return re.sub(r"\d+", "N", reason)


def summarize_fallbacks(observations: Iterable[Observation]) -> list[FallbackSummary]:
    """Summarize the `auto` startups that fell back to native, pre- and post-merge."""
    groups: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.assigned_arm == "auto":
            groups[(observation.gpu_type, observation.test_case, observation.role)].append(
                observation
            )
    summaries = []
    for (gpu_type, test_case, role), members in groups.items():
        fallbacks = [obs for obs in members if obs.classification == AUTO_FALLBACK]
        if not fallbacks:
            continue
        reasons = Counter(_reason_pattern(obs.fallback_reason) for obs in fallbacks)
        top_reason = reasons.most_common(1)[0][0]
        categories = Counter(obs.fallback_category for obs in fallbacks)
        summaries.append(
            FallbackSummary(
                gpu_type=gpu_type,
                test_case=test_case,
                role=role,
                n_auto=len(members),
                n_activated=sum(obs.classification == RANK_STRIPED_ACTIVATED for obs in members),
                n_fallback=len(fallbacks),
                n_post_merge_fallback=sum(obs.post_merge for obs in fallbacks),
                categories=", ".join(f"{name} {count}" for name, count in categories.most_common()),
                top_reason=top_reason,
                example_reason=next(
                    obs.fallback_reason
                    for obs in fallbacks
                    if _reason_pattern(obs.fallback_reason) == top_reason
                ),
            )
        )
    return sorted(summaries, key=lambda s: (-s.n_fallback, s.gpu_type, s.test_case, s.role))


def _query(version: str, since_ms: int | None, until_ms: int | None, size: int) -> dict:
    filters: list[dict] = [
        {"term": {"s_checkpoint_io_experiment_version": version}},
        {"term": {"b_startup_observation_primary_row": True}},
    ]
    created: dict[str, int] = {}
    if since_ms is not None:
        created["gte"] = since_ms
    if until_ms is not None:
        created["lt"] = until_ms
    if created:
        filters.append({"range": {"ts_created": created}})
    return {
        "size": size,
        "_source": list(ROW_FIELDS),
        "sort": [{"ts_created": "asc"}, {"s_startup_observation_id": "asc"}],
        "query": {"bool": {"filter": filters}},
    }


def fetch_rows(
    version: str, since_ms: int | None, until_ms: int | None, page_size: int
) -> list[Row]:
    """Read the primary startup rows of one experiment version from OpenSearch.

    Reading needs `OPEN_SEARCH_DB_BASE_URL` but no credentials. The rows are paged with
    `search_after`, so there is no limit on their number.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from open_search_db import PERF_SANITY_PROJECT_NAME, OpenSearchDB

    rows: list[Row] = []
    query = _query(version, since_ms, until_ms, page_size)
    while True:
        response = OpenSearchDB.queryFromOpenSearchDB(query, PERF_SANITY_PROJECT_NAME)
        if response is None:
            raise RuntimeError(
                "The OpenSearch query failed; check that OPEN_SEARCH_DB_BASE_URL is set."
            )
        hits = response.json()["hits"]["hits"]
        rows.extend(hit["_source"] for hit in hits)
        print(f"Read {len(rows)} rows", file=sys.stderr)
        if len(hits) < page_size:
            return rows
        query["search_after"] = hits[-1]["sort"]


def load_rows(args: argparse.Namespace) -> list[Row]:
    """Read rows from the `--rows-json` cache when it matches the query, else from OpenSearch."""
    query = {"version": args.experiment_version, "since": args.since, "until": args.until}
    cache: Path | None = args.rows_json
    if cache is not None and cache.exists() and not args.refresh:
        saved = json.loads(cache.read_text())
        if saved.get("query") != query:
            raise ValueError(
                f"{cache} holds rows of {saved.get('query')}, not {query}; pass --refresh."
            )
        return saved["rows"]
    rows = fetch_rows(
        args.experiment_version, _date_ms(args.since), _date_ms(args.until), args.page_size
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"query": query, "rows": rows}))
    return rows


def _date_ms(date: str | None) -> int | None:
    if date is None:
        return None
    parsed = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _date_argument(text: str) -> str:
    try:
        _date_ms(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from error
    return text


def _regex_argument(text: str) -> re.Pattern[str]:
    try:
        return re.compile(text)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"invalid regular expression {text!r}: {error}") from error


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100 * value:+.1f}%"


def _seconds(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1f} s"


def _interval(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"{100 * low:+.1f} to {100 * high:+.1f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows
    ]
    return lines + [""]


def render_report(
    args: argparse.Namespace,
    rows: Sequence[Row],
    observations: Sequence[Observation],
    summaries: Sequence[CaseSummary],
    pairs: Sequence[Pair],
    fallbacks: Sequence[FallbackSummary],
) -> str:
    """Render the Markdown report."""
    post_merge = [obs for obs in observations if obs.post_merge]
    created = [obs.created_ms for obs in observations if obs.created_ms]
    titles = {comparison.name: comparison for comparison in COMPARISONS}
    lines = [
        "# Checkpoint I/O CI startup experiment: RANK-STRIPED vs NATIVE",
        "",
        f"- Experiment version `{args.experiment_version}`; metric `d_{args.metric}` (rank 0 of "
        "each server, the maximum over the servers of a role).",
        f"- {len(rows)} primary startup rows"
        + (f" created {_iso(min(created))} to {_iso(max(created))} UTC" if created else "")
        + f"; {len({obs.startup_id for obs in post_merge})} post-merge startups from "
        f"{len({(obs.pipeline, obs.root_build) for obs in post_merge})} root builds of the "
        + ", ".join(sorted({obs.pipeline for obs in post_merge}))
        + " pipelines.",
        "- Post-merge observations by classification: "
        + ", ".join(
            f"{name} {count}"
            for name, count in Counter(obs.classification for obs in post_merge).most_common()
        )
        + ".",
        "- Runtime image of post-merge observations, by source: "
        + ", ".join(
            f"{name} {count}"
            for name, count in Counter(obs.image_source for obs in post_merge).most_common()
        )
        + ".",
        f"- Pairs: same test case, GPU type, role, pipeline, branch and runtime image, at most "
        f"{args.pair_window} "
        f"root builds apart; 95% intervals from {args.bootstrap} bootstrap resamples. Tables "
        f"below list tests with at least {args.min_pairs} pairs.",
        "",
    ]
    for comparison in COMPARISONS:
        members = [s for s in summaries if s.comparison == comparison.name]
        lines += [
            f"## {comparison.title}: {comparison.treatment} vs {comparison.control}",
            "",
        ]
        rollups = (
            roll_up(members, lambda s: "all tests", args.min_pairs)
            + roll_up(members, lambda s: s.gpu_type, args.min_pairs)
            + roll_up(members, lambda s: f"{s.model} ({s.gpu_type})", args.min_pairs)
        )
        lines += _table(
            ("Group", "Tests", "Pairs", "Geomean change", "Faster", "Slower", "Inconclusive"),
            (
                (
                    r.group,
                    r.n_tests,
                    r.n_pairs,
                    _percent(r.geomean_change),
                    r.n_faster,
                    r.n_slower,
                    r.n_tests - r.n_faster - r.n_slower,
                )
                for r in rollups
            ),
        )
        listed = sorted(
            (s for s in members if s.n_pairs >= args.min_pairs),
            key=lambda s: (s.gpu_type, s.change),
        )
        lines += _table(
            (
                "Test case",
                "GPU",
                "Role",
                "Pairs",
                "Change",
                "95% interval",
                "Pairs faster",
                f"{comparison.treatment} median (p10-p90)",
                f"{comparison.control} median (p10-p90)",
            ),
            (
                (
                    s.test_case,
                    s.gpu_type,
                    s.role,
                    s.n_pairs,
                    _percent(s.change),
                    _interval(s.ci_low, s.ci_high),
                    f"{100 * s.share_treatment_faster:.0f}%",
                    f"{_seconds(s.treatment_median)} ({s.treatment_p10:.0f}-{s.treatment_p90:.0f})",
                    f"{_seconds(s.control_median)} ({s.control_p10:.0f}-{s.control_p90:.0f})",
                )
                for s in listed
            ),
        )
    lines += [
        "## First and repeated loads (heuristic)",
        "",
        "A startup is a repeated load if an earlier startup of the same root build, stage and "
        "host loaded the same model; its checkpoint is then likely in the page cache. Changes "
        "are medians over the pooled pairs of all tests.",
        "",
    ]
    lines += _table(
        ("Comparison", "Loads", "Pairs", "Median change", "95% interval"),
        (
            (
                titles[split.comparison].title,
                split.load,
                split.n_pairs,
                _percent(split.change),
                _interval(split.ci_low, split.ci_high),
            )
            for split in split_by_load(pairs, args.bootstrap, args.seed)
        ),
    )
    lines += ["## `auto` fallbacks to NATIVE", ""]
    lines += _table(
        (
            "GPU",
            "Test case",
            "Role",
            "auto startups",
            "Activated",
            "Fallbacks (post-merge)",
            "Categories",
            "Most common reason (example)",
        ),
        (
            (
                f.gpu_type,
                f.test_case,
                f.role,
                f.n_auto,
                f.n_activated,
                f"{f.n_fallback} ({f.n_post_merge_fallback})",
                f.categories,
                f.example_reason,
            )
            for f in fallbacks
        ),
    )
    lines += [
        "## Caveats",
        "",
        "- Observational CI data, not a controlled benchmark: CI does not record whether the "
        "checkpoint was in the page cache, and nodes, storage load and co-scheduled jobs vary.",
        "- Only successful launches are uploaded. Timings are rank 0's per server.",
        "- Rows record the runtime image only since PR #19174; older rows use the image named "
        "at their commit, which needs those commits in the local checkout.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        if not records:
            return
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _pair_record(pair: Pair, metric: str) -> dict[str, object]:
    return {
        "comparison": pair.comparison,
        "test_case": pair.treatment.test_case,
        "gpu_type": pair.treatment.gpu_type,
        "role": pair.treatment.role,
        "pipeline": pair.treatment.pipeline,
        "branch": pair.treatment.branch,
        "image": pair.treatment.image,
        "treatment_root_build": pair.treatment.root_build,
        "control_root_build": pair.control.root_build,
        "treatment_seconds": pair.treatment.metrics[metric],
        "control_seconds": pair.control.metrics[metric],
        "change": math.expm1(pair.log_ratio),
        "treatment_repeat_load": pair.treatment.repeat_load,
        "control_repeat_load": pair.control.repeat_load,
        "treatment_startup_id": pair.treatment.startup_id,
        "control_startup_id": pair.control.startup_id,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for the outputs")
    parser.add_argument(
        "--experiment-version",
        default=DEFAULT_EXPERIMENT_VERSION,
        help="value of s_checkpoint_io_experiment_version to read",
    )
    parser.add_argument(
        "--since", type=_date_argument, help="UTC date YYYY-MM-DD; rows created on or after it"
    )
    parser.add_argument(
        "--until", type=_date_argument, help="UTC date YYYY-MM-DD; rows created before it"
    )
    parser.add_argument(
        "--gpu", action="append", default=[], help="GPU type to keep, e.g. gb300; repeatable"
    )
    parser.add_argument(
        "--test-regex",
        type=_regex_argument,
        help="keep test cases that match this regular expression",
    )
    parser.add_argument("--metric", choices=METRICS, default=METRICS[0], help="timing to compare")
    parser.add_argument(
        "--pair-window",
        type=int,
        default=3,
        help="largest root-build distance within a pair; arms alternate, so 1 is adjacent",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=5, help="pairs a test needs for the tables and roll-ups"
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument(
        "--rows-json",
        type=Path,
        help="cache of the queried rows: read if it exists and matches the query, else written",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="query OpenSearch even if --rows-json exists"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="git checkout used to look up the CI image of each commit",
    )
    parser.add_argument("--page-size", type=int, default=2000, help="rows per OpenSearch query")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report and return the process exit code."""
    args = parse_args(argv)
    rows = load_rows(args)
    test_regex: re.Pattern[str] | None = args.test_regex
    gpus = {gpu.lower() for gpu in args.gpu}
    observations = [
        observation
        for row in rows
        for observation in observations_from_row(row)
        if (not gpus or observation.gpu_type in gpus)
        and (test_regex is None or test_regex.search(observation.test_case))
    ]
    observations = mark_repeat_loads(resolve_images(observations, git_image_lookup(args.repo_root)))
    summaries: list[CaseSummary] = []
    pairs: list[Pair] = []
    for comparison in COMPARISONS:
        comparison_summaries, comparison_pairs = compare(
            observations, comparison, args.metric, args.pair_window, args.bootstrap, args.seed
        )
        summaries += comparison_summaries
        pairs += comparison_pairs
    fallbacks = summarize_fallbacks(observations)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(
        render_report(args, rows, observations, summaries, pairs, fallbacks)
    )
    _write_csv(out_dir / "per_test.csv", [asdict(summary) for summary in summaries])
    _write_csv(out_dir / "pairs.csv", [_pair_record(pair, args.metric) for pair in pairs])
    _write_csv(out_dir / "fallbacks.csv", [asdict(fallback) for fallback in fallbacks])
    print(f"Wrote {out_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
