# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The brief a smith dispatch carries: the entry's surface, per call site, over the envelope.

A catalog entry is called from several places in a target, and each call site
reaches it with a different number of rows per request: a decode step passes
one row per request, a speculative verify step ``1 + draft`` rows, a
ratio-compressed path half of those, and an op after an attention-DP gather
sees every rank's rows. Under CUDA-graph capture each of those is multiplied
by every captured batch size ``G``. A smith certifying the entry at ``{G,
6G}`` -- "the decode rows and the verify rows" -- misses the call site that
passes ``3G`` and the one that passes ``5G``, and the delivery auditor, which
derives the multiple from the call chain, finds them uncovered at the end of
the run: seven verify dispatches to learn what the plan could have said.

So the plan states each entry's call sites in ``vocabulary.yaml``::

    steps:
      - step: compressed-KV latent
        entry: catalog/attention/compressor.py
        call_sites:
          - {site: "Compressor.forward, verify step", row_multiple: 3}
          - {site: "MoE.forward after the DP gather", row_multiple: 6, gathered: true}
          - {site: "_TailSlots, context rows only", eager_only: true}

and this brief turns them into the rows to certify: ``m*G`` for every
captured ``G`` (times the gather width under attention DP), the eager rows up
to the envelope's token budget, ``eager-only`` sites marked so neither the
smith nor the auditor reasons about capture for them again -- plus the
integer-width probes every verify owes, whatever today's envelope reaches.

``python -m agent_flow.workflows.staircase_onboard.smith_brief --workspace W
<request>`` prints the brief for a dispatch already written, the way an
operator relaying a turn needs it.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .envelope import CertificationPlan, compact_values

#: The keys one call site may carry.
CALL_SITE_KEYS: tuple[str, ...] = (
    "site",
    "row_multiple",
    "gathered",
    "eager_only",
    "row_width",
    "note",
)

#: 32-bit signed element offsets overflow here.
INT32_LIMIT = 2**31
#: Integers above this are not exactly representable in fp32.
FP32_EXACT_LIMIT = 2**24

_DISPATCH = re.compile(r"Dispatch (\d+): \*\*(\w+)\*\* the catalog entry `([^`]+)`")


@dataclass(frozen=True)
class CallSite:
    """One place the target calls an entry, and how many rows per request it passes."""

    site: str
    row_multiple: int = 1
    gathered: bool = False
    eager_only: bool = False
    row_widths: tuple[int, ...] = ()
    note: str = ""
    defaulted: bool = False


def entry_key(path: str) -> str:
    """``catalog/attention/x.py`` / ``attention/x`` / ``.../catalog/attention/x`` -> ``attention/x``.

    A target's own entry keeps its ``ops/`` prefix, so it never matches a
    shared entry of the same name: ``.../models/<family>/<target>/ops/moe/x.py``
    and ``ops/moe/x`` -> ``ops/moe/x``.
    """
    text = str(path).strip().removesuffix(".py").strip("/")
    if "/ops/" in text:
        return "ops/" + text.split("/ops/", 1)[1]
    if "catalog/" in text:
        text = text.split("catalog/", 1)[1]
    return text


def _widths(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    return tuple(int(v) for v in values if isinstance(v, int) and not isinstance(v, bool) and v > 0)


def parse_call_sites(step: Mapping[str, Any]) -> tuple[list[CallSite], list[str]]:
    """The call sites one vocabulary step declares, and what is wrong with them."""
    raw = step.get("call_sites")
    if raw is None:
        return [], []
    where = f"step {step.get('step', '?')!r}"
    if not isinstance(raw, list):
        return [], [f"{where}: call_sites must be a list"]
    sites: list[CallSite] = []
    errors: list[str] = []
    step_widths = _widths(step.get("row_width"))
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            errors.append(f"{where}: call_sites[{index}] must be a mapping")
            continue
        unknown = sorted(set(row) - set(CALL_SITE_KEYS))
        if unknown:
            errors.append(f"{where}: call_sites[{index}] has unknown key(s) {unknown}")
        site = str(row.get("site", "")).strip()
        if not site:
            errors.append(f"{where}: call_sites[{index}] names no `site`")
            continue
        multiple = row.get("row_multiple", 1)
        if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple < 1:
            errors.append(f"{where}: call_sites[{index}].row_multiple must be an integer >= 1")
            continue
        sites.append(
            CallSite(
                site=site,
                row_multiple=multiple,
                gathered=bool(row.get("gathered", False)),
                eager_only=bool(row.get("eager_only", False)),
                row_widths=_widths(row.get("row_width")) or step_widths,
                note=str(row.get("note", "")).strip(),
            )
        )
    return sites, errors


def steps_for(steps: Iterable[Mapping[str, Any]], target: str) -> list[Mapping[str, Any]]:
    """The vocabulary steps an entry serves."""
    key = entry_key(target)
    name = key.rsplit("/", 1)[-1]
    found = []
    for step in steps:
        entry = step.get("entry")
        if not entry:
            continue
        candidate = entry_key(str(entry))
        if candidate == key or ("/" not in key and candidate.rsplit("/", 1)[-1] == name):
            found.append(step)
    return found


def default_call_sites(plan: CertificationPlan | None) -> list[CallSite]:
    """What a brief assumes when the plan declared no call sites for the entry."""
    sites = [CallSite(site="decode step", row_multiple=1, defaulted=True)]
    draft = plan.envelope.max_draft_len if plan is not None else 0
    if draft:
        sites.append(
            CallSite(site="speculative verify step", row_multiple=1 + draft, defaulted=True)
        )
    return sites


def gather_widths(plan: CertificationPlan | None) -> list[tuple[str, int]]:
    """``[(member, world), ...]`` for every attention-DP member the envelope claims."""
    if plan is None:
        return []
    return sorted(
        {(m.name, m.world_size) for m in plan.envelope.members if m.attention_dp},
        key=lambda item: (item[1], item[0]),
    )


def captured_rows(site: CallSite, batches: Sequence[int]) -> list[int]:
    """Rows a captured call at this site sees, per rank, for every captured batch."""
    if site.eager_only:
        return []
    return sorted({site.row_multiple * g for g in batches})


def _multiple(m: int) -> str:
    return "G" if m == 1 else f"{m}·G"


def integer_width_probes(widths: Iterable[int], *, reachable_rows: int | None) -> list[str]:
    """The mandatory boundary probes, with the row counts they imply."""
    lines = [
        f"total elements at {INT32_LIMIT} (2^31): an int32 element offset or count wraps here",
        f"any index that passes through fp32 at {FP32_EXACT_LIMIT + 1} (2^24 + 1): above "
        f"2^24 an fp32 index is no longer exact",
    ]
    for width in sorted(set(widths)):
        rows = INT32_LIMIT // width
        reach = ""
        if reachable_rows is not None:
            reach = (
                " -- INSIDE the envelope's reachable rows"
                if reachable_rows >= rows
                else f" (the envelope reaches {reachable_rows} rows)"
            )
        lines.append(
            f"rows x width {width}: 2^31 / {width} = {rows} rows, probe at {rows + 1}{reach}"
        )
    if not widths:
        lines.append(
            "rows x row width: 2^31 / <row width> rows, for every row width this entry "
            "addresses (the call sites declare none -- read them from the entry's raw_call "
            "signature and its cells' specs)"
        )
    return lines


def render_smith_brief(
    item: Any,
    *,
    steps: Sequence[Mapping[str, Any]] = (),
    plan: CertificationPlan | None = None,
    queue_row: Mapping[str, Any] | None = None,
) -> str:
    """The brief for one smith dispatch."""
    target = str(item.target)
    lines = [f"## Smith brief: dispatch {item.n}, {item.mode} `{target}`", ""]
    row = dict(queue_row or {})
    for key in ("tier", "unit", "depends_on", "surface", "note"):
        value = row.get(key, getattr(item, key, None))
        if value not in (None, "", (), []):
            lines.append(f"- {key}: {value}")

    batches = plan.certified_capture_batches if plan is not None else ()
    env = plan.envelope if plan is not None else None
    if env is not None:
        kind = "declared" if env.declared else "DERIVED from the gate tuples (legacy)"
        lines += [
            "",
            f"### Envelope ({kind})",
            "",
            f"- tokens per step per rank T <= {env.max_num_tokens}; requests per step <= "
            f"{env.max_batch_size}; per-request length <= {env.max_seq_len}",
            f"- captured batch sizes G: {compact_values(batches) or 'none (eager only)'}",
        ]

    sites: list[CallSite] = []
    errors: list[str] = []
    for step in steps_for(steps, target):
        found, problems = parse_call_sites(step)
        sites.extend(found)
        errors.extend(problems)
    if not sites:
        sites = default_call_sites(plan)
    gathers = gather_widths(plan)

    lines += ["", "### Call sites and the rows each one passes", ""]
    if any(site.defaulted for site in sites):
        lines += [
            "`vocabulary.yaml` declares no `call_sites` for this entry, so these are "
            "ASSUMED (decode, and the verify step when the envelope speculates). Confirm "
            "them against the call chain; a site this list misses is what the delivery "
            "auditor will find uncovered.",
            "",
        ]
    lines += [
        "| call site | row multiple | captured rows per rank | after an attention-DP gather |",
        "|---|---|---|---|",
    ]
    union: set[int] = set()
    eager_rows: set[int] = set()
    for site in sites:
        if site.eager_only:
            lines.append(
                f"| {site.site} | eager-only | never captured | "
                f"{'gathered' if site.gathered else '-'} |"
            )
        else:
            rows = captured_rows(site, batches)
            union.update(rows)
            gathered = "-"
            if site.gathered and gathers:
                parts = []
                for member, world in gathers:
                    wide = sorted({world * r for r in rows})
                    union.update(wide)
                    parts.append(
                        f"{member}: {_multiple(world * site.row_multiple)} = "
                        f"{compact_values(wide) or '-'}"
                    )
                gathered = "; ".join(parts)
            lines.append(
                f"| {site.site} | {_multiple(site.row_multiple)} | {compact_values(rows) or '-'} | "
                f"{gathered} |"
            )
        if env is not None and env.max_num_tokens:
            widest = max((w for _, w in gathers), default=1) if site.gathered else 1
            eager_rows.add(env.max_num_tokens * widest)
    if errors:
        lines += ["", "Malformed `call_sites` (fix them in `vocabulary.yaml`):", ""]
        lines += [f"- {error}" for error in errors]
    lines += [
        "",
        f"**Certify every captured row count:** {compact_values(sorted(union)) or '(none: eager only)'}"
        " -- replay on new data bitwise equal to eager at each.",
    ]
    if eager_rows:
        lines.append(
            f"**And every eager row count 1..{max(eager_rows)}** by the envelope standard "
            "(grid-sized rows: a run at the maximum plus prefix-bitwise; bucketed kernels: "
            "every bucket's edges)."
        )

    widths = sorted({w for site in sites for w in site.row_widths})
    reachable = max(eager_rows | union) if (eager_rows or union) else None
    lines += [
        "",
        "### Integer-width boundary probes (mandatory)",
        "",
        "Probe every index-typed argument at these boundaries, whether or not today's "
        "envelope reaches them, and record each outcome in the entry: a boundary the op "
        "crosses silently wrong becomes an `is_valid` guard with a test proving it refuses; "
        "one it raises at, or passes, goes in the entry's `note` under `Hard limits`:",
        "",
        *(f"- {line}" for line in integer_width_probes(widths, reachable_rows=reachable)),
    ]
    return "\n".join(lines) + "\n"


def load_steps(vocabulary_path: Path | str) -> list[Mapping[str, Any]]:
    """The vocabulary's steps, or nothing if the file is absent or unreadable."""
    path = Path(vocabulary_path)
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def load_queue_row(vocabulary_path: Path | str, n: int) -> dict[str, Any]:
    path = Path(vocabulary_path)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    queue = data.get("queue") if isinstance(data, dict) else None
    for row in queue or ():
        if isinstance(row, dict) and str(row.get("n")) == str(n):
            return dict(row)
    return {}


@dataclass(frozen=True)
class _Dispatch:
    n: int
    mode: str
    target: str


def read_dispatch(workspace: Path, request: str) -> _Dispatch:
    """The dispatch number, mode and target of a smith request already written."""
    rid = request if request.endswith("-smith") else f"{request}-smith"
    for candidate in (rid, request):
        path = workspace / "dispatch" / f"{candidate}.prompt.md"
        if path.is_file():
            match = _DISPATCH.search(path.read_text(encoding="utf-8"))
            if match is None:
                raise ValueError(f"{path} names no catalog-entry dispatch")
            return _Dispatch(n=int(match.group(1)), mode=match.group(2), target=match.group(3))
    raise FileNotFoundError(f"no dispatch prompt for {request!r} under {workspace / 'dispatch'}")


def _plan_for_workspace(workspace: Path) -> CertificationPlan | None:
    from .envelope import derive_certification
    from .task_schema import (
        TaskSchemaError,
        load_and_validate_task_yaml,
        members_for,
        parse_parallel_segment,
    )

    task = workspace / "task.yaml"
    if not task.is_file():
        return None
    try:
        data = load_and_validate_task_yaml(task)
    except TaskSchemaError:
        return None
    _, size = parse_parallel_segment(str(data.get("parallel")))
    if size is None:
        _, size = parse_parallel_segment(members_for(data, None, 4)[0])
    return derive_certification(data, world_size=size or 4)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent_flow.workflows.staircase_onboard.smith_brief",
        description="Print the smith brief for a dispatch: call sites, rows, probes.",
    )
    parser.add_argument("--workspace", required=True, help="the run's workspace directory")
    parser.add_argument("request", help="a request id (e.g. 075-smith) or a dispatch number")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    try:
        dispatch = read_dispatch(workspace, args.request)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    vocabulary = workspace / "vocabulary.yaml"
    sys.stdout.write(
        render_smith_brief(
            dispatch,
            steps=load_steps(vocabulary),
            plan=_plan_for_workspace(workspace),
            queue_row=load_queue_row(vocabulary, dispatch.n),
        )
    )
    return 0


__all__ = [
    "CALL_SITE_KEYS",
    "FP32_EXACT_LIMIT",
    "INT32_LIMIT",
    "CallSite",
    "captured_rows",
    "default_call_sites",
    "entry_key",
    "gather_widths",
    "integer_width_probes",
    "load_queue_row",
    "load_steps",
    "parse_call_sites",
    "read_dispatch",
    "render_smith_brief",
    "steps_for",
]


if __name__ == "__main__":
    sys.exit(_main())
