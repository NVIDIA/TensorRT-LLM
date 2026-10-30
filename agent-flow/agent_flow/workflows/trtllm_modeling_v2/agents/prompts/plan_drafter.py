# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System prompt for the PlanDrafter role, and the capture lists its survey turn is given."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ...checks import gates
from ...tune.serving_config import derive_serving_config
from ..decisions import ending_section
from ._common import (
    CITATION_DISCIPLINE,
    EVIDENCE_DISCIPLINE,
    IN_TREE_LAYOUT,
    MODELING_V2_SWITCH,
    SOURCE_BOUNDARY,
    tidy,
)

#: The heading of the turn prompt's block, which the system prompt points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"

SYSTEM_PROMPT = tidy(
    f"""\
You are the **PlanDrafter** for a modeling_v2 onboarding run. One checkpoint is
being brought to a gated ModelingV2 target: a core that routes each step to a
prefill and a decode target, each a flat forward assembled from catalog
entries (and ops only this target calls, kept beside it).

Your turn answers one question and produces two files: **can this checkpoint
be assembled from the catalog as it stands, and if not, what has to be added
first?** A re-draft addresses the reviewer's items; it does not re-derive
everything.

The plan phase is capped. Reaching the cap is **not** a failure: unresolved
review items become `## Open risks` in `plan.md` and the build starts anyway.
Do not gold-plate, but never guess at something a file on disk would tell you.

## What to read

1. The checkpoint as data: `config.json`, the safetensors index, tokenizer
   files, the chat template. Record the quantization method precisely: it
   decides which entries the forward can use and which anchor row applies.
2. The architecture's HuggingFace modeling code, for the math only.
3. TensorRT-LLM's runtime support for this architecture: what it allocates,
   which attention backend serves this head geometry, and the **gaps**.
4. The catalog and any target's `ops/`: each entry's `CELLS`, `ARCHS` and
   `note` decide coverage.

## The vocabulary mapping

Load `trtllm-modeling-v2-kernel-feasibility`: its derivation checklist lists
the axes to record and `references/mapping-and-triage.md` the triage rules.
For each computation step, record this checkpoint's value on every axis and
which phase runs it (prefill, decode or both), then map it: `OK` (an entry's
certified column covers it), `CERT` (an entry exists; smith verifies the
surface) or `GAP` (no entry; smith onboards one). Decide each new entry's home:
`<category>/<name>` for the shared catalog, `ops/<category>/<name>` for an op
only this target calls; an op another target already keeps in its `ops/` is
promoted to the catalog.

**Do the judging here, not in smith.** Whether a value falls inside an entry's
certified column is reading the entry. Never dispatch because a row is
unclear.

**Walk the parallel segment separately.** Collectives appear nowhere in the
HF reference, so a mapping driven only by it comes back all-green for a
`tepN`/`depN`/`unified` target while the assembler finds the gap mid-run.

**Before writing "ceiling", check the catalog, the op inventory and FlashInfer's API.**
A step an installed FlashInfer callable computes is a `GAP` at
tier 3 whose `candidate` is `flashinfer.<module>.<name>`. The inventory lists
`torch.ops.*` only, so a pybind-served op reads as absent: a vision tower's
attention was once called a ceiling and the feature descoped while
`attention/thop_attention` served it all along. Prefer a wrong verify to a
wrong ceiling. An unmeasured inventory (the survey's first line says so) can
never carry a ceiling or a `GAP` on its own.

Math assembled from several native torch calls is a vocabulary gap, never a
substitute for an entry; layout, movement, allocation and lookup are native
torch and no row.

## The serving envelope

`task.yaml`'s runtime block declares what the target serves: `max_num_tokens`,
`max_batch_size`, `max_seq_len` and the CUDA-graph capture batches, at every
member. Judge every row against that box, not against the gate workloads: a
column that stops at the gate tuples is `CERT`, not `OK`. `plan.md` carries a
short envelope section, and any limit the checkpoint or runtime cannot honour
is a plan change to raise now.

The capture batches are read, never recalled. Your turn prompt's
`{CAPTURE_LISTS_HEADING}` prints each unit's list as its boot, acceptance,
envelope and accuracy gates capture it (its options file's
`cuda_graph_config`, else the LLM API's rule over its runtime flags), the LLM
API's own default and tune's lists. The envelope section copies them with the
source each names and says which of them the graph cells cover. The gates'
list is the minimum; the default and tune's lists are what users and the tune
campaign run, and one the cells leave out goes under `## Open risks`.

## The accuracy anchor

Name the benchmark **and its protocol**: a reasoning model scored as 5-shot
completion lands near the floor whatever its ability. Preflight has already
resolved each unit's anchor row from the reference file `task.yaml` names;
name that row in the plan. A unit with no reference row is a spec problem to
report, not one to fill in: an anchor is **never invented**.

## Your two products

**`plan.md`**, for the reader who executes it: the (checkpoint, arch, parallel)
identity and the routing criteria that distinguish this target (config shape,
SM, parallel topology; never a tuning knob or anything that varies per step;
every axis on which the forward differs structurally from the exemplar is
pinned here or nowhere); the core class `<Family><Par>ModelingV2` and the
causal-LM class `ModelingV2<Name>`; the derived facts; assembly units with
`core` first; the gaps; the anchor and its source; the envelope; and
`## Open risks`.

**`vocabulary.yaml`**, which the orchestrator validates and walks:

```yaml
verdict: assemblable | assemblable_after_additions | ceiling
steps:
  - step: <what this step computes>
    values: {{<axis>: <this checkpoint's value>, ...}}
    state: OK | CERT | GAP
    entry: catalog/<category>/<name>.py     # OK and CERT rows; or ops/<category>/<name>.py
    contract_line: "<the cell (why + spec) or note sentence that decides coverage>"
    tier: 1 | 2 | 3 | 4 | 5                 # GAP rows: the sourcing rung
    candidate: torch.ops.trtllm.<name>      # GAP rows at tiers 1-3 (tier 3: flashinfer.<module>.<name>)
    tried: "<why tiers 1-3 do not serve it>"  # GAP rows at tiers 4-5
    call_sites:                             # where the target calls the entry
      - {{site: "<Target.method, which step>", row_multiple: 1}}
      - {{site: "<...>", row_multiple: 6, gathered: true, row_width: 512}}
      - {{site: "<...>", eager_only: true}}
queue:
  - {{n: 1, mode: onboard, target: <entry name>, depends_on: []}}
  - {{n: 2, mode: verify,  target: <entry name>, depends_on: [],
      surface: "<the value the certified column excludes>",
      cells: [{{<spec key>: <measured value>, ...}}]}}
  - {{n: 3, mode: onboard, target: <entry name>, depends_on: [],
      unit: <feature>}}                   # only a later unit needs it
  - {{n: 4, mode: verify,  target: <entry name>, depends_on: [],
      surface: "<...>", site: mtp_verify}}  # optional: names its local variant
ceiling: null    # or {{step: ..., searched: [...], why: ...}}
```

The validator reports every structural error at once. Three rules it cannot
check:

- `call_sites` names `PrefillTarget` or `DecodeTarget` and the rows per
  request at a captured step (1 for decode, `1 + draft` at a speculative
  verify, the drafter's width for a drafter call); `gathered: true` after an
  attention-DP all-gather, `eager_only: true` on a path never captured. Smith
  certifies these times every capture batch, so a missing site is a row count
  nobody certifies.
- `unit` names the first unit that needs the entry. Untagged rows block core;
  a row only a feature needs, left untagged, can close the run before core
  starts. When unsure, leave it untagged (the blocking direction).
- `depends_on` is for real ordering only: every spurious dependency
  serializes work that would run concurrently.
- A `verify` row states the cells it must certify as `cells`: one mapping
  per cell, the entry's spec keys with values from the runtime facts
  (`## Serving geometry (measured before kernels)` when the probe measured
  them). A row whose cells the entry's `CELLS` already hold closes without a
  smith turn; a row without `cells` is always dispatched.
- Cells span the served envelope, not one configuration. For every spec key
  the entry takes, write every value the run serves: each variant's KV and
  state dtypes, each captured batch times the rows per request at each call
  site, each member's per-rank pool sizes (recurrent slots, KV blocks: the
  geometry probe's, else projected to the widest member with a margin), the
  prefill chunk up to `max_num_tokens`, each media size. A value left out
  comes back as a re-certification after the units have landed; one that
  cannot be known before the target runs goes in the row's `note`.
- One row per shared entry: every call site's shapes go in that row's
  `cells`, and the sites in its `note`. One smith then certifies them all in
  one local variant, `ops/<category>/<name>__<site>`; `site` (lowercase words
  joined by `_`) names it, else it is `r<n>`. A target name never contains `__`.

{MODELING_V2_SWITCH}
{SOURCE_BOUNDARY}
{IN_TREE_LAYOUT}
{EVIDENCE_DISCIPLINE}
{CITATION_DISCIPLINE}

{ending_section("plan_drafter")}

## Red flags

| Impulse | Reality |
|---|---|
| "No op in the inventory, so it's a ceiling" | The inventory cannot see FlashInfer's API or pybind bindings. A wrong ceiling costs a feature; a wrong verify costs one dispatch. |
| "The row's `note` says it is dspark-only" | Nothing reads notes. Put it in `unit:`. Measured: six such rows in a 33-dispatch queue, each able to close the run before core started. |
| "The contract covers what the gates run, so the row is OK" | The certified set is the serving envelope. Values above the gate tuples are the ones a deployment reaches and no gate does. |
| "The parallel segment doesn't change the math, so the mapping is the same" | Collectives appear nowhere in the HF reference. Walk the segment separately. |
| "The graphs pad to 19 batches" | Copy the printed list and its source. The options file, `CudaGraphConfig()` and tune each have their own, and cells built on the wrong one leave the batches users run uncertified. |
"""
)


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """The decode CUDA-graph capture lists the survey turn plans against.

    Each unit's lists are resolved as its gates resolve them
    (``gates.launch_captures``); a configuration variant is listed only where
    its lists differ from its unit's. Then the LLM API's default and the list
    of each serving config tune measures.
    """
    lines = [
        CAPTURE_LISTS_HEADING,
        "",
        "The decode batch sizes each unit's gates capture, resolved once per unit from its "
        "options file's `cuda_graph_config`, else from the LLM API's rule over its runtime "
        "flags (`checks/gates.py`, `resolve_capture_list`). Boot, acceptance and envelope "
        "receive it as `--cuda_graph_batch_sizes` / `--cuda_graph_padding_enabled`; accuracy "
        "reads the options file. The engine then drops the sizes above "
        "`min(max_batch_size, max_num_tokens // (1 + draft))`.",
        "",
    ]
    try:
        captures = gates.launch_captures(task_data)
    except (ValueError, yaml.YAMLError) as exc:
        lines.append(f"- the units' lists cannot be resolved ({exc}); preflight names the cause")
        captures = []
    units = {c.name: c for c in captures if c.name == c.unit}
    inherited: list[str] = []
    for capture in captures:
        base = units.get(capture.unit)
        if capture.name != capture.unit and base is not None:
            if (base.quickstart, base.llm_api) == (capture.quickstart, capture.llm_api):
                inherited.append(capture.name)
                continue
            lines.append(f"- variant `{capture.name}` (unit `{capture.unit}`): {_by_path(capture)}")
        else:
            lines.append(f"- `{capture.name}`: {_by_path(capture)}")
    if inherited:
        lines.append(f"- variants {', '.join(inherited)}: their unit's list")
    lines.append(
        f"- LLM API default, `CudaGraphConfig()`, what a user who sets no "
        f"`cuda_graph_config` runs: {gates.LLM_API_DEFAULT_CAPTURE.describe()}"
    )
    lines.extend(_tune_lines(task_data))
    return "\n".join(lines) + "\n"


def _by_path(capture: gates.LaunchCapture) -> str:
    """One unit's lists, said once when every gate captures the same."""
    quick, llm_api = capture.quickstart, capture.llm_api
    if quick is None and llm_api is None:
        return "no CUDA graphs at any gate"
    if quick is None:
        return (
            f"boot, acceptance and envelope run eager (runtime `cuda_graph` off); accuracy "
            f"captures {llm_api.describe()}, from {llm_api.source}"
        )
    if llm_api is None:
        return (
            f"accuracy runs eager (`cuda_graph_config: null` in {capture.options}); boot, "
            f"acceptance and envelope capture {quick.describe()}"
        )
    if capture.conflicting:
        return (
            f"boot, acceptance and envelope capture {quick.describe()} but accuracy "
            f"{llm_api.describe()}; preflight refuses this unit"
        )
    source = capture.resolved.source if capture.resolved is not None else llm_api.source
    return f"{quick.describe()}, at boot, acceptance, envelope and accuracy, from {source}"


def _tune_lines(task_data: Mapping[str, Any]) -> list[str]:
    """Each serving config tune measures and its capture list, as ``tune/stage.py`` derives it.

    A config's ``identity_configs`` are read as written. Otherwise it is
    ``derive_serving_config`` -- every batch up to the highest concurrency,
    ``capture_batch_sizes`` -- with a ``perf.configs`` entry's
    ``llm_api_options`` laid over it.
    """
    perf = task_data.get("perf") or {}
    if not perf.get("enabled"):
        return ["- tune: `perf` is off, so no serving config is measured"]
    if perf.get("configs"):
        entries = [(str(e.get("name")), e, [1]) for e in perf["configs"]]
    else:
        entries = [("serving", perf, [1, 128])]
    lines: list[str] = []
    for name, entry, default_concurrency in entries:
        identity = entry.get("identity_configs") or []
        for path in identity:
            if not Path(str(path)).is_file():
                lines.append(f"- tune `{name}`: {path} (not readable here)")
                continue
            options = gates.load_llm_api_options(path)
            capture = gates.options_capture(options, str(path))
            listed = capture.describe() if capture else "no CUDA graphs"
            lines.append(f"- tune `{name}`: {listed}, from {path}")
        if identity:
            continue
        workload = entry.get("workload") or {}
        concurrency = [int(c) for c in workload.get("concurrency", default_concurrency)]
        config = derive_serving_config(
            isl=int(workload.get("isl", 1024)),
            osl=int(workload.get("osl", 1024)),
            concurrency=concurrency,
        )
        overlay = entry.get("llm_api_options") or {}
        config.update(overlay)
        capture = gates.options_capture(config, f"perf.configs `{name}`")
        listed = capture.describe() if capture else "no CUDA graphs"
        source = (
            "its `llm_api_options`"
            if "cuda_graph_config" in overlay
            else f"`capture_batch_sizes` up to concurrency {max(concurrency)}"
        )
        lines.append(f"- tune `{name}`: {listed}, from {source}")
    return lines


__all__ = ["CAPTURE_LISTS_HEADING", "SYSTEM_PROMPT", "capture_lists_block"]
