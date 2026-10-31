# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker tags: the section each puts in a task's dispatch, and the line the planner picks it by.

A task's tags are its kind, then the ``tags`` the planner gave it
(:data:`~...run.graph.WORKER_TAGS`). Its dispatch carries
the section of each, in that order (:func:`tag_section`), so a task that does
two kinds of work is briefed on both. A tag says what one kind of work asks;
how the turn ends is the role's, and its system prompt's (:mod:`.worker`).
"""

from __future__ import annotations

from typing import Iterable

from ...operators.coverage import MAX_ROW_CELLS, TIER_DESCRIPTION
from .survey import CAPTURE_LISTS_HEADING

#: What the planner chooses a task's kind and tags by: one line per tag.
TAG_SUMMARIES: dict[str, str] = {
    "survey": "map the checkpoint's computation steps onto the catalog: `plan.md` and "
    "`vocabulary.yaml`",
    "kernel": "one catalog entry: onboard a missing one, verify an existing one at new values, "
    "or promote the target's local variants into the catalog",
    "assemble": "the target itself: modeling, weights, routing",
    "feature": "one feature, or one configuration variant, on a target whose core is built",
    "integrate": "land work that did not integrate on its own, resolving its conflicts",
    "investigate": "find the root cause of a failing check; writes a diagnosis, never the target",
    "tune": "make the target faster: as a kind, the performance campaign on the target as it "
    "is when the task starts; as a tag, one measured speedup",
}

#: Tags another tag brings with it: a feature is work on the target.
IMPLIED: dict[str, tuple[str, ...]] = {"feature": ("assemble",)}

_LADDER = "\n".join(f"{tier}. {text}" for tier, text in sorted(TIER_DESCRIPTION.items()))

_SURVEY = f"""\
## Tag `survey` — the checkpoint, mapped onto the catalog

Your task answers one question in two files: **can this checkpoint be
assembled from the catalog as it stands, and if not, what has to be added
first?** A re-draft addresses what the planner sent back; it does not
re-derive everything. Do not gold-plate, but never guess at what a file on
disk would tell you.

Read the checkpoint as data: its quantization method decides which entries
apply and which anchor row is right, and an entry's `CELLS`, `ARCHS` and
`note` decide what it covers. Read the HuggingFace modeling code for the
math; TensorRT-LLM's support for this architecture: what the runtime
allocates, and which attention backend serves this geometry (the classes on
`AttentionBackend`, sparse ones included; `tree_map.md` says where they are);
and every entry in the catalog and in each target's `ops/`. `surface.md`
lists them all; your dispatch shows each list's most relevant names first.
The config's traps: a multimodal checkpoint
nests the language model's geometry under `text_config`; a per-layer pattern
(`layer_types`, a hybrid layer pattern) is not a sliding window; a
speculative drafter may sit under `mtp.*` while its `dspark_*` keys say it is
DSpark. List every structural key you cannot place: each is a candidate for a
step the catalog has no word for, and this is the cheapest moment to find a
ceiling.

**The mapping.** For each computation step, record this checkpoint's value on
every axis the step's entry takes and the phase that runs it (prefill, decode
or both), then map it: `OK` (an entry's cells cover it), `CERT` (an entry
exists; a kernel task verifies the values) or `GAP` (no entry; a kernel task
onboards one, at the cheapest rung below that serves it). Judge coverage
here, by reading the entry: never queue a row because it is unclear. Walk the
parallel segment separately: its collectives appear nowhere in the HF
reference. Layout, movement, allocation and lookup are native torch and no
row; math composed from several torch calls is a gap, never a substitute for
an entry.

The sourcing ladder, cheapest first:

{_LADDER}

**Before writing a ceiling, look past `torch.ops.trtllm`**: FlashInfer's
public API as installed and the engine's pybind bindings serve steps no
trtllm op does. A vision tower's attention was once called a ceiling while
`thop_attention` served it all along. Prefer a wrong verify, which costs one
dispatch, to a wrong ceiling, which costs a feature.

**Before writing new code (tier 4 or 5), look at the prior art.** Your
dispatch's prior-art section lists what this tree already implements that
shares the checkpoint's words, and the limits the engine's kernels set for
this geometry. A GAP row that writes new code names in `tried` each lead that
shares its words and why it does not serve; `plan.md` states the design for
each engine limit. A survey review holds the draft to both and sends it back
once with what it left unconsidered. Look outside the tree too: how another
engine (vLLM, SGLang) implements this model is evidence for each step's rung,
and a library call it makes may serve a step no trtllm op does.

**The envelope.** `task.yaml`'s runtime block declares what the target serves
at every member: `max_num_tokens`, `max_batch_size`, `max_seq_len` and the
CUDA-graph capture batches. Judge every row against that box, not against the
gate workloads: cells that stop at the gates' shapes are `CERT`, not `OK`.
The capture batches are read, never recalled: copy them from your dispatch's
`{CAPTURE_LISTS_HEADING}` with the source each names, and put a list the
cells leave out under `## Open risks`.

**The anchor.** Name each unit's benchmark and its protocol (a reasoning
model scored as 5-shot completion lands near the floor whatever its ability),
from the reference row preflight resolved. A unit with no row is a spec
problem to report: an anchor is **never invented**.

**`plan.md`**, for the reader who executes it: the (checkpoint, arch,
parallel) identity and the routing criteria that distinguish this target
(config shape, SM, parallel topology; never a tuning knob), the core class
`<Family><Par>ModelingV2` and the causal-LM class `ModelingV2<Name>`, the
derived facts, the assembly units with `core` first, the gaps, the anchor,
the envelope, and `## Open risks`.

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

The validator reports every structural error at once. What it cannot check:

- `call_sites` names `PrefillTarget` or `DecodeTarget` and the rows per
  request at a captured step (1 for decode, `1 + draft` at a speculative
  verify, the drafter's width for a drafter call); `gathered: true` after an
  attention-DP all-gather, `eager_only: true` on a path never captured. A
  missing site is a row count nobody certifies.
- `unit` names the first unit that needs the entry; nothing reads a row's
  `note` for it. Untagged rows block `core`; when unsure, leave it untagged.
- `depends_on` is for real ordering only: a spurious one serializes work.
- A `verify` row states its `cells`: one mapping per cell, the entry's spec
  keys with values from the runtime facts. The kernel writer certifies each
  cell, so cover each axis of the envelope, not their product: every dtype
  each variant runs, each call site's rows per request at the smallest and
  the largest captured batch (and where the entry's code switches path),
  each member's per-rank pool sizes, one prefill chunk at `max_num_tokens`.
  Vary one axis at a time from a base cell; a row holds at most
  {MAX_ROW_CELLS} cells.
- One row per shared entry: every call site's shapes go in its `cells`, the
  sites in its `note`. `site` (lowercase words joined by `_`) names its local
  variant, else it is `r<n>`; a target name never contains `__`.
"""

_KERNEL = """\
## Tag `kernel` — one catalog entry

**One task, one entry**: onboard an op no entry wraps, verify an existing
entry at values its cells do not hold, or promote this target's local
variants into the catalog. Your dispatch says which, and where the files
land. **Load the `trtllm-modeling-v2-entry` skill before implementing, verifying or
promoting, even if verification changes no code.** It defines entry/test
certification and variant promotion. Native torch is not an entry (there is no
`catalog/torch/`): report it and write nothing. Nor is a torch fallback
inside the wrapper: it certifies nothing, so land the failure instead.

A kernel this run writes starts from the closest existing implementation
where there is one (FlashInfer, CUTLASS, vLLM, SGLang, another target's
`ops/`): adapt it, keep its license header and name its origin in the
entry's docstring.

When your dispatch states the cells, certify exactly those, plus one at the
envelope's largest row; a value you notice beyond them goes in the `note` as
uncertified. Certify measured values, not estimates: `runtime_facts.yaml`
holds what the engines of passed gates captured and allocated. Certify on a
run of the whole test file and the claims gates at your final commit.
Delivery runs the delivered tests again at the integrated HEAD, and holds
only if they pass there.

Beyond the entry, its test and the two files the skill allows, you write
nothing: anything broken outside the entry is a symptom to report, not to fix.
"""

_ASSEMBLE = """\
## Tag `assemble` — the target

A target, laid out as the README says, is a core that routes each step to a
`PrefillTarget` and a `DecodeTarget`, each a flat forward assembled from
entries. Read an existing target end to end before writing. The forward is
derived from `vocabulary.yaml` (the steps), the checkpoint's files as data,
the HF reference for the math, and each entry's `.py` (how to call it and
what is certified). An axis on which this checkpoint differs from the
exemplar is re-derived, or becomes a routing criterion if it changes the
forward's structure.

The invariants. The source-level gates check parts of 5 to 7; no gate checks
1 and 2:

1. **Closed vocabulary**: every TensorRT-LLM op the forward reaches goes
   through an entry; native torch is called directly, as glue.
2. **Routing decides identity**: the target carries no configuration
   asserts, and tuning knobs are never routing inputs.
3. **Weights are declared once**, in the weight table, and load coverage
   holds both ways: every parameter filled, every checkpoint key consumed or
   named in `expected_unconsumed` with why.
4. **Graph safety**: no `.item()`, `.cpu()` or data-dependent host branch in
   the forward; it survives CUDA-graph capture the first time.
5. **Every argument at the call site**, and `step_args` once at the top of
   `forward`: the engine surface the target reads is checked by running
   `step_args`, never by a hand-kept list.
6. **The core dispatches**: `PrefillTarget` and `DecodeTarget` run the step
   and share no body.
7. **Self-containment**: no code shared across targets.

A constant that bounds rows, tokens, batch or slots is derived from an engine
argument, never from what the gates happened to serve, and checked against
`runtime_facts.yaml`: what the engines of passed gates measured.

Run a gate yourself to debug, with the command your dispatch names. **A
shortfall is a debug signal, not a verdict**: back every re-gate with one
explicit causal hypothesis, and after three hypothesis-backed fixes without
improvement, stop and block with `accuracy_shortfall` and the trail. Turning
a runtime switch off (block reuse, say) to get a green gate hides a defect,
unless `task.yaml`'s runtime block turns it off. The anchor is read-only. A
call no entry covers is a `vocabulary_gap`: block as soon as you know, naming
it. Formatting is not your job: the operator lints the branch once, before it
becomes a PR.
"""

_FEATURE = """\
## Tag `feature` — one feature on the target

The target's core is built and its gates pass. Your task adds one feature
(your task's `ref`, one of `task.yaml`'s `features`), or makes the target
serve one configuration variant as well. **Every check the target passed
keeps passing**: with the feature off, the target runs as it did. Your task's
checks say what the feature must pass. A feature that needs an entry the
vocabulary lacks blocks as a `vocabulary_gap`, as the core would. **Load the
`trtllm-modeling-v2-features` skill first**: where the feature lives in the
engine, what the target must honour for it, and how to prove it from a CPU
check up to its gate.
"""

_INTEGRATE = """\
## Tag `integrate` — land work that did not integrate on its own

Your task lands changes that conflicted with the target as it moved: the
tasks and branches its goal names. Read each side's receipt and report first,
then resolve every conflict so that both intents stand. Prove the join on the
merged tree: the source-level gates, and the tests of every entry and target
file the merged changes touched. If the two cannot both stand, block, naming
the files and why.
"""

_INVESTIGATE = """\
## Tag `investigate` — why a check failed

Your task localizes a failure before anyone fixes it. As the task's kind, its
worktree is a scratch copy nobody integrates and you change nothing that
lands. Read the failing check's evidence first: its log, and the receipts of
the tasks it covered.

- Reproduce first where the tree already runs it: the in-tree test nearest
  the failure (its configuration, one test through the spool) and the stock
  path. A failure they share is the engine's or the site's, not the target's.
- Bisect with cheap probes before full runs: the same prompts under
  `modeling_v2="off"` and `"require"`; one module in isolation against the HF
  module or the built-in one, with the checkpoint's real weights, at the
  shapes the target runs; per-layer hidden-state diffs; prefill against
  decode; CUDA graphs on against off.
- Back every experiment with one explicit causal hypothesis, written down
  first: a rerun without one buys nothing.
- Never widen a tolerance; the anchor and its tolerance are read-only.

Your product is the diagnosis: what fails (the op, the layer, the phase), the
evidence that shows it, and the change you propose, with the task that should
make it. When no probe could localize the failure, block, with each
hypothesis and its result.
"""

_TUNE = """\
## Tag `tune` — making the target faster

You make **one** thing faster, at the values the target actually passes: one
entry, or one run of consecutive entries fused into one. Your task or request
names it.

1. **Measure first**, on this machine, at the cells the target runs: the
   decode batches CUDA graphs capture and the prefill chunk. One benchmark
   session per revision, with warmup and timing iterations inside it. Report
   the median and within-session spread; do not repeat full benchmark jobs.
2. **Take one approach**: another implementation of the op (an engine op, an
   in-tree callable, or FlashInfer as installed: a kernel only a newer
   FlashInfer has is not available), a new kernel (Triton; CuTe DSL or CUDA
   only where Triton cannot reach the floor, saying why), or a fusion: one new
   entry whose single call computes what several consecutive entries did. A
   new entry follows the `kernel` rules in full. Shared modules and kernels
   are fair game; the stock model's forward is not.
3. **Prove it correct and faster**: the entry's test passes within its band
   against `reference`, the source-level gates pass, and the orchestrator's
   identical micro/module harness verifies the gain on both revisions. A
   claimed speedup is not evidence. Never claim an end-to-end number from a
   local benchmark: e2e is reserved for serving/system-wide interactions, the
   campaign's checkpoints and the final tree. The accuracy benchmark runs once,
   on the final tree.
4. **Keep it only if it won.** A change that did not win lands nothing and is
   reported with its numbers: a measured "no" is a result.

Matching the baseline bit for bit is not a goal: a faster design may
accumulate in another order and differ in the last bits, as long as the
difference is the error its arithmetic explains. Never trade accuracy for
speed (no wider band, no lower precision, no dropped cell), and name every
change of numerics you made.
"""

#: Each tag's section of the dispatch.
TAG_PROMPTS: dict[str, str] = {
    "survey": _SURVEY,
    "kernel": _KERNEL,
    "assemble": _ASSEMBLE,
    "feature": _FEATURE,
    "integrate": _INTEGRATE,
    "investigate": _INVESTIGATE,
    "tune": _TUNE,
}


def task_tags(kind: str, tags: Iterable[str] = ()) -> tuple[str, ...]:
    """A task's tags in briefing order: each tag after those it brings with it, none twice."""
    ordered: list[str] = []

    def add(tag: str) -> None:
        for implied in IMPLIED.get(tag, ()):
            add(implied)
        if tag in TAG_PROMPTS and tag not in ordered:
            ordered.append(tag)

    for tag in (kind, *tags):
        add(tag)
    return tuple(ordered)


def tag_section(tags: Iterable[str]) -> str:
    """The dispatch's sections for ``tags``, in order; ``""`` when none is a worker tag."""
    return "\n".join(TAG_PROMPTS[tag] for tag in tags if tag in TAG_PROMPTS)


__all__ = [
    "IMPLIED",
    "TAG_PROMPTS",
    "TAG_SUMMARIES",
    "tag_section",
    "task_tags",
]
