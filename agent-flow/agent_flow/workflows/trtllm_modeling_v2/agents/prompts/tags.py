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

from ...onboard.vocabulary import TIER_DESCRIPTION
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

Read the checkpoint as data (`config.json`, the safetensors index, the
quantization config, the tokenizer and chat template: the quantization method
decides which entries apply and which anchor row is right), the HuggingFace
modeling code for the math, TensorRT-LLM's support for this architecture
(what the runtime allocates, which attention backend serves this geometry),
and every entry in the catalog and in each target's `ops/`: an entry's
`CELLS`, `ARCHS` and `note` decide what it covers.

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
`thop_attention` served it all along. Prefer a wrong verify to a wrong
ceiling.

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
- `unit` names the first unit that needs the entry. Untagged rows block
  `core`; when unsure, leave it untagged.
- `depends_on` is for real ordering only: a spurious one serializes work.
- A `verify` row states its `cells`: one mapping per cell, the entry's spec
  keys with values from the runtime facts, spanning the envelope: every dtype
  each variant runs, each captured batch times each call site's rows per
  request, each member's per-rank pool sizes, the prefill chunk up to
  `max_num_tokens`.
- One row per shared entry: every call site's shapes go in its `cells`, the
  sites in its `note`. `site` (lowercase words joined by `_`) names its local
  variant, else it is `r<n>`; a target name never contains `__`.

| Impulse | Reality |
|---|---|
| "No op in `torch.ops.trtllm`, so it's a ceiling" | FlashInfer's API and the pybind bindings are outside it. A wrong ceiling costs a feature; a wrong verify costs one dispatch. |
| "The row's `note` says only the speculator needs it" | Nothing reads notes. Put it in `unit:`. |
| "The cells cover what the gates run, so the row is OK" | The certified set is the serving envelope: the values above the gate tuples are the ones a deployment reaches. |
| "The graphs pad to 19 batches" | Copy the printed list and its source; cells built on the wrong one leave the batches users run uncertified. |
"""

_KERNEL = """\
## Tag `kernel` — one catalog entry

**One task, one entry**: onboard an op no entry wraps, verify an existing
entry at values its cells do not hold, or promote this target's local
variants into the catalog. Your dispatch says which, and where the files
land. **Load the `trtllm-modeling-v2-entry` skill before writing anything**:
it is the procedure for an entry and its test, local variants and the
promote. Read `catalog/_op.py` and two or three entries of the same category
with their tests first; yours follows their shape.

An entry is one `OpWrapper` subclass per file, and the class is the contract:
`__call__` (one kernel-launching call, every argument explicit, nothing read
off `self`), `reference` (plain torch, fp32-accumulated, never the op under
test or another entry), `is_valid` (refuses what the op accepts and answers
wrongly), `CELLS`, `ARCHS` and `note`. Fusion is legal only inside that one
call. Native torch is not an entry (there is no `catalog/torch/`): report it
and write nothing.

Certification has three states:

- **certified** — inside a cell, on an arch in `ARCHS`, because you watched
  the entry's whole test file pass here, in a run that post-dates the last
  write to every file of the entry, in both trees;
- **failed** — the op demonstrably fails at a value: land the entry anyway,
  the value outside `CELLS`, refused by `is_valid` where the op answers
  silently wrong, the measurement in the `note`;
- **unknown** — a value outside every cell, or an arch not in `ARCHS`.

When your dispatch states the cells, certify exactly those, plus one at the
envelope's largest row; a value you notice beyond them goes in the `note` as
uncertified. Certify measured values, not estimates: `runtime_facts.yaml`
holds what the engines of passed gates captured and allocated. Iterate on the
cells you change (`pytest -k`); run the whole test file and the claims gates
once, at the end, through the command `trtllm-modeling-v2 tests affected
--workspace <the workspace> --head <your commit> --paths <the test file>`
prints (the CLI is in the workspace's `bin/`). That run is recorded with your
commit: delivery holds only on a recorded pass.

You write the entry and its test and nothing else, besides an `__init__.py`
for a new `ops/<category>/` and, when the device has no `Arch` member yet,
that one line in `catalog/_op.py` in a commit of its own. Anything broken
outside the entry is a symptom to report, not to fix.

| Impulse | Reality |
|---|---|
| "The engine call fails; I'll fall back to torch inside the wrapper" | A torch fallback certifies nothing. Land the failure instead. |
| "It passes with rtol=0.5" | Tolerance inflation manufactures green. Derive the band in the test, or land the failure. |
| "It obviously works on this arch too" | `ARCHS` comes from runs, not plausibility. |
| "The test passed, so `ARCHS` can name this arch" | Only if that run post-dates every file of the entry, in both trees. |
"""

_ASSEMBLE = """\
## Tag `assemble` — the target

A target is `models/<family>/<ckpt>__<arch>__<parallel>/`: `modeling.py` and
`weights.py`, routed by `models/<family>/routing.py`; a core that routes each
step to a `PrefillTarget` and a `DecodeTarget`, each a flat forward assembled
from entries. Read an existing target end to end before writing. The forward
is derived from `vocabulary.yaml` (the steps), the checkpoint's files as
data, the HF reference for the math, and each entry's `.py` (how to call it
and what is certified). An axis on which this checkpoint differs from the
exemplar is re-derived, or becomes a routing criterion if it changes the
forward's structure.

The invariants, which the source-level gates check:

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
improvement, stop and block with `accuracy_shortfall` and the trail. The
anchor is read-only. A call no entry covers is a `vocabulary_gap`: block as
soon as you know, naming it. Formatting is not your job: the operator lints
the branch once, before it becomes a PR.

| Impulse | Reality |
|---|---|
| "HF's forward does it op by op; transliterating that is safest" | HF defines the math, the catalog the vocabulary. The target is the fused form of HF's semantics. |
| "The exemplar computes X, this config says Y: add an assert so it refuses the mismatch" | Targets carry no configuration asserts. Re-derive X, or make it a routing criterion. |
| "Prefill and decode do the same thing here; one shared base saves 200 lines" | The two targets share no body. Duplicate it. |
| "It would pass with block reuse disabled" | Turning a switch off to get a green gate hides a defect, unless `task.yaml`'s runtime block turns it off. |
| "8192 rows is plenty for this buffer" | Plenty for the gates is not the envelope. Size it from `max_num_tokens`. |
"""

_FEATURE = """\
## Tag `feature` — one feature on the target

The target's core is built and its gates pass. Your task adds one feature
(your task's `ref`, one of `task.yaml`'s `features`), or makes the target
serve one configuration variant as well. **Every check the target passed
keeps passing**: with the feature off, the target runs as it did. Learn how
the engine supports the feature from its code (the LLM API argument that
turns it on, and what the runtime then hands the model) and wire the target
to that interface. Your task's checks say what the feature must pass. A
feature that needs an entry the vocabulary lacks blocks as a
`vocabulary_gap`, as the core would.
"""

_INTEGRATE = """\
## Tag `integrate` — land work that did not integrate on its own

Your task lands changes that conflicted with the target as it moved: the
tasks and branches its goal names. Read each side's receipt and report first,
then resolve every conflict so that both intents stand; never take one side
wholesale. Leave no conflict marker behind (`git diff --check`). Prove the
join on the merged tree: the source-level gates, and the tests of every entry
and target file the merged changes touched. If the two cannot both stand,
block, naming the files and why.
"""

_INVESTIGATE = """\
## Tag `investigate` — why a check failed

Your task localizes a failure before anyone fixes it. As the task's kind, its
worktree is a scratch copy nobody integrates and you change nothing that
lands. Read the failing check's evidence first: its log, and the receipts of
the tasks it covered.

- Bisect with cheap probes before full runs: the same prompts under
  `modeling_v2="off"` and `"require"`; one module in isolation against the HF
  module or the built-in one, with the checkpoint's real weights, at the
  shapes the target runs; per-layer hidden-state diffs; prefill against
  decode; CUDA graphs on against off.
- Back every experiment with one explicit causal hypothesis, written down
  first. Same-protocol accuracy reruns move 0.1-0.3 points: a rerun without a
  hypothesis buys nothing.
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
   decode batches CUDA graphs capture and the prefill chunk. Warm up,
   synchronize, and report the median and the run-to-run spread. Record the
   baseline before you change anything.
2. **Take one approach**: another implementation of the op (an engine op, an
   in-tree callable, or FlashInfer as installed: a kernel only a newer
   FlashInfer has is not available), a new kernel (Triton; CuTe DSL or CUDA
   only where Triton cannot reach the floor, saying why), or a fusion: one new
   entry whose single call computes what several consecutive entries did. A
   new entry follows the `kernel` rules in full.
3. **Prove it correct and faster**: the entry's test passes within its band
   against `reference`, the source-level gates pass, and the micro-benchmark
   win is larger than the spread you measured. You never claim an end-to-end
   number: the orchestrator measures that, and runs the accuracy gate on
   every change it keeps.
4. **Keep it only if it won.** A change that did not win lands nothing and is
   reported with its numbers: a measured "no" is a result.

Matching the baseline bit for bit is not a goal: a faster design may
accumulate in another order and differ in the last bits, as long as the
difference is the error its arithmetic explains. Never trade accuracy for
speed (no wider band, no lower precision, no dropped cell), and name every
change of numerics you made.

| Impulse | Reality |
|---|---|
| "This should be ~2x faster" | Measure it. Micro-benchmark numbers in the decision; end-to-end from the orchestrator. |
| "Two fusions, one candidate: saves a sweep" | One hypothesis per change, or the attribution is garbage. |
| "The stock model is fast here; let me see how it does it" | Its model-specific files are embargoed. Shared modules and kernels are fair game; its forward is not. |
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
