---
name: staircase-kernel-feasibility
description: Load when deciding whether a checkpoint can be assembled from the current kernel catalog, and what has to be added first — read the checkpoint and its HF reference for the computation-step list, map each step onto the catalog, and produce vocabulary.yaml with the dispatch queue. Also answers it standalone, before committing to an onboarding.
---

# staircase-kernel-feasibility

Answers one question: **can this checkpoint be assembled from the catalog as
it stands, and if not, what has to be added first?**

It runs inside the run's **plan phase** — the PlanDrafter does this judging
itself, and nobody is dispatched to do it — but it is worth running on its
own: before committing hours to an onboarding, it tells you whether the model
is reachable at all. Its product is a file, not a conclusion in someone's
context: the mapping is referenced by every dispatch that follows and it is
the recovery state after a session cut.

You do this yourself. You do not dispatch to answer it — see "Why the judging
stays here".

## Flow

### 1. Semantic pre-audit

Read the checkpoint's `config.json`, its safetensors index (key names, shapes,
dtypes), and the architecture's HF reference modeling code — for
**computational semantics only**, what the model computes.

Walk `references/derivation-checklist.md` and record this checkpoint's value
on every axis. The output is a **computation-step list**: each step of the
forward with the values that step will be called at.

**The parallel axis is not in the checkpoint.** Collectives are an artifact of
the topology, so they appear nowhere in the HF reference; a pre-audit driven
only by that reading comes back all-green for a `tepN`/`depN`/`unified` target
and the assembler discovers the gap mid-run. Walk the segment separately: what
the topology partitions, what must be reduced or gathered to put it back
together, and whether entries for that exist. `references/trtllm-ops.md`'s `comm`
category is where the candidates are.

### 2. Vocabulary mapping

Map every step onto
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml` and the
entries' contracts. Each row lands in one of three states — the judging rules
are in `references/mapping-and-triage.md`:

| state | meaning |
|---|---|
| **OK** | an entry exists and its certified column covers this checkpoint's values |
| **CERT** | an entry exists; this surface is not covered |
| **GAP** | no entry |

### 3. Ceiling check

For every GAP row, look for a candidate in `references/trtllm-ops.md`. **No
candidate is the vocabulary ceiling** —

**except that the inventory cannot see pybind bindings, and one of the most
capable entries in the catalog is one.** `references/trtllm-ops.md` is generated
from the torch schema registry, so it lists `torch.ops.*` only.
`attention/thop_attention` wraps `tensorrt_llm.bindings.internal.thop.attention`,
a pybind binding carried as an approved exception — it is not in the inventory
and never will be. A row whose only server is a binding therefore comes back
empty from an inventory search: **correctly empty, and wrongly read as a
ceiling.** This has cost a feature once: a vision tower's cache-free attention
was recorded as a ceiling and the feature descoped, when `thop_attention` served
it all along in a no-KV-cache mode (all four pool arguments `None`) that
trtllm's own diffusion stack runs.

So before writing "ceiling", check the **catalog** as well as the inventory, and
ask whether an existing entry's *uncertified* surface already covers the step —
a `VERIFY` row is not a gap. Prefer a wrong "verify" to a wrong "ceiling": the
first costs a dispatch, the second costs a feature — this checkpoint cannot be assembled
today, and you can say so here, before dispatching anyone. Record the step,
what it computes, and why nothing upstream provides it.

The mirror surface is glue — layout, movement, allocation, lookup — and a
missing **computation** kernel (norm, rope, activation, linear, attention,
quantization) is a ceiling rather than something to build there. **One narrow
exception**, and it is narrow on purpose: a computation may become a mirror when
all three hold —

1. it is **exactly one torch callable**, not a computation assembled from
   several mirrors;
2. **no trtllm op serves it**, established by a search closed over the inventory
   *and* the pybind surface the inventory cannot show;
3. upstream owns its correctness, which is what lets a mirror be contract-free.

Composing a computation out of several mirrors is still a ceiling, and reaching
for a mirror while a trtllm op exists is still wrong — the catalog prefers the
engine's kernels and a mirror is the last resort, not the convenient one.
Precedent: `catalog/torch/gelu_tanh.py`, added after the candidate set for
non-gated `gelu_tanh` at bf16 was searched closed.
<!-- PORT: verify — `catalog/torch/gelu_tanh.py` is not in the in-tree catalog
     (`torch/` holds add, concat, copy_, embedding, empty, expand, pad,
     reshape, split, transpose, view_dtype). The precedent is kept because the
     rule it illustrates stands on its own; confirm whether the entry was
     dropped in the move or is still to be ported. -->

### 4. Write `vocabulary.yaml`

`<workspace>/vocabulary.yaml` — the run's typed mapping. Out of tree this was
prose, because the only thing reading it was another language model; here the
orchestrator reads it, builds its dispatch queue from it, and **rejects a
malformed one**. The schema is authoritative and lives at
`agent_flow/workflows/staircase_onboard/vocabulary.py`; what follows is what
it validates.

```yaml
verdict: assemblable | assemblable_after_additions | ceiling

steps:                                  # every computation step, both chapters
  - step: <what this step computes>
    values: {<axis>: <this checkpoint's value>, ...}
    state: OK | CERT | GAP
    entry: catalog/<category>/<name>.py                     # OK and CERT rows
    contract_line: "<the contract sentence that decides coverage>"
    candidate: torch.ops.trtllm.<name>                      # GAP rows

queue:                                  # one dispatch per entry, in dependency order
  - {n: 1, mode: onboard, target: <entry name>, depends_on: []}
  - {n: 2, mode: verify,  target: <entry name>, depends_on: [1],
     surface: "<the value the certified column excludes>"}
  - {n: 3, mode: onboard, target: <entry name>, depends_on: [],
     unit: <feature>}                   # optional; core when absent

ceiling: null                           # or {step: ..., searched: [...], why: ...}
```

Chapter 1 is the `OK`/`CERT` rows (entries this checkpoint will use); chapter 2
is the `GAP` rows (callables with no entry). The `queue` is the two chapters as
dispatches — `CERT` → `mode: verify` → `staircase-kernel-verify`, `GAP` →
`mode: onboard` → `staircase-kernel-onboard`. An `OK` row dispatches nobody.

Every rule below is enforced in Python, and every violation is reported in one
error rather than one per round-trip:

| rule | why |
|---|---|
| `verdict` is exactly one of `assemblable` / `assemblable_after_additions` / `ceiling` | it is the headline, and the orchestrator branches on it |
| every `steps` row names a non-empty `step` and a `state` of `OK` / `CERT` / `GAP` | a row with no state is not a mapping |
| an `OK` or `CERT` row names an `entry` **and** quotes a `contract_line` | the quoted line is the evidence for the judgement; without it nobody can audit the row without redoing the reading |
| a `GAP` row names a `candidate` | a GAP with no candidate anywhere *is* the ceiling — say so in `ceiling` rather than leaving the row bare |
| `queue[].n` is an integer and unique across the queue | it is the dispatch's identity and what `depends_on` points at |
| `queue[].mode` is `onboard` or `verify` | it selects the skill smith loads |
| `queue[].target` names exactly one catalog entry | one dispatch, one entry |
| **no `target` appears under both modes** | the two chapters do not overlap (below) |
| `depends_on` is a list of dispatch numbers, all present in the queue | an unknown dependency cannot be ordered, and a cycle is rejected when the queue is layered into concurrent batches |
| a `verify` dispatch names a `surface` | it becomes smith's prompt almost verbatim, so vagueness here costs a whole run |
| `queue[].unit`, when present, is one of the run's assembly units — `core` or a feature from `task.yaml` | it decides what a failed dispatch blocks (below); a unit the run does not build could only be guessed at |
| `verdict: ceiling` carries a `ceiling` mapping naming the exact `step` and what was `searched` | a ceiling is the one verdict that ends the run, so it carries its evidence or it is not a verdict |
| any other verdict leaves `ceiling` unset | the two must agree |
| `assemblable` means an empty queue; `assemblable_after_additions` means a non-empty one | "nothing has to be added first" plus three dispatches is not a verdict |

The two chapters do not overlap, and an entry onboarded in chapter 2 never
appears in chapter 1: smith writes its test at this checkpoint's values by
construction. The schema enforces it, and the error names both modes it saw.

Order the queue by dependency. The category (verify or onboard) says nothing
about order — a chapter-1 row whose test state must be built by a chapter-2
entry simply sits after it. Everything with no dependency runs **concurrently**,
so `depends_on` is for real ordering only: a spurious edge serializes a batch
and costs wall-clock directly.

Tag every row that only a feature increment needs with `unit:` — the first
unit that uses the entry. Queue those rows anyway: every row is dispatched in
the smith phase, concurrently with core's, so a feature's entries cost no extra
wall-clock. The tag changes only what a **failure** blocks. An untagged row
reads as `core`, and a `core` row that fails stops the run before assembly. A
tagged row that fails is deferred instead, and retried once before its unit's
first assembler turn; the run stops only if that retry fails too, and then at
that unit rather than before core. Measured on one run: a 33-dispatch queue
with six feature-only rows — host-offload gathers, a vision-tower verify and
speculator-only surfaces among them — any one of which could close the run
before core started.

- Tag the **first** unit that needs the entry, in `task.yaml` feature order.
- Leave the tag off when core uses the entry at all, or when you are unsure.
  Untagged is the blocking direction, so it is the safe one.
- A row that a `core` row `depends_on` is core's, transitively, whatever its
  own tag says: the orchestrator will not start core without it.
- A `note` saying "dspark only" is not a tag. Nothing reads notes.

## Why the judging stays here

Reading a contract's certified column is what contracts are *for*. The entry
spec puts it plainly: "Whoever calls this op reads the contract and nothing
else — no trtllm source", and "Nothing outside this entry reads trtllm source.
The contract must stand alone." A contract only smith can interpret would be a
failed contract.
<!-- PORT: verify — the first sentence is verbatim in
     `.claude/skills/staircase-kernel-onboard/references/entry-spec.md`; the
     second was attributed to staircase's `smith.md` agent definition and is
     not verbatim in the ported prompt
     (`agent_flow/workflows/staircase_onboard/prompts/smith.py`), which carries
     the same doctrine as a red-flag row. Re-attribute or re-quote. -->

What genuinely needs smith is a row with **no entry**: only reading source
tells you what that callable is, whether it can be one entry, and what it
needs beforehand.

So: **judge here, dispatch to act.** Never dispatch because a row is unclear —
read the contract again.

## Report

Whether run inside an onboarding or standalone, say:

- the verdict: assemblable / assemblable after N additions / ceiling
- the counts by state, and the queue length
- for a ceiling: the exact step, and what was searched
- for CERT rows: the value and the contract line that excludes it — this is
  what smith is dispatched with, so vagueness here costs a whole run
