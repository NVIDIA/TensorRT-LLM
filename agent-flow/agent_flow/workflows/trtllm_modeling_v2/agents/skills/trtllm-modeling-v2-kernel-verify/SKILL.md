---
name: trtllm-modeling-v2-kernel-verify
description: Load when smith must check an existing catalog entry against the values a checkpoint will pass — confirm the entry's ARCHS names this architecture and its CELLS cover the values, and where they do not, add the cells in a local variant, drive them through its test, and certify only from the observed run. Also covers the promote dispatch, which merges a target's local variants into the shared catalog.
---

# trtllm-modeling-v2-kernel-verify

One run, one entry, one surface. The entry already exists; the question is
whether what this checkpoint passes is inside what its cells certify.

The spec is shared: read
`.claude/skills/trtllm-modeling-v2-kernel-onboard/references/entry-spec.md` (where an
entry lives, the `OpWrapper` members, certification, the test recipe) and
`.claude/skills/trtllm-modeling-v2-kernel-onboard/references/contract-evidence.md`
(what makes a sentence in an entry defensible). This skill only says what
differs.

The entry is the `OpWrapper` subclass in its `.py` — in the shared catalog
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py`)
or in a target's own `ops/<category>/<name>.py` — and its test is at the
matching path in the tests tree (see the entry spec's table). There is no
prose contract beside it, no index row and no receipt frontmatter to read or
write.

**A shared entry is verified in a local variant.** Your dispatch prompt's
"Where it lands" names it: `ops/<category>/<name>__<site>.py` in this target's
own `ops/`, and its test beside the target's other op tests. Start it as a
copy of the entry as it stands, file and test, keeping the class and the
module-level instance name, and do everything below there; the shared files
stay as they are, because other tasks on this entry are verifying it beside
you, each in a variant of its own. The target imports the variant. Before
delivery one `promote` dispatch merges every variant into the catalog
(`references/promote.md`). Wherever this skill says "the entry", read "the
variant".

## Outcomes

**Covered.** Some cell in `CELLS` already includes these values — read each
cell's `why` and `spec`, and the `note` for what is stated as outside — and
`ARCHS` names the architecture this run targets. Change nothing. Report which
cells cover the surface.

That check is not a formality. An arch not in `ARCHS` is **unknown**, not
"probably fine", and a value between two cells is covered only if the op's
behaviour is provably uniform across them (a cell per bucket edge, per tile
boundary, per code path — the `why`s say which). Architecture is the whole
key — there is no version pin in tree, because a target moves with the trunk,
and the entry's test runs in pre-merge, so a merged entry's cells are re-proved
on every commit. All of that costs a file read and no GPU.

**A defect outside the entry.** When a value fails because of shared code
the entry wraps (an upstream kernel, a library, the engine) and not the
entry's wrapper or its test, stop as soon as you have measured it: commit the
guard and the failing test, and end `FAILED_RECEIPT` with the `engine_defect`
block in your decision file (`affects`, `trigger`, `detect`, `proposal`; your
prompt shows its shape). The orchestrator adds the task that fixes it; the
remaining cells go to the re-certification that follows the fix.

**Not covered.** Extend it:

1. **Probe first where the semantics are undocumented**, then freeze what the
   probes pinned into cells. Do not write the assertion before you know what
   the op does at this value
2. **Add the cells to the entry's `CELLS`**, one `Cell(why=..., spec=...)` per
   distinct thing covered, and extend the test's `_build` only as far as the
   new `spec` keys need. The test stays parametrized off `op.CELLS`, arms
   `certifying`, and gates `__call__` against the entry's own `reference`
   within the band the test states — never a local reference, never another
   entry. Randomness is seeded; any band wider than torch's dtype-aware
   default is the test's own, with its derivation
3. **Move what the `note` said was uncertified for this surface** into cells,
   from observed runs only, and rewrite the `note` to match. This is the part
   that gets skipped and it is the part the next target reads
4. **Iterate on the cells you add or change** (`pytest -k`), not the whole
   file: each whole-file run costs what every cell in it costs
5. **Run the entry's whole test file and the claims gates once, at the end**,
   with every edit made (this arch in `ARCHS` if it is new), and certify from
   that run. A multi-rank entry's world size is a cell axis
6. Commit the entry and its test (the variant's, for a shared entry), and
   nothing else, through the commit helper your prompt names

A verify touches nothing but the entry and its test: the catalog is its entry
files, with no index to keep in step. If a shared entry's summary has to
change, it is the class docstring.

`__call__`'s signature usually stays untouched. If it does not — if covering this surface
needs a signature change — that is worth saying loudly in the report, because
every existing caller is affected and the promote has to reconcile it with
every other variant. An entry that sits in another target's `ops/` is verified
the same way, in a local variant started from that target's copy; the promote
moves it into the shared catalog.

**The op may simply fail at this value.** That is a valid product: the value
is not a cell; if the op accepts it and answers wrongly, `is_valid` refuses it
and a guard test proves the refusal; the `note` records what was measured.
Commit it, the entry and its test and nothing else, and report
`FAILED_RECEIPT`: a failed receipt whose commits change only the entry's own
files is integrated as it stands. It is a real constraint on every target
that would have passed that value, and finding it here is far cheaper than
finding it in an accuracy gate. A fix outside the entry (the engine, the
target) is not yours to make: name it in the report, and the planner gives
it a task; a commit of yours outside the entry keeps the whole branch out of
the target until the task's next attempt.

## Test cost

The entry's test file runs in pre-merge CI on every commit. A new cell fits
**10 s and 40 GiB** of device memory; a cell that must reach a bound (a pool
at the `kv_cache_fraction` bound, a 2^31-element probe) is marked `large` and
run once in your turn; cells of one call that one process can drive are one
parametrized test. The entry spec's "Test cost" has the marker and how to
apply it. Report the file's wall time from the certifying run.

**Pre-merge does not run `large` cells:** pre-merge runs the suite with `-m
"not large"`, so a `large` cell is proved in your turn and in the stage that
runs the rest, not on every commit. The marker must be registered in
`tests/unittest/pytest.ini` (under `markers`). That file is outside your
entry: when `large` is not listed there, say so in your report as a symptom,
and the planner gives an engine task the one line; never edit it yourself.

## Promote

A `promote` dispatch merges a target's local variants of one or more shared
entries into the catalog, points the target at the catalog and certifies the
merged entries once. Its procedure is `references/promote.md`; read it before
writing anything.

## The dispatch sets the scope

When your dispatch carries a `cells` block, or "The surface to certify", that
is the whole of this turn: certify exactly those cells, plus one cell at the
envelope's largest row. The workflow chose them from the target's calls and
the measured geometry (`runtime_facts.yaml`); a value you notice beyond them
goes in the `note` as uncertified, for the dispatch that needs it. Without a
stated scope, the section below is the scope.

## The surface is the envelope, at every call site's row multiple

The serving envelope is the runtime block of the run's `task.yaml`, and
`vocabulary.yaml` gives, for each place the target calls this entry, the rows
per request it passes (`G`, `3·G`, `6·G`, ...; times the gather width after an
attention-DP all-gather; `eager_only` for a site that is never captured).
Certify what those reach, not what one gate happened to launch:

- the captured row counts it lists by bucket boundary (each bucket's smallest
  and largest, where the kernel picks a path by size), replay on new data
  bitwise equal to eager;
- every eager row count up to the envelope's token budget, by the envelope
  standard (a run at the maximum plus prefix-bitwise for grid-sized rows;
  every bucket's edges for a kernel that picks by size);
- pools at the largest size the GPU allocates at the envelope's settings, with
  rows addressed at the far end.

The cells you write say the envelope and the call-site multiples they cover —
in their `why`s, with the rows in their `spec`s — so a reader reads the
cell list instead of re-deriving rows. Check the call sites against the call
chain and say in your report which you confirmed.

## Integer-width boundaries

For each **index-typed** argument — offsets, indices, positions, page and
slot tables, row counts — compute where it crosses an integer-width boundary:

- **2^31 elements**: an int32 element count or offset wraps;
- **2^31 / row width rows**, for every row width the entry addresses: a row
  index multiplied by the width in 32 bits wraps there (2^31 / 512 = 2^22
  rows), and the kernel writes out of place without raising;
- **2^24 + 1** for any index that passes through fp32: above 2^24 an fp32
  integer is no longer exact.

**When the envelope reaches a boundary, probing it is required**, and the
outcome goes in the entry as a **hard limit**: an `is_valid` guard where the
op crosses it silently, a `Hard limits` row in the `note` otherwise. When the
envelope does not reach it, write the computed boundary in the `note` without
probing. Either way the limit is written down, so a change that lifts a
memory cap and enlarges the envelope can re-read it: one kernel computed its
output offset in 32 bits and silently wrote rows at and above 2^25 out of
place after a prefill-tiling fix moved the reachable rows past it. Probing
without the full allocation, and the table format, are in
`references/integer-width-probes.md`.

## When the surface neighbours a known intermittent defect

`entry-spec.md` states the rule and the case that produced it: prove the
harness can *see* the defect before reporting a clean result. It applies with
full force here — extending coverage next to a known-flaky surface is exactly
where a blind harness returns a confident, meaningless green.

## Report

- entry path, the surface requested, and the verdict: already covered / newly
  certified / failed
- `ARCHS` and the cells as they now read, and what the new cells pinned
- the sourcing tier the entry's `__call__` wraps, also the decision file's
  `tier`, and the test file's wall time in the certifying run
- the hard limits the integer-width probes established
- **what remains uncertified** on this entry — the next caller reads this
- every claim matched against disk before sending
