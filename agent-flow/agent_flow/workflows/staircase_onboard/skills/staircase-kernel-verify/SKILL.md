---
name: staircase-kernel-verify
description: Load when smith must check an existing catalog entry against the values a checkpoint will pass — confirm the entry's ARCHS names this architecture and its CELLS cover the values, and where they do not, add the cells, drive them through the entry's test, and certify only from the observed run.
---

# staircase-kernel-verify

One run, one entry, one surface. The entry already exists; the question is
whether what this checkpoint passes is inside what its cells certify.

The spec is shared: read
`.claude/skills/staircase-kernel-onboard/references/entry-spec.md` (where an
entry lives, the `OpWrapper` members, certification, the test recipe) and
`.claude/skills/staircase-kernel-onboard/references/contract-evidence.md`
(what makes a sentence in an entry defensible). This skill only says what
differs.

The entry is the `OpWrapper` subclass in its `.py` — in the shared catalog
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py`)
or in a target's own `ops/<category>/<name>.py` — and its test is at the
matching path in the tests tree (see the entry spec's table). There is no
prose contract beside it, no index row and no receipt frontmatter to read or
write.

## Two outcomes

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
4. **Rerun the entry's whole test file**, not just the new cells, and certify
   from that run: add this arch to `ARCHS` if it is new. A multi-rank entry's
   world size is a cell axis
5. Do not format or lint during iteration — the repo's `pre-commit` runs once
   at delivery. Revalidate that the run post-dates the last write to every
   file of the entry, in both trees, then commit pathspec-scoped — the entry
   and its test, e.g. for a shared entry:

```bash
git add tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py \
        tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py
git commit -m "feat(catalog): certify <name> at <surface>"
```

A verify touches nothing but the entry and its test: the catalog is its entry
files, with no index to keep in step. If a shared entry's summary has to
change, it is the class docstring.

`__call__`'s signature usually stays untouched. If it does not — if covering this surface
needs a signature change — that is worth saying loudly in the report, because
every existing caller is affected; and if the entry sits in another target's
`ops/`, a verify for this run is a **promotion** (entry and test move to the
shared catalog in one commit) — report it and let the orchestrator dispatch it.

**The op may simply fail at this value.** That is a valid product: the value
is not a cell; if the op accepts it and answers wrongly, `is_valid` refuses it
and a guard test proves the refusal; the `note` records what was measured.
Land it and report `FAILED_RECEIPT`. It is a real constraint on every target
that would have passed that value, and finding it here is far cheaper than
finding it in an accuracy gate.

## The surface is the envelope, at every call site's row multiple

Your dispatch prompt carries a **smith brief**: the run's serving envelope and,
for each place the target calls this entry, the rows per request it passes
(`G`, `3·G`, `6·G`, ...; times the gather width after an attention-DP
all-gather; `eager-only` for a site that is never captured). Certify what the
brief lists, not what one dump showed:

- every captured row count it lists, replay on new data bitwise equal to eager;
- every eager row count up to the envelope's token budget, by the envelope
  standard (a run at the maximum plus prefix-bitwise for grid-sized rows;
  every bucket's edges for a kernel that picks by size);
- pools at the largest size the GPU allocates at the envelope's settings, with
  rows addressed at the far end.

The cells you write say the envelope and the call-site multiples they cover —
in their `why`s, with the rows in their `spec`s — so the auditor reads the
cell list instead of re-deriving rows. A
brief that says its call sites were ASSUMED is a prompt to check them against
the call chain and say in your report which you found.

## Integer-width boundary probes (mandatory)

Every verify probes each **index-typed** argument -- offsets, indices,
positions, page and slot tables, row counts -- at the integer-width
boundaries, and records the outcome in the entry as a **hard limit** — an
`is_valid` guard where the op crosses it silently, a `Hard limits` row in the
`note` otherwise — **whether or not today's envelope reaches it**:

- **2^31 elements**: an int32 element count or offset wraps;
- **2^31 / row width rows**, for every row width the entry addresses: a row
  index multiplied by the width in 32 bits wraps there (2^31 / 512 = 2^22
  rows), and the kernel writes out of place without raising;
- **2^24 + 1** for any index that passes through fp32: above 2^24 an fp32
  integer is no longer exact.

The brief lists the row counts for this entry's widths and marks any that fall
inside the envelope. How to probe without the full allocation, and the table
format, are in `references/integer-width-probes.md`.

The reason this is mandatory rather than envelope-driven: an entry certified
"up to the envelope of the day" stops being true the day another change
removes the memory cap that kept the envelope small. One kernel computed its
output offset in 32 bits and silently wrote rows at and above 2^25 out of
place; its cells were correct for the envelope they were certified at, and a
prefill-tiling fix that lifted a memory cap moved the reachable rows past it.
A hard limit in the entry is what makes that change visible: **any change
that removes a memory cap re-reads the hard limits of every entry it feeds.**

## When the surface neighbours a known intermittent defect

`entry-spec.md` states the rule and the case that produced it: prove the
harness can *see* the defect before reporting a clean result. It applies with
full force here — extending coverage next to a known-flaky surface is exactly
where a blind harness returns a confident, meaningless green.

## Report

- entry path, the surface requested, and the verdict: already covered / newly
  certified / failed
- `ARCHS` and the cells as they now read, and what the new cells pinned
- the hard limits the integer-width probes established
- **what remains uncertified** on this entry — the next caller reads this
- every claim matched against disk before sending
