# Mapping and triage — reading an entry to place a step

How to turn one computation step plus this checkpoint's values into one of
three states, and what each state dispatches.

## Reading the certified column

An entry is an `OpWrapper` class
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py`), and five
places in it bound what it certifies. Read all five; `ARCHS` alone is not the
answer.

| where | what it bounds |
|---|---|
| `ARCHS` | the **architectures** the cells were driven on. An arch not in it is **unknown** — not passed, not failed |
| `CELLS` | the input configurations certified, each a `Cell(why=..., spec=...)`; a multi-rank entry's world size is a `spec` axis |
| `__call__` signature | the arguments, by the op's own names — what a call site must supply, every call |
| `is_valid` | inputs the op accepts and answers wrongly; a value it refuses is outside the entry for good |
| `note` | metadata consumed, preconditions, arch quirks, hard limits, and what is stated as not certified |

Two reading rules that have each cost a run:

**The cells are the column.** A value is covered when a cell includes it, or
when the cells' `why`s say the op is uniform across the range it falls in (a
cell per bucket edge, per tile boundary, per code path). "Cells at head counts
8, 16, 128" do not certify 32. A `note` stating a rule backed by cells at its
edges does.

**An arch this run targets that is not in `ARCHS` makes the row CERT, not
OK**, however many cells there are. Architecture is the whole key — there is
no version pin in tree to drift against — and a merged entry's test runs in
pre-merge, so its cells are re-proved on every commit.

## The outcomes

| situation | state | dispatch |
|---|---|---|
| shared entry exists, a cell covers this checkpoint's values, `ARCHS` names this run's arch | **OK** | nothing — the row is chapter 1's evidence, not a dispatch |
| shared entry exists, some value falls outside every cell | **CERT** | `trtllm-modeling-v2-kernel-verify`, with the value and the cell or `note` line that excludes it |
| another target keeps the op in its own `ops/` | **GAP** (promotion) | `trtllm-modeling-v2-kernel-onboard`, target `<category>/<name>`, naming the existing entry in `note` — it and its test move to the shared catalog and the cells are driven again |
| no entry, but `references/trtllm-ops.md` has a candidate callable | **GAP** (tier 1) | `trtllm-modeling-v2-kernel-onboard`, with the candidate and the step it serves; target `ops/<category>/<name>` when only this target calls it |
| no entry and no trtllm op, but a FlashInfer public callable computes the step (`surface.md`, "FlashInfer API") | **GAP** (tier 3) | `trtllm-modeling-v2-kernel-onboard`, candidate `flashinfer.<module>.<name>`: the entry's `__call__` calls it directly. FlashInfer is a pinned dependency; most of its kernels have no trtllm op, so the op inventory cannot see them |
| no entry and no candidate | **ceiling** | nothing — the run ends here. **First rule out FlashInfer's API and a pybind binding**: the inventory is generated from the torch schema registry and lists `torch.ops.*` only, so a step served by a binding (`thop_attention` is one) searches empty and reads as a ceiling. Check the catalog's existing entries and their *uncertified* surfaces before concluding |

One more case that dispatches nobody: **a step plain torch serves**. Native
torch calls are not entries — tensor layout, movement, allocation, lookup
(reshape, split, concat, copy, empty, expand, embedding, residual add) — and the
target calls torch directly; assembler writes those itself. It is glue by
default: a computation the HF reference expresses as a named module is never
*composed* from torch calls in the target, and a missing kernel there is a GAP
(tier 4 or 5: an entry this run writes) or a ceiling.

**The one exception**, narrow by design: a computation may be a direct torch
call when it is **exactly one torch callable**, **no trtllm op and no FlashInfer
callable serves it** (searched closed over the inventory, the survey's FlashInfer
API *and* the pybind surface the inventory cannot show), and upstream owns its
correctness. Such a step is neither GAP nor
ceiling, and not a `steps` row — list it in the plan as a direct torch call,
with the search that closed it.

## Precision in a CERT row

A CERT row becomes smith's dispatch prompt almost verbatim, so vagueness costs
a whole run. It must carry:

- the entry path
- **the value**, not the axis: `nope=192, v=256, H=16 per rank, fp8 pool`, not
  "MLA geometry"
- **the cell or `note` line that excludes it**, quoted. If you cannot quote
  one, the row is probably OK and you are dispatching on a hunch

## Two failure modes in this step

**Reading the source instead of the entry.** Nothing forbids you from
opening it — but if an entry's cells and `note` cannot answer whether a value
is covered, that is a **defect in the entry**, and patching around it by
reading source leaves the next reader with the same problem. Record the
defect; the fix is smith adding cells from an observed run, which is what
`trtllm-modeling-v2-kernel-verify` does.

**Grouping rows to "save dispatches".** One dispatch, one entry. Entries that
share an expensive test fixture are still separate dispatches — catalog tests
may not import each other, so there is no fixture to share, only the knowledge
of how to build one, and that travels in smith's reports.
