---
name: trtllm-modeling-v2-entry
description: Load before writing or changing a ModelingV2 catalog entry (an OpWrapper subclass and its GPU test) - onboarding an op, verifying an entry at new values, promoting a target's local variants into the catalog, or adding a faster entry. The procedure the orchestrator relies on - where the files go, what the test must do, how a run certifies, and how variants merge.
---

# trtllm-modeling-v2-entry

The contract is code: `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py`
(`OpWrapper`, `Cell`, `Arch`) and the claims gates that read it. Read them,
then two or three entries of the same category with their tests
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/flashinfer_rmsnorm.py`
is the smallest). This skill is only what the code does not say.

## Where the files go

| target in your dispatch | entry | test |
|---|---|---|
| `<category>/<name>` (shared) | `catalog/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| `ops/<category>/<name>` (this target only) | `<target>/ops/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/models/<family>/<target>/<category>/test_modeling_v2_<name>.py` |

**A shared entry is written as a local variant first**: `<target>/ops/<category>/<name>__<site>.py`
and `test_modeling_v2_<name>__<site>.py` beside the target's other op tests,
the paths your dispatch's "Where it lands" names. The class and its
module-level instance keep the entry's name, the target imports the variant,
and the shared files stay as they are: other tasks on the entry work beside
you in variants of their own. A verify starts its variant as a copy of the
entry as it stands, file and test. An entry spans two trees, so "every file
of the entry" means both. A new `ops/<category>/` gets an `__init__.py` with
the copyright header.

## The entry

- `__call__` makes **one** kernel-launching call (a `torch.ops.trtllm.*` op,
  or at tier 3 one FlashInfer public callable) and reads nothing off `self`.
  It returns what it writes, even for an out-variant or in-place op, and
  mirrors the op's schema argument for argument, by name: a drifting
  positional list is a silent wrong answer.
- A FlashInfer callable is imported at module scope, named in the class
  docstring, and called with `backend=` explicit where it takes one (`auto`
  resolves per architecture). Where a trtllm op wraps the same kernel, wrap
  the op. Any other Python callable needs the run's explicit authorization.
- `reference` takes the arguments a cell supplies, by `__call__`'s names, and
  never calls the op, the entry or another entry.
- `is_valid` accepts every argument `__call__` does, and guards only pure
  metadata (shape, stride, dtype, device) observed **on this machine** to
  fail silently. Restating an op's loud validation swaps its error for a
  worse one.
- The `note` holds what the members cannot: the metadata the op reads, the
  preconditions a caller establishes, arch quirks, and the hard limits below.
- A device whose arch has no `Arch` member gets one line in `catalog/_op.py`,
  in a commit of its own.

## The test

- Parametrize off `op.CELLS` and arm the `certifying` fixture of
  `tests/unittest/_torch/modeling_v2/_validating.py` in every cell test: it
  refuses to finish unless the entry's own `__call__` and `reference` ran. A
  test against a local copy of the reference certifies nothing.
- Gate `__call__` against `reference` in a band the test states and derives
  (the dtype and accumulation order it accounts for). A band wider than
  torch's dtype-aware default also runs the wrong variants (swapped operand
  halves, an omitted permutation) and records how far outside it each lands.
- Each input `is_valid` refuses gets a guard test (`with validating(op)` and
  `pytest.raises`), after showing unguarded that the op answers wrongly there.
  The unguarded call touches only memory the test owns; a value that faults or
  writes out of place is shown in a subprocess, or only recorded in the
  `note`.
- **Size persistent state from the cache manager, never from your batch.** A
  KV, recurrent or conv pool's slot count is what capacity planning
  allocated; a test where `pool_slots == batch_size` cannot see a
  slot-addressing defect, because both readings index the same row. **Make
  the slot count a `spec` axis**, at the width the engine allocates at the
  envelope's settings.
- Stateful ops run on minimal **real** metadata and caches, never mocks.
  A collective follows the in-tree shape (`tests/unittest/_torch/modeling_v2/comm/`):
  a rank body driven through `_rank_job`, every rank asserting, a deadline,
  the world size a cell axis.
- Assert `torch.cuda.is_available()` up front, no skip paths, seed all
  randomness. Import the entry by its absolute path; share no fixture with
  another entry's test.

## Test cost

The test runs in pre-merge CI on every commit.

- **Budget: a new cell fits 10 s and 40 GiB of device memory.**
- A cell that must reach a bound (a pool at the `kv_cache_fraction` bound, a
  2^31-element probe) is `large`: `large=True` in its `spec`, parametrized as
  `pytest.param(cell, marks=pytest.mark.large)`. Pre-merge runs `-m "not large"`.
  The marker must be registered in `tests/unittest/pytest.ini` under `markers`;
  where it is not, say so in your report rather than editing that file.
- **One launch for many shapes.** Cells one process can drive are one
  parametrized test in one pytest process.
- **Iterate on what you change, certify once.** Drive the cells you change
  with `pytest -k`; the whole file and the claims gates
  (`tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py`,
  `tests/unittest/_torch/modeling_v2/test_modeling_v2_no_stale_claims.py`)
  run once, at the end, through the test-cache command your prompt names.

## Integer-width boundaries

For each index-typed argument (offsets, positions, page and slot tables, row
counts) compute where it crosses 2^31 elements, 2^31 / row width rows (for
each width it addresses) and 2^24 + 1 through fp32. When the envelope reaches
a boundary, probe it, and record the outcome as a hard limit: an `is_valid`
guard where the op crosses it silently, a `Hard limits` row in the `note`
otherwise. When it does not, write the computed boundary in the `note`.

## Certifying

Add this device's arch to `ARCHS` (from `torch.cuda.get_device_capability()`)
only after the whole test file passed here, in a run that post-dates the last
write to every file of the entry. A value the op demonstrably fails at is
knowledge, and lands: outside `CELLS`, refused by `is_valid` if the op answers
silently wrong, measured in the `note`. A failed receipt whose commits change
only the entry's own files is integrated as it stands; one that changes
anything else stays on the task's branch, and the task's next attempt starts
from it.

## A macro

When the op cannot be one kernel-launching call (its body composes modules,
registered ops or tensor glue in Python), write no entry: name its
constituents in your decision, each with its proposed entry and tier, and
omit one an existing entry already serves.

## Promote

A `promote` merges every local variant of each named entry into its shared
home, in one commit:

1. `__call__` and `reference` stay the entry's. Where a variant's differ,
   the merged entry takes the form every call site can use and every caller
   is updated in the same commit; when no one form serves them all, write
   nothing and block, naming the difference.
2. `CELLS`, `ARCHS` and the `is_valid` guards are the unions of the variants'
   (one cell per distinct `spec`, keeping its `large` mark), and the `note`
   carries every measurement they recorded.
3. One test file at the shared home, `_build` covering every `spec` key a
   variant used; a cell keeps the band it was certified under, the tighter
   where two differ.
4. The target, and any other target that kept its own copy, imports the
   catalog; the variants and their tests go (`git rm`).
5. The merged tests and the claims gates run once, in one pytest process. A
   cell that passed in its variant and fails merged is a conflict to report,
   never to land.

## Report

The entry's path, the op it wraps and its tier, `ARCHS` and `CELLS` as
committed, the hard limits, the certifying command and its wall time, and
**what remains uncertified**: the next caller reads it.
