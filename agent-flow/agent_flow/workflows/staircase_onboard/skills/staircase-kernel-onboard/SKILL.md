---
name: staircase-kernel-onboard
description: Load when smith must add a catalog entry for a TensorRT-LLM callable that has none — read the source, fix the single-call boundary, write the OpWrapper entry (raw_call, reference, CELLS, ARCHS) and its GPU test in the shared catalog or the target's own ops/, drive the cells, sync the index. Also covers the macro verdict, where the run produces a decomposition report and dispatches one child per constituent instead.
---

# staircase-kernel-onboard

One run, one entry. Three possible outcomes, all of them valid products:

| outcome | when |
|---|---|
| a new entry, its cells driven and passing, `ARCHS` naming this arch, index synced | the callable is one kernel-launching call |
| an entry whose failing value is refused, not certified — `is_valid` guards it if the op answers wrongly, the `note` records what was measured | wrapper and test are right and the op demonstrably fails at a value the call site needs. **Land it anyway** — that is knowledge |
| a decomposition report, no entry | the callable turns out to be a macro |

References: `references/entry-spec.md` is the shared spec (atomicity, where an
entry goes, the `OpWrapper` members, certification, index).
`references/contract-evidence.md` is what makes a sentence in an entry
defensible. `references/decomposition.md` is the macro path. Read the first
two before writing anything.

An entry is **one `OpWrapper` subclass and its GPU test**, and there is no
contract `.md`: the class is the contract (`raw_call`, `reference`, `is_valid`,
`compare`, `CELLS`, `ARCHS`, `note`), and the test that drives its cells is what
keeps it true. It lives in one of two homes, which your dispatch target names:

| target | entry | test |
|---|---|---|
| `<category>/<name>` (shared) | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| `ops/<category>/<name>` (this target only) | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/ops/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py` |

Either way the entry spans two trees, and every rule below that says "every
file of the entry" means both.

## Flow

1. **Locate the callable.** `.claude/skills/staircase-kernel-feasibility/references/trtllm-ops.md` (the generated inventory,
   with schemas) or, under the run's launch prefix and declared interpreter,
   `agent_flow/workflows/staircase_onboard/tools/list_trtllm_ops.py -f <substr>`.
   Every listed op is callable as `torch.ops.trtllm.<name>`. Then read the
   implementation and its Python call path until you can answer: what does one
   call compute, what state or metadata must exist first, which dtypes and
   arches does it route.
2. **Fix the boundary, the category and the home.** One entry = one op
   invocation at the Python level; fusion is legal only *inside* that call.
   Categories: `gemm attention moe norm activation quantization comm ssm`,
   chosen by compute role rather than dtype. Native torch is not an entry —
   there is no `torch/` category; if your verdict is that plain torch serves
   the step, report it and write nothing. No category fits → stop and report.
   The home is the one your dispatch target names; if you find the op is also
   called by another target, say so in the report rather than moving it.
   **A macro verdict here ends this path**: go to `references/decomposition.md`.
3. **Write the entry and its test**, to the spec. `raw_call` makes the one op
   call and reads nothing off `self`; it returns what it writes. `CELLS` are
   the configurations the smith brief lists, one `Cell(why=..., spec=...)` per
   distinct thing covered. The test parametrizes off `op.CELLS`, arms the
   `certifying` fixture, and adds a guard test per input `is_valid` refuses.
   The test file is named `test_modeling_v2_<name>.py`, always — that name is
   what keeps it from colliding with upstream's test of the same op, and what
   the CI selector matches. A new `ops/<category>/` directory gets an
   `__init__.py` with the copyright header.
4. **Run the test on GPU**, under the run's launch prefix and declared
   interpreter:
   `<prefix> <python> -m pytest <the test path from the table above>`.
   One command per entry, multi-rank ops included — a collective's test starts
   its own ranks rather than asking the caller for a launcher. Iterate until
   every cell passes, or until you have established a genuine op failure. A
   fake pass is not a product.
5. **Certify from that observed run only**: add this device's arch to `ARCHS`
   (from `torch.cuda.get_device_capability()`; if `Arch` has no member for it,
   add one in `catalog/_op.py` in a commit of its own), and nothing else.
6. **Shared entries only: write the index fragment**
   `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.d/<category>/<name>.yaml`
   (shape in `references/entry-spec.md`). Never edit `catalog/index.yaml`: the
   orchestrator regenerates it from the committed fragments. A target's
   `ops/` entry has no index row.
7. **Do not format or lint during iteration.** The repo's `pre-commit` runs
   once, at delivery, and covers formatting, import order and the general lint
   rules in one pass. What the retired per-entry recipe taught is why that pass
   is not narrowed: an import-only check (`ruff check --select I`) restricts
   ruff to import rules, so an unused variable passes it, passes the formatter,
   and passes a type check — an entry once shipped with a dead `F841` in its
   test because the recipe stopped at the first two. If you do run a checker by
   hand, run it **by name on your own files**, so a sweep does not reformat
   entries you did not touch.
8. **Revalidate the run.** Steps 5–7 can all write to the entry after the
   test last ran — a cell added, a helper renamed, a formatter rewriting the
   test file. The run you certify from is valid only if it **post-dates the
   last write to every file of the entry, in both trees**. Check it
   mechanically against the files' mtimes, not by recollection: a
   certification that predated its own test file has shipped here before, and
   it read as green. Stale → rerun the whole test file.
9. **Commit**, pathspec-scoped to exactly your files. Shared:

```bash
git add tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py \
        tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py \
        tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.d/<category>/<name>.yaml
git commit -m "feat(catalog): onboard <name> — <one line>"
```

   A target's own op (`T` is its `models/<family>/<ckpt>__<arch>__<parallel>`):

```bash
git add tensorrt_llm/_torch/_experimental/modeling_v2/$T/ops/<category>/<name>.py \
        tests/unittest/_torch/modeling_v2/$T/<category>/test_modeling_v2_<name>.py
git commit -m "feat(ops): onboard <name> for <target> — <one line>"
```

   plus the new `ops/<category>/__init__.py` when you created one.

10. **Report.**

## The macro path

On a macro verdict, the run's product is the report and the subtree:

1. write `references/decompositions/<op>.md` to the checklist in
   `references/decomposition.md`
2. **commit it before dispatching anything** — your context does not survive a
   session cut, and the report plus `git log` is what lets a fresh smith
   continue
3. write each child's prompt into the journal, then dispatch one
   `Agent(smith)` per constituent in dependency order, each loading this skill
4. report the subtree: which constituents landed, with which `ARCHS` and cells, and
   what remains

A child does not dispatch. One that finds its own callable is a macro writes a
report and returns it.

<!-- PORT: verify — where a committed decomposition report lands in tree is
     unresolved. The skills are materialized into each run's workspace, so a
     report written to `references/decompositions/` is not obviously the thing
     `git log` preserves; the alternatives are the packaged skill directory
     (`agent_flow/workflows/staircase_onboard/skills/staircase-kernel-onboard/
     references/decompositions/`) or the workspace `reports/`. The source also
     disagreed with itself: `entry-spec.md` said
     `docs/references/<op-name>-decomposition.md` while this file and
     `decomposition.md` said `references/decompositions/<op>.md`. Unified here
     on the latter; needs an owner's decision. -->

## Report

- entry path and home (shared / `ops/` of which target); the wrapped op's
  qualified name; `ARCHS` and `CELLS` as committed (each cell's `why`); the
  test's command; files touched — in **both** trees
- **what remains uncertified** on this entry — values outside every cell,
  inputs `is_valid` refuses, hard limits — the next caller reads this
- **the decomposition map**, whenever the request turned out composite: every
  constituent in call order, each tagged `existing entry: <path>` /
  `not onboarded: <callable>` / `gap`. Every symbol you name must be verified
  resolvable in the installed engine — never cite one you have not resolved.
  This map is why the composite can stay a black box afterwards: you opened it
  once so nobody has to again
- symptoms: out-of-scope problems observed (trtllm bugs, env issues)
- every claim about the artifacts must match disk. Re-check before sending
