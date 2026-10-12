---
name: staircase-kernel-onboard
description: Load when smith must add a catalog entry for a TensorRT-LLM callable that has none — read the source, fix the single-call boundary, write contract + wrapper + GPU test, record the receipt, sync the index. Also covers the macro verdict, where the run produces a decomposition report and dispatches one child per constituent instead.
---

# staircase-kernel-onboard

One run, one entry. Three possible outcomes, all of them valid products:

| outcome | when |
|---|---|
| a new entry, contracted, tested, receipt recorded, index synced | the callable is one kernel-launching call |
| an entry with `status: failed` plus a short note | wrapper and test are right and the op demonstrably fails here. **Land it anyway** — that is knowledge |
| a decomposition report, no entry | the callable turns out to be a macro |

References: `references/entry-spec.md` is the shared spec (atomicity, the
three pieces, receipts, index). `references/contract-evidence.md` is what
makes a contract sentence defensible. `references/decomposition.md` is the
macro path. Read the first two before writing anything.

An entry **spans two trees**: the contract and the wrapper live in
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/`, the GPU
test lives at
`tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py`.
Every rule below that says "every file of the entry" means both trees.

## Flow

1. **Locate the callable.** `.claude/skills/staircase-kernel-feasibility/references/trtllm-ops.md` (the generated inventory,
   with schemas) or, under the run's launch prefix and declared interpreter,
   `agent_flow/workflows/staircase_onboard/tools/list_trtllm_ops.py -f <substr>`.
   Every listed op is callable as `torch.ops.trtllm.<name>`. Then read the
   implementation and its Python call path until you can answer: what does one
   call compute, what state or metadata must exist first, which dtypes and
   arches does it route.
2. **Fix the boundary and the category.** One entry = one op invocation at the
   Python level; fusion is legal only *inside* that call. Categories:
   `gemm attention moe norm activation quantization comm ssm`, chosen by
   compute role rather than dtype. `torch/` is reserved for hand-added
   mirrors — **never write there**, including when your verdict is that a
   torch callable is the answer: report that verdict and let the orchestrator
   decide, since a computation mirror widens what the surface is for. No
   category fits → stop and report.
   **A macro verdict here ends this path**: go to `references/decomposition.md`.
3. **Write the three pieces**, to the spec. The test file is named
   `test_modeling_v2_<name>.py`, always — that name is what keeps it from
   colliding with upstream's test of the same op, and what the CI selector
   matches.
4. **Run the test on GPU**, under the run's launch prefix and declared
   interpreter:
   `<prefix> <python> -m pytest tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py`.
   One command per entry, multi-rank ops included — a collective's test spawns
   its own ranks rather than asking the caller for a launcher. Iterate until
   it passes, or until you have established a genuine op failure. A fake pass
   is not a product.
5. **Record the receipt** from that observed run only.
6. **Write the entry's index fragment**
   `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.d/<category>/<name>.yaml`
   (shape in `references/entry-spec.md`). Never edit `catalog/index.yaml`: the
   orchestrator regenerates it from the committed fragments.
7. **Do not format or lint during iteration.** The repo's `pre-commit` runs
   once, at delivery, and covers formatting, import order and the general lint
   rules in one pass. What the retired per-entry recipe taught is why that pass
   is not narrowed: an import-only check (`ruff check --select I`) restricts
   ruff to import rules, so an unused variable passes it, passes the formatter,
   and passes a type check — an entry once shipped with a dead `F841` in its
   test because the recipe stopped at the first two. If you do run a checker by
   hand, run it **by name on your own files**, so a sweep does not reformat
   entries you did not touch.
8. **Revalidate the receipt.** Steps 5–7 can all write to the entry after the
   test last ran — a fixture added, a helper renamed, a formatter rewriting
   the test file. A receipt is valid only if it **post-dates the last write to
   every file in the entry, in both trees**. Check it mechanically against the
   files' mtimes, not by recollection: a receipt that predated its own test
   file has shipped here before, and it read as green.
9. **Commit**, pathspec-scoped to exactly your files:

```bash
git add tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.md \
        tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py \
        tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py \
        tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.d/<category>/<name>.yaml
git commit -m "feat(catalog): onboard <name> — <one line>"
```

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
4. report the subtree: which constituents landed, with which receipts, and
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

- entry path; the wrapped op's qualified name; the receipt as recorded; what
  the test covered (shapes, dtypes); files touched — in **both** trees
- **the decomposition map**, whenever the request turned out composite: every
  constituent in call order, each tagged `existing entry: <path>` /
  `not onboarded: <callable>` / `gap`. Every symbol you name must be verified
  resolvable in the installed engine — never cite one you have not resolved.
  This map is why the composite can stay a black box afterwards: you opened it
  once so nobody has to again
- symptoms: out-of-scope problems observed (trtllm bugs, env issues)
- every claim about the artifacts must match disk. Re-check before sending
