---
name: trtllm-modeling-v2-kernel-onboard
description: Load when smith must add a catalog entry for a TensorRT-LLM callable that has none — read the source, fix the single-call boundary, write the OpWrapper entry (__call__, reference, is_valid, CELLS, ARCHS) and its GPU test in the target's own ops/ (a shared entry as a local variant the promote moves into the catalog), drive the cells. Also covers the macro verdict, where the run produces a decomposition report and dispatches one child per constituent instead.
---

# trtllm-modeling-v2-kernel-onboard

One run, one entry. Four possible outcomes, all of them valid products:

| outcome | when |
|---|---|
| a new entry, its cells driven and passing, `ARCHS` naming this arch | the callable is one kernel-launching call |
| an entry whose failing value is refused, not certified — `is_valid` guards it if the op answers wrongly, the `note` records what was measured | wrapper and test are right and the op demonstrably fails at a value the call site needs. **Land it anyway** — that is knowledge: a `FAILED_RECEIPT` whose commits are the entry's files only is integrated as it stands |
| a decomposition report, no entry | the callable turns out to be a macro |
| a `FAILED_RECEIPT` naming an engine defect, handed back early | the call fails in shared code the entry wraps (an upstream kernel, a library, the engine), not in the wrapper or its test: stop once it is measured, commit the guard and the failing test, and put the `engine_defect` block (`affects`, `trigger`, `detect`, `proposal`) in your decision file. The orchestrator adds the task that fixes it |

When your dispatch carries a `cells` block, or "The surface to certify",
certify exactly those cells, plus one at the envelope's largest row: the
workflow chose them from the target's calls and the measured geometry. A
value you notice beyond them goes in the `note` as uncertified.

References: `references/entry-spec.md` is the shared spec (atomicity, where an
entry goes, the `OpWrapper` members, certification).
`references/contract-evidence.md` is what makes a sentence in an entry
defensible. `references/decomposition.md` is the macro path. Read the first
two before writing anything.

An entry is **one `OpWrapper` subclass in its own file and its GPU test**,
and nothing else — no prose contract beside it, no index row: the class is the
contract (`__call__`, `reference`, `is_valid`, `CELLS`, `ARCHS`, `note`), and
the test that drives its cells is what keeps it true. It lives in one of two homes, which your dispatch target names:

| target | entry | test |
|---|---|---|
| `<category>/<name>` (shared) | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| `ops/<category>/<name>` (this target only) | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/ops/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py` |

A shared entry is written first as a **local variant** in this target's own
`ops/`, `ops/<category>/<name>__<site>.py` and its test
`test_modeling_v2_<name>__<site>.py` in the target's tests: your dispatch
prompt's "Where it lands" names the two files. The class and its module-level
instance keep the entry's name (`<name> = <Class>()`), and the target imports
the variant. Before delivery a `promote` dispatch moves every variant into the
shared home above (`trtllm-modeling-v2-kernel-verify`,
`references/promote.md`), so other tasks on the same entry never wait for
yours. Either way the entry spans two trees, and every rule below that says
"every file of the entry" means both.

## Flow

1. **Locate the callable.** `.claude/skills/trtllm-modeling-v2-kernel-feasibility/references/trtllm-ops.md` (the generated inventory,
   with schemas) or, under the run's launch prefix and declared interpreter,
   `agent_flow/workflows/trtllm_modeling_v2/tools/list_trtllm_ops.py -f <substr>`.
   Every listed op is callable as `torch.ops.trtllm.<name>`. A tier-3
   dispatch names a FlashInfer callable instead (`flashinfer.<module>.<name>`,
   listed in the workspace's `surface.md`); read its signature and docstring
   under the run's declared interpreter (`python -c "import inspect,
   flashinfer.<module> as m; print(inspect.signature(m.<name>)); print(m.<name>.__doc__)"`)
   and its source in the installed package. Then read the implementation and
   its Python call path until you can answer: what does one call compute, what
   state or metadata must exist first, which dtypes, arches and `backend=`
   choices does it route.
2. **Fix the boundary, the category and the home.** One entry = one op
   invocation at the Python level; fusion is legal only *inside* that call.
   Categories: `gemm attention moe norm activation quantization comm ssm`,
   chosen by compute role rather than dtype. Native torch is not an entry —
   there is no `torch/` category; if your verdict is that plain torch serves
   the step, report it and write nothing. No category fits → stop and report.
   The home is the one your dispatch target names, and a shared one lands
   where your dispatch prompt's "Where it lands" says; if you find the op is
   also called by another target, say so in the report rather than moving it.
   **A macro verdict here ends this path**: go to `references/decomposition.md`.
3. **Write the entry and its test**, to the spec. `__call__` makes the one op
   call, takes every argument explicitly and reads nothing off `self`; it
   returns what it writes. The file ends with the entry's one module-level
   instance (`<name> = <Class>()`), which is what targets import. `CELLS` are
   the configurations the entry's call sites reach, one `Cell(why=..., spec=...)` per
   distinct thing covered. The test parametrizes off `op.CELLS`, arms
   `_validating.py`'s `certifying` fixture (or runs the call under its
   `validating`), gates `__call__` against `reference` within a band it states
   and derives, and adds a guard test per input `is_valid` refuses.
   The test file is named `test_modeling_v2_<name>.py`, always — that name is
   what keeps it from colliding with upstream's test of the same op, and what
   the CI selector matches. A new `ops/<category>/` directory gets an
   `__init__.py` with the copyright header.
4. **Run the test on GPU**, under the run's launch prefix and declared
   interpreter:
   `<prefix> <python> -m pytest <the test path from the table above>`.
   One command per entry, multi-rank ops included — a collective's test starts
   its own ranks rather than asking the caller for a launcher. Iterate on the
   cells you are working on (`-k`) until every cell passes, or until you have
   established a genuine op failure; each cell within the budget in
   `references/entry-spec.md` ("Test cost"). A fake pass is not a product.
5. **Certify from one run at the end**: add this device's arch to `ARCHS`
   (from `torch.cuda.get_device_capability()`; if `Arch` has no member for it,
   add one in `catalog/_op.py` in a commit of its own), and nothing else, then
   run the whole test file and the claims gates once. That run counts only if
   it post-dates the last write to every file of the entry.
6. **Commit** exactly the entry's files in both trees (a shared entry's
   local variant and its test), plus a new `ops/<category>/__init__.py` when
   you created one, through the commit helper your prompt names.
7. **Report.**

## The macro path

On a macro verdict, the run's product is the report and the subtree:

1. write `references/decompositions/<op>.md` to the checklist in
   `references/decomposition.md`
2. **commit it before dispatching anything** — your context does not survive a
   session cut, and the report plus `git log` is what lets a fresh smith
   continue
3. return `MACRO` with a `constituents` list (the smith prompt gives its
   shape); the orchestrator queues one dispatch per constituent, in
   dependency order, each loading this skill
4. report what remains for the constituents to cover

A child does not dispatch. One that finds its own callable is a macro writes a
report and returns it.

<!-- PORT: verify — where a committed decomposition report lands in tree is
     unresolved. The skills are materialized into each run's workspace, so a
     report written to `references/decompositions/` is not obviously the thing
     `git log` preserves; the alternatives are the packaged skill directory
     (`agent_flow/workflows/trtllm_modeling_v2/skills/trtllm-modeling-v2-kernel-onboard/
     references/decompositions/`) or the workspace `reports/`. The source also
     disagreed with itself: `entry-spec.md` said
     `docs/references/<op-name>-decomposition.md` while this file and
     `decomposition.md` said `references/decompositions/<op>.md`. Unified here
     on the latter; needs an owner's decision. -->

## Report

- entry path and home (shared / `ops/` of which target); the wrapped op's
  qualified name and the sourcing tier it is (also the decision file's
  `tier`); `ARCHS` and `CELLS` as committed (each cell's `why`); the test's
  command and its wall time in the certifying run; files touched — in
  **both** trees
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
