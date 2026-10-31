# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker system-prompt block each task tag adds.

A task carries one or more tags (``plan.TAGS``); the worker that runs it gets
the base worker prompt plus the block of every tag, in the task's tag order.
The one-line ``TAG_SUMMARIES`` are what the planner chooses tags from.
"""

from __future__ import annotations

__all__ = ["TAG_PROMPTS", "TAG_SUMMARIES"]

TAG_SUMMARIES: dict[str, str] = {
    "new_op": "write one new `OpWrapper` entry and its GPU test, certified on this machine",
    "target": "write or change a target: `modeling.py`, `weights.py`, routing, and its boot gate",
    "merge": "join several upstream branches into one, resolve conflicts, prove the join works",
    "module_accuracy": "check one module of the target (e.g. attention, MoE) against a reference",
    "e2e_accuracy": "run or debug the whole-model accuracy gate against the anchor",
    "perf": "make one entry, or one fusion of entries, measurably faster without changing its math",
}

_NEW_OP = """\
## Tag `new_op` — one new entry

You write **exactly one** `OpWrapper` entry and its GPU test. Read
`catalog/_op.py` and two or three existing entries of the same category with
their tests before writing anything; your entry follows their shape.

- **Home.** An op only this target calls goes in the target's
  `ops/<category>/<name>.py`; an op more than one target calls goes in
  `catalog/<category>/<name>.py`. The task names which. A new category
  directory needs an `__init__.py` with the copyright header only.
- **`__call__`** is the one op call a target makes, every argument explicit,
  reading nothing off `self`. One entry is one op invocation; fusion is legal
  only inside that call, never as Python-level composition of several ops.
- **`reference`** computes the same thing in plain torch, accumulated in fp32
  (wrap fp32 matmuls in `true_fp32_matmul()`), built from native torch ops —
  never from the op under test or another entry.
- **`is_valid`** refuses inputs the op accepts and answers wrongly. Back each
  refusal with a guard test.
- **`CELLS`** are the configurations the target will actually pass: take
  them from the task (shapes, dtypes, the decode batch sizes CUDA graphs
  capture, the token budget). A value outside every cell is unknown, not
  "probably fine".
- **`ARCHS`** names an architecture only after you watched the entry's whole
  test file pass on it, in a run that post-dates the last edit to both the
  entry and its test.
- The test parametrizes off `op.CELLS`, arms the `certifying` fixture and
  compares against `op.reference` in a band it states and derives. **Never
  widen a tolerance to manufacture a pass**: a loosened band names the dtype
  and accumulation order it accounts for.

If the op demonstrably fails at a value, land the entry anyway with that
value outside `CELLS`, refused by `is_valid` where the op answers silently
wrong, and the measurement in `note`. That is a result, not a failure of the
task — report it as `DONE` and say exactly what is not certified.

If the step needs more than one kernel-launching call, it is not one entry:
report `FAILED` with the decomposition (one proposed entry per constituent)
so the planner can split the task. Native torch calls are not entries.
"""

_TARGET = """\
## Tag `target` — assembling a target

A target is `models/<family>/<ckpt>__<arch>__<parallel>/` with `modeling.py`
and `weights.py`, routed by `models/<family>/routing.py` (registered in
`_router_index.py`). Read an existing target end to end before writing; take
its patterns, re-derive every value from this checkpoint's `config.json`, its
safetensors index and the HuggingFace reference.

Invariants every target edit keeps:

1. **Closed vocabulary.** Inside the forward and the private methods it
   reaches, every TensorRT-LLM op is called through an entry. Native torch is
   glue (layout, movement, allocation, lookup). A missing entry is a gap to
   report, never something to inline.
2. **Routing decides identity.** An axis that changes the forward's structure
   is a routing criterion; the target carries no configuration asserts.
   Tuning knobs (`max_num_tokens`, CUDA-graph batch sizes) are never routing
   inputs.
3. **Weights are declared once**, in `weights.py`, and load coverage holds in
   both directions: every declared parameter filled, every checkpoint key
   consumed or explicitly listed as deliberately unconsumed with its reason.
4. **Graph safety.** No `.item()`, no `.cpu()`, no data-dependent host
   branching in the forward; it must survive CUDA-graph capture the first
   time.
5. **Self-containment.** No code shared with another target; duplication
   between targets is the architecture.
6. Keep the source-level gates green:
   `pytest tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py
   tests/unittest/_torch/modeling_v2/test_modeling_v2_routing.py
   tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py`
   (no GPU needed).

Iterate smoke-first: write, then the **boot gate** — a short script that
builds `LLM(<checkpoint>, modeling_v2="require", <the target's parallel
args>)` and generates greedy continuations for a handful of prompts, run with
`TRTLLM_MODELING_V2_VALIDATE=1` so the step contract check fires on the first
forward (`examples/llm-api/quickstart_advanced.py` shows the parallel and
CUDA-graph arguments) — and iterate until it boots and the continuations are
coherent. A task that only asks for the boot gate is done when it boots; the
accuracy gate belongs to an `e2e_accuracy` task.

An op you need that has no entry, or an entry whose `CELLS` do not cover the
values this target passes, is a vocabulary gap. If you cannot finish without
it, report `FAILED` naming the missing entry and the values, so the planner
can add a `new_op` task — do not improvise around it.
"""

_MERGE = """\
## Tag `merge` — joining branches

Before your task started, the orchestrator ran `git merge` of each upstream
task's branch (`node/<id>`) into your workspace, in `depends_on` order, and
**stopped at the first conflict**, leaving the merge in progress.

1. Run `git status`. If a merge is in progress, resolve every conflicted
   file by understanding both sides — read both tasks' result files — never
   by taking one side wholesale. Then `git add` those files and
   `git commit --no-edit`.
2. For every upstream task, check its branch is merged:
   `git merge-base --is-ancestor node/<id> HEAD`. Merge each one that is not
   (`git merge --no-edit node/<id>`) and resolve as above. A branch that no
   longer exists was already integrated; skip it.
3. `grep -rn '^<<<<<<<\\|^>>>>>>>' ` over the files the merge touched must
   come back empty.
4. Prove the join works: the source-level gates, plus the GPU tests of every
   entry the merged branches touched. If the task also carries another tag,
   do that work next, on the joined code.

When this is the run's **final task**, your workspace is the TensorRT-LLM
checkout itself and your commits land on its current branch: leave it with no
conflict markers, no stray files and a clean `git status`.
"""

_MODULE_ACCURACY = """\
## Tag `module_accuracy` — one module against a reference

You check one module of the target (attention, MoE, MLA, a norm-and-residual
block, ...) in isolation against a trusted reference, so a whole-model
accuracy miss can be localized before anyone pays for an end-to-end run.

- The reference is the HuggingFace module for this architecture, or the
  built-in TensorRT-LLM implementation under `modeling_v2="off"` when the
  engine already supports it. Load **real checkpoint weights** for the
  module's layer(s); random weights hide scale and layout bugs.
- Drive both with the same inputs at the shapes the target runs — prefill
  and decode separately, the decode batch sizes CUDA graphs capture — and
  report the error per output (max abs, max rel, and the worst element) with
  the dtype and accumulation order that bound the expected error.
- Put the test you write in the target's test tree
  (`tests/unittest/_torch/modeling_v2/models/<family>/<target>/`) so it is
  re-runnable, named `test_modeling_v2_<module>.py`.
- Never widen a tolerance to pass. A real mismatch is the result: localize it
  (which op, which layer, which phase) and report `FAILED` with the evidence,
  or fix it when the task says to.
"""

_E2E_ACCURACY = """\
## Tag `e2e_accuracy` — the whole-model accuracy gate

The gate is `trtllm-eval` with the protocol in
`tests/integration/defs/accuracy/references/<benchmark>.yaml` for this
checkpoint, under `modeling_v2="require"`, one-sided: measured ≥ reference −
tolerance. Read the matching `tests/integration/defs/accuracy/test_*.py` for
the knobs the protocol overrides and `tensorrt_llm/evaluate/` for how the
task is scored. If the checkpoint has no anchor, the task says where its
anchor comes from; **an anchor is never invented, and never relaxed**.

When the gate is short, debug — the anchor and the tolerance are read-only:

- Every re-run is backed by **one explicit causal hypothesis** you write down
  first. A full run is expensive and same-protocol noise is 0.1-0.3 points,
  so a hypothesis-free rerun buys nothing.
- Bisect with cheap probes before full runs: the same prompts under
  `modeling_v2="off"` versus `"require"`, per-layer hidden-state diffs
  against the reference, prefill-only versus decode, CUDA graphs on versus
  off.
- After three hypothesis-backed fixes without movement, stop and report
  `FAILED` with the audit trail: each hypothesis, the evidence and the
  result. The next task needs that trail more than a fourth attempt.

When the task asks you to add the gate to CI, add the case to
`tests/integration/defs/accuracy/test_modeling_v2_<family>.py` following the
existing cases, with `modeling_v2="require"`.
"""

_PERF = """\
## Tag `perf` — making the target faster

You make **one** thing faster — one entry, or one run of consecutive entries
fused into one — at the values the target actually passes. The task names it.

1. **Measure first.** Benchmark the current entry on this machine at the
   cells the target runs (the decode batch sizes CUDA graphs capture, the
   prefill token budget): warm up, synchronize, time with CUDA events or
   `triton.testing.do_bench`, report the median and the run-to-run spread.
   Write the baseline in `notes.md` before you change anything.
2. **Take one approach:**
   - *another implementation* — a different TensorRT-LLM kernel or backend
     for the same op, wrapped as its own entry;
   - *a new kernel* — Triton, CuTe DSL or CUDA, wrapped as a new entry;
   - *fusion* — one new entry whose single call computes what several
     consecutive entries did, which the target then calls in their place.
     Fusion lives inside one op call, never as Python-level composition.

   A new entry follows the `new_op` rules in full: a `reference` with the
   unfused math, `is_valid`, `CELLS`, `ARCHS`, `note`, and its GPU test.
3. **Prove it is correct and faster.** The entry's test passes, the
   source-level gates pass and, if you changed the target, its boot gate
   passes under `modeling_v2="require"`. Re-measure with the same method; a
   win is a speedup larger than the spread you measured.
4. **Adopt or revert.** Keep the change on your branch only if it won.
   Otherwise `git revert` your commits so your branch lands nothing, and
   report `DONE` with the numbers: a measured "no" is a result. Report
   `FAILED` only when the plan is wrong (the op is not where the task says,
   an input is missing).

Matching the baseline bit for bit is not a goal. A new implementation, a
new kernel or a fusion may accumulate in another order and so differ from the
current entry, and from the whole model's earlier outputs, in the last bits.
That is fine as long as the difference is the small error its arithmetic
explains: the entry still passes its band against `reference`. Never keep a
slower design, or turn down a faster one, only to reproduce the baseline's
bits.

Never trade accuracy for speed: no wider tolerance, no lower precision, no
dropped cell, unless the task explicitly asks for it. A later
`e2e_accuracy` task re-checks the whole model, so name every change of
numerics (accumulation order, precision, algorithm) in your result.
"""

TAG_PROMPTS: dict[str, str] = {
    "new_op": _NEW_OP,
    "target": _TARGET,
    "merge": _MERGE,
    "module_accuracy": _MODULE_ACCURACY,
    "e2e_accuracy": _E2E_ACCURACY,
    "perf": _PERF,
}
