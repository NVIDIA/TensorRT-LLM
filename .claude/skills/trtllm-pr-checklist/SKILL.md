---
name: trtllm-pr-checklist
description: >-
  Systematic pre-submission and review checklist for TensorRT-LLM pull requests.
  Use when preparing a change for review, reviewing someone else's PR, or
  revising a PR in response to reviewer / CodeRabbit feedback. Covers design fit
  and blast radius on core components, correctness across backends and
  parallelism modes, validation-versus-testing, CI registration, code quality,
  and whether the PR's claims are backed by evidence. Complements
  `trtllm-code-contribution` (which covers the implementation phase) — use this
  one at the point where human review is requested.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Pull Request Checklist

This skill enforces TensorRT-LLM's contribution standards at the review boundary,
and reduces the manual review burden on maintainers. It is the checkpoint between
"the code works" and "a human should spend time on this".

Scope boundary:

| Skill | When |
|---|---|
| `trtllm-codebase-exploration` | Before writing code — survey what already exists |
| `trtllm-code-contribution` | While writing code — implementation patterns and pitfalls |
| **`trtllm-pr-checklist`** (this) | **At review time — prepare, review, or revise a PR** |

## Modes

Select the mode from the request, then work through the sections below.

### Mode A — Author self-check (before requesting review)

Go through every section against your own diff. Fix what you can; write the rest
into the PR description as explicit caveats. Do not silently skip a section — if
it does not apply, say so and say why.

### Mode B — Reviewer

Assess someone else's PR against every section. Produce a report with a summary
header (overall assessment + the 1-3 items that most need attention), then
per-section notes. Distinguish **blocking** issues from **suggestions**. Where
evidence is missing, ask for the specific evidence rather than guessing.

### Mode C — Re-review (revising after feedback)

1. Read **all** comments and threads on the PR, including CodeRabbit's
   (`.coderabbit.yaml` configures it) and any resolved-but-unaddressed threads.
2. Triage each concrete concern into: fix in code / reply with rationale /
   out of scope, file a follow-up.
3. For each, agree the plan with your human operator before acting. Do not
   mass-apply CodeRabbit suggestions without judgment — many are noise.
4. Re-run the sections that the change touches.
5. **Refresh the PR description.** A stale description after three rounds of
   changes is one of the most common and most avoidable review frictions.

## Operating contract

This skill is designed for a human in the loop.

- Running programmatically: write notes on **every** section so the report is
  complete, with a clear header summarizing findings and areas needing attention.
- Running interactively: present the report, then a concrete list of actionable
  follow-ups and design decisions the human must make. Then iterate.
- Never mark an item as passing that you did not actually check. "Not verified"
  is a valid and useful outcome; a false pass is not.

---

## 1. Design Fit

Does the change fit TensorRT-LLM's architecture, or does it bolt onto it?

### 1.1 Blast radius on core components

These files and directories are shared by nearly every deployment. Changes here
carry high risk and must be structured to be minimally invasive:

| Area | Paths |
|---|---|
| Executor loop | `tensorrt_llm/_torch/pyexecutor/py_executor.py`, `py_executor_creator.py`, `_util.py` |
| Model engine | `tensorrt_llm/_torch/pyexecutor/model_engine.py`, `cuda_graph_runner.py` |
| Scheduling | `tensorrt_llm/_torch/pyexecutor/scheduler/`, `executor_request_queue.py` |
| Resources / KV cache | `tensorrt_llm/_torch/pyexecutor/resource_manager.py`, `pyexecutor/kv_cache/`, `seq_slot_manager.py` |
| Sampling / decoding | `tensorrt_llm/_torch/pyexecutor/sampler/` |
| Public config | `tensorrt_llm/llmapi/llm_args.py`, `llmapi/llm.py`, `llmapi/llm_utils.py` |
| Attention | `tensorrt_llm/_torch/attention/` (`attention.py`, `mla.py`, `backends/`, `kernels/`) |
| MoE | `tensorrt_llm/_torch/moe/fused_moe/` |
| Executor abstraction | `tensorrt_llm/executor/executor.py` |
| C++ core | `cpp/tensorrt_llm/batch_manager/`, `cpp/tensorrt_llm/kernels/` |

Checks:

- Could the new functionality live in a new file / subclass / backend instead of
  adding a branch to one of the above?
- How many new conditional branches does this add to a hot loop?
- Before touching anything under `tensorrt_llm/_torch/attention/`, read
  `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md`. Before touching
  `tensorrt_llm/_torch/moe/`, read
  `tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md` and use the
  `trtllm-moe-develop` skill.
- Changes under `tensorrt_llm/visual_gen/` or `_torch/visual_gen/` must satisfy
  `_torch/visual_gen/ENGINEERING_CRITERIA.md`, and `tensorrt_llm/visual_gen/` is
  a user-facing API surface — confirm a public API change is actually intended.

### 1.2 Replication

Does the change duplicate something that already exists? Run the
`trtllm-codebase-exploration` searches before accepting new code as necessary.

Replication is sometimes acceptable by design:

- **New model implementations** (`_torch/models/modeling_*.py`) — boilerplate
  repeats by convention. But layer subcomponents (attention, MoE, RMSNorm, RoPE)
  should be reused from `_torch/modules/`, `_torch/attention/`, and
  `_torch/moe/`, not re-derived.
- **New attention or MoE backends** — isolation from existing backends is often
  the point.
- **Speculative decoding mechanisms** — shared architecture discourages
  inheritance; still reuse common subcomponents.
- **Per-architecture kernel specializations** — the same algorithm specialized
  for SM90 / SM100 / SM103 legitimately repeats structure.

Outside those cases, duplication needs an explicit justification in the PR.

### 1.3 Complexity budget

Line count is a coarse flag, not a verdict — but an unexplained large diff is
itself a review finding. Exclude tests and docs from the count.

| Change type | Expected | Above this, justify explicitly |
|---|---|---|
| Bug fix | < 100 lines | > 250 lines |
| Perf optimization | < 250 lines | > 500 lines |
| New feature on existing paths | < 400 lines | > 800 lines, or consider splitting |
| New model / new backend | no useful bound | — judge by reuse instead |

Redesigning core components or changing core interfaces **in order to fix a bug**
is discouraged and needs strong justification. If a bug fix requires an
interface change, say so prominently and consider splitting the refactor out.

Also check: does this add a new abstraction layer, a new config field, or a new
env var? Each is permanent surface area. Is it earning its keep?

### 1.4 Correctness and compatibility across shared paths

TensorRT-LLM's compatibility matrix is large. Trace every axis the change
touches, and state in the PR which ones were verified versus reasoned about.

| Axis | What to check |
|---|---|
| Python / C++ boundary | `_torch/pyexecutor/` and the shared C++ core it drives through nanobind (batch manager, KV cache manager, decoder) — a contract change on either side reaches the other |
| Parallelism | TP, PP, CP, EP, and attention DP (`mapping.*_size`). CP in particular breaks "all tokens are local" assumptions |
| Attention backends | `_torch/attention/backends/` (`trtllm.py`, `flashinfer.py`, `vanilla.py`, `fmha/`, `sparse/`, `fp4_mla/`) — a change to `attention.py` reaches all of them |
| Hardware | SM89 / SM90 / SM100 / SM103 / SM120. Fused kernels differ numerically across generations (e.g. SM90 FMHA online-softmax merge vs SM100+ single pass) |
| Quantization | FP8, NVFP4, FP4 KV cache, W4A8, per-tensor vs per-block scales |
| Serving topology | aggregate and disaggregated (`_torch/disaggregation/`, NIXL/UCX/MPI) |
| Runtime modes | CUDA graph vs eager, chunked prefill, overlap scheduler, spec decode |
| Vertical | VisualGen shares ops/kernels — check if a shared op changed |

Cross-check against `docs/source/features/feature-combination-matrix.md` and
update it if the change adds or removes a supported combination.

**Trace callers, not just the changed function.** Grep every call site of a
modified function and every sibling implementation of a modified interface.

### 1.5 Tradeoffs and regression risk

Applies especially to default-value changes, backend selection changes, and
heuristic thresholds.

- If this speeds up one configuration, which configuration could it slow down?
  Was that one measured, or only assumed?
- Does a new fast path have a guard whose semantics match its cost model?
  (A threshold on per-sequence length does not bound a cost that scales with
  total packed tokens.)
- Does enabling a feature by default reduce numerical accuracy anywhere?
- Complex optimizations with 3+ guard conditions, correctness-critical paths, or
  hardware-specific behavior should ship **disabled by default** with env-var or
  config opt-in until broadly validated.

State the tradeoffs in the PR. An unstated tradeoff discovered by a reviewer
costs a round trip; discovered after merge, it costs a revert.

---

## 2. Validation and Testing

These are two different obligations. Both are required; neither substitutes for
the other.

### 2.1 Validation — did you actually run it?

Every functional change must be *validated* by running it:

| Change type | Minimum validation |
|---|---|
| Bug fix | Reproduce the bug, apply the fix, confirm it is gone. State the reproducer |
| New feature | Run it end to end. Include the invocation |
| Perf optimization | Benchmark before/after on a stated configuration |
| Refactor | Unit tests for minor refactors; E2E serving run for significant ones |
| Accuracy-affecting | `trtllm-eval` score before/after, or a published baseline comparison |

Shipping a bug fix without having run TensorRT-LLM to confirm it resolves the
issue is not acceptable. If you could not validate (no access to the hardware,
no reproducer), say so explicitly in the PR — do not leave it ambiguous.

Use `trtllm-test-specialist` / `trtllm-case-executor` to run these; use
`LLM_MODELS_ROOT` for tests needing weights.

### 2.2 Testing — what goes into the repo

Not every change needs a new test, and more tests are not better.

- Tests should target paths that are **error-prone, regression-prone, or
  complex to interact with**. Local validation covers baseline correctness.
- Recommend **deleting** redundant, trivial, or near-duplicate parametrizations.
  Over-parametrized test files are one of the most common review complaints.
  Each parametrized case should exercise a *distinct code path*.
- **Verify the test actually reaches the new code.** This is the single most
  valuable check in this section. TensorRT-LLM fails silently into fallbacks:
  - A backend-selection test must assert the intended backend was selected, not
    just that output is correct.
  - A CUDA-graph test must assert the graph was captured/replayed, not padded out.
  - A quantization test must assert the quantized kernel ran, not a dequantized
    reference path.
  - A spec-decode correctness test must assert a **nonzero acceptance rate** —
    unless the test's purpose is specifically to verify all-draft-rejection, in
    which case assert zero and say so.
  - A feature-flag test (chunked prefill, overlap scheduler, piecewise graphs)
    must assert the feature actually engaged for the batch under test, not
    merely that the flag was accepted by the config.

### 2.3 Test reliability

Analyze new tests for flakiness before CI finds it for you. Frequent sources in
this repo:

- **Cross-hardware tolerances** — bit-exact or tight-tolerance assertions break
  when kernel selection changes across SM versions. Gate with `get_sm_version()`
  or widen tolerance deliberately, with a comment on why that tolerance.
- **Multi-GPU / multi-node** — NCCL init, MPI rank setup, and timeouts. Prefer
  existing fixtures in `tests/unittest/_torch/` over hand-rolled setup.
- **Background threads** — cache managers spawn `ThreadPoolExecutor` threads
  that outlive tests; add `pytestmark = pytest.mark.threadleak(enabled=False)`.
- **Shared state / ordering** — GPU memory not released between parametrizations,
  module-level caches, env vars set by one test leaking into another.
- **Model download / `LLM_MODELS_ROOT`** — mark appropriately so the test is
  skipped rather than failing where weights are unavailable.

### 2.4 CI registration — will the test actually run?

A test that CI never executes provides zero protection.

- Register integration tests in the right per-GPU list under
  `tests/integration/test_lists/test-db/` (`l0_h100.yml`, `l0_b200.yml`,
  `l0_dgx_b200.yml`, `l0_gb200_multi_gpus.yml`, ...). Pick the list matching the
  hardware the test actually needs.
- Confirm which stage will pick it up:
  `python scripts/test_to_stage_mapping.py --tests "<test_name>"`
- Unit tests under `tests/unittest/` run in pre-merge CI automatically.
- If you must waive a known failure, add it to
  `tests/integration/test_lists/waives.txt` **with an NVBug link** — never a
  bare waive.
- Trigger CI with `/bot run`. Add `--extra-stage "<stage>, <stage>"` when the
  change needs a stage outside the default pipeline. Use `--disable-fail-fast`
  only when you actually need all stages despite an early failure. `/bot help`
  lists current commands.
- Check the test's resource needs are actually available in the target stage
  (GPU count, node count, memory).

---

## 3. Code Quality and Style

`CODING_GUIDELINES.md` is authoritative — this section covers what reviewers
most often have to raise anyway.

### 3.1 Comments and docstrings

- Do not over-comment. Prefer self-explanatory code and precise names.
- Comment the *why* of non-obvious logic, not the *what* of obvious logic.
- Design-decision narrative belongs in the **PR description**, not in code
  comments.
- Docstrings: short private helpers usually do not need one. Public APIs and
  custom kernels do — for kernels, document input/output **shapes, dtypes,
  layouts, and assumptions**, which are otherwise unrecoverable from the code.
- **Stale comments are a defect.** If a semantic changed, grep for every comment,
  docstring, variable name, and test description that referenced the old meaning
  and update them in the same commit.

### 3.2 Helpers and shared logic

- A helper called from exactly one site and doing something trivial should be
  inlined.
- A condition or expression repeated at two or more sites should be extracted —
  in the same commit, not a follow-up.

### 3.3 Repository-specific hard requirements

These fail CI or review if missed:

- **NVIDIA copyright header** on all new files; update the year on modified files.
- **Pydantic discipline** for user-facing config classes (`BaseLlmArgs` and
  anything used in its fields) — see `CODING_GUIDELINES.md` "Pydantic Guidelines".
  `StrictBaseModel`, no custom `__init__`.
- **LLM args / nested config changes** must run
  `python3 scripts/generate_llm_args_golden_manifest.py` and commit
  `tensorrt_llm/usage/llm_args_golden_manifest.json`. New fields require
  telemetry/privacy CODEOWNER approval.
- **API stability** — `tests/unittest/api_stability` protects committed LLM API
  signatures. If reference files change, classify with the `api-compatible` or
  `api-breaking` label; `api-breaking` also requires `BREAKING` in the title.
  Request review from the API code owners. See
  `docs/source/developer-guide/api-change.md`.
- **Narrow exception handling** — no bare `except:`.
- **pre-commit** — run `pre-commit run --all-files` before committing. Hooks
  rewrite files in place; re-stage and commit again. Respect the Group A / Group
  B legacy-file rules in `CODING_GUIDELINES.md` — do not mass-reformat Group B
  files as a side effect.
- **DCO** — every commit signed off via `git commit -s`; let git generate the
  line. Do not attribute AI tools in the sign-off, and do not add co-authors
  unless explicitly instructed.

### 3.4 Documentation and examples

Update alongside the code, in the same PR:

- `docs/source/models/supported-models.md` for new models
- `examples/` for new user-facing capabilities
- `docs/source/features/feature-combination-matrix.md` for compatibility changes
- `docs/source/features/` for feature behavior changes
- `.github/tava_architecture_diagram.md` for significant design changes
- `AGENTS.md` if workflows, commands, or conventions changed
- `.github/CODEOWNERS` if ownership changed
- For CLI docs, prefer `--config` over `--extra_llm_api_options`

---

## 4. Pull Request Contents

This section is about the PR itself, not the code. As author, produce these; as
reviewer, assess them and name what is missing.

### 4.1 Title

Format (enforced in CI by the `Check PR Title Format` job in
`.github/workflows/pr-check.yml`; see also `CONTRIBUTING.md` and
`.github/pull_request_template.md`):

```
[JIRA ticket / NVBugs ID / GitHub issue / None][type] Summary
```

- Ticket: `[TRTLLM-1234]`, `[https://nvbugs/1234567]`, `[#1234]`, or `[None]`
- Type (lowercase): `fix`, `feat`, `perf`, `doc`, `infra`, `chore`, `test`, ...
- Examples: `[TRTLLM-5516][perf] Optimize CUDA graph padding`,
  `[https://nvbugs/5334370][fix] Fix one-model EAGLE3`,
  `[None][chore] Minor clean-up`

A title that does not match is rejected by CI before review starts.

### 4.2 Description and context

- A 1-2 line summary a reviewer can read first and understand the change from.
- Link the issue, the NVBug, and any predecessor / blocking / follow-up PRs.
- If the PR is large, explain why it cannot be split.
- **One concern per PR.** Unrelated fixes belong in separate PRs with stated
  dependencies.
- No commented-out code, no leftover debug prints, no unrelated file churn.

### 4.3 Claims and supporting evidence

State plainly what the PR accomplishes, then back each claim.

| Claim type | Required evidence |
|---|---|
| Bug fix | Root cause + reproducer + confirmation the reproducer now passes |
| New feature | A working usage example that was actually run |
| Refactor | Motivation + validation at a level matching the scope |
| Perf | Config-complete before/after numbers (below) |
| Accuracy | Eval task, dataset, and score before/after |

A performance claim is not reviewable unless it states **all** of:

- GPU type and count (and node count, if multi-node)
- Model and checkpoint, including quantization
- Parallelism: TP / PP / EP / DP
- ISL / OSL and concurrency (or the dataset used)
- Aggregate vs disaggregated
- The tool and command (`trtllm-bench` / `trtllm-serve` + load generator)
- The metric: throughput, TTFT, TPOT, or e2e latency — and which one improved

Consider `examples/configs/database/` for a pareto-optimized baseline
configuration rather than a hand-tuned strawman comparison.

Microbenchmark-only evidence is usually insufficient for a perf claim — a kernel
speedup that does not move an end-to-end serving metric is not yet a user-visible
win. If end-to-end validation was not possible (e.g. no access to the hardware),
state that explicitly rather than omitting it.

### 4.4 Root-cause analysis

For bug fixes, explain what actually caused the problem, how it was found, and
why this fix addresses the cause rather than the symptom. A patch that makes a
symptom disappear without a root cause is a future recurrence. Where an NVBug
exists, keep its RCCA fields consistent with the PR.

### 4.5 Implementation details and open questions

Call out design decisions, rejected alternatives, known limitations, and the
specific places where you want maintainer attention. Flag risks explicitly —
reviewers should not have to discover them.

### 4.6 Process

- Branch pushed to your fork; PR opened against `NVIDIA/TensorRT-LLM`.
- Target `main`, unless fixing a release-branch bug — for NVIDIA developers,
  submit to the branch named in the NVBug **Keywords** field.
- Add the `release blocker` label if this could delay a release.
- Auto-assigned reviewers are appropriate for the areas touched.
- New dependencies scanned for license and vulnerabilities.
- **Every checkbox in the PR description must be checked before merge.** The
  `Check PR Checklist Resolution` job in `.github/workflows/pr-check.yml` fails
  the PR if *any* `- [ ]` remains unchecked anywhere in the body — including
  checkboxes you added yourself outside the template. Two consequences:
  - Write follow-up items as plain bullets, not checkboxes, or they block merge.
  - To mark a template item as not applicable, strike through the **entire**
    item text: `- [ ] ~~Not applicable: no API change in this PR~~`. A partial
    strikethrough does not count as resolved.
- **AI-assisted contributions** follow the existing `AGENTS.md` rules: the
  DCO sign-off line must never attribute an AI tool, and no co-authors are added
  unless the human author explicitly asks for them. TensorRT-LLM does not
  currently require a separate AI-assistance statement in the PR description, so
  do not add one unprompted. What does not change either way: the human submitter
  is responsible for having reviewed and understood every changed line, and the
  evidence requirements in §4.3 apply identically regardless of how the code was
  produced.

---

## Summary checklist

Condensed form, for working through locally or pasting into a **review comment**.

Do **not** paste this into the PR description as-is: CI fails the PR if the body
contains any unchecked checkbox (see §4.6).

**Design**
- [ ] Core-component blast radius minimized; new logic isolated where possible
- [ ] No unjustified duplication of existing functionality
- [ ] Diff size appropriate for the change type, or justified
- [ ] Python/C++ boundary, parallelism (TP/PP/CP/EP/DP), SM versions,
      quantization, and agg/disagg paths traced — verified vs reasoned stated
- [ ] Tradeoffs and regression risks stated; complex optimizations opt-in

**Validation & Testing**
- [ ] Change was actually run; reproducer / command / benchmark stated
- [ ] Tests target error-prone paths; redundant parametrizations removed
- [ ] Tests provably reach the new path (no silent fallback)
- [ ] Flakiness considered (tolerances, multi-GPU, threads, shared state)
- [ ] Registered in the correct `test-db` list; stage confirmed via
      `test_to_stage_mapping.py`; any waive has an NVBug link

**Code Quality**
- [ ] Comments justified and current; no stale semantics
- [ ] Copyright headers; Pydantic rules; golden manifest regenerated if needed
- [ ] API stability handled + correct label if reference files changed
- [ ] `pre-commit run --all-files` clean; no Group B collateral reformatting
- [ ] Docs, examples, compatibility matrix, CODEOWNERS updated as needed
- [ ] All commits DCO signed off; no fix-on-fix chains

**PR Contents**
- [ ] Title matches `[ticket][type] Summary`
- [ ] Summary + links; single concern; large size justified
- [ ] Every claim has matching evidence; perf claims are config-complete
- [ ] Root cause explained for bug fixes
- [ ] Limitations, risks, and open questions called out
