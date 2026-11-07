<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engineering Rules for KV Cache Manager V2 and KV Cache Compression Work

Two entry skills share these rules:
- `trtllm-kv-cache-manager-v2` (`../SKILL.md`), for KV cache manager V2
  development;
- `trtllm-kv-cache-compression`
  (`../../trtllm-kv-cache-compression/SKILL.md`), for KV cache compression
  framework development.

The repository docs and the code are the source of truth for what the code
does. This file is the workflow on top of them: how to design and gate a
change, and the rules that reviews keep enforcing. Paths are relative to this
file.

These failure patterns recur in AI-assisted work on KV cache manager V2 and
the KV cache compression framework. The first one matters most; every
section below exists to prevent one of them:

1. **The feature does not work.** A stub or placeholder, a path that is
   never wired into the executor, a demonstration on one model presented as
   general support, or "done" without an end-to-end run.
2. **Defensive, conservative, and messy code.** Try/except around code that
   should not fail, fallbacks, checks for states that cannot happen, options
   nobody asked for, and flags, wrappers, or parallel paths added to avoid
   changing the code that owns the behavior; dead paths and jargon.
3. **Test bloat and test gaming.** Many tests that mock the feature away,
   compare the code with itself, or can never fail; tests edited, skipped,
   or loosened until they pass.
4. **Ignoring the existing framework.** A new mechanism is built where
   configuration or an existing extension point would have worked.
5. **Non-compliant code.** Repository rules for config changes, tests, docs,
   and commits are skipped.
6. **No performance check.** A feature is called done without an end-to-end
   comparison in the production configuration.

## Design Workflow

1. **Behavior contract.** What changes for the user, and on which model
   families. The goal is every family KV cache manager V2 serves (see "The
   Goal" in `design-principles.md`). Also state what stays identical when
   the feature is off: everything.
2. **Survey.** Spend a short time finding mechanisms that already do part of
   the job: life cycles (including sliding-window ones), retention
   priorities, reuse policies, extra buffers, the KV cache compression
   manager's lifecycle hooks, cold-page codecs, sparse-attention hooks, the
   attention metadata's mixed-precision staging copy, transceiver layout
   resolution. Use what fits. The survey shapes the design; it does not stop
   new code that the feature needs. See `design-principles.md`, section 2.
3. **Placement.** Put each change where the behavior belongs: configuration,
   a field, an existing extension point, the feature's own package, shared
   Python runtime code, or native code. Prefer a higher level when it can
   express the feature, and go lower with a written reason when it cannot.
   For changes to shared Python runtime code or native code, list every
   function you will change, by layer.
4. **Owner boundaries.** Place each responsibility in a row of the owner map.
5. **Compatibility.** Decide each axis in
   `../../trtllm-kv-cache-compat-testing/SKILL.md` as supported, rejected, or
   n/a.
6. **Performance budget.** List the per-step costs the feature adds and the
   expected benefit, and pick the baselines
   (`../../perf-kv-cache-tuning/SKILL.md`).
7. **Validation plan.** Follow the order in the compression development
   guide (`docs/source/developer-guide/kv-cache-compression-development.md`):
   1. the accuracy or output-quality contract against the uncompressed path;
   2. real end-to-end activation on every claimed model family, and the
      claimed compatibility combinations;
   3. end-to-end performance;
   4. automated regression tests: small functional pytest checks (the path
      runs, bookkeeping matches exactly, unsupported combinations are
      rejected, the feature activated, exact operations are byte-exact,
      layout invariants hold) and a few integration smoke tests that only
      prove activation.

   Stages 1 to 3 are local real runs, and so are the encode/decode
   round-trip accuracy and the end-to-end accuracy and performance checks
   that the guide lists under stage 4. Write their commands and results in
   the PR description, not in pytest. Each stage is its own deliverable,
   with its own evidence.
8. **User-facing surface.** Docs, an example, manifest and schema
   regeneration for config changes, API-stability references.

Write the gate before editing:

```markdown
## Design Gate
- Behavior contract: <what changes; "feature off = identical to main">
- Model families: <family -> supported (end-to-end run) | rejected (where,
  why, follow-up)>
- Existing mechanisms surveyed: <mechanism -> reused | why not>
- Placement: <config | field | extension point | feature package |
  shared Python runtime code | native>; reason for going lower: <...>
- Shared functions touched (layer x function): <table, or "none">
- Owner placement: <responsibility -> owner-map row>
- Compatibility table: <axis -> supported / rejected (where) / n/a>
- Per-step cost and expected benefit: <...>; baselines: <...>
- Validation plan: <local real runs, reported in the PR description:
  accuracy contract / end-to-end activation and compatibility / end-to-end
  performance; regression tests: functional pytest / activation smoke tests>
- Names: <the full name of each new concept, used unchanged in code,
  tests, docs, and the PR>
- Docs and config artifacts to update: <...>
```

If a reviewer can name a model family or a compatibility axis with no
decision, or an existing mechanism the design duplicates, the gate is not
passed.

### When the Framework Is Not Enough

Use existing mechanisms where they fit, and build what is missing where they
do not. When a feature needs something the framework cannot express:
1. prototype it in your own package and measure it;
2. write down why the existing extension points do not fit;
3. propose an algorithm-neutral extension point (hook, capability, buffer role,
   life-cycle property, codec interface) that defaults to today's behavior;
4. get the owners' decision;
5. land the extension point with unchanged-behavior tests, then the method
   on top.

Section 11 of `design-principles.md` has the criteria and examples from
main. Kernels follow
`../../trtllm-kv-cache-compression/references/adding-kernels.md`.

## Engineering Rules

Details and rationale are in `design-principles.md`. The incidents
behind them are in `lessons-learned.md`.

- **The feature first.** The deliverable is a working feature. Wire the whole
  path first (configuration, KV cache manager V2, attention metadata,
  executor callbacks, outputs) and run it end to end on a real model in the
  production configuration. Then refine.
- **Every model family.** Make the feature work on every model family KV
  cache manager V2 serves: dense models with multi-head or grouped-query
  attention, multi-head latent attention (MLA) models, sparse-attention
  models such as DeepSeek-V4, sliding-window and hybrid models, hybrid
  recurrent models, multimodal models, and speculative-decoding draft
  layers. A family that cannot be supported yet is rejected at configuration
  time and listed as a gap with its reason, never silently skipped. Cover
  every GPU architecture the feature claims the same way, for example Ampere
  (SM80, SM86), Hopper (SM90: H100, H200), and Blackwell (SM100: B200,
  GB200; SM103: B300, GB300; SM120, SM121), since kernels, shared-memory
  limits, and numerics can differ between them.
- **Fit, do not force.** Use configuration and existing extension points when
  they fit the feature. When they do not, change the code that owns the
  behavior or add what is missing. Do not bend the feature to fit an
  existing mechanism, and do not add a flag, a wrapper, a new abstraction
  layer, or a parallel path to avoid touching the right file. Do not route
  around an owner's component with a fallback.
- **Keep method logic out of shared code.** The runtime, KV cache manager
  V2, attention kernels, and C++ never branch on a method's or model's name.
  Shared changes extend an algorithm-neutral contract.
- **Feature-off invariance.** With the feature disabled, outputs, memory, speed,
  errors, and API are identical to main. Gate every new path on the feature's
  configuration. A bookkeeping fix that would change disabled runs is
  proposed separately.
- **Fail fast.** Reject unsupported combinations at configuration time, or at
  executor construction before KV cache manager V2 allocation when the check
  needs the device or the engine. No version sniffing, no try/except-import
  fallbacks, no half-applied state. Load offline artifacts at
  initialization, never calibrate during inference.
- **Direct code, not defensive code.** Write the code the feature needs and
  nothing "just in case": no try/except around code that should not fail,
  no fallback that hides an error or changes results, no `getattr` or
  `hasattr` defaults for attributes that always exist, no checks for states
  that validation already rules out, and no options nobody asked for. Every
  guard names the real failure it handles. A fallback that gives the same
  results under a resource limit, is counted or logged, and is tested is
  fine: for example, TriAttention scores requests one at a time when its
  optional batch workspace does not fit in memory.
- **Few, strong tests.** Each test exercises the real path and fails if the
  feature breaks. Do not write tests that mock the feature away, compute
  their expected values with the code under test, check trivial getters, or
  repeat another test with different numbers. Never edit, skip, delete, or
  loosen a test, or special-case test inputs, to make it pass: fix the code,
  or stop and report the conflict. When an approved behavior change makes an
  expectation obsolete, update the test and say why in the PR. Keep pytest
  small, with few GPU tests; check lossy numerics and end-to-end behavior
  locally on real runs, and write the commands and results in the PR
  description.
- **Preserve algorithm semantics.** Do not change selection rules or
  granularity, and do not remove modes for convenience. Parallel runs must
  match single-GPU semantics.
- **Hot-path discipline.** No per-step host loops, synchronizations,
  allocations, or blanket memsets. Allocate once, update in place, and keep
  CUDA-graph inputs static.
- **Names and text.** Name things by behavior, not mechanism. Give each
  concept one name, preferably the one the code or the public docs already
  use, and write it in full everywhere: identifiers, test names and
  parameter IDs, comments, log and error messages, docs, PR text, and
  replies to people. Never cut it to a fragment. The mixed-precision high
  window is `mixed_precision_high_window` in code and "mixed-precision high
  window" in text, never `window`; a test of the FP8 KV cache says "FP8 KV
  cache" (`fp8_kv_cache`), never just "FP8" or "KV". Use an abbreviation only
  if it is common knowledge (LLM, GPU, CUDA, `attn`) or after writing "full
  name (ABBR)" at its first use in each document. Comments explain the code
  as it stands; history goes in the PR. Use no internal jargon, codenames,
  version labels, or milestone names anywhere.
- **Code size is a consideration, not a target.** A working feature comes
  first. While adding it, check whether an existing mechanism can carry it,
  or whether it can merge
  into the existing framework, instead of writing a new standalone
  implementation. Add everything the requirement needs. Never remove
  features, modes, model support, or optimizations to make a change smaller.
  Read a large diff as a sign that the design may duplicate something, and
  test the reuse decision.
- **On-topic diffs.** One concern per PR. Measure the diff after each fix
  round. Delete only what the change itself makes obsolete.

## What Machines Check, and What Only Review Catches

| Rule | Enforced by | Reviewer action |
|---|---|---|
| Formatting, lint, spelling | pre-commit (ruff, yapf, clang-format, codespell, and others) | None if the hooks ran on commit |
| Developer Certificate of Origin (DCO) sign-off | commit-msg hook (`scripts/dco_check.py`) | Use `git commit -s`; no AI attribution lines |
| New config fields have descriptions | `tests/unittest/llmapi/test_llm_args.py` | None |
| LLM-args golden manifest is current | `tests/unittest/usage/test_llmapi_config_telemetry_docs.py` | Regenerate with `scripts/generate_llm_args_golden_manifest.py`, and review the diff |
| Serve schema snapshots are current | `scripts/generate_trtllm_serve_schemas.py --check`, CPU schema tests | Regenerate; never hand-edit |
| Public API signatures | `tests/unittest/api_stability/` | Update the references deliberately |
| Test-list entries exist | pre-commit `scripts/check_test_list.py --validate` | Also check the test is on a stage with the right GPU |
| No raw pinned-memory calls in runtime code | pre-commit `scripts/check_pinned_memory_usage.py` | Use `maybe_pin_memory()` or `prefer_pinned()` from `tensorrt_llm/_utils.py` |
| NVIDIA copyright header on new files, year updated on modified files | Nothing | Check every new and touched file (`CODING_GUIDELINES.md`) |
| User-facing config classes follow the Pydantic rules | Partly (field descriptions are tested) | `StrictBaseModel`, `Field(description=...)`, `Literal` for closed sets, validators raising `ValueError`, `status="prototype"` on new experimental settings (`CODING_GUIDELINES.md`) |
| Feature-off invariance | Nothing | Read every new branch; run touched suites with the feature off |
| Reuse of existing mechanisms; no new layers | Nothing | Compare with the design gate's survey |
| No method or model branches in shared code or C++ | Nothing | Grep the diff outside the feature package |
| No defensive code: silent fallbacks, version sniffing, checks for impossible states | Nothing | Read every `try`/`except`, `getattr` or `hasattr` default, `isinstance` check, and capability check the change adds; each must name a real failure |
| Per-step host work, syncs, memsets | Nothing | Read the call sites and their guards |
| The feature works end to end on every claimed model family | Nothing | Ask for the end-to-end run per family, and for the rejection and its reason for every other family |
| Tests fail when the feature breaks | Nothing | Break the feature on purpose and confirm the new tests fail. Reject mock-only and self-comparing tests, and any test that was edited, skipped, or loosened to pass |
| Compatibility decided and tested | Nothing | Check the compatibility table against the tests |
| Activation proven, performance and accuracy measured | Nothing | Ask for the evidence (`../../perf-kv-cache-tuning/SKILL.md`, `../../trtllm-kv-cache-accuracy-eval/SKILL.md`) |
| One full name per concept; abbreviations defined at first use | Nothing | Read new identifiers, test IDs, comments, and docs for fragments and undefined abbreviations |

## Agent Working Rules

- **Finish the feature before polishing it.** Get the full path working end
  to end first; refactors, extra options, and cleanups come after. A change
  called done has no stubs, placeholders, or "TODO: implement".
- **Read, don't infer.** Before answering "does X happen once or every
  step", "which engine or backend runs this", or "is this our bug", read the
  call site and its guard, or run it. Never rely on a function name, a
  docstring, or a README layout.
- **Verify behavior, not labels.** A green test or a 100% score can mean the
  path never ran. Design a check that must change if the mechanism works,
  and show it.
- **Honest status.** Say "submitted", "running", or "passed" only with
  evidence of that state. "Done" means the user's goal is met and checked,
  not that files were edited. Keep committed `HEAD` separate from local
  uncommitted changes in every report.
- **Do not change the experiment to fix the result.** Do not change benchmark
  settings, test configurations, page sizes, pool ratios, reuse policies, or
  thresholds to make something pass. Propose the deviation and ask first.
  Results from a deviated setup are reference only.
- **Do not shrink scope.** Run the full requested matrix and the full dataset.
  If something blocks, say which command failed and which decision is
  missing.
- **Stay in scope.** Do not refactor, reformat, or "improve" code the task did
  not ask for. Do not touch another component's files without saying why.
- **Match references exactly.** When asked to align with a reference
  implementation, use its exact option names, accepted values, precedence,
  and defaults. Report any unavoidable difference.
- **Root-cause before fixing.** Reproduce, locate the first point where the
  behavior diverges, and fix the cause. A retry, a wider tolerance, or a waive
  is not a fix.
- **Do not game tests.** When a test fails, fix the code. When a test
  contradicts the requirement, stop and report it. Never edit, skip, delete,
  or loosen the test, and never special-case its inputs. Updating a test
  because an approved behavior change made its expectation obsolete is not
  gaming it; say why in the PR.
- **Decide open design choices before building.** For each:
  - inventory the existing code;
  - present the options with scope, benefit, and impact on other owners,
    including "change nothing";
  - get a decision before implementing;
  - record the rejected options and why.

  Do not implement a choice that belongs to the user or an owner and present
  it afterwards.
- **Answer the question that was asked, first.** Put the answer before any
  progress narrative. If the question cannot be answered yet, say what is
  missing.
- **Talk in the same full names.** In answers, reports, and review replies,
  use the full names the code and docs use, define an abbreviation at its
  first use, and never use a label you coined during the work. If the reader
  would have to ask what a word means, replace it.
- **Test the code you ship.** Use the latest code on a matching build. Run
  long experiments from a frozen snapshot of code and wheel, not from a
  branch that is still changing.
- **Triage CI failures before touching code.** Find the common failure
  point. Check whether main and unrelated tests fail the same way, and look
  for an existing fix or waive. Change code only for failures the change
  caused.

## PR and Review Hygiene

- **Scope.** Keep the diff in scope: no unrelated files, renames, re-wraps,
  or constant changes. A rename in a shared file pulls in another owner's
  review for nothing.
- **Split by owner and risk.**
  - Land shared-runtime integration separately from the method.
  - Ship the basic version for standard models first.
  - Keep stacked PRs strictly additive over their base.
- **Rebase.** Rebase on the latest base, and resolve only the conflicts.
  Drop local fixes that the base already contains. On a shared PR, merge main
  instead of rebasing or force-pushing.
- **Debugging must not grow the PR.** Measure the diff against main after
  every round of fixes, and report it. Fix crashes and wrong results, not
  hypothetical robustness, and simplify after fixing.
- **CI timing.** Do not push while the PR's CI is running or about to rerun.
  A new head commit restarts the whole pipeline. Documentation-only and
  waive changes can wait until CI passes.
- **Clean branch.** Debug probes, extra NVIDIA Tools Extension (NVTX)
  ranges, micro-benchmarks, and design notes stay out of the PR branch.
- **Order of work.** Before pushing a feature change, run the required hooks
  and the end-to-end gate (`../../trtllm-kv-cache-compat-testing/SKILL.md`). For
  documentation-only changes and small agreed review fixes, the hooks are
  enough: push, then run the remaining checks and report them as pending.
  Never call a change done before the checks that matter pass, such as CUDA
  graphs on and the feature activated.
- **Description.** Use the title format from the root `AGENTS.md`:
  `[JIRA/NVBUG/None][type] description`, for example `[None][feat] ...`. It
  takes precedence over any other title format.
  Give:
  - what the change enables;
  - the flow;
  - a short module table;
  - one quantified result;
  - the commands and results of the local lossy-numeric, accuracy,
    end-to-end, and performance runs;
  - explicit scope limits.

  Keep claims conservative and precise; unvalidated paths are called
  unvalidated.
- **Review comments.** Handle them one at a time: quote, analyze, agree on
  the fix, then edit. Apply the agreed fix to every peer call site. Keep
  replies short and on the reviewer's actual concern; leave out internal
  status.
- **Owner bugs.** When another owner's bug blocks you:
  - prove it is real and not your setup;
  - document its trigger and origin;
  - fix it in its own commit or PR, or skip it and continue.

## Review Output

```markdown
## Findings
- [High] <file:line> <correctness, ownership, or invariance issue>
- [Medium] <file:line> <reuse, compatibility, or performance issue>
- [Low] <file:line> <naming, comment, or diff hygiene>

## Design Fit
- Existing mechanisms reused: <yes/no; which were missed>
- Extension level justified: <yes/no>
- Owner boundaries respected: <yes/no; which owner-map row>
- Feature-off invariance: <checked how>

## Evidence
- Compatibility: <axis -> test | rejection>
- Activation: <how shown>
- Performance: <comparison summary or "missing">
- Accuracy: <summary or "n/a: lossless" or "missing">
```

If there are no findings, say so and list the remaining test, performance, or
accuracy risk.

## Related Repository Skills

Where they are installed:
- `trtllm-codebase-exploration`: run it before the design survey to find
  existing infrastructure;
- `trtllm-code-contribution`: fork workflow, sign-off, and PR conventions;
- `kernel-triton-writing` and `kernel-cute-writing`: method and conversion
  kernels;
- the `perf-*` profiling skills, used through `perf-kv-cache-tuning`.

## References

- `design-principles.md`: the review bar, with rationale.
- `lessons-learned.md`: recurring review findings and the rules they
  produced, by category.
- `kv-cache-architecture.md`: a map of the stack, with the invariants changes
  most often break.
- `../../trtllm-kv-cache-compression/references/adding-kernels.md`: choosing,
  writing, wiring, and testing kernels (Triton, CuTe DSL, CUDA C++ with torch
  custom ops or nanobind).
