---
name: trtllm-kv-cache-develop
description: >-
  Entry point for designing, implementing, debugging, reviewing, or
  refactoring KV-cache features in the TensorRT-LLM PyTorch backend: KV Cache
  Manager V2 (KVCM V2) layer groups, pools, life cycles, block and partial
  reuse, reuse policies and retention, the KV cache compression framework,
  cold-page compression, mixed-precision KV cache, and sparse or evicting
  attention that changes KV state, plus the executor, attention-metadata, and
  disaggregated-transfer code they touch. Use before writing or reviewing any
  such change. Gives the required reading, owner boundaries, design workflow
  and gate, the engineering rules that reviews keep enforcing, agent working
  rules, and how to add kernels, and routes to the KVCM V2, method-specific,
  compatibility, performance, and accuracy skills.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM KV-Cache Development

The repository docs and the code are the source of truth for what the
KV-cache stack does. This skill is the workflow on top of them:
- what to read;
- which boundaries a change must respect;
- how to design and gate a change;
- the rules that reviews of KV-cache work keep enforcing.

It is the entry point. The other KV-cache skills assume it. In these skills,
"KVCM" means KV Cache Manager V2 unless V1 is named.

Four failure patterns recur in AI-assisted KV-cache work. Every section below
exists to prevent one of them:

1. **Ignoring the existing framework.** A new mechanism is built where
   configuration or an existing extension point would have worked.
2. **Non-compliant code.** Repository rules for config changes, tests, docs,
   and commits are skipped.
3. **Messy code.** New abstraction layers, dead paths, silent fallbacks, and
   jargon pile up.
4. **No performance check.** A feature is called done without an end-to-end
   A/B in the production configuration.

## Route the Task

| Task | Load next |
|---|---|
| Add or change a compression method: iteration-driven (eviction, compaction) or storage-bound | `trtllm-kv-cache-compression` |
| Cold-page codecs, the NVFP4 cold format, RoPE precision for cold pages | `trtllm-kv-cache-cold-page` |
| The mixed-precision KV cache (`mixed_nvfp4_*` dtypes, first/last N tokens) | `trtllm-kv-cache-mixed-precision` |
| Whether a feature works with reuse, CUDA graphs, overlap, disaggregated serving, speculative decoding, or parallelism; test design | `trtllm-kv-cache-compat-testing` |
| Throughput, latency, memory, profiling, tuning | `perf-kv-cache-tuning` |
| Accuracy of a lossy or state-changing feature | `trtllm-kv-cache-accuracy-eval` |
| Change KVCM V2 itself: life cycles, reuse or retention rules, buffers, pool sizing, tiers, or its C++ core | `trtllm-kv-cache-manager-v2` |
| A sparse attention algorithm that selects or skips KV inside attention (`SparseAttentionConfig`) | `docs/source/developer-guide/sparse-attention-development-guide.md`, then the compatibility, performance, and accuracy skills above |

## Required Context

Read before proposing a design or editing code:

1. `CODING_GUIDELINES.md` and the root `AGENTS.md`.
2. `docs/source/features/kvcache.md`: manager selection, reuse, partial
   reuse, retention, mixed precision, offloading.
3. `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md`, at least
   "KV-cache and decode-time semantics", "Evaluating New Attention", and
   "Anti-Patterns".
4. Before touching the KVCM V2 core, `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`
   and `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md`. The
   latter covers ownership, invariants, the concurrency model, and high-risk
   areas.
5. `docs/source/features/feature-combination-matrix.md`.
6. The feature's own docs:
   - `docs/source/features/kv-cache-compression.md`;
   - `docs/source/developer-guide/kv-cache-compression-development.md`;
   - `docs/source/developer-guide/kv-cache-cold-page-codec.md`;
   - `docs/source/features/sparse-attention.md`;
   - `docs/source/developer-guide/sparse-attention-development-guide.md`;
   - `docs/source/features/disagg-serving.md`;
   - `docs/source/developer-guide/kv-transfer.md`.
7. The target code, its tests, and its recent history (`git log -- <files>`):
   who changed it, and why.

When docs and code disagree, the code and its tests win. Say so in the change,
and either fix the doc in the same PR or report it as a follow-up. Known
stale spots to read past:
- some docs still mention a pure-Python KVCM V2 backend, or ask for tests on
  "both backends". There is one backend, the C++ core;
- the sparse-attention development guide still offers KV cache manager V1
  for auxiliary pools. Put side buffers on KVCM V2 through
  `_extra_buffers_per_layer` instead;
- the root `AGENTS.md` lists NIXL, UCX, or MPI for KV exchange, and
  `kv-transfer.md` says a V1 context worker can talk to a V2 generation
  worker. Both predate the deprecation: the current transceiver uses NIXL,
  with KVCM V2 on both sides;
- the opening sections of `kvcache.md` ("The Basics", "Reuse Across
  Requests") describe V1 pools. V2 sizes one pool per layer group, from
  `pool_ratio` or `avg_seq_len`;
- the "Multi-backend Support" overview in `disagg-serving.md` describes the
  deprecated C++ transceiver. Use its transceiver configuration section
  instead;
- the "Required KVCM Refactoring" section of the cold-page codec design is
  already implemented. Read it as invariants, not as open work;
- the note in `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md` that
  streaming KV events have no implementation predates the native streaming
  path. The "KV Cache Events" section of `kvcache.md` is current;
- line anchors in GitHub links can drift. Locate symbols by name.

## Build on Current Paths Only

| Build on | Do not build on (deprecated, removed, or legacy) |
|---|---|
| KV Cache Manager V2 (`use_kv_cache_manager_v2: true`, the default) | KV cache manager V1. It is deprecated and scheduled for removal, together with its V1-based subclasses, such as the hybrid managers in `kv_cache/mamba_cache_manager/legacy.py`. A few features still depend on V1 while they gain V2 support; new features target V2 and do not reuse V1-specific code paths |
| The Python cache transceiver (`transceiver_runtime: PYTHON`) on the NIXL backend, with KVCM V2 on both workers | The deprecated C++ cache transceiver, which is selected by `transceiver_runtime: CPP`, by an explicit null, or by `auto` when it falls back. Also the UCX, MPI and MOONCAKE transceiver backends, `backend: DEFAULT`, and the `TRTLLM_USE_*_KVCACHE` backend selectors. New features target the Python transceiver |
| Explicit settings in recipes and tests (`use_kv_cache_manager_v2: true`, `transceiver_runtime: PYTHON`) | `auto` values that can resolve to a deprecated path, and example or curated configurations that pin `use_kv_cache_manager_v2: false` |
| Current speculative-decoding configurations (MTP, EAGLE3, and the other one-model modes) | The deprecated `NGramDecodingConfig` and `UserProvidedDecodingConfig` |
| The C++ KVCM V2 core, driven through `tensorrt_llm/_torch/pyexecutor/kv_cache/` | A pure-Python KVCM V2 core. Main has none, and no code reads the `TLLM_KV_CACHE_MANAGER_V2_BACKEND` variable that some docs and tests still mention |
| `tensorrt_llm/_torch/attention/` (`backends/`, `attention.py`, `ATTENTION_DEVELOPER_GUIDE.md`) | The deprecated compatibility shims `tensorrt_llm._torch.attention_backend` and `tensorrt_llm._torch.modules.attention` |
| `tensorrt_llm/_torch/pyexecutor/kv_cache/` and `tensorrt_llm/_torch/disaggregation/` | Flat module paths from before the KV cache managers and the transceiver were regrouped |
| The KV cache compression framework (`KvCacheCompressionConfig`) for work between steps or at tier boundaries | Sparse-attention hooks for KV reduction that does not change the attention computation |
| Current field names: `cold_tier_offload_min_priority`, `block_reuse_config.policy`, `block_reuse_config.enable_branch_snapshot`, `mamba_state_config.periodic_snapshot_interval` | Old names: `secondary_offload_min_priority` (alias of `cold_tier_offload_min_priority`); `block_reuse_config.block_reuse_policy` (rejected; renamed to `policy`); `mamba_state_config.enable_branch_snapshot` (alias of `block_reuse_config.enable_branch_snapshot`); `mamba_state_cache_interval` (alias of `mamba_state_config.periodic_snapshot_interval`); `sink_token_length` (ignored on the PyTorch backend) |
| `tests/unittest/_torch/pyexecutor/` for executor tests, `tests/unittest/_torch/pyexecutor/kv_cache/` for the KVCM V2 wrapper, `tests/unittest/kv_cache_manager_v2_tests/` for KVCM V2 core behavior | `tests/unittest/_torch/executor/`, the name from before the executor tests were renamed |

Before relying on any path, confirm it on the current main. Search open pull
requests for planned removals or renames of the files you will touch.

## Owner Map

| Layer | Location | Owns | Must not |
|---|---|---|---|
| User configuration | `tensorrt_llm/llmapi/llm_args.py` (`KvCacheConfig`, `KvCacheCompressionConfig` subclasses, `SparseAttentionConfig`) | Knobs, defaults, validation, rejection of unsupported combinations | Hold runtime state; switch features off silently |
| Executor construction | `tensorrt_llm/_torch/pyexecutor/_util.py`, `py_executor_creator.py` | Building managers in the right order, compatibility admission, KV budget estimation | Contain method-specific runtime logic |
| Executor loop and resource managers | `tensorrt_llm/_torch/pyexecutor/py_executor.py`, `resource_manager.py` | Request lifecycle, scheduling, the order of resource callbacks | Know a KV layout or a method's name |
| KVCM V2, executor side | `tensorrt_llm/_torch/pyexecutor/kv_cache/` (`kv_cache_manager_v2.py`, `mixed_precision_staging.py`, `mamba_cache_manager/mamba_cache_manager_v2.py` for hybrid models), plus model- and backend-specific subclasses under `tensorrt_llm/_torch/attention/backends/` that declare side buffers through `_extra_buffers_per_layer` | Mapping layers to layer groups, pools, life cycles, priorities and reuse policies; extra buffers; the page tables that mixed-precision staging reads (the staging copy itself belongs to the attention metadata) | Make algorithm decisions; branch on one method |
| KVCM V2 core | `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, surfaced by `tensorrt_llm/runtime/kv_cache_manager_v2/` | Pages, slots, pools, the radix tree, eviction, migration, events, publication, release | Know models or methods |
| Compression methods | `tensorrt_llm/_torch/kv_cache_compression/`, plus the manager base in `resource_manager.py` | Cadence, request-local state, selection, transforms, method kernels | Allocate or publish pages; release capacity except through the KVCM API after publishing completion |
| Attention | `tensorrt_llm/_torch/attention/` (backends, metadata) | Consuming the published active representation; per-step metadata; declaring backend side buffers through the manager hooks | Own storage policy beyond those declarations; grow per-feature kernel branches |
| Disaggregated transfer | `tensorrt_llm/_torch/disaggregation/` (Python transceiver, NIXL backend) | Moving published state between context and generation workers; resolving layout and transfer ids on both sides | Assume layouts the sender did not publish; target the deprecated C++ transceiver |

## Design Workflow

1. **Behavior contract.** What changes for the user. Also state what stays
   identical when the feature is off: everything.
2. **Survey.** Find the existing mechanisms that already do most of the job:
   life cycles and window pools, retention priorities, reuse policies, extra
   buffers, lifecycle hooks, cold-page codecs, sparse-attention hooks,
   metadata staging, transceiver layout resolution. See
   `references/design-principles.md`, section 2.
3. **Extension level.** Use the highest level that works: configuration,
   then a field, then an existing extension point, then the feature's own
   package, then shared Python core, then native code. Write a reason for every
   step down. For shared-core or native changes, list every function you will
   change, by layer.
4. **Owner boundaries.** Place each responsibility in a row of the owner map.
5. **Compatibility.** Decide each axis in `trtllm-kv-cache-compat-testing` as
   supported, rejected, or n/a.
6. **Performance budget.** List the per-step costs the feature adds and the
   expected benefit, and pick the baselines (`perf-kv-cache-tuning`).
7. **Validation plan.** Follow the order in the compression development
   guide:
   1. the accuracy or output-quality contract against the uncompressed path;
   2. real end-to-end activation and the claimed compatibility combinations;
   3. end-to-end performance;
   4. automated regression tests.

   Unit tests protect what these stages established; they do not replace
   them. Each stage is its own deliverable, with its own evidence.
8. **User-facing surface.** Docs, an example, manifest and schema
   regeneration for config changes, API-stability references.

Write the gate before editing:

```markdown
## KV-Cache Design Gate
- Behavior contract: <what changes; "feature off = identical to main">
- Existing mechanisms surveyed: <mechanism -> reused | why not>
- Extension level: <config | field | extension point | feature package |
  shared Python core | native>; reason for each step down: <...>
- Shared functions touched (layer x function): <table, or "none">
- Owner placement: <responsibility -> owner-map row>
- Compatibility table: <axis -> supported / rejected (where) / n/a>
- Per-step cost and expected benefit: <...>; baselines: <...>
- Validation plan: <accuracy contract / e2e activation and compatibility /
  e2e perf / regression tests>
- Docs and config artifacts to update: <...>
```

If a reviewer can name an existing mechanism the design duplicates, or an
axis with no decision, the gate is not passed.

### When the Framework Is Not Enough

Be strict about reuse and open to real innovation. If a valuable idea cannot
be expressed with existing mechanisms:
1. prototype it in your own package and measure it;
2. write down why each rung of the extension ladder fails;
3. propose an algorithm-neutral seam (hook, capability, buffer role,
   life-cycle property, codec interface) that defaults to today's behavior;
4. get the owners' decision;
5. land the seam with unchanged-behavior tests, then the method on top.

Section 11 of `references/design-principles.md` has the criteria and examples
from main. Kernels follow `references/adding-kernels.md`.

## Engineering Rules

Details and rationale are in `references/design-principles.md`. The incidents
behind them are in `references/lessons-learned.md`.

- **Reuse before building.** Configuration first, then existing extension
  points, then the feature package. Do not add a new abstraction layer for a
  small feature. Do not route around an owner's component with a fallback.
- **Keep method logic out of shared code.** The runtime, KVCM, attention
  kernels, and C++ never branch on a method's or model's name. Shared changes
  extend an algorithm-neutral contract.
- **Feature-off invariance.** With the feature disabled, outputs, memory, speed,
  errors, and API are identical to main. Gate every new path on the feature's
  configuration. A bookkeeping fix that would change disabled runs is
  proposed separately.
- **Fail fast.** Reject unsupported combinations at configuration time, or at
  executor construction before KVCM allocation when the check needs the device
  or the engine. No version sniffing, no try/except-import fallbacks, no
  half-applied state. Load offline artifacts at initialization, never
  calibrate during inference.
- **Preserve algorithm semantics.** Do not change selection rules or
  granularity, and do not remove modes for convenience. Parallel runs must
  match single-GPU semantics.
- **Hot-path discipline.** No per-step host loops, synchronizations,
  allocations, or blanket memsets. Allocate once, update in place, and keep
  CUDA-graph inputs static.
- **Names and text.** Name things by behavior, not mechanism. Comments explain
  the code as it stands; history goes in the PR. Use no internal jargon,
  version labels, or milestone names anywhere.
- **Code size is a consideration, not a target.** Before adding a feature,
  check whether an existing mechanism can carry it, or whether it can merge
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
| DCO sign-off | commit-msg hook (`scripts/dco_check.py`) | Use `git commit -s`; no AI attribution lines |
| New config fields have descriptions | `tests/unittest/llmapi/test_llm_args.py` | None |
| LLM-args golden manifest is current | `tests/unittest/usage/test_llmapi_config_telemetry_docs.py` | Regenerate with `scripts/generate_llm_args_golden_manifest.py`, and review the diff |
| Serve schema snapshots are current | `scripts/generate_trtllm_serve_schemas.py --check`, CPU schema tests | Regenerate; never hand-edit |
| Public API signatures | `tests/unittest/api_stability/` | Update the references deliberately |
| Test-list entries exist | pre-commit `scripts/check_test_list.py --validate` | Also check the test is on a stage with the right GPU |
| No raw pinned-memory calls in runtime code | pre-commit `scripts/check_pinned_memory_usage.py` | Use `maybe_pin_memory()` or `prefer_pinned()` from `tensorrt_llm/_utils.py` |
| NVIDIA copyright header on new files, year updated on modified files | Nothing | Check every new and touched file (`CODING_GUIDELINES.md`) |
| User-facing config classes follow the Pydantic rules | Partly (field descriptions are tested) | `StrictBaseModel`, `Field(description=...)`, `Literal` for closed sets, validators raising `ValueError`, `status="prototype"` on new experimental knobs (`CODING_GUIDELINES.md`) |
| Feature-off invariance | Nothing | Read every new branch; run touched suites with the feature off |
| Reuse of existing mechanisms; no new layers | Nothing | Compare with the design gate's survey |
| No method or model branches in shared code or C++ | Nothing | Grep the diff outside the feature package |
| No silent fallbacks, no version sniffing | Nothing | Read every `except`, `getattr` default, and capability check |
| Per-step host work, syncs, memsets | Nothing | Read the call sites and their guards |
| Compatibility decided and tested | Nothing | Check the compatibility table against the tests |
| Activation proven, performance and accuracy measured | Nothing | Ask for the evidence (`perf-kv-cache-tuning`, `trtllm-kv-cache-accuracy-eval`) |

## Agent Working Rules

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
- **Clean branch.** Debug probes, extra NVTX ranges, micro-benchmarks, and
  design notes stay out of the PR branch.
- **Order of work.** Before pushing a feature change, run the required hooks
  and the end-to-end gate (`trtllm-kv-cache-compat-testing`). For
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
- Performance: <A/B summary or "missing">
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

- `references/design-principles.md`: the review bar, with rationale.
- `references/lessons-learned.md`: recurring review findings and the rules
  they produced, by category.
- `references/adding-kernels.md`: choosing, writing, wiring, and testing
  kernels for KV-cache features (Triton, CuTe DSL, CUDA C++ with torch ops or
  nanobind).
- `references/kv-cache-architecture.md`: a map of the KV-cache stack, with
  the invariants changes most often break.
