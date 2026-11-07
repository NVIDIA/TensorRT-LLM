---
name: trtllm-kv-cache-compat-testing
description: >-
  Decide and prove how a TensorRT-LLM KV-cache feature combines with the rest
  of the system, and design its tests. Use when adding or changing a KV cache
  manager V2 behavior, reuse policy, KV cache compression method, cold-page
  codec, mixed-precision KV cache path, or sparse/evicting attention, and
  whenever the user asks whether a KV-cache feature works with block reuse,
  partial reuse, CUDA graphs, the overlap scheduler, chunked prefill,
  disaggregated serving, speculative decoding, tensor parallelism, attention
  data parallelism, pipeline parallelism, beam search, sliding-window or
  hybrid models, multi-head latent attention or sparse attention backends, or
  KV cache offload tiers. Covers the compatibility decision per axis, the test
  layers, what belongs in pytest, CI test-list registration, waives, and which
  tests to run.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# KV-Cache Feature Compatibility and Testing

KV-cache features rarely fail on their own. They fail when they meet reuse,
CUDA graphs, the overlap scheduler, disaggregated serving, or speculative
decoding. Decide each combination up front, test what you claim, and reject
the rest before execution. Read the entry skill for the area first:
`trtllm-kv-cache-manager-v2` (`../trtllm-kv-cache-manager-v2/SKILL.md`) for
KV cache manager V2 changes, or `trtllm-kv-cache-compression`
(`../trtllm-kv-cache-compression/SKILL.md`) for KV cache compression
methods.

## Step 1: Decide the Compatibility Table Up Front

Fill this table in the design gate, before writing code, and keep it updated
in the PR description. The per-axis details, failure modes, and smallest
tests are in `references/compatibility-matrix.md`.

| Axis | Decision | Test or rejection |
|---|---|---|
| Block reuse | supported / rejected / n/a | |
| Partial reuse and reuse policies | | |
| CUDA graphs | | |
| Overlap scheduler | | |
| Chunked prefill | | |
| Request abort, preemption, suspend/resume | | |
| Disaggregated serving (Python transceiver, `backend: NIXL`, KV cache manager V2 on both workers) | | |
| Tensor, attention data, context, and pipeline parallelism | | |
| Speculative decoding (each mode) | | |
| Beam search | | |
| Sliding-window attention, variable sliding-window attention (VSWA), hybrid recurrent models | | |
| Multi-head latent attention (MLA) and sparse attention backends | | |
| GPU architecture: every family the feature claims, for example Ampere (SM80, SM86), Hopper (SM90), datacenter Blackwell (SM100, SM103), consumer Blackwell (SM120, SM121) | | |
| KV cache offload tiers and connectors | | |
| Reuse identity: low-rank adaptation (LoRA), cache salt, multimodal inputs | | |
| KV cache events and KV-cache-aware routing | | |
| Pool rebalance (`enable_kv_pool_rebalance`) | | |
| Prefill CUDA graphs (`prefill_cuda_graph_backend`) | | |
| Feature-off invariance | always required | |

Rules for the decision:

- **Reuse and disaggregated serving are high-value combinations.** Preserve
  them whenever the feature's semantics allow, and test them. Rejecting them
  must be a reasoned decision written in the PR, never a silent fallback.
- **Reject early.** Put every unsupported combination in the feature's
  validator, with an actionable message: at configuration time, or at
  executor construction before KV cache manager V2 allocation when the check
  needs the device or the engine. Do not branch at runtime into a degraded
  path.
- **Admission predicates are not fallbacks.** A `supports_*()` method reports
  a capability. It must not switch the feature off quietly.
- **Untested means rejected, for new work.** A combination nobody ran is not
  "supported", and a new feature rejects it. Combinations that existing code
  already allows but nobody has validated are labeled "Untested" in docs and
  matrices. Say plainly when a path works but is unvalidated, for example
  draft KV caches that are transformed but whose accuracy was never measured.
- **Never edit a feature out of a configuration to get past a rejection.**
  Dropping speculative decoding, reuse, or CUDA graphs from an experiment or
  test so that it runs changes what is being measured. Add minimal support,
  or report the rejection.
- **Keep docs in sync.** When the decision changes a documented capability,
  update the feature page and, if relevant,
  `docs/source/features/feature-combination-matrix.md`.
- **Re-derive, do not copy, the general matrix.** The general combination
  table does not distinguish KV cache manager V1 from KV cache manager V2, or
  the transceiver runtimes. Some of its cells are wrong for the default
  KV cache manager V2; for example, KV cache manager V2 rejects block reuse
  with context-parallel Helix. Confirm each cell you rely on in the current
  code.

## Step 2: Layer the Tests

| Layer | Purpose | Typical location |
|---|---|---|
| CPU configuration and admission | Parsing, defaults, serialization, rejection of unsupported combinations with the right error | `tests/unittest/llmapi/`, the feature's CPU tests |
| KV cache manager and bookkeeping | Pools, page counts, lengths, reuse counts, hook cadence, state release, item by item | `tests/unittest/_torch/pyexecutor/kv_cache/`, `tests/unittest/kv_cache_manager_v2_tests/` |
| Kernels and data movement | Byte-exact movement and compaction; layout of cold-page codecs and mixed-precision KV cache conversions (spans, scale groups, lossless parts); non-contiguous, partial and large batches. Lossy kernel numerics are checked locally and reported in the PR description | `tests/unittest/_torch/kv_cache_compression/`, C++ unit tests |
| Small end-to-end smoke | The production executor runs with the feature and activation is proven. Claimed combinations are checked locally on real runs (the end-to-end gate in Step 3), with the commands and results in the PR description | A few integration tests |
| Accuracy | Model-quality contract | Separate evaluation with `trtllm-kv-cache-accuracy-eval` (`../trtllm-kv-cache-accuracy-eval/SKILL.md`), not pytest |
| Performance | Throughput and latency contract | Separate measurement with `perf-kv-cache-tuning` (`../perf-kv-cache-tuning/SKILL.md`), not pytest |

## What Belongs in pytest

- **Yes**:
  - the path runs;
  - bookkeeping is equal item by item (pages, lengths, reuse counts, pool
    occupancy after drain);
  - outputs are byte-exact where the operation is meant to be exact;
  - invalid combinations are rejected;
  - the feature activated (counters or recorded calls).
- **No**:
  - accuracy thresholds;
  - tolerance-based comparisons of lossy paths against a hand-written numeric
    reference;
  - end-to-end quality checks.

  A numeric reference can be wrong in ways the kernel is not, for example by
  grouping quantization dimensions in a different order. Such a test fails on
  some hardware while the kernel is bit-correct. Test lossy paths through
  invariants instead:
  - lossless spans are byte-exact;
  - scale geometry is right;
  - the round trip is stable: re-encoding a decoded page reproduces the same
    bytes where the format promises it.

  Keep pytest small, with few GPU tests; check lossy kernel numerics, such as
  the encode/decode round-trip accuracy for every supported runtime dtype,
  and end-to-end behavior locally on real runs, and write the commands and
  results in the PR description. The KV cache compression development guide
  lists that round-trip accuracy check among its automated tests; run it
  locally instead.
- **Mocks**: a bare `MagicMock` attribute is truthy, so gates like
  `if getattr(cfg, "feature", None):` pass on mocks and hide the real branch.
  Use real configuration objects or `spec`'d mocks.
- **Skips**: a skip condition must not hide that the fix is never exercised.
  Check that the new test actually runs on the CI stage it is registered for.

## Step 3: Register and Run

- **Register new tests.** Files under `tests/unittest/` do not run in CI
  unless they are listed in `tests/integration/test_lists/test-db/*.yml`. Add
  the test next to related entries on a stage with the right GPU, for
  example `l0_cpu.yml`, `l0_a10.yml`, `l0_h100.yml`, or `l0_b200.yml`.
  - The pre-commit test-list validator checks that the entries exist.
  - `python scripts/test_to_stage_mapping.py --tests "<name>"` shows where a
    test runs.
  - Mark tests that need no GPU with `@pytest.mark.cpu_only`. GPU stages run
    with `-m "not cpu_only"` and deselect them; the CPU stage picks them up.
    Do not put CPU-only tests inside a module that skips itself without a
    GPU, or they will run nowhere.
  - Check `tests/integration/test_lists/waives.txt` before claiming that a test
    runs in CI: whole directories can be waived. Run waived tests yourself
    when your change touches them.
- **Run, at minimum**:
  1. the tests you added or changed;
  2. the neighbor tests of every module you touched. For a
     KV cache manager V2 change, run its suites under
     `tests/unittest/kv_cache_manager_v2_tests/` and
     `tests/unittest/_torch/pyexecutor/kv_cache/`, and the disaggregated tests
     if transfer metadata changed;
  3. the tests of the features that share the code you changed, such as reuse,
     speculative decoding, and sparse backends.

  Record what you ran, what was skipped, and why.
- **End-to-end gate.** Green unit tests alone do not justify a push. Before
  pushing a feature change, and after every merge from main or refactor, run
  the feature end to end on real models, with CUDA graphs (padding on) and
  the overlap scheduler on, and show activation in every run. Run each
  configuration below, or reject the combination in the validator:
  1. two model families, one per execution path;
  2. block reuse: a second request that shares a long prefix, with reused
     blocks reported for both arms (the configurations being compared);
  3. disaggregated serving: a context and generation pair on the Python
     transceiver, compared with aggregated serving on the same requests;
  4. each speculative-decoding mode the feature claims.
- **Waives**: entries in `tests/integration/test_lists/waives.txt` match exact
  test identifiers.
  - When a fix makes a waived test pass, remove the waive in the same change,
    or in a follow-up that names the fix.
  - Never waive a failure the change introduced.

## Step 4: Feature-Off Invariance

Run the touched suites with the feature disabled, and compare with main. With
the feature off, the following must be unchanged:
- outputs;
- memory use and the KV cache memory budget;
- speed;
- error messages;
- the public API.

Gate new code on the feature's configuration so that disabled runs take the
original path. If you find a more correct bookkeeping rule for shared code
that would change disabled-run behavior, describe it separately and let the
owner decide. Do not fold it into the feature.

## Review Checklist

```markdown
## Compatibility review
- Compatibility table present and complete: <yes/no>
- Unsupported combinations rejected in the validator with clear errors: <list>
- Claimed combinations each have evidence: <axis -> test | local run>
- Activation asserted (not just config parsed): <how>
- New tests registered in test-db: <stage/file>
- Neighbor suites run: <list>; skipped: <list + reason>
- Feature-off invariance checked: <how>
- Docs updated for capability changes: <files>
```

## References

- `references/compatibility-matrix.md`: per-axis failure modes, smallest
  tests, and evidence.
- `docs/source/features/feature-combination-matrix.md`: the user-facing
  combination table.
- `docs/source/developer-guide/kv-cache-compression-development.md`
  ("Validation"): the KV cache compression development guide's validation
  stages and test lists for compression methods. Two of its test-list items
  run locally, not in pytest, with the commands and results in the PR
  description: "encode/decode round-trip accuracy for every supported runtime
  dtype" and "aligned end-to-end accuracy and performance checks".
