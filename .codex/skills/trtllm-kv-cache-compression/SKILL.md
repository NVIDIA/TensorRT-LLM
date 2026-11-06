---
name: trtllm-kv-cache-compression
description: >-
  Entry point for KV cache compression framework development in the
  TensorRT-LLM PyTorch backend: KvCacheCompressionConfig subclasses,
  create_kv_cache_compression_manager and its admission checks, the
  KVCacheCompressionManager base and its five lifecycle hooks,
  iteration-driven methods such as TriAttention eviction with physical
  compaction, storage-bound methods that provide cold-page codecs to KV cache
  manager V2, and method kernels. Read it first for any new KV cache
  compression method, any change to the KV cache compression manager base,
  hooks, factory, admission, or compaction, and any review of a KV cache
  compression PR. Builds on KV cache manager V2 (trtllm-kv-cache-manager-v2,
  ../trtllm-kv-cache-manager-v2/SKILL.md), whose engineering rules it shares.
  For cold-page codecs also read trtllm-kv-cache-cold-page
  (../trtllm-kv-cache-cold-page/SKILL.md).
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# KV Cache Compression Framework

`docs/source/developer-guide/kv-cache-compression-development.md` defines
the KV cache compression framework:
- the architecture and configuration contract;
- the lifecycle hooks and the storage-bound APIs;
- the ownership table;
- the steps for adding a method;
- the order of validation.

This skill is the entry point for KV cache compression framework
development and the workflow on top of that guide: what to read, which owner
boundaries a change must respect, the contracts the guide leaves implicit,
the rules only a reviewer enforces, and the design-gate and review
templates.

The KV cache compression framework sits on KV cache manager V2 only. Do not
build compression on the deprecated KV cache manager V1, and do not model a
new method on sparse-attention algorithms that still depend on KV cache
manager V1. RocketKV is a sparse-attention method, not a compression method.
For the KV cache manager V2 contracts a method relies on (lengths,
publication, resize, reuse), the current and deprecated paths, and the owner
map, read `trtllm-kv-cache-manager-v2`
(`../trtllm-kv-cache-manager-v2/SKILL.md`). The design workflow, engineering
rules, agent working rules, and PR hygiene that both entry skills share are
in `../trtllm-kv-cache-manager-v2/references/engineering-rules.md`.

## Hard Rules

The job is a working method: rules 1 and 2 come first, and the rest keep
the result correct and reviewable.
`../trtllm-kv-cache-manager-v2/references/engineering-rules.md` and
`../trtllm-kv-cache-manager-v2/references/design-principles.md` explain each
one.

1. Deliver the method end to end first. Wire the whole path (configuration,
   the KV cache compression manager and its hooks, KV cache manager V2,
   attention, outputs) and show it running on a real model in the
   production configuration before polishing anything.
2. Make it work on every model family KV cache manager V2 serves (dense,
   multi-head latent attention (MLA), sparse-attention, sliding-window and
   hybrid, hybrid recurrent, and multimodal models, and speculative-decoding
   draft layers) and on every GPU architecture it claims, such as Hopper
   (H100, H200) and Blackwell (B200, GB200, B300, GB300). A family or
   architecture that cannot be supported yet is rejected at configuration
   time and listed as a gap, never silently skipped.
3. Read the required context below before designing or editing, and build
   only on current paths (KV cache manager V2, the Python cache
   transceiver, `tensorrt_llm/_torch/attention/`), never on deprecated ones.
4. Change the code where the behavior belongs. Use an existing mechanism
   when it fits; when it does not, extend it or write the new code. Do not
   add a flag, a wrapper, or a parallel path to avoid touching the right
   file.
5. Write direct code: no defensive try/except, no fallback that hides an
   error or changes results, no checks for states that cannot happen, no
   options nobody asked for. A counted fallback that gives the same results
   under a resource limit, such as per-request work when an optional batch
   workspace does not fit in memory, is fine. Reject unsupported
   combinations at configuration time.
6. With the method off, outputs, memory, speed, errors, and API stay
   identical to main.
7. Write few, strong tests: each one runs the real path and fails if the
   method breaks. Never edit, skip, or loosen a test to make it pass; fix
   the code, or stop and report the conflict. When an approved behavior
   change makes an expectation obsolete, update the test and say why in the
   PR. Keep pytest small, with few GPU tests; check lossy numerics and
   end-to-end behavior locally on real runs, and write the commands and
   results in the PR description.
8. Decide and test every compatibility axis, including block reuse, CUDA
   graphs, the overlap scheduler, disaggregated serving, speculative
   decoding, and parallelism (`../trtllm-kv-cache-compat-testing/SKILL.md`),
   and measure performance end to end in the production configuration
   (`../perf-kv-cache-tuning/SKILL.md`).
9. Never remove a feature, mode, model support, or optimization to make
   code smaller, and never change an experiment's settings to make a result
   pass.
10. Give each concept one full name and use it everywhere: identifiers, test
    names, comments, docs, PR text, and replies to people. Write
    "mixed-precision high window", never "window"; "FP8 KV cache", never
    "FP8" or "KV". Abbreviate only with common knowledge (LLM, GPU, CUDA,
    `attn`) or after writing "full name (ABBR)" at first use.
11. Say "done", "passed", or "pushed" only with evidence you can show.

## Route the Task

Paths are relative to this file.

| Task | Read next |
|---|---|
| Cold-page compression: cold-page codecs, the NVFP4 cold-page format, rotary position embedding (RoPE) precision | `../trtllm-kv-cache-cold-page/SKILL.md` |
| An iteration-driven method (eviction, compaction): lengths, protected tokens, cadence, CUDA graphs, speculative decoding, parallelism | `references/iteration-driven-methods.md` |
| Method kernels | `references/adding-kernels.md` |
| A change to KV cache manager V2 itself, or a KV cache manager V2 contract a method relies on | `../trtllm-kv-cache-manager-v2/SKILL.md` |
| Whether a method works with reuse, CUDA graphs, the overlap scheduler, disaggregated serving, speculative decoding, or parallelism; test design | `../trtllm-kv-cache-compat-testing/SKILL.md` |
| Throughput, latency, memory, profiling, tuning | `../perf-kv-cache-tuning/SKILL.md` |
| Accuracy of a lossy method | `../trtllm-kv-cache-accuracy-eval/SKILL.md` |

## Required Context

1. The development guide above, in full. Its GitHub line anchors can be
   stale; locate symbols by name. Also `CODING_GUIDELINES.md` and the root
   `AGENTS.md`.
2. `docs/source/features/kv-cache-compression.md` and the recipes in
   `examples/kv_cache_compression/`.
3. `docs/source/developer-guide/kv-cache-cold-page-codec.md`, for
   storage-bound work.
4. Code:
   - `KvCacheCompressionConfig`, its subclasses, and
     `KvCacheCompressionConfigType` in `tensorrt_llm/llmapi/llm_args.py`;
   - `create_kv_cache_compression_manager()` and the admission checks next to
     it in `tensorrt_llm/_torch/pyexecutor/_util.py`;
   - `KVCacheCompressionManager` in
     `tensorrt_llm/_torch/pyexecutor/resource_manager.py`;
   - the method packages under `tensorrt_llm/_torch/kv_cache_compression/`:
     `triattention/`, `compaction.py`, `quantization_for_cold_page/`.
5. Tests:
   - `tests/unittest/_torch/pyexecutor/kv_cache/test_kv_cache_compression_manager.py`;
   - `tests/unittest/_torch/kv_cache_compression/`.

When the guide and the code disagree, the code and tests win. Name the
mismatch, and fix the guide in the same change or report it.

One deliberate difference: the guide's "Turn validated behavior into
regression gates" lists "encode/decode round-trip accuracy for every
supported runtime dtype" and "aligned end-to-end accuracy and performance
checks" as automated tests. Under hard rule 7, check both locally on real
runs instead, and write the commands and results in the PR description.
Re-run these local checks in every later PR that changes the method's
kernels or data path, and report them there.

## How to Use It

One compression method can be enabled per LLM instance, on
KV cache manager V2. In Python, pass `kv_cache_compression_config=` with a
`TriAttentionKvCacheCompressionConfig` or a
`ColdPageQuantizationCompressionConfig`; cold pages also need a nonzero Host
tier. In a YAML file for `trtllm-serve` or `trtllm-bench --config`:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
kv_cache_compression_config:
  algorithm: triattention        # always set the discriminator
  budget: 2048
  beta: 128
  eviction_mode: union
  calibration_path: /path/to/calibration.pt
```

Complete examples:
- `docs/source/features/kv-cache-compression.md`;
- `examples/kv_cache_compression/` (one recipe per method, including
  activation checks);
- `examples/llm-api/llm_kv_cache_compression.py`.

Confirm activation in every run. A parsed configuration is not evidence:
- for an iteration-driven method, look for compressed-token counts growing
  and eviction rounds running;
- for a cold-page method, look for cold-page codec calls on compressed
  life cycles.

## Composing an Iteration-Driven Pipeline

A round is a fixed sequence of stages. Keep each stage in its place:

| Stage | Where it runs | Notes |
|---|---|---|
| Bind (once) | Host, outside CUDA graphs | Fix geometry, allocate workspaces, load and validate calibration, warm up kernels and compile them just in time |
| `on_request_init` | Host | Create request-local state; reject requests the method cannot handle |
| `on_generation_step_begin` | Host | Remember the prepared batch when the overlap scheduler needs it |
| Forward | Device, inside CUDA graphs | Attention reads the compacted KV cache; the method adds nothing here |
| Due check | Host, in `on_generation_step_end` | From published counts only; no device synchronization |
| Snapshot | Host to device | Block offsets and metadata for this round, in pinned buffers |
| Score | Device, current stream | Method kernels, batched over requests, layers, and heads |
| Fold across ranks | Device | Gather or reduce, so every tensor-parallel rank selects the same tokens |
| Select | Device | Top-k with explicit tie-breaking |
| Compact | Device | `build_compaction_params()` and `compact()` from `compaction.py` |
| Publish | Host | Add the evicted count to `py_num_compressed_tokens` |
| Shrink | Host, through KV cache manager V2 | `resize(capacity, None)` on the target and draft KV caches, after the completion event |
| `on_request_finish` | Host | Release request-local state; never free blocks |

Kernels for the score and select stages follow
`references/adding-kernels.md`.

## How a Method Is Wired on Main

**Startup**

1. The configuration parses by `algorithm` into one subclass of the
   `KvCacheCompressionConfigType` union. Mixed-precision KV cache and any
   compression configuration are rejected together.
2. A config with `changes_physical_kv_length = True` makes attention turn
   rotary position embedding (RoPE) fusion off.
3. The executor's KV cache creator calls
   `create_kv_cache_compression_manager()` before it builds any
   KV cache manager V2. The function:
   - rejects context-parallel Helix;
   - then dispatches with an `if` per algorithm.

   Each branch calls `validate_kv_cache_compression_compatibility()` (GPU
   architecture, block reuse, speculative modes) before building its
   KV cache compression manager. A branch may return no KV cache compression
   manager when the method has nothing to do; for example, the cold-page
   branch does so when the active KV cache is already NVFP4. An algorithm
   without a branch skips validation, logs a warning, and runs uncompressed.
4. A KV cache compression manager with `provides_cold_page_codec = True`
   becomes the compression provider of the target KV cache manager V2 and of
   a separate one-model draft KV cache manager V2. A KV cache compression
   manager with `uses_iteration_lifecycle = True` is registered as a resource
   manager.
5. `bind_kv_cache_managers(target, draft)` runs after KV cache manager V2
   construction. It requires KV cache manager V2, and sets
   `kv_compression_manages_history` from `changes_physical_kv_length`. The
   KV cache compression manager is then moved last in the resource-manager
   order.

**Every iteration (read `resource_manager.py` for the exact conditions)**

| Hook | Fired from | When |
|---|---|---|
| `on_request_init(request)` | `prepare_resources` | For each context request on its first context chunk |
| `on_generation_step_begin(batch)` | `prepare_resources` | On every call, including context-only batches |
| `on_context_step_end(requests)` | `update_resources` | Once per iteration, for the requests whose last context chunk ran |
| `on_generation_step_end(batch)` | `update_resources` | On every call, after KV cache manager V2 has updated capacity for the step |
| `on_request_finish(request)` | `free_resources` | On completion or abort, before KV cache manager V2 frees the request's blocks |

All hooks default to no-op. There is no attention-time hook.

Ordering consequences:
- `prepare_resources` runs after KV cache manager V2 has allocated the step's
  KV cache.
- `free_resources` iterates in reverse, so `on_request_finish` sees the
  blocks still allocated. It must never free blocks itself.
- In the overlap-scheduler loop, iteration N+1's `prepare_resources` runs
  before iteration N's `update_resources`. A step-end mutation must protect
  the tokens of requests that are already in the prepared next batch.

**What `changes_physical_kv_length = True` turns on.** Declare it only when
the method really makes physical and logical lengths diverge.
- Generation-step resizes stop touching the history length.
- Generation blocks are no longer committed for reuse. Only the prompt prefix
  stays reusable.
- The steady generation and steady multi-token prediction (MTP)
  input-preparation fast paths are disabled.
- RoPE fusion is disabled in attention.
- Some sparse-cache features, such as sparse offload for DeepSeek-V4, reject
  it.

Points two to four (no generation-block reuse, no steady fast paths, no RoPE
fusion) are performance costs. Measure them; do not discover them later. The
last point is a compatibility limit.

**Publishing a length change.** Update `request.py_num_compressed_tokens`
first, then shrink with `cache.resize(new_capacity, None)`, which leaves
history untouched. Input preparation subtracts the compressed count from the
cached-token count it gives attention. RoPE positions stay logical. Never
derive positions from the compacted physical length.

**Storage-bound methods** have no executor hook. KV cache manager V2 calls the
cold-page codec when it moves pages between the GPU and Host or Disk.
Host-to-Disk moves copy the encoded blob unchanged. See
`trtllm-kv-cache-cold-page` (`../trtllm-kv-cache-cold-page/SKILL.md`).

## Owner Boundaries

| Owner | Owns | Must not |
|---|---|---|
| Method config (`llm_args.py`) | `algorithm`, method fields, `changes_physical_kv_length`, `supports_block_reuse()`, `supports_speculative_decoding()` | Put method fields on the common base; act as runtime switches |
| Admission and factory (`_util.py`) | Rejecting real unsupported combinations; one branch per method; local imports of heavy method code | Hold method policy beyond admission; leave a requested method silently off |
| `KVCacheCompressionManager` subclass (method package) | Cadence, request-local state, selection, kernel launches, publishing compressed counts, returning trailing pages through `resize(new_capacity, None)` after publishing | Release a request's blocks directly; shrink capacity any other way; reach into KV cache manager V2 internals the KV cache compression framework does not expose |
| Generic compaction (`compaction.py`) | Moving kept tokens and protected tails inside KV cache manager V2 pools | Select tokens |
| KV cache manager V2 | Pages, slots, pools, mappings, migration, events, release, rollback | Learn a method's name or scores |
| Attention backends | Reading the published active KV cache | Gain compression-specific branches |

## Adding an Iteration-Driven Method

Follow "Adding a compression method" in the guide. These steps are easy to
miss:

1. **Configuration and exports.**
   - Subclass `KvCacheCompressionConfig` with a unique `algorithm` literal.
   - Add the class to `KvCacheCompressionConfigType`, and export it from
     `tensorrt_llm/llmapi/__init__.py`.
   - Regenerate the LLM-args golden manifest and the serve schemas, and
     update the API-stability reference.
   - Follow the telemetry rules that
     `tests/unittest/usage/test_llmapi_config_telemetry_docs.py` enforces:
     the `algorithm` value is captured, and paths and other user data never
     are. Extend its discriminator test with the new algorithm.
2. **Admission.** Add the method's architecture gate. Add an **explicit**
   speculative-decoding branch: the current check is an `if`/`else`, so a new
   method otherwise inherits another method's allowed modes. Add explicit
   rejections for combinations you have not tested, such as disaggregated
   serving, beam search, or sparse attention. A missing guard is not support.
3. **Factory.** Add the branch, call the validator, and import the method
   package locally. Decide whether the KV cache compression manager must also
   exist during the KV cache memory budget estimation pass, which runs before
   the real KV cache is built. Test that choice; that pass sizes the KV cache
   memory budget, so a KV cache compression manager that allocates memory
   must be present there.
4. **KV cache compression manager.**
   - Subclass `KVCacheCompressionManager` in a new package under
     `tensorrt_llm/_torch/kv_cache_compression/`.
   - Call `super().bind_kv_cache_managers()` first.
   - Implement only the hooks you need.
   - Reuse `build_compaction_params()` and `compact()` for physical moves.
   - Publish counts before resizing.
5. **Stream ordering.** Order compaction on the stream that the next forward
   pass and KV cache manager V2 use, and record a completion event before any
   consumer reads the result.
6. **Wiring is automatic.** Registration, binding, ordering, and teardown
   (`shutdown()`) need no executor or KV cache manager V2 change. Do not add
   method branches to `py_executor.py`, KV cache manager V2, or C++.
7. **Tests and docs.**
   - Unit tests go under `tests/unittest/_torch/kv_cache_compression/`, and
     framework cases into `test_kv_cache_compression_manager.py`.
   - Add a section to the feature page, a recipe under
     `examples/kv_cache_compression/`, and an option in
     `examples/llm-api/llm_kv_cache_compression.py`.

The deeper contracts (protected tails, the overlap scheduler, CUDA graphs,
speculative decoding, parallelism, kernels) are in
`references/iteration-driven-methods.md`.

## Changing the Framework Itself

A change to `KVCacheCompressionManager`, its hooks, the factory, admission, or
`compaction.py` affects every method. Before making it:

1. Show that no existing hook or resource-manager callback can carry the
   need. Name at least one method other than yours, on main or concretely
   planned, that needs the change.
2. Keep the change algorithm-neutral.
   - A new hook defaults to a no-op.
   - It fires from an existing resource-manager callback.
   - It receives only what the framework already owns: requests, the batch,
     and the bound KV cache managers.
3. Keep existing methods unchanged. Run
   `tests/unittest/_torch/kv_cache_compression/` and
   `test_kv_cache_compression_manager.py`, and show that every method's
   behavior and hook cadence are the same as before.
4. Update the hook and ownership tables of the development guide in the same
   change.

## What Machines Check, and What Only Review Catches

| Rule | Enforced by | Reviewer action |
|---|---|---|
| Config fields described; manifest and telemetry current | `tests/unittest/llmapi/test_llm_args.py`, `tests/unittest/usage/test_llmapi_config_telemetry_docs.py` | Check that the telemetry opt-in excludes paths and user data |
| Union and export include the class | Config dispatch tests | Check `llmapi/__init__.py` too |
| Factory branch exists | Nothing: a missing branch only warns | Run the method and show activation |
| Explicit speculative-decoding branch | Nothing | Read the admission diff |
| Capability methods truthful | Nothing | Match each `supports_*()` value to a test |
| No method name outside its package and config | Nothing | Grep the diff |
| Publish before resize; return pages only through `resize`; never release a request's blocks in hooks | Partly, through unit tests | Read the step-end path |
| No calibration while serving | Nothing | Check artifacts load at initialization |
| Accuracy and end-to-end activation | Nothing: CI has no compression integration or accuracy tests | Require the evaluation evidence in the PR; see `trtllm-kv-cache-accuracy-eval` (`../trtllm-kv-cache-accuracy-eval/SKILL.md`) |

## Design Gate

Extend the general design gate in
`../trtllm-kv-cache-manager-v2/references/engineering-rules.md` with:

```markdown
## Compression Design Gate
- Integration path: <iteration-driven | storage-bound | both>
- Hooks used and why: <...>
- Protected tokens (prompt, sinks, tails, next prepared batch): <...>
- changes_physical_kv_length: <value; accepted costs: no generation-block reuse,
  no steady fast paths, no RoPE fusion>
- Capabilities: block reuse <T/F>; speculative modes <list>; evidence for each:
  <pytest check | local run in the PR description>
- Explicit rejections added: <combination -> check>
- Reused KV cache compression framework parts: <compaction, lifecycle hooks, cold-page codec adapter>
- Estimation-pass behavior: <built | skipped; why; test>
- Calibration artifact: <none | field + offline tool>
- Per-step cost and placement: <...>
- Accuracy tasks that actually trigger the method: <...>
```

## Test Selection

- Framework, factory, admission, binding:
  `tests/unittest/_torch/pyexecutor/kv_cache/test_kv_cache_compression_manager.py`.
- Config dispatch and serialization: `tests/unittest/llmapi/test_llm_args.py`.
  Telemetry: `tests/unittest/usage/test_llmapi_config_telemetry_docs.py`.
- Method kernels, pipeline, overlap scheduler, compacting draft and target
  KV caches together, byte-for-byte compaction checks against a reference
  implementation: the files in `tests/unittest/_torch/kv_cache_compression/`.
  Many require a Blackwell (SM100 family) GPU.
- KV cache manager V2 interaction:
  `tests/unittest/_torch/pyexecutor/kv_cache/test_kvcm2_integration.py`
  and `tests/unittest/kv_cache_manager_v2_tests/` when lengths, capacity,
  release, or cold-page codec setup and calls change.
- Register any new test path in the test-db; see
  `trtllm-kv-cache-compat-testing` (`../trtllm-kv-cache-compat-testing/SKILL.md`).

## Review Output

```markdown
## Findings
- [High] <file:line> <ownership, ordering, invariance, or correctness issue>
- [Medium] <file:line> <capability, admission, compatibility, or performance issue>
- [Low] <file:line> <naming, docs, diff hygiene>

## Fit With the KV Cache Compression Framework
- Hooks minimal and correctly placed: <yes/no>
- Method logic contained in its package and config: <yes/no; leaks>
- Admission explicit (speculative-decoding branch; untested combinations rejected): <yes/no>
- Guide update needed: <yes/no + section>

## Evidence
- Activation: <...>; compatibility table: <...>; accuracy: <...>; performance: <...>
```

## References

- `references/iteration-driven-methods.md`: protected tails, overlap-scheduler
  and CUDA-graph safety, compaction, speculative decoding, parallelism,
  kernels, and TriAttention as a worked example.
- `references/adding-kernels.md`: choosing, writing, wiring, and testing
  kernels for compression methods, cold-page codecs, mixed-precision KV cache
  conversion, and KV cache manager V2 data movement (Triton, CuTe DSL, CUDA
  C++ with torch custom ops or nanobind).
- Shared with `trtllm-kv-cache-manager-v2`, under
  `../trtllm-kv-cache-manager-v2/references/`:
  - `engineering-rules.md`: the design workflow and design gate,
    engineering rules, what machines check, agent working rules, PR
    hygiene, and the review output format;
  - `design-principles.md`: the review bar, with rationale;
  - `lessons-learned.md`: recurring review findings and the rules they
    produced, by category;
  - `kv-cache-architecture.md`: a map of the stack around KV cache manager
    V2, with the invariants changes most often break.
