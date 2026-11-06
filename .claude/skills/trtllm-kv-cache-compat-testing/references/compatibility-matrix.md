<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV-Cache Feature Compatibility Matrix

Every KV-cache feature has to say where it stands on each axis below. Give one
of these answers per axis:
- **supported and tested**;
- **rejected early, with a clear error**, at configuration time or at
  executor construction;
- **not applicable**, with the reason.

"Probably works" is not an answer. A new feature rejects every combination it
has not tested. Combinations that existing code already allows but nobody has
validated are labeled **Untested** in docs and matrices, never "supported".

For each axis, this file lists what typically breaks, the smallest test that
catches it, and the evidence a reviewer should ask for. Check the general
combination table in `docs/source/features/feature-combination-matrix.md`
first; it may already constrain the feature.

Keep pytest small, with few GPU tests. The smallest tests below that compare
model outputs end to end (feature on vs off, CUDA graphs, the overlap
scheduler, parallelism, disaggregated serving, speculative decoding, model
families, GPU architectures) are checked locally on real runs; write their
commands and results in the PR description. Bookkeeping, rejection,
activation, byte-exact, and layout checks stay in pytest; see "What Belongs
in pytest" in `../SKILL.md`.

## Reuse Axes

### Block reuse (full-block prefix reuse)

- **What breaks**:
  - a block whose bytes were transformed after publication (compacted, evicted
    into, re-encoded) is matched by a later request as if it still held the
    original tokens;
  - the reused prefix is read in a different representation than it was
    written in;
  - retained copies (mixed-precision high windows, side caches) are reused
    without their companion state.
- **Smallest test**: two requests share a long prefix. Run the second after the
  first finishes, then assert the following.
  1. The reused-token count is what the design predicts. Check it; a hit rate
     that silently drops to zero passes every output check.
  2. The output matches the run without reuse. Compare token for token for
     lossless features; for lossy features, use the documented accuracy
     contract, measured with `trtllm-kv-cache-accuracy-eval`
     (`../../trtllm-kv-cache-accuracy-eval/SKILL.md`). Check this locally on
     real runs and write the commands and results in the PR description.
  3. Reusing a prefix that was itself compressed, restored, or partly evicted
     behaves as designed.
- **Evidence**: per-request reuse counters for both arms (the configurations
  being compared). State whether the prefix blocks keep their identity, and
  which blocks the feature mutates.

### Partial reuse and reuse policies

- **What breaks**:
  - a partially matched block is copied without the feature's extra state;
  - a sliding-window, sink, or retained-copy life cycle is matched under the
    wrong policy, so the whole prefix is truncated to its first blocks;
  - retention priorities evict the companion copy first, which silently
    shortens every later reuse.
- **Smallest test**:
  - a prefix that ends mid-block;
  - a prefix whose retained copy was evicted under memory pressure, followed by
    a request that should reuse it.

  Assert the reused length for both.
- **Evidence**: the reuse policy per life cycle (for example required,
  optional, or private), and why the feature's state belongs where it is.
- **Also test**:
  - Re-serving an identical prompt whose length is not a multiple of the
    block size, with greedy continuations compared against a block-aligned
    control.
  - Preempted generations under the `MAX_UTILIZATION` scheduler policy, which
    re-prefill and may reuse their own blocks, so cached tokens can exceed
    the prompt length.
  - Hybrid models whose recurrent-state snapshots bound how far a prefix can
    be reused.

## Execution-Mode Axes

### CUDA graphs

- **What breaks**:
  - host-side metadata rebuilt with new tensors, so captured addresses go
    stale;
  - shapes that depend on the batch;
  - work that is only valid for real requests running on padding or dummy
    requests;
  - allocations during capture;
  - per-step work sized to the captured maximum without anyone noticing.
- **Smallest test**: run the same requests with CUDA graphs on and off and
  compare outputs. Include a batch size that needs padding, and generation
  lengths that cross page boundaries. Fill spare pages and padded slots with
  NaN in a test run to prove they are never read: kernels must stop at the
  real length. The KV cache manager V2 debug fills do this without custom
  code: `TRTLLM_KV_FRESH_PAGE_FILL=nan` poisons newly handed-out pages, and
  `TRTLLM_KV_GUARD_PAGE` adds a guard page. Know their limits:
  - The fill depends on the KV cache dtype. An unpacked FP8 KV cache has no
    fill kernel, so the fill is skipped with a warning; run the check with a
    BF16 or FP16 KV cache instead. A packed NVFP4 pool cannot hold NaN, so a
    non-zero fill lands as +6.0; compare outputs with the fill on and off,
    which must be identical.
  - The fill covers the model's own layers, not the layers the manager
    appends: the mixed-precision high pool is not filled. The log reports
    one total for the first fill, not each pool. To check a pool the fill
    does not reach, fill and check it with test instrumentation.
  - For an NVFP4 KV cache, the fill writes the packed key and value data,
    not the block-scale buffers, which have their own storage. An unwritten
    scale keeps its old value in both the fill-on and the fill-off run, so
    the comparison can miss that read; check the scale buffers with separate
    instrumentation.
  - Only some paths point masked entries at a guard page: the FlashInfer
    sliding-window path, and the DeepSeek-V4 sparse cache manager with its
    own guard pages. On other paths, the reservation log does not show that
    masked entries use it; check those paths with their own instrumentation.
- **Evidence**: the CUDA graph is still replayed for the feature's steps
  (CUDA graph launch counts or a profile). The feature does not force eager
  mode.
- **Known traps**:
  - With CUDA graphs on, a final context chunk of one token can run as a
    decode row inside the CUDA graph batch (`promote_final_contexts`).
    Forward-time feature code that keys on "context request" must handle that
    case. Test a prompt whose last chunk is a single token.
  - Dummy requests for CUDA graph padding and warm-up appear in scheduled
    batches, and never pass through request initialization. Skip them
    (`request.is_dummy`) in every hook and per-request loop.

### Overlap scheduler

- **What breaks**: the scheduler prepares step N+1 while step N runs, so
  feature state is one step stale.
  - A KV cache mutation between steps races with the in-flight step.
  - A request finishes while its next step is already queued.
- **Smallest test**:
  - compare outputs with the overlap scheduler on and off on the same
    requests, including requests that finish at different steps;
  - repeat several times. If run-to-run divergence appears only with the
    overlap scheduler, consider three causes: a missing dependency between
    GPU streams, fixed with an event that the consumer stream waits on; a
    host buffer that the next step overwrites while an asynchronous copy
    still reads it (KV cache manager V2 guards its host page-table rows
    against this in `_prepare_host_rows`); and kernel inputs, such as batch
    shapes, that occur only with the overlap scheduler. A full device
    synchronization changes the timing of all three, so it narrows the
    search but does not identify the cause. Confirm the cause from the
    kernel inputs and from which buffers are still in use.
- **Evidence**: where the feature's mutation is ordered relative to the
  in-flight step, and what makes it safe.
- **Never** conclude that a feature "cannot use the overlap scheduler", and
  never turn the overlap scheduler off as the fix. When one step must wait,
  for example on the feature's completion event, make only that step wait.

### Chunked prefill

- **What breaks**:
  - per-request hooks fire on every chunk instead of the final one;
  - spans and staging buffers computed per chunk instead of per request;
  - partial pages at chunk boundaries;
  - long prompts rejected because chunked prefill is off.
- **Smallest test**: a prompt several chunks long, with the feature on. Compare
  against unchunked prefill, and assert hook cadence (final chunk only, where
  that is the contract). Use small chunk sizes with long inputs and batches
  larger than one.
- **Evidence**: hook call counts per request. Accumulate per-chunk state
  incrementally, and decide (score, evict, quantize a window of tokens) once
  the whole prompt exists. A chunked run that scores better than an unchunked
  one is a bug.

### Request lifecycle

- **What breaks**: state leaks on abort or cancellation, suspend/resume or
  preemption, early stop, or streaming.
- **Smallest test**: abort a request mid-generation, preempt and resume one,
  then assert that request-local state and pages are released and that a
  following request is unaffected.
- **High-pressure case**: serve far more requests than fit, with prompts
  longer than any sliding window and the overlap scheduler on, so that
  requests are suspended as eviction victims while their next step is in
  flight. A token count that advanced in flight while capacity did not
  produced `-1` tail page entries at page boundaries before. Assert that no
  tail entry is `-1` after resize or resume.
- **Evidence**: pool occupancy returns to its initial value after the requests
  drain.

## Distribution Axes

### Disaggregated serving

- **What breaks**:
  - context and generation workers disagree on layout: pools, layer groups,
    dtypes, per-layer KV-head counts, window sizes, or retained copies;
  - the transceiver sends only the main KV cache pool, not the feature's extra
    state;
  - the feature is enabled on one side only;
  - reuse on the generation side assumes state that was never transferred.
- **Smallest test**: a context-plus-generation pair with the feature on both
  sides, compared against aggregated serving on the same requests.
  - Also run a mismatched configuration and assert it is rejected.
  - Target the Python transceiver (`transceiver_runtime: PYTHON`) with the
    NIXL backend and KV cache manager V2 on both workers. The C++ cache
    transceiver is deprecated; do not add features to it.
- **Evidence**: which buffers are transferred, and how the receiver resolves
  their geometry.
- Before declaring disaggregated serving unsupported, read the exact gates in
  the newest code. Make it work through correct base configuration and
  layer-id mapping, not special cases.
- Set `cache_transceiver_config.backend: NIXL` on both workers. With the
  backend unset, a worker starts but brings up no cache transceiver, and it
  rejects the disaggregated requests routed to it. A test that forgot the
  backend tests nothing.
- By default the whole prompt's KV cache moves in one transfer after prefill.
  With `cache_transceiver_config.enable_pipelined_transfer`, each completed
  prefill chunk is sent while later chunks compute. If the feature claims
  disaggregated support together with chunked prefill, test both modes.

### Tensor, attention data, context, and pipeline parallelism

- **What breaks**:
  - KV heads are sharded, so per-rank decisions diverge across ranks;
  - per-layer tables assume all layers are local;
  - staging buffers and budgets are sized with global head counts;
  - attention data parallelism mixes requests from different ranks.
- **Smallest test**: tensor parallelism (TP) at TP=1 against TP=N on the
  same requests. Algorithm semantics must match TP=1: a selection or eviction
  decision that depends on the head shard must be reduced or defined so that
  all ranks agree. Split per-head calibration with the standard contiguous
  head split, and all-reduce or gather the scores. Check how sibling methods
  handle TP before designing your own scheme. Add a pipeline-parallel run if
  layers are split across ranks.
- **Evidence**: what is computed per rank and what is agreed across ranks.

### Speculative decoding

- **What breaks**:
  - draft layers that share the target KV cache manager get no feature state,
    or the wrong state;
  - an independent draft KV cache is ignored;
  - rejected draft tokens are rewound in the main KV cache pool but not in the
    feature's extra state;
  - compression of the target and the draft disagree, as with a draft that
    keeps tokens the target evicted.
- **Smallest test**: every speculative mode the feature claims, with rejections
  forced by a low-acceptance draft. Compare outputs against speculative
  decoding off. The cases relevant to the KV cache are:
  - draft layers that live in the target KV cache manager V2, as in one-model
    multi-token prediction (MTP);
  - one-model drafters with their own architecture;
  - drafters with a separate draft KV cache manager V2.

  Two-model speculative decoding has been removed; do not design for it.
- **Evidence**: which KV caches the draft layers use, and how rewind reaches
  every copy.
- **Design rule**: decide draft handling once, when the feature's KV cache
  manager is created.
  - Draft layers with the target's KV cache geometry follow the target's
    policy.
  - Drafters with their own architecture keep a plain KV cache, or the
    combination is rejected.
  - An explicitly pinned draft dtype is respected.
  - Include a speculative-decoding run in every end-to-end gate.

### Beam search

- **What breaks**: shared prompt pages and per-beam copies are not handled by
  the feature's state.
- **Default**: reject beam search unless the feature implements and tests it.

## Model and Layout Axes

### Sliding window, variable sliding window, and hybrid models

- **What breaks**:
  - sliding-window layers and full-attention layers are in different layer
    groups, so a feature that assumes one layer group mis-sizes pools;
  - sliding-window eviction drops tokens the feature expects to keep;
  - recurrent state pools in hybrid models are sized together with KV cache
    pools.
- **Smallest test**: one model with alternating sliding-window and
  full-attention layers, and one hybrid model, if claimed. Check pool sizing
  in the startup log and the output against the feature-off run.
- **Rules**:
  - Follow the existing sliding-window conventions: whether bounds are
    inclusive, and what value means "disabled".
  - Mask out-of-window entries rather than zero-filling them. A `-1` page
    entry must never reach a kernel as an address.
  - Decide skip rules by layer or page type before looking at dtypes.

### Multi-head latent attention (MLA) and sparse attention layouts

- **What breaks**:
  - the latent KV cache has a single buffer with trailing rotary position
    embedding (RoPE) dimensions;
  - sparse attention backends carry auxiliary or side caches, such as indexer
    keys, that the feature does not transform or transfer;
  - compressed-token layouts have their own geometry.
- **Smallest test**: each claimed MLA or sparse attention backend, with the
  auxiliary caches checked to be untouched, or transformed consistently.

### KV-cache tiers and connectors

- **What breaks**: offload and onboard of pages the feature transformed;
  connectors that copy raw pages without the feature's metadata.
- **Smallest test**: force offload under memory pressure, then onboard and
  reuse, for each tier claimed: Host or Disk.
- **Default**: reject KV cache connectors unless they are tested.

## Reuse-Adjacent Axes

### Reuse identity: low-rank adaptation (LoRA), cache salt, multimodal inputs

- **What breaks**: block keys include the reuse scope (LoRA id, salt) and the
  tokens. Multimodal requests add their own identity. A feature that rewrites
  tokens, keys, or committed pages can let a request match another scope's
  prefix, or miss its own.
- **Smallest test**:
  - the same tokens with different salts or LoRA ids never share blocks;
  - a repeated multimodal prompt with the same inputs reuses, and one with
    different inputs does not.

### KV cache events and KV-cache-aware routing

- **What breaks**: events must describe complete, published blocks. A
  feature that adds life cycles, rewrites committed pages, or changes block
  identity can emit events for blocks that are not reusable, or break the
  hashes routers compute.
- **Smallest test**: with events enabled, the stored and removed events match
  the blocks that a later request actually reuses.

### Pool rebalance

- **What breaks**: `enable_kv_pool_rebalance` moves capacity between layer
  groups at runtime. A feature that adds a layer group, or relies on a fixed
  split, can be starved or resized under it.
- **Smallest test**: run with pool rebalance on under a workload that shifts
  between layer groups, and check that the feature's layer group keeps
  working and its reuse depth is reported.

### Prefill CUDA graphs

- **What breaks**: prefill CUDA graphs (`prefill_cuda_graph_backend`) capture
  context steps, so context-time feature work must follow the same
  static-buffer rules as decode CUDA graphs.
- **Smallest test**: the feature with prefill CUDA graphs on and off, with
  outputs compared.

### GPU architectures

- **What breaks**:
  - a kernel built or tuned for one architecture family fails, is skipped,
    or produces different numerics on another;
  - shared-memory limits and supported data types differ between families;
  - a feature validated on only one GPU family is untested on the others.
- **Smallest test**: run the end-to-end path on each architecture the
  feature claims. KV cache manager V2 itself runs on more families than most
  kernels, for example Ampere (SM80, SM86: A100, A10), Hopper (SM90: H100,
  H200), datacenter Blackwell (SM100: B200, GB200; SM103: B300, GB300), and
  consumer Blackwell (SM120, SM121: RTX PRO 6000, GB10). Check what the
  code you build on supports, and reject at configuration time only the
  architectures the feature cannot support.
- **Evidence**: a local end-to-end run on each claimed architecture, with the
  commands and results in the PR description. CI does not need a copy per
  architecture: register the feature's few GPU tests on a stage that has the
  right GPU, such as `l0_h100.yml`, `l0_b200.yml`, `l0_b300.yml`,
  `l0_gb200_multi_gpus.yml`, or `l0_gb300.yml`.

## Feature-Off Invariance

Not an axis, but always required. See Step 4 of `../SKILL.md`.
