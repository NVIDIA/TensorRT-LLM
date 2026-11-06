<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lessons Learned in KV Cache Manager V2 and KV Cache Compression Work

These rules summarize findings that came up again and again in reviews of
AI-assisted work on KV cache manager V2 and the KV cache compression
framework in this repository. The areas covered are:
- the KV cache compression framework;
- token eviction and physical compaction;
- cold-page compression;
- the mixed-precision KV cache;
- KV cache manager V2 reuse;
- performance and accuracy validation for all of the above.

Each entry is a rule followed by its rationale. The skills cite these rules;
read the category you are working in before starting.

## The Most Common Findings

1. Deliver the feature: wire the whole path, run it end to end, and make it
   work on every model family and GPU architecture it claims. A
   demonstration on one model is not done.
2. Write direct code. No defensive try/except, fallbacks that hide errors
   or change results, checks for impossible states, or flags and parallel
   paths added to avoid changing the code that owns the behavior.
3. Keep tests few and strong. Never edit, skip, or loosen a test to make it
   pass; update it only when an approved behavior change makes its
   expectation obsolete, and say why in the PR.
4. Use the mechanism the codebase already has when it fits; do not build a
   duplicate.
5. Read the code, logs, or results before claiming anything. Never invent
   behavior, defaults, numbers, or quotes.
6. Claim "done", "pushed", or "verified" only with a checkable artifact.
7. Add only the code the feature needs: no class, file, wrapper, manager, or
   kernel that duplicates an existing one.
8. Measure performance only in the production configuration, on the workload
   the feature is for, with a trace you have read.
9. Align accuracy evaluation with the official protocol, on full datasets and
   full repeat counts.
10. Keep behavior identical when the feature is off, and keep the diff in
    scope.
11. Never shrink or expand the scope, and never change experiment settings,
    without asking.
12. Name things by behavior and give each concept one full name, used
    everywhere. Write plain comments, define every abbreviation at first
    use, and use no internal jargon.
13. Answer the question that was asked, first.

## 1. Design and Architecture

- **Reuse existing mechanisms.** Read the relevant code paths and their call
  sites before designing. Prefer configuration, then a parameter on an existing
  mechanism, then a new path. Windowing, scale handling, page locking, layouts,
  and metadata usually exist already. The mixed-precision KV cache needed only
  an extra sliding-window-with-sinks life cycle, not new pool logic.
- **No duplicate layers.** Add the classes and files the feature needs, but
  no class, wrapper, registry, runner, or subsystem that duplicates an
  existing one. A separate manager per method, extra
  "representation" layers, or wrappers around a single hook add review cost
  and are removed again in review.
- **Methods are subclasses with hooks.** A compression method subclasses the
  KV cache compression manager (`KVCacheCompressionManager`) and overrides
  only the hooks it needs. It never gets a parallel manager or an owner class
  outside the KV cache compression framework.
- **Shared core stays generic.**
  - Method logic lives in the method's package and configuration.
  - Shared code, KV cache manager V2, attention, and C++ never branch on a
    method's or model's name.
  - Native code exposes one generic hook. Policy, layout, and kernel launches
    stay in the method.
  - One callback per migration batch, never per page.
- **Prefer the executor-side layer over the KV cache manager V2 native
  core.** The KV cache manager V2 native core is C++. Configuration, the
  executor-side wrapper, and existing callbacks are where features belong.
  Change the KV cache manager V2 native core only for an algorithm-neutral
  contract and with evidence, such as a profile.
- **Respect ownership.** "Do not touch" means another component's core and
  storage logic. Adapt on your side, put code where it belongs, and write
  real bugs in other components down for their owners. Do not modify another
  author's open PR.
- **Keep the feature invisible to unrelated subsystems.** Speculative-decoding
  interfaces, model code, attention kernels, and the executor loop should not
  learn that the feature exists. Rewriting another subsystem's interfaces is
  out of scope for a feature change.
- **Generic steps take narrow inputs.** A generic compaction step takes the
  cache layout and keep decisions through its public entry point. It does not
  take algorithm scores or an opaque dictionary of everything.
- **Bookkeeping code does not launch kernels.** Management code switches state
  and calls the owner's API. It does not allocate tensors or run per-step
  PyTorch ops.
- **Branch on the feature's state, not on a dtype, shape, or model name.**
  Checks keyed to today's format or pool count broke when a second format
  arrived. Design interfaces keyed by format so the next one plugs in without
  renames.
- **Do not change algorithm or parallel semantics on your own.** Selection
  rules, granularity, normalization, and tie-breaking follow the reference.
  Under tensor parallelism, every rank must reach the same decision as a
  single GPU; independent per-rank selection silently changes the algorithm.
- **Decide open choices before building.** Inventory the code, present the
  options (including "change nothing") with scope, benefit, and impact on
  others, get a decision, and record the rejected options. Do not implement a
  choice that belongs to the user or an owner and present it afterwards.
- **Vendor small, stable utilities** rather than taking a fragile dependency
  or adding indirection.
- **Extend through existing hooks.** Disaggregated support for the
  mixed-precision KV cache needed one override of the transfer-layer-id hook
  and no change to transfer code.
- **Methods hold a KV cache manager; they do not become one.** A compression
  method holds a normal KV cache manager V2. It never subclasses KV cache
  manager V2, writes its private fields, or puts its name into the KV cache
  manager V2 native core.
- **Fix bad state where it is produced.** A `-1` page index that crashed a
  kernel came from the KV cache manager's capacity accounting. Making the
  kernel tolerate it would have hidden the bug.
- **Fix every reader of a contract.** A page-index scale fix applied to one
  page-table getter missed the flat variant that another attention backend
  reads, which caused an illegal memory access.
- **Follow existing conventions; fix the new code.** A new chunk aligner
  assumed a shifted cumulative-sum layout while everything else used the
  inclusive one. The fix belonged in the new code, not in the convention.

## 2. Code Quality

- **Name by observable behavior, in the codebase's own vocabulary.** For
  example, use "generation", not "decode", where the codebase says
  generation. Do not use a flag name that suggests data loss when nothing is
  lost. Name pools by role (`mixed_precision_high`, `mixed_precision_low`),
  not by the trick behind them.
- **Follow the neighbors exactly.** Use existing option names, accepted values,
  defaults, the config schema pattern, import style, and the way optional
  features are wired. When an equivalent option exists elsewhere in the
  repository, align with it.
- **Comments are one or two plain lines on what the code does.** Explain
  special cases, constants, and borrowed code, and say where borrowed code
  comes from. History belongs in the PR description.
- **No internal shorthand anywhere.** No variant codes, round numbers, version
  labels for kernels, milestone names, or run bookkeeping in code, docs,
  reports, or replies.
- **One full name per concept, in code and in text.** Names cut down to a
  fragment keep coming back in review: a variable called `window` for the
  mixed-precision high window, or "FP8" and "KV" in test names and
  explanations where the FP8 KV cache is meant. The reader cannot tell which
  window, or which FP8, is meant. Write the full name every time, in the
  identifier, the comment, and the explanation. Abbreviate only with common
  knowledge (LLM, GPU, CUDA, `attn`) or after writing "full name (ABBR)" at
  first use.
- **One production path.** No fallbacks, version sniffing, KV cache manager
  V1 paths, hard caps, or separate eager-mode and CUDA-graph implementations.
  Shapes derive from the deployment's configuration.
- **Fail fast, at configuration or initialization time.** A missing parameter
  that surfaces later as a shape mismatch, a calibration path that silently
  produces identity scales, or an early `return` that skips an unsupported
  KV cache manager all hide errors from the user.
- **Validate once, in one place.** Delete checks for states that earlier
  validation already rules out, such as the KV cache manager version,
  architecture, alignment, or null pointers, especially on hot paths. Do not
  repeat a compatibility check inside an attention backend.
- **Use explicit contracts.** Access attributes directly, put typed fields on
  the owning class, and keep internals out of the public API. Do not churn
  existing `getattr` code outside your change.
- **Simplify by removing redundancy.** Remove dead paths, probes, superseded
  implementations, and the parameters and tests that served them, by reading
  the code end to end. Measure the diff after every round; a
  "simplification" that grows the diff is not one.
- **Code size follows the requirement.** Lines of code are a consideration,
  not a hard limit. Before adding a feature, check whether an existing
  mechanism or the existing framework can carry it, rather than writing a new
  standalone implementation, and test that decision.
- **Simplification never removes features.** Removing a conversion path, an
  optional calibrated input, an algorithm mode, a performance optimization,
  or a supported model family is a regression, whatever it saves in lines.
  When the user asks to delete something specific, delete it without arguing.
- **Keep debug probes and debug ranges out of the branch.** Keep
  micro-benchmarks and design notes out too.

## 3. Performance

- **Production configuration only.** Keep CUDA graphs and the overlap
  scheduler on, use the recommended serving settings, and limit
  instrumentation to NVIDIA Tools Extension (NVTX). A "profiling
  configuration" with CUDA graphs off misstates step time by multiples, and
  its results have to be redone.
- **Benchmark the workload the feature is for.**
  - Use large batches with continuous batching, long inputs and outputs, and
    distinct prompts.
  - Include the stress case where every request triggers the feature in the
    same step.
  - Identical prompts with block reuse on inflate throughput several-fold.
- **Back every number with a trace you have read.**
  - Map the whole per-step pipeline first.
  - Capture a steady-state window that contains the event, marked with NVTX.
  - Separate host from GPU, exposed from hidden, and the feature's own cost
    from shared costs.
  - Use the full kernel list; truncated lists produced phantom negative costs.
- **Compare like with like.** Keep the GPU type, model, parallelism, and
  build the same, and change one setting at a time. Understand each setting,
  such as the difference between chunk size and token limits, before
  sweeping it.
- **Never perturb the baseline.** A harness that adds per-step work, or a
  control call that serializes migrations, invalidates the comparison.
- **Prove the tested path ran.** File names and labels are not evidence. A
  setting written into a result's name but never passed to the command, or a
  guard that routes one model to another kernel, silently measures something
  else. Check the logs or the trace.
- **Separate kernel claims from system claims.** Check a kernel's share of
  end-to-end time before optimizing it. Per-transfer or per-kernel claims need
  a controlled test with fixed shapes, not an end-to-end replay window where
  each arm (one configuration being compared) sees different requests.
- **Turn savings into capacity.** Eviction must physically free blocks, so
  that saved memory becomes a larger batch. Report the largest batch each arm
  can run, and check peak memory.
- **Work out where the feature should win before sweeping**, and reproduce the
  baseline frontier yourself. Size each arm's memory and batch to its own
  maximum, and report the standard Pareto view.
- **Reproduce reference experiments setting by setting.** Read the
  reference's scripts, configurations, and raw outputs, and tabulate every
  difference. A silently mismatched configuration is equivalent to fabricated
  data.
- **Ground memory models in the real configuration.** A staging buffer sized
  as maximum batch times maximum length can be orders of magnitude too large.
  Count each buffer once in the memory estimate.
- **Launch at the batched granularity.** A per-layer or per-page launch can
  make a compressed offload slower than a plain copy.
- **Treat implausible results as bugs.** Non-monotonic curves, a larger cache
  with a lower hit rate, or more disk hits with compression point to a broken
  setup, not to a new phenomenon.
- **Optimize in planned stages.** Estimate the possible gain, change one thing,
  re-measure against the previous stage, and run a quick accuracy check. Weigh
  every gain against the code it adds.
- **Make reports reviewable.**
  - Plan the full table first and mark unfinished cells "pending".
  - Show absolute values with units.
  - State the setup and the commit.
  - Cross-check every figure against the raw data, and keep scripts and raw
    data together.

## 4. Feature Compatibility

- **Check capabilities once, before construction.** Declare block-reuse,
  speculative-decoding, and physical-length capabilities on the method, check
  every combination in one place before any manager is created, and reject
  with a clear message.
- **Keep the high-value combinations.** CUDA graphs, block reuse,
  disaggregated serving, all main attention backends, and multi-head latent
  attention (MLA), grouped-query attention (GQA), and multi-head attention
  (MHA) must keep working. Only feature-local fused shortcuts may be
  disabled. Do not blanket-disable reuse: a generation-only eviction method
  does not conflict with prefix reuse of prompt blocks.
- **Overlap-scheduler differences are bugs.** Outputs must match with the
  overlap scheduler on and off. Never conclude that a method "cannot use the
  overlap scheduler", and never turn the overlap scheduler off as a fix. When
  a step must wait, make only that step wait.
- **Stay inside the existing CUDA-graph capture.**
  - Add no method-specific capture or warm-up machinery.
  - Keep captured regions free of host syncs and data-dependent branches.
  - Make kernels stop at the real length on padded CUDA-graph slots.
  - Fill spare pages with NaN to prove they are never read.
- **Chunked prefill: accumulate per chunk, decide once.** Score or evict only
  after the last chunk. A chunked run that scores better than an unchunked one
  is a bug, not a feature.
- **Speculative decoding: decide draft handling once, when the KV cache
  manager is created.**
  - Draft layers with the target model's KV cache geometry follow the target
    model's policy.
  - Drafters with their own architecture keep a plain cache, or the
    combination is rejected.
  - Respect an explicitly pinned draft dtype.
  - Compress only finalized tokens, and keep the draft tail.
- **Disaggregated serving: verify, do not assume.** Read the exact gates in the
  newest code before saying it is unsupported. Make it work through correct
  base configuration and layer-id mapping, not special cases. Layer-id mapping
  for new layers broke it once.
- **Sliding-window layers need explicit handling.** Use the existing
  sliding-window conventions. Mask out-of-window entries instead of
  zero-filling them: a `-1` entry reached a kernel as an address. Decide skip
  rules by layer or page type before dtype.
- **Never delete a feature from a configuration to get past a rejection.** An
  experiment that drops speculative decoding to avoid a rejection no longer
  measures the intended setup.
- **Say plainly when a combination is unvalidated.**

## 5. Testing

- **Unit tests are functional.** Pytest checks five things:
  - configurations start and run;
  - bookkeeping matches exactly (pages, lengths, layer ids, which layers take
    the feature);
  - unsupported configurations are rejected;
  - outputs are byte-exact where the operation is meant to be exact;
  - layouts and invariants hold (spans, scale groups, byte-exact lossless
    parts, identical re-encode where the format promises it).

  Kernel-numeric references with tolerances, accuracy thresholds, and
  model-quality checks stay out of unit tests, and GPU tests stay few. Check
  lossy numerics and end-to-end behavior locally on real runs, and write the
  commands and results in the PR description; re-run them in every PR that
  changes the feature's kernels or data path. Small end-to-end smoke tests
  that only prove activation belong in the integration tests; see
  `trtllm-kv-cache-compat-testing`
  (`../../trtllm-kv-cache-compat-testing/SKILL.md`).
- **Run end to end on real models before pushing a feature change**, and after
  every merge or refactor. Use one run per execution path, for example two
  model families and a speculative-decoding model, with CUDA graphs and the
  overlap scheduler on. Green unit tests alone miss interactions such as the
  overlap scheduler with eviction.
- **Deliver the validation matrix you planned.** Unit tests are not a
  substitute for planned end-to-end and profiling cells.
- **Tests fail loudly.** Do not skip, waive, revert, or disable your way past a
  failure. "1 skipped, exit 0" is not a pass. Gates must not accept `Mock`
  objects because they are truthy. A version-gated skip can make a fix look
  verified when it never ran.
- **Know what CI runs.** New test files are not picked up automatically.
  Register them in the per-GPU test lists, keep them light, and check that they
  are not deselected (for example, CPU-only tests inside a GPU-gated module).
- **Triage CI failures with evidence.** Read every failing log. Compare with a
  main build and nearby runs. Classify each failure as caused by the change,
  by someone else, or flaky. When your fix repairs a waived test, unwaive it.
- **Run the hooks on every commit**, including documentation-only commits.
- **Run your tests plus the existing tests of everything you touched.** A
  branch that is green on its own test files can still break the existing
  tests of the modules it modified.
- **Test the code you ship.** Use the latest code on a matching build, and a
  frozen snapshot for long experiments.

## 6. Accuracy Evaluation

- **Align with the official protocol by running the reference yourself.**
  Match:
  - the chat template and thinking mode;
  - sampling;
  - maximum length;
  - repeats;
  - budget and mode;
  - the scorer.

  Read the settings recorded in result files, not script defaults. Instruct
  models need their chat template.
- **Align in stages.** First, the baseline matches the published score.
  Second, a lossless arm runs under the same memory pressure. Only then do the
  lossy arms run. A ten-point baseline gap makes every comparison meaningless.
- **Full datasets and full repeat counts.** Small runs are quick probes only.
  Always state the sample count.
- **Size repeats to the noise, and give baselines the same seeds.** Identical
  scores across sampled repeats are a red flag. Know what each seed controls.
- **Read raw outputs.** Check truncation, loops, and missing answer boxes, and
  score every arm with the same extractor. Apparent drops are often
  truncation artifacts, and arms scored by different extractors produce fake
  regressions.
- **Treat implausible numbers as setup bugs** until proven otherwise.
  Examples: a lossier arm scoring higher, more disk hits with compression, or
  a runtime-length average above the output length.
- **Prove the mechanism engaged at the intended level.** For example, report
  the share of reused KV cache that came from compressed pages. Arms labeled
  with a pressure level they never reached are invalid.
- **When porting a method, compare stage by stage, bitwise.** Compare scores,
  selections, eviction decisions, and the first generated token against the
  reference implementation, starting from a single deterministic request.
- **Tie every score to a recorded, clean source tree and recipe.** Record the
  commit, uncommitted changes, build, configuration, and time. Copy rerun
  commands verbatim from the original log; a command retyped from memory can
  drop a setting such as the chat template.
- **Test lossy cold-page codecs over repeated round trips.** Offload is a
  move, so every round trip re-encodes. Compare raw bit patterns after 1, 2,
  4, and 8 round trips.
- **A configured feature that does nothing must change nothing.** With a
  cold-page codec configured but no page moved, outputs must be
  token-identical to the lossless control.
- **When a benchmark saturates, add a harder one** instead of over-reading
  small differences.

## 7. PR and Review Hygiene

- **One purpose per PR.**
  - A KV cache compression framework PR carries no method kernels.
  - Independent fixes go on their own branch off main.
  - Documentation can go separately.
  - Revert unrelated edits, comment rewording, and copyright churn.
- **Debugging must not grow the PR.** Measure the diff against main after every
  fix round. Fix crashes and wrong results, not hypothetical robustness, and
  simplify after fixing. Audit rounds that add a test and a guard for every
  finding grow the PR and pull in new code-owner groups.
- **Feature-off behavior is identical to main.** Present a more correct
  internal fix that changes non-feature behavior as a separate choice.
- **Split by owner and risk.** Keep stacked PRs strictly additive, and rebase
  onto the latest base resolving only conflicts.
- **Push protocol.** Push only when asked.
  - Before pushing a feature change, run the hooks and the end-to-end gate.
  - For documentation-only changes or small agreed review fixes, the hooks
    are enough. Push, then run the remaining checks and report them as
    pending.
  - Report the pushed state with the remote reference.
  - Do not push while a PR's CI is running; a new head commit restarts the
    whole pipeline.
  - On a shared PR, merge main instead of rebasing.
- **Descriptions are short and plain, modeled on the repository's PRs.**
  - Use the title format.
  - Give the purpose and the benefit, one quantified result, and the scope
    limits.
  - Write the commands and results of the local lossy-numeric, accuracy,
    end-to-end, and performance runs.
  - Leave out stale status notes and internal links.
  - Integrate new text into the existing description instead of appending an
    "Updates" section.
- **Claim only what was validated.** In public docs and support matrices, mark
  unsupported features "No" and unverified ones "Untested". Present options as
  exploratory, and promise no accuracy you have not measured.
- **Review comments: one at a time, in order.** Quote the comment, analyze it
  against the code, propose a short reply and change, and wait for the
  decision. Replies are short, plain, and copyable, with no internal status.
- **Docs and reports put the logic first.** Open with the point, then the
  mechanism, then supporting detail. Keep terms and numbers consistent
  everywhere, and lead with the general idea before the example.
- **When polishing someone's text, change only the wording.** Keep its meaning,
  structure, links, and data.

## 8. Agent Working Discipline

- **Verify instead of guessing.** Read the code, logs, or results, or rerun,
  before claiming anything. Typical invented facts: races, kernel limits,
  "needs C++", "runs only once", "this is the default".
- **"Done" requires an artifact.** Keep "edited", "verified", "pushed", and
  "merged" distinct. Check a push with the remote reference; a failed push
  must never be reported as done.
- **Report progress with evidence.** Never say "almost done" without results.
  Monitor every launched job, run independent arms in parallel, and do not
  let one blocked arm stall the rest.
- **Never shrink or expand scope on your own.** No subsets, no dropped cells,
  no simulated variants, no lowered targets, no extra unrequested work.
- **Never change protocol parameters, ratios, run windows, or hardware
  without asking.** Never silently deviate from an agreed design, layout,
  numerics, or dispatch.
- **Answer the question first.** Put the answer before the narrative, and
  answer every sub-question.
- **Speak plainly.** Principle first, then the design idea, then details, in
  the codebase's own terms.
- **Find the root cause.** Do not work around a problem: disabling the
  overlap scheduler, masking instead of evicting, zero-filling, waiving, and
  calling something noise are all workarounds.
- **Check the branch, the build, and the configuration before running.** Stale
  builds and forgotten instructions wasted days.
- **Estimate runtime and results before launching.** When something is far
  off, stop launching and find the cause.
- **Keep valid results.** Judge runs by their outputs, do not rerun everything
  for a small issue, and never overwrite earlier reports.
- **Read handoff docs, prior reports, and recorded lessons** before planning
  similar work.
- **Deliver the requested artifact first.** Raise extra concerns afterwards,
  and do not over-verify a quick task.
- **Turn repeated corrections into standing rules**, in the project's skills
  and docs, so later sessions do not repeat them.
- **Supervise delegated work.** Results from a sub-agent that look
  suspiciously clean need a spot check against the raw evidence. When a
  long chain fails, rerun the last known-good end-to-end path first.
- **One engine per GPU set.** Launching the LLM API script under `mpirun -n N`
  started N engines on the same GPUs, and produced false reports of KV cache
  corruption.

## 9. Subsystem Lessons

### KV cache compression framework

- Compression is a memory-lifecycle operation. Sparse attention is a compute
  optimization. They are separate configurations, and there is one compression
  method per instance.
- The KV cache compression manager talks to KV cache manager V2, never to
  attention metadata.
- The KV cache compression manager runs after KV cache manager V2 has updated
  lengths for the step. Its five hooks map onto the executor's resource
  callbacks.
- Store per-request results, such as compressed-token counts, on the request,
  so the engine reads correct lengths without engine changes.
- Generic pieces belong to the KV cache compression framework: a compaction
  mover that knows only the layout and the keep list, and the cold-page codec
  hooks for tier migration.
- Calibration is offline, and the runtime only loads artifacts.

### Eviction and compaction

- Evict only finalized generation tokens. Keep the prompt, which keeps prefix
  reuse valid. Keep the speculative tail.
- Eviction must physically free blocks. Resize once per round, and check that
  attention really reads the compressed length.
- All eviction modes, including per-head and per-layer-per-head, are part of
  the algorithm. Check the reference code, not only the paper's prose, before
  claiming equivalence.
- Score normalization that is global over a row cannot be tile-fused without
  changing semantics.
- Do not score sliding-window layers. Compact them to the latest sliding
  window plus the protected tail, as `compaction.py` does. Wait on the
  compaction event before freeing or reusing tail pages.
- Reduce the method to its essential kernels: score, top-k, compact. Launch
  across layers, not in a per-layer loop. Reuse the repository's top-k and
  compaction kernels.
- Use 64-bit offsets, and bound reads by the current length.

### Cold pages

- The runtime KV cache stays at native precision. Pages are compressed only
  at the tier boundary, so a page carries at most one lossy rounding per
  offload.
- The cold-page codec's input is the hot KV cache, which may already be an
  FP8 KV cache. A higher-precision cold-page format cannot recover
  information that is already lost.
- Fuse quantization into the tier transfer. The encode must not be slower
  than the copy it replaces.
- Store only what the format needs. Padding inherited from GPU-kernel
  alignment wasted the space compression saved.
- Partial pages work because scale groups stay within one token and global
  scales are calibrated offline.
- Rotary position embedding (RoPE) precision options must match the naming,
  accepted values, and defaults of equivalent active KV cache options. Check
  that a switch reaches every path.
- Compress the large non-RoPE latent of MLA caches. Leave indexer keys,
  compressor state, and recurrent state lossless.
- A commit path that adopts an existing same-prefix page must not let a
  lower-precision copy replace freshly computed higher-precision KV cache
  entries.

### Mixed-precision KV cache

- Build the mixed-precision high pool on the sliding-window-with-sinks life
  cycle, and reuse `pool_ratio`. Do not add a new ratio setting or a KV cache
  manager subclass.
- Separate the storage format from the compute dtype: the feature converts
  storage, and attention computes in the high-precision dtype.
- On main, the mixed-precision high pool holds only KV cache entries that
  attention computed, never dequantized low-precision data. A design that
  fills mixed-precision high pages from the mixed-precision low pool changes
  this property; make that an explicit decision, and evaluate its accuracy.
- Gate everything on the feature, and name things "mixed-precision high/low".
- Zero the mixed-precision staging copy once after allocation. Confirm its
  rebuild frequency from the call-site guard.
- For sparse models, reuse the existing top-k gather, as the high-row overlay
  path does, so the converted rows scale with top-k, not with the maximum
  length.
- Reject unsupported KV cache managers explicitly instead of skipping them
  silently.

### KV cache manager V2 reuse

- Model reuse exactly as the implementation does it:
  - radix tree match, then pruning by REQUIRED life cycles;
  - the commit policy and its default;
  - adoption of existing pages on commit;
  - release of retained pages under per-conversation retention.
- Report losses to capacity eviction separately from missed reuse.
- With several life cycles, check that pool ratios route the savings to the
  pool that serves reuse. A large per-page saving can become only a few
  percent more slots in the life cycle that matters.
- Under memory pressure, capacity accounting must cover the next token during
  eviction and suspension. A lagging capacity lets a kernel read a `-1` page
  index.
- Ratio-style configurations can change meaning between versions. Verify
  their semantics before porting a published configuration.

### Sparse attention

- Algorithms that read query and key content need post-RoPE values, so they
  turn RoPE fusion off with a per-method flag.
- Methods that need attention metadata are attention-modifying methods, not
  KV cache compression methods.
- Using global addresses as page indices risks 32-bit overflow.

## 10. Recurring Agent Failure Modes

| Failure mode | What it looked like | Countermeasure |
|---|---|---|
| Unfinished feature | Stubs and placeholders, a path never wired into the executor, one model or one GPU shown as general support | End-to-end run on every claimed model family and GPU architecture |
| Defensive and conservative code | Try/except around everything, fallbacks, checks for impossible states, new flags and parallel paths instead of changing the code that owns the behavior | Direct code; change the code that owns the behavior |
| Test bloat and test gaming | Mock-only tests, expected values computed by the code under test, tests that cannot fail; tests edited, skipped, or loosened until they pass | Few strong tests that fail when the feature breaks; stop and report a conflicting test |
| Inventing facts | Plausible code structure, defaults, limits, or numbers that were never read or measured | Read or run before claiming |
| Unsupported status claims | "Almost done", "verified", "pushed" with nothing behind it | Artifacts for every status word |
| Over-engineering, then defending it | New managers, wrappers, kernels, fallbacks, and checks; diffs that grow under "simplify" | Extension ladder; diff measured each round |
| Scope creep and silent changes | Edits to other owners' code; changed configs, ratios, layouts, or numerics | Change only what was asked; propose deviations |
| Silent scope shrinking | Subsets, single runs, dropped cells, simulated variants | Full matrix and full datasets |
| Workarounds instead of fixes | Disabling the overlap scheduler, zero-filling, waiving, reverting, "it's noise" | Root cause first |
| Weakened experiments | CUDA graphs off, crafted inputs, identical prompts with block reuse, features that never triggered | Production config; activation evidence |
| Rationalizing anomalies | Stories for impossible results; blaming the model | Treat as a setup bug |
| Jargon and buried answers | Coined labels, names cut to a fragment, undefined abbreviations; answers to a different question | One full name per concept; abbreviations defined at first use; answer first |
| Repeating corrected mistakes | The same layer, subset, or format reappearing after correction | Write the rule down and re-read it |
