---
name: trtllm-kv-cache-accuracy-eval
description: >-
  Plan, run, and report accuracy evaluations for lossy or state-changing
  TensorRT-LLM KV-cache features: KV cache quantization and mixed-precision KV cache,
  cold-page compression, token eviction or compaction (for example
  TriAttention), sparse attention that drops KV cache entries, and reuse
  policies that rebuild or approximate reused prefixes. Use before claiming
  such a KV-cache feature preserves accuracy, when comparing it against
  FP8 KV cache or full KV cache references, when its score drops or looks too
  good, and when writing the accuracy section of its PR or design doc. Covers
  protocol alignment, reference arms (the configurations a feature is
  compared against), activation evidence, seeds and standard errors,
  truncation and extraction artifacts, run hygiene, and the report format.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Accuracy Evaluation for KV-Cache Features

A lossy KV-cache feature is defined by its accuracy contract. A small tensor
error does not mean unchanged model output, and an unchanged score on a task
that never triggers the feature means nothing. Treat accuracy as an evidence
workflow with a written protocol, not as one number.

Functional tests, accuracy evaluation, and performance measurement are
separate tasks. Do the one that was asked, and label the evidence honestly.
Read the entry skill for the area first:
`trtllm-kv-cache-manager-v2` (`../trtllm-kv-cache-manager-v2/SKILL.md`) for
KV cache manager V2 changes, or `trtllm-kv-cache-compression`
(`../trtllm-kv-cache-compression/SKILL.md`) for KV cache compression
methods.

## Step 1: Fix the Protocol Before Running

Write the protocol down first, and copy it verbatim into every rerun. Silent
drift between runs is the most common reason scores cannot be compared.

| Item | Source of truth |
|---|---|
| Sampling (temperature, top-p, top-k, seed policy) | The model's `generation_config.json` and the benchmark's official harness or paper |
| Number of samples per prompt and how scores are averaged | The benchmark's official protocol, for example average pass rate over N samples |
| Prompt construction | The official harness: chat template on or off, system prompt, thinking mode, few-shot format |
| Maximum input and output lengths | The official setting. For reasoning models, the output limit must hold the full reasoning |
| Scorer and answer extraction | The official scorer, used for every arm and every result directory |
| Dataset version, split, and subset rules | The official harness. Report any filtering |

- Prompt construction is the most hidden trap. A reasoning model evaluated
  with a plain prompt instead of its chat template and thinking mode can score
  half of its published number.
- Greedy decoding for a model whose official protocol samples is not a valid
  reference, even if the score looks better.
- When a paper's harness differs from the model's official recipe, choose one
  deliberately, say which, and keep it for every arm (one configuration being
  compared).
- When the comparison is against an external reference implementation, run
  that implementation's own code. Reuse its prompts, sampling, and scorer for
  the TensorRT-LLM arms instead of re-implementing its logic.

Run the TensorRT-LLM arms through the repository's evaluation entry point
instead of writing a new harness: `trtllm-eval` (`tensorrt_llm/evaluate/`),
with every arm using the same configuration file apart from the feature's own
keys. The accuracy test infrastructure and reference scores live under
`tests/integration/defs/accuracy/`. Do not add an accuracy test or a
reference score for the feature there or anywhere else in pytest: run the
arms locally on real runs, and write the commands and results in the PR
description. Re-run them in every later PR that changes the feature's
kernels or data path. When a benchmark's official protocol needs its own harness, run
that harness against a TensorRT-LLM server, and keep the harness identical
across arms.

Align in stages, each run through the TensorRT-LLM path (not another
library):
1. The baseline reproduces the published score of the model and benchmark
   within noise. A gap of several points makes every later comparison
   meaningless; resolve it first.
2. Where storage, transfer, or reuse is involved, a lossless arm runs under
   the same memory pressure. This separates the effect of the mechanism from
   the effect of the lossy transform.
3. Only then run the lossy arms.

## Step 2: Choose Arms and Tasks

- **Reference arms**: the full or uncompressed path with the same model,
  backend, sampling, and limits. For precision features, include the model
  dtype or the FP8 KV cache, and the closest existing lossy option, for
  example the pure NVFP4 KV cache next to the mixed-precision KV cache.
- **Tasks that exercise the feature.** A generation-time eviction method on a
  task with short outputs never evicts, so it measures the dense path. A reuse
  feature on prompts with no shared prefix never reuses. Pick tasks by the
  feature's risk:

| Feature risk | Task family |
|---|---|
| Losing distant context (quantized history, eviction, cold pages) | Long-context retrieval and multi-needle tasks, in the style of multi-round coreference resolution (MRCR), needle-in-a-haystack, and RULER |
| Error accumulating over long generation | Long reasoning: competition math, science question answering with long chains of thought |
| Reused or rebuilt prefixes (reuse policies, restored cold pages) | Multi-turn or forced-reuse variants, where the second turn must read the reused prefix |
| Tokens near the start or end of the context (first-N/last-N policies) | Tasks whose answer depends on instructions at the start and the most recent turn |

- Short multiple-choice benchmarks rarely stress KV-cache features. Use them
  only as sanity checks.

## Step 3: Prove the Feature Was Active

A configuration that parses is not activation evidence. For every arm, record
proof that the intended path ran and that its result was consumed:

- the resolved configuration from the startup log: KV cache dtype, pools, method,
  window sizes;
- method counters: compression rounds, evicted tokens, encoded and decoded
  pages, conversions into the mixed-precision staging copy;
- for reuse features, reuse ratios per request. In a forced-reuse evaluation,
  confirm that the reused fraction is what the design predicts, and treat an
  arm whose reuse collapsed as a different experiment.

Control and report how much the mechanism engaged in each run: the fraction
of tokens evicted, pages offloaded and restored, tokens served from the
mixed-precision low pool. The setting you vary must move that number. An arm
labeled "30% compressed" that measured 10% is a 10% arm. Count only the
benchmark's own requests, not filler traffic.

## Step 4: Run Hygiene

- One engine per GPU set, one output directory per arm and seed, and the
  launch line checked before the runs start. See `perf-kv-cache-tuning`
  (`../perf-kv-cache-tuning/SKILL.md`) for the engine rules.
- Start with a quick probe: a few prompts, end to end, checking lengths,
  finish reasons, parse success, and runtime. Then run the full set.
  Final numbers come from the full dataset, never a convenient subset.
- Long prompts must be admitted. Enable chunked prefill or raise the token
  limits so that no request is rejected. A rejected request that the harness
  silently drops biases the score toward short inputs.
- Before reading any score, check:
  - the number of records equals the expected count;
  - errors and tracebacks are zero;
  - generations are non-empty and not degenerate (repetition loops);
  - extraction parsed the answers. Inspect a handful of raw outputs: a score
    of zero may mean a broken parser, not a broken model.

## Step 5: Treat Noise and Artifacts Correctly

- **Seeds and standard errors.** Run several seeds per arm. Report the mean,
  the standard error (SE), and paired differences against the reference on
  the same prompts. Differences inside about two SEs are "no measurable
  difference". Add seeds before claiming a gap.
- **Truncation.** Count `finish_reason == length` per arm. An apparent drop
  that comes from hitting the output limit is a truncation artifact. Fix the
  limit, or report the truncated items separately.
- **Nondeterminism is not regression.** Different outputs across runs prove
  nondeterminism, not lower accuracy. Only a measured score difference shows
  a regression.
- **Zero variance is a red flag.** Identical scores across sampled repeats
  usually mean that sampling or seeding is not doing what you think. Check
  what each seed controls before reporting.
- **Check output lengths.** A lossy method can change how long the model
  generates. Before attributing a gap, compare output-length distributions
  and truncation counts across arms.
- **Check invariants.** Before publishing a figure, check that:
  - averages of runtime lengths do not exceed true output lengths;
  - a lossier arm does not beat a less lossy one without an explanation;
  - runs with crashed seeds are not averaged with complete ones.
- **Same scorer.** Before comparing results from different directories or
  dates, confirm they used the same scorer, extraction, and harness version.
  Keep the provenance (commit, settings) for every reference band you quote.
- **Repeated lossy transforms.** If production can re-apply the transform, as
  with re-offloaded pages, repeated eviction rounds, or rebuilt
  mixed-precision high windows, evaluate that path too. For cold-page codecs,
  compare raw bit patterns after 1, 2, 4, and 8 round trips, on real KV cache
  data and on extreme values.
- **Configured but inactive changes nothing.** With the feature configured but
  never triggered (no page moved, no round due), outputs must be
  token-identical to the control. If they are not, there is a bug in setup
  or in shared state. Engines sharing GPUs are one cause.
- **Porting a method.** Compare against the reference implementation stage by
  stage and bitwise: scores, selected indices, eviction decisions, and the
  first generated token. Start from one deterministic request. Comparing an
  implementation with itself proves nothing.
- **Provenance.** Tie every score to a recorded tree and recipe: commit,
  uncommitted changes, build, configuration, time. Copy rerun commands
  verbatim from the original log; a retyped command can silently drop a
  setting such as the chat template.
- **Saturated benchmarks.** If a benchmark cannot separate the arms, add a
  harder one, for example more needles or a longer context, instead of
  over-reading small differences.

## Report Template

```markdown
## Accuracy: <feature> vs <reference arms>
- Build: <commit>; local changes: <none | diff hash>
- Model and checkpoint, backend, KV cache dtype, feature settings
- Harness and version, dataset and split, prompt template and thinking mode
- Sampling, samples per prompt, seeds, maximum input and output lengths
- Activation evidence: <counters or log lines, reuse ratios>

| Task | Arm | Score (mean +/- SE) | Delta vs reference (paired, +/- SE) | Truncated | Errors |
|---|---|---|---|---|---|

- Verdict: <within noise | regression of X +/- Y on Z>; scope not tested: <list>
```

## References

- `../perf-kv-cache-tuning/SKILL.md`: engine-per-GPU rules and measurement
  hygiene shared with performance runs.
- `../trtllm-kv-cache-manager-v2/references/lessons-learned.md`: evaluation
  incidents behind these rules.
- `docs/source/developer-guide/kv-cache-compression-development.md`
  ("Validation"): the required order of accuracy, functional, and
  performance evidence for compression methods.
