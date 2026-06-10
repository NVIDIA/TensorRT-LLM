# VisualGen Engineering Criteria

VisualGen aims to deliver **SOL performance** for image and video
generation with **world-class quality**.

In the agentic-coding era, implementing models and features has gotten
dramatically cheaper — but shaping the project around them now takes
a larger share of engineering effort.

This document aligns the high-level engineering principles and
guidelines that hold VisualGen to its **world-class quality** bar, for
engineers ***and the agents*** alike.

## 1. New model support

Adding a HuggingFace checkpoint. See
[`examples/visual_gen/api_walkthrough.py`](../../../examples/visual_gen/api_walkthrough.py) for the
end-to-end workflow.

1. **One PR.** All requirements below land in the same model-enabling
   PR.
2. **Register the pipeline.** Add every supported/tested checkpoint
   to the `hf_ids` list so it surfaces via
   `VisualGen.supported_models()`.
3. **Model-specific knobs.** Route through `pipeline_config` and the
   registered `extra_param_specs` (surfaced as `extra_params` on
   `VisualGenParams`), not top-level `VisualGenArgs` /
   `VisualGenParams` fields (§2).
4. **Example in CI.** At least one example per model series,
   exercised in pre-merge CI.
5. **E2E vs reference framework.** One per task per checkpoint (T2I /
   T2V / TI2V) with LPIPS — or an equivalent metric — at an explicit
   threshold, comparing against a baseline image/video produced by a
   reference framework.

## 2. API

Public surface: `tensorrt_llm/visual_gen/`. The API must be cheap to
extend and cheap to keep stable; apply Occam's Razor.

1. **Team review, not just PR approval.** Public API changes
   (additions, renames, semantic shifts on `VisualGen` /
   `VisualGenArgs` / `VisualGenParams`) are reviewed in a team sync
   before merge — PR approval alone is not sufficient.
2. **General concepts, one-sentence docs.** Top-level fields name
   concepts that generalize across models (`quant_config`,
   `parallel_config`, `attention_config`). A field whose docstring
   doesn't speak for itself is a signal to re-examine the use case,
   motivate it with an example, and discuss case by case.
3. **Model-specific stays model-specific.** Knobs that only make sense
   for one model live in `pipeline_config` / `extra_params` via the
   registry. Promote to a top-level field only after the knob
   generalizes.
4. **Internalize what can be deduced.** If a value is derivable from
   platform/model/workload and the user impact of a wrong guess is
   insignificant, let the engine pick. Example: fused vs unfused kernel
   selection isn't a field.
5. **Pydantic discipline.** Follow
   [`CODING_GUIDELINES.md`](../../../CODING_GUIDELINES.md) § Pydantic
   Guidelines; the `Field(description=...)` is the API doc.
6. **Symmetric offline + serve.** Anything user-visible round-trips
   through both `--visual_gen_args` YAML and `trtllm-serve`.

## 3. Features

Covers performance optimization and other general features.

1. **Reproducibility (perf work).** Document the setup somewhere
   discoverable — PR description, in-repo scripts/configs, or a
   linked report.
2. **Perf evidence (perf work).** Before/after numbers on the
   targeted workloads in the PR; a comprehensive linked perf report
   is preferred. General features without perf claims skip this.
3. **Lossy work.** Quantization, sparsity, approximate kernels —
   trades quality for speed/memory. Report relevant accuracy metric
   numbers in the PR (LPIPS, VBench, or equivalent), plus a linked
   report with generated content alongside the reference for visual
   comparison. The accuracy study and its acceptance bar are
   reviewed in sync meetings before the PR lands.
4. **Lossless work.** Kernel fusion, scheduling, parallelism —
   mathematically equivalent up to floating-point accumulation.
   Report LPIPS against the existing golden; a very small diff is
   acceptable, a larger diff is a red flag.
5. **Knob exposure follows §2.** No new public API surface unless
   adding a knob; always-on optimizations (deducible from
   arch/dtype/workload) stay internal.
6. **Tests.** E2E coverage required for important features and
   feature combinations. See §5 for cross-cutting test discipline.

## 4. Examples & docs

Examples and the public API carry **specifics**, protected by CI so
they stay honest. Docs carry **framing** — introductions, architecture,
when to reach for which knob — material that survives releases.

1. **Examples are tutorials, not tests.** A new example earns its
   place by demonstrating a distinct concept — new model/task, new
   public-API surface, or new serving mode.
2. **Examples are standalone.** Each example is a single file
   calling the public API directly — no CLI wrappers, no helper
   modules, no example-side abstractions. If multiple examples want
   to share code, refactor the API; don't hide it behind an example
   abstraction.
3. **Examples run in CI.** Every example verifies its generated
   content with a metric — examples that aren't tested rot. See §5
   for the test pattern.
4. **Specifics live in code.** The registry, `extra_param_specs`,
   and `default_params` are the source of truth; docs point at the
   API rather than enumerate. Putting specifics in docs is the rot
   path — TRT-LLM's
   [supported-models list](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html#id32)
   is the cautionary tale.
5. **Developer guides co-locate with code.** Long-form subsystem
   guidance lives next to the code it governs (precedent:
   [`ATTENTION_DEVELOPER_GUIDE.md`](../modules/ATTENTION_DEVELOPER_GUIDE.md),
   [`MOE_DEVELOPER_GUIDE.md`](../modules/fused_moe/MOE_DEVELOPER_GUIDE.md)).

## 5. Tests

Test scope is decided per-PR but is always required; the PR ships its
own tests. Tests scale with feature scope, not lines of code.

Where tests live:

| Test category | Target directory |
| --- | --- |
| Example tests | `tests/integration/defs/examples/visual_gen/` |
| E2E tests | `tests/integration/defs/visual_gen/` |
| Perf tests | `tests/integration/defs/perf/` |
| Unit tests (public) | `tests/unittest/visual_gen/` |
| Unit tests (internal) | `tests/unittest/_torch/visual_gen/` |

1. **Tiers.** A cheap **sanity** check (content was generated, valid
   shape, non-black frame) runs broadly; the **quality gate** (LPIPS
   / VBench against a reference) runs where quality matters.
2. **Keep E2E cheap.** Reduced denoising steps and resolution are
   typical (FLUX 4 steps, LTX-2 8 steps, 256×256). Variant matrices
   stay offline.
3. **Drop redundant CI checks.** If a check costs hours per PR to
   defend against something a higher-level test already catches, drop
   it.
