# VisualGen API Refactor — Design Discussion

> **Status**: Draft — under active discussion
> **Authors**: Inference Engine Team
> **Date**: 2026-03-18
> **Codebase Ref**: [`e71a200`](https://github.com/NVIDIA/TensorRT-LLM/commit/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Landscape Survey (Compact)](#2-landscape-survey)
3. [Phase 1 — Engine Initialization](#3-phase-1--engine-initialization)
4. [Phase 2 — Request Construction](#4-phase-2--request-construction)
5. [Phase 3 — Execution](#5-phase-3--execution)
6. [Phase 4 — Output & Post-Processing](#6-phase-4--output--post-processing)
7. [Phase 5 — Lifecycle Management & Observability](#7-phase-5--lifecycle-management--observability)
8. [Cross-Cutting: Streaming Readiness](#8-cross-cutting-streaming-readiness)
9. [Cross-Cutting: Module & Directory Structure](#9-cross-cutting-module--directory-structure)
10. [Cross-Cutting: Naming Conventions](#10-cross-cutting-naming-conventions)
11. [Proposed API Shape (End-to-End)](#11-proposed-api-shape-end-to-end)
12. [Summary of Recommendations](#12-summary-of-recommendations)
13. [Open Questions](#13-open-questions)
14. [Appendix: Framework Source Code References](#appendix-framework-source-code-references)

---

## 1. Executive Summary

TensorRT-LLM's VisualGen API is marked `prototype`. Before graduating to `stable`, we must address design issues across the full lifecycle — from engine init through request processing to output handling. This document organizes the discussion along that lifecycle, benchmarks against diffusers, SGLang Diffusion, and vLLM-omni (with source code references), and proposes directions.

**Design principles:**

- **Familiar to the ecosystem**: Users from diffusers/SGLang/vLLM should feel at home.
- **Lifecycle clarity**: Engine config (how to run) → request params (what to generate) → output (what you get back). Each phase has a clear, separate API surface.
- **Extensibility without explosion**: New models shouldn't touch the core API contract.
- **Streaming-ready**: Shapes should accommodate progressive output without redesign.
- **Minimal surprise**: Defaults come from the model, not hardcoded in params.
- **Symmetry with the LLM API**: Where patterns from [`LLM`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L132) / [`SamplingParams`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/sampling_params.py) / [`RequestOutput`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L55) apply, reuse them. Where they don't (diffusion is not autoregressive), diverge deliberately.

---

## 2. Landscape Survey

### Compact Comparison

| Aspect | Diffusers | SGLang Diffusion | vLLM-Omni | TRT-LLM (current) |
|--------|-----------|------------------|-----------|-------------------|
| **Engine init** | `Pipeline.from_pretrained(model)` | `Server(model_path=...)` | `Omni(model=...)` | `VisualGen(model_path=..., diffusion_args=...)` |
| **Engine config** | Pipeline init kwargs | `ServerArgs` + `PipelineConfig` | `EngineArgs` | `VisualGenArgs` (Pydantic) |
| **Request params** | Flat `__call__(**kwargs)` | `SamplingParams` dataclass (model subclasses) | `OmniDiffusionSamplingParams` (mega-dataclass) | `VisualGenParams` dataclass |
| **Model-specific defaults** | Per-pipeline `__call__` defaults | `ClassVar` on subclasses (`_default_height`) | Pass-through (None → model decides) | Hardcoded on `VisualGenParams` |
| **Model-specific params** | Per-pipeline kwargs | Fields on model subclass | `extra_args: dict` | Flat fields on shared dataclass |
| **Output type** | PIL Image (default) / np / pt | File path / frames | Base64 PNG (OpenAI) | Raw `torch.Tensor` |
| **Encoding utils** | `export_to_video()` in utils | Built into `SamplingParams` | In API server layer | `MediaStorage` in `serve/` |
| **Input type** | `prompt: str \| list[str]` | `SamplingParams.prompt` | `list[OmniPromptType]` | `VisualGenInputs` (union type) |
| **Prompt in params?** | Yes (kwarg to `__call__`) | Yes (field on `SamplingParams`) | Separate (`OmniDiffusionRequest.prompts`) | Separate (`inputs` arg) |
| **Batch API** | `prompt=["a", "b"]` (list) | Per-request | `OmniDiffusionRequest.prompts` list | `VisualGenInputs` accepts `Sequence` |

### Critical Lessons from Other Frameworks

1. **SGLang's default-value bug** ([Issue #20078](https://github.com/sgl-project/sglang/issues/20078), Mar 2026): Generic defaults overrode model-specific ones, causing `guidance_scale=7.5` instead of `0.0` for distilled models. **Our current hardcoded defaults in `VisualGenParams` create the same risk.**

2. **vLLM-omni's mega-dataclass anti-pattern**: [`OmniDiffusionSamplingParams`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/inputs/data.py) has ~100+ fields mixing user params with runtime state (`latents`, `timesteps`, `step_index`, `past_key_values`). Convenient internally; terrible as a public API. **We must keep request wire-types separate from user-facing params.**

3. **Diffusers' prompt-in-kwargs simplicity**: Users write `pipeline(prompt="cat", height=512)`. No separate input type. The prompt IS a parameter. This is the simplest API in the ecosystem.

Source code references in [Appendix](#appendix-framework-source-code-references).

---

## 3. Phase 1 — Engine Initialization

### Current State

```python
# Current API
visual_gen = VisualGen(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    diffusion_args=VisualGenArgs(parallel=ParallelConfig(dit_cfg_size=2)),
)
```

[Source: `VisualGen.__init__`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L474-L490)

Internally: `VisualGen.__init__` → creates `DiffusionRemoteClient` → spawns worker processes → each worker runs `PipelineLoader.load()` → `DiffusionModelConfig.from_pretrained()` → `AutoPipeline.from_config()` → `BasePipeline.__init__` → `warmup()` → sends READY signal.

### Issues

#### 3.1 `VisualGenArgs` is in `_torch/` (private package) but user-facing

[`VisualGenArgs`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/config.py#L298) lives in `tensorrt_llm/_torch/visual_gen/config.py`. The `_torch` prefix implies private, but users must import and instantiate `VisualGenArgs` to configure the engine. Its sub-configs (`ParallelConfig`, `CompilationConfig`, `AttentionConfig`, etc.) are also user-facing.

This is inconsistent with the LLM API pattern where [`LlmArgs`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm_args.py#L2371) (and `TorchLlmArgs`, `TrtLlmArgs`) lives in `tensorrt_llm/llmapi/llm_args.py` — a **public** module, not under `_torch/`. Putting user-facing config in a `_`-prefixed module sends the wrong signal to contributors and violates the convention that `_torch/` is internal implementation.

**What's in `config.py` today:**

| Class | User-facing? | Should be public? |
|-------|:---:|:---:|
| `VisualGenArgs` | Yes — users instantiate it | Yes |
| `ParallelConfig` | Yes — users tune parallelism | Yes |
| `CompilationConfig` | Yes — users configure warmup shapes | Yes |
| `TorchCompileConfig` | Yes — users enable/disable torch.compile | Yes |
| `CudaGraphConfig` | Yes — users enable/disable CUDA graphs | Yes |
| `PipelineConfig` | Yes — users configure offloading | Yes |
| `AttentionConfig` | Yes — users select attention backend | Yes |
| `TeaCacheConfig` | Yes — users enable/configure TeaCache | Yes |
| `PipelineComponent` | Yes — users specify skip_components | Yes |
| `DiffusionModelConfig` | No — created internally by `PipelineLoader` | No |

**Recommendation**: Move user-facing config classes out of `_torch/` to a public module, mirroring the LLM API pattern:

**Option A** (parallel to `llm_args.py`): Create `llmapi/visual_gen_args.py`

```
tensorrt_llm/llmapi/
├── llm_args.py            # LlmArgs, TorchLlmArgs, TrtLlmArgs, KvCacheConfig, ...
├── visual_gen_args.py     # NEW: VisualGenArgs, ParallelConfig, CompilationConfig, ...
├── visual_gen.py          # VisualGen, VisualGenParams
└── ...
```

This is the most consistent approach: `llm_args.py` ↔ `visual_gen_args.py`. Users import the same way: `from tensorrt_llm.llmapi import VisualGenArgs, ParallelConfig`.

**Option B** (if we later create `visualgenapi/`): Move to `visualgenapi/args.py`

```
tensorrt_llm/visualgenapi/
├── args.py                # VisualGenArgs + sub-configs
├── visual_gen.py          # VisualGen
└── ...
```

**In both cases**: `DiffusionModelConfig` (internal) stays in `_torch/visual_gen/config.py`. The public classes are moved out; the internal config remains in the internal module.

**Re-export**: `tensorrt_llm/__init__.py` continues to export `from .llmapi import VisualGenArgs, ParallelConfig, ...` so the user import path (`from tensorrt_llm import VisualGenArgs`) doesn't change.

#### 3.2 `VisualGenArgs` mixes engine config with internal fields

`VisualGenArgs` contains both user-facing fields (`checkpoint_path`, `device`, `dtype`, `quant_config`, sub-configs) and implementation details (`skip_warmup`, `skip_components`, `force_dynamic_quantization`). The sub-configs ([`CompilationConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/config.py#L229), [`TorchCompileConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/config.py#L207), etc.) are genuinely user-facing for production tuning.

**Recommendation**: Acceptable as-is for the field mix — `skip_warmup` and `skip_components` are arguably user-facing (advanced users need them). The Pydantic `StrictBaseModel` with `extra="forbid"` provides good guardrails. Document which sub-configs matter for typical vs advanced users.

However, `VisualGenArgs` currently defines `to_dict()` and `from_dict()` (lines 403–414 of `config.py`), which violate `CODING_GUIDELINES.md`: *"Avoid defining `to_dict()` methods — prefer Pydantic's built-in `model_dump()`. Avoid defining `from_dict()` / `from_kwargs()` methods — prefer constructing the class directly."* These should be removed; callers should use `args.model_dump()` and `VisualGenArgs(**config_dict)` directly. `from_yaml()` is fine (the guidelines only prohibit `from_dict` / `from_kwargs`).

#### 3.3 Constructor naming: `diffusion_args` → ?

**Motivation**: The parameter name `diffusion_args` leaks the implementation detail that these are "diffusion" models, which is part of the broader "Diffusion" prefix elimination (§10.3).

| Option | Constructor | Rationale |
|--------|------------|-----------|
| **A — `args`** | `VisualGen(model_path="...", args=VisualGenArgs(...))` | Short, matches the type name (`VisualGenArgs`). Consistent with common Python patterns. |
| **B — `config`** | `VisualGen(model_path="...", config=VisualGenArgs(...))` | Semantic — "config" communicates engine configuration. But the type is `*Args`, not `*Config`, creating a slight mismatch. |
| **C — `**kwargs`** | `VisualGen(model_path="...", parallel=ParallelConfig(...))` | Matches `LLM(model=..., **kwargs)` where kwargs are fields of `LlmArgs`. More ergonomic but loses explicit `VisualGenArgs` construction. |

**Recommendation**: **Option A — `args`**. It's the simplest rename and matches the type name. Option C is a larger design change that can be considered later.

#### 3.4 Constructor naming: `model_path` → `model`

LLM uses `model=`. VisualGen uses `model_path=`. The `_path` suffix suggests only local file paths are accepted, but HuggingFace Hub IDs work too (e.g. `"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"`). The name `model` is also the standard parameter in diffusers (`Pipeline.from_pretrained(model)`), SGLang (`Server(model_path=...)`), and vLLM (`Omni(model=...)`).

| Option | Constructor | Matches |
|--------|------------|---------|
| **Keep `model_path`** | `VisualGen(model_path="Wan-AI/...")` | SGLang's `model_path` |
| **Rename to `model`** | `VisualGen(model="Wan-AI/...")` | LLM API, diffusers, vLLM |

**Recommendation**: Rename to `model` for consistency with `LLM(model=...)` and the broader ecosystem. If backward compatibility is needed, keep `model_path` as a deprecated alias.

---

## 4. Phase 2 — Request Construction

This is the most contentious area. Three interconnected questions:
1. Should prompt and params be separate arguments or a single request object?
2. How do we handle model-specific parameters without the dataclass exploding?
3. What should default values be?

### 4.1 Input + Params: Merge or Separate?

#### Current State

```python
output = visual_gen.generate(
    inputs="A cat sitting on a windowsill",  # VisualGenInputs
    params=VisualGenParams(height=480, width=832, ...),  # separate params
)
```

[`VisualGenInputs`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/inputs/data.py#L107-L110) is `Union[VisualGenPromptInputs, Sequence[VisualGenPromptInputs]]`, where `VisualGenPromptInputs` is `Union[str, List[int], VisualGenTextPrompt, VisualGenTokensPrompt]`.

The prompt types `VisualGenTextPrompt` and `VisualGenTokensPrompt` add `negative_prompt` on top of the LLM types. But that's essentially the only extra field — the input type is very thin.

#### How Others Handle This

| Framework | Pattern | Prompt in params? |
|-----------|---------|-------------------|
| **Diffusers** | `pipeline(prompt="cat", height=512, ...)` | Yes — flat kwargs |
| **SGLang** | `SamplingParams(prompt="cat", height=480, ...)` | Yes — field on params |
| **vLLM-omni** | `OmniDiffusionRequest(prompts=[...], sampling_params=...)` | No — separate |
| **LLM API** | `llm.generate(inputs="text", sampling_params=SamplingParams(...))` | No — separate |

For **LLM**, separating prompt from params makes sense because `SamplingParams` is genuinely reusable — you often generate 100 different prompts with the same temperature/top_p. The prompt is the "input data"; params are "control knobs".

For **visual generation**, the picture is different:
- The "prompt" (text) is just one of potentially many inputs: text, negative text, reference image, mask, last frame.
- Resolution, num_frames, and seed are often per-request (unlike LLM where temperature is shared).
- The distinction between "input" (the creative content) and "params" (the technical config) is blurrier.

#### Discussion: Two Options

> **Note on return type**: The signatures below use `VisualGenOutput`, a proposed wrapper around `MediaOutput` that adds request-level metadata. See §6.1 for the motivation, design, and naming discussion.

**Option A: Merge into a single request object**

One self-contained object per request:

```python
def generate(
    self,
    request: Union[VisualGenRequest, List[VisualGenRequest]],
) -> Union[VisualGenOutput, List[VisualGenOutput]]:
```

```python
@dataclass
class VisualGenRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    image: Optional[Union[str, List[str]]] = None
    mask: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    num_frames: Optional[int] = None
    frame_rate: Optional[float] = None
    num_images_per_prompt: int = 1
    extra_params: Optional[dict] = None
```

Pros: Each request is self-contained with its own resolution/seed. Matches how diffusers works conceptually. No separate input type. Clean for batch with different params per item.
Cons: Can't share params across requests (minor — create a helper). Departs from the LLM `generate(inputs, sampling_params)` pattern.

**Option B: Keep separate, but prompt is just `str` + conditioning lives on params**

```python
def generate(
    self,
    prompt: Union[str, List[str]],
    params: Optional[VisualGenParams] = None,
) -> Union[VisualGenOutput, List[VisualGenOutput]]:
```

Where `VisualGenParams` holds negative_prompt, image, mask along with other params. The first arg is purely the text prompt.

Pros: Simplest signature. `prompt` is always a string (or list of strings). Everything else is in params.
Cons: Negative prompt is semantically an "input", not a "param". But diffusers treats it the same way (`pipeline(prompt=..., negative_prompt=...)`), so users expect it.

#### Recommendation

**Option B is recommended** — keep prompt and params separate with the simplest possible signature:

1. **`prompt` is always `str` (or `List[str]`)**: No separate `VisualGenPrompt` type. Text prompt is the first-class input; conditioning inputs (image, mask, negative_prompt) live on `VisualGenParams` alongside generation knobs, matching how diffusers handles them.

2. **`params` defaults to `None`** (meaning use model defaults): This gives the ergonomic `visual_gen.generate("A cat")` one-liner.

3. **Support both single and batch**: `prompt` accepts `str` or `list`, matching [`LLM.generate()`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L289-L306) which also accepts `Union[PromptInputs, Sequence[PromptInputs]]`.

4. **`params` can be single or list**: Single params shared across batch, or per-request params list. Again matches [`LLM.generate()`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L292-L293) which accepts `Optional[Union[SamplingParams, List[SamplingParams]]]`.

```python
# Minimal
result = visual_gen.generate("A cat on a windowsill")

# With params
result = visual_gen.generate("A cat", params=VisualGenParams(height=480, width=832, seed=42))

# Batch with shared params
results = visual_gen.generate(["A cat", "A dog"], params=VisualGenParams(seed=42))

# Batch with per-request params
results = visual_gen.generate(
    ["A cat", "A dog"],
    params=[VisualGenParams(height=480), VisualGenParams(height=720)],
)

# With conditioning (image/mask on params, not on prompt)
result = visual_gen.generate(
    "Make it snow",
    params=VisualGenParams(image="summer.png", num_frames=81),
)
```

### 4.2 Default Values: `None` = Model Default

#### Current Problem

[`VisualGenParams`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L433-L438) hardcodes: `height=720`, `width=1280`, `num_inference_steps=50`, `guidance_scale=5.0`, `max_sequence_length=512`, `seed=42`. These are wrong for most models:

| Model | Ideal height | Ideal width | guidance_scale | num_inference_steps |
|-------|-------------|-------------|----------------|---------------------|
| Wan 1.3B | 480 | 832 | 5.0 | 50 |
| FLUX.2 | 1024 | 1024 | 0.0 (distilled) | 28 |
| LTX-2 | 768 | 1280 | varies | varies |

#### Recommendation

Make most fields default to `None`. Semantics: `None` = "use what the model and engine config define".

```python
@dataclass
class VisualGenParams:
    # Core generation params — None means model default
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None                  # see seed discussion below
    max_sequence_length: Optional[int] = None

    # Video params
    num_frames: Optional[int] = None            # None → model default (1 for image, 81 for video)
    frame_rate: Optional[float] = None

    # Common optional params
    negative_prompt: Optional[str] = None
    num_images_per_prompt: int = 1              # 1 is always sensible as default

    # Model-specific params (see §4.3)
    extra_params: Optional[Dict[str, Any]] = None
```

#### 4.2.1 `seed` Default: Deterministic vs Random

The current default is `seed=42` (deterministic). Making `seed=None` (random) the default means every existing user who relied on reproducible output — even unknowingly — gets different results on every run. This is exactly the kind of "minimal surprise" violation the design principles call out.

| Option | Default | Behavior |
|--------|---------|----------|
| **Keep `seed=42`** | Deterministic | Safe for prototype→stable transition; user must opt into randomness |
| **`seed=None` (random)** | Non-deterministic | Matches diffusers convention; more "production-like" |
| **`seed=0`** | Deterministic but arbitrary | Signals "we pick a seed" without implying `42` is special |

**Recommendation**: Keep a deterministic default (either `42` or `0`) for the graduation. Require `seed=None` for explicit randomness. The `seed_used` field on `VisualGenOutput` (§6.1) makes the actual seed discoverable either way.

#### 4.2.2 Remove `output_type` from `VisualGenParams`

The current `VisualGenParams.output_type: str = "pt"` allows users to choose output format. However, the proposed output model (`MediaOutput` with `to_pil()`, `to_bytes()`, `save()` methods, §6.2) makes this field redundant — `MediaOutput` always returns tensors, and conversion is a post-processing concern on the output object, not a request parameter.

Keeping both creates confusion: does `output_type` change what `MediaOutput.video` contains, or is it ignored?

**Recommendation**: Remove `output_type` from `VisualGenParams`. Conversion belongs on the output object, not the request params.

**Resolution flow**: `User-specified (explicit) → Model pipeline defaults → Hardcoded fallback (last resort)`

Each `BasePipeline` subclass implements `default_generation_params() → dict` that returns its model-specific defaults. The executor merges before calling `pipeline.infer()`.

**Default resolution at request time**: When `params=None` or individual fields are `None`, the executor resolves defaults by calling `self.pipeline.default_generation_params()` on the worker side before invoking `pipeline.infer()`. This keeps model-specific defaults co-located with the model pipeline implementation.

### 4.3 Model-Specific Params: `extra_params` Discoverability

#### The Problem

With `extra_params: dict`, how does the user know:
- What keys are valid for the loaded model?
- What types and ranges are expected?
- What the defaults are?

#### How Others Handle This

| Framework | Approach | Discoverability |
|-----------|----------|-----------------|
| **Diffusers** | Per-pipeline `__call__` docstring + type hints | IDE completion ✓, but only if you know the pipeline class |
| **SGLang** | Model subclass with typed fields | IDE completion ✓, but must import the right subclass |
| **vLLM-omni** | "Incompatible values will result in errors from the underlying pipeline" | No discoverability ✗ |

#### Recommendation: Three-Layer Approach

The examples below use a type called `ExtraParamSchema` to describe each extra parameter's type, default, range, and description. Before showing the design, here is the naming discussion for this type.

#### Naming: `ExtraParamSchema`

| Option | Name | Rationale |
|--------|------|-----------|
| **A** | `ParamSpec` | Short and descriptive. But **collides with `typing.ParamSpec`** (PEP 612), a well-known Python typing construct with completely different semantics. Would confuse static analysis tools and developers. |
| **B** | `ExtraParamSchema` | Descriptive — it's a schema describing an extra parameter. No collisions. |
| **C** | `ParamDescriptor` | Descriptive, but "descriptor" has its own Python meaning (`__get__`/`__set__` protocol). |
| **D** | `ParamDefinition` | Clear, but less conventional in API contexts where "schema" is the standard term. |

**Recommendation**: **Option B — `ExtraParamSchema`**. Avoids the `typing.ParamSpec` collision, is self-describing, and follows the convention of "schema" for type/validation metadata.

#### Three-Layer Design

**Layer 1 — Runtime introspection (primary)**:

```python
engine = VisualGen(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Discover supported extra params for the loaded model
schema = engine.supported_extra_params
# Returns: {
#     "guidance_scale_2": ExtraParamSchema(type=float, default=None, description="Second guidance scale for dual-guidance models"),
#     "boundary_ratio": ExtraParamSchema(type=float, default=None, range=(0.0, 1.0), description="..."),
# }

# Use them
result = engine.generate("A dragon", params=VisualGenParams(
    extra_params={"guidance_scale_2": 3.0}
))
```

Each `BasePipeline` subclass declares its supported extra params via a class method:

```python
class WanPipeline(BasePipeline):
    @classmethod
    def extra_param_specs(cls) -> dict[str, ExtraParamSchema]:
        return {
            "guidance_scale_2": ExtraParamSchema(type=float, default=None, description="..."),
            "boundary_ratio": ExtraParamSchema(type=float, default=None, range=(0.0, 1.0)),
        }
```

**Layer 2 — Validation at request time**:

When `extra_params` is provided, the executor validates keys against the pipeline's declared specs before forwarding. Unknown keys → clear error message listing valid keys. Wrong types → clear error with expected type.

**Layer 3 — Documentation**:

Each model's deployment guide (`docs/source/deployment-guide/`) documents its `extra_params` with examples.

**Future evolution**: If a model-specific param becomes universally needed (e.g., if all models end up supporting `guidance_scale_2`), promote it to a typed field on `VisualGenParams`. The `extra_params` dict is the extensibility escape hatch, not the long-term home.

### 4.4 Additional Design Issue: Internal Wire Format Duplication

Currently, [`generate_async()`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L572-L598) manually copies 20+ fields from `VisualGenParams` to `DiffusionRequest`. Every time a field is added to `VisualGenParams`, it must also be added to `DiffusionRequest` and the copy code.

**Recommendation**: The internal `DiffusionRequest` should embed or reference the params directly:

```python
@dataclass
class DiffusionRequest:
    request_id: int
    prompt: List[str]
    negative_prompt: Optional[str]
    params: VisualGenParams  # embed the whole object instead of field-by-field copy
    image: Optional[Union[str, List[str]]] = None
    mask: Optional[str] = None
```

This eliminates the field-by-field copy and ensures the internal request always carries the full params. The pipeline's `infer()` method reads from `request.params` instead of top-level fields.

> **Future work**: The `Diffusion` prefix on internal types (`DiffusionRequest`, `DiffusionResponse`, `DiffusionExecutor`, `DiffusionModelConfig`, etc.) leaks an implementation detail while the product brand is `VisualGen`. No peer framework uses "Diffusion" as a prefix either — SGLang uses `GenerateReqInput`, vLLM-omni uses `OmniDiffusionRequest`, TRT-LLM LLM uses `GenerationRequest`. These internal types should be renamed to `VisualGen*` in a follow-up pass once the API-facing renames are settled. See §10.3 for the full rename map.

---

## 5. Phase 3 — Execution

### 5.1 `generate()` Signature

Following the recommendations from §4:

```python
class VisualGen:
    def generate(
        self,
        prompt: Union[str, List[str]],
        params: Optional[Union[VisualGenParams, List[VisualGenParams]]] = None,
    ) -> Union[VisualGenOutput, List[VisualGenOutput]]:
        """Synchronous generation.

        Args:
            prompt: Text prompt string or list of prompt strings.
            params: Generation parameters (including conditioning inputs
                    like image, mask, negative_prompt). None = model defaults.
                    Single params shared across batch, or per-request list.

        Returns:
            Single VisualGenOutput or list (matching prompt shape).
        """

    def generate_async(
        self,
        prompt: str,
        params: Optional[VisualGenParams] = None,
    ) -> VisualGenResult:
        """Async generation for a single request. Returns immediately."""
```

Note: `generate_async` is single-request only (matching [`LLM.generate_async`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L374-L387)). `generate()` handles batching by calling `generate_async` per item internally. The return type `VisualGenResult` (renamed from `DiffusionGenerationResult`) is discussed in §5.2.

#### 5.1.1 Batch Error Semantics

If 2 of 5 batch requests fail, what happens?

**Option A — Raise on first error (current behavior):** `generate()` raises `VisualGenError` immediately. Simpler, but the caller loses partial results. Matches the current behavior where a `RuntimeError` is raised if `response.error_msg` is set.

**Option B — Return all with per-item errors:** `generate()` always returns a full list. Failed items have `VisualGenOutput.error` set and `VisualGenOutput.output` is `None`. Callers must check each result. More robust for production batch workflows.

**Edge case**: When `params` is a list but its length doesn't match `inputs`, this should be a validation error at the `generate()` boundary, not deep in the executor.

**Recommendation**: Option B is more aligned with the LLM API's approach (each `RequestOutput` carries its own state). At minimum, the chosen behavior must be documented in the `generate()` docstring.

### 5.2 Async Handle: Renaming `DiffusionGenerationResult`

The [current async handle](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L366-L405) is called `DiffusionGenerationResult`. Two issues:

1. **"Diffusion" prefix** — being eliminated across the codebase (see §10.3). The product brand is `VisualGen`.
2. **"Result" suffix** — the object is an async handle (a future), not a result itself. `result()` is a *method* on the handle. The LLM API uses `GenerationResult` for this role, but "Result" conflates the handle with the value it produces.

| Option | Name | Rationale |
|--------|------|-----------|
| **A** | `VisualGenResult` | Matches LLM API's `GenerationResult` pattern. But inherits the "Result" ambiguity. |
| **B** | `VisualGenResult` | Clarifies async semantics — this is a future/handle, not the result itself. Matches Python's `asyncio.Future` convention. |
| **C** | `VisualGenGenerationResult` | Verbose. Adds nothing over option A. |
| **D** | `VisualGenAsyncHandle` | Descriptive but unusual — no Python library uses "Handle" for this pattern. |

**Recommendation**: **Option A — `VisualGenResult`**. This matches the established convention: TRT-LLM LLM uses `GenerationResult`, SGLang uses `GenerationResult` — neither highlights async behavior in the type name despite both being async handles. Consistency with peer frameworks outweighs the semantic precision of `Future`. The `VisualGen*` prefix avoids collision with the LLM API's `GenerationResult`.

#### Interface Design

Consider aligning the interface with Python's `asyncio` conventions:

```python
class VisualGenResult:
    """Future-like object for async generation."""

    async def result(self, timeout: Optional[float] = None) -> VisualGenOutput:
        """Await the result."""

    def result_sync(self, timeout: Optional[float] = None) -> VisualGenOutput:
        """Blocking wait for result (convenience for non-async code)."""

    def cancel(self) -> bool:
        """Attempt to cancel the request. Returns True if successfully cancelled."""

    @property
    def done(self) -> bool:
        """Whether generation has completed."""
```

### 5.3 Additional Issue: No Progress Feedback

Video generation can take minutes. Users have no visibility into progress. Before full streaming, a progress callback is a lightweight solution:

```python
result = visual_gen.generate(
    "A dragon flying over mountains",
    params=VisualGenParams(num_inference_steps=50),
    on_progress=lambda step, total: print(f"Step {step}/{total}"),
)
```

**Recommendation**: Add an optional `on_progress: Optional[Callable[[int, int], None]]` parameter to `generate()`. The executor periodically reports denoising step progress. This is much simpler than full streaming and covers the most common need (progress bars).

However, this only covers the sync path. Async users — including the serving layer — get no progress visibility. Three options for extending to async:

**Option A — Add `on_progress` to `VisualGenResult`:**

```python
future = engine.generate_async("A dragon", params=params)
future.on_progress = lambda step, total: ...
```

**Option B — Accept async callbacks in `generate_async`:**

```python
async def generate_async(
    self,
    inputs: ...,
    params: ...,
    on_progress: Optional[Callable[[int, int], Union[None, Awaitable[None]]]] = None,
) -> VisualGenResult:
```

**Option C — Defer to the streaming design:**

Progress is fundamentally streaming information. If the streaming API (§8) is near-term, the callback may be unnecessary complexity. `stream()` would yield intermediate `VisualGenOutput` objects with a `progress: float` field, covering both sync and async use cases.

Regardless of the option chosen, the callback threading model must be documented: which thread does the callback execute on (worker thread? event loop thread? caller thread?)? A blocking callback on the event loop thread would stall all requests.

---

## 6. Phase 4 — Output & Post-Processing

### 6.1 Output Object Design

#### Current State

[`generate()`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L492-L518) returns [`MediaOutput`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/output.py#L10-L29) directly. [`DiffusionResponse`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/executor.py#L63-L74) wraps it internally but is discarded — the user never sees `request_id`, `error_msg`, or any metadata.

Compare with LLM's [`RequestOutput`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L55) which carries `request_id`, `prompt`, `prompt_token_ids`, `outputs`, `finished`, `context_logits`.

#### Recommendation: Wrap `MediaOutput` in a Request-Level Object

**Motivation**: `MediaOutput` is a raw tensor container — it has no request metadata (request ID, seed used, timing, error state, streaming progress). The LLM API's `RequestOutput` wraps `CompletionOutput` with exactly this kind of metadata. Without a wrapper, the VisualGen API forces users to track request↔result correlation externally and loses production-essential data like latency breakdowns.

**Proposed design**:

```python
@dataclass
class VisualGenOutput:
    """Result of a single visual generation request."""

    request_id: int
    output: MediaOutput                         # the generated media
    finished: bool = True                       # streaming-readiness (always True for now)
    error: Optional[str] = None                 # error message if generation failed
    seed_used: int = 0                          # actual seed (useful when input seed is None/random)
    prompt: Optional[str] = None                # echo back the input prompt (for logging/tracing)
    metrics: Optional[VisualGenMetrics] = None   # performance metrics

@dataclass
class VisualGenMetrics:
    queue_ms: float = 0.0          # time spent waiting in queue
    preprocess_ms: float = 0.0     # input preparation (e.g., text encoding)
    inference_ms: float = 0.0      # core generation time
    postprocess_ms: float = 0.0    # output conversion (e.g., latent decoding)
    total_ms: float = 0.0          # end-to-end
```

Why these fields:
- `seed_used`: When user passes `seed=None` (random), they need to know the actual seed to reproduce results.
- `prompt`: Useful for correlating results in batch scenarios and logging.
- `metrics`: Essential for production performance analysis. The pipeline already has timing data internally. The breakdown (`queue_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`) uses abstract names that don't leak pipeline internals — `preprocess_ms` covers input preparation (e.g., text encoding) and `postprocess_ms` covers output conversion (e.g., latent decoding), so the fields stay meaningful across different backends and modalities. And also enables benchmarking per-step metric from the user side.
- `finished`: For future streaming. Today it's always `True`.

#### Naming: `VisualGenOutput` and `VisualGenMetrics`

**For the wrapper type:**

| Option | Name | Rationale |
|--------|------|-----------|
| **A** | `GenerationOutput` | Generic. But collides with potential LLM API types — `GenerationResult` already exists there. |
| **B** | `RequestOutput` | Matches LLM API naming. But would collide with `tensorrt_llm.llmapi.RequestOutput` if both are imported. |
| **C** | `VisualGenOutput` | Brand-prefixed. Avoids all collisions. Consistent with `VisualGenArgs`, `VisualGenParams`. |
| **D** | `MediaGenerationOutput` | Modality-specific but verbose. |

**Recommendation**: **Option C — `VisualGenOutput`**. The `VisualGen*` prefix avoids namespace collisions in a codebase where users commonly `from tensorrt_llm import LLM, VisualGen`. The LLM API already has `RequestOutput` and `GenerationResult`, so generic names risk ambiguity. `MediaOutput` stays as-is — it's already modality-specific and doesn't collide.

**For the metrics type:**

| Option | Name | Rationale |
|--------|------|-----------|
| **A** | `GenerationMetrics` | Generic. Could be confused with LLM metrics. |
| **B** | `VisualGenMetrics` | Consistent with the `VisualGen*` prefix convention. Aligns with SGLang (`RequestMetrics`) and vLLM-Omni (`RequestE2EStats`) naming — `metrics` is the ecosystem-standard term for this data, which naturally extends beyond pure timing to include memory, throughput, etc. |

**Recommendation**: **Option B — `VisualGenMetrics`**. Follows the same prefix convention as `VisualGenOutput`.

### 6.2 `MediaOutput` Convenience Methods & Encoding Format

#### Current State

Users must import [`MediaStorage`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/serve/media_storage.py#L510) from `tensorrt_llm.serve.media_storage` and call static methods with the correct arguments:

```python
from tensorrt_llm.serve.media_storage import MediaStorage
MediaStorage.save_video(output.video, "output.avi", frame_rate=params.frame_rate)
```

#### Recommendation: Methods on `MediaOutput` with explicit format control

```python
@dataclass
class MediaOutput:
    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None   # carries frame_rate, height, width, etc.

    def save(
        self,
        path: str,
        format: Optional[str] = None,    # None → infer from extension
        frame_rate: Optional[float] = None,  # override metadata
        quality: int = 95,               # for lossy image formats (JPEG/WEBP)
        codec: Optional[str] = None,     # for video: 'h264', 'mjpeg', etc.
    ) -> str:
        """Save the output to a file.

        Format is inferred from the file extension by default:
        - '.png', '.jpg', '.webp' → image save
        - '.mp4', '.avi' → video save (with optional audio muxing)

        Args:
            path: Output file path.
            format: Explicit format override (e.g. 'mp4', 'png').
                    If None, inferred from path extension.
            frame_rate: Frame rate for video. If None, uses metadata
                        from generation (e.g. 24.0).
            quality: Quality for lossy formats (1-100).
            codec: Video codec. If None, uses best available.

        Returns:
            Path where the file was saved.
        """

    def to_pil(self) -> "PIL.Image.Image":
        """Convert image output to PIL Image. Raises if no image data."""

    def to_bytes(
        self,
        format: str = "png",   # REQUIRED — no inference from path here
        frame_rate: Optional[float] = None,
        quality: int = 95,
    ) -> bytes:
        """Encode output to bytes buffer.

        Args:
            format: Encoding format ('png', 'jpg', 'webp' for images;
                    'mp4', 'avi' for video).
            frame_rate: Frame rate for video encoding.
            quality: Quality for lossy formats.

        Returns:
            Encoded bytes.
        """
```

Key design decisions:
- **`save()` infers format from extension** (like [`MediaStorage.save_video`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/serve/media_storage.py#L660-L663) already does), with `format` override.
- **`to_bytes()` requires explicit format** — no path to infer from, so must be explicit.
- **`metadata` dict carries `frame_rate`** from generation, so `save("output.mp4")` works without specifying frame_rate again. User can override.
- **Audio muxing is automatic**: If `video` and `audio` are both populated, `save("output.mp4")` muxes them together (MP4+AAC). This matches user expectation for LTX-2 outputs.
- **Dependencies are lazy-imported**: PIL, ffmpeg are imported only when `save()`/`to_pil()`/`to_bytes()` are called.

Updated example:

```python
# Before (current)
from tensorrt_llm.serve.media_storage import MediaStorage
MediaStorage.save_video(output.video, "output.avi", frame_rate=params.frame_rate)

# After
result = visual_gen.generate("A cat on a windowsill")
result.output.save("cat.mp4")                          # infers format, uses model's frame_rate
result.output.save("cat.avi", codec="mjpeg")            # explicit codec
result.output.to_bytes(format="mp4", quality=80)        # for serving
result.output.to_pil()                                  # for image models
```

### 6.3 Where to Put the Encoding Implementation

`MediaStorage` currently lives in [`tensorrt_llm/serve/media_storage.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/serve/media_storage.py). The `serve/` module is for HTTP serving infrastructure (OpenAI server, routers, etc.) — media encoding doesn't belong there.

**Option A: `tensorrt_llm/_torch/visual_gen/encoding.py`**

Move the encoding logic (video/image encoding, ffmpeg interaction) into the visual_gen internal package:

```
tensorrt_llm/_torch/visual_gen/
├── encoding.py          # VideoEncoder, ImageEncoder, ffmpeg utils (refactored from MediaStorage)
├── output.py            # MediaOutput (calls encoding.py internally)
└── ...
```

Pros: Co-located with visual gen. Encoding is an implementation detail of `MediaOutput.save()`.
Cons: `_torch/` is the PyTorch backend — media encoding is a general utility and doesn't belong there. The `serve/` layer also needs encoding for HTTP responses, creating an awkward cross-layer dependency.

**Option B: `tensorrt_llm/media/encoding.py`** (recommended)

```
tensorrt_llm/
├── media/
│   ├── __init__.py
│   └── encoding.py      # VideoEncoder, ImageEncoder
└── ...
```

Pros: Media encoding is a general utility, not specific to the `_torch` backend. Clean separation of concerns — both `_torch/visual_gen/` and `serve/` can import from `media/` without cross-layer coupling.
Cons: Creates a new top-level module for one file initially, though it's likely to grow as multimodal output support expands.

**Option C: `tensorrt_llm/utils/media.py`**

Put it alongside other utilities.

Pros: Minimal structural change.
Cons: `utils/` becomes a dumping ground.

**Recommendation**: **Option B**. Media encoding is a general-purpose utility that shouldn't live under `_torch/`, which is specifically the PyTorch backend. A dedicated `media/` module provides a clean import path for both the visual gen pipeline (`_torch/visual_gen/`) and the serving layer (`serve/`).

For backward compatibility, keep `MediaStorage` in `serve/` as a thin wrapper that delegates to `media/encoding.py`, and deprecate it.

> **Note**: `MediaOutput` is a public API type and also needs to move out of `_torch/`. Its placement is discussed alongside the broader module structure in §9.

---

## 7. Phase 5 — Lifecycle Management & Observability

### 7.1 Resource Lifecycle

The current [`VisualGen`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L609-L628) uses four cleanup mechanisms: `atexit.register()`, `__del__()`, `__enter__/__exit__()`, and explicit `shutdown()`. This matches the [LLM pattern](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/llm.py#L267-L268) and is acceptable.

### 7.2 Error Handling

Currently, errors surface as `RuntimeError(f"Generation failed: {response.error_msg}")` — a string.

#### Landscape

All three frameworks use raise-on-error (no error fields on result objects) and have minimal exception hierarchies:

| Aspect | TRT-LLM LLM API | vLLM-Omni | SGLang Diffusion |
|--------|-----------------|-----------|------------------|
| **Error model** | Raise-on-error | Raise-on-error | Log-and-continue (per-prompt in batch) |
| **Custom exceptions** | `RequestError(RuntimeError)` — flat, no `LLMError` base | `VLLMValidationError(ValueError)` — not diffusion-specific | None |
| **Error on result object** | No | No | No |
| **Runtime error type** | `RequestError` (via background handler) | Bare `Exception` | Bare `Exception` |

None have invested in a diffusion-specific exception hierarchy.

#### Recommendation

Start with a single base exception:

```python
class VisualGenError(RuntimeError):
    """Base exception for VisualGen operations."""
    ...
```

- Extends `RuntimeError` (consistent with TRT-LLM's `RequestError(RuntimeError)`)
- Replaces the current bare `RuntimeError(f"Generation failed: ...")`
- Gives users a single `except VisualGenError` catch-all

Defer sub-classes (`InvalidParamsError`, `GenerationError`, `EncodingError`, etc.) until there are concrete use cases — e.g., a serving layer that retries transient errors but rejects invalid params immediately.

### 7.3 Warm-Up Interaction

[`BasePipeline.warmup()`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/pipeline.py#L410-L441) runs automatically during `__init__`. The user controls it via `VisualGenArgs.skip_warmup` and `VisualGenArgs.compilation.resolutions`/`num_frames`.

**Issue**: Users may want to add warm-up shapes after init (e.g., after learning their traffic patterns).

**Recommendation**: Expose `VisualGen.warmup(shapes: List[Tuple[int, int, int]])` as an explicit method. Calling it discards all previously compiled CUDA graphs and cached resources from prior warm-ups (including the auto-warmup at init), then warms up from scratch with the new shapes. This avoids accumulating stale resources and gives users a clean reset when they learn their actual traffic patterns.

### 7.4 Observability / Tracing (Low Priority)

VisualGen has no tracing support. The LLM API has full OpenTelemetry (OTLP) integration:

**How the LLM API does it:**

1. **Configuration** — `LlmArgs.otlp_traces_endpoint` (optional URL). On init, `LLM.__init__()` calls `tracing.init_tracer("trt.llm", endpoint)` to set up a `TracerProvider` with a `BatchSpanProcessor`.

2. **Header propagation** — `LLM.generate_async()` accepts `trace_headers: Mapping[str, str]`. The serving layer (`openai_server.py`) extracts `traceparent`/`tracestate` from incoming HTTP headers and passes them through. The headers are stored on `GenerationResult.trace_headers`.

3. **Span creation** — When a request completes, `GenerationResultBase.do_tracing()` creates a span named `"llm_request"` with `SpanKind.SERVER`, rooted in the caller's trace context. It records:
   - Request attributes: `temperature`, `top_p`, `top_k`, `max_tokens`, `n`, `request_id`
   - Usage: `prompt_tokens`, `completion_tokens`
   - Latency: `time_to_first_token`, `e2e`, `time_in_queue`, `kv_cache_transfer_time`
   - Events: `kv_cache_transfer_start`/`end` with timestamps

4. **Key files** — `llmapi/tracing.py` (tracer init, span utilities, `SpanAttributes` enum), `executor/result.py` (`do_tracing()` on `GenerationResultBase`).

**What VisualGen would need:**

1. **`VisualGenArgs.otlp_traces_endpoint`** — reuse the same `tracing.init_tracer()` from `llmapi/tracing.py` (no need to duplicate).

2. **`trace_headers` on `VisualGen.generate()`** — pass through to `VisualGenResult`, same as the LLM API pattern.

3. **Visual-gen-specific `SpanAttributes`** — diffusion has different metrics than LLM (no TTFT or token counts). Relevant attributes:
   - `gen_ai.request.id`, `gen_ai.request.model`
   - `visual_gen.resolution` (height x width), `visual_gen.num_frames`
   - `visual_gen.num_inference_steps`, `visual_gen.guidance_scale`
   - `visual_gen.latency.e2e`, `visual_gen.latency.pipeline` (denoising), `visual_gen.latency.encoding`

4. **`do_tracing()` on `VisualGenResult`** — create a `"visual_gen_request"` span on completion, analogous to the LLM's `"llm_request"` span.

**Recommendation** (future): Reuse `llmapi/tracing.py` infrastructure (tracer init, span export, context propagation). Add visual-gen-specific span attributes and a `do_tracing()` method on the result type. This is additive and non-breaking — defer until serving-side observability is a priority.

---

## 8. Cross-Cutting: Streaming Readiness

We do NOT implement streaming now, but the API shape must be forward-compatible.

### Design Approach

Rather than designing a complex chunk/stream protocol now, we ensure the API shapes accommodate it:

1. **`VisualGenOutput.finished: bool`** — Today always `True`. In streaming mode, intermediate outputs have `finished=False`.

2. **`generate()` returns `VisualGenOutput`, not raw `MediaOutput`** — The wrapper gives room for metadata.

3. **Future `stream()` method**:

```python
# Future addition — does NOT require changing generate()
async def stream(
    self,
    prompt: str,
    params: Optional[VisualGenParams] = None,
) -> AsyncIterator[VisualGenOutput]:
    """Yield partial results as they become available.

    For video: each yielded VisualGenOutput.output.video contains
    frames generated so far (growing tensor).
    VisualGenOutput.finished is True on the last yield.
    """
```

4. **No `ChunkType` enum needed**: The `VisualGenOutput` wrapper already tells the consumer what modalities are populated (check `output.video is not None`, `output.audio is not None`). A progress field (`VisualGenOutput.progress: float`) gives 0→1 visibility.

5. **Non-streaming `generate()` wraps `stream()` internally** in the future:

```python
def generate(self, inputs, params):
    # Current: direct execution
    # Future: could wrap stream() — all chunks → final MediaOutput
    ...
```

This is the same pattern the team proposed (`generate` wraps `stream`), but without introducing new abstractions (`ChunkType`, `DiffusionChunk`) until we need them.

---

## 9. Cross-Cutting: Module & Directory Structure

### Current Structure

```
tensorrt_llm/
├── __init__.py              # exports VisualGen, VisualGenParams, VisualGenArgs, MediaOutput
├── llmapi/
│   ├── __init__.py          # re-exports VisualGen, VisualGenParams
│   ├── visual_gen.py        # VisualGen, VisualGenParams, DiffusionRemoteClient, DiffusionGenerationResult
│   └── ...
├── _torch/visual_gen/
│   ├── __init__.py          # exports everything (public + internal mixed)
│   ├── config.py            # VisualGenArgs + 7 sub-configs + DiffusionModelConfig
│   ├── executor.py          # DiffusionExecutor + DiffusionRequest + DiffusionResponse
│   ├── output.py            # MediaOutput
│   ├── pipeline.py          # BasePipeline
│   ├── pipeline_loader.py   # PipelineLoader
│   ├── pipeline_registry.py # AutoPipeline, @register_pipeline
│   ├── models/              # wan/, flux/, ltx2/
│   └── ...
├── inputs/data.py           # VisualGenInputs mixed with LLM input types
└── serve/media_storage.py   # MediaStorage (encoding)
```

### Three Options (Elaborated)

> **Note**: The directory listings below use proposed type names from earlier sections (`VisualGenOutput` from §6.1, `VisualGenResult` from §5.2). Internal types keep their current `Diffusion*` names (see §4.4 future work note).

#### Option A: New top-level `visualgenapi/` (parallel to `llmapi/`)

```
tensorrt_llm/
├── llmapi/                     # LLM public API — no VisualGen here
│   ├── __init__.py
│   └── ...
├── visualgenapi/               # NEW: VisualGen public API surface
│   ├── __init__.py             # exports: VisualGen, VisualGenParams, VisualGenOutput
│   ├── visual_gen.py           # VisualGen class + VisualGenResult
│   ├── params.py               # VisualGenParams
│   ├── output.py               # VisualGenOutput, VisualGenMetrics
│   └── _client.py              # internal: DiffusionRemoteClient
├── media/                      # NEW: media data container + encoding (see §6.3)
│   ├── __init__.py             # exports: MediaOutput, VideoEncoder, ImageEncoder
│   ├── output.py               # MediaOutput
│   └── encoding.py             # VideoEncoder, ImageEncoder (moved from serve/)
├── _torch/visual_gen/          # Internal implementation (unchanged)
│   ├── executor.py             # DiffusionExecutor, DiffusionRequest (internal, rename deferred)
│   └── ...
└── __init__.py                 # top-level: from .visualgenapi import VisualGen, ...
```

Note: `VisualGenArgs` moves to `visualgenapi/` (or `llmapi/`, see §3.1), not staying in `_torch/`. This is consistent with the LLM API pattern where `BaseLlmArgs`/`TrtLlmArgs`/`TorchLlmArgs` all live in `llmapi/llm_args.py`, not in `_torch/`.

Pros:
- Cleanest separation of public vs internal.
- `visualgenapi/` is the "boundary" — everything inside `_torch/visual_gen/` is explicitly internal.
- Mirrors the `llmapi/` / `_torch/` split that exists for LLM.
- New team members immediately know where the API lives.
- **Code ownership**: The VisualGen team can own `visualgenapi/` independently. Changes don't require `llmapi-dev` review, which is focused on LLM APIs.

Cons:
- New top-level module. Changes to `__init__.py`, CI, docs.
- VisualGen is still small (1 class, 2-3 supporting types). A whole module may be premature.
- `from tensorrt_llm import VisualGen` still works (via `__init__.py`), so users don't care about the internal module name.

#### Option B: Keep in `llmapi/`, clarify public vs internal

```
tensorrt_llm/
├── llmapi/
│   ├── __init__.py             # exports: VisualGen, VisualGenArgs, VisualGenParams, VisualGenOutput
│   ├── visual_gen.py           # VisualGen + VisualGenParams + VisualGenOutput (public)
│   ├── visual_gen_args.py      # NEW: VisualGenArgs + sub-configs (moved from _torch/)
│   ├── _visual_gen_client.py   # internal: DiffusionRemoteClient (rename deferred)
│   └── ...
├── media/                      # NEW: media data container + encoding (see §6.3)
│   ├── __init__.py             # exports: MediaOutput, VideoEncoder, ImageEncoder
│   ├── output.py               # MediaOutput
│   └── encoding.py             # VideoEncoder, ImageEncoder (moved from serve/)
├── _torch/visual_gen/
│   ├── __init__.py             # only exports what _visual_gen_client.py needs
│   ├── config.py               # DiffusionModelConfig (internal, rename deferred)
│   ├── executor.py             # DiffusionExecutor, DiffusionRequest (internal, rename deferred)
│   └── ...
```

Pros:
- Minimal structural change. Just rename/reorganize within existing modules.
- Pragmatic — VisualGen IS an API alongside LLM; sharing `llmapi/` is reasonable.
- Underscore prefix on internal files (`_visual_gen_client.py`) is enough.

Cons:
- `llmapi/` name implies "LLM API". Having VisualGen there is semantically odd.
- As VisualGen grows, `llmapi/` becomes a dumping ground.
- **Code ownership friction**: `llmapi/` is owned by the `llmapi-dev` team, which is focused on LLM APIs. VisualGen changes in `llmapi/` would require their review, adding process overhead for the VisualGen team and review burden for `llmapi-dev` reviewers who may lack VisualGen context.

#### Option C: New `tensorrt_llm/visual_gen/` (non-underscore) as the public API re-export layer

```
tensorrt_llm/
├── visual_gen/                 # NEW: thin public re-export layer
│   ├── __init__.py             # re-exports VisualGen, VisualGenParams, VisualGenOutput
│   ├── visual_gen.py           # VisualGen class (moved from llmapi/)
│   ├── params.py               # VisualGenParams
│   └── output.py               # VisualGenOutput, VisualGenMetrics
├── media/                      # media data container + encoding (see §6.3)
│   ├── __init__.py             # exports: MediaOutput, VideoEncoder, ImageEncoder
│   ├── output.py               # MediaOutput
│   └── encoding.py             # VideoEncoder, ImageEncoder
├── _torch/visual_gen/          # Internal implementation
│   └── ...
├── llmapi/
│   ├── __init__.py             # NO VisualGen exports
│   └── ...
```

Pros:
- `tensorrt_llm.visual_gen` is a natural, discoverable namespace.
- Not inside `llmapi/` — no semantic confusion.
- Not inside `_torch/` — clearly public.
- **Code ownership**: Like Option A, the VisualGen team can own `visual_gen/` independently without requiring `llmapi-dev` review.

Cons:
- Could conflict with `_torch/visual_gen/` naming (both are "visual_gen").
- Import paths like `from tensorrt_llm.visual_gen import VisualGen` vs `tensorrt_llm._torch.visual_gen.config import VisualGenArgs` — the duplication may confuse contributors.

#### Recommendation

**Option B now, Option A as a future milestone.**

Rationale:
- Option B is the minimum viable change. It clarifies public vs internal without module restructuring.
- The key improvement: `llmapi/__init__.py` explicitly lists what's public; internal types get underscore prefixes.
- The code ownership concern (VisualGen changes requiring `llmapi-dev` review) is real but manageable short-term — the VisualGen surface in `llmapi/` is small and well-scoped.
- If/when VisualGen grows to 5+ public classes or we add more generation modalities (e.g., audio-only, 3D), graduate to Option A (`visualgenapi/`) or Option C (`visual_gen/`) to resolve the ownership friction.
- Users always do `from tensorrt_llm import VisualGen` regardless — they don't see the internal module path.

---

## 10. Cross-Cutting: Naming Conventions

This section covers naming holistically — from the top-level "brand name" down to individual field names. Naming is the API. Users read names before they read docs.

### 10.1 The "VisualGen" Brand — Is It Right?

The current name `VisualGen` was chosen to be modality-agnostic across image and video. But LTX-2 also generates **audio**. And future models may generate 3D meshes, point clouds, or other non-visual media.

| Candidate | Covers Image? | Covers Video? | Covers Audio? | Covers 3D? | Ecosystem precedent |
|-----------|:---:|:---:|:---:|:---:|-----|
| `VisualGen` | ✓ | ✓ | ✗ | ✗ | — |
| `MediaGen` | ✓ | ✓ | ✓ | ✓ | — |
| `DiffusionEngine` | ✓ | ✓ | ✓ | ✓ | SGLang, vLLM-omni use "diffusion" internally |
| `MultiModalGen` | ✓ | ✓ | ✓ | ✓ | vLLM-omni: `multimodal_gen` |
| `GenerativeEngine` | ✓ | ✓ | ✓ | ✓ | Too generic |

**Discussion:**

- `VisualGen` is good enough for now. Audio accompanying video is still "visual generation with audio output." Standalone audio models (e.g., MusicGen) are a stretch.
- `MediaGen` is more accurate but may be confused with Meta's MediaGen model.
- `DiffusionEngine` leaks implementation — not all future models may use diffusion (e.g., autoregressive image models like Janus, Anole).
- `MultiModalGen` overlaps with multimodal LLM understanding (which is a different concept).

**Recommendation**: Keep `VisualGen` for now. The name is established in examples, docs, and the codebase. If we add standalone audio or 3D generation, reconsider then. The output type (`MediaOutput`) is already modality-agnostic, which is the right abstraction.

### 10.2 Naming Convention Rules (Proposed)

Establish explicit patterns that mirror the LLM API. The goal: a user who knows the LLM API can predict VisualGen class names.

| Suffix | Role | LLM API Example | VisualGen Example |
|--------|------|-----------------|-------------------|
| `*Args` | Engine/model loading config (init-time) | `LlmArgs`, `TorchLlmArgs`, `TrtLlmArgs` | `VisualGenArgs` |
| `*Config` | Sub-configuration objects (composable, nested) | `KvCacheConfig`, `CudaGraphConfig`, `SchedulerConfig` | `ParallelConfig`, `CompilationConfig`, `AttentionConfig` |
| `*Params` | Per-request generation parameters | `SamplingParams` | `VisualGenParams` |
| `*Output` | User-facing result objects | `RequestOutput`, `CompletionOutput` | `VisualGenOutput`, `MediaOutput` |
| `*Result` | Async result handles | `GenerationResult` | `VisualGenResult` (renamed from `DiffusionGenerationResult`) |
| `*Request` | Internal wire format (not user-facing) | — (internal) | `DiffusionRequest` (keep for now, rename deferred — see §4.4) |
| `*Response` | Internal wire format (not user-facing) | — (internal) | `DiffusionResponse` (keep for now) |
| `*Error` | Exception types | — | `VisualGenError`, `InvalidParamsError`, etc. |
| `*Prompt` | Input type dicts | `TextPrompt`, `TokensPrompt` | N/A — prompt is plain `str` (see §4.1) |

**Key rule: Internal types live in internal modules.** The `_torch/visual_gen/` module path signals "internal." Internal types keep their current `Diffusion*` names for now; renaming them to `VisualGen*` is deferred as future work (see §4.4).

### 10.3 Eliminating the "Diffusion" Prefix

**Motivation**: The `Diffusion` prefix leaks an implementation detail ("diffusion models") into type names. The product brand is `VisualGen`, and the module is already `visual_gen/`. As the product expands to non-diffusion architectures (e.g., autoregressive video models), the "Diffusion" prefix becomes misleading.

Currently, seven classes use the `Diffusion` prefix. **This refactor only renames the public API type** (`DiffusionGenerationResult`). Internal types keep their current names for now — renaming them is deferred as future work (see §4.4).

| Current | Visibility | Action | Rationale |
|---------|-----------|--------|-----------|
| [`DiffusionGenerationResult`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L366) | **Public** | **Rename → `VisualGenResult`** | See §5.2 for naming options. |
| [`DiffusionRequest`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/executor.py#L20) | Internal | Keep (future work) | Defer — refactor `params` embedding first (§4.4) |
| [`DiffusionResponse`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/executor.py#L63) | Internal | Keep (future work) | Defer |
| [`DiffusionExecutor`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/executor.py#L77) | Internal | Keep (future work) | Defer |
| [`DiffusionRemoteClient`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/llmapi/visual_gen.py#L52) | Internal | Keep (future work) | Defer |
| [`DiffusionModelConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/config.py#L468) | Internal | Keep (future work) | Defer |
| [`DiffusionStepProtocol`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/protocols.py#L36) | Internal (deep) | Keep (future work) | Defer |

The related function `run_diffusion_worker` — keep for now (future work).

### 10.4 `*Args` vs `*Config` Consistency

Currently:
- Engine-level: `VisualGenArgs` (uses `*Args` — matches `LlmArgs`)
- Sub-configs: `ParallelConfig`, `CompilationConfig`, `AttentionConfig`, `TeaCacheConfig`, `PipelineConfig` (use `*Config`)
- Internal model config: `DiffusionModelConfig` (uses `*Config`)

This is actually correct and consistent with the LLM API:
- `LlmArgs` → engine config (top-level)
- `KvCacheConfig`, `CudaGraphConfig`, `SchedulerConfig` → sub-configs

**The rule**: The top-level user-facing engine configuration object uses `*Args`. Composable sub-configurations use `*Config`. This is consistent; no change needed.

However, there's an inconsistency in the **constructor parameter name** (`diffusion_args`). See §3.3 for the motivation, options, and recommendation to rename it to `args`.

### 10.5 `VisualGenParams` vs `SamplingParams`

In the LLM world, `SamplingParams` controls how tokens are sampled (temperature, top_p, top_k). In diffusion, the analogous concept is "generation parameters" (height, width, steps, guidance_scale, seed). The LLM API's `SamplingParams` name doesn't translate well to diffusion.

| Candidate | Rationale |
|-----------|-----------|
| `VisualGenParams` | Current name. Clear, specific. |
| `GenerationParams` | More generic. Could apply to LLM too (confusing). |
| `SamplingParams` | Overloaded with the LLM meaning. |
| `InferenceParams` | Too generic — "inference" is a broad term. |

**Recommendation**: Keep `VisualGenParams`. It's clear, specific to visual generation, and doesn't collide with `SamplingParams`. The `VisualGen*` prefix is the brand identifier.

### 10.6 Output & Future Naming Summary

The naming decisions for `VisualGenOutput`, `VisualGenMetrics`, and `VisualGenResult` are discussed in context where these types are first introduced:

- **`VisualGenOutput` and `VisualGenMetrics`** — see §6.1 for motivation, options, and recommendation.
- **`VisualGenResult`** — see §5.2 for motivation, options, and recommendation.

For a consolidated view of how these map to the LLM API:

| LLM API | VisualGen API | Role | Discussion |
|---------|---------------|------|------------|
| `CompletionOutput` | `MediaOutput` (keep) | The generated content | — |
| `RequestOutput` | `VisualGenOutput` (NEW) | Request-level wrapper with metadata | §6.1 |
| `GenerationResult` | `VisualGenResult` (renamed) | Async handle | §5.2 |

### 10.7 Input Type Naming

Current:
- `VisualGenTextPrompt` — a `TypedDict` with `prompt` and optional `negative_prompt`
- `VisualGenTokensPrompt` — a `TypedDict` with `prompt_token_ids`
- `VisualGenPromptInputs` — union type
- `VisualGenInputs` — union including batch

**Issues:**
- With the decision to use plain `str` as the prompt type (§4.1 Option B), none of these structured input types are needed in the public API.
- Conditioning inputs (negative_prompt, image, mask) live on `VisualGenParams`, not on a prompt dict.
- `VisualGenTokensPrompt` exposes token IDs, which is an LLM concept — diffusion text encoders handle tokenization internally.

**Recommendation**: Drop all four types from the public API:
- `VisualGenTextPrompt` — no longer needed; prompt is `str`
- `VisualGenTokensPrompt` — drop; raw token IDs are not a user-facing concept for visual generation
- `VisualGenPromptInputs` — drop; the union simplifies to `Union[str, List[str]]` inline
- `VisualGenInputs` — drop; same reason
- Remove from `inputs/data.py`; if any internal code still needs structured prompt dicts, use underscore-prefixed internal types

### 10.8 Benchmark/Serving Helper Naming

The codebase also has:
- [`VisualGenRequestInput`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/serve/scripts/benchmark_visual_gen.py#L67) in `serve/scripts/benchmark_visual_gen.py`
- [`VisualGenSampleRequest`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/bench/benchmark/visual_gen_utils.py#L28), [`VisualGenRequestOutput`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/bench/benchmark/visual_gen_utils.py#L35), [`VisualGenBenchmarkMetrics`](https://github.com/NVIDIA/TensorRT-LLM/blob/e71a200c4cd83238568ed6bcd2aa0cd7dd539c90/tensorrt_llm/bench/benchmark/visual_gen_utils.py#L47) in `bench/`

These are consistent with the `VisualGen*` brand. No changes needed, but they should evolve if the core types are renamed (e.g., if `VisualGenRequestOutput` wraps the new `VisualGenOutput`).

### 10.9 Pipeline Class Naming (Internal)

Current model pipeline classes: `WanPipeline`, `Flux2Pipeline`, `LTX2Pipeline`, `FluxPipeline`, `WanImageToVideoPipeline`. Base: `BasePipeline`.

These are internal (in `_torch/visual_gen/models/`) and users never see them. The naming is consistent and clear. **No changes needed.**

However, the registry name strings should stay aligned:
```python
@register_pipeline("WanPipeline")
class WanPipeline(BasePipeline): ...
```

### 10.10 Summary: Full Rename Map

| Current Name | New Name | Type | File Location |
|-------------|----------|------|---------------|
| **Public types** | | | |
| `VisualGen` | Keep | Class | `llmapi/visual_gen.py` |
| `VisualGenArgs` | Keep (remove `to_dict`/`from_dict`, see §3.2) | Pydantic model | `llmapi/visual_gen_args.py` (move from `_torch/visual_gen/config.py`, see §3.1) |
| `VisualGenParams` | Keep (convert to Pydantic `StrictBaseModel`, see §13 Q1) | Pydantic model | `llmapi/visual_gen.py` |
| `MediaOutput` | Keep (move out of `_torch/`, see §9) | Dataclass | `_torch/visual_gen/output.py` |
| `DiffusionGenerationResult` | → `VisualGenResult` | Class | `llmapi/visual_gen.py` |
| `VisualGenTextPrompt` | → Drop (prompt is plain `str`, see §4.1) | TypedDict | `inputs/data.py` |
| `VisualGenTokensPrompt` | → Drop | TypedDict | `inputs/data.py` |
| `VisualGenPromptInputs` | → Drop (inline as `Union[str, List[str]]`) | Type alias | — |
| `VisualGenInputs` | → Drop | Type alias | — |
| — | `VisualGenOutput` (NEW) | Pydantic model | `llmapi/visual_gen.py` |
| — | `VisualGenMetrics` (NEW) | Pydantic model | `llmapi/visual_gen.py` |
| — | `ExtraParamSchema` (NEW) | Pydantic model | `llmapi/visual_gen.py` |
| **Internal types (rename deferred — see §4.4)** | | | |
| `DiffusionRequest` | Keep (future: → `VisualGen*`) | Dataclass | `_torch/visual_gen/executor.py` |
| `DiffusionResponse` | Keep (future) | Dataclass | `_torch/visual_gen/executor.py` |
| `DiffusionExecutor` | Keep (future) | Class | `_torch/visual_gen/executor.py` |
| `DiffusionRemoteClient` | Keep (future) | Class | `llmapi/visual_gen.py` |
| `DiffusionModelConfig` | Keep (future) | Pydantic model | `_torch/visual_gen/config.py` |
| `run_diffusion_worker` | Keep (future) | Function | `_torch/visual_gen/executor.py` |
| `MediaStorage` | → Deprecate; encoding moves to `media/encoding.py` | Class | `serve/media_storage.py` |
| **Constructor params** | | | |
| `VisualGen(model_path=...)` | → `VisualGen(model=...)` (see §3.4) | Param | `llmapi/visual_gen.py` |
| `VisualGen(diffusion_args=...)` | → `VisualGen(args=...)` (see §3.3) | Param | `llmapi/visual_gen.py` |

### 10.11 Naming in Peer Frameworks (Reference)

For context, here's how peer frameworks name their equivalent concepts:

| Concept | Diffusers | SGLang | vLLM-Omni | TRT-LLM LLM API | **TRT-LLM VisualGen (proposed)** |
|---------|-----------|--------|-----------|-----------------|----------------------------------|
| Engine class | `StableDiffusionPipeline` | `Server` | `Omni` | `LLM` | `VisualGen` |
| Engine config | `pipeline.from_pretrained(...)` kwargs | `ServerArgs` | `EngineArgs` | `LlmArgs` | `VisualGenArgs` |
| Sub-configs | — | Various | — | `KvCacheConfig`, etc. | `ParallelConfig`, etc. |
| Request params | `__call__(**kwargs)` | `SamplingParams` | `OmniDiffusionSamplingParams` | `SamplingParams` | `VisualGenParams` |
| Input type | `prompt: str` | `SamplingParams.prompt` | `OmniPromptType` | `PromptInputs` | `str` (plain prompt) |
| Output (content) | `FluxPipelineOutput` | file / frames | base64 bytes | `CompletionOutput` | `MediaOutput` |
| Output (request) | — | — | — | `RequestOutput` | `VisualGenOutput` |
| Async handle | — | — | — | `GenerationResult` | `VisualGenResult` |
| Internal request | — | `GenerateReqInput` | `OmniDiffusionRequest` | — | `DiffusionRequest` (keep, future rename) |
| Internal config | — | `PipelineConfig` | — | — | `DiffusionModelConfig` (keep, future rename) |

---

## 11. Proposed API Shape (End-to-End)

```python
from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
from tensorrt_llm.llmapi import ParallelConfig  # after move from _torch/ per §3.1

# ─── Phase 1: Engine Init ───
engine = VisualGen(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",    # renamed from model_path per §3.4
    args=VisualGenArgs(parallel=ParallelConfig(dit_cfg_size=2)),
)

# ─── Phase 2+3: Request + Execute ───

# Minimal — all model defaults
result = engine.generate("A cat sitting on a windowsill")

# Explicit params
params = VisualGenParams(height=480, width=832, seed=42, num_inference_steps=50)
result = engine.generate("A cat sitting on a windowsill", params=params)

# Conditioning inputs (i2v) — image on params, not on prompt
result = engine.generate(
    "Make it snow",
    params=VisualGenParams(image="summer.jpg", num_frames=81),
)

# Batch with shared params
results = engine.generate(
    ["A cat on a windowsill", "A dog in a park"],
    params=VisualGenParams(height=480, width=832, seed=42),
)

# Batch with per-request params (different resolutions)
results = engine.generate(
    ["A cat", "A dog"],
    params=[
        VisualGenParams(height=480, width=832),
        VisualGenParams(height=720, width=1280),
    ],
)

# Model-specific params
result = engine.generate("A dragon", params=VisualGenParams(
    extra_params={"guidance_scale_2": 3.0, "boundary_ratio": 0.5},
))

# Async
future = engine.generate_async("A cat on a windowsill", params=params)
result = await future.result()

# With progress
result = engine.generate(
    "A long video of a sunset",
    params=VisualGenParams(num_frames=161),
    on_progress=lambda step, total: print(f"{step}/{total}"),
)

# ─── Phase 4: Output ───
print(result.seed_used)              # 42
print(result.timing.inference_ms)    # 12345.6

# Raw tensor access (zero-copy, engine philosophy)
video_tensor = result.output.video   # torch.Tensor (T, H, W, C) uint8
image_tensor = result.output.image   # for image models

# Convenience save (format from extension)
result.output.save("cat.mp4")                         # auto frame_rate from metadata
result.output.save("cat.avi", codec="mjpeg")           # explicit codec
result.output.save("cat_frame.png")                    # save first frame as image

# Encoding to bytes (explicit format required)
video_bytes = result.output.to_bytes(format="mp4")
image_bytes = result.output.to_bytes(format="png")

# PIL Image (image models)
pil_img = result.output.to_pil()

# ─── Phase 5: Lifecycle ───
engine.shutdown()
# Or: with VisualGen(...) as engine: ...
```

---

## 12. Summary of Recommendations

### By Lifecycle Phase

| Phase | Recommendation | Priority | Breaking Change? |
|-------|---------------|----------|-----------------|
| **Init** | Move `VisualGenArgs` + sub-configs from `_torch/` to `llmapi/visual_gen_args.py` (match `llm_args.py` pattern, see §3.1) | High | No (re-export preserves user imports) |
| **Init** | Rename `diffusion_args` → `args`/`config` | Medium | Yes (param name) |
| **Request** | `params` defaults to `None` (model defaults); most fields → `Optional[None]` | High | Yes (behavior) |
| **Request** | Add `extra_params: dict` for model-specific overflow; remove model-specific flat fields | High | Yes (field removal) |
| **Request** | `extra_params` validated at request time with schema from pipeline | High | No (additive) |
| **Request** | Support single + batch via Union types (match LLM pattern) | Medium | Yes (signature) |
| **Request** | Embed `VisualGenParams` in internal wire type (eliminate field-by-field copy) | Medium | No (internal) |
| **Execute** | Rename future: `DiffusionGenerationResult` → `VisualGenResult` | Low | Yes (type name) |
| **Execute** | Add optional `on_progress` callback | Medium | No (additive) |
| **Output** | `generate()` returns `VisualGenOutput` (wraps `MediaOutput`) with `seed_used`, `timing`, `finished` | High | Yes (return type) |
| **Output** | Add `MediaOutput.save()`, `.to_pil()`, `.to_bytes()` with format/codec/quality args | High | No (additive) |
| **Output** | `MediaOutput.metadata` carries `frame_rate` from generation | Medium | No (additive) |
| **Output** | Move `MediaOutput` out of `_torch/` — placement discussed in §9 | Medium | No (re-export preserves imports) |
| **Output** | Move encoding from `serve/media_storage.py` → `media/encoding.py` | Medium | No (internal) |
| **Lifecycle** | Define exception hierarchy (`VisualGenError`, `InvalidParamsError`, etc.) | Medium | No (additive) |
| **Lifecycle** | Expose `warmup(shapes)` method for post-init warm-up | Low | No (additive) |
| **Structure** | Stay in `llmapi/` for now; prefix internal types with underscore | Low | No (internal) |
| **Naming** | Establish `*Args`/`*Config`/`*Params`/`*Output` suffix conventions (see §10.2) | High | Convention |
| **Naming** | Rename public `DiffusionGenerationResult` → `VisualGenResult`; internal `Diffusion*` types keep names (future work, see §4.4, §10.3) | Medium | Yes (`generate_async` users) |
| **Naming** | Rename `DiffusionGenerationResult` → `VisualGenResult` (public) | Medium | Yes |
| **Naming** | Drop structured input types (`VisualGenTextPrompt`, `VisualGenTokensPrompt`, `VisualGenPromptInputs`, `VisualGenInputs`); prompt is plain `str` (see §4.1, §10.7) | Medium | Yes |
| **Naming** | Rename constructor param `diffusion_args` → `args` (see §10.4) | Medium | Yes |
| **Naming** | Use `VisualGen*` prefix for output/result types: `VisualGenOutput`, `VisualGenResult`, `VisualGenMetrics` (see §10.6) | Medium | Yes |
| **Init** | Rename `model_path` → `model` for consistency with `LLM(model=...)` (see §3.4) | Medium | Yes (param name) |
| **Init** | Remove `to_dict()` / `from_dict()` per coding guidelines (see §3.2) | Low | Yes (method removal) |
| **Request** | Make `VisualGenParams` a Pydantic `StrictBaseModel`, not a dataclass (see §13 Q1) | High | Yes (base class) |
| **Request** | Remove `output_type` field — redundant with `MediaOutput` convenience methods (see §4.2) | Medium | Yes (field removal) |
| **Request** | Choose deterministic `seed` default to avoid silent behavior break (see §4.2) | High | Yes (behavior) |
| **Request** | Specify and document image/conditioning input format on `VisualGenParams` (see §4.1) | High | No (documentation) |
| **Execute** | Define batch error semantics: raise-on-first vs return-all-with-errors (see §5.1) | High | Yes (contract) |
| **Execute** | Address thread safety of `req_counter` before stable graduation (see §13 Q10) | Low | No (internal) |
| **Execute** | Rename `ParamSpec` → `ExtraParamSchema` to avoid `typing.ParamSpec` collision (see §4.3) | Low | Yes (type name) |
| **Cross-cutting** | Define deprecation/migration strategy for prototype→stable (see §13 Q11) | Medium | Process |

---

## 13. Open Questions

1. **`VisualGenParams` should be Pydantic, not a dataclass.**
   `CODING_GUIDELINES.md` states: *"When defining any user-facing configuration classes [...] **always** use Pydantic classes rather than dataclasses or vanilla classes."* `VisualGenParams` is the most-touched user-facing type in the entire API. Making it a dataclass loses `extra="forbid"` (typos silently pass), `model_validator` (cross-field validation like `height % 8 == 0`), and `Field(description=...)`. The QPS overhead concern is valid for LLM `SamplingParams` (thousands of requests/second), but visual generation runs at single-digit requests/second — Pydantic construction overhead (~100μs) is negligible relative to multi-second denoising. The LLM API's `SamplingParams` predates the Pydantic mandate and is a known exception; new user-facing types should not replicate that exception. **Use `StrictBaseModel` with `extra="forbid"` and `Field(description=...)` for all user-facing fields.**

2. **Should `generate()` accept `params=None` for full model defaults?**
   - `visual_gen.generate("A cat")` is the best possible UX.
   - Concern: Users might not realize they can tune quality. Mitigation: document model-specific defaults in deployment guides.
   - **Tentative**: Yes. Match diffusers' `pipeline("A cat")` simplicity.

3. **How does `extra_params` map to OpenAI-compatible serving?**
   - The serving layer converts OpenAI request fields → `VisualGenParams`. Unknown fields → `extra_params`.
   - Or: the serving layer has its own extension fields that map to specific `extra_params` keys.

4. **Should `MediaOutput.save()` support batch tensors directly?**
   - Currently `save_video` strips batch dim. Should `save()` save only the first item, or require unbatched input?
   - **Tentative**: `MediaOutput` represents a single generation result (not a batch). Batching is at the `VisualGenOutput` level.

5. **Audio muxing in `save()`**: When `output.video` and `output.audio` are both populated (LTX-2), should `save("output.mp4")` automatically mux audio? What codec? (MP4+AAC via ffmpeg)

6. **LoRA support**: SGLang has extensive LoRA support for diffusion. Should `VisualGenParams` (or a separate argument) support LoRA adapters?

8. **Multiple outputs per prompt**: `num_images_per_prompt > 1` — does `VisualGenOutput.output` contain one `MediaOutput` or a list? Proposal: Always one `MediaOutput`, but `image` field shape gains a batch dim. If so, `to_pil()` should return `List[PIL.Image.Image]`, and `save()` needs behavior specification (save first only? save all with index suffix?).

9. **Batch error handling strategy**: When `generate()` processes a batch and some items fail, should it raise on the first error (simpler, current behavior) or return all results with per-item `error` fields (more robust for production)? See §5.1.1. This affects the `VisualGenOutput` contract.

10. **Thread safety of `req_counter`**: `VisualGen.req_counter` is incremented without a lock in `generate_async()`. If multiple threads call `generate_async()` concurrently, request IDs can collide. Options: use `itertools.count()` (atomic under CPython GIL), `threading.Lock`, or `asyncio`-based counter. Low risk for current usage, but should be addressed before stable graduation.

11. **Deprecation / migration strategy from `prototype` → `stable`**: When breaking changes are made (renaming `DiffusionGenerationResult`, changing `diffusion_args` parameter name, etc.), should old names be kept as deprecated aliases for one release cycle? Should there be a migration guide? Since the API is `prototype`, breaking is acceptable, but the transition plan should be documented.

12. **`model_path` → `model` rename**: Should the constructor parameter be renamed from `model_path` to `model` for consistency with `LLM(model=...)` and the diffusers/vLLM ecosystem? See §3.4.

13. **Sub-config re-export story**: Which sub-configs (`ParallelConfig`, `CompilationConfig`, `AttentionConfig`, etc.) should be re-exported from the top-level `tensorrt_llm` namespace vs only available at `tensorrt_llm.llmapi`? The LLM API re-exports key configs (`KvCacheConfig`, etc.) at the top level for convenience.

---

## Appendix: Framework Source Code References

### Diffusers

| File | Content |
|------|---------|
| [`utils/outputs.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/outputs.py) | `BaseOutput` class — `OrderedDict`-based output base |
| [`utils/export_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/export_utils.py) | `export_to_video()`, `export_to_gif()` utilities |
| [`pipelines/flux/pipeline_output.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_output.py) | `FluxPipelineOutput` — `.images: list[PIL.Image.Image] \| np.ndarray` |
| [`pipelines/wan/pipeline_output.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_output.py) | `WanPipelineOutput` — `.frames: torch.Tensor` |

### SGLang Diffusion

| File | Content |
|------|---------|
| [`configs/sample/sampling_params.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/sample/sampling_params.py) | `SamplingParams` base (~964 lines). `height: int \| None = None`, `_default_height: ClassVar[int \| None] = None`. Model subclasses override. `__post_init__` applies model defaults. `_validate()` for self-consistency. |
| [`configs/pipeline_configs/base.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/base.py) | `ModelTaskType` enum (`T2V`, `T2I`, `I2V`, `TI2V`, etc.), pipeline config base |
| [Issue #20078](https://github.com/sgl-project/sglang/issues/20078) | Bug: diffusers backend ignoring model-specific defaults |
| [PR #20080](https://github.com/sgl-project/sglang/pull/20080) | Fix: resolve model-specific `sampling_param_cls` via config registry |

### vLLM-Omni

| File | Content |
|------|---------|
| [`inputs/data.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/inputs/data.py) | `OmniDiffusionSamplingParams` (~100+ fields, mixes user params with runtime state: `latents`, `timesteps`, `step_index`, `past_key_values`). `extra_args: dict` for overflow. |
| [`diffusion/request.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/request.py) | `OmniDiffusionRequest` — wraps `list[OmniPromptType]` + `OmniDiffusionSamplingParams` |
| [Architecture overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/) | OmniStage, E/P/D/G disaggregation, CFG companion flow |
| [Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/) | OpenAI DALL-E compatible, base64 PNG output, "pass-through" param handling |

### Streaming Research

| Reference | Key Insight |
|-----------|-------------|
| [StreamWise (arxiv 2603.05800)](https://arxiv.org/abs/2603.05800) | Multi-modal orchestration, adaptive quality, sub-second startup |
| [HiStream (arxiv 2512.21338)](https://arxiv.org/abs/2512.21338) | Autoregressive video gen, 76-107x faster with spatial/temporal/timestep compression |
| [WaveSpeed AI](https://wavespeed.ai/landing/real-time-video-generation) | Frame-streaming via WebSocket/WebRTC, sub-500ms latency |
| [fal.ai streaming](https://docs.fal.ai/documentation/development/streaming) | SSE for diffusion step previews, callback-based progress |

---

*This document is a living artifact. Please add comments, counter-proposals, and concerns inline or in the discussion thread.*
