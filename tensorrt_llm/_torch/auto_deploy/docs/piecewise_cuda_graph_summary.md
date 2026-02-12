# Piecewise CUDA Graph — Beginner-Friendly Code Walkthrough

> **Audience**: Developers new to the TensorRT-LLM / AutoDeploy codebase.
> **Scope**: All new & modified files for the piecewise CUDA graph feature.

______________________________________________________________________

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
1. [High-Level Architecture](#2-high-level-architecture)
1. [How AutoDeploy Works (Context for Newcomers)](#3-how-autodeploy-works-context-for-newcomers)
1. [File-by-File Walkthrough](#4-file-by-file-walkthrough)
   - 4.1 [piecewise_utils.py (NEW)](#41-piecewise_utilspy-new)
   - 4.2 [inplace_ops.py (NEW)](#42-inplace_opspy-new)
   - 4.3 [piecewise_runner.py (NEW)](#43-piecewise_runnerpy-new)
   - 4.4 [torch_cudagraph.py (MODIFIED)](#44-torch_cudagraphpy-modified)
   - 4.5 [compile_model.py (MODIFIED)](#45-compile_modelpy-modified)
   - 4.6 [default.yaml (MODIFIED)](#46-defaultyaml-modified)
1. [End-to-End Flow: From Config to Runtime](#5-end-to-end-flow-from-config-to-runtime)
1. [Key Concepts Explained](#6-key-concepts-explained)
1. [Bugs Encountered & How They Were Fixed](#7-bugs-encountered--how-they-were-fixed)
1. [Known Limitations](#8-known-limitations)
1. [How to Use It](#9-how-to-use-it)

______________________________________________________________________

## 1. What Problem Does This Solve?

### The Before: Monolithic CUDA Graphs (Decode-Only)

A **CUDA graph** records a sequence of GPU operations and replays them with near-zero
CPU overhead. AutoDeploy already supported this via `CapturedGraph`, but with a critical
limitation: the **entire model** was captured as a **single** CUDA graph.

This works for **decode-only batches** (1 token per sequence, fixed batch size) because
all tensor shapes are fixed. But for **prefill** (processing variable-length input
sequences) or **mixed batches** (prefill + decode together), the shapes vary — the
number of tokens, sequence lengths, and metadata all change per request. A monolithic
CUDA graph cannot handle this.

| Batch Type     | Before (monolithic only)    |
|---------------|-----------------------------|
| Decode-only    | ✅ CUDA graph (fast)        |
| Prefill        | ❌ Eager (slow)             |
| Mixed          | ❌ Eager (slow)             |

### The After: Piecewise CUDA Graphs

**Piecewise CUDA graph** solves this by splitting the model at "dynamic" operation
boundaries (attention, SSM, etc.) and capturing only the **static** segments (MLPs,
norms, projections) as individual CUDA graphs. Dynamic ops run eagerly between the
captured segments:

```
┌──────────────┐     ┌───────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────────┐
│  Segment 0   │────▶│ Attention │────▶│  Segment 1   │────▶│ Attention │────▶│  Segment 2   │
│ (CUDA Graph) │     │  (eager)  │     │ (CUDA Graph) │     │  (eager)  │     │ (CUDA Graph) │
└──────────────┘     └───────────┘     └──────────────┘     └───────────┘     └──────────────┘
```

| Batch Type     | After (dual-mode)           |
|---------------|-----------------------------|
| Decode-only    | ✅ Monolithic CUDA graph    |
| Prefill        | ✅ **Piecewise CUDA graph** |
| Mixed          | ✅ **Piecewise CUDA graph** |

______________________________________________________________________

## 2. High-Level Architecture

The feature adds a **dual-mode** execution strategy:

```
               ┌─────────────────────────────┐
               │      Runtime Forward         │
               └──────────┬──────────────────┘
                          │
               ┌──────────▼──────────────────┐
               │  Is batch decode-only?       │
               └──────┬───────────┬──────────┘
                 YES  │           │  NO (has prefill)
                      ▼           ▼
           ┌──────────────┐  ┌──────────────────────────────┐
           │  Monolithic  │  │  num_tokens matches bucket?  │
           │  CapturedGraph│  └──────┬───────────┬──────────┘
           │  (1 graph     │   YES   │           │  NO
           │   replay)    │         ▼           ▼
           └──────────────┘  ┌──────────────┐  ┌────────────┐
                             │  Piecewise   │  │   Eager    │
                             │  CapturedGraph│  │  Fallback  │
                             │  (per-segment│  └────────────┘
                             │   replay)    │
                             └──────────────┘
```

Three classes orchestrate this:

- **`CapturedGraph`** (pre-existing) — captures the whole model as one CUDA graph for
  decode-only batches.
- **`PiecewiseCapturedGraph`** (NEW) — splits the model, captures static segments, runs
  dynamic ops eagerly. Used for prefill/mixed batches.
- **`DualModeCapturedGraph`** (NEW) — the runtime dispatcher that decides which path to
  take based on the batch type.

______________________________________________________________________

## 3. How AutoDeploy Works (Context for Newcomers)

AutoDeploy (AD) is a deployment path within TensorRT-LLM that takes a HuggingFace model,
optimizes it through a series of transforms, and runs it with the TRT-LLM runtime.

### The Pipeline (simplified)

```
HuggingFace model
    │
    ▼  [build_model]       — Load on meta device
    ▼  [export_to_gm]      — torch.export → FX GraphModule
    ▼  [pattern_matcher]    — Fuse ops, match patterns
    ▼  [sharding]           — Tensor parallel
    ▼  [load_weights]       — Move to GPU
    ▼  [cache_init]         — Replace attention with cached versions, add KV cache
    ▼  [compile_model]      — ★ THIS IS WHERE CUDA GRAPHS ARE CAPTURED ★
    │
    ▼  ADEngine.forward()   — Runtime: receive requests, run model, return logits
```

After `cache_init`, the model's attention ops have been replaced with **custom ops** like
`auto_deploy::flashinfer_attention_mha_with_cache`. These custom ops interact with the
KV cache, handle variable-length sequences, and have data-dependent control flow — making
them **un-capturable** by CUDA graphs for mixed/prefill batches.

### Key Files Newcomers Should Know

| File | Purpose |
|------|---------|
| `config/default.yaml` | Defines all transform steps and their config |
| `transform/library/compile_model.py` | The `compile_model` transform — entry point for CUDA graph setup |
| `compile/backends/torch_cudagraph.py` | The `torch-cudagraph` backend — CUDA graph capture logic |
| `compile/compiler.py` | Abstract `CompilerBackend` base class + registry |
| `shim/ad_executor.py` | `ADEngine` — the runtime engine that calls `model(**args)` |
| `shim/interface.py` | `CachedSequenceInterface` — manages input generation for the model |

______________________________________________________________________

## 4. File-by-File Walkthrough

### 4.1 `piecewise_utils.py` (NEW)

**Location**: `tensorrt_llm/_torch/auto_deploy/compile/piecewise_utils.py`
**Purpose**: Graph splitting infrastructure — identifies dynamic ops and splits the FX
graph at their boundaries.

#### What's in it

**1. Dynamic ops registry** (lines 27–66):

A catalog of ALL custom ops that cannot be captured in CUDA graphs for mixed/prefill
batches. There are 13 ops across 5 categories:

| Category | Ops | Why uncapturable |
|----------|-----|-----------------|
| Attention (×3) | `flashinfer_attention_mha_with_cache`, `triton_attention_flattened_mha_with_cache`, `torch_cached_attention_with_cache` | Kernel grid depends on per-sequence lengths |
| SSM (×3) | `triton_cached_ssm`, `torch_cached_ssm`, `flashinfer_cached_ssm` | Python-level branching on `batch_info_host` |
| CausalConv (×2) | `triton_cached_causal_conv1d`, `cuda_cached_causal_conv1d` | Branches on prefill vs decode |
| Delta Rule (×1) | `fla_cached_delta_rule` | Branches on prefill vs decode |
| Metadata (×2) | `flashinfer_attention_prepare_metadata`, `mamba_ssm_prepare_metadata` | CPU math on CUDA tensors |

**2. `is_dynamic_cached_op(node)`** (lines 69–95):

Takes an FX graph `Node` and returns `True` if it's one of the 13 dynamic ops. Handles
different target representations (OpOverload, qualname, string) and strips `.default`
suffixes for matching.

**3. `SplitInfo` dataclass** (lines 98–110):

A simple container returned by the split function:

```python
@dataclass
class SplitInfo:
    split_gm: GraphModule          # The split graph with submod_0, submod_1, ...
    num_submodules: int            # Total count
    dynamic_submod_indices: List[int]  # Which submods are dynamic (run eagerly)
    static_submod_indices: List[int]   # Which submods are static (CUDA graph)
```

**4. `split_graph_at_dynamic_ops(gm)`** (lines 112–211):

The core splitting function. It:

1. Walks every node in the FX graph
1. Assigns **partition IDs**: static ops share a partition, each dynamic op gets its own
1. Calls `torch.fx.passes.split_module()` to perform the actual split
1. Returns `SplitInfo` with which submodules are dynamic vs static

Example: a model with 2 attention layers produces:

```
submod_0 (static)  — embedding + norm + QKV projection
submod_1 (dynamic) — attention layer 1
submod_2 (static)  — output projection + MLP + norm + QKV projection
submod_3 (dynamic) — attention layer 2
submod_4 (static)  — output projection + final norm + LM head
```

**5. `swap_to_inplace_ops(gm)`** (lines 239–322):

A graph transform that replaces each dynamic op with its **inplace variant** (see §4.2).
For each dynamic op node, it:

1. Inserts a `torch.empty_like(q)` output buffer before the op
1. Replaces the op with its inplace version (takes `output` as extra arg)
1. Replaces all downstream uses of the original output with the buffer

This ensures outputs live at **fixed memory addresses** — required for CUDA graph segments
to read from stable pointers.

Skips:

- CausalConv ops (already mutate their input in-place)
- Metadata ops (produce metadata, not model activations)

______________________________________________________________________

### 4.2 `inplace_ops.py` (NEW)

**Location**: `tensorrt_llm/_torch/auto_deploy/compile/inplace_ops.py`
**Purpose**: Registers 7 inplace variants of dynamic custom ops.

#### Why inplace ops?

In piecewise CUDA graphs, each static segment is captured as a separate CUDA graph. The
graph replays using the **exact same memory addresses** it was captured with. If a dynamic
op (running eagerly between two segments) allocates a **new** output tensor each time,
the next CUDA graph segment would read from a stale address.

Inplace ops solve this: they take a pre-allocated `output` tensor and write into it via
`output.copy_(result)`. The output tensor's address never changes, so the next segment
always reads from the correct location.

#### What's in it

7 inplace op registrations, each following the same pattern:

```python
@torch.library.custom_op(
    "auto_deploy::original_op_name_inplace", mutates_args=("output",)
)
def original_op_name_inplace(
    ... all original args ...,
    output: torch.Tensor,      # ← NEW: pre-allocated output buffer
) -> None:                      # ← Returns None (writes into output instead)
    result = torch.ops.auto_deploy.original_op_name(... all original args ...)
    output.copy_(result)        # ← Copy into the fixed-address buffer

@original_op_name_inplace.register_fake
def original_op_name_inplace_fake(..., output) -> None:
    pass  # No-op for tracing (shapes are known from the output buffer)
```

The mapping from original → inplace is stored in `ORIGINAL_TO_INPLACE`:

| Original Op | Inplace Variant |
|-------------|-----------------|
| `flashinfer_attention_mha_with_cache` | `flashinfer_attention_mha_with_cache_inplace` |
| `triton_attention_flattened_mha_with_cache` | `triton_attention_flattened_mha_with_cache_inplace` |
| `torch_cached_attention_with_cache` | `torch_cached_attention_with_cache_inplace` |
| `triton_cached_ssm` | `triton_cached_ssm_inplace` |
| `torch_cached_ssm` | `torch_cached_ssm_inplace` |
| `flashinfer_cached_ssm` | `flashinfer_cached_ssm_inplace` |
| `fla_cached_delta_rule` | `fla_cached_delta_rule_inplace` |

Note: CausalConv ops already mutate `input` in-place (declared with `mutates_args={"input"}`),
so they don't need inplace wrappers.

______________________________________________________________________

### 4.3 `piecewise_runner.py` (NEW)

**Location**: `tensorrt_llm/_torch/auto_deploy/compile/piecewise_runner.py`
**Purpose**: Manages CUDA graph warmup → capture → replay for a **single static segment**.

#### The State Machine

Each `ADPiecewiseRunner` wraps one static submodule and implements a per-`num_tokens`
state machine:

```
                     ┌─────────────────────────────────────────────────┐
                     │            Per-num_tokens Entry                  │
                     │  (e.g., num_tokens=128)                         │
                     │                                                 │
   call 1-3:         │  ┌──────────┐     call 4:    ┌─────────┐       │
   ─────────────────▶│  │ WARMUP   │──────────────▶│ CAPTURE  │       │
                     │  │ (eager,  │               │ (record  │       │
                     │  │  track   │               │  CUDA    │       │
                     │  │  ptrs)   │               │  graph)  │       │
                     │  └──────────┘               └────┬─────┘       │
                     │                                  │              │
                     │         calls 5+:                ▼              │
                     │  ┌──────────────────────────────────────┐       │
                     │  │  REPLAY (copy inputs → replay graph) │       │
                     │  └──────────────────────────────────────┘       │
                     └─────────────────────────────────────────────────┘
```

#### Key Datastructures

**`SegmentEntry`** (lines 37–55): State for one `(num_tokens)` configuration:

- `warmup_count` — how many warmup iterations have run
- `cuda_graph` — the captured CUDA graph (None until capture)
- `static_inputs` — the frozen input tensors used during graph capture
- `dynamic_indices` — which input slots are activations (need copy during replay)
- `static_output` — the output tensor from the captured graph (returned on replay)
- `_warmup_data_ptrs` — tracks `data_ptr()` per arg across warmup iterations

#### The Smart Pointer Tracking Optimization (lines 132–170)

This is the most important optimization in this file. The problem: when capturing a CUDA
graph, all input tensors must be at **fixed addresses**. The naive approach is to
`clone()` ALL tensor inputs. But for a 7B model, this duplicates multi-GB weight tensors
— causing OOM.

The insight: **weight tensors already have stable addresses** (they're model parameters,
allocated once). Only **activation tensors** change address between calls (allocated by
the previous layer).

During the warmup phase (3 iterations), the runner records `data_ptr()` of every tensor
argument:

- **Call 1**: Record all `data_ptr()` values
- **Calls 2–3**: Compare with recorded values. If a tensor's `data_ptr()` changed → it's
  a dynamic activation. If unchanged → it's a weight (stable address).

At capture time:

- **Weights**: referenced directly (zero-copy, saves GBs)
- **Activations**: cloned into fixed-address buffers (only these need copying on replay)

#### The `num_tokens` Context (lines 77–87)

A subtlety: after splitting, some submodules receive intermediate tensors (SSM metadata,
chunk indices) whose `shape[0]` is NOT `num_tokens`. So inferring `num_tokens` from input
shapes is unreliable.

Solution: a **class-level** variable `_current_num_tokens` that the orchestrator
(`PiecewiseCapturedGraph`) sets before each forward pass. ALL runners read from this
shared context.

```python
class ADPiecewiseRunner(nn.Module):
    _current_num_tokens: Optional[int] = None  # Class-level, shared by ALL instances

    @classmethod
    def set_current_num_tokens(cls, num_tokens):
        cls._current_num_tokens = num_tokens
```

______________________________________________________________________

### 4.4 `torch_cudagraph.py` (MODIFIED)

**Location**: `tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py`
**Purpose**: The `torch-cudagraph` compile backend. This file had the existing
`CapturedGraph` class and `TorchCudagraphCompiler`. The changes add 3 new classes
and a helper.

#### Pre-existing code (unchanged)

- **`CapturedGraph`** (lines 82–219): The monolithic CUDA graph class. Captures the
  entire model as one graph per batch size. Used for decode-only batches. This code
  is **not modified** — it works exactly as before.

#### New: `_submod_has_cuda_ops()` (lines 41–67)

A helper that checks if a static submodule actually contains any GPU-kernel-launching
operations. When two dynamic ops are adjacent (e.g., `prepare_metadata` → `SSM`), the
static partition between them is **empty** — it only contains FX plumbing like `getitem`,
`reshape`, etc. Capturing an empty CUDA graph triggers a PyTorch warning and wastes
resources.

This function inspects the submodule's FX graph and returns `False` if it only has
trivial ops (defined in `_TRIVIAL_CALL_FUNCTIONS` and `_TRIVIAL_CALL_METHODS`). These
trivial submodules are then **skipped** — they run eagerly instead of being wrapped in
`ADPiecewiseRunner`.

#### New: `PiecewiseCapturedGraph` (lines 222–377)

The **orchestrator** for piecewise CUDA graphs. It manages the full lifecycle:

**`prepare()`** (lines 242–318):

1. **Phase A**: Creates a new `GraphModule` that **shares weights** with the original
   but has its own copy of the FX graph (avoids OOM from `deepcopy`). Then calls
   `swap_to_inplace_ops()` to replace dynamic ops with inplace variants.
1. **Phase B**: Calls `split_graph_at_dynamic_ops()` to split the graph into submodules.
1. **Phase C**: Wraps each static submodule in an `ADPiecewiseRunner`. Skips trivial
   submodules (no CUDA ops). Shares a single CUDA graph memory pool across all runners.

**`warmup_and_capture()`** (lines 320–354):
For each `num_tokens` bucket (e.g., 128, 256, 512):

1. Generate synthetic inputs via `get_args_kwargs(nt)`
1. Set the shared `_current_num_tokens` context
1. Run `warmup_iters + 1` iterations through the split graph
   - First 3 calls: warmup (each `ADPiecewiseRunner` runs eagerly, tracks pointers)
   - 4th call: capture (each runner records its CUDA graph)

**`forward()`** (lines 356–377):
Sets `_current_num_tokens`, runs the split graph (each submodule handles its own
warmup/capture/replay), clears context.

#### New: `DualModeCapturedGraph` (lines 380–464)

The **runtime dispatcher** — the top-level module returned to the engine. At inference
time, every forward pass goes through this class:

**`_is_decode_only()`** (lines 414–434):
Checks `batch_info_host[0]` (num_prefill). If 0 → decode-only batch.

**`_get_num_tokens()`** (lines 436–446):
Extracts total token count from `input_ids.numel()`.

**`forward()`** (lines 448–464):

```
if decode-only → monolithic CapturedGraph (single graph replay)
elif num_tokens matches a bucket → piecewise CapturedGraph (per-segment replay)
else → eager fallback (no CUDA graph)
```

#### Modified: `TorchCudagraphCompiler` (lines 467–535)

Extended with new parameters:

- `piecewise_enabled: bool` — toggle for dual-mode
- `piecewise_num_tokens: List[int]` — bucket sizes for piecewise capture
- `get_mixed_args_kwargs_for_compile: Callable` — generates prefill/mixed inputs

The `compile()` method now has two paths:

- `piecewise_enabled=False` (default): returns a `CapturedGraph` (same as before)
- `piecewise_enabled=True`: builds both a `CapturedGraph` (decode) and a
  `PiecewiseCapturedGraph` (prefill/mixed), wraps them in `DualModeCapturedGraph`

______________________________________________________________________

### 4.5 `compile_model.py` (MODIFIED)

**Location**: `tensorrt_llm/_torch/auto_deploy/transform/library/compile_model.py`
**Purpose**: The `compile_model` transform — the entry point that creates and configures
the CUDA graph compiler.

#### New: `_generate_default_piecewise_num_tokens()` (lines 19–40)

When `piecewise_num_tokens: null` in config, this generates power-of-2 bucket sizes:

```
max_num_tokens=8192 → [64, 128, 256, 512, 1024, 2048, 4096, 8192]
```

#### Modified: `CompileModelConfig` (lines 43–66)

Two new config fields:

```python
piecewise_enabled: bool = False
piecewise_num_tokens: Optional[List[int]] = None
```

#### New: `_get_mixed_args_kwargs()` (lines 92–111)

Generates synthetic inputs for piecewise CUDA graph capture. This is the function passed
to `PiecewiseCapturedGraph.warmup_and_capture()`.

Key design:

- For small buckets (≤8 tokens): generates a pure prefill request (single sequence)
- For larger buckets (>8 tokens): generates a **mixed** batch — one long prefill sequence
  - one decode token. This exercises both code paths in the dynamic ops.

```python
# Example: num_tokens=256
cm.info.set_example_sequence(
    input_ids=[
        [1] * 255,   # prefill sequence (255 tokens)
        [1],          # decode sequence (1 token)
    ],
)
```

#### Modified: `_apply_to_full_model()` (lines 116–138)

Now passes `_get_mixed_args_kwargs` and auto-generated bucket sizes to the compiler when
`piecewise_enabled=True`.

______________________________________________________________________

### 4.6 `default.yaml` (MODIFIED)

**Location**: `tensorrt_llm/_torch/auto_deploy/config/default.yaml`
**Purpose**: Default configuration for all AutoDeploy transforms.

Two new lines added under `compile_model:`:

```yaml
compile_model:
  stage: compile
  expect_mem_change: true
  run_per_gm: false
  cuda_graph_batch_sizes: null
  backend: torch-compile
  piecewise_enabled: true             # ← NEW
  piecewise_num_tokens: [7, 16]       # ← NEW (example fixed buckets)
```

Note: In production, `piecewise_num_tokens: null` would auto-generate power-of-2 buckets.
The `[7, 16]` shown here is likely for testing/development purposes.

______________________________________________________________________

## 5. End-to-End Flow: From Config to Runtime

### Build Time (model compilation)

```
1. default.yaml declares:
     compile_model:
       backend: torch-cudagraph
       piecewise_enabled: true
       piecewise_num_tokens: [128, 256, 512]

2. CompileModel._apply_to_full_model() is called by the transform pipeline
   │
   ├─ Creates _get_args_kwargs() for decode batch arg generation
   ├─ Creates _get_mixed_args_kwargs() for prefill/mixed batch arg generation
   ├─ Auto-generates buckets if piecewise_num_tokens is null
   │
   └─ Instantiates TorchCudagraphCompiler(
         model, piecewise_enabled=True,
         piecewise_num_tokens=[128, 256, 512],
         get_mixed_args_kwargs_for_compile=_get_mixed_args_kwargs,
         cuda_graph_batch_sizes=[1, 2, 4, 8],
         ...
      )

3. TorchCudagraphCompiler.compile():
   │
   ├─ Build monolithic CapturedGraph for decode:
   │   └─ Capture entire model for batch_sizes=[1, 2, 4, 8]
   │
   ├─ Build PiecewiseCapturedGraph for prefill/mixed:
   │   │
   │   ├─ prepare():
   │   │   ├─ GraphModule(model, deepcopy(model.graph))  ← shallow copy (share weights)
   │   │   ├─ swap_to_inplace_ops(gm)                     ← replace attention/SSM with inplace
   │   │   ├─ split_graph_at_dynamic_ops(gm)               ← split at dynamic boundaries
   │   │   └─ Wrap static submods in ADPiecewiseRunner     ← skip trivial ones
   │   │
   │   └─ warmup_and_capture():
   │       For each num_tokens in [512, 256, 128] (descending):
   │         ├─ Generate mixed batch args
   │         ├─ Set ADPiecewiseRunner._current_num_tokens = nt
   │         └─ Run 4 iterations (3 warmup + 1 capture) through split graph
   │
   └─ Return DualModeCapturedGraph(monolithic, piecewise)
```

### Runtime (inference)

```
ADEngine.forward(scheduled_requests, ...)
  │
  ├─ _prepare_inputs() → populates cache_seq_interface.named_args
  │
  └─ _compute_logits():
       │
       └─ self.model(**named_args)
            │
            ▼  (self.model is DualModeCapturedGraph)
            │
            DualModeCapturedGraph.forward(**kwargs):
              │
              ├─ Check batch_info_host[0] (num_prefill):
              │
              ├─ IF num_prefill == 0 (decode-only):
              │   └─ monolithic.forward(**kwargs)
              │       └─ Copy inputs → replay single CUDA graph → return output
              │
              ├─ ELIF num_tokens matches bucket (e.g., 256):
              │   └─ piecewise.forward(**kwargs, num_tokens=256)
              │       ├─ Set ADPiecewiseRunner._current_num_tokens = 256
              │       └─ split_gm(**kwargs):
              │            ├─ submod_0 (ADPiecewiseRunner): copy inputs → replay CG
              │            ├─ submod_1 (dynamic attention): run eagerly
              │            ├─ submod_2 (ADPiecewiseRunner): copy inputs → replay CG
              │            ├─ submod_3 (dynamic attention): run eagerly
              │            └─ submod_4 (ADPiecewiseRunner): copy inputs → replay CG
              │
              └─ ELSE (no matching bucket):
                  └─ piecewise.original_model(**kwargs)
                      └─ Full eager execution (no CUDA graph)
```

______________________________________________________________________

## 6. Key Concepts Explained

### 6.1 Why "Inplace" Ops?

When CUDA graph **Segment A** is captured, its output tensor is at address `0xABC`.
When **Segment B** is captured right after, it reads from `0xABC`.

At **replay** time, if the dynamic op between A and B allocates a **new** output tensor
at `0xDEF`, Segment B still reads from `0xABC` — stale data! 💥

Inplace ops fix this: they always write to the same pre-allocated buffer at `0xABC`.

```
Without inplace:     Segment A → output@0xABC → Attention → NEW output@0xDEF → Segment B reads 0xABC 💥
With inplace:        Segment A → output@0xABC → Attention(output=buf@0xABC) → Segment B reads 0xABC ✅
```

### 6.2 Why Not Reuse TRTLLM's Piecewise Implementation?

TRTLLM already has piecewise CUDA graphs (`tensorrt_llm/_torch/compilation/piecewise_optimizer.py`),
but it's tightly coupled to the native runtime:

- 6+ global state flags
- Hardcoded split on `trtllm::attn_custom_op_inplace` (not AD's ops)
- Depends on `TensorWrapper` C++ bindings

AD needs a clean, self-contained implementation that splits on AD's own custom ops and
fits into AD's `CompilerBackend` registry.

### 6.3 Why Exact `num_tokens` Match (No Padding)?

When you pad `input_ids` from 7→64, `batch_info_host` still says there are 7 real tokens.
Dynamic ops like attention/SSM read `batch_info_host` and produce outputs for 7 tokens.
The next static CUDA graph segment expects 64 tokens → **shape mismatch** 💥.

In TRTLLM's native pipeline, attention is "shape-preserving" (output shape = input shape,
regardless of real tokens). In AD, multiple dynamic ops read metadata and produce
variable-sized outputs, making padding inconsistent.

Current solution: exact match only. If `num_tokens` doesn't match any bucket, fall back
to eager.

### 6.4 What Is `batch_info_host`?

A tensor `[num_prefill, num_prefill_tokens, num_decode]` that tells the model about the
current batch composition. It's used everywhere:

- `num_prefill == 0` → decode-only batch
- `num_prefill_tokens` → how many tokens are from prefill sequences
- This is how `DualModeCapturedGraph` decides which execution path to take

### 6.5 FX Graph and GraphModule

PyTorch FX captures a model's forward pass as a **graph** of operations. Each operation
is a `Node` with:

- `op`: type of operation (`call_function`, `call_module`, `placeholder`, `output`)
- `target`: what to call (e.g., `torch.ops.auto_deploy.flashinfer_attention_mha_with_cache`)
- `args`: input nodes

A `GraphModule` is an `nn.Module` backed by an FX graph — you can inspect, transform,
and split it programmatically. This is what makes piecewise splitting possible: we walk
the graph, identify dynamic ops, and use `torch.fx.passes.split_module` to partition it.

### 6.6 CUDA Graph Memory Pool

All `ADPiecewiseRunner` instances share a single CUDA graph **memory pool** (passed via
the `graph_pool` parameter). This means CUDA graphs across different segments allocate
from the same pool, reducing total GPU memory overhead.

______________________________________________________________________

## 7. Bugs Encountered & How They Were Fixed

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | **OOM from `copy.deepcopy(model)`** | Duplicated ALL weight tensors (~14GB for 7B) | `GraphModule(model, copy.deepcopy(model.graph))` — share weights, copy only the graph DAG |
| 2 | **OOM from cloning weight tensors in runner** | `clone()` on ALL tensor args including multi-GB MoE weights | Smart pointer tracking: only clone activations, reference weights directly |
| 3 | **Missing `mamba_ssm_prepare_metadata`** | Not registered as dynamic op → placed inside static segment → CPU math during graph capture → crash | Added to `_METADATA_PREP_OPS` |
| 4 | **Wrong `num_tokens` inference** | Inferred from first tensor's `shape[0]`, but `input_ids` is `[1, N]` (gives 1, not N) | Explicit `num_tokens` parameter passed from `DualModeCapturedGraph` |
| 5 | **Shape mismatch from padding** | Pad `input_ids` 7→64 but `batch_info_host` still says 7 → dynamic ops produce 7-token output → next segment expects 64 | Removed all padding, use exact match only |
| 6 | **"CUDA Graph is empty" warning** | Two adjacent dynamic ops → empty static partition → empty CUDA graph | `_submod_has_cuda_ops()` check: skip trivial submodules |

______________________________________________________________________

## 8. Known Limitations

1. **Exact match only**: Piecewise CUDA graphs only fire when `num_tokens` exactly matches
   a pre-captured bucket. In production most prefill lengths won't hit exact buckets, so
   they fall back to eager. Future fix: JIT capture (capture on first occurrence of new
   `num_tokens`) or scheduler-level padding.

1. **First-call capture latency**: Each new `num_tokens` bucket requires 3 warmup + 1
   capture iterations before CUDA graph is available.

1. **Memory per bucket**: Each captured bucket holds its own set of static input/output
   buffers. With many buckets × many segments, memory can grow. Mitigated by shared graph
   pools across runners.

1. **SSM ops are split points**: Even though SSM math is theoretically capturable
   (fixed-size chunk operations), the Python-level implementation has data-dependent
   branching that prevents capture. Future optimization: refactor SSM ops to remove
   branching.

______________________________________________________________________

## 9. How to Use It

### In a YAML config

```yaml
compile_model:
  backend: torch-cudagraph
  piecewise_enabled: true
  # Explicit buckets:
  piecewise_num_tokens: [128, 256, 512, 1024]
  # Or auto-generate (power-of-2 from 64 to max_num_tokens):
  # piecewise_num_tokens: null
  cuda_graph_batch_sizes: [1, 2, 4, 8]
```

### What the config fields mean

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | str | `torch-compile` | Must be `torch-cudagraph` for CUDA graphs |
| `piecewise_enabled` | bool | `false` | Enable dual-mode (monolithic + piecewise) |
| `piecewise_num_tokens` | list or null | `null` | Bucket sizes for piecewise capture. `null` = auto-generate power-of-2 |
| `cuda_graph_batch_sizes` | list or null | `null` | Batch sizes for monolithic decode graphs |

### Verify it's working

In the logs, look for:

```
TorchCudagraphCompiler: dual-mode enabled (monolithic + piecewise)
PiecewiseCapturedGraph: swapped N ops to inplace variants
Piecewise split: M submodules (X static, Y dynamic)
PiecewiseCapturedGraph: prepared with M submodules (...)
PiecewiseCapturedGraph: warming up for num_tokens=512
PiecewiseCapturedGraph: captured graphs for num_tokens=512
```

At runtime, if piecewise CG is being used you'll see:

```
# (debug level) when falling back to eager:
DualModeCapturedGraph: num_tokens=137 not in captured buckets {128, 256, 512}, falling back to eager
```

______________________________________________________________________

## Appendix: File Change Summary

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `compile/piecewise_utils.py` | **NEW** | 324 | Dynamic ops registry, graph splitting, inplace op swap |
| `compile/inplace_ops.py` | **NEW** | 423 | 7 inplace op variants + original→inplace mapping |
| `compile/piecewise_runner.py` | **NEW** | 270 | Per-segment CUDA graph state machine with smart pointer tracking |
| `compile/backends/torch_cudagraph.py` | **MODIFIED** | 536 | Added `PiecewiseCapturedGraph`, `DualModeCapturedGraph`, `_submod_has_cuda_ops()`, extended `TorchCudagraphCompiler` |
| `transform/library/compile_model.py` | **MODIFIED** | 145 | Added `piecewise_enabled`, `piecewise_num_tokens` config, `_get_mixed_args_kwargs()`, auto-bucket generation |
| `config/default.yaml` | **MODIFIED** | 209 | Added `piecewise_enabled` and `piecewise_num_tokens` fields |
| `shim/ad_executor.py` | **NO CHANGE** | — | Dispatch is handled inside `DualModeCapturedGraph`, no executor changes needed |
