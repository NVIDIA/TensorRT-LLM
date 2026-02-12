# Piecewise CUDA Graph for AutoDeploy — Implementation Plan

## 1. Background & Motivation

Today, AutoDeploy's CUDA graph support (via `torch-cudagraph` backend) only works for
**decode-only** batches. This is because `CapturedGraph` captures the entire model as a
single monolithic CUDA graph, which requires all tensor shapes to be fixed — impossible
when prefill sequences have variable lengths.

**Piecewise CUDA graph** solves this by splitting the model at dynamic-shape boundaries
(attention, SSM, etc.) and capturing only the **static segments** (MLP, projections,
norms) as individual CUDA graphs. Dynamic ops run eagerly between the captured segments.

```
┌──────────────┐     ┌───────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────────┐
│  Segment 0   │────▶│ Attention │────▶│  Segment 1   │────▶│ Attention │────▶│  Segment 2   │
│ (CUDA Graph) │     │  (eager)  │     │ (CUDA Graph) │     │  (eager)  │     │ (CUDA Graph) │
└──────────────┘     └───────────┘     └──────────────┘     └───────────┘     └──────────────┘
```

This gives us CUDA graph speedups for prefill and mixed batches, not just decode.

## 2. Key Design Decisions

### 2.1. AD-owned PiecewiseRunner (not reusing TRTLLM's)

TRTLLM's `PiecewiseRunner` (in `tensorrt_llm/_torch/compilation/piecewise_optimizer.py`)
is tightly coupled to the native TRTLLM runtime:

- 6+ global state flags (`get_piecewise_cuda_graph_flag`, `set_piecewise_running`, etc.)
- Hardcoded split on `trtllm::attn_custom_op_inplace` (not AD's ops)
- Depends on `TensorWrapper` C++ bindings via `make_weak_ref()`

AD will own a clean, self-contained implementation that:

- Has no global state dependencies
- Splits on AD's own custom ops
- Fits into AD's `CompilerBackend` registry

### 2.2. Inplace Attention Ops (not copy-based)

TRTLLM's approach uses inplace attention (`mutates_args=("output")`) so that each CUDA
graph segment's output lives at a fixed memory address that the next segment can read
from. AD will follow the same pattern:

- Create inplace placeholder variants of AD's attention/SSM/conv ops
- Each takes an `output` tensor as a parameter and writes into it
- This ensures zero extra memory copies between segments at runtime

### 2.3. Dual-Mode: Monolithic CG for Decode + Piecewise CG for Prefill/Mixed

We need perf gains for **both** decode and prefill/mixed. The optimal strategy is:

- **Decode-only batches**: Use monolithic CUDA graph (existing `CapturedGraph`).
  Captures the entire model as a single graph — fastest possible, zero segment-
  transition overhead.
- **Prefill/mixed batches**: Use piecewise CUDA graph (new `PiecewiseCapturedGraph`).
  Captures static segments only — gives CUDA graph speedups where monolithic can't
  work at all.

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
              │  .forward()  │    YES   │           │  NO
              │  (1 graph    │         ▼           ▼
              │   replay)    │  ┌──────────────┐  ┌────────────┐
              └──────────────┘  │  Piecewise   │  │   Eager    │
                                │  CapturedGraph│  │  Fallback  │
                                │  .forward()  │  └────────────┘
                                └──────────────┘
```

This is controlled by a `piecewise_enabled` flag on the `TorchCudagraphCompiler`:

```yaml
compile_model:
  backend: torch-cudagraph
  piecewise_enabled: true           # NEW — enables dual-mode
  cuda_graph_batch_sizes: [1, 2, 4, 8]  # for monolithic decode graphs
  piecewise_num_tokens: [64, 128, 256, 512]  # NEW — for piecewise prefill graphs
```

When `piecewise_enabled=false` (default): behavior is unchanged (monolithic only,
decode-only batches get CUDA graph, prefill/mixed falls back to eager).

When `piecewise_enabled=true`: **both** paths are active:

- Monolithic graphs are captured for `cuda_graph_batch_sizes` (decode)
- Piecewise graphs are captured for `piecewise_num_tokens` (prefill/mixed)

### 2.4. Prefill/Mixed vs Decode-Only — Runtime Dispatch

At the runtime level, `ScheduledRequests.can_run_cuda_graph` currently returns `True`
only for decode-only batches (no `context_requests`):

```python
# tensorrt_llm/_torch/pyexecutor/scheduler.py
@property
def can_run_cuda_graph(self) -> bool:
    return (not self.context_requests)  # True only for decode-only
```

With `piecewise_enabled=true`, we **keep monolithic CG for decode** and **add piecewise
CG for mixed/prefill**:

| Batch Type    | `piecewise_enabled=false` | `piecewise_enabled=true` |
|---------------|---------------------------|--------------------------|
| Decode-only   | ✅ Monolithic CG          | ✅ Monolithic CG (same)  |
| Mixed/Prefill | ❌ Eager fallback         | ✅ **Piecewise CG** (if num_tokens matches) |

The dispatch logic lives entirely in `DualModeCapturedGraph.forward()`:

1. Check if batch is decode-only (via `batch_info_host[0] == 0`) → monolithic CG
1. Else, check if `num_tokens` exactly matches a pre-captured bucket → piecewise CG
1. Else → fall back to eager

### 2.5. Mamba / SSM / CausalConv — Analysis

#### Why SSM is different from attention

SSM (Mamba) is fundamentally **linear attention** — O(n) per token, not O(n²) like
quadratic attention. The underlying math does NOT depend on individual sequence lengths:

- `mamba_chunk_scan_combined` processes all tokens in fixed-size chunks
- Kernel grid dimensions depend on `total_tokens / chunk_size`, NOT per-sequence lengths
- `cu_seqlens` is just data passed to the kernel, not shape-determining

By contrast, attention kernel grids DO depend on per-sequence lengths:

- `context_attention_kv_flattened` has grid `(num_prefill, n_heads, ceil(max_seq_len/32))`
- `max(seq_len)` changes per batch → kernel configuration changes → truly uncapturable

#### Why SSM is still a split point today (implementation, not math)

Despite the math being fixed-shape, the **Python-level custom op implementation** has
data-dependent control flow that prevents CUDA graph capture:

```python
# In triton_cached_ssm:
num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()  # ← CPU sync
preallocated_ssm_out_p = preallocated_ssm_out[:num_prefill_tokens]      # ← dynamic slice
preallocated_ssm_out_d = preallocated_ssm_out[num_prefill_tokens:...]   # ← dynamic slice

# In torch_cached_ssm:
if s == 1:           # ← data-dependent Python branch
    # decode path
else:
    # prefill path — iterates over sequences with for-loop
```

These baked-in branches and dynamic slices make SSM uncapturable for **mixed batches**
where the prefill/decode split changes per iteration.

#### Capturability summary by batch type

| Batch Type | Attention | SSM | Why |
|------------|-----------|-----|-----|
| **Decode-only** | ✅ Capturable | ✅ Capturable | All shapes fixed, `s==1` branch deterministic |
| **Pure prefill** (fixed total tokens) | ❌ Grid depends on `max(seq_len)` | ⚠️ Theoretically capturable | SSM grids are fixed; only Python branching is the blocker |
| **Mixed batch** | ❌ Dynamic split | ❌ Dynamic split | Both branch on `num_prefill` vs `num_decode` |

#### Recommendation: treat SSM as split point now, optimize later

For the initial implementation, **SSM/conv/delta ops are treated as split points**
alongside attention. This is the safe, correct approach.

**Future optimization**: Refactor SSM ops to remove data-dependent branching (e.g.,
always run both prefill and decode paths with zero-length inputs for the inactive
path). This would make SSM capturable even in mixed batches, reducing the number of
graph segments for hybrid models (Bamba, NemotronH). This is a follow-up task, not
a blocker.

#### Split point summary

| Op Type | Custom Op | Inplace? | Split Point? | Notes |
|---------|-----------|----------|-------------|-------|
| Attention | `flashinfer_attention_mha_with_cache` | No → need inplace | Yes | Grid depends on seq_len distribution |
| Attention | `triton_attention_flattened_mha_with_cache` | No → need inplace | Yes | Grid depends on `max(seq_len)` |
| Attention | `torch_cached_attention_with_cache` | No → need inplace | Yes | Python for-loop over sequences |
| SSM | `triton_cached_ssm` | No → need inplace | Yes (for now) | Math is capturable; Python branching is not |
| SSM | `torch_cached_ssm` | No → need inplace | Yes (for now) | Has per-sequence Python for-loop |
| SSM | `flashinfer_cached_ssm` | No → need inplace | Yes (for now) | Same branching pattern |
| CausalConv | `triton_cached_causal_conv1d` | Yes (`mutates_args={"input"}`) | Yes | Branches on prefill vs decode |
| CausalConv | `cuda_cached_causal_conv1d` | Yes (`mutates_args={"input"}`) | Yes | Branches on prefill vs decode |
| Delta Rule | `fla_cached_delta_rule` | No → need inplace | Yes | Branches on prefill vs decode |
| Metadata | `flashinfer_attention_prepare_metadata` | N/A (metadata) | Yes | CPU math on CUDA tensors |
| Metadata | `mamba_ssm_prepare_metadata` | N/A (metadata) | Yes | CPU math on CUDA tensors |

### 2.6. Why TRTLLM's Padding Works but AD's Doesn't

#### How TRTLLM Pads (seq_len=7, bucket=8)

Native TRTLLM pads at the engine level in `_prepare_tp_inputs_no_cache()`:

```python
# tensorrt_llm/_torch/pyexecutor/model_engine.py
if attn_metadata.padded_num_tokens is not None:
    self.input_ids_cuda[num_tokens:padded_num_tokens].fill_(0)     # pad [7:8] with 0
    self.position_ids_cuda[num_tokens:padded_num_tokens].fill_(0)  # pad [7:8] with 0
    virtual_num_tokens = padded_num_tokens                         # = 8

inputs = {
    'input_ids': self.input_ids_cuda[:virtual_num_tokens],         # [8 tokens]
    'position_ids': self.position_ids_cuda[:virtual_num_tokens],   # [8 positions]
    'attn_metadata': attn_metadata,                                # UNCHANGED — still says 7 real
}
```

The critical property that makes this work:

```
┌─────────────────────────────────────────────────────────────┐
│  input_ids = [tok0..tok6, 0]  (8 tokens, position 7 = pad) │
│  attn_metadata = {seq_lens: [7], num_contexts: 1}  (REAL)   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────▼──────────────────────┐
        │  Static Segment 0 (CUDA Graph, N=8)   │
        │  embedding → norm → QKV projection    │
        │  Input: [8] tokens → Output: q[8]     │
        │  ✅ Shape=8, matches captured graph    │
        └────────────────┬──────────────────────┘
                         │
        ┌────────────────▼──────────────────────┐
        │  Dynamic: Attention (eager)           │
        │  Input: q[8], k[8], v[8]              │
        │  BUT: attn_metadata.seq_lens = [7]    │
        │  → FlashAttention cu_seqlens = [0,7]  │
        │  → Kernel processes tokens 0-6 ONLY   │
        │  → Token 7 = garbage in/out           │
        │                                       │
        │  OUTPUT BUFFER: sized from q.shape[0] │
        │  = torch.empty(8, H)                  │
        │  Inplace: output[0:7] = real result   │
        │           output[7] = garbage          │
        │  ✅ Output SHAPE is still [8, H]      │
        └────────────────┬──────────────────────┘
                         │
        ┌────────────────▼──────────────────────┐
        │  Static Segment 1 (CUDA Graph, N=8)   │
        │  o_proj → residual → MLP → norm       │
        │  Input: [8, H] ✅ Shape matches       │
        │  (position 7 = garbage, but harmless) │
        └────────────────┬──────────────────────┘
                         │
                ... repeat for all layers ...
                         │
        ┌────────────────▼──────────────────────┐
        │  Output: hidden_states[8, H]          │
        │  Only [:7] used for logits/sampling   │
        │  Position 7 discarded                 │
        └───────────────────────────────────────┘
```

**TRTLLM's key property: attention is shape-preserving.**

| Property | TRTLLM Attention |
|----------|-----------------|
| Input shape | `q[8, heads, dim]` — padded |
| Metadata | `attn_metadata` with real `seq_lens=[7]` |
| Kernel behavior | Processes only tokens 0-6 (via `cu_seqlens`) |
| Output buffer | `create_output(q)` → sized from `q.shape[0]` = **8** |
| Output shape | **`[8, H]` — same as input, regardless of real tokens** |

The attention inplace op writes real results into positions 0-6 and leaves position 7
unchanged. The **shape is always \[padded, ...\]** — matching the captured CUDA graph.

#### Why This Doesn't Work for AD

| Dimension | TRTLLM | AD |
|-----------|--------|-----|
| Dynamic ops | Only `attn_custom_op_inplace` (1 type) | 13 ops: attention×3, SSM×3, conv×2, delta×1, metadata×2 |
| Metadata | `attn_metadata` (rich Python object, unchanged by padding) | `batch_info_host` (simple tensor: `[num_prefill, num_prefill_tokens, num_decode]`) |
| Shape preservation | ✅ Output shape = input shape (always padded) | ❌ `prepare_metadata` produces **variable-size** outputs |
| What controls real tokens | `cu_seqlens` inside FlashAttention kernel | Python-level branching on `batch_info_host.tolist()` |

**The killer is `prepare_metadata` + `batch_info_host`:**

When we pad `input_ids` from 7→8 but `batch_info_host` still says `[1, 7, 0]` (7 real
prefill tokens), the `prepare_metadata` op reads `batch_info_host` and computes metadata
for 7 tokens. Then the SSM/attention op processes those 7 tokens and produces an output
shaped for 7 tokens. The next static CUDA graph segment expects 8 tokens → **BOOM**.

```
TRTLLM: input[8] → attn(metadata:7) → output[8] → static CG[8]    ✅
AD:     input[8] → prepare_metadata(batch_info:7) → SSM(metadata:7) → output[7]
        → static CG expects [8]                                      💥
```

#### How to Fix This (Future Work)

Three approaches, from cleanest to most pragmatic:

1. **Scheduler-level padding** (cleanest, highest effort):
   The scheduler generates `batch_info_host = [1, 8, 0]` and pads `cu_seqlens` etc.
   to reflect 8 tokens. Every dynamic op sees consistent metadata. This requires changes
   outside AutoDeploy (in the scheduler/executor) and careful handling so the padding
   token doesn't corrupt attention (needs a "null" slot in KV cache or cu_seqlens that
   excludes the padding token from real attention computation).

1. **Make ALL dynamic ops shape-preserving** (medium effort):
   Modify every dynamic op to always produce output sized from the input tensor (like
   TRTLLM's attention). This means `prepare_metadata` would produce metadata for 8
   tokens, SSM would process 8 tokens, etc. The 8th token would be garbage but the
   shapes would be consistent. This is invasive — requires changing all 13 dynamic ops.

1. **JIT capture at runtime** (most pragmatic, lowest effort):
   Don't pad at all. Instead, capture a new piecewise graph the first time each
   `num_tokens` is seen, and cache it for replay on subsequent calls with the same
   count. No metadata inconsistency because no padding. Downside: first-call latency
   for each unique `num_tokens` (warmup + capture), and memory grows with unique counts.

**Current approach**: Exact `num_tokens` match only. No padding. Falls back to eager
if no pre-captured bucket matches. Option 3 (JIT capture) is the recommended next step.

## 3. Implementation Tasks

### Phase 1: Graph Splitting Infrastructure ✅

**Task 1.1: Create `piecewise_utils.py`** ✅

- Location: `tensorrt_llm/_torch/auto_deploy/compile/piecewise_utils.py`
- `is_dynamic_cached_op()` — checks all 13 dynamic op types (attn×3, SSM×3, conv×2, delta×1, metadata×2)
- `split_graph_at_dynamic_ops(gm) -> SplitInfo` — partitions FX graph using `split_module`
- `swap_to_inplace_ops(gm) -> int` — graph transform: original → inplace dynamic ops
- Returns `SplitInfo` with split graph + dynamic/static submodule indices

**Task 1.2: Create `ADPiecewiseRunner`** ✅

- Location: `tensorrt_llm/_torch/auto_deploy/compile/piecewise_runner.py`
- Lightweight `nn.Module` wrapping a single static submodule
- State machine: warmup → capture → replay per `num_tokens`
- **Key optimization**: During warmup, tracks `data_ptr()` of each arg to identify
  static (weight) vs dynamic (activation) tensors. Only clones dynamic tensors —
  saves GBs of GPU memory for models with large weight tensors (e.g., MoE experts)
- **num_tokens context**: Uses class-level `_current_num_tokens` set by the
  orchestrator, NOT inferred from arg shapes (unreliable in piecewise-split models
  where intermediate tensors like SSM metadata don't have num_tokens as dim 0)

### Phase 2: Inplace Attention/SSM Ops ✅

**Task 2.1–2.2: Create inplace variants** ✅

- Location: `tensorrt_llm/_torch/auto_deploy/compile/inplace_ops.py`
- 7 inplace variants: attention×3, SSM×3, delta×1
- Each takes `output` tensor and writes via `output.copy_(result)`
- `ORIGINAL_TO_INPLACE` mapping for graph transform

**Task 2.3: Graph transform** ✅

- `swap_to_inplace_ops()` in `piecewise_utils.py`
- Inserts `torch.empty_like(q)` output buffer before each dynamic op
- Replaces op with inplace variant, replaces all uses with output buffer
- Skips causal_conv (already inplace) and prepare_metadata (metadata only)

### Phase 3: Backend Integration ✅

**Task 3.1–3.5: Dual-mode support** ✅

- `PiecewiseCapturedGraph` — split graph orchestrator
- `DualModeCapturedGraph` — runtime dispatcher (decode vs prefill/mixed)
- `TorchCudagraphCompiler` — extended with `piecewise_enabled` flag
- `CompileModelConfig` — extended with piecewise config fields
- `default.yaml` — added `piecewise_enabled: false`, `piecewise_num_tokens: null`

### Phase 4: Runtime Integration ✅

**Task 4.1: Mixed batch arg generation** ✅

- `_get_mixed_args_kwargs()` creates single-sequence prefill inputs for capture
- Auto-generates power-of-2 bucket sizes when `piecewise_num_tokens: null`

### Phase 5: Runtime Token Matching — ⚠️ EXACT MATCH ONLY (No Padding)

#### The Problem

`piecewise_num_tokens` is a **discrete list of bucket sizes** (e.g., `[64, 128, 256]`).
At runtime, a prefill/mixed batch can arrive with **any** total token count.

#### Current Approach: Exact Match (No Padding)

`DualModeCapturedGraph.forward()` uses **exact `num_tokens` match** only:

```python
def forward(self, *args, **kwargs):
    if self._is_decode_only(**kwargs):
        return self.monolithic(*args, **kwargs)

    num_tokens = self._get_num_tokens(**kwargs)  # e.g., input_ids.numel()

    if num_tokens in self._captured_num_tokens:  # exact match only
        return self.piecewise(*args, num_tokens=num_tokens, **kwargs)

    # No match → eager fallback
    return self.piecewise.original_model(*args, **kwargs)
```

This means piecewise CG only fires when the total prefill token count **exactly**
matches a pre-captured bucket size. For other lengths, the model runs in eager mode.
See §2.6 for a full analysis of why padding doesn't work in AD's architecture.

## 4. Bugs Found & Fixed During Testing

### Bug 4.1: OOM from `copy.deepcopy(model)` ✅ FIXED

**Symptom**: `torch.OutOfMemoryError` during `PiecewiseCapturedGraph.prepare()`.
**Root cause**: `copy.deepcopy(model)` duplicated ALL weight tensors on GPU (~14GB for 7B model).
**Fix**: `GraphModule(model, copy.deepcopy(model.graph))` — shares parameters, copies only the
FX graph DAG (metadata, not weights). Zero-copy for parameters.

### Bug 4.2: OOM from cloning weight tensors in ADPiecewiseRunner ✅ FIXED

**Symptom**: OOM during CUDA graph capture for submodules with many weight args (MoE models).
**Root cause**: `entry.static_inputs = [a.clone() for a in flat_args]` cloned ALL tensor args,
including huge MoE expert weight tensors (GBs each).
**Fix**: During warmup, track `data_ptr()` of each arg. Args with stable addresses (weights)
are referenced directly (zero-copy). Only args with changing addresses (activations) are cloned.

### Bug 4.3: Missing `mamba_ssm_prepare_metadata` in dynamic ops registry ✅ FIXED

**Symptom**: `cudaErrorStreamCaptureUnsupported` during CUDA graph capture.
**Root cause**: `mamba_ssm_prepare_metadata` was not registered as a dynamic op, so it was
placed inside a static submodule. It does CPU math on CUDA tensors (`math.ceil(...)`)
which is illegal during stream capture.
**Fix**: Added `"auto_deploy::mamba_ssm_prepare_metadata"` to `_METADATA_PREP_OPS`.

### Bug 4.4: Wrong `num_tokens` inference in PiecewiseCapturedGraph ✅ FIXED

**Symptom**: Shape mismatches during CUDA graph replay.
**Root cause**: `PiecewiseCapturedGraph.forward()` inferred `num_tokens` from the first
tensor's `shape[0]`. For `input_ids = [1, num_tokens]`, this gives `1` (batch dim), not
`num_tokens`. Also, in kwargs iteration order, the first tensor might not be `input_ids`.
**Fix**: `num_tokens` is now passed explicitly from `DualModeCapturedGraph` to
`PiecewiseCapturedGraph` as a keyword argument.

### Bug 4.5: Shape mismatch from top-level padding ✅ FIXED (by removing padding)

**Symptom**: `RuntimeError: size of tensor a (64) must match size of tensor b (7)`.
**Root cause**: Padding `input_ids` from 7→64 without updating `batch_info_host` (still says 7).
Dynamic ops use real metadata → produce 7-token outputs → next static segment expects 64.
**Fix**: Removed all padding/truncation logic. Now uses exact `num_tokens` match only.
See §2.6 for the full architectural analysis.

### Bug 4.6: "CUDA Graph is empty" for trivial static submodules ✅ FIXED

**Symptom**: PyTorch warning `The CUDA Graph is empty` during capture.
**Root cause**: When two dynamic ops are adjacent in the graph (e.g., `prepare_metadata` →
`SSM`, or `attention` → `prepare_metadata`), the static partition between them contains
only FX graph plumbing (`getitem`, `reshape`) — no actual CUDA kernel launches. Wrapping
these in `ADPiecewiseRunner` and capturing a CUDA graph produces an empty graph.
**Fix**: Added `_submod_has_cuda_ops()` check in `PiecewiseCapturedGraph.prepare()`.
Trivial submodules (no `call_module`, no non-trivial `call_function`/`call_method`) are
now skipped — they run eagerly instead of being wrapped in `ADPiecewiseRunner`. This
eliminates the warning and avoids unnecessary CUDA graph overhead for no-op segments.

## 5. File Changes Summary

| File | Change | Status |
|------|--------|--------|
| `compile/piecewise_utils.py` | **NEW** — `is_dynamic_cached_op()`, `split_graph_at_dynamic_ops()`, `swap_to_inplace_ops()` | ✅ |
| `compile/piecewise_runner.py` | **NEW** — `ADPiecewiseRunner` with warmup ptr tracking + class-level num_tokens context | ✅ |
| `compile/inplace_ops.py` | **NEW** — All 7 inplace variants + `ORIGINAL_TO_INPLACE` mapping | ✅ |
| `compile/backends/torch_cudagraph.py` | **MODIFY** — `PiecewiseCapturedGraph` (with trivial-submod filtering), `DualModeCapturedGraph` (exact match), `_submod_has_cuda_ops()`, dual-mode `TorchCudagraphCompiler` | ✅ |
| `transform/library/compile_model.py` | **MODIFY** — `piecewise_enabled`, `piecewise_num_tokens` in config + `_generate_default_piecewise_num_tokens()` + `_get_mixed_args_kwargs` | ✅ |
| `config/default.yaml` | **MODIFY** — added `piecewise_enabled: false`, `piecewise_num_tokens: null` (null = auto-generate) | ✅ |
| `shim/ad_executor.py` | **NO CHANGE NEEDED** — dispatch handled inside `DualModeCapturedGraph` | ✅ |

**Key design choices:**

- All inplace variants consolidated in a single `compile/inplace_ops.py`
- `ADPiecewiseRunner` uses class-level context for `num_tokens` (not arg shape inference)
- Weight tensors are referenced (zero-copy), only activations are cloned
- Shallow FX graph copy (`GraphModule(model, deepcopy(graph))`) to avoid duplicating weights
- Trivial static submodules (no CUDA ops) are skipped, not wrapped — eliminates empty CG warnings

## 6. Testing Strategy

1. **Unit tests for graph splitting**: Verify that a simple model graph is correctly
   split at attention boundaries.
1. **Unit tests for PiecewiseRunner**: Verify warmup → capture → replay state machine.
1. **Integration test**: Run a small model (e.g., Llama-2-7b) with piecewise enabled
   and verify correctness against eager baseline.
1. **Hybrid model test**: Run a Mamba/Nemotron model to verify SSM split points work.
1. **Performance benchmark**: Compare piecewise vs eager for mixed batches to quantify
   the speedup.

## 7. Risks & Open Questions

1. **Low cache hit rate with exact match**: Without padding, piecewise CG only fires
   when `num_tokens` exactly matches a bucket. In production, this is rare — most
   prefill lengths won't hit an exact power-of-2 bucket. **Mitigation**: JIT capture
   (capture on first occurrence) or scheduler-level padding (future work). See §2.6.

1. **Memory overhead**: Each captured CUDA graph segment holds its own memory pool.
   With many segments × many `num_tokens` values, memory usage could grow. May need
   to limit `piecewise_num_tokens` or share pools across segments. (Graph pools ARE
   shared across runners via `ADPiecewiseRunner.graph_pool`.)

1. **Inplace op correctness**: The inplace ops must produce bit-identical results to
   their non-inplace counterparts. Need thorough numerical validation.

1. **Interaction with `torch.compile`**: If `torch-opt` backend is used (torch.compile +
   cudagraph), piecewise should still work. The compiled submodules would be captured
   inside `ADPiecewiseRunner`.

## 8. Priority & Phasing — IMPLEMENTATION STATUS

| Phase | Tasks | Status |
|-------|-------|--------|
| **Phase 1** | Graph splitting + runner | ✅ IMPLEMENTED |
| **Phase 2** | Inplace ops + graph transform | ✅ IMPLEMENTED |
| **Phase 3** | Dual-mode backend (monolithic decode + piecewise prefill) | ✅ IMPLEMENTED |
| **Phase 4** | Runtime integration (capture-time arg generation) | ✅ IMPLEMENTED |
| **Phase 5** | Runtime token matching | ⚠️ EXACT MATCH ONLY (see §2.6 for analysis) |
| **Bug fixes** | OOM, shape mismatch, missing ops, empty CG | ✅ ALL FIXED (see §4) |
| **Testing** | Unit + integration + perf | 🔲 TODO |

### Known Limitations:

- **No runtime padding** — piecewise CG only fires on exact `num_tokens` match (see §2.6)
- **First-call capture latency** — warmup + capture adds latency for each new bucket size

### Files Created/Modified:

```
NEW:   compile/piecewise_utils.py     — Graph splitting infrastructure (13 dynamic ops)
NEW:   compile/piecewise_runner.py    — CUDA graph segment runner (ptr tracking, context)
NEW:   compile/inplace_ops.py         — 7 inplace op variants + mapping

MOD:   compile/backends/torch_cudagraph.py  — PiecewiseCapturedGraph + DualMode + Compiler
                                              + _submod_has_cuda_ops() trivial-segment filter
MOD:   transform/library/compile_model.py   — Config + mixed batch args + auto-bucket gen
MOD:   config/default.yaml                  — piecewise_enabled + piecewise_num_tokens
```

### Usage:

```yaml
# In config YAML or programmatic override:
compile_model:
  backend: torch-cudagraph
  piecewise_enabled: true
  piecewise_num_tokens: [128, 256, 512, 1024]  # or null for auto-generate
  cuda_graph_batch_sizes: [1, 2, 4, 8]
```

### Next Steps (Priority Order):

1. **JIT capture at runtime** — capture piecewise graphs on first occurrence of a new
   `num_tokens` value, cache for subsequent replays (eliminates exact-match limitation,
   no padding needed, no metadata inconsistency)
1. **Scheduler-level padding** — pad requests to bucket sizes at the scheduler level so
   ALL metadata is consistent (enables pad-to-nearest-bucket like native TRTLLM).
   Requires changes outside AD, in the scheduler/executor.
1. **SSM op refactoring** — remove data-dependent branching to make SSM capturable
   (reduces segment count for hybrid models)
