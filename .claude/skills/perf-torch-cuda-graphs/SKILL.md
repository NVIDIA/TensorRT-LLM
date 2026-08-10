---
name: perf-torch-cuda-graphs
description: >-
  Apply CUDA Graphs to PyTorch workloads — API selection (torch.compile, PyTorch
  make_graphed_callables, TE make_graphed_callables, MCore CudaGraphManager,
  FullCudaGraphWrapper, manual torch.cuda.graph), code compatibility, capture
  workflows, dynamic pattern handling, and troubleshooting.
  Triggers: CUDA graph, torch.cuda.graph, make_graphed_callables, reduce-overhead,
  graph capture, graph replay, kernel launch overhead, CudaGraphManager,
  FullCudaGraphWrapper, full-iteration graph, stream capture.
tags:
  - cuda-graph
  - optimization
  - pytorch
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# CUDA Graphs for PyTorch

CUDA Graphs capture a sequence of GPU operations once and replay them with
minimal CPU overhead. This skill guides applying CUDA Graphs to PyTorch
training and inference workloads using native PyTorch APIs, Transformer
Engine, and Megatron-LM.

## When to Use

Reach for this skill when you encounter:

- **Triggers**: User wants to optimize with CUDA Graphs, reduce kernel launch
  overhead, or speed up training/inference loops
- **Symptoms**: Low GPU utilization (<80%), many small kernel launches (<50 us
  each), CPU-bound training, high kernel launch latency visible in Nsight
  Systems profiles
- **Keywords**: "CUDA graph", "torch.cuda.graph", "make_graphed_callables",
  "reduce-overhead", "graph capture", "graph replay", "kernel launch overhead",
  "CudaGraphManager", "FullCudaGraphWrapper", "full-iteration graph", "stream
  capture"

Do NOT use this skill for:

- General PyTorch performance tuning unrelated to kernel launch overhead
- CUDA kernel development or custom CUDA C++ code
- Host-device sync elimination only (use **perf-torch-sync-free** skill instead)
- Nsight Systems profiling (use **perf-nsight-systems** skill)
- TensorFlow/JAX graph compilation (different APIs entirely)

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| PyTorch | >= 1.10 | `torch.cuda.graph()` available |
| CUDA | >= 11.0 | Graph update APIs |
| GPU | NVIDIA (any) | Required for CUDA |
| Nsight Systems | any | Optional, for profiling |
| APEX | any | Optional, for capturable optimizers |
| Transformer Engine | >= 2.2 | Optional, for FP8-aware graphing |
| Megatron-LM | core >= 0.14.0 | Optional, for CudaGraphManager / FullCudaGraphWrapper |

## API Selection Guide

Choose the API based on your framework and performance needs.

| Situation | API | Workflow |
|-----------|-----|---------|
| Quick experiment, unknown graph boundaries | `torch.compile(mode="reduce-overhead")` | Workflow 2 |
| Training, need autograd, no FP8/PP | `torch.cuda.make_graphed_callables()` | Workflow 3 |
| Any PyTorch model, FP8 or PP support | TE `make_graphed_callables` | Workflow 4 |
| Megatron-LM, per-layer, automatic | MCore `CudaGraphManager` | Workflow 5 |
| Maximum perf, full-iteration capture | MCore `FullCudaGraphWrapper` | Workflow 6 |
| Full manual control, custom pipelines | `torch.cuda.graph()` | Workflow 7 |

**Decision flowchart:**

1. Using Megatron-LM with FP8/PP?
   - Yes, want maximum perf with static workload --> Workflow 6 (FullCudaGraphWrapper)
   - Yes, want per-layer automatic graphing --> Workflow 5 (CudaGraphManager)
   - Yes, want manual control over what gets graphed --> Workflow 4 (TE make_graphed_callables)
2. Using Transformer Engine without Megatron?
   - Yes, need FP8 or PP --> Workflow 4 (TE make_graphed_callables)
3. General PyTorch?
   - Want zero effort, okay with fragmented graphs --> Workflow 2 (torch.compile)
   - Want autograd support, training loop --> Workflow 3 (PyTorch make_graphed_callables)
   - Want full manual control --> Workflow 7 (torch.cuda.graph)

**Strategy:** Start with the highest-level API available for your framework.
Move to lower-level APIs only if you need more control, hit limitations, or
do not achieve the expected performance improvement.

## Workflows

### Workflow 1: Profile and Decide Whether Graphs Help

Goal: Determine if CUDA Graphs will benefit your workload before investing
effort.

1. Profile with Nsight Systems:
   ```bash
   nsys profile --cuda-graph-trace=graph python train.py
   ```
2. Check GPU utilization -- if already >95%, graphs won't help much.
3. Look for gaps between kernel launches (CPU overhead) and many small kernels
   (<50 us each). These are the targets for graphing.
4. Annotate regions of interest to correlate idle GPU time with code:
   ```python
   with torch.cuda.nvtx.range("forward"):
       output = model(input)
   ```
5. Estimate benefit: count kernels per iteration. Workloads with hundreds of
   small kernels and <80% GPU utilization are strong candidates.

Expected result: Identified bottleneck regions with low GPU occupancy between
kernels. Proceed to the appropriate workflow from the API Selection Guide.

### Workflow 2: torch.compile(mode="reduce-overhead")

Goal: Automatic CUDA Graph capture with zero manual effort.

When to use: Quick experiment, unknown graph boundaries, already using
`torch.compile`.

Steps:

1. Decorate the training step with `@torch.compile(mode="reduce-overhead")`:
   ```python
   @torch.compile(mode="reduce-overhead")
   def train_step(model, x, target, criterion):
       output = model(x)
       loss = criterion(output, target)
       loss.backward()
       return loss
   ```
2. Run the training loop normally -- graphs are captured automatically.
3. Profile with Nsight Systems to see captured graphs:
   ```bash
   nsys profile --cuda-graph-trace=graph python train.py
   ```
4. If you see too many small graphs (graph fragmentation), check for graph
   breaks: `.item()`, `print()`, data-dependent control flow. Fix these or
   escalate to Workflow 3+.

Trade-offs:
- Zero effort, but may create fragmented small graphs.
- Limited control over what gets graphed.
- Graph fragmentation limits performance gains compared to manual approaches.

### Workflow 3: torch.cuda.make_graphed_callables()

Goal: Training with autograd support. Separate forward/backward graphs.

When to use: Training with custom loops, non-FP8, need autograd.

Steps:

1. Prepare sample inputs matching training batch shape:
   ```python
   sample_input = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
   ```
2. Create the graphed model:
   ```python
   graphed_model = torch.cuda.make_graphed_callables(
       model, (sample_input,), num_warmup_iters=3
   )
   ```
3. Use `graphed_model` as a drop-in replacement in the training loop:
   ```python
   for data, target in dataloader:
       optimizer.zero_grad()
       output = graphed_model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
   ```
4. If using AMP, set `cache_enabled=False`:
   ```python
   for data, target in dataloader:
       optimizer.zero_grad()
       with torch.amp.autocast("cuda", cache_enabled=False):
           output = graphed_model(data)
           loss = criterion(output, target)
       loss.backward()
       optimizer.step()
   ```
5. If using DDP, construct DDP on a side stream and use 11 warmup iters:
   ```python
   os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
   s = torch.cuda.Stream()
   with torch.cuda.stream(s):
       model = DistributedDataParallel(model)
   torch.cuda.current_stream().wait_stream(s)

   graphed_model = torch.cuda.make_graphed_callables(
       model, (sample_input,), num_warmup_iters=11
   )
   ```

Limitations:
- No double backward (higher-order gradients).
- No module hooks during capture.
- Module structure is frozen after graphing (no add/remove parameters).
- Argument signature must match `sample_args` exactly.

### Workflow 4: TE make_graphed_callables

Goal: Per-callable graphing with FP8 support and pipeline parallelism.

When to use: FP8 training, PP with manual scheduling, non-Megatron models
needing FP8, or any PyTorch model that needs FP8-aware CUDA Graphs.

Steps:

1. Import and configure:
   ```python
   from transformer_engine.pytorch.graph import make_graphed_callables
   from transformer_engine.pytorch.fp8 import fp8_autocast
   ```
2. Prepare sample inputs (one per callable per microbatch per chunk):
   ```python
   sample_args = tuple(
       (torch.randn(batch_size, seq_len, hidden_size, device="cuda"),)
       for _ in range(num_callables * num_microbatches)
   )
   ```
3. Define pipeline schedule if using PP (1-indexed chunk IDs, positive=fwd,
   negative=bwd):
   ```python
   # Example: 2 chunks, 3 microbatches
   layer_order = [1, 2, 1, 2, 1, 2, -2, -1, -2, -1, -2, -1]
   ```
4. Wrap layers in CUDA Graphs:
   ```python
   graphed_layers = make_graphed_callables(
       tuple(layers),
       sample_args=sample_args,
       fp8_enabled=True,
       fp8_recipe=fp8_recipe,
       fp8_weight_caching=True,
       _order=layer_order,  # None for no PP
   )
   ```
5. Training loop -- wrap with `fp8_autocast` during replay:
   ```python
   with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
       for layer in graphed_layers[start:end]:
           x = layer(x, is_first_microbatch=(mb_idx == 0))
   # FP8 scaling auto-updated on fp8_autocast exit
   optimizer.step()
   ```

Key points:
- **AOT capture**: Graphs captured before the training loop when you call
  `make_graphed_callables()`.
- **Replay order must match `_order`**: The training loop must execute graphs
  in the same interleaved order as specified during capture.
- **`fp8_autocast` required during replay**: Without it, FP8 state is not
  properly configured.
- **Weight caching**: `fp8_weight_caching=True` caches FP8 weight
  quantization across microbatches; pass `is_first_microbatch` kwarg to
  control when weights are requantized.

For full API details, see `references/api-te-megatron.md`.

### Workflow 5: MCore CudaGraphManager (Per-Layer)

Goal: Automatic per-layer graphing for Megatron-LM training.

When to use: Megatron-LM training, especially with PP > 1. Default choice
for Megatron users.

Steps:

1. Enable via CLI flags (no code changes needed):
   ```bash
   python pretrain_gpt.py \
       --enable-cuda-graph \
       --cuda-graph-num-warmup-steps 3
   ```
2. Or enable via Python config:
   ```python
   config = TransformerConfig(
       enable_cuda_graph=True,
       cuda_graph_num_warmup_steps=3,
   )
   ```
3. Training loop is unchanged -- graphs are captured automatically after
   warmup iterations.

Key points:
- **Megatron layers only**: Works with `TransformerLayer` and `MambaLayer`.
- **JIT capture**: Records execution order during warmup, captures graphs
  after warmup completes, then replays on subsequent iterations.
- **Automatic FP8 handling**: Uses `fp8_autocast(..., _graph=True)` to skip
  per-layer amax reduction; reduction happens once after all backward graphs.
- **Automatic PP support**: Handles microbatch interleaving automatically.
- **Memory savings**: Set `cuda_graph_share_io_buffers=True` to share I/O
  buffers between layers (requires no operations between layers).
- **Memory pool strategy**: Default uses separate pools per microbatch for
  graph reuse. Set `cuda_graph_use_single_mempool=True` for shared pool
  (higher graph count but may reduce fragmentation).

### Workflow 6: MCore FullCudaGraphWrapper (Full-Iteration)

Goal: Maximum performance. Captures forward+backward for all microbatches
as a single graph.

When to use: Maximum performance priority, static workloads, Megatron-LM
training.

Steps:

1. Enable via CLI flags:
   ```bash
   python pretrain_gpt.py \
       --enable-cuda-graph \
       --cuda-graph-scope full_iteration \
       --cuda-graph-warmup-steps 1 \
       --te-rng-tracker \
       --no-check-for-nan-in-loss-and-grad
   ```
2. Ensure all forward+backward code is capturable (no `.item()`, no NaN
   check, no dynamic control flow).
3. Optimizer remains in eager mode by default (outside the graph). Can be
   included inside the graph for maximum performance.

Key points:
- **Only 2 graphs total**: One for training, one for validation.
- **`--te-rng-tracker` required**: Standard RNG uses CPU scalars that cannot
  be captured; TE RNG uses device tensors compatible with graphs.
- **`--no-check-for-nan-in-loss-and-grad` mandatory**: NaN checking uses
  `.item()` which requires CPU-GPU sync, forbidden during capture.
- **StaticBufferLoader**: Pre-allocates input buffers for all microbatches
  during warmup.
- **Optimizer in/out of graph**: Inside = maximum performance (all optimizer
  kernels captured). Outside = more flexible (can change optimizer/LR without
  recapture).
- **JIT capture**: Graph captured during training at iteration
  `warmup_steps + 1`.

### Workflow 7: torch.cuda.graph() (Manual)

Goal: Full control over capture and replay. Custom pipelines, full-iteration
capture without Megatron.

When to use: Need fine-grained control, non-Megatron full-iteration capture,
custom pipelines.

**Inference pattern:**

1. Pre-allocate static input/output tensors:
   ```python
   static_input = torch.randn(batch_size, *shape, device="cuda")
   ```
2. Warmup on a side stream (3 iterations, 11 for DDP):
   ```python
   s = torch.cuda.Stream()
   with torch.cuda.stream(s):
       for _ in range(3):
           _ = model(static_input)
   torch.cuda.current_stream().wait_stream(s)
   ```
3. Capture the graph:
   ```python
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       static_output = model(static_input)
   ```
4. Replay loop -- update inputs via `.copy_()`, clone outputs:
   ```python
   for data in loader:
       static_input.copy_(data)
       g.replay()
       result = static_output.clone()
   ```

**Full training pattern (fwd+bwd+optimizer in one graph):**

```python
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

static_input = torch.randn(batch_size, *shape, device="cuda")
static_target = torch.randint(0, num_classes, (batch_size,), device="cuda")

# Warmup
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", cache_enabled=False):
            out = model(static_input)
            loss = criterion(out, static_target)
        loss.backward()
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", cache_enabled=False):
        static_output = model(static_input)
        static_loss = criterion(static_output, static_target)
    static_loss.backward()

# Replay loop
for data, target in loader:
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
    optimizer.step()
```

**DDP setup:**

```python
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    model = DistributedDataParallel(model)

# 11 warmup iterations for DDP
with torch.cuda.stream(s):
    for _ in range(11):
        out = model(static_input)
        out.sum().backward()
torch.cuda.current_stream().wait_stream(s)

# Capture on the same side stream
with torch.cuda.graph(g):
    static_output = model(static_input)
```

**Memory pool sharing for multiple graphs:**

```python
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    out1 = model_a(static_in_a)

# Second graph shares first graph's memory pool
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2, pool=g1.pool()):
    out2 = model_b(static_in_b)
```

**Custom RNG registration:**

```python
gen = torch.cuda.default_generators[0]
g = torch.cuda.CUDAGraph()
g.register_generator_state(gen)
with torch.cuda.graph(g):
    out = model(static_input)  # RNG state properly captured
```

### Navigating Between Workflows

- **torch.compile gives insufficient speedup** --> escalate to
  `make_graphed_callables` (Workflow 3) for larger, fewer graphs.
- **make_graphed_callables can't handle FP8/PP** --> TE
  `make_graphed_callables` (Workflow 4).
- **Need Megatron per-layer automatic** --> CudaGraphManager (Workflow 5).
- **Want maximum perf** --> FullCudaGraphWrapper (Workflow 6) or manual
  full-iteration capture (Workflow 7).
- **Something too hard to graph** --> partial capture (graph what you can,
  leave the rest in eager mode).
- **User wants best absolute perf** --> skip directly to Workflow 6
  (Megatron) or Workflow 7 (manual).
- **Start small, expand progressively**: Begin with one module/layer. Verify
  correctness. Then expand to more layers, full forward pass, add backward,
  and eventually full iteration with optimizer.

## Making Code Graph-Compatible

These principles apply to all workflows. Code inside the captured region must
satisfy three constraints.

### Principle 1: GPU-Only

Only GPU operations are captured. CPU-side code (Python logic, I/O, logging)
executes during capture but is eliminated during replay.

Violations:
- File I/O: `data = torch.load("file.pt")` won't reload on replay
- CPU preprocessing: `tokens = tokenizer.encode(text)` won't re-tokenize
- Logging: `print(f"Step {i}")` won't print during replay
- CPU RNG: `random.randint(0, 10)` won't regenerate
- CPU bookkeeping: `buffer.append(tensor)` won't populate during replay

Fix: Move all CPU-side operations outside the graphed region.

### Principle 2: Sync-Free

No CPU-GPU synchronization inside the graph. The CPU queues work continuously
without waiting for GPU results.

Violations:
- `.item()` to get scalar values
- `.cpu()` to move tensors for inspection
- `torch.cuda.synchronize()` or `stream.synchronize()`
- `print(tensor)` (implicitly syncs)

Fix: **Invoke the perf-torch-sync-free skill** for systematic detection and
elimination of sync points. Use `torch.cuda.set_sync_debug_mode("warn")` to
find hidden syncs.

### Principle 3: Static

All operations, control flow, memory addresses, and shapes must be fixed
across all replays.

Violations and fixes:

| Dynamic aspect | Fix |
|---------------|-----|
| `if loss > threshold:` | `torch.where(condition, a, b)` |
| `input = new_tensor` (address changes) | Pre-allocate + `.copy_()` |
| Python scalars (lr, temperature) | GPU tensor + `.fill_()` |
| Variable batch size / sequence length | Padding or bucketing |
| MoE / dynamic routing | Partial graphing |

For detailed patterns, see `references/patterns-dynamic.md`.

### Compatibility Checklist

Verify every item before attempting capture:

- [ ] No `.item()`, `.cpu()`, `.numpy()`, `print(tensor)` inside graph
- [ ] No `torch.cuda.synchronize()` or `stream.synchronize()`
- [ ] No `if tensor_value:` -- use `torch.where()` instead
- [ ] All inputs pre-allocated, updated via `.copy_()`
- [ ] All shapes fixed (use padding or bucketing for variable sizes)
- [ ] Python scalars --> GPU tensors with `.fill_()`
- [ ] Output tensors `.clone()`d before next replay
- [ ] `cache_enabled=False` with `torch.amp.autocast`
- [ ] Custom RNG generators registered with `graph.register_generator_state()`
- [ ] Use `graphsafe_get_state()` / `graphsafe_set_state()` for RNG
- [ ] Warmup completed (3 standard, 11 for DDP)
- [ ] DDP: `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`, construct on side stream
- [ ] DDP: NCCL >= 2.9.6 for full graph capture
- [ ] Libraries/extensions use `torch.cuda.current_stream()`, not default stream
- [ ] No pinned memory allocation during capture (triggers hidden event query)
- [ ] `activation_checkpointing`: `preserve_rng_state=False`
- [ ] Global tensors used in graph kept alive (not deleted/reassigned)
- [ ] No `torch.compile` functions inside manual capture without prior warmup
- [ ] Gradient clipping uses sync-free `clip_grad_norm_` (PyTorch >= 1.13)

For the complete checklist with references, see `references/patterns-compatibility.md`.

## Output Formats

**Success indicators:**
- `g.replay()` completes without errors
- Outputs match eager mode within tolerance (`torch.allclose`)
- Nsight Systems profile shows single graph launch replacing many kernels
- GPU utilization increases, training/inference latency decreases

**Key metrics:**

| Metric | How to Check |
|--------|-------------|
| Correctness | `torch.allclose(eager, graphed, rtol=1e-5)` |
| Speedup | Wall-clock time comparison |
| GPU utilization | `nvidia-smi` or Nsight Systems timeline |
| Memory overhead | `torch.cuda.memory_summary()` |

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `StreamCaptureUnsupported` (900) | Sync op during capture (`.item()`, `.cpu()`) | Move sync outside graph |
| `StreamCaptureInvalidated` (901) | Background thread (e.g., pin_memory) | `capture_error_mode="thread_local"` |
| `StreamCaptureUnjoined` (904) | Side stream didn't rejoin capture stream | `capture_stream.wait_stream(side_stream)` |
| `StreamCaptureImplicit` (906) | AccumulateGrad on default stream | Warmup on side stream before capture |
| Illegal memory access | Input tensor freed/reassigned | Keep persistent ref, use `.copy_()` |
| Wrong numerical results | Dynamic behavior frozen at capture | See `references/patterns-compatibility.md` |
| OOM with multiple graphs | Pools can't share memory | `pool=g1.pool()` for sequential graphs |
| No speedup | Already GPU-bound or wrong capture scope | Profile with nsys first (Workflow 1) |
| FP8 scaling corruption | TE without `fp8_autocast` during replay | Wrap with `fp8_autocast(enabled=True)` |
| PP replay order mismatch | Wrong execution order during replay | Match `_order` / capture sequence exactly |
| FullCudaGraphWrapper capture fail | NaN check or sync enabled | `--no-check-for-nan-in-loss-and-grad` |
| RNG failure with FullCudaGraphWrapper | Standard RNG not capturable | `--te-rng-tracker` |
| DDP capture failure | Async error handling watchdog | `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` |
| DDP AccumulateGrad on default stream | DDP constructed on default stream | Construct DDP in side stream context |
| Autocast cache invalidation | Cached cast tensors freed on exit | `cache_enabled=False` |

For detailed troubleshooting, see `references/troubleshooting.md`.

## Finding More Information

Use this 3-tier lookup hierarchy -- start at Tier 1 and escalate only when
needed.

### Tier 1: This File (SKILL.md)

You are reading it now. The workflows, compatibility checklist, and error
table above cover the most common tasks. Search this file first before going
deeper.

### Tier 2: references/ Directory

The `references/` directory beside this file contains distilled reference
material -- API details, patterns, and troubleshooting pages.

**How to search:**
1. Grep for your keyword across `references/` -- headers are designed to be
   grep-friendly.
2. Read only the file that grep points you to. Do not read every file.

Available references:
- `references/api-pytorch.md` -- PyTorch CUDA Graph APIs (`torch.cuda.graph`,
  `make_graphed_callables`, `torch.compile reduce-overhead`)
- `references/api-te-megatron.md` -- TE `make_graphed_callables`,
  CudaGraphManager, FullCudaGraphWrapper implementations
- `references/patterns-compatibility.md` -- GPU-only, sync-free, and static
  principles with full checklist
- `references/patterns-dynamic.md` -- Dynamic control flow, tensors, scalars,
  shapes: workarounds and patterns
- `references/troubleshooting.md` -- Capture failures, numerical errors,
  memory issues, performance issues

### Tier 3: Original Documentation

If Tiers 1-2 do not answer the question, consult the original sources:
- **NVIDIA guide**: `https://docs.nvidia.com/dl-cuda-graph/latest/index.html`
- **PyTorch docs**: `https://docs.pytorch.org/docs/stable/notes/cuda.html`
  (CUDA Graphs section)
- **TE docs**: `https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html`
- **Megatron Core docs**: `https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html`

Return to Tier 2 afterward and consider whether the answer should be distilled
into the references directory for next time.
