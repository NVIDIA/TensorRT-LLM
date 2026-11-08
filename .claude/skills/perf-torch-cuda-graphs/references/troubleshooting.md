<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# CUDA Graph Troubleshooting

## Capture Failures (Explicit Errors)

### Error Code Reference

| Code | Name | Cause | Fix |
|------|------|-------|-----|
| 900 | `StreamCaptureUnsupported` | Forbidden op during capture | Remove sync ops, pre-allocate pinned memory |
| 901 | `StreamCaptureInvalidated` | External work/thread interrupted capture | Use `thread_local` capture mode |
| 904 | `StreamCaptureUnjoined` | Side stream didn't rejoin before end | Add `capture_stream.wait_stream(side_stream)` |
| 905 | `StreamCaptureIsolation` | Dependency on uncaptured work | Move dependency inside capture or before it |
| 906 | `StreamCaptureImplicit` | Forbidden dependency on default stream | Initialize on side stream, warmup before capture |

### Synchronization During Capture (Error 900)

**Triggers:** `.item()`, `.cpu()`, `.numpy()`, `print(tensor)`, `torch.cuda.synchronize()`

**Fix:** Move all sync operations outside `torch.cuda.graph()` context.

### Pinned Memory Allocation (Error 900 in Global Mode)

**Triggers:** `cudaHostAlloc`, `cudaFreeHost`, hidden `cudaEventQuery()` in allocator

**Fix:** Pre-allocate pinned memory before capture, or use `pin_memory=False` in DataLoader.

### Stream Fork-Join Violations

**Missing entry sync (Error 900):**
```python
# Fix: side stream must wait for capture stream
side_stream.wait_stream(capture_stream)
```

**Missing exit sync (Error 904):**
```python
# Fix: capture stream must wait for side stream
capture_stream.wait_stream(side_stream)
```

**External dependency (Error 905):**
```python
# Fix: use external events for cross-graph sync
event = torch.cuda.Event(external=True)
```

### Gradient Tape Default Stream Conflict (Error 906)

**Cause:** `AccumulateGrad` created on default stream, capture on side stream.

**Fix:** Warmup on side stream before capture:
```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):  # 11 for DDP
        out = model(static_input)
        out.sum().backward()
torch.cuda.current_stream().wait_stream(s)
```

### DataLoader pin_memory Thread (Error 901)

**Cause:** Background `pin_memory` thread calls `cudaEventQuery()` during capture.

**Fix:** Set `capture_error_mode="thread_local"`, or `pin_memory=False`.

### RNG Failures

| Error | Cause | Fix |
|-------|-------|-----|
| "increase offset for generator not in capture" | Custom generator unregistered | `g.register_generator_state(gen)` |
| Runtime error on `get_rng_state()` | CPU-side RNG API during capture | Use `gen.graphsafe_get_state()` |
| "current_seed during capture" | Activation checkpointing | Set `preserve_rng_state=False` |
| "incompatible with .grad()" | Reentrant checkpointing | Use `use_reentrant=False` |
| torch.compile + RNG | First compilation saves RNG state | Warmup compiled functions before capture |

---

## Numerical Errors (Silent — No Error Messages)

The most dangerous failure mode: code runs, results are wrong.

### Dynamic Behavior Frozen at Capture Time

| Pattern | Symptom | Fix |
|---------|---------|-----|
| Python `if/else` on tensor | Always takes branch from capture | `torch.where()` |
| Dynamic loop count | Fixed iteration count | Fixed-range loops |
| Tensor reassignment | Reads stale address | `.copy_()` in-place |
| Python scalar parameter | Baked as constant | GPU tensor + `.fill_()` |
| Shape change | Truncated/corrupted output | Padding or bucketing |

### Output Tensor Reuse

Graph outputs occupy static memory reused every replay. Collecting outputs across
iterations without cloning gives multiple references to the same (overwritten) buffer.

**Fix:** `results.append(static_output.clone())` after every `g.replay()`.

### Memory Pool Sharing Corruption

**Replay order mismatch:** Capturing as [fwd0, bwd0, fwd1, bwd1] but replaying as
[fwd0, fwd1, bwd0, bwd1] — backward passes overwrite intermediate activations.

**Fix:** Match replay order to capture order, or use separate pools per graph.

**Concurrent replay:** Two graphs sharing a pool replayed on different streams race.

**Fix:** Separate pools or serialize with stream sync.

### Missing Operations

**Library stream unawareness:** External code launching ops on non-capture stream.
Ops execute during capture (looking correct) but are absent from the graph.

**Fix:** Ensure libraries use `torch.cuda.current_stream()`.

### Debugging Strategy

Compare eager vs graphed execution layer-by-layer:
```python
for i, (eager, graph) in enumerate(zip(eager_outs, graph_outs)):
    diff = (eager - graph).abs().max()
    if diff > 1e-5:
        print(f"Divergence at layer {i}: max diff = {diff}")
        break
```

---

## Memory Issues

### Illegal Memory Access / Segfault

| Cause | Fix |
|-------|-----|
| Input tensor garbage-collected | Keep persistent reference |
| Tensor variable reassigned | Use `.copy_()` instead |
| CPU tensor freed before H2D copy | Maintain CPU tensor lifetime |
| Host pointer array freed (grouped GEMM) | Keep host arrays alive |

### Out of Memory

| Cause | Fix |
|-------|-----|
| Static inputs accumulate across graphs | Share input buffers |
| Pools can't share memory | `pool=g1.pool()` for sequential graphs |
| Post-capture tensors can't use graph pool | Expand capture range |
| Fragmentation across pools | `expandable_segments:True` in allocator |
| `record_stream()` defers recycling | `graph_capture_record_stream_reuse:True` (PT 2.9+) |
| AccumulateGrad stream mismatch | Initialize on capture stream |
| `cudaFree` suppressed during capture | Free tensors before entering capture |

### Debugging OOM

```python
# Enable memory history recording
torch.cuda.memory._record_memory_history()

# Run your code...

# Check stats
print(torch.cuda.memory_summary())
```

Compare Reserved memory (not just Allocated) between graph and non-graph runs.

---

## Performance Issues

### Insufficient Speedup

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Already GPU-bound | GPU util >95% in nsys | Graphs won't help |
| Wrong capture scope | Bottleneck outside graph | Profile, move capture |
| Too many small graphs | `torch.compile` graph breaks | Remove `.item()`, control flow |
| Input copy overhead | Large input tensors | Use `out=` parameter, write directly |
| DDP allreduce serialized | `make_graphed_callables` | Use `torch.cuda.graph()` for full iteration |
| Channel serialization | >32 concurrent streams | Limit streams or `CUDA_DEVICE_MAX_CONNECTIONS=128` |

### Profiling

```bash
# Graph-level profiling
nsys profile --cuda-graph-trace=graph python train.py

# Kernel-level profiling (higher overhead)
nsys profile --cuda-graph-trace=node python train.py
```

---

## TE / MCore-Specific Issues

| Issue | Context | Fix |
|-------|---------|-----|
| FP8 scaling corruption | TE `make_graphed_callables` without `fp8_autocast` during replay | Wrap replay with `fp8_autocast(enabled=True)` |
| PP replay order mismatch | TE/CudaGraphManager with wrong execution order | Ensure replay matches `_order` / capture sequence |
| `--no-check-for-nan-in-loss-and-grad` missing | FullCudaGraphWrapper capture failure | Add required flag (NaN check causes `.item()` sync) |
| `--te-rng-tracker` missing | FullCudaGraphWrapper RNG failure | Required for device-tensor RNG state |
| Weight caching stale | TE with `fp8_weight_caching=True` after significant weight change | Set `is_first_microbatch=True` or recapture |
| Buffer sharing failure | CudaGraphManager with ops between layers | Disable `cuda_graph_share_io_buffers` or move ops inside layers |
| Single mempool OOM | CudaGraphManager with `cuda_graph_use_single_mempool=True` | Switch to default (separate pools per microbatch) |
| Graph count explosion | CudaGraphManager with many layers x microbatches | Use separate pools (default) for graph reuse across microbatches |
