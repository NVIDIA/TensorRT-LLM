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

# Hotspot Classification Details

Extended classification reference with per-type indicators, code examples, and severity guidance.

---

## SYNC (Synchronization Barriers)
Forces CPU to wait for GPU completion. **Most impactful to fix.**
```python
# Explicit sync
torch.cuda.synchronize()
stream.synchronize()

# Implicit sync (data transfer)
tensor.item()           # Scalar extraction
tensor.cpu()            # Device transfer
tensor.numpy()          # Requires CPU tensor
tensor.tolist()         # Full tensor to Python list

# Async operation wait
future.wait()
event.synchronize()
```

**Fix via:** SYNC patterns in [patterns/sync-alloc.md](patterns/sync-alloc.md) (pre-compute on CPU, batch scalar extraction, GPU-side conditionals, non-blocking transfers).

---

## ALLOC (Memory Allocation)
GPU memory allocation is expensive; CPU allocation adds Python overhead.
```python
# GPU allocation in hot path
torch.zeros(size, device='cuda')
torch.empty(size, device='cuda')
tensor.clone()
tensor.contiguous()     # May allocate if not already contiguous
tensor.to('cuda')       # Allocation + copy

# CPU allocation overhead
[None] * large_n        # Python list allocation
dict.copy()             # Dictionary copy
```

**Fix via:** ALLOC patterns in [patterns/sync-alloc.md](patterns/sync-alloc.md) (pre-allocate at init, reusable buffer pool, avoid unnecessary clones).

---

## HOIST (Invariant Recomputation)
Per-layer recomputation of values that are identical across layers (step-invariant). Highest impact when the recomputed value involves `.item()` (combines HOIST + SYNC).
```python
# Per-layer function re-derives step-invariant values
def forward_per_layer(self, metadata, ...):
    pool = metadata.kv_cache_manager.get_unique_primary_pool()  # Same every layer
    stride = num_layers * tokens_per_block                       # Same every layer
    num_ctx = host_lengths[:n].sum().item()                      # .item() = SYNC!
```

**Detection heuristics:**
- In line_profiler: function called N_layers times/step (Hits = N_layers) with high Total time
- Inspect inputs: if only `layer_id` changes per call, everything else is invariant
- Check if the metadata object already carries the computed values
- `.item()` in per-layer code = HOIST + HOST_SYNC compound pattern

**Fix via:** HOIST patterns in [patterns/loop-iteration.md](patterns/loop-iteration.md) (Variant A: cache on first call; Variant B: pass from step level).

---

## PYLOOP (Python Loop Overhead)
Python interpreter overhead per iteration; vectorization can yield 10-100x speedup.
```python
# High overhead patterns
for i in range(len(items)):
    result = items[i].process()

for req in requests:    # If requests is large (>100)
    handle(req)

[f(x) for x in large_list]
```

**Fix via:** PYLOOP patterns in [patterns/loop-iteration.md](patterns/loop-iteration.md) (vectorize with tensor ops, torch.where, batch processing).

---

## REDUNDANT_ITER (Redundant Iteration)
Multiple passes over the same collection when data could be collected in one pass.
```python
# BEFORE: Two separate iterations over the same data
for request in all_requests:
    categorize(request)  # First pass

request_ids = [r.id for r in all_requests]  # Second pass -- redundant!

# AFTER: Collect during the existing iteration
request_ids = []
for request in all_requests:
    request_ids.append(request.id)  # Piggyback on first pass
    categorize(request)
```

**Fix via:** REDUNDANT_ITER patterns in [patterns/loop-iteration.md](patterns/loop-iteration.md) (collect during existing iteration, slicing instead of re-iteration).

---

## DEAD_WORK (Unnecessary Object Construction)
Creating objects or calling methods whose results are never used.
```python
# BEFORE: Always construct, then check if needed
for req in requests:
    params = ExpensiveParams(data=req.data)  # Always constructed
    params.process()                          # Always called
    if params.has_content():                  # Almost always False!
        use(params)

# AFTER: Guard with cheap check first
for req in requests:
    if req.data:                              # Cheap check
        params = ExpensiveParams(data=req.data)
        params.process()
        if params.has_content():
            use(params)
```

**Fix via:** DEAD_WORK patterns in [patterns/loop-iteration.md](patterns/loop-iteration.md) (guard with cheap check, model-level feature guard).

---

## CONTAINER (Container Operations)
Dictionary/set operations have O(1) average but constant factor matters in hot loops.
```python
# Repeated lookups
for req in requests:
    val = config_dict[req.key]      # Hash computation each time
    if req.id in seen_set:          # Set membership test

# Attribute access overhead
for req in requests:
    req.field1                       # __getattribute__ call
    req.field2
```

**Fix via:** CONTAINER patterns in [patterns/python-overhead.md](patterns/python-overhead.md) (cache dict lookups, local variable caching, `__slots__`).

---

## FUNCALL (Function Call Overhead)
Python function calls have ~100ns overhead; adds up in tight loops.
```python
# Property access in loop
for req in requests:
    x = req.computed_property       # Calls getter each time

# Repeated method calls
for item in items:
    result = helper_function(item)  # Call overhead per item
```

**Fix via:** FUNCALL patterns in [patterns/python-overhead.md](patterns/python-overhead.md) (hoist invariant calls, eliminate redundant assignments, inline hot functions).

---

## BOUNDARY (Python-C++ Boundary Crossings)
Accessing C++ properties or methods via bindings (nanobind/pybind11). Each crossing costs ~250ns.
```python
# nanobind-exposed property on a C++ object
for req in requests:             # N iterations
    val = req.cpp_property       # ~250ns per crossing (nanobind __get__)
    flag = req.another_property  # ~250ns again

# Detection via line_profiler:
# Look for high Hits with Per Hit ~0.2-0.5us on attribute access lines
# where the attribute is defined in C++ (check class with nanobind/pybind11)
```

**Detection heuristics:**
- Per Hit ~0.2-0.5us on a simple attribute access (pure Python attrs are ~0.05us)
- The accessed object's class is defined in C++ and exposed via nanobind or pybind11
- `Hits` count matches loop iteration count (not a nested loop artifact)

**Fix via:** BOUNDARY patterns in [patterns/python-overhead.md](patterns/python-overhead.md) (feature-gate, shadow with Python attribute at construction time).

---

## ENUM_CONSTRUCT (Expensive Enum Construction)
Python `Enum(int_value)` is expensive (~1.7us) due to metaclass lookup. Adds up in hot loops when a single value dominates >95% of calls.

**Detection heuristics:**
- Per Hit ~1.5-2.0us on a line that constructs an Enum from an int
- `Hits` count matches loop iteration count
- One enum value dominates >95% of cases (check with a Counter or sampling)
- The enum class uses standard `enum.Enum` or `enum.IntEnum`

**Python version note:** `Enum(value)` cost varies by Python version. Python 3.11+ improved enum construction speed (~0.8-1.0us), but the raw-int-comparison pattern is still faster (~0.05us). The ~1.7us figure above is from Python 3.10. Always verify with line_profiler on the target runtime.

**Fix via:** ENUM_CONSTRUCT patterns in [patterns/python-overhead.md](patterns/python-overhead.md) (raw int comparison, batch-collect with deferred conversion).

---

## GIL (Global Interpreter Lock / Threading)
Threading contention and synchronization primitives.
```python
# Lock contention
with self.lock:                     # May block waiting for lock
    shared_state.update(...)

# Queue operations
item = queue.get(timeout=0.1)       # Blocking call
queue.put(result)

# Thread coordination
event.wait()
condition.notify_all()
```

**Detection heuristics:**
- In line_profiler: Per Hit ~100us-10ms on `lock.acquire()`, `queue.get()`, `event.wait()`
- Thread CPU utilization low despite multiple active threads
- In nsys: CPU gaps correlating with Python thread scheduling, not GPU or NCCL

**Fix via:** GIL_CONTENTION patterns in [patterns/system.md](patterns/system.md) (fine-grained locking, batch under lock, thread-local storage).

---

## GC (Garbage Collection)
Python GC pauses causing periodic latency spikes. **Low priority for throughput** but can
dominate tail latency (P99/P999). Particularly impactful for inference servers with large
long-lived model objects — the collector repeatedly scans millions of parameter/tensor
objects that will never be freed.

**line_profiler cannot detect GC** — GC runs between profiled lines and is invisible
to line-level timing. Use nsys with NVTX markers instead.

**Detection:**
- Enable GC NVTX markers: `TLLM_PROFILE_RECORD_GC=1` (with nsys)
- In nsys timeline: periodic GPU idle gaps that don't correlate with kernels or API calls
- Check `gc.get_stats()` for high collection counts and collected object counts
- P99/P999 latency disproportionately higher than P50 (GC pauses hit tail)

**Fix via:** GC_MGMT patterns in [patterns/system.md](patterns/system.md):
1. `gc.freeze()` after model load — exempt long-lived objects from scanning (highest impact)
2. `gc.disable()` during latency-sensitive sections — eliminate pauses in bounded hot loops
3. `gc.set_threshold()` tuning — reduce collection frequency when freeze is not possible
4. Object reuse (connects to ALLOC patterns) — reduce GC pressure from temporary allocations

---

## COMM (Communication Overhead)
Host-side overhead from distributed communication in TP/PP inference paths.
```python
# Unnecessary barrier in hot loop
dist.barrier()                        # Stalls all ranks

# Repeated small NCCL launches
dist.all_reduce(tensor_a)            # Separate launch
dist.all_reduce(tensor_b)            # Separate launch

# Group creation in hot path
group = dist.new_group(ranks)        # ~ms collective operation!
```

**Detection heuristics:**
- In nsys: NCCL ratio (M5) > 0.20, or `ncclAllReduce` gaps between forward steps
- In line_profiler: `dist.all_reduce()`, `dist.barrier()`, `dist.new_group()` with high Per Hit
- In code: `dist.barrier()` inside the inference loop, or multiple sequential `dist.all_reduce()` calls

**Fix via:** Communication patterns in [communication-patterns.md](communication-patterns.md) (NCCL_BATCH, BARRIER_REMOVE, COMM_OVERLAP, DIST_WRAPPER, REDUCE_SCATTER, GROUP_CACHE).

---

## CUSTOM_OP (Python Tensor Op Chain)
A chain of Python tensor operations (view, slice, reinterpret, cast) that could be replaced by a single C++ custom op.

**Detection heuristics:**
- 3+ consecutive tensor manipulation lines (view, to, slice, contiguous) before a kernel launch
- Per Hit ~5-50us on each line in line_profiler
- High Hits count matching per-layer iteration count
- Total impact: `chain_overhead * calls_per_step` (e.g., 55us x 124 = 6.8ms)

See the CUSTOM_OP section in [patterns/gpu-graph.md](patterns/gpu-graph.md) for before/after code examples and fix strategies.

---

## GRAPH_BREAK (CUDA Graph Capture Break)
An operation that prevents CUDA graph capture of surrounding graph-safe code. The op itself may be fast, but it forces all nearby operations to run eagerly.

**Common triggers:** `.item()`, `torch.cumsum` with dynamic shapes, `index.Tensor` with dynamic indices, C extension calls opaque to CUDA graph runtime.

**Detection heuristics:**
- In nsys: `cudaLaunchKernel` (eager) appearing between `cudaGraphLaunch` sequences
- Use eager vs graph kernel classification (from analysis skill kernel-level drill-down) to identify affected kernels
- Check for `stop_partition` flags in framework graph partitioner logs

**Fix via:** GRAPH_EXPAND (partition poisoning) or GRAPH_SPLIT (split monolithic ops) patterns in [patterns/gpu-graph.md](patterns/gpu-graph.md).

---

## HOST_SYNC (Host-Device Synchronization in Forward Path)
A more specific variant of SYNC: `.item()`, `.cpu()`, or `cudaMemcpy(D2H)` calls specifically within the per-layer forward pass, where they execute N_layers times per step.

**Key distinction from SYNC:** SYNC in step-level code (1x per step) may be acceptable. HOST_SYNC in per-layer code (N_layers x per step) is almost always a bug.

**Detection heuristics:**
- `.item()`, `.cpu()`, `.numpy()` inside a function with Hits = N_layers in line_profiler
- In nsys: `cudaStreamSynchronize` appearing N_layers times per step
- Per Hit ~50-500us depending on GPU queue depth
- Total = per_hit x N_layers (e.g., 200us x 61 = 12.2ms)

**Fix via:** HOIST Variant B in [patterns/loop-iteration.md](patterns/loop-iteration.md) or SYNC Pattern 1 in [patterns/sync-alloc.md](patterns/sync-alloc.md).

---

## SERIALIZE (Serialization/Deserialization)
Converting between Python objects and wire formats in request processing or IPC paths.
```python
# JSON operations
data = json.loads(request_body)
response = json.dumps(result)

# Protobuf/pickle
obj = pickle.loads(data)
serialized = message.SerializeToString()
```

**Detection heuristics:**
- In line_profiler: Per Hit ~10-1000us on `pickle.dumps/loads`, `json.dumps/loads`, `SerializeToString()`
- CPU-bound gaps during request handling phases (no GPU or NCCL activity in nsys)
- Scope: typically per-step (request processing) or per-request (IPC)

**Fix via:** SERIALIZE patterns in [patterns/system.md](patterns/system.md):
1. Use `msgpack` instead of JSON/pickle for structured metadata (5-10x faster)
2. Use `SharedMemory` for same-node large buffer IPC (zero-copy)
3. Use raw buffer transfer for tensor IPC (skip pickle overhead)

---

## COMPUTE (Actual Computation)
Lines performing real work (arithmetic, comparisons, data structure operations) that cannot be eliminated — only made faster via algorithmic changes.

**Detection heuristics:**
- In line_profiler: moderate Per Hit (~1-10us) on lines doing actual logic (not framework calls)
- Not attributable to any other hotspot type (not sync, not allocation, not boundary crossing)
- Reducing iteration count or data size proportionally reduces time

**When to stop:** If all remaining hotspots are COMPUTE, host overhead optimization has reached its limit. Further improvement requires algorithmic changes or moving work off the hot path entirely.
