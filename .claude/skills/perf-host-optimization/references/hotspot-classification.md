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

**Mitigation patterns:**
```python
# Pattern 1: Feature-gate — skip boundary crossing entirely when feature unused
_uses_feature = self.model.uses_feature()  # One-time check
for req in requests:
    if _uses_feature:            # Only cross boundary when actually needed
        val = req.cpp_property

# Pattern 2: Shadow with Python attribute at construction time
class PyRequest:
    def __init__(self, cpp_req):
        self.cached_prop = cpp_req.cpp_property  # One crossing at init
        # ...later in hot loop:
        # use self.cached_prop instead of cpp_req.cpp_property
```

---

## ENUM_CONSTRUCT (Expensive Enum Construction)
Python `Enum(int_value)` is expensive (~1.7us) due to metaclass lookup. Adds up in hot loops when a single value dominates >95% of calls.
```python
# BEFORE: Enum construction every iteration
for req in requests:                          # N iterations
    mode = RequestMode(req.raw_mode_int)      # ~1.7us per Enum() call
    if mode == RequestMode.GENERATE:          # Almost always GENERATE
        handle_generate(req)

# AFTER: Compare raw int, construct only on rare path
_GENERATE_INT = RequestMode.GENERATE.value    # Cache the int value
for req in requests:
    if req.raw_mode_int == _GENERATE_INT:     # ~0.05us int compare
        handle_generate(req)
    else:
        mode = RequestMode(req.raw_mode_int)  # Rare path: construct enum
        handle_other(req, mode)
```

**Detection heuristics:**
- Per Hit ~1.5-2.0us on a line that constructs an Enum from an int
- `Hits` count matches loop iteration count
- One enum value dominates >95% of cases (check with a Counter or sampling)
- The enum class uses standard `enum.Enum` or `enum.IntEnum`

**Python version note:** `Enum(value)` cost varies by Python version. Python 3.11+ improved enum construction speed (~0.8-1.0us), but the raw-int-comparison pattern is still faster (~0.05us). The ~1.7us figure above is from Python 3.10. Always verify with line_profiler on the target runtime.

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

---

## GC (Garbage Collection)
Python GC pauses causing periodic latency spikes. **Low priority** — typically a
small fraction of total host time but can dominate tail latency (P99/P999).

**line_profiler cannot detect GC** — GC runs between profiled lines and is invisible
to line-level timing. Use nsys with NVTX markers instead.

Detection:
- Enable GC NVTX markers: `TLLM_PROFILE_RECORD_GC=1` (with nsys)
- In nsys timeline: look for periodic pauses that don't correlate with kernels or API calls
- Check `gc.get_stats()` for collection counts per generation

Mitigation (if GC is confirmed as a problem):
```python
import gc
# Disable automatic GC and collect manually at known safe points
gc.disable()
# ... hot loop ...
gc.collect()  # Explicit collection between batches
```

```python
# Reduce GC pressure by reusing objects
# BEFORE: allocating new list each iteration
for step in steps:
    temp = [process(x) for x in batch]  # New list every step

# AFTER: pre-allocate and reuse
temp = [None] * max_batch_size
for step in steps:
    for i, x in enumerate(batch):
        temp[i] = process(x)
```

---

## SERIALIZE (Serialization/Deserialization)
Converting between Python objects and wire formats.
```python
# JSON operations
data = json.loads(request_body)
response = json.dumps(result)

# Protobuf/pickle
obj = pickle.loads(data)
serialized = message.SerializeToString()
```
