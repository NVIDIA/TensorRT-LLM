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

# Optimization Patterns

Reference catalog of optimization patterns by hotspot type. Consult the relevant section after classifying a hotspot.

---

## REDUNDANT_ITER: Eliminate Redundant Iterations

**Pattern 1: Collect data during existing iteration**
```python
# BEFORE: Categorize in one loop, then build ID list in another
for request in scheduled_requests.generation_requests:
    if is_extend(request):
        extend_requests.append(request)
    else:
        generation_requests.append(request)

# Later: REDUNDANT second pass over the same data
self.previous_request_ids = [
    request.py_request_id
    for request in scheduled_requests.generation_requests
]

# AFTER: Collect IDs during the categorization loop
all_gen_request_ids = []
for request in scheduled_requests.generation_requests:
    all_gen_request_ids.append(request.py_request_id)  # Piggyback
    if is_extend(request):
        extend_requests.append(request)
    else:
        generation_requests.append(request)

# Later: Reuse the already-collected list
self.previous_request_ids = all_gen_request_ids
```

**Pattern 2: Use slicing instead of re-iteration**
```python
# BEFORE: Build full list, then build subset via separate iteration
all_ids = []
for req in all_requests:
    all_ids.append(req.id)
gen_ids = [req.id for req in generation_requests]  # Redundant!

# AFTER: Slice from the already-built list
all_ids = []
for req in all_requests:
    all_ids.append(req.id)
gen_ids = all_ids[num_context_requests:]  # O(n) slice, no attribute access
```

---

## DEAD_WORK: Skip Unnecessary Construction

**Pattern 1: Guard with cheap truthiness check**
```python
# BEFORE: Always construct object even when data is empty/None
for req in requests:
    params = MultimodalParams(multimodal_data=req.py_multimodal_data)
    params.strip_for_generation()
    if params.has_content():
        process(params)

# AFTER: Skip entirely when data is empty (common for text-only models)
for req in requests:
    if req.py_multimodal_data:  # Cheap truthiness check
        params = MultimodalParams(multimodal_data=req.py_multimodal_data)
        params.strip_for_generation()
        if params.has_content():
            process(params)
```

**Pattern 2: Model-level feature guard**
```python
# BEFORE: Check per-request feature that depends on model architecture
for req in requests:
    if self.model.supports_multimodal():  # Method call per request
        handle_multimodal(req)

# AFTER: Cache at init time or before loop
_supports_multimodal = self.model.supports_multimodal()
for req in requests:
    if _supports_multimodal:
        handle_multimodal(req)
```

---

## FUNCALL: Cache Invariant Method Results

**Pattern 1: Hoist invariant method calls out of loops**
```python
# BEFORE: Method call inside hot loop (result never changes)
for request in generation_requests:  # 279K iterations
    if self.mapping.has_cp_helix():  # Same result every time!
        handle_helix(request)

# AFTER: Cache before loop
_has_cp_helix = self.mapping.has_cp_helix()
for request in generation_requests:
    if _has_cp_helix:
        handle_helix(request)
```

**Pattern 2: Eliminate redundant assignments inside loops**
```python
# BEFORE: Assign constant inside loop
for request in requests:
    for beam in range(beam_width):
        first_beam = 0                   # Assigned every iteration!
        if beam == first_beam:
            process_first_beam(request)

# AFTER: Compare directly with literal
for request in requests:
    for beam in range(beam_width):
        if beam == 0:                    # No assignment needed
            process_first_beam(request)
```

**Pattern 3: Inline hot functions**
```python
# BEFORE: function call in tight loop
def is_valid(req):
    return req.status == Status.ACTIVE and req.tokens > 0

for req in requests:
    if is_valid(req):             # Function call overhead
        process(req)

# AFTER: inline the logic
for req in requests:
    if req.status == Status.ACTIVE and req.tokens > 0:
        process(req)
```

---

## SYNC: Remove/Batch Synchronization

**Pattern 1: Batch scalar extraction**
```python
# BEFORE: sync every iteration (N syncs)
for req in requests:
    val = tensor[req.idx].item()  # SYNC per iteration!
    process(val)

# AFTER: single sync, then Python iteration
vals = tensor.cpu().numpy()       # One sync
for req in requests:
    val = vals[req.idx]           # Pure Python indexing
    process(val)
```

**Pattern 2: Defer synchronization**
```python
# BEFORE: sync immediately after kernel
output = model(input)
result = output.item()            # SYNC - blocks until kernel completes

# AFTER: overlap compute with other work
output = model(input)             # Async launch
do_other_work()                   # CPU work while GPU runs
result = output.item()            # Sync only when needed
```

**Pattern 3: Use non-blocking transfers**
```python
# BEFORE: blocking transfer
cpu_tensor = gpu_tensor.cpu()     # Blocking

# AFTER: non-blocking with stream
cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
torch.cuda.current_stream().synchronize()  # Explicit sync point
```

---

## ALLOC: Pre-allocate Buffers

**Pattern 1: Reusable buffer pool**
```python
# BEFORE: allocate every iteration
for batch in batches:
    temp = torch.zeros(batch_size, hidden_dim, device='cuda')
    process(batch, temp)

# AFTER: pre-allocated buffer with slice reuse
max_batch_size = max(len(b) for b in batches)
temp_buffer = torch.zeros(max_batch_size, hidden_dim, device='cuda')
for batch in batches:
    temp = temp_buffer[:len(batch)]
    temp.zero_()                  # In-place reset (no allocation)
    process(batch, temp)
```

**Pattern 2: Avoid unnecessary clones**
```python
# BEFORE: defensive cloning
def process(tensor):
    working = tensor.clone()      # Allocation!
    working += 1
    return working

# AFTER: in-place or documented mutation
def process(tensor):
    tensor += 1                   # In-place (document side effect)
    return tensor

# OR: caller provides output buffer
def process(tensor, out):
    torch.add(tensor, 1, out=out)
    return out
```

**Pattern 3: Contiguous check before operation**
```python
# BEFORE: always call contiguous
x = tensor.contiguous()           # May allocate

# AFTER: check first
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
# Or use ops that handle non-contiguous input
```

---

## PYLOOP: Vectorize Operations

**Pattern 1: Replace Python loop with tensor ops**
```python
# BEFORE: Python iteration (slow)
results = []
for x in items:
    results.append(x * 2 + 1)
results = torch.stack(results)

# AFTER: vectorized (fast)
items_tensor = torch.stack(items)
results = items_tensor * 2 + 1
```

**Pattern 2: Use torch.where instead of conditional loop**
```python
# BEFORE: conditional per element
for i, val in enumerate(tensor):
    if val > threshold:
        output[i] = val * scale
    else:
        output[i] = default

# AFTER: vectorized conditional
output = torch.where(tensor > threshold, tensor * scale, default)
```

**Pattern 3: Batch request processing**
```python
# BEFORE: process one request at a time
for req in requests:
    req.output = self.sampler.sample(req.logits)

# AFTER: batch all requests
all_logits = torch.stack([req.logits for req in requests])
all_outputs = self.sampler.sample_batch(all_logits)
for req, out in zip(requests, all_outputs):
    req.output = out
```

---

## CONTAINER: Optimize Data Structure Access

**Pattern 1: Cache dictionary lookups**
```python
# BEFORE: repeated dict access
for req in requests:
    strategy = self.strategies[req.strategy_type]  # Hash + lookup each time
    strategy.process(req)

# AFTER: group by key first
from collections import defaultdict
by_strategy = defaultdict(list)
for req in requests:
    by_strategy[req.strategy_type].append(req)
for strategy_type, reqs in by_strategy.items():
    strategy = self.strategies[strategy_type]      # One lookup per group
    for req in reqs:
        strategy.process(req)
```

**Pattern 2: Use local variable for repeated attribute access**
```python
# BEFORE: repeated attribute lookup
for req in requests:
    self.config.param1             # __getattribute__ each time
    self.config.param2

# AFTER: cache in local
config = self.config               # One lookup
param1, param2 = config.param1, config.param2
for req in requests:
    use(param1, param2)
```

**Pattern 3: Use slots for frequently instantiated classes**
```python
# BEFORE: regular class (dict-based attributes)
class Request:
    def __init__(self, id, data):
        self.id = id
        self.data = data

# AFTER: slots (faster attribute access, less memory)
class Request:
    __slots__ = ['id', 'data']
    def __init__(self, id, data):
        self.id = id
        self.data = data
```

---

## BOUNDARY: Reduce Python-C++ Crossings

> **Scoping note**: BOUNDARY optimizations (feature-gating, shadowing) only help when the boundary crossing is inside a **hot loop** (high Hits count in line_profiler). A single boundary crossing outside a loop (~250ns) is negligible. Focus on crossings where `Hits * Per Hit` exceeds ~100us total.

**Pattern 1: Feature-gate to skip entirely when feature unused**
```python
# BEFORE: Always access C++ property even when feature is disabled
for req in requests:
    multimodal_data = req.py_multimodal_data  # nanobind crossing ~250ns
    if multimodal_data:
        process_multimodal(multimodal_data)

# AFTER: Gate on model-level feature flag (checked once)
_has_multimodal = self.model_config.has_multimodal()  # One-time check
for req in requests:
    if _has_multimodal:
        multimodal_data = req.py_multimodal_data  # Only cross when needed
        if multimodal_data:
            process_multimodal(multimodal_data)
```

**Pattern 2: Shadow with Python attribute at construction time**
```python
# BEFORE: Access C++ property every iteration
for req in requests:
    req_id = req.py_request_id      # nanobind crossing ~250ns
    req_type = req.request_type     # nanobind crossing ~250ns
    process(req_id, req_type)

# AFTER: Shadow at request creation time (once per request lifetime)
class PyRequest:
    """Python wrapper that caches frequently-accessed C++ properties."""
    __slots__ = ['_cpp_req', 'request_id', 'request_type']

    def __init__(self, cpp_req):
        self._cpp_req = cpp_req
        self.request_id = cpp_req.py_request_id   # One crossing at init
        self.request_type = cpp_req.request_type   # One crossing at init

# In hot loop: pure Python attribute access (~0.05us)
for req in py_requests:
    process(req.request_id, req.request_type)
```

---

## ENUM_CONSTRUCT: Defer Expensive Enum Construction

**Pattern 1: Raw int comparison, construct only on rare path**
```python
# BEFORE: Construct Enum every iteration (~1.7us each)
for req in requests:
    state = RequestState(req.raw_state)       # ~1.7us per call
    if state == RequestState.GENERATION:
        handle_generation(req)
    elif state == RequestState.CONTEXT:
        handle_context(req)

# AFTER: Compare raw ints for dominant case, construct only on rare path
_GENERATION_INT = RequestState.GENERATION.value
_CONTEXT_INT = RequestState.CONTEXT.value
for req in requests:
    raw = req.raw_state
    if raw == _GENERATION_INT:                # ~0.05us int compare
        handle_generation(req)
    elif raw == _CONTEXT_INT:                 # ~0.05us int compare
        handle_context(req)
    else:
        state = RequestState(raw)             # Rare: construct for unknown
        handle_other(req, state)
```

**Pattern 2: Batch-collect raw ints, then batch-convert non-default values**
```python
# BEFORE: Construct enum for every request to categorize
categories = {}
for req in requests:
    mode = RequestMode(req.raw_mode)          # ~1.7us * N
    categories.setdefault(mode, []).append(req)

# AFTER: Categorize by raw int, only construct enums for rare categories
_DEFAULT_MODE_INT = RequestMode.GENERATE.value
default_reqs = []
other_by_int = {}
for req in requests:
    raw = req.raw_mode
    if raw == _DEFAULT_MODE_INT:
        default_reqs.append(req)              # No enum construction
    else:
        other_by_int.setdefault(raw, []).append(req)

# Construct enums only for the non-default categories (typically 0-2 entries)
categories = {RequestMode.GENERATE: default_reqs}
for raw_int, reqs in other_by_int.items():
    categories[RequestMode(raw_int)] = reqs   # Rare: few constructions
```

---

## CACHE: Memoization and Caching

**Pattern 1: Instance-level caching**
```python
# BEFORE: recompute every call
def get_strategy(self, request):
    return compute_expensive_strategy(request.params)  # Expensive!

# AFTER: cache by slot/key
def get_strategy(self, request):
    cached = self._strategy_cache.get(request.slot_id)
    if cached is not None:
        return cached
    strategy = compute_expensive_strategy(request.params)
    self._strategy_cache[request.slot_id] = strategy
    return strategy
```

**Pattern 2: Use functools.lru_cache for pure functions**
```python
from functools import lru_cache

# BEFORE: repeated computation
def compute_mask(seq_len, dtype):
    return create_attention_mask(seq_len, dtype)

# AFTER: cached (for hashable arguments)
@lru_cache(maxsize=128)
def compute_mask(seq_len, dtype):
    return create_attention_mask(seq_len, dtype)
```

---

## Compound Patterns

Some hotspots combine multiple types. Address them in the listed order (outer → inner).

### FUNCALL + BOUNDARY
A Python function call wraps a C++ boundary crossing. Eliminate the function call first (inline), then gate or shadow the boundary crossing.
```python
# BEFORE: function call + boundary crossing per iteration
def get_request_id(req):           # FUNCALL: ~100ns call overhead
    return req.py_request_id       # BOUNDARY: ~250ns nanobind crossing

for req in requests:
    rid = get_request_id(req)      # ~350ns total per call

# AFTER: inline + shadow
for req in requests:
    rid = req.request_id           # Shadowed Python attr: ~50ns
```

### DEAD_WORK + REDUNDANT_ITER
Unnecessary object construction combined with a redundant pass. Eliminate the dead work first (guard with cheap check), then merge the surviving iteration into an existing loop.
```python
# BEFORE: construct always + iterate twice
for req in requests:
    params = Params(req.data)      # DEAD_WORK: always constructed
    if params.has_content():
        use(params)
ids = [r.id for r in requests]     # REDUNDANT_ITER: second pass

# AFTER: guard + merge
ids = []
for req in requests:
    ids.append(req.id)             # Merged into single pass
    if req.data:                   # Guard: skip when empty
        params = Params(req.data)
        if params.has_content():
            use(params)
```

### PYLOOP + BOUNDARY
A Python loop body whose dominant cost is boundary crossings (C++ property access per iteration). Gate or shadow the boundary crossings first, then consider vectorizing the remaining loop body.
```python
# BEFORE: loop with boundary crossings
for req in requests:                          # PYLOOP: N iterations
    rid = req.py_request_id                   # BOUNDARY: ~250ns
    rtype = req.request_type                  # BOUNDARY: ~250ns
    process(rid, rtype)                       # ~500ns/iter from crossings alone

# AFTER: shadow at creation + pure Python loop
# (At request creation time, cache properties on a Python wrapper)
for req in py_requests:
    process(req.request_id, req.request_type) # ~50ns/iter (Python attrs)
```

### ENUM_CONSTRUCT + PYLOOP
Expensive enum construction inside a Python loop. Replace enum construction with raw int comparison first, then consider vectorizing the remaining loop body if possible.
```python
# BEFORE: enum + loop
for req in requests:                         # PYLOOP
    mode = RequestMode(req.raw_mode)         # ENUM_CONSTRUCT: ~1.7us
    if mode == RequestMode.GENERATE:
        handle(req)

# AFTER: raw int compare (eliminates enum cost within loop)
_GEN = RequestMode.GENERATE.value
for req in requests:
    if req.raw_mode == _GEN:                 # ~0.05us
        handle(req)
```

---

## CPython Pitfalls

Micro-optimizations in CPython can have counterintuitive results. **Always validate with re-profiling.**

### Pitfall 1: Ternary expressions can be slower than attribute access

```python
# SEEMS FASTER but is actually SLOWER:
beam_width = request.sampling_config.beam_width if _use_beam_search else 1
# CPython ternary bytecode: LOAD, POP_JUMP, LOAD_CONST, JUMP — ~14us/hit

# ACTUALLY FASTER (simple attribute chain):
beam_width = request.sampling_config.beam_width
# CPython attribute chain: LOAD_ATTR, LOAD_ATTR — ~5us/hit
```

**Why:** The conditional expression generates branch bytecode that CPython evaluates less efficiently than a straightforward attribute access chain, even when the condition is a local variable. The branch prediction overhead in the interpreter loop outweighs the saved attribute access.

### Pitfall 2: Local variable caching only helps for method calls, not simple attributes

```python
# HELPS (~1.5s savings for 279K iterations):
_has_cp_helix = self.mapping.has_cp_helix()  # Method call -> cache result
for req in requests:
    if _has_cp_helix: ...

# MARGINAL (< 0.1s savings):
_is_dummy = self.is_draft_model  # Simple attribute -> already fast
for req in requests:
    if _is_dummy: ...
```

**Why:** `self.attr` is a single `LOAD_ATTR` instruction (~5us). Caching it as a local saves ~1-2us per access. Only worthwhile for **method calls** (which have call frame overhead) or **chained attributes** (`self.a.b.c`).

### Pitfall 3: Adding conditionals to "skip" loop iterations can cost more than the iterations

```python
# SLOWER: Added check costs more than the skipped work
for req in requests:
    if req.needs_processing:  # Extra check on every iteration
        do_work(req)          # Work is cheap

# FASTER: Just do the work unconditionally
for req in requests:
    do_work(req)
```

**Rule of thumb:** Only add a guard condition if the skipped work costs >10us per iteration AND the condition itself is <2us (simple attribute/truthiness check).

### Pitfall 4: `isinstance()` in hot loops (~200-300ns MRO traversal)

```python
# SLOWER: isinstance() walks the MRO chain each call
for req in requests:
    if isinstance(req, GenerationRequest):  # ~200-300ns MRO traversal
        handle_generation(req)

# FASTER: use a type tag or duck-typing attribute
for req in requests:
    if req.is_generation:                   # ~50ns attribute check
        handle_generation(req)
```

**Why:** `isinstance()` must traverse the Method Resolution Order (MRO) and check each base class. For deep inheritance hierarchies or ABCs, this can be 200-300ns per call. In a hot loop with 1000+ iterations, this adds up. Prefer a boolean attribute or type tag set at construction time.

### Pitfall 5: `in` on list vs set (O(n) vs O(1))

```python
# SLOWER: O(n) membership test on list
blocked_ids = [1, 2, 3, ..., 100]          # list
for req in requests:
    if req.id in blocked_ids:               # O(n) linear scan

# FASTER: O(1) membership test on set
blocked_ids = {1, 2, 3, ..., 100}          # set
for req in requests:
    if req.id in blocked_ids:               # O(1) hash lookup
```

**Why:** `in` on a `list` is O(n) — it performs a linear scan. For lists with >10 elements in a hot loop, convert to a `set` or `frozenset` before the loop. The set construction cost is amortized after ~2-3 loop iterations for a 100-element collection.

### Pitfall 6: Generator expression vs list comprehension in `sum`/`min`/`max`

```python
# SLOWER: list comprehension materializes the full list
total = sum([req.token_count for req in requests])  # Allocates list

# FASTER: generator expression (no intermediate allocation)
total = sum(req.token_count for req in requests)     # Lazy evaluation
```

**Why:** A list comprehension `[expr for x in iter]` allocates a full list in memory before passing it to `sum()`. A generator expression `(expr for x in iter)` yields values lazily with no intermediate allocation. For large collections, this saves both memory and time. The difference is most pronounced with >1000 elements.

### Pitfall 7: String formatting in hot paths (use tuple keys instead)

```python
# SLOWER: f-string formatting every iteration
cache = {}
for req in requests:
    key = f"{req.model_id}_{req.batch_size}_{req.dtype}"  # ~500ns string alloc
    if key not in cache:
        cache[key] = compute(req)

# FASTER: tuple key (no string allocation, faster hashing)
for req in requests:
    key = (req.model_id, req.batch_size, req.dtype)       # ~50ns tuple creation
    if key not in cache:
        cache[key] = compute(req)
```

**Why:** f-string formatting allocates a new string object each time, involving concatenation and `__format__` calls. Tuples of ints/strings are cheaper to create and hash faster than equivalent string keys. Use tuple keys for dictionary lookups in hot loops.
