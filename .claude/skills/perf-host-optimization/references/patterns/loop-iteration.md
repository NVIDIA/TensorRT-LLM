# Loop & Iteration Patterns

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

## HOIST: Hoist Invariants Out of Hot Loops

A function called once per layer (N times/step) re-derives values that are identical across layers, or re-derives values already known at step level. This is one of the highest-impact patterns because per-layer overhead compounds (100us x 61 layers = 6ms).

**SAFETY RULE**: Caching/hoisting is ONLY safe inside code that runs every step (eager custom ops, non-graph-captured paths). Never cache inside CUDA-graph-captured code — the Python body only executes at capture time, so cached values become stale on replay.

**Detection**:
1. Profile the function — measure wall time per call
2. Inspect inputs: which change per layer? (usually only `layer_id`)
3. Everything else (metadata, pool views, tensor slices, stride calculations) is invariant
4. Check: does the metadata object already carry the needed values? If so, the per-layer code is a redundant re-derivation.

### Variant A: Cache on First Call

Use when the invariant is derived from GPU-side state that is not available at step-level preparation time.

```python
# BEFORE: 124 calls/step, 105-154us each = 16ms total
def transform_and_prepare_pool_view(self, ...):
    pool = metadata.kv_cache_manager.get_unique_primary_pool()  # repeated
    pool = pool.squeeze(2).view(-1, 1, head_dim)                # repeated
    stride_factor = num_layers * tokens_per_block                # repeated
    block_table = metadata.block_table[num_ctx:num_seqs]        # repeated

# AFTER: 1 compute + 123 cache hits = ~1ms total
def _ensure_pool_view_cached(self):
    if self._pool_cache_valid:
        return  # HIT — early return
    # MISS — compute and store
    self._cached_pool_view = pool.squeeze(2).view(...)
    self._cached_stride_factor = ...
    self._pool_cache_valid = True

# Invalidate at step boundaries
def prepare(self, ...):
    self._pool_cache_valid = False
```

### Variant B: Pass from Step Level

Use when the value is already known on CPU from batch metadata or request bookkeeping. This is the preferred variant when applicable — it avoids caching entirely by passing the value explicitly.

**Key principle**: When the re-derivation involves `.item()`, this becomes an instance of the SYNC pattern. `value.sum().item()` does TWO expensive things: (a) launches a reduction kernel, (b) synchronizes host-device. If you already know the answer from CPU-side bookkeeping, never ask the GPU.

```python
# BEFORE: Every layer, in C++ attention op:
num_contexts = 0
for idx in range(num_seqs):
    if request_types[idx] != RequestType.CONTEXT:
        break
    num_contexts += 1
num_ctx_tokens = host_context_lengths[:num_contexts].sum().item()
# This .item() is a HOST-DEVICE SYNC — 61 times per step!

# AFTER: Computed once in Python metadata, passed as explicit parameter
# In prepare() / build_metadata():
metadata.num_contexts = sum(1 for r in requests if r.is_context)
metadata.num_ctx_tokens = sum(r.context_length for r in context_requests)
# In per-layer forward:
attention(..., opt_num_contexts=metadata.num_contexts,
          opt_num_ctx_tokens=metadata.num_ctx_tokens)
```

### Instance-Level Caching (General)

For non-per-layer hotspots, standard memoization applies:

```python
# Instance-level cache by key
def get_strategy(self, request):
    cached = self._strategy_cache.get(request.slot_id)
    if cached is not None:
        return cached
    strategy = compute_expensive_strategy(request.params)
    self._strategy_cache[request.slot_id] = strategy
    return strategy
```

```python
# functools.lru_cache for pure functions with hashable arguments
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_mask(seq_len, dtype):
    return create_attention_mask(seq_len, dtype)
```
