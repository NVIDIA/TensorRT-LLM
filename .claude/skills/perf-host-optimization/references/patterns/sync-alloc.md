# Synchronization & Allocation Patterns

---

## SYNC: Remove/Batch Synchronization

**Cost model**: One `.item()` on a busy GPU pipeline = 50-500us (depends on queue depth). In single-stream workloads (common in inference), `.item()` triggers `cudaStreamSynchronize` which blocks the host until ALL previously queued work on that stream completes. With N layers x M syncs/layer, this compounds: 61 layers x 2 calls = 122 syncs x 200us avg = **24ms of pure stall time**.

**Pattern 1: Pre-compute on CPU (preferred)**
```python
# BEFORE: per-layer sync to get a value that's already known on CPU
def forward_per_layer(self, ...):
    num_ctx = host_context_lengths.slice(0, 0, num_contexts).sum().item()
    # This .item() is a HOST-DEVICE SYNC — 61 times per step!

# AFTER: Computed once in metadata builder, passed as parameter
# In prepare() or build_metadata():
metadata.num_ctx_tokens = sum(ctx_lengths)  # Pure CPU
# In per-layer forward:
def forward_per_layer(self, ..., num_ctx_tokens):
    # Use num_ctx_tokens directly — no sync needed
```

**Pattern 2: Batch scalar extraction**
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

**Pattern 3: Use GPU-side conditionals**
```python
# BEFORE: sync to check a condition on CPU
if tensor.sum().item() > threshold:  # SYNC!
    result = path_a(tensor)
else:
    result = path_b(tensor)

# AFTER: keep everything on GPU
result = torch.where(tensor.sum() > threshold, path_a(tensor), path_b(tensor))
```

**Pattern 4: Defer to step boundaries**

When `.item()` appears in per-layer code (N_layers x per step), hoist it to step-level preparation. See HOIST Variant B in [loop-iteration.md](loop-iteration.md) for the full pattern with concrete examples.

**Pattern 5: Use non-blocking transfers**
```python
# BEFORE: blocking transfer
cpu_tensor = gpu_tensor.cpu()     # Blocking

# AFTER: non-blocking with stream
cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
torch.cuda.current_stream().synchronize()  # Explicit sync point
```

**Fix strategy priority** (in order of preference):
1. Pre-compute on CPU — if the value comes from batch metadata, never ask the GPU
2. GPU-side conditionals — keep everything on GPU
3. Defer to step boundaries — if you MUST sync, do it once per step, not per layer
4. CUDA events for ordering — use `event.record()` + `stream.wait_event()` instead of full sync

---

## ALLOC: Pre-allocate Buffers

**Cost model**: PyTorch's caching allocator is fast (~1-3us) but NOT free. It walks a free list, checks size classes, and may trigger `cudaMalloc`. More importantly, allocation + Python setup around it (view, dtype cast) adds 10-30us. At per-layer frequency: 7 allocs x 61 layers = 427 allocs/step, even at 3us each = 1.3ms. With Python setup: 427 x 20us = 8.5ms.

**Detection**: Search for `torch.empty`, `torch.zeros`, `torch.ones` inside forward methods. Check: are the shapes known at init time? (max_batch, max_tokens, head_dim, etc.) Count: allocations per layer x layers per step = total allocations.

**Pattern 1: Pre-allocate at init, reuse via slicing**
```python
# BEFORE: 7 allocations per layer x 61 layers = 427 allocs/step
def forward_per_layer(self, ...):
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    fused_q = torch.empty([num_tokens, num_heads, dim], dtype=dtype, device=device)
    # ... 5 more allocations

# AFTER: Allocated once at init, reused via slicing
def __init__(self, ...):
    self._cu_q_seqlens = torch.empty(max_batch + 1, dtype=torch.int32, device='cuda')
    self._fused_q = torch.empty([max_tokens, num_heads, dim], dtype=dtype, device='cuda')

def forward_per_layer(self, ...):
    cu_q_seqlens = self._cu_q_seqlens[:num_seqs + 1]
    fused_q = self._fused_q[:num_tokens]
```

**Pattern 2: Reusable buffer pool**
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

**Pattern 3: Avoid unnecessary clones**
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

**Pattern 4: Contiguous check before operation**
```python
# BEFORE: always call contiguous
x = tensor.contiguous()           # May allocate

# AFTER: check first
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
# Or use ops that handle non-contiguous input
```
