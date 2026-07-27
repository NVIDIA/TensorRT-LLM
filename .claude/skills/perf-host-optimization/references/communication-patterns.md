# Communication Overhead Patterns

Optimization patterns for host-side communication overhead in distributed TRT-LLM inference (TP/PP). Use when nsys analysis shows high NCCL ratio (M5 > 0.20) or when inter-step gaps correlate with distributed coordination.

---

## NCCL_BATCH: Group Operations to Reduce Launch Overhead

Each NCCL call has ~5-10us host overhead for kernel launch setup.

**Detection:**
- In nsys: many small `ncclAllReduce`/`ncclAllGather` calls with gaps between them
- In line_profiler: multiple `dist.all_reduce()` calls in sequence within a single function
- Per-step impact: `num_separate_calls * 5-10us`

**Pattern 1: Use coalescing manager**
```python
# BEFORE: N separate NCCL launches (N * 5-10us overhead)
dist.all_reduce(grad_weight)     # Launch 1
dist.all_reduce(grad_bias)       # Launch 2
dist.all_reduce(grad_norm)       # Launch 3

# AFTER: single batched launch via coalescing (PyTorch 2.0+)
with dist._coalescing_manager():
    dist.all_reduce(grad_weight)
    dist.all_reduce(grad_bias)
    dist.all_reduce(grad_norm)
# Fused into single NCCL group call
```

**Pattern 2: Flatten and fuse tensors**
```python
# BEFORE: N all-reduces for N small tensors
for tensor in tensor_list:
    dist.all_reduce(tensor)       # N launches

# AFTER: flatten, single all-reduce, unflatten
flat = torch._utils._flatten_dense_tensors(tensor_list)
dist.all_reduce(flat)             # 1 launch
results = torch._utils._unflatten_dense_tensors(flat, tensor_list)
```

**Impact:** Reduces launch overhead from O(n) to O(1) for n operations.

---

## BARRIER_REMOVE: Eliminate Unnecessary dist.barrier()

`dist.barrier()` synchronizes ALL ranks. The slowest rank determines completion time.

**Detection:**
- In nsys: `ncclAllReduce` (barrier implementation) appearing between forward steps with no data payload
- In code: `dist.barrier()` calls inside the inference loop
- Impact: ~10-100us per barrier + straggler wait time (unbounded)

**Pattern: Remove barriers where collectives provide implicit sync**
```python
# BEFORE: explicit barrier in hot loop
for step in range(steps):
    output = model(input)
    dist.barrier()                # Unnecessary! All ranks wait for slowest
    dist.all_reduce(output)       # Already synchronizes ranks

# AFTER: collectives are already synchronizing
for step in range(steps):
    output = model(input)
    dist.all_reduce(output)       # Implicit sync between ranks
```

**When barrier IS needed:** Only when ranks diverge in control flow (e.g., rank 0 writes a file, others read it). Never inside the forward step loop.

---

## COMM_OVERLAP: Async All-Reduce with Compute Overlap

Launch communication early, compute independent work, wait only when the result is needed.

**Detection:**
- In nsys: GPU idle during all-reduce (no compute kernels overlapping NCCL kernels)
- In code: blocking `dist.all_reduce()` followed by independent compute
- Scope: per-layer (TP all-reduce after each attention/FFN block)

**Pattern: Overlap all-reduce with next layer's independent compute**
```python
# BEFORE: blocking all-reduce stalls GPU between layers
for i, layer in enumerate(model.layers):
    output = layer(input)
    dist.all_reduce(output)         # GPU idle during communication
    input = output

# AFTER: async all-reduce overlaps with next layer's compute
prev_work = None
for i, layer in enumerate(model.layers):
    output = layer(input)
    work = dist.all_reduce(output, async_op=True)  # Non-blocking
    if prev_work is not None:
        prev_work.wait()            # Wait for PREVIOUS layer's all-reduce
    prev_work = work
    input = output
if prev_work is not None:
    prev_work.wait()
```

**Impact:** Hides communication latency behind compute. Effective when compute time >= communication time per layer.

---

## DIST_WRAPPER: Amortize torch.distributed Python Overhead

Each `dist.*` call adds ~2-5us of Python dispatch overhead (argument validation, ProcessGroup lookup, tensor checks).

**Detection:**
- In line_profiler: Per Hit ~2-5us on `dist.all_reduce` / `dist.send` lines
- Many such calls per step (e.g., separate all-reduce per tensor)
- Total: `num_calls * 2-5us` Python-side overhead

**Pattern: Batch P2P operations**
```python
# BEFORE: separate send/recv calls (N * 2-5us Python overhead)
for i in range(n_peers):
    dist.send(tensor_list[i], dst=peers[i])     # Python dispatch each call

# AFTER: batched P2P (single Python dispatch)
ops = []
for i in range(n_peers):
    ops.append(dist.P2POp(dist.isend, tensor_list[i], peers[i]))
reqs = dist.batch_isend_irecv(ops)              # Single dispatch for all
for req in reqs:
    req.wait()
```

**Impact:** Reduces Python dispatch overhead from O(n) to O(1) for n P2P operations.

---

## REDUCE_SCATTER: Use Fused Collective

`all_reduce` followed by scatter is a common pattern in TP. `reduce_scatter` does both in one operation with half the communication volume.

**Detection:**
- In code: `dist.all_reduce(full_tensor)` followed by `chunk = full_tensor.chunk(world_size)[rank]`
- In nsys: all-reduce on a tensor that is immediately sliced by rank

**Pattern:**
```python
# BEFORE: all_reduce + manual scatter (2x communication volume)
dist.all_reduce(full_tensor)
my_chunk = full_tensor.chunk(world_size)[rank]

# AFTER: fused reduce_scatter (1x communication volume)
output = torch.empty(chunk_size, device='cuda')
dist.reduce_scatter(output, list(full_tensor.chunk(world_size)))
```

**Impact:** ~50% less communication volume. Single NCCL call instead of two operations.

---

## GROUP_CACHE: Cache Process Groups

`dist.new_group()` is an expensive collective operation (~ms). Never create groups in the hot path.

**Detection:**
- In line_profiler: Per Hit ~1-10ms on `dist.new_group()` lines
- `Hits` count > 1 for the same set of ranks

**Pattern:**
```python
# BEFORE: create group every call
def all_reduce_subset(tensor, ranks):
    group = dist.new_group(ranks)         # ~ms collective operation!
    dist.all_reduce(tensor, group=group)

# AFTER: cache groups by rank tuple
_group_cache = {}
def all_reduce_subset(tensor, ranks):
    key = tuple(ranks)
    if key not in _group_cache:
        _group_cache[key] = dist.new_group(ranks)
    dist.all_reduce(tensor, group=_group_cache[key])
```

**Impact:** Eliminates ~ms collective overhead per call after the first invocation for each rank set.

---

## Risk Classification

| Pattern | Risk | Validation |
|---------|------|------------|
| BARRIER_REMOVE | Zero | Verify collectives already provide needed sync |
| GROUP_CACHE | Zero | Re-profile only |
| NCCL_BATCH | Low | Compare outputs (tensor fusion may affect precision) |
| DIST_WRAPPER | Low | Compare outputs |
| REDUCE_SCATTER | Low | Compare outputs (different reduction order) |
| COMM_OVERLAP | Medium | Verify data dependencies between layers |
