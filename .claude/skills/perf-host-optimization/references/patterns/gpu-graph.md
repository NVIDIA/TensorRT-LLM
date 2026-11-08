# GPU Graph & Custom Op Patterns

---

## CUSTOM_OP: Replace Python Tensor Ops with C++ Custom Ops

**Pattern**: A chain of Python tensor operations (view, slice, reinterpret, cast) adds host overhead between kernel launches. Every Python tensor op, no matter how "free" computationally, costs 5-50us of host overhead (Python interpreter + torch dispatch). A 10-op Python chain = 50-500us of pure waste per call.

**Detection**:
1. In nsys, find gaps where the GPU is idle but no kernel is being prepared
2. In Python profiler, find hot lines — they will be `.view()`, `.to()`, `.contiguous()`, slicing, `torch.empty()`, etc.
3. Count the number of Python ops in the chain
4. Multiply: `chain_overhead * calls_per_step` = total impact

**Fix**:
1. Write a C++ custom op (CUDA kernel) that does the entire chain in one launch
2. Accept the original (pre-view) tensors as inputs with explicit size/stride params
3. Do all reinterpretation, casting, striding inside the kernel
4. Register via `TORCH_LIBRARY_FRAGMENT` + `TORCH_LIBRARY_IMPL`

**One kernel launch = ~3-5us total overhead**, vs 50-500us for a Python chain.

**Example 1: Index computation**
```python
# BEFORE: Triton kernel with 12-line Python preamble (view chains, stride checks)
# 154us host gap per call x 124 calls = 19ms
def convert_req_index_to_global(block_table, ...):
    block_table = block_table.view(-1)
    indices = block_table[offsets:offsets + count]
    # ... 10 more Python tensor ops
    triton_kernel[grid](indices, ...)

# AFTER: C++ CUDA kernel accepts raw tensors + strides as int params
# Kernel does block_table lookup, index math, output write — all on GPU
# 25us per call x 124 calls = 3ms
torch.ops.trtllm.convert_req_index_to_global(block_table, offsets, count, ...)
```

**Example 2: Cache scatter with reinterpretation**
```python
# BEFORE: 14 lines of Python view/slice/reinterpret before kernel launch
k_fp8_bytes = k_fp8.view(-1).view(torch.uint8).view(num_tokens, head_dim)
k_scale_flat = k_scale.view(-1)
if k_scale_flat.stride(-1) != 1:
    k_scale_flat = torch.as_strided(k_scale_flat.contiguous(), ...)
k_scale_bytes = k_scale_flat.view(torch.uint8).view(num_tokens, scale_size)
# ... more slicing
torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8_bytes, k_scale_bytes, ...)

# AFTER: Pass original tensors + num_tokens; C++ does reinterpret_cast internally
torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8, k_scale, k_cache,
                                             metadata.slot_mapping_fp8,
                                             metadata.slot_mapping_scale,
                                             num_tokens)
```

---

## GRAPH_SPLIT: Split Custom Ops for CUDA Graph Granularity

**Pattern**: A monolithic custom op contains both graph-capturable (pure tensor math) and non-capturable (metadata-dependent, dynamic) parts. The entire op is excluded from CUDA graph capture.

**Detection**:
1. Identify custom ops that are piecewise graph partition boundaries
2. Inspect the op body: which parts are pure tensor computation (GEMM, norm, RoPE)?
3. Which parts access batch metadata, do conditional branching, or modify state?
4. If >50% of the op's GPU time is pure tensor math, it is a split candidate

**Fix**:
1. Split into two custom ops:
   - **Op A (graph-capturable)**: Pure tensor math (projections, norms, quantization)
     - No batch metadata access
     - No tensor slicing by dynamic `num_tokens`
     - Fixed-shape inputs (padded to max)
   - **Op B (eager)**: Metadata-dependent work (cache update, indexing, attention dispatch)
     - Slices inputs to actual `num_tokens`
     - Accesses KV cache manager, batch structure
     - Contains conditional ctx/gen branching
2. Register Op A with whatever graph capture mechanism your framework uses
3. Return intermediates from Op A -> pass explicitly to Op B (no stashing on `self`)

**Safety rules**:
- Never stash tensors on `self` between graph-captured ops — CUDA graph replays the originally captured GPU pointers, but `self.tensor` may be reassigned to a different allocation between replays, causing the graph to read/write stale memory
- All `num_tokens` slicing must happen in the eager op — graph capture needs fixed shapes
- Disable dynamic control flow under torch compile

**Example**:
```
BEFORE: 1 custom op (mla_custom_op_inplace) — everything eager
  [projections + indexer + attention] = 43 kernels, all eager

AFTER: 2 custom ops
  Op 1 (mla_dsa_proj) — CUDA graph captured:
    [GEMM + LayerNorm + RoPE + FP8 quant] = 20 kernels, graph-captured
  Op 2 (mla_dsa_attn_inplace) — eager:
    [k_cache update + sparse indexer + attention dispatch] = 23 kernels, eager

Result: 20 kernels moved from eager -> graph = ~600us/layer saved
```

---

## GRAPH_EXPAND: Expand CUDA Graph Capture Coverage

**Pattern**: Code regions that COULD be graph-captured are running eagerly because (a) the graph partitioner gives up too early (partition poisoning), or (b) compilable and non-compilable ops are interleaved.

### Partition Poisoning

One non-capturable op (cumsum, index.Tensor) triggers `stop_partition = True`, killing graph capture for ALL subsequent operations in the same scope.

**Detection**:
1. Use eager vs graph kernel classification to find graph-capturable kernels running eagerly
2. Trace back: find the specific op that triggered `stop_partition`
3. Verify that ops after the trigger are structurally graph-safe

**Fix strategies**:
1. **Exclude-but-continue**: Make the trigger op its own excluded segment, let partitioning continue for ops before and after
2. **Hoist the trigger**: Move the non-capturable op outside the compiled region (pre-compute in eager Python, pass result as input tensor)
3. **Separate compilation**: Use a separate `@torch.compile` for the region after the trigger

**Example**:
```
BEFORE: cumsum at MTP line 987 triggers stop_partition
  -> 234 subsequent kernels (3 MTP decoder layers) ALL run eagerly
  -> 85ms of Python bubbles

AFTER (hoist): Pre-compute cumsum outside compiled region
  -> MTP layers get their own @torch.compile -> piecewise graph capture
  -> 85ms -> ~5ms (metadata updates only)
```

### Mixed Compile-Safety

A function mixes tensor math with `.item()`, Python loops, or C extensions.

**Fix strategies**:
1. **Separate concerns**: Split into `_compute_part()` (pure tensor math, `@torch.compile` safe) and `_metadata_part()` (stays eager)
2. **Hoist closures**: `@torch.compile` on a nested function creates a NEW compiled function each call. Hoist to a module method or standalone function.
3. **Replace dynamic with static**: Replace `.item()` with pre-computed values. Replace Python lists with pre-allocated tensors.

**Example**:
```python
# BEFORE: @torch.compile on nested function — recompiles every call
def forward(self):
    @torch.compile
    def prepare_position_ids():  # NEW compiled fn each call!
        ...

# AFTER: Hoisted to instance method — compiled once, reused
@torch.compile
def prepare_position_ids(self, ...):
    ...
```

**Safety check**: Before expanding graph capture, verify that ALL newly-captured ops are truly graph-safe:
- No `.item()` calls (host-device sync)
- No C extension calls (opaque to CUDA graph)
- No Python loops with dynamic trip counts
- No `torch.autograd` operations
