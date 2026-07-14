# Optimization Strategy

Strategic guidance for ordering, measuring, and scoping host overhead optimizations. Consult before starting the optimization loop to avoid common methodology pitfalls.

---

## Zero-Risk-First Ordering

**Principle**: Always implement zero-risk optimizations before medium/high-risk ones.

### Risk Taxonomy

| Risk Level | Description | Validation | Examples |
|------------|-------------|------------|----------|
| **Zero** | Changes WHEN computation happens, not WHAT. Output is bit-identical. | Re-profile only | Caching read-only values, pre-allocating buffers, hoisting loop invariants |
| **Low** | Same math in C++ instead of Python. Output should be bit-identical. | Compare outputs | Replacing Python tensor ops with C++ custom ops |
| **Medium** | Same computation but different execution order or dispatch mechanism. | Warmup success + output comparison | Changing graph partition boundaries, restructuring custom op splits |
| **High** | Different computation that should produce equivalent results. | Numerical tolerance checks | Fusing operations, combining attention paths, algorithm changes |

### Why This Ordering Matters

1. **Free gains first**: Zero-risk optimizations provide guaranteed speedups with no correctness risk
2. **Cleaner profiling**: Each subsequent round profiles a cleaner baseline with fewer confounding variables
3. **Methodology confidence**: Consistent measurements from zero-risk changes validate your profiling setup before you make riskier changes
4. **Rollback simplicity**: If a medium/high-risk change introduces subtle correctness issues, you want the zero-risk gains already locked in

### Mapping to Optimization Patterns

| Risk Level | Patterns |
|------------|----------|
| Zero | HOIST, ALLOC (pre-allocate), FUNCALL (cache invariants), REDUNDANT_ITER, DEAD_WORK, CONTAINER, ENUM_CONSTRUCT, GC_MGMT (freeze/disable), BARRIER_REMOVE, GROUP_CACHE |
| Low | CUSTOM_OP, BOUNDARY (shadow with Python attr), SYNC (batch/defer), NCCL_BATCH, DIST_WRAPPER, REDUCE_SCATTER, SERIALIZE (msgpack/raw buffer) |
| Medium | GRAPH_SPLIT, GRAPH_EXPAND, PYLOOP (vectorize with same numerics), COMM_OVERLAP, GIL_CONTENTION (fine-grained locking) |
| High | Algorithm fusion, PYLOOP (vectorize with different numerics) |

---

## Metric Traps

Five common measurement traps that lead to wasted optimization effort.

### Trap 1: Kernel Duration vs Launch Gap

A kernel that takes 1us on GPU but has 500us of host overhead before it contributes 501us to end-to-end latency. **Don't optimize the kernel — optimize the gap.**

Use inter-kernel gap analysis (from the analysis skill's kernel-level drill-down) to measure gaps, not kernel durations.

### Trap 2: Wrong Statistic for the Question

| Question | Right Statistic | Wrong Statistic |
|----------|----------------|-----------------|
| Typical per-step behavior | Median | Mean (skewed by outliers) |
| Tail latency / SLA compliance | P99 / P999 | Mean or median |
| Aggregate impact on throughput | Total (sum) | Per-call average |

Don't let a few large outliers pull your attention away from 10,000 small gaps that add up to more total impact.

### Trap 3: Per-Call vs Total Impact

A 10us optimization x 12,000 calls/step = **120ms total savings**.
A 5ms optimization x 3 calls/step = **15ms total savings**.

**Total impact drives priority**, not per-call cost. Always multiply `per_call_savings * calls_per_step` to compute the actual throughput improvement.

### Trap 4: Graph Kernel Count != Performance

More CUDA graph segments does not mean better performance. Each segment's replay dispatch costs 50-100us. Fewer, larger segments are better than many small ones.

**Example**: Splitting one monolithic eager region into 10 graph segments adds 10 x 75us = 750us of dispatch overhead. If the segments only save 500us of eager launch overhead, you net **-250us** (a regression).

### Trap 5: Allocation Count, Not Allocation Size

A 1KB allocation and a 1GB allocation cost nearly the same host overhead (~1-3us for PyTorch caching allocator, plus 10-30us for Python setup). Reducing the **number** of allocations matters more than reducing sizes.

**Prioritize by**: `num_allocations_per_step`, not `total_bytes_allocated`.

---

## Three Scopes of Host Overhead

Every host overhead source falls into one of three scopes. The scope determines the fix strategy and the total impact.

| Scope | Frequency | Fix Strategy | Examples |
|-------|-----------|--------------|----------|
| **Per-step** | 1x per forward step | Pre-compute in step-level preparation (`prepare()`, `build_metadata()`) | Batch metadata, cumsum, request type counting |
| **Per-layer** | Nx per step (N = num_layers) | Cache across layers, eliminate redundancy (HOIST pattern) | Pool view computation, plan() params, rope table lookup |
| **Per-kernel** | NxK per step (K = kernels/layer) | Replace Python with C++ (CUSTOM_OP), fuse kernels | View chains, tensor slicing, dtype casts |

### Impact Calculation

The total impact of an overhead source depends on its scope:

```
Per-kernel:  10us overhead x 43 kernels x 61 layers = 26ms/step
Per-layer:  100us overhead x 61 layers              =  6ms/step
Per-step:     3ms overhead x 1                       =  3ms/step
```

**Priority**: Per-kernel > per-layer > per-step (by total impact, not per-call cost).

### Scope Identification

When profiling reveals a hotspot, determine its scope before choosing a fix:

1. **Check Hits count** in line_profiler: Hits = 1 -> per-step, Hits = N_layers -> per-layer, Hits >> N_layers -> per-kernel
2. **Check call site**: Is the function called from `forward()` (per-step), from a per-layer loop, or from within a kernel dispatch chain?
3. **Multiply**: `per_call_time * hits_per_step` = total impact. This determines whether the optimization is worth the effort.

---

## Pattern Selection Guide

Use this decision tree to map profiler output to the correct hotspot type and fix pattern.

### From line_profiler Signatures

| Per Hit | Hits Pattern | Line Content | Hotspot Type | Fix Pattern |
|---------|-------------|--------------|--------------|-------------|
| ~50-500us | = N_layers | `.item()`, `.cpu()`, `.numpy()` | HOST_SYNC | HOIST Variant B (pass from step level) or SYNC Pattern 1 (pre-compute on CPU) |
| ~50-500us | = 1 (per-step) | `.item()`, `.cpu()`, `synchronize()` | SYNC | SYNC Pattern 2-5 (batch, GPU-side conditional, defer, non-blocking) |
| ~5-50us | >> N_layers | `.view()`, `.to()`, `.contiguous()`, slicing | CUSTOM_OP | Replace Python tensor chain with C++ custom op |
| ~1.5-2.0us | = loop count | `Enum(int_value)` construction | ENUM_CONSTRUCT | Raw int comparison, construct only on rare path |
| ~1-10us | = loop count | `torch.empty(...)`, `.clone()`, `.zeros()` | ALLOC | Pre-allocate at init, reuse via slicing |
| ~0.2-0.5us | = loop count | `req.cpp_property` (C++ binding attribute) | BOUNDARY | Feature-gate or shadow with Python attribute |
| ~0.05-0.1us | = loop count | `self.method()` returning constant result | FUNCALL | Cache result before loop |
| ~100us-10ms | any | `lock.acquire()`, `queue.get()`, `event.wait()` | GIL | Fine-grained locking, batch under lock, thread-local |
| ~10-1000us | per-step | `pickle.dumps/loads`, `json.dumps/loads` | SERIALIZE | msgpack, SharedMemory, raw buffer |
| N/A | N/A | Two loops over same collection | REDUNDANT_ITER | Merge into single pass |
| N/A | N/A | Object always constructed, rarely used | DEAD_WORK | Guard with cheap truthiness check |

**Key discriminators:**
- Per Hit ~0.2-0.5us on attribute access -> BOUNDARY (pure Python attrs are ~0.05us)
- Per Hit ~1.5-2.0us on constructor call -> ENUM_CONSTRUCT (normal constructors are ~0.1-0.5us)
- Hits = N_layers on `.item()` -> HOST_SYNC (not just SYNC -- per-layer scope is critical)

### From nsys Analysis Handoff

When the `perf-host-analysis` skill provides root cause findings (including kernel-level drill-down), map them to optimization patterns:

| Analysis Finding | Hotspot Type | Fix Pattern | Reference |
|-----------------|--------------|-------------|-----------|
| GPU idle ratio > 0.30, host prep confirmed (M3b+M3c) | Multiple | Start with top NVTX regression from Root Cause | SKILL.md "Using Analysis Skill Results" |
| Inter-kernel gaps > 50us, eager kernels between graph segments | CUSTOM_OP or GRAPH_BREAK | C++ custom op or GRAPH_EXPAND | [patterns/gpu-graph.md](patterns/gpu-graph.md) |
| Graph coverage ratio < expected | GRAPH_BREAK | GRAPH_EXPAND (partition poisoning) or GRAPH_SPLIT | [patterns/gpu-graph.md](patterns/gpu-graph.md) |
| NCCL ratio > 0.20 | COMM | See communication-patterns.md | [communication-patterns.md](communication-patterns.md) |
| Straggler rank with extra host work | Per-rank | Optimize coordinator-only code paths | analysis skill kernel-level drill-down |
| GC NVTX markers in hot loop | GC | gc.freeze(), gc.disable() | [patterns/system.md](patterns/system.md) GC_MGMT |
| Periodic GPU idle gaps uncorrelated with kernels | GC | gc.freeze() after model load | [patterns/system.md](patterns/system.md) GC_MGMT |

### Quick Triage Flowchart

```
1. Is the hotspot .item()/.cpu()/.numpy()?
   YES -> Is Hits = N_layers?
          YES -> HOST_SYNC -> HOIST Variant B
          NO  -> SYNC -> Pattern 1-5 by context
   NO  -> continue

2. Is it a chain of .view()/.to()/.contiguous()?
   YES -> CUSTOM_OP -> C++ custom op
   NO  -> continue

3. Is it Enum(int) construction?
   YES -> ENUM_CONSTRUCT -> raw int compare
   NO  -> continue

4. Is it torch.empty/zeros/clone in a loop?
   YES -> ALLOC -> pre-allocate at init
   NO  -> continue

5. Is it accessing a C++ binding attribute (Per Hit ~0.2-0.5us)?
   YES -> BOUNDARY -> feature-gate or shadow
   NO  -> continue

6. Is it a lock/queue/event operation?
   YES -> GIL -> fine-grained locking
   NO  -> continue

7. Is it dist.all_reduce/barrier/send?
   YES -> COMM -> see communication-patterns.md
   NO  -> continue

8. None of the above -> classify as FUNCALL, CONTAINER, PYLOOP,
   REDUNDANT_ITER, DEAD_WORK, or COMPUTE based on code structure
```
