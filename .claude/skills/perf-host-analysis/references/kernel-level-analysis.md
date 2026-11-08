# Kernel-Level Analysis Techniques

GPU kernel-level analysis techniques for diagnosing host overhead at sub-operation granularity. Use these after Phase 2 (Root Cause) identifies specific operations to drill into, or when NVTX-level analysis lacks the resolution to pinpoint the bottleneck source.

---

## Inter-Kernel Gap Analysis (Bubble Profiling)

**When to use**: GPU utilization is low but individual GPU kernels are fast. You suspect the overhead is *between* kernels rather than *within* them.

### Method

1. Capture an nsys trace with CUDA API + kernel correlation
2. For each consecutive kernel pair on the same stream, compute:
   ```
   gap = kernel[i+1].start - kernel[i].end
   ```
3. Bucket gaps by size:

   | Bucket | Typical Source |
   |--------|---------------|
   | < 1us | Normal kernel dispatch pipeline |
   | 1-5us | CUDA graph segment replay overhead |
   | 5-10us | `cudaGraphLaunch` dispatch |
   | 10-50us | Light Python dispatch between graph segments |
   | 50-100us | End-to-end gap between CUDA graph segment replays (includes Python dispatch loop) |
   | 100-500us | Python interpreter overhead between kernel launches |
   | 500us-1ms | Heavy Python processing (tensor view chains, metadata preparation) |
   | 1-5ms | Python interpreter overhead in eager code paths |
   | > 5ms | Host-device sync (`.item()`) or runtime object creation (`cudaEventCreate`) |

4. The bucket distribution reveals the bottleneck type:
   - **Dominated by 50-100us**: Piecewise graph replay dispatch overhead
   - **Dominated by 1-5ms**: Python interpreter overhead between kernel launches
   - **Dominated by >5ms**: Host-device synchronization or runtime object creation
5. Identify the top-N largest gaps and map each to source code via kernel name correlation or NVTX annotations

### Key Insight

The gap between two kernel launches is **pure host overhead** — the GPU is idle, waiting for the CPU to issue the next launch. This is the optimization target.

### Example

```
TRT-LLM bubble distribution:
  50-100us: 13,543 gaps, 956ms total (39.6%)   <- piecewise graph replay overhead
  1-5ms:    200 gaps, 313ms total (13.0%)       <- Python in eager MTP tail
  >5ms:     21 gaps, 237ms total (9.8%)         <- cudaEventCreateWithFlags
```

### SQL Query Template

```sql
-- Compute inter-kernel gaps on the same stream
WITH ordered_kernels AS (
    SELECT
        start AS k_start,
        end AS k_end,
        deviceId,
        streamId,
        ROW_NUMBER() OVER (PARTITION BY deviceId, streamId ORDER BY start) AS rn
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    a.k_end AS prev_end,
    b.k_start AS next_start,
    (b.k_start - a.k_end) AS gap_ns
FROM ordered_kernels a
JOIN ordered_kernels b
    ON a.deviceId = b.deviceId
    AND a.streamId = b.streamId
    AND a.rn + 1 = b.rn
WHERE b.k_start > a.k_end  -- Only positive gaps
ORDER BY gap_ns DESC
LIMIT 100;
```

---

## Eager vs Graph Kernel Classification

**When to use**: The system uses piecewise CUDA graph capture and you need to know what is captured vs what runs eagerly.

### Method

1. In nsys, filter kernels by launch API:
   - `cudaGraphLaunch` -> graph-captured (good)
   - `cudaLaunchKernel` / `cuLaunchKernel` -> eager (potential overhead)
2. Compute graph coverage ratio:
   ```
   graph_coverage = total_graph_kernels / total_kernels
   ```
3. For each eager kernel, trace back to source code via:
   - Kernel name -> known kernel mapping (e.g., `triton_red_fused_cumsum_sub_0` -> `torch.cumsum`)
   - NVTX annotations if available
   - Launch timestamp correlation with Python profiler

### Key Insight

Eager kernels are not inherently bad — attention custom ops that depend on variable batch metadata (dynamic shapes, KV cache state) must remain eager. The problem is when **graph-capturable work** (GEMMs, norms, allreduce) runs eagerly because something upstream broke graph partitioning.

### Graph Break Detection

Common causes of graph partition breaks:
- `.item()` calls (host-device synchronization)
- `torch.cumsum` / `index.Tensor` with dynamic shapes
- C extension calls opaque to the CUDA graph runtime
- Python control flow depending on tensor values
- `torch.autograd` operations

When a single non-capturable op triggers `stop_partition = True`, it can kill graph capture for **all subsequent ops** in the same scope. See the GRAPH_EXPAND pattern in the optimization skill's [patterns/gpu-graph.md](../../perf-host-optimization/references/patterns/gpu-graph.md) for fix strategies.

### Example

234 MTP kernels ran eagerly because one `cumsum` op triggered `stop_partition = True`, killing graph capture for all subsequent ops.

---

## Repeating-Pattern Source Mapping

**When to use**: The same kernel pattern repeats N times (once per transformer layer). You need to map each kernel to its source code and quantify per-group overhead.

### Method

1. Isolate one repetition of the pattern (e.g., kernels between consecutive attention ops)
2. Number each kernel 0..K-1
3. For each kernel, identify:
   - Kernel name -> operation type (GEMM, LayerNorm, RoPE, elementwise, scatter, etc.)
   - Preceding gap duration -> host overhead source
   - Source file:line via kernel name + code path tracing
4. Group into functional blocks (projections, indexer, attention dispatch)
5. Sum gaps per group -> prioritize groups with highest total overhead

### Output Format

```
Group  | Kernels  | Operation              | Total Gap | Priority
-------|----------|------------------------|-----------|--------
A      | 0-19     | Indexer projections     | 340us     | Medium
B      | 20-29    | Sparse attention indexer | 180us     | Low
C      | 30       | Post-DSA GEMM           | 15us      | Low
D      | 31-42    | MLA attention dispatch   | 526us     | HIGH
```

### Key Insight

Per-layer overhead compounds: a 526us overhead in Group D x 61 layers = **32ms** of pure host stall per step. Focus on the group with the highest `gap * num_layers` product.

---

## Straggler / Rank Imbalance Detection

**When to use**: Multi-GPU (TP/PP) inference has unexplained slowdowns despite balanced kernel execution times.

### Method

1. Capture nsys traces on ALL ranks simultaneously
2. For each NCCL collective, measure per-rank arrival time (timestamp of the rank's entry into the collective)
3. Compute:
   ```
   straggler_rank = argmax(arrival_time) per collective
   ```
4. If one rank is consistently the straggler (>80% of iterations), investigate:
   - Extra host work on that rank (coordinator duties, metadata preparation)
   - Asymmetric kernel launches (one rank launches more kernels)
   - Memory pressure differences (one rank does more allocation)

### GPU Queue Depth Positive Feedback Loop

A straggler rank can create a self-reinforcing performance degradation:

```
Rank 0 has extra host work
  -> arrives late to NCCL collective
  -> short NCCL wait (no time to build GPU queue)
  -> shallow GPU queue -> GPU idles between launches
  -> arrives even later to next NCCL collective
  -> self-reinforcing degradation
```

This feedback loop means that a **small** host overhead imbalance (e.g., coordinator rank doing 500us extra metadata work) can amplify into a **large** per-step slowdown (e.g., 5ms) because the shallow GPU queue causes gaps between every kernel launch on the straggler rank.

### Fix Strategies

1. **Reduce host overhead on the straggler rank** — apply optimization techniques from the optimization skill specifically to the coordinator-only code paths
2. **Pre-launch kernels to build GPU queue depth** before the NCCL sync point — ensure enough kernels are queued so the GPU stays busy during the wait
3. **Balance coordinator duties** — distribute metadata preparation across ranks rather than concentrating it on rank 0

### Detection via nsys SQL

```sql
-- Find NCCL collective start times per rank
-- Compare arrival times across ranks for the same collective instance
SELECT
    k.shortName,
    k.deviceId AS rank,
    k.start AS arrival_time,
    k.end - k.start AS duration
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%nccl%'
ORDER BY k.start;
```
