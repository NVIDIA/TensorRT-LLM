# Rubin GVR dispatch and locality

The FP32 variable-length entry uses a shape and resource based policy on SM107. The policy chooses existing GVR register, register-cluster, streaming and streaming-cluster kernels and their sampling parameters. It does not inspect scores, model names or layer identities. Other architectures, deterministic FP32, FP16/BF16 and the legacy batch-uniform entries retain their existing dispatch. Uncovered shapes use the original route. The sparse N table uses 2,048-candidate buckets; register capacity is recomputed from the actual envelope before applying a rule.

## Prepared locality execution

`gvr_topk_locality.PreparedRowTopK` is an opt-in interface for a caller that already owns row-local input memory. It requires CUDA locality APIs from cuda-bindings 13.4 or later. It neither moves logits nor changes the allocator. `row_counts[d]` describes a contiguous range of rows allocated in locality domain `d`; the counts must cover all rows. Domain count, per-domain SM resources and stream membership are queried from CUDA. No equal partition or fixed SM count is assumed.

The policy can retain tuned global execution, select the original global route, or issue one GVR invocation per nonempty domain. `local_preserve` plans with the full row count and physical SM profile, then binds each invocation to its actual row slice. `local_partition` plans using each domain's row count and queried SM count. The cluster kernel still derives some internal occupancy flags from the actual slice size; preserving the host plan does not force those device flags to use the full batch. Every concurrent domain invocation owns a separate workspace.

Only measured FP32 shape buckets have automatic locality choices. K=512/1024/2048 and FP16/BF16 remain accepted, with global fallback for unmeasured dtypes. Devices with no CUDA locality capability also retain global execution. The policy normalizes rows by available SM count, and the executor verifies actual resource limits and row ownership before using a local method. Input layouts that do not match the declared domain ownership, and unbalanced row assignments outside the measured resource proportions, use global execution.

```python
import torch

from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_topk_locality import PreparedRowTopK

# logits already contains caller-owned, domain-local row ranges.
# kv_lens and max_seq_len are uncompressed token counts, as in run_varlen.
prepared = PreparedRowTopK(
    logits,
    kv_lens,
    indices,
    max_seq_len=max_seq_len,
    compress_ratio=4,
    row_counts=row_counts,
    co_local=True,
)
prepared()  # Warm outside capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    prepared()
graph.replay()

# Destroy all graphs that reference the prepared call before closing it.
graph.reset()
prepared.close()
```

Preparation performs resource queries, placement checks, compilation, workspace allocation and initialization outside capture. Replay issues only the bound kernels and necessary stream-event dependencies. The measured local interval includes the fork and join. Values and device lengths may change in place within the envelope; tensor addresses, shapes, placement and resource ownership must stay fixed. A prepared call is not reentrant: simultaneous invocations need separate prepared calls and workspaces. This row API uses `next_n=1`; MTP batches continue to use `run_varlen`.

`co_local=False` retains global execution and requires no domain resource creation. An optional `LocalityDomainExecutor` can be supplied for resource reuse; it must be on the same device and remain alive while local calls or their graphs can execute. Its `coscheduled_sm_count` describes the SM grouping guaranteed by CUDA for cluster scheduling, not the total number of SMs in a domain. CUDA returns the actual number of SMs available in each domain for that grouping.
