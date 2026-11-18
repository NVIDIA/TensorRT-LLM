# Rubin GVR dispatch and locality

## Dispatch

SM107 variable-length dispatch supports FP32, FP16 and BF16 with K=512/1024/2048. Selection depends on dtype, shape and SM resources. FP32 selects existing register, cluster and streaming kernels with capacity checks. FP16/BF16 select native code-radix and PCW options. Host option selection preserves the actual compilation target and device identity in cache keys.

Sparse K=2048 low-precision batches use the resident selector. Sparse 16-CTA FP16 clusters use 1,024 threads per CTA with proportionally fewer vectors per thread. BF16 PCW accepts 16-byte row alignment for N>2048; shorter rows require 64-element alignment. PCW uses 512 histogram bins for K=2048 and 1,024 otherwise, with the bin count included in the compilation cache key.

## Locality execution

`PreparedRowTopK` verifies domain ownership of complete padded rows before selecting `local_preserve`. Optional `row_counts` must match that ownership. Domain counts and SM budgets are queried dynamically; row assignments must follow the resource proportions. Ordinary allocations, unknown ownership and unsupported resource configurations use tuned global execution. `run_varlen` remains independent of locality detection.

`local_preserve` selects the host plan using the full batch and physical SM count, then binds each domain's row slice to a separate workspace. Device logic uses the actual slice size. Coscheduled SM grouping is four; the final kernel cluster requirement must fit that grouping. This grouping specifies cluster scheduling support, not the total SM budget per domain.

Locality supports FP32/FP16/BF16 and K=512/1024/2048 on SM107 and later when CUDA locality APIs are available. `gvr_topk_locality_policy.py` defines the shape thresholds. Open upper bounds do not override kernel, indexing or allocation limits, and do not guarantee a speedup.

## Remainder backfill

Automatic BACKFILL requires verified localized FP32 input, K=512/1024, N≥131072 and B≥737, where B=ceil(BS×208/physical_SM_count). Within N≤262143 and B≤4096, eligible B intervals are [737,832], [921,1024] and [1473,1664]. Above either upper bound, all shapes meeting the lower thresholds are eligible. FP16/BF16 and K=2048 do not automatically request BACKFILL.

The executor distributes available remainder in aligned increments while preserving domain budget proportions. Unsupported requests or invalid returned budgets retain the original partitions; other driver errors propagate. Added SMs are not guaranteed to physically belong to the preferred domain. Explicit `sm_counts` override `backfill`, and caller-owned executors retain their requested resource policy.

## CUDA Graph usage

Prepare and warm before capture. Resource discovery, ownership checks, compilation and workspace initialization occur during preparation. Replay contains the bound kernels and stream-event dependencies.

```python
import torch

from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_topk_locality import PreparedRowTopK

prepared = PreparedRowTopK(
    logits,
    kv_lens,
    indices,
    max_seq_len=max_seq_len,
    compress_ratio=4,
)
prepared()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    prepared()
graph.replay()
graph.reset()
prepared.close()
```

Lengths use uncompressed token counts. This API supports `next_n=1`; MTP batches use `run_varlen`. Values and lengths may change within the prepared envelope, but addresses, shapes, ownership and resources must remain valid until captured graphs are destroyed. Concurrent calls require separate prepared instances and workspaces.

`co_local=False` disables locality; `True` still requires verified allocation ownership. The caller owns placement: preparation does not copy or reallocate logits. An optional `LocalityDomainExecutor` must use the same device and outlive all calls and graphs referencing it.
