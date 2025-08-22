
# Parallelism in TensorRT-LLM

Parallelism across multiple GPUs is required when either
* the model cannot fit in a single GPU’s memory, or
* a single GPU cannot deliver the desired performance.

---

## Attention Module

TensorRT-LLM currently supports two strategies:

**Tensor Parallelism (TP)** — best for small batch sizes
**Data Parallelism (DP)** — best for large batch sizes

### Tensor Parallelism

* The GEMM weights before and after the attention kernel are evenly sharded across GPUs, as are the attention `num_heads`.
* Exceptions
  1. **DeepSeek-R1**: the `fused_A` GEMM is *not* sharded.
  2. **GQA / MQA / MLA**: if `num_heads < tensor_parallel_size`, the KV-cache is replicated on every GPU.

### Data Parallelism

* All GEMM weights are **replicated** on every GPU.
* The KV-cache is **partitioned**, because different user requests are routed to different DP ranks.

### How to enable

```yaml
# TP-8
tensor_parallel_size: 8
enable_attention_dp: false    # default
# DP-8
tensor_parallel_size: 8
enable_attention_dp: true
```

## FFN Module

### Dense models

Tensor Parallelism is supported for the FFN layers of dense models.

### Mixture of Experts (MoE)

MoE replaces a single FFN with multiple experts. A router selects the top-k experts for each token and dispatches the corresponding hidden states.
TensorRT-LLM supports three execution patterns: Tensor Parallelism (TP), Expert Parallelism (EP) and Hybrid (TP × EP)

* **TP** - Every expert’s weight matrix is sliced across all GPUs. Each GPU sees all tokens.
* **EP** -  Full weights of each expert reside on a single GPU. Each GPU only sees tokens routed to its local experts.
* **Hybrid ETP** - Each GPU stores a subset of experts (EP) and shards those weights further (TP), balancing workload and kernel efficiency.

#### How to enable

```yaml
# TP only
tensor_parallel_size: 8
moe_tensor_parallel_size: 8

# EP only
tensor_parallel_size: 8
moe_expert_parallel_size: 8

# Hybrid (TP-4 × EP-2)
tensor_parallel_size: 8      # 4 × 2
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 2
```

The product of `moe_tensor_parallel_size` and `moe_expert_parallel_size` must equal `tensor_parallel_size`.

## Pipeline Parallelism

Enable pipeline parallelism with:

```yaml
pipeline_parallel_size: 4    # example
```
