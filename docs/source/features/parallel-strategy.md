# Parallelism in TensorRT-LLM

Parallelism across multiple GPUs becomes necessary when either
* the model cannot fit in a single GPU’s memory, or
* a single GPU cannot deliver the desired performance.

TensorRT-LLM supports multiple parallelism strategies (listed below) for deployment on both single and multiple nodes.
* Tensor Parallel
* Pipeline Parallel
* Data Parallel
* Expert Parallel
* Context Parallel

# Module-level Guide
## Attention Module

TensorRT-LLM currently supports two strategies:

- **Tensor Parallelism (TP)** — best for small batch sizes
- **Data Parallelism (DP)** — best for large batch sizes

### Tensor Parallelism (TP)

* The GEMM weights before and after the attention kernel are evenly sharded across GPUs, as are the attention `num_heads`.
* Exceptions
  1. **DeepSeek-R1**: the `fused_A` GEMM is *not* sharded.
  2. **GQA / MQA / MLA**: if `num_heads < tensor_parallel_size`, the KV-cache is replicated on every GPU.

### Data Parallelism (DP)

* All GEMM weights are **replicated** on every GPU.
* The KV-cache is **partitioned**, because different user requests are routed to different DP ranks.

### How to Enable

To deploy a model with the above parallel strategies using `trtllm-serve` or run benchmarking with `trtllm-bench`, create a YAML configuration file named `parallel_config.yaml` as follows:

```bash
cat <<EOF > parallel_config.yaml
# TP-8
tensor_parallel_size: 8
enable_attention_dp: false    # default
# DP-8
tensor_parallel_size: 8
enable_attention_dp: true
EOF
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

### How to Enable

To deploy a model with the above parallel strategies using `trtllm-serve` or run benchmarking with `trtllm-bench`, create a YAML configuration file named `parallel_config.yaml` as follows:

```bash
cat <<EOF > parallel_config.yaml
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
EOF
```
```{note}
The product of `moe_tensor_parallel_size` and `moe_expert_parallel_size` must equal `tensor_parallel_size`.
```
