# Parallelism in TensorRT LLM

Parallelism across multiple GPUs becomes necessary when either
* the model cannot fit in a single GPU’s memory, or
* a single GPU cannot deliver the desired performance.

TensorRT LLM supports multiple parallelism strategies for deployment on both single and multiple nodes:
* **Tensor Parallel (TP)** - Shards model weights across GPUs
* **Pipeline Parallel (PP)** - Distributes model layers across GPUs
* **Data Parallel (DP)** - Replicates model across GPUs for different requests
* **Expert Parallel (EP)** - Distributes experts across GPUs for MoE models
* **Context Parallel (CP)** - Distributes context processing across GPUs
* **Wide Expert Parallel (Wide-EP)** - Advanced EP with load balancing for large-scale MoE models

## Overview of Parallelism Strategies

### Tensor Parallelism (TP)
Tensor parallelism splits the model weights across multiple GPUs. Each GPU holds a portion of the weights and processes the same input tokens, with results combined through communication.

**Best for:** Small batch sizes, memory-constrained scenarios

### Pipeline Parallelism (PP)
Pipeline parallelism distributes different layers of the model across multiple GPUs. Each GPU processes a subset of layers, with activations passed between GPUs.

**Best for:** Large models that don't fit in single GPU memory

### Data Parallelism (DP)
Data parallelism replicates the entire model across multiple GPUs. Each GPU processes different requests independently.

**Best for:** Large batch sizes, high throughput scenarios

### Expert Parallelism (EP)
Expert parallelism is specifically designed for Mixture of Experts (MoE) models, where different experts are distributed across GPUs.

**Best for:** MoE models with high expert count

### Context Parallelism (CP)
Context parallelism distributes the processing of long sequences across multiple GPUs.

**Best for:** Long context scenarios

### Wide Expert Parallelism (Wide-EP)
Wide-EP is an advanced form of expert parallelism that addresses the inherent workload imbalance in large-scale MoE models through intelligent load balancing and expert replication.

**Best for:** Large-scale MoE models like DeepSeek-V3/R1, LLaMA4, Qwen3

## Module-level Parallelism Guide

### Attention Module

TensorRT LLM supports two strategies for attention modules:

- **Tensor Parallelism (TP)** — best for small batch sizes
- **Data Parallelism (DP)** — best for large batch sizes

#### Tensor Parallelism (TP)

* The GEMM weights before and after the attention kernel are evenly sharded across GPUs, as are the attention `num_heads`.
* Exceptions:
  1. **DeepSeek-R1**: the `fused_A` GEMM is *not* sharded.
  2. **GQA / MQA / MLA**: if `num_heads < tensor_parallel_size`, the KV-cache is replicated on every GPU.

#### Data Parallelism (DP)

* All GEMM weights are **replicated** on every GPU.
* The KV-cache is **partitioned**, because different user requests are routed to different DP ranks.

#### How to Enable Attention Parallelism

To deploy a model with the above parallel strategies using `trtllm-serve` or run benchmarking with `trtllm-bench`, create a YAML configuration file named `parallel_config.yaml`:

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

then set `--config parallel_config.yaml` in `trtllm-serve` or `trtllm-bench`.

### Multimodal encoders

`multimodal_config.encoder_data_parallel_size` controls encoder parallelism
independently of LLM attention parallelism. Omitting it (or setting it to
`null`) preserves the model's existing encoder mapping. It does not turn on
encoder weight replication when LLM attention DP is enabled.

An explicit value divides the LLM TP group into that many encoder replicas.
Each replica uses `tensor_parallel_size / encoder_data_parallel_size` TP ranks.
The value must divide `tensor_parallel_size`; PP and CP must both be 1.
For example, on two GPUs:

| LLM attention | `enable_attention_dp` | `encoder_data_parallel_size` | Encoder |
|---|---|---|---|
| TP2 | `false` | `1` | TP2 |
| TP2 | `false` | `2` | DP2 |
| DP2 | `true` | `1` | TP2 |
| DP2 | `true` | `2` | DP2 |

```yaml
tensor_parallel_size: 2
enable_attention_dp: true
multimodal_config:
  encoder_data_parallel_size: 1  # Encoder TP2, with LLM attention DP2
```

With LLM TP, encoder DP distributes cache-miss items by physical encoder token
cost and gathers their embeddings in prompt order. With LLM attention DP and a
fully replicated encoder, each rank encodes only its own requests. With LLM
attention DP and encoder TP, all ranks participate in a common item-encoding
step, including ranks with only cache hits or no multimodal work. Prepared
inputs are shared one request owner's scheduled batch at a time; CUDA payloads
stay on the GPU, and only the owner keeps the returned embeddings in its cache.

Explicit encoder TP is supported by the native Qwen2.5-VL, Qwen3-VL, Mistral3,
EXAONE 4.5, and Gemma4 vision integrations. Gemma4 audio, Hugging Face Qwen2-VL,
and the encoder-free Gemma4 Unified front ends support encoder DP but do not
shard their weights. When a model enables multiple encoders, each must support
the requested configuration. For example, a Gemma4 model with audio enabled
rejects explicit encoder TP even though its vision encoder supports TP.
Encoder DP currently excludes encoder-only execution, encoder CUDA graphs, and
side-stream prefetch. Encoder TP with LLM attention DP requires item scheduling
and also excludes encoder CUDA graphs and side-stream prefetch. Models without
item scheduling, including Gemma4, reject shared encoder TP with attention DP
at startup.

Model integrations declare these capabilities through
`MultimodalModelMixin.get_multimodal_encoder_parallel_modes(model_config)`.
The method returns supported modes for each encoder enabled by that config,
including its projector. Image and video can share a `vision` entry:

```python
{"vision": frozenset({"tp", "dp"}), "audio": frozenset({"dp"})}
```

The model loader validates each encoder before constructing weights. A hybrid
encoder TP/DP configuration requires both modes. An empty mapping means explicit
encoder parallelism has not been integrated; omitting the setting continues to
use the existing model path. These declarations describe implemented capabilities;
model-specific accuracy and performance validation are separate requirements.

### FFN Module

#### Dense Models

Tensor Parallelism is supported for the FFN layers of dense models.

#### Mixture of Experts (MoE)

MoE replaces a single FFN with multiple experts. A router selects the top-k experts for each token and dispatches the corresponding hidden states.

TensorRT LLM supports three execution patterns for MoE:

* **TP** - Every expert's weight matrix is sliced across all GPUs. Each GPU sees all tokens.
* **EP** - Full weights of each expert reside on a single GPU. Each GPU only sees tokens routed to its local experts.
* **Hybrid ETP** - Each GPU stores a subset of experts (EP) and shards those weights further (TP), balancing workload and kernel efficiency.

#### How to Enable MoE Parallelism

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

## Wide Expert Parallelism (Wide-EP)

Wide Expert Parallelism (Wide-EP) is TensorRT LLM's advanced solution for large-scale MoE model inference. It addresses the challenges of traditional expert parallelism through intelligent load balancing and expert replication strategies.

### Motivation for Wide-EP

Large-scale MoE models like DeepSeek-V3/R1, LLaMA4, and Qwen3 use fine-grained expert designs that introduce new challenges:

- **High memory demands** for expert weights
- **Inherent expert-level workload imbalance** due to sparse execution patterns
- **Communication overhead** in distributed expert parallelism
- **Hot expert problem** where certain experts receive significantly more tokens than others

### Key Features of Wide-EP

#### 1. Expert Replication and Load Balancing
Wide-EP introduces the concept of **expert slots** that are decoupled from specific experts. This allows:
- Multiple replicas of hot experts across different GPUs
- Dynamic expert placement based on workload patterns
- Both offline and online load balancing strategies

#### 2. Custom EP Communication Kernels
- Optimized for NVIDIA GB200 Multi-Node NVLink (MNNVL)
- Efficient all-to-all communication for expert dispatch and combine
- Reduced communication overhead compared to traditional EP

#### 3. Expert Parallelism Load Balancer (EPLB)
- **Offline EPLB**: Pre-computed expert placement based on historical workload statistics
- **Online EPLB**: Dynamic expert placement that adapts to real-time traffic patterns
- Layer-wise weight redistribution to minimize inference disruption

### Architecture Overview

Wide-EP separates the concepts of **experts** and **slots**:
- **Expert**: The concept from the model's perspective (e.g., Expert 0, Expert 1, etc.)
- **Slot**: The concept from the model engine's perspective (e.g., Slot 0, Slot 1, etc.)

The system maintains a routing table that maps Expert IDs to Slot IDs, which can be updated by the load balancing policy.


### Best Practices

1. **Start with offline EPLB** for production deployments with known workload patterns
2. **Use online EPLB** for dynamic workloads or when traffic patterns change frequently
3. **Monitor expert statistics** to understand workload distribution
4. **Tune max_num_tokens** based on your memory constraints and EP size
5. **Test with representative datasets** to validate load balancing effectiveness

### References

- [Technical Blog: Scaling Expert Parallelism in TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437)
- [EPLB Implementation](https://github.com/deepseek-ai/EPLB)

For detailed implementation examples and advanced usage, see:
- [`examples/wide_ep/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/): Complete Wide-EP examples
- [`examples/wide_ep/ep_load_balancer/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer/): Load balancing tools
- [`examples/wide_ep/slurm_scripts/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/slurm_scripts/): Cluster deployment scripts
