# Wide Expert Parallelism (Wide-EP) in TensorRT-LLM

TensorRT-LLM's Wide Expert Parallelism (Wide-EP) feature enables efficient inference of large-scale Mixture-of-Experts (MoE) models by scaling expert parallelism beyond traditional limits. This feature addresses the inherent workload imbalance challenges in large-scale MoE models and provides both offline and online load balancing capabilities.

## Overview

Large-scale MoE models like DeepSeek-V3/R1, LLaMA4, and Qwen3 use fine-grained expert designs that introduce new challenges for inference systems:

- **High memory demands** for expert weights
- **Inherent expert-level workload imbalance** due to sparse execution patterns
- **Communication overhead** in distributed expert parallelism

Wide-EP solves these challenges through:

- **Custom EP communication kernels** optimized for NVIDIA GB200 Multi-Node NVLink (MNNVL)
- **Expert Parallelism Load Balancer (EPLB)** with both offline and online modes
- **Dynamic expert placement and replication** strategies
- **Layer-wise weight redistribution** to minimize inference disruption

## Configuration Options

### MoE Configuration Parameters

An example yaml file to enable wide EP with load balancer:
```yaml
moe_config:
    backend: WideEP
    max_num_tokens: 9216
    load_balancer: ./moe_load_balancer.yaml
```

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `backend` | MoE backend type | `CUTLASS` | Set to `WideEP` to enable wide EP |
| `max_num_tokens` | If set, at most max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time.  | `None` | If the number of tokens exceeds max_num_tokens, the input tensors will be split into chunks and a for loop will be used. |
| `load_balancer` | Configuration for MoE load balancing | `None` | Set path to the yaml file |

### Load Balancer Configuration

An example yaml file to configure online EP balancer:
```yaml
num_slots: 288
layer_updates_per_iter: 1
```

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `num_slots` | Total number of expert slots | `None` | Must be ≥ total experts |
| `layer_updates_per_iter` | Number of layers updated per iteration | `0` | `0` = offline, `>0` = online |

## Quick Start

### 1. Experiment with Load Balancer

Refer to the [ep_load_balancer](./ep_load_balancer/) directory.

### 2. Execute Wide-EP on SLURM Clusters

Refer to the [slurm_scripts](./slurm_scripts/) directory.

## References

- [Technical Blog: Scaling Expert Parallelism in TensorRT-LLM](docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)

For detailed implementation examples and advanced usage, see the subdirectories:
- [`ep_load_balancer/`](ep_load_balancer/): Load balancing tools and examples
- [`slurm_scripts/`](slurm_scripts/): Cluster deployment scripts
