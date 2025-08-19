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

## Quick Start

### Prerequisites

* GPU: GB200 NVL72, H20, or RTX 6000D.
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

For GB200 NVL72, to make sure that Multi-Node NVLink (MNNVL) is correctly setup, check if the path `/dev/nvidia-caps-imex-channels` exists in the container. If the path doesn't exist, mount it when launching the Docker container.

For more information on NVIDIA IMEX service for NVLink networks, refer to https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html.

### Configurations

An example yaml file to enable wide EP:
```yaml
moe_config:
    backend: WIDEEP
    max_num_tokens: 9216
    load_balancer: moe_load_balancer.yaml # (optional) enable load balancer
```

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `backend` | MoE backend type | `CUTLASS` | Set to `WIDEEP` to enable wide EP |
| `max_num_tokens` | If set, at most max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time.  | `None` | If the number of tokens exceeds max_num_tokens, the input tensors will be split into chunks and a for loop will be used. |
| `load_balancer` | Configuration for MoE load balancing | `None` | Set path to the yaml file |

#### Online Load Balancer Configuration

```yaml
moe_config:
    backend: WIDEEP
    max_num_tokens: 9216
    load_balancer:
        num_slots: 288
        layer_updates_per_iter: 1
```

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `num_slots` | Total number of expert slots | `None` | Must be ≥ total experts |
| `layer_updates_per_iter` | Number of layers updated per iteration | `0` | `0` = offline, `>0` = online |

#### Offline Load Balancer Configuration

Refer to the [Offline EP Load Balancer](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#offline-ep-load-balancer) documentation.

*Online EP Load Balancer is more suitable for production deployment needs to react timely to the online traffic changes.*

### Execute Wide-EP on SLURM Clusters

Refer to the [slurm_scripts](./slurm_scripts/) directory, which reuses [disaggregated slurm scripts](../disaggregated/slurm/) to automatically generate configuration files and submit jobs to SLURM clusters.

## Trouble shooting

### Transparent HugePages failure

When getting exception `madvise(MADV_HUGEPAGE) failed.`, check if Transparent Hugepages has been enabled.
```bash
>$ cat /sys/kernel/mm/transparent_hugepage/enabled
always [madvise] never
>$ cat /sys/kernel/mm/transparent_hugepage/defrag
always defer defer+madvise [madvise] never
```
If `never` is highlighted, enable Transparent HugePages by the following command.
```bash
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
```

### GB200 NUMA binding

GPU memory are also on NUMA nodes on GB200 and system can also use that. Bind memory to CPU nodes to avoid GPU memory being used as host memory.
```bash
numactl -m 0,1 <command>
```

### Shared Memory Clean Up on EPLB

To achieve online load balance, all expert weights are stored in shared host memory. 4 ranks on same GB200 node share the same expert weights to save memory. Normally, these shared host memory will be cleaned up at process exit, but they may not get chance to be cleaned if an abnormal exit happens.

In that case, when seeing the following (or similar) error message:
```
FileExistsError: [Errno 17] File exists: '/moe_shared_l0_lr0_all'
```
you need to manually check `/dev/shm` directory and delete `/dev/shm/moe_shared_*` if any.

### Disaggregated serving related issues

Refer to the [Troubleshooting and FAQ](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/advanced/disaggregated-service.md#troubleshooting-and-faq) section of Disaggregated-Service.

## References

- Technical Blog: Scaling Expert Parallelism in TensorRT-LLM
  - [Part 1: Design and Implementation of Large-scale EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
  - [Part 2: Performance Status and Optimization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md)

For detailed implementation examples and advanced usage, see the subdirectories:
- [`ep_load_balancer/`](ep_load_balancer/): Load balancing tools and examples
- [`slurm_scripts/`](slurm_scripts/): Cluster deployment scripts
