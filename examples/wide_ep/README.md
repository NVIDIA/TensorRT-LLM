# Wide Expert Parallelism (Wide-EP) in TensorRT-LLM

TensorRT-LLM's Wide Expert Parallelism (Wide-EP) feature enables efficient inference of large-scale Mixture-of-Experts (MoE) models by scaling expert parallelism beyond traditional limits. This feature addresses the inherent workload imbalance challenges in large-scale MoE models and provides both offline and online load balancing capabilities.

## Overview

Large-scale MoE models like DeepSeek-V3/R1, Kimi K2 Thinking, LLaMA4, and Qwen3 use fine-grained expert designs that introduce new challenges for inference systems:

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

* GPU: GB200 NVL72, GB300 NVL72, H20, or RTX 6000D.
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

For GB200/GB300 NVL72, to make sure that Multi-Node NVLink (MNNVL) is correctly setup, check if the path `/dev/nvidia-caps-imex-channels` exists in the container. If the path doesn't exist, mount it when launching the Docker container.

For more information on NVIDIA IMEX service for NVLink networks, refer to https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html.

#### Coherent Driver-Based Memory Management (CDMM)

Starting from R580 Driver, [Coherent Driver-Based Memory Management (CDMM)](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06/index.html#hardware-software-support) for GB200 platforms is introduced. With CDMM, the driver manages GPU memory instead of the OS. CDMM avoids OS onlining of the GPU memory and the exposing of the GPU memory as a NUMA node to the OS. In Wide-EP, online EPLB needs host threads to be able to access the GPU memory to do the weights update.

When CDMM mode is off, GPU memory is exposed as NUMA nodes, so no additional prerequisites are required.

When CDMM mode is on, GPU memory doesn't exist in NUMA nodes. In that case, if online EPLB is needed, [GDRCopy](https://github.com/NVIDIA/gdrcopy?tab=readme-ov-file#build-and-installation) needs to be installed.

When GDRCopy is installed and the kernel module is loaded, you should be able to see the device file `/dev/gdrdrv` and kernel module `gdrdrv` by `lsmod`. The device file needs to be mapped into the container.

* For docker, this can be done by adding a device mapping like `--device=/dev/gdrdrv:/dev/gdrdrv`.
* For slurm with enroot, `--container-mounts="/dev/gdrdrv:/dev/gdrdrv"` needs to be added when starting containers and environment variable `export ENROOT_ALLOW_DEV=yes` needs to be set.

### Online Load Balancer Configurations

An example yaml file to enable wide EP:
```yaml
moe_config:
    backend: WIDEEP
    max_num_tokens: 9216
    load_balancer:
      num_slots: 288
      layer_updates_per_iter: 1 # (optional) enable online load balancer
```

#### `backend`

 - MoE backend type, defaults to `CUTLASS`.
 - Currently, TensorRT LLM has multiple MoE backends that support wide EP, including `WIDEEP`, `CUTLASS`, `TRTLLM` and `CUTEDSL`. There are on-going efforts to refactor the backends so that we don't necessarily need a specific `WIDEEP` backend, and each other backend will support wide EP functionality.

#### `max_num_tokens`

If set, at most `max_num_tokens` tokens will be sent to `torch.ops.trtllm.fused_moe` at the same time. If the number of tokens exceeds `max_num_tokens`, the input tensors will be split into chunks and a for loop will be used.

#### `load_balancer`

Configuration for MoE load balancing, users can directly set `num_slots` and `layer_updates_per_iter` as online EPLB settings, while set path to a YAML file that also includes `initial_global_assignments` for offline EPLB.

#### `num_slots`

Total number of expert slots, must be ≥ total experts. Three typical settings:

1. Set to 0. MoE load balancing is disabled.
2. Set to number of total experts, such as 256 for DeepSeek R1.
3. Set to number of total experts + EP size, such as 288 for DeepSeek R1, 32-way EP.
   * This means there is 1 extra expert on each EP rank, so that there is more room for the per-rank token distribution to be more balanced.

#### `layer_updates_per_iter`

Number of layers updated per iteration, defaults to `0`. `0` means offline, while `>0` means online EPLB.

### Offline Load Balancer Configuration

Refer to the [Offline EP Load Balancer](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#offline-ep-load-balancer) documentation.

*Note: Online EP Load Balancer is more suitable for production deployments that need to react timely to online traffic changes.*

### Execute Wide-EP on SLURM Clusters

Refer to the [slurm_scripts](./slurm_scripts/) directory, which reuses [disaggregated slurm scripts](../disaggregated/slurm/) for submitting jobs to SLURM clusters.

## Troubleshooting

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

### GB200/GB300 NVL72 NUMA binding

GPU memory is also on NUMA nodes on GB200/GB300 NVL72 and the system can also use that. Bind memory to CPU nodes to avoid GPU memory being used as host memory.
```bash
numactl -m 0,1 <command>
```

### Shared Memory on EPLB

To achieve online load balancing, all expert weights are stored in shared host memory. Four ranks on the same GB200/GB300 NVL72 node share the same expert weights to save memory.

There is one environment variable `TRTLLM_EPLB_SHM_NAME` to specify the base name of the shared memory. This environment variable may need to be specified if there are multiple instances on one node. If not, you can ignore it.

The default value of `TRTLLM_EPLB_SHM_NAME` is `moe_shared`. When `TRTLLM_EPLB_SHM_NAME` is set to `moe_shared`, the shared memory segments will be named as `moe_shared_l0_lr0_all`, `moe_shared_l1_lr0_all`, and so on. Here `l0` means the first layer with EPLB, and `lr0` means that it is the part loaded by local rank 0, and `all` means it contains all expert weights of each expert.

Normally, these shared memory segments will be cleaned up automatically at process exit. However, they may not get the chance to be cleaned up if an abnormal exit occurs. Therefore, EPLB will automatically clean up leftover shared memory with the same name that already exists before creating new segments.

If you experience an abnormal exit and are concerned about the shared memory usage before the next run, you need to manually check the `/dev/shm` directory and delete any `/dev/shm/moe_shared_*` files if present.

#### Manual Cleanup Commands

For manual cleanup of shared memory, you can use the following commands:

```bash
# List all moe_shared related shared memory
ls -la /dev/shm/moe_shared_*

# Remove all moe_shared related shared memory
rm -f /dev/shm/moe_shared_*

# Or remove specific shared memory segments
rm -f /dev/shm/moe_shared_l0_lr0_all
```

**Warning:** Be careful when removing shared memory manually, as this may affect running processes that depend on these shared memory segments.

### Host OOM

Since EPLB requires all experts to be loaded on host memory, when some models (such as Kimi K2 Thinking) have larger weights size, it's possible seeing host OOM issues, as the following:

```log
Loading weights: 100%|█████████████████████| 1408/1408 [03:43<00:00,  6.30it/s]
 0: [12/04/2025-18:38:28] [TRT-LLM] [RANK 0] [I] moe_load_balancer finalizing model...
 1: [nvl72136-T14:452151:0:452151] Caught signal 7 (Bus error: nonexistent physical address)
 1: ==== backtrace (tid: 452151) ====
 1:  0  /usr/local/ucx//lib/libucs.so.0(ucs_handle_error+0x2cc) [0xffff9638274c]
 1:  1  /usr/local/ucx//lib/libucs.so.0(+0x328fc) [0xffff963828fc]
 1:  2  /usr/local/ucx//lib/libucs.so.0(+0x32c78) [0xffff96382c78]
```
This can be addressed by mounting `tmpfs:/dev/shm:size=640G` when launching the Docker container, to increase the shm size that the container can access.

### Disaggregated serving related issues

Refer to the [Troubleshooting and FAQ](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md#troubleshooting-and-faq) section of Disaggregated-Service.

## References

To understand more details on wide EP and the optimizations we've added, refer to the technical blog series: Scaling Expert Parallelism in TensorRT-LLM
  - [Part 1: Design and Implementation of Large-scale EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
  - [Part 2: Performance Status and Optimization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md)
  - [Part 3: Pushing the Performance Boundary](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md)

To review how wide EP helps with Blackwell's leading inference benchmarks, also read these recent blog posts:
* [NVIDIA Blackwell Leads on SemiAnalysis InferenceMAX™ v1 Benchmarks](https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benchmarks/)
* [NVIDIA Blackwell Raises Bar in New InferenceMAX Benchmarks, Delivering Unmatched Performance and Efficiency](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/)

For detailed implementation examples and advanced usage, see the subdirectories:
- [`ep_load_balancer/`](ep_load_balancer/): Load balancing tools and examples
- [`slurm_scripts/`](slurm_scripts/): Cluster deployment scripts
