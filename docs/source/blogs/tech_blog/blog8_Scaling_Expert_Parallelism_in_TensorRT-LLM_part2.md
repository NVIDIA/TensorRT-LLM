# Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization)

This blog post continues our previous work on [Scaling Expert Parallelism in TensorRT LLM (Part 1: Design and Implementation of Large-scale EP)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md), where we introduced the fundamental design and implementation of large-scale Expert Parallelism (EP) in TensorRT LLM. Building upon that foundation, we have made significant performance improvements through various optimizations, achieving better throughput and latency for large-scale MoE models.

*By NVIDIA TensorRT LLM Team*

## Table of Contents
- [Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization)](#scaling-expert-parallelism-in-tensorrt-llm-part-2-performance-status-and-optimization)
  - [Table of Contents](#table-of-contents)
  - [Optimization Highlights](#optimization-highlights)
    - [Kernel Optimizations](#kernel-optimizations)
      - [MoE Auxiliary Kernels](#moe-auxiliary-kernels)
      - [Communication Kernels](#communication-kernels)
    - [Expert Parallelism Load Balancer (EPLB)](#expert-parallelism-load-balancer-eplb)
      - [Attempts at Online EPLB Implementation](#attempts-at-online-eplb-implementation)
        - [1. Initial Approach for Weight Updating - cudaMemcpyAsync](#1-initial-approach-for-weight-updating---cudamemcpyasync)
        - [2. Avoiding Deadlock - Multithreaded CPU Copy with Managed Memory](#2-avoiding-deadlock---multithreaded-cpu-copy-with-managed-memory)
        - [3. NUMA Memory to Prevent Page Migration](#3-numa-memory-to-prevent-page-migration)
        - [4. Addressing the TLB Thrashing Issue](#4-addressing-the-tlb-thrashing-issue)
    - [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
    - [Host Overhead Optimization](#host-overhead-optimization)
      - [Reduce Binding and Inter-Process Communication Overhead](#reduce-binding-and-inter-process-communication-overhead)
      - [Support Stream Interval](#support-stream-interval)
  - [End-to-End Performance](#end-to-end-performance)
  - [Future Work](#future-work)
    - [Further Performance Optimization](#further-performance-optimization)
  - [Acknowledgements](#acknowledgements)

## Optimization Highlights

Following the introduction of the fundamental design and implementation of large-scale Expert Parallelism (EP) in TensorRT LLM in our [previous blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md), the TensorRT LLM team has focused on optimizing the large EP implementation to improve performance.

At the kernel level, we analyzed kernel duration and optimized performance by either improving existing kernels or developing new kernels that perform better. At the system level, we refined and optimized the EPLB implementation (which also helps reduce kernel scalability issues), integrated additional features such as MTP, and optimized host overhead to prevent Python code from slowing down inference.

### Kernel Optimizations

Our initial kernel breakdown and analysis revealed several key observations about performance impacts when Expert Parallelism (EP) scales up:

1. **MoE GEMM duration decreases** as EP size increases, which is expected behavior.
2. **Attention kernel performance** remains unaffected by increased EP size, demonstrating good scalability.
3. **Communication and some MoE kernels** do not scale well and require optimization.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_kernel_breakdown.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Kernel breakdown when scaling EP without EPLB.</em></sub></p>

We have made improvements to the MoE auxiliary kernels, including `expandInputRowsKernel`, `doActivationKernel`, and `finalizeMoeRoutingKernel`, and to the communication kernels by replacing `AllGather` with a newly developed `AllToAllPrepare` kernel. Additionally, since the `ReduceScatter` and `AlltoAll` kernels do not scale well due to EP imbalance, we optimized the EPLB implementation to improve the scalability of those kernels.

#### MoE Auxiliary Kernels

We observed that given a fixed per-GPU batch size, `expandInputRowsKernel`, `doActivationKernel`, and `finalizeMoeRoutingKernel` showed increased execution time with larger EP size. However, their workload should remain constant regardless of EP size.

Before MoE group GEMMs, `M` tokens are expanded to `M * topK` tokens, which are routed to experts hosted on different ranks. Hence, on average only `M * topK / EP` expanded tokens are valid on each rank (those routed to experts hosted on that rank). The original kernels launch a thread block for each expanded token. Each thread block detects if the token is valid; if so, it proceeds with the computation; otherwise, the thread block exits. For a large EP size, the valid tokens are sparse (`1 / EP`), so most thread blocks are launched for invalid tokens and do nothing, which is wasteful.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_moe_aux_kernels1.png" width="400">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparsity of valid expanded tokens. For DeepSeek-R1 deployed with EP 32, a batch of 12 tokens are expanded to 96 tokens, but only 3 are valid on rank 0.</em></sub></p>

Therefore, we modified the kernels so that thread blocks are launched for valid tokens only. This addressed the scalability issue.

Note that the number of valid tokens is data-dependent. To guarantee CUDA graph compatibility, we cannot rely on any data-dependent information on the host. Thus, we further modified the kernels to use persistent thread blocks, which control the loop based on the valid token number on the device.

This optimization was implemented in [PR 5215](https://github.com/NVIDIA/TensorRT-LLM/pull/5215), with the following performance improvement:

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_moe_aux_kernels2.png">
</figure>
</div>
<p align="center"><sub><em>Figure 3: Optimization effect on MoE auxiliary kernels. (Left) Before optimization, kernel time increases with EP size. (Right) After optimization, kernel time remains constant with EP size.</em></sub></p>

#### Communication Kernels

As introduced in our [previous blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md#ep-communication-kernels-implementation), we developed EP communication kernels to transfer hidden state tensors of MoE. In the original design, each rank needs to determine which tokens it needs to send and receive, along with the expert IDs and scaling factors selected by those tokens. We initially used `allgather` to collect expert IDs and scaling factors, then each rank calculated the required metadata. However, we found that although the transmission size of this data is not large, the performance of `allgather` is unsatisfactory and may become a performance bottleneck when EP size increases. Therefore, we developed new communication kernels to optimize this process.

First, a kernel counts the number of tokens needed to be transferred to another rank and transfers the count to that rank. Then each rank can calculate the index information for subsequent alltoall kernels. Finally, an alltoall kernel transfers expert IDs and scaling factors. These kernels make EP more scalable because the communication size no longer increases with EP size. The implementation of the communication part of these kernels is similar to the previous communication kernel of hidden states, are used in a FIFO manner. But an important difference is that these kernels use release-acquire instructions to ensure memory consistency, which has the advantage of being able to support various forms of data more flexibly. Although it is not as efficient as LL128 primitive in terms of performance, it is more helpful for fast iteration before the functionality converges.

Note that although these kernels achieve better performance compared to `allgather`, there is still considerable room for optimization, especially in latency-bound scenarios.

This optimization was implemented in [PR 5570](https://github.com/NVIDIA/TensorRT-LLM/pull/5570), with the following performance improvement:

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_communication_kernel.png">
</figure>
</div>
<p align="center"><sub><em>Figure 4: Optimization effect on communication kernels.</em></sub></p>

### Expert Parallelism Load Balancer (EPLB)

As introduced in our [previous blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md#ep-load-balancer), EP-level workload imbalance is common for large-scale EP inference across multiple datasets and has significant performance impacts. TensorRT LLM implements a set of functionalities to address this issue. We have refined the code and improved the usability of this feature, and the benefits of EPLB are directly reflected in kernel duration improvements.

The core challenge with EP scaling is that different experts receive varying amounts of work based on the routing decisions made by the MoE layer. This imbalance becomes more pronounced as EP size increases, leading to scenarios where some GPUs are heavily loaded while others remain underutilized. The Expert Parallelism Load Balancer (EPLB) addresses this by dynamically redistributing expert assignments to achieve better load balance across all participating GPUs.

EPLB operates in two main modes:
- **Static EPLB**: Pre-computed expert-to-GPU mappings based on historical data patterns
- **Online EPLB**: Dynamic runtime redistribution that adapts to real-time workload patterns

While Static EPLB provides good baseline improvements, Online EPLB offers the potential for optimal load balancing by responding to actual runtime patterns. However, implementing Online EPLB presented several unexpected technical challenges, particularly around weight synchronization and memory management in GPU clusters.

In the previous [Kernel Optimizations](#kernel-optimizations) section, we noted that `reduce_scatter` and `alltoall` kernels do not show good scalability, with load imbalance being the major root cause. After applying proper EPLB strategy, those kernels perform well even when EP size scales to larger extents.

#### Attempts at Online EPLB Implementation

We discussed the [high-level design](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md#high-level-design-introduction) and [implementation considerations](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md#online-ep-load-balancer) of Online EPLB in our previous blog. However, several unexpected issues arose during implementation.

These issues primarily stem from the weight updating mechanism.

##### 1. Initial Approach for Weight Updating - cudaMemcpyAsync

Our initial approach for weight updating was straightforward. Since GPU kernels from the model forward thread read weights, we placed weights directly in GPU memory using `cudaMalloc` and used a separate non-blocking stream to invoke multiple `cudaMemcpyAsync` calls for weight updates. After implementing the first version of the prototype, we discovered that with CUDA Graph enabled, the model forward thread and the weight updating thread could deadlock.

After investigation, we found the root cause: both `cudaGraphLaunch` and `cudaMemcpyAsync` were competing for the same mutex inside CUDA. In our implementation with layer-wise weight updating, the GPU needs to synchronize with the CPU during model forward passes. This creates kernels that wait for CPU signals indicating that updates are complete and MoE weights are safe to use. These waiting kernels block subsequent kernels.

Since LLM models contain numerous kernels, `cudaGraphLaunch` may need to wait for previous kernels to finish to acquire sufficient resources for launch completion. When waiting kernels are blocked by the CPU, `cudaGraphLaunch` is also blocked. The CPU thread responsible for unblocking this process is the weight update thread, which should signal completion when weight updating finishes. However, since our initial implementation used `cudaMemcpyAsync` for weight updating, it needed to acquire the CUDA mutex before starting memcpy operations. Unfortunately, this mutex was held by `cudaGraphLaunch` in the model forward thread, which was waiting for the weight updating thread to complete. This created a deadlock scenario.

To resolve the deadlock, we needed to break the dependency cycle. While the model forward thread must depend on the weight updating thread for correctness, the weight updating process should not wait for `cudaGraphLaunch` in the model forward thread. Our solution was to use alternative methods instead of `cudaMemcpyAsync` to avoid competing for the same mutex with `cudaGraphLaunch` and other CUDA APIs.

##### 2. Avoiding Deadlock - Multithreaded CPU Copy with Managed Memory

Since weight updating is handled by CPU threads and we wanted to avoid interfering with GPU model forward passes while avoiding mutex contention in `cudaMemcpyAsync`, we chose to use CPU threads for copying operations. To achieve this, we needed MoE weights to be accessible by the CPU while remaining physically located on the GPU to provide high bandwidth for MoE forward passes.

On GB200 systems, the C2C link between CPU and GPU allows CPU access to GPU memory, with GPU memory treated as NUMA nodes. Although the CUDA Driver API doesn't directly support this in CUDA 12.9, one option is to use `cudaMallocManaged` for MoE weights and use `cudaMemAdvise` to set the GPU as the preferred location while enabling CPU access. The CPU copy implementation was straightforward, but we still needed to detect system topology and bind to CPU cores belonging to the same NUMA nodes as the GPU's host NUMA node.

After completing this implementation, CUDA Graph worked well with weight updating and we began seeing end-to-end performance benefits using Online EPLB in some configurations. However, we soon encountered issues with managed memory. Although the preferred location of managed memory was set to GPU, and on GB200 it typically remains on GPU when accessed by CPU, we still observed page migration when GPU memory usage approached capacity limits. The bottom half of the UVM interrupt service process for each GPU consumed 100% of one CPU core's time, causing severe slowdowns when approaching GPU memory limits. To address this, we needed GPU memory that was accessible by CPU without triggering page migration.

##### 3. NUMA Memory to Prevent Page Migration

On GB200 systems, the Grace CPU and Blackwell GPU are connected via C2C links, enabling mutual memory access. GPU memories are also exposed to the OS as NUMA nodes. Running `numactl -H` on GB200 nodes shows output similar to this:

```text
# numactl -H
available: 34 nodes (0-33)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
node 0 size: 489935 MB
node 0 free: 370318 MB
node 1 cpus: 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
node 1 size: 489795 MB
node 1 free: 465004 MB
node 2 cpus:
node 2 size: 188416 MB
node 2 free: 188415 MB
node 3 cpus:
node 3 size: 0 MB
node 3 free: 0 MB
...
node 9 cpus:
node 9 size: 0 MB
node 9 free: 0 MB
node 10 cpus:
node 10 size: 188416 MB
node 10 free: 188416 MB
...
node 18 cpus:
node 18 size: 188416 MB
node 18 free: 188416 MB
...
node 26 cpus:
node 26 size: 188416 MB
node 26 free: 188416 MB
...
node distances:
node   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33
  0:  10  40  80  80  80  80  80  80  80  80  80  80  80  80  80  80  80  80  120  120  120  120  120  120  120  120  120  120  120  120  120  120  120  120
  1:  40  10  120  120  120  120  120  120  120  120  120  120  120  120  120  120  120  120  80  80  80  80  80  80  80  80  80  80  80  80  80  80  80  80
  2:  80  120  10  11  11  11  11  11  11  11  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40
  3:  80  120  11  10  11  11  11  11  11  11  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40
...
  9:  80  120  11  11  11  11  11  11  11  10  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40
...
```

In this configuration, `node 0` and `node 1` are Grace CPU nodes, each with 72 CPU cores and 480GB of memory. `node 2`, `node 10`, `node 18`, and `node 26` represent NVIDIA GB200 GPUs, which have no CPU cores but contain memory. Additional NUMA nodes (3-9, 11-17, 19-25, 27-33) are reserved for MIG instances and show 0 MB memory size. For brevity, we only show `node 3` and `node 9` in the example.

It's possible to allocate system memory on a GPU's NUMA node using `numa_alloc_onnode` (e.g., NUMA node 2 for GPU 0), then register that memory with the GPU using `cudaHostRegister` to make it accessible as host system memory. This allows both CPU and GPU to access the memory, and our testing showed that bandwidth appears nearly identical to normal device memory from the GPU's perspective.

This approach resolved page migration issues, and Online EPLB worked well for large batch sizes per GPU (e.g., 256). However, when investigating smaller batch sizes (32 or 64), we found that MoE GEMM kernel execution time could be higher than without Online EPLB—increasing from 75 µs to 93 µs for the first group GEMM of MoE with EP size 16. Further experiments revealed that when running group GEMM multiple times in the same layer, only the first execution suffered from this slowdown. By adding a warmup kernel that read only one value from 64 KB of weights, we found this simple warmup kernel consumed more than half the execution time of the group GEMM kernel. More interestingly, when running this warmup kernel in parallel with other kernels (using only 14 CTAs), those other kernels also became extremely slow. Based on these observations, we concluded that we were encountering TLB thrashing.

##### 4. Addressing the TLB Thrashing Issue

On GB200 systems, the default page size is 64 KB, which can be verified with:

```text
# getconf PAGE_SIZE
65536
```

The `numa_alloc_onnode` function may use this page size, which is too small for efficient GPU kernel execution. Linux systems support [HugeTLB Pages](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html), and on GB200 systems, the huge page size is 512 MB:

```text
# cat /proc/meminfo
MemTotal:       1774995776 kB
MemFree:        1651165696 kB
MemAvailable:   1671517696 kB
...
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:     524288 kB
Hugetlb:               0 kB
```

By using huge pages, we can significantly reduce the number of required TLB entries and avoid TLB thrashing. Our implementation approach:

- Use `mmap` to allocate address space aligned to 512 MB boundaries
- Use `mbind` to bind the memory to the GPU's NUMA node (e.g., NUMA node 2 for GPU 0)
- Request huge pages using `madvise` with the `MADV_HUGEPAGE` flag
- Register the memory with the GPU using `cudaHostRegister`

This approach provides memory that is located on the GPU, accessible by the host, uses large pages instead of small ones, and doesn't trigger page migration. One consideration is that huge page allocation requires memory allocation at the granularity of one page (512 MB), which could cause significant memory waste with separate allocations. Since our primary use case involves MoE weights that are allocated at model load time and persist throughout the model's lifetime, we implemented a simple memory pool to minimize waste.

Since our implementation relies on huge pages and `madvise`, Transparent Hugepages must be enabled on the system. Without this, you may encounter the exception `madvise(MADV_HUGEPAGE) failed.`. To verify that Transparent Hugepages is properly configured:

```bash
>$ cat /sys/kernel/mm/transparent_hugepage/enabled
always [madvise] never
>$ cat /sys/kernel/mm/transparent_hugepage/defrag
always defer defer+madvise [madvise] never
```

In the output above, the value in square brackets indicates the current setting. If `never` is highlighted instead of `madvise`, you can enable Transparent HugePages with:

```bash
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
```

After implementing huge pages, we found that warmup kernels now execute in only 4 µs without slowing down other kernels. Additionally, group GEMM kernel performance matches that achieved without Online EPLB, both with and without warmup operations. This optimization was implemented in [PR 5963](https://github.com/NVIDIA/TensorRT-LLM/pull/5963), and we achieved additional performance improvements using Online EPLB on the Pareto curve.

### Multi-Token Prediction (MTP)

MTP allows verifying and accepting several draft tokens in a single iteration, which is very beneficial for scenarios that prefer low latency. TensorRT LLM has supported MTP, and we refer to our previous [MTP blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.md#mtp-implementation-in-tensorrt-llm) for more details on the implementation.

For large EP, we have also extended the implementation so that it works well with online EPLB. This was implemented in [PR 5213](https://github.com/NVIDIA/TensorRT-LLM/pull/5213).

### Host Overhead Optimization

Since large-scale EP enables extensive parallelism that includes both expert parallelism and attention data parallelism, the total batch size of one iteration scales with the number of total GPUs involved in the calculation. One outcome is that this significantly increases the number of requests and responses that the system must handle, putting huge pressure on Python threads. The Global Interpreter Lock (GIL) makes the situation worse, since multi-threading won't help under heavy system workloads. When the workload prefers higher throughput, it could even appear that highly optimized CUDA kernels are faster than CPU operation execution, and the GPU could be idle waiting for the CPU to finish the work.

To address the increased host overhead when scaling parallelism in the system, we added optimizations to performance hot spots to reduce single-thread pressure.

#### Reduce Binding and Inter-Process Communication Overhead

TensorRT LLM is designed to be composed of both C++ and Python code, so that C++ can handle the most performance-sensitive parts while Python handles higher-level logic. As we try to put more logic into Python to make the program easier to read and debug, there are still frequent conversations through binding interfaces between C++ and Python. Besides, since most of the logic is implemented in Python, there are several layers of implementation that communicate with each other through inter-process communication overhead. Frequent binding calls and serialization/deserialization introduced by inter-process communication slow down the core library.

To improve program efficiency, we used environment variables introduced in the [performance analysis guidance](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-analysis.md) to measure and profile CPU overhead, and improved performance by reducing and reusing different binding calls as much as possible, and delaying Python object deserialization to avoid duplicated serialization and reduce message size when doing inter-process communication. This optimization was added in [PR 5224](https://github.com/NVIDIA/TensorRT-LLM/pull/5224). We have also reduced Python garbage collection (GC) impacts in [PR 5141](https://github.com/NVIDIA/TensorRT-LLM/pull/5141).

To enable powerful NVTX markers for easier analysis of host overheads, TensorRT LLM provides several useful environment variables:

```bash
export TLLM_NVTX_DEBUG=1 # enables more NVTX markers
export TLLM_PROFILE_RECORD_GC=1 # enables GC collection hint
export TLLM_PROFILE_START_STOP=100-150 # enable specific iterations profiling
```

#### Support Stream Interval

As mentioned previously, one outcome of large-scale workloads is that they significantly increase the number of requests and responses that the system must handle, putting huge pressure on Python threads. When the GPU finishes one iteration of calculation, a batch of responses are generated under streaming mode. For each response, TensorRT LLM must perform detokenization so that output IDs are converted to strings, and OpenAI API protocol objects need to be initialized so that responses can be returned to the user. This becomes time-consuming, especially when the number of responses is huge and the CPU must process them on each iteration. One observation from the user side will be reduced streaming performance when compared to non-streaming.

To address this problem, TensorRT LLM has supported a feature called stream interval. Instead of handling all responses on each iteration, a user-specified `stream_interval` `N` indicates that responses will be handled and returned every `N` iterations. This way, on each iteration, there will still be one output ID generated, but it won't be returned to users immediately (except for the first token for the sake of time-to-first-token latency). Instead, tokens accumulate for `N` iterations, and one response is created to handle those `N` generated tokens, which greatly reduces pressure on the CPU side by giving more time for the CPU to catch up. Meanwhile, users can still get streamed output.

This feature was added in [PR 5284](https://github.com/NVIDIA/TensorRT-LLM/pull/5284), and we have verified that it works effectively to reduce host overhead. In most cases, setting `stream_interval` to 2 or 4 should close the gap (if any) between streaming and non-streaming modes. The feature can be enabled by setting the following in the YAML extra config file:

```yaml
stream_interval: 4
```

## End-to-End Performance

To demonstrate the benefits of large-scale EP, we compared performance on EP16 and EP32 with EP4 and EP8 as baselines, on GB200 NVL72 using DeepSeek R1 FP4 [checkpoints](https://huggingface.co/nvidia/DeepSeek-R1-FP4).

We explored different workloads including 1k-ISL 1k-OSL, 4k-ISL 1k-OSL, and 8k-ISL 1k-OSL. To quickly collect these data points and ensure that generation nodes are saturated, we used the `TLLM_BENCHMARK_REQ_QUEUES_SIZE` environment variable when benchmarking so that the workload can quickly reach a balanced point. The numbers are measured on commit `0cf2f6f154b4a5765d89945b20aa3449b2be7933` with a translation-task dataset, and generated by post-processing the per-iteration log.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_perf-1k-1k-dep.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 5: DeepSeek R1 throughput on ISL/OSL 1k/1k.</em></sub></p>

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_perf-4k-1k-dep.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 6: DeepSeek R1 throughput on ISL/OSL 4k/1k.</em></sub></p>

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_perf-8k-1k-dep.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 7: DeepSeek R1 throughput on ISL/OSL 8k/1k.</em></sub></p>

When enabling MTP, there is an extra performance boost compared to the baseline. We conducted end-to-end experiments and compared to EP4 and EP8 as baselines, seeing up to 6.17x per-GPU output throughput improvement. The numbers are measured with `trtllm-serve` enabling multiple features like large EP, disaggregated serving, EPLB, MTP, and using an OpenAI API client [tool](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py) that sends requests to the server and collects performance metrics.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog8_perf-8k-1k-e2e-mtp.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 8: DeepSeek R1 throughput on ISL/OSL 8k/1k with MTP enabled.</em></sub></p>

To reproduce the numbers, refer to the [`examples/wide_ep/slurm_scripts`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/slurm_scripts) directory. The scripts there demonstrate how to launch TensorRT LLM disaggregated serving with large-scale EP and other features enabled on a SLURM cluster.

## Future Work

### Further Performance Optimization

We are planning to implement more performance optimizations for the large EP implementation, including optimizing the `concat_qkv` operation for the context phase, quantizing `Wo_GEMM` to FP4, supporting low-precision `All2All` operations, and fusing some `All2All` kernels into one. We will also explore integrating more features such as PDL.

## Acknowledgements

This work represents an outstanding example of collaborative engineering excellence within the TensorRT LLM team. The successful implementation and optimization of large-scale Expert Parallelism required coordinated efforts across multiple domains - from low-level CUDA kernel optimizations to high-level system architecture design. The dedication and technical expertise demonstrated by our team members throughout this project has been truly remarkable.

Large-scale Expert Parallelism represents one of the important workloads for users productive scenarios, enabling efficient deployment of large MoE models. The performance improvements achieved through this work demonstrate the transformative potential of expert parallelism at scale, and this work opens new possibilities for deploying increasingly sophisticated AI models in production environments.
