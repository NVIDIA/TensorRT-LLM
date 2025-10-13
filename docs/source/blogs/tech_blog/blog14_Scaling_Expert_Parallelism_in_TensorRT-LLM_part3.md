# Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary)

This blog post is a continuation of previous posts:
* [Scaling Expert Parallelism in TensorRT LLM (Part 1: Design and Implementation of Large-scale EP)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
* [Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md)

In this blog post, we focus on performance optimization, diving deeper into techniques such as lower precision, network structure refactoring, and aggressive kernel fusion. We hope this analysis and optimization process brings new inspiration to your model inference optimization work.

*By NVIDIA TensorRT LLM Team*

## Table of Contents
- [Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary)](#scaling-expert-parallelism-in-tensorrt-llm-part-3-pushing-the-performance-boundary)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Lower precision](#lower-precision)
    - [wo GEMM FP4 quantization](#wo-gemm-fp4-quantization)
    - [Low precision `AlltoAll`](#low-precision-alltoall)
    - [FP8 context FMHA support](#fp8-context-fmha-support)
  - [Rethink network structure](#rethink-network-structure)
    - [MTP LM head tensor parallelism](#mtp-lm-head-tensor-parallelism)
    - [Context phase Q/K/V `concat` optimization](#context-phase-qkv-concat-optimization)
  - [More kernel overlap, fusion and optimization](#more-kernel-overlap-fusion-and-optimization)
    - [Overlap kernels using programmatic dependent launch (PDL)](#overlap-kernels-using-programmatic-dependent-launch-pdl)
    - [Fuse several `AlltoAll` kernels](#fuse-several-alltoall-kernels)
    - [Fuse `add` (sparse exp and shared exp) into local reduction](#fuse-add-sparse-exp-and-shared-exp-into-local-reduction)
    - [Optimize PyTorch native `copy` and `concat` using `torch.compile`](#optimize-pytorch-native-copy-and-concat-using-torchcompile)
  - [End-to-End Performance](#end-to-end-performance)
  - [Acknowledgements](#acknowledgements)

## Overview

Let's firstly take a look at how the network structure looks like before we did the optimizations, to give an overall review on how the workloads look like:

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_overview_before_opt.png" width="600">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Network structure overview before optimization</em></sub></p>

In this third blog of our scaling Expert Parallelism (EP) series, we push the performance boundaries of large-scale EP on NVIDIA GB200 NVL72 through multiple optimization techniques. Building upon the foundation established in [part 1](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md) and [part 2](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md), this blog explores three key optimization pillars: **lower precision computation** (including FP4 quantization for wo GEMM, low-precision AlltoAll communication, and FP8 context FMHA), **network structure rethinking** (featuring MTP LM head tensor parallelism and context phase Q/K/V concatenation elimination), and **aggressive kernel fusion and overlap** (leveraging Programmatic Dependent Launch, fused AlltoAll operations, and torch.compile optimizations). These optimizations collectively deliver significant end-to-end performance improvements for wide-EP scenarios on NVIDIA GB200 NVL72, for DeepSeek R1 with its specialized Multi-head Latent Attention (MLA) mechanism. Each technique is carefully designed to maintain accuracy while maximizing performance, demonstrating the power of combining algorithmic innovation with deep hardware awareness.

## Lower precision

### wo GEMM FP4 quantization

The wo GEMM is the final linear layer within the multi-head attention block that produces the final outputs. While DeepSeek R1's MLA modifies the initial projections for keys and values, the wo GEMM operator remains a critical and standard component for finalizing the attention computation. In the term, "wo" is the abbreviation for the weight matrix for the output.

We've evaluated that quantizing the wo GEMM to FP4 still satisfies the accuracy requirements, maintaining a similar MTP accept rate (AR) while improving end-to-end performance. The [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) team has published checkpoints that additionally quantize the wo module in attention layers to FP4 on HuggingFace:
* https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2
* https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2

In TensorRT LLM, this is supported by [PR 6393](https://github.com/NVIDIA/TensorRT-LLM/pull/6393). To utilize the checkpoints, simply use the LLM API or `trtllm-serve` to load them. Refer to [deploy-with-tensorrt-llm](https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2#deploy-with-tensorrt-llm) for more details.

### Low precision `AlltoAll`

In wide-EP MoE, the combine phase (after experts finish FC2) performs an all-to-all to return each token’s expert outputs to its origin rank, followed by a per-token reduce over top-k experts.

This step is typically bandwidth-bound when FC2 outputs are in BF16 or FP16. We introduce a low-precision AlltoAll that transmits these combine payloads in NVFP4 instead of BF16/FP16, then dequantizes back on the receiver before the local reduction.

During combine, we temporarily quantize the per-token expert outputs to NVFP4 (e2m1 values with per-16-element E4M3 scale factors plus a global scale) inside shared memory, send the compact representation across GPUs, and dequantize back to the original dtype on the receiving side. Indices and routing-related small tensors remain in their native types.

Since we quantize only for transport and outputs are dequantized back to the working dtype before the per-token reduction, we observe negligible accuracy impact; tolerances comparable to a quant-dequant roundtrip are sufficient. This feature is supported by [PR 7155](https://github.com/NVIDIA/TensorRT-LLM/pull/7155) and [PR 7898](https://github.com/NVIDIA/TensorRT-LLM/pull/7898).

### FP8 context FMHA support

FP8 context FMHA is a technique that uses the FP8 data format to accelerate the FMHA/MLA computation during the context phase of a model. This combination is designed to improve TTFT and prefill throughput, particularly when processing long contexts, without significantly sacrificing accuracy.

In the context phase, the K and V can be stored in FP8 format, which is often referred to as FP8 KV Cache. Using FP8 KV cache can significantly save GPU memory, which is especially beneficial for long input sequences.
However, since Q is in BF16 format, FMHA will also be performed in BF16 format, which cannot benefit from FP8 Tensor Core.

With FP8 context FMHA, we first quantize Q into FP8 format, which aligns with FP8 K and V, and then leverage FP8 Tensor Core for FMHA/MLA. Since the context phase is compute-bound and Tensor Core has much higher FP8 FLOPS than BF16 FLOPS, the speed-up becomes more pronounced as the input sequence length grows.

Since FP8 context FMHA can maintain accuracy very close to the BF16 baseline, we enable it automatically when users use FP8 KV cache on Hopper or Blackwell. This is supported by [PR 7610](https://github.com/NVIDIA/TensorRT-LLM/pull/7610) and [PR 7612](https://github.com/NVIDIA/TensorRT-LLM/pull/7612).

## Rethink network structure

### MTP LM head tensor parallelism

The LM (language modeling) head is responsible for converting the `hidden_states` computed by previous decode layers to `logits`. It's a linear layer with weights in the shape of `(vocab_size, hidden_size)`, outputting logits with the shape of `(batch_size, seqlen, vocab_size)`. We are primarily interested in the logits corresponding to the last token of the input sequence, so the logits will finally be `(batch_size, vocab_size)`.

When MTP is enabled, the number of tokens that MTP layers handle will be equal to the batch size, while the main model will handle `(1 + MTP) * batch_size` tokens, which makes the LM head computation on MTP layers easier to fall into the memory-bound range, and 256 tokens is the empirical boundary between memory-bound and math-bound. This leads to an optimization idea: if we keep the calculation memory-bound but reduce the size of weights that need to be loaded, there could be performance benefits.

Based on this analysis, we conducted experiments on the following scenario: a DeepSeek R1 EP32 case with attention DP and MTP-3 enabled, where the local per-rank batch size is 32. Before the optimization, there is 32-way data parallelism, so each MTP module on each rank processes 32 tokens for LM head calculation.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_MTP_parallel_1.png" width="500">
</figure>
</div>
<p align="center"><sub><em>Figure 2: MTP LM head computation before optimization</em></sub></p>

In the optimization, we first perform an `AllGather` on every 4 GPUs, so that each GB200 node has all tokens prepared for the following TP4 calculation. Then, we split LM head weights on the token dimension to fit those 4 GPUs and perform 4-way TP. Afterwards, we collect the local argmax logits on each TP rank, do a round of `AllGather` to collect that, and find the global argmax logits across all TP ranks. Collecting the local argmax logits firstly helps with minimizing communication and argmax computation overheads. Finally, we split logits to guarantee correctness.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_MTP_parallel_2.png" width="500">
</figure>
</div>
<p align="center"><sub><em>Figure 3: MTP LM head computation after applying tensor parallelism</em></sub></p>

*Some layers are omitted in the diagrams above to keep the example simple.*

Note that we can expand the TP to 8-way to utilize multi-node NVLink, as long as we still achieve performance gains from reducing weight loading time in memory-bound scenarios.

This feature is supported by [PR 7571](https://github.com/NVIDIA/TensorRT-LLM/pull/7571) and [PR 7891](https://github.com/NVIDIA/TensorRT-LLM/pull/7891).

### Context phase Q/K/V `concat` optimization

In the standard attention mechanism, Q/K/V are derived from the same hidden states through `GEMM_Q`/`GEMM_K`/`GEMM_V` operations, and TensorRT LLM typically merges the weights of these three GEMMs in advance, executing a single `GEMM_QKV` to obtain a large contiguous tensor QKV, which is then used as the input to the attention kernels.

However, DeepSeek's MLA is a special attention module where Q/K/V are obtained by applying different downsampling-upsampling processes to the hidden states. Additionally, Q and K are divided into two parts: with RoPE and without RoPE, so a contiguous QKV tensor cannot be obtained directly.

In the initial implementation of context MLA, due to input format constraints of the attention kernels, TensorRT LLM had to explicitly concatenate the Q/K/V tensors into one contiguous QKV tensor, resulting in extra memory and time overhead, which became more significant in wide EP scenarios.

Recently, we introduced a new input format for the context MLA kernels called "separate qkv". As the name implies, these attention kernels now support three separate Q/K/V tensors as direct inputs. [PR 6538](https://github.com/NVIDIA/TensorRT-LLM/pull/6538) refactors the MLA process to eliminate the need for concatenating Q/K/V, saving copy operations and significantly improving prefill latency in wide EP scenarios.

## More kernel overlap, fusion and optimization

The team has implemented aggressive kernel fusion, overlap, and optimization to reduce kernel launch overheads and overall kernel duration. This includes overlapping kernels using PDL, fusing several `AlltoAll` kernels through refactoring, fusing sparse exp and shared exp `add` into local reduction, fusing `memset` into `expandinputrow`, fusing `finalizeMoeRouting` into FC2, and removing the `swizzle` kernel after `AlltoAll`. The following three representative examples demonstrate the common ideas behind these optimizations.

### Overlap kernels using programmatic dependent launch (PDL)

The Programmatic Dependent Launch (PDL) mechanism allows a dependent secondary kernel to launch before the primary kernel it depends on in the same CUDA stream has finished executing. Refer to the [official documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) for more details. TensorRT LLM has been utilizing this feature to optimize end-to-end performance.

We have introduced this feature to the kernels used by the wide EP workflow as well. The implementation is in [PR 7977](https://github.com/NVIDIA/TensorRT-LLM/pull/7977). We inserted the `cudaTriggerProgrammaticLaunchCompletion` API with all thread blocks in the primary kernel, which signals that it's ready for the secondary kernel to launch, and then call the `cudaGridDependencySynchronize` API in the secondary kernel, which blocks until all primary kernels the secondary kernel depends on have completed and flushed results to global memory. The following example from the [official documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#api-description) demonstrates how PDL is supported in TensorRT LLM, the only difference is that we inserted `cudaTriggerProgrammaticLaunchCompletion` and `cudaGridDependencySynchronize` to the same kernel so that it can both overlap with the front and subsequent kernels.
```c
__global__ void primary_kernel() {
   // Initial work that should finish before starting secondary kernel

   // Trigger the secondary kernel
   cudaTriggerProgrammaticLaunchCompletion();

   // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
   // Independent work

   // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
   cudaGridDependencySynchronize();

   // Dependent work
}
```

We have verified the accuracy after the modification to ensure that computation results are not affected by incorrect memory reads and writes. With this premise, we made those kernels overlap as much as possible for performance considerations. In TensorRT LLM, PDL can be enabled by setting the environment variable `TRTLLM_ENABLE_PDL` to `1`, and we may introduce this as an official API in the future.

The effect of enabling PDL can be clearly observed using [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems). Taking `moeComputeRouteKernel`, `computeCountAndIndiceDevice` and `computeCumsumDevice` kernels as an example, they are executed in order when disabling PDL:

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_pdloff.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: The profiling results of disabling PDL.</em></sub></p>

The following profiling results show how the three kernels overlap after enabling PDL.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_pdlon.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 5: The profiling results of enabling PDL.</em></sub></p>

*The above profiles were generated by using commit [84d2f12](https://github.com/NVIDIA/TensorRT-LLM/tree/84d2f1281857fbb1662b14603d3123cf327ac94f) on the main branch. They may change in future versions.*

For tips on using Nsys to profile and analyze TensorRT LLM performance, refer to [Coordinating with NVIDIA Nsight Systems Launch](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-analysis.md#coordinating-with-nvidia-nsight-systems-launch).

### Fuse several `AlltoAll` kernels

To better support communication fusion—including `hiddenStates` during dispatch, low-precision ScalingFactor, MoE's `tokenSelectedExpert` and scales, as well as supporting low-precision communication during dispatch and handling potential non-alignment issues in original data, we redesigned and reimplemented `AlltoAll`.

Taking the dispatch of four fields as an example, the data flow is shown in Figure 6.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_alltoall_dataflow.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 6: The data flow of new Alltoall kernel</em></sub></p>

The sending process is as follows:
- The first step loads the original data according to the data alignment in global memory, using TMA to load into shared memory as `unAlignedData`.
- Next, in shared memory, all fields are aligned to 16-byte boundaries and different fields are concatenated together to form `alignedData`.
- If low-precision communication is needed, the aligned data is quantized into low-precision `lowPrecisionData`. Currently, quantization is only supported for a single field.
- Next, corresponding encoding is performed according to the protocol. For example, with LL128, each 128 bytes contains 120 bytes of valid data and 8 bytes of flags. To avoid bank conflicts during encoding in shared memory, we select different flag positions for different packets, and the final encoded data is stored in `protoPackedData+Flag`.
- Finally, the proto-encoded `protoPackedData+Flag` is written to the remote GPU's workspace.

For the receiver, it only needs to check the flag at the corresponding position in the workspace to confirm whether the data is ready. If ready, the original data is decoded in the reverse manner of sending and written to the corresponding tensors.

Through this approach, we can support sending and receiving multiple arbitrarily aligned fields in a fused manner and support low-precision communication during the combine process. This feature was implemented in [PR 6973](https://github.com/NVIDIA/TensorRT-LLM/pull/6973).

### Fuse `add` (sparse exp and shared exp) into local reduction

To reduce the number of kernel launches and achieve better overlap at the tail of the MoE module, we've fused the shared-expert add into the local reduction kernel that aggregates top-k experts. This removes the extra add operator without increasing the reduce operator's overhead. It also achieves single write-out and lower bandwidth occupancy.

The optimization is compatible with NVFP4 combine without requiring any API changes and brings no accuracy impact. It was added by [PR 7422](https://github.com/NVIDIA/TensorRT-LLM/pull/7422).

### Optimize PyTorch native `copy` and `concat` using `torch.compile`

We have observed several inefficient `copy` and `concat` operations on context phase in wide EP scenarios, and one significant case is copying `k_nope` in the MLA module. As mentioned in previous section, Q and K are divided into two parts in DeepSeek MLA: with RoPE and without RoPE. In context phase, head size of nope will be 128, and that of rope will be 64, which adds up to 192 head size. However, the FMHA kernel will directly read Q and K with head size 192, which means that we have to prepare the full Q and K using `copy` and `concat`.

On ISL/OSL 8k/1k, batch size 1 cases, on context phase, we observed that the `copy` operation takes 306us, which is clearly suboptimal. If we try to calculate a theoretical duration, considering 8 TB/sec HBM3e bandwidth, the formula would roughly be:
```
( ISL 8192 * k_nope_size 128 * num_heads 128 * 2 bytes * read/write 2 ) / ( 8 TB/sec * efficiency 0.8 ) = 80 us
```

To optimize the operator, we simply added `torch.compile` decorator to the operation, and the kernel duration directly drops to 107us, which is greatly reduced and already on a promising level. [PR 8044](https://github.com/NVIDIA/TensorRT-LLM/pull/8044) implemented the changes. This is an outstanding example demonstrating the power of `torch.compile`, and showing the process of analyzing and optimizing without heavily hand-crafting kernels.

## End-to-End Performance

After applying the optimizations above, the network structure is cleaner. For example, `o_proj` and `A2A tokens` now compute in lower precision, and operators like `add` of sparse‑expert and shared‑expert is now fused into the `reduction`. The optimized parts are marked in **bold**.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_overview_after_opt.png" width="600">
</figure>
</div>
<p align="center"><sub><em>Figure 7: Network structure overview after optimization</em></sub></p>

We measured one round of performance and compared it with the baseline (main branch in July). With the optimizations mentioned above, we can see a significant performance improvement.
<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog14_perf.png" width="600">
</figure>
</div>
<p align="center"><sub><em>Figure 8: End-to-End Performance on Aug 31st</em></sub></p>

*Note: The numbers were collected on August 31st. Some optimizations mentioned above were not yet added at that time.*

To review how wide EP helps with Blackwell's leading inference benchmarks, also read these recent blog posts:
* [NVIDIA Blackwell Leads on SemiAnalysis InferenceMAX™ v1 Benchmarks](https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benchmarks/)
* [NVIDIA Blackwell Raises Bar in New InferenceMAX Benchmarks, Delivering Unmatched Performance and Efficiency](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/)

## Acknowledgements
This is a great continuation of previous work on TensorRT-LLM wide EP and another demonstration of excellent teamwork. It stems from brilliant performance optimization ideas, solid performance analysis and benchmarking, and rapid engineering support and implementation. By sharing these experiences, we hope to help more people who are interested in deploying large-scale LLM models on NVIDIA GPUs to run AI faster.
