# Combining Guided Decoding and Speculative Decoding: Making CPU and GPU Cooperate Seamlessly

*By NVIDIA TensorRT LLM Team and the XGrammar Team*

## Table of Contents
- [Combining Guided Decoding and Speculative Decoding: Making CPU and GPU Cooperate Seamlessly](#combining-guided-decoding-and-speculative-decoding-making-cpu-and-gpu-cooperate-seamlessly)
  - [Table of Contents](#table-of-contents)
  - [Background and Challenges](#background-and-challenges)
    - [Motivation](#motivation)
    - [Guided Decoding](#guided-decoding)
    - [Speculative Decoding](#speculative-decoding)
    - [Two Challenges](#two-challenges)
  - [Trace Grammar State for Draft Token Proposal and Rejection](#trace-grammar-state-for-draft-token-proposal-and-rejection)
    - [Target Model](#target-model)
    - [Draft Model](#draft-model)
  - [Make Grammar Computation Capturable by CUDA Graph](#make-grammar-computation-capturable-by-cuda-graph)
    - [CUDA Callback](#cuda-callback)
    - [Integration to TensorRT LLM Python Runtime](#integration-to-tensorrt-llm-python-runtime)
    - [CUDA Graph Compatibility: Grammar Computation](#cuda-graph-compatibility-grammar-computation)
    - [CUDA Graph Compatibility: Mask Applying Kernel](#cuda-graph-compatibility-mask-applying-kernel)
    - [Troubleshooting: Data Race between Host and CUDA Callback](#troubleshooting-data-race-between-host-and-cuda-callback)
    - [Troubleshooting: Deadlock by GIL and CUDA Mutex](#troubleshooting-deadlock-by-gil-and-cuda-mutex)
  - [Performance and Analysis](#performance-and-analysis)
  - [Acknowledgements](#acknowledgements)

## Background and Challenges

### Motivation

As part of our effort to bridge gaps in feature combinations, we enabled guided decoding with many important LLM inference features in TensorRT LLM over the last two months:

* Overlap scheduler: [PR 6000](https://github.com/NVIDIA/TensorRT-LLM/pull/6000)
* CUDA graph padding: [PR 6774](https://github.com/NVIDIA/TensorRT-LLM/pull/6774)
* Disaggregated serving: [PR 6704](https://github.com/NVIDIA/TensorRT-LLM/pull/6704)
* Speculative decoding (two-model implementation): [PR 6300](https://github.com/NVIDIA/TensorRT-LLM/pull/6300)
* Speculative decoding (one-model implementation): [PR 6948](https://github.com/NVIDIA/TensorRT-LLM/pull/6948)

More complicated (higher-order) combinations are also supported; for example, we can run DeepSeek-R1 with guided decoding, overlap scheduler, CUDA graph, [attention data parallelism (ADP)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog10_ADP_Balance_Strategy.md), [multiple token prediction (MTP)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.md) and [disaggregated serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md)​ all enabled.

Among all these tasks, combining guided decoding with one-model speculative decoding is the most challenging one, and it achieves the best performance for low-latency or throughput@latency scenarios. This blog post shares the overall design, implementation details, and performance analysis.

### Guided Decoding

Guided decoding (or interchangeably constrained decoding, structured generation) guarantees that the LLM outputs are amenable to a user-specified grammar (e.g., JSON schema), which is particularly useful for LLM agents. For example, guided decoding can help an LLM generate function arguments that strictly conform to function signatures. Thus, the LLM can correctly call external tools and integrate the tool calling results for a better response.

For a request at the prefill phase, guided decoding creates an initial grammar state (i.e., grammar compilation), and generates a mask tensor indicating which tokens from the vocabulary are allowed for the first generated token (i.e., mask gen). At each generation phase, guided decoding advances the grammar state based on the last generated token (i.e., grammar advance), and generates a mask tensor for the next token. The mask will be applied to the logits to mask out the disallowed tokens before sampling (i.e., mask applying), which ensures the next token is amenable to the grammar constraints.

TensorRT LLM integrates third-party grammar backends (e.g., [XGrammar](https://github.com/mlc-ai/xgrammar), [LLGuidance](https://github.com/guidance-ai/llguidance)) for the grammar computation. Currently, these grammar backends are implemented on CPU, so the grammar computation introduces significant CPU overhead. Fortunately, this can be overlapped with the GPU computation, achieving [near-zero overhead](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar). The core idea is that at every iteration, we should first launch the model forward to make the GPU busy, and then compute grammar compilation/advance and mask gen on CPU. Once both the computations finish, the mask can be applied to the logits before sampling.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_constrained_decoding_pipeline_overlap.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Top: guided decoding timeline without overlapping. Bottom: guided decoding timeline with overlapping. (This figure is from the XGrammar paper.)</em></sub></p>

### Speculative Decoding

Speculative decoding is a crucial feature in low-latency or throughput@latency LLM inference scenarios. For each request, a lightweight drafter proposes several draft tokens, and then the target model verifies the draft tokens in parallel. Hopefully, most draft tokens are accepted, and thus multiple tokens are generated in a single target model forward. Compared with normal LLM inference where each model forward generates a single token, speculative decoding offers the potential to generate more tokens per iteration by leveraging more computation. This improves the arithmetic intensity and reduces the required number of iterations.

TensorRT LLM has two kinds of speculative decoding implementations, namely the one-model and two-model implementations. The one-model implementation launches a single CUDA graph for a target model forward together with multiple draft model forwards. This is more difficult to implement and is coupled with the modeling code, but it offers the best performance. The two-model implementation decouples the target and draft models into separate CUDA graphs, which is much more flexible and offers better feature coverage. There are ongoing efforts to close the gaps between the two implementations.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_one_model_vs_two_model.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Top: GPU timeline of one-model speculative decoding. Bottom: GPU timeline of two-model speculative decoding.</em></sub></p>

### Two Challenges

When combining guided decoding and speculative decoding, two challenges arise. First, at each generation iteration, speculative decoding proposes multiple draft tokens, some of which might be rejected in the verification step. The draft token proposal and rejection are not transparent to guided decoding. Specifically, this can be broken down into two views:

* For the target model, guided decoding should advance the grammar state and generate the mask for every draft token. If some draft tokens are rejected, guided decoding should rollback the grammar state to the last accepted token.
* For the draft model, without grammar constraints, some draft tokens may violate the grammar and thus be forcefully rejected in the verification step. Clearly, this hurts the acceptance rate. Hence, guided decoding should also intervene on the logits for every draft token generation if possible.
  * Some speculative algorithms propose draft tokens recurrently by computing logits and sampling (e.g., the standard draft-target model, EAGLE or MTP), similarly to a standard LLM. In that case, guided decoding can apply grammar constraints in a similar mask gen and applying way.
  * Some drafting algorithms work without logits sampling, which require other ways to apply the grammar constraints.

Second, specific to the one-model speculative decoding where a single CUDA graph contains multiple (draft and target) model forwards, the CPU-GPU synchronization becomes challenging. Note that for every step $i$, there are two event waits:

* The host waits for the *token event* that indicates the readiness of CPU tokens from step $i-1$.
* The model forward stream waits for the *mask event* that indicates the readiness of GPU masks from step $i$.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_cpu_gpu_synchronization_for_multiple_steps.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 3: The CPU-GPU synchronization for multiple model forwards.</em></sub></p>

Note that in the two-model implementation, the sampling is excluded from the CUDA graphs for better flexibility (Figure 2). From the CPU perspective, this offers a timing for the grammar computation. In particular, the mask event wait can be inserted between the CUDA graph replay and sampling, effectively making the GPU wait for the GPU masks asynchronously copied from CPU.

However, the CUDA graph of the one-model implementation contains multiple forwards, inevitably including the sampling operations. Hence, there is no timing for the grammar computation. The most outstanding problem is that when replaying the CUDA graph, the mask event wait cannot be inserted before sampling. An alternative is capturing the events and waits in the CUDA graph, but it is still ineffective because the grammar computation is on CPU and thus not capturable. Once such a CUDA graph is launched to replay, the GPU does not wait for any newly recorded events, so it is impossible to block the GPU for the readiness of masks.

## Trace Grammar State for Draft Token Proposal and Rejection

### Target Model

For a target model forward, a request should have one new token and multiple draft tokens from the last verification step and drafter, respectively. For each token in the sequence, guided decoding should advance the grammar state and fill the mask tensor. Before sampling, the masks should be applied to the corresponding logits. After verification, the grammar state should be rolled back by the number of rejected tokens.

Compared to guided decoding with non-speculative decoding, the rollback operation is newly introduced. Thankfully, it has built-in support by grammar backends like [XGrammar](https://github.com/mlc-ai/xgrammar/blob/v0.1.21/python/xgrammar/matcher.py#L341-L350) and [LLGuidance](https://github.com/guidance-ai/llguidance/blob/v1.1.1/python/llguidance/_lib.pyi#L363-L366).

Before proceeding to the draft model view, note that the LLM can generate correct outputs as long as we apply grammar constraints on the target model, because any draft tokens violating the grammar will be forcefully rejected by the verification step. However, this hurts the acceptance rate.

### Draft Model

As aforementioned, we can apply grammar constraints for draft tokens in a similar mask gen and applying way for speculative algorithms based on recurrent logits sampling. Specifically, for the first drafting step, guided decoding advances the grammar state using the last new token. For the following drafting steps, the grammar state is advanced using the last draft token. Each step should fill and apply the mask to the corresponding draft model logits before sampling. 

After the drafting process, the grammar state should be rolled back to the original state, so that the subsequent target model forward can have the correct grammar state. If the draft and target models share the same vocabulary, then the grammar computation is exactly the same so the masks can be reused.

One special case is EAGLE3, whose draft model has a [pruned vocabulary](https://github.com/SafeAILab/EAGLE/blob/58d1de099fe315645a82fe002e46586d54efe405/eagle/traineagle3/config.json#L22-L23) compared to the target model. For instance, LLaMA 3.1 has a 128k vocabulary size, while the corresponding EAGLE3 drafter has a vocabulary containing the most frequent 32k tokens. This saves some computation of the lm_head GEMM. Note that grammar is built on the target model’s vocabulary, so the produced mask cannot be directly applied to the logits of the draft model. EAGLE3 provides a special [d2t](https://github.com/SafeAILab/EAGLE/blob/d7161f9f94aaa345654d9b4045931145811d4d03/eagle/traineagle3/cnets.py#L673-L681) tensor that maps draft token IDs to target token IDs. [PR 7481](https://github.com/NVIDIA/TensorRT-LLM/pull/7481) fuses this d2t mapping to the mask applying kernel.

> **Note:** Here we focus on the chain-based speculative algorithms. A tree-based algorithm would further complicate the implementation; in particular, guided decoding should traverse the drafting tree, advance and rollback grammar states accordingly.

## Make Grammar Computation Capturable by CUDA Graph

### CUDA Callback

CUDA graph can help eliminate the CPU overhead, which is an important technique in the LLM inference systems, especially for the generation phase. As aforementioned, the one-model speculative decoding implementation launches a single CUDA graph to compute multiple draft and target model forwards. This makes the CPU-GPU synchronization challenging: the sampling operation depends on masks computed on CPU, but the GPU is not able to wait for the readiness of any CPU computation once the CUDA graph is launched.

CUDA callback [`cudaLaunchHostFunc`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f) can launch a host function to a CUDA stream. (The host function should not call any CUDA API.) This has two crucial implications:

* CUDA events and event waits can be inserted before and after the host functions, which can be used to synchronize the CPU and GPU computation.
* The host functions can be captured and replayed by CUDA graph.

Hence, we can launch grammar computation along with other auxiliary host functions as CUDA callbacks to a CUDA stream. The CUDA graph should capture and replay multiple model forwards and corresponding grammar computation all together. To achieve CPU-GPU overlapping, the grammar computation should be placed on a dedicated CUDA stream. Specifically, for every step $i$:

* The grammar stream:
  * waits for the *token event* that indicates the readiness of CPU tokens from step $i-1$;
  * performs grammar advance and mask gen (CUDA callback);
  * asynchronously copies the CPU masks to GPU;
  * records the *mask event*.
* The model forward stream:
  * computes model forward using the last GPU tokens;
  * waits for the *mask event* that indicates the readiness of GPU masks;
  * applies the mask to logits and then samples new tokens;
  * asynchronously copies the GPU tokens to CPU;
  * records the *token event*.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_cpu_gpu_synchronization_for_multiple_steps_by_cuda_callback.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: The CPU-GPU synchronization for multiple model forwards by CUDA callback.</em></sub></p>

### Integration to TensorRT LLM Python Runtime

We surveyed some off-the-shelf Python bindings implementations of `cudaLaunchHostFunc`, but it turned out that they do not work well with CUDA graph (e.g., CUDA-Python [Issue 790](https://github.com/NVIDIA/cuda-python/issues/790), cupy [Issue 9274](https://github.com/cupy/cupy/issues/9274)). The probable reason is that the intermediate wrapper data structures are released once the callback is executed; hence, even though the callback is captured by CUDA graph, it cannot be replayed multiple times.

We implement our own bindings to `cudaLaunchHostFunc` — [`launch_hostfunc`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp#L76). Specifically, `launch_hostfunc` packs the Python function and arguments to an [intermediate data structure](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp#L33) and calls `cudaLaunchHostFunc` to launch a [trampoline function](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp#L49) to a CUDA stream. The trampoline function unpacks the intermediate data structure and invokes the Python function with the arguments. Note that `launch_hostfunc` offers great flexibility — it can launch an arbitrary Python function (without any CUDA API calls) as a CUDA callback. Hence, the grammar computation logics can still be implemented in Python.

When CUDA graph is capturing, `launch_hostfunc` does not release the intermediate data structure, so it is accessible during CUDA graph replay. The intermediate data structures can be manually released via [`free_hostfunc_user_data`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp#L97); otherwise, they are automatically cleaned up when the Python interpreter exists. If CUDA graph is disabled (e.g., prefill phase), the intermediate data structure should be released promptly to avoid memory leak. Specifically, the trampoline function automatically releases it once the callback finishes execution.

In Python, we provide a decorator `hostfunc` which casts an arbitrary Python function to a CUDA callback. For example, run the below code snippet:

```python
import torch
from tensorrt_llm._torch.hostfunc import hostfunc

@hostfunc
def increase(x: torch.Tensor):
    x.add_(1)

x = torch.zeros(10, dtype=torch.int32)

stream = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    increase(x)
    increase(x)
torch.cuda.synchronize()

with torch.cuda.stream(stream):
    for _ in range(10):
        g.replay()

torch.cuda.synchronize()
print(x)
```

The output would look like:

```txt
tensor([20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=torch.int32)
```

Note that the CUDA graph increases the tensor twice, and it is replayed for ten times, so the tensor should be totally increased by 20 times. Clearly, the output validates that the CUDA graph capture and replay are successful.

As the final step, we implemented a variant of `GuidedDecoder` — [`CapturableGuidedDecoder`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L405). It reuses most logics from `GuidedDecoder`, but the grammar computation and some auxiliary methods are decorated by `hostfunc`, making it capturable by CUDA graph.

### CUDA Graph Compatibility: Grammar Computation

Once captured, CUDA graph can be launched to run the same GPU kernels as many times as needed. Note that the replayed kernels are always executed using the fixed input and output memory addresses. By filling input buffers with new data, we can run the same work on new data. This pattern also applies to CUDA callback, except that the input and output buffers are on CPU. 

Guided decoder manages the below buffers and resources:

* [Request states](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L20): All the necessary request information affecting grammar computation, including the user-specified grammar, the last new token and draft tokens.
* [Grammar states](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L167-L168): The grammar states managed by grammar backends. By leveraging the grammar backends, guided decoder advances grammar states and fills mask tensors.
* [New tokens tensor](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L419-L422): The tensor values are copied from the newly computed GPU tokens, and used to update the last new token or draft tokens of the request states.
* [Mask tensor](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L175-L177): The tensor values are filled according to the grammar states and then copied to GPU masks, which will be used to apply to logits.

The buffers are stored in fixed memories, and the resources are accessed via fixed pointers. This makes grammar computation compatible with CUDA graph. The buffers and resources are connected via slot IDs. In the runtime, each request is assigned with an exclusive slot ID (0 <= slot ID < `max_batch_size`) upon the first scheduling. The slot ID is occupied until the request is finished and removed from the scheduler.

When the runtime schedules a new batch of requests, the guided decoder updates the request states on the host. After that, all the other operations (grammar compilation/advance, mask gen, buffer copying, etc.) happen on CUDA streams and should be capturable by CUDA graph. More specifically, buffer copying should be asynchronous, and the other CPU computation should be CUDA callbacks.

### CUDA Graph Compatibility: Mask Applying Kernel

The mask applying kernel takes a batch of logits and masks as the input, and modifies the logits in-place. Specifically, the masked-out (disallowed by grammar) token logits are assigned a value of negative infinity, so that they are impossible to be sampled as the next tokens.

Note that currently CUDA graph is enabled for the generation phase only, and the draft length is fixed for all requests. This greatly simplifies the effort for CUDA graph compatibility. Given a `batch_size` and `max_num_draft_tokens`, the logits tensor is of shape `(batch_size * (1 + max_num_draft_tokens), vocab_size)`. Clearly, we can fill the first `(batch_size * (1 + max_num_draft_tokens))` rows of the mask tensor accordingly, and pass the mask tensor address to the kernel.

Some requests may have no grammar constraints. For such requests, we can fill the corresponding masks with all ones (allowed by grammar) so the logits will not be modified by the kernel, but this causes unnecessary computation. To resolve this, a token-level mask tensor is introduced. The tensor values are filled with zeros for requests without grammar constraints. The kernel reads these mask values and skips the rows with mask values being zero.

### Troubleshooting: Data Race between Host and CUDA Callback

Similar to GPU kernels, CUDA callbacks are asynchronously executed on CUDA streams. Note that both normal host functions and CUDA callbacks can access the same CPU memory addresses, so it can easily cause a data race.

In the initial implementation, `CapturableGuidedDecoder` directly reads request states from [`ScheduledRequests`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/scheduler.py#L18). However, the `ScheduledRequests` is shared through an executor iteration and thus probably modified by other executor components. This creates a potential data race scenario:

* Guided decoder launches a CUDA callback, which will read some request states from `ScheduledRequests`;
* Some other executor components inplace modify `ScheduledRequests`;
* The CUDA callback is executed, reading some modified request states from `ScheduledRequests`.

Clearly, the CUDA callback may read unexpected data. This data race motivates a dedicated request states class — [`GuidedRequest`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L20). It is a request snapshot created for guided decoder only, so it will never be modified by other components. It is also possible that the guided decoder itself may access request states via both normal host functions and CUDA callbacks, so we adopt a protocol that the request snapshots should be created on the host, and then accessed only via CUDA callbacks. This prevents potential data race within an executor iteration.

When overlap scheduler is enabled, another data race scenario exists between executor iterations:

* Iteration $i$ launches CUDA callbacks, which will read request states from a fixed address;
* Iteration $i+1$ updates the request states;
* Iteration $i$'s CUDA callbacks are executed, reading request states updated by iteration $i+1$.

Again, the CUDA callbacks may read unexpected data. A straightforward solution is letting the request state update wait for CUDA callback execution, but this effectively disables overlap scheduling. To resolve this issue and also unblock overlap scheduling, a [queue](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/pyexecutor/guided_decoder.py#L417) is introduced. For each iteration, a new batch of request states is put into the queue; then, a CUDA callback is launched to fetch a new batch of request states from the queue, and all the subsequent CUDA callbacks access the newly fetched request states. This allows the co-existence of the request snapshots of two (or even more) iterations, which prevents potential data race between iterations.

### Troubleshooting: Deadlock by GIL and CUDA Mutex

After the first version was implemented, the program intermittently encountered a hang issue when `CapturableGuidedDecoder` is enabled. By checking out the callstack, we found that it was hanging on completely irrelevant kernel launches or some other CUDA API calls. With further investigation, we discovered that the hang issue was caused by a deadlock between the Python GIL and a CUDA mutex.

As documented, a CUDA callback must not make any CUDA API calls. This implies that the CUDA callback execution and CUDA API calls compete for the same mutex. Note that the trampoline function needs to [acquire the GIL](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp#L52) before calling the Python code. Hence, when executing Python code by a CUDA callback, it acquires a CUDA mutex and then the GIL. In the meanwhile, the Python main thread may hold the GIL and make CUDA API calls, so it acquires the GIL and then the CUDA mutex. The two threads acquire the two locks in opposite orders, which creates a deadlock pattern.

This deadlock can be resolved if the Python main thread can release the GIL for CUDA API calls. TensorRT LLM Python runtime is built on PyTorch. Thankfully, PyTorch releases the GIL for most CUDA API calls, even including PyTorch custom operators. However, we find two exceptions in PyTorch 2.8. When creating a device tensor using a shape depending on data from another device tensor, it triggers an implicit and synchronized D2H copy, and this D2H copy is executed with GIL being held ([Issue 163062](https://github.com/pytorch/pytorch/issues/163062)). This can be reproduced by the below code snippet:

```python
import torch

x = torch.randint(0, 100, (100,), dtype=torch.int64, device='cuda')
y = torch.zeros(100, x.max(), dtype=torch.int64, device='cuda')
```

The other case is that `torch.compile` kernels are called with GIL being held ([Issue 163061](https://github.com/pytorch/pytorch/issues/163061)), although Triton kernels are called with GIL released. Hence, we have to avoid any problematic operators and disable `torch.compile` when using CUDA callback to Python code ([PR 7871](https://github.com/NVIDIA/TensorRT-LLM/pull/7871)), until these issues are fixed by PyTorch.

Another source of risk comes from some runtime components that are implemented in C++ and exposed as Python bindings; they may make CUDA API calls as well. By default, Python bindings do not release GIL. Hence, we swept these Python bindings and released GIL properly ([PR 6948](https://github.com/NVIDIA/TensorRT-LLM/pull/6948)).

After all these efforts, the hang issue disappears. It is generally recommended to release the GIL when calling C++ code from Python; even without the context of CUDA callback, this is beneficial for multi-threading performance. However, we acknowledge the limitation that it is difficult to make sure that every such place has been properly handled, and that future code changes do not introduce any risks.

> **Note:** Theoretically, the GIL-free Python ([PEP 703](https://peps.python.org/pep-0703)) could be another remedy.

## Performance and Analysis

We benchmark the performance of guided decoding on two datasets [JSON Mode Eval](https://huggingface.co/datasets/NousResearch/json-mode-eval) and [JSON Schema Bench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench). The models are [LLaMA 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [LLaMA 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), the GPUs are H200 and the grammar backend is XGrammar.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_pareto_curve_json_mode_eval_llama_3.1_8b.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 5: Pareto curve on LLaMA 3.1 8B TP1 on H200, JSON Mode Eval. The concurrency ranges from 1 to 128.</em></sub></p>


<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_pareto_curve_json_mode_eval_llama_3.3_70b.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 6: Pareto curve on LLaMA 3.3 70B TP4 on H200, JSON Mode Eval. The concurrency ranges from 1 to 128.</em></sub></p>

Figures 5 and 6 present the Pareto curves on JSON Mode Eval for LLaMA 3.1 8B and LLaMA 3.3 70B, respectively. Speculative decoding achieves significant speedup for low-latency or throughput@latency scenarios. In particular, the speedup can be up to ~2x for batch size 1. The one-model EAGLE3 implementation is more performant than the two-model EAGLE3, and this performance gap is amplified for small models. This is reasonable, because the one-model implementation captures more workloads into a single CUDA graph, which results in less (if any) exposed CPU overhead.

Note that although NGram is a two-model implementation, it performs surprisingly well. This is because JSON Mode Eval is an information extraction task. Each prompt contains the JSON schema and all the information required by the response, so the NGram has a high acceptance rate on this dataset.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_pareto_curve_json_schema_bench_llama_3.1_8b.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 7: Pareto curve on LLaMA 3.1 8B TP1 on H200, JSON Schema Bench. The concurrency ranges from 1 to 128.</em></sub></p>

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog12_pareto_curve_json_schema_bench_llama_3.3_70b.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 8: Pareto curve on LLaMA 3.3 70B TP4 on H200, JSON Schema Bench. The concurrency ranges from 1 to 128.</em></sub></p>

Figures 7 and 8 show the results on JSON Schema Bench. The one-model EAGLE3 achieves the best performance across almost all scenarios. Note that the NGram becomes less performant since the task is no longer an information extraction task, although the JSON schemas are still present in the prompts.

| Dataset | Model | EAGLE3 | EAGLE3 w/o draft | NGram |
| :-----: | :---: | :----: | :--------------: | :---: |
| JSON Mode Eval    | LLaMA 3.1 8B  | 2.86 | 2.65 | 2.59 |
| JSON Mode Eval    | LLaMA 3.3 70B | 2.72 | 2.60 | 2.44 |
| JSON Schema Bench | LLaMA 3.1 8B  | 2.55 | 2.33 | 1.89 |
| JSON Schema Bench | LLaMA 3.3 70B | 2.50 | 2.30 | 1.87 |

<p align="center"><sub><em>Table 1: Average acceptance lengths per iteration for EAGLE3 and NGram. The acceptance length includes the golden token. The draft length is 3. "EAGLE3 w/o draft" means the draft model does not apply grammar constraints.</em></sub></p>

Table 1 lists the average acceptance lengths per iteration. We perform an ablation experiment where the draft model does not apply grammar constraints. As presented, this does decrease acceptance rates, but by a slighter margin than expected. Note that it introduces extra overheads to apply grammar constraints on the draft model:

* In the drafting loop, the extra mask applying kernels slightly contribute to the GPU time.
* If the drafting forwards are too fast to hide the grammar computation, the exposed CPU time will cause bubbles in the GPU timeline.

These extra overheads could partially offset the benefits from the improved acceptance.

## Acknowledgements

This work demonstrates an outstanding example of cross-team collaboration between the TensorRT LLM and XGrammar teams. We sincerely appreciate the support from everyone who contributed to making this happen.

We acknowledge that it is built on top of the tremendous existing foundations from the community. In particular, some designs were inspired by vLLM [PR 14702](https://github.com/vllm-project/vllm/pull/14702) and SGLang [PR 6499](https://github.com/sgl-project/sglang/pull/6499). In addition, special thanks go to the authors who proposed the speculative algorithms like EAGLE/MTP, and the grammar backend projects like XGrammar/LLGuidance.
