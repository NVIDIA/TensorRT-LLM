# Skip Softmax Attention: A Drop-in Sparse Attention Technique for Accelerating Long-Context Inference

In the previous [tech blog](https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/tech_blog/blog15_Sparse_Attention_in_TensorRT-LLM.md) (TODO: Update link), we introduced the framework to support sparse attention in TensorRT-LLM. The methods we covered, no matter KV cache compression after the context phase, or the sparse tokens prediction in the generation phase, all require some **runtime** modifications. Therefore, they are relatively complex to implement and apply. More importantly, the additional operations compared to the full attention brings computational overhead, which would be detrimental to the performance gain of the core attention computation. Whether those methods are beneficial depends on the specific scenarios, e.g., if the context length is not long enough, enabling those methods may result in negative performance impact. On the other hand, Skip Softmax is only an approximation method of the attention kernel computation, making it compatible with nearly all the other features, such as FP8 attention, KV cache reuse, chunked prefill etc.

In this blog, we introduce **Skip Softmax Attention**, a drop-in sparse attention technique that is designed to accelerate the existing pretrained models that use standard attention mechanisms like MHA, GQA, or MLA. Skip Softmax Attention based on top of the Flash Attention algorithm and only requires modifying the existing **attention kernels**. Due to this simplicity, the end-to-end performance gain is more predictable.

## Method Overview

The idea of Skip Softmax Attention is to compare the local maximum ($\tilde{m}_i^{(j)}$) of $Q \cdot K^T$ with the running global maximum ($m_i^{(j)}$), and skip the softmax (exp) and BMM2 calculation for blocks that are below a certain threshold $\lambda$:
$$
   \tilde{m}_i^{(j)} - m_i^{(j)} < \lambda.
$$
In this way, we can indirectly control the sparsity via the threshold. Note that the threshold is inversely proportional to the context length, i.e., the longer the context, the smaller the threshold is needed to achieve the same sparsity.

The method is fully dynamic, and can be applied to both the prefilling and decoding. The algorithm of Skip Softmax Attention is described in the paper [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/pdf/2512.12087). We have also published a [Developer Blog](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/) for explanation. Please refer to these resources for in-depth dive into the algorithm details. We will focus on the application of Skip Softmax Attention in TensorRT-LLM to accelerate long-context inference.
<img src="../media/tech_blog16_blasst.jpg" alt="BLASST Illustration" style="width: 50%; min-width: 300px; display: block; margin: auto;" />

## Example Usage

Enabling Skip Softmax Attention is pretty simple: we only need to configure the `SkipSoftmaxAttentionConfig` and pass it to the `LLM` API:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig

sparse_attention_config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

# Additionally, the threshold_scale_factor for prefill and decode could be separately configured.
sparse_attention_config = SkipSoftmaxAttentionConfig(threshold_scale_factor={"prefill": 1000.0, "decode": 500.0})

llm = LLM(
   model="Qwen/Qwen3-30B-A3B-Instruct-2507",
   sparse_attention_config=sparse_attention_config,
   # Other LLM arguments...
)
```

The configuration could also be specified through the extra LLM API options YAML file. An example to launch an OpenAI-compatible endpoint is shown below:

```bash
cat >extra_llm_api_options.yaml <<EOF
sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 1000.0

# Additionally, the threshold_scale_factor for prefill and decode could be separately configured.
cat >extra_llm_api_options.yaml <<EOF
sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor:
        prefill: 1000.0
        decode: 500.0
EOF

trtllm-serve Qwen/Qwen3-30B-A3B-Instruct-2507 --extra_llm_api_options extra_llm_api_options.yaml
```

The actual threshold value equals the `threshold_scale_factor` divided by the context length. [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) will support the calibration to automatically determine the required value given the target sparsity. We will use `Qwen3-30B-A3B-Instruct-2507` as the example model for testing, and the calibrated threshold scale factors are listed below:

| Target Sparsity | Threshold Scale Factor (Prefill) | Threshold Scale Factor (Decode) |
|:---------------:|:----------------------------:|:----------------------------:|
| 0.0             | 0.0                          | 0.0                          |
| 0.5             | 85.97                        | 55.48                        |
| 0.6             | 333.81                       | 118.90                       |
| 0.7             | 469.03                       | 254.23                       |
| 0.8             | 718.14                       | 418.73                       |
| 0.9             | 1418.14                      | 863.14                       |

## Accuracy Evaluation
We evaluate the accuracy of Skip Softmax Attention using LongBench V1 and V2. LongBench V1 is a comprehensive benchmark for medium-to-long context understanding, comprising prompts typically at the scale of 10k tokens. LongBench V2 is a harder benchmark that contains longer sequences, and we test up to 256k due to the limit of the native context window of the model. Both versions are integrated into the TensorRT-LLM accuracy test suite, `trtllm-eval`. Here are the example scripts to run the accuracy evaluation:

```bash
# Run LongBench V1 with a single GPU.
cat >extra_llm_api_options.yaml <<EOF
sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 
        prefill: ${thr_prefill}
        decode: ${thr_decode}
EOF
trtllm-eval --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max_batch_size 256 --max_num_tokens 100000 --kv_cache_free_gpu_memory_fraction 0.8 --extra_llm_api_options extra_llm_api_options.yaml longbench_v1
```

```bash
# Run LongBench V2.
# Due to the long sequence length of the prompts and long evaluation time, we use 8 GPUs.
cat >extra_llm_api_options.yaml <<EOF
cuda_graph_config: null
sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 
        prefill: ${thr_prefill}
        decode: ${thr_decode}
moe_config:
    max_num_tokens: 32768  # Chunk MoE execution to avoid OOM.
EOF
trtllm-eval --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
--max_batch_size 1 --max_num_tokens 262144 --tp_size 8 --ep_size 1 \
--kv_cache_free_gpu_memory_fraction 0.8 --extra_llm_api_options extra_llm_api_options.yaml \
longbench_v2 \
--length medium \
--max_output_length 256 --max_input_length 262144
```

The evaluation results are summarized in the table below:

| Target Sparsity | LongBench V1 Overall Accuracy | LongBench V2 Overall Accuracy |
|:---------------:|:----------------------------:|:----------------------------:|
| 0.0             | 47.77                        | 34.42                        |
| 0.5             | 47.43                        | 33.48                        |
| 0.6             | 47.47                       | 33.02                        |
| 0.7             | 47.21                     | 33.02                        |
| 0.8             | 46.50                     | 33.02                        |
| 0.9             | 45.97                       | 33.02                        |

(Note that the number of samples in LongBench V2 is very small (~200), so the result is subject to large variance. You may see non-monotonic situations where higher sparsity results in higher accuracy.)

These results demonstrate that Skip Softmax Attention is safe to use without significant accuracy degradation. 


## Performance Benchmark
Skip Softmax Attention is supported on both Hopper and Blackwell GPUs, based on the SoTA performance of the TensorRT-LLM's attention kernels. Hopper prefilling is implemented in [fmha_v2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/fmha_v2), Hopper decoding is implemented in [XQA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/xqa), and Blackwell is implemented in [trtllm-gen](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/trtllmGenKernels).

We benchmarked the performance of the attention kernels under different achieved sparsity by specifying the threshold.

**B200, seqlen=16k, prefilling:**

| Sparsity % (BF16) | TFLOP/s (BF16) | Speedup (BF16) | Sparsity % (FP8) | TFLOP/s (FP8) | Speedup (FP8) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.00 | 1029.13 | 1.00 | 0.00 | 1523.57 | 1.00 |
| 0.77 | 1016.7 | 0.99 | 0.77 | 1527.17 | 1.00 |
| 8.79 | 1104.53 | 1.07 | 8.81 | 1556.14 | 1.02 |
| 15.51 | 1159.46 | 1.13 | 15.50 | 1587.11 | 1.04 |
| 22.80 | 1180.55 | 1.15 | 22.78 | 1624.8 | 1.07 |
| 29.98 | 1248.99 | 1.21 | 29.96 | 1668.9 | 1.10 |
| 36.82 | 1294.44 | 1.26 | 36.82 | 1714.68 | 1.13 |
| 43.19 | 1314.27 | 1.28 | 43.13 | 1763.96 | 1.16 |
| 49.13 | 1367.18 | 1.33 | 49.05 | 1815.13 | 1.19 |
| 59.72 | 1461.65 | 1.42 | 59.65 | 1925.32 | 1.26 |
| 68.65 | 1536.93 | 1.49 | 68.59 | 2041.63 | 1.34 |
| 76.09 | 1610.21 | 1.56 | 76.01 | 2155.09 | 1.41 |
| 82.20 | 1676.53 | 1.63 | 82.17 | 2266.71 | 1.49 |
| 87.25 | 1753.55 | 1.70 | 87.14 | 2370.11 | 1.56 |
| 94.16 | 1838.36 | 1.79 | 94.12 | 2559.36 | 1.68 |
| 98.46 | 1882.98 | 1.83 | 98.46 | 2731.16 | 1.79 |

**B200, seqlen=64k, prefilling:**

| Sparsity % (BF16) | TFLOP/s (BF16) | Speedup (BF16) | Sparsity % (FP8) | TFLOP/s (FP8) | Speedup (FP8) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.00 | 1038.26 | 1.00 | 0.00 | 1621.41 | 1.00 |
| 0.19 | 1036.03 | 1.00 | 0.19 | 1626.86 | 1.00 |
| 38.87 | 1302.78 | 1.25 | 38.87 | 1861.54 | 1.15 |
| 49.15 | 1376.42 | 1.33 | 49.13 | 1962.05 | 1.21 |
| 56.90 | 1436.79 | 1.38 | 56.87 | 2051.47 | 1.27 |
| 62.96 | 1489.71 | 1.43 | 62.94 | 2131.31 | 1.31 |
| 67.83 | 1535.26 | 1.48 | 67.81 | 2202.96 | 1.36 |
| 71.85 | 1575.72 | 1.52 | 71.83 | 2267.48 | 1.40 |
| 75.26 | 1612.14 | 1.55 | 75.24 | 2326.69 | 1.43 |
| 80.82 | 1673.38 | 1.61 | 80.79 | 2435.71 | 1.50 |
| 85.27 | 1723.23 | 1.66 | 85.24 | 2536.55 | 1.56 |
| 88.91 | 1771.61 | 1.71 | 88.88 | 2632.71 | 1.62 |
| 91.85 | 1811.59 | 1.74 | 91.82 | 2720.56 | 1.68 |
| 94.16 | 1842.88 | 1.77 | 94.13 | 2797.27 | 1.73 |
| 97.21 | 1889.81 | 1.82 | 97.19 | 2919.07 | 1.80 |
| 99.61 | 1925.68 | 1.85 | 99.61 | 3049.29 | 1.88 |


**B200, seqlen=16k, decoding:**

| Sparsity % (BF16) | TB/s (BF16) | Speedup (BF16) | Sparsity % (FP8) | TB/s (FP8) | Speedup (FP8) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.00 | 2.01 | 1.00 | 0.00 | 1.14 | 1.00 |
| 0.00 | 1.95 | 0.97 | 0.00 | 1.08 | 0.95 |
| 0.00 | 1.96 | 0.97 | 0.00 | 1.09 | 0.95 |
| 14.53 | 1.96 | 0.97 | 14.53 | 1.08 | 0.95 |
| 29.05 | 1.97 | 0.98 | 28.72 | 1.09 | 0.96 |
| 45.78 | 1.98 | 0.99 | 46.11 | 1.11 | 0.97 |
| 54.73 | 2.03 | 1.01 | 54.73 | 1.13 | 0.99 |
| 56.42 | 2.05 | 1.02 | 56.42 | 1.13 | 0.99 |

**B200, seqlen=64k, decoding:**

| Sparsity % (BF16) | TB/s (BF16) | Speedup (BF16) | Sparsity % (FP8) | TB/s (FP8) | Speedup (FP8) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.00 | 4.36 | 1.00 | 0.00 | 3.08 | 1.00 |
| 0.00 | 4.27 | 0.98 | 0.00 | 2.79 | 0.91 |
| 3.38 | 4.28 | 0.98 | 3.23 | 2.79 | 0.91 |
| 25.14 | 4.50 | 1.03 | 25.68 | 2.84 | 0.92 |
| 52.70 | 4.86 | 1.12 | 52.61 | 2.85 | 0.93 |
| 72.30 | 4.90 | 1.12 | 72.01 | 2.97 | 0.96 |
| 81.76 | 5.20 | 1.19 | 81.85 | 3.02 | 0.98 |
| 85.09 | 5.24 | 1.20 | 85.04 | 3.03 | 0.98 |

For prefilling, the maximum speedup is ~1.8x. Another advantage of Skip Softmax Attention is that it can further boost performance on top of FP8 attention,.

TODO: Do not list table for kernel performance, use bar chart like FA3 instead. https://www.together.ai/blog/flashattention-3
TODO: Separately list Blackwell and Hopper perf.


We also benchmark the end-to-end performance to demonstrate the benefit of Skip Softmax Attention. Due to the quadratic complexity of the attention, the TTFT in long-context scenarios is often a severe blocker for real-world usage. Skip Softmax Attention can significantly reduce the TTFT by accelerating the prefilling kernel, and the TPOT can also be reduced if the context length is long enough. 

E2E benchmark could be performed using `trtllm-bench`. We can export the LongBench V1 and V2 dataset for performance benchmarking, by adding `--dump_as_text` and `--dump_path` when running `trtllm-eval`. After getting the data of format required by `trtllm-bench`, we can do E2E using the following commands:
```bash
trtllm-bench --model Qwen/Qwen3-30B-A3B-Instruct-2507 --throughput --dataset ${longbench_v1_dataset}  --concurrency 256 --max_batch_size 256 --max_num_tokens 100000 --extra_llm_api_options extra_llm_api_options.yaml --warmup 0 --streaming --report_json longbench_v1_perf.json
```
```bash
trtllm-bench --model Qwen/Qwen3-30B-A3B-Instruct-2507 --throughput --dataset ${longbench_v2_dataset}  --concurrency 1 --tp 8 --ep 1 --max_batch_size 1 --max_num_tokens 262144 --extra_llm_api_options extra_llm_api_options.yaml --warmup 0 --streaming --report_json longbench_v2_perf.json
```

| Target Sparsity | TTFT/ms (Hopper) | TPOT/ms (Hopper) | TTFT/ms (Blackwell) | TPOT/ms (Blackwell) |
|:---------------:|:----:|:----:|:----:|:----:| 
| 0.0             |      |      |      |      |
| 0.5             |      |      |      |      |
| 0.6             |      |      |      |      |
| 0.7             |      |      |      |      |
| 0.8             |      |      |      |      |
| 0.9             |      |      |      |      |

TODO: Fill data.



TODO: Compare with MInference.

## Conclusion
Skip Softmax Attention is a kernel-based solution for accelerating the attention. Due to the design that BMM1 ($Q \cdot K^T$) in the attention kernel is not skipped, the performance gain is capped to 1.8x at kernel level. Nevertheless, it excels at achieving high sparsity with minimal accuracy degradation, and is especially effective in the medium-to-long context (10k-100k) scenarios where previous methods like MInference cannot well handle. The drop-in nature of Skip Softmax Attention makes it a flexible, easy-to-use method for accelerating long-context inference. MLA support for Skip Softmax Attention will be added in the future, and the Skip Softmax Attention kernels will be available in FlashInfer for adoptions by the open-source community.
