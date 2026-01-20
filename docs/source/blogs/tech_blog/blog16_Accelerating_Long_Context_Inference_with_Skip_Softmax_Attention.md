# Accelerating Long-Context Inference with Skip Softmax Attention

In the previous [tech blog](https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/tech_blog/blog15_Sparse_Attention_in_TensorRT-LLM.md) (TODO: Update link), we introduced the framework to support sparse attention in TensorRT-LLM. The methods we covered, no matter KV cache compression after the context phase, or the sparse tokens prediction in the generation phase, all require some **runtime** modifications. Therefore, they are relatively complex to implement and apply. More importantly, the additional operations compared to the full attention brings computational overhead, which would be detrimental to the performance gain of the core attention computation. Whether those methods are beneficial depends on the specific scenarios, e.g., if the context length is not long enough, enabling those methods may result in negative performance impact.

In this blog, we introduce **Skip Softmax Attention**, a drop-in sparse attention technique that is designed to accelerate the existing pretrained models that use standard attention mechanisms like MHA, GQA, or MLA. Skip Softmax Attention based on top of the Flash Attention algorithm and only requires modifying the existing **attention kernels**. Due to this simplicity, the end-to-end performance gain is more predictable. In addition, it is only an approximation method of the attention kernel computation, making it compatible with nearly all the other features, such as FP8 attention, KV cache reuse, chunked prefill etc.

## Method Overview

The idea of Skip Softmax Attention is to compare the local maximum ($\tilde{m}_i^{(j)}$) of $Q \cdot K^T$ with the running global maximum ($m_i^{(j)}$), and skip the softmax (exp) and BMM2 calculation for blocks that are below a certain threshold $\lambda$:
$$
   \tilde{m}_i^{(j)} - m_i^{(j)} < \lambda.
$$
In this way, we can indirectly control the sparsity via the threshold. As described in the [paper](https://arxiv.org/pdf/2512.12087), the threshold is set to be inversely proportional to the context length, i.e., the longer the context, the smaller the threshold is needed to achieve the same sparsity.

The method is fully dynamic, and can be applied to both the prefilling and decoding. The algorithm of Skip Softmax Attention is described in the paper [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/pdf/2512.12087). We have also published a [Developer Blog](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/) for explanation. Please refer to these resources for in-depth dive into the algorithm details. We will focus on the application of Skip Softmax Attention in TensorRT-LLM to accelerate long-context inference.

<p align="center">
  <img src="../media/tech_blog16_blasst.jpg" alt="BLASST Illustration" style="width: 50%; min-width: 300px; display: block; margin: auto;" />
</p>

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
We evaluate the accuracy of Skip Softmax Attention using LongBench V1 and V2. LongBench V1 is a comprehensive benchmark for medium-to-long context understanding, with average sequence length of 10k tokens. LongBench V2 is a harder benchmark that contains longer sequences, and we pick its `medium` subset and truncate the prompt length to 256k due to the limit of the native context window of the model. The average sequence length of LongBench V2 is 130k tokens.

The evaluation results are summarized in the table below:

| Target Sparsity | LongBench V1 Overall Accuracy | LongBench V2 Overall Accuracy |
|:---------------:|:----------------------------:|:----------------------------:|
| 0.0             | 47.77                        | 36.28                        |
| 0.5             | 47.43                        | 38.14                        |
| 0.6             | 47.47                       | 39.53                        |
| 0.7             | 47.21                     | 39.53                        |
| 0.8             | 46.50                     | 37.21                        |
| 0.9             | 45.97                       | 37.21                        |

(Note that the number of samples in LongBench V2 is very small (~200), so the result is subject to large variance. You will see non-monotonic relationship between sparsity and accuracy. We recommend to look at LongBench V1 for inspecting the accuracy loss trend.)

These results demonstrate that Skip Softmax Attention is safe to use without significant accuracy degradation.


## Performance Benchmark
Skip Softmax Attention is supported on both Hopper and Blackwell GPUs, based on the SoTA performance of the TensorRT-LLM's attention kernels. Hopper prefilling is implemented in [fmha_v2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/fmha_v2), Hopper decoding is implemented in [XQA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/xqa), and Blackwell is implemented in [trtllm-gen](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/trtllmGenKernels).

### Kernel Performance

We provide the performance data of the attention kernels under different achieved sparsity by specifying the threshold. The micro-benchmarking is performed under these configs: q_heads=64, kv_heads=4, head_dim=128, seqlen=16k/64k. Both BF16 and FP8 attention are supported. For prefilling, batch size is set to 1; for decoding, batch size is 64.

As a reference, the baseline performance data **without** Skip Softmax Attention are listed below (you can fill in the numbers).

**Prefill Baseline:**

| GPU | Seqlen | Precision | TFLOP/s | Duration µs |
|:---:|:-----:|:---------:|--------:|--------------:|
| H200 | 16k | BF16 | 594.05 | 7403.50 |
| H200 | 16k | FP8  | 852.81 | 5157.12 |
| H200 | 64k | BF16 | 610.30 | 115301.89 |
| H200 | 64k | FP8  | 873.60 | 80550.30 |
| B200 | 16k | BF16 | 1029.13 | 4273.56 |
| B200 | 16k | FP8  | 1523.57 | 2886.67 |
| B200 | 64k | BF16 | 1038.26 | 67775.65 |
| B200 | 64k | FP8  | 1621.41  | 43399.72 |

**Decode Baseline:**

| GPU | Seqlen | Precision | Bandwidth TB/s | Duration µs |
|:---:|:-----:|:---------:|-----------------:|--------------:|
| H200 | 16k | BF16 | 4.391 | 489.06 |
| H200 | 16k | FP8  | 3.158 | 340.01 |
| H200 | 64k | BF16 | 4.410 | 1947.83 |
| H200 | 64k | FP8  | 3.221 | 1333.43 |
| B200 | 16k | BF16 | 7.082 | 303.23 |
| B200 | 16k | FP8  | 5.457 | 196.76 |
| B200 | 64k | BF16 | 7.102 | 1209.51 |
| B200 | 64k | FP8  | 5.683 | 755.76 |

The following figures plot **speedup vs. achieved sparsity**.

**Kernel performance (grouped by GPU):**

<table style="width: 100%; border: 0;">
  <tr>
    <td style="width: 50%; padding: 0 8px; vertical-align: top;">
      <p align="center"><b>Hopper (H200)</b></p>
      <p align="center"><b>Prefill</b></p>
      <img src="../media/tech_blog16_hopper_prefill.png" alt="Hopper prefill speedup vs sparsity" style="width: 100%; min-width: 280px; display: block; margin: auto;" />
      <p align="center"><b>Decode</b></p>
      <img src="../media/tech_blog16_hopper_decode.png" alt="Hopper decode speedup vs sparsity" style="width: 100%; min-width: 280px; display: block; margin: auto;" />
    </td>
    <td style="width: 50%; padding: 0 8px; vertical-align: top;">
      <p align="center"><b>Blackwell (B200)</b></p>
      <p align="center"><b>Prefill</b></p>
      <img src="../media/tech_blog16_blackwell_prefill.png" alt="Blackwell prefill speedup vs sparsity" style="width: 100%; min-width: 280px; display: block; margin: auto;" />
      <p align="center"><b>Decode</b></p>
      <img src="../media/tech_blog16_blackwell_decode.png" alt="Blackwell decode speedup vs sparsity" style="width: 100%; min-width: 280px; display: block; margin: auto;" />
    </td>
  </tr>
</table>

For prefilling, the maximum speedup is ~1.8x. Another advantage of Skip Softmax Attention is that it can further boost performance on top of FP8 attention,.


### End-to-end Performance

We also benchmark the end-to-end performance to demonstrate the benefit of Skip Softmax Attention. Due to the quadratic complexity of the attention, the TTFT in long-context scenarios is often a severe blocker for real-world usage. Skip Softmax Attention can significantly reduce the TTFT by accelerating the prefilling kernel, and the TPOT can also be reduced if the context length is long enough.  

E2E benchmark could be performed using `trtllm-bench` (see the reproduction section for the exact commands).
LongBench V1
avg ISL = 10k

| Target Sparsity | TTFT/ms (H200) | TPOT/ms (H200) | TTFT/ms (B200) | TPOT/ms (B200) |
|:--------------:|------------------:|-----------------:|--------------------:|--------------------:|
| 0.0            | 9419.61           | 1731.80          | 4997.07             | 955.49              |
| 0.5            | 9321.41           | 1712.62          | 4701.96             | 899.31              |
| 0.6            | 9226.75           | 1701.59          | 4680.72             | 895.33              |
| 0.7            | 9065.09           | 1672.45          | 4634.84             | 889.68              |
| 0.8            | 8778.14           | 1622.27          | 4531.42             | 870.22              |
| 0.9            | 8618.86           | 1596.62          | 4475.78             | 861.18              |


LongBench V2
avg ISL = 130k

| Target Sparsity | TTFT/ms (H200) | TPOT/ms (H200) | TTFT/ms (B200) | TPOT/ms (B200) |
|:--------------:|------------------:|-----------------:|--------------------:|--------------------:|
| 0.0            | 16277.58          | 9.32             | 7370.71             | 6.34                |
| 0.5            | 15487.28          | 8.57             | 6655.98             | 6.30                |
| 0.6            | 15020.24          | 8.57             | 6431.65             | 6.25                |
| 0.7            | 14921.12          | 8.42             | 6355.43             | 6.24                |
| 0.8            | 14465.74          | 8.41             | 6192.77             | 6.26                |
| 0.9            | 13791.37          | 8.40             | 6043.06             | 6.27                |


## Reproduction

### Accuracy evaluation (LongBench V1/V2)
> Please manually `pip install lm_eval==0.4.9.2` before reproducing the results. There is a known bug in the current version of `lm_eval` in `requirements-dev.txt`.

Both LongBench V1 and V2 are integrated into the TensorRT-LLM accuracy test suite, `trtllm-eval`. Here are the example scripts to run the accuracy evaluation:

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

### End-to-end performance (TTFT/TPOT)
We can export the LongBench V1 and V2 dataset for performance benchmarking by adding `--dump_as_text` and `--dump_path` when running `trtllm-eval`. After getting the data of format required by `trtllm-bench`, we can do E2E using:

```bash
trtllm-bench --model Qwen/Qwen3-30B-A3B-Instruct-2507 --throughput --dataset ${longbench_v1_dataset} --concurrency 256 --max_batch_size 256 --max_num_tokens 100000 --extra_llm_api_options extra_llm_api_options.yaml --warmup 0 --streaming --report_json longbench_v1_perf.json
```

```bash
trtllm-bench --model Qwen/Qwen3-30B-A3B-Instruct-2507 --throughput --dataset ${longbench_v2_dataset} --concurrency 1 --tp 8 --ep 1 --max_batch_size 1 --max_num_tokens 262144 --extra_llm_api_options extra_llm_api_options.yaml --warmup 0 --streaming --report_json longbench_v2_perf.json
```


TODO: Compare with MInference.

## Conclusion
Skip Softmax Attention is a kernel-based solution for accelerating the attention. Due to the design that BMM1 ($Q \cdot K^T$) in the attention kernel is not skipped, the performance gain is capped to 1.8x at kernel level. Nevertheless, it excels at achieving high sparsity with minimal accuracy degradation, and is especially effective in the medium-to-long context (10k-100k) scenarios where previous methods like MInference cannot well handle. The drop-in nature of Skip Softmax Attention makes it a flexible, easy-to-use method for accelerating long-context inference. MLA support for Skip Softmax Attention will be added in the future, and the Skip Softmax Attention kernels will be available in FlashInfer for adoptions by the open-source community.
