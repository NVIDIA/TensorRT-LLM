# README on Skip-Softmax Attention using fmha_v2

## Notice

This a NVIDIA work in progress. You must follow the Non-Disclosure Agreement (NDA) for collaboration to access and use the code. Do not share or distribute the work to any 3rd party (even internal) without permission.

## Setup

We provide a git patch based on TensorRT-LLM tag v1.2.0rc1. Clone TensorRT-LLM and checkout:
```bash
git checkout tags/v1.2.0rc1
```

Apply the patch:
```bash
git apply 9784.patch
```

Then build TensorRT-LLM as usual.

## Introduction

Skip-softmax attention is a technique to perform dynamic sparse attention *within* the attention kernel. It can accelerate the attention kernel without any other runtime modification. It is only controlled by a single threshold parameter, and skip the exp and BMM2 if the following holds:

```c++
   exp(local_max - global_max) < skip_softmax_threshold
```

Larger threshold means better sparsity (performance) but lower accuracy. Typicall threshold should between 0 and 1, and longer context requires lower threshold to achieve the same sparsity.

We provide the patch based on fmha_v2 to work on Hopper, as well as the integration inside TensorRT-LLM.

Kernel perf data, e2e accuracy & TTFT data can be found on【腾讯文档】(Confidential) fmha_v2 Skip Softmax Attention
https://docs.qq.com/sheet/DZUNBU3BuQ2JkUEd1?tab=BB08J2

## Usage

### Enable Skip-Softmax Attention via LLM API

Add the following to your LLM API config YAML:
```yaml
sparse_attention_config:
  threshold: ${thr}
```

Or pass in the `SkipSoftmaxAttentionConfig` object to LLM API's `sparse_attention_config` argument (see eval_longbench_v2.py for example).

### Evaluate Accuracy using LongBench-v2

```bash
cd examples/longbench
```

```bash
git clone https://github.com/NVIDIA/LongBench.git path/to/LongBench
```

Exaluate Qwen3-30B-A3B-Instruct-2507 on single H200:

```bash
python eval_longbench_v2.py \
    --model_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --longbench_path path/to/LongBench/ \
    --attention_backend TRTLLM \
    --output_dir results/Qwen3-30B-A3B-Instruct-2507/medium/SkipSoftmax/thr_${thr} \
    --length medium \
    --skip_softmax_threshold "${thr}"
done
```

Evaluate Qwen3-235B-A22B-Instruct-2507-FP8 on 8x H200 (use with caution: each data point could take 2 hours):
```bash
for thr in 0 0.2 0.4 0.6 0.8 1 2 5; do
  echo "======== Running with threshold ${thr} ========"
  python eval_longbench_v2.py \
    --model_path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --longbench_path path/to/LongBench/ \
    --attention_backend TRTLLM \
    --output_dir results/Qwen3-235B-A22B-Instruct-2507-FP8/medium/SkipSoftmax/thr_${thr} \
    --length medium \
    --skip_softmax_threshold "${thr}" \
    --tensor_parallel_size 8 \
    --kv_cache_fraction 0.5 \
    --max_batch_size 1 \
    --moe_max_num_tokens 33280
done
```