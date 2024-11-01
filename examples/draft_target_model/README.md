# Draft-Target-Model Speculative Decoding

This document shows how to build and run a model using Draft-Target-Model speculative decoding (also known as `Speculative-Sampling`, [`Paper`](https://arxiv.org/abs/2302.01318)) in TensorRT-LLM on single GPU, or single node multiple GPU.

## Overview

The Draft-Target-Model involves the use of two distinct models trained independently but sharing the same vocabulary: a smaller Draft model and a larger Target model. For example, GPT 125M / 6.7B models can serve as the Draft / Target model.

There are two styles of using Draft-Target-Model in TensorRT-LLM now. The first one is using TensorRT-LLM-BLS in Triton, which more information and detailed steps can be found in [speculative decoding documentation](../../docs/source/speculative_decoding.md). The second one is using it directly in TensorRT-LLM, which steps can be found in this document and the code can be found in [examples/run.py](../run.py).

Draft-Target-Model has 4 additional hyperparameters that you need to specify to control the process of generation:
- `draft_len`: the number of tokens the draft model generated in one iteration, which the range is from 4 to 10 in common usage. Empirically, the larger the value is, the higher acceptance ratio but higher overhead is expected at the same time, so the right balance based on the models and application scenarios needs to be found.
- `draft_model_device_list`: the index list of device(s) to run the draft model. The length of it must be the same as the TP size of the draft model engine. For instances, `draft_model_device_list=[1]` means using tp_size=1 and GPU 1 for draft model, `draft_model_device_list=[4,5,6,7]` means using tp=4 and GPU from 4 to 7 for draft model.
- `target_model_device_list`: the index list of device(s) to run the target model. The length of it must be the same as the TP size of the target model engine. For instances, `draft_model_device_list=[0]` means using tp_size=1 and GPU 0 for target model, `draft_model_device_list=[2,3]` means using tp=2 and GPU from 2 to 3 for target model.
- `use_logits`: there are two methods to accept tokens proposed by draft model. When `use_logits=True`, the draft tokens are accepted based on the ratio of the logits from draft and target model (modified rejection sampling method in the original paper); When `use_logits=False`, the draft tokens are accepted based on per-token comparison with target predictions regardless of the logits.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8 (both draft and target model)
  * Paged KV Cache
  * Tensor Parallel

## Usage

### Build draft and target engines

+ We use a open-source `llama-v2-7B/13B` models as both draft and target model in this example.
+ `--use_paged_context_fmha=enable` must be specified since we need KVcache reuse for draft / target model.
+ `--gather_generation_logits` is optional. In original paper, we accept the tokens by comparing logits of draft and target models, so this parameter is needed. But for simplification, we can accept the tokens by comparing the output token directly, in this occasion, we can skip this parameter.
+ `--speculative_decoding_mode=draft_tokens_external` and `--max_draft_len` must be specified for target model.

```bash
cd examples/llama

python3 convert_checkpoint.py \
    --model_dir=<Path To Llama-v2-7B repo> \
    --output_dir=./ckpt-draft \
    --dtype=float16

python3 convert_checkpoint.py \
    --model_dir=<Path To Llama-v2-13B repo> \
    --output_dir=./ckpt-target \
    --dtype=float16

trtllm-build \
    --checkpoint_dir ./ckpt-draft \
    --output_dir=./draft-engine \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --gather_generation_logits \
    --max_batch_size=4 \
    --max_input_len=3200 \
    --max_seq_len=4800

trtllm-build \
    --checkpoint_dir=./ckpt-target \
    --output_dir=./target-engine \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --gather_generation_logits \
    --speculative_decoding_mode=draft_tokens_external \
    --max_draft_len=10 \
    --max_batch_size=4 \
    --max_input_len=3200 \
    --max_seq_len=4800
```

### Run decoding

+ `--draft_engine_dir` and `--engine_dir` must be specified for the draft and target engines.
+ `--draft_target_model_config` is corresponding configuration of Draft-Target-Model, we can see its definition in [util.py](../util.py).
  + As an example, `[4,[0],[1],False]` means `draft_len=4`, device of draft model is `GPU0`, device of target model is `GPU1`, and use tokens rather than logits to accept.
+ Only CPP session (using executor as low-level API) is supported, while Python session (`--use_py_session`) is not supported.

```bash
cd examples/llama

python3 ../run.py \
    --tokenizer_dir gpt2-medium \
    --draft_engine_dir ./draft-engine \
    --engine_dir ./target-engine \
    --draft_target_model_config="[4,[0],[1],True]" \
    --kv_cache_free_gpu_memory_fraction=0.4 \
    --max_output_len=256 \
    --input_text="How does Draft-Sampling work?"
```
