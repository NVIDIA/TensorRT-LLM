# Prompt-Lookup Speculative Decoding

This document shows how to build and run a model using Prompt-Lookup speculative decoding (supported as `ASSISTED_GENERATION` in transformers and vLLM, source: [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/tree/main)) in TensorRT-LLM on single GPU, or single node multiple GPU.

## Overview

The Prompt-Lookup has 3 additional hyperparameters that you need to specify to control the process of generation:
- `prompt_lookup_num_tokens`: the number of tokens we extract from input prompt or previous generated output as draft tokens in one iteration, which the range is from 4 to 10 in common usage. Empirically, the larger the value is, the higher acceptance ratio but higher overhead is expected at the same time, so the right balance based on the models and application scenarios needs to be found.
- `max_matching_ngram_size`: the number of tokens we get from the tail of the generated output as a pattern, which is used to match in input prompt or previous generated output. Empirically, the larger the value is, the more precise context can be matched from the existed sequence, indicating higher acceptance ratio, but the higher probability of miss-match and higher overhead appear, which fall back to normal generation (one token per iteration).
- `device_list`: the index list of device(s) to run the model. The length of it must be the same as the TP size of the draft model engine. For instances, `device_list=[0]` means using tp_size=1 and GPU 0 for the model, `device_list=[4,5,6,7]` means using tp=4 and GPU from 4 to 7 for the model.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8
  * Paged KV Cache
  * Tensor Parallel

## Usage

### Build engines

+ We use an open-source `llama-v2-13B` models in this example.
+ `--use_paged_context_fmha=enable` must be specified since we need KVcache reuse in this approach.
+ `--speculative_decoding_mode=draft_tokens_external` must be specified.
+ `--max_draft_len` must be specified larger or equal to `prompt_lookup_num_tokens`.

```bash
cd examples/llama

python3 convert_checkpoint.py \
    --model_dir=<Path To Llama-v2-13B repo> \
    --output_dir=./ckpt-target \
    --dtype=float16

trtllm-build \
    --checkpoint_dir=./ckpt-target \
    --output_dir=./target-engine \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --speculative_decoding_mode=draft_tokens_external \
    --max_draft_len=10 \
    --max_batch_size=4 \
    --max_input_len=3200 \
    --max_seq_len=4800
```

### Run decoding

+ `---prompt_lookup_config` is corresponding configuration of Prompt-Lookup, we can see its usage in [util.py](../util.py).
  + As an example, `[10,2,[0]]` means `prompt_lookup_num_tokens=10`, `max_matching_ngram_size=2`, and  device of target model is `GPU0`.
+ `--kv_cache_enable_block_reuse` must be specified for this approach.
+ Only CPP session is supported, so `--use_py_session` must not be specified.
+ `--num_beams` can not be specified as larger than 1 since beam search is not supported in this approach yet.

```bash
cd examples/llama

python3 ../run.py \
    --tokenizer_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --prompt_lookup_config="[10,2,[0]]" \
    --max_output_len=256 \
    --kv_cache_enable_block_reuse \
    --input_text="How does Draft-Sampling work?"
```

## Run summarization tasks

```bash
cd examples/llama

python ../summarize.py \
    --test_hf \
    --test_trt_llm \
    --check_accuracy \
    --hf_model_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --batch_size=1 \
    --prompt_lookup_config="[10,2,[0]]" \
    --kv_cache_enable_block_reuse
```
