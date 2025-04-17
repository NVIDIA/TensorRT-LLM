# Prompt-Lookup Speculative Decoding

This document shows how to build and run a model using Prompt-Lookup speculative decoding (supported as `ASSISTED_GENERATION` in transformers and vLLM, source: [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/tree/main)) in TensorRT-LLM on single GPU, or single node multiple GPU.

## Overview

We provide two styles of workflow to run Prompt-Lookup (named V1 and V2 respectively) now. V1 is similar to the Draft-Target-Model workflow, running in orchestrator mode and calling `runner.generate()` multiple times to get outputs, which is more flexible for customizing but slightly more overhead. V2 is similar to the Look-Ahead workflow, running in leader mode and calling `runner.generate()` only one time to get outputs, which provides higher performance but fixed process.

The Prompt-Lookup has 4 additional hyperparameters that you need to specify to control the process of generation:
- `prompt_lookup_num_tokens`: the number of tokens we extract from input prompt or previous generated output as draft tokens in one iteration, which the range is from 4 to 10 in common usage (default value: 4). Empirically, the larger the value is, the higher acceptance ratio but higher overhead is expected at the same time, so the right balance based on the models and application scenarios needs to be found.
- `max_matching_ngram_size`: the number of tokens we get from the tail of the input prompt or generated output as a pattern, which is used to match in input prompt or previous generated output (default value: 2). Empirically, the larger the value is, the more precise context can be matched from the existed sequence, indicating higher acceptance ratio, but the higher probability of miss-match and higher overhead appear, which fall back to normal generation (one token per iteration).
- `candidate_set_size`: the number of cached matches of a pattern (default value: 2). In default algorithm, the first (the oldest) match in the candidate set is always used as draft tokens, so it's equivalent to use any positive integers greater than 1 for this parameter. Furthermore, `candidate_set_size=1` uses the last (latest) match as draft tokens (the candidate match updates every time when new match appears).
- `device_list`: [optional] the index list of device(s) to run the model in V1 workflow. The length of it must be the same as the TP size of the draft model engine. For instances, `device_list=[0]` means using tp_size=1 and GPU 0 for the model, `device_list=[4,5,6,7]` means using tp=4 and GPU from 4 to 7 for the model. V2 workflow is used if this parameter is not specified.

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

# V1 workflow
python3 ../run.py \
    --tokenizer_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --prompt_lookup_config="[10,2,[0]]" \
    --max_output_len=256 \
    --kv_cache_enable_block_reuse \
    --input_text="How does Draft-Sampling work?"

# V2 workflow
python3 ../run.py \
    --tokenizer_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --prompt_lookup_config="[10,2]" \
    --max_output_len=256 \
    --kv_cache_enable_block_reuse \
    --input_text="How does Draft-Sampling work?"
```

## Run summarization tasks

```bash
cd examples/llama

# V1 workflow
python ../summarize.py \
    --test_hf \
    --test_trt_llm \
    --check_accuracy \
    --hf_model_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --batch_size=1 \
    --prompt_lookup_config="[10,2,[0]]" \
    --kv_cache_enable_block_reuse

# V2 workflow
python ../summarize.py \
    --test_hf \
    --test_trt_llm \
    --check_accuracy \
    --hf_model_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --batch_size=1 \
    --prompt_lookup_config="[10,2]" \
    --kv_cache_enable_block_reuse
```
