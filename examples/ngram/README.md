# NGram Speculative Decoding

This document shows how to build and run a model using NGram speculative decoding (supported as `ASSISTED_GENERATION` in transformers and vLLM, source: [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/tree/main)) in TensorRT LLM on single GPU, or single node multiple GPU.

## Overview

We provide two styles of workflow to run NGram (named V1 and V2 respectively) now. V1 is in TRT workflow and similar to the Draft-Target-Model workflow, running in orchestrator mode and calling `runner.generate()` multiple times to get outputs, which is more flexible for customizing but slightly more overhead. V2 is in pytorch workflow and similar to the Look-Ahead workflow, running in leader mode and calling `runner.generate()` only one time to get outputs, which provides higher performance but fixed process.

The NGram has 3 additional hyperparameters that you need to specify to control the process of generation:
- `max_draft_len`: the maximum number of tokens provided as draft tokens in one iteration, which is usually from 4 to 10 in common usage (default value: 4). Empirically, the larger the value is, the higher acceptance rate but higher overhead is expected at the same time, so the right balance based on the models and application scenarios needs to be found.
- `max_matching_ngram_size`: the maximum number of tokens extracted from the tail of the input prompt or generated output as a pattern, which is used to search corresponding draft tokens (default value: 2). Empirically, the larger the value is, the more precise context can be matched from the existed sequence, indicating higher acceptance rate, but the higher probability of miss-match and higher overhead appear, which fall back to normal generation (one token per iteration).
- `device_list`: the index list of device(s) to run the model in V1 workflow. The length of it must be the same as the TP size of the draft model engine. For instances, `device_list=[0]` means using tp_size=1 and GPU 0 for the model, `device_list=[4,5,6,7]` means using tp=4 and GPU from 4 to 7 for the model. This parameter is neddless in V2 workflow.

+ For example, the process of getting draft tokens using `max_draft_len=2` and `max_matching_ngram_size=4` with a sentence `prefix=[..., t1, t2, t3, t4]` is like below:

```Python
pattern = prefix[:-2]                               # pattern=[t3, t4] (length=2)
if pattern in pool and len(pool[pattern]) == 4:     # assuming it is {(t3, t4): (t5, t6, t7, t8)}
    return pool[pattern]                            # draft token = [t5, t6, t7, t8]
elif pattern in pool and len(pool[pattern]) == <4:  # assuming it is {(t3, t4): (t9, t10, t11)}
    return pool[pattern]                            # draft token = [t9, t10, t11]
pattern = prefix[:-1]                               # Try shorter pattern if no candidate of length=2 exists, pattern=[t4] (length=1)
if pattern in pool and len(pool[pattern]) == 4:     # The same process as above
    return pool[pattern]
elif pattern in pool and len(pool[pattern]) == <4:
    return pool[pattern]
return None                                         # No any candidate exists
```

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8
  * Paged KV Cache
  * Tensor Parallel

## Usage

### V1 workflow

+ We use an open-source `llama-v2-13B` models in this example.
+ `--use_paged_context_fmha=enable` must be specified since we need KVcache reuse in this approach.
+ `--speculative_decoding_mode=draft_tokens_external` must be specified.
+ `--max_draft_len` must be specified as the length maximum of the draft tokens.
+ `--ngram_config` is corresponding configuration of NGram, we can see its usage in [util.py](../util.py).
  + As an example, `[10,2,[0]]` means `max_draft_len=10`, `max_matching_ngram_size=2`, and device of target model is `GPU0`.
+ `--kv_cache_enable_block_reuse` must be specified for this approach.
+ Only CPP session is supported, so `--use_py_session` must not be specified.
+ `--num_beams` can not be specified as larger than 1 since beam search is not supported in this approach yet.

```bash
# Build engine
python3 examples/models/core/llama/convert_checkpoint.py \
    --model_dir <Path To Llama-v2-13B repo> \
    --output_dir ./ckpt-target \
    --dtype float16

trtllm-build \
    --checkpoint_dir ./ckpt-target \
    --output_dir ./target-engine \
    --gemm_plugin float16 \
    --use_paged_context_fmha enable \
    --speculative_decoding_mode draft_tokens_external \
    --max_draft_len 10 \
    --max_batch_size 4 \
    --max_input_len 3200 \
    --max_seq_len 4800

# Run decoding
python3 examples/run.py \
    --tokenizer_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --ngram_config "[10,2,[0]]" \
    --max_output_len 256 \
    --kv_cache_enable_block_reuse \
    --input_text "How does Draft-Sampling work?"

# Run summarization tasks
python examples/summarize.py \
    --test_hf \
    --test_trt_llm \
    --check_accuracy \
    --hf_model_dir <Path To Llama-v2-7B repo> \
    --engine_dir ./target-engine \
    --batch_size 1 \
    --ngram_config "[10,2,[0]]" \
    --kv_cache_enable_block_reuse
```

### V2 workflow

```bash
python3 examples/llm-api/quickstart_advanced.py \
    --spec_decode_max_draft_len 4 \
    --max_matching_ngram_size 2 \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```
