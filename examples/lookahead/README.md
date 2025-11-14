# Lookahead Speculative Decoding

This document shows how to build and run a model using Lookahead speculative decoding ([Break the Sequential Dependency of LLM Inference Using Lookahead Decoidng](https://arxiv.org/pdf/2402.02057)) in TensorRT-LLM.

## Overview

Lookahead decoding algorithm operates through two parallel computation branches within the same LLM - a lookahead branch that generates n-grams using a fixed-sized 2D window, and a verification branch that validates promising n-gram candidates.

Lookahead algorithm is configured with a tuple of `(windows_size, ngram_size, verification_set_size)` or, shortly, `(W, N, G)`.
+ `windows_size` is the Jacobi window size, meaning number of n-grams in lookahead branch that explores future draft tokens.
+ `ngram_size` is the n-gram size, meaning the maximum number of draft tokens accepted per iteration.
+ `verification_set_size` is the maximum number of n-grams considered for verification, meaning the number of draft token beam hypotheses.

You can enable Lookahead decoding for any of decoder-only autoregressive LLM models without any fine-tuning. Some TensorRT LLM models might not work with Lookahead due to the missing head size in the speculative decoding XQA attention kernels. Lookahead performance greatly depends on the base model, hardware, batch size, sequence length, and the dataset. It is recommended to profile various configurations to find the best `(W, N, G)` configuration given the setup.

Specify the Lookahead related flags in three places:

1. *Build the engine*

To build an engine with Lookahead support, specify the `--speculative_decoding_mode lookahead_decoding` and `--max_draft_len` arguments.
For Lookahead, the `max_draft_len` is defined as:
```python
def max_draft_len(windows_size, ngram_size, verification_set_size):
    return (0 if (ngram_size == 1) else ngram_size - 2)
        + (windows_size - 1 + verification_set_size) * (ngram_size - 1)
```

2. *Setup TensorRT LLM runtime*
When TensorRT LLM server starts, the server reserves resources according to the `executor_lookahead_config`. `executor_lookahead_config` is noted as `(W, N, G)`. Ensure the `max_draft_len` derived from `executor_lookahead_config` equals to the `max_draft_len` specified in the engine-building phase -- `--max_draft_len == max_draft_len(W, N, G)`.

3. *Setup the request*
Each request can specify a Lookahead configuration, noted as `(w, n, g)`. If none are specified, the `executor_lookahead_config` is used. The minimum Lookahead config `(1, 1, 0)` forces non speculative, autoregressive mode. The meaningful minimum configuration is `(2, 2, 1)`. Ensure the Lookahead configuration for each request satisfies `w <= W, n <= N, g <= G`.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8
  * Paged KV Cache
  * Inflight-fused-batching
  * C++ runtime
  * Tensor Parallel

## Usage
### Convert Checkpoint

This example is based on the Vicuna-7b v1.3 model, a fine-tuned Llama model.
Checkpoint conversion is similar to any standard autoregressive model, such as the models located in the [examples/models/core/llama](../../examples/models/core/llama) directory.

```bash
MODEL_DIR=/path/to/vicuna-7b-v1.3
ENGINE_DIR=tmp/engine
CKPT_DIR=tmp/engine/ckpt

python3 examples/models/core/llama/convert_checkpoint.py    \
    --model_dir=$MODEL_DIR                      \
    --output_dir=$CKPT_DIR                      \
    --dtype=float16                             \
    --tp_size=1                                 \
    --pp_size=1
```

### Build engine

```bash
trtllm-build                        \
    --checkpoint_dir=$CKPT_DIR      \
    --output_dir=$ENGINE_DIR        \
    --gpt_attention_plugin=float16  \
    --gemm_plugin=float16           \
    --max_batch_size=32             \
    --max_input_len=1024            \
    --max_seq_len=2048              \
    --max_beam_width=1              \
    --log_level=error               \
    --max_draft_len=83              \
    --speculative_decoding_mode=lookahead_decoding
```

### Run decoding

+ `--lookahead_config` is a server-level configuration of Lookahead decoding in the form of a triplet `[W, N, G]`. Note that `run.py` and `summarize.py` interfaces allow to set only per server but not per-request config.

Run `examples/run.py` to generate sequences.
```bash
python examples/run.py          \
    --tokenizer_dir=$MODEL_DIR  \
    --engine_dir=$ENGINE_DIR    \
    --max_output_len=32         \
    --lookahead_config=[7,7,7]  \
    --log_level=verbose         \
    --input_text 'Once upon' 'To be, or not' 'Be not afraid of greatness'
```

Run `examples/summarize.py` to summarize the CNN daily dataset.
```bash
python examples/summarize.py    \
    --test_hf                   \
    --test_trt_llm              \
    --hf_model_dir=$MODEL_DIR   \
    --engine_dir=$ENGINE_DIR    \
    --data_type=fp16            \
    --lookahead_config=[7,7,7]
```
