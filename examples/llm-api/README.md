# LLM API Examples

Please refer to the [official documentation](https://nvidia.github.io/TensorRT-LLM/llm-api/) including [customization](https://nvidia.github.io/TensorRT-LLM/examples/customization.html) for detailed information and usage guidelines regarding the LLM API.

## Measure prefill/decode logprob consistency

```bash
python3 llm_prefill_decode_consistency.py \
    --model /path/to/checkpoint --tp-size 1 \
    --prompt-lengths 127 128 129 511 512 513 --max-tokens 64 \
    --output /tmp/prefill-decode.json
```

The probe generates once and scores the exact prompt and response tokens with
full prefill twice. It normalizes both paths' logits with the same FP32 CPU
`log_softmax`; the API-reported logprobs are retained as a separate normalization
check. It saves token IDs, per-token scores, error percentiles, and tail fractions.
The first response token comes from the initial prefill and is reported separately
from the subsequent decode tokens. Repeated prefill scoring measures prefill
repeatability; this probe does not measure batch invariance or decode determinism.

The baseline uses one request at a time, eager execution, NCCL allreduce, FP32 SSM
state, no prefix reuse, no chunked prefill, and no speculative decoding. Extra LLM
options can be supplied through `--config options.json`; options that break the
controlled comparison are rejected. Cache capacity is based on available memory;
hybrid KV/SSM models can need substantially more cache memory than a token-only
estimate suggests. Check runtime logs for reductions of the requested sequence
limit when overriding the cache budget. Prompt text is encoded without a chat template
and repeated/truncated to the specified lengths. For custom inputs, use `--prompts`
with a JSON list of objects containing `id` and either `prompt` or `prompt_token_ids`.

This is a developer diagnostic, with no assumed numerical tolerance. Once a model
and platform have a measured baseline, `--max-abs-dlogprob VALUE` makes exceeding
that bound on any decode token fail the run. A report whose status is `running`
is incomplete and must not be treated as a passing result. GPU results depend on
the checkpoint, precision, backends, and hardware recorded in the report.

The CPU scoring tests and GPU regressions live in
`tests/unittest/llmapi/test_prefill_decode_consistency*.py` and
`test_prefill_decode_cache_consistency.py`. The GPU tests build a fixed, two-layer
BF16 Llama checkpoint locally. They check scoring alignment and repeated prefill,
then compare fixed continuations across whole/chunked prefill, cold/warm prefix
caches, unrelated intervening requests, and eager/CUDA graph execution. Cache
hits and actual context calls are checked so these paths cannot silently go
untested. Default-path tests use a tolerance specific to that small checkpoint.
On SM100/SM103, deterministic-mode tests select the TRTLLM attention backend and
require exact score equality across chunking and CUDA graphs. The executor
disables prefix reuse under `FORCE_DETERMINISTIC=1`; actual prefix hits are
covered separately by the default-mode tests.

For the supported BF16, TP=1 runtime paths, enable `FORCE_DETERMINISTIC=1` before
starting the engine and select `attn_backend="TRTLLM"`. GDN and Mamba2 also require
`kv_cache_config.mamba_ssm_cache_dtype="float32"`. The mode fixes GEMM reductions,
causal attention, recurrent state updates and BF16 CUTLASS expert tactics across
prefill and decode. See the [deterministic attention documentation](../../docs/source/features/attention.md#deterministic-prefill-and-generation-arithmetic)
for attention shape limits. These computations can substantially slow long
prefills; benchmark the intended workload. This mode does not establish a
whole-model equality guarantee for quantization, MLA or multi-GPU reductions.


## Run the advanced usage example script:

```bash
# FP8 + TP=2
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --tp_size 2

# FP8 (e4m3) kvcache
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --kv_cache_dtype fp8

# BF16 + TP=8
python3 quickstart_advanced.py --model_dir nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 --tp_size 8

# Nemotron Nano hybrid (SSM) models require disabling cache reuse in kv cache
python3 quickstart_advanced.py --model_dir nvidia/NVIDIA-Nemotron-Nano-9B-v2 --disable_kv_cache_reuse --max_batch_size 8
```

## Run the multimodal example script:

```bash
# default inputs
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality image [--use_cuda_graph]

# user inputs
# supported modes:
# (1) N prompt, N media (N requests are in-flight batched)
# (2) 1 prompt, N media
# Note: media should be either image or video. Mixing image and video is not supported.
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality video --prompt "Tell me what you see in the video briefly." "Describe the scene in the video briefly." --media "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4" "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4" --max_tokens 128 [--use_cuda_graph]
```

## Run the speculative decoding script:

```bash
# NGram drafter
python3 quickstart_advanced.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --spec_decode_algo NGRAM \
    --spec_decode_max_draft_len 4 \
    --max_matching_ngram_size 2 \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```

```bash
# Draft Target
python3 quickstart_advanced.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --spec_decode_algo draft_target \
    --spec_decode_max_draft_len 5 \
    --draft_model_dir meta-llama/Llama-3.2-1B-Instruct \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```
