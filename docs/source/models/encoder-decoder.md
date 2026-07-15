<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use encoder-decoder models with the PyTorch backend

TensorRT LLM can run supported Hugging Face encoder-decoder checkpoints directly
with the PyTorch backend. You do not need to convert the checkpoint or build a
TensorRT engine. The LLM API treats the supplied prompt as the encoder input and
automatically starts the decoder with the checkpoint's
`decoder_start_token_id` or `bos_token_id`.

This guide covers text-to-text generation with the following Hugging Face
architectures:

| Hugging Face architecture | Model families and examples |
| --- | --- |
| `T5ForConditionalGeneration` | T5, Flan-T5, and ByT5, for example `google/flan-t5-small` |
| `BartForConditionalGeneration` | BART checkpoints |
| `MBartForConditionalGeneration` | mBART checkpoints |

Whisper is not currently supported by the PyTorch encoder-decoder path. Support
is in progress.

This is the PyTorch LLM path, not the legacy TensorRT encoder-decoder workflow
under `examples/models/core/enc_dec/`.

mBART architecture loading is available. When a BART or mBART checkpoint
defines `forced_bos_token_id`, the PyTorch backend seeds that token in the
decoder prefix. Source- and target-language selection remains
checkpoint-specific, so validate the configured language tokens before
deployment. Refer to [Understand BART and mBART decoder tokens](#understand-bart-and-mbart-decoder-tokens)
for BOS, EOS, and output-limit behavior.

## Feature support

The following table describes the supported and recommended configurations.

| Feature | Support | Notes |
| --- | --- | --- |
| KV cache manager V1 | Yes; recommended | This is the default. It supports greedy decoding, beam search, batching, the overlap scheduler, decoder CUDA graphs, and tensor parallelism. |
| KV cache manager V2 | Yes | Set `use_kv_cache_manager_v2=True`. It currently requires `max_beam_width=1`, so use greedy or sampling with a single sequence rather than beam search. |
| Greedy decoding | Yes | Set `temperature=0.0`. |
| Beam search | Yes with V1 | Configure `max_beam_width` when constructing `LLM`, then set `use_beam_search=True` in `SamplingParams`. |
| Attention backend | `TRTLLM` | Use this backend for encoder-decoder models. It is required when `tensor_parallel_size > 1`. |
| Decoder CUDA graphs | Yes | `CudaGraphConfig` captures decoder work. V1 supports greedy and beam search; V2 supports its single-beam path. |
| Encoder CUDA graphs | No | `EncodeCudaGraphConfig` is disabled for encoder-decoder models. The encoder runs eagerly. |
| Overlap scheduler | Yes | Enabled by default. V1 supports greedy decoding and beam search; V2 remains limited to `max_beam_width=1`. |
| Tensor parallelism | Yes | Use `tensor_parallel_size > 1` with `attn_backend="TRTLLM"`. Attention head counts must be divisible by the TP size. |
| Pipeline parallelism | No | Keep `pipeline_parallel_size=1`. |
| Context parallelism | No | Keep `context_parallel_size=1`. |
| Attention data parallelism | No | Keep `enable_attention_dp=False`. |
| Chunked prefill | Not supported for the encoder phase | Set `enable_chunked_prefill=False`. The complete encoder input must fit in the iteration token budget. |
| Piecewise CUDA graph | No | Do not set `torch_compile_config.enable_piecewise_cuda_graph=True`. |

BF16 is the recommended model dtype. Validate accuracy with your checkpoint and
task before deploying a different precision or quantization configuration.

## Choose the attention backend

Use `attn_backend="TRTLLM"` for encoder-decoder models. T5 self-attention needs
this backend to apply relative attention bias, and tensor parallel
encoder-decoder execution explicitly requires it.

The `TRTLLM` backend can internally select optimized kernels when the hardware
and request are eligible. For example, compatible operations on Blackwell can
use FlashInfer TRTLLM-Gen kernels. This internal selection is different from
setting `attn_backend="FLASHINFER"`. Cases such as T5 relative attention bias or
beam-expanded self-attention can fall back to another kernel within the
`TRTLLM` backend; no customer-side backend change is needed.

## Run basic generation

Install TensorRT LLM using the [installation guide](../installation/installation-guide.md)
and make sure the checkpoint is accessible either from the Hugging Face Hub or
from a local directory.

The following example uses KV cache manager V1, greedy decoding, and the overlap
scheduler:

```python
from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams, SchedulerConfig


model = "google/flan-t5-small"

with LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    dtype="bfloat16",
    disable_overlap_scheduler=False,
    enable_chunked_prefill=False,
    max_batch_size=4,
    max_input_len=512,
    max_num_tokens=2048,
    max_seq_len=512,
    kv_cache_config=KvCacheConfig(
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
        use_kv_cache_manager_v2=False,
    ),
    scheduler_config=SchedulerConfig(use_python_scheduler=True),
) as llm:
    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
    )
    result = llm.generate(
        "translate English to German: The house is wonderful.",
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(result.outputs[0].text)
```

Use the task format expected by the checkpoint. For example, T5 translation
checkpoints commonly expect a task prefix such as `translate English to
German:`, while a summarization checkpoint expects the source document.

The LLM API performs these encoder-decoder-specific steps automatically:

1. Tokenizes the supplied string as the encoder input.
2. Runs the encoder once and retains its output for cross-attention.
3. Initializes the decoder from the checkpoint's decoder start token.
4. Generates decoder tokens and returns the detokenized decoder output.

Do not prepend a decoder start token to the source prompt. If you pass token IDs
instead of text, pass only the encoder-side token IDs:

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(model)
source_text = "translate English to German: The house is wonderful."
source_token_ids = tokenizer.encode(source_text, add_special_tokens=True)
result = llm.generate(source_token_ids, sampling_params=sampling_params)
```

### Configure an mBART tokenizer

mBART tokenization depends on the source language. Create the Hugging Face
tokenizer with `src_lang` and pass that tokenizer to `LLM` so string prompts
receive the correct source-language token:

```python
from transformers import AutoTokenizer

from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams


model = "/path/to/mbart-large-50-many-to-one-mmt"
tokenizer = AutoTokenizer.from_pretrained(model, src_lang="ro_RO")

with LLM(
    model=model,
    tokenizer=tokenizer,
    backend="pytorch",
    attn_backend="TRTLLM",
    dtype="bfloat16",
    enable_chunked_prefill=False,
    kv_cache_config=KvCacheConfig(cross_kv_cache_fraction=0.5),
) as llm:
    result = llm.generate(
        "Şeful ONU spune că nu există o soluţie militară în Siria.",
        sampling_params=SamplingParams(max_tokens=64, temperature=0.0),
        use_tqdm=False,
    )
    print(result.outputs[0].text)
```

For this many-to-one checkpoint, `generation_config.json` selects English with
the `en_XX` forced BOS token. For other mBART checkpoints, confirm that
`decoder_start_token_id`, `forced_bos_token_id`, `eos_token_id`, and the
tokenizer language settings select the source and target languages you intend
to serve.

### Understand BART and mBART decoder tokens

When a BART or mBART checkpoint defines `forced_bos_token_id`, the PyTorch
backend initializes the decoder with the following internal prefix:

```text
[decoder_start_token_id, forced_bos_token_id]
```

For example, BART-large-CNN uses `[2, 0]`. Customers provide only the encoder
input; do not prepend either decoder token. By default, the returned token IDs
exclude `decoder_start_token_id` but include `forced_bos_token_id`, so the
BART-large-CNN output begins with token ID 0.

The forced BOS token counts against `SamplingParams.max_tokens`. Consequently,
a request using this prefix requires `max_tokens` to be at least 2. The runtime
uses the remaining token budget for model-selected tokens. This behavior is the
same for greedy decoding and beam search and does not require a customer logits
processor.

EOS is a stopping token rather than a forced final token. The runtime uses
`SamplingParams.end_id`, which defaults to the tokenizer's `eos_token_id`. If
the model generates EOS before the output limit, the returned token IDs include
EOS and `finish_reason` is `"stop"`. Set `ignore_eos=True` to continue decoding
past EOS.

The runtime does not inject `forced_eos_token_id` when a sequence reaches
`max_tokens`. It preserves the model-selected final token and reports
`finish_reason="length"`.

## Run a batch

Pass a list of strings to batch inputs. The strings can have different tokenized
lengths:

```python
sources = [
    "translate English to German: The house is wonderful.",
    "translate English to German: The book is on the table.",
]

results = llm.generate(sources, sampling_params=sampling_params, use_tqdm=False)
for source, result in zip(sources, results):
    print(f"source={source!r} output={result.outputs[0].text!r}")
```

`max_num_tokens` must cover the encoder tokens admitted in an iteration as well
as decoder work. Increase it for larger batches or longer source sequences.

## Choose KV cache manager V1 or V2

Encoder-decoder execution uses two KV cache pools:

- The self-attention pool stores decoder-side KV states.
- The cross-attention pool stores encoder-derived K/V states used by every
  decoder layer.

`cross_kv_cache_fraction` is required for every encoder-decoder model. It divides
the configured KV cache memory budget between the two pools. A value of `0.5`
is a reasonable starting point:

```python
kv_cache_config = KvCacheConfig(
    free_gpu_memory_fraction=0.8,
    cross_kv_cache_fraction=0.5,
    use_kv_cache_manager_v2=False,
)
```

Increase `cross_kv_cache_fraction` when long encoder inputs exhaust the cross
pool. Decrease it when long decoder outputs or wide beams exhaust the
self-attention pool. The two fractions are related as follows:

```text
cross-attention pool = total KV cache budget * cross_kv_cache_fraction
self-attention pool  = total KV cache budget * (1 - cross_kv_cache_fraction)
```

V1 is the default and should be the first choice for production deployments.
To evaluate V2, change only the manager selection and keep beam width equal to
one:

```python
kv_cache_config = KvCacheConfig(
    free_gpu_memory_fraction=0.8,
    cross_kv_cache_fraction=0.5,
    use_kv_cache_manager_v2=True,
)

llm = LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    max_beam_width=1,
    kv_cache_config=kv_cache_config,
)
```

KV cache manager V2 is a prototype feature and rejects configurations with a
maximum beam width greater than one.

## Use beam search

Beam search requires KV cache manager V1. The maximum beam width is a runtime
capacity setting and must be specified when constructing `LLM`:

```python
beam_width = 4

with LLM(
    model="/path/to/bart-large-cnn",
    backend="pytorch",
    attn_backend="TRTLLM",
    dtype="bfloat16",
    max_beam_width=beam_width,
    enable_chunked_prefill=False,
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
        use_kv_cache_manager_v2=False,
    ),
) as llm:
    beam_params = SamplingParams(
        best_of=beam_width,
        max_tokens=64,
        n=beam_width,
        temperature=0.0,
        use_beam_search=True,
    )
    result = llm.generate(
        "The engineering team released a faster inference service on Monday. "
        "The update improves batching, lowers latency, and adds detailed "
        "monitoring for operators.",
        sampling_params=beam_params,
        use_tqdm=False,
    )

    for hypothesis in result.outputs:
        print(hypothesis.text)
```

`best_of` sets the beam width and must not exceed `LLM.max_beam_width`. `n`
sets the number of returned hypotheses and must not exceed `best_of`. Set `n=1`
to return only the best hypothesis.

Beam search expands decoder-side cache and compute requirements. Include this
expansion when sizing the self-attention KV pool and CUDA graph batch sizes.

## Enable decoder CUDA graphs

Pass `CudaGraphConfig` to capture and replay decoder iterations:

```python
from tensorrt_llm.llmapi import CudaGraphConfig


llm = LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    max_batch_size=8,
    cuda_graph_config=CudaGraphConfig(
        max_batch_size=8,
        enable_padding=True,
    ),
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
    ),
)
```

This configuration captures decoder work only; the encoder continues to run
eagerly. With beam search, graph batch sizes must cover the active decoder
sequences after beam expansion. Padding lets nearby runtime batch sizes reuse a
captured graph.

Do not use `EncodeCudaGraphConfig` for an encoder-decoder model. The runtime
warns and disables it. Piecewise CUDA graphs through `TorchCompileConfig` are
also unsupported for this model type.

## Control the overlap scheduler

The PyTorch backend enables the overlap scheduler by default. The examples set
`disable_overlap_scheduler=False` explicitly to make that choice visible:

```python
llm = LLM(
    model=model,
    backend="pytorch",
    disable_overlap_scheduler=False,
    kv_cache_config=KvCacheConfig(cross_kv_cache_fraction=0.5),
)
```

Overlap is not restricted to KV cache manager V1. Both V1 and V2 enter the same
overlap executor loop, and that loop contains V2-specific resource handling.
V2 remains limited to `max_beam_width=1`. Set
`disable_overlap_scheduler=True` when debugging.

## Use tensor parallelism

Set `tensor_parallel_size` to the number of GPUs over which to shard the model:

```python
with LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    context_parallel_size=1,
    enable_attention_dp=False,
    enable_chunked_prefill=False,
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
        use_kv_cache_manager_v2=False,
    ),
) as llm:
    result = llm.generate(source_text, sampling_params=sampling_params)
```

For single-node execution through the LLM API, do not add an `mpirun` prefix.
TensorRT LLM starts the worker processes. The selected TP size must divide the
encoder and decoder attention head counts. Cross-attention KV head duplication
is not supported, so its KV head count must also be divisible by the TP size.

Tensor parallelism currently requires `attn_backend="TRTLLM"`. Pipeline
parallelism, context parallelism, and attention DP are rejected for
encoder-decoder models.

## Serve an encoder-decoder model

The following configuration starts a greedy Flan-T5 service with the PyTorch
backend. Save it as `enc-dec-config.yaml`:

```yaml
attn_backend: TRTLLM
dtype: bfloat16
disable_overlap_scheduler: false
enable_chunked_prefill: false
max_beam_width: 1
max_input_len: 512
max_num_tokens: 2048
max_seq_len: 512
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
  cross_kv_cache_fraction: 0.5
  use_kv_cache_manager_v2: false
scheduler_config:
  use_python_scheduler: true
```

Start the server:

```bash
trtllm-serve google/flan-t5-small \
    --backend pytorch \
    --max_batch_size 4 \
    --config enc-dec-config.yaml
```

Send the source text through the completions endpoint:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "google/flan-t5-small",
      "prompt": "translate English to German: The house is wonderful.",
      "max_tokens": 64,
      "temperature": 0.0
    }'
```

To serve beam search, use KV cache manager V1, restart the server with
`max_beam_width` set to the desired maximum, and add the following request
fields:

```json
{
  "use_beam_search": true,
  "best_of": 4,
  "n": 1
}
```

For tensor parallel serving, add `--tp_size <N>` to `trtllm-serve` and keep the
attention backend set to `TRTLLM`.

## Size the runtime

Use these guidelines as a starting point:

- Set `max_input_len` to at least the maximum tokenized encoder input length.
- Set `max_seq_len` to at least the larger of the maximum encoder input length
  and maximum decoded sequence length. The current encoder-decoder runtime uses
  this value while sizing both phases.
- Set `max_num_tokens` high enough for all encoder tokens admitted together and
  for the active decoder tokens. This is especially important for mixed-length
  batches.
- Increase `max_batch_size` for more concurrent requests. Beam width multiplies
  the number of active decoder sequences but not the number of source requests.
- Tune `free_gpu_memory_fraction` first, then tune
  `cross_kv_cache_fraction` based on whether the cross-attention or
  self-attention pool is exhausted.

## Performance

The following benchmarks compare the PyTorch backend with the legacy TensorRT
encoder-decoder path for large-batch inference. The measurements use BF16 on
one H100 80 GB GPU with greedy decoding, an output limit of 128 tokens, and
mixed encoder input lengths from 260 to 440 tokens. The Flan-T5-XL results are
the average of ten timed runs after three warmup runs. The BART results are the
average of 20 timed runs after five warmup runs. Executed-token throughput
includes the terminal EOS token when a sequence emits it.

The PyTorch configuration uses the `TRTLLM` attention backend, the overlap
scheduler, the Python scheduler, decoder CUDA graphs with padding, KV cache
manager V1, `max_input_len=512`, `max_seq_len=1024`, and
`max_num_tokens=65536`. Block reuse and chunked prefill are disabled. The KV
cache uses `free_gpu_memory_fraction=0.3` and
`cross_kv_cache_fraction=0.5`.

The legacy TensorRT configuration uses separate BF16 encoder and decoder
engines built for batch size 128 and beam width 1. The encoder supports 512
input tokens and 65,536 tokens per iteration; the decoder supports a sequence
length of 129. The benchmark runs these engines through `ModelRunnerCpp` with
greedy `top_k=1` decoding and the same KV cache fractions. For BART, the legacy
TensorRT benchmark starts the decoder with token IDs `[2, 0]` and generates at
most 127 more tokens. The PyTorch LLM API applies the same decoder prefix
internally and counts token ID 0 as the first output token; customers do not
need to provide the decoder prefix. Both paths use token ID 2 as EOS and stop
when the model generates it naturally. If a sequence reaches the output limit,
it retains the model-selected final token and reports a length stop instead of
forcing EOS. This setup also lets beam search begin after the shared decoder
prefix without a per-step Python logits processor.

### Flan-T5-XL

For Flan-T5-XL, the PyTorch backend performs on par with the legacy TensorRT
path, with slightly lower latency and higher executed-token throughput across
the tested batch sizes.

| Batch size | Legacy TensorRT latency | PyTorch latency | PyTorch latency improvement over legacy TensorRT | Legacy TensorRT executed tokens/s | PyTorch executed tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 727.6 ms | 706.1 ms | 3.0% | 3,153 | 3,312 |
| 64 | 1,225.0 ms | 1,136.7 ms | 7.2% | 3,863 | 4,184 |
| 128 | 2,056.8 ms | 1,999.3 ms | 2.8% | 4,601 | 4,768 |

### BART-large-CNN

For BART-large-CNN, the PyTorch backend has 21.9% to 36.0% higher latency than
the legacy TensorRT path across the tested batch sizes.

| Batch size | Legacy TensorRT latency | PyTorch latency | PyTorch latency difference | Legacy TensorRT executed tokens/s | PyTorch executed tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 229.9 ms | 280.2 ms | 21.9% slower | 12,611 | 10,662 |
| 64 | 252.2 ms | 343.0 ms | 36.0% slower | 22,007 | 16,209 |
| 128 | 352.3 ms | 472.7 ms | 34.2% slower | 31,544 | 23,518 |

Performance depends on the model, request distribution, decoding settings, and
GPU configuration. Benchmark with a representative workload before deployment.

## Troubleshooting

### `cross_kv_cache_fraction` is required

Every encoder-decoder runtime needs a cross-attention KV pool. Add
`KvCacheConfig(cross_kv_cache_fraction=...)`; `0.5` is a reasonable initial
value. Do not set this field for a decoder-only model.

### `decoder_start_token_id` is required

The checkpoint must define `decoder_start_token_id` or `bos_token_id` in its
Hugging Face model or generation configuration. Use a checkpoint with a complete
`config.json` and, when applicable, `generation_config.json`.

### KV cache manager V2 fails with beam search

V2 currently requires `max_beam_width=1`. Select V1 by setting
`use_kv_cache_manager_v2=False` before enabling beam search.

### Tensor parallel initialization is rejected

Check all of the following:

- `attn_backend` is `TRTLLM`.
- Encoder, decoder, and cross-attention head counts are divisible by the TP
  size.
- `pipeline_parallel_size=1` and `context_parallel_size=1`.
- `enable_attention_dp=False`.

### CUDA graphs do not capture the encoder

This is expected. `CudaGraphConfig` accelerates decoder iterations only. The
encoder path runs eagerly, and `EncodeCudaGraphConfig` is disabled for
encoder-decoder models.

### Output quality differs from the Hugging Face example

Confirm that the source uses the task prefix and language settings expected by
the checkpoint. Also compare the same model dtype, beam width, length penalty,
EOS stopping behavior, and forced BOS configuration. Small numerical
differences can change lower-ranked beam hypotheses when scores are close.
