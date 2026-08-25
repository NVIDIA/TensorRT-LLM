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

This guide covers text-to-text generation and speech-to-text transcription with
the following Hugging Face architectures:

| Hugging Face architecture | Model families and examples |
| --- | --- |
| `T5ForConditionalGeneration` | T5, Flan-T5, and ByT5, for example `google/flan-t5-small` |
| `BartForConditionalGeneration` | BART checkpoints |
| `MBartForConditionalGeneration` | mBART checkpoints |
| `WhisperForConditionalGeneration` | Whisper automatic speech recognition (ASR), for example `openai/whisper-large-v3` |

mBART architecture loading is available. When a BART or mBART checkpoint
defines `forced_bos_token_id`, the PyTorch backend seeds that token in the
decoder prefix. Source- and target-language selection remains
checkpoint-specific, so validate the configured language tokens before
deployment. Refer to [Understand BART and mBART decoder tokens](#understand-bart-and-mbart-decoder-tokens)
for BOS, EOS, and output-limit behavior.

Whisper consumes audio rather than a text prompt. The sections up to
[Transcribe audio with Whisper](#transcribe-audio-with-whisper) describe the
text-to-text models; the runtime configuration, KV cache manager, beam search,
CUDA graph, and parallelism guidance in this guide applies to all architectures.

## Feature support

The following table describes the supported and recommended configurations.

| Feature | Support | Notes |
| --- | --- | --- |
| KV cache manager V1 | Yes; recommended | This is the default. It supports greedy decoding, beam search, batching, the overlap scheduler, decoder CUDA graphs, and tensor parallelism. |
| KV cache manager V2 | Yes | Set `use_kv_cache_manager_v2=True`. It currently requires `max_beam_width=1`, so use greedy or sampling with a single sequence rather than beam search. |
| Greedy decoding | Yes | Set `temperature=0.0`. |
| Beam search | Yes with V1 | Configure `max_beam_width` when constructing `LLM`, then set `use_beam_search=True` in `SamplingParams`. |
| Attention backend | `TRTLLM` | Use this backend for encoder-decoder models. It is required when `tensor_parallel_size > 1`. |
| Decoder CUDA graphs | Yes, except in FP32 | `CudaGraphConfig` captures decoder work. V1 supports greedy and beam search; V2 supports its single-beam path. FP32 encoder-decoder models decline capture at engine init and log a warning instead of failing. |
| Encoder CUDA graphs | Yes | Set `encoder_cuda_graph_config=EncodeCudaGraphConfig(...)` and `encoder_max_batch_size`. Usually set `encoder_max_batch_size` lower than `max_batch_size`. The `TRTLLM` attention backend is required. Text encoders also require `num_tokens` and `seq_lens`; a fixed-shape feature encoder such as Whisper derives both from the model and needs only `batch_sizes`. |
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

## Transcribe audio with Whisper

Whisper is an audio encoder-decoder model for speech transcription and
translation. It differs from the text-to-text models in this guide in two ways:
the encoder consumes audio instead of a text prompt, and the text prompt, when
supplied, sets the decoder task prefix rather than the encoder input.

Pass one audio clip per request through `multi_modal_data["audio"]`:

```python
import soundfile

from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams, SchedulerConfig


model = "openai/whisper-large-v3"
wave, sample_rate = soundfile.read("utterance.wav")

with LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    max_batch_size=4,
    # Cross-KV pool capacity. The default (1024) is smaller than the 1500
    # encoder positions that every Whisper request produces.
    max_input_len=1500,
    max_num_tokens=3000,
    enable_chunked_prefill=False,
    kv_cache_config=KvCacheConfig(
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
    ),
    scheduler_config=SchedulerConfig(use_python_scheduler=True),
) as llm:
    result = llm.generate(
        {"prompt": "", "multi_modal_data": {"audio": [(wave, sample_rate)]}},
        sampling_params=SamplingParams(max_tokens=96, temperature=0.0),
        use_tqdm=False,
    )
    print(result.outputs[0].text)
```

Set `max_input_len` to at least 1500. Every Whisper request produces 1500 encoder
positions regardless of clip length, and the cross-KV pool is sized from `max_input_len`.

The audio item accepts a file path or URL, an `(array, sample_rate)` tuple, or a
`{"array": ..., "sampling_rate": ...}` mapping. Supply exactly one clip per
request, at the checkpoint's sampling rate, which is 16 kHz for the published
Whisper checkpoints. Clips shorter than the 30-second window are zero-padded;
longer clips and other sampling rates are rejected rather than silently
truncated or resampled. Long-form chunked transcription is not part of this
path.

### Select the language and task

An empty text prompt selects the checkpoint default, which is English
transcription (`<|startoftranscript|>[<|en|>][<|transcribe|>]<|notimestamps|>`).
A non-empty text prompt replaces that decoder prefix verbatim and is how you
override the language or switch to translation. It must begin with
`<|startoftranscript|>`:

```python
prompt = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
result = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"audio": [(wave, sample_rate)]}},
    sampling_params=SamplingParams(max_tokens=96, temperature=0.0),
)
```

The prefix counts against the decoder position table, so `SamplingParams.max_tokens`
is capped to the space remaining in it. A request asking for more logs a warning
and proceeds with the capped budget. Pre-tokenized `prompt_token_ids` are not
consumed on this path.

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

## Enable encoder and decoder CUDA graphs

Configure the decoder and encoder graph grids separately:

```python
from tensorrt_llm.llmapi import CudaGraphConfig, EncodeCudaGraphConfig


llm = LLM(
    model=model,
    backend="pytorch",
    attn_backend="TRTLLM",
    max_batch_size=8,
    encoder_max_batch_size=2,
    encoder_max_num_tokens=2048,
    cuda_graph_config=CudaGraphConfig(
        max_batch_size=8,
        enable_padding=True,
    ),
    encoder_cuda_graph_config=EncodeCudaGraphConfig(
        batch_sizes=[1, 2],
        num_tokens=[128, 256, 512, 1024, 2048],
        seq_lens=[128, 256, 512, 1024],
        enable_padding=True,
    ),
    enable_encoder_decoder_mixed_cuda_graph=True,
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.8,
        cross_kv_cache_fraction=0.5,
    ),
)
```

`cuda_graph_config` controls decoder and mixed decoder graphs.
`encoder_cuda_graph_config` controls encoder-forward graph buckets for batch
size, total packed tokens, and maximum sequence length. The
`encoder_max_batch_size` value is the hard encoder capacity and admission
limit. With beam search, decoder graph batch sizes must cover the active
decoder sequences after beam expansion.

Which encoder buckets you must supply depends on the model. A text encoder,
such as BART or T5, packs a variable number of tokens per request, so
`num_tokens` and `seq_lens` are part of its key space and are required; leaving
either unset is rejected at `LLM(...)` for any architecture TensorRT-LLM
recognizes, and at model engine initialization otherwise, since only the loaded
model states with certainty which kind of encoder it has. An encoder whose input is a
fixed-shape per-request feature tensor, such as Whisper's fixed 30-second
zero-padded audio waveform (the mel transform runs inside the encoder, so the
per-request input is the waveform itself, not a spectrogram), produces the same
number of encoder positions for every request, so both lists follow from the
model and are derived rather than configured. For those models
`batch_sizes` alone enables capture, and any `num_tokens` or `seq_lens` you set
is ignored.

Batch sizes that do not fit `encoder_max_num_tokens` divided by the model's
encoder output length are dropped, and the encoder stays eager when none fit.
Size the encoder token budget for the largest bucket before setting the
buckets: Whisper emits 1500 encoder positions per request, so `batch_sizes` up
to 8 needs `encoder_max_num_tokens` of at least 12000. `encoder_max_num_tokens`
falls back to `max_num_tokens` when unset, which is a decoder-sized number and
usually too small.

```python
from tensorrt_llm.llmapi import EncodeCudaGraphConfig


llm = LLM(
    model="openai/whisper-large-v3",
    backend="pytorch",
    attn_backend="TRTLLM",
    max_batch_size=8,
    encoder_max_batch_size=8,
    # 8 buckets * 1500 encoder positions. Leave this at the default and the
    # 4 and 8 buckets are silently dropped.
    encoder_max_num_tokens=12000,
    encoder_cuda_graph_config=EncodeCudaGraphConfig(batch_sizes=[1, 2, 4, 8]),
    # ... the remaining Whisper settings from "Transcribe audio with Whisper",
    # whose `max_batch_size=4` this example raises to 8
)
```

`max_batch_size` controls the total decoder concurrency, while
`encoder_max_batch_size` controls encoder microbatch admission. For better
performance, tune `encoder_max_batch_size`, `encoder_max_num_tokens`, and the
encoder CUDA graph buckets together for the production workload. Start with
`encoder_max_batch_size` smaller than `max_batch_size`, such as 2 versus 8,
then adjust the limits and capture buckets based on benchmark results.

`enable_encoder_decoder_mixed_cuda_graph` is primarily a performance option. It
reduces CPU launch overhead for decoder iterations that mix newly admitted
context requests with ongoing generation requests. The option defaults to
`True`, but becomes effective only when the encoder and decoder graph
configurations produce usable capture shapes. Set it to `False` to disable
mixed graphs while retaining the separate encoder and decoder CUDA graphs.

Passing `EncodeCudaGraphConfig` through `cuda_graph_config` remains unsupported
for encoder-decoder models; pass it through `encoder_cuda_graph_config`
instead. Piecewise CUDA graphs through `TorchCompileConfig` are also
unsupported for this model type.

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
max_batch_size: 8
encoder_max_batch_size: 2
encoder_max_num_tokens: 1024
max_beam_width: 1
max_input_len: 512
max_num_tokens: 2048
max_seq_len: 512
cuda_graph_config:
  max_batch_size: 8
  enable_padding: true
encoder_cuda_graph_config:
  batch_sizes: [1, 2]
  num_tokens: [128, 256, 512, 1024]
  seq_lens: [128, 256, 512]
  enable_padding: true
enable_encoder_decoder_mixed_cuda_graph: true
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
- Set `max_num_tokens` high enough for the active decoder tokens.
- Set `encoder_max_num_tokens` high enough for all encoder tokens in one
  encoder microbatch. This is especially important for mixed-length batches.
- Increase `max_batch_size` for more concurrent requests. Start with a smaller
  `encoder_max_batch_size`, such as 2 when `max_batch_size=8`, to bound encoder
  memory and admission cost without reducing decoder concurrency. Beam width
  multiplies the number of active decoder sequences but not the number of
  source requests.
- Tune `free_gpu_memory_fraction` first, then tune
  `cross_kv_cache_fraction` based on whether the cross-attention or
  self-attention pool is exhausted.

## Performance

Configure encoder, decoder, and mixed decoder CUDA graphs for the expected
serving workload. With representative capture buckets, the PyTorch backend can
outperform the legacy TensorRT encoder-decoder path while avoiding the engine
build and checkpoint conversion steps.

For example, a BF16 FLAN-T5 Large serving benchmark on one H100 80 GB GPU used
encoder CUDA graphs, padded decoder CUDA graphs, and mixed encoder-decoder CUDA
graphs. Compared with the legacy TensorRT path, the PyTorch backend delivered
65.8%, 12.6%, and 11.8% higher request throughput at concurrencies 8, 32, and
64, respectively. P99 latency was 51.6%, 31.0%, and 33.5% lower.

Follow [Enable encoder and decoder CUDA graphs](#enable-encoder-and-decoder-cuda-graphs)
and choose capture buckets that cover the batch sizes, packed encoder token
counts, and sequence lengths expected in production. Capture grids that omit
common runtime shapes fall back to eager execution and can lose these
performance benefits.

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

### Encoder CUDA graphs fall back to eager execution

Check that `encoder_cuda_graph_config` and `encoder_max_batch_size` are set,
that the encoder graph buckets cover the request shape, and that
`attn_backend="TRTLLM"`. Unsupported shapes and attention backends fall back to
eager encoder execution.

### `num_tokens` or `seq_lens` unset is rejected at engine construction

A text encoder needs both bucket lists, so the engine raises rather than
silently running eager. Supply them, or drop `encoder_cuda_graph_config` if you
do not want encoder graphs. A fixed-shape feature encoder such as Whisper does
not hit this: it derives both from the model and needs only `batch_sizes`.

### Output quality differs from the Hugging Face example

Confirm that the source uses the task prefix and language settings expected by
the checkpoint. Also compare the same model dtype, beam width, length penalty,
EOS stopping behavior, and forced BOS configuration. Small numerical
differences can change lower-ranked beam hypotheses when scores are close.
