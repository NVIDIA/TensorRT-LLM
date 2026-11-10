# Text watermarking

The PyTorch backend supports context-keyed Gumbel and dual-key Gumbel text
watermarks for ordinary stochastic sampling. A matching detector can score
generated token IDs on the CPU, without model weights.

## Python API

Configure the engine with a private unsigned 64-bit key. Watermarking is enabled
by default on that engine; `SamplingParams(watermarking=False)` disables it for
an individual request. An engine without `watermark_config` rejects requests
that explicitly set `watermarking=True`.

```python
import os

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.watermarking import WatermarkConfig, WatermarkDetector

config = WatermarkConfig(
    algorithm="gumbel",  # or "dual_key_gumbel"
    key=int(os.environ["WATERMARK_KEY"]),
    context_width=4,
)
llm = LLM(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    tensor_parallel_size=4,
    watermark_config=config,
)
output = llm.generate(
    "Explain the water cycle in detail.",
    SamplingParams(max_tokens=256, temperature=1.0, top_p=0.95),
)
result = WatermarkDetector(config).detect(
    output.outputs[0].token_ids,
    prompt_token_ids=output.prompt_token_ids,
    threshold=0.01,
)
print(result.detected, result.p_value, result.num_scored_tokens)
```

Generation and detection must use the same key, algorithm, context width,
deduplication history limit, and tokenizer. Pass the original prompt token IDs
to score the beginning of the completion. Without the prompt, the detector
uses the first `context_width` completion tokens as unscored context.

## Serving

Set `watermark_config` in the server YAML passed to `--config`. The example key
below is public example data; replace it with a private value for deployment.

```yaml
tensor_parallel_size: 4
watermark_config:
  algorithm: dual_key_gumbel
  key: 123456789
  context_width: 4
  deduplicate_contexts_max_history: 8192
  alpha: 0.1
```

```bash
trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --config config.yaml
```

Both `/v1/completions` and `/v1/chat/completions` accept the optional
`watermarking` boolean. With the OpenAI Python client, pass
`extra_body={"watermarking": False}` to opt out. Omitting the field or sending
`null` follows the engine default. The key is configured on the server and is
never a per-request field.

## Sampling and detection behavior

Watermark selection uses the probabilities after temperature, penalties,
logits processors, token bans, and top-k/top-p/min-p filtering. It does not
introduce tokens excluded by these filters. Greedy rows retain ordinary greedy
sampling, and reported log probabilities describe the original sampling
distribution. Watermarking keeps its context on the device so it includes the
newest sampled token under the overlap scheduler. Its random function is
independent of tensor-parallel rank.

The last `context_width` token IDs determine keyed candidate noise. Gumbel
sampling selects the token with maximum `log(probability) + Gumbel(noise)`.
Dual-key Gumbel chooses between two domain-separated noise streams, using
`alpha` as the probability of selecting the secondary stream. Detection scores
both streams and corrects the minimum p-value for the two tests.

The same key and context produce the same candidate noise. Multiple
completions requested with `n > 1` can therefore produce identical output,
including their initial tokens. Greedy sampling does not add a watermark.

Contexts repeated within the most recent
`deduplicate_contexts_max_history` positions use ordinary sampling. Increasing
that limit increases per-active-sequence device memory. The detector excludes
every repeated context across the entire scored text, including contexts that
generation has already evicted from its bounded history, to keep repeated
keyed noise from counting as independent evidence. Short completions,
repeated contexts, and distributions concentrated on a single token provide
less detection evidence.

This implementation uses TensorRT-LLM's version-1 MurmurHash3/Philox random
function. Its token-level watermark format is not interchangeable with vLLM's
watermark implementation or detector. A watermark detection is statistical
evidence and does not provide cryptographic authentication.

## Supported combinations

Watermarking supports ordinary aggregated PyTorch decoding, including tensor
parallelism, streaming, overlap scheduling, and independent `n` completions.
Beam search, speculative decoding, and disaggregated serving are rejected when
watermarking is enabled. SynthID-Text is not implemented.
