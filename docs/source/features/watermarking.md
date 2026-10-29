# Text Watermarking

TensorRT-LLM can embed a statistical watermark in generated text while it samples tokens, so that an
operator holding the secret key can later test whether a piece of text came from the service. Some
regions require AI-generated content to be marked this way.

The watermark uses a keyed Gumbel-max draw in place of the random draw of the final token selection.
Temperature, top-k, top-p and min-p are applied first, so the output distribution is unchanged on
average over keys. Detection needs only the token ids, the tokenizer and the key.

## Enable it

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.watermark import WatermarkConfig

llm = LLM(model="Qwen/Qwen3-1.7B", watermark_config=WatermarkConfig(key=123456789))
out = llm.generate(["Write a short note about GPUs."], SamplingParams(temperature=0.8, max_tokens=200))
```

`WatermarkConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `algorithm` | `"gumbel"` | `"gumbel"` or `"gumbel_dual_key"`. Dual-key additionally supports speculative decoding with rejection sampling (see below). |
| `key` | required | Secret integer in `[0, 2**64)`. Detection needs the same value. Excluded from `repr`. |
| `context_width` | 4 | Number of preceding tokens that seed each decision. Larger values are less robust to edits. |
| `deduplicate_contexts` | `"single_turn"` | A repeated context would repeat its token and count as independent evidence, so it falls back to ordinary sampling. `"none"` disables this. |
| `deduplicate_contexts_max_history` | 8192 | How many generated positions back the repeat search reaches. `None` means the whole generation. |
| `allow_request_opt_out` | `True` | Whether a request may set `SamplingParams.watermarking=False`. Set to `False` to enforce the watermark on every request that can carry one. |
| `prf` | `"philox"` | Pseudo-random function. Fast but not cryptographic: it does not protect against key recovery or forgery. |

Per request, `SamplingParams.watermarking` (and the `watermarking` field of the OpenAI-compatible
completion and chat requests) is `None` or `True` to request a watermark and `False` to opt out.

## Speculative decoding

Watermarking works with the one-model speculative modes (Eagle3, MTP, DraftTarget and the others that
share their verification), with CUDA graphs, the overlap scheduler and chunked prefill.

- **Exact-match verification (the default).** A draft token is accepted only when it equals the token
  the target itself sampled at that position, so the output is exactly what plain decoding with the
  same key produces, and the watermark strength does not depend on the acceptance rate.
- **Rejection sampling (`use_rejection_sampling=True`).** Accepted draft tokens were not sampled from
  the target, so they need care. Two modes are supported:
  - `algorithm="gumbel_dual_key"` draws the draft tokens under the key and the token that follows the
    accepted prefix under a second key derived from it. The two draws and the accept decisions are
    independent, so the output distribution stays exact. The detector tests every token against both
    keys. Requires an autoregressive drafter (Eagle3, MTP, DraftTarget).
  - `allow_target_only_watermarking=True` leaves accepted draft tokens unmarked and watermarks only the
    token that follows them. The signal weakens as the acceptance rate rises.

  Without one of these, a watermarked request fails at admission.

## Detect it

```python
from tensorrt_llm.watermark import GumbelWatermarkDetector, WatermarkConfig

detector = GumbelWatermarkDetector(WatermarkConfig(key=123456789))
result = detector.detect(output_token_ids, prompt_token_ids=prompt_token_ids)
print(result.p_value, result.is_watermarked)
```

Tokenize text with the tokenizer used for generation. The configuration must match generation: key,
`context_width`, PRF and deduplication settings. Without `prompt_token_ids` the first `context_width`
positions are not scored. A fixed key reuses random values across documents, so the nominal p-value
assumes independent inputs; measure the false-positive rate on representative unwatermarked text
before relying on a decision. Detection outputs can help an attacker imitate or strip the watermark.

## Compatibility

Watermarking works with CUDA graphs, the overlap scheduler, chunked prefill, tensor, pipeline, expert and
attention data parallelism, disaggregated serving (configure both workers identically), and the one-model
speculative modes described above. A logits post-processor edits the logits before the watermark draw, so
tokens it bans are never sampled and the mark steers among the rest (not combinable with speculative
decoding, which the executor already rejects). With KV cache reuse, the first request matches an engine
without reuse; a later request that hits the cache can differ in its sampled text through bf16 numerics, as
it does without the watermark, and is still detected. Guided decoding, sliding window attention, LoRA and
Helix parallelism are untested. See the
[feature combination matrix](feature-combination-matrix.md).

## Limitations

- Greedy (`temperature=0`) and beam-search requests cannot carry a watermark and are generated normally.
- Sampling becomes deterministic given the context. Requests with `n > 1` and the same prompt produce
  the same text until their contexts diverge, and repeated contexts are not watermarked.
- Low-entropy output (code, short answers) and heavily edited text carry little signal.
- Guided decoding with the watermark is untested. The grammar mask is applied before the watermark draw, so
  the mark can only steer among the tokens the grammar allows.
- Not supported: NGram and user-provided drafts (verified in the sampler), tree (dynamic tree)
  speculation, relaxed acceptance for thinking, and ragged verification. The engine rejects these
  combinations: at startup for NGram and user-provided drafts, at admission for the others.
