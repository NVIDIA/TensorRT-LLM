# Sampling

The PyTorch backend supports a wide variety of features, listed below:

| Forward Pass       | Sampling Strategies              | Sampling Features              |
|--------------------|----------------------------------|--------------------------------|
| No drafting        |  Greedy                          | Guided Decoding                |
| Draft target model |  TopP                            | Plugging Logits Post-Processor |
| Eagle 3            |  TopK                            | Temperature                    |
| Ngram              |  TopK + TopP                     | MinP                           |
|                    |  Beam Search                     | Embedding / Logits Bias        |
|                    |  Best of / n (composable)        | Stop criteria                  |
|                    |  Rejection sampling (composable) | Return Logits                  |
|                    |                                  | Return LogProbs                |
|                    |                                  | TopK LogProbs                  |
|                    |                                  | Penalties                      |

## General usage

Torch Sampler is the only sampling backend and is used for all requests; there
is nothing to configure.

Here is an example to run a model with basic usage of sampling parameters. This example prepares two identical prompts which will give different results due to the sampling parameters chosen:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

It is also possible to specify different sampling parameters on a per-prompt basis:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params_0 = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
sampling_params_1 = SamplingParams(
        top_k=4,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"],
            [sampling_params_0,
            sampling_params_1])
```

### Model generation config defaults

The PyTorch backend can use compatible sampling defaults explicitly specified
in a model's `generation_config.json`. This behavior is opt-in:

```python
from tensorrt_llm import LLM

llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          generation_config='auto')
```

For `trtllm-serve`, enable it on the command line:

```bash
trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --generation-config auto
```

or in the server YAML configuration:

```yaml
generation_config: auto
```

The `generation_config` option has two modes:

* `trtllm` (default) keeps the TRT-LLM sampling behavior and defaults.
* `auto` loads supported sampling values from the model's
  `generation_config.json`.

In `auto` mode, values are resolved in this order:

1. A value explicitly specified by the request.
2. A value explicitly present in `generation_config.json`.
3. The existing default for the LLM API or serving protocol.

The supported fields are `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, `no_repeat_ngram_size`, `length_penalty`, and
`early_stopping` when its value is a boolean or integer. Defaults synthesized
by Hugging Face Transformers for fields absent from the JSON file are not
applied.

TRT-LLM's existing model-specific handling of `eos_token_id`, BART
`forced_bos_token_id`, and Whisper suppression tokens remains active in both
modes.

### LLM API sampling behavior when using Torch Sampler

* The sampling is controlled via `SamplingParams`.

* By default (`temperature = top_p = top_k = None`), greedy sampling is used
  (unless min-p or top-p decay is active, see below). With
  `generation_config='auto'`, values explicitly specified in the model's
  `generation_config.json` take the place of these defaults; see
  [Model generation config defaults](#model-generation-config-defaults).

* If either `temperature = 0`, `top_p = 0`, `top_k = 1`, and/or `min_p = 1`, is specified,
  sampling is greedy, irrespective of the values of the remaining parameters.

* Otherwise, sampling proceeds according to the specified sampling parameter values and any
  unspecified parameters default to `top_k = 0`, `top_p = 1`, `min_p = 0`, `temperature = 1.0`:

  * The logits are scaled by `1/temperature` before applying softmax to compute probabilities.
    Sampling is performed according to these probabilities.

  * If `top_k = 0` (or `top_k = vocab_size`), `top_p = 1` and `min_p = 0`, the output tokens
    are sampled from the entire vocabulary.

  * If `0 < min_p < 1` is specified, the sampling is restricted to the tokens whose probability
    is at least `min_p` times the probability of the most likely token ("min-p sampling").
    When combined with `top_k` and/or `top_p`, `min_p` is applied first.

  * If `1 < top_k < vocab_size` is specified, the sampling is restricted to
    the `top_k` highest-probability tokens.

  * If `0 < top_p < 1.0` is specified, the sampling is further restricted to a minimal subset
    of highest-probability tokens with total probability greater than `top_p` ("nucleus sampling").
    In particular, the probability of the lowest-probability token in the selected
    subset is greater or equal than the probability of any not selected token.
    When combined with `top_k`, the probabilities of the tokens selected by `top_k` are rescaled
    such that they sum to one before `top_p` is applied.

  * The implementation does not guarantee any particular treatment of tied probabilities.

* Top-P decay is supported: if `top_p_decay < 1` is specified, the effective `top_p` is
  multiplied by `top_p_decay` after every sampled token, bounded from below by `top_p_min`
  (default `1e-6`), and reset to the initial `top_p` whenever the token `top_p_reset_ids`
  is sampled (default `-1`, which never matches a token). Out-of-range values
  (`top_p_decay` or `top_p_min` outside `(0, 1]`, negative `top_p_reset_ids`) are rejected.

  * An active top-p decay implies top-p sampling even if `top_p` is unspecified or `top_p = 1`
    (the initial `top_p` then defaults to 1). However, explicitly requested greedy sampling
    (`temperature = 0`, `top_p = 0`, and/or `top_k = 1`) takes precedence over top-p decay.

  * Top-P decay is not supported in combination with beam search or with speculative decoding
    modes that route draft tokens through the Torch Sampler; such requests are rejected.

* Positive Min-P is not supported in combination with one-model speculative decoding. Such
  requests are rejected at admission.

* Occurrence penalties are supported: `repetition_penalty`, `presence_penalty` and
  `frequency_penalty` discourage (or encourage) the model from reusing tokens it has
  already seen. All three rewrite the logits before temperature scaling, driven by the
  occurrence history of the prompt plus everything generated so far. Writing `c` for the
  number of times a token has occurred in that history:

  * `repetition_penalty` (default `1.0`) rescales the logit of every token with `c > 0`:
    the logit is divided by the penalty when it is non-negative and multiplied by it when
    it is negative. The two branches move a positive and a negative logit the same way, so
    a value `> 1` always pushes a seen token down, and a value `< 1` always pulls it up.
    Must be `> 0`.

  * `presence_penalty` (default `0.0`) subtracts the penalty itself from every token with
    `c > 0`. The amount does not depend on `c`, so it controls whether a token reappears,
    not how often.

  * `frequency_penalty` (default `0.0`) subtracts the penalty multiplied by `c`, so the
    more often a token has already been produced, the harder it is pushed down.

  * `prompt_ignore_length` (default `0`) excludes the first N prompt tokens from the
    presence and frequency counts. Those ignored tokens still count for
    `repetition_penalty`. Values `<= 0` have no effect, and values larger than the prompt
    are clamped to the prompt length.

  * With beam search the occurrence history is kept per beam rather than per request:
    each beam is penalized against the tokens on its own path, and whenever a beam
    continues another one it inherits that beam's history. The prompt seeds every beam
    alike, so `prompt_ignore_length` applies to all of them equally.

  * With one-model speculative decoding the penalties must be enabled at deploy time
    with `enable_penalty: true` in the speculative decoding config, because they need
    an occurrence workspace that is allocated up front. While the flag is off, a
    request that sets any of the three is rejected at admission rather than decoded
    without them. Tree speculation (`eagle_choices` or a dynamic tree) is not
    supported and such requests are rejected even when the flag is on. Only the
    target distribution is penalized; the draft model proposes from its unpenalized
    distribution, which leaves the sampled result unchanged but can lower the
    acceptance rate as the penalty grows.

* If `no_repeat_ngram_size = n` is specified, any token that would recreate an `n`-gram already
  present in the sequence (prompt included) is excluded from sampling. `None` or `0` disables
  the restriction.

### Performance

The Torch Sampler leverages the optimized sampling kernels provided by
[FlashInfer](https://docs.flashinfer.ai/api/sampling.html), which is a required
dependency for the Torch Sampler. The sampler also uses the
[sorting-free implementations](https://flashinfer.ai/2025/03/10/sampling.html)
whenever possible. This optimization does not compute the complete set of token sampling probabilities
(after top-k / top-p masking etc.), which typically can be omitted unless requested by the user or
required for speculative decoding (rejection sampling).

Moreover, Torch Sampler internally batches requests with compatible sampling parameters. This
can greatly reduce the overall latency of the sampling step when request batches are comprised
of requests with very heterogeneous sampling strategies (e.g. a mix of requests using greedy and top-p-after-top-k sampling).

## Advanced sampling mode (speculative decoding)

For one-model speculative decoding (e.g. MTP-Eagle one-model), the per-request
advanced sampler applies a `top_k` mask, a temperature softmax, and a `top_p`
filter before sampling each draft/target token. When a deployment fixes its
sampling configuration such that a filter is always disabled (`top_k = 0` /
`top_k = vocab_size`, or `top_p = 1`), that filter's kernel is pure overhead.

`advanced_sampling_mode` (on `DecodingBaseConfig`, so it is available to any
speculative config) lets you skip those redundant kernels for a fixed deploy
config. The output is identical to `FULL` whenever the skipped filter is already
disabled, so this is a lossless throughput optimization for advanced use cases:

| Mode | `top_k` kernel | `top_p` kernel |
|---|---|---|
| `full` (default) | applied | applied |
| `no_topk` | **skipped** | applied |
| `no_topp` | applied | **skipped** |
| `no_topk_no_topp` | **skipped** | **skipped** |

Notes:

* `full` is the default and always safe; the specialization is opt-in.
* `advanced_sampling_mode` and `use_rejection_sampling` are independent: every mode
  works with rejection sampling on or off; the flag no longer gates the mode choice.
* `no_topp` and `no_topk_no_topp` disable `top_p`, switching the sampler from the
  fused `top_p_sampling_from_probs` to the cheaper `sampling_from_probs`; `no_topk`
  keeps `top_p`.
* Greedy requests are handled natively (via a sentinel temperature that makes the
  softmax collapse to a one-hot argmax), so any mode supports mixed greedy +
  sampling batches without a special case.
* `advanced_sampling_mode` is a deploy-time choice; it is *not* part of the CUDA
  graph key, so it adds no extra warmup graphs.

```python
from tensorrt_llm.llmapi import MTPDecodingConfig

spec_config = MTPDecodingConfig(
    max_draft_len=3,
    advanced_sampling_mode="no_topk_no_topp",  # temperature-only deploy config
)
```

## Beam search

Beam search is a decoding strategy that maintains multiple candidate sequences (beams) during text generation, exploring different possible continuations to find higher quality outputs. Unlike greedy decoding or sampling, beam search considers multiple hypotheses simultaneously.

To enable beam search, you must:

1. Enable the `use_beam_search` option in the `SamplingParams` object
2. Set the `max_beam_width` parameter in the `LLM` class to match the `best_of` parameter in `SamplingParams`

Parameter Configuration:
- `best_of`: Controls the number of beams processed during generation (beam width)
- `n`: Controls the number of output sequences returned (can be less than `best_of`)
- If `best_of` is omitted, the number of beams processed defaults to `n`
- `max_beam_width` in the `LLM` class must equal `best_of` in `SamplingParams`
- `length_penalty`: Controls how beams of different lengths are compared. Candidate beams are
  ranked by `cum_log_prob / length**length_penalty`, where `length` is the number of generated
  tokens. The default (`0.0`) ranks beams by their raw cumulative log-probability, which favors
  shorter sequences; values above `0.0` favor longer sequences. The `cumulative_logprob` values
  returned with the outputs remain unnormalized.
- `beam_search_diversity_rate`: Encourages beams to diverge from each other. During beam
  expansion, `diversity_rate * source_beam_index` is added to each candidate's ranking score,
  boosting candidates that expand from lower-ranked beams so that the selected beams do not all
  descend from the single strongest beam. Here `source_beam_index` is the rank of the beam a
  candidate expands from among the current step's input beams, ordered by their cumulative
  log-probability (`0` for the strongest beam, `1` for the next, and so on). The default (`0.0`)
  disables the adjustment.
- `early_stopping`: Controls when beam search stops. It is a three-state setting following
  Hugging Face: `1` (the default) ends generation as soon as `best_of` finished candidates
  exist; `0` and `2` are exhaustive, keeping a pool of finished candidates and continuing while
  an unfinished beam could still outscore the worst of them. The two differ in how optimistic
  that bound is: `0` measures attainability against the beams' current length, `2` ("never")
  against `max_seq_len` when `length_penalty > 0`. Any other integer is treated as `2`.

Beam search rejects the following combinations, raising an error at admission:

- **Disaggregated serving.** The pool of finished candidates the context server builds is not
  part of the handoff, so a completion found there would be silently dropped. Use
  `best_of=1` on a disaggregated deployment.
- **A decreasing `beam_width_array`.** Only non-decreasing schedules are supported; the
  semantics of narrowing mid-decode are not defined.
- **A `best_of` other than `max_beam_width`.** Every request in an engine runs at the same
  beam width, which admission enforces so that a mismatch is reported against the offending
  request. Mixing widths is a forward-time failure that aborts the whole batch: note that
  admission compares `best_of` against `max_beam_width` only, so requests whose
  `beam_width_array` puts them at different per-iteration widths in the same step still
  reach that failure.

The following example demonstrates beam search with a beam width of 4, returning the top 3 sequences:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          max_beam_width=4,   # must equal SamplingParams.best_of
    )
sampling_params = SamplingParams(
        best_of=4,   # must equal LLM.max_beam_width
        use_beam_search=True,
        n=3,         # return top 3 sequences
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

### Over the OpenAI-compatible API

`length_penalty` and `early_stopping` now default to `null` in the HTTP schema, deferring to
the engine defaults (`0.0` and `1`) rather than restating them. Previously the schema defaulted
`length_penalty` to `1.0`, so a beam-search request that did not set it was normalizing scores
by sequence length; the same request now ranks by the raw cumulative log-probability. Set
`"length_penalty": 1.0` explicitly to keep the old ranking.

`early_stopping` accepts `false`, `true` and `"never"` over HTTP, mirroring HuggingFace, and is
translated to the engine's `0` / `1` / `2`. Integers outside that set are rejected by the
schema rather than silently reinterpreted.

## Logits processor

Logits processors allow you to modify the logits produced by the network before sampling, enabling custom generation behavior and constraints.

To use a custom logits processor:

1. Create a custom class that inherits from [`LogitsProcessor`](source:tensorrt_llm/sampling_params.py#L48) and implements the `__call__` method
2. Pass an instance of this class to the `logits_processor` parameter of `SamplingParams`

The following example demonstrates logits processing:

```python
import torch
from typing import List, Optional

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.sampling_params import LogitsProcessor

class MyCustomLogitsProcessor(LogitsProcessor):
    def __call__(self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int]
    ) -> None:
        # Implement your custom inplace logits processing logic
        logits *= logits

llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params = SamplingParams(
        logits_processor=MyCustomLogitsProcessor()
    )
llm.generate(["Hello, my name is"], sampling_params)
```

You can find a more detailed example on logits processors [here](source:examples/llm-api/llm_logits_processor.py).
