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

## General usage

There are two sampling backends available.

* Torch Sampler
* TRTLLM Sampler

Torch Sampler currently supports a superset of features of TRTLLM Sampler, and is intended as the long-term solution. One can specify which sampler to use explicitly with:

```python
from tensorrt_llm import LLM

# Chooses TorchSampler explicitly
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          sampler_type="TorchSampler")

# Chooses TRTLLMSampler explicitly
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          sampler_type="TRTLLMSampler")
```

By default, the sampling backend is chosen to be `auto`. This will use:

* TRTLLM Sampler when using Beam Search.
* Torch Sampler otherwise.

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

### LLM API sampling behavior when using Torch Sampler

* The sampling is controlled via `SamplingParams`.

* By default (`temperature = top_p = top_k = None`), greedy sampling is used.

* If either `temperature = 0`, `top_p = 0`, and/or `top_k = 1`, is specified, sampling is greedy,
  irrespective of the values of the remaining parameters.

* Otherwise, sampling proceeds according to the specified sampling parameter values and any
  unspecified parameters default to `top_k = 0`, `top_p = 1`, `temperature = 1.0`:

  * The logits are scaled by `1/temperature` before applying softmax to compute probabilities.
    Sampling is performed according to these probabilities.

  * If `top_k = 0` (or `top_k = vocab_size`) and `top_p = 1`, the output tokens are sampled
    from the entire vocabulary.

  * If `1 < top_k < vocab_size` is specified, the sampling is restricted to
    the `top_k` highest-probability tokens.

  * If `0 < top_p < 1.0` is specified, the sampling is further restricted to a minimal subset
    of highest-probability tokens with total probability greater than `top_p` ("nucleus sampling").
    In particular, the probability of the lowest-probability token in the selected
    subset is greater or equal than the probability of any not selected token.
    When combined with `top_k`, the probabilities of the tokens selected by `top_k` are rescaled
    such that they sum to one before `top_p` is applied.

  * The implementation does not guarantee any particular treatment of tied probabilities.

### Performance

The Torch Sampler leverages the optimized sampling kernels provided by
[FlashInfer](https://docs.flashinfer.ai/api/sampling.html). The sampler
also uses the [sorting-free implementations](https://flashinfer.ai/2025/03/10/sampling.html)
whenever possible. This optimization does not compute the complete set of token sampling probabilities
(after top-k / top-p masking etc.), which typically can be omitted unless requested by the user or
required for speculative decoding (rejection sampling).
In case of unexpected problems, the use of FlashInfer in Torch Sampler can
be disabled via the `disable_flashinfer_sampling` config option (note that this option is likely
to be removed in a future TensorRT LLM release).

Moreover, Torch Sampler internally batches requests with compatible sampling parameters. This
can greatly reduce the overall latency of the sampling step when request batches are comprised
of requests with very heterogeneous sampling strategies (e.g. a mix of requests using greedy and top-p-after-top-k sampling).

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
