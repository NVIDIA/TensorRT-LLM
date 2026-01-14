# Sampling
The PyTorch backend supports most of the sampling features that are supported on the C++ backend, such as temperature, top-k and top-p sampling, beam search, stop words, bad words, penalty, context and generation logits, log probability and logits processors

## General usage

To use the feature:

1. Enable the `enable_trtllm_sampler` option in the `LLM` class
2. Pass a [`SamplingParams`](source:tensorrt_llm/sampling_params.py#L125) object with the desired options to the `generate()` function

The following example prepares two identical prompts which will give different results due to the sampling parameters chosen:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          enable_trtllm_sampler=True)
sampling_params = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

Note: The `enable_trtllm_sampler` option is not currently supported when using speculative decoders, such as MTP or Eagle-3, so there is a smaller subset of sampling options available.

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
3. Disable overlap scheduling using the `disable_overlap_scheduler` parameter of the `LLM` class
4. Disable the usage of CUDA Graphs by passing `None` to the `cuda_graph_config` parameter of the `LLM` class

Parameter Configuration:
- `best_of`: Controls the number of beams processed during generation (beam width)
- `n`: Controls the number of output sequences returned (can be less than `best_of`)
- If `best_of` is omitted, the number of beams processed defaults to `n`
- `max_beam_width` in the `LLM` class must equal `best_of` in `SamplingParams`

The following example demonstrates beam search with a beam width of 4, returning the top 3 sequences:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          enable_trtllm_sampler=True,
          max_beam_width=4,   # must equal SamplingParams.best_of
          disable_overlap_scheduler=True,
          cuda_graph_config=None)
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
