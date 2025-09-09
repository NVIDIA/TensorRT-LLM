# Sampling
The PyTorch backend supports most of the sampling features that are supported on the C++ backend, such as temperature, top-k and top-p sampling, beam search, stop words, bad words, penalty, context and generation logits, log probability, guided decoding and logits processors

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

## Guided decoding

Guided decoding controls the generation outputs to conform to pre-defined structured formats, ensuring outputs follow specific schemas or patterns.

The PyTorch backend supports guided decoding with the XGrammar and Low-level Guidance (llguidance) backends and the following formats:
- JSON schema
- JSON object
- Regular expressions
- Extended Backus-Naur form (EBNF) grammar
- Structural tags

To enable guided decoding, you must:

1. Set the `guided_decoding_backend` parameter to `'xgrammar'` or `'llguidance'` in the `LLM` class
2. Create a [`GuidedDecodingParams`](source:tensorrt_llm/sampling_params.py#L14) object with the desired format specification
    * Note: Depending on the type of format, a different parameter needs to be chosen to construct the object (`json`, `regex`, `grammar`, `structural_tag`).
3. Pass the `GuidedDecodingParams` object to the `guided_decoding` parameter of the `SamplingParams` object

The following example demonstrates guided decoding with a JSON schema:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import GuidedDecodingParams

llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          guided_decoding_backend='xgrammar')
structure = '{"title": "Example JSON", "type": "object", "properties": {...}}'
guided_decoding_params = GuidedDecodingParams(json=structure)
sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
    )
llm.generate("Generate a JSON response", sampling_params)
```

You can find a more detailed example on guided decoding [here](source:examples/llm-api/llm_guided_decoding.py).

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
