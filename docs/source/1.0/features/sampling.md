# Sampling
The PyTorch backend supports most of the sampling features that are supported on the C++ backend, such as temperature, top-k and top-p sampling, beam search, stop words, bad words, penalty, context and generation logits, and log probs.

## General usage

In order to use this feature, it is necessary to enable option `enable_trtllm_sampler` in the `LLM` class, and pass a `SamplingParams` object with the desired options as well. The following example prepares two identical prompts which will give different results due to the sampling parameters chosen:

```python
from tensorrt_llm import LLM
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

When using speculative decoders such as MTP or Eagle-3, the `enable_trtllm_sampler` option is not yet supported and therefore the subset of sampling options available is more restricted.

## Beam search

Beam search is a decoding strategy that maintains multiple candidate sequences (beams) during text generation, exploring different possible continuations to find higher quality outputs. Unlike greedy decoding or sampling, beam search considers multiple hypotheses simultaneously.

To enable beam search, you must:

1. Enable the `use_beam_search` option in the `SamplingParams` object
2. Set the `max_beam_width` parameter in the `LLM` class to match the `best_of` parameter in `SamplingParams`
3. Disable overlap scheduling using the `disable_overlap_scheduler` parameter of the `LLM` class

Parameter Configuration:
- `best_of`: Controls the number of beams processed during generation (beam width)
- `n`: Controls the number of output sequences returned (can be less than `best_of`)
- If `best_of` is omitted, it will be implicitly set to `n`
- `max_beam_width` in the `LLM` class must equal `best_of` in `SamplingParams`

The following example demonstrates beam search with a beam width of 4, returning the top 3 sequences:

```python
from tensorrt_llm import LLM
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          enable_trtllm_sampler=True,
          max_beam_width=4,   # must equal SamplingParams.best_of
          disable_overlap_scheduler=True)
sampling_params = SamplingParams(
        best_of=4,   # must equal LLM.max_beam_width
        use_beam_search=True,
        n=3,         # return top 3 sequences
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```
