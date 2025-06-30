# Sampling

The PyTorch backend supports most of the sampling features that are supported on the C++ backend, such as temperature, top-k and top-p sampling, stop words, bad words, penalty, context and generation logits, and log probs.

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
