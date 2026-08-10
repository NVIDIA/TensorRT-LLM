# LLM Common Customizations

## Quantization

TensorRT LLM runs quantized models from pre-quantized checkpoints. Use a checkpoint quantized with [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (for example, the ready-made FP8/NVFP4 checkpoints published on the [NVIDIA Hugging Face hub](https://huggingface.co/nvidia)), and the quantization configuration is detected automatically when the model loads:

``` python
from tensorrt_llm import LLM

llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8")
```

Refer to the [quantization feature documentation](../features/quantization.md) for the supported formats per GPU architecture and instructions on quantizing your own model.

## Sampling

SamplingParams can customize the sampling strategy to control LLM generated responses, such as beam search, temperature, and many others.

As an example, to enable beam search with a beam width of 4, configure the engine limit with `max_beam_width` and request beam search through `SamplingParams`:

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(<llama_model_path>, max_beam_width=4)
# Let the LLM object generate text with the default sampling strategy, or
# you can create a SamplingParams object as well with several fields set manually
sampling_params = SamplingParams(n=4, use_beam_search=True)

for output in llm.generate(<prompt>, sampling_params=sampling_params):
    print(output)
```

Refer to the [class documentation](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html#tensorrt_llm.llmapi.SamplingParams) for the complete list of fields.

## Runtime Customization

Runtime behavior such as KV cache management and GPU memory allocation can be customized with dedicated configuration classes like `kv_cache_config` and `peft_cache_config` passed to the `LLM` constructor. Refer to the following example:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(<llama_model_path>,
          kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.8))
```

Refer to the [LLM API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html) for all available configuration classes.

## Tokenizer Customization

By default, the LLM API uses transformers’ `AutoTokenizer`. You can override it with your own tokenizer by passing it when creating the LLM object. Refer to the following example:

```python
llm = LLM(<llama_model_path>, tokenizer=<my_faster_one>)
```

The LLM() workflow should use your tokenizer instead.

It is also possible to input token IDs directly without `Tokenizers` with the following code. The code produces token IDs without text because the tokenizer is not used.

``` python
llm = LLM(<llama_model_path>)

for output in llm.generate([32, 12]):
    ...
```

### Disable Tokenizer

For performance considerations, you can disable the tokenizer by passing `skip_tokenizer_init=True` when creating `LLM`. In this case, `LLM.generate` and `LLM.generate_async` will expect prompt token ids as input. Refer to the following example:

```python
llm = LLM(<llama_model_path>, skip_tokenizer_init=True)
for output in llm.generate([[32, 12]]):
    print(output)
```

You will get something like:
```python
RequestOutput(request_id=1, prompt=None, prompt_token_ids=[1, 15043, 29892, 590, 1024, 338], outputs=[CompletionOutput(index=0, text='', token_ids=[518, 10858, 4408, 29962, 322, 306, 626, 263, 518, 10858, 20627, 29962, 472, 518, 10858, 6938, 1822, 306, 626, 5007, 304, 4653, 590, 4066, 297, 278, 518, 11947, 18527, 29962, 2602, 472], cumulative_logprob=None, logprobs=[])], finished=True)
```

Note that the `text` field in `CompletionOutput` is empty since the tokenizer is deactivated.

## Generation

### Asyncio-Based Generation

With the LLM API, you can also perform asynchronous generation with the `generate_async` method. Refer to the following example:

```python
llm = LLM(model=<llama_model_path>)

async for output in llm.generate_async(<prompt>, streaming=True):
    print(output)
```

When the `streaming` flag is set to `True`, the `generate_async` method will return a generator that yields each token as soon as it is available. Otherwise, it returns a generator that waits for and yields only the final results.

### Future-Style Generation

The result of the `generate_async` method is a [Future-like](https://docs.python.org/3/library/asyncio-future.html#asyncio.Future) object, it doesn't block the thread unless the `.result()` is called.

```python
# This will not block the main thread
generation = llm.generate_async(<prompt>)
# Do something else here
# call .result() to explicitly block the main thread and wait for the result when needed
output = generation.result()
```

The `.result()` method works like the [result](https://docs.python.org/3/library/asyncio-future.html#asyncio.Future.result) method in the Python Future, you can specify a timeout to wait for the result.

```python
output = generation.result(timeout=10)
```

There is an async version, where the `.aresult()` is used.

```python
generation = llm.generate_async(<prompt>)
output = await generation.aresult()
```
