# Advanced Usage

## Quantization
By simply setting several flags in the `LLM`, TensorRT-LLM can quantize the HuggingFace model automatically. For example, to perform an Int4 AWQ quantization, the following code triggers the model quantization.

``` python
from tensorrt_llm.hlapi import QuantConfig, QuantAlgo

quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)

llm = LLM(<model-dir>, quant_config=quant_config)
```


## Customization

### Customizing sampling with `SamplingParams`
With SamplingParams, you can customize the sampling strategy, such as beam search, temperature, and so on.

To enable beam search with a beam size of 4, set the `sampling_params` as follows:

```python
from tensorrt_llm.hlapi import LLM, SamplingParams, BuildConfig

build_config = BuildConfig()
build_config.max_beam_width = 4

llm = LLM(<llama_model_path>, build_config=build_config)
# Let the LLM object generate text with the default sampling strategy, or
# you can create a SamplingParams object as well with several fields set manually
sampling_params = SamplingParams(beam_width=4) # current limitation: beam_width should be equal to max_beam_width

for output in llm.generate(<prompt>, sampling_params=sampling_params):
    print(output)
```

`SamplingParams` manages and dispatches fields to C++ classes including:
* [SamplingConfig](https://nvidia.github.io/TensorRT-LLM/_cpp_gen/runtime.html#_CPPv4N12tensorrt_llm7runtime14SamplingConfigE)
* [OutputConfig](https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html#_CPPv4N12tensorrt_llm8executor12OutputConfigE)

Please refer to these classes for more details.


### Build configuration
Apart from the arguments mentioned above, you can also customize the build configuration with the `build_config` class and other arguments borrowed from the lower-level APIs. For example:

```python
llm = LLM(<model-path>,
          build_config=BuildConfig(
            max_new_tokens=4096,
            max_batch_size=128,
            max_beam_width=4))
```

### Runtime customization
Similar to `build_config`, you can also customize the runtime configuration with the `runtime_config`, `peft_cache_config` or other arguments borrowed from the lower-level APIs. For example:


```python
from tensorrt_llm.hlapi import LLM, KvCacheConfig

llm = LLM(<llama_model_path>,
          kv_cache_config=KvCacheConfig(
            max_new_tokens=128,
            free_gpu_memory_fraction=0.8))
```

### Tokenizer Customization

By default, the high-level API uses transformers’ `AutoTokenizer`. You can override it with your own tokenizer by passing it when creating the LLM object. For example:

```python
llm = LLM(<llama_model_path>, tokenizer=<my_faster_one>)
```

The LLM() workflow should use your tokenizer instead.

It is also possible to input token IDs directly without Tokenizers with the following code, note that the result will be also IDs without text since the tokenizer is not used.

``` python
llm = LLM(<llama_model_path>)

for output in llm.generate([32, 12]):
    ...
```

### Disable Tokenizer
For performance considerations, you can disable the tokenizer by passing `skip_tokenizer_init=True` when creating `LLM`. In this case, `LLM.generate` and `LLM.generate_async` will expect prompt token ids as input. For example:

```python
llm = LLM(<llama_model_path>)
for output in llm.generate([[32, 12]], skip_tokenizer_init=True):
    print(output)
```

You will get something like:
```python
RequestOutput(request_id=1, prompt=None, prompt_token_ids=[1, 15043, 29892, 590, 1024, 338], outputs=[CompletionOutput(index=0, text='', token_ids=[518, 10858, 4408, 29962, 322, 306, 626, 263, 518, 10858, 20627, 29962, 472, 518, 10858, 6938, 1822, 306, 626, 5007, 304, 4653, 590, 4066, 297, 278, 518, 11947, 18527, 29962, 2602, 472], cumulative_logprob=None, logprobs=[])], finished=True)
```

Note that the `text` field in `CompletionOutput` is empty since the tokenizer is deactivated.


## Generation

### `asyncio`-based generation
With the high-level API, you can also perform asynchronous generation with the `generate_async` method. For example:

```python
llm = LLM(model=<llama_model_path>)

async for output in llm.generate_async(<prompt>, streaming=True):
    print(output)
```

When the `streaming` flag is set to `True`, the `generate_async` method will return a generator that yields the token results as soon as they are available. Otherwise, it will return a generator that yields the final results only.

### Future-style generation
The result of the `generate_async` method is a Future-like object, it doesn't block the thread unless the `.result()` is called.

```python
# This will not block the main thread
generation = llm.generate_async(<prompt>)
# Do something else here
# call .result() to explicitly block the main thread and wait for the result when needed
output = generation.result()
```

The `.result()` method works like the [result](https://docs.python.org/zh-cn/3/library/asyncio-future.html#asyncio.Future.result) method in the Python Future, you can specify a timeout to wait for the result.

```python
output = generation.result(timeout=10)
```

There is an async version, where the `.aresult()` is used.

```python
generation = llm.generate_async(<prompt>)
output = await generation.aresult()
```
