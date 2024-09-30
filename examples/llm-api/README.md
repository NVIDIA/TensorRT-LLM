# High-level API
We are working on a Python high-level API(HLAPI) for LLM workflow, which is still in incubation and may change later.
Here we show you a preview of how it works and how to use it.

Note that the APIs are not stable and we appreciate your patience and understanding as we improve this API.

## HLAPI Supported Model
* LLaMA (including variants Mistral, Mixtral, InternLM)
* GPT (including variants Starcoder-1/2, Santacoder)
* Gemma-1/2
* Phi-1/2/3
* ChatGLM (including variants glm-10b, chatglm, chatglm2, chatglm3, glm4)
* QWen-1/1.5/2
* Falcon
* Baichuan-1/2
* GPT-J

## Quick start

Please install the required packages first:

```bash
pip install -r requirements.txt
```

Here is a simple example to show how to use the HLAPI:

Firstly, import the `LLM` and `SamplingParams` from the `tensorrt_llm` package, and create an LLM object with a HuggingFace (HF) model directly. Here we use the TinyLlama model as an example, `LLM` will download the model from the HuggingFace model hub automatically. You can also specify local models, either in HF format, TensorRT-LLM engine format or TensorRT-LLM checkpoint format.

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

Secondly, generate text with the `generate` method of the `LLM` object directly with a batch of prompts, the `sampling_params` is optional, and you can customize the sampling strategy with it.

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Please refer to the [LLM quickstart](./quickstart_example.py) for the complete example.

## Examples

You can refer to the scripts in the current directory for all of the examples.

## Model preparation
The `LLM` class supports four kinds of model inputs:

1. **HuggingFace model name**: triggers a download from the HuggingFace model hub, e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in the quickstart.
1. **Local HuggingFace models**: uses a locally stored HuggingFace model.
2. **Local TensorRT-LLM engine**: built by `trtllm-build` tool or saved by the HLAPI
3. **Local TensorRT-LLM checkpoints**: converted by `convert_checkpoint.py` script in the examples

All kinds of the model inputs can be seamlessly integrated with the HLAPI, and the `LLM(model=<any-model-path>)` construcotr can accommodate models in any of the above formats.

Let's delve into the preparation of the three kinds of local model formats.

### Option 1: From HuggingFace models

Given its popularity, the TRT-LLM HLAPI chooses to support HuggingFace format as one of the start points, to use the HLAPI on LLaMA models, you need to run the following conversion script provided in [transformers/llama](https://huggingface.co/docs/transformers/main/model_doc/llama) or [transformers/llama2](https://huggingface.co/docs/transformers/main/model_doc/llama2) to convert the Meta checkpoint to HuggingFace format.

For instance, when targeting the LLaMA2 7B model, the official way to retrieve the model is to visit the [LLaMA2 model page](https://huggingface.co/docs/transformers/main/en/model_doc/llama2), normally you need to submit a request for the model file.

To convert the checkpoint files, a script from transformers is required, thus please also clone the transformers repo with the following code:

```sh
git clone https://github.com/huggingface/transformers.git
```

Finally, the command to convert the checkpoint files to HuggingFace format is as follows:

``` sh
python <transformers-dir>/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir Llama-2-7b --model_size 7B --output_dir llama-hf-7b
```

That should produce a HuggingFace format model in `./llama-hf-7b`, which could be used by the HLAPI.

### Option 2: From TensorRT-LLM engine
There are two ways to build the TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the HuggingFace model directly with the `trtllm-build` tool, and save the engine to disk for later use.  Please consult the LLaMA's [README](../llama/README.md).
2. Use the HLAPI to save one:

```python
llm = LLM(<model-path>)

# Save engine to local disk
llm.save(<engine-dir>)
```

### Option 3: From TensorRT-LLM checkpoint
In each model example, there is a `convert_checkpoint.py` to convert third-party models to TensorRT-LLM checkpoint for further usage.
The HLAPI could seamlessly accept the checkpoint, and build the engine in the backend.
For step-by-step guidance on checkpoint conversion, please refer to the LLaMA's [README](../llama/README.md).


## Basic usage
To use the API, import the `LLM` from the `tensorrt_llm` package and create an LLM object with a HuggingFace model directly.
For example:

``` python
from tensorrt_llm import LLM

llm = LLM(model=<llama_model_path>)
```

It will trigger TRT-LLM engine building in the backend, and create a HuggingFace tokenizer by default to support an end-to-end generation.

To generate text, use the `generate` method of the `LLM` object directly with a batch of prompts, for example:

``` python
prompts = ["To tell a story"]
for output in llm.generate(prompts):
    print(output)
```

The output might be something like:

``` python
RequestOutput(request_id=2, prompt='To tell a story', prompt_token_ids=[1, 1763, 2649, 263, 5828], outputs=[CompletionOutput(index=0, text=', you need to have a beginning, a middle, and an end.\nThe beginning is the introduction of the characters and the setting.\nThe middle is', token_ids=[29892, 366, 817, 304, 505, 263, 6763, 29892, 263, 7256, 29892, 322, 385, 1095, 29889, 13, 1576, 6763, 338, 278, 18707, 310, 278, 4890, 322, 278, 4444, 29889, 13, 1576, 7256, 338], cumulative_logprob=None, logprobs=[])], finished=True)
```

You can also dump the runtime engine to disk, and load from the engine file directly in the next run to save the engine building time from the HuggingFace model.

``` python
# dump the llm
llm.save(<engine-path>)

# next time
llm = LLM(model=<engine-path>)
```

In other words, the `model_dir` could accept either a HugggingFace model, a built TensorRT-LLM engine, or a TensorRT-LLM checkpoint, and the `LLM()` will do the rest work silently for end-to-end execution.

## Quantization

By simply setting several flags in the `LLM`, TensorRT-LLM can quantize the HuggingFace model automatically. For example, to perform an Int4 AWQ quantization, the following code will trigger the model quantization.


``` python
from tensorrt_llm.hlapi import QuantConfig, QuantAlgo

quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)

llm = LLM(<model-dir>, quant_config=quant_config)
```

## Parallelism

### Tensor Parallelism
It is easy to enable Tensor Parallelism in the HLAPI. For example, setting `parallel_config.tp_size=2` to perform a 2-way parallelism:

```python
from tensorrt_llm.hlapi import LLM

llm = LLM(<llama_model_path>,
          tensor_parallel_size=2)
```

### Pipeline Parallelism
Similar to Tensor Parallelism, you can enable Pipeline Parallelism in the HLAPI with following code:

```python
llm = LLM(<llama_model_path>,
          pipeline_parallel_size=4)
```

### Automatic Parallelism (in preview)

By simply enabling `auto_parallel` in the `LLM` class, TensorRT-LLM can parallelize the model automatically. For example, setting `world_size` to perform a 2-way parallelism:

``` python
from tensorrt_llm import LLM

llm = LLM(<llama_model_path>, auto_parallel=True, world_size=2)
```

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

## LLM pipeline configuration

### Build configuration
Apart from the arguments mentioned above, you can also customize the build configuration with the `build_config` class and other arguments borrowed from the lower-level APIs. For example:

```python
llm = LLM(<model-path>,
          build_config=BuildConfig(
            max_num_tokens=4096,
            max_batch_size=128,
            max_beam_width=4))
```

### Runtime customization
Similar to `build_config`, you can also customize the runtime configuration with the `runtime_config`, `peft_cache_config` or other arguments borrowed from the lower-level APIs. For example:


```python
from tensorrt_llm.hlapi import LLM, KvCacheConfig

llm = LLM(<llama_model_path>,
          kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.8))
```

### Tokenizer customization

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

### Disabling tokenizer
For performance considerations, you can disable the tokenizer by passing `skip_tokenizer_init=True` when creating `LLM`. In this case, `LLM.generate` and `LLM.generate_async` will expect prompt token ids as input. For example:

```python
llm = LLM(<llama_model_path>)
for output in llm.generate([[32, 12]]):
    print(output)
```

You will get something like:
```python
RequestOutput(request_id=1, prompt=None, prompt_token_ids=[1, 15043, 29892, 590, 1024, 338], outputs=[CompletionOutput(index=0, text='', token_ids=[518, 10858, 4408, 29962, 322, 306, 626, 263, 518, 10858, 20627, 29962, 472, 518, 10858, 6938, 1822, 306, 626, 5007, 304, 4653, 590, 4066, 297, 278, 518, 11947, 18527, 29962, 2602, 472], cumulative_logprob=None, logprobs=[])], finished=True)
```

Note that the `text` field in `CompletionOutput` is empty since the tokenizer is deactivated.

### Build caching
Although the HLAPI runs the engine building in the background, you can also cache the built engine to disk and load it in the next run to save the engine building time.

To enable the build cache, there are two ways to do it:

1. Use the environment variable: `export TLLM_HLAPI_BUILD_CACHE=1` to enable the build cache globally, and optionally export `TLLM_HLAPI_BUILD_CACHE_ROOT` to specify the cache root directory.
2. Pass the `enable_build_cache` to the `LLM` constructor

The build cache will reuse the built engine if all the building settings are the same, or it will rebuild the engine.

NOTE: The build cache monitors the model path and build settings, if you change the weights while keeping the same model path, the build cache will not detect the change and reuse the old engine.
