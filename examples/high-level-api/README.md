# High-level API
We are working on a high-level API for LLM workflow, which is still in incubation and may change later.
Here we show you a preview of how it works and how to use it.

Note that the APIs are not stable and only support the LLaMA model. We appreciate your patience and understanding as we improve this API.

You can refer to [llm_examples.py](llm_examples.py) for all of the examples, and run it with the [run_examples.sh](./run_examples.sh) script, the command is as follows:

```sh
./run_examples.sh <llama-model-path>
```

For 7B, 13B models those could be held in a single GPU, it should run all the examples automatically and print the results.

For larger models, such as LLaMA v2 70B, at least two H100/A100 cards are required, the following command could be used to start a parallel task with Tensor Parallelism enabled.

``` sh
python3 llm_examples.py --task run_llm_on_tensor_parallel \
    --prompt="<prompt>" \
    --hf_model_dir=<llama-model-path>
```

## Model preparation

Given its popularity, the TRT-LLM high-level API chooses to support HuggingFace format as start point, to use the high-level API on LLaMA models, you need to run the following conversion script provided in [transformers/llama](https://huggingface.co/docs/transformers/main/model_doc/llama) or [transformers/llama2](https://huggingface.co/docs/transformers/main/model_doc/llama2) to convert the Meta checkpoint to HuggingFace format.

For instance, specially for a LLaMA2 7B model, the official way to retrieve the model is to visit the [LLaMA2 7B model page](https://huggingface.co/transformers/llama2-7B), normally you need to submit a request for the the model file.
For a quick start, you can also download the model checkpoint files directly from [modelscope.cn](https://www.modelscope.cn/models/shakechen/Llama-2-7b/files), the command is as follows:

``` sh
git clone https://www.modelscope.cn/shakechen/Llama-2-7b.git
```

And to convert the checkpoint files, a script from transformers is required, thus please also clone the transformers repo with the following code:

```sh
git clone https://github.com/huggingface/transformers.git
```

Finally, the command to convert the checkpoint files to HuggingFace format is as follows:

``` sh
python <transformers-dir>/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir Llama-2-7b --model_size 7B --output_dir llama-hf-7b
```

That should produce a HuggingFace format model in `./llama-hf-7b`, which could be used by the high-level API.

## Basic usage
To use the API, import the `LLM` and `ModelConfig` from the `tensorrt_llm` package and create an LLM object with a HuggingFace model directly.
For example:

``` python
from tensorrt_llm import LLM, ModelConfig

config = ModelConfig(model_dir=<llama_model_path>)
llm = LLM(config)
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
GenerationOutput(text="with a picture.\nI'm a writer, but I'm also a photographer.")
```

You can also dump the runtime engine to disk, and load from the engine file directly in the next run to save the engine building time from the HuggingFace model.

``` python
# dump the llm
llm.save(<engine-path>)

# next time
config = ModelConfig(model_dir=<engine-path>)
llm = LLM(config)
```

In other words, the `model_dir` could accept either a HugggingFace model or a built TensorRT-LLM engine, the `LLM()` will do the rest work silently.


## Quantization

By simply setting several flags in the ModelConfig, TensorRT-LLM can quantize the HuggingFace model automatically. For example, to perform an Int4 AWQ quantization, the following code will trigger the model quantization.


``` python
config = ModelConfig(model_dir=<llama_model_path>)

config.quant_config.init_from_description(
                       quantize_weights=True,
                       use_int4_weights=True,
                       per_group=True)

llm = LLM(config)
```

## Asynchronous generation
With the high-level API, you can also perform asynchronous generation with the `generate_async` method. For example:

```python
config = ModelConfig(model_dir=<llama_model_path>)

llm = LLM(config, async_mode=True)

async for output in llm.generate_async(<prompt>, streaming=True):
    print(output)
```

When the `streaming` flag is set to `True`, the `generate_async` method will return a generator that yields the token results as soon as they are available. Otherwise, it will return a generator that yields the final results only.


## Customization

By default, the high-level API uses transformers’ `AutoTokenizer`. You can override it with your own tokenizer by passing it when creating the LLM object. For example:

```python
llm = LLM(config, tokenizer=<my_faster_one>)
```

The LLM() workflow should use your tokenizer instead.

It is also possible to input token IDs directly without Tokenizers with the following code:

``` python
llm = LLM(config, enable_tokenizer=False)

for output in llm.generate([32, 12]): ...
```
