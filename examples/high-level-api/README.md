# High-level API
We are working on a high-level API for LLM workflow, which is still in incubation and may change later.
Here we show you a preview of how it works and how to use it.

Note that the APIs are not stable and only support LLaMA model on single-node-single-gpu with limited optimization.
We appreciate your patience and understanding as we improve this API.

## Basic usage
To use the API, import the `LLM` and `ModelConfig` from `tensorrt_llm` package and create an LLM object with a HuggingFace model directly.
For example:

``` python
from tensorrt_llm.hlapi.llm import LLM, ModelConfig

config = ModelConfig(model_dir=<llama_model_path>)
llm = LLM(config)
```

It will trigger TRT-LLM engine building in the backend, and create HuggingFace tokenizer by default to support an end-to-end generation.

To generate text, use the `__call__` method of `LLM` object directly with a batch of prompts, for example:

``` python
prompts = ["To tell a story"]
for output in llm(prompts):
    print(output)
```

The output might be something like:

``` python
GenerationPiece(index=0, text="with a picture.\nI'm a writer, but I'm also a photographer.")
```

You can also dump the runtime engine to disk, and load from the engine file directly in the next run to save the engine building time from HuggingFace model.

``` python
# dump the llm
llm.save(<engine-path>)

# next time
config = ModelConfig(model_dir=<engine-path>)
llm = LLM(config)
```


## Customization

By default, the high-level API uses transformers’ `AutoTokenizer`. You can override it with your own tokenizer by passing it when creating the LLM object. For example:

```python
llm = LLM(config, tokenizer=<my_faster_one>)
```

Besides tokenizer, you can also override the model by passing in an in-memory model, that will save much efforts for a highly-customed model.

``` python
class MyModel(Module): ...

my_model = MyModel(...)
llm = LLM(model=my_model)
```
