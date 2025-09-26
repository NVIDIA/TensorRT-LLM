# TensorRT LLM Python Plugin

TensorRT LLM provides a Python plugin interface to integrate TensorRT LLM with pure Python.

+ `openai_triton_plugin`: plugin package
+ `build_lookup.py`: Build a TensorRT engine with TensorRT LLM Python plugin
+ `run_lookup.py`: Run the engine and compare the result with PyTorch

## Plugin Definition

The following code shows how to create a look-up plugin.
We only need to do a few things to define a TensorRT LLM plugin.

1. Inherit the `PluginBase`.
2. Register the plugin class to TensorRT LLM by using `@trtllm_plugin("your_plugin_name")`.
3. Define an `__init__` function and initialize the base class.
4. Define a shape and dtype inference function.
5. Define the compute flow.

```python
@trtllm_plugin("TritonLookUp")
class LookUpPlugin(PluginBase):

    def __init__(self, use_torch_tensor, fp32_output):
        super().__init__()
        self.use_torch_tensor = use_torch_tensor
        self.fp32_output = fp32_output

    def shape_dtype_inference(self, inputs: Sequence[SymTensor]) -> SymTensor:
        shape = inputs[1].shape
        shape[0] = inputs[0].shape[0] + inputs[1].shape[0] - inputs[1].shape[0]
        return SymTensor(
            inputs[1].dtype if not self.fp32_output else torch.float32, shape)

    def forward(self, inputs: Sequence[TensorWrapper],
                outputs: Sequence[TensorWrapper]):
        assert len(inputs) == 2
        assert inputs[0].dtype in [torch.int32 or torch.int64]
        assert inputs[1].dtype in [torch.float32, torch.float16, torch.bfloat16]
        assert (self.fp32_output and outputs[0].dtype
                == torch.float32) or outputs[0].dtype == inputs[1].dtype

        x = inputs[0]
        y = inputs[1]
        z = outputs[0]
        if self.use_torch_tensor:
            x = convert_to_torch_tensor(x)
            y = convert_to_torch_tensor(y)
            z = convert_to_torch_tensor(z)
        MAX_BLOCK_NUM = 65536
        MAX_BLOCK_SIZE = 512
        grid = lambda meta: (min(MAX_BLOCK_NUM, x.shape[0]) * min(
            MAX_BLOCK_SIZE, y.shape[1]), )
        lookup_kernel[grid](x, y, z, y.shape[0], y.shape[1], x.shape[0])

```

## Adding a TensorRT LLM Plugin to a Network

You only need an instance of the plugin object and then call it with `tensorrt_llm.Tensor` as input arguments.

```python
builder = tensorrt_llm.Builder()
network = builder.create_network()
with tensorrt_llm.net_guard(network):
    x = Tensor(name='x',
               shape=index_shape,
               dtype=tensorrt_llm.str_dtype_to_trt('int32'))
    y = Tensor(name='y',
               shape=(vocab_size, n_embed),
               dtype=torch_dtype_to_trt(dtype))

    def lookup(x, y):
        lookup_plugin = LookUpPlugin(False)
        return lookup_plugin(x, y)

    output = lookup(x, y)
    output.mark_output('output', torch_dtype_to_str(dtype))
```

## Plugin Code Structure

Because TensorRT LLM performs plugin registration when importing the custom TensorRT LLM plugin, there are some code structure conventions to register the plugin at runtime.

```text
plugin_lib
├──__init__.py
├──lookup_plugin.py
└──lookup_kernel.py
```

The `__init__.py` file imports all the plugins in the plugin package.
With this convention, users only need to import the plugin package to register the plugins and do not need to manually import them.

```python
# __init__.py
from .lookup_plugin import LookUpPlugin

__all__ = ["LookUpPlugin"]
```

## Deserialize an Engine with TensorRT LLM Plugin

During deserialization, TensorRT needs to find the user-defined plugin. Thus, we need to import the plugin once to register them. If the plugin follows the code structure convention, users only need to import that package to register all the custom plugins.

```python
from tensorrt_llm.runtime.session import Session, TensorInfo

import openai_triton_plugin  # isort: skip

if __name__ == "__main__":

    def run_engine(dtype):
        output_dir = Path('tmp') / torch_dtype_to_str(dtype)

        engine_path = output_dir / "lookup.engine"

        with engine_path.open('rb') as f:
            session = Session.from_serialized_engine(f.read())
```
