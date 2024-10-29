# TRT-LLM Python Plugin

TRT-LLM provides an python plugin interface for users to integrate plugin to TRT-LLM with pure python.

+ `openai_triton_plugin`: plugin package
+ `build_lookup.py`: Build a TensorRT engine with TRT-LLM plugin
+ `run_lookup.py`: Run the engine and compare the result with pytorch

## Plugin Definition

The following code gives a simple example to create a look up plugin. We only need to do a few things to define a TRT-LLM plugin.

1. Inherit the `PluginBase`
2. Register the plugin class to TRT-LLM by using `@trtllm_plugin("your_plugin_name")`
3. Define `__init__` function and initialize base class
4. Define shape & dtype inference function
5. Define the compute flow

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

## Adding a TRT-LLM Plugin to Network

You only needs to instance a plugin object and then call it with `tensorrt_llm.Tensor` as input arguments.

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

Since we do plugin registration when importing the custom TRT-LLM plugin, so there would be some convention on code structure for users to register the plugin at runtime.

```
plugin_lib
├──__init__.py
├──lookup_plugin.py
└──lookup_kernel.py
```

Say we have such plugin package. The `__init__.py` should import all the plugins in the plugin packages, so that the plugin users only need to import the plugin package to register all the plugin, and no need to manually import them.

```python
# __init__.py
from .lookup_plugin import LookUpPlugin

__all__ = ["LookUpPlugin"]
```

## Deserialize an Engine with TRT-LLM Plugin

During the deserialization, TRT needs to find the user defined plugin. Thus, we need to import the plugin once to register them. If the plugin has the recommended code structure, users only need to import that package to register all the custom plugin.

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
