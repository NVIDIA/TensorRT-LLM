(graph-rewriting)=


# Graph Rewriting Module

TensorRT-LLM uses a declarative approach to define neural networks and contains
techniques to optimize the underlying graph.  It provides a wrapper similar to PyTorch's Module. When a user invokes the `forward` method, the layers are lowered to TensorRT's `ILayer`s and become part of an `INetworkDefinition`. The Graph Rewriting (GW) module can be used to manipulate the network at the `ILayer`/`INetworkDefinition` level.

## When to Use Graph Rewriting?

For network manipulation, there are two options in TensorRT-LLM:

1. **Module Rewriting:** This method modifies the members of `Module` instances before triggering the `forward` method (that is, creating the TensorRT graph). It works on the highest level of the network representation and facilitates the modification of sequences of operations (like modifying the GEMM + activation for SmoothQuant),

2. **Graph Rewriting:** Graph Rewriting manipulates TensorRT's `INetworkDefinition` after the `forward` method is triggered. It operates at a finer-grained `ILayer` level and can alter the structure across multiple Module instances. It is typically used for layer fusion.

Graph Rewriting (GW) is ideally used in the following conditions:

1. When only `ILayer`/`INetworkDefinition` is available,
2. When Module Rewriting would lead to nested control flow or scattered functionality.

## Graph Rewriting APIs

Several core APIs are provided for Graph Rewriting:

### Tensor-Related Methods

- `Tensor.get_parent`: Get the `ILayer` that produces this tensor,
- `Tensor.get_users`: Get the consumer `ILayer`s of this tensor,
- `replace_all_uses_with`: Replace this tensor with another tensor in all consumer `ILayer`s.

### FLayerInfo for Retrieving High-Level Information for a Functional

For all the layers located in `functional.py`, the original input information is missing once lowered to `INetworkDefinition`, especially for TensorRT plugins, which are opaque in the Python world. `FLayerInfo` holds their original information as a high-level signature containing inputs like `Tensor`, Python attributes, and more. There is a Network-wise singleton called `FLayerInfoMemo` to map each `ILayer` to its corresponding `FLayerInfo`.

For `FLayerInfo`:

- `FLayerInfo.replace_input_with`: Replace some input tensor with another tensor,
- `FLayerInfo.replace_output_uses_with`: Redirect the usage of the original output tensors to a set of new tensors.

For `FLayerInfoMemo`:

- `FLayerInfoMemo.instance()`: Get the singleton instance,
- `FLayerInfoMemo.get`: Get the corresponding `FLayerInfo` for an `ILayer`.

`FLayerInfo` remains consistent with the actual `ILayer` during GW, making it safe to use.

### Pattern and Pattern Manager

There are two kinds of patterns:

- `PatternRewriter`: Used for defining a rewriting pattern, which actually alters the network.
  - `match`: Match the pattern; returns true if a layer is matched,
  - `rewrite`: Manipulate a layer,
  - `match_and_rewrite`: Combines both `match` and `rewrite`, used for complex states that need to pass from `match` to `rewrite`.

- `PatternAnalyzer`: Used for defining an analysis pattern, which collects information from the network.
  - `match`: Match the pattern,
  - `analyze`: Perform analysis on a list of layers.

There are two managers for managing multiple `PatternRewriter` or `PatternAnalyzer`:

- `RewritePatternManager`:
  - `add`: Add a pattern with its label and benefit; the benefit specifies its privilege,
  - `get`: Get a pattern by label,
  - `rewrite`: Apply the rewriting patterns contained to a network.

- `AnalysisPatternManager`:
  - `add`: Add a pattern with its label and benefit; the benefit specifies its privilege,
  - `get`: Get a pattern by label,
  - `analyze`: Apply the analysis patterns contained to a network.

### @record_signature to Decorate Functionals Requiring FLayerInfo

The `@record_signature` decorator is used to record the `FLayerInfo` for a functional. While FLayerInfo is vital for GW when analyzing or rewriting certain functionals, it is used in an "add as needed" manner. If you are adding GW patterns, ensure that the functional requires the `@record_signature` decorator.

## Classical Workflow

There are specific routines for defining a GW pattern. Let's start with a simple example: replacing a sum layer with a subtract layer, which can also be found in the `test_graph_rewriting.py` file.

```python
class NaivePatternRewriter_ReplaceAddWithSub(PatternRewriter):

    def __init__(self):
        super().__init__('replace_add_with_sub',
                         root_layer={trt.LayerType.ELEMENTWISE},
                         separate_match_rewrite=True)

    def match(self, layer: Layer):
        # The rewriter will stop at the first matched layer, and then the Rewriter will enter the rewrite() to do the rewriting.
        return layer.as_layer().op == trt.ElementWiseOperation.SUM

    def rewrite(self, layer: Layer) -> None:
        # The layer here should be an Elementwise_SUM layer.
        with net_guard(layer.network):
            # There are several stages to replace some subgraph with another subgraph:

            # Stage 1: Get the input tensors and output tensors of the subgraph to replace.
            # - For Elementwise_SUM, there are two inputs and one output.
            a, b = layer.get_inputs(0, 1)
            o = layer.get_outputs(0)[0]

            # Stage 2: Create a new subgraph that takes the old one's inputs.
            # - Here we insert an Elementwise_SUB layer, and 'c' is the output.
            c = a - b

            # Stage 3: Redirect all the layers depending on the outputs of the old subgraph to the new subgraph's.
            # - After this, the SUM becomes dangling and will be pruned by TensorRT when building the engine.
            # - Note that there is no API in TensorRT python to remove a layer explicitly; `replace_all_uses_with` is the only way to "remove" a layer.
            o.replace_all_uses_with(c)

            # Stage 4: Mark all the layers in the old subgraph as removed.
            # - This helps the PatternRewriter to skip the removed layers.
            layer.mark_as_removed()
```

In this example, we deal with `ILayer` rather than Plugins, so `FLayerInfo` is unnecessary. As illustrated in the `rewrite` method, there are four stages that are shared across nearly all rewrite patterns.

Note that in GW, we **NEVER** rewrite a layer directly. Instead, we do it in two steps: first, create another layer with the same input and deprive all the users of the original outputs, redirecting them to the outputs of the new layers. In this way, the old layer will be dangling and pruned automatically by TensorRT during the engine building phase. This is a limitation of TensorRT since remove-layer-like APIs are not available in Python.

In Stage 2, we rely on operators and layers commonly used during the network building phase. Ideally, you can replace them with any network structure during GW.

For the usage of `FLayerInfo`, let's rewrite the `gpt_attention` to enable the `remove-padding` feature. `gpt_attention` is actually

 a TensorRT plugin, so we need `FLayerInfo` to hold the original Tensor-wise inputs to help create new `gpt_attention` layers.

```python
class GPTAttentionPluginRemovePaddingRewritePass(PatternRewriter):

    def __init__(self):
        super().__init__('gpt_attention_plugin_remove_padding',
                         root_layer={trt.LayerType.PLUGIN_V2})

    def match_and_rewrite(self, layer: Layer) -> bool:
        if layer.as_layer().type != trt.LayerType.PLUGIN_V2 or \
                layer.as_layer().plugin.plugin_namespace != 'tensorrt_llm' or \
                layer.as_layer().plugin.plugin_type != 'GPTAttention':
            return False

        # Retrieve the FLayerInfo
        flayer = FLayerInfoMemo.instance().get(layer.name)
        assert flayer
        # Although the layer is a plugin, which is a black box, we get some high-level input information from the FLayerInfo.
        tensor_input: Tensor = flayer.get_input('qkv')
        if tensor_input.shape[0] == 1:  # Already in remove-padding mode
            return False

        # Some information could be passed in from external
        assert self.args is not None, "args should be passed in from RewritePatternManager.rewrite()"
        batch_size, in_len, hidden_size = self.args['batch_size'], self.args['in_len'], self.args['hidden_size']

        with net_guard(layer.network):
            new_inputs = flayer.clone_inputs()

            # Step 1: Create new inputs and replace the original arglist.
            input = Tensor(
                name='qkv',
                dtype=trt.float16,
                shape=(1, batch_size * in_len, hidden_size),
            )
            new_inputs['qkv'] = input

            # Step 2: Create a new plugin instance.
            new_outs = gpt_attention(**new_inputs)

            # Step 3: Deprive all the users of the old plugin instance.
            flayer.replace_outputs_uses_with(layer.network, new_outs)

            # Step 4: Remove the old plugin instance.
            layer.mark_as_removed()

        return True
```

This is quite similar to the first example, with the focus on the `FLayerInfo` part. Through the code below, we can get the original inputs of this layer, enabling us to alter the inputs related to remove-padding and create a new layer to replace it.

```python
flayer = FLayerInfoMemo.instance().get(layer.name)
assert flayer
```

```python
new_inputs = flayer.clone_inputs()

# Step 1: Create new inputs and replace the original arglist.
input = Tensor(
    name='tensor',
    dtype=trt.float16,
    shape=(1, batch_size * in_len, hidden_size),
)
new_inputs['tensor'] = input

# Step 2: Create a new plugin instance.
new_outs = gpt_attention(**new_inputs)
```

For real examples, please refer to the `FuseAttentionWithBiasPass` in the `graph_rewriting.py`.
