(add-model)=

# Adding a Model

```{caution}
The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.
```

> [!WARNING]
> This page describes the **legacy** TensorRT engine-build workflow for adding a model.
> For new projects, follow the PyTorch backend guide: [Adding a New Model](../../torch/adding_new_model.md)
> (also published as [models/adding-new-model](../../models/adding-new-model.md)).

This document describes how to add a typical decoder-only model in the **legacy** TensorRT LLM engine workflow.

## Step 1. Write Modeling Part

TensorRT LLM provides different levels of APIs:

- Low-level functions, for example, `concat`, `add`, and `sum`.
- Basic layers, such as, `Linear` and `LayerNorm`.
- High-level layers, such as, `MLP` and `Attention`.
- Base class for typical decoder-only models, such as, `DecoderModelForCausalLM`.

1. Create a model directory in `tensorrt_llm/models`, for example `my_model`.
2. Write a `model.py` with TensorRT LLM's APIs

```python
class MyDecoderLayer(Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.config = config
        self.input_layernorm = LayerNorm(...)
        self.attention = Attention(...)
        self.post_layernorm = LayerNorm(...)
        self.mlp = MLP(...)

    def forward(self, hidden_states, ...):
        # decoder layer forward
        return hidden_states

class MyModel(Module):
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.vocab_embedding = Embedding(...)
        self.layers = DecoderLayerList(MyDecoderLayer, config)
        self.ln_f = LayerNorm(...)

    def forward(self, input_ids, ...):
        # model forward
        return hidden_states


class MyModelForCausalLM(DecoderModelForCausalLM):
    def __init__(self, config: PretrainedConfig):
        transformer = MyModel(config)
        lm_head = ColumnLinear(...)
        super().__init__(config, transformer, lm_head)
```


## Step 2. Implement Weight Conversion

The weights from source framework need to be converted and bound to the new added TensorRT LLM model. Here is an example of converting HuggingFace weights:

```python
class MyModelForCausalLM(DecoderModelForCausalLM):
    @classmethod
    def from_hugging_face(
            cls,
            hf_model_dir,
            dtype='float16',
            mapping: Optional[Mapping] = None) -> MyModelForCausalLM
        # create a TensorRT LLM MyModelForCausalLM model object
        # convert HuggingFace checkpoint to TensorRT LLM expected weights dict
        # load the weights to MyModelForCausalLM object
```

Historically, an optional `convert_checkpoint.py` script lived under `examples/my_model/` for offline weights conversion. That script and the `trtllm-build` CLI were removed with the TensorRT backend; do not add new convert/build scripts.

## Step 3. Register New Model

Please register the new model class `MyModelForCausalLM` in `tensorrt_llm/models/__init__.py`.

## Step 4. Verify New Model

The legacy verification flow depended on per-model `convert_checkpoint.py`, the removed `trtllm-build` CLI, and deleted helpers `examples/run.py` / `examples/summarize.py`. Those entry points no longer exist.

For current end-to-end verification on the PyTorch backend, follow [Adding a New Model](../../torch/adding_new_model.md) and use `trtllm-serve` or the LLM API instead of building a TensorRT engine.

## Reference

This page is retained next to the legacy [workflow](./workflow.md) and [checkpoint](./checkpoint.md) documents. For the supported path to add a model, see [torch/adding_new_model.md](../../torch/adding_new_model.md).
