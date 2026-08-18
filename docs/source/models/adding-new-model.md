# Adding a New Model

This guide provides a step-by-step process for adding a new model to the PyTorch backend.

The following sections describe the steps for adding a new model:

1. [Prerequisites](#prerequisites)
1. [Step-by-Step Guide](#step-by-step-guide)
    1. [Model Configuration](#model-configuration)
    1. [Model Definition](#model-definition)
    1. [Weight Loading](#weight-loading)
    1. [Model Registration](#model-registration)
        1. [Core Models](#core-models)
        1. [Out-of-Tree Models](#out-of-tree-models)


## Prerequisites

Before you begin, ensure you have the following:
- A working installation of TensorRT-LLM. Follow the [Build from Source](../installation/build-from-source.md) install guide.

## Step-by-Step Guide

### Model Configuration

Suppose you want to support a new model named `MyModel`. If the model is already supported in Hugging Face Transformers, adapt the PyTorch modeling code and reuse the Hugging Face configuration class. For example, `tensorrt_llm/_torch/models/modeling_llama.py` was adapted from the Hugging Face [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) implementation and reuses its configuration class:

```python
from transformers import LlamaConfig
```

If the model is not registered in Hugging Face Transformers, define the configuration class in `configuration_mymodel.py` by following the Hugging Face [configuration_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py) implementation:

```python
from transformers.configuration_utils import PretrainedConfig

class MyConfig(PretrainedConfig):
    def __init__(self, ...):
        ...
```

### Model Definition

Remove any unnecessary code, such as training-specific code, and then rewrite the required PyTorch modules. For a typical Transformer decoder model, implement `modeling_mymodel.py` as follows:

```python
from typing import Optional

import torch
from torch import nn
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

from configuration_mymodel import MyConfig


class MyAttention(Attention):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: Optional[int] = None):
        # Use model_config to initialize the Attention module
        super().__init__(...)


class MyDecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: int):
        super().__init__()
        # Use model_config to initialize the submodules
        self.input_layernorm = ...
        self.self_attn = MyAttention(model_config, layer_idx)
        self.post_attention_layernorm = ...
        self.mlp = ...

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        # Define the forward computation of a single decoder layer
        ...


class MyModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(model_config)
        # Use model_config to initialize the submodules
        self.embed_tokens = ...
        self.layers = nn.ModuleList([
            MyDecoderLayer(model_config, layer_idx) for layer_idx in range(model_config.pretrained_config.num_hidden_layers)
        ])

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):
        # Define the forward computation of the model
        ...


class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(MyModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
```

`MyAttention` inherits from the TensorRT-LLM `Attention` module in `tensorrt_llm/_torch/modules/attention.py`. This inheritance makes the attention computation compatible with the PyTorch runtime. Adapt the module inputs as follows:

- The `attn_metadata` stores metadata from the batched input and key-value (KV) cache for the attention backend. The runtime creates and passes this metadata. Ensure that `attn_metadata` is passed correctly to the attention module.
- The input tensors, such as `input_ids`, `position_ids`, and `hidden_states`, use packed mode. The first dimension corresponds to the number of tokens in a batch.

Additionally, `MyDecoderLayer`, `MyModel`, and `MyModelForCausalLM` are subclasses of `DecoderLayer`, `DecoderModel`, and `DecoderModelForCausalLM`, respectively. The base classes define interfaces and provide generic scaffolding for defining model layers and loading weights.

Optionally, you can replace the native PyTorch modules with TensorRT-LLM implementations to enable features or improve performance:
- `Linear` (in `tensorrt_llm/_torch/modules/linear.py`): Enables tensor parallelism and quantization.
- `Embedding` (in `tensorrt_llm/_torch/modules/embedding.py`): Enables tensor parallelism for embedding.
- `RotaryEmbedding` (in `tensorrt_llm/_torch/modules/rotary_embedding.py`): Enables performant rotary embedding.
- `RMSNorm` (in `tensorrt_llm/_torch/modules/rms_norm.py`): Enables performant RMS norm.

For a concrete example, refer to `tensorrt_llm/_torch/models/modeling_llama.py`.

### Weight Loading

The base class `DecoderModelForCausalLM` provides a `load_weights` method that loads the weights from the checkpoint file and assigns them to the corresponding layers in the model. However, if the default method does not work for `MyModelForCausalLM`, you need to implement your own `load_weights`:

```python
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):

    def load_weights(self, weights: dict):
        # Define the weight loading logic
        ...
```

For example, the Hugging Face LLaMA model uses three linear layers for query, key, and value (Q/K/V) projections, resulting in three weight tensors in the checkpoint:

```python
>>> weights
{
    ...,
    "model.layers.0.self_attn.q_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.k_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.v_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    ...,
}
```

However, the TensorRT-LLM LLaMA model fuses the three layers into one linear layer:

```python
>>> llama.model.layers[0].self_attn.qkv_proj.weight.data
torch.Tensor([hidden_size * 3, hidden_size])
```

The `load_weights` method must collect the three weight tensors from the original checkpoint, concatenate them, and assign them to the fused linear layer. Tensor parallelism and quantization add complexity to this process. Use the predefined module-level `load_weights` methods, such as those for `Linear` and `Embedding`, when implementing the model-level `load_weights` method.

Overall, `load_weights` should handle any discrepancy between `MyModelForCausalLM` and the weights loaded from the checkpoint, so that `MyModelForCausalLM` can perform forward computation equivalent to the original model.

### Model Registration

Register the new model so that the PyTorch runtime can recognize it. Add the `register_auto_model` decorator to `MyModelForCausalLM`:

```python
from tensorrt_llm._torch.models.modeling_utils import register_auto_model

@register_auto_model("MyModelForCausalLM")
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
       ...
```

#### Core Models

To add the new model to the core models, place `modeling_mymodel.py` and, if needed, `configuration_mymodel.py` in `tensorrt_llm/_torch/models`. Then, import the modeling code in `tensorrt_llm/_torch/models/__init__.py`:

```python
from .modeling_mymodel import MyModelForCausalLM

__all__ = [
    ...,
    "MyModelForCausalLM",
]
```

#### Out-of-Tree Models

Alternatively, register the new model as an out-of-tree model to use it without modifying the TensorRT-LLM codebase. Place `modeling_mymodel.py` and, if needed, `configuration_mymodel.py` in your working directory, and import the modeling code in your script:

```python
from tensorrt_llm import LLM
import modeling_mymodel

def main():
    llm = LLM(...)

if __name__ == '__main__':
    main()
```

For an out-of-tree modeling example, refer to `examples/llm-api/out_of_tree_example`. The model is implemented in `modeling_opt.py`. Run the example with the following command:

```bash
python examples/llm-api/out_of_tree_example/main.py
```
