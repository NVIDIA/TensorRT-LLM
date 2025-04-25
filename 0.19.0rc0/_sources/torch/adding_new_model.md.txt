# Adding a New Model in PyTorch Backend

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Guide](#step-by-step-guide)
    1. [Model Configuration](#model-configuration)
    2. [Model Definition](#model-definition)
    3. [Weight Loading](#weight-loading)
    4. [Model Registration](#model-registration)
        1. [Core Models](#core-models)
        2. [Out-of-Tree Models](#out-of-tree-models)

## Introduction

This guide provides a step-by-step process for adding a new model in PyTorch Backend.

## Prerequisites

Before you begin, ensure you have the following:
- A working installation of TensorRT-LLM. Follow these [instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/build-from-source-linux.md).

## Step-by-Step Guide

### Model Configuration

Suppose you want to support a new model named `MyModel`. If the model is already supported in HuggingFace's transformers, you should bring the PyTorch modeling code and reuse HuggingFace's configuration class. For example, our `tensorrt_llm/_torch/models/modeling_llama.py` was adapted from HuggingFace's [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py); in the modeling code, we reuse the configuration class:

```python
from transformers import LlamaConfig
```

If the model is not registered in HuggingFace's transformers, you need to define the configuration class in your `configuration_mymodel.py` following HuggingFace's [configuration_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py):

```python
from transformers.configuration_utils import PretrainedConfig

class MyConfig(PretrainedConfig):
    def __init__(self, ...):
        ...
```

### Model Definition

Remove any unnecessary code (e.g., training-specific code), and then rewrite some PyTorch modules. For a typical Transformer decoder model, you need to implement your `modeling_mymodel.py` like this:

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
                input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
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

Note that `MyAttention` inherits from our `Attention` module (in `tensorrt_llm/_torch/modules/attention.py`), so that the attention computation is compatible with our PyTorch runtime. Related to this, module inputs should also be adapted:

- The `attn_metadata` stores the metadata from the batched input and KV cache for the attention backend. It is created by and passed from the runtime, and model developers need to ensure that `attn_metadata` is correctly passed to the attention module.
- The input tensors (i.e., `input_ids`, `position_ids`, `hidden_states`) are in the packed mode. The first dimension corresponds to the number of tokens in a batch.

Additionally, `MyDecoderLayer`, `MyModel`, and `MyModelForCausalLM` are subclasses of `DecoderLayer`, `DecoderModel`, and `DecoderModelForCausalLM` respectively. The base classes define interfaces and provide a generic scaffolding to define model layers, load weights, etc.

Optionally, you may replace the native PyTorch modules with our implementations to enable features or achieve higher performance:
- `Linear` (in `tensorrt_llm/_torch/modules/linear.py`): Enables tensor parallelism and quantization.
- `Embedding` (in `tensorrt_llm/_torch/modules/embedding.py`): Enables tensor parallelism for embedding.
- `RotaryEmbedding` (in `tensorrt_llm/_torch/modules/rotary_embedding.py`): Enables performant rotary embedding.
- `RMSNorm` (in `tensorrt_llm/_torch/modules/rms_norm.py`): Enables performant RMS norm.

For a concrete reference, check out `tensorrt_llm/_torch/models/modeling_llama.py`.

### Weight Loading

The base class `DecoderModelForCausalLM` provides a `load_weights` method that loads the weights from the checkpoint file and assigns them to the corresponding layers in the model. However, if the default method does not work for `MyModelForCausalLM`, you need to implement your own `load_weights`:

```python
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):

    def load_weights(self, weights: dict):
        # Define the weight loading logic
        ...
```

For example, Huggingface's LLaMA model uses three linear layers for Q/K/V projections, resulting in three weight tensors in the checkpoint:

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

However, our LLaMA model fuses the three layers into one linear layer:

```python
>>> llama.model.layers[0].self_attn.qkv_proj.weight.data
torch.Tensor([hidden_size * 3, hidden_size])
```

Hence, `load_weights` needs to collect the three weight tensors from the original checkpoint, concatenate them, and assign them to the fused linear layer. Considering tensor parallelism and quantization, the process would be more complicated. We recommend calling the predefined module-level `load_weights` (e.g., `Linear` and `Embedding`) when implementing your model-level `load_weights` method.

Overall, `load_weights` should handle any discrepancy between `MyModelForCausalLM` and the weights loaded from the checkpoint, so that `MyModelForCausalLM` can perform forward computation equivalent to the original model.

### Model Registration

The new model needs to be registered so that it can be recognized by the PyTorch runtime. The registration can be done simply by adding the `register_auto_model` decorator for `MyModelForCausalLM`:

```python
from tensorrt_llm._torch.models.modeling_utils import register_auto_model

@register_auto_model("MyModelForCausalLM")
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
       ...
```

#### Core Models

To add the new model to core models, `modeling_mymodel.py` (and potentially `configuration_mymodel.py`) should be placed in `tensorrt_llm/_torch/models`. Then, you need to import the modeling code in `tensorrt_llm/_torch/models/__init__.py`:

```python
from .modeling_mymodel import MyModelForCausalLM

__all__ = [
    ...,
    "MyModelForCausalLM",
]
```

#### Out-of-Tree Models

Alternatively, you can register the new model as an out-of-tree model, so that you can use the new model without touching the TensorRT-LLM codebase. To do so, place `modeling_mymodel.py` (and potentially `configuration_mymodel.py`) in your working directory, and import the modeling code in your script:

```python
from tensorrt_llm._torch import LLM
import modeling_mymodel

def main():
    llm = LLM(...)

if __name__ == '__main__':
    main()
```

We provide an out-of-tree modeling example in `examples/pytorch/out_of_tree_example`. The model is implemented in `modeling_opt.py` and you can run the example by:

```bash
python examples/pytorch/out_of_tree_example/main.py
```
