# TensorRT-LLM Model Weights Loader

## Overview

The weights loader is designed for easily converting and loading external weight checkpoints into TensorRT-LLM models.

## Workflow

Weight checkpoints can be generated from all sources, and may have different naming and data layouts compared to TRT-LLM's requirements. E.g.:

```bash
# HuggingFace LLaMA checkpoints
{
    "model.embed_tokens.weight": torch.Tensor([vocab_size, hidden_size])
    "model.layers.0.input_layernorm.weight": torch.Tensor([hidden_size]),
    "model.layers.0.mlp.down_proj.weight": torch.Tensor([hidden_size, inter_size]),
    "model.layers.0.mlp.gate_proj.weight": torch.Tensor([inter_size, hidden_size]),
    "model.layers.0.mlp.up_proj.weight": torch.Tensor([inter_size, hidden_size]),
    "model.layers.0.post_attention_layernorm.weight": torch.Tensor([hidden_size]),
    "model.layers.0.self_attn.q_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.k_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.v_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.o_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    ...,
}
# TensorRT-LLM expected weights
{
    "transformer.vocab_embedding.weight": torch.Tensor([vocab_size, hidden_size])
    "transformer.layers.0.input_layernorm.weight": torch.Tensor([hidden_size]),
    "transformer.layers.0.mlp.down_proj.weight": torch.Tensor([hidden_size, inter_size]),
    "transformer.layers.0.mlp.gate_proj.weight": torch.Tensor([inter_size, hidden_size]),
    "transformer.layers.0.mlp.up_proj.weight": torch.Tensor([inter_size, hidden_size]),
    "transformer.layers.0.post_layernorm.weight": torch.Tensor([hidden_size]),
    "transformer.layers.0.attention.qkv.weight": torch.Tensor([hidden_size * 3, hidden_size]), # Different layout
    "transformer.layers.0.attention.dense.weight": torch.Tensor([hidden_size, hidden_size]),
    ...,
}
```

Conversion means converting the dictionary of `{external_keys:external_weights}` into `{tllm_keys:tllm_weights}`, it includes changing the naming logic and data layouts, and is contains of the following parts:

1. Translate a TRT-LLM parameter name into external-format name(s).
2. Loading tensor slice(s) according to the translated names.
3. Postprocess the tensor(s) into target layout.

### Translator

TRT-LLM parameter names are translated in units of sections divided by dots. E.g.:

|    TensorRT-LLM key     | `transformer` |.| `layers` |.| `0` |.| `attention` |.|  `dense` |.| `weight` |
| :---------------------: | :-----------: |-| :------: |-|:---:|-| :---------: |-| :------: |-| :------: |
| Translated external key |    `model`    |.| `layers` |.| `0` |.| `self_attn` |.| `o_proj` |.| `weight` |

The mapping between TRT-LLM keywords and HF keywords are described in `tllm_to_externel_key_dict` of `ModelWeightsLoader` class object. \
If any of the mappings has one-to-multiple corresponding, the translated key will get multiplied accordingly. E.g.:

|         TensorRT-LLM key and related keyword mapping         | Translated external keys |
| :----------------------------------------------------------: | :----------------------: |
| `transformer.layers.0.attention.qkv.weight`<br>`{"qkv":[q_proj, k_proj, v_proj]}` | `model.layers.0.self_attn.q_proj.weights`<br>`model.layers.0.self_attn.k_proj.weights`<br>`model.layers.0.self_attn.v_proj.weights`|
|   `transformer.layers.0.mlp.fc.weight`<br>`{"weight":[qweight, qzeros, scales]}`  | `model.layers.0.mlp.gate_proj.qweight`<br>`model.layers.0.mlp.gate_proj.qzeros`<br>`model.layers.0.mlp.gate_proj.scales`|

The default `tllm_to_externel_key_dict` is based on HF LLaMA as:

```python
class ModelWeightsLoader:
    def __init__(self, model_dir, customized_key_dict: dict = {}) -> None:
        ...
        self.tllm_to_externel_key_dict = {
            "transformer": "model",
            "vocab_embedding": "embed_tokens",
            "lm_head": "lm_head",
            "ln_f": "norm",
            "attention": "self_attn",
            "qkv": ["q_proj", "k_proj", "v_proj"],
            "dense": "o_proj",
            "gate": "up_proj",
            "proj": "down_proj",
            "fc": "gate_proj",
            "input_layernorm": "input_layernorm",
            "post_layernorm": "post_attention_layernorm",
        }
        self.tllm_to_externel_key_dict.update(customized_key_dict)
        ...
```

It can be updated through passing `customized_key_dict` when initializing `ModelWeightsLoader`.

The dictionary will also get updated according to the layer classes. When iterating over parameters,
if the layer class has attribute `tllm_to_externel_key_dict`, for keywords exist both in the default one and the layer-specified one,
the weight loader will translate according to the layer attribute with higher priority.
This can enable the support for different quantization precisions automatically.


### Loading function

The loading function can load an arbitrary tensor slice according to its `key`, `tp_size`, `tp_dim` and `tp_rank`.

The template for loading function is as following.

```python
def load_tensor(self, key, tp_size, tp_dim, tp_rank):
    # Retrieve file pointer index
    if key in self.shard_map:
        ptr_idx = self.shard_map[key]
    else:
        return None

    # Load tensor from the corresponding shard
    if self.format == ModelWeightsFormat.SAFETENSORS:
        tensor = self.shards[ptr_idx].get_slice(key)
        tensor_shape = tensor.get_shape()
    else:
        ...

    # Shard and return a tensor slice
    slice_shape = ...
    return tensor[slice_shape]
```

When initializing the `ModelWeightsLoader` object, the file format will be derived from `model_dir` through `detect_format`. The following formats are supported for now:

 * Directory contains or file named `*.safetensors` (Recommended, has better performance)
 * Directory contains or file named `*.bin`
 * Directory contains or file named `*.pth`

To support other formats or in-memory loaded models, the format need to be claimed in `ModelWeightsFormat`, `detect_format()`, `preload()` and `load_tensor()`.

### Postprocessing functions

After translation and loading, a TRT-LLM key will become a tensor or a list of tensors, which is the input of postprocessing functions. \
Operations including QKV concatenating, MoE weight stacking and weight-only quantization can be handled here.
The template of postprocessing function is:

```python
# Example for 1-1 weights mapping
class CustomizedModuleA(Module):
    def __init__(...):
        super().__init__(...)
        ...
        self.tp_dim = 0    # Need to set or inherit from parent class

    def postprocess(self, tllm_key, weights, **kwargs):
        weights = proc(weights)
        return {tllm_key: weights}

# Example for multiple-multiple weights mapping
class CustomizedModuleB(Module):
    def __init__(...):
        super().__init__(...)
        ...
        self.tp_dim = 0    # Need to set or inherit from parent class
        # The default value of "weights" in tllm_to_externel_key_dict will be override
        self.tllm_to_externel_key_dict = {"weight": ["qweight", "qzeros", "scales"]}

    def postprocess(self, tllm_key, weights, **kwargs):
        # Skipped the postprocess of zeros and weights_scaling_factor
        # They are loaded in the postprocess of weight
        config = kwargs.get("config", None) # Passed through kwargs by default
        if not tllm_key.endswith("weight"):
            return {}
        # The order in weights is defined in tllm_to_externel_key_dict
        qweight, qzeros, scales = weights
        proccessed_weight, proccessed_zeros = proc(qweight, qzeros, config.num_heads)
        return {
            tllm_key: proccessed_weight,
            tllm_key.replace("weight", "zeros"): proccessed_zeros,
            tllm_key.replace("weight", "weights_scaling_factor"): scales,
        }
```

## Examples

The `ModelWeightsLoader` class can support different models with the following levels:

### Natively supported models
For models with native support, users can call the default weight loader without any other operations.
```python
# Using the model weights loader for LLaMA
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
loader = ModelWeightsLoader(external_checkpoint_dir)
loader.generate_tllm_weights(trtllm_model)
```
For calibration-free quantization precisions, passing a properly quantized `trtllm_model` will let the weight loader load at the given precision accordingly. The configurations will be read from `trtllm_model.config` automatically. For now, LLaMA family models using the default `tllm_to_externel_key_dict` is supported natively.

### Models with customized key names
For models with different naming logic, users can still call the default weight loader with `customized_key_dict` specified.
```python
# Using the model weights loader for the LLM part of LLaVA
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
llava_dict = {
    "transformer": "language_model.model",
    "lm_head": "language_model.lm_head"
}
loader = ModelWeightsLoader(external_checkpoint_dir, llava_dict)
loader.generate_tllm_weights(trtllm_model)
```
Users need to specify the different part from the default `tllm_to_externel_key_dict`. The loader still have support across different precisions.
The support for LLaVA and Exaone is in `LLaMAForCausalLM.from_hugging_face()` of [model.py](../../../tensorrt_llm/models/llama/model.py), and can also be taken as examples.

### Models with customized weight layout
For models with different weight layout, users can write the conversion loop explicitly and do customized operations.
```python
# Using the model weights loader for BLOOM
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
bloom_dict = {
    "transformer": "",
    "layers": "h",
    "ln_f": "ln_f",
    "lm_head": "word_embeddings",
    "ln_embed": "word_embeddings_layernorm",
    "vocab_embedding": "word_embeddings",
    "attention": "self_attention",
    "qkv": "query_key_value",
    "dense": "dense",
    "fc": "dense_h_to_4h",
    "proj": "dense_4h_to_h",
    "post_layernorm": "post_attention_layernorm",
}
loader = ModelWeightsLoader(external_checkpoint_dir, bloom_dict)
# See ModelWeightsLoader.generate_tllm_weights()
loader.update_key_mapping(trtllm_model)
tllm_weights = {}
for tllm_key, _ in tqdm(trtllm_model.named_parameters()):
    if tllm_key.endswith("qkv"):
        # Passing the callable handle
        tllm_weights.update(loader.load(tllm_key, preprocess=customized_preprocess))
    else:
        tllm_weights.update(loader.load(tllm_key))
loader.fill(tllm_weights)
```
This will apply `preprocess` after `load_tensor()` and before `postprocess`, and demonstrates how to convert the loaded shard into default HF layout. The loader still have support for precisions quantized from FP16/BF16 (e.g. INT8-wo/INT4-wo), the other precisions may require special operations, and can be addressed inside the `preprocess` function.
The support for Qwen-1 is in `QWenForCausalLM.from_hugging_face()` of [model.py](../../../tensorrt_llm/models/qwen/model.py), and can also be taken as example.

### Fully customized
If the model weights loader cannot satisfy the requirements, users can write the conversion loop totally on their own.
```python
tllm_weights = {}
for tllm_key, param in tqdm(trtllm_model.named_parameters()):
    # Load from external checkpoints
    # The load_tensor() function can also be called here
    tensor = ...
    # Convert tensor and set the values according to the config
    if trtllm_model.config.quantization.quant_algo == xxx:
        ...
    else:
        ...
    param.value = tensor
```
In this mode, every precision require user's own support.

## Trouble shooting
The weights loader is an experimental feature for now, and is enabled for LLaMA family models and Qwen models by default.

If users are encountered with failure caused by `ModelWeightsLoader`, a workaround is passing environmental variable `TRTLLM_DISABLE_UNIFIED_CONVERTER=1` to disable the model weights loader and fallback to the legacy path.

This workaround will be removed in future version after the LLaMA/Qwen weights conversion is stable.
