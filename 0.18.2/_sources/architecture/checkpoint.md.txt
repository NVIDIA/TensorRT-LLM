# TensorRT-LLM Checkpoint

## Overview

The earlier versions (pre-0.8 version) of TensorRT-LLM were developed with a very aggressive timeline. For those versions, emphasis was not put on defining a unified workflow. Now that TensorRT-LLM has reached some level of feature richness, the development team has decided to put more effort into unifying the APIs and workflow of TensorRT-LLM. This file documents the workflow around TensorRT-LLM checkpoint and the set of CLI tools to generate checkpoint, build engines, and evaluate engines.

There are three steps in the workflow:

1. Convert weights from different source frameworks into TensorRT-LLM checkpoint.
2. Build the TensorRT-LLM checkpoint into TensorRT engines with a unified build command.
3. Load the engines to TensorRT-LLM model runner and evaluate with different evaluation tasks.

```
NeMo -------------
                  |
HuggingFace ------
                  |   convert                             build                    load
Modelopt ---------  ----------> TensorRT-LLM Checkpoint --------> TensorRT Engine ------> TensorRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
```

## Prepare the TensorRT-LLM Checkpoint

TensorRT-LLM aims at supporting different sources:

1. Trained models from NVIDIA NeMo, Microsoft DeepSpeed, and JAX
2. Quantized models from NVIDIA Modelopt
3. Popular models from HuggingFace

TensorRT-LLM defines its own checkpoint format. A checkpoint directory includes:

1. One config `json` file, which contains several model hyper-parameters.
2. One or several rank weights files, each file contains a dictionary of tensors (weights).
The different files are loaded by different ranks in a multi-GPU (multi-process) scenario.

### Config

| Field                                  | Type       | Default Value       |
| :------------------------------------- | :--------- | :------------------ |
| architecture                           | string     | mandatory           |
| dtype                                  | string     | mandatory           |
| logits_dtype                           | string     | 'float32'           |
| vocab_size                             | int        | mandatory           |
| max_position_embeddings                | int        | null                |
| hidden_size                            | int        | mandatory           |
| num_hidden_layers                      | int        | mandatory           |
| num_attention_heads                    | int        | mandatory           |
| num_key_value_heads                    | int        | num_attention_heads |
| hidden_act                             | string     | mandatory           |
| intermediate_size                      | int        | null                |
| norm_epsilon                           | float      | 1e-5                |
| position_embedding_type                | string     | 'learned_absolute'  |
| mapping.world_size                     | int        | 1                   |
| mapping.tp_size                        | int        | 1                   |
| mapping.pp_size                        | int        | 1                   |
| quantization.quant_algo                | str        | null                |
| quantization.kv_cache_quant_algo       | str        | null                |
| quantization.group_size                | int        | 64                  |
| quantization.has_zero_point            | bool       | False               |
| quantization.pre_quant_scale           | bool       | False               |
| quantization.exclude_modules           | list       | null                |

`mapping.world_size` means `mapping` is a dictionary containing the `world_size` sub field.

```json
{
    "architecture": "OPTForCausalLM",
    "mapping": {
        "world_size": 1
    }
}
```

Supported quantization algorithm list:

- W8A16
- W4A16
- W4A16_AWQ
- W4A8_AWQ
- W4A16_GPTQ
- FP8
- W8A8_SQ_PER_CHANNEL

Supported KV cache quantization algorithm list:

- FP8
- INT8

The config field is extensible, a model could add its own specific config fields.
For example, OPT model has a `do_layer_norm_before` field.

Here is the model specific config list:

| Field                                  | Type       | Default Value       |
| :------------------------------------- | :--------- | :------------------ |
| OPT                                    |            |                     |
| do_layer_norm_before                   | bool       | False               |
|                                        |            |                     |
| Falcon                                 |            |                     |
| bias                                   | bool       | True                |
| new_decoder_architecture               | bool       | False               |
| parallel_attention                     | bool       | False               |

### Rank Weights

Like PyTorch, the tensor (weight) name is a string containing hierarchical information,
which is uniquely mapped to a certain parameter of a TensorRT-LLM model.

For example, each transformer layer of the OPT model contains an `Attention` layer, an `MLP` layer. and two `LayerNorm` layers.

#### Attention Weights

The `Attention` layer contains two `Linear` layers, qkv and dense; each `Linear` layer contains one weight and one bias.
There are four tensors (weights) in total, whose names are:

- `transformer.layers.0.attention.qkv.weight`
- `transformer.layers.0.attention.qkv.bias`
- `transformer.layers.0.attention.dense.weight`
- `transformer.layers.0.attention.dense.bias`

where `transformer.layers.0.attention` is the prefix name, indicating that the weights/biases are in the Attention module of the 0-th transformer layer.

#### MLP Weights

The `MLP` layer also contains two `Linear` layers, fc and proj; each `Linear` layer contains one weight and one bias.
There are four tensors (weights) in total, whose names are:

- `transformer.layers.0.mlp.fc.weight`
- `transformer.layers.0.mlp.fc.bias`
- `transformer.layers.0.mlp.proj.weight`
- `transformer.layers.0.mlp.proj.bias`

where `transformer.layers.0.mlp` is the prefix name, indicating that the weights/biases are in the MLP module of the 0-th transformer layer.

#### LayerNorm Weights

Each of the two `LayerNorm` layers, namely `input_layernorm` and `post_layernorm`, contains one weight and one bias.
There are four tensors (weights) in total, whose names are:

- `transformer.layers.0.input_layernorm.weight`
- `transformer.layers.0.input_layernorm.bias`
- `transformer.layers.0.post_layernorm.weight`
- `transformer.layers.0.post_layernorm.bias`

where `transformer.layers.0.input_layernorm` and `transformer.layers.0.post_layernorm` are prefix names for the two `layernorm` modules.

#### KV Cache Quantization Scaling Factors

If we quantize the model, there will be different tensors (depending on the quantization method applied).
For example, if we quantize the KV cache, the `Attention` layer will have this extra scaling factor:

- `transformer.layers.0.attention.kv_cache_scaling_factor`

#### FP8 Quantization Scaling Factors

Here is the FP8 scaling factors of `attention.qkv` linear layer:

- `transformer.layers.0.attention.qkv.activation_scaling_factor`
- `transformer.layers.0.attention.qkv.weights_scaling_factor`

#### AWQ Quantization Scaling Factors

Here is the AWQ scaling factors of `mlp.fc` linear layer:

- `transformer.layers.0.mlp.fc.weights_scaling_factor`
- `transformer.layers.0.mlp.fc.prequant_scaling_factor`

    ```{note}
    The linear weights in TensorRT-LLM checkpoint always follows (`out_feature`, `in_feature`) shape, whereas some quantized linear in TensorRT-LLM implemented by plugin may use (`in_feature`, `out_fature`) shape. The `trtllm-build` command adds a transpose operation to post-process it.

### Example

Let's take OPT as an example and deploy the model with tensor parallelism 2:

```bash
cd examples/opt
python3 convert_checkpoint.py --model_dir ./opt-125m \
                --dtype float16 \
                --tp_size 2 \
                --output_dir ./opt/125M/trt_ckpt/fp16/2-gpu/
```

Here is the checkpoint directory:

```
./opt/125M/trt_ckpt/fp16/1-gpu/
    config.json
    rank0.safetensors
    rank1.safetensors
```

Here is the `config.json`:

```json
{
    "architecture": "OPTForCausalLM",
    "dtype": "float16",
    "logits_dtype": "float32",
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": 768,
    "vocab_size": 50272,
    "position_embedding_type": "learned_absolute",
    "max_position_embeddings": 2048,
    "hidden_act": "relu",
    "mapping": {
        "world_size": 2,
        "tp_size": 2
    },
    "use_parallel_embedding": false,
    "embedding_sharding_dim": 0,
    "do_layer_norm_before": true,
}
```

## Build Checkpoint into TensorRT Engine

TensorRT-LLM provides a unified build command: `trtllm-build`. Before using it,
you may need to add it to the `PATH`.

```bash
export PATH=/usr/local/bin:$PATH

trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/125M/trt_engines/fp16/2-gpu/
```

## Make Evaluation

```bash
mpirun -n 2 --allow-run-as-root \
    python3 ../summarize.py --engine_dir ./opt/125M/trt_engines/fp16/2-gpu/ \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_dir opt-125m \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=14
```
