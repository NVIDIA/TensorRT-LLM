# New Workflow

## Overview

There are 3 steps in the new workflow:

1. convert weights from different source frameworks into TensorRT-LLM checkpoint
2. build the TensorRT-LLM checkpoint into TensorRT engine(s) with a unified build command
3. load the engine(s) to TensorRT-LLM model runner and make evaluation with different evaluation tasks

```txt
NeMo -------------
                  |
HuggingFace ------
                  |   convert                       build                load
AMMO -------------  ----------> TRT-LLM Checkpoint --------> TRT Engine ------> TRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
```

## Prepare TensorRT-LLM Checkpoint

There are different kinds of sources we want to support:

1. trained models from NeMo/DeepSpeed/JAX
2. quantized models from AMMO
3. popular models from HuggingFace

TensorRT-LLM defines its own checkpoint format. A checkpoint directory include:

1. One config json file, which contains several model hyper-parameters
2. One or several rank weights files, each rank file contains a dictionary of tensors(weights)

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
| use_prompt_tuning                      | bool       | false               |
| mapping.world_size                     | int        | 1                   |
| mapping.tp_size                        | int        | 1                   |
| mapping.pp_size                        | int        | 1                   |
| quantization.use_smooth_quant          | bool       | false               |
| quantization.per_channel               | bool       | false               |
| quantization.per_token                 | bool       | false               |
| quantization.per_group                 | bool       | false               |
| quantization.group_size                | int        | 64                  |
| quantization.int8_kv_cache             | bool       | false               |
| quantization.enable_fp8                | bool       | false               |
| quantization.fp8_kv_cache              | bool       | false               |
| quantization.use_weight_only           | bool       | false               |
| quantization.weight_only_precision     | string     | 'int8'              |

The config field is extensible, a model could add its own specific config fields.
For example, OPT model has a `do_layer_norm_before` field.

### Rank Weights

Like PyTorch, the tensor(weight) name is a string containing hierarchical information,
which is uniquely mapped to a certain parameter of a TensorRT-LLM model.

For example, the `Attention` layer contains 2 `Linear` layer, qkv and dense.
Each linear layer contains one weight and one bias.
So, there are 4 tensors(weights) in total, whose names are:

- "xxx.qkv.weight"
- "xxx.qkv.bias"
- "xxx.dense.weight"
- "xxx.dense.bias"

`xxx` is the prefix name. If we quantize the KV cache, we will have extra 2 scaling factors:

- "xxx.kv_orig_quant_scale"
- "xxx.kv_quant_orig_scale"

If we do FP8 quantize, we will have extra 4 scaling factors:

- "xxx.qkv.activation_scaling_factor"
- "xxx.qkv.weights_scaling_factor"
- "xxx.dense.activation_scaling_factor"
- "xxx.dense.weights_scaling_factor"

### Example

Let's take OPT as an example, say we want to deploy the model with tensor parallelism 2:

```bash
cd examples/opt
python3 convert_checkpoint.py --model_dir ./opt-125m \
                --dtype float16 \
                --world_size 2 \
                --output_dir ./opt/125M/trt_ckpt/fp16/2-gpu/
```

Here is the checkpoint directory:

```txt
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
    "quantization": {
        "use_weight_only": false,
        "weight_only_precision": "int8"
    },
    "mapping": {
        "world_size": 2,
        "tp_size": 2
    },
    "use_parallel_embedding": false,
    "embedding_sharding_dim": 0,
    "share_embedding_table": false,
    "do_layer_norm_before": true,
    "use_prompt_tuning": false
}
```

## Build Checkpoint into TensorRT Engine

TensorRT-LLM provides a unified build command: `trtllm-llm`. Before using it,
you may need to add it to the `PATH`

```bash
export PATH=/usr/local/bin:$PATH

trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_output_len 100 \
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
