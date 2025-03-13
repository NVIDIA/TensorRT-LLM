(lora)=

## Run gpt-2b + LoRA using Executor / cpp runtime

First build a model with LoRA and inflight-batching enabled.

```bash
git-lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
git-lfs clone https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0
BASE_MODEL=llama-7b-hf

python examples/llama/convert_checkpoint.py --model_dir ${BASE_MODEL} \
    --output_dir /tmp/llama_7b/trt_ckpt/fp16/1-gpu/ \
    --dtype float16

trtllm-build --checkpoint_dir /tmp/llama_7b/trt_ckpt/fp16/1-gpu/ \
    --output_dir /tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/ \
    --remove_input_padding enable \
    --gpt_attention_plugin float16 \
    --context_fmha enable \
    --paged_kv_cache enable \
    --gemm_plugin float16 \
    --lora_plugin float16 \
    --max_batch_size 128 \
    --max_input_len 512 \
    --max_seq_len 562 \
    --lora_dir Japanese-Alpaca-LoRA-7b-v0 \
    --max_lora_rank 8 \
    --lora_target_modules "attn_q" "attn_k" "attn_v"
```

To pass LoRAs into the cpp runtime they must be converted to the format below.
The script below will convert a Hugging Face LoRA model to the correct NumPy tensor.

```bash
python3 tensorrt_llm/examples/hf_lora_convert.py -i Japanese-Alpaca-LoRA-7b-v0 -o Japanese-Alpaca-LoRA-7b-v0-weights --storage-type float16
python3 tensorrt_llm/examples/hf_lora_convert.py -i luotuo-lora-7b-0.1 -o luotuo-lora-7b-0.1-weights --storage-type float16
```

Refer to the [tensorrtllm_backend documentation](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/lora.md) for a Multi-LoRA example using Triton.

### LoRA tensor format details

To run inference using `Executor`, a `Request` must have a `LoraConfig` that contains a `task_id`, `weights` and `config` parameters.

`task_id` the unique task ID for the given LoRA.

To perform inference with a specific LoRA for the first time, `task_id`, `weights`, and `config` must all be given. The LoRA will be cached, so that subsequent requests for the same task only require `task_id`.
If the cache is full, the oldest LoRA will be evicted to make space for new ones. An error is returned if `task_id` is not cached.

`weights` contains the weights for all the LoRAs. Currently, this should include weights for all TP and PP ranks.
The weights tensor has the shape `[num_lora_modules_layers, D x Hi + Ho x D ]`. The last dimension holds the in / out adapter weights for the associated module (for example, `attn_qkv`) and model layer.

Each of the in / out tensors are first flattened and then concatenated together in the format above.
The first dimension (of size `num_lora_module_layers`) has an entry for each module-layer (that is, there is an entry for `attn_q layer1` and another for `attn_k layer1`).

`D=adapter_size (i.e. R value), Hi=hidden_size_in, Ho=hidden_size_out.`

`config` is a configuration tensor which identifies the moduleId, layerId, and adapter size of each element of `LoraWeights`. It has the shape `[num_lora_modules_layers, 3]`. The last dimension holds `[module_id, layer_idx, adapter_size D (i.e. R value)]`.

This feature supports LoRAs as described in https://arxiv.org/pdf/2106.09685.pdf

#### Example LoRA tensors

Here is an example of `LoraWeights` and `LoraConfig` tensors for a model with tp=1, pp=1, 4 layers, and a hidden size of 4.
The following tensors are for a LoRA which has a `q` and `k` adapter.

```
# loraConfig
[
  [1, 0, 2]
  [2, 0, 4]
  [1, 1, 2]
  [2, 1, 4]
  [1, 2, 2]  # Note that the final 2 layers only adapt `q`
  [1, 3, 8]
]
# Note: The loraConfig tensor configures the loraWeights tensor.
#       The contents of each row of loraWeights is specified be the corresponding row in loraConfig

# loraWeights
# Note: that 'in weights' and 'out weights' are 'A' and 'B' in the LoRA paper.
[
  [ <2 x 4 in weights>, <4 x 2 out weights> <padding> ]  # `q` adapter for layer 0
  [ <4 x 4 in weights>, <4 x 4 out weights> <padding> ]  # `k` adapter for layer 0
  [ <2 x 4 in weights>, <4 x 2 out weights> <padding> ]  # `q` adapter for layer 1
  [ <4 x 4 in weights>, <4 x 4 out weights> <padding> ]  # `k` adapter for layer 1
  [ <2 x 4 in weights>, <4 x 2 out weights> <padding> ]  # `q` adapter for layer 2
  [ <8 x 4 in weights>, <4 x 8 out weights>           ]  # `q` adapter for layer 3. Note the final layer has a adapter size of 8
]

```

#### LoRA Module id mapping

| module name (as specified in `convert_checkpoint.py` scripts) | module id | description |
| --------------------------------------------- | --------- | ----------- |
| attn_qkv | 0 | compbined qkv adapter |
| attn_q | 1 | q adapter |
| attn_k | 2 | k adapter |
| attn_v | 3 | v adapter |
| attn_dense | 4 | adapter for the dense layer in attention |
| mlp_h_to_4h | 5 | for llama2 adapter for gated mlp layer after attention / RMSNorm: up projection |
| mlp_4h_to_h | 6 | for llama2 adapter for gated mlp layer after attention / RMSNorm: down projection |
| mlp_gate | 7 | for llama2 adapter for gated mlp later after attention / RMSNorm: gate |
| cross_attn_qkv | 8 | compbined qkv adapter for cross attention |
| cross_attn_q | 9 | q adapter for cross attention |
| cross_attn_k | 10 | k adapter for cross attention |
| cross_attn_v | 11 | v adapter for cross attention |
| cross_attn_dense | 12 | adapter for the dense layer in cross attention |
| moe_h_to_4h | 13 | for mixtral adapter for expert mlp layer: up projection |
| moe_4h_to_h | 14 | for mixtral adapter for expert mlp layer: down projection |
| moe_gate | 15 | for mixtral adapter for expert mlp layer: gate |
| moe_router | 16 | for mixtral adapter for expert router layer |
| mlp_router | 17 | for qwen2-moe adapter for shared expert gate layer |
| mlp_gate_up | 18 | adapter for gated mlp layer after attention / RMSNorm: gate + up projection |

#### LoraCache configuration

The core idea is that we will have a fixed size, 2-level LoRA cache in TRT-LLM. The higher level cache resides on the host and the lower level is on GPU (distinct from the existing KV cache). Sizes of both are user configurable.

The CPU cache is configured to be a max size.  The GPU cache is configured to a percentage of free GPU memory after engine load. As requests come in LoRAs are stored in the host cache.

As requests are scheduled for execution LoRAs are loaded into the GPU cache.

#### LoRA with tensor parallel

The partition of tensor parallel for LoRA is special. There are two cases: `RowLinear` and `ColumnLinear`. Assume we have a linear layer and the input feature size is `K` and the output feature size is `N`. Then, the shape of the weight is `[K, N]`.

First, consider this linear layer is a `ColumnLinear` layer. When we partition the weight, we split the weight by column with `tp_size`. Then, there are `tp_size` split weights and the shapes of these weights are `[K, N // tp_size]`. When we apply LoRA adapter on such `ColumnLinear` layer, the shapes of original two weights are `[K, lora_rank]` and `[lora_rank, N]`. So, we only partition the second weight and get `tp_size` split weights with shapes `[lora_rank, N // tp_size]`. For the first weight, each GPU maintains the same entire weight (with shape `[K, lora_rank]`).

Next, consider this linear layer is a `RowLinear` layer. When we partition the weight, we split the weight by row with `tp_size`. Then, there are `tp_size` split weights and the shapes of these weights are `[K // tp_size, N]`. When we apply LoRA adapter on such `RowLinear` layer, the shapes of original two weights are `[K, lora_rank]` and `[lora_rank, N]`. So, we only partition the first weight and get `tp_size` split weights with shapes `[K // tp_size, lora_rank]`. For the second weight, each GPU maintains the same entire weight (with shape `[lora_rank, N]`).

#### DoRA

TRTLLM supports DoRA as described in https://arxiv.org/abs/2402.09353 . To enable DoRA, you must add the additional `--dora_plugin enable` flag to the `trtllm-build` command.

The DoRA scales must be normalized before they are submitted to TRTLLM in an inference request. The normalization requires the base model weights. To normalize your adapter you may use the script provided in `tensorrt_llm/examples/dora/normalize_weights.py`.

When using DoRA, the format of `LoraWeights` and `LoraConfig` changes slightly.
The shape of `LoraConfig` becomes `[module_id, layer_idx, adapter_size D (i.e. R value), is_dora]`, with `is_dora` a boolean flag that determines whether the supplied adapter contains DoRA scales or not. If the old config shape is used, it is assumed the adapter does not have DoRA scales.
The shape of `LoraWeights` becomes `[num_lora_modules_layers, D x Hi + Ho x D + Ho]`, and the last `Ho` values are the DoRA scale vector.
