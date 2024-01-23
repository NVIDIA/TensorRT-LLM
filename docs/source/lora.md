## Run gpt-2b + LoRA using GptManager / cpp runtime

First build a model with LoRA and inflight-batching enabled.

```
BASE_MODEL=llama-7b-hf

python3 tensorrt_llm/examples/llama/build.py --model_dir ${BASE_MODEL} \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir "/tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/" \
                --max_batch_size 128 \
                --max_input_len 512 \
                --max_output_len 50 \
                --use_lora_plugin float16 \
                --lora_target_modules "attn_q" "attn_k" "attn_v" \
                --use_inflight_batching \
                --paged_kv_cache \
                --max_lora_rank 8 \
                --world_size 1 --tp_size 1
```

To pass LoRAs into the cpp runtime they must be converted to the format below.
The script below will convert a huggingface LoRA model to the correct numpy tensors.

```
git-lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
git-lfs clone https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0

python3 tensorrt_llm/examples/hf_lora_convert.py -i Japanese-Alpaca-LoRA-7b-v0 -o Japanese-Alpaca-LoRA-7b-v0-weights --storage-type float16
python3 tensorrt_llm/examples/hf_lora_convert.py -i luotuo-lora-7b-0.1 -o luotuo-lora-7b-0.1-weights --storage-type float16
```

See tensorrtllm_backend [docs](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/README.md) for a Multi-LoRA example using Triton.

### LoRA tensor format details

To run inference with LoRA weights using GptManager, InferenceRequests must have LoraWeights (lora_weights) and LoraConfig (lora_config) parameters.

`LoraWeights` contains the weights for all the LoRAs. Currently this should include weight for all tp and pp ranks.
The weights tensor has the shape `[ num_lora_modules_layers, D x Hi + Ho x D ]`. the last dimension holds the in / out adapter weights for the associated module (e.g. attn_qkv) and model layer.
Each of the in / out tensors are first flattened and then concatenated together in the format above.
The first dimension (of size `num_lora_module_layers`) has an entry for each module-layer (ie there is an entry for attn_q layer1 and another for attn_k layer1).

`D=adapter_size (i.e. R value), Hi=hidden_size_in, Ho=hidden_size_out.`

`LoraConfig` is a configuration tensor which identifies the moduleId, layerId, and adapter size of each element of `LoraWeights`.
It has the shape `[num_lora_modules_layers, 3]`.
The last dimension holds `[ module_id, layer_idx, adapter_size D (i.e. R value) ]`

Reference: This feature supports LoRAs as described in https://arxiv.org/pdf/2106.09685.pdf

#### Example LoRA tensors
Here is an example of loraWeights and loraConfig tensors for a model with tp=1, pp=1, 4 layers, and a hidden size of 4.
The tensors below are for a LoRA which has a `q` and `k` adapter.

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

See LoraModule::ModuleType for model id mapping

| module name (as specified in build.py scripts | module id | description |
| --------------------------------------------- | --------- | ----------- |
| attn_qkv | 0 | compbined qkv adapter |
| attn_q | 1 | q adapter |
| attn_k | 2 | k adapter |
| attn_v | 3 | v adapter |
| attn_dense | 4 | adapter for the dense layer in attention |
| mlp_h_to_4h | 5 | for llama2 adapter for gated mlp layer after attention / RMSNorm: up projection |
| mlp_4h_to_h | 6 | for llama2 adapter for gated mlp layer after attention / RMSNorm: down projection |
| mlp_gate | 7 | for llama2 adapter for gated mlp later after attention / RMSNorm: gate |
