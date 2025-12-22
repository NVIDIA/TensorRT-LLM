# Phi-4-MultiModal model

This document outlines the procedures for executing Phi-4-Multimodal (phi4-mm) using TensorRT LLM. The implementation supports both single and multi-GPU configurations via the PyTorch backend. Additionally, ModelOpt was employed to derive FP8 and NVFP4 checkpoints from the source [BF16 repository](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).


## Overview
* Supported BF16, FP8, NVFP4 model formats.
* Supported single and multi-GPUs inference.
* Added support for KV cache reuse and chunked prefill for phi4-mm models
* Enabled LoRA support for multi-modal inputs.
* Configurable RoPE scaling: The model defaults to Long RoPE but automatically switches to Short RoPE when `--max_seq_len` is set to 4096 or lower.

## Usage
### Offline batch inference
```
python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality image --load_lora --auto_model_name Phi4MMForCausalLM

python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality audio --load_lora --auto_model_name Phi4MMForCausalLM

python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality image_audio --load_lora --auto_model_name Phi4MMForCausalLM
```

### TRTLLM-serve
```
cat > lora_llmapi_config.yml<<EOF
kv_cache_config:
    free_gpu_memory_fraction: 0.6
lora_config:
  swap_gate_up_proj_lora_b_weight: false
  max_loras: 2
  max_cpu_loras: 2
  max_lora_rank: 320
  lora_target_modules:
    - attn_qkv
    - attn_dense
    - mlp_gate_up
    - mlp_4h_to_h
  trtllm_modules_to_hf_modules:
    attn_qkv: qkv_proj
    attn_dense: o_proj
    mlp_gate_up: gate_up_proj
    mlp_4h_to_h: down_proj
EOF

trtllm-serve  \
<model_folder_path> \
--backend pytorch \
--trust_remote_code \
--config lora_llmapi_config.yml
```

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-4-multimodal-instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the natural environment in the image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
            }
          }
        ]
      }
    ],
    "lora_request": {
      "lora_name": "lora1",
      "lora_int_id": 0,
      "lora_path": "<path_to_vision_lora_folder>"
    },
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

## Notes

* Model Download: Please use `git clone git@hf.co:microsoft/Phi-4-multimodal-instruct` to download the model. Do not use snapshot downloads, as they cause runtime errors due to specific directory structure requirements (see `tensorrt_llm/_torch/models/modeling_phi4mm.py`).

* Transformers Compatibility: The Phi-4-MM model is currently incompatible with the latest transformers library, despite the presence of related code. If you need to use the transformers implementation, please refer to [this discussion](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/70).
