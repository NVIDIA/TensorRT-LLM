# Phi-4-MultiModal model
This document explains how to run Phi-4-Multimodal (phi4-mm) using TensorRT LLM and run on a single or multiple GPUs with pytorch backend. The original BF16 phi4-mm HF repository is https://huggingface.co/microsoft/Phi-4-multimodal-instruct, and we used ModelOpt to get the FP8 and NVFP4 versions.

## Overview
* We supported phi4-mm models with BF16, FP8 and NVFP4 formats.
* We supported kv-cache-reuse and chunked-prefill for phi4-mm models.
* We supported LoRA inputs for different modalities.
* We supported to set short/long RoPE, by setting `--max_seq_len`. When `max_seq_len <=4096`, it will be short RoPE. Default is long RoPE.

## Usage
### Offline batch inference with LoRA support
```
python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality image --load_lora --auto_model_name Phi4MMForCausalLM

python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality audio --load_lora --auto_model_name Phi4MMForCausalLM

python examples/llm-api/quickstart_multimodal.py --model_dir <model_folder_path> --modality image_audio --load_lora --auto_model_name Phi4MMForCausalLM
```

### TRTLLM-serve with LoRA support
```
cat > lora-extra-llm-api-config.yml<<EOF
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
--extra_llm_api_options lora-extra-llm-api-config.yml \
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
* About HF model downloading, please use `git clone git@hf.co:microsoft/Phi-4-multimodal-instruct` (snapshot download will raise error when running the model) since we assumed some specific folder architectures for phi4-mm, see tensorrt_llm/_torch/models/modeling_phi4mm.py

* About phi4-mm HF model, it is not compatible with the latest transformers even there are some codes related with phi4-mm. If you want to use the transformers codes, you can refer to https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/70.
