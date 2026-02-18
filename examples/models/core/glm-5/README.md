# GLM-5

This guide walks you through running the GLM-5 model using TensorRT-LLM with the PyTorch backend.

GLM-5 uses Multi-Latent Attention (MLA) with DeepSeek Sparse Attention (DSA). It shares the same architecture as DeepSeek V3.2 and reuses the `DeepseekV32ForCausalLM` code path in TensorRT-LLM.

> [!NOTE]
> This guide assumes that you replace placeholder values (e.g., `<YOUR_MODEL_DIR>`) with the appropriate paths.

## Table of Contents

- [GLM-5](#glm-5)
  - [Hardware Requirements](#hardware-requirements)
  - [Downloading the Model Weights](#downloading-the-model-weights)
  - [Prerequisites](#prerequisites)
  - [Serving](#serving)
  - [Notes and Troubleshooting](#notes-and-troubleshooting)

## Hardware Requirements

| GPU | GLM-5 FP4 |
|-----|-----------|
| B200/GB200 | 8 GPUs (SM100) |

## Downloading the Model Weights

The following checkpoints are available:

1. [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) — Official BF16 checkpoint.
2. [warnold-nv/GLM-5-nvfp4-v1](https://huggingface.co/warnold-nv/GLM-5-nvfp4-v1) — Unofficial NVFP4 checkpoint for experimentation only. *Quantized with ModelOpt by Will Arnold.*

```bash
git lfs install
git clone https://huggingface.co/warnold-nv/GLM-5-nvfp4-v1 <YOUR_MODEL_DIR>
```

## Prerequisites

| Item | Details |
|------|---------|
| Hardware | 8× B200 GPUs (SM100) |
| Container | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3` with this PR installed |

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html) for how to build TensorRT-LLM from source and start a TRT-LLM Docker container.

```bash
docker run --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/your/models:/workspace/models \
    -it nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 bash
```

> **Everything below runs inside the Docker container.**

### Config modifications

GLM-5 uses `model_type: "glm_moe_dsa"`, which requires `transformers>=5.0.2`. TensorRT-LLM currently ships with `transformers==4.57.1`, so the model type and tokenizer class must be updated for the config to load correctly.

Edit `<YOUR_MODEL_DIR>/config.json` — change `model_type`:

```json
{
    "model_type": "deepseek_v32"
}
```

Edit `<YOUR_MODEL_DIR>/tokenizer_config.json` — apply the following two changes:

1. Change `"tokenizer_class"` from `"TokenizersBackend"` to `"PreTrainedTokenizerFast"`.
2. Rename the key `"extra_special_tokens"` to `"additional_special_tokens"`.

---

## Serving

### trtllm-serve

Below is an example B200 serving configuration for min-latency in FP4. **Treat this as a starting point — tune for your model and workload to achieve the best performance.**

#### B200 FP4 min-latency config

```bash
cat >./config.yml <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 1024
enable_attention_dp: false
enable_chunked_prefill: true
kv_cache_config:
    enable_block_reuse: false
    dtype: fp8
# speculative_config:
#   decoding_type: MTP
#   num_nextn_predict_layers: 1
stream_interval: 10
EOF
```

#### Launch trtllm-serve OpenAI-compatible API server

```bash
trtllm-serve \
  <YOUR_MODEL_DIR> \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 8192 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --config ./config.yml
```

> [!WARNING]
> You may encounter OOM issues with some configurations. Try reducing `kv_cache_free_gpu_mem_fraction` to a smaller value as a workaround. If using a max-throughput config, reduce `max_num_tokens` to `3072` to avoid OOM.

To query the server:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "<YOUR_MODEL_DIR>",
      "prompt": "Write a short poem about a cat and a dog",
      "max_tokens": 256,
      "temperature": 0.6
  }'
```

## Notes and Troubleshooting

- **Model Directory:** Update `<YOUR_MODEL_DIR>` with the actual path where the model weights reside.
- **GPU Memory:** Adjust `--max_batch_size` and `--max_num_tokens` if you encounter out-of-memory errors.
- **Configuration Files:** Verify that the configuration files are correctly formatted to avoid runtime issues.
- **Architecture:** GLM-5 reuses the DeepSeek V3.2 attention implementation (`DeepseekV32Attention`), which includes a built-in DSA indexer that routes context attention through absorption mode. DSA parameters (`index_n_heads`, `index_head_dim`, `index_topk`) are read automatically from the model config.
