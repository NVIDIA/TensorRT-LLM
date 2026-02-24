# Nemotron-nano-v2-VL

## Model series
 * https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
 * https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8
 * https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD

## Support Matrix
  * BF16 / FP8 / FP4
  * Tensor Parallel / Pipeline Parallel
  * Inflight Batching
  * PAGED_KV_CACHE
  * checkpoint type: Huggingface (HF)
  * Image / video multimodal input


## Offline batch inference example CMDs
 * Taking BF16 model as an example below, you can change to FP8 / FP4 ckpt.

 * Image modality input:

```bash
python3 examples/llm-api/quickstart_multimodal.py --model_dir nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 --disable_kv_cache_reuse --max_batch_size 128 --trust_remote_code
```

 * Image modality input with chunked_prefill:

```bash
python3 examples/llm-api/quickstart_multimodal.py --model_dir nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 --disable_kv_cache_reuse --max_batch_size 128 --trust_remote_code --enable_chunked_prefill --max_num_tokens=256
```

 * Video modality input:

```bash
python3 examples/llm-api/quickstart_multimodal.py --model_dir nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 --disable_kv_cache_reuse --max_batch_size 128 --trust_remote_code --modality video --max_num_tokens 131072
```

 * Video modality input with Efficient video sampling (EVS):

```bash
TLLM_VIDEO_PRUNING_RATIO=0.9 python3 examples/llm-api/quickstart_multimodal.py --model_dir nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 --disable_kv_cache_reuse --max_batch_size 128 --trust_remote_code --modality video --max_num_tokens 131072
```

## Online serving example CMDs

 * Taking BF16 model as an example below, you can change to FP8 / FP4 ckpt.

```bash
# Create extra config file.
cat > ./config.yml << EOF
kv_cache_config:
  enable_block_reuse: false
  mamba_ssm_cache_dtype: float32
EOF

# CMD to launch serve without EVS.
trtllm-serve  \
nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16\
--host 0.0.0.0 \
--port 8000 \
--backend pytorch \
--max_batch_size 16 \
--max_num_tokens 131072 \
--trust_remote_code \
--media_io_kwargs "{\"video\": {\"fps\": 2, \"num_frames\": 128} }" \
--config config.yml

# CMD to launch serve with EVS (video_pruning_ratio=0.9).
TLLM_VIDEO_PRUNING_RATIO=0.9 trtllm-serve  \
nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16\
--host 0.0.0.0 \
--port 8000 \
--backend pytorch \
--max_batch_size 16 \
--max_num_tokens 131072 \
--trust_remote_code \
--media_io_kwargs "{\"video\": {\"fps\": 2, \"num_frames\": 128} }" \
--config config.yml
```

# Known issue:
 * Don't set too large batch size, otherwise the Mamba cache might raise OOM error.
 * Video modality cannot support chunked prefill yet.
 * Prefix-caching is not supported for Nemotron-nano-v2-VL yet .
