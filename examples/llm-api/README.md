# LLM API Examples

Please refer to the [official documentation](https://nvidia.github.io/TensorRT-LLM/llm-api/) including [customization](https://nvidia.github.io/TensorRT-LLM/examples/customization.html) for detailed information and usage guidelines regarding the LLM API.


## Run the advanced usage example script:

```bash
# FP8 + TP=2
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --tp_size 2

# FP8 (e4m3) kvcache
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --kv_cache_dtype fp8

# BF16 + TP=8
python3 quickstart_advanced.py --model_dir nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 --tp_size 8

# Nemotron-H requires disabling cache reuse in kv cache
python3 quickstart_advanced.py --model_dir nvidia/Nemotron-H-8B-Base-8K --disable_kv_cache_reuse --max_batch_size 8
```

## Run the multimodal example script:

```bash
# default inputs
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality image [--use_cuda_graph]

# user inputs
# supported modes:
# (1) N prompt, N media (N requests are in-flight batched)
# (2) 1 prompt, N media
# Note: media should be either image or video. Mixing image and video is not supported.
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality video --prompt "Tell me what you see in the video briefly." "Describe the scene in the video briefly." --media "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4" "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4" --max_tokens 128 [--use_cuda_graph]
```

## Run the speculative decoding script:

```bash
# NGram drafter
python3 quickstart_advanced.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --spec_decode_algo NGRAM \
    --spec_decode_max_draft_len 4 \
    --max_matching_ngram_size 2 \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```

```bash
# Draft Target
python3 quickstart_advanced.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --spec_decode_algo draft_target \
    --spec_decode_max_draft_len 5 \
    --draft_model_dir meta-llama/Llama-3.2-1B-Instruct \
    --disable_overlap_scheduler \
    --disable_kv_cache_reuse
```
