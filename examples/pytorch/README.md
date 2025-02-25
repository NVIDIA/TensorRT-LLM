# TRT-LLM with PyTorch

Run the quick start script:

```bash
python3 quickstart.py
```

Run the advanced usage example script:

```bash
# BF16
python3 quickstart_advanced.py --model_dir meta-llama/Llama-3.1-8B-Instruct

# FP8
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8

# BF16 + TP=2
python3 quickstart_advanced.py --model_dir meta-llama/Llama-3.1-8B-Instruct --tp_size 2

# FP8 + TP=2
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --tp_size 2

# FP8(e4m3) kvcache
python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --kv_cache_dtype fp8
```

Run the multimodal example script:

```bash
# default inputs
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --model_type vila --modality image

# user inputs
# supported modes:
# (1) N prompt, N media (N requests are in-flight batched)
# (2) 1 prompt, N media
# Note: media should be either image or video. Mixing image and video is not supported.
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --model_type vila --modality video --prompt "Tell me what you see in the video briefly." "Describe the scene in the video briefly." --media "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4" "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4" --max_tokens 128
```
