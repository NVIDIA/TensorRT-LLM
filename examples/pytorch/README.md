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
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B

# user inputs
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --prompt "Describe the scene" "What do you see in the image?" --data "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png" "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg" --max_tokens 64
```
