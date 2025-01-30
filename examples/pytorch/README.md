# TRT-LLM with PyTorch

Run example:

```bash
# BF16
python3 quickstart.py --model_dir meta-llama/Llama-3.1-8B-Instruct

# FP8
python3 quickstart.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8

# BF16 + TP=2
python3 quickstart.py --model_dir meta-llama/Llama-3.1-8B-Instruct --tp_size 2

# FP8 + TP=2
python3 quickstart.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --tp_size 2

# FP8(e4m3) kvcache
python3 quickstart.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --kv_cache_dtype fp8
```
