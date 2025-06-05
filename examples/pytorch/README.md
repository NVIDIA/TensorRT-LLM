# TRT-LLM with PyTorch

## Run the quick start script:

```bash
python3 quickstart.py
```

## Run the advanced usage example script:

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

### Supported Models
|             Architecture             | Model                                                        | HuggingFace Example                                          | Modality |
| :----------------------------------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :------: |
|   `BertForSequenceClassification`    | BERT-based                                                   | `textattack/bert-base-uncased-yelp-polarity`                 |    L     |
|       `DeepseekV3ForCausalLM`        | DeepSeek-V3                                                  | `deepseek-ai/DeepSeek-V3 `                                   |    L     |
|         `Gemma3ForCausalLM`          | Gemma3                                                       | `google/gemma-3-1b-it`                                       |    L     |
|`HCXVisionForCausalLM`| HyperCLOVAX-SEED-Vision | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | L + V |
|          `LlavaLlamaModel`           | VILA                                                         | `Efficient-Large-Model/NVILA-8B`                             |  L + V   |
| `LlavaNextForConditionalGeneration`  | LLaVA-NeXT                                                   | `llava-hf/llava-v1.6-mistral-7b-hf`                          |  L + V   |
|          `LlamaForCausalLM`          | Llama 3 <br> Llama 3.1 <br> Llama 2 <br> LLaMA               | `meta-llama/Meta-Llama-3.1-70B`                              |    L     |
|   `Llama4ForConditionalGeneration`   | Llama 4 Scout <br> Llama 4 Maverick                          | `meta-llama/Llama-4-Scout-17B-16E-Instruct` <br> `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |  L + V   |
|         `MistralForCausalLM`         | Mistral                                                      | `mistralai/Mistral-7B-v0.1`                                  |    L     |
|         `MixtralForCausalLM`         | Mixtral                                                      | `mistralai/Mixtral-8x7B-v0.1`                                |    L     |
|   `MllamaForConditionalGeneration`   | Llama 3.2                                                    | `meta-llama/Llama-3.2-11B-Vision`                            |    L     |
|        `NemotronForCausalLM`         | Nemotron-3 <br> Nemotron-4 <br> Minitron                     | `nvidia/Minitron-8B-Base`                                    |    L     |
|        `NemotronHForCausalLM`        | Nemotron-H                                                   | `nvidia/Nemotron-H-8B-Base-8K` <br> `nvidia/Nemotron-H-47B-Base-8K` <br> `nvidia/Nemotron-H-56B-Base-8K` |    L     |
|       `NemotronNASForCausalLM`       | LLamaNemotron <br> LlamaNemotron Super <br> LlamaNemotron Ultra | `nvidia/Llama-3_1-Nemotron-51B-Instruct` <br> `nvidia/Llama-3_3-Nemotron-Super-49B-v1` <br> `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` |    L     |
|          `Qwen2ForCausalLM`          | QwQ, Qwen2                                                   | `Qwen/Qwen2-7B-Instruct`                                     |    L     |
|     `Qwen2ForProcessRewardModel`     | Qwen2-based                                                  | `Qwen/Qwen2.5-Math-PRM-7B`                                   |    L     |
|        `Qwen2ForRewardModel`         | Qwen2-based                                                  | `Qwen/Qwen2.5-Math-RM-72B`                                   |    L     |
|  `Qwen2VLForConditionalGeneration`   | Qwen2-VL                                                     | `Qwen/Qwen2-VL-7B-Instruct`                                  |  L + V   |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL                                                   | `Qwen/Qwen2.5-VL-7B-Instruct`                                |  L + V   |

Note:
- L: Language only
- L + V: Language and Vision multimodal support
- Llama 3.2 accepts vision input, but our support currently limited to text only.

## Run the speculative decoding script:

```bash
# NGram drafter
python3 examples/pytorch/quickstart_advanced.py \
    --max_matching_ngram_size=2 \
    --spec_decode_nextn=4
```
