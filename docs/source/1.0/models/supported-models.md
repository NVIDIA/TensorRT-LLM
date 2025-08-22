# Supported Models

The following is a table of supported models for the PyTorch backend:

| Architecture                         | Model                              | HuggingFace Example                          | Modality |
| ------------------------------------ | ---------------------------------- | -------------------------------------------- | -------- |
| `BertForSequenceClassification`      | BERT-based                         | `textattack/bert-base-uncased-yelp-polarity` | L        |
| `DeciLMForCausalLM`                  | Nemotron                           | `nvidia/Llama-3_1-Nemotron-51B-Instruct`     | L        |
| `DeepseekV3ForCausalLM`              | DeepSeek-V3                        | `deepseek-ai/DeepSeek-V3`                    | L        |
| `Gemma3ForCausalLM`                  | Gemma 3                            | `google/gemma-3-1b-it`                       | L        |
| `LlavaLlamaModel`                    | VILA                               | `Efficient-Large-Model/NVILA-8B`             | L + V    |
| `LlavaNextForConditionalGeneration`  | LLaVA-NeXT                         | `llava-hf/llava-v1.6-mistral-7b-hf`          | L + V    |
| `LlamaForCausalLM`                   | Llama 3.1, Llama 3, Llama 2, LLaMA | `meta-llama/Meta-Llama-3.1-70B`              | L        |
| `Llama4ForConditionalGeneration`     | Llama 4                            | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  | L        |
| `MistralForCausalLM`                 | Mistral                            | `mistralai/Mistral-7B-v0.1`                  | L        |
| `MixtralForCausalLM`                 | Mixtral                            | `mistralai/Mixtral-8x7B-v0.1`                | L        |
| `MllamaForConditionalGeneration`     | Llama 3.2                          | `meta-llama/Llama-3.2-11B-Vision`            | L        |
| `NemotronForCausalLM`                | Nemotron-3, Nemotron-4, Minitron   | `nvidia/Minitron-8B-Base`                    | L        |
| `NemotronNASForCausalLM`             | NemotronNAS                        | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`     | L        |
| `Qwen2ForCausalLM`                   | QwQ, Qwen2                         | `Qwen/Qwen2-7B-Instruct`                     | L        |
| `Qwen2ForProcessRewardModel`         | Qwen2-based                        | `Qwen/Qwen2.5-Math-PRM-7B`                   | L        |
| `Qwen2ForRewardModel`                | Qwen2-based                        | `Qwen/Qwen2.5-Math-RM-72B`                   | L        |
| `Qwen2VLForConditionalGeneration`    | Qwen2-VL                           | `Qwen/Qwen2-VL-7B-Instruct`                  | L + V    |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL                         | `Qwen/Qwen2.5-VL-7B-Instruct`                | L + V    |
| `Qwen3ForCausalLM`                   | Qwen3                              | `Qwen/Qwen3-8B`                              | L        |
| `Qwen3MoeForCausalLM`                | Qwen3MoE                           | `Qwen/Qwen3-30B-A3B`                         | L        |

Note:
- L: Language only
- L + V: Language and Vision multimodal support
- Llama 3.2 accepts Vision input, but our support is currently limited to Language only.

## Model-Feature Support Matrix(Key Models)

Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.

| Model Architecture/Feature     | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | MTP | EAGLE-3(One Model Engine) | EAGLE-3(Two Model Engine) | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Sliding Window Attention | Logits Post Processor | Guided Decoding |
| ------------------------------ | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | --- | ------------------------- | ------------------------- | ------------- | ---------------- | -------------- | ------------------------ | --------------------- | --------------- |
| DeepseekV3ForCausalLM          | Yes               | Yes        | Yes                        | Yes                   | Yes [^1]        | Yes | No                        | No                        | Yes           | Yes              | Yes [^2]       | N/A                      | Yes                   | Yes             |
| Qwen3MoeForCausalLM            | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                       | Yes                       | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| Llama4ForConditionalGeneration | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                       | Yes                       | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |


[^1]: Chunked Prefill for MLA can only be enabled on SM100.
[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100 and in BF16/FP8 KV cache dtype.
