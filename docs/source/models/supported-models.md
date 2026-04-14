(support-matrix)=
# Supported Models

The following is a table of supported models for the PyTorch backend:

| Architecture                         | Model                              | HuggingFace Example                          |
| ------------------------------------ | ---------------------------------- | -------------------------------------------- |
| `BertForSequenceClassification`      | BERT-based                         | `textattack/bert-base-uncased-yelp-polarity` |
| `Cohere2ForCausalLM`                 | Command A                          | `CohereLabs/c4ai-command-a-03-2025`          |
| `DeciLMForCausalLM`                  | Nemotron                           | `nvidia/Llama-3_1-Nemotron-51B-Instruct`     |
| `DeepseekV3ForCausalLM`              | DeepSeek-V3, Kimi-K2               | `deepseek-ai/DeepSeek-V3`                    |
| `DeepseekV32ForCausalLM`             | DeepSeek-V3.2                      | `deepseek-ai/DeepSeek-V3.2`                  |
| `Exaone4ForCausalLM`                 | EXAONE 4.0                         | `LGAI-EXAONE/EXAONE-4.0-32B`                 |
| `ExaoneMoEForCausalLM`               | K-EXAONE                           | `LGAI-EXAONE/K-EXAONE-236B-A23B`             |
| `Gemma3ForCausalLM`                  | Gemma 3                            | `google/gemma-3-1b-it`                       |
| `Gemma3nForConditionalGeneration` [^8]| Gemma 3n                           | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it` |
| `Gemma4ForConditionalGeneration` [^7]| Gemma 4                            | `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` |
| `Glm4MoeForCausalLM`                 | GLM-4.5, GLM-4.6, GLM-4.7          | `THUDM/GLM-4-100B-A10B`                      |
| `Glm4MoeLiteForCausalLM` [^6]        | GLM-4.7-Flash                      | `zai-org/GLM-4.7-Flash`                      |
| `GlmMoeDsaForCausalLM`               | GLM-5                              | `zai-org/GLM-5`                              |
| `GptOssForCausalLM`                  | GPT-OSS                            | `openai/gpt-oss-120b`                        |
| `KimiK25ForConditionalGeneration`    | Kimi-K2.5                          | `moonshotai/Kimi-K2.5`                       |
| `LlamaForCausalLM`                   | Llama 3.1, Llama 3, Llama 2, LLaMA | `meta-llama/Meta-Llama-3.1-70B`              |
| `Llama4ForConditionalGeneration`     | Llama 4                            | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  |
| `MiniMaxM2ForCausalLM`               | MiniMax M2/M2.1                    | `MiniMaxAI/MiniMax-M2`                       |
| `MistralForCausalLM`                 | Mistral                            | `mistralai/Mistral-7B-v0.1`                  |
| `MixtralForCausalLM`                 | Mixtral                            | `mistralai/Mixtral-8x7B-v0.1`                |
| `MllamaForConditionalGeneration`     | Llama 3.2                          | `meta-llama/Llama-3.2-11B-Vision`            |
| `NemotronForCausalLM`                | Nemotron-3, Nemotron-4, Minitron   | `nvidia/Minitron-8B-Base`                    |
| `NemotronHForCausalLM`               | Nemotron-3-Nano, Nemotron-3-Super  | `nvidia/nvidia-nemotron-v3`                  |
| `NemotronNASForCausalLM`             | NemotronNAS                        | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`     |
| `Phi3ForCausalLM`                    | Phi-4                              | `microsoft/Phi-4`                            |
| `Qwen2ForCausalLM`                   | QwQ, Qwen2                         | `Qwen/Qwen2-7B-Instruct`                     |
| `Qwen2ForProcessRewardModel`         | Qwen2-based                        | `Qwen/Qwen2.5-Math-PRM-7B`                   |
| `Qwen2ForRewardModel`                | Qwen2-based                        | `Qwen/Qwen2.5-Math-RM-72B`                   |
| `Qwen3ForCausalLM`                   | Qwen3                              | `Qwen/Qwen3-8B`                              |
| `Qwen3MoeForCausalLM`                | Qwen3MoE                           | `Qwen/Qwen3-30B-A3B`                         |
| `Qwen3NextForCausalLM`               | Qwen3Next                          | `Qwen/Qwen3-Next-80B-A3B-Thinking`           |
| `Qwen3_5MoeForCausalLM` [^5]         | Qwen3.5-MoE                        | `Qwen/Qwen3.5-397B-A17B`                     |


## Model-Feature Support Matrix (Key Models)

Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.

| Model Architecture/Feature     | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | MTP | EAGLE-3(One Model Engine) — Linear | EAGLE-3(One Model Engine) — Dynamic | EAGLE-3(Two Model Engine) | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Sliding Window Attention | Logits Post Processor | Guided Decoding |
| ------------------------------ | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | --- | ---------------------------------- | ----------------------------------- | ------------------------- | ------------- | ---------------- | -------------- | ------------------------ | --------------------- | --------------- |
| `DeepseekV3ForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes [^1]        | Yes | No                                 | No                                  | No                        | Yes           | Yes              | Yes [^2]       | N/A                      | Yes                   | Yes             |
| `DeepseekV32ForCausalLM`         | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | No                                 | No                                  | No                        | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Glm4MoeForCausalLM`             | Yes               | Yes        | Yes                        | Untested              | Yes             | Yes | No                                 | No                                  | No                        | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `Qwen3MoeForCausalLM`            | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                                | Yes                                 | Yes                       | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Qwen3NextForCausalLM` [^3]          | Yes                | Yes        | Yes                         | Untested                    | Yes              | No  | No                                 | No                                  | No                        | Yes            | Yes               | No             | No                       | Untested                    | Untested              |
| `Llama4ForConditionalGeneration` | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                                | Yes                                 | Yes                       | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `GptOssForCausalLM`            | Yes              | Yes         | Yes                        | Yes                   | Yes             | No   | Yes                                | No                                  | Yes [^4]                   | Yes           | Yes              | Yes             | N/A                      | Yes                    | Yes             |
| `Qwen3_5MoeForCausalLM` [^5]  | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                                 | No                                  | No                        | Yes           | Untested         | Yes       | N/A                      | Untested              | Untested        |
| `Glm4MoeLiteForCausalLM` [^6] | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                                 | No                                  | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `NemotronHForCausalLM` (Super) | Yes               | Yes        | Untested                   | Untested              | Yes             | Yes | No                                 | No                                  | No                        | Yes           | Yes              | Untested       | N/A                      | Untested              | Untested        |

[^1]: Chunked Prefill for MLA can only be enabled on SM100/SM103.
[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103 and in BF16/FP8 KV cache dtype.
[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.
[^4]: Overlap scheduler isn't supported when using EAGLE-3(Two Model Engine) for GPT-OSS.
[^5]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml).
[^6]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml).
[^7]: Text-only support via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See AD configs for [MoE](../../../examples/auto_deploy/model_registry/configs/gemma4_moe.yaml) and [dense](../../../examples/auto_deploy/model_registry/configs/gemma4_dense.yaml).
[^8]: Text-only support via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/gemma3n_e2b_it.yaml).


# Multimodal Feature Support Matrix (PyTorch Backend)

| Model Architecture/Feature           | Overlap Scheduler | CUDA Graph | Chunked Prefill | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Logits Post Processor | EPD Disaggregated Serving | Modality  |
| ------------------------------------ | ----------------- | ---------- | --------------- | ------------- | ---------------- | -------------- | --------------------- | ------------------------- | --------- |
| `Gemma3ForConditionalGeneration`     | Yes               | Yes        | N/A             | Yes           | Yes              | N/A            | Yes                   | No                        | L + I     |
| `HCXVisionForCausalLM`               | Yes               | Yes        | No              | Yes           | Yes              | Yes            | Yes                   | No                        | L + I     |
| `LlavaLlamaModel (VILA)`             | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        | L + I + V |
| `LlavaNextForConditionalGeneration`  | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I     |
| `Llama4ForConditionalGeneration`     | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        | L + I     |
| `Mistral3ForConditionalGeneration`   | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I     |
| `NemotronH_Nano_VL_V2`               | Yes               | Yes        | Yes             | Yes           | Yes              | N/A            | Yes                   | No                        | L + I + V |
| `Phi4MMForCausalLM`                  | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I + A |
| `Qwen2VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I + V |
| `Qwen2_5_VLForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLMoeForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |

Note:
- L: Language
- I: Image
- V: Video
- A: Audio

# Visual Generation Models

TensorRT-LLM provides beta support for diffusion-based image and video generation.
For full documentation, see the [Visual Generation](./visual-generation.md) page.

## Supported Models

| HuggingFace Model ID | Tasks |
|---|---|
| `black-forest-labs/FLUX.1-dev` | Text-to-Image |
| `black-forest-labs/FLUX.2-dev` | Text-to-Image |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-Video |
| `Lightricks/LTX-2` | Text-to-Video (with Audio), Image-to-Video (with Audio) |

## Feature Matrix

| Model | TeaCache | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve |
|---|---|---|---|---|---|---|---|
| **FLUX.1** | Yes | No [^vg1] | Yes | No | Yes | Yes | Yes |
| **FLUX.2** | Yes | No [^vg1] | Yes | No | Yes | Yes | Yes |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.2** | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **LTX-2** | No | Yes | Yes | No | No | Yes | Yes |

[^vg1]: FLUX models use embedded guidance and do not have a separate negative prompt path, so CFG parallelism is not applicable.
