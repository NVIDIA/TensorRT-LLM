(support-matrix)=
# Supported Models

The following is a table of supported models for the PyTorch backend:

| Architecture                         | Model                                      | HuggingFace Example                          |
| ------------------------------------ | ------------------------------------------ | -------------------------------------------- |
| `BertForSequenceClassification`      | BERT-based                                 | `textattack/bert-base-uncased-yelp-polarity` |
| `Cohere2ForCausalLM`                 | Command A, Aya Expanse                     | `CohereLabs/c4ai-command-a-03-2025`          |
| `DeciLMForCausalLM`                  | Nemotron                                   | `nvidia/Llama-3_1-Nemotron-51B-Instruct`     |
| `DeepseekV2ForCausalLM` [^7]        | DeepSeek-V2, Coder-V2                      | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` |
| `DeepseekV3ForCausalLM`              | DeepSeek-V3, DeepSeek-R1                   | `deepseek-ai/DeepSeek-V3`                    |
| `DeepseekV32ForCausalLM`             | DeepSeek-V3.2                              | `deepseek-ai/DeepSeek-V3.2`                  |
| `Exaone4ForCausalLM`                 | EXAONE 4.0                                 | `LGAI-EXAONE/EXAONE-4.0-32B`                 |
| `ExaoneForCausalLM` [^7]            | EXAONE 3.5                                 | `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`       |
| `ExaoneMoEForCausalLM`               | K-EXAONE                                   | `LGAI-EXAONE/K-EXAONE-236B-A23B`             |
| `Gemma2ForCausalLM` [^7]            | Gemma 2                                    | `google/gemma-2-9b-it`                       |
| `Gemma3ForCausalLM`                  | Gemma 3                                    | `google/gemma-3-1b-it`                       |
| `GemmaForCausalLM` [^7]             | Gemma, CodeGemma                           | `google/gemma-1.1-7b-it`                     |
| `Glm4MoeForCausalLM`                 | GLM-4.5, GLM-4.6, GLM-4.7, GLM-5          | `THUDM/GLM-4-100B-A10B`                      |
| `Glm4MoeLiteForCausalLM` [^6]       | GLM-4.7-Flash                              | `zai-org/GLM-4.7-Flash`                      |
| `GptOssForCausalLM`                  | GPT-OSS                                    | `openai/gpt-oss-120b`                        |
| `GraniteMoeHybridForCausalLM` [^7]  | Granite 4.0 HybridMoE                      | `ibm-granite/granite-4.0-micro`              |
| `HunYuanDenseV1ForCausalLM` [^7]    | Hunyuan Dense, Hunyuan-MT                  | `tencent/Hunyuan-7B-Instruct`                |
| `HunYuanMoEV1ForCausalLM` [^7]      | Hunyuan MoE                                | `tencent/Hunyuan-A13B-Instruct`              |
| `InternLM2ForCausalLM` [^7]         | InternLM 3                                 | `internlm/internlm3-8b-instruct`            |
| `JambaForCausalLM` [^7]             | Jamba                                      | `ai21labs/AI21-Jamba-1.5-Mini`               |
| `KimiK2ForCausalLM` [^7]            | Kimi K2.5                                  | `moonshotai/Kimi-K2.5`                       |
| `LlamaForCausalLM`                   | Llama 3.1, Llama 3, Llama 2, CodeLlama, SmolLM, Falcon 3 | `meta-llama/Meta-Llama-3.1-70B`  |
| `Llama4ForConditionalGeneration`     | Llama 4                                    | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  |
| `MiniMaxM2ForCausalLM`               | MiniMax M2/M2.1/M2.5                      | `MiniMaxAI/MiniMax-M2`                       |
| `MistralForCausalLM`                 | Mistral, Codestral, Ministral              | `mistralai/Mistral-7B-v0.1`                  |
| `MixtralForCausalLM`                 | Mixtral                                    | `mistralai/Mixtral-8x7B-v0.1`                |
| `MllamaForConditionalGeneration`     | Llama 3.2                                  | `meta-llama/Llama-3.2-11B-Vision`            |
| `NemotronFlashForCausalLM` [^7]     | Nemotron Flash                             | `nvidia/Nemotron-Flash-3B-Instruct`          |
| `NemotronForCausalLM`                | Nemotron-3, Nemotron-4, Minitron           | `nvidia/Minitron-8B-Base`                    |
| `NemotronHForCausalLM`               | Nemotron-H, Nemotron-3-Nano, Nemotron-3-Super | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| `NemotronNASForCausalLM`             | NemotronNAS                                | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`     |
| `Olmo2ForCausalLM` [^7]             | OLMo 2, OLMo 3                            | `allenai/Olmo-3-7B-Instruct`                |
| `OpenELMForCausalLM` [^7]           | OpenELM                                    | `apple/OpenELM-3B-Instruct`                 |
| `Phi3ForCausalLM`                    | Phi-4, Phi-3                               | `microsoft/Phi-4`                            |
| `Qwen2ForCausalLM`                   | QwQ, Qwen2, Qwen2.5, Seed-Coder           | `Qwen/Qwen2-7B-Instruct`                     |
| `Qwen2ForProcessRewardModel`         | Qwen2-based                                | `Qwen/Qwen2.5-Math-PRM-7B`                   |
| `Qwen2ForRewardModel`                | Qwen2-based                                | `Qwen/Qwen2.5-Math-RM-72B`                   |
| `Qwen2MoeForCausalLM` [^7]          | Qwen2-MoE, Seed-OSS, MiMo                 | `XiaomiMiMo/MiMo-V2-Flash`                  |
| `Qwen3ForCausalLM`                   | Qwen3, Qwen3.5 Dense                      | `Qwen/Qwen3-8B`                              |
| `Qwen3MoeForCausalLM`                | Qwen3-MoE, Qwen3-Coder                    | `Qwen/Qwen3-30B-A3B`                         |
| `Qwen3NextForCausalLM`               | Qwen3Next                                  | `Qwen/Qwen3-Next-80B-A3B-Thinking`           |
| `Qwen3_5MoeForCausalLM` [^5]        | Qwen3.5-MoE                               | `Qwen/Qwen3.5-397B-A17B`                     |
| `Starcoder2ForCausalLM` [^7]        | StarCoder 2                                | `bigcode/starcoder2-15b`                     |


## Model-Feature Support Matrix (Key Models)

Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.

| Model Architecture/Feature     | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | MTP | EAGLE-3(One Model Engine) | EAGLE-3(Two Model Engine) | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Sliding Window Attention | Logits Post Processor | Guided Decoding |
| ------------------------------ | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | --- | ------------------------- | ------------------------- | ------------- | ---------------- | -------------- | ------------------------ | --------------------- | --------------- |
| `DeepseekV3ForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes [^1]        | Yes | No                        | No                        | Yes           | Yes              | Yes [^2]       | N/A                      | Yes                   | Yes             |
| `DeepseekV32ForCausalLM`         | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | No                        | No                        | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Glm4MoeForCausalLM`             | Yes               | Yes        | Yes                        | Untested              | Yes             | Yes | No                        | No                        | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `Qwen3MoeForCausalLM`            | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                       | Yes                       | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Qwen3NextForCausalLM` [^3]          | Yes                | Yes        | No                         | Untested                    | Yes              | No  | No                        | No                        | Yes            | Yes               | No             | No                       | Untested                    | Untested              |
| `Llama4ForConditionalGeneration` | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                       | Yes                       | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `GptOssForCausalLM`            | Yes              | Yes         | Yes                        | Yes                   | Yes             | No   | Yes                       | Yes [^4]                   | Yes           | Yes              | Yes             | N/A                      | Yes                    | Yes             |
| `Qwen3_5MoeForCausalLM` [^5]  | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Yes       | N/A                      | Untested              | Untested        |
| `Glm4MoeLiteForCausalLM` [^6] | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `DeepseekV2ForCausalLM` [^7]  | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `Gemma2ForCausalLM` [^7]      | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `GemmaForCausalLM` [^7]       | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `GraniteMoeHybridForCausalLM` [^7] | Yes          | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `HunYuanDenseV1ForCausalLM` [^7] | Yes             | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `JambaForCausalLM` [^7]       | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `KimiK2ForCausalLM` [^7]      | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `NemotronFlashForCausalLM` [^7] | Yes              | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `OpenELMForCausalLM` [^7]     | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `ExaoneForCausalLM` [^7]     | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `HunYuanMoEV1ForCausalLM` [^7] | Yes             | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `InternLM2ForCausalLM` [^7] | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `Olmo2ForCausalLM` [^7]     | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `Qwen2MoeForCausalLM` [^7]  | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `Starcoder2ForCausalLM` [^7]  | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No                        | No                        | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |

[^1]: Chunked Prefill for MLA can only be enabled on SM100/SM103.
[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103 and in BF16/FP8 KV cache dtype.
[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.
[^4]: Overlap scheduler isn't supported when using EAGLE-3(Two Model Engine) for GPT-OSS.
[^5]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml).
[^6]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml).
[^7]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend.


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

For diffusion-based image and video generation models, see the [Visual Generation](./visual-generation.md) documentation.
