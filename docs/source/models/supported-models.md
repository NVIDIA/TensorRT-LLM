(support-matrix)=
# Supported Models

The following is a table of supported models for the PyTorch backend:

| Architecture                         | Model                              | HuggingFace Example                          |
| ------------------------------------ | ---------------------------------- | -------------------------------------------- |
| `AfmoeForCausalLM`                   | Arcee Foundation MoE (Trinity)     | `arcee-ai/Trinity-Mini`                      |
| `BertForSequenceClassification`      | BERT-based                         | `textattack/bert-base-uncased-yelp-polarity` |
| `Cohere2ForCausalLM`                 | Command A                          | `CohereLabs/c4ai-command-a-03-2025`          |
| `DeciLMForCausalLM`                  | Nemotron                           | `nvidia/Llama-3_1-Nemotron-51B-Instruct`     |
| `DeepSeekV2ForCausalLM` [^5]         | DeepSeek V2                        | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` |
| `DeepseekV3ForCausalLM`              | DeepSeek-V3, Kimi-K2               | `deepseek-ai/DeepSeek-V3`                    |
| `DeepseekV32ForCausalLM`             | DeepSeek-V3.2                      | `deepseek-ai/DeepSeek-V3.2`                  |
| `DeepseekV4ForCausalLM` [^11]        | DeepSeek-V4                        | `deepseek-ai/DeepSeek-V4-Pro`                |
| `ExaoneForCausalLM` [^5]             | EXAONE 3.5                         | `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`        |
| `Exaone4ForCausalLM`                 | EXAONE 4.0                         | `LGAI-EXAONE/EXAONE-4.0-32B`                 |
| `ExaoneMoEForCausalLM`               | K-EXAONE                           | `LGAI-EXAONE/K-EXAONE-236B-A23B`             |
| `Gemma3ForCausalLM`                  | Gemma 3                            | `google/gemma-3-1b-it`                       |
| `Gemma3nForConditionalGeneration` [^7]| Gemma 3n                           | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it` |
| `Gemma4ForConditionalGeneration`     | Gemma 4                            | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it` [^6], `google/gemma-4-31B-it` [^6] |
| `Gemma4UnifiedForConditionalGeneration` | Gemma 4 12B Unified (encoder-free) | `google/gemma-4-12B`, `google/gemma-4-12B-it`              |
| `Glm4MoeForCausalLM`                 | GLM-4.5, GLM-4.6, GLM-4.7          | `THUDM/GLM-4-100B-A10B`                      |
| `Glm4MoeLiteForCausalLM` [^5]        | GLM-4.7-Flash                      | `zai-org/GLM-4.7-Flash`                      |
| `GlmMoeDsaForCausalLM`               | GLM-5                              | `zai-org/GLM-5`                              |
| `GraniteForCausalLM` [^5]            | Granite 3, Granite Guardian 3      | `ibm-granite/granite-3.1-8b-instruct`, `ibm-granite/granite-3.3-8b-instruct`, `ibm-granite/granite-guardian-3.2-5b` |
| `GraniteMoeHybridForCausalLM` [^5]   | Granite 4.0 Hybrid MoE             | `ibm-granite/granite-4.0-h-small`            |
| `GptOssForCausalLM`                  | GPT-OSS                            | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`  |
| `HunYuanDenseForCausalLM` [^5]       | Hunyuan Dense                      | `tencent/Hunyuan-7B-Instruct`                |
| `HunYuanMoEForCausalLM` [^5]         | Hunyuan MoE                        | `tencent/Hunyuan-A13B-Instruct`              |
| `InternLM3ForCausalLM` [^5]          | InternLM3                          | `internlm/internlm3-8b-instruct`             |
| `KimiK25ForConditionalGeneration`    | Kimi-K2.5                          | `moonshotai/Kimi-K2.5`                       |
| `LagunaForCausalLM`                  | Laguna-XS                          | `poolside/laguna-XS.2`                       |
| `LlamaForCausalLM`                   | Llama 3.1, Llama 3, Llama 2, LLaMA | `meta-llama/Meta-Llama-3.1-70B`              |
| `Llama4ForConditionalGeneration`     | Llama 4                            | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  |
| `MiniMaxM2ForCausalLM` [^5]          | MiniMax M2/M2.1/M2.7              | `MiniMaxAI/MiniMax-M2.7`                    |
| `MiniMaxM3SparseForConditionalGeneration` [^12]| MiniMax-M3                       | `MiniMaxAI/MiniMax-M3`                      |
| `MistralForCausalLM`                 | Mistral                            | `mistralai/Mistral-7B-v0.1`                  |
| `MixtralForCausalLM`                 | Mixtral                            | `mistralai/Mixtral-8x7B-v0.1`                |
| `MllamaForConditionalGeneration`     | Llama 3.2                          | `meta-llama/Llama-3.2-11B-Vision`            |
| `NemotronForCausalLM`                | Nemotron-3, Nemotron-4, Minitron   | `nvidia/Minitron-8B-Base`                    |
| `NemotronHForCausalLM`               | Nemotron-3-Nano, Nemotron-3-Super, Nemotron-3-Ultra | `nvidia/nvidia-nemotron-v3`                  |
| `NemotronNASForCausalLM`             | NemotronNAS                        | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`     |
| `Olmo3ForCausalLM` [^5]              | OLMo 3, OLMo 3.1                   | `allenai/Olmo-3.1-32B-Instruct`              |
| `OpenELMForCausalLM` [^5]            | OpenELM                            | `apple/OpenELM-270M-Instruct`                |
| `Phi3ForCausalLM`                    | Phi-4                              | `microsoft/Phi-4`                            |
| `Qwen2ForCausalLM`                   | QwQ, Qwen2                         | `Qwen/Qwen2-7B-Instruct`                     |
| `Qwen2ForProcessRewardModel`         | Qwen2-based                        | `Qwen/Qwen2.5-Math-PRM-7B`                   |
| `Qwen2ForRewardModel`                | Qwen2-based                        | `Qwen/Qwen2.5-Math-RM-72B`                   |
| `Qwen3ForCausalLM`                   | Qwen3                              | `Qwen/Qwen3-8B`                              |
| `Qwen3MoeForCausalLM`                | Qwen3MoE                           | `Qwen/Qwen3-30B-A3B`                         |
| `Qwen3NextForCausalLM`               | Qwen3Next                          | `Qwen/Qwen3-Next-80B-A3B-Thinking`           |
| `Qwen3_5MoeForCausalLM`              | Qwen3.5-MoE                        | `Qwen/Qwen3.5-397B-A17B`                     |
| `SeedOssForCausalLM` [^5]            | Seed OSS, Seed-Coder               | `ByteDance-Seed/Seed-OSS-36B-Instruct`       |
| `SkyworkR1V2ForConditionalGeneration` [^5] | Skywork R1V2, Skywork SWE    | `Skywork/Skywork-R1V2-38B`                   |
| `SmolLM3ForCausalLM` [^5]            | SmolLM3                            | `HuggingFaceTB/SmolLM3-3B`                   |
| `Step3p7ForConditionalGeneration` [^8]| Step-3.7-Flash                    | `stepfun-ai/Step-3.7-Flash`                  |


## Model-Feature Support Matrix (Key Models)

Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.

| Model Architecture/Feature       | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | MTP | EAGLE-3 — Linear | EAGLE-3 — Dynamic | DFlash | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Sliding Window Attention | Logits Post Processor | Guided Decoding |
| -------------------------------- | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | --- | ---------------- | ----------------- | ------ | ------------- | ---------------- | -------------- | ------------------------ | --------------------- | --------------- |
| `DeepseekV3ForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes [^1]        | Yes | No               | No                | No     | Yes           | Yes              | Yes [^2]       | N/A                      | Yes                   | Yes             |
| `DeepseekV32ForCausalLM`         | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | No               | No                | No     | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `DeepseekV4ForCausalLM` [^11]    | Yes               | Yes        | Yes                        | Untested              | Yes             | Yes | No               | No                | No     | Yes           | Yes              | Untested       | Yes                      | Untested              | Untested        |
| `Glm4MoeForCausalLM`             | Yes               | Yes        | Yes                        | Untested              | Yes             | Yes | No               | No                | No     | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `Qwen3MoeForCausalLM`            | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes              | Yes               | No     | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Qwen3NextForCausalLM` [^3]      | Yes               | Yes        | Yes                        | Untested              | Yes             | No  | No               | No                | No     | Yes           | Yes              | No             | No                       | Untested              | Untested        |
| `Qwen3_5MoeForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | No               | No                | No     | Yes           | Untested         | Yes            | N/A                      | Untested              | Untested        |
| `Llama4ForConditionalGeneration` | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes              | Yes               | No     | Yes           | Yes              | Untested       | N/A                      | Yes                   | Yes             |
| `GptOssForCausalLM`              | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes              | No                | Yes    | Yes           | Yes              | Yes            | N/A                      | Yes                   | Yes             |
| `Glm4MoeLiteForCausalLM` [^5]    | Yes               | Yes        | Untested                   | Untested              | Yes             | No  | No               | No                | No     | Yes           | Untested         | Untested       | N/A                      | Untested              | Untested        |
| `NemotronHForCausalLM`           | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | No               | No                | No     | Yes           | Yes              | Yes            | N/A                      | Untested              | Untested        |
| `Gemma4ForConditionalGeneration` | Untested          | Yes        | Untested                   | No                    | Yes             | No  | No               | No                | No     | Yes           | Untested         | No             | Yes                      | Untested              | Untested        |
| `Gemma4UnifiedForConditionalGeneration` | Untested          | Untested   | Untested                   | No                    | Yes             | No  | No               | No                | No     | Yes           | Untested         | No             | Yes                      | Untested              | Untested        |
| `Step3p7ForConditionalGeneration`| Yes               | Yes        | Yes                        | Untested              | Untested        | Yes | No               | No                | No     | Yes           | Untested         | Untested       | Yes                      | Untested              | Untested        |
| `MiniMaxM3SparseForConditionalGeneration` [^12] | Yes               | Yes        | Yes                        | Untested              | Untested        | No  | No               | No                | No     | Yes           | Untested         | No             | N/A                      | Untested              | Untested        |

[^1]: Chunked Prefill for MLA can only be enabled on SM100/SM103.
[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103 and in BF16/FP8 KV cache dtype.
[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.
[^5]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD Configs](../../../examples/auto_deploy/model_registry/configs).
[^6]: Also supports text-only inference via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend.
[^7]: Text-only support via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend.
[^8]: Supports text and image inputs. The vision tower runs in BF16 even when the text decoder is quantized (FP8 block-scale or NVFP4). The text decoder is also usable standalone (text-only) via the `Step3p5ForCausalLM` architecture.
[^9]: Audio modality only supported on E2B/E4B variants.
[^10]: Audio requires a checkpoint with a `sound_config` and is supported only on the full (non-disaggregated) model path, not the EPD disaggregated path.
[^11]: DeepSeek-V4 is only supported on Blackwell GPUs (`SM100+`). See the [DeepSeek-V4 example README](../../../examples/models/core/deepseek_v4/README.md) for setup and parallelism.
[^12]: Supports text, image, and video inputs over the block-sparse attention path. The published MXFP8 checkpoint is dequantized on load so the runtime sees an effectively BF16 model. The text decoder is also usable standalone (text-only) via the `MiniMaxM3SparseForCausalLM` architecture. KV cache reuse and MTP are not supported on the sparse-attention path in this release.
[^13]: The Cosmos 3 family also supports visual generation through the VisualGen API. See [Visual Generation Models](#visual-generation-models).

# Multimodal Feature Support Matrix (PyTorch Backend)

| Model Architecture/Feature           | Overlap Scheduler | CUDA Graph | Chunked Prefill | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Logits Post Processor | EPD Disaggregated Serving | Modality  |
| ------------------------------------ | ----------------- | ---------- | --------------- | ------------- | ---------------- | -------------- | --------------------- | ------------------------- | --------- |
| `Exaone4_5_ForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I + V |
| `Gemma3ForConditionalGeneration`     | Yes               | Yes        | N/A             | Yes           | Yes              | N/A            | Yes                   | No                        | L + I     |
| `Gemma4ForConditionalGeneration`     | Untested          | Yes        | Yes             | Yes           | Untested         | No             | Untested              | No                        | L + I + V + A [^9] |
| `Gemma4UnifiedForConditionalGeneration` | Untested          | Untested   | Untested        | Yes           | Untested         | No             | Untested              | No                        | L + I + A |
| `HCXVisionForCausalLM`               | Yes               | Yes        | No              | Yes           | Yes              | Yes            | Yes                   | No                        | L + I     |
| `LlavaLlamaModel (VILA)`             | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        | L + I + V |
| `LlavaNextForConditionalGeneration`  | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I     |
| `Llama4ForConditionalGeneration`     | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        | L + I     |
| `Mistral3ForConditionalGeneration`   | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I     |
| `NemotronH_Nano_VL_V2`               | Yes               | Yes        | Yes             | Yes           | Yes              | N/A            | Yes                   | Yes                       | L + I + V + A [^10] |
| `Phi4MMForCausalLM`                  | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I + A |
| `Qwen2VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | No                        | L + I + V |
| `Qwen2_5_VLForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLMoeForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Yes                   | Yes                       | L + I + V |
| `Step3p7ForConditionalGeneration`    | Yes               | Yes        | Untested        | Yes           | Untested         | Untested       | Untested              | Untested                  | L + I     |
| `MiniMaxM3SparseForConditionalGeneration` [^12] | Yes               | Yes        | Untested        | Yes           | Untested         | No             | Untested              | Untested                  | L + I + V |
| `Cosmos3ForConditionalGeneration` [^13] | Yes               | Yes        | Yes             | Yes           | Yes              | Yes            | Untested              | Untested                  | L + I + V |
| `Qwen3_5ForConditionalGeneration`    | Yes               | Yes        | Untested        | Yes           | Yes              | No             | Untested              | Yes                       | L + I + V |
| `Qwen3_5MoeForConditionalGeneration` | Yes               | Yes        | Untested        | Yes           | Yes              | No             | Untested              | Yes                       | L + I + V |

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
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Text-to-Video, Image-to-Video |
| `Lightricks/LTX-2` | Text-to-Video (with Audio), Image-to-Video (with Audio) |
| `Qwen/Qwen-Image` | Text-to-Image |
| `Qwen/Qwen-Image-2512` | Text-to-Image |
| `nvidia/Cosmos3-Nano` | Text-to-Image, Text-to-Video, Image-to-Video |
| `nvidia/Cosmos3-Super` | Text-to-Image, Text-to-Video, Image-to-Video |

### Feature Matrix

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve | Attention2D | Ring Attention | Tensor Parallelism |
|---|---|---|---|---|---|---|---|---|---|--|--|--|
| **FLUX.1** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **FLUX.2** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.2** | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **LTX-2** | Yes | Yes | No | Yes | Yes | No | No | Yes | Yes | Yes | Yes | No |
| **Qwen-Image** [^2] | Yes | Yes | No | No | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| **Cosmos3** | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes |

[^vg1]: FLUX models use embedded guidance and do not have a separate negative prompt path, so CFG parallelism is not applicable.
