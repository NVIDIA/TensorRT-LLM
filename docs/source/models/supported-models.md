(support-matrix)=
# Supported Models

The following is a table of supported models for the PyTorch backend:

| Architecture                         | Model                              | HuggingFace Example                          |
| ------------------------------------ | ---------------------------------- | -------------------------------------------- |
| `AfmoeForCausalLM`                   | Arcee Foundation MoE (Trinity)     | `arcee-ai/Trinity-Mini`                      |
| `BartForConditionalGeneration`       | BART                               | `facebook/bart-large-cnn`                    |
| `BertForSequenceClassification`      | BERT-based                         | `textattack/bert-base-uncased-yelp-polarity` |
| `Cohere2ForCausalLM`                 | Command A                          | `CohereLabs/c4ai-command-a-03-2025`          |
| `DeciLMForCausalLM`                  | Nemotron                           | `nvidia/Llama-3_1-Nemotron-51B-Instruct`     |
| `DeepseekV3ForCausalLM`              | DeepSeek-V3, Kimi-K2               | `deepseek-ai/DeepSeek-V3`                    |
| `DeepseekV32ForCausalLM`             | DeepSeek-V3.2                      | `deepseek-ai/DeepSeek-V3.2`                  |
| `DeepseekV4ForCausalLM` [^11]        | DeepSeek-V4                        | `deepseek-ai/DeepSeek-V4-Pro`                |
| `DeepseekV41ForCausalLM`            | DeepSeek-V4.1 (text decoder)       | —                                          |
| `DeepseekV41ForConditionalGeneration` | DeepSeek-V4.1                    | —                                          |
| `Exaone4ForCausalLM`                 | EXAONE 4.0                         | `LGAI-EXAONE/EXAONE-4.0-32B`                 |
| `ExaoneMoeForCausalLM`               | K-EXAONE, K-EXAONE 2.0             | `LGAI-EXAONE/K-EXAONE-236B-A23B`, `LGAI-EXAONE/K-EXAONE-2.0-750B-A37B` |
| `Gemma3ForCausalLM`                  | Gemma 3                            | `google/gemma-3-1b-it`                       |
| `Gemma4ForConditionalGeneration`     | Gemma 4                            | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` |
| `Gemma4UnifiedForConditionalGeneration` | Gemma 4 12B Unified (encoder-free) | `google/gemma-4-12B`, `google/gemma-4-12B-it`              |
| `Gemma4AssistantForCausalLM`         | Gemma 4 MTP assistant              | `google/gemma-4-E2B-it-assistant`, `google/gemma-4-E4B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant`, `google/gemma-4-31B-it-assistant` |
| `Glm4MoeForCausalLM`                 | GLM-4.5, GLM-4.6, GLM-4.7          | `THUDM/GLM-4-100B-A10B`                      |
| `GlmMoeDsaForCausalLM`               | GLM-5, GLM-5.2, GLM-5.3            | `zai-org/GLM-5`, `zai-org/GLM-5.2`, `zai-org/GLM-5.3` |
| `Glm5NextForConditionalGeneration` [^20] | GLM-5.3-Flash                 | `zai-org/GLM-5.3-Flash`                     |
| `GptOssForCausalLM`                  | GPT-OSS                            | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`  |
| `KimiK25ForConditionalGeneration`    | Kimi-K2.5                          | `moonshotai/Kimi-K2.5`                       |
| `KimiK3ForConditionalGeneration` [^15]| Kimi-K3                            | `moonshotai/Kimi-K3`                         |
| `KimiLinearForCausalLM` [^15]        | Kimi-K3 (text decoder)             | `moonshotai/Kimi-K3`                         |
| `LagunaForCausalLM`                  | Laguna-XS                          | `poolside/laguna-XS.2`                       |
| `LiLiCorrDraftModel`                | [LiLiCorr draft model](../features/speculative-decoding.md#dflash) | Compatible checkpoint via `DFlashDecodingConfig.speculative_model` |
| `LlamaForCausalLM`                   | Llama 3.1, Llama 3, Llama 2, LLaMA | `meta-llama/Meta-Llama-3.1-70B`              |
| `Llama4ForConditionalGeneration`     | Llama 4                            | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  |
| `MBartForConditionalGeneration`      | mBART                              | `facebook/mbart-large-50-many-to-one-mmt`    |
| `MiniCPMV4_6ForConditionalGeneration` [^14]| MiniCPM-V 4.6                    | `openbmb/MiniCPM-V-4.6`                      |
| `MiniMaxM3SparseForConditionalGeneration` [^12]| MiniMax-M3                       | `MiniMaxAI/MiniMax-M3`                      |
| `MistralForCausalLM`                 | Mistral                            | `mistralai/Mistral-7B-v0.1`                  |
| `MixtralForCausalLM`                 | Mixtral                            | `mistralai/Mixtral-8x7B-v0.1`                |
| `MllamaForConditionalGeneration`     | Llama 3.2                          | `meta-llama/Llama-3.2-11B-Vision`            |
| `NemotronForCausalLM`                | Nemotron-3, Nemotron-4, Minitron   | `nvidia/Minitron-8B-Base`                    |
| `NemotronHForCausalLM`               | Nemotron-3-Nano, Nemotron-3-Super, Nemotron-3-Ultra | `nvidia/nvidia-nemotron-v3`                  |
| `NemotronNASForCausalLM`             | NemotronNAS                        | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`     |
| `Phi3ForCausalLM`                    | Phi-4                              | `microsoft/Phi-4`                            |
| `Qwen2ForCausalLM`                   | QwQ, Qwen2                         | `Qwen/Qwen2-7B-Instruct`                     |
| `Qwen2ForProcessRewardModel`         | Qwen2-based                        | `Qwen/Qwen2.5-Math-PRM-7B`                   |
| `Qwen2ForRewardModel`                | Qwen2-based                        | `Qwen/Qwen2.5-Math-RM-72B`                   |
| `Qwen3ForCausalLM`                   | Qwen3                              | `Qwen/Qwen3-8B`                              |
| `Qwen3ForTextEmbedding`              | Qwen3-Embedding                    | `Qwen/Qwen3-Embedding-8B`                    |
| `Qwen3MoeForCausalLM`                | Qwen3MoE                           | `Qwen/Qwen3-30B-A3B`                         |
| `Qwen3NextForCausalLM`               | Qwen3Next                          | `Qwen/Qwen3-Next-80B-A3B-Thinking`           |
| `Qwen3_5MoeForCausalLM`              | Qwen3.8-MoE, Qwen3.5-MoE           | `Qwen/Qwen3.8-2.4T-A95B`, `Qwen/Qwen3.5-397B-A17B` |
| `Qwen4ExpForCausalLM`                | Qwen3.8-Flash-Next (text)          | `Qwen/Qwen3.8-Flash-Next`                    |
| `Qwen4ExpForConditionalGeneration`   | Qwen3.8-Flash-Next                 | `Qwen/Qwen3.8-Flash-Next`                    |
| `Step3p7ForConditionalGeneration` [^8]| Step-3.7-Flash                    | `stepfun-ai/Step-3.7-Flash`                  |
| `T5ForConditionalGeneration`         | T5, Flan-T5, ByT5                  | `google/flan-t5-small`                       |
| `WhisperForConditionalGeneration`    | Whisper                            | `openai/whisper-large-v3`                    |


## Models Supported outside main branch

Some newly released models are first supported on a dedicated feature branch, with release
containers published from that branch, before the support is merged into `main`. The entries
below are temporary: each entry is removed once the model's support, including its deployment
guide, is fully merged into `main` and available in a regular release. Until then, use the
linked branch and container rather than `main`.

The linked containers are preview releases published from the side branch and are retained on
NGC for roughly two months. If a tag is no longer available, check the
[release tags page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)
for a newer tag published from the same branch, or build the container from the branch sources
following its [Docker guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/README.md).

| Model | Side branch | Deployment guide | Release container |
| ----- | ----------- | ---------------- | ----------------- |
| Muse Spark Open | [`feat/musespark-open`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/musespark-open) | [Muse Spark Open deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/docs/source/deployment-guide/deployment-guide-for-musespark-open-on-trtllm.md), [speculative decoding guide](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/docs/source/deployment-guide/deployment-guide-for-musespark-open-speculative-decoding-on-trtllm.md) | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0a0.dev202610171236` |
| MiniMax-M3.1-Flash | [`feat/minimax-m3.1-flash`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/minimax-m3.1-flash) | [Serving MiniMax-M3.1-Flash](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/minimax-m3.1-flash/docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md#serving-minimax-m31-flash) in the MiniMax-M3 deployment guide | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0a0.dev202610230958` |
| Beam (Reflection AI) | [`feat/reflection-beam`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/reflection-beam) | [Beam deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/reflection-beam/docs/source/deployment-guide/deployment-guide-for-beam-on-trtllm.md) | No container published yet — build from the branch sources per the guide |

## Model-Feature Support Matrix (Key Models)

Note: Support for other models may vary. Features marked "N/A" are not applicable to the model architecture.
This matrix covers language-only requests, including the text-decoder path of multimodal architectures. The multimodal
matrix below independently covers requests with actual image, video, or audio input, so a feature can have different
statuses for the same architecture in the two matrices.

| Model Architecture/Feature       | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | Speculative Decoding [^18] | Torch Sampler | KV Cache Reuse | Sliding Window Attention | Logits Post Processor | Guided Decoding |
| -------------------------------- | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | -------------------------- | ------------- | -------------- | ------------------------ | --------------------- | --------------- |
| `DeepseekV3ForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes [^1]        | MTP                        | Yes           | Yes [^2]       | N/A                      | Yes                   | Yes             |
| `DeepseekV32ForCausalLM`         | Yes               | Yes        | Yes                        | Yes                   | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `GlmMoeDsaForCausalLM`           | Yes               | Yes        | Yes                        | Yes                   | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `DeepseekV4ForCausalLM` [^11]    | Yes               | Yes        | Yes                        | Untested              | Yes             | MTP                        | Yes           | Yes            | Yes                      | Yes                   | Yes             |
| `Glm4MoeForCausalLM`             | Yes               | Yes        | Yes                        | Untested              | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes [^16]       |
| `Qwen3MoeForCausalLM`            | Yes               | Yes        | Yes                        | Yes                   | Yes             | EAGLE-3 (Linear, Dynamic)  | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `Qwen3NextForCausalLM` [^3]      | Yes               | Yes        | Yes                        | Untested              | Yes             | No                         | Yes           | Yes [^19]      | No                       | Yes                   | Yes             |
| `Qwen3_5MoeForCausalLM`          | Yes               | Yes        | Yes                        | Yes                   | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `Qwen4ExpForCausalLM`            | Yes               | Yes        | Yes                        | Yes                   | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `Llama4ForConditionalGeneration` | Yes               | Yes        | Yes                        | Yes                   | Yes             | EAGLE-3 (Linear, Dynamic)  | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `GptOssForCausalLM`              | Yes               | Yes        | Yes                        | Yes                   | Yes             | EAGLE-3 (Linear), DFlash   | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `KimiK3ForConditionalGeneration` [^15] [^17] | Yes               | Yes        | Yes                        | Yes                   | Yes             | DSpark                     | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `NemotronHForCausalLM`           | Yes               | Yes        | Yes                        | Yes                   | Yes             | MTP                        | Yes           | Yes            | N/A                      | Yes                   | Yes             |
| `Gemma4ForConditionalGeneration` | Yes                | Yes        | Yes                        | No                    | Yes             | MTP                        | Yes           | No             | Yes                      | Yes                   | Yes             |
| `Gemma4UnifiedForConditionalGeneration` | Yes                | Yes        | Yes                        | No                    | Yes             | No                         | Yes           | No             | Yes                      | Yes                   | Yes             |
| `Step3p7ForConditionalGeneration`| Yes               | Yes        | Yes                        | Untested              | Untested        | MTP                        | Yes           | No             | Yes                      | Yes                   | Yes             |
| `MiniMaxM3SparseForConditionalGeneration` [^12] | Yes               | Yes        | Yes                        | Untested              | Untested        | EAGLE-3 (Linear), DSpark [^21] | Yes           | No             | N/A                      | Yes                   | Yes             |

[^1]: Chunked Prefill for MLA can only be enabled on SM90/SM100/SM103/SM107/SM120.
[^2]: KV cache reuse for MLA can only be enabled on SM90/SM100/SM103/SM107/SM120/SM121 and in BF16/FP8 KV cache dtype.
[^3]: Qwen3-Next-80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark.
[^8]: Supports text and image inputs. The vision tower runs in BF16 even when the text decoder is quantized (FP8 block-scale or NVFP4). The text decoder is also usable standalone (text-only) via the `Step3p5ForCausalLM` architecture.
[^9]: Audio modality only supported on E2B/E4B variants.
[^10]: Audio requires a checkpoint with a `sound_config` and is supported only on the full (non-disaggregated) model path, not the EPD disaggregated path.
[^11]: DeepSeek-V4 is only supported on Blackwell GPUs (`SM100+`). See the [DeepSeek-V4 example README](source:examples/models/core/deepseek_v4/README.md) for setup and parallelism.
[^12]: Supports text, image, and video inputs over the block-sparse attention path. The published MXFP8 checkpoint is dequantized on load so the runtime sees an effectively BF16 model. The text decoder is also usable standalone (text-only) via the `MiniMaxM3SparseForCausalLM` architecture. KV cache reuse and MTP are not supported on the sparse-attention path in this release.
[^13]: Cosmos3-Nano and Cosmos3-Super Reasoner checkpoints use `Cosmos3ForConditionalGeneration`. Cosmos3-Edge Reasoner uses `Cosmos3EdgeForConditionalGeneration` (SigLIP2 vision encoder, 4608-11520-2048 projector, dense ReLU-squared decoder, interleaved mRoPE) and is BF16. The same Cosmos 3 checkpoints also support visual generation through the VisualGen API. See [Visual Generation Models](#visual-generation-models).
[^14]: Requires `transformers>=5.7.0`: MiniCPM-V 4.6 was upstreamed into transformers as a native model type (`minicpmv4_6`) and the checkpoint ships no remote code (`auto_map`) to fall back on. The Qwen3.5-hybrid text tower runs in BF16. Image, video, and text inputs are supported in this release (video reuses the same NaViT-packed vision path as image via `MiniCPMV4_6InputProcessor`).
[^15]: Kimi K3 is only supported on NVIDIA Blackwell GPUs (`SM100` family); see the [Kimi K3 deployment guide](../deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md). Which recipes fit is set by the attention layout rather than by the GPU count: DEP16 (`enable_attention_dp: true`) replicates the BF16 non-expert weights on every rank (114 GB) on top of the MXFP4 routed experts at 16-way expert parallelism (90 GB), needing 210 GB per rank, so it requires GB300-class per-GPU memory. TEP16 (`enable_attention_dp: false`) shards those non-expert weights instead and needs 115 GB per rank; it is validated end-to-end on GB200 (`SM100`) at 16 GPUs. On B200 (`SM100`), kernel and module support is functional and covered by CI. Note that the FP8 weight-read path (TRTLLM-14765) does not relax the DEP16 requirement: the conversion runs after the weights are already resident in BF16, so it lowers the steady-state footprint but not the load-time peak.
[^16]: Guided decoding for `Glm4MoeForCausalLM` is currently supported only with the `xgrammar` backend; `llguidance` is not working.
[^17]: Kimi K3 has no MTP or EAGLE-3 head, and its DSpark checkpoints are not compatible with plain `DFlash`. Speculators-format DSpark checkpoints require `aux_hidden_state_layer_ids` to be a nonempty list of positive integers; booleans, floats and strings are rejected rather than coerced.
[^18]: NGram and standalone Suffix Automaton (SA) use model-free drafting on the PyTorch backend, so they are not listed in individual entries. This does not imply universal end-to-end support: compatibility depends on each model's multi-token verification and cache-management paths and may be untested or explicitly restricted.
[^19]: KV cache reuse for hybrid recurrent-attention models requires an explicit recurrent-state snapshot policy, such as `kv_cache_config.mamba_state_config.periodic_snapshot_interval`; the model default disables reuse when no snapshot policy is configured.
[^20]: Supports text, image, and video inputs, MTP (including with attention data parallelism), and FP8 KV cache. Image and video inputs require `transformers==5.17.0`. Beam search is not supported. See the [GLM-5.3-Flash deployment guide](../deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md) for setup and feature-specific requirements.
[^21]: MiniMax-M3.1-Flash ships a standalone DSpark drafter in its `dspark/` directory (architecture `DSparkMiniMaxDraftModel`). Load it with `speculative_config: {decoding_type: DSpark, speculative_model: <checkpoint>/dspark, max_draft_len: 7}` (the drafter's block size is 7 and `max_draft_len` must match it). Keep KV cache block reuse enabled (the default): with `kv_cache_config.enable_block_reuse: false`, or on a disaggregated generation server, prompts of about 131k tokens or more fail with `Draft KV cache context resize failed`.

# Encoder-Decoder Feature Support Matrix (PyTorch Backend)

The following capabilities apply to the supported encoder-decoder architectures. For configuration guidance and
limitations, see [Use encoder-decoder models with the PyTorch backend](./encoder-decoder.md).

| Model Architecture/Feature | Overlap Scheduler | Decoder CUDA Graph | Encoder CUDA Graph | KV Cache Manager V1 | KV Cache Manager V2 | Beam Search | Tensor Parallelism | Pipeline Parallelism | Chunked Prefill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BartForConditionalGeneration` | Yes | Yes (except FP32) | Yes | Yes | Yes (single beam) | Yes (V1 only) | Yes | No | No (encoder phase) |
| `MBartForConditionalGeneration` | Yes | Yes (except FP32) | Yes | Yes | Yes (single beam) | Yes (V1 only) | Yes | No | No (encoder phase) |
| `T5ForConditionalGeneration` | Yes | Yes (except FP32) | Yes | Yes | Yes (single beam) | Yes (V1 only) | Yes | No | No (encoder phase) |
| `WhisperForConditionalGeneration` | Yes | Yes (except FP32) | No (feature inputs) | Yes | Yes (single beam) | Yes (V1 only) | Yes | No | No (encoder phase) |

Decoder CUDA graphs support greedy and beam-search decoding with KV cache manager V1 and single-beam decoding with
V2. Encoder CUDA graphs support the token-input BART, mBART, and T5 families; Whisper's feature-driven audio encoder
runs eagerly. Use the `TRTLLM` attention backend for encoder-decoder models; tensor parallelism also requires attention
head counts divisible by the tensor parallel size. Chunked prefill is not supported for the encoder phase, so the
complete encoder input must fit in the iteration token budget.

# Multimodal Feature Support Matrix (PyTorch Backend)

Entries in this matrix require a request with at least one listed non-language modality; language-only test evidence is
not used to mark a feature as supported here.

| Model Architecture/Feature           | Overlap Scheduler | CUDA Graph | Chunked Prefill | Torch Sampler | KV Cache Reuse | Logits Post Processor | EPD Disaggregated Serving | Modality  |
| ------------------------------------ | ----------------- | ---------- | --------------- | ------------- | -------------- | --------------------- | ------------------------- | --------- |
| `Exaone4_5_ForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | No                        | L + I + V |
| `Gemma3ForConditionalGeneration`     | Yes               | Yes        | N/A             | Yes           | N/A            | Yes                   | No                        | L + I     |
| `Gemma4ForConditionalGeneration`     | Yes                | Yes        | Yes             | Yes           | No             | Yes                   | No                        | L + I + V + A [^9] |
| `Gemma4UnifiedForConditionalGeneration` | Yes                | Yes        | Untested        | Yes           | No             | Yes                   | No                        | L + I + A |
| `HCXVisionForCausalLM`               | Yes               | Yes        | No              | Yes           | Yes            | Yes                   | No                        | L + I     |
| `LlavaLlamaModel (VILA)`             | Yes               | Yes        | No              | Yes           | No             | Yes                   | No                        | L + I + V |
| `LlavaNextForConditionalGeneration`  | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | Yes                       | L + I     |
| `Llama4ForConditionalGeneration`     | Yes               | Yes        | No              | Yes           | No             | Yes                   | No                        | L + I     |
| `MiniCPMV4_6ForConditionalGeneration` [^14] | Yes               | Untested   | Untested        | Yes           | Untested       | Untested              | No                        | L + I + V |
| `Mistral3ForConditionalGeneration`   | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | No                        | L + I     |
| `NemotronH_Nano_VL_V2`               | Yes               | Yes        | Yes             | Yes           | N/A            | Yes                   | Yes                       | L + I + V + A [^10] |
| `Phi4MMForCausalLM`                  | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | No                        | L + I + A |
| `Qwen2VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | No                        | L + I + V |
| `Qwen2_5_VLForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLForConditionalGeneration`    | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | Yes                       | L + I + V |
| `Qwen3VLMoeForConditionalGeneration` | Yes               | Yes        | Yes             | Yes           | Yes            | Yes                   | Yes                       | L + I + V |
| `Step3p7ForConditionalGeneration`    | Yes               | Yes        | Untested        | Yes           | Untested       | Yes                   | Untested                  | L + I     |
| `MiniMaxM3SparseForConditionalGeneration` [^12] | Yes               | Yes        | Untested        | Yes           | No             | Yes                   | Untested                  | L + I + V |
| `Cosmos3ForConditionalGeneration` [^13] | Yes               | Yes        | Yes             | Yes           | Yes            | Untested              | Untested                  | L + I + V |
| `Cosmos3EdgeForConditionalGeneration` [^13] | Yes               | Yes        | Untested        | Yes           | Untested       | Untested              | Untested                  | L + I + V |
| `Qwen3_5ForConditionalGeneration`    | Yes               | Yes        | Untested        | Yes           | No             | Untested              | Yes                       | L + I + V |
| `Qwen3_5MoeForConditionalGeneration` | Yes               | Yes        | Untested        | Yes           | No             | Untested              | Yes                       | L + I + V |
| `Qwen4ExpForConditionalGeneration`   | Untested          | Untested   | Untested        | Yes           | Untested       | Yes                   | No                        | L + I     |

Note:
- L: Language
- I: Image
- V: Video
- A: Audio

## Multimodal Encoder Optimizations

The following optimizations are available to models that implement
`MultimodalModelMixin`.

### Encoder CUDA graphs

For encoders with CUDA graph support, set a maximum batch size in the model's
modality configuration. For example, pass this YAML through `trtllm-serve --config`:

```yaml
multimodal_config:
  encoder_cuda_graph:
    vision:
      max_batch_size: 8
```

The encoder adapter automatically generates capture shapes for batch sizes 1 through 8,
within `encoder_max_num_tokens` (or `max_num_tokens` when unset). Use the optional
per-modality `max_num_tokens` to limit capture memory and startup work. Batch size counts
independent encoder contexts: images, video frame groups, or audio chunks as defined by
the encoder. It does not count concurrent API requests or the scheduler's atomic items.
Larger or unsupported shapes fall back to eager execution. Graphs remain disabled by default.

Each supported tower reads its own modality entry. An audio adapter uses the same settings,
with its own shape, mask, and output rules:

```yaml
multimodal_config:
  encoder_cuda_graph:
    vision:
      max_batch_size: 8
    audio:
      max_batch_size: 4
      max_num_tokens: 2048
```

The second example requires a model with graph-enabled vision and audio towers. The current
production integrations are Qwen3-VL and related Qwen vision encoders, RADIO (`vision`), and
GLM5-Next (`image` instead of `vision`). The shared audio adapter interface is available,
but production audio towers still need an implementation and model validation. Requesting an
unsupported modality fails before any graphs are captured.

Advanced users can supply `buckets: [[256, 1], [512, 2]]` instead of `max_batch_size`
and `max_num_tokens`. Each pair specifies total real encoder tokens and an exact context count.
Automatic shapes use per-context capacities starting at 256 tokens (or the available budget
when smaller), doubling up to each batch size's token budget. Adapters can override these
shapes for additional alignment constraints. Padded replay is enabled by default; disabling it
requires an exact token-count match. Encoder graphs and encoder side-stream prefetch are
mutually exclusive.

### Side streams and embeddings cache

| Model Architecture | Multimodal Encoder Side Stream | Multimodal Embeddings Cache |
| ------------------ | ------------------------------ | --------------------------- |
| `Gemma4ForConditionalGeneration` | Yes | Yes |
| `Gemma4UnifiedForConditionalGeneration` | Yes | Yes |
| `Mistral3ForConditionalGeneration` | Yes | Yes |
| `Qwen3VLForConditionalGeneration` | Yes | Yes |
| `Qwen3VLMoeForConditionalGeneration` | Yes | Yes |
| `Qwen3_5ForConditionalGeneration` | Yes | Yes |
| `Qwen3_5MoeForConditionalGeneration` | Yes | Yes |

- **Multimodal encoder side stream** prefetches encoder work for pending requests on a separate
  CUDA stream, allowing it to overlap with work on the main stream. Set
  `multimodal_config.encoder_side_stream_max_ahead` to a positive value to enable it; the value
  limits the number of prefetched requests that can be ahead of admission. This option is mutually
  exclusive with `multimodal_config.encoder_cuda_graph` and can increase peak GPU memory use. It
  can be combined with the multimodal embeddings cache so side-stream cache hits skip encoder work
  and misses populate the cache.
- **Multimodal embeddings cache** is a per-model, cross-request LRU cache of encoder embeddings,
  stored per multimodal item. Set `multimodal_config.encoder_cache_max_bytes` to its capacity (for
  example, `"512MiB"`), or `0` to disable cross-request reuse. The capacity applies per rank; ranks
  do not share a cache.
  - On the inline encoder path, a request whose items all hit the cache skips the encoder, and a
    partially cached request encodes only its missing items. Inline caching supports only
    single-modality requests. When combined with side-stream prefetch, peak memory is the cache
    capacity plus any in-flight prefetched encoder inputs and outputs.
  - Models with item-level encoder scheduling use the item path unless
    `multimodal_config.encoder_scheduling_policy` is `DISABLED`. Cached items are reused before
    encoder work is selected, so they use none of the per-iteration encoder item or token budget.
    Only missing items are encoded, mixed-modality requests are cached per item, and newly encoded
    items populate the cache for later requests. With KV cache manager V2, items covered by confirmed
    KV reuse are not encoded. With chunked prefill, each chunk waits only for its required items
    and releases encoder outputs after use. KV cache manager V1 keeps whole-request encoder
    scheduling. The cache is never smaller than the output of one encoder iteration, even when
    `encoder_cache_max_bytes` is `0`. Item scheduling does not
    support side-stream prefetch or pipeline parallelism yet. With `pipeline_parallel_size > 1`,
    the `DEFAULT` policy, explicit or unset, logs a warning and falls back to the inline path, and
    the `EAGER` policy raises an error.

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
| `Lightricks/LTX-2.3` | Text-to-Video (with Audio), Video-to-Video (with Audio) |
| `MiniMaxAI/MiniMax-H3` | Text-to-Video, First/Last-Frame-to-Video, Reference-to-Video (all with Audio) |
| `Qwen/Qwen-Image` | Text-to-Image |
| `Qwen/Qwen-Image-2512` | Text-to-Image |
| `Qwen/Qwen-Image-Layered` | Image-to-Image |
| `Qwen/Qwen-Image-Edit-2511` | Image Editing (text+images-to-image) |
| `nvidia/Cosmos3-Nano` | Text-to-Image, Text-to-Video, Image-to-Video |
| `nvidia/Cosmos3-Super` | Text-to-Image, Text-to-Video, Image-to-Video |
| `nvidia/Cosmos3-Super-Text2Image-4Step` | Text-to-Image (DMD2-distilled, fixed 4-step schedule) |
| `nvidia/Cosmos3-Super-Image2Video-4Step` | Image-to-Video (DMD2-distilled, fixed 4-step schedule) |
| `nvidia/Cosmos3-Edge` | Text-to-Image, Text-to-Video, Image-to-Video (Nemotron-dense backbone, 480p-native) |

### Feature Matrix

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve | Attention2D | Ring Attention | Tensor Parallelism | Text Encoder TP |
|---|---|---|---|---|---|---|---|---|---|--|--|--|---|
| **FLUX.1** | Yes | Yes | Yes | No | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **FLUX.2** | Yes | Yes | Yes | No | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Wan 2.2** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **LTX-2** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| **LTX-2.3** | Yes | Yes | No | No | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| **MiniMax-H3** | Yes | Yes | No | No | Yes | Yes | No | Yes | Yes | No | No | Yes | Yes |
| **Qwen-Image** | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No | No |
| **Qwen-Image-Layered** | Yes | No | No | No | No | No | Yes | Yes | Yes | No | No | No | No |
| **Qwen-Image-Edit-2511** | Yes | Yes | No | Yes | Yes | No | Yes | Yes | Yes | No | No | No | No |
| **Cosmos3** | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No |
