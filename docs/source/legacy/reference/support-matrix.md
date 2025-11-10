# Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on NVIDIA GPUs. The following sections provide a list of supported GPU architectures as well as important features implemented in TensorRT-LLM.

## Models (PyTorch Backend)

| Architecture | Model | HuggingFace Example | Modality |
|--------------|-------|---------------------|----------|
| `BertForSequenceClassification` | BERT-based | `textattack/bert-base-uncased-yelp-polarity` | L |
| `DeciLMForCausalLM` | Nemotron | `nvidia/Llama-3_1-Nemotron-51B-Instruct` | L |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3 `| L |
| `Exaone4ForCausalLM` | EXAONE 4.0 | `LGAI-EXAONE/EXAONE-4.0-32B` | L |
| `Gemma3ForCausalLM` | Gemma 3 | `google/gemma-3-1b-it` | L |
| `Gemma3ForConditionalGeneration` | Gemma 3 | `google/gemma-3-27b-it` | L + I |
| `HCXVisionForCausalLM` | HyperCLOVAX-SEED-Vision | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | L + I |
| `LlavaLlamaModel` | VILA | `Efficient-Large-Model/NVILA-8B` | L + I + V |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | `llava-hf/llava-v1.6-mistral-7b-hf` | L + I |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA | `meta-llama/Meta-Llama-3.1-70B` | L |
| `Llama4ForConditionalGeneration` | Llama 4 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | L + I |
| `MistralForCausalLM` | Bielik | `speakleash/Bielik-11B-v2.2-Instruct` | L |
| `MistralForCausalLM` | Mistral | `mistralai/Mistral-7B-v0.1` | L |
| `Mistral3ForConditionalGeneration` | Mistral3 | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | L + I |
| `MixtralForCausalLM` | Mixtral | `mistralai/Mixtral-8x7B-v0.1` | L |
| `MllamaForConditionalGeneration` | Llama 3.2 | `meta-llama/Llama-3.2-11B-Vision` | L |
| `NemotronForCausalLM` | Nemotron-3, Nemotron-4, Minitron | `nvidia/Minitron-8B-Base` | L |
| `NemotronNASForCausalLM` | NemotronNAS | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | L |
| `Phi3ForCausalLM` | Phi-4  | `microsoft/Phi-4` | L |
| `Phi4MMForCausalLM` | Phi-4-multimodal | `microsoft/Phi-4-multimodal-instruct` | L + I + A |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/Qwen2-7B-Instruct` | L |
| `Qwen2ForProcessRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-PRM-7B` | L |
| `Qwen2ForRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-RM-72B` | L |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | L + I + V |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | L + I + V |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B` | L |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B` | L |

Note:
- L: Language
- I: Image
- V: Video
- A: Audio

## Models (TensorRT Backend)

### LLM Models

- [Arctic](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/arctic)
- [Baichuan/Baichuan2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/baichuan)
- [BART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [BERT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/bert)
- [BLOOM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/bloom)
- [ByT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [ChatGLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/chatglm-6b)
- [ChatGLM2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/chatglm2-6b)
- [ChatGLM3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/chatglm3-6b-32k)
- [Code LLaMA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama)
- [DBRX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/dbrx)
- [Exaone](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/exaone)
- [FairSeq NMT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [Falcon](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/falcon)
- [Flan-T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec) [^encdec]
- [Gemma/Gemma2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma)
- [GLM-4](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/glm-4-9b)
- [GPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gpt)
- [GPT-J](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/gptj)
- [GPT-Nemo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gpt)
- [GPT-NeoX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/gptneox)
- [Granite-3.0](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/granite)
- [Grok-1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/grok)
- [InternLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples//models/contrib/internlm)
- [InternLM2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/internlm2)
- [LLaMA/LLaMA 2/LLaMA 3/LLaMA 3.1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama)
- [Mamba](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/mamba)
- [mBART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [Minitron](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/nemotron)
- [Mistral](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama)
- [Mistral NeMo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama)
- [Mixtral](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/mixtral)
- [MPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/mpt)
- [Nemotron](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/nemotron)
- [mT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/opt)
- [Phi-1.5/Phi-2/Phi-3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/phi)
- [Qwen/Qwen1.5/Qwen2/Qwen3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/qwen)
- [Qwen-VL](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/qwenvl)
- [RecurrentGemma](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/recurrentgemma)
- [Replit Code](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/mpt) [^replitcode]
- [RoBERTa](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/bert)
- [SantaCoder](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gpt)
- [Skywork](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/skywork)
- [Smaug](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/contrib/smaug)
- [StarCoder](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gpt)
- [T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec)
- [Whisper](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/whisper)


### Multi-Modal Models [^multimod]

- [BLIP2 w/ OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [BLIP2 w/ T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [CogVLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal) [^bf16only]
- [Deplot](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [Fuyu](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [Kosmos](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [LLaVA-v1.5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [LLaVa-Next](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [LLaVa-OneVision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [Nougat](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [Phi-3-vision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [Video NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [VILA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [MLLaMA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)
- [LLama 3.2 VLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/multimodal)


(support-matrix-hardware)=
## Hardware

The following table shows the supported hardware for TensorRT-LLM.

If a GPU architecture is not listed, the TensorRT-LLM team does not develop or test the software on the architecture and support is limited to community support.
In addition, older architectures can have limitations for newer software releases.

```{list-table}
:header-rows: 1
:widths: 20 80

* -
  - Hardware Compatibility
* - Operating System
  - TensorRT-LLM requires Linux x86_64 or Linux aarch64.
* - GPU Model Architectures
  -
    - [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
    - [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
    - [NVIDIA Grace Hopper Superchip](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
    - [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
    - [NVIDIA Ada Lovelace Architecture](https://www.nvidia.com/en-us/technologies/ada-architecture/)
    - [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
```

(support-matrix-software)=
## Software

The following table shows the supported software for TensorRT-LLM.

```{list-table}
:header-rows: 1
:widths: 20 80

* -
  - Software Compatibility
* - Container
  - [25.08](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
* - TensorRT
  - [10.13](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
* - Precision
  -
    - Blackwell (SM100/SM120) - FP32, FP16, BF16, FP8, FP4, INT8, INT4
    - Hopper (SM90) - FP32, FP16, BF16, FP8, INT8, INT4
    - Ada Lovelace (SM89) - FP32, FP16, BF16, FP8, INT8, INT4
    - Ampere (SM80, SM86) - FP32, FP16, BF16, INT8, INT4[^smgte89]
```

[^replitcode]: Replit Code is not supported with the transformers 4.45+.

[^smgte89]: INT4 AWQ and GPTQ with FP8 activations require SM >= 89.

[^encdec]: Encoder-Decoder provides general encoder-decoder functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, and so on.

[^multimod]: Multi-modal provides general multi-modal functionality that supports many multi-modal architectures such as BLIP2 family, LLaVA family, and so on.

[^bf16only]: Only supports bfloat16 precision.


```{note}
Support for FP8 and quantized data types (INT8 or INT4) is not implemented for all the models. Refer to {ref}`precision` and [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder for additional information.
```
