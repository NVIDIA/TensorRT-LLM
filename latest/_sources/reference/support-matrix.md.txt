(support-matrix)=

# Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on NVIDIA GPUs. The following sections provide a list of supported GPU architectures as well as important features implemented in TensorRT-LLM.

## Models

### LLM Models

- [Arctic](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/arctic)
- [Baichuan/Baichuan2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/baichuan)
- [BART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [BERT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bert)
- [BLOOM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bloom)
- [ByT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [GLM/ChatGLM/ChatGLM2/ChatGLM3/GLM-4](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/chatglm)
- [Code LLaMA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
- [DBRX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/dbrx)
- [Exaone](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/exaone)
- [FairSeq NMT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [Falcon](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/falcon)
- [Flan-T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec) [^encdec]
- [Gemma/Gemma2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gemma)
- [GPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
- [GPT-J](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptj)
- [GPT-Nemo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
- [GPT-NeoX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptneox)
- [Granite-3.0](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/granite)
- [Grok-1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/grok)
- [InternLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm)
- [InternLM2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm2)
- [LLaMA/LLaMA 2/LLaMA 3/LLaMA 3.1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
- [Mamba](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mamba)
- [mBART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [Minitron](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/nemotron)
- [Mistral](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
- [Mistral NeMo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
- [Mixtral](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral)
- [MPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt)
- [Nemotron](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/nemotron)
- [mT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/opt)
- [Phi-1.5/Phi-2/Phi-3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi)
- [Qwen/Qwen1.5/Qwen2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen)
- [Qwen-VL](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwenvl)
- [RecurrentGemma](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/recurrentgemma)
- [Replit Code](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt) [^replitcode]
- [RoBERTa](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bert)
- [SantaCoder](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
- [Skywork](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/skywork)
- [Smaug](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/smaug)
- [StarCoder](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
- [T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
- [Whisper](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper)


### Multi-Modal Models [^multimod]

- [BLIP2 w/ OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [BLIP2 w/ T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [CogVLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal) [^bf16only]
- [Deplot](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Fuyu](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Kosmos](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [LLaVA-v1.5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [LLaVa-Next](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [LLaVa-OneVision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Nougat](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Phi-3-vision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Video NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [VILA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [MLLaMA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [LLama 3.2 VLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)


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
  - [25.03](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
* - TensorRT
  - [10.9](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
* - Precision
  -
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
