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
- [Grok-1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/grok)
- [InternLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm)
- [InternLM2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm2)
- [LLaMA/LLaMA 2/LLaMA 3/LLaMA 3.1](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
- [Mamba](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mamba)
- [mBART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
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
- [Replit Code](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt)
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
- [NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Nougat](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Phi-3-vision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [Video NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
- [VILA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)


(support-matrix-hardware)=
## Hardware

The following table shows the supported hardware for TensorRT-LLM.

If a GPU is not listed, it is important to note that TensorRT-LLM is expected to work on GPUs based on the Volta, Turing, Ampere, Hopper, and Ada Lovelace architectures. Certain limitations may, however, apply.

```{list-table}
:header-rows: 1
:widths: 20 80

* -
  - Hardware Compatibility
* - Operating System
  - TensorRT-LLM requires Linux x86_64 or Windows.
* - GPU Model Architectures
  -
    - [NVIDIA Hopper H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
    - [NVIDIA L40S GPU](https://www.nvidia.com/en-us/data-center/l40s/)
    - [NVIDIA Ada Lovelace GPU](https://www.nvidia.com/en-us/technologies/ada-architecture/)
    - [NVIDIA Ampere A100 GPU](https://www.nvidia.com/en-us/data-center/a100/)
    - [NVIDIA A30 GPU](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
    - [NVIDIA Turing T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/)
    - [NVIDIA Volta V100 GPU](https://www.nvidia.com/en-us/data-center/v100/) (experimental)
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
  - [24.07](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
* - TensorRT
  - [10.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
* - Precision
  -
    - Hopper (SM90) - FP32, FP16, BF16, FP8, INT8, INT4
    - Ada Lovelace (SM89) - FP32, FP16, BF16, FP8, INT8, INT4
    - Ampere (SM80, SM86) - FP32, FP16, BF16, INT8, INT4[^smgte89]
    - Turing (SM75) - FP32, FP16, INT8[^smooth], INT4
    - Volta (SM70) - FP32, FP16, INT8[^smooth], INT4[^smlt75]
```

[^smooth]: INT8 SmoothQuant is not supported on SM70 and SM75.

[^smlt75]: INT4 AWQ and GPTQ are not supported on SM < 75.

[^smgte89]: INT4 AWQ and GPTQ with FP8 activations require SM >= 89.

[^encdec]: Encoder-Decoder provides general encoder-decoder functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, and so on.

[^multimod]: Multi-modal provides general multi-modal functionality that supports many multi-modal architectures such as BLIP2 family, LLaVA family, and so on.

[^bf16only]: Only supports bfloat16 precision.


```{note}
Support for FP8 and quantized data types (INT8 or INT4) is not implemented for all the models. Refer to {ref}`precision` and [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder for additional information.
```
