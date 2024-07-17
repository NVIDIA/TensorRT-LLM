(support-matrix)=

# Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on NVIDIA GPUs. The following sections provide a list of supported GPU architectures as well as important features implemented in TensorRT-LLM.

(support-matrix-hardware)=
## Hardware

The following table shows the supported hardware for TensorRT-LLM.

If a GPU is not listed, it is important to note that TensorRT-LLM is expected to work on GPUs based on the Volta, Turing, Ampere, Hopper, and Ada Lovelace architectures. Certain limitations may, however, apply.

```{eval-rst}
.. list-table::
    :header-rows: 1
    :widths: 20 80

    * -
      - Hardware Compatibility
    * - Operating System
      - TensorRT-LLM requires Linux x86_64 or Windows.
    * - GPU Model Architectures
      -
            - `NVIDIA Hopper H100 GPU <https://www.nvidia.com/en-us/data-center/h100/>`_
            - `NVIDIA L40S GPU <https://www.nvidia.com/en-us/data-center/l40s/>`_
            - `NVIDIA Ada Lovelace GPU <https://www.nvidia.com/en-us/technologies/ada-architecture/>`_
            - `NVIDIA Ampere A100 GPU <https://www.nvidia.com/en-us/data-center/a100/>`_
            - `NVIDIA A30 GPU <https://www.nvidia.com/en-us/data-center/products/a30-gpu/>`_
            - `NVIDIA Turing T4 GPU <https://www.nvidia.com/en-us/data-center/tesla-t4/>`_
            - `NVIDIA Volta V100 GPU <https://www.nvidia.com/en-us/data-center/v100/>`_ (experimental)
```

(support-matrix-software)=
## Software

The following table shows the supported software for TensorRT-LLM.

```{eval-rst}
.. list-table::
    :header-rows: 1
    :widths: 20 80

    * -
      - Software Compatibility
    * - Container
      - [24.05](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    * - TensorRT
      - [10.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    * - Precision
      -
        - Hopper (SM90) - FP32, FP16, BF16, FP8, INT8, INT4
        - Ada Lovelace (SM89) - FP32, FP16, BF16, FP8, INT8, INT4
        - Ampere (SM80, SM86) - FP32, FP16, BF16, INT8, INT4(3)
        - Turing (SM75) - FP32, FP16, INT8(1), INT4
        - Volta (SM70) - FP32, FP16, INT8(1), INT4(2)
    * - Models
      -
        - [Arctic](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/arctic)
        - [Baichuan](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/baichuan)
        - [BART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
        - [BERT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bert)
        - [BLOOM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bloom)
        - [ByT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
        - [ChatGLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/chatglm)
        - [DBRX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/dbrx)
        - [FairSeq NMT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
        - [Falcon](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/falcon)
        - [Flan-T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec) (4)
        - [Gemma](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gemma)
        - [GPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
        - [GPT-J](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptj)
        - [GPT-Nemo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)
        - [GPT-NeoX](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptneox)
        - [InternLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm)
        - [InternLM2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm2)
        - [LLaMA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
        - [LLaMA-v2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
        - [Mamba](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mamba)
        - [mBART](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
        - [Mistral](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral)
        - [MPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt)
        - [mT5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec)
        - [OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/opt)
        - [Phi-1.5/Phi-2/Phi-3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi)
        - [Qwen/Qwen1.5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen)
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
    * - Multi-Modal Models (5)
      -
        - [BLIP2 w/ OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [BLIP2 w/ T5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [CogVLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)(6)
        - [Deplot](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [Fuyu](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [Kosmos](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [LLaVA-v1.5](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [Nougat](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [Phi-3-vision](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [Video NeVA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
        - [VILA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)
```

(1) INT8 SmoothQuant is not supported on SM70 and SM75.<br>
(2) INT4 AWQ and GPTQ are not supported on SM < 75.<br>
(3) INT4 AWQ and GPTQ with FP8 activations require SM >= 89.<br>
(4) [Encoder-Decoder](https://github.com/NVIDIA/TensorRT-LLM/tree/main/main/examples/enc_dec) provides general encoder-decoder functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, and so on.
(5) Multi-modal provides general multi-modal functionality that supports many multi-modal architectures such as BLIP2 family, LLaVA family, and so on.
(6) Only supports bfloat16 precision.


```{note}
Support for FP8 and quantized data types (INT8 or INT4) is not implemented for all the models. Refer to {ref}`precision` and [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder for additional information.
```
