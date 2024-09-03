(support-matrix)=

# Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on NVIDIA GPUs. The following sections provide a list of supported GPU architectures as well as important features implemented in TensorRT-LLM.

## Models
### LLMs


```{eval-rst}
.. list-table::
    :header-rows: 1
    :widths: 20 80

* - Model List
      -
            - `Arctic <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/arctic>`_
            - `Baichuan/Baichuan2 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/baichuan>`_
            - `BART <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `BERT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bert>`_
            - `BLOOM <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bloom>`_
            - `ByT5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `GLM/ChatGLM/ChatGLM2/ChatGLM3/GLM-4 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/chatglm>`_
            - `Code LLaMA <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama>`_
            - `DBRX <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/dbrx>`_
            - `Exaone <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/exaone>`_
            - `FairSeq NMT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `Falcon <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/falcon>`_
            - `Flan-T5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_ (4)
            - `Gemma/Gemma2 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gemma>`_
            - `GPT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt>`_
            - `GPT-J <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptj>`_
            - `GPT-Nemo <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt>`_
            - `GPT-NeoX <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gptneox>`_
            - `Grok-1 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/grok>`_
            - `InternLM <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm>`_
            - `InternLM2 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/internlm2>`_
            - `LLaMA/LLaMA 2/LLaMA 3/LLaMA 3.1 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama>`_
            - `Mamba <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mamba>`_
            - `mBART <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `Mistral <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama>`_
            - `Mistral NeMo <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama>`_
            - `Mixtral <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral>`_
            - `MPT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt>`_
            - `Nemotron <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/nemotron>`_
            - `mT5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `OPT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/opt>`_
            - `Phi-1.5/Phi-2/Phi-3 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi>`_
            - `Qwen/Qwen1.5/Qwen2 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen>`_
            - `Qwen-VL <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwenvl>`_
            - `RecurrentGemma <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/recurrentgemma>`_
            - `Replit Code <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mpt>`_
            - `RoBERTa <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/bert>`_
            - `SantaCoder <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt>`_
            - `Skywork <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/skywork>`_
            - `Smaug <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/smaug>`_
            - `StarCoder <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt>`_
            - `T5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec>`_
            - `Whisper <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper>`_
```


### Multi-Modal Models(5)

```{eval-rst}
.. list-table::
    :header-rows: 1
    :widths: 20 80

* - Model List
      -
            - `BLIP2 w/ OPT <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `BLIP2 w/ T5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `CogVLM <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_ (6)
            - `Deplot <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `Fuyu <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `Kosmos <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `LLaVA-v1.5 <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `LLaVa-Next <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `NeVA <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `Nougat <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `Phi-3-vision <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `Video NeVA <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
            - `VILA <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal>`_
```


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
      - `24.07 <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_
    * - TensorRT
      - `10.3 <https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html>`_
    * - Precision
      -
        - Hopper (SM90) - FP32, FP16, BF16, FP8, INT8, INT4
        - Ada Lovelace (SM89) - FP32, FP16, BF16, FP8, INT8, INT4
        - Ampere (SM80, SM86) - FP32, FP16, BF16, INT8, INT4(3)
        - Turing (SM75) - FP32, FP16, INT8(1), INT4
        - Volta (SM70) - FP32, FP16, INT8(1), INT4(2)

```

(1) INT8 SmoothQuant is not supported on SM70 and SM75.<br>
(2) INT4 AWQ and GPTQ are not supported on SM < 75.<br>
(3) INT4 AWQ and GPTQ with FP8 activations require SM >= 89.<br>
(4) Encoder-Decoder provides general encoder-decoder functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, and so on.
(5) Multi-modal provides general multi-modal functionality that supports many multi-modal architectures such as BLIP2 family, LLaVA family, and so on.
(6) Only supports bfloat16 precision.



```{note}
Support for FP8 and quantized data types (INT8 or INT4) is not implemented for all the models. Refer to {ref}`precision` and [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder for additional information.
```
