# LLM API Introduction

The LLM API is a high-level Python API designed to streamline LLM inference workflows.

It supports a broad range of use cases, from single-GPU setups to multi-GPU and multi-node deployments, with built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) and the [Triton Inference Server](https://github.com/triton-inference-server/server).

While the LLM API simplifies inference workflows with a high-level interface, it is also designed with flexibility in mind. Under the hood, it uses a PyTorch-native and modular backend, making it easy to customize, extend, or experiment with the runtime.

## Table of Contents
- [Quick Start Example](#quick-start-example)
- [Supported Models](#supported-models)
- [Tips and Troubleshooting](#tips-and-troubleshooting)

## Quick Start Example
A simple inference example with TinyLlama using the LLM API:

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

For more advanced usage including distributed inference, multimodal, and speculative decoding, please refer to this [README](../../../examples/llm-api/README.md).

## Model Input

The `LLM()` constructor accepts either a Hugging Face model ID or a local model path as input.

### 1. Using a Model from the Hugging Face Hub

To load a model directly from the [Hugging Face Model Hub]((https://huggingface.co/)), simply pass its model ID (i.e., repository name) to the LLM constructor. The model will be automatically downloaded:

```python
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

You can also use [quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) (FP4, FP8, etc) of popular models provided by NVIDIA in the same way.

### 2. Using a Local Hugging Face Model

To use a model from local storage, first download it manually:

```console
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
```

Then, load the model by specifying a local directory path:

```python
llm = LLM(model=<local_path_to_model>)
```

> **Note:** Some models require accepting specific [license agreements]((https://ai.meta.com/resources/models-and-libraries/llama-downloads/)). Make sure you have agreed to the terms and authenticated with Hugging Face before downloading.


## Supported Models


| Models                                                    |                [Model Class Name](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/models)                | HuggingFace Model ID Example                                                                                                          | Modality |
| :-------------------------------------------------------------- | :----------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------ | :------: |
| BERT-based                                                      |    `BertForSequenceClassification`   | `textattack/bert-base-uncased-yelp-polarity`                                                                                          |     L    |
| DeepSeek-V3                                                     |        `DeepseekV3ForCausalLM`       | `deepseek-ai/DeepSeek-V3 `                                                                                                            |     L    |
| Gemma3                                                          |          `Gemma3ForCausalLM`         | `google/gemma-3-1b-it`                                                                                                                |     L    |
| HyperCLOVAX-SEED-Vision                                         |        `HCXVisionForCausalLM`        | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B`                                                                               |   L + V  |
| VILA                                                            |           `LlavaLlamaModel`          | `Efficient-Large-Model/NVILA-8B`                                                                                                      |   L + V  |
| LLaVA-NeXT                                                      |  `LlavaNextForConditionalGeneration` | `llava-hf/llava-v1.6-mistral-7b-hf`                                                                                                   |   L + V  |
| Llama 3 <br> Llama 3.1 <br> Llama 2 <br> LLaMA                  |          `LlamaForCausalLM`          | `meta-llama/Meta-Llama-3.1-70B`                                                                                                       |     L    |
| Llama 4 Scout <br> Llama 4 Maverick                             |   `Llama4ForConditionalGeneration`   | `meta-llama/Llama-4-Scout-17B-16E-Instruct` <br> `meta-llama/Llama-4-Maverick-17B-128E-Instruct`                                      |   L + V  |
| Mistral                                                         |         `MistralForCausalLM`         | `mistralai/Mistral-7B-v0.1`                                                                                                           |     L    |
| Mixtral                                                         |         `MixtralForCausalLM`         | `mistralai/Mixtral-8x7B-v0.1`                                                                                                         |     L    |
| Llama 3.2                                                       |   `MllamaForConditionalGeneration`   | `meta-llama/Llama-3.2-11B-Vision`                                                                                                     |     L    |
| Nemotron-3 <br> Nemotron-4 <br> Minitron                        |         `NemotronForCausalLM`        | `nvidia/Minitron-8B-Base`                                                                                                             |     L    |
| Nemotron-H                                                      |        `NemotronHForCausalLM`        | `nvidia/Nemotron-H-8B-Base-8K` <br> `nvidia/Nemotron-H-47B-Base-8K` <br> `nvidia/Nemotron-H-56B-Base-8K`                              |     L    |
| LLamaNemotron <br> LlamaNemotron Super <br> LlamaNemotron Ultra |       `NemotronNASForCausalLM`       | `nvidia/Llama-3_1-Nemotron-51B-Instruct` <br> `nvidia/Llama-3_3-Nemotron-Super-49B-v1` <br> `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` |     L    |
| QwQ, Qwen2                                                      |          `Qwen2ForCausalLM`          | `Qwen/Qwen2-7B-Instruct`                                                                                                              |     L    |
| Qwen2-based                                                     |     `Qwen2ForProcessRewardModel`     | `Qwen/Qwen2.5-Math-PRM-7B`                                                                                                            |     L    |
| Qwen2-based                                                     |         `Qwen2ForRewardModel`        | `Qwen/Qwen2.5-Math-RM-72B`                                                                                                            |     L    |
| Qwen2-VL                                                        |   `Qwen2VLForConditionalGeneration`  | `Qwen/Qwen2-VL-7B-Instruct`                                                                                                           |   L + V  |
| Qwen2.5-VL                                                      | `Qwen2_5_VLForConditionalGeneration` | `Qwen/Qwen2.5-VL-7B-Instruct`                                                                                                         |   L + V  |


- **L**: Language model only
- **L + V**: Language and Vision multimodal support
- Llama 3.2 accepts vision input, but our support currently limited to text only.

> **Note:** For the most up-to-date list of supported models, you may refer to the [TensorRT-LLM model definitions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/models).


## Tips and Troubleshooting

The following tips typically assist new LLM API users who are familiar with other APIs that are part of TensorRT-LLM:

- RuntimeError: only rank 0 can start multi-node session, got 1

  There is no need to add an `mpirun` prefix for launching single node multi-GPU inference with the LLM API.

  For example, you can run `python llm_inference_distributed.py` to perform multi-GPU on a single node.

- Hang issue on Slurm Node

  If you experience a hang or other issue on a node managed with Slurm, add prefix `mpirun -n 1 --oversubscribe --allow-run-as-root` to your launch script.

  For example, try `mpirun -n 1 --oversubscribe --allow-run-as-root python llm_inference_distributed.py`.

- MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode 1.

  Because the LLM API relies on the `mpi4py` library, put the LLM class in a function and protect the main entrypoint to the program under the `__main__` namespace to avoid a [recursive spawn](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor) process in `mpi4py`.

  This limitation is applicable for multi-GPU inference only.

- Cannot quit after generation

  The LLM instance manages threads and processes, which may prevent its reference count from reaching zero. To address this issue, there are two common solutions:
  1. Wrap the LLM instance in a function, as demonstrated in the quickstart guide. This will reduce the reference count and trigger the shutdown process.
  2. Use LLM as an contextmanager, with the following code: `with LLM(...) as llm: ...`, the shutdown methed will be invoked automatically once it goes out of the `with`-statement block.
