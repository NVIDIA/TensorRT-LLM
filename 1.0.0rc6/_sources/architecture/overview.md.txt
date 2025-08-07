(architecture-overview)=

# TensorRT-LLM Architecture

TensorRT-LLM is a toolkit to assemble optimized solutions to perform Large Language Model (LLM) inference. It offers a Model Definition API to define models and compile efficient [TensorRT](https://developer.nvidia.com/tensorrt) engines for NVIDIA GPUs. It also contains Python and C++ components to build runtimes to execute those engines as well as backends for the [Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server) to easily create web-based services for LLMs. TensorRT-LLM supports multi-GPU and multi-node configurations (through MPI).

As a user, the very first step to create an inference solution is to either define your own model or select a pre-defined network architecture (refer to {ref}`models` for the list of models supported by TensorRT-LLM). Once defined, that model must be trained using a training framework (training is outside of the scope of TensorRT-LLM). For pre-defined models, checkpoints can be downloaded from various providers. To illustrate that point, a lot of examples in TensorRT-LLM use model weights obtained from the [Hugging Face](https://huggingface.co) hub and trained using [NVIDIA Nemo](https://developer.nvidia.com/nemo) or [PyTorch](https://pytorch.org).

Equipped with the model definition and the weights, a user must use TensorRT-LLM's Model Definition API to recreate the model in a way that can be compiled by TensorRT into an efficient engine. For ease of use, TensorRT-LLM already supports a handful of standard models.

Together with the Model Definition API to describe models, TensorRT-LLM provides users with components to create a runtime that executes the efficient TensorRT engine. Runtime components offer beam-search, along with extensive sampling functionalities such as top-K and top-P sampling. The exhaustive list can be found in the documentation of the {ref}`gpt-runtime`. The C++ runtime is the recommended runtime.

TensorRT-LLM also includes Python and C++ backends for NVIDIA Triton Inference Server to assemble solutions for LLM online serving. The C++ backend implements in-flight batching as explained in the {ref}`executor` documentation and is the recommended backend.

## Model Weights

TensorRT-LLM is a library for LLM inference, and so to use it, you need to supply a set of trained weights. You can either use your own model weights trained in a framework like [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/) or pull a set of pretrained weights from repositories like the Hugging Face Hub.
