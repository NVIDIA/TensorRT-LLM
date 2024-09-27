(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT-LLM. Specifically, this Quick Start Guide enables you to quickly get setup and send HTTP requests using TensorRT-LLM.

## Prerequisites

- This quick start uses the Meta Llama 3.1 model. This model is subject to a particular [license](https://llama.meta.com/llama-downloads/). To download the model files, agree to the terms and [authenticate with Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct?clone=true).

- Complete the [installation](./installation/linux.md) steps.

- Pull the weights and tokenizer files for the chat-tuned variant of the Llama 3.1 8B model from the [Hugging Face Hub](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

  ```console
  git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
  ```

(quick-start-guide-compile)=
## Compile the Model into a TensorRT Engine

Use the [Llama model definition](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) from the `examples/llama` directory of the GitHub repository.
The model definition is a minimal example that shows some of the optimizations available in TensorRT-LLM.

```console
# From the root of the cloned repository, start the TensorRT-LLM container
make -C docker release_run LOCAL_USER=1

# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****

# Convert the model into TensorRT-LLM checkpoint format
cd examples/llama
pip install -r requirements.txt
pip install --upgrade transformers # Llama 3.1 requires transformer 4.43.0+ version.
python3 convert_checkpoint.py --model_dir Meta-Llama-3.1-8B-Instruct --output_dir llama-3.1-8b-ckpt

# Compile model
trtllm-build --checkpoint_dir llama-3.1-8b-ckpt \
    --gemm_plugin float16 \
    --output_dir ./llama-3.1-8b-engine
```

When you create a model definition with the TensorRT-LLM API, you build a graph of operations from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) primitives that form the layers of your neural network. These operations map to specific kernels; prewritten programs for the GPU.

In this example, we included the `gpt_attention` plugin, which implements a FlashAttention-like fused attention kernel, and the `gemm` plugin, that performs matrix multiplication with FP32 accumulation. We also called out the desired precision for the full model as FP16, matching the default precision of the weights that you downloaded from Hugging Face. For more information about plugins and quantizations, refer to the [Llama example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) and {ref}`precision` section.

## Run the Model

Now that you have the model engine, run the engine and perform inference.

```console
python3 ../run.py --engine_dir ./llama-3.1-8b-engine  --max_output_len 100 --tokenizer_dir Meta-Llama-3.1-8B-Instruct --input_text "How do I count to nine in French?"
```

## Deploy with Triton Inference Server

To create a production-ready deployment of your LLM, use the [Triton Inference Server backend for TensorRT-LLM](https://github.com/triton-inference-server/tensorrtllm_backend) to leverage the TensorRT-LLM C++ runtime for rapid inference execution and include optimizations like in-flight batching and paged KV caching. Triton Inference Server with the TensorRT-LLM backend is available as a [pre-built container through NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

1. Pull down the example model repository so that Triton Inference Server can read the model and any associated metadata.

    ```bash
    # After exiting the TensorRT-LLM Docker container
    cd ..
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
    cd tensorrtllm_backend
    cp ../TensorRT-LLM/examples/llama/out/*   all_models/inflight_batcher_llm/tensorrt_llm/1/
    ```

    The `tensorrtllm_backend` repository includes the skeleton of a model repository under `all_models/inflight_batcher_llm/` that you can use.

2. Copy the model you compiled ({ref}`quick-start-guide-compile`) to the example model repository.

3. Modify the configuration files from the model repository. Specify the path to the compiled model engine, the tokenizer, and how to handle memory allocation for the KV cache when performing inference in batches.

    ```bash
    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
        decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,\
    max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,\
    max_num_sequences:4

    python tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
        tokenizer_type:llama,tokenizer_dir:Meta-Llama-3.1-8B-Instruct

    python tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
        tokenizer_type:llama,tokenizer_dir:Meta-Llama-3.1-8B-Instruct
    ```

4. Start Triton Inference Server in the container. Specify `world_size`, which is the number of GPUs the model was built for, and point to the `model_repo` that was just set up.

    ```bash
    docker run -it --rm --gpus all --network host --shm-size=1g \
    -v $(pwd)/all_models:/all_models \
    -v $(pwd)/scripts:/opt/scripts \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3

    # Log in to huggingface-cli to get tokenizer
    huggingface-cli login --token *****

    # Install python dependencies
    pip install sentencepiece protobuf

    # Launch Server
    python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
    ```

## Send Requests

Use one of the Triton Inference Server client libraries or send HTTP requests to the generated endpoint. To get started, you can use the more fully featured client script or the following command:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
"text_input": "How do I count to nine in French?",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""]
}
}'
```

## LLM API
The LLM API is a Python API to setup & infer with TensorRT-LLM directly in python.It allows for optimizing models by specifying a HuggingFace repo name or a model checkpoint. The LLM API handles checkpoint conversion, engine building, engine loading, and model inference, all from one python object.

Note that these APIs are in incubation, they may change and  supports the [following models](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/index.html#supported-model), which will increase in coming release. We appreciate your patience and understanding as we improve this API.

Here is a simple example to show how to use the LLM API with TinyLlama.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

To learn more about the LLM API, check out the [](llm-api-examples/index) and [](llm-api/index).

## Next Steps

In this Quick Start Guide, you:

- Installed and built TensorRT-LLM
- Retrieved the model weights
- Compiled and ran the model
- Deployed the model with Triton Inference Server
- Sent HTTP requests

For more examples, refer to:

- [examples/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) for showcases of how to run a quick benchmark on latest LLMs.

## Related Information

- [Best Practices Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md)
- [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
