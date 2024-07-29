(quick-start-guide)=

# Quick Start Guide

This is the starting point to try out TensorRT-LLM. Specifically, this Quick Start Guide enables you to quickly get setup and send HTTP requests using TensorRT-LLM.

## Prerequisites

The steps below use the Llama 2 model, which is subject to a particular [license](https://llama.meta.com/llama-downloads/). To download the necessary model files, agree to the terms and [authenticate with Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf?clone=true).

## Launch the Docker

Please be sure to complete the [installation](./installation/linux.md) steps before proceeding with the following steps.

## Retrieve the Model Weights

Pull the weights and tokenizer files for the chat-tuned variant of the 7B parameter Llama 2 model from the [Hugging Face Hub](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

```bash
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

(quick-start-guide-compile)=
## Compile the Model into a TensorRT Engine

Use the included [Llama model definition](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama). This is a minimal example that includes some of the optimizations available in TensorRT-LLM.

```bash
# Launch the Tensorrt-LLM container
make -C docker release_run LOCAL_USER=1

# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****

# Convert the model into TensorrtLLM checkpoint format
cd exammples/llama
python3 convert_checkpoint.py --model_dir meta-llama/Llama-2-7b-chat-hf --output_dir llama-2-7b-ckpt

# Compile model
trtllm-build --checkpoint_dir llama-2-7b-ckpt \
    --gemm_plugin float16 \
    --output_dir ./llama-2-7b-engine
```

When you created the model definition with the TensorRT-LLM API, you built a graph of operations from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) primitives that formed the layers of your neural network. These operations map to specific kernels; prewritten programs for the GPU.

In this example, we included the `gpt_attention` plugin, which implements a FlashAttention-like fused attention kernel, and the `gemm` plugin, that performs matrix multiplication with FP32 accumulation. We also called out the desired precision for the full model as FP16, matching the default precision of the weights that you downloaded from Hugging Face. For more information about plugins and quantizations, refer to the [Llama example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) and {ref}`precision` section.

## Run the Model

Now that you’ve got your model engine, its time to run it.

```bash
python3 ../run.py --engine_dir ./llama-2-7b-engine  --max_output_len 100 --tokenizer_dir meta-llama/Llama-2-7b-chat-hf --input_text "How do I count to nine in French?"
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

3. Modify the configuration files from the model repository. Specify, where the compiled model engine is, what tokenizer to use, and how to handle memory allocation for the KV cache when performing inference in batches.

    ```bash
    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
        decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,\
    max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,\
    max_num_sequences:4

    python tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
        tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-chat-hf

    python tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
        tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-chat-hf
    ```

4. Start the Docker container and launch the Triton Inference server. Specify `world size`, which is the number of GPUs the model was built for, and point to the `model_repo` that was just set up.

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

## Next Steps

In this Quick Start Guide, you:

- Installed and built TensorRT-LLM
- Retrieved the model weights
- Compiled and ran the model
- Deployed the model with Triton Inference Server
- Sent HTTP requests

For more examples, refer to:

- [examples/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) for showcases of how to run a quick benchmark on latest LLMs.

## Links
 - [Best Practices Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md)
 - [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
