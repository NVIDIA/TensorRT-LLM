# How to launch Llama4 Maverick + Eagle3 TensorRT LLM server

Artificial Analysis has benchmarked the Llama4 Maverick with Eagle3 enabled TensorRT LLM server running at over [1000 tokens per second per user on 8xB200 GPUs](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/). This implementation leverages NVIDIA's TensorRT LLM combined with speculative decoding using the Eagle3 model to further boost performance.

In the guide below, we will walk you through how to launch your own high-performance Llama4 Maverick with Eagle3 enabled TensorRT LLM server, from build to deployment.  (Note that your specific performance numbers may vary—speculative decoding speedups depend upon the dataset!)

## Prerequisites

- 8x NVIDIA B200 GPUs in a single node (we have a forthcoming guide for getting great performance on H100)
- CUDA Toolkit 12.8 or later
- Docker with NVIDIA Container Toolkit installed
- Fast SSD storage for model weights
- Access to Llama4 Maverick and Eagle3 model checkpoints
- A love of speed

## Download Artifacts

* [NVIDIA Llama 4 Maverick 17B 128E Instruct FP8](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8)
* [NVIDIA Llama 4 Maverick 17B 128E Eagle3 BF16](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3)

In [Step 4: Start the TensorRT LLM server](#step-4-start-the-tensorrt-llm-server), `/path/to/maverick` and `/path/to/eagle` refer to the download paths of the above respective models.

## Launching the server

### Step 1: Clone the repository

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

The last command, `git lfs pull`, ensures all large files stored with Git LFS are properly downloaded. If `git lfs` is not installed, please install following [Install Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

### Step 2: Prepare the TensorRT LLM release Docker image


#### Option 1. Use weekly release NGC docker image
TensorRT LLM provides weekly release [docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release)

#### Option 2. Build TensorRT LLM Docker image (Alternative way)
If you want to compile a specific TensorRT LLM commit, you can build the docker image by checking out the specific branch or commit and running a make command. This may take 15-30 minutes depending on your system.

```
make -C docker release_build
```

### Step 3: (Optional) Tag and push the Docker image to your registry

If you want to use this image on multiple machines or in a cluster:

```
docker tag tensorrt_llm/release:latest docker.io/<username>/tensorrt_llm:main
docker push docker.io/<username>/tensorrt_llm:main
```

Replace `<username>` with your Docker Hub username or your private registry path.

### Step 4: Start the TensorRT LLM server

This command launches the server with Llama4 Maverick as the main model and Eagle3 as the draft model for speculative decoding. Make sure you have downloaded both model checkpoints before running this command.

**Important:** Replace `/path/to/maverick` and `/path/to/eagle` with the actual paths to your Maverick and Eagle3 model checkpoints on your host machine, downloaded in the [Download Artifacts](#download-artifacts) stage

```
docker run -d --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 --gpus=all -e "TRTLLM_ENABLE_PDL=1" \
    -v /path/to/maverick:/config/models/maverick -v /path/to/eagle:/config/models/eagle \
    docker.io/<username>/tensorrt_llm:main sh \
        -c "echo -e 'enable_autotuner: false\nenable_attention_dp: false\nenable_min_latency: true\ncuda_graph_config:\n  max_batch_size: 8\nspeculative_config:\n  decoding_type: Eagle\n  max_draft_len: 3\n  speculative_model_dir: /config/models/eagle\n  eagle3_one_model: true\nkv_cache_config:\n  enable_block_reuse: false' > c.yaml && \
        TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=True \
        trtllm-serve /config/models/maverick \
            --host 0.0.0.0 --port 8000 \
            --tp_size 8 --ep_size 1 \
            --trust_remote_code --extra_llm_api_options c.yaml \
            --kv_cache_free_gpu_memory_fraction 0.75"
```

This command:
- Runs the container in detached mode (`-d`)
- Sets up shared memory and stack limits for optimal performance
- Maps port 8000 from the container to your host
- Enables all GPUs with tensor parallelism across all 8 GPUs
- Creates a configuration file for speculative decoding with Eagle3
- Configures memory settings for optimal throughput

After running this command, the server will initialize, which may take several minutes as it loads and optimizes the models.

You can query the health/readiness of the server using
```
curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health"
```

When the 200 code is returned the server is ready for queries.  Note that the very first query may take longer due to initialization and compilation.

### Step 5: Test the server with a sample request

Once the server is running, you can test it with a simple curl request:

```
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "Llama4-eagle",
        "messages": [{"role": "user", "content": "Why is NVIDIA a great company?"}],
        "max_tokens": 1024
    }' -w "\n"

# {"id":"chatcmpl-e752184d1181494c940579c007ab2c5f","object":"chat.completion","created":1748018634,"model":"Llama4-eagle","choices":[{"index":0,"message":{"role":"assistant","content":"NVIDIA is considered a great company for several reasons:\n\n1. **Innovative Technology**: NVIDIA is a leader in the development of graphics processing units (GPUs) and high-performance computing hardware. Their GPUs are used in a wide range of applications, from gaming and professional visualization to artificial intelligence (AI), deep learning, and autonomous vehicles.\n2. ...","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":17,"total_tokens":552,"completion_tokens":535}}
```

The server exposes a standard OpenAI-compatible API endpoint that accepts JSON requests. You can adjust parameters like `max_tokens`, `temperature`, and others according to your needs.


### Step 6: (Optional) Monitor server logs

To view the logs of the running container:

```
docker ps # get the container id
docker logs -f <container_id>
```

This is useful for troubleshooting or monitoring performance statistics reported by the server.

### Step 7: (Optional) Stop the server

When you're done with the server:

```
docker ps # get the container id
docker kill <container_id>
```

## Troubleshooting Tips

- If you encounter CUDA out-of-memory errors, try reducing `max_batch_size` or `max_seq_len`
- Ensure your model checkpoints are compatible with the expected format
- For performance issues, check GPU utilization with `nvidia-smi` while the server is running
- If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed
- For connection issues, make sure port 8000 is not being used by another application

## Performance Tuning

The configuration provided is optimized for 8xB200 GPUs, but you can adjust several parameters for your specific workload.

**Note:** This configuration is optimized for minimum latency (`enable_min_latency: true`). When increasing the concurrency of requests, the tokens per second (TPS) per user degrades rapidly. This setup is designed to maximize single-user performance rather than high-concurrency throughput. For workloads with many concurrent users, you may need to adjust the configuration accordingly.

- `max_batch_size`: Controls how many requests can be batched together
- `max_draft_len`: The number of tokens Eagle can speculate ahead
- `kv_cache_free_gpu_memory_fraction`: Controls memory allocation for the KV cache
