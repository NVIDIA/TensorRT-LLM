# Quick Start Recipe for Qwen3 Next on TensorRT LLM - Blackwell & Hopper Hardware

## Introduction

This is a functional quick-start guide for running the Qwen3-Next model on TensorRT LLM. It focuses on a working setup with recommended defaults. Additional performance optimizations and support will be rolled out in future updates.

## Prerequisites

* GPU: NVIDIA Blackwell or Hopper Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* BF16 model: [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)

## Deployment Steps

### Run Docker Container

Build and run the docker container. See the [Docker guide](../../../docker/README.md) for details.
```
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=qwen3-next-local

make -C docker release_run IMAGE_NAME=tensorrt_llm IMAGE_TAG=qwen3-next-local LOCAL_USER=1
```

### Creating the TensorRT LLM Server config

We create a YAML configuration file `/tmp/config.yml` for the TensorRT LLM Server with the following content:

```shell
EXTRA_LLM_API_FILE=/tmp/config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
enable_attention_dp: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 720
moe_config:
    backend: TRTLLM
stream_interval: 20
num_postprocess_workers: 4
kv_cache_config:
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.6
EOF
```


### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server with the Qwen3-Next model from within the container.

```shell
trtllm-serve Qwen/Qwen3-Next-80B-A3B-Thinking \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 16 \
    --max_num_tokens 4096 \
    --tp_size 4 \
    --pp_size 1 \
    --ep_size 4 \
    --trust_remote_code \
    --extra_llm_api_options ${EXTRA_LLM_API_FILE}
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

These options are used directly on the command line when you start the `trtllm-serve` process.

#### `--tp_size`

* **Description:** Sets the **tensor-parallel size**. This should typically match the number of GPUs you intend to use for a single model instance.

#### `--ep_size`

* **Description:** Sets the **expert-parallel size** for Mixture-of-Experts (MoE) models. Like `tp_size`, this should generally match the number of GPUs you're using. This setting has no effect on non-MoE models.

#### `--kv_cache_free_gpu_memory_fraction`

* **Description:** A value between `0.0` and `1.0` that specifies the fraction of free GPU memory to reserve for the KV cache after the model is loaded. Since memory usage can fluctuate, this buffer helps prevent out-of-memory (OOM) errors.
* **Recommendation:** If you experience OOM errors, try reducing this value to `0.7` or lower.


#### `--max_batch_size`

* **Description:** The maximum number of user requests that can be grouped into a single batch for processing. The actual max batch size that can be achieved depends on total sequence length (input + output).

#### `--max_num_tokens`

* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch.

#### `--max_seq_len`

* **Description:** The maximum possible sequence length for a single request, including both input and generated output tokens. We won't specifically set it. It will be inferred from model config.

#### `--trust_remote_code`

* **Description:** Allows TensorRT LLM to download models and tokenizers from Hugging Face. This flag is passed directly to the Hugging Face API.


#### Extra LLM API Options (YAML Configuration)

These options provide finer control over performance and are set within a YAML file passed to the `trtllm-serve` command via the `--extra_llm_api_options` argument.

#### `cuda_graph_config`

* **Description**: A section for configuring CUDA graphs to optimize performance.

* **Options**:

  * `enable_padding`: If `"true"`, input batches are padded to the nearest `cuda_graph_batch_size`. This can significantly improve performance.

    **Default**: `false`

  * `max_batch_size`: Sets the maximum batch size for which a CUDA graph will be created.

    **Default**: `0`

    **Recommendation**: Set this to the same value as the `--max_batch_size` command-line option.

#### `moe_config`

* **Description**: Configuration for Mixture-of-Experts (MoE) models.

* **Options**:

  * `backend`: The backend to use for MoE operations.
    **Default**: `CUTLASS`

See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full list of options which can be used in the `extra_llm_api_options`.

## Testing API Endpoint

### Basic Test

Start a new terminal on the host to test the TensorRT LLM server you just launched.

You can query the health/readiness of the server using:

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

After the TensorRT LLM server is set up and shows Application startup complete, you can send requests to the server.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 1024,
    "top_p": 1.0
}' -w "\n"
```

Here is an example response:

```
{"id":"chatcmpl-64ac201c77bf46a7a3a4eca7759b1fd8","object":"chat.completion","created":1759022940,"model":"Qwen/Qwen3-Next-80B-A3B-Thinking","choices":[{"index":0,"message":{"role":"assistant","content":"Okay, the user is asking \"Where is New York?\" Hmm, this seems straightforward but I need to be careful. New York could mean different things—maybe they're confused about the city versus the state. \n\nFirst thought: Are they a tourist planning a trip? Or maybe a student doing homework? Could even be someone国外 who's only heard \"New York\" in movies and isn't sure if it's a city or state. \n\nI should clarify both possibilities immediately. People often mix them up. Like, if someone says \"I'm going to New York\" they're probably talking about NYC, but technically New York State is bigger. \n\nLet me break it down: \n- New York City (NYC) is the famous one—Manhattan, skyscrapers, Times Square. \n- Then New York State (NY) is the whole state, which includes NYC but also upstate areas like Albany (the capital), Buffalo, and even the Adirondacks. \n\nWait, should I mention that NYC is in New York State? Yeah, that's crucial. Otherwise they might think it's two separate things. Also, where is the state located? Northeast US, borders other states like Pennsylvania, New Jersey... maybe name a few neighbors for context. \n\nOh! And the city vs state confusion is super common. Like, people say \"New York\" for the city but forget it's part of a larger state. Should I give examples? \n- \"If someone says 'hot dogs in New York' they mean NYC\" \n- But \"the New York State Police\" covers the whole state\" \n\n...Maybe add a fun fact? Like how NYC is just one city within the state? Or that the state capital isn't NYC but Albany? Yeah, that'll help clear confusion. \n\nAlso—should I ask if they meant the city or the state? Nah, better to cover both upfront. Keep it simple but precise. No jargon. They might not know terms like \"metropolitan area\" or \"boroughs,\" so explain \"boroughs\" as neighborhoods (Manhattan, Brooklyn etc.). \n\nCoordinates: 40.7°N, 74°W? Maybe overkill unless they're into geography. But for the state, roughly 42°N to 45°N latitude? Hmm, probably unnecessary. Stick to landmarks: borders Lake Erie, Atlantic coast... \n\nWait—user might be asking literally for GPS coordinates? Unlikely, but just in case I can add: \"Geographically, New York City is at approximately 40.7°N latitude and 74.0°W longitude...\" but only if the answer feels too dry otherwise. \n\nPriority: Clear, simple, correct. No assuming prior knowledge. Say \"New York can refer to...\" then split into two cases. End with \"Most people mean the city when they say 'New York' in casual talk.\" That’s the key takeaway. \n\n...Double-check: Is Albany really the capital? Yes. And NYC is the largest city but not capital. Got it. Also, New York State is the 27th largest by area? Maybe not relevant. Stick to basics. \n\nFinal structure: \n1. \"New York can mean two things: the city or the state.\" \n2. For city: \n   - Where it is (Northeast US) \n   - Inside New York State \n   - Famous parts (Manhattan etc.) \n3. For state: \n   - Larger area, includes city \n   - Borders (PA, NJ, CT, MA, VT, Canada) \n   - Capital is Albany (not NYC!) \n4. Clear summary: \"In everyday talk, 'New York' usually means NYC.\" \n\nYeah. That covers it without overwhelming them. No markdown, just plain English. Keep it warm—add a \"hope that helps!\" at the end.\n</think>\n\nThat's a common point of confusion! **\"New York\" can refer to two different things**, and it's important to distinguish between them:\n\n### 1. **New York City (NYC)**  \n   - **What it is**: The largest and most famous city in the United States, known for landmarks like Times Square, Central Park, the Statue of Liberty, and Wall Street.  \n   - **Where it is**:  \n     - Located in the **northeastern United States**.  \n     - Situated at the mouth of the **Hudson River**, where it meets the **Atlantic Ocean**.  \n     - Part of **New York State** (see below).  \n   - **Geographic details**:  \n     - Coordinates: Approximately **40.7° N latitude, 74.0° W longitude**.  \n     - Composed of **5 boroughs**: Manhattan (the \"city\" most people picture), Brooklyn, Queens, The Bronx, and Staten Island.  \n     - Panoramic view of NYC (including Brooklyn and New Jersey skyline):","reasoning_content":null,"reasoning":null,"tool_calls":[]},"logprobs":null,"finish_reason":"length","stop_reason":null,"mm_embedding_handle":null,"disaggregated_params":null,"avg_decoded_tokens_per_iter":1.0}],"usage":{"prompt_tokens":15,"total_tokens":1039,"completion_tokens":1024},"prompt_token_ids":null}
```

### Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size` or `max_seq_len`.
* Ensure your model checkpoints are compatible with the expected format.
* For performance issues, check GPU utilization with nvidia-smi while the server is running.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.



## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this first creating a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/qwen3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model Qwen/Qwen3-Next-80B-A3B-Thinking \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --tokenize-on-client \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
EOF
chmod +x bench.sh
```

To achieve max through-put, with attention DP on, one needs to sweep up to `concurrency = max_batch_size * num_gpus`.

If you want to save the results to a file add the following options.

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py) 

Run `bench.sh` to begin a serving benchmark. This will take a long time if you run all the concurrencies mentioned in the above `bench.sh` script.

```shell
./bench.sh
```
