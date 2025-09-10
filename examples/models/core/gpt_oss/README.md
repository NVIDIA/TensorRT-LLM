# GPT-OSS

## Overview

GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the other weights are in bf16.

## MoE Support Matrix

In MoE, the weights are pre-quantized to mxfp4. The activation can be in either bf16 (Hopper) or mxfp8 (Blackwell), with similar accuracy. FP8 activation with per-tensor scaling factor has limited support. Note that the per-tensor scaling factor needs to be calculated dynamically during inference with the official mxfp4 checkpoints, which may negatively impact perf. The configs in **bold** are the recommended configs for the official checkpoints.

| device | Activation | Weight | Supported moe_backend | MMA|
|----------|----------|----------|----------|----------|
| Hopper | **bf16** | mxfp4 | **TRITON**, CUTLASS | simulated mxfp4, HGMMA |
| Hopper | fp8 | mxfp4 | CUTLASS (not enabled) | simulated mxfp4, QGMMA |
| Blackwell | **mxfp8** | mxfp4 | **CUTLASS, TRTLLM** | UTCQMMA |
| Blackwell | fp8 | mxfp4 | CUTLASS, TRTLLM | UTCQMMA |
| Blackwell | fp8 | mxfp4 | TRITON (experimental) | NA |
| Blackwell | bf16 | mxfp4 | TRTLLM | simulated mxfp4, UTCHMMA |


| moe_backend | TP | EP | AlltoAll |
|----------|----------|----------|----------|
| CUTLASS | yes | yes | yes |
| TRTLLM | yes | yes | no |
| TRITON | no | yes | no |

For best performance, use the `TRITON` moe_backend on Hopper for both latency and throughput cases. Use `CUTLASS` for throughput cases and `TRTLLM` for latency cases on Blackwell.

## KV Cache Support Matrix

|   device  | bf16 kv cache dtype | fp8 kv cache dtype |
|:---------:|:-------------------:|--------------------|
|   Hopper  | yes                 | no                 |
| Blackwell | yes                 | yes                |

On Blackwell GPUs, support for FP8 KV cache is available, allowing the key-value cache to be stored in FP8 format. This reduces memory usage and bandwidth requirements, which can lead to higher throughput and improved overall performance, especially for large batch sizes or long sequence generation.

## Harmony Examples

### Function Calling

OpenAI MoE models support function calling. Here is an example based on [XGrammar](https://github.com/mlc-ai/xgrammar)'s structural tag.

First, launch a server with XGrammar enabled:

```bash
trtllm-serve <model>
```

Run the [openai_chat_client_function_calling.py](./openai_chat_client_function_calling.py) script, which queries the LLM server in two steps:

1. **First step:**
   - The client provides function definitions and a user prompt to the LLM server
   - Instead of answering the prompt directly, the LLM server responds with a selected function and corresponding arguments based on the user prompt
   - XGrammar's structural tag ensures the arguments conform to the function definition

2. **Second step:**
   - The client calls the selected function with the arguments and retrieves the results
   - The client provides the chat history and function call results to the LLM server
   - The LLM server provides the response based on the function call results

For example, you can query "What is the weather like in SF?" with the following command:

```bash
python openai_chat_client_function_calling.py \
    --model <model> \
    --prompt "What is the weather like in SF?"
```

The output would look similar to:

```txt
[USER PROMPT] What is the weather like in SF?
[RESPONSE 1] [COT] Need to call get_current_weather.
[RESPONSE 1] [FUNCTION CALL] get_current_weather(**{'location': 'San Francisco, CA'})
[RESPONSE 2] It’s a bright, sunny day in San Francisco with the temperature around 20 °C (68 °F). Enjoy the pleasant weather!
```

The function call works successfully:
- In `[RESPONSE 1]`, the LLM selects the correct function `get_current_weather` and provides the appropriate arguments.
- In `[FUNCTION CALL]`, the client parses the LLM response and executes the function call.
- In `[RESPONSE 2]`, the LLM integrates the function call results into its final answer.

Let's try another query "What is the weather like in NY and SF?" with the following command:

```bash
python openai_chat_client_function_calling.py \
    --model <model> \
    --prompt "What is the weather like in NY and SF?"
```

The output would look like:

```txt
[USER PROMPT] What is the weather like in NY and SF?
[RESPONSE 1] [COT] Need to call get_multiple_weathers.
[RESPONSE 1] [FUNCTION CALL] get_multiple_weathers(**{'locations': ['New York, NY', 'San Francisco, CA'], 'format': 'celsius'})
[RESPONSE 2] Here’s a quick snapshot of the current weather in both cities:

| City | Weather | Temperature |
|------|---------|-------------|
| New York | ☀️ Sunny | 20 °C |
| San Francisco | ☀️ Sunny | 20 °C |
```

Once again, the function call works successfully, this time using a different function: `get_multiple_weathers`.

## Using OpenAI Triton Kernels for MoE

OpenAI ships a set of Triton kernels optimized for its MoE models. TensorRT-LLM can leverage these kernels; enable them with the steps below:

1. **Build and install Triton** (tested with the commit below):

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
# Specific commit verified with TensorRT-LLM
git checkout f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f
pip install -r python/requirements.txt          # build-time dependencies
pip install wheel build
python3 setup.py bdist_wheel
pip install ./dist/*.whl
```

2. **Expose the Triton kernels to TensorRT-LLM**
   The kernels are not packaged in the wheel, so set the environment variable `TRITON_ROOT` to your Triton clone:

```bash
export TRITON_ROOT=/local/user/triton
# TensorRT-LLM expects the kernels at:
#   $TRITON_ROOT/python/triton_kernels
```

3. **Select Triton as the MoE backend**

• **trtllm-serve** (or other similar commands) — add this snippet to the YAML file passed via `--extra_llm_api_options`:

```yaml
moe_config:
  backend: TRITON
```

• **Example scripts** (e.g. `examples/llm-api/quickstart_advanced.py`) — pass the CLI flag:

```bash
--moe_backend TRITON
```
