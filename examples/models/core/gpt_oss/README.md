# GPT-OSS

## Overview

GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the other weights are in bf16.

## MoE Support Matrix

In MoE, the weights are pre-quantized to mxfp4. The activation can be in either bf16 (Hopper) or mxfp8 (Blackwell), with similar accuracy.

| device | Activation | Weight | Supported moe_backend |
|----------|----------|----------|----------|
| Hopper | bf16 | mxfp4 | **TRITON**, CUTLASS |
| Blackwell | mxfp8 | mxfp4 | CUTLASS, TRTLLM |


| moe_backend | TP | EP | AlltoAll |
|----------|----------|----------|----------|
| CUTLASS | yes | yes | yes |
| TRTLLM | yes | yes | no |
| TRITON | no | yes | no |

For best performance, use the `TRITON` moe_backend on Hopper for both latency and throughput cases. Use `CUTLASS` for throughput cases and `TRTLLM` for latency cases on Blackwell.

## Harmony Examples

### Function Calling

OpenAI MoE models support function calling. Here is an example based on [XGrammar](https://github.com/mlc-ai/xgrammar)'s structural tag.

First, launch a server with XGrammar enabled:

```bash
cat > ./extra_llm_api_options.yaml <<EOF
guided_decoding_backend: xgrammar
EOF

trtllm-serve <model> \
    --backend pytorch \
    --extra_llm_api_options extra_llm_api_options.yaml
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
[RESPONSE 1] <|channel|>analysis<|message|>The user asks: "What is the weather like in SF?" They want the weather in SF. SF likely refers to San Francisco. We need to get the current weather. We can use get_current_weather function. We need to provide location string "San Francisco, CA". We can also ask for format? By default celsius. But maybe user expects Fahrenheit? They didn't specify. We can provide celsius or Fahrenheit. We can choose default celsius. But maybe better to provide Fahrenheit because US. But default is celsius. We can provide both? We can call function with format "fahrenheit" to be user-friendly. But the function default is celsius. We can override. Let's call get_current_weather with location "San Francisco, CA" and format "fahrenheit". Then we will get the weather. Then we will respond with friendly tone. We need to call the function.<|end|><|start|>assistant<|channel|>commentary to=get_current_weather <|constrain|>json<|message|>{
  "location": "San Francisco, CA",
 "format": "fahrenheit"
}<|call|>
[FUNCTION CALL] get_current_weather(**{'location': 'San Francisco, CA', 'format': 'fahrenheit'})
[RESPONSE 2] <|channel|>analysis<|message|>The user asked: "What is the weather like in SF?" We have fetched the weather: sunny true, temperature 68 (F). We need to respond in a friendly tone. Provide a friendly answer: "It's sunny and 68°F in San Francisco." Possibly add a friendly comment. Also ask if they want more details.<|end|><|start|>assistant<|channel|>final<|message|>Sure thing! It’s a pleasant 68 °F in San Francisco right now, and the sun is shining. 🌞

Anything else you'd like to know about the weather or maybe some fun things to do in the city today?<|return|>
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
[RESPONSE 1] <|channel|>analysis<|message|>The user asks: "What is the weather like in NY and SF?" They want the weather in New York and San Francisco. We need to provide the weather. We can use the function get_multiple_weathers. We need to provide the list of city and state strings. For New York, we can use "New York, NY". For San Francisco, "San Francisco, CA". We can call get_multiple_weathers with those two locations. We should specify format? The default is celsius. But maybe the user might want Fahrenheit? They didn't specify. We can just use default celsius. But maybe we can provide both? But the function only returns one format. We can just use default celsius. But we can also ask the user? But the user asked "What is the weather like in NY and SF?" We can just provide the weather. We can call the function. Then we will get the weather data. Then we can respond with a friendly tone. We need to call the function.<|end|><|start|>assistant<|channel|>commentary to=get_multiple_weathers <|constrain|>json<|message|>{"locations":["New York, NY","San Francisco, CA"]}<|call|>
[FUNCTION CALL] get_multiple_weathers(**{'locations': ['New York, NY', 'San Francisco, CA']})
[RESPONSE 2] <|channel|>analysis<|message|>The user asked: "What is the weather like in NY and SF?" We called get_multiple_weathers with locations ["New York, NY","San Francisco, CA"]. The function returned: [{"sunny": true, "temperature": 20}, {"sunny": true, "temperature": 20}]. That seems to be a list of two objects, each with sunny: true, temperature: 20. But we need to interpret the function output. The function get_multiple_weathers presumably returns a list of weather data for each location. But the returned data is ambiguous: we don't know which corresponds to which location. But we can assume the order matches the input order: first is New York, second is San Francisco. The temperature is 20 degrees Celsius? The function didn't specify units, but default is celsius. So 20°C. And sunny: true. So both are sunny and 20°C. We should respond in a friendly tone, summarizing the weather for both cities. We can mention that it's sunny and 20°C in both New York and San Francisco. We can also mention that it's a nice day. We can ask if they want more details. We should not mention the function call. We should just provide the answer.<|end|><|start|>assistant<|channel|>final<|message|>Sure thing! Here’s the scoop:

- **New York, NY**: It’s sunny and a comfortable 20 °C (68 °F).
- **San Francisco, CA**: Also sunny with a pleasant 20 °C (68 °F).

Looks like both coasts are enjoying a bright, mild day. Let me know if you’d like a forecast for later or any other details!<|return|>
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
