# EXAONE

This document shows how to build and run [EXAONE](https://huggingface.co/LGAI-EXAONE) models in TensorRT-LLM.

- [EXAONE](#exaone)
  - [Support Matrix](#support-matrix)
  - [Supported Models](#supported-models)
    - [EXAONE-3.0](#exaone-30)
    - [EXAONE-Deep](#exaone-deep)
    - [EXAONE-4.0](#exaone-40)
    - [K-EXAONE](#k-exaone)
  - [PyTorch flow](#pytorch-flow)
    - [Running EXAONE-4.0](#running-exaone-40)
    - [Running K-EXAONE](#running-k-exaone)
      - [MoE Backend Options](#moe-backend-options)
    - [PyTorch flow Quantization](#pytorch-flow-quantization)
      - [FP8 Quantization](#fp8-quantization)
      - [NVFP4 Quantization](#nvfp4-quantization)
  - [Running the TensorRT LLM Server](#running-the-tensorrt-llm-server)
    - [Running Aggregated TensorRT LLM Server](#running-aggregated-tensorrt-llm-server)
      - [Creating the Extra Options Configuration](#creating-the-extra-options-configuration)
      - [Launch trtllm-serve OpenAI-compatible API server](#launch-trtllm-serve-openai-compatible-api-server)
    - [Running Disaggregated TensorRT LLM Server](#running-disaggregated-tensorrt-llm-server)
      - [Step 1: Set Environment Variables](#step-1-set-environment-variables)
      - [Step 2: Create Configuration Files](#step-2-create-configuration-files)
      - [Step 3: Launch the Disaggregated Server](#step-3-launch-the-disaggregated-server)
  - [TRT flow](#trt-flow)
    - [Convert checkpoint and build TensorRT engine(s)](#convert-checkpoint-and-build-tensorrt-engines)
    - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
    - [SmoothQuant](#smoothquant)
    - [Groupwise quantization (AWQ)](#groupwise-quantization-awq)
      - [W4A16 AWQ with FP8 GEMM (W4A8 AWQ)](#w4a16-awq-with-fp8-gemm-w4a8-awq)
    - [Run Engine](#run-engine)
  - [Troubleshootings](#troubleshootings)
    - [Troubleshootings for EXAONE-4.0](#troubleshootings-for-exaone-40)
    - [Troubleshootings for K-EXAONE](#troubleshootings-for-k-exaone)

## Support Matrix
  * FP16
  * BF16
  * Tensor Parallel (TP)
  * Expert Parallel (EP) (K-EXAONE only)
  * Attention Data Parallel (ADP) (K-EXAONE only)
  * Disaggregated Serving
  * MTP (Multi Token Prediction)
  * FP8
  * INT8 & INT4 Weight-Only
  * INT8 SmoothQuant
  * INT4 AWQ & W4A8 AWQ
  * NVFP4 (K-EXAONE only)

## Supported Models

**Note:**
- **EXAONE-3.0** & **EXAONE-Deep** are supported using the [TRT Flow](#trt-flow).
- **EXAONE-4.0** & **K-EXAONE** are supported using the [PyTorch flow](#pytorch-flow).

Please refer to the corresponding sections below for usage instructions and examples for each model.

### EXAONE-3.0

Download the HuggingFace FP32 checkpoints of EXAONE-3.0 model. We support EXAONE-3.0 families but here, we only use the `EXAONE-3.0-7.8B-Instruct` model for the example.

```bash
export HF_MODEL_DIR=hf_models/exaone
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct $HF_MODEL_DIR
```

### EXAONE-Deep

Download the HuggingFace checkpoints of EXAONE-Deep model. Here, we only use the `EXAONE-Deep-2.4B` model for the example. We can use the same procedure as EXAONE-3.0 to convert the weights and build the TensorRT engine.

```bash
export HF_MODEL_DIR=hf_models/exaone_deep
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B $HF_MODEL_DIR
```

### EXAONE-4.0

Download the HuggingFace checkpoints of the EXAONE-4.0 model. Here, we use the `EXAONE-4.0-32B` model as an example. EXAONE-4.0 is supported only via the PyTorch flow.

```bash
export HF_MODEL_DIR=hf_models/exaone4
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B $HF_MODEL_DIR
```

### K-EXAONE

K-EXAONE is a Mixture of Experts (MoE) model based on the EXAONE architecture. It features a hybrid architecture with both dense and MoE layers, sliding window attention, and supports FP8 and NVFP4 quantization for efficient inference.

Download the HuggingFace checkpoints of the K-EXAONE model:

```bash
export HF_MODEL_DIR=hf_models/kexaone
git clone https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B $HF_MODEL_DIR
```

## PyTorch flow

### Running EXAONE-4.0
To quickly run EXAONE-4.0 models, you can use [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py):

```bash
python ../../../llm-api/quickstart_advanced.py --model_dir $HF_MODEL_DIR
```

The output will be like:
```bash
[0] Prompt: 'Hello, my name is', Generated text: " [Your Name], and I'm a [Your Profession]. I'm here to learn and share with you.\n\nBest regards,\n[Your Name]\n\nThis letter is concise, professional, and clearly states who you are and what you're here for. It's a good starting point"
[1] Prompt: 'The capital of France is', Generated text: ' Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris'
[2] Prompt: 'The future of AI is', Generated text: ' not just about technology but also about how we choose to use it. We must ensure that AI is developed and deployed in a way that benefits all of humanity, not just a select few. This means prioritizing ethical considerations, transparency, and accountability in AI development. It also means involving diverse stakeholders in the conversation about AI'
```

### Running K-EXAONE

K-EXAONE is a Mixture of Experts model that benefits from multiple parallelism strategies. You can run it with tensor parallelism (TP), expert parallelism (EP), and attention data parallelism (ADP):

```bash
python ../../../llm-api/quickstart_advanced.py \
    --model_dir $HF_MODEL_DIR \
    --tp_size 8 \
    --moe_ep_size 8 \
    --enable_attention_dp \
    --apply_chat_template
```
The output will be like:
```bash
[0] Prompt: '<|user|>\nHello, my name is<|endofturn|>\n<|assistant|>\n<think>\n', Generated text: 'Okay, the user started with "Hello, my name is" and then stopped. Hmm, they probably forgot to finish their sentence. \n\nI should figure out what they need. Since they\'re introducing themselves, maybe they want help completing their introduction. Or perhaps they\'re testing how I respond to incomplete messages.\n\nLet me check the context again. The user\'s message is cut off'
[1] Prompt: '<|user|>\nThe capital of France is<|endofturn|>\n<|assistant|>\n<think>\n', Generated text: 'Okay, the user asked, "The capital of France is". Hmm, this seems like a straightforward question. But let me think deeper.\n\nFirst, the user might be testing basic knowledge. Or perhaps they\'re a student learning geography. Alternatively, they could be someone verifying information, maybe for a project or trivia.\n\nWait, the user\'s query is incomplete incomplete.'
[2] Prompt: '<|user|>\nThe future of AI is<|endofturn|>\n<|assistant|>\n<think>\n', Generated text: 'Okay, the user asked, "The future of AI is..." and stopped. Hmm, they probably want me to complete that thought or elaborate on what the future of AI entails.\n\nFirst, I need to figure out what the user is really looking for. They might be a student researching AI, a professional trying to understand industry trends, or just someone curious about where AI is heading.\n\n\n\nSince
```

#### MoE Backend Options

K-EXAONE supports the following MoE backends:

| Backend | Description |
|---------|-------------|
| `CUTLASS` | Default backend, optimized for general use cases |
| `TRTLLM` | TensorRT-LLM backend using TRT-LLM Gen kernels, optimized for low-latency inference |
| `WIDEEP` | Wide expert parallelism backend for cases where EP size exceeds the number of experts |

You can specify the MoE backend using the `--moe_backend` argument:

```bash
python ../../../llm-api/quickstart_advanced.py \
    --model_dir $HF_MODEL_DIR \
    --tp_size 8 \
    --moe_ep_size 8 \
    --enable_attention_dp \
    --moe_backend CUTLASS
```

#### MTP (Multi-Token Prediction)

K-EXAONE has 1 MTP layer. To run with MTP, use [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py) with additional options:

```bash
python ../../../llm-api/quickstart_advanced.py \
    --model_dir $HF_MODEL_DIR \
    --tp_size 8 \
    --moe_ep_size 8 \
    --enable_attention_dp \
    --spec_decode_algo MTP \
    --spec_decode_max_draft_len N \
    --use_one_model
```

`N` is the number of MTP modules. When `N` is equal to `0`, which means that MTP is not used (default). When `N` is greater than `0`, which means that `N` MTP modules are enabled. In the current implementation, the weight of each MTP module is shared.

### PyTorch flow Quantization

For PyTorch flow, TRT-LLM supports quantized formats generated by [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer). You can either use pre-quantized models from the HuggingFace model hub, or generate quantized models yourself using the instructions below.

First, clone the [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) repository:

```bash
git clone https://github.com/NVIDIA/Model-Optimizer
cd Model-Optimizer/examples/llm_ptq
```

For more information, please refer to the official [Model Optimizer documentation](https://github.com/NVIDIA/Model-Optimizer).

#### FP8 Quantization

FP8 quantization provides a good balance between model accuracy and inference performance. To quantize a model to FP8 format:

```bash
python3 hf_ptq.py --model $HF_MODEL_DIR --quant fp8 --export_fmt hf
```

#### NVFP4 Quantization

NVFP4 (4-bit floating point) quantization enables memory-efficient inference with reduced GPU memory footprint. To quantize a model to NVFP4 format:

```bash
python3 hf_ptq.py --model $HF_MODEL_DIR --quant nvfp4 --export_fmt hf
```

## Running the TensorRT LLM Server

This section describes how to deploy the K-EXAONE model using the TensorRT LLM server with an OpenAI-compatible API endpoint.
Make sure `HF_MODEL_DIR` points to your EXAONE checkpoint directory.

The examples in this section are intended as a minimal, runnable demonstration and are not fully performance-optimized. For more features and performance tuning, please refer the documents below.
- [Disaggregated Serving examples](../../../disaggregated/README.md)
- [Disaggregated Serving feature guide](../../../../docs/source/features/disagg-serving.md)
- [Recommended LLM API configuration settings](../../../configs/README.md) (see also `examples/configs/curated/`)

### Running Aggregated TensorRT LLM Server

The aggregated server runs all components (context and generation phases) on the same set of GPUs, which is suitable for single-node deployments.

#### Creating the Extra Options Configuration

Create a YAML configuration file to specify advanced options such as attention data parallelism, CUDA graph settings, and MoE backend configuration:

```bash
cat <<EOF > configs.yaml
enable_attention_dp: true
trust_remote_code: true
cuda_graph_config:
  max_batch_size: 2048
  enable_padding: true
moe_config:
  backend: CUTLASS  # The TRTLLM backend is recommended for the Blackwell architecture.
kv_cache_config:
  enable_block_reuse: true  # Please disable the block reuse feature when conducting performance benchmarking.
  max_attention_window: [128, 128, 128, 131072]  # This allows KV cache manager to possibly improve memory efficiency.
  free_gpu_memory_fraction: 0.9
  dtype: "auto"
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 50
  timeout_iters: 1
num_postprocess_workers: 4  # Can mitigate the postprocessing overhead (e.g. detokenization)
EOF
```

#### Launch trtllm-serve OpenAI-compatible API server

Start the server using `trtllm-serve` with the PyTorch backend. This launches an OpenAI-compatible API server that can handle chat completions and text generation requests:

```bash
trtllm-serve \
  $HF_MODEL_DIR \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 2048 \
  --max_num_tokens 8192 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --config ./configs.yaml
```

Once the server is running, you can send requests to `http://localhost:8000/v1/completions` using the OpenAI API format.

### Running Disaggregated TensorRT LLM Server

Disaggregated serving separates the context (prefill) and generation (decode) phases onto different GPU sets, enabling better resource utilization and improved throughput. This example demonstrates a single-node disaggregated deployment using 8 GPUs (4 for context, 4 for generation). For more details, see the [Disaggregated Serving documentation](../../../disaggregated/README.md).

#### Step 1: Set Environment Variables

Configure the parallelism and buffer settings:

```bash
# Buffer size for KV cache transfer between context and generation servers
export MAX_TOKENS_IN_BUFFER=8192

# Model parallelism configuration
export TP_SIZE=4
export MOE_EP_SIZE=4
export ENABLE_ATTENTION_DP=true
```

#### Step 2: Create Configuration Files

**Context server configuration (`ctx_extra-llm-api-config.yaml`):**

```bash
cat > ctx_extra-llm-api-config.yaml << EOF
backend: pytorch
trust_remote_code: true
disable_overlap_scheduler: true
enable_chunked_prefill: true

tensor_parallel_size: $TP_SIZE
moe_expert_parallel_size: $MOE_EP_SIZE
pipeline_parallel_size: 1
enable_attention_dp: $ENABLE_ATTENTION_DP

cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: $MAX_TOKENS_IN_BUFFER
EOF
```

**Generation server configuration (`gen_extra-llm-api-config.yaml`):**

```bash
cat > gen_extra-llm-api-config.yaml << EOF
backend: pytorch
trust_remote_code: true
disable_overlap_scheduler: false
enable_chunked_prefill: true

tensor_parallel_size: $TP_SIZE
moe_expert_parallel_size: $MOE_EP_SIZE
pipeline_parallel_size: 1
enable_attention_dp: $ENABLE_ATTENTION_DP

cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: $MAX_TOKENS_IN_BUFFER
EOF
```

**Disaggregated orchestrator configuration (`disagg_config.yaml`):**

```bash
cat > disagg_config.yaml << EOF
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
    - "localhost:8002"
EOF
```

#### Step 3: Launch the Disaggregated Server

Start all components in the following order:

```bash
# 1. Start context server (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-serve $HF_MODEL_DIR \
    --host localhost --port 8001 --enable_chunked_prefill \
    --extra_llm_api_options ./ctx_extra-llm-api-config.yaml &> log_ctx.log &

# 2. Start generation server (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 trtllm-serve $HF_MODEL_DIR \
    --host localhost --port 8002 --enable_chunked_prefill \
    --extra_llm_api_options ./gen_extra-llm-api-config.yaml &> log_gen.log &

# 3. Start disaggregated orchestrator
trtllm-serve disaggregated -c disagg_config.yaml -t 360 -r 1200 &> log_disagg.log &
```

Once all servers are running, you can send requests to `http://localhost:8000/v1/completions` using the OpenAI API format.


## TRT flow

The next section describes how to convert weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format. We will use LLaMA's [convert_checkpoint.py](../llama/convert_checkpoint.py) for EXAONE models and then build the model with `trtllm-build`.

### Convert checkpoint and build TensorRT engine(s)

```bash
# Build a single-GPU float16 engine from HF weights.

# Build the EXAONE model using a single GPU and FP16.
python ../llama/convert_checkpoint.py \
    --model_dir $HF_MODEL_DIR \
    --output_dir trt_models/exaone/fp16/1-gpu \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp16/1-gpu \
    --output_dir trt_engines/exaone/fp16/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using a single GPU and apply INT8 weight-only quantization.
python ../llama/convert_checkpoint.py \
    --model_dir $HF_MODEL_DIR \
    --output_dir trt_models/exaone/int8_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int8 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/int8_wq/1-gpu \
    --output_dir trt_engines/exaone/int8_wq/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using a single GPU and apply INT4 weight-only quantization.
python ../llama/convert_checkpoint.py \
    --model_dir $HF_MODEL_DIR \
    --output_dir trt_models/exaone/int4_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int4 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/int4_wq/1-gpu \
    --output_dir trt_engines/exaone/int4_wq/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using 2-way tensor parallelism and FP16.
python ../llama/convert_checkpoint.py \
    --model_dir $HF_MODEL_DIR \
    --output_dir trt_models/exaone/fp16/2-gpu \
    --tp_size 2 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp16/2-gpu \
    --output_dir trt_engines/exaone/fp16/2-gpu \
    --gemm_plugin auto
```
> **NOTE**: EXAONE model is not supported with `--load_by_shard`.

### FP8 Post-Training Quantization

The examples below use the NVIDIA ModelOpt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and apply FP8 quantization.
python ../../../quantization/quantize.py \
    --model_dir $HF_MODEL_DIR \
    --dtype float16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir trt_models/exaone/fp8/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp8/1-gpu \
    --output_dir trt_engines/exaone/fp8/1-gpu \
    --gemm_plugin auto
```

### SmoothQuant

The examples below use the NVIDIA ModelOpt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and apply INT8 SmoothQuant.
python ../../../quantization/quantize.py \
    --model_dir $HF_MODEL_DIR \
    --dtype float16 \
    --qformat int8_sq \
    --output_dir trt_models/exaone/int8_sq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/int8_sq/1-gpu \
    --output_dir trt_engines/exaone/int8_sq/1-gpu \
    --gemm_plugin auto
```

### Groupwise quantization (AWQ)

The examples below use the NVIDIA ModelOpt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and apply INT4 AWQ.
python ../../../quantization/quantize.py \
    --model_dir $HF_MODEL_DIR \
    --dtype float16 \
    --qformat int4_awq \
    --output_dir trt_models/exaone/int4_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/int4_awq/1-gpu \
    --output_dir trt_engines/exaone/int4_awq/1-gpu \
    --gemm_plugin auto
```

#### W4A16 AWQ with FP8 GEMM (W4A8 AWQ)
For Hopper GPUs, TRT-LLM also supports employing FP8 GEMM for accelerating linear layers. This mode is noted with `w4a8_awq` for Modelopt and TRT-LLM, in which both weights and activations are converted from W4A16 to FP8 for GEMM calculation.

Please make sure your system contains a Hopper GPU before trying the commands below.

```bash
# Build the EXAONE model using a single GPU and apply W4A8 AWQ.
python ../../../quantization/quantize.py \
    --model_dir $HF_MODEL_DIR \
    --dtype float16 \
    --qformat w4a8_awq \
    --output_dir trt_models/exaone/w4a8_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/w4a8_awq/1-gpu \
    --output_dir trt_engines/exaone/w4a8_awq/1-gpu \
    --gemm_plugin auto
```


### Run Engine
Test your engine with the [run.py](../run.py) script:

```bash
python3 ../../../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir $HF_MODEL_DIR \
    --engine_dir trt_engines/exaone/fp16/1-gpu

# Run with 2 GPUs
mpirun -n 2 --allow-run-as-root \
    python3 ../../../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir $HF_MODEL_DIR \
    --engine_dir trt_engines/exaone/fp16/2-gpu

python ../../../summarize.py \
    --test_trt_llm \
    --data_type fp16 \
    --hf_model_dir $HF_MODEL_DIR \
    --engine_dir trt_engines/exaone/fp16/1-gpu
```

For more examples regarding EXAONE-3.0 & EXAONE-Deep's TRT flow, see [`examples/models/core/llama/README.md`](../llama/README.md)



## Troubleshootings

### Troubleshootings for EXAONE-4.0

The following error may occur during quantization:
```bash
torch._dynamo.exc.Unsupported: Graph break under GenericContextWrappingVariable
Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
Hint: Move the offending context manager(s) to outside the compiled region.
Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
```

This error may indicate an incompatibility between `torch.compile()` and the `HybridCache` module of the transformers library. As a result, [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt) cannot perform PTQ with HybridCache.

Temporarily switching to `DynamicCache` when creating PTQ models could help address the issue. This can be done by updating the `cache_implementation` field in the `generation_config.json` file located in the model checkpoint directory, for example:
```json
# generation_config.json
{
    // Change "hybrid" to "dynamic" to run PTQ.
    // Revert this to "hybrid" after quantization is complete.
    "cache_implementation": "hybrid",
    ...
}
```
For models with sliding window attention, DynamicCache is less memory-efficient than HybridCache because it retains the entire key-value cache. However, this does not break the model's attention logic, as the cache implementation is separated from the attention computation itself. This trade-off is acceptable for the PTQ process, which is a one-time procedure. Our tests confirm that this workaround does not degrade accuracy on MMLU or GSM8K benchmarks with the default ModelOpt settings.

### Troubleshootings for K-EXAONE

K-EXAONE is a Mixture of Experts (MoE) model which activates 8 experts per token. When not enough tokens are given during the PTQ, some experts on some layers might not be activated and will not produce proper weights.

To address this issue, provide enough data samples during calibration by increasing `calib_size` and `calib_seq` parameters:

**FP8 Quantization:**
```bash
cd Model-Optimizer/examples/llm_ptq
python3 hf_ptq.py --model hf_models/$MODEL_NAME --quant fp8 --export_fmt hf --calib_size 8192 --calib_seq 1024
```

**NVFP4 Quantization:**
```bash
cd Model-Optimizer/examples/llm_ptq
python3 hf_ptq.py --model hf_models/$MODEL_NAME --quant nvfp4 --export_fmt hf --calib_size 8192 --calib_seq 1024
```
