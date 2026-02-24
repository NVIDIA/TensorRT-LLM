# HyperCLOVAX

This document shows how to build and run a [HyperCLOVAX](https://huggingface.co/naver/hyperclovax) model in TensorRT-LLM.


- [HyperCLOVAX](#hyperclovax)
  - [Support Matrix](#support-matrix)
  - [Supported Models](#supported-models)
    - [HyperCLOVAX-SEED-Text](#hyperclovax-seed-text)
    - [HyperCLOVAX-SEED-Vision](#hyperclovax-seed-vision)
  - [PyTorch flow](#pytorch-flow)
    - [LLM](#llm)
    - [Multimodal](#multimodal)
  - [TRT flow](#trt-flow)
    - [Convert checkpoint and build TensorRT engine(s)](#convert-checkpoint-and-build-tensorrt-engines)
    - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
    - [SmoothQuant](#smoothquant)
    - [Groupwise quantization (AWQ)](#groupwise-quantization-awq)
        - [W4A16 AWQ with FP8 GEMM (W4A8 AWQ)](#w4a16-awq-with-fp8-gemm-w4a8-awq)
    - [Run Engine](#run-engine)

## Support Matrix
  * FP16
  * BF16
  * Tensor Parallel
  * FP8
  * INT8 & INT4 Weight-Only
  * INT8 SmoothQuant
  * INT4 AWQ & W4A8 AWQ

## Supported Models
### HyperCLOVAX-SEED-Text

Download the HuggingFace checkpoints of the HyperCLOVAX-SEED-Text model. We support HyperCLOVAX-SEED-Text family, but here we will use the `HyperCLOVAX-SEED-Text-Instruct-0.5B` model as an example.

```bash
export MODEL_NAME=HyperCLOVAX-SEED-Text-Instruct-0.5B
git clone https://huggingface.co/naver-hyperclovax/$MODEL_NAME hf_models/$MODEL_NAME
```

### HyperCLOVAX-SEED-Vision
Download the HuggingFace checkpoints of the HyperCLOVAX-SEED-Vision model. We support the HyperCLOVAX-SEED-Vision model in [PyTorch flow](../../../llm-api).

```bash
export MODEL_NAME=HyperCLOVAX-SEED-Vision-Instruct-3B
git clone https://huggingface.co/naver-hyperclovax/$MODEL_NAME hf_models/$MODEL_NAME
```

## PyTorch flow

### LLM
To quickly run HyperCLOVAX-SEED-Text, you can use [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py):

```bash
pip install -r requirements.txt

python ../../../llm-api/quickstart_advanced.py --model_dir hf_models/$MODEL_NAME
```

The output will be like:
```bash
[0] Prompt: 'Hello, my name is', Generated text: ' [name] and I am a [position] at [company name]. I am interested in learning more about the [industry] and would like to discuss this further with you. I would appreciate it if you could provide me with a list of questions to ask you. Here are some questions that I would like to ask'
[1] Prompt: 'The president of the United States is', Generated text: ' the head of the executive branch, which is responsible for the day-to-day administration of the country. The president is the head of the executive branch, which is responsible for the day-to-day administration of the country. The president is the head of the executive branch, which is responsible for the day-to-day administration of the'
[2] Prompt: 'The capital of France is', Generated text: ' Paris, which is the largest city in the country. It is home to the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its rich history, cultural heritage, and culinary delights. The city is a hub for art, fashion, and entertainment, and is home'
[3] Prompt: 'The future of AI is', Generated text: " not just about technology, but about how we use it to improve our lives. It's about creating a world where technology and humanity work together to solve complex problems, make decisions, and enhance our quality of life. As we continue to develop and integrate AI into our daily lives, it's essential to consider the ethical implications"
```

### Multimodal
To quickly run HyperCLOVAX-SEED-Vision, you can use [examples/llm-api/quickstart_multimodal.py](../../../llm-api/quickstart_multimodal.py):

```bash
pip install -r requirements.txt

python ../../../llm-api/quickstart_multimodal.py --model_dir hf_models/$MODEL_NAME
```

The output will be like:
```bash
[0] Prompt: 'Describe the natural environment in the image.', Generated text: '이미지는 흐린 날씨에 거친 바다를 보여줍니다. 하늘은 어둡고 무거운 구름으로 덮여 있으며, 바다는 거센 파도가 치며 매우 거칠어 보입니다. 파도는 크고 흰 거품을 내며 부서지고 있고, 파도의 형태는 매우 역동적이며 에너지가 넘치는 모습입니다.'
[1] Prompt: 'Describe the object and the weather condition in the image.', Generated text: '이 이미지는 화창한 날씨에 촬영된 것으로 보입니다. 하늘은 맑고 푸른색을 띠고 있으며, 구름 몇 조각이 떠 있는 것을 볼 수 있습니다. 사진의 중앙에는 거대한 화강암 절벽이 우뚝 솟아 있으며, 그 절벽은 매우 가파른 경사를 가지고'
[2] Prompt: 'Describe the traffic condition on the road in the image.', Generated text: '이미지 속 도로의 교통 상태는 비교적 원활해 보입니다. 여러 차선이 있고, 차선마다 차량들이 일정한 간격을 유지하며 주행하고 있습니다. 도로의 왼쪽 차선에는 여러 대의 차량이 있고, 오른쪽 차선에도 몇 대의 차량이 보입니다. 도로의 중앙에는 파란'
```

For more information, you can refer to [examples/llm-api](../../../llm-api).

## TRT flow
The next section describes how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format. We will use llama's [convert_checkpoint.py](../../core/llama/convert_checkpoint.py) for the HyperCLOVAX model and then build the model with `trtllm-build`.

### Convert checkpoint and build TensorRT engine(s)

```bash
pip install -r requirements.txt

# Build a single-GPU float16 engine from HF weights.

# Build the HyperCLOVAX model using a single GPU and FP16.
python ../../core/llama/convert_checkpoint.py \
    --model_dir hf_models/$MODEL_NAME \
    --output_dir trt_models/$MODEL_NAME/fp16/1-gpu \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/fp16/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/fp16/1-gpu \
    --gemm_plugin auto

# Build the HyperCLOVAX model using a single GPU and apply INT8 weight-only quantization.
python ../../core/llama/convert_checkpoint.py \
    --model_dir hf_models/$MODEL_NAME \
    --output_dir trt_models/$MODEL_NAME/int8_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int8 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/int8_wq/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/int8_wq/1-gpu \
    --gemm_plugin auto

# Build the HyperCLOVAX model using a single GPU and apply INT4 weight-only quantization.
python ../../core/llama/convert_checkpoint.py \
    --model_dir hf_models/$MODEL_NAME \
    --output_dir trt_models/$MODEL_NAME/int4_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int4 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/int4_wq/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/int4_wq/1-gpu \
    --gemm_plugin auto

# Build the HyperCLOVAX model using 2-way tensor parallelism and FP16.
python ../../core/llama/convert_checkpoint.py \
    --model_dir hf_models/$MODEL_NAME \
    --output_dir trt_models/$MODEL_NAME/fp16/2-gpu \
    --tp_size 2 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/fp16/2-gpu \
    --output_dir trt_engines/$MODEL_NAME/fp16/2-gpu \
    --gemm_plugin auto
```

### FP8 Post-Training Quantization

The examples below use the NVIDIA Modelopt toolkit for the model quantization process.

First, make sure the Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the HyperCLOVAX model using a single GPU and apply FP8 quantization.
python ../../../quantization/quantize.py \
    --model_dir hf_models/$MODEL_NAME \
    --dtype float16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir trt_models/$MODEL_NAME/fp8/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/fp8/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/fp8/1-gpu \
    --gemm_plugin auto
```

### SmoothQuant

The examples below use the NVIDIA Modelopt toolkit for the model quantization process.

First, make sure the Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the HyperCLOVAX model using a single GPU and apply INT8 SmoothQuant.
python ../../../quantization/quantize.py \
    --model_dir hf_models/$MODEL_NAME \
    --dtype float16 \
    --qformat int8_sq \
    --output_dir trt_models/$MODEL_NAME/int8_sq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/int8_sq/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/int8_sq/1-gpu \
    --gemm_plugin auto
```

### Groupwise quantization (AWQ)

The examples below use the NVIDIA Modelopt toolkit for the model quantization process.

First, make sure the Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the HyperCLOVAX model using a single GPU and apply INT4 AWQ.
python ../../../quantization/quantize.py \
    --model_dir hf_models/$MODEL_NAME \
    --dtype float16 \
    --qformat int4_awq \
    --output_dir trt_models/$MODEL_NAME/int4_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/int4_awq/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/int4_awq/1-gpu \
    --gemm_plugin auto
```

#### W4A16 AWQ with FP8 GEMM (W4A8 AWQ)
For Hopper GPUs, TRT-LLM also supports using FP8 GEMM for accelerating linear layers. This mode is denoted as `w4a8_awq` for Modelopt and TRT-LLM, where both weights and activations are converted from W4A16 to FP8 for GEMM calculation.

Please ensure your system contains a Hopper GPU before trying the commands below.

```bash
# Build the HyperCLOVAX model using a single GPU and apply W4A8 AWQ.
python ../../../quantization/quantize.py \
    --model_dir hf_models/$MODEL_NAME \
    --dtype float16 \
    --qformat w4a8_awq \
    --output_dir trt_models/$MODEL_NAME/w4a8_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/$MODEL_NAME/w4a8_awq/1-gpu \
    --output_dir trt_engines/$MODEL_NAME/w4a8_awq/1-gpu \
    --gemm_plugin auto
```

### Run Engine
Test your engine with the [run.py](../run.py) script:

```bash
python3 ../../../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir hf_models/$MODEL_NAME \
    --engine_dir trt_engines/$MODEL_NAME/fp16/1-gpu

# Run with 2 GPUs
mpirun -n 2 --allow-run-as-root \
    python3 ../../../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir hf_models/$MODEL_NAME \
    --engine_dir trt_engines/$MODEL_NAME/fp16/2-gpu

python ../../../summarize.py \
    --test_trt_llm \
    --data_type fp16 \
    --hf_model_dir hf_models/$MODEL_NAME \
    --engine_dir trt_engines/$MODEL_NAME/fp16/1-gpu
```

The TensorRT LLM HyperCLOVAX implementation is based on the LLaMA model. The implementation can be found in [llama/model.py](../../../../tensorrt_llm/models/llama/model.py).
For more examples, see [`examples/models/core/llama/README.md`](../../core/llama/README.md)
