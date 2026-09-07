# HyperCLOVAX

> [!WARNING]
> The legacy TensorRT engine-build workflow (`convert_checkpoint.py` /
> `trtllm-build` / `run.py`) was **removed** with the TensorRT backend.
> Use the PyTorch flow below via
> [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
> or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html).

This document shows how to build and run a [HyperCLOVAX](https://huggingface.co/naver/hyperclovax) model in TensorRT-LLM.


- [HyperCLOVAX](#hyperclovax)
  - [Support Matrix](#support-matrix)
  - [Supported Models](#supported-models)
    - [HyperCLOVAX-SEED-Text](#hyperclovax-seed-text)
    - [HyperCLOVAX-SEED-Vision](#hyperclovax-seed-vision)
  - [PyTorch flow](#pytorch-flow)
    - [LLM](#llm)
    - [Multimodal](#multimodal)
  - [TRT flow (removed)](#trt-flow-removed)

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


## TRT flow (removed)

The legacy engine-build path documented here previously
(`examples/models/core/llama/convert_checkpoint.py`, `trtllm-build`, and
`examples/run.py`) was removed together with the TensorRT backend. Those paths
and the `trtllm-build` CLI are no longer in the tree.

Use the **PyTorch flow** section above (`trtllm-serve` / LLM API) instead.
