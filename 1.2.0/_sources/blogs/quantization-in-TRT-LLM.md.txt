# Speed up inference with SOTA quantization techniques in TRT-LLM

The deployment and inference speed of LLMs are often impeded by limitations in memory capacity, memory bandwidth, and computation power. Quantization emerges as a vital strategy to address these bottlenecks, involving representing weights and activations with lower-precision data types like [FP8](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/).

In this blog, we provide an overview of the quantization features in TensorRT-LLM, share benchmark, and offer best practices of selecting the appropriate quantization methods tailored to your specific use case.

## Quantization in TensorRT-LLM
TensorRT LLM offers a best-in-class unified quantization toolkit to significantly speedup DL/GenAI deployment on NVIDIA hardware, while maintaining model accuracy. This toolkit is designed with easy-of-use in mind. You can follow [this user guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) to quantize [supported LLMs](../reference/support-matrix.md#models) with a few lines of codes. We currently focus on providing SOTA **Post-Training Quantization (PTQ)** and will soon expand to more model optimization techniques in the near future.

## Benchmark

### Performance
In the following benchmark, we highlight the acceleration of a few popular models at a small batch size without imposing latency constraints. It's important to note that in scenarios where there's a latency constraint in your application, TRT-LLM can achieve an even greater performance improvement. Using LLaMA-v2-7B as an example, when the first token latency is constrained to be under 500ms, quantization with FP8 and a batch size of 16 achieves a notable **2.3x inference speedup** compared to FP16 on a H100.

| Model       | Batch Size | Speedup (FP8 v.s. FP16) | Speedup (INT8 SQ v.s. FP16) |
| ----------- | :--------: | :---------------------: | :-------------------------: |
| GPT-J       |     1      |          1.40x          |            1.40x            |
| GPT-J       |     8      |          1.44x          |            1.30x            |
| LLaMA-v2-7B |     1      |          1.51x          |            1.47x            |
| LLaMA-v2-7B |     8      |          1.40x          |            1.32x            |

*The above benchmarks were run with Input Length=1024, Output Length=128, and TP=1 on H100 80GB.

### Accuracy

| Model        | Quantization Methods | MMLU Baseline (FP16) | MMLU Post-quantization | MMLU Loss |
| ------------ | :------------------: | :------------------: | :--------------------: | :-------: |
| Falcon-180B  |         FP8          |         70.4         |          70.3          |   0.14%   |
|              |       INT8-SQ        |         70.4         |          68.6          |   2.56%   |
|              |       INT4-AWQ       |         70.4         |          69.8          |   0.85%   |
| Falcon-40B   |         FP8          |         56.1         |          55.6          |   0.89%   |
|              |       INT8-SQ        |         56.1         |          54.7          |   2.50%   |
|              |       INT4-AWQ       |         56.1         |          55.5          |   1.07%   |
| LLaMA-v2-70B |         FP8          |         69.1         |          68.5          |   0.87%   |
|              |       INT8-SQ        |         69.1         |          67.2          |   2.75%   |
|              |       INT4-AWQ       |         69.1         |          68.4          |   1.01%   |
| MPT-30B      |         FP8          |         47.5         |          47.4          |   0.21%   |
|              |       INT8-SQ        |         47.5         |          46.8          |   1.47%   |
|              |       INT4-AWQ       |         47.5         |          46.5          |   2.11%   |



## Best practices to choose the right quantization methods
A quantization method comprises three primary components:
1. Weight precision format
2. Activation precision format
3. Calibration algorithms

Typically, in the context of small-batch inference scenarios (batch size ≤ 4), the key consideration is memory bandwidth, making weight-only quantization methods the preferred choice. Conversely, for large-batch inference scenarios, such as serving scenarios (batch size ≥ 16), both memory bandwidth and computation density become crucial factors. Consequently, it's recommended to opt for a quantization method that has both weight and activation quantized. For batch size ≥ 16, the choice of quantization method can be model specific. We suggest to prioritize using FP8 first, as we typically see it offers the best performance and accuracy. If the results do not meet your specific use case, you can further experiment with Int8 SmoothQuant (Int8 SQ) followed by AWQ and/or GPTQ.

Based on specific use cases, users might have different tolerances on accuracy impact and calibration time. The table below summarizes the tradeoffs* to consider when choosing a quantization method. You can also learn more about precision formats in our [documentation](https://nvidia.github.io/TensorRT-LLM/reference/precision.html).

| Quantization Methods     | Performance Improvement (batch size <= 4) | Performance Improvement (batch size >= 16) | Accuracy Impact | Calibration Time** |
| :----------------------- | :---------------------------------------: | :----------------------------------------: | :-------------: | :----------------: |
| FP8 (W8A8)               |                  Medium                   |                   Medium                   |    Very Low     |      Minutes       |
| Int8 SQ (W8A8)           |                  Medium                   |                   Medium                   |     Medium      |      Minutes       |
| Int8 weight-only (W8A16) |                  Medium                   |                    Low                     |       Low       |    Not Required    |
| Int4 weight-only (W4A16) |                   High                    |                    Low                     |      High       |    Not Required    |
| Int4 AWQ (W4A16)         |                   High                    |                    Low                     |       Low       |  Tens of Minutes   |
| Int4 GPTQ                |                   High                    |                    Low                     |       Low       |  Tens of Minutes   |
| Int4-FP8 AWQ (W4A8)      |                   High                    |                   Medium                   |       Low       |  Tens of Minutes   |

\* The performance and impact are measured on 10+ popular LLMs. We'll follow up with more data points.
** Calibration time is subject to the actual model size.

We note that TensorRT LLM also offers INT8 and FP8 quantization for KV cache. KV cache differs from normal activation because it occupies non-negligible persistent memory under scenarios like large batch sizes or long context lengths. If you're using KV cache on Hopper & Ada GPUs, We recommend using FP8 KV cache over Int8 because the former has a lower accuracy impact than the latter in most tested cases. When switching from FP16 KV cache to FP8 KV cache, it also enables you to run 2-3x larger batch size on H100 machine for models like GPT-J which further brings about 1.5x performance benefit.

## What’s coming next
TensorRT LLM continues to make improvements on our quantization features, such as Int4-FP8 AWQ (W4A8) public examples and more model supports. Please stay tuned for our upcoming releases.
