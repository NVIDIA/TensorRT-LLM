(fp8-quantization)=

# FP8 Quantization

Quantization is a technique that allows models to run in lower precisions like int8 and fp8 while maintaining acceptable output quality. Running in lower precisions can greatly boost performance, significantly increasing throughput and decreasing latency. The tradeoff is a drop in output quality, but in many cases the output quality is still acceptable and many real world deployments utilize quantization. If you want to learn more about quantization refer to [Mastering LLM Techniques - Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

This section walks through enabling fp8 quantization and highlight some fp8 quantization specific configuration options for boosting performance. It also continues the case study of Llama-3.3-70B split across 4 H100-sxm-80GB GPUs via tensor parallelism and showcase the effects of enabling these configuration options on performance.

> Disclaimer: While performance numbers shown here are real, they are only for demonstration purposes. Differences in environment, SKU, interconnect, and workload can all significantly affect performance and lead to your results differing from what is shown here.

## Enabling Quantization

To enable quantization you need to configure the `QuantConfig` class and pass it to the `quant_config` parameter of the LLM class. At a minimum the `quant_algo` parameter, which sets the quantization algorithm (fp8, fp8 per token, int8awq, etc.) must be specified. You can find all supported quantization algorithms and other configurable options for `QuantConfig` in the LLM-API->Reference section of the docs. While it is not required if you are using weights/checkpoints from that are already quantized, if you are using an fp16 checkpoint then you also need to specify the calibration dataset that will be used to determine the quantization scales via `CalibConfig`. `CalibConfig` provides several options for setting the calibration dataset that can also be referenced in the LLM-API->Reference section of the docs. Although TensorRT-LLM supports several other types of quantization, this guide focuses on fp8.


Here is an example of building and saving an fp8 engine from a bf16 checkpoint (Note that fp8 is supported only on devices with compute capability > 8.9 - Ada, Hopper, Blackwell, and beyond):
```python
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig

def main():

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)

    calib_config = CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096
    )

    build_config = BuildConfig(
        max_num_tokens=2048,
        max_batch_size=512,
    )

    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.multiple_profiles = True

    llm = LLM(
        model="/path/to/Llama-3.3-70B",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        build_config=build_config,
        quant_config=quant_config,
        calib_config=calib_config
    )

    llm.save("baseline_fp8_engine")

if __name__ == '__main__':
    main()
```

For an example of how to build an fp8 engine using the [TensorRT-LLM CLI workflow](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) flow see [TensorRT-LLM LLaMA examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama). In short you first run [`examples/quantization/quantize.py`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) to quantize and convert the model checkpoint to TensorRT-LLM format and then use `trtllm-build`.

> ***Note: While quantization aims to preserve model accuracy this is not guaranteed and it is extremely important you check that the quality of outputs remains sufficient after quantization.***

## FP8 "Baseline" Performance

Benchmarking the engine produced by the example above yielded the following performance results. Note that we enabled some of the build flags we mentioned [earlier](./useful-build-time-flags.md) (multiple profiles, paged_context_fmha) and also tuned max batch size and max num tokens. This is done to give a sense of what performance is achievable if you tune an fp8 engine but exclude options that have been tailored for quantization. We recommend disabling the gemm plugin for quantized engines which is why it is not included here (it is off by default). Reduce fusion has a quantization specific optimization that will be covered later. For the remainder of this page we will refer to this setup as the "baseline" numbers for fp8.


| Metric                           | Value     |
| -------------------------------- | --------- |
| Token Throughput (tokens/sec)    | 3389.5305 |
| Request Throughput (req/sec)     | 1.6550    |
| Average Time To First Token (ms) | 96.1597   |
| Average Inter-Token Latency (ms) | 12.4248   |


## Quantized KV-Cache

By default the KV-Cache is not quantized but TensorRT-LLM supports quantizing the KV-Cache to further improve performance. However, quantizing the model more aggressively also increases the risk of model output quality degrading so it is important to check that when using this feature.

### Enabling Quantized KV Cache

The LLM-API exposes the quantization algorithm to be used for kv cache via the `kv_cache_quant_algo` field in `QuantConfig`. To enable fp8 kv cache, you would modify `QuantConfig` as such:

```python
quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                           kv_cache_quant_algo=QuantAlgo.FP8)
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--kv_cache_dtype fp8` to [`examples/quantization/quantize.py`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization).

### Performance with Quantized KV Cache

| Metric                           | Baseline  | FP8 KV-Cache ON |
| -------------------------------- | --------- | --------------- |
| Token Throughput (tokens/sec)    | 3389.5305 | 5299.6372       |
| Request Throughput (req/sec)     | 1.6550    | 2.5877          |
| Average Time To First Token (ms) | 96.1597   | 97.1287         |
| Average Inter-Token Latency (ms) | 12.4248   | 12.5496         |

## Reduce Norm Fusion with User Buffers for Llama Models

The [Reduce Norm Fusion](./useful-build-time-flags.md#reduce-norm-fusion-plugin-for-llama-models) feature is supported for fp8. An additional optimization called "User Buffers" is also supported for fp8 models. The user buffer feature aims to eliminate extra copies from the local buffer to the shared buffer in the communication kernel, leading to improved end-to-end performance.


### Enabling Reduce Norm Fusion with User Buffers


To enable reduce norm fusion with user buffers, add the following lines below `BuildConfig`'s initialization

```python
build_config.plugin_config.reduce_fusion = True
build_config.plugin_config.user_buffer = True
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--reduce_fusion enable` and `--user_buffer enable` to `trtllm-build` to enable the feature.

> Note: You must have enabled `reduce_fusion` in order to enable `user_buffer`

### Performance with Reduce Norm Fusion + User Buffers:

Reduce Norm Fusion + User Buffer ON: Same engine previously referred to as FP8 KV-Cache ON.

Reduce Norm Fusion + User Buffer ON: Previous example with reduce fusion and user buffers enabled. Max-num tokens set to 16384 and max-batch size set to 512 after tuning.


| Metric                           | Reduce Norm Fusion + User Buffer OFF | Reduce Norm Fusion + User Buffer ON |
| -------------------------------- | ------------------------------------ | ----------------------------------- |
| Token Throughput (tokens/sec)    | 5299.6372                            | 5980.7842                           |
| Request Throughput (req/sec)     | 2.5877                               | 2.9203                              |
| Average Time To First Token (ms) | 97.1287                              | 82.2679                             |
| Average Inter-Token Latency (ms) | 12.5496                              | 12.6975                             |

## GEMM + SwiGLU Fusion in Gated-MLP

The GEMM + SwiGLU fusion in Gated-MLP combines two Matmul operations and one SwiGLU operation into a single kernel. Currently this is only supported for FP8 precision on Hopper. While this fusion improves performance, it can slightly reduce accuracy in FP8 PTQ because one quantization scaling factor is discarded.

We recommend enabling this feature for large models running on Hopper with FP8 precision.We do not recommend enabling this feature for very small workloads or if the
accuracy loss is unacceptable.

### Enabling GEMM + SwiGLU Fusion

To enable the GEMM + SwiGLU fusion, add the following lines below `BuildConfig`'s initialization

```python
build_config.plugin_config.gemm_swiglu_plugin = 'fp8'
```
For small batch size cases where latency is important, you can replace the above line with

```python
build_config.plugin_config.low_latency_gemm_swiglu_plugin = 'fp8'
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--gemm_swiglu_plugin=fp8` or `--low_latency_gemm_swiglu_plugin=fp8` for the low latency case (only include one or the other) to `trtllm-build`.

### Performance with GEMM + SwiGLU Fusion


| Metric                           | GEMM + SwiGLU fusion OFF | GEMM + SwiGLU fusion ON |
| -------------------------------- | ------------------------ | ----------------------- |
| Token Throughput (tokens/sec)    | 5980.7842                | 5976.7977               |
| Request Throughput (req/sec)     | 2.9203                   | 2.9184                  |
| Average Time To First Token (ms) | 82.2679                  | 81.8841                 |
| Average Inter-Token Latency (ms) | 12.6975                  | 11.7031                 |

In this case, the GEMM + SwiGLU plugin performs almost equivalently to when it was disabled. The throughput drop is within run to run variance and the TTFT and ITL improvements are slight. However, we found that when paired with the low latency gemm plugin discussed next, enabling this feature was necessary for getting the maximum throughput.

## Low Latency GEMM Plugin

Previously we mentioned the [GEMM Plugin](./useful-build-time-flags.md#gemm-plugin) feature. Although it has fp8 support we recommend disabling it (by default it is disabled). However for low-latency scenarios in fp8 we recommend trying the low latency GEMM plugin to see if it is effective for your workload.

### Enabling Low Latency GEMM plugin

To enable the low latency GEMM plugin, add the following lines below `BuildConfig`'s initialization

```python
build_config.plugin_config.low_latency_gemm_plugin = 'fp8'
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--low_latency_gemm_plugin=fp8` to `trtllm-build` to enable the feature. Again, **we recommend disabling the gemm plugin for fp8** so if you are passing `--gemm_plugin=fp8` to `trtllm-build` we recommend removing that.

###  Performance with Low Latency GEMM plugin

Low Latency GEMM ON: Same configuration as previous example but with low latency GEMM plugin enabled. Max num tokens was set to 16384 and max-batch size was set to 512 after tuning.

| Metric                           | Low Latency GEMM OFF | Low Latency GEMM ON |
| -------------------------------- | -------------------- | ------------------- |
| Token Throughput (tokens/sec)    | 5976.7977            | 6049.1625           |
| Request Throughput (req/sec)     | 2.9184               | 2.9537              |
| Average Time To First Token (ms) | 81.8841              | 88.0162             |
| Average Inter-Token Latency (ms) | 11.7031              | 10.8225             |

In this case, enabling the low-latency gemm plugin actually provided a meaningful boost to throughput. Additionally it also improved ITL but at the expense of TTFT. Furthermore, when used without the gemm+swiglu fusion, performance was actually worse than with out the plugin turned on. This suggests that for this workload the low-latency gemm plugin was choosing a worse kernel for the gemm right before the swiglu, but once that was handled by the gemm+swiglu fusion custom kernel, the rest of the kernels the low-latency gemm plugin was choosing was better than the baseline, resulting in improved performance. This underscores the importance of benchmarking different settings as the impact of this plugin is highly workload dependent. If possible some grid searching can be useful for extremely performance sensitive workloads

## Conclusion

Overall leveraging quantization can provide significant uplifts in performance. Here are the performance uplifts from our tuned fp8 model as compared to the tuned fp16 numbers we reached in the [previous page of guide](./tuning-max-batch-size-and-max-num-tokens.md)

| Metric                           | Tuned FP16 Model | Tuned FP8 Model | % Improvement |
| -------------------------------- | ---------------- | --------------- | ------------- |
| Token Throughput (tokens/sec)    | 2474.2581        | 6049.1625       | 144.48        |
| Request Throughput (req/sec)     | 1.2081           | 2.9537          | 144.49        |
| Average Time To First Token (ms) | 147.5742         | 88.0162         | 40.36         |
| Average Inter-Token Latency (ms) | 14.6852          | 10.8225         | 26.30         |

Additionally, compared to the fp8 baseline numbers (the baseline numbers had some degree of tuning, see [Baseline Performance](./fp8-quantization.md#fp8-baseline-performance) for details), we received the following performance uplifts from enabling the flags discussed above:

| Metric                           | Baseline FP8 Model | Tuned FP8 Model | % Improvement |
| -------------------------------- | ------------------ | --------------- | ------------- |
| Token Throughput (tokens/sec)    | 3389.5305          | 6049.1625       | 78.47         |
| Request Throughput (req/sec)     | 1.6550             | 2.9537          | 78.47         |
| Average Time To First Token (ms) | 96.1597            | 88.0162         | 8.47          |
| Average Inter-Token Latency (ms) | 12.4248            | 10.8225         | 12.90         |

As mentioned previously, the caveat with leveraging quantization are potential drops in accuracy, and we strongly recommend having a way to test whether model output quality is acceptable before attempting to use quantization. That said, many real world cases successfully use quantization and the significant performance boosts it enables are often worth the effort to see if it is a fit.

### Summary of Configuration Option Recommendations:

1. Quantized KV-cache: Typically provides significant throughput boost. We recommend turning it on as long as output quality is still acceptable with the feature enabled.
2. Reduce fusion + user buffers: This feature is only supported on fp8 Llama and Mistral/Mixtral models. Effectiveness is workload dependent so we recommend turning it on and benchmarking to check.
3. Gemm + Swiglu Plugin: This feature is only supported on fp8 models with Swiglu operators like Llama, Mixtral etc. Like reduce fusion effectiveness is workload dependent and we recommend sanity checking effectiveness. Has increased risk of affecting accuracy since it drops a quantization scale.
4. Low-Latency GEMM plugin: Effectiveness is workload dependent so we recommend turning it on and benchmarking. Effectiveness can be affected by other flags as we saw in our case study, so if possible benchmarking various combinations of configuration options is ideal.
