(useful-build-time-flags)=

# Useful Build-Time Flags

This page presents several build-time flags, set via the LLM-API's `BuildConfig` class that you can enable to improve upon the baseline performance. Build-time refers to the fact that these flags affect how the TensorRT-LLM engine is built and cannot be changed without rebuilding the engine. For each flag there is an explanation of what it does, a description of how to enable it, and then an example of running it through the benchmarking flow described in [Benchmarking Default Performance](./benchmarking-default-performance.md) to showcase its impact on performance. All options compatible with `trtllm-build` can be found in the Command Line Reference section of the docs.

> Disclaimer: While performance numbers shown here are real, they are only for demonstration purposes. Differences in environment, SKU, interconnect, and workload can all significantly affect performance and lead to your results differing from what is shown here.

## Multiple Profiles

TensorRT-LLM is built on TensorRT, which handles engine building through "optimization profiles" defining min, optimal, and max input tensor shapes. TensorRT optimizes for the optimal shape while supporting the range between min and max.

TensorRT-LLM abstracts away the need to create optimization profiles although flags like max_batch_size and max_num_tokens (covered later) influence how they are created. By default, only one profile is created.

During inference serving, varying request loads can pose different tensor shapes to the engine. TensorRT addresses this by allowing multiple profiles, which TensorRT-LLM supports via the BuildConfig option in the LLM-API. Enabling multiple profiles increases build times but has no performance downsides, so it is recommended for production builds.

The only thing to watch out for is that enabling this can lead to slightly different outputs when the same prompt is run multiple times as different profiles and consequently kernels might be used depending on the request load. However this variance should not affect output quality so it is safe to enable this flag as long as you don't need completely deterministic outputs.

### Enabling building with multiple profiles

Below is an example of how you can modify the baseline example to enable multiple profiles.

```python
from tensorrt_llm import LLM, BuildConfig

def main():
    build_config = BuildConfig()
    build_config.plugin_config.multiple_profiles = True

    llm = LLM(
        model="/scratch/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4,
        build_config=build_config
    )

    llm.save("build_flags_multiple_profiles")

if __name__ == '__main__':
    main()
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--multiple_profiles` to `trtllm-build` to enable the feature.


### Performance with multiple profiles

Baseline refers to the engine that was benchmarked in the previous Benchmarking Default Performance page.

| Metric                           | Baseline  | Multiple Profiles ON |
| -------------------------------- | --------- | -------------------- |
| Token Throughput (tokens/sec)    | 1564.3040 | 1861.0881            |
| Request Throughput (req/sec)     | 0.7638    | 0.9087               |
| Average Time To First Token (ms) | 147.6976  | 145.8958             |
| Average Inter-Token Latency (ms) | 31.3276   | 19.6452              |

As you can see, enabling multiple profiles significantly improves the metrics across the board.

## Paged Context Attention

By default all the tokens of the prompt of a new request are processed in one iteration as the context phase. Enabling paged context attention allows TensorRT-LLM to break the context phase into chunks and handle the prompt over several iterations. This is particularly useful for workloads with large input length. In the worst case, this feature can provide a small performance hit in benchmarking runs (<2%) so it can be safely enabled. This feature is discussed further in the [next page](./tuning-max-batch-size-and-max-num-tokens.md#revisiting-paged-context-attention-and-context-chunking) of the guide.


### Enabling Paged Context Attention

Add the following line to our multiple profiles example from above to enable paged context attention

```python
build_config.plugin_config.use_paged_context_fmha=True
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--use_paged_context_fmha` to `trtllm-build` to enable the feature.

### Performance

Paged Context OFF refers to the same engine shown as Multiple Profiles ON in the previous example.

| Metric                           | Paged Context OFF | Paged Context ON |
| -------------------------------- | ----------------- | ---------------- |
| Token Throughput (tokens/sec)    | 1861.0881         | 1866.6684        |
| Request Throughput (req/sec)     | 0.9087            | 0.9115           |
| Average Time To First Token (ms) | 145.8958          | 145.4089         |
| Average Inter-Token Latency (ms) | 19.6452           | 19.6523          |

In this case enabling paged context attention provides a small boost to performance, but a rerun of our tests found this to be within run to run variance of around 10 tok/s for token throughput and 2ms for average time to first token (ITL was stable with <1ms and request throughput corresponded directly to token throughput). In other cases naively enabling it might actually provide a small hit to performance. However, further guidance on how to reason about this flag and why we recommend enabling it is discussed in the [next page](./tuning-max-batch-size-and-max-num-tokens.md#revisiting-paged-context-attention-and-context-chunking) as it is closely intertwined with how TensorRT-LLM schedules requests as well as the max-num tokens flag.

## GEMM Plugin

TensorRT allows you to add "plugins" or custom kernels that can be used instead of the kernels that TensorRT selects for particular operations. TensorRT-LLM has a host of custom plugins that are specifically tailored to speed up supported modules. The GEMM plugin utilizes NVIDIA cuBLASLt and some custom kernels to perform GEMM operations. On FP16 and BF16, it’s recommended to be enabled for better performance and smaller GPU memory usage. On FP8, it’s recommended to be disabled.

### Enabling GEMM Plugin

Add the following line to the multiple profiles example from above to enable paged context attention.

```python
build_config.plugin_config.gemm_plugin = 'auto'
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--gemm_plugin auto` to `trtllm-build` to enable the feature. `'auto'` tells the GEMM plugin to have the same type as the model (fp16, bf16, etc). It is fine to leave it on auto unless you are trying to do mixed precision.

### Performance with GEMM Plugin

GEMM Plugin OFF refers to the same engine shown as Paged Context ON in the previous example.

| Metric                           | GEMM Plugin OFF | GEMM Plugin ON |
| -------------------------------- | --------------- | -------------- |
| Token Throughput (tokens/sec)    | 1866.6684       | 2033.2640      |
| Request Throughput (req/sec)     | 0.9115          | 0.9928        |
| Average Time To First Token (ms) | 145.4089        | 147.8307       |
| Average Inter-Token Latency (ms) | 19.6523         | 15.4133        |

In this case the GEMM plugin greatly improves throughput as well as ITL, with a slight hit to TTFT.

## Reduce Norm Fusion Plugin for Llama models:

TensorRT-LLM has custom kernels for AllReduce operations that are enabled by default. This feature extends this functionality by fusing the ResidualAdd and LayerNorm kernels that run after AllReduce into the AllReduce kernel, resulting in a single kernel that handles those operations and improves end-to-end performance. This feature is currently only available for Llama models. It is most beneficial in workloads that are generation-phase heavy. For extremely context-phase heavy workloads its worth checking performance with and without this. Additionally, since this is an optimization for AllReduce, it is only beneficial for cases with tensor-parallelism. For scenarios only using pipeline parallelism this should stay disabled since pipeline parallelism doesn't require any AllReduce operations.

### Enabling Reduce Norm Fusion Plugin

Add the following line to the multiple profiles example from above to enable paged context attention.

```python
build_config.plugin_config.reduce_fusion = True
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) pass `--reduce_fusion enable` to `trtllm-build` to enable the feature.

### Performance with Reduce Norm Fusion

Reduce Fusion OFF refers to the same engine shown as GEMM Plugin ON in the previous example.

| Metric                           | REDUCE FUSION OFF | REDUCE FUSION ON |
| -------------------------------- | ----------------- | ---------------- |
| Token Throughput (tokens/sec)    | 2033.2640        | 2044.2628        |
| Request Throughput (req/sec)     | 0.9928            | 0.9982           |
| Average Time To First Token (ms) | 147.8307          | 146.6628         |
| Average Inter-Token Latency (ms) | 15.4133           | 14.4493          |

For the ISL/OSL pair of 2048/2048 enabling the reduce norm fusion plugin slightly improves performance all around. However, test reruns found that with run to run variance, in the worst case, they performed at par. Again this flag's effectiveness is dependent on the workload so users should check whether it provides meaningful performance boosts in their case.


## Pipeline Parallel Reduce Scatter Optimization

This feature adds a pipeline parallelism optimization with ReduceScatter + AllGather targeting large mixture of experts models.
This can be enabled via the LLM-API as such
```python
    build_config.plugin_config.pp_reduce_scatter = True
```

If you are using the [CLI flow for building engines](./benchmarking-default-performance.md#building-and-saving-engines-via-cli) flow you can enable this feature by adding `--pp_reduce_scatter` to `trtllm-build`.

As the Llama model is not a MoE model this flag was not included as part of the case study.

## Conclusion

Overall, enabling these flags can greatly boost performance. However, the degree to which they are effective can vary from workload to workload, and it's recommended that you run sanity checks on your workloads to verify performance.

The case-study example showed that enabling these flags provided the following performance uplifts from the baseline numbers. This included significant boosts in Token Throughput, Request Throughput, and Average Inter-Token Latency. TTFT remained largely unchanged.

| Metric                           | Baseline  | Build-Time Flags ON | % Improvement |
| -------------------------------- | --------- | ------------------- | ------------- |
| Token Throughput (tokens/sec)    | 1564.3040 | 2044.2628           | 30.68         |
| Request Throughput (req/sec)     | 0.7638    | 0.9982              | 30.69         |
| Average Time To First Token (ms) | 147.6976  | 146.6628            | 0.70          |
| Average Inter-Token Latency (ms) | 31.3276   | 14.4493             | 53.88         |

### Summary of Configuration Option Recommendations:

1. Multiple profiles: Always enable. It may increase build times a little but will only ever help performance. Enabling might cause engine to produce slightly different outputs when the same prompt is run multiple times depending on request load but it should not affect output quality, see [Multiple Profiles section](./useful-build-time-flags.md#multiple-profiles) for explanation.
2. Paged Context Attention: In the worst case it may hurt performance a little initially but typically helps with request scheduling and boosts performance after further tuning of max batch size and max num tokens. More on this topic is discussed in the next page.
3. GEMM Plugin: It's recommended to enable it for FP16 and BF16 models as it usually helps. However, it is a good idea to benchmark your workload and double check that it is helping.
4. Reduce Fusion: This feature is only supported on Llama and Mistral/Mixtral models. Effectiveness is workload dependent and it's recommend that you benchmark your workload with and without it and compare the results.
