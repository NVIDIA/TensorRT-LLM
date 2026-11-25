# Recommended LLM API Configuration Settings

The [curated recipes](curated/) contain recommended [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/) performance settings for popular models. They can be used out-of-the-box with `trtllm-serve` via the `--config` CLI flag, or you can adjust them to your specific use case.

The Core and Spark performance tests select shared benchmark settings from [model_presets.yml](model_presets.yml). See the [performance test configuration guide](../../tests/integration/defs/perf/README_release_test.md#41-pytorch-model-configuration) for `cfg:` syntax.

For model-specific deployment guides, please refer to the [official documentation](https://nvidia.github.io/TensorRT-LLM/deployment-guide/index.html).
