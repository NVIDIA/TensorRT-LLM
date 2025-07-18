# Micro Benchmarks

This folder contains benchmarks for specific components in TRT-LLM,
using [google-benchmark](https://github.com/google/benchmark/tree/main)

## Building

To build add the `--micro_benchmark` flag to `build_wheel.py` or pass `-DBUILD_MICRO_BENCHMARKS=ON` to cmake

## Benchmark Documentations

### Mixture Of Experts Backend Benchmark

> [!CAUTION]
> Disclaimer this benchmark is intended for developers to help evaluating the impact of new optimisations. This benchmark does not meet the same quality standards as other parts of TRT-LLM. Please use with caution

Target `mixtureOfExpertsBackendBenchmark`

This benchmark covers the backend used by the `MixtureOfExperts` plugin. It allows you to benchmark different MOE
configurations without building a TRT engine.

Usage:

```bash
./mixtureOfExpertsBackendBenchmark

# or

./mixtureOfExpertsBackendBenchmark --input_file <JSON benchmark definition>
```

For more information see:

```
./mixtureOfExpertsBackendBenchmark --help
```

The `gen-moe-workload-file.py` is a helper script that can generate workload files for MOE benchmarks. This is useful
for sharing or comparing configurations, such as when generating a reproduction case for a performance bug
