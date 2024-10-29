# TensorRT-LLM Benchmarks

## Overview

There are currently three workflows to benchmark TensorRT-LLM:
* [C++ benchmarks](./cpp)
  - The recommended workflow that uses TensorRT-LLM C++ API and can take advantage of the latest features of TensorRT-LLM.
* [Python benchmarks](./python)
  - The Python benchmarking scripts can only benchmark the Python runtime, which do not support the latest features, such as in-flight batching.
* [The Python benchmarking suite](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html)
  - This benchmarker is native to TensorRT-LLM and is a Python benchmarker for reproducing and testing the performance of TensorRT-LLM.
  - _NOTE_: This benchmarking suite is a current work in progress and is prone to large changes.
