# TensorRT-LLM Benchmarks

## Overview

There are currently two workflows to benchmark TensorRT-LLM:
* [`trtllm-bench`](../docs/source/performance/perf-benchmarking.md)
  - `trtllm-bench` is native to TensorRT-LLM and is a Python benchmarker for reproducing and testing the performance of TensorRT-LLM.
  - _NOTE_: This benchmarking suite is a current work in progress and is prone to large changes.
* [C++ benchmarks](./cpp)
  - The recommended workflow that uses TensorRT-LLM C++ API and can take advantage of the latest features of TensorRT-LLM.
