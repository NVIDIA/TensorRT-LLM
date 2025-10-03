While defaults are expected to provide solid performance, TensorRT-LLM has several configurable options that can improve performance for your particular workload. This guide is meant to help you tune TensorRT-LLM to extract the best performance for your use case. It covers several of the most helpful tunable parameters and provides intuition for thinking about them. This guide also doubles as an example of how to work with TensorRT-LLM's LLM-API and its TRTLLM-Bench benchmarking workflow.

This guide uses Llama-3.3-70b on 4 H100-sxm-80GB connected via NVLink as a case study and focuses on optimizing performance on input sequence length/output sequence length of 2048/2048. Case study sections throughout this guide reference internal performance testing and results to help reinforce the conclusions and recommendations given.

## Prerequisite Knowledge

This guide expects you have some familiarity with the following concepts

- Phases of Inference: Context (Prefill) Phase and Generation Phase
- Inflight Batching
- Tensor Parallelism and Pipeline Parallelism
- Quantization

 Please refer to [Mastering LLM Techniques - Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) for an introduction to these concepts.

## Table of Contents
<!--Actual table of contents with links is not written here because sphinx autogenerates it.-->
