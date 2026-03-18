.. TensorRT LLM documentation master file, created by
   sphinx-quickstart on Wed Sep 20 08:35:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TensorRT LLM's Documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: Getting Started

   overview.md
   quick-start-guide.md
   installation/index.rst
   supported-hardware.md


.. toctree::
   :maxdepth: 2
   :caption: Deployment Guide
   :name: Deployment Guide

   examples/llm_api_examples.rst
   examples/trtllm_serve_examples
   examples/dynamo_k8s_example.rst
   deployment-guide/index.rst
   deployment-guide/configuring-cpu-affinity.md

.. toctree::
   :maxdepth: 2
   :caption: Models
   :name: Models

   models/supported-models.md
   models/visual-generation.md
   models/adding-new-model.md



.. toctree::
   :maxdepth: 2
   :caption: CLI Reference
   :name: CLI Reference

   commands/trtllm-bench
   commands/trtllm-eval
   commands/trtllm-serve/index


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   llm-api/index.md
   llm-api/reference.rst


.. toctree::
   :maxdepth: 2
   :caption: Features

   features/feature-combination-matrix.md
   features/attention.md
   features/disagg-serving.md
   features/kvcache.md
   features/long-sequence.md
   features/lora.md
   features/multi-modality.md
   features/overlap-scheduler.md
   features/paged-attention-ifb-scheduler.md
   features/parallel-strategy.md
   features/quantization.md
   features/sampling.md
   features/additional-outputs.md
   features/guided-decoding.md
   features/speculative-decoding.md
   features/checkpoint-loading.md
   features/auto_deploy/auto-deploy.md
   features/ray-orchestrator.md
   features/torch_compile_and_piecewise_cuda_graph.md
   features/helix.md
   features/kv-cache-connector.md
   features/sparse-attention.md


.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer-guide/overview.md
   developer-guide/perf-analysis.md
   developer-guide/perf-benchmarking.md
   developer-guide/ci-overview.md
   developer-guide/dev-containers.md
   developer-guide/api-change.md
   developer-guide/kv-transfer.md


.. toctree::
   :maxdepth: 2
   :caption: Blogs

   Blog18: Optimizing MoE Communication with One-Sided AlltoAll Over NVLink <blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md>
   Blog17: Sparse Attention in TensorRT LLM <blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md>
   Blog16: Accelerating Long-Context Inference with Skip Softmax Attention <blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md>
   Blog15: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs <blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md>
   Blog14: Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary) <blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md>
   Blog13: Inference Time Compute Implementation in TensorRT LLM <blogs/tech_blog/blog13_Inference_Time_Compute_Implementation_in_TensorRT-LLM.md>
   Blog12: Combining Guided Decoding and Speculative Decoding: Making CPU and GPU Cooperate Seamlessly <blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.md>
   Blog11: Running GPT-OSS-120B with Eagle3 Speculative Decoding on GB200/B200 (TensorRT LLM) <blogs/tech_blog/blog11_GPT_OSS_Eagle3.md>
   Blog10: ADP Balance Strategy <blogs/tech_blog/blog10_ADP_Balance_Strategy.md>
   Blog9: Running a High Performance GPT-OSS-120B Inference Server with TensorRT LLM <blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.md>
   Blog8: Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization) <blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md>
   Blog7: N-Gram Speculative Decoding in TensorRT LLM <blogs/tech_blog/blog7_NGram_performance_Analysis_And_Auto_Enablement.md>
   Blog6: How to launch Llama4 Maverick + Eagle3 TensorRT LLM server <blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.md>
   Blog5: Disaggregated Serving in TensorRT LLM <blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md>
   Blog4: Scaling Expert Parallelism in TensorRT LLM (Part 1: Design and Implementation of Large-scale EP) <blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md>
   Blog3: Optimizing DeepSeek R1 Throughput on NVIDIA Blackwell GPUs: A Deep Dive for Developers <blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md>
   Blog2: DeepSeek R1 MTP Implementation and Optimization <blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.md>
   Blog1: Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs <blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md>
   blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
   blogs/H200launch.md
   blogs/XQA-kernel.md
   blogs/H100vsA100.md


.. toctree::
   :maxdepth: 2
   :caption: Quick Links

   Releases <https://github.com/NVIDIA/TensorRT-LLM/releases>
   Github Code <https://github.com/NVIDIA/TensorRT-LLM>
   Roadmap <https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3Aroadmap>

.. toctree::
   :maxdepth: 2
   :caption: Use TensorRT Engine
   :hidden:

   legacy/tensorrt_quickstart.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
