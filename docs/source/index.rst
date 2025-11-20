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


.. toctree::
   :maxdepth: 2
   :caption: Deployment Guide
   :name: Deployment Guide

   examples/llm_api_examples.rst
   examples/trtllm_serve_examples
   examples/dynamo_k8s_example.rst
   deployment-guide/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Models
   :name: Models

   models/supported-models.md
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
   features/speculative-decoding.md
   features/checkpoint-loading.md
   features/auto_deploy/auto-deploy.md
   features/ray-orchestrator.md
   features/torch_compile_and_piecewise_cuda_graph.md

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
   :glob:

   blogs/tech_blog/*
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
