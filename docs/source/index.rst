.. TensorRT-LLM documentation master file, created by
   sphinx-quickstart on Wed Sep 20 08:35:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TensorRT-LLM's Documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: Getting Started

   overview.md
   quick-start-guide.md
   installation/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

.. toctree::
   :maxdepth: 2
   :caption: CLI Reference
   :hidden:
   commands/trtllm-serve

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   llm-api/*
   examples/index.rst
   examples/customization.md
   examples/llm_api_examples
   examples/trtllm_serve_examples



.. toctree::
   :maxdepth: 2
   :caption: Software Architecture
   :name: Software Architecture

   architecture/overview.md
   architecture/core-concepts.md
   architecture/checkpoint.md
   architecture/workflow.md
   architecture/add-model.md

.. toctree::
   :maxdepth: 2
   :caption: Features
   :name: Features

   advanced/gpt-attention.md
   advanced/gpt-runtime.md
   advanced/executor.md
   advanced/graph-rewriting.md
   advanced/inference-request.md
   advanced/lora.md
   advanced/expert-parallelism.md
   advanced/kv-cache-management.md
   advanced/kv-cache-reuse.md
   advanced/speculative-decoding.md
   advanced/disaggregated-service.md


.. toctree::
   :maxdepth: 2
   :caption: Links
   :hidden:

   blogs/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
