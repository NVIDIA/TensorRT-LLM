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
   release-notes.md


.. toctree::
   :maxdepth: 2
   :caption: Installation
   :name: Installation

   .. installation/overview.md

   installation/linux.md
   installation/build-from-source-linux.md
   installation/windows.md
   installation/build-from-source-windows.md

.. toctree::
   :maxdepth: 2
   :caption: Architecture
   :name: Architecture

   architecture/overview.md
   architecture/core-concepts.md
   architecture/checkpoint.md
   architecture/workflow.md
   architecture/add-model.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced
   :name: Advanced

   advanced/gpt-attention.md
   advanced/gpt-runtime.md
   advanced/graph-rewriting.md
   advanced/batch-manager.md
   advanced/inference-request.md
   advanced/lora.md
   advanced/expert-parallelism.md

.. toctree::
   :maxdepth: 2
   :caption: Performance
   :name: Performance

   performance/perf-overview.md
   performance/perf-best-practices.md
   performance/perf-analysis.md


.. toctree::
   :maxdepth: 2
   :caption: Reference
   :name: Reference

   reference/troubleshooting.md
   reference/support-matrix.md

   .. reference/upgrading.md

   reference/precision.md
   reference/memory.md


.. toctree::
   :maxdepth: 2
   :caption: C++ API
   :hidden:

   _cpp_gen/executor.rst
   _cpp_gen/runtime.rst

.. toctree::
   :maxdepth: 2
   :caption: Blogs
   :hidden:

   blogs/H100vsA100.md
   blogs/H200launch.md
   blogs/Falcon180B-H200.md
   blogs/quantization-in-TRT-LLM.md
   blogs/XQA-kernel.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
