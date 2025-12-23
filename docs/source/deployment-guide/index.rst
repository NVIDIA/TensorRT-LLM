Model Recipes
================

Quick Start for Popular Models
-------------------------------

The table below contains ``trtllm-serve`` commands that can be used to easily deploy popular models including DeepSeek-R1, gpt-oss, Llama 4, Qwen3, and more.

We maintain LLM API configuration files for these models containing recommended performance settings in two locations:

* **Curated Examples**: `examples/configs/curated <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs/curated>`_ - Hand-picked configurations for common scenarios.
* **Comprehensive Database**: `examples/configs/database <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs/database>`_ - A more comprehensive set of known-good configurations for various GPUs and traffic patterns.

The TensorRT LLM Docker container makes these config files available at ``/app/tensorrt_llm/examples/configs/curated`` and ``/app/tensorrt_llm/examples/configs/database`` respectively. You can reference them as needed:

.. code-block:: bash

   export TRTLLM_DIR="/app/tensorrt_llm" # path to the TensorRT LLM repo in your local environment

.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-quick-start-isl-osl
   :end-before: .. end-note-quick-start-isl-osl

This table is designed to provide a straightforward starting point; for detailed model-specific deployment guides, check out the guides below.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20 30

   * - Model Name
     - GPU
     - Inference Scenario
     - Config
     - Command
   * - `DeepSeek-R1 <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`_
     - H100, H200
     - Max Throughput
     - `deepseek-r1-throughput.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/deepseek-r1-throughput.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/curated/deepseek-r1-throughput.yaml``
   * - `DeepSeek-R1 <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`_
     - B200, GB200
     - Max Throughput
     - `deepseek-r1-deepgemm.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/deepseek-r1-deepgemm.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/curated/deepseek-r1-deepgemm.yaml``
   * - `DeepSeek-R1 (NVFP4) <https://huggingface.co/nvidia/DeepSeek-R1-FP4>`_
     - B200, GB200
     - Max Throughput
     - `deepseek-r1-throughput.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/deepseek-r1-throughput.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-FP4 --config ${TRTLLM_DIR}/examples/configs/curated/deepseek-r1-throughput.yaml``
   * - `DeepSeek-R1 (NVFP4) <https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2>`_
     - B200, GB200
     - Min Latency
     - `deepseek-r1-latency.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/deepseek-r1-latency.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/curated/deepseek-r1-latency.yaml``
   * - `gpt-oss-120b <https://huggingface.co/openai/gpt-oss-120b>`_
     - Any
     - Max Throughput
     - `gpt-oss-120b-throughput.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/gpt-oss-120b-throughput.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/curated/gpt-oss-120b-throughput.yaml``
   * - `gpt-oss-120b <https://huggingface.co/openai/gpt-oss-120b>`_
     - Any
     - Min Latency
     - `gpt-oss-120b-latency.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/gpt-oss-120b-latency.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/curated/gpt-oss-120b-latency.yaml``
   * - `Qwen3-Next-80B-A3B-Thinking <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking>`_
     - Any
     - Max Throughput
     - `qwen3-next.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/qwen3-next.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-Next-80B-A3B-Thinking --config ${TRTLLM_DIR}/examples/configs/curated/qwen3-next.yaml``
   * - Qwen3 family (e.g. `Qwen3-30B-A3B <https://huggingface.co/Qwen/Qwen3-30B-A3B>`_)
     - Any
     - Max Throughput
     - `qwen3.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/qwen3.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-30B-A3B --config ${TRTLLM_DIR}/examples/configs/curated/qwen3.yaml`` (swap to another Qwen3 model name as needed)
   * - `Llama-3.3-70B (FP8) <https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8>`_
     - Any
     - Max Throughput
     - `llama-3.3-70b.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/llama-3.3-70b.yaml>`_
     - ``trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP8 --config ${TRTLLM_DIR}/examples/configs/curated/llama-3.3-70b.yaml``
   * - `Llama 4 Scout (FP8) <https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8>`_
     - Any
     - Max Throughput
     - `llama-4-scout.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/llama-4-scout.yaml>`_
     - ``trtllm-serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --config ${TRTLLM_DIR}/examples/configs/curated/llama-4-scout.yaml``

Model-Specific Deployment Guides
---------------------------------

The deployment guides below provide more detailed instructions for serving specific models with TensorRT LLM.

.. toctree::
   :maxdepth: 1
   :name: Deployment Guides

   deployment-guide-for-deepseek-r1-on-trtllm.md
   deployment-guide-for-llama3.3-70b-on-trtllm.md
   deployment-guide-for-llama4-scout-on-trtllm.md
   deployment-guide-for-gpt-oss-on-trtllm.md
   deployment-guide-for-qwen3-on-trtllm.md
   deployment-guide-for-qwen3-next-on-trtllm.md
   deployment-guide-for-kimi-k2-thinking-on-trtllm.md

Comprehensive Configuration Database
------------------------------------

The table below lists all available pre-configured model scenarios in the TensorRT LLM configuration database. Each row represents a specific model, GPU, and performance profile combination with recommended request settings.

.. include:: config_table.rst
