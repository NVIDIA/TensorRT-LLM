trtllm-serve
============

About
-----

The ``trtllm-serve`` command launches an OpenAI-compatible inference server that provides the following capabilities:

API Endpoints
------------

OpenAI-Compatible Endpoints:
- ``/v1/models`` - List available models
- ``/v1/completions`` - Text completion API
- ``/v1/chat/completions`` - Chat completion API

Additional Endpoints:
- ``/health`` - Server health check
- ``/metrics`` - Runtime statistics and performance metrics
- ``/version`` - Server version information

For detailed API specifications, refer to the `OpenAI API Reference <https://platform.openai.com/docs/api-reference>`__.

Quick Start
----------

Basic server startup command:

.. code-block:: bash

   trtllm-serve <model> [--backend pytorch --tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

For complete command syntax and all available options, see :ref:`syntax`.

API Usage Examples
----------------

Chat API
~~~~~~~~

Using Python client:

.. literalinclude:: ../../../examples/serve/openai_chat_client.py
    :language: python
    :linenos:

Using curl:

.. literalinclude:: ../../../examples/serve/curl_chat_client.sh
    :language: bash
    :linenos:

Completions API
~~~~~~~~~~~~~~~

Using Python client:

.. literalinclude:: ../../../examples/serve/openai_completion_client.py
    :language: python
    :linenos:

Using curl:

.. literalinclude:: ../../../examples/serve/curl_completion_client.sh
    :language: bash
    :linenos:

Multimodal Model Support
----------------------

For multimodal models (e.g., Qwen2-VL), follow these steps:

1. Create a configuration file:

.. code-block:: bash

   cat >./extra-llm-api-config.yml<<EOF
   kv_cache_config:
       enable_block_reuse: false
       free_gpu_memory_fraction: 0.6
   EOF

2. Start the server with the configuration:

.. code-block:: bash

   trtllm-serve Qwen/Qwen2-VL-7B-Instruct \
       --extra_llm_api_options ./extra-llm-api-config.yml \
       --backend pytorch

Multimodal API Examples
~~~~~~~~~~~~~~~~~~~~~

Using Python client:

.. literalinclude:: ../../../examples/serve/openai_completion_client_for_multimodal.py
    :language: python
    :linenos:

Using curl:

.. literalinclude:: ../../../examples/serve/curl_chat_client_for_multimodal.sh
    :language: bash
    :linenos:

Performance Benchmarking
----------------------

We recommend using ``genai-perf`` for performance testing:

1. Install the benchmark tool:
.. code-block:: bash

   pip install genai-perf

2. Start the server (see :ref:`Quick Start`)

3. Run the benchmark:
.. literalinclude:: ../../../examples/serve/genai_perf_client.sh
    :language: bash
    :linenos:

For detailed benchmarking guidance, refer to the `genai-perf README <https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md>`_.

Multi-node Deployment with Slurm
------------------------------

Example deployment of DeepSeek-V3 across two nodes:

1. Create configuration:
.. code-block:: bash

    echo -e "enable_attention_dp: true\npytorch_backend_config:\n  enable_overlap_scheduler: true" > extra-llm-api-config.yml

2. Launch with Slurm:
.. code-block:: bash

    srun -N 2 -w [NODES] \
        --output=benchmark_2node.log \
        --ntasks 16 --ntasks-per-node=8 \
        --mpi=pmix --gres=gpu:8 \
        --container-image=<CONTAINER_IMG> \
        --container-mounts=/workspace:/workspace \
        --container-workdir /workspace \
        bash -c "trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --backend pytorch --max_batch_size 161 --max_num_tokens 1160 --tp_size 16 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"

For more details on multi-node deployment, see the `trtllm-llmapi-launch source code <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/trtllm-llmapi-launch>`_.

Metrics and Monitoring
--------------------

.. note::

   The metrics endpoint is currently in beta.

   PyTorch backend metrics are less comprehensive than TensorRT backend metrics.
   Some metrics (e.g., CPU memory usage) are not available for PyTorch backend.
   Enabling metrics collection may slightly impact performance.

To enable metrics collection for PyTorch backend:

1. Create a configuration file:
.. code-block:: yaml

   # extra-llm-api-config.yml
   pytorch_backend_config:
    enable_iter_perf_stats: true

2. Start server with configuration:
.. code-block:: bash

   trtllm-serve <model> \
     --extra_llm_api_options <path-to-extra-llm-api-config.yml> \
     [--backend pytorch --tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

3. Query metrics:
.. code-block:: bash

   curl -X GET http://<host>:<port>/metrics

Example metrics output:
.. code-block:: json

   [
       {
           "gpuMemUsage": 56401920000,
           "inflightBatchingStats": {
               ...
           },
           "iter": 1,
           "iterLatencyMS": 16.505143404006958,
           "kvCacheStats": {
               ...
           },
           "newActiveRequestsQueueLatencyMS": 0.0007503032684326172
       }
   ]

Command Reference
---------------

.. click:: tensorrt_llm.commands.serve:main
   :prog: trtllm-serve
   :nested: full
