trtllm-serve
============

About
-----

The ``trtllm-serve`` command starts an OpenAI compatible server that supports the following endpoints:

- ``/v1/models``
- ``/v1/completions``
- ``/v1/chat/completions``

For information about the inference endpoints, refer to the `OpenAI API Reference <https://platform.openai.com/docs/api-reference>`__.

The server also supports the following endpoints:

- ``/health``
- ``/metrics``
- ``/version``

The ``metrics`` endpoint provides runtime-iteration statistics such as GPU memory use and inflight-batching details.

Starting a Server
-----------------

The following abbreviated command syntax shows the commonly used arguments to start a server:

.. code-block:: bash

   trtllm-serve <model> [--backend pytorch --tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

For the full syntax and argument descriptions, refer to :ref:`syntax`.

Inference Endpoints
-------------------

After you start the server, you can send inference requests through completions API and Chat API, which are compatible with corresponding OpenAI APIs. We use `TinyLlama-1.1B-Chat-v1.0 <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`_ for examples in the following sections.

Chat API
~~~~~~~~

You can query Chat API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../examples/serve/openai_chat_client.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../examples/serve/curl_chat_client.sh
    :language: bash
    :linenos:

Completions API
~~~~~~~~~~~~~~~

You can query Completions API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../examples/serve/openai_completion_client.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../examples/serve/curl_completion_client.sh
    :language: bash
    :linenos:

Multimodal Serving
~~~~~~~~~~~~~~~~~

For multimodal models (e.g., Qwen2-VL), you'll need to create a configuration file and start the server with additional options:

First, create a configuration file:

.. code-block:: bash

   cat >./extra-llm-api-config.yml<<EOF
   kv_cache_config:
       enable_block_reuse: false
   EOF

Then, start the server with the configuration file:

.. code-block:: bash

   trtllm-serve Qwen/Qwen2-VL-7B-Instruct \
       --extra_llm_api_options ./extra-llm-api-config.yml \
       --backend pytorch

Completions API
~~~~~~~~~~~~~~~

You can query Completions API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../examples/serve/openai_completion_client_for_multimodal.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../examples/serve/curl_completion_client_for_multimodal.sh
    :language: bash
    :linenos:

Benchmark
---------

You can use any benchmark clients compatible with OpenAI API to test serving performance of ``trtllm_serve``, we recommend ``genai-perf`` and here is a benchmarking recipe.

First, install ``genai-perf`` with ``pip``:

.. code-block:: bash

   pip install genai-perf

Then, :ref:`start a server<Starting a Server>` with ``trtllm-serve`` and ``TinyLlama-1.1B-Chat-v1.0``.

Finally, test performance with the following command:

.. literalinclude:: ../../../examples/serve/genai_perf_client.sh
    :language: bash
    :linenos:

Refer to `README <https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md>`_ of ``genai-perf`` for more guidance.

Multi-node Serving with Slurm
-----------------------------

You can deploy `DeepSeek-V3 <https://huggingface.co/deepseek-ai/DeepSeek-V3>`_ model across two nodes with Slurm and ``trtllm-serve``

.. code-block:: bash

    echo -e "enable_attention_dp: true\npytorch_backend_config:\n  enable_overlap_scheduler: true" > extra-llm-api-config.yml

    srun -N 2 -w [NODES] \
        --output=benchmark_2node.log \
        --ntasks 16 --ntasks-per-node=8 \
        --mpi=pmix --gres=gpu:8 \
        --container-image=<CONTAINER_IMG> \
        --container-mounts=/workspace:/workspace \
        --container-workdir /workspace \
        bash -c "trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --backend pytorch --max_batch_size 161 --max_num_tokens 1160 --tp_size 16 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"

See `the source code <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/trtllm-llmapi-launch>`_ of ``trtllm-llmapi-launch`` for more details.

Metrics Endpoint
----------------

.. note::

   This endpoint is beta maturity.

   The statistics for the PyTorch backend are beta and not as comprehensive as those for the TensorRT backend.

   Some fields, such as CPU memory usage, are not available for the PyTorch backend.

   Enabling ``enable_iter_perf_stats`` in the PyTorch backend can impact performance slightly, depending on the serving configuration.

The ``/metrics`` endpoint provides runtime-iteration statistics such as GPU memory use and inflight-batching details.
For the TensorRT backend, these statistics are enabled by default.
However, for the PyTorch backend, specified with the ``--backend pytorch`` argument, you must explicitly enable iteration statistics logging by setting the `enable_iter_perf_stats` field in a YAML configuration file as shown in the following example:

.. code-block:: yaml

   # extra-llm-api-config.yml
   pytorch_backend_config:
    enable_iter_perf_stats: true

Then start the server and specify the ``--extra_llm_api_options`` argument with the path to the YAML file as shown in the following example:

.. code-block:: bash

   trtllm-serve <model> \
     --extra_llm_api_options <path-to-extra-llm-api-config.yml> \
     [--backend pytorch --tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

After at least one inference request is sent to the server, you can fetch the runtime-iteration statistics by polling the `/metrics` endpoint:

.. code-block:: bash

   curl -X GET http://<host>:<port>/metrics

*Example Output*

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

Syntax
------

.. click:: tensorrt_llm.commands.serve:main
   :prog: trtllm-serve
   :nested: full
