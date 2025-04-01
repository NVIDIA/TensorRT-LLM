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

After you start the server, you can send inference requests through completions API and Chat API, which are compatible with corresponding OpenAI APIs.

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
