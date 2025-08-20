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

   trtllm-serve <model> [--tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

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
~~~~~~~~~~~~~~~~~~

For multimodal models, you need to create a configuration file and start the server with additional options due to the following limitations:

* TRT-LLM multimodal is currently not compatible with ``kv_cache_reuse``
* Multimodal models require ``chat_template``, so only the Chat API is supported

To set up multimodal models:

First, create a configuration file:

.. code-block:: bash

   cat >./extra-llm-api-config.yml<<EOF
   kv_cache_config:
       enable_block_reuse: false
   EOF

Then, start the server with the configuration file:

.. code-block:: bash

   trtllm-serve Qwen/Qwen2-VL-7B-Instruct \
       --extra_llm_api_options ./extra-llm-api-config.yml

Multimodal Chat API
~~~~~~~~~~~~~~~~~~~

You can query Completions API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../examples/serve/openai_completion_client_for_multimodal.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../examples/serve/curl_chat_client_for_multimodal.sh
    :language: bash
    :linenos:

Multimodal Modality Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TRT-LLM multimodal supports the following modalities and data types (depending on the model):

**Text**

* No type specified:

  .. code-block:: json

     {"role": "user", "content": "What's the capital of South Korea?"}

* Explicit "text" type:

  .. code-block:: json

     {"role": "user", "content": [{"type": "text", "text": "What's the capital of South Korea?"}]}

**Image**

* Using "image_url" with URL:

  .. code-block:: json

     {"role": "user", "content": [
         {"type": "text", "text": "What's in this image?"},
         {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
     ]}

* Using "image_url" with base64-encoded data:

  .. code-block:: json

     {"role": "user", "content": [
         {"type": "text", "text": "What's in this image?"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
     ]}

.. note::
   To convert images to base64-encoded format, use the utility function
   :func:`tensorrt_llm.utils.load_base64_image`. Refer to the
   `load_base64_image utility <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/utils/load_base64_image.py>`__
   for implementation details.

**Video**

* Using "video_url":

  .. code-block:: json

     {"role": "user", "content": [
         {"type": "text", "text": "What's in this video?"},
         {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
     ]}

**Audio**

* Using "audio_url":

  .. code-block:: json

     {"role": "user", "content": [
         {"type": "text", "text": "What's in this audio?"},
         {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
     ]}



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
        bash -c "trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --max_batch_size 161 --max_num_tokens 1160 --tp_size 16 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"

See `the source code <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/trtllm-llmapi-launch>`_ of ``trtllm-llmapi-launch`` for more details.

Metrics Endpoint
----------------

.. note::

   The metrics endpoint for the default PyTorch backend are in beta and are not as comprehensive as those for the TensorRT backend.

   Some fields, such as CPU memory usage, are not yet available for the PyTorch backend.

   Enabling ``enable_iter_perf_stats`` in the PyTorch backend can slightly impact performance, depending on the serving configuration.

The ``/metrics`` endpoint provides runtime iteration statistics such as GPU memory usage and KV cache details.

For the default PyTorch backend, iteration statistics logging is enabled by setting the ``enable_iter_perf_stats`` field in a YAML file:

.. code-block:: yaml

   # extra_llm_config.yaml
   enable_iter_perf_stats: true

Start the server and specify the ``--extra_llm_api_options`` argument with the path to the YAML file:

.. code-block:: bash

   trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --extra_llm_api_options extra_llm_config.yaml

After sending at least one inference request to the server, you can fetch runtime iteration statistics by polling the ``/metrics`` endpoint.
Since the statistics are stored in an internal queue and removed once retrieved, it's recommended to poll the endpoint shortly after each request and store the results if needed.

.. code-block:: bash

   curl -X GET http://localhost:8000/metrics

Example output:

.. code-block:: json

    [
        {
            "gpuMemUsage": 76665782272,
            "iter": 154,
            "iterLatencyMS": 7.00688362121582,
            "kvCacheStats": {
                "allocNewBlocks": 3126,
                "allocTotalBlocks": 3126,
                "cacheHitRate": 0.00128,
                "freeNumBlocks": 101253,
                "maxNumBlocks": 101256,
                "missedBlocks": 3121,
                "reusedBlocks": 4,
                "tokensPerBlock": 32,
                "usedNumBlocks": 3
            },
            "numActiveRequests": 1
            ...
        }
    ]



Syntax
------

.. click:: tensorrt_llm.commands.serve:main
   :prog: trtllm-serve
   :nested: full

Besides the above examples, `trtllm-serve` is also used as an entrypoint for performance benchmarking.
Please refer to `Performance Benchmarking with `trtllm-serve` <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-serve/trtllm-serve-bench.md>` for more details.
