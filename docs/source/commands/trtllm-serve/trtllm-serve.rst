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
- ``/start_profile`` (prototype)
- ``/stop_profile`` (prototype)

The ``metrics`` endpoint provides runtime-iteration statistics such as GPU memory use and inflight-batching details. The ``start_profile`` and ``stop_profile`` endpoints control iteration-scoped profiling of the backend engine; see :ref:`runtime-profiling-endpoints` below.

Starting a Server
-----------------

The following abbreviated command syntax shows the commonly used arguments to start a server:

.. code-block:: bash

   trtllm-serve <model> [--tp_size <tp> --pp_size <pp> --ep_size <ep> --host <host> --port <port>]

For the full syntax and argument descriptions, refer to :ref:`syntax`.

Inference Endpoints
-------------------

After you start the server, you can send inference requests through completions API, Chat API and Responses API, which are compatible with corresponding OpenAI APIs. We use `TinyLlama-1.1B-Chat-v1.0 <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`_ for examples in the following sections.

Chat API
~~~~~~~~

You can query Chat API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../../examples/serve/openai_chat_client.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../../examples/serve/curl_chat_client.sh
    :language: bash
    :linenos:

Completions API
~~~~~~~~~~~~~~~

You can query Completions API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../../examples/serve/openai_completion_client.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../../examples/serve/curl_completion_client.sh
    :language: bash
    :linenos:

Responses API
~~~~~~~~~~~~~~~

You can query Responses API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../../examples/serve/openai_responses_client.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../../examples/serve/curl_responses_client.sh
    :language: bash
    :linenos:


More openai compatible examples can be found in the `compatibility examples <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/serve/compatibility>`_ directory.

Multimodal Serving
~~~~~~~~~~~~~~~~~~

For multimodal models, you need to create a configuration file and start the server with additional options due to the following limitations:

* TRT-LLM multimodal is currently not compatible with ``kv_cache_reuse``
* Multimodal models require ``chat_template``, so only the Chat API is supported

To set up multimodal models:

First, create a configuration file:

.. code-block:: bash

   cat >./config.yml<<EOF
   kv_cache_config:
       enable_block_reuse: false
   EOF

Then, start the server with the configuration file:

.. code-block:: bash

   trtllm-serve Qwen/Qwen2-VL-7B-Instruct \
       --config ./config.yml

Multimodal Chat API
~~~~~~~~~~~~~~~~~~~

You can query Completions API with any http clients, a typical example is OpenAI Python client:

.. literalinclude:: ../../../../examples/serve/openai_completion_client_for_multimodal.py
    :language: python
    :linenos:

Another example uses ``curl``:

.. literalinclude:: ../../../../examples/serve/curl_chat_client_for_multimodal.sh
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

**Image embeddings**

It is also possible to directly provide the image embeddings to use by the multimodal
model.

* Using "image_embeds" with base64-encoded data:

  .. code-block:: json

     {"role": "user", "content": [
         {"type": "text", "text": "What's in this image?"},
         {"type": "image_embeds", "image_embeds": {"data": "{image_embeddings_base64}"}}}
     ]}

.. note::
   The contents of `image_embeddings_base64` can be generated by base64-encoding
   the result of serializing a tensor via `torch.save`.

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



Visual Generation Serving
~~~~~~~~~~~~~~~~~~~~~~~~~

``trtllm-serve`` supports diffusion-based visual generation models (FLUX.1, FLUX.2, Wan2.1, Wan2.2) for image and video generation. When a diffusion model directory is provided (detected by the presence of ``model_index.json``), the server automatically launches in visual generation mode with dedicated endpoints.

.. note::
   VisualGen is in **beta** stage. APIs, supported models, and optimization options are actively evolving and may change in future releases.

.. code-block:: bash

   # Video generation (Wan)
   trtllm-serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
       --visual_gen_args config.yml

   # Image generation (FLUX)
   trtllm-serve black-forest-labs/FLUX.2-dev \
       --visual_gen_args config.yml

The ``--visual_gen_args`` flag accepts a YAML file that configures quantization, parallelism, and TeaCache. Available visual generation endpoints include ``/v1/images/generations``, ``/v1/videos``, ``/v1/videos/generations``, and video management APIs.

For full details, see the :doc:`../../models/visual-generation.md` feature documentation. Example client scripts are available in the `examples/visual_gen/serve/ <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/serve>`_ directory.

Multi-node Serving with Slurm
-----------------------------

You can deploy `DeepSeek-V3 <https://huggingface.co/deepseek-ai/DeepSeek-V3>`_ model across two nodes with Slurm and ``trtllm-serve``

.. code-block:: bash

    echo -e "enable_attention_dp: true\npytorch_backend_config:\n  enable_overlap_scheduler: true" > config.yml

    srun -N 2 -w [NODES] \
        --output=benchmark_2node.log \
        --ntasks 16 --ntasks-per-node=8 \
        --mpi=pmix --gres=gpu:8 \
        --container-image=<CONTAINER_IMG> \
        --container-mounts=/workspace:/workspace \
        --container-workdir /workspace \
        bash -c "trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --max_batch_size 161 --max_num_tokens 1160 --tp_size 16 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.95 --config ./config.yml"

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

Start the server and specify the ``--config`` argument with the path to the YAML file:

.. code-block:: bash

   trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --config config.yaml

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

.. _runtime-profiling-endpoints:

Runtime Profiling Endpoints
---------------------------

.. note::

   The ``/start_profile`` and ``/stop_profile`` endpoints are **prototype** and only apply to the default PyTorch backend (``PyExecutor``). They are no-ops on the TensorRT backend. APIs, parameters, and behaviour may change in future releases.

The server exposes two HTTP endpoints that control iteration-scoped profiling of the backend engine at runtime, without restarting ``trtllm-serve`` and without setting ``TLLM_PROFILE_START_STOP`` / ``TLLM_TORCH_PROFILE_TRACE``. Under TP/PP > 1, both endpoints broadcast the window to every rank, so each rank writes its own chrome trace (distinguished by a ``rank-<N>`` suffix in the filename) and all ranks profile the same iterations in lockstep.

POST /start_profile
~~~~~~~~~~~~~~~~~~~

Start a profile window. All fields in the JSON body are optional.

.. list-table::
   :header-rows: 1
   :widths: 18 15 50

   * - Field
     - Type
     - Description
   * - ``output_dir``
     - string
     - Directory where chrome traces are written. Defaults to ``$TLLM_TORCH_PROFILER_DIR`` if set, otherwise ``/tmp``. One trace file per rank is written as ``trtllm-trace-<profile_id>-rank-<N>.json``; ``<profile_id>`` is a timestamp + short uuid so successive cycles don't collide.
   * - ``num_steps``
     - int
     - Number of engine iterations to capture. When set, the engine stops the window automatically and you do not need to call ``/stop_profile``. When omitted, profiling runs until ``/stop_profile`` is called.
   * - ``start_step``
     - int (default ``0``)
     - Iterations to skip after the call before profiling starts. Use to exclude warmup-like transients from the capture.
   * - ``activities``
     - list of string
     - Subset of ``["CPU", "GPU", "CUDA_PROFILER"]`` (default ``["CPU", "GPU"]``). When ``["CUDA_PROFILER"]`` is specified, ``torch.profiler`` is not started and only ``cudaProfilerStart`` / ``cudaProfilerStop`` are called — this composes cleanly with ``nsys profile -c cudaProfilerApi``.

Response: ``200`` with ``{"message": "Profiling started"}`` on success. ``409`` with ``{"success": false, "message": ...}`` if a profile window is already active or pending (the backend does not silently queue a second window).

POST /stop_profile
~~~~~~~~~~~~~~~~~~

Stop an in-progress profile window and flush the chrome trace to disk. Takes no body. The call blocks until ``export_chrome_trace()`` has run on the backend thread, so by the time the handler returns the file is on disk. Call this only if ``/start_profile`` was invoked without ``num_steps``; otherwise the window self-terminates.

Example
~~~~~~~

.. code-block:: bash

   # Start a 30-iteration profile window into /tmp/traces.
   curl -s -X POST http://127.0.0.1:8000/start_profile \
       -H "Content-Type: application/json" \
       -d '{"output_dir": "/tmp/traces", "num_steps": 30}'

   # ... drive load via /v1/completions or /v1/chat/completions ...

   # If you started the window without num_steps, stop it:
   curl -s -X POST http://127.0.0.1:8000/stop_profile

   # Inspect the resulting chrome traces.
   ls /tmp/traces/
   # trtllm-trace-1778480665-7993d824-rank-0.json
   # trtllm-trace-1778480665-7993d824-rank-1.json   (TP=2)

Open the trace files in `ui.perfetto.dev <https://ui.perfetto.dev/>`_ or ``chrome://tracing``. Each captured iteration is wrapped in an SGLang-compatible ``step[EXTEND bs=N toks=M]`` or ``step[DECODE bs=N]`` user-annotation scope so traces from ``trtllm-serve`` and SGLang can be compared side-by-side.

For a full benchmarking + profiling workflow (including multi-cycle capture under steady-state load), see :doc:`run-benchmark-with-trtllm-serve`.

.. _configuring-with-yaml-files:

Configuring with YAML Files
----------------------------

You can configure various options of ``trtllm-serve`` using YAML files by setting the ``--config`` option to the path of a YAML file. Explicit CLI flags take precedence over values in the YAML; un-set CLI flags fall back to the YAML.

.. include:: ../../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias

The yaml file is configuration of `tensorrt_llm.llmapi.LlmArgs <https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs>`_, the class has multiple levels of hierarchy, to configure the top level arguments like ``max_batch_size``, the yaml file should be like:

.. code-block:: yaml

   max_batch_size: 8

To configure the nested level arguments like ``moe_config.backend``, the yaml file should be like:

.. code-block:: yaml

   moe_config:
       backend: CUTLASS

Syntax
------

This syntax section lists all command line arguments for ``trtllm-serve``'s subcommands. Some of the arguments are accompanied with a stability tag indicating their development status. Refer to our `API Reference <https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html>`__ for details

.. click:: tensorrt_llm.commands.serve:main
   :prog: trtllm-serve
   :nested: full

Besides the above examples, `trtllm-serve` is also used as an entrypoint for performance benchmarking.
Please refer to `Performance Benchmarking with `trtllm-serve` <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-serve/trtllm-serve-bench.md>` for more details.
