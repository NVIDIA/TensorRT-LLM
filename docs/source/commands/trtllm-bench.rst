trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT-LLM engines. It provides three main subcommands for different benchmarking scenarios:

**Common Options for All Commands:**

**Usage:**
.. code-block:: bash

    trtllm-bench [OPTIONS] <subcommand> [OPTIONS]

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--model``, ``-m``
     - HuggingFace model name (required)
   * - ``--model_path``
     - Path to local HuggingFace checkpoint
   * - ``--workspace``, ``-w``
     - Directory for intermediate files (default: /tmp)
   * - ``--log_level``
     - Logging level (default: info)


build
-----
Build TensorRT-LLM engines optimized for benchmarking.

**Usage:**
.. code-block:: bash

    trtllm-bench -m <model_name> build [OPTIONS]

**Key Options:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--tp_size``, ``-tp``
     - Number of tensor parallel shards (default: 1)
   * - ``--pp_size``, ``-pp``
     - Number of pipeline parallel shards (default: 1)
   * - ``--quantization``, ``-q``
     - Quantization algorithm (e.g., fp8, int8_sq, nvfp4)
   * - ``--max_seq_len``
     - Maximum total sequence length for requests
   * - ``--dataset``
     - Dataset file to extract sequence statistics for engine optimization
   * - ``--max_batch_size``
     - Maximum number of requests the engine can schedule
   * - ``--max_num_tokens``
     - Maximum number of batched tokens the engine can schedule
   * - ``--target_input_len``
     - Target average input length for tuning heuristics
   * - ``--target_output_len``
     - Target average output length for tuning heuristics

**Engine Build Modes:**
The build command supports three mutually exclusive optimization modes:

1. **Dataset-based**: Use ``--dataset`` to analyze sequence statistics and optimize engine parameters
2. **IFB Scheduler**: Use ``--max_batch_size`` and ``--max_num_tokens`` for manual tweaking of inflight batching
3. **Tuning Heuristics**: Use ``--target_input_len`` and ``--target_output_len`` for heuristic-based optimization

throughput
----------
Run throughput benchmarks to measure the engine's processing capacity under load.

**Usage:**
.. code-block:: bash

    trtllm-bench -m <model_name> throughput [OPTIONS]

**Key Options:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--engine_dir``
     - Path to the serialized TRT-LLM engine
   * - ``--backend``
     - Backend choice (pytorch, _autodeploy) -- unspecified for TensorRT.
   * - ``--extra_llm_api_options``
     - Path to YAML file that overwrites trtllm-bench parameters
   * - ``--max_batch_size``
     - Maximum runtime batch size
   * - ``--max_num_tokens``
     - Maximum runtime tokens the engine can accept
   * - ``--max_seq_len``
     - Maximum sequence length
   * - ``--beam_width``
     - Number of search beams (default: 1)
   * - ``--kv_cache_free_gpu_mem_fraction``
     - GPU memory fraction for KV cache (default: 0.90)
   * - ``--dataset``
     - Dataset file for benchmark input
   * - ``--eos_id``
     - End-of-sequence token (-1 to disable)
   * - ``--modality``
     - Modality of multimodal requests (image, video)
   * - ``--max_input_len``
     - Maximum input sequence length for multimodal models (default: 4096)
   * - ``--num_requests``
     - Number of requests to process (0 for all)
   * - ``--warmup``
     - Number of warmup requests before benchmarking
   * - ``--target_input_len``
     - Target average input length for tuning heuristics
   * - ``--target_output_len``
     - Target average output length for tuning heuristics
   * - ``--tp``
     - Tensor parallelism size (default: 1)
   * - ``--pp``
     - Pipeline parallelism size (default: 1)
   * - ``--ep``
     - Expert parallelism size
   * - ``--cluster_size``
     - Expert cluster parallelism size
   * - ``--concurrency``
     - Number of concurrent requests to process
   * - ``--streaming``
     - Enable streaming output mode
   * - ``--report_json``
     - Path to save benchmark report
   * - ``--iteration_log``
     - Path to save iteration logging
   * - ``--output_json``
     - Path to save output tokens
   * - ``--enable_chunked_context``
     - Enable chunking in prefill stage for enhanced throughput
   * - ``--scheduler_policy``
     - KV cache scheduler policy (guaranteed_no_evict, max_utilization)

**Performance Features:**
- Supports both streaming and non-streaming modes
- Configurable concurrency for load testing
- Comprehensive reporting with detailed statistics

latency
-------
Run low-latency benchmarks optimized for minimal response time.

**Usage:**
.. code-block:: bash

    trtllm-bench -m <model_name> latency [OPTIONS]

**Key Options:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--engine_dir``
     - Path to the serialized TRT-LLM engine (required)
   * - ``--kv_cache_free_gpu_mem_fraction``
     - GPU memory fraction for KV cache (default: 0.90)
   * - ``--dataset``
     - Dataset file for benchmark input
   * - ``--num_requests``
     - Number of requests to process (0 for all)
   * - ``--warmup``
     - Number of warmup requests (default: 2)
   * - ``--concurrency``
     - Number of concurrent requests (default: 1)
   * - ``--beam_width``
     - Number of search beams for beam search
   * - ``--medusa_choices``
     - Path to YAML file defining Medusa tree for speculative decoding
   * - ``--report_json``
     - Path to save benchmark report
   * - ``--iteration_log``
     - Path to save iteration logging


Examples
--------

Build an engine optimized for a specific dataset (TensorRT backend only):
.. code-block:: bash

    trtllm-bench -m <model_name> build --dataset <dataset_path> --tp_size <tp_size> --pp_size <pp_size> --quantization <quantization>

Run throughput benchmark (PyTorch):
.. code-block:: bash

    trtllm-bench -m <model_name> throughput --backend pytorch --dataset <dataset_path> --tp_size <tp_size> --pp_size <pp_size>

Run throughput benchmark (TensorRT):
.. code-block:: bash

    trtllm-bench -m <model_name> throughput --engine_dir <engine_path> --dataset <dataset_path>

Run latency benchmark:
.. code-block:: bash

    trtllm-bench -m <model_name> --engine_dir <engine_path> --kv_cache_free_gpu_mem_fraction <kv_cache_free_gpu_mem_fraction> --dataset <dataset_path> --num_requests <num_requests> --warmup <warmup> --concurrency <concurrency> --beam_width <beam_width> --medusa_choices <medusa_choices> --report_json <report_json> --iteration_log <iteration_log>

Dataset Preparation
------------------
trtllm-bench is designed to work with the ``prepare_dataset.py`` script, which generates benchmark datasets in the required format. The prepare_dataset script supports:

**Dataset Types:**
- Real datasets from various sources
- Synthetic datasets with normal or uniform token distributions
- LoRA task-specific datasets

**Key Features:**
- Tokenizer integration for proper text preprocessing
- Configurable random seeds for reproducible results
- Support for LoRA adapters and task IDs
- Output in JSON format compatible with trtllm-bench

.. important::
   The ``--stdout`` flag is **required** when using prepare_dataset.py with trtllm-bench to ensure proper data streaming format.

**prepare_dataset.py CLI Options:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--tokenizer``
     - Tokenizer directory or HuggingFace model name (required)
   * - ``--output``
     - Output JSON filename (default: preprocessed_dataset.json)
   * - ``--stdout``
     - Print output to stdout with JSON dataset entry on each line (**required for trtllm-bench**)
   * - ``--random-seed``
     - Random seed for token generation (default: 420)
   * - ``--task-id``
     - LoRA task ID (default: -1)
   * - ``--rand-task-id``
     - Random LoRA task range (two integers)
   * - ``--lora-dir``
     - Directory containing LoRA adapters
   * - ``--log-level``
     - Logging level: info or debug (default: info)

**prepare_dataset.py Subcommands:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Subcommand
     - Description
   * - ``dataset``
     - Process real datasets from various sources
   * - ``token_norm_dist``
     - Generate synthetic datasets with normal token distribution
   * - ``token_unif_dist``
     - Generate synthetic datasets with uniform token distribution

**Usage Example:**
.. code-block:: bash

    python prepare_dataset.py --tokenizer meta-llama/Meta-Llama-3.3-8B --stdout dataset --output benchmark_data.jsonl

This workflow allows you to:
1. Prepare datasets using ``prepare_dataset.py`` with the required ``--stdout`` flag
2. Build optimized engines with ``trtllm-bench build`` using the prepared dataset
3. Run comprehensive benchmarks with ``trtllm-bench throughput`` or ``trtllm-bench latency``
