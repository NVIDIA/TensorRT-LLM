trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT LLM engines. It provides three main subcommands for different benchmarking scenarios:

**Common Options for All Commands:**

**Usage:**

.. click:: tensorrt_llm.commands.bench:main
   :prog: trtllm-bench
   :nested: full
   :commands: throughput, latency, build



prepare_dataset.py
===========================

trtllm-bench is designed to work with the `prepare_dataset.py <https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/prepare_dataset.py>`_ script, which generates benchmark datasets in the required format. The prepare_dataset script supports:

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

**Usage:**

prepare_dataset
-------------------

.. code-block:: bash

    python prepare_dataset.py [OPTIONS]

**Options**

----

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

dataset
-------------------

Process real datasets from various sources.

.. code-block:: bash

    python prepare_dataset.py dataset [OPTIONS]

**Options**

----

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--input``
     - Input dataset file or directory (required)
   * - ``--max-input-length``
     - Maximum input sequence length (default: 2048)
   * - ``--max-output-length``
     - Maximum output sequence length (default: 512)
   * - ``--num-samples``
     - Number of samples to process (default: all)
   * - ``--format``
     - Input format: json, jsonl, csv, or txt (default: auto-detect)


token_norm_dist
-------------------

Generate synthetic datasets with normal token distribution.

.. code-block:: bash

    python prepare_dataset.py token_norm_dist [OPTIONS]

**Options**

----

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--num-requests``
     - Number of requests to be generated (required)
   * - ``--input-mean``
     - Normal distribution mean for input tokens (required)
   * - ``--input-stdev``
     - Normal distribution standard deviation for input tokens (required)
   * - ``--output-mean``
     - Normal distribution mean for output tokens (required)
   * - ``--output-stdev``
     - Normal distribution standard deviation for output tokens (required)


token_unif_dist
-------------------

Generate synthetic datasets with uniform token distribution

.. code-block:: bash

    python prepare_dataset.py token_unif_dist [OPTIONS]

**Options**

----

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--num-requests``
     - Number of requests to be generated (required)
   * - ``--input-min``
     - Uniform distribution minimum for input tokens (required)
   * - ``--input-max``
     - Uniform distribution maximum for input tokens (required)
   * - ``--output-min``
     - Uniform distribution minimum for output tokens (required)
   * - ``--output-max``
     - Uniform distribution maximum for output tokens (required)
