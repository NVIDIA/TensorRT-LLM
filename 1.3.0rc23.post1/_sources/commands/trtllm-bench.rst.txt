trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT LLM engines. It provides three main subcommands for different benchmarking scenarios:

.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias

Syntax
------

.. click:: tensorrt_llm.commands.bench:main
   :prog: trtllm-bench
   :nested: full
   :commands: throughput, latency, build



Dataset preparation
------------------

prepare-dataset
^^^^^^^^^^^^^^^

trtllm-bench ships a ``prepare-dataset`` subcommand which generates benchmark datasets in the required format. It supports:

**Dataset Types:**

- Real datasets from various sources
- Synthetic datasets with normal or uniform token distributions
- LoRA task-specific datasets

**Key Features:**

- Tokenizer integration for proper text preprocessing
- Configurable random seeds for reproducible results
- Support for LoRA adapters and task IDs
- Output in JSON format compatible with trtllm-bench

.. note::
   The tokenizer is taken from the model passed to ``trtllm-bench --model``. Use ``--output`` to write the dataset to a file, or ``--stdout`` to stream it with a JSON dataset entry on each line.

**Usage:**

prepare-dataset
"""""""""""""""

.. code-block:: bash

    trtllm-bench --model <model> prepare-dataset [OPTIONS]

**Options**

----

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--output``
     - Output JSON filename (default: preprocessed_dataset.json)
   * - ``--stdout``
     - Print output to stdout with a JSON dataset entry on each line instead of writing a file
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

real-dataset
""""""""""""

Process real datasets from various sources.

.. code-block:: bash

    trtllm-bench --model <model> prepare-dataset real-dataset [OPTIONS]

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


token-norm-dist
"""""""""""""""

Generate synthetic datasets with normal token distribution.

.. code-block:: bash

    trtllm-bench --model <model> prepare-dataset token-norm-dist [OPTIONS]

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


token-unif-dist
"""""""""""""""

Generate synthetic datasets with uniform token distribution

.. code-block:: bash

    trtllm-bench --model <model> prepare-dataset token-unif-dist [OPTIONS]

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
