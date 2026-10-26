trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT LLM engines. It provides subcommands for different benchmarking scenarios.

.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias

Syntax
------

.. click:: tensorrt_llm.commands.bench:main
   :prog: trtllm-bench
   :nested: full
   :commands: throughput, latency, prepare-dataset



Dataset preparation
-------------------

**Dataset Types:**

- Real datasets from various sources
- Synthetic datasets with normal or uniform token distributions
- LoRA task-specific datasets

**Key Features:**

- Tokenizer integration for proper text preprocessing — the tokenizer is resolved from the model passed to ``trtllm-bench --model``
- Configurable random seeds for reproducible results
- Support for LoRA adapters and task IDs
- Output in JSON format compatible with trtllm-bench
