trtllm-bench
===========================

trtllm-bench is a comprehensive benchmarking tool for TensorRT LLM engines. It provides subcommands for different benchmarking scenarios:

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
-------------------

prepare-dataset
^^^^^^^^^^^^^^^

.. click:: tensorrt_llm.commands.bench:main
   :prog: trtllm-bench
   :nested: full
   :commands: prepare-dataset
