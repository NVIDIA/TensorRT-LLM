trtllm-eval
===========

About
-----

The ``trtllm-eval`` command provides developers with a unified entry point for accuracy evaluation. It shares the core evaluation logic with the `accuracy test suite <https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests/integration/defs/accuracy>`_ of TensorRT-LLM.

``trtllm-eval`` is built on the offline API -- LLM API. Compared to the online ``trtllm-serve``, the offline API provides clearer error messages and simplifies the debugging workflow.

The following tasks are currently supported:

.. list-table::
   :header-rows: 1
   :widths: 20 25 15 15 15

   * - Dataset
     - Task
     - Metric
     - Default ISL
     - Default OSL
   * - CNN Dailymail
     - summarization
     - rouge
     - 924
     - 100
   * - MMLU
     - QA; multiple choice
     - accuracy
     - 4,094
     - 2
   * - GSM8K
     - QA; regex matching
     - accuracy
     - 4,096
     - 256
   * - GPQA
     - QA; multiple choice
     - accuracy
     - 32,768
     - 4,096
   * - JSON mode eval
     - structured generation
     - accuracy
     - 1,024
     - 512

.. note::

    ``trtllm-eval`` originates from the TensorRT-LLM accuracy test suite and serves as a lightweight utility for verifying and debugging accuracy. At this time, ``trtllm-eval`` is intended solely for development and is not recommended for production use.

Usage and Examples
------------------

Some evaluation tasks (e.g., GSM8K and GPQA) depend on the ``lm_eval`` package. To run these tasks, you need to install ``lm_eval`` with:

.. code-block:: bash

   pip install -r requirements-dev.txt

Alternatively, you can install the ``lm_eval`` version specified in ``requirements-dev.txt``.

Here are some examples:

.. code-block:: bash

   # Evaluate Llama-3.1-8B-Instruct on MMLU
   trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct mmlu

   # Evaluate Llama-3.1-8B-Instruct on GSM8K
   trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct gsm8k

   # Evaluate Llama-3.3-70B-Instruct on GPQA Diamond
   trtllm-eval --model meta-llama/Llama-3.3-70B-Instruct gpqa_diamond

The ``--model`` argument accepts either a Hugging Face model ID or a local checkpoint path. By default, ``trtllm-eval`` runs the model with the PyTorch backend; you can pass ``--backend tensorrt`` to switch to the TensorRT backend.

Alternatively, the ``--model`` argument also accepts a local path to pre-built TensorRT engines. In this case, you should pass the Hugging Face tokenizer path to the ``--tokenizer`` argument.

For more details, see ``trtllm-eval --help`` and ``trtllm-eval <task> --help``.



Syntax
------

.. click:: tensorrt_llm.commands.eval:main
   :prog: trtllm-eval
   :nested: full
