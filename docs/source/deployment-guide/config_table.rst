.. start-config-table-note
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-traffic-patterns
   :end-before: .. end-note-traffic-patterns
.. end-config-table-note

.. start-Qwen/Qwen3-235B-A22B

.. _Qwen/Qwen3-235B-A22B:

`Qwen3-235B-A22B <https://huggingface.co/Qwen/Qwen3-235B-A22B>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 15 13 20 25

   * - GPU
     - Performance Profile
     - ISL / OSL
     - Concurrency
     - Config
     - Command
   * - GB200
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp1_conc4.yaml``
   * - 4xGB200
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp4_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc1.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc1.yaml``
   * - 4xGB200
     - Balanced
     - 1024 / 1024
     - 2
     - `1k1k_tp4_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc2.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc2.yaml``
   * - 4xGB200
     - Balanced
     - 1024 / 1024
     - 4
     - `1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc4.yaml``
   * - 4xGB200
     - Max Throughput
     - 1024 / 1024
     - 8
     - `1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp4_conc8.yaml``
   * - 4xGB200
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp4_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc1.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc1.yaml``
   * - 4xGB200
     - Balanced
     - 1024 / 8192
     - 2
     - `1k8k_tp4_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc2.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc2.yaml``
   * - 4xGB200
     - Balanced
     - 1024 / 8192
     - 8
     - `1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc8.yaml``
   * - 4xGB200
     - Max Throughput
     - 1024 / 8192
     - 16
     - `1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp4_conc16.yaml``
   * - 4xGB200
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp4_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc1.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc1.yaml``
   * - 4xGB200
     - Balanced
     - 8192 / 1024
     - 2
     - `8k1k_tp4_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc2.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc2.yaml``
   * - 4xGB200
     - Balanced
     - 8192 / 1024
     - 4
     - `8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc4.yaml``
   * - 4xGB200
     - Max Throughput
     - 8192 / 1024
     - 8
     - `8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp4_conc8.yaml``
   * - 8xGB200
     - Min Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc16.yaml``
   * - 8xGB200
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc32.yaml``
   * - 8xGB200
     - Low Latency
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc64.yaml``
   * - 8xGB200
     - Balanced
     - 1024 / 1024
     - 128
     - `1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc128.yaml``
   * - 8xGB200
     - Balanced
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc256.yaml``
   * - 8xGB200
     - High Throughput
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc512.yaml``
   * - 8xGB200
     - High Throughput
     - 1024 / 1024
     - 1024
     - `1k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc1024.yaml``
   * - 8xGB200
     - Max Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k1k_tp8_conc2048.yaml``
   * - 8xGB200
     - Min Latency
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc32.yaml``
   * - 8xGB200
     - Low Latency
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc64.yaml``
   * - 8xGB200
     - Low Latency
     - 1024 / 8192
     - 128
     - `1k8k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc128.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc128.yaml``
   * - 8xGB200
     - Balanced
     - 1024 / 8192
     - 256
     - `1k8k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc256.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc256.yaml``
   * - 8xGB200
     - High Throughput
     - 1024 / 8192
     - 512
     - `1k8k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc512.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc512.yaml``
   * - 8xGB200
     - High Throughput
     - 1024 / 8192
     - 1024
     - `1k8k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc1024.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc1024.yaml``
   * - 8xGB200
     - Max Throughput
     - 1024 / 8192
     - 2048
     - `1k8k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc2048.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/1k8k_tp8_conc2048.yaml``
   * - 8xGB200
     - Min Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc16.yaml``
   * - 8xGB200
     - Low Latency
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc32.yaml``
   * - 8xGB200
     - Low Latency
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc64.yaml``
   * - 8xGB200
     - Balanced
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc128.yaml``
   * - 8xGB200
     - Balanced
     - 8192 / 1024
     - 256
     - `8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc256.yaml``
   * - 8xGB200
     - High Throughput
     - 8192 / 1024
     - 512
     - `8k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc512.yaml``
   * - 8xGB200
     - High Throughput
     - 8192 / 1024
     - 1024
     - `8k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc1024.yaml``
   * - 8xGB200
     - Max Throughput
     - 8192 / 1024
     - 2048
     - `8k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve Qwen/Qwen3-235B-A22B --config ${TRTLLM_DIR}/examples/configs/database/Qwen/Qwen3-235B-A22B/GB200/8k1k_tp8_conc2048.yaml``

.. end-Qwen/Qwen3-235B-A22B

.. start-deepseek-ai/DeepSeek-R1-0528

.. _deepseek-ai/DeepSeek-R1-0528:

`DeepSeek-R1 <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 15 13 20 25

   * - GPU
     - Performance Profile
     - ISL / OSL
     - Concurrency
     - Config
     - Command
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 2
     - `1k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - `1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 1024
     - `1k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 4096
     - `1k1k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc4096.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc4096.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 8192
     - `1k1k_tp8_conc8192.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc8192.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc8192.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 2
     - `1k8k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 128
     - `1k8k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 256
     - `1k8k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 512
     - `1k8k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc512.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 1024
     - `1k8k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1024.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 2048
     - `1k8k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2048.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 2
     - `8k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Balanced
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 512
     - `8k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 1024
     - `8k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 2048
     - `8k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 4096
     - `8k1k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc4096.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc4096.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 2
     - `1k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Balanced
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 128
     - `1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc128.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc256.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc512.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 1024
     - `1k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1024.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2048.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 2
     - `1k8k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc2.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 128
     - `1k8k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc128.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 256
     - `1k8k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc256.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 2
     - `8k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc2.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc128.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc256.yaml``

.. end-deepseek-ai/DeepSeek-R1-0528

.. start-nvidia/DeepSeek-R1-0528-FP4-v2

.. _nvidia/DeepSeek-R1-0528-FP4-v2:

`DeepSeek-R1 (NVFP4) <https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 15 13 20 25

   * - GPU
     - Performance Profile
     - ISL / OSL
     - Concurrency
     - Config
     - Command
   * - 4xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc4.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc8.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Balanced
     - 1024 / 1024
     - 64
     - `1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 128
     - `1k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc128.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc256.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 4096
     - `1k1k_tp4_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc4096.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc4096.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 8192
     - `1k1k_tp4_conc8192.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc8192.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc8192.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 2048
     - `1k8k_tp4_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp4_conc2048.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp4_conc2048.yaml``
   * - 4xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc4.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc8.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Balanced
     - 8192 / 1024
     - 64
     - `8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 128
     - `8k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc128.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc256.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 2048
     - `8k1k_tp4_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc2048.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc2048.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 2
     - `1k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 128
     - `1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 1024
     - `1k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc1.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 2
     - `1k8k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc2.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 128
     - `1k8k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 256
     - `1k8k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 512
     - `1k8k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc512.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 1024
     - `1k8k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc1024.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 2
     - `8k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 512
     - `8k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 1024
     - `8k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc1024.yaml``

.. end-nvidia/DeepSeek-R1-0528-FP4-v2

.. start-openai/gpt-oss-120b

.. _openai/gpt-oss-120b:

`gpt-oss-120b <https://huggingface.co/openai/gpt-oss-120b>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 15 13 20 25

   * - GPU
     - Performance Profile
     - ISL / OSL
     - Concurrency
     - Config
     - Command
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc16.yaml``
   * - B200_NVL
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp1_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc4.yaml``
   * - B200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc8.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc16.yaml``
   * - B200_NVL
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp1_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc16.yaml``
   * - B200_NVL
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp1_conc64.yaml``
   * - 2xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc4.yaml``
   * - 2xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc8.yaml``
   * - 2xB200_NVL
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc16.yaml``
   * - 2xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc32.yaml``
   * - 2xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp2_conc64.yaml``
   * - 2xB200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc4.yaml``
   * - 2xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc8.yaml``
   * - 2xB200_NVL
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc16.yaml``
   * - 2xB200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc32.yaml``
   * - 2xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc64.yaml``
   * - 2xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 256
     - `1k8k_tp2_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc256.yaml``
   * - 2xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc4.yaml``
   * - 2xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc8.yaml``
   * - 2xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc16.yaml``
   * - 2xB200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc32.yaml``
   * - 2xB200_NVL
     - Balanced
     - 8192 / 1024
     - 64
     - `8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc64.yaml``
   * - 2xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 768
     - `8k1k_tp2_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc768.yaml``
   * - 2xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 1024
     - `8k1k_tp2_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc1024.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc1024.yaml``
   * - 2xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 1280
     - `8k1k_tp2_conc1280.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc1280.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc1280.yaml``
   * - 4xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc4.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc8.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 64
     - `1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - `1k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc128.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc256.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 1280
     - `1k1k_tp4_conc1280.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1280.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1280.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 1536
     - `1k1k_tp4_conc1536.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1536.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc1536.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 2560
     - `1k1k_tp4_conc2560.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc2560.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc2560.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 3584
     - `1k1k_tp4_conc3584.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc3584.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc3584.yaml``
   * - 4xB200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc4.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc8.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 10
     - `1k8k_tp4_conc10.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc10.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc10.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 16
     - `1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 32
     - `1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Balanced
     - 1024 / 8192
     - 64
     - `1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 128
     - `1k8k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc128.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 384
     - `1k8k_tp4_conc384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc384.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 640
     - `1k8k_tp4_conc640.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc640.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc640.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 896
     - `1k8k_tp4_conc896.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc896.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc896.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 1536
     - `1k8k_tp4_conc1536.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc1536.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc1536.yaml``
   * - 4xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp4_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 2
     - `8k1k_tp4_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc2.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc4.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc8.yaml``
   * - 4xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 10
     - `8k1k_tp4_conc10.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc10.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc10.yaml``
   * - 4xB200_NVL
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc256.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 1536
     - `8k1k_tp4_conc1536.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1536.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1536.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 1792
     - `8k1k_tp4_conc1792.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1792.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc1792.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 3584
     - `8k1k_tp4_conc3584.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc3584.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc3584.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 2
     - `1k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 384
     - `1k1k_tp8_conc384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc384.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 640
     - `1k1k_tp8_conc640.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc640.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc640.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 768
     - `1k1k_tp8_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc768.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 1024
     - 896
     - `1k1k_tp8_conc896.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc896.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc896.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 1024
     - 1024
     - `1k1k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1024.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 1792
     - `1k1k_tp8_conc1792.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1792.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc1792.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 3072
     - `1k1k_tp8_conc3072.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc3072.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc3072.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 4096
     - `1k1k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4096.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4096.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 4608
     - `1k1k_tp8_conc4608.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4608.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc4608.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 5120
     - `1k1k_tp8_conc5120.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc5120.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc5120.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 6144
     - `1k1k_tp8_conc6144.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc6144.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc6144.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 7168
     - `1k1k_tp8_conc7168.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc7168.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc7168.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 8192
     - `1k1k_tp8_conc8192.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc8192.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc8192.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 10240
     - `1k1k_tp8_conc10240.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc10240.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc10240.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 14336
     - `1k1k_tp8_conc14336.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc14336.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc14336.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 16384
     - `1k1k_tp8_conc16384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16384.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 2
     - `1k8k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Balanced
     - 1024 / 8192
     - 512
     - `1k8k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 768
     - `1k8k_tp8_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc768.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 1024
     - `1k8k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1024.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1024.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 1280
     - `1k8k_tp8_conc1280.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1280.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1280.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 1792
     - `1k8k_tp8_conc1792.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1792.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc1792.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 2048
     - `1k8k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2048.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 3072
     - `1k8k_tp8_conc3072.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc3072.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc3072.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 4096
     - `1k8k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc4096.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc4096.yaml``
   * - 8xB200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc4.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc8.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Low Latency
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc128.yaml``
   * - 8xB200_NVL
     - Balanced
     - 8192 / 1024
     - 384
     - `8k1k_tp8_conc384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc384.yaml``
   * - 8xB200_NVL
     - Balanced
     - 8192 / 1024
     - 512
     - `8k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc512.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 640
     - `8k1k_tp8_conc640.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc640.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc640.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 896
     - `8k1k_tp8_conc896.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc896.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc896.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 2048
     - `8k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc2048.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 3072
     - `8k1k_tp8_conc3072.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc3072.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc3072.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 4096
     - `8k1k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc4096.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc4096.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 5120
     - `8k1k_tp8_conc5120.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc5120.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc5120.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc16.yaml``
   * - H200_SXM
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp1_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc4.yaml``
   * - H200_SXM
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc8.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc16.yaml``
   * - H200_SXM
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp1_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc16.yaml``
   * - H200_SXM
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp1_conc64.yaml``
   * - 2xH200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc4.yaml``
   * - 2xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc8.yaml``
   * - 2xH200_SXM
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc16.yaml``
   * - 2xH200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc32.yaml``
   * - 2xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc64.yaml``
   * - 2xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 2560
     - `1k1k_tp2_conc2560.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc2560.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc2560.yaml``
   * - 2xH200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc4.yaml``
   * - 2xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc8.yaml``
   * - 2xH200_SXM
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc16.yaml``
   * - 2xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc32.yaml``
   * - 2xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp2_conc64.yaml``
   * - 2xH200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc4.yaml``
   * - 2xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc8.yaml``
   * - 2xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc16.yaml``
   * - 2xH200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc32.yaml``
   * - 2xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc64.yaml``
   * - 2xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp2_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc256.yaml``
   * - 2xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 384
     - `8k1k_tp2_conc384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc384.yaml``
   * - 4xH200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc4.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc8.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc16.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc32.yaml``
   * - 4xH200_SXM
     - Balanced
     - 1024 / 1024
     - 64
     - `1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc64.yaml``
   * - 4xH200_SXM
     - Balanced
     - 1024 / 1024
     - 128
     - `1k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc128.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 384
     - `1k1k_tp4_conc384.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc384.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc384.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 1024
     - `1k1k_tp4_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc1024.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc1024.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 3584
     - `1k1k_tp4_conc3584.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc3584.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc3584.yaml``
   * - 4xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 5120
     - `1k1k_tp4_conc5120.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc5120.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc5120.yaml``
   * - 4xH200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc4.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc8.yaml``
   * - 4xH200_SXM
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc16.yaml``
   * - 4xH200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - `1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc32.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc64.yaml``
   * - 4xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 512
     - `1k8k_tp4_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc512.yaml``
   * - 4xH200_SXM
     - Min Latency
     - 8192 / 1024
     - 2
     - `8k1k_tp4_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc2.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc4.yaml``
   * - 4xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc8.yaml``
   * - 4xH200_SXM
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc16.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc32.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc64.yaml``
   * - 4xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 768
     - `8k1k_tp4_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc768.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 1024 / 1024
     - 1
     - `1k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 2
     - `1k1k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc2.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 4
     - `1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 8
     - `1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc256.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 1024
     - 512
     - `1k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc512.yaml``
   * - 8xH200_SXM
     - Balanced
     - 1024 / 1024
     - 640
     - `1k1k_tp8_conc640.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc640.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc640.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 768
     - `1k1k_tp8_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc768.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 896
     - `1k1k_tp8_conc896.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc896.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc896.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 1280
     - `1k1k_tp8_conc1280.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1280.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1280.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 1536
     - `1k1k_tp8_conc1536.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1536.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1536.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 1792
     - `1k1k_tp8_conc1792.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1792.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc1792.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 2048
     - `1k1k_tp8_conc2048.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc2048.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc2048.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 3072
     - `1k1k_tp8_conc3072.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc3072.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc3072.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 4096
     - `1k1k_tp8_conc4096.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc4096.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc4096.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 6144
     - `1k1k_tp8_conc6144.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc6144.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc6144.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 1024 / 8192
     - 1
     - `1k8k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 2
     - `1k8k_tp8_conc2.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc2.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc2.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 4
     - `1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 8
     - `1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Balanced
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 128
     - `1k8k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc128.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 256
     - `1k8k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc256.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc256.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 768
     - `1k8k_tp8_conc768.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc768.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc768.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 896
     - `1k8k_tp8_conc896.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc896.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc896.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 1024
     - `1k8k_tp8_conc1024.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1024.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1024.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 1280
     - `1k8k_tp8_conc1280.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1280.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc1280.yaml``
   * - 8xH200_SXM
     - Min Latency
     - 8192 / 1024
     - 1
     - `8k1k_tp8_conc1.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc1.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc1.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 4
     - `8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc4.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 8
     - `8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc8.yaml``
   * - 8xH200_SXM
     - Low Latency
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Balanced
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 128
     - `8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc128.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 512
     - `8k1k_tp8_conc512.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc512.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc512.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 640
     - `8k1k_tp8_conc640.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc640.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc640.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 1536
     - `8k1k_tp8_conc1536.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc1536.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc1536.yaml``

.. end-openai/gpt-oss-120b
