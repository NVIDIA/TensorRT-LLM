.. start-config-table-note
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-traffic-patterns
   :end-before: .. end-note-traffic-patterns
.. end-config-table-note

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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Min Latency
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - Min Latency
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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - Min Latency
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
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --config ${TRTLLM_DIR}/examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc64.yaml``

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
     - Balanced
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - High Throughput
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
     - Max Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp4_conc256.yaml``
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
     - High Throughput
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
     - Max Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp4_conc256.yaml``
   * - 8xB200_NVL
     - Min Latency
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
     - High Throughput
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
     - Max Throughput
     - 1024 / 1024
     - 256
     - `1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/1k1k_tp8_conc256.yaml``
   * - 8xB200_NVL
     - Min Latency
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
     - Max Throughput
     - 8192 / 1024
     - 256
     - `8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --config ${TRTLLM_DIR}/examples/configs/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/8k1k_tp8_conc256.yaml``

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
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc32.yaml``
   * - 2xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp2_conc64.yaml``
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc16.yaml``
   * - 2xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc32.yaml``
   * - 2xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp2_conc64.yaml``
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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp4_conc64.yaml``
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
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp4_conc64.yaml``
   * - 4xB200_NVL
     - Min Latency
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc16.yaml``
   * - 4xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc32.yaml``
   * - 4xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp4_conc64.yaml``
   * - 8xB200_NVL
     - Min Latency
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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k1k_tp8_conc64.yaml``
   * - 8xB200_NVL
     - Min Latency
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
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/1k8k_tp8_conc64.yaml``
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc16.yaml``
   * - 8xB200_NVL
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc32.yaml``
   * - 8xB200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/B200/8k1k_tp8_conc64.yaml``
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
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc32.yaml``
   * - 2xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp2_conc64.yaml``
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc16.yaml``
   * - 2xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc32.yaml``
   * - 2xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp2_conc64.yaml``
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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc16.yaml``
   * - 4xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc32.yaml``
   * - 4xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp4_conc64.yaml``
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
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc32.yaml``
   * - 4xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp4_conc64.yaml``
   * - 4xH200_SXM
     - Min Latency
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
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp4_conc64.yaml``
   * - 8xH200_SXM
     - Min Latency
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
     - Balanced
     - 1024 / 1024
     - 16
     - `1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 1024
     - 32
     - `1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k1k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - Min Latency
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
     - Balanced
     - 1024 / 8192
     - 16
     - `1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 1024 / 8192
     - 32
     - `1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/1k8k_tp8_conc64.yaml``
   * - 8xH200_SXM
     - Min Latency
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
     - Balanced
     - 8192 / 1024
     - 16
     - `8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc16.yaml``
   * - 8xH200_SXM
     - High Throughput
     - 8192 / 1024
     - 32
     - `8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc32.yaml``
   * - 8xH200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --config ${TRTLLM_DIR}/examples/configs/database/openai/gpt-oss-120b/H200/8k1k_tp8_conc64.yaml``

.. end-openai/gpt-oss-120b
