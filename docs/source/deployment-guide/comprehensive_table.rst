.. start-deepseek-ai/DeepSeek-R1-0528

.. _deepseek-ai/DeepSeek-R1-0528:

deepseek-ai/DeepSeek-R1-0528
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 18 10 20 25

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
     - `deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - `deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - `deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - `deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - `deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml``

.. end-deepseek-ai/DeepSeek-R1-0528

.. start-nvidia/DeepSeek-R1-0528-FP4-v2

.. _nvidia/DeepSeek-R1-0528-FP4-v2:

nvidia/DeepSeek-R1-0528-FP4-v2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 18 10 20 25

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
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 32
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 32
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 64
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 64
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 256
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 256
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 32
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 32
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 64
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 64
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 128
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 128
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 256
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 256
     - `deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml``

.. end-nvidia/DeepSeek-R1-0528-FP4-v2

.. start-openai/gpt-oss-120b

.. _openai/gpt-oss-120b:

openai/gpt-oss-120b
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 18 10 20 25

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
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - `gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - `gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml``

.. end-openai/gpt-oss-120b
