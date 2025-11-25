.. start-deepseek-ai/DeepSeek-R1-0528

.. _deepseek-ai/DeepSeek-R1-0528:

`DeepSeek-R1 <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 20 25 15 13

   * - GPU
     - Performance Profile
     - Config
     - Command
     - Best ISL / OSL
     - Best Concurrency
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Balanced
     - `fp8_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Max Throughput
     - `fp8_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp8_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Balanced
     - `fp8_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Max Throughput
     - `fp8_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml``
     - 8192 / 1024
     - 64
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml``
     - 1024 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml``
     - 1024 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml``
     - 1024 / 1024
     - 16
   * - H200_SXM
     - Balanced
     - `fp8_h200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml``
     - 1024 / 1024
     - 32
   * - H200_SXM
     - Max Throughput
     - `fp8_h200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml``
     - 1024 / 1024
     - 64
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml``
     - 8192 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml``
     - 8192 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp8_h200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml``
     - 8192 / 1024
     - 16
   * - H200_SXM
     - Balanced
     - `fp8_h200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml``
     - 8192 / 1024
     - 32
   * - H200_SXM
     - Max Throughput
     - `fp8_h200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml``
     - 8192 / 1024
     - 64

.. end-deepseek-ai/DeepSeek-R1-0528

.. start-nvidia/DeepSeek-R1-0528-FP4-v2

.. _nvidia/DeepSeek-R1-0528-FP4-v2:

`DeepSeek-R1 (NVFP4) <https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 20 25 15 13

   * - GPU
     - Performance Profile
     - Config
     - Command
     - Best ISL / OSL
     - Best Concurrency
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml``
     - 1024 / 1024
     - 128
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml``
     - 1024 / 1024
     - 128
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml``
     - 1024 / 1024
     - 256
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml``
     - 1024 / 1024
     - 256
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml``
     - 8192 / 1024
     - 64
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml``
     - 8192 / 1024
     - 64
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp4_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml``
     - 8192 / 1024
     - 128
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp8_conc128.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml``
     - 8192 / 1024
     - 128
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp4_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml``
     - 8192 / 1024
     - 256
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp8_conc256.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml>`_
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml``
     - 8192 / 1024
     - 256

.. end-nvidia/DeepSeek-R1-0528-FP4-v2

.. start-openai/gpt-oss-120b

.. _openai/gpt-oss-120b:

`gpt-oss-120b <https://huggingface.co/openai/gpt-oss-120b>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1
   :widths: 12 15 20 25 15 13

   * - GPU
     - Performance Profile
     - Config
     - Command
     - Best ISL / OSL
     - Best Concurrency
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml``
     - 1024 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml``
     - 1024 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml``
     - 1024 / 1024
     - 16
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml``
     - 1024 / 1024
     - 32
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml``
     - 1024 / 1024
     - 64
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml``
     - 1024 / 8192
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml``
     - 1024 / 8192
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml``
     - 1024 / 8192
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml``
     - 1024 / 8192
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml``
     - 1024 / 8192
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml``
     - 1024 / 8192
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml``
     - 1024 / 8192
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml``
     - 1024 / 8192
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml``
     - 1024 / 8192
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml``
     - 1024 / 8192
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml``
     - 1024 / 8192
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml``
     - 1024 / 8192
     - 16
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml``
     - 1024 / 8192
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml``
     - 1024 / 8192
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml``
     - 1024 / 8192
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml``
     - 1024 / 8192
     - 32
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml``
     - 1024 / 8192
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml``
     - 1024 / 8192
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml``
     - 1024 / 8192
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml``
     - 1024 / 8192
     - 64
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml``
     - 8192 / 1024
     - 4
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml``
     - 8192 / 1024
     - 8
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Min Latency
     - `fp4_b200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml``
     - 8192 / 1024
     - 16
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Balanced
     - `fp4_b200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml``
     - 8192 / 1024
     - 32
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml``
     - 8192 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml``
     - 8192 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml``
     - 8192 / 1024
     - 64
   * - B200_NVL
     - Max Throughput
     - `fp4_b200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml``
     - 8192 / 1024
     - 64
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml``
     - 1024 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml``
     - 1024 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml``
     - 1024 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml``
     - 1024 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml``
     - 1024 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml``
     - 1024 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml``
     - 1024 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml``
     - 1024 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml``
     - 1024 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml``
     - 1024 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml``
     - 1024 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml``
     - 1024 / 1024
     - 16
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml``
     - 1024 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml``
     - 1024 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml``
     - 1024 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml``
     - 1024 / 1024
     - 32
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml``
     - 1024 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml``
     - 1024 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml``
     - 1024 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml``
     - 1024 / 1024
     - 64
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml``
     - 1024 / 8192
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml``
     - 1024 / 8192
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml``
     - 1024 / 8192
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml``
     - 1024 / 8192
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml``
     - 1024 / 8192
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml``
     - 1024 / 8192
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml``
     - 1024 / 8192
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml``
     - 1024 / 8192
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml``
     - 1024 / 8192
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml``
     - 1024 / 8192
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml``
     - 1024 / 8192
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_1k8k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml``
     - 1024 / 8192
     - 16
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k8k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml``
     - 1024 / 8192
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k8k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml``
     - 1024 / 8192
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k8k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml``
     - 1024 / 8192
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_1k8k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml``
     - 1024 / 8192
     - 32
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k8k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml``
     - 1024 / 8192
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k8k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml``
     - 1024 / 8192
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k8k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml``
     - 1024 / 8192
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_1k8k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml``
     - 1024 / 8192
     - 64
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp1_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml``
     - 8192 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp2_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml``
     - 8192 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp4_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml``
     - 8192 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp8_conc4.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml``
     - 8192 / 1024
     - 4
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp1_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml``
     - 8192 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp2_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml``
     - 8192 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp4_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml``
     - 8192 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp8_conc8.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml``
     - 8192 / 1024
     - 8
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp1_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml``
     - 8192 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp2_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml``
     - 8192 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp4_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml``
     - 8192 / 1024
     - 16
   * - H200_SXM
     - Min Latency
     - `fp4_h200_trt_8k1k_tp8_conc16.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml``
     - 8192 / 1024
     - 16
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_8k1k_tp1_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml``
     - 8192 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_8k1k_tp2_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml``
     - 8192 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_8k1k_tp4_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml``
     - 8192 / 1024
     - 32
   * - H200_SXM
     - Balanced
     - `fp4_h200_trt_8k1k_tp8_conc32.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml``
     - 8192 / 1024
     - 32
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_8k1k_tp1_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml``
     - 8192 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_8k1k_tp2_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml``
     - 8192 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_8k1k_tp4_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml``
     - 8192 / 1024
     - 64
   * - H200_SXM
     - Max Throughput
     - `fp4_h200_trt_8k1k_tp8_conc64.yaml <https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml>`_
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml``
     - 8192 / 1024
     - 64

.. end-openai/gpt-oss-120b
