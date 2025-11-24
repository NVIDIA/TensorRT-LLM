deepseek-ai/DeepSeek-R1-0528
============================

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
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/B200/deepseek_r1_0528_fp8_b200_trt_8k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_1k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml
     - ``trtllm-serve deepseek-ai/DeepSeek-R1-0528 --extra_llm_api_options tensorrt_llm/configure/database/deepseek-ai/DeepSeek-R1-0528/H200/deepseek_r1_0528_fp8_h200_trt_8k1k_tp8_conc64.yaml``

nvidia/DeepSeek-R1-0528-FP4-v2
==============================

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
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc128.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 128
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc128.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 256
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp4_conc256.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 256
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_1k1k_tp8_conc256.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 128
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc128.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 128
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc128.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 256
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp4_conc256.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 256
     - tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml
     - ``trtllm-serve nvidia/DeepSeek-R1-0528-FP4-v2 --extra_llm_api_options tensorrt_llm/configure/database/nvidia/DeepSeek-R1-0528-FP4-v2/B200/deepseek_r1_0528_fp4_v2_fp4_b200_trt_8k1k_tp8_conc256.yaml``

openai/gpt-oss-120b
===================

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
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k1k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_1k8k_tp8_conc64.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc4.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc8.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc16.yaml``
   * - B200_NVL
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc16.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc32.yaml``
   * - B200_NVL
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc32.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp1_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp2_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp4_conc64.yaml``
   * - B200_NVL
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/B200/gpt_oss_120b_fp4_b200_trt_8k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k1k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 1024 / 8192
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 1024 / 8192
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 1024 / 8192
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_1k8k_tp8_conc64.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 4
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc4.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 8
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc8.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc16.yaml``
   * - H200_SXM
     - Min Latency
     - 8192 / 1024
     - 16
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc16.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc32.yaml``
   * - H200_SXM
     - Balanced
     - 8192 / 1024
     - 32
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc32.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp1_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp2_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp4_conc64.yaml``
   * - H200_SXM
     - Max Throughput
     - 8192 / 1024
     - 64
     - tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml
     - ``trtllm-serve openai/gpt-oss-120b --extra_llm_api_options tensorrt_llm/configure/database/openai/gpt-oss-120b/H200/gpt_oss_120b_fp4_h200_trt_8k1k_tp8_conc64.yaml``
