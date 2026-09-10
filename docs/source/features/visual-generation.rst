.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Visual Generation (Beta)
========================

This section documents optimization features for VisualGen image and video
generation. Use the guides below for configuration, supported options, and
interactions between features.

For supported models and an overview of VisualGen, see
:doc:`../models/visual-generation`. For usage examples, see
:doc:`../examples/visual_gen_examples`.

.. toctree::
   :maxdepth: 1

   CUDA Graphs <visualgen-cuda-graph>
   Quantized Attention <visualgen-quantized-attention>
   Sparse Attention <visualgen-sparse-attention>

Related technical blogs
-----------------------

These articles explain the design choices and benchmark results behind
VisualGen optimizations:

* :doc:`Scaling video generation across NVL72 <../blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM>`
  covers multi-GPU parallelism and scaling results.
* :doc:`Quantization and sparse attention for video generation <../blogs/tech_blog/blog28_Accelerating_Video_Generation_with_GEMM_Quantization_Attention_Quantization_and_Skip_Softmax_Attention_in_TensorRT-LLM>`
  covers GEMM quantization, quantized attention, and Skip Softmax Attention,
  including performance and visual-quality trade-offs.
