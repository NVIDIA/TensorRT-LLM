.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Post-Load Fusion Stage
======================

Post-load fusion applies performance optimizations that need loaded weights,
device tensors, or the final post-sharding graph structure. This stage includes
kernel fusions for quantized linear layers, MoE, normalization, activation, RoPE,
and related inference patterns.

.. trtllm_auto_deploy_transform_stage:: post_load_fusion
