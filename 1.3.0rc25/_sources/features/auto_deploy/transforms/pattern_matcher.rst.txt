.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Pattern Matching Stage
======================

Pattern matching canonicalizes model-specific PyTorch graphs into AutoDeploy's
standard graph representation. These transforms identify attention, MoE,
normalization, quantization, activation, and layout patterns before sharding and
post-load fusion run.

.. trtllm_auto_deploy_transform_stage:: pattern_matcher
