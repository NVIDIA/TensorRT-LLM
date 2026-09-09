.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Cache Initialization Stage
==========================

Cache initialization rewrites attention and recurrent state operations for
cached inference. This stage prepares runtime cache resources such as KV-cache
storage, SSM state, residual hidden-state capture, and model-specific cache
metadata.

.. trtllm_auto_deploy_transform_stage:: cache_init
