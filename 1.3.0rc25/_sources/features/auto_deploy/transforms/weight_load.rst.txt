.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Weight Loading Stage
====================

Weight loading materializes model weights and moves required state to the target
device after graph structure and sharding decisions have been made. This stage
bridges graph preparation and weight-dependent fusion.

.. trtllm_auto_deploy_transform_stage:: weight_load
