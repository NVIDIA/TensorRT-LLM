.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Sharding Stage
==============

Sharding determines and applies distributed execution layout. These transforms
identify tensor, expert, and batch-matmul sharding choices, then apply graph
rewrites and communication hints needed for multi-rank execution.

.. trtllm_auto_deploy_transform_stage:: sharding
