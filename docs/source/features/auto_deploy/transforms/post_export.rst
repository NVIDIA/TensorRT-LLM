.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Post-Export Stage
=================

Post-export transforms remove low-level export artifacts and simple no-op graph
patterns. This keeps later pattern-matching, sharding, and fusion passes focused
on meaningful graph structure.

.. trtllm_auto_deploy_transform_stage:: post_export
