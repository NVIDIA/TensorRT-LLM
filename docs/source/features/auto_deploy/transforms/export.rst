.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Export Stage
============

Export converts the model into a graph representation that later stages can
inspect and rewrite. After this point, transforms operate on graph structure
rather than only on the original model object.

.. trtllm_auto_deploy_transform_stage:: export
