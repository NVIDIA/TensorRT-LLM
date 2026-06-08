.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Factory Stage
=============

Factory transforms create or wrap the starting model object for AutoDeploy. This
stage establishes the module that later graph, weight-loading, cache, and
runtime transforms will optimize.

.. trtllm_auto_deploy_transform_stage:: factory
