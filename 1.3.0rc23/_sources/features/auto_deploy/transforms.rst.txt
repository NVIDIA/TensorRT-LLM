.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

AutoDeploy Transforms
=====================

This section documents the AutoDeploy transform interfaces and registered
pipeline transforms. Use the stage pages to find where a transform runs in the
optimization pipeline, what graph or runtime change it performs, and which
configuration fields are available.

For an overview of how transforms fit into the AutoDeploy pipeline, see
:doc:`auto-deploy`. For information on configuring which transforms run and in
what order, see :doc:`advanced/expert_configurations`.

.. toctree::
   :maxdepth: 1

   transforms/core
   transforms/factory
   transforms/export
   transforms/post_export
   transforms/pattern_matcher
   transforms/sharding
   transforms/weight_load
   transforms/post_load_fusion
   transforms/cache_init
   transforms/visualize
   transforms/compile
   transforms/additional
