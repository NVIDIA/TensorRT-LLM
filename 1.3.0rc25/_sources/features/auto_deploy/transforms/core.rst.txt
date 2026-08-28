.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Core Transform APIs
===================

Common Transform Configuration
------------------------------

Most transforms accept these common fields. Stage pages also show
transform-specific configuration models when a transform extends this base
configuration.

.. autopydantic_model:: tensorrt_llm._torch.auto_deploy.transform.interface.TransformConfig
   :members:
   :show-inheritance:
   :no-index:

Transform Interface
-------------------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.interface
   :members:
   :undoc-members:
   :show-inheritance:

Optimizer
---------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Graph Module Visualizer
-----------------------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.graph_module_visualizer
   :members:
   :undoc-members:
   :show-inheritance:
