.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Compilation Stage
=================

Compilation is the final transform stage before execution. It applies
runtime-oriented and compiler-oriented changes after graph structure, weights,
and caches are ready, such as multi-stream kernels, final cleanup, and CUDA graph
or ``torch.compile`` execution.

.. trtllm_auto_deploy_transform_stage:: compile
