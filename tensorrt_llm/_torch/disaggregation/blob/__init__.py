# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

No module here imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time.

- ``store`` (the private driver boundary) and ``keys`` import only the standard library. How an
  object is laid out in a store is its driver's private format.
- ``backend`` (the public backend over a driver) imports only ``store``, ``keys`` and
  ``base.backend``, plus numpy and ``tensorrt_llm.logger``.
"""
