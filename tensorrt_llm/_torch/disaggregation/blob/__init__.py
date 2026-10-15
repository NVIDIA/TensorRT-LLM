# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

No module here imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time.

- ``store`` (the private driver boundary) and ``keys`` import only the standard library. How an
  object is laid out in a store is its driver's private format.
- ``backend`` (the public backend over a driver) imports only ``store``, ``keys`` and
  ``base.backend``, plus numpy and ``tensorrt_llm.logger``.
- ``mooncake`` (the Mooncake driver) imports only ``store`` and ``tensorrt_llm.logger``; it takes
  an SDK object already set up.
- ``mooncake_setup`` (reading a store config, opening the SDK) imports only ``mooncake``, plus
  yaml and ``tensorrt_llm.logger``; its ``open_sdk`` is the one place that imports
  ``mooncake.store``, when called.
- ``assembly`` builds a backend for one lender from ``backend`` and a driver opener.
"""
