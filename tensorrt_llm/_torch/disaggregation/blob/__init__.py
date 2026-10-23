# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

Read top down; a fetch or publish flows ``BlobBackend`` -> ``StoreDriver`` -> store SDK:

- ``backend``: ``BlobBackend``, the public ``base.backend.Backend``. It cuts each view into
  batches and hands them to a ``store.StoreDriver`` on worker threads. ``RegionBinding``
  registers the lender's regions the driver can access, under the lender's hold.
- ``store`` and ``keys``: the private driver boundary, which shares only ``MemoryKind`` with the
  public contract, and the store key encoding. How an object is laid out in a store is its
  driver's private format.

No module imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time.
``README.md`` next to this file says how to write a driver and how to assemble a backend.
"""
