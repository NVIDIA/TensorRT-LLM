# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

Read top down; a fetch or publish flows ``BlobBackend`` -> ``StoreDriver`` -> store SDK:

- ``backend``: ``BlobBackend``, the public ``base.backend.Backend``. It checks each view, cuts it
  into calls and hands each to a ``store.StoreDriver`` as one ``RowBatch``; the driver runs the
  store's work on its own threads and tells the backend of changes through ``on_change``.
  ``RegionBinding`` registers the lender's regions the driver can access, under the lender's hold,
  one by one or, for a ``RegistersRegionsTogether`` driver, as one batch.
- ``store`` and ``keys``: the private driver boundary, which shares only ``MemoryKind`` with the
  public contract, and the store key encoding. How an object is laid out in a store is its
  driver's private format.
- ``mooncake``: ``MooncakeDriver``, a ``StoreDriver`` over a set-up Mooncake SDK client.

No module imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time.
``README.md`` next to this file says how to write a driver and how to assemble a backend.
"""
