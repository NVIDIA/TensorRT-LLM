# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

Read top down; a fetch or publish flows ``BlobBackend`` -> ``StoreDriver`` -> store SDK:

- ``backend``: ``BlobBackend``, the public ``base.backend.Backend``. It hands each layer group of
  a view to a ``store.StoreDriver`` as one ``put`` or ``get`` of a ``store.RowBatch`` (split only
  at the driver's ``max_rows_per_op``, when it declares one) and keeps the books; the driver
  queues and runs the store's work on threads of its own. ``RegionBinding`` registers the lender's
  regions the driver can access, under the lender's hold.
- ``store`` and ``keys``: the private driver boundary, which shares only ``MemoryKind`` with the
  public contract, and the store key encoding. How an object is laid out in a store is its
  driver's private format.
- ``mooncake``: ``MooncakeDriver``, a ``StoreDriver`` over a set-up Mooncake SDK client.
- ``mooncake_setup``: reads a store config and opens the SDK client; ``open_mooncake_driver``
  reads the config's ``blob`` section with the caller's reader (``BlobConfig`` for the backend)
  before it opens the SDK, and returns the driver with what the reader returned.

No module imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time; only
``mooncake_setup.open_sdk`` imports ``mooncake.store``, when called. Each module's imports are
checked by ``test_mooncake_setup.py::test_blob_modules_import_only_their_layer``.
"""
