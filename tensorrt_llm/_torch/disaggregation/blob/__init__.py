# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

So far this package holds the private driver boundary (``store.py``) and the key encoding
(``keys.py``). Nothing here imports an optional store SDK at import time, and the only thing
shared with the public backend contract is ``MemoryKind``.
"""
