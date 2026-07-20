# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from typing import Optional


class KvCacheCompressionMode(IntEnum):
    """Algorithm-level traits of a KV-cache compression method.

    Configs map their ``algorithm`` string to a member here; feature gates
    read the ``is_*`` trait predicates (algorithm dispatch itself matches the
    config's ``algorithm`` string).
    """

    TRIATTENTION = auto()
    NONE = auto()

    def is_eviction_method(self):
        """Whether this method physically evicts cached tokens."""
        return self == KvCacheCompressionMode.TRIATTENTION

    @staticmethod
    def from_string(name: Optional[str]) -> "KvCacheCompressionMode":
        if name is None:
            return KvCacheCompressionMode.NONE
        try:
            return KvCacheCompressionMode[name.upper()]
        except KeyError:
            return KvCacheCompressionMode.NONE
