# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from typing import Optional


class KvCacheCompressionMode(IntEnum):
    """Algorithm-level traits of a KV-cache compression method.

    Configs map their ``algorithm`` string to a member here; callers read the
    ``is_*`` predicates instead of comparing strings.
    """

    TRIATTENTION = auto()
    NONE = auto()

    def is_eviction_method(self) -> bool:
        """Return whether this mode physically evicts cached tokens."""
        return self in (KvCacheCompressionMode.TRIATTENTION,)

    @staticmethod
    def from_string(name: Optional[str]) -> "KvCacheCompressionMode":
        if name is None:
            return KvCacheCompressionMode.NONE
        try:
            return KvCacheCompressionMode[name.upper()]
        except KeyError:
            return KvCacheCompressionMode.NONE
