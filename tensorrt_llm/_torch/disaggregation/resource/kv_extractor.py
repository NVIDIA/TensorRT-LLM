# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.cache_specs.page_table import PageTable
from tensorrt_llm._utils import nvtx_range


class KVRegionExtractorV1(RegionExtractorBase):
    """Unit base addresses of a rank's cache, described by its :class:`PageTable`.

    Built from a page table (a peer's) or from a ``KVCacheManagerV2`` (this rank's).
    """

    def __init__(self, kv_arg):
        if isinstance(kv_arg, PageTable):
            self._page_table = kv_arg
        else:
            self._page_table = build_page_table_from_manager(kv_arg)
        self._data_layout = DataLayout.HND

    @property
    def page_table(self) -> PageTable:
        return self._page_table

    @nvtx_range("KVRegionExtractorV1.extract")
    def extract(self, region_ids: np.ndarray, pool_idx: int = 0) -> SpecRegion:
        """One base pointer per valid unit id (pool slot) of pool ``pool_idx``.

        Each ptr = base_address + slot_id * slot_stride_bytes, the start of the unit's slot. Which
        bytes inside the slot move is the transfer group's run plan.
        """
        pool = self._page_table.pools[pool_idx]
        # Filter out invalid ids (BAD_PAGE_INDEX = -1).
        valid = region_ids >= 0
        ptrs = pool.base_address + pool.slot_stride_bytes * region_ids[valid]
        return SpecRegion(memory=MemRegionGroup(ptrs=ptrs, bytes_per_region=pool.slot_bytes))


def build_page_table_from_manager(manager) -> PageTable:
    """The page table of a cache manager; see ``cache_specs.build_page_table``."""
    from tensorrt_llm._torch.disaggregation.cache_specs.build import build_page_table

    return build_page_table(manager)
