# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared masking predicates for PrimTS attention kernels."""


def kv_tile_needs_right_mask(tile_offset_k, tile_size_kv, visible_k_end):
    """Return whether a KV tile crosses a row-visible right bound."""
    return tile_offset_k + tile_size_kv > visible_k_end


def kv_tile_is_fully_visible(
    tile_offset_k,
    tile_size_kv,
    visible_k_begin,
    visible_k_end,
):
    """Return whether a KV tile is inside every row's visible interval.

    All intervals are half-open.  The arguments may be Python integers in
    host-side tests or CuTe DSL integer values while tracing a kernel.
    """
    return (tile_offset_k >= visible_k_begin) & (
        tile_offset_k + tile_size_kv <= visible_k_end
    )
