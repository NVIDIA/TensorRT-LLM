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
"""Manager page tables built around hand-written page layouts."""

from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import DevicePool, ManagerLayout


def manager_layout(layout, pool_bases, kv_heads=0, num_slots=128):
    """The plain-value page table a manager would build around ``layout``.

    Each layer group has one internal layer per global layer, in order; pool ``i`` of pool group
    ``g`` lies at ``pool_bases[g][i]`` with ``num_slots`` slots; attention groups count ``kv_heads``
    heads on the rank.
    """
    groups = layout.layer_groups
    return ManagerLayout(
        page_layout=layout,
        tokens_per_block=layout.tokens_per_page,
        pool_group_of=tuple(g.pool_group for g in groups),
        windows=tuple(g.kept_window for g in groups),
        sink_blocks=tuple(g.num_sink_blocks for g in groups),
        recurrent=tuple(g.kind == "state" for g in groups),
        sparse=tuple(False for _ in groups),
        layers=tuple(g.layers for g in groups),
        shards=tuple((g.shard.count, g.shard.index) for g in groups),
        page_bytes={g: sum(widths) for g, widths in layout.pool_slot_bytes.items()},
        device_pools={
            g: tuple(
                DevicePool(group=g, index=i, base=base, slot_bytes=width, num_slots=num_slots)
                for i, (base, width) in enumerate(zip(pool_bases[g], widths))
            )
            for g, widths in layout.pool_slot_bytes.items()
        },
        named_groups=tuple(True for _ in groups),
        local_layers=tuple(tuple(enumerate(g.layers)) for g in groups),
        kv_heads=tuple(kv_heads if g.kind == "attention" else 0 for g in groups),
    )
