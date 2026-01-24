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

from typing import Iterator, NamedTuple, NewType, TypeAlias, cast

from ._common import SlidingWindowSize
from ._config import KVCacheManagerConfig
from ._utils import TypedIndexList, div_up, typed_enumerate


class LifeCycle(NamedTuple):
    window_size: SlidingWindowSize
    num_sink_blocks: int  # div_up(num_sink_tokens, tokens_per_block)

    @staticmethod
    def make(
        window_size: SlidingWindowSize, num_sink_tokens: int | None, tokens_per_block: int
    ) -> "LifeCycle":
        assert tokens_per_block > 0
        assert window_size is None or window_size > 0
        assert num_sink_tokens is None or num_sink_tokens >= 0
        assert num_sink_tokens in (None, 0) or window_size is not None
        num_sink_blocks = div_up(num_sink_tokens or 0, tokens_per_block)
        return LifeCycle(window_size, num_sink_blocks)


LifeCycleId = NewType("LifeCycleId", int)

# For public exposure
LayerGroupId: TypeAlias = LifeCycleId


class LifeCycleRegistry:
    __slots__ = ("_life_cycle_list", "_life_cycle_id_dict")
    _life_cycle_list: TypedIndexList[LifeCycleId, LifeCycle]
    _life_cycle_id_dict: dict[LifeCycle, LifeCycleId]

    def __init__(self, config: KVCacheManagerConfig) -> None:
        self._life_cycle_list = cast(TypedIndexList[LifeCycleId, LifeCycle], [])
        self._life_cycle_id_dict = dict[LifeCycle, LifeCycleId]()
        for layer in config.layers:
            details = LifeCycle.make(
                layer.window_size, layer.num_sink_tokens, config.tokens_per_block
            )
            if details not in self._life_cycle_id_dict:
                assert len(self._life_cycle_id_dict) == len(self._life_cycle_list), (
                    "corrupted life cycle registry"
                )
                self._life_cycle_list.append(details)
                self._life_cycle_id_dict[details] = LifeCycleId(len(self._life_cycle_list) - 1)

    def get_life_cycle(self, id: LifeCycleId) -> LifeCycle:
        return self._life_cycle_list[id]

    def get_id(self, life_cycle_details: LifeCycle) -> LifeCycleId:
        return self._life_cycle_id_dict[life_cycle_details]

    @property
    def size(self) -> LifeCycleId:
        assert len(self._life_cycle_list) == len(self._life_cycle_id_dict), (
            "corrupted life cycle registry"
        )
        return LifeCycleId(len(self._life_cycle_list))

    def __iter__(self) -> Iterator[LifeCycle]:
        return iter(self._life_cycle_list)

    def __getitem__(self, idx: LifeCycleId) -> LifeCycle:
        return self._life_cycle_list[idx]

    def items(self) -> Iterator[tuple[LifeCycleId, LifeCycle]]:
        return typed_enumerate(self.get())

    def get(self) -> TypedIndexList[LifeCycleId, LifeCycle]:
        return self._life_cycle_list

    def __contains__(self, lc: LifeCycle) -> bool:
        return lc in self._life_cycle_id_dict
