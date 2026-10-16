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
"""Plan complete K3 checkpoints without owning or moving native cache pages.

A producer must supply page indices from an independently pinned exact native
checkpoint, never from a request's advancing recurrent state. This module only
plans byte ranges and validates the completion manifest. Native pinning,
retirement, request adoption and Store transfers belong to their existing
owners; a successful manifest check alone never authorizes cache publication.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .keys import BlockHashChain, KeyNamespace

if TYPE_CHECKING:
    from ..kv_cache_layout import KvCacheLayout


@dataclass(frozen=True)
class CheckpointComponent:
    """A Store object and the ordered byte ranges of one pinned page."""

    key: str
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]
    private: bool


@dataclass(frozen=True)
class K3CheckpointPlan:
    """Full-block checkpoint: complete attention prefix plus exact KDA state."""

    marker: str
    endpoint_tokens: int
    components: tuple[CheckpointComponent, ...]

    @property
    def private_keys(self) -> tuple[str, ...]:
        """Only endpoint-private state belongs in a turn-retention manifest."""
        return tuple(component.key for component in self.components if component.private)

    def completion_manifest(self) -> bytes:
        """Marker payload, published only after every component PUT retires.

        Local addresses/slot numbers are excluded: another owner's allocation
        must reconstruct the same manifest for the same representation/prefix.
        """
        return json.dumps(
            {
                "schema": 1,
                "endpoint_tokens": self.endpoint_tokens,
                "components": [[c.key, list(c.sizes), c.private] for c in self.components],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def matches_manifest(self, payload: bytes) -> bool:
        """Reject a different or incomplete representation before issuing GETs."""
        return payload == self.completion_manifest()

    def all_components_succeeded(self, results: Mapping[str, bool]) -> bool:
        """Require exactly one successful result for every planned object.

        The caller must separately prove all I/O retired before publishing a
        native restore, including when any result is a miss or failure.
        """
        expected = {component.key for component in self.components}
        return set(results) == expected and all(value is True for value in results.values())


def plan_k3_checkpoint(
    layout: "KvCacheLayout",
    chain: BlockHashChain,
    page_indices: Mapping[int, Sequence[tuple[int, int]]],
    *,
    namespace: str,
    representation_id: str,
    rank: int,
    world_size: int,
    endpoint_tokens: int,
) -> K3CheckpointPlan:
    """Describe complete committed K3 state using the existing connector layout.

    ``representation_id`` must identify weights, per-layer dtypes/quantization,
    dimensions and buffer layout, and change whenever any of them changes.
    It must be supplied by the producer's layout identity, not an image name.
    This helper does not duplicate the generic lender's representation hashing.

    Supported here: full attention and KDA groups with ssm_state + conv_state
    for every recurrent layer. Sliding-window/FP4-tail layouts, sparse groups,
    and additional recurrent roles require their own adapters. No unsupported
    serving guard is removed by constructing a plan.
    """
    if not namespace or not representation_id:
        raise ValueError("checkpoint requires namespace and representation identity")
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("invalid checkpoint shard")
    block_size = layout.tokens_per_block
    if block_size <= 0 or chain.tokens_per_block != block_size:
        raise ValueError("checkpoint block geometry does not match its hash chain")
    if type(endpoint_tokens) is not int or endpoint_tokens <= 0 or endpoint_tokens % block_size:
        raise ValueError("checkpoint requires a positive full-block endpoint")
    end_block = endpoint_tokens // block_size
    if len(chain.hashes) < end_block:
        raise ValueError("hash chain does not cover checkpoint endpoint")
    groups = sorted(layout.groups, key=lambda group: group.layer_group_id)
    group_ids = {group.layer_group_id for group in groups}
    if not groups or len(group_ids) != len(groups) or set(page_indices) != group_ids:
        raise ValueError("checkpoint must cover every layout group exactly once")
    layers = [layer for group in groups for layer in group.layer_ids]
    if len(set(layers)) != len(layers):
        raise ValueError("layer belongs to more than one checkpoint group")
    # Hash arbitrary identities so embedded separators cannot alias another key.
    identity = hashlib.sha256(representation_id.encode()).hexdigest()
    prefix = f"{namespace}/k3-v1/{identity}/w{world_size}r{rank}"
    endpoint_hash = hashlib.sha256(chain.hashes[end_block - 1]).hexdigest()
    components = []
    recurrent_count = attention_count = 0
    for group in groups:
        if group.window_size is not None:
            raise ValueError("checkpoint planner does not support sliding-window groups")
        if (
            not group.regions
            or not group.layer_ids
            or len(set(group.layer_ids)) != len(group.layer_ids)
        ):
            raise ValueError("checkpoint group has incomplete layer or region metadata")
        roles: dict[int, set[str]] = {layer: set() for layer in group.layer_ids}
        for region in group.regions:
            if region.size <= 0 or region.stride < region.size or region.num_slots <= 0:
                raise ValueError("invalid checkpoint byte range")
            for buffer in region.buffers:
                if buffer.layer_id not in roles or buffer.role in roles[buffer.layer_id]:
                    raise ValueError("checkpoint roles do not match group layers")
                if buffer.expansion != 1:
                    raise ValueError("checkpoint planner requires unexpanded pages")
                roles[buffer.layer_id].add(buffer.role)
        recurrent = any(
            "ssm_state" in layer_roles or "conv_state" in layer_roles
            for layer_roles in roles.values()
        )
        if recurrent:
            if any(layer_roles != {"ssm_state", "conv_state"} for layer_roles in roles.values()):
                raise ValueError("every KDA layer requires both recurrent and convolution state")
            expected = [end_block - 1]
            recurrent_count += 1
        else:
            if any("key" not in layer_roles for layer_roles in roles.values()):
                raise ValueError("every attention layer requires a key payload")
            expected = list(range(end_block))
            attention_count += 1
        slots = list(page_indices[group.layer_group_id])
        if [ordinal for ordinal, _ in slots] != expected:
            raise ValueError("checkpoint is missing pages or has a different recurrent endpoint")
        if len({slot for _, slot in slots}) != len(slots):
            raise ValueError("distinct attention blocks alias the same page")
        if any(type(slot) is not int for _, slot in slots):
            raise ValueError("checkpoint page slots must be integers")
        ns = KeyNamespace(
            prefix,
            "attention",
            rank,
            world_size,
            group.layer_group_id,
            block_size,
            group.bytes_per_page,
        )
        for ordinal, slot in slots:
            key = (
                f"{prefix}/group/{group.layer_group_id}/{endpoint_hash}"
                if recurrent
                else ns.key(chain.hashes[ordinal])
            )
            components.append(
                CheckpointComponent(
                    key,
                    tuple(region.address_of(slot) for region in group.regions),
                    tuple(region.size for region in group.regions),
                    recurrent,
                )
            )
    if not recurrent_count or not attention_count:
        raise ValueError("K3 checkpoint requires attention and recurrent groups")
    return K3CheckpointPlan(
        f"{prefix}/complete/{endpoint_hash}", endpoint_tokens, tuple(components)
    )
