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
"""CPU transfer-plan contracts; native ownership and GPU I/O are not simulated."""

from dataclasses import replace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.k3_checkpoint import (
    plan_k3_checkpoint,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain

pytestmark = pytest.mark.cpu_only


def region(layer, role, base, size):
    return KvCacheRegion(base, size, size + 16, 8, (KvCacheBufferRef(layer, role),))


@pytest.fixture
def layout():
    # Separate regions exercise scales and mixed payload sizes as opaque bytes.
    return KvCacheLayout(
        4,
        (
            KvCacheLayerGroupLayout(
                0, (0,), None, (region(0, "key", 1000, 64), region(0, "kv_scale", 2000, 16))
            ),
            KvCacheLayerGroupLayout(
                1,
                (1,),
                None,
                (region(1, "ssm_state", 3000, 128), region(1, "conv_state", 5000, 32)),
            ),
        ),
    )


def plan(layout, slots=None, *, tokens=None, **overrides):
    chain = BlockHashChain(4)
    chain.extend(list(range(8)) if tokens is None else tokens)
    args = dict(
        namespace="test",
        representation_id="weights/layout/bf16",
        rank=0,
        world_size=1,
        endpoint_tokens=8,
    )
    args.update(overrides)
    return plan_k3_checkpoint(
        layout, chain, {0: [(0, 1), (1, 2)], 1: [(1, 3)]} if slots is None else slots, **args
    )


def test_complete_plan_includes_attention_scales_recurrent_and_convolution(layout):
    result = plan(layout)
    assert len(result.components) == 3
    assert result.components[0].addresses == (1080, 2032)
    assert result.components[0].sizes == (64, 16)
    assert result.components[2].addresses == (3432, 5144)
    assert result.components[2].sizes == (128, 32)
    assert result.private_keys == (result.components[2].key,)
    assert result.marker.rsplit("/", 1)[1] == result.private_keys[0].rsplit("/", 1)[1]


def test_different_native_addresses_have_identical_identity(layout):
    source = plan(layout)
    moved = replace(
        layout,
        groups=tuple(
            replace(g, regions=tuple(replace(r, base=r.base + 10000) for r in g.regions))
            for g in layout.groups
        ),
    )
    target = plan(moved, {0: [(0, 4), (1, 5)], 1: [(1, 6)]})
    assert source.completion_manifest() == target.completion_manifest()
    assert source.components[0].addresses != target.components[0].addresses
    assert target.matches_manifest(source.completion_manifest())


def test_recurrent_state_is_endpoint_private_attention_is_shared(layout):
    short = plan(layout, {0: [(0, 1)], 1: [(0, 3)]}, endpoint_tokens=4)
    long = plan(layout)
    assert short.components[0].key == long.components[0].key
    assert short.private_keys != long.private_keys
    assert not short.matches_manifest(long.completion_manifest())


@pytest.mark.parametrize(
    "overrides",
    [
        dict(representation_id="other-dtype"),
        dict(rank=1, world_size=2),
        dict(namespace="other-job"),
    ],
)
def test_incompatible_representation_shard_or_job_cannot_reuse_marker(layout, overrides):
    assert not plan(layout).matches_manifest(plan(layout, **overrides).completion_manifest())


def test_different_prefix_does_not_reuse_recurrent_checkpoint(layout):
    assert plan(layout).marker != plan(layout, tokens=[99, *range(1, 8)]).marker


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_partial_restore_cannot_be_complete(layout, missing):
    p = plan(layout)
    results = {c.key: True for c in p.components}
    del results[p.components[missing].key]
    assert not p.all_components_succeeded(results)


@pytest.mark.parametrize("value", [False, None, 1])
def test_failure_or_non_boolean_result_cannot_publish(layout, value):
    p = plan(layout)
    results = {c.key: True for c in p.components}
    results[p.components[-1].key] = value
    assert not p.all_components_succeeded(results)


def test_complete_retired_result_set_and_marker_contract(layout):
    p = plan(layout)
    results = {c.key: True for c in p.components}
    assert p.all_components_succeeded(results)
    results["unexpected-object"] = True
    assert not p.all_components_succeeded(results)
    assert not p.matches_manifest(b"{}")
    assert not p.matches_manifest(p.completion_manifest()[:-1])


@pytest.mark.parametrize(
    "slots",
    [
        {0: [(0, 1)], 1: [(1, 3)]},
        {0: [(0, 1), (1, 2)], 1: [(0, 3)]},
        {0: [(0, 1), (1, 2)]},
        {0: [(0, 1), (1, 1)], 1: [(1, 3)]},
        {0: [(0, 1), (1, 2)], 1: [(1, 3)], 2: []},
    ],
)
def test_incomplete_wrong_endpoint_or_aliasing_pages_rejected(layout, slots):
    with pytest.raises(ValueError):
        plan(layout, slots)


@pytest.mark.parametrize("slot", [-1, 8])
def test_invalid_slot_is_rejected_before_any_address_is_used(layout, slot):
    with pytest.raises(IndexError):
        plan(layout, {0: [(0, 1), (1, 2)], 1: [(1, slot)]})


@pytest.mark.parametrize(
    "roles", [("ssm_state",), ("conv_state",), ("ssm_state", "conv_state", "ple_ngram_context")]
)
def test_incomplete_or_unhandled_recurrent_roles_rejected(layout, roles):
    group = replace(
        layout.groups[1],
        regions=tuple(region(1, role, 3000 + i * 1000, 32) for i, role in enumerate(roles)),
    )
    with pytest.raises(ValueError, match="both recurrent and convolution"):
        plan(replace(layout, groups=(layout.groups[0], group)))


def test_missing_attention_role_rejected(layout):
    group = replace(layout.groups[0], regions=(region(0, "kv_scale", 1000, 16),))
    with pytest.raises(ValueError, match="key payload"):
        plan(replace(layout, groups=(group, layout.groups[1])))


@pytest.mark.parametrize("endpoint", [0, 3, 12, True])
def test_endpoint_must_be_aligned_and_covered(layout, endpoint):
    with pytest.raises(ValueError):
        plan(layout, endpoint_tokens=endpoint)


def test_sliding_window_requires_separate_native_lifecycle_adapter(layout):
    with pytest.raises(ValueError, match="sliding-window"):
        plan(replace(layout, groups=(replace(layout.groups[0], window_size=4), layout.groups[1])))


def test_same_size_different_region_partition_is_not_same_manifest(layout):
    group = replace(
        layout.groups[0], regions=(region(0, "key", 1000, 48), region(0, "kv_scale", 2000, 32))
    )
    other = plan(replace(layout, groups=(group, layout.groups[1])))
    assert not plan(layout).matches_manifest(other.completion_manifest())
