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
"""Row and part names: who shares a name and who never does, including the ranks of a
tensor-parallel group. One full name and one part name are pinned: stored objects stay reachable
only while the bytes stay the same."""

import hashlib

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import (
    KEY_BYTES,
    KEY_FORMAT_VERSION,
    NAME_BYTES,
    Identity,
    namespace,
)

pytestmark = pytest.mark.cpu_only

WHOLE = (1, 0)


def _keys(n, salt=0):
    return np.array(
        [[(salt * 31 + i * 7 + b) % 256 for b in range(KEY_BYTES)] for i in range(n)],
        dtype=np.uint8,
    )


def _identity(scope=b"", layout_id=b"layout", layers=((0,),), shards=None):
    return Identity(scope, layout_id, layers, shards or [WHOLE] * len(layers))


def _names(identity, layer_group=0, n=3, salt=0):
    return [row.tobytes() for row in identity.names(layer_group, _keys(n, salt))]


def _golden():
    return Identity(
        hashlib.sha256(b"golden-compute").digest(),
        hashlib.sha256(b"golden-layout").digest(),
        [(4, 5), (0, 1, 2, 3)],
        [(4, 3), WHOLE],
    )


def test_a_name_is_namespace_then_key_then_the_layer_group_s_identity():
    # Local group 0 holds the largest layer, so it is canonical group 5 of six.
    layers = [(50,), (0,), (10,), (20,), (30,), (40,)]
    identity = _identity(layers=layers, shards=[(4, 3)] + [WHOLE] * 5)
    rows = identity.names(0, _keys(3))
    assert rows.shape == (3, NAME_BYTES) and rows.dtype == np.uint8
    names = [row.tobytes() for row in rows]
    keys = _keys(3)
    for i, name in enumerate(names):
        assert len(name) == NAME_BYTES == 54
        assert name[: len(identity.namespace)] == identity.namespace
        assert name[len(identity.namespace) : -6] == keys[i].tobytes()
        # canonical group 5, shard 3 of 4: big-endian uint16 each
        assert name[-6:] == bytes([0, 5, 0, 4, 0, 3])
    assert len(set(names)) == 3


def test_one_full_name_is_pinned():
    """Any change to these bytes must bump ``KEY_FORMAT_VERSION``, so old objects become misses."""
    key = np.frombuffer(bytes(range(KEY_BYTES)), dtype=np.uint8).reshape(1, KEY_BYTES)
    assert KEY_FORMAT_VERSION == 2
    assert _golden().names(0, key)[0].tobytes().hex() == (
        "ac0cbcea5a240bde17f6bac29386876a"
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "000100040003"
    )


def test_part_names_are_pinned():
    """A part is named by the layout's first 8 bytes and the canonical groups it serves."""
    identity = _golden()
    assert identity.part_name([0]) == "94ede4f3bb6ba356:lg1"
    assert identity.part_name([1]) == "94ede4f3bb6ba356:lg0"
    assert identity.part_name([1, 0, 0]) == "94ede4f3bb6ba356:lg0+1"


def test_part_names_do_not_depend_on_the_scope():
    layers = [(0,), (1,)]
    a = _identity(b"model", layers=layers)
    b = _identity(b"other model", layers=layers)
    assert [a.part_name([g]) for g in range(2)] == [b.part_name([g]) for g in range(2)]
    assert a.namespace != b.namespace


def test_a_replicated_group_is_one_share_of_one():
    assert _names(_identity())[0][-4:] == bytes([0, 1, 0, 0])


def test_instances_that_may_exchange_bytes_compute_the_same_names():
    a = _identity(b"model", layers=[(0,), (1,)])
    b = _identity(b"model", layers=[(0,), (1,)])
    assert _names(a, 1) == _names(b, 1)


def test_local_group_numbering_does_not_leak_into_names():
    # The same canonical groups, numbered the other way round locally, name the same objects.
    a = _identity(layers=[(0,), (1,)])
    b = _identity(layers=[(1,), (0,)])
    assert _names(a, 0) == _names(b, 1) and _names(a, 1) == _names(b, 0)
    assert _names(a, 0) != _names(a, 1)
    assert a.part_name([0]) == b.part_name([1])


@pytest.mark.parametrize("other", [(b"model", b"other-layout"), (b"other-model", b"layout")])
def test_a_different_scope_or_layout_never_collides(other):
    a = _identity(b"model", b"layout")
    b = _identity(*other)
    assert not set(_names(a)) & set(_names(b))


def test_namespace_fields_are_length_prefixed():
    assert namespace(b"a", b"bc") != namespace(b"ab", b"c")


def test_an_unknown_layer_group_or_an_empty_layout_is_rejected():
    with pytest.raises(ValueError, match="unknown layer group"):
        _identity().names(1, _keys(1))
    with pytest.raises(ValueError, match="layout_id"):
        _identity(b"model", b"")
    with pytest.raises(ValueError, match="layout_id"):
        namespace(b"model", b"")


def test_a_scope_longer_than_its_length_prefix_is_rejected():
    assert len(namespace(b"s" * 0xFFFF, b"layout")) == 16
    with pytest.raises(ValueError, match="at most 65535"):
        namespace(b"s" * 0x10000, b"layout")


# -- shards --------------------------------------------------------------------------------------


def test_a_replicated_group_is_named_alike_on_every_rank():
    """Every buffer of the group holds the same bytes on every rank (a single KV head, a
    replicated indexer): the group is whole on each rank, so ranks share its names."""
    ranks = [_identity(b"model", layers=[(0, 1), (2, 3)]) for _ in range(4)]
    assert len({tuple(_names(identity)) for identity in ranks}) == 1


def test_a_head_sharded_group_is_named_per_share():
    layers = [(0, 1), (2, 3)]
    ranks = [_identity(b"model", layers=layers, shards=[(4, r), (4, r)]) for r in range(4)]
    names = [set(_names(identity)) for identity in ranks]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not names[i] & names[j]


def test_groups_of_one_rank_are_judged_one_by_one():
    """A replicated group stays shared across ranks even when another group of the same rank is
    head-sharded."""
    layers = [(0, 1), (2, 3)]
    a = _identity(b"m", layers=layers, shards=[(2, 0), WHOLE])
    b = _identity(b"m", layers=layers, shards=[(2, 1), WHOLE])
    assert _names(a, 1) == _names(b, 1)
    assert not set(_names(a, 0)) & set(_names(b, 0))


@pytest.mark.parametrize("shard", [(2, 2), (0, 0), (2, -1)])
def test_a_shard_names_a_share_of_its_count(shard):
    with pytest.raises(ValueError, match="shard"):
        _identity(shards=[shard])


def test_every_layer_group_has_a_shard():
    with pytest.raises(ValueError, match="shards"):
        Identity(b"", b"layout", [(0,), (1,)], [WHOLE])
