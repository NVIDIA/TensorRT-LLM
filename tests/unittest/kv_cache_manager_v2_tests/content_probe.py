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
"""One backend's answers to the content-named entry points, as plain data.

Importable under the loaded backend and runnable as a script under the other;
the caller diffs the two dicts. Only what the shared surface promises.
"""

from __future__ import annotations

import json
import os
import sys

TOKENS_PER_BLOCK = 8
NUM_BLOCKS = 3
GPU_QUOTA = 16 << 20
HOST_QUOTA = 16 << 20
DISK_QUOTA = 16 << 20
# Well-formed, right-width, and in no tree: the shape of a stale router hint.
ABSENT_KEY = bytes(range(32))


def _disk_path() -> str:
    for candidate in ("/workspace/", "/tmp/nvidia-mps/", "/tmp"):
        # Writable, as the message below already claims: a disk tier opens files
        # here while the manager is built, so mere existence fails later instead.
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    raise RuntimeError("no writable directory for a disk cache tier")


def _config():
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
    )

    # Three tiers so every answer pool_group_descs_at can give is reachable:
    # device memory, colder memory, and a level that is not memory at all.
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[
            GpuCacheTierConfig(quota=GPU_QUOTA),
            HostCacheTierConfig(quota=HOST_QUOTA),
            DiskCacheTierConfig(quota=DISK_QUOTA, path=_disk_path()),
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(layer_id),
                buffers=[
                    BufferConfig(role="key", size=8192),
                    BufferConfig(role="value", size=8192),
                ],
            )
            for layer_id in range(2)
        ],
    )


def commit_chain(manager, tokens) -> None:
    """Prefill and commit ``tokens``, leaving the blocks in the tree for reuse."""
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import TemporaryCudaStream

    with TemporaryCudaStream([]) as stream_ctx:
        kv = manager.create_kv_cache(None, tokens)
        assert kv.resume(stream_ctx.handle)
        assert kv.resize(len(tokens))
        already = kv.num_committed_tokens
        if already < len(tokens):
            kv.commit(tokens[already:])
        kv.stop_committing()
        kv.close()
    stream_ctx.take_finish_event().synchronize()


def chain_keys(tokens):
    """The root key followed by one key per whole block, as a holder receives it."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, sequence_to_blockchain_keys

    return [key for _, key in sequence_to_blockchain_keys(TOKENS_PER_BLOCK, ReuseScope(), tokens)]


def _servable(manager, keys, life_cycles):
    found = manager.servable_chain(list(keys), list(life_cycles))
    return None if found is None else [int(found[0]), int(found[1])]


class KeyLike(bytes):
    """A bytes subclass, which both backends must treat as bytes."""


def _refusal(fn, *args) -> str:
    """What a shared entry point does with a name it must not answer.

    Recorded rather than asserted: refusing differently is the same defect one
    level down as answering differently.
    """
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 -- the type is the answer
        return f"raised:{type(exc).__name__}"
    if hasattr(result, "close"):
        result.close()
        return "held"
    return f"answered:{result!r}"


def _held_page_counts(manager, keys, life_cycles):
    """How many pages a content-named cache actually holds, per life cycle."""
    held = manager.create_kv_cache_from_keys(list(keys))
    try:
        return [
            len(list(held.get_aggregated_page_indices(lc, valid_only=True))) for lc in life_cycles
        ]
    finally:
        held.close()


def _pool_shape(descs):
    """The part of a pool-group layout that is not this backend's own business."""
    if descs is None:
        return None
    return [len(list(desc.pools)) for desc in descs]


def _pool_group_answers(manager):
    # Imported here, like every other manager import in this module: the backend
    # is fixed at import time, so nothing may be pulled in at module scope.
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    tiers = list(manager.cache_tier_list)
    answers = {
        tier.name: _pool_shape(_introspection.pool_group_descs_at(manager, level))
        for level, tier in enumerate(tiers)
    }
    # A level this build does not have is not memory either, at both ends since
    # the level arrives as a plain int. Recorded rather than asserted: None
    # versus a raise is exactly what the two backends have to agree on.
    for name, level in (("beyond_last", len(tiers)), ("before_first", -1)):
        try:
            answers[name] = _pool_shape(_introspection.pool_group_descs_at(manager, level))
        except (IndexError, KeyError, OverflowError) as exc:
            answers[name] = f"raised:{type(exc).__name__}"
    return answers


def collect() -> dict:
    """Build a manager, name a chain in it, and record every answer."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager, TokenId, _introspection
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once

    init_cuda_once()
    manager = KVCacheManager(_config())
    try:
        tokens = [TokenId(i) for i in range(NUM_BLOCKS * TOKENS_PER_BLOCK)]
        commit_chain(manager, tokens)
        keys = chain_keys(tokens)
        # Same namespace, different content: the root resolves and nothing under it does.
        missing = chain_keys([TokenId(9000 + i) for i in range(NUM_BLOCKS * TOKENS_PER_BLOCK)])
        life_cycles = list(_introspection.attention_life_cycle_ids(manager))

        try:
            unnamed = manager.create_kv_cache_from_keys([])
        except (ValueError, RuntimeError) as exc:
            empty_chain_refusal = type(exc).__name__
        else:
            unnamed.close()
            empty_chain_refusal = None

        # Ends part-way into a block: stop_committing commits that last block at
        # its real coverage, so the tail names a block that is not whole.
        partial = [TokenId(5000 + i) for i in range(NUM_BLOCKS * TOKENS_PER_BLOCK + 2)]
        commit_chain(manager, partial)
        partial_keys = chain_keys(partial)
        num_life_cycles = len(_introspection.life_cycle_pool_group_indices(manager))

        held = manager.create_kv_cache_from_keys(list(keys))
        try:
            held_drop_plan = _refusal(held.plan_committed_block_drop)
        finally:
            held.close()

        hot = _introspection.pool_group_descs_at(manager, 0)
        return {
            "tokens_per_block": int(manager.tokens_per_block),
            "cache_tiers": [tier.name for tier in manager.cache_tier_list],
            "life_cycles": life_cycles,
            "keys": [key.hex() for key in keys],
            "servable": {
                "empty": _servable(manager, [], []),
                "whole": _servable(manager, keys, life_cycles),
                "prefix": _servable(manager, keys[:-1], life_cycles),
                "root_only": _servable(manager, keys[:1], life_cycles),
                "forked": _servable(manager, keys[:-1] + [ABSENT_KEY], life_cycles),
                # The same fork named by a bytes subclass: PyBytes_Check takes
                # one, so an exact type check on either side would refuse a name
                # the other answers.
                "forked_subclass": _servable(
                    manager, keys[:-1] + [KeyLike(ABSENT_KEY)], life_cycles
                ),
                "missing": _servable(manager, missing, life_cycles),
                "no_life_cycles": _servable(manager, keys, []),
                "partial_whole_prefix": _servable(manager, partial_keys[:-1], life_cycles),
                "partial_block": _servable(manager, partial_keys, life_cycles),
            },
            # Every one of these must be a refusal, and the same refusal: a
            # guard only one backend has is a peer reading a fraction of a
            # block, or an unowned page, as if it were content.
            "refusals": {
                "life_cycle_below_first": _refusal(manager.servable_chain, list(keys), [-1]),
                "life_cycle_past_last": _refusal(
                    manager.servable_chain, list(keys), [num_life_cycles]
                ),
                "short_key": _refusal(manager.servable_chain, [ABSENT_KEY[:31]], life_cycles),
                "long_key": _refusal(manager.servable_chain, [ABSENT_KEY + b"\x00"], life_cycles),
                "bytearray_key": _refusal(
                    manager.servable_chain, [bytearray(ABSENT_KEY)], life_cycles
                ),
                "str_key": _refusal(manager.servable_chain, [ABSENT_KEY.hex()], life_cycles),
                "short_key_held": _refusal(manager.create_kv_cache_from_keys, [ABSENT_KEY[:31]]),
                "partial_block_held": _refusal(
                    manager.create_kv_cache_from_keys, list(partial_keys)
                ),
            },
            "held_drop_plan": held_drop_plan,
            "held_page_counts": {
                "whole": _held_page_counts(manager, keys, life_cycles),
                "missing": _held_page_counts(manager, missing, life_cycles),
            },
            "empty_chain_refusal": empty_chain_refusal,
            "pool_groups": _pool_group_answers(manager),
            # The hot level is memory by construction, so this is the layout
            # already published -- said as a call rather than a second copy.
            "hot_matches_published": _pool_shape(hot)
            == _pool_shape(list(manager.pool_group_descs)),
        }
    finally:
        manager.clear_reusable_blocks()
        manager.shutdown()


if __name__ == "__main__":
    # Written where the caller asked rather than to stdout, which already carries
    # tensorrt_llm's version banner. Parsing around it would be a filter that
    # rots the first time an import adds a line; a file of our own cannot.
    _out_path = sys.argv[1]
    try:
        _answers = collect()
    except ImportError as exc:
        # This backend is not built here at all, which is a different fact from
        # it answering differently, and must not read as one.
        _answers = {"unavailable": f"{type(exc).__name__}: {exc}"}
    with open(_out_path, "w", encoding="utf-8") as _fh:
        json.dump(_answers, _fh)
