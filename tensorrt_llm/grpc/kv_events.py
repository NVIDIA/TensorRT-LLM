# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert TensorRT-LLM KV-cache events to the smg-grpc-proto KvEventBatch wire format.

Pure and engine-free (imports only the generated proto): consumes the event
dicts produced by ``LLM.get_kv_cache_events*`` and produces
``common_pb2.KvEventBatch`` messages for the gateway's cache-aware router.
Only ``stored`` and ``removed`` events have a wire equivalent; ``created`` and
``updated`` are skipped.
"""
from typing import Optional

from smg_grpc_proto.generated import common_pb2

_U64_MASK = 0xFFFFFFFFFFFFFFFF
_I64_SIGN_BIT = 0x8000000000000000
_U64_MODULUS = 0x10000000000000000


def to_int64(value: int) -> int:
    """Reduce an unsigned 64-bit hash to a signed int64 for the proto.

    The gateway uses the hash only as a node identity, so a deterministic
    64-bit reduction is safe and must match the other engine bridges.
    """
    masked = int(value) & _U64_MASK
    if masked >= _I64_SIGN_BIT:
        masked -= _U64_MODULUS
    return masked


def convert_event(event: dict, event_id: int) -> Optional[common_pb2.KvCacheEvent]:
    """Convert one TRT-LLM event dict to a proto KvCacheEvent (or None to skip)."""
    data = event.get("data") or {}
    etype = data.get("type")

    if etype == "stored":
        blocks = []
        for block in data.get("blocks", []):
            token_ids = [int(t["token_id"]) for t in block.get("tokens", [])]
            blocks.append(common_pb2.KvBlock(
                block_hash=to_int64(block["block_hash"]),
                token_ids=token_ids,
                block_size=len(token_ids),
            ))
        stored = common_pb2.KvBlocksStored(blocks=blocks)
        parent = data.get("parent_hash")
        if parent is not None:
            stored.parent_block_hash = to_int64(parent)
        return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

    if etype == "removed":
        return common_pb2.KvCacheEvent(
            event_id=event_id,
            removed=common_pb2.KvBlocksRemoved(
                block_hashes=[to_int64(h) for h in data.get("block_hashes", [])]),
        )

    # created / updated / unknown -> no wire equivalent
    return None


def convert_batch(event: dict, seq_num: int) -> Optional[common_pb2.KvEventBatch]:
    """Wrap one converted event in a KvEventBatch, or None if it has no wire form.

    ``seq_num`` is a gateway-facing monotonic counter (the batch sequence
    number); the proto ``event_id`` keeps TRT-LLM's own monotonic id.
    """
    proto_event = convert_event(event, int(event.get("event_id", seq_num)))
    if proto_event is None:
        return None
    batch = common_pb2.KvEventBatch(sequence_number=seq_num, events=[proto_event])
    dp_rank = event.get("attention_dp_rank")
    if dp_rank is not None:
        batch.dp_rank = dp_rank
    return batch
