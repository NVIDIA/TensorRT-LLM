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
"""Key-sharded decode Top-K merge for the DSA indexer under Helix CP (switchboard field
``dsa_helix_key_shard``: set by the generation profile of a target that asks for it).

Under Helix every CP rank already holds only its own KV pages: global page ``g`` (64 tokens) of a
sequence lives on CP rank ``g % cp`` as that rank's local page ``g // cp``. With the key-sharded
merge on, the Helix decode indexer computes the logits over the local pages into the merge
workspace, takes an exact local Top-L and runs the pack / merge of ``key_shard.py`` over the CP
group instead of ``merge_helix_topk`` (all-gather of every rank's full local Top-K plus a Top-K
over ``cp * K`` candidates per row and layer). The merge writes this rank's winners as local token
indices, which is what the Helix sparse MLA reads.
"""

from tensorrt_llm._torch import decode_switches


def helix_key_shard_enabled() -> bool:
    """Switch of the key-sharded merge under Helix CP (switchboard field ``dsa_helix_key_shard``)."""
    return decode_switches.active().dsa_helix_key_shard
