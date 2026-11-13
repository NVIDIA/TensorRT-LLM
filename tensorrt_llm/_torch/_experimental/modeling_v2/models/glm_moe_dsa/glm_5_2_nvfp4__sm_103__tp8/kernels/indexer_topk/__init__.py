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
"""indexer_topk: indexer logits and top-k selection (the index_selection contract), GEN target full layers.

Registered entry points (``dispatch.get_impl("indexer_topk", name)``):

- ``indexer_decode``: :func:`indexer_decode.try_indexer_decode`, the adapter
  hook for the generation branch of ``Indexer.sparse_attn_indexer``.
- ``decode_topk``: :func:`indexer_decode.decode_topk`, the tensor-level entry
  used by tests and microbenchmarks.
- ``cp_indexer_decode``: :func:`cross_rank_indexer_decode.try_cp_indexer_decode`, the
  adapter hook of the cross_rank_topk context-partitioned variant (selected with
  ``cross_rank_topk``).
- ``cp_decode_topk``: :func:`cross_rank_indexer_decode.cp_decode_topk`, its
  tensor-level entry.
"""

from ...dispatch import register
from .cross_rank_indexer_decode import cp_decode_topk, try_cp_indexer_decode
from .indexer_decode import decode_topk, paged_mqa_logits, try_indexer_decode

register("indexer_topk", "indexer_decode", try_indexer_decode)
register("indexer_topk", "decode_topk", decode_topk)
register("indexer_topk", "cp_indexer_decode", try_cp_indexer_decode)
register("indexer_topk", "cp_decode_topk", cp_decode_topk)

__all__ = [
    "cp_decode_topk",
    "decode_topk",
    "paged_mqa_logits",
    "try_cp_indexer_decode",
    "try_indexer_decode",
]
