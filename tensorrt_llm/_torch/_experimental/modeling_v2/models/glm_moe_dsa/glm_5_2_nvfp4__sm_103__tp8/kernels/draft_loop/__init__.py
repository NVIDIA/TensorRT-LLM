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
"""draft_loop: MTP draft-loop overhead (embedding, bookkeeping, DSA metadata).

Entry points registered with ``dispatch`` (looked up by the call sites only
after ``dispatch.is_enabled("draft_loop")``):

``draft_embedding(native_embedding) -> callable``
    Single-launch stand-in for the native draft-token embedding.
``draft_loop_ops(worker, attn_metadata, spec_metadata) -> DraftLoopOps | None``
    Per-forward fused acceptance, draft-0 preparation, step-0 row gather,
    per-step metadata update and post-loop finalize; None keeps the native
    one-model MTP loop.
``draft_step_update(...)``
    The per-step metadata update alone (used by ``DraftLoopOps``).
"""

from ... import dispatch
from .draft_loop import DraftLoopOps, make_draft_loop_ops
from .draft_step import draft_step_update
from .embedding import DraftEmbedding, draft_embedding, wrap_draft_embedding

dispatch.register("draft_loop", "draft_embedding", wrap_draft_embedding)
dispatch.register("draft_loop", "draft_loop_ops", make_draft_loop_ops)
dispatch.register("draft_loop", "draft_step_update", draft_step_update)

__all__ = [
    "DraftEmbedding",
    "DraftLoopOps",
    "draft_embedding",
    "draft_step_update",
    "make_draft_loop_ops",
    "wrap_draft_embedding",
]
