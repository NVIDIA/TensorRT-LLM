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
"""decode_prologue: the target decode-metadata prologue and the target embedding in one launch.

decode_prologue extends draft_loop (``draft_loop``): its launch reuses draft_loop's DeepGEMM
schedule reproduction (``draft_loop/draft_step.py``) and draft_loop's embedding semantics
(``draft_loop/embedding.py``) for the target step instead of the draft steps.

Entry point: :class:`prologue.DecodePrologueHook`, which ``runtime.make_runtime_hooks``
declares to the engine as the step loop's ``decode_prologue`` and hands to the decode
target. Calling it runs :func:`prologue.target_prologue` at the start of
``DecoderInputPreparer.preprocess_inputs`` (True when it replaced the native prologue, False
to keep the native code); its ``take_embedding`` (:func:`prologue.take_embedding`)
returns the rows the launch embedded to the decode target's forward, or None.
"""

from ...dispatch import register
from .prologue import DecodePrologueHook, prologue_ineligibility, take_embedding, target_prologue

register("decode_prologue", "target_prologue", target_prologue)
register("decode_prologue", "take_embedding", take_embedding)

__all__ = ["DecodePrologueHook", "prologue_ineligibility", "take_embedding", "target_prologue"]
