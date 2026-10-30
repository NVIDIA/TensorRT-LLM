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
"""deferred_input_prep's executor-loop domain of its store writer, and its debug level.

deferred_input_prep has three parts, each a fast path of the plan
(``dispatch.PART_PATHS``): the staged input preparation (``prep``, the step-input
stager), the sampler store writer (``store``) and the overlap executor loop on the
execution stream (``stream``). ``runtime.make_runtime_hooks`` declares only ``stream``
to the engine; no target serves the other two, which have no engine hook.
``TRTLLM_STAIRCASE_GLM_LL_DISABLE`` turns each served part off.

deferred_input_prep's sampler store writer is validated only in the overlap executor
loop of an executor without attention data parallelism. The target's executor events
declare the overlap loop's attention-DP setting when it starts
(``declare_overlap_loop``); until a loop of this process has declared that domain, and
once any loop has declared another one, the writer keeps the native path
(``loop_domain``). The input preparation stager checks attention DP and the overlap
scheduler itself, per preparation.

``TRTLLM_MODELING_V2_VALIDATE=1`` (``dispatch.validate_mode``) builds the stager
with ``debug_level=1``, the guard: between the start of a staged preparation and
its launch, a torch dispatch mode and a Triton launch hook raise on any operation
whose tensor arguments overlap an already staged destination, and every staged
destination is compared at the launch with its bytes at staging time (raises if
anything wrote it in between). Destination bytes are unchanged, so oracle digests
remain comparable with the native run. ``debug_level=2`` (tests only) adds poison:
every destination that the launch overwrites is filled with ``0xA5`` bytes when it
is staged, so a reader of any kind (also kernels the hooks cannot see) consumes
garbage and the oracle fails.
"""

from typing import Optional

# Executor-loop domain of this process: declared by the overlap loop at its start.
_LOOP_STATE = {"declared": False, "reason": "", "logged": False}
_LOOP_UNDECLARED = (
    "no overlap executor loop declared its domain; the store writer is validated only there"
)
_LOOP_NO_ARGUMENT = "the overlap executor loop did not pass its attention-DP setting"
_LOOP_ATTENTION_DP = "attention DP: the store writer is not validated with it"


def declare_overlap_loop(enable_attention_dp: Optional[bool]) -> None:
    """Record the attention-DP setting of an overlap executor loop that starts.

    ``None``: the caller did not pass it. A declaration outside the validated
    domain is final for the process.
    """
    if enable_attention_dp is None:
        reason = _LOOP_NO_ARGUMENT
    elif enable_attention_dp:
        reason = _LOOP_ATTENTION_DP
    else:
        reason = ""
    _LOOP_STATE["declared"] = True
    if reason and not _LOOP_STATE["reason"]:
        _LOOP_STATE["reason"] = reason


def loop_domain() -> bool:
    """Whether the store writer may engage.

    Logs the first decision of the process (``deferred_input_prep/loop_domain``).
    """
    state = _LOOP_STATE
    reason = state["reason"] if state["declared"] else _LOOP_UNDECLARED
    if not state["logged"]:
        from ...dispatch import log_capability

        state["logged"] = True
        log_capability("deferred_input_prep", "loop_domain", not reason, reason)
    return not reason
