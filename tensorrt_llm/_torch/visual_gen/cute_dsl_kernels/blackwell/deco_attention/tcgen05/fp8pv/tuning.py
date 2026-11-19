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
"""Schedule constants of the tcgen05 attention kernel per compile target (target.arch_name(): "sm_100", "sm_103",
"sm_107"). What follows from an instruction a target has (the QK and PV MMA widths, the reducing TMEM load) is asked
of deco_attention.target where it is used, not listed here."""

from typing import NamedTuple

from ... import target


class Tuning(NamedTuple):
    # The load warp's wait for a free K/V ring slot: 0 waits by test_wait; n > 0 polls try_wait with an n ns
    # nanosleep between probes.
    load_sleep_ns: int
    # The MMA warp probes softmax warpgroup 0's S-empty credit (try_wait) before issuing PV0; without the probe it
    # waits for the credit before QK1.
    s_probe: bool
    # The rescale of O_j (the anchor moved) loops over its 16-column chunks instead of being unrolled into the
    # softmax step.
    rescale_loop: bool
    # The softmax step does the bookkeeping of a means batch (block-sum read-back, zeroing, A-operand store, row-sum
    # add) in one branch on the step after the batch ended.
    vs_batched: bool


# The values of the schedule constants per compile target (fields: Tuning).
TUNING = {
    "sm_100": Tuning(load_sleep_ns=32, s_probe=False, rescale_loop=True, vs_batched=False),
    "sm_103": Tuning(load_sleep_ns=0, s_probe=True, rescale_loop=False, vs_batched=False),
    "sm_107": Tuning(load_sleep_ns=0, s_probe=False, rescale_loop=False, vs_batched=True),
}

# K/V ring slots, each holding K(i) and V(i-1). Three: the means operands of V smoothing follow the ring in shared
# memory and must stay within the address range of a shared-memory descriptor.
KV_STAGE = 3


def tuning():
    """The schedule constants of the current compile target."""
    return TUNING[target.arch_name()]
