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
"""Where the shared expert of a MegaMoE layer runs, relative to the routed launch.

:class:`SharedExpertPlanMixin` gives a ``TrtllmCutedslMegaMoeNvfp4Impl`` subclass the per-step answers the MiniMax-M3.1
DEP target's MoE forward asks (``minimax_m3_1_flash_nvfp4__sm_107__dep/tail.py``): whether the shared expert runs after
the routed launch on the calling stream (``shared_expert_serial``), and where it runs beside it on the aux stream,
whether its MXFP8 linears use the pinned GEMM (``shared_corun_pin``). Each answer is the ``local_plan`` policy on the
step's launch bucket with the knobs read once, when the backend was built, so every step a captured bucket serves gets
the same answer.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from . import local_plan


class SharedExpertPlanMixin:
    """The shared-expert placement of a MegaMoE backend; mix in ahead of ``TrtllmCutedslMegaMoeNvfp4Impl``."""

    def _init_shared_expert_plan(self) -> None:
        """Read the knobs once, validating each, so the model's decisions and the backend's always agree."""
        self._lever_environ = local_plan.lever_environ(os.environ)
        local_plan.shared_serial_max_rows(self._lever_environ)
        local_plan.shared_serial_min_ep_size(self._lever_environ)
        local_plan.shared_corun_pin_above_rows(self._lever_environ)
        local_plan.shared_corun_band(self._lever_environ)

    def megamoe_launch_bucket(self, max_tokens_per_rank: Optional[int]) -> int:
        """The smallest token bucket among the launches of a step whose largest rank has ``max_tokens_per_rank`` rows
        (``None``: one launch at the full bucket).

        A step above the top bucket runs as near-equal chunks, each of at least half the top bucket.
        """
        top = self._maxt_buckets[-1]
        if max_tokens_per_rank is not None and max_tokens_per_rank > top:
            max_tokens_per_rank = (top + 1) // 2
        return self._select_launch_max_tokens(max_tokens_per_rank)

    def shared_expert_serial(self, max_tokens_per_rank: Optional[int]) -> bool:
        """Whether the shared expert of a step whose largest rank has ``max_tokens_per_rank`` rows runs after this
        backend's launches instead of beside them on the aux stream (``local_plan.shared_expert_serial``)."""
        return local_plan.shared_expert_serial(
            self.megamoe_launch_bucket(max_tokens_per_rank), self.ep_size, self._lever_environ
        )

    def shared_corun_pin(self, max_tokens_per_rank: Optional[int]) -> bool:
        """Whether the shared expert of such a step, beside this backend's launches on the aux stream, runs its MXFP8
        linears on the pinned GEMM (``local_plan.shared_corun_pin``)."""
        return local_plan.shared_corun_pin(
            self.megamoe_launch_bucket(max_tokens_per_rank), self._lever_environ
        )

    def shared_corun_pin_kernel(self) -> str:
        """The GEMM a pinned shared expert runs (``local_plan.shared_corun_pin_kernel``)."""
        return local_plan.shared_corun_pin_kernel(self._lever_environ)

    def shared_corun_band(self) -> Optional[Tuple[int, int]]:
        """The co-run band ``(LO, HI)``, rows per rank (``local_plan.shared_corun_band``)."""
        return local_plan.shared_corun_band(self._lever_environ)

    def default_shared_corun_band(self, band: Tuple[int, int]) -> bool:
        """Take ``band`` (``(LO, HI)`` rows per rank) as this backend's co-run band where the knobs it was built with
        set none and do not disable it; returns whether it took it. Must run before any launch is captured."""
        if (
            local_plan.SHARED_CORUN_BAND_ENV in self._lever_environ
            or self._lever_environ.get(local_plan.SHARED_CORUN_BAND_DISABLE_ENV) == "1"
        ):
            return False
        value = f"{band[0]}:{band[1]}"
        local_plan.shared_corun_band({local_plan.SHARED_CORUN_BAND_ENV: value})
        self._lever_environ[local_plan.SHARED_CORUN_BAND_ENV] = value
        return True
