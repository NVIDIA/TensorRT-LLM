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
"""The MegaMoE runner of the MiniMax-M3.1 local backend, and the launches it records.

``local_runner_class()`` is the shared ``Sm100MegaMoENvfp4Runner`` with:

* the M3.1 default tactics (``local_plan.DEFAULT_TACTICS``) ahead of the shared tables;
* the generation-phase FC1 gather index's padding ids set out of range once per local workspace
  (``TRTLLM_M31_DISABLE_MEGAMOE_GATHER_FILL=1`` turns it off);
* the next eager launch after ``record_next_launch`` recorded as a :class:`RecordedLaunch` for
  ``take_recorded_launch``.

The switches are read once, at import, so every rank and every step of a process sees the same setting.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import threading
import weakref
from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from . import local_plan

__all__ = [
    "GATHER_INDEX_OOB_BASE",
    "RecordedLaunch",
    "fill_gather_index_padding",
    "local_runner_class",
    "record_next_launch",
    "recording_live",
    "take_recorded_launch",
]

#: The region of the generation-phase kernels' local workspace that maps FC1 pool rows to source rows.
GATHER_INDEX_REGION = "genphase.token_comm.gather_index"
#: First padding id: out of range for every source plane, so the FC1 gather reads zeros for it.
GATHER_INDEX_OOB_BASE = 1 << 30

_GATHER_FILL_ENABLED = local_plan.megamoe_local_requested(os.environ) and not local_plan.switch_set(
    os.environ, local_plan.GATHER_FILL_DISABLE_ENV
)
# The switches of local_plan.DEFAULT_TACTICS, read once.
_TACTIC_ENVIRON: Dict[str, str] = {
    name: os.environ[name] for name, _ in local_plan.DEFAULT_TACTICS.values() if name in os.environ
}


@dataclasses.dataclass(frozen=True)
class RecordedLaunch:
    """An eager launch: the compiled kernel, its launch arguments, the fenced workspace pairing it ran on and the
    fence record (``fence_refs``) of that pairing's allocations.

    ``keepalive`` holds the launch's input and output tensors, so their memory cannot be reused while the launch is
    recorded. Its workspaces are not held: ``recording_live`` checks that the pairing is still the one fenced for
    the launch.
    """

    compiled: Any
    launch_kwargs: Dict[str, Any]
    fence_key: Tuple
    fence_refs: Any = None
    keepalive: Tuple[torch.Tensor, ...] = ()


def recording_live(launch: RecordedLaunch) -> bool:
    """Whether ``launch`` may be relaunched: its workspace pairing is fenced, both workspaces are the allocations it
    ran on, and no fence ran on the pairing since.

    Every fence stores a new record of the pairing's allocations, so an address reused by a new allocation (which
    is fenced before use) or a re-fence of the same allocations both fail the identity check.
    """
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    return (
        launch.fence_refs is not None
        and megamoe_op._megamoe_fence_key_live(launch.fence_key)
        and megamoe_op._MEGAMOE_FENCED_KEY_WS_REFS.get(launch.fence_key) is launch.fence_refs
    )


# Per thread: ``armed`` while the next eager launch is to be recorded, ``launch`` the one recorded.
_LAST_LAUNCH = threading.local()
# Local workspaces whose gather-index padding ids are set: data pointer -> weakref of the storage.
_FILLED_WORKSPACES: Dict[int, Any] = {}


def record_next_launch() -> None:
    """Record the next eager launch of a runner on this thread (a launch under graph capture records nothing)."""
    _LAST_LAUNCH.armed = True
    _LAST_LAUNCH.launch = None


def take_recorded_launch() -> Optional[RecordedLaunch]:
    """The launch recorded since ``record_next_launch`` on this thread, else None; stops recording."""
    launch = getattr(_LAST_LAUNCH, "launch", None)
    _LAST_LAUNCH.armed = False
    _LAST_LAUNCH.launch = None
    return launch


def fill_gather_index_padding(kernel: Any, local_workspace: torch.Tensor) -> int:
    """Set the gather-index region of ``kernel``'s local workspace to distinct out-of-range row ids.

    The token-comm sort writes only the routed pool rows of the region, but the FC1 gather warps also read the ids
    of each token tile's padding rows. The gather descriptor's row bound turns an out-of-range id into a zero fill
    with no memory access, whereas id 0 (what an allocation may hold) is a valid row that every padding row then
    reads. Padding columns lie outside the dynamic MMA N extent, so no output changes. The region is neither
    zeroed on allocation nor tail-reset, so the ids persist for the workspace's lifetime.

    Returns the number of ids written: 0 when the kernel has no such region.
    """
    workspace = getattr(kernel, "_device_workspace", None)
    if workspace is None:
        return 0
    try:
        offset = int(workspace.offset(GATHER_INDEX_REGION))
        nbytes = int(workspace.nbytes(GATHER_INDEX_REGION))
    except KeyError:
        return 0
    ids = local_workspace[offset : offset + nbytes].view(torch.int32)
    ids.copy_(
        torch.arange(
            GATHER_INDEX_OOB_BASE,
            GATHER_INDEX_OOB_BASE + ids.numel(),
            dtype=torch.int32,
            device=ids.device,
        )
    )
    return ids.numel()


def _workspace_filled(local_workspace: torch.Tensor) -> bool:
    ref = _FILLED_WORKSPACES.get(local_workspace.data_ptr())
    return ref is not None and ref() is not None


@functools.cache
def local_runner_class() -> Optional[type]:
    """The M3.1 MegaMoE runner class, or None when the MegaMoE op is unavailable."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        return None

    class MegaMoELocalRunner(megamoe_op.Sm100MegaMoENvfp4Runner):
        """``Sm100MegaMoENvfp4Runner`` with the M3.1 default tactics, gather-index fill and launch recording."""

        def _default_tactic(self, num_tokens: int) -> Tuple:
            entry = local_plan.default_tactic(self._tactic_lookup_key(), _TACTIC_ENVIRON)
            if entry is None:
                return super()._default_tactic(num_tokens)
            switch, tactic = entry
            logger.info_once(
                f"MegaMoECuteDslLocal: the {self.max_tokens_per_rank}-row bucket at EP{self.world_size} runs "
                f"tactic {tactic} ({switch}=1 turns it off).",
                key=f"megamoe_local_tactic_{self.world_size}_{self.max_tokens_per_rank}",
            )
            return tuple(tactic)

        def _resolve_kernel_descriptor(
            self, tactic: Tuple, *, allow_fallback: bool
        ) -> Tuple[Any, Tuple]:
            kernel, resolved = super()._resolve_kernel_descriptor(
                tactic, allow_fallback=allow_fallback
            )
            if allow_fallback and megamoe_op._unpack_tactic(resolved) != megamoe_op._unpack_tactic(
                tactic
            ):
                logger.warning_once(
                    f"MegaMoECuteDslLocal: the {self.max_tokens_per_rank}-row bucket's default tactic {tactic} "
                    f"did not build; it runs {resolved}.",
                    key=f"megamoe_local_tactic_fallback_{self.world_size}_{self.max_tokens_per_rank}",
                )
            return kernel, resolved

        def _launch_compiled(
            self,
            compiled: Any,
            runtime_kwargs: dict,
            *,
            kernel: Any,
            local_workspace: torch.Tensor,
            fence_key: Tuple,
        ) -> None:
            capturing = torch.cuda.is_current_stream_capturing()
            if _GATHER_FILL_ENABLED and not capturing and not _workspace_filled(local_workspace):
                rows = fill_gather_index_padding(kernel, local_workspace)
                _FILLED_WORKSPACES[local_workspace.data_ptr()] = weakref.ref(
                    local_workspace.untyped_storage()
                )
                if rows:
                    logger.info_once(
                        f"MegaMoECuteDslLocal: the generation-phase gather index pads with out-of-range ids "
                        f"({local_plan.GATHER_FILL_DISABLE_ENV}=1 turns it off).",
                        key="megamoe_local_gather_fill",
                    )
            compiled(**runtime_kwargs)
            if getattr(_LAST_LAUNCH, "armed", False):
                _LAST_LAUNCH.armed = False
                if not capturing:
                    _LAST_LAUNCH.launch = RecordedLaunch(
                        compiled,
                        dict(runtime_kwargs),
                        fence_key,
                        fence_refs=megamoe_op._MEGAMOE_FENCED_KEY_WS_REFS.get(fence_key),
                    )

    return MegaMoELocalRunner
