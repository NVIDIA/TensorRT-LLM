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
"""Modules whose memory belongs to a label: the loaded model's weights.

:func:`claim_module_memory` records a weak reference to a module; when a
report is built, the module's current parameters and buffers are mapped onto
the allocator's live blocks. It needs no history recording, costs nothing
until a report is built, and follows the module through anything that
replaces its storage -- fused or quantised weights, a weight update, sleep
and wake.
"""

from __future__ import annotations

import itertools
import weakref

import torch

from tensorrt_llm.logger import logger

__all__ = ["claim_module_memory", "claimed_modules", "claimed_storages", "clear_claims"]

#: Modules whose parameters and buffers belong to a label, held weakly.
#: Resolved to storages each time a report is built rather than once at
#: load: an address recorded once keeps its label after the block is freed
#: and reused by an unrelated allocation, while walking the module sees only
#: the storage it holds now -- including after sleep/wake or a weight
#: update has replaced it.
_claimed_modules: list[tuple[weakref.ref[torch.nn.Module], str]] = []


def claim_module_memory(module: torch.nn.Module, label: str) -> bool:
    """Attribute ``module``'s parameters and buffers to ``label``.

    Needs no history recording and does no work beyond taking a weak
    reference; the storages are resolved when a report is built. Recorded
    whether or not the monitor is on, so a report asked for later
    (``LLM.get_memory_report``) still names the weights; holding a weak
    reference costs nothing. Returns whether the claim was recorded.
    """
    try:
        reference = weakref.ref(module)
    except TypeError:
        logger.debug(f"memory monitor: cannot weakly reference {type(module).__name__}")
        return False
    _claimed_modules[:] = [
        (ref, name) for ref, name in _claimed_modules if ref() is not None and ref() is not module
    ]
    _claimed_modules.append((reference, label))
    return True


def claimed_modules() -> list[torch.nn.Module]:
    """Claimed modules that are still alive.

    Lets a provider whose objects hang off the model -- with no registry of
    their own -- walk it without holding it.
    """
    return [module for module in (ref() for ref, _ in list(_claimed_modules)) if module is not None]


def clear_claims() -> None:
    """Forget every claimed module."""
    _claimed_modules.clear()


def claimed_storages() -> list[tuple[int, str]]:
    """``(storage pointer, label)`` for every CUDA tensor a claimed module holds now."""
    found = []
    for reference, label in list(_claimed_modules):
        module = reference()
        if module is None:
            continue
        try:
            tensors = itertools.chain(module.parameters(), module.buffers())
            for tensor in tensors:
                if tensor.is_cuda:
                    found.append((tensor.untyped_storage().data_ptr(), label))
        except (RuntimeError, NotImplementedError) as exc:
            # A tensor subclass without plain storage; skip the module, not the report.
            logger.debug(f"memory monitor: could not walk the {label!r} module ({exc})")
    return found
