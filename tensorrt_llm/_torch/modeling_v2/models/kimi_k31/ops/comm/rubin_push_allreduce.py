# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pushed MNNVL all-reduce of the K3.1 Rubin targets: a builder and helpers, not OpWrappers.

A producer may store its bf16 output straight into the Lamport buffers of the MNNVL all-reduce
that follows it; the consumer's kernel, or a finish kernel, then completes the all-reduce bitwise
like the C++ kernel the call replaces. Every kernel launch goes through the entries of
``ops/comm/rubin_lamport_push.py``.

The shared ``AllReduce`` / ``MNNVLAllReduce`` are not edited, so the pushed all-reduce runs on
the shared MNNVL workspace (same dict keys, same 9-word ``buffer_flags`` layout), with its
one-shot threshold.

The targets' attention all-reduce builds its workspace once, with
:func:`rubin_lamport_push_workspace` (defined with the entries in
``ops/comm/rubin_lamport_push.py``, called while the runtime is built): the step then pushes
and finishes on it with no per-call state.
"""

from __future__ import annotations

from tensorrt_llm.mapping import Mapping

from .rubin_lamport_push import LamportPushWs, rubin_lamport_push_workspace


def use_fabric_handle(mapping: Mapping) -> bool:
    """Whether the family's own multicast buffers use fabric handles: across nodes.

    The shared workspace also takes them on one node under ``TRTLLM_FORCE_MNNVL_AR=1``; POSIX FD
    handles are equally valid on one node.
    """
    return mapping.is_multi_node()


__all__ = [
    "LamportPushWs",
    "rubin_lamport_push_workspace",
    "use_fabric_handle",
]
