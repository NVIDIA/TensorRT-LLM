# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What each CSA2 compute provider does, for the module's per-forward choices.

``DeepseekV41Attention`` asks these capabilities instead of the provider's name. The native
trtllm-gen provider (``NATIVE_COMPUTE``) attends through ``CSA2TrtllmAttention.forward``; every other
provider is a compute helper (``CSA2PrimsTS``, ``CSA2FlashMLA``, ``CSA2FlashInfer``) that also
attends the module's tiles through ``forward_context`` and ``forward_generation``.
"""

from __future__ import annotations

from typing import ClassVar

import torch


class CSA2ComputeCapabilities:
    """A provider's capabilities; a provider declares those it has."""

    # ``attend_rows`` over the projected BF16 query rows reproduces ``forward`` (pass graphs).
    supports_row_split: ClassVar[bool] = False
    # Generation steps the backend admits take the direct sparse decode (``direct_decode.py``).
    serves_direct_decode: ClassVar[bool] = False
    # A decode layer with final selections compacts them on a side stream beside the projections.
    prepares_early_selection: ClassVar[bool] = False
    # The attention waits for a query projected on the auxiliary stream (``query_ready``), so
    # Q and KV projections can overlap.
    waits_query_ready: ClassVar[bool] = False
    # The attention reads the projection's unit-scale E4M3 query as is.
    reads_e4m3_query: ClassVar[bool] = False
    # The attention can write the grouped MXFP8 O-A input of context and generation tiles.
    takes_mxfp8_output: ClassVar[bool] = False

    def writes_mxfp8_output(self, device: torch.device, rows: int, width: int) -> bool:
        """Whether a context-only or generation-only tile of ``rows`` rows and sparse ``width``
        writes the grouped MXFP8 O-A input it is offered, and then no BF16 output."""
        return False


class _NativeCompute(CSA2ComputeCapabilities):
    """The native trtllm-gen provider."""

    supports_row_split = True
    serves_direct_decode = True
    waits_query_ready = True


NATIVE_COMPUTE = _NativeCompute()
