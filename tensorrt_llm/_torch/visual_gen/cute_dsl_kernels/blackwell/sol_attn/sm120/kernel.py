# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
"""SM120 kernel recipe."""

from .mainloop import SolAttnForwardSm120


def make_kernel(
    *,
    debug_route_trace: bool = False,
    prefetch_first_exact_k: bool = True,
    prefetch_next_route_k: bool = True,
):
    return SolAttnForwardSm120(
        debug_route_trace=debug_route_trace,
        prefetch_first_exact_k=prefetch_first_exact_k,
        prefetch_next_route_k=prefetch_next_route_k,
    )


__all__ = ["make_kernel"]
