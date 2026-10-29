# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only schedule selection for GLM local prefill.

This implements the measured prefill policy; it installs no serving hooks,
queries no device data and chooses no router arithmetic split. Callers supply
the actual device SM count and native shared-stream/tracing policy. Unsupported
token counts return to the caller unchanged; supported counts have only CuTe
DSL paths. Compile and pack weights before capture, never inside this selector.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedGeometry:
    """Selected up family and logical CTA demand before persistent-grid capping."""

    family: str
    splitk_ctas: int
    wide_ctas: int
    num_sms: int


def in_prefill_domain(tokens: int) -> bool:
    """Select actual invocation rows, not scheduler batch/context-token sums."""
    if type(tokens) is not int or tokens < 0:
        raise ValueError("nonnegative integer token count required")
    return 65 <= tokens <= 8192


def select_shared_geometry(tokens: int, num_sms: int) -> SharedGeometry:
    """Use actual CTA demand; the two thresholds are not fixed to 80 SMs."""
    if not in_prefill_domain(tokens):
        raise ValueError("shared selector requires T in 65..8192")
    if type(num_sms) is not int or num_sms < 2 or num_sms % 2:
        raise ValueError("positive even actual SM count required")
    # Interleaved gate/value has1024 columns. Split-K uses M64/N64/SK2;
    # the wide two-CTA family has logical M256/N256 and two CTAs per tile.
    splitk_ctas = ((tokens + 63) // 64) * (1024 // 64) * 2
    wide_ctas = ((tokens + 255) // 256) * (1024 // 256) * 2
    family = (
        "splitk_up64_s12_k2"
        if splitk_ctas <= num_sms
        else "cta2_up128_s8"
        if wide_ctas < num_sms
        else "cta2_up256_s6"
    )
    return SharedGeometry(family, splitk_ctas, wide_ctas, num_sms)


def memory_cta_cap(num_sms: int) -> int:
    """Use eight CTA waves for byte-zero and unfused local-add kernels."""
    if type(num_sms) is not int or not 1 <= num_sms <= 4096:
        raise ValueError("actual positive SM count required")
    return 8 * num_sms
