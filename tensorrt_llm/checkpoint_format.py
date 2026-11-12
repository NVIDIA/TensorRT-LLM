# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from strenum import StrEnum


class CheckpointFormat(StrEnum):
    """Built-in checkpoint loader formats. Custom registrations can use strings."""

    HF = "HF"
    MX = "MX"
    MISTRAL = "mistral"
    MISTRAL_LARGE_3 = "mistral_large_3"
