# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


class PPCommTag(IntEnum):
    """
    Unique tags for pipeline parallelism communication.
    """

    TERMINATION = 20000
    SCHEDULE_RESULT = 20001
    EXECUTED_BATCH_NUM = 20002
    SAMPLE_STATE = 20003
