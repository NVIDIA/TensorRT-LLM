# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .deepseekv32_parser import DeepSeekV32Parser


class DeepSeekV41Parser(DeepSeekV32Parser):
    """Tool parser for the DeepSeek V4.1 spaced DSML format.

    V4.1 retains V4's DSML structure but puts a leading space in each tag
    name: `` calls``, `` invoke``, and `` parameter``. Those spaces are part
    of the model's token contract, not cosmetic formatting.
    """

    _CALLS_TAG = " calls"
    _INVOKE_TAG = " invoke"
    _PARAMETER_TAG = " parameter"
