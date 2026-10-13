# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .deepseekv32_parser import DeepSeekV32Parser


class DeepSeekV41Parser(DeepSeekV32Parser):
    """Tool parser for the DeepSeek V4.1 spaced DSML format.

    V4.1 retains V4's DSML structure but puts a leading space in each tag
    name: `` calls``, `` invoke``, and `` parameter``. Those spaces are part
    of the model's token contract, not cosmetic formatting.
    """

    _INVOKE_HEADER_PREFIX = '<｜DSML｜ invoke name="'  # nosec B105
    _INVOKE_BEGIN_TEMPLATE = '<｜DSML｜ invoke name="{name}">'  # nosec B105

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = "<｜DSML｜ calls>"  # nosec B105
        self.eot_token = "</｜DSML｜ calls>"  # nosec B105
        self.invoke_begin_regex = r'<｜DSML｜ invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜ invoke>"  # nosec B105
        self.parameter_regex = (
            r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>'
            r"(.*?)</｜DSML｜ parameter>"
        )
