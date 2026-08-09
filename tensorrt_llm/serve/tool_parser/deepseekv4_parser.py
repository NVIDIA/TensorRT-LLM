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

from .deepseekv32_parser import DeepSeekV32Parser


class DeepSeekV4Parser(DeepSeekV32Parser):
    """Tool parser for the DeepSeek V4 DSML tool call format."""

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"  # nosec B105
        self.eot_token = "</｜DSML｜tool_calls>"  # nosec B105
