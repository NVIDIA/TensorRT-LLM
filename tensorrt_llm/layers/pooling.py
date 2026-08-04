# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional, Tuple

from ..functional import avg_pool2d
from ..module import Module


class AvgPool2d(Module):

    def __init__(self,
                 kernel_size: Tuple[int],
                 stride: Optional[Tuple[int]] = None,
                 padding: Optional[Tuple[int]] = (0, 0),
                 ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_szie = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return avg_pool2d(input, self.kernel_szie, self.stride, self.padding,
                          self.ceil_mode, self.count_include_pad)
