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
from ....functional import allreduce, pow, select, stack
from ....layers import GroupNorm
from ....mapping import Mapping
from ....module import Module


class DistriGroupNorm(Module):

    def __init__(self,
                 module: GroupNorm,
                 mapping: Mapping = Mapping(),
                 is_first_layer: bool = False):
        super().__init__()
        self.mapping = mapping
        self.module = module

    def forward(self, x, *args, **kwargs):
        mapping = self.mapping
        module = self.module
        n, c, h, w = x.shape
        num_groups = module.num_groups
        group_size = c // num_groups

        x = x.view([n, num_groups, group_size, h, w])
        x_mean = x.mean(dim=4, keepdim=True).mean(dim=(3, 2), keepdim=True)
        x2_mean = pow(x, 2.0).mean(dim=4, keepdim=True).mean(dim=(3, 2),
                                                             keepdim=True)
        mean = stack([x_mean, x2_mean], dim=0)
        mean = allreduce(mean, mapping.tp_group)
        mean = mean / (mapping.tp_size * 1.0)
        x_mean = select(mean, 0, 0)
        x2_mean = select(mean, 0, 1)
        var = x2_mean - pow(x_mean, 2.0)
        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + module.eps).sqrt()
        output = (x - x_mean) / std
        output = output.view([n, c, h, w])
        if module.affine:
            output = output * module.weight.value.view([1, -1, 1, 1])
            output = output + module.bias.value.view([1, -1, 1, 1])

        return output
