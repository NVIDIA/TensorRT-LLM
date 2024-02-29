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
from functools import partial

from ...functional import avg_pool2d, interpolate, silu, view
from ...layers import (AvgPool2d, Conv2d, ConvTranspose2d, GroupNorm, Linear,
                       Mish)
from ...module import Module


class Upsample2D(Module):

    def __init__(self,
                 channels: int,
                 use_conv=False,
                 use_conv_transpose=False,
                 out_channels=None) -> None:
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.use_conv_transpose = use_conv_transpose
        self.use_conv = use_conv
        if self.use_conv_transpose:
            self.conv = ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = Conv2d(self.channels,
                               self.out_channels, (3, 3),
                               padding=(1, 1))
        else:
            self.conv = None

    def forward(self, hidden_states, output_size=None):
        assert not hidden_states.is_dynamic()
        batch, channels, _, _ = hidden_states.size()
        assert channels == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if output_size is None:
            hidden_states = interpolate(hidden_states,
                                        scale_factor=2.0,
                                        mode="nearest")
        else:
            hidden_states = interpolate(hidden_states,
                                        size=output_size,
                                        mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample2D(Module):

    def __init__(self,
                 channels,
                 use_conv=False,
                 out_channels=None,
                 padding=1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = (2, 2)

        if use_conv:
            self.conv = Conv2d(self.channels,
                               self.out_channels, (3, 3),
                               stride=stride,
                               padding=(padding, padding))
        else:
            assert self.channels == self.out_channels
            self.conv = AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states):
        assert not hidden_states.is_dynamic()
        batch, channels, _, _ = hidden_states.size()
        assert channels == self.channels

        #TODO add the missing pad function
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(Module):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = GroupNorm(num_groups=groups,
                               num_channels=in_channels,
                               eps=eps,
                               affine=True)
        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1))

        if temb_channels is not None:
            self.time_emb_proj = Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = GroupNorm(num_groups=groups_out,
                               num_channels=out_channels,
                               eps=eps,
                               affine=True)
        self.conv2 = Conv2d(out_channels,
                            out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1))

        if non_linearity == "swish":
            self.nonlinearity = lambda x: silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = silu

        self.upsample = self.downsample = None
        #TODO (guomingz) add the fir kernel supporting.
        if self.up:
            if kernel == "sde_vp":
                self.upsample = partial(interpolate,
                                        scale_factor=2.0,
                                        mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:

            if kernel == "sde_vp":
                self.downsample = partial(avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels,
                                               use_conv=False,
                                               padding=1,
                                               name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        if self.use_in_shortcut:
            self.conv_shortcut = Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0))
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))
            new_shape = list(temb.size())
            new_shape.extend([1, 1])  #[:,:,None,None] ->view
            hidden_states = hidden_states + view(temb, new_shape)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor +
                         hidden_states) / self.output_scale_factor
        return output_tensor
