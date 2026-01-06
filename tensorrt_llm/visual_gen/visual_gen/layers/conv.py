# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.distributed as dist


class ConvParallelStride2(torch.nn.Module):
    def __init__(self, module, chunk_dim, pad_before_conv=(0, 1, 0, 1)):
        super().__init__()

        self.kernel_size = getattr(module, "kernel_size", (3, 3))
        self.padding = getattr(module, "padding", (0, 0))
        assert self.padding == (0, 0), "padding in conv2d is not supported"
        self.stride = (2, 2)  # only support stride 2
        self.chunk_dim = chunk_dim

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        chunk_kernel_size = self.kernel_size[self.chunk_dim - 2]
        d = chunk_kernel_size - 1
        self.padding_left = d // 2
        self.padding_right = d - self.padding_left
        self.padding_flag = self.padding_left

        self.left = pad_before_conv[0]
        self.right = pad_before_conv[1]
        self.top = pad_before_conv[2]
        self.bottom = pad_before_conv[3]
        if chunk_dim == 2:
            self.before_conv_module = torch.nn.ZeroPad2d((self.left, self.right, 0, 0))
            self.last_conv_module = torch.nn.ZeroPad2d((0, 0, self.top, self.bottom))
        elif chunk_dim == 3:
            self.before_conv_module = torch.nn.ZeroPad2d((0, 0, self.top, self.bottom))
            self.last_conv_module = torch.nn.ZeroPad2d((self.left, self.right, 0, 0))
        else:
            raise ValueError(f"chunk_dim {chunk_dim} is not supported")

        self.rank = dist.get_rank()
        self.module = module

    def pad_context(self, h):
        if self.padding_flag == 0:
            return h

        # because stride is 2, so we don't need to send to right
        send_to_left = torch.narrow(h, self.chunk_dim, 0, self.padding_left).contiguous()

        if self.rank != dist.get_world_size() - 1:
            right_context = torch.zeros_like(send_to_left)
            dist.recv(right_context, src=self.rank + 1)
        if self.rank != 0:
            dist.send(send_to_left, dst=self.rank - 1)

        if self.rank != dist.get_world_size() - 1:
            h_with_context = torch.cat([h, right_context], dim=self.chunk_dim)
        else:
            h_with_context = h

        if self.rank == dist.get_world_size() - 1:
            h_with_context = self.last_conv_module(h_with_context)

        return h_with_context

    def forward(self, hidden_states):
        if self.padding_flag == 0:
            return self.module.forward(hidden_states)

        hidden_states = self.before_conv_module(hidden_states)
        hidden_states = self.pad_context(hidden_states)
        hidden_states = self.module.forward(hidden_states)
        return hidden_states


class ConvParallelStride1(torch.nn.Module):
    def __init__(self, module, chunk_dim, adj_groups):
        super().__init__()
        self.kernel_size = getattr(module, "kernel_size", (1, 1, 1))

        if module.__class__.__name__ == "WanCausalConv3d":
            self.pad_width = module._padding[0]
            self.pad_height = module._padding[3]
        elif module.__class__.__name__ == "Conv2d":
            self.pad_width = module.padding[1]
            self.pad_height = module.padding[0]
        else:
            raise ValueError("only WanCausalConv3d and Conv2d are supported")

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)

        self.chunk_dim = chunk_dim

        # chunk_dim from shape [N,C,D,H,W], kernel shape is [D, H, W]
        # chunk_dim from shape [N,C,H,W], kernel shape is [H, W]
        chunk_kernel_size = self.kernel_size[self.chunk_dim - 2]

        if module.__class__.__name__ == "WanCausalConv3d":
            assert self.pad_width == self.kernel_size[2] // 2
            assert self.pad_height == self.kernel_size[1] // 2
        elif module.__class__.__name__ == "Conv2d":
            assert self.pad_width == self.kernel_size[1] // 2
            assert self.pad_height == self.kernel_size[0] // 2
        else:
            raise ValueError("only WanCausalConv3d and Conv2d are supported")

        d = chunk_kernel_size - 1
        self.padding_left = d // 2
        self.padding_right = d - self.padding_left
        self.padding_flag = self.padding_left

        self.rank = dist.get_rank()
        self.module = module
        self.adj_groups = adj_groups

    def pad_context(self, h):
        if self.padding_flag == 0:
            return h

        send_to_left = torch.narrow(h, self.chunk_dim, 0, self.padding_left).contiguous()
        send_to_right = torch.narrow(
            h, self.chunk_dim, h.shape[self.chunk_dim] - self.padding_right, self.padding_right
        ).contiguous()

        recv_from_left = torch.zeros_like(send_to_right)
        recv_from_right = torch.zeros_like(send_to_left)

        # do all gather between rank and rank+1 to share padding context to rank and rank + 1 at same time.
        # do two all gather in total:
        # first all gather:
        # for rank % 2 == 0, send send_to_left to rank-1, recv recv_from_left from rank-1
        # for rank % 2 == 1, send send_to_right to rank+1, recv recv_from_right from rank+1
        # second all gather:
        # for rank % 2 == 0, send send_to_right to rank+1, recv recv_from_right from rank+1
        # for rank % 2 == 1, send send_to_left to rank-1, recv recv_from_left from rank-1

        if self.rank % 2 == 0:
            # first all gather:
            if self.rank != 0:
                # not the first rank, have left context
                padding_list = [recv_from_left, send_to_left]
                dist.all_gather(padding_list, send_to_left, group=self.adj_groups[self.rank - 1])
            # second all gather:
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [send_to_right, recv_from_right]
                dist.all_gather(padding_list, send_to_right, group=self.adj_groups[self.rank])
        else:
            # first all gather:
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [send_to_right, recv_from_right]
                dist.all_gather(padding_list, send_to_right, group=self.adj_groups[self.rank])
            # second all gather:
            if self.rank != 0:
                # not the first rank, have left context
                padding_list = [recv_from_left, send_to_left]
                dist.all_gather(padding_list, send_to_left, group=self.adj_groups[self.rank - 1])

        h_with_context = torch.cat([recv_from_left, h, recv_from_right], dim=self.chunk_dim)
        return h_with_context

    def forward(self, hidden_states, cache_x=None, *args, **kwargs):
        if self.padding_flag == 0:
            print("padding=0, return old_forward")
            return self.module(hidden_states, cache_x, *args, **kwargs)

        hidden_states = self.pad_context(hidden_states)
        if cache_x is not None:
            cache_x = self.pad_context(cache_x)

        if self.module.__class__.__name__ == "WanCausalConv3d":
            result = self.module(hidden_states, cache_x, *args, **kwargs)
        elif self.module.__class__.__name__ == "Conv2d":
            result = self.module(hidden_states, *args, **kwargs)
        else:
            raise ValueError("only WanCausalConv3d and Conv2d are supported")

        # Determine padding and dimension based on chunk_dim
        pad_length = self.pad_height if self.chunk_dim == 3 else self.pad_width

        # Remove padding from the result
        start_index = pad_length
        end_index = result.shape[self.chunk_dim] - pad_length
        result = result.narrow(self.chunk_dim, start_index, end_index - start_index)

        return result
