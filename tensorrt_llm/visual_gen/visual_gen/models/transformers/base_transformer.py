# Adapted from: https://github.com/ali-vilab/TeaCache
# @article{
#   title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
#   author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
#   journal={arXiv preprint arXiv:2411.19108},
#   year={2024}
# }
#
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

import logging

import numpy as np
import torch
import torch.nn as nn

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.parallel import DiTParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.models.utils import check_transformer_blocks

logger = logging.getLogger(__name__)


class ditBaseTransformer(nn.Module):
    """Base transformer class for dit models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-specific states for TeaCache
        self.is_even = True
        self.teacache_coefficients = None
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None

    def _calc_teacache_distance(self, modulated_inp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance between the modulated input and the previous modulated input.
        If the distance is less than the threshold, skip the computation.
        Return:
            should_calc: bool, whether to compute the block
            x: torch.Tensor, the hidden states after the computation
        """
        if not TeaCacheConfig.enable_teacache():
            return True, x

        ret_steps = TeaCacheConfig.ret_steps()
        is_even = True
        cnt_stride = 1
        if PipelineConfig.do_classifier_free_guidance:
            if not ret_steps:
                ret_steps = 2  # must calculate in the first step
            if DiTParallelConfig.cfg_size() == 2:
                is_even = True if DiTParallelConfig.cfg_rank() == 0 else False
                cnt_stride = 2
            else:
                is_even = True if TeaCacheConfig.cnt() % 2 == 0 else False
        else:
            if not ret_steps:
                ret_steps = 1  # must calculate in the first step
        self.is_even = is_even
        if self.is_even:
            if TeaCacheConfig.cnt() < ret_steps or TeaCacheConfig.cnt() >= TeaCacheConfig.cutoff_steps():
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.teacache_coefficients)
                xx = (
                    ((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean())
                    .cpu()
                    .item()
                )
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean())
                    .cpu()
                    .item()
                )

                if self.accumulated_rel_l1_distance_even < TeaCacheConfig.teacache_thresh():
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()
            if not should_calc:
                x += self.previous_residual_even
        else:
            if TeaCacheConfig.cnt() < ret_steps or TeaCacheConfig.cnt() >= TeaCacheConfig.cutoff_steps():
                should_calc = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.teacache_coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean())
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_odd < TeaCacheConfig.teacache_thresh():
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()
            if not should_calc:
                x += self.previous_residual_odd

        TeaCacheConfig.increment_cnt(cnt_stride)
        if TeaCacheConfig.cnt() >= TeaCacheConfig.num_steps():
            TeaCacheConfig.set_cnt(0)

        return should_calc, x

    def _update_teacache_residual(self, block_input: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        if self.is_even:
            self.previous_residual_even = block_output - block_input
        else:
            self.previous_residual_odd = block_output - block_input

    def transformer_block_names(self):
        """
        Return the name of the blocks in the transformer, e.g. ["blocks"] for Wan, ["transformer_blocks", "single_transformer_blocks"] for Flux
        This will guide torch compile and WeightManagedBlocks to work.
        """
        if hasattr(self, "_transformer_block_names"):
            return self._transformer_block_names

        self._transformer_block_names = []
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleList):
                if check_transformer_blocks(child):
                    self._transformer_block_names.append(name)
        return self._transformer_block_names
