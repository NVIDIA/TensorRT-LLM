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

from visual_gen.configs.op_manager import AttentionOpManager
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.utils import cp_wrapper, joint_sequence_wrapper, ring_wrapper, ulysses_wrapper
from visual_gen.utils.auto_tuner import get_auto_tuner
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class ditAttnProcessor:
    def __init__(self, force_type: str = None):
        logger.debug("Initializing ditAttnProcessor")
        self.name = None  # module name in the model
        self.attn_impl = None
        self.cached_attn_impl = dict()
        self.force_type = force_type  # force to use a specific attention type

    def select_attn_impl(self, attn_inputs: torch.Tensor = None):
        if self.force_type is not None:
            attn_type = self.force_type
            if attn_type not in self.cached_attn_impl:
                self.cached_attn_impl[attn_type] = AttentionOpManager.get_impl(attn_type)
            self.attn_impl = self.cached_attn_impl[attn_type]
            return
        attn_type = AttentionOpManager.attn_type

        current_denoising_step = PipelineConfig.current_denoising_step
        current_dit_block_id = PipelineConfig.current_dit_block_id

        if attn_type == "auto":
            auto_tuner = get_auto_tuner()
            if auto_tuner is None:
                raise RuntimeError("AutoTuner is not initialized")
            if auto_tuner.mode == "inference":
                layer_name = self.name
                assert layer_name is not None, f"layer_name is not set for {type(self)}"
                assert attn_inputs is not None, "attn_inputs is required for op selection"
                mse_threshold = AttentionOpManager.mse_threshold
                cosine_similarity_threshold = AttentionOpManager.cosine_similarity_threshold
                attn_type = auto_tuner.select_best_impl(
                    step=current_denoising_step,
                    layer_name=layer_name,
                    op_type=AttentionOpManager.op_type(),
                    inputs=attn_inputs,
                    mse_threshold=mse_threshold,
                    cosine_similarity_threshold=cosine_similarity_threshold,
                )
            else:
                assert (
                    auto_tuner.mode == "tuning"
                ), f"Got unexpected auto tuner mode: {auto_tuner.mode}, choices: inference, tuning"
                # No impl is selected for now, we need to tune firstly
                return
        else:
            if AttentionOpManager.num_timesteps_high_precision != 0.0:
                num_inference_steps = PipelineConfig.num_inference_steps
                assert current_denoising_step is not None, "current_denoising_step is not set"
                assert num_inference_steps is not None, "num_inference_steps is not set"
                if current_denoising_step < num_inference_steps * AttentionOpManager.num_timesteps_high_precision:
                    attn_type = AttentionOpManager.high_precision_attn_type
                    logger.debug(
                        f"Force using `{attn_type}` attention for layer: {self.name} at timestep {current_denoising_step}"
                    )
            if AttentionOpManager.num_layers_high_precision != 0.0:
                num_dit_layers = PipelineConfig.num_dit_layers
                assert current_dit_block_id is not None, "current_dit_block_id is not set"
                assert num_dit_layers is not None, "num_dit_layers is not set"
                if current_dit_block_id < num_dit_layers * AttentionOpManager.num_layers_high_precision:
                    attn_type = AttentionOpManager.high_precision_attn_type
                    logger.debug(
                        f"Force using `{attn_type}` attention for layer: {self.name} at layer {current_dit_block_id}"
                    )
        logger.debug(
            f"Selecting attention: `{attn_type}` at timestep: {current_denoising_step}, layer: {current_dit_block_id}"
        )
        if attn_type not in self.cached_attn_impl:
            self.cached_attn_impl[attn_type] = AttentionOpManager.get_impl(attn_type)
        self.attn_impl = self.cached_attn_impl[attn_type]

    @ring_wrapper
    @ulysses_wrapper
    @cp_wrapper
    @joint_sequence_wrapper
    # @torch.cuda.nvtx.range("ditAttn.visual_gen_attn")
    def visual_gen_attn(
        self,
        query,
        key,
        value,
        tensor_layout,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        joint_seq_length=0,
        valid_joint_seq_length=None,
        joint_strategy="none",
    ):
        """
        :param joint_seq_length: The joint sequence length. Joint sequence means the part contians by every rank and not split along sequence dimension.
        :param valid_joint_seq_length: The valid tokens in joint sequence. Its shape should be (B,), record valid number of tokens for each batch. None means all joint sequence tokens are valid.
        :param joint_strategy: The strategy to handle the joint sequence. Choices: `none`, `front`, `rear`, `none` means not joint, `front` and `rear` indicating whether the text is concatenated to the front or rear of the image sequence, respectively.
        """
        auto_tuner = get_auto_tuner()
        attn_inputs = {
            "query": query,
            "key": key,
            "value": value,
            "tensor_layout": tensor_layout,
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,
            "return_lse": return_lse,
        }
        if AttentionOpManager.attn_type == "auto" and auto_tuner is not None and auto_tuner.mode == "tuning":
            layer_name = self.name
            step = PipelineConfig.current_denoising_step
            assert layer_name is not None, f"layer_name is not set for {type(self)}"
            assert step is not None, "current_denoising_step of the pipelineis not set"
            outputs = auto_tuner.tune(
                layer_name=layer_name,
                op_manager=AttentionOpManager,
                baseline_impl="default",
                inputs=attn_inputs,
                step=step,
            )
            return outputs

        self.select_attn_impl(attn_inputs)
        return self.attn_impl(**attn_inputs)
