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

from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineConfig:
    """
    Pipeline configuration class implemented as a singleton.

    Attributes:
        transformer_type: Type of the model being used
        current_denoising_step: Current iteration step in the denoising process
        current_dit_block_id: Current DiT layer
        num_dit_layers: Total number of DiT layers
        num_inference_steps: Total number of inference steps
    """

    _instance = None
    _initialized = False

    # Class attributes for configuration
    transformer_type = None
    current_denoising_step = None  # current iteration step in the denoising process
    current_dit_block_id = None  # current DiT layer
    # Conditional Free Guidance type, such as "None", "positive", "negative".
    # "None" means no CFG, "positive" means in the positive prompt stage, "negative" means in the negative prompt stage.
    cfg_type = None
    num_dit_layers = None  # total number of DiT layers
    num_inference_steps = None  # total number of inference steps
    seq_len = None  # sequence length
    seq_len_padded = None  # padded sequence length
    seq_len_all_ranks = None  # sequence length all ranks, [seq_rank_0, seq_rank_1, ...]
    seq_len_cur_ring_group = None  # sequence length current ring group, [seq_ring_0, seq_ring_1, ...]
    ulysses_seq_all_ring_ranks = None  # ulysses sequence length all ranks in ring group, [seq_ring_0, seq_ring_1, ...]
    model_wise_offloading = []  # model-wise offloading models, such as ["text_encoder", "image_encoder"]
    block_wise_offloading = []  # block-wise offloading models, such as ["transformer"]
    offloading_stride = 0  # offloading stride for block-wise offloading, if stride is 0, not enabled
    enable_torch_compile = True  # enable torch compile
    torch_compile_models = (
        []
    )  # models to compile with torch compile, such as ["transformer", "image_encoder", "text_encoder"]
    torch_compile_mode = "default"  # torch compile mode, such as "default", "max-autotune"
    fuse_qkv = True
    do_classifier_free_guidance = False
    in_refiner_stage = False  # whether in the refiner stage, some models have refiner, such as HunyuanImage
    int8_ulysses = False  # whether to use 8-bit quantization for Ulysses all-to-all communication
    fuse_qkv_in_ulysses = (
        False  # whether to fuse q, k, v communication into single operation for ulysses parallelization
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure initialization only happens once
        if not self._initialized:
            self.__class__._initialized = True

    @classmethod
    @torch.compiler.disable
    def set_config(cls, **kwargs):
        """
        Set configuration parameters.

        Args:
            transformer_type: Type of the model being used
            current_denoising_step: Current iteration step in the denoising process
            current_dit_block_id: Current DiT layer
            num_inference_steps: Total number of inference steps
            num_dit_layers: Total number of DiT layers
        """
        logger.debug(f"Setting pipeline configuration: {kwargs}")
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def set_uneven_cp_config(cls, seq_len, seq_len_padded, seq_len_cur_rank, ditParallelConfig):

        cls.seq_len = seq_len
        cls.seq_len_padded = seq_len_padded

        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        seq_len_cur_rank = torch.tensor([seq_len_cur_rank], dtype=torch.int32, device=device)
        gather_list = [torch.empty_like(seq_len_cur_rank) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gather_list, seq_len_cur_rank)
        cls.seq_len_all_ranks = torch.cat(gather_list, dim=0).cpu()

        if ditParallelConfig.ulysses_size() == 1:
            cls.seq_len_cur_ring_group = cls.seq_len_all_ranks[torch.tensor(ditParallelConfig.ring_ranks())]
            return
        if ditParallelConfig.ring_size() == 1:
            seq_len_cur_ulysses_group = cls.seq_len_all_ranks[torch.tensor(ditParallelConfig.ulysses_ranks())]
            cls.ulysses_seq_all_ring_ranks = [torch.sum(seq_len_cur_ulysses_group, dtype=torch.int32)]
            return

        cls.seq_len_cur_ring_group = cls.seq_len_all_ranks[torch.tensor(ditParallelConfig.ring_ranks())]
        seq_len_cur_ulysses_group = cls.seq_len_all_ranks[torch.tensor(ditParallelConfig.ulysses_ranks())]
        ulysses_seq_cur_ring_rank = torch.sum(seq_len_cur_ulysses_group, dtype=torch.int32)
        gather_list = [torch.empty(1, dtype=torch.int32, device=device) for _ in range(ditParallelConfig.ring_size())]
        torch.distributed.all_gather(gather_list, ulysses_seq_cur_ring_rank, group=ditParallelConfig.ring_group())
        cls.ulysses_seq_all_ring_ranks = torch.cat(gather_list, dim=0)

    @classmethod
    def get_config(cls):
        """
        Get current configuration as a dictionary.

        Returns:
            dict: Current configuration parameters
        """
        return {
            "transformer_type": cls.transformer_type,
            "current_denoising_step": cls.current_denoising_step,
            "current_dit_block_id": cls.current_dit_block_id,
            "num_inference_steps": cls.num_inference_steps,
            "num_dit_layers": cls.num_dit_layers,
            "seq_len": cls.seq_len,
            "seq_len_padded": cls.seq_len_padded,
            "seq_len_all_ranks": cls.seq_len_all_ranks,
            "seq_len_cur_ring_group": cls.seq_len_cur_ring_group,
            "ulysses_seq_all_ring_ranks": cls.ulysses_seq_all_ring_ranks,
        }

    @classmethod
    def reset(cls):
        """Reset configuration to default values."""
        cls.transformer_type = None
        cls.current_denoising_step = None
        cls.current_dit_block_id = None
        cls.num_inference_steps = None
        cls.num_dit_layers = None
        cls.seq_len = None
        cls.seq_len_padded = None
        cls.seq_len_all_ranks = None
        cls.seq_len_cur_ring_group = None
        cls.ulysses_seq_all_ring_ranks = None

    @classmethod
    def is_initial_state(cls) -> bool:
        """
        Check if the configuration is in its initial state.

        Returns:
            bool: True if all attributes are at their default values, False otherwise
        """
        return (
            cls.transformer_type is None
            and cls.current_denoising_step is None
            and cls.current_dit_block_id is None
            and cls.num_inference_steps is None
            and cls.num_dit_layers is None
            and cls.seq_len is None
            and cls.seq_len_padded is None
            and cls.seq_len_all_ranks is None
            and cls.seq_len_cur_ring_group is None
            and cls.ulysses_seq_all_ring_ranks is None
        )
