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


import os
import pprint

import torch

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import AttentionOpManager, LinearOpManager, SparseVideogenConfig, SparseVideogenConfig2
from visual_gen.configs.parallel import (
    DiTParallelConfig,
    RefinerDiTParallelConfig,
    T5ParallelConfig,
    VAEParallelConfig,
    init_dist,
)
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def setup_pipeline_config(**pipe_config) -> None:
    """Setup pipeline configuration"""
    logger.debug(f"Pipeline configuration: {pipe_config}")
    PipelineConfig.set_config(**pipe_config)


def setup_attn_config(**attn_config) -> None:
    """Setup attention configuration"""
    logger.info(f"Attention configuration: {attn_config}")
    AttentionOpManager.set_attn_config(
        attn_type=attn_config["type"],
        attn_choices=attn_config.get("choices", ["default", "sage-attn"]),
        high_precision_attn_type=attn_config.get("high_precision_attn_type", "default"),
        num_timesteps_high_precision=attn_config.get("num_timesteps_high_precision", 0.0),
        num_layers_high_precision=attn_config.get("num_layers_high_precision", 0.0),
        cosine_similarity_threshold=attn_config.get("cosine_similarity_threshold", None),
        mse_threshold=attn_config.get("mse_threshold", None),
        record_io_tensors=attn_config.get("record_io_tensors", False),
    )
    if attn_config["type"] in ["sparse-videogen", "auto"]:
        svg_config = {
            "num_sampled_rows": attn_config.get("num_sampled_rows", 64),
            "sample_mse_max_row": attn_config.get("sample_mse_max_row", 10000),
            "sparsity": attn_config.get("sparsity", 0.25),
        }
        SparseVideogenConfig.update(**svg_config)

    if attn_config["type"] == "sparse-videogen2":
        svg2_config = {
            "num_q_centroids": attn_config.get("num_q_centroids", 100),
            "num_k_centroids": attn_config.get("num_k_centroids", 500),
            "top_p_kmeans": attn_config.get("top_p_kmeans", 0.9),
            "min_kc_ratio": attn_config.get("min_kc_ratio", 0.1),
            "kmeans_iter_init": attn_config.get("kmeans_iter_init", 50),
            "kmeans_iter_step": attn_config.get("kmeans_iter_step", 2),
        }
        SparseVideogenConfig2.update(**svg2_config)


def setup_linear_config(**linear_config) -> None:
    """Setup linear configuration"""
    logger.info(f"Linear configuration: {linear_config}")
    LinearOpManager.set_linear_config(
        linear_type=linear_config["type"],
        linear_recipe=linear_config["recipe"],
        linear_choices=linear_config.get("choices", ["default"]),
        cosine_similarity_threshold=linear_config.get("cosine_similarity_threshold", None),
        mse_threshold=linear_config.get("mse_threshold", None),
        record_io_tensors=linear_config.get("record_io_tensors", False),
    )


def setup_teacache_config(**teacache_config) -> None:
    enable_teacache = teacache_config.get("enable_teacache", False)
    if not enable_teacache:
        logger.debug("TeaCache disabled")
        return
    use_ret_steps = teacache_config.get("use_ret_steps", True)
    teacache_thresh = teacache_config.get("teacache_thresh", 0.2)
    TeaCacheConfig.set_config(
        enable_teacache=enable_teacache,
        use_ret_steps=use_ret_steps,
        teacache_thresh=teacache_thresh,
    )
    logger.info(f"TeaCache config: {teacache_config}")


def setup_parallel_config(**parallel_config) -> None:
    """Setup parallel configuration

    kwargs for parallel configuration:
        dit_dp_size: Data parallel size for DiT
        dit_tp_size: Tensor parallel size for DiT
        dit_ulysses_size: Ulysses parallel size for DiT
        dit_ring_size: Ring parallel size for DiT
        dit_cp_size: Context parallel size for DiT
        dit_cfg_size: Classifier-free guidance size for DiT
    """
    logger.debug("Setting up parallel configuration")

    init_dist(device_type="cuda")

    is_parallel = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    logger.debug(f"Parallel configuration: {parallel_config}")

    DiTParallelConfig.set_config(
        dp_size=parallel_config.get("dit_dp_size", 1),
        tp_size=parallel_config.get("dit_tp_size", 1),
        ulysses_size=parallel_config.get("dit_ulysses_size", 1),
        ring_size=parallel_config.get("dit_ring_size", 1),
        cp_size=parallel_config.get("dit_cp_size", 1),
        cfg_size=parallel_config.get("dit_cfg_size", 1),
        fsdp_size=parallel_config.get("dit_fsdp_size", 1),
    )

    RefinerDiTParallelConfig.set_config(
        dp_size=parallel_config.get("refiner_dit_dp_size", 1),
        tp_size=parallel_config.get("refiner_dit_tp_size", 1),
        ulysses_size=parallel_config.get("refiner_dit_ulysses_size", 1),
        ring_size=parallel_config.get("refiner_dit_ring_size", 1),
        cp_size=parallel_config.get("refiner_dit_cp_size", 1),
        cfg_size=parallel_config.get("refiner_dit_cfg_size", 1),
        fsdp_size=parallel_config.get("refiner_dit_fsdp_size", 1),
    )

    T5ParallelConfig.set_config(
        fsdp_size=parallel_config.get("t5_fsdp_size", 1),
    )

    disable_vae_parallel = parallel_config.get("disable_parallel_vae", False)
    if not is_parallel:
        if not disable_vae_parallel:
            logger.info("Not in parallel environment, disable vae parallel")
            disable_vae_parallel = True

    VAEParallelConfig.set_config(
        disable_parallel_vae=disable_vae_parallel,
        parallel_vae_split_dim=parallel_config.get("parallel_vae_split_dim", "width"),
    )

    if is_parallel:
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank)


def create_default_dit_configs() -> dict:
    configs = {
        "pipeline": {
            "enable_torch_compile": True,
            "torch_compile_models": "transformer",
            "torch_compile_mode": "default",
            "fuse_qkv": True,
        },
        "attn": {
            "type": "default",
            "choices": "default,sage-attn",
            "cosine_similarity_threshold": 0.999,
            "mse_threshold": 0.01,
            "sparsity": 0.25,
            "num_timesteps_high_precision": 0.0,
            "num_layers_high_precision": 0.0,
            "high_precision_attn_type": "default",
        },
        "linear": {
            "type": "default",
            "choices": "default,trtllm-fp8-blockwise,trtllm-fp8-per-tensor",
            "cosine_similarity_threshold": 0.999,
            "mse_threshold": 0.01,
            "recipe": "dynamic",
        },
        "parallel": {
            "disable_parallel_vae": False,
            "parallel_vae_split_dim": "width",
            "dit_dp_size": 1,
            "dit_tp_size": 1,
            "dit_ulysses_size": 1,
            "dit_ring_size": 1,
            "dit_cp_size": 1,
            "dit_cfg_size": 1,
            "dit_fsdp_size": 1,
            "refiner_dit_dp_size": 1,
            "refiner_dit_tp_size": 1,
            "refiner_dit_ulysses_size": 1,
            "refiner_dit_ring_size": 1,
            "refiner_dit_cp_size": 1,
            "refiner_dit_cfg_size": 1,
            "refiner_dit_fsdp_size": 1,
            "t5_fsdp_size": 1,
        },
        "teacache": {
            "enable_teacache": False,
            "use_ret_steps": True,
            "teacache_thresh": 0.2,
            "ret_steps": 0,
            "cutoff_steps": 50,
        },
    }
    return configs


def _setup_envs() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"


def setup_configs(**kwargs) -> None:
    """Setup all configurations"""
    _setup_envs()
    configs = create_default_dit_configs()
    configs.update(kwargs)
    # INSERT_YOUR_CODE
    pp = pprint.PrettyPrinter(indent=2, width=100, compact=False, sort_dicts=False)
    logger.info("Configs:\n" + pp.pformat(configs))
    setup_pipeline_config(**configs.get("pipeline", {}))
    setup_attn_config(**configs.get("attn", {}))
    setup_linear_config(**configs.get("linear", {}))
    setup_teacache_config(**configs.get("teacache", {}))
    setup_parallel_config(**configs.get("parallel", {}))
