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

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import UMT5EncoderModel

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.parallel import DiTParallelConfig, T5ParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.base_transformer import ditBaseTransformer
from visual_gen.models.utils import GemmGeluProcessor, WeightManagedBlocks
from visual_gen.utils.logger import MemoryMonitor, get_logger, log_execution_time

logger = get_logger(__name__)


class ditBasePipeline(DiffusionPipeline):
    """Base pipeline class for dit models.

    This class extends the DiffusionPipeline with pre and post processing hooks.
    """

    def _fsdp(self, **kwargs):
        param_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        logger.debug(f"FSDP param dtype: {param_dtype}")
        fsdp_kwargs = {
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
            )
        }
        for name in self.__dict__:
            model = self.__dict__[name]
            logger.debug(f"pipeline.config: {name} {type(model)}")
            if isinstance(model, UMT5EncoderModel) and T5ParallelConfig.fsdp_size() > 1:
                model = model.to(param_dtype)
                logger.info(f"FSDP {name}:{type(model)}")
                for block in model.encoder.block:
                    fully_shard(block, mesh=T5ParallelConfig.fsdp_device_mesh(), **fsdp_kwargs)
                fully_shard(model, mesh=T5ParallelConfig.fsdp_device_mesh(), **fsdp_kwargs)
            if isinstance(model, ditBaseTransformer) and DiTParallelConfig.fsdp_size() > 1:
                model = model.to(param_dtype)
                logger.info(f"FSDP {name}:{type(model)}")
                for block in model.blocks:
                    fully_shard(block, mesh=DiTParallelConfig.fsdp_device_mesh(), **fsdp_kwargs)
                fully_shard(model, mesh=DiTParallelConfig.fsdp_device_mesh(), **fsdp_kwargs)

    def _fuse_qkv(self, transformer):
        if not PipelineConfig.fuse_qkv:
            logger.debug("Fusing qkv is disabled")
            return

        for name, module in transformer.named_modules():
            if hasattr(module, "fuse_projections"):
                assert not module.fused_projections, "Attention module already fused"
                logger.debug(f"Fusing qkv for {name}")
                module.fuse_projections()
                # Replace fused Linear to ditLinear
                if hasattr(module, "to_qkv"):
                    # self attention
                    linear = ditLinear.from_linear(module.to_qkv)
                    linear.weight.data.copy_(module.to_qkv.weight)
                    if module.to_qkv.bias is not None:
                        linear.bias.data.copy_(module.to_qkv.bias)
                    linear.name = name + ".to_qkv"
                    setattr(module, "to_qkv", linear)
                    # delete original linear modules
                    delattr(module, "to_q")
                    delattr(module, "to_k")
                    delattr(module, "to_v")
                if hasattr(module, "to_kv"):
                    # cross attention
                    linear = ditLinear.from_linear(module.to_kv)
                    linear.weight.data.copy_(module.to_kv.weight)
                    if module.to_kv.bias is not None:
                        linear.bias.data.copy_(module.to_kv.bias)
                    linear.name = name + ".to_kv"
                    setattr(module, "to_kv", linear)
                    # delete original linear modules
                    delattr(module, "to_k")
                    delattr(module, "to_v")
                if hasattr(module, "to_added_qkv"):
                    # added projections for SD3 and others.
                    linear = ditLinear.from_linear(module.to_added_qkv)
                    linear.weight.data.copy_(module.to_added_qkv.weight)
                    if module.to_added_qkv.bias is not None:
                        linear.bias.data.copy_(module.to_added_qkv.bias)
                    linear.name = name + ".to_added_qkv"
                    setattr(module, "to_added_qkv", linear)
                    # delete original linear modules
                    delattr(module, "add_q_proj")
                    delattr(module, "add_k_proj")
                    delattr(module, "add_v_proj")

    def _fuse_gemm_gelu(self, transformer):
        gemm_gelu_checker = GemmGeluProcessor()
        gemm_gelu_checker.process_model(transformer)

    def torch_compile(self):
        logger.debug("Setting up torch compile")
        if PipelineConfig.enable_torch_compile:
            if isinstance(PipelineConfig.torch_compile_models, str):
                PipelineConfig.torch_compile_models = [PipelineConfig.torch_compile_models]
            torch_compile_models = PipelineConfig.torch_compile_models.copy()
            torch_compile_mode = PipelineConfig.torch_compile_mode
            for name in torch_compile_models:
                model = getattr(self, name)
                if model is None:
                    raise ValueError(f"Model {name} is not found in the pipeline")
                if isinstance(model, ditBaseTransformer):
                    transformer_blocks = []
                    for block_name in model.transformer_block_names():
                        if hasattr(model, block_name):
                            transformer_blocks.append(block_name)
                    if not transformer_blocks:
                        logger.warning(f"No transformer blocks found in {name}")
                        continue
                    for block_name in transformer_blocks:
                        logger.info(f"Torch compile {name}.{block_name}, compile mode: {torch_compile_mode}")
                        compiled_blocks = []
                        for block in getattr(model, block_name):
                            block = torch.compile(block, mode=torch_compile_mode)
                            compiled_blocks.append(block)
                        setattr(model, block_name, torch.nn.ModuleList(compiled_blocks))
                else:
                    logger.info(f"Torch compile {name}, compile mode: {torch_compile_mode}")
                    model = torch.compile(model, mode=torch_compile_mode)
                PipelineConfig.torch_compile_models.remove(name)

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        """Post-processing hook after load model checkpoints in 'from_pretrained' method."""
        raise NotImplementedError("Subclass should implement this method")

    @property
    def _execution_device(self):
        """
        Override the _execution_device in diffusers's pipeline to return the current device.
        """
        if PipelineConfig.model_wise_offloading or PipelineConfig.block_wise_offloading:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return super()._execution_device

    def enable_async_cpu_offload(
        self, model_wise: List[str] = None, block_wise: List[str] = None, offloading_stride: int = 1
    ) -> None:
        """CPU offload for dit pipeline

        Args:
            model_wise: List of models for model-wise offloading, such as ["text_encoder"]. The models in this list will be offloaded to CPU after the forward pass.
            block_wise: List of blocks for block-wise offloading, such as ["transformer"]. The models in this list will be offloaded to CPU in a block-wise manner. For example, for block in the transformer.blocks, the block idx % offloading_stride == 0 will be offloaded to CPU. And when block[idx] is computing, it will copy block[idx+offloading_stride]'s weight to GPU. So that copy from the offloaded weight to GPU can be overlapped with the forward pass. Currently, only support offloading the transformer.blocks.
            offloading_stride: Stride for block-wise offloading. If stride is 0, not enabled.

        """
        logger.debug(
            f"dit CPU offload: model_wise={model_wise}, block_wise={block_wise}, offloading_stride={offloading_stride}"
        )

        if block_wise is None:
            block_wise = []
        if model_wise is None:
            model_wise = []
        if set(model_wise) & set(block_wise):
            raise ValueError(f"model_wise and block_wise cannot have the same model, got {model_wise} and {block_wise}")

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        for name in block_wise:
            if not isinstance(getattr(self, name), ditBaseTransformer):
                raise ValueError(f"Currently, only support block-wise offloading for transformer, got {name}")
            if offloading_stride <= 0:
                logger.warning(
                    f"Block-wise offloading of {name} is not enabled because offloading_stride={offloading_stride}"
                )
                continue
            model = getattr(self, name)
            if model is None:
                logger.warning(f"Model {name} is not found in the pipeline")
                continue
            if PipelineConfig.enable_torch_compile:
                self.torch_compile()
            for block_name in model.transformer_block_names():
                block = WeightManagedBlocks(getattr(model, block_name))
                logger.info(
                    f"Set aync block-wise offloading for {name}.{block_name} with offloading_stride={offloading_stride}"
                )
                block.setup_offloading(offloading_stride)
                setattr(model, block_name, block)
            model.to(device)
            for module in model.modules():
                if isinstance(module, ditLinear) and module.offloading:
                    # offload the weight to cpu
                    offloading_weight = module.get_offloading_weights()
                    offloading_weight.data = offloading_weight.data.cpu()
            PipelineConfig.set_config(block_wise_offloading=block_wise, offloading_stride=offloading_stride)
        torch.cuda.empty_cache()

        skiped_models = []
        for name in self.__dict__:
            model = self.__dict__[name]
            if name in block_wise:
                continue
            if isinstance(model, torch.nn.Module):
                if name in model_wise:
                    supported_models = ["text_encoder", "image_encoder", "transformer"]
                    if name not in supported_models:
                        raise ValueError(
                            f"Currently, only support model-wise offloading for {supported_models}, got {name}"
                        )
                    logger.info(
                        f"Skip copying {name} to device {device} because it uses model-wise offloading strategy and will be dynamically copied to device before forward pass."
                    )
                    # Don't need to init these model in gpu, they will be dynamically copy to gpu before forward pass.
                    skiped_models.append(name)
                    logger.info(f"Set model-wise offloading for {name}")
                    PipelineConfig.set_config(model_wise_offloading=model_wise)
                    continue
                else:
                    logger.info(f"Copy {name} to device {device}")
                    model.to(device)
        if set(skiped_models) != set(model_wise):
            logger.warning(
                f"Some model-wise offloading models are not found in the pipeline: {set(model_wise) - set(skiped_models)}"
            )

    @classmethod
    @log_execution_time("visual_gen.pipeline")
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load pretrained model and call _after_load hook.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
            *args: Variable length argument list to pass to parent class
            **kwargs: Arbitrary keyword arguments to pass to parent class

        Returns:
            ditBasePipeline: A new instance of the pipeline with pretrained weights loaded
        """

        # Call parent's from_pretrained to get base pipeline
        with MemoryMonitor("visual_gen.pipeline", "model_loading"):
            pipeline = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Convert pipeline to this class type
        pipeline.__class__ = cls
        pipeline._after_load(pretrained_model_name_or_path, *args, **kwargs)

        # FSDP
        pipeline._fsdp(**kwargs)

        # Torch compile
        pipeline.torch_compile()

        logger.info("Pipeline loaded successfully")
        return pipeline

    def dit_dp_split(self, batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds):
        logger.debug(f"Data parallel split: batch_size={batch_size}")

        # Check if batch size is compatible with data parallel size
        dit_dp_size = DiTParallelConfig.get_instance().dp_size()
        if dit_dp_size == 1:
            logger.debug("No data parallel splitting needed (dp_size=1)")
            return batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds

        logger.info(f"Splitting data for DP: batch_size={batch_size}, dp_size={dit_dp_size}")

        if batch_size < dit_dp_size:
            error_msg = f"Batch size ({batch_size}) must be greater than or equal to data parallel size ({dit_dp_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure batch size is divisible by dp_size
        if batch_size % dit_dp_size != 0:
            error_msg = f"Batch size ({batch_size}) must be divisible by data parallel size ({dit_dp_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Split data for data parallel
        dp_rank = DiTParallelConfig.dp_rank()
        local_batch_size = batch_size // dit_dp_size
        start_idx = dp_rank * local_batch_size
        end_idx = start_idx + local_batch_size

        logger.debug(f"DP rank {dp_rank}: processing indices {start_idx}:{end_idx}")

        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            # Ensure continuous data in each rank
            prompt = prompt[start_idx:end_idx]
            if negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                negative_prompt = negative_prompt[start_idx:end_idx]
        else:
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds[start_idx:end_idx]
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds[start_idx:end_idx]

        logger.debug(f"Data split completed for rank {dp_rank}")
        return local_batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds

    def dit_dp_gather(self, latents):
        logger.debug("Gathering results from data parallel processes")

        # Gather results from all data parallel processes
        if DiTParallelConfig.get_instance().dp_size() > 1:
            dp_group = DiTParallelConfig.dp_group()
            if dp_group is not None:
                logger.debug(f"Gathering latents with shape {latents.shape}")
                gathered_latents = [torch.zeros_like(latents) for _ in range(DiTParallelConfig.dp_size())]
                dist.all_gather(gathered_latents, latents, group=dp_group)
                # Reorder the gathered results to match the original batch order
                latents = torch.cat([gathered_latents[i] for i in range(DiTParallelConfig.dp_size())], dim=0)
                logger.debug(f"Gathered latents shape: {latents.shape}")
        else:
            logger.debug("No gathering needed (dp_size=1)")

        return latents

    def cfg_parallel(
        self,
        transformer,
        cfg_positive_inputs: Optional[Dict] = None,
        cfg_negative_inputs: Optional[Dict] = None,
    ):
        logger.debug("Executing CFG parallel processing")

        # Check if we should use cfg parallel
        dit_cfg_size = DiTParallelConfig.cfg_size()
        assert dit_cfg_size == 2, "cfg_size must be 2"

        cfg_rank = DiTParallelConfig.cfg_rank()
        cfg_group = DiTParallelConfig.cfg_group()

        logger.debug(f"CFG parallel: rank={cfg_rank}, size={dit_cfg_size}")

        transformer_inputs = cfg_positive_inputs
        if cfg_rank == 0:
            PipelineConfig.set_config(cfg_type="positive")
            logger.debug("Using positive prompt embeds for CFG rank 0")
        else:
            PipelineConfig.set_config(cfg_type="negative")
            transformer_inputs = cfg_negative_inputs
            logger.debug("Using negative prompt embeds for CFG rank 1")

        with MemoryMonitor("visual_gen.pipeline", "transformer_forward"):
            noise_pred = transformer(
                **transformer_inputs,
            )[0]

        def gather_noise_pred(noise_pred):
            # Gather results from both ranks
            logger.debug("Gathering CFG results")
            gathered_noise = [torch.zeros_like(noise_pred) for _ in range(dit_cfg_size)]
            dist.all_gather(gathered_noise, noise_pred, group=cfg_group)
            return gathered_noise

        gathered_noise = gather_noise_pred(noise_pred)
        noise_cond = gathered_noise[0]
        noise_uncond = gathered_noise[1]
        return noise_cond, noise_uncond

    def _reset_teacache_config(self, num_inference_steps, do_classifier_free_guidance):
        logger.debug(f"Updating TeaCache config for {num_inference_steps} inference steps")
        step_size = 2 if do_classifier_free_guidance else 1
        if TeaCacheConfig.use_ret_steps():
            TeaCacheConfig.set_config(
                cutoff_steps=num_inference_steps * step_size, num_steps=num_inference_steps * step_size, cnt=0
            )
            logger.debug(
                f"TeaCache ret_steps mode: cutoff_steps={num_inference_steps * step_size}, num_steps={num_inference_steps * step_size}, cnt=0"
            )
        else:
            TeaCacheConfig.set_config(
                cutoff_steps=num_inference_steps * step_size - 2, num_steps=num_inference_steps * step_size, cnt=0
            )
            logger.debug(
                f"TeaCache standard mode: cutoff_steps={num_inference_steps * step_size - 2}, num_steps={num_inference_steps * step_size}, cnt=0"
            )

    def visual_gen_transformer(
        self,
        transformer,
        current_denoising_step,
        num_inference_steps,
        cfg_positive_inputs: Optional[Dict],
        cfg_negative_inputs: Optional[Dict] = None,
        do_classifier_free_guidance: bool = False,
    ) -> Tuple[torch.Tensor, None | torch.Tensor]:
        """
        This function is designed to handle the CFG parallel processing and set some global configs

        Args:
            transformer: The transformer model to use.
            current_denoising_step: The current denoising step.
            num_inference_steps: The number of inference steps.
            cfg_positive_inputs: The inputs produced by positive prompt, which must be provided.
            cfg_negative_inputs: The inputs produced by negative prompt, which must be provided if do_classifier_free_guidance is True.
            do_classifier_free_guidance: Whether to use classifier-free guidance.
        Outputs:
            noise_cond: The noise prediction for the positive prompt, i.e., cfg_positive_inputs.
            noise_uncond: The noise prediction for the negative prompt, i.e., cfg_negative_inputs.
        """
        logger.debug(f"transformer forward: current_denoising_step={current_denoising_step}")

        if cfg_positive_inputs is None:
            raise ValueError("cfg_positive_inputs is required")

        if current_denoising_step == 0:
            self._reset_teacache_config(num_inference_steps, do_classifier_free_guidance)

        PipelineConfig.set_config(
            do_classifier_free_guidance=do_classifier_free_guidance,
            current_denoising_step=current_denoising_step,
            num_inference_steps=num_inference_steps,
            num_dit_layers=transformer.config.num_layers,
        )

        if DiTParallelConfig.cfg_size() > 1:
            logger.debug("Enabled CFG parallel processing")
            assert do_classifier_free_guidance, "do_classifier_free_guidance must be True"
            assert cfg_negative_inputs is not None, "cfg_negative_inputs is required"
            return self.cfg_parallel(
                transformer,
                cfg_positive_inputs,
                cfg_negative_inputs,
            )
        else:
            logger.debug("Disabled CFG parallel processing")

            PipelineConfig.set_config(cfg_type="positive")
            with MemoryMonitor("visual_gen.pipeline", "transformer_forward_positive"):
                noise_cond = transformer(
                    **cfg_positive_inputs,
                )[0]

            noise_uncond = None
            if do_classifier_free_guidance:
                logger.debug("Computing negative prompt prediction")
                PipelineConfig.set_config(cfg_type="negative")
                with MemoryMonitor("visual_gen.pipeline", "transformer_forward_negative"):
                    noise_uncond = transformer(
                        **cfg_negative_inputs,
                    )[0]

        return noise_cond, noise_uncond
