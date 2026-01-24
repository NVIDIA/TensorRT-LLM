# Adopted from https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/transformers/transformer_qwenimage.py
# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

import gc
import types
from typing import Optional

import torch
from common import (  # noqa: E402
    BaseArgumentParser,
    autotuning,
    benchmark_inference,
    create_dit_config,
    generate_autotuner_dir,
    generate_output_path,
    log_args_and_timing,
    save_output,
    validate_parallel_config,
)
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from visual_gen import setup_configs
from visual_gen.layers import apply_visual_gen_linear, apply_visual_gen_norm, ditAttnProcessor
from visual_gen.models.utils import apply_async_cpu_offloading
from visual_gen.utils import cudagraph_wrapper, get_logger
from visual_gen.utils.parallel import dit_sp_gather, dit_sp_split

logger = get_logger(__name__)


class ditQwenDoubleStreamAttnProcessor2_0(ditAttnProcessor):
    def __init__(self):
        super().__init__()
        logger.debug("ditQwenDoubleStreamAttnProcessor2_0 initialized")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)"
            )

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = self.visual_gen_attn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            tensor_layout="NHD",
            joint_seq_length=seq_txt,
            valid_joint_seq_length=None,
            joint_strategy="front",
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


def _dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes=None,
    txt_seq_lens=None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs=None,
    return_dict=True,
):
    """The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    hidden_states = dit_sp_split(hidden_states, dim=1, allow_uneven=False)
    image_rotary_emb = list(image_rotary_emb)
    image_rotary_emb[0] = dit_sp_split(image_rotary_emb[0], dim=0, allow_uneven=False)

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    output = dit_sp_gather(output, dim=1)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def load_and_setup_pipeline(args):
    """Load and configure the Flux pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    def enable_visual_gen(pipe):
        if not args.enable_async_cpu_offload:
            pipe = pipe.to(torch.cuda.current_device())

        # import ipdb; ipdb.set_trace()
        for name, module in pipe.transformer.named_modules():
            # only replace self-attention layers
            # the cross-attention has very small kv, thus doesn't need cp and we don't split its kv
            if isinstance(module, Attention):
                attn_processor = ditQwenDoubleStreamAttnProcessor2_0()
                attn_processor.name = name
                module.set_processor(attn_processor)
        apply_visual_gen_linear(pipe.transformer, load_parameters=True)
        apply_visual_gen_norm(
            pipe.transformer,
            rmsnorm=["norm_q", "norm_k", "norm_added_q", "norm_added_k"],
            load_parameters=True,
        )

        # replace the dit `forward` method, which supports sequence parallel.
        pipe.transformer.forward = types.MethodType(_dit_forward, pipe.transformer)

        if not args.disable_torch_compile:
            pipe.transformer = torch.compile(pipe.transformer, mode=args.torch_compile_mode)
            pipe.text_encoder = torch.compile(pipe.text_encoder)
            pipe.vae = torch.compile(pipe.vae)

        if args.enable_async_cpu_offload:
            pipe.transformer
            apply_async_cpu_offloading(
                pipe.transformer,
                transformer_blocks_name="transformer_blocks",
                offloading_stride=args.visual_gen_block_cpu_offload_stride,
            )
            modules = [(n, getattr(pipe, n, None)) for n in dir(pipe)]
            modules = [(n, m) for n, m in modules if isinstance(m, torch.nn.Module)]
            for n, module in modules:
                if n != "transformer":
                    module = module.to(torch.cuda.current_device())
        else:
            pipe = pipe.to(torch.cuda.current_device())

    enable_visual_gen(pipe)

    return pipe


def run_inference(pipe, args, enable_autotuner: bool = False):
    """Run warmup and actual inference."""

    def inference_fn(warmup: bool = False):
        if warmup:
            num_inference_steps = 2
        else:
            num_inference_steps = args.num_inference_steps

        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=torch.Generator(device=torch.cuda.current_device()).manual_seed(
                args.random_seed
            ),
        ).images[0]

        return image

    autotuner_dir = args.autotuner_result_dir
    if enable_autotuner and not args.skip_autotuning:
        if autotuner_dir is None:
            autotuner_dir = generate_autotuner_dir(args)
        autotuning(inference_fn, autotuner_dir)

    # Apply CUDAGraphs
    if args.enable_cuda_graph:
        assert args.attn_type != "sage-attn", "sage-attn has accuracy issue when enable cudagraph"
        assert not (
            args.enable_async_cpu_offload
            or args.enable_sequential_cpu_offload
            or args.enable_model_cpu_offload
        ), "CudaGraph is not supported when using cpu offload"
        pipe.dit._dit_forward = cudagraph_wrapper(pipe.dit._dit_forward)

    image, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=True,
        profile=args.profile,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return image, elapsed_time


def main():
    """Main function for Qwen-Image text-to-image generation."""
    # Setup argument parser
    parser = BaseArgumentParser("Qwen-Image Text-to-Image Generation")

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
    }

    model_name = "Qwen/Qwen-Image"
    # Set defaults for Qwen-Image
    parser.set_defaults(
        model_path=model_name,
        prompt="""A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition""",
        negative_prompt=" ",
        width=1664,
        height=928,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        random_seed=42,
    )

    args = parser.parse_args()

    # Add positive magic to prompt
    args.prompt = args.prompt + positive_magic["en"]

    enable_autotuner = False
    if args.linear_type == "auto" or args.attn_type == "auto":
        enable_autotuner = True
        if not args.disable_torch_compile:
            logger.warning("Disable torch compile when using autotuner")
            args.disable_torch_compile = True
        if args.enable_async_cpu_offload:
            logger.warning("Disable visual_gen cpu offload when using autotuner")
            args.enable_async_cpu_offload = False

    # Validate configuration
    validate_parallel_config(args)

    # Generate output path
    output_path = generate_output_path(args, save_type="png")

    # Load pipeline
    pipe = load_and_setup_pipeline(args)

    # Run inference
    image, elapsed_time = run_inference(pipe, args, enable_autotuner)

    # delete pipe to free cudagraph, otherwise might hang
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    # Log results and save output
    log_args_and_timing(args, elapsed_time)
    save_output(image, output_path, output_type="image")

    logger.info(
        f"Peak gpu memory usage: {torch.cuda.max_memory_reserved(torch.cuda.current_device()) / 1024 / 1024} MB"
    )


if __name__ == "__main__":
    main()
