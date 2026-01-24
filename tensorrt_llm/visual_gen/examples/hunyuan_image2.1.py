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
import os
import types
from typing import Dict, Optional, Union

import torch._dynamo
import torch.distributed as dist
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
from einops import rearrange
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
from hyimage.models.hunyuan.modules.modulate_layers import apply_gate
from hyimage.models.hunyuan.modules.token_refiner import IndividualTokenRefinerBlock
from visual_gen import setup_configs
from visual_gen.configs.parallel import (
    DiTParallelConfig,
    RefinerDiTParallelConfig,
    VAEParallelConfig,
)
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers import apply_visual_gen_linear, apply_visual_gen_norm, ditAttnProcessor
from visual_gen.models.vaes.Hunyuan_vae import ditHunyuanVAE2D
from visual_gen.utils import cudagraph_wrapper, get_logger
from visual_gen.utils.parallel import dit_dp_gather, dit_dp_split, dit_sp_gather, dit_sp_split

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.recompile_limit = 128

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch  # noqa: E402

logger = get_logger(__name__)


class visual_gen_hunyuan_attention(ditAttnProcessor):
    def __call__(
        self,
        q,
        k,
        v,
        attn_mode="flash",
        text_mask=None,
    ):
        """Multi-modal attention function that processes image and text sequences."""
        query, encoder_query = q
        key, encoder_key = k
        value, encoder_value = v

        assert attn_mode == "flash"  # Only flash attention is implemented for now
        sequence_length = query.size(1)
        encoder_sequence_length = encoder_query.size(1)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        # replace with visual_gen_attn
        hidden_states = self.visual_gen_attn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            dropout_p=0.0,
            scale=None,
            joint_seq_length=encoder_sequence_length,
            valid_joint_seq_length=text_mask,
            joint_strategy="rear",
        )

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)

        attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        b, s, a, d = attn.shape
        attn = attn.reshape(b, s, -1)

        return attn


def _preprocess_text(
    text_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    extra_kwargs: dict,
):
    """Preprocess text to remove padding and mask. This is useful for cudagraph usage."""
    text_states = dit_dp_split(text_states, dim=0)
    encoder_attention_mask = dit_dp_split(encoder_attention_mask, dim=0)
    if (
        "byt5_text_states" in extra_kwargs
        and extra_kwargs["byt5_text_states"].shape[0] == text_states.shape[0] * 2
    ):
        extra_kwargs["byt5_text_states"] = dit_dp_split(extra_kwargs["byt5_text_states"], dim=0)
    if (
        "byt5_text_mask" in extra_kwargs
        and extra_kwargs["byt5_text_mask"].shape[0] == text_states.shape[0] * 2
    ):
        extra_kwargs["byt5_text_mask"] = dit_dp_split(extra_kwargs["byt5_text_mask"], dim=0)

    valid_text_len = encoder_attention_mask.sum(dim=1)
    # todo: remove this assertion by supporting variable length text in attention. This assertion is for cudagraph usage.
    assert valid_text_len.max() == valid_text_len.min(), (
        "valid_text_len must be the same for all samples"
    )
    text_states = text_states[:, : valid_text_len.max(), :]
    encoder_attention_mask = encoder_attention_mask[:, : valid_text_len.max()]
    if "byt5_text_states" in extra_kwargs:
        byt5_text_states = extra_kwargs["byt5_text_states"]
        byt5_text_mask = extra_kwargs["byt5_text_mask"]
        valid_byt5_text_len = byt5_text_mask.sum(dim=1)
        assert valid_byt5_text_len.max() == valid_byt5_text_len.min(), (
            "valid_byt5_text_len must be the same for all samples"
        )
        byt5_text_states = byt5_text_states[:, : valid_byt5_text_len.max(), :]
        byt5_text_mask = byt5_text_mask[:, : valid_byt5_text_len.max()]
        extra_kwargs["byt5_text_states"] = byt5_text_states
        extra_kwargs["byt5_text_mask"] = byt5_text_mask
        valid_text_len = valid_text_len + byt5_text_mask.sum(dim=1)

    # convert text mask to text length, since we don't use mask inplementation for attention
    valid_text_len = valid_text_len.cpu()
    return text_states, encoder_attention_mask, extra_kwargs, valid_text_len


def _preprocess_pos_embed(
    self,
    x: torch.Tensor,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
):
    """Copy positional embeddings to the device in advance to avoid H2D copy and synchronize, which harm performance and may break cudagraph."""
    input_shape = x.shape
    # Calculate spatial dimensions and get rotary embeddings
    if len(input_shape) == 5:
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        if freqs_cos is None or freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
    elif len(input_shape) == 4:
        _, _, oh, ow = x.shape
        th, tw = (
            oh // self.patch_size[0],
            ow // self.patch_size[1],
        )
        if freqs_cos is None or freqs_sin is None:
            assert freqs_cos is None and freqs_sin is None, (
                "freqs_cos and freqs_sin must be both None or both not None"
            )
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((th, tw))
    else:
        raise ValueError(f"Unsupported hidden_states shape: {x.shape}")
    freqs_cos = freqs_cos.to(x.device)
    freqs_sin = freqs_sin.to(x.device)
    return freqs_cos, freqs_sin


def _dit_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    text_states: torch.Tensor,
    valid_text_len: torch.Tensor,
    output_features: bool = False,
    output_features_stride: int = 8,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    return_dict: bool = False,
    guidance=None,
    extra_kwargs=None,
    *,
    timesteps_r: Optional[torch.LongTensor] = None,
    is_refiner: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass for the transformer.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input image tensor.
    timestep : torch.LongTensor
        Timestep tensor.
    text_states : torch.Tensor
        Text embeddings.
    encoder_attention_mask : torch.Tensor
        Attention mask for text.
    output_features : bool, optional
        Whether to output intermediate features.
    output_features_stride : int, optional
        Stride for outputting features.
    freqs_cos, freqs_sin : torch.Tensor, optional
        Precomputed rotary embeddings.
    return_dict : bool, optional
        Not supported.
    guidance : torch.Tensor, optional
        Guidance vector for distillation.
    extra_kwargs : dict, optional
        Extra arguments for ByT5.
    timesteps_r : torch.LongTensor, optional
        Additional timestep for MeanFlow.

    Returns:
    -------
    tuple
        (img, features_list, shape)
    """
    if is_refiner:
        PipelineConfig.set_config(in_refiner_stage=True)
    else:
        PipelineConfig.set_config(in_refiner_stage=False)

    hidden_states = dit_dp_split(hidden_states, dim=0)
    timestep = dit_dp_split(timestep, dim=0)
    if guidance is not None and guidance.shape[0] == hidden_states.shape[0] * 2:
        guidance = dit_dp_split(guidance, dim=0)

    if guidance is None:
        guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
    img = x = hidden_states
    t = timestep
    txt = text_states
    input_shape = x.shape

    # Calculate spatial dimensions and get rotary embeddings
    if len(input_shape) == 5:
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        if freqs_cos is None or freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
    elif len(input_shape) == 4:
        tt = None
        _, _, oh, ow = x.shape
        th, tw = (
            oh // self.patch_size[0],
            ow // self.patch_size[1],
        )
        if freqs_cos is None or freqs_sin is None:
            assert freqs_cos is None and freqs_sin is None, (
                "freqs_cos and freqs_sin must be both None or both not None"
            )
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((th, tw))
    else:
        raise ValueError(f"Unsupported hidden_states shape: {x.shape}")

    with torch.cuda.nvtx.range("HunyuanImage.img_in"):
        img = self.img_in(img)

    # Prepare modulation vectors
    with torch.cuda.nvtx.range("HunyuanImage.time_in"):
        vec = self.time_in(t)

    # MeanFlow support: merge timestep and timestep_r if available
    if self.use_meanflow:
        assert self.time_r_in is not None, "use_meanflow is True but time_r_in is None"
    if timesteps_r is not None:
        assert self.time_r_in is not None, "timesteps_r is not None but time_r_in is None"
        vec_r = self.time_r_in(timesteps_r)
        vec = (vec + vec_r) / 2

    # Guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        with torch.cuda.nvtx.range("HunyuanImage.guidance_in"):
            vec = vec + self.guidance_in(guidance)

    # Embed image and text
    if self.text_projection == "linear":
        with torch.cuda.nvtx.range("HunyuanImage.txt_in"):
            txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        with torch.cuda.nvtx.range("HunyuanImage.txt_in"):
            txt = self.txt_in(txt, t, None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    if self.glyph_byT5_v2:
        byt5_text_states = extra_kwargs["byt5_text_states"]
        with torch.cuda.nvtx.range("HunyuanImage.byt5_in"):
            byt5_txt = self.byt5_in(byt5_text_states)
        with torch.cuda.nvtx.range("HunyuanImage.reorder_txt_token"):
            # txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)
            txt = torch.cat([byt5_txt, txt], dim=1)

    with torch.cuda.nvtx.range("HunyuanImage.dit_sp_split"):
        # split image and freqs_cos, freqs_sin for sequence parallel
        img = dit_sp_split(img, dim=1, allow_uneven=False)
        freqs_cos = dit_sp_split(freqs_cos, dim=0)
        freqs_sin = dit_sp_split(freqs_sin, dim=0)

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # # Calculate cu_seqlens and max_s for flash attention
    # cu_seqlens, max_s = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens, max_s = None, None  # they are unused in visual_gen

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    text_mask = valid_text_len
    # Pass through double stream blocks
    with torch.cuda.nvtx.range("HunyuanImage.double_blocks"):
        for block in self.double_blocks:
            double_block_args = [img, txt, vec, freqs_cis, text_mask, cu_seqlens, max_s]
            img, txt = block(*double_block_args)

    # Merge txt and img to pass through single stream blocks
    x = torch.cat((img, txt), 1)
    features_list = [] if output_features else None

    with torch.cuda.nvtx.range("HunyuanImage.single_blocks"):
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    cu_seqlens,
                    max_s,
                ]
                x = block(*single_block_args)
                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

    img = x[:, :img_seq_len, ...]

    # Final layer
    with torch.cuda.nvtx.range("HunyuanImage.final_layer"):
        img = self.final_layer(img, vec)

    img = dit_sp_gather(img, dim=1)

    img = dit_dp_gather(img, dim=0)

    # Unpatchify based on input shape
    if len(input_shape) == 5:
        img = self.unpatchify(img, tt, th, tw)
        shape = (tt, th, tw)
    elif len(input_shape) == 4:
        img = self.unpatchify_2d(img, th, tw)
        shape = (th, tw)
    else:
        raise ValueError(f"Unsupported input_shape: {input_shape}")

    assert not return_dict, "return_dict is not supported."

    if output_features:
        features_list = torch.stack(features_list, dim=0)
        features_list = dit_sp_gather(features_list, dim=2)
        features_list = dit_dp_gather(features_list, dim=1)
    else:
        features_list = None

    return img, features_list, shape


def dit_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    text_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    output_features: bool = False,
    output_features_stride: int = 8,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    return_dict: bool = False,
    guidance=None,
    extra_kwargs=None,
    *,
    timesteps_r: Optional[torch.LongTensor] = None,
):
    with (
        torch.no_grad(),
        torch.cuda.nvtx.range(
            f"dit_forward, ring_size: {DiTParallelConfig.ring_size()}, ulysses_size: {DiTParallelConfig.ulysses_size()}, cp_size: {DiTParallelConfig.cp_size()}"
        ),
    ):
        text_states, encoder_attention_mask, extra_kwargs, valid_text_len = _preprocess_text(
            text_states, encoder_attention_mask, extra_kwargs
        )
        # cast text_states to bfloat16 for speedup
        text_states = text_states.to(hidden_states.dtype)
        freqs_cos, freqs_sin = self._preprocess_pos_embed(hidden_states, freqs_cos, freqs_sin)
        output = self._dit_forward(
            hidden_states,
            timestep,
            text_states,
            valid_text_len,
            output_features,
            output_features_stride,
            freqs_cos,
            freqs_sin,
            return_dict,
            guidance,
            extra_kwargs,
            timesteps_r=timesteps_r,
            is_refiner=False,
        )
    return output


def refiner_dit_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    text_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    output_features: bool = False,
    output_features_stride: int = 8,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    return_dict: bool = False,
    guidance=None,
    extra_kwargs=None,
    *,
    timesteps_r: Optional[torch.LongTensor] = None,
):
    with (
        torch.no_grad(),
        torch.cuda.nvtx.range(
            f"refiner_dit_forward, ring_size: {RefinerDiTParallelConfig.ring_size()}, ulysses_size: {RefinerDiTParallelConfig.ulysses_size()}, cp_size: {RefinerDiTParallelConfig.cp_size()}"
        ),
    ):
        text_states, encoder_attention_mask, extra_kwargs, valid_text_len = _preprocess_text(
            text_states, encoder_attention_mask, extra_kwargs
        )
        # cast text_states to bfloat16 for speedup
        text_states = text_states.to(hidden_states.dtype)
        freqs_cos, freqs_sin = self._preprocess_pos_embed(hidden_states, freqs_cos, freqs_sin)
        output = self._dit_forward(
            hidden_states,
            timestep,
            text_states,
            valid_text_len,
            output_features,
            output_features_stride,
            freqs_cos,
            freqs_sin,
            return_dict,
            guidance,
            extra_kwargs,
            timesteps_r=timesteps_r,
            is_refiner=True,
        )
    return output


def individual_token_refiner_block_forward(self, x, c, attn_mask):
    gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
    norm_x = self.norm1(x)
    qkv = self.self_attn_qkv(norm_x)
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    q = self.self_attn_q_norm(q).to(v)
    k = self.self_attn_k_norm(k).to(v)
    # use visual_gen attention for cuda graph tracing
    attn = self.attn.visual_gen_attn(q, k, v, tensor_layout="NHD")
    b, s, _, _ = q.shape
    attn = attn.reshape(b, s, -1)
    # attn = attention(q, k, v, attn_mask=attn_mask)
    x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
    x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
    return x


def load_and_setup_pipeline(args):
    """Load and configure the Flux pipeline."""
    # Create dit configuration
    dit_configs = create_dit_config(args)
    setup_configs(**dit_configs)

    # Create pipeline
    pipe = HunyuanImagePipeline.from_pretrained(
        args.model_path,
        use_fp8=False,
        enable_stage1_offloading=False,  # offload models in stage1 pipeline when reprompt or refiner is working
        enable_reprompt_model_offloading=False,  # offload reprompt model after finishing
        enable_refiner_offloading=False,  # offload refiner model after finishing
        enable_text_encoder_offloading=False,  # offload text encoder after finishing
        enable_full_dit_offloading=False,  # offload during text encoding and latent decoding
        enable_vae_offloading=False,  # offload vae after finishing
        enable_byt5_offloading=False,  # offload byt5 after finishing
        device="cuda",
    )

    def enable_visual_gen(pipe, is_refiner: bool = False):
        # replace the attention with visual_gen_hunyuan_attention
        for block in pipe.dit.double_blocks:
            block.core_attn = visual_gen_hunyuan_attention()
        for block in pipe.dit.single_blocks:
            block.core_attn = visual_gen_hunyuan_attention()

        apply_visual_gen_linear(pipe.dit, load_parameters=True)
        apply_visual_gen_norm(
            pipe.dit,
            rmsnorm=[
                "img_attn_q_norm",
                "img_attn_k_norm",
                "txt_attn_q_norm",
                "txt_attn_k_norm",
                "q_norm",
                "k_norm",
            ],
            load_parameters=True,
        )

        # replace the forward with dit_forward, which supports parallel and remove padding in text.
        pipe.dit._dit_forward = types.MethodType(_dit_forward, pipe.dit)
        pipe.dit._preprocess_pos_embed = types.MethodType(_preprocess_pos_embed, pipe.dit)
        if is_refiner:
            pipe.dit.forward = types.MethodType(refiner_dit_forward, pipe.dit)
        else:
            pipe.dit.forward = types.MethodType(dit_forward, pipe.dit)

        # redefine the individual token refiner block with visual_gen attention for cuda graph tracing
        def _replace_individual_token_refiner_block(module):
            if isinstance(module, IndividualTokenRefinerBlock):
                setattr(module, "attn", ditAttnProcessor())
                module.forward = types.MethodType(individual_token_refiner_block_forward, module)

        pipe.dit.apply(_replace_individual_token_refiner_block)

        if not VAEParallelConfig.disable_parallel_vae:
            config = ditHunyuanVAE2D.load_config(pipe.config.vae_config.load_from)
            visual_gen_hunyuan_vae = ditHunyuanVAE2D.from_config(config)
            visual_gen_hunyuan_vae.load_checkpoint(pipe.config.vae_config.load_from)
            visual_gen_hunyuan_vae.parallel_vae(split_dim=VAEParallelConfig.parallel_vae_split_dim)
            visual_gen_hunyuan_vae.to("cuda")
            pipe.vae = visual_gen_hunyuan_vae
            torch.cuda.empty_cache()

        if not args.disable_torch_compile:
            pipe.dit = torch.compile(pipe.dit, mode=args.torch_compile_mode)
            pipe.text_encoder = torch.compile(pipe.text_encoder)
            pipe.vae = torch.compile(pipe.vae)

    enable_visual_gen(pipe)
    if args.use_refiner:
        enable_visual_gen(pipe.refiner_pipeline, is_refiner=True)

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
            width=args.width,
            height=args.height,
            use_reprompt=args.use_reprompt,
            use_refiner=args.use_refiner,
            num_inference_steps=num_inference_steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            seed=args.random_seed,
        )
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
        # todo: refiner has accuracy issue when enable cudagraph
        # if args.use_refiner:
        #     pipe.refiner_pipeline.dit._dit_forward = cudagraph_wrapper(pipe.refiner_pipeline.dit._dit_forward)

    image, elapsed_time = benchmark_inference(
        inference_fn,
        warmup=True,
        random_seed=args.random_seed,
        enable_autotuner=enable_autotuner,
        autotuner_dir=autotuner_dir,
    )
    return image, elapsed_time


def main():
    """Main function for HunyuanImage-2.1 text-to-image generation."""
    # Setup argument parser
    parser = BaseArgumentParser("HunyuanImage-2.1 Text-to-Image Generation")

    # Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
    model_name = "hunyuanimage-v2.1-distilled"
    # Set defaults for HunyuanImage-2.1
    parser.set_defaults(
        model_path=model_name,
        prompt="A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style.",
        # Examples of supported resolutions and aspect ratios for HunyuanImage-2.1:
        # 16:9  -> width=2560, height=1536
        # 4:3   -> width=2304, height=1792
        # 1:1   -> width=2048, height=2048
        # 3:4   -> width=1792, height=2304
        # 9:16  -> width=1536, height=2560
        # Please use one of the above width/height pairs for best results.
        width=2048,
        height=2048,
        use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
        use_refiner=False,  # Enable refiner model
        # For the distilled model, use 8 steps for faster inference.
        # For the non-distilled model, use 50 steps for better quality.
        num_inference_steps=8 if "distilled" in model_name else 50,
        guidance_scale=3.25 if "distilled" in model_name else 3.5,
        shift=4 if "distilled" in model_name else 5,
        random_seed=649151,
    )

    args = parser.parse_args()

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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
