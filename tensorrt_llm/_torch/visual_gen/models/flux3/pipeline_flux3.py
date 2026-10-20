# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from io import BytesIO
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torch import Tensor

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
    Flux3ImageVAE,
    Flux3ImageVAEConfig,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import Flux3FlowMatchScheduler
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder import (
    Qwen3VLEmbedder,
    Qwen3VLEmbedderParams,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_lora import apply_flux3_runtime_lora
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Model
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import (
    BasePipeline,
    ExtraParamSchema,
    RefSlotSpec,
    RoleSpec,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.logger import logger

I2I_COND_T = 0
I2I_COND_L = -1


def prc_img(
    x: Tensor, t_coord: Optional[Tensor] = None, l_coord: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    _, h, w = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = x.permute(1, 2, 0).reshape(h * w, -1)
    return x, x_ids


def prc_txt(
    x: Tensor, t_coord: Optional[Tensor] = None, l_coord: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txt"
    length, _ = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(length, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def _batched(fn: Callable) -> Callable:
    def wrapped(
        x: Tensor, t_coord: Optional[Tensor] = None, l_coord: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        pairs = [
            fn(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
            for i in range(len(x))
        ]
        xs, ids = zip(*pairs)
        return torch.stack(xs), torch.stack(ids)

    return wrapped


batched_prc_img = _batched(prc_img)
batched_prc_txt = _batched(prc_txt)


def _compress_time(t_ids: Tensor) -> Tensor:
    t_max = torch.max(t_ids)
    remap = torch.zeros((t_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    uniq = torch.unique(t_ids, sorted=True)
    remap[uniq] = torch.arange(len(uniq), device=t_ids.device, dtype=t_ids.dtype)
    return remap[t_ids]


def scatter_ids(x: Tensor, x_ids: Tensor):
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_c = _compress_time(t_ids)
        t = int(torch.max(t_ids_c)) + 1
        h = int(torch.max(h_ids)) + 1
        w = int(torch.max(w_ids)) + 1

        flat_ids = t_ids_c * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out_list.append(out.reshape(1, t, h, w, ch).permute(0, 4, 1, 2, 3))
    return out_list


def sample_image_latents(
    *,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    image_rope_scale: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    copies: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    if min(height, width) < 16 or height % 16 or width % 16:
        raise ValueError("FLUX.3 image dimensions must be positive multiples of 16")
    h, w = height // 16, width // 16
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn((h * w, 128), generator=rng).unsqueeze(0)
    x_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)
    ).unsqueeze(0)
    orig_ids = x_ids.to(device)
    x_ids = x_ids.to(dtype=torch.float32)
    x_ids[..., 1:3] = x_ids[..., 1:3] * image_rope_scale
    x = x.repeat(copies, 1, 1).to(device=device, dtype=dtype)
    x_ids = x_ids.repeat(copies, 1, 1).to(device=device)
    return x, x_ids, orig_ids


def pack_i2i_image_condition(
    latents: List[Tensor],
    *,
    device: torch.device,
    l_start: int = I2I_COND_L,
    image_rope_scale: float = 0.5,
    copies: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    x_cond_parts = []
    x_cond_id_parts = []
    for i_ref, latent in enumerate(latents):
        x_ref, x_ref_ids = prc_img(
            latent,
            t_coord=torch.tensor([I2I_COND_T], device=latent.device, dtype=torch.long),
            l_coord=torch.tensor([l_start - i_ref], device=latent.device, dtype=torch.long),
        )
        x_cond_parts.append(x_ref)
        x_cond_id_parts.append(x_ref_ids)

    x_cond = torch.cat(x_cond_parts, dim=0).unsqueeze(0)
    x_cond_ids = torch.cat(x_cond_id_parts, dim=0).unsqueeze(0)
    x_cond_ids = x_cond_ids.to(dtype=torch.float32)
    x_cond_ids[..., 1:3] = x_cond_ids[..., 1:3] * image_rope_scale
    x_cond_timesteps = torch.zeros(x_cond.shape[:2], device=x_cond.device, dtype=x_cond.dtype)
    x_cond = x_cond.repeat(copies, 1, 1).to(device=device, dtype=torch.bfloat16)
    x_cond_ids = x_cond_ids.repeat(copies, 1, 1).to(device=device)
    x_cond_timesteps = x_cond_timesteps.repeat(copies, 1).to(device=device, dtype=torch.bfloat16)
    return x_cond, x_cond_ids, x_cond_timesteps


def _load_image_pixels(source: str | bytes, *, device: torch.device) -> Tensor:
    """Prepare a reference image as FP32 [1, 3, H, W] in [-1, 1]."""
    with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        scale = min(1.0, 4096 / math.sqrt(image.width * image.height))
        if scale < 1:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
        width, height = image.width // 16 * 16, image.height // 16 * 16
        if min(width, height) == 0:
            raise ValueError("FLUX.3 reference images must be at least 16 pixels on each axis")
        left, top = (image.width - width) // 2, (image.height - height) // 2
        pixels = np.array(image.crop((left, top, left + width, top + height)))
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 127.5
        - 1
    )


def _as_image_grid(tokens: Tensor, orig_ids: Tensor) -> Tensor:
    return scatter_ids(tokens.float(), orig_ids[:1])[0].squeeze(2)


@register_pipeline(
    "Flux3Pipeline",
    hf_ids=[
        "black-forest-labs/flux-3-image-klein-base",
        "black-forest-labs/flux-3-image-klein",
    ],
    defaults={"variant": None},
    doc="FLUX.3 image generation and editing.",
)
class Flux3Pipeline(BasePipeline):
    """FLUX.3 image generation with shared text encoder and image VAE."""

    supports_image_edit = True

    @property
    def default_warmup_resolutions(self) -> list[tuple[int, int]]:
        return [(1024, 1024)]

    @property
    def default_warmup_num_frames(self) -> list[int]:
        return [1]

    def warmup_cache_key(self, height: int, width: int, num_frames: int) -> tuple:
        return (height, width)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (
            autotune_context,
            autotune_enabled,
        )

        if not autotune_enabled():
            self._run_flux3_warmup(height, width, num_frames, steps)
            return

        # Tune before capture so CUDA graphs use the selected tactics.
        with self.disallow_cuda_graph_capture(), autotune_context(self.device, self.rank):
            self._run_flux3_warmup(height, width, num_frames, steps)
        if self.pipeline_config.cuda_graph.enable:
            with autotune_context(self.device, self.rank, tune=False):
                self._run_flux3_warmup(height, width, num_frames, steps)

    def _run_flux3_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        ctx = self._encode_prompt("warmup", "")
        for mode in self.pipeline_config.compilation.modes or ["t2i", "i2i"]:
            kwargs = dict(ctx=ctx, height=height, width=width, num_inference_steps=steps)
            if mode == "i2i":
                kwargs["image_cond"] = torch.zeros(1, 3, height, width, device=self.device)
            self.forward(mode, **kwargs)

    def _setup_cuda_graphs(self):
        """
        Install the forward wrapper during initialization. The loader compiles
        individual blocks afterward; CUDA graph capture occurs during warmup.
        """
        if not self.pipeline_config.cuda_graph.enable:
            return

        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        self.transformer.register_cuda_graph_extra_key_fns(runner)
        compile_note = " (with torch.compile)" if self.pipeline_config.torch_compile.enable else ""
        logger.info(f"CUDA graph runner: wrapping transformer.forward{compile_note}")
        self.transformer.forward = runner.wrap(self.transformer.forward)
        self._cuda_graph_runners["transformer"] = runner

    def _init_transformer(self) -> None:
        logger.info("Creating FLUX.3 transformer with quantization support...")
        model_config = self.pipeline_config.model_configs["transformer"]
        vgm = self.pipeline_config.visual_gen_mapping
        if (
            getattr(model_config.pretrained_config, "flow_map", False)
            and vgm is not None
            and vgm.cfg_size > 1
        ):
            raise ValueError("FLUX.3 flow-map checkpoints do not support CFG parallelism")
        if self.pipeline_config.runtime_lora is not None:
            model_config.extra_attrs["force_unfused_lora_topology"] = True
        self.transformer = Flux3Model(model_config=model_config)

    def _setup_runtime_lora(self) -> None:
        runtime_lora = self.pipeline_config.runtime_lora
        if runtime_lora is None or self._runtime_lora_applications:
            return
        components = runtime_lora.target_components or ["transformer"]
        if len(components) != len(set(components)):
            raise ValueError("runtime_lora_config.target_components contains duplicates")
        unknown = set(components) - {"transformer"}
        if unknown and runtime_lora.strict:
            raise ValueError(f"Unknown FLUX.3 runtime LoRA target components: {sorted(unknown)}")
        if unknown:
            logger.warning(f"Runtime LoRA skipping non-transformer components: {sorted(unknown)}")
        if "transformer" not in components:
            return
        application = apply_flux3_runtime_lora(self.transformer, runtime_lora)
        self._runtime_lora_applications = [application]
        self.transformer.prepare_primsts_weights()

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        """Select only Qwen for offloading; keep the DiT and VAE resident."""
        return (("text_encoder",),)

    def offload_pipeline_components(self) -> dict[str, torch.nn.Module]:
        return {"text_encoder": self.text_encoder} if self.text_encoder is not None else {}

    def extra_offload_component_names(self) -> set[str]:
        # The loader validates stages before loading the auxiliary components.
        return {"text_encoder"}

    def load_standard_components(
        self, checkpoint_dir: str, device: torch.device, skip_components: list | None = None
    ) -> None:
        package = self.pipeline_config.extra_attrs["flux3_checkpoint"]
        skip_components = skip_components or []
        if "text_encoder" not in skip_components:
            config = package.config["text_encoder"]
            encoder_device = (
                torch.device("cpu")
                if "text_encoder" in self.offloader.requested_components()
                else device
            )
            self.text_encoder = (
                Qwen3VLEmbedder(
                    str(package.asset(config["path"])),
                    Qwen3VLEmbedderParams(
                        output_layer=config["layers"],
                        max_length=config["max_tokens"],
                        pad_multiple=config["pad_multiple"],
                        torch_dtype=config["dtype"],
                        use_compile=self.pipeline_config.torch_compile.enable,
                    ),
                    tokenizer_spec=str(package.asset(config["tokenizer"])),
                )
                .to(encoder_device)
                .eval()
            )
        if "vae" not in skip_components and "image_vae" not in skip_components:
            config = package.config["vae"]
            self.vae = Flux3ImageVAE(Flux3ImageVAEConfig())
            self.vae.load_state_dict(load_file(str(package.asset(config["path"]))))
            self.vae = self.vae.to(device=device, dtype=getattr(torch, config["dtype"])).eval()

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        package = self.pipeline_config.extra_attrs["flux3_checkpoint"]
        return load_file(str(package.weights))

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading FLUX.3 transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self._setup_cache_acceleration()

    def setup_parallel_vae(self) -> None:
        if self.pipeline_config.parallel.parallel_vae_size > 1:
            raise NotImplementedError(
                "FLUX.3 parallel VAE is not supported; use parallel_vae_size=1"
            )
        super().setup_parallel_vae()

    def torch_compile(self) -> None:
        """Per-block compile, overridden because FLUX.3 keeps its
        content blocks in ModuleDicts of ModuleLists that the base's flat-
        ModuleList discovery misses."""
        tc_config = self.pipeline_config.torch_compile
        if tc_config.enable_fullgraph:
            raise NotImplementedError(
                "FLUX.3 fullgraph compilation is not supported; use enable_fullgraph=False"
            )

        transformer = self.transformer
        if transformer is None:
            logger.warning("torch.compile: FLUX.3 transformer not initialized, skipping")
            return

        def _block_lists():
            for name, child in transformer.named_children():
                if isinstance(child, torch.nn.ModuleList) and len(child) > 0:
                    yield name, child
                elif isinstance(child, torch.nn.ModuleDict):
                    for key, sub in child.items():
                        if isinstance(sub, torch.nn.ModuleList) and len(sub) > 0:
                            yield f"{name}.{key}", sub

        for label, blocks in _block_lists():
            logger.info(f"torch.compile: transformer.{label} ({len(blocks)} blocks)")
            for i in range(len(blocks)):
                # Preserve BF16 intermediate rounding.
                blocks[i] = torch.compile(
                    blocks[i],
                    dynamic=None,
                    fullgraph=tc_config.enable_fullgraph,
                    options={"emulate_precision_casts": True},
                )

    @property
    def default_generation_params(self) -> dict:
        sampler = self.pipeline_config.extra_attrs["flux3_checkpoint"].config["sampler"]
        return dict(
            height=None,
            width=None,
            num_inference_steps=sampler["steps"],
            guidance_scale=sampler["guidance"],
            seed=0,
        )

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            "image_reference": RefSlotSpec(
                modality="image", roles=[RoleSpec(role="reference", min=0, max=None)]
            )
        }

    @property
    def extra_param_specs(self) -> dict:
        return {
            "mode": ExtraParamSchema(
                type="str",
                default=None,
                description="t2i or i2i; inferred from image references when omitted.",
            ),
            "pag_single_layers": ExtraParamSchema(
                type="list",
                default=None,
                description="Single-block indices for perturbed-attention guidance.",
            ),
            "pag_strength": ExtraParamSchema(
                type="float", default=1.0, range=(0.0, 1.0), description="PAG blend strength."
            ),
        }

    def _encode_prompt(self, prompt: str, negative_prompt: str | None) -> list[Tensor]:
        if self.transformer.flow_map and negative_prompt:
            raise ValueError("FLUX.3 flow-map checkpoints do not support negative prompts")
        prompts = [prompt] if self.transformer.flow_map else [negative_prompt or "", prompt]
        with self.offloader.context_if_requested("text_encoder"):
            return self.text_encoder.encode(prompts)

    @staticmethod
    def _to_uint8_image(x: Tensor) -> Tensor:
        """Convert VAE pixels in [-1, 1] to uint8 [B, H, W, C]."""
        x = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        return (x * 255).round().to(torch.uint8)

    def _package_output(self, out: Tensor | None) -> PipelineOutput:
        return PipelineOutput(image=self._to_uint8_image(out) if out is not None else None)

    @staticmethod
    def _text_position_ids(ctx: Tensor) -> Tensor:
        return torch.stack([prc_txt(sample)[1] for sample in ctx])

    def _run_denoise(
        self,
        *,
        x: Tensor,
        x_ids: Tensor,
        ctx,
        num_inference_steps: int | None,
        cfg: float | None,
        pag_single_layers: list[int] | None,
        pag_strength: float,
        cond_kwargs: dict | None = None,
        timer: CudaPhaseTimer | None = None,
    ) -> Tensor:
        contexts = list(ctx.split(1)) if isinstance(ctx, Tensor) else ctx
        cond_kwargs = cond_kwargs or {}
        flow_map = self.transformer.flow_map
        package = self.pipeline_config.extra_attrs["flux3_checkpoint"]
        sampler = package.config["sampler"]
        if num_inference_steps is None:
            num_inference_steps = sampler["steps"]
        if cfg is None:
            cfg = sampler["guidance"] if not flow_map else None
        if flow_map and cfg is not None:
            raise ValueError("FLUX.3 flow-map checkpoints sample without CFG; omit guidance")
        if flow_map and pag_single_layers:
            raise ValueError("FLUX.3 flow-map checkpoints do not support PAG")
        if flow_map and len(contexts) != 1:
            raise ValueError("FLUX.3 flow-map checkpoints require one conditional context")
        vgm = self.pipeline_config.visual_gen_mapping
        cfg_parallel = not flow_map and vgm is not None and vgm.cfg_size >= 2 and cfg > 1
        compute_dtype = self.transformer.model_config.torch_dtype
        # Match the image-small reference's BF16 latent updates.
        latent = x[0].to(compute_dtype)
        samples = []
        if flow_map:
            indices = (0,)
        elif cfg_parallel:
            indices = (1 if vgm.is_cfg_conditional else 0,)
        else:
            indices = (0, 1)
        for index in indices:
            text_ids = self._text_position_ids(contexts[index])
            sample = {"text": (contexts[index][0], text_ids[0]), "image": (latent, x_ids[0])}
            if "x_image_cond" in cond_kwargs:
                sample["image_cond"] = (
                    cond_kwargs["x_image_cond"][0],
                    cond_kwargs["x_image_cond_ids"][0],
                )
            samples.append(sample)
        packed = pack(
            samples, {"text": self.transformer.context_in_dim, **self.transformer.in_channels}
        )
        inputs = packed.kwargs()
        inputs["_flux3_uniform_timesteps"] = True
        prepared = self.transformer.prepare_static_inputs(**inputs)
        scheduler = Flux3FlowMatchScheduler(package.config["sampler"]["alpha"])
        scheduler.set_timesteps(num_inference_steps)
        target_kwargs = {}
        if flow_map:
            target_kwargs["timesteps_ref"] = inputs["x_text_timesteps"].new_empty(1)
        if self.cache_accelerator is not None and self.cache_accelerator.is_enabled():
            self.cache_accelerator.refresh(num_inference_steps)
        pag_samples = (0,) if not flow_map and indices[0] == 0 else ()
        if timer is not None:
            timer.mark_denoise_start()
        for step_index, timestep in self._profile_denoise_steps(scheduler.timesteps):
            inputs["x_image_timesteps"].fill_(timestep.item())
            if flow_map:
                target_kwargs["timesteps_ref"].fill_(scheduler.sigmas[step_index + 1].item())
            predictions = self.transformer(
                **inputs,
                **prepared,
                **target_kwargs,
                pag_single_layers=pag_single_layers,
                pag_strength=pag_strength,
                pag_samples=pag_samples,
            )["x_image"]
            if flow_map:
                velocity = predictions
            elif cfg_parallel:
                gathered = [torch.empty_like(predictions) for _ in range(vgm.cfg_size)]
                torch.distributed.all_gather(
                    gathered, predictions.contiguous(), group=vgm.cfg_group
                )
                conditional, unconditional = gathered[:2]
            else:
                unconditional, conditional = predictions.chunk(2)
            if not flow_map:
                velocity = unconditional + cfg * (conditional - unconditional)
            latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
            inputs["x_image"] = torch.cat([latent] * len(indices))
        return latent.unsqueeze(0)

    def _sample_image_latents(self, height: int, width: int, seed: int):
        device = next(self.transformer.parameters()).device
        return sample_image_latents(
            height=height,
            width=width,
            seed=seed,
            device=device,
            image_rope_scale=self.pipeline_config.extra_attrs["flux3_checkpoint"].config["dit"][
                "image_rope_scale"
            ],
            copies=1 if self.transformer.flow_map else 2,
        )

    def _prepare_t2i(
        self,
        ctx,
        height=1024,
        width=1024,
        num_inference_steps=None,
        cfg=None,
        seed=0,
        pag_single_layers=None,
        pag_strength=1.0,
        timer: CudaPhaseTimer | None = None,
    ):
        x_image, x_image_ids, x_image_orig_ids = self._sample_image_latents(height, width, seed)
        latent = self._run_denoise(
            x=x_image,
            x_ids=x_image_ids,
            ctx=ctx,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
            timer=timer,
        )
        return latent, x_image_orig_ids

    def _prepare_i2i(
        self,
        ctx,
        image_cond,
        height=1024,
        width=1024,
        num_inference_steps=None,
        cfg=None,
        seed=0,
        pag_single_layers=None,
        pag_strength=1.0,
        timer: CudaPhaseTimer | None = None,
    ):
        x_image, x_image_ids, x_image_orig_ids = self._sample_image_latents(height, width, seed)
        device = x_image.device
        image_vae_dtype = next(self.vae.parameters()).dtype
        images = [image_cond] if isinstance(image_cond, Tensor) else image_cond
        image_latents = [self.vae.encode(image.to(image_vae_dtype))[0] for image in images]
        x_image_cond, x_image_cond_ids, _ = pack_i2i_image_condition(
            image_latents,
            device=device,
            image_rope_scale=self.pipeline_config.extra_attrs["flux3_checkpoint"].config["dit"][
                "image_rope_scale"
            ],
            copies=1 if self.transformer.flow_map else 2,
        )
        latent = self._run_denoise(
            x=x_image,
            x_ids=x_image_ids,
            ctx=ctx,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
            cond_kwargs=dict(
                x_image_cond=x_image_cond,
                x_image_cond_ids=x_image_cond_ids,
            ),
            timer=timer,
        )
        return latent, x_image_orig_ids

    @torch.inference_mode()
    def generate_t2i(
        self,
        ctx: list[Tensor],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        cfg: float | None = None,
        seed: int = 0,
        pag_single_layers: Optional[List[int]] = None,
        pag_strength: float = 1.0,
    ) -> Tensor | None:
        """Generate from CFG text embeddings, or one conditional context for a flow map."""
        latent, orig_ids = self._prepare_t2i(
            ctx=ctx,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            seed=seed,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
        )
        return self._decode_latents((latent, orig_ids))

    @torch.inference_mode()
    def generate_i2i(
        self,
        ctx: list[Tensor],
        image_cond: Tensor | list[Tensor],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        cfg: float | None = None,
        seed: int = 0,
        pag_single_layers: Optional[List[int]] = None,
        pag_strength: float = 1.0,
    ) -> Tensor | None:
        """Generate an image conditioned on one or more reference images."""
        latent, orig_ids = self._prepare_i2i(
            ctx=ctx,
            image_cond=image_cond,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            seed=seed,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
        )
        return self._decode_latents((latent, orig_ids))

    def _decode_latents(self, latent_result: tuple) -> Tensor | None:
        latent, ids = latent_result
        return self.decode_latents(
            _as_image_grid(latent, ids),
            lambda grid: self.vae.decode(grid.to(next(self.vae.parameters()).dtype)),
        )

    def forward(self, mode: str, **kwargs) -> Tensor | None:
        if mode not in ("t2i", "i2i"):
            raise ValueError(f"Unsupported FLUX.3 image mode: {mode}")
        return {"t2i": self.generate_t2i, "i2i": self.generate_i2i}[mode](**kwargs)

    @torch.inference_mode()
    def infer(self, req) -> PipelineOutput:
        p = req.params
        extra = p.extra_params or {}
        mode = extra.get("mode") or ("i2i" if p.image_reference else "t2i")
        if mode not in ("t2i", "i2i"):
            raise ValueError(f"Unsupported FLUX.3 image mode: {mode}")
        if isinstance(req.prompt, (list, tuple)) and len(req.prompt) != 1:
            raise ValueError("FLUX.3 serves one prompt per request")
        if mode == "i2i" and not p.image_reference:
            raise ValueError("i2i requires at least one image reference")
        if mode == "t2i" and p.image_reference:
            raise ValueError("Use mode=i2i with image references")
        prompt = req.prompt[0] if isinstance(req.prompt, (list, tuple)) else req.prompt
        if self.transformer.flow_map:
            if p.guidance_scale is not None:
                raise ValueError("FLUX.3 flow-map checkpoints sample without CFG; omit guidance")
            if extra.get("pag_single_layers"):
                raise ValueError("FLUX.3 flow-map checkpoints do not support PAG")
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        ctx = self._encode_prompt(prompt, p.negative_prompt)
        images = [
            _load_image_pixels(ref.content, device=self.device) for ref in p.image_reference or []
        ]
        width, height = self.pipeline_config.extra_attrs["flux3_checkpoint"].config["sampler"][
            "size"
        ]
        if images:
            h, w = images[0].shape[-2:]
            scale = math.sqrt(width * height / (w * h))
            width, height = max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)
        kwargs = dict(
            ctx=ctx,
            height=p.height or height,
            width=p.width or width,
            num_inference_steps=p.num_inference_steps,
            cfg=p.guidance_scale,
            seed=p.seed,
            pag_single_layers=extra.get("pag_single_layers"),
            pag_strength=extra.get("pag_strength", 1.0),
            timer=timer,
        )
        if mode == "i2i":
            kwargs["image_cond"] = images
        from .flux3_gemm_config import autotune_context

        with autotune_context(self.device, self.rank, tune=False):
            latent = {"t2i": self._prepare_t2i, "i2i": self._prepare_i2i}[mode](**kwargs)
        timer.mark_post_start()
        result = self._package_output(self._decode_latents(latent))
        timer.mark_end()
        return timer.fill(result)
