# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoTransformerPipeline — standalone Visual Gen pipeline for the auto path.

Not a `BasePipeline` subclass. The handwritten path's `BasePipeline` makes
strong assumptions about the load lifecycle (`MetaInitMode` wrap →
`_materialize_meta_tensors` → `load_transformer_weights` →
`load_standard_components` → ...) that don't apply to the auto path:
Diffusers' `DiffusionPipeline.from_pretrained` loads weights and standard
components eagerly in one call, and that call must run outside any
meta-tensor dispatch.

Rather than subclass `BasePipeline` and silently no-op the lifecycle hooks
(which led to a "subclass-as-marker" anti-pattern flagged in code review),
this class stands on its own and exposes only the surface the executor and
loader actually consume:

* ``infer(req: DiffusionRequest) → MediaOutput`` — executor entry point
* ``warmup()`` / ``warmup_cache_key`` / ``_warmed_up_shapes`` — warmup contract
* ``default_generation_params`` / ``extra_param_specs`` — executor defaulting
* ``mapping`` / ``model_config`` / ``transformer`` / ``vae`` / ... — view-only

The loader (`pipeline_loader.py`) detects this class via ``isinstance`` and
branches around the handwritten-only hooks. Construction is invoked from
``pipeline_loader.PipelineLoader._create_pipeline`` under
``pipeline_mode in {"auto", "fallback-and-unregistered"}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..output import MediaOutput

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig
    from ..executor import DiffusionRequest


# Family-name → default warmup shapes / frame counts. Used when the user
# didn't pin `model_config.compilation.resolutions`/`num_frames`. Tokens are
# matched against `type(diffusers_pipe).__name__`.
_DEFAULT_RES_BY_FAMILY: Tuple[Tuple[str, List[Tuple[int, int]]], ...] = (
    ("Wan", [(720, 1280)]),
    ("LTX2", [(512, 768)]),
    ("LTX", [(512, 704)]),
    ("Flux2", [(1024, 1024)]),
    ("Flux", [(1024, 1024)]),
    ("PixArt", [(1024, 1024)]),
    ("Sana", [(1024, 1024)]),
    ("StableDiffusion3", [(1024, 1024)]),
)

_DEFAULT_FRAMES_BY_FAMILY: Tuple[Tuple[str, List[int]], ...] = (
    ("Wan", [81]),
    ("LTX2", [121]),
    ("LTX", [33]),
)


def _match_family(name: str, table: Tuple[Tuple[str, Any], ...], default: Any) -> Any:
    for tok, value in table:
        if tok in name:
            return value
    return default


class AutoTransformerPipeline(nn.Module):
    """Standalone pipeline for the auto path. NOT a `BasePipeline` subclass —
    see module docstring for why.

    All heavy lifting (Diffusers load + capture + rewrite + compile) happens
    in `__init__`. There is no separate "load weights" / "load standard
    components" phase: by the time `__init__` returns, the pipeline is ready
    to serve `warmup()` and `infer(req)`.
    """

    def __init__(
        self,
        model_config: "DiffusionModelConfig",
        checkpoint_dir: str,
    ) -> None:
        super().__init__()
        # Surface known divergences vs the handwritten path BEFORE we start
        # the expensive load — operator sees the warnings even if Diffusers
        # crashes mid-load.
        self._warn_silent_divergences(model_config)

        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.mapping: Mapping = getattr(model_config, "mapping", None) or Mapping()
        self._warmed_up_shapes: set = set()
        self._checkpoint_dir = checkpoint_dir
        # Components that the executor / introspection code may read.
        # Populated below from the inner Diffusers pipeline.
        self.transformer: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None

        # Lazy import to keep auto_pipeline.py / pipeline.py / families/* import
        # cycles tame.
        from .auto_pipeline import AutoDiffusersPipeline

        compile_cfg = getattr(model_config, "torch_compile", None)
        enable_compile = bool(getattr(compile_cfg, "enable_torch_compile", False))
        vgm = getattr(model_config, "visual_gen_mapping", None)

        self._inner: AutoDiffusersPipeline = AutoDiffusersPipeline(
            checkpoint_dir,
            model_config,
            visual_gen_mapping=vgm,
            enable_torch_compile=enable_compile,
            # `reduce-overhead` (CUDA graphs) is unsafe when Diffusers
            # re-allocates the per-step `latents` tensor. See
            # `auto_pipeline.AutoDiffusersPipeline.__init__` docstring.
            torch_compile_mode="default",
        )
        dpipe = self._inner.diffusers_pipe
        self._family_name = type(dpipe).__name__
        self.transformer = self._inner.captured_transformer
        self.vae = getattr(dpipe, "vae", None)
        self.text_encoder = getattr(dpipe, "text_encoder", None)
        self.tokenizer = getattr(dpipe, "tokenizer", None)
        self.scheduler = getattr(dpipe, "scheduler", None)

    # ------------------------------------------------------------------
    # Pre-construction divergence warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _warn_silent_divergences(model_config: "DiffusionModelConfig") -> None:
        """Log warnings for `VisualGenArgs` fields the auto path silently
        no-ops. Mirrors what the handwritten path supports but auto doesn't.

        Listed here (not in `AutoDiffusersPipeline`) because these are
        product-level features the operator selects via config; the inner
        factory is a Diffusers-shaped utility unaware of `VisualGenArgs`.
        """
        # Cache acceleration (TeaCache / Cache-DiT). `cache_backend` is
        # `Optional[Literal["teacache", "cache_dit"]]` — falsy means no
        # cache backend selected; we don't need a `!= "none"` clause.
        cache_backend = getattr(model_config, "cache_backend", None)
        if cache_backend:
            logger.warning(
                f"AutoTransformerPipeline: cache_backend={cache_backend!r} is "
                "a no-op on the auto path "
                "(transformer wrapper's `cache_context` returns nullcontext)."
            )
        # Per-layer quantization.
        if getattr(model_config, "quant_config_dict", None):
            logger.warning(
                "AutoTransformerPipeline: per-layer `quant_config_dict` is not "
                "wired on the auto path — only the global `quant_config` is "
                "applied to all Linears."
            )
        # Parallel VAE.
        mapping = getattr(model_config, "mapping", None)
        ws = getattr(mapping, "world_size", 1) if mapping is not None else 1
        if getattr(model_config, "enable_parallel_vae", False) and ws > 1:
            logger.warning(
                "AutoTransformerPipeline: enable_parallel_vae=True is a no-op "
                "on the auto path (no `vae_adapter_class` registered)."
            )
        # CUDA graphs.
        cuda_graph_cfg = getattr(model_config, "cuda_graph", None)
        if cuda_graph_cfg is not None and getattr(cuda_graph_cfg, "enable_cuda_graph", False):
            logger.warning(
                "AutoTransformerPipeline: cuda_graph.enable_cuda_graph=True is "
                "a no-op on the auto path. Use "
                "`torch_compile.enable_torch_compile=True` instead "
                "(Inductor codegen, mode='default')."
            )

    # ------------------------------------------------------------------
    # Executor / loader duck-typed surface
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def dtype(self) -> torch.dtype:
        if self.transformer is not None:
            try:
                return next(self.transformer.parameters()).dtype
            except StopIteration:
                pass
        return torch.float32

    @property
    def device(self) -> torch.device:
        return self._inner.device if self._inner is not None else torch.device("cpu")

    @property
    def default_generation_params(self) -> dict:
        """Executor consumes this for `_merge_defaults`. The auto path has
        no opinion — Diffusers pipelines define their own per-call defaults
        and `infer()` drops `None` fields before calling the pipe.
        """
        return {}

    @property
    def extra_param_specs(self) -> dict:
        """No model-specific schema-validated extras on the auto path."""
        return {}

    def warmup_cache_key(self, height: int, width: int, num_frames: int) -> tuple:
        # Image families' Diffusers `__call__` doesn't take `num_frames`;
        # video families do.
        if not self._is_video_family():
            return (height, width)
        return (height, width, num_frames)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return _match_family(self._family_name, _DEFAULT_RES_BY_FAMILY, [(1024, 1024)])

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return _match_family(self._family_name, _DEFAULT_FRAMES_BY_FAMILY, [1])

    @property
    def default_warmup_steps(self) -> int:
        return 2

    def _resolve_warmup_plan(self) -> Tuple[List[Tuple[int, int, int]], int]:
        """Same precedence as `BasePipeline.resolve_warmup_plan`."""
        warmup_cfg = self.model_config.compilation
        if warmup_cfg.resolutions is not None or warmup_cfg.num_frames is not None:
            resolutions = warmup_cfg.resolutions or self.default_warmup_resolutions
            num_frames_list = warmup_cfg.num_frames or self.default_warmup_num_frames
        else:
            resolutions = self.default_warmup_resolutions
            num_frames_list = self.default_warmup_num_frames
        import itertools

        shapes = [(h, w, f) for (h, w), f in itertools.product(resolutions, num_frames_list)]
        return shapes, self.default_warmup_steps

    def warmup(self) -> None:
        """Run warmup inference to trigger torch.compile + populate
        `_warmed_up_shapes`. OOM and other failures fast-fail at startup —
        matches the handwritten `BasePipeline.warmup` contract.
        """
        import time

        shapes, steps = self._resolve_warmup_plan()
        if not shapes:
            logger.info("Warmup disabled (no warmup shapes)")
            return
        logger.info(f"AutoTransformerPipeline warmup: {len(shapes)} shapes, {steps} steps each")
        t0 = time.time()
        for h, w, f in shapes:
            logger.info(f"Warmup: {h}x{w}, {f} frames, {steps} steps")
            self._run_warmup(h, w, f, steps)
            torch.cuda.synchronize()
        self._warmed_up_shapes = {self.warmup_cache_key(h, w, f) for h, w, f in shapes}
        logger.info(f"Warmup completed in {time.time() - t0:.2f}s")

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        kwargs: Dict[str, Any] = {
            "prompt": "warmup",
            "height": height,
            "width": width,
            "num_inference_steps": steps,
        }
        if self._is_video_family():
            kwargs["num_frames"] = num_frames
            kwargs["output_type"] = "pt"
        self._inner._pipe(**kwargs)

    # ------------------------------------------------------------------
    # Family classification (used by infer + warmup)
    # ------------------------------------------------------------------

    def _is_video_family(self) -> bool:
        return any(tok in self._family_name for tok in ("Wan", "LTX"))

    def _is_audio_visual_family(self) -> bool:
        return "LTX2" in self._family_name

    def _is_distilled_no_negative(self) -> bool:
        # FLUX.1 / FLUX.2 (incl. schnell, dev, Kontext, klein) are
        # guidance-distilled and reject `negative_prompt`. `startswith("Flux")`
        # subsumes `Flux2` — listed once for clarity.
        return self._family_name.startswith("Flux")

    # ------------------------------------------------------------------
    # Executor entry point
    # ------------------------------------------------------------------

    def infer(self, req: "DiffusionRequest") -> MediaOutput:
        """Translate a `DiffusionRequest` into Diffusers pipe kwargs, call the
        pipe, wrap the result as `MediaOutput`.
        """
        generator = torch.Generator(device=self.device).manual_seed(req.seed)
        kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "height": req.height,
            "width": req.width,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "generator": generator,
        }
        if req.negative_prompt and not self._is_distilled_no_negative():
            kwargs["negative_prompt"] = req.negative_prompt
        if req.num_frames is not None and self._is_video_family():
            kwargs["num_frames"] = req.num_frames
        if req.frame_rate is not None and self._is_audio_visual_family():
            kwargs["frame_rate"] = req.frame_rate
        if req.max_sequence_length is not None and any(
            tok in self._family_name
            for tok in ("StableDiffusion3", "Flux", "Wan", "PixArt", "LTX", "Sana")
        ):
            kwargs["max_sequence_length"] = req.max_sequence_length
        if req.extra_params:
            kwargs.update(req.extra_params)
        # Re-apply the distilled-no-negative guard AFTER extra_params merge —
        # otherwise a user passing `extra_params={"negative_prompt": ...}` can
        # smuggle the kwarg back into a FLUX-family call that rejects it.
        if self._is_distilled_no_negative():
            kwargs.pop("negative_prompt", None)
        # Drop None values so Diffusers uses each pipeline's defaults.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["output_type"] = "pt"

        result = self._inner._pipe(**kwargs)
        return _wrap_diffusers_output(result, self._family_name)


# ---------------------------------------------------------------------------
# Diffusers pipeline output → MediaOutput
# ---------------------------------------------------------------------------


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Scale [0, 1] float tensor → uint8."""
    return x.clamp(0, 1).mul(255).round().to(torch.uint8)


def _wrap_diffusers_output(result: Any, family_name: str) -> MediaOutput:
    """Adapt a Diffusers pipeline result (`.images`, `.frames`, `.audio`) to
    the unified `MediaOutput(image=..., video=..., audio=...)` shape.

    Conventions (matching the handwritten path):
    - `image`: `(H, W, C)` uint8 — first image of batch.
    - `video`: `(T, H, W, C)` uint8 — first sample of batch.
    - `audio`: tensor as-returned by the pipeline.
    """
    out = MediaOutput()
    is_video_family = any(tok in family_name for tok in ("Wan", "LTX"))

    images = getattr(result, "images", None)
    if images is not None and not is_video_family:
        # Diffusers image pipelines with `output_type="pt"` return
        # `(B, C, H, W)` float in [0, 1].
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            out.image = _to_uint8(images[0]).permute(1, 2, 0).contiguous()
        return out

    frames = getattr(result, "frames", None)
    if frames is not None:
        # WAN: `(B, T, H, W, C)`; LTX-Video / LTX-2: `(B, T, C, H, W)`.
        # Use the family hint instead of shape-sniffing to avoid the
        # degenerate `(T, H, W, C=3)` ↔ `(T, C=3, H, W)` ambiguity.
        if isinstance(frames, torch.Tensor):
            vid = frames[0] if frames.dim() == 5 else frames
            if isinstance(vid, torch.Tensor) and vid.dim() == 4:
                if "Wan" in family_name:
                    # Already (T, H, W, C).
                    out.video = _to_uint8(vid).contiguous()
                else:
                    # LTX family: (T, C, H, W) → permute to (T, H, W, C).
                    out.video = _to_uint8(vid).permute(0, 2, 3, 1).contiguous()
        elif isinstance(frames, (list, tuple)) and frames:
            first = frames[0]
            if isinstance(first, torch.Tensor) and first.dim() == 4:
                vid = first
                if "LTX" in family_name:
                    vid = vid.permute(0, 2, 3, 1)
                out.video = _to_uint8(vid).contiguous()

    audio = getattr(result, "audio", None)
    if audio is not None and isinstance(audio, torch.Tensor):
        out.audio = audio
    return out
