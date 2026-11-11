import logging
from enum import Enum

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.model_protocol import LTXModelProtocol
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import AttentionOps, attention_label
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.modality import Modality
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import LTXRopeType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.transformer_args import (
    AdaLNTimestepEmbedding,
    BlockPerturbationsProcessor,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import to_denoised

logger = logging.getLogger(__name__)


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module, Disposable):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_ops: AttentionOps | None = None,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        use_prompt_adaln_single: bool = True,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
    ):
        super().__init__()
        attention_ops = attention_ops or AttentionOps()
        # Log the attention backends this transformer is built with. Reading the resolved
        # ``label`` off the ops reports whatever was selected -- AUTOMATIC, an explicit pin
        # (PYTORCH/FA3/FA4/SDPA_*), or a directly supplied callable -- so this is the
        # single source of truth for which kernel a build uses. Fires once per build.
        logger.info(
            "Building transformer with attention backends -- self: %s, masked: %s",
            attention_label(attention_ops.attention_function),
            attention_label(attention_ops.masked_attention_function),
        )
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_prompt_adaln_single = use_prompt_adaln_single
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        self.use_keyframes_abs_pos_embedding = use_keyframes_abs_pos_embedding
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_ops=attention_ops,
            apply_gated_attention=apply_gated_attention,
            ff_bias=ff_bias,
            audio_ff_bias=audio_ff_bias,
        )
        # Hook for per-block input prep. Compile transforms in `compiling.py`
        # wrap (not replace) this with a processor that also marks the seq dim
        # dynamic, so any caller customisation here is preserved as the inner.
        self.block_input_processor = BlockPerturbationsProcessor()

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _keyframes_embedding(self) -> torch.Tensor | None:
        """Look the parameter up on each call.
        Deliberately not a captured reference: this parameter is absent from checkpoints that
        predate it, so it is created on the meta device and only later materialized -- which
        replaces the parameter object. A stale reference would keep pointing at the meta tensor.
        """
        return getattr(self, "keyframes_abs_pos_embedding", None)

    @property
    def supports_keyframes_abs_pos_embedding(self) -> bool:
        """Whether this model has a usable keyframe absolute-position embedding.
        False for models built without the flag, and also for a model whose config set the flag
        but whose checkpoint carried no weight for it (the parameter would still be on ``meta``).
        """
        embedding = self._keyframes_embedding()
        return embedding is not None and not embedding.is_meta

    def enable_keyframes_abs_pos_embedding(self) -> None:
        """Ensure the keyframe embedding exists and holds real zeros, creating it if needed.
        Covers both ways a loaded model can arrive without a usable parameter:
        - The checkpoint predates the feature entirely, so its config never set
          ``use_keyframes_abs_pos_embedding`` and no parameter was built. It is created here.
        - The config did set the flag but the checkpoint carried no weight for it. Models are
          built on the meta device and loaded with ``strict=False, assign=True``, so such a
          parameter stays on ``meta`` and would fail at the first forward.
        Idempotent, and never overwrites a parameter that already holds real storage -- a
        checkpoint carrying a trained embedding keeps it. The preprocessors resolve the parameter
        through a provider on every call, so enabling it after the model is built is safe.
        A zero embedding is an exact no-op, so this only makes the marker *harmless*, not
        meaningful: it does not give an untrained checkpoint the keyframe capability.
        """
        if not self.model_type.is_video_enabled():
            raise ValueError("The keyframe absolute-position embedding is a video-stream parameter")

        existing = self._keyframes_embedding()
        if existing is not None and not existing.is_meta:
            return

        reference = self.patchify_proj.weight
        shape = existing.shape if existing is not None else (1, self.inner_dim)
        dtype = existing.dtype if existing is not None else reference.dtype
        self.use_keyframes_abs_pos_embedding = True
        self.keyframes_abs_pos_embedding = torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device=reference.device))

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        # Marks tokens whose latent encodes a single standalone pixel frame. Zero-initialized, so a
        # checkpoint that predates it behaves identically until the parameter is trained.
        self.keyframes_abs_pos_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.inner_dim)) if self.use_keyframes_abs_pos_embedding else None
        )

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_ops: AttentionOps | None,
        apply_gated_attention: bool,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=ff_bias,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=audio_ff_bias,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_ops=attention_ops,
                )
                for _ in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks."""
        return len(self.transformer_blocks)

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Process transformer blocks for LTX."""
        for block_idx, block in enumerate(self.transformer_blocks):
            if video is not None:
                video = self.block_input_processor(
                    video,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                )
            if audio is not None:
                audio = self.block_input_processor(
                    audio,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                )

            if self._enable_gradient_checkpointing and self.training:
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    use_reentrant=False,
                )
            else:
                video, audio = block(video=video, audio=audio)

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        timestep_embedding: AdaLNTimestepEmbedding,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        embedded_timestep = timestep_embedding.per_token_embedding()
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Materialize the no-perturbation mask here (eager); a None config means "perturb nothing"
        # -> all-keep masks. The block loop never builds masks.
        if perturbations is None:
            ref = (video_args or audio_args).x
            perturbations = BatchedPerturbationConfig.empty(ref.shape[0], self.num_blocks, ref.device, ref.dtype)
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.timestep_embedding,
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.timestep_embedding,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module, Disposable):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks."""
        return self.velocity_model.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module, Disposable):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks."""
        return self.velocity_model.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps.per_token()) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps.per_token()) if ax is not None else None
        return denoised_video, denoised_audio
