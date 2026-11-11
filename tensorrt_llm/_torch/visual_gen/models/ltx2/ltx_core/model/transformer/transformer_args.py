from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.adaln import AdaLayerNormSingle
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.modality import Modality
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)

#: Returns the model's keyframe absolute-position embedding, or ``None`` when the model has none.
#: A provider rather than the tensor itself because a parameter the checkpoint did not supply is
#: materialized later, which *replaces* the parameter object -- a directly-held reference would
#: keep pointing at the stale meta tensor.
KeyframesEmbeddingProvider = Callable[[], torch.Tensor | None]


def apply_keyframes_absolute_embedding(
    hidden_states: torch.Tensor,
    keyframes_mask: torch.Tensor | None,
    embedding_provider: KeyframesEmbeddingProvider | None,
) -> torch.Tensor:
    """Add the learned keyframe marker to single-pixel-frame tokens.
    Applied to projected hidden states, immediately after ``patchify_proj``. The embedding is
    zero-initialized, so this is an exact no-op until it is trained -- and a no-op forever for
    models built without ``use_keyframes_abs_pos_embedding``, whose provider yields ``None``.
    Args:
        hidden_states: ``(B, T, D)`` projected tokens.
        keyframes_mask: ``(B, T, 1)`` marker, or ``None`` for "no token is marked".
        embedding_provider: Resolves the ``(1, D)`` embedding, or ``None`` if the model has none.
    """
    if embedding_provider is None or keyframes_mask is None:
        return hidden_states
    embedding = embedding_provider()
    if embedding is None:
        return hidden_states
    mask = (keyframes_mask > 0).to(dtype=hidden_states.dtype)
    return hidden_states + mask * embedding.to(dtype=hidden_states.dtype)


@dataclass(frozen=True)
class AdaLNTimestepEmbedding:
    """Projected AdaLN timesteps and the token index that selects their rows.
    Transformer blocks consume ``modulation`` and carry ``embedding`` to the model
    output head, which applies the final scale and shift.
    Compact-table indices have the batch offset folded in and address the first two
    dimensions of any projected table passed to ``gather``.
    """

    modulation: torch.Tensor
    embedding: torch.Tensor
    token_indices: torch.Tensor | None = None

    def gather(self, values: torch.Tensor) -> torch.Tensor:
        """Expand a compact table to one row per token."""
        if self.token_indices is None:
            return values
        return values.reshape(-1, *values.shape[2:])[self.token_indices]

    def per_token_embedding(self) -> torch.Tensor:
        return self.gather(self.embedding)


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timestep_embedding: AdaLNTimestepEmbedding
    positional_embeddings: tuple[torch.Tensor, torch.Tensor]
    cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = (
        None  # Additive log-space self-attention bias (B, 1, T, T), None = full attention
    )
    # Per-block perturbation state, precomputed by `LTXModel._process_transformer_blocks`
    # so the block forward needs no per-block identity. The bool shortcuts
    # (`*_all_perturbed`, `cross_attn_skip_all`) are Python bools that Dynamo specialises
    # on — fine because they're stable across denoising steps for a fixed perturbation
    # config.
    self_attn_perturbation_mask: torch.Tensor | None = None
    self_attn_all_perturbed: bool = False
    cross_attn_perturbation_mask: torch.Tensor | None = None
    cross_attn_skip_all: bool = False


class BlockPerturbationsProcessor:
    """Per-block preparation of ``TransformerArgs``.
    The base implementation returns a copy of ``args`` with this block's
    precomputed perturbation flags and masks attached. Subclasses can layer in
    operations that must run on each block's inputs but stay outside the
    compile boundary -- e.g. ``torch._dynamo.mark_dynamic`` for
    shape-polymorphic block compilation (see ``compiling.py``). Swapping the
    processor on an ``LTXModel`` instance is how compile transforms opt in to
    such behaviour without baking it into the model's forward.
    ``self_attn_perturbation_mask`` is None when all or none of the batch is
    perturbed (the attention call can take the shortcut path). ``cross_attn_*``
    is None when every sample skips the cross-attention entirely.
    """

    def _block_guards(
        self,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_type: PerturbationType,
        cross_type: PerturbationType,
    ) -> tuple[bool, bool, bool]:
        """The values the block's attention branches specialise on: ``(self_mask_present, all_self,
        all_cross)``. Self-attn has three paths -- none/partial/all -- so it needs both whether a
        blend mask is present (partial) and ``all_self`` (skip); cross-attn multiplies its mask
        unconditionally when it runs, so only ``all_cross`` (skip the whole block) matters. Mask
        VALUES below this ride in as runtime data and do not change which path runs."""
        all_self = perturbations.all_in_batch(self_type, block_idx)
        any_self = perturbations.any_in_batch(self_type, block_idx)
        all_cross = perturbations.all_in_batch(cross_type, block_idx)
        return (any_self and not all_self), all_self, all_cross

    def __call__(
        self,
        args: "TransformerArgs",
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> "TransformerArgs":
        self_mask_present, all_self, all_cross = self._block_guards(
            perturbations, block_idx, self_attn_type, cross_attn_type
        )
        return replace(
            args,
            self_attn_perturbation_mask=perturbations.mask(self_attn_type, block_idx) if self_mask_present else None,
            self_attn_all_perturbed=all_self,
            cross_attn_perturbation_mask=None if all_cross else perturbations.mask(cross_attn_type, block_idx),
            cross_attn_skip_all=all_cross,
        )

    def graph_signature(self, perturbations: BatchedPerturbationConfig) -> Hashable:
        """Identity of the block graph for this perturbation config: the per-block guard values the
        attention branches specialise on (see ``_block_guards``). A CUDA-graph capture keyed on it
        recaptures when -- and only when -- the block would recompile; mask values below this
        granularity ride in as copy-in runtime data and never change the graph."""
        assert perturbations.block_masks_cpu is not None, "graph signature needs the host mask mirror"
        num_blocks = perturbations.block_masks_cpu.shape[1]
        # (self-attn, cross-attn) types per modality, as _process_transformer_blocks applies them.
        modality_types = (
            (PerturbationType.SKIP_VIDEO_SELF_ATTN, PerturbationType.SKIP_A2V_CROSS_ATTN),
            (PerturbationType.SKIP_AUDIO_SELF_ATTN, PerturbationType.SKIP_V2A_CROSS_ATTN),
        )
        return tuple(
            self._block_guards(perturbations, block_idx, self_type, cross_type)
            for block_idx in range(num_blocks)
            for self_type, cross_type in modality_types
        )


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        keyframes_embedding_provider: KeyframesEmbeddingProvider | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        self.keyframes_embedding_provider = keyframes_embedding_provider

    def _prepare_timestep(
        self, timestep: torch.Tensor, adaln: AdaLayerNormSingle, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare context for transformer blocks."""
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Prepare self-attention mask by converting [0,1] values to additive log-space bias.
        Input shape: 3D ``(B, T_q, T_k)`` with values in [0, 1]. The dense form
        is ``(B, T, T)``; broadcastable forms like ``(1, 1, T)`` (key-only
        padding) or ``(B, 1, T)`` are also valid and yield a correspondingly
        broadcastable output.
        Output shape: ``(B, 1, T_q, T_k)`` (heads dim inserted) with 0.0 for
        full attention and a large negative value for masked positions.
        Positions with attention_mask <= 0 are fully masked (mapped to the dtype's minimum
        representable value). Strictly positive entries are converted via log-space for
        smooth attenuation, with small values clamped for numerical stability.
        Returns None if input is None (no masking).
        """
        if attention_mask is None:
            return None

        # Convert [0, 1] attention mask to additive log-space bias:
        #   1.0 -> log(1.0) = 0.0  (no bias, full attention)
        #   0.0 -> finfo.min        (fully masked)
        # Must stay elementwise: boolean indexing would sync the host on every forward.
        finfo = torch.finfo(x_dtype)
        bias = torch.where(
            attention_mask > 0,
            torch.log(attention_mask.clamp(min=finfo.tiny)).to(x_dtype),
            torch.full_like(attention_mask, finfo.min, dtype=x_dtype),
        )
        return bias.unsqueeze(1)  # (B, 1, T_q, T_k) for head broadcast

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,  # noqa: ARG002
    ) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        x = apply_keyframes_absolute_embedding(x, modality.keyframes_mask, self.keyframes_embedding_provider)
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps.values, self.adaln, batch_size, modality.latent.dtype
        )
        timestep_indices = modality.timesteps.indices
        if timestep_indices is not None:
            # Fold the batch into the index so a consumer resolves batch and value with one
            # gather over the table flattened to (B * K, ...). Every consumer of
            # `AdaLNTimestepEmbedding` reads them in that flattened form.
            batch_offsets = torch.arange(batch_size, device=x.device, dtype=torch.int32)
            timestep_indices = timestep_indices + batch_offsets[:, None] * timestep.shape[1]
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
            )
        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timestep_embedding=AdaLNTimestepEmbedding(timestep, embedded_timestep, timestep_indices),
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        keyframes_embedding_provider: KeyframesEmbeddingProvider | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
            keyframes_embedding_provider=keyframes_embedding_provider,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args

        if cross_modality.sigma.numel() > 1:
            if cross_modality.sigma.shape[0] != modality.timesteps.values.shape[0]:
                raise ValueError("Cross modality sigma must have the same batch size as the modality")
            if cross_modality.sigma.ndim != 1:
                raise ValueError("Cross modality sigma must be a 1D tensor")

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            modality_timesteps=modality.timesteps.values,
            cross_modality_sigma=cross_modality.sigma,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        modality_timesteps: torch.Tensor,
        cross_modality_sigma: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare A-V cross-attention AdaLN inputs."""
        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            (modality_timesteps * timestep_scale_multiplier).flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])

        gate_noise_timestep, _ = self.cross_gate_adaln(
            (cross_modality_sigma * timestep_scale_multiplier * av_ca_factor).flatten(),
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
