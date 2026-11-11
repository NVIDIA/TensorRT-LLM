from dataclasses import dataclass, replace

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.feed_forward import FeedForward
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import LTXRopeType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.transformer_args import AdaLNTimestepEmbedding, TransformerArgs
from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        attention_ops: AttentionOps | None = None,
    ):
        super().__init__()

        if attention_ops is None:
            attention_ops = AttentionOps()
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep_embedding: AdaLNTimestepEmbedding,
        indices: slice,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        hidden_dim = scale_shift_table.shape[1]
        modulation = timestep_embedding.modulation if modulation is None else modulation
        modulation = modulation.reshape(batch_size, -1, num_ada_params, hidden_dim)[:, :, indices, :]
        modulation = timestep_embedding.gather(modulation)
        return (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=modulation.device, dtype=modulation.dtype)
            + modulation
        ).unbind(dim=2)

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep_embedding: AdaLNTimestepEmbedding,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            timestep_embedding,
            scale_shift_indices,
            scale_shift_timestep,
        )
        gate_table = scale_shift_table[num_scale_shift_values:, :].to(
            device=gate_timestep.device, dtype=gate_timestep.dtype
        )
        gate_modulation = gate_timestep.reshape(batch_size, -1, 1, gate_table.shape[-1])
        gate_ada_values = (gate_table[None, None] + gate_modulation).unbind(dim=2)

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep_embedding: AdaLNTimestepEmbedding,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation.
        ``x_normed`` is the RMS-normalized post-self-attention output (produced by
        the PostSA composite); this method does not normalize again.
        """
        if cross_attention_adaln:
            shift_q, scale_q = self.get_ada_values(
                scale_shift_table, x_normed.shape[0], timestep_embedding, slice(6, 8)
            )
            (gate,) = self.get_ada_values(scale_shift_table, x_normed.shape[0], timestep_embedding, slice(8, 9))
            return apply_cross_attention_adaln(
                x_normed,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        return attn(x_normed, context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timestep_embedding, slice(0, 3)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            vx = vx + vx_msa_out * vgate_msa
            vx_normed = rms_norm(vx, eps=self.norm_eps)
            del vgate_msa, norm_vx, vx_msa_out
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timestep_embedding,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del vx_normed

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timestep_embedding, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )
            ax = ax + ax_msa_out * agate_msa
            ax_normed = rms_norm(ax, eps=self.norm_eps)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timestep_embedding,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ax_normed

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.timestep_embedding,
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = rms_norm(vx_pre_av, eps=self.norm_eps) * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.timestep_embedding,
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = rms_norm(ax_pre_av, eps=self.norm_eps) * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.timestep_embedding,
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = rms_norm(ax_pre_av, eps=self.norm_eps) * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.timestep_embedding,
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = rms_norm(vx_pre_av, eps=self.norm_eps) * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                del scale_ca_video_v2a, shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled
            del vx_pre_av, ax_pre_av

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timestep_embedding, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timestep_embedding, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply query/key AdaLN modulation then cross-attention.
    ``x_normed`` is already RMS-normalized by the PostSA composite; this only
    applies the affine (scale/shift) modulation, so normalization is not repeated.
    """
    batch_size = x_normed.shape[0]
    # K/V modulation. With the prompt-side AdaLN MLP disabled (use_prompt_adaln_single=False),
    # prompt_timestep is None and only the static per-block table applies, so K/V are
    # timestep-independent and cacheable across denoising/AR steps. Otherwise the timestep-
    # conditioned MLP output is added on top. 
    kv_modulation = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
    if prompt_timestep is not None:
        kv_modulation = kv_modulation + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    shift_kv, scale_kv = kv_modulation.unbind(dim=2)
    attn_input = x_normed * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
