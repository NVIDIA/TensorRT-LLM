import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.model_protocol import ModelConfigurator
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.model import LTXModel, LTXModelType
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.rope import LTXRopeType
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.text_projection import create_caption_projection
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.utils import check_config_value


class LTXModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX model.
    Used to create an LTX model from a configuration dictionary.
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> LTXModel:
        config = metadata.get("config", {})
        # Build caption projections for 19B models (projection handled in transformer).
        caption_projection, audio_caption_projection = _build_caption_projections(config, is_av=True)

        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_audio_video_cross_attention", True)
        check_config_value(config, "share_ff", False)
        check_config_value(config, "av_cross_ada_norm", True)
        check_config_value(config, "use_middle_indices_grid", True)
        check_config_value(config, "num_attention_heads", config.get("audio_num_attention_heads", float("nan")))

        return LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            caption_projection=caption_projection,
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
            # Default True preserves the prompt-side AdaLN MLP for existing checkpoints. KV-cacheable
            # checkpoints set use_prompt_adaln_single=false, dropping the timestep-dependence of the
            # cross-attention K/V so they can be computed once per prompt and reused across steps.
            use_prompt_adaln_single=config.get("use_prompt_adaln_single", True),
            # Default True keeps backwards compatibility: pre-2.5 checkpoints lack these
            # keys and retain FFN biases. LTX 2.5 (gemma4) sets ff_bias=false.
            ff_bias=config.get("ff_bias", True),
            audio_ff_bias=config.get("audio_ff_bias", True),
            # Only generated-keyframe checkpoints set this; defaults off so older ones are untouched.
            use_keyframes_abs_pos_embedding=config.get("use_keyframes_abs_pos_embedding", False),
        )


class LTXVideoOnlyModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX video only model.
    Used to create an LTX video only model from a configuration dictionary.
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> LTXModel:
        config = metadata.get("config", {})
        # Build caption projection for 19B model (projection handled in transformer).
        caption_projection, _ = _build_caption_projections(config, is_av=False)

        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            caption_projection=caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
            # Default True preserves the prompt-side AdaLN MLP for existing checkpoints. KV-cacheable
            # checkpoints set use_prompt_adaln_single=false, making the cross-attention K/V
            # timestep-independent (computed once per prompt, reused across steps).
            use_prompt_adaln_single=config.get("use_prompt_adaln_single", True),
            # Default True keeps backwards compatibility with pre-2.5 checkpoints.
            ff_bias=config.get("ff_bias", True),
            # Only generated-keyframe checkpoints set this; defaults off so older ones are untouched.
            use_keyframes_abs_pos_embedding=config.get("use_keyframes_abs_pos_embedding", False),
        )


class LTXAudioOnlyModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX audio only model.
    Builds an audio-only LTX model (``model_type=AudioOnly``) so the video
    transformer weights are never instantiated or loaded. Useful for
    text-to-audio inference where the video branch is unused.
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> LTXModel:
        config = metadata.get("config", {})
        # Build audio caption projection for 19B models (projection handled in transformer).
        _, audio_caption_projection = _build_caption_projections(config, is_av=True)

        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.AudioOnly,
            num_layers=config.get("num_layers", 48),
            norm_eps=config.get("norm_eps", 1e-06),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )


def _build_caption_projections(
    config: dict,
    is_av: bool,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """Build caption projections for the transformer when projection is NOT in the text encoder.
    19B models: projection is in the transformer (caption_proj_before_connector=False).
    22B models: projection is in the text encoder, so no projections are created here.
    Args:
        config: Full model config dict (must contain "transformer" key).
        is_av: Whether this is an audio-video model. When False, audio projection is skipped.
    Returns:
        Tuple of (video_caption_projection, audio_caption_projection), both None for 22B models.
    """
    transformer_config = config.get("transformer", {})
    if transformer_config.get("caption_proj_before_connector", False):
        return None, None

    with torch.device("meta"):
        caption_projection = create_caption_projection(transformer_config)
        audio_caption_projection = create_caption_projection(transformer_config, audio=True) if is_av else None
    return caption_projection, audio_caption_projection


LTXV_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTXV_MODEL_COMFY_PREFIX_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
)


def transformer_checkpoint_selection_ops(config: dict) -> SDOps | None:
    """Exclude connector weights owned by the embeddings processor when configured."""
    if not config.get("transformer", {}).get("use_embeddings_connector", False):
        return None
    return SDOps(
        "transformer-connectors-in-embeddings-processor",
        excluded_prefixes=("video_embeddings_connector.", "audio_embeddings_connector."),
    )


LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTXV_AUDIO_ONLY_MODEL_COMFY_MAP")
    .with_matching(prefix="model.diffusion_model.", contains="audio_attn1")
    .with_matching(prefix="model.diffusion_model.", contains="audio_attn2")
    .with_matching(prefix="model.diffusion_model.", contains="audio_ff")
    .with_matching(prefix="model.diffusion_model.", contains="audio_patchify")
    .with_matching(prefix="model.diffusion_model.", contains="audio_proj_out")
    .with_matching(prefix="model.diffusion_model.", contains="audio_adaln_single")
    .with_matching(prefix="model.diffusion_model.", contains="audio_prompt")
    .with_matching(prefix="model.diffusion_model.", contains="audio_scale_shift_table")
    .with_replacement("model.diffusion_model.", "")
)
