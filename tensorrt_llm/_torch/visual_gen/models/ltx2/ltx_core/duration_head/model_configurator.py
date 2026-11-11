from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.duration_head.duration_head import DurationHead
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.model_protocol import ModelConfigurator

# DurationHead's tensors live under a "duration_head." key prefix, already exported in
# PyTorch's nn.MultiheadAttention layout. Layout-agnostic on purpose: this strips the
# prefix regardless of whether it's read from its own checkpoint file or from a shared
# one alongside other model weights.
DURATION_HEAD_KEY_OPS = (
    SDOps("DURATION_HEAD_KEY_OPS").with_matching(prefix="duration_head.").with_replacement("duration_head.", "")
)


class DurationHeadConfigurator(ModelConfigurator[DurationHead]):
    """Configurator for DurationHead.
    Video/audio projection dims mirror the main transformer's connector dims
    (``cross_attention_dim`` / ``audio_cross_attention_dim``). The head's own
    hyperparameters have no finalized config schema yet, so they are read from a
    placeholder ``duration_head`` sub-dict with JAX-matching defaults.
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> DurationHead:
        config = metadata.get("config", {})
        transformer_config = config.get("transformer", {})
        duration_head_config = config.get("duration_head", {})

        return DurationHead(
            video_cross_attention_dim=transformer_config.get("cross_attention_dim", 4096),
            audio_cross_attention_dim=transformer_config.get("audio_cross_attention_dim", 2048),
            pooler_hidden_dim=duration_head_config.get("pooler_hidden_dim", 256),
            num_queries=duration_head_config.get("num_queries", 1),
            num_pooler_heads=duration_head_config.get("num_pooler_heads", 4),
            mlp_hidden=duration_head_config.get("mlp_hidden", 256),
        )
