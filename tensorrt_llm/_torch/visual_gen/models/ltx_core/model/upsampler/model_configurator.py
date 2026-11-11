from tensorrt_llm._torch.visual_gen.models.ltx_core.model.model_protocol import ModelConfigurator
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.model import LatentUpsampler


class LatentUpsamplerConfigurator(ModelConfigurator[LatentUpsampler]):
    """
    Configurator for LatentUpsampler model.
    Used to create a LatentUpsampler model from a configuration dictionary.
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> LatentUpsampler:
        config = metadata.get("config", {})
        in_channels = config.get("in_channels", 128)
        mid_channels = config.get("mid_channels", 512)
        num_blocks_per_stage = config.get("num_blocks_per_stage", 4)
        dims = config.get("dims", 3)
        spatial_upsample = config.get("spatial_upsample", True)
        temporal_upsample = config.get("temporal_upsample", False)
        spatial_scale = config.get("spatial_scale", 2.0)
        rational_resampler = config.get("rational_resampler", False)
        return LatentUpsampler(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            dims=dims,
            spatial_upsample=spatial_upsample,
            temporal_upsample=temporal_upsample,
            spatial_scale=spatial_scale,
            rational_resampler=rational_resampler,
        )
