"""Video VAE package."""


__all__ = [
    "AUTO_TILING",
    "CHANNELS_LAST_3D_WEIGHTS",
    "DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER",
    "MEMORY_EFFICIENT_DECODE",
    "VAE_DECODER_COMFY_KEYS_FILTER",
    "VAE_ENCODER_COMFY_KEYS_FILTER",
    "AutoTiling",
    "ConvVideoDecoder",
    "DiffusionVideoDecoder",
    "DimensionSizeConfig",
    "PipelineTiling",
    "TileSizeConfig",
    "TilingConfig",
    "VideoDecoder",
    "VideoDecoderConfigurator",
    "VideoEncoder",
    "VideoEncoderConfigurator",
    "diffvae_tiling_geometry_from_vae_config",
    "estimate_diffusion_decoder_weight_bytes",
    "get_video_chunks_number",
    "is_diffusion_video_vae",
    "video_decoder_sd_ops_for_checkpoint",
]

# Defer optional decoder dependencies until the requested decoder is used.
import importlib

_LAZY_EXPORTS = {'CHANNELS_LAST_3D_WEIGHTS': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.memory_efficient_decode', 'CHANNELS_LAST_3D_WEIGHTS'), 'MEMORY_EFFICIENT_DECODE': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.memory_efficient_decode', 'MEMORY_EFFICIENT_DECODE'), 'DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER'), 'VAE_DECODER_COMFY_KEYS_FILTER': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'VAE_DECODER_COMFY_KEYS_FILTER'), 'VAE_ENCODER_COMFY_KEYS_FILTER': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'VAE_ENCODER_COMFY_KEYS_FILTER'), 'VideoDecoderConfigurator': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'VideoDecoderConfigurator'), 'VideoEncoderConfigurator': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'VideoEncoderConfigurator'), 'diffvae_tiling_geometry_from_vae_config': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'diffvae_tiling_geometry_from_vae_config'), 'estimate_diffusion_decoder_weight_bytes': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'estimate_diffusion_decoder_weight_bytes'), 'is_diffusion_video_vae': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'is_diffusion_video_vae'), 'video_decoder_sd_ops_for_checkpoint': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.model_configurator', 'video_decoder_sd_ops_for_checkpoint'), 'ConvVideoDecoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae', 'ConvVideoDecoder'), 'DiffusionVideoDecoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae', 'DiffusionVideoDecoder'), 'VideoDecoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae', 'VideoDecoder'), 'VideoEncoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae', 'VideoEncoder'), 'get_video_chunks_number': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae', 'get_video_chunks_number'), 'AUTO_TILING': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'AUTO_TILING'), 'AutoTiling': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'AutoTiling'), 'DimensionSizeConfig': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'DimensionSizeConfig'), 'PipelineTiling': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'PipelineTiling'), 'TileSizeConfig': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'TileSizeConfig'), 'TilingConfig': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling', 'TilingConfig')}

def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module, symbol = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
