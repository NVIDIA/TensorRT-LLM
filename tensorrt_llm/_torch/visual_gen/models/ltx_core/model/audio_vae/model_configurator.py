from collections.abc import Mapping

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.attention import AttentionType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import AudioDecoder, AudioEncoder
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.causality_axis import CausalityAxis
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.vocoder import MelSTFT, Vocoder, VocoderWithBWE
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.common.normalization import NormType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.model_protocol import ModelConfigurator
from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import check_config_value


def _vocoder_from_config(
    cfg: dict,
    apply_final_activation: bool = True,
    output_sampling_rate: int | None = None,
) -> Vocoder:
    """Instantiate a Vocoder from a flat config dict.
    Args:
        cfg: Vocoder config dict (keys match Vocoder constructor args).
        apply_final_activation: Whether to apply tanh/clamp at the output.
        output_sampling_rate: Explicit override for the output sample rate.
            When None, reads from cfg["output_sampling_rate"] (default 24000).
    """
    return Vocoder(
        resblock_kernel_sizes=cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        upsample_rates=cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        upsample_initial_channel=cfg.get("upsample_initial_channel", 1024),
        resblock=cfg.get("resblock", "1"),
        output_sampling_rate=(
            output_sampling_rate if output_sampling_rate is not None else cfg.get("output_sampling_rate", 24000)
        ),
        activation=cfg.get("activation", "snake"),
        use_tanh_at_final=cfg.get("use_tanh_at_final", True),
        apply_final_activation=apply_final_activation,
        use_bias_at_final=cfg.get("use_bias_at_final", True),
    )


class VocoderConfigurator(ModelConfigurator[Vocoder]):
    """Configurator that auto-detects the checkpoint format.
    Returns a plain Vocoder for pre-ltx-2.3 checkpoints (flat config) or a
    VocoderWithBWE for ltx-2.3+ checkpoints (nested "vocoder" + "bwe" config).
    """

    @classmethod
    def from_metadata(cls, metadata: dict) -> Vocoder | VocoderWithBWE:
        cfg = metadata.get("config", {}).get("vocoder", {})

        if "bwe" not in cfg:
            check_config_value(cfg, "resblock", "1")
            check_config_value(cfg, "stereo", True)
            return _vocoder_from_config(cfg)

        vocoder_cfg = cfg.get("vocoder", {})
        bwe_cfg = cfg["bwe"]

        check_config_value(vocoder_cfg, "resblock", "AMP1")
        check_config_value(vocoder_cfg, "stereo", True)
        check_config_value(vocoder_cfg, "activation", "snakebeta")
        check_config_value(bwe_cfg, "resblock", "AMP1")
        check_config_value(bwe_cfg, "stereo", True)
        check_config_value(bwe_cfg, "activation", "snakebeta")

        vocoder = _vocoder_from_config(
            vocoder_cfg,
            output_sampling_rate=bwe_cfg["input_sampling_rate"],
        )
        bwe_generator = _vocoder_from_config(
            bwe_cfg,
            apply_final_activation=False,
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
        )
        mel_stft = MelSTFT(
            filter_length=bwe_cfg["n_fft"],
            hop_length=bwe_cfg["hop_length"],
            win_length=bwe_cfg["n_fft"],
            n_mel_channels=bwe_cfg["num_mels"],
        )
        return VocoderWithBWE(
            vocoder=vocoder,
            bwe_generator=bwe_generator,
            mel_stft=mel_stft,
            input_sampling_rate=bwe_cfg["input_sampling_rate"],
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
            hop_length=bwe_cfg["hop_length"],
        )


def _strip_vocoder_prefix(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the leading 'vocoder.' prefix exactly once.
    Uses removeprefix instead of str.replace so that BWE keys like
    'vocoder.vocoder.conv_pre' become 'vocoder.conv_pre' (not 'conv_pre').
    Works identically for legacy keys like 'vocoder.conv_pre' → 'conv_pre'.
    """
    return {key.removeprefix("vocoder."): inputs[key]}


VOCODER_COMFY_KEYS_FILTER = (
    SDOps("VOCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vocoder.")
    .with_kv_operation(_strip_vocoder_prefix, key_prefix="vocoder.")
)


class AudioDecoderConfigurator(ModelConfigurator[AudioDecoder]):
    @classmethod
    def from_metadata(cls, metadata: dict) -> AudioDecoder:
        audio_vae_cfg = metadata.get("config", {}).get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {})
        model_params = model_cfg.get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables_cfg.get("mel_bins")

        return AudioDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=ddconfig.get("attn_resolutions", {8, 16, 32}),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            dropout=ddconfig.get("dropout", 0.0),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )


class AudioEncoderConfigurator(ModelConfigurator[AudioEncoder]):
    @classmethod
    def from_metadata(cls, metadata: dict) -> AudioEncoder:
        audio_vae_cfg = metadata.get("config", {}).get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {})
        model_params = model_cfg.get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        n_fft = stft_cfg.get("filter_length", 1024)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables_cfg.get("mel_bins")

        return AudioEncoder(
            ch=ddconfig.get("ch", 128),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=ddconfig.get("attn_resolutions", {8, 16, 32}),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            double_z=ddconfig.get("double_z", True),
            dropout=ddconfig.get("dropout", 0.0),
            resamp_with_conv=ddconfig.get("resamp_with_conv", True),
            in_channels=ddconfig.get("in_channels", 2),
            attn_type=AttentionType(ddconfig.get("attn_type", "vanilla")),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )


AUDIO_VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("AUDIO_VAE_DECODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="audio_vae.decoder.")
    .with_matching(prefix="audio_vae.per_channel_statistics.")
    .with_replacement("audio_vae.decoder.", "")
    .with_replacement("audio_vae.per_channel_statistics.", "per_channel_statistics.")
)


AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="audio_vae.encoder.")
    .with_matching(prefix="audio_vae.per_channel_statistics.")
    .with_replacement("audio_vae.encoder.", "")
    .with_replacement("audio_vae.per_channel_statistics.", "per_channel_statistics.")
)
