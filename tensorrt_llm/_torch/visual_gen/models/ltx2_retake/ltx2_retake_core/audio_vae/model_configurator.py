# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from ....ltx2.ltx2_core.audio_vae.causality_axis import CausalityAxis
from ....ltx2.ltx2_core.normalization import NormType
from .audio_vae import AudioEncoder


class AudioEncoderConfigurator:
    """Create an AudioEncoder from the LTX-2 native config dict."""

    @classmethod
    def from_config(cls, config: dict) -> AudioEncoder:
        audio_vae_cfg = config.get("audio_vae", {})
        model_params = audio_vae_cfg.get("model", {}).get("params", {})
        ddconfig = model_params.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})
        variables_cfg = audio_vae_cfg.get("variables", {})

        sample_rate = model_params.get("sampling_rate", 16000)
        mel_hop_length = stft_cfg.get("hop_length", 160)
        n_fft = stft_cfg.get("filter_length", 1024)
        is_causal = stft_cfg.get("causal", True)
        mel_bins = (
            ddconfig.get("mel_bins")
            or mel_cfg.get("n_mel_channels")
            or variables_cfg.get("mel_bins")
        )
        if mel_bins is None:
            raise ValueError("LTX-2 audio VAE config does not define the mel-bin count.")

        # The native encoder implements only vanilla attention.
        attn_type = ddconfig.get("attn_type", "vanilla")
        if attn_type != "vanilla":
            raise ValueError(
                f"LTX-2 native audio encoder supports attn_type='vanilla'; got {attn_type!r}."
            )

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
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", True),
            norm_type=NormType(ddconfig.get("norm_type", "pixel")),
            causality_axis=CausalityAxis(ddconfig.get("causality_axis", "height")),
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )
