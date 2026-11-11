from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import upload_from_host


class VideoPixelShape(NamedTuple):
    """
    Shape of the tensor representing the video pixel array. Assumes BGR channel format.
    """

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    Field order matches the (frame/time, height, width) axis layout used by
    latent tensors and meshgrid coordinates elsewhere in the codebase.
    """

    time: int
    height: int
    width: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, height=32, width=32)

    @classmethod
    def from_blocks(cls, blocks: list, patch_size: int) -> "SpatioTemporalScaleFactors":
        """Derive the scale factors from a VAE encoder/decoder block list.
        Each ``compress_*`` block halves (encoder) or doubles (decoder) its target
        axes by a stride of 2, independent of any channel ``multiplier``. The initial
        patchify contributes an extra ``patch_size`` of spatial compression. Deriving
        the factors from the blocks keeps a single source of truth that stays correct
        across VAE variants (e.g. the 32x32x8 default and the 16x16x4 variant) instead
        of relying on a hardcoded constant.
        """
        spatial_steps = 0
        temporal_steps = 0
        for block_name, _ in blocks:
            if block_name.startswith(("compress_space", "compress_all")):
                spatial_steps += 1
            if block_name.startswith(("compress_time", "compress_all")):
                temporal_steps += 1
        spatial = patch_size * (2**spatial_steps)
        return cls(time=2**temporal_steps, height=spatial, width=spatial)

    @classmethod
    def from_model_config(cls, model_config: dict) -> "SpatioTemporalScaleFactors":
        """Derive the video scale factors from a checkpoint's model config dict.
        Reads the embedded VAE block list (see ``from_blocks``). Falls back to the
        default when the config carries no VAE block list -- either no ``vae`` section
        or a ``vae`` section without encoder/decoder blocks (e.g. audio-only
        checkpoints), where video tools are never built.
        """
        vae_config = model_config.get("vae", {})
        blocks = vae_config.get("encoder_blocks") or vae_config.get("decoder_blocks")
        if not blocks:
            return cls.default()
        return cls.from_blocks(blocks, vae_config.get("patch_size", 4))


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """
    Shape of the tensor representing video in VAE latent space.
    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def token_count(self) -> int:
        """Number of tokens after patchification with the default patch size of 1."""
        return self.frames * self.height * self.width

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors.time + 1
        height = shape.height // scale_factors.height
        width = shape.width // scale_factors.width

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def token_count(self) -> int:
        """Number of tokens after patchification."""
        return self.frames

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class HostDeviceTensorPair:
    """A tensor together with its host mirror."""

    host_tensor: torch.Tensor
    device_tensor: torch.Tensor

    @classmethod
    def on_device(cls, tensor: torch.Tensor, device: torch.device) -> "HostDeviceTensorPair":
        """``tensor``'s host mirror and a copy on ``device``.
        Reads the device back only when ``tensor`` lives there; a host tensor is uploaded without
        waiting for the device.
        """
        host = tensor.cpu()
        device_tensor = tensor if tensor.device == device else upload_from_host(host, device)
        return cls(host_tensor=host, device_tensor=device_tensor)


@dataclass(frozen=True)
class Audio:
    """
    Container for decoded audio samples and metadata.
    Attributes:
        waveform: Audio waveform tensor.
        sampling_rate: Sampling rate (Hz) of the waveform.
    """

    waveform: torch.Tensor
    sampling_rate: int

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))


@dataclass(frozen=True)
class GeneratedKeyframeLayout:
    """Where a state's generated-keyframe slot tokens live, and what they represent.
    Recorded by :class:`~ltx_core.conditioning.types.keyframe_slots.VideoGeneratedKeyframeSlots`
    when it appends the slots, so they can later be located and extracted *exactly* rather
    than by assuming they are the trailing tokens. Conditioning items are applied in list
    order and each appends to the end, so a state built with slots plus any other appending
    conditioning item has no fixed trailing layout.
    Attributes:
        pixel_frame_indices: Target pixel-frame index of each slot, in token order.
        tokens_per_keyframe: Number of tokens one slot occupies (one latent frame's worth).
        first_token: Index of the first slot token in the token sequence.
    """

    pixel_frame_indices: tuple[int, ...]
    tokens_per_keyframe: int
    first_token: int

    @property
    def num_keyframes(self) -> int:
        return len(self.pixel_frame_indices)

    @property
    def num_tokens(self) -> int:
        return self.num_keyframes * self.tokens_per_keyframe

    @property
    def token_slice(self) -> slice:
        return slice(self.first_token, self.first_token + self.num_tokens)


@dataclass(frozen=True)
class DenoiseMask:
    """A per-token denoising mask and the table of its distinct values.
    Conditioning builds ``tensor`` on the host, editing it in place or replacing it, with
    ``values`` unset. ``compact`` tabulates the distinct values there, where ``torch.unique`` needs
    no device synchronization, and moves both to the device; the denoising loop only sees the
    compacted form, and nothing may edit ``tensor`` after it. ``values`` must hold exactly the
    values present: a missing one silently modulates its tokens with a neighbouring value's
    conditioning, and an extra one changes the shape the modulation is projected at.
    ``zero_index`` is the position of ``0.0`` in ``values``, recorded on the host by ``compact``
    (``None`` when no token is fully conditioned), so padding can reuse that entry.
    """

    tensor: torch.Tensor
    values: torch.Tensor | None = None
    zero_index: int | None = None

    def compact(self, device: torch.device) -> "DenoiseMask":
        """Tabulate the distinct mask values on the host and move mask and table to ``device``."""
        if self.tensor.device.type != "cpu":
            raise ValueError(f"denoise mask must be built on the host, got {self.tensor.device}")
        values = torch.unique(self.tensor, sorted=True)
        zeros = (values == 0).nonzero()
        zero_index = int(zeros[0, 0]) if zeros.numel() else None
        return DenoiseMask(upload_from_host(self.tensor, device), upload_from_host(values, device), zero_index)

    def with_same_values(self, tensor: torch.Tensor) -> "DenoiseMask":
        """Replace the tensor with a reordering or batch repeat of the same token values."""
        return DenoiseMask(tensor, self.values, self.zero_index)

    def clone(self) -> "DenoiseMask":
        return DenoiseMask(
            self.tensor.clone(), self.values.clone() if self.values is not None else None, self.zero_index
        )


@dataclass(frozen=True)
class LatentState:
    """
    State of latents during the diffusion denoising process.
    Attributes:
        latent: The current noisy latent tensor being denoised.
        denoise_mask: Mask encoding the denoising strength for each token (1 = full denoising, 0 = no denoising).
        positions: Positional indices for each latent element, used for positional embeddings.
        clean_latent: Initial state of the latent before denoising, may include conditioning latents.
        attention_mask: Optional 2D self-attention mask of shape (B, T, T). Values in [0, 1] where 1 = full attention,
            0 = no attention. None means full attention everywhere. Built incrementally by conditioning items.
        keyframes_mask: Optional per-token marker of shape (B, T, 1) -- same layout as
            ``denoise_mask`` -- non-zero on tokens whose latent encodes a *single standalone pixel
            frame* rather than the usual multi-frame span. That set is the target's first latent
            frame (the video encoder is causal, so its first temporal latent frame covers 1 pixel
            frame while the rest cover 8) plus any generated keyframe slots. Selects the tokens
            that receive the model's learned keyframe absolute-position embedding; ignored
            entirely by models built without ``use_keyframes_abs_pos_embedding``.
        generated_keyframe_layout: Set when generated keyframe slots were appended; locates them.
        generated_keyframes: Populated by ``clear_conditioning`` when a layout is present: the
            denoised slot content as an unpatchified ``(B, C, K, H, W)`` latent, one latent frame
            per keyframe. Each frame must be decoded as a standalone one-frame clip, never as a
            K-frame video -- a causal decode would blend slots that were never adjacent.
        frozen: When True, this stream is held fixed: token denoising is disabled (``denoise_mask``
            should be all zeros; pipeline builders enforce that) and the scalar noise level used for
            prompt / cross-modality AdaLN gates is forced to 0 when the state is converted for the
            transformer.
    """

    latent: torch.Tensor
    denoise_mask: DenoiseMask
    positions: torch.Tensor
    clean_latent: torch.Tensor
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None
    generated_keyframes: torch.Tensor | None = None
    frozen: bool = False

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            keyframes_mask=self.keyframes_mask.clone() if self.keyframes_mask is not None else None,
            generated_keyframe_layout=self.generated_keyframe_layout,
            generated_keyframes=(self.generated_keyframes.clone() if self.generated_keyframes is not None else None),
            frozen=self.frozen,
        )
