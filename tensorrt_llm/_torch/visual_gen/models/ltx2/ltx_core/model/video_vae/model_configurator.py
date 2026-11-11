from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import safetensors
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.checkpoint import CheckpointReader
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import KeyMatching, KeyReplacement, KeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.model_protocol import ModelConfigurator
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.dsl_kernels import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    NA_SOFTMAX_BOUND_BUFFER,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import ConvVideoDecoder, DiffusionVideoDecoder, VideoDecoder, VideoEncoder

_CONV_VAE_CLASS_NAME = "CausalVideoAutoencoder"


def _vae_class_name_from_metadata(metadata: dict) -> str:
    """Return the top-level ``vae._class_name`` from checkpoint metadata."""
    return metadata.get("config", {}).get("vae", {}).get("_class_name", _CONV_VAE_CLASS_NAME)


def is_diffusion_video_vae(checkpoint_path: str) -> bool:
    """Whether ``checkpoint_path`` metadata describes a diffusion video VAE.
    Decision is based on ``config.vae._class_name`` (same field
    ``VideoDecoderConfigurator`` uses), not on how the path was supplied.
    A standalone conv-VAE extract must still get conv SDOps / memory-efficient
    decode.
    """
    with CheckpointReader.open((Path(checkpoint_path),)) as reader:
        metadata = reader.read_checkpoint_metadata()
    return _vae_class_name_from_metadata(metadata) != _CONV_VAE_CLASS_NAME


def _prepare_video_encoder_kwargs(config: dict) -> dict:
    """Extract ``VideoEncoder`` init kwargs from a ``vae`` configuration dictionary.
    Two checkpoint layouts:
    - Flat ``CausalVideoAutoencoder`` (LTX-2): fields live on ``vae`` itself.
      Latent width is ``latent_channels``; top-level ``out_channels`` is decoder
      RGB and must not be used as the encoder's latent size.
    - Nested ``CausalDiffusionVAE``: fields live under ``vae.encoder`` with
      ``blocks`` / ``out_channels`` naming (``out_channels`` is the latent width).
    """
    if "encoder" in config:
        encoder_config = config["encoder"]
        out_channels = encoder_config.get("out_channels", config.get("latent_channels", 128))
        encoder_blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
    else:
        encoder_config = config
        out_channels = config.get("latent_channels", 128)
        encoder_blocks = config.get("encoder_blocks", [])

    return {
        "convolution_dimensions": encoder_config.get("dims", config.get("dims", 3)),
        "in_channels": encoder_config.get("in_channels", 3),
        "out_channels": out_channels,
        "encoder_blocks": encoder_blocks,
        "patch_size": encoder_config.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(
            encoder_config.get(
                "spatial_padding_mode",
                config.get("encoder_spatial_padding_mode", "zeros"),
            )
        ),
    }


class VideoEncoderConfigurator(ModelConfigurator[VideoEncoder]):
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> VideoEncoder:
        config = metadata.get("config", {}).get("vae", {})
        return VideoEncoder(**_prepare_video_encoder_kwargs(config))


def _build_conv_video_decoder(config: dict) -> ConvVideoDecoder:
    """Build a ``ConvVideoDecoder`` from a ``vae`` configuration dictionary."""
    return ConvVideoDecoder(
        convolution_dimensions=config.get("dims", 3),
        in_channels=config.get("latent_channels", 128),
        out_channels=config.get("out_channels", 3),
        decoder_blocks=config.get("decoder_blocks", []),
        patch_size=config.get("patch_size", 4),
        norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
        causal=config.get("causal_decoder", False),
        timestep_conditioning=config.get("timestep_conditioning", True),
        decoder_spatial_padding_mode=PaddingModeType(config.get("spatial_padding_mode", "reflect")),
        base_channels=config.get("decoder_base_channels", 128),
    )


def _build_diffusion_video_decoder(config: dict) -> DiffusionVideoDecoder:
    """Build a ``DiffusionVideoDecoder`` from a ``vae`` configuration dictionary.
    Real checkpoints (``_class_name: "CausalDiffusionVAE"``) nest decoder
    hyperparameters under ``vae.decoder``; fall back to the flat ``vae`` dict
    itself for configs that put them there directly. Architecture fields
    absent from config (e.g. ``t_emb_dim``, stage ladders) are left at the
    class defaults by omitting them from the constructor call.
    """
    decoder_config = config.get("decoder", config)
    # Optional architecture overrides - only pass when present so class defaults apply.
    architecture: dict = {}
    if "stage_channels" in decoder_config:
        architecture["stage_channels"] = tuple(decoder_config["stage_channels"])
    if "stage_depths" in decoder_config:
        architecture["stage_depths"] = tuple(decoder_config["stage_depths"])
    if "stage_kernels" in decoder_config:
        architecture["stage_kernels"] = tuple(tuple(kernel) for kernel in decoder_config["stage_kernels"])
    if "upsamples" in decoder_config:
        architecture["upsamples"] = tuple(
            (tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"]
        )
    if "stage5_kernel" in decoder_config:
        architecture["stage5_kernel"] = tuple(decoder_config["stage5_kernel"])
    if "stage5_channels" in decoder_config:
        architecture["stage5_channels"] = decoder_config["stage5_channels"]

    return DiffusionVideoDecoder(
        in_channels=decoder_config.get("in_channels", config.get("latent_channels", 128)),
        out_channels=decoder_config.get("out_channels", 3),
        patch_size=decoder_config.get("patch_size", 4),
        head_dim=decoder_config.get("head_dim", decoder_config.get("na_head_dim", 64)),
        t_emb_dim=decoder_config.get("t_emb_dim", 384),
        default_num_inference_steps=decoder_config.get("default_num_inference_steps", 2),
        timestep_scale_multiplier=decoder_config.get("timestep_scale_multiplier", 1.0),
        # Sibling of "decoder" at the top vae level (CausalDiffusionVAE.from_config
        # reads it there too), not nested under decoder_config.
        model_output_type=config.get("model_output_type", "v"),
        **architecture,
    )


def stage4_channels_from_decoder_config(decoder_config: dict) -> int:
    """Channels at stage-4 input (= output of deterministic stages 1-3).
    ``stage_channels[3]`` after the first three upsample hops.
    """
    if "stage_channels" not in decoder_config:
        raise ValueError("VAE decoder config missing stage_channels")
    channels = decoder_config["stage_channels"]
    if len(channels) < 4:
        raise ValueError(f"stage_channels must have length >= 4, got {len(channels)}")
    dim = int(channels[3])
    if dim < 1:
        raise ValueError(f"stage4_channels must be >= 1, got {dim}")
    return dim


def stage5_channels_from_decoder_config(decoder_config: dict) -> int:
    """Stage-5 feature width from VAE config: ``num_heads * head_dim``.
    Prefers explicit ``stage5_channels``, else ``stage_channels[-1]``. Both must be
    divisible by ``head_dim`` (required in config as ``head_dim`` / ``na_head_dim``).
    """
    if "head_dim" in decoder_config:
        head_dim = int(decoder_config["head_dim"])
    elif "na_head_dim" in decoder_config:
        head_dim = int(decoder_config["na_head_dim"])
    else:
        raise ValueError("VAE decoder config missing head_dim / na_head_dim")
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    if "stage5_channels" in decoder_config:
        dim = int(decoder_config["stage5_channels"])
    elif "stage_channels" in decoder_config:
        dim = int(decoder_config["stage_channels"][-1])
    else:
        raise ValueError("VAE decoder config missing stage5_channels / stage_channels")
    if dim % head_dim != 0:
        raise ValueError(f"stage5 dim {dim} is not divisible by head_dim={head_dim}")
    return (dim // head_dim) * head_dim


def diffvae_tiling_geometry_from_vae_config(config: dict) -> dict:
    """Resolve DiffVAE tiling geometry from a ``vae`` config dict (no module build).
    Returns kwargs for :func:`ltx_core.model.video_vae.diffusion_tiling.recommended_decode_tiling_config`
    except resolution / memory fields: ``tile_halos``, ``pixel_scale``, ``min_tile_size_s4``,
    ``patch_size``, ``stage5_channels``, ``stage4_channels``, ``upsample_strides``,
    ``natten_trailing_pad_latent_frames``, ``out_channels``.
    """
    # Local import: avoid a hard cycle at module import time.
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae import diffusion_tiling as dt  # noqa: PLC0415
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae import diffusion_video_decoder as dvd  # noqa: PLC0415

    decoder_config = config.get("decoder", config)
    stage_kernels = (
        tuple(tuple(k) for k in decoder_config["stage_kernels"])
        if "stage_kernels" in decoder_config
        else dvd._L_STAGE_KERNELS
    )
    stage_depths = (
        tuple(decoder_config["stage_depths"]) if "stage_depths" in decoder_config else dvd._DIFF_STAGE_DEPTHS_DEFAULT
    )
    upsamples = (
        tuple((tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"])
        if "upsamples" in decoder_config
        else dvd._L_UPSAMPLES
    )
    stage5_kernel = (
        tuple(decoder_config["stage5_kernel"]) if "stage5_kernel" in decoder_config else dvd._DIFF_STAGE5_KERNEL_DEFAULT
    )
    patch_size = int(decoder_config.get("patch_size", 4))
    up3_stride = upsamples[3][0]
    return {
        "tile_halos": dt.compute_tile_halos(
            stage_kernels[3],
            stage_depths[3],
            stage5_kernel,
            stage_depths[-1],
            up3_stride,
        ),
        "pixel_scale": dt.stage4_to_pixel_scale_factors(up3_stride, patch_size),
        "min_tile_size_s4": dt.compute_tile_min_size(stage_kernels[3], stage5_kernel, up3_stride),
        "patch_size": patch_size,
        "stage5_channels": stage5_channels_from_decoder_config(decoder_config),
        "stage4_channels": stage4_channels_from_decoder_config(decoder_config),
        "upsample_strides": tuple(stride for stride, _reduction in upsamples),
        "natten_trailing_pad_latent_frames": (stage_kernels[0][0] // 2) * 2,
        "out_channels": int(decoder_config.get("out_channels", 3)),
    }


def estimate_diffusion_decoder_weight_bytes(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Sum on-disk nbytes for ``vae.decoder`` / ``decoder`` tensors (header only) at ``dtype``."""
    element_size = dtype.itemsize
    path = Path(checkpoint_path)
    total = 0
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118  # safetensors SafeOpen is not a Mapping
            if not (key.startswith(("vae.decoder.", "decoder.")) or "per_channel_statistics" in key):
                continue
            shape = handle.get_slice(key).get_shape()
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n * element_size
    return total


class VideoDecoderConfigurator(ModelConfigurator[VideoDecoder]):
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> VideoDecoder:
        config = metadata.get("config", {}).get("vae", {})
        if _vae_class_name_from_metadata(metadata) == _CONV_VAE_CLASS_NAME:
            return _build_conv_video_decoder(config)
        return _build_diffusion_video_decoder(config)


# Accept both monolithic ``vae.*`` prefixes and Comfy-split bare ``encoder.*`` /
# ``decoder.*`` keys (standalone video VAE files under ``vae/``).
# LTX-2.0 checkpoints also store latent statistics the VAE does not own; only mean-of-means and std-of-means bind.
_UNUSED_LATENT_STATISTICS = (
    "per_channel_statistics.channel",
    "per_channel_statistics.mean-of-stds",
)
VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_DECODER_COMFY_KEYS_FILTER", excluded_prefixes=_UNUSED_LATENT_STATISTICS)
    .with_matching(prefix="vae.decoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.decoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    # Comfy-split bare keys: strip the leading ``decoder.`` after the vae. rule above.
    .with_replacement("decoder.", "")
)

VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_ENCODER_COMFY_KEYS_FILTER", excluded_prefixes=_UNUSED_LATENT_STATISTICS)
    .with_matching(prefix="vae.encoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.encoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    .with_replacement("encoder.", "")
)


_DIFFUSION_DECODER_PREFIXES = ("vae.decoder.", "decoder.")
# Linear weight leaf -> sibling gate folded into it (W <- g * W, b <- g * b).
_GATE_FOLD_TARGETS = (
    (".attn.proj.weight", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
)
_TYPE_EMB_KEY = "type_emb"
_CONV_IN_WEIGHT_KEY = "conv_in.weight"
_UPSAMPLE3_WEIGHT_KEY = "upsamples.3.proj.weight"
_UPSAMPLE3_BIAS_KEY = "upsamples.3.proj.bias"
_CONTEXT_PROJ_WEIGHT_SUFFIX = ".context_proj.weight"
_CONTEXT_PROJ_BIAS_SUFFIX = ".context_proj.bias"
_K_NORM_SUFFIX = ".attn.k_norm.weight"


def _strip_diffusion_decoder_prefix(key: str) -> str | None:
    """Strip ``vae.decoder.`` or bare ``decoder.``; keep channel statistics; None for other keys."""
    for prefix in _DIFFUSION_DECODER_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    if key.startswith("vae.per_channel_statistics."):
        return key.removeprefix("vae.")
    if key.startswith("per_channel_statistics."):
        return key
    return None


def _diffusion_prefix_stage() -> SDOps:
    """Normalize monolith ``vae.decoder.*`` and Comfy-split bare ``decoder.*`` keys."""
    return (
        SDOps("diffvae-prefix")
        .with_matching(prefix="vae.decoder.")
        .with_matching(prefix="decoder.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_matching(prefix="per_channel_statistics.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("decoder.", "")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
        .with_replacement("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        .with_replacement("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
    )


def _split_fused_qkv(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split fused ``...qkv.weight`` / ``...qkv.bias`` into the ``to_q`` / ``to_k`` / ``to_v`` linears."""
    value = inputs[key]
    if value.shape[0] % 3:
        raise ValueError(f"fused QKV {key!r} leading dimension is not divisible by 3")
    leaf = "weight" if key.endswith(".weight") else "bias"
    width = value.shape[0] // 3
    prefix = key[: -len(leaf)]
    return {
        f"{prefix}to_{part}.{leaf}": chunk.detach().clone()
        for part, chunk in zip(("q", "k", "v"), value.split(width, dim=0), strict=True)
    }


def _drop(_key: str, _inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {}


def _diffusion_qkv_and_coarse_stage() -> SDOps:
    """Split fused QKV tensors and drop the bundled preview/coarse head (``coarse_*``)."""
    return (
        SDOps("diffvae-qkv-and-coarse")
        .with_kv_operation(_drop, key_prefix="coarse_")
        .with_kv_operation(_split_fused_qkv, key_suffix=".qkv.weight")
        .with_kv_operation(_split_fused_qkv, key_suffix=".qkv.bias")
    )


def _fold_gate(weight_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold a static gate into its Linear weight and bias, consuming the gate; pass through without one."""
    weight = inputs[weight_key]
    suffix, gate_suffix = next((pair for pair in _GATE_FOLD_TARGETS if weight_key.endswith(pair[0])), (None, None))
    if suffix is None or gate_suffix is None:
        raise ValueError(f"unsupported gate-fold target: {weight_key!r}")
    gate_key = weight_key.removesuffix(suffix) + gate_suffix
    bias_key = weight_key.removesuffix(".weight") + ".bias"
    gate = inputs.get(gate_key)
    bias = inputs.get(bias_key)
    if gate is None:
        return dict(inputs)
    if gate.ndim != 1 or gate.shape[0] != weight.shape[0]:
        raise ValueError(f"gate {gate_key!r} disagrees with {weight_key!r}")
    folded = (weight.float() * gate.float().unsqueeze(1)).to(weight.dtype)
    result = {weight_key: folded}
    if bias is not None:
        if bias.shape != gate.shape:
            raise ValueError(f"gate {gate_key!r} disagrees with {bias_key!r}")
        result[bias_key] = (bias.float() * gate.float()).to(bias.dtype)
    return result


def _diffusion_gate_fold_stage() -> SDOps:
    """Fold standalone distilled checkpoints' gates so they load into the ungated ``DiffusionNABlock``."""
    stage = SDOps("diffvae-gate-fold")
    for suffix, gate_suffix in _GATE_FOLD_TARGETS:
        auxiliary_key_specs = (
            (KeyMatching(suffix=suffix), KeyReplacement(suffix, gate_suffix)),
            (KeyMatching(suffix=".weight"), KeyReplacement(".weight", ".bias")),
        )
        stage = stage.with_kv_operation(
            KeyValueOperation(_fold_gate, auxiliary_key_specs=auxiliary_key_specs), key_suffix=suffix
        )
    return stage


def _emit_zero_type_emb(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pass ``conv_in.weight`` through and synthesize a zero ``type_emb`` beside it.
    ``type_emb`` is registered unconditionally on the decoder but only exists in
    keyframe-trained checkpoints. Zeros keep pre-keyframe checkpoints loadable and are
    exactly the no-op the tag has when untrained (it is added to the keyframe latents,
    and zero means "no tag"). Attached to ``conv_in.weight`` because it always exists and
    its ``(out, in)`` shape carries the latent width.
    """
    value = inputs[key]
    return {key: value, _TYPE_EMB_KEY: torch.zeros(value.shape[1], dtype=value.dtype, device=value.device)}


def _diffusion_type_emb_stage() -> SDOps:
    return SDOps("diffvae-zero-type-emb").with_kv_operation(_emit_zero_type_emb, key_suffix=_CONV_IN_WEIGHT_KEY)


def _softmax_row_bound(value: torch.Tensor) -> torch.Tensor:
    from ltx_kernels.vae import softmax_row_bound  # noqa: PLC0415

    return softmax_row_bound(value)


def _emit_na_softmax_bound(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pass ``attn.k_norm.weight`` through and emit the DSL kernel's softmax bound beside it.
    The CuTe DSL neighborhood-attention kernel replaces the online row max with a fixed
    per-row offset built from ``sqrt(head_dim) * max|k_norm.weight|``. Deriving it here
    means it is computed once, during load, into the buffer
    ``DiffVAEMode.BLACKWELL_DSL`` registered -- rather than per forward, where reading it
    off the weight would be a device-to-host sync inside a compiled region.
    """
    value = inputs[key]
    bound_key = key.removesuffix(_K_NORM_SUFFIX) + f".attn.{NA_SOFTMAX_BOUND_BUFFER}"
    return {key: value, bound_key: _softmax_row_bound(value)}


def _abstract_na_softmax_bound(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    bound_key = key.removesuffix(_K_NORM_SUFFIX) + f".attn.{NA_SOFTMAX_BOUND_BUFFER}"
    return {key: inputs[key], bound_key: torch.empty((), dtype=torch.float32, device="meta")}


def _diffusion_softmax_bound_stage() -> SDOps:
    return SDOps("diffvae-na-softmax-bound").with_kv_operation(
        KeyValueOperation(_emit_na_softmax_bound, _abstract_na_softmax_bound), key_suffix=_K_NORM_SUFFIX
    )


def pack_fused_upsample_ctx_weights(
    w_upsample: torch.Tensor,
    b_upsample: torch.Tensor,
    w_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``context_proj(pixel_shuffle(upsample(.)))`` into packed Linear weights.
    Thin wrapper over :func:`ltx_kernels.vae.pack_fused_ctx_weights` so SDOps tests
    can import from ``model_configurator`` without a hard kernels dependency at
    module import time.
    """
    from ltx_kernels.vae import pack_fused_ctx_weights  # noqa: PLC0415

    return pack_fused_ctx_weights(w_upsample, b_upsample, w_ctx, b_ctx)


def _stage4_fusion(context_keys: tuple[str, ...]) -> KeyValueOperation:
    """Pack ``upsamples.3.proj`` into every gate-folded ``context_proj`` the fused DSL block reads."""

    def outputs(tensors: Mapping[str, torch.Tensor], fuse: bool) -> dict[str, torch.Tensor]:
        if _UPSAMPLE3_BIAS_KEY not in tensors:
            raise ValueError(f"stage-4 fusion needs {_UPSAMPLE3_BIAS_KEY}")
        result = dict(tensors)
        for weight_key in (key for key in context_keys if key.endswith(_CONTEXT_PROJ_WEIGHT_SUFFIX)):
            bias_key = weight_key.removesuffix(".weight") + ".bias"
            if bias_key not in tensors:
                raise ValueError(f"stage-4 fusion needs {bias_key!r}")
            prefix = weight_key.removesuffix(_CONTEXT_PROJ_WEIGHT_SUFFIX)
            if fuse:
                weight, bias = pack_fused_upsample_ctx_weights(
                    tensors[_UPSAMPLE3_WEIGHT_KEY], tensors[_UPSAMPLE3_BIAS_KEY], tensors[weight_key], tensors[bias_key]
                )
            else:
                weight = torch.empty(
                    tensors[_UPSAMPLE3_WEIGHT_KEY].shape, dtype=tensors[weight_key].dtype, device="meta"
                )
                bias = torch.empty(tensors[_UPSAMPLE3_BIAS_KEY].shape, dtype=tensors[bias_key].dtype, device="meta")
            result[f"{prefix}.{FUSED_CTX_WEIGHT_BUFFER}"] = weight
            result[f"{prefix}.{FUSED_CTX_BIAS_BUFFER}"] = bias
        return result

    return KeyValueOperation(
        lambda _key, tensors: outputs(tensors, fuse=True),
        lambda _key, tensors: outputs(tensors, fuse=False),
        auxiliary_key_specs=(_UPSAMPLE3_BIAS_KEY, *context_keys),
    )


def _diffusion_stage4_fusion_stage(context_keys: tuple[str, ...]) -> SDOps:
    """Name the fusion by its context key set, which the checkpoint header selects."""
    digest = sha256("\0".join(context_keys).encode()).hexdigest()[:12]
    return SDOps(f"diffvae-stage4-fusion:{digest}").with_kv_operation(
        _stage4_fusion(context_keys), key_suffix=_UPSAMPLE3_WEIGHT_KEY
    )


# Header-independent DiffVAE decoder stages: prefixes, QKV split, coarse drop and gate fold.
# ``video_decoder_sd_ops_for_checkpoint`` adds what a checkpoint's header calls for.
DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER = (
    _diffusion_prefix_stage(),
    _diffusion_qkv_and_coarse_stage(),
    _diffusion_gate_fold_stage(),
)


def video_decoder_sd_ops_for_checkpoint(
    checkpoint_path: str | Path,
    *,
    diffusion_vae: bool | None = None,
    na_dsl_kernel: bool = False,
) -> tuple[SDOps, ...]:
    """Pick decoder SDOps stages from checkpoint metadata (conv vs diffusion).
    For diffusion VAEs, the header selects the stages: a zero ``type_emb`` when the
    checkpoint has none, and every gate must have its fold target.
    ``na_dsl_kernel`` additionally derives the CuTe DSL kernel's softmax bound from each
    ``k_norm`` weight and fuses ``upsamples[3]`` into every block's ``context_proj``, so
    the fused block reads stage-4 directly instead of a full-resolution context. Must be
    paired with ``DiffVAEMode.BLACKWELL_DSL`` (registers the receiving buffers); apply
    alone leaves NaN / meta / empty buffers that fail at first forward. ``VideoDecoder``
    ties both to one flag -- custom builders must do the same.
    Pass ``diffusion_vae`` when the caller already ran ``is_diffusion_video_vae`` to
    skip a second metadata open.
    """
    if diffusion_vae is None:
        diffusion_vae = is_diffusion_video_vae(str(checkpoint_path))
    if not diffusion_vae:
        return (VAE_DECODER_COMFY_KEYS_FILTER,)
    with CheckpointReader.open((Path(checkpoint_path),)) as reader:
        _, metadata = reader.scan()
    keys = {short for key in metadata if (short := _strip_diffusion_decoder_prefix(key)) is not None}
    for suffix, gate_suffix in _GATE_FOLD_TARGETS:
        for gate in (key for key in keys if key.endswith(gate_suffix)):
            target = gate.removesuffix(gate_suffix) + suffix
            if target not in keys:
                raise ValueError(f"gate {gate!r} has no fold target {target!r}")
    stages = list(DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER)
    if _TYPE_EMB_KEY not in keys:
        stages.append(_diffusion_type_emb_stage())
    if na_dsl_kernel:
        if _UPSAMPLE3_WEIGHT_KEY not in keys or _UPSAMPLE3_BIAS_KEY not in keys:
            raise ValueError(
                f"DiffVAE checkpoint {checkpoint_path!r} has no {_UPSAMPLE3_WEIGHT_KEY} and bias, which "
                "na_dsl_kernel needs to fuse the stage-4 hop into context_proj"
            )
        context = tuple(
            sorted(key for key in keys if key.endswith((_CONTEXT_PROJ_WEIGHT_SUFFIX, _CONTEXT_PROJ_BIAS_SUFFIX)))
        )
        stages.extend((_diffusion_stage4_fusion_stage(context), _diffusion_softmax_bound_stage()))
    return tuple(stages)
