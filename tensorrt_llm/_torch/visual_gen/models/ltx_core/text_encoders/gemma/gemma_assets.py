"""In-memory Gemma HuggingFace assets (load + HF builders + TE pack helpers).
Supports two on-disk layouts via :class:`GemmaAssets`:
* HF directory root (flat or nested ``_readout_proj``)
* Packed single-file TE ``.safetensors`` with embedded config / tokenizer / sidecars
HF objects (config, tokenizer, processor) are built from the in-memory buffers. Pack helpers
embed a ``gemma_root`` (+ projections) into a TE file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedTokenizerFast, ProcessorMixin
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.video_processing_utils import BaseVideoProcessor

from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import find_matching_file

logger = logging.getLogger(__name__)

GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"

_REQUIRED_ASSET_FILENAMES = (
    "tokenizer_config.json",
    "processor_config.json",
)
# Older / Comfy packs may store these as metadata strings instead of ``hf_asset__`` tensors.
_METADATA_FALLBACK_FILENAMES = (
    *_REQUIRED_ASSET_FILENAMES,
    "chat_template.jinja",
    "generation_config.json",
)
# Collect by pattern (not allowlist) so future processor sidecars are not silently dropped.
_ASSET_GLOBS = ("*.json", "*.jinja")
_ASSET_GLOB_EXCLUDES = frozenset({"config.json", "tokenizer.json"})


def is_safetensors_file(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix == ".safetensors"


@dataclass(frozen=True, slots=True)
class GemmaAssets:
    """In-memory Gemma HF assets loaded from a directory root or a single TE file."""

    source: str
    config_dict: Mapping[str, Any]
    tokenizer_json: bytes
    sidecars: Mapping[str, bytes]
    weight_paths: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> GemmaAssets:
        p = Path(path)
        if p.is_dir():
            return cls.from_root(p)
        if is_safetensors_file(p):
            return cls.from_single_file(p)
        raise FileNotFoundError(f"Gemma path {path!r} is neither a directory nor a .safetensors file.")

    @classmethod
    def from_root(cls, root: Path) -> GemmaAssets:
        """Load an HF Gemma directory (flat or nested ``_readout_proj``)."""
        root = Path(root)
        config_path = find_matching_file(str(root), "config.json")
        tokenizer_path = find_matching_file(str(root), "tokenizer.json")
        config_dict = json.loads(config_path.read_bytes())
        tokenizer_json = tokenizer_path.read_bytes()

        # Nested roots can duplicate basenames; keep the first hit (matches find_matching_file).
        sidecars: dict[str, bytes] = {}
        for pattern in _ASSET_GLOBS:
            for path in sorted(root.rglob(pattern)):
                if not path.is_file() or path.name in _ASSET_GLOB_EXCLUDES:
                    continue
                sidecars.setdefault(path.name, path.read_bytes())

        assets = cls(
            source=str(root),
            config_dict=config_dict,
            tokenizer_json=tokenizer_json,
            sidecars=sidecars,
            weight_paths=resolve_gemma_weight_paths(str(root)),
        )
        assets._require_sidecars(*_REQUIRED_ASSET_FILENAMES)
        return assets

    @classmethod
    def from_single_file(cls, path: Path) -> GemmaAssets:
        path = Path(path)
        with safetensors.safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            raw_config = meta.get(GEMMA_CONFIG_METADATA_KEY)
            if raw_config is None:
                raise ValueError(
                    f"Safetensors text-encoder {path} is missing metadata key "
                    f"{GEMMA_CONFIG_METADATA_KEY!r} (JSON-encoded HuggingFace config)."
                )
            config_dict = json.loads(raw_config)

            keys = set(f.keys())
            if TOKENIZER_JSON_TENSOR_KEY not in keys:
                raise ValueError(f"Safetensors text-encoder {path} is missing tensor {TOKENIZER_JSON_TENSOR_KEY!r}.")
            tokenizer_json = _tensor_to_bytes(f.get_tensor(TOKENIZER_JSON_TENSOR_KEY))

            sidecars: dict[str, bytes] = {}
            for key in keys:
                if not key.startswith(HF_ASSET_TENSOR_PREFIX):
                    continue
                name = key.removeprefix(HF_ASSET_TENSOR_PREFIX)
                sidecars[name] = _tensor_to_bytes(f.get_tensor(key))

            for name in _METADATA_FALLBACK_FILENAMES:
                if name in sidecars or name not in meta:
                    continue
                sidecars[name] = meta[name].encode()

        assets = cls(
            source=str(path),
            config_dict=config_dict,
            tokenizer_json=tokenizer_json,
            sidecars=sidecars,
            weight_paths=(str(path.resolve()),),
        )
        assets._require_sidecars(*_REQUIRED_ASSET_FILENAMES)
        return assets

    def sidecar_bytes(self, name: str) -> bytes:
        try:
            return self.sidecars[name]
        except KeyError as exc:
            raise KeyError(f"Gemma assets from {self.source!r} are missing sidecar {name!r}.") from exc

    def sidecar_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.sidecar_bytes(name))

    def _require_sidecars(self, *names: str) -> None:
        missing = [name for name in names if name not in self.sidecars]
        if missing:
            raise ValueError(
                f"Gemma assets from {self.source!r} are missing required sidecar(s): {', '.join(missing)}. "
                f"Embed as {HF_ASSET_TENSOR_PREFIX}<name> (or metadata for small JSON)."
            )


TOKENIZER_MAX_LENGTH = 1024

# ``added_tokens_decoder`` is for from_pretrained reconstruction; as a kwarg it TypeErrors.
_TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)

_SUBPROCESSOR_BASES: dict[str, type] = {
    "image_processor": BaseImageProcessor,
    "feature_extractor": FeatureExtractionMixin,
    "video_processor": BaseVideoProcessor,
}

# gemma3 stores image-processor config in preprocessor_config.json, not nested in processor_config.
_SUBPROCESSOR_SIDECAR_FALLBACKS: dict[str, tuple[str, ...]] = {
    "image_processor": ("preprocessor_config.json",),
    "feature_extractor": ("preprocessor_config.json",),
    "video_processor": ("video_preprocessor_config.json", "preprocessor_config.json"),
}


def _resolve_transformers_class(name: str, base: type) -> type:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Invalid transformers class name {name!r}.")
    cls = getattr(transformers, name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, base):
        raise ValueError(f"transformers.{name} is not a subclass of {base.__name__}.")
    return cls


def _resolve_subprocessor_class(attribute: str, type_name: str) -> type:
    """Resolve a Processor sub-component class from its declared ``*_type`` name.
    For ``image_processor``, prefer the PIL backend (``*Pil``) when available. That matches
    the legacy disk loader's ``AutoImageProcessor.from_pretrained(..., use_fast=False)``
    (now ``backend="pil"``). The torchvision ``Gemma4ImageProcessor`` path produces
    slightly different pixel values and shifts I2V enhance goldens.
    """
    base = _SUBPROCESSOR_BASES[attribute]
    if attribute == "image_processor" and not type_name.endswith("Pil"):
        pil_cls = getattr(transformers, f"{type_name}Pil", None)
        if isinstance(pil_cls, type) and issubclass(pil_cls, base):
            return pil_cls
    return _resolve_transformers_class(type_name, base)


def _chat_template_from_assets(assets: GemmaAssets) -> str | None:
    if (tpl := assets.sidecars.get("chat_template.jinja")) is not None:
        return tpl.decode()
    raw = assets.sidecars.get("chat_template.json")
    if raw is None:
        return None
    loaded = json.loads(raw)
    if isinstance(loaded, str):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("chat_template"), str):
        return loaded["chat_template"]
    return None


def build_gemma_hf_config(assets: GemmaAssets) -> PretrainedConfig:
    model_type = assets.config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Gemma config from {assets.source!r} is missing model_type.")
    try:
        config_cls = CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Gemma model_type={model_type!r} in assets from {assets.source!r}.") from exc
    return config_cls.from_dict(dict(assets.config_dict))


def build_gemma_hf_tokenizer(assets: GemmaAssets, max_length: int = TOKENIZER_MAX_LENGTH) -> PreTrainedTokenizerFast:
    tokenizer_cfg = assets.sidecar_json("tokenizer_config.json")
    kwargs = {k: v for k, v in tokenizer_cfg.items() if k not in _TOKENIZER_CONFIG_SKIP}
    if (tpl := _chat_template_from_assets(assets)) is not None:
        kwargs.setdefault("chat_template", tpl)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(assets.tokenizer_json),
        model_max_length=max_length,
        **kwargs,
    )


def build_gemma_processor(assets: GemmaAssets, hf_tokenizer: PreTrainedTokenizerBase) -> ProcessorMixin:
    processor_cfg = assets.sidecar_json("processor_config.json")
    processor_class_name = processor_cfg.get("processor_class")
    if not processor_class_name:
        raise ValueError(f"processor_config.json from {assets.source!r} is missing processor_class.")
    processor_class = _resolve_transformers_class(processor_class_name, ProcessorMixin)

    components: dict[str, Any] = {"tokenizer": hf_tokenizer}
    for name in processor_class.get_attributes():
        if name in components:
            continue
        if name not in _SUBPROCESSOR_BASES:
            raise ValueError(
                f"Unsupported processor attribute {name!r} in {processor_class_name} (assets from {assets.source!r})."
            )
        if name in processor_cfg and isinstance(processor_cfg[name], dict):
            sub_cfg = dict(processor_cfg[name])
        else:
            sub_cfg = None
            for fname in _SUBPROCESSOR_SIDECAR_FALLBACKS.get(name, ()):
                if fname in assets.sidecars:
                    sub_cfg = assets.sidecar_json(fname)
                    break
            if sub_cfg is None:
                raise ValueError(
                    f"No nested {name!r} config in processor_config.json and no fallback sidecar "
                    f"for assets from {assets.source!r}."
                )
        type_key = f"{name}_type"
        type_name = sub_cfg.get(type_key)
        if not type_name:
            raise ValueError(f"Missing {type_key!r} for processor attribute {name!r} (assets from {assets.source!r}).")
        concrete = _resolve_subprocessor_class(name, type_name)
        components[name] = concrete.from_dict(sub_cfg)

    extra = {
        k: v for k, v in processor_cfg.items() if k not in processor_class.get_attributes() and k != "processor_class"
    }
    return processor_class(**components, **extra)


def resolve_gemma_weight_paths(gemma_model_path: str) -> tuple[str, ...]:
    path = Path(gemma_model_path)
    if is_safetensors_file(path):
        return (str(path.resolve()),)
    model_folder = find_matching_file(gemma_model_path, "model*.safetensors").parent
    return tuple(str(p) for p in sorted(model_folder.rglob("*.safetensors")))


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().numpy()
    if arr.dtype == np.uint8:
        return arr.tobytes()
    # Comfy may store tokenizer_json as int8 (or other integer dtypes).
    return arr.astype(np.uint8).tobytes()


def _bytes_to_uint8_tensor(data: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


def _flatten_gemma4_unified_keys_for_comfy(key: str) -> str:
    """Map HF ``gemma4_unified`` keys to ComfyUI's LTXAV TE module names.
    * LM: ``model.language_model.*`` → ``model.*`` so ``detect_te_model`` sees
      ``model.layers.0.post_feedforward_layernorm.weight``.
    * Vision / modality towers: HF ``model.vision_embedder.*`` /
      ``model.embed_vision.*`` / ``model.embed_audio.*`` → Comfy
      ``vision_model.*`` / ``multi_modal_projector.*`` / ``audio_projector.*``.
      Without this remap, Comfy ``load_sd(..., strict=False)`` leaves those 11
      tensors at random init (see ``docs/ltx-2.5-gemma4-missing-tensors.md``).
    """
    if key.startswith("model.language_model."):
        return "model." + key.removeprefix("model.language_model.")
    if key.startswith("model.vision_embedder."):
        return "vision_model." + key.removeprefix("model.vision_embedder.")
    if key.startswith("model.embed_vision."):
        return "multi_modal_projector." + key.removeprefix("model.embed_vision.")
    if key.startswith("model.embed_audio."):
        return "audio_projector." + key.removeprefix("model.embed_audio.")
    return key


def build_text_encoder_tensors_from_gemma_root(
    gemma_root: str,
    projection_tensors: dict[str, torch.Tensor],
    *,
    gemma_version: str | None = None,
    comfy_flat_lm_keys: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Assemble TE safetensors payloads (weights + assets) from an HF ``gemma_root``.
    When ``comfy_flat_lm_keys`` is True (default), HF ``gemma4_unified`` keys are rewritten to
    ComfyUI's LTXAV TE layout (flat LM under ``model.*``, plus ``vision_model.*`` /
    ``multi_modal_projector.*`` / ``audio_projector.*``). The pipeline loader accepts both
    the Comfy layout and legacy HF tower names via :func:`get_gemma_ops`.
    ``gemma_version`` is an *assertion*, not a stamp: pass the version the diffusion
    checkpoint expects and packing fails if this root disagrees. Default ``None`` packs
    whatever the root declares.
    Returns ``(tensors, metadata_str_values)`` ready for ``safetensors.torch.save_file``.
    """
    assets = GemmaAssets.from_root(Path(gemma_root))
    config = dict(assets.config_dict)
    declared_version = config.get("gemma_version")
    if gemma_version is not None:
        # Assert against the root; do not overwrite a declared version (would defeat load-time checks).
        if declared_version is not None and declared_version != gemma_version:
            raise ValueError(
                f"Gemma version mismatch: caller expects gemma_version={gemma_version!r}, but the "
                f"Gemma config at {gemma_root!r} declares gemma_version={declared_version!r}."
            )
        if declared_version is None:
            logger.warning(
                "Gemma config at %s declares no gemma_version; stamping %r from the caller.",
                gemma_root,
                gemma_version,
            )
            config["gemma_version"] = gemma_version

    tensors: dict[str, torch.Tensor] = {}
    for wp in assets.weight_paths:
        with safetensors.safe_open(wp, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open is not a Mapping; keys() is the only iterator
                out_key = _flatten_gemma4_unified_keys_for_comfy(key) if comfy_flat_lm_keys else key
                tensors[out_key] = f.get_tensor(key)

    tensors.update(projection_tensors)
    tensors[TOKENIZER_JSON_TENSOR_KEY] = _bytes_to_uint8_tensor(assets.tokenizer_json)
    for name, data in assets.sidecars.items():
        tensors[f"{HF_ASSET_TENSOR_PREFIX}{name}"] = _bytes_to_uint8_tensor(data)

    metadata = {
        "format": "pt",
        GEMMA_CONFIG_METADATA_KEY: json.dumps(config),
    }
    return tensors, metadata


def save_text_encoder_safetensors(
    output_path: str | Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)
