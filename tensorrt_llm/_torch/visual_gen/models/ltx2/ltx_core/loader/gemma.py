"""Build Gemma text encoders with the unit loader and established model operations."""

from __future__ import annotations

from pathlib import Path

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.simple_builder import SimpleBuilder
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.text_encoders.gemma.encoders.encoder_configurator import GemmaTextEncoderConfigurator, get_gemma_ops
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.text_encoders.gemma.gemma_assets import GemmaAssets, build_gemma_hf_config


def _builder(root: str | Path, allowed_types: frozenset[str]) -> SimpleBuilder[LTXGemmaTextEncoder]:
    path = Path(root).resolve()
    assets = GemmaAssets.load(path)
    model_type = build_gemma_hf_config(assets).model_type
    if model_type not in allowed_types:
        expected = ", ".join(sorted(allowed_types))
        raise ValueError(f"expected {expected} assets at {path}, got {model_type!r}")
    sd_ops, module_ops = get_gemma_ops(str(path), assets=assets)
    configurator = GemmaTextEncoderConfigurator.with_gemma_model_path(str(path))
    return SimpleBuilder(configurator, tuple(Path(weight) for weight in assets.weight_paths), (sd_ops,), module_ops)


def gemma3_builder(root: str | Path) -> SimpleBuilder[LTXGemmaTextEncoder]:
    """Build Gemma 3 from an HF root or packed text-encoder file."""
    return _builder(root, frozenset({"gemma3"}))


def gemma4_builder(root: str | Path) -> SimpleBuilder[LTXGemmaTextEncoder]:
    """Build dense or unified Gemma 4 from an HF root or packed text-encoder file."""
    return _builder(root, frozenset({"gemma4", "gemma4_unified"}))
