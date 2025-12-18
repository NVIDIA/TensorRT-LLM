"""Post-export transform to tag VLM-specific mask routing metadata on attention nodes.

This transform annotates `torch.ops.auto_deploy.torch_attention` FX nodes with:

  node.meta["mask_kind"] in {"full", "sliding", "none"}

The intent is to enable downstream KV-cache insertion (e.g., FlashInfer) to plumb optional custom
VLM masks only when the exported model is multimodal and the layer-type signal is available.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _mask_kind_from_layer_type(layer_type: str) -> str:
    if layer_type == "full_attention":
        return "full"
    if layer_type == "sliding_attention":
        return "sliding"
    return "none"


def _is_vlm_config(cfg: Any) -> bool:
    """Best-effort check whether a config corresponds to a multimodal/VLM model."""
    if cfg is None:
        return False
    # Common HF VLM markers.
    if getattr(cfg, "image_token_id", None) is not None:
        return True
    if hasattr(cfg, "vision_config"):
        return True
    if hasattr(cfg, "vision_model"):
        return True
    if hasattr(cfg, "mm_vision_tower") or hasattr(cfg, "vision_tower"):
        return True
    # Nested multimodal configs often contain both text_config and vision_config.
    if hasattr(cfg, "text_config") and hasattr(cfg, "vision_config"):
        return True
    return False


def _build_mask_kind_by_module(sub_mod: torch.nn.Module) -> Dict[str, str]:
    """Build a mapping {qualified_module_name: mask_kind} for attention modules.

    This mapping is later consumed by the post-export transform `tag_vlm_mask_kind` via FX node
    metadata (`nn_module_stack`) to deterministically tag each `torch_attention` node.
    """
    cfg = getattr(sub_mod, "config", None)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None and hasattr(cfg, "text_config"):
        layer_types = getattr(cfg.text_config, "layer_types", None)

    out: Dict[str, str] = {}
    for name, m in sub_mod.named_modules():
        # Prefer explicit per-layer attention_type if present.
        attn_type = getattr(m, "attention_type", None)
        if isinstance(attn_type, str):
            out[name] = _mask_kind_from_layer_type(attn_type)
            continue

        # Fallback: infer from config.layer_types using layer_idx (Gemma3-style).
        layer_idx = getattr(m, "layer_idx", None)
        if (
            isinstance(layer_idx, int)
            and isinstance(layer_types, (list, tuple))
            and 0 <= layer_idx < len(layer_types)
        ):
            lt = layer_types[layer_idx]
            if isinstance(lt, str):
                out[name] = _mask_kind_from_layer_type(lt)
                continue

    return out


class TagVlmMaskKindConfig(TransformConfig):
    """Configuration for tagging VLM mask_kind on attention nodes."""

    enabled: bool = Field(default=True)


@TransformRegistry.register("tag_vlm_mask_kind")
class TagVlmMaskKind(BaseTransform):
    config: TagVlmMaskKindConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TagVlmMaskKindConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Gating + mapping are provided by export_to_gm via gm.meta.
        ad_meta: Dict = getattr(gm, "meta", {}) or {}
        is_vlm = bool(ad_meta.get("ad_is_vlm", False))
        mask_kind_by_module: Dict[str, str] = ad_meta.get("ad_mask_kind_by_module", {}) or {}
        if not is_vlm or not mask_kind_by_module:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        num_tagged = 0
        for n in gm.graph.nodes:
            # Handle both OpOverloadPacket and OpOverload targets.
            if not (
                is_op(n, torch.ops.auto_deploy.torch_attention)
                or is_op(n, torch.ops.auto_deploy.torch_attention.default)
            ):
                continue
            module_path = _get_innermost_module_path(n)
            if module_path is None:
                continue

            mk = _lookup_mask_kind(mask_kind_by_module, module_path)
            if mk not in ("full", "sliding", "none"):
                mk = "none"

            if n.meta.get("mask_kind", None) != mk:
                n.meta["mask_kind"] = mk
                num_tagged += 1

        # Tagging node.meta does not change graph semantics or shapes.
        return gm, TransformInfo(
            skipped=False, num_matches=num_tagged, is_clean=True, has_valid_shapes=True
        )


def _get_innermost_module_path(n: Node) -> Optional[str]:
    """Best-effort get innermost module path from nn_module_stack metadata."""
    stack = n.meta.get("nn_module_stack", None)
    if isinstance(stack, dict) and stack:
        # torch.fx records stack as {qualname: ModuleType, ...}.
        # Prefer the deepest/most-specific qualname (most segments) rather than relying on dict order.
        try:
            keys = list(stack.keys())
            # Strip "L__self__" prefix if present
            innermost = max(keys, key=lambda k: (k.count("."), len(k)))
            innermost = re.sub(r"^L__self__", "", innermost)
            return innermost
        except Exception:
            return None
    return None


def _lookup_mask_kind(mask_kind_by_module: Dict[str, str], module_path: str) -> str:
    """Lookup mask kind with fallback to suffix matches (handles differing root prefixes)."""
    if module_path in mask_kind_by_module:
        return mask_kind_by_module[module_path]
    # Try stripping leading components until we find a match.
    parts = module_path.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in mask_kind_by_module:
            return mask_kind_by_module[suffix]
    return "none"
