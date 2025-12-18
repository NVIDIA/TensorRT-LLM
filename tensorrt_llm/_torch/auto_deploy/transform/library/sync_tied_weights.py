"""Transform to sync tied weights after submodule export and weight loading.

When a submodule is exported to a GraphModule, weight tying between parameters
inside and outside the exported submodule can break. This transform restores
the tying by making non-exported parameters reference the exported parameters'
tensors.

This transform runs AFTER weights are loaded (stage: post_load_fusion) so it can
directly sync the already-loaded weights.

This is particularly important for VLM models like Gemma3 where:
- embed_tokens.weight is inside the exported language_model
- lm_head.weight is outside (at parent level)
- They share the same weight via _tied_weights_keys
"""

from typing import List, Set, Tuple, Type

import torch
import torch.nn as nn

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _get_tied_weight_pairs(mod: nn.Module) -> List[Tuple[str, str]]:
    """Extract tied weight pairs from model's _tied_weights_keys attribute.

    HF models can declare tied weights in multiple formats:
    1. Dict format: {"lm_head.weight": "model.embed_tokens.weight"} - explicit dst->src mapping
    2. List format: ["lm_head.weight"] - just lists the tied key, src is from get_input_embeddings()

    For list format, we use get_input_embeddings() and get_output_embeddings() to determine
    the actual tying relationship.

    Args:
        mod: The model to extract tied weight pairs from.

    Returns:
        List of (dst_key, src_key) tuples where dst is tied TO src.
        Returns empty list if no tied weights are declared.
    """
    tied_keys = getattr(mod, "_tied_weights_keys", None)
    if not tied_keys:
        return []

    # Dict format: explicit mapping {"dst": "src"}
    if isinstance(tied_keys, dict):
        return list(tied_keys.items())

    # List/set format: this typically means word embeddings are tied
    # Check config.tie_word_embeddings (HF's standard flag) to confirm
    if isinstance(tied_keys, (list, tuple, set)):
        # Check if tie_word_embeddings is enabled (HF's standard config flag)
        config = getattr(mod, "config", None)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", None)

        # Also check text_config for VLM models
        if tie_word_embeddings is None and config is not None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                tie_word_embeddings = getattr(text_config, "tie_word_embeddings", None)

        if not tie_word_embeddings:
            ad_logger.debug(
                f"_tied_weights_keys={tied_keys} but tie_word_embeddings is not True, skipping"
            )
            return []

        # tie_word_embeddings=True and we have a list like ["lm_head.weight"]
        # Use HF's standard methods to find the actual tied modules
        input_embeddings = None
        output_embeddings = None
        input_embed_key = None
        output_embed_key = None

        try:
            if hasattr(mod, "get_input_embeddings"):
                input_embeddings = mod.get_input_embeddings()
            if hasattr(mod, "get_output_embeddings"):
                output_embeddings = mod.get_output_embeddings()
        except Exception:
            pass
        if input_embeddings is None or output_embeddings is None:
            ad_logger.warning(
                f"tie_word_embeddings=True but get_input_embeddings/get_output_embeddings "
                f"returned None (input={input_embeddings}, output={output_embeddings})"
            )
            return []

        # Find the parameter paths for input and output embeddings
        for name, submod in mod.named_modules():
            if submod is input_embeddings:
                input_embed_key = f"{name}.weight" if name else "weight"
            if submod is output_embeddings:
                output_embed_key = f"{name}.weight" if name else "weight"
        if input_embed_key and output_embed_key and input_embed_key != output_embed_key:
            # output (lm_head) is tied TO input (embed_tokens)
            ad_logger.debug(
                f"Inferred tied weight pair: {output_embed_key} -> {input_embed_key} "
                f"(tie_word_embeddings=True)"
            )
            return [(output_embed_key, input_embed_key)]

        ad_logger.warning(
            f"tie_word_embeddings=True but could not find embedding paths: "
            f"input={input_embed_key}, output={output_embed_key}"
        )
        return []

    return []


def _get_exported_submodule_keys(mod: nn.Module) -> List[str]:
    """Infer which submodules were exported by detecting GraphModules.

    Args:
        mod: The root model to search for exported submodules.

    Returns:
        List of submodule key paths that are GraphModules (i.e., were exported).
    """
    exported_keys = []
    for name, submod in mod.named_modules():
        if isinstance(submod, torch.fx.GraphModule):
            exported_keys.append(name)
    return exported_keys


def _detect_cross_boundary_tied_weights(
    mod: nn.Module,
    exported_submodule_keys: List[str],
) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """Detect tied weights that cross the export boundary.

    When a submodule is exported, weight tying between parameters inside and outside
    the exported submodule can break. This function identifies such cross-boundary pairs.

    The exported parameter becomes the canonical source of truth because it's embedded
    in the GraphModule's graph (via get_attr nodes) and cannot be easily changed.

    Args:
        mod: The root model containing both exported and non-exported submodules.
        exported_submodule_keys: List of submodule key paths that were exported.

    Returns:
        Tuple of:
        - List of (dst_key, src_key) pairs that have cross-boundary tying
        - Set of canonical keys (the exported ones that are sources of truth)
    """
    tied_pairs = _get_tied_weight_pairs(mod)
    if not tied_pairs:
        return [], set()

    def is_in_exported(key: str) -> bool:
        """Check if parameter key is inside an exported submodule."""
        for sub in exported_submodule_keys:
            if sub == "":  # Full model exported (root is GraphModule)
                return True
            if key.startswith(f"{sub}."):
                return True
        return False

    cross_boundary_pairs = []
    canonical_keys = set()
    for dst_key, src_key in tied_pairs:
        src_exported = is_in_exported(src_key)
        dst_exported = is_in_exported(dst_key)

        if src_exported == dst_exported:
            # Both exported or both not exported - no cross-boundary issue
            # Existing deduplication handles both-exported case
            continue

        # Cross-boundary case: one exported, one not
        cross_boundary_pairs.append((dst_key, src_key))

        # Determine which is canonical (exported)
        if src_exported:
            canonical_keys.add(src_key)
        else:
            canonical_keys.add(dst_key)

    return cross_boundary_pairs, canonical_keys


def _sync_tied_weights(
    mod: nn.Module,
    cross_boundary_pairs: List[Tuple[str, str]],
    canonical_keys: Set[str],
) -> int:
    """Sync tied weights by making non-canonical weights point to canonical weights.

    This function should be called AFTER weights are loaded. It makes the non-exported
    weight (e.g., lm_head.weight) point to the same tensor as the exported weight
    (e.g., embed_tokens.weight).

    Args:
        mod: The root model with loaded weights.
        cross_boundary_pairs: List of (dst_key, src_key) pairs with cross-boundary tying.
        canonical_keys: Set of parameter keys that are canonical (exported).

    Returns:
        Number of weights successfully synced.
    """
    synced_count = 0
    for dst_key, src_key in cross_boundary_pairs:
        # Determine canonical vs redirect keys
        if src_key in canonical_keys:
            canonical_key = src_key
            redirect_key = dst_key
        else:
            canonical_key = dst_key
            redirect_key = src_key

        try:
            # Get the loaded canonical parameter
            canonical_param = mod.get_parameter(canonical_key)

            # Parse redirect key into module path and param name
            parts = redirect_key.rsplit(".", 1)
            if len(parts) > 1:
                redirect_mod = mod.get_submodule(parts[0])
                redirect_name = parts[1]
            else:
                redirect_mod = mod
                redirect_name = parts[0]

            # Remove from _parameters so it's not a registered parameter
            # (prevents double-counting in state_dict, optimizer, etc.)
            if redirect_name in redirect_mod._parameters:
                del redirect_mod._parameters[redirect_name]

            # Sync: make redirect point to the canonical tensor
            setattr(redirect_mod, redirect_name, canonical_param)
            ad_logger.info(f"Synced tied weight: {redirect_key} -> {canonical_key} (canonical)")
            synced_count += 1
        except Exception as e:
            ad_logger.warning(f"Failed to sync tied weight {redirect_key} -> {canonical_key}: {e}")

    return synced_count


class SyncTiedWeightsConfig(TransformConfig):
    """Configuration for the sync tied weights transform."""

    pass  # No configuration options needed for now


@TransformRegistry.register("sync_tied_weights")
class SyncTiedWeights(BaseTransform):
    """Sync tied weights that cross the export boundary.

    This transform runs AFTER weights are loaded (stage: post_load_fusion).
    It detects GraphModules to infer which submodules were exported, then
    syncs any tied weights that cross the export boundary.

    For example, in Gemma3 VLM:
    - language_model is exported to GraphModule (contains embed_tokens.weight)
    - lm_head is at parent level (not exported)
    - _tied_weights_keys declares lm_head.weight -> embed_tokens.weight
    - This transform makes lm_head.weight reference embed_tokens.weight
    """

    config: SyncTiedWeightsConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return SyncTiedWeightsConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # Infer exported submodules by detecting GraphModules
        exported_keys = _get_exported_submodule_keys(mod)
        if not exported_keys:
            # No GraphModules found - nothing to sync
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Detect cross-boundary tied weights
        cross_boundary_pairs, canonical_keys = _detect_cross_boundary_tied_weights(
            mod, exported_keys
        )

        if not cross_boundary_pairs:
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Directly sync the weights (weights are already loaded at this point)
        synced_count = _sync_tied_weights(mod, cross_boundary_pairs, canonical_keys)

        return mod, TransformInfo(
            skipped=False,
            num_matches=synced_count,
            is_clean=True,
            has_valid_shapes=True,
        )
