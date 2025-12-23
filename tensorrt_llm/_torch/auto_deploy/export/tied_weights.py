"""Handle cross-boundary tied weights during export.

When a submodule is exported to a GraphModule, weight tying between parameters
inside and outside the exported submodule can break. This module provides utilities
to detect and sync these cross-boundary tied weights via load_state_dict hooks.

This is particularly important for VLM models like Gemma3 where:
- embed_tokens.weight is inside the exported language_model
- lm_head.weight is outside (at parent level)
- They share the same weight via _tied_weights_keys
"""

from typing import List, Set, Tuple

import torch.nn as nn

from ..utils.logger import ad_logger


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


def _detect_cross_boundary_pairs(
    tied_pairs: List[Tuple[str, str]],
    exported_submodule_keys: List[str],
) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """Detect which tied weight pairs cross the export boundary.

    Args:
        tied_pairs: List of (dst_key, src_key) tied weight pairs.
        exported_submodule_keys: List of submodule key paths that were exported.

    Returns:
        Tuple of:
        - List of (dst_key, src_key) pairs that cross the boundary
        - Set of canonical keys (the exported ones that are sources of truth)
    """

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


def add_load_hook_for_cross_boundary_tied_weights(
    mod: nn.Module,
    exported_submodule_keys: List[str],
) -> None:
    """Register a post-hook to sync tied weights that cross the export boundary.

    When a submodule is exported to a GraphModule, weight tying between parameters
    inside and outside the exported submodule can break. This function registers a
    load_state_dict post-hook that syncs these weights after loading.

    The exported parameter becomes the canonical source of truth because it's embedded
    in the GraphModule's graph (via get_attr nodes) and cannot be easily changed.

    Args:
        mod: The root model containing both exported and non-exported submodules.
        exported_submodule_keys: List of submodule key paths that were exported.
    """
    tied_pairs = _get_tied_weight_pairs(mod)
    if not tied_pairs:
        return

    cross_boundary_pairs, canonical_keys = _detect_cross_boundary_pairs(
        tied_pairs, exported_submodule_keys
    )

    if not cross_boundary_pairs:
        return

    # Track whether we've already synced (hook fires once per checkpoint shard)
    synced = set()

    def sync_tied_weights_hook(module, incompatible_keys):
        """Post-hook to sync tied weights after load_state_dict completes."""
        for dst_key, src_key in cross_boundary_pairs:
            # Determine canonical vs redirect keys
            if src_key in canonical_keys:
                canonical_key = src_key
                redirect_key = dst_key
            else:
                canonical_key = dst_key
                redirect_key = src_key

            # Skip if already synced (hook fires multiple times for sharded checkpoints)
            if redirect_key in synced:
                continue

            try:
                # Get the loaded canonical parameter
                canonical_param = module.get_parameter(canonical_key)

                # Parse redirect key into module path and param name
                parts = redirect_key.rsplit(".", 1)
                if len(parts) > 1:
                    redirect_mod = module.get_submodule(parts[0])
                    redirect_name = parts[1]
                else:
                    redirect_mod = module
                    redirect_name = parts[0]

                # Safeguard: check if redirect weight was sharded differently
                # If so, it would have a different shape than canonical - this is an error
                redirect_param = getattr(redirect_mod, redirect_name, None)
                if redirect_param is not None and hasattr(redirect_param, "shape"):
                    if redirect_param.shape != canonical_param.shape:
                        raise ValueError(
                            f"Tied weight sharding mismatch! "
                            f"{redirect_key} has shape {redirect_param.shape} but "
                            f"{canonical_key} has shape {canonical_param.shape}. "
                            f"Tied weights must be sharded identically."
                        )

                # Remove from _parameters so it's not a registered parameter
                # (prevents double-counting in state_dict, optimizer, etc.)
                if redirect_name in redirect_mod._parameters:
                    del redirect_mod._parameters[redirect_name]

                # Sync: make redirect point to the canonical tensor
                setattr(redirect_mod, redirect_name, canonical_param)
                synced.add(redirect_key)
                ad_logger.info(f"Synced tied weight: {redirect_key} -> {canonical_key} (canonical)")
            except ValueError:
                # Re-raise ValueError (sharding mismatch) - this is a critical error
                raise
            except Exception as e:
                ad_logger.warning(
                    f"Failed to sync tied weight {redirect_key} -> {canonical_key}: {e}"
                )

    # Register the post-hook on the root module
    mod.register_load_state_dict_post_hook(sync_tied_weights_hook)
    ad_logger.debug(
        f"Registered cross-boundary tied weight hook for {len(cross_boundary_pairs)} pairs"
    )
