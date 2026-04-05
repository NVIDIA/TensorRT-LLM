#!/usr/bin/env python3
"""Module-Level Graph Comparison: AD vs HF Debugging Tool.

This module compares AutoDeploy (AD) outputs with HuggingFace (HF) model outputs
at module boundaries to identify which module first diverges.

Compares at:
    - embed_tokens: Embedding layer output
    - self_attn: Attention layer output
    - block_sparse_moe: MoE layer output
    - norm: Final normalization output
    - lm_head: Output projection

Usage at breakpoint in optimizer.py:
    from tensorrt_llm._torch.auto_deploy.utils.graph_debug_compare import run_comparison
    run_comparison(mod, cm, self.factory)
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .debug_interpreter import run_interpreter_with_captures
from .graph_debug_utils import compare_tensors, load_debug_artifacts
from .logger import extract_graph_metadata

# ============================================================================
# Scatter Plot Configuration
# ============================================================================
SCATTER_MAX_POINTS = 50000  # Max points to plot (subsample if more)


def plot_tensor_scatter(
    hf_tensor: torch.Tensor,
    ad_tensor: torch.Tensor,
    out_png: str,
    title: str,
    max_points: int = SCATTER_MAX_POINTS,
    seed: int = 0,
) -> tuple:
    """Flatten two tensors, subsample, and produce an HF vs AD scatter plot.

    Args:
        hf_tensor: HuggingFace reference tensor
        ad_tensor: AutoDeploy tensor to compare
        out_png: Output path for PNG file
        title: Plot title
        max_points: Maximum number of points to plot (subsamples if more)
        seed: Random seed for subsampling

    Returns:
        Tuple of (correlation, num_points_plotted)
    """
    hf_flat = hf_tensor.detach().float().cpu().reshape(-1)
    ad_flat = ad_tensor.detach().float().cpu().reshape(-1)

    min_len = min(hf_flat.numel(), ad_flat.numel())
    if min_len == 0:
        print(f"      Warning: Scatter skipped (empty tensors): {title}")
        return float("nan"), 0

    hf_flat = hf_flat[:min_len]
    ad_flat = ad_flat[:min_len]
    finite_mask = torch.isfinite(hf_flat) & torch.isfinite(ad_flat)
    hf_flat = hf_flat[finite_mask]
    ad_flat = ad_flat[finite_mask]

    if hf_flat.numel() == 0:
        print(f"      Warning: Scatter skipped (no finite points): {title}")
        return float("nan"), 0

    hf_np = hf_flat.numpy()
    ad_np = ad_flat.numpy()

    n_points = hf_np.size
    if n_points > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, size=max_points, replace=False)
        hf_np = hf_np[indices]
        ad_np = ad_np[indices]
        n_points = hf_np.size

    corr = float(np.corrcoef(hf_np, ad_np)[0, 1]) if hf_np.size >= 2 else float("nan")

    png_dir = os.path.dirname(out_png)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    lo = float(min(hf_np.min(), ad_np.min()))
    hi = float(max(hf_np.max(), ad_np.max()))
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(hf_np, ad_np, s=1, alpha=0.25, linewidths=0)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.8)
    plt.xlabel("HF (reference)")
    plt.ylabel("AutoDeploy")
    plt.title(f"{title}\nPearson r = {corr:.5f}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"      Scatter saved: {out_png} (corr={corr:.4f}, points={n_points})")
    return corr, n_points


def _find_ad_weight(
    ad_state: Dict[str, torch.Tensor],
    hf_module_path: str,
    hf_key: str,
) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[str]]:
    """Find matching AD weight with multiple strategies.

    Handles:
    1. _orig_mod. prefix with dot notation (new format)
    2. Fused MoE weights (experts stacked into single tensors)
    3. Underscore format (legacy fallback)

    Returns:
        (weight_tensor, matched_key, slice_info) or (None, None, None) if not found
    """
    # Strategy 1: _orig_mod. prefix with dot notation
    orig_mod_key = f"_orig_mod.{hf_module_path}.{hf_key}"
    if orig_mod_key in ad_state:
        return ad_state[orig_mod_key], orig_mod_key, None

    # Strategy 2: Check for fused MoE weights
    expert_match = re.match(r"experts\.(\d+)\.(w[123])\.weight", hf_key)
    if expert_match:
        expert_idx = int(expert_match.group(1))
        weight_type = expert_match.group(2)

        # Find fused weight key
        for ad_key in ad_state.keys():
            if weight_type in ("w1", "w3") and "fused_moe_w3_w1_stacked" in ad_key:
                fused = ad_state[ad_key]
                intermediate_size = fused.shape[1] // 2
                if weight_type == "w3":
                    extracted = fused[expert_idx, :intermediate_size, :]
                    return extracted, ad_key, f"[{expert_idx}, :{intermediate_size}, :]"
                else:  # w1
                    extracted = fused[expert_idx, intermediate_size:, :]
                    return extracted, ad_key, f"[{expert_idx}, {intermediate_size}:, :]"
            elif weight_type == "w2" and "fused_moe_w2_stacked" in ad_key:
                extracted = ad_state[ad_key][expert_idx]
                return extracted, ad_key, f"[{expert_idx}]"

    # Strategy 3: Original underscore format (fallback)
    ad_prefix = hf_module_path.replace(".", "_") + "_"
    ad_key_candidate = ad_prefix + hf_key.replace(".", "_")
    if ad_key_candidate in ad_state:
        return ad_state[ad_key_candidate], ad_key_candidate, None

    # Strategy 4: Suffix/contains match (flexible matching)
    for ad_k, ad_v in ad_state.items():
        if ad_key_candidate in ad_k or ad_k.endswith(ad_key_candidate):
            return ad_v, ad_k, None
        # Also try matching just the param name with prefix
        param_suffix = hf_key.replace(".", "_")
        if ad_k.endswith("_" + param_suffix) and ad_prefix.rstrip("_") in ad_k:
            return ad_v, ad_k, None

    return None, None, None


# ============================================================================
# Configuration
# ============================================================================
# Tolerances for bfloat16 comparison - tightened to catch 1-bit errors
# bfloat16 has ~7 bits of mantissa. 0.007812 is 2^-7.
# We want to flag diffs > 1e-4 even for values > 1.0.
ATOL = 1e-3
RTOL = 1e-3  # 0.01% relative tolerance


# ============================================================================
# Stage 1: Coarse Module Boundary Comparison
# ============================================================================


def _discover_hf_modules(hf_model) -> List[str]:
    """Dynamically discover key modules from the HF model structure.

    For GLM-4.7-Flash (and similar models):
    - embed_tokens: Embedding layer
    - layers.0.self_attn: Layer 0 attention
    - layers.0.mlp: Layer 0 FFN (dense MLP)
    - layers.1.self_attn: Layer 1 attention
    - layers.1.mlp: Layer 1 FFN (MoE)
    - norm: Final layer norm
    - lm_head: Output projection

    Returns:
        List of module paths to hook
    """
    modules_to_hook = []

    # Find the inner model (usually accessed via .model attribute)
    inner_model = getattr(hf_model, "model", hf_model)

    # 1. embed_tokens
    if hasattr(inner_model, "embed_tokens"):
        modules_to_hook.append("model.embed_tokens")
    elif hasattr(inner_model, "embeddings"):
        modules_to_hook.append("model.embeddings")

    # 2. Layer 0 and Layer 1 attention and FFN
    # GLM-4.7-Flash: Layer 0 has dense MLP, Layer 1+ has MoE
    layers = getattr(inner_model, "layers", None)
    if layers is not None:
        attn_names = ["self_attn", "attention", "attn"]
        ffn_names = ["block_sparse_moe", "moe", "mlp", "feed_forward", "ffn"]

        # Layer 0 (dense MLP)
        if len(layers) > 0:
            layer0 = layers[0]
            for attn_name in attn_names:
                if hasattr(layer0, attn_name):
                    modules_to_hook.append(f"model.layers.0.{attn_name}")
                    break
            for ffn_name in ffn_names:
                if hasattr(layer0, ffn_name):
                    modules_to_hook.append(f"model.layers.0.{ffn_name}")
                    break

        # Layer 1 (MoE for GLM-4.7-Flash)
        if len(layers) > 1:
            layer1 = layers[1]
            for attn_name in attn_names:
                if hasattr(layer1, attn_name):
                    modules_to_hook.append(f"model.layers.1.{attn_name}")
                    break
            for ffn_name in ffn_names:
                if hasattr(layer1, ffn_name):
                    modules_to_hook.append(f"model.layers.1.{ffn_name}")
                    break

    # 3. Final norm
    norm_names = ["norm", "final_layernorm", "ln_f"]
    for norm_name in norm_names:
        if hasattr(inner_model, norm_name):
            modules_to_hook.append(f"model.{norm_name}")
            break

    # 4. lm_head (on the outer model)
    if hasattr(hf_model, "lm_head"):
        modules_to_hook.append("lm_head")
    elif hasattr(hf_model, "output"):
        modules_to_hook.append("output")

    print(f"[_discover_hf_modules] Discovered modules: {modules_to_hook}")
    return modules_to_hook


def _build_module_mapping(hf_modules: List[str]) -> Dict[str, str]:
    """Build a mapping from HF module paths to simplified AD keys.

    For GLM-4.7-Flash, we need layer-specific keys:
    - model.layers.0.self_attn -> layer0_self_attn
    - model.layers.0.mlp -> layer0_mlp
    - model.layers.1.self_attn -> layer1_self_attn
    - model.layers.1.mlp -> layer1_mlp

    Args:
        hf_modules: List of HF module paths

    Returns:
        Dict mapping HF paths to simplified keys
    """
    mapping = {}
    for hf_path in hf_modules:
        parts = hf_path.split(".")
        if len(parts) == 1:
            # Simple case: "lm_head" -> "lm_head"
            mapping[hf_path] = hf_path
        elif parts[-1] in ("embed_tokens", "embeddings"):
            mapping[hf_path] = "embed_tokens"
        elif parts[-1] in ("self_attn", "attention", "attn"):
            # Check if this is a layer-specific module: model.layers.N.self_attn
            if len(parts) >= 4 and parts[1] == "layers":
                layer_idx = parts[2]
                mapping[hf_path] = f"layer{layer_idx}_self_attn"
            else:
                mapping[hf_path] = "self_attn"
        elif parts[-1] in ("block_sparse_moe", "moe", "mlp", "feed_forward", "ffn"):
            # Check if this is a layer-specific module: model.layers.N.mlp
            if len(parts) >= 4 and parts[1] == "layers":
                layer_idx = parts[2]
                mapping[hf_path] = f"layer{layer_idx}_mlp"
            else:
                mapping[hf_path] = "ffn"
        elif parts[-1] in ("norm", "final_layernorm", "ln_f"):
            mapping[hf_path] = "norm"
        else:
            # Fallback: use the last part
            mapping[hf_path] = parts[-1]

    print(f"[_build_module_mapping] Module mapping: {mapping}")
    return mapping


class HFHookCapture:
    """Context manager to capture HF model inputs and outputs at module boundaries."""

    def __init__(self, model, module_names: List[str]):
        """Initialize hook capture.

        Args:
            model: HuggingFace model
            module_names: List of module names to hook (e.g., ["embed_tokens", "self_attn"])
        """
        self.model = model
        self.module_names = module_names
        self.captured: Dict[str, Any] = {}  # outputs
        self.captured_inputs: Dict[str, Any] = {}  # inputs
        self.handles = []

    def __enter__(self):
        for name in self.module_names:
            module = self._find_module(name)
            if module is not None:
                # Register pre-hook to capture inputs
                pre_handle = module.register_forward_pre_hook(self._make_pre_hook(name))
                self.handles.append(pre_handle)
                # Register forward hook to capture outputs
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _find_module(self, name: str):
        """Find a module by name in the model."""
        # Try direct access first
        try:
            module = self.model
            for part in name.split("."):
                module = getattr(module, part)
            return module
        except AttributeError:
            pass

        # Search in named_modules
        for full_name, mod in self.model.named_modules():
            if full_name.endswith(name) or name in full_name:
                return mod

        print(f"Warning: Could not find module '{name}'")
        return None

    def _clone_value(self, v: Any) -> Any:
        """Clone a tensor value, preserving structure for tuples/lists."""
        if isinstance(v, torch.Tensor):
            return v.detach().clone()
        elif isinstance(v, tuple):
            return tuple(self._clone_value(x) for x in v)
        elif isinstance(v, list):
            return [self._clone_value(x) for x in v]
        return v

    def _make_pre_hook(self, name: str):
        """Create a pre-hook to capture inputs."""

        def pre_hook(module, args):
            # Capture all tensor args
            cloned_args = tuple(self._clone_value(a) for a in args)
            self.captured_inputs[name] = cloned_args

        return pre_hook

    def _make_hook(self, name: str):
        """Create a forward hook to capture outputs."""

        def hook(module, input, output):
            # Handle different output types
            self.captured[name] = self._clone_value(output)

        return hook


def stage1_coarse_comparison(
    hf_model,
    ad_gm,
    ad_metadata: Dict[str, Any],
    input_ids: torch.Tensor,
    ad_named_args: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Tuple[bool, Optional[str], Dict[str, Dict[str, Any]]]:
    """Run Stage 1: Coarse module boundary comparison.

    Args:
        hf_model: HuggingFace model
        ad_gm: AD GraphModule (final)
        ad_metadata: AD metadata dict
        input_ids: Input token IDs
        ad_named_args: Optional dict of named args to feed AD GraphModule
        device: Device to run on

    Returns:
        Tuple of:
        - all_passed: True if all comparisons pass
        - first_failing_module: Name of first failing module (or None)
        - results: Dict of comparison results per module
    """

    def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
        if t is None:
            return {"status": "none"}
        if not isinstance(t, torch.Tensor):
            return {"status": "non_tensor", "type": str(type(t))}
        if t.numel() == 0:
            return {"shape": tuple(t.shape), "dtype": str(t.dtype), "status": "empty"}
        tf = t.float()
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "min": tf.min().item(),
            "max": tf.max().item(),
            "has_nan": torch.isnan(tf).any().item(),
            "has_inf": torch.isinf(tf).any().item(),
        }

    def _hf_tensor_leaves(v: Any) -> List[torch.Tensor]:
        """Deterministically extract tensor leaves from HF outputs (tensor / tuple / list)."""
        if isinstance(v, torch.Tensor):
            return [v]
        if isinstance(v, (tuple, list)):
            return [t for t in v if isinstance(t, torch.Tensor)]
        return []

    print("\n" + "=" * 80)
    print("STAGE 1: Coarse Module Boundary Comparison")
    print("=" * 80)

    # Dynamically discover modules from the HF model
    hf_modules = _discover_hf_modules(hf_model)

    # Move models to device
    hf_model = hf_model.to(device)
    ad_gm = ad_gm.to(device)
    input_ids = input_ids.to(device)

    # Run HF model with hooks
    print("\nRunning HF model with hooks...")
    with torch.inference_mode():
        with HFHookCapture(hf_model, hf_modules) as hf_capture:
            _ = hf_model(input_ids)
            hf_captured = hf_capture.captured
            hf_captured_inputs = hf_capture.captured_inputs

    print(f"HF captured modules (outputs): {list(hf_captured.keys())}")
    print(f"HF captured modules (inputs): {list(hf_captured_inputs.keys())}")

    # Build module mapping dynamically from discovered modules
    module_mapping = _build_module_mapping(hf_modules)

    # Find AD output nodes by pattern matching on node names
    # GLM-4.7-Flash node naming patterns (from exported graph):
    #   - model_embed_tokens_embedding
    #   - model_layers_0_self_attn_o_proj_torch_linear_simple_* (layer 0 attn)
    #   - model_layers_0_mlp_down_proj_torch_linear_simple_* (layer 0 dense MLP)
    #   - model_layers_1_self_attn_o_proj_torch_linear_simple_* (layer 1 attn)
    #   - model_layers_1_mlp_add_* (layer 1 MoE output)
    #   - model_norm_mul_*
    #   - lm_head_torch_linear_simple_*
    ad_node_patterns = {
        "embed_tokens": r"^model_embed_tokens_embedding$",
        # Layer 0: dense MLP
        "layer0_self_attn": r"^model_layers_0_self_attn_o_proj_torch_linear_simple_\d+$",
        "layer0_mlp": r"^model_layers_0_mlp_down_proj_torch_linear_simple_\d+$",
        # Layer 1: MoE
        "layer1_self_attn": r"^model_layers_1_self_attn_o_proj_torch_linear_simple_\d+$",
        "layer1_mlp": r"^model_layers_1_mlp_add_\d+$",
        # Final outputs
        "norm": r"^model_norm_mul_\d+$",
        "lm_head": r"^lm_head_torch_linear_simple_\d+$",
    }

    # Find matching nodes in the AD graph
    all_node_names = [n.name for n in ad_gm.graph.nodes]
    ad_output_nodes: Dict[str, Optional[str]] = {}

    print("\nMatching AD nodes by pattern:")
    for key, pattern in ad_node_patterns.items():
        matches = [n for n in all_node_names if re.match(pattern, n)]
        if matches:
            # Take the last match (usually the final output of that module)
            ad_output_nodes[key] = matches[-1]
            print(f"  [{key}] -> {matches[-1]}")
        else:
            ad_output_nodes[key] = None
            print(f"  [{key}] -> NO MATCH for pattern: {pattern}")

    # Build list of nodes to capture
    ad_capture_nodes = set(n for n in ad_output_nodes.values() if n is not None)

    # Build inputs for the AD graph in placeholder order
    placeholders = [n.target for n in ad_gm.graph.nodes if n.op == "placeholder"]
    ad_inputs = {}
    missing_inputs = []
    ad_source = ad_named_args or {}
    for name in placeholders:
        if name in ad_source:
            val = ad_source[name]
            ad_inputs[name] = val.to(device) if isinstance(val, torch.Tensor) else val
        elif name == "input_ids":
            ad_inputs[name] = input_ids
        else:
            missing_inputs.append(name)

    if missing_inputs:
        print(f"WARNING: Missing inputs for placeholders: {missing_inputs}")

    with torch.inference_mode():
        ad_captured = run_interpreter_with_captures(
            ad_gm,
            ad_capture_nodes,
            **ad_inputs,
        )

    print(f"AD captured nodes: {list(ad_captured.keys())}")

    # Compare at each boundary (inputs and outputs)
    results = {}
    all_passed = True
    first_failing_module = None

    def _compare_module_weights(
        hf_model,
        ad_gm,
        hf_module_path: str,
        device: str,
    ) -> Dict[str, Any]:
        """Compare weights between HF module and AD using direct key matching.

        Both state_dicts use the same dot-notation keys, so just compare directly.
        """
        hf_state = hf_model.state_dict()
        ad_state = ad_gm.state_dict()

        # DEBUG: Print state dict keys for MoE modules
        if "mlp" in hf_module_path and "layers.1" in hf_module_path:
            print("\n  [DEBUG] State Dict Keys for MoE layer:")
            print("  [DEBUG] HF state dict keys (mlp/experts):")
            hf_moe_keys = sorted([k for k in hf_state.keys() if "layers.1.mlp" in k])
            for k in hf_moe_keys:
                shape = tuple(hf_state[k].shape) if hasattr(hf_state[k], "shape") else "N/A"
                print(f"    HF: {k} -> shape={shape}")

            print("  [DEBUG] AD state dict keys (mlp/experts):")
            ad_moe_keys = sorted(
                [k for k in ad_state.keys() if "layers.1.mlp" in k or "layers_1_mlp" in k]
            )
            for k in ad_moe_keys:
                shape = tuple(ad_state[k].shape) if hasattr(ad_state[k], "shape") else "N/A"
                print(f"    AD: {k} -> shape={shape}")

            # Also check if there are any mismatches in key sets
            hf_set = set(hf_moe_keys)
            ad_set = set(ad_moe_keys)
            only_in_hf = hf_set - ad_set
            only_in_ad = ad_set - hf_set
            if only_in_hf:
                print(f"  [DEBUG] Keys ONLY in HF: {sorted(only_in_hf)}")
            if only_in_ad:
                print(f"  [DEBUG] Keys ONLY in AD: {sorted(only_in_ad)}")

            # DEBUG: Check what the graph actually references (get_attr nodes)
            print("  [DEBUG] AD graph 'get_attr' nodes for MoE weights:")
            for node in ad_gm.graph.nodes:
                if node.op == "get_attr" and (
                    "layers_1_mlp" in node.target or "layers.1.mlp" in node.target
                ):
                    # Get the actual tensor from the graph
                    try:
                        attr_val = ad_gm
                        for part in node.target.split("."):
                            attr_val = getattr(attr_val, part)
                        shape = tuple(attr_val.shape) if hasattr(attr_val, "shape") else "N/A"
                        # Check if this tensor is same object as state_dict
                        state_key = node.target
                        in_state = state_key in ad_state
                        same_obj = in_state and (
                            ad_state[state_key].data_ptr() == attr_val.data_ptr()
                        )
                        print(
                            f"    get_attr: {node.target} -> shape={shape},"
                            f" in_state_dict={in_state},"
                            f"same_obj={same_obj}"
                        )
                    except Exception as e:
                        print(f"    get_attr: {node.target} -> ERROR: {e}")

        # Filter to keys matching this module path
        prefix = hf_module_path + "."
        hf_keys = [k for k in hf_state.keys() if k.startswith(prefix)]

        if not hf_keys:
            return {"status": "skip", "reason": f"No HF weights for '{hf_module_path}'"}

        weight_comparisons = []
        for hf_key in hf_keys:
            hf_weight = hf_state[hf_key]

            if hf_key not in ad_state:
                weight_comparisons.append(
                    {
                        "hf_key": hf_key,
                        "match": False,
                        "reason": "not in AD",
                    }
                )
                continue

            ad_weight = ad_state[hf_key]
            hf_w = hf_weight.to(device)
            ad_w = ad_weight.to(device)

            if hf_w.shape != ad_w.shape:
                weight_comparisons.append(
                    {
                        "hf_key": hf_key,
                        "match": False,
                        "reason": f"shape: HF={tuple(hf_w.shape)} AD={tuple(ad_w.shape)}",
                    }
                )
                continue

            diff = (hf_w.float() - ad_w.float()).abs()
            max_diff = diff.max().item()
            match = max_diff < 1e-6

            # DEBUG: Print first few values for mismatching expert weights
            if (
                not match
                and "experts" in hf_key
                and ("gate_up_proj" in hf_key or "down_proj" in hf_key)
            ):
                hf_flat = hf_w.flatten()[:10].tolist()
                ad_flat = ad_w.flatten()[:10].tolist()
                print(f"\n  [DEBUG] Weight value comparison for {hf_key}:")
                print(f"    HF first 10 values: {[f'{v:.4f}' for v in hf_flat]}")
                print(f"    AD first 10 values: {[f'{v:.4f}' for v in ad_flat]}")
                # Check if AD values look like random initialization (typically small random values)
                ad_mean = ad_w.float().mean().item()
                ad_std = ad_w.float().std().item()
                hf_mean = hf_w.float().mean().item()
                hf_std = hf_w.float().std().item()
                print(f"    HF stats: mean={hf_mean:.6f}, std={hf_std:.6f}")
                print(f"    AD stats: mean={ad_mean:.6f}, std={ad_std:.6f}")

            weight_comparisons.append(
                {
                    "hf_key": hf_key,
                    "match": match,
                    "max_diff": max_diff,
                }
            )

        all_match = all(c.get("match", False) for c in weight_comparisons)
        return {
            "match": all_match,
            "comparisons": weight_comparisons,
            "num_weights": len(weight_comparisons),
            "num_matched": sum(1 for c in weight_comparisons if c.get("match", False)),
        }

    print("\n--- Comparison Results (using node name patterns) ---")
    for hf_name, ad_key in module_mapping.items():
        module_results: Dict[str, Any] = {}
        module_passed = True
        print(f"\n[{ad_key}] Comparing HF '{hf_name}' vs AD node")

        # 1. Compare weights FIRST (always)
        print("  [weights] Comparing weights...")
        weight_result = _compare_module_weights(hf_model, ad_gm, hf_name, device)
        module_results["weights"] = weight_result

        if weight_result.get("status") == "skip":
            print(f"  [weights] SKIP: {weight_result.get('reason')}")
        elif weight_result.get("status") == "error":
            print(f"  [weights] ERROR: {weight_result.get('reason')}")
        else:
            num_weights = weight_result.get("num_weights", 0)
            num_matched = weight_result.get("num_matched", 0)
            if weight_result.get("match"):
                print(f"  [weights] PASS: All {num_weights} weights match")
            else:
                print(f"  [weights] FAIL: {num_matched}/{num_weights} weights match")
                # Print which weights don't match
                for comp in weight_result.get("comparisons", []):
                    if not comp.get("match", False):
                        reason = comp.get("reason", f"max_diff={comp.get('max_diff', '?')}")
                        print(f"    - MISMATCH: {comp['hf_key']} ({reason})")
                module_passed = False

        # 2. Compare activations
        hf_tensor = hf_captured.get(hf_name)
        ad_node_name = ad_output_nodes.get(ad_key)

        if hf_tensor is None:
            print(f"  [outputs] SKIP: HF module '{hf_name}' not captured")
            module_results["outputs"] = {"status": "skip", "reason": "HF not captured"}
            results[ad_key] = module_results
            continue

        # Get first tensor from HF output (handles tuples)
        hf_tensors = _hf_tensor_leaves(hf_tensor)
        if not hf_tensors:
            print(f"  [outputs] SKIP: HF output has no tensors (type={type(hf_tensor)})")
            module_results["outputs"] = {"status": "skip", "reason": "no tensor leaves"}
            results[ad_key] = module_results
            continue

        hf_out = hf_tensors[0]  # Use first tensor for comparison
        print(f"  [HF] shape={hf_out.shape}, dtype={hf_out.dtype}")

        if ad_node_name is None:
            print(f"  [AD] No matching node found for '{ad_key}' pattern")
            module_results["outputs"] = {"status": "no_match", "reason": "pattern not found"}
            module_passed = False
        elif ad_node_name not in ad_captured:
            print(f"  [AD] Node '{ad_node_name}' not captured")
            module_results["outputs"] = {"status": "not_captured", "node": ad_node_name}
            module_passed = False
        else:
            ad_out = ad_captured[ad_node_name]
            print(f"  [AD] node='{ad_node_name}', shape={ad_out.shape}, dtype={ad_out.dtype}")

            # Compare tensors
            comparison = compare_tensors(hf_out, ad_out, atol=ATOL, rtol=RTOL)
            module_results["outputs"] = comparison | {"node": ad_node_name}

            if comparison["match"]:
                print(f"  [outputs] PASS (max_diff={comparison['max_diff']:.6f})")
            else:
                print(
                    f"  [outputs] FAIL (max_diff={comparison['max_diff']:.6f}, "
                    f"mean_diff={comparison['mean_diff']:.6f})"
                )
                module_passed = False

                # Save scatter plot on failure
                if output_dir:
                    scatter_png = os.path.join(output_dir, f"{ad_key}_output_scatter.png")
                    plot_tensor_scatter(hf_out, ad_out, scatter_png, f"{ad_key}: HF vs AD")

        results[ad_key] = module_results
        if not module_passed:
            all_passed = False
            if first_failing_module is None:
                first_failing_module = ad_key
                # Summarize the issue
                weights_ok = weight_result.get("match", False) or weight_result.get("status") in (
                    "skip",
                    "error",
                )
                outputs_ok = module_results.get("outputs", {}).get("match", False)
                if weights_ok and not outputs_ok:
                    print(f"  -> First divergence at {ad_key}: weights OK, activations DIFFER")
                elif not weights_ok:
                    print(f"  -> First divergence at {ad_key}: weights MISMATCH")

    return all_passed, first_failing_module, results


# ============================================================================
# RoPE Comparison Helpers
# ============================================================================


def _discover_rope_nodes(gm) -> List[str]:
    """Dynamically discover RoPE-related nodes from the graph.

    Looks for patterns like:
    - position_ids placeholder
    - q_proj, k_proj linear nodes
    - rms_norm nodes
    - view, transpose, slice, cat, add nodes in self_attn

    Returns:
        List of node names to capture for RoPE comparison
    """
    nodes_to_capture = []
    node_names = [n.name for n in gm.graph.nodes]

    # Always try to capture position_ids
    if "position_ids" in node_names:
        nodes_to_capture.append("position_ids")

    # Pattern-based discovery for layer 0 self_attn
    patterns = [
        # Q/K projections (look for linear ops in self_attn)
        r".*layers_0_self_attn.*q_proj.*",
        r".*layers_0_self_attn.*k_proj.*",
        # RMSNorm
        r".*rms_norm.*",
        # View/reshape operations
        r".*layers_0_self_attn.*view.*",
        # Transpose
        r".*layers_0_self_attn.*transpose.*",
        # Slice (for partial RoPE)
        r".*layers_0_self_attn.*slice.*",
        # Unsqueeze (cos/sin)
        r".*layers_0_self_attn.*unsqueeze.*",
        # Cat (concat after RoPE)
        r".*layers_0_self_attn.*cat.*",
        # Add (rotated parts)
        r".*layers_0_self_attn.*add.*",
    ]

    for pattern in patterns:
        for name in node_names:
            if re.match(pattern, name) and name not in nodes_to_capture:
                nodes_to_capture.append(name)

    print(f"[_discover_rope_nodes] Discovered {len(nodes_to_capture)} nodes for RoPE comparison")
    if nodes_to_capture:
        # Print first few as sample
        sample = nodes_to_capture[:5]
        print(f"  Sample nodes: {sample}{'...' if len(nodes_to_capture) > 5 else ''}")

    return nodes_to_capture


def capture_hf_upstream_ops(hf_model, input_ids, device: str = "cuda"):
    """Capture upstream ops (q_proj, k_proj, q_norm, k_norm, view, transpose) from HF model.

    Returns dict with outputs from each operation in the self_attn forward path.
    """
    captured = {}
    hooks = []

    # Find the first self_attn module
    self_attn = None
    for name, module in hf_model.named_modules():
        if "layers.0.self_attn" in name and hasattr(module, "q_proj"):
            self_attn = module
            break

    if self_attn is None:
        print("[HF Upstream] WARNING: Could not find layers.0.self_attn module")
        return captured

    # Register hooks on submodules
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured[key] = output.clone().detach()
            elif isinstance(output, tuple) and len(output) > 0:
                captured[key] = (
                    output[0].clone().detach() if isinstance(output[0], torch.Tensor) else output
                )

        return hook

    # Hook q_proj, k_proj, v_proj
    if hasattr(self_attn, "q_proj"):
        hooks.append(self_attn.q_proj.register_forward_hook(make_hook("q_proj_output")))
    if hasattr(self_attn, "k_proj"):
        hooks.append(self_attn.k_proj.register_forward_hook(make_hook("k_proj_output")))
    if hasattr(self_attn, "v_proj"):
        hooks.append(self_attn.v_proj.register_forward_hook(make_hook("v_proj_output")))

    # Hook q_norm, k_norm (if present)
    if hasattr(self_attn, "q_norm"):
        hooks.append(self_attn.q_norm.register_forward_hook(make_hook("q_norm_output")))
    if hasattr(self_attn, "k_norm"):
        hooks.append(self_attn.k_norm.register_forward_hook(make_hook("k_norm_output")))

    try:
        hf_model.to(device)
        input_ids_device = input_ids.to(device) if hasattr(input_ids, "to") else input_ids
        with torch.inference_mode():
            _ = hf_model(input_ids_device)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return captured


def capture_hf_rope_outputs(hf_model, input_ids, device: str = "cuda"):
    """Capture Q/K before and after RoPE from HF model by patching apply_rotary_pos_emb.

    Returns dict with:
        - 'q_before_rope', 'k_before_rope': Q/K inputs to apply_rotary_pos_emb
        - 'q_after_rope', 'k_after_rope': Q/K outputs from apply_rotary_pos_emb
        - 'cos', 'sin': rotation values
        - 'position_ids': position IDs used
        - 'q_rot_part', 'k_rot_part': The sliced rotary parts (first rotary_dim elements)
        - 'q_pass_part', 'k_pass_part': The pass-through parts (remaining elements)
    """
    captured = {}

    # Find the modeling module dynamically
    model_class = type(hf_model)
    modeling_module = __import__(model_class.__module__, fromlist=[""])

    if not hasattr(modeling_module, "apply_rotary_pos_emb"):
        print("[HF RoPE] WARNING: Could not find apply_rotary_pos_emb in modeling module")
        return captured

    # Save original function
    original_apply_rotary = modeling_module.apply_rotary_pos_emb

    def capturing_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Capture Q/K BEFORE RoPE (inputs to the function)
        captured["q_before_rope"] = q.clone().detach()
        captured["k_before_rope"] = k.clone().detach()
        captured["cos"] = cos.clone().detach()
        captured["sin"] = sin.clone().detach()

        # Capture position_ids if provided
        if position_ids is not None:
            captured["position_ids"] = (
                position_ids.clone().detach()
                if isinstance(position_ids, torch.Tensor)
                else position_ids
            )

        # For partial RoPE, capture the sliced parts before rotation
        rotary_dim = cos.shape[-1]
        captured["rotary_dim"] = rotary_dim
        captured["q_rot_part"] = q[..., :rotary_dim].clone().detach()
        captured["q_pass_part"] = q[..., rotary_dim:].clone().detach()
        captured["k_rot_part"] = k[..., :rotary_dim].clone().detach()
        captured["k_pass_part"] = k[..., rotary_dim:].clone().detach()

        # Also capture cos/sin after unsqueeze (to match what's used in multiplication)
        cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
        sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
        captured["cos_unsqueezed"] = cos_unsqueezed.clone().detach()
        captured["sin_unsqueezed"] = sin_unsqueezed.clone().detach()

        # Call original
        q_embed, k_embed = original_apply_rotary(q, k, cos, sin, position_ids, unsqueeze_dim)

        # Capture Q/K after RoPE (already includes concat with pass-through for partial RoPE)
        captured["q_after_rope"] = q_embed.clone().detach()
        captured["k_after_rope"] = k_embed.clone().detach()

        return q_embed, k_embed

    # Patch
    modeling_module.apply_rotary_pos_emb = capturing_apply_rotary_pos_emb

    try:
        hf_model.to(device)
        input_ids_device = input_ids.to(device) if hasattr(input_ids, "to") else input_ids
        with torch.inference_mode():
            _ = hf_model(input_ids_device)
    finally:
        # Restore original
        modeling_module.apply_rotary_pos_emb = original_apply_rotary

    return captured


class CapturingInterpreter:
    """Run AD GraphModule and capture intermediate node outputs."""

    def __init__(self, gm, capture_node_names: List[str]):
        self.gm = gm
        self.capture_node_names = set(capture_node_names)
        self.captured = {}

    def run(self, *args, **kwargs):
        """Execute graph and capture specified nodes."""
        from torch.fx.interpreter import Interpreter

        interp = Interpreter(self.gm)

        # Store original run_node
        original_run_node = interp.run_node

        def capturing_run_node(n):
            result = original_run_node(n)
            if n.name in self.capture_node_names:
                if isinstance(result, torch.Tensor):
                    self.captured[n.name] = result.clone().detach()
                else:
                    self.captured[n.name] = result
            return result

        interp.run_node = capturing_run_node
        output = interp.run(*args, **kwargs)
        return output


def _compare_tensor(
    name: str, hf_tensor: torch.Tensor, ad_tensor: torch.Tensor, threshold: float = 0.01
):
    """Helper to compare two tensors and print results."""
    # Handle layout differences (HF uses BNSD, AD may use different layout)
    if hf_tensor.shape != ad_tensor.shape:
        print(f"[{name}] Shape mismatch: HF {hf_tensor.shape} vs AD {ad_tensor.shape}")
        # Try to transpose if shapes are permuted
        if sorted(hf_tensor.shape) == sorted(ad_tensor.shape):
            print("  -> Shapes contain same dims, may need transpose")
        return False

    diff = (hf_tensor.float() - ad_tensor.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    match = max_diff < threshold
    status = "MATCH" if match else "MISMATCH"
    print(f"[{name}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} -> {status}")

    if not match:
        # Print more details for mismatches
        print(
            f"  HF:  min={hf_tensor.min():.6f}, max={hf_tensor.max():.6f}, mean={hf_tensor.float().mean():.6f}"
        )
        print(
            f"  AD:  min={ad_tensor.min():.6f}, max={ad_tensor.max():.6f}, mean={ad_tensor.float().mean():.6f}"
        )

    return match


def _find_ad_node_by_pattern(ad_captured: Dict, pattern: str) -> Optional[str]:
    """Find an AD node by pattern matching.

    Args:
        ad_captured: Dict of captured AD tensors
        pattern: Regex pattern to match node names

    Returns:
        First matching node name, or None if not found
    """
    for node_name in ad_captured.keys():
        if re.search(pattern, node_name):
            return node_name
    return None


def compare_upstream_ops(hf_upstream: Dict, ad_captured: Dict, output_dir: Optional[str] = None):
    """Compare upstream ops (q_proj, k_proj, q_norm, k_norm, view, transpose) between HF and AD.

    This traces backwards from the RoPE slice to find where divergence begins.

    Args:
        hf_upstream: Dict of captured HF tensors
        ad_captured: Dict of captured AD tensors
        output_dir: Optional directory to save scatter plots
    """
    print("\n" + "=" * 70)
    print("[UPSTREAM OPS DEBUG] Comparing proj -> norm -> view -> transpose")
    print("=" * 70)

    if not ad_captured:
        print("  [SKIP] No AD nodes captured for upstream comparison")
        return True

    all_match = True

    # Mapping from HF keys to AD node patterns (regex)
    # Use patterns instead of hardcoded names to support different models
    comparisons = [
        # (name, hf_key, ad_pattern, description)
        ("q_proj", "q_proj_output", r".*layers_0_self_attn.*q_proj.*linear", "Q projection"),
        ("k_proj", "k_proj_output", r".*layers_0_self_attn.*k_proj.*linear", "K projection"),
        ("q_norm", "q_norm_output", r".*rms_norm.*[23]$", "Q RMSNorm"),
        ("k_norm", "k_norm_output", r".*rms_norm.*[01]$", "K RMSNorm"),
        ("q_view", None, r".*layers_0_self_attn.*view.*1", "Q after view"),
        ("k_view", None, r".*layers_0_self_attn.*view(?!.*1)", "K after view"),
        ("q_transpose", None, r".*layers_0_self_attn.*transpose.*1", "Q after transpose"),
        ("k_transpose", None, r".*layers_0_self_attn.*transpose.*2", "K after transpose"),
    ]

    for name, hf_key, ad_pattern, desc in comparisons:
        print(f"\n--- {name}: {desc} ---")

        # Find AD tensor by pattern matching
        ad_key = _find_ad_node_by_pattern(ad_captured, ad_pattern)
        if ad_key is None:
            print(f"  [AD] No node matching pattern '{ad_pattern}'")
            continue

        ad_tensor = ad_captured.get(ad_key)
        if ad_tensor is None:
            print(f"  [AD] {ad_key}: NOT CAPTURED")
            continue

        print(f"  [AD] {ad_key}: shape={ad_tensor.shape}, dtype={ad_tensor.dtype}")
        print(
            f"       stats: min={ad_tensor.min():.6f}, max={ad_tensor.max():.6f}, mean={ad_tensor.float().mean():.6f}"
        )

        # Get HF tensor (if available)
        if hf_key is not None and hf_key in hf_upstream:
            hf_tensor = hf_upstream[hf_key]
            print(f"  [HF] shape={hf_tensor.shape}, dtype={hf_tensor.dtype}")
            hf_mean = hf_tensor.float().mean()
            print(
                f"       stats: min={hf_tensor.min():.6f}, max={hf_tensor.max():.6f}, "
                f"mean={hf_mean:.6f}"
            )

            # Compare if shapes match (or can be reshaped)
            if hf_tensor.shape == ad_tensor.shape:
                if not _compare_tensor(name, hf_tensor, ad_tensor):
                    all_match = False
                # Save scatter plot
                if output_dir:
                    scatter_png = os.path.join(output_dir, f"upstream_{name}_scatter.png")
                    plot_tensor_scatter(hf_tensor, ad_tensor, scatter_png, f"Upstream: {name}")
            else:
                # Try to flatten and compare
                hf_flat = hf_tensor.flatten()
                ad_flat = ad_tensor.flatten()
                if hf_flat.shape == ad_flat.shape:
                    print(f"  [Note] Comparing flattened: {hf_flat.shape}")
                    if not _compare_tensor(f"{name}_flat", hf_flat, ad_flat):
                        all_match = False
                    # Save scatter plot for flattened comparison
                    if output_dir:
                        scatter_png = os.path.join(output_dir, f"upstream_{name}_flat_scatter.png")
                        plot_tensor_scatter(
                            hf_flat, ad_flat, scatter_png, f"Upstream: {name} (flat)"
                        )
                else:
                    print(f"  [Shape mismatch] HF {hf_tensor.shape} vs AD {ad_tensor.shape}")
        else:
            print(f"  [HF] {hf_key}: NOT CAPTURED (no hook available)")

    # Summary
    print("\n" + "-" * 70)
    if all_match:
        print("[UPSTREAM OPS] All ops MATCH - divergence is downstream")
    else:
        print("[UPSTREAM OPS] Ops MISMATCH - found divergence source above")
    print("-" * 70)

    return all_match


def compare_rope_inputs(hf_captured: Dict, ad_captured: Dict, output_dir: Optional[str] = None):
    """Compare all RoPE inputs between HF and AD to find divergence source.

    Compares:
    - position_ids
    - cos/sin after unsqueeze
    - Q/K rot parts before RoPE
    - Q/K pass parts

    Args:
        hf_captured: Dict of captured HF tensors
        ad_captured: Dict of captured AD tensors
        output_dir: Optional directory to save scatter plots
    """
    print("\n" + "=" * 70)
    print("[RoPE INPUTS DEBUG] Comparing intermediate values BEFORE RoPE")
    print("=" * 70)

    all_match = True

    # 1. Compare position_ids
    print("\n--- Position IDs ---")
    if "position_ids" in hf_captured and "position_ids" in ad_captured:
        hf_pos = hf_captured["position_ids"]
        ad_pos = ad_captured["position_ids"]
        print(f"[HF] position_ids: shape={hf_pos.shape}, dtype={hf_pos.dtype}")
        print(f"     values: {hf_pos.flatten()[:10].tolist()}...")
        print(f"[AD] position_ids: shape={ad_pos.shape}, dtype={ad_pos.dtype}")
        print(f"     values: {ad_pos.flatten()[:10].tolist()}...")

        if hf_pos.shape == ad_pos.shape:
            if torch.equal(hf_pos, ad_pos):
                print("[position_ids] -> MATCH")
            else:
                print("[position_ids] -> MISMATCH")
                all_match = False
        else:
            print(f"[position_ids] Shape mismatch: HF {hf_pos.shape} vs AD {ad_pos.shape}")
            all_match = False
    else:
        print(
            f"[position_ids] HF captured: {'position_ids' in hf_captured}, AD captured: {'position_ids' in ad_captured}"
        )

    # 2. Compare cos/sin after unsqueeze
    print("\n--- Cos/Sin after unsqueeze ---")
    # HF cos/sin are already captured with unsqueeze
    if "cos_unsqueezed" in hf_captured:
        hf_cos = hf_captured["cos_unsqueezed"]
        print(f"[HF] cos_unsqueezed: shape={hf_cos.shape}, dtype={hf_cos.dtype}")

        # Find AD cos unsqueeze node by pattern
        ad_cos_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*unsqueeze.*[3]")
        if ad_cos_key and ad_cos_key in ad_captured:
            ad_cos = ad_captured[ad_cos_key]
            print(f"[AD] cos_unsqueezed ({ad_cos_key}): shape={ad_cos.shape}, dtype={ad_cos.dtype}")

            # Note: shapes may differ in batch/head dims
            if hf_cos.shape[-1] == ad_cos.shape[-1]:  # at least head_dim matches
                # Compare the actual cos values (may need to broadcast/squeeze)
                hf_cos_flat = hf_cos.squeeze()
                ad_cos_flat = ad_cos.squeeze()
                if hf_cos_flat.shape == ad_cos_flat.shape:
                    if not _compare_tensor("cos", hf_cos_flat, ad_cos_flat):
                        all_match = False
                    # Save scatter plot for cos
                    if output_dir:
                        scatter_png = os.path.join(output_dir, "rope_cos_scatter.png")
                        plot_tensor_scatter(hf_cos_flat, ad_cos_flat, scatter_png, "RoPE: cos")
                else:
                    print(
                        f"  [cos] After squeeze: HF {hf_cos_flat.shape} vs AD {ad_cos_flat.shape}"
                    )
        else:
            print("  [AD] No cos unsqueeze node found")

    if "sin_unsqueezed" in hf_captured:
        hf_sin = hf_captured["sin_unsqueezed"]
        print(f"[HF] sin_unsqueezed: shape={hf_sin.shape}, dtype={hf_sin.dtype}")

        ad_sin_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*unsqueeze.*[4]")
        if ad_sin_key and ad_sin_key in ad_captured:
            ad_sin = ad_captured[ad_sin_key]
            print(f"[AD] sin_unsqueezed ({ad_sin_key}): shape={ad_sin.shape}, dtype={ad_sin.dtype}")

            hf_sin_flat = hf_sin.squeeze()
            ad_sin_flat = ad_sin.squeeze()
            if hf_sin_flat.shape == ad_sin_flat.shape:
                if not _compare_tensor("sin", hf_sin_flat, ad_sin_flat):
                    all_match = False
                # Save scatter plot for sin
                if output_dir:
                    scatter_png = os.path.join(output_dir, "rope_sin_scatter.png")
                    plot_tensor_scatter(hf_sin_flat, ad_sin_flat, scatter_png, "RoPE: sin")
            else:
                print(f"  [sin] After squeeze: HF {hf_sin_flat.shape} vs AD {ad_sin_flat.shape}")
        else:
            print("  [AD] No sin unsqueeze node found")

    # 3. Compare Q/K rot parts BEFORE RoPE
    print("\n--- Q/K rot parts BEFORE RoPE ---")

    # Q rot part: HF is BNSD, AD may use different layouts
    if "q_rot_part" in hf_captured:
        hf_q_rot = hf_captured["q_rot_part"]
        print(f"[HF] q_rot_part: shape={hf_q_rot.shape}, dtype={hf_q_rot.dtype}")

        ad_q_rot_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*slice.*[3]")
        if ad_q_rot_key and ad_q_rot_key in ad_captured:
            ad_q_rot = ad_captured[ad_q_rot_key]
            print(
                f"[AD] q_rot_part ({ad_q_rot_key}): shape={ad_q_rot.shape}, dtype={ad_q_rot.dtype}"
            )

            if not _compare_tensor("q_rot_part", hf_q_rot, ad_q_rot):
                all_match = False
            # Save scatter plot for q_rot_part
            if output_dir and hf_q_rot.shape == ad_q_rot.shape:
                scatter_png = os.path.join(output_dir, "rope_q_rot_part_scatter.png")
                plot_tensor_scatter(hf_q_rot, ad_q_rot, scatter_png, "RoPE: Q rot part")
        else:
            print("  [AD] No Q rot slice node found")

    if "k_rot_part" in hf_captured:
        hf_k_rot = hf_captured["k_rot_part"]
        print(f"[HF] k_rot_part: shape={hf_k_rot.shape}, dtype={hf_k_rot.dtype}")

        ad_k_rot_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*slice.*[5]")
        if ad_k_rot_key and ad_k_rot_key in ad_captured:
            ad_k_rot = ad_captured[ad_k_rot_key]
            print(
                f"[AD] k_rot_part ({ad_k_rot_key}): shape={ad_k_rot.shape}, dtype={ad_k_rot.dtype}"
            )

            if not _compare_tensor("k_rot_part", hf_k_rot, ad_k_rot):
                all_match = False
            # Save scatter plot for k_rot_part
            if output_dir and hf_k_rot.shape == ad_k_rot.shape:
                scatter_png = os.path.join(output_dir, "rope_k_rot_part_scatter.png")
                plot_tensor_scatter(hf_k_rot, ad_k_rot, scatter_png, "RoPE: K rot part")
        else:
            print("  [AD] No K rot slice node found")

    # 4. Compare Q/K pass parts
    print("\n--- Q/K pass parts (should be unchanged) ---")

    if "q_pass_part" in hf_captured:
        hf_q_pass = hf_captured["q_pass_part"]
        print(f"[HF] q_pass_part: shape={hf_q_pass.shape}, dtype={hf_q_pass.dtype}")

        ad_q_pass_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*slice.*[4]")
        if ad_q_pass_key and ad_q_pass_key in ad_captured:
            ad_q_pass = ad_captured[ad_q_pass_key]
            print(
                f"[AD] q_pass_part ({ad_q_pass_key}): shape={ad_q_pass.shape}, dtype={ad_q_pass.dtype}"
            )

            if not _compare_tensor("q_pass_part", hf_q_pass, ad_q_pass):
                all_match = False
            # Save scatter plot for q_pass_part
            if output_dir and hf_q_pass.shape == ad_q_pass.shape:
                scatter_png = os.path.join(output_dir, "rope_q_pass_part_scatter.png")
                plot_tensor_scatter(hf_q_pass, ad_q_pass, scatter_png, "RoPE: Q pass part")
        else:
            print("  [AD] No Q pass slice node found")

    if "k_pass_part" in hf_captured:
        hf_k_pass = hf_captured["k_pass_part"]
        print(f"[HF] k_pass_part: shape={hf_k_pass.shape}, dtype={hf_k_pass.dtype}")

        ad_k_pass_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*slice.*[6]")
        if ad_k_pass_key and ad_k_pass_key in ad_captured:
            ad_k_pass = ad_captured[ad_k_pass_key]
            print(
                f"[AD] k_pass_part ({ad_k_pass_key}): shape={ad_k_pass.shape}, dtype={ad_k_pass.dtype}"
            )

            if not _compare_tensor("k_pass_part", hf_k_pass, ad_k_pass):
                all_match = False
            # Save scatter plot for k_pass_part
            if output_dir and hf_k_pass.shape == ad_k_pass.shape:
                scatter_png = os.path.join(output_dir, "rope_k_pass_part_scatter.png")
                plot_tensor_scatter(hf_k_pass, ad_k_pass, scatter_png, "RoPE: K pass part")
        else:
            print("  [AD] No K pass slice node found")

    # Summary
    print("\n" + "-" * 70)
    if all_match:
        print("[RoPE INPUTS] All inputs MATCH - divergence is in RoPE computation itself")
    else:
        print("[RoPE INPUTS] Inputs MISMATCH - divergence source found in inputs above")
    print("-" * 70)

    return all_match


def compare_rope_outputs(hf_captured: Dict, ad_captured: Dict, output_dir: Optional[str] = None):
    """Compare Q/K after RoPE between HF and AD.

    Args:
        hf_captured: Dict of captured HF tensors
        ad_captured: Dict of captured AD tensors
        output_dir: Optional directory to save scatter plots
    """
    print("\n" + "=" * 70)
    print("[RoPE COMPARISON] Q/K after partial RoPE")
    print("=" * 70)

    # HF outputs
    if "q_after_rope" in hf_captured:
        hf_q = hf_captured["q_after_rope"]
        print(f"[HF] Q after RoPE: shape={hf_q.shape}, dtype={hf_q.dtype}")
        print(f"      stats: min={hf_q.min():.6f}, max={hf_q.max():.6f}, mean={hf_q.mean():.6f}")
    else:
        print("[HF] Q after RoPE: NOT CAPTURED")
        return

    if "k_after_rope" in hf_captured:
        hf_k = hf_captured["k_after_rope"]
        print(f"[HF] K after RoPE: shape={hf_k.shape}, dtype={hf_k.dtype}")
        print(f"      stats: min={hf_k.min():.6f}, max={hf_k.max():.6f}, mean={hf_k.mean():.6f}")
    else:
        print("[HF] K after RoPE: NOT CAPTURED")
        return

    # AD outputs - look for cat nodes that represent Q/K after RoPE
    # Use pattern matching to find the concatenated Q/K outputs
    ad_q_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*cat.*[3]")
    ad_k_key = _find_ad_node_by_pattern(ad_captured, r".*layers_0_self_attn.*cat.*[4]")

    if ad_q_key and ad_q_key in ad_captured:
        ad_q = ad_captured[ad_q_key]
        print(f"[AD] Q after RoPE ({ad_q_key}): shape={ad_q.shape}, dtype={ad_q.dtype}")
        print(f"      stats: min={ad_q.min():.6f}, max={ad_q.max():.6f}, mean={ad_q.mean():.6f}")

        # Compare Q
        if hf_q.shape == ad_q.shape:
            q_diff = (hf_q - ad_q).abs()
            print(f"[Q DIFF] max_diff={q_diff.max():.6f}, mean_diff={q_diff.mean():.6f}")
            if q_diff.max() < 0.01:
                print("  -> Q MATCH")
            else:
                print("  -> Q MISMATCH")
            # Save scatter plot for Q after RoPE
            if output_dir:
                scatter_png = os.path.join(output_dir, "rope_q_after_scatter.png")
                plot_tensor_scatter(hf_q, ad_q, scatter_png, "RoPE: Q after")
        else:
            print(f"  -> Shape mismatch: HF {hf_q.shape} vs AD {ad_q.shape}")
    else:
        print(f"[AD] Q after RoPE: NOT CAPTURED (looked for 'cat_3' in {list(ad_captured.keys())})")

    if ad_k_key and ad_k_key in ad_captured:
        ad_k = ad_captured[ad_k_key]
        print(f"[AD] K after RoPE ({ad_k_key}): shape={ad_k.shape}, dtype={ad_k.dtype}")
        print(f"      stats: min={ad_k.min():.6f}, max={ad_k.max():.6f}, mean={ad_k.mean():.6f}")

        # Compare K
        if hf_k.shape == ad_k.shape:
            k_diff = (hf_k - ad_k).abs()
            print(f"[K DIFF] max_diff={k_diff.max():.6f}, mean_diff={k_diff.mean():.6f}")
            if k_diff.max() < 0.01:
                print("  -> K MATCH")
            else:
                print("  -> K MISMATCH")
            # Save scatter plot for K after RoPE
            if output_dir:
                scatter_png = os.path.join(output_dir, "rope_k_after_scatter.png")
                plot_tensor_scatter(hf_k, ad_k, scatter_png, "RoPE: K after")
        else:
            print(f"  -> Shape mismatch: HF {hf_k.shape} vs AD {ad_k.shape}")
    else:
        print(f"[AD] K after RoPE: NOT CAPTURED (looked for 'cat_4' in {list(ad_captured.keys())})")


# ============================================================================
# Inline Comparison Entry Point (for use at breakpoint)
# ============================================================================


def _detect_model_architecture(hf_model, ad_gm) -> str:
    """Detect model architecture for specialized debug handling.

    Returns:
        "glm" - GLM-4.7-Flash with MLA and fused RoPE (torch_mla, torch_rope_with_qk_interleaving)
        "minimax" - MiniMax with partial RoPE (slice/cat patterns, block_sparse_moe)
        "llama" - Standard Llama-style attention
        "unknown" - Fallback
    """
    # Check HF model config
    config = getattr(hf_model, "config", None)
    if config is not None:
        arch = getattr(config, "architectures", [""])[0] if hasattr(config, "architectures") else ""
        model_type = getattr(config, "model_type", "")

        # GLM-4.7-Flash detection
        if "Glm4MoeLite" in arch or "glm4_moe_lite" in model_type:
            return "glm"

        # MiniMax detection
        if "MiniMax" in arch or "minimax" in model_type.lower():
            return "minimax"

    # Check AD graph for custom ops
    node_names = [n.name for n in ad_gm.graph.nodes]
    node_str = " ".join(node_names)

    # GLM uses these custom ops
    if "torch_mla" in node_str and "torch_rope_with_qk_interleaving" in node_str:
        return "glm"

    # MiniMax uses block_sparse_moe and has slice/cat patterns for partial RoPE
    if "block_sparse_moe" in node_str:
        # Check for partial RoPE pattern (slice -> cat)
        if "slice" in node_str and "cat" in node_str:
            return "minimax"

    # Llama-style
    if "self_attn" in node_str and "rotary" in node_str.lower():
        return "llama"

    return "unknown"


def run_comparison(mod, cm, factory, device: str = "cuda", output_dir: Optional[str] = None):
    """Run HF vs AD comparison inline with live GraphModule.

    Usage at breakpoint in optimizer.py:
        from tensorrt_llm._torch.auto_deploy.utils.graph_debug_compare import run_comparison
        run_comparison(mod, cm, self.factory, output_dir="debug_scatter_plots")

    Args:
        mod: The transformed AD GraphModule
        cm: CachedSequenceInterface with named_args
        factory: ModelFactory with checkpoint_path and config
        device: Device to run on
        output_dir: Optional directory to save scatter plots for each block comparison
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Scatter plots will be saved to: {output_dir}")
    # Get model path and num_layers from factory
    model_path = getattr(factory, "checkpoint_path", None) or getattr(factory, "model", None)
    num_layers = (
        factory.model_kwargs.get("num_hidden_layers", 1) if hasattr(factory, "model_kwargs") else 1
    )

    if model_path is None:
        print("ERROR: Could not get model path from factory")
        return

    print("=" * 80)
    print("INLINE COMPARISON: AD vs HF")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Num layers: {num_layers}")

    # Extract metadata from live model
    print("\nExtracting metadata from AD model...")
    ad_metadata = extract_graph_metadata(mod)

    # Get inputs from cm
    input_ids = cm.named_args.get("input_ids")
    if input_ids is None:
        print("ERROR: No input_ids in cm.named_args")
        return

    # Load HF model
    print(f"\nLoading HuggingFace model: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_hidden_layers = num_layers
    config.use_cache = False

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hf_model.eval()

    # ========================================================================
    # Detect model architecture for specialized debug
    # ========================================================================
    model_arch = _detect_model_architecture(hf_model, mod)
    print(f"\n[Model Architecture] Detected: {model_arch}")

    # ========================================================================
    # RoPE Comparison: Only for models with unfused RoPE (MiniMax-style)
    # GLM-4.7-Flash uses fused torch_rope_with_qk_interleaving custom op
    # ========================================================================
    if model_arch == "minimax":
        print("\n" + "=" * 70)
        print("[RoPE DEBUG] Capturing upstream ops and Q/K after RoPE from both HF and AD")
        print("=" * 70)

        # 1a. Capture HF upstream ops (q_proj, k_proj, q_norm, k_norm)
        print("\n[HF] Running forward with upstream hooks (q_proj, k_proj, q_norm, k_norm)...")
        hf_upstream_captured = capture_hf_upstream_ops(hf_model, input_ids, device=device)
        print(f"  Captured: {list(hf_upstream_captured.keys())}")

        # 1b. Capture HF Q/K after RoPE
        print("\n[HF] Running forward with RoPE capture...")
        hf_rope_captured = capture_hf_rope_outputs(hf_model, input_ids, device=device)

        # 2. Capture AD intermediate nodes for RoPE comparison
        print("\n[AD] Running forward with node capture...")
        # Dynamically discover RoPE-related nodes from the graph
        rope_capture_nodes = _discover_rope_nodes(mod)

        # Prepare AD inputs
        ad_inputs = []
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                arg_name = node.name
                if arg_name in cm.named_args:
                    val = cm.named_args[arg_name]
                    if isinstance(val, torch.Tensor):
                        ad_inputs.append(val.to(device))
                    else:
                        ad_inputs.append(val)
                else:
                    ad_inputs.append(None)

        # Run AD with capturing
        ad_capturer = CapturingInterpreter(mod, rope_capture_nodes)
        with torch.inference_mode():
            mod.to(device)
            try:
                _ = ad_capturer.run(*ad_inputs)
            except Exception as e:
                print(f"[AD] Forward failed: {e}")

        # 3. Compare upstream ops first (proj -> norm -> view -> transpose)
        compare_upstream_ops(hf_upstream_captured, ad_capturer.captured, output_dir=output_dir)

        # 4. Compare RoPE inputs (to find divergence source)
        compare_rope_inputs(hf_rope_captured, ad_capturer.captured, output_dir=output_dir)

        # 5. Compare RoPE outputs
        compare_rope_outputs(hf_rope_captured, ad_capturer.captured, output_dir=output_dir)
    else:
        print("\n" + "=" * 70)
        print(f"[RoPE DEBUG] Skipped - {model_arch} uses fused RoPE custom ops")
        print("  GLM-4.7-Flash: torch_rope_with_qk_interleaving + torch_mla")
        print("  Use Stage 1 coarse comparison with debug markers instead.")
        print("=" * 70)

    # ========================================================================
    # Original Coarse Comparison
    # ========================================================================

    # Run comparison
    all_passed, failing_module, results = stage1_coarse_comparison(
        hf_model,
        mod,
        ad_metadata,
        input_ids,
        ad_named_args=cm.named_args,
        device=device,
        output_dir=output_dir,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_passed:
        print("All module boundaries match between HF and AD!")
    else:
        print(f"Divergence detected in module: {failing_module}")

    return all_passed, failing_module, results


# ============================================================================
# Main (for standalone usage)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Module-Level Graph Comparison Tool")
    parser.add_argument(
        "--debug-dir",
        type=str,
        required=True,
        help="Directory containing debug dumps (from AD_DUMP_DEBUG_DIR)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="cyankiwi/MiniMax-M2-BF16",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save scatter plots for each block comparison",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Module-Level Graph Comparison: AD vs HF")
    print("=" * 80)
    print(f"Debug dir: {debug_dir}")
    print(f"HF model: {args.hf_model}")
    print(f"Num layers: {args.num_layers}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir or '(not saving scatter plots)'}")

    # Load debug artifacts
    print("\nLoading debug artifacts...")
    final_gm, final_metadata, inputs = load_debug_artifacts(debug_dir, "final")

    if final_gm is None:
        print("ERROR: Could not load final GraphModule")
        return 1

    if final_metadata is None or not final_metadata:
        print("ERROR: Could not load final metadata")
        return 1

    # Load HF model
    print("\nLoading HuggingFace model...")
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    config.num_hidden_layers = args.num_layers
    config.use_cache = False

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hf_model.eval()

    # Prepare input
    if inputs is not None and "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        ad_named_args = {"input_ids": input_ids}
    else:
        # Create dummy input
        print("Using dummy input (no saved inputs found)")
        input_ids = torch.randint(0, 1000, (1, 8))
        ad_named_args = {"input_ids": input_ids}

    # ========================================================================
    # Module-Level Comparison
    # ========================================================================
    all_passed, failing_module, _ = stage1_coarse_comparison(
        hf_model,
        final_gm,
        final_metadata,
        input_ids,
        ad_named_args=ad_named_args,
        device=args.device,
        output_dir=args.output_dir,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_passed:
        print("All module boundaries match between HF and AD!")
    else:
        print(f"Divergence detected in module: {failing_module}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
