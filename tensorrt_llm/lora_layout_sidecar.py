# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parser for the optional ``lora_layout.json`` adapter sidecar.

The sidecar lets routed-expert Mixture-of-Experts (MoE) LoRA adapters declare
which side (``A`` or ``B``) of each module is shared across experts -- the
"outer" / residual-stream side. The fused-MoE op honors these flags via
``LoraParams::*_shared_a/b``, zero-offsetting the per-expert pointer
arithmetic in the kernel's ``setupLoraWorkspace``.

In this release the on-disk layout is unchanged: the loader still replicates
the shared tensor across ``num_experts`` before packing it into the C++ LoRA
cache (the cache row size is determined by ``LoraModule`` config, not by the
packed blob). The sidecar establishes the convention end-to-end, propagating
the flags from disk through the loader and per-request assembler into the
fused-MoE op. A follow-up will switch the loader and C++ cache to native
unreplicated storage for actual GPU-memory savings.

Sidecar schema (all fields optional; missing modules default to per-expert):

.. code-block:: json

    {
      "version": 1,
      "moe_shared_outer": {
        "moe_h_to_4h": {"shared_side": "A"},
        "moe_gate":    {"shared_side": "A"},
        "moe_4h_to_h": {"shared_side": "B"}
      }
    }
"""
import json
import os
from typing import Dict, Optional

# Shared-side value for a single module: "A", "B", or None for per-expert.
SharedSide = Optional[str]

# Mapping from the TRT-LLM LoRA module name to the kernel-side module
# prefix the fused-MoE op uses. Matches the slot mapping in
# ``fused_moe_cutlass._extract_moe_lora_tensors``:
#   moe_gate    -> fc1   (up projection)
#   moe_4h_to_h -> fc2   (down projection)
#   moe_h_to_4h -> gated (gate-side up projection in SwiGLU)
MODULE_TO_KERNEL_PREFIX: Dict[str, str] = {
    "moe_gate": "fc1",
    "moe_4h_to_h": "fc2",
    "moe_h_to_4h": "gated",
}

# Per-side boolean flags accepted by ``torch.ops.trtllm.fused_moe``.
KERNEL_FLAG_KEYS = (
    "fc1_shared_a",
    "fc1_shared_b",
    "fc2_shared_a",
    "fc2_shared_b",
    "gated_shared_a",
    "gated_shared_b",
)

LORA_LAYOUT_FILENAME = "lora_layout.json"


class LoraLayoutError(ValueError):
    """Raised when ``lora_layout.json`` is present but malformed."""


def all_false_flags() -> Dict[str, bool]:
    """Return a fresh dict of the six kernel flag keys, all set to False.

    Useful as a default for adapters with no ``lora_layout.json`` sidecar.
    """
    return {key: False for key in KERNEL_FLAG_KEYS}


def load_lora_layout_sidecar(
        model_dir: str) -> Optional[Dict[str, SharedSide]]:
    """Read ``<model_dir>/lora_layout.json`` if it exists.

    Args:
        model_dir: Directory containing the adapter (alongside
            ``adapter_config.json`` / ``adapter_model.*``).

    Returns:
        A dict mapping each present LoRA module name to its
        ``shared_side`` ("A" or "B" or ``None``), or ``None`` when the
        sidecar file is absent.

    Raises:
        LoraLayoutError: if the file is present but cannot be parsed
        or violates the schema.
    """
    path = os.path.join(model_dir, LORA_LAYOUT_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise LoraLayoutError(f"Failed to parse {path}: {e}") from e
    except OSError as e:
        raise LoraLayoutError(f"Failed to read {path}: {e}") from e
    return parse_lora_layout(data, source=path)


def parse_lora_layout(data: Dict,
                      source: str = "<dict>") -> Dict[str, SharedSide]:
    """Validate a parsed sidecar dict and return its per-module map.

    Unknown top-level keys are tolerated for forward compatibility, but
    unknown module names under ``moe_shared_outer`` raise so that typos
    or stale conventions surface early.

    Args:
        data: The decoded JSON object.
        source: A diagnostic identifier used in error messages (file path
            or ``"<dict>"`` when parsing in-memory).

    Returns:
        Dict mapping each present module name to its shared side.

    Raises:
        LoraLayoutError: on schema violations.
    """
    if not isinstance(data, dict):
        raise LoraLayoutError(
            f"{source}: top-level value must be a JSON object, got "
            f"{type(data).__name__}.")
    version = data.get("version", 1)
    if version != 1:
        raise LoraLayoutError(
            f"{source}: unsupported lora_layout version {version!r}; only "
            f"version 1 is recognized.")
    moe = data.get("moe_shared_outer", {})
    if not isinstance(moe, dict):
        raise LoraLayoutError(
            f"{source}: 'moe_shared_outer' must be a JSON object.")
    result: Dict[str, SharedSide] = {}
    for module_name, spec in moe.items():
        if module_name not in MODULE_TO_KERNEL_PREFIX:
            raise LoraLayoutError(
                f"{source}: unknown MoE LoRA module {module_name!r}; "
                f"supported: {sorted(MODULE_TO_KERNEL_PREFIX)}.")
        if not isinstance(spec, dict):
            raise LoraLayoutError(
                f"{source}: 'moe_shared_outer[{module_name}]' must be a "
                "JSON object.")
        side = spec.get("shared_side")
        if side not in ("A", "B", None):
            raise LoraLayoutError(
                f"{source}: 'moe_shared_outer[{module_name}].shared_side' "
                f"must be 'A', 'B' or null; got {side!r}.")
        result[module_name] = side
    return result


def layout_to_kernel_flags(
        layout: Optional[Dict[str, SharedSide]]) -> Dict[str, bool]:
    """Translate a per-module shared-side map into the six-bool dict
    consumed by the fused-MoE op via ``lora_params["moe_shared_flags"]``.

    Modules missing from ``layout`` (or an entirely missing layout) yield
    all-False, preserving the default per-expert offset behavior.

    Args:
        layout: Per-module shared-side mapping (output of
            :func:`load_lora_layout_sidecar` or
            :func:`parse_lora_layout`), or ``None`` when no sidecar was
            provided.

    Returns:
        Dict with the six keys defined by :data:`KERNEL_FLAG_KEYS`,
        each ``True`` iff that side of that module is shared.
    """
    flags = all_false_flags()
    if not layout:
        return flags
    for module_name, side in layout.items():
        if side is None:
            continue
        prefix = MODULE_TO_KERNEL_PREFIX[module_name]
        if side == "A":
            flags[f"{prefix}_shared_a"] = True
        elif side == "B":
            flags[f"{prefix}_shared_b"] = True
    return flags
