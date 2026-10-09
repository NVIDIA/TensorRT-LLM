# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from pathlib import Path

import torch
from test_common.llm_data import hf_id_to_local_model_dir

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleDrafterForCausalLM
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory


def _load_valid_safetensors_index(index_path: Path):
    """Load a safetensors index JSON, skipping invalid and Git-LFS pointer files."""
    if not index_path.exists():
        return None

    try:
        index_text = index_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if index_text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    try:
        index = json.loads(index_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(index, dict):
        return None

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return None

    return index


def _analyze_mtp_weight_loading(model_path: Path, model):
    """Analyze weight loading for MTP models with safetensors index.

    MTP checkpoints use multiple safetensors files with an index. This function
    loads the checkpoint keys from the index and applies the model's
    _checkpoint_conversion_mapping to determine which keys will be loaded.

    Args:
        model_path: Path to the MTP model directory
        model: The instantiated model

    Returns:
        Tuple of (loaded_keys, missing_keys, unexpected_keys)
    """
    # Load checkpoint keys from safetensors index
    index_path = model_path / "model.safetensors.index.json"
    index = _load_valid_safetensors_index(index_path)
    if index is None:
        raise ValueError(
            "Expected a valid safetensors index JSON. "
            f"Path was missing, malformed, or a Git-LFS pointer: {index_path}"
        )

    # Get MTP-specific checkpoint keys (those starting with "mtp.")
    checkpoint_keys_original = [k for k in index["weight_map"].keys() if k.startswith("mtp.")]
    if not checkpoint_keys_original:
        raise ValueError(f"No mtp.* keys found in safetensors index: {index_path}")

    # Apply _checkpoint_conversion_mapping (same logic as hf.py _remap_param_names_load_hook)
    conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    checkpoint_keys_remapped = []

    for key in checkpoint_keys_original:
        new_key = key
        if conversion_mapping:
            for pattern, replacement in conversion_mapping.items():
                new_key = re.sub(pattern, replacement, new_key)
        checkpoint_keys_remapped.append(new_key)

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint_keys_remapped)

    # Calculate differences
    loaded_keys = checkpoint_keys & model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - model_keys

    return loaded_keys, missing_in_checkpoint, unexpected_in_checkpoint


def test_nemotron_mtp_model_with_weights():
    """Test NemotronH MTP model weight loading using EagleDrafterFactory.

    This test verifies that:
    1. EagleDrafterFactory can create EagleDrafterForCausalLM with NemotronH layers
    2. Weights are correctly loaded with the mtp.* -> model.* key mapping
    3. All expected model parameters are loaded from the MTP checkpoint

    The MTP model uses a checkpoint that contains both backbone.* and mtp.* keys.
    Only mtp.* keys are loaded. Shared parameters (embed_tokens, lm_head) are NOT
    created in the model (load_embedding_from_target=True, load_lm_head_from_target=True),
    so they don't appear as missing keys. They are shared from the target model at runtime.
    """
    print("\n" + "=" * 80)
    print("Test: NemotronH MTP model weight loading (via EagleDrafterFactory)")
    print("=" * 80)

    mtp_model_name = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    mtp_model_path = hf_id_to_local_model_dir(mtp_model_name)
    mtp_path = Path(mtp_model_path)
    index_path = mtp_path / "model.safetensors.index.json"

    # Check for a valid index JSON and verify it has mtp.* keys.
    index = _load_valid_safetensors_index(index_path)
    assert index is not None, (
        "Expected a valid safetensors index JSON. "
        f"Path was missing, malformed, or a Git-LFS pointer: {index_path}"
    )
    mtp_source_keys = {k for k in index["weight_map"].keys() if k.startswith("mtp.")}
    assert mtp_source_keys, f"Expected at least one mtp.* key in {index_path}"

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create factory - use EagleDrafterFactory for NemotronH MTP
    print("Creating EagleDrafterFactory...")
    factory = EagleDrafterFactory(
        model=mtp_model_path,
        skip_loading_weights=False,
    )

    # Build model using factory
    print("Building model via factory.build_model('meta')...")
    model = factory.build_model("meta")
    print(f"Model type: {type(model).__name__}")
    print(f"Model config type: {type(model.config).__name__}")

    # Verify model type is EagleDrafterForCausalLM
    assert isinstance(model, EagleDrafterForCausalLM), (
        f"Expected EagleDrafterForCausalLM, got {type(model).__name__}"
    )

    # Analyze weight loading
    print("\n--- Weight Loading Analysis ---")
    loaded_keys, missing_keys, unexpected_keys = _analyze_mtp_weight_loading(mtp_path, model)

    print(f"Total model parameters: {len(loaded_keys) + len(missing_keys)}")
    print(f"Total MTP checkpoint keys: {len(loaded_keys) + len(unexpected_keys)}")
    print(f"✅ Weights to be loaded: {len(loaded_keys)}")
    print(f"⚠️  Missing in checkpoint (should be 0): {len(missing_keys)}")
    print(f"⚠️  Unexpected in checkpoint (should be 0): {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing keys (expected - shared from target model):")
        for key in sorted(missing_keys):
            if "embed_tokens" in key:
                print(f"  - {key} (shared embedding from target)")
            elif "lm_head" in key:
                print(f"  - {key} (shared lm_head from target)")
            else:
                print(f"  - {key}")

    if unexpected_keys:
        print("\nUnexpected keys (should not happen):")
        for key in sorted(unexpected_keys):
            print(f"  - {key}")

    print("--- End Weight Analysis ---\n")

    # Verify expected missing and unexpected keys
    # MTP checkpoint does NOT contain embed_tokens or lm_head, but that's OK because:
    # - embed_tokens: shared from target model (load_embedding_from_target=True → model doesn't create it)
    # - lm_head: shared from target model (load_lm_head_from_target=True → model doesn't create it)
    # Since neither parameter is created in the model, they don't appear as missing keys.
    # Note: For NemotronH, layers_handle_final_norm=True, so the wrapper doesn't create self.norm.
    # The final norm is inside the layers (final_layernorm), which IS in the checkpoint.
    expected_missing_keys = set()  # All model params are loaded; shared params aren't created
    expected_unexpected_keys = set()  # All checkpoint keys should be used

    assert missing_keys == expected_missing_keys, (
        f"Unexpected missing keys.\n"
        f"Expected: {expected_missing_keys}\n"
        f"Got: {missing_keys}\n"
        f"Extra missing: {missing_keys - expected_missing_keys}\n"
        f"Not missing (but expected): {expected_missing_keys - missing_keys}"
    )

    assert unexpected_keys == expected_unexpected_keys, (
        f"Unexpected keys in checkpoint.\n"
        f"Expected: {expected_unexpected_keys}\n"
        f"Got: {unexpected_keys}\n"
        f"Extra unexpected: {unexpected_keys - expected_unexpected_keys}"
    )

    print("✅ Weight loading analysis matches expected missing/unexpected keys!")

    # Load weights using factory
    print("Loading weights via factory.load_or_random_init()...")
    factory.load_or_random_init(model, device, disable_preload=True)
    print("Weights loaded successfully via factory interface!")

    model.eval()
    print("✅ NemotronH MTP model created and weights loaded successfully!")
