# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Reading LoRA checkpoints off disk -- the HF and NeMo formats.

This half knows nothing about ``LoraManager``; everything here turns files into
plain weights and metadata.
"""

import io
import json
import logging
import re
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import yaml

from tensorrt_llm.models.convert_utils import get_model_path, load_state_dict

from .ownership import DecoderLoraOwner

logger = logging.getLogger(__name__)


NEMO_SUPPORTED_LORA_MODULES = {"attn_qkv"}


def _validate_owner_prefix(
    key: str, prefix: str, owner: DecoderLoraOwner, source: str, *, layered: bool = True
) -> None:
    if source == "HF":
        prefix = prefix.removeprefix("base_model.model.")
        if prefix == "base_model.model":
            prefix = ""
        prefixes = owner.hf_prefixes
        if not layered:
            prefixes = tuple(p.rpartition(".")[0] for p in prefixes)
    else:
        prefixes = owner.nemo_prefixes
    if prefix not in prefixes:
        raise ValueError(
            f"LoRA key '{key}' is outside selected owner 'decoder' "
            f"(module path '{owner.module_path or '<root>'}'): "
            f"unsupported {source} prefix '{prefix}'. "
            "Encoder/projector LoRA is not supported. Use a decoder-only adapter checkpoint."
        )


def _validate_unparsed_key_owner(key: str, owner: DecoderLoraOwner, source: str) -> None:
    normalized = key.removeprefix("base_model.model.") if source == "HF" else key
    prefixes = owner.hf_prefixes if source == "HF" else owner.nemo_prefixes
    if not any(normalized.startswith(f"{prefix}.") for prefix in prefixes):
        _validate_owner_prefix(key, normalized, owner, source)


def _check_duplicate_destination(
    destinations: dict, destination: tuple, key: str, owner: DecoderLoraOwner
) -> None:
    if destination in destinations:
        raise ValueError(
            f"LoRA keys '{destinations[destination]}' and '{key}' have a duplicate "
            f"destination {destination} for selected owner 'decoder' "
            f"(module path '{owner.module_path or '<root>'}')."
        )
    destinations[destination] = key


def get_all_nemo_lora_weights(
    lora_weights: Dict[str, torch.Tensor],
    *,
    owner: DecoderLoraOwner = DecoderLoraOwner(),
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract and organize NeMo LoRA weights by layer and direction.

    Args:
        lora_weights: Dictionary mapping weight keys to tensors from NeMo checkpoint
        owner: Decoder module and accepted checkpoint prefixes

    Returns:
        Dictionary mapping layer_idx -> {direction -> tensor} where direction is 'in' or 'out'

    Raises:
        KeyError: If unsupported keys are found or layer extraction fails
        ValueError: If a key belongs to another owner or duplicates a destination
    """
    layer_weights: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
    adapter_key = "self_attention.adapter_layer.lora_kqv_adapter"
    layer_pattern = re.compile(r"(.*\.layers)\.(\d+)\..*")
    destinations = {}
    for key, weights in lora_weights.items():
        _validate_unparsed_key_owner(key, owner, "NeMo")
        if adapter_key in key:
            m = layer_pattern.match(key)
            if m is None:
                if not key.endswith(("linear_in.weight", "linear_out.weight")):
                    continue
                raise KeyError(
                    f"Failed to extract layer index from key {key} using pattern {layer_pattern.pattern}"
                )
            _validate_owner_prefix(key, m.group(1), owner, "NeMo")
            if key.endswith("linear_in.weight"):
                inout = "in"
            elif key.endswith("linear_out.weight"):
                inout = "out"
            else:
                continue
            layer_idx = int(m.group(2))
            _check_duplicate_destination(
                destinations, ("attn_qkv", layer_idx, None, inout), key, owner
            )
            layer_weights[layer_idx][inout] = weights
        else:
            raise KeyError(f"unsupported key {key} from Nemo LoRA weights")
    return layer_weights


HF_LORA_PATTERN = re.compile(
    r"(.*)\.(\d+)\.(\w+)\.(\w+|\w+\.\w+|(\w+)\.(\d+)\.(\w+))\.(?:lora_(?:(A|B)\.weight|(magnitude)_vector)|weight_(m_wdecomp).weight)"
)


def iterate_hf_lora(
    iter_fn,
    lora_weights: Dict[str, torch.Tensor],
    hf_modules: Set[str],
    component: Optional[str] = None,
    *,
    owner: DecoderLoraOwner = DecoderLoraOwner(),
    hf_modules_to_trtllm_modules: Optional[Dict[str, str]] = None,
):
    """Iterate over HuggingFace LoRA weights and call iterator function for each weight.

    Args:
        iter_fn: Function to call for each weight with signature
        (layer_idx, hf_module, expert_idx, inout_or_mag, weights)
        lora_weights: Dictionary mapping weight keys to tensors from HF checkpoint
        hf_modules: Set of supported HF module names
        component: Optional component name to filter by (e.g., 'decoder')
        owner: Decoder module and accepted checkpoint prefixes
        hf_modules_to_trtllm_modules: Canonical destinations for detecting HF alias collisions

    Returns:
        Nested dictionary structure organizing the weights

    Raises:
        KeyError: If unsupported keys are found
        ValueError: If a key belongs to another owner or duplicates a destination
    """
    all_weights = defaultdict(lambda: defaultdict(dict))
    destinations = {}
    module_representations = {}
    pattern = HF_LORA_PATTERN
    for key, weights in lora_weights.items():
        m = pattern.fullmatch(key)
        if not m:
            saved_module = re.search(r"(?:^|\.)(?:lm_head|embed_tokens)(?:\.|$)", key)
            if saved_module is not None:
                _validate_owner_prefix(key, key[: saved_module.start()], owner, "HF", layered=False)
                continue
            _validate_unparsed_key_owner(key, owner, "HF")
            raise KeyError(f"unsupported key {key} from HF LoRA weights")
        _validate_owner_prefix(key, m.group(1), owner, "HF")
        layer_idx = int(m.group(2))
        expert_idx = m.group(6)
        if expert_idx is not None:
            expert_idx = int(expert_idx)
        is_moe = expert_idx is not None
        if is_moe:
            expert_name = m.group(5)
            module_name = m.group(7)
            hf_module = m.group(3) + "." + expert_name + "." + module_name
        else:
            module_name = m.group(4)
            hf_module = m.group(3) + "." + module_name
        if hf_module not in hf_modules:
            hf_module = module_name

            # If module_name contains dots (e.g., "shared_expert.down_proj"),
            # extract just the final component (e.g., "down_proj").
            # Skip this fallback for shared_expert modules to avoid
            # silently mapping them to the wrong mlp_* module type.
            if hf_module not in hf_modules and "." in hf_module:
                if not hf_module.startswith(("shared_expert.", "shared_experts.")):
                    final_component = hf_module.split(".")[-1]
                    if final_component in hf_modules:
                        hf_module = final_component

            if hf_module not in hf_modules:
                # Skip modules not in the supported mapping (only log once per module type)
                if hf_module not in getattr(iterate_hf_lora, "_warned_modules", set()):
                    logger.warning(
                        f"Skipping unsupported LoRA module '{hf_module}'. "
                        f"LoRA weights for this module will be ignored."
                    )
                    if not hasattr(iterate_hf_lora, "_warned_modules"):
                        iterate_hf_lora._warned_modules = set()
                    iterate_hf_lora._warned_modules.add(hf_module)
                continue  # Skip this module

        is_lora_a_or_b = m.group(8) is not None
        if is_lora_a_or_b:
            inout_or_mag = "in" if m.group(8) == "A" else "out"
        else:
            inout_or_mag = "magnitude"

        trtllm_module = (
            hf_modules_to_trtllm_modules[hf_module]
            if hf_modules_to_trtllm_modules is not None
            else hf_module
        )
        module_destination = (trtllm_module, layer_idx)
        previous_is_moe, previous_key = module_representations.setdefault(
            module_destination, (is_moe, key)
        )
        if previous_is_moe != is_moe:
            raise ValueError(
                f"LoRA keys '{previous_key}' and '{key}' mix flat and expert-indexed weights "
                f"for destination {module_destination} and selected owner 'decoder' "
                f"(module path '{owner.module_path or '<root>'}')."
            )
        _check_duplicate_destination(
            destinations, (trtllm_module, layer_idx, expert_idx, inout_or_mag), key, owner
        )
        if component is not None and component not in m.group(1):
            continue
        iter_fn(layer_idx, hf_module, expert_idx, inout_or_mag, weights)
        if not is_moe:
            all_weights[layer_idx][hf_module][inout_or_mag] = weights
        else:
            all_weights[layer_idx][hf_module].setdefault(expert_idx, {})
            all_weights[layer_idx][hf_module][expert_idx][inout_or_mag] = weights
    return all_weights


def get_all_hf_lora_weights(
    lora_weights: Dict[str, torch.Tensor],
    hf_modules: Set[str],
    component: Optional[str] = None,
    *,
    owner: DecoderLoraOwner = DecoderLoraOwner(),
    hf_modules_to_trtllm_modules: Optional[Dict[str, str]] = None,
):
    """Extract and organize all HuggingFace LoRA weights by layer and module.

    Args:
        lora_weights: Dictionary mapping weight keys to tensors from HF checkpoint
        hf_modules: Set of supported HF module names
        component: Optional component name to filter by (e.g., 'decoder')
        owner: Decoder module and accepted checkpoint prefixes
        hf_modules_to_trtllm_modules: Canonical destinations for detecting HF alias collisions

    Returns:
        Nested dictionary organizing weights by layer, module, and potentially expert
    """

    def iter_fn(layer_idx, hf_module, expert_idx, inout, weights):
        if expert_idx is None:
            all_weights[layer_idx][hf_module][inout] = weights
        else:
            all_weights[layer_idx][hf_module].setdefault(expert_idx, {})
            all_weights[layer_idx][hf_module][expert_idx][inout] = weights

    all_weights = defaultdict(lambda: defaultdict(dict))
    iterate_hf_lora(
        iter_fn,
        lora_weights,
        hf_modules,
        component,
        owner=owner,
        hf_modules_to_trtllm_modules=hf_modules_to_trtllm_modules,
    )
    return all_weights


def get_hf_target_modules(
    lora_weights,
    hf_modules,
    *,
    owner: DecoderLoraOwner = DecoderLoraOwner(),
    hf_modules_to_trtllm_modules: Optional[Dict[str, str]] = None,
):
    """Validate ownership and return supported HF targets present in the checkpoint."""

    def iter_fn(layer_idx, hf_module, expert_idx, inout, weights):
        hf_target_modules.add(hf_module)

    hf_target_modules = set()
    iterate_hf_lora(
        iter_fn,
        lora_weights,
        hf_modules,
        owner=owner,
        hf_modules_to_trtllm_modules=hf_modules_to_trtllm_modules,
    )
    return hf_target_modules


def invert_module_mapping(
    trtllm_modules_to_hf_modules: Dict[str, Union[str, List[str]]],
) -> Dict[str, str]:
    """Invert module mapping from TensorRT LLM -> HF to HF -> TensorRT-LLM.

    Args:
        trtllm_modules_to_hf_modules: Mapping from TensorRT LLM module names to HF module names
                                     (values can be strings or lists of strings)

    Returns:
        Dictionary mapping HF module names to TensorRT LLM module names
    """
    hf_modules_to_trtllm_modules: Dict[str, str] = {}
    for k, hf_modules in trtllm_modules_to_hf_modules.items():
        if isinstance(hf_modules, list):
            for hf_module in hf_modules:
                hf_modules_to_trtllm_modules[hf_module] = k
        else:
            hf_modules_to_trtllm_modules[hf_modules] = k
    return hf_modules_to_trtllm_modules


def norm_dora_magnitude(
    W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, m: torch.Tensor, scaling: float = 1.0
):
    new_weight_v = W0 + (B @ A) * scaling
    norm_m = m.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()
    return norm_m


class HfLoraLoader:
    def __init__(self, lora_dirs: List[str], *, owner: DecoderLoraOwner = DecoderLoraOwner()):
        self.owner = owner
        self.lora_target_modules = []
        self.is_valid = False
        self.lm_head = None
        self.embed_tokens = None
        self.vocab_size = 0

        if len(lora_dirs) == 0:
            return

        for lora_dir in lora_dirs:
            model_path = get_model_path(lora_dir, "adapter_model")
            if model_path is None:
                raise ValueError(f"adapter_model file does not exist in {lora_dir}")
            config_file = Path(f"{lora_dir}/adapter_config.json")
            if not config_file.exists():
                raise ValueError(f"{config_file} does not exist")
            if not config_file.is_file():
                raise ValueError(f"{config_file} is not a file")
        self.is_valid = True

        lora_dir = lora_dirs[0]
        with open(f"{lora_dir}/adapter_config.json") as f:
            adapter_config = json.load(f)

        model_path = get_model_path(lora_dir, "adapter_model")
        if model_path is None:
            raise ValueError(f"adapter_model file does not exist in {lora_dir}")
        lora_weight = load_state_dict(model_path)
        self.lora_weight = lora_weight
        if adapter_config.get("modules_to_save") is not None:
            if "lm_head" in adapter_config["modules_to_save"]:
                self.lm_head = lora_weight["base_model.model.lm_head.weight"]
                self.vocab_size = self.lm_head.shape[0]

            if "embed_tokens" in adapter_config["modules_to_save"]:
                self.embed_tokens = lora_weight["base_model.model.model.embed_tokens.weight"]

    def get_target_modules(self, trtllm_modules_to_hf_modules):
        hf_modules_to_trtllm_modules = invert_module_mapping(trtllm_modules_to_hf_modules)
        lora_target_modules = set()
        if self.is_valid:
            hf_target_modules = get_hf_target_modules(
                self.lora_weight,
                hf_modules=set(hf_modules_to_trtllm_modules.keys()),
                owner=self.owner,
                hf_modules_to_trtllm_modules=hf_modules_to_trtllm_modules,
            )
            for m in hf_target_modules:
                trtllm_module = hf_modules_to_trtllm_modules[m]
                lora_target_modules.add(trtllm_module)
        return list(lora_target_modules)

    def get_lora_dtype(self) -> Optional[torch.dtype]:
        """Return the common input/output dtype across all LoRA modules."""
        lora_dtypes = set()
        for key, weight in self.lora_weight.items():
            match = HF_LORA_PATTERN.match(key)
            if match is not None and match.group(8) in ("A", "B"):
                lora_dtypes.add(weight.dtype)
        if len(lora_dtypes) > 1:
            dtype_names = ", ".join(sorted(str(dtype) for dtype in lora_dtypes))
            raise ValueError(f"LoRA checkpoint weights must have one dtype; found {dtype_names}.")
        return next(iter(lora_dtypes)) if lora_dtypes else None


@lru_cache(maxsize=128)
def _find_nemo_files_single_path(lora_path: str) -> List[str]:
    """Find .nemo files from a single path (file or directory).

    This function is cached per individual path to maximize cache efficiency
    when the same paths appear in different collections.

    Args:
        lora_path: A single path that can be either:
                  - Direct path to a .nemo file
                  - Directory containing .nemo files (will auto-detect *.nemo)

    Returns:
        List[str]: List of paths to .nemo files found in this single path

    Raises:
        ValueError: If path doesn't exist, no .nemo files found, or invalid file type
    """
    path = Path(lora_path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")

    if path.is_file():
        if path.suffix == ".nemo":
            return [str(path)]
        else:
            raise ValueError(f"{path} is not a .nemo file")
    elif path.is_dir():
        nemo_files_in_dir = list(path.glob("*.nemo"))
        if not nemo_files_in_dir:
            raise ValueError(f"No .nemo files found in directory {path}")
        return [str(f) for f in nemo_files_in_dir]
    else:
        raise ValueError(f"{path} is neither a file nor a directory")


def find_nemo_files(lora_dirs: List[str]) -> List[str]:
    """Find all .nemo files from a list of directories or file paths.

    This function is optimized for repeated calls at generation time by using an internal LRU cache
    on individual paths, which maximizes cache efficiency when the same paths
    appear in different collections.

    Args:
        lora_dirs: List of paths that can be either:
                  - Direct paths to .nemo files
                  - Directories containing .nemo files (will auto-detect *.nemo)

    Returns:
        List[str]: List of paths to .nemo files

    Raises:
        ValueError: If a path doesn't exist, no .nemo files are found in a directory
        path, or a file path is of invalid file type
    """
    if len(lora_dirs) == 0:
        return []

    all_nemo_files: List[str] = []
    for lora_path in lora_dirs:
        nemo_files_for_path = _find_nemo_files_single_path(lora_path)
        all_nemo_files.extend(nemo_files_for_path)

    if not all_nemo_files:
        raise ValueError("No .nemo files found in the provided paths")

    return all_nemo_files


class NemoLoraLoader:
    def __init__(self, lora_dirs: List[str], *, owner: DecoderLoraOwner = DecoderLoraOwner()):
        """Initialize NemoLoraLoader with paths to .nemo files or directories.

        Args:
            lora_dirs: List of paths that can be either:
                      - Direct paths to .nemo files
                      - Directories containing .nemo files (will auto-detect *.nemo)

        Note: The parameter name 'lora_dirs' is misleading - it can accept both
              directories and files. This is a design flaw that should be fixed
              in a future version (e.g., rename to 'lora_paths').
        """
        self.owner = owner
        self.lora_dirs = lora_dirs
        self.lora_target_modules = []
        self.is_valid = False

        if len(lora_dirs) == 0:
            return

        for lora_file in lora_dirs:
            path = Path(lora_file)
            if not path.exists():
                raise ValueError(f"{path} does not exist")
        self.is_valid = True
        self.lora_target_modules = list(NEMO_SUPPORTED_LORA_MODULES)

    def get_target_modules(self):
        """Get target modules for NeMo LoRA.

        Unlike the HF loader, this method does not accept trtllm_modules_to_hf_modules
        as an argument since the module mapping is hardcoded for NeMo LoRA support.

        Returns:
            List[str]: List of target module names supported by NeMo LoRA
        """
        return self.lora_target_modules

    def validate_weights(self) -> None:
        """Reject unsupported owners and ambiguous destinations in each archive."""
        for model_file in find_nemo_files(self.lora_dirs):
            _, weights = unpack_nemo_weights(model_file)
            get_all_nemo_lora_weights(weights, owner=self.owner)


def unpack_nemo_weights(nemo_archive_path: str) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    """Unpack model config and weights from a NeMo .nemo archive file.

    Args:
        nemo_archive_path: Path to the .nemo archive file

    Returns:
        Tuple of (model_config_dict, model_weights_dict)

    Raises:
        Exception: If required files cannot be extracted from the archive
    """
    with tarfile.open(nemo_archive_path) as tar:
        try:
            model_weights_file = tar.extractfile("model_weights.ckpt")
            model_config_file = tar.extractfile("model_config.yaml")
        except KeyError:
            try:
                model_weights_file = tar.extractfile("./model_weights.ckpt")
                model_config_file = tar.extractfile("./model_config.yaml")
            except KeyError:
                err_str = "Both model_weights paths not found in the tar archive."
                raise Exception(err_str)

        if model_weights_file is None or model_config_file is None:
            raise Exception("Could not extract model weights or config files")

        model_config_content = model_config_file.read()
        model_config_dict = yaml.safe_load(model_config_content)

        model_weights_bytes = model_weights_file.read()
        model_weights_dict = torch.load(
            io.BytesIO(model_weights_bytes),
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        return model_config_dict, model_weights_dict
