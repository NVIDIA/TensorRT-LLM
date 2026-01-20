import io
import itertools
import json
import logging
import re
import tarfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import yaml

from tensorrt_llm.bindings import internal as tb_internal

from ._utils import pad_vocab_size, release_gc, str_dtype_to_torch, torch_to_numpy
from .layers.linear import ColumnLinear
from .lora_helper import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
    get_missing_qkv_modules_from_lora_modules,
)
from .mapping import Mapping
from .models.convert_utils import get_model_path, load_state_dict, split_matrix_tp

if TYPE_CHECKING:
    from .runtime import ModelConfig

NEMO_SUPPORTED_LORA_MODULES = {"attn_qkv"}

logger = logging.getLogger(__name__)


def _check_lora_in_out(
    layer_idx: int, lora_module: str, available_matrices: Dict, source_identifier: str
) -> None:
    """Check that 'in' and 'out' matrices are present."""
    missing = []
    if "in" not in available_matrices:
        missing.append("'in' matrix (lora_A equivalent)")
    if "out" not in available_matrices:
        missing.append("'out' matrix (lora_B equivalent)")

    if missing:
        raise ValueError(
            f"Layer {layer_idx} is missing required {' and '.join(missing)} for {lora_module} "
            f"in LoRA weights from {source_identifier}. "
            f"LoRA adapters must contain both 'in' and 'out' matrices for all layers. "
            f"Please check if the LoRA checkpoint is complete or was corrupted during loading."
        )


def _is_moe_module_weights(module_weights: Dict) -> bool:
    """Check if module weights represent MoE (integer expert indices with nested dicts)."""
    if not module_weights:
        return False

    # All keys should be integers (expert indices) and values should be dicts
    return all(isinstance(k, int) for k in module_weights.keys()) and all(
        isinstance(v, dict) for v in module_weights.values()
    )


def get_all_nemo_lora_weights(
    lora_weights: Dict[str, torch.Tensor],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract and organize NeMo LoRA weights by layer and direction.

    Args:
        lora_weights: Dictionary mapping weight keys to tensors from NeMo checkpoint

    Returns:
        Dictionary mapping layer_idx -> {direction -> tensor} where direction is 'in' or 'out'

    Raises:
        KeyError: If unsupported keys are found or layer extraction fails
    """
    layer_weights: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
    adapter_key = "self_attention.adapter_layer.lora_kqv_adapter"
    layer_pattern = re.compile(r".*\.layers\.(\d+)\..*")
    for key, weights in lora_weights.items():
        if adapter_key in key:
            if key.endswith("linear_in.weight"):
                inout = "in"
            elif key.endswith("linear_out.weight"):
                inout = "out"
            else:
                continue
            m = layer_pattern.match(key)
            if m is None:
                raise KeyError(
                    f"Failed to extract layer index from key {key} using pattern {layer_pattern.pattern}"
                )
            layer_idx = int(m.group(1))
            layer_weights[layer_idx][inout] = weights
        else:
            raise KeyError(f"unsupported key {key} from Nemo LoRA weights")
    return layer_weights


# The pattern is {layer_prefix:1}.{layer_idx:2}.{module_prefix:3}.{module_name or {expert_name:5}.{expert_idx:6}.{module_name:7} :4}.lora_{A|B:8}.weight  # noqa: E501
HF_LORA_PATTERN = re.compile(
    r"(.*)\.(\d+)\.(\w+)\.(\w+|\w+\.\w+|(\w+)\.(\d+)\.(\w+))\.(?:lora_(?:(A|B)\.weight|(magnitude)_vector)|weight_(m_wdecomp).weight)"
)


def iterate_hf_lora(
    iter_fn,
    lora_weights: Dict[str, torch.Tensor],
    hf_modules: Set[str],
    component: Optional[str] = None,
):
    """Iterate over HuggingFace LoRA weights and call iterator function for each weight.

    Args:
        iter_fn: Function to call for each weight with signature
        (layer_idx, hf_module, expert_idx, inout_or_mag, weights)
        lora_weights: Dictionary mapping weight keys to tensors from HF checkpoint
        hf_modules: Set of supported HF module names
        component: Optional component name to filter by (e.g., 'decoder')

    Returns:
        Nested dictionary structure organizing the weights

    Raises:
        KeyError: If unsupported keys are found
        AssertionError: If HF module is not in supported list
    """
    all_weights = defaultdict(lambda: defaultdict(dict))
    pattern = HF_LORA_PATTERN
    for key, weights in lora_weights.items():
        m = pattern.match(key)
        if not m:
            if "lm_head" not in key and "embed_tokens" not in key:
                raise KeyError(f"unsupported key {key} from HF LoRA weights")
            continue
        if component is not None and component not in m.group(1):
            continue
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
            assert hf_module in hf_modules, (
                f"hf_module {hf_module} is not in supported list {hf_modules}"
            )

        is_lora_a_or_b = m.group(8) is not None
        if is_lora_a_or_b:
            inout_or_mag = "in" if m.group(8) == "A" else "out"
        else:
            inout_or_mag = "magnitude"

        iter_fn(layer_idx, hf_module, expert_idx, inout_or_mag, weights)
        if not is_moe:
            all_weights[layer_idx][hf_module][inout_or_mag] = weights
        else:
            all_weights[layer_idx][hf_module].setdefault(expert_idx, {})
            all_weights[layer_idx][hf_module][expert_idx][inout_or_mag] = weights
    return all_weights


def get_all_hf_lora_weights(
    lora_weights: Dict[str, torch.Tensor], hf_modules: Set[str], component: Optional[str] = None
):
    """Extract and organize all HuggingFace LoRA weights by layer and module.

    Args:
        lora_weights: Dictionary mapping weight keys to tensors from HF checkpoint
        hf_modules: Set of supported HF module names
        component: Optional component name to filter by (e.g., 'decoder')

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
    iterate_hf_lora(iter_fn, lora_weights, hf_modules, component)
    return all_weights


def get_hf_target_modules(lora_weights, hf_modules):
    def iter_fn(layer_idx, hf_module, expert_idx, inout, weights):
        hf_target_modules.add(hf_module)

    hf_target_modules = set()
    iterate_hf_lora(iter_fn, lora_weights, hf_modules)
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


@dataclass
class LoraModelConfig:
    lora_target_modules: list[str]
    trtllm_modules_to_hf_modules: dict[str, str]
    hidden_size: int
    dtype: str
    swap_gate_up_proj_lora_b_weight: bool = True


class HfLoraLoader:
    def __init__(self, lora_dirs: List[str]):
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
            )
            for m in hf_target_modules:
                trtllm_module = hf_modules_to_trtllm_modules[m]
                lora_target_modules.add(trtllm_module)
        return list(lora_target_modules)


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
    def __init__(self, lora_dirs: List[str]):
        """Initialize NemoLoraLoader with paths to .nemo files or directories.

        Args:
            lora_dirs: List of paths that can be either:
                      - Direct paths to .nemo files
                      - Directories containing .nemo files (will auto-detect *.nemo)

        Note: The parameter name 'lora_dirs' is misleading - it can accept both
              directories and files. This is a design flaw that should be fixed
              in a future version (e.g., rename to 'lora_paths').
        """
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


def load_nemo_lora(model, lora_config: LoraConfig):
    lora_loader = NemoLoraLoader(lora_config.lora_dir)

    if not lora_loader.is_valid:
        raise ValueError(f"Failed to load NeMo LoRA from {lora_config.lora_dir}")

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.lora_target_modules


def load_torch_hf_lora(lora_config: LoraConfig):
    """This is a shortned version of load_hf_lora that is used for torch models.

    Main problem is model.config in legacy code is custom (defined in the legacy code) whereas
    pivot model config is the transformer's one.
    """
    # TODO smor- need to comibe with load_hf_lora
    if not lora_config.trtllm_modules_to_hf_modules:
        lora_config.trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules()

    assert len(lora_config.lora_dir) == 1, "Expecting only a single lora dir"
    lora_loader = HfLoraLoader(lora_config.lora_dir)

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.get_target_modules(
            lora_config.trtllm_modules_to_hf_modules
        )

    if len(lora_config.lora_target_modules) == 0:
        raise ValueError(
            "lora_target_modules is empty. "
            "Please specify lora_target_modules or provide lora_dir to infer lora_target_modules."
        )

    missing_qkv_modules = LoraManager.get_missing_qkv_modules(lora_config.lora_target_modules)
    lora_config.lora_target_modules.extend(missing_qkv_modules)


def load_torch_nemo_lora(lora_config: LoraConfig):
    """Load NeMo LoRA checkpoint for PyTorch workflow.

    This is a PyTorch-specific loader for NeMo LoRA checkpoints, similar to
    load_torch_hf_lora but handling NeMo checkpoint format. NeMo uses a combined
    "attn_qkv" module rather than separate Q, K, V modules, so no missing QKV
    module handling is needed.

    Note: This function only sets up the configuration. For PyTorch workflow,
    the actual weight loading happens later via LoraManager when requests are
    made with LoRA UIDs.

    Args:
        lora_config: LoRA configuration with lora_ckpt_source="nemo"

    Raises:
        ValueError: If NeMo LoRA directory is invalid or unsupported modules are specified
    """
    lora_config.trtllm_modules_to_hf_modules = {"attn_qkv": "attn_qkv"}

    assert len(lora_config.lora_dir) == 1, "Expecting only a single lora dir"
    lora_loader = NemoLoraLoader(lora_config.lora_dir)

    if not lora_loader.is_valid:
        raise ValueError(f"Failed to load NeMo LoRA from {lora_config.lora_dir}")

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.get_target_modules()

    if len(lora_config.lora_target_modules) == 0:
        raise ValueError(
            "lora_target_modules is empty. "
            "Please specify lora_target_modules or provide lora_dir to infer lora_target_modules."
        )

    unsupported_modules = set(lora_config.lora_target_modules) - NEMO_SUPPORTED_LORA_MODULES
    if unsupported_modules:
        raise ValueError(
            f"NeMo LoRA only supports {NEMO_SUPPORTED_LORA_MODULES} modules, "
            f"but got unsupported modules: {unsupported_modules}. "
            f"NeMo LoRA does not support embedding, lm_head, or MLP adapters."
        )


def load_torch_lora(lora_config: LoraConfig):
    """Load LoRA checkpoint for PyTorch workflow.

    This function routes to the appropriate loader based on lora_ckpt_source.

    Args:
        lora_config: LoRA configuration with lora_ckpt_source set to "hf" or "nemo"

    Raises:
        ValueError: If lora_ckpt_source is not supported
    """
    if lora_config.lora_ckpt_source == "nemo":
        load_torch_nemo_lora(lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        load_torch_hf_lora(lora_config)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}. "
            f"Supported sources: 'hf', 'nemo'"
        )


def load_hf_lora(
    model,
    lora_config: LoraConfig,
    trtllm_modules_to_hf_modules: Optional[Dict[str, str]] = None,
):
    trtllm_modules_to_hf_modules = (
        trtllm_modules_to_hf_modules or get_default_trtllm_modules_to_hf_modules()
    )
    lora_config.trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules

    lora_loader = HfLoraLoader(lora_config.lora_dir)

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.get_target_modules(
            trtllm_modules_to_hf_modules
        )
    if len(lora_config.lora_target_modules) == 0:
        raise ValueError(
            "lora_target_modules is empty. "
            "Please specify lora_target_modules or provide lora_dir to infer lora_target_modules."
        )

    missing_qkv_modules = LoraManager.get_missing_qkv_modules(lora_config.lora_target_modules)
    lora_config.lora_target_modules.extend(missing_qkv_modules)

    if lora_loader.is_valid:
        config = model.config
        torch_dtype = str_dtype_to_torch(config.dtype)
        # the lora checkpoint might finetune the embedding
        if lora_loader.vocab_size != 0:
            config.vocab_size = lora_loader.vocab_size
        mapping = config.mapping
        if mapping.is_first_pp_rank() and lora_loader.embed_tokens is not None:
            weight = lora_loader.embed_tokens
            if config.use_parallel_embedding:
                weight = split_matrix_tp(
                    weight,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=config.embedding_sharding_dim,
                )
            if model.transformer.vocab_embedding.weight.raw_value.shape != weight.shape:
                model.transformer.vocab_embedding = model.transformer.vocab_embedding.__class__(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.hidden_size,
                    dtype=config.dtype,
                    tp_size=mapping.tp_size if config.use_parallel_embedding else 1,
                    tp_group=mapping.tp_group if config.use_parallel_embedding else None,
                    sharding_dim=config.embedding_sharding_dim,
                    tp_rank=mapping.tp_rank,
                )
            model.transformer.vocab_embedding.weight.value = weight.to(torch_dtype)
        if mapping.is_last_pp_rank() and lora_loader.lm_head is not None:
            weight = lora_loader.lm_head
            vocab_size = lora_loader.vocab_size
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                weight = torch.from_numpy(
                    np.pad(
                        torch_to_numpy(weight),
                        ((0, pad_width), (0, 0)),
                        "constant",
                        constant_values=0,
                    )
                )
            else:
                vocab_size_padded = vocab_size
            if model.lm_head.weight.raw_value.shape != weight.shape:
                model.lm_head = ColumnLinear(
                    config.hidden_size,
                    vocab_size_padded,
                    bias=False,
                    dtype=config.dtype,
                    tp_group=mapping.tp_group,
                    tp_size=mapping.tp_size,
                    gather_output=True,
                )
            model.lm_head.weight.value = split_matrix_tp(
                weight,
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            ).to(torch_dtype)


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
            io.BytesIO(model_weights_bytes), map_location=torch.device("cpu")
        )

        return model_config_dict, model_weights_dict


class LoraManager(object):
    LORA_MODULE_IDS = {
        "attn_qkv": 0,
        "attn_q": 1,
        "attn_k": 2,
        "attn_v": 3,
        "attn_dense": 4,
        "mlp_h_to_4h": 5,
        "mlp_4h_to_h": 6,
        "mlp_gate": 7,
        "cross_attn_qkv": 8,
        "cross_attn_q": 9,
        "cross_attn_k": 10,
        "cross_attn_v": 11,
        "cross_attn_dense": 12,
        "moe_h_to_4h": 13,
        "moe_4h_to_h": 14,
        "moe_gate": 15,
        "moe_router": 16,
        "mlp_router": 17,
        "mlp_gate_up": 18,
    }

    def __init__(
        self,
        *,
        mapping: Mapping,
        model_config: "ModelConfig",
        cpp_peft_cache_manager: tb_internal.batch_manager.PeftCacheManager | None = None,
    ):
        """Constructor.

        Args:
            mapping (Mapping): Parallelism related information.
            model_config (ModelConfig): model configuration python class instance.
            cpp_peft_cache_manager (PeftCacheManager, optional): used by is_adapter_in_cpu_cache method, that's used for
                a performance optimization with LoRA of not sending the LoRA adapter weights with every LLM request when
                the adapter is already loaded in the LoRA CPU cache.
        """
        # _lora_uid_to_low_ranks: dict[str -> dict[int -> dict[str -> int]]]
        # {
        #     uid: {
        #         0: {
        #             lora_module: int
        #         }, # layer_0_rank,
        #         1: {
        #             lora_module: int
        #         }, # layer_1_rank,
        #         ...
        #     }
        # }

        # _lora_weights_pointers_list: dict[str -> dict[int -> dict[str -> [Tensor, Tensor]]]]
        # {
        #     uid: {
        #         0: {
        #             lora_module: [t_in, t_out]
        #         }, # layer_0,
        #         1: {
        #             lora_module: [t_in, t_out]
        #         }, # layer_1,
        #         ...
        #     }
        # }

        self._lora_uid_counter = 0
        self._lora_uid_to_low_ranks: Dict[str, Dict[int, Dict[str, int]]] = {}
        # hold the torch tensors and prevent them from being freed
        # TODO(enweiz): free device tensors if it's used for c++ runtime only
        self._lora_weights: List[torch.Tensor] = []
        self._lora_weights_pointers_list: Dict[str, Dict[int, Dict[str, List[int]]]] = {}
        self._cpp_lora_weights: Dict[str, torch.Tensor] = {}  # on cpu
        self._cpp_lora_config: Dict[str, torch.Tensor] = {}  # on cpu
        self.lora_target_modules: List[str] = []
        self._mapping = mapping
        self._model_config = model_config
        self._cpp_peft_cache_manager = cpp_peft_cache_manager

    def is_adapter_in_cpu_cache(self, adapter_uid: int) -> bool:
        """Best effort to check if a LoRA adapter is in the LoRA CPU cache.

        If no cpp_peft_cache_manager instance was given at the construction of this LoraManager instance, then False is
        returned.
        """
        return (
            self._cpp_peft_cache_manager.is_task_cached(adapter_uid)
            if self._cpp_peft_cache_manager
            else False
        )

    @staticmethod
    def get_missing_qkv_modules(lora_target_modules: List[str]) -> List[str]:
        return get_missing_qkv_modules_from_lora_modules(lora_target_modules)

    @property
    def missing_qkv_modules(self) -> List[str]:
        return LoraManager.get_missing_qkv_modules(self.lora_target_modules)

    def load_from_ckpt(
        self,
        model_dirs_or_files: List[str],
        model_config: Union["ModelConfig", LoraModelConfig],
        uids: Optional[List[str]] = None,
        ckpt_source: str = "hf",
    ) -> List[str]:
        """Returns the adapter UIDs that were loaded by this call.

        Note that when an adapter was already loaded before this call, it would not be
        included in the returned list of UIDs.
        """
        if ckpt_source == "hf":
            return self.load_from_hf(
                model_dirs=model_dirs_or_files,
                model_config=model_config,
                uids=uids,
            )
        elif ckpt_source == "nemo":
            # Find all .nemo files from directories or files
            nemo_files = find_nemo_files(model_dirs_or_files)

            # Pass the actual .nemo files to the loader
            return self.load_from_nemo(
                model_files=nemo_files,
                model_config=model_config,
                uids=uids,
            )
        else:
            assert False, f"{self.__class__.__name__} does not support source {ckpt_source}"

    def load_from_nemo(
        self,
        model_files: List[str],
        model_config: Union["ModelConfig", LoraModelConfig],
        uids: Optional[List[str]] = None,
    ) -> List[str]:
        """Returns the adapter UIDs that were loaded by this call.

        Note that when an adapter was already loaded before this call, it would not be
        included in the returned list of UIDs.
        """
        if uids is None:
            uids = [self._generate_uid() for _ in range(len(model_files))]
        assert len(uids) == len(model_files)

        new_uids, new_model_files = [], []
        for uid, model_file in zip(uids, model_files):
            if uid in self._lora_uid_to_low_ranks:
                continue
            new_uids.append(uid)
            new_model_files.append(model_file)

        if len(new_uids) == 0:
            return new_uids

        self.lora_target_modules = model_config.lora_target_modules

        def load_from_model_file(uid, model_file):
            if uid not in self._cpp_lora_weights:
                self._cpp_lora_weights[uid] = []  # Will be converted to tensor later
            if uid not in self._cpp_lora_config:
                self._cpp_lora_config[uid] = []  # Will be converted to tensor later

            _, nemo_weights = unpack_nemo_weights(model_file)
            all_lora_weights = get_all_nemo_lora_weights(nemo_weights)

            self._lora_uid_to_low_ranks[uid] = {}
            self._lora_weights_pointers_list[uid] = {}
            for layer_idx in sorted(all_lora_weights.keys()):
                self._lora_uid_to_low_ranks[uid][layer_idx] = {}
                self._lora_weights_pointers_list[uid][layer_idx] = {}

                for lora_module in self.lora_target_modules:
                    if lora_module not in NEMO_SUPPORTED_LORA_MODULES:
                        warnings.warn(
                            f"LoRA module '{lora_module}' not supported in NeMo loading for "
                            f"layer {layer_idx}, skipping. NeMo LoRA currently only supports "
                            f"{NEMO_SUPPORTED_LORA_MODULES} modules."
                        )
                        self._lora_uid_to_low_ranks[uid][layer_idx][lora_module] = 0
                        continue

                    if lora_module == "attn_qkv":
                        # Validate required matrices are present
                        _check_lora_in_out(
                            layer_idx=layer_idx,
                            lora_module=lora_module,
                            available_matrices=all_lora_weights[layer_idx],
                            source_identifier=f"file {model_file}",
                        )

                        t_in = all_lora_weights[layer_idx]["in"]
                        t_out = all_lora_weights[layer_idx]["out"]
                    else:
                        t_in = None
                        t_out = None

                    if t_in is not None and t_out is not None:
                        t_in = t_in.cuda().to(str_dtype_to_torch(model_config.dtype)).contiguous()
                        t_out = t_out.cuda().to(str_dtype_to_torch(model_config.dtype)).contiguous()
                        rank = t_in.shape[0]
                        self._lora_uid_to_low_ranks[uid][layer_idx][lora_module] = int(rank)
                        self._lora_weights_pointers_list[uid][layer_idx][lora_module] = [
                            t_in.data_ptr(),
                            t_out.data_ptr(),
                            0,
                        ]

                        # prevent torch free this buffer
                        self._lora_weights.append(t_in)
                        self._lora_weights.append(t_out)
                        self._cpp_lora_weights[uid].append(
                            torch.concatenate([t_in.flatten().cpu(), t_out.flatten().cpu()])
                        )
                        self._cpp_lora_config[uid].append(
                            torch.tensor(
                                [self.LORA_MODULE_IDS[lora_module], layer_idx, int(rank)],
                                dtype=torch.int32,
                            )
                        )

            max_weight_size = max(w.size(0) for w in self._cpp_lora_weights[uid])
            self._cpp_lora_weights[uid] = torch.stack(
                [
                    torch.nn.functional.pad(w, (0, max_weight_size - w.size(0)))
                    for w in self._cpp_lora_weights[uid]
                ]
            )
            self._cpp_lora_config[uid] = torch.stack([c for c in self._cpp_lora_config[uid]])

        for uid, model_file in zip(new_uids, new_model_files):
            load_from_model_file(uid, model_file)
            release_gc()

        if new_uids:
            logger.info(f"Successfully loaded NeMo LoRA adapters with UIDs: {new_uids}")
        return new_uids

    def load_from_hf(
        self,
        model_dirs: List[str],
        model_config: Union["ModelConfig", LoraModelConfig],
        uids: Optional[List[str]] = None,
        component: Optional[str] = None,
    ) -> List[str]:
        """Returns the adapter UIDs that were loaded by this call.

        Note that when an adapter was already loaded before this call, it would not be
        included in the returned list of UIDs.

        Lora config of https://huggingface.co/hfl/chinese-alpaca-2-lora-7b.

        {
            "base_model_name_or_path": "/Llama-2-7b-hf",
            "bias": "none",
            "enable_lora": null,
            "fan_in_fan_out": false,
            "inference_mode": true,
            "lora_alpha": 128.0,
            "lora_dropout": 0.05,
            "merge_weights": false,
            "modules_to_save": [
                "embed_tokens",
                "lm_head"
            ],
            "peft_type": "LORA",
            "r": 64,
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            "task_type": "CAUSAL_LM"

        }

        keys in adapter_model.bin:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.up_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.up_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.down_proj.lora_A.weight torch.Size([64, 11008])
            base_model.model.model.layers.0.mlp.down_proj.lora_B.weight torch.Size([4096, 64])
            ...

        """
        if uids is None:
            uids = [self._generate_uid() for _ in range(len(model_dirs))]
        assert len(uids) == len(model_dirs)

        new_uids, new_model_dirs = [], []
        for uid, model_dir in zip(uids, model_dirs):
            if uid in self._lora_uid_to_low_ranks:
                continue
            new_uids.append(uid)
            new_model_dirs.append(model_dir)

        if len(new_uids) == 0:
            return new_uids

        lora_hf_configs = []
        for model_dir in new_model_dirs:
            with open(f"{model_dir}/adapter_config.json", "r") as f:
                config = json.load(f)
                lora_hf_configs.append(config)

        self.lora_target_modules = model_config.lora_target_modules
        hf_modules_to_trtllm_modules = invert_module_mapping(
            model_config.trtllm_modules_to_hf_modules
        )
        hf_modules = set(hf_modules_to_trtllm_modules.keys())

        def preprocess_lora_weights(lora_model, model_config):
            # Swap weights of gate_up_proj
            if getattr(model_config, "swap_gate_up_proj_lora_b_weight", True):
                for key, value in lora_model.items():
                    if "gate_up_proj.lora_B.weight" in key:
                        original_weights = value.contiguous().clone()
                        half_split = original_weights.shape[0] // 2
                        first_half = original_weights[:half_split, :]
                        second_half = original_weights[half_split:, :]
                        value = torch.cat((second_half, first_half), dim=0)
                        lora_model[key] = value
            return lora_model

        def interleave_fused_lora_weights_for_tp(
            weight: torch.Tensor, rank_dim: int, tp_size: int, part_sizes: List[int]
        ) -> List[torch.Tensor]:
            """Interleaves fused LoRA modules weights for TP.
            e.g.  In case of attn_qkv: Convert t_out=torch.cat([Wq, Wk, Wv]) to
                  torch.cat([Wq_rank0, Wk_rank0, Wv_rank0, ..., Wq_rankN, Wk_rankN, Wv_rankN])
                  where N=TP size.
            """  # noqa: D205
            assert weight.shape[rank_dim] == sum(part_sizes)

            # Split the weights into their respective parts. e.g. weight -> [Wq, Wk, Wv] for attn_qkv.
            weight_parts = [
                weight.narrow(rank_dim, sum(part_sizes[:i]), part_sizes[i])
                for i in range(len(part_sizes))
            ]
            for i in range(len(part_sizes)):
                assert weight_parts[i].shape[rank_dim] % tp_size == 0

            # Split each part into tp_size chunks.
            # e.g. [Wq, Wk, Wv] -> [[Wq_rank0, ..., Wq_rankN], [Wk_rank0, ..., Wk_rankN], [Wv_rank0, ..., Wv_rankN]]
            # where N is TP size, for attn_qkv.
            weight_parts_tp_weights = [
                torch.split(
                    weight_parts[i], weight_parts[i].shape[rank_dim] // tp_size, dim=rank_dim
                )
                for i in range(len(part_sizes))
            ]

            # Interleave the parts across TP ranks and flatten the list of lists into a single list.
            # e.g. [[Wq_rank0, ..., Wq_rankN], [Wk_rank0, ..., Wk_rankN], [Wv_rank0, ..., Wv_rankN]]
            # -> [Wq_rank0, Wk_rank0, Wv_rank0, ..., Wq_rankN, Wk_rankN, Wv_rankN] where N is TP size, for attn_qkv.
            return list(itertools.chain.from_iterable(zip(*weight_parts_tp_weights)))

        def prepare_fused_lora_modules_for_tp(
            lora_module: str, t_out: torch.Tensor, rank_dim: int
        ) -> torch.Tensor:
            """Reorders fused LoRA modules weights for TP. This is required since HF stores the parts weights
            sequentially, whereas with TP>1 we need them to be interleaved so they would be sharded correctly.

            See interleave_fused_lora_weights_for_tp for more details.
            """  # noqa: D205
            tp_size = self._mapping.tp_size
            if tp_size == 1:
                return t_out
            part_sizes = []
            if lora_module == "mlp_gate_up":
                assert t_out.shape[rank_dim] % 2 == 0
                half_size = t_out.shape[rank_dim] // 2
                part_sizes = [half_size, half_size]
            elif lora_module == "attn_qkv":
                # The sizes are multiplied by tp_size because num_heads and num_kv_heads here were already
                # divided by tp_size in tensorrt_llm/_torch/model_config.py::ModelConfig.get_bindings_model_config
                q_size = self._model_config.head_size * self._model_config.num_heads * tp_size
                kv_size = self._model_config.head_size * self._model_config.num_kv_heads * tp_size
                part_sizes = [q_size, kv_size, kv_size]

            if part_sizes:
                interleaved_parts = interleave_fused_lora_weights_for_tp(
                    t_out, rank_dim, tp_size, part_sizes
                )
                # Concatenate them all after interleaving, as the CPP implementation expects the full non-split weights.
                t_out = torch.cat(interleaved_parts, dim=rank_dim)
            return t_out

        def load_from_model_dir(uid, model_dir, hf_config):
            if uid not in self._cpp_lora_weights:
                self._cpp_lora_weights[uid] = []  # Will be converted to tensor later
            if uid not in self._cpp_lora_config:
                self._cpp_lora_config[uid] = []  # Will be converted to tensor later

            lora_model = load_state_dict(get_model_path(model_dir, "adapter_model"))
            if lora_model is None:
                raise ValueError(f"Failed to load adapter_model from {model_dir}")
            lora_model = preprocess_lora_weights(lora_model, model_config)
            all_weights = get_all_hf_lora_weights(lora_model, hf_modules, component)
            rank = int(hf_config["r"])
            rs_lora = bool(hf_config.get("use_rslora", False))

            self._lora_uid_to_low_ranks[uid] = {}
            self._lora_weights_pointers_list[uid] = {}
            for layer_idx in sorted(all_weights.keys()):
                layer_weights = all_weights[layer_idx]
                self._lora_uid_to_low_ranks[uid][layer_idx] = {}
                self._lora_weights_pointers_list[uid][layer_idx] = {}

                for lora_module in self.missing_qkv_modules:
                    hf_module = model_config.trtllm_modules_to_hf_modules[lora_module]
                    if isinstance(hf_module, list):
                        hf_module = hf_module[0]
                    layer_weights[hf_module] = {
                        "in": torch.zeros(rank, model_config.hidden_size),
                        "out": torch.zeros(model_config.hidden_size, rank),
                    }

                for hf_module, module_weights in layer_weights.items():
                    lora_module = hf_modules_to_trtllm_modules[hf_module]
                    if lora_module not in self.lora_target_modules:
                        warnings.warn(
                            f"LoRA module '{lora_module}' not in target modules {self.lora_target_modules}, skipping."
                        )
                        self._lora_uid_to_low_ranks[uid][layer_idx][lora_module] = 0
                        continue

                    has_expert_indices = _is_moe_module_weights(module_weights)

                    if has_expert_indices:  # MoE
                        # Validate and extract matrices in one pass
                        expert_indices = sorted(module_weights.keys())
                        t_in_list, t_out_list = [], []
                        for expert_idx in expert_indices:
                            expert_weights = module_weights[expert_idx]
                            _check_lora_in_out(
                                layer_idx=layer_idx,
                                lora_module=f"{lora_module}_expert_{expert_idx}",
                                available_matrices=expert_weights,
                                source_identifier=f"directory {model_dir}",
                            )
                            t_in_list.append(expert_weights["in"])
                            t_out_list.append(expert_weights["out"])

                        t_in = torch.stack(t_in_list)
                        t_out = torch.stack(t_out_list)
                        for weights in module_weights.values():
                            if "mag" in weights:
                                # TODO(oargov): this might work, but I had no MoE DoRA models to test
                                raise ValueError("DoRA with MoE is not supported")
                        t_mag = None
                    else:
                        # Not MoE - validate required matrices are present
                        _check_lora_in_out(
                            layer_idx=layer_idx,
                            lora_module=lora_module,
                            available_matrices=module_weights,
                            source_identifier=f"directory {model_dir}",
                        )

                        t_in = module_weights["in"]
                        t_out = module_weights["out"]
                        t_mag = module_weights.get("magnitude", None)

                    is_dora = t_mag is not None
                    rank_dim = 1 if has_expert_indices else 0
                    t_out = prepare_fused_lora_modules_for_tp(lora_module, t_out, rank_dim)

                    effective_rank = t_in.shape[rank_dim]

                    t_in = t_in.cuda().contiguous()
                    t_out = t_out.cuda().contiguous()
                    if is_dora and t_mag is not None:
                        t_mag = t_mag.cuda().contiguous()

                    if rs_lora:
                        scale = float(hf_config["lora_alpha"]) / np.sqrt(effective_rank)
                    else:
                        scale = float(hf_config["lora_alpha"]) / effective_rank
                    t_out = t_out * scale
                    t_in = t_in.to(str_dtype_to_torch(model_config.dtype))
                    t_out = t_out.to(str_dtype_to_torch(model_config.dtype))
                    if is_dora and t_mag is not None:
                        t_mag = t_mag.to(str_dtype_to_torch(model_config.dtype))

                    self._lora_uid_to_low_ranks[uid][layer_idx][lora_module] = effective_rank
                    self._lora_weights_pointers_list[uid][layer_idx][lora_module] = [
                        t_in.data_ptr(),
                        t_out.data_ptr(),
                        t_mag.data_ptr() if (is_dora and t_mag is not None) else 0,
                    ]

                    # prevent torch free this buffer
                    self._lora_weights.append(t_in)
                    self._lora_weights.append(t_out)
                    if is_dora and t_mag is not None:
                        self._lora_weights.append(t_mag)

                    t_in_cpu = t_in.flatten().cpu()
                    t_out_cpu = t_out.flatten().cpu()
                    weights_to_concat = [t_in_cpu, t_out_cpu]

                    if is_dora and t_mag is not None:
                        t_mag_cpu = t_mag.flatten().cpu()
                        weights_to_concat.append(t_mag_cpu)

                    self._cpp_lora_weights[uid].append(torch.cat(weights_to_concat))
                    self._cpp_lora_config[uid].append(
                        torch.tensor(
                            [self.LORA_MODULE_IDS[lora_module], layer_idx, effective_rank, is_dora],
                            dtype=torch.int32,
                        )
                    )

            max_weight_size = max(w.size(0) for w in self._cpp_lora_weights[uid])
            self._cpp_lora_weights[uid] = torch.stack(
                [
                    torch.nn.functional.pad(w, (0, max_weight_size - w.size(0)))
                    for w in self._cpp_lora_weights[uid]
                ]
            )
            self._cpp_lora_config[uid] = torch.stack([c for c in self._cpp_lora_config[uid]])

        for uid, model_dir, hf_config in zip(new_uids, new_model_dirs, lora_hf_configs):
            load_from_model_dir(uid, model_dir, hf_config)
            release_gc()

        return new_uids

    @property
    def lora_weights(self):
        return self._lora_weights

    @property
    def lora_weights_pointers_list(self):
        return self._lora_weights_pointers_list

    @property
    def cpp_lora_weights(self):
        return self._cpp_lora_weights

    @property
    def cpp_lora_config(self):
        return self._cpp_lora_config

    def uid_to_low_ranks(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_low_ranks[uid]

    def _generate_uid(self):
        while str(self._lora_uid_counter) in self._lora_uid_to_low_ranks:
            self._lora_uid_counter += 1
        uid = str(self._lora_uid_counter)
        self._lora_uid_counter += 1
        return uid

    @property
    def num_lora_adapters(self):
        return len([uid for uid in self._lora_uid_to_low_ranks if uid != "-1"])

    def save_lora_weights_to_bin(self, out_dir):
        def save_val(val, dir, key, tp_num=None, write_npy=False):
            ext = "npy" if write_npy else "bin"
            suffix = ext if tp_num is None else f"{tp_num}.{ext}"
            if write_npy:
                np.save(dir / f"model.{key}.{suffix}", val)
            else:
                val.tofile(dir / f"model.{key}.{suffix}")

        if isinstance(out_dir, str):
            out_dir_path = Path(out_dir)
        elif isinstance(out_dir, Path):
            out_dir_path = out_dir
        else:
            assert False
        for uid in self.cpp_lora_weights:
            if uid == "-1":
                continue

            all_weights = np.expand_dims(torch_to_numpy(self.cpp_lora_weights[uid]), 0)
            all_configs = np.expand_dims(torch_to_numpy(self.cpp_lora_config[uid]), 0)

            uid_path = out_dir_path / f"{uid}"
            uid_path.mkdir(parents=True, exist_ok=True)
            save_val(all_weights, uid_path, "lora_weights", tp_num=None, write_npy=True)
            save_val(all_configs, uid_path, "lora_config", tp_num=None, write_npy=True)

    def input_buffers(self, lora_uids, mapping: Mapping, num_layers: int):
        inputs = {}
        for layer_idx in mapping.pp_layers(num_layers):
            for lora_module in self.lora_target_modules + self.missing_qkv_modules:
                lora_ranks_ = []
                lora_ptrs_ = []
                for lora_uid in lora_uids:
                    lora_rank = 0
                    lora_ptrs = [0, 0, 0]

                    if lora_uid != "-1":
                        low_ranks = self.uid_to_low_ranks(lora_uid)

                        if (
                            layer_idx in low_ranks
                            and lora_module in low_ranks[layer_idx].keys()
                            and low_ranks[layer_idx][lora_module] != 0
                        ):
                            lora_rank = low_ranks[layer_idx][lora_module]
                            lora_ptrs = self.lora_weights_pointers_list[lora_uid][layer_idx][
                                lora_module
                            ]

                    lora_ranks_.append(lora_rank)
                    lora_ptrs_.append(lora_ptrs)

                inputs[f"{lora_module}_lora_ranks_{layer_idx}"] = torch.IntTensor(lora_ranks_)
                inputs[f"{lora_module}_lora_weights_pointers_{layer_idx}"] = torch.LongTensor(
                    lora_ptrs_
                )
        return inputs
