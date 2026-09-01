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
"""Runtime LoRA adapter management, and the entry points that build it."""

import itertools
import json
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._utils import release_gc, str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import get_model_path, load_state_dict

from .config import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
    get_missing_qkv_modules_from_lora_modules,
)
from .loaders import (
    NEMO_SUPPORTED_LORA_MODULES,
    HfLoraLoader,
    NemoLoraLoader,
    find_nemo_files,
    get_all_hf_lora_weights,
    get_all_nemo_lora_weights,
    invert_module_mapping,
    unpack_nemo_weights,
)

if TYPE_CHECKING:
    from tensorrt_llm.runtime import ModelConfig


logger = logging.getLogger(__name__)


_FP8_LORA_TMA_ALIGNMENT = 16


@lru_cache(maxsize=1)
def _warn_native_fp8_lora_capability_query_unavailable() -> None:
    logger.warning(
        "Native FP8 LoRA capability query is unavailable; adapter weights "
        "will fall back to the model compute dtype. Check that the "
        "TensorRT-LLM libraries match the Python package and are loaded."
    )


def _native_fp8_lora_kernels_available(device_capability: Tuple[int, int]) -> bool:
    kernel_support_query = getattr(torch.ops.trtllm, "lora_grouped_gemm_supports_fp8", None)
    if kernel_support_query is None:
        _warn_native_fp8_lora_capability_query_unavailable()
        return False
    major, minor = device_capability
    return kernel_support_query(major * 10 + minor)


def supports_native_fp8_lora(device_capability: Tuple[int, int]) -> bool:
    """Return whether compiled native FP8 LoRA kernels support a CUDA capability."""
    return _native_fp8_lora_kernels_available(device_capability)


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


def _validate_fp8_lora_alignment(
    *,
    rank: int,
    input_size: int,
    output_size: int,
    layer_idx: int,
    lora_module: str,
) -> None:
    dimensions = {
        "rank": rank,
        "input size": input_size,
        "output size": output_size,
    }
    misaligned_dimensions = {
        name: size for name, size in dimensions.items() if size % _FP8_LORA_TMA_ALIGNMENT != 0
    }
    if misaligned_dimensions:
        formatted_dimensions = ", ".join(
            f"{name}={size}" for name, size in misaligned_dimensions.items()
        )
        raise ValueError(
            f"FP8 LoRA weights on SM90/SM100 require rank, input size, and output size "
            f"to be multiples of {_FP8_LORA_TMA_ALIGNMENT} for 128-bit TMA alignment. "
            f"Layer {layer_idx} module '{lora_module}' has {formatted_dimensions}. "
            f"Use aligned adapter dimensions or non-FP8 LoRA weights."
        )


@dataclass
class LoraModelConfig:
    lora_target_modules: list[str]
    trtllm_modules_to_hf_modules: dict[str, str]
    hidden_size: int
    dtype: str
    swap_gate_up_proj_lora_b_weight: bool = True


def load_torch_hf_lora(lora_config: LoraConfig) -> Optional[torch.dtype]:
    """Load an HF LoRA checkpoint for the PyTorch workflow.

    Populates lora_config (trtllm_modules_to_hf_modules and inferred
    lora_target_modules) from the HF adapter directory. The actual weights are
    loaded later by LoraManager when requests arrive with LoRA UIDs.

    Returns:
        The common LoRA weight dtype, or ``None`` when no homogeneous LoRA
        weights are present.
    """
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
    return lora_loader.get_lora_dtype()


def load_torch_nemo_lora(lora_config: LoraConfig) -> Optional[torch.dtype]:
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

    Returns:
        ``None`` because NeMo LoRA weights use the model compute dtype.

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
    return None


def load_torch_lora(lora_config: LoraConfig) -> Optional[torch.dtype]:
    """Load LoRA checkpoint for PyTorch workflow.

    This function routes to the appropriate loader based on lora_ckpt_source.

    Args:
        lora_config: LoRA configuration with lora_ckpt_source set to "hf" or "nemo"

    Returns:
        The configured adapter's homogeneous dense weight dtype, if available.

    Raises:
        ValueError: If lora_ckpt_source is not supported
    """
    if lora_config.lora_ckpt_source == "nemo":
        return load_torch_nemo_lora(lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        return load_torch_hf_lora(lora_config)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}. "
            f"Supported sources: 'hf', 'nemo'"
        )


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
        "shared_expert_h_to_4h": 19,
        "shared_expert_4h_to_h": 20,
        "shared_expert_gate": 21,
        "mamba_in_proj": 22,
        "mamba_out_proj": 23,
        "moe_latent_fc1": 24,
        "moe_latent_fc2": 25,
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
        # When cpp_peft_cache_manager is provided (PyTorch backend), the C++
        # PeftCacheManager manages its own GPU cache with proper eviction.
        # The Python-side GPU tensors are only needed by the legacy TRT backend
        # which reads raw data_ptr() values via input_buffers().
        self._retain_device_tensors = cpp_peft_cache_manager is None
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
                        if self._retain_device_tensors:
                            self._lora_weights_pointers_list[uid][layer_idx][lora_module] = [
                                t_in.data_ptr(),
                                t_out.data_ptr(),
                                0,
                            ]
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
            lora_model = load_state_dict(get_model_path(model_dir, "adapter_model"))
            if lora_model is None:
                raise ValueError(f"Failed to load adapter_model from {model_dir}")
            lora_model = preprocess_lora_weights(lora_model, model_config)
            all_weights = get_all_hf_lora_weights(lora_model, hf_modules, component)
            rank = int(hf_config["r"])
            rs_lora = bool(hf_config.get("use_rslora", False))
            model_dtype = str_dtype_to_torch(model_config.dtype)
            supports_native_fp8 = supports_native_fp8_lora(torch.cuda.get_device_capability())

            def get_output_dtype(module_weights):
                if _is_moe_module_weights(module_weights):
                    output_dtypes = {
                        weights["out"].dtype
                        for weights in module_weights.values()
                        if "out" in weights
                    }
                    return next(iter(output_dtypes)) if len(output_dtypes) == 1 else model_dtype
                return module_weights["out"].dtype if "out" in module_weights else model_dtype

            def uses_native_fp8(module_weights):
                return (
                    get_output_dtype(module_weights) == torch.float8_e4m3fn and supports_native_fp8
                )

            for layer_weights in all_weights.values():
                placeholder_dtype = next(
                    (
                        weights["out"].dtype
                        for weights in layer_weights.values()
                        if not _is_moe_module_weights(weights) and "out" in weights
                    ),
                    model_dtype,
                )
                for lora_module in self.missing_qkv_modules:
                    hf_module = model_config.trtllm_modules_to_hf_modules[lora_module]
                    if isinstance(hf_module, list):
                        hf_module = hf_module[0]
                    layer_weights[hf_module] = {
                        "in": torch.zeros(rank, model_config.hidden_size, dtype=placeholder_dtype),
                        "out": torch.zeros(model_config.hidden_size, rank, dtype=placeholder_dtype),
                    }

            cache_dtypes = {
                torch.float8_e4m3fn if uses_native_fp8(module_weights) else model_dtype
                for layer_weights in all_weights.values()
                for hf_module, module_weights in layer_weights.items()
                if hf_modules_to_trtllm_modules[hf_module] in self.lora_target_modules
            }
            if len(cache_dtypes) > 1:
                dtype_names = ", ".join(sorted(str(dtype) for dtype in cache_dtypes))
                raise ValueError(
                    "A LoRA adapter must use one PEFT cache dtype across all modules; "
                    f"{model_dir} requires {dtype_names}. Mixing native FP8 modules "
                    "with compute-dtype modules is not supported."
                )

            cpp_lora_weights = []
            cpp_lora_config = []
            uid_to_low_ranks = {}
            lora_weights_pointers = {}
            retained_lora_weights = []
            for layer_idx in sorted(all_weights.keys()):
                layer_weights = all_weights[layer_idx]
                uid_to_low_ranks[layer_idx] = {}
                lora_weights_pointers[layer_idx] = {}

                for hf_module, module_weights in layer_weights.items():
                    lora_module = hf_modules_to_trtllm_modules[hf_module]
                    if lora_module not in self.lora_target_modules:
                        warnings.warn(
                            f"LoRA module '{lora_module}' not in target modules {self.lora_target_modules}, skipping."
                        )
                        uid_to_low_ranks[layer_idx][lora_module] = 0
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
                    # TODO: Enable SM120/SM121 after validating the native FP8 LoRA kernel there.
                    use_fp8_kernel = uses_native_fp8(module_weights)
                    if use_fp8_kernel:
                        if t_in.dtype != t_out.dtype:
                            raise ValueError(
                                "FP8 LoRA input and output weights must have the same dtype; "
                                f"got {t_in.dtype} and {t_out.dtype} for layer {layer_idx} "
                                f"module {lora_module}"
                            )
                        _validate_fp8_lora_alignment(
                            rank=effective_rank,
                            input_size=t_in.shape[-1],
                            output_size=t_out.shape[-2],
                            layer_idx=layer_idx,
                            lora_module=lora_module,
                        )
                        if is_dora:
                            raise NotImplementedError(
                                "DoRA is not supported with FP8 LoRA weights on SM90/SM100"
                            )

                    t_in = t_in.cuda().contiguous()
                    t_out = t_out.cuda().contiguous()
                    if is_dora and t_mag is not None:
                        t_mag = t_mag.cuda().contiguous()

                    if rs_lora:
                        scale = float(hf_config["lora_alpha"]) / np.sqrt(effective_rank)
                    else:
                        scale = float(hf_config["lora_alpha"]) / effective_rank

                    if use_fp8_kernel:
                        # Keep weights in FP8 for the native SM90/SM100 kernel.
                        # FP8 has no scalar multiply, so scale through BF16.
                        fp8_max = torch.finfo(t_out.dtype).max
                        t_out = (
                            (t_out.to(torch.bfloat16) * scale)
                            .clamp(-fp8_max, fp8_max)
                            .to(t_out.dtype)
                        )
                    else:
                        # Other architectures require the model compute dtype.
                        t_in = t_in.to(model_dtype)
                        t_out = t_out.to(model_dtype)
                        t_out = t_out * scale
                        if is_dora and t_mag is not None:
                            t_mag = t_mag.to(model_dtype)

                    uid_to_low_ranks[layer_idx][lora_module] = effective_rank
                    if self._retain_device_tensors:
                        lora_weights_pointers[layer_idx][lora_module] = [
                            t_in.data_ptr(),
                            t_out.data_ptr(),
                            t_mag.data_ptr() if (is_dora and t_mag is not None) else 0,
                        ]
                        retained_lora_weights.append(t_in)
                        retained_lora_weights.append(t_out)
                        if is_dora and t_mag is not None:
                            retained_lora_weights.append(t_mag)

                    t_in_cpu = t_in.flatten().cpu()
                    t_out_cpu = t_out.flatten().cpu()
                    weights_to_concat = [t_in_cpu, t_out_cpu]

                    if is_dora and t_mag is not None:
                        t_mag_cpu = t_mag.flatten().cpu()
                        weights_to_concat.append(t_mag_cpu)

                    cpp_lora_weights.append(torch.cat(weights_to_concat))
                    cpp_lora_config.append(
                        torch.tensor(
                            [self.LORA_MODULE_IDS[lora_module], layer_idx, effective_rank, is_dora],
                            dtype=torch.int32,
                        )
                    )

            max_weight_size = max(w.size(0) for w in cpp_lora_weights)
            packed_lora_weights = torch.stack(
                [
                    torch.nn.functional.pad(w, (0, max_weight_size - w.size(0)))
                    for w in cpp_lora_weights
                ]
            )
            packed_lora_config = torch.stack(cpp_lora_config)

            self._cpp_lora_weights[uid] = packed_lora_weights
            self._cpp_lora_config[uid] = packed_lora_config
            self._lora_uid_to_low_ranks[uid] = uid_to_low_ranks
            self._lora_weights_pointers_list[uid] = lora_weights_pointers
            self._lora_weights.extend(retained_lora_weights)

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
