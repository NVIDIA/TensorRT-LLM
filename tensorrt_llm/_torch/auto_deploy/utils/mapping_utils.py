# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities for working with Mapping objects in auto_deploy.

This module provides:
1. MappingSerializer - Serialize/deserialize Mapping for @torch.library.custom_op
2. from_config() - Initialize Mapping from LlmArgs configuration
"""

from typing import List, Optional

from tensorrt_llm.mapping import Mapping

from .logger import ad_logger


class MappingSerializer:
    """Serializer for Mapping objects to/from list[int] for custom ops.

    **Why this class exists:**

    PyTorch's @torch.library.custom_op decorator has strict requirements on parameter
    types and does NOT support composite types like objects or Dict[str, Any]. It only
    supports primitives (int, float, bool, str), Tensors, and simple collections
    (List[Tensor], List[int], Optional[...]).

    The natural approach would be to use Mapping.to_dict() and pass a Dict[str, int],
    but this is not possible. Therefore, we serialize the Mapping into a flat list of
    integers (List[int]), which IS supported by @torch.library.custom_op.

    **Design:**

    This class encapsulates the serialization format as an implementation detail.
    The internal indices (_WORLD_SIZE, _TP_SIZE, etc.) are private and users only
    interact with the public serialize() and deserialize() methods.

    **Usage:**

        # In graph transformation (sharding.py)
        config = MappingSerializer.serialize(mapping, max_num_tokens=8192)
        # Pass config as List[int] to custom op

        # In custom op implementation (trtllm_moe.py)
        mapping = MappingSerializer.deserialize(config)
        # Now have full Mapping object with all fields

    **Compatibility:**

    The serialization format can be extended by adding new indices at the end
    without breaking existing code (forward compatibility).

    See Also:
        - PyTorch custom ops docs: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
        - GitHub issue requesting Dict support: https://github.com/pytorch/pytorch/issues/143466
    """

    # Internal indices for the serialized list (implementation details - do not use directly)
    _WORLD_SIZE = 0
    _RANK = 1
    _GPUS_PER_NODE = 2
    _CP_SIZE = 3
    _TP_SIZE = 4
    _PP_SIZE = 5
    _MOE_TP_SIZE = 6
    _MOE_TP_RANK = 7
    _MOE_CLUSTER_SIZE = 8
    _MOE_CLUSTER_RANK = 9
    _MOE_EP_SIZE = 10
    _MOE_EP_RANK = 11
    _ATTN_TP_SIZE = 12
    _ATTN_CP_SIZE = 13
    _CP_CONFIG = 14  # Note: cp_config can be None, we'll use 0 for None
    _ENABLE_ATTENTION_DP = 15  # bool -> int (0/1)
    _ENABLE_LM_HEAD_TP_IN_ADP = 16  # bool -> int (0/1)
    _MAX_NUM_TOKENS = 17  # Custom field for MoE workspace allocation
    _LENGTH = 18

    @classmethod
    def serialize(cls, mapping: Mapping, max_num_tokens: int = 0) -> List[int]:
        """Serialize Mapping to list[int] for custom ops.

        Args:
            mapping: The Mapping object to serialize (contains all distributed config)
            max_num_tokens: Maximum number of tokens for workspace allocation (MoE-specific)

        Returns:
            List of integers encoding all fields from Mapping, compatible with
            @torch.library.custom_op parameter requirements
        """
        config = [0] * cls._LENGTH
        config[cls._WORLD_SIZE] = mapping.world_size
        config[cls._RANK] = mapping.rank
        config[cls._GPUS_PER_NODE] = mapping.gpus_per_node
        config[cls._CP_SIZE] = mapping.cp_size
        config[cls._TP_SIZE] = mapping.tp_size
        config[cls._PP_SIZE] = mapping.pp_size
        config[cls._MOE_TP_SIZE] = mapping.moe_tp_size
        config[cls._MOE_TP_RANK] = mapping.moe_tp_rank
        config[cls._MOE_CLUSTER_SIZE] = mapping.moe_cluster_size
        config[cls._MOE_CLUSTER_RANK] = mapping.moe_cluster_rank
        config[cls._MOE_EP_SIZE] = mapping.moe_ep_size
        config[cls._MOE_EP_RANK] = mapping.moe_ep_rank
        config[cls._ATTN_TP_SIZE] = mapping.attn_tp_size
        config[cls._ATTN_CP_SIZE] = mapping.attn_cp_size
        # cp_config is Optional and can be None, empty dict, or int - encode all as 0 if not a valid int
        if mapping.cp_config is None or (
            isinstance(mapping.cp_config, dict) and not mapping.cp_config
        ):
            config[cls._CP_CONFIG] = 0
        elif isinstance(mapping.cp_config, int):
            config[cls._CP_CONFIG] = mapping.cp_config
        else:
            # Non-empty dict or other type - encode as 0 (not supported in serialization)
            config[cls._CP_CONFIG] = 0
        config[cls._ENABLE_ATTENTION_DP] = int(mapping.enable_attention_dp)
        config[cls._ENABLE_LM_HEAD_TP_IN_ADP] = int(mapping.enable_lm_head_tp_in_adp)
        config[cls._MAX_NUM_TOKENS] = max_num_tokens
        return config

    @classmethod
    def get_max_num_tokens(cls, config: Optional[List[int]] = None) -> int:
        """Extract max_num_tokens from serialized config.

        This is a separate method because max_num_tokens is not part of the
        Mapping object itself - it's an additional parameter for MoE workspace allocation.

        Args:
            config: Serialized mapping configuration from serialize()

        Returns:
            max_num_tokens value, or 0 if config is None or too short
        """
        if config is None or len(config) <= cls._MAX_NUM_TOKENS:
            return 0
        return config[cls._MAX_NUM_TOKENS]

    @classmethod
    def deserialize(cls, config: Optional[List[int]] = None) -> Mapping:
        """Reconstruct Mapping from serialized list[int].

        Args:
            config: Serialized mapping configuration from serialize().
                    If None, returns a Mapping with all default values.

        Returns:
            Mapping object with all fields restored from the serialized config,
            or a default Mapping if config is None.
        """
        # If config is None, return a default Mapping
        if config is None:
            return Mapping()

        def _get(idx: int, default: int = 1) -> int:
            """Safely extract value from config with default fallback."""
            return config[idx] if len(config) > idx else default

        def _get_bool(idx: int, default: bool = False) -> bool:
            """Extract boolean value (stored as 0/1)."""
            return bool(_get(idx, int(default)))

        # cp_config: 0 means None, otherwise use the value
        cp_config_val = _get(cls._CP_CONFIG, 0)
        cp_config = None if cp_config_val == 0 else cp_config_val

        return Mapping(
            world_size=_get(cls._WORLD_SIZE, 1),
            rank=_get(cls._RANK, 0),
            gpus_per_node=_get(cls._GPUS_PER_NODE, 8),
            cp_size=_get(cls._CP_SIZE, 1),
            tp_size=_get(cls._TP_SIZE, 1),
            pp_size=_get(cls._PP_SIZE, 1),
            moe_tp_size=_get(cls._MOE_TP_SIZE, 1),
            moe_cluster_size=_get(cls._MOE_CLUSTER_SIZE, 1),
            moe_ep_size=_get(cls._MOE_EP_SIZE, 1),
            attn_tp_size=_get(cls._ATTN_TP_SIZE, 1),
            attn_cp_size=_get(cls._ATTN_CP_SIZE, 1),
            cp_config=cp_config,
            enable_attention_dp=_get_bool(cls._ENABLE_ATTENTION_DP, False),
            enable_lm_head_tp_in_adp=_get_bool(cls._ENABLE_LM_HEAD_TP_IN_ADP, False),
        )

    @classmethod
    def from_config(cls, ad_config, world_size: int, rank: int) -> Mapping:
        """Initialize Mapping from LlmArgs configuration.

        This method extracts the distributed mapping configuration from
        ad_config.transforms['detect_sharding']['dist_mapping'] and creates
        a proper Mapping object.

        When enable_attention_dp=True, enforces 1D MoE parallelism:
        - dp_tp_only=False (default): EP-only (moe_ep_size=world_size, moe_tp_size=1)
        - dp_tp_only=True: TP-only (moe_tp_size=world_size, moe_ep_size=1)

        2D MoE parallelism (EP+TP) is NOT supported with attention-DP.

        Args:
            ad_config: LlmArgs configuration object containing transform configs
            world_size: Total number of processes in the distributed setup
            rank: Current process rank

        Returns:
            Mapping object initialized from config, with fallback to defaults

        Example:
            >>> mapping = MappingSerializer.from_config(ad_config, world_size=8, rank=0)
            >>> print(mapping.moe_ep_size)  # Access MoE expert parallelism size
        """
        # Extract config from transforms
        sharding_config = ad_config.transforms.get("detect_sharding", {})
        dist_mapping_config = sharding_config.get("dist_mapping", {})
        enable_attention_dp = sharding_config.get("enable_attention_dp", False)
        dp_tp_only = sharding_config.get("dp_tp_only", False)

        # Determine MoE parallelism dimensions
        if enable_attention_dp:
            # EP + TP 2D parallelism is currently NOT supported with attention-DP.
            # Enforce 1D parallelism to avoid broken 2D case.
            if dp_tp_only:
                # TP-only: all experts on all GPUs, use allgather + reducescatter
                moe_tp_size = world_size
                moe_ep_size = 1
                ad_logger.info(
                    f"Attention-DP with TP-only MoE: moe_tp_size={moe_tp_size}, moe_ep_size={moe_ep_size}"
                )
            else:
                # EP-only: experts sharded across GPUs, use all-to-all dispatch/combine
                moe_ep_size = world_size
                moe_tp_size = 1
                ad_logger.info(
                    f"Attention-DP with EP-only MoE: moe_ep_size={moe_ep_size}, moe_tp_size={moe_tp_size}"
                )
        else:
            # No attention-DP: use dist_mapping config or defaults
            moe_tp_size = dist_mapping_config.get("moe_tp", 1)
            moe_ep_size = dist_mapping_config.get("moe_ep", world_size)

        # Create Mapping with proper distributed configuration
        try:
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=dist_mapping_config.get("tp", world_size),
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                moe_cluster_size=dist_mapping_config.get("moe_cluster", 1),
                enable_attention_dp=enable_attention_dp,
            )
        except ValueError as e:
            ad_logger.warning(f"Invalid parallel grid config: {e}")
            ad_logger.warning("Defaulting to TP-only sharding (EP only for MoE)")
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=world_size,
                moe_tp_size=1,
                moe_ep_size=world_size,
                moe_cluster_size=1,
            )

        return mapping

    @classmethod
    def print_grid(cls, mapping: Mapping) -> str:
        """Pretty print the grid of the mapping."""
        return (
            "process grid: [TP, MoE_TP, MoE_EP] = "
            + f"[{mapping.tp_size}, {mapping.moe_tp_size}, {mapping.moe_ep_size}]"
        )

    @classmethod
    def print_rank(cls, mapping: Mapping) -> str:
        """Pretty print the rank of the mapping."""
        return f"rank: [{mapping.rank}, {mapping.moe_tp_rank}, {mapping.moe_ep_rank}]"
