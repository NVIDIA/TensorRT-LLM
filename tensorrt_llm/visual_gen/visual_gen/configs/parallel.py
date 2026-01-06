# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def init_dist(device_type: str = "cuda"):
    # Validate device type
    if device_type not in ["cuda", "cpu"]:
        error_msg = f"Unsupported device type: {device_type}"
        logger.error(f"{error_msg}")
        raise ValueError(error_msg)

    if not dist.is_initialized():
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        RANK = int(os.environ.get("RANK", 0))
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        if WORLD_SIZE > 1:
            logger.debug(f"Initializing distributed process group with backend: {device_type}")
            if device_type == "cuda":
                dist.init_process_group(
                    backend="nccl", rank=RANK, world_size=WORLD_SIZE, device_id=torch.device(f"cuda:{LOCAL_RANK}")
                )
            elif device_type == "cpu":
                dist.init_process_group(backend="gloo")
            else:
                raise ValueError(f"Unsupported device type: {device_type}")
            logger.debug(f"Distributed process group initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
        else:
            logger.debug("No distributed process group needed")


@dataclass
class BaseParallelConfig:
    """Base configuration class for different types of parallelism with singleton pattern.

    Attributes:
        tp_size (int): Tensor parallel degree, defaults to 1
        ulysses_size (int): Ulysses parallel degree, defaults to 1
        ring_size (int): Ring attention parallel degree, defaults to 1
        cp_size (int): Context parallel degree, defaults to 1
        dp_size (int): Data parallel degree, defaults to 1
        cfg_size (int): Classifier-free guidance parallel degree, defaults to 1
    """

    _instances: ClassVar[Dict[Type["BaseParallelConfig"], "BaseParallelConfig"]] = {}
    _initialized: ClassVar[Dict[Type["BaseParallelConfig"], bool]] = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._initialized[cls] = False
            logger.debug(f"[{cls.__name__}] Created new {cls.__name__} instance")
        return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        if not self._initialized[self.__class__]:
            super().__init__(*args, **kwargs)
            self._initialized[self.__class__] = True
            logger.debug(f"[{self.__class__.__name__}] Initialized {self.__class__.__name__}")
            # parallel config to store the parallel config
            self._tp_size = 1
            self._ulysses_size = 1
            self._ring_size = 1
            self._cp_size = 1
            self._dp_size = 1
            self._cfg_size = 1
            self._redundant_size = 1
            self._fsdp_size = 1
            # device mesh to store the device mesh
            self._device_mesh = None
            # fsdp device mesh to store the fsdp device mesh
            self._fsdp_device_mesh = None
            # cached device mesh to avoid re-creating device mesh
            self._cached_device_mesh = {}
            # Initialize device mesh after validation
            self.set_device_mesh()
            self.set_fsdp_device_mesh()

    def __del__(self):
        """Cleanup distributed process group when the config is destroyed."""
        if hasattr(self, "_device_mesh") and self._device_mesh is not None:
            if dist.is_initialized():
                logger.info(f"[{self.__class__.__name__}] Cleaning up distributed process group")
                dist.destroy_process_group()

    def set_device_mesh(self, device_type: str = "cuda") -> torch.distributed.DeviceMesh:
        """Initialize the device mesh for distributed training.

        Args:
            device_type (str): The type of device to use, defaults to "cuda"

        Returns:
            torch.distributed.DeviceMesh: The initialized device mesh

        Note:
            The device mesh is created in the order: dp -> cfg -> ring -> ulysses -> tp
        """
        if self._device_mesh is not None:
            logger.debug(f"[{self.__class__.__name__}] Device mesh already initialized")
            return self._device_mesh

        total_parallel_size = self.get_total_parallel_size()
        world_size = self.get_world_size()
        if world_size % total_parallel_size != 0:
            raise ValueError(
                f"World size ({world_size}) is not divisible by total parallel size ({total_parallel_size})"
            )

        logger.debug(
            f"[{self.__class__.__name__}] Setting up device mesh with total parallel size: {total_parallel_size}"
        )

        if total_parallel_size == 1:
            logger.debug(f"[{self.__class__.__name__}] No parallelism needed, skipping device mesh setup")
            self._device_mesh = None
            return self._device_mesh

        # Create mesh dimensions in the specified order
        mesh_dims = []
        mesh_sizes = []

        self._redundant_size = 1
        if world_size != total_parallel_size:
            # Note that this redundant dimension is not used for communication,
            # it is only used to ensure parallel when world_size != total_parallel_size
            mesh_dims.append("redundant")
            self._redundant_size = world_size // total_parallel_size
            mesh_sizes.append(self._redundant_size)
            logger.debug(
                f"[{self.__class__.__name__}] Added redundant dimension: {self._redundant_size}, world_size={world_size}, total_parallel_size={total_parallel_size}"
            )

        if str(self) in self._cached_device_mesh:
            self._device_mesh = self._cached_device_mesh[str(self)]
            return self._device_mesh

        # Add dimensions in order: dp -> cfg -> ring -> ulysses -> tp
        if self._dp_size > 1:
            mesh_dims.append("dp")
            mesh_sizes.append(self._dp_size)
            logger.debug(f"[{self.__class__.__name__}] Added DP dimension: {self._dp_size}")
        if self._cfg_size > 1:
            mesh_dims.append("cfg")
            mesh_sizes.append(self._cfg_size)
            logger.debug(f"[{self.__class__.__name__}] Added CFG dimension: {self._cfg_size}")
        if self._ring_size > 1:
            mesh_dims.append("ring")
            mesh_sizes.append(self._ring_size)
            logger.debug(f"[{self.__class__.__name__}] Added Ring dimension: {self._ring_size}")
        if self._cp_size > 1:
            mesh_dims.append("cp")
            mesh_sizes.append(self._cp_size)
            logger.debug(f"[{self.__class__.__name__}] Added CP dimension: {self._cp_size}")
        if self._ulysses_size > 1:
            mesh_dims.append("ulysses")
            mesh_sizes.append(self._ulysses_size)
            logger.debug(f"[{self.__class__.__name__}] Added Ulysses dimension: {self._ulysses_size}")
        if self._tp_size > 1:
            mesh_dims.append("tp")
            mesh_sizes.append(self._tp_size)
            logger.debug(f"[{self.__class__.__name__}] Added TP dimension: {self._tp_size}")

        if not mesh_dims:
            logger.debug(f"[{self.__class__.__name__}] No mesh dimensions needed")
            self._device_mesh = None
        else:
            # Initialize the device mesh with the specified dimensions
            logger.info(f"[{self.__class__.__name__}] Creating device mesh: dims={mesh_dims}, sizes={mesh_sizes}")
            self._device_mesh = init_device_mesh(device_type, tuple(mesh_sizes), mesh_dim_names=tuple(mesh_dims))
            logger.info(f"[{self.__class__.__name__}] Device mesh created successfully")

        if str(self) not in self._cached_device_mesh:
            self._cached_device_mesh[str(self)] = self._device_mesh

        return self._device_mesh

    def set_fsdp_device_mesh(self, device_type: str = "cuda") -> torch.distributed.DeviceMesh:
        """Initialize the FSDP device mesh.

        Args:
            device_type (str): The type of device to use, defaults to "cuda"
        """
        if self._fsdp_device_mesh is not None:
            logger.debug(f"[{self.__class__.__name__}] FSDP device mesh already initialized")
            return self._fsdp_device_mesh

        if self._fsdp_size <= 1:
            logger.debug(f"[{self.__class__.__name__}] FSDP size is 1, skipping FSDP device mesh setup")
            self._fsdp_device_mesh = None
            return self._fsdp_device_mesh
        world_size = self.get_world_size()
        if world_size % self._fsdp_size != 0:
            raise ValueError(f"World size ({world_size}) is not divisible by FSDP size ({self._fsdp_size})")
        hsdp_size = world_size // self._fsdp_size
        # only ranks in same fsdp group need to communicate with each other,
        # ranks in different fsdp group don't need to communicate with each other.
        # if gpu memory is sufficient, this might be helpful to reduce communication overhead.
        self._fsdp_device_mesh = init_device_mesh(
            device_type, (hsdp_size, self._fsdp_size), mesh_dim_names=("hsdp", "fsdp")
        )
        logger.info(
            f"[{self.__class__.__name__}] FSDP device mesh created successfully: hsdp_size={hsdp_size}, fsdp_size={self._fsdp_size}"
        )

        return self._fsdp_device_mesh

    @classmethod
    def device_mesh(cls) -> Optional[torch.distributed.DeviceMesh]:
        """Get the current device mesh.

        Returns:
            Optional[torch.distributed.DeviceMesh]: The current device mesh, or None if not initialized
        """
        instance = cls.get_instance()
        return instance._device_mesh

    @classmethod
    def fsdp_device_mesh(cls) -> Optional[torch.distributed.DeviceMesh]:
        """Get the FSDP device mesh.

        Returns:
            Optional[torch.distributed.DeviceMesh]: The FSDP device mesh, or None if not initialized
        """
        instance = cls.get_instance()
        return instance._fsdp_device_mesh

    @classmethod
    def get_group(cls, group_name: str) -> Optional[torch.distributed.ProcessGroup]:
        """Get the process group.

        Args:
            group_name (str): The name of the process group
        """
        instance = cls.get_instance()
        if instance._device_mesh is None:
            return None
        return instance._device_mesh.get_group(group_name)

    @classmethod
    def dp_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the data parallel process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The data parallel process group, or None if dp_size=1
        """
        instance = cls.get_instance()
        if instance._dp_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("dp")
        logger.debug(f"[{cls.__name__}] Retrieved DP group: {group}")
        return group

    @classmethod
    def cfg_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the classifier-free guidance process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The cfg process group, or None if cfg_size=1
        """
        instance = cls.get_instance()
        if instance._cfg_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("cfg")
        logger.debug(f"[{cls.__name__}] Retrieved CFG group: {group}")
        return group

    @classmethod
    def ring_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the ring attention process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The ring attention process group, or None if ring_size=1
        """
        instance = cls.get_instance()
        if instance._ring_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("ring")
        logger.debug(f"[{cls.__name__}] Retrieved Ring group: {group}")
        return group

    @classmethod
    def cp_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the context parallel process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The context parallel process group, or None if cp_size=1
        """
        instance = cls.get_instance()
        if instance._cp_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("cp")
        logger.debug(f"[{cls.__name__}] Retrieved CP group: {group}")
        return group

    @classmethod
    def ulysses_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the Ulysses process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The Ulysses process group, or None if ulysses_size=1
        """
        instance = cls.get_instance()
        if instance._ulysses_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("ulysses")
        logger.debug(f"[{cls.__name__}] Retrieved Ulysses group: {group}")
        return group

    @classmethod
    def tp_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the tensor parallel process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The tensor parallel process group, or None if tp_size=1
        """
        instance = cls.get_instance()
        if instance._tp_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("tp")
        logger.debug(f"[{cls.__name__}] Retrieved TP group: {group}")
        return group

    @classmethod
    def fsdp_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the FSDP process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The FSDP process group, or None if fsdp_size=1
        """
        instance = cls.get_instance()
        if instance._fsdp_size <= 1 or instance._fsdp_device_mesh is None:
            return None
        group = cls.get_group("fsdp")
        logger.debug(f"[{cls.__name__}] Retrieved FSDP group: {group}")
        return group

    @classmethod
    def all_groups(cls) -> Dict[str, Optional[torch.distributed.ProcessGroup]]:
        """Get all process groups.

        Returns:
            Dict[str, Optional[torch.distributed.ProcessGroup]]: Dictionary containing all process groups
        """
        groups = {
            "dp": cls.dp_group(),
            "cfg": cls.cfg_group(),
            "ring": cls.ring_group(),
            "cp": cls.cp_group(),
            "ulysses": cls.ulysses_group(),
            "tp": cls.tp_group(),
            "fsdp": cls.fsdp_group(),
        }
        logger.debug(f"[{cls.__name__}] Retrieved all process groups: {list(groups.keys())}")
        return groups

    @classmethod
    def global_rank(cls) -> int:
        """Get the global rank.

        Returns:
            int: global rank (0 to world_size-1)
        """
        if dist.is_initialized():
            return dist.get_rank()
        else:
            return 0

    @classmethod
    def get_local_rank(cls, group_name: str) -> int:
        """Get the local rank.

        Returns:
            int: local rank (0 to world_size-1)
        """
        instance = cls.get_instance()
        if instance._device_mesh is None:
            raise RuntimeError("Device mesh not initialized")
        if group_name not in instance._device_mesh.mesh_dim_names:
            raise RuntimeError(f"Group {group_name} not found in device mesh")
        return instance._device_mesh.get_local_rank(group_name)

    @classmethod
    def dp_rank(cls) -> int:
        """Get the local rank in the data parallel group.

        Returns:
            int: Local rank in the data parallel group (0 to dp_size-1)
        """
        instance = cls.get_instance()
        if instance._dp_size <= 1:
            return 0
        rank = cls.get_local_rank("dp")
        logger.debug(f"[{cls.__name__}] DP rank: {rank}")
        return rank

    @classmethod
    def cfg_rank(cls) -> int:
        """Get the local rank in the classifier-free guidance group.

        Returns:
            int: Local rank in the cfg group (0 to cfg_size-1)
        """
        instance = cls.get_instance()
        if instance._cfg_size <= 1:
            return 0
        rank = cls.get_local_rank("cfg")
        logger.debug(f"[{cls.__name__}] CFG rank: {rank}")
        return rank

    @classmethod
    def ring_rank(cls) -> int:
        """Get the local rank in the ring attention group.

        Returns:
            int: Local rank in the ring attention group (0 to ring_size-1)
        """
        instance = cls.get_instance()
        if instance._ring_size <= 1:
            return 0
        rank = cls.get_local_rank("ring")
        logger.debug(f"[{cls.__name__}] Ring rank: {rank}")
        return rank

    @classmethod
    def ring_ranks(cls) -> List[int]:
        """Get all the local ranks in the ring attention group."""
        instance = cls.get_instance()
        return instance._device_mesh["ring"].mesh.flatten().tolist()

    @classmethod
    def cp_rank(cls) -> int:
        """Get the local rank in the context parallel group.

        Returns:
            int: Local rank in the context parallel group (0 to cp_size-1)
        """
        instance = cls.get_instance()
        if instance._cp_size <= 1:
            return 0
        rank = cls.get_local_rank("cp")
        logger.debug(f"[{cls.__name__}] CP rank: {rank}")
        return rank

    @classmethod
    def ulysses_rank(cls) -> int:
        """Get the local rank in the Ulysses group.

        Returns:
            int: Local rank in the Ulysses group (0 to ulysses_size-1)
        """
        instance = cls.get_instance()
        if instance._ulysses_size <= 1:
            return 0
        rank = cls.get_local_rank("ulysses")
        logger.debug(f"[{cls.__name__}] Ulysses rank: {rank}")
        return rank

    @classmethod
    def ulysses_ranks(cls) -> List[int]:
        """Get all the local ranks in the Ulysses group."""
        instance = cls.get_instance()
        return instance._device_mesh["ulysses"].mesh.flatten().tolist()

    @classmethod
    def tp_rank(cls) -> int:
        """Get the local rank in the tensor parallel group.

        Returns:
            int: Local rank in the tensor parallel group (0 to tp_size-1)
        """
        instance = cls.get_instance()
        if instance._tp_size <= 1:
            return 0
        rank = cls.get_local_rank("tp")
        logger.debug(f"[{cls.__name__}] TP rank: {rank}")
        return rank

    @classmethod
    def all_ranks(cls) -> Dict[str, int]:
        """Get local ranks in all parallel groups.

        Returns:
            Dict[str, int]: Dictionary containing local ranks in all parallel groups
        """
        return {
            "dp": cls.dp_rank(),
            "cfg": cls.cfg_rank(),
            "ring": cls.ring_rank(),
            "cp": cls.cp_rank(),
            "ulysses": cls.ulysses_rank(),
            "tp": cls.tp_rank(),
        }

    @classmethod
    def get_instance(cls) -> "BaseParallelConfig":
        """Get the singleton instance of parallel configuration.

        Returns:
            BaseParallelConfig: The singleton instance of the configuration

        Note:
            This method ensures that each subclass has its own singleton instance.
            For example, VAEParallelConfig.get_instance() and DiTParallelConfig.get_instance()
            will return different instances.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._initialized[cls] = False
            # Initialize the instance
            instance = cls._instances[cls]
            instance.__init__()
        return cls._instances[cls]

    @classmethod
    def clear_instance(cls) -> None:
        """Clear the singleton instance of the configuration.

        This method is mainly used for testing purposes.
        """
        if cls in cls._instances:
            del cls._instances[cls]
        if cls in cls._initialized:
            del cls._initialized[cls]

    @classmethod
    @torch.compiler.disable
    def set_config(
        cls,
        tp_size: int = 1,
        ulysses_size: int = 1,
        ring_size: int = 1,
        cp_size: int = 1,
        dp_size: int = 1,
        cfg_size: int = 1,
        fsdp_size: int = 1,
    ) -> None:
        """Set the configuration values for parallelism and reinitialize device mesh.

        Args:
            tp_size: Tensor parallel degree, defaults to 1
            ulysses_size: Ulysses parallel degree, defaults to 1
            ring_size: Ring attention parallel degree, defaults to 1
            cp_size: Context parallel degree, defaults to 1
            dp_size: Data parallel degree, defaults to 1
            cfg_size: Classifier-free guidance parallel degree, defaults to 1
            fsdp_size: Fully sharded data parallel degree, defaults to 1
        Raises:
            ValueError: If parallel configuration is invalid
        """
        # Validate input values first
        if not isinstance(tp_size, int) or tp_size < 1:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")
        if not isinstance(ulysses_size, int) or ulysses_size < 1:
            raise ValueError(f"ulysses_size must be a positive integer, got {ulysses_size}")
        if not isinstance(ring_size, int) or ring_size < 1:
            raise ValueError(f"ring_size must be a positive integer, got {ring_size}")
        if not isinstance(cp_size, int) or cp_size < 1:
            raise ValueError(f"cp_size must be a positive integer, got {cp_size}")
        if not isinstance(dp_size, int) or dp_size < 1:
            raise ValueError(f"dp_size must be a positive integer, got {dp_size}")
        if not isinstance(cfg_size, int) or cfg_size < 1:
            raise ValueError(f"cfg_size must be a positive integer, got {cfg_size}")

        # Special validation for cfg_size
        if cfg_size not in (1, 2):
            raise ValueError(f"cfg_size must be either 1 or 2, got {cfg_size}")

        if cp_size > 1:
            assert (
                ring_size == 1 and ulysses_size == 1
            ), f"cp_size > 1 is only supported when ring_size and ulysses_size are 1, but got ring_size={ring_size}, ulysses_size={ulysses_size}"

        # Check total parallel size before updating configuration
        total_size = tp_size * ulysses_size * ring_size * cp_size * dp_size * cfg_size
        if total_size > torch.cuda.device_count():
            raise ValueError(
                f"Total parallel size ({total_size}) exceeds available GPU count ({torch.cuda.device_count()})"
            )

        instance = cls.get_instance()

        # Update configuration
        instance._tp_size = tp_size
        instance._ulysses_size = ulysses_size
        instance._ring_size = ring_size
        instance._cp_size = cp_size
        instance._dp_size = dp_size
        instance._cfg_size = cfg_size
        if total_size > 1:
            # Reinitialize device mesh
            instance._device_mesh = None
            instance.set_device_mesh()
        instance._fsdp_size = fsdp_size
        if fsdp_size > 1:
            instance._fsdp_device_mesh = None
            instance.set_fsdp_device_mesh()
        instance.check_parallel_size()

    def get_world_size(self) -> int:
        """Get the world size.

        Returns:
            int: World size
        """
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def check_parallel_size(self) -> bool:
        """Check if the total parallel size matches the world size.

        Returns:
            bool: True if the total parallel size matches the world size, False otherwise
        """
        total_parallel_size = self.get_total_parallel_size()
        is_valid = True
        # check dp, tp, ulysses, ring, cfg
        if total_parallel_size == 1:
            logger.debug(f"[{self.__class__.__name__}] Total parallel size is 1, skipping check")
        elif total_parallel_size * self._redundant_size != self.get_world_size():
            logger.info(
                f"[{self.__class__.__name__}] Total parallel size {total_parallel_size} * redundant size {self._redundant_size} does not match world size {self.get_world_size()}"
            )
            is_valid = False
        # check fsdp
        if self._fsdp_size == 1:
            logger.debug(f"[{self.__class__.__name__}] FSDP size is 1, skipping check")
        elif self.get_world_size() % self._fsdp_size != 0:
            is_valid = False
        if not is_valid:
            par_size = f"DP: {self._dp_size}, TP: {self._tp_size}, ULYSSES: {self._ulysses_size}, RING: {self._ring_size}, CP: {self._cp_size}, CFG: {self._cfg_size}, FSDP: {self._fsdp_size}, REDUNDANT: {self._redundant_size}"
            raise ValueError(f"Parallel size {par_size} does not match world size ({self.get_world_size()})")
        return is_valid

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: Dictionary containing all parallel configurations
        """
        return {
            "tp_size": self._tp_size,
            "ulysses_size": self._ulysses_size,
            "ring_size": self._ring_size,
            "cp_size": self._cp_size,
            "dp_size": self._dp_size,
            "cfg_size": self._cfg_size,
            "fsdp_size": self._fsdp_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseParallelConfig":
        """Create a BaseParallelConfig instance from a dictionary.

        Args:
            config_dict (dict): Dictionary containing parallel configurations

        Returns:
            BaseParallelConfig: New instance with values from the dictionary
        """
        return cls(**config_dict)

    @classmethod
    def get_total_cp_size(cls) -> int:
        """Calculate the total context parallel size by multiplying all context parallel degrees.

        Returns:
            int: Total parallel size
        """
        return cls.ulysses_size() * cls.ring_size() * cls.cp_size()

    @classmethod
    def get_total_parallel_size(cls) -> int:
        """Calculate the total parallel size by multiplying all parallel degrees.

        Returns:
            int: Total parallel size
        """
        return cls.tp_size() * cls.ulysses_size() * cls.ring_size() * cls.cp_size() * cls.dp_size() * cls.cfg_size()

    def __str__(self) -> str:
        """String representation of the parallel configuration."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  tp_size={self._tp_size},\n"
            f"  ulysses_size={self._ulysses_size},\n"
            f"  ring_size={self._ring_size},\n"
            f"  cp_size={self._cp_size},\n"
            f"  dp_size={self._dp_size},\n"
            f"  cfg_size={self._cfg_size},\n"
            f"  fsdp_size={self._fsdp_size},\n"
            f"  redundant_size={self._redundant_size}\n"
            f")"
        )

    @classmethod
    def check_process_groups(cls) -> bool:
        """Check if all necessary process groups are properly initialized.

        Returns:
            bool: True if all necessary process groups are initialized, False otherwise
        """
        instance = cls.get_instance()
        if instance._device_mesh is None:
            return False

        # Check if all groups that should exist are properly initialized
        if instance._dp_size > 1 and cls.dp_group() is None:
            return False
        if instance._cfg_size > 1 and cls.cfg_group() is None:
            return False
        if instance._ring_size > 1 and cls.ring_group() is None:
            return False
        if instance._cp_size > 1 and cls.cp_group() is None:
            return False
        if instance._ulysses_size > 1 and cls.ulysses_group() is None:
            return False
        if instance._tp_size > 1 and cls.tp_group() is None:
            return False

        return True

    @classmethod
    def dp_size(cls) -> int:
        """Get the data parallel size.

        Returns:
            int: Data parallel size
        """
        return cls.get_instance()._dp_size

    @classmethod
    def tp_size(cls) -> int:
        """Get the tensor parallel size.

        Returns:
            int: Tensor parallel size
        """
        return cls.get_instance()._tp_size

    @classmethod
    def cfg_size(cls) -> int:
        """Get the classifier-free guidance parallel size.

        Returns:
            int: Classifier-free guidance parallel size
        """
        return cls.get_instance()._cfg_size

    @classmethod
    def ulysses_size(cls) -> int:
        """Get the Ulysses parallel size.

        Returns:
            int: Ulysses parallel size
        """
        return cls.get_instance()._ulysses_size

    @classmethod
    def ring_size(cls) -> int:
        """Get the ring attention parallel size.

        Returns:
            int: Ring attention parallel size
        """
        return cls.get_instance()._ring_size

    @classmethod
    def cp_size(cls) -> int:
        """Get the context parallel size.
        Only allGather KV to reduce the number of communication.

        Returns:
            int: context parallel size
        """
        return cls.get_instance()._cp_size

    @classmethod
    def sp_size(cls) -> int:
        """Get the sequence parallel size.

        Returns:
            int: Sequence parallel size
        """
        return cls.get_instance()._ulysses_size * cls.get_instance()._ring_size * cls.get_instance()._cp_size

    @classmethod
    def fsdp_size(cls) -> int:
        """Get the fully sharded data parallel size.

        Returns:
            int: Fully sharded data parallel size
        """
        return cls.get_instance()._fsdp_size


class T5ParallelConfig(BaseParallelConfig):
    """Singleton configuration class for T5 parallelism."""


class DiTParallelConfig(BaseParallelConfig):
    """Singleton configuration class for DiT parallelism."""


class RefinerDiTParallelConfig(BaseParallelConfig):
    """Singleton configuration class for Refiner's DiT parallelism."""


@dataclass
class VAEParallelConfig:
    """Configuration class for VAE parallelism."""

    disable_parallel_vae: bool = False
    parallel_vae_split_dim: str = "width"  # "width" or "height"

    @classmethod
    def set_config(cls, **kwargs):
        for key, value in kwargs.items():
            if key in cls.__dataclass_fields__:
                if key == "parallel_vae_split_dim":
                    if value not in ["width", "height"]:
                        raise ValueError(f"Invalid value for parallel_vae_split_dim: {value}")
                setattr(cls, key, value)
            else:
                raise ValueError(f"Invalid key: {key}")


@torch.compiler.disable
def get_dit_parallel_config():
    if PipelineConfig.in_refiner_stage:
        return RefinerDiTParallelConfig
    else:
        return DiTParallelConfig


@contextmanager
def dit_parallel_config_context(
    tp_size: Optional[int] = None,
    ulysses_size: Optional[int] = None,
    ring_size: Optional[int] = None,
    cp_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    cfg_size: Optional[int] = None,
    fsdp_size: Optional[int] = None,
):
    """
    Context manager for temporarily modifying DiTParallelConfig settings.

    Args:
        tp_size: Tensor parallel degree
        ulysses_size: Ulysses parallel degree
        ring_size: Ring attention parallel degree
        cp_size: Context parallel degree
        dp_size: Data parallel degree
        cfg_size: Classifier-free guidance parallel degree
        fsdp_size: Fully sharded data parallel degree

    Usage:
        with dit_parallel_config_context(tp_size=2, cp_size=4):
            # DiTParallelConfig is temporarily modified
            model_forward()
        # DiTParallelConfig is restored to original values
    """
    # Get the current DiTParallelConfig instance
    config = DiTParallelConfig.get_instance()

    # Store original values
    original_values = {
        "tp_size": config._tp_size,
        "ulysses_size": config._ulysses_size,
        "ring_size": config._ring_size,
        "cp_size": config._cp_size,
        "dp_size": config._dp_size,
        "cfg_size": config._cfg_size,
        "fsdp_size": config._fsdp_size,
    }

    try:
        # Apply new configuration values (only if provided)
        new_config = {}
        if tp_size is not None:
            new_config["tp_size"] = tp_size
        if ulysses_size is not None:
            new_config["ulysses_size"] = ulysses_size
        if ring_size is not None:
            new_config["ring_size"] = ring_size
        if cp_size is not None:
            new_config["cp_size"] = cp_size
        if dp_size is not None:
            new_config["dp_size"] = dp_size
        if cfg_size is not None:
            new_config["cfg_size"] = cfg_size
        if fsdp_size is not None:
            new_config["fsdp_size"] = fsdp_size

        # Apply the new configuration if any values were provided
        if new_config:
            # Merge with original values to ensure all parameters are set
            full_config = original_values.copy()
            full_config.update(new_config)
            config.set_config(**full_config)

        yield config

    finally:
        # Restore original configuration
        config.set_config(**original_values)
