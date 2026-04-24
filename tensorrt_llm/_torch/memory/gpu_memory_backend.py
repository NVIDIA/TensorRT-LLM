# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from typing import Iterator, Optional, Protocol, runtime_checkable

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_MODE_ALIASES = ("rw", "ro", "auto")


@runtime_checkable
class GPUMemoryBackend(Protocol):
    def connect(self) -> bool:
        ...

    @property
    def is_rw(self) -> Optional[bool]:
        ...

    def has_committed_weights(self) -> bool:
        ...

    def total_bytes(self) -> int:
        ...

    def mem_pool_scope(self, device: Optional[torch.device] = None) -> Iterator[None]:
        ...

    def materialize_module(self, model: nn.Module) -> None:
        ...

    def finalize_write(self, model: nn.Module) -> int:
        ...

    def defer_finalize_write(self, model: nn.Module) -> None:
        ...

    def finalize_pending_write(self) -> int:
        ...

    def move_untracked_params(self, model: nn.Module) -> None:
        ...

    def cleanup(self) -> None:
        ...


class GMSBackend:
    DEFAULT_TAG = "weights"

    def __init__(
        self,
        socket_path: Optional[str],
        mapping: Mapping,
        mode: str = "auto",
        tag: str = DEFAULT_TAG,
    ) -> None:
        if mode not in _MODE_ALIASES:
            raise ValueError(
                f"GMS mode must be one of {_MODE_ALIASES}, got {mode!r}")

        self._socket_path = socket_path
        self._mapping = mapping
        self._mode = mode
        self._tag = tag
        self._device_index = torch.cuda.current_device()
        self._client = None
        self._is_rw: Optional[bool] = None
        self._pending_model: Optional[nn.Module] = None

    def connect(self) -> bool:
        try:
            from gpu_memory_service.client.torch.allocator import (
                get_or_create_gms_client_memory_manager,
            )
            from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
            from gpu_memory_service.common.utils import get_socket_path
            from gpu_memory_service.integrations.common.patches import patch_empty_cache
        except ImportError:
            logger.warning(
                "gpu_memory_service is not installed; LoadFormat.GMS is unavailable.")
            return False

        mode_map = {
            "rw": RequestedLockType.RW,
            "ro": RequestedLockType.RO,
            "auto": RequestedLockType.RW_OR_RO,
        }

        socket_path = self._socket_path
        if socket_path is None:
            socket_path = get_socket_path(self._device_index, self._tag)
        self._socket_path = socket_path

        try:
            self._client = get_or_create_gms_client_memory_manager(
                socket_path,
                self._device_index,
                mode=mode_map[self._mode],
                tag=self._tag,
            )
        except Exception as e:
            logger.warning(
                "Failed to connect to GMS at %s (mode=%s, tag=%s): %s",
                socket_path,
                self._mode,
                self._tag,
                e,
            )
            self._client = None
            return False

        self._is_rw = self._client.granted_lock_type == GrantedLockType.RW
        try:
            patch_empty_cache()
        except Exception as e:
            logger.debug("GMS patch_empty_cache failed (non-fatal): %s", e)

        logger.info(
            "Connected to GMS at %s (mode=%s, granted=%s, tag=%s)",
            socket_path,
            self._mode,
            "RW" if self._is_rw else "RO",
            self._tag,
        )
        return True

    @property
    def is_rw(self) -> Optional[bool]:
        return self._is_rw

    def has_committed_weights(self) -> bool:
        if self._client is None:
            return False
        try:
            from gpu_memory_service.common.locks import GrantedLockType

            return self._client.granted_lock_type == GrantedLockType.RO
        except Exception:
            return False

    def total_bytes(self) -> int:
        if self._client is None:
            return 0
        return int(self._client.total_bytes)

    @contextmanager
    def mem_pool_scope(
        self,
        device: Optional[torch.device] = None,
    ) -> Iterator[None]:
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if self._is_rw is False:
            raise RuntimeError(
                "GMS mem_pool_scope() is only valid in RW mode (this client was granted RO)."
            )

        from gpu_memory_service.client.torch.allocator import gms_use_mem_pool

        target_device = device
        if target_device is None:
            target_device = torch.device("cuda", self._device_index)

        with gms_use_mem_pool(self._tag, target_device):
            yield

    def move_untracked_params(self, model: nn.Module) -> None:
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service.client.torch.module import _iter_module_tensors
        from gpu_memory_service.client.torch.tensor import _tensor_from_pointer

        gms_client = self._client
        seen: set[int] = set()

        with torch.no_grad():
            for _name, tensor, tensor_type in _iter_module_tensors(model):
                if tensor_type != "parameter" or tensor is None or not tensor.is_cuda:
                    continue

                storage_ptr = tensor.untyped_storage().data_ptr()
                if storage_ptr in seen:
                    continue
                seen.add(storage_ptr)

                if _ptr_in_gms(gms_client, int(tensor.data_ptr())):
                    continue

                nbytes = _storage_nbytes(tensor)
                base_va = gms_client.create_mapping(size=nbytes, tag=self._tag)
                replacement = _tensor_from_pointer(
                    int(base_va),
                    list(tensor.shape),
                    list(tensor.stride()),
                    tensor.dtype,
                    self._device_index,
                )
                replacement.copy_(tensor)
                tensor.data = replacement

    def finalize_write(self, model: nn.Module) -> int:
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if self._is_rw is False:
            raise RuntimeError("GMS finalize_write() is only valid in RW mode.")

        self._pending_model = None

        from gpu_memory_service.client.torch.module import register_module_tensors
        from gpu_memory_service.integrations.common.utils import finalize_gms_write

        register_module_tensors(self._client, model)
        bytes_committed = int(self._client.total_bytes)
        torch.cuda.synchronize()
        finalize_gms_write(self._client)
        self._is_rw = False
        logger.info(
            "GMS RW->RO: committed %.2f GiB at %s (tag=%s)",
            bytes_committed / (1 << 30),
            self._socket_path,
            self._tag,
        )
        return bytes_committed

    def defer_finalize_write(self, model: nn.Module) -> None:
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if self._is_rw is False:
            raise RuntimeError("GMS defer_finalize_write() is only valid in RW mode.")
        self._pending_model = model

    def finalize_pending_write(self) -> int:
        if self._pending_model is None:
            return 0

        model = self._pending_model
        self._pending_model = None
        return self.finalize_write(model)

    def materialize_module(self, model: nn.Module) -> None:
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service.client.torch.module import materialize_module_from_gms
        from tensorrt_llm._torch.modules.linear import Linear

        materialize_module_from_gms(
            self._client,
            model,
            device_index=self._device_index,
        )

        for module in model.modules():
            if isinstance(module, Linear):
                module._weights_presharded = True

        logger.info(
            "GMS RO: materialized weights from %s (tag=%s, tp_rank=%d/%d, total_bytes=%.2f GiB)",
            self._socket_path,
            self._tag,
            self._mapping.tp_rank,
            self._mapping.tp_size,
            int(self._client.total_bytes) / (1 << 30),
        )

    def cleanup(self) -> None:
        if self._client is None:
            return

        try:
            from gpu_memory_service.client.torch.allocator import (
                evict_gms_client_memory_manager,
            )

            client = self._client
            try:
                client.close()
            except Exception:
                pass
            evict_gms_client_memory_manager(client)
            logger.info("GMS: disconnected from %s", self._socket_path)
        except Exception as e:
            logger.warning("GMS cleanup error: %s", e)
        finally:
            self._client = None
            self._pending_model = None


def _ptr_in_gms(gms_client, ptr: int) -> bool:
    mappings = getattr(gms_client, "mappings", None)
    if not isinstance(mappings, MappingABC):
        mappings = getattr(gms_client, "_mappings", None)
    if not isinstance(mappings, MappingABC) or not mappings:
        return False

    for mapping in mappings.values():
        base = int(getattr(mapping, "va", 0))
        size = getattr(mapping, "aligned_size", None)
        if not isinstance(size, int):
            size = getattr(mapping, "size", 0)
        size = int(size)
        if base and size and base <= ptr < base + size:
            return True
    return False


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())
