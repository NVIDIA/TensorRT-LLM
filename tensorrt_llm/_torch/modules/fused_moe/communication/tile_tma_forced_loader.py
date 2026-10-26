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

"""Fail-closed loader for the packaged Push-Combine side DSO.

The SM107 TensorRT-LLM wheel bundles one DSO that registers the
``moe_a2a_push_*`` Torch schemas. This module resolves that package artifact
(or an explicit expert override), binds the process to the exact inode that was
hashed and loaded, validates its exported build ID, and hands the SHA256-keyed
``kernel_build_id`` to the selector for workspace and CUDA-graph identity.
"""

import ctypes
import hashlib
import importlib
import inspect
import os
import stat
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .base import Communication


@dataclass(frozen=True)
class TileTmaDsoIdentity:
    """Identity of the exact unified DSO object loaded into this process."""

    build_id_symbol: str
    exported_build_id: str
    kernel_build_id: str
    dso_path: str
    dso_sha256: str
    dso_device: int
    dso_inode: int
    dso_size: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable identity record."""

        return asdict(self)


def _hash_open_file(dso_fd: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(os.dup(dso_fd), "rb") as dso_stream:
        for chunk in iter(lambda: dso_stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exported_build_id(dso_fd: int, symbol: str) -> str:
    dso = ctypes.CDLL(f"/proc/self/fd/{dso_fd}")
    try:
        read_build_id = getattr(dso, symbol)
    except AttributeError as error:
        raise RuntimeError(
            f"Tile-TMA DSO does not export required build-id symbol {symbol}; "
            "refusing non-unified (per-arm or foreign) DSO substitution"
        ) from error
    read_build_id.argtypes = []
    read_build_id.restype = ctypes.c_char_p
    exported = read_build_id()
    if exported is None:
        raise RuntimeError("Tile-TMA DSO exported a null build ID")
    try:
        return exported.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("Tile-TMA DSO build ID is not valid UTF-8") from error


def _optional_ctor_kwargs(
    selector_class: type[Communication], identity: TileTmaDsoIdentity
) -> dict:
    """Pass the loader handle only if the selector ctor declares a slot for it."""

    try:
        parameters = inspect.signature(selector_class.__init__).parameters
    except (TypeError, ValueError):
        return {}
    optional = {}
    if "loader_handle" in parameters:
        optional["loader_handle"] = identity
    if "loader_identity" in parameters:
        optional["loader_identity"] = identity.as_dict()
    return optional


PUSH_COMBINE_FORCE_METHOD = "PUSH_COMBINE"
PUSH_COMBINE_DSO_ENV = "TRTLLM_TILE_PUSH_COMBINE_DSO"
PUSH_COMBINE_PACKAGED_DSO = "tile_push_comm_ops.so"
PUSH_COMBINE_BUILD_ID_SYMBOL = "trtllm_tile_push_combine_build_id"
PUSH_COMBINE_EXPORTED_BUILD_ID = "tile_push_combine:v2"
# The one legal push wire recipe: packed-interleaved [Q|SF] (recipe id 3).
PUSH_COMBINE_RECIPE = "nvfp4_packed"
# The complete K4 consumer surface.
PUSH_COMBINE_SUPPORTED_EP_SIZES = (2, 4, 8, 16, 32)

PUSH_COMBINE_SELECTOR_CLASS_NAME = "NVLinkOneSidedTilePushCombine"
_PUSH_COMBINE_SELECTOR_MODULE = "nvlink_one_sided_tile_nvfp4_push"

_PUSH_LOAD_LOCK = threading.Lock()
_LOADED_PUSH_IDENTITY: TileTmaDsoIdentity | None = None


def validate_push_combine_ep_size(ep_size: int) -> int:
    """Validate ep_size against the complete Push-Combine K4 surface."""

    ep_size = int(ep_size)
    if ep_size not in PUSH_COMBINE_SUPPORTED_EP_SIZES:
        raise ValueError(
            f"push-combine ep_size must be one of {PUSH_COMBINE_SUPPORTED_EP_SIZES} "
            f"(the supported K4 consumer surface), got EP{ep_size}"
        )
    return ep_size


def get_packaged_push_combine_dso_path() -> str:
    """Return the package-relative Push DSO path."""

    package_root = Path(__file__).resolve().parents[4]
    return str(package_root / "libs" / PUSH_COMBINE_PACKAGED_DSO)


def resolve_push_combine_dso_path() -> str:
    """Resolve an expert override or the DSO bundled by the normal wheel build."""

    override = os.environ.get(PUSH_COMBINE_DSO_ENV)
    if override == "":
        raise RuntimeError(f"{PUSH_COMBINE_DSO_ENV} must not be empty when set")
    dso_path = override if override is not None else get_packaged_push_combine_dso_path()
    if not os.path.isfile(dso_path):
        source = f"{PUSH_COMBINE_DSO_ENV} override" if override else "bundled TensorRT-LLM wheel"
        raise RuntimeError(
            f"push-combine DSO from the {source} is missing: {dso_path}. "
            "Build/install a TensorRT-LLM wheel that targets SM107 (Push-Combine support), or set "
            f"{PUSH_COMBINE_DSO_ENV} to an expert override."
        )
    return dso_path


def _import_push_combine_selector_class() -> type[Communication]:
    package = __name__.rsplit(".", 1)[0]
    try:
        module = importlib.import_module(f"{package}.{_PUSH_COMBINE_SELECTOR_MODULE}")
    except ImportError as error:
        raise RuntimeError(
            f"Unable to import the push-combine module {_PUSH_COMBINE_SELECTOR_MODULE}: {error}"
        ) from error
    selector_class = getattr(module, PUSH_COMBINE_SELECTOR_CLASS_NAME, None)
    if selector_class is None:
        raise RuntimeError(
            f"{_PUSH_COMBINE_SELECTOR_MODULE} has no attribute {PUSH_COMBINE_SELECTOR_CLASS_NAME}"
        )
    return selector_class


def _load_push_combine_dso(dso_path: str) -> TileTmaDsoIdentity:
    real_dso_path = os.path.realpath(dso_path)
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    dso_fd = os.open(real_dso_path, open_flags)
    try:
        dso_stat = os.fstat(dso_fd)
        if not stat.S_ISREG(dso_stat.st_mode):
            raise RuntimeError(f"push-combine DSO is not a regular file: {real_dso_path}")
        dso_sha256 = _hash_open_file(dso_fd)

        global _LOADED_PUSH_IDENTITY
        with _PUSH_LOAD_LOCK:
            if _LOADED_PUSH_IDENTITY is not None:
                if _LOADED_PUSH_IDENTITY.dso_sha256 != dso_sha256:
                    raise RuntimeError(
                        "A different push-combine DSO is already loaded in this "
                        f"process (loaded sha256={_LOADED_PUSH_IDENTITY.dso_sha256}, "
                        f"requested sha256={dso_sha256}); the moe_a2a_push_* "
                        "schemas may register only once"
                    )
                return _LOADED_PUSH_IDENTITY

            fd_path = f"/proc/self/fd/{dso_fd}"
            torch.ops.load_library(fd_path)
            exported_build_id = _read_exported_build_id(dso_fd, PUSH_COMBINE_BUILD_ID_SYMBOL)
            if exported_build_id != PUSH_COMBINE_EXPORTED_BUILD_ID:
                raise RuntimeError(
                    "push-combine DSO build-id mismatch: exported "
                    f"{exported_build_id!r}, expected "
                    f"{PUSH_COMBINE_EXPORTED_BUILD_ID!r}"
                )
            identity = TileTmaDsoIdentity(
                build_id_symbol=PUSH_COMBINE_BUILD_ID_SYMBOL,
                exported_build_id=exported_build_id,
                kernel_build_id=f"{exported_build_id}:{dso_sha256}",
                dso_path=real_dso_path,
                dso_sha256=dso_sha256,
                dso_device=int(dso_stat.st_dev),
                dso_inode=int(dso_stat.st_ino),
                dso_size=int(dso_stat.st_size),
            )
            _LOADED_PUSH_IDENTITY = identity
            logger.info(
                f"Loaded push-combine DSO: build_id={exported_build_id}, dso_sha256={dso_sha256}"
            )
            return identity
    finally:
        os.close(dso_fd)


def get_loaded_push_combine_identity() -> dict[str, str | int] | None:
    """Return the process-wide loaded push-combine identity, if any."""

    with _PUSH_LOAD_LOCK:
        return None if _LOADED_PUSH_IDENTITY is None else _LOADED_PUSH_IDENTITY.as_dict()


def create_push_combine_strategy(
    *,
    dso_path: str,
    mapping: Mapping,
    num_slots: int,
    top_k: int,
    max_num_tokens_per_rank: int,
    payload_in_workspace: bool,
    hidden_size: int,
    dtype: torch.dtype,
) -> Communication:
    """Construct the forced Push-Combine strategy for a supported EP size."""

    ep_size = validate_push_combine_ep_size(mapping.moe_ep_size)
    if hidden_size != 7168 or top_k != 8 or dtype is not torch.bfloat16:
        raise ValueError(
            "push-combine has a fixed BF16/H7168/top-k8 activation contract; "
            f"got dtype={dtype}, hidden_size={hidden_size}, top_k={top_k}"
        )
    if max_num_tokens_per_rank <= 0:
        raise ValueError("push-combine max_num_tokens_per_rank must be positive")

    selector_class = _import_push_combine_selector_class()
    identity = _load_push_combine_dso(dso_path)

    build_id_prefix = getattr(selector_class, "BUILD_ID_PREFIX", None)
    if build_id_prefix is not None:
        if not isinstance(build_id_prefix, str) or not build_id_prefix.endswith(":"):
            raise RuntimeError(
                f"push-combine selector {selector_class.__name__} has an invalid BUILD_ID_PREFIX"
            )
        if identity.exported_build_id != build_id_prefix[:-1]:
            raise RuntimeError(
                "push-combine DSO build-id mismatch: exported "
                f"{identity.exported_build_id!r}, expected {build_id_prefix[:-1]!r}"
            )

    strategy = selector_class(
        mapping,
        recipe=PUSH_COMBINE_RECIPE,
        num_slots=num_slots,
        top_k=top_k,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        payload_in_workspace=payload_in_workspace,
        hidden_size=hidden_size,
        dtype=dtype,
        num_experts=None,
        use_low_precision_combine=False,
        kernel_build_id=identity.kernel_build_id,
        **_optional_ctor_kwargs(selector_class, identity),
    )
    if not strategy.supports_post_quant_dispatch():
        raise RuntimeError("push-combine requires post-quant dispatch support")
    strategy._tile_push_dso_identity = identity.as_dict()
    logger.info(
        "Constructed forced push-combine communication selector: "
        f"recipe={PUSH_COMBINE_RECIPE}, EP={ep_size}, "
        f"selector={selector_class.__name__}, dso_sha256={identity.dso_sha256}"
    )
    return strategy


def create_push_combine_strategy_from_env(
    *,
    mapping: Mapping,
    num_slots: int,
    top_k: int,
    max_num_tokens_per_rank: int,
    payload_in_workspace: bool,
    hidden_size: int,
    dtype: torch.dtype,
) -> Communication:
    """Construct Push-Combine using the bundled DSO or an expert override."""

    dso_path = resolve_push_combine_dso_path()
    return create_push_combine_strategy(
        dso_path=dso_path,
        mapping=mapping,
        num_slots=num_slots,
        top_k=top_k,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        payload_in_workspace=payload_in_workspace,
        hidden_size=hidden_size,
        dtype=dtype,
    )
