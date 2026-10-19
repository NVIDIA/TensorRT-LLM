# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime capability probing and build-time EP bootstrap for MegaMoE."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

__all__ = [
    "MegaMoeCuteDslUnavailable",
    "is_megamoe_cute_dsl_runtime_available",
    "maybe_init_torch_dist_under_mpi",
    "resolve_ep_process_group",
]


class MegaMoeCuteDslUnavailable(RuntimeError):
    """The active environment cannot supply the required MegaMoE runtime."""


_RUNTIME_PROBE_CACHE: dict[int, bool | str] = {}


def is_megamoe_cute_dsl_runtime_available() -> tuple[bool, str | None]:
    """Return whether the CUDA 13 Cutlass DSL runtime exposes all symbols the
    ported MegaMoE CuteDSL kernel needs.

    Stricter than ``IS_CUTLASS_DSL_AVAILABLE``, which only confirms that
    ``cutlass`` / ``cutlass.cute`` import cleanly. The MegaMoE kernel
    ABI also requires ``cutlass.torch.from_dlpack``, ``cutlass._mlir``,
    ``cute_nvgpu`` MMA atoms, and async-copy helpers. SM107 additionally
    requires ``cutlass.utils.rubin_helpers``. Older builds return
    ``(False, reason)``.

    Returns ``(True, None)`` on success or ``(False, reason)`` with an
    actionable message. The result is cached for the process lifetime.
    """
    sm_version = get_sm_version() if torch.cuda.is_available() else -1
    cached = _RUNTIME_PROBE_CACHE.get(sm_version)
    if cached is True:
        return True, None
    if isinstance(cached, str):
        return False, cached

    if not IS_CUTLASS_DSL_AVAILABLE:
        reason = (
            "Cutlass DSL is not importable on this environment; install "
            "nvidia-cutlass-dsl[cu13] to enable MegaMoECuteDsl."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.pipeline  # noqa: F401
        import cutlass.torch  # noqa: F401
        from cutlass._mlir import ir  # noqa: F401
        from cutlass.base_dsl.native_struct import native_struct  # noqa: F401
        from cutlass.cutlass_dsl import (  # noqa: F401
            Int32,
            Int64,
            Uint8,
            dsl_user_op,
            extract_mlir_values,
            new_from_mlir_values,
        )
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires CUDA 13 Cutlass DSL symbols; got "
            f"ImportError={e!r}. Install nvidia-cutlass-dsl[cu13]>=4.5.0 "
            f"(see PR #14354)."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    try:
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires cutlass.cute.nvgpu.tcgen05 + cpasync; "
            f"missing {e!r}. Install a Blackwell-capable cutlass-dsl wheel."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    if sm_version == 107:
        try:
            try:
                import cutlass.memory as cutlass_memory
            except ImportError:
                import cutlass.utils as cutlass_memory
            import cutlass.utils.rubin_helpers as rubin_helpers

            if not hasattr(rubin_helpers, "make_blockscaled_trivial_tiled_mma"):
                raise ImportError(
                    "cutlass.utils.rubin_helpers lacks make_blockscaled_trivial_tiled_mma"
                )
            if not hasattr(cutlass_memory, "get_smem_capacity_in_bytes"):
                raise ImportError("Cutlass DSL lacks get_smem_capacity_in_bytes")
            if cutlass_memory.get_smem_capacity_in_bytes("sm_107") <= 0:
                raise ValueError("invalid SM107 shared-memory capacity")
            if cute.arch.get_max_tmem_alloc_cols("sm_107") <= 0:
                raise ValueError("invalid SM107 tensor-memory capacity")
        except (
            AttributeError,
            ImportError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            reason = (
                f"MegaMoECuteDsl on SM107 requires Rubin support in the "
                f"Cutlass DSL wheel; missing {e!r}. Use a Rubin-capable "
                f"Cutlass DSL build."
            )
            _RUNTIME_PROBE_CACHE[sm_version] = reason
            return False, reason

    try:
        from ....moe.kernels.cutedsl_megamoe import (  # noqa: F401
            BlackwellInferenceMegaMoE,
            RubinInferenceGenphaseMegaMoE,
            RubinInferenceLocalMegaMoE,
            RubinInferenceMegaMoE,
        )
    except ImportError as e:
        reason = (
            f"Ported MegaMoE NVFP4 kernel package failed to import: "
            f"{e!r}. Verify tensorrt_llm/_torch/moe/kernels/"
            f"cutedsl_megamoe is in the install tree."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    _RUNTIME_PROBE_CACHE[sm_version] = True
    return True, None


def maybe_init_torch_dist_under_mpi(ep_size: int) -> None:
    """Bootstrap a torch.distributed NCCL WORLD group under the MPI orchestrator.

    The MPI orchestrator (trtllm-bench / mpirun) never calls
    ``init_process_group``, but the symmetric-memory rendezvous needs a
    ProcessGroup; for pure DEP a node-local NCCL WORLD suffices. No-op
    under Ray or single-rank.
    """
    if not dist.is_available() or dist.is_initialized():
        return
    from tensorrt_llm._utils import MPI, mpi_comm, mpi_rank, mpi_world_size

    try:
        world = mpi_world_size()
        rank = mpi_rank()
    except (MPI.Exception, RuntimeError, ValueError) as e:
        logger.debug(
            f"[MegaMoECuteDsl] MPI rank query failed ({e!r}); skipping torch.distributed bootstrap."
        )
        return
    if world <= 1:
        return
    # Check pure DEP BEFORE any collective: under PP/CP hybrids some ranks
    # never construct a MegaMoE module, so the bcast/init below would hang
    # instead of failing loud. Rank-identical inputs -> lockstep raise.
    if world != ep_size:
        raise ValueError(
            f"MegaMoECuteDsl requires pure DEP (MPI WORLD == EP) to "
            f"bootstrap torch.distributed: got MPI world={world}, "
            f"EP={ep_size}."
        )

    # Rank 0 draws a free port (a fixed one collides across same-node
    # jobs). The bcast is UNCONDITIONAL and rank 0 alone decides
    # (honoring ITS pre-set MASTER_* env): gating the collective on
    # per-rank env presence would desync the communicator when the vars
    # are exported on only a subset of ranks. No bind-race retry: a sound
    # one needs cross-rank failure agreement, and the close()->bind()
    # window is negligible next to the fixed-port collision this replaces.
    def _pick_rendezvous() -> tuple[str, str]:
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")
        if addr and port:
            return (addr, port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        try:
            addr = socket.gethostbyname(socket.gethostname())
        except OSError:
            # Unresolvable hostname (minimal containers / broken DNS):
            # this bootstrap is documented single-node, loopback works.
            addr = "127.0.0.1"
        return (addr, str(port))

    host, port = mpi_comm().bcast(_pick_rendezvous() if rank == 0 else None, root=0)
    os.environ["MASTER_ADDR"] = str(host)
    os.environ["MASTER_PORT"] = str(port)
    # The executor already bound this rank's device before the model was
    # built (base_worker: global_mpi_rank() % device_count). Re-deriving it
    # here could pick a different GPU (e.g. a disagg ctx/gen pair sharing a
    # node) and would move only the layers built after this point, so bind
    # NCCL to the current device instead of selecting one.
    device_id = None
    if torch.cuda.is_available():
        device_id = torch.device("cuda", torch.cuda.current_device())
    logger.info(
        f"[MegaMoECuteDsl] torch.distributed not initialized under MPI; "
        f"bootstrapping NCCL WORLD group (rank={rank}/{world}, "
        f"device={device_id}, "
        f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}) for the "
        f"EP rendezvous."
    )
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        rank=rank,
        world_size=world,
        device_id=device_id,
    )


def resolve_ep_process_group(
    mapping: Mapping, *, ep_size: int, layer_idx: int | None
) -> dist.ProcessGroup:
    """Return the torch.distributed ProcessGroup for the EP sub-world.

    Prefer the mapping group and otherwise accept WORLD only for pure DEP.
    """
    maybe_init_torch_dist_under_mpi(ep_size)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "MegaMoECuteDsl requires torch.distributed to be initialized "
            "before module construction (mpirun or Ray) for multi-rank "
            "execution."
        )
    try:
        pg = mapping.moe_ep_group_pg
        log_fn = logger.info if layer_idx == 0 else logger.debug
        log_fn(
            f"[MegaMoECuteDsl] layer={layer_idx} using mapping.moe_ep_group_pg (DeviceMesh path)."
        )
        return pg
    except (NotImplementedError, AttributeError):
        pass
    world_size = dist.get_world_size()
    if ep_size == world_size:
        log_fn = logger.info if layer_idx == 0 else logger.debug
        log_fn(
            f"[MegaMoECuteDsl] layer={layer_idx} using dist.group.WORLD "
            f"(EP == world_size == {world_size})."
        )
        return dist.group.WORLD
    raise RuntimeError(
        f"MegaMoECuteDsl: cannot resolve EP ProcessGroup. The current "
        f"mapping does not expose ``moe_ep_group_pg`` and EP "
        f"({ep_size}) is a strict subset of world ({world_size})."
    )
