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
"""NCCL EP utilities backed by the nccl4py wheel's ``nccl.ep`` package.

Owns the long-lived NCCL EP resources (communicator, group, persistent receive
NDTensors) for the MoE NcclEP communication strategy. Per-step dispatch handles
are created in ``communication/nccl_ep.py``.
"""

from typing import Optional

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_MIN_NCCL_RUNTIME_VERSION = "2.30.4"
_MIN_NCCL_EP_INT32_TOPK_VERSION = "0.2"
_NCCL_RUNTIME_ERRORS = (RuntimeError, OSError)
_NCCL_AVAILABILITY_ERRORS = (ImportError,) + _NCCL_RUNTIME_ERRORS

_nccl_ep_installed: Optional[bool] = None


def is_nccl_ep_installed() -> bool:
    """Return True iff ``nccl.ep`` is usable.

    Requires that ``nccl.ep`` imports cleanly AND the loaded ``libnccl.so``
    runtime version is >= 2.30.4.
    """
    global _nccl_ep_installed
    if _nccl_ep_installed is not None:
        return _nccl_ep_installed
    try:
        import nccl
        from packaging.version import Version

        runtime = nccl.get_version().nccl.version
        if runtime < Version(_MIN_NCCL_RUNTIME_VERSION):
            logger.info(
                f"NCCL EP disabled: libnccl runtime {runtime} "
                f"< required {_MIN_NCCL_RUNTIME_VERSION}"
            )
            _nccl_ep_installed = False
            return False
        import nccl.ep  # noqa: F401

        _nccl_ep_installed = True
    except _NCCL_AVAILABILITY_ERRORS as e:
        logger.info(f"NCCL EP disabled: nccl.ep is not usable ({e!r})")
        _nccl_ep_installed = False
    return _nccl_ep_installed


def _nccl_ep_supports_int32_topk_idx() -> bool:
    """Return True when the loaded libnccl_ep supports int32 input topk_idx."""
    try:
        import nccl
        from packaging.version import Version

        nccl_ep_info = nccl.get_version().nccl_ep
        nccl_ep_version = nccl_ep_info.version if nccl_ep_info is not None else None
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        logger.info(
            f"NCCL EP int32 topk_idx disabled: could not determine libnccl_ep version ({e!r})"
        )
        return False

    if nccl_ep_version is None:
        logger.info("NCCL EP int32 topk_idx disabled: libnccl_ep version is not available")
        return False

    if nccl_ep_version < Version(_MIN_NCCL_EP_INT32_TOPK_VERSION):
        logger.info(
            f"NCCL EP int32 topk_idx disabled: libnccl_ep {nccl_ep_version} "
            f"< required {_MIN_NCCL_EP_INT32_TOPK_VERSION}"
        )
        return False

    return True


# Singleton EP context keyed by (ep_size, ep_rank, max_tokens, num_experts,
# hidden, max_top_k, layout).
_ep_group_cache: dict = {}
_ep_group_refcounts: dict = {}


class NcclEpContext:
    """Long-lived NCCL EP group + receive buffers, shared across NcclEP instances.

    Owns the :class:`nccl.ep.Group`, the source :class:`nccl.core.Communicator`,
    and the rank-major LL persistent receive buffers (tokens, top-k idx / weights,
    per-source-rank counter) wrapped as
    :class:`nccl.ep.Tensor` descriptors.

    Per-step routing handles (``Handle``) are created in ``NcclEP``, not here.
    """

    def __init__(
        self,
        mapping: Mapping,
        num_experts: int,
        max_tokens_per_rank: int,
        hidden_size: int,
        max_top_k: int,
        layout: Optional[int] = None,
    ):
        import nccl.core as nccl_core
        from nccl.ep import Algorithm, Group, GroupConfig, Layout, Tensor

        from tensorrt_llm._utils import mpi_comm

        self.mapping = mapping
        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.max_top_k = max_top_k
        self.hidden_size = hidden_size
        self.layout = Layout.RANK_MAJOR if layout is None else Layout(layout)
        self.max_recv_tokens = self.ep_size * max_tokens_per_rank

        # topk_idx dtype passed to the EP runtime. NCCL-EP < 0.2 asserts
        # int64 in ncclEpUpdateHandle; 0.2+ supports TRT-LLM's native int32
        # routing ids and avoids the per-iter widening conversion.
        self.topk_idx_dtype = torch.int32 if _nccl_ep_supports_int32_topk_idx() else torch.int64
        self._v0_2_features_enabled = self.topk_idx_dtype == torch.int32

        # NCCL-EP v0.2+ may expose a configurable receive expert-id kind.
        # Within that version-gated path, detect whether the linked binding supports a
        # configurable recv_topk_idx kind on LayoutInfo. When the field
        # is present we set it to GLOBAL and skip the post-dispatch
        # local->global rewrite; otherwise the kernel writes LOCAL ids
        # unconditionally (older nccl-ep builds) and the dispatch
        # wrapper applies torch.where to restore the global contract
        # NVLinkOneSided also advertises.
        self.kernel_writes_global_ids = False
        self._expert_id_kind_global = None
        if self._v0_2_features_enabled:
            try:
                from nccl.bindings.nccl_ep import ExpertIdKind as _ExpertIdKind
                from nccl.bindings.nccl_ep import LayoutInfo as _LowLayoutInfo

                self.kernel_writes_global_ids = hasattr(_LowLayoutInfo(), "recv_topk_idx_kind")
                self._expert_id_kind_global = (
                    int(_ExpertIdKind.GLOBAL) if self.kernel_writes_global_ids else None
                )
            except (ImportError, AttributeError):
                pass

        # NCCL-EP v0.2+ may support opportunistic zero-copy dispatch.
        # When the Pythonic GroupConfig facade exposes `zero_copy` (i.e.,
        # the wheel was built against a libnccl_ep.so that has the field
        # in ncclEpGroupConfig_t), we allocate a VMM-backed,
        # window-registered dispatch output buffer; the LL dispatch
        # opportunistically picks zero-copy when recv_x->win_hdl is set
        # (nvlink-only + rank-major). The config flag itself stays
        # AUTO/OFF -- strict zero_copy=ON requires combine inputs to be
        # windowed too, which would force a caller-side interface change
        # (the MLP output is caller-owned). The C-side strict-ON check
        # remains in the library for future use.
        self.zerocopy_enabled = self._v0_2_features_enabled and "zero_copy" in getattr(
            GroupConfig, "__dataclass_fields__", {}
        )

        # MPI sub-communicator scoped to the EP group. Mirrors the
        # DeepEPLowLatency pattern (see deep_ep_utils.py:104): split
        # MPI_COMM_WORLD by pp_rank so each pipeline stage gets its own EP
        # comm, keyed by moe_ep_rank. Avoids the wheel's
        # nccl.ep.get_nccl_comm_from_group() helper which requires
        # torch.distributed.init_process_group() -- the test infrastructure
        # (mpi_pool_executor) and microbenchmarks use MPI4PY only.
        self._ep_mpi_comm = mpi_comm().Split(mapping.pp_rank, mapping.moe_ep_rank)
        ep_world_rank = self._ep_mpi_comm.Get_rank()
        ep_world_size = self._ep_mpi_comm.Get_size()
        unique_id = nccl_core.get_unique_id() if ep_world_rank == 0 else None
        unique_id = self._ep_mpi_comm.bcast(unique_id, root=0)
        self.comm = nccl_core.Communicator.init(
            nranks=ep_world_size,
            rank=ep_world_rank,
            unique_id=unique_id,
        )

        cfg = GroupConfig(
            algorithm=Algorithm.LOW_LATENCY,
            num_experts=num_experts,
            max_dispatch_tokens_per_rank=max_tokens_per_rank,
            max_recv_tokens_per_rank=self.max_recv_tokens,
            max_token_bytes=hidden_size * 2,  # bfloat16
        )
        self.ep_group = Group.create(self.comm, cfg)

        logger.info(
            f"NCCL EP group created: ep_size={self.ep_size}, "
            f"num_experts={num_experts}, max_tokens_per_rank={max_tokens_per_rank}, "
            f"hidden_size={hidden_size}, max_top_k={max_top_k}, "
            f"layout={self.layout.name}"
        )

        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)

        # Dispatch output tokens: 3D [ep_size, max_tokens_per_rank, hidden]
        # for LL rank-major. When zerocopy is enabled the buffer must be
        # VMM-backed (cuMemMap) so ncclCommWindowRegister's internal
        # cuMemGetAddressRange call succeeds -- torch's caching
        # allocator returns plain cudaMalloc memory which fails that
        # check with CUDA_ERROR_INVALID_VALUE. Allocate via
        # nccl.core.mem_alloc (VMM-backed) then build a zero-copy torch
        # view over the raw pointer via the TRT-LLM CAI wrapper.
        token_shape = (self.ep_size, max_tokens_per_rank, hidden_size)
        token_nbytes = self.ep_size * max_tokens_per_rank * hidden_size * 2
        self._output_tokens_alloc = None
        self._recv_x_window = None
        if self.zerocopy_enabled:
            self._output_tokens_alloc = nccl_core.mem_alloc(
                token_nbytes,
                device=device_id,
            )
            self.output_tokens_buf = convert_to_torch_tensor(
                TensorWrapper(
                    int(self._output_tokens_alloc.handle),
                    dtype=torch.bfloat16,
                    shape=token_shape,
                )
            )
            self._recv_x_window = self.comm.register_window(
                self._output_tokens_alloc,
            )
        else:
            self.output_tokens_buf = torch.empty(
                *token_shape,
                dtype=torch.bfloat16,
                device=device,
            )
        # Received topk indices: int32 [ep_size, max_tokens_per_rank, max_top_k]
        # for the LL rank-major dispatch contract. -1 marks invalid rows.
        # Downstream consumers want 2D [max_recv, max_top_k]; flatten via view.
        self.recv_topk_idx_buf = torch.empty(
            self.ep_size,
            max_tokens_per_rank,
            max_top_k,
            dtype=torch.int32,
            device=device,
        )
        # Received topk weights: float32 [ep_size, max_tokens_per_rank, max_top_k]
        self.recv_topk_weights_buf = torch.empty(
            self.ep_size,
            max_tokens_per_rank,
            max_top_k,
            dtype=torch.float32,
            device=device,
        )
        # Per-source-rank received-token counter (passed via
        # LayoutInfo.src_rank_counters at dispatch time).
        self.recv_rank_counter_buf = torch.empty(
            self.ep_size,
            dtype=torch.int32,
            device=device,
        )
        # Wrap each persistent buffer as a Tensor descriptor. Torch owns the
        # storage; the descriptor only carries shape + a pointer (+ window
        # handle on dispatch output when zerocopy is on, so libnccl_ep's
        # opportunistic LL zero-copy path can fire).
        if self.zerocopy_enabled and self._recv_x_window is not None:
            self.output_tokens_nd = Tensor(
                self.output_tokens_buf,
                window=self._recv_x_window,
                window_offset=0,
            )
        else:
            self.output_tokens_nd = Tensor(self.output_tokens_buf)
        self.recv_topk_idx_nd = Tensor(self.recv_topk_idx_buf)
        self.recv_topk_weights_nd = Tensor(self.recv_topk_weights_buf)
        self.recv_rank_counter_nd = Tensor(self.recv_rank_counter_buf)

    def get_stream(self) -> int:
        """Current CUDA stream as a raw int handle (accepted by ``nccl.ep`` APIs)."""
        return torch.cuda.current_stream().cuda_stream

    def destroy(self):
        """Release EP group, NCCL comm, and MPI sub-comm in LIFO order.

        Avoids relying on Python GC ordering between the group, the comm it
        was built from, and the MPI sub-comm seeding it: the group must go
        first (uses the comm), then ``finalize`` + ``destroy`` on the comm
        (the recommended nccl4py pattern), then ``Free`` on the MPI comm.
        """
        if self.ep_group is not None:
            try:
                self.ep_group.destroy()
            except _NCCL_RUNTIME_ERRORS as e:
                logger.warning(f"NCCL EP group destroy error: {e}")
            self.ep_group = None

        # Deregister windows before the comm goes away. close() is
        # idempotent and local; the comm would auto-close any leftover
        # windows on destroy, but explicit LIFO release matches the rest
        # of this teardown path. Both dispatch-output and combine-input
        # windows are registered only when zerocopy is on.
        for attr in ("_combine_input_window", "_recv_x_window"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.close()
                except _NCCL_RUNTIME_ERRORS as e:
                    logger.warning(f"NCCL EP window close error ({attr}): {e}")
                setattr(self, attr, None)

        # Drop torch view + EP descriptor before freeing the underlying
        # NCCL-allocated Buffer (CAI view doesn't refcount the source).
        # close() the cuda.core.Buffer to call nccl.core.mem_free; the
        # alloc is only populated when zerocopy is enabled.
        self.output_tokens_nd = None
        self.output_tokens_buf = None
        if getattr(self, "_output_tokens_alloc", None) is not None:
            try:
                self._output_tokens_alloc.close()
            except _NCCL_RUNTIME_ERRORS as e:
                logger.warning(f"NCCL EP recv_x buffer free error: {e}")
            self._output_tokens_alloc = None

        if self.comm is not None:
            try:
                self.comm.finalize()
                self.comm.destroy()
            except _NCCL_RUNTIME_ERRORS as e:
                logger.warning(f"NCCL EP comm destroy error: {e}")
            self.comm = None

        if self._ep_mpi_comm is not None:
            from mpi4py import MPI

            try:
                self._ep_mpi_comm.Free()
            except MPI.Exception as e:
                logger.warning(f"EP MPI sub-comm free error: {e}")
            self._ep_mpi_comm = None


def get_nccl_ep_context(
    mapping: Mapping,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    max_top_k: int,
    layout: Optional[int] = None,
) -> NcclEpContext:
    """Get or create a singleton :class:`NcclEpContext` for the given configuration."""
    from nccl.ep import Layout

    if layout is None:
        layout = Layout.RANK_MAJOR
    key = (
        mapping.moe_ep_size,
        mapping.moe_ep_rank,
        max_tokens_per_rank,
        num_experts,
        hidden_size,
        max_top_k,
        int(layout),
    )
    if key not in _ep_group_cache:
        _ep_group_cache[key] = NcclEpContext(
            mapping,
            num_experts,
            max_tokens_per_rank,
            hidden_size,
            max_top_k,
            layout,
        )
    _ep_group_refcounts[key] = _ep_group_refcounts.get(key, 0) + 1
    return _ep_group_cache[key]


def release_nccl_ep_context(ctx: Optional[NcclEpContext]) -> None:
    """Release one reference to a cached :class:`NcclEpContext`."""
    if ctx is None:
        return

    key = next((key for key, cached_ctx in _ep_group_cache.items() if cached_ctx is ctx), None)
    if key is None:
        return

    refcount = _ep_group_refcounts.get(key, 0) - 1
    if refcount > 0:
        _ep_group_refcounts[key] = refcount
        return

    _ep_group_refcounts.pop(key, None)
    cached_ctx = _ep_group_cache.pop(key)
    try:
        cached_ctx.destroy()
    except _NCCL_RUNTIME_ERRORS as e:
        logger.warning(f"Error destroying NCCL EP context: {e}")


def destroy_all_nccl_ep_contexts():
    """Destroy all cached NCCL EP contexts (call at process teardown)."""
    for ctx in list(_ep_group_cache.values()):
        try:
            ctx.destroy()
        except _NCCL_RUNTIME_ERRORS as e:
            logger.warning(f"Error destroying NCCL EP context: {e}")
    _ep_group_cache.clear()
    _ep_group_refcounts.clear()
