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
"""The post_moe_hop contract: native raw NVFP4 MoE tuple -> post-reduction hidden/residual.

``PostMoeHop`` replaces, for one target MoE layer, the native multi-node
chain "MoE runner finalize -> shared add -> MNNVL all-reduce with residual and
RMSNorm" by one fused launch that consumes the runner's ``do_finalize=False``
outputs. It returns the same ``(normed_hidden, new_residual)`` pair as
``AllReduce(fusion_op=RESIDUAL_RMS_NORM)``.

Eligibility is decided before the MoE runner is asked for raw output, from
rank-identical facts only, so ranks never diverge in collective count or
order:

* when the hop is built (collective): the rank-local capability, the raw-tuple
  contract the MoE producer declares (``supports_unfinalized_output()``,
  hidden size, dtype, top-k), the post_moe_hop workspace allocation on every rank;
* per call (``can_run``): the token count.

After the runner the hop checks the raw tuple's metadata. A dtype or shape
that differs from the declared contract is a producer error and raises
``PostMoeHopContractError`` (the runner already skipped its finalize, so there is no
native path left to take). A layout the kernels cannot read directly (inner
stride, contiguity, a base or row pitch not aligned to 16 bytes) is copied
into an aligned contiguous tensor on this rank only, which leaves the
collective and the arithmetic unchanged; such copies are counted
(``PostMoeHopWorkspace.layout_repairs``) and logged once per tensor.

Device errors (a peer that never publishes within ``timeout_s``, a
permutation entry beyond the FC2 rows) set the sticky per-rank error word;
see ``health.check_health`` for the host-side report.
"""

import contextlib
import os
import platform
from typing import Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping

from ... import dispatch
from .workspace import PostMoeHopWorkspace, _all_ranks_agree

FEATURE_ID = "post_moe_hop"
DEFAULT_MAX_ROWS = 16
DEFAULT_BLOCK = 1024
DEFAULT_NUM_WARPS = 4
# Upper bound of any legitimate cross-rank skew at the hop, including the
# first decode-shaped forward, in which neighbouring packages may compile
# kernels per rank; below the executor's 300 s hang detector.
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CLUSTER = 8
IMPLS = ("triton", "cluster")
_SUPPORTED_WORLDS = (2, 4, 8)
_SCALE_DTYPES = (torch.bfloat16, torch.float32)
_VECTOR_BYTES = 16


class PostMoeHopContractError(RuntimeError):
    """The raw MoE tuple does not match the contract its producer declared."""


def _local_capability(
    mapping: Mapping,
    hidden_size: int,
    dtype: torch.dtype,
    block: int,
    num_warps: int,
    require_multi_node: bool,
) -> Tuple[bool, str]:
    """Rank-local preconditions; the caller converges them across the TP group."""
    if mapping.enable_attention_dp:
        return False, "attention DP"
    if mapping.has_cp():
        return False, "context parallelism"
    if mapping.tp_size not in _SUPPORTED_WORLDS:
        return False, f"tp_size {mapping.tp_size} not in {_SUPPORTED_WORLDS}"
    if require_multi_node and not mapping.is_multi_node():
        return False, "single-node TP group (native fused MoE all-reduce applies)"
    if dtype != torch.bfloat16:
        return False, f"dtype {dtype}"
    if block < 128 or block & (block - 1):
        return False, f"block {block} is not a power of two >= 128"
    if block < 256 * num_warps:
        return False, f"block {block} < 256 * num_warps ({256 * num_warps})"
    if hidden_size % 8 or -(-hidden_size // block) > 64:
        return False, f"hidden {hidden_size} unsupported for block {block}"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        return False, f"SM {major}x < 90"
    if "aarch64" not in platform.machine().lower() and os.environ.get("TLLM_TEST_MNNVL") != "1":
        return False, "MNNVL multicast workspace requires an aarch64 NVLink domain"
    try:
        import triton  # noqa: F401

        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    except ImportError as exc:
        return False, f"import failed: {exc}"
    if not MnnvlMemory.supports_mnnvl() and os.environ.get("TLLM_TEST_MNNVL") != "1":
        return False, "MNNVL not supported on this device"
    return True, ""


def producer_capability(
    producer, hidden_size: int, top_k: int, dtype: torch.dtype
) -> Tuple[bool, str]:
    """Static raw-tuple contract of a MoE module, read from its own declaration.

    ``producer`` is the layer's MoE module (``ConfigurableMoE`` or a backend
    leaf). It qualifies when its backend declares that ``do_finalize=False``
    returns the unfinalized ``(gemm2_output, expert_weights,
    expanded_idx_to_permuted_idx)`` triple, leaves the reduction to the
    caller, and matches the hop's hidden size, activation dtype and top-k.
    """
    backend = getattr(producer, "backend", producer)
    declares = getattr(backend, "supports_unfinalized_output", None)
    if not callable(declares) or not declares():
        return False, (
            f"MoE backend {type(backend).__name__} does not declare the unfinalized "
            "raw-tuple output"
        )
    if getattr(producer, "reduce_results", False):
        return False, "MoE producer reduces its own output"
    if getattr(producer, "hidden_size", hidden_size) != hidden_size:
        return False, f"MoE hidden size {producer.hidden_size} != {hidden_size}"
    if getattr(producer, "dtype", None) not in (None, dtype):
        return False, f"MoE dtype {producer.dtype} != {dtype}"
    routing = getattr(producer, "routing_method", None)
    experts_per_token = getattr(routing, "experts_per_token", top_k)
    if experts_per_token != top_k:
        return False, f"MoE top-k {experts_per_token} != {top_k}"
    return True, ""


def _vector_ready(t: torch.Tensor, row_major_2d: bool) -> bool:
    """Whether the kernels may read ``t`` with 16-byte vectors at its strides."""
    if t.data_ptr() % _VECTOR_BYTES or t.stride(-1) != 1:
        return False
    if row_major_2d:
        return (t.stride(0) * t.element_size()) % _VECTOR_BYTES == 0
    return t.is_contiguous()


class PostMoeHop:
    """Fused post-MoE finalize + cross-rank reduction + residual + RMSNorm.

    Args:
        mapping: TP mapping of the target model (non-ADP).
        hidden_size: model hidden size ``H``.
        top_k: routed experts per token.
        dtype: activation dtype (bf16 only).
        max_rows: largest token count the hop accepts per call.
        block: hidden columns per program (power of two dividing ``H``).
        num_warps: Triton warps per program.
        timeout_s: deadline of every cross-rank poll; on expiry the call sets
            the sticky error word, writes NaN outputs and completes.
        use_pdl: launch with programmatic dependent launch; defaults to the
            native ``TRTLLM_ENABLE_PDL`` policy (on for SM90+ unless set to 0).
        impl: ``"triton"`` (chunked programs, square sums exchanged through a
            local Lamport slot) or ``"cluster"`` (CuTe DSL, one thread block
            cluster per row, square sums reduced through distributed shared
            memory). Shapes the cluster kernel does not cover use Triton.
        cluster: CTAs per row for the cluster kernel.
        require_multi_node: restrict to TP groups that span hosts (the native
            single-node fused path already exists); tests may disable it.
        raw_producer: the layer's MoE module; its declared raw-tuple contract
            (``producer_capability``) is part of the collective decision.
            ``None`` skips that check (tests that feed synthetic tuples).
        local_veto: test-only reason that makes this rank incapable.
        fault_delay_ns: test-only protocol fault injection (scrambled store
            delays inside the kernels); production keeps 0.
    """

    def __init__(
        self,
        mapping: Mapping,
        hidden_size: int,
        top_k: int,
        dtype: torch.dtype = torch.bfloat16,
        max_rows: int = DEFAULT_MAX_ROWS,
        block: int = DEFAULT_BLOCK,
        num_warps: int = DEFAULT_NUM_WARPS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        use_pdl: Optional[bool] = None,
        impl: str = "cluster",
        cluster: int = DEFAULT_CLUSTER,
        require_multi_node: bool = True,
        raw_producer: Optional[object] = None,
        local_veto: Optional[str] = None,
        fault_delay_ns: int = 0,
    ):
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.dtype = dtype
        self.max_rows = max_rows
        self.block = block
        self.num_warps = num_warps
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.timeout_ns = int(timeout_s * 1e9)
        self.fault_delay_ns = fault_delay_ns
        self.workspace: Optional[PostMoeHopWorkspace] = None
        if use_pdl is None:
            use_pdl = (
                os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0"
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
            )
        self.use_pdl = use_pdl
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
        self.cluster = cluster
        self.impl = impl
        if impl == "cluster":
            from .cute_kernel import cluster_kernel_supported

            why = cluster_kernel_supported(hidden_size, top_k, mapping.tp_size, cluster)
            if why is not None:
                dispatch.log_capability(FEATURE_ID, "cluster_kernel", False, why)
                self.impl = "triton"

        local_ok, reason = _local_capability(
            mapping, hidden_size, dtype, block, num_warps, require_multi_node
        )
        if local_ok and raw_producer is not None:
            local_ok, reason = producer_capability(raw_producer, hidden_size, top_k, dtype)
        if local_veto is not None:
            local_ok, reason = False, local_veto
        self.engaged = False
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            comm = PostMoeHopWorkspace.comm_for(mapping)
            if not _all_ranks_agree(comm, local_ok):
                reason = reason or "another rank is not capable"
            else:
                self.workspace, reason = PostMoeHopWorkspace.get_or_create(
                    mapping, hidden_size, max_rows
                )
                self.engaged = self.workspace is not None
        if self.engaged and self.impl == "cluster":
            self._precompile_cluster_kernel(comm)
        self.reason = reason
        self._repairs_logged = set()
        dispatch.log_capability(FEATURE_ID, f"post_moe_hop[{self.impl}]", self.engaged, reason)

    def _precompile_cluster_kernel(self, comm) -> None:
        """Compile the cluster kernel now, outside any CUDA graph capture.

        Covers both shared-output variants with bf16 routing weights (the
        TRTLLM backend's runner emits them in the activation dtype); other
        variants compile on first use. Ranks agree on the outcome, so a
        compile failure moves every rank to the Triton kernel together.
        """
        from .cute_kernel import compiled_cluster_hop
        from .kernels import STATE_WORDS

        ok = True
        try:
            for has_shared in (True, False):
                compiled_cluster_hop(
                    self.hidden_size,
                    self.top_k,
                    self.mapping.tp_size,
                    self.max_rows,
                    self.cluster,
                    has_shared,
                    False,
                    self.use_pdl,
                    STATE_WORDS,
                    False,
                    self.fault_delay_ns,
                )
        except Exception as exc:  # noqa: BLE001 - converged across ranks below
            ok = False
            dispatch.log_capability(FEATURE_ID, "cluster_kernel_compile", False, repr(exc))
        if not _all_ranks_agree(comm, ok):
            self.impl = "triton"

    def can_run(self, num_rows: int) -> bool:
        """Whether a call with ``num_rows`` tokens takes the post_moe_hop path (rank-identical)."""
        return self.engaged and 0 < num_rows <= self.max_rows

    # ------------------------------------------------------------------
    # routed_moe_into_hop entry: reachable only when the ``routed_moe_into_hop`` feature is
    # selected; nothing else calls it, so the post_moe_hop is unchanged otherwise.
    # ------------------------------------------------------------------
    ROUTED_MODES = ("row", "identity", "fp32")
    # fp32: no finalize kernel and no raw rows; identity
    # is the bit-exact alternative. The selector's variant table of
    # routed_moe_into_hop is the one source of the default and of the names
    # its knob selects; row stays a mode of the hop that no knob selects.
    DEFAULT_ROUTED_MODE = dispatch.NUMERICAL_VARIANTS["routed_moe_into_hop"][1][0]
    assert set(dispatch.NUMERICAL_VARIANTS["routed_moe_into_hop"][1]) <= set(ROUTED_MODES)

    @staticmethod
    def selected_routed_mode() -> str:
        """routed_moe_into_hop's routed-input mode selected by the environment.

        The variable
        (``TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION``) is
        parsed by the selector (``dispatch.feature_variants``), the parser of
        the feature record, so the record and the hop name the same variant;
        unset or empty selects ``DEFAULT_ROUTED_MODE``.

        Raises:
            ValueError: the variable names no routed_moe_into_hop mode.
        """
        return dispatch.feature_variants(("routed_moe_into_hop",))["routed_moe_into_hop"]

    def enable_routed_input(self, mode: Optional[str] = None) -> bool:
        """Collective: prepare the hop for a routed input written by routed_moe.

        ``mode`` (default: ``selected_routed_mode()``, i.e.
        ``TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION`` as the
        selector parses it) selects what routed_moe writes and the cluster-kernel
        variant that reads it: "row" a finalized bf16 row, "identity" the
        identity-layout raw rows and their scales, "fp32" an FP32 accumulator
        of the top-k contributions (cleared by the hop after reading).
        Allocates the graph-stable buffer of that mode in device memory (also
        when called during the served model construction, see
        ``_routed_device_buffer``), checks it, and precompiles its variants;
        every TP rank must succeed with the same mode, otherwise the feature
        stays off on all of them. May be called again (collectively, outside
        capture) to switch modes.
        """
        mode = mode or self.selected_routed_mode()
        if mode not in self.ROUTED_MODES:
            raise ValueError(
                f"routed_moe_into_hop routed-input mode {mode!r} not in {self.ROUTED_MODES}"
            )
        if getattr(self, "_routed_enabled", None) is not None and self.routed_mode == mode:
            return self._routed_enabled
        ok = self.engaged and self.impl == "cluster"
        if ok:
            try:
                from .cute_kernel import compiled_cluster_hop
                from .kernels import STATE_WORDS

                for has_shared in (True, False):
                    compiled_cluster_hop(
                        self.hidden_size,
                        self.top_k,
                        self.mapping.tp_size,
                        self.max_rows,
                        self.cluster,
                        has_shared,
                        False,
                        self.use_pdl,
                        STATE_WORDS,
                        False,
                        self.fault_delay_ns,
                        mode,
                    )
                if mode == "row" and self._needs_routed_buffer("_routed_buffer"):
                    self._routed_buffer = self._routed_device_buffer(
                        (self.max_rows, self.hidden_size), self.dtype
                    )
                if mode == "identity" and (
                    self._needs_routed_buffer("_routed_rows")
                    or self._needs_routed_buffer("_routed_scales")
                ):
                    self._routed_rows = self._routed_device_buffer(
                        (self.max_rows * self.top_k, self.hidden_size), self.dtype
                    )
                    self._routed_scales = self._routed_device_buffer(
                        (self.max_rows, self.top_k), self.dtype
                    )
                if mode == "fp32" and self._needs_routed_buffer("_routed_acc"):
                    # the routed kernel accumulates into it; the hop clears
                    # the rows it read, so it is zero before every call
                    self._routed_acc = self._routed_device_buffer(
                        (self.max_rows, self.hidden_size), torch.float32
                    )
                problem = self._routed_buffer_problem(mode)
                if problem:
                    dispatch.log_capability(
                        "routed_moe_into_hop", "post_moe_routed_input_buffer", False, problem
                    )
                    ok = False
            except Exception as exc:  # noqa: BLE001 - converged across ranks below
                dispatch.log_capability(
                    "routed_moe_into_hop", "post_moe_routed_input_compile", False, repr(exc)
                )
                ok = False
        comm = PostMoeHopWorkspace.comm_for(self.mapping)
        # The mode comes from each rank's environment: require the same mode
        # on every rank (one agreement per mode, in a fixed order), so no rank
        # writes or reads another variant than its peers.
        if not [m for m in self.ROUTED_MODES if _all_ranks_agree(comm, mode == m)]:
            dispatch.log_capability(
                "routed_moe_into_hop",
                "post_moe_routed_input_mode",
                False,
                f"routed-input modes differ across ranks (this rank: {mode})",
            )
            ok = False
        self._routed_enabled = _all_ranks_agree(comm, ok)
        self.routed_mode = mode
        dispatch.log_capability(
            "routed_moe_into_hop", f"post_moe_routed_input[{mode}]", self._routed_enabled
        )
        return self._routed_enabled

    _ROUTED_BUFFERS = {
        "row": ("_routed_buffer",),
        "identity": ("_routed_rows", "_routed_scales"),
        "fp32": ("_routed_acc",),
    }

    def _needs_routed_buffer(self, name: str) -> bool:
        buf = getattr(self, name, None)
        return buf is None or buf.is_meta

    def _routed_device_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A zeroed routed-input buffer in device memory on the workspace's device.

        The served model loader constructs the model, and with it this hop,
        inside ``MetaInitMode``: a dispatch mode that turns ``empty`` /
        ``empty_like`` into meta tensors (data pointer 0) and afterwards
        materializes only module parameters and buffers. The hop is not a
        module, so its buffers are allocated with every dispatch mode
        suspended.
        """
        try:
            from torch.utils._python_dispatch import _disable_current_modes
        except ImportError:  # the result is still checked by _routed_buffer_problem
            _disable_current_modes = contextlib.nullcontext
        with _disable_current_modes():
            return torch.zeros(shape, dtype=dtype, device=self.workspace.state.device)

    def _routed_buffer_problem(self, mode: str) -> str:
        """Why ``mode``'s buffers cannot be handed to the kernels ("" when they can)."""
        device = self.workspace.state.device
        for name in self._ROUTED_BUFFERS[mode]:
            buf = getattr(self, name, None)
            if buf is None or buf.is_meta or buf.device != device:
                where = None if buf is None else buf.device
                return f"{name} is on {where}, not in device memory on {device}"
            if not buf.is_contiguous() or buf.data_ptr() % _VECTOR_BYTES:
                return f"{name} is not a contiguous 16-byte aligned buffer"
        return ""

    def routed_input_buffer(self, num_rows: int) -> torch.Tensor:
        """Mode "row": the ``[num_rows, H]`` view routed_moe writes its finalized output into."""
        return self._routed_buffer[:num_rows]

    def routed_raw_buffers(self, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mode "identity": ``(rows [num_rows * top_k, H], scales [num_rows, top_k])``."""
        return self._routed_rows[: num_rows * self.top_k], self._routed_scales[:num_rows]

    def routed_acc_buffer(self, num_rows: int) -> torch.Tensor:
        """Mode "fp32": the zeroed FP32 ``[num_rows, H]`` accumulator."""
        return self._routed_acc[:num_rows]

    def _routed_view(self, num_rows: int) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:
        """``(mode, buffer view, scales)`` the hop reads in its routed-input mode."""
        if self.routed_mode == "fp32":
            return "fp32", self.routed_acc_buffer(num_rows), None
        if self.routed_mode == "identity":
            rows, scales = self.routed_raw_buffers(num_rows)
            return "identity", rows, scales
        return "row", self.routed_input_buffer(num_rows), None

    def hop_routed(
        self,
        routed: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The hop on a routed input written by routed_moe (routed_moe_into_hop).

        ``routed`` is the buffer view ``routed_output_into_hop`` returned (a
        view of this hop's own buffer in its routed-input mode). Same
        arithmetic as ``__call__`` after its finalize: ``bf16(routed +
        shared)``, rank-ordered FP32 sum over the TP group, residual added
        once, RMSNorm.

        Raises:
            ValueError: ``can_run`` is false or the routed input is not enabled.
            PostMoeHopContractError: ``routed`` is not this hop's buffer for the row
                count, or a dtype / shape differs from the contract.
        """
        num_rows = residual.shape[0]
        if not (self.can_run(num_rows) and getattr(self, "_routed_enabled", False)):
            raise ValueError(
                "routed_moe_into_hop hop called for an ineligible shape or before enable_routed_input()"
            )
        mode, expect, scales = self._routed_view(num_rows)
        rows_h = (num_rows, self.hidden_size)
        problems = []
        if (
            routed.data_ptr() != expect.data_ptr()
            or routed.dtype != expect.dtype
            or tuple(routed.shape) != tuple(expect.shape)
            or tuple(routed.stride()) != tuple(expect.stride())
        ):
            problems.append(
                f"routed {routed.dtype} {tuple(routed.shape)} is not the hop's {mode} buffer"
            )
        if residual.dtype != self.dtype or tuple(residual.shape) != rows_h:
            problems.append(f"residual {residual.dtype} {tuple(residual.shape)}")
        if shared_output is not None and (
            shared_output.dtype != self.dtype or tuple(shared_output.shape) != rows_h
        ):
            problems.append(f"shared_output {shared_output.dtype} {tuple(shared_output.shape)}")
        if norm_weight.dtype != self.dtype or tuple(norm_weight.shape) != (self.hidden_size,):
            problems.append(f"norm_weight {norm_weight.dtype} {tuple(norm_weight.shape)}")
        if problems:
            raise PostMoeHopContractError(
                "routed_moe_into_hop routed input violates the hop's contract: "
                f"{'; '.join(problems)}"
            )
        if shared_output is not None:
            shared_output = self._vector_layout("shared_output", shared_output)
        residual = self._vector_layout("residual", residual)
        norm_weight = self._vector_layout("norm_weight", norm_weight)
        from .cute_kernel import launch_cluster_hop_routed

        out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        workspace = self.workspace
        launch_cluster_hop_routed(
            routed=routed,
            shared_output=shared_output,
            residual=residual,
            norm_weight=norm_weight,
            out=out,
            residual_out=residual_out,
            mc_buffer=workspace.mc,
            uc_buffer=workspace.uc,
            partials=workspace.partials,
            state=workspace.state,
            rank=self.mapping.tp_rank,
            world=self.mapping.tp_size,
            max_rows=self.max_rows,
            eps=eps,
            cluster=self.cluster,
            use_pdl=self.use_pdl,
            timeout_ns=self.timeout_ns,
            top_k=self.top_k,
            mode=mode,
            scales=scales,
            fault_delay_ns=self.fault_delay_ns,
        )
        return out, residual_out

    def _contract_violations(
        self,
        fc2_output,
        expert_scale_factor,
        expanded_idx_to_permuted_idx,
        shared_output,
        residual,
        norm_weight,
    ) -> list:
        """Dtype/shape differences from the declared raw-tuple contract (rank-identical)."""
        num_rows = residual.shape[0]
        rows_h = (num_rows, self.hidden_size)
        problems = []
        if residual.dtype != self.dtype or tuple(residual.shape) != rows_h:
            problems.append(f"residual {residual.dtype} {tuple(residual.shape)}")
        if (
            fc2_output.dtype != self.dtype
            or fc2_output.dim() != 2
            or fc2_output.shape[1] < self.hidden_size
        ):
            problems.append(f"fc2_output {fc2_output.dtype} {tuple(fc2_output.shape)}")
        if expert_scale_factor.dtype not in _SCALE_DTYPES or tuple(expert_scale_factor.shape) != (
            num_rows,
            self.top_k,
        ):
            problems.append(
                f"expert_scale_factor {expert_scale_factor.dtype} "
                f"{tuple(expert_scale_factor.shape)}"
            )
        if expanded_idx_to_permuted_idx.dtype != torch.int32 or tuple(
            expanded_idx_to_permuted_idx.shape
        ) != (num_rows, self.top_k):
            problems.append(
                f"expanded_idx_to_permuted_idx {expanded_idx_to_permuted_idx.dtype} "
                f"{tuple(expanded_idx_to_permuted_idx.shape)}"
            )
        if shared_output is not None and (
            shared_output.dtype != self.dtype or tuple(shared_output.shape) != rows_h
        ):
            problems.append(f"shared_output {shared_output.dtype} {tuple(shared_output.shape)}")
        if norm_weight.dtype != self.dtype or tuple(norm_weight.shape) != (self.hidden_size,):
            problems.append(f"norm_weight {norm_weight.dtype} {tuple(norm_weight.shape)}")
        return problems

    def _vector_layout(self, name: str, t: torch.Tensor, row_major_2d: bool = False):
        """Return ``t`` or a 16-byte-aligned contiguous copy of it (rank-local)."""
        if _vector_ready(t, row_major_2d):
            return t
        misalignment, strides = t.data_ptr() % _VECTOR_BYTES, tuple(t.stride())
        if row_major_2d:
            t = t[:, : self.hidden_size]
        fixed = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        fixed.copy_(t)
        self.workspace.layout_repairs += 1
        if name not in self._repairs_logged:
            self._repairs_logged.add(name)
            from tensorrt_llm.logger import logger

            logger.warning(
                f"[glm_moe_dsa] post_moe_hop/hop: {name} is not ready for 16-byte vector "
                f"access (data_ptr % 16 = {misalignment}, strides {strides}); "
                "copying it before every such call"
            )
        return fixed

    def __call__(
        self,
        fc2_output: torch.Tensor,
        expert_scale_factor: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        *,
        trace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rmsnorm(reduced + residual), reduced + residual)``.

        Args:
            fc2_output: bf16 ``[N_padded, >=H]`` raw FC2 rows of the native runner.
            expert_scale_factor: ``[R, top_k]`` routing weights (bf16 or fp32).
            expanded_idx_to_permuted_idx: int32 ``[R, top_k]``; negative = no local
                row; entries ``>= N_padded`` are skipped and set error bit 4.
            shared_output: bf16 ``[R, H]`` rank-local shared-expert partial, or ``None``.
            residual: bf16 ``[R, H]`` stock residual (added once, after the reduction).
            norm_weight: bf16 ``[H]`` next-layer RMSNorm weight.
            eps: RMSNorm epsilon.
            trace: diagnostics only; int64 ``[programs, 8]`` phase timestamps.

        Raises:
            ValueError: ``can_run`` is false for this row count.
            PostMoeHopContractError: a dtype or shape differs from the raw-tuple contract.
        """
        if not self.can_run(residual.shape[0]):
            raise ValueError(
                "post_moe_hop hop called for an ineligible shape; check can_run() first"
            )
        problems = self._contract_violations(
            fc2_output,
            expert_scale_factor,
            expanded_idx_to_permuted_idx,
            shared_output,
            residual,
            norm_weight,
        )
        if problems:
            raise PostMoeHopContractError(
                "raw MoE tuple violates the declared native raw-tuple contract after the runner "
                f"skipped its finalize: {'; '.join(problems)}"
            )
        fc2_output = self._vector_layout("fc2_output", fc2_output, row_major_2d=True)
        expert_scale_factor = self._vector_layout("expert_scale_factor", expert_scale_factor)
        expanded_idx_to_permuted_idx = self._vector_layout(
            "expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx
        )
        if shared_output is not None:
            shared_output = self._vector_layout("shared_output", shared_output)
        residual = self._vector_layout("residual", residual)
        norm_weight = self._vector_layout("norm_weight", norm_weight)

        out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        workspace = self.workspace
        if self.impl == "cluster" and trace is None:
            from .cute_kernel import launch_cluster_hop

            launch_cluster_hop(
                fc2_output=fc2_output,
                expert_scale_factor=expert_scale_factor,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                shared_output=shared_output,
                residual=residual,
                norm_weight=norm_weight,
                out=out,
                residual_out=residual_out,
                mc_buffer=workspace.mc,
                uc_buffer=workspace.uc,
                partials=workspace.partials,
                state=workspace.state,
                rank=self.mapping.tp_rank,
                world=self.mapping.tp_size,
                max_rows=self.max_rows,
                eps=eps,
                cluster=self.cluster,
                use_pdl=self.use_pdl,
                timeout_ns=self.timeout_ns,
                fault_delay_ns=self.fault_delay_ns,
            )
            return out, residual_out
        from .kernels import launch_post_moe_hop

        launch_post_moe_hop(
            fc2_output=fc2_output,
            expert_scale_factor=expert_scale_factor,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            shared_output=shared_output,
            residual=residual,
            norm_weight=norm_weight,
            out=out,
            residual_out=residual_out,
            mc_buffer=workspace.mc,
            uc_buffer=workspace.uc,
            partials=workspace.partials,
            state=workspace.state,
            rank=self.mapping.tp_rank,
            world=self.mapping.tp_size,
            max_rows=self.max_rows,
            eps=eps,
            block=self.block,
            num_warps=self.num_warps,
            timeout_ns=self.timeout_ns,
            use_pdl=self.use_pdl,
            trace=trace,
            fault_delay_ns=self.fault_delay_ns,
        )
        return out, residual_out

    def error_bits(self) -> int:
        """Sticky device error word of this rank (synchronizes); 0 means healthy."""
        return 0 if self.workspace is None else self.workspace.error_bits()

    def raise_if_error(self) -> None:
        """Raise if any program of any earlier call failed on this rank."""
        bits = self.error_bits()
        if bits:
            raise RuntimeError(
                f"post_moe_hop post-MoE hop reported error bits {bits:#x} on TP rank "
                f"{self.mapping.tp_rank}; the run is invalid and the workspace "
                "needs a collective reset()"
            )

    def reset(self) -> bool:
        """Collective workspace reset after an error; see ``PostMoeHopWorkspace.reset``."""
        return self.workspace.reset() if self.workspace is not None else True


def native_finalize_reference(
    fc2_output: torch.Tensor,
    expert_scale_factor: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """Torch version of the runner's finalize: fp32 FMA chain over top-k, one bf16 rounding.

    Test oracle. Each step ``acc = fma(w_k, x_k, acc)`` is evaluated in fp64
    and rounded to fp32 (the bf16 x fp32 product is exact in fp64), which
    reproduces the native kernel's fused multiply-add in ``k`` order. Entries
    outside ``[0, N_padded)`` contribute nothing, like in the post_moe_hop kernels.
    """
    num_rows, top_k = expanded_idx_to_permuted_idx.shape
    idx = expanded_idx_to_permuted_idx.to(torch.long)
    valid = (idx >= 0) & (idx < fc2_output.shape[0])
    safe = torch.where(valid, idx, torch.zeros_like(idx))
    rows = fc2_output[safe.view(-1), :hidden_size].view(num_rows, top_k, hidden_size)
    weights = expert_scale_factor.double()
    acc = torch.zeros((num_rows, hidden_size), dtype=torch.float32, device=fc2_output.device)
    for k in range(top_k):
        step = (acc.double() + weights[:, k : k + 1] * rows[:, k].double()).float()
        acc = torch.where(valid[:, k : k + 1], step, acc)
    return acc.to(fc2_output.dtype)
