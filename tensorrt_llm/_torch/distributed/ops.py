import math
import os
import platform
import sys
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._mnnvl_utils import HelixCpMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.distributed.allreduce_helper import \
    CustomAllReduceHelper
from tensorrt_llm._torch.distributed.symm_mem_allreduce import \
    SymmetricMemoryAllReduce
from tensorrt_llm._torch.q35_env import register as _q35_register
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AllReduceStrategy, MoEAllReduceParams)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Feature flag: GEMM→NCCL-window zero-copy (writes GEMM output directly into
# the window buffer so the allreduce needs no extra copy).  On by default
# (1); set TLLM_NCCL_SYMMETRIC_ZERO_COPY=0 to disable.
# Evaluated once at import — O(1) module-global lookup on every call,
# equivalent to a C++ static-bool cached env var.
_NCCL_SYMMETRIC_ZERO_COPY: bool = (os.environ.get(
    "TLLM_NCCL_SYMMETRIC_ZERO_COPY", "1") == "1")

_MNNVL_ONE_SHOT_THRESHOLD_BYTES = 64 * 1024 * 8 * 2

# Qwen3.5 low-latency comm flags (default off), read once at import.
# TRTLLM_Q35_AR_LOG=1: log which path every AllReduce construction site and call shape takes
# (MNNVL one-shot / two-shot, C++ AUTO / NCCL) and every all-gather call site.
_Q35_AR_LOG: bool = os.environ.get("TRTLLM_Q35_AR_LOG", "0") == "1"
_Q35_AR_NORM_FUSE: bool
_Q35_MNNVL_PRESIZE: bool


@_q35_register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global _Q35_AR_NORM_FUSE, _Q35_MNNVL_PRESIZE
    # TRTLLM_Q35_AR_NORM_FUSE=1: MNNVL serves RMS_NORM as RESIDUAL_RMS_NORM with a zero residual, so the
    # callers can fold their standalone RMSNorms into the all-reduce (see q35_zero_residual).
    _Q35_AR_NORM_FUSE = os.environ.get("TRTLLM_Q35_AR_NORM_FUSE", "0") == "1"
    # TRTLLM_Q35_MNNVL_PRESIZE=1: the MNNVL all-reduce workspace of a TP group is created once, at its first
    # use (model construction), large enough for the model's allreduce_max_num_tokens rows, and is never
    # re-created afterwards; a call that would need more takes the C++ / NCCL fallback instead. Without it the
    # workspace grows lazily, i.e. at the first long prefill (KV-cache profiling), and every McastGPUBuffer
    # construction is a collective fabric allocation. The one construction left per TP group is fenced
    # (_q35_fenced_mcast_buffer): a workaround for a known staging-buffer overrun in the multicast handle
    # exchange; a no-op once the library fix is present.
    _Q35_MNNVL_PRESIZE = os.environ.get("TRTLLM_Q35_MNNVL_PRESIZE", "0") == "1"


# TRTLLM_Q35_PROBE_NO_AR=1: ATTRIBUTION PROBE ONLY, WRONG OUTPUTS. Every all-reduce is replaced by
# its local epilogue (residual add + RMSNorm [+ FP8 quant]) without any communication, to bound
# the exposed all-reduce cost of a decode step against the stock step.
_Q35_PROBE_NO_AR: bool = os.environ.get("TRTLLM_Q35_PROBE_NO_AR", "0") == "1"
# TRTLLM_Q35_MCAST_STRAY_LOG=1: DIAGNOSTIC for the fenced construction of TRTLLM_Q35_MNNVL_PRESIZE
# (_q35_fenced_mcast_buffer). Before every McastGPUBuffer construction on ranks > 0, log which live pinned torch
# tensors (and which driver allocation) sit next to the handle-exchange staging buffer; constructions where that
# memory is not mapped are fenced instead.
_Q35_MCAST_STRAY_LOG: bool = os.environ.get("TRTLLM_Q35_MCAST_STRAY_LOG",
                                            "0") == "1"
# Largest token count for which the zero-residual buffer is used (decode / spec-verify shapes).
_Q35_ZERO_RESIDUAL_MAX_TOKENS = 1024

_thread_local = threading.local()

_q35_zero_residuals: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def q35_zero_residual(shape, dtype: torch.dtype,
                      device: torch.device) -> Optional[torch.Tensor]:
    """A read-only zero tensor of ``shape`` (at most 1024 rows of ``shape[-1]``), or None.

    Feeding it as the residual of RESIDUAL_RMS_NORM gives RMSNorm(allreduce(x)) and, as the
    residual output, allreduce(x) itself: the kernel adds the residual to the reduced sum, and adding
    zero changes no value. One buffer per (device, dtype) is shared by every caller. It is only ever
    allocated outside CUDA-graph capture (a capture would record its memset without running it); a
    call that would need it first inside a capture gets None and must take the unfused path.
    """
    shape = tuple(shape)
    hidden = shape[-1]
    numel = math.prod(shape)
    if numel > _Q35_ZERO_RESIDUAL_MAX_TOKENS * hidden:
        return None
    key = (torch.device(device), dtype)
    buf = _q35_zero_residuals.get(key)
    if buf is None or buf.numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.zeros(max(numel, _Q35_ZERO_RESIDUAL_MAX_TOKENS * hidden),
                          dtype=dtype,
                          device=device)
        _q35_zero_residuals[key] = buf
    return buf[:numel].view(shape)


def q35_mnnvl_fusable(allreduce: Optional["AllReduce"],
                      num_tokens: int) -> bool:
    """Whether ``allreduce`` serves fused calls of ``num_tokens`` rows on MNNVL (the zero-residual
    norm folds use it only then, so they never land on the C++ / NCCL fallback)."""
    return (allreduce is not None and allreduce.mnnvl_allreduce is not None
            and 0 < num_tokens <= _Q35_ZERO_RESIDUAL_MAX_TOKENS)


def _q35_caller_site(skip_files: Tuple[str, ...] = ()) -> str:
    """file:line of the first frame outside this module (and ``skip_files``)."""
    frame = sys._getframe(1)
    while frame is not None and (frame.f_code.co_filename == __file__
                                 or os.path.basename(
                                     frame.f_code.co_filename) in skip_files):
        frame = frame.f_back
    if frame is None:
        return "?"
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


_q35_logged: set = set()


def _q35_log_once(key, message: str, warn: bool = False) -> None:
    if key in _q35_logged:
        return
    _q35_logged.add(key)
    (logger.warning if warn else logger.info)(message)


def get_allreduce_workspace(mapping: Mapping) -> torch.LongTensor:
    if not hasattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}'):
        setattr(_thread_local, f'allreduce_workspaces_{mapping.pp_rank}', {})

    allreduce_workspaces = getattr(_thread_local,
                                   f'allreduce_workspaces_{mapping.pp_rank}')
    if mapping not in allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_auto(
                mapping.tp_size, support_deterministic=False),
        )
        allreduce_workspaces[mapping] = (ipc_buffers, workspace)
    return allreduce_workspaces[mapping][1]


def allocate_low_presicion_allreduce_workspace(mapping: Mapping) -> None:
    if not hasattr(_thread_local, 'lowprecision_allreduce_workspaces'):
        _thread_local.lowprecision_allreduce_workspaces = {}
    lowprecision_allreduce_workspaces = _thread_local.lowprecision_allreduce_workspaces
    if mapping not in lowprecision_allreduce_workspaces:
        ipc_buffers, workspace = CustomAllReduceHelper.allocate_lowprecision_workspace(
            mapping,
            CustomAllReduceHelper.max_workspace_size_lowprecision(
                mapping.tp_size),
        )
        lowprecision_allreduce_workspaces[mapping] = (ipc_buffers, workspace)
        CustomAllReduceHelper.initialize_lowprecision_buffers(
            workspace, mapping.tp_size)
    return


def _q35_mnnvl_presize_bytes(mapping: Mapping, dtype: torch.dtype) -> int:
    """TRTLLM_Q35_MNNVL_PRESIZE: bytes per Lamport buffer for the largest call of the model being built
    (allreduce_max_num_tokens rows of allreduce_hidden_size, recorded by AllReduce.__init__), or 0."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
    tokens = AllReduceRunner._prealloc_max_num_tokens
    hidden = AllReduceRunner._prealloc_hidden_size
    if not tokens or not hidden:
        return 0
    return MNNVLAllReduce.get_required_workspace_size(tokens, hidden,
                                                      mapping.tp_size, dtype)


_q35_cudart_lib = None
# (tp_rank, fenced, outcome) of every fenced construction (tests / diagnostics); outcome: "landed" (the fence
# caught the overrun where expected), "none" (fence untouched: a library with the fix), "elsewhere" (fence bytes
# changed somewhere else), "" (not fenced)
_q35_fence_log: List[Tuple[int, bool, str]] = []


def _q35_cudart():
    """ctypes handle on the CUDA runtime that libtensorrt_llm.so allocates its pinned host memory with."""
    global _q35_cudart_lib
    if _q35_cudart_lib is None:
        import ctypes
        paths = []
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split()[-1]
                if "libcudart.so" in os.path.basename(
                        path) and path not in paths:
                    paths.append(path)
        lib = ctypes.CDLL(paths[0] if paths else "libcudart.so")
        lib.cudaMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        ]
        lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        _q35_cudart_lib = lib
    return _q35_cudart_lib


def _q35_fenced_mcast_buffer(workspace_size_bytes: int, mapping: Mapping,
                             use_fabric_handle: bool, comm) -> McastGPUBuffer:
    """McastGPUBuffer(...) with a pinned-host fence after its handle-exchange staging buffer
    (TRTLLM_Q35_MNNVL_PRESIZE).

    Workaround for a known staging-buffer overrun in the multicast handle exchange; a no-op once the
    library fix is present. The pinned host allocator hands a just-freed block of the same size back
    first, so a contiguous run of same-size blocks is allocated, its first block freed (the staging buffer
    lands there) and the rest held as the fence until the construction is done.
    """
    import ctypes

    def construct():
        return McastGPUBuffer(workspace_size_bytes, mapping.tp_size,
                              mapping.tp_rank, mapping.local_rank,
                              use_fabric_handle, comm.py2f())

    if not use_fabric_handle or mapping.tp_rank == 0:
        _q35_fence_log.append((mapping.tp_rank, False, ""))
        return construct()
    rt = _q35_cudart()
    size = mapping.tp_size * 64
    span = mapping.tp_rank * 4096 + 64
    held, run = [], []
    fenced = False
    try:
        while len(held) < 1024:
            p = ctypes.c_void_p()
            if rt.cudaMallocHost(ctypes.byref(p), size) != 0 or not p.value:
                break
            held.append(p.value)
            if len(run) >= 2 and p.value - run[-1] != run[1] - run[0]:
                run = [
                ]  # a new chunk: start the run over (the old blocks stay held)
            if len(run) == 1 and not 0 < p.value - run[0] <= 4096:
                run = []
            run.append(p.value)
            if len(run) >= 2 and run[-1] + (run[1] - run[0]) - run[0] >= span:
                fenced = True
                break
        if fenced:
            for p in run[1:]:
                ctypes.memset(p, 0x5A, size)
            held.remove(run[0])
            rt.cudaFreeHost(run[0])
        buf = construct()
        outcome = ""
        if fenced:
            blk = run[0] + mapping.tp_rank * 4096
            pattern = b"\x5a" * size
            changed = [
                b for b in run[1:] if ctypes.string_at(b, size) != pattern
            ]
            if not changed:
                outcome = "none"  # fixed library: no stray write at all
            elif all(b <= blk < b + size for b in changed) and ctypes.string_at(
                    blk, 64) != pattern[:64]:
                outcome = "landed"
            else:
                outcome = "elsewhere"
                _q35_log_once(
                    ("mcast_fence_miss", mapping.tp_rank),
                    "[Q35 comm] McastGPUBuffer wrote into its fence at an unexpected place",
                    warn=True)
        else:
            _q35_log_once(
                ("mcast_fence_none", mapping.tp_rank),
                "[Q35 comm] could not build a pinned fence for McastGPUBuffer",
                warn=True)
        _q35_fence_log.append((mapping.tp_rank, fenced, outcome))
        return buf
    finally:
        for p in held:
            rt.cudaFreeHost(p)


def _q35_host_mapping(addr: int):
    """(start, end) of the readable and writable mapping containing ``addr``, or None (also for reserved
    PROT_NONE ranges such as the driver's guard regions)."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            lo, hi = (int(x, 16) for x in fields[0].split("-"))
            if lo <= addr < hi:
                return (lo, hi) if fields[1].startswith("rw") else None
    return None


def _q35_stray_target_report(mapping: Mapping, staging: int, chunk, target: int,
                             mapped: bool) -> str:
    """Diagnostic: the driver allocation and the live pinned torch tensors covering ``target``."""
    import ctypes
    import gc
    import warnings
    drv = "?"
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        start, rsize = ctypes.c_uint64(0), ctypes.c_size_t(0)
        e1 = cuda.cuPointerGetAttribute(ctypes.byref(start), 11,
                                        ctypes.c_uint64(target))
        e2 = cuda.cuPointerGetAttribute(ctypes.byref(rsize), 12,
                                        ctypes.c_uint64(target))
        drv = (
            f"driver range +{target - start.value} in [{start.value:#x}, +{rsize.value})"
            if e1 == 0 and e2 == 0 else f"no driver allocation (err {e1}/{e2})")
    except OSError as e:
        drv = f"no libcuda ({e})"
    owners, chunk_shapes, pinned, in_chunk = [], [], 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for o in gc.get_objects():
            # diagnostic scan over arbitrary objects: any object that is not a plain pinned CPU tensor is skipped
            try:
                if type(o) is not torch.Tensor and not (
                        isinstance(o, torch.Tensor)
                        and type(o).__module__.startswith("torch")):
                    continue
                if o.device.type != "cpu" or not o.is_pinned():
                    continue
                st = o.untyped_storage()
                lo, n = st.data_ptr(), st.nbytes()
            except Exception:  # noqa: BLE001
                continue
            pinned += 1
            if chunk is not None and lo < chunk[1] and lo + n > chunk[0]:
                in_chunk += 1
                if in_chunk <= 8:
                    chunk_shapes.append(f"{tuple(o.shape)}@+{lo - chunk[0]}")
            if lo < target + 64 and lo + n > target:
                owners.append(
                    f"{tuple(o.shape)} {o.dtype} storage {n} B at +{target - lo}"
                )
    return (
        f"staging {staging:#x} at +{staging - chunk[0] if chunk else -1} of "
        f"{(chunk[1] - chunk[0]) if chunk else 0} B chunk, target +{mapping.tp_rank * 4096}: "
        f"{'mapped' if mapped else 'UNMAPPED (would SIGSEGV; fenced instead)'}; {drv}; "
        f"{pinned} live pinned torch tensors, {in_chunk} in the chunk {chunk_shapes}, covering the target: "
        f"{owners or 'none'}")


def _q35_stray_diag_construct(workspace_size_bytes: int, mapping: Mapping,
                              use_fabric_handle: bool, comm) -> McastGPUBuffer:
    """TRTLLM_Q35_MCAST_STRAY_LOG (diagnostic): construct like the stock path and log the fence target's owners.
    The construction is collective: whatever the diagnostic does, every rank must reach it."""
    import ctypes
    import time

    def construct():
        return McastGPUBuffer(workspace_size_bytes, mapping.tp_size,
                              mapping.tp_rank, mapping.local_rank,
                              use_fabric_handle, comm.py2f())

    if not use_fabric_handle or mapping.tp_rank == 0:
        return construct()
    try:
        rt = _q35_cudart()
        size = mapping.tp_size * 64
        p = ctypes.c_void_p()
        rt.cudaMallocHost(ctypes.byref(p), size)
        staging = p.value
        rt.cudaFreeHost(
            staging)  # same-size LIFO reuse: the staging buffer lands here
        target = staging + mapping.tp_rank * 4096
        chunk = _q35_host_mapping(staging)
        tmap = _q35_host_mapping(target)
        mapped = tmap is not None and tmap[1] >= target + 64
        old = ctypes.string_at(target, 64) if mapped else None
        tag = (
            f"[Q35 comm] stray write (rank {mapping.tp_rank}, {workspace_size_bytes} B workspace, "
            f"t={time.time():.1f}): " +
            _q35_stray_target_report(mapping, staging, chunk, target, mapped))
    except Exception as e:  # noqa: BLE001  (diagnostic only; never skip the collective construction)
        logger.warning(f"[Q35 comm] stray write diagnostic failed: {e!r}")
        return construct()
    if not mapped:
        logger.warning(tag)
        return _q35_fenced_mcast_buffer(workspace_size_bytes, mapping,
                                        use_fabric_handle, comm)
    buf = construct()
    new = ctypes.string_at(target, 64)
    logger.warning(
        f"{tag}; target bytes {'CHANGED' if new != old else 'unchanged'} "
        f"(before: {old[:16].hex()}..., zero={old == bytes(64)})")
    return buf


def get_or_scale_allreduce_mnnvl_workspace(
        mapping: Mapping,
        dtype: torch.dtype,
        buffer_size_bytes: Optional[int] = None) -> Optional[Dict]:
    """
    WORKSPACE is a entire memory allocation used for allreduce, while BUFFER refers to single lamport buffer.
    Each WORKSPACE contains NUM_LAMPORT_BUFFERS buffers.

    With TRTLLM_Q35_MNNVL_PRESIZE the first call creates the workspace at its final size and later calls
    never re-create it: one that needs a larger buffer gets None (the caller falls back).
    """

    NUM_LAMPORT_BUFFERS = 3

    # Use MNNVLAllReduce class to share across threads
    allreduce_mnnvl_workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces

    if _Q35_MNNVL_PRESIZE:
        ws = allreduce_mnnvl_workspaces.get(mapping)
        if ws is not None:
            if ws["buffer_size_bytes"] < (buffer_size_bytes or 0):
                _q35_log_once(
                    ("mnnvl_sealed", mapping.tp_size, mapping.tp_rank),
                    f"[Q35 comm] MNNVL workspace is sized once (TRTLLM_Q35_MNNVL_PRESIZE): a call needing "
                    f"{buffer_size_bytes} bytes per buffer (> {ws['buffer_size_bytes']}) takes the fallback",
                    warn=True)
                return None
            return ws
        # never below the default initial size (1024 tokens x 8192 hidden)
        buffer_size_bytes = max(
            buffer_size_bytes or 0, _q35_mnnvl_presize_bytes(mapping, dtype),
            1024 * 8192 * torch.tensor([], dtype=dtype).element_size())

    # A safe method to get the element size of the dtype
    elem_size = torch.tensor([], dtype=dtype).element_size()
    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
    use_fabric_handle = force_mn or mapping.is_multi_node()

    if mapping not in allreduce_mnnvl_workspaces or allreduce_mnnvl_workspaces[
            mapping]["buffer_size_bytes"] < (buffer_size_bytes or 0):
        # Initial buffer to be large enough to support 1024 tokens * 8192 hidden_dim
        init_buffer_size_bytes = max(1024 * 8192 * elem_size, buffer_size_bytes
                                     or 0)
        # Creating the workspace if it doesn't exist
        if mapping not in allreduce_mnnvl_workspaces:
            # Do the communicator split if there is no communicator in the workspace
            comm = mpi_comm().Split(
                int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank),
                mapping.tp_rank)
            # Use the predefined buffer size if no buffer size is provided
            buffer_size_bytes = buffer_size_bytes or init_buffer_size_bytes
            if mapping.tp_rank == 0:
                logger.debug(
                    f"[MNNVL] Creating workspace for pp_rank {mapping.pp_rank}, tp_size {mapping.tp_size} with {buffer_size_bytes} bytes"
                )

        else:
            comm = allreduce_mnnvl_workspaces[mapping]["mpi_comm"]
            # Safeguard against when buffer_size_bytes is None
            req_buffer_size_bytes = buffer_size_bytes or init_buffer_size_bytes
            # Increase the buffer size in 8 MiB granularity to avoid frequently scaling the buffer
            buffer_size_bytes = math.ceil(req_buffer_size_bytes /
                                          (8 * 1024 * 1024)) * (8 * 1024 * 1024)
            logger.debug(
                f"[MNNVL] Requested {req_buffer_size_bytes} bytes, is larger than the current workspace size. Scaling workspace for pp_rank {mapping.pp_rank}, tp_size {mapping.tp_size} from {allreduce_mnnvl_workspaces[mapping]['buffer_size_bytes']} to {buffer_size_bytes} bytes"
            )
        if _Q35_AR_LOG:
            import traceback
            stack = " <- ".join(
                f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                for f in reversed(traceback.extract_stack()[-9:-1]))
            logger.info(
                f"[Q35 comm] MNNVL workspace {'scaled' if mapping in allreduce_mnnvl_workspaces else 'created'}"
                f" (rank {mapping.rank}): {buffer_size_bytes} bytes per buffer; called from {stack}"
            )
        # Each workspace contains NUM_LAMPORT_BUFFERS buffers.
        workspace_size_bytes = NUM_LAMPORT_BUFFERS * buffer_size_bytes
        # Pass the pre-split MPI communicator's Fortran handle to avoid redundant splitting in C++
        if _Q35_MNNVL_PRESIZE:
            mcast_buf_handle = _q35_fenced_mcast_buffer(workspace_size_bytes,
                                                        mapping,
                                                        use_fabric_handle, comm)
        elif _Q35_MCAST_STRAY_LOG:
            mcast_buf_handle = _q35_stray_diag_construct(
                workspace_size_bytes, mapping, use_fabric_handle, comm)
        else:
            mcast_buf_handle = McastGPUBuffer(
                workspace_size_bytes,
                mapping.tp_size,
                mapping.tp_rank,
                mapping.local_rank,
                use_fabric_handle,  # whether to use fabric handle or POSIX FD ipc
                comm.py2f(),  # Fortran handle for the MPI communicator
            )

        # We use per FP32 element in the buffer for lamport sync
        buffer = mcast_buf_handle.get_uc_buffer(mapping.tp_rank,
                                                (workspace_size_bytes //
                                                 (torch.float32.itemsize), ),
                                                torch.float32, 0)
        buffer.fill_(-0.0)
        # Wait until the initialization is done
        torch.cuda.synchronize()
        comm.Barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # The flag should be binded to each buffer allocation
        # Layout: [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count ptr]
        num_bytes_to_clear = [0] * 4
        buffer_flags = torch.tensor(
            [0, 2, buffer_size_bytes, 0, *num_bytes_to_clear, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

        allreduce_mnnvl_workspaces[mapping] = {
            "handle": mcast_buf_handle,
            "uc_buffer": buffer,
            "buffer_flags": buffer_flags,
            "buffer_size_bytes": buffer_size_bytes,
            "mpi_comm": comm,
        }
    return allreduce_mnnvl_workspaces[mapping]


def userbuffers_allreduce_finalize(
        input: torch.Tensor,
        force_applying_finalize: bool = False) -> torch.Tensor:
    output = torch.ops.trtllm.userbuffers_allreduce_finalize(
        input, force_applying_finalize)
    return output


def get_output_info(input: torch.Tensor, dim: int) -> List[int]:
    dim = dim % input.ndim
    output_shape = [
        val if idx != dim else -1 for idx, val in enumerate(input.shape)
    ]
    numel_base = -math.prod(output_shape)
    return {'output_shape': output_shape, 'numel_base': numel_base}


def filter_valid_input(
        input_list: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], List[bool]]:
    func_valid = lambda x: x is not None
    valid_list = list(map(func_valid, input_list))
    input_list = list(filter(func_valid, input_list))
    return input_list, valid_list


def restore_full_output(valid_outputs: List[torch.Tensor],
                        valid_list: List[bool]) -> List[torch.Tensor]:
    idx = 0
    full_outputs = []
    for v in valid_list:
        full_outputs.append(valid_outputs[idx] if v else None)
        idx += int(v)
    return full_outputs


def _allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    group: List[int],
    rank: int,
    group_boxed: Optional[object] = None,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Performs a collective all-gather across the given parallel group, for the given rank.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        group (List[int]): The list of ranks to participate in the all-gather.
        rank (int): The rank of the current process.
        group_boxed (object): The boxed ProcessGroup object for the list of ranks, if available.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    if len(group) == 1:
        return input

    if sizes is not None:
        assert len(sizes) == len(group)
        if isinstance(input, torch.Tensor):
            assert input.shape[dim] == sizes[rank]
        else:
            assert all([
                val.shape[dim] == sizes[rank] for val in input
                if val is not None
            ])
    # Inputs are reshaped in this way to pass necessary shape information to the allgather op
    if isinstance(input, torch.Tensor):
        if mpi_disabled():
            torch_op = torch.ops.trtllm.allgather_pg
        else:
            torch_op = torch.ops.trtllm.allgather

        output_info = get_output_info(input, dim)
        input = input.contiguous().view(-1, output_info['numel_base'])
    else:
        input, valid = filter_valid_input(input)
        if mpi_disabled():
            torch_op = torch.ops.trtllm.allgather_list_pg
        else:
            torch_op = torch.ops.trtllm.allgather_list

        output_info = [get_output_info(val, dim) for val in input]
        input = [
            val.contiguous().view(-1, val_info['numel_base'])
            for val, val_info in zip(input, output_info)
        ]

    if mpi_disabled():
        output = torch_op(input, sizes, group, group_boxed)
    else:
        output = torch_op(input, sizes, group)

    def convert_output(x, x_info):
        if dim == 0:
            x = x.view(x_info['output_shape'])
        else:
            if sizes is None:
                x_list = x.chunk(len(group))
            else:
                x_list = x.split(sizes)
            x = torch.cat([x.reshape(x_info['output_shape']) for x in x_list],
                          dim=dim)
        return x

    if isinstance(input, torch.Tensor):
        output = convert_output(output, output_info)
    else:
        output = [
            convert_output(val, val_info)
            for val, val_info in zip(output, output_info)
        ]
        output = restore_full_output(output, valid)
    return output


def allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather across the TP group.

    If 'sizes' is 'None', the input tensors in the different ranks must have the same shape.
    Otherwise, 'sizes[i]' must be 'input.shape[dim]' at rank i, and the input tensors in
    the different ranks can only differ in shape at dimension `dim`.

    The input tensors in the same TP group are concatenated at dimension 'dim' to produce the output tensor.
    If 'sizes' is 'None', 'output.shape[dim] = input.shape[dim] * tp_group_size'.
    Otherwise, 'output.shape[dim] = sum(sizes)'.

    That operation is implemented using a torch op that wraps the NCCL all-gather collective operation or
    the NCCL group call of a series of NCCL broadcast collective operations. See the following materials for details.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        mapping (Mapping): The parallel mapping.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    group_boxed = mapping.tp_group_pg.boxed() if mpi_disabled() else None
    if _Q35_AR_LOG and len(mapping.tp_group) > 1:
        site = _q35_caller_site()
        shape = tuple(input.shape) if isinstance(input,
                                                 torch.Tensor) else len(input)
        _q35_log_once(("allgather", site, shape),
                      f"[Q35 comm] allgather {site}: tp={mapping.tp_size} "
                      f"input={shape} dim={dim} -> NCCL all-gather")
    return _allgather(input, mapping.tp_group, mapping.tp_rank, group_boxed,
                      dim, sizes)


def cp_allgather(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    '''
    Add an operation that performs a collective all-gather across the CP group.

    See `allgather` for more details on the inputs and implementation constraints.

    Args:
        input (Union[Tensor, List[Tensor]]): The input tensor or tensor list.
        mapping (Mapping): The parallel mapping.
        dim (int): Gather along given dimension. By default -1.
        sizes(Optional[List[int]]): An optional list indicating 'input.shape[dim]' in all ranks. By default None.
    Returns:
        The gathered tensor or tensor list.
    '''
    group_boxed = mapping.cp_group_pg.boxed() if mpi_disabled() else None
    return _allgather(input, mapping.cp_group, mapping.cp_rank, group_boxed,
                      dim, sizes)


def alltoall_helix(
    inputs: List[torch.Tensor],
    group: List[int],
) -> List[torch.Tensor]:
    '''
    Add an operation that performs a collective all-to-all across a given group.
    The operation is implemented using a torch op that wraps a NCCL group call of a series of
    NCCL send/recv operations to implement the all-to-all. See the following materials for details.
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all,
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html.
    Args:
        inputs (List[Tensor]): The input tensors.
            Its length must be a multiple of the group size,
            and all tensors in a group must have the same shape.
            All input tensors must be contiguous.
        group (List[int]): The group of ranks to participate in the all-to-all.
    Returns:
        The output tensors.
        For each group of input tensors (of size group size),
        there is one output tensor with shape (group size, *input shape).
    '''
    n_ranks = len(group)
    if n_ranks == 1:
        return inputs

    assert n_ranks > 0, "group must be non-empty"
    assert n_ranks == len(set(group)), "group must be unique"

    assert len(inputs) % n_ranks == 0,\
        "inputs length must be a multiple of the group size"
    num_lists = len(inputs) // n_ranks
    for il in range(num_lists):
        ref_input = inputs[il * n_ranks]
        assert all([inputs[i].shape == ref_input.shape for i in range(il * n_ranks + 1, (il + 1) * n_ranks)]),\
            "all input tensors in a group must have the same shape"

    return torch.ops.trtllm.alltoall_helix(inputs, group, num_lists)


class HelixAllToAllNative:
    """
    Manager for Helix All-to-All operations with MNNVL workspace management.

    Exchanges data along the cp_size dimension:
    - partial_o: [..., cp_size, kv_lora_rank] half-precision
    - softmax_stats: [..., cp_size, 2] float32
    """

    # Global cache: mapping -> instance
    _cache: Dict[Mapping, "HelixAllToAllNative"] = {}

    def __init__(self, mapping: Mapping, workspace: HelixCpMnnvlMemory,
                 workspace_tensor: torch.Tensor):
        """Private constructor - use get() instead."""
        self.mapping = mapping
        self.workspace = workspace
        self.workspace_tensor = workspace_tensor

    @staticmethod
    def get(mapping: Mapping) -> "HelixAllToAllNative":
        """
        Get or create a HelixAllToAllNative instance for the given configuration.

        Args:
            mapping: TensorRT-LLM mapping object containing cp_size and cp_rank

        Returns:
            Cached or newly-created HelixAllToAllNative instance
        """
        if mapping not in HelixAllToAllNative._cache:
            logger.info(
                f"Rank {mapping.cp_rank} initializing HelixCpMnnvlMemory for Helix"
            )
            MnnvlMemory.initialize()

            # Get workspace size (in bytes)
            workspace_size_per_rank = _tllm_internal.thop.get_helix_workspace_size_per_rank(
                mapping.cp_size)

            # Allocate MNNVL memory using CP communicator for Helix
            workspace = HelixCpMnnvlMemory(mapping, workspace_size_per_rank)
            workspace_tensor = workspace.as_torch_strided_tensor(torch.uint64)

            torch.ops.trtllm.initialize_helix_workspace(workspace_tensor,
                                                        mapping.cp_rank,
                                                        mapping.cp_size)
            torch.cuda.synchronize()
            HelixCpMnnvlMemory.get_comm(mapping).barrier()

            HelixAllToAllNative._cache[mapping] = HelixAllToAllNative(
                mapping, workspace, workspace_tensor)

        return HelixAllToAllNative._cache[mapping]

    def alltoall_native(self, partial_o: torch.Tensor,
                        softmax_stats: torch.Tensor):
        """
        Perform all-to-all data exchange.

        Args:
            partial_o: Tensor with shape [..., cp_size, kv_lora_rank], dtype half.
            softmax_stats: Tensor with shape [..., cp_size, 2], dtype float32.

        Returns:
            Tuple of (partial_o_out, softmax_stats_out) with same shapes as inputs.
        """
        partial_o_out, softmax_stats_out = torch.ops.trtllm.alltoall_helix_native(
            partial_o,
            softmax_stats,
            self.workspace_tensor,
            self.mapping.cp_rank,
            self.mapping.cp_size,
        )

        return partial_o_out, softmax_stats_out


def reducescatter(
    input: Union[torch.Tensor, List[torch.Tensor]],
    mapping: Mapping,
    dim: int = -1,
    sizes: Optional[List[int]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if mapping.tp_size == 1:
        return input

    if sizes is not None:
        assert len(sizes) == len(mapping.tp_group)
        sum_split_size = sum(sizes)
        if isinstance(input, torch.Tensor):
            assert input.shape[dim] == sum_split_size
        else:
            assert all([
                val.shape[dim] == sum_split_size for val in input
                if val is not None
            ])

    def convert_input(x, x_info):
        # Inputs are reshaped in this way to pass necessary shape information to the reducescatter op
        if dim == 0:
            x = x.contiguous().view(-1, x_info['numel_base'])
        else:
            if sizes is None:
                x_list = x.chunk(mapping.tp_size, dim=dim)
            else:
                x_list = x.split(sizes, dim=dim)
            x = torch.cat([x.reshape(-1, x_info['numel_base']) for x in x_list])
        return x

    if isinstance(input, torch.Tensor):
        if mpi_disabled():
            torch_op = torch.ops.trtllm.reducescatter_pg
        else:
            torch_op = torch.ops.trtllm.reducescatter
        output_info = get_output_info(input, dim)
        input = convert_input(input, output_info)
    else:
        input, valid = filter_valid_input(input)
        if mpi_disabled():
            torch_op = torch.ops.trtllm.reducescatter_list_pg
        else:
            torch_op = torch.ops.trtllm.reducescatter_list
        output_info = [get_output_info(val, dim) for val in input]
        input = [
            convert_input(val, val_info)
            for val, val_info in zip(input, output_info)
        ]

    if mpi_disabled():
        output = torch_op(input, sizes, mapping.tp_group,
                          mapping.tp_group_pg.boxed())
    else:
        output = torch_op(input, sizes, mapping.tp_group)

    if isinstance(input, torch.Tensor):
        output = output.view(output_info['output_shape'])
    else:
        output = [
            val.view(val_info['output_shape'])
            for val, val_info in zip(output, output_info)
        ]
        output = restore_full_output(output, valid)
    return output


class MNNVLAllReduce(nn.Module):
    """A specialized AllReduce implementation for Multi-Node NVLink communication.

    This class handles the MNNVL-specific allreduce operations, which can be more efficient
    for certain operations when using NVLink for multi-node communication. The oneshot kernel
    is deterministic across ranks. For TP sizes up to 8, it uses a rank-specialized
    fast path that keeps the local value in registers and volatile-loads only peers
    from the Lamport buffer. Larger world sizes use a compact deterministic fallback.
    """
    allreduce_mnnvl_workspaces: Dict[Mapping, Dict] = {}

    SUPPORTED_FUSION_OPS: frozenset[AllReduceFusionOp] = frozenset({
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
    })

    def __init__(self, mapping: Mapping, dtype: torch.dtype):
        super().__init__()
        self.mapping = mapping
        self.dtype = dtype
        if dtype not in MNNVLAllReduce.get_supported_dtypes() or (
                mapping.has_cp()):
            # This is safe as we always capture the exception when create this object
            raise ValueError(
                f"MNNVL all reduce only supports dtype {MNNVLAllReduce.get_supported_dtypes()} and without cp."
            )

        # Initialize the workspace
        get_or_scale_allreduce_mnnvl_workspace(self.mapping, self.dtype)

    @staticmethod
    def get_supported_dtypes():
        return (torch.float16, torch.bfloat16, torch.float32)

    # Check if MNNVL is supported
    @staticmethod
    def is_mnnvl(mapping: Mapping, dtype: torch.dtype) -> bool:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        # Add a bypass so that we can run the unittest on single-node
        is_testing = os.environ.get("TLLM_TEST_MNNVL", "0") == "1"
        return is_testing or (dtype in MNNVLAllReduce.get_supported_dtypes() and
                              not mapping.has_cp() and mapping.is_multi_node()
                              and MnnvlMemory.supports_mnnvl()
                              and is_on_aarch64)

    @staticmethod
    def get_required_workspace_size(num_tokens: int, hidden_dim: int,
                                    group_size: int, dtype: torch.dtype) -> int:
        elem_size = torch.tensor([], dtype=dtype).element_size()
        # This should match the heuristic in allreduceOp.cpp.
        is_one_shot = (num_tokens * hidden_dim * group_size * elem_size
                       <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES)
        if is_one_shot:
            # For one-shot, each rank needs to store num_tokens * group_size tokens
            workspace_size = num_tokens * hidden_dim * group_size * elem_size
        else:
            # For two-shot, each rank stores a slices of tokens. We need to round up to the nearest group_size.
            # 2 Stage is required for the two-shot allreduce.
            workspace_size = 2 * math.ceil(
                num_tokens / group_size) * group_size * hidden_dim * elem_size
        return workspace_size

    def forward(
        self,
        input: torch.Tensor,
        all_reduce_params: AllReduceParams,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass for MNNVL AllReduce.

        Args:
            input (torch.Tensor): Input tensor to be reduced; last dim is the hidden dimension.
            all_reduce_params (Optional[AllReduceParams]): Parameters for fused operations.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: Reduced tensor(s). Output tensors
            preserve the input's N-D shape (NVFP4 quant output has its last dim halved; the
            NVFP4 scale-factor output is 1-D).
        """

        fusion_op = all_reduce_params.fusion_op
        hidden_dim = input.shape[-1]
        num_tokens = input.numel() // hidden_dim

        residual = all_reduce_params.residual
        kernel_op = fusion_op
        if fusion_op == AllReduceFusionOp.RMS_NORM:
            # TRTLLM_Q35_AR_NORM_FUSE: RMSNorm of the reduced sum is the residual variant with a
            # zero residual (the kernel adds the residual to the reduced sum; adding zero changes
            # no value). Without the flag RMS_NORM keeps returning None (caller's fallback path).
            residual = q35_zero_residual(
                input.shape, input.dtype, input.device
            ) if _Q35_AR_NORM_FUSE and input.dtype == self.dtype else None
            if residual is None:
                return None
            kernel_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

        workspace_size_bytes = self.get_required_workspace_size(
            num_tokens, hidden_dim, self.mapping.tp_size, self.dtype)

        # We use uint32_t to store workspace size related info. Safeguard against overflow.
        if workspace_size_bytes >= 2**32 - 1:
            # Raise an error so we can fallback to other allreduce strategies
            raise ValueError(
                f"[MNNVL AllReduce] Required workspace {workspace_size_bytes} bytes exceeds uint32 limits "
                f"for shard ({num_tokens}, {hidden_dim}), TP {self.mapping.tp_size}."
            )

        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping,
            self.dtype,
            buffer_size_bytes=workspace_size_bytes,
        )
        if workspace is None:  # TRTLLM_Q35_MNNVL_PRESIZE: sized once, too small for this call
            return None

        # We don't expect the buffer to be directly used in this level. The tensor is only used for passing the pointer to the kernel
        buffer_base = workspace["uc_buffer"].view(self.dtype).view(3, -1)
        # The buffer flags is tied to the buffer and used to save the state of the buffer
        buffer_flags = workspace["buffer_flags"]

        is_fusion = kernel_op != AllReduceFusionOp.NONE
        if is_fusion and kernel_op not in MNNVLAllReduce.SUPPORTED_FUSION_OPS:
            return None

        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            input,
            all_reduce_params.norm_weight,  # gamma
            residual,  # residual
            all_reduce_params.eps,  # epsilon
            buffer_base,  # comm_buffer
            buffer_flags,  # buffer_flags
            is_fusion,  # rmsnorm_fusion
            all_reduce_params.scale,  # scale
            int(kernel_op),
        )
        if fusion_op == AllReduceFusionOp.RMS_NORM:
            # RMS_NORM returns only the normalized tensor (its residual output is the plain sum).
            return outputs[0]
        return tuple(outputs) if is_fusion else outputs[0]


class AllReduce(nn.Module):

    def __init__(self,
                 mapping: Mapping,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        """
        AllReduce is a module that performs an all-reduce operation on a tensor.

        Args:
            mapping (Mapping):  The parallel mapping config.
            strategy (AllReduceStrategy):
                The following all-reduce strategies are supported:

                - SYMM_MEM: Uses PyTorch's symmetric memory with MULTIMEM hardware instructions.
                  Falls back automatically if not supported.

                - UB: AllReduce uses user-buffer based all-reduce kernel.

                - NCCL: Use NCCL allreduce.

                - MIN_LATENCY: AllReduce uses MIN_LATENCY mode kernel.

                - AUTO: AUTO chooses the best available strategy. Will try MNNVL,
                  then choose between NCCL and MIN_LATENCY based on a heuristic policy.

                - LOWPRECISION: AllReduce quantizes data to lower precision for transmission.
                  Should only be used on topologies with PCIe switches and without NVLink.
                  This strategy may result in some precision loss but can improve performance
                  on specific hardware configurations.

            All strategies support the following operations:
                - NONE (AllReduce only)
                - RESIDUAL_RMS_NORM
                - RESIDUAL_RMS_NORM_QUANT_FP8
                - RESIDUAL_RMS_NORM_QUANT_NVFP4
                - RESIDUAL_RMS_NORM_OUT_QUANT_FP8
                - RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4

            Note: NCCL, UB, and LOWPRECISION strategies only support consequent kernel calls
        instead of fused operations.

        Note:
            For the reference implementation for each pattern, please refer to the following unit test:
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/multi_gpu/test_allreduce.py

            The LOWPRECISION strategy can be selected either by directly specifying it in the constructor.
        """

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy
        self.mnnvl_allreduce = None
        self.symm_mem_allreduce = None
        self._disable_mpi = mpi_disabled()

        self.all_reduce_op = torch.ops.trtllm.allreduce_pg if self._disable_mpi else torch.ops.trtllm.allreduce

        # Propagate model-level prealloc config to AllReduceRunner once per
        # process.  extra_attrs is only active during model __init__, so we
        # read it here and stash the values as class-level attributes that
        # AllReduceRunner.forward can use during the autotuner warm-up phase.
        extra_attrs = get_model_extra_attrs()
        if extra_attrs:
            from tensorrt_llm._torch.custom_ops.torch_custom_ops import \
                AllReduceRunner
            max_num_tokens = extra_attrs.get('allreduce_max_num_tokens')
            hidden_size = extra_attrs.get('allreduce_hidden_size')
            if max_num_tokens is not None:
                AllReduceRunner._prealloc_max_num_tokens = max_num_tokens
            if hidden_size is not None:
                AllReduceRunner._prealloc_hidden_size = hidden_size
            prealloc_dtype = extra_attrs.get('allreduce_dtype')
            if prealloc_dtype is not None:
                AllReduceRunner._prealloc_dtype = prealloc_dtype

        if self.mapping.tp_size > 1 and not self.mapping.enable_attention_dp:
            # Initialize Symmetric Memory AllReduce if needed (before workspace allocation)
            if self.strategy == AllReduceStrategy.SYMM_MEM:
                try:
                    symm_mem = SymmetricMemoryAllReduce(
                        self.mapping,
                        dtype=dtype if dtype else torch.bfloat16,
                    )
                    if not symm_mem.disabled:
                        self.symm_mem_allreduce = symm_mem
                        logger.info(
                            f"SymmetricMemoryAllReduce (MULTIMEM) is enabled with fallback support for world_size={self.mapping.tp_size}"
                        )
                        # Keep SYMM_MEM strategy but allocate workspace for fallback to regular allreduce
                    else:
                        logger.info(
                            f"SymmetricMemoryAllReduce is disabled (not supported or unavailable), falling back to AUTO strategy"
                        )
                        # Fall back to AUTO if SYMM_MEM can't be enabled
                        self.strategy = AllReduceStrategy.AUTO
                except Exception as e:
                    logger.info(
                        f"Symmetric Memory AllReduce can't be enabled due to {e}, falling back to AUTO strategy"
                    )
                    self.symm_mem_allreduce = None
                    # Fall back to AUTO if SYMM_MEM initialization fails
                    self.strategy = AllReduceStrategy.AUTO

            # Allocate workspace for strategies that need it
            # Note: SYMM_MEM now also needs workspace for fallback scenarios (fused ops, etc.)
            # Only UB doesn't need workspace
            if self.strategy != AllReduceStrategy.UB:
                if self.strategy == AllReduceStrategy.LOWPRECISION:
                    allocate_low_presicion_allreduce_workspace(self.mapping)
                if self.strategy not in (AllReduceStrategy.UB,
                                         AllReduceStrategy.NCCL,
                                         AllReduceStrategy.NCCL_SYMMETRIC):
                    self.workspace = get_allreduce_workspace(self.mapping)

            # Initialize MNNVL if using AUTO or MNNVL strategy
            if self.strategy in (AllReduceStrategy.AUTO,
                                 AllReduceStrategy.MNNVL):
                # Try to initialize MNNVL
                if MNNVLAllReduce.is_mnnvl(self.mapping, dtype):
                    # ALWAYS capture the exception when creating this instance
                    try:
                        self.mnnvl_allreduce = MNNVLAllReduce(
                            self.mapping, dtype) if dtype else None
                    except Exception as e:
                        logger.debug(
                            f"MNNVL AllReduce can't be enabled due to {e}.")
                        self.mnnvl_allreduce = None
                else:
                    logger.debug(
                        f"MNNVLAllReduce can't be enabled due to failing the is_mnnvl check."
                    )
                    self.mnnvl_allreduce = None

        self._q35_site = None
        if _Q35_AR_LOG and self.mapping.tp_size > 1:
            self._q35_site = _q35_caller_site(("linear.py", "embedding.py"))
            multi_node = self.mapping.is_multi_node()
            if self.mnnvl_allreduce is not None:
                route = "MNNVL (fabric Lamport one-/two-shot)"
            elif self.symm_mem_allreduce is not None:
                route = "SYMM_MEM for NONE, C++ for fused ops"
            elif multi_node:
                route = (
                    f"C++ {self.strategy.name} -> NCCL across nodes "
                    f"[MNNVL not used: {'no dtype' if dtype is None else 'unsupported'}]"
                )
            else:
                route = f"C++ {self.strategy.name} (IPC kernels / NCCL)"
            _q35_log_once(
                ("ar_init", self._q35_site, route),
                f"[Q35 comm] AllReduce {self._q35_site}: tp={self.mapping.tp_size}"
                f"{' across nodes' if multi_node else ''} strategy={self.strategy.name} "
                f"dtype={dtype} -> {route}",
                warn=multi_node and self.mnnvl_allreduce is None)

    def _q35_log_call(self, input: torch.Tensor,
                      params: Optional[AllReduceParams], path: str) -> None:
        op = params.fusion_op if params is not None else AllReduceFusionOp.NONE
        hidden = input.shape[-1]
        tokens = input.numel() // hidden
        if path == "MNNVL":
            nbytes = tokens * hidden * self.mapping.tp_size * input.element_size(
            )
            path = ("MNNVL one-shot"
                    if nbytes <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES else
                    "MNNVL two-shot + Lamport RMSNorm"
                    if op != AllReduceFusionOp.NONE else "MNNVL two-shot")
        _q35_log_once(
            ("ar_call", self._q35_site, int(op), tokens, path),
            f"[Q35 comm] AllReduce call {self._q35_site}: "
            f"op={AllReduceFusionOp(op).name} rows={tokens} hidden={hidden} "
            f"dtype={input.dtype} -> {path}",
            warn=path.startswith("C++") and self.mapping.is_multi_node())

    def _q35_probe_local(
        self, input: torch.Tensor, params: Optional[AllReduceParams]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """TRTLLM_Q35_PROBE_NO_AR: the all-reduce's local epilogue only (no communication)."""
        op = params.fusion_op if params is not None else AllReduceFusionOp.NONE
        if op == AllReduceFusionOp.NONE:
            return input
        from ..custom_ops import (flashinfer_fused_add_rmsnorm,
                                  flashinfer_rmsnorm)
        if op == AllReduceFusionOp.RMS_NORM:
            return flashinfer_rmsnorm(input, params.norm_weight, params.eps)
        # One in-place fused add + RMSNorm kernel, like the fused all-reduce it replaces (the
        # all-reduce input and the incoming residual are dead after the call); only the shared
        # zero residual is copied first.
        residual = params.residual
        if any(residual.data_ptr() >= z.data_ptr() and residual.data_ptr() <
               z.data_ptr() + z.numel() * z.element_size()
               for z in _q35_zero_residuals.values()):
            residual = residual.clone()
        hidden = input
        flashinfer_fused_add_rmsnorm(hidden, residual, params.norm_weight,
                                     params.eps)
        if op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
            return hidden, residual
        quant, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            hidden, params.scale)
        if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
            return quant, residual
        if op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
            return hidden, quant, residual
        raise NotImplementedError(f"probe: fusion op {op}")

    def uses_nccl_symmetric_memory_window(self) -> bool:
        """Return True if this allreduce can use an NCCL window output buffer.

        Requires TLLM_NCCL_SYMMETRIC_ZERO_COPY=1 AND NCCL_SYMMETRIC/NCCL/AUTO
        strategy AND tp_size > 1 AND MPI not disabled.
        """
        return (_NCCL_SYMMETRIC_ZERO_COPY and self.strategy
                in (AllReduceStrategy.NCCL_SYMMETRIC, AllReduceStrategy.NCCL,
                    AllReduceStrategy.AUTO) and self.mapping.tp_size > 1
                and not self._disable_mpi)

    @property
    def output_buffer_kind(self) -> int:
        """Buffer kind callers should use when allocating the tensor that will
        be passed into this allreduce.

        Returns int(BufferKind.NCCL_WINDOW) when zero-copy window output is
        active, int(BufferKind.DEFAULT) otherwise.  The value depends solely on
        compile-time constants so it is safe to branch on inside torch.compile.
        """
        return (int(BufferKind.NCCL_WINDOW)
                if self.uses_nccl_symmetric_memory_window() else int(
                    BufferKind.DEFAULT))

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        '''
        The input tensors in the different ranks must have the same shape.
        The output tensor will have that same shape with the input tensor.
        The output tensor will be replicated among the TP group.
        Note that it is not an in-place operation like torch.distributed.all_reduce.

        That operation is implemented using a torch op that wraps the NCCL all-reduce
        collective operation and custom one-shot/two-shot allreduce kernels. See
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
        for details.

        Args:
            input (Tensor): The input tensor.
            all_reduce_params (AllReduceParams): The parameters for the fused ops into the allreduce op.
        Returns:
            A tensor lists with different tensor outptus according to the fusion_op.
            NONE: [hidden_states]
            RMS_NORM: [hidden_states]
            RESIDUAL_RMS_NORM: [hidden_states, residual]
            RESIDUAL_RMS_NORM_QUANT_FP8: [norm_quant, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_FP8: [norm, norm_quant, residual]
            RESIDUAL_RMS_NORM_QUANT_NVFP4: [norm_quant_fp4, scale_factor, residual]
            RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: [norm, norm_quant_fp4, scale_factor, residual]
        '''
        if self.mapping.tp_size == 1 or (all_reduce_params is not None
                                         and all_reduce_params.enable_allreduce
                                         == False):
            return input

        input = input.contiguous()  # Underlying op requires contiguous input

        if _Q35_PROBE_NO_AR:
            return self._q35_probe_local(input, all_reduce_params)

        allreduce_strategy = self.strategy

        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        # Try Symmetric Memory AllReduce first if available
        # Note: Currently only supports NONE fusion op (plain allreduce)
        if self.symm_mem_allreduce and all_reduce_params.fusion_op == AllReduceFusionOp.NONE:
            symm_mem_output = self.symm_mem_allreduce(input)
            if symm_mem_output is not None:
                logger.debug(
                    f"Using SymmetricMemoryAllReduce (MULTIMEM) for input shape {input.shape}"
                )
                return symm_mem_output
        elif self.symm_mem_allreduce and all_reduce_params.fusion_op != AllReduceFusionOp.NONE:
            # Log once per rank that we're skipping symm_mem due to fusion
            logger.debug_once(
                f"Skipping SymmetricMemoryAllReduce for fused operation (fusion_op={all_reduce_params.fusion_op}), using regular allreduce",
                key=(self.mapping.tp_rank, all_reduce_params.fusion_op,
                     "debug_fusion_skip"),
            )

        # Try MNNVL AllReduce if symm_mem didn't handle it
        if self.mnnvl_allreduce:
            mnnvl_output = self.mnnvl_allreduce(
                input, all_reduce_params=all_reduce_params)
            if mnnvl_output is not None:
                if self._q35_site is not None:
                    self._q35_log_call(input, all_reduce_params, "MNNVL")
                return mnnvl_output

        if self._q35_site is not None:
            self._q35_log_call(
                input, all_reduce_params,
                f"C++ {'tunable_allreduce' if allreduce_strategy == AllReduceStrategy.AUTO else 'allreduce'}"
                f" strategy={allreduce_strategy.name}"
                f"{' (NCCL across nodes)' if self.mapping.is_multi_node() else ''}"
            )

        # Fall back to regular AllReduce if specialized methods are not available or not applicable
        # Make sure the strategy is AUTO since allreduceOp does not have the branch for MNNVL/SYMM_MEM
        if allreduce_strategy in (AllReduceStrategy.MNNVL,
                                  AllReduceStrategy.SYMM_MEM):
            allreduce_strategy = AllReduceStrategy.AUTO
        if (allreduce_strategy == AllReduceStrategy.NCCL_SYMMETRIC
                and os.environ.get("TRTLLM_Q35_AR_NO_NCCL_SYMM", "0") == "1"):
            # TRTLLM_Q35_AR_NO_NCCL_SYMM (see custom_ops/torch_custom_ops.py): plain NCCL instead.
            allreduce_strategy = AllReduceStrategy.NCCL

        additional_args = {}
        if self._disable_mpi:
            # Get ProcessGroup from mapping
            pg = self.mapping.tp_group_pg
            assert pg is not None, "TP ProcessGroup not initialised"
            additional_args = {
                "rank": torch.distributed.get_rank(),
                "pg": pg.boxed(),
            }

        # In case that AutoTuner brings potential perf regression
        # TODO: Remove this if no perf regression is observed.
        disable_allreduce_autotune = os.environ.get(
            "TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0") == "1"

        if allreduce_strategy == AllReduceStrategy.AUTO and not disable_allreduce_autotune and not self._disable_mpi:
            output = torch.ops.trtllm.tunable_allreduce(
                input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                scale=all_reduce_params.scale,
                bias=all_reduce_params.bias,
                workspace=self.workspace,
                group=self.mapping.tp_group,
                strategy=allreduce_strategy,
                op=all_reduce_params.fusion_op,
                eps=all_reduce_params.eps,
                trigger_completion_at_end=all_reduce_params.
                trigger_completion_at_end,
            )
        else:
            output = self.all_reduce_op(
                input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                scale=all_reduce_params.scale,
                bias=all_reduce_params.bias,
                workspace=self.workspace,
                group=self.mapping.tp_group,
                strategy=allreduce_strategy,
                op=all_reduce_params.fusion_op,
                eps=all_reduce_params.eps,
                trigger_completion_at_end=all_reduce_params.
                trigger_completion_at_end,
                **additional_args,
            )

        return output if len(output) > 1 else output[0]


class MoEAllReduce(nn.Module):

    def __init__(self, mapping: Mapping):
        """
        MoEAllReduce is a module that performs a specific fused MoE reduction
        followed by a regular AR + RMS norm.

        Args:
            mapping (Mapping):  The parallel mapping config.

        Notes:
            * min latency mode:

            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = torch.sum(active_experts_token_input *
                                        scale.unsqueeze(-1),
                                        dim=0)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)

            * regular mode:

            Support pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
            expert_reduction = local_reduction(input, expanded_idx_to_permuted_idx, expert_scale_factor)
            output_add = expert_reduction + shared_expert_output
            output_residual = output_add + residual
            output_hidden_states = rms_norm(output_residual, norm_weight, eps)
        """
        super().__init__()
        self.mapping = mapping
        self.workspace = get_allreduce_workspace(self.mapping)
        # Pls keep this value in sync with the kOneShotMaxToken in moeAllReduceFusionKernels.h
        self.max_token = 128

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: MoEAllReduceParams,
    ) -> torch.Tensor:

        assert all_reduce_params.is_valid(), "MoEAllReduceParams is not valid"

        if all_reduce_params.is_cutlass_min_latency:
            """
            Args:
            residual: residual tensor
            norm_weight: RMS norm weight
            device_num_experts: number of experts per device
            scale_input: experts to token score
            active_experts_token_input: per token per expert input
            token_input: per token input, shared expert output
            eps: epsilon for RMSNorm

            Output:
                hidden_states: hidden_states of the model
                residual: residual tensor
            """

            return torch.ops.trtllm.moe_allreduce(
                active_experts_token_input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                device_num_experts=all_reduce_params.device_num_experts,
                scale_input=all_reduce_params.expert_scale_factor,
                token_input=all_reduce_params.shared_expert_output,
                workspace=self.workspace,
                rank=self.mapping.tp_rank,
                nranks=self.mapping.tp_size,
                eps=all_reduce_params.eps,
            )
        else:
            if all_reduce_params.residual is not None:
                assert all_reduce_params.residual.shape[
                    0] <= self.max_token, "Num tokens must be less than or equal to max_token"

            return torch.ops.trtllm.moe_finalize_allreduce(
                input=input,
                residual=all_reduce_params.residual,
                norm_weight=all_reduce_params.norm_weight,
                expanded_idx_to_permuted_idx=all_reduce_params.
                expanded_idx_to_permuted_idx,
                shared_expert_output=all_reduce_params.shared_expert_output,
                expert_scale_factor=all_reduce_params.expert_scale_factor,
                workspace=self.workspace,
                rank=self.mapping.tp_rank,
                nranks=self.mapping.tp_size,
                eps=all_reduce_params.eps,
            )


def all_to_all_4d(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    All-to-all for 4D tensors (batch, seq, heads, head_dim).

    Redistributes a 4D tensor along two dimensions using all-to-all communication.
    This is used for Ulysses-style sequence parallelism to transform between:
    - Sequence sharding [B, S/P, H, D] → Head sharding [B, S, H/P, D]
    - Head sharding [B, S, H/P, D] → Sequence sharding [B, S/P, H, D]

    Args:
        input: Input tensor with shape [batch, seq, heads, head_dim]
        scatter_dim: Dimension to split and scatter (1 for seq, 2 for heads)
        gather_dim: Dimension to gather (1 for seq, 2 for heads)
        process_group: PyTorch distributed process group. If None, uses default process group.

    Returns:
        Redistributed tensor with same shape as input

    Example:
        # Transform from sequence sharding to head sharding
        # Input: [B, S/P, H, D] (each rank has S/P sequence)
        output = all_to_all_4d(input, scatter_dim=2, gather_dim=1, process_group=pg)
        # Output: [B, S, H/P, D] (each rank has H/P heads)

        # Transform back from head sharding to sequence sharding
        output = all_to_all_4d(input, scatter_dim=1, gather_dim=2, process_group=pg)
    """
    # Only support PyTorch distributed mode (not MPI mode)
    if not mpi_disabled():
        raise NotImplementedError(
            "all_to_all_4d currently only supports PyTorch distributed mode. "
            "MPI mode is not supported.")

    # Get world size from process group
    world_size = torch.distributed.get_world_size(group=process_group)

    # If world_size is 1, no communication needed
    if world_size == 1:
        return input

    # Validate dimensions
    assert scatter_dim in [1, 2], "scatter_dim must be 1 (seq) or 2 (heads)"
    assert gather_dim in [1, 2], "gather_dim must be 1 (seq) or 2 (heads)"
    assert scatter_dim != gather_dim, "scatter_dim and gather_dim must be different"

    batch, seq, heads, head_dim = input.shape

    # Validate that the scatter dimension is divisible by world_size
    scatter_size = input.shape[scatter_dim]
    assert scatter_size % world_size == 0, \
        f"Dimension {scatter_dim} size {scatter_size} must be divisible by world_size {world_size}"

    # For all-to-all, we need to:
    # 1. Split input along scatter_dim into world_size chunks
    # 2. Send chunk i to rank i
    # 3. Receive chunk from each rank and concatenate along gather_dim

    # Reshape for all-to-all: move scatter_dim chunks to a new dimension
    if scatter_dim == 1:  # Scatter along seq dimension
        # [B, S, H, D] -> [B, P, S/P, H, D] where P = world_size
        input_reshaped = input.view(batch, world_size, seq // world_size, heads,
                                    head_dim)
        # Transpose to group by destination rank: [B, P, S/P, H, D] -> [P, B, S/P, H, D]
        input_transposed = input_reshaped.permute(1, 0, 2, 3, 4).contiguous()
    else:  # scatter_dim == 2, scatter along heads dimension
        # [B, S, H, D] -> [B, S, P, H/P, D] where P = world_size
        input_reshaped = input.view(batch, seq, world_size, heads // world_size,
                                    head_dim)
        # Transpose to group by destination rank: [B, S, P, H/P, D] -> [P, B, S, H/P, D]
        input_transposed = input_reshaped.permute(2, 0, 1, 3, 4).contiguous()

    # Flatten to [P * ...] for all-to-all communication
    # Shape: [P, B, ...] -> [P * B * ...]
    input_flat = input_transposed.flatten()
    output_flat = torch.empty_like(input_flat)

    # Perform all-to-all communication using PyTorch distributed
    # all_to_all_single splits input into world_size chunks and exchanges them
    torch.distributed.all_to_all_single(output_flat,
                                        input_flat,
                                        group=process_group)

    # Reshape output back to [P, B, ...] form
    output_transposed = output_flat.view_as(input_transposed)

    # Transpose back and reshape to final form
    if gather_dim == 1:  # Gather along seq dimension
        # [P, B, S/P, H, D] -> [B, P, S/P, H, D]
        output_reshaped = output_transposed.permute(1, 0, 2, 3, 4).contiguous()
        # [B, P, S/P, H, D] -> [B, S, H, D] where S = P * (S/P)
        # When scattering heads and gathering seq: seq needs to be multiplied, heads needs to be divided
        if scatter_dim == 2:
            # Scattered heads, so we have H/P heads and need to gather S/P -> S sequence
            gathered_seq = seq * world_size
            sharded_heads = heads // world_size
            output = output_reshaped.view(batch, gathered_seq, sharded_heads,
                                          head_dim)
        else:
            # Scattered seq (should be impossible if gather_dim == 1), keep as is
            output = output_reshaped.view(batch, seq, heads, head_dim)
    else:  # gather_dim == 2, gather along heads dimension
        # [P, B, S, H/P, D] -> [B, S, P, H/P, D]
        output_reshaped = output_transposed.permute(1, 2, 0, 3, 4).contiguous()
        # [B, S, P, H/P, D] -> [B, S, H, D] where H = P * (H/P)
        # When scattering seq and gathering heads: heads needs to be multiplied, seq needs to be divided
        if scatter_dim == 1:
            # Scattered seq, so we have S/P seq and need to gather H/P -> H heads
            gathered_heads = heads * world_size
            sharded_seq = seq // world_size
            output = output_reshaped.view(batch, sharded_seq, gathered_heads,
                                          head_dim)
        else:
            # Scattered heads (should be impossible if gather_dim == 2), keep as is
            output = output_reshaped.view(batch, seq, heads, head_dim)

    return output


def all_to_all_5d(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    All-to-all for 5D tensors with a fused QKV dimension.

    Operates on [B, S, 3, H, D] tensors where dim 2 is the QKV count.
    Used for Ulysses sequence parallelism with fused QKV to reduce the
    number of all-to-all collectives from 3 (one per Q/K/V) to 1.

    Supported scatter/gather combinations:
    - scatter_dim=3 (heads), gather_dim=1 (seq): [B, S/P, 3, H, D] -> [B, S, 3, H/P, D]
    - scatter_dim=1 (seq), gather_dim=3 (heads): [B, S, 3, H/P, D] -> [B, S/P, 3, H, D]
    """
    if not mpi_disabled():
        raise NotImplementedError(
            "all_to_all_5d currently only supports PyTorch distributed mode.")

    world_size = torch.distributed.get_world_size(group=process_group)
    if world_size == 1:
        return input

    assert input.dim() == 5, f"Expected 5D tensor, got {input.dim()}D"
    assert scatter_dim in [1, 3] and gather_dim in [1, 3]
    assert scatter_dim != gather_dim

    batch, seq, qkv_count, heads, head_dim = input.shape
    assert input.shape[scatter_dim] % world_size == 0, \
        f"Dim {scatter_dim} size {input.shape[scatter_dim]} not divisible by world_size {world_size}"

    if scatter_dim == 3 and gather_dim == 1:
        # [B, S/P, 3, H, D] -> [B, S, 3, H/P, D]
        sharded_heads = heads // world_size
        inp = input.reshape(batch, seq, qkv_count, world_size, sharded_heads,
                            head_dim)
        inp = inp.permute(3, 0, 1, 2, 4,
                          5).contiguous()  # [P, B, S/P, 3, H/P, D]

        out_flat = torch.empty_like(inp.flatten())
        torch.distributed.all_to_all_single(out_flat,
                                            inp.flatten(),
                                            group=process_group)
        out = out_flat.view_as(inp)

        out = out.permute(1, 0, 2, 3, 4,
                          5).contiguous()  # [B, P, S/P, 3, H/P, D]
        gathered_seq = seq * world_size
        return out.reshape(batch, gathered_seq, qkv_count, sharded_heads,
                           head_dim)

    else:  # scatter_dim == 1, gather_dim == 3
        # [B, S, 3, H/P, D] -> [B, S/P, 3, H, D]
        sharded_seq = seq // world_size
        inp = input.reshape(batch, world_size, sharded_seq, qkv_count, heads,
                            head_dim)
        inp = inp.permute(1, 0, 2, 3, 4,
                          5).contiguous()  # [P, B, S/P, 3, H/P, D]

        out_flat = torch.empty_like(inp.flatten())
        torch.distributed.all_to_all_single(out_flat,
                                            inp.flatten(),
                                            group=process_group)
        out = out_flat.view_as(inp)

        out = out.permute(1, 2, 3, 0, 4,
                          5).contiguous()  # [B, S/P, 3, P, H/P, D]
        gathered_heads = heads * world_size
        return out.reshape(batch, sharded_seq, qkv_count, gathered_heads,
                           head_dim)


class MiniMaxAllReduceRMS(nn.Module):

    def __init__(self, mapping: Mapping):
        super().__init__()
        self.mapping = mapping
        self.workspace = get_allreduce_workspace(self.mapping)

    def forward(self, input: torch.Tensor, rms_weights: torch.Tensor,
                eps: float):
        return torch.ops.trtllm.minimax_allreduce_rms(input, rms_weights,
                                                      self.workspace,
                                                      self.mapping.tp_rank,
                                                      self.mapping.tp_size, eps,
                                                      True)

    def forward_qk(self, q: torch.Tensor, k: torch.Tensor,
                   rms_weights_q: torch.Tensor, rms_weights_k: torch.Tensor,
                   eps: float):
        """Fused Q+K RMS norm with allreduce. Returns (q_out, k_out)."""
        out_list = torch.ops.trtllm.minimax_allreduce_rms_qk(
            q,
            k,
            rms_weights_q,
            rms_weights_k,
            self.workspace,
            self.mapping.tp_rank,
            self.mapping.tp_size,
            eps,
            True,
        )
        return (out_list[0], out_list[1])
