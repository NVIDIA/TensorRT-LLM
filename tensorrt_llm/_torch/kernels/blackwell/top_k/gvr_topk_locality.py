# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared row-local GVR on dynamically queried CUDA locality resources."""

import operator
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cuda

from . import gvr_topk_decode_self_sampling_host as gvr
from .gvr_topk_locality_policy import choose_locality


def _check(result: tuple, operation: str) -> object:
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation}: {result[0]}")
    if len(result) == 2:
        return result[1]
    return result[1:]


@dataclass
class _Domain:
    ordinal: int
    sm_count: int
    context: cuda.CUgreenCtx
    driver_stream: cuda.CUstream
    stream: torch.cuda.ExternalStream


class LocalityDomainExecutor:
    """Own dynamically queried domain green contexts; prepare outside capture."""

    def __init__(
        self,
        device: int = 0,
        coscheduled_sm_count: int = 8,
        sm_counts: tuple[int, ...] | None = None,
    ) -> None:
        self.device = device
        properties = torch.cuda.get_device_properties(device)
        self.sm_version = properties.major * 10 + properties.minor
        self.physical_sm_count = properties.multi_processor_count
        self.coscheduled_sm_count = coscheduled_sm_count
        self.domains: list[_Domain] = []
        self.closed = False
        with torch.cuda.device(device):
            torch.cuda.init()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Create locality resources before CUDA graph capture")
            torch.cuda.current_stream()
            count = int(
                _check(
                    cuda.cuDeviceGetAttribute(
                        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device
                    ),
                    "locality domain count",
                )
            )
            if count < 1:
                raise RuntimeError("The device does not expose CUDA locality domains")
            if sm_counts is not None:
                sm_counts = tuple(operator.index(c) for c in sm_counts)
                if len(sm_counts) != count or any(c <= 0 for c in sm_counts):
                    raise ValueError("sm_counts must provide a positive budget for every domain")
            resource = _check(
                cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
                "device SM resource",
            )
            params = []
            for ordinal in range(count):
                param = cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
                param.smCount = 0 if sm_counts is None else sm_counts[ordinal]
                param.coscheduledSmCount = coscheduled_sm_count
                param.flags = int(
                    cuda.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
                )
                param.localityDomainId = ordinal
                params.append(param)
            resources, remainder = _check(
                cuda.cuDevSmResourceSplit(count, resource, 0, params), "split SM resources"
            )
            self.remainder_sms = int(remainder.sm.smCount)
            try:
                for ordinal, partition in enumerate(resources):
                    self._add_domain(ordinal, partition)
            except RuntimeError:
                self.close()
                raise

    def _add_domain(self, ordinal: int, partition: cuda.CUdevResource) -> None:
        desc = _check(cuda.cuDevResourceGenerateDesc([partition], 1), "resource descriptor")
        context = _check(
            cuda.cuGreenCtxCreate(
                desc, self.device, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            ),
            "green context",
        )
        try:
            driver_stream = _check(
                cuda.cuGreenCtxStreamCreate(context, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                "green stream",
            )
        except RuntimeError:
            cuda.cuGreenCtxDestroy(context)
            raise
        try:
            stream = torch.cuda.ExternalStream(int(driver_stream), device=self.device)
            actual = _check(
                cuda.cuStreamGetDevResource(
                    driver_stream, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
                ),
                "stream SM resource",
            )
            if int(actual.sm.localityDomainId) != ordinal:
                raise RuntimeError("Green stream does not belong to the requested locality domain")
            if int(actual.sm.smCount) != int(partition.sm.smCount):
                raise RuntimeError("Green stream SM count differs from its partition")
        except RuntimeError:
            cuda.cuStreamDestroy(driver_stream)
            cuda.cuGreenCtxDestroy(context)
            raise
        self.domains.append(
            _Domain(ordinal, int(actual.sm.smCount), context, driver_stream, stream)
        )

    def close(self) -> None:
        """Synchronize owned streams and release their green contexts."""
        if self.closed:
            return
        for domain in self.domains:
            _check(cuda.cuStreamSynchronize(domain.driver_stream), "synchronize green stream")
        for domain in reversed(self.domains):
            _check(cuda.cuStreamDestroy(domain.driver_stream), "destroy green stream")
            _check(cuda.cuGreenCtxDestroy(domain.context), "destroy green context")
        self.domains.clear()
        self.closed = True

    def __enter__(self) -> "LocalityDomainExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _Bound:
    def __init__(self, x, lengths, output, n, cr, rows, sms, sm, *, use_rubin_tuning=True):
        self.workspace = torch.zeros(gvr.workspace_bytes(), device=x.device, dtype=torch.uint8)
        self.logits, self.lengths, self.output = x, lengths, output
        compile_guard = gvr._COMPILE_LOCK_16BIT if x.dtype != torch.float32 else nullcontext()
        with compile_guard, torch.cuda.device(x.device):
            self.launcher = gvr._varlen_launcher(
                rows,
                x.shape[1],
                output.shape[1],
                n,
                1,
                cr,
                sms,
                sm,
                dtype=x.dtype,
                device_index=x.get_device(),
                use_rubin_tuning=use_rubin_tuning,
            )
        self.ws = gvr.kernel_view(self.workspace)

    def __call__(self):
        lc = self.launcher
        x, lengths, out = self.logits, self.lengths, self.output
        if lc[0] in ("reg_clus", "complement"):
            lc[1](x, out, lengths, out, lc[2])
        elif lc[0] in ("reg", "clus"):
            lc[1](x, out, lengths, out, *lc[2])
        else:
            lc[1](x, out, out, self.ws, *lc[2], lengths, *lc[3])


class PreparedRowTopK:
    """One non-reentrant call, including graph-safe fork/join when locality wins.

    Prepare and warm before graph capture. Keep this object, its input allocation,
    and its resource executor alive until all captured graphs are destroyed.
    Values and per-row lengths may change; addresses and row ownership may not.
    ``co_local=True`` requires each declared row range to be in its named domain.
    The caller owns placement; this API does not copy or reallocate logits.
    ``max_seq_len`` and ``kv_lens`` use uncompressed token counts, matching
    ``run_varlen``. This row API uses ``next_n=1``; MTP batches use ``run_varlen``.
    """

    def __init__(
        self,
        logits,
        kv_lens,
        indices,
        *,
        max_seq_len,
        compress_ratio=1,
        row_counts=None,
        co_local=False,
        executor=None,
    ):
        self.executor = None
        self.owns_executor = False
        self.closed = False
        success = False
        try:
            self._initialize(
                logits,
                kv_lens,
                indices,
                max_seq_len=max_seq_len,
                compress_ratio=compress_ratio,
                row_counts=row_counts,
                co_local=co_local,
                executor=executor,
            )
            success = True
        finally:
            if not success:
                if self.owns_executor and self.executor is not None:
                    self.executor.close()
                self.closed = True

    def _initialize(
        self,
        logits,
        kv_lens,
        indices,
        *,
        max_seq_len,
        compress_ratio,
        row_counts,
        co_local,
        executor,
    ):
        tensors = (logits, kv_lens, indices)
        if any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
            raise ValueError("All inputs must be CUDA tensors")
        self.device = logits.get_device()
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare GVR before CUDA graph capture")
        if any(t.get_device() != self.device for t in tensors):
            raise ValueError("All inputs must be on the same device")
        if logits.ndim != 2 or indices.ndim != 2 or logits.shape[0] < 1:
            raise ValueError("Logits and output must be nonempty row matrices")
        rows, width = logits.shape
        k = indices.shape[1]
        if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Logits must be FP32, FP16 or BF16")
        if indices.dtype != torch.int32 or kv_lens.dtype != torch.int32:
            raise ValueError("Output and lengths must be int32")
        if kv_lens.shape != (rows,) or indices.shape != (rows, k):
            raise ValueError("Lengths and output must match logits rows")
        if not kv_lens.is_contiguous() or not indices.is_contiguous() or logits.stride(1) != 1:
            raise ValueError("Lengths/output must be contiguous, and logits columns contiguous")
        max_seq_len = operator.index(max_seq_len)
        compress_ratio = operator.index(compress_ratio)
        if compress_ratio not in (1, 4) or max_seq_len < 0:
            raise ValueError("Invalid uncompressed envelope or compression ratio")
        max_seq_len = max(max_seq_len // compress_ratio, 1)
        if k not in (512, 1024, 2048) or width < 1:
            raise ValueError("Invalid K, compressed envelope, or compression ratio")
        stride = logits.stride(0) if rows > 1 else width
        if stride < width or stride % (16 // logits.element_size()) or logits.data_ptr() % 16:
            raise ValueError("Logits require a nonoverlapping, vector-aligned row layout")
        need = logits.storage_offset() + rows * stride
        if need * logits.element_size() > logits.untyped_storage().nbytes():
            raise ValueError("Backing storage must cover the padded row layout")
        max_seq_len = min(max_seq_len, stride)
        x = logits.as_strided((rows, stride), (stride, 1))
        props = torch.cuda.get_device_properties(self.device)
        sm = props.major * 10 + props.minor
        sms = props.multi_processor_count
        self.mode = "tuned"
        self.executor = None
        self.owns_executor = False
        self.closed = False
        self.parts = []
        self.plans = []
        counts = None if row_counts is None else tuple(operator.index(c) for c in row_counts)
        if counts is not None and (any(c < 0 for c in counts) or sum(counts) != rows):
            raise ValueError("row_counts must cover logits exactly")
        choice = choose_locality(logits.dtype, k, max_seq_len, rows, sms, sm) if co_local else None
        if choice is not None and choice[0] == "global_original":
            self.mode = "global_original"
            choice = None
        if choice is not None:
            capability = cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, self.device
            )
            if capability[0] != cuda.CUresult.CUDA_SUCCESS or int(capability[1]) < 1:
                choice = None
        if choice is not None and counts is not None:
            mode, budget = choice
            if executor is None:
                executor = LocalityDomainExecutor(self.device, budget)
                self.owns_executor = True
                self.executor = executor
            if executor.closed or executor.device != self.device:
                raise ValueError("Executor is closed or on another device")
            if len(counts) != len(executor.domains):
                if self.owns_executor:
                    executor.close()
                raise ValueError("row_counts needs one entry for every queried domain")
            self.executor = executor
            if executor.coscheduled_sm_count == budget:
                # Query every allocation-granularity interval, including the end.
                # Ordinary allocations without a definite domain fall back globally.
                prop = cuda.CUmemAllocationProp()
                prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
                prop.location.type = (
                    cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
                )
                prop.location.localized.deviceId = self.device
                prop.location.localized.localityDomainId = 0
                unit = int(
                    _check(
                        cuda.cuMemGetAllocationGranularity(
                            prop,
                            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                        ),
                        "locality granularity",
                    )
                )
                valid = True
                offset = 0
                for domain, count in zip(executor.domains, counts):
                    if count:
                        begin = x.data_ptr() + offset * stride * x.element_size()
                        end = begin + count * stride * x.element_size()
                        addresses = [
                            begin,
                            end - 1,
                            *range(((begin // unit) + 1) * unit, end, unit),
                        ]
                        for ptr in addresses:
                            result = cuda.cuPointerGetAttribute(
                                cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                ptr,
                            )
                            if (
                                result[0] != cuda.CUresult.CUDA_SUCCESS
                                or int(result[1]) != domain.ordinal
                            ):
                                valid = False
                                break
                    offset += count
                # Unbalanced ownership can exhaust one domain while leaving others idle.
                # Use the measured policy only when the assigned rows follow SM resources.
                total = sum(d.sm_count for d in executor.domains)
                balanced = all(
                    abs(c - rows * d.sm_count / total) <= 1
                    for c, d in zip(counts, executor.domains)
                )
                if valid and balanced:
                    offset = 0
                    prepared = []
                    for domain, count in zip(executor.domains, counts):
                        if count:
                            plan_rows = rows if mode == "local_preserve" else count
                            plan_sms = sms if mode == "local_preserve" else domain.sm_count
                            plan = gvr._route_fp32_varlen(
                                plan_rows, max(max_seq_len, k + 1), stride, k, plan_sms, sm
                            )
                            if plan["cluster"] > executor.coscheduled_sm_count:
                                prepared = []
                                break
                            bound = _Bound(
                                x[offset : offset + count],
                                kv_lens[offset : offset + count],
                                indices[offset : offset + count],
                                max_seq_len,
                                compress_ratio,
                                plan_rows,
                                plan_sms,
                                sm,
                            )
                            prepared.append((domain, bound, torch.cuda.Event()))
                            self.plans.append(plan)
                        offset += count
                    if prepared:
                        self.mode = mode
                        self.parts = prepared
        if not self.parts:
            if self.owns_executor and self.executor is not None:
                self.executor.close()
            self.executor = None
            self.owns_executor = False
            self.parts = [
                (
                    None,
                    _Bound(
                        x,
                        kv_lens,
                        indices,
                        max_seq_len,
                        compress_ratio,
                        rows,
                        sms,
                        sm,
                        use_rubin_tuning=self.mode != "global_original",
                    ),
                    None,
                )
            ]
            self.plans = []
        self.start = torch.cuda.Event()
        self.logits, self.kv_lens, self.indices = logits, kv_lens, indices
        # Finish workspace initialization on the caller's stream before capture.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        ready.synchronize()

    def __call__(self):
        with torch.cuda.device(self.device):
            if self.closed or (self.executor is not None and self.executor.closed):
                raise RuntimeError("Prepared call or its executor is closed")
            if self.mode in ("global_original", "tuned"):
                self.parts[0][1]()
                return
            parent = torch.cuda.current_stream(self.device)
            self.start.record(parent)
            for domain, bound, done in self.parts:
                domain.stream.wait_event(self.start)
                with torch.cuda.stream(domain.stream):
                    bound()
                    done.record(domain.stream)
            for _, _, done in self.parts:
                parent.wait_event(done)

    def close(self):
        """Call after destroying graphs which reference this prepared invocation."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        if self.owns_executor and self.executor is not None:
            self.executor.close()
        self.parts.clear()
        self.closed = True
