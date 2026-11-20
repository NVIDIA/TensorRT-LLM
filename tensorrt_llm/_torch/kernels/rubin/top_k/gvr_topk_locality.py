# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared row-local GVR on dynamically queried CUDA locality resources."""

import operator
from contextlib import nullcontext
from dataclasses import dataclass
from math import gcd, lcm

import torch
from cuda.bindings import driver as cuda

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling_host as gvr

from .gvr_topk_locality_policy import COSCHEDULED_SM_COUNT, choose_backfill, choose_locality


class _CudaError(RuntimeError):
    def __init__(self, status: cuda.CUresult, operation: str) -> None:
        super().__init__(f"{operation}: {status}")
        self.status = status
        self.operation = operation


def _check(result: tuple, operation: str) -> object:
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise _CudaError(result[0], operation)
    if len(result) == 2:
        return result[1]
    return result[1:]


def _backfill_counts(
    counts: tuple[int, ...], alignments: tuple[int, ...], remainder: int
) -> tuple[int, ...]:
    """Grow aligned budgets without changing the domain ownership proportions."""
    if not counts or min(counts) <= 0 or remainder <= 0:
        return counts
    divisor = gcd(*counts)
    ratios = tuple(c // divisor for c in counts)
    quantum = lcm(*(a // gcd(a, r) for a, r in zip(alignments, ratios)))
    factor = remainder // sum(ratios) // quantum * quantum
    return tuple(c + factor * r for c, r in zip(counts, ratios))


@dataclass
class _Domain:
    ordinal: int
    sm_count: int
    context: cuda.CUgreenCtx | None
    driver_stream: cuda.CUstream | None = None
    stream: torch.cuda.ExternalStream | None = None


class LocalityDomainExecutor:
    """Own dynamically queried domain green contexts; prepare outside capture.

    Grouping bounds cluster scheduling, not total domain SMs. BACKFILL preserves
    budget proportions but does not guarantee physical locality of added SMs.
    """

    def __init__(
        self,
        device: int = 0,
        coscheduled_sm_count: int = COSCHEDULED_SM_COUNT,
        sm_counts: tuple[int, ...] | None = None,
        *,
        backfill: bool = False,
    ) -> None:
        self.device = device
        properties = torch.cuda.get_device_properties(device)
        self.sm_version = properties.major * 10 + properties.minor
        self.physical_sm_count = properties.multi_processor_count
        self.coscheduled_sm_count = coscheduled_sm_count
        self.domains: list[_Domain] = []
        self.closed = False
        self.backfill_requested = backfill
        self.backfill_applied = False
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
            if backfill and sm_counts is None:
                resources, remainder = self._try_backfill(resources, remainder, params)
            self.remainder_sms = int(remainder.sm.smCount)
            try:
                for ordinal, partition in enumerate(resources):
                    self._add_domain(ordinal, partition)
            except BaseException as error:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise

    def _try_backfill(
        self,
        resources: list[cuda.CUdevResource],
        remainder: cuda.CUdevResource,
        params: list[cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS],
    ) -> tuple[list[cuda.CUdevResource], cuda.CUdevResource]:
        flag = getattr(cuda.CUdevSmResourceGroup_flags, "CU_DEV_SM_RESOURCE_GROUP_BACKFILL", None)
        if flag is None:
            return resources, remainder
        counts = tuple(int(r.sm.smCount) for r in resources)
        alignments = tuple(
            lcm(self.coscheduled_sm_count, max(1, int(r.sm.smCoscheduledAlignment)))
            for r in resources
        )
        requested = _backfill_counts(counts, alignments, int(remainder.sm.smCount))
        if requested == counts:
            return resources, remainder
        for param, count in zip(params, requested):
            param.smCount = count
            param.flags |= int(flag)
        root = _check(
            cuda.cuDeviceGetDevResource(
                self.device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            ),
            "backfill SM resource",
        )
        result = cuda.cuDevSmResourceSplit(len(params), root, 0, params)
        if result[0] in (
            cuda.CUresult.CUDA_ERROR_INVALID_VALUE,
            cuda.CUresult.CUDA_ERROR_NOT_SUPPORTED,
            cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,
        ):
            return resources, remainder
        filled, rest = _check(result, "backfill SM split")
        if (
            tuple(int(r.sm.smCount) for r in filled) != requested
            or any(int(r.sm.smCoscheduledAlignment) < self.coscheduled_sm_count for r in filled)
            or sum(requested) + int(rest.sm.smCount) != self.physical_sm_count
        ):
            return resources, remainder
        self.backfill_applied = True
        return filled, rest

    def _add_domain(self, ordinal: int, partition: cuda.CUdevResource) -> None:
        desc = _check(cuda.cuDevResourceGenerateDesc([partition], 1), "resource descriptor")
        context = _check(
            cuda.cuGreenCtxCreate(
                desc, self.device, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            ),
            "green context",
        )
        domain = _Domain(ordinal, int(partition.sm.smCount), context)
        self.domains.append(domain)
        domain.driver_stream = _check(
            cuda.cuGreenCtxStreamCreate(context, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
            "green stream",
        )
        domain.stream = torch.cuda.ExternalStream(int(domain.driver_stream), device=self.device)
        actual = _check(
            cuda.cuStreamGetDevResource(
                domain.driver_stream, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            ),
            "stream SM resource",
        )
        if int(actual.sm.localityDomainId) != ordinal:
            raise RuntimeError("Green stream does not belong to the requested locality domain")
        if int(actual.sm.smCount) != domain.sm_count:
            raise RuntimeError("Green stream SM count differs from its partition")

    def close(self) -> None:
        """Disable execution and release owned handles; failed releases may be retried."""
        self.closed = True
        error = None
        for domain in reversed(self.domains):
            if domain.driver_stream is not None:
                try:
                    _check(
                        cuda.cuStreamSynchronize(domain.driver_stream), "synchronize green stream"
                    )
                except RuntimeError as exc:
                    error = error or exc
                try:
                    _check(cuda.cuStreamDestroy(domain.driver_stream), "destroy green stream")
                    domain.driver_stream = None
                    domain.stream = None
                except RuntimeError as exc:
                    error = error or exc
            # Retain the context while a failed stream release is still outstanding.
            if domain.context is not None and domain.driver_stream is None:
                try:
                    _check(cuda.cuGreenCtxDestroy(domain.context), "destroy green context")
                    domain.context = None
                except RuntimeError as exc:
                    error = error or exc
        self.domains[:] = [
            d for d in self.domains if d.context is not None or d.driver_stream is not None
        ]
        if error is not None:
            raise error

    def __enter__(self) -> "LocalityDomainExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _Bound:
    def __init__(self, x, lengths, output, n, cr, rows, sms, sm):
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
            )
        self.ws = gvr.kernel_view(self.workspace)
        self.cluster_size = getattr(self.launcher, "cluster_size", None)

    def __call__(self):
        lc = self.launcher
        x, lengths, out = self.logits, self.lengths, self.output
        if lc[0] in ("reg_clus", "complement"):
            lc[1](x, out, lengths, out, lc[2])
        elif lc[0] in ("reg", "clus"):
            lc[1](x, out, lengths, out, *lc[2])
        else:
            lc[1](x, out, out, self.ws, *lc[2], lengths, *lc[3])


def _allocation_rows(x: torch.Tensor, device: int) -> tuple[int, ...] | None:
    """Verify complete padded rows before selecting any locality policy."""
    if any(
        not callable(getattr(cuda, name, None))
        for name in (
            "cuPointerGetAttribute",
            "cuDeviceGetAttribute",
            "cuMemGetAllocationGranularity",
        )
    ):
        return None
    attribute = getattr(
        cuda.CUpointer_attribute, "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL", None
    )
    domain_attribute = getattr(
        cuda.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None
    )
    location_type = getattr(
        cuda.CUmemLocationType, "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN", None
    )
    if attribute is None or domain_attribute is None or location_type is None:
        return None
    first = cuda.cuPointerGetAttribute(attribute, x.data_ptr())
    if first[0] != cuda.CUresult.CUDA_SUCCESS or int(first[1]) < 0:
        return None
    capability = cuda.cuDeviceGetAttribute(domain_attribute, device)
    if capability[0] != cuda.CUresult.CUDA_SUCCESS or int(capability[1]) < 1:
        return None
    domains = int(capability[1])
    if int(first[1]) >= domains:
        return None
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = location_type
    prop.location.localized.deviceId = device
    unit = 0
    for ordinal in range(domains):
        prop.location.localized.localityDomainId = ordinal
        granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if granularity[0] != cuda.CUresult.CUDA_SUCCESS or int(granularity[1]) < 1:
            return None
        unit = gcd(unit, int(granularity[1]))
    counts = [0] * domains
    previous = 0
    row_bytes = x.stride(0) * x.element_size()
    for row in range(x.shape[0]):
        begin = x.data_ptr() + row * row_bytes
        end = begin + row_bytes
        result = cuda.cuPointerGetAttribute(attribute, begin)
        if result[0] != cuda.CUresult.CUDA_SUCCESS:
            return None
        domain = int(result[1])
        if domain < previous or domain >= domains:
            return None
        for address in (end - 1, *range(((begin // unit) + 1) * unit, end, unit)):
            result = cuda.cuPointerGetAttribute(attribute, address)
            if result[0] != cuda.CUresult.CUDA_SUCCESS or int(result[1]) != domain:
                return None
        counts[domain] += 1
        previous = domain
    return tuple(counts)


class PreparedRowTopK:
    """Non-reentrant, graph-capturable top-k over caller-owned row allocations.

    Verify complete padded-row ownership before applying the full-batch plan to
    domain slices; ordinary or ineligible inputs retain tuned global execution.
    Preparation never copies logits. Separate concurrent calls need separate instances.
    The upstream producer must allocate its logits output with complete padded rows
    in matching locality domains; ordinary producer outputs use tuned global execution.
    Prepare and warm before capture; retain allocations and executor until graphs
    are destroyed. Values and lengths may change, but addresses and ownership may not.
    ``co_local=None`` or ``True`` probes ownership and falls back when ineligible;
    ``False`` skips the probe. ``mode`` reports the selected execution path.
    This opt-in API is not used by the model TopK module.
    Lengths are uncompressed token counts. Only ``next_n=1`` is supported.
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
        co_local=None,
        executor=None,
    ):
        self.executor = None
        self.owns_executor = False
        self.closed = False
        self.parts = []
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
        except BaseException as error:
            self.closed = True
            if self.owns_executor and self.executor is not None:
                try:
                    self.executor.close()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise

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
        if indices.data_ptr() % 16:
            raise ValueError("Output base must be 16-byte aligned")
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
        verified = None
        if co_local is not False and sm >= 107:
            verified = _allocation_rows(x, self.device)
        if verified is not None and counts is not None and counts != verified:
            verified = None
        counts = verified
        self.locality_allocation = verified is not None
        choice = (
            choose_locality(logits.dtype, k, max_seq_len, rows, sms, sm)
            if self.locality_allocation
            else None
        )
        if choice is not None:
            capability = cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, self.device
            )
            if capability[0] != cuda.CUresult.CUDA_SUCCESS or int(capability[1]) < 1:
                choice = None
        if choice is not None and counts is not None:
            mode, budget = choice
            if executor is None:
                try:
                    executor = LocalityDomainExecutor(
                        self.device,
                        budget,
                        backfill=choose_backfill(logits.dtype, k, max_seq_len, rows, sms, sm),
                    )
                except _CudaError as error:
                    if not (
                        error.operation == "split SM resources"
                        and error.status == cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION
                    ):
                        raise
                else:
                    self.owns_executor = True
                    self.executor = executor
        if choice is not None and counts is not None and executor is not None:
            mode, budget = choice
            if executor.closed or executor.device != self.device:
                raise ValueError("Executor is closed or on another device")
            if len(counts) != len(executor.domains):
                raise ValueError("row_counts needs one entry for every queried domain")
            self.executor = executor
            if executor.coscheduled_sm_count == budget:
                # Row ownership must match the domain resource proportions.
                total = sum(d.sm_count for d in executor.domains)
                balanced = all(
                    abs(c - rows * d.sm_count / total) <= 1
                    for c, d in zip(counts, executor.domains)
                )
                if balanced:
                    offset = 0
                    prepared = []
                    for domain, count in zip(executor.domains, counts):
                        if count:
                            plan = None
                            if logits.dtype == torch.float32:
                                plan = gvr._route_fp32_varlen(
                                    rows, max(max_seq_len, k + 1), stride, k, sms, sm
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
                                rows,
                                sms,
                                sm,
                            )
                            if plan is None:
                                # Check the final cluster size after native dispatch overrides.
                                if bound.cluster_size is None:
                                    raise RuntimeError("Native launcher has no cluster metadata")
                                plan = {"kernel": bound.launcher[0], "cluster": bound.cluster_size}
                                if plan["cluster"] > executor.coscheduled_sm_count:
                                    prepared = []
                                    break
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
            if self.mode == "tuned":
                self.parts[0][1]()
                return
            parent = torch.cuda.current_stream(self.device)
            capturing = torch.cuda.is_current_stream_capturing()
            submitted = []
            try:
                self.start.record(parent)
                for domain, bound, done in self.parts:
                    submitted.append((domain, done))
                    domain.stream.wait_event(self.start)
                    with torch.cuda.stream(domain.stream):
                        bound()
                        done.record(domain.stream)
                for _, _, done in self.parts:
                    parent.wait_event(done)
            except BaseException:
                self.closed = True
                if not capturing:
                    for domain, done in submitted:
                        try:
                            done.record(domain.stream)
                            parent.wait_event(done)
                        except BaseException:
                            try:
                                domain.stream.synchronize()
                            except BaseException:
                                pass  # Preserve the submission error after best-effort draining.
                raise

    def close(self):
        """Call after destroying graphs which reference this prepared invocation."""
        if self.closed and not self.parts and not self.owns_executor:
            return
        self.closed = True
        error = None
        if self.parts:
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as exc:
                error = exc
        if self.owns_executor and self.executor is not None:
            try:
                self.executor.close()
            except RuntimeError as exc:
                error = error or exc
            if not self.executor.domains:
                self.owns_executor = False
        self.parts.clear()
        if error is not None:
            raise error
