# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-expert use of MegaMoE's complete fused FC1/activation/FC2 kernel."""

from itertools import product
from threading import Condition, RLock

import torch

from ..autotuner import AutoTuner, DynamicTensorSpec, TunableRunner, TuningConfig

__all__ = [
    "release_shared_fc12_cache",
    "run_shared_fc12",
    "shared_fc12_cache_info",
    "shared_fc12_candidate_tactics",
    "shared_fc12_tactic_descriptor",
]

_SHARED_FC12_OP = "trtllm::megamoe_shared_fc12"
_CATALOG_VERSION = 1
_SharedFc12Tactic = tuple[int, int, int, str, bool]
_CANDIDATES = tuple(
    product((1, 2), (64, 128, 256), (128, 256), ("grid_stride", "atomic_counter"), (False, True))
)


def shared_fc12_candidate_tactics() -> tuple[_SharedFc12Tactic, ...]:
    """Finite common search space; larger clusters/K and extra TMA stages are not searched."""
    return _CANDIDATES


def shared_fc12_tactic_descriptor(tactic: _SharedFc12Tactic) -> dict:
    tactic = tuple(tactic)
    if tactic not in _CANDIDATES:
        raise ValueError(f"Unsupported shared FC12 tactic: {tactic}")
    ctas, tile_n, tile_k, work_id_mode, fc2_use_bulk = tactic
    return {
        "tactic": list(tactic),
        "mma_tiler_mnk": [128 * ctas, tile_n, tile_k],
        "cluster_shape_mn": [ctas, 1],
        "use_2cta_instrs": ctas == 2,
        "work_id_mode": work_id_mode,
        "fc2_use_bulk": fc2_use_bulk,
        "fc2_tma_stages": 1,
        "fc1_epi_flag_batch": 1,
        "fc2_epi_flag_batch": 1,
    }


def _token_buckets(max_tokens: int) -> tuple[int, ...]:
    if max_tokens <= 0:
        raise ValueError("shared FC12 capacity must be positive")
    return tuple(
        sorted(
            {n for n in (1, 128, 256, 512, 1024, 2048, 4096, 8192) if n < max_tokens} | {max_tokens}
        )
    )


def _make_kernel(
    device_index, hidden_size, intermediate_size, max_tokens, sm_count, swiglu_limit, tactic
):
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import OperandMajorMode

    from .cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from .cutedsl_megamoe.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_fc12_kernel import (
        BlockScaledSwapAbFc12Kernel,
    )
    from .cutedsl_megamoe.quant_def import CombineFormat

    properties = torch.cuda.get_device_properties(device_index)
    if not 0 < sm_count <= properties.multi_processor_count:
        raise ValueError("shared FC12 sm_count must be within the device SM count")
    descriptor = shared_fc12_tactic_descriptor(tactic)
    cluster_size = descriptor["cluster_shape_mn"][0]
    hardware_clusters = utils.HardwareInfo(device_index).get_max_active_clusters(cluster_size)
    launch_clusters = min(sm_count // cluster_size, hardware_clusters)
    if launch_clusters <= 0:
        raise ValueError("shared FC12 requires at least one active cluster")
    problem = ProblemDesc(
        {
            "expert_count": 1,
            "intermediate_gateup_size": 2 * intermediate_size,
            "hidden_size": hidden_size,
            "quant_kind": "mxfp8_e4m3",
            "a_major_mode": OperandMajorMode.K,
            "b_major_mode": OperandMajorMode.K,
            "combine_format": CombineFormat.parse("bf16"),
            "gate_up_clamp": swiglu_limit,
        }
    )
    implementation = ImplDesc(
        {
            **{key: value for key, value in descriptor.items() if key != "tactic"},
            "mma_tiler_mnk": tuple(descriptor["mma_tiler_mnk"]),
            "cluster_shape_mn": tuple(descriptor["cluster_shape_mn"]),
            "group_hint": launch_clusters,
            "token_padding_block": 64,
            "sf_padding_block": 128,
            "max_tokens": max_tokens,
            "launch_cluster_count": launch_clusters,
        }
    )
    kernel = BlockScaledSwapAbFc12Kernel(problem, implementation)
    return kernel, implementation, properties, hardware_clusters, launch_clusters


class _SharedFc12Runner:
    def __init__(
        self,
        device_index: int,
        hidden_size: int,
        intermediate_size: int,
        max_tokens: int,
        sm_count: int,
        swiglu_limit: float | None,
        stream_handle: int,
        *,
        tactic: _SharedFc12Tactic | None = None,
    ) -> None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace

        device = torch.device("cuda", device_index)
        properties = torch.cuda.get_device_properties(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create the shared FC12 runner before CUDA graph capture")
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.reserved_sms = properties.multi_processor_count - sm_count
        self.stream_handle = stream_handle
        if tactic is None:
            tactic = (
                (1, 256, 128, "atomic_counter", True)
                if self.reserved_sms > 0
                else (2, 128, 128, "grid_stride", True)
            )
        self.tactic = tuple(tactic)
        kernel, implementation, properties, hardware_clusters, launch_clusters = _make_kernel(
            device_index,
            hidden_size,
            intermediate_size,
            max_tokens,
            sm_count,
            swiglu_limit,
            self.tactic,
        )
        local_bytes, shared_bytes = kernel.get_workspace_sizes()
        if shared_bytes:
            raise RuntimeError("shared FC12 must not allocate a communication workspace")
        self.workspace = torch.empty(local_bytes, dtype=torch.uint8, device=device)
        local_zero_bytes, shared_zero_bytes = kernel.require_zero_workspace_leading_bytes
        if shared_zero_bytes:
            raise RuntimeError("shared FC12 must not initialize communication state")
        self.workspace_pointer = self.workspace.data_ptr()
        self.workspace_reset_bytes = local_zero_bytes
        self._memset_async = cuda.cuMemsetD8Async
        self._cuda_success = cuda.CUresult.CUDA_SUCCESS
        self._cuda_stream = cuda.CUstream(stream_handle)
        # A slice supplies the current E=1 token count without a per-call GPU write.
        self.token_counts = torch.arange(max_tokens + 1, dtype=torch.int32, device=device)

        token_rows = cute.sym_int64(divisibility=1)
        scale_rows = cute.sym_int64(divisibility=128)
        padded_gateup = (2 * intermediate_size + 127) // 128 * 128
        fc1_sf_columns = padded_gateup * (hidden_size // 32)
        fc2_sf_columns = hidden_size * (intermediate_size // 32)
        fake_arguments = {
            "activation": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (token_rows, hidden_size),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            "fc1_weight": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (1, hidden_size, 2 * intermediate_size),
                stride_order=(2, 0, 1),
                assumed_align=16,
            ),
            "activation_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU,
                (scale_rows, hidden_size // 32),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            "fc1_weight_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU, (1, fc1_sf_columns), stride_order=(1, 0), assumed_align=16
            ),
            "fc2_weight": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (1, intermediate_size, hidden_size),
                stride_order=(2, 0, 1),
                assumed_align=16,
            ),
            "fc2_weight_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU, (1, fc2_sf_columns), stride_order=(1, 0), assumed_align=16
            ),
            "fc2_output": make_fake_compact_tensor(
                cutlass.BFloat16,
                (token_rows, 1, hidden_size),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            "local_workspace": make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            "max_active_clusters": hardware_clusters,
            "stream": make_fake_stream(),
            "expert_token_sizes": make_fake_compact_tensor(
                cutlass.Int32, (1,), stride_order=(0,), assumed_align=4
            ),
            "expert_token_prefix_sum": None,
            "topk_scores": None,
            "fc1_alpha": None,
            "fc2_alpha": None,
            "fc1_norm_const": None,
        }
        self.compiled = cute.compile[cute.EnableTVMFFI(True)](kernel, **fake_arguments)

    def __call__(
        self,
        activation: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
    ) -> torch.Tensor:
        token_rows = activation.shape[0]
        quantized, scales = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(
            activation, reserved_sms=self.reserved_sms
        )
        output = torch.empty(
            (token_rows, 1, self.hidden_size), dtype=torch.bfloat16, device=activation.device
        )
        # The standalone FC12 readiness counters require reset before every invocation.
        (status,) = self._memset_async(
            self.workspace_pointer, 0, self.workspace_reset_bytes, self._cuda_stream
        )
        if status != self._cuda_success:
            raise RuntimeError(f"shared FC12 asynchronous counter reset failed: {status}")
        self.compiled(
            activation=quantized,
            fc1_weight=fc1_weight.transpose(1, 2),
            activation_sf=scales.view(torch.float8_e8m0fnu).view(-1, self.hidden_size // 32),
            fc1_weight_sf=fc1_weight_sf.view(torch.float8_e8m0fnu),
            fc2_weight=fc2_weight.transpose(1, 2),
            fc2_weight_sf=fc2_weight_sf.view(torch.float8_e8m0fnu),
            fc2_output=output,
            local_workspace=self.workspace_pointer,
            stream=self.stream_handle,
            expert_token_sizes=self.token_counts[token_rows : token_rows + 1],
            expert_token_prefix_sum=None,
            topk_scores=None,
            fc1_alpha=None,
            fc2_alpha=None,
            fc1_norm_const=None,
        )
        return output[:, 0, :]


_SHARED_FC12_CACHE_LOCK = RLock()
_SHARED_FC12_CACHE_CONDITION = Condition(_SHARED_FC12_CACHE_LOCK)
_ACTIVE_FC12_CALLS: dict[tuple[int, int], int] = {}
_RELEASING_FC12_DEVICES: set[int] = set()
_RELEASING_FC12_STREAMS: set[tuple[int, int]] = set()
_RUNNER_CACHE: dict[tuple, _SharedFc12Runner] = {}
_RUNNER_CACHE_HITS = 0
_RUNNER_CACHE_MISSES = 0


def _begin_shared_fc12_release(
    device_index: int, stream_handle: int | None
) -> tuple[int, int] | None:
    stream_key = (device_index, stream_handle) if stream_handle is not None else None
    with _SHARED_FC12_CACHE_CONDITION:
        while (
            device_index in _RELEASING_FC12_DEVICES
            or (
                stream_key is None
                and any(device == device_index for device, _ in _RELEASING_FC12_STREAMS)
            )
            or (stream_key is not None and stream_key in _RELEASING_FC12_STREAMS)
        ):
            _SHARED_FC12_CACHE_CONDITION.wait()
        if stream_key is None:
            _RELEASING_FC12_DEVICES.add(device_index)
        else:
            _RELEASING_FC12_STREAMS.add(stream_key)
        while any(
            count and device == device_index and (stream_handle is None or stream == stream_handle)
            for (device, stream), count in _ACTIVE_FC12_CALLS.items()
        ):
            _SHARED_FC12_CACHE_CONDITION.wait()
    return stream_key


def _end_shared_fc12_release(device_index: int, stream_key: tuple[int, int] | None) -> None:
    with _SHARED_FC12_CACHE_CONDITION:
        if stream_key is None:
            _RELEASING_FC12_DEVICES.discard(device_index)
        else:
            _RELEASING_FC12_STREAMS.discard(stream_key)
        _SHARED_FC12_CACHE_CONDITION.notify_all()


def _begin_shared_fc12_call(device_index: int, stream_handle: int) -> tuple[int, int]:
    key = (int(device_index), int(stream_handle))
    with _SHARED_FC12_CACHE_CONDITION:
        while key[0] in _RELEASING_FC12_DEVICES or key in _RELEASING_FC12_STREAMS:
            _SHARED_FC12_CACHE_CONDITION.wait()
        _ACTIVE_FC12_CALLS[key] = _ACTIVE_FC12_CALLS.get(key, 0) + 1
    return key


def _end_shared_fc12_call(key: tuple[int, int]) -> None:
    with _SHARED_FC12_CACHE_CONDITION:
        remaining = _ACTIVE_FC12_CALLS[key] - 1
        if remaining:
            _ACTIVE_FC12_CALLS[key] = remaining
        else:
            del _ACTIVE_FC12_CALLS[key]
            _SHARED_FC12_CACHE_CONDITION.notify_all()


def _get_runner(
    device_index: int,
    hidden_size: int,
    intermediate_size: int,
    max_tokens: int,
    sm_count: int,
    swiglu_limit: float | None,
    stream_handle: int,
    tactic: _SharedFc12Tactic | None = None,
) -> _SharedFc12Runner:
    global _RUNNER_CACHE_HITS, _RUNNER_CACHE_MISSES
    key = (
        device_index,
        hidden_size,
        intermediate_size,
        max_tokens,
        sm_count,
        swiglu_limit,
        stream_handle,
        tactic,
    )
    with _SHARED_FC12_CACHE_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is not None:
            _RUNNER_CACHE_HITS += 1
            return runner
        _RUNNER_CACHE_MISSES += 1
        with torch.cuda.device(device_index):
            runner = _SharedFc12Runner(
                device_index,
                hidden_size,
                intermediate_size,
                max_tokens,
                sm_count,
                swiglu_limit,
                stream_handle,
                tactic=tactic,
            )
        _RUNNER_CACHE[key] = runner
        return runner


def shared_fc12_cache_info() -> dict[str, int | None]:
    """Return process-cache counters for diagnostics."""
    with _SHARED_FC12_CACHE_LOCK:
        return {
            "hits": _RUNNER_CACHE_HITS,
            "misses": _RUNNER_CACHE_MISSES,
            "maxsize": None,
            "currsize": len(_RUNNER_CACHE),
        }


class SharedFc12TunableRunner(TunableRunner):
    # AutoTuner uses this dictionary to prime persisted winners during model warmup.
    kernel_cache: dict[tuple, _SharedFc12Runner] = {}

    def __init__(
        self,
        device_index,
        hidden_size,
        intermediate_size,
        max_tokens,
        sm_count,
        swiglu_limit,
        stream_handle,
        input_signature,
    ):
        properties = torch.cuda.get_device_properties(device_index)
        self.device_index = device_index
        self.sm_count = sm_count
        self.reserved_sms = properties.multi_processor_count - sm_count
        self.max_tokens = max_tokens
        self.stream_handle = stream_handle
        self._runner_args = (
            device_index,
            hidden_size,
            intermediate_size,
            max_tokens,
            sm_count,
            swiglu_limit,
            stream_handle,
        )
        self._unique_id = (
            _CATALOG_VERSION,
            device_index,
            properties.major,
            properties.minor,
            properties.multi_processor_count,
            sm_count,
            hidden_size,
            intermediate_size,
            max_tokens,
            swiglu_limit,
            "K128_input_K32_intermediate_R128c4_gateup16",
            input_signature,
        )
        self.buckets = _token_buckets(max_tokens)
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=0,
                    dim_idx=0,
                    gen_tuning_buckets=self.buckets,
                    map_to_tuning_buckets=self.token_bucket,
                ),
            ),
            tune_max_num_tokens=max_tokens,
            use_cold_l2_cache=True,
            use_cuda_graph=False,
        )
        self._valid_tactics = None
        self._compiled_runners = {}
        self._compile_lock = RLock()

    def unique_id(self):
        return self._unique_id

    def token_bucket(self, tokens: int) -> int:
        if not 0 < tokens <= self.max_tokens:
            raise ValueError("shared FC12 token count exceeds its configured capacity")
        return next(bucket for bucket in self.buckets if tokens <= bucket)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        if self._valid_tactics is None:
            valid = []
            with torch.cuda.device(self.device_index):
                for tactic in shared_fc12_candidate_tactics():
                    try:
                        _make_kernel(*self._runner_args[:6], tactic)
                    except (ValueError, AssertionError):
                        continue
                    else:
                        valid.append(tactic)
            self._valid_tactics = tuple(valid)
        return list(self._valid_tactics)

    def forward(self, inputs, *, tactic=-1, do_preparation=False, **kwargs):
        if do_preparation:
            return None
        if tactic == -1:
            if AutoTuner.get().is_tuning_mode:
                raise RuntimeError("shared FC12 profiling did not select a valid tactic")
            tactic = (
                (1, 256, 128, "atomic_counter", True)
                if self.reserved_sms > 0
                else (2, 128, 128, "grid_stride", True)
            )
        tactic = tuple(tactic)
        with self._compile_lock:
            runner = self._compiled_runners.get(tactic)
            if runner is None:
                runner = _get_runner(*self._runner_args, tactic)
                self._compiled_runners[tactic] = runner
                self.kernel_cache[(*self._runner_args, tactic)] = runner
        return runner(*inputs)


_tunable_runners: dict[tuple, SharedFc12TunableRunner] = {}


def release_shared_fc12_cache(device_index: int, stream_handle: int | None = None) -> None:
    """Drain and release shared-FC12 runners for one executor stream.

    Dispatches already in progress finish before teardown. New dispatches for
    the same stream wait, while unrelated streams and devices remain usable.
    Passing no stream handle retains the process-exit fallback that releases
    every runner on the device.
    """
    device_index = int(device_index)
    stream_handle = None if stream_handle is None else int(stream_handle)
    stream_key = _begin_shared_fc12_release(device_index, stream_handle)

    def matches(key: tuple) -> bool:
        return int(key[0]) == device_index and (
            stream_handle is None or int(key[6]) == stream_handle
        )

    try:
        with torch.cuda.device(device_index):
            if stream_handle is None:
                torch.cuda.synchronize(device_index)
            else:
                torch.cuda.ExternalStream(stream_handle, device=device_index).synchronize()
        with _SHARED_FC12_CACHE_CONDITION:
            tunable_keys = [key for key in _tunable_runners if matches(key)]
            for key in tunable_keys:
                _tunable_runners.pop(key)._compiled_runners.clear()
            kernel_keys = [key for key in SharedFc12TunableRunner.kernel_cache if matches(key)]
            for key in kernel_keys:
                SharedFc12TunableRunner.kernel_cache.pop(key, None)
            runner_keys = [key for key in _RUNNER_CACHE if matches(key)]
            for key in runner_keys:
                _RUNNER_CACHE.pop(key, None)
    finally:
        _end_shared_fc12_release(device_index, stream_key)


def run_shared_fc12(
    activation: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    *,
    swiglu_limit: float | None,
    sm_count: int,
    max_tokens: int = 8192,
) -> torch.Tensor:
    """Run one complete FC12 kernel after the existing shared-expert input quantizer.

    ``activation`` is contiguous BF16 ``[T, H]``. FP8 weights have shapes
    ``[1, 2*I, H]`` (gate/up interleaved in groups of 16) and ``[1, H, I]``.
    Their uint8 scales are flattened R128c4 UE8M0 ``[1, padded_rows * K/32]``.
    ``H`` and ``I`` must be divisible by 128; output is BF16 ``[T, H]``.
    ``max_tokens`` is a fixed model capacity, not the current batch size.
    """
    if activation.dtype != torch.bfloat16 or not activation.is_cuda:
        raise ValueError("shared FC12 requires CUDA BF16 activations")
    if activation.ndim != 2 or not activation.is_contiguous():
        raise ValueError("shared FC12 requires contiguous [T, H] activations")
    if max_tokens <= 0 or activation.shape[0] > max_tokens:
        raise ValueError("shared FC12 token count exceeds its configured capacity")
    hidden_size = activation.shape[1]
    if fc2_weight.ndim != 3 or fc2_weight.shape[:2] != (1, hidden_size):
        raise ValueError("shared FC12 weight must have shape [1, H, I]")
    intermediate_size = fc2_weight.shape[2]
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("shared FC12 hidden and intermediate dimensions must be multiples of 128")
    if fc1_weight.shape != (1, 2 * intermediate_size, hidden_size):
        raise ValueError("shared FC1 weight must have shape [1, 2*I, H]")
    for weight in (fc1_weight, fc2_weight):
        if weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous():
            raise ValueError("shared FC12 weights must be contiguous E4M3 tensors")
    for scale in (fc1_weight_sf, fc2_weight_sf):
        if scale.dtype != torch.uint8 or not scale.is_contiguous():
            raise ValueError("shared FC12 scales must be contiguous uint8 tensors")
    if any(
        t.device != activation.device
        for t in (fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf)
    ):
        raise ValueError("shared FC12 inputs and weights must be on the same CUDA device")
    if activation.shape[0] == 0:
        return torch.empty_like(activation)
    stream = torch.cuda.current_stream(activation.device)
    inputs = [activation, fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf]
    input_signature = tuple(
        (str(t.dtype), (None, hidden_size) if index == 0 else tuple(t.shape), tuple(t.stride()))
        for index, t in enumerate(inputs)
    )
    runner_key = (
        activation.device.index,
        hidden_size,
        intermediate_size,
        max_tokens,
        sm_count,
        swiglu_limit,
        stream.cuda_stream,
        input_signature,
    )
    active_key = _begin_shared_fc12_call(activation.device.index, stream.cuda_stream)
    try:
        with _SHARED_FC12_CACHE_LOCK, torch.cuda.device(activation.device):
            runner = _tunable_runners.get(runner_key)
            if runner is None:
                runner = _tunable_runners[runner_key] = SharedFc12TunableRunner(*runner_key)
        tuner = AutoTuner.get()
        selected_runner, tactic = tuner.choose_one(
            _SHARED_FC12_OP, [runner], runner.tuning_config, inputs
        )
        return selected_runner(inputs, tactic=tactic)
    finally:
        _end_shared_fc12_call(active_key)
