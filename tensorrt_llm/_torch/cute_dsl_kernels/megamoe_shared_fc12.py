# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-expert use of MegaMoE's complete fused FC1/activation/FC2 kernel."""

from copy import deepcopy
from functools import lru_cache
from itertools import product

import torch

from ..autotuner import AutoTuner, DynamicTensorSpec, TunableRunner, TuningConfig

__all__ = [
    "run_shared_fc12",
    "shared_fc12_cache_info",
    "shared_fc12_candidate_tactics",
    "shared_fc12_tactic_descriptor",
    "shared_fc12_autotune_state",
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
        cluster_size = implementation["cluster_shape_mn"][0]
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
        self.audit = {
            "kernel_class": type(kernel).__name__,
            "tactic": list(self.tactic),
            "fc2_use_bulk": implementation["fc2_use_bulk"],
            "fc2_tma_stages": implementation["fc2_tma_stages"],
            "upstream_commit": "522fbb019942ad6bbe0f418dba02541f837ea46c",
            "kernel_descriptor_architecture": kernel.architecture,
            "device_compute_capability": [properties.major, properties.minor],
            "input_quantizer": "fp8_quantize_1x128_packed_ue8m0.sm_budget",
            "input_quantization_block": 128,
            "fc1_output_quantization_block": 32,
            "requested_sm_count": sm_count,
            "input_quantizer_reserved_sms": self.reserved_sms,
            "launch_cluster_count": launch_clusters,
            "work_id_mode": implementation["work_id_mode"],
            "launch_grid": [cluster_size, 1, launch_clusters],
            "mma_tiler_mnk": list(implementation["mma_tiler_mnk"]),
            "cluster_shape_mn": list(implementation["cluster_shape_mn"]),
            "use_2cta_instrs": implementation["use_2cta_instrs"],
            "max_tokens": max_tokens,
            "dynamic_token_dimension": True,
            "routing_or_communication": False,
            "reset_strategy": "cuMemsetD8Async_before_fc12_same_stream",
            "workspace_reset_bytes": self.workspace_reset_bytes,
        }

    def __call__(
        self,
        activation: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
    ) -> torch.Tensor:
        token_rows = activation.shape[0]
        with torch.cuda.nvtx.range("MEGAMOE_SHARED_INPUT_QUANT"):
            quantized, scales = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(
                activation, reserved_sms=self.reserved_sms
            )
        output = torch.empty(
            (token_rows, 1, self.hidden_size), dtype=torch.bfloat16, device=activation.device
        )
        with torch.cuda.nvtx.range("MEGAMOE_SHARED_FC12"):
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


@lru_cache(maxsize=None)
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
    with torch.cuda.device(device_index):
        return _SharedFc12Runner(
            device_index,
            hidden_size,
            intermediate_size,
            max_tokens,
            sm_count,
            swiglu_limit,
            stream_handle,
            tactic=tactic,
        )


def shared_fc12_cache_info() -> dict[str, int | None]:
    """Compiled runners are keyed by capacity/tactic, not the current token count."""
    return _get_runner.cache_info()._asdict()


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
        self._rejected_tactics = {}
        self._compiled_runners = {}
        self._cache_keys = {}
        self._selections = {}

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
                    except (ValueError, AssertionError) as error:
                        self._rejected_tactics[tactic] = {
                            "tactic": list(tactic),
                            "stage": "descriptor_resources",
                            "reason": str(error),
                        }
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
        runner = self._compiled_runners.get(tactic)
        if runner is None:
            try:
                runner = _get_runner(*self._runner_args, tactic)
            except Exception as error:
                # Preserve compilation failures for audit; AutoTuner retains its exception policy.
                self._rejected_tactics[tactic] = {
                    "tactic": list(tactic),
                    "stage": "compile",
                    "reason": str(error),
                }
                raise
            self._compiled_runners[tactic] = runner
            self.kernel_cache[(*self._runner_args, tactic)] = runner
        return runner(*inputs)

    def record_selection(self, tuner, inputs, tactic):
        bucket = self.token_bucket(inputs[0].shape[0])
        key = self._cache_keys.get(bucket)
        if key is None:
            key = tuner.profiling_cache.get_cache_key(
                _SHARED_FC12_OP, self, tuple(t.shape for t in inputs), self.tuning_config
            )
            self._cache_keys[bucket] = key
        entry = tuner.profiling_cache.cache.get(key)
        fallback = tactic == -1
        selected = -1 if fallback else tuple(tactic)
        cache_hit = (
            entry is not None and not fallback and entry[1] != -1 and tuple(entry[1]) == selected
        )
        active_capture = tuner._active_capture is not None
        replay = active_capture and tuner._active_capture.is_replaying()
        signature = (bucket, selected, tuner.is_tuning_mode, active_capture, replay, cache_hit)
        row = self._selections.get(signature)
        tokens = inputs[0].shape[0]
        if row is None:
            row = self._selections[signature] = {
                "bucket": bucket,
                "tactic": list(selected) if not fallback else -1,
                "production_cache_key": repr(key),
                "cache_hit": cache_hit,
                "min_time_ms": float(entry[2]) if cache_hit else None,
                "is_tuning_mode": tuner.is_tuning_mode,
                "active_capture": active_capture,
                "replay": replay,
                "fallback": fallback,
                "calls": 0,
                "actual_token_min": tokens,
                "actual_token_max": tokens,
            }
        row["calls"] += 1
        row["actual_token_min"] = min(row["actual_token_min"], tokens)
        row["actual_token_max"] = max(row["actual_token_max"], tokens)

    def audit_state(self):
        return {
            "unique_id": repr(self.unique_id()),
            "device_index": self.device_index,
            "stream_handle": self.stream_handle,
            "sm_count": self.sm_count,
            "reserved_sms": self.reserved_sms,
            "max_tokens": self.max_tokens,
            "tuning_buckets": list(self.buckets),
            "use_cuda_graph": self.tuning_config.use_cuda_graph,
            "use_cold_l2_cache": self.tuning_config.use_cold_l2_cache,
            "timing_scope": ["input_quantization", "counter_reset", "complete_fc12"],
            "timing_metric": "AutoTuner_mean_gpu_time_ms_full_runner_call",
            "distributed_tuning_strategy": self.tuning_config.distributed_tuning_strategy.value,
            "valid_tactics": (
                None if self._valid_tactics is None else [list(t) for t in self._valid_tactics]
            ),
            "rejected_tactics": list(self._rejected_tactics.values()),
            "selections": list(self._selections.values()),
            "compiled_runners": [r.audit for r in self._compiled_runners.values()],
        }


_tunable_runners: dict[tuple, SharedFc12TunableRunner] = {}


def shared_fc12_autotune_state() -> dict:
    """Snapshot actual dispatches; fallback/capture replay are not autotuned serving winners."""
    return deepcopy(
        {
            "schema_version": 1,
            "custom_op": _SHARED_FC12_OP,
            "catalog_version": _CATALOG_VERSION,
            "candidate_catalog": [shared_fc12_tactic_descriptor(t) for t in _CANDIDATES],
            "instances": [runner.audit_state() for runner in _tunable_runners.values()],
        }
    )


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
    with torch.cuda.device(activation.device):
        runner = _tunable_runners.get(runner_key)
        if runner is None:
            runner = _tunable_runners[runner_key] = SharedFc12TunableRunner(*runner_key)
        tuner = AutoTuner.get()
        selected_runner, tactic = tuner.choose_one(
            _SHARED_FC12_OP, [runner], runner.tuning_config, inputs
        )
        output = selected_runner(inputs, tactic=tactic)
        selected_runner.record_selection(tuner, inputs, tactic)
        return output
