# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from visual_gen.configs.op_manager import AttentionOpManager, BaseOpManager, LinearOpManager
from visual_gen.configs.parallel import get_dit_parallel_config
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.logger import get_logger

matplotlib.use("Agg")  # Use non-interactive backend

logger = get_logger(__name__)


class TunableParam:
    """A class representing a tunable parameter with its value and acceptable range.

    Supports various types including str, int, float, torch.Tensor, torch.nn.Parameter, etc.
    """

    def __init__(
        self,
        value: Any,
        param_range: Optional[Union[List, Tuple, Dict]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize a tunable parameter.

        Args:
            value: The current value of the parameter
            param_range: The acceptable range/choices for the parameter:
                - For numeric types: (min_value, max_value) or [min_value, max_value]
                - For discrete choices: [choice1, choice2, ...]
                - For complex types: dict with constraints
            name: Optional name for the parameter
            description: Optional description
        """
        self.value = value
        self.param_range = param_range
        self.name = name
        self.description = description
        self.param_type = type(value).__name__

        # Validate initial value is within range
        if param_range is not None:
            self._validate_value(value)

    def _validate_value(self, value: Any) -> bool:
        """Validate if a value is within the acceptable range."""
        if self.param_range is None:
            return True

        if isinstance(self.param_range, (list, tuple)):
            if len(self.param_range) == 2 and all(
                isinstance(x, (int, float)) for x in self.param_range
            ):
                # Numeric range [min, max]
                if isinstance(value, (int, float)):
                    return self.param_range[0] <= value <= self.param_range[1]
            else:
                # Discrete choices
                return value in self.param_range
        elif isinstance(self.param_range, dict):
            # Custom validation rules for complex types
            return True  # Implement custom validation logic as needed

        return True

    def set_value(self, new_value: Any) -> bool:
        """Set a new value for the parameter if it's within range.

        Returns:
            bool: True if value was set successfully, False otherwise
        """
        if self._validate_value(new_value):
            self.value = new_value
            return True
        else:
            logger.warning(f"Value {new_value} is not within acceptable range {self.param_range}")
            return False

    def get_value(self) -> Any:
        """Get the current value of the parameter."""
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter to a serializable dictionary."""
        result = {
            "name": self.name,
            "param_type": self.param_type,
            "param_range": self.param_range,
            "description": self.description,
        }

        # Handle different value types for serialization
        if isinstance(self.value, torch.Tensor):
            result["value"] = {
                "type": "torch.Tensor",
                "shape": list(self.value.shape),
                "dtype": str(self.value.dtype),
                "data": self.value.detach().cpu().tolist(),
            }
        elif isinstance(self.value, torch.nn.Parameter):
            result["value"] = {
                "type": "torch.nn.Parameter",
                "shape": list(self.value.shape),
                "dtype": str(self.value.dtype),
                "data": self.value.detach().cpu().tolist(),
                "requires_grad": self.value.requires_grad,
            }
        else:
            result["value"] = self.value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TunableParam":
        """Create a TunableParam instance from a dictionary."""
        value = data["value"]

        # Reconstruct complex types
        if isinstance(value, dict) and "type" in value:
            if value["type"] == "torch.Tensor":
                tensor_data = torch.tensor(value["data"])
                if len(value["shape"]) > 0:
                    tensor_data = tensor_data.reshape(value["shape"])
                value = tensor_data
            elif value["type"] == "torch.nn.Parameter":
                tensor_data = torch.tensor(value["data"])
                if len(value["shape"]) > 0:
                    tensor_data = tensor_data.reshape(value["shape"])
                value = torch.nn.Parameter(
                    tensor_data, requires_grad=value.get("requires_grad", True)
                )

        return cls(
            value=value,
            param_range=data.get("param_range"),
            name=data.get("name"),
            description=data.get("description"),
        )


class OpTuningMetaInfo:
    def __init__(
        self,
        op_type: str,
        inputs_info: Dict[str, Any],
    ):
        self.op_type = op_type
        self.inputs_info = inputs_info
        k, v = list(self.inputs_info.items())[0]
        self.input_key = k
        self.input_size = int(torch.prod(torch.tensor(v["shape"])))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "inputs_info": self.inputs_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpTuningMetaInfo":
        return cls(
            op_type=data["op_type"],
            inputs_info=data["inputs_info"],
        )

    def __gt__(self, other: "OpTuningMetaInfo") -> bool:
        return self.input_size > other.input_size

    def __lt__(self, other: "OpTuningMetaInfo") -> bool:
        return self.input_size < other.input_size

    def __eq__(self, other: "OpTuningMetaInfo") -> bool:
        is_equal = True
        if self.op_type != other.op_type:
            is_equal = False
        if self.inputs_info != other.inputs_info:
            is_equal = False
        return is_equal

    def __str__(self) -> str:
        return f"{self.input_key} size:{self.input_size}"

    def __hash__(self) -> int:
        def _dict_to_hashable(d):
            """Convert a dictionary to a hashable tuple of sorted items."""
            if isinstance(d, dict):
                return tuple(sorted((k, _dict_to_hashable(v)) for k, v in d.items()))
            elif isinstance(d, (list, tuple)):
                return tuple(_dict_to_hashable(item) for item in d)
            else:
                return d

        return hash(
            (
                self.op_type,
                _dict_to_hashable(self.inputs_info),
            )
        )


class OpTuningResult:
    def __init__(
        self,
        impl_type: str,
        elapsed_time_ms: List[float],
        op_outputs: Dict[str, torch.Tensor],
        is_baseline: bool = False,
        accuracy_metrics: Dict[str, List[float]] = None,
    ):
        self.impl_type = impl_type
        self.is_baseline = is_baseline
        self.elapsed_time_ms = elapsed_time_ms
        self.op_outputs = op_outputs
        self.accuracy_metrics = accuracy_metrics if accuracy_metrics is not None else {}

    def get_elapsed_time_mean(self) -> float:
        """Get the mean of elapsed_time_ms."""
        if not self.elapsed_time_ms:
            return 0.0
        return sum(self.elapsed_time_ms) / len(self.elapsed_time_ms)

    def get_accuracy_metric_mean(self, metric_name: str) -> float:
        """Get the mean value of a specific accuracy metric."""
        if metric_name not in self.accuracy_metrics:
            return 0.0

        metric_values = self.accuracy_metrics[metric_name]
        if isinstance(metric_values, list) and len(metric_values) > 0:
            return sum(metric_values) / len(metric_values)
        elif isinstance(metric_values, (int, float)):
            return float(metric_values)
        else:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        # Ensure accuracy_metrics are JSON serializable and calculate statistics
        serializable_metrics = {}
        if self.accuracy_metrics is not None:
            for key, value in self.accuracy_metrics.items():
                if isinstance(value, list) and len(value) > 0:
                    # Handle list values - calculate statistics
                    serializable_metrics[key] = sum(value) / len(value)  # mean
                    serializable_metrics[key + "_max"] = max(value)
                    serializable_metrics[key + "_min"] = min(value)
                    # Keep original list for backward compatibility
                    serializable_metrics[key + "_raw"] = value
                else:
                    # Handle scalar values or empty lists
                    serializable_metrics[key] = value

        # Calculate statistics for elapsed_time_ms
        elapsed_time_stats = {}
        if self.elapsed_time_ms:
            elapsed_time_stats["elapsed_time_ms"] = sum(self.elapsed_time_ms) / len(
                self.elapsed_time_ms
            )  # mean
            elapsed_time_stats["elapsed_time_ms_max"] = max(self.elapsed_time_ms)
            elapsed_time_stats["elapsed_time_ms_min"] = min(self.elapsed_time_ms)
        else:
            elapsed_time_stats["elapsed_time_ms"] = 0
            elapsed_time_stats["elapsed_time_ms_max"] = 0
            elapsed_time_stats["elapsed_time_ms_min"] = 0

        return {
            "impl_type": self.impl_type,
            "is_baseline": self.is_baseline,
            "elapsed_time_ms": self.elapsed_time_ms,  # Keep original list for backward compatibility
            "elapsed_time_stats": elapsed_time_stats,
            "accuracy_metrics": serializable_metrics,
            # Note: op_outputs are not serialized to save space
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpTuningResult":
        """Create an OpTuningResult instance from a dictionary."""
        return cls(
            impl_type=data["impl_type"],
            elapsed_time_ms=data["elapsed_time_ms"],
            op_outputs={},  # Empty dict since we don't serialize op_outputs
            is_baseline=data.get("is_baseline", False),
            accuracy_metrics=data.get("accuracy_metrics", {}),
        )


class TuningResult:
    # Class-level cache to ensure same instances for identical (step, cfg_type, layer_name)
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, step: int, cfg_type: str, layer_name: str):
        """Ensure that instances with same step, cfg_type, layer_name are identical."""
        key = (step, cfg_type, layer_name)

        with cls._lock:
            if key not in cls._instances:
                instance = super(TuningResult, cls).__new__(cls)
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, step: int, cfg_type: str, layer_name: str):
        # Only initialize once per unique instance
        if hasattr(self, "_initialized"):
            return

        self.step = step
        self.cfg_type = cfg_type
        self.layer_name = layer_name
        # key: meta_info, value: Dict[impl_type, OpTuningResult]
        self.results_of_each_shape = {}
        self._initialized = True

    def _find_matching_meta_info(self, target_meta_info: OpTuningMetaInfo):
        """Find existing meta_info that matches the target_meta_info."""
        for existing_meta_info in self.results_of_each_shape.keys():
            if existing_meta_info == target_meta_info:
                return existing_meta_info
        return None

    def add_result(self, meta_info: OpTuningMetaInfo, result: OpTuningResult):
        """Add tuning result to results_of_each_shape.

        Args:
            meta_info: Operation metadata info as key
            result: Tuning result to add/merge
        """
        # Check if we already have a matching meta_info
        existing_meta_info = self._find_matching_meta_info(meta_info)

        if existing_meta_info is None:
            # First time seeing this meta_info, create a new dict with this result
            self.results_of_each_shape[meta_info] = {result.impl_type: result}
        else:
            # meta_info already exists, check if there's a result with same impl_type
            existing_results = self.results_of_each_shape[existing_meta_info]

            if result.impl_type in existing_results:
                # Found existing result with same impl_type: merge the data
                existing_result = existing_results[result.impl_type]
                existing_result.elapsed_time_ms.extend(result.elapsed_time_ms)

                # Merge accuracy_metrics
                for metric_key, metric_values in result.accuracy_metrics.items():
                    if metric_key in existing_result.accuracy_metrics:
                        existing_result.accuracy_metrics[metric_key].extend(metric_values)
                    else:
                        existing_result.accuracy_metrics[metric_key] = metric_values.copy()
            else:
                # No existing result with same impl_type: add new result to the dict
                existing_results[result.impl_type] = result

    def search_result(self, meta_info: OpTuningMetaInfo) -> Dict[str, OpTuningResult]:
        """Search the op tuning result according to the meta_info."""
        if meta_info in self.results_of_each_shape:
            return self.results_of_each_shape[meta_info]
        return None

    def fuzzy_search_result(self, meta_info: OpTuningMetaInfo) -> Dict[str, OpTuningResult]:
        """Fuzzy search the op tuning result according to the meta_info."""
        result = self.search_result(meta_info)
        if result is None:
            # fuzzy search by compare output size
            sorted_meta_infos = sorted(self.results_of_each_shape.keys())
            if meta_info < sorted_meta_infos[0]:
                result = self.results_of_each_shape[sorted_meta_infos[0]]
            elif meta_info > sorted_meta_infos[-1]:
                result = self.results_of_each_shape[sorted_meta_infos[-1]]
            else:
                for i in range(len(sorted_meta_infos) - 1):
                    if sorted_meta_infos[i] < meta_info and sorted_meta_infos[i + 1] > meta_info:
                        result = self.results_of_each_shape[sorted_meta_infos[i + 1]]
                        break
        return result


class AutoTuner:
    """AutoTuner class for recording and managing I/O tensors during pipeline execution.

    Features:
    1. Automatic denoising step detection
    2. I/O tensor recording per step
    3. Context manager support for global access
    4. Save/load functionality
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        mode: str,
        result_dir: str,
        max_memory_gb: float = 20.0,
        attention_temperature: float = 1.0,
        attention_block_size: int = 128,
    ):
        """Initialize AutoTuner instance.

        Args:
            mode: Mode of operation, either "tuning" or "inference"
            result_dir: Directory to save tuning results and heatmaps
            max_memory_gb: Maximum memory usage for io_tensors in GB
            save_attention_heatmaps: Whether to save attention heatmaps during tuning
            attention_temperature: Temperature parameter for attention softmax (default: 1.0)
            attention_block_size: Block size for max pooling to reduce heatmap resolution (default: 128)
        """
        assert mode in ["tuning", "inference"], "Mode must be either 'tuning' or 'inference'"
        self.mode = mode
        self.result_dir = result_dir
        self.linear_choices = LinearOpManager.linear_choices
        self.attn_choices = AttentionOpManager.attn_choices

        # set save_attention_heatmaps from env
        save_attention_heatmaps = os.getenv("SAVE_ATTENTION_HEATMAPS", "False").lower() == "true"
        logger.info(f"Save attention heatmaps: {save_attention_heatmaps}")

        self.io_tensors_per_step: Dict[int, Dict[str, Any]] = {}
        self.tuning_results_per_step_per_layer: Dict[
            int, Dict[str, Dict[str, Dict[str, OpTuningResult]]]
        ] = {}
        self.tunable_params: Dict[str, TunableParam] = {}
        self._step_counter = 0

        # Memory management for io_tensors
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3  # Convert GB to bytes
        self._saved_chunks_index = 0  # Index for saved chunk files
        self._saved_chunks_metadata: Dict[int, str] = {}  # Maps chunk_index to file_path
        self._estimated_memory_usage = 0  # Estimated memory usage in bytes

        # Attention heatmap saving configuration
        self.save_attention_heatmaps = save_attention_heatmaps
        self.attention_temperature = attention_temperature
        self.attention_block_size = attention_block_size
        self.heatmap_dir = None
        if save_attention_heatmaps and mode == "tuning":
            self.heatmap_dir = os.path.join(result_dir, "attention_heatmaps")
            os.makedirs(self.heatmap_dir, exist_ok=True)
            logger.info(f"Attention heatmaps will be saved to {self.heatmap_dir}")

        if mode == "tuning":
            if not os.path.exists(result_dir):
                logger.info(f"Creating result directory {result_dir}")
                os.makedirs(result_dir)
            self.result_dir = result_dir
        else:
            if not os.path.isdir(result_dir):
                raise FileNotFoundError(f"Result path {result_dir} does not exist")
            self.result_dir = result_dir

    def __enter__(self) -> "AutoTuner":
        """Context manager entry."""
        with AutoTuner._lock:
            AutoTuner._instance = self
            if self.mode == "inference":
                self.load_tuning_results()

        logger.debug("AutoTuner context started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        with AutoTuner._lock:
            if self.mode == "tuning":
                self.export_tuning_results()
                self._save_and_clear_io_tensors_chunk()
            AutoTuner._instance = None
        logger.debug("AutoTuner context ended")

    @classmethod
    def get_instance(cls) -> Optional["AutoTuner"]:
        """Get the current global AutoTuner instance."""
        with cls._lock:
            return cls._instance

    def benchmark(
        self,
        op_manager: BaseOpManager,
        impl_type: str,
        inputs: Dict[str, torch.Tensor],
        is_baseline: bool = False,
    ) -> OpTuningResult:
        """Benchmark the given implementation with the given inputs."""
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        op_impl = op_manager.get_impl(impl_type)

        torch.cuda.synchronize()
        # Timing
        start_event.record()
        op_outputs = op_impl(**inputs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        op_type = op_manager.op_type()
        logger.debug(
            f"Benchmark: op type {op_type}, impl type {impl_type} took {elapsed_time_ms:.3f} ms"
        )

        return OpTuningResult(
            impl_type,
            [round(elapsed_time_ms, 3)],
            op_outputs,
            is_baseline=is_baseline,
            accuracy_metrics={},
        )

    def _get_op_inputs(
        self, impl_type: str, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get the op inputs from the inputs dictionary."""
        selected_inputs = inputs.copy()

        special_inputs = {}
        if "special_inputs" in inputs:
            special_inputs = selected_inputs["special_inputs"]
            del selected_inputs["special_inputs"]
        else:
            return selected_inputs

        if impl_type in special_inputs:
            selected_inputs.update(special_inputs[impl_type])
        return selected_inputs

    def _get_inputs_info(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        inputs_info = {}
        for k, v in inputs.items():
            if v is None or v == {}:
                continue
            if isinstance(v, torch.Tensor):
                inputs_info[k] = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                }
            elif k == "special_inputs":
                continue
            else:
                inputs_info[k] = v
        return inputs_info

    def tune(
        self,
        layer_name: str,
        op_manager: BaseOpManager,
        baseline_impl,
        inputs: Dict[str, torch.Tensor],
        accuracy_metrics: List[str] = ["cosine_similarity", "mse"],
        step: Optional[int] = None,
    ) -> None:
        """Tune the given implementation with the given inputs.
        We will search the best implementation from the available implementations and their tunable parameters.
        For exmaple, we compare the performance of the "DefaultAttn", "SageAttn", "BlockwiseSparseAttn". For "BlockwiseSparseAttn", we will further tune the sparsity of the attention mask.

        Args:
            layer_name: The name of the layer to tune
            op_manager: The op manager to tune, such as AttentionOpManager or LinearOpManager
            baseline_impl: The baseline implementation to be compared with, such as "default"
            inputs: The inputs to the op
            record_io_tensors: Whether to record the input and output tensors of baseline
            accuracy_metrics: The accuracy metrics to tune. If None, it will use ["cosine_similarity", "mse"].
            step: Current denoising step. If None, it will use the current step from PipelineConfig.current_denoising_step.
        """
        available_impls = op_manager.get_registered_types()
        tunable_impls = []
        if op_manager.op_type() == "linear":
            tunable_impls = self.linear_choices.copy()
            if (
                "svd-nvfp4" in tunable_impls
                and "svd_qweight" not in inputs["special_inputs"]["svd-nvfp4"]
            ):
                tunable_impls.remove("svd-nvfp4")
        elif op_manager.op_type() == "attention":
            tunable_impls = self.attn_choices
        else:
            raise ValueError(f"Unsupported op manager type: {op_manager.op_type()}")

        for impl_type in tunable_impls:
            if impl_type not in available_impls:
                logger.error(
                    f"Implementation {impl_type} is not in the available implementations: {available_impls}"
                )

        if "trtllm-fp8-blockwise" in tunable_impls:
            rows, cols = inputs["weight"].shape[-2:]
            if rows % 128 != 0 or cols % 128 != 0:
                logger.debug(
                    f"layer {layer_name} weight shape {rows}x{cols} is not divisible by 128, cannot use trtllm-fp8-blockwise"
                )
                tunable_impls.remove("trtllm-fp8-blockwise")
        assert baseline_impl in tunable_impls, (
            f"Baseline implementation {baseline_impl} is not in the available implementations: {tunable_impls}"
        )

        if step is None:
            if PipelineConfig.current_denoising_step is not None:
                step = PipelineConfig.current_denoising_step
            else:
                step = self._step_counter
                self._step_counter += 1

        # warmup in step 0
        if step == 0:
            for impl_type in tunable_impls:
                logger.debug(f"Warming up {impl_type} for step {step}, layer {layer_name}")
                current_inputs = self._get_op_inputs(impl_type, inputs)
                _ = self.benchmark(op_manager, impl_type, current_inputs)

        baseline_result = self.benchmark(
            op_manager, baseline_impl, self._get_op_inputs(baseline_impl, inputs), is_baseline=True
        )
        if PipelineConfig.cfg_type is None or PipelineConfig.cfg_type == "positive":
            cfg_type = "positive"
        else:
            cfg_type = "negative"
        if op_manager.record_io_tensors:
            io_tensors = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    io_tensors[key] = value.detach().cpu().clone()
            output_tensors = baseline_result.op_outputs
            if isinstance(output_tensors, torch.Tensor):
                output_tensors = [output_tensors]
            for i, output_tensor in enumerate(output_tensors):
                io_tensors[f"baseline_output_{i}"] = output_tensor.detach().cpu().clone()
            self.record_io_tensor_per_step_per_layer(io_tensors, step, cfg_type, layer_name)

        # Save attention heatmaps if this is an attention layer
        if self.save_attention_heatmaps and "query" in inputs and "key" in inputs:
            query = inputs["query"]
            key = inputs["key"]

            # Check if query and key have the expected shape [batch_size, num_heads, seq_len, head_dim]
            if len(query.shape) == 4 and len(key.shape) == 4 and query.shape == key.shape:
                cfg_type = (
                    "positive"
                    if PipelineConfig.cfg_type is None or PipelineConfig.cfg_type == "positive"
                    else "negative"
                )

                # Compute attention correlation
                attention_weights = self._compute_attention_weights(
                    query,
                    key,
                    temperature=self.attention_temperature,
                    block_size=self.attention_block_size,
                    device=query.device,
                )

                # Save heatmaps
                self._save_heatmaps_per_head(attention_weights, step, layer_name, cfg_type)

                logger.debug(f"Saved attention heatmaps for {layer_name} at step {step}")
            else:
                logger.warning(
                    f"Skipping attention heatmaps for {layer_name}: unexpected tensor shapes"
                )

        # op meta info
        inputs_info = self._get_inputs_info(inputs)
        op_tuning_meta_info = OpTuningMetaInfo(op_manager.op_type(), inputs_info)

        benchmark_impls = tunable_impls.copy()
        benchmark_impls.remove(baseline_impl)
        for impl_type in benchmark_impls:
            result = self.benchmark(op_manager, impl_type, self._get_op_inputs(impl_type, inputs))
            # ring Attention may return lse
            baseline_output = (
                baseline_result.op_outputs[0]
                if isinstance(baseline_result.op_outputs, tuple)
                else baseline_result.op_outputs
            )
            current_output = (
                result.op_outputs[0] if isinstance(result.op_outputs, tuple) else result.op_outputs
            )
            for metric in accuracy_metrics:
                if metric == "cosine_similarity":
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        baseline_output.flatten().float(), current_output.flatten().float(), dim=0
                    )
                    if dist.is_initialized():
                        dist.all_reduce(cosine_similarity, op=dist.ReduceOp.AVG)
                    # Convert tensor to scalar for JSON serialization
                    result.accuracy_metrics[metric] = [cosine_similarity.item()]
                    logger.debug(
                        f"Cosine similarity between baseline and {impl_type}: {cosine_similarity:.3f}"
                    )
                elif metric == "mse":
                    mse = torch.nn.functional.mse_loss(baseline_output, current_output)
                    mse = mse.mean()
                    if dist.is_initialized():
                        dist.all_reduce(mse, op=dist.ReduceOp.AVG)
                    # Convert tensor to scalar for JSON serialization
                    result.accuracy_metrics[metric] = [mse.item()]
                    logger.debug(f"MSE between baseline and {impl_type}: {mse:.3f}")
                else:
                    raise ValueError(f"Unsupported accuracy metric: {metric}")

            # Free the memory of the op outputs after all metrics are computed
            del result.op_outputs
            torch.cuda.empty_cache()

            self.record_tuning_result_per_step_per_layer(
                op_tuning_meta_info, result, step, cfg_type, layer_name
            )

        outputs = baseline_result.op_outputs.clone()
        del baseline_result.op_outputs
        torch.cuda.empty_cache()
        self.record_tuning_result_per_step_per_layer(
            op_tuning_meta_info, baseline_result, step, cfg_type, layer_name
        )

        return outputs

    def select_best_impl(
        self,
        step: int,
        layer_name: str,
        op_type: str,
        inputs: Dict[str, torch.Tensor],
        cosine_similarity_threshold: float = None,
        mse_threshold: float = None,
    ) -> str:
        """Select the best implementation from the available implementations."""
        if cosine_similarity_threshold is not None:
            assert cosine_similarity_threshold >= 0 and cosine_similarity_threshold <= 1, (
                "Cosine similarity threshold must be between 0 and 1"
            )
        if mse_threshold is not None:
            assert mse_threshold >= 0, "MSE threshold must be non-negative"

        if step is None:
            if PipelineConfig.current_denoising_step is not None:
                step = PipelineConfig.current_denoising_step
            else:
                step = self._step_counter
                self._step_counter += 1
        if step not in self.tuning_results_per_step_per_layer:
            raise ValueError(f"No tuning results found for step {step}")
        if PipelineConfig.cfg_type is None or PipelineConfig.cfg_type == "positive":
            cfg_type = "positive"
        else:
            cfg_type = "negative"
        if cfg_type not in self.tuning_results_per_step_per_layer[step]:
            raise ValueError(f"No tuning results found for cfg_type {cfg_type} in step {step}")
        if layer_name not in self.tuning_results_per_step_per_layer[step][cfg_type]:
            raise ValueError(
                f"No tuning results found for layer_name {layer_name} in step {step} and cfg_type {cfg_type}"
            )
        tuning_results = self.tuning_results_per_step_per_layer[step][cfg_type][layer_name]
        cfg_type = PipelineConfig.cfg_type
        if cfg_type is None:
            cfg_type = "positive"
        else:
            cfg_type = "negative"
        # op meta info
        inputs_info = self._get_inputs_info(inputs)
        op_tuning_meta_info = OpTuningMetaInfo(op_type, inputs_info)

        op_tuning_result = tuning_results.fuzzy_search_result(op_tuning_meta_info)
        if op_tuning_result is None:
            raise RuntimeError(f"No tuning result found for step {step}, layer_name {layer_name}")

        if "best_impl" in op_tuning_result:
            return op_tuning_result["best_impl"].impl_type

        best_impl = self._find_best_impl(
            op_tuning_result, cosine_similarity_threshold, mse_threshold
        )
        # Cache the best impl so that we don't need to re-select it for each inference
        op_tuning_result["best_impl"] = op_tuning_result[best_impl]
        return best_impl

    def _find_best_impl(
        self,
        op_tuning_result: TuningResult,
        cosine_similarity_threshold: float,
        mse_threshold: float,
    ) -> str:
        if "best_impl" in op_tuning_result:
            logger.debug("Best impl exists, will be overwritten")
        best_impl = None
        best_result = None
        found_baseline = False
        for impl_type, result in op_tuning_result.items():
            if impl_type == "meta_info":
                continue
            if result.is_baseline:
                best_impl = impl_type
                best_result = result
                found_baseline = True
                break

        if not found_baseline:
            raise RuntimeError("No baseline implementation found")

        for impl_type, result in op_tuning_result.items():
            if impl_type == "meta_info" or result.is_baseline:
                continue
            if cosine_similarity_threshold is not None:
                assert "cosine_similarity" in result.accuracy_metrics, (
                    "Cosine similarity metric not found in tuning results"
                )
                # Use precomputed mean value if available, otherwise calculate on the fly
                cosine_sim = result.accuracy_metrics.get("cosine_similarity", 0.0)
                if cosine_sim < cosine_similarity_threshold:
                    logger.debug(
                        f"Cosine similarity {cosine_sim} is less than threshold {cosine_similarity_threshold}, skipping"
                    )
                    continue
            if mse_threshold is not None:
                assert "mse" in result.accuracy_metrics, "MSE metric not found in tuning results"
                # Use precomputed mean value if available, otherwise calculate on the fly
                mse_val = result.accuracy_metrics.get("mse", float("inf"))
                if mse_val > mse_threshold:
                    logger.debug(
                        f"MSE {mse_val} is greater than threshold {mse_threshold}, skipping"
                    )
                    continue
            # Use elapsed time mean for comparison
            if result.get_elapsed_time_mean() < best_result.get_elapsed_time_mean():
                best_impl = impl_type
                best_result = result
        return best_impl

    def _create_tuning_results_per_step_per_layer(
        self, step: int, cfg_type: str, layer_name: str
    ) -> None:
        """Create the tuning results for the current pipeline step."""
        assert step is not None, "Step must be provided"
        assert layer_name is not None, "Layer name must be provided"
        if step not in self.tuning_results_per_step_per_layer:
            self.tuning_results_per_step_per_layer[step] = {}

        if cfg_type not in self.tuning_results_per_step_per_layer[step]:
            self.tuning_results_per_step_per_layer[step][cfg_type] = {}
        if layer_name not in self.tuning_results_per_step_per_layer[step][cfg_type]:
            self.tuning_results_per_step_per_layer[step][cfg_type][layer_name] = {}
        return self.tuning_results_per_step_per_layer[step][cfg_type][layer_name]

    def record_tuning_result_per_step_per_layer(
        self,
        meta_info: OpTuningMetaInfo,
        result: OpTuningResult,
        step: int,
        cfg_type: str,
        layer_name: str,
    ) -> None:
        """Record the tuning result for the current pipeline step and layer_name.
        Storage structure: step → cfg_type → layer_name → impl_type → result
        """
        self._create_tuning_results_per_step_per_layer(step, cfg_type, layer_name)
        tuning_results = TuningResult(step, cfg_type, layer_name)
        tuning_results.add_result(meta_info, result)
        self.tuning_results_per_step_per_layer[step][cfg_type][layer_name] = tuning_results

    def record_io_tensor_per_step_per_layer(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        step: int,
        cfg_type: str,
        layer_name: str,
    ) -> None:
        """Record input/output tensors for the current pipeline step.

        Args:
            layer_name: The name of the layer to record
            tensor_dict: Dictionary containing tensors to record
            step: Optional step number. If None, auto-detect from PipelineConfig.current_denoising_step.
        """
        assert step is not None, "Step must be provided"

        logger.debug(f"Recording io tensors for step {step}")

        # Initialize step record if not exists
        if step not in self.io_tensors_per_step:
            self.io_tensors_per_step[step] = {}
        if cfg_type not in self.io_tensors_per_step[step]:
            self.io_tensors_per_step[step][cfg_type] = {}
        if layer_name not in self.io_tensors_per_step[step][cfg_type]:
            self.io_tensors_per_step[step][cfg_type][layer_name] = {}
        # Record tensors with metadata
        io_data = self.io_tensors_per_step[step][cfg_type][layer_name]
        for name, tensor in tensor_dict.items():
            if name in io_data:
                logger.warning(
                    f"Tensor {name} already exists for step {step}, layer {layer_name}, skipping"
                )
                continue
            if isinstance(tensor, torch.Tensor):
                io_data[name] = tensor.detach().cpu().clone()
        self._estimated_memory_usage += self._calculate_io_tensors_memory(io_data)
        if self._estimated_memory_usage > self.max_memory_bytes:
            memory_gb = self._estimated_memory_usage / (1024**3)
            logger.info(
                f"IO tensors memory usage ({memory_gb:.2f} GB) exceeds limit ({self.max_memory_gb} GB)"
            )
            self._save_and_clear_io_tensors_chunk()

    def clear_recorded_data(self) -> None:
        """Clear all recorded data."""
        self.io_tensors_per_step.clear()
        self.tuning_results_per_step_per_layer.clear()
        self.tunable_params.clear()
        self._step_counter = 0
        self._estimated_memory_usage = 0
        self._saved_chunks_index = 0
        self._saved_chunks_metadata.clear()

    @torch.no_grad()
    def _compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        temperature: float = 1.0,
        block_size: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Compute attention correlation (softmax of query @ key^T) for each head.

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            temperature: Temperature for softmax (default: 1.0 for no scaling)
            block_size: Block size for max pooling to reduce resolution

        Returns:
            Attention weights of shape [batch_size, num_heads, reduced_seq_len, reduced_seq_len] (on CPU)
        """
        # Move tensors to CPU to save GPU memory; Move to GPU to speed up computation
        query = query.bfloat16().to(device)
        key = key.bfloat16().to(device)

        # Scale by sqrt(head_dim) as is standard in attention
        scale = 1.0 / np.sqrt(query.size(-1))

        # Compute attention scores: query @ key^T
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale / temperature

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply max pooling to reduce resolution
        attention_weights = F.max_pool2d(
            attention_weights, kernel_size=block_size, stride=block_size
        )

        # Move back to CPU to save GPU memory
        attention_weights_cpu = attention_weights.float().cpu()

        return attention_weights_cpu

    def _save_heatmaps_per_head(
        self, attention_weights: torch.Tensor, step: int, layer_name: str, cfg_type: str
    ) -> None:
        """Create and save raw high-resolution heatmap images for each attention head.
        The image resolution matches the attention weight matrix shape exactly.

        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            step: Denoising step
            layer_name: Name of the attention layer
            cfg_type: Configuration type (positive/negative)
        """
        if not self.save_attention_heatmaps or self.heatmap_dir is None:
            return

        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Take the first batch
        weights_batch = attention_weights[0].numpy()

        # Clean layer name for filename
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Create raw heatmaps for each head
        for head_idx in range(num_heads):
            head_weights = weights_batch[head_idx]  # Shape: [seq_len, seq_len]

            # Create figure with exact pixel size matching the attention matrix
            # Set DPI to 1 so that figure size directly corresponds to pixel size
            fig = plt.figure(figsize=(seq_len, seq_len), dpi=1)
            ax = fig.add_axes([0, 0, 1, 1])  # Use entire figure area

            # Remove all axes decorations
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            # Display the attention weights as image
            # origin='upper' puts the origin at top-left (query=0 at top, key=0 at left)
            # extent controls the data coordinates of the image
            im = ax.imshow(
                head_weights,
                cmap="viridis",  # Use a good colormap
                aspect="equal",
                interpolation="nearest",
                origin="upper",
                extent=[0, seq_len, seq_len, 0],
            )  # [left, right, bottom, top]

            dit_parallel_config = get_dit_parallel_config()
            if dit_parallel_config.ring_size() > 1:
                raise NotImplementedError(
                    "Ring attention is not supported for attention heatmaps, please use ulysses instead"
                )

            if dit_parallel_config.ulysses_size() > 1:
                global_head_idx = dit_parallel_config.ulysses_rank() * num_heads + head_idx
            else:
                global_head_idx = head_idx
            # Save with exact resolution
            head_filename = (
                f"step_{step:03d}_{cfg_type}_{clean_layer_name}_head_{global_head_idx:02d}.jpg"
            )
            head_path = os.path.join(self.heatmap_dir, head_filename)

            # Save without any padding or borders
            plt.savefig(
                head_path,
                format="jpg",
                dpi=1,  # 1:1 pixel mapping
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
                edgecolor="none",
            )
            plt.close()

        logger.debug(
            f"Saved {num_heads} attention heatmaps for {layer_name} ({cfg_type}) at step {step}"
        )

    def _calculate_io_tensors_memory(self, io_tensors: Dict[str, torch.Tensor]) -> int:
        """Calculate estimated memory usage of io_tensors_per_step in bytes."""
        total_memory = 0
        for tensor in io_tensors.values():
            if isinstance(tensor, torch.Tensor):
                total_memory += tensor.numel() * tensor.element_size()
        return total_memory

    def _save_and_clear_io_tensors_chunk(self) -> None:
        """Save current io_tensors_per_step to a chunk file and clear it."""
        if not self.io_tensors_per_step:
            return

        chunk_filename = f"io_tensors_chunk_{self._saved_chunks_index}.pt"
        chunk_path = os.path.join(self.result_dir, chunk_filename)

        # Save the current data
        torch.save(self.io_tensors_per_step, chunk_path)

        # Record the chunk metadata
        self._saved_chunks_metadata[self._saved_chunks_index] = chunk_path

        # Clear current data and update counters
        self.io_tensors_per_step.clear()
        self._estimated_memory_usage = 0
        self._saved_chunks_index += 1

        logger.info(f"Saved io_tensors chunk {self._saved_chunks_index - 1} to {chunk_path}")

    def export_tuning_results(self, result_path: str = None) -> None:
        """Export the tuning results to a file."""
        if dist.is_initialized() and dist.get_rank() != 0:
            # Non-rank0 processes wait for rank0 to finish writing
            if dist.is_initialized():
                dist.barrier()
            return
        if result_path is None:
            result_path = os.path.join(self.result_dir, "tuning_results.json")
        # Convert OpTuningResult objects to serializable dictionaries
        serializable_results = {}
        for step, step_data in self.tuning_results_per_step_per_layer.items():
            serializable_results[step] = {}
            for cfg_type, cfg_data in step_data.items():
                serializable_results[step][cfg_type] = {}
                for layer_name, tuning_result in cfg_data.items():
                    serializable_results[step][cfg_type][layer_name] = {}
                    # tuning_result is a TuningResult instance
                    if isinstance(tuning_result, TuningResult):
                        for meta_info, result_dict in tuning_result.results_of_each_shape.items():
                            meta_key = str(meta_info)
                            serializable_results[step][cfg_type][layer_name][meta_key] = {
                                "meta_info": meta_info.to_dict(),
                                "results": {
                                    impl_type: result.to_dict()
                                    for impl_type, result in result_dict.items()
                                },
                            }

        with open(result_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Exported tuning results to {result_path}")

        # Ensure all processes wait for rank0 to finish writing
        if dist.is_initialized():
            dist.barrier()

    def load_tuning_results(self) -> None:
        """Load the tuning results from a file."""
        result_path = os.path.join(self.result_dir, "tuning_results.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Tuning results file {result_path} does not exist")

        with open(result_path, "r") as f:
            serializable_results = json.load(f)

        # Clear TuningResult cache to avoid conflicts during loading
        TuningResult._instances.clear()

        linear_cossim_threshold = LinearOpManager.cosine_similarity_threshold
        attn_cossim_threshold = AttentionOpManager.cosine_similarity_threshold
        linear_mse_threshold = LinearOpManager.mse_threshold
        attn_mse_threshold = AttentionOpManager.mse_threshold

        # Convert dictionaries back to TuningResult objects
        self.tuning_results_per_step_per_layer = {}
        for step_str, step_data in serializable_results.items():
            step = int(step_str)  # JSON keys are strings
            self.tuning_results_per_step_per_layer[step] = {}
            for cfg_type, cfg_data in step_data.items():
                self.tuning_results_per_step_per_layer[step][cfg_type] = {}
                for layer_name, layer_data in cfg_data.items():
                    # Create a new TuningResult instance
                    tuning_result = TuningResult(step, cfg_type, layer_name)

                    # Load each meta_info and its results
                    for meta_key, meta_result_data in layer_data.items():
                        if (
                            isinstance(meta_result_data, dict)
                            and "meta_info" in meta_result_data
                            and "results" in meta_result_data
                        ):
                            # Reconstruct OpTuningMetaInfo
                            meta_info = OpTuningMetaInfo.from_dict(meta_result_data["meta_info"])

                            # Reconstruct dict of OpTuningResult
                            result_dict = {}
                            for impl_type, result_dict_data in meta_result_data["results"].items():
                                result_dict[impl_type] = OpTuningResult.from_dict(result_dict_data)

                            # pre-compute best_impl to accelerate search
                            op_type = meta_info.op_type
                            if op_type == LinearOpManager.op_type():
                                cosine_similarity_threshold = linear_cossim_threshold
                                mse_threshold = linear_mse_threshold
                            elif op_type == AttentionOpManager.op_type():
                                cosine_similarity_threshold = attn_cossim_threshold
                                mse_threshold = attn_mse_threshold
                            else:
                                raise ValueError(f"Unsupported op_type: {op_type}")
                            best_impl = self._find_best_impl(
                                result_dict, cosine_similarity_threshold, mse_threshold
                            )
                            result_dict["best_impl"] = result_dict[best_impl]

                            # Add to TuningResult
                            tuning_result.results_of_each_shape[meta_info] = result_dict
                        else:
                            logger.warning(
                                f"Unexpected data structure for {meta_key}: {meta_result_data}"
                            )

                    self.tuning_results_per_step_per_layer[step][cfg_type][layer_name] = (
                        tuning_result
                    )

        logger.info(f"Loaded tuning results from {result_path}")

        best_impl_result_path = os.path.join(
            self.result_dir,
            f"best_impl_results_linear_cossim_{linear_cossim_threshold}_mse_{linear_mse_threshold}_attn_cossim_{attn_cossim_threshold}_mse_{attn_mse_threshold}.json",
        )
        self.export_tuning_results(best_impl_result_path)


# Convenience function for global access
def get_auto_tuner() -> Optional[AutoTuner]:
    """Get the current global AutoTuner instance."""
    return AutoTuner.get_instance()
