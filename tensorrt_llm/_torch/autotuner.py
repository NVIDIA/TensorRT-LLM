import ast
import contextlib
import copy
import enum
import fcntl
import inspect
import itertools
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from cuda.bindings import driver

import tensorrt_llm
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Unique tag to avoid collisions with other comms
PP_COMM_TAG_AUTOTUNING = 30000


class DistributedTuningStrategy(enum.Enum):
    """
    Strategy for distributed tuning.
    Args:
        BROADCAST: One rank (rank 0) tunes and broadcasts results to others
        INDEPENDENT: Each rank tunes independently (default for non-comm ops)
        MERGE: All ranks participate in tuning and reach merge
        PARALLEL: All ranks participate in tuning with partial tactics
    """
    BROADCAST = "broadcast"
    INDEPENDENT = "independent"
    MERGE = "merge"
    PARALLEL = "parallel"


@dataclass(slots=True, unsafe_hash=True)
class DynamicTensorSpec:
    """
    A specification for a dynamic tensor dimension.
    Args:
        input_idx: The index of the input tensor.
        dim_idx: The index of the dimension to tune.
        gen_tuning_buckets: A tuple of values to try or a function generating values.
        map_to_tuning_buckets: A function to map dimensions to tuning buckets during inference.
    """
    input_idx: int
    dim_idx: int
    gen_tuning_buckets: Union[Tuple[int], Callable] = ()
    map_to_tuning_buckets: Callable = lambda x: x


@dataclass(slots=True, unsafe_hash=True)
class ConstraintSpec:
    """
    A specification for a constraint on a tensor dimension.
    Args:
        input_idx: The index of the input tensor.
        dim_idx: The index of the dimension to constrain.
        infer_shape: A function to infer the shape of the dimension.
    """
    input_idx: int
    dim_idx: int
    infer_shape: Callable


@dataclass(kw_only=True)
class TuningConfig:
    """Configuration for autotuning.

    This class specifies all the tuning configurations for a single tuning process.
    Args:
        dynamic_tensor_specs (Tuple[DynamicTensorSpec]): Specifications for how different tensor dimensions
            should be tuned to optimize performance. Each spec defines:
            - Which input tensor dimension is dynamic
            - How to generate tuning values
            - How to map dimensions to tuning values during inference

            Example:
                >>> config = TuningConfig(
                ...     dynamic_tensor_specs=(
                ...         DynamicTensorSpec(
                ...             input_idx=0,
                ...             dim_idx=1,
                ...             gen_tuning_buckets=(32, 64, 128),
                ...             map_to_tuning_buckets=lambda x: ((x + 31) // 32) * 32
                ...         ),
                ...     )
                ... )
        constraint_specs (Tuple[ConstraintSpec]): Specifications for constraints on tensor dimensions.
            Each spec defines:
            - Which input tensor dimension is constrained
            - How to infer the shape of the dimension based on other dimensions

            Example:
                >>> config = TuningConfig(
                ...     constraint_specs=(
                ...         ConstraintSpec(
                ...             input_idx=1,
                ...             dim_idx=2,
                ...             infer_shape=lambda shapes: shapes[0][0] * 2
                ...         ),
                ...     )
                ... )
        tune_max_num_tokens (int): The maximum saturation number of tokens to be tuned.
            During the inference, the input tensor will be saturated with the same value. Or if
            any value is provided to the choose_one function, the input tensor will be saturated
            with the provided value.
            If not provided, the autotuner will not consider the max num tokens.
        inputs_pre_hook (Callable): A function that takes a list of input tensors, returns a list of modified input tensors.
            It is called before the input tensors are prepared for the tuning process to match the real data distribution.
        use_cold_l2_cache (bool): Whether to use cold L2 cache.
            This flag is to create circular buffer of input tensors to avoid L2 cache hits to simulate cold L2 cache.
            Notice that not all tuning processes can benefit from this feature.
        use_cuda_graph (bool): Whether to use CUDA graph for the tuning process.
        distributed_tuning_strategy (DistributedTuningStrategy): Strategy for distributed tuning.
    """
    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: Tuple[ConstraintSpec, ...] = ()
    tune_max_num_tokens: int = None
    inputs_pre_hook: Callable = None
    use_cold_l2_cache: bool = False
    use_cuda_graph: bool = True
    distributed_tuning_strategy: DistributedTuningStrategy = DistributedTuningStrategy.INDEPENDENT


@dataclass(unsafe_hash=True)
class StaticDim:
    val: int

    def _opt(self):
        return self.val


@dataclass(unsafe_hash=True)
class DynamicDim:
    '''Range of one dimension'''
    min: int
    opt: int
    max: int

    def _opt(self):
        return self.opt


Dim = Union[DynamicDim, StaticDim]


@dataclass
class OptimizationProfile:
    '''Ranges of all tensors, all dimension
    '''
    shapes: List[List[Dim]] = field(default_factory=lambda: [[]])

    def get_hash_key(self):
        return self.get_opt_shapes()

    def get_opt_shapes(self):
        '''Only the opt shapes are considered as hash key
        '''
        # TODO: remove duplicate shape generation
        opt_shapes = []
        for t in self.shapes:
            opt_shapes.append(tuple([d._opt() for d in t]))
        return tuple(opt_shapes)


#TODO: can/shall we use the torch builtin FakeTensor class?
@dataclass
class FakeTensor:
    dtype: torch.dtype
    device: torch.device
    shape: List[Dim]


class TunableRunner(ABC):

    @abstractmethod
    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[Any]:
        """One tactic corresponding to one cuda kernel normally, but how to interpret the meaning
        of tactic is pure internal details of the runner.

        The autotuner will just pass the tactic value to the forward w/o. any knowledge on what the tactic
        means. User can choose to implement their own types of tactic for flexibility, such as using a dict-typed
        to represent a collection of named configs.

        The type of the tactic is arbitrary. But serialization/deserialization of the cache requires that the type is compatible with json.dumps/json.loads.
        To evaluate if a type of tactic is compatible with current workflow, try the following code:
            *  assert YOUR_TACTIC_OBJECT == eval(repr(YOUR_TACTIC_OBJECT))

        tactic==-1 has special meaning, means the fallback kernel which should be able to implement any shapes
        This fallback tactic is needed for 2 reasons:
            * when the autotuner cannot find a valid tactic in it's cache.
            * in eager mode, w/o autotunning the custom op should have at least one kernel, which makes the autotuning
              process an optional process, such that user can opt out.

        We choose not to have a standalone can_implement function, the tactics returned by get_valid_tactics should return
        valid kernel for these given input tensors.
        """
        return [-1]

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(
            self,
            /,  # tensors are position only
            inputs: List[torch.Tensor],
            *,  # all others are keyword args only
            tactic: Any = -1,
            do_preparation: bool = False,
            **kwargs) -> Any:
        """Forward pass for tunable runners.

        Args:
            inputs: List of input tensors (position-only argument)
            tactic: A arbitrary type that represents a specific kernel config.
                    For instance, it can be an integer number that specifies the unique ID of the implementation tactic to use.
                    -1 (default) represents the fallback tactic that must be implemented
                    to handle any input shapes when autotuning is disabled.
            do_preparation: When True, allows one-time setup operations to be performed
                          before tactic evaluation begins. These operations are excluded
                          from the performance measurements during autotuning. Notice that
                          anything prepared in this phase should be persistent in the forward
                          and can be accessed by the following forward calls.

        Returns:
            Any: Output of the forward pass.

        """
        raise NotImplementedError

    def unique_id(self):
        """
        Returns a tuple of the unique id of the runner. The unique id will be converted to a string for the cache key.
        A common practice is to return a tuple of the runner's attributes, for example:
            return (self.output_dtype, self.attribute_1, ...)

        Returns:
            Any: The unique id of the runner, which can be converted to a string for the cache key.
        """
        return tuple(self.__dict__.values())


@contextlib.contextmanager
def autotune(tune_mode: bool = True, cache_path: str = None):
    """Context manager for autotuning with distributed support.

    Args:
        tune_mode: Whether to enable tuning mode
        cache_path: Path to save/load cache files
    """
    autotuner = AutoTuner.get()
    rank = autotuner.mapping.rank

    # if cache_path is provided, use the rank-specific file
    tune_required = tune_mode
    if cache_path is not None:
        # check if the rank-specific file exists
        # if the rank-specific file exists, load it
        file_exists = os.path.exists(cache_path)
        if file_exists:
            logger.info(f"[Autotuner] Loading cache from {cache_path}")
            autotuner.profiling_cache.load_cache(cache_path, rank)

    # record the old tuning mode
    old_mode = autotuner.is_tuning_mode
    autotuner.is_tuning_mode = tune_required
    autotune_enabled = tune_required and not old_mode

    if autotune_enabled:
        logger.info("[Autotuner] Autotuning process starts ...")

    try:
        yield
    finally:
        autotuner.is_tuning_mode = old_mode
        if autotune_enabled:
            logger.info("[Autotuner] Autotuning process ends")

        # save cache
        if cache_path is not None:
            logger.info(f"[Autotuner] Saving cache to {cache_path}")
            autotuner.profiling_cache.save_cache(cache_path, rank)


@dataclass
class AutoTunerStatistics:
    """Statistics collected by the AutoTuner.

    Attributes:
        cache_misses (int): Number of cache misses requiring fallback
        cache_miss_config_collection (Dict[str, Set[OptimizationProfile]]): Collection of configs that caused cache misses
        failed_profiling_count (Dict[str, int]): Number of failed profiling attempts per operation
        tuned_op_profiled_configs (Dict[str, int]): Profiled configurations per operation
        tuned_op_time_cost (Dict[str, float]): Time cost per operation
    """
    cache_misses: int = 0
    cache_miss_config_collection: Dict[str,
                                       Set[tuple]] = field(default_factory=dict)
    failed_profiling_count: Dict[str, Set[Tuple[str, TunableRunner,
                                                OptimizationProfile]]] = field(
                                                    default_factory=dict)
    tuned_op_profiled_configs: Dict[str, int] = field(default_factory=dict)
    tuned_op_time_cost: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a string representation of collected statistics.
        """
        stats_str = ""
        stats_str += f"Cache misses: {self.cache_misses}\n"
        if self.cache_miss_config_collection:
            stats_str += "Cache miss config collection:\n"
            for op, profiles in sorted(
                    self.cache_miss_config_collection.items()):
                stats_str += f"  {op}:\n"
                for profile in sorted(profiles, key=str):
                    stats_str += f"    - Config: {profile}\n"

        if self.tuned_op_profiled_configs:
            stats_str += "Tuned operations:\n"
            for op in sorted(self.tuned_op_profiled_configs.keys()):
                successful = self.tuned_op_profiled_configs[op]
                failed = len(self.failed_profiling_count[op])
                stats_str += f"  {op}:\n"
                stats_str += f"    - Successful configs: {successful}\n"
                stats_str += f"    - Failed profiling count: {failed}\n"
                if failed > 0:
                    stats_str += f"    - Failed profiling combinations:\n"
                    for failed_key in self.failed_profiling_count[op]:
                        stats_str += f"      - {failed_key}\n"

        if self.tuned_op_time_cost:
            stats_str += "Tuned operations time cost:\n"
            for op in sorted(self.tuned_op_time_cost.keys()):
                stats_str += f"  {op}: {self.tuned_op_time_cost[op] * 1000:.4f} milliseconds\n"

        return stats_str


class AutoTunerProfilingCache:
    """AutoTunerCache for caching profiling results.

    The profiling cache can be serialized to disk for persistence across sessions:
        - Use save_cache() to save the cache after tuning
        - Use load_cache() to restore cached results before inference
        - JSON format provides human-readable output and cross-platform compatibility
    """

    def __init__(self):
        self.cache: Dict[Tuple, Tuple] = dict()

        # Cache metadata for local storage and validation
        self.lib_version = tensorrt_llm.__version__
        self.creation_timestamp = time.time()
        # gpu_platform
        self.device_name = torch.cuda.get_device_name()
        self.device_capability = torch.cuda.get_device_capability()

    def __setitem__(self, cache_key: Tuple, value: Tuple) -> None:
        self.cache[cache_key] = value

    def __getitem__(self, cache_key: Tuple) -> Tuple:
        return self.cache[cache_key]

    def __len__(self) -> int:
        return len(self.cache)

    def clear(self) -> None:
        self.cache.clear()

    def fallback_entry(self) -> Tuple:
        # runner_id = 0, tactic = -1
        return 0, -1, float('inf')

    def search_cache(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        apply_map_to_tuning_buckets: bool = True,
    ) -> Tuple[bool, int, int, Dict[str, Any], OptimizationProfile]:
        """Search for cached profiling results matching the current configuration.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            profile (OptimizationProfile): Optimization profile
            apply_map_to_tuning_buckets: If True, apply map_to_tuning_buckets for runtime cache lookups.
                If False, use raw bucket values for tuning cache storage.

        Returns:
            A tuple containing:
            [is_cache_hit, runner_id, tactic, stored_profile]
            runner_id is the index in the current runners list
        """
        for idx, r in enumerate(runners):
            if (cache_key := self.get_cache_key(
                    custom_op, r, input_shapes, tuning_config,
                    apply_map_to_tuning_buckets)) in self.cache:
                # Return the current index in runners list, not the cached runner_id
                cached_runner_id, tactic, min_time = self.cache[cache_key]
                return True, idx, tactic, min_time

        return False, *self.fallback_entry()

    def get_cache_key(
        self,
        custom_op: str,
        runner: TunableRunner,
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        apply_map_to_tuning_buckets: bool = True,
    ) -> Tuple:
        return (
            custom_op,
            runner.__class__.__name__,
            str(runner.unique_id()),
            AutoTuner.get()._find_nearest_profile(
                input_shapes,
                tuning_config.dynamic_tensor_specs,
                tuning_config.constraint_specs,
                tuning_config.tune_max_num_tokens,
                apply_map_to_tuning_buckets,
            ),
        )

    def merge_cache_data(self, cache_data: Dict[Tuple, Tuple]):
        self.cache.update(cache_data)

    def get_specific_custom_op(self, custom_op: str) -> Dict[Tuple, Tuple]:
        return {k: v for k, v in self.cache.items() if k[0] == custom_op}

    def save_cache(self, file_path: Union[str, Path], rank: int) -> None:
        """Save the profiling cache to disk in JSON format.

        Args:
            file_path: Path where to save the cache

        Raises:
            IOError: If file cannot be written

        Note:
            The cache is saved in JSON format which provides human-readable output.
            Some type information may be lost for complex tactic objects.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            serialized_rank_cache_data = self._serialize_cache_data()
            with open(file_path, 'a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                content = f.read()
                if content.strip():
                    current_cache = json.loads(content)
                else:
                    current_cache = {
                        "metadata": self._serialize_metadata(),
                    }
                f.seek(0)
                f.truncate()
                current_cache[f"rank_{rank}"] = serialized_rank_cache_data
                json.dump(current_cache, f, indent=2, default=str)
            logger.info(
                f"[AutoTuner] Successfully saved cache to {file_path} using JSON format"
            )
        except Exception as e:
            logger.error(f"[AutoTuner] Failed to save cache with JSON: {e}")
            raise

    def load_cache(self, file_path: Union[str, Path], rank: int) -> None:
        """Load the profiling cache from disk in JSON format.

        Args:
            file_path: Path to the cache file

        Raises:
            FileNotFoundError: If cache file doesn't exist
            IOError: If file cannot be read

        Note:
            Loading will replace the current cache contents. The cache is loaded
            from JSON format.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                current_cache_contents = json.load(f)
                self._deserialize_metadata(current_cache_contents["metadata"])
                assert f"rank_{rank}" in current_cache_contents, f"Rank {rank} cache not found in {file_path}"
            self.cache = self._deserialize_cache_data(
                current_cache_contents[f'rank_{rank}'])
            logger.info(
                f"[AutoTuner] Successfully loaded cache from {file_path} using JSON format"
            )
        except Exception as e:
            logger.error(f"[AutoTuner] Failed to load cache with JSON: {e}")
            raise

    def _serialize_metadata(self) -> Dict[str, Any]:
        return {
            "lib_version": self.lib_version,
            "creation_timestamp": self.creation_timestamp,
            "device_name": self.device_name,
            "device_capability": self.device_capability,
        }

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> None:
        self.lib_version = metadata["lib_version"]
        self.creation_timestamp = metadata["creation_timestamp"]
        self.device_name = metadata["device_name"]
        self.device_capability = metadata["device_capability"]

    def _serialize_cache_data(self) -> Dict[str, Any]:
        """Convert the profiling cache to a JSON-serializable format.

        Returns:
            Dictionary that can be serialized to JSON

        Note:
            This method handles the conversion of complex objects to JSON-compatible
            representations. Some type information may be lost in the conversion.
        """
        serializable_cache = {}

        for key, value in self.cache.items():
            # Convert any simple object to string for JSON compatibility
            key_str = str(key)
            runner_id, tactic, min_time = value
            tactic_str = repr(tactic)
            try:
                assert tactic == ast.literal_eval(
                    tactic_str
                ), f"Tactic is not compatible with json.dumps/json.loads"
            except Exception as e:
                logger.warning_once(
                    f"[AutoTuner] Could not serialize tactic: {tactic_str} for cache key {key_str} due to {e}. Deserialization may fail.",
                    key=tactic_str)

            serializable_cache[key_str] = {
                "runner_id": runner_id,
                "tactic": tactic_str,
                "min_time": min_time,
            }

        return serializable_cache

    def _deserialize_cache_data(
            self, cache_data: Dict[str, Any]) -> Dict[Tuple, Tuple]:
        """Convert JSON-serialized cache back to the original format.

        Args:
            serializable_cache: Dictionary loaded from JSON

        Returns:
            Profiling cache in the original format

        Note:
            This attempts to reconstruct the original data structures but may not
            perfectly preserve all type information, especially for complex tactic objects.
        """
        cache = {}

        for key_str, value in cache_data.items():
            # Reconstruct the tuple key safely
            try:
                key = ast.literal_eval(key_str)
            except (ValueError, SyntaxError):
                logger.warning(
                    f"[AutoTuner] Could not reconstruct cache key: {key_str}")
                continue
            try:
                tactic = ast.literal_eval(value["tactic"])
            except (ValueError, TypeError):
                logger.warning_once(
                    f"[AutoTuner] Could not deserialize tactic: {value['tactic']} for cache key {key_str}",
                    key=value["tactic"])

            runner_id = value["runner_id"]
            min_time = value["min_time"]

            cache[key] = (runner_id, tactic, min_time)

        return cache


class AutoTuner:
    """AutoTuner for optimizing TensorRT LLM operations.

    This class handles automatic performance tuning of tensor operations by profiling
    different implementations and caching the best performing configurations.

    Args:
        warmup (int): Number of warmup iterations before profiling (default: 3)
        repeat (int): Number of profiling iterations for averaging (default: 10)
        stream_delay_micro_secs (int): Delay on CUDA stream before the profiled kernel runs in microseconds (default: 1000)
    """
    _CUDA_GRAPH_DELAY_MICRO_SECS = 100
    _instance = None

    def __init__(self, warmup=2, repeat=10, stream_delay_micro_secs=1000):
        # Increase log level for AutoTuner associated logger`
        self._log_level_to_info = os.getenv(
            "TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO", '0') == '1'
        self._debug_logger = logger.info if self._log_level_to_info else logger.debug

        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache = AutoTunerProfilingCache()
        self.is_tuning_mode = False

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        # Current captured choose_one() contexts
        self._active_capture: Optional['AutoTuner.TacticsCapture'] = None
        # Last captured choose_one() contexts
        self._last_capture: Optional['AutoTuner.TacticsCapture'] = None

        # Dsitributed tuning state
        self._map_op_to_distributed_strategy: Dict[
            str, DistributedTuningStrategy] = {}
        self._dist: Optional[Distributed] = None
        self._has_received_cache: bool = False
        self.mapping: Mapping = Mapping()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = AutoTuner()
        return cls._instance

    class TacticsCapture:
        """Object returned by capture() that can be iterated to get all tactic combinations.

        This class encapsulates all state related to capturing and replaying tactics:
        - Captured execution contexts
        - Generated tactic configurations
        - Current replay state (which config and call index)
        """

        runner_tactic_comb_checkers: List[Callable] = []

        def __init__(self, autotuner):
            # State for captured contexts
            self._captured_contexts: List[Dict[str, Any]] = []
            self._context_tactics_lists: Optional[List[List[Tuple[int,
                                                                  Any]]]] = None
            # State for replay mode
            self._replay_runner_tactic_list: Optional[List[Tuple[int,
                                                                 int]]] = None
            self._replay_context_idx: int = 0

        def __iter__(self):
            """Iterate through all tactic configurations.

            For single context: yields (runner, tactic)
            For multiple contexts: yields ((runner_ctx0, tactic_ctx0), (runner_ctx1, tactic_ctx1), ...)
            """
            if self._context_tactics_lists is None:
                self._context_tactics_lists = self._generate_context_tactics_lists(
                )

            # Generate cartesian product from context and tactics where all_configrations[i][ctx] = (runner, tactic)
            # Such that each element in all_configrations is a replay of multiple contexts of all possible replays
            for config in itertools.product(*self._context_tactics_lists):
                # config is a tuple of (runner_idx, tactic) for each context
                # Convert to (runner, tactic) format for user
                runner_tactic_pairs = []
                for ctx_idx, (runner_idx, tactic) in enumerate(config):
                    runners = self._captured_contexts[ctx_idx]['runners']
                    runner = runners[runner_idx]
                    runner_tactic_pairs.append((runner, tactic))

                if not all(
                        checker(runner_tactic_pairs) for checker in
                        self.__class__.runner_tactic_comb_checkers):
                    continue

                yield tuple(runner_tactic_pairs)

        def _generate_context_tactics_lists(self):
            """Generate all valid tactic combinations."""
            if not self._captured_contexts:
                raise RuntimeError(
                    "No context available for testing.\n"
                    "Use capture() to capture the operation context first:\n"
                    "  with AutoTuner.get().capture() as tactics_capture:\n"
                    "      output = operation.forward(...)\n")

            # Collect valid tactics for each context separately
            context_tactics_lists = []

            for context in self._captured_contexts:
                runners = context['runners']
                inputs = context['inputs']
                kwargs = context.get('kwargs', {})

                # Collect all valid (runner, tactic) combinations for this context
                tactics_lists = []
                for runner_idx, runner in enumerate(runners):
                    valid_tactics = runner.get_valid_tactics(
                        inputs, OptimizationProfile(), **kwargs)
                    for tactic in valid_tactics:
                        tactics_lists.append((runner_idx, tactic))
                context_tactics_lists.append(tactics_lists)

            return context_tactics_lists

        def is_replaying(self) -> bool:
            """Check if this TacticsCapture is currently in replay mode."""
            return self._replay_runner_tactic_list is not None

        @classmethod
        def register_runner_tactic_comb_checker(cls, checker: Callable):
            cls.runner_tactic_comb_checkers.append(checker)
            return checker

    def choose_one(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
        **kwargs,
    ) -> Tuple:
        """Choose the best runner and tactic combination through performance profiling.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            tuning_config (TuningConfig): Configuration for the tuning process
            inputs (List[torch.Tensor]): Input tensors for profiling
            **kwargs: Arbitrary keyword arguments, will be passed to get_valid_tactics and forward method of each runner

        Returns:
            Tuple: A tuple containing:
                - The selected runner implementation
                - The best tactic ID for that runner (-1 if using fallback)
                - The best config for that runner (if configs is not empty)

        Note:
            The method profiles different implementations and tactics to find the
            optimal combination based on performance measurements. It caches results
            to avoid redundant profiling of the same configuration.
            Although runners[0] with tactic=-1 is always treated as the fallback runner.
            Runner authors are suggested to provide a fallback implementation for each runner to avoid potential issues.
        """

        # Check if we're in replay mode via active TacticsCapture
        if self._active_capture is not None and self._active_capture.is_replaying(
        ):
            tactics_capture = self._active_capture
            call_idx = tactics_capture._replay_context_idx

            assert call_idx < len(tactics_capture._replay_runner_tactic_list
                                  ), "call_idx out of range"
            assert call_idx < len(
                tactics_capture._captured_contexts), "call_idx out of range"
            assert len(tactics_capture._replay_runner_tactic_list) == len(
                tactics_capture._captured_contexts)

            # Check if we have a forced tactic for this call and both custom_op match
            captured_custom_op = tactics_capture._captured_contexts[
                call_idx].get('custom_op')
            if captured_custom_op != custom_op:
                raise RuntimeError(
                    f"Custom op mismatch in kernel testing mode.\n"
                    f"Expected operation: '{captured_custom_op}'\n"
                    f"Actual operation: '{custom_op}'\n"
                    f"Context index: {call_idx}\n"
                    f"Make sure the forward() call in test mode uses the same operation as captured."
                )

            runner_idx, tactic = tactics_capture._replay_runner_tactic_list[
                call_idx]
            # Increment context counter
            tactics_capture._replay_context_idx += 1
            # Reset counter after all contexts have been used
            if tactics_capture._replay_context_idx >= len(
                    tactics_capture._replay_runner_tactic_list):
                tactics_capture._replay_context_idx = 0
            return (runners[runner_idx], tactic)

        # Capture context for testing all underlying kernels
        if self._active_capture is not None and not self._active_capture.is_replaying(
        ):
            self._active_capture._captured_contexts.append({
                'custom_op': custom_op,
                'runners': runners,
                'tuning_config': tuning_config,
                'inputs': inputs,
                'kwargs': kwargs,
            })

        input_shapes = tuple(self._get_input_sizes(inputs))
        is_cache_hit, best_runner_id, best_tactic, min_time = self.profiling_cache.search_cache(
            custom_op,
            runners,
            input_shapes,
            tuning_config,
            apply_map_to_tuning_buckets=True)

        # Early return if it's not tuning, use cache found one or fallback one
        if not self.is_tuning_mode:
            best_runner = runners[best_runner_id]
            # TODO: check the stored runner and tactic can implement this shape here
            # Log the cache miss. Expect no cache miss in inference.
            if not is_cache_hit:
                logger.warning_once(
                    f"[AutoTuner] {custom_op} using the fallback tactic, due to cache miss on input shapes={input_shapes}",
                    key=(custom_op, "warning_autotuning_cache_miss_fallback"))

            return (best_runner, best_tactic)

        # If it's tuning mode and cache hit, return the best runner and tactic to avoid redundant profiling.
        if self.is_tuning_mode and is_cache_hit:
            return (runners[best_runner_id], best_tactic)

        # PP rank does not have cache hit, so we try to receive the cache from the previous rank
        # Notice that only under tuning mode, pp_recv will be called
        self.cache_pp_recv()

        assert len(runners) > 0, "At least one runner is required"
        assert all([isinstance(r, TunableRunner) for r in runners]), \
            "All Given runners must be subclass of TunableRunner"

        # Record the distributed tuning strategy for the custom_op
        self._map_op_to_distributed_strategy[
            custom_op] = tuning_config.distributed_tuning_strategy

        tuning_start_time = time.perf_counter()
        profiles = self._optimization_profiles(tuning_config, inputs)

        # Initialize the statistics for the custom_op
        if custom_op not in self.stats.tuned_op_profiled_configs:
            self.stats.tuned_op_profiled_configs[custom_op] = 0
        if custom_op not in self.stats.failed_profiling_count:
            self.stats.failed_profiling_count[custom_op] = set()
        new_tuning_failure_occurred = False

        # Synchronize ranks before profiling
        if self._should_current_rank_tune(
                tuning_config.distributed_tuning_strategy):
            for p in profiles:
                tensors = self._prepare_input_tensors(p, inputs)
                is_cache_hit, *_ = self.profiling_cache.search_cache(
                    custom_op,
                    runners,
                    p.get_opt_shapes(),
                    tuning_config,
                    apply_map_to_tuning_buckets=False,
                )
                if not is_cache_hit:
                    # Initialize runner and tactic as None in case of no valid tactic or runners are found
                    with nvtx_range(f"{custom_op}, shape {p.get_opt_shapes()}"):
                        best_runner_id, best_tactic, min_time, has_tuning_failure_occurred = self._profile_runners(
                            custom_op, runners, tensors, p, tuning_config,
                            **kwargs)
                    new_tuning_failure_occurred = new_tuning_failure_occurred or has_tuning_failure_occurred

        self._maybe_sync_cache_data(tuning_config.distributed_tuning_strategy,
                                    custom_op)

        # If failed profiling tactics occurs, log the error.
        if new_tuning_failure_occurred:
            logger.warning_once(
                f"[Autotuner] New tuning error occurs:"
                f"Total failed profiling tactics occurs: {len(self.stats.failed_profiling_count[custom_op])} for custom_op={custom_op}. "
                f"This will not block the tuning process. "
                f"Please set TLLM_LOG_LEVEL=WARNING to find out when the tactic profiling fails. "
                f"Set TLLM_LOG_LEVEL=DEBUG to get more details of the failures.",
                key=(custom_op, "warning_autotuning_tuning_error_summary"),
            )

        # Get the best runner and tactic from cache
        # If no valid tactic is found, the fallback runner and tactic will be used
        _, runner_id, tactic, _ = self.profiling_cache.search_cache(
            custom_op, runners, input_shapes, tuning_config)

        tuning_end_time = time.perf_counter()
        self.stats.tuned_op_time_cost[
            custom_op] = self.stats.tuned_op_time_cost.get(
                custom_op, 0) + tuning_end_time - tuning_start_time
        return (runners[runner_id], tactic)

    def _profile_runners(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_tensors: List[torch.Tensor],
        profile: OptimizationProfile,
        tuning_config: TuningConfig,
        **kwargs,
    ) -> float:
        """Profile runners and select the best tactic.

        For multi-rank profiling, only rank 0 performs the actual profiling
        to avoid sync issues when different ranks select different tactics.
        The results are then broadcasted to all other ranks.
        """

        min_time = float('inf')
        has_tuning_failure_occurred = False
        best_runner_id, best_tactic = None, None
        # If the inputs_pre_hook is provided, it will be called before profiling.
        if tuning_config.inputs_pre_hook is not None:
            input_tensors = tuning_config.inputs_pre_hook(input_tensors)
        for runner_id, runner in enumerate(runners):
            # TODO: use FakeTensor here.
            runner_arg_names = {
                p.name
                for p in inspect.signature(runner.forward).parameters.values()
            }
            all_valid_tactics = runner.get_valid_tactics(
                input_tensors, profile, **kwargs)

            valid_tactics = self._maybe_parallelize_tactics(
                all_valid_tactics, tuning_config.distributed_tuning_strategy)
            if "do_preparation" in runner_arg_names and len(valid_tactics) > 0:
                runner(
                    input_tensors,
                    tactic=-1,
                    do_preparation=True,
                    **kwargs,
                )

            for tac in valid_tactics:
                try:
                    with nvtx_range(f"r{runner_id}, tactic {tac}"):
                        time_measured = self._profile_single_kernel(
                            runner=runner,
                            inputs=input_tensors,
                            tactic=tac,
                            tuning_config=tuning_config,
                            use_cuda_graph=tuning_config.use_cuda_graph,
                            **kwargs,
                        )
                except Exception as e:
                    # Handle None tensors for optional inputs
                    shapes = self._get_input_sizes(input_tensors)
                    logger.warning_once(
                        f"[Autotuner] Failed when profiling runner={runner}, tactic={tac}, shapes={shapes}. Error: {e}",
                        key=(custom_op, "warning_autotuning_profile_failure"),
                    )

                    # Record the failed profiling combinations
                    self.stats.failed_profiling_count[custom_op].add(
                        self.profiling_cache.get_cache_key(
                            custom_op,
                            runner,
                            profile.get_opt_shapes(),
                            tuning_config,
                            apply_map_to_tuning_buckets=False))

                    # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                    # or some runtime error occurs during profiling.
                    time_measured = float('inf')
                    has_tuning_failure_occurred = True
                if time_measured < min_time:
                    min_time = time_measured
                    best_runner_id, best_tactic = runner_id, tac

        if best_runner_id is not None:
            # At least one valid (runner, tactic) pair is found
            cache_key = self.profiling_cache.get_cache_key(
                custom_op,
                runners[best_runner_id],
                profile.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False)

            self._debug_logger(
                f"[Autotuner] Profiling runner={runners[best_runner_id]}, tactic={best_tactic} for cache_key={cache_key}."
            )
            # inspect call stack
            # TODO: use named tuple to make it more readable
            self.profiling_cache[cache_key] = (best_runner_id, best_tactic,
                                               min_time)

            self.stats.tuned_op_profiled_configs[custom_op] += 1
        else:
            logger.warning_once(
                f"[Autotuner] No valid runner/tactic was found for custom_op={custom_op}, input_shapes={profile.get_opt_shapes()}. "
                f"At least one valid (runner, tactic) pair is required. "
                f"If get_valid_tactics is intended to return empty list, please ensure that this profile is not valid for the custom_op "
                f"and should not occurs during the inference stage, or fallback tactic is implemented. Otherwise, the the tuning process will crash.",
                key=(custom_op, "warning_autotuning_no_valid_tactic"),
            )

        return best_runner_id, best_tactic, min_time, has_tuning_failure_occurred

    def _get_input_sizes(self, inputs: List[torch.Tensor]) -> List[torch.Size]:

        # Handle None tensors for optional inputs and non-Tensor scalar values
        sizes = [
            input.size() if isinstance(input, torch.Tensor) else torch.Size(
                (0, )) for input in inputs
        ]

        return sizes

    def _profile_single_kernel(
        self,
        runner: TunableRunner,
        inputs: List[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        use_cuda_graph: bool = False,
        **kwargs,
    ) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (Any): Tactic to use for this profiling run

        Returns:
            Average execution time in milliseconds

        Note:
            The method performs warmup runs, then measures multiple iterations
            to get an average execution time. Stream synchronization and delays
            are used to ensure accurate timing.
        """
        input_tensor_batches = self._prepare_input_tensors_with_batches(
            inputs, tuning_config)

        stream = torch.cuda.current_stream()
        # If the warm up time is longer than 0.5ms, we will profile the kernel with fewer repeats.
        profile_fewer_repeat = 2
        short_profile_threshold_ms = 1

        avg_time = float('inf')

        def pure_profile(stream: torch.cuda.Stream, repeat: int):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.stream(stream):
                if use_cuda_graph:
                    with torch.cuda.graph(graph):
                        for r in range(repeat):
                            runner(
                                input_tensor_batches[r %
                                                     len(input_tensor_batches)],
                                tactic=tactic,
                                **kwargs,
                            )

                stream.synchronize()
                if tuning_config.distributed_tuning_strategy == DistributedTuningStrategy.MERGE:
                    # Currently only AllReduce will use this strategy, and only MPI parallel will enable tuning.
                    # TODO: Unified tp barrier for both MPIDist and TorchDist.
                    if hasattr(self._dist, "tp_comm"):
                        self._dist.tp_comm.barrier()

                # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
                if use_cuda_graph:
                    delay_kernel(self._CUDA_GRAPH_DELAY_MICRO_SECS, stream)
                else:
                    delay_kernel(self.stream_delay_micro_secs, stream)

                start.record()

                if use_cuda_graph:
                    graph.replay()
                else:
                    for r in range(repeat):
                        runner(
                            input_tensor_batches[r % len(input_tensor_batches)],
                            tactic=tactic,
                            **kwargs,
                        )

                end.record()
                stream.synchronize()

                return start.elapsed_time(end) / repeat

        # warm up, no timing
        for _ in range(self.warmup):
            runner(input_tensor_batches[-1], tactic=tactic, **kwargs)

        fewer_repeat_avg_time = pure_profile(stream, profile_fewer_repeat)

        disable_short_profile = os.environ.get(
            "TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE", "0") == "1"
        if fewer_repeat_avg_time > short_profile_threshold_ms and not disable_short_profile:
            # directly use the few repeat estimated time to avoid redundant profiling
            avg_time = fewer_repeat_avg_time
        else:
            # profile the kernel with the full repeat to get precise time
            avg_time = pure_profile(stream, self.repeat)

        shapes = self._get_input_sizes(inputs)
        self._debug_logger(
            f"[Autotuner] Profiled runner={runner}, tactic={tactic}, shapes={shapes}: {avg_time:.6f}ms."
        )

        return avg_time

    def _optimization_profiles(
            self, tuning_config: TuningConfig,
            inputs: List[torch.Tensor]) -> List[OptimizationProfile]:
        """Generate optimization profiles for autotuning.

        Args:
            tuning_config (TuningConfig): Tuning configuration
            inputs (List[torch.Tensor]): List of input tensors

        Returns:
            List of OptimizationProfile objects representing different configurations

        Note:
            This method performs a cartesian product of all possible dimension
            combinations specified in dynamic_tensor_specs.
        """
        # every dimension created from the concrete input tensor shape
        # generate some dynamic dimension description based on the dynamic_tensors

        # Zero handles the case where a TRTLLM op has optional or scalar inputs.
        base_profile = OptimizationProfile(
            [[StaticDim(x) for x in t.size()]
             if isinstance(t, torch.Tensor) else [StaticDim(0)]
             for t in inputs])

        generated_profiles: List[OptimizationProfile] = []

        dynamic_dims = []

        for spec in tuning_config.dynamic_tensor_specs:
            assert callable(spec.gen_tuning_buckets) or isinstance(spec.gen_tuning_buckets, (list, tuple)), \
                "The given dynamic dimension must provide a opt value generation function or a list of opt values"
            if callable(spec.gen_tuning_buckets):
                if tuning_config.tune_max_num_tokens is None:
                    # Use the current input size as the opt value
                    opt_shapes = spec.gen_tuning_buckets(
                        base_profile.shapes[spec.input_idx][spec.dim_idx].val)
                else:
                    # Use the tune_max_num_tokens as the opt value
                    opt_shapes = spec.gen_tuning_buckets(
                        tuning_config.tune_max_num_tokens)
            else:
                # Default values is an empty tuple, means that user does not want to tune this dimension.
                opt_shapes = spec.gen_tuning_buckets
            # Add the current input value as one of the opt values
            opt_shapes = set(opt_shapes)
            if tuning_config.tune_max_num_tokens is not None:
                opt_shapes.add(
                    min(
                        tuning_config.tune_max_num_tokens,
                        base_profile.shapes[spec.input_idx][spec.dim_idx].val,
                    ))
            else:
                opt_shapes.add(
                    base_profile.shapes[spec.input_idx][spec.dim_idx].val)
            opt_shapes = sorted(list(opt_shapes))
            opt_shapes_max = tuple(opt_shapes[1:]) + (float('inf'), )
            opt_shapes_max = {
                v1: v2
                for v1, v2 in zip(opt_shapes, opt_shapes_max)
            }
            dynamic_dims.append(
                (spec.input_idx, spec.dim_idx, opt_shapes_max, opt_shapes))

        # grid search, do cartesian product for all the dynamic axis
        dim_grids = itertools.product(*[d[-1] for d in dynamic_dims])
        for opt_point in dim_grids:
            p = copy.deepcopy(base_profile)
            for pos, (input_idx, dim_idx, opt_shapes_max,
                      opt_shapes) in enumerate(dynamic_dims):
                opt_value = opt_point[pos]
                #TODO: fix me, how to set the min and max?
                min_value = opt_value
                max_value = opt_shapes_max[opt_value]
                p.shapes[input_idx][dim_idx] = DynamicDim(
                    min_value, opt_value, max_value)

            # Adjust the profile to satisfy the constraints
            for spec in tuning_config.constraint_specs:
                min_value = opt_value = max_value = spec.infer_shape(
                    p.get_opt_shapes())
                if p.shapes[spec.input_idx] == [StaticDim(0)]:
                    continue
                p.shapes[spec.input_idx][spec.dim_idx] = DynamicDim(
                    min_value, opt_value, max_value)
            generated_profiles.append(p)
            self._debug_logger(f"[Autotuner] Generated profile: {p}")
        return generated_profiles

    @classmethod
    @lru_cache(maxsize=None)
    def _find_nearest_profile(
        cls,
        shapes: Tuple[torch.Size],
        dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...],
        constraint_specs: Tuple[ConstraintSpec, ...],
        tune_max_num_tokens: int = None,
        apply_map_to_tuning_buckets: bool = True,
    ) -> Tuple:
        """Find the nearest optimization profile for given inputs
        User can define their own nearest profile generation method to reduce the host overhead.

        Args:
            shapes: Tuple of input tensor shapes
            tuning_config: Tuning configuration
            apply_map_to_tuning_buckets: If True, apply map_to_tuning_buckets for runtime cache lookups.
                If False, use raw bucket values for tuning cache storage.

        Return:
            Tuple: A tuple containing:
                - attributes: Tuple of runner attributes, sorted.
                - profile: Tuple of input tensor shapes
        """
        base_profile = list(list(shape) for shape in shapes)

        for spec in dynamic_tensor_specs:
            # During runtime: apply map_to_tuning_buckets to map input to bucket
            # During tuning: no mapper, use raw bucket value
            if apply_map_to_tuning_buckets:
                base_profile[spec.input_idx][
                    spec.dim_idx] = spec.map_to_tuning_buckets(
                        base_profile[spec.input_idx][spec.dim_idx])

            if tune_max_num_tokens is not None:
                base_profile[spec.input_idx][spec.dim_idx] = min(
                    base_profile[spec.input_idx][spec.dim_idx],
                    tune_max_num_tokens)

        # associated dimensions dependent on other free dynamic dimensions, so assign -1 in the profile
        for spec in constraint_specs:
            if base_profile[spec.input_idx] == [0]:
                continue
            base_profile[spec.input_idx][spec.dim_idx] = -1

        return tuple(tuple(shape) for shape in base_profile)

    def _create_tensor_like(self, origin_tensor: torch.Tensor,
                            dims: List[Dim]) -> torch.Tensor:
        """Create a new tensor matching the properties of the original tensor.

        Args:
            origin_tensor (torch.Tensor): Template tensor to match
            dims (List[Dim]): List of dimensions for the new tensor

        Returns:
            New tensor with specified dimensions and matching properties

        Note:
            Creates a zero tensor with the same dtype and device as the original,
            but with dimensions specified by the dims parameter.
        """
        dtype = origin_tensor.dtype
        device = origin_tensor.device
        shapes = []
        for i, d in enumerate(dims):
            if isinstance(d, StaticDim):
                assert d.val == origin_tensor.shape[i]
                shapes.append(d.val)
            else:
                # TODO: how to make sure the created Tensor has the min/max info
                assert isinstance(d, DynamicDim)
                shapes.append(d.opt)

        if len(dims) == 2 and isinstance(dims[0], DynamicDim) and isinstance(
                dims[1], StaticDim) and (dtype == torch.int32
                                         or dtype == torch.int64):
            # We should be carefully about int values, since they might be index like topk_index.
            # We want to keep them legal, so just repeating input tensor.
            repeat_times = (shapes[0] + origin_tensor.shape[0] -
                            1) // origin_tensor.shape[0]
            dup_tensor = origin_tensor.repeat(repeat_times, 1)[:shapes[0]]
            return dup_tensor

        # TODO: FIXME, sometimes the content of the tensor can affect the performance, like MOE
        # One solution is to manituplate the tensor content to make it more like the real data
        # during the tuning process. This can by controlled in the preparation phase by the runner.
        # It must not use all zero tensors. Otherwise the timing results become unreliable.
        if dtype == torch.float4_e2m1fn_x2:
            return torch.randint(-5, 5, shapes,
                                 device=device).to(torch.uint8).view(dtype)
        else:
            return torch.randint(-5, 5, shapes, device=device).to(dtype)

    def _prepare_input_tensors(
            self, profile: OptimizationProfile,
            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        tensors = []
        for i, p in enumerate(profile.shapes):
            if any(isinstance(d, DynamicDim) for d in p):
                tensor = self._create_tensor_like(inputs[i], p)
            else:
                tensor = inputs[i]
            tensors.append(tensor)
        return tensors

    def _prepare_input_tensors_with_batches(
        self,
        inputs: List[torch.Tensor],
        tuning_config: TuningConfig,
    ) -> List[List[torch.Tensor]]:
        if not tuning_config.use_cold_l2_cache:
            return [inputs]

        one_buffer_bytes = sum(
            input.numel() *
            input.element_size() if isinstance(input, torch.Tensor) else 0
            for input in inputs)
        if one_buffer_bytes <= 0:
            self._debug_logger(
                "[Autotuner] No tensor inputs or zero-sized tensors; falling back to single-batch profiling."
            )
            return [inputs]

        num_buffers = self._get_l2_cache_size_in_bytes(
        ) * 3 // one_buffer_bytes + 1
        num_buffers = min(num_buffers, self.repeat + 1)

        inputs_list = [inputs]
        for _ in range(num_buffers - 1):
            inputs_list.append(
                list(t.clone() if isinstance(t, torch.Tensor) else t
                     for t in inputs))

        self._debug_logger(
            f"[Autotuner] use_cold_l2_cache={tuning_config.use_cold_l2_cache}, use {num_buffers} different tensors for profiling"
        )
        return inputs_list

    def clear_cache(self) -> None:
        """Clear the profiling cache."""
        self.profiling_cache.clear()

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()

    def print_profiling_cache(self):
        self._debug_logger(f"[Autotuner] The profiling_cache entries:")
        self._debug_logger(
            f"[Autotuner] Cache contents: (custom_op, runner, hash(attributes), shape_profiles) -> (runner_id, tactic, shape_profile(ignored))"
        )
        for key, value in self.profiling_cache.cache.items():
            runner_id, tactic, min_time = value
            self._debug_logger(
                f"[Autotuner] {key}: (runner_id={runner_id}, tactic={tactic}, min_time={min_time})"
            )

        self.print_statistics()

    def print_statistics(self):
        self._debug_logger(f"[Autotuner] The statistics:")
        for line in self.stats.__str__().split("\n"):
            self._debug_logger(line)

    @contextlib.contextmanager
    def capture(self):
        """Context manager for capturing execution contexts for testing.

        Returns a TacticsCapture object that can be iterated to get all valid
        (runner, tactic) combinations.

        Example:
            >>> # Single context case
            >>> with AutoTuner.get().capture() as tactics_capture:
            ...     y = custom_op.forward(x)
            >>>
            >>> for runner, tactic in tactics_capture:
            ...     with AutoTuner.get().replay(runner, tactic):
            ...         y = custom_op.forward(x)

            >>> # Multiple contexts case
            >>> with AutoTuner.get().capture() as tactics_capture:
            ...     y = custom_op1.forward(x)
            ...     z = custom_op2.forward(y)
            >>>
            >>> for config in tactics_capture:
            ...     with AutoTuner.get().replay(config):
            ...         y = custom_op1.forward(x)
            ...         z = custom_op2.forward(y)
        """
        tactics_capture = self.TacticsCapture(self)
        self._active_capture = tactics_capture
        try:
            yield tactics_capture
        finally:
            self._active_capture = None
            self._last_capture = tactics_capture

    @contextlib.contextmanager
    def replay(self, *config: Tuple[Tuple[TunableRunner, int], ...]):
        """Context manager for replaying with specific runner/tactic configuration.

        Args:
            config:
                - A tuple of (runner, tactic) pairs. The tuple size matches the number of captured choose_one() contexts.
        """
        # Parse config argument
        if len(config) == 1:
            if isinstance(config[0], tuple):
                # Multiple contexts: replay(((r0,t0), (r1,t1), ...))
                runner_tactic_pairs = list(config[0])
            else:
                # Also handle single context passed as replay((runner, tactic))
                runner_tactic_pairs = [config[0]]
        else:
            raise ValueError(
                f"Invalid config for replay: {config}\n"
                "Expected replay(((runner, tactic), (runner, tactic), ...))")

        # Find the TacticsCapture to use
        tactics_capture = self._active_capture or self._last_capture

        if tactics_capture is None:
            raise RuntimeError(
                "No TacticsCapture available for replay. "
                "Make sure you've called capture() before replay().")

        # Temporarily set as active capture during replay
        prev_active = self._active_capture
        self._active_capture = tactics_capture

        runner_tactic_list = []
        for ctx_idx, (runner, tactic) in enumerate(runner_tactic_pairs):
            runners = tactics_capture._captured_contexts[ctx_idx]['runners']
            runner_idx = runners.index(runner)
            runner_tactic_list.append((runner_idx, tactic))

        self._debug_logger(
            f"[Autotuner][replay]: Testing configuration: {runner_tactic_list}")

        # Replay the contexts with given (runner, tactic) pairs
        tactics_capture._replay_runner_tactic_list = runner_tactic_list
        tactics_capture._replay_context_idx = 0

        try:
            yield
        finally:
            tactics_capture._replay_runner_tactic_list = None
            tactics_capture._replay_context_idx = 0
            # Restore previous active capture state
            self._active_capture = prev_active

    def _get_l2_cache_size_in_bytes(self, device_id: int = 0) -> int:
        device = self._checkCudaErrors(driver.cuDeviceGet(device_id))
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                device,
            ))

    def _checkCudaErrors(self, result) -> Any:
        status = result[0]
        if status != driver.CUresult.CUDA_SUCCESS:
            code = getattr(status, "value", status)
            raise RuntimeError(
                f"CUDA error code={code}({self._cudaGetErrorEnum(status)})")
        # CUDA APIs always return the status as the first element of the result tuple
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]

    def _cudaGetErrorEnum(self, error) -> str:
        from cuda.bindings import nvrtc
        if isinstance(error, driver.CUresult):
            err, name = driver.cuGetErrorName(error)
            return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    def setup_distributed_state(self, mapping: Mapping, dist: Distributed):
        """Setup distributed communication state for autotuning."""
        self.mapping = mapping
        self._dist = dist
        self._debug_logger(
            f"[AutoTuner] Whether using distributed tuning: {self._is_distributed()}"
        )

    def _is_distributed(self) -> bool:
        """Check if we're in a distributed environment."""
        return self.mapping is not None and self.mapping.tp_size > 1 and self._dist is not None

    def _maybe_parallelize_tactics(
            self, all_valid_tactics: List[Any],
            strategy: DistributedTuningStrategy) -> List[Any]:
        """Parallelize tactics across all TP ranks if strategy is PARALLEL."""
        if strategy == DistributedTuningStrategy.PARALLEL:
            # only distribute across TP ranks
            # each TP rank will only tune the tactics that are assigned to it
            tp_size = self.mapping.tp_size
            tp_rank = self.mapping.tp_rank
            valid_tactics = []
            for idx, tactic in enumerate(all_valid_tactics):
                if idx % tp_size == tp_rank:
                    valid_tactics.append(tactic)
            return valid_tactics
        else:
            return all_valid_tactics

    def _maybe_sync_cache_data(self, strategy: DistributedTuningStrategy,
                               custom_op: str):
        """Synchronize cache data across all ranks."""
        if not self._is_distributed():
            return

        if strategy == DistributedTuningStrategy.BROADCAST:
            self._broadcast_cache_data(custom_op)
        elif strategy == DistributedTuningStrategy.INDEPENDENT:
            return
        elif strategy == DistributedTuningStrategy.MERGE:
            self._merge_cache_data(custom_op)
        elif strategy == DistributedTuningStrategy.PARALLEL:
            self._merge_cache_data(custom_op)
        else:
            logger.error(
                f"[AutoTuner] Unknown distributed tuning strategy: {strategy}, falling back to independent"
            )
            return

    def _merge_cache_data(self, custom_op: str):
        cache_data = self.profiling_cache.get_specific_custom_op(custom_op)
        merged_cache_data = dict()
        all_cache_data = self._dist.tp_cp_allgather(obj=cache_data)

        for data in all_cache_data:
            for key, value in data.items():
                current_time = merged_cache_data.get(key, [
                    float('inf'),
                ])[-1]
                if value[-1] < current_time:
                    merged_cache_data[key] = value

        self.profiling_cache.merge_cache_data(merged_cache_data)

    def _broadcast_cache_data(
        self,
        custom_op: str,
    ) -> None:
        """Broadcast tactics from root rank to all other ranks."""
        cache_data = self.profiling_cache.get_specific_custom_op(custom_op)
        root = 0
        cache_data = self._dist.tp_cp_broadcast(obj=cache_data, root=root)

        self.profiling_cache.merge_cache_data(cache_data)

    def _should_current_rank_tune(self,
                                  strategy: DistributedTuningStrategy) -> bool:
        """Determine if this rank should perform tuning based on strategy."""
        if not self._is_distributed():
            return True

        if strategy == DistributedTuningStrategy.BROADCAST:
            # Only rank 0 tunes
            return self.mapping.rank == 0
        elif strategy in {
                DistributedTuningStrategy.INDEPENDENT,
                DistributedTuningStrategy.MERGE,
                DistributedTuningStrategy.PARALLEL,
        }:
            # All ranks tune independently
            return True
        else:
            logger.error(
                f"[AutoTuner] Unknown distributed tuning strategy: {strategy}, falling back to independent"
            )
            return True

    def cache_pp_recv(self):
        if self.mapping.has_pp() and not self.mapping.is_first_pp_rank(
        ) and not self._has_received_cache:
            self._debug_logger(
                f"[AutoTuner] Receiving cache data from previous pp rank {self.mapping.prev_pp_rank()}"
            )
            profiling_cache = self._dist.recv_object(
                src=self.mapping.prev_pp_rank(),
                tag=PP_COMM_TAG_AUTOTUNING,
            )
            # Guarantee that only receive cache once during a single warm-up run
            # Notice that this flag should be reset after each warm-up run because isend is always called
            self._has_received_cache = True
            self.profiling_cache.merge_cache_data(profiling_cache)

    def cache_pp_send(self):
        if self.mapping.has_pp() and not self.mapping.is_last_pp_rank():
            self._debug_logger(
                f"[AutoTuner] Sending cache data to next pp rank {self.mapping.next_pp_rank()}"
            )
            self._dist.isend_object(
                self.profiling_cache.cache,
                dest=self.mapping.next_pp_rank(),
                tag=PP_COMM_TAG_AUTOTUNING,
            ).wait()

    def clean_pp_flag(self):
        self._has_received_cache = False
