import contextlib
import copy
import inspect
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch

from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.logger import logger


@dataclass(kw_only=True)
class TuningConfig:
    """Configuration for autotuning.

    This class specifies all the tuning configurations for a single tuning process.
    Args:
        dynamic_tensors (Dict[int, Dict[int, Tuple[Union[List[int], Callable], Callable]]]):
            how different tensor dimensions should be tuned to optimize performance. It allows
            defining which input tensor dimensions are dynamic and how they should be tuned
            by providing shape generators and rounding rules.

            A nested dictionary specifying tuning rules:
            - First level key: Input tensor index (0-based)
            - Second level key: Dimension index to tune (0-based)
            - Value: Tuple of (shape_generator, round_rule) where:
                - shape_generator: List of values to try or function generating values
                - round_rule: Function to round dimensions to valid values during inference

            Example:
                >>> config = TuningConfig(
                ...     dynamic_tensors={
                ...         0: {  # First input tensor
                ...             1: (  # Second dimension
                ...                 [32, 64, 128],  # Try these sizes
                ...                 lambda x: ((x + 31) // 32) * 32  # Round to multiple of 32
                ...             )
                ...         }
                ...     }
                ... )
        constraints (Dict[int, Dict[int, Callable]]):
            A nested dictionary specifying constraints on the dimensions:
            - First level key: Input tensor index (0-based)
            - Second level key: Dimension index to constrain (0-based)
            - Value: Function to apply to the dimension

            Example:
            >>> config = TuningConfig(
                ...     constraints={
                ...         1: {    # constrained tensor index
                ...             2: lambda shapes: shapes[0][0] * 2  # constrained dimension index and constraint function
                ...         }
                ...     }
                ... )
    """
    dynamic_tensors: Dict[int,
                          Dict[int,
                               Tuple[List[int],
                                     Callable]]] = field(default_factory=dict)
    constraints: Dict[int, Dict[int, Callable]] = field(default_factory=dict)


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
    shapes: List[List[Dim]]

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
    def get_valid_tactics(self, inputs: List[FakeTensor]) -> List[int]:
        """One tactic corresponding to one cuda kernel normally, but how to interpret the meaning
        of tactic is pure internal details of the runner.

        The autotuner will just pass the tactic value to the forward w/o any knowledge on what the tactic
        means.

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
            tactic: int = -1,
            do_preparation: bool = False) -> Any:
        """Forward pass for tunable runners.

        Args:
            inputs: List of input tensors (position-only argument)
            tactic: Integer ID specifying which implementation tactic to use.
                   -1 (default) represents the fallback tactic that must be implemented
                   to handle any input shapes when autotuning is disabled.
            do_preparation: When True, allows one-time setup operations to be performed
                          before tactic evaluation begins. These operations are excluded
                          from the performance measurements during autotuning. Notice that
                          anything prepared in this phase should be persistent in the forward
                          and can be accessed by the following forward calls.

        Returns:
            Any: Output of the forward pass

        """
        raise NotImplementedError


@contextlib.contextmanager
def autotune(tune_mode: bool = True):
    old_mode = AutoTuner.get().is_tuning_mode
    AutoTuner.get().is_tuning_mode = tune_mode
    autotune_enabled = tune_mode and not old_mode
    if autotune_enabled:
        logger.info("[Autotuner]: Autotuning process starts ...")
    try:
        yield
    finally:
        AutoTuner.get().is_tuning_mode = old_mode
        if autotune_enabled:
            logger.info("[Autotuner]: Autotuning process ends")


@dataclass
class AutoTunerStatistics:
    """Statistics collected by the AutoTuner.

    Attributes:
        cache_misses (int): Number of cache misses requiring fallback
        cache_miss_config_collection (Dict[str, Set[OptimizationProfile]]): Collection of configs that caused cache misses
        failed_profiling_count (Dict[str, int]): Number of failed profiling attempts per operation
        tuned_op_total_configs (Dict[str, int]): Total configurations tried per operation
        tuned_op_successful_configs (Dict[str, int]): Successful configurations per operation
    """
    cache_misses: int = 0
    cache_miss_config_collection: Dict[str, Set[OptimizationProfile]] = field(
        default_factory=dict)
    failed_profiling_count: Dict[str, Set[Tuple[str, TunableRunner,
                                                OptimizationProfile]]] = field(
                                                    default_factory=dict)
    tuned_op_total_configs: Dict[str, int] = field(default_factory=dict)
    tuned_op_successful_configs: Dict[str, int] = field(default_factory=dict)

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

        if self.tuned_op_total_configs:
            stats_str += "Tuned operations:\n"
            for op in sorted(self.tuned_op_total_configs.keys()):
                total = self.tuned_op_total_configs[op]
                successful = self.tuned_op_successful_configs.get(op, 0)
                failed = len(self.failed_profiling_count.get(op, set()))
                success_rate = (successful / total * 100) if total > 0 else 0
                stats_str += f"  {op}:\n"
                stats_str += f"    - Total configs tried: {total}\n"
                stats_str += f"    - Successful configs: {successful}\n"
                stats_str += f"    - Failed profiling count: {failed}\n"
                if failed > 0:
                    stats_str += f"    - Failed profiling combinations:\n"
                    for failed_key in self.failed_profiling_count[op]:
                        stats_str += f"      - {failed_key}\n"
                stats_str += f"    - Success rate: {success_rate:.1f}%\n"

        return stats_str


class AutoTuner:
    """AutoTuner for optimizing TensorRT-LLM operations.

    This class handles automatic performance tuning of tensor operations by profiling
    different implementations and caching the best performing configurations.

    Args:
        warmup (int): Number of warmup iterations before profiling (default: 3)
        repeat (int): Number of profiling iterations for averaging (default: 10)
        stream_delay_micro_secs (int): Delay on CUDA stream before the profiled kernel runs in microseconds (default: 1000)
    """
    _instance = None

    def __init__(self, warmup=3, repeat=10, stream_delay_micro_secs=1000):
        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache = {}
        self.is_tuning_mode = False

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        self.profiling_debug = True

    @classmethod
    def get(cls):
        if cls._instance == None:
            cls._instance = AutoTuner()
        return cls._instance

    def search_cache(
        self, custom_op: str, runners: List[TunableRunner],
        profile: OptimizationProfile
    ) -> Tuple[bool, TunableRunner, int, OptimizationProfile]:
        """Search for cached profiling results matching the current configuration.
        """
        for r in runners:
            cache_key = self.get_cache_key(custom_op, r, profile)
            if cache_key in self.profiling_cache:
                return True, *self.profiling_cache[cache_key]

        return False, runners[0], -1, profile

    def choose_one(self, custom_op: str, runners: List[TunableRunner],
                   tuning_config: TuningConfig, inputs: List[torch.Tensor],
                   **kwargs) -> Tuple[TunableRunner, int]:
        """Choose the best runner and tactic combination through performance profiling.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            tuning_config (TuningConfig): Configuration for the tuning process
            inputs (List[torch.Tensor]): Input tensors for profiling
            **kwargs: Arbitrary keyword arguments, will be passed to get_valid_tactics and forward method of each runner

        Returns:
            Tuple[TunableRunner, int]: A tuple containing:
                - The selected runner implementation
                - The best tactic ID for that runner (-1 if using fallback)

        Note:
            The method profiles different implementations and tactics to find the
            optimal combination based on performance measurements. It caches results
            to avoid redundant profiling of the same configuration.
            Although runners[0] with tactic=-1 is always treated as the fallback runner.
            Runner authors are suggested to provide a fallback implementation for each runner to avoid potential issues.
        """

        assert len(runners) > 0, "At least one runner is required"
        assert all([isinstance(r, TunableRunner) for r in runners]), \
            "All Given runners must be subclass of TunableRunner"

        profile = self._find_nearest_profile(tuning_config.dynamic_tensors,
                                             tuning_config.constraints, inputs)

        # Early return if it's not tuning, use cache found one or fallback one
        if not self.is_tuning_mode:
            is_cache_hit, runner, tactic, stored_profile = self.search_cache(
                custom_op, runners, profile)

            # TODO: check the stored runner and tactic can implement this shape here
            # Should not directly try (runner, tactic) here, or it will hurt a lot of inference perf.

            # Record the cache miss config.
            # Expect no cache miss in inference. Thus, any cache miss should be recorded.
            if not is_cache_hit:
                self.stats.cache_misses += 1
                if custom_op not in self.stats.cache_miss_config_collection:
                    self.stats.cache_miss_config_collection[custom_op] = set()
                self.stats.cache_miss_config_collection[custom_op].add(
                    profile.get_hash_key())
                logger.debug(f"[AutoTunner]: Using fallback tactic")
                assert runner == runners[0] \
                    and tactic == -1, f"Should use fallback runner {runners[0]} and tactic {-1}, but got runner {runner} and tactic {tactic}"

            logger.debug(
                f"[AutoTuner]: Using {runner} {tactic} for profile:{stored_profile}"
            )
            return runner, tactic

        profiles = self._optimization_profiles(tuning_config.dynamic_tensors,
                                               tuning_config.constraints,
                                               inputs)
        # Record the total configs to try
        self.stats.tuned_op_total_configs[custom_op] = len(profiles)

        for p in profiles:
            tensors = [
                self._create_tensor_like(orig_tensor, dims)
                for dims, orig_tensor in zip(p.shapes, inputs)
            ]
            is_cache_hit, runner, tactic, stored_profile = self.search_cache(
                custom_op, runners, p)
            if not is_cache_hit:
                min_time = float('inf')
                # Initialize runner and tactic as None in case of no valid tactic or runners are found
                runner, tactic = None, None
                for r in runners:
                    # TODO: use FakeTensor here.
                    valid_tactics = r.get_valid_tactics(tensors)
                    runner_arg_names = {
                        p.name
                        for p in inspect.signature(
                            r.forward).parameters.values()
                    }
                    if "do_preparation" in runner_arg_names:
                        r(tensors, tactic=-1, do_preparation=True, **kwargs)
                    for tac in valid_tactics:
                        try:
                            time_measured = self._profile_single_kernel(
                                r, tensors, tac, **kwargs)
                        except Exception as e:
                            logger.error(
                                f"[Autotuner]: Failed when profiling {r} {tac}, shapes={[t.size() for t in tensors]}. Error occurred: {e}"
                            )

                            # Record the failed profiling combinations
                            if custom_op not in self.stats.failed_profiling_count:
                                self.stats.failed_profiling_count[
                                    custom_op] = set()
                            self.stats.failed_profiling_count[custom_op].add(
                                self.get_cache_key(custom_op, r, p))

                            # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                            # or some runtime error occurs during profiling.
                            time_measured = float('inf')
                        if time_measured < min_time:
                            min_time = time_measured
                            runner, tactic = r, tac
                if runner is not None:
                    # At least one valid (runner, tactic) pair is found
                    cache_key = self.get_cache_key(custom_op, runner, p)
                    self.profiling_cache[cache_key] = (runner, tactic, p)
                    self.stats.tuned_op_successful_configs[
                        custom_op] = self.stats.tuned_op_successful_configs.get(
                            custom_op, 0) + 1
                    logger.debug(
                        f"[Autotuner]: profiling chosen runner: {runner} {tactic} for {cache_key}"
                    )

        # Get the best runner and tactic from cache
        # If no valid tactic is found, the fallback runner and tactic will be used
        _, runner, tactic, _ = self.search_cache(custom_op, runners, profile)

        return runner, tactic

    def _profile_single_kernel(self, runner: TunableRunner,
                               inputs: List[torch.Tensor], tactic: int,
                               **kwargs) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (int): Tactic ID to use for this profiling run

        Returns:
            Average execution time in milliseconds

        Note:
            The method performs warmup runs, then measures multiple iterations
            to get an average execution time. Stream synchronization and delays
            are used to ensure accurate timing.
        """
        stream = torch.cuda.current_stream()
        # warm up, no timing
        for _ in range(self.warmup):
            runner(inputs, tactic=tactic, **kwargs)
        stream.synchronize()

        # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
        # TODO: This is build time sensitive, O(tactic_num * impl_num * num_profile * tunable_ops)
        # Consider apply a preprofiling to estimate the kernel execution time, then decide the necessity.
        delay_kernel(self.stream_delay_micro_secs, stream)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream=stream)
        for _ in range(self.repeat):
            runner(inputs, tactic=tactic, **kwargs)
        end.record(stream=stream)
        stream.synchronize()

        avg_time = start.elapsed_time(end) / self.repeat
        logger.debug(
            f"[Autotuner]: profiling {runner} {tactic}, shapes={[t.size() for t in inputs]}, avg_time {avg_time}"
        )

        return avg_time

    def _find_nearest_profile(
            self, dynamic_tensors: Dict, constraints: Dict,
            inputs: List[torch.Tensor]) -> OptimizationProfile:
        """Find the nearest optimization profile for given inputs.

        Args:
            dynamic_tensors (Dict): Dictionary specifying tunable dimensions
            constraints (Dict): Dictionary specifying constraints on the dimensions
            inputs (List[torch.Tensor]): Input tensors to find profile for

        Returns:
            OptimizationProfile: Profile with dimensions rounded to nearest valid values

        Note:
            This method uses the rounding rules specified in dynamic_tensors to
            find valid dimensions closest to the actual input dimensions.
        """
        base_profile = OptimizationProfile([[StaticDim(x) for x in t.size()]
                                            for t in inputs])
        for input_idx in dynamic_tensors:
            for dim_idx in dynamic_tensors[input_idx]:
                _, shape_round_rule = dynamic_tensors[input_idx][dim_idx]
                dim_val = base_profile.shapes[input_idx][dim_idx].val
                nearest_opt_shape = shape_round_rule(dim_val)
                base_profile.shapes[input_idx][dim_idx] = StaticDim(
                    nearest_opt_shape)

        # Adjust the profile to satisfy the constraints
        for input_idx in constraints:
            for dim_idx in constraints[input_idx]:
                constraint = constraints[input_idx][dim_idx]
                min_value = 0
                max_value = base_profile.shapes[input_idx][dim_idx].val
                base_profile.shapes[input_idx][dim_idx] = DynamicDim(
                    min_value, constraint(base_profile.get_opt_shapes()),
                    max_value)

        return base_profile

    def _optimization_profiles(
            self, dynamic_tensors: Dict, constraints: Dict,
            inputs: List[torch.Tensor]) -> List[OptimizationProfile]:
        """Generate optimization profiles for autotuning.

        Args:
            dynamic_tensors (Dict): Dictionary specifying which dimensions to tune
            constraints (Dict): Dictionary specifying constraints on the dimensions
            inputs (List[torch.Tensor]): List of input tensors

        Returns:
            List of OptimizationProfile objects representing different configurations

        Note:
            This method performs a cartesian product of all possible dimension
            combinations specified in dynamic_tensors.
        """
        assert all(isinstance(_, torch.Tensor)
                   for _ in inputs)  # all args must be tensors

        # every dimension created from the concrete input tensor shape
        # generate some dynamic dimension description based on the dynamic_tensors
        base_profile = OptimizationProfile([[StaticDim(x) for x in t.size()]
                                            for t in inputs])

        generated_profiles: List[OptimizationProfile] = []

        dynamic_dims = []

        for input_idx in dynamic_tensors:
            for dim_idx in dynamic_tensors[input_idx]:
                shape_generater, shape_round_rule = dynamic_tensors[input_idx][
                    dim_idx]
                assert inspect.isfunction(shape_generater) or isinstance(shape_generater, (list, tuple)), \
                    "The given dynamic dimension must provide a opt value generation function or a list of opt values"
                if inspect.isfunction(shape_generater):
                    opt_shapes = shape_generater(
                        base_profile.shapes[input_idx][dim_idx].val)
                else:
                    opt_shapes = shape_generater
                dynamic_dims.append((input_idx, dim_idx, opt_shapes))

        # grid search, do cartesian product for all the dynamic axis
        dim_grids = itertools.product(*[d[-1] for d in dynamic_dims])
        for opt_point in dim_grids:
            p = copy.deepcopy(base_profile)
            for pos, (input_idx, dim_idx, _) in enumerate(dynamic_dims):
                opt_value = opt_point[pos]
                #TODO: fix me, how to set the min and max?
                min_value = 0
                max_value = base_profile.shapes[input_idx][dim_idx].val
                p.shapes[input_idx][dim_idx] = DynamicDim(
                    min_value, opt_value, max_value)

            # Adjust the profile to satisfy the constraints
            for input_idx in constraints:
                for dim_idx in constraints[input_idx]:
                    constraint = constraints[input_idx][dim_idx]
                    min_value = 0
                    max_value = base_profile.shapes[input_idx][dim_idx].val
                    p.shapes[input_idx][dim_idx] = DynamicDim(
                        min_value, constraint(p.get_opt_shapes()), max_value)
            generated_profiles.append(p)
            logger.debug(f"[Autotuner]: generated profile: {p}")
        return generated_profiles

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
        for d in dims:
            if isinstance(d, StaticDim):
                shapes.append(d.val)
            else:
                # TODO: how to make sure the created Tensor has the min/max info
                assert isinstance(d, DynamicDim)
                shapes.append(d.opt)
        # TODO: FIXME, sometimes the content of the tensor can affect the performance, like MOE
        # One solution is to manituplate the tensor content to make it more like the real data
        # during the tuning process. This can by controlled in the preparation phase by the runner.
        return torch.zeros(shapes, dtype=dtype, device=device)

    def get_cache_key(
            self, custom_op: str, runner: TunableRunner,
            profile: OptimizationProfile) -> Tuple[str, str, Tuple, Tuple]:
        """Generate a unique cache key for the given custom operation, runner, inputs, and profile.

        Args:
            custom_op (str): Name of the custom operation
            runner (TunableRunner): Runner implementation
            profile (OptimizationProfile): Optimization profile

        Returns:
            Tuple[str, str, Tuple, Tuple]: A tuple containing:
                - custom_op: Operation name
                - runner_key: Runner class name
                - attribute_key: Tuple of runner attributes
                - profile_key: Profile hash key
        """
        # TODO: Eliminate the overhead of the cache key creation. env_key has not been added to the cache key yet.
        # NOTE: Attribute names affect the hash order
        attributes = {
            k: v
            for k, v in runner.__dict__.items()
            if not callable(v) and not k.startswith("_")
        }
        attribute_key = tuple(attributes[key]
                              for key in sorted(attributes.keys()))
        profile_key = profile.get_hash_key()
        runner_key = runner.__class__.__name__
        return (custom_op, runner_key, attribute_key, profile_key)

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()
