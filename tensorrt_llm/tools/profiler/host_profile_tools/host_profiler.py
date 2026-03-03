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
"""Host-side profiler for analyzing CPU overhead in the PyExecutor.

This module provides a HostProfiler class that wraps line_profiler to measure
line-by-line execution time of critical functions in the executor worker thread.

Usage:
    Set environment variable TLLM_LINE_PROFILER_PATH to enable:
        TLLM_LINE_PROFILER_PATH=./lp_results.txt pytest ...

    Or use programmatically:
        profiler = HostProfiler(output_path="./results.txt")
        profiler.start()
        # ... run code ...
        profiler.stop()
"""

import importlib
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from tensorrt_llm.logger import logger

# Environment variable to enable line_profiler output path.
LINE_PROFILER_PATH_ENV_VAR = "TLLM_LINE_PROFILER_PATH"

# Environment variable to specify additional functions to profile (comma-separated).
# Format: "module.Class.method,module.Class.method2,..."
LINE_PROFILER_FUNCTIONS_ENV_VAR = "TLLM_LINE_PROFILER_FUNCTIONS"


@dataclass
class ProfileTarget:
    """Represents a function to be profiled.

    Supports both class methods and standalone module-level functions.

    Examples:
        Class method:
            ProfileTarget("module.path", "ClassName", "method_name")
            -> resolves to module.path.ClassName.method_name

        Standalone function:
            ProfileTarget("module.path", None, "function_name")
            -> resolves to module.path.function_name
    """

    module_path: str
    class_name: Optional[str]  # None for standalone functions
    method_name: str

    @property
    def full_path(self) -> str:
        if self.class_name is None:
            return f"{self.module_path}.{self.method_name}"
        return f"{self.module_path}.{self.class_name}.{self.method_name}"

    @property
    def is_standalone(self) -> bool:
        """Returns True if this is a standalone function (not a class method)."""
        return self.class_name is None

    def resolve(self) -> Optional[Callable]:
        """Resolve the target to an actual function object.

        Returns:
            The unwrapped method/function (inner function), or None if resolution fails.

        Note:
            We MUST unwrap decorated functions (e.g., @torch.inference_mode,
            @nvtx_range) because line_profiler traces by __code__ identity.
            When a function is wrapped by decorators like @torch.inference_mode(),
            the wrapper's __code__ is different from the actual function's __code__.
            The wrapper only contains a few lines that call the inner function,
            so line_profiler would only see those wrapper lines, not the actual
            function body we want to profile.

            By unwrapping, we get the actual function's __code__ which allows
            line_profiler to trace the real function lines.
        """
        try:
            module = importlib.import_module(self.module_path)

            if self.is_standalone:
                # Standalone module-level function
                func = getattr(module, self.method_name)
            else:
                # Class method
                cls = getattr(module, self.class_name)
                func = getattr(cls, self.method_name)

            # Unwrap decorated functions to get the actual inner function.
            # This is necessary for @torch.inference_mode(), @nvtx_range(), etc.
            # Without unwrapping, line_profiler would only trace the wrapper's
            # __code__, not the actual function body.
            while hasattr(func, "__wrapped__"):
                func = func.__wrapped__

            return func
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to resolve profile target {self.full_path}: {e}")
            return None


# Default functions to profile for host overhead analysis
# Hierarchical config: {module_path: {class_name: [method_names]}}
# Use None as class_name for standalone module-level functions
#
# Wildcard support:
#   - ["*"] as method list: Profile all methods of the class
#   - {None: ["*"]}: Profile all standalone functions in the module
#   - {"*": ["*"]}: Profile all classes and all their methods in the module
_PYEXEC = "tensorrt_llm._torch.pyexecutor"
_DEFAULT_PROFILE_CONFIG: Dict[str, Dict[Optional[str], List[str]]] = {
    f"{_PYEXEC}.py_executor": {
        "PyExecutor": [
            "_prepare_and_schedule_batch",
            "_schedule",
            "_forward_step",
            "_sample_async",
            "_update_requests",
            "_update_request_states",
            "_fetch_and_activate_new_requests",
            "_handle_responses",
            "_handle_canceled_requests",
            "_enqueue_responses",
        ],
    },
    f"{_PYEXEC}.sampler": {
        "TorchSampler": [
            "sample_async",
            "update_requests",
            "_process_requests",
            "_write_finish_reasons",
            "_prepare_beam_search",
            "_select_generated_logits",
            "_sample_batched_by_strategy",
        ],
        # Standalone module-level functions (use None as class_name)
        None: [
            "_group_requests_by_strategy_key",
        ],
    },
    f"{_PYEXEC}.resource_manager": {
        "ResourceManager": ["prepare_resources", "update_resources", "free_resources"],
        "KVCacheManager": ["prepare_resources", "update_resources"],
    },
    f"{_PYEXEC}.scheduler": {
        "RequestScheduler": ["schedule_request"],
    },
    f"{_PYEXEC}.executor_request_queue": {
        "ExecutorRequestQueue": [
            "_fetch_new_requests_attention_tp",
            "_fetch_new_requests_attention_dp",
            "_fetch_and_process_requests",
            "_merge_requests",
            "fetch_new_requests",
        ],
    },
}


def _get_all_methods_from_class(cls: type) -> List[str]:
    """Get all user-defined methods from a class (excluding inherited, dunder, nested classes, and properties).

    Only includes items that have actual executable code (__code__ attribute):
    - Regular instance methods (def foo(self): ...)
    - Static methods (@staticmethod)
    - Class methods (@classmethod)

    Excludes:
    - Nested classes (e.g., dataclasses like Args, Store)
    - Properties (usually trivial getters, complex to profile)
    - Constants and type aliases
    - Dunder methods (__init__, __repr__, etc.)

    Args:
        cls: The class to introspect.

    Returns:
        List of method names defined directly on the class.
    """
    import inspect

    methods = []
    for name, member in cls.__dict__.items():
        # Skip dunder methods
        if name.startswith("__") and name.endswith("__"):
            continue

        # Skip nested classes
        if inspect.isclass(member):
            continue

        # Skip properties (usually trivial, complex to handle with line_profiler)
        if isinstance(member, property):
            continue

        # Extract the underlying function from descriptors
        func = None
        if isinstance(member, staticmethod):
            func = member.__func__
        elif isinstance(member, classmethod):
            func = member.__func__
        elif inspect.isfunction(member):
            func = member

        # Only include if we have a valid function with executable code
        if func is not None and hasattr(func, "__code__"):
            methods.append(name)

    return methods


def _get_all_functions_from_module(module_path: str) -> List[str]:
    """Get all user-defined functions from a module (excluding imported ones).

    Args:
        module_path: The module path to introspect.

    Returns:
        List of function names defined directly in the module.
    """
    import inspect

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Failed to import module {module_path} for introspection: {e}")
        return []

    functions = []
    for name, member in inspect.getmembers(module, inspect.isfunction):
        # Only include functions defined in this module (not imported)
        if member.__module__ == module_path:
            # Skip private functions starting with underscore if desired
            # For profiling, we likely want to include them
            functions.append(name)
    return functions


def _get_all_classes_from_module(module_path: str) -> List[str]:
    """Get all user-defined classes from a module (excluding imported ones).

    Args:
        module_path: The module path to introspect.

    Returns:
        List of class names defined directly in the module.
    """
    import inspect

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Failed to import module {module_path} for introspection: {e}")
        return []

    classes = []
    for name, member in inspect.getmembers(module, inspect.isclass):
        # Only include classes defined in this module (not imported)
        if member.__module__ == module_path:
            classes.append(name)
    return classes


def _expand_profile_config(
    config: Dict[str, Dict[Optional[str], List[str]]],
) -> List[ProfileTarget]:
    """Expand hierarchical config into a flat list of ProfileTarget objects.

    Args:
        config: Hierarchical config mapping module_path -> class_name -> method_names.
                Use None as class_name for standalone module-level functions.

                Special markers:
                - "*" in method list: Profile all methods of the class
                - "*" as class_name key with ["*"]: Profile all classes and their methods
                - None as class_name key with ["*"]: Profile all standalone functions

    Returns:
        List of ProfileTarget objects.

    Examples:
        # Profile all methods of TorchSampler class:
        {"module.sampler": {"TorchSampler": ["*"]}}

        # Profile all standalone functions in a module:
        {"module.sampler": {None: ["*"]}}

        # Profile all classes and all their methods in a module:
        {"module.sampler": {"*": ["*"]}}

        # Mix explicit and wildcard:
        {"module.sampler": {"TorchSampler": ["*"], "OtherClass": ["specific_method"]}}
    """
    targets = []
    for module_path, classes in config.items():
        for class_name, methods in classes.items():
            # Handle wildcard class_name: "*" means all classes in the module
            if class_name == "*":
                if methods == ["*"]:
                    # Profile all methods of all classes in the module
                    all_classes = _get_all_classes_from_module(module_path)
                    for cls_name in all_classes:
                        try:
                            module = importlib.import_module(module_path)
                            cls = getattr(module, cls_name)
                            all_methods = _get_all_methods_from_class(cls)
                            for method_name in all_methods:
                                targets.append(ProfileTarget(module_path, cls_name, method_name))
                        except (ImportError, AttributeError) as e:
                            logger.warning(f"Failed to introspect {module_path}.{cls_name}: {e}")
                continue

            # Handle wildcard methods: ["*"] means all methods of the class/module
            if methods == ["*"]:
                if class_name is None:
                    # All standalone functions in the module
                    all_funcs = _get_all_functions_from_module(module_path)
                    for func_name in all_funcs:
                        targets.append(ProfileTarget(module_path, None, func_name))
                else:
                    # All methods of a specific class
                    try:
                        module = importlib.import_module(module_path)
                        cls = getattr(module, class_name)
                        all_methods = _get_all_methods_from_class(cls)
                        for method_name in all_methods:
                            targets.append(ProfileTarget(module_path, class_name, method_name))
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Failed to introspect {module_path}.{class_name}: {e}")
                continue

            # Normal case: explicit method list
            for method_name in methods:
                targets.append(ProfileTarget(module_path, class_name, method_name))
    return targets


DEFAULT_PROFILE_TARGETS: List[ProfileTarget] = _expand_profile_config(_DEFAULT_PROFILE_CONFIG)


class HostProfiler:
    """Host-side profiler for measuring CPU overhead in the executor.

    This class wraps line_profiler to provide line-by-line timing analysis
    of critical functions in the PyExecutor worker thread.

    Attributes:
        output_path: Path to save profiling results.
        targets: List of ProfileTarget objects specifying functions to profile.
        enabled: Whether profiling is currently active.

    Example:
        >>> profiler = HostProfiler(output_path="./results.txt")
        >>> profiler.add_target(
        ...     ProfileTarget(
        ...         module_path="my_module",
        ...         class_name="MyClass",
        ...         method_name="my_method",
        ...     )
        ... )
        >>> profiler.start()
        >>> # ... run code ...
        >>> profiler.stop()
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        targets: Optional[List[ProfileTarget]] = None,
        use_defaults: bool = True,
    ):
        """Initialize the host profiler.

        Args:
            output_path: Path to save results. If None, uses env var TLLM_LINE_PROFILER_PATH.
            targets: List of ProfileTarget objects. If None and use_defaults=True,
                     uses DEFAULT_PROFILE_TARGETS.
            use_defaults: Whether to include default profile targets.
        """
        self.output_path = output_path or os.environ.get(LINE_PROFILER_PATH_ENV_VAR)
        self.targets: List[ProfileTarget] = []
        self._line_profiler = None
        self._enabled = False

        # Add default targets if requested
        if use_defaults:
            self.targets.extend(DEFAULT_PROFILE_TARGETS)

        # Add custom targets
        if targets:
            self.targets.extend(targets)

        # Parse additional targets from environment variable
        self._parse_env_targets()

    def _parse_env_targets(self) -> None:
        """Parse additional profile targets from environment variable.

        Supported formats:
            - module.Class.method  -> class method
            - module::function     -> standalone function (uses :: as delimiter)
        """
        env_funcs = os.environ.get(LINE_PROFILER_FUNCTIONS_ENV_VAR, "")
        if not env_funcs:
            return

        for func_path in env_funcs.split(","):
            func_path = func_path.strip()
            if not func_path:
                continue

            # Check for standalone function format: "module::function"
            if "::" in func_path:
                parts = func_path.rsplit("::", 1)
                if len(parts) != 2:
                    logger.warning(
                        f"Invalid standalone function path '{func_path}'. "
                        "Expected format: module.path::function_name"
                    )
                    continue
                module_path, method_name = parts
                self.targets.append(
                    ProfileTarget(
                        module_path=module_path,
                        class_name=None,  # Standalone function
                        method_name=method_name,
                    )
                )
            else:
                # Class method format: "module.Class.method"
                parts = func_path.rsplit(".", 2)
                if len(parts) < 3:
                    logger.warning(
                        f"Invalid function path '{func_path}'. Expected format: "
                        "module.Class.method (class method) or module::function (standalone)"
                    )
                    continue

                # Handle nested module paths
                method_name = parts[-1]
                class_name = parts[-2]
                module_path = ".".join(parts[:-2])

                self.targets.append(
                    ProfileTarget(
                        module_path=module_path,
                        class_name=class_name,
                        method_name=method_name,
                    )
                )

    def add_target(self, target: ProfileTarget) -> "HostProfiler":
        """Add a profile target.

        Args:
            target: The ProfileTarget to add.

        Returns:
            Self for chaining.
        """
        self.targets.append(target)
        return self

    def add_function(
        self,
        module_path: str,
        class_name: Optional[str],
        method_name: str,
    ) -> "HostProfiler":
        """Add a function to profile by specifying its path.

        Args:
            module_path: The module path (e.g., "tensorrt_llm._torch.pyexecutor.sampler")
            class_name: The class name (e.g., "TorchSampler"), or None for standalone functions
            method_name: The method/function name (e.g., "_process_requests")

        Returns:
            Self for chaining.

        Examples:
            # Add a class method
            profiler.add_function("my.module", "MyClass", "my_method")

            # Add a standalone function
            profiler.add_function("my.module", None, "my_function")
        """
        return self.add_target(
            ProfileTarget(
                module_path=module_path,
                class_name=class_name,
                method_name=method_name,
            )
        )

    def add_standalone_function(
        self,
        module_path: str,
        function_name: str,
    ) -> "HostProfiler":
        """Add a standalone module-level function to profile.

        This is a convenience method for adding standalone functions
        (not class methods).

        Args:
            module_path: The module path (e.g., "tensorrt_llm._torch.pyexecutor.sampler")
            function_name: The function name (e.g., "_group_requests_by_strategy_key")

        Returns:
            Self for chaining.
        """
        return self.add_target(
            ProfileTarget(
                module_path=module_path,
                class_name=None,
                method_name=function_name,
            )
        )

    def clear_targets(self) -> "HostProfiler":
        """Clear all profile targets (including defaults).

        This is useful when you want to start with a clean slate and add
        only specific targets, or when you want to replace all default
        targets with a custom set.

        Returns:
            Self for chaining.

        Example:
            # Clear defaults and add only specific targets
            profiler = HostProfiler(use_defaults=True)
            profiler.clear_targets().add_function("my.module", "MyClass", "method")

            # Or start fresh without defaults
            profiler = HostProfiler(use_defaults=False)
        """
        self.targets.clear()
        return self

    @property
    def is_available(self) -> bool:
        """Check if line_profiler is available."""
        return True

    @property
    def should_profile(self) -> bool:
        """Check if profiling should be enabled (output path is set)."""
        return self.output_path is not None

    @property
    def enabled(self) -> bool:
        """Check if profiling is currently active."""
        return self._enabled

    def start(self) -> bool:
        """Start profiling.

        Returns:
            True if profiling started successfully, False otherwise.
        """
        if not self.should_profile:
            logger.info("Line profiler not enabled (no output path specified)")
            return False

        if not self.is_available:
            logger.warning("line_profiler not installed. Install with: pip install line_profiler")
            return False

        if self._enabled:
            logger.warning("Line profiler already started")
            return True

        try:
            from line_profiler import LineProfiler

            self._line_profiler = LineProfiler()

            # Add all target functions
            resolved_count = 0
            for target in self.targets:
                func = target.resolve()
                if func is not None:
                    logger.info(
                        f"line profiler func code ID: {id(func.__code__)}, target: {target.full_path}"
                    )
                    self._line_profiler.add_function(func)
                    resolved_count += 1

            if resolved_count == 0:
                logger.warning("No profile targets could be resolved")
                self._line_profiler = None
                return False

            self._line_profiler.enable()
            self._enabled = True
            self._profiler_thread_id = threading.current_thread().ident
            logger.info(
                f"Line profiler enabled with {resolved_count}/{len(self.targets)} targets. "
                f"Thread ID: {self._profiler_thread_id}, Thread name: {threading.current_thread().name}. "
                f"Results will be saved to: {self.output_path}"
            )
            dump_profiler_functions()
            return True

        except Exception as e:
            logger.error(f"Failed to start line profiler: {e}")
            self._line_profiler = None
            return False

    def stop(self) -> bool:
        """Stop profiling and save results.

        Returns:
            True if results were saved successfully, False otherwise.
        """
        if not self._enabled or self._line_profiler is None:
            return False

        try:
            self._line_profiler.disable()
            self._enabled = False

            # Save results
            with open(self.output_path, "w") as f:
                self._line_profiler.print_stats(stream=f)

            logger.info(f"Line profiler results saved to: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save line profiler results: {e}")
            return False

        finally:
            self._line_profiler = None

    @contextmanager
    def profile(self):
        """Context manager for profiling.

        Usage:
            with profiler.profile():
                # ... code to profile ...
        """
        started = self.start()
        try:
            yield self
        finally:
            if started:
                self.stop()

    def get_stats_string(self) -> Optional[str]:
        """Get profiling stats as a string (without saving to file).

        Returns:
            Stats string if profiling is active, None otherwise.
        """
        if self._line_profiler is None:
            return None

        import io

        stream = io.StringIO()
        self._line_profiler.print_stats(stream=stream)
        return stream.getvalue()

    def list_targets(self) -> List[str]:
        """List all configured profile targets.

        Returns:
            List of target paths.
        """
        return [t.full_path for t in self.targets]


# Global profiler instance for use in worker thread
_global_profiler: Optional[HostProfiler] = None


def get_global_profiler() -> Optional[HostProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def dump_profiler_functions() -> None:
    """Print all functions registered with the line profiler for debugging.

    Only dumps on rank 0 to avoid interleaved output from multiple ranks.
    """
    # Import here to avoid circular imports and handle cases where MPI is not initialized
    try:
        from tensorrt_llm._utils import mpi_rank

        if mpi_rank() != 0:
            return
    except Exception:
        pass  # If MPI is not available, proceed (single-rank case)

    profiler = get_global_profiler()
    if profiler is None or profiler._line_profiler is None:
        logger.info("No line profiler active")
        return

    lp = profiler._line_profiler
    logger.info(f"=== Line Profiler State: {len(lp.functions)} functions registered ===")
    for func in lp.functions:
        logger.info(f"  {func.__module__}.{func.__qualname__}, code id: {id(func.__code__)}")
    logger.info("=== End Line Profiler State ===")


def set_global_profiler(profiler: Optional[HostProfiler]) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


@contextmanager
def host_profiler_context(enable: bool = True, output_path: Optional[str] = None):
    """Context manager for host profiling in the worker thread.

    This is the main entry point for profiling in PyExecutor._event_loop_wrapper.

    Args:
        output_path: Path to save results. If None, uses env var.

    Usage:
        with host_profiler_context():
            # ... event loop code ...
    """
    if not enable:
        yield None
        return

    profiler = HostProfiler(output_path=output_path)
    set_global_profiler(profiler)

    started = profiler.start()
    try:
        yield profiler
    finally:
        if started:
            profiler.stop()
        set_global_profiler(None)
