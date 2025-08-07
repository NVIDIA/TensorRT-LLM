# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from contextlib import contextmanager
from typing import Any, Callable, Optional


class TimeoutManager:
    """
    A utility class for managing timeout in test cases.

    This class helps reduce boilerplate code for timeout handling in test cases
    by providing a simple interface to track remaining time and execute operations
    with automatic timeout checking.
    """

    def __init__(self, initial_timeout: Optional[float] = None):
        """
        Initialize the timeout manager.

        Args:
            initial_timeout: Initial timeout value in seconds. If None, no timeout is enforced.
        """
        self._initial_timeout = initial_timeout
        self._remaining_timeout = initial_timeout
        self._start_time = None

    @property
    def remaining_timeout(self) -> Optional[float]:
        """Get the remaining timeout value."""
        return self._remaining_timeout

    def reset(self, timeout: Optional[float] = None) -> None:
        """
        Reset the timeout manager with a new timeout value.

        Args:
            timeout: New timeout value. If None, uses the initial timeout.
        """
        self._remaining_timeout = timeout if timeout is not None else self._initial_timeout
        self._start_time = None

    def check_timeout(self, phase_name: str = "operation") -> None:
        """
        Check if timeout has been exceeded and raise TimeoutError if so.

        Args:
            phase_name: Name of the current phase for error message.

        Raises:
            TimeoutError: If timeout has been exceeded.
        """
        if self._remaining_timeout is not None and self._remaining_timeout <= 0:
            raise TimeoutError(f"Timeout exceeded after {phase_name} phase!")

    @contextmanager
    def timed_operation(self, phase_name: str = "operation"):
        """
        Context manager for timing an operation and updating remaining timeout.

        Args:
            phase_name: Name of the phase for timeout checking.

        Yields:
            None

        Raises:
            TimeoutError: If timeout is exceeded after the operation.
        """
        if self._remaining_timeout is None:
            # No timeout enforcement
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            operation_time = time.time() - start_time
            self._remaining_timeout -= operation_time
            self.check_timeout(phase_name)

    def execute_with_timeout(self,
                             operation: Callable[[], Any],
                             phase_name: str = "operation",
                             **kwargs) -> Any:
        """
        Execute an operation with timeout tracking.

        Args:
            operation: The operation to execute.
            phase_name: Name of the phase for timeout checking.
            **kwargs: Additional arguments to pass to the operation.

        Returns:
            The result of the operation.

        Raises:
            TimeoutError: If timeout is exceeded after the operation.
        """
        with self.timed_operation(phase_name):
            return operation(**kwargs)

    def call_with_timeout(self,
                          func: Callable,
                          *args,
                          phase_name: str = "operation",
                          **kwargs) -> Any:
        """
        Call a function with timeout tracking.

        Args:
            func: The function to call.
            *args: Positional arguments for the function.
            phase_name: Name of the phase for timeout checking.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            TimeoutError: If timeout is exceeded after the function call.
        """
        with self.timed_operation(phase_name):
            return func(*args, **kwargs)


def create_timeout_manager(
        timeout_from_marker: Optional[float] = None) -> TimeoutManager:
    """
    Create a TimeoutManager instance from a timeout marker value.

    Args:
        timeout_from_marker: Timeout value from pytest marker.

    Returns:
        A TimeoutManager instance.
    """
    return TimeoutManager(timeout_from_marker)


# Convenience decorator for test functions
def with_timeout_management(func: Callable) -> Callable:
    """
    Decorator to automatically inject timeout management into test functions.

    This decorator expects the test function to have a 'timeout_from_marker' parameter
    and automatically creates a TimeoutManager instance.

    Args:
        func: The test function to decorate.

    Returns:
        The decorated function.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract timeout_from_marker from kwargs
        timeout_from_marker = kwargs.get('timeout_from_marker')

        # Create timeout manager
        timeout_manager = create_timeout_manager(timeout_from_marker)

        # Add timeout_manager to kwargs
        kwargs['timeout_manager'] = timeout_manager

        return func(*args, **kwargs)

    return wrapper
