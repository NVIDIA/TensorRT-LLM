#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def parse_positive_concurrency(value: object) -> int:
    """Parse a positive benchmark concurrency from YAML configuration.

    Args:
        value: Integer or numeric string from ``benchmark.concurrency_list``.

    Returns:
        The parsed positive integer.

    Raises:
        ValueError: If ``value`` is not an integer or is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"benchmark.concurrency_list must be a positive integer, got {value!r}")

    try:
        concurrency = int(value)
    except ValueError as error:
        raise ValueError(
            f"benchmark.concurrency_list must be a positive integer, got {value!r}"
        ) from error

    if concurrency <= 0:
        raise ValueError(f"benchmark.concurrency_list must be a positive integer, got {value!r}")
    return concurrency
