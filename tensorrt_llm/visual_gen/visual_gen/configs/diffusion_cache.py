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

import logging

logger = logging.getLogger(__name__)


class TeaCacheConfig:
    """Configuration for TeaCache."""

    _instance = None
    _enable_teacache = False
    _teacache_thresh = 0.2  # "Higher speedup will cause to worse quality"
    _use_ret_steps = False  # "Using Retention Steps will result in faster generation speed and better generation quality."
    _ret_steps = None
    _cutoff_steps = None
    _num_steps = None
    _cnt = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TeaCacheConfig, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def check_status(cls) -> bool:
        """Check if the TeaCacheConfig is valid by checking if any attribute is None.

        Returns:
            bool: True if all attributes are set (not None), False otherwise.

        Raises:
            ValueError: If any required attribute is None, with details about which attributes are None.
        """
        none_attrs = [
            attr
            for attr in [
                "_enable_teacache",
                "_cnt",
                "_teacache_thresh",
                "_use_ret_steps",
                "_ret_steps",
                "_cutoff_steps",
                "_num_steps",
            ]
            if getattr(cls, attr) is None
        ]

        if none_attrs:
            logger.warning(
                f"The following TeaCacheConfig attributes are None: {', '.join(none_attrs)}"
            )
            return False

        return True

    @classmethod
    def set_config(
        cls,
        enable_teacache: bool = None,
        teacache_thresh: float = None,
        use_ret_steps: bool = None,
        ret_steps: int = None,
        cutoff_steps: int = None,
        num_steps: int = None,
        cnt: int = None,
    ):
        """Set TeaCache configuration parameters."""
        if enable_teacache is not None:
            cls._enable_teacache = enable_teacache
        if cnt is not None:
            cls._cnt = cnt
        if teacache_thresh is not None:
            cls._teacache_thresh = teacache_thresh
        if use_ret_steps is not None:
            cls._use_ret_steps = use_ret_steps
        if ret_steps is not None:
            cls._ret_steps = ret_steps
        if cutoff_steps is not None:
            cls._cutoff_steps = cutoff_steps
        if num_steps is not None:
            cls._num_steps = num_steps

    @classmethod
    def reset(cls):
        """Reset all TeaCache state variables."""
        cls._cnt = 0

    # Property-style access for attributes
    @classmethod
    def enable_teacache(cls) -> bool:
        return cls._enable_teacache

    @classmethod
    def teacache_thresh(cls) -> float:
        return cls._teacache_thresh

    @classmethod
    def use_ret_steps(cls) -> bool:
        return cls._use_ret_steps

    @classmethod
    def ret_steps(cls) -> int:
        return cls._ret_steps

    @classmethod
    def cutoff_steps(cls) -> int:
        return cls._cutoff_steps

    @classmethod
    def num_steps(cls) -> int:
        return cls._num_steps

    @classmethod
    def cnt(cls) -> int:
        return cls._cnt

    @classmethod
    def increment_cnt(cls, step: int = 1) -> None:
        """Increment cnt by specified step and return the new value.

        Args:
            step (int): The increment size. Defaults to 1.
        """
        cls._cnt += step

    @classmethod
    def set_cnt(cls, cnt: int) -> None:
        """Set cnt to a specific value."""
        cls._cnt = cnt

    @classmethod
    def to_dict(cls) -> dict:
        """Convert TeaCacheConfig to a dictionary."""
        return {
            "enable_teacache": cls._enable_teacache,
            "teacache_thresh": cls._teacache_thresh,
            "use_ret_steps": cls._use_ret_steps,
            "ret_steps": cls._ret_steps,
            "cutoff_steps": cls._cutoff_steps,
        }
