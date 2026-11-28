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
"""The errors a streaming session raises to its caller."""

__all__ = [
    "SessionEnded",
    "SessionError",
    "SessionLeaseExpired",
    "SessionQueueFull",
]


class SessionError(RuntimeError):
    """Base class for session failures raised to the client."""


class SessionQueueFull(SessionError):
    """A queued input has no room; retry after the next advance."""


class SessionEnded(SessionError):
    """The session has a terminal result or is closing; no more input is admitted."""


class SessionLeaseExpired(SessionError):
    """A result's storage was returned to the worker; the view is no longer valid."""
