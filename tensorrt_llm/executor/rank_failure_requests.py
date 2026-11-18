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
"""The error of the requests that a rank failure ends under Wide-EP fault tolerance."""

from .utils import RequestError


class RankFailureError(RequestError):
    """Error of a request that a Wide-EP rank failure ended, safe to retry.

    Args:
        message: The error message.
        resubmit: Whether running the request again can succeed before a replacement joins.
            False for a request that fails the same way until then.
    """

    def __init__(
        self,
        message: str = "A rank failure ended the request; sending it again can succeed",
        resubmit: bool = True,
    ):
        super().__init__(message)
        self.resubmit = resubmit
