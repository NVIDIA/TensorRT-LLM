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

# This import inexplicably starts a thread!
# This causes problems for our test infra. The issue is that TRTLLM will import
# this module. If the import happens before the test starts, there are no problems.
# But if the import happens lazily after the test starts, pytest will think you leaked
# the thread. We thus do the import here to prevent thread leak issues cropping up when messing
# with the import statements in tests.
from torch._inductor import lowering  # NOQA
