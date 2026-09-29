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
"""Everything selection does that is not pytest: stdlib only, no pytest import.

These modules take plain values and raise `SelectionError`; turning that into a
`pytest.UsageError` is `plugin.py`'s job, and `plugin.py` is the only module in
the package that imports pytest. So every decision here can be exercised
without pytest.

`qa_selection/__init__.py` lists the modules in import order; each one imports
only the ones above it.
"""
