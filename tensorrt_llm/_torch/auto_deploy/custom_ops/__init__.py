# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom ops and make sure they are all registered."""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    __all__.append(module_name)
    importlib.import_module(f"{__name__}.{module_name}")

# Recursively import subpackages and modules so their side-effects (e.g.,
# op registrations) are applied even when nested in subdirectories.
for _, full_name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    __all__.append(full_name)
    importlib.import_module(full_name)
