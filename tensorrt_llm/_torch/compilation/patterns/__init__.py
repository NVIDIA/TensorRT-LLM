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

from inspect import signature
from typing import Callable

from torch._inductor.pattern_matcher import (Match, PatternExpr,
                                             PatternMatcherPass,
                                             register_graph_pattern)

MATCHER_SUBSYSTEM = "torch_compile"


def _register_pattern_replacement(
    pattern: PatternExpr,
    replacement: Callable[..., object],
    custom_pass: PatternMatcherPass,
    *,
    extra_check: Callable[[Match], bool] = lambda match: True,
) -> None:
    """Replace an explicit FX pattern using its named captures.

    Graph patterns already describe the search operations. Register them
    directly so the matcher traces only the replacement, using the matched
    inputs' shapes and dtypes.
    """
    argnames = tuple(signature(replacement).parameters)

    @register_graph_pattern(pattern, extra_check, pass_dict=custom_pass)
    def replace(match: Match, *args, **kwargs):
        match.replace_by_example(replacement,
                                 [kwargs[name] for name in argnames])
