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
"""Fake VisualGen generator for CPU tests of the VISUAL_GEN serving role.

Kept in its own module so that only tests that build a VisualGen app import the
``tensorrt_llm.visual_gen`` tree: ``OpenAIServer`` detects VisualGen generators
by probing ``sys.modules`` for ``tensorrt_llm.visual_gen.visual_gen``, so
importing this module is what makes ``_is_visual_gen_instance`` able to say yes.
"""

import itertools
from typing import Any

from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.visual_gen import VisualGen

__all__ = ["FakeVisualGen"]


class FakeVisualGen(VisualGen):
    """VisualGen generator fake; subclassing keeps ``isinstance`` checks true."""

    def __init__(self, model: str = "fake-visual-gen") -> None:
        # Never run VisualGen.__init__ (it loads a pipeline). VisualGen.shutdown,
        # which __del__ calls, returns early when executor is None.
        self.model = model
        self.args = VisualGenArgs()
        self.executor = None
        self._req_counter = itertools.count()

    def generate_async(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("FakeVisualGen does not generate.")
