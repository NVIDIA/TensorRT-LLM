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
"""Put the modeling_v2 switch in front of the ranks that resolve the model.

Shared by the whole-model gates rather than copied into each of them: what
follows is easy to get subtly wrong, and two copies would drift.

``TRTLLM_MODELING_V2`` is read by ``modeling_v2_resolve``, called from
``AutoModelForCausalLM._resolve_class`` -- which runs in a worker rank, not in
the process running the test. Assigning to ``os.environ`` here does not
reliably reach that rank: MPI caches the environment at import time and spawns
from that snapshot, which is why ``worker_main`` re-applies ``env_overrides``
by hand at its top. So the switch travels as an ``LLM(...)`` field rather than
as a shell export, and it then reaches both sides -- ``LLM.__init__`` applies
the same overrides to the calling process before it checks the mode itself.

These cases used to read the variable instead and skip unless the launching
shell had exported it. Nothing in CI exports it, so all of them skipped and the
stage reported green without ever measuring a modeling_v2 target -- the same
misattribution ``require`` mode exists to prevent, reached through the test
harness rather than through the model.
"""

from typing import Any, Dict

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_ENV


def modeling_v2_llm_args(
    mode: str, monkeypatch: pytest.MonkeyPatch, **extra: str
) -> Dict[str, Any]:
    """``LLM(...)`` kwargs that build the model under ``TRTLLM_MODELING_V2=mode``.

    ``extra`` carries any further environment switch the case needs the ranks
    to see; it travels by the same route.

    ``monkeypatch`` is not decoration. ``LLM`` applies its overrides to the
    calling process as well and never puts them back, so with no teardown the
    first case asking for ``"require"`` would leave every later case in the
    session asking for it too -- and ``require`` raises on an architecture with
    no target, so unrelated tests downstream would fail with this file's name
    nowhere in the traceback.
    """
    overrides = {MODELING_V2_ENV: mode, **extra}
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    return {"env_overrides": overrides}
