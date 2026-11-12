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
"""``persistent_experts`` with a kernel-option override against the default options, on a GPU.

Every tactic field only reorders or retimes work: each output tile keeps one accumulator over K
ascending (see ``persistent_experts.TACTIC_RANGES``). So the rows of a launch built with the
``MUSESPARK_OPEN_PERSISTENT_EXPERTS_OPTIONS`` override must equal, bit for bit, the rows of the
launch built with the default options on the same inputs: ``torch.equal``, no tolerance.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    persistent_experts as PE,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="persistent_experts is built for sm_103a",
)

ROWS = 8
WINDOW = 32


def _inputs(device):
    g = torch.Generator(device=device).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device, generator=g)

    def scales(*shape):  # E4M3 bytes 0x30..0x40: positive, about 0.5..2, never NaN
        return torch.randint(0x30, 0x41, shape, dtype=torch.uint8, device=device, generator=g)

    ids = torch.stack(
        [torch.randperm(WINDOW, device=device, generator=g)[: PE.TOP_K] for _ in range(ROWS)]
    ).to(torch.int32)  # every slot local, no repeats
    scalar = torch.full((WINDOW,), 1e-2, dtype=torch.float32, device=device)
    return (
        codes(ROWS, PE.HIDDEN // 2),
        scales(ROWS * PE.HIDDEN // PE.SF_VEC),
        ids,
        torch.rand(ROWS, PE.TOP_K, dtype=torch.bfloat16, device=device),
        codes(WINDOW, 2 * PE.INTER, PE.HIDDEN // 2),
        scales(WINDOW * 2 * PE.INTER * PE.HIDDEN // PE.SF_VEC),
        codes(WINDOW, PE.HIDDEN, PE.INTER // 2),
        scales(WINDOW * PE.HIDDEN * PE.INTER // PE.SF_VEC),
        scalar,
        scalar.clone(),
        scalar.clone(),
    )


def _run(monkeypatch, override, args):
    # The environment is read once per device and cached: start each launch from an empty cache.
    monkeypatch.setattr(PE, "_DEVICE_OPTIONS", {})
    if override is None:
        monkeypatch.delenv(PE.OPTIONS_ENV, raising=False)
    else:
        monkeypatch.setenv(PE.OPTIONS_ENV, override)
    options = PE._device_options(args[0].device)
    rows, _, index = PE.persistent_experts(*args)
    torch.cuda.synchronize()
    return options, rows, index


@pytest.mark.parametrize("override", ["order=0,balance=0", "num_stages=3,acc_stages=2"])
def test_override_rows_are_bitwise_equal_to_default(monkeypatch, override):
    args = _inputs(torch.device("cuda"))
    default_options, default_rows, default_index = _run(monkeypatch, None, args)
    options, rows, index = _run(monkeypatch, override, args)

    # The override took effect (otherwise the comparison below is the default against itself).
    changed = dict(item.split("=") for item in override.split(","))
    for key, value in changed.items():
        assert options[key] == int(value) != default_options[key], key

    assert torch.isfinite(default_rows.float()).all()
    assert default_rows.float().abs().max() > 0  # a zero or unwritten output would pass equal
    assert torch.equal(index, default_index)
    assert torch.equal(rows, default_rows)
