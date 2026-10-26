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
"""CPU tests of ``DenseConfig``, the dense DWDP settings setup and the runtime read.

``DwdpConfig.enable_dense`` is the only public setting; the other fields keep the values every
launch uses and are set by tests only.
"""

import pytest

from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.types import DEFAULT_MIN_BYTES, SegmentKind
from tensorrt_llm.llmapi.llm_args import DwdpConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "dwdp_config,enabled",
    [(None, False), (DwdpConfig(dwdp_size=2), False), (DwdpConfig(enable_dense=True), True)],
    ids=["no-dwdp-config", "expert-only", "dense"],
)
def test_a_launch_sets_only_whether_dense_dwdp_is_on(
    dwdp_config: DwdpConfig | None, enabled: bool
) -> None:
    """``from_dwdp_config`` takes ``enable_dense``; every other field keeps its default."""
    assert DenseConfig.from_dwdp_config(dwdp_config) == DenseConfig(enabled=enabled)
    config = DenseConfig.from_dwdp_config(dwdp_config)
    assert config.segments == (SegmentKind.ATTN, SegmentKind.MOE)
    assert (config.min_bytes, config.store_chunk_mb, config.strict) == (DEFAULT_MIN_BYTES, 0, False)


def test_segments_are_canonical_and_start_with_attention() -> None:
    """Strings are converted and put in canonical order; repeats collapse."""
    assert DenseConfig(segments=("moe", "attn", "moe")).segments == (
        SegmentKind.ATTN,
        SegmentKind.MOE,
    )
    assert DenseConfig(segments=("attn",)).segments == (SegmentKind.ATTN,)


@pytest.mark.parametrize(
    "segments,message",
    [(("moe",), "must include the attention segment"), (("ffn",), "has an unknown segment")],
)
def test_invalid_segments_raise(segments: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DenseConfig(segments=segments)


@pytest.mark.parametrize("name", ["min_bytes", "store_chunk_mb"])
def test_sizes_are_non_negative(name: str) -> None:
    with pytest.raises(ValueError, match=f"DenseConfig.{name} must be non-negative"):
        DenseConfig(**{name: -1})
