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

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

mamba2_mixer_mod = pytest.importorskip("tensorrt_llm._torch.modules.mamba.mamba2_mixer")
if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
else:
    Mamba2Mixer = mamba2_mixer_mod.Mamba2Mixer


def test_mamba2_mixer_post_load_weights_caches_derived_state():
    mixer = Mamba2Mixer.__new__(Mamba2Mixer)
    nn.Module.__init__(mixer)
    mixer.norm = SimpleNamespace(is_nvfp4=False)
    mixer.A = torch.tensor([1.0, 2.0])
    mixer.dt_bias = torch.tensor([3.0, 4.0])
    mixer.D = torch.tensor([5.0, 6.0])
    mixer.head_dim = 2
    mixer.d_state = 3

    mixer.post_load_weights()

    assert mixer._A_expanded.shape == (2, 2, 3)
    assert mixer._dt_bias_expanded.shape == (2, 2)
    assert mixer._D_expanded.shape == (2, 2)
    assert not hasattr(mixer, "_weights_transformed")


# Non-MTP decode through flashinfer: 16 heads x 64, 2 groups x 128, so the x/B/C
# column views of zxbcdt start 128-byte aligned.
_BATCH, _SLOTS = 8, 20
_HEADS, _HEAD_DIM, _GROUPS, _D_STATE, _D_CONV = 16, 64, 2, 128, 4
_D_INNER = _HEADS * _HEAD_DIM
_CONV_DIM = _D_INNER + 2 * _GROUPS * _D_STATE
_D_IN_PROJ = _D_INNER + _CONV_DIM + _HEADS
# zxbcdt is a column slice of a wider buffer: (buffer row width, first column).
_LAYOUTS = {
    "aligned": (_D_IN_PROJ + 64, 0),  # padded rows, 5280-byte stride
    "misaligned_pointer": (_D_IN_PROJ + 64, 8),  # x/B/C start 16 bytes off
    "misaligned_stride": (_D_IN_PROJ + 4, 0),  # 5160-byte stride (not 16x)
}


def _make_decode_mixer() -> Mamba2Mixer:
    with torch.device("cuda"):
        mixer = Mamba2Mixer(
            d_model=256,
            d_state=_D_STATE,
            d_conv=_D_CONV,
            nheads=_HEADS,
            n_groups=_GROUPS,
            head_dim=_HEAD_DIM,
            chunk_size=128,
            layer_idx=0,
            dtype=torch.bfloat16,
        )
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for p in (mixer.conv1d.weight, mixer.conv1d.bias, mixer.D, mixer.dt_bias):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.5)
        mixer.A.copy_(-0.5 - torch.rand(mixer.A.shape, generator=gen, device="cuda"))
    mixer.post_load_weights()
    assert mixer._use_flashinfer
    return mixer


def _column_ptrs(zxbcdt: torch.Tensor) -> list[int]:
    """Row-0 pointers of the x, B and C columns of zxbcdt."""
    cols = (_D_INNER, 2 * _D_INNER, 2 * _D_INNER + _GROUPS * _D_STATE)
    return [zxbcdt[0, col].data_ptr() for col in cols]


def _run_decode(
    mixer: Mamba2Mixer, monkeypatch: pytest.MonkeyPatch, layout: str, entry: str = "forward_core"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run one decode step on fresh inputs (same values for every layout).

    Returns the output, the whole SSM state, zxbcdt and the x/B/C tensors that
    the state update got.
    """
    width, first = _LAYOUTS[layout]
    gen = torch.Generator(device="cuda").manual_seed(1)
    values = torch.randn(_BATCH, _D_IN_PROJ, generator=gen, device="cuda")
    conv_state = torch.randn(_SLOTS, _CONV_DIM, _D_CONV - 1, generator=gen, device="cuda")
    ssm_state = torch.randn(_SLOTS, _HEADS, _HEAD_DIM, _D_STATE, generator=gen, device="cuda")
    slots = torch.randperm(_SLOTS, generator=gen, device="cuda")[:_BATCH].int()
    zxbcdt = torch.zeros(_BATCH, width, dtype=torch.bfloat16, device="cuda")
    zxbcdt = zxbcdt[:, first : first + _D_IN_PROJ]
    zxbcdt.copy_(values)
    conv_state = conv_state.to(torch.bfloat16)
    ssm_state = ssm_state.to(torch.bfloat16)
    out = torch.zeros(_BATCH, _D_INNER, dtype=torch.bfloat16, device="cuda")

    layer_cache = SimpleNamespace(conv=conv_state, temporal=ssm_state)
    metadata = SimpleNamespace(
        num_contexts=0,
        seq_lens=torch.ones(_BATCH, dtype=torch.int32),
        num_ctx_tokens=0,
        num_tokens=_BATCH,
        kv_cache_manager=SimpleNamespace(
            mamba_layer_cache=lambda layer_idx: layer_cache, is_speculative=lambda: False
        ),
        mamba_metadata=SimpleNamespace(state_indices=slots),
    )
    seen: list[tuple[torch.Tensor, ...]] = []
    state_update = mixer.selective_state_update_func

    def spy(
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        seen.append((x, B, C))
        return state_update(state, x, dt, A, B, C, D, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(mixer, "selective_state_update_func", spy)
        if entry == "custom_op":
            # The boundary op that torch.compile and breakable CUDA graphs use.
            m.setattr(
                mamba2_mixer_mod,
                "_extract_mamba2_extra_attrs",
                lambda layer_idx: (metadata, mixer, None),
            )
            torch.ops.trtllm.mamba2_custom_op_inplace(zxbcdt, mixer.layer_idx_str, out)
        else:
            mixer.forward_core(zxbcdt, metadata, metadata.mamba_metadata, None, out)
    torch.cuda.synchronize()
    assert len(seen) == 1
    return out, ssm_state, zxbcdt, seen[0]


def _run_decode_with_copies(
    mixer: Mamba2Mixer, monkeypatch: pytest.MonkeyPatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: the aligned layout with the x/B/C copies forced."""
    with monkeypatch.context() as m:
        m.setattr(mamba2_mixer_mod, "_flashinfer_row_views_ok", lambda *tensors: False)
        out, ssm_state, _, xbc = _run_decode(mixer, monkeypatch, "aligned")
    assert all(t.is_contiguous() for t in xbc)
    return out, ssm_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("entry", ["forward_core", "custom_op"])
def test_mamba2_decode_passes_aligned_strided_views(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixer = _make_decode_mixer()
    out, ssm_state, zxbcdt, xbc = _run_decode(mixer, monkeypatch, "aligned", entry)
    # The state update reads x/B/C in place, through the padded row stride.
    assert [t.data_ptr() for t in xbc] == _column_ptrs(zxbcdt)
    assert all(t.stride(0) == zxbcdt.stride(0) == _D_IN_PROJ + 64 for t in xbc)

    ref_out, ref_ssm_state = _run_decode_with_copies(mixer, monkeypatch)
    assert torch.equal(out, ref_out)
    assert torch.equal(ssm_state, ref_ssm_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("layout", ["misaligned_pointer", "misaligned_stride"])
def test_mamba2_decode_copies_misaligned_views(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixer = _make_decode_mixer()
    out, ssm_state, zxbcdt, xbc = _run_decode(mixer, monkeypatch, layout)
    # The state update gets contiguous copies, not the views.
    assert all(t.is_contiguous() for t in xbc)
    assert not {t.data_ptr() for t in xbc} & set(_column_ptrs(zxbcdt))

    ref_out, ref_ssm_state = _run_decode_with_copies(mixer, monkeypatch)
    assert torch.equal(out, ref_out)
    assert torch.equal(ssm_state, ref_ssm_state)
