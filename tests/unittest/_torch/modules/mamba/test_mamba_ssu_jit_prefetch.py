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
"""JIT compile ahead of the Mamba2 decode's FlashInfer module.

The module FlashInfer's ``selective_state_update`` builds at a layer's first
single-token decode step is named from the layer at model construction, and
the FlashInfer helper rebuilds the module's spec from that name.
"""

from types import SimpleNamespace

import pytest
import torch

ssu_uri = pytest.importorskip("flashinfer.jit.mamba.selective_state_update")


def _mixer(cache_dtype, rounding=False, philox_rounds=5, head_dim=64, d_state=128):
    """A Mamba2Mixer holding only what its decode step's module follows from."""
    Mamba2Mixer = pytest.importorskip("tensorrt_llm._torch.modules.mamba.mamba2_mixer").Mamba2Mixer

    mixer = Mamba2Mixer.__new__(Mamba2Mixer)
    torch.nn.Module.__init__(mixer)
    mixer._use_flashinfer = True
    mixer._mamba_ssm_cache_dtype = cache_dtype
    mixer._stochastic_rounding_for_flashinfer = rounding
    mixer._philox_rounds = philox_rounds
    mixer.head_dim, mixer.d_state = head_dim, d_state
    mixer.norm = torch.nn.Module()
    mixer.norm.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16, device="meta"))
    return mixer


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "args",
    [
        (torch.float16, torch.bfloat16, None, 64, 128, 1, 5),
        (torch.bfloat16, torch.bfloat16, None, 128, 64, 1, 0),
        (torch.float32, torch.float16, torch.float32, 64, 256, 4, 0),
    ],
    ids=["fp16_rounding", "bf16", "fp32_scale_mtp"],
)
@pytest.mark.parametrize("suffix", ["_sm100", "_sm90", ""])
def test_the_helper_reads_the_module_arguments_from_its_name(args, suffix):
    """Every argument of the module's generator is read back from the name
    FlashInfer gives it, and the generator matches the name's suffix."""
    from tensorrt_llm._torch import jit_prefetch_flashinfer_helper as helper

    state, inp, scale, dim, dstate, ntokens, philox = args
    f32, i32, i64 = torch.float32, torch.int32, torch.int64
    expected = (state, inp, f32, f32, i32, scale, dim, dstate, ntokens, i32, i64, philox)
    generators = {
        "_sm100": ssu_uri.gen_selective_state_update_sm100_module,
        "_sm90": ssu_uri.gen_selective_state_update_sm90_module,
        "": ssu_uri.gen_selective_state_update_module,
    }
    generator, read = helper.ssu_args(ssu_uri.get_selective_state_update_uri(*expected) + suffix)
    assert read == expected
    assert generator is generators[suffix]


@pytest.mark.cpu_only
def test_a_speculative_model_names_no_decode_module():
    """With speculative decoding the generation steps verify drafts, through
    another module or kernel, so the single-token module is not built."""
    from tensorrt_llm._torch.modules.mamba import jit_prefetch_flashinfer as jpf

    model = torch.nn.Sequential(_mixer(torch.float16))
    model.model_config = SimpleNamespace(spec_config=object())
    assert jpf.flashinfer_modules(model) == []
    model[0]._mamba_ssm_cache_dtype = None
    assert jpf.decode_module_name(model[0], 10) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "cache_dtype,rounding",
    [
        (torch.float16, True),
        (torch.float16, False),
        (torch.bfloat16, False),
        (torch.float32, False),
    ],
    ids=["fp16_rounding", "fp16", "bf16", "fp32"],
)
def test_the_named_module_is_the_one_the_decode_step_builds(monkeypatch, cache_dtype, rounding):
    """The module named for a layer is the one FlashInfer resolves for the
    layer's single-token decode call, as ``Mamba2Mixer.forward_core`` makes
    it: the SSM cache in its dtype, ``x``/``B``/``C`` in the activation dtype,
    ``dt`` in float32, the expanded float32 ``A``, ``D`` and ``dt_bias``, the
    int32 state indices, and a Philox seed when rounding stochastically."""
    import importlib

    from tensorrt_llm._torch.modules.mamba import jit_prefetch_flashinfer as jpf

    # The module: flashinfer.mamba re-exports its function under the same name.
    fi_ssu = importlib.import_module("flashinfer.mamba.selective_state_update")

    class Resolved(Exception):
        pass

    seen = []

    def resolve(device, *args, **kwargs):
        seen.append((args, kwargs))
        raise Resolved

    monkeypatch.setattr(fi_ssu, "get_selective_state_update_module", resolve)
    heads, groups, head_dim, d_state, batch = 8, 2, 64, 128, 3
    dev = torch.device("cuda")
    act = torch.bfloat16
    kwargs = dict(
        z=None,
        dt_bias=torch.zeros(heads, head_dim, device=dev),
        dt_softplus=True,
        state_batch_indices=torch.arange(batch, dtype=torch.int32, device=dev),
        out=torch.empty(batch, heads, head_dim, dtype=act, device=dev),
    )
    if rounding:
        kwargs.update(rand_seed=torch.zeros(1, dtype=torch.int64, device=dev), philox_rounds=5)
    with pytest.raises(Resolved):
        fi_ssu.selective_state_update(
            torch.zeros(batch + 2, heads, head_dim, d_state, dtype=cache_dtype, device=dev),
            torch.zeros(batch, heads, head_dim, dtype=act, device=dev),
            torch.zeros(batch, heads, head_dim, device=dev),
            torch.zeros(heads, head_dim, d_state, device=dev),
            torch.zeros(batch, groups, d_state, dtype=act, device=dev),
            torch.zeros(batch, groups, d_state, dtype=act, device=dev),
            torch.zeros(heads, head_dim, device=dev),
            **kwargs,
        )
    ((args, kw),) = seen
    # get_selective_state_update_module(device, state, input, weight, A, index,
    # dim, dstate, ntokens, cu_seqlens, num_accepted, state_scale=, philox_rounds=)
    built = ssu_uri.get_selective_state_update_uri(
        *args[:5], kw.get("state_scale_dtype"), *args[5:10], kw.get("philox_rounds", 0)
    ) + jpf._sm_suffix(torch.cuda.get_device_capability()[0])
    # Two layers of one shape name one module.
    layers = [_mixer(cache_dtype, rounding, head_dim=head_dim, d_state=d_state) for _ in range(2)]
    model = torch.nn.Sequential(*layers)
    model.model_config = SimpleNamespace(spec_config=None)
    assert jpf.flashinfer_modules(model) == [built]
