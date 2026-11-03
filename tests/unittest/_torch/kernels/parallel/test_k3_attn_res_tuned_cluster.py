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
"""``k3_attn_res.tuned_cluster_size``: the vectorized attention-residual kernel with one or two CTAs per
token gives the same bits as with four, for every add mode, so the tuned choice changes speed only. It is the
default for Kimi K3.1 (``enable_tuned_cluster``) and opt-in elsewhere (TRTLLM_K3_ATTN_RES_TUNED_CLUSTER=1)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an SM100/SM103 GPU",
)

HIDDEN = 7168


def _k3():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.kernels.blackwell import k3_attn_res

    return k3_attn_res


def _run(k3, t, mode, cluster):
    m = t["layer"].shape[0]
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    upd = torch.empty_like(t["layer"]) if mode else None
    k3.run_k3_attn_res(
        t["layer"],
        addends,
        t["block"],
        t["proj"],
        t["norm_w"],
        t["out_w"],
        1e-6,
        1e-5,
        out,
        upd,
        exact=False,
        cluster=cluster,
        add_rn=mode == 1,
    )
    torch.cuda.synchronize()
    assert out.shape[0] == m
    return out, upd


def _inputs(m, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        layer=rn(m, HIDDEN),
        addend=rn(m, HIDDEN, sc=0.5),
        shared=rn(m, HIDDEN, sc=0.5),
        block=rn(n - 1, m, HIDDEN),
        proj=rn(HIDDEN, sc=0.05),
        norm_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 4, 6, 7, 9])
@pytest.mark.parametrize("m", [28, 42, 84, 112, 128, 144])
def test_tuned_cluster_is_bitwise(m: int, n: int, mode: int, monkeypatch):
    k3 = _k3()
    t = _inputs(m, n, seed=31 * m + n)
    ref = _run(k3, t, mode, 4)
    for c in (1, 2):
        got = _run(k3, t, mode, c)
        for a, b in zip(got, ref):
            if a is None:
                continue
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (m, n, mode, c)
    # The tuned default dispatch (Kimi K3.1) gives the same bits as well.
    monkeypatch.setattr(k3, "_TUNED_CLUSTER", True)
    dflt = _run(k3, t, mode, None)
    for a, b in zip(dflt, ref):
        if a is not None:
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (m, n, mode, "default")


@pytest.mark.parametrize("env", [None, "0", "1"])
def test_tuned_cluster_default(env, monkeypatch):
    """Off by default (Kimi K3); a Kimi K3.1 model enables it unless the env var says otherwise."""
    k3 = _k3()
    monkeypatch.setattr(k3, "_TUNED_CLUSTER_ENV", env)
    monkeypatch.setattr(k3, "_TUNED_CLUSTER", env == "1")
    k3.enable_tuned_cluster()
    assert k3._TUNED_CLUSTER == (env != "0")


def test_tuned_cluster_policy():
    k3 = _k3()
    assert k3.tuned_cluster_size(7, 3, False) == 4
    assert k3.tuned_cluster_size(28, 3, False) == 1
    assert k3.tuned_cluster_size(28, 9, True) == 4
    assert k3.tuned_cluster_size(56, 7, False) == 2
    assert k3.tuned_cluster_size(112, 5, True) == 4
    assert k3.tuned_cluster_size(112, 6, True) == 1
    assert k3.tuned_cluster_size(112, 9, False) == 1
