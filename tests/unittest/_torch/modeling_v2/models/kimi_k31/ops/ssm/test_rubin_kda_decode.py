# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family kda_decode entry as the Rubin targets call it, over chained decode steps (sm_107).

The Rubin targets reuse the family's one-token decode entry (it takes the FP32 state they keep). Its
own test drives its cells one step at a time; here the Rubin batches run several steps on the same
pools, each step's projections new, eagerly and as one CUDA graph replayed per step: the state and
conv windows a step leaves are the next step's input, so the op must stay on the reference chain.
The chained steps also take their projections as the specialized targets pass them: views of one
stacked [q | k | v | g | f_a | b] GEMM output.
"""

import math

import pytest
import torch
from _kda_ops_util import (
    EPS,
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    bits,
    clone_strided,
    padded_slots,
)
from _rubin_kda_prefill_inputs import stacked_projection

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import (
    kda_decode,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin target shapes: driven on sm_107",
)

# The entry's decode bands per step; three chained steps keep the state within 3x.
TOL_OUT, TOL_STATE = 2e-2, 3e-3
STEPS = 3


def _layer(heads: int, batch: int, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = heads * HEAD_DIM
    slots = batch + 4
    conv = padded_slots(slots, (3 * dim, 3), torch.bfloat16)
    conv.copy_(torch.randn(conv.shape, generator=gen, device="cuda") * 0.5)
    state = padded_slots(slots, (heads, HEAD_DIM, HEAD_DIM), torch.float32)
    state.copy_(torch.randn(state.shape, generator=gen, device="cuda") * 0.05)
    zeros = torch.zeros(dim, dtype=torch.bfloat16, device="cuda")
    return dict(
        heads=heads,
        gen=gen,
        conv=conv,
        weights=dict(
            w_q_t=(torch.randn(4, dim, generator=gen, device="cuda") * 0.3).to(torch.bfloat16),
            w_k_t=(torch.randn(4, dim, generator=gen, device="cuda") * 0.3).to(torch.bfloat16),
            w_v_t=(torch.randn(4, dim, generator=gen, device="cuda") * 0.3).to(torch.bfloat16),
            bias_q=zeros,
            bias_k=zeros,
            bias_v=zeros,
            a_log=torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen)),
            dt_bias=torch.empty(dim, device="cuda").uniform_(
                math.log(1e-3), math.log(1e-1), generator=gen
            ),
            onorm_weight=1 + torch.randn(HEAD_DIM, generator=gen, device="cuda") * 0.1,
        ),
        state=state,
        indices=torch.randperm(slots, generator=gen, device="cuda")[:batch].to(torch.int32),
        out=torch.empty(batch, 1, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
    )


def _step_rows(lay: dict, batch: int, stacked: bool = False) -> dict:
    """One step's projection outputs: the fused [q|k|v|g] GEMM and the [f_a|b|pad] projection, or
    with `stacked` both as views of one `stacked_projection`."""
    heads, gen = lay["heads"], lay["gen"]
    dim = heads * HEAD_DIM
    if stacked:
        proj = stacked_projection(batch, heads, gen, "cuda")
        qkvg, merged = proj[:, : 4 * dim], proj[:, 4 * dim :]
    else:
        qkvg = torch.randn(batch, 4 * dim, generator=gen, device="cuda").to(torch.bfloat16)
        width = (HEAD_DIM + heads + 7) // 8 * 8
        merged = torch.randn(batch, width, generator=gen, device="cuda").to(torch.bfloat16)
    gate = torch.randn(batch, dim, generator=gen, device="cuda").to(torch.bfloat16)
    return dict(qkvg=qkvg, merged=merged, gate=gate)


def _args(
    lay: dict, rows: dict, conv: torch.Tensor, state: torch.Tensor, out: torch.Tensor
) -> dict:
    heads = lay["heads"]
    dim = heads * HEAD_DIM

    def heads_view(t):
        return t.unflatten(-1, (heads, HEAD_DIM)).unsqueeze(0)

    qkvg = rows["qkvg"]
    return dict(
        x_q=heads_view(qkvg[:, :dim]),
        x_k=heads_view(qkvg[:, dim : 2 * dim]),
        x_v=heads_view(qkvg[:, 2 * dim : 3 * dim]),
        conv_state_q=conv[:, :dim],
        conv_state_k=conv[:, dim : 2 * dim],
        conv_state_v=conv[:, 2 * dim :],
        g=heads_view(rows["gate"]),
        beta=rows["merged"][:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0),
        onorm_g=heads_view(qkvg[:, 3 * dim :]),
        ssm_state_indices=lay["indices"],
        state=state,
        apply_onorm=True,
        update_conv_cache=True,
        use_lower_bound=True,
        apply_beta_sigmoid=True,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        onorm_eps=EPS,
        output=out,
        **lay["weights"],
    )


@pytest.mark.parametrize(
    "heads,batch,stacked",
    [(96, 12, False), (12, 16, False), (12, 1, False), (96, 12, True), (12, 16, True)],
    ids=["DEP16-b12", "helix8-b16", "TEP8-b1", "DEP16-b12-stacked", "helix8-b16-stacked"],
)
@torch.inference_mode()
def test_chained_steps_follow_the_reference(heads, batch, stacked) -> None:
    lay = _layer(heads, batch, seed=heads * 100 + batch)
    ref_conv, ref_state = clone_strided(lay["conv"]), clone_strided(lay["state"])
    ref_out = torch.empty_like(lay["out"])
    with validating(kda_decode):
        for step in range(STEPS):
            rows = _step_rows(lay, batch, stacked)
            kda_decode(**_args(lay, rows, lay["conv"], lay["state"], lay["out"]))
            kda_decode.reference(**_args(lay, rows, ref_conv, ref_state, ref_out))
            torch.cuda.synchronize()
            assert_scaled_close(lay["out"], ref_out, TOL_OUT, f"step {step} output")
            assert torch.equal(bits(lay["conv"]), bits(ref_conv)), f"step {step} conv windows"
    for slot in lay["indices"].tolist():
        assert_scaled_close(lay["state"][slot], ref_state[slot], TOL_STATE, f"state slot {slot}")


@pytest.mark.parametrize("heads,batch", [(96, 12), (12, 16)], ids=["DEP16-b12", "helix8-b16"])
@torch.inference_mode()
def test_graph_replay_per_step_matches_eager(heads, batch) -> None:
    """One captured decode replayed per step equals the eager steps bit for bit."""
    lay = _layer(heads, batch, seed=7)
    eager_conv, eager_state = clone_strided(lay["conv"]), clone_strided(lay["state"])
    eager_out = torch.empty_like(lay["out"])
    static = _step_rows(lay, batch)
    kda_decode(
        **_args(
            lay,
            static,
            clone_strided(lay["conv"]),
            clone_strided(lay["state"]),
            torch.empty_like(lay["out"]),
        )
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kda_decode(**_args(lay, static, lay["conv"], lay["state"], lay["out"]))
    for step in range(STEPS):
        rows = _step_rows(lay, batch)
        for name in rows:
            static[name].copy_(rows[name])
        graph.replay()
        kda_decode(**_args(lay, rows, eager_conv, eager_state, eager_out))
        torch.cuda.synchronize()
        assert torch.equal(bits(lay["out"]), bits(eager_out)), f"step {step} output"
        assert torch.equal(bits(lay["conv"]), bits(eager_conv)), f"step {step} conv"
        assert torch.equal(bits(lay["state"]), bits(eager_state)), f"step {step} state"
