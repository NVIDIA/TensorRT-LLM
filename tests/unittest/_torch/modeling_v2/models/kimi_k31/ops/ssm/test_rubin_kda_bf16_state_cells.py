# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin KDA entries on a BF16 recurrent state, called as the three layouts call them (sm_107).

The default target lets a deployment keep its KDA state in BF16 (``mamba_ssm_cache_dtype:
bfloat16``). Each case runs an entry on a BF16 pool in the cache manager's page layout: the verify
at draft lengths 1-7 on one batch per kernel form (1, 8 and 32 requests at 12 heads, 1 and 12 at
96), and its graph replay on new inputs; the prefill with the snapshots of block reuse; and the
one-token decode, whose kernel takes an FP32 state only, so the default target keeps a BF16 state
to speculative decoding.
"""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("K3.1 Rubin (sm_107) KDA cells", allow_module_level=True)

from _kda_ops_util import EPS, padded_slots  # noqa: E402
from _rubin_kda_prefill_inputs import (  # noqa: E402
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    clone_written,
    coalesced_pool,
    prefill_args,
    stacked_columns,
    stacked_projection,
    step_schedule,
)
from _rubin_kda_verify_inputs import MUTATED, clone_mutated, verify_args  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import (  # noqa: E402
    kda_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (  # noqa: E402,E501
    rubin_kda_mtp_verify,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (  # noqa: E402
    rubin_kda_prefill,
)

# _validating, and the verify entry's own test for its output and write checks
__extra_import_path__ = ["../../../..", "."]
from _validating import validating  # noqa: E402
from test_rubin_kda_mtp_verify import check_output, check_writes  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="runs the sm_107 kernels",
)

#: Each layout's verify call: (layouts, KDA heads per rank). TEP8 and helix8 run 12 of the 96 heads
#: (KDA TP8), DEP16 all 96 (attention DP).
_VERIFY_CALLS = (("tep8_helix8", 12), ("dep16", 96))
#: Requests per rank, one batch per kernel form the layout's calls take: a cluster pair per (head,
#: request) tile while there are at most 64 tiles, then one CTA per tile while the tiles fit in one
#: wave, then the two-CTA kernel. Each form compiles per head count and draft length.
_BATCHES = {"tep8_helix8": (1, 8, 32), "dep16": (1, 12)}
_DRAFT_LENS = tuple(range(1, 8))
#: State slots of the pool, more than any batch here.
_POOL_SLOTS = 40

_VERIFY_CELLS = [
    pytest.param(
        dict(
            N=n,
            H=heads,
            num_spec=k,
            pool=_POOL_SLOTS,
            stacked=True,
            state_dtype="bfloat16",
            out="bf16",
        ),
        id=f"{name}-b{n}k{k}",
    )
    for name, heads in _VERIFY_CALLS
    for n in _BATCHES[name]
    for k in _DRAFT_LENS
]
#: One captured call per verify kernel: TEP8 b1k7 (one wave) and DEP16 b12k7 (two CTAs per SM).
_REPLAY_CELLS = [
    pytest.param(dict(N=1, H=12, num_spec=7, pool=4, stacked=True), id="tep8_helix8-b1k7"),
    pytest.param(dict(N=12, H=96, num_spec=7, pool=16, stacked=True), id="dep16-b12k7"),
]

_RAGGED = dict(seq_lens=(100, 64, 300, 37), slots=(7, 2, 10, 4), fresh=(True, False, True, False))
_PREFILL_CELLS = [
    pytest.param(
        dict(
            heads=12,
            stacked=True,
            snapshot_rows=(20, 21, -1, 22),
            snapshot_chunks=(1, 1, 0, 0),
            **_RAGGED,
        ),
        id="tep8-snapshots",
    ),
    pytest.param(
        dict(
            heads=96,
            stacked=True,
            seq_lens=(300, 130),
            slots=(1, 2),
            fresh=(True, False),
            snapshot_rows=(5, 6),
            snapshot_chunks=(4, 2),
        ),
        id="dep16-snapshots",
    ),
]

# The prefill entry's bands (its note and GPU test): output and BF16 state rows within 2e-2 of the
# largest magnitude.
PREFILL_TOL_OUT = PREFILL_TOL_STATE = 2e-2


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _verify_call(spec: dict, seed: int, device) -> dict:
    """`rubin_kda_mtp_verify`'s arguments (`verify_args`), the state pool in the cache manager's page
    layout: each slot's state contiguous, the slot stride a whole number of pages."""
    args = verify_args(spec, seed, device)
    state = args["recurrent_state"]
    pool = coalesced_pool(state.shape[0], tuple(state.shape[1:]), state.dtype, device)
    pool.copy_(state)
    args["recurrent_state"] = pool
    return args


# ----------------------------------------------------------------------------- verify


@pytest.mark.parametrize("spec", _VERIFY_CELLS)
@torch.inference_mode()
def test_verify_on_a_bf16_state(spec, request) -> None:
    op = rubin_kda_mtp_verify
    args = _verify_call(spec, _seed(request.node.callspec.id), "cuda")
    ref_args = clone_mutated(args)
    used = set(args["ssm_state_indices"].tolist())
    untouched = [s for s in range(spec["pool"]) if s not in used]
    before = args["recurrent_state"][untouched].clone()
    with validating(op):
        out = op(**args)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()
    check_output(out, ref, spec)
    assert args["recurrent_state"].dtype is torch.bfloat16
    check_writes(args, ref_args, untouched, before)


def test_the_verify_cells_take_both_kernels() -> None:
    """At each head count some cell has at most one (head, request) tile per SM (the one-wave
    kernel) and some cell more (the two-CTA kernel), on this GPU's SM count."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    for name, heads in _VERIFY_CALLS:
        tiles = [heads * n for n in _BATCHES[name]]
        assert min(tiles) <= sms < max(tiles), f"{name}: tiles {tiles} against {sms} SMs"


@pytest.mark.parametrize("spec", _REPLAY_CELLS)
@torch.inference_mode()
def test_verify_replay_on_new_inputs_equals_eager(spec) -> None:
    """Captured on one batch's inputs, replayed on another's: the eager call's output and writes."""
    op = rubin_kda_mtp_verify
    spec = dict(spec, state_dtype="bfloat16")
    captured = _verify_call(spec, seed=5, device="cuda")
    op(**clone_mutated(captured))  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = op(**captured)
    new = _verify_call(spec, seed=6, device="cuda")
    for name, value in new.items():
        if isinstance(value, torch.Tensor):
            captured[name].copy_(value)
    eager = clone_mutated(new)
    graph.replay()
    out_eager = op(**eager)
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)
    for name in MUTATED:
        assert torch.equal(captured[name], eager[name]), name


# ----------------------------------------------------------------------------- prefill


@pytest.mark.parametrize("spec", _PREFILL_CELLS)
@torch.inference_mode()
def test_prefill_on_a_bf16_state(spec, request) -> None:
    op = rubin_kda_prefill
    spec = dict(spec, state_dtype="bfloat16")
    args = prefill_args(spec, _seed(request.node.callspec.id))
    seq_order, cu_chunks = step_schedule(spec["seq_lens"], "cuda")
    args.update(seq_order=seq_order, cu_chunks=cu_chunks)
    ref_args = clone_written(args, ("state_pool",))
    with validating(op):
        out = op(**args)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()

    assert out.dtype == torch.bfloat16 and out.shape == args["q"].shape
    assert_scaled_close("output", out, ref, PREFILL_TOL_OUT)
    pool, ref_pool = args["state_pool"], ref_args["state_pool"]
    assert pool.dtype is torch.bfloat16
    written = set(spec["slots"]) | {r for r in spec.get("snapshot_rows", ()) if r >= 0}
    for row in range(pool.shape[0]):
        if row in written:
            assert_scaled_close(f"state row {row}", pool[row], ref_pool[row], PREFILL_TOL_STATE)
        else:
            assert torch.equal(pool[row], ref_pool[row]), f"unselected row {row} changed"


# ----------------------------------------------------------------------------- decode


def _decode_call(heads: int, batch: int, state_dtype: torch.dtype, seed: int) -> dict:
    """`kda_decode`'s arguments as the layouts pass them: q, k, v, the output gate and beta as views
    of one stacked projection, the forget-gate logits from their own GEMM, the three section views
    of the packed conv pool and the state pool, both with padded slots."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = heads * HEAD_DIM
    slots = batch + 4

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    conv = padded_slots(slots, (3 * dim, 3), torch.bfloat16)
    conv.copy_(randn(*conv.shape, scale=0.5))
    state = padded_slots(slots, (heads, HEAD_DIM, HEAD_DIM), state_dtype)
    state.copy_(randn(*state.shape, scale=0.05))
    proj = stacked_projection(batch, heads, gen, "cuda")
    b0 = stacked_columns(heads)["b"]

    def heads_view(t):
        return t.unflatten(-1, (heads, HEAD_DIM)).unsqueeze(0)

    zeros = torch.zeros(dim, dtype=torch.bfloat16, device="cuda")
    return dict(
        x_q=heads_view(proj[:, :dim]),
        x_k=heads_view(proj[:, dim : 2 * dim]),
        x_v=heads_view(proj[:, 2 * dim : 3 * dim]),
        w_q_t=randn(4, dim, scale=0.3).to(torch.bfloat16),
        w_k_t=randn(4, dim, scale=0.3).to(torch.bfloat16),
        w_v_t=randn(4, dim, scale=0.3).to(torch.bfloat16),
        bias_q=zeros,
        bias_k=zeros,
        bias_v=zeros,
        conv_state_q=conv[:, :dim],
        conv_state_k=conv[:, dim : 2 * dim],
        conv_state_v=conv[:, 2 * dim :],
        a_log=randn(heads, scale=0.5),
        g=heads_view(randn(batch, dim).to(torch.bfloat16)),
        dt_bias=randn(dim, scale=0.5),
        beta=proj[:, b0 : b0 + heads].unsqueeze(0),
        onorm_g=heads_view(proj[:, 3 * dim : 4 * dim]),
        onorm_weight=1 + randn(HEAD_DIM, scale=0.1),
        ssm_state_indices=torch.randperm(slots, generator=gen, device="cuda")[:batch].to(
            torch.int32
        ),
        state=state,
        apply_onorm=True,
        update_conv_cache=True,
        use_lower_bound=True,
        apply_beta_sigmoid=True,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        onorm_eps=EPS,
        output=torch.empty(batch, 1, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
    )


@torch.inference_mode()
def test_decode_refuses_a_bf16_state() -> None:
    """The call that runs on an FP32 pool is refused on a BF16 one. The op checks the state dtype
    before any head-count dispatch, so one head count (12, TEP8 and helix8) covers it."""
    op = kda_decode
    with validating(op):
        op(**_decode_call(12, 16, torch.float32, seed=28))
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="state must be a CUDA float32 tensor"):
        with validating(op):
            op(**_decode_call(12, 16, torch.bfloat16, seed=28))
