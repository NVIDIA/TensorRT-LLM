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
"""Native attention → attention_output_tail → native reduction/norm, multi-rank, the target's GlmMLA.forward.

Runs with the plan that runs only ``attention_output_tail`` (opt-in:
``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED=1``),
one process per GPU under
``srun --mpi=pmix``. Rank ``r`` builds TP8 rank ``r`` of real layers 2 and 3
(``real_mla_harness.AttentionStack``: all attention weights real, FP8 KV
cache filled by a real prefill) and runs the GEN step (R=6, one request, MTP
draft 5) as a decode step through ``GlmMLA.forward``, so the latent comes from the native DSA
attention kernel inside the target's DSA attention op. Each rank's output
feeds the native reduction exactly as the decoder layer does:

* layer 2 (dense destination): ``o_proj`` reduces inside ``Linear.forward``
  (``AllReduceParams(enable_allreduce=True)``), then the real
  ``post_attention_layernorm(hidden, residual)``;
* layer 3 (MoE destination): ``o_proj`` returns the rank-local partial
  (``enable_allreduce=False``), then ``AllReduce(RESIDUAL_RMS_NORM)`` with the
  real ``post_attention_layernorm`` weight.

With eight ranks on two nodes (``-N2``) this is the GEN role's TP8 group;
with fewer ranks the reduction sums the shards of ranks ``0 .. world-1``
only, which still exercises the bridge.

The native DSA attention output varies slightly between identical runs, so
each arm is judged against its own oracle: every rank records the latent its
tail received (``LatentRecorder``), the FP64 tail oracle of that latent (with
the native BF16 rounding of ``v``) is summed over ranks in FP64, and the
residual add and RMSNorm are applied in FP64. attention_output_tail's normalized RMS and max
errors against its oracle must stay within the frozen attention_output_tail budget relative to
native's errors against native's oracle. A second native run (A/A) is
reported for scale. Results are gathered from all ranks before any assertion,
so a failure on one rank cannot leave the others waiting in a collective.

The attention_output_tail contract ("no communication is hidden inside attention_output_tail"): each
``o_proj`` gets its own view of the shared all-reduce that counts calls and flags any call
made while an ``AttnTail`` entry point runs. In both arms every layer call
must reach the all-reduce exactly once, outside the tail, with the params
object the caller passed: from ``Linear.forward`` natively, from
``MLA.forward`` with attention_output_tail. A second test captures the attention_output_tail step
(both destinations, real all-reduces) in a CUDA graph and replays it with new
inputs against the same per-arm FP64 oracles.
"""

import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_mla_harness as harness  # noqa: E402
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

_CKPT = tail_fixtures.checkpoint_dir()
HIDDEN = 6144
LAYERS = [2, 3]


def _comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (_CKPT and os.path.isdir(_CKPT)), reason="checkpoint not available"),
    pytest.mark.skipif(
        os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED") != "1",
        reason="needs attention_output_tail selected (opt-in)",
    ),
    pytest.mark.skipif(_comm().Get_size() < 2, reason="needs >= 2 MPI ranks"),
    pytest.mark.threadleak(enabled=False),
]

_SETUP = {}
TAIL_ENTRY_POINTS = ("arm", "before_attention", "defer_w_uv", "project")


class _InTail:
    """Set while an ``AttnTail`` entry point runs (class-level hooks, so
    they compose with the harness's per-instance latent recorder)."""

    active = False


def _hook_tail_entry_points():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
        AttnTail,
    )

    if getattr(AttnTail, "_contract_o_hooked", False):
        return
    for name in TAIL_ENTRY_POINTS:
        entry = getattr(AttnTail, name)

        def run(self, *args, _entry=entry, **kwargs):
            _InTail.active = True
            try:
                return _entry(self, *args, **kwargs)
            finally:
                _InTail.active = False

        setattr(AttnTail, name, run)
    AttnTail._contract_o_hooked = True


class _CountingAllReduce:
    """One ``o_proj``'s view of the shared ``AllReduce``: forwards every
    call and records whether it was made inside the attention_output_tail and with which
    params object."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, input, all_reduce_params=None):
        self.calls.append((_InTail.active, all_reduce_params))
        return self.inner(input, all_reduce_params=all_reduce_params)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _setup():
    if "s" in _SETUP:
        return _SETUP["s"]
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    comm = _comm()
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(
        world_size=world, tp_size=world, rank=rank, gpus_per_node=torch.cuda.device_count()
    )
    stack = harness.AttentionStack(
        _CKPT, LAYERS, tp_rank=rank, ctx=4096, rows=6, fast_paths="attention_output_tail"
    )
    # As in the model: o_proj owns an all-reduce; the decoder layer enables
    # it for dense layers and fuses it into the post-attention norm for MoE.
    allreduce = AllReduce(mapping=mapping, dtype=torch.bfloat16)
    for attn in stack.attn.values():
        attn.o_proj.reduce_output = True
        attn.o_proj.all_reduce = _CountingAllReduce(allreduce)
    _hook_tail_entry_points()
    eps = stack.pretrained.rms_norm_eps
    norms = {}
    for layer in LAYERS:
        norm = RMSNorm(hidden_size=HIDDEN, eps=eps, dtype=torch.bfloat16).cuda()
        norm.weight.data.copy_(
            tail_fixtures.read_tensor(
                _CKPT, f"model.layers.{layer}.post_attention_layernorm.weight"
            ).cuda()
        )
        norms[layer] = norm
    _SETUP["s"] = (rank, world, stack, norms, allreduce, eps)
    return _SETUP["s"]


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _chain(
    stack, norms, allreduce, eps, attention_output_tail, inputs, residual, params=None, sync=True
):
    """GEN step of layers 2 (dense) and 3 (MoE) with their native reductions.

    Returns the outputs and the latent each layer's tail received; ``params``
    (if given) receives the ``AllReduceParams`` object passed to each layer.
    """
    from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams

    stack.set_attn_tail(attention_output_tail)
    pos = stack.position_ids()
    params = {} if params is None else params
    params[2] = AllReduceParams(enable_allreduce=True)
    params[3] = AllReduceParams(enable_allreduce=False)
    # Outputs are cloned right after they are produced: an all-reduce result
    # may live in a workspace that the next all-reduce reuses.
    out = {}
    with stack.scope(), harness.LatentRecorder(stack) as recorder:
        reduced = stack.forward(2, inputs[2], pos, params[2])
        out["dense_reduced"] = reduced.clone()
        hs2, res2 = norms[2](reduced, residual.clone())
        out["dense_hidden"], out["dense_residual"] = hs2.clone(), res2.clone()
        partial = stack.forward(3, inputs[3], pos, params[3])
        out["moe_partial"] = partial.clone()
        hs3, res3 = allreduce(
            partial,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual.clone(),
                norm_weight=norms[3].weight,
                eps=eps,
                trigger_completion_at_end=False,
            ),
        )
        out["moe_hidden"], out["moe_residual"] = hs3.clone(), res3.clone()
        latents = dict(recorder.latents)
    if sync:
        torch.cuda.synchronize()
    return out, latents


def _take_calls(stack, params, label):
    """The attention_output_tail contract: one all-reduce call per layer, outside the tail, with
    the caller's params object. Returns failure strings and clears."""
    failures = []
    for layer in LAYERS:
        calls = stack.attn[layer].o_proj.all_reduce.calls
        ok = len(calls) == 1 and not calls[0][0] and calls[0][1] is params[layer]
        if not ok:
            failures.append(
                f"{label} layer{layer}: o_proj all-reduce calls "
                f"(in_tail, same_params) = "
                f"{[(c[0], c[1] is params[layer]) for c in calls]}"
            )
        calls.clear()
    return failures


def _oracle(stack, norms, eps, latents, residual):
    """FP64 references of ``_chain``'s outputs from the recorded latents."""
    import numpy as np
    from mpi4py import MPI

    def summed(local):
        send = local.cpu().numpy()
        recv = np.empty_like(send)
        _comm().Allreduce(send, recv, op=MPI.SUM)
        return torch.from_numpy(recv).cuda()

    def norm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.double()

    res_in = residual.double()
    partial = {}
    for layer in LAYERS:
        attn = stack.attn[layer]
        partial[layer], _ = tail_reference.oracle_attn_tail(
            latents[layer], attn.v_b_proj, attn.o_proj.weight, stack.rows
        )
    dense = summed(partial[2])
    moe = summed(partial[3])
    return {
        "dense_reduced": dense,
        "dense_hidden": norm(dense + res_in, norms[2].weight),
        "dense_residual": dense + res_in,
        "moe_partial": partial[3],
        "moe_hidden": norm(moe + res_in, norms[3].weight),
        "moe_residual": moe + res_in,
    }


@pytest.mark.parametrize("step", [0, 1])
def test_native_producer_attn_tail_native_reduction(step):
    rank, world, stack, norms, allreduce, eps = _setup()
    inputs = {layer: stack.gen_inputs(layer, step=step) for layer in LAYERS}
    gen = torch.Generator(device="cuda").manual_seed(4321 + step)
    residual = torch.randn(
        (stack.rows, HIDDEN), generator=gen, device="cuda", dtype=torch.float32
    ).bfloat16()
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    for layer in LAYERS:
        stack.attn[layer].o_proj.all_reduce.calls.clear()
    failures, report = [], []
    params = {}
    native, lat_native = _chain(stack, norms, allreduce, eps, False, inputs, residual, params)
    failures += _take_calls(stack, params, "native")
    native_b, _ = _chain(stack, norms, allreduce, eps, False, inputs, residual, params)
    failures += _take_calls(stack, params, "native_b")
    attention_output_tail, lat_attn_tail = _chain(
        stack, norms, allreduce, eps, True, inputs, residual, params
    )
    failures += _take_calls(stack, params, "attention_output_tail")
    report.append("contract_o_calls=" + ("ok" if not failures else "BAD"))
    ref_native = _oracle(stack, norms, eps, lat_native, residual)
    ref_attn_tail = _oracle(stack, norms, eps, lat_attn_tail, residual)

    for layer in LAYERS:
        tail = stack.attn_tail_tails[layer]
        # As in the model (default TLLM_NCCL_SYMMETRIC_ZERO_COPY, AUTO
        # strategy, TP > 1), the o_proj all-reduce may use an NCCL window,
        # but the native o_proj GEMM output is an ordinary F.linear buffer,
        # so attention_output_tail still takes the call.
        report.append(
            f"layer{layer}: nccl_window_allreduce="
            f"{stack.attn[layer].o_proj.all_reduce.uses_nccl_symmetric_memory_window()} "
            f"static_reason={tail.static_reason} "
            f"engaged={tail.engaged_calls - engaged[layer]}"
        )
        if tail.static_reason is not None or (tail.engaged_calls != engaged[layer] + 1):
            failures.append(report[-1])
    for key in native:
        e_attn_tail = tail_reference.error_stats(attention_output_tail[key], ref_attn_tail[key])
        e_nat = tail_reference.error_stats(native[key], ref_native[key])
        aa = tail_reference.error_stats(native_b[key], native[key].double())
        report.append(
            f"{key}: rel_rms attention_output_tail={e_attn_tail['rel_rms']:.2e} "
            f"native={e_nat['rel_rms']:.2e} "
            f"max attention_output_tail={e_attn_tail['max_abs']:.2e} "
            f"native={e_nat['max_abs']:.2e} "
            f"(A/A rel_rms {aa['rel_rms']:.2e})"
        )
        if (
            e_attn_tail["rel_rms"] > 1.25 * e_nat["rel_rms"] + 1e-6
            or e_attn_tail["max_abs"] > 2 * e_nat["max_abs"] + 2**-8 * e_attn_tail["ref_max"]
        ):
            failures.append(report[-1])
    digests = tuple(
        _digest(attention_output_tail[k])
        for k in ("dense_reduced", "dense_hidden", "moe_hidden", "moe_residual")
    )
    gathered = _comm().allgather((rank, failures, report, digests))
    if rank == 0:
        for r, _, rep, _ in gathered:
            print(
                f"[attention_output_tail-mla-forward-mpi] rank={r}/{world} step={step} | "
                + " | ".join(rep)
            )
    bad = [(r, f) for r, f, _, _ in gathered if f]
    assert not bad, bad
    assert len({d for _, _, _, d in gathered}) == 1, "reduced outputs differ across ranks"


def test_graph_replay_native_reduction():
    """The attention_output_tail step of both destinations, real all-reduces included, captured
    in one CUDA graph and replayed with new inputs and residuals; each replay
    is judged like the eager step, against native eager on the same inputs."""
    rank, world, stack, norms, allreduce, eps = _setup()
    static_in = {layer: stack.gen_inputs(layer, step=0).clone() for layer in LAYERS}
    static_res = torch.zeros((stack.rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    for layer in LAYERS:
        stack.attn[layer].o_proj.all_reduce.calls.clear()
    failures, report = [], []
    params = {}
    for _ in range(2):  # eager warm-up (JIT, per-layer state), as the engine
        _chain(stack, norms, allreduce, eps, True, static_in, static_res, params)
        failures += _take_calls(stack, params, "warm-up")
    _comm().Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lat = _chain(
            stack, norms, allreduce, eps, True, static_in, static_res, params, sync=False
        )
    failures += _take_calls(stack, params, "capture")
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    digests = []
    for step in range(1, 4):
        for layer in LAYERS:
            static_in[layer].copy_(stack.gen_inputs(layer, step=step))
        gen = torch.Generator(device="cuda").manual_seed(4321 + step)
        static_res.copy_(
            torch.randn(
                (stack.rows, HIDDEN), generator=gen, device="cuda", dtype=torch.float32
            ).bfloat16()
        )
        _comm().Barrier()
        graph.replay()
        torch.cuda.synchronize()
        got = {k: v.clone() for k, v in captured.items()}
        lat = {k: v.clone() for k, v in captured_lat.items()}
        native, lat_native = _chain(
            stack,
            norms,
            allreduce,
            eps,
            False,
            {layer: static_in[layer].clone() for layer in LAYERS},
            static_res.clone(),
            params,
        )
        failures += _take_calls(stack, params, f"native step{step}")
        ref_attn_tail = _oracle(stack, norms, eps, lat, static_res)
        ref_native = _oracle(stack, norms, eps, lat_native, static_res)
        for key in got:
            e_attn_tail = tail_reference.error_stats(got[key], ref_attn_tail[key])
            e_nat = tail_reference.error_stats(native[key], ref_native[key])
            report.append(
                f"step{step} {key}: rel_rms "
                f"graph={e_attn_tail['rel_rms']:.2e} "
                f"native={e_nat['rel_rms']:.2e}"
            )
            if (
                e_attn_tail["rel_rms"] > 1.25 * e_nat["rel_rms"] + 1e-6
                or e_attn_tail["max_abs"] > 2 * e_nat["max_abs"] + 2**-8 * e_attn_tail["ref_max"]
            ):
                failures.append(report[-1])
        digests.append(
            tuple(_digest(got[k]) for k in ("dense_reduced", "dense_hidden", "moe_hidden"))
        )
    if engaged != {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}:
        failures.append("replays ran tail Python")
    gathered = _comm().allgather((rank, failures, report, tuple(digests)))
    if rank == 0:
        for r, _, rep, _ in gathered:
            print(
                f"[attention_output_tail-mla-forward-mpi-graph] rank={r}/{world} | "
                + " | ".join(rep)
            )
    bad = [(r, f) for r, f, _, _ in gathered if f]
    assert not bad, bad
    assert len({d for _, _, _, d in gathered}) == 1, "replayed reduced outputs differ across ranks"
    del graph
