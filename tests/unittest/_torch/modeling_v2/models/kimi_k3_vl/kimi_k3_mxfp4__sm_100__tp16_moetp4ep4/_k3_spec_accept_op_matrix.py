# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the target's ``spec/k3_spec_accept`` op with vocabulary-sharded logits: the
cells with ``columns``, on W ranks of one NVLink domain (default 4), the W ranks one TP group.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_spec_accept_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _k3_spec_accept_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of calls inside one W-rank job (every call on a workspace rotates its three
Lamport buffers, so the calls are one sequence, not independent cases). The collected entry point is
``test_modeling_v2_k3_spec_accept_op_matrix.py``.

Every rank draws the whole step from one seed (the bf16 logits of every shard, the drafts and the state) and passes
its own shard of ``columns`` columns with the workspace of ``op.workspace(mapping, push_copies)``; ``push_copies`` > 1
makes every rank fill that many exchange slots (TP16's 16-slot exchange on 4 ranks). The reference is the entry's
``reference`` on the group's shards in rank order. Each cell runs its steps with the state carried, a different rank
launching late at every step: every output and every state tensor bit for bit against the reference on every rank,
and the outputs identical across the ranks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_spec_accept requires CUDA devices"

DEADLINE_S = 900

R = None
t = None  # the one-GPU test module: the entry, its cells' state and step builders
accept_op = None  # the kernel's op module (the workspace)


def check_sharded_cells() -> None:
    """Every sharded cell: ``steps`` calls with the state carried, rank ``step % W`` launching each call late."""
    world = R.world
    for cell in [c for c in t.k3_spec_accept.CELLS if "columns" in c.spec]:
        spec = cell.spec
        shard, batch, drafts = spec["columns"], spec["batch"], spec["drafts"]
        vocab = shard * world
        ws = accept_op.workspace(R.mapping, spec["push_copies"])
        gen = torch.Generator(device="cuda").manual_seed(t.cell_seed(cell))
        st_op = t.make_state(gen, batch, vocab)
        st_ref = t.clone_state(st_op)
        for step in range(spec["steps"]):
            t.set_dummies((st_op, st_ref), batch, step)
            x = t.step_logits(
                gen, batch * (drafts + 1), vocab, spec["logits"], step, world
            ).bfloat16()
            draft = t.drafts_for(gen, x, batch, drafts, step)
            shards = [x[:, r * shard : (r + 1) * shard].contiguous() for r in range(world)]
            assert R.same_on_ranks(x, draft), (cell.why, step, "the ranks drew different steps")
            R.barrier()
            R.late(step % world)
            with t.validating(t.k3_spec_accept):
                got = t.k3_spec_accept(
                    logits=shards[R.rank],
                    draft=draft,
                    force=spec["force"],
                    block=spec["block"],
                    ws_uc=ws["uc"],
                    ws_mc=ws["mc"],
                    ws_flags=ws["flags"],
                    rank=ws["rank"],
                    slots=ws["slots"],
                    push_copies=ws["push_copies"],
                    shard_offset=R.rank * shard,
                    **st_op,
                )
            want = t.k3_spec_accept.reference(
                logits=shards,
                draft=draft,
                force=spec["force"],
                block=spec["block"],
                slots=ws["slots"],
                **st_ref,
            )
            torch.cuda.synchronize()
            bad = t.differing(got, want, t.OUTPUTS)
            assert not bad, (cell.why, step, bad)
            bad = t.differing(t.state_of(st_op), t.state_of(st_ref), t.STATE_KEYS)
            assert not bad, (cell.why, step, bad)
            assert R.same_on_ranks(*got), (cell.why, step, "the ranks' outputs differ")
        R.barrier()


CHECKS = [check_sharded_cells]


def _run_one_rank(args) -> int:
    global R, t, accept_op
    R = ls.Rank(args)
    import test_modeling_v2_k3_spec_accept as one_gpu

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_spec_accept import (  # noqa: E501
        op,
    )

    t, accept_op = one_gpu, op
    with torch.inference_mode():
        return ls.run_checks(R, CHECKS)


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
