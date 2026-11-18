# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-sharded AttnRes (wide verify shapes, M >= 256): rank r owns rows
[r * per, (r + 1) * per), per = ceil(M / TP), runs every AttnRes on them only
(``ClusterAttnRes(out_mc=True)``: its output rows go to every rank's Lamport
all-gather box) and ``GatherRows`` turns the box into the plain rows the
in-projections, the MoE front and the LM head read."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.mailbox import poll, rearm
from ..common.primitives import at, launch, store_words, thread


def owned(rows, tp, rank):
    """(first row, row count) of ``rank``'s share of ``rows``."""
    per = -(-rows // tp)
    first = min(rank * per, rows)
    return first, min(per, rows - first)


class ShardBox:
    """The row-sharded AttnRes mailboxes at ``cap`` rows (the deployment's largest
    graph), all Lamport-armed and re-armed in the call that reads them:

    - ``gathers``: three all-gather boxes [cap, 7168] (multicast): 0 the o-projection
      owners' x_moe rows, 1 the other AttnRes outputs (x_attn, the final rows), 2 the
      aux prefix rows; ``GatherRows`` copies them into the plain rows;
    - ``owners``: the o-projection row owners' receive slots [TP, per_cap, 7168]
      (per_cap = ceil(cap / TP)), with every rank's base (``owner_peers``);
    - ``ring``: the wide MoE tail's row-owner ring [1, per_cap, 7168], with every rank's
      base (``ring_peers``)."""

    def __init__(self, group, device, tp, rank, cap):
        from ..oproj.prepared import OprojMailbox

        self.tp, self.rank, self.cap = tp, rank, cap
        self.per_cap = per_cap = -(-cap // tp)
        self.gathers = [
            OprojMailbox(group, device, tp, rank, (cap, 7168)) for _ in range(3)
        ]
        self.owners = OprojMailbox(group, device, tp, rank, (tp, per_cap, 7168))
        self.ring = OprojMailbox(group, device, tp, rank, (1, per_cap, 7168))
        i64 = dict(dtype=torch.int64, device=device)
        self.owner_peers = torch.tensor(self.owners.handle.buffer_ptrs, **i64)
        self.ring_peers = torch.tensor(self.ring.handle.buffer_ptrs, **i64)


class GatherRows:
    """One 224-thread CTA per row: poll the all-gather box row (896 16-byte
    fragments, four per thread), copy it into the plain row and re-arm it."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, args[1].shape[0], 224)

    @cute.kernel
    def kernel(self, args):
        box, out = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()  # ``out``'s previous readers are done
        cute.arch.griddepcontrol_launch_dependents()
        base = at(box, token, tid * 8)
        got = poll(base, 224 * 8, 4)
        for i in cutlass.range_constexpr(4):
            rearm(base + i * 224 * 8)
            words = tuple(got[i, k] for k in range(4))
            store_words(at(out, token, (tid + 224 * i) * 8), words)


@cache
def compile_gather(rows, cap, device):
    with torch.accelerator.device_index(device):
        box, out = fake((cap, 7168)), fake((rows, 7168))
        return cute.compile(GatherRows(), box, out, make_fake_stream())
