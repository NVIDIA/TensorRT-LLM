# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_query_wait entry, compiled and called as ``FlatMlaQuery`` does:
it returns on a published mailbox without counting a timeout or writing it, and with the rows
published by another stream while it polls it returns only after they land. The target releases
dependents after its polls. The standalone kernel also supports immediate release."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_query_wait import (  # noqa: E501
    SENTINEL,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_query_wait import (  # noqa: E501
    dsa_query_wait as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the query wait runs where the K3.1 ULL target does (SM10x)",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

CAPACITY = 64


def _kernel(mailbox: torch.Tensor, rows: int, release_after_poll: bool):
    """(compiled QueryWait, mailbox view, spins, spins view, stream) as ``FlatMlaQuery`` builds them."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_ull_query_wait import (  # noqa: E501
        QueryWait,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8 import (  # noqa: E501
        module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8.ops.cute_dsl.latent_moe_tail.primitives import (  # noqa: E501
        to_cute,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501
        query_wait_kernel,
    )

    host = module("cutedsl.common.host")
    view = to_cute(mailbox[:rows], 16)
    kernel = query_wait_kernel(rows)
    assert kernel.release_after_poll is True
    if not release_after_poll:
        kernel = QueryWait(kernel.chunks, kernel.record, release_after_poll=False)
    compiled = cute.compile(kernel, view, host.fake((4,), None, torch.int32, 4), make_fake_stream())
    spins = torch.zeros(4, dtype=torch.int32, device="cuda")
    return compiled, view, spins, to_cute(spins, 4), host.stream(mailbox.device)


@pytest.mark.parametrize(
    "release_after_poll", [False, True], ids=["release_at_once", "release_after_poll"]
)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, release_after_poll) -> None:
    rows = cell.spec["rows"]
    mailbox = torch.full((CAPACITY, 96, 576), SENTINEL, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(rows)
    # Published FP8 query bytes: anything but four 0xFF bytes in one word.
    data = torch.randint(0, 0xFF, (rows, 96, 576), generator=g, device="cuda", dtype=torch.uint8)
    compiled, view, spins, spins_view, stream = _kernel(mailbox, rows, release_after_poll)
    torch.cuda.synchronize()
    if not cell.spec["late"]:
        mailbox[:rows].copy_(data)
        with validating(op):
            op(compiled, view, spins_view, stream)
        torch.cuda.synchronize()
    else:
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(main)
        # The publish is enqueued first: no launch may follow the polling kernel on the host (a
        # lazily loaded module, e.g. _sleep's, would wait for the resident poller: a deadlock).
        with torch.cuda.stream(side):
            torch.cuda._sleep(20_000_000)  # about 10 ms: the kernel is polling well before it lands
            mailbox[:rows].copy_(data)
        with validating(op):
            op(compiled, view, spins_view, stream)
        # What the next copy on the stream reads: the published rows only if the wait waited.
        seen = mailbox[:rows].clone()
        torch.cuda.synchronize()
        assert torch.equal(seen, data), "the stream went on before the rows were published"
    assert op.reference(compiled, view, spins_view, stream) is None
    assert int(spins.sum()) == 0, "a poll timed out"
    assert torch.equal(mailbox[:rows], data), "the wait wrote the mailbox"
    assert bool((mailbox[rows:] == SENTINEL).all())
