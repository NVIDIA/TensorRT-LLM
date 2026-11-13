# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SE_SIDE: at 8-way expert parallelism on a 212-SM device, the MegaMoE launches of the buckets above 64 up
to 128 rows per rank run the shared expert beside the routed launch on a 28-SM green-context side stream, instead of
after it (vLLM's VLLM_MEGA_MOE_SHARED_SIDE layout).

The routed generation-phase kernel launches 184 CTAs, so 28 SMs sit idle under it. The side stream holds exactly those
28 (``ops.green_partition``: a 184-SM main part, coscheduled in 4s, and the remainder, %smid-probed disjoint), so the
shared expert can never take an SM the persistent routed grid needs: its 184 CTAs always fit beside it. The routed
chain (router, route-quant, MegaMoE) keeps the main stream and its PDL chain. The shared expert forks after the
post-attention norm (its MXFP8 hand-off is published by then) and joins before the tail's add, as the co-run band's
aux-stream path does (``tail._routed_and_shared``).

The backends take the co-run band ``CORUN_BAND`` (``corun_band.install``), so those buckets run the shared expert beside
the launch everywhere: the fenceless top-k exchange and the token-comm PDL are off there, on every rank, as for any
co-running bucket. The buckets above 32 up to 64 rows keep DEP_CORUN_BAND's aux stream and pin; every other bucket stays
serial. Inside the side band a rank whose own rows are below the pinned table's 33 runs its shared expert serially on
the main stream (FlashInfer's low-M path may pick clustered tactics the 28 SMs cannot launch). On the side stream the
shared expert's hand-off GEMMs take the 2-CTA-cluster entries of the pinned table at 33..64 rows (DEP_SHARED_PIN's own
4-CTA gate_up tactic cannot launch on 28 SMs) and DEP_GEMM_PROVIDER's cuBLASLt at 65..128 rows. Every one of those is
the default tactic bit for bit, so the outputs are the serial path's.

The install is all or nothing across ranks: every rank must split its device (184 / 28, disjoint) and serve every MoE
layer through the tail's MoE forward, else no rank engages (the co-run band changes a cross-rank kernel protocol).
Built in ``cache_derived_state``, before any capture. ``TRTLLM_M31_DISABLE_DEP_SE_SIDE=1`` installs nothing: the band
stays DEP_CORUN_BAND's and those buckets stay serial. Read at import.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Callable, Iterable, Iterator, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

DEP_SE_SIDE = not m31_disabled("DEP_SE_SIDE")
#: The SMs the routed generation-phase kernel's grid needs (184 CTAs, one per SM) and the device's SM count.
MAIN_SMS = 184
NUM_SMS = 212
EP_SIZE = 8
#: Launch buckets (rows per rank) above LO up to HI that take the side stream.
SIDE_BAND = (64, 128)
#: The co-run band the backends take where the side engages: DEP_CORUN_BAND's 32..64 plus the side band.
CORUN_BAND = (32, 128)
#: A rank's own rows below this run the shared expert serially inside the side band (the pinned table's floor).
MIN_SIDE_ROWS = 33
#: The pinned-table overrides on the side stream: none, so the shared expert's GEMMs keep the table's 2-CTA entries.
SIDE_PIN_OVERRIDES: dict = {}

#: ``placement`` result for a step whose shared expert runs serially inside the side band.
SERIAL = "serial"

_ON_SIDE: ContextVar[bool] = ContextVar("dep_se_side_on_side", default=False)
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_SE_SIDE {outcome}: {detail}")


def on_side() -> bool:
    """Whether the calls made now run on the side stream (shared_mxfp8 reads it to pick side-safe tactics)."""
    return _ON_SIDE.get()


@contextlib.contextmanager
def side_calls() -> Iterator[None]:
    token = _ON_SIDE.set(True)
    try:
        yield
    finally:
        _ON_SIDE.reset(token)


def _moe_backend(layer: torch.nn.Module):
    moe = getattr(layer, "block_sparse_moe", None)
    experts = getattr(moe, "experts", None)
    return moe, getattr(experts, "backend", experts)


def layer_decline_reason(
    layer: torch.nn.Module, serves: Callable[[torch.nn.Module], bool]
) -> Optional[str]:
    """Why a decoder layer's MoE block cannot take the side stream, or None. A layer without a MoE block is not one."""
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl

    moe, backend = _moe_backend(layer)
    if moe is None:
        return None
    if getattr(moe, "shared_experts", None) is None:
        return "the MoE block has no shared expert"
    if not isinstance(backend, TrtllmCutedslMegaMoeNvfp4Impl) or backend.ep_size != EP_SIZE:
        return f"the MoE backend is not the {EP_SIZE}-way MegaMoE"
    if not serves(moe):
        return "the MoE forward is not DEP_TAIL's (the stock forward would co-run on the plain aux stream)"
    if not hasattr(backend, "shared_corun_band") or not hasattr(backend, "_lever_environ"):
        return "the MoE backend has no shared-expert co-run band on this runtime"
    try:
        from tensorrt_llm._torch.moe.fused_moe.mega_moe import megamoe_local_plan as plan
    except ImportError:
        return "the MoE backend has no shared-expert co-run band on this runtime"

    if (
        plan.SHARED_CORUN_BAND_ENV in backend._lever_environ
        or backend.shared_corun_band() is not None
    ):
        return "the backend was built with an explicit co-run band"
    if backend._lever_environ.get(plan.SHARED_CORUN_BAND_DISABLE_ENV) == "1":
        return f"the backend was built with {plan.SHARED_CORUN_BAND_DISABLE_ENV}=1"
    return None


def local_decline_reason(
    layers: Iterable[torch.nn.Module], serves: Callable[[torch.nn.Module], bool], device: int
) -> Tuple[Optional[str], Optional[object]]:
    """This rank's reason not to engage (or None) and, where it can, its device's partition."""
    if not DEP_SE_SIDE:
        return "off (TRTLLM_M31_DISABLE_DEP_SE_SIDE=1)", None
    moes = 0
    for layer in layers:
        reason = layer_decline_reason(layer, serves)
        if reason is not None:
            return reason, None
        moes += getattr(layer, "block_sparse_moe", None) is not None
    if moes == 0:
        return "no MoE layers", None
    from ..ops import green_partition

    try:
        part = green_partition.get(device, MAIN_SMS)
    except Exception as err:  # noqa: BLE001 - any driver refusal declines the item, on every rank
        return f"the green partition failed: {type(err).__name__}: {err}", None
    if part.num_sms != NUM_SMS:
        return (
            f"a {part.num_sms}-SM device (the routed grid's 184 / 28 split is for {NUM_SMS})",
            None,
        )
    if (
        not part.disjoint
        or len(part.main_sms) != MAIN_SMS
        or len(part.side_sms) != NUM_SMS - MAIN_SMS
    ):
        return f"the green partition is {part.describe()}", None
    return None, part


def _all_ranks(ok: bool) -> bool:
    from tensorrt_llm._utils import mpi_allgather

    return all(bool(x) for x in mpi_allgather(bool(ok)))


def install(
    layers: Iterable[torch.nn.Module],
    serves: Callable[[torch.nn.Module], bool],
    device: Optional[int] = None,
    all_ranks: Callable[[bool], bool] = _all_ranks,
) -> Tuple[int, Optional[str], Optional[object]]:
    """Give every MoE block the side stream where every rank can: returns (MoE blocks, the decline reason or None,
    the partition or None). A collective: every rank of the model calls it once, before any capture."""
    layers = list(layers)
    device = torch.cuda.current_device() if device is None else device
    reason, part = local_decline_reason(layers, serves, device)
    # A disabled rank still joins the agreement (contributing False) so a mixed kill switch cannot hang the others.
    if not all_ranks(reason is None):
        return 0, reason or "another rank declined", None
    count = 0
    for layer in layers:
        moe, _ = _moe_backend(layer)
        if moe is not None:
            moe._dep_se_side_stream = part.side_stream
            count += 1
    return count, None, part


def placement(moe: torch.nn.Module, backend, max_rows: Optional[int], rows: int):
    """The side stream, ``SERIAL`` or None (the step keeps its other placement) for one MoE forward."""
    stream = getattr(moe, "_dep_se_side_stream", None)
    if stream is None or max_rows is None:
        return None
    bucket = backend.megamoe_launch_bucket(max_rows)
    if (
        bucket is None
        or not SIDE_BAND[0] < bucket <= SIDE_BAND[1]
        or backend.shared_expert_serial(max_rows)
    ):
        return None
    if rows < MIN_SIDE_ROWS:
        _log_once(
            "serial",
            f"bucket {bucket}: a rank's {rows} rows run the shared expert after the routed launch",
        )
        return SERIAL
    _log_once(
        "taken",
        f"bucket {bucket}: the shared expert on the 28-SM side stream beside the routed launch",
    )
    return stream


def engagement(count: int, reason: Optional[str], part) -> str:
    if count:
        return (
            f"on {count} MoE layers at buckets {SIDE_BAND[0] + 1}..{SIDE_BAND[1]} rows "
            f"({part.describe()}, smid-probed; co-run band {CORUN_BAND[0]}:{CORUN_BAND[1]})"
        )
    return f"not installed: {reason}" if reason else "not installed"
