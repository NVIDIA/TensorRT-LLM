# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive EPLB load balancer and its per-layer handle.

Statistics and slot routing run inside the forward, so CUDA graphs capture them and no host work
is needed per step.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import socket
import threading
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pynvml
import torch
import torch.distributed as dist

import tensorrt_llm
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..moe_load_balancer import BaseMoeLoadBalancer
from . import calibrate
from .calibrate import CALIBRATION
from .policy import (
    ExpertGeometry,
    ExpertStatistics,
    TimeModel,
    active_slots,
    build_route_tables,
    statistics_buffers,
)
from .runtime import GATHER_SLICE_S, AdaptiveRuntime, DecisionConfig, DecisionProcess
from .transport import (
    PeerSlotTransport,
    _plan_layer,
    find_backends,
    migrate_model_weights,
    peer_copy_gap,
    prepare_layers,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    from .transport import AdaptiveEplbBackend


def time_layer_copy(
    transport: PeerSlotTransport, layer: int, geometry: ExpertGeometry, stream: torch.cuda.Stream
) -> float:
    """Milliseconds per replica of the copies of one layer's install, at the slowest rank;
    collective over EP.

    Every rank copies into its free slots of ``layer`` as many experts as an install places on a
    GPU, each from the home slots of a rank half the EP group away. The ranks copy at the same
    time, once every rank's blocks hold their weights, and the same copies run once untimed
    first. No route table points at a free slot, so the copies change no result.
    """
    rank, size, homes = transport.ep_rank, geometry.ep_size, geometry.local_experts
    replicas = active_slots(geometry) // size - homes
    experts = [
        (layer, homes + j, (rank + size // 2 + j) % size, j % homes) for j in range(replicas)
    ]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    transport.allgather(None)
    for _ in range(2):
        transport.copy(experts, stream, start, end)
        end.synchronize()
    return max(transport.allgather(start.elapsed_time(end))) / max(1, replicas)


TERMS = ("per_active_slot_us", "route_us")  # a calibration file's optional costs per bucket (us)


def nvlink_domain_error(comm: object) -> tuple[str | None, int]:
    """Why the EP group of ``comm`` is not one NVLink domain (``None`` when it is), and its hosts.

    Every GPU needs its NVLink up (``MnnvlMemory.supports_mnnvl``); a group over several hosts
    needs one NVLink fabric (MNNVL): the same NVML fabric cluster and clique on every GPU.
    Collective over ``comm``.
    """
    up, fabric = False, None
    try:  # every rank reaches the vote below: a failure here only marks this GPU
        up = MnnvlMemory.supports_mnnvl()
        pynvml.nvmlInit()
        info = pynvml.c_nvmlGpuFabricInfoV_t()  # pynvml structures take no keyword arguments
        info.version = pynvml.nvmlGpuFabricInfo_v2
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
        if info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED:
            fabric = (bytes(info.clusterUuid).hex(), int(info.cliqueId))
    except Exception:  # noqa: BLE001 - whatever fails, this rank still votes
        pass
    views = comm.allgather((up, socket.gethostname(), fabric))
    hosts = len({host for _, host, _ in views})
    if not all(up for up, _, _ in views):
        return "a GPU of the EP group has no NVLink with all links up", hosts
    if hosts > 1 and (views[0][2] is None or len({f for _, _, f in views}) > 1):
        return "the EP group spans hosts that share no NVLink fabric (MNNVL)", hosts
    return None, hosts


def calibration_identity(
    backend: object, ep_size: int, hosts: int, tokens: Sequence[int], num_slots: int
) -> dict:
    """What a calibration holds for: the GPU, the EP group, its slots and the decode kernels."""
    return dict(
        device=torch.cuda.get_device_name(),
        sm=get_sm_version(),
        sm_count=torch.cuda.get_device_properties().multi_processor_count,
        ep_size=int(ep_size),
        hosts=int(hosts),
        num_slots=int(num_slots),
        **backend.decode_calibration_identity(list(tokens)),
    )


def load_time_model(
    backend: object,
    path: str,
    expert_bytes: int,
    identity: dict,
    tokens: Sequence[int],
    text: str | None = None,
) -> tuple[TimeModel | None, str | None]:
    """The decode time model of ``backend`` from the calibration file at ``path``, or ``text`` that
    ``path`` names (per CUDA graph bucket a scale and, where it gives them, :data:`TERMS`);
    ``(None, why)`` when it does not hold for the ``identity`` of this run or a decode bucket of
    ``tokens`` tokens per GPU. ``ValueError`` when the file is malformed."""
    doc = json.loads(Path(path).read_text() if text is None else text)
    if not isinstance(doc, dict) or (doc.get("schema"), doc.get("version")) != CALIBRATION:
        return None, f"{path} is not a calibration file of version {CALIBRATION[1]}"
    parts = [doc.get(key) for key in ("identity", "rates", "measurement")]
    buckets = doc.get("buckets")
    if not (
        all(isinstance(part, dict) for part in parts)
        and isinstance(parts[0].get("tactics"), dict)
        and isinstance(buckets, list)
        and buckets
        and all(isinstance(bucket, dict) for bucket in buckets)
    ):
        raise ValueError(f"{path} does not have the layout of a calibration file")

    def real(x, low=0.0):  # a finite number, above 0 or at least low
        return type(x) in (int, float) and (x > 0 or x >= low) and x < math.inf

    reals = {f"rates.{k}": parts[1].get(k) for k in ("hbm_bytes_per_s", "fp4_dense_flops_per_s")}
    bad = [key for key, x in reals.items() if not real(x, math.inf)]
    sizes = {f"buckets[{i}].tokens": b.get("tokens") for i, b in enumerate(buckets)}
    bad += [key for key, x in sizes.items() if type(x) is not int or x <= 0]
    if bad:  # the rates positive and the tokens integers, or the file is malformed
        raise ValueError(f"{path}: {', '.join(bad)} out of range")
    for b in buckets:  # a scale not positive or a cost negative: no model there, as when refused
        if not (real(b.get("scale"), math.inf) and all(real(b[k]) for k in TERMS if k in b)):
            b.update(held=False, scale=1.0, **{k: 0.0 for k in TERMS if k in b})
    theirs, per_bucket = doc["identity"], ("tactics", "launch_buckets", "tile_rows")
    wrong = [key for key in identity if key not in per_bucket and theirs.get(key) != identity[key]]
    bucket_of = {b["tokens"]: b for b in buckets}
    wrong += [f"bucket {t}" for t in tokens if t not in bucket_of]
    wrong += [
        f"{key} of bucket {t}"
        for key in per_bucket
        for t in tokens
        if t in bucket_of
        and (theirs.get(key) or {}).get(str(t)) != identity.get(key, {}).get(str(t))
    ]
    if wrong:
        return None, f"{path} does not hold for this run: {', '.join(wrong)} differ or are missing"
    rates = (doc["rates"]["hbm_bytes_per_s"], doc["rates"]["fp4_dense_flops_per_s"])
    known = identity.get("tile_rows", {})
    tiles = {t: b.get("tile_rows") or known.get(str(t)) for t, b in bucket_of.items()}
    if not all(tiles.values()):
        raise ValueError(
            f"{path}: no tile rows for buckets {[t for t, n in tiles.items() if not n]}"
        )
    fields = backend.decode_time_model(expert_bytes, rates, tile_rows=tiles[max(tiles)])
    own = dict(zip(TERMS, (fields["per_active_slot_us"], 0.0)))  # a bucket without them keeps these
    by_rows = tuple(  # a bucket with its measured read cost solves the shares by time
        (t * identity["top_k"], float(b["scale"]), *(float(b.get(k, x)) for k, x in own.items()))
        + (TERMS[0] in b, tiles[t], b.get("held", True) is not False)
        for t, b in sorted(bucket_of.items())
    )
    if doc["measurement"].get("trtllm_version") != tensorrt_llm.__version__:
        logger.warning(
            f"{path} was measured with TensorRT LLM {doc['measurement'].get('trtllm_version')}"
        )
    _, scale, read, route, timed, *_ = by_rows[-1]
    fields.update(scale=scale, per_active_slot_us=read)
    return TimeModel(**fields, scales=by_rows, route_us=route, time_shares=timed), None


def ep_gather(group: dist.ProcessGroup) -> Callable:
    """``gather(send, rows, leave)`` on ``group``, waiting in slices until done or ``leave`` is set."""

    def gather(send: torch.Tensor, rows: list[torch.Tensor], leave: threading.Event) -> None:
        work = dist.all_gather(rows, send, group=group, async_op=True)
        while not work.is_completed():
            try:
                work.wait(timeout=timedelta(seconds=GATHER_SLICE_S))
            except RuntimeError:  # the slice timed out, or the all-gather failed
                if leave.is_set() and not work.is_completed():
                    raise
        work.wait()  # raises the all-gather's error

    return gather


NO_LAYER = "no MoE layer runs on an implementation that supports adaptive EPLB"
"""Why adaptive EPLB stays off on a model whose MoE layers all run outside the balancer."""


class AdaptiveMoeLayerLoadBalancer:
    """Graph-safe layer routes and statistics, without standard host migration hooks.

    Args:
        manager: The model-level balancer that owns this layer.
        layer_idx: The MoE layer's index in the model.
        expert_count: The layer's routed experts.
        slot_count_per_rank: Weight slots per EP rank, home copies included.
        repeated_count: Calls of the layer per forward (MTP repeats).
    """

    produces_replica_plan = False
    stages_routes = True

    def __init__(
        self,
        manager: AdaptiveMoeLoadBalancer,
        *,
        layer_idx: int,
        expert_count: int,
        slot_count_per_rank: int,
        repeated_count: int = 1,
    ) -> None:
        self.manager = manager
        self.layer_idx = int(layer_idx)
        self.expert_count = int(expert_count)
        self.num_slots = int(slot_count_per_rank) * manager.ep_size
        self.repeated_count = int(repeated_count)
        self.statistics: ExpertStatistics | None = None
        self._tables: list[torch.Tensor] = []
        self._slot_dtype = torch.int32
        self._route_staging: Callable[[int, int, torch.device], object | None] | None = None
        self._pending: tuple[torch.Tensor, bool] | None = None
        self._route_into = None
        # Local token rows of the forward the calls so far covered (a chunked forward's offset).
        self._rows_seen = 0

    def get_repeat_count(self) -> int:
        return self.repeated_count

    def is_static_routing(self) -> bool:
        return False

    def is_dynamic_routing(self) -> bool:
        """False: no host statistics, host weight migration or stage hooks."""
        return False

    def records_statistics(self) -> bool:
        """True: the MoE modules pass every forward's routed experts to the statistics hooks."""
        return True

    def need_load_shared_weights(self) -> bool:
        return False

    def get_load_expert_ids(self) -> list[int]:
        local = self.expert_count // self.manager.ep_size
        start = self.manager.ep_rank * local
        return list(range(start, start + local))

    def home_layout(self) -> list[int]:
        """Every expert on its home slot (:meth:`~.policy.ExpertGeometry.home_layout`)."""
        geometry = ExpertGeometry(self.expert_count, self.num_slots, self.manager.ep_size)
        return geometry.home_layout().tolist()

    def set_initial_weight_assignments(self, initial_weight_assignments: Sequence[int]) -> None:
        """Nothing to set: every layer starts at its home layout."""

    def prepare_device_state(
        self,
        slot_dtype: torch.dtype,
        device: torch.device | str,
        route_staging: Callable[[int, int, torch.device], object | None] | None = None,
    ) -> None:
        """Bind the layer's statistics and route tables on ``device`` before the first forward.

        ``slot_dtype`` is the dtype the backend reads slot ids in and ``route_staging`` its
        ``route_staging`` method, if it has one.
        """
        manager = self.manager
        flag = manager.recording_flag(device)
        self.statistics = ExpertStatistics.of(manager.statistics(device), self.layer_idx, flag)
        tables = manager.route_tables(device)[self.layer_idx]
        self._tables = tables.split([self.expert_count] * 2 + [self.num_slots] * 2)
        self._table_row = tables
        self._off = torch.zeros(1, dtype=torch.int32, device=tables.device)
        self._slot_dtype = slot_dtype
        self._route_staging = route_staging
        self._route_into = torch.ops.trtllm.adaptive_eplb_route_into

    def update_local_statistic(
        self,
        local_raw_expert_ids: torch.Tensor,
        is_first_stage: bool,
        is_last_stage: bool,
        allreduce: object = None,
    ) -> None:
        """Keep the routes of a chunk or repeat; the next :meth:`route` of the same ids adds them to
        the statistics.

        Forwards that are not recorded (context steps, warmup) record nothing, except under CUDA
        graph capture, where the device flag gates every replay.
        """
        statistics = self.statistics
        pending, self._pending = self._pending, None
        if pending is not None:
            statistics.record(*pending)
        if self.manager.recording or (
            statistics.rows.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            self._pending = (self._counted(local_raw_expert_ids, is_first_stage), is_first_stage)

    def _counted(self, ids: torch.Tensor, first: bool) -> torch.Tensor:
        """The rows of ``ids`` (a call's ``[tokens, top_k]``) that count: all of them, or in a batch
        with context requests those of the generation tokens (``manager.counted_rows``), placed
        after the rows of the forward's earlier chunks. A slice, which :meth:`route` records apart.
        """
        rows = self.manager.counted_rows
        if first:
            self._rows_seen = 0
        start, n = self._rows_seen, ids.shape[0]
        self._rows_seen += n
        if rows is None:
            return ids
        return ids[min(max(rows[0] - start, 0), n) : min(max(rows[1] - start, 0), n)]

    update_statistic_with_local_ids = update_local_statistic

    def router_route(self, first: bool) -> tuple:
        """The adaptive route for a backend whose dispatch router maps and counts this layer's routes.

        Args:
            first: Whether this is the layer's first call in the forward, which counts the forward.

        Returns:
            None in a batch with context requests (only its generation rows count, which the
            separate launches record), else
            ``(table, rows, forwards, enabled, first)``: the layer's packed route table row, its
            statistics rows and forward count, the recording flag where :meth:`update_local_statistic`
            would keep the routes (recording, or under CUDA graph capture, where the flag gates every
            replay; else a flag that is never set), and ``first``.
        """
        if self.manager.counted_rows is not None:
            return None  # a batch with context requests counts part of its rows: separate launches
        pending, self._pending = self._pending, None
        if pending is not None:
            self.statistics.record(*pending)
        statistics = self.statistics
        counted = self.manager.recording or (
            statistics.rows.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        enabled = statistics.enabled if counted else self._off
        return self._table_row, statistics.rows, statistics.forwards, enabled, bool(first)

    def route(
        self,
        token_selected_experts: torch.Tensor,
        offset_by_ep_rank: bool = False,
        *,
        overlap_fn: Callable[[], None] | None = None,
        token_final_scales: torch.Tensor | None = None,
        quantize: tuple | None = None,
    ) -> torch.Tensor:
        """Map every routed expert to a slot, in one launch that also adds the kept routes to the
        statistics.

        Ids outside the experts map to ``num_slots``. With ``token_final_scales`` (``[tokens,
        top_k]`` float32) and a backend that stages routes, the launch writes the slots and weights
        into the backend's staging and returns a view of it, which the backend does not stage again.
        Under the staging's live-row word the rows of padding requests are staged as -1 and 0 and
        not counted, as the backend's kernel drops them; other launches count every row.
        With ``quantize`` (``x, global_scale, x_fp4, x_sf``), the launch also quantizes ``x``.

        Args:
            token_selected_experts: ``[tokens, top_k]`` expert ids.
            offset_by_ep_rank: Unused; kept for the balancer interface.
            overlap_fn: Work to run while the route launch is in flight.
            token_final_scales: Gate weights to stage beside the slots, as described above.
            quantize: Activation to quantize in the same launch, as described above.

        Returns:
            The slot ids, ``[tokens, top_k]``, or a view of the backend's staging.
        """
        ids = token_selected_experts
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] is not ids:
            self.statistics.record(*pending)
            pending = None
        staging = None
        if (
            self._route_staging is not None
            and token_final_scales is not None
            and token_final_scales.dtype == torch.float32
            and token_final_scales.is_contiguous()
            and token_final_scales.shape == ids.shape
        ):
            staging = self._route_staging(*ids.shape, ids.device)
        if staging is None:
            out = torch.empty_like(ids, dtype=self._slot_dtype)
            weights = (None, None)
        else:
            out = staging.topk_idx
            weights = (token_final_scales, staging.topk_weights)
        stats = (None, None, None, True) if pending is None else (*self.statistics, pending[1])
        self._route_into(
            ids,
            *self._tables,
            self.manager.ep_rank,
            out,
            *weights,
            *stats,
            None if staging is None else staging.live_row_word,
            *(quantize or ()),
        )
        if overlap_fn is not None:
            overlap_fn()
        return out[: ids.shape[0]]


class AdaptiveMoeLoadBalancer(BaseMoeLoadBalancer):
    """Model-level adaptive EPLB balancer of rank ``ep_rank`` of ``ep_size``.

    Statistics and routes live on the device, so a forward outside ``MoeLoadBalancerIterContext``
    (the first replay after a graph capture) is recorded like any other.

    Args:
        ep_rank: This rank's index in the expert-parallel group.
        ep_size: The expert-parallel group size.
        config: The ``moe_load_balancer`` configuration of mode ``adaptive``.
        mapping: The model's parallel mapping.
    """

    def __init__(
        self, *, ep_rank: int, ep_size: int, config: MoeAdaptiveLoadBalancerConfig, mapping: Mapping
    ) -> None:
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.config = config
        self.mapping = mapping
        self.num_slots = int(config.num_slots)
        self.num_local_slots = self.num_slots // self.ep_size
        self.layers: list[AdaptiveMoeLayerLoadBalancer] = []
        self.next_layer_repeated_count: int | None = None
        self.in_iter = False
        self.is_shutdown = False
        self._previous_balancer: BaseMoeLoadBalancer | None = None
        self.enable_statistic = False
        self.enable_update_weights = False
        self.transport: PeerSlotTransport | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self.runtime: AdaptiveRuntime | None = None
        # Why adaptive EPLB stays off on this platform (:meth:`prepare_runtime`), if it does.
        self.off_reason: str | None = None
        # Whether the forwards now running record their routes.
        self.recording = False
        # The local token rows ``[start, stop)`` that count in a batch that also runs context
        # requests (its generation tokens); None when every row counts.
        self.counted_rows: tuple[int, int] | None = None
        # Whether a window is open: the device flag counts only while this and ``recording`` hold.
        # Without a runtime no window is scheduled and every recorded forward counts; the runtime
        # closes and opens windows on its own schedule (``on_window``).
        self.window_open = True
        self._recording_flag: torch.Tensor | None = None
        self._flag_value = 0
        self._route_tables: torch.Tensor | None = None
        self._statistics: dict[str, torch.Tensor] | None = None
        # What before_capture calibrates and starts the runtime with, if prepare_runtime found none.
        self._pending: tuple | None = None

    def is_static_routing(self) -> bool:
        return False

    def set_repeated_for_next_layer(self, repeated_count: int) -> None:
        if repeated_count <= 0:
            raise ValueError("repeat count must be positive")
        self.next_layer_repeated_count = int(repeated_count)

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> AdaptiveMoeLayerLoadBalancer:
        """Register one MoE layer and return its handle; ``ValueError`` if it does not fit the config."""
        self.config.validate_expert_capacity(expert_count)
        if int(slot_count_per_rank) != self.num_local_slots:
            raise ValueError(
                f"layer slot count {slot_count_per_rank} differs from the configured "
                f"{self.num_local_slots} local slots"
            )
        layer = AdaptiveMoeLayerLoadBalancer(
            self,
            layer_idx=len(self.layers),
            expert_count=expert_count,
            slot_count_per_rank=slot_count_per_rank,
            repeated_count=self.next_layer_repeated_count or 1,
        )
        self.next_layer_repeated_count = None
        self.layers.append(layer)
        return layer

    def _nothing_before_runtime(self, *args: object, **kwargs: object) -> None:
        """Nothing to do: the slot weights move in :meth:`prepare_runtime`."""

    register_weight_slots_after_to_cuda = _nothing_before_runtime
    finalize_model = _nothing_before_runtime
    set_warm_up_iter_count = _nothing_before_runtime

    def prepare_runtime(self, model: torch.nn.Module, decode_tokens: Sequence[int] = ()) -> None:
        """Check the deployment, move the slot weights into MNNVL memory, time one layer's install
        copy (:func:`time_layer_copy`) and start the runtime under the backend's time model.
        Collective over EP; runs before the KV cache is sized.

        ``decode_tokens`` are the tokens per GPU per forward of every decode bucket the deployment
        runs. Every rank checks, without a kernel launch, that it balances a layer, that the EP
        group is one NVLink domain (:func:`nvlink_domain_error`), that it can copy experts from
        every EP rank (:meth:`platform_gap`) and, under ``time_model`` ``file``, that the file holds
        for this run (:func:`load_time_model`); the ranks vote, and where any check fails on any
        rank, or the ranks balance different layers, every rank logs why and keeps adaptive EPLB
        off: the layers serve their home layouts and record no statistics. Under ``auto``, rank 0's
        cached calibration of this run serves every rank; without one, :meth:`before_capture`
        calibrates the time model and starts the runtime.
        """
        backends = find_backends(model, self.layers)
        first = self.layers[0].layer_idx if self.layers else None
        backend = backends.get(first)
        device = torch.device("cuda", torch.cuda.current_device())
        prepare_layers(backends, self.layers, device)
        transport = PeerSlotTransport(self.mapping)
        why, hosts = nvlink_domain_error(transport)
        mode, path, digest = self.config.time_model, self.config.time_model_file, ""
        time_model, identity, tokens = None, {}, list(decode_tokens)
        try:  # this rank's checks; whatever they raise becomes its vote, so every rank votes
            why = why or (self.platform_gap(backend) if self.layers else NO_LAYER)
            if why is None and mode == "off":
                why = "time_model is off, so no time model prices the installs"
            if why is None:
                expert_bytes = _plan_layer(first, backend).expert_bytes
            if why is None and mode == "file":  # auto: the warmup's launches name the tactics
                identity, tokens = self._identity(backend, hosts, tokens)
                time_model, why = load_time_model(backend, path, expert_bytes, identity, tokens)
                if time_model is not None:
                    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except Exception as error:  # noqa: BLE001 - the vote carries any error to every rank
            why = f"{type(error).__name__}: {error}"
            logger.warning(f"adaptive EPLB start-up check failed on EP rank {self.ep_rank}: {why}")
        votes = transport.allgather((why, digest, len(self.layers), calibrate.digest(identity)))
        stopped = [f"{gap} (EP rank {rank})" for rank, (gap, *_) in enumerate(votes) if gap]
        if not stopped and len({file for _, file, *_ in votes}) > 1:
            stopped = [f"the EP ranks read different files at {path}"]
        if not stopped and len({v[2:4] for v in votes}) > 1:
            stopped = ["the EP ranks balance different MoE layers or decode kernels"]
        if stopped:
            self.off_reason = stopped[0]
            logger.warning(
                f"adaptive EPLB is off: {self.off_reason}; the MoE layers keep their home layouts"
            )
            return
        self.transport = transport
        self.copy_stream = migrate_model_weights(backends, transport, device)
        geometry = ExpertGeometry(self.layers[0].expert_count, self.num_slots, self.ep_size)
        copy_ms = time_layer_copy(transport, first, geometry, self.copy_stream)
        logger.info(
            f"adaptive EPLB balances {len(self.layers)} MoE layers; one layer's install copy "
            f"takes {copy_ms:.3f} ms per replica at start-up"
        )
        if time_model is None:  # auto: cached or calibrated in the CUDA graph warmup
            self._pending = (backend, hosts, expert_bytes, tokens, copy_ms)
            self.off_reason = "its time model comes in the CUDA graph warmup, not run yet"
            logger.info(f"adaptive EPLB stays off until {self.off_reason}")
            return
        self.start_runtime(time_model, self.mapping.gpus_per_node, copy_ms)

    def _identity(self, backend: object, hosts: int, tokens: list[int]) -> tuple[dict, list[int]]:
        """The calibration identity of this run and its decode buckets that one launch takes."""
        identity = calibration_identity(backend, self.ep_size, hosts, tokens, self.num_slots)
        return identity, [t for t in tokens if str(t) in identity["tactics"]]

    def before_capture(self) -> None:
        """Under ``time_model`` ``auto``, take rank 0's cached calibration of this run, else calibrate
        the decode time model (:func:`~.calibrate.calibrate`) and cache it, and start the runtime on
        it; or log why adaptive EPLB stays off. Once, between the CUDA graph warmup's first pass,
        which launched every decode bucket (autotuned or not, their tactics are known now), and its
        capture. Collective over EP; nothing to do otherwise."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        backend, hosts, expert_bytes, tokens, copy_ms = pending
        identity, path, cached, model, why = {}, None, None, None, None
        try:
            identity, tokens = self._identity(backend, hosts, tokens)
            path = calibrate.cache_path(self.config, identity)
            cached = calibrate.read_cache(path) if self.ep_rank == 0 else None
        except Exception as error:  # noqa: BLE001 - voted below
            why = f"{type(error).__name__}: {error}"
        votes = self.transport.allgather((why, calibrate.digest(identity), cached))
        why = next((f"{gap} (EP rank {rank})" for rank, (gap, *_) in enumerate(votes) if gap), None)
        if why is None and len({digest for _, digest, _ in votes}) > 1:
            why = "the EP ranks resolve different decode kernels"
        if why is None and votes[0][2] is not None:  # one text, one identity: one outcome
            try:
                model, stale = load_time_model(
                    backend, str(path), expert_bytes, identity, tokens, votes[0][2]
                )
            except Exception as error:  # noqa: BLE001 - a file that does not parse is stale
                stale = f"{type(error).__name__}: {error}"
            logger.info(
                f"adaptive EPLB time model cached at {path}" + (f": {stale}" if stale else "")
            )
        if why is None and model is None:
            model, why = self._calibrate(backend, identity, expert_bytes, tokens, path)
        if why is not None:
            self.off_reason = f"no decode time model: {why}"
            logger.warning(f"adaptive EPLB is off: {self.off_reason}")
            return
        self.off_reason = None
        self.start_runtime(model, self.mapping.gpus_per_node, copy_ms)

    def _calibrate(self, backend, identity, expert_bytes, tokens, path) -> tuple:
        """``(time model, None)`` calibrated and cached, or ``(None, why)``. Collective over EP."""
        planes = [backend.get_parameter(name) for name, _ in backend.slot_weight_planes()]
        state = [self._route_tables, *(self._statistics or {}).values(), self._recording_flag]
        state = [t for t in state + planes if t is not None]
        before, model = calibrate.snapshot(state), None
        try:
            text, why, times = calibrate.calibrate(
                backend,
                tokens,
                identity=identity,
                expert_bytes=expert_bytes,
                num_experts=self.layers[0].expert_count,
                num_slots=self.num_slots,
                rank=self.ep_rank,
                gather=self.transport.allgather,
                settings=calibrate.Settings.of(self.config),
            )
            if times and path is not None and self.ep_rank == 0:  # the designs' times, to replay
                calibrate.write_cache(path.with_suffix(".times.json"), json.dumps(times))
            for b in json.loads(text)["buckets"] if text and self.ep_rank == 0 else ():
                keys = ("scale", "se", "holdout_z", "rmse_us", "median_us", "designs", "held")
                line = ", ".join(f"{k} {b[k]}" for k in keys + ("per_active_slot_us",))
                (logger.info if b["held"] else logger.warning)(
                    f"adaptive EPLB bucket {b['tokens']}: {line}"
                )
            if why is None:
                model, why = load_time_model(
                    backend, str(path), expert_bytes, identity, tokens, text
                )
        except Exception as error:  # noqa: BLE001 - voted below
            why = f"{type(error).__name__}: {error}"
        changed = not calibrate.unchanged(before, calibrate.snapshot(state))
        votes = self.transport.allgather((changed, None if model is not None else why))
        if any(c for c, _ in votes):  # every rank raises, none waits in a collective
            changed = [rank for rank, (c, _) in enumerate(votes) if c]
            raise RuntimeError(f"the calibration changed the weights or balancer state: {changed}")
        why = next((f"{gap} (EP rank {rank})" for rank, (_, gap) in enumerate(votes) if gap), None)
        if why is not None:
            return None, f"the calibration failed: {why}"
        failed = calibrate.write_cache(path, text) if self.ep_rank == 0 else None
        if failed:
            logger.warning(f"adaptive EPLB keeps its calibration in memory only: {failed}")
        return model, None

    def platform_gap(self, backend: AdaptiveEplbBackend) -> str | None:
        """Why the installs cannot run on this rank, or ``None``: they copy experts from every EP
        rank (:func:`~.transport.peer_copy_gap`). Whether a time model prices ``backend`` on this
        GPU is :func:`load_time_model`'s check."""
        return peer_copy_gap(self.mapping)

    def start_runtime(
        self, model: TimeModel, ranks_per_node: int, copy_ms: float | None = None
    ) -> None:
        """Start the windows, the decision process and the installs, the decisions taking
        ``copy_ms`` per replica copied among the copies they time. Collective over EP."""
        if not dist.is_initialized():
            raise RuntimeError(
                "adaptive EPLB needs torch.distributed: its statistics gathers run on a gloo group "
                "of the EP ranks, and a MoE backend that supports it initializes the default group"
            )
        geometry = ExpertGeometry(self.layers[0].expert_count, self.num_slots, self.ep_size)
        homes = np.array([layer.home_layout() for layer in self.layers], dtype=np.int64)
        rows = self.config.rows_per_expert
        decision = DecisionProcess(
            DecisionConfig(
                geometry, model, homes, self.ep_rank, ranks_per_node, rows, copy_ms=copy_ms
            )
        )
        # The runtime's worker thread gathers on a group of its own: a group the main thread also
        # uses would interleave their collectives differently on different ranks. Its window and
        # copy gathers alternate on every rank, so one group carries both.
        ranks = sorted(self.transport.allgather(dist.get_rank()))
        group = dist.new_group(ranks, backend="gloo", use_local_synchronization=True)

        device = torch.device("cuda", torch.cuda.current_device())
        self.runtime = AdaptiveRuntime(
            self.layers,
            self.transport,
            self.copy_stream,
            decision,
            ep_gather(group),
            device,
            self.route_tables(device),
            self.statistics(device),
            on_window=self._set_window_open,
        )

    def statistics(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Every layer's statistics in contiguous buffers (:func:`~.policy.statistics_buffers`)."""
        if self._statistics is None:
            self._statistics = statistics_buffers(
                len(self.layers), self.layers[0].expert_count, device
            )
        return self._statistics

    def route_tables(self, device: torch.device | str) -> torch.Tensor:
        """The route tables of all layers, one int32 row of ``2 * (experts + slots)`` per layer.

        Every layer starts with the table of its home layout.
        """
        if self._route_tables is None:
            layer = self.layers[0]
            home = build_route_tables(layer.home_layout(), None, layer.expert_count, self.num_slots)
            self._route_tables = torch.tensor(
                [home.packed()] * len(self.layers), dtype=torch.int32, device=device
            )
        return self._route_tables

    def recording_flag(self, device: torch.device | str) -> torch.Tensor:
        """The ``[1]`` int32 flag the statistics read: set for generation steps outside warmup
        while the runtime's window is open (:meth:`_sync_recording_flag`)."""
        if self._recording_flag is None:
            self._recording_flag = torch.full(
                (1,), self._flag_value, dtype=torch.int32, device=device
            )
        return self._recording_flag

    def observe_scheduled_batch(
        self,
        num_generation_requests: int,
        generation_only: bool,
        is_warmup: bool,
        generation_progress: Callable[[int, int], tuple[int, int, int]],
        generation_rows: tuple[int, int] | None = None,
    ) -> None:
        """Record the statistics of generation steps outside warmup only, and pass the generation
        requests' progress to the runtime when it asks for it.

        A batch that also runs context requests records the rows of its generation tokens only,
        ``generation_rows`` (``[start, stop)`` of the forward's local tokens); without them it
        records nothing. Skipping such batches whole left a window without a single forward on a
        rank that admits a request at every call, and the window's other ranks lost it too.
        """
        mixed = not generation_only
        counted = generation_rows is not None and generation_rows[1] > generation_rows[0]
        recording = num_generation_requests > 0 and not is_warmup and (counted or not mixed)
        if self._pending is not None and not is_warmup:  # serving began, no warmup calibrated it
            self._pending, self.off_reason = None, "no CUDA graph warmup calibrated its time model"
            logger.warning(f"adaptive EPLB is off: {self.off_reason}")
        if self.off_reason is not None or getattr(self.runtime, "failure", None) is not None:
            recording = False  # nothing reads the statistics of adaptive EPLB that is off
        self.recording = recording
        self.counted_rows = None
        if recording and mixed:
            self.counted_rows = (int(generation_rows[0]), int(generation_rows[1]))
        self._sync_recording_flag()
        if self.runtime is not None and self.runtime.wants_request_progress:
            self.runtime.add_request_progress(num_generation_requests, generation_progress)

    def _set_window_open(self, open_: bool) -> None:
        self.window_open = bool(open_)
        self._sync_recording_flag()

    def _sync_recording_flag(self) -> None:
        """Set the device flag the statistics ops and the router read to whether the forwards now
        running count: generation steps outside warmup, while the runtime's window is open. A
        stream-ordered write at a fixed address, so captured graphs read it at every replay."""
        value = int(self.recording and self.window_open)
        if value != self._flag_value:
            self._flag_value = value
            if self._recording_flag is not None:
                self._recording_flag.fill_(value)

    def iteration_stats(self) -> dict[str, int] | None:
        """The runtime's counters (:meth:`~.runtime.AdaptiveRuntime.counters`) once it runs."""
        return None if self.runtime is None else self.runtime.counters()

    def start_iter(self) -> None:
        if self.in_iter:
            raise RuntimeError("already in a MoE load-balancer iteration")
        self.in_iter = True
        if self.runtime is not None and self.enable_update_weights:
            if not torch.cuda.is_current_stream_capturing():
                self.runtime.boundary()

    def end_iter(self) -> None:
        if not self.in_iter:
            raise RuntimeError("not in a MoE load-balancer iteration")
        self.in_iter = False

    def shutdown(self, local_safe: bool = True) -> bool:
        """Stop the runtime once every EP rank is safe, else return ``False``. Collective over EP."""
        if self.is_shutdown:
            return True
        local_safe = bool(local_safe and not self.in_iter)
        gathers = 0 if self.runtime is None else self.runtime.gathers_posted
        if self.transport is not None:
            votes = self.transport.allgather((local_safe, gathers))  # and the most all-gathers
            local_safe, gathers = all(v[0] for v in votes), max(v[1] for v in votes)
        if not local_safe:
            return False
        if self.runtime is not None:
            self.runtime.close(gathers)
        if self.transport is not None:
            self.copy_stream.synchronize()
        self.is_shutdown = True
        return True
