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
"""Preflight checks for the MegaMoE DeepGEMM NVLink SymmBuffer rendezvous.

Why this exists
---------------
``get_symm_buffer_for_mega_moe`` (vendored DeepGEMM, ``deep_gemm/mega``)
runs four collective-ish steps back to back with **no synchronize between
them**::

    symm_mem.empty(num_bytes)  # 1  allocate
    symm_mem.rendezvous(buffer)  # 2  exchange IPC/fabric handles
    buffer.zero_()  # 3  first write through the symmetric window
    group.barrier()
    synchronize()  # 4  first sync -- where any fault surfaces

Because step 4 is the first synchronization point, a fault raised in steps
1-3 is reported *there*, as an opaque error on a line that did nothing but
wait. ``cudaErrorNvlinkUncorrectable`` at step 4 therefore carries no
indication of which step broke, which rank broke, or whether the CUDA
context was already poisoned on entry.

This module adds the two things that were missing:

1. A **drain** before the rendezvous, so a pre-existing (sticky) CUDA fault
   inherited from weight loading or an earlier layer is attributed to where
   it came from instead of to the SymmBuffer.
2. The **capability gate** every sibling NVLink path already has.
   ``MnnvlMemory.supports_mnnvl()`` gates ``deep_ep``, ``nvlink_one_sided``,
   ``nvlink_two_sided``, ``nvlink_two_sided_flashinfer`` and
   ``_torch/distributed/ops.py``; the ``enable_symm_mem_for_group`` +
   1-byte multicast probe pattern gates ``symm_mem_allreduce``,
   ``symm_mem_allgather`` and ``cute_dsl_megamoe_custom_op``. The DeepGEMM
   SymmBuffer path had neither.

Default behaviour is **diagnostic only**: every check is reported on one
greppable line and nothing changes the control flow, so enabling this
cannot turn a currently-passing configuration red. Set
``TLLM_MEGA_MOE_SYMM_STRICT=1`` to make a failed capability check raise
instead of warn.

See NVBug 6713426.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

# Single greppable token for log post-processing across ranks.
_TAG = "MEGAMOE_SYMM_PREFLIGHT"


def _strict() -> bool:
    return os.environ.get("TLLM_MEGA_MOE_SYMM_STRICT", "0") == "1"


def _fabric_identity(dev_id: int) -> Dict[str, Any]:
    """Best-effort NVML GPU fabric identity for this device.

    Returns a dict that always has a ``fabric`` key. Every lookup is
    ``getattr``-probed rather than assumed: the fabric-info API name and
    its field names vary across NVML/pynvml versions, and this must never
    be the thing that breaks a run. A missing API is reported explicitly
    (``fabric=no-api``) rather than silently omitted, so an unusable probe
    is visible in the log instead of looking like a clean result.

    Only *equality across ranks* is interpreted downstream -- no field
    encoding is decoded here, so a version difference cannot make this
    produce a wrong verdict, only an unavailable one.
    """
    out: Dict[str, Any] = {"fabric": "unknown"}
    try:
        import pynvml
    except ImportError:
        out["fabric"] = "no-pynvml"
        return out

    try:
        try:
            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001 - already-initialized is fine
            pass
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
    except Exception as e:  # noqa: BLE001
        out["fabric"] = f"nvml-error:{type(e).__name__}"
        return out

    fn = None
    for name in ("nvmlDeviceGetGpuFabricInfoV", "nvmlDeviceGetGpuFabricInfo"):
        fn = getattr(pynvml, name, None)
        if fn is not None:
            out["fabric_api"] = name
            break
    if fn is None:
        out["fabric"] = "no-api"
        return out

    try:
        info = fn(handle)
    except Exception as e:  # noqa: BLE001
        out["fabric"] = f"query-failed:{type(e).__name__}"
        return out

    out["fabric"] = "ok"
    for field in ("clusterUuid", "cliqueId", "state", "status", "healthMask"):
        val = getattr(info, field, None)
        if val is None:
            continue
        if isinstance(val, (bytes, bytearray)):
            val = val.hex()
        out[field] = val
    return out


# Outcomes of stage 2 that mean "nothing is wrong with symmetric memory here".
_BENIGN_PROBE = ("ok", "no-group-name", "single-rank")


def preflight_failed(fields: Dict[str, Any], *, ep_size: int) -> bool:
    """Decide whether the collected preflight fields indicate a real problem.

    Pure function of the collected fields so the verdict can be tested
    directly, including the cases that must NOT trip it.

    A single-rank group never reaches symmetric memory, so nothing it
    reports can be a fault. ``supports_mnnvl`` only counts as a failure
    when it is literally ``False``: a string such as
    ``"check-failed:NVMLError"`` means the *check* was unusable, which is
    reported but is not evidence that the capability is absent -- treating
    an unusable probe as a negative result is how a broken probe starts
    manufacturing conclusions.
    """
    if ep_size <= 1:
        return False
    if fields.get("supports_mnnvl") is False:
        return True
    return str(fields.get("probe", "")) not in _BENIGN_PROBE


def symm_buffer_preflight(pg, *, layer_idx: int, num_bytes_hint: int | None = None) -> None:
    """Drain, then gate, then describe -- immediately before the DG rendezvous.

    ``pg`` is the expert-parallel ``ProcessGroup`` that the SymmBuffer will
    rendezvous over. Runs once per SymmBuffer cache key (build time only,
    from ``post_load_weights`` -> ``cache_derived_state``), so the added
    synchronize costs nothing at steady state and does not perturb the
    steady-state launch timing the way a global ``CUDA_LAUNCH_BLOCKING``
    would.
    """
    dist_up = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(pg) if dist_up else -1
    dev_id = torch.cuda.current_device()
    fields: Dict[str, Any] = {
        "layer": layer_idx,
        "host": socket.gethostname(),
        "grank": dist.get_rank() if dist_up else -1,
        "eprank": rank,
        "epsize": pg.size(),
        "dev": dev_id,
    }
    if num_bytes_hint is not None:
        fields["bytes"] = num_bytes_hint

    # -- Stage 0: drain ------------------------------------------------
    # Attribute a *pre-existing* fault to its origin instead of to the
    # SymmBuffer. Without this, a sticky error from weight loading is
    # first observed by the synchronize inside SymmBuffer.__init__ and
    # reads as if the rendezvous caused it.
    try:
        torch.cuda.synchronize()
        fields["drain"] = "clean"
    except Exception as e:  # noqa: BLE001
        fields["drain"] = f"PRE_EXISTING_FAULT:{type(e).__name__}"
        logger.error(
            f"[{_TAG}] {fields} -- CUDA was already faulted BEFORE the DeepGEMM "
            f"SymmBuffer rendezvous; the fault originates upstream of MegaMoE "
            f"(weight load or an earlier layer), not in symmetric memory. "
            f"Original error: {e}"
        )
        raise

    # -- Stage 1: capability gate (parity with every sibling NVLink path)
    try:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        fields["supports_mnnvl"] = bool(MnnvlMemory.supports_mnnvl())
    except Exception as e:  # noqa: BLE001
        fields["supports_mnnvl"] = f"check-failed:{type(e).__name__}"

    # -- Stage 2: group enablement + 1-byte multicast probe ------------
    # Same pattern as symm_mem_allreduce / symm_mem_allgather /
    # cute_dsl_megamoe_custom_op: a pooled handle's multicast_ptr can read
    # non-zero locally even when the group spans multiple NVSwitch
    # domains, so the probe is what distinguishes "multicast is real" from
    # "multicast looks available on this rank".
    try:
        import torch.distributed._symmetric_memory as torch_symm_mem

        group_name = str(getattr(pg, "group_name", "")) or None
        fields["group_name"] = group_name
        if pg.size() == 1:
            # DeepGEMM allocates with ``torch.empty`` and fabricates the
            # handle for a single-rank group -- symmetric memory is never
            # touched, so probing it here would test a path the real
            # allocation does not take.
            fields["probe"] = "single-rank"
        elif group_name is None:
            fields["probe"] = "no-group-name"
        else:
            torch_symm_mem.enable_symm_mem_for_group(group_name)
            probe = torch_symm_mem.empty(
                1, device=torch.device(f"cuda:{dev_id}"), dtype=torch.uint8
            )
            handle = torch_symm_mem.rendezvous(probe, group_name)
            mc = int(getattr(handle, "multicast_ptr", 0) or 0)
            fields["probe_mc_ptr"] = hex(mc)
            fields["probe"] = "ok" if mc != 0 else "MULTICAST_UNAVAILABLE"
    except Exception as e:  # noqa: BLE001
        fields["probe"] = f"FAILED:{type(e).__name__}"
        fields["probe_err"] = str(e)[:200]

    # -- Stage 3: fabric identity (compared across ranks in post-processing)
    fields.update(_fabric_identity(dev_id))

    bad = preflight_failed(fields, ep_size=pg.size())
    line = f"[{_TAG}] {fields}"
    if bad:
        if _strict():
            raise RuntimeError(
                f"{line} -- MegaMoE DeepGEMM SymmBuffer preflight failed and "
                f"TLLM_MEGA_MOE_SYMM_STRICT=1"
            )
        logger.warning(
            f"{line} -- NVLink symmetric-memory prerequisites are not "
            f"satisfied; the DeepGEMM SymmBuffer rendezvous is proceeding "
            f"anyway (set TLLM_MEGA_MOE_SYMM_STRICT=1 to make this fatal)."
        )
    else:
        logger.info(line)
