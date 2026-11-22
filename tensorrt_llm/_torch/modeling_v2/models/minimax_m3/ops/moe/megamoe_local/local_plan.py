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
"""Host planning of the MiniMax-M3.1 MegaMoE local backend (``TrtllmCutedslMegaMoeLocalNvfp4Impl``).

The switches of the backend and its levers, the shared-expert placement around its launches (serial after the launch,
or beside it on the aux stream with an optional pinned GEMM), and the deployments the backend cannot serve.

Pure Python with no torch dependency: everything here is integer bookkeeping that has to be identical on every rank.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

DISABLE_ENV = "TRTLLM_M31_DISABLE_MEGAMOE_LOCAL"
SHARED_SERIAL_DISABLE_ENV = "TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL"
SHARED_CORUN_PIN_DISABLE_ENV = "TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_PIN"
SHARED_SERIAL_MAX_ROWS_ENV = "TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS"
#: Default largest token bucket (rows per rank) whose MegaMoE launches run the shared expert after the routed launch
#: instead of beside it on the aux stream: every generation-phase bucket.
SHARED_SERIAL_MAX_ROWS = 1024
SHARED_SERIAL_MIN_EP_SIZE_ENV = "TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MIN_EP_SIZE"
#: Default smallest expert-parallel size whose steps may run the shared expert serially: every size.
SHARED_SERIAL_MIN_EP_SIZE = 1
#: ``LO:HI`` rows per rank: the launch buckets above LO up to HI keep the shared expert beside the launch. Unset: none.
SHARED_CORUN_BAND_ENV = "TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND"
#: ``1``: no co-run band, whatever ``SHARED_CORUN_BAND_ENV`` or a default says.
SHARED_CORUN_BAND_DISABLE_ENV = "TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_BAND"
SHARED_CORUN_PIN_ABOVE_ROWS_ENV = "TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_ABOVE_ROWS"
#: Default token bucket (rows per rank) above which a shared expert beside the MegaMoE launch on the aux stream runs
#: its MXFP8 linears on the pinned GEMM, per pin kernel: the CuTe DSL GEMM pins every bucket that runs beside the
#: launch, the native GEMM only the larger ones.
SHARED_CORUN_PIN_ABOVE_ROWS = 64
SHARED_CORUN_PIN_CUTE_DSL_ABOVE_ROWS = 32
SHARED_CORUN_PIN_KERNEL_ENV = "TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_KERNEL"
#: The pinned GEMM: FlashInfer's CuTe DSL GEMM (the default) or the native CUTLASS GEMM.
SHARED_CORUN_PIN_KERNELS = ("cute_dsl", "cutlass")
#: The backend's levers; ``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_<LEVER>=1`` turns one off.
LEVERS = (
    "ROUTEQ",
    "TAIL",
    "ROUTE_BESIDE_SHARED",
    "ROUTEQ_SPLIT_TAIL",
    "ROUTEQ_PORT",
)

NUM_DOMAINS = 2


def megamoe_local_requested(environ: Mapping[str, str]) -> bool:
    """On unless the master DISABLE switch is set."""
    return environ.get(DISABLE_ENV) != "1"


def lever_enabled(lever: str, environ: Mapping[str, str]) -> bool:
    """A lever is on unless the backend is off or its own DISABLE switch is set."""
    if lever not in LEVERS:
        raise ValueError(f"unknown MegaMoE-local lever {lever!r}; known: {LEVERS}")
    return megamoe_local_requested(environ) and environ.get(f"{DISABLE_ENV}_{lever}") != "1"


def _non_negative_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from None
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return number


def shared_serial_max_rows(environ: Mapping[str, str]) -> int:
    """The serial row threshold: ``TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS``, else ``SHARED_SERIAL_MAX_ROWS``."""
    return _non_negative_int(environ, SHARED_SERIAL_MAX_ROWS_ENV, SHARED_SERIAL_MAX_ROWS)


def shared_serial_min_ep_size(environ: Mapping[str, str]) -> int:
    """The smallest expert-parallel size that runs serially: ``TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MIN_EP_SIZE``,
    else ``SHARED_SERIAL_MIN_EP_SIZE``."""
    return _non_negative_int(environ, SHARED_SERIAL_MIN_EP_SIZE_ENV, SHARED_SERIAL_MIN_EP_SIZE)


def shared_corun_band(environ: Mapping[str, str]) -> Optional[Tuple[int, int]]:
    """The co-run band ``(LO, HI)`` from ``TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND=LO:HI``, else None; None with
    ``TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_BAND=1``."""
    if environ.get(SHARED_CORUN_BAND_DISABLE_ENV) == "1":
        return None
    value = environ.get(SHARED_CORUN_BAND_ENV)
    if value is None:
        return None
    low, separator, high = value.partition(":")
    try:
        band = (int(low), int(high))
    except ValueError:
        band = None
    if not separator or band is None or not 0 <= band[0] < band[1]:
        raise ValueError(
            f"{SHARED_CORUN_BAND_ENV} must be LO:HI rows per rank with 0 <= LO < HI, got {value!r}"
        )
    return band


def shared_expert_serial(
    launch_bucket: Optional[int], ep_size: int, environ: Mapping[str, str]
) -> bool:
    """Whether a MegaMoE launch of token bucket ``launch_bucket`` (rows per rank), at ``ep_size``-way expert
    parallelism, runs the shared expert serially.

    The test is on the launch's bucket, not its row count, so every batch that a captured bucket serves gets the
    same answer. A shared expert on the aux stream beside the persistent MegaMoE kernel can stall it, so every
    generation-phase bucket runs it serially by default; ``TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS`` narrows that
    scope, ``TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND=LO:HI`` keeps the aux stream for the buckets above LO up to HI, and
    ``TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL=1`` keeps it at every size. An unknown bucket runs serially.
    """
    if environ.get(SHARED_SERIAL_DISABLE_ENV) == "1":
        return False
    if ep_size < shared_serial_min_ep_size(environ):
        return False
    band = shared_corun_band(environ)
    if band is not None and launch_bucket is not None and band[0] < launch_bucket <= band[1]:
        return False
    return launch_bucket is None or launch_bucket <= shared_serial_max_rows(environ)


def shared_corun_pin_kernel(environ: Mapping[str, str]) -> str:
    """The pinned GEMM: ``TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_KERNEL`` (``cute_dsl`` or ``cutlass``), else
    ``cute_dsl``."""
    kernel = (
        environ.get(SHARED_CORUN_PIN_KERNEL_ENV, "").strip().lower() or SHARED_CORUN_PIN_KERNELS[0]
    )
    if kernel not in SHARED_CORUN_PIN_KERNELS:
        raise ValueError(
            f"{SHARED_CORUN_PIN_KERNEL_ENV}={environ[SHARED_CORUN_PIN_KERNEL_ENV]!r}: expected one of "
            f"{', '.join(SHARED_CORUN_PIN_KERNELS)}"
        )
    return kernel


def shared_corun_pin_above_rows(environ: Mapping[str, str]) -> int:
    """The pin's row floor: ``TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_ABOVE_ROWS``, else the pin kernel's default
    (``SHARED_CORUN_PIN_CUTE_DSL_ABOVE_ROWS`` for ``cute_dsl``, ``SHARED_CORUN_PIN_ABOVE_ROWS`` for ``cutlass``)."""
    default = (
        SHARED_CORUN_PIN_CUTE_DSL_ABOVE_ROWS
        if shared_corun_pin_kernel(environ) == "cute_dsl"
        else SHARED_CORUN_PIN_ABOVE_ROWS
    )
    return _non_negative_int(environ, SHARED_CORUN_PIN_ABOVE_ROWS_ENV, default)


def shared_corun_pin(launch_bucket: Optional[int], environ: Mapping[str, str]) -> bool:
    """Whether a shared expert beside a MegaMoE launch of token bucket ``launch_bucket`` (rows per rank) on the aux
    stream runs its MXFP8 linears on the pinned GEMM (``shared_corun_pin_kernel``).

    Only above ``shared_corun_pin_above_rows`` rows per rank; smaller buckets keep the automatic backend choice. An
    unknown bucket (one launch at the full bucket) is pinned. Off with
    ``TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_PIN=1``.
    """
    if environ.get(SHARED_CORUN_PIN_DISABLE_ENV) == "1":
        return False
    return launch_bucket is None or launch_bucket > shared_corun_pin_above_rows(environ)


def lever_environ(environ: Mapping[str, str]) -> Dict[str, str]:
    """The entries of ``environ`` that the MegaMoE-local levers, the shared-serial scope and the shared-corun pin
    read."""
    return {
        key: value
        for key, value in environ.items()
        if key.startswith("TRTLLM_M31_DISABLE_MEGAMOE")
        or key
        in (
            SHARED_SERIAL_MAX_ROWS_ENV,
            SHARED_SERIAL_MIN_EP_SIZE_ENV,
            SHARED_CORUN_BAND_ENV,
            SHARED_CORUN_PIN_ABOVE_ROWS_ENV,
            SHARED_CORUN_PIN_KERNEL_ENV,
        )
    }


def vd_static_rejection(
    *,
    ep_size: int,
    num_experts: int,
    num_slots: int,
    moe_tp_size: int,
    eplb_enabled: bool,
    world_size: Optional[int] = None,
) -> Optional[str]:
    """Why the MegaMoE local backend cannot serve this deployment, or None if it can.

    A load balancer (static or dynamic) migrates the stock expert tensors, whose halves the backend keeps per
    locality domain, so any EPLB declines it.
    """
    if eplb_enabled:
        return "an expert load balancer is registered (static or dynamic EPLB)"
    if num_slots != num_experts:
        return f"num_slots={num_slots} differs from num_experts={num_experts} (redundant experts)"
    if moe_tp_size != 1:
        return f"moe_tp_size={moe_tp_size}; the split needs moe_tp_size == 1"
    if ep_size < 2:
        return f"ep_size={ep_size}; the split needs a multi-rank EP group"
    if num_experts % (NUM_DOMAINS * ep_size) != 0:
        return f"num_experts={num_experts} is not divisible by {NUM_DOMAINS} * ep_size={ep_size}"
    if world_size is not None and world_size != ep_size:
        return f"torch.distributed WORLD={world_size} differs from EP={ep_size}"
    return None
