# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selection policy for the compact (two CTAs per SM) ``trtllm::kda_mtp_decode`` layout.

The in-tree verify kernel (``kda_mtp_decode.py``) runs one 512-thread
CTA per (head, request), one CTA per SM. ``feat/k3_26.08`` carries a compact
shared-memory / TMEM layout. It needs at most 111 KiB of smem and
256 TMEM columns per CTA, and on SM107 a 64-register launch bound and named barrier
ids below 5, so two CTAs co-reside on an SM once ``H * N`` exceeds the SM count. One CTA's
load phase then overlaps the other's tensor-core chain. That kernel is kept next to the in-tree one
(``kda_mtp_decode_compact.py``, launcher in ``cute_dsl_kimi_k3_kda_mtp_compact_ops``)
instead of being merged into it. ``cute_dsl_kimi_k3_kda_mtp_ops.kda_mtp_decode_impl``
asks :func:`select_compact_layout` which launcher to compile for each call.

This module is pure Python on purpose: the in-tree path imports only it, and the
vendored kernel is imported only when the compact layout is first selected.

Switches. They are startup settings: snapshotted when this module is imported (set
them before the serving workers start) and parsed on the first call on each device.

* ``TRTLLM_KDA_MTP_COMPACT``: unset or ``0`` keeps the in-tree kernel (the default
  on this branch; ``feat/k3_26.08`` defaults to ``1``); ``1`` selects the compact
  layout wherever the policy below holds. Any other value raises ``ValueError``.
* ``TRTLLM_KDA_MTP_COMPACT_QUANTIZED``: unset = on for SM107 only; ``0``/``1``.
  Allows the compact layout for the MXFP8-output variant.
* ``TRTLLM_KDA_MTP_COMPACT_TWO_CTAS``: unset = on for SM107 only; ``0``/``1``.
  Launch bound ``min_blocks_per_mp=2`` and named barrier ids 2..4, which SM107 needs
  for two compact CTAs to actually co-reside.

The two sub-switches are read (and must be ``0``/``1``) only when the compact layout
is enabled, so the default path depends on ``TRTLLM_KDA_MTP_COMPACT`` alone. The
Rubin recipes set all three to ``1``. The policy is ``feat/k3_26.08``'s:
SM100/103/107, one CTA per tile (``H * N > SPLIT_MAX_TILES``), more tiles than SMs,
a bf16 recurrent state, the fused output norm, ``num_spec <= 7`` on SM107 and
``<= 3`` on SM100/103 (``feat/k3_26.08``: ``<= 3``), and MXFP8 output only with
``TRTLLM_KDA_MTP_COMPACT_QUANTIZED``. Every other call runs the in-tree kernel.
``num_spec`` 4..7 (DSpark draft lengths 5..7) fit two CTAs per SM since the Kb hi tile
moved into the dead A3 rows and the cumulative gates stay in registers (112,640 B of
dynamic smem per CTA at ``num_spec = 7``, budget 115,712 B). They are validated on
SM107 only; with one CTA per SM (the SM100/103 default) the compact layout runs at the
in-tree kernel's speed, so those architectures keep the bound of 3.

Precedence: ``TRTLLM_KDA_MTP_COMPACT=1`` together with ``TRTLLM_KDA_MTP_BF16_MATH=1``
raises ``ValueError`` on the first call, whatever the shape. The compact kernel
implements the 3xTF32 recurrence only, and the BF16-operand path exists only in the
in-tree kernel. With the compact switch unset or ``0``, ``TRTLLM_KDA_MTP_BF16_MATH``
behaves exactly as before.

Parity: the compact kernel performs the same arithmetic as the in-tree kernel's
default (3xTF32) path and reads and writes the same replay caches. Outputs, the
committed state, the conv windows and the replay caches are therefore bitwise
identical, and a slot may move between the two kernels from one call to the next
(for example when the batch drops below one wave).
"""

import os
from typing import Optional, Tuple

import torch

_COMPACT_ENV = "TRTLLM_KDA_MTP_COMPACT"
_COMPACT_QUANTIZED_ENV = "TRTLLM_KDA_MTP_COMPACT_QUANTIZED"
_COMPACT_TWO_CTAS_ENV = "TRTLLM_KDA_MTP_COMPACT_TWO_CTAS"
# Architectures of the feat/k3_26.08 policy (B200/GB200, B300/GB300, VR200).
_COMPACT_ARCHS = ((10, 0), (10, 3), (10, 7))
# The chunk form pads the token axis (2 * NUM_SPEC + 1 rows) to 16: NUM_SPEC <= 7.
# num_spec 4..7 is validated on SM107 only; SM100/103 keep feat/k3_26.08's bound.
_COMPACT_MAX_NUM_SPEC = 7
_COMPACT_MAX_NUM_SPEC_SM100 = 3

# Startup snapshot of the switches (raw strings; None = unset).
_KDA_MTP_COMPACT = os.environ.get(_COMPACT_ENV)
_KDA_MTP_COMPACT_QUANTIZED = os.environ.get(_COMPACT_QUANTIZED_ENV)
_KDA_MTP_COMPACT_TWO_CTAS = os.environ.get(_COMPACT_TWO_CTAS_ENV)
# device index -> (sm, enabled, quantized, two_ctas)
_kda_compact_policy_cache = {}
# device index -> SM count
_num_sms_cache = {}


def _parse_switch(name: str, value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _device_index(device: torch.device) -> int:
    dev = torch.device(device)
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _kda_compact_policy(device: torch.device) -> Tuple[Tuple[int, int], bool, bool, bool]:
    """Return ``(sm, enabled, quantized, two_ctas)`` for ``device``.

    Computed once per device index from the startup snapshot. The device-dependent
    defaults (``_QUANTIZED`` / ``_TWO_CTAS`` on for SM107) match ``feat/k3_26.08``.
    """
    idx = _device_index(device)
    if idx not in _kda_compact_policy_cache:
        sm = torch.cuda.get_device_capability(idx)
        enabled = _parse_switch(_COMPACT_ENV, _KDA_MTP_COMPACT, False)
        quantized = two_ctas = False
        if enabled:
            rubin = sm == (10, 7)
            quantized = _parse_switch(_COMPACT_QUANTIZED_ENV, _KDA_MTP_COMPACT_QUANTIZED, rubin)
            two_ctas = _parse_switch(_COMPACT_TWO_CTAS_ENV, _KDA_MTP_COMPACT_TWO_CTAS, rubin)
        _kda_compact_policy_cache[idx] = (sm, enabled, quantized, two_ctas)
    return _kda_compact_policy_cache[idx]


def _num_sms(device: torch.device) -> int:
    idx = _device_index(device)
    if idx not in _num_sms_cache:
        _num_sms_cache[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _num_sms_cache[idx]


def select_compact_layout(
    *,
    device: torch.device,
    split_v: int,
    num_tiles: int,
    state_dtype: torch.dtype,
    fuse_output_norm: bool,
    num_spec: int,
    quantize_output: bool,
    bf16_math: bool,
) -> Tuple[bool, bool]:
    """Return ``(compact, two_ctas)`` for one ``kda_mtp_decode`` call.

    ``num_tiles`` is ``HV * N``. ``compact`` selects the vendored launcher
    ``_run_kda_decode_mtp_compact``; ``two_ctas`` its SM107 co-residency variant.
    ``(False, False)`` keeps the in-tree kernel.

    Raises:
        ValueError: ``TRTLLM_KDA_MTP_COMPACT=1`` with ``TRTLLM_KDA_MTP_BF16_MATH=1``
            (``bf16_math``), or an invalid switch value.
    """
    sm, enabled, quantized, two_ctas = _kda_compact_policy(device)
    if not enabled:
        return False, False
    if bf16_math:
        raise ValueError(
            f"{_COMPACT_ENV}=1 and TRTLLM_KDA_MTP_BF16_MATH=1 are mutually exclusive: the "
            "compact KDA MTP verify layout implements the 3xTF32 recurrence only, and the "
            "BF16-operand path exists only in the in-tree kernel. Unset one of them."
        )
    compact = (
        sm in _COMPACT_ARCHS
        and split_v == 1
        and num_tiles > _num_sms(device)
        and state_dtype is torch.bfloat16
        and bool(fuse_output_norm)
        and num_spec <= (_COMPACT_MAX_NUM_SPEC if sm == (10, 7) else _COMPACT_MAX_NUM_SPEC_SM100)
        and (not quantize_output or quantized)
    )
    return compact, compact and two_ctas
