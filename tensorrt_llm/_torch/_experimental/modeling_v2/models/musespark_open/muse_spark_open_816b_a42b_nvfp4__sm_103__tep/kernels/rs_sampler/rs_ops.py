# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Library load and op wrappers of the q-store kernels (``rs_qstore.cu``, torch library ``musespark_open_rs``).

The q records (``rs_common.cuh`` ``QRecord``, ``uint8 [R, 1024]``, row = spec-decode slot) carry the draft
distribution q of each slot's last drafts in a sparse form the fused verify reads. These four ops keep them coherent
with the stock dense ``spec_metadata.draft_probs [R, 7, Vd]`` on the steps that run the stock path:

* ``init``: every record empty (kind NONE, valid 0);
* ``write_onehot``: after the stock context one-hot, the context slots' records are ONEHOT0;
* ``sparsify``: after the stock block draft sampler, the generation slots' records are FROM_DENSE (ids from the
  selector output's non ``-inf`` entries, q copied from ``draft_probs``);
* ``densify``: before a stock rejection verify, consume every valid position, scattering a SAMPLED record's q into
  the dense row first.

Thin wrappers only: no knob is read and nothing is cached besides the loaded library. Every op is a plain launch on
the current stream with static shapes (CUDA-graph safe); ``kernel_attributes`` resolves the kernels so that nothing
loads during a capture. ``slots`` is an int64 1-D CUDA tensor (a view of ``spec_metadata.batch_slot_ids``) and
``diag`` the ``int64 [rs_common.DIAG_WORDS]`` counters (``rs_common.Diag``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from . import rs_common

LIBRARY_NAME = "musespark_open_rs"
SOURCE = Path(__file__).resolve().parent / "rs_qstore.cu"
#: The q-store kernels, in the order of ``kernel_attributes``.
KERNELS = ("rsInitKernel", "rsWriteOnehotKernel", "rsSparsifyKernel", "rsDensifyKernel")
#: ``cudaFuncAttributes`` fields ``kernel_attributes`` reports per kernel.
ATTRIBUTE_FIELDS = ("num_regs", "local_size_bytes", "shared_size_bytes", "max_threads_per_block")

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    """The ``-gencode`` target of this device (``107a`` on Rubin)."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"the rs q-store kernels are built for sm_100a and newer, not sm_{major}{minor}"
        )
    return f"{major}{minor}a"


def build_kwargs() -> Dict[str, object]:
    """Arguments of ``torch.utils.cpp_extension.load`` for this device (no fast math: ``rs_common.NVCC_FLAGS``)."""
    target = arch_target()
    return dict(
        name=f"{LIBRARY_NAME}_sm{target}",
        sources=[str(SOURCE)],
        extra_cuda_cflags=list(rs_common.NVCC_FLAGS)
        + [f"-gencode=arch=compute_{target},code=sm_{target}"],
        extra_include_paths=[str(rs_common.INCLUDE_DIR)],
        is_python_module=False,
        verbose=False,
    )


def load_library() -> str:
    """Build (if needed) and load the library once per process; its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            from torch.utils.cpp_extension import load

            path = load(**build_kwargs())
            _LOADED = str(path) if path else LIBRARY_NAME
        return _LOADED


def _ops():
    load_library()
    return torch.ops.musespark_open_rs


def new_records(num_slots: int, device) -> torch.Tensor:
    """A ``uint8 [num_slots, RECORD_BYTES]`` record store on ``device``, initialized by ``init``."""
    records = torch.empty((num_slots, rs_common.RECORD_BYTES), dtype=torch.uint8, device=device)
    init(records)
    return records


def new_diag(device) -> torch.Tensor:
    """Zeroed ``int64 [DIAG_WORDS]`` counters on ``device``."""
    return torch.zeros(rs_common.DIAG_WORDS, dtype=torch.int64, device=device)


def init(records: torch.Tensor) -> None:
    """Every record to the empty state: ids -1, qf 0, draft -1, valid 0, kind NONE, mode 0, flags 0."""
    _ops().init(records)


def write_onehot(
    records: torch.Tensor, slots: torch.Tensor, diag: torch.Tensor, check: bool = False
) -> None:
    """The ONEHOT0 record (id 0 with q 1 at every position, drafts 0, valid 0x7F) for each slot in ``slots``."""
    _ops().write_onehot(records, slots, diag, check)


def selector_structure_ok(logits: torch.Tensor, vocab: int) -> bool:
    """Whether ``logits`` can be the stock selector output of the block draft sampler: FP32 ``[G, 7, vocab]`` with a
    contiguous last dimension."""
    return (
        logits.dtype == torch.float32
        and logits.dim() == 3
        and logits.shape[1] == rs_common.POS
        and logits.shape[-1] == vocab
        and logits.stride(-1) == 1
    )


def sparsify(
    records: torch.Tensor,
    slots: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
    stored_vocab: int,
    tokens: torch.Tensor,
    diag: torch.Tensor,
    structural_ok: Optional[bool] = None,
    check: bool = False,
) -> None:
    """The FROM_DENSE record of each slot ``slots[g]`` from selector output row ``logits[g]`` and the dense
    ``probs[slot, :7, :stored_vocab]`` the stock sampler drew ``tokens[g]`` from.

    ``structural_ok`` (default: ``selector_structure_ok(logits, logits.shape[-1])``) says the logits are the selector
    output; when it is False the records become kind NONE with valid 0 and ``logits`` is not read.
    """
    if structural_ok is None:
        structural_ok = logits.dim() == 3 and selector_structure_ok(logits, logits.shape[-1])
    _ops().sparsify(records, slots, logits, probs, stored_vocab, tokens, structural_ok, diag, check)


def densify(
    records: torch.Tensor,
    slots: torch.Tensor,
    probs: torch.Tensor,
    stored_vocab: int,
    diag: torch.Tensor,
    check: bool = False,
) -> None:
    """Before a stock rejection verify of ``slots``: rewrite ``probs[slot, t, :stored_vocab]`` from each valid
    position t of a SAMPLED record, and consume every valid position of every record."""
    _ops().densify(records, slots, probs, stored_vocab, diag, check)


def kernel_attributes(device=None) -> Dict[str, Dict[str, int]]:
    """Resolve the q-store kernels on ``device`` (default: current) and return their ``cudaFuncAttributes``."""
    anchor = torch.empty(1, dtype=torch.uint8, device=device if device is not None else "cuda")
    flat = list(_ops().kernel_attributes(anchor))
    width = len(ATTRIBUTE_FIELDS)
    return {
        name: dict(zip(ATTRIBUTE_FIELDS, flat[i * width : (i + 1) * width]))
        for i, name in enumerate(KERNELS)
    }


def record_fields(records: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Views of ``records`` (``uint8 [R, RECORD_BYTES]``): ids int32 [R, 7, 16], qf fp32 [R, 7, 16], draft int32
    [R, 7], valid int32 [R], kind uint8 [R], mode uint8 [R], pos_flags uint8 [R, 7]."""
    c = rs_common
    num = records.shape[0]

    def words(begin: int, end: int, dtype: torch.dtype) -> torch.Tensor:
        return records[:, begin:end].view(dtype)

    ids = words(c.OFFSET_IDS, c.OFFSET_QF, torch.int32).view(num, c.POS, c.LANES)
    qf = words(c.OFFSET_QF, c.OFFSET_DRAFT, torch.float32).view(num, c.POS, c.LANES)
    draft = words(c.OFFSET_DRAFT, c.OFFSET_VALID, torch.int32)
    valid = words(c.OFFSET_VALID, c.OFFSET_KIND, torch.int32).view(num)
    kind = records[:, c.OFFSET_KIND]
    mode = records[:, c.OFFSET_MODE]
    pos_flags = records[:, c.OFFSET_POS_FLAGS : c.OFFSET_POS_FLAGS + c.POS]
    return ids, qf, draft, valid, kind, mode, pos_flags
