# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused all-reduce boundary's copy of the library's Lamport protocol, pinned to the library.

``csrc/mhcArBoundary.cu`` runs the one-shot Lamport all-reduce on the same workspace as the
library's, so its layout and rotation (namespace ``lamport``) must be the library's. The library
keeps them private: ``LamportComm`` in ``allReduceFusionKernels.cu`` (copied in
``MiniMaxReduceRMSKernel.cu``) reads them and ``allocate_allreduce_fusion_workspace`` builds them,
none of it in a header. This reads all of them, so changing either side alone fails here instead of
desynchronizing the two kernels at run time. A source check: it needs the repository's sources, not
a device.
"""

import re
from pathlib import Path

import pytest

# tests/unittest/_torch/modeling_v2/models/glm5_next/<target>/comm -> repo root
_ROOT = Path(__file__).resolve().parents[8]
_TARGET_CSRC = (
    _ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/glm5_next"
    / "glm_5_3_flash_fp8__sm_10x__tp4/csrc/mhcArBoundary.cu"
)
_COMM_KERNELS = _ROOT / "cpp/tensorrt_llm/kernels/communicationKernels"
_LAMPORT_COMMS = (
    _COMM_KERNELS / "allReduceFusionKernels.cu",
    _COMM_KERNELS / "MiniMaxReduceRMSKernel.cu",
)
_ALLOCATOR = _ROOT / "tensorrt_llm/_torch/distributed/allreduce_helper.py"

# The target's constants (in terms of its group size kRanks); LamportComm's statements below set the same.
_CONSTANTS = {
    "kBufferSlot": "2 * kRanks",
    "kFlagSlot": "3 * kRanks",
    "kLayoutSlot": "3 * kRanks + 1",
    "kTableSize": "3 * kRanks + 2",
    "kCounterWord": "0",
    "kFlagWord": "2",
    "kClearSizeWord": "0",
    "kCommSizeWord": "1",
    "kNumBuffers": "3",
}
# LamportComm's statements that set those values (whitespace-insensitive).
_LAMPORT_COMM_STATEMENTS = (
    # kFlagSlot, kCounterWord
    "counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];",
    # kFlagSlot, kFlagWord
    "flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];",
    # kLayoutSlot, kClearSizeWord
    "clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];",
    # kLayoutSlot, kCommSizeWord
    "auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];",
    # dataBuffer
    "int data_offset = flag_value % 3;",
    # clearBuffer: the previous call's buffer
    "int clear_offset = (flag_value + 2) % 3;",
    # kBufferSlot
    "data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;",
    "clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;",
    # nextFlag, and the clear size and counter reset that go with it
    "*flag_ptr = (flag_value + 1) % 3;",
    "*clear_ptr = new_clear_size;",
    "*counter_ptr = 0;",
)
# The one-shot kernel's clear size: the bytes it wrote, as the target writes it.
_ONE_SHOT_CLEAR_SIZE = (
    "comm.update(static_cast<int64_t>(params.size) * NRanks * sizeof(DType));",
    "int clear_access = static_cast<int>((comm.clear_size + sizeof(float4) - 1) / sizeof(float4));",
)
# The allocator's pointer table, flag words and layout words.
_ALLOCATOR_STATEMENTS = (
    "lamport_buffers = IpcMemory(mapping, 3 * lamport_buffers_size, is_p2p_supported)",
    'flag_buffer = torch.tensor([0, 0, 0], dtype=torch.int, device="cuda")',
    'layout_buffer = torch.tensor([0, lamport_buffers_size], dtype=torch.int64, device="cuda")',
    "ipc_buffers.serialize() + ipc_barriers.serialize() + lamport_buffers.serialize()"
    " + [flag_buffer.data_ptr()] + [layout_buffer.data_ptr()]",
)
# The target's rotation, in terms of its constants.
_TARGET_ROTATION = (
    "int dataBuffer(int flag) { return flag % kNumBuffers; }",
    "int clearBuffer(int flag) { return (flag + kNumBuffers - 1) % kNumBuffers; }",
    "int nextFlag(int flag) { return (flag + 1) % kNumBuffers; }",
)

for _path in (_TARGET_CSRC, *_LAMPORT_COMMS, _ALLOCATOR):
    if not _path.is_file():
        pytest.skip(f"the repository's sources are not here: no {_path}", allow_module_level=True)


def _squash(text: str) -> str:
    """``text`` without whitespace, so formatting does not matter."""
    return re.sub(r"\s+", "", text)


def _lamport_namespace() -> str:
    source = _TARGET_CSRC.read_text()
    match = re.search(r"namespace lamport\s*\{(.*?)\}\s*// namespace lamport", source, re.S)
    assert match, f"{_TARGET_CSRC.name} has no namespace lamport"
    return match.group(1)


def _struct_lamport_comm(path: Path) -> str:
    source = path.read_text()
    match = re.search(r"struct LamportComm\s*\{(.*?)\n\};", source, re.S)
    assert match, f"{path.name} has no struct LamportComm"
    return match.group(1)


def test_target_constants():
    body = _lamport_namespace()
    found = {
        name: _squash(value) for name, value in re.findall(r"constexpr int (k\w+) = ([^;]+);", body)
    }
    assert found == {name: _squash(value) for name, value in _CONSTANTS.items()}
    squashed = _squash(body)
    for statement in _TARGET_ROTATION:
        assert _squash(statement) in squashed, statement


@pytest.mark.parametrize("path", _LAMPORT_COMMS, ids=lambda p: p.name)
def test_library_lamport_comm(path):
    squashed = _squash(_struct_lamport_comm(path))
    missing = [s for s in _LAMPORT_COMM_STATEMENTS if _squash(s) not in squashed]
    assert not missing, (
        f"LamportComm in {path.name} changed; update namespace lamport in "
        f"{_TARGET_CSRC.name} and this test together: {missing}"
    )


def test_library_one_shot_clear_size():
    squashed = _squash(_LAMPORT_COMMS[0].read_text())
    for statement in _ONE_SHOT_CLEAR_SIZE:
        assert _squash(statement) in squashed, statement
    # The target clears whole 16-byte stores, rounded up, and records the bytes it wrote.
    target = _squash(_TARGET_CSRC.read_text())
    assert _squash("i < (clearSize + 15) / 16;") in target
    assert (
        _squash(
            "*clearPtr = static_cast<long long>(numRows) * kH * kRanks"
            " * static_cast<long long>(sizeof(__nv_bfloat16));"
        )
        in target
    )


def test_workspace_allocator():
    squashed = _squash(_ALLOCATOR.read_text())
    missing = [s for s in _ALLOCATOR_STATEMENTS if _squash(s) not in squashed]
    assert not missing, (
        f"the all-reduce workspace layout in {_ALLOCATOR.name} changed; update namespace lamport "
        f"in {_TARGET_CSRC.name} and this test together: {missing}"
    )
