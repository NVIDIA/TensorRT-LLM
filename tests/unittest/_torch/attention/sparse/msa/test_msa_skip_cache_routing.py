# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No buffer a captured graph writes on the device is routed through a skipping upload (K3 audit #150, check #6).

The MSA metadata's ``_upload_block_table`` (a ``DeviceCopyBuffer`` per destination) skips an upload when its record
says the destination already holds the values. That is sound only while nothing else writes the destination: a graph
replay that rewrites it on the device (``on_update_kv_lens`` / the MSA KV-length update for the lengths, slots and
valid-block counts, ``_preprocess_inputs`` for ``kv_lens_cuda``), or another graph key's captured forward writing the
same pooled CUDA graph metadata block, leaves the record describing values the device no longer holds. This test scans
``tensorrt_llm/_torch`` and fails if any of those buffers is the destination of a skipping helper; they have to go
through ``device_copy.copy`` / ``device_copy.fill_tail`` without ``unchanged`` (which skips the write too).

CPU only: it parses the sources.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

pytestmark = pytest.mark.cpu_only

_ROOT = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
# Buffers a captured forward writes on the device, and pooled CUDA graph metadata buffers that are never skipped.
_DEVICE_WRITTEN = {
    "msa_seq_lens_cuda",
    "msa_cu_kv_lens",
    "msa_out_cache_loc",
    "msa_n_valid_blocks",
    "_msa_prefill_n_valid_blocks",
    "kv_lens_cuda",
    "prompt_lens_cuda",
    "msa_short_table",
    "msa_qo_lens_cuda",
    "msa_cu_q_lens",
    "msa_q_batch_row",
    "msa_q_intra",
}
# Helpers that may skip the write; their first argument is the destination (``_upload_block_table(dst, src)``).
_SKIPPING = {"_upload_block_table"}
# Calls that skip the write when passed ``unchanged`` (anything but a literal False); their second argument is the
# destination (``device_copy.copy(site, dst, src)``, ``device_copy.fill_tail(site, buf, start, value)``).
_SKIPPING_WHEN_UNCHANGED = {"copy", "fill_tail"}


def _receiver_names(node) -> set:
    """Attribute / variable names along the receiver chain of an expression (``self.a[:n].view(...)`` -> {"a"})."""
    names = set()
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Call)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        node = node.func if isinstance(node, ast.Call) else node.value
    if isinstance(node, ast.Name):
        names.add(node.id)
    return names


def _may_skip(call: ast.Call) -> bool:
    """Whether the call passes ``unchanged`` as anything but a literal False."""
    return any(
        kw.arg == "unchanged"
        and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
        for kw in call.keywords
    )


def _skipped_device_buffers(tree: ast.AST):
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if name in _SKIPPING:
            dst = node.args[0]
        elif name in _SKIPPING_WHEN_UNCHANGED and len(node.args) > 1 and _may_skip(node):
            dst = node.args[1]
        else:
            continue
        bad = _receiver_names(dst) & _DEVICE_WRITTEN
        if bad:
            hits.append((node.lineno, name, sorted(bad)))
    return hits


def test_no_device_written_buffer_goes_through_the_skip_cache() -> None:
    found = []
    for path in sorted(_ROOT.rglob("*.py")):
        src = path.read_text()
        if "unchanged" not in src and not any(helper in src for helper in _SKIPPING):
            continue
        for line, helper, bad in _skipped_device_buffers(ast.parse(src)):
            found.append(f"{path.relative_to(_ROOT).as_posix()}:{line} {helper}({', '.join(bad)})")
    assert not found, "device-written buffers routed through the H2D skip cache:\n" + "\n".join(
        found
    )


def test_the_scan_sees_a_skipped_device_buffer() -> None:
    """The scan itself: a skipping write of a device-written buffer is reported, a never-skipping one is not."""
    bad = ast.parse(
        "self._upload_block_table(self.msa_seq_lens_cuda[:b], host)\n"
        "m._upload_block_table(m.msa_out_cache_loc, host)\n"
        "device_copy.copy(S.SEQ_LENS, self.msa_seq_lens_cuda[:b], host, unchanged=same)\n"
        "device_copy.fill_tail(S.TAIL, self.msa_out_cache_loc, n, 0, unchanged=True)"
    )
    ok = ast.parse(
        "device_copy.copy(S.SEQ_LENS, self.msa_seq_lens_cuda[:b], host)\n"
        "device_copy.copy(S.SEQ_LENS, self.msa_seq_lens_cuda[:b], host, unchanged=False)\n"
        "device_copy.copy(S.BLOCK_TABLE, self.msa_block_table[:b], host, unchanged=same)\n"
        "self._upload_block_table(self.msa_block_table[:b], h)"
    )
    assert [h[1] for h in _skipped_device_buffers(bad)] == ["_upload_block_table"] * 2 + [
        "copy",
        "fill_tail",
    ]
    assert _skipped_device_buffers(ok) == []
