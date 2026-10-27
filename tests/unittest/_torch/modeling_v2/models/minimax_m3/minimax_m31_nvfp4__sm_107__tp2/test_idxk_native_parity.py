# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer GPU gate: compare the source-built ops to the frozen cut-8 library.

Set M31_IDXK_CANDIDATE_LIBRARY to the built libth_common.so and
M31_IDXK_REFERENCE_LIBRARY to the shipped index-K library. Separate processes
avoid duplicate registration of the m31idxk namespace. This QA test is not in a
CI list because its reference binary is an external campaign artifact.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REFERENCE_MD5 = "227f3572511bc6cd8aae49844a103e66"


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _run_library(library, output):
    torch.ops.load_library(library)
    ops = torch.ops.m31idxk
    results = {}
    for count in (0, 1, 16, 64, 128, 224, 448, 8192, 16384):
        page_count = max(1, (count + 127) // 128 + 1)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for index_dtype in (torch.int32, torch.int64):
                for padding in (0, 32):
                    torch.manual_seed(67)
                    # Generate on CPU so both processes consume identical bytes.
                    rows = (torch.randn(count, 128) * 2).to(dtype).cuda()
                    loc = torch.arange(count, dtype=index_dtype, device="cuda")
                    loc[::17] = -1
                    storage = torch.full(
                        (page_count, 9216 + padding), 0x5A, dtype=torch.uint8, device="cuda"
                    )
                    pages = storage[:, :9216]
                    sink = torch.zeros(
                        page_count, 1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda"
                    )
                    # Keep padded sink rows zero; invalid slots are skipped.
                    sink.view(-1, 128)[:count] = rows.to(torch.float8_e4m3fn)
                    ids = torch.arange(page_count - 1, -1, -1, dtype=torch.int32, device="cuda")
                    gathered = torch.empty(
                        page_count, 128, 128, dtype=torch.float8_e4m3fn, device="cuda"
                    )
                    for from_sink in (False, True):

                        def invoke():
                            if from_sink:
                                ops.nvfp4_idxk_write_from_sink(sink, loc, count, pages)
                            else:
                                ops.nvfp4_idxk_write(rows, loc, pages)
                            ops.nvfp4_idxk_gather_fp8(pages, ids, gathered)

                        for captured in (False, True):
                            storage.fill_(0x5A)
                            if captured:
                                stream = torch.cuda.Stream()
                                stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    invoke()
                                torch.cuda.current_stream().wait_stream(stream)
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    invoke()
                                graph.replay()
                            else:
                                invoke()
                            torch.cuda.synchronize()
                            key = f"{count}/{dtype}/{index_dtype}/{padding}/{from_sink}/{captured}"
                            results[key] = (_digest(storage), _digest(gathered))
    Path(output).write_text(json.dumps(results, sort_keys=True))


def test_idxk_source_build_bitwise_to_cut8(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    candidate = os.environ.get("M31_IDXK_CANDIDATE_LIBRARY")
    reference = os.environ.get("M31_IDXK_REFERENCE_LIBRARY")
    if not (candidate and reference):
        pytest.skip(
            "M31_IDXK_CANDIDATE_LIBRARY / M31_IDXK_REFERENCE_LIBRARY not set (developer gate)"
        )
    assert Path(candidate).resolve() != Path(reference).resolve()
    assert hashlib.md5(Path(reference).read_bytes()).hexdigest() == REFERENCE_MD5
    outputs = []
    for label, library in (("reference", reference), ("candidate", candidate)):
        output = tmp_path / f"{label}.json"
        subprocess.run([sys.executable, __file__, library, str(output)], check=True)
        outputs.append(json.loads(output.read_text()))
    assert len(outputs[0]) == 288
    assert outputs[0] == outputs[1], "source-built index-K differs from shipped bytes"


if __name__ == "__main__":
    _run_library(sys.argv[1], sys.argv[2])
