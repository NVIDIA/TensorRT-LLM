# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 24-row two-shot o-projection consumer's carve-out.

MR8's ``CONSUMER_CARVEOUT`` gives the two-shot consumer the 100 % shared-memory carve-out at 32 /
64 rows on Rubin only; ``oproj_consumer_carveout`` sets it on the 24-row consumer of a
``PreparedOproj`` (a driver attribute of the compiled function), only on Rubin, only on rows the
owner runs two-shot and only where the table sets nothing. ``carve_oproj_consumers`` does it per
owner. On an SM 10.x GPU the real 24-row consumer gets the attribute and the
other compiled functions keep theirs.
"""

import importlib
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
TP, RUBIN, GB300 = 16, (10, 7), (10, 3)


def _mr8(name):
    return importlib.import_module(f"{_FAMILY}.kernels.vendor.vllm_mr8").module(name)


def _modeling():
    return importlib.import_module(f"{_PKG}.modeling")


def _capability(value):
    return mock.patch.object(torch.cuda, "get_device_capability", lambda *a, **k: value)


def _prepared():
    """A ``PreparedOproj`` stand-in: fused 8, one-shot 16, two-shot 24 / 32."""
    codes = {8: ("fused8",), 16: ("p16", "c16"), 24: ("p24", "c24"), 32: ("p32", "c32")}
    return NS(codes=codes, twoshots={24, 32})


def _record():
    calls = []
    host = _mr8("cutedsl.common.host")
    patch = mock.patch.object(
        host,
        "prepare_shared_carveout",
        lambda code, index, value: calls.append((code, index, value)),
    )
    return calls, patch


def test_the_vendored_table_leaves_24_rows_out() -> None:
    oproj = _mr8("cutedsl.oproj.prepared")
    with _capability(RUBIN):
        assert {b: oproj.consumer_carveout(TP, b, None) for b in (16, 24, 32, 64)} == {
            16: None, 24: None, 32: 100, 64: 100,
        }  # fmt: skip
        assert oproj.twoshot_batches(TP) >= {24, 32}
    with _capability(GB300):
        assert oproj.consumer_carveout(TP, 32, None) is None


def test_only_the_two_shot_rows_the_table_misses() -> None:
    modeling = _modeling()
    assert modeling.OPROJ_CARVEOUT_ROWS == (24,) and modeling.OPROJ_CARVEOUT == 100
    calls, patch = _record()
    with _capability(RUBIN), patch:
        done = modeling.oproj_consumer_carveout(
            _prepared(), TP, (8, 16, 24, 32, 40), torch.device("cuda", 3)
        )
    assert done == [24]  # 8 fused, 16 one-shot, 32 in the table, 40 not prepared
    assert calls == [("c24", 3, 100)]


def test_rubin_only() -> None:
    calls, patch = _record()
    with _capability(GB300), patch:
        done = _modeling().oproj_consumer_carveout(_prepared(), TP, (24,), torch.device("cuda", 0))
    assert done == [] and calls == []


def test_the_owner_carves_each_prepared_layer() -> None:
    modeling = _modeling()
    prepared = {layer: _prepared() for layer in (1, 3)}
    calls, patch = _record()
    with _capability(RUBIN), patch:
        modeling.carve_oproj_consumers("kda", prepared, TP, 0, torch.device("cuda", 0))
    assert calls == [("c24", 0, 100), ("c24", 0, 100)]


def _carveout(compiled, index):
    import cuda.bindings.driver as cuda

    _, function = _mr8("cutedsl.common.host").resolve_kernel(compiled, index)
    attribute = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    err, value = cuda.cuFuncGetAttribute(attribute, function)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    return value


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="compiles MR8's o-projection pair for an SM10x GPU",
)
def test_the_24_row_consumer_gets_the_carve_out(monkeypatch) -> None:
    """The real two-shot pairs at 24 and 32 rows (the KDA / DSA o-projection, K 768, reduced by the
    producer, batched consumer): with the device's capability admitted, only the 24-row consumer's
    attribute changes, to 100 (32 keeps the table's, the producers theirs)."""
    modeling, oproj = _modeling(), _mr8("cutedsl.oproj.prepared")
    index = torch.cuda.current_device()
    monkeypatch.setattr(
        modeling, "OPROJ_CARVEOUT_CAPABILITIES", (tuple(torch.cuda.get_device_capability()),)
    )
    codes = {}
    for b in (24, 32):
        shape = (768, b, TP, 2, True, False, False, 64, True, False, False, False)
        codes[b] = tuple(
            oproj.compile_oproj(
                kind, *shape, kind == "consumer", True, kind == "producer", 1, False
            )
            for kind in ("producer", "consumer")
        )
    before = {(b, i): _carveout(codes[b][i], index) for b in codes for i in (0, 1)}
    prepared = NS(codes=codes, twoshots={24, 32})
    done = modeling.oproj_consumer_carveout(prepared, TP, (24,), torch.device("cuda", index))
    after = {(b, i): _carveout(codes[b][i], index) for b in codes for i in (0, 1)}
    assert done == [24]
    assert after[(24, 1)] == 100
    assert {k: v for k, v in after.items() if k != (24, 1)} == {
        k: v for k, v in before.items() if k != (24, 1)
    }
