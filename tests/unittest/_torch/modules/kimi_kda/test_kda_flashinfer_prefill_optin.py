# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer KDA prefill is opt-in: ``TLLM_KDA_FLASHINFER_PREFILL=1``; the default is in-tree.

Unset or ``0`` must leave the in-tree prefill and the K3 metadata untouched. Set to ``1`` the
mixer must match the in-tree prefill within the port's numerics on the same weights, inputs
and pools: output, recurrent-state rows and convolution-state rows, with every other row of
both pools bitwise unchanged.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_flashinfer_schedule import (
    FLASHINFER_PREFILL_ENV,
    KDAFlashInferSchedule,
    flashinfer_prefill_requested,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_flashinfer_kda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="FlashInfer KDA requires SM100/SM103/SM107",
)


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_switch_values(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv(FLASHINFER_PREFILL_ENV, raising=False)
    else:
        monkeypatch.setenv(FLASHINFER_PREFILL_ENV, value)
    assert flashinfer_prefill_requested() is expected


@pytest.mark.parametrize("value", ["2", "true", "on", ""])
def test_switch_rejects_other_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(FLASHINFER_PREFILL_ENV, value)
    with pytest.raises(ValueError, match=FLASHINFER_PREFILL_ENV):
        flashinfer_prefill_requested()


def test_default_import_does_not_load_the_flashinfer_ops() -> None:
    """With the switch unset, importing the mixer and metadata leaves the FlashInfer ops unloaded."""
    ops = "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops"
    code = (
        "import sys\n"
        "import tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer\n"
        "import tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata\n"
        f"print({ops!r} in sys.modules)\n"
    )
    env = {key: value for key, value in os.environ.items() if key != FLASHINFER_PREFILL_ENV}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"


@needs_cuda
def test_metadata_schedule_exists_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

    monkeypatch.delenv(FLASHINFER_PREFILL_ENV, raising=False)
    default = KimiK3MambaMetadata(max_batch_size=8, chunk_size=128)
    assert default._kda_fi_schedule is None
    assert not any(hasattr(default, name) for name in ("kda_seq_order", "kda_cu_chunks"))

    monkeypatch.setenv(FLASHINFER_PREFILL_ENV, "1")
    opted = KimiK3MambaMetadata(max_batch_size=8, chunk_size=128)
    assert opted._kda_fi_schedule is not None
    assert opted.kda_max_seqlen == 0
    assert opted.kda_seq_order.numel() == 0 and opted.kda_cu_chunks.numel() == 1


@needs_cuda
def test_schedule_orders_by_length_and_counts_16_token_chunks() -> None:
    schedule = KDAFlashInferSchedule(max_batch_size=8)
    metadata = SimpleNamespace()
    schedule.prepare(metadata, [17, 64, 1, 64, 33])
    torch.cuda.synchronize()
    assert metadata.kda_max_seqlen == 64
    # Stable descending length order; prefix sums of ceil(length / 16) in request order.
    assert metadata.kda_seq_order.tolist() == [1, 3, 4, 0, 2]
    assert metadata.kda_cu_chunks.tolist() == [0, 2, 6, 7, 11, 14]
    assert metadata.kda_seq_order.dtype == metadata.kda_cu_chunks.dtype == torch.int32

    schedule.prepare(metadata, [])
    assert metadata.kda_max_seqlen == 0
    assert metadata.kda_seq_order.numel() == 0 and metadata.kda_cu_chunks.numel() == 1


@needs_flashinfer_kda
@torch.inference_mode()
def test_default_mixer_never_enters_the_flashinfer_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fla")
    import tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer as mixer_module
    from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import (
        make_attention,
        make_batch,
    )

    attention = make_attention(6, hidden_size=768, flashinfer=False)
    assert attention._kda_flashinfer_prefill is False

    def refuse(*args, **kwargs):
        raise AssertionError("the FlashInfer prefill ran with the switch off")

    monkeypatch.setattr(mixer_module, "forward_prefill_flashinfer", refuse)
    batch = make_batch(6, [17, 64, 150], hidden_size=768, initial="mixed")
    output = attention.forward_prefill(
        batch.hidden, batch.cu, batch.metadata, batch.n, batch.conv, batch.state, batch.indices
    )
    assert torch.isfinite(output).all()


@needs_flashinfer_kda
@pytest.mark.parametrize(
    "heads, hidden, lengths",
    [
        (6, 768, [1, 17, 64, 1000, 1623]),
        (6, 768, [1623]),  # one request: the Small-BH route
        (96, 7168, [150, 1623]),
        (96, 7168, [1623]),  # K3 heads, one request (Small-BH where 2 x 96 fits the SMs)
    ],
)
@pytest.mark.parametrize("initial", ["fresh", "mixed"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_flashinfer_prefill_matches_intree_prefill(
    heads: int, hidden: int, lengths: list[int], initial: str, state_dtype: torch.dtype
) -> None:
    pytest.importorskip("fla")
    from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import (
        compare,
        make_attention,
        make_batch,
    )

    intree = make_attention(heads, hidden_size=hidden, flashinfer=False)
    port = make_attention(heads, hidden_size=hidden, flashinfer=True)
    batch = make_batch(heads, lengths, hidden_size=hidden, initial=initial)
    state0 = batch.state.to(state_dtype)
    conv_intree, state_intree = batch.conv.clone(), state0.clone()
    conv_port, state_port = batch.conv.clone(), state0.clone()

    expected = intree.forward_prefill(
        batch.hidden, batch.cu, batch.metadata, batch.n, conv_intree, state_intree, batch.indices
    )
    actual = port.forward_prefill(
        batch.hidden, batch.cu, batch.metadata, batch.n, conv_port, state_port, batch.indices
    )

    rows = batch.indices.long()
    for name, got, want in (
        ("output", actual, expected),
        ("recurrent state", state_port[rows], state_intree[rows]),
    ):
        metrics = compare(got, want)
        assert metrics["passed"], (name, metrics)
    # Both prefills store the same projected conv inputs (same FP8 GEMM) as the window.
    torch.testing.assert_close(conv_port[rows], conv_intree[rows], rtol=0, atol=0)
    # Rows outside the batch keep their pre-call contents on both paths.
    untouched = [i for i in range(state0.shape[0]) if i not in set(rows.tolist())]
    for state, conv in ((state_port, conv_port), (state_intree, conv_intree)):
        torch.testing.assert_close(state[untouched], state0[untouched], rtol=0, atol=0)
        torch.testing.assert_close(conv[untouched], batch.conv[untouched], rtol=0, atol=0)
