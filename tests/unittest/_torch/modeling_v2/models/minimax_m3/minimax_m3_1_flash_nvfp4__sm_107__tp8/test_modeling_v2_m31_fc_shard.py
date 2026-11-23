# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC_SHARD on the sm_107 TP8 target: the drafter's fc as a column shard per rank + an all-gather.

Each 768-column shard uses a validated configuration of ``mv2_minimax_m3::minimax_m31_fc_shard_gemm`` and equals the
matching columns of ``torch.nn.functional.linear(x, w)`` for every shard and every row count up to MAX_ROWS.
This covers real-scale and 2^+-12 block-scaled inputs, eager and in a CUDA graph. The gathered shards are the full fc.
The wrapper
takes a shard only for the fc it was built from and the checked row counts. Default on;
TRTLLM_M31_DISABLE_FC_SHARD=1 turns it off."""

import importlib
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm import m31_fc_shard

K, N, TP = 30720, 6144, 8
COLS = N // TP


def _op_or_skip():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm_107 TP8 target runs on SM107 only")
    op = m31_fc_shard.fc_shard_op()
    assert op is not None, "this build has no mv2_minimax_m3::minimax_m31_fc_shard_gemm"
    return op


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cpu").manual_seed(2026)
    return (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).cuda()


@pytest.fixture(scope="module")
def fc(weight: torch.Tensor) -> torch.nn.Linear:
    _op_or_skip()
    linear = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    return linear


@pytest.fixture(scope="module")
def shards(fc: torch.nn.Linear) -> list[m31_fc_shard.FcShard]:
    result = []
    for rank in range(TP):
        shard, why = m31_fc_shard.FcShard._check(
            fc, TP, rank, SimpleNamespace(tp_size=TP, tp_rank=rank)
        )
        assert why is None and shard is not None, why
        result.append(shard)
    return result


def _inputs(rows, stress):
    g = torch.Generator(device="cpu").manual_seed(100 + rows)
    x = torch.randn(rows, K, generator=g)
    if stress:
        e = torch.randint(-12, 13, (rows, K // 128), generator=g).float()
        x = x * torch.pow(2.0, e).repeat_interleave(128, dim=1)
    return x.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("rows", list(range(1, m31_fc_shard.MAX_ROWS + 1)))
def test_every_shard_is_the_full_fc_columns(
    weight: torch.Tensor, shards: list[m31_fc_shard.FcShard], rows: int, stress: bool
) -> None:
    x = _inputs(rows, stress)
    full = F.linear(x, weight)
    parts = []
    for r in range(TP):
        got = shards[r].partial(x)
        assert torch.equal(got, full[:, r * COLS : (r + 1) * COLS]), r
        parts.append(got)
    assert torch.equal(torch.cat(parts, dim=-1), full)


def test_the_shard_in_a_cuda_graph(
    weight: torch.Tensor, shards: list[m31_fc_shard.FcShard]
) -> None:
    x = _inputs(8, True)
    shard = shards[3]
    out = torch.empty(8, COLS, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out.copy_(shard.partial(x))  # set the algo up outside the capture
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(shard.partial(x))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, F.linear(x, weight)[:, 3 * COLS : 4 * COLS])


def test_the_wrapper_shards_checks_and_gathers(
    fc: torch.nn.Linear, shards: list[m31_fc_shard.FcShard], monkeypatch: pytest.MonkeyPatch
) -> None:
    x = _inputs(7, False)
    s0 = shards[0]
    assert s0.applies(fc, 7) and s0.applies(fc, m31_fc_shard.MAX_ROWS)
    assert not s0.applies(fc, m31_fc_shard.MAX_ROWS + 1) and not s0.applies(fc, 0)
    other = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
    assert not s0.applies(other, 7)
    parts = [s.partial(x) for s in shards]
    assert all(p.shape == (7, COLS) for p in parts)
    # The all-gather concatenates the ranks' tensors in rank order along dim.
    monkeypatch.setattr(
        m31_fc_shard, "allgather", lambda y, mapping, dim: torch.cat(parts, dim=dim)
    )
    assert torch.equal(s0.gather(parts[0]), fc(x))


def test_a_non_applicable_fc_keeps_the_stock_path():
    _op_or_skip()
    fc = torch.nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
    shard, why = m31_fc_shard.FcShard._check(fc, TP, 0, SimpleNamespace(tp_size=TP, tp_rank=0))
    assert shard is None and "do not apply" in why


def test_the_native_operator_reports_a_configuration_rejection(weight: torch.Tensor) -> None:
    op = _op_or_skip()
    configuration = list(m31_fc_shard.PINNED_CFG)
    configuration[2] = 0
    inputs = torch.zeros(1, K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="not supported.*cuBLAS status"):
        op(inputs, weight[:COLS].contiguous(), configuration)


@pytest.fixture
def factory_fc() -> Iterator[torch.nn.Linear]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is required for the factory checks")
    with torch.random.fork_rng():
        yield torch.nn.Linear(32, 16, bias=False, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize(
    ("supported_splits", "reject_first"),
    [(22, False), (9, True), (9, False), (None, False)],
    ids=["first-config", "second-config", "second-config-mismatch", "no-compatible-config"],
)
def test_the_factory_selects_only_a_compatible_configuration(
    factory_fc: torch.nn.Linear,
    monkeypatch: pytest.MonkeyPatch,
    supported_splits: int | None,
    reject_first: bool,
) -> None:
    mapping = SimpleNamespace(tp_size=2, tp_rank=0)
    notices: list[str] = []
    last_configuration: tuple[int, ...] = ()

    def probe_operator(
        inputs: torch.Tensor, shard_weight: torch.Tensor, configuration: list[int]
    ) -> torch.Tensor:
        nonlocal last_configuration
        last_configuration = tuple(configuration)
        if configuration[2] == 22 and supported_splits == 9 and reject_first:
            raise NotImplementedError("the configuration is unavailable")
        reference = factory_fc(inputs)[:, : shard_weight.shape[0]]
        configuration_matches = configuration[2] == supported_splits
        if configuration[2] == 22 and supported_splits == 9 and not reject_first:
            configuration_matches = inputs.shape[0] < m31_fc_shard.MAX_ROWS
        if not configuration_matches:
            return torch.full_like(reference, float("nan"))
        return reference

    monkeypatch.setattr(m31_fc_shard, "fc_shard_op", lambda: probe_operator)
    monkeypatch.setattr(m31_fc_shard.logger, "info", notices.append)
    monkeypatch.setattr(
        m31_fc_shard, "allgather", lambda flags, mapping, dim: flags.expand(mapping.tp_size)
    )
    shard = m31_fc_shard.FcShard.build(factory_fc, mapping)
    if supported_splits is None:
        assert shard is None
        assert any("SPLITK_NUM=22" in notice and "SPLITK_NUM=9" in notice for notice in notices)
    else:
        assert shard is not None
        rows = m31_fc_shard.MAX_ROWS if supported_splits == 9 and not reject_first else 3
        inputs = torch.ones(rows, 32, dtype=torch.bfloat16, device="cuda")
        assert torch.equal(shard.partial(inputs), factory_fc(inputs)[:, :8])
        assert any(f"configuration {last_configuration}" in notice for notice in notices)


@pytest.mark.parametrize("failing_boundary", ["probe", "reference"])
def test_the_factory_keeps_stock_fc_when_the_reference_check_fails(
    factory_fc: torch.nn.Linear, monkeypatch: pytest.MonkeyPatch, failing_boundary: str
) -> None:
    mapping = SimpleNamespace(tp_size=2, tp_rank=0)
    module, name = (torch, "randn") if failing_boundary == "probe" else (F, "linear")
    operation = getattr(module, name)
    failure_injected = False
    agreement_ran = False

    def fail_once(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal failure_injected
        if not failure_injected:
            failure_injected = True
            raise RuntimeError("the reference check is unavailable")
        return operation(*args, **kwargs)

    def unused_operator(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("a failed reference check must not run the shard")

    def decline_agreement(flags: torch.Tensor, mapping: SimpleNamespace, dim: int) -> torch.Tensor:
        nonlocal agreement_ran
        agreement_ran = True
        assert not bool(flags.any())
        return flags.expand(mapping.tp_size)

    monkeypatch.setattr(module, name, fail_once)
    monkeypatch.setattr(m31_fc_shard, "fc_shard_op", lambda: unused_operator)
    monkeypatch.setattr(m31_fc_shard, "allgather", decline_agreement)
    assert m31_fc_shard.FcShard.build(factory_fc, mapping) is None
    assert failure_injected and agreement_ran


@pytest.mark.parametrize(
    "native_error", [torch.OutOfMemoryError, RuntimeError], ids=["allocation", "device-setup"]
)
def test_the_factory_keeps_stock_fc_when_native_setup_fails(
    factory_fc: torch.nn.Linear, monkeypatch: pytest.MonkeyPatch, native_error: type[RuntimeError]
) -> None:
    mapping = SimpleNamespace(tp_size=2, tp_rank=0)
    attempts = 0
    local_failure_reported = False

    def fail_once(
        inputs: torch.Tensor, shard_weight: torch.Tensor, configuration: list[int]
    ) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise native_error("native setup is unavailable")
        return factory_fc(inputs)[:, : shard_weight.shape[0]]

    def rank_agreement(flags: torch.Tensor, mapping: SimpleNamespace, dim: int) -> torch.Tensor:
        nonlocal local_failure_reported
        local_failure_reported = not bool(flags.any())
        return flags.expand(mapping.tp_size)

    monkeypatch.setattr(m31_fc_shard, "fc_shard_op", lambda: fail_once)
    monkeypatch.setattr(m31_fc_shard, "allgather", rank_agreement)
    shard = m31_fc_shard.FcShard.build(factory_fc, mapping)
    assert attempts == 1
    assert shard is None and local_failure_reported


@pytest.mark.parametrize(
    "env, expected", [({}, True), ({"TRTLLM_M31_DISABLE_FC_SHARD": "1"}, False)]
)
def test_fc_shard_default_on_and_kill_switch(monkeypatch, env, expected):
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
        m31_switches,
    )

    monkeypatch.delenv("TRTLLM_M31_DISABLE_FC_SHARD", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(m31_switches)
    try:
        assert importlib.reload(dspark).FC_SHARD is expected
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_FC_SHARD", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(dspark)
    assert dspark.FC_SHARD is True
