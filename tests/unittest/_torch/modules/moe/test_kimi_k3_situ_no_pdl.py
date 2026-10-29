# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-PDL launch of the Kimi K3 shared expert's SiTU activation beside the all-to-all dispatch.

In an attention-DP MoE layer (DEP16) the fused ``trtllm::situ_and_mul`` of the shared expert,
``SituAndMul(overlaps_moe_dispatch=True)``, launches without programmatic dependent launch (PDL) for decode
batches (at most 1024 rows) on SM107 when ``TRTLLM_K3_DEP_SITU_NO_PDL`` asks for it: from 128 rows with ``auto``, for
every decode batch with ``1``, never when it is unset or ``0`` (opt-in: the default is the PDL launch). The tests check
that

* the selection (``situ_without_pdl``) follows the layer kind, the row count, the platform and the switch;
* the launch the op issues follows the selection;
* the no-PDL launch writes exactly the bytes of the PDL launch and can be captured in a CUDA graph.
"""

import pytest
import torch

from tensorrt_llm._torch.modules import situ
from tensorrt_llm._torch.modules.situ import SituAndMul

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

ROWS = [1, 48, 127, 128, 192, 1024, 1025]


@pytest.fixture
def platform(monkeypatch):
    """Pretend to be on (or off) the platform the switch was validated on, wherever the test runs."""

    def set_platform(supported):
        monkeypatch.setattr(situ, "mk1_platform_supported", lambda: supported)

    return set_platform


@pytest.fixture
def switch(monkeypatch):
    def set_switch(value):
        if value is None:
            monkeypatch.delenv(situ.ENV_SITU_NO_PDL, raising=False)
        else:
            monkeypatch.setenv(situ.ENV_SITU_NO_PDL, value)

    return set_switch


@pytest.mark.parametrize("num_rows", ROWS, ids=lambda v: f"rows{v}")
@pytest.mark.parametrize(
    "value", [None, "", "0", "auto", "1"], ids=["unset", "empty", "off", "auto", "on"]
)
@pytest.mark.parametrize("overlaps_moe_dispatch", [False, True], ids=["other", "beside_dispatch"])
def test_selection(platform, switch, overlaps_moe_dispatch, value, num_rows):
    platform(True)
    switch(value)
    decode = num_rows <= 1024
    expected = (
        overlaps_moe_dispatch and decode and (value == "1" or (value == "auto" and num_rows >= 128))
    )
    assert situ.situ_without_pdl(num_rows, overlaps_moe_dispatch) == expected


@pytest.mark.parametrize("value", [None, "auto", "1"], ids=["unset", "auto", "on"])
def test_unsupported_platform_keeps_the_pdl_launch(platform, switch, value):
    platform(False)
    switch(value)
    assert not situ.situ_without_pdl(256, True)


BAD_VALUES = ["2", "yes", "-1", "Auto"]


@pytest.mark.parametrize("value", BAD_VALUES)
def test_bad_switch_value_is_rejected(platform, switch, value):
    platform(True)
    switch(value)
    with pytest.raises(ValueError, match=situ.ENV_SITU_NO_PDL):
        situ.situ_without_pdl(256, True)


# A bad value is rejected whatever else the selection would have decided first: a rank whose batch is too large, whose
# layer is not beside a dispatch or whose platform is not the validated one must raise like every other rank, not walk
# on into the next collective while its peers raise.
@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize("supported", [False, True], ids=["other_platform", "validated_platform"])
@pytest.mark.parametrize("num_rows", [48, 1025], ids=lambda v: f"rows{v}")
@pytest.mark.parametrize("overlaps_moe_dispatch", [False, True], ids=["other", "beside_dispatch"])
def test_bad_switch_value_is_rejected_before_any_other_decision(
    platform, switch, overlaps_moe_dispatch, num_rows, supported, value
):
    platform(supported)
    switch(value)
    with pytest.raises(ValueError, match=situ.ENV_SITU_NO_PDL):
        situ.situ_without_pdl(num_rows, overlaps_moe_dispatch)


@pytest.mark.parametrize("enable_pdl", [None, "0"], ids=["pdl_default", "pdl_off"])
@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize("overlaps_moe_dispatch", [False, True], ids=["other", "beside_dispatch"])
def test_bad_switch_value_is_rejected_when_the_module_is_built(
    monkeypatch, switch, overlaps_moe_dispatch, value, enable_pdl
):
    """At the start of the server, on every rank, with or without PDL: never inside a step."""
    if enable_pdl is None:
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", enable_pdl)
    switch(value)
    with pytest.raises(ValueError, match=situ.ENV_SITU_NO_PDL):
        SituAndMul(
            beta=4.0,
            linear_beta=25.0,
            use_fused_activation=True,
            overlaps_moe_dispatch=overlaps_moe_dispatch,
        )


@pytest.mark.parametrize(
    "value", [None, "", "0", "auto", "1"], ids=["unset", "empty", "off", "auto", "on"]
)
def test_good_switch_values_build_the_module(switch, value):
    switch(value)
    SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True, overlaps_moe_dispatch=True)


class _LaunchRecorder:
    """Stands in for ``situ_and_mul_kernel``: records the ``launch_pdl`` of every launch, then launches."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launch_pdl = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(**kwargs):
            self.launch_pdl.append(kwargs["launch_pdl"])
            return launch(**kwargs)

        return run


@requires_cuda
@pytest.mark.parametrize("num_rows", [48, 128, 1025], ids=lambda v: f"rows{v}")
@pytest.mark.parametrize("value", [None, "0", "auto", "1"], ids=["unset", "off", "auto", "on"])
@pytest.mark.parametrize("overlaps_moe_dispatch", [False, True], ids=["other", "beside_dispatch"])
def test_launch_follows_selection(
    monkeypatch, platform, switch, overlaps_moe_dispatch, value, num_rows
):
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._utils import get_sm_version

    platform(True)
    switch(value)
    recorder = _LaunchRecorder(situ.situ_and_mul_kernel)
    monkeypatch.setattr(situ, "situ_and_mul_kernel", recorder)
    x = torch.randn(num_rows, 4096, dtype=torch.bfloat16, device="cuda")
    SituAndMul(
        beta=4.0,
        linear_beta=25.0,
        use_fused_activation=True,
        overlaps_moe_dispatch=overlaps_moe_dispatch,
    )(x)
    torch.cuda.synchronize()
    default_pdl = bool(get_env_enable_pdl() and get_sm_version() >= 90)
    assert recorder.launch_pdl == [
        default_pdl and not situ.situ_without_pdl(num_rows, overlaps_moe_dispatch)
    ]


@requires_cuda
@pytest.mark.parametrize("linear_beta", [25.0, None], ids=["linear", "no_linear"])
@pytest.mark.parametrize(
    "num_rows,two_d",
    [(1, 12288), (7, 736), (48, 12288), (192, 12288), (256, 12288), (384, 12288), (33, 2050)],
    ids=lambda v: str(v),
)
def test_no_pdl_launch_writes_the_bytes_of_the_pdl_launch(
    platform, switch, num_rows, two_d, linear_beta
):
    platform(True)
    torch.manual_seed(3)
    x = torch.randn(num_rows, two_d, dtype=torch.bfloat16, device="cuda") * 4.0
    switch("0")
    expected = torch.ops.trtllm.situ_and_mul(x, 4.0, linear_beta, True)
    switch("1")
    actual = torch.ops.trtllm.situ_and_mul(x, 4.0, linear_beta, True)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@requires_cuda
def test_no_pdl_launch_is_cuda_graph_capturable(platform, switch):
    platform(True)
    switch("1")
    torch.manual_seed(5)
    x = torch.randn(192, 12288, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0, True)  # compile outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0, True)
    x2 = torch.randn_like(x)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    switch("0")
    assert torch.equal(out, torch.ops.trtllm.situ_and_mul(x2, 4.0, 25.0, True))


@requires_cuda
def test_no_pdl_launch_is_logged_once_per_shape(monkeypatch, platform, switch):
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._utils import get_sm_version

    if not (get_env_enable_pdl() and get_sm_version() >= 90):
        pytest.skip(
            "every launch is without PDL here, none is a no-PDL launch of the shared expert"
        )
    platform(True)
    switch("auto")
    lines = []
    monkeypatch.setattr(situ.logger, "log", lambda severity, *msg: lines.append(" ".join(msg)))
    for rows in (192, 256, 64):
        situ.logger._appeared_keys.discard(f"situ_without_pdl_{rows}_2048")
    x = torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")
    shared_expert = SituAndMul(
        beta=4.0, linear_beta=25.0, use_fused_activation=True, overlaps_moe_dispatch=True
    )
    other = SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True)
    for rows in (192, 192, 256, 64):  # 64 rows: below the threshold of ``auto``, the PDL launch
        shared_expert(x[:rows])
    other(x[:192])  # a dense MLP keeps the PDL launch and stays silent
    torch.cuda.synchronize()
    logged = [line for line in lines if "without PDL" in line]
    assert len(logged) == 2, lines
    assert "rows=192" in logged[0] and "rows=256" in logged[1]
    assert situ.ENV_SITU_NO_PDL in logged[0]
