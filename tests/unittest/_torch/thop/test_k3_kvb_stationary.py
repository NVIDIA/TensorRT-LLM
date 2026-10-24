# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B-stationary MXFP8 GEMM tactics (K3 MLA kv_b decompression, FP8 output) match the dense kernel bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin MXFP8 GEMM requires SM107",
)

STATIONARY_ENV = "TRTLLM_CUTEDSL_MXFP8_STATIONARY"
K_TILES = 4  # K = 512 in 128-wide K tiles
# The production kv_b tactic (the tuner's pick in the CTX workers): 1-CTA 128 x 256 tile, UNSWAPPED
# operands (for M >= 256 the base menu offers no swapped tactic), CLC dynamic scheduling (the DWDP compute-side defaults
# pin it).
DENSE_TACTIC = ("base", (128, 256, 128), (128, 256, 64), (1, 1), False, False, "clc_dynamic", "m")
# (mma tiler M, mma instruction M, cluster shape, resident N blocks)
STATIONARY_CONFIGS = [
    (256, 256, (2, 1), 2),
    (256, 256, (2, 1), 1),
    (128, 128, (1, 1), 1),
]


def _stationary_tactic(config, swap_ab, num_a_stage=0):
    tiler_m, inst_m, cluster, n_res = config
    return (
        "stationary",
        (tiler_m, 256, 128),
        (inst_m, 256, 64),
        cluster,
        swap_ab,
        n_res,
        num_a_stage,
    )


def _make_inputs(m, n, k=512, seed=107):
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    b, b_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    return [a, b, a_sf, b_sf, alpha], alpha


@pytest.mark.parametrize("use_tvm_ffi", [False, True])
@pytest.mark.parametrize("swap_ab", [True, False], ids=["swap", "noswap"])
@pytest.mark.parametrize(
    "config", STATIONARY_CONFIGS, ids=["2cta_nres2", "2cta_nres1", "1cta_nres1"]
)
# N blocks of 256: 8 (even) and 7 (odd: the last group of n_res = 2 runs a missing block on zero-filled B).
@pytest.mark.parametrize("n", [2048, 1792])
# M = 5 blocks of 256 (odd), and 17 blocks with a partial last block.
@pytest.mark.parametrize("m", [1280, 4097])
def test_stationary_matches_dense(monkeypatch, m, n, config, swap_ab, use_tvm_ffi):
    monkeypatch.setenv(STATIONARY_ENV, "1")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    inputs, alpha = _make_inputs(m, n)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn, use_tvm_ffi=use_tvm_ffi)
    tactic = _stationary_tactic(config, swap_ab)
    # A non-unit alpha rounds through BF16; large values saturate to the E4M3 range; inf and NaN take the same
    # path as in the dense epilogue.
    for scale in (1.0, 0.5, 1.3, 32.0, float("inf"), float("-inf"), float("nan")):
        alpha.fill_(scale)
        expected = runner(inputs, tactic=DENSE_TACTIC)
        actual = runner(inputs, tactic=tactic)
        assert actual.shape == expected.shape == (m, n)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"alpha {scale}: {int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())} bytes differ"
        )


@pytest.mark.parametrize(
    "config", STATIONARY_CONFIGS, ids=["2cta_nres2", "2cta_nres1", "1cta_nres1"]
)
def test_stationary_few_clusters_matches_dense(monkeypatch, config):
    """With all but a few SMs reserved the persistent grid has 4 clusters (8 CTAs), so every cluster runs many steps.

    One step per cluster (the other tests' shapes) never reaches the steady state: the A ring wrapping across
    steps, the accumulator stages being freed before the MMA warp reuses them, B being reloaded, and the store
    warp's later-step path.
    """
    monkeypatch.setenv(STATIONARY_ENV, "1")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", str(sms - 8))
    # Swapped operands: 4097 rows give 17 blocks of 256 (a partial last one, an odd group tail for n_res 2), 2048
    # columns 8 blocks (2-CTA) or 16 (1-CTA): 72 / 72 / 272 steps over 4 / 4 / 8 CTAs.
    inputs, alpha = _make_inputs(4097, 2048)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    tactic = _stationary_tactic(config, True)
    for scale in (1.0, 0.3):
        alpha.fill_(scale)
        expected = runner(inputs, tactic=DENSE_TACTIC)
        actual = runner(inputs, tactic=tactic)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"alpha {scale}: {int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())} bytes differ"
        )


@pytest.mark.parametrize(
    "config", [STATIONARY_CONFIGS[0], STATIONARY_CONFIGS[2]], ids=["2cta_nres2", "1cta_nres1"]
)
def test_stationary_production_shape_matches_dense(monkeypatch, config):
    """The CTX kv_b GEMM at the production N (24576) and a 16320-token chunk (a partial last block): about 30 steps per
    cluster on the whole device, swapped operands as the autotuner picks them."""
    monkeypatch.setenv(STATIONARY_ENV, "1")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    inputs, alpha = _make_inputs(16320, 24576)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    tactic = _stationary_tactic(config, True)
    for scale in (1.0, 0.3):
        alpha.fill_(scale)
        expected = runner(inputs, tactic=DENSE_TACTIC)
        actual = runner(inputs, tactic=tactic)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"alpha {scale}: {int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())} bytes differ"
        )


@pytest.mark.parametrize("num_a_stage", [2, 3, 4, 5])
def test_stationary_a_ring_depth(monkeypatch, num_a_stage):
    """A shallow A ring (the TMA warp waits on every stage) only changes the speed."""
    monkeypatch.setenv(STATIONARY_ENV, "1")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    inputs, _ = _make_inputs(2304, 1024)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    expected = runner(inputs, tactic=DENSE_TACTIC)
    for config in STATIONARY_CONFIGS:
        if config[3] > 1 and num_a_stage < K_TILES:
            continue  # rejected, see test_stationary_rejects_a_ring_shorter_than_a_block
        actual = runner(inputs, tactic=_stationary_tactic(config, True, num_a_stage))
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_stationary_rejects_a_ring_shorter_than_a_block(monkeypatch):
    """n_res = 2 keeps a whole A block (K_TILES stages) in the ring until its second output tile: a shorter ring must
    fail at compile time, not deadlock."""
    monkeypatch.setenv(STATIONARY_ENV, "1")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    inputs, _ = _make_inputs(512, 512)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    with pytest.raises(Exception, match="A ring of 2 stages"):
        runner(inputs, tactic=_stationary_tactic(STATIONARY_CONFIGS[0], True, 2))


def test_stationary_tactics_are_gated(monkeypatch):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    monkeypatch.delenv(STATIONARY_ENV, raising=False)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    assert runner._stationary_tactics(65536, 24576, 512) == []
    key_off = runner.unique_id()
    assert "stationary" not in key_off

    monkeypatch.setenv(STATIONARY_ENV, "1")
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn)
    # Only the switched-on runs get a new autotuner key (unswitched runs keep the old one); BF16 output never does.
    assert runner.unique_id() == key_off + ("stationary",)
    assert "stationary" not in CuteDSLMXFP8RubinLinear(torch.bfloat16).unique_id()
    tactics = runner._stationary_tactics(65536, 24576, 512)
    # 1-CTA (n_res 1) and 2-CTA (n_res 2) tiles, swapped operands only (whatever the base menu offers): the no-swap
    # tactics lose to them while the DWDP gather writes into HBM.
    assert len(tactics) == 2
    assert all(t[0] == "stationary" and len(t) == 7 and t[6] == 0 for t in tactics)
    assert {t[4] for t in tactics} == {True}
    # Short K only (at most 4 K tiles), large M and N only, FP8 output only.
    assert runner._stationary_tactics(65536, 24576, 1024) == []
    assert runner._stationary_tactics(2048, 24576, 512) == []
    assert CuteDSLMXFP8RubinLinear(torch.bfloat16)._stationary_tactics(65536, 24576, 512) == []


@pytest.fixture
def clean_autotuner():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner

    AutoTuner.get().clear_cache()
    yield
    AutoTuner.get().clear_cache()


@pytest.fixture
def ctx_worker_env(monkeypatch):
    """The CTX worker's environment (fabric_gather.apply_compute_side_defaults) and the flag under test."""
    monkeypatch.setenv(STATIONARY_ENV, "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "30")
    monkeypatch.setenv("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", "clc_dynamic")


@pytest.mark.parametrize("m", [4352, 16320, 65536])
def test_real_menu_offers_the_stationary_tactics(ctx_worker_env, m):
    """The tuner's own tactic list for the kv_b GEMM (get_valid_tactics, as the autotuner calls it) holds the production
    dense tactic and both stationary tactics: for M >= 256 the base menu has only unswapped tactics, the stationary ones
    must be offered regardless."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import OptimizationProfile
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    inputs, _ = _make_inputs(m, 24576)
    menu = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn, use_tvm_ffi=True).get_valid_tactics(
        inputs, OptimizationProfile()
    )
    assert DENSE_TACTIC in menu, "the production dense tactic is not offered"
    stationary = [t for t in menu if t[0] == "stationary"]
    assert len(stationary) == 2 and all(t[4] is True for t in stationary), stationary
    assert not any(t[0] == "base" and t[4] for t in menu), (
        "the base menu offers swapped tactics at M >= 256"
    )


@pytest.mark.parametrize("m", [16320, 65536])
def test_stationary_is_chosen_and_run_by_the_autotuner(
    ctx_worker_env, monkeypatch, clean_autotuner, m
):
    """The op path of the model (tune once, then run the cached tactic): the kv_b GEMM picks a stationary tactic over
    the production dense tactic and returns the dense kernel's bytes.

    The full dense menu is several hundred kernel compiles, so the tuning menu is cut to the production dense tactic
    and the stationary tactics (the unmodified menu is checked by test_real_menu_offers_the_stationary_tactics)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    full_menu = CuteDSLMXFP8RubinLinear.get_valid_tactics
    offered = []

    def pruned_menu(self, inputs, profile, **kwargs):
        tactics = full_menu(self, inputs, profile, **kwargs)
        offered.append(tactics)
        return [t for t in tactics if t[0] == "stationary" or t == DENSE_TACTIC]

    monkeypatch.setattr(CuteDSLMXFP8RubinLinear, "get_valid_tactics", pruned_menu)
    inputs, _ = _make_inputs(m, 24576)
    args = (inputs[0], inputs[1], inputs[2], inputs[3], torch.float8_e4m3fn)
    with autotune(skip_dynamic_tuning_buckets=True):
        tuned = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(*args)
    cached = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(*args)
    runner = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn, use_tvm_ffi=True)
    _, best = AutoTuner.get().choose_one(
        "trtllm::cute_dsl_mxfp8_gemm_rubin", [runner], runner.__class__.tuning_config, inputs
    )
    assert offered, "the tuner never asked for the tactics"
    assert all(DENSE_TACTIC in menu for menu in offered), (
        "the production dense tactic is not offered"
    )
    assert all(sum(t[0] == "stationary" for t in menu) == 2 for menu in offered)
    assert best[0] == "stationary", f"the autotuner chose {best}"
    expected = runner(inputs, tactic=DENSE_TACTIC)
    for out in (tuned, cached):
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def test_flags_are_strict_switches(monkeypatch):
    """Both CX3 flags take 0 or 1 only (like the other K3 switches): "off" or "true" must not silently pick a side."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels import fc1_situ_fast, mxfp8_stationary

    for env, read in (
        ("TRTLLM_CUTEDSL_MXFP8_STATIONARY", mxfp8_stationary),
        ("TRTLLM_CUTEDSL_FC1_SITU_FAST", fc1_situ_fast),
    ):
        monkeypatch.delenv(env, raising=False)
        assert read() is False
        monkeypatch.setenv(env, "1")
        assert read() is True
        monkeypatch.setenv(env, "0")
        assert read() is False
        for bad in ("off", "true", "2", ""):
            monkeypatch.setenv(env, bad)
            with pytest.raises(ValueError, match=env):
                read()
